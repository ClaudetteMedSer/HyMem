from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable

from hymem import portability
from hymem import redaction
from hymem import session as session_log
from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import canonicalize as canon
from hymem.dreaming.runner import DreamReport, run_dreaming
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.query.augment import AugmentedContext, augment, build_token_overlap_index
from hymem.query.conflicts import Conflict, find_conflicts
from hymem.query.entities import TimelineEntry, timeline as query_timeline

log = logging.getLogger("hymem.api")


class HyMem:
    """Public API for the Hermes host.

    Hermes is responsible for:
      - constructing one HyMem per agent process (or per project root),
      - calling `log_message` for every conversational turn,
      - calling `augment` before sending a user turn to its model,
      - calling `dream` during idle windows (or via its own scheduler),
      - providing an LLMClient — required only for `dream`, not for `augment`.
    """

    def __init__(
        self,
        config: HyMemConfig,
        *,
        llm: LLMClient | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config
        self._llm = llm
        self._embed = embedding_client
        self._conn: sqlite3.Connection | None = None
        self._read_conn: sqlite3.Connection | None = None
        self._initialized = False
        # Token-overlap index for entity expansion in augment(). Built lazily
        # on first augment, invalidated after dreaming since dreaming is the
        # only thing that mutates the canonical set.
        self._token_overlap_index: dict[str, list[str]] | None = None

    # ---- lifecycle ---------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = core_db.connect(self.config.db_path)
        if not self._initialized:
            core_db.initialize(self._conn)
            core_db.backfill_entity_mentions(self._conn)
            self._initialized = True
        return self._conn

    @property
    def read_conn(self) -> sqlite3.Connection:
        if self._read_conn is None:
            self.conn  # ensure the write connection is initialized first
            self._read_conn = core_db.connect(self.config.db_path)
            # Load vec extension before query_only so semantic edge KNN uses
            # the vec0 fast path instead of falling back to python cosine.
            core_db._load_vec_extension(self._read_conn)
            self._read_conn.execute("PRAGMA query_only = ON")
        return self._read_conn

    def close(self) -> None:
        if self._read_conn is not None:
            self._read_conn.close()
            self._read_conn = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False

    def set_llm(self, llm: LLMClient) -> None:
        self._llm = llm

    def set_embedding_client(self, embedding_client: EmbeddingClient) -> None:
        self._embed = embedding_client

    def fork(self) -> "HyMem":
        """Return a new HyMem on the same database with its own SQLite
        connection, reusing this instance's LLM and embedding clients.

        Used to run background dreaming on a separate connection so it does
        not collide with live ingestion on the primary connection.
        """
        return HyMem(self.config, llm=self._llm, embedding_client=self._embed)

    # ---- session log -------------------------------------------------

    def open_session(self, session_id: str) -> None:
        with core_db.transaction(self.conn):
            session_log.open_session(self.conn, session_id)

    def close_session(self, session_id: str) -> None:
        with core_db.transaction(self.conn):
            session_log.close_session(self.conn, session_id)

    def _prepare_content(self, content: str) -> str:
        """Apply the ingest-boundary guards (size cap, then secret redaction)
        before any message text reaches SQLite. Truncation happens first so the
        redactor never has to scan an unbounded string."""
        cap = self.config.max_message_chars
        if cap and len(content) > cap:
            log.warning(
                "log_message: content of %d chars exceeds max_message_chars=%d; truncating",
                len(content), cap,
            )
            content = content[:cap] + "\n[TRUNCATED]"
        if self.config.redact_secrets:
            content = redaction.redact(content)
        return content

    def log_message(self, session_id: str, role: str, content: str) -> int:
        content = self._prepare_content(content)
        with core_db.transaction(self.conn):
            session_log.open_session(self.conn, session_id)
            return session_log.append_message(self.conn, session_id, role, content)

    def log_messages(
        self, session_id: str, turns: Iterable[tuple[str, str]]
    ) -> list[int]:
        """Append a batch of (role, content) turns in a single transaction.

        One BEGIN IMMEDIATE for the whole batch instead of one per message.
        """
        prepared = [(role, self._prepare_content(content)) for role, content in turns]
        with core_db.transaction(self.conn):
            session_log.open_session(self.conn, session_id)
            return [
                session_log.append_message(self.conn, session_id, role, content)
                for role, content in prepared
            ]

    # ---- query-time --------------------------------------------------

    def augment(
        self, user_message: str, *, session_id: str | None = None
    ) -> AugmentedContext:
        cap = self.config.max_query_chars
        if cap and len(user_message) > cap:
            log.warning(
                "augment: query of %d chars exceeds max_query_chars=%d; truncating",
                len(user_message), cap,
            )
            user_message = user_message[:cap]
        if self._token_overlap_index is None:
            self._token_overlap_index = build_token_overlap_index(
                self.read_conn, write_conn=self.conn
            )
        return augment(
            self.read_conn, self.config, user_message,
            embedding_client=self._embed,
            llm=self._llm,
            token_overlap_index=self._token_overlap_index,
            session_id=session_id,
        )

    def timeline(self, entity: str) -> list["TimelineEntry"]:
        """First-seen active edge per predicate for `entity`, oldest first.

        Answers "when did we start using X?" from `knowledge_graph.first_seen`
        without re-asking the user. `entity` may be a surface form; it's
        resolved through the alias table. Read-only.
        """
        return query_timeline(self.read_conn, entity)

    def conflicts(self) -> list[Conflict]:
        """Return detected contradictions in the knowledge graph. Read-only.

        Surfaces edges that disagree — competing objects under an exclusive
        predicate, or a subject/object pair joined by opposing predicates.
        """
        return find_conflicts(self.read_conn)

    # ---- dreaming ----------------------------------------------------

    def dream(self, *, session_ids: Iterable[str] | None = None) -> DreamReport:
        if self._llm is None:
            raise RuntimeError(
                "HyMem.dream requires an LLMClient. Pass one to the constructor "
                "or call set_llm() before dreaming."
            )
        ids = list(session_ids) if session_ids is not None else None
        report = run_dreaming(
            self.conn,
            self.config,
            self._llm,
            session_ids=ids,
            embedding_client=self._embed,
        )
        # Dreaming may have added, retracted, or merged canonicals — invalidate
        # the token-overlap index so the next augment() rebuilds it.
        self._token_overlap_index = None
        return report

    def invalidate_query_caches(self) -> None:
        """Clear query-side caches (token-overlap index).

        Call after an *external* write to the DB — e.g. a forked HyMem instance
        completed a background dream cycle — so this instance's next augment()
        rebuilds the index from fresh state. In-process `dream()`,
        `merge_canonical()`, and `retract_edge()` already self-invalidate.
        """
        self._token_overlap_index = None

    def recent_dream_runs(self, limit: int = 20) -> list[dict]:
        """Return the last N dream_runs rows as dicts, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM dream_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def dream_status(self) -> dict:
        """Operator-visibility snapshot of the dreaming/extraction backlog.

        Pure SQL via `read_conn` — no LLM or embedding calls, no writes — so it
        works even when no LLM/embedding client is configured. Useful to explain
        the re-extraction surge that follows a `prompt_version` bump: when the
        version changes, every chunk is "pending" again and the next dream(s)
        reprocess the whole backlog, which can take minutes.

        Returns a dict with:
          - `pending_chunks`: chunks with NO `processed_chunks` row for the
            CURRENT `config.prompt_version` (i.e. still owed an extraction pass).
          - `total_chunks`: total chunk count.
          - `prompt_version`: the current `config.prompt_version`.
          - `in_progress`: True iff a `run_lock` row named 'dreaming' exists.
            This is intentionally coarse — a *stale* lock (e.g. from a crashed
            dream) reads as in_progress until it expires. Lock heartbeat/TTL is
            handled in the dreaming runner; this method does not reimplement it.
          - `last_run`: the most recent `dream_runs` row as a dict, or None if
            no dream has ever run.
        """
        conn = self.read_conn
        pv = self.config.prompt_version

        pending_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks "
            "WHERE id NOT IN ("
            "    SELECT chunk_id FROM processed_chunks WHERE prompt_version = ?"
            ")",
            (pv,),
        ).fetchone()[0]
        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        in_progress = (
            conn.execute(
                "SELECT 1 FROM run_lock WHERE name = 'dreaming' LIMIT 1"
            ).fetchone()
            is not None
        )
        last_row = conn.execute(
            "SELECT * FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

        return {
            "pending_chunks": pending_chunks,
            "total_chunks": total_chunks,
            "prompt_version": pv,
            "in_progress": in_progress,
            "last_run": dict(last_row) if last_row is not None else None,
        }

    # ---- portability -------------------------------------------------

    def export(self, path: str | Path) -> dict[str, int]:
        """Write the canonical state (sessions, chunks, edges, episodes,
        procedures, profile entries) to `path` as JSON Lines. Returns per-kind
        row counts. Useful for backups, project migration, and external
        inspection. Read-only.
        """
        return portability.export_jsonl(self.conn, path)

    def import_(self, path: str | Path) -> dict[str, int]:
        """Load a JSON Lines export written by `export()`. Additive and
        idempotent (INSERT-OR-IGNORE, sessions first); returns per-kind counts
        of rows inserted. Best run against a fresh database. Invalidates the
        query-side caches afterwards.
        """
        result = portability.import_jsonl(self.conn, path)
        self._token_overlap_index = None
        return result

    # ---- maintenance -------------------------------------------------

    def register_alias(self, surface: str, canonical: str) -> None:
        with core_db.transaction(self.conn):
            canon.register_alias(self.conn, surface, canonical)

    def merge_canonical(self, keep: str, drop: str) -> None:
        with core_db.transaction(self.conn):
            canon.merge(self.conn, keep, drop)
            self.conn.execute(
                "DELETE FROM token_overlap_index WHERE canonical = ?", (drop,)
            )
        self._token_overlap_index = None

    def retract_edge(self, subject: str, predicate: str, object: str) -> bool:
        """Mark an edge as retracted. Idempotent. Returns True if an edge was found and updated, False otherwise.

        Subjects/objects are normalized through the alias table — pass the surface
        form (e.g., 'MedFlow') and HyMem resolves to the canonical id.

        Only acts on edges with status='active'; calling again on an already
        retracted edge returns False.
        """
        with core_db.transaction(self.conn):
            subj = canon.resolve(self.conn, subject)
            obj = canon.resolve(self.conn, object)
            row = self.conn.execute(
                "SELECT id FROM knowledge_graph "
                "WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ? "
                "AND status = 'active'",
                (subj, predicate, obj),
            ).fetchone()
            if row is None:
                return False
            self.conn.execute(
                "UPDATE knowledge_graph "
                "SET status = 'retracted', "
                "    neg_evidence = neg_evidence + 1, "
                "    last_seen = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (row["id"],),
            )
            # Store feedback for future extraction improvement
            evidence_rows = self.conn.execute(
                """SELECT chunk_id FROM kg_evidence 
                   WHERE edge_id = ? AND polarity = 1
                   ORDER BY extracted_at DESC LIMIT 5""",
                (row["id"],),
            ).fetchall()
            for er in evidence_rows:
                chunk_row = self.conn.execute(
                    "SELECT text FROM chunks WHERE id = ?", (er["chunk_id"],)
                ).fetchone()
                if chunk_row:
                    snippet = chunk_row["text"][:600]
                    self.conn.execute(
                        """INSERT OR IGNORE INTO extraction_feedback
                           (chunk_id, chunk_text_snippet, extracted_subject,
                            extracted_predicate, extracted_object, feedback_type)
                           VALUES (?, ?, ?, ?, ?, 'retracted')""",
                        (er["chunk_id"], snippet, subject, predicate, object),
                    )
            for c in (subj, obj):
                still_active = self.conn.execute(
                    "SELECT 1 FROM knowledge_graph "
                    "WHERE (subject_canonical = ? OR object_canonical = ?) "
                    "AND status = 'active' LIMIT 1",
                    (c, c),
                ).fetchone()
                if still_active is None:
                    self.conn.execute(
                        "DELETE FROM token_overlap_index WHERE canonical = ?", (c,)
                    )
        self._token_overlap_index = None
        return True

    def mark_procedure_stale(self, procedure_id: str) -> bool:
        """Flag a procedure as wrong / outdated. Idempotent. Returns True if an
        active procedure was found and updated, False otherwise.

        Symmetric to `retract_edge`, but for procedural memory: when Hermes
        surfaces a procedure via `augment()` (a `ProcedureHit`) and the user
        marks it stale, the row is flipped to status='stale' — which removes it
        from future `_procedure_search` results — and its `confidence` is
        knocked down by `cfg.procedure_stale_confidence_factor` so the negative
        signal survives even if the procedure is later re-extracted.

        Only acts on procedures with status='active'; calling again on an
        already-stale procedure returns False.
        """
        with core_db.transaction(self.conn):
            row = self.conn.execute(
                "SELECT id FROM procedures WHERE id = ? AND status = 'active'",
                (procedure_id,),
            ).fetchone()
            if row is None:
                return False
            self.conn.execute(
                "UPDATE procedures "
                "SET status = 'stale', confidence = confidence * ? "
                "WHERE id = ?",
                (self.config.procedure_stale_confidence_factor, procedure_id),
            )
        return True
