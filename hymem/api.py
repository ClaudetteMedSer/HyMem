from __future__ import annotations

import logging
import sqlite3
from typing import Iterable

from hymem import session as session_log
from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import canonicalize as canon
from hymem.dreaming.runner import DreamReport, run_dreaming
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.query.augment import AugmentedContext, augment
from hymem.query.conflicts import Conflict, find_conflicts

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

    def log_message(self, session_id: str, role: str, content: str) -> int:
        with core_db.transaction(self.conn):
            session_log.open_session(self.conn, session_id)
            return session_log.append_message(self.conn, session_id, role, content)

    def log_messages(
        self, session_id: str, turns: Iterable[tuple[str, str]]
    ) -> list[int]:
        """Append a batch of (role, content) turns in a single transaction.

        One BEGIN IMMEDIATE for the whole batch instead of one per message.
        """
        with core_db.transaction(self.conn):
            session_log.open_session(self.conn, session_id)
            return [
                session_log.append_message(self.conn, session_id, role, content)
                for role, content in turns
            ]

    # ---- query-time --------------------------------------------------

    def augment(self, user_message: str) -> AugmentedContext:
        return augment(
            self.read_conn, self.config, user_message,
            embedding_client=self._embed,
            llm=self._llm,
        )

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
        return run_dreaming(
            self.conn,
            self.config,
            self._llm,
            session_ids=ids,
            embedding_client=self._embed,
        )

    def recent_dream_runs(self, limit: int = 20) -> list[dict]:
        """Return the last N dream_runs rows as dicts, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM dream_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- maintenance -------------------------------------------------

    def register_alias(self, surface: str, canonical: str) -> None:
        with core_db.transaction(self.conn):
            canon.register_alias(self.conn, surface, canonical)

    def merge_canonical(self, keep: str, drop: str) -> None:
        with core_db.transaction(self.conn):
            canon.merge(self.conn, keep, drop)

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
            return True
