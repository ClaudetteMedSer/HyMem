"""Guarded single-chunk Phase-1 retry for Hermes1's unpublished extraction.

Preview builds a source/producer/retry fingerprint without provider calls or
writes. Apply needs exact reviewed digests, a new private backup, and the
dreaming lease. Eight logical completions bound the maintained LLM transport to
at most 24 HTTP attempts. Failed extraction records only a fourth retry; full
validated success is staged on a private clone before atomic live publication.
No source text, model reply, vector, or credential is printed.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import sys
import tempfile
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite_vec

from hymem import bootstrap
from hymem.config import HyMemConfig
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.core import db
from hymem.core.graph import live_edge_predicate
from hymem.core.vectors import decode_vector
from hymem.dreaming import phase1, runner
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.chunks import Chunk
from hymem.dreaming.embeddings import (
    PendingEdgeEmbeddings, fetch_edge_embeddings, persist_edge_embeddings,
)
from hymem.extraction.producer import phase1_generation_binding
from hymem.reembed import _configured_embedder


LIVE_PATHS = (
    Path("/opt/stacks/hermes/instance1/home/.hermes/hymem.sqlite"),
    Path("/home/node/.hermes/hymem.sqlite"),
)
CALL_LIMIT = 8
HTTP_LIMIT = 24
EXPECTED_ATTEMPTS = 3
EXPECTED_OBSERVATIONS = 37


@dataclass(frozen=True)
class Plan:
    chunk: Chunk
    evidence_id: int
    edge_id: int
    source_digest: str
    generation_digest: str
    retry_digest: str
    claim_digest: str
    state_digest: str
    live_ids: frozenset[int]
    aggregation_state: tuple[int, int, int]

    def summary(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_digest,
            "generation_sha256": self.generation_digest,
            "retry_sha256": self.retry_digest,
            "state_sha256": self.state_digest,
            "legacy_observations": EXPECTED_OBSERVATIONS,
            "quarantined_attempts": EXPECTED_ATTEMPTS,
            "live_direct_edges": len(self.live_ids),
            "max_logical_completions": CALL_LIMIT,
            "max_provider_attempts": HTTP_LIMIT,
        }


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, default=str,
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[list]:
    return [list(row) for row in conn.execute(sql, params).fetchall()]


def _claim_digest(conn: sqlite3.Connection, chunk_id: str) -> str:
    observations = _rows(
        conn, "SELECT * FROM kg_claim_observations WHERE chunk_id=? "
        "ORDER BY edge_id,source_session_id,source_message_id,evidence_kind",
        (chunk_id,),
    )
    evidence_rows = _rows(
        conn, "SELECT * FROM kg_evidence WHERE chunk_id=? ORDER BY id",
        (chunk_id,),
    )
    edge_ids = tuple(sorted({
        int(row["edge_id"])
        for row in conn.execute(
            "SELECT edge_id FROM kg_evidence WHERE chunk_id=?", (chunk_id,),
        )
    }))
    if not evidence_rows or not edge_ids:
        raise ValueError("reviewed claim rows are missing")
    return _digest([
        observations,
        evidence_rows,
        _rows(conn, "SELECT * FROM knowledge_graph WHERE id IN ("
              + ",".join("?" for _ in edge_ids) + ") ORDER BY id",
              edge_ids),
        _rows(conn, "SELECT * FROM processed_chunks WHERE chunk_id=?",
              (chunk_id,)),
        _rows(conn, "SELECT * FROM kg_claim_extraction_outcomes "
              "WHERE chunk_id=?", (chunk_id,)),
        _rows(conn, "SELECT * FROM current_phase1_publications "
              "WHERE chunk_id=?", (chunk_id,)),
        _rows(conn, "SELECT * FROM kg_edge_lifecycle "
              "WHERE edge_id IN ("
              + ",".join("?" for _ in edge_ids) + ") ORDER BY id",
              edge_ids),
    ])


def _is_live(path: Path) -> bool:
    return any(
        path.resolve() == known.resolve()
        or (path.exists() and known.exists() and path.samefile(known))
        for known in LIVE_PATHS
    )


def _lease_guard(conn: sqlite3.Connection, holder: str | None) -> None:
    rows = conn.execute(
        "SELECT holder FROM run_lock WHERE name='dreaming'"
    ).fetchall()
    if holder is None:
        if rows:
            raise ValueError("dreaming lease is present")
    elif len(rows) != 1 or rows[0][0] != holder:
        raise ValueError("operator lost dreaming lease")


def _aggregation_state(conn: sqlite3.Connection) -> tuple[int, int, int]:
    return (
        int(conn.execute(
            "SELECT COUNT(*) FROM aggregation_enabled_root_material"
        ).fetchone()[0]),
        int(conn.execute(
            "SELECT revision FROM aggregation_material_clock WHERE id=1"
        ).fetchone()[0]),
        int(conn.execute(
            "SELECT COUNT(*) FROM aggregation_publication_state"
        ).fetchone()[0]),
    )


def _plan(conn: sqlite3.Connection, generation: dict,
          *, holder: str | None = None) -> Plan:
    schema = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    if schema is None or int(schema[0]) != 64:
        raise ValueError("schema mismatch")
    _lease_guard(conn, holder)
    evidence_rows = conn.execute(
        "SELECT id,edge_id,chunk_id,polarity FROM kg_evidence "
        "WHERE evidence_kind='extraction' AND provenance_status='canonical' "
        "AND is_current=1 AND published_at IS NULL"
    ).fetchall()
    if len(evidence_rows) != 1 or int(evidence_rows[0]["polarity"]) != 1:
        raise ValueError("reviewed orphan cohort changed")
    orphan = evidence_rows[0]
    chunk_id = str(orphan["chunk_id"])
    chunk_row = conn.execute(
        "SELECT id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,source_manifest_version,source_manifest_count "
        "FROM chunks WHERE id=? AND chunk_kind='extraction'",
        (chunk_id,),
    ).fetchone()
    if chunk_row is None or chunk_row["source_manifest_version"] != (
        "claim-source-manifest-v1"
    ) or int(chunk_row["source_manifest_count"] or 0) != 1:
        raise ValueError("orphan lacks reviewed exact source manifest")
    members = conn.execute(
        "SELECT ordinal,source_message_id,source_session_id,"
        "source_coverage_chunk_id,source_coverage_version "
        "FROM chunk_message_sources WHERE chunk_id=? ORDER BY ordinal",
        (chunk_id,),
    ).fetchall()
    if len(members) != 1 or int(members[0]["ordinal"]) != 0:
        raise ValueError("orphan source membership changed")
    chunk = Chunk(
        str(chunk_row["id"]), str(chunk_row["session_id"]),
        int(chunk_row["start_message_id"]),
        int(chunk_row["end_message_id"]),
        str(chunk_row["salience_reason"]), str(chunk_row["text"]),
        (int(members[0]["source_message_id"]),),
    )
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    if len(sources) != 1 or sources[0].message_id != (
        chunk.source_message_ids[0]
    ):
        raise ValueError("orphan exact source proof is invalid")
    if len(phase1._claim_source_record(sources[0])) != 1206:
        raise ValueError("orphan source shape changed")

    prompt = str(generation["extraction_cache_key"])
    generation_key = str(generation["generation_key"])
    attempts = conn.execute(
        "SELECT * FROM chunk_extraction_attempts WHERE chunk_id=? "
        "ORDER BY prompt_version", (chunk_id,),
    ).fetchall()
    selected = [row for row in attempts if row["prompt_version"] == prompt]
    if (
        len(attempts) != 2 or len(selected) != 1
        or selected[0]["phase1_generation_key"] != generation_key
        or int(selected[0]["attempts"]) != EXPECTED_ATTEMPTS
        or selected[0]["last_failure_reason"] != "branch_incomplete"
    ):
        raise ValueError("reviewed retry quarantine changed")
    observations = conn.execute(
        "SELECT * FROM kg_claim_observations WHERE chunk_id=? "
        "ORDER BY edge_id,source_session_id,source_message_id,evidence_kind",
        (chunk_id,),
    ).fetchall()
    if (
        len(observations) != EXPECTED_OBSERVATIONS
        or sum(row["evidence_id"] == orphan["id"]
               for row in observations) != 1
        or any(row["phase1_generation_key"] is not None
               for row in observations)
        or conn.execute(
            "SELECT 1 FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (chunk_id,),
        ).fetchone() is not None
        or conn.execute(
            "SELECT 1 FROM current_phase1_publications WHERE chunk_id=?",
            (chunk_id,),
        ).fetchone() is not None
    ):
        raise ValueError("reviewed unpublished claim state changed")
    source_material = [
        list(chunk_row), [list(row) for row in members],
        [list(source.__dict__.values()) for source in sources],
    ]
    retry_material = [list(row) for row in attempts]
    claim_digest = _claim_digest(conn, chunk_id)
    state_material = [source_material, retry_material, claim_digest]
    live_ids = frozenset(int(row[0]) for row in conn.execute(
        f"SELECT id FROM knowledge_graph WHERE {live_edge_predicate()}"
    ))
    aggregation = _aggregation_state(conn)
    if aggregation != (0, 19, 1):
        raise ValueError("reviewed aggregation policy changed")
    return Plan(
        chunk, int(orphan["id"]), int(orphan["edge_id"]),
        _digest(source_material), _digest(generation),
        _digest(retry_material), claim_digest, _digest(state_material),
        live_ids, aggregation,
    )


def _load_server_process_env(proc_root: Path = Path("/proc")) -> None:
    allowed = {
        "HYMEM_ROOT", "HYMEM_DREAM_MAX_FACTS_PER_SESSION",
        "HYMEM_LLM_API_KEY", "HYMEM_LLM_BASE_URL",
        "HYMEM_LLM_MODEL", "HYMEM_LLM_THINKING", "HYMEM_LLM_EXTRA_BODY",
        "HYMEM_LLM_DEPLOYMENT_REVISION", "HYMEM_LLM_DEPLOYMENT_TENANT",
        "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
        "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM",
        "HYMEM_EMBEDDING_PIN_DIMENSION",
        "HYMEM_EMBEDDING_DEPLOYMENT_REVISION",
        "HYMEM_EMBEDDING_DEPLOYMENT_TENANT",
        "HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP",
        "HYMEM_EMBEDDING_TIMEOUT_SECONDS",
    }
    candidates: dict[str, list[dict[str, str]]] = {
        "hymem-honcho": [], "hymem-server": [],
    }
    honcho_switches: list[dict[str, str]] = []
    for process in sorted(proc_root.iterdir()):
        if not process.name.isdecimal():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
            tokens = {argument.rsplit(b"/", 1)[-1] for argument in argv}
            role = (
                "hymem-honcho"
                if tokens & {b"hymem-honcho", b"hymem.honcho.app"}
                else "hymem-server"
                if tokens & {b"hymem-server", b"hymem.server"}
                else None
            )
            if role is None:
                continue
            entries = (process / "environ").read_bytes().split(b"\0")
            variables = dict(item.split(b"=", 1) for item in entries
                             if b"=" in item)
            if (
                variables.get(b"HYMEM_EMBEDDING_BASE_URL")
                != b"http://embedding-server:8766/v1"
                or not variables.get(b"HYMEM_LLM_API_KEY")
                or not variables.get(b"HYMEM_EMBEDDING_API_KEY")
            ):
                continue
            selected = {
                key.decode("ascii"): value.decode("utf-8")
                for key, value in variables.items()
                if key.decode("ascii") in allowed
            }
        except (OSError, UnicodeError, ValueError):
            continue
        if selected not in candidates[role]:
            candidates[role].append(selected)
        if role == "hymem-honcho":
            switches = {
                key: variables[key.encode("ascii")].decode("utf-8")
                for key in (
                    "HYMEM_AGGREGATION_NODES_ENABLED",
                    "HYMEM_AGGREGATION_DIGEST_ENABLED",
                )
                if key.encode("ascii") in variables
            }
            if switches not in honcho_switches:
                honcho_switches.append(switches)
    if (
        any(len(values) != 1 for values in candidates.values())
        or candidates["hymem-honcho"][0]
        != candidates["hymem-server"][0]
        or len(honcho_switches) != 1
    ):
        raise ValueError("matching Hermes1 producer environments unavailable")
    for key in tuple(os.environ):
        if key.startswith(("HYMEM_LLM_", "HYMEM_EMBEDDING_")) or key in {
            "HYMEM_ROOT", "HYMEM_DREAM_MAX_FACTS_PER_SESSION",
            "HYMEM_AGGREGATION_NODES_ENABLED",
            "HYMEM_AGGREGATION_DIGEST_ENABLED",
        }:
            del os.environ[key]
    os.environ.update(candidates["hymem-honcho"][0])
    os.environ.update(honcho_switches[0])


def _clients_and_config(path: Path):
    env = bootstrap.resolve_env()
    if not env.has_llm_key or env.root != LIVE_PATHS[1].parent:
        raise ValueError("configured Hermes1 LLM/root unavailable")
    llm = OpenAICompatibleClient(
        api_key=env.llm_api_key, base_url=env.llm_base_url,
        model=env.llm_model,
    )
    try:
        cfg_kwargs = {}
        if env.aggregation_nodes_enabled is not None:
            cfg_kwargs["aggregation_nodes_enabled"] = (
                env.aggregation_nodes_enabled
            )
        if env.aggregation_digest_enabled is not None:
            cfg_kwargs["aggregation_digest_enabled"] = (
                env.aggregation_digest_enabled
            )
        cfg = HyMemConfig(root=env.root, **cfg_kwargs)
        if (
            not llm.transport_integrity_ok
            or cfg.chunk_extraction_max_attempts != EXPECTED_ATTEMPTS
        ):
            raise ValueError("reviewed Phase-1 producer contract unavailable")
        generation = phase1_generation_binding(cfg.prompt_version, llm)
        if not generation["producer"]["identity_exact"]:
            raise ValueError("Phase-1 producer identity is not exact")
        embedding_env, embedder = _configured_embedder(
            apply=True, allow_local=False,
        )
        route = urlsplit(embedding_env.embedding_base_url)
        if (
            embedding_env.embedding_backend != "openai_compatible"
            or route.scheme != "http"
            or route.hostname != "embedding-server"
            or route.port != 8766
            or route.path.rstrip("/") != "/v1"
            or route.username is not None or route.password is not None
            or route.query or route.fragment
        ):
            raise ValueError("embedding route is not Hermes1 local service")
        model, dim = embedding_storage_identity(embedder)
        if dim != 384:
            raise ValueError("embedding dimension changed")
        return cfg, llm, embedder, generation, model, dim
    except BaseException:
        llm.close()
        raise


def _verify_vectors(conn: sqlite3.Connection, model: str, dim: int,
                    expected_ids: frozenset[int]) -> None:
    metadata = dict(conn.execute(
        "SELECT key,value FROM schema_meta WHERE key IN ('vec_model','vec_dim')"
    ).fetchall())
    if metadata != {"vec_model": model, "vec_dim": str(dim)}:
        raise ValueError("edge vector producer changed")
    actual = {int(row[0]) for row in conn.execute(
        "SELECT rowid FROM vec_edges"
    )}
    if actual != expected_ids:
        raise ValueError("physical edge mirror ID mismatch")
    checked = 0
    for row in conn.execute(
        "SELECT v.embedding,c.vector_json FROM vec_edges v "
        "JOIN knowledge_graph kg ON kg.id=v.rowid "
        "LEFT JOIN edge_embeddings c ON c.edge_text="
        "(kg.subject_canonical || ' ' || kg.predicate || ' ' || "
        "kg.object_canonical) AND c.model=? AND c.dim=?",
        (model, dim),
    ):
        if row["vector_json"] is None or row["embedding"] != db._pack_vector(
            decode_vector(row["vector_json"])
        ):
            raise ValueError("physical edge vector cache mismatch")
        checked += 1
    if checked != len(expected_ids):
        raise ValueError("physical edge vector cardinality changed")


def _live_texts(conn: sqlite3.Connection) -> dict[int, str]:
    return {
        int(row["id"]): " ".join((
            row["subject_canonical"], row["predicate"],
            row["object_canonical"],
        ))
        for row in conn.execute(
            "SELECT id,subject_canonical,predicate,object_canonical "
            f"FROM knowledge_graph WHERE {live_edge_predicate()}"
        )
    }


def _create_backup(target: Path, destination: Path, plan: Plan,
                   generation: dict, holder: str,
                   model: str, dim: int) -> None:
    parent = destination.parent
    if (
        not destination.is_absolute() or not parent.is_dir()
        or parent.is_symlink() or parent.resolve(strict=True) != parent
        or stat.S_IMODE(parent.stat().st_mode) != 0o700
        or destination.exists() or destination.is_symlink()
        or destination.resolve() == target.resolve()
    ):
        raise ValueError("backup path must be new, distinct, and private")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600)
    os.close(fd)
    try:
        with closing(sqlite3.connect(
            target.resolve().as_uri() + "?mode=ro", uri=True,
        )) as src:
            with closing(sqlite3.connect(destination)) as out:
                src.backup(out)
        # The backup must represent the pre-repair store, not strand this
        # operator's temporary lease if it is ever restored.
        with closing(sqlite3.connect(destination, isolation_level=None)) as out:
            removed = out.execute(
                "DELETE FROM run_lock WHERE name='dreaming' AND holder=?",
                (holder,),
            ).rowcount
            if removed != 1:
                raise ValueError("backup lease snapshot changed")
            checkpoint = out.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if checkpoint is None or checkpoint[0] != 0:
                raise ValueError("backup checkpoint failed")
        with db.read_snapshot(destination) as check:
            sqlite_vec.load(check)
            saved = _plan(check, generation)
            if (
                saved != plan
                or check.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or check.execute("PRAGMA foreign_key_check").fetchone()
                is not None
            ):
                raise ValueError("backup differs from reviewed target state")
            _verify_vectors(check, model, dim, plan.live_ids)
    except BaseException:
        _delete_sqlite_files(destination)
        raise


def _assert_integrity(conn: sqlite3.Connection) -> None:
    if (
        conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
        or conn.execute("PRAGMA foreign_key_check").fetchone() is not None
    ):
        raise ValueError("postflight SQLite integrity failed")


def _delete_sqlite_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def _staged_success(backup: Path, extraction: phase1.ChunkExtraction,
                    plan: Plan, cfg: HyMemConfig, dedup_vectors: dict,
                    embedder: object, model: str, dim: int,
                    generation: dict) -> tuple[PendingEdgeEmbeddings, Path]:
    with tempfile.NamedTemporaryFile(
        prefix="hymem-phase1-stage-", suffix=".sqlite",
        dir=backup.parent, delete=False,
    ) as temporary:
        stage_path = Path(temporary.name)
    try:
        with closing(sqlite3.connect(
            backup.resolve().as_uri() + "?mode=ro", uri=True,
        )) as source:
            with closing(sqlite3.connect(stage_path)) as target:
                source.backup(target)
        with closing(db.connect(stage_path)) as staged:
            sqlite_vec.load(staged)
            if _plan(staged, generation) != plan:
                raise ValueError("staging baseline changed")
            with db.transaction(staged):
                phase1.persist_chunk_results(
                    staged, plan.chunk, extraction,
                    prompt_version=cfg.prompt_version, cfg=cfg,
                    dedup_vectors=dedup_vectors,
                )
            pending = fetch_edge_embeddings(staged, embedder)
            if pending is None or (pending.model, pending.dim) != (model, dim):
                raise ValueError("staged edge vector producer changed")
            with db.transaction(staged):
                persist_edge_embeddings(staged, pending)
                _verify_vectors(
                    staged, model, dim,
                    frozenset(pending.edge_text_by_id),
                )
                _assert_integrity(staged)
            if (
                staged.execute(
                    "SELECT COUNT(*) FROM current_phase1_publications "
                    "WHERE chunk_id=?", (plan.chunk.id,),
                ).fetchone()[0] != 1
                or staged.execute(
                    "SELECT COUNT(*) FROM chunk_extraction_attempts "
                    "WHERE chunk_id=? AND prompt_version=?",
                    (plan.chunk.id, generation["extraction_cache_key"]),
                ).fetchone()[0] != 0
                or staged.execute(
                    "SELECT is_current FROM kg_evidence WHERE id=?",
                    (plan.evidence_id,),
                ).fetchone()[0] != 0
                or _aggregation_state(staged) != plan.aggregation_state
            ):
                raise ValueError("staged claim publication incomplete")
            return pending, stage_path
    except BaseException:
        _delete_sqlite_files(stage_path)
        raise


def _apply(target: Path, backup: Path, expected: Plan, cfg: HyMemConfig,
           llm: OpenAICompatibleClient, embedder: object,
           generation: dict, model: str, dim: int) -> dict[str, object]:
    if "completion_call_limit" not in inspect.signature(
        phase1.extract_chunk_results
    ).parameters:
        raise ValueError("bounded Phase-1 path is not deployed")
    with closing(db.connect(target)) as conn:
        sqlite_vec.load(conn)
        with db.transaction(conn):
            if _plan(conn, generation) != expected:
                raise ValueError("target changed before lease acquisition")
            holder = secrets.token_hex(16)
            inserted = conn.execute(
                "INSERT INTO run_lock(name,acquired_at,holder) "
                "VALUES ('dreaming',CURRENT_TIMESTAMP,?) "
                "ON CONFLICT(name) DO NOTHING", (holder,),
            ).rowcount
            if inserted != 1:
                raise ValueError("dreaming lease is occupied")
        fence = None
        heartbeat = None
        stage_path = None
        try:
            fence = db.activate_transaction_lease_fence(
                conn, name="dreaming", holder=holder,
            )
            heartbeat = runner._DreamLeaseHeartbeat(
                conn, holder, interval_seconds=30,
            )
            heartbeat.start()
            if _plan(conn, generation, holder=holder) != expected:
                raise ValueError("target changed after lease acquisition")
            _verify_vectors(conn, model, dim, expected.live_ids)
            _create_backup(
                target, backup, expected, generation, holder, model, dim,
            )
            heartbeat.check()
            runner._refresh_lock(conn, holder)
            before_http = int(llm.request_attempts)
            extraction = phase1.extract_chunk_results(
                conn, expected.chunk, llm,
                prompt_version=cfg.prompt_version,
                phase1_generation=generation,
                completion_call_limit=CALL_LIMIT,
            )
            heartbeat.check()
            runner._refresh_lock(conn, holder)
            actual_http = int(llm.request_attempts) - before_http
            if extraction is None or (
                extraction.completion_calls > CALL_LIMIT
                or extraction.provider_attempts > HTTP_LIMIT
                or actual_http > HTTP_LIMIT
                or actual_http != extraction.provider_attempts
                or extraction.phase1_generation != generation
                or not extraction.source_validated
            ):
                raise ValueError("bounded extraction contract changed")
            if extraction.failed:
                with db.transaction(conn):
                    if _plan(conn, generation, holder=holder) != expected:
                        raise ValueError("target changed before failure audit")
                    phase1.persist_chunk_results(
                        conn, expected.chunk, extraction,
                        prompt_version=cfg.prompt_version, cfg=cfg,
                    )
                    retry = conn.execute(
                        "SELECT attempts,phase1_generation_key "
                        "FROM chunk_extraction_attempts "
                        "WHERE chunk_id=? AND prompt_version=?",
                        (expected.chunk.id,
                         generation["extraction_cache_key"]),
                    ).fetchone()
                    if (
                        retry is None or int(retry["attempts"]) != 4
                        or retry["phase1_generation_key"]
                        != generation["generation_key"]
                        or conn.execute(
                            "SELECT 1 FROM current_phase1_publications "
                            "WHERE chunk_id=?", (expected.chunk.id,),
                        ).fetchone() is not None
                        or conn.execute(
                            "SELECT is_current FROM kg_evidence WHERE id=?",
                            (expected.evidence_id,),
                        ).fetchone()[0] != 1
                        or _claim_digest(conn, expected.chunk.id)
                        != expected.claim_digest
                        or frozenset(_live_texts(conn)) != expected.live_ids
                        or _aggregation_state(conn) != expected.aggregation_state
                    ):
                        raise ValueError("failed extraction changed claim authority")
                    _verify_vectors(conn, model, dim, expected.live_ids)
                    _assert_integrity(conn)
                return {
                    "status": "held_after_bounded_failure",
                    "completion_calls": extraction.completion_calls,
                    "provider_attempts": extraction.provider_attempts,
                    "retry_attempts": 4,
                    "failure_reason": extraction.failure_reason,
                    "live_direct_edges": len(expected.live_ids),
                }

            dedup_vectors = phase1.prepare_dedup_vectors(
                conn, extraction, cfg, embedder,
            )
            # The normal writer may skip semantic dedup if its best-effort
            # prefetch fails. The operator route requires every new candidate.
            from hymem.dreaming import canonicalize
            candidate_texts = set()
            for triple in extraction.triples:
                subject = canonicalize.resolve(conn, triple.subject)
                object_ = canonicalize.resolve(conn, triple.object)
                if conn.execute(
                    "SELECT 1 FROM knowledge_graph WHERE "
                    "subject_canonical=? AND predicate=? AND object_canonical=?",
                    (subject, triple.predicate, object_),
                ).fetchone() is None:
                    candidate_texts.add(
                        f"{subject} {triple.predicate} {object_}"
                    )
            if set(dedup_vectors) != candidate_texts:
                raise ValueError("dedup vector prefetch incomplete")
            heartbeat.check()
            runner._refresh_lock(conn, holder)
            pending, stage_path = _staged_success(
                backup, extraction, expected, cfg, dedup_vectors,
                embedder, model, dim, generation,
            )
            heartbeat.check()
            runner._refresh_lock(conn, holder)
            if phase1_generation_binding(cfg.prompt_version, llm) != generation:
                raise ValueError("Phase-1 producer changed before publication")
            if embedding_storage_identity(embedder) != (model, dim):
                raise ValueError("embedding producer changed before publication")
            with db.transaction(conn):
                if _plan(conn, generation, holder=holder) != expected:
                    raise ValueError("target changed before publication")
                _verify_vectors(conn, model, dim, expected.live_ids)
                phase1.persist_chunk_results(
                    conn, expected.chunk, extraction,
                    prompt_version=cfg.prompt_version, cfg=cfg,
                    dedup_vectors=dedup_vectors,
                )
                live_texts = _live_texts(conn)
                if live_texts != pending.edge_text_by_id:
                    raise ValueError("live graph differs from staged publication")
                persisted = persist_edge_embeddings(conn, pending)
                live_ids = frozenset(live_texts)
                _verify_vectors(conn, model, dim, live_ids)
                if (
                    conn.execute(
                        "SELECT COUNT(*) FROM current_phase1_publications "
                        "WHERE chunk_id=?", (expected.chunk.id,),
                    ).fetchone()[0] != 1
                    or conn.execute(
                        "SELECT COUNT(*) FROM chunk_extraction_attempts "
                        "WHERE chunk_id=? AND prompt_version=?",
                        (expected.chunk.id,
                         generation["extraction_cache_key"]),
                    ).fetchone()[0] != 0
                    or conn.execute(
                        "SELECT is_current FROM kg_evidence WHERE id=?",
                        (expected.evidence_id,),
                    ).fetchone()[0] != 0
                    or _aggregation_state(conn) != expected.aggregation_state
                ):
                    raise ValueError("live claim publication incomplete")
                _assert_integrity(conn)
            return {
                "status": "published", "completion_calls":
                extraction.completion_calls,
                "provider_attempts": extraction.provider_attempts,
                "observations": len(extraction.triples),
                "live_direct_edges": len(live_ids),
                "embedded_edge_vectors": persisted,
                "vector_mirror_mismatches": 0,
            }
        finally:
            try:
                if stage_path is not None:
                    _delete_sqlite_files(stage_path)
            finally:
                try:
                    if heartbeat is not None:
                        heartbeat.stop()
                finally:
                    try:
                        if fence is not None:
                            db.deactivate_transaction_lease_fence(fence)
                    finally:
                        released = runner._release_lock(conn, holder)
                        if not released and sys.exc_info()[0] is None:
                            raise ValueError(
                                "operator dreaming lease release failed"
                            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--server-process-env", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--expect-source")
    parser.add_argument("--expect-generation")
    parser.add_argument("--expect-retry")
    parser.add_argument("--expect-state")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    llm = embedder = None
    try:
        live = _is_live(args.db)
        if live and args.db.resolve() != LIVE_PATHS[1]:
            raise ValueError("live operator must run inside Hermes1 container")
        if live and not args.server_process_env:
            raise ValueError("live operator requires server process environment")
        if args.apply and not args.allow_provider_calls:
            raise ValueError("apply requires explicit provider-call approval")
        if args.server_process_env:
            _load_server_process_env()
        cfg, llm, embedder, generation, model, dim = _clients_and_config(
            args.db,
        )
        with db.read_snapshot(args.db) as snapshot:
            sqlite_vec.load(snapshot)
            plan = _plan(snapshot, generation)
            _verify_vectors(snapshot, model, dim, plan.live_ids)
            _assert_integrity(snapshot)
        if not args.apply:
            print(json.dumps({
                "status": "preview", "live_target": live, **plan.summary(),
            }, sort_keys=True))
            return 0
        if live and not args.allow_live:
            raise ValueError("live target requires --allow-live")
        if args.backup is None:
            raise ValueError("apply requires new private backup")
        expectations = (
            args.expect_source, args.expect_generation,
            args.expect_retry, args.expect_state,
        )
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in expectations
        ) or expectations != (
            plan.source_digest, plan.generation_digest,
            plan.retry_digest, plan.state_digest,
        ):
            raise ValueError("reviewed source/generation/retry/state changed")
        result = _apply(
            args.db, args.backup, plan, cfg, llm, embedder,
            generation, model, dim,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__},
                         sort_keys=True))
        return 1
    finally:
        for client in (embedder, llm):
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
