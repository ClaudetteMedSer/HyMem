"""Retire historical unpublished Phase-3 evidence and refresh edge vectors.

Preview is read-only. Apply requires a reviewed candidate fingerprint and a new
private backup path. The embedding endpoint must be the local Hermes1 service.
No claim text, source text, vectors, or credentials are printed.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite_vec

from hymem.core import db
from hymem.core.embedding_batches import embed_bounded
from hymem.core.graph import live_edge_predicate
from hymem.core.time import normalize_iso_timestamp
from hymem.core.vectors import decode_vector
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.bitemporal import recompute_edge_interval
from hymem.dreaming.embeddings import PendingEdgeEmbeddings, persist_edge_embeddings
from hymem.dreaming.evidence import ledger_counts, reconcile_edge_counts
from hymem.reembed import _configured_embedder


LIVE_PATHS = (
    Path("/opt/stacks/hermes/instance1/home/.hermes/hymem.sqlite"),
    Path("/home/node/.hermes/hymem.sqlite"),
)
REASON = "unpublished_historical_phase3_retired"
BASELINE = (464, 455, 9759, 443)
LIVE_BASELINE_DIGEST = (
    "a069a37b8cb1fece8fc3ad6924288b009a1ee92f92bb61090e25579897cac981"
)
LIVE_COHORT_DIGEST = (
    "b8d3197381ed7fac0928b54dbff013da193ded8f050b281aa07275a9af76f39a"
)
LIVE_PRE_IDS_DIGEST = (
    "17dd1127f9e8143f0cf15964dd7200b554dad3ede7cbb13b118b8ca7e4aa0659"
)
LIVE_POST_IDS_DIGEST = (
    "a582cbbe94f474669cf9faf79582f24aef8e5609bcd39bf00ee70c99088f3eca"
)
SELECTOR = (
    "e.evidence_kind IN ('reinforcement','decay') "
    "AND e.provenance_status='canonical' AND e.is_current=1 "
    "AND e.published_at IS NULL"
)


@dataclass(frozen=True)
class Plan:
    evidence_ids: tuple[int, ...]
    edge_ids: tuple[int, ...]
    digest: str
    cohort_digest: str
    lifecycle_digest: str
    live_ids: frozenset[int]
    aggregation_state: tuple[int, int, int]
    prior_retired: int

    def summary(self) -> dict[str, object]:
        return {
            "candidate_rows": len(self.evidence_ids),
            "affected_edges": len(self.edge_ids),
            "candidate_sha256": self.digest,
            "live_direct_edges": len(self.live_ids),
            "previously_retired": self.prior_retired,
            "aggregation_root_enabled": self.aggregation_state[0],
        }


def _is_live_path(path: Path) -> bool:
    return any(
        path.resolve() == known.resolve()
        or (path.exists() and known.exists() and path.samefile(known))
        for known in LIVE_PATHS
    )


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


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()).hexdigest()


def _cohort_digest(rows: list[sqlite3.Row]) -> str:
    excluded = {"is_current", "superseded_at", "superseded_reason"}
    columns = [key for key in rows[0].keys() if key not in excluded] if rows else []
    return _digest([[row[key] for key in columns] for row in rows])


def _retired_cohort_digest(conn: sqlite3.Connection) -> tuple[int, str]:
    rows = conn.execute(
        "SELECT * FROM kg_evidence WHERE is_current=0 "
        "AND superseded_reason=? "
        "AND evidence_kind IN ('reinforcement','decay') ORDER BY id",
        (REASON,),
    ).fetchall()
    return len(rows), _cohort_digest(rows)


def _lifecycle_digest(conn: sqlite3.Connection,
                      edge_ids: tuple[int, ...]) -> str:
    if not edge_ids:
        return _digest([])
    rows = conn.execute(
        "SELECT * FROM kg_edge_lifecycle WHERE edge_id IN ("
        + ",".join("?" for _ in edge_ids) + ") ORDER BY id",
        edge_ids,
    ).fetchall()
    return _digest([list(row) for row in rows])


def _assert_dream_idle(conn: sqlite3.Connection) -> None:
    # A stale lease is still ambiguous here; a dream may be reclaiming it.
    # Refuse any dreaming row instead of trying to arbitrate ownership.
    if conn.execute(
        "SELECT 1 FROM run_lock WHERE name='dreaming' LIMIT 1"
    ).fetchone() is not None:
        raise ValueError("dreaming lease is present")


def _plan(conn: sqlite3.Connection) -> Plan:
    schema = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    if schema is None or int(schema[0]) != 64:
        raise ValueError("schema mismatch")
    _assert_dream_idle(conn)
    rows = conn.execute(
        f"SELECT e.* FROM kg_evidence e WHERE {SELECTOR} ORDER BY e.id"
    ).fetchall()
    for row in rows:
        expected = (
            (row["evidence_kind"] == "reinforcement"
             and row["polarity"] == 1
             and row["weight_source"] == "fixed_unattributed_comention:1")
            or (row["evidence_kind"] == "decay"
                and row["polarity"] == -1
                and row["weight_source"] == "fixed_soft_decay:1")
        )
        if not (
            expected and row["evidence_weight"] == 1
            and row["extraction_prompt_version"] is None
            and str(row["extracted_at"]).startswith("2026-09-05 ")
            and row["source_message_id"] is not None
            and row["source_session_id"] is not None
            and row["source_coverage_chunk_id"] is not None
            and row["source_coverage_version"] is not None
            and conn.execute(
                "SELECT 1 FROM kg_claim_observations WHERE evidence_id=? LIMIT 1",
                (row["id"],),
            ).fetchone() is None
        ):
            raise ValueError("unpublished synthetic evidence has an unknown shape")
    ids = tuple(int(row["id"]) for row in rows)
    edge_ids = tuple(sorted({int(row["edge_id"]) for row in rows}))
    graph_rows = conn.execute(
        "SELECT * FROM knowledge_graph WHERE id IN ("
        + ",".join("?" for _ in edge_ids) + ") ORDER BY id",
        edge_ids,
    ).fetchall() if edge_ids else []
    if len(graph_rows) != len(edge_ids):
        raise ValueError("affected graph edge is missing")
    material = [
        [list(row) for row in rows],
        [list(row) for row in graph_rows],
    ]
    digest = _digest(material)
    live_ids = frozenset(int(row[0]) for row in conn.execute(
        f"SELECT id FROM knowledge_graph WHERE {live_edge_predicate()}"
    ))
    retired = int(conn.execute(
        "SELECT COUNT(*) FROM kg_evidence WHERE superseded_reason=? "
        "AND evidence_kind IN ('reinforcement','decay') AND is_current=0",
        (REASON,),
    ).fetchone()[0])
    return Plan(
        ids, edge_ids, digest, _cohort_digest(rows),
        _lifecycle_digest(conn, edge_ids), live_ids,
        _aggregation_state(conn), retired,
    )


def _edge_texts(conn: sqlite3.Connection, ids: tuple[int, ...] | None = None
                ) -> dict[int, str]:
    if ids is None:
        rows = conn.execute(
            "SELECT id,subject_canonical,predicate,object_canonical "
            f"FROM knowledge_graph WHERE {live_edge_predicate()}"
        )
    elif ids:
        rows = conn.execute(
            "SELECT id,subject_canonical,predicate,object_canonical "
            "FROM knowledge_graph WHERE id IN ("
            + ",".join("?" for _ in ids) + ")",
            ids,
        )
    else:
        return {}
    return {
        int(row["id"]): " ".join((
            row["subject_canonical"], row["predicate"],
            row["object_canonical"],
        ))
        for row in rows
    }


def _verify_vectors(conn: sqlite3.Connection, *, model: str, dim: int,
                    expected_ids: frozenset[int]) -> None:
    metadata = dict(conn.execute(
        "SELECT key,value FROM schema_meta WHERE key IN ('vec_model','vec_dim')"
    ).fetchall())
    if metadata.get("vec_model") != model or metadata.get("vec_dim") != str(dim):
        raise ValueError("physical edge mirror producer identity changed")
    physical = {int(row[0]) for row in conn.execute("SELECT rowid FROM vec_edges")}
    if physical != expected_ids:
        raise ValueError("physical edge mirror does not match live graph")
    rows = conn.execute(
        "SELECT vec.rowid,vec.embedding,cache.vector_json FROM vec_edges vec "
        "JOIN knowledge_graph kg ON kg.id=vec.rowid "
        "LEFT JOIN edge_embeddings cache ON cache.edge_text="
        "(kg.subject_canonical || ' ' || kg.predicate || ' ' || kg.object_canonical) "
        "AND cache.model=? AND cache.dim=?",
        (model, dim),
    )
    matched = 0
    for row in rows:
        if row["vector_json"] is None:
            raise ValueError("current model vector cache has a gap")
        try:
            vector = decode_vector(row["vector_json"])
            identical = row["embedding"] == db._pack_vector(vector)
        except (TypeError, ValueError, OverflowError):
            identical = False
        if not identical:
            raise ValueError("physical edge vector differs from durable cache")
        matched += 1
    if matched != len(expected_ids):
        raise ValueError("physical edge mirror cardinality mismatch")


def _load_server_embedding_env() -> None:
    for process in sorted(Path("/proc").iterdir()):
        if not process.name.isdecimal():
            continue
        try:
            entries = (process / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        variables = dict(item.split(b"=", 1) for item in entries if b"=" in item)
        if b"HYMEM_EMBEDDING_BASE_URL" not in variables:
            continue
        for key, value in variables.items():
            if key.startswith(b"HYMEM_EMBEDDING_"):
                os.environ[key.decode()] = value.decode()
        return
    raise RuntimeError("server embedding configuration unavailable")


def _local_embedder(*, server_process_env: bool):
    if server_process_env:
        _load_server_embedding_env()
    cfg, client = _configured_embedder(apply=True, allow_local=False)
    endpoint = urlsplit(cfg.embedding_base_url)
    if (
        cfg.embedding_backend != "openai_compatible"
        or endpoint.scheme != "http"
        or endpoint.hostname != "embedding-server"
        or endpoint.port != 8766
        or endpoint.path.rstrip("/") != "/v1"
    ):
        close = getattr(client, "close", None)
        if callable(close):
            close()
        raise ValueError("embedding endpoint is not Hermes1 local service")
    return client


def _create_backup(target: Path, destination: Path, baseline: Plan) -> None:
    if destination.exists() or destination.resolve() == target.resolve():
        raise ValueError("backup path must be new and distinct")
    mode = stat.S_IMODE(destination.parent.stat().st_mode)
    if mode & 0o077:
        raise ValueError("backup directory must be private")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600)
    os.close(fd)
    try:
        with sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True) as src:
            with sqlite3.connect(destination) as backup:
                src.backup(backup)
        with db.read_snapshot(destination) as check:
            sqlite_vec.load(check)
            saved = _plan(check)
            if (
                saved.digest != baseline.digest
                or saved.cohort_digest != baseline.cohort_digest
                or saved.lifecycle_digest != baseline.lifecycle_digest
                or saved.live_ids != baseline.live_ids
                or saved.aggregation_state != baseline.aggregation_state
                or saved.prior_retired != baseline.prior_retired
                or check.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or check.execute("PRAGMA foreign_key_check").fetchone() is not None
            ):
                raise ValueError("backup differs from reviewed baseline")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _apply(conn: sqlite3.Connection, baseline: Plan, client: object,
           *, expect_restored: int) -> dict[str, object]:
    model, dim = embedding_storage_identity(client)
    if dim != 384:
        raise ValueError("embedding dimension differs from live baseline")
    _verify_vectors(conn, model=model, dim=dim, expected_ids=baseline.live_ids)
    affected = _edge_texts(conn, baseline.edge_ids)
    missing = sorted({text for text in affected.values() if conn.execute(
        "SELECT 1 FROM edge_embeddings WHERE edge_text=? AND model=? AND dim=?",
        (text, model, dim),
    ).fetchone() is None})
    vectors = embed_bounded(
        client, missing, identity=lambda: embedding_storage_identity(client),
        expected_model=model, required_dim=dim,
    )
    if embedding_storage_identity(client) != (model, dim):
        raise ValueError("embedding producer changed")
    prepared = dict(zip(missing, vectors))
    with db.transaction(conn):
        # BEGIN IMMEDIATE fences a new lease until this repair commits or rolls
        # back. Recheck after the slow embedding prefetch and before mutations.
        _assert_dream_idle(conn)
        current = _plan(conn)
        if (
            current.digest != baseline.digest
            or current.cohort_digest != baseline.cohort_digest
            or current.lifecycle_digest != baseline.lifecycle_digest
            or current.live_ids != baseline.live_ids
            or current.aggregation_state != baseline.aggregation_state
            or current.prior_retired != baseline.prior_retired
        ):
            raise ValueError("database changed after preview")
        _verify_vectors(conn, model=model, dim=dim,
                        expected_ids=baseline.live_ids)
        now = normalize_iso_timestamp(
            conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0],
            context="synthetic retirement",
        )
        with db.evidence_mutation(conn):
            changed = conn.execute(
                "UPDATE kg_evidence SET is_current=0,superseded_at=?,"
                "superseded_reason=? WHERE id IN ("
                + ",".join("?" for _ in baseline.evidence_ids) + ") "
                "AND is_current=1 AND published_at IS NULL",
                (now, REASON, *baseline.evidence_ids),
            ).rowcount
        if changed != len(baseline.evidence_ids):
            raise ValueError("candidate evidence changed")
        reconcile_edge_counts(conn, baseline.edge_ids)
        for edge_id in baseline.edge_ids:
            recompute_edge_interval(conn, edge_id)
            edge = conn.execute(
                "SELECT pos_evidence,neg_evidence FROM knowledge_graph WHERE id=?",
                (edge_id,),
            ).fetchone()
            if edge is None or tuple(edge) != ledger_counts(conn, edge_id):
                raise ValueError("affected edge counts do not match ledger")
        after = _plan(conn)
        restored = after.live_ids - baseline.live_ids
        if (
            after.evidence_ids
            or len(restored) != expect_restored
            or (baseline.live_ids - after.live_ids)
            or not restored.issubset(baseline.edge_ids)
            or after.aggregation_state != baseline.aggregation_state
            or _lifecycle_digest(conn, baseline.edge_ids)
            != baseline.lifecycle_digest
            or _digest(sorted(after.live_ids)) != LIVE_POST_IDS_DIGEST
        ):
            raise ValueError("graph poststate differs from reviewed repair")
        live_texts = _edge_texts(conn)
        uncached_live = {text for text in live_texts.values() if conn.execute(
            "SELECT 1 FROM edge_embeddings WHERE edge_text=? AND model=? AND dim=?",
            (text, model, dim),
        ).fetchone() is None}
        if not uncached_live.issubset(prepared):
            raise ValueError("a live edge lacks a prepared vector")
        persisted = persist_edge_embeddings(
            conn, PendingEdgeEmbeddings(
                edge_text_by_id=live_texts,
                new_text_vectors={text: prepared[text] for text in uncached_live},
                truly_new_texts=set(uncached_live), dim=dim, model=model,
                cache_hits=0,
            ),
        )
        _verify_vectors(conn, model=model, dim=dim,
                        expected_ids=after.live_ids)
        if (
            _plan(conn).aggregation_state != baseline.aggregation_state
            or conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or conn.execute("PRAGMA foreign_key_check").fetchone() is not None
        ):
            raise ValueError("postflight integrity or aggregation changed")
    return {
        "status": "applied", "retired": changed,
        "affected_edges": len(baseline.edge_ids),
        "restored_live_edges": len(restored),
        "provider_embedded_texts": len(missing),
        "embedded_vectors": persisted,
        "live_direct_edges": len(after.live_ids),
        "vector_mirror_mismatches": 0,
        "aggregation_state_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--server-process-env", action="store_true")
    parser.add_argument("--expect-candidates", type=int)
    parser.add_argument("--expect-edges", type=int)
    parser.add_argument("--expect-live-before", type=int)
    parser.add_argument("--expect-restored", type=int)
    parser.add_argument("--expect-digest")
    args = parser.parse_args()
    try:
        with db.read_snapshot(args.db) as snapshot:
            sqlite_vec.load(snapshot)
            plan = _plan(snapshot)
        live = _is_live_path(args.db)
        if plan.aggregation_state[0] != 0:
            raise ValueError("aggregation root policy changed")
        if not args.apply:
            print(json.dumps({"status": "preview", "live_target": live,
                              **plan.summary()}, sort_keys=True))
            return 0
        if live and not args.allow_live:
            raise ValueError("live apply requires --allow-live")
        if args.backup is None:
            raise ValueError("apply requires a private --backup destination")
        if not all((
            args.expect_candidates is not None,
            args.expect_edges is not None,
            args.expect_live_before is not None,
            args.expect_restored is not None,
            isinstance(args.expect_digest, str) and len(args.expect_digest) == 64,
        )):
            raise ValueError("apply requires exact reviewed expectations")
        if (
            not plan.evidence_ids and args.expect_candidates > 0
            and plan.prior_retired == args.expect_candidates
            and (not live or plan.prior_retired == BASELINE[0])
        ):
            if (
                args.expect_candidates != BASELINE[0]
                or args.expect_edges != BASELINE[1]
                or args.expect_live_before != BASELINE[2]
                or args.expect_restored != BASELINE[3]
                or len(plan.live_ids) != args.expect_live_before + args.expect_restored
                or plan.aggregation_state != (0, 19, 1)
                or (live and args.expect_digest != LIVE_BASELINE_DIGEST)
                or _digest(sorted(plan.live_ids)) != LIVE_POST_IDS_DIGEST
            ):
                raise ValueError("repaired live baseline changed")
            client = _local_embedder(server_process_env=args.server_process_env)
            try:
                model, dim = embedding_storage_identity(client)
                if dim != 384:
                    raise ValueError("repaired embedding dimension changed")
                with db.read_snapshot(args.db) as repaired:
                    sqlite_vec.load(repaired)
                    confirmed = _plan(repaired)
                    retired_count, retired_digest = _retired_cohort_digest(repaired)
                    if (
                        confirmed != plan
                        or retired_count != BASELINE[0]
                        or retired_digest != LIVE_COHORT_DIGEST
                    ):
                        raise ValueError("retired cohort changed")
                    _verify_vectors(repaired, model=model, dim=dim,
                                    expected_ids=plan.live_ids)
                    if (
                        repaired.execute("PRAGMA integrity_check").fetchone()[0]
                        != "ok"
                        or repaired.execute("PRAGMA foreign_key_check").fetchone()
                        is not None
                    ):
                        raise ValueError("repaired store integrity changed")
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            print(json.dumps({"status": "already_applied", **plan.summary()},
                             sort_keys=True))
            return 0
        if (
            len(plan.evidence_ids) != args.expect_candidates
            or len(plan.edge_ids) != args.expect_edges
            or len(plan.live_ids) != args.expect_live_before
            or plan.digest != args.expect_digest
        ):
            raise ValueError("reviewed baseline changed")
        if live and (
            len(plan.evidence_ids), len(plan.edge_ids), len(plan.live_ids),
            args.expect_restored,
        ) != BASELINE:
            raise ValueError("Hermes1 live baseline changed")
        if live and plan.digest != LIVE_BASELINE_DIGEST:
            raise ValueError("Hermes1 reviewed cohort changed")
        if (
            plan.cohort_digest != LIVE_COHORT_DIGEST
            or _digest(sorted(plan.live_ids)) != LIVE_PRE_IDS_DIGEST
        ):
            raise ValueError("reviewed source cohort or live graph changed")
        if live and plan.aggregation_state != (0, 19, 1):
            raise ValueError("Hermes1 aggregation baseline changed")
        if plan.prior_retired:
            raise ValueError("partial prior synthetic retirement exists")
        _create_backup(args.db, args.backup, plan)
        client = _local_embedder(server_process_env=args.server_process_env)
        try:
            with closing(db.connect(args.db)) as conn:
                sqlite_vec.load(conn)
                result = _apply(conn, plan, client,
                                expect_restored=args.expect_restored)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
