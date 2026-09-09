"""Bounded, resumable repair of existing vector mirrors; never runs an LLM.

This is not source regeneration. Missing mirrors, unproven historical rows,
and aggregation publications requiring new material are deliberately excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sqlite3

from hymem.core import db
from hymem.core.graph import live_edge_predicate
from hymem.core.vectors import encode_vector
from hymem.deadline import DeadlineExceeded, MonotonicDeadline, use_deadline
from hymem.embedding_health import MIRROR_TABLES, _valid_vector, scan_embedding_health
from hymem.dreaming.aggregation_material import embedding_execution_identity
from hymem.extraction.embeddings import embedding_text_hash

_KEYS = ("chunk_id", "message_id", "edge_text", "episode_id", "fact_id", "node_id")
_SHADOWS = ("vec_chunks", "vec_messages", "vec_edges", "vec_episodes", "vec_facts")


class ConfigurationError(ValueError):
    """Safe operational reason; never stores rejected configuration bytes."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass
class RepairReport:
    mode: str
    status: str = "incomplete"
    scanned: int = 0
    current: int = 0
    pending: int = 0
    repaired: int = 0
    blocked: int = 0
    rebuild_required: int = 0
    provider_batches: int = 0
    sweep_complete: bool = False
    sweep_blocked: int = 0
    error_code: str | None = None
    health_before: dict = field(default_factory=dict)
    health_after: dict = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "complete" else 1


def _identity(embedder):
    binding, model, dim = embedding_execution_identity(embedder)
    if binding["identity_exact"] is not True or binding["reuse_scope"] != "durable" or dim is None:
        raise RuntimeError("inexact producer")
    if getattr(embedder, "fallback_reason", None):
        raise RuntimeError("embedding fallback refused")
    return binding, model, dim


def _compatible(row, table, model, dim):
    return (row["model"] == model and row["dim"] == dim
            and (table not in ("episode_embeddings", "aggregation_node_embeddings")
                 or row["embedding_producer_key"] == model)
            and _valid_vector(row["vector_json"], dim))


def _source(conn, index, row):
    """Return exact stored text, revalidation token, and shadow row ids.

    The same read-only proof is evaluated before the provider and under the
    publication writer lock. No raw-message or source-manifest repair occurs.
    """
    key = row[_KEYS[index]]
    if index == 0:
        from hymem.dreaming.chunks import Chunk
        from hymem.dreaming.phase1 import _claim_sources_for_chunk
        source = conn.execute("SELECT rowid AS vector_rowid,* FROM chunks WHERE id=? AND chunk_kind='extraction'", (key,)).fetchone()
        if source is None:
            return None
        chunk = Chunk(*(source[name] for name in (
            "id", "session_id", "start_message_id", "end_message_id", "salience_reason", "text",
        )))
        proofs = _claim_sources_for_chunk(conn, chunk)
        if not proofs or source["text"] != "\n".join(f"{proof.role}: {proof.content}" for proof in proofs):
            return None
        return source["text"], (tuple(source), tuple(proofs)), (source["vector_rowid"],)
    if index == 1:
        from hymem.dreaming.lossless import validate_message_coverage_artifact
        from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
        if row["source_coverage_version"] != LOSSLESS_COVERAGE_VERSION:
            return None
        proof = validate_message_coverage_artifact(
            conn, message_id=key, chunk_id=row["source_coverage_chunk_id"],
            coverage_version=row["source_coverage_version"],
        )
        frontier = conn.execute("SELECT coverage_message_id FROM sessions WHERE id=?", (proof.session_id,)).fetchone()
        if proof.role not in ("user", "assistant") or frontier is None or frontier[0] is None or key > frontier[0]:
            return None
        return proof.content, proof, (key,)
    if index == 2:
        # Fence cheap exact-text selection before the correlated live proof.
        # Do not cap candidates here: earlier same-text historical owners must
        # not crowd out a later live owner or hide the >64-live-owner guard.
        sources = conn.execute(
            "WITH candidates AS MATERIALIZED (SELECT * FROM knowledge_graph "
            "WHERE subject_canonical || ' ' || predicate || ' ' || object_canonical = ?) "
            f"SELECT * FROM candidates WHERE {live_edge_predicate('candidates')} "
            "ORDER BY id LIMIT 65", (key,),
        ).fetchall()
        if not sources or len(sources) > 64:
            return None
        return key, tuple(tuple(item) for item in sources), tuple(item["id"] for item in sources)
    if index == 3:
        from hymem.dreaming.aggregation_provenance import episode_input_proof
        proof = episode_input_proof(conn, key, with_session_prefix=False)
        source = conn.execute(
            "SELECT e.* FROM episodes e JOIN sessions s ON s.id=e.session_id WHERE e.id=? "
            "AND (e.digest_generation IS NULL OR e.digest_generation=s.digest_published_generation)", (key,),
        ).fetchone()
        if proof is None or source is None:
            return None
        rowid = db.episode_vector_rowid(key)
        # Match the ordinary publisher's collision rejection without retaining
        # an unbounded episode-id map. A collision anywhere in this row's
        # ownership set must not overwrite another episode's shadow.
        if any(other[0] != key and db.episode_vector_rowid(other[0]) == rowid
               for other in conn.execute("SELECT id FROM episodes")):
            return None
        return f"{source['title']}\n{source['summary']}", (tuple(source), proof.proof_hash), (rowid,)
    if index == 4:
        from hymem.dreaming.facts import load_fact_source_manifest
        proof = load_fact_source_manifest(conn, key)
        source = conn.execute("SELECT * FROM narrative_facts WHERE id=?", (key,)).fetchone()
        if not proof or source is None:
            return None
        return source["text"], (tuple(source), proof), (key,)
    raise ValueError("aggregation material is not embedding-only repairable")


def _safe_source(conn, index, row):
    try:
        return _source(conn, index, row)
    except (RuntimeError, ValueError, TypeError, KeyError):
        return None


def _cursor(conn, state_key):
    row = conn.execute("SELECT value FROM schema_meta WHERE key=?", (state_key,)).fetchone()
    if row is None:
        return 0, None, 0
    state = json.loads(row[0])
    if (type(state) is not list or len(state) != 3 or type(state[0]) is not int
            or not 0 <= state[0] < len(MIRROR_TABLES) or type(state[2]) is not int or state[2] < 0
            or (state[1] is not None and type(state[1]) is not int)):
        raise ValueError("invalid repair cursor")
    return tuple(state)


def _save_cursor(conn, key, index, after, blocked):
    conn.execute("INSERT INTO schema_meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, json.dumps([index, after, blocked], separators=(",", ":"))))


def _publish(conn, index, old, source, vector, model, dim):
    """Update one existing mirror and only its shadow ids, never provenance."""
    table, key = MIRROR_TABLES[index], _KEYS[index]
    fields = "vector_json=?,model=?,dim=?"
    values = [encode_vector(vector), model, dim]
    if index != 2:
        fields += ",text_hash=?"
        values.append(embedding_text_hash(source[0]))
    if index == 3:
        fields += ",embedding_producer_key=?"
        values.append(model)
    conn.execute(f"UPDATE {table} SET {fields} WHERE {key}=?", (*values, old[key]))
    if db.has_vec_table(conn, table=_SHADOWS[index]):
        for rowid in source[2]:
            conn.execute(f"DELETE FROM {_SHADOWS[index]} WHERE rowid=?", (rowid,))
            conn.execute(f"INSERT INTO {_SHADOWS[index]}(rowid,embedding) VALUES (?,?)", (rowid, db._pack_vector(vector)))


def repair(conn, embedder, *, apply=False, batch_size=16, max_items=256, timeout_seconds=60.0):
    """One bounded sweep segment. Apply rejects caller-owned transactions.

    Counts describe this invocation; sweep_blocked also includes earlier
    segments. Cursor advances past healthy/blocked rows and resets at EOF.
    A later invocation retries earlier failures in the next sweep. Provider
    errors keep their batch position, so a transient outage cannot skip work.
    """
    report = RepairReport("apply" if apply else "dry_run")
    if conn.in_transaction:
        report.status, report.error_code = "error", "caller_transaction"
        return report
    if (type(batch_size) is not int or not 1 <= batch_size <= 64
            or type(max_items) is not int or not 1 <= max_items <= 100000
            or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 3600):
        report.status, report.error_code = "error", "invalid_bounds"
        return report
    from hymem.dreaming.runner import (
        _DreamLeaseHeartbeat, _acquire_lock, _new_lease_token, _refresh_lock, _release_lock,
    )
    holder, heartbeat, fence = None, None, None
    deadline = MonotonicDeadline.after(timeout_seconds)
    try:
        with use_deadline(deadline):
            expected = _identity(embedder)
            _binding, model, dim = expected
            if db.schema_version(conn) != db.EXPECTED_SCHEMA_VERSION:
                report.status, report.error_code = "blocked", "current_schema_required"
                return report
            # Every mirror and FK is inspected without source text/provider
            # work. Corrupt/orphan stores fail closed before even taking a lease.
            conn.set_progress_handler(lambda: int(deadline.expired), 10000)
            health = scan_embedding_health(conn, live_model=model, live_dim=dim)
            report.health_before = asdict(health)
            deadline.check()
            if health.foreign_keys.status != "valid" or health.status == "unavailable":
                report.status, report.error_code = "blocked", "store_integrity"
                return report
            state_key = "embedding_repair_scan_v1:" + hashlib.sha256(f"{model}\0{dim}".encode()).hexdigest()
            if apply:
                holder = _new_lease_token()
                if not _acquire_lock(conn, holder):
                    holder = None
                    report.status, report.error_code = "blocked", "lease_busy"
                    return report
                fence = db.activate_transaction_lease_fence(conn, name="dreaming", holder=holder)
                heartbeat = _DreamLeaseHeartbeat(conn, holder, interval_seconds=30)
                heartbeat.start()
                index, after, sweep_blocked = _cursor(conn, state_key)
            else:
                index, after, sweep_blocked = 0, None, 0

            def guard():
                deadline.check()
                if heartbeat is not None:
                    heartbeat.check()
                    db._assert_transaction_lease_owned(conn)
                if _identity(embedder) != expected:
                    raise RuntimeError("producer changed")

            while report.scanned < max_items:
                guard()
                table, key = MIRROR_TABLES[index], _KEYS[index]
                limit = min(batch_size, max_items - report.scanned)
                rows = conn.execute(
                    f"SELECT rowid AS _repair_rowid,* FROM {table} " + ("WHERE rowid>? " if after is not None else "")
                    + "ORDER BY rowid LIMIT ?", ((after, limit) if after is not None else (limit,)),
                ).fetchall()
                if not rows:
                    index += 1
                    after = None
                    if index == len(MIRROR_TABLES):
                        report.sweep_complete = True
                        report.sweep_blocked = sweep_blocked
                        # An external writer or VACUUM/REPLACE can change an
                        # earlier row while a multi-run cursor is in flight.
                        # A cursor at EOF alone is never evidence of recovery.
                        after_health = scan_embedding_health(conn, live_model=model, live_dim=dim)
                        report.health_after = asdict(after_health)
                        guard()
                        if apply:
                            with db.transaction(conn):
                                guard()
                                _save_cursor(conn, state_key, 0, None, 0)
                                guard()
                        break
                    continue
                candidates = []
                batch_blocked = 0
                for row in rows:
                    deadline.check()
                    report.scanned += 1
                    compatible = _compatible(row, table, model, dim)
                    if index == 5:
                        from hymem.dreaming.aggregation_provenance import load_current_aggregation_node_proof
                        proof = load_current_aggregation_node_proof(conn, row[key], embedding_client=embedder) if compatible else None
                        if proof is None:
                            report.rebuild_required += 1
                            report.blocked += 1
                            batch_blocked += 1
                        else:
                            report.current += 1
                        continue
                    source = _safe_source(conn, index, row)
                    if source is None:
                        report.blocked += 1
                        batch_blocked += 1
                    elif compatible and (index == 2 or row["text_hash"] == embedding_text_hash(source[0])):
                        report.current += 1
                    else:
                        report.pending += 1
                        candidates.append((row, source))
                if apply:
                    guard()
                    # Recheck proof immediately before sending source text.
                    if any(_safe_source(conn, index, row) != source for row, source in candidates):
                        raise RuntimeError("source changed")
                    _refresh_lock(conn, holder)
                    if candidates:
                        report.provider_batches += 1
                    vectors = embedder.embed([source[0] for _, source in candidates]) if candidates else []
                    guard()
                    if len(vectors) != len(candidates) or any(not _valid_vector(encode_vector(vector), dim) for vector in vectors):
                        raise RuntimeError("invalid provider vectors")
                    count = 0
                    with db.transaction(conn), db.embedding_mutation(conn):
                        guard()
                        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
                            raise RuntimeError("store integrity changed")
                        for (row, source), vector in zip(candidates, vectors):
                            current = conn.execute(f"SELECT rowid AS _repair_rowid,* FROM {table} WHERE {key}=?", (row[key],)).fetchone()
                            if current is None or tuple(current) != tuple(row) or _safe_source(conn, index, row) != source:
                                raise RuntimeError("source or mirror changed")
                        if candidates:
                            db.ensure_vec_table(conn, dim, model=model)
                        for (row, source), vector in zip(candidates, vectors):
                            _publish(conn, index, row, source, vector, model, dim)
                            count += 1
                        # Keep only opaque local rowids, never edge/source text,
                        # in the local operational cursor. Updates preserve ids.
                        _save_cursor(conn, state_key, index, rows[-1]["_repair_rowid"], sweep_blocked + batch_blocked)
                        guard()
                    report.repaired += count
                    report.pending -= count
                sweep_blocked += batch_blocked
                report.sweep_blocked = sweep_blocked
                after = rows[-1]["_repair_rowid"]
            report.status = ("blocked" if report.sweep_blocked else
                             "complete" if report.sweep_complete and (apply or report.pending == 0)
                             and after_health.status == "compatible" and after_health.foreign_keys.status == "valid" else "incomplete")
    except DeadlineExceeded:
        report.status, report.error_code = "incomplete", "deadline"
    except db.LeaseOwnershipLost:
        report.status, report.error_code = "blocked", "lease_lost"
    except Exception:
        # No provider exception text, row id, source bytes, endpoint, or cursor
        # may leak through the operational report.
        report.status, report.error_code = (("incomplete", "deadline") if deadline.expired else ("error", "repair_failed"))
    finally:
        conn.set_progress_handler(None, 0)
        if heartbeat is not None:
            heartbeat.stop()
        if fence is not None:
            db.deactivate_transaction_lease_fence(fence)
        if holder is not None:
            _release_lock(conn, holder)
    return report


@contextmanager
def _open_existing(path, *, apply):
    # URI rw refuses creation even if a file vanishes between preflight/open.
    conn = sqlite3.connect(Path(path).resolve().as_uri() + ("?mode=rw" if apply else "?mode=ro"), uri=True, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        db.register_read_authority_functions(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=1000")
        if apply:
            conn.create_function("hymem_embedding_mutation_authorized", 0,
                lambda: int(db._connection_authority_key(conn) in db._EMBEDDING_MUTATION_KEYS.get()))
        else:
            conn.execute("PRAGMA query_only=ON")
        yield conn
    finally:
        conn.close()


def _configured_embedder(*, apply, allow_local):
    from hymem.bootstrap import resolve_env
    raw_dim = os.environ.get("HYMEM_EMBEDDING_DIM")
    if raw_dim is not None and (not raw_dim.isascii() or not raw_dim.isdecimal() or int(raw_dim) <= 0):
        raise ConfigurationError("invalid_embedding_dimension")
    cfg = resolve_env()
    if cfg.embedding_fallback_reason:
        raise ConfigurationError({
            "remote_embedding_endpoint_rejected": "embedding_endpoint_rejected",
            "remote_embedding_credentials_missing": "embedding_credentials_missing",
        }.get(cfg.embedding_fallback_reason, "embedding_fallback_refused"))
    if cfg.embedding_backend == "local_feature_hash":
        if apply and not allow_local:
            raise ConfigurationError("local_apply_requires_allow_local")
        from hymem.extraction.embeddings import LocalHashEmbeddingClient
        client = LocalHashEmbeddingClient(dim_value=cfg.embedding_dim, model_name=cfg.embedding_model)
    else:
        if not (cfg.has_embedding_key and cfg.embedding_pin_dimension and cfg.embedding_deployment_revision and cfg.embedding_deployment_tenant):
            raise ConfigurationError("exact_remote_producer_configuration_required")
        from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
        client = OpenAICompatibleEmbeddingClient(
            api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url, model=cfg.embedding_model,
            dim=cfg.embedding_dim, pin_dimension=True, deployment_revision=cfg.embedding_deployment_revision,
            deployment_tenant=cfg.embedding_deployment_tenant,
        )
    return cfg, client


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write repaired existing vectors (default: dry-run)")
    parser.add_argument("--allow-local", action="store_true", help="explicitly allow intentional local feature-hash apply")
    parser.add_argument("--db", type=Path, help="existing current-schema database (default: HYMEM_ROOT/hymem.sqlite)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=256, help="maximum repair-cursor rows, including healthy/blocked rows; excludes read-only integrity/proof audits")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    client = None
    report = RepairReport("apply" if args.apply else "dry_run")
    try:
        cfg, client = _configured_embedder(apply=args.apply, allow_local=args.allow_local)
        with _open_existing(args.db or cfg.root / "hymem.sqlite", apply=args.apply) as conn:
            report = repair(conn, client, apply=args.apply, batch_size=args.batch_size,
                            max_items=args.max_items, timeout_seconds=args.timeout_seconds)
    except ConfigurationError as exc:
        report.status, report.error_code = "error", exc.code
    except sqlite3.Error:
        report.status, report.error_code = "error", "existing_store_open_failed"
    except Exception:
        report.status, report.error_code = "error", "configuration_or_store_open"
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                report.status, report.error_code = "error", "client_cleanup"
    if args.json:
        print(json.dumps(asdict(report), sort_keys=True))
    else:
        print(f"{report.mode}: {report.status}; scanned={report.scanned} current={report.current} "
              f"pending={report.pending} repaired={report.repaired} blocked={report.blocked} "
              f"rebuild_required={report.rebuild_required} sweep_complete={report.sweep_complete} "
              f"error={report.error_code or 'none'}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
