"""Read-only, source-aware checks for a HyMem daily health report.

Run on the host that owns the v64 database:

    python tools/diagnostics/hymem_daily_triage.py --db /path/to/hymem.sqlite

The output contains counts and hashed session identifiers, never memory text.
It does not call providers, migrate the database, or change retry state.
Read the production dream logs for fact failure reasons and compare the stored
vector identity with the live embedding producer before declaring health.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hymem.core import db
from hymem.core.graph import live_edge_predicate
from hymem.core.time import EVENT_CLOCK_SKEW_SECONDS
from hymem.core.vectors import decode_vector
from hymem.dreaming.aggregation_provenance import episode_input_proof
from hymem.dreaming.facts import fact_diagnostic_session_key
from hymem.extraction.embeddings import embedding_text_hash


def _count(conn, sql: str, params: tuple = ()) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def _session_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def collect(path: Path, *, recent_runs: int = 10) -> dict:
    """Collect one coherent SQLite snapshot without requiring sqlite-vec."""
    with db.read_snapshot(path) as conn:
        # Ordinary tables remain readable when the examining host has no
        # sqlite-vec extension. This PRAGMA only affects this query-only handle.
        conn.execute("PRAGMA writable_schema = ON")
        schema = int(conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0])
        if schema != db.EXPECTED_SCHEMA_VERSION:
            raise ValueError(f"schema_{schema}_expected_{db.EXPECTED_SCHEMA_VERSION}")

        runs = [
            {
                "id": int(row["id"]),
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "top_level_error": row["error"] is not None,
                "fact_failures": int(row["fact_failures"]),
                "chunks_seen": int(row["chunks_seen"]),
                "chunks_processed": int(row["chunks_processed"]),
            }
            for row in conn.execute(
                "SELECT id,started_at,ended_at,error,fact_failures,"
                "chunks_seen,chunks_processed FROM dream_runs "
                "ORDER BY id DESC LIMIT ?", (recent_runs,)
            )
        ]
        fact_retries = [
            {
                "fact_log_session_key": fact_diagnostic_session_key(str(row["id"])),
                "retry_count": int(row["facts_retry_count"] or 0),
                "quarantined": bool(row["facts_quarantined"]),
                "cursor_prompt_version": row["facts_cursor_prompt_version"],
                "retry_key_sha256": (
                    hashlib.sha256(str(row["facts_retry_config_version"]).encode()).hexdigest()
                    if row["facts_retry_config_version"] is not None else None
                ),
            }
            for row in conn.execute(
                "SELECT id,facts_retry_count,facts_quarantined,"
                "facts_cursor_prompt_version,facts_retry_config_version "
                "FROM sessions WHERE facts_retry_count>0 OR facts_quarantined=1 "
                "ORDER BY id"
            )
        ]

        extraction_chunks = _count(
            conn, "SELECT COUNT(*) FROM chunks WHERE chunk_kind='extraction'"
        )
        terminal = _count(conn, "SELECT COUNT(*) FROM chunk_extraction_terminal_losses")
        terminal_with_graph_evidence = _count(
            conn, "SELECT COUNT(DISTINCT loss.chunk_id) "
            "FROM chunk_extraction_terminal_losses loss "
            "JOIN kg_evidence evidence ON evidence.chunk_id=loss.chunk_id"
        )
        canonical_unpublished_extraction = _count(
            conn, "SELECT COUNT(*) FROM kg_evidence "
            "WHERE evidence_kind='extraction' AND provenance_status='canonical' "
            "AND is_current=1 AND published_at IS NULL"
        )
        canonical_unpublished_extraction_chunks = _count(
            conn, "SELECT COUNT(DISTINCT chunk_id) FROM kg_evidence "
            "WHERE evidence_kind='extraction' AND provenance_status='canonical' "
            "AND is_current=1 AND published_at IS NULL"
        )

        # Migration v38 deliberately stamped pre-cursor episode publications
        # `legacy`. The current dream status correctly treats that wire shape
        # as unverified; identify this exact historical subtype without changing
        # its health gate or accepting a malformed modern cursor as legacy.
        legacy_digest_predicate = (
            "s.digest_published_generation='legacy' "
            "AND s.coverage_message_id IS NULL "
            "AND s.digest_cursor_message_id IS NULL "
            "AND s.digest_cursor_partial_message_id IS NULL "
            "AND s.digest_cursor_offset=0 "
            "AND s.digest_cursor_prompt_version IS NULL "
            "AND EXISTS (SELECT 1 FROM episodes e WHERE e.session_id=s.id "
            "AND e.digest_generation='legacy' AND e.source_manifest_complete=0) "
            "AND NOT EXISTS (SELECT 1 FROM episodes e WHERE e.session_id=s.id "
            "AND (e.digest_generation IS NOT 'legacy' "
            "OR e.source_manifest_complete<>0))"
        )
        legacy_unverified_sessions = _count(
            conn, "SELECT COUNT(*) FROM sessions s WHERE " + legacy_digest_predicate
        )
        legacy_unverified_episodes = _count(
            conn,
            "SELECT COUNT(*) FROM episodes episode JOIN sessions s "
            "ON s.id=episode.session_id WHERE " + legacy_digest_predicate,
        )
        legacy_other_state = _count(
            conn, "SELECT COUNT(*) FROM sessions s WHERE "
            "s.digest_published_generation='legacy' AND NOT ("
            + legacy_digest_predicate + ")"
        )

        episodes = conn.execute(
            "SELECT e.id,e.title,e.summary,e.source_manifest_complete,"
            "ee.episode_id,ee.model,ee.dim,ee.embedding_producer_key,"
            "ee.text_hash,ee.vector_json "
            "FROM episodes e JOIN sessions s ON s.id=e.session_id "
            "LEFT JOIN episode_embeddings ee ON ee.episode_id=e.id "
            "WHERE e.digest_generation IS NULL "
            "OR e.digest_generation=s.digest_published_generation"
        ).fetchall()
        meta = dict(conn.execute(
            "SELECT key,value FROM schema_meta WHERE key IN ('vec_model','vec_dim')"
        ).fetchall())
        model = meta.get("vec_model")
        dim = int(meta["vec_dim"]) if meta.get("vec_dim") is not None else None
        episode_counts = {
            "active": len(episodes),
            "source_proof_valid": 0,
            "source_manifest_unpublished": 0,
            "published_source_proof_invalid": 0,
            "proof_validation_errors": 0,
            "source_proof_valid_without_stored_identity_embedding": 0,
            "stored_embedding_rows": 0,
            "stored_embedding_on_unproved_episode": 0,
        }
        expected_episode_vec = {}
        for row in episodes:
            manifest_published = bool(row["source_manifest_complete"])
            validation_error = False
            try:
                proof_valid = manifest_published and (
                    episode_input_proof(
                        conn, str(row["id"]), with_session_prefix=False
                    ) is not None
                )
            except (sqlite3.Error, ValueError, TypeError, RuntimeError):
                proof_valid = False
                validation_error = True
            has_embedding = row["episode_id"] is not None
            current_embedding = (
                has_embedding and model is not None and dim is not None
                and row["model"] == model and row["dim"] == dim
                and row["embedding_producer_key"] == model
                and row["text_hash"] == embedding_text_hash(
                    f"{row['title']}\n{row['summary']}"
                )
            )
            vector = None
            if current_embedding:
                try:
                    vector = db._finite_vec(
                        decode_vector(row["vector_json"]), dim
                    )
                    current_embedding = vector is not None
                except (TypeError, ValueError, OverflowError, UnicodeError):
                    current_embedding = False
            if proof_valid and current_embedding:
                rowid = db.episode_vector_rowid(str(row["id"]))
                if rowid in expected_episode_vec:
                    raise ValueError("episode_vector_rowid_collision")
                expected_episode_vec[rowid] = db._pack_vector(vector)
            if not manifest_published:
                episode_counts["source_manifest_unpublished"] += 1
            elif proof_valid:
                episode_counts["source_proof_valid"] += 1
            else:
                episode_counts["published_source_proof_invalid"] += 1
            episode_counts["proof_validation_errors"] += int(validation_error)
            episode_counts["stored_embedding_rows"] += int(has_embedding)
            episode_counts["stored_embedding_on_unproved_episode"] += int(
                has_embedding and not proof_valid
            )
            episode_counts["source_proof_valid_without_stored_identity_embedding"] += int(
                proof_valid and not current_embedding
            )

        edge_rows = conn.execute(
            "SELECT id,subject_canonical,predicate,object_canonical "
            f"FROM knowledge_graph WHERE {live_edge_predicate()}"
        ).fetchall()
        # Reuse the same status, majority, valid-time, and direct-edge gates as
        # live_edge_predicate. Subtract its full authority result to isolate
        # only direct edges rejected by the graph's evidence authority gate.
        graph_pre_authority = (
            "kg.status='active' AND kg.invalid_at IS NULL "
            "AND kg.pos_evidence>kg.neg_evidence AND kg.derived=0 "
            "AND (kg.valid_at IS NULL OR hymem_timestamp_at_or_before("
            "kg.valid_at,strftime('%Y-%m-%dT%H:%M:%fZ','now',"
            f"'+{EVENT_CLOCK_SKEW_SECONDS} seconds'))=1)"
        )
        authority_excluded_edges = _count(
            conn, "SELECT COUNT(*) FROM knowledge_graph kg WHERE "
            + graph_pre_authority + " AND NOT (" + live_edge_predicate("kg") + ")"
        )
        unpublished_on_eligible_edges = conn.execute(
            "SELECT COUNT(*) AS evidence_rows,COUNT(DISTINCT ev.chunk_id) "
            "AS source_chunks FROM kg_evidence ev "
            "JOIN knowledge_graph kg ON kg.id=ev.edge_id "
            "WHERE ev.provenance_status='canonical' AND ev.is_current=1 "
            "AND ev.published_at IS NULL AND " + graph_pre_authority
        ).fetchone()
        edge_texts = {
            int(row["id"]): (
                f"{row['subject_canonical']} {row['predicate']} "
                f"{row['object_canonical']}"
            ) for row in edge_rows
        }
        current_cache = {}
        if model is not None and dim is not None:
            for row in conn.execute(
                "SELECT edge_text,vector_json FROM edge_embeddings "
                "WHERE model=? AND dim=?", (model, dim)
            ):
                try:
                    vector = db._finite_vec(decode_vector(row["vector_json"]), dim)
                except (TypeError, ValueError, OverflowError, UnicodeError):
                    vector = None
                if vector is not None:
                    current_cache[str(row["edge_text"])] = db._pack_vector(vector)
        expected_vec = {
            edge_id: current_cache[triple]
            for edge_id, triple in edge_texts.items() if triple in current_cache
        }
        vec_available = db._load_vec_extension(conn)
        vector_shadow = {
            "checked": False,
            "reason": "vec_edges_missing" if vec_available else "sqlite_vec_unavailable",
        }
        if vec_available and db.has_vec_table(conn, table="vec_edges"):
            actual_vec = {
                int(row[0]): bytes(row[1])
                for row in conn.execute("SELECT rowid,embedding FROM vec_edges")
            }
            vector_shadow = {
                "checked": True,
                "missing_live_cached_ids": len(expected_vec.keys() - actual_vec.keys()),
                "extra_ids": len(actual_vec.keys() - expected_vec.keys()),
                "different_vectors": sum(
                    actual_vec[key] != expected_vec[key]
                    for key in expected_vec.keys() & actual_vec.keys()
                ),
            }
        episode_shadow = {
            "checked": False,
            "reason": "vec_episodes_missing" if vec_available else "sqlite_vec_unavailable",
        }
        if vec_available and db.has_vec_table(conn, table="vec_episodes"):
            actual_episode_vec = {
                int(row[0]): bytes(row[1])
                for row in conn.execute("SELECT rowid,embedding FROM vec_episodes")
            }
            episode_shadow = {
                "checked": True,
                "missing_ids": len(expected_episode_vec.keys() - actual_episode_vec.keys()),
                "extra_ids": len(actual_episode_vec.keys() - expected_episode_vec.keys()),
                "different_vectors": sum(
                    actual_episode_vec[key] != expected_episode_vec[key]
                    for key in expected_episode_vec.keys() & actual_episode_vec.keys()
                ),
            }

        run_date_candidates = [
            {
                "procedure_id_sha256": hashlib.sha256(
                    str(row["id"]).encode("utf-8")
                ).hexdigest(),
                "name_sha256": hashlib.sha256(
                    str(row["name"]).encode("utf-8")
                ).hexdigest(),
                "session_sha256": _session_key(str(row["session_id"])),
                "status": row["status"],
                "description_sha256": hashlib.sha256(
                    str(row["description"] or "").encode("utf-8")
                ).hexdigest(),
                "steps_sha256": hashlib.sha256(
                    str(row["steps"] or "").encode("utf-8")
                ).hexdigest(),
                "unretired_publication_rows": int(row["unretired_publication_rows"]),
            }
            for row in conn.execute(
                "SELECT p.id,p.name,p.session_id,p.status,p.description,p.steps,"
                "COUNT(d.procedure_id) AS unretired_publication_rows "
                "FROM procedures p LEFT JOIN procedure_digest_publications d "
                "ON d.procedure_id=p.id AND d.retired=0 "
                "WHERE lower(p.name) LIKE '%run_date%' "
                "GROUP BY p.id ORDER BY p.id"
            )
        ]

        return {
            "schema_version": schema,
            "recent_runs": runs,
            "fact_retry_rows": fact_retries,
            "extraction_chunks": extraction_chunks,
            "terminal_source_loss_chunks": terminal,
            "terminal_with_existing_graph_evidence": terminal_with_graph_evidence,
            "current_canonical_extraction_evidence_without_published_at": (
                canonical_unpublished_extraction
            ),
            "source_chunks_with_unpublished_canonical_extraction_evidence": (
                canonical_unpublished_extraction_chunks
            ),
            "current_canonical_evidence_without_published_at_on_eligible_direct_edges": (
                int(unpublished_on_eligible_edges["evidence_rows"])
            ),
            "source_chunks_with_such_evidence": (
                int(unpublished_on_eligible_edges["source_chunks"])
            ),
            "active_direct_edges_excluded_by_authority": authority_excluded_edges,
            "legacy_unverified_digest_sessions": legacy_unverified_sessions,
            "legacy_unverified_episode_rows": legacy_unverified_episodes,
            "legacy_digest_stamp_other_state_sessions": legacy_other_state,
            "episodes": episode_counts,
            "stored_vec_identity": {
                "model": model,
                "dim": dim,
                "live_producer_attested": False,
            },
            "vec_episodes": episode_shadow,
            "edge_embeddings_cache_rows": _count(conn, "SELECT COUNT(*) FROM edge_embeddings"),
            "live_direct_edges": len(edge_texts),
            "live_direct_edges_with_valid_stored_identity_cache": len(expected_vec),
            "vec_edges": vector_shadow,
            "run_date_candidate_scope": "name_contains_run_date",
            "run_date_procedure_candidates": run_date_candidates,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100:
        parser.error("--runs must be between 1 and 100")
    print(json.dumps(collect(args.db, recent_runs=args.runs), sort_keys=True))


if __name__ == "__main__":
    main()
