#!/usr/bin/env python3
"""Private, bounded R7 extraction capture and same-output persistence replay.

Run with `python -I` and a private writable /work. Terminal JSON is metadata
only; the private JSON artifacts contain source text and must stay private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import stat
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

# `python -I` ignores PYTHONPATH; import only the explicitly mounted release.
sys.path.insert(0, "/candidate")

CHUNK_ID = "chk_12238afe485d5b2f4e7975b0d20b096ec20414d9"
EXPECTED_GENERATION = "hymem-phase1-generation-v1:6075085e12e32e1b790e49b99b8c3bb50718b18be0f28762d1582c58ee8e35eb"
SOURCE = Path("/reference/source.sqlite")
WORK = Path("/work")
CAPTURE = WORK / "claim-conflict-capture.json"
VECTORS = WORK / "claim-conflict-vectors.json"
MAX_COMPLETIONS = 16
MAX_HTTP = 48
USAGE = {"completion_calls": 0, "http_attempts": 0}
STAGE = "startup"
ENV_KEYS = frozenset({
    "HYMEM_LLM_API_KEY", "HYMEM_LLM_BASE_URL", "HYMEM_LLM_MODEL",
    "HYMEM_LLM_THINKING", "HYMEM_LLM_EXTRA_BODY",
    "HYMEM_LLM_DEPLOYMENT_REVISION", "HYMEM_LLM_DEPLOYMENT_TENANT",
    "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
    "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
    "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM",
    "HYMEM_EMBEDDING_PIN_DIMENSION",
    "HYMEM_EMBEDDING_DEPLOYMENT_REVISION",
    "HYMEM_EMBEDDING_DEPLOYMENT_TENANT",
    "HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP",
    "HYMEM_EMBEDDING_TIMEOUT_SECONDS",
    "HYMEM_AGGREGATION_NODES_ENABLED", "HYMEM_AGGREGATION_DIGEST_ENABLED",
})


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_digest(conn: sqlite3.Connection) -> str:
    """Hash SQLite's logical dump, excluding journal/header byte variation."""
    digest = hashlib.sha256()
    for statement in conn.iterdump():
        digest.update(statement.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _private_write(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.chmod(path, 0o600)


def _load_runtime_env(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("runtime_environment_not_private_regular_file")
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in values.items()
    ):
        raise ValueError("runtime environment must be a JSON string map")
    if set(values) - ENV_KEYS:
        raise ValueError("runtime environment contains unapproved key names")
    for key in tuple(os.environ):
        if key.startswith(("HYMEM_", "DEEPSEEK_", "OPENAI_")):
            del os.environ[key]
    for key, value in values.items():
        os.environ[key] = value
    os.environ["HYMEM_ROOT"] = str(WORK)


def _clone(name: str):
    from hymem.core.db import connect, _load_vec_extension

    target = WORK / name
    if target.exists():
        raise FileExistsError(target)
    source = sqlite3.connect(f"file:{SOURCE}?mode=ro&immutable=1", uri=True)
    try:
        dest = sqlite3.connect(target)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    os.chmod(target, 0o600)
    conn = connect(target)
    try:
        has_vec = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%USING vec0%'"
        ).fetchone()
        if has_vec and not _load_vec_extension(conn):
            raise RuntimeError("snapshot_vector_extension_unavailable")
    except BaseException:
        conn.close()
        raise
    return conn, target


def _chunk(conn):
    from hymem.dreaming.chunks import Chunk

    row = conn.execute(
        "SELECT id,session_id,start_message_id,end_message_id,salience_reason,text,"
        "source_manifest_version,source_manifest_count "
        "FROM chunks WHERE id=? AND chunk_kind='extraction'", (CHUNK_ID,)
    ).fetchone()
    if row is None:
        raise ValueError("target extraction chunk absent")
    mids = tuple(int(r[0]) for r in conn.execute(
        "SELECT source_message_id FROM chunk_message_sources "
        "WHERE chunk_id=? ORDER BY ordinal", (CHUNK_ID,)
    ))
    if mids != (902, 903) or len(row["text"]) != 4606:
        raise ValueError("target source manifest or chunk size changed")
    if (row["source_manifest_version"] != "claim-source-manifest-v1"
            or row["source_manifest_count"] != 2):
        raise ValueError("target source coverage version changed")
    return Chunk(row["id"], row["session_id"], int(row["start_message_id"]),
                 int(row["end_message_id"]), row["salience_reason"], row["text"], mids)


def _current_publication_count(conn) -> int:
    # A pending chunk may have a valid publication from an older producer.
    return conn.execute(
        "SELECT COUNT(*) FROM current_phase1_publications "
        "WHERE chunk_id=? AND phase1_generation_key=?",
        (CHUNK_ID, EXPECTED_GENERATION),
    ).fetchone()[0]


def _offline_preflight() -> dict:
    # Publication views require the same connection-local authority functions
    # as production. A plain sqlite3 connection cannot execute those views.
    conn, _ = _clone("preflight.sqlite")
    try:
        chunk = _chunk(conn)
        publications = _current_publication_count(conn)
        cited = conn.execute(
            "SELECT COUNT(*) FROM kg_claim_observations WHERE source_message_id IN (902,903) "
            "AND phase1_generation_key=?", (EXPECTED_GENERATION,),
        ).fetchone()[0]
        registered = conn.execute(
            "SELECT COUNT(*) FROM phase1_generations WHERE generation_key=?",
            (EXPECTED_GENERATION,),
        ).fetchone()[0]
        if publications or cited or registered != 1:
            raise ValueError("target already has published/current-generation observations")
        _load_runtime_env(Path("/run/runtime-env.json"))
        client = _client()
        try:
            from hymem.extraction.producer import phase1_generation_binding
            if phase1_generation_binding(_cfg().prompt_version, client)["generation_key"] != EXPECTED_GENERATION:
                raise ValueError("preflight_generation_mismatch")
        finally:
            client.close()
        from hymem.dreaming.phase1 import _claim_sources_for_chunk
        if len(_claim_sources_for_chunk(conn, chunk)) != 2:
            raise ValueError("preflight_coverage_mismatch")
        return {"status": "ready", "source_sha256": _hash(SOURCE),
                "target_chunk_id": CHUNK_ID, "source_message_ids": list(chunk.source_message_ids),
                "published_count": publications, "same_generation_observation_count": cited,
                "expected_generation_registry_count": registered,
                "runtime_generation_verified": True, "source_coverage_verified": True,
                "completion_calls": 0, "http_attempts": 0,
                "capture_exists": CAPTURE.exists()}
    finally:
        conn.close()


def _cfg():
    from hymem.config import HyMemConfig
    return HyMemConfig(root=WORK)


def _client():
    from hymem.bootstrap import resolve_env
    from hymem.contrib.openai_client import OpenAICompatibleClient
    env = resolve_env()
    if (not env.llm_api_key or env.llm_base_url.rstrip("/") != "https://api.deepseek.com"
            or env.llm_model != "deepseek-flash"):
        raise ValueError("unapproved_runtime_llm_identity")
    return OpenAICompatibleClient(
        api_key=env.llm_api_key, base_url=env.llm_base_url, model=env.llm_model
    )


def _encode_capture(chunk, extraction) -> bytes:
    return json.dumps({
        "schema": "claim-conflict-capture-v1",
        "chunk": asdict(chunk),
        "extraction": asdict(extraction),
        "source_sha256": _hash(SOURCE),
        "generation_key": EXPECTED_GENERATION,
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
       allow_nan=False).encode("utf-8")


def _decode_capture():
    from hymem.dreaming.chunks import Chunk
    from hymem.dreaming.lossless import CoveredMessage
    from hymem.dreaming.phase1 import ChunkExtraction
    from hymem.extraction.markers import Marker
    from hymem.extraction.triples import Triple

    payload = json.loads(CAPTURE.read_text())
    if payload["schema"] != "claim-conflict-capture-v1":
        raise ValueError("capture schema changed")
    raw = payload["extraction"]
    raw["triples"] = [Triple(**item) for item in raw["triples"]]
    raw["markers"] = [Marker(**item) for item in raw["markers"]]
    raw["failure_details"] = tuple(raw["failure_details"])
    raw["claim_sources"] = {
        int(mid): CoveredMessage(**item)
        for mid, item in raw["claim_sources"].items()
    }
    payload["chunk"]["source_message_ids"] = tuple(
        payload["chunk"]["source_message_ids"]
    )
    return payload, Chunk(**payload["chunk"]), ChunkExtraction(**raw)


def _embedding():
    from hymem.bootstrap import resolve_env
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient

    env = resolve_env()
    if env.embedding_fallback_reason:
        raise ValueError("configured embedding authority unavailable")
    if env.embedding_dim != 384:
        raise ValueError("embedding dimension differs from frozen 384-space")
    if env.embedding_backend != "openai_compatible":
        raise ValueError("actual pinned embedding producer unavailable")
    if not (env.embedding_pin_dimension and env.embedding_deployment_revision
            and env.embedding_deployment_tenant):
        raise ValueError("embedding producer lacks exact pinned authority")
    return OpenAICompatibleEmbeddingClient(
        api_key=env.embedding_api_key, base_url=env.embedding_base_url,
        model=env.embedding_model, dim=env.embedding_dim,
        pin_dimension=env.embedding_pin_dimension,
        deployment_revision=env.embedding_deployment_revision,
        deployment_tenant=env.embedding_deployment_tenant,
    )


def capture() -> dict:
    global STAGE
    from hymem.bootstrap import resolve_env
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.deadline import MonotonicDeadline, use_deadline
    from hymem.dreaming import phase1
    from hymem.dreaming.runner import _CountingPhase1LLM
    from hymem.extraction.producer import phase1_generation_binding

    conn, path = _clone("capture.sqlite")
    client = None
    embedding = None
    llm = None
    original_extract = phase1.extract_chunk
    try:
        chunk = _chunk(conn)
        cfg = _cfg()
        STAGE = "client_identity"
        client = _client()
        llm = _CountingPhase1LLM(client)
        generation = phase1_generation_binding(cfg.prompt_version, llm)
        if generation["generation_key"] != EXPECTED_GENERATION:
            raise ValueError("producer generation does not match frozen expected key")

        def limited_extract(actual_client, text, *, source_records=None):
            return original_extract(actual_client, text,
                                    source_records=source_records,
                                    completion_call_limit=MAX_COMPLETIONS)

        phase1.extract_chunk = limited_extract
        try:
            STAGE = "paid_extraction"
            with use_deadline(MonotonicDeadline.after(600)):
                extraction = phase1.extract_chunk_results(
                    conn, chunk, llm, prompt_version=cfg.prompt_version,
                    phase1_generation=generation,
                )
        finally:
            phase1.extract_chunk = original_extract
        if llm.completion_calls > MAX_COMPLETIONS or llm.provider_attempts > MAX_HTTP:
            raise RuntimeError("paid request budget exceeded")
        if extraction is None:
            raise ValueError("target chunk already published")
        STAGE = "save_extraction"
        _private_write(CAPTURE, _encode_capture(chunk, extraction))
        if extraction.failed:
            return {"status": "extraction_failed", "failure_reason": extraction.failure_reason,
                    "completion_calls": llm.completion_calls,
                    "http_attempts": llm.provider_attempts}
        STAGE = "prepare_embeddings"
        embedding = _embedding()
        dedup = phase1.prepare_dedup_vectors(conn, extraction, cfg, embedding)
        if extraction.triples and not dedup:
            raise ValueError("dedup vector preparation produced no candidates")
        _private_write(VECTORS, json.dumps({
            "schema": "claim-conflict-vectors-v1",
            "model": getattr(dedup, "model", None),
            "dim": getattr(dedup, "dim", None),
            "vectors": dict(dedup),
            "source_sha256": _hash(SOURCE),
            "capture_sha256": _hash(CAPTURE),
            "generation_key": EXPECTED_GENERATION,
        }, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
           allow_nan=False).encode("utf-8"))
        return {"status": "captured", "triples": len(extraction.triples),
                "dedup_vectors": len(dedup), "completion_calls": llm.completion_calls,
                "http_attempts": llm.provider_attempts,
                "generation_key": generation["generation_key"],
                "capture_path": str(CAPTURE), "source_sha256": _hash(SOURCE)}
    finally:
        phase1.extract_chunk = original_extract
        if llm is not None:
            USAGE["completion_calls"] = llm.completion_calls
            USAGE["http_attempts"] = llm.provider_attempts
        conn.close()
        try:
            if embedding is not None:
                embedding.close()
        finally:
            if client is not None:
                client.close()


def replay(*, dedup_enabled: bool, name: str) -> dict:
    from hymem.dreaming import evidence, phase1
    from hymem.dreaming.evidence import prompt_generation
    from hymem.dreaming.phase1 import _PreparedDedupVectors

    payload, saved_chunk, extraction = _decode_capture()
    if extraction.failed or not VECTORS.is_file():
        raise ValueError("capture has no successful extraction with dedup vectors")
    vector_payload = json.loads(VECTORS.read_text())
    if vector_payload["schema"] != "claim-conflict-vectors-v1":
        raise ValueError("dedup vector schema changed")
    if vector_payload["source_sha256"] != payload["source_sha256"]:
        raise ValueError("dedup vectors bind another source")
    if (vector_payload["capture_sha256"] != _hash(CAPTURE)
            or vector_payload["generation_key"] != EXPECTED_GENERATION):
        raise ValueError("dedup vectors bind another extraction")
    dedup = _PreparedDedupVectors(model=vector_payload["model"],
                                   dim=vector_payload["dim"])
    dedup.update(vector_payload["vectors"])
    if payload["source_sha256"] != _hash(SOURCE):
        raise ValueError("source snapshot changed since capture")
    if payload["generation_key"] != EXPECTED_GENERATION:
        raise ValueError("captured producer generation changed")
    conn, path = _clone(name)
    before = _logical_digest(conn)
    old_record = evidence.record_claim_observation
    observations = []
    def record(*args, **kwargs):
        evidence_id = kwargs["evidence_id"]
        new = conn.execute("SELECT interpretation_key,value_text,value_numeric,"
                           "value_unit,temporal_scope FROM kg_evidence WHERE id=?",
                           (evidence_id,)).fetchone()
        old = conn.execute(
            "SELECT o.chunk_id,o.polarity,o.interpretation_key,"
            "e.value_text,e.value_numeric,e.value_unit,e.temporal_scope "
            "FROM kg_claim_observations o JOIN kg_evidence e ON e.id=o.evidence_id "
            "WHERE o.edge_id=? AND o.source_session_id=? AND o.source_message_id=? "
            "AND o.evidence_kind='extraction' AND o.prompt_generation=? "
            "AND o.phase1_generation_key IS ?",  # detect collision in same generation
            (kwargs["edge_id"], kwargs["source_session_id"],
             kwargs["source_message_id"], prompt_generation(kwargs["prompt_version"]),
             kwargs["phase1_generation_key"]),
        ).fetchall()
        for row in old:
            changed = []
            if row["polarity"] != kwargs["polarity"]:
                changed.append("polarity")
            if new is not None:
                for field in ("interpretation_key", "value_text", "value_numeric",
                              "value_unit", "temporal_scope"):
                    if row[field] != new[field]:
                        changed.append(field)
            observations.append({"edge_id": kwargs["edge_id"],
                                 "source_message_id": kwargs["source_message_id"],
                                 "old_chunk_id": row["chunk_id"],
                                 "new_chunk_id": kwargs["chunk_id"],
                                 "changed_fields": changed})
        return old_record(*args, **kwargs)
    try:
        chunk = _chunk(conn)
        if chunk != saved_chunk:
            raise ValueError("target chunk changed since capture")
        cfg = replace(_cfg(), triple_dedup_enabled=dedup_enabled)
        evidence.record_claim_observation = record
        try:
            conn.execute("BEGIN IMMEDIATE")
            phase1.persist_chunk_results(
                conn, chunk, extraction, prompt_version=cfg.prompt_version,
                cfg=cfg, dedup_vectors=(dedup if dedup_enabled else {}),
                in_cycle_edges=phase1.new_in_cycle_pool(),
            )
            conn.execute("COMMIT")
            status = "persisted"
            error_type = None
        except Exception as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            status = "rejected"
            error_type = type(exc).__name__
            if str(exc) != "same-generation claim observations disagree":
                raise
        finally:
            evidence.record_claim_observation = old_record
        after = _logical_digest(conn)
        if status == "rejected" and after != before:
            raise RuntimeError("rejected replay changed logical SQLite state")
        return {"dedup_enabled": dedup_enabled, "status": status,
                "error_type": error_type, "observation_collisions": observations,
                "clone_path": str(path), "source_sha256": _hash(SOURCE),
                "logical_digest_before": before, "logical_digest_after": after}
    finally:
        evidence.record_claim_observation = old_record
        conn.close()


def main() -> None:
    global STAGE
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("offline", "live", "replay"))
    parser.add_argument("--env", type=Path, default=Path("/run/runtime-env.json"))
    args = parser.parse_args()
    STAGE = args.mode
    WORK.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(WORK, 0o700)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_before = _hash(SOURCE)
    if args.mode == "offline":
        result = _offline_preflight()
    elif args.mode == "live":
        if CAPTURE.exists() or VECTORS.exists():
            raise FileExistsError("private paid capture already exists")
        _load_runtime_env(args.env)
        result = capture()
    else:
        if not CAPTURE.is_file():
            raise FileNotFoundError(CAPTURE)
        result = {"dedup_on": replay(dedup_enabled=True, name="dedup-on.sqlite"),
                  "dedup_off": replay(dedup_enabled=False, name="dedup-off.sqlite")}
    if _hash(SOURCE) != source_before:
        raise RuntimeError("source snapshot changed")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Never print provider exceptions, endpoint paths, credentials, text,
        # or traceback frames from the private reproduction.
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "stage": STAGE, **USAGE}, sort_keys=True))
        sys.exit(1)
