#!/usr/bin/env python3
"""Bounded embedding-only recovery check on a private clone.

The mounted runtime JSON is read for HYMEM_EMBEDDING_* keys only. No LLM is
constructed. Terminal output is metadata; source text and errors remain 0600.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import traceback
import types

sys.path.insert(0, "/candidate")

SOURCE = Path("/reference/source.sqlite")
WORK = Path("/work")
MAX_TEXTS_PER_CALL = 16
MAX_CHARS_PER_CALL = 128_000
MAX_UTF8_BYTES_PER_CALL = 512_000
MAX_EMBEDDING_HTTP = 128
MAX_SECONDS = 900
EMBEDDING_BASE_URL = "http://embedding-server:8766/v1"
EMBEDDING_BASE_URL_SHA256 = "081b277dc2e1bc5cb0aa00b7d5c4688377d5a36a53840535b3e6e1e1369676ae"
STAGE = "startup"


class BudgetStop(BaseException):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_json(path: Path, value: object) -> None:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True,
                     allow_nan=False, separators=(",", ":")).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def load_embedding_env(path: Path, root: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("runtime_environment_not_private_regular_file")
    values = json.loads(path.read_bytes())
    if not isinstance(values, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in values.items()
    ):
        raise ValueError("runtime_environment_invalid")
    selected = {key: value for key, value in values.items()
                if key.startswith("HYMEM_EMBEDDING_")}
    for key in tuple(os.environ):
        if key.startswith(("HYMEM_", "OPENAI_", "DEEPSEEK_")):
            del os.environ[key]
    os.environ.update(selected)
    os.environ["HYMEM_ROOT"] = str(root)
    if not selected.get("HYMEM_EMBEDDING_API_KEY"):
        raise ValueError("embedding_credentials_missing")


def clone_source(target: Path) -> None:
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    source = sqlite3.connect(f"file:{SOURCE}?mode=ro&immutable=1", uri=True)
    try:
        dest = sqlite3.connect(target)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()


def open_clone(path: Path):
    from hymem.core.db import connect, _load_vec_extension
    conn = connect(path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%USING vec0%'"
        ).fetchone()
        if exists and not _load_vec_extension(conn):
            raise RuntimeError("snapshot_vector_extension_unavailable")
        return conn
    except BaseException:
        conn.close()
        raise


def make_embedder():
    from hymem.bootstrap import resolve_env
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
    from hymem.extraction.embeddings import CachedEmbeddingClient

    cfg = resolve_env()
    if (cfg.embedding_base_url != EMBEDDING_BASE_URL
            or hashlib.sha256(cfg.embedding_base_url.encode()).hexdigest()
            != EMBEDDING_BASE_URL_SHA256):
        raise ValueError("embedding_endpoint_identity_invalid")
    if (cfg.embedding_fallback_reason is not None
            or cfg.embedding_backend != "openai_compatible"
            or cfg.embedding_dim != 384
            or not cfg.embedding_pin_dimension
            or not cfg.embedding_deployment_revision
            or not cfg.embedding_deployment_tenant
            or not cfg.embedding_api_key):
        raise ValueError("embedding_runtime_identity_invalid")
    transport = OpenAICompatibleEmbeddingClient(
        api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url,
        model=cfg.embedding_model, dim=cfg.embedding_dim,
        pin_dimension=cfg.embedding_pin_dimension,
        deployment_revision=cfg.embedding_deployment_revision,
        deployment_tenant=cfg.embedding_deployment_tenant,
    )
    return CachedEmbeddingClient(transport)


def census(conn, embedder) -> dict:
    """Exact pending/cache census with read-only SQL and no embed() call."""
    from hymem.core.vectors import decode_vector
    from hymem.dreaming.embeddings import (
        _embedding_identity, _fetch_cached_vectors, _finite_embedding_vector,
        chunk_embedding_id_batches,
    )
    from hymem.extraction.embeddings import embedding_text_hash

    model, dim = _embedding_identity(embedder)
    if dim != 384:
        raise ValueError("embedding_space_changed")
    scanned_ids: list[str] = []
    pending_ids: list[str] = []
    batches = 0
    pending = cached = misses = miss_chars = miss_bytes = planned_calls = uncallable = 0
    for ids in chunk_embedding_id_batches(conn):
        batches += 1
        if len(ids) > MAX_TEXTS_PER_CALL:
            raise ValueError("planned_batch_row_limit_exceeded")
        rows = conn.execute(
            "SELECT c.id,c.text,e.vector_json,e.model,e.dim,e.text_hash "
            "FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id "
            "WHERE c.chunk_kind='extraction' AND c.id IN (" +
            ",".join("?" * len(ids)) + ") ORDER BY c.id", ids,
        ).fetchall()
        if len(rows) != len(ids):
            raise ValueError("planned_batch_rows_changed")
        scanned_ids.extend(row["id"] for row in rows)
        pending_rows = []
        for row in rows:
            text_hash = embedding_text_hash(row["text"])
            stored = None
            if (row["text_hash"] == text_hash and row["model"] == model
                    and row["dim"] == dim):
                try:
                    stored = _finite_embedding_vector(
                        decode_vector(row["vector_json"]), expected_dim=dim)
                except (AttributeError, UnicodeError, TypeError, ValueError):
                    stored = None
            if stored is None:
                pending_rows.append((row["id"], row["text"], text_hash))
        if not pending_rows:
            continue
        pending += len(pending_rows)
        pending_ids.extend(row[0] for row in pending_rows)
        cached_by_hash = _fetch_cached_vectors(
            conn, [row[2] for row in pending_rows], model, expected_dim=dim)
        batch_misses = [row for row in pending_rows if row[2] not in cached_by_hash]
        cached += len(pending_rows) - len(batch_misses)
        misses += len(batch_misses)
        miss_chars += sum(len(row[1]) for row in batch_misses)
        miss_bytes += sum(len(row[1].encode("utf-8")) for row in batch_misses)
        if batch_misses:
            if (len(batch_misses) > MAX_TEXTS_PER_CALL
                    or sum(len(row[1]) for row in batch_misses) > MAX_CHARS_PER_CALL
                    or sum(len(row[1].encode("utf-8")) for row in batch_misses)
                    > MAX_UTF8_BYTES_PER_CALL):
                uncallable += 1
            else:
                planned_calls += 1
    return {"model_sha256": hashlib.sha256(model.encode()).hexdigest(),
            "dimension": dim, "scanned_batches": batches,
            "scanned_chunks": len(scanned_ids), "pending_chunks": pending,
            "cached_pending_chunks": cached, "remote_miss_texts": misses,
            "remote_miss_chars": miss_chars, "remote_miss_utf8_bytes": miss_bytes,
            "planned_http_calls": planned_calls,
            "uncallable_remote_batches": uncallable,
            "scanned_ids_sha256": hashlib.sha256("\n".join(scanned_ids).encode()).hexdigest(),
            "pending_ids_sha256": hashlib.sha256("\n".join(pending_ids).encode()).hexdigest(),
            "last_chunk_id_sha256": hashlib.sha256(
                (scanned_ids[-1] if scanned_ids else "").encode()).hexdigest()}


def graph_audit(conn) -> dict:
    from hymem.dreaming.canonicalize import find_canonical_drift
    from hymem.dreaming.evidence import count_mismatches

    integrity = conn.execute("PRAGMA integrity_check").fetchall()
    foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
    graph_hash = hashlib.sha256()
    for table in ("knowledge_graph", "kg_evidence", "kg_claim_observations"):
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid"):
            graph_hash.update(json.dumps(tuple(row), default=str,
                                         ensure_ascii=True).encode())
            graph_hash.update(b"\n")
    logical_hash = hashlib.sha256()
    tables = conn.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    for table in tables:
        name = table["name"]
        if (name in ("chunk_embeddings", "embedding_cache")
                or name.startswith("vec_") or "_fts_" in name):
            continue
        logical_hash.update(name.encode("utf-8") + b"\0")
        logical_hash.update((table["sql"] or "").encode("utf-8") + b"\0")
        quoted = '"' + name.replace('"', '""') + '"'
        predicate = (" WHERE key NOT IN ('vec_dim','vec_model')"
                     if name == "schema_meta" else "")
        for row in conn.execute(f"SELECT * FROM {quoted}{predicate} ORDER BY rowid"):
            logical_hash.update(json.dumps(tuple(row), default=str,
                                           ensure_ascii=True).encode() + b"\n")
    return {"integrity_ok": len(integrity) == 1 and integrity[0][0] == "ok",
            "foreign_key_findings": len(foreign),
            "canonical_drift_findings": len(find_canonical_drift(conn)),
            "ledger_count_mismatches": len(count_mismatches(conn)),
            "graph_core_sha256": graph_hash.hexdigest(),
            "nonembedding_logical_sha256": logical_hash.hexdigest()}


class Admission:
    def __init__(self, code: types.CodeType):
        self.code = code
        self.http_attempts = 0
        self.texts = 0
        self.characters = 0
        self.utf8_bytes = 0
        self.reason = None

    def profile(self, frame, event: str, _value) -> None:
        if event != "call" or frame.f_code is not self.code:
            return
        texts = frame.f_locals.get("texts")
        if not isinstance(texts, (list, tuple)):
            self.reason = "embedding_payload_invalid"
            raise BudgetStop("embedding_payload_invalid")
        count = len(texts)
        chars = sum(len(item) for item in texts if type(item) is str)
        try:
            utf8_bytes = sum(len(item.encode("utf-8")) for item in texts
                             if type(item) is str)
        except UnicodeError:
            self.reason = "embedding_payload_invalid"
            raise BudgetStop("embedding_payload_invalid") from None
        if (any(type(item) is not str for item in texts)
                or count > MAX_TEXTS_PER_CALL or chars > MAX_CHARS_PER_CALL
                or utf8_bytes > MAX_UTF8_BYTES_PER_CALL):
            self.reason = "embedding_payload_limit"
            raise BudgetStop("embedding_payload_limit")
        if not count:
            return
        if self.http_attempts >= MAX_EMBEDDING_HTTP:
            self.reason = "embedding_http_budget"
            raise BudgetStop("embedding_http_budget")
        self.http_attempts += 1
        self.texts += count
        self.characters += chars
        self.utf8_bytes += utf8_bytes


def safe_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if name in {"ValueError", "RuntimeError", "TypeError", "OSError",
                            "TimeoutError", "ConnectionError", "BudgetStop"} else "Exception"


def safe_frames(exc: BaseException) -> list[dict]:
    pattern = re.compile(r"hymem(?:/[A-Za-z_][A-Za-z_0-9]*)*/[A-Za-z_][A-Za-z_0-9]*\.py\Z")
    frames = []
    for index, (frame, line) in enumerate(traceback.walk_tb(exc.__traceback__)):
        if index >= 96:
            break
        filename = frame.f_code.co_filename
        if filename.startswith("/candidate/hymem/"):
            relative = filename[len("/candidate/"):]
            name = frame.f_code.co_name
            if pattern.fullmatch(relative) and name.isidentifier() and 1 <= line <= 100000:
                frames.append({"path": relative, "function": name, "line": line})
    return frames[-12:]


def run(mode: str, expected_source_sha: str, env_path: Path) -> dict:
    global STAGE
    from hymem.core.db import transaction
    from hymem.deadline import MonotonicDeadline, use_deadline
    from hymem.dreaming.embeddings import (
        chunk_embedding_id_batches, fetch_chunk_embeddings, persist_chunk_embeddings,
    )
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient

    if sha(SOURCE) != expected_source_sha:
        raise ValueError("source_snapshot_pin_mismatch")
    target_dir = WORK / mode
    target_dir.mkdir(mode=0o700, exist_ok=False)
    clone = target_dir / "hymem.sqlite"
    clone_source(clone)
    load_embedding_env(env_path, target_dir)
    STAGE = "client_identity"
    embedder = make_embedder()
    conn = None
    admission = None
    result = {"status": "error", "source_sha256": expected_source_sha,
              "cleanup_ok": False, "source_unchanged": False}
    primary = None
    try:
        conn = open_clone(clone)
        before = census(conn, embedder)
        result["before"] = before
        result["graph_before"] = graph_audit(conn)
        if before["uncallable_remote_batches"]:
            result["status"] = "budget_stopped"
            result["budget_reason"] = "planned_payload_limit"
            return result
        if before["planned_http_calls"] > MAX_EMBEDDING_HTTP:
            result["status"] = "budget_stopped"
            result["budget_reason"] = "planned_http_budget"
            return result
        if mode == "offline":
            result.update({"status": "ready", "http_attempts": 0,
                           "provider_request_attempts": 0})
            return result
        STAGE = "live_embedding"
        admission = Admission(OpenAICompatibleEmbeddingClient._embed_with_locked_transport.__code__)
        batches = persisted = cache_hits = 0
        progress = target_dir / "progress.jsonl"
        fd = os.open(progress, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as progress_stream:
            sys.setprofile(admission.profile)
            try:
                deadline = MonotonicDeadline.after(MAX_SECONDS)
                with use_deadline(deadline):
                    for ids in chunk_embedding_id_batches(conn):
                        deadline.check()
                        batches += 1
                        pending = fetch_chunk_embeddings(conn, embedder, chunk_ids=ids)
                        if pending is not None:
                            private_json(target_dir / f"pending-{batches:03d}.json",
                                         asdict(pending))
                            with transaction(conn):
                                persisted += persist_chunk_embeddings(conn, pending)
                            cache_hits += pending.cache_hits
                        record = {"batch": batches,
                                  "last_chunk_id_sha256": hashlib.sha256(ids[-1].encode()).hexdigest(),
                                  "persisted_total": persisted,
                                  "http_attempts": admission.http_attempts}
                        progress_stream.write(json.dumps(record, sort_keys=True).encode() + b"\n")
                        progress_stream.flush()
                        os.fsync(progress_stream.fileno())
                    deadline.check()
            finally:
                sys.setprofile(None)
        result.update({"status": "completed", "batches_processed": batches,
                       "vectors_persisted": persisted, "cache_hits": cache_hits})
        result["after"] = census(conn, embedder)
        result["graph_after"] = graph_audit(conn)
        result["graph_unchanged"] = result["graph_after"] == result["graph_before"]
        result["full_pending_coverage"] = result["after"]["pending_chunks"] == 0
        if not result["full_pending_coverage"] or not result["graph_unchanged"]:
            result["status"] = "incomplete"
    except BaseException as exc:
        primary = exc
        result["status"] = "budget_stopped" if isinstance(exc, BudgetStop) else "captured_failure"
        result["error_type"] = safe_type(exc)
        result["candidate_frames"] = safe_frames(exc)
        result["stage"] = STAGE
        try:
            private_json(target_dir / "failure.json", {
                "type": type(exc).__name__,
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            })
            result["failure_captured"] = True
        except BaseException:
            result["failure_captured"] = False
            result["status"] = "error"
    finally:
        sys.setprofile(None)
        if admission is not None:
            result.update({"http_attempts": admission.http_attempts,
                           "remote_texts_admitted": admission.texts,
                           "remote_chars_admitted": admission.characters})
            result["remote_utf8_bytes_admitted"] = admission.utf8_bytes
            if admission.reason is not None:
                result["budget_reason"] = admission.reason
        try:
            reported = embedder.request_attempts
            result["provider_request_attempts"] = reported if type(reported) is int else None
            if admission is not None:
                result["accounting_verified"] = (type(reported) is int
                                                 and 0 <= reported <= admission.http_attempts
                                                 <= MAX_EMBEDDING_HTTP)
            else:
                result["accounting_verified"] = type(reported) is int and reported == 0
            embedder.close()
            result["cleanup_ok"] = True
        except BaseException:
            result["cleanup_ok"] = False
        if conn is not None:
            try:
                conn.close()
            except BaseException:
                result["cleanup_ok"] = False
        try:
            result["source_unchanged"] = sha(SOURCE) == expected_source_sha
            result["clone_sha256"] = sha(clone)
        except BaseException:
            result["source_unchanged"] = False
            result["status"] = "error"
            result["error_code"] = "final_hash_failed"
        if not result["cleanup_ok"] or not result.get("accounting_verified"):
            result["status"] = "error"
            result["error_code"] = "cleanup_or_accounting_unverified"
    return result


def main() -> int:
    global STAGE
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("offline", "live"))
    parser.add_argument("--source-sha256", default=os.environ.get("CLAIM_SOURCE_SHA256"))
    parser.add_argument("--env", type=Path, default=Path("/run/runtime-env.json"))
    args = parser.parse_args()
    if not isinstance(args.source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", args.source_sha256) is None:
        raise ValueError("source_pin_invalid")
    if not WORK.is_dir() or WORK.is_symlink() or stat.S_IMODE(WORK.stat().st_mode) != 0o700:
        raise ValueError("private_work_directory_invalid")
    STAGE = args.mode
    result = run(args.mode, args.source_sha256, args.env)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return (0 if result["status"] in ("ready", "completed", "budget_stopped", "captured_failure")
            and result["cleanup_ok"] and result["source_unchanged"] else 1)


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink():
                private_json(WORK / "setup-failure.json", {
                    "type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                })
                captured = True
        except BaseException:
            pass
        print(json.dumps({"status": "error", "error_type": safe_type(exc),
                          "stage": STAGE, "candidate_frames": safe_frames(exc),
                          "failure_captured": captured},
                         sort_keys=True))
        rc = 1
    raise SystemExit(rc)
