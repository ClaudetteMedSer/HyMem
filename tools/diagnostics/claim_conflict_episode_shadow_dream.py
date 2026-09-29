#!/usr/bin/env python3
"""Bounded private v64 dream; explicitly upgrade the v63 clone before use.

The producer/HTTP/capture instrumentation is the separately pinned reviewed
worker. This adapter adds only a pre-provider schema64 admission boundary.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import sys
import traceback

sys.path.insert(0, "/candidate")

OLD_WORKER = Path("/diag/claim_conflict_episode_shadow_dream_instrumented.py")
OLD_WORKER_SHA = "e095f534da3457c5f24f7688a8f2eb6bc6b4d6f8981b630abec26c2de49739d8"
WORK = Path("/work")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def load_old():
    if (not OLD_WORKER.is_file() or OLD_WORKER.is_symlink()
            or hashlib.sha256(OLD_WORKER.read_bytes()).hexdigest() != OLD_WORKER_SHA):
        raise RuntimeError("instrumented_worker_pin_drift")
    spec = importlib.util.spec_from_file_location("v64_pinned_instrumentation", OLD_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upgrade_clone(old, target: Path) -> None:
    """Only the private clone is writable; source remains pinned and read-only."""
    from hymem.core.db import connect, initialize, schema_version, _load_vec_extension

    old.original_clone_source(target)
    conn = connect(target)
    try:
        if schema_version(conn) != 63:
            raise RuntimeError("reference_schema_not_v63")
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%USING vec0%'"
        ).fetchone() and not _load_vec_extension(conn):
            raise RuntimeError("snapshot_vector_extension_unavailable")
        initialize(conn)
        if schema_version(conn) != 64:
            raise RuntimeError("clone_schema_upgrade_incomplete")
        if not any(str(row[1]) == "local_replay_proof" for row in conn.execute(
                "PRAGMA table_info(kg_claim_extraction_outcomes)")):
            raise RuntimeError("clone_proof_column_missing")
        if conn.execute("SELECT COUNT(*) FROM kg_claim_extraction_outcomes "
                        "WHERE local_replay_proof IS NOT NULL").fetchone()[0] != 0:
            raise RuntimeError("clone_historical_proof_invented")
        check = conn.execute("PRAGMA integrity_check").fetchall()
        if len(check) != 1 or check[0][0] != "ok" or conn.execute(
                "PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("clone_upgrade_integrity_failed")
    finally:
        conn.close()
    if old.sha(old.SOURCE) != old.SOURCE_SHA256:
        raise RuntimeError("reference_changed_during_clone_upgrade")


def main() -> int:
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    old = load_old()
    old.original_clone_source = old.clone_source
    old.clone_source = lambda target: upgrade_clone(old, target)
    # The pinned worker retains its exact SDK identity/profile hooks and
    # validates 128 completions, 384 LLM/512 embedding HTTP, 896 total, 2700s.
    return old.main()


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink() and stat.S_IMODE(WORK.stat().st_mode) == 0o700:
                raw = json.dumps({
                    "type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(
                        type(exc), exc, exc.__traceback__)),
                }, sort_keys=True, ensure_ascii=True).encode()
                fd = os.open(WORK / "setup-failure-v64.json",
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                captured = True
        except BaseException:
            pass
        safe_type = type(exc).__name__ if type(exc).__name__ in {
            "RuntimeError", "ValueError", "OSError", "OperationalError",
            "IntegrityError", "TypeError", "KeyError"} else "Exception"
        print(json.dumps({"status": "error", "error_type": safe_type,
                          "stage": "v64_setup", "failure_captured": captured}, sort_keys=True))
        rc = 1
    raise SystemExit(rc)
