#!/usr/bin/env python3
"""Network-free persistence replay of one private capture on one pinned source tree."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import stat
import sys

CAPTURE_HELPER = Path("/diag/claim_conflict_next_capture.py")
CAPTURE_HELPER_SHA256 = "4ad2819309f51e96eb7b7195d4e215b1d53081ba9551df89cd9c657c23d3178c"
SOURCE_SHA256 = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
WORK = Path("/work")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("private_capture_file_invalid")


def load_helper():
    if sha(CAPTURE_HELPER) != CAPTURE_HELPER_SHA256:
        raise ValueError("capture_helper_pin_mismatch")
    spec = importlib.util.spec_from_file_location("claim_conflict_next_capture_pinned", CAPTURE_HELPER)
    if spec is None or spec.loader is None:
        raise ValueError("capture_helper_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reason(exc: BaseException) -> str:
    # Equality checks use only fixed application messages. The exception text
    # itself may contain private source values and never enters the receipt.
    known = {
        "same-generation claim observations disagree": "same_generation_observation_disagreement",
        "canonical evidence source provenance collides": "canonical_evidence_provenance_collision",
        "canonical edge merge found conflicting evidence audit": "canonical_evidence_audit_collision",
        "canonical edge merge found conflicting claim authority": "canonical_claim_authority_collision",
        "claim extraction outcome requires a published source manifest": "missing_published_source_manifest",
        "duplicate claim citation in validated extraction": "duplicate_validated_claim_citation",
        "same prompt generation claim extraction outcomes disagree": "same_generation_extraction_outcome_disagreement",
    }
    return known.get(str(exc), "unclassified_exception")


def clone_digest(helper, name: str) -> str | None:
    target = WORK / name
    if not target.is_file() or target.is_symlink():
        return None
    from hymem.core.db import connect, _load_vec_extension
    conn = connect(target)
    try:
        has_vec = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%USING vec0%'"
        ).fetchone()
        if has_vec and not _load_vec_extension(conn):
            raise RuntimeError("replay_clone_vector_extension_unavailable")
        return helper._logical_digest(conn)
    finally:
        conn.close()


def run_arm(helper, *, dedup_enabled: bool, name: str, baseline_digest: str) -> dict:
    try:
        result = helper.replay(dedup_enabled=dedup_enabled, name=name)
        return {
            "status": result["status"],
            "reason_code": ("same_generation_observation_disagreement"
                            if result["status"] == "rejected" else None),
            "error_type": result["error_type"],
            "collision_count": len(result["observation_collisions"]),
            "changed_field_sets": sorted({tuple(item["changed_fields"])
                                          for item in result["observation_collisions"]}),
            "logical_digest_before": result["logical_digest_before"],
            "logical_digest_after": result["logical_digest_after"],
            "rollback_preserved": (result["logical_digest_after"] == baseline_digest
                                   if result["status"] == "rejected" else None),
        }
    except Exception as exc:
        diagnostic_error = None
        try:
            after = clone_digest(helper, name)
        except Exception as diagnostic_exc:
            after = None
            diagnostic_error = (type(diagnostic_exc).__name__
                                if type(diagnostic_exc).__module__ == "builtins" else "Exception")
        code = reason(exc)
        rollback = after == baseline_digest if after is not None else None
        status = ("rejected" if code != "unclassified_exception" and rollback is True
                  else "setup_failure" if after is None else "execution_failure")
        return {
            "status": status, "reason_code": code,
            "error_type": type(exc).__name__ if type(exc).__module__ == "builtins" else "Exception",
            "candidate_frames": helper._safe_frames(exc),
            "logical_digest_before": baseline_digest,
            "logical_digest_after": after,
            "rollback_preserved": rollback,
            "diagnostic_error_type": diagnostic_error,
        }


def main() -> None:
    logging.disable(logging.CRITICAL)
    if sha(Path("/reference/source.sqlite")) != SOURCE_SHA256:
        raise ValueError("reference_snapshot_pin_mismatch")
    for name in ("claim-conflict-capture.json", "claim-conflict-vectors.json"):
        private_file(WORK / name)
    helper = load_helper()
    if helper.EXPECTED_SOURCE_SHA256 != SOURCE_SHA256:
        raise ValueError("capture_source_pin_mismatch")
    payload, chunk, extraction = helper._decode_capture()
    if extraction.failed or not (WORK / "claim-conflict-vectors.json").is_file():
        raise ValueError("capture_not_successful")
    if payload["source_sha256"] != SOURCE_SHA256 or chunk.id != helper.CHUNK_ID:
        raise ValueError("capture_identity_mismatch")
    control, _ = helper._clone("control.sqlite")
    try:
        baseline_digest = helper._logical_digest(control)
    finally:
        control.close()
    result = {
        "status": "replayed", "capture_sha256": sha(helper.CAPTURE),
        "vectors_sha256": sha(helper.VECTORS),
        "source_sha256": SOURCE_SHA256,
        "dedup_on": run_arm(helper, dedup_enabled=True,
                            name="dedup-on.sqlite", baseline_digest=baseline_digest),
        "dedup_off": run_arm(helper, dedup_enabled=False,
                             name="dedup-off.sqlite", baseline_digest=baseline_digest),
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # No exception text, source lines, private data, or external paths.
        kind = type(exc).__name__ if type(exc).__module__ == "builtins" else "Exception"
        print(json.dumps({"status": "error", "reason_code": "replay_setup_failed",
                          "error_type": kind}, sort_keys=True))
        sys.exit(1)
