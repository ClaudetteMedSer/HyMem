"""Pinned rollout adapter: load sqlite-vec before each migration inspection.

All actions and guards come from the sealed original helper. Deploy this file
beside that original; existing rollout receipts remain untouched.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re

ORIGINAL_SHA256 = "222a4c1222d8a36df94bda6203c53aeeab3a74248142efc39b996f2ac224ebe0"


def load_original(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("original_helper_unsafe")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("original_helper_pin_mismatch")
    spec = importlib.util.spec_from_file_location("_hymem_v64_rollout_sealed", path)
    original = importlib.util.module_from_spec(spec)
    # Execute precisely the bytes whose seal was checked.
    exec(compile(raw, str(path), "exec"), original.__dict__)
    connection = "c=db.connect(path)\ntry:\n"
    if original.MIGRATION.count(connection) != 2:
        raise RuntimeError("migration_connection_shape_changed")
    original.MIGRATION = original.MIGRATION.replace(
        connection,
        connection + "    need(db._load_vec_extension(c),'vec_extension_unavailable')\n",
    )
    return original


original = load_original(Path(__file__).with_name("hymem_v64_rollout.py"))
MIGRATION = original.MIGRATION


if __name__ == "__main__":
    try:
        raise SystemExit(original.main())
    except Exception as exc:
        code = str(exc) if type(exc) is RuntimeError and re.fullmatch("[a-z_]+", str(exc)) else type(exc).__name__
        print(json.dumps({"status": "failed", "preflight_rejected": True, "failure_code": code}))
        raise SystemExit(1)
