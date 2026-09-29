"""Pinned v2 adapter with explicit, verified stopped SQLite WAL checkpoint."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3

ORIGINAL_SHA256 = "222a4c1222d8a36df94bda6203c53aeeab3a74248142efc39b996f2ac224ebe0"
V2_SHA256 = "f13af8054b0cd1e75e13926caa030e02bf1597c6d61b07320dd695db9a27e3f6"


def load_v2(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("v2_helper_unsafe")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != V2_SHA256:
        raise RuntimeError("v2_helper_pin_mismatch")
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location("_hymem_v64_v2_sealed", path))
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if module.ORIGINAL_SHA256 != ORIGINAL_SHA256:
        raise RuntimeError("original_helper_pin_mismatch")
    return module


v2 = load_v2(Path(__file__).with_name("hymem_v64_rollout_v2.py"))
original = v2.original
MIGRATION = original.MIGRATION
# Reuse the sealed fingerprint implementation, including vector shadow tables,
# but include schema_meta and perform no initialize or migration calls.
SNAPSHOT = MIGRATION[:MIGRATION.index("c=db.connect(path)")].replace("if r[0]!='schema_meta'", "").replace(" AND name NOT LIKE 'sqlite_%'", "") + r'''
import shutil,tempfile
# connect configures WAL; operate only on an ephemeral copy of the read-only
# frozen input so the application's normal connection setup cannot alter it.
workspace=tempfile.TemporaryDirectory(dir=path.parent)
copy=pathlib.Path(workspace.name)/'snapshot.sqlite'
shutil.copyfile(path,copy)
path=copy
c=db.connect(path)
try:
    need(db._load_vec_extension(c),'vec_extension_unavailable')
    c.execute('PRAGMA query_only=ON')
    need(db.schema_version(c)==63,'checkpoint_schema_not_63');health(c)
    columns,rows=snapshot(c)
    schema=[list(r) for r in c.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
finally:
    c.close()
    workspace.cleanup()
print(json.dumps({'status':'passed','schema_version':63,'columns':columns,'rows':rows,'schema':schema}))
'''


def checkpoint(path):
    """Physical checkpoint only; callers must establish stopped backup guards."""
    path = Path(path)
    original.regular(path)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            original.regular(sidecar)
    c = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=0)
    try:
        result = c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        original.need(result is not None and tuple(result) in ((0, 0, 0), (0, -1, -1)),
                      "checkpoint_busy_or_unverified")
        wal = Path(str(path) + "-wal")
        original.need(not wal.exists() or (not wal.is_symlink() and wal.stat().st_size == 0),
                      "checkpoint_wal_not_empty")
    finally:
        c.close()
    wal = Path(str(path) + "-wal")
    original.need(not wal.exists() or (not wal.is_symlink() and wal.stat().st_size == 0),
                  "checkpoint_wal_not_empty")
    return list(result)


def snapshot_offline(self, path):
    # Same sealed offline runtime, source and isolation as migration; frozen
    # backup is mounted read-only and query_only independently forbids writes.
    cmd = ["docker", "run", "--rm", "--name", "hymem-v64-" + self.stage.name,
           "--pull", "never", "--network", "none", "--user", "1000:1000", "--read-only",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "128",
           "--memory", "2g", "--cpus", "2", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
           "--tmpfs", "/database:rw,noexec,nosuid,size=256m,mode=1777",
           "--mount", "type=bind,src=" + str(self.source) + ",dst=/source,readonly",
           "--mount", "type=bind,src=" + str(original.RUNTIME) + ",dst=/home/node/hymem-env,readonly",
           "--mount", "type=bind,src=" + str(path) + ",dst=/database/" + path.name + ",readonly",
           "--entrypoint", "/home/node/hymem-env/bin/python3", original.IMAGE, "-I", "-B", "-c",
           "DBPATH=" + repr("/database/" + path.name) + "\n" + SNAPSHOT]
    result = json.loads(self.run(cmd, timeout=600))
    original.need(result.get("status") == "passed" and result.get("schema_version") == 63
                  and all(k in result for k in ("columns", "rows", "schema")), "checkpoint_snapshot_unverified")
    return result


def verified_checkpoint(self, backup):
    self.inspect(stopped=True)
    self.preserve()
    original.verify_files(self.stage / "source-backup", self.old)
    frozen = self.stage / "stopped.sqlite"
    original.need(original.sha(frozen) == backup["sha256"], "stopped_backup_drift")
    before = snapshot_offline(self, frozen)
    self.inspect(stopped=True)
    result = checkpoint(original.DB)
    self.inspect(stopped=True)
    after_backup = self.backup("checkpointed.sqlite")
    after = snapshot_offline(self, self.stage / "checkpointed.sqlite")
    original.need(before == after, "checkpoint_logical_rows_changed")
    original.need(original.sha(frozen) == backup["sha256"]
                  and original.sha(self.stage / "checkpointed.sqlite") == after_backup["sha256"],
                  "checkpoint_backup_drift")
    self.preserve()
    original.verify_files(self.stage / "source-backup", self.old)
    self.inspect(stopped=True)
    wal = Path(str(original.DB) + "-wal")
    original.need(not wal.exists() or (not wal.is_symlink() and wal.stat().st_size == 0),
                  "checkpoint_wal_not_empty")
    return {"result": result, "logical_sha256": original.digest(before), "backup": after_backup}


# Execute the sealed stop body with one explicit insertion, retaining every
# original precondition, backup, rehearsal, source and runtime guard.
import inspect
import textwrap
stop_source = textwrap.dedent(inspect.getsource(original.Rollout.stop))
needle = '    return self.receipt("stop", {"backup": backup, "fresh_backup_migration": checked,\n'
original.need(stop_source.count(needle) == 1, "stop_shape_changed")
stop_source = stop_source.replace(needle,
    '    checkpoint_receipt = verified_checkpoint(self, backup)\n' + needle)
stop_source = stop_source.replace('"production_db_sha256": sha(DB)})',
                                '"production_db_sha256": sha(DB), "checkpoint": checkpoint_receipt})')
namespace = dict(original.__dict__, verified_checkpoint=verified_checkpoint)
exec(compile(stop_source, __file__, "exec"), namespace)
original.Rollout.stop = namespace["stop"]


if __name__ == "__main__":
    try:
        raise SystemExit(original.main())
    except Exception as exc:
        code = str(exc) if type(exc) is RuntimeError and re.fullmatch("[a-z_]+", str(exc)) else type(exc).__name__
        print(json.dumps({"status": "failed", "preflight_rejected": True, "failure_code": code}))
        raise SystemExit(1)
