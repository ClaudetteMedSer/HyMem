"""Host-side, receipt-bound Hermes1 schema64 rollout (no implicit SSH).

Run one action per invocation on Afrodite, after reviewing and sealing config.
Failures leave the service/database in place for explicit operator recovery.
This helper never installs dependencies, restores a database, or prints raw
subprocess output. Packaging differences fail closed for a separate review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import traceback

CANDIDATE_PIN = "5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576"
BASELINE_PIN = "1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8"
DOCTOR_PIN = "2ac786c6590aa7fbece726e6380981906d514fefe9f664a3ca4f7b1d1de7c15b"
PHASE1_PIN = "31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136"
IMAGE = "sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5"
HOME = Path("/opt/stacks/hermes/instance1/home")
LIVE = HOME / "HyMem"
RUNTIME = HOME / "hymem-env"
DB = HOME / ".hermes/hymem.sqlite"
HEX = re.compile(r"[0-9a-f]{64}\Z")


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def regular(path):
    path = Path(path)
    need(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "unsafe_file")
    return path


def safe_rel(name):
    p = PurePosixPath(name)
    need(name == p.as_posix() and bool(p.parts) and not p.is_absolute()
         and ".." not in p.parts and not any(x.startswith(".") for x in p.parts)
         and p.parts[0] in ("hymem", "tests", "benchmarks", "references", "tools", "README.md", "pyproject.toml"),
         "unsafe_manifest_path")
    return p


def safe_target(root, name):
    p = root.joinpath(*safe_rel(name).parts)
    for parent in (p, *p.parents):
        need(not parent.is_symlink(), "source_path_symlink")
        if parent == root:
            break
    return p


def manifest(path, pin, count, grouped=False):
    regular(path)
    raw = json.loads(Path(path).read_bytes())
    need((sha(path) if grouped else digest(raw)) == pin, "manifest_pin_mismatch")
    files = {}
    for group in ("source_sha256", "test_sha256", "auxiliary_sha256") if grouped else (None,):
        for name, value in (raw[group] if group else raw).items():
            safe_rel(name)
            need(name not in files and isinstance(value, str) and HEX.fullmatch(value), "manifest_shape")
            files[name] = value
    need(len(files) == count, "manifest_count")
    return files


def verify_files(root, files):
    need(root.is_dir() and not root.is_symlink(), "source_root_invalid")
    for name, pin in files.items():
        p = safe_target(root, name)
        need(sha(regular(p)) == pin, "source_hash_drift")


def inventory(root):
    """Private full runtime/config identity; symlinks are recorded, never followed."""
    root = Path(root)
    need(root.exists() and not root.is_symlink(), "preserved_root_invalid")
    result = {}
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for p in paths:
        key = p.relative_to(root).as_posix() if p != root else "."
        info = p.lstat()
        value = {"mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid}
        if p.is_symlink():
            value["link"] = os.readlink(p)
        elif p.is_file():
            value["sha256"] = sha(p)
        elif p.is_dir():
            value["directory"] = True
        else:
            need(False, "preserved_special_file")
        result[key] = value
    return result


def exclusive_json(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def read_sealed(path, pin):
    need(isinstance(pin, str) and HEX.fullmatch(pin), "unresolved_receipt_pin")
    regular(path)
    need(sha(path) == pin, "receipt_seal_drift")
    return json.loads(Path(path).read_bytes())


def gates(cfg):
    for role in ("fullsuite", "paid_postflight"):
        gate = cfg["gates"][role]
        value = read_sealed(gate["path"], gate["sha256"])
        need(value.get("status") == "passed" and value.get("candidate_manifest_sha256") == CANDIDATE_PIN
             and value.get("role") == role and value.get("root_reviewed") is True,
             "candidate_gate_not_passed")
        # Root-normalized, separately sealed metadata receipts are required.
        # Raw historical receipts are deliberately not inferred as approval.
        if role == "fullsuite":
            need(type(value.get("collected")) is int and value["collected"] > 0
                 and value.get("collected") == value.get("expected_collected")
                 and type(value.get("passed")) is int and value["passed"] > 0
                 and value["passed"] + value.get("skipped", 0) + value.get("xfailed", 0) == value["collected"]
                 and value.get("skipped", 0) == value.get("expected_skipped", 0)
                 and value.get("xfailed", 0) == value.get("expected_xfailed", 0)
                 and value.get("xpassed", 0) == 0
                 and value.get("failed") == 0 and value.get("errors") == 0
                 and value.get("exit_code") == 0 and value.get("cleanup_verified") is True,
                 "fullsuite_collection_or_cleanup_missing")
        else:
            for key in ("claims", "ledger", "canonical", "same_generation", "integrity", "foreign_keys", "episode_vectors"):
                need(value.get("checks", {}).get(key) is True, "candidate_gate_checks_missing")
            need(value.get("aggregation_passed") is True and value.get("paid_calls", 0) > 0,
                 "paid_gate_missing")


MIGRATION = r'''
import hashlib,json,pathlib,sys
sys.path.insert(0,'/source')
from hymem.core import db
path=pathlib.Path(DBPATH)
def need(ok,code):
    if not ok: raise RuntimeError(code)
def q(name):return '"'+name.replace('"','""')+'"'
def snapshot(c,columns=None):
    result={}
    if columns is None:
        columns={r[0]:[x[1] for x in c.execute('PRAGMA table_info('+q(r[0])+')')]
                 for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                 if r[0]!='schema_meta'}
    for table,cols in columns.items():
        rows=[]
        for row in c.execute('SELECT '+','.join(q(x) for x in cols)+' FROM '+q(table)):
            vals=[{'bytes':x.hex()} if isinstance(x,bytes) else x for x in row]
            rows.append(hashlib.sha256(json.dumps(vals,separators=(',',':'),allow_nan=False).encode()).digest())
        h=hashlib.sha256()
        for row in sorted(rows):h.update(row)
        result[table]={'count':len(rows),'sha256':h.hexdigest()}
    return columns,result
def health(c):
    need([x[0] for x in c.execute('PRAGMA integrity_check')]==['ok'],'integrity')
    need(c.execute('PRAGMA foreign_key_check').fetchone() is None,'foreign_keys')
c=db.connect(path)
try:
    before=db.schema_version(c);need(before==63,'before_schema');health(c)
    columns,old=snapshot(c)
    db.initialize(c);need(db.schema_version(c)==64,'after_schema');health(c)
    _,new=snapshot(c,columns);need(old==new,'durable_rows_changed')
    need(c.execute('SELECT COUNT(*) FROM kg_claim_extraction_outcomes WHERE local_replay_proof IS NOT NULL').fetchone()[0]==0,'historical_proof_minted')
    _,all_after=snapshot(c)
finally:c.close()
c=db.connect(path)
try:
    db.initialize(c);need(db.schema_version(c)==64,'reopen_schema');health(c)
    _,reopened=snapshot(c);need(reopened==all_after,'reopen_rows_changed')
finally:c.close()
print(json.dumps({'status':'passed','schema_before':63,'schema_after':64,'preserved_tables':len(old),
 'rows_preserved':True,'historical_proofs_null':True,'reopen_stable':True,'integrity_ok':True,'foreign_keys_ok':True}))
'''


class Rollout:
    def __init__(self, cfg, config_sha):
        self.cfg = cfg
        self.config_sha = config_sha
        self.stage = Path(cfg["stage"])
        self.source = Path(cfg["candidate"])
        need(self.stage.parent == HOME / ".hermes/benchmarks"
             and re.fullmatch(r"hymem-v64-rollout-[a-z0-9-]+", self.stage.name), "fresh_stage_name_required")
        need(not self.stage.parent.is_symlink(), "stage_parent_symlink")
        if self.stage.exists():
            need(self.stage.is_dir() and not self.stage.is_symlink()
                 and stat.S_IMODE(self.stage.stat().st_mode) == 0o700
                 and self.stage.stat().st_uid == os.geteuid(), "stage_identity_invalid")
        need(cfg.get("candidate_manifest_sha256") == CANDIDATE_PIN and cfg.get("image") == IMAGE,
             "candidate_or_image_pin")
        self.files = manifest(cfg["candidate_manifest"], CANDIDATE_PIN, 481)
        self.old = manifest(cfg["baseline_manifest"], BASELINE_PIN, 479, grouped=True)
        self.old.update({"hymem/doctor.py": DOCTOR_PIN, "hymem/dreaming/phase1.py": PHASE1_PIN})
        need(set(self.old) <= set(self.files), "unexpected_source_deletion")
        verify_files(self.source, self.files)
        gates(cfg)
        census = cfg["census"]
        self.census = read_sealed(census["path"], census["sha256"])
        need(self.census.get("root_reviewed") is True and self.census.get("schema") == 63
             and self.census.get("image") == IMAGE and self.census.get("container_id"), "fresh_census_required")
        need(self.old["pyproject.toml"] == self.files["pyproject.toml"], "packaging_changed_separate_install_review")
        need(isinstance(self.census.get("preserved_paths"), dict)
             and str(RUNTIME) in self.census["preserved_paths"]
             and len(self.census["preserved_paths"]) >= 3, "runtime_config_wrapper_inventory_required")

    def run(self, cmd, timeout=60):
        # stderr/stdout stay private; callers see only bounded known metadata.
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if self.stage.is_dir() and not self.stage.is_symlink():
            fd, log = tempfile.mkstemp(prefix="private-subprocess-", dir=self.stage)
            with os.fdopen(fd, "wb") as stream:
                stream.write(p.stdout)
                stream.write(b"\nPRIVATE STDERR\n")
                stream.write(p.stderr)
        need(p.returncode == 0, "subprocess_failed_inspect_private_logs")
        need(len(p.stdout) < 2_000_000, "subprocess_output_oversized")
        return p.stdout

    def inspect(self, stopped=False):
        obj = json.loads(self.run(["docker", "inspect", "hermes-1"]))[0]
        need(obj["Name"] == "/hermes-1" and obj["Image"] == IMAGE
             and obj["Id"] == self.census["container_id"]
             and digest(obj["Config"]) == self.census["docker_config_sha256"], "container_identity_drift")
        s = obj["State"]
        if stopped:
            need(s["Status"] == "exited" and not s["Running"] and s["Pid"] == 0
                 and not s["OOMKilled"] and s["ExitCode"] != 137, "service_not_normally_stopped")
        else:
            need(s["Running"] and not s["OOMKilled"], "service_not_running")
        return s

    def preserve(self):
        for path, pin in self.census["preserved_paths"].items():
            need(isinstance(pin, str) and HEX.fullmatch(pin) and digest(inventory(Path(path))) == pin,
                 "runtime_config_wrapper_drift")

    def verify_before(self):
        verify_files(LIVE,self.old)
        for name in set(self.files)-set(self.old):
            target=safe_target(LIVE,name)
            need(not os.path.lexists(target),"new_candidate_path_already_present")

    def idle(self):
        self.inspect()
        port = self.census["health_port"]
        need(type(port) is int and 1 <= port <= 65535, "health_port_required")
        script = ("import json,urllib.request; b='http://127.0.0.1:" + str(port) + "';"
                  "h=json.load(urllib.request.urlopen(b+'/health',timeout=5));"
                  "s=json.load(urllib.request.urlopen(b+'/dream-status',timeout=5));"
                  "print(json.dumps({'health':h,'in_progress':s['in_progress']}))")
        value = json.loads(self.run(["docker", "exec", "hermes-1", "/home/node/hymem-env/bin/python3", "-I", "-B", "-c", script]))
        need(value == {"health": {"status": "ok", "backend": "hymem"}, "in_progress": False}, "service_not_idle")
        # Bind the exact current honcho process and effective environment without printing it.
        pid = self.census["honcho_pid"]
        need(type(pid) is int and pid > 0, "honcho_pid_required")
        raw = self.run(["docker", "exec", "hermes-1", "/home/node/hymem-env/bin/python3", "-I", "-B", "-c",
                        "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('/proc/" + str(pid) + "/environ').read_bytes()).hexdigest())"])
        need(raw.decode().strip() == self.census["honcho_environ_sha256"], "effective_environment_drift")

    def receipt(self, action, value):
        body = {"status": "passed", "action": action, "config_sha256": self.config_sha,
                "candidate_manifest_sha256": CANDIDATE_PIN, **value}
        exclusive_json(self.stage / (action + ".json"), body)
        return body

    def prior(self, name):
        value = json.loads(regular(self.stage / (name + ".json")).read_bytes())
        need(value.get("status") == "passed" and value.get("config_sha256") == self.config_sha
             and value.get("candidate_manifest_sha256") == CANDIDATE_PIN, "prior_receipt_invalid")
        return value

    def intent(self, action):
        need(not (self.stage / (action + ".json")).exists(), "action_already_completed")
        exclusive_json(self.stage / (action + "-intent.json"), {"action": action, "config_sha256": self.config_sha})

    def backup(self, name):
        target = self.stage / name
        regular(DB)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        src = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True, timeout=30)
        dst = sqlite3.connect(target)
        try:
            src.backup(dst, pages=1024, sleep=0.1)
        finally:
            dst.close()
            src.close()
        return {"sha256": sha(target), "bytes": target.stat().st_size}

    def offline(self, db_path, source):
        cmd = ["docker", "run", "--rm", "--name", "hymem-v64-" + self.stage.name,
               "--pull", "never", "--network", "none", "--user", "1000:1000", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "128",
               "--memory", "2g", "--cpus", "2", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
               "--tmpfs", "/database:rw,noexec,nosuid,size=256m,mode=1777",
               "--mount", "type=bind,src=" + str(source) + ",dst=/source,readonly",
               "--mount", "type=bind,src=" + str(RUNTIME) + ",dst=/home/node/hymem-env,readonly",
               "--mount", "type=bind,src=" + str(db_path) + ",dst=/database/" + db_path.name,
               "--entrypoint", "/home/node/hymem-env/bin/python3", IMAGE, "-I", "-B", "-c",
               "DBPATH=" + repr("/database/" + db_path.name) + "\n" + MIGRATION]
        value = json.loads(self.run(cmd, timeout=600))
        need(value.get("status") == "passed" and value.get("schema_after") == 64
             and all(value.get(k) is True for k in ("rows_preserved", "historical_proofs_null", "reopen_stable", "integrity_ok", "foreign_keys_ok")),
             "migration_verification_failed")
        return value

    def stage_action(self):
        need(not self.stage.exists(), "stage_already_exists")
        self.idle()
        self.preserve()
        self.verify_before()
        self.stage.mkdir(mode=0o700)
        self.intent("stage")
        # Snapshot config/runtime/wrappers before any production mutation.
        backup = self.stage / "preserved-backup"
        backup.mkdir(mode=0o700)
        for index, path in enumerate(self.census["preserved_paths"]):
            src = Path(path)
            dst = backup / str(index)
            if src.is_dir():
                shutil.copytree(src, dst, symlinks=True)
            else:
                shutil.copy2(src, dst)
            need(digest(inventory(dst)) == self.census["preserved_paths"][path], "preserved_backup_drift")
        delta = {n: p for n, p in self.files.items() if self.old.get(n) != p}
        return self.receipt("stage", {"source_delta_files": len(delta), "source_delta_sha256": digest(delta), "install_required": False})

    def rehearse(self):
        self.prior("stage")
        self.idle()
        self.preserve()
        self.verify_before()
        self.intent("rehearse")
        baseline = self.backup("pre-rehearsal.sqlite")
        path = self.stage / "rehearsal.sqlite"
        shutil.copy2(self.stage / "pre-rehearsal.sqlite", path)
        result = self.offline(path, self.source)
        need(sha(self.stage / "pre-rehearsal.sqlite") == baseline["sha256"], "rehearsal_baseline_drift")
        return self.receipt("rehearse", {"backup": baseline, "migration": result, "migrated_sha256": sha(path)})

    def stop(self):
        self.prior("rehearse")
        self.idle()
        self.preserve()
        self.verify_before()
        self.intent("stop")
        self.run(["docker", "stop", "-t", "120", "hermes-1"], timeout=150)
        self.inspect(stopped=True)
        backup = self.backup("stopped.sqlite")
        check = self.stage / "stopped-check.sqlite"
        shutil.copy2(self.stage / "stopped.sqlite", check)
        checked = self.offline(check, self.source)
        need(sha(self.stage / "stopped.sqlite") == backup["sha256"], "stopped_backup_drift")
        source_backup = self.stage / "source-backup"
        source_backup.mkdir(mode=0o700)
        for n in self.old:
            p = safe_target(LIVE, n)
            dst = source_backup / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        verify_files(source_backup, self.old)
        self.preserve()
        return self.receipt("stop", {"backup": backup, "fresh_backup_migration": checked,
                                     "production_db_sha256": sha(DB)})

    def stopped_guard(self):
        stop = self.prior("stop")
        self.inspect(stopped=True)
        self.preserve()
        # A file-only offline mount must never omit uncheckpointed WAL data.
        wal = Path(str(DB) + "-wal")
        need(not wal.exists() or (not wal.is_symlink() and wal.stat().st_size == 0),
             "stopped_wal_requires_explicit_checkpoint_review")
        verify_files(self.stage / "source-backup", self.old)
        need(sha(self.stage / "stopped.sqlite") == stop["backup"]["sha256"], "stopped_backup_drift")
        for index, path in enumerate(self.census["preserved_paths"]):
            need(digest(inventory(self.stage / "preserved-backup" / str(index)))
                 == self.census["preserved_paths"][path], "preserved_backup_drift")
        return stop

    def apply(self):
        stop = self.stopped_guard()
        need(sha(DB) == stop["production_db_sha256"], "stopped_db_drift")
        self.verify_before()
        self.intent("apply")
        for name, pin in sorted(self.files.items()):
            if self.old.get(name) == pin:
                continue
            target = safe_target(LIVE, name)
            info = target.stat() if target.exists() else None
            need(info is None or (info.st_uid == os.geteuid() and info.st_gid in os.getgroups()), "source_owner_drift")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=".v64-", dir=target.parent)
            with os.fdopen(fd, "wb") as stream:
                stream.write((self.source / name).read_bytes())
                os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode) if info else 0o644)
                if info:
                    os.fchown(stream.fileno(), info.st_uid, info.st_gid)
                stream.flush()
                os.fsync(stream.fileno())
            need(sha(temp) == pin, "source_replace_pin")
            os.replace(temp, target)
            dirfd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        verify_files(LIVE, self.files)
        self.preserve()
        return self.receipt("apply", {"source_files_verified": 481, "install_performed": False})

    def migrate(self):
        self.prior("apply")
        stop = self.stopped_guard()
        need(sha(DB) == stop["production_db_sha256"], "stopped_db_drift")
        verify_files(LIVE, self.files)
        self.intent("migrate")
        result = self.offline(DB, LIVE)
        self.preserve()
        return self.receipt("migrate", {"migration": result, "production_db_sha256": sha(DB)})

    def start(self):
        migration = self.prior("migrate")
        self.prior("apply")
        self.stopped_guard()
        verify_files(LIVE, self.files)
        need(sha(DB) == migration["production_db_sha256"], "post_migration_db_drift")
        self.intent("start")
        self.run(["docker", "start", "hermes-1"], timeout=120)
        self.inspect()
        self.preserve()
        return self.receipt("start", {"postdeploy_verification_required": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "rehearse", "stop", "apply", "migrate", "start"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    need(not sys.flags.optimize, "optimized_execution_forbidden")
    need(os.geteuid() == 1000 and HOME.is_dir(), "afrodite_host_identity_required")
    cfg = read_sealed(args.config, args.config_sha256)
    rollout = Rollout(cfg, args.config_sha256)
    try:
        value = getattr(rollout, "stage_action" if args.action == "stage" else args.action)()
    except Exception:
        # No raw provider content, environment, paths, or traceback in stdout.
        if rollout.stage.is_dir() and not rollout.stage.is_symlink():
            fd = os.open(rollout.stage / (args.action + "-private-error.txt"), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as stream:
                traceback.print_exc(file=stream)
        print(json.dumps({"status": "failed", "action": args.action, "inspect_private_evidence": True}))
        return 1
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        code=str(exc) if type(exc) is RuntimeError and re.fullmatch('[a-z_]+',str(exc)) else type(exc).__name__
        print(json.dumps({"status": "failed", "preflight_rejected": True,"failure_code":code}))
        raise SystemExit(1)
