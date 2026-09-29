"""Fresh schema64 production gate. Preparation alone grants no launch authority.

The reviewed launch manifest and every receipt stay in a private host directory.
No remote operations, automatic retries, environment overrides or derived SQL
repairs are provided. Execute only after root independently seals the manifest.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import stat
import sys
import threading

CANDIDATE = "5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576"
SESSION_SHA = "8066d4f38539a067ff89c6cc7b130f38511e83af181321fe9efb58e8b38b698f"
TARGET = "chk_c14182771bd8e2a7e583f7d693cee29a5fb159fe"
GENERATION = "hymem-phase1-generation-v1:6075085e12e32e1b790e49b99b8c3bb50718b18be0f28762d1582c58ee8e35eb"
BOUNDS = {"completions": 128, "llm_http": 384, "embedding_http": 512,
          "total_http": 896, "deadline_seconds": 2700,
          "embedding_texts": 16, "embedding_chars": 128000,
          "embedding_utf8_bytes": 512000}
AUTHORIZATION = b"reviewed-schema64-production-targeted-dream-v1"


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private_file(path):
    info = Path(path).lstat()
    need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600,
         "private_regular_file_required")


def save(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    need(len(raw) <= 65536, "receipt_too_large")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(manifest_path, manifest_sha):
    private_file(manifest_path)
    need(sha(manifest_path) == manifest_sha, "reviewed_manifest_pin_drift")
    value = json.loads(Path(manifest_path).read_bytes())
    need(value.get("version") == "schema64-production-targeted-dream-v1"
         and value.get("root_reviewed") is True
         and value.get("candidate_sha256") == CANDIDATE
         and value.get("session_sha256") == SESSION_SHA
         and value.get("generation") == GENERATION
         and value.get("bounds") == BOUNDS, "launch_contract_invalid")
    stage = Path(value["stage"])
    need(stage.is_absolute() and stage.is_dir() and not stage.is_symlink()
         and stat.S_IMODE(stage.stat().st_mode) == 0o700, "private_stage_invalid")
    files = value["candidate_files"]
    need(len(files) == 481 and hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == CANDIDATE,
        "exact481_manifest_invalid")
    source = Path(value["source"])
    for relative, digest in files.items():
        path = Path(relative)
        need(not path.is_absolute() and ".." not in path.parts,
             "candidate_path_invalid")
        need(not (source / path).is_symlink() and sha(source / path) == digest,
             "candidate_source_drift")
    for label in ("worker", "meter", "supervisor"):
        item = value[label]
        private_file(item["path"])
        need(sha(item["path"]) == item["sha256"], "helper_pin_drift")
    need(Path(value["worker"]["path"]).resolve() == Path(__file__).resolve(),
         "worker_identity_changed")
    # Receipts are independently produced and then externally hash sealed.
    for label in ("full_suite", "private_paid_postflight", "deployment_start",
                  "production_preflight", "role_profiles"):
        item = value["gates"][label]
        private_file(item["path"])
        need(sha(item["path"]) == item["sha256"], "required_gate_receipt_drift")
        receipt = json.loads(Path(item["path"]).read_bytes())
        need(receipt.get("verified") is True
             and receipt.get("candidate_sha256") == CANDIDATE,
             "required_gate_not_verified")
    return value


def runtime_env(manifest):
    processes = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"hymem.honcho" in args or any(a.rsplit(b"/", 1)[-1] == b"hymem-honcho" for a in args):
            processes.append(proc)
    need(len(processes) == 1, "honcho_process_count")
    values = {}
    for item in (processes[0] / "environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            if key.startswith((b"HYMEM_", b"DEEPSEEK_", b"OPENAI_")):
                values[os.fsdecode(key)] = os.fsdecode(value)
    # The private seal includes secret values without disclosing them.
    digest = hashlib.sha256(json.dumps(values, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    need(digest == manifest["runtime_env_sha256"], "effective_runtime_drift")
    # Aggregation flags may differ legitimately by role and may be absent.
    # Their exact values/absence are already bound by the effective-env seal.
    need(values.get("HYMEM_LLM_MODEL") == "deepseek-flash"
         and "HYMEM_LLM_EXTRA_BODY" not in values,
         "production_role_policy_changed")
    return {"PATH": "/home/node/hymem-env/bin:/usr/bin:/bin", "HOME": "/home/node",
            "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", **values}


def idle_and_session(conn, meter):
    from hymem.core.db import schema_version
    need(schema_version(conn) == 64, "production_schema_changed")
    need(conn.execute("SELECT COUNT(*) FROM dream_runs WHERE ended_at IS NULL").fetchone()[0] == 0,
         "durable_dream_already_open")
    need(conn.execute("SELECT COUNT(*) FROM run_lock WHERE name='dreaming'").fetchone()[0] == 0,
         "production_dream_lease_present")
    session = meter.target_session(conn)
    need(hashlib.sha256(session.encode()).hexdigest() == SESSION_SHA, "target_session_changed")
    return session


class CounterJournal:
    """Reuse frozen meter admissions; discard all provider payload capture."""
    def append(self, _name, _value):
        pass


def worker(manifest_path, manifest_sha):
    need(sys.stdin.buffer.read() == AUTHORIZATION, "worker_authorization_missing")
    manifest = verify(manifest_path, manifest_sha)
    runtime = runtime_env(manifest)
    need(all(os.environ.get(k) == v for k, v in runtime.items()), "worker_runtime_changed")
    sys.path.insert(0, manifest["source"])
    meter = load_module("production_reviewed_meter", manifest["meter"]["path"])
    need(meter.TARGET_CHUNK == TARGET and meter.GENERATION == GENERATION
         and meter.MAX_COMPLETIONS == BOUNDS["completions"]
         and meter.MAX_LLM_HTTP_ATTEMPTS == BOUNDS["llm_http"]
         and meter.MAX_EMBEDDING_HTTP_ATTEMPTS == BOUNDS["embedding_http"]
         and meter.MAX_HTTP_ATTEMPTS == BOUNDS["total_http"]
         and meter.MAX_EMBEDDING_TEXTS == BOUNDS["embedding_texts"]
         and meter.MAX_EMBEDDING_CHARS == BOUNDS["embedding_chars"]
         and meter.MAX_EMBEDDING_UTF8_BYTES == BOUNDS["embedding_utf8_bytes"],
         "reviewed_meter_contract_changed")
    from hymem.bootstrap import build_from_env, shutdown_instance
    from hymem.deadline import MonotonicDeadline
    logging.disable(logging.CRITICAL)
    hy = probe = None
    result = {"status": "failed", "cleanup_ok": False,
              "independent_repair_postflight_required": True,
              "session_convergence_verified": False}
    try:
        hy = build_from_env()
        need(hy._phase1_generation["generation_key"] == GENERATION, "generation_changed")
        need(not hy.dream_status().get("in_progress"), "production_dream_already_active")
        session = idle_and_session(hy.conn, meter)
        previous = hy.conn.execute("SELECT COALESCE(MAX(id),0) FROM dream_runs").fetchone()[0]
        codes = meter.code_points()
        # No extraction/prepersist observers: never copy production private rows.
        probe = meter.Probe(CounterJournal(), Path("/unused"), completion_code=codes[0],
            attempt_code=codes[1], embedding_code=codes[2],
            extraction_code=None, persist_code=None)
        sys.setprofile(probe.profile)
        threading.setprofile(probe.profile)
        report = hy.dream(session_ids=[session], deadline=MonotonicDeadline.after(2700))
        sys.setprofile(None)
        threading.setprofile(None)
        runs = hy.conn.execute("SELECT id,ended_at,error,skipped_locked FROM dream_runs WHERE id>?", (previous,)).fetchall()
        owned = [row for row in runs if not row["skipped_locked"]]
        need(not report.skipped_locked and len(owned) == 1 and owned[0]["ended_at"]
             and owned[0]["error"] is None, "dream_ownership_or_result_unverified")
        need(not hy.conn.execute("SELECT 1 FROM run_lock WHERE name='dreaming'").fetchone(),
             "dream_lease_cleanup_unverified")
        result["owned_run_id"] = owned[0]["id"]
        result["report"] = {k: v for k, v in asdict(report).items() if v is None or type(v) in (int, float, bool)}
        # A bounded targeted dream can finish its owned run and repair episode
        # vectors while durable digest/profile/fact work remains. Completion is
        # invocation evidence; root's independent vector/publication postflight
        # must determine repair, and convergence is never inferred here.
        status = hy.dream_status()
        result["after"] = {key: status.get(key) for key in (
            "pending_chunks", "pending_digests", "pending_profiles", "pending_facts",
            "pending_aggregation", "summary_degraded_sessions", "summary_missing_sessions")}
        result["target_current_publications"] = hy.conn.execute(
            "SELECT COUNT(*) FROM current_phase1_publications WHERE chunk_id=? AND phase1_generation_key=?",
            (TARGET, GENERATION)).fetchone()[0]
        need(result["target_current_publications"] == 1, "target_publication_unverified")
        need(not probe.budget_reason and not probe.capture_error, "paid_budget_or_capture_failed")
        result["status"] = "completed"
    except BaseException:
        result["error_code"] = "worker_failed_inspect_before_retry"
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        if probe:
            result.update(completion_calls=probe.completions, http_attempts=probe.attempts,
                llm_http_attempts=probe.llm_attempts, embedding_http_attempts=probe.embedding_attempts,
                budget_reason=probe.budget_reason)
        if hy:
            try:
                result["accounting_verified"] = (probe is not None
                    and hy._llm.request_attempts == probe.llm_attempts
                    and hy._embed.request_attempts == probe.embedding_attempts)
                result["token_usage_available"] = hy._llm.token_usage_available is True
                if result["token_usage_available"]:
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        result[name] = getattr(hy._llm, name)
                need(result["accounting_verified"] and result["token_usage_available"], "accounting_unverified")
            except BaseException:
                result["status"] = "failed"
            try:
                result["cleanup_ok"] = shutdown_instance(hy) is True
            except BaseException:
                pass
        if not result["cleanup_ok"]:
            result["status"] = "failed"
        save(Path(manifest["stage"]) / "production-dream-result.json", result)
    return 0 if result["status"] == "completed" else 1


def supervise(manifest_path, manifest_sha):
    manifest = verify(manifest_path, manifest_sha)
    env = runtime_env(manifest)
    module = load_module("production_reviewed_supervisor", manifest["supervisor"]["path"])
    def cancelled(_signum, _frame):
        # Convert termination into the reviewed supervisor's exception cleanup
        # path, which owns and closes the worker process group before returning.
        raise KeyboardInterrupt
    previous_term = signal.signal(signal.SIGTERM, cancelled)
    try:
        outcome = module.supervise_invocation(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "worker",
             str(manifest_path), manifest_sha], cwd=manifest["source"], env=env,
            output_dir=Path(manifest["stage"]) / "production-dream-invocation-v1",
            timeout_seconds=2760, cleanup_seconds=10, stdin_bytes=AUTHORIZATION,
            output_limit_bytes=2 * 1024 * 1024)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    save(Path(manifest["stage"]) / "production-dream-supervisor.json", asdict(outcome))
    return 0 if outcome.status == "completed" and outcome.safe_to_continue else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "worker"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("manifest_sha256")
    args = parser.parse_args()
    try:
        rc = (worker if args.mode == "worker" else supervise)(args.manifest, args.manifest_sha256)
    except BaseException:
        print('{"status":"failed_inspect_before_retry"}')
        rc = 1
    raise SystemExit(rc)
