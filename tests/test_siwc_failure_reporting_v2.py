"""Offline controls for the frozen SIWC checkpoint failure vocabulary."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.diagnostics import siwc_lme_diagnostic_progress_v4 as progress


ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle")


def test_versioned_source_identity_is_explicit() -> None:
    import hashlib

    runner = ROOT / progress.RUNNER_RELATIVE
    assert runner.is_file()
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == progress.RUNNER_SHA256
    assert progress.RUN_SCHEMA == "siwc-lme-semantic-diagnostic-v2"
    assert progress.RUNNER_RELATIVE.endswith("siwc_lme_diagnostic_v2.py")


def test_frozen_worker_and_checkpoint_projection(tmp_path, monkeypatch) -> None:
    # A fresh interpreter imports the exact frozen candidate, without allowing
    # pytest's local package imports to substitute the workspace's stale v5.
    script = r'''
import importlib.util
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

repo = pathlib.Path(sys.argv[1])
frozen = pathlib.Path(sys.argv[2])
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

old = module("offline_old_runner", repo / "tools/diagnostics/siwc_lme_diagnostic_v1.py")
new = module("offline_new_runner", repo / "tools/diagnostics/siwc_lme_diagnostic_v2.py")
reader = module("offline_new_reader", repo / "tools/diagnostics/siwc_lme_diagnostic_progress_v4.py")
loaded = new.import_source_only(frozen, frozen / "source-map.json",
    new.ACCEPTED_INVENTORY_SHA256)
lme, strictness, protocol = loaded["lme"], loaded["strictness"], loaded["protocol"]
raw = {"cycles": 0, "max_cycles": 100, "timeout_s": 10800.0,
    "elapsed_s": 10800.0, "complete": False, "healthy": False,
    "failure_reason": "timeout_during_cycle", "reports": [],
    "final_status": {}, "quarantined": {}}
canonical = lme.canonicalize_lme_indexing_summary(raw)
assert protocol._validate_versioned_indexing(canonical,
    require_healthy=True, allow_failure=True) is False
assert strictness.bounded_failure_text("indexing_rejected") == "unspecified_failure"
assert strictness.bounded_failure_text("indexing_failure:timeout_during_cycle") == "indexing_failure:timeout_during_cycle"
assert reader.INDEXING_CODES == (
    strictness._INDEXING_FAILURE_CODES & protocol._INDEXING_FAILURE_CODES)
assert reader.WORKER_EXCEPTION_TYPES == new.WORKER_EXCEPTION_TYPES
for code in reader.INDEXING_CODES:
    projected = "indexing_failure:" + code
    assert strictness.bounded_failure_text(projected) == projected
for name in reader.WORKER_EXCEPTION_TYPES:
    projected = "worker_failure:" + name
    assert strictness.bounded_failure_text(projected) == projected
assert strictness.bounded_failure_text("indexing_failure:phase1_producer_unavailable") == "unspecified_failure"

class Budget:
    def __init__(self): self.halts = []
    def halt(self, code): self.halts.append(code)
class Client:
    usage_complete = True
    def __init__(self): self.closed = False
    def close(self): self.closed = True
class Accounted:
    counts = {"reader": 0}
    instances = []
    result = True
    def __init__(self):
        self.reconciled = False
        self.instances.append(self)
    def reconcile(self): self.reconciled = True; return self.result
class Observer:
    def __init__(self): self.captured = []
    def capture_pair(self, key): self.captured.append(key)
class Adapter:
    summary = canonical
    instances = []
    diagnostic_indexing = {"admitted": True}
    def __init__(self, *args, **kwargs):
        self.last_indexing_summary = self.summary
        self.closed = False
        self.instances.append(self)
    def open(self): pass
    def close(self): self.closed = True

saved = []
clients = []
def make_dual(*args):
    client = Client(); clients.append(client); return client
def atomic_private(path, value):
    saved.append((path.name, value))
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))

loaded["candidate"] = frozen / "candidate"
loaded["prior"].old.ChatBridge = lambda *args: object()
loaded["prior"].old.make_adapter_class = lambda *args: object()
loaded["prior"].atomic_private = atomic_private
loaded["diagnostic"].make_diagnostic_adapter_class = lambda *args: Adapter
for runner in (old, new):
    runner.make_dual = make_dual
    runner.AccountedClient = lambda *args: Accounted()
    runner._memory_client = lambda *args: object()

exc = lme.IndexingConvergenceError("secret should not escape", {"unsafe": "secret"})
def raise_failure(*args, **kwargs): raise exc
lme.evaluate_question = raise_failure

with tempfile.TemporaryDirectory() as temp:
    root = pathlib.Path(temp)
    template = json.loads(sys.stdin.read())
    vocabulary_manifest = {"schema": "offline", "expected_count": 1,
        "expected_ids_hash": strictness.content_hash(["q0"]),
        "scored_run": True}
    vocabulary_manifest["run_id"] = strictness.content_hash(vocabulary_manifest)
    for index, code in enumerate(sorted(reader.QUESTION_FAILURE_CODES)):
        with strictness.AtomicCheckpoint(root / ("vocabulary-%02d.json" % index),
                manifest=vocabulary_manifest, expected_ids=["q0"],
                scored=True) as checkpoint:
            checkpoint.record("q0", row=None, failure=code)
            assert checkpoint.finalize()["entries"]["q0"]["failure"] == code
    def run(runner, summary, failure):
        Adapter.summary = summary
        budget, observer = Budget(), Observer()
        output = root / ("old" if runner is old else "new") / failure
        output.mkdir(parents=True)
        result = runner._question_worker(loaded, budget, None,
            {"question_id": "q0"}, 0, output, 10800, observer)
        assert budget.halts == ["question_failure"]
        assert observer.captured == ["question.0"]
        assert Adapter.instances[-1].closed and clients[-1].closed
        assert (output / "q-0000" / "private-indexing.json").is_file()
        assert result["projection"] is None and result["accounting"] is None
        code = result["stop_code"]
        manifest = {"schema": "offline", "expected_count": 1,
            "expected_ids_hash": strictness.content_hash(["q0"]),
            "scored_run": True}
        manifest["run_id"] = strictness.content_hash(manifest)
        with strictness.AtomicCheckpoint(output / "checkpoint.json",
                manifest=manifest, expected_ids=["q0"], scored=True) as checkpoint:
            checkpoint.record("q0", row=None, failure=code)
            state = checkpoint.finalize()
        durable = state["entries"]["q0"]["failure"]
        assert "secret" not in json.dumps(state)
        return code, durable

    old_code, old_durable = run(old, canonical, "indexing")
    assert old_code == "indexing_rejected" and old_durable == "unspecified_failure"
    new_code, new_durable = run(new, canonical, "indexing")
    assert new_code == new_durable == "indexing_failure:timeout_during_cycle"

    malformed = dict(canonical)
    malformed["failure"] = {"code": "unsafe_secret", "exception_type": None}
    code, durable = run(new, malformed, "malformed")
    assert code == durable == "worker_failure:IndexingConvergenceError"
    malformed = dict(canonical)
    malformed["cycles"] = -1
    code, durable = run(new, malformed, "badshape")
    assert code == durable == "worker_failure:IndexingConvergenceError"

    lme.evaluate_question = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("private path and secret"))
    code, durable = run(new, canonical, "valueerror")
    assert code == durable == "worker_failure:ValueError"
    class PrivateFault(Exception): pass
    lme.evaluate_question = lambda *args, **kwargs: (_ for _ in ()).throw(
        PrivateFault("private path and secret"))
    code, durable = run(new, canonical, "unknown")
    assert code == durable == "worker_failure:Exception"

    # The unchanged success and accounting paths still reconcile the ledger,
    # capture the observer, and close both resources.
    lme.evaluate_question = lambda *args, **kwargs: {"question_id": "q0"}
    new.validate_diagnostic_row = lambda *args: {"question_id": "q0", "correct": True}
    for label, reconciles in (("success", True), ("accounting", False)):
        Accounted.result = reconciles
        budget, observer = Budget(), Observer()
        output = root / "new" / label
        output.mkdir(parents=True)
        result = new._question_worker(loaded, budget, None,
            {"question_id": "q0"}, 0, output, 10800, observer)
        assert Accounted.instances[-1].reconciled
        assert Adapter.instances[-1].closed and clients[-1].closed
        assert observer.captured == ["question.0"]
        if reconciles:
            assert result["projection"]["correct"] is True
            assert result["accounting"] == {"reader": 0}
            assert result["stop_code"] is None and budget.halts == []
        else:
            assert result["stop_code"] == "worker_failure:RuntimeError"
            assert budget.halts == ["question_failure"]
    coordinator = {}
    for label in ("worker_runtime_failure", "not_started_after_campaign_stop"):
        path = root / ("coordinator-" + label + ".json")
        with strictness.AtomicCheckpoint(path, manifest=template["manifest"],
                expected_ids=template["expected_ids"], scored=True) as checkpoint:
            checkpoint.record(template["expected_ids"][0], row=None, failure=label)
            running = json.loads(path.read_text())
            for qid in template["expected_ids"][1:]:
                checkpoint.record(qid, row=None, failure="not_started_after_campaign_stop")
            terminal = checkpoint.finalize()
        assert running["entries"]["q0"]["failure"] == "unspecified_failure"
        assert terminal["entries"]["q0"]["failure"] == "unspecified_failure"
        coordinator[label] = {"running": running, "terminal": terminal}
    mixed_path = root / "mixed-running.json"
    with strictness.AtomicCheckpoint(mixed_path, manifest=template["manifest"],
            expected_ids=template["expected_ids"], scored=True) as checkpoint:
        checkpoint.record("q0", row=None, failure="worker_runtime_failure")
        checkpoint.record("q1", row={"question_id": "q1", "correct": True,
            "benchmark_failure": None, "diagnostic_kind": "strict_healthy",
            "strict_indexing_healthy": True, "quarantined_chunks": 0,
            "summary_degraded_sessions": 0, "context_sha": "0" * 64})
        mixed_running = json.loads(mixed_path.read_text())
print(json.dumps({"status": "frozen_worker_checkpoint_ok",
    "coordinator": coordinator, "mixed_running": mixed_running}))
'''
    template, receipt = _checkpoint("unspecified_failure")
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", script,
        str(ROOT), str(FROZEN)], input=json.dumps(template), text=True,
        capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stderr[-3000:]
    observed = json.loads(completed.stdout)
    assert observed["status"] == "frozen_worker_checkpoint_ok"
    for states in observed["coordinator"].values():
        for state in states.values():
            if state["status"] == "complete":
                assert set(state["counts"]) == progress.COUNT_FIELDS, state["counts"]
                assert state["counts"] == {"expected": 4, "attempted": 4,
                    "unique_attempted": 4, "total_attempts": 4,
                    "completed": 0, "failed": 4, "missing": 0}, state["counts"]
            counts, completed_count, _, _, _ = progress.checkpoint(state, receipt)
            assert completed_count == 0
            assert counts is None or counts["failed"] == 4

    # Drive inspect through real frozen checkpoint bytes. Only receipt and
    # runtime identities are mocked; the checkpoint validator and projection
    # are the reader's production path.
    root = tmp_path / ".hymem-siwc-lme-diagnostic-offlinefixture"
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    receipt_sha = "a" * 64
    (root / "launch-attempt.json").write_text(json.dumps({
        "receipt_sha256": receipt_sha, "one_shot": True}))
    (root / progress.EXECUTION_MARKER).write_bytes(json.dumps({
        "receipt_sha256": receipt_sha, "execution_started": True},
        sort_keys=True, separators=(",", ":")).encode("ascii"))
    checkpoint_path = run_dir / "diagnostic-checkpoint.json"
    monkeypatch.setattr(progress, "checked_root", lambda path: path)
    monkeypatch.setattr(progress, "receipt", lambda path, digest:
        receipt | {"expected_cgroup": "/offline"})
    monkeypatch.setattr(progress, "live_resource", lambda group: {"denials": 0})
    states = observed["coordinator"]["worker_runtime_failure"]
    checkpoint_path.write_text(json.dumps(states["running"]))
    monkeypatch.setattr(progress, "runtime", lambda pinned: ("running_verified", False))
    running = progress.inspect(root, receipt_sha)
    assert running["status"] == "checkpoint_running"
    assert running["question_failure_codes"] == ["unspecified_failure", None, None, None]
    assert "q0" not in json.dumps(running)

    checkpoint_path.write_text(json.dumps(observed["mixed_running"]))
    mixed = progress.inspect(root, receipt_sha)
    assert observed["mixed_running"]["entries"]["q1"]["status"] == "completed"
    assert "q2" not in observed["mixed_running"]["entries"]
    assert mixed["question_failure_codes"] == ["unspecified_failure", None, None, None]

    checkpoint_path.write_text(json.dumps(states["terminal"]))
    monkeypatch.setattr(progress, "runtime", lambda pinned: ("failed_exit_cleaned", True))
    terminal = progress.inspect(root, receipt_sha)
    assert terminal["status"] == "terminal_failure_without_result"
    assert terminal["runtime_cleanup_verified"] is True
    assert terminal["question_failure_codes"] == ["unspecified_failure"] * 4
    assert "q0" not in json.dumps(terminal)

    unsafe = json.loads(json.dumps(states["running"]))
    unsafe["entries"]["q0"]["failure"] = "worker_failure:PrivateFault"
    unsafe["entries"]["q0"]["row"]["benchmark_failure"] = "worker_failure:PrivateFault"
    checkpoint_path.write_text(json.dumps(unsafe))
    with pytest.raises(ValueError, match="checkpoint_failure_invalid"):
        progress.inspect(root, receipt_sha)


def _checkpoint(failure: str) -> tuple[dict, dict]:
    import hashlib
    import json

    ids = [f"q{i}" for i in range(4)]
    canonical = lambda value: "sha256:" + hashlib.sha256(json.dumps(value,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {"selected_row_sha256": ["0" * 64] * 4,
        "source_sha256": {"benchmarks/chatgpt_plan_responses_v6.py": "1" * 64,
                          "benchmarks/chatgpt_plan_lme_v1.py": "2" * 64}}
    manifest = {"schema": progress.RUN_SCHEMA, "mode": progress.MODE,
        "canonical_r9_artifact": False, "official_model_score": False,
        "candidate_map_sha256": progress.MAP_SHA256,
        "dataset_sha256": progress.DATASET_SHA256,
        "selected_row_sha256": receipt["selected_row_sha256"],
        "selected_source_order": "first_n", "expected_count": 4,
        "expected_ids_hash": canonical(ids), "scored_run": True,
        "diagnostic_helper_sha256": progress.HELPER_SHA256,
        "billing_policy": progress.BILLING_POLICY,
        "runner_sha256": progress.RUNNER_SHA256,
        "transport_sha256": "1" * 64, "bridge_sha256": "2" * 64,
        "grant_identity_sha256": progress.GRANT_IDENTITY_SHA256,
        "rerolls": 0, "limits": {name: dict(zip(
            ("turns", "known_tokens", "seconds"), values))
            for name, values in progress.LIMITS.items()}
            | {"indexing_seconds": 10800, "workers": 4}}
    manifest["run_id"] = canonical(manifest)
    entry = {"status": "failed", "attempts": 1, "failure": failure,
        "row": {"question_id": ids[0], "correct": False,
                "benchmark_failure": failure}}
    value = {"schema": progress.CHECKPOINT_SCHEMA, "status": "complete",
        "scored": True, "verdict_key": "correct", "manifest": manifest,
        "expected_ids": ids, "entries": {ids[0]: entry},
        "run_id": manifest["run_id"],
        "counts": {"expected": 4, "attempted": 1, "unique_attempted": 1,
                   "total_attempts": 1, "completed": 0, "failed": 1,
                   "missing": 3}}
    return value, receipt


@pytest.mark.parametrize("code", [
    "indexing_failure:timeout_during_cycle", "worker_failure:ValueError",
    "worker_failure:Exception", "unspecified_failure",
])
def test_new_reader_accepts_exact_question_codes(code: str) -> None:
    value, receipt = _checkpoint(code)
    counts, completed, _, _, _ = progress.checkpoint(value, receipt)
    assert counts["failed"] == 1 and completed == 0


@pytest.mark.parametrize("code", [
    "indexing_rejected", "question_failure", "worker_runtime_failure",
    "not_started_after_campaign_stop",
    "worker_failure:PrivateFault", "indexing_failure:private_secret",
    "worker_failure:ValueError: raw private text",
])
def test_new_reader_rejects_legacy_or_untrusted_question_codes(code: str) -> None:
    value, receipt = _checkpoint(code)
    with pytest.raises(ValueError, match="checkpoint_failure_invalid"):
        progress.checkpoint(value, receipt)
