"""Independent offline verdict controls for the retained alias replay."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_alias_replay_host as host


@pytest.fixture
def replay_worker():
    path = Path(__file__).parents[1] / "claim_conflict_instrumented_replay.py"
    spec = importlib.util.spec_from_file_location("alias_replay_root_worker", path)
    worker = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        spec.loader.exec_module(worker)
    finally:
        sys.path[:] = original_path
    return worker


def _integrity():
    return {
        "integrity_ok": True, "foreign_key_findings": 0,
        "canonical_drift_findings": 0, "ledger_count_mismatches": 0,
        "same_generation_disagreeing_groups": 0,
    }


def _persisted(before="a", after="b"):
    return {
        "status": "persisted", "reason_code": None,
        "logical_digest_before": before * 64, "logical_digest_after": after * 64,
        "rollback_preserved": None, "pool_count_before": 0, "pool_count_after": 1,
    }


def _rejected(rollback=True):
    return {
        "status": "rejected", "reason_code": "alias_owned_state_guard",
        "rollback_preserved": rollback, "failure_captured": True,
        "logical_digest_before": "a" * 64, "logical_digest_after": "a" * 64,
        "error_type": "ValueError", "candidate_frames": [],
        "diagnostic_error_type": None, "pool_count_before": 0,
    }


def _run(monkeypatch, worker, tmp_path, first, second, *, before=None, after=None):
    from hymem.extraction.contract import extraction_cache_key

    cache_key = extraction_cache_key("fixture")
    class Conn:
        closed = False

        def execute(self, query, *args):
            if "current_phase1_publications" in query:
                assert args[0] == ("fixture-chunk", cache_key, worker.GENERATION)
                return SimpleNamespace(fetchone=lambda: (
                    int(bool(attempts) and first["status"] == "persisted"),
                ))
            if "phase1_generations" in query:
                return SimpleNamespace(fetchone=lambda: (cache_key,))
            return SimpleNamespace(fetchone=lambda: (1,))

        def close(self):
            self.closed = True

    conn = Conn()
    attempts = []
    integrity_results = iter([before or _integrity(), after or _integrity()])

    def attempt(_conn, _raw, *, dedup_enabled, label):
        attempts.append((dedup_enabled, label))
        return deepcopy(first if len(attempts) == 1 else second)

    monkeypatch.setattr(worker, "WORK", tmp_path)
    monkeypatch.setattr(worker, "clone", lambda *_: None)
    monkeypatch.setattr(worker, "open_clone", lambda *_: conn)
    monkeypatch.setattr(worker, "sha", lambda *_: "c" * 64)
    monkeypatch.setattr(worker, "reconstruct", lambda *a, **k: (
        SimpleNamespace(id="fixture-chunk", session_id="fixture-session"),
        SimpleNamespace(phase1_generation={"generation_key": worker.GENERATION,
                                           "extraction_cache_key": cache_key}),
        None, None, None,
    ))
    monkeypatch.setattr(worker, "integrity", lambda *_: next(integrity_results))
    monkeypatch.setattr(worker, "attempt", attempt)
    result = worker.run_arm({"prompt_version": "fixture", "extraction": {
        "phase1_generation": {"generation_key": worker.GENERATION,
                              "extraction_cache_key": cache_key}}},
                            tmp_path / "source.sqlite",
                            dedup_enabled=True, label="fixture")
    assert conn.closed
    return result, attempts


def test_run_arm_baseline_rejection_needs_rollback_and_does_not_repeat(monkeypatch, replay_worker, tmp_path):
    result, attempts = _run(monkeypatch, replay_worker, tmp_path, _rejected(), None)
    assert result["status"] == "completed"
    assert result["exact_repeat"] is None
    assert len(attempts) == 1


@pytest.mark.parametrize("failure", ["rollback", "execution", "repeat_rejected", "repeat_changed"])
def test_run_arm_rejects_incomplete_or_non_idempotent_evidence(monkeypatch, replay_worker, tmp_path, failure):
    first, second = _persisted(), _persisted("b", "b")
    if failure == "rollback":
        first, second = _rejected(False), None
    elif failure == "execution":
        first, second = {**_rejected(), "status": "execution_failure"}, None
    elif failure == "repeat_rejected":
        second = _rejected()
    else:
        second = _persisted("b", "d")
    result, _ = _run(monkeypatch, replay_worker, tmp_path, first, second)
    assert result["status"] == "inconclusive"


@pytest.mark.parametrize("which", ["before", "after"])
@pytest.mark.parametrize("finding", [
    "integrity_ok", "foreign_key_findings", "canonical_drift_findings",
    "ledger_count_mismatches", "same_generation_disagreeing_groups",
])
def test_run_arm_never_accepts_any_integrity_finding(monkeypatch, replay_worker, tmp_path, which, finding):
    bad = _integrity()
    bad[finding] = False if finding == "integrity_ok" else 1
    result, _ = _run(monkeypatch, replay_worker, tmp_path,
                     _persisted(), _persisted("b", "b"), **{which: bad})
    assert result["status"] == "inconclusive"


def test_run_arm_accepts_success_only_with_unchanged_exact_repeat(monkeypatch, replay_worker, tmp_path):
    result, attempts = _run(monkeypatch, replay_worker, tmp_path,
                            _persisted(), _persisted("b", "b"))
    assert result["status"] == "completed"
    assert result["exact_repeat_unchanged"] is True
    assert attempts == [(True, "fixture-first"), (True, "fixture-exact-repeat")]


def _metadata(mode):
    baseline = mode == "baseline"
    arms = {}
    for name in ("dedup_on", "dedup_off"):
        arms[name] = {
            "status": "completed", "dedup_enabled": name == "dedup_on",
            "first": _rejected() if baseline else _persisted(),
            "exact_repeat": None if baseline else _persisted("b", "b"),
            "exact_repeat_unchanged": None if baseline else True,
            "published_before": 0, "published_after_first": 0 if baseline else 1,
            "published_after_repeat": None if baseline else 1,
            "integrity_before": _integrity(), "integrity_after": _integrity(),
            "database_sha256": "c" * 64,
        }
    return {
        "status": "replayed", "capture_index": 1,
        "capture_sha256": "d" * 64, "snapshot_sha256": "e" * 64,
        "phase1_sha256": host.PHASE1_SHA, **arms,
    }


def _receipt():
    return {"capture_sha256": "d" * 64, "snapshot_sha256": "e" * 64}


@pytest.mark.parametrize("mode", ["baseline", "fixed"])
def test_host_verdict_accepts_only_expected_comparison_shapes(mode):
    host.verdict(mode, host.project(_metadata(mode)), _receipt())


@pytest.mark.parametrize("mutation", [
    "wrong_reason", "rollback_missing", "failure_not_captured", "publication_created",
])
def test_host_cannot_accept_unproven_baseline_rejection(mutation):
    raw = _metadata("baseline")
    arm = raw["dedup_on"]
    if mutation == "wrong_reason":
        arm["first"]["reason_code"] = "same_generation_observation_disagreement"
    elif mutation == "rollback_missing":
        arm["first"]["rollback_preserved"] = False
    elif mutation == "failure_not_captured":
        arm["first"]["failure_captured"] = False
    else:
        arm["published_after_first"] = 1
    with pytest.raises(RuntimeError, match="baseline_not_alias_guard_rollback"):
        host.verdict("baseline", host.project(raw), _receipt())


@pytest.mark.parametrize("mutation", [
    "first_noop", "missing_publication", "already_published", "repeat_changed",
    "repeat_disconnected", "repeat_rejected",
])
def test_host_cannot_accept_false_fixed_success(mutation):
    raw = _metadata("fixed")
    arm = raw["dedup_off"]
    if mutation == "first_noop":
        arm["first"]["logical_digest_before"] = arm["first"]["logical_digest_after"]
    elif mutation == "missing_publication":
        arm["published_after_first"] = 0
    elif mutation == "already_published":
        arm["published_before"] = 1
    elif mutation == "repeat_changed":
        arm["exact_repeat"]["logical_digest_after"] = "f" * 64
    elif mutation == "repeat_disconnected":
        arm["exact_repeat"] = _persisted("c", "c")
    else:
        arm["exact_repeat"] = _rejected()
    with pytest.raises(RuntimeError, match="fixed_not_idempotent_publication"):
        host.verdict("fixed", host.project(raw), _receipt())


@pytest.mark.parametrize("field", ["capture_sha256", "snapshot_sha256", "phase1_sha256"])
def test_host_rejects_every_input_hash_mismatch(field):
    raw = _metadata("fixed")
    raw[field] = "0" * 64
    with pytest.raises(RuntimeError, match="replay_input_identity_drift"):
        host.verdict("fixed", host.project(raw), _receipt())


@pytest.mark.parametrize("mode", ["baseline", "fixed"])
def test_host_never_accepts_missing_logical_digest_as_comparison_evidence(mode):
    raw = _metadata(mode)
    raw["dedup_on"]["first"]["logical_digest_before"] = None
    if mode == "baseline":
        raw["dedup_on"]["first"]["logical_digest_after"] = None
    with pytest.raises(RuntimeError):
        host.verdict(mode, host.project(raw), _receipt())


def test_host_capture_index_is_an_integer_not_boolean():
    raw = _metadata("baseline")
    raw["capture_index"] = True
    with pytest.raises(RuntimeError):
        host.project(raw)


def test_host_export_contains_no_private_payload():
    raw = _metadata("baseline")
    raw["private_input"] = "PRIVATE SOURCE SENTINEL"
    raw["dedup_on"]["first"]["traceback"] = "PRIVATE SOURCE SENTINEL"
    raw["dedup_off"]["private_exception"] = "PRIVATE SOURCE SENTINEL"
    assert "PRIVATE SOURCE SENTINEL" not in repr(host.project(raw))


@pytest.mark.parametrize("mode", ["baseline", "fixed"])
def test_each_arm_is_networkless_without_runtime_credentials_or_live_store(mode):
    helper = SimpleNamespace(RUNTIME=Path("/approved/runtime"), IMAGE="sha256:" + "1" * 64)
    command, mounts = host.configure(helper, mode)
    assert command[command.index("--network") + 1] == "none"
    assert command[-2:] == ["--capture-index", "1"]
    assert [destination for _, destination, rw in mounts if rw] == ["/work"]
    assert {destination for _, destination, _ in mounts} == {
        "/candidate", "/diag/claim_conflict_instrumented_replay.py", "/capture",
        "/work", "/home/node/hymem-env",
    }
    assert (str(host.CAPTURE), "/capture", False) in mounts
    assert "--read-only" in command
