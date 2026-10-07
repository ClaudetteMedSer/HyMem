"""Offline controls for the once-only recovery check; no host or provider work."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/siwc_lme_recovery_check_v1.py"
spec = importlib.util.spec_from_file_location("recovery_v1_offline", PATH)
assert spec and spec.loader
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class FakeBudget:
    latest = None

    def __init__(self, limits, *, max_in_flight):
        assert limits == (1, 160000, 180) and max_in_flight == 1
        self.turns = 0
        self.tokens = 0
        self.complete = True
        self.stop = None
        FakeBudget.latest = self

    def snapshot(self):
        return {"turns": self.turns, "known_tokens": self.tokens,
                "usage_complete": self.complete, "in_flight": 0,
                "reserved": 0, "stop_code": self.stop}


class FakeBroker:
    calls = 0

    def __init__(self, state, runtime):
        assert str(state) == recovery.OWNER and str(runtime) == recovery.RUNTIME
        FakeBroker.calls += 1
        self.identity_digest = recovery.GRANT_SHA

    def close(self):
        pass


class FakeClient:
    mode = "success"
    calls_total = 0
    last = None

    def __init__(self, broker, budget, question_id, limits):
        assert type(broker) is FakeBroker and question_id == "recovery"
        assert limits == (1, 160000, 180)
        self.budget = budget
        self.calls = self.successes = self.failures = self.http_attempts = 0
        self.first_failure = None
        FakeClient.last = self

    def complete(self, request):
        FakeClient.calls_total += 1
        self.calls += 1
        self.http_attempts += 1
        self.budget.turns += 1
        assert request.system == recovery.SYSTEM and request.user == recovery.USER
        assert request.response_format == "text" and request.max_tokens == 64
        if self.mode == "success":
            self.successes = 1
            self.budget.tokens = 13
            return recovery.EXPECTED
        self.failures = 1
        self.budget.complete = False
        code, status = {"503": ("subscription_sharing_user_unavailable", 503),
                        "403": ("chatpass_v2_scope_not_authorized", 403),
                        "429": ("subscription_sharing_usage_limit_exceeded", 429)}[self.mode]
        self.first_failure = {"code": code, "phase": "http", "turn_admitted": True,
                              "unknown_usage": True, "http_status": status,
                              "wire_observation": {"ignored": "private"}}
        self.budget.stop = code
        raise RuntimeError("private provider body must not leave host")

    def diagnostic_summary(self):
        return {"calls": self.calls, "successes": self.successes,
                "failures": self.failures, "internal_http_attempts": self.http_attempts,
                "first_failure": self.first_failure}

    def close(self):
        pass


def fake_loaded():
    return {"siwc": SimpleNamespace(SharedBudget=FakeBudget,
                                      SIWCLMEClient=FakeClient,
                                      _CODES=frozenset({"subscription_sharing_user_unavailable",
                                          "chatpass_v2_scope_not_authorized",
                                          "subscription_sharing_usage_limit_exceeded"}),
                                      owner=SimpleNamespace(CredentialBroker=FakeBroker)),
            "warm": SimpleNamespace(BudgetLimits=lambda *args: args),
            "request_type": lambda **kwargs: SimpleNamespace(**kwargs)}


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeBroker.calls = 0
    FakeClient.calls_total = 0
    FakeClient.mode = "success"
    FakeClient.last = None


def run_execute(monkeypatch, tmp_path, mode):
    root = tmp_path / ".hymem-siwc-lme-diagnostic-recovery-abcdefgh"
    root.mkdir()
    digest = "a" * 64
    FakeClient.mode = mode
    written = {}
    monkeypatch.setattr(recovery, "root_checked", lambda path: path)
    monkeypatch.setattr(recovery, "receipt_checked", lambda path, value: {"unit": "u.service"})
    monkeypatch.setattr(recovery, "read_private", lambda path, cap: {"receipt_sha256": digest,
                                                                      "one_shot": True})
    monkeypatch.setattr(recovery, "exact_private", lambda *args: None)
    monkeypatch.setattr(recovery, "execute_containment", lambda receipt: tmp_path)
    monkeypatch.setattr(recovery, "source_only", lambda path: (None, fake_loaded()))
    monkeypatch.setattr(recovery, "failure_codes_readonly", lambda path:
                        fake_loaded()["siwc"]._CODES)
    monkeypatch.setattr(recovery, "resource", lambda group:
                        {"peak": 4, "denials": 0, "oom": 0, "oom_kill": 0})
    monkeypatch.setattr(recovery, "write_once", lambda path, value, cap=8192:
                        written.setdefault(path.name, value))
    result = recovery.execute(root, digest)
    assert FakeClient.calls_total == 1
    assert FakeBroker.calls == 1
    assert written["recovery-execution.json"] == {"receipt_sha256": digest,
                                                   "execution_started": True}
    return result, written["recovery-result.json"]


def test_one_success_has_complete_positive_usage(monkeypatch, tmp_path, capsys):
    code, result = run_execute(monkeypatch, tmp_path, "success")
    assert code == 0 and result["check_passed"] is True
    assert result["calls"] == result["successes"] == result["admitted_turns"] == 1
    assert result["known_tokens"] == 13 and result["usage_complete"] is True
    assert recovery.EXPECTED not in json.dumps(result)
    assert "private" not in capsys.readouterr().out


@pytest.mark.parametrize("mode,status", [("503", 503), ("403", 403), ("429", 429)])
def test_failed_admitted_turn_stops_with_unknown_usage(monkeypatch, tmp_path, mode, status, capsys):
    code, result = run_execute(monkeypatch, tmp_path, mode)
    assert code == 1 and result["check_passed"] is False
    assert result["admitted_turns"] == 1 and result["known_tokens"] == 0
    assert result["usage_complete"] is False
    assert result["first_failure"]["unknown_usage"] is True
    assert result["first_failure"]["http_status"] == status
    assert "private" not in json.dumps(result) + capsys.readouterr().out


def test_receipt_checks_self_and_source_before_reader(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    sources = {recovery.SELF: "x" * 64}
    expected = recovery.receipt_for(root, sources)
    path = root / "recovery-receipt.json"
    path.write_bytes(recovery.canonical(expected))
    monkeypatch.setattr(recovery, "UID", path.stat().st_uid)
    path.chmod(0o600)
    monkeypatch.setattr(recovery, "identity_files", lambda _: sources)
    called = []
    monkeypatch.setattr(recovery, "source_tree_readonly", lambda _: called.append(True))
    digest = recovery.sha(path)
    assert recovery.receipt_checked(root, digest) == expected and called == [True]
    with pytest.raises(ValueError, match="receipt_pin_invalid"):
        recovery.receipt_checked(root, "0" * 64)
    sources[recovery.SELF] = "y" * 64
    with pytest.raises(ValueError, match="receipt_identity_invalid"):
        recovery.receipt_checked(root, digest)


def test_inspection_does_not_load_model_or_owner(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(recovery, "root_checked", lambda path: path)
    monkeypatch.setattr(recovery, "receipt_checked", lambda path, digest:
                        recovery.receipt_for(root, {recovery.SELF: "a" * 64}))
    monkeypatch.setattr(recovery, "failure_codes_readonly", lambda path: frozenset())
    monkeypatch.setattr(recovery, "load_pinned", lambda *args: pytest.fail("model import"))
    monkeypatch.setattr(recovery, "source_only", lambda *args: pytest.fail("owner access"))
    result = recovery.inspect(root, "a" * 64)
    assert result["status"] == "prepared_not_launched"
    assert result["recovery_verified"] is False


def test_launch_consumes_attempt_before_ambiguous_dispatch(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "empty").mkdir()
    (root / "tmp").mkdir()
    receipt = recovery.receipt_for(root, {recovery.SELF: "a" * 64})
    digest = "a" * 64
    monkeypatch.setattr(recovery, "root_checked", lambda path: path)
    monkeypatch.setattr(recovery, "receipt_checked", lambda path, value: receipt)
    monkeypatch.setattr(recovery, "load_pinned_after_identity", lambda path:
                        SimpleNamespace(host_admission=lambda: None))
    monkeypatch.setattr(recovery, "source_only", lambda path: None)
    monkeypatch.setattr(recovery, "bus_env", lambda: {})
    def ambiguous(*args, **kwargs):
        assert (root / "recovery-attempt.json").exists()
        raise TimeoutError("ambiguous")
    monkeypatch.setattr(recovery.subprocess, "run", ambiguous)
    result = recovery.launch(root, digest)
    assert result["never_retry"] is True and result["launch_command_returncode"] == -1
    with pytest.raises(ValueError, match="already_attempted"):
        recovery.launch(root, digest)


def test_finite_failure_projection_rejects_private_or_unbounded_text():
    allowed = frozenset({"subscription_sharing_user_unavailable"})
    failure = {"code": "subscription_sharing_user_unavailable", "phase": "http",
               "turn_admitted": True, "unknown_usage": True, "http_status": 503}
    assert recovery.safe_failure(failure, allowed) == failure["code"]
    with pytest.raises(ValueError, match="failure_metadata_invalid"):
        recovery.safe_failure({**failure, "raw_body": "private"}, allowed)
    with pytest.raises(ValueError, match="failure_metadata_invalid"):
        recovery.safe_failure({**failure, "code": "x" * 1000}, allowed)
    with pytest.raises(ValueError, match="failure_metadata_invalid"):
        recovery.safe_failure({**failure, "code": "invented_safe_looking_code"}, allowed)


def test_resource_fault_blocks_model_admission(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    digest = "a" * 64
    monkeypatch.setattr(recovery, "root_checked", lambda path: path)
    monkeypatch.setattr(recovery, "receipt_checked", lambda path, value: {})
    monkeypatch.setattr(recovery, "read_private", lambda path, cap:
                        {"receipt_sha256": digest, "one_shot": True})
    monkeypatch.setattr(recovery, "execute_containment", lambda receipt: tmp_path)
    monkeypatch.setattr(recovery, "source_only", lambda path: (None, fake_loaded()))
    monkeypatch.setattr(recovery, "failure_codes_readonly", lambda path:
                        fake_loaded()["siwc"]._CODES)
    monkeypatch.setattr(recovery, "resource", lambda group:
                        {"peak": 4, "denials": 1, "oom": 0, "oom_kill": 0})
    monkeypatch.setattr(recovery, "exact_private", lambda *args: None)
    monkeypatch.setattr(recovery, "write_once", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="precall_resource_fault"):
        recovery.execute(root, digest)
    assert FakeClient.calls_total == FakeBroker.calls == 0


def test_reader_keeps_finite_overshoot_failure_and_rejects_bool_counts(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "recovery-attempt.json").write_bytes(recovery.canonical(
        {"receipt_sha256": "a" * 64, "one_shot": True}))
    (root / "recovery-execution.json").write_bytes(recovery.canonical(
        {"receipt_sha256": "a" * 64, "execution_started": True}))
    (root / "recovery-result.json").write_text("{}")
    digest = "a" * 64
    result = {"schema": recovery.SCHEMA, "receipt_sha256": digest,
              "response_match": True, "calls": 1, "successes": 1, "failures": 0,
              "http_attempts": 1, "admitted_turns": 1, "known_tokens": 200000,
              "usage_complete": True, "in_flight": 0, "reserved": 0,
              "first_failure": None, "stop_code": None,
              "resource": {"peak": 4, "denials": 0, "oom": 0, "oom_kill": 0},
              "check_passed": False}
    def read(path, cap):
        if path.name == "recovery-attempt.json":
            return {"receipt_sha256": digest, "one_shot": True}
        if path.name == "recovery-execution.json":
            return {"receipt_sha256": digest, "execution_started": True}
        return result
    monkeypatch.setattr(recovery, "root_checked", lambda path: path)
    monkeypatch.setattr(recovery, "receipt_checked", lambda path, value:
                        recovery.receipt_for(root, {recovery.SELF: "a" * 64}))
    monkeypatch.setattr(recovery, "failure_codes_readonly", lambda path: frozenset())
    monkeypatch.setattr(recovery, "read_private", read)
    monkeypatch.setattr(recovery, "exact_private", lambda *args: None)
    monkeypatch.setattr(recovery, "service_state", lambda receipt: ("terminal", True, True))
    projection = recovery.inspect(root, digest)
    assert projection["status"] == "terminal_check_failed"
    assert projection["known_tokens"] == 200000 and not projection["recovery_verified"]
    result["admitted_turns"] = True
    with pytest.raises(ValueError, match="result_count_invalid"):
        recovery.inspect(root, digest)
