"""Independent no-inference access-check accounting and privacy controls."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_staged_v3 as staged
from hymem.extraction.llm import LLMRequest
from tools.diagnostics import luna_subscription_access_check_v1 as check


PRIVATE = "INVENTED_PRIVATE_PROVIDER_OUTPUT"


@pytest.mark.parametrize("failure", [True, False])
def test_first_failure_cannot_dispatch_second_turn_or_export_private_text(monkeypatch, tmp_path, failure):
    warm = staged.warm
    calls = []
    def complete(self, request, *args, **kwargs):
        calls.append(self.question_id)
        self.budget.reserve(self.question_id)
        self.budget.before_turn(self.question_id, {"auth": "chatgpt", "model": "gpt-6-luna",
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": [{"remaining_percent": 75}]})
        self.budget.settle(self.question_id, used=None if failure else 42,
            turn_started=True, failure="usage_unknown" if failure else None)
        self.requested_controls.append({"output_schema_sent": bool(args),
            "output_schema_acknowledged": bool(args)})
        if failure:
            raise warm.ConcurrentStop("usage_unknown")
        return PRIVATE
    monkeypatch.setattr(warm.WarmSubscriptionClient, "complete", complete)
    monkeypatch.setattr(staged.StagedSubscriptionClient, "complete_stage", complete)
    monkeypatch.setattr(check, "_live_containment", lambda receipt: True)
    monkeypatch.setattr(check, "_resources", lambda receipt: (0, 0))
    loaded = {"warm": warm, "staged": staged, "request_type": LLMRequest,
              "binary": Path("/unused"), "candidate": Path(__file__).resolve().parents[3]}
    (tmp_path / "private-access").mkdir(mode=0o700)
    result = check.run_access(loaded, tmp_path.resolve(), {})
    assert len(calls) == (1 if failure else 2)
    assert PRIVATE not in json.dumps(result)


def clean_result():
    value = {"schema": check.SCHEMA, "status": "transport_verified", "turns": 2,
        "known_tokens": 84, "usage_complete": True, "reserved": 0, "in_flight": 0,
        "ordinary_completed": True, "staged_completed": True,
        "schema_acknowledged": True, "staged_response_valid": False,
        "client_cleanup_verified": True, "resource_denials": 0, "resource_oom": 0,
        "containment_verified": True}
    if "first_failure" in check.RESULT_FIELDS:
        value["first_failure"] = None
    if "stop_code" in check.RESULT_FIELDS:
        value["stop_code"] = None
    return value


@pytest.mark.parametrize("key,value", [("status", []), ("status", {}),
    ("staged_response_valid", 1), ("staged_response_valid", 0), ("staged_response_valid", 0.0),
    ("turns", True), ("known_tokens", True), ("resource_denials", False),
    ("client_cleanup_verified", False), ("resource_oom", 1), ("resource_denials", 1),
    ("in_flight", 1), ("reserved", 1), ("known_tokens", 32001),
    ("usage_complete", False), ("containment_verified", False), ("turns", 1)])
def test_public_reader_rejects_forged_success(key, value):
    item = clean_result()
    assert check.validate_result(item) is True
    item[key] = value
    assert check.validate_result(item) is False


def test_public_reader_rejects_private_fields():
    item = clean_result()
    item["error_message"] = PRIVATE
    assert check.validate_result(item) is False


@pytest.mark.parametrize("populated,expected", [("0", True), ("1", False), ("unknown", False)])
def test_recursive_cleanup_uses_kernel_descendant_population(monkeypatch, tmp_path, populated, expected):
    group = tmp_path / "group"
    group.mkdir()
    for name in ("cgroup.procs", "cgroup.threads"):
        (group / name).write_text("")
    (group / "cgroup.events").write_text("populated " + populated + "\n")
    monkeypatch.setattr(check, "_group_policy", lambda path: True)
    assert check._recursive_empty(group) is expected


def test_exit_zero_with_surviving_descendant_is_not_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(check, "_unit_values", lambda receipt: {"ActiveState": "active",
        "SubState": "exited", "MainPID": "0", "ControlGroup": "",
        "Result": "success", "ExecMainStatus": "0"})
    monkeypatch.setattr(check, "_group", lambda receipt: tmp_path)
    monkeypatch.setattr(check, "_recursive_empty", lambda group: False)
    runtime, cleanup = check._terminal_runtime({"expected_cgroup": "/invented"})
    assert runtime == "unverified" and cleanup is False


def test_running_label_requires_correct_live_group(monkeypatch, tmp_path):
    monkeypatch.setattr(check, "_unit_values", lambda receipt: {"ActiveState": "active",
        "SubState": "running", "MainPID": "123", "ControlGroup": "/wrong",
        "Result": "success", "ExecMainStatus": "0"})
    monkeypatch.setattr(check, "_group", lambda receipt: tmp_path)
    assert check._terminal_runtime({"expected_cgroup": "/invented"}) == ("unverified", False)


def test_private_first_failure_prose_cannot_escape_serializer():
    item = clean_result()
    item["status"] = "failed"
    item["first_failure"] = {"code": "fixed_other", "phase": "turn-events",
                             "message": PRIVATE}
    assert not check.validate_result(item, staged.warm.serialize_failure)
    item["first_failure"] = staged.warm.serialize_failure(item["first_failure"])
    assert check.validate_result(item, staged.warm.serialize_failure)
    assert PRIVATE not in json.dumps(item)


def test_failed_fixture_blocks_prepare_before_any_admission_or_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(check, "_root", lambda root: root)
    launcher = SimpleNamespace(host_admission=lambda: pytest.fail("admission after bad fixture"))
    monkeypatch.setattr(check, "verify_sources", lambda root: (launcher, {}))
    def broken(loaded):
        raise ValueError("invented_fixture_defect")
    monkeypatch.setattr(check, "build_fixture", broken)
    with pytest.raises(ValueError, match="invented_fixture_defect"):
        check.prepare(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_source_drift_prevents_launch_before_marker_or_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(check, "_root", lambda root: root)
    def drift(root):
        raise ValueError("launcher_source_drift")
    monkeypatch.setattr(check, "verify_sources", drift)
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **k: pytest.fail("dispatch on drift"))
    with pytest.raises(ValueError, match="launcher_source_drift"):
        check.launch(tmp_path, "a" * 64)
    assert list(tmp_path.iterdir()) == []


def test_real_receipt_gate_blocks_second_ambiguous_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(check, "_root", lambda root: root)
    launcher = SimpleNamespace(host_admission=lambda: None, _bus_env=lambda: {})
    monkeypatch.setattr(check, "verify_sources", lambda root: (launcher, {}))
    receipt = {"schema": check.SCHEMA, "unit": "invented.service"}
    monkeypatch.setattr(check, "receipt_for", lambda *a: receipt)
    check._write_once(tmp_path / "access-receipt.json", receipt)
    digest = check._sha(tmp_path / "access-receipt.json")
    for name in ("private-access", "access-empty", "access-tmp"):
        (tmp_path / name).mkdir(mode=0o700)
    calls = []
    def ambiguous(*a, **k):
        calls.append(1)
        raise subprocess.TimeoutExpired("systemd-run", 20)
    monkeypatch.setattr(check.subprocess, "run", ambiguous)
    with pytest.raises(subprocess.TimeoutExpired):
        check.launch(tmp_path, digest)
    with pytest.raises(ValueError, match="already_attempted"):
        check.launch(tmp_path, digest)
    assert calls == [1]


@pytest.mark.parametrize("kind", ["digest", "contents", "marker"])
def test_receipt_fails_closed_on_drift(monkeypatch, tmp_path, kind):
    expected = {"schema": check.SCHEMA, "unit": "invented.service"}
    monkeypatch.setattr(check, "receipt_for", lambda *a: expected)
    saved = expected if kind != "contents" else {**expected, "unit": "wrong.service"}
    check._write_once(tmp_path / "access-receipt.json", saved)
    digest = check._sha(tmp_path / "access-receipt.json")
    if kind == "digest":
        digest = "0" * 64
    check._write_once(tmp_path / "access-attempt.json", {
        "receipt_sha256": digest if kind != "marker" else "0" * 64, "one_shot": True})
    with pytest.raises(ValueError):
        check._receipt(tmp_path, digest, None, attempted=True)
