"""Root-authored adversarial verification of the observed pilot boundary."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


DIAG = Path(__file__).resolve().parents[1]
SECRET = "SYNTHETIC_UNTRUSTED_PRIVATE_PAYLOAD"
VALUES = [None, True, False, 0, -1, 1.5, SECRET, [], {}, [SECRET], {SECRET: SECRET}]


def load(name):
    spec = importlib.util.spec_from_file_location("root_review_" + name, DIAG / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = load("luna_observed_lme_progress")
launcher = load("luna_observed_lme_launch")


def failure():
    return {"code": "fixed_other", "phase": "run", "rpc": "turn/events",
        "process_index": 3, "request_index": 2, "retired_count": 1,
        "queue_count": 0, "known_tokens": 100, "process_age_seconds": 2.5,
        "turn_admitted": True, "known_usage": False, "usage_complete": False,
        "failure_family": "unexpected_notification", "event_family": "error",
        "app_server_error": {"identity": "matched", "will_retry": False,
                             "error_class": "usageLimitExceeded"}}


def test_valid_failure_preserves_safe_diagnostics():
    assert reader.first_failure_summary(failure()) == failure()


@pytest.mark.parametrize("field", sorted(reader._CORE))
def test_every_required_failure_key_must_be_present(field):
    value = failure()
    del value[field]
    assert reader.first_failure_summary(value) is None


@pytest.mark.parametrize("field", sorted(failure()))
@pytest.mark.parametrize("value", VALUES)
def test_untrusted_failure_shapes_never_crash_or_export_text(field, value):
    payload = failure()
    payload[field] = value
    assert SECRET not in json.dumps(reader.first_failure_summary(payload))


@pytest.mark.parametrize("field", ["identity", "will_retry", "error_class", "http_status_code"])
@pytest.mark.parametrize("value", VALUES)
def test_nested_error_shapes_are_total_and_source_free(field, value):
    payload = failure()
    payload["app_server_error"][field] = value
    assert SECRET not in json.dumps(reader.first_failure_summary(payload))


@pytest.mark.parametrize("field", ["category", "code"])
@pytest.mark.parametrize("value", VALUES)
def test_nested_rpc_shapes_are_total_and_source_free(field, value):
    payload = failure()
    payload.pop("event_family")
    payload.pop("failure_family")
    payload.pop("app_server_error")
    payload.update(code="rpc_failure:turn/start", rpc="turn/start",
                   rpc_error={"category": "internal_error", "code": -32603})
    payload["rpc_error"][field] = value
    assert SECRET not in json.dumps(reader.first_failure_summary(payload))


def test_reviewed_hashes_are_exact():
    sha = lambda name: hashlib.sha256((DIAG / name).read_bytes()).hexdigest()
    assert reader.RUNNER_SHA256 == sha("luna_observed_lme.py")
    assert reader.LAUNCHER_SHA256 == sha("luna_observed_lme_launch.py")
    assert reader.WARM_V3_SHA256 == hashlib.sha256(
        (DIAG.parents[1] / "benchmarks/codex_subscription_warm_v3.py").read_bytes()).hexdigest()


def test_healthy_terminal_still_requires_original_health_and_accounting(tmp_path, monkeypatch):
    fixture = load("tests/test_luna_subscription_profiled_v2_progress")
    root, receipt, safe, result, source = fixture._fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(reader.base, "bounded_json", lambda path: source[str(path)])
    safe["schema"] = result["schema"] = reader.SCHEMA
    safe["runner_sha256"] = reader.RUNNER_SHA256
    safe["candidate_source_map_sha256"] = reader.ground.GROUNDED_MAP_SHA256
    for value in (safe, result):
        value.update(effective_warm_transport_sha256=reader.WARM_V3_SHA256,
                     inherited_warm_transport_sha256=reader.WARM_V2_SHA256)
    assert reader.verify_terminal(root, receipt, safe, result)["validated"]
    for field in ("effective_warm_transport_sha256", "inherited_warm_transport_sha256"):
        expected = safe[field]
        safe[field] = "0" * 64
        assert not reader.verify_terminal(root, receipt, safe, result)["validated"]
        safe[field] = expected
    for field in ("usage_complete", "ok"):
        expected = safe[field]
        safe[field] = False
        assert not reader.verify_terminal(root, receipt, safe, result)["validated"]
        safe[field] = expected
    result["stage_accounting"]["q-0000"]["reader"]["known_tokens"] += 1
    assert not reader.verify_terminal(root, receipt, safe, result)["validated"]
    result["stage_accounting"]["q-0000"]["reader"]["known_tokens"] -= 1
    safe["first_failure"] = failure()
    assert not reader.verify_terminal(root, receipt, safe, result)["validated"]
