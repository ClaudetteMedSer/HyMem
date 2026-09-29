"""Independent root checks: source binding and untrusted diagnostic projection."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
import queue
import shutil
import subprocess
import sys

import pytest

from benchmarks import codex_subscription_warm_v3 as warm


SECRET = "SYNTHETIC_PRIVATE_TEXT_DO_NOT_EXPORT"
JSON_VALUES = [None, True, False, 0, -1, 65536, 1.5, SECRET, [], {}, [SECRET], {SECRET: SECRET}]


def event():
    return {"method": "error", "params": {
        "threadId": "root-thread", "turnId": "root-turn", "willRetry": False,
        "error": {"message": SECRET, "additionalDetails": SECRET,
                  "codexErrorInfo": "usageLimitExceeded"}}}


def failure():
    return {"code": "fixed_other", "phase": "run", "rpc": "turn/events",
            "process_age_seconds": 1.0, "process_index": 1, "request_index": 1,
            "retired_count": 0, "queue_count": 0, "turn_admitted": True,
            "known_usage": False, "known_tokens": 0, "usage_complete": False}


@pytest.mark.parametrize("value", JSON_VALUES)
@pytest.mark.parametrize("field", ["params", "threadId", "turnId", "willRetry",
                                     "error", "message", "codexErrorInfo",
                                     "httpStatusCode", "turnKind"])
def test_every_json_shape_is_total_and_never_exports_text(field, value):
    payload = event()
    params = payload["params"]
    if field == "params":
        payload[field] = value
    elif field in {"threadId", "turnId", "willRetry", "error"}:
        params[field] = value
    elif field == "httpStatusCode":
        params["error"]["codexErrorInfo"] = {
            "responseStreamDisconnected": {field: value, "message": SECRET}}
    elif field == "turnKind":
        params["error"]["codexErrorInfo"] = {"activeTurnNotSteerable": {field: value}}
    else:
        params["error"][field] = value
    result = warm._app_error(payload, "root-thread", "root-turn")
    encoded = json.dumps(result)
    assert SECRET not in encoded
    assert "root-thread" not in encoded and "root-turn" not in encoded


@pytest.mark.parametrize("field", list(failure()) + ["event_family", "failure_family",
                                                   "app_server_error", "rpc_error"])
@pytest.mark.parametrize("value", [SECRET, {"message": SECRET}, [SECRET]])
def test_serializer_revalidates_every_public_field(field, value):
    payload = failure()
    payload[field] = value
    projected = warm.serialize_failure(payload)
    assert SECRET not in json.dumps(projected)


@pytest.mark.parametrize("nested", [
    {"identity": "matched", "error_class": SECRET},
    {"identity": "matched", "will_retry": SECRET},
    {"identity": "matched", "error_class": "usageLimitExceeded", "message": SECRET},
    {"identity": SECRET},
    {"identity": "matched", "http_status_code": {"secret": SECRET}},
])
def test_serializer_does_not_trust_nested_app_server_fields(nested):
    payload = {**failure(), "app_server_error": nested}
    assert SECRET not in json.dumps(warm.serialize_failure(payload))


@pytest.mark.parametrize("thread,turn", [(None, None), ("other", "root-turn"),
                                        ("root-thread", "other"), ("", "")])
def test_unbound_or_foreign_identity_never_attributes_error_details(thread, turn):
    observed = warm._app_error(event(), thread, turn)
    assert observed.get("identity") != "matched"
    assert "error_class" not in observed and "will_retry" not in observed


def test_flat_bundle_loads_verified_sibling_not_shadowed_package(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("codex_subscription.py", "codex_subscription_concurrent_v2.py",
                 "codex_subscription_warm_v2.py", "codex_subscription_warm_v3.py"):
        shutil.copyfile(root / "benchmarks" / name, tmp_path / name)
    program = r'''
import importlib.util, pathlib, sys, types
bundle = pathlib.Path(sys.argv[1])
pkg = types.ModuleType("benchmarks")
sys.modules["benchmarks"] = pkg
poison = types.ModuleType("benchmarks.codex_subscription_warm_v2")
sys.modules[poison.__name__] = poison
pkg.codex_subscription_warm_v2 = poison
hymem = types.ModuleType("hymem")
extract = types.ModuleType("hymem.extraction")
llm = types.ModuleType("hymem.extraction.llm")
llm.LLMRequest = type("LLMRequest", (), {})
sys.modules.update({"hymem": hymem, "hymem.extraction": extract,
                    "hymem.extraction.llm": llm})
spec = importlib.util.spec_from_file_location("root_pinned_v3", bundle / "codex_subscription_warm_v3.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
assert mod.v2 is not poison
assert pathlib.Path(mod.v2.__file__).resolve() == bundle / "codex_subscription_warm_v2.py"
assert mod.base.MODEL == "gpt-6-luna"
print("flat_binding_verified")
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(tmp_path)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "flat_binding_verified"


def test_v2_and_base_source_pins_unchanged():
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    pins = {
        "codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
        "codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
        "codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    }
    for name, expected in pins.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


def wire_session(monkeypatch, events):
    """Exercise the real warm RPC/queue/router; substitute only pipe writes."""
    session = object.__new__(warm.WarmSession)
    session.created_at = warm.time.monotonic()
    session.deadline = session.created_at + 30
    session.next_id = 0
    session.events = queue.Queue()
    session.pending = deque()
    session.stage = "startup"
    session.active_thread = "root-thread"
    session.active_turn = None
    session.last_event = None
    session.rpc_error = None
    session.request_id = None
    session.retired_threads = set()
    session.starting_events = []
    session.warning_targets = []
    session.initialized_result = None
    session.initialized_sent = True
    for value in events:
        session.events.put(value)

    def send(self, method, params, *, notification=False):
        self.next_id += 1
        return self.next_id

    monkeypatch.setattr(warm.base.StdioSession, "send", send)
    return session


@pytest.mark.parametrize("before_response", [True, False])
@pytest.mark.parametrize("foreign_turn", [True, False])
def test_actual_wire_rpc_binds_before_classifying_queued_errors(monkeypatch, before_response, foreign_turn):
    response = {"id": 1, "result": {"turn": {"id": "root-turn", "status": "inProgress"}}}
    error = event()
    if foreign_turn:
        error["params"]["turnId"] = "foreign"
    events = [error, response] if before_response else [response, error]
    session = wire_session(monkeypatch, events)
    with pytest.raises(warm.base.SubscriptionTransportError, match="unexpected_notification:error"):
        warm.base._run_turn(session, "root-thread", "synthetic user")
    assert session.next_id == 1  # No retry or second turn.
    assert session.stage == "turn/events"
    assert session.last_event["event_family"] == "error"
    detail = session.last_event["app_server_error"]
    if foreign_turn:
        assert detail == {"identity": "mismatch"}
    else:
        assert detail == {"identity": "matched", "error_class": "usageLimitExceeded",
                          "will_retry": False}
    assert SECRET not in json.dumps(session.last_event)


def test_real_wire_rpc_error_is_not_misidentified_as_turn_event(monkeypatch):
    session = wire_session(monkeypatch, [{"id": 1, "error": {
        "code": -32603, "message": SECRET, "data": {"secret": SECRET}}}])
    with pytest.raises(warm.base.SubscriptionTransportError, match="rpc_failure:turn/start"):
        warm.base._run_turn(session, "root-thread", "synthetic user")
    assert session.stage == "turn/start" and session.last_event is None
    assert session.rpc_error == {"category": "internal_error", "code": -32603}


def test_real_wire_valid_response_and_usage_remain_unchanged(monkeypatch):
    item = {"type": "agentMessage", "id": "root-item", "phase": "final_answer", "text": "{}"}
    session = wire_session(monkeypatch, [
        {"id": 1, "result": {"turn": {"id": "root-turn", "status": "inProgress"}}},
        {"method": "item/started", "params": {"threadId": "root-thread", "turnId": "root-turn", "item": item}},
        {"method": "item/completed", "params": {"threadId": "root-thread", "turnId": "root-turn", "item": item}},
        {"method": "thread/tokenUsage/updated", "params": {"threadId": "root-thread", "turnId": "root-turn", "tokenUsage": {"total": {"totalTokens": 11}}}},
        {"method": "turn/completed", "params": {"threadId": "root-thread", "turn": {"id": "root-turn", "status": "completed"}}},
    ])
    assert warm.base._run_turn(session, "root-thread", "synthetic user") == ("{}", 11)
    assert session.next_id == 1 and session.last_event is None and session.rpc_error is None
