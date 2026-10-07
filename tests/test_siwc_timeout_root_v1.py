"""Independent timeout-observation controls; invented fixtures, no live I/O."""
import copy
from email.message import Message
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest

from benchmarks import chatgpt_plan_responses_v6 as old
from benchmarks import chatgpt_plan_responses_v7 as wire
from tests.test_chatgpt_plan_responses_root_v1 import terminal
from tests.test_chatgpt_plan_responses_root_v5 import sequence


ROOT = Path(__file__).resolve().parents[1]


def test_consumed_sources_and_terminal_evidence_remain_unchanged():
    pins = {
        "benchmarks/chatgpt_plan_responses_v6.py": "811bff13ebc4b24ebd22cad16542c3b58597dc04538a1d7deb5085df7190a28f",
        "benchmarks/chatgpt_plan_lme_v1.py": "857aeebc2695ac2bc014643f023acd1751c5f7086c0196cd4d7a3d8489569ca5",
        "tools/diagnostics/siwc_lme_diagnostic_v3.py": "81f055ed3c64c7df03f4e038ae24d452691d37f04df1f07d7dcad375ebc4dae0",
        "tools/diagnostics/siwc_lme_diagnostic_progress_v5.py": "6719e840088ae88c35b5e07ddff1646aeecd67f890329133fb3ae226371d9868",
        "tools/diagnostics/siwc_lme_diagnostic_launch_v2.py": "fee76e6ebe40ca754d92eb8a90958db014ae39fbe59ebbad8cf5eaee86f25aa6",
        "docs/plans/2026-10-01-siwc-n_jualpy-terminal-metadata.json": "6986829e07205814a9243a74c0a09ff0b00083e3757106173328e3432a0da94e",
    }
    for relative, digest in pins.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_route_request_and_caps_are_identical():
    for name in ("HOST", "PATH", "MODEL", "MAX_WALL_SECONDS", "MAX_WIRE_BYTES",
                 "MAX_LINE_BYTES", "MAX_MEANINGFUL_EVENTS", "MAX_OUTPUT_CHARS",
                 "MAX_REQUEST_BYTES"):
        assert getattr(wire, name) == getattr(old, name)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    for output_schema in (None, schema):
        assert wire.build_request("invented system", "invented user", output_schema) == (
            old.build_request("invented system", "invented user", output_schema))


@pytest.mark.parametrize("kind", ["ordinary", "missing", "null", "empty"])
def test_identical_successful_parsing_and_usage(kind):
    events = [terminal()] if kind == "ordinary" else sequence(kind)
    before = copy.deepcopy(events)
    assert wire.parse_stream_events(events) == old.parse_stream_events(events)
    assert events == before


@pytest.mark.parametrize("field,value", [
    ("model", "invented-other-model"), ("status", "incomplete"),
    ("usage", None), ("output", {}), ("output", []),
    ("error", {"message": "PRIVATE_ROOT_SENTINEL"}),
])
def test_identical_terminal_rejection(field, value):
    event = terminal()
    event["response"][field] = value
    with pytest.raises(old.TransportError) as previous:
        old.parse_stream_events([copy.deepcopy(event)])
    with pytest.raises(wire.TransportError) as current:
        wire.parse_stream_events([copy.deepcopy(event)])
    assert current.value.code == previous.value.code
    assert current.value.stream_observation == previous.value.stream_observation
    assert "PRIVATE_ROOT_SENTINEL" not in repr(current.value)


@pytest.mark.parametrize("code", sorted(old.PROVIDER_CODES))
def test_explicit_provider_denial_semantics_unchanged(code):
    events = [{"type": "response.failed", "response": {
        "error": {"code": code, "message": "PRIVATE_ROOT_SENTINEL"}}}]
    with pytest.raises(old.TransportError) as previous:
        old.parse_stream_events(copy.deepcopy(events))
    with pytest.raises(wire.TransportError) as current:
        wire.parse_stream_events(copy.deepcopy(events))
    assert previous.value.code == current.value.code == code
    assert "PRIVATE_ROOT_SENTINEL" not in repr(current.value)
    assert current.value.timeout_observation is None


def _root_child(send, slots, mode, timeout):
    # This hook is installed in the spawned process too. A fixture mistake
    # cannot turn a timeout reproduction into an account or network request.
    def prohibit_network(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise AssertionError("offline_network_forbidden")
    sys.addaudithook(prohibit_network)
    if mode == "startup":
        time.sleep(timeout + 2)
        return

    class Response:
        status = 200
        headers = Message()
        def __init__(self):
            event = terminal() if mode == "completion_then_stall" else {"type": "response.created"}
            self.lines = iter((b"data: " + json.dumps(event).encode() + b"\n", b"\n"))
        def getheader(self, name, default=None):
            return "text/event-stream" if name == "Content-Type" else default
        def readline(self, size):
            if mode == "stream":
                time.sleep(timeout + 2)
            if mode in {"completion_then_stall", "event_then_stall"}:
                try:
                    return next(self.lines)
                except StopIteration:
                    time.sleep(timeout + 2)
            return b""

    class Connection:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, *args, **kwargs):
            if mode == "request":
                time.sleep(timeout + 2)
        def getresponse(self):
            if mode == "headers":
                time.sleep(timeout + 2)
            return Response()
        def close(self):
            if mode == "close":
                time.sleep(timeout + 2)

    wire.http.client.HTTPSConnection = Connection
    wire.ssl.create_default_context = lambda: None
    if mode == "parse":
        def parser(events, tracker):
            list(events)
            time.sleep(timeout + 2)
        wire._v6.parse_stream_events = parser
    wire._child(send, slots, wire.Credentials("invented-not-a-token"),
                wire.build_request("invented", "invented"), timeout)


@pytest.mark.parametrize("mode,phase,event_count,completion", [
    ("startup", "unknown", None, None),
    ("request", "request_send", 0, False),
    ("headers", "headers_wait", 0, False),
    ("stream", "stream_read", 0, False),
    ("parse", "parse", 0, False),
    ("close", "response_close", 0, False),
    ("event_then_stall", "stream_read", 1, False),
    ("completion_then_stall", "stream_read", 1, True),
])
def test_actual_child_timeout_phase_without_network(mode, phase, event_count, completion):
    children_before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_root_child, (mode,), 1.0)
    elapsed = time.monotonic() - started
    exc = caught.value
    assert exc.code == "timeout" and elapsed < 4
    observation = wire.sanitize_timeout_observation(exc.timeout_observation)
    assert observation is not None and observation == exc.timeout_observation
    assert observation["child_phase"] == phase
    assert observation["event_count"] == event_count
    assert observation["completion_seen"] is completion
    assert observation["result_ready"] is (None if phase == "unknown" else False)
    assert observation["timeout_allowance_ms"] == 1000
    assert observation["parent_elapsed_ms"] >= 900
    assert observation["parent_timeout_site"] in {"result_wait", "result_recv", "deadline_after_recv"}
    assert "invented" not in json.dumps(observation)
    assert {child.pid for child in multiprocessing.active_children()} == children_before


def _partial_result(send, slots, timeout):
    progress = wire._Progress(slots, timeout)
    progress.mark("child_entry")
    progress.mark("result_ipc", ready=True, ipc=True)
    os.write(send.fileno(), struct.pack("!i", 10000) + b"invented-partial")
    time.sleep(timeout + 2)


def test_partial_result_delivery_remains_bounded_and_attributed():
    before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(wire.TransportError) as caught:
        wire._run_child(_partial_result, (), 1)
    assert time.monotonic() - started < 4
    observation = caught.value.timeout_observation
    assert caught.value.code == "timeout"
    assert observation["parent_timeout_site"] == "result_recv"
    assert observation["child_phase"] == "result_ipc"
    assert observation["result_ready"] and observation["result_ipc_started"]
    assert "invented-partial" not in repr(caught.value)
    assert {child.pid for child in multiprocessing.active_children()} == before


def _observation():
    return {"child_phase": "stream_read", "parent_timeout_site": "result_wait",
            "last_progress_elapsed_ms": 1, "parent_elapsed_ms": 999,
            "timeout_allowance_ms": 1000, "elapsed_saturated": False,
            "snapshot_valid": True, "wire_bytes": 90, "event_count": 1,
            "completion_seen": True, "result_ready": False,
            "result_ipc_started": False, "child_alive_when_sampled": True}


@pytest.mark.parametrize("value", [True, -1, 1.1, float("inf"), float("nan"),
                                  "PRIVATE_ROOT_SENTINEL", [], {}, 10**100])
def test_all_timeout_count_fields_reject_nonfinite_or_noninteger_data(value):
    for key in ("last_progress_elapsed_ms", "parent_elapsed_ms",
                "timeout_allowance_ms", "wire_bytes", "event_count"):
        assert wire.sanitize_timeout_observation({**_observation(), key: value}) is None


def test_timeout_metadata_is_copy_isolated_and_repr_resanitized():
    source = _observation()
    exc = wire.TransportError("timeout", timeout_observation=source)
    source["child_phase"] = "PRIVATE_ROOT_SENTINEL"
    assert exc.timeout_observation["child_phase"] == "stream_read"
    exc.timeout_observation["child_phase"] = "PRIVATE_ROOT_SENTINEL"
    assert "PRIVATE_ROOT_SENTINEL" not in repr(exc)
    assert wire.TransportError("quota_failure", timeout_observation=_observation()).timeout_observation is None


def test_actual_timeout_metadata_reaches_ledger_without_invented_usage(monkeypatch):
    from benchmarks import chatgpt_plan_lme_v2 as bridge
    from hymem.extraction.llm import LLMRequest
    broker = object.__new__(bridge.owner.CredentialBroker)
    broker.identity_digest = "a" * 64
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire", lambda self, **kw:
                        bridge.owner.CredentialLease("invented-token", int(time.time()) + 900))
    limits = bridge.warm.BudgetLimits(5, 10000, 300)
    budget = bridge.SharedBudget(limits, max_in_flight=4)
    attempts = []
    def invoke(*args, **kwargs):
        attempts.append(kwargs["timeout"])
        return wire._run_child(_root_child, ("completion_then_stall",), 1)
    client = bridge.SIWCLMEClient(broker, budget, "q", limits, response_call=invoke)
    with pytest.raises(bridge.BridgeError, match="^timeout$"):
        client.complete(LLMRequest("invented", "invented"))
    summary = bridge.validate_summary_projection(client.diagnostic_summary())
    first = copy.deepcopy(summary["first_failure"])
    assert first["timeout_observation"]["completion_seen"] is True
    assert first["timeout_observation"]["result_ready"] is False
    assert first["unknown_usage"] is True
    assert summary["schema"] == "siwc_lme_summary_v2"
    assert summary["calls"] == summary["failures"] == summary["internal_http_attempts"] == 1
    assert summary["successes"] == summary["known_tokens"] == 0
    state = budget.snapshot()
    assert state["turns"] == 1 and state["known_tokens"] == 0
    assert state["usage_complete"] is False and state["stopped"]
    assert state["reserved"] == state["in_flight"] == 0
    assert state["first_failure"]["timeout_observation"] == first["timeout_observation"]
    summary["first_failure"]["timeout_observation"]["child_phase"] = "PRIVATE_ROOT_SENTINEL"
    with pytest.raises(bridge.warm.ConcurrentStop):
        client.complete(LLMRequest("invented", "invented"))
    assert len(attempts) == 1
    assert client.diagnostic_summary()["first_failure"] == first


@pytest.mark.parametrize("drift", ["none", "v7_origin", "v6_origin"])
def test_isolated_source_import_and_origin_guards_without_external_io(drift):
    script = r'''
import json, pathlib, sys
root, drift = sys.argv[1:]
sys.path.insert(0, root)
def prohibit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"}:
        raise RuntimeError("external_io_forbidden")
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if path.endswith("/auth.json") or "/.codex/" in path or "/.hymem-siwc-owner" in path:
            raise RuntimeError("credential_state_forbidden")
sys.addaudithook(prohibit)
from benchmarks import chatgpt_plan_responses_v7 as v7
from benchmarks import chatgpt_plan_responses_v6 as v6
if drift == "v7_origin": v7.__file__ = "/invented/wrong-origin.py"
if drift == "v6_origin": v6.__file__ = "/invented/wrong-origin.py"
try:
    from benchmarks import chatgpt_plan_lme_v2 as bridge
except RuntimeError as exc:
    if drift == "none" or str(exc) != "pinned_siwc_source_mismatch": raise
    print(json.dumps({"drift_rejected": True}))
else:
    assert drift == "none"
    assert bridge.transport is v7 and bridge.transport_v6 is v6
    assert bridge.MAX_INVOCATION == 120.0
    print(json.dumps({"source_import_ok": True}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT), drift],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == (
        {"source_import_ok": True} if drift == "none" else {"drift_rejected": True})
