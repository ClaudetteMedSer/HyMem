"""Offline controls for the frozen source-bound first-fault seam."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


BUNDLE = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle")
CAPTURE = Path(__file__).resolve().parents[1] / "tools/diagnostics/luna_application_fault_capture_v1.py"


def _run(body: str) -> dict:
    setup = r'''
import importlib.util, json, sys
from pathlib import Path
bundle, capture_path = map(Path, sys.argv[1:3])
sys.path[:0] = [str(bundle / "candidate"), str(bundle / "code")]
from hymem.extraction import chunk
from hymem.dreaming.runner import _CountingPhase1LLM, _HeartbeatLLMClient
from benchmarks.codex_subscription_warm_v9 import SharedBudget, BudgetLimits
spec = importlib.util.spec_from_file_location("fault_capture", capture_path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
loaded = {"candidate": bundle / "candidate", "code": bundle / "code", "chunk": chunk}
limits = BudgetLimits(turns=12, known_tokens=160000, seconds=600)
def budget():
    value = SharedBudget(limits, max_in_flight=1)
    value.register("q", limits)
    return value
'''
    result = subprocess.run([sys.executable, "-B", "-c", setup + body,
                             str(BUNDLE), str(CAPTURE)], capture_output=True,
                            text=True, timeout=25)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="frozen bundle unavailable")
def test_actual_chunk_after_return_fault_stops_before_swallow_and_restores():
    state = _run(r'''
spec = importlib.util.spec_from_file_location("frozen_runner", bundle / "code/tools/diagnostics/luna_lme_diagnostic_v8.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
b = budget()
class Dual:
    key = "q"
    def __init__(self): self.calls = 0; self.budget = b
    def complete(self, request):
        b.reserve("q")
        b.before_turn("q", {"auth":"chatgpt", "model":"gpt-6-luna", "config_isolation_admitted":True,
                            "inference_enabled":False, "quota_windows":[{"remaining_percent":100}]})
        self.calls += 1
        b.settle("q", used=10, turn_started=True)
        return '{"triples":[],"markers":[],"complete":true}'
dual = Dual()
accounted = runner.AccountedClient(dual, bundle / "candidate")
memory = runner._memory_client({}, accounted)
beats = [0]
def heartbeat():
    beats[0] += 1
    if beats[0] == 2: raise RuntimeError("PRIVATE AFTER RETURN")
counting = _CountingPhase1LLM(_HeartbeatLLMClient(memory, heartbeat))
original = chunk._failure
probe = capture.FirstApplicationFaultCapture(loaded, b)
try:
    with probe:
        chunk.extract_chunk(counting, "private source content", completion_call_limit=2)
except capture.FirstApplicationFaultStop:
    stopped = True
else:
    stopped = False
print(json.dumps({"stopped": stopped, "restored": chunk._failure is original,
                  "calls": dual.calls, "reconciled": accounted.reconcile(),
                  "stop": b.stop_code, "snapshot": probe.snapshot()}))
''')
    assert state["stopped"] and state["restored"]
    assert state["calls"] == 1 and state["reconciled"]
    assert state["stop"] == "application_fault"
    assert state["snapshot"]["first"] == {"kind": "call_failure", "phase": "heartbeat_boundary",
                                             "exception": "runtime_error", "gate_family": "none"}


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="frozen bundle unavailable")
def test_pre_return_first_immutable_cleanup_and_nested_rejection():
    state = _run(r'''
b = budget()
original = chunk._failure
probe = capture.FirstApplicationFaultCapture(loaded, b)
with probe:
    try:
        with capture.FirstApplicationFaultCapture(loaded, b): pass
    except ValueError: nested = True
    else: nested = False
    class Failing:
        def complete(self, request): raise ValueError("PRIVATE INPUT 123")
    try: chunk.extract_chunk(Failing(), "private source", completion_call_limit=1)
    except capture.FirstApplicationFaultStop: stopped = True
    else: stopped = False
    first = probe.snapshot()
    probe.record_top_level(RuntimeError("PRIVATE SECOND"), phase="question_path")
    probe.record_cleanup(OSError("PRIVATE CLEANUP"))
    final = probe.snapshot()
print(json.dumps({"nested":nested,"stopped":stopped,"restored":chunk._failure is original,
                  "first":first,"final":final}))
''')
    assert state["nested"] and state["stopped"] and state["restored"]
    assert state["first"]["first"] == state["final"]["first"]
    assert state["final"]["first"]["exception"] == "value_error"
    assert state["final"]["cleanup"]["exception"] == "os_error"
    assert "PRIVATE" not in json.dumps(state)


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="frozen bundle unavailable")
def test_validation_source_and_intentional_controls():
    state = _run(r'''
b = budget()
probe = capture.FirstApplicationFaultCapture(loaded, b, intentional_types=(TimeoutError,))
intentional = probe.record_top_level(TimeoutError("private"), phase="question_path")
class PrivateValueError(Exception): pass
other = probe.record_top_level(PrivateValueError("private"), phase="question_path")
valid = probe.snapshot()
mutations = []
for key, value in (("schema", 1), ("first", {"kind":"top_level", "phase":"question_path",
                                               "exception":"private", "gate_family":"none"}),
                   ("cleanup", {"kind":"cleanup", "phase":"cleanup",
                                "exception":"runtime_error", "gate_family":False})):
    modified = dict(valid); modified[key] = value
    mutations.append(capture.validate_snapshot(modified) is None)
detached = capture.validate_snapshot(valid)
detached["first"]["exception"] = "runtime_error"
source_rejected = False
original = chunk._failure
chunk._failure = lambda *_: None
try:
    try: capture.FirstApplicationFaultCapture(loaded, b)
    except ValueError: source_rejected = True
finally:
    chunk._failure = original
print(json.dumps({"intentional":intentional,"other":other,"first":probe.snapshot()["first"],
                  "mutations":mutations,"detached":probe.snapshot()["first"]["exception"],
                  "source_rejected":source_rejected}))
''')
    assert state["intentional"] is False and state["other"] is True
    assert state["first"]["exception"] == "other"
    assert state["mutations"] == [True, True, True]
    assert state["detached"] == "other" and state["source_rejected"]


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="frozen bundle unavailable")
def test_base_exception_restoration_and_grounding_family():
    state = _run(r'''
b = budget()
original = chunk._failure
probe = capture.FirstApplicationFaultCapture(loaded, b)
try:
    with probe:
        raise KeyboardInterrupt()
except KeyboardInterrupt:
    pass
gate = chunk.GroundingGateError("source:invalid")
recorded = probe.record_top_level(gate, phase="question_path")
snapshot = probe.snapshot()
invalid = dict(snapshot)
invalid["first"] = dict(snapshot["first"], gate_family="private")
print(json.dumps({"restored":chunk._failure is original, "recorded":recorded,
                  "first":snapshot["first"], "invalid":capture.validate_snapshot(invalid) is None}))
''')
    assert state["restored"] and state["recorded"] and state["invalid"]
    assert state["first"]["exception"] == "grounding_gate_error"
    assert state["first"]["gate_family"] == "source"


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="frozen bundle unavailable")
def test_real_grounding_wrapper_records_one_known_cause():
    state = _run(r'''
b = budget()
good = {"triples":[{"subject":"user","predicate":"uses","object":"sqlite","polarity":1}],
        "markers":[],"complete":True}
empty = {"triples":[],"markers":[],"complete":True}
class Staged:
    def __init__(self): self.calls = 0; self.grounding_calls = 0
    def complete(self, request):
        self.calls += 1
        return json.dumps(good if self.calls == 1 else empty)
    def complete_stage(self, request, batch, stage, recheck):
        self.grounding_calls += 1
        raise RuntimeError("PRIVATE GROUNDING SOURCE")
client = Staged()
probe = capture.FirstApplicationFaultCapture(loaded, b)
try:
    with probe:
        chunk.extract_chunk(client, "The user uses sqlite.", completion_call_limit=4)
except capture.FirstApplicationFaultStop:
    stopped = True
else:
    stopped = False
print(json.dumps({"stopped":stopped,"calls":client.calls,"grounding_calls":client.grounding_calls,
                  "first":probe.snapshot()["first"]}))
''')
    assert state["stopped"] and state["grounding_calls"] == 1
    assert state["first"]["kind"] == "call_failure"
    assert state["first"]["exception"] == "runtime_error"
    assert "PRIVATE" not in json.dumps(state)


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="frozen bundle unavailable")
def test_spoofed_code_identity_and_first_checkpoint_callback():
    state = _run(r'''
b = budget()
original = chunk._failure
namespace = {"__name__":chunk.__name__}
exec(compile("def _failure(reason, *details): return None", chunk.__file__, "exec"), namespace)
chunk._failure = namespace["_failure"]
try:
    try: capture.FirstApplicationFaultCapture(loaded, b)
    except ValueError: spoof_rejected = True
    else: spoof_rejected = False
finally:
    chunk._failure = original
calls = []
def on_first(snapshot):
    calls.append(snapshot)
    snapshot["first"]["exception"] = "private"
probe = capture.FirstApplicationFaultCapture(loaded, b, on_first=on_first)
probe.record_top_level(ValueError("PRIVATE MESSAGE"), phase="ingest")
first = probe.snapshot()
probe.record_top_level(RuntimeError("PRIVATE SECOND"), phase="dream")
bad_budget = budget()
def broken(_snapshot): raise RuntimeError("PRIVATE SINK")
broken_probe = capture.FirstApplicationFaultCapture(loaded, bad_budget, on_first=broken)
try: broken_probe.record_top_level(ValueError("private"), phase="client_setup")
except capture.FirstApplicationFaultStop: escaped = True
else: escaped = False
class BadHalt:
    def halt(self, code): raise RuntimeError("PRIVATE HALT")
halt_probe = capture.FirstApplicationFaultCapture(loaded, BadHalt())
try: halt_probe.record_top_level(ValueError("private"), phase="adapter_open")
except capture.FirstApplicationFaultStop: halt_escaped = True
else: halt_escaped = False
print(json.dumps({"spoof_rejected":spoof_rejected,"calls":len(calls),"first":first["first"],
                  "escaped":escaped,"halt_escaped":halt_escaped,"bad_stop":bad_budget.stop_code,
                  "bad_first":broken_probe.snapshot()["first"]}))
''')
    assert state["spoof_rejected"] and state["calls"] == 1
    assert state["first"]["exception"] == "value_error"
    assert state["first"]["phase"] == "ingest"
    assert state["escaped"] and state["bad_stop"] == "capture_checkpoint_failure"
    assert state["halt_escaped"]
    assert state["bad_first"]["exception"] == "value_error"
    assert "PRIVATE" not in json.dumps(state)
