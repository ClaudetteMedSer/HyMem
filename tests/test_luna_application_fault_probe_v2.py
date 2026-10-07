"""Offline finite-schema and one-question orchestration controls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools/diagnostics/luna_application_fault_probe_v2.py"
spec = importlib.util.spec_from_file_location("application_fault_probe_v2_test", SOURCE)
probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(probe)


class CaptureStub:
    @staticmethod
    def validate_snapshot(value):
        if (type(value) is dict and set(value) == {"schema", "first", "cleanup"}
                and value["schema"] == "test-capture" and value["cleanup"] is None
                and (value["first"] is None or value["first"] == {
                    "kind": "top_level", "phase": "ingest", "exception": "runtime_error",
                    "gate_family": "none"})):
            return {"schema": value["schema"], "first": (
                dict(value["first"]) if value["first"] is not None else None), "cleanup": None}
        return None


def valid_result():
    return {"schema": probe.SCHEMA, "status": "inconclusive", "stop": "none", "phase": None,
            "selected_row_sha256": "0" * 64,
            "first_snapshot": {"schema": "test-capture", "first": None, "cleanup": None},
            "turns": 0, "known_tokens": 0, "usage_complete": True, "in_flight": 0,
            "reserved": 0, "stages": {key: {field: 0 for field in probe.COUNTS}
                                       for key in probe.STAGES},
            "resource": {"current": 1, "peak": 1, "limit": 256, "denials": 0},
            "checkpoint_durable": False, "adapter_cleanup_ok": True,
            "client_cleanup_ok": True, "accounting_reconciled": True}


@pytest.mark.parametrize("field,value", [
    ("turns", True), ("turns", 13), ("known_tokens", float("nan")),
    ("usage_complete", 1), ("checkpoint_durable", "yes"), ("status", "PRIVATE"),
    ("status", "budget_stop"),
    ("stop", "PRIVATE"), ("phase", "PRIVATE"), ("resource", {"current": False,
        "peak": 1, "limit": 256, "denials": 0}),
    ("resource", {"current": 2, "peak": 1, "limit": 256, "denials": 0}),
])
def test_result_rejects_malformed_metadata(field, value):
    row = valid_result()
    row[field] = value
    with pytest.raises(ValueError):
        probe.validate_result(row, CaptureStub)


def test_result_exact_fields_stage_and_snapshot_bounds():
    row = valid_result()
    validated = probe.validate_result(row, CaptureStub)
    validated["stages"]["extraction"]["attempts"] = 5
    assert row["stages"]["extraction"]["attempts"] == 0
    for mutate in (lambda r: r.update(private="PRIVATE"),
                   lambda r: r["stages"]["extraction"].update(private="PRIVATE"),
                   lambda r: r["stages"]["extraction"].update(attempts=False),
                   lambda r: r["first_snapshot"].update(private="PRIVATE"),
                   lambda r: r.update(checkpoint_durable=True),
                   lambda r: r.update(status="application_fault")):
        row = valid_result()
        mutate(row)
        with pytest.raises(ValueError):
            probe.validate_result(row, CaptureStub)


def test_status_and_accounting_consistency_fail_closed():
    for mutate in (
        lambda r: r.update(usage_complete=False),
        lambda r: r.update(in_flight=1),
        lambda r: r.update(adapter_cleanup_ok=False),
        lambda r: r.update(accounting_reconciled=False),
        lambda r: r.update(resource=None),
        lambda r: r.update(resource={"current": 1, "peak": 1, "limit": 256, "denials": 1}),
        lambda r: r.update(known_tokens=1),
        lambda r: r.update(status="provider_denial", stop="timeout"),
        lambda r: r.update(status="transport_stop", stop="quota_exhausted"),
    ):
        row = valid_result()
        mutate(row)
        with pytest.raises(ValueError):
            probe.validate_result(row, CaptureStub)


def test_resource_rejects_extra_strings_and_invalid_numbers():
    for value in (None, {"current": 1, "peak": 1, "limit": 256, "denials": 0,
                         "private": "secret"},
                  {"current": 1, "peak": 1, "limit": 256, "denials": float("nan")},
                  {"current": 1, "peak": 1, "limit": 255, "denials": 0}):
        with pytest.raises(ValueError):
            probe._resource(value)


def test_atomic_checkpoint_is_write_once_and_private(tmp_path):
    target = tmp_path / "first-fault.json"
    value = {"schema": "test-capture", "first": {"kind": "top_level", "phase": "ingest",
        "exception": "runtime_error", "gate_family": "none"}, "cleanup": None}
    probe._write_once(target, value)
    assert target.stat().st_mode & 0o777 == 0o600
    assert json.loads(target.read_text()) == value
    with pytest.raises(FileExistsError):
        probe._write_once(target, {"secret": "PRIVATE"})
    assert json.loads(target.read_text()) == value


def test_selects_exact_second_source_row_and_rejects_drift():
    questions = [{"question": f"q{i}", "haystack_sessions": [],
                  "haystack_session_ids": [], "haystack_dates": []} for i in range(4)]
    loaded = {"questions": questions, "dataset": Path("/private/test-data"),
              "protocol": object(), "prior": SimpleNamespace(
                  SelectedQuestions=lambda *_: iter(questions))}
    assert probe.select_question(loaded) is questions[1]
    loaded["prior"] = SimpleNamespace(SelectedQuestions=lambda *_: iter(questions[:1] + [
        dict(questions[1], question="tampered")] + questions[2:]))
    with pytest.raises(ValueError, match="drift"):
        probe.select_question(loaded)


def _offline_harness(monkeypatch, *, failure=None, cleanup_failure=False,
                     turns=0, usage_complete=True):
    events = []
    class Stop(BaseException):
        pass
    class Budget:
        def __init__(self, limits, max_in_flight):
            assert (limits.turns, limits.known_tokens, limits.seconds, max_in_flight) == (
                12, 160_000, 600, 1)
            self.stop_code = None
        def register(self, key, limits):
            assert key == "q-0001"
        def halt(self, code):
            if self.stop_code is None:
                self.stop_code = code
        def snapshot(self):
            return {"stop_code": self.stop_code, "stopped": self.stop_code is not None,
                    "turns": turns, "known_tokens": turns * 10,
                    "usage_complete": usage_complete,
                    "in_flight": 0, "reserved": 0, "questions": {"q-0001": {
                        "turns": turns, "known_tokens": turns * 10,
                        "usage_complete": usage_complete}}}
    class Limits:
        def __init__(self, *, turns, known_tokens, seconds):
            self.turns, self.known_tokens, self.seconds = turns, known_tokens, seconds
    class FirstCapture:
        def __init__(self, loaded, budget, *, intentional_types, on_first):
            self.budget, self.on_first = budget, on_first
            self.first = None
            self.cleanup = None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def record_top_level(self, exc, *, phase):
            if self.first is not None:
                return False
            self.first = {"kind": "top_level", "phase": "ingest",
                          "exception": "runtime_error", "gate_family": "none"}
            try:
                self.on_first(self.snapshot())
            except BaseException:
                self.budget.halt("capture_checkpoint_failure")
                raise Stop()
            self.budget.halt("application_fault")
            return True
        def record_cleanup(self, exc):
            self.cleanup = True
            self.budget.halt("cleanup_failure")
        def snapshot(self):
            return {"schema": "test-capture", "first": self.first, "cleanup": None}
    class Accounted:
        def __init__(self, client, candidate):
            self.counts = ({"extraction": {"attempts": turns, "returned": turns,
                             "turns": turns, "known_tokens": turns * 10}}
                           if turns else {})
        def reconcile(self):
            return True
    class Client:
        def close(self):
            events.append("client_close")
    class Adapter:
        def __init__(self, *args, **kwargs):
            events.append("adapter_construct")
        def open(self):
            events.append("open")
        def ingest_sessions(self, sessions, ids, dates, *, namespace):
            events.append(("ingest", sessions, ids, dates, namespace))
            if failure == "ingest":
                raise RuntimeError("PRIVATE INPUT MESSAGE")
        def dream_and_wait(self, *, timeout, max_cycles, require_healthy):
            assert 0 < timeout <= 540 and max_cycles == 100 and require_healthy is True
            events.append("dream")
            if failure == "dream":
                raise RuntimeError("PRIVATE OUTPUT MESSAGE")
        def close(self):
            events.append("adapter_close")
            if cleanup_failure:
                raise RuntimeError("PRIVATE CLEANUP MESSAGE")
        def search(self, *args, **kwargs):
            raise AssertionError("forbidden search")
        def answer_question_raw(self, *args, **kwargs):
            raise AssertionError("forbidden answer")
        def judge_answer_raw(self, *args, **kwargs):
            raise AssertionError("forbidden judge")
    def make_dual(loaded, budget, key, limits, output):
        budget.register(key, limits)
        events.append("client_setup")
        return Client()
    runner = SimpleNamespace(
        make_dual=make_dual,
        AccountedClient=Accounted,
        _memory_client=lambda *args: object())
    capture = SimpleNamespace(FirstApplicationFaultCapture=FirstCapture,
        FirstApplicationFaultStop=Stop, validate_snapshot=CaptureStub.validate_snapshot)
    selected = {"question": "INVENTED QUERY", "haystack_sessions": [[{"role": "user",
        "content": "INVENTED CONTENT"}]], "haystack_session_ids": ["PRIVATE ID"],
        "haystack_dates": ["2020-01-01"]}
    loaded = {"warm": SimpleNamespace(BudgetLimits=Limits, SharedBudget=Budget,
                                       ConcurrentStop=Stop),
              "candidate": Path("/private/candidate"),
              "prior": SimpleNamespace(old=SimpleNamespace(make_adapter_class=lambda *_: Adapter)),
              "lme": object(), "protocol": object(), "strictness": object(),
              "summary_classifier": object(),
              "diagnostic": SimpleNamespace(make_diagnostic_adapter_class=lambda *args: args[-1])}
    monkeypatch.setattr(probe, "verify_sources", lambda *_: None)
    monkeypatch.setattr(probe, "select_question", lambda *_: selected)
    return loaded, runner, capture, events


@pytest.mark.parametrize("failure,cleanup_failure,status", [
    (None, False, "inconclusive"), ("ingest", False, "application_fault"),
    ("dream", True, "unverified"),
])
def test_offline_orchestration_first_checkpoint_and_cleanup(
        tmp_path, monkeypatch, failure, cleanup_failure, status):
    loaded, runner, capture, events = _offline_harness(
        monkeypatch, failure=failure, cleanup_failure=cleanup_failure)
    resource_calls = []
    def resource():
        resource_calls.append(1)
        return {"current": 1, "peak": 1, "limit": 256, "denials": 0}
    output = tmp_path / "fresh"
    result = probe.run_probe(loaded, runner, capture, output=output,
                             containment_verified=True, binary_sha256="0" * 64,
                             resource_check=resource)
    assert result["status"] == status
    assert result["phase"] == (failure if failure else None)
    assert events[:3] == ["client_setup", "adapter_construct", "open"]
    assert events[3] == ("ingest", [[{"role": "user", "content": "INVENTED CONTENT"}]],
                         ["PRIVATE ID"], ["2020-01-01"], "INVENTED QUERY")
    assert events[-2:] == ["adapter_close", "client_close"]
    assert len(resource_calls) >= (3 if failure else 2)
    if failure:
        assert result["checkpoint_durable"] is True
        assert (output / "first-fault.json").is_file()
    assert "PRIVATE" not in (output / "probe-result.json").read_text()
    assert "PRIVATE" not in json.dumps(result)
    with pytest.raises(ValueError, match="gate"):
        probe.run_probe(loaded, runner, capture, output=output,
                        containment_verified=True, binary_sha256="0" * 64,
                        resource_check=resource)


def test_twelve_turn_cap_without_fault_is_inconclusive(tmp_path, monkeypatch):
    loaded, runner, capture, _ = _offline_harness(monkeypatch, turns=12)
    result = probe.run_probe(loaded, runner, capture, output=tmp_path / "cap",
        containment_verified=True, binary_sha256="0" * 64,
        resource_check=lambda: {"current": 1, "peak": 1, "limit": 256, "denials": 0})
    assert result["turns"] == 12 and result["known_tokens"] == 120
    assert result["status"] == "inconclusive"
    assert result["accounting_reconciled"] is True


def test_unknown_usage_marks_result_unverified(tmp_path, monkeypatch):
    loaded, runner, capture, _ = _offline_harness(monkeypatch, turns=1,
                                                   usage_complete=False)
    result = probe.run_probe(loaded, runner, capture, output=tmp_path / "unknown",
        containment_verified=True, binary_sha256="0" * 64,
        resource_check=lambda: {"current": 1, "peak": 1, "limit": 256, "denials": 0})
    assert result["turns"] == 1 and result["known_tokens"] == 10
    assert result["usage_complete"] is False
    assert result["status"] == "unverified"


def test_partial_client_setup_captures_first_before_unwind(tmp_path, monkeypatch):
    loaded, runner, capture, events = _offline_harness(monkeypatch)
    def broken(*args):
        events.append("client_setup")
        raise RuntimeError("PRIVATE SETUP FAILURE")
    runner.make_dual = broken
    result = probe.run_probe(loaded, runner, capture, output=tmp_path / "partial",
        containment_verified=True, binary_sha256="0" * 64,
        resource_check=lambda: {"current": 1, "peak": 1, "limit": 256, "denials": 0})
    assert events == ["client_setup"]
    assert result["status"] == "application_fault"
    assert result["phase"] == "client_setup"
    assert result["checkpoint_durable"] is True


def test_first_checkpoint_survives_resource_observer_failure(tmp_path, monkeypatch):
    loaded, runner, capture, _ = _offline_harness(monkeypatch, failure="ingest")
    calls = [0]
    def resource():
        calls[0] += 1
        if calls[0] == 2:
            return {"current": True, "peak": 1, "limit": 256, "denials": 0}
        return {"current": 1, "peak": 1, "limit": 256, "denials": 0}
    output = tmp_path / "resource-fault"
    result = probe.run_probe(loaded, runner, capture, output=output,
        containment_verified=True, binary_sha256="0" * 64, resource_check=resource)
    assert result["status"] == "unverified"
    assert result["checkpoint_durable"] is True
    assert json.loads((output / "first-fault.json").read_text())["first"]["phase"] == "ingest"
    assert result["first_snapshot"]["first"]["exception"] == "runtime_error"


def test_terminal_resource_failure_keeps_finite_stop(tmp_path, monkeypatch):
    loaded, runner, capture, _ = _offline_harness(monkeypatch)
    calls = [0]
    def resource():
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("PRIVATE CGROUP ERROR")
        return {"current": 1, "peak": 1, "limit": 256, "denials": 0}
    result = probe.run_probe(loaded, runner, capture, output=tmp_path / "terminal-fault",
        containment_verified=True, binary_sha256="0" * 64, resource_check=resource)
    assert result["status"] == "unverified"
    assert result["stop"] == "resource_observer_unverified"
    assert "PRIVATE" not in json.dumps(result)


def test_first_checkpoint_callback_failure_stops_without_private_export(tmp_path, monkeypatch):
    loaded, runner, capture, _ = _offline_harness(monkeypatch, failure="ingest")
    original_write = probe._write_once
    def failing_first(path, value):
        if path.name == "first-fault.json":
            raise OSError("PRIVATE CHECKPOINT FAILURE")
        return original_write(path, value)
    monkeypatch.setattr(probe, "_write_once", failing_first)
    output = tmp_path / "callback-failure"
    result = probe.run_probe(loaded, runner, capture, output=output,
        containment_verified=True, binary_sha256="0" * 64,
        resource_check=lambda: {"current": 1, "peak": 1, "limit": 256, "denials": 0})
    assert result["status"] == "unverified"
    assert result["stop"] == "capture_checkpoint_failure"
    assert result["checkpoint_durable"] is False
    assert result["first_snapshot"]["first"]["exception"] == "runtime_error"
    assert "PRIVATE" not in (output / "probe-result.json").read_text()


def test_actual_assembled_source_only_graph_rejects_live_probe(tmp_path):
    original = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle")
    repaired = Path("/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle")
    if not (original.is_dir() and repaired.is_dir()):
        pytest.skip("accepted frozen source bundles unavailable")
    bundle = tmp_path / "bundle"
    shutil.copytree(original / "code", bundle / "code")
    shutil.copytree(repaired / "candidate", bundle / "candidate")
    shutil.copy2(repaired / "source-map.json", bundle / "source-map.json")
    for name in ("luna_lme_diagnostic_v9.py", "luna_application_fault_capture_v2.py"):
        shutil.copy2(SOURCE.parent / name, bundle / "code/tools/diagnostics" / name)
    child = r'''
import importlib.util,json,sys
from pathlib import Path
bundle,probe_path=map(Path,sys.argv[1:3])
sys.path[:0]=[str(bundle/'candidate'),str(bundle/'code')]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module
runner=load('source_bound_runner',bundle/'code/tools/diagnostics/luna_lme_diagnostic_v9.py')
capture=load('source_bound_capture',bundle/'code/tools/diagnostics/luna_application_fault_capture_v2.py')
probe=load('source_bound_probe',probe_path)
loaded=runner.import_source_only(bundle,bundle/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
denied=False
try:probe.verify_sources(loaded,runner,capture,'0'*64)
except ValueError:denied=True
from hymem.extraction import chunk
from hymem.dreaming.runner import _CountingPhase1LLM,_HeartbeatLLMClient
from benchmarks.codex_subscription_warm_v9 import SharedBudget,BudgetLimits
limits=BudgetLimits(12,160000,600)
constructor_budget=SharedBudget(limits,max_in_flight=1)
private=bundle/'private-offline-constructor';private.mkdir(mode=0o700)
loaded['binary']=Path('/private/offline/no-inference-binary')
constructed=runner.make_dual(loaded,constructor_budget,'q-0001',limits,private)
registered=list(constructor_budget.snapshot()['questions'])
constructed.close()
budget=SharedBudget(limits,max_in_flight=1);budget.register('q-0001',limits)
class Dual:
    key='q-0001'
    def __init__(self):self.budget=budget;self.ordinary=0;self.structured=0
    def complete(self,request):
        budget.reserve(self.key)
        budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna',
            'config_isolation_admitted':True,'inference_enabled':False,
            'quota_windows':[{'remaining_percent':100}]})
        self.ordinary+=1;budget.settle(self.key,used=10,turn_started=True)
        return json.dumps({'triples':[{'subject':'user','predicate':'uses','object':'sqlite','polarity':1}]
            if self.ordinary==1 else [],'markers':[],'complete':True})
    def complete_stage(self,request,batch,stage,recheck):
        budget.reserve(self.key)
        budget.before_turn(self.key,{'auth':'chatgpt','model':'gpt-6-luna',
            'config_isolation_admitted':True,'inference_enabled':False,
            'quota_windows':[{'remaining_percent':100}]})
        self.structured+=1;budget.settle(self.key,used=11,turn_started=True)
        return json.dumps({'verdicts':[{'index':0,'supported':True,'reason':'support'}]})
dual=Dual();accounted=runner.AccountedClient(dual,bundle/'candidate')
memory=runner._memory_client(loaded,accounted)
client=_CountingPhase1LLM(_HeartbeatLLMClient(memory,lambda:None))
fault=capture.FirstApplicationFaultCapture({**loaded,'runner':runner},budget)
try:
    with fault:chunk.extract_chunk(client,'The user uses sqlite.',completion_call_limit=4)
except capture.FirstApplicationFaultStop:pass
print(json.dumps({'source_only':loaded['source_only'],'denied':denied,
    'registered':registered,
    'ordinary':dual.ordinary,'structured':dual.structured,'reconciled':accounted.reconcile(),
    'first':fault.snapshot()['first']}))
'''
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", child,
                                str(bundle), str(SOURCE)], capture_output=True,
                               text=True, timeout=45)
    assert completed.returncode == 0, completed.stderr
    state = json.loads(completed.stdout)
    assert state["source_only"] is True and state["denied"] is True
    assert state["registered"] == ["q-0001"]
    assert state["ordinary"] >= 1 and state["structured"] >= 1
    assert state["reconciled"] is True
