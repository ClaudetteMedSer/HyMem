from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

DIAG = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DIAG / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load("grounding_builder_test", "luna_grounding_candidate.py")
probe = _load("grounding_probe_test", "luna_grounding_probe.py")


def test_derived_prompt_is_exact_three_insertions():
    frozen = Path("/private/tmp/hymem-r9-summary-20260927.XzIhDt/candidate/hymem/extraction/prompts/__init__.py")
    if not frozen.is_file():
        pytest.skip("private frozen source unavailable")
    old = frozen.read_bytes()
    new = builder.derive_prompt(old)
    assert builder.sha(new) == probe.GROUNDED_PROMPT_SHA256
    assert new.count(b"PREDICATE_GROUNDING_VERSION") == 2
    assert new.count(b"Predicate grounding ({predicate_grounding_version})") == 1
    assert old[:old.index(b'_CHUNK_EXTRACTION_SYSTEM_TEMPLATE')] in new
    with pytest.raises(ValueError, match="original_prompt_drift"):
        builder.derive_prompt(old + b"\n")


def test_schedule_is_interleaved_and_order_reversed():
    units = probe.schedule()
    assert len(units) == 44
    assert [unit[3] for unit in units[:2]] == ["baseline", "candidate"]
    assert [unit[3] for unit in units[22:24]] == ["candidate", "baseline"]
    assert sum(unit[1] == "canary" for unit in units) == 4
    assert len(set(units)) == 44


@dataclass(frozen=True)
class Request:
    system: str
    user: str
    response_format: str = "json"
    max_tokens: int = 1024
    temperature: float = 0.0


def test_mapping_preserves_all_other_request_fields():
    delegate = SimpleNamespace(complete=lambda request: json.dumps({"system": request.system}),
                               budget=SimpleNamespace(halt=lambda code: None))
    wrapped = probe.MappingClient(delegate, {"old": "new"}, "candidate")
    answer = wrapped.complete(Request("old", "sensitive private user", max_tokens=17))
    assert json.loads(answer) == {"system": "new"}
    meta = wrapped.request_log[0]
    assert meta["before_user_sha256"] == meta["after_user_sha256"]
    assert meta["before_request"]["user"] == meta["after_request"]["user"]
    assert meta["before_request"]["max_tokens"] == meta["after_request"]["max_tokens"]
    with pytest.raises(probe.MappingFault, match="unexpected_system_prompt"):
        wrapped.complete(Request("unknown", "x"))


def test_private_evidence_failure_halts_before_delegate_call():
    calls = []
    stops = []
    delegate = SimpleNamespace(complete=lambda request: calls.append(request),
                               budget=SimpleNamespace(halt=stops.append))
    client = probe.MappingClient(delegate, {"old": "new"}, "candidate",
        on_request=lambda *_: (_ for _ in ()).throw(OSError("private disk detail")))
    with pytest.raises(probe.MappingFault, match="private_evidence_write_failure"):
        client.complete(Request("old", "private content"))
    assert calls == []
    assert stops == ["private_evidence_write_failure"]


def test_real_frozen_extractor_does_not_swallow_mapping_fault():
    frozen = Path("/private/tmp/hymem-r9-summary-20260927.XzIhDt/candidate")
    if not frozen.is_dir():
        pytest.skip("private frozen source unavailable")
    script = """
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from hymem.extraction import chunk
from luna_grounding_cases import CASES
from luna_grounding_probe import MappingClient, MappingFault
calls=[]
stops=[]
delegate=SimpleNamespace(complete=lambda request: calls.append(request), budget=SimpleNamespace(halt=stops.append))
wrapped=MappingClient(delegate, {}, 'candidate')
case=CASES[0]
try:
    chunk.extract_chunk(wrapped, case.text, source_records=case.source_records, completion_call_limit=8)
except MappingFault:
    assert calls == [] and stops == ['unexpected_system_prompt']
else:
    raise AssertionError('mapping fault swallowed')
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(frozen), str(DIAG)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class FakeBudget:
    def __init__(self, limits, *, max_in_flight):
        assert (limits.turns, limits.known_tokens, limits.seconds, max_in_flight) == (192, 2_000_000, 1800, 1)
        self.stop_code = None

    def snapshot(self):
        return {"stopped": self.stop_code is not None, "stop_code": self.stop_code,
                "usage_complete": True, "in_flight": 0, "reserved": 0,
                "turns": 0, "known_tokens": 0}

    def halt(self, code):
        self.stop_code = code


def fake_limits(turns, known_tokens, seconds):
    return SimpleNamespace(turns=turns, known_tokens=known_tokens, seconds=seconds)


class FakeConcurrentStop(BaseException):
    pass


class FakeClient:
    observed_turns = 0
    observed_tokens = 0
    usage_complete = True

    def __init__(self, fail_close=False):
        self.fail_close = fail_close

    def close(self):
        if self.fail_close:
            raise RuntimeError("private cleanup detail")


class FakeChunk:
    @staticmethod
    def extract_chunk(client, text, *, source_records, completion_call_limit):
        assert completion_call_limit == 8
        case = next(case for case in probe.cases.CASES if case.source_records == source_records)
        return SimpleNamespace(failed=False, triples=case.expected, markers=[],
                               entity_type_hints={}, entity_property_hints={},
                               completion_calls=0)


def _fake_canary(*args, **kwargs):
    return {"schema": "luna-experimental-canary-v2", "passed": True,
            "fixture_sha256": "fixed", "matched_core_claims": 2,
            "expected_core_claims": 2, "completion_calls": 0,
            "execution_path_valid": True, "core_execution_path_exact": True,
            "initial_prepartition_leaves": 1, "failure_code": None}


def test_all_units_continue_after_semantic_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.pinned.old, "experimental_canary", _fake_canary)
    original_grade = probe.cases.grade_case
    def grade(case_id, result):
        value = original_grade(case_id, result)
        if case_id == "preference_only":
            value["passed"] = False
        return value
    monkeypatch.setattr(probe.cases, "grade_case", grade)
    state = probe.run(binary="unused", concurrent=SimpleNamespace(
        BudgetLimits=fake_limits, SharedBudget=FakeBudget,
        ConcurrentStop=FakeConcurrentStop),
        warm=SimpleNamespace(WarmSession=object), canary=None, chunk=FakeChunk,
        mapping={"x": "y"}, output=tmp_path,
        client_factory=lambda *args: FakeClient())
    assert state["completed_and_clean"] is True
    assert state["completed_units"] == 44
    assert sum(not item["passed"] for item in state["units"]) == 4


def test_cleanup_failure_stops_without_reroll(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.pinned.old, "experimental_canary", _fake_canary)
    state = probe.run(binary="unused", concurrent=SimpleNamespace(
        BudgetLimits=fake_limits, SharedBudget=FakeBudget,
        ConcurrentStop=FakeConcurrentStop),
        warm=SimpleNamespace(WarmSession=object), canary=None, chunk=FakeChunk,
        mapping={"x": "y"}, output=tmp_path,
        client_factory=lambda *args: FakeClient(fail_close=True))
    assert state["completed_units"] == 1
    assert state["budget"]["stop_code"] == "cleanup_failure"
    assert state["completed_and_clean"] is False


def test_call_accounting_mismatch_stops_before_next_unit(tmp_path, monkeypatch):
    class BadCountChunk(FakeChunk):
        @staticmethod
        def extract_chunk(*args, **kwargs):
            result = FakeChunk.extract_chunk(*args, **kwargs)
            result.completion_calls = 1
            return result
    monkeypatch.setattr(probe.pinned.old, "experimental_canary", _fake_canary)
    state = probe.run(binary="unused", concurrent=SimpleNamespace(
        BudgetLimits=fake_limits, SharedBudget=FakeBudget,
        ConcurrentStop=FakeConcurrentStop),
        warm=SimpleNamespace(WarmSession=object), canary=None, chunk=BadCountChunk,
        mapping={"x": "y"}, output=tmp_path,
        client_factory=lambda *args: FakeClient())
    assert state["completed_units"] == 1
    assert state["budget"]["stop_code"] == "call_accounting_mismatch"


def test_process_group_liveness_stops_before_next_unit(tmp_path, monkeypatch):
    class Session:
        def __init__(self, *args, **kwargs):
            self.process = SimpleNamespace(pid=123456)
    class Client(FakeClient):
        def __init__(self, binary, budget, key, cap, *, session_factory, **kwargs):
            super().__init__()
            session_factory()
    monkeypatch.setattr(probe.pinned.old, "experimental_canary", _fake_canary)
    monkeypatch.setattr(probe.os, "killpg", lambda pid, signal: None)
    state = probe.run(binary="unused", concurrent=SimpleNamespace(
        BudgetLimits=fake_limits, SharedBudget=FakeBudget,
        ConcurrentStop=FakeConcurrentStop),
        warm=SimpleNamespace(WarmSession=Session, WarmSubscriptionClient=Client),
        canary=None, chunk=FakeChunk, mapping={"x": "y"}, output=tmp_path)
    assert state["completed_units"] == 1
    assert state["budget"]["stop_code"] == "process_group_remaining"


def test_final_private_write_failure_preserves_public_paid_work(tmp_path, monkeypatch):
    result = {"completed_and_clean": True, "completed_units": 2,
              "process_groups_absent": True, "all_candidate_passed": False,
              "units": [{"id": "r1-control-preference_only-baseline", "passed": True},
                        {"id": "r1-control-preference_only-candidate", "passed": False}],
              "budget": {"turns": 9, "known_tokens": 12345,
                         "usage_complete": True, "in_flight": 0},
              "campaign_stop": None}
    monkeypatch.setattr(probe, "private_write", lambda *_: (_ for _ in ()).throw(
        OSError("private disk full")))
    public = {"schema": probe.SCHEMA}
    probe.finish_public_result(public, result, tmp_path, {"runner_sha256": "fixed"})
    assert public["completed_and_clean"] is False
    assert public["stop_code"] == "private_result_write_failure"
    assert public["completed_units"] == 2
    assert public["units"] == result["units"]
    assert public["budget"]["known_tokens"] == 12345
