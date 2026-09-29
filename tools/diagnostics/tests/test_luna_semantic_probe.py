"""Network-free controls for the bounded semantic probe core."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from hymem.extraction import grounding
from tools.diagnostics import luna_semantic_cases as cases
from tools.diagnostics import luna_semantic_probe as probe


class FakeJudge:
    def __init__(self, case, *, override=None, bad_binding=False, bad_recheck=False):
        self.case = case
        self.override = override
        self.bad_binding = bad_binding
        self.bad_recheck = bad_recheck
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        wire = json.loads(request.user)
        recheck = len(self.requests) == 2
        verdicts = []
        for index, expected in enumerate(self.case.expected):
            status = ("supported" if recheck else
                      self.override or next(iter(sorted(expected.statuses))))
            if recheck and self.bad_recheck:
                status = "replace_predicate"
            predicate = (wire["batch"]["candidates"][index]["predicate"]
                         if status == "supported" else
                         expected.predicate if status == "replace_predicate" else None)
            evidence = ([{"source_message_id": self.case.triples[index].source_message_id,
                          "region": region, "quote": quote}
                         for region, quote in expected.evidence]
                        if status in {"supported", "replace_predicate"} else [])
            if status in {"supported", "replace_predicate"} and not evidence:
                evidence = [{"source_message_id": self.case.triples[index].source_message_id,
                             "region": "owned", "quote": self.case.sources[0].content[:30]}]
            verdicts.append({"index": index, "status": status,
                             "predicate": predicate, "evidence": evidence})
        return json.dumps({"schema": "source-grounding-v1",
                           "batch_sha256": ("0" * 64 if self.bad_binding else
                                            wire["batch_sha256"]),
                           "complete": True, "verdicts": verdicts})


def test_all_invented_controls_exact_schedule_and_private_prewrite():
    assert cases.suite_sha256() == probe.CASE_SHA256
    total = 0
    for case in cases.cases():
        events = []
        judge = FakeJudge(case)
        def record(event):
            events.append(event)
            if event["phase"].endswith("before_dispatch"):
                assert len(judge.requests) == len([x for x in events
                    if x["phase"].endswith("returned")])
        result = probe.run_control(case, judge, grounding, record=record)
        assert result["passed"], (case.case_id, result)
        assert len(judge.requests) == result["new_calls"]
        assert result["new_calls"] == (2 if case.category == "correction" else 1)
        for request in judge.requests:
            assert all(e.rationale not in request.user and e.rationale not in request.system
                       for e in case.expected)
        total += result["new_calls"]
    assert total == 26


def test_wrong_initial_is_scored_once_without_speculative_recheck():
    case = next(c for c in cases.cases() if c.case_id == "correct_use_to_prefer")
    judge = FakeJudge(case, override="unsupported")
    result = probe.run_control(case, judge, grounding, record=lambda _: None)
    assert result["outcome"] == "missed_recovery"
    assert result["new_calls"] == len(judge.requests) == 1


def test_unsafe_accept_and_binding_failure_are_separate():
    case = next(c for c in cases.cases() if c.case_id == "unrelated_citation")
    accepted = probe.run_control(case, FakeJudge(case, override="supported"),
                                 grounding, record=lambda _: None)
    assert accepted["false_support_indexes"] == [0]
    assert accepted["outcome"] == "false_support"
    bad = probe.run_control(case, FakeJudge(case, bad_binding=True), grounding,
                            record=lambda _: None)
    assert bad["outcome"] == "malformed"
    assert bad["malformed_code"] == "response:binding"


def test_second_correction_fails_without_third_call():
    case = next(c for c in cases.cases() if c.case_id == "correct_use_to_prefer")
    judge = FakeJudge(case, bad_recheck=True)
    result = probe.run_control(case, judge, grounding, record=lambda _: None)
    assert result["outcome"] == "malformed"
    assert result["recheck_failed"]
    assert result["missed_recovery_indexes"] == [0]
    assert result["new_calls"] == len(judge.requests) == 2


def test_prewrite_failure_prevents_dispatch():
    case = cases.cases()[0]
    judge = FakeJudge(case)
    with pytest.raises(OSError):
        probe.run_control(case, judge, grounding,
                          record=lambda _: (_ for _ in ()).throw(OSError()))
    assert judge.requests == []


def test_hybrid_only_forwards_grounding_and_checks_exact_ordinary_bytes():
    class Paid:
        observed_turns = 0
        observed_tokens = 0
        usage_complete = True
        def complete(self, request):
            self.observed_turns += 1
            self.observed_tokens += 7
            return "answer"
    ordinary = grounding.LLMRequest(system="s", user="u")
    pair = (asdict(ordinary), "ordinary answer")
    retained = (pair,) * 8
    paid = Paid()
    events = []
    hybrid = probe.HybridReplayClient(paid, retained, record=events.append,
                                      source_ids=(101, 102))
    assert hybrid.complete(ordinary) == "ordinary answer"
    assert hybrid.observed_turns == 1 and paid.observed_turns == 0
    with pytest.raises(probe.ProbeStop) as exc:
        hybrid.complete(grounding.LLMRequest(system="different", user="u"))
    assert exc.value.code == "ordinary_request_mismatch"
    assert paid.observed_turns == 0
    for _ in range(3):
        hybrid.complete(ordinary)
    for sid in (101, 102, 102):
        if sid == 102 and hybrid.new_calls == 1:
            for _ in range(4):
                hybrid.complete(ordinary)
        wire = {"batch": {"candidates": [{"source_message_id": sid}]},
                "batch_sha256": "x"}
        request = grounding.LLMRequest(system="ground", user=json.dumps(wire))
        assert hybrid.complete(request) == "answer"
    assert hybrid.new_calls == paid.observed_turns == 3
    assert hybrid.observed_turns == 11
    assert hybrid.observed_tokens == 21
    assert len([e for e in events if e["phase"].endswith("before_dispatch")]) == 3
    with pytest.raises(probe.ProbeStop):
        hybrid.complete(request)


def test_private_journal_requires_fresh_private_directory(tmp_path):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    journal = probe.PrivateJournal(directory)
    journal.record("control-00", {"phase": "initial_before_dispatch", "request": {"user": "secret"}})
    files = list(directory.iterdir())
    assert len(files) == 1 and files[0].stat().st_mode & 0o077 == 0
    with pytest.raises(probe.ProbeStop):
        probe.PrivateJournal(directory)
