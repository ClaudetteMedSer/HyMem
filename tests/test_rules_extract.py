"""Idea B write-side — LLM durability tagger (`hymem/rules_extract.py`).

The semantic instrument that replaces the lexical classifier's ~14% ceiling.
These lock the pieces the live routing depends on: batch parsing is index-aligned
and precision-safe on garbage; the confidence threshold gates minting; the
fastpath shortcuts lexical hits WITHOUT an LLM call and sends only the ambiguous
rest; and every failure mode degrades to "don't mint" (never a spurious rule).
"""

from __future__ import annotations

import json

import pytest

from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.rules_extract import (
    _parse_batch,
    judge_durability_batch,
    route_decisions,
)


class _EchoJudge:
    """A fake judge that reads the markers out of the request and returns a
    standing verdict for each — recording how many markers each call carried,
    so tests can assert sub-batching."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def complete(self, req: LLMRequest) -> str:
        body = req.user.split("Markers:", 1)[1].rsplit("Return", 1)[0].strip()
        payload = json.loads(body)
        self.calls.append(len(payload))
        return json.dumps([
            {"index": m["index"], "standing": True, "confidence": 0.9, "rule": m["statement"]}
            for m in payload
        ])


# ── batch parsing: index-aligned, precision-safe ────────────────────────────

def test_parse_batch_wellformed():
    raw = ('[{"index":0,"standing":true,"confidence":0.9,"rule":"Never use X"},'
           '{"index":1,"standing":false,"confidence":0.1,"rule":null}]')
    out = _parse_batch(raw, 2)
    assert out[0].standing and out[0].rule == "Never use X" and out[0].confidence == 0.9
    assert not out[1].standing and out[1].rule is None


def test_parse_batch_standing_without_rule_still_routes():
    # A correct standing verdict with no canonical rewrite must NOT be dropped —
    # the raw statement is the fallback rule text downstream.
    out = _parse_batch('[{"index":0,"standing":true,"confidence":0.95}]', 1)
    assert out[0].standing and out[0].rule is None and out[0].confidence == 0.95


def test_parse_batch_verdict_string_shape():
    # deepseek-v4-flash returned {"verdict": "STANDING RULE"} instead of the spec.
    # Honor it, and default the absent confidence high (a crisp verdict).
    out = _parse_batch('[{"index":0,"verdict":"STANDING RULE"}]', 1)
    assert out[0].standing and out[0].confidence == 1.0


def test_parse_batch_verdict_oneoff_is_false():
    out = _parse_batch('[{"index":0,"verdict":"one-off"},{"index":1,"label":"not standing"}]', 2)
    assert not out[0].standing and not out[1].standing


def test_parse_batch_missing_index_uses_position():
    out = _parse_batch('[{"standing":true,"confidence":0.9,"rule":"x"}]', 1)
    assert out[0].standing and out[0].rule == "x"


def test_parse_batch_no_verdict_signal_stays_nonrouting():
    # an object with no standing/verdict/label signal keeps the safe default.
    out = _parse_batch('[{"index":0,"confidence":0.9}]', 1)
    assert not out[0].standing


def test_parse_batch_missing_and_out_of_range_default_nonrouting():
    # index 5 is invalid for n=2; both slots stay non-standing (nothing minted).
    out = _parse_batch('[{"index":5,"standing":true,"confidence":1.0,"rule":"x"}]', 2)
    assert all(not j.standing for j in out)


def test_parse_batch_garbage_is_safe():
    assert all(not j.standing for j in _parse_batch("not json at all", 3))


def test_confidence_is_clamped():
    raw = ('[{"index":0,"standing":true,"confidence":5,"rule":"x"},'
           '{"index":1,"standing":true,"confidence":-2,"rule":"y"}]')
    out = _parse_batch(raw, 2)
    assert out[0].confidence == 1.0 and out[1].confidence == 0.0


# ── batched call: empty short-circuits, errors degrade ──────────────────────

def test_judge_batch_empty_makes_no_call():
    stub = StubLLMClient(default="[]")
    assert judge_durability_batch(stub, []) == []
    assert stub.calls == []


def test_judge_batch_splits_into_subbatches():
    # 5 markers at batch_size=2 → three calls (2,2,1), globally index-aligned.
    judge = _EchoJudge()
    markers = [("style", f"marker {i}") for i in range(5)]
    out = judge_durability_batch(judge, markers, batch_size=2)
    assert judge.calls == [2, 2, 1]
    assert len(out) == 5 and all(j.standing for j in out)
    assert [j.index for j in out] == [0, 1, 2, 3, 4]
    assert [j.rule for j in out] == [f"marker {i}" for i in range(5)]


def test_judge_batch_subbatch_failure_is_isolated():
    # A judge that fails on the 2nd call: only that slice degrades, not the rest.
    class _FlakyJudge:
        def __init__(self):
            self.n = 0

        def complete(self, req):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("slice down")
            body = req.user.split("Markers:", 1)[1].rsplit("Return", 1)[0].strip()
            payload = json.loads(body)
            return json.dumps([{"index": m["index"], "standing": True,
                                "confidence": 0.9, "rule": m["statement"]} for m in payload])

    markers = [("style", f"m{i}") for i in range(4)]
    out = judge_durability_batch(_FlakyJudge(), markers, batch_size=2)
    assert [j.standing for j in out] == [True, True, False, False]  # 2nd slice degraded only


def test_judge_batch_error_degrades_to_non_standing():
    class Boom:
        def complete(self, req):
            raise RuntimeError("llm down")

    out = judge_durability_batch(Boom(), [("style", "x"), ("style", "y")])
    assert len(out) == 2 and all(not j.standing for j in out)


# ── routing modes ───────────────────────────────────────────────────────────

def test_route_lexical_parity():
    markers = [("style", "Use British spelling"),
               ("correction", "The meeting is Tuesday, not Monday")]
    out = route_decisions(markers, mode="lexical", llm=None, confidence_min=0.75)
    assert out[0].route is True            # style routes by kind
    assert out[1].route is False           # non-imperative correction one-off


def test_route_llm_confidence_threshold():
    raw = '[{"index":0,"standing":true,"confidence":0.8,"rule":"Never use Mongo"}]'
    hi = route_decisions([("rejection", "rejects mongo")], mode="llm",
                         llm=StubLLMClient(default=raw), confidence_min=0.75)
    assert hi[0].route and hi[0].text == "Never use Mongo"   # canonical, not raw
    lo = route_decisions([("rejection", "rejects mongo")], mode="llm",
                         llm=StubLLMClient(default=raw), confidence_min=0.90)
    assert not lo[0].route                                   # 0.8 < 0.90


def test_route_llm_verdict_shape_routes_with_statement_fallback():
    # end-to-end: judge returns the drifted {"verdict":...} shape, no rule/conf.
    # It must route (conf defaults high) and fall back to the raw statement text.
    stub = StubLLMClient(default='[{"index":0,"verdict":"STANDING RULE"}]')
    out = route_decisions([("rejection", "the user rejects MongoDB")], mode="llm",
                          llm=stub, confidence_min=0.75)
    assert out[0].route and out[0].text == "the user rejects MongoDB"


def test_route_fastpath_shortcuts_lexical_without_calling_llm():
    stub = StubLLMClient(default="[]")
    out = route_decisions([("style", "Use British spelling")], mode="llm_fastpath",
                          llm=stub, confidence_min=0.75)
    assert out[0].route and out[0].source_mode == "lexical_fastpath"
    assert stub.calls == []                                  # no LLM needed for a lexical hit


def test_route_fastpath_sends_ambiguous_to_llm():
    raw = '[{"index":0,"standing":false,"confidence":0.1,"rule":null}]'
    stub = StubLLMClient(default=raw)
    out = route_decisions([("rejection", "rejects the LOWER() patch")],
                          mode="llm_fastpath", llm=stub, confidence_min=0.75)
    assert not out[0].route                                  # LLM says one-off
    assert len(stub.calls) == 1                              # the ambiguous marker was judged


def test_route_llm_without_client_is_precision_safe():
    # mode wants an LLM but none supplied → mint nothing rather than guess.
    out = route_decisions([("rejection", "rejects mongo")], mode="llm",
                          llm=None, confidence_min=0.5)
    assert not out[0].route


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        route_decisions([("style", "x")], mode="bogus", llm=None, confidence_min=0.5)
