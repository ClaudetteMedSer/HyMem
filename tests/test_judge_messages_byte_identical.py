"""Byte-equality pin for the judge message construction.

The fixture was captured from UNMODIFIED judge_answer (2026-08-31, before the
return_raw refactor) via a stub LLM on two REAL rows: one pool row (KU, ideal =
the real gold from a dataset reparse) and one control row (ABS, ideal = the
legacy ideal_answer — byte-identical to what A prompted). The refactor must
not change the bytes: "callers unchanged" is a claim about call sites, the
risk is in the construction. This test asserts both deep equality and
json-dump byte equality against the pre-refactor capture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
from beam_adapter import _judge_messages, judge_answer  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "data" / "judge_messages_fixture.json"


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def test_refactored_construction_is_byte_identical(fixture):
    for item in fixture:
        inp = item["inputs"]
        got = _judge_messages(inp["question"], inp["ideal"], inp["rubric"], inp["answer"])
        expected = item["messages"]
        # Deep equality catches structural drift…
        assert got == expected, f"{item['label']}: messages differ"
        # …byte equality catches encoding / whitespace / f-string drift.
        a = json.dumps(got, ensure_ascii=False)
        b = json.dumps(expected, ensure_ascii=False)
        assert a == b, f"{item['label']}: json bytes differ"


def test_judge_answer_default_callers_unchanged(fixture):
    # return_raw defaults False → exact old payload (no judge_raw key).
    for item in fixture:
        inp = item["inputs"]

        class Stub:
            def chat(self, messages, temperature=None, max_tokens=None):
                return '{"scores": [1], "total_score": 1.0}'

        res = judge_answer(Stub(), inp["question"], inp["ideal"], inp["rubric"], inp["answer"])
        assert res == {"score": 1.0, "scores": [1]}, f"{item['label']}: default shape changed"


def test_return_raw_adds_judge_raw(fixture):
    item = fixture[0]
    inp = item["inputs"]

    class Stub:
        def chat(self, messages, temperature=None, max_tokens=None):
            return '{"scores": [1], "total_score": 1.0}'

    res = judge_answer(Stub(), inp["question"], inp["ideal"], inp["rubric"], inp["answer"],
                       return_raw=True)
    assert res["score"] == 1.0 and res["scores"] == [1]
    assert res["judge_raw"] == '{"scores": [1], "total_score": 1.0}'


def test_return_raw_on_parse_failure_keeps_signature(fixture):
    # The silent-0 shape: parse failure must still return {"score": 0.0, "scores": []}
    # PLUS judge_raw so the rejudge can distinguish it from a real 0.0.
    item = fixture[0]
    inp = item["inputs"]

    class Stub:
        def chat(self, messages, temperature=None, max_tokens=None):
            return "not json at all"

    res = judge_answer(Stub(), inp["question"], inp["ideal"], inp["rubric"], inp["answer"],
                       return_raw=True)
    assert res["score"] == 0.0 and res["scores"] == []
    assert res["judge_raw"] == "not json at all"
