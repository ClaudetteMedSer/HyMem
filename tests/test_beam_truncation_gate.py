"""B2 v0.2 §3: three classes, and the one in the middle must not be counted.

B2 v0.1 voided because a judge scored a row 1.0, wrote a long explanation, and
ran out of tokens mid-sentence. The regex found no closing brace and returned
the sentinel {"score": 0.0, "scores": []}.

The first draft of the fix said: that's judge variation, count it as a changed
row. That was WRONG. The 0.0 is fabricated -- neither the judge's actual score
nor a real 0.0 -- so counting it puts a parse artifact into the statistic as a
measurement, which is the exact class the gate exists to exclude, arriving
through the back door with the gate's blessing.

Correct: the plumbing is fine, so it must not VOID; the row is unreadable, so
it must not COUNT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import beam_adapter as ba  # noqa: E402

TRUNCATED = ('{"scores": [1], "total_score": 1.0, "explanation": "The response '
             'includes numeric error status codes')


def test_the_b2_row_is_recognised_as_truncation():
    """The actual reply that voided B2 v0.1."""
    assert ba.is_truncation(TRUNCATED, "length") is True


def test_finish_reason_alone_is_not_the_separator():
    """Both failure shapes carry "length". Gating on it alone would call a
    broken pin a long explanation."""
    assert ba.is_truncation("", "length") is False
    assert ba.is_truncation("   ", "length") is False


def test_the_v4_flash_trap_is_not_classed_as_truncation():
    """Empty content + length is the trap. It arrives here already rewritten by
    the falsy raise into an [LLM_ERROR string, and belongs in the explicit
    bucket -- not excused as a long explanation."""
    assert ba.is_truncation("[LLM_ERROR: empty content (finish=length, ...)]",
                            "length") is False


def test_a_complete_stop_is_not_truncation():
    assert ba.is_truncation('{"scores": [1]}', "stop") is False


def test_an_unknown_finish_reason_is_not_truncation():
    """Artifacts predating finish_reason capture record None. Unknown must not
    silently become excusable -- absence is not evidence of truncation."""
    assert ba.is_truncation(TRUNCATED, None) is False


def test_the_ceiling_is_five_percent_of_the_sample():
    """Above it, too much of the sample is unreadable to say anything about
    the rest -- report the rate, do not interpret."""
    assert ba.TRUNCATION_CEILING == 8


def test_a_truncated_row_never_becomes_a_counted_score(captured):
    """The end-to-end property. judge_answer hands back a FABRICATED 0.0 for a
    truncated reply; nothing downstream may treat it as the judge's verdict."""
    captured["reply"] = {"choices": [{"message": {"content": TRUNCATED},
                                      "finish_reason": "length"}]}
    llm = ba.LLMClient("deepseek-chat", "k")
    out = ba.judge_answer(llm, "q", "ideal", ["r"], "a", return_raw=True)
    assert out["score"] == 0.0 and out["scores"] == []       # the fabrication
    assert '"scores": [1]' in out["judge_raw"]               # what was really said
    assert ba.is_truncation(out["judge_raw"], out["judge_finish_reason"])


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class _Resp:
        def __init__(self, p):
            self._p = p

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["body"] = json
        return _Resp(seen["reply"])

    monkeypatch.setattr(ba.http, "post", fake_post)
    return seen
