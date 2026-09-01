r"""The judge's verdict and the parser's failure shared one column.

`re.search(r'\{[^}]+\}')` stops at the FIRST `}`. A judge explanation quoting
a brace -- code, JSON, a `${template}` literal -- produced a fragment,
json.loads raised, and the except path emitted a sentinel 0.0 that no reader
could tell from a real 0.0. Observed on a COMPLETE, valid, finish_reason=stop
reply in which the judge had written `scores: [1]`.

The fix must be strictly MORE PERMISSIVE, never different: anything the old
regex parsed must parse identically. That is the pre-registered falsifier and
it is pinned here rather than eyeballed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import beam_adapter as ba  # noqa: E402

# The reply that voided B2 v0.1 and v0.2, verbatim from the preserved artifact.
B2_REPLY = ('{"scores": [1], "total_score": 1.0, "explanation": "The response '
            "includes numeric status codes in the example 'Error "
            "${response.status}: ${response.statusText}' and mentions status "
            'codes in the summary, satisfying the criterion."}')


def naive(raw):
    """The old parser, kept verbatim so the falsifier compares against the
    real thing rather than a description of it."""
    m = re.search(r'\{[^}]+\}', raw.replace('\n', ' '))
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def test_the_row_that_voided_two_runs_is_recovered():
    obj, how = ba.extract_judge_json(B2_REPLY)
    assert how == "recovered"
    assert obj["scores"] == [1]
    assert naive(B2_REPLY) is None, "the old parser must still fail, or this proves nothing"


def test_the_recovered_row_now_scores_what_the_judge_said(monkeypatch):
    """End to end: 1.0, not the sentinel 0.0 it was recorded as."""
    class Stub:
        def chat(self, messages, temperature=None, max_tokens=None):
            return B2_REPLY

    out = ba.judge_answer(Stub(), "q", "ideal", ["states the code"], "a", return_raw=True)
    assert out["score"] == 1.0 and out["scores"] == [1]
    assert out["judge_parse"] == "recovered"


@pytest.mark.parametrize("raw", [
    '{"scores": [1, 0], "total_score": 0.5}',
    '{"scores": []}',
    '{"scores": [1]}  trailing prose',
    'preamble {"scores": [0]} postamble',
    '{"scores": [1, 1, 0]}\n',
])
def test_strictly_more_permissive_never_different(raw):
    """THE PRE-REGISTERED FALSIFIER. Every reply the old regex could read must
    read identically now. A different answer on an already-working input means
    the fix is wrong, not better."""
    old = naive(raw)
    new, how = ba.extract_judge_json(raw)
    assert old is not None, "fixture must be one the OLD parser handled"
    assert new == old
    assert how == "ok"


@pytest.mark.parametrize("raw", [
    "",
    "not json at all",
    "[LLM_ERROR: empty content (finish=length, reasoning=40 chars)]",
    '{"scores": [1], "explanation": "cut off mid',       # genuinely truncated
    '{"scores": [1', 
])
def test_an_unreadable_reply_stays_unreadable(raw):
    """The sentinel must survive for replies that really cannot be read --
    otherwise the fix trades a false 0.0 for a fabricated score, which is the
    same defect facing the other way."""
    obj, how = ba.extract_judge_json(raw)
    assert obj is None and how == "unreadable"


def test_a_brace_inside_a_string_does_not_end_the_object():
    obj, how = ba.extract_judge_json('{"scores": [1], "note": "a } inside"}')
    assert obj["scores"] == [1] and how == "recovered"


def test_an_escaped_quote_does_not_reopen_the_string():
    """\\" inside a string must not flip the in-string state, or the scan ends
    the object at the wrong brace."""
    obj, _ = ba.extract_judge_json(r'{"scores": [1], "note": "he said \"} \" ok"}')
    assert obj["scores"] == [1]


def test_nested_objects_are_matched_to_their_own_close():
    obj, how = ba.extract_judge_json('{"scores": [1], "meta": {"a": 1}}')
    assert obj["scores"] == [1] and obj["meta"] == {"a": 1}
    assert how == "recovered"


def test_the_first_complete_object_wins_not_the_last():
    obj, _ = ba.extract_judge_json('{"scores": [1]} {"scores": [0]}')
    assert obj["scores"] == [1]


def test_a_malformed_first_object_does_not_block_a_valid_later_one():
    """A stray unbalanced brace early in the prose must not swallow the real
    verdict."""
    obj, how = ba.extract_judge_json('{not json} then {"scores": [1]}')
    assert obj is not None and obj["scores"] == [1]
