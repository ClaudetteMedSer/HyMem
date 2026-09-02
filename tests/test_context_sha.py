"""The fingerprint that turns a bound into a partition.

`churn_decompose` attributes run-to-run answer churn to our retrieval or to
the provider's decoder by asking whether the reader was handed the same input.
The artifacts only recorded COUNTS (n_episodes, n_facts, ability_used, ...),
which cannot tell "the same 15 episodes" from "15 different ones" -- so the
split was a lower bound on retrieval and an upper bound on the decoder, the
same caveat `guard_score.fired_subset` carries.

`context_sha` hashes the rendered reader prompt, which closes the gap for
every future run at no extra call. These tests pin the two properties that
make it worth recording: it moves when the reader's input moves, and
`answer_question` keeps the exact contract three other adapters import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import longmemeval_adapter as lme  # noqa: E402


class _FakeLLM:
    """Records the messages it was handed and returns a fixed reply."""

    def __init__(self):
        self.messages = None

    def chat(self, messages, temperature=None, max_tokens=None):
        self.messages = messages
        return "the answer"


def mem(content, mtype="episode"):
    return {"type": mtype, "content": content}


def ask(memories, **kw):
    llm = _FakeLLM()
    ans, sha = lme.answer_question_raw(llm, memories, "Q?", **kw)
    return ans, sha, llm


def test_the_same_context_hashes_the_same():
    _, a, _ = ask([mem("alpha"), mem("beta")])
    _, b, _ = ask([mem("alpha"), mem("beta")])
    assert a == b


def test_different_episode_TEXT_at_the_same_COUNT_changes_the_hash():
    """The whole reason the hash exists. Both calls hand over two episodes,
    so every count field the artifact records is identical -- and the old
    fingerprint would have called these two runs the same context."""
    _, a, _ = ask([mem("alpha"), mem("beta")])
    _, b, _ = ask([mem("gamma"), mem("delta")])
    assert a != b


def test_episode_ORDER_changes_the_hash():
    """The reader sees a list, and position matters to it."""
    _, a, _ = ask([mem("alpha"), mem("beta")])
    _, b, _ = ask([mem("beta"), mem("alpha")])
    assert a != b


def test_the_question_is_in_the_hash():
    llm = _FakeLLM()
    a = lme.answer_question_raw(llm, [mem("alpha")], "first?")[1]
    b = lme.answer_question_raw(llm, [mem("alpha")], "second?")[1]
    assert a != b


def test_the_system_prompt_is_in_the_hash():
    """The MR branch swaps BOTH the system prompt and the memory list. A
    fingerprint over the user turn alone would call two different reader
    configurations identical."""
    a = lme.context_sha([{"role": "system", "content": "one"},
                         {"role": "user", "content": "same"}])
    b = lme.context_sha([{"role": "system", "content": "two"},
                         {"role": "user", "content": "same"}])
    assert a != b


def test_the_field_boundary_cannot_be_forged_by_moving_text_across_it():
    """Concatenating without a boundary would hash ("ab","c") and ("a","bc")
    identically, and those are different prompts."""
    a = lme.context_sha([{"role": "user", "content": "ab"},
                         {"role": "user", "content": "c"}])
    b = lme.context_sha([{"role": "user", "content": "a"},
                         {"role": "user", "content": "bc"}])
    assert a != b


@pytest.mark.parametrize("delim", ["\x00", ":"])
def test_a_field_containing_the_delimiter_cannot_forge_a_collision(delim):
    """Any pure delimiter scheme is injective only while no field contains
    the delimiter, and retrieved turns are arbitrary text. Both cases below
    are genuine collisions under `field + D + field + D`; neither may collide
    here. The colon case is the one that matters -- it is the character this
    implementation actually writes.

    Found by two mutations that dropped a separator and survived: the test
    that should have caught them passed on the role prefix alone, which meant
    the scheme had never been checked for injectivity at all."""
    a = lme.context_sha([{"role": "a", "content": "b"},
                         {"role": "c", "content": "d"}])
    b = lme.context_sha([{"role": f"a{delim}b", "content": f"c{delim}d"}])
    assert a != b


def test_a_length_prefix_cannot_be_forged_either():
    """`len:field` is only injective because the length is read before the
    colon. ("1", "2:x") and ("12", ":x") must stay distinct."""
    a = lme.context_sha([{"role": "1", "content": "2:x"}])
    b = lme.context_sha([{"role": "12", "content": ":x"}])
    assert a != b


def test_answer_question_still_returns_a_bare_string():
    """beam_adapter, locomo_adapter and msc_adapter all import this name and
    use the result as text. The split must be invisible to them."""
    llm = _FakeLLM()
    out = lme.answer_question(llm, [mem("alpha")], "Q?")
    assert isinstance(out, str) and out == "the answer"


def test_the_raw_variant_returns_the_answer_first():
    ans, sha, _ = ask([mem("alpha")])
    assert ans == "the answer"
    assert len(sha) == 64 and set(sha) <= set("0123456789abcdef")


def test_the_hash_covers_what_the_llm_was_actually_handed():
    """Not a re-derivation that could drift from the real prompt."""
    _, sha, llm = ask([mem("alpha")])
    assert sha == lme.context_sha(llm.messages)
