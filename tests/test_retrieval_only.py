"""A cheap run that must describe the expensive one exactly.

`--retrieval-only` exists to measure `f` without paying for 500 reader calls
and 500 judge calls. The whole construction rests on one property: the prompt
it fingerprints must be byte-identical to the prompt a full run would have
sent. If the two paths render separately they will drift, and the cheap
measurement will describe a run nobody ever performs.

So the prompt is built in ONE place and both paths go through it, and that is
what these tests pin -- along with the two structural guards: the reader and
judge cannot be reached at all under the flag, and every scorer refuses the
verdict-free artifact it produces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import longmemeval_adapter as lme  # noqa: E402
import run_registry as rr  # noqa: E402


class _FakeLLM:
    def __init__(self):
        self.messages = None
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        self.messages = messages
        return "an answer"


def mem(content, mtype="episode"):
    return {"type": mtype, "content": content}


# ------------------------------------------- the one load-bearing property

@pytest.mark.parametrize("kw", [
    {},
    {"ability": "MR"},
    {"ability": "TR", "question_date": "2026-09-03"},
    {"ability": "PF", "permissive_default": True},
    {"distilled": ["a kept line"], "narrative_facts": ["a fact"]},
])
def test_the_cheap_path_fingerprints_exactly_what_the_full_path_sends(kw):
    """Across every branch that swaps the system prompt or the memory list."""
    memories = [mem("alpha"), mem("beta"),
                {"type": "message_hit", "content": "[user] gamma"}]
    llm = _FakeLLM()
    _, sha = lme.answer_question_raw(llm, memories, "Q?", **kw)
    built = lme.build_answer_messages(memories, "Q?", **kw)
    assert llm.messages == built, "the two paths rendered different prompts"
    assert sha == lme.context_sha(built)


def test_the_prompt_is_rendered_in_exactly_one_place():
    """Two renderings that agree today are two renderings that can diverge."""
    src = (_BENCH / "longmemeval_adapter.py").read_text()
    assert src.count("def _answer_messages(") == 1
    assert src.count("messages = [\n") == 1, \
        "only _answer_messages may assemble the reader's message list"


# ------------------------------------------------------ the reader is gone

def test_the_poison_client_raises_rather_than_answering():
    """A flag that merely skips a branch can be routed around by a later
    refactor, and the failure would be silent and expensive."""
    with pytest.raises(AssertionError, match="retrieval-only"):
        lme.PoisonLLM("judge").chat([{"role": "user", "content": "x"}])


def test_the_poison_client_names_the_path_it_guards():
    with pytest.raises(AssertionError, match="reader"):
        lme.PoisonLLM("reader").chat([])


def test_the_counter_counts_and_still_delegates():
    inner = _FakeLLM()
    c = lme.CountingLLM(inner)
    assert c.chat([{"role": "user", "content": "abcd"}]) == "an answer"
    assert c.calls == 1 and c.prompt_chars == 4
    assert inner.calls == 1, "the real client still did the work"


def test_the_counter_passes_through_attributes_of_the_real_client():
    """The run summary reads call_count and total_tokens off whatever client
    it was handed."""
    inner = _FakeLLM()
    inner.call_count = 7
    assert lme.CountingLLM(inner).call_count == 7


# ---------------------------------------------- the artifact cannot be scored

def test_a_retrieval_only_artifact_is_recognised_from_its_config():
    assert rr.is_retrieval_only({"config": {"retrieval_only": True}})


def test_it_is_recognised_from_its_rows_when_the_config_is_missing():
    """An artifact hand-edited, or written before the flag existed, must not
    slip through on a missing key."""
    assert rr.is_retrieval_only(
        {"per_question": [{"retrieval_only": True}, {"retrieval_only": True}]})


def test_a_normal_artifact_is_not_mistaken_for_one():
    assert not rr.is_retrieval_only(
        {"config": {}, "per_question": [{"correct": True}]})


def test_an_empty_artifact_is_not_retrieval_only():
    """`all()` over no rows is True, which would make every empty artifact
    unscoreable for the wrong reason."""
    assert not rr.is_retrieval_only({"per_question": []})
    assert not rr.is_retrieval_only({})


def test_a_partly_marked_artifact_is_not_treated_as_retrieval_only():
    """Mixed rows are a corrupt artifact, not a cheap run; the refusal here
    would hide that rather than surface it."""
    assert not rr.is_retrieval_only(
        {"per_question": [{"retrieval_only": True}, {"correct": True}]})


def test_the_adapter_and_the_registry_do_not_keep_two_definitions():
    assert lme.is_retrieval_only({"config": {"retrieval_only": True}}) is True
    src = (_BENCH / "longmemeval_adapter.py").read_text()
    assert "from run_registry import is_retrieval_only as _impl" in src


# ------------------------------------------------------------------- the flag

def test_the_flag_exists_and_defaults_off():
    import argparse
    import assert_arm
    ns = assert_arm.parse_adapter_argv(["--scales", "S"])
    assert ns["retrieval_only"] is False
    on = assert_arm.parse_adapter_argv(["--scales", "S", "--retrieval-only"])
    assert on["retrieval_only"] is True
    assert isinstance(argparse.ArgumentParser(), argparse.ArgumentParser)


def test_distillation_gets_its_own_client_so_the_reader_can_be_poisoned():
    """Distillation is part of RETRIEVAL and genuinely fires under the flag,
    so it cannot share a client with the reader — otherwise the reader's
    client has to stay real and "no reader call" is only ever a property of
    an if-statement."""
    src = (_BENCH / "longmemeval_adapter.py").read_text()
    assert "distill_llm or llm, question, memories" in src
    assert 'answer_llm = PoisonLLM("reader")' in src
    assert 'judge_llm = PoisonLLM("judge")' in src


# ------------------------- every scorer refuses the verdict-free artifact

def _ro_pair():
    def art(lever, shas):
        return {"config": {"episode_granularity_enabled": lever,
                           "retrieval_only": True},
                "per_question": [
                    {"question_id": f"q{i}", "correct": None,
                     "retrieval_only": True, "context_sha": s,
                     "question_type": "multi-session", "n_episodes": 5}
                    for i, s in enumerate(shas)]}
    return art(False, ["a", "b", "c"]), art(True, ["a", "B", "C"])


def test_guard_score_refuses_a_retrieval_only_artifact():
    """It has 500 rows and no verdicts. `accuracy` would read every row as
    unscored and print 0.0, and a 0% arm beside a 69% arm looks like a
    catastrophic regression rather than a category error."""
    import guard_score as gs
    a, b = _ro_pair()
    lines: list[str] = []
    verdict, detail = gs.report(a, b, out=lines.append)
    text = "\n".join(lines)
    assert verdict == "INCOMPLETE"
    assert detail["retrieval_only"] == ["A", "B"]
    assert "REFUSED" in text
    assert "VERDICT" not in text and "NO REGRESSION" not in text


def test_guard_score_refuses_when_only_ONE_arm_is_retrieval_only():
    """The dangerous case: a real arm beside a verdict-free one is exactly
    the shape that would print a 0-versus-69 'regression'."""
    import guard_score as gs
    a, _ = _ro_pair()
    b = {"config": {"episode_granularity_enabled": True},
         "per_question": [{"question_id": f"q{i}", "correct": True,
                           "question_type": "multi-session"}
                          for i in range(3)]}
    lines: list[str] = []
    verdict, detail = gs.report(a, b, out=lines.append)
    assert verdict == "INCOMPLETE"
    assert detail["retrieval_only"] == ["A"]


def test_the_concentration_model_refuses_a_retrieval_only_artifact():
    import concentration_model as cm
    a, b = _ro_pair()
    b["config"]["episode_granularity_enabled"] = False   # same arm otherwise
    lines: list[str] = []
    res = cm.report(a, b, out=lines.append)
    assert res["refused"] is True
    assert "gain:" not in "\n".join(lines)


def test_churn_decompose_refuses_a_retrieval_only_artifact():
    import churn_decompose as cd
    a, b = _ro_pair()
    with pytest.raises(ValueError, match="no verdicts"):
        cd.decompose(a, b)
