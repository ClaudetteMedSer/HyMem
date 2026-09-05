"""Characterisation tests for `longmemeval_adapter.judge_answer` — the SINGLE
function that produces every canonical benchmark number in this repo.

Why this file exists
--------------------
`benchmarks/locomo_adapter.py:421` and `benchmarks/msc_adapter.py:502` both
import `judge_answer` from `longmemeval_adapter`. One function therefore scores
LoCoMo (68.2%), LME (68.4%) and MSC (~84.0%). It is two lines::

    raw = llm.chat(messages, temperature=0.0, max_tokens=10)
    return parse_judge_verdict(raw)

Until 2026-08-25 the second line was `return "yes" in raw.lower()`, and most of
this file was written to pin that. See "The 2026-08-25 parse fix" below for what
changed and what deliberately did not.

(BEAM is NOT in the blast radius: `beam_adapter.py:725` defines its own
rubric-scoring `judge_answer` returning a dict. Pinned below so a future reader
does not over-scope a re-baseline.)

These are CHARACTERISATION tests, not a wish-list. They pin what the function
does TODAY so that a fix — which re-baselines all three numbers at once — is a
deliberate, visible act rather than a silent drift. Where the behaviour is
correct it is pinned as correct. Where it is wrong the test name starts with
`test_DEFECT_` and the docstring names the defect.

Nothing here spends an LLM token: every judge reply is canned through the house
`StubLLMClient` (hymem.extraction.llm), adapted to the benchmark's `.chat()`
shape. No live client is ever constructed.

What is pinned, in one paragraph
--------------------------------
1. The DECISION RULE is `parse_judge_verdict`: word-boundary `\byes\b` /
   `\bno\b`, first verdict token wins, negated affirmatives and the
   `[LLM_ERROR: ...]` sentinel score `False`, no verdict token scores `False`.
   It replaced `"yes" in raw.lower()` — an unanchored substring test that was
   correct on compliant replies and on the empty reply, and wrong on "not yes"
   and on any reply containing "yes" INSIDE another word ("yesterday", "eyes").
   "yesterday" was not a hypothetical: temporal-reasoning is a scored category
   on both LME and LoCoMo, and the judge\'s own reasoning about it is exactly
   where that word appears. The old rule never consulted the "no" half of a
   reply, so every conflict resolved to correct.
2. The CRITERION for the 5 non-abstention branches is CONTAINMENT ("answer yes
   if the response contains the correct answer"). The `_abs` branch asks a
   different question ("does the model correctly identify the question as
   unanswerable"). The asymmetry matters because the containment instructions
   are BYTE-IDENTICAL for a committed answer and for a hedged refusal that
   recites the gold value — pinned in
   `test_DEFECT_containment_instructions_cannot_separate_a_refusal_from_an_answer`.
   Whether a real judge exploits that latitude is an LLM-behaviour question this
   offline file cannot answer; `benchmarks/judge_audit.py` is the instrument
   that measures it, and its thresholds are pre-registered in its docstring.
3. A judge-side `[LLM_ERROR: ...]` sentinel is scored as "wrong answer" —
   now BY CONSTRUCTION rather than by luck, but still indistinguishable from a
   genuine "no" at the call site. No caller checks it; the two re-judge paths
   check only the ANSWER for that sentinel. Containment was fixed; VISIBILITY
   was not, and that half is still pinned as a defect below.
4. `judge_answer` returns a bool and DISCARDS `raw`. That is precisely why
   points 1 and 3 went uncounted until `benchmarks/judge_audit.py` recorded
   `raw` on the 2026-08-25 LME run.

The 2026-08-25 parse fix
------------------------
The audit\'s spend pass recorded all 500 raw judge replies and measured C2
non-compliance = 0.00%: every reply was a bare yes/no. That made the anchored
rule a PROVEN no-op on the only corpus of real judge replies in existence, so it
was landed in that window rather than at the next judge migration — when the
rule and the data would otherwise change in the same step. `judge_audit.py
--verify-parse` is the standing check, and it must report zero flips.

Two shapes were deliberately NOT changed, and both are pinned so the choice
cannot be quietly revisited:

  * "yes and no" still scores correct (first token wins). Resolving a hedging
    judge to `False` decides what it MEANT — a criterion question (D1, open at
    WATCH), not a parse question.
  * A truncated fragment carrying the bare word ("...whether a yes would be")
    still scores correct. Separating it needs the reply\'s structure, and
    `max_tokens=10` is part of the frozen comparability contract.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest

from hymem.extraction.llm import LLMRequest, StubLLMClient

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))


def _install_offline_requests_shim() -> None:
    """`longmemeval_adapter` does `import requests as http` at module scope, and
    this interpreter has no `requests` (the same reason test_fact_probe,
    test_raptor_cluster_probe and test_rerank_ab do not collect).

    The shim is deliberately HOSTILE: every attribute access raises. The judge
    path must never touch the network, and if it ever starts to, these tests
    must fail loudly rather than quietly exercise a live client. If `requests`
    is genuinely installed we leave it alone and the real module is used.
    """
    if "requests" in sys.modules:
        return
    try:  # pragma: no cover - environment-dependent
        import requests  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType("requests")

    def _forbidden(name: str):
        raise AssertionError(
            f"the judge path touched the network (requests.{name}) — "
            "these tests must stay offline"
        )

    shim.__getattr__ = _forbidden  # type: ignore[attr-defined]
    sys.modules["requests"] = shim


_install_offline_requests_shim()

from longmemeval_adapter import (  # noqa: E402
    get_judge_prompt,
    judge_answer,
    parse_judge_verdict,
)

# The audit instrument. Imported at the top rather than beside its own section
# because the decision-rule tests join against it: several of them assert that
# the FROZEN legacy rule still disagrees with the landed one, which is what
# stops `shipping_verdict` being re-synced and the before/after diff going
# vacuous. Only stdlib is pulled in at its import time.
import judge_audit as JA  # noqa: E402

# Every question_type the judge prompt knows. Sourced from the branches of
# `get_judge_prompt`; the `_abs` variants are the abstention route.
_NON_ABS_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
)


class StubJudgeClient:
    """The benchmark `LLMClient` shape (`.chat(messages, temperature, max_tokens)`)
    backed by the house `StubLLMClient`.

    Written as an adapter rather than a fresh fake so the canned-response
    semantics are the sanctioned ones (fixture keys matched by substring against
    system + user, `default` for the rest). Records the exact call arguments so
    the temperature / max_tokens comparability contract can be pinned.
    """

    def __init__(self, reply: str = "", fixtures: dict[str, str] | None = None):
        self._stub = StubLLMClient(fixtures=fixtures or {}, default=reply)
        self.calls: list[dict] = []

    def chat(self, messages: list, temperature: float = 0.1,
             max_tokens: int = 1024) -> str:
        self.calls.append({"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens})
        system = "".join(m["content"] for m in messages if m["role"] == "system")
        user = "".join(m["content"] for m in messages if m["role"] == "user")
        return self._stub.complete(
            LLMRequest(system=system, user=user, response_format="text",
                       max_tokens=max_tokens, temperature=temperature)
        )


def _judge(reply: str, question_type: str = "single-session-user") -> bool:
    return judge_answer(StubJudgeClient(reply), question_type,
                        "What is her personal best?", "3:42", "irrelevant")


# ── 0. The stub itself binds ────────────────────────────────────────────
# A canned-reply harness that silently ignored its canned reply would make
# every test below vacuous — the whole file would pass on a constant. Prove the
# reply reaches the function and that different replies produce different
# verdicts before asserting anything about the verdicts.

def test_stub_reply_actually_reaches_judge_answer():
    """Anti-vacuity floor for this entire file."""
    client = StubJudgeClient("yes")
    assert judge_answer(client, "single-session-user", "q", "a", "r") is True
    assert len(client.calls) == 1, "judge_answer did not call the client"
    assert judge_answer(StubJudgeClient("no"), "single-session-user",
                        "q", "a", "r") is False, \
        "verdict did not depend on the canned reply — every test here is vacuous"


# ── 1. The decision rule: `parse_judge_verdict` (was `"yes" in raw.lower()`) ─

@pytest.mark.parametrize("reply", ["yes", "Yes", "YES", "Yes.", "yes\n",
                                   "  \n\t yes", "**Yes**", '"yes"', "- Yes"])
def test_compliant_affirmatives_score_correct(reply):
    """CORRECT, pinned as correct. `.lower()` plus an unanchored `in` absorbs
    case, leading whitespace, punctuation and markdown decoration — the whole
    class of benign non-compliance a real judge produces."""
    assert _judge(reply) is True


@pytest.mark.parametrize("reply", ["no", "No", "NO", "No.", "no\n",
                                   "  no", "**No**",
                                   "No, the response does not contain it."])
def test_compliant_negatives_score_incorrect(reply):
    """CORRECT, pinned as correct. Note the last case: under the LEGACY rule a
    verbose "no" survived only because it happened to contain no "yes" anywhere.
    Since 2026-08-25 it is correct by construction — the same shape carrying a
    "yesterday" now also scores no, which is the fix directly below."""
    assert _judge(reply) is False


def test_empty_reply_scores_incorrect():
    """CORRECT, pinned as correct — and the safe direction. An empty judge reply
    fails closed (scores the answer wrong) rather than open."""
    assert _judge("") is False


def test_UNFIXED_ambiguous_yes_and_no_still_scores_correct():
    """DELIBERATELY UNCHANGED by the 2026-08-25 parse fix, and pinned so the
    choice cannot be quietly revisited.

    A judge that answers "yes and no" has declined to commit, and first-token-wins
    credits the answer. Scoring it `False` is a defensible reading of the judge
    instruction ("Answer yes or no only"), but it decides what a non-committal
    judge MEANT — a CRITERION question, which is D1, still open at WATCH. The
    parse fix was landed on the strength of being a proven no-op on the recorded
    corpus; folding an unmeasured criterion decision into it would have spent
    exactly the property that made it safe to land."""
    assert _judge("yes and no") is True
    assert _judge("Yes and no — it depends.") is True


def test_FIXED_negated_yes_scores_incorrect():
    """WAS a defect: the legacy rule read the token and never the polarity, so a
    NEGATED affirmative inverted the judge's meaning and scored CORRECT."""
    assert _judge("not yes") is False
    assert _judge("NOT YES") is False
    assert _judge("The correct answer here is not yes.") is False
    assert _judge("that is never a yes") is False


def test_the_negation_rule_does_not_fire_on_a_judge_that_said_yes():
    """The negation rule is the one part of the fix that can create a NEW
    misscore, in the opposite direction, so it is bounded and controlled here.

    `judge_audit._NEGATED_YES` is deliberately looser (`[^.]{0,20}?` between the
    negation and the token) because it is a COUNTER: over-matching inflates a
    bucket already reported as a lower bound, which is conservative. A DECISION
    rule cannot be loose — over-matching silently marks a correct answer wrong —
    so the landed regex requires the negation to sit ADJACENT to the token.

    This test is the join proving the two really differ. If someone re-syncs the
    decision rule to the audit's regex "for consistency", it fails."""
    hedged_yes = "it is not incorrect, yes"
    assert _judge(hedged_yes) is True, \
        "the negation rule flipped a reply whose verdict was yes"
    assert JA._NEGATED_YES.search(hedged_yes), \
        "the audit's counter no longer over-matches — the asymmetry this test " \
        "documents is gone and the tightening is no longer motivated"


@pytest.mark.parametrize("reply", [
    "The model refers to yesterday, so no.",
    "No — the response only mentions what she did yesterday.",
    "no, the model just describes eyes",
])
def test_FIXED_substring_rule_now_has_a_word_boundary(reply):
    """The highest-exposure defect, and the one that was NOT hypothetical.

    `"yes" in raw.lower()` was unanchored, so "yesterday" and "eyes" fired it.
    Each reply here is an unambiguous NO from the judge and each scored CORRECT.

    Why this was not a curiosity: temporal-reasoning is a scored category on both
    LME (`temporal-reasoning`) and LoCoMo (`category 2`), and its judge prompt
    invites day-arithmetic reasoning ("do not penalize off-by-one errors for the
    number of days"). "yesterday" is the single likeliest word in a
    non-compliant temporal judge reply, and it was a false CORRECT every time.
    """
    assert _judge(reply) is False
    assert JA.shipping_verdict(reply) is True, \
        "the frozen legacy rule no longer scores this correct — the baseline " \
        "copy has been re-synced and the before/after diff is now vacuous"


def test_FIXED_the_no_half_of_a_reply_is_now_consulted():
    """The legacy rule had no "no" test at all, so no conflict could be detected
    and every conflict resolved to CORRECT regardless of order. The first
    word-boundary verdict token now wins, which scores all three as the judge
    meant — including the ORDER-DEPENDENT pair, which is what proves the rule
    reads position and is not just returning `False` on anything ambiguous."""
    assert _judge("no\n\nyes") is False
    assert _judge("yes\n\nno") is True
    assert _judge("No. Well, actually, yes.") is False


@pytest.mark.parametrize("reply,expected", [
    # A reasoning preamble cut off at ~10 tokens, carrying no verdict at all.
    ("Let me analyse the model response against the correct", False),
    # The same shape, cut off just after a "yes" that is not the verdict.
    ("The question is whether a yes would be", True),
])
def test_DEFECT_truncated_reply_is_scored_as_a_verdict(reply, expected):
    """DELIBERATELY UNFIXED by the 2026-08-25 parse fix, and pinned as a defect.

    `max_tokens=10` truncates any non-compliant reply mid-sentence, and the
    fragment is then scored AS IF it were a verdict — silently, in whichever
    direction the fragment happens to fall. Word-boundary matching does not help
    here: the second case carries a real, bare "yes" that simply is not the
    verdict. Separating it needs the reply's STRUCTURE, and `max_tokens=10` is
    itself part of the frozen comparability contract, so the fix is a
    re-baseline rather than a parse change.

    This is the failure mode the deepseek-v4-flash migration already hit from
    the other side (a reasoning preamble corrupting the yes/no parse, worked
    around with `thinking: disabled` rather than by hardening the parse), so the
    exposure is documented as real for this judge family, not theoretical."""
    assert _judge(reply) is expected


def test_FIXED_a_judge_outage_is_now_distinguishable_from_a_genuine_no():
    """D3, CLOSED 2026-08-26. Was `test_DEFECT_..._still_invisible_to_the_caller`.

    Two halves, landed a day apart and for different reasons.

    CONTAINMENT (2026-08-25, rode along with the parse fix): the sentinel is
    rejected explicitly, so an outage message that happens to contain the word
    "yes" (`[LLM_ERROR: unexpected token 'yes']`) cannot be read as the judge
    saying the answer was CORRECT. Provably inert — 0 judge-side sentinels over
    500 recorded replies.

    VISIBILITY (2026-08-26): `judge_scored` returns `correct=None` for a
    sentinel — UNSCORED, not wrong — and every call site across the three
    adapters routes through it. `judge_answer` keeps its bare bool and its
    fail-closed `False`, unchanged, because it is the certified function; the
    channel is a sibling, not a modification.

    The old defect asserted `error_verdict == genuine_no`. THAT EQUALITY IS
    WHAT MOVED: at the scoring boundary they are now different values."""
    outage = "[LLM_ERROR: Connection reset by peer]"

    # The certified function is untouched: still a bool, still fail-closed.
    assert _judge(outage) is False
    assert _judge(outage) == _judge("no")

    # The channel the call sites actually use tells them apart.
    from longmemeval_adapter import judge_scored
    err_verdict, err_raw = judge_scored(StubJudgeClient(outage),
                                        "single-session-user", "q", "a", "r")
    no_verdict, no_raw = judge_scored(StubJudgeClient("no"),
                                      "single-session-user", "q", "a", "r")
    assert err_verdict is None and no_verdict is False
    assert err_verdict != no_verdict, (
        "an outage and a genuine 'no' are indistinguishable again — that "
        "indistinguishability WAS the defect"
    )
    assert err_raw == outage and no_raw == "no"

    # The half that was fixed first: rejected by construction, not by luck.
    assert _judge("[LLM_ERROR: unexpected token 'yes' in response]") is False
    assert JA.shipping_verdict("[LLM_ERROR: unexpected token 'yes' in response]") \
        is True, "the legacy rule scored this sentinel CORRECT — that is the " \
                 "hazard the explicit check removed"


def test_FIXED_the_raw_reply_survives_into_the_caller():
    """Was `test_DEFECT_raw_reply_is_discarded`, the reason judge_audit exists.

    `judge_answer` discarded `raw`, so the rate of non-compliant replies was not
    merely unmeasured but unmeasurABLE from any stored run — judge_audit had to
    re-judge 500 rows to measure something that had already been produced once
    and thrown away. `judge_answer_raw` returns it, the adapters persist it as
    `judge_raw`, and the next audit is free.

    The three replies are deliberately all COMPLIANT affirmatives differing only
    in formatting: that isolates the claim to the RETURN CHANNEL. Under the old
    bare-bool return all three collapsed to one value; now each survives
    verbatim while the verdict stays identical."""
    from longmemeval_adapter import judge_answer_raw

    replies = ["yes", "Yes.", "  YES  \n"]
    assert len(set(replies)) == 3
    got = [judge_answer_raw(StubJudgeClient(r), "single-session-user",
                            "q", "a", "r") for r in replies]
    assert [v for v, _raw in got] == [True, True, True]
    assert [raw for _v, raw in got] == replies, "raw reply is still discarded"
    # And the bool-returning wrapper still erases it, unchanged.
    assert len({_judge(r) for r in replies}) == 1


def test_judge_call_shape_is_the_comparability_contract():
    """CORRECT, pinned as correct — and pinned because it is frozen. Every
    canonical number was produced at temperature 0.0 / max_tokens 10 with a
    single user-role message and no system prompt. Changing any of these three
    re-baselines LoCoMo, LME and MSC simultaneously."""
    client = StubJudgeClient("yes")
    judge_answer(client, "single-session-user", "q", "a", "r")
    (call,) = client.calls
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 10
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"


# ── 2. The criterion: what the judge is actually asked ──────────────────

_CONTAINMENT = "Please answer yes if the response contains the correct answer."
_ABSTENTION = ("Please answer yes if the model correctly identifies the "
               "question as unanswerable.")


@pytest.mark.parametrize("qtype", _NON_ABS_TYPES)
def test_non_abstention_branches_all_use_a_containment_criterion(qtype):
    """Five of the six non-abstention branches instruct CONTAINMENT verbatim.
    The sixth (`single-session-preference`) is a rubric-satisfaction criterion —
    asserted separately so this parametrisation cannot pass by asserting nothing.
    """
    prompt = get_judge_prompt(qtype, "q", "gold", "resp")
    if qtype == "single-session-preference":
        assert "Please answer yes if the response satisfies the desired response." in prompt
        assert _CONTAINMENT not in prompt
    else:
        assert _CONTAINMENT in prompt


def test_abstention_branch_asks_a_different_and_correct_question():
    """The `_abs` branch is the one that gets it RIGHT: it asks whether the model
    identified the question as unanswerable, not whether some string is present.
    Pinned as correct, and pinned as DIFFERENT — the asymmetry is the finding."""
    abs_prompt = get_judge_prompt("single-session-user_abs", "q", "expl", "resp")
    assert _ABSTENTION in abs_prompt
    assert _CONTAINMENT not in abs_prompt
    assert "unanswerable" in abs_prompt

    plain_prompt = get_judge_prompt("single-session-user", "q", "gold", "resp")
    assert "unanswerable" not in plain_prompt
    assert abs_prompt != plain_prompt


def test_DEFECT_containment_instructions_cannot_separate_a_refusal_from_an_answer():
    """The core criterion defect, stated as something offline code CAN prove.

    Take one gold value and two model responses: a committed answer, and a hedged
    refusal that recites the gold while explicitly declining to answer. Render
    both judge prompts. Everything the judge is TOLD — the criterion, the
    question, the gold, the closing instruction — is byte-identical; the sole
    difference is the response being quoted. The criterion supplies no clause
    about the model committing to the value, so the judge is given no ground to
    separate them and is in fact instructed toward "yes" for both, because the
    refusal literally contains the correct answer.

    Note what this test does and does not establish. It proves the instructions
    are identical and that the criterion is containment. It does NOT measure how
    often a real judge scores such a refusal correct — that needs LLM spend and
    is `benchmarks/judge_audit.py`'s job, against pre-registered thresholds.

    Contrast the `_abs` branch, which for the same shape asks the RIGHT question
    and would score the refusal on its refusal, not on its recitation.
    """
    gold = "3:42"
    committed = "Her personal best is 3:42."
    refusal = ("I can't tell which of these is her personal best, though the "
               "context mentions 3:42.")

    p_committed = get_judge_prompt("single-session-user", "What is her PB?",
                                   gold, committed)
    p_refusal = get_judge_prompt("single-session-user", "What is her PB?",
                                 gold, refusal)

    # The recitation really is verbatim inside the refusal branch's prompt.
    assert gold in p_refusal.split("Model Response:")[1]

    # Strip the quoted response from each; what remains is what the judge is told.
    instructions_committed = p_committed.replace(committed, "<RESPONSE>")
    instructions_refusal = p_refusal.replace(refusal, "<RESPONSE>")
    assert instructions_committed == instructions_refusal, (
        "the judge receives different instructions for the two shapes — "
        "if this ever fails, the criterion gained a refusal clause and this "
        "defect is FIXED"
    )
    assert _CONTAINMENT in instructions_refusal

    # And the abstention branch, on the same pair, asks the discriminating question.
    p_abs = get_judge_prompt("single-session-user_abs", "What is her PB?",
                             gold, refusal)
    assert _ABSTENTION in p_abs


def test_unknown_question_type_raises():
    """CORRECT, pinned as correct: an unmapped type fails loudly rather than
    silently falling through to a default criterion. `match=` is required here —
    a bare `pytest.raises(NotImplementedError)` would pass on any unrelated
    NotImplementedError raised anywhere in the call."""
    with pytest.raises(NotImplementedError, match="Unknown question type"):
        get_judge_prompt("not-a-real-type", "q", "a", "r")


# ── 3. The blast radius: who routes through this function ───────────────
# The premise of the whole audit. Asserted from the SOURCE via ast, and every
# assertion is a positive existence check with a non-zero count, because an ast
# walk that finds nothing at all otherwise reads as a clean pass (this codebase
# has already shipped one import test with exactly that hole).

# The adapters call `judge_scored` since the 2026-08-26 D3 fix. That is still
# the SAME judge: judge_scored -> judge_answer_raw -> parse_judge_verdict, and
# `judge_answer` is now `judge_answer_raw(...)[0]`. What changed is the channel
# (a sentinel returns None = UNSCORED instead of False), not the decision rule.
# Both names are accepted so the premise survives the rename; the delegation
# chain that makes them equivalent is asserted separately below.
_JUDGE_ENTRYPOINTS = ("judge_answer", "judge_scored")


def _import_sources(path: Path, name: str) -> list[ast.ImportFrom]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and n.module == "longmemeval_adapter"
            and any(a.name == name for a in n.names)]


@pytest.mark.parametrize("adapter", ["locomo_adapter.py", "msc_adapter.py"])
def test_locomo_and_msc_import_the_lme_judge(adapter):
    """One function, three canonical baselines."""
    hits = [h for name in _JUDGE_ENTRYPOINTS
            for h in _import_sources(_BENCH / adapter, name)]
    assert len(hits) >= 1, (
        f"{adapter} no longer imports the LME judge entry point — "
        "the single-judge premise of this audit has changed"
    )


def test_the_judge_entry_points_all_reduce_to_one_decision_rule():
    """What keeps the blast-radius premise true across the D3 rename.

    Three names now exist, and they must remain ONE rule: `judge_scored` and
    `judge_answer` both delegate to `judge_answer_raw`, which parses with
    `parse_judge_verdict`. If a call site were ever given its own parse, the
    single-judge premise — and with it C1 = 0.00% covering all three
    benchmarks — would quietly stop holding."""
    from longmemeval_adapter import judge_answer_raw, judge_scored

    client = StubJudgeClient("yes")
    assert judge_answer(client, "single-session-user", "q", "a", "r") is True
    assert judge_answer_raw(StubJudgeClient("yes"),
                            "single-session-user", "q", "a", "r") == (True, "yes")
    assert judge_scored(StubJudgeClient("no"),
                        "single-session-user", "q", "a", "r") == (False, "no")
    # The channel, and the only behavioural difference between the two.
    verdict, raw = judge_scored(StubJudgeClient("[LLM_ERROR: reset by peer]"),
                                "single-session-user", "q", "a", "r")
    assert verdict is None and raw.startswith("[LLM_ERROR")
    assert judge_answer(StubJudgeClient("[LLM_ERROR: reset by peer]"),
                        "single-session-user", "q", "a", "r") is False


def test_beam_is_NOT_in_the_blast_radius():
    """Scope guard, so a re-baseline is not over-scoped. BEAM defines its own
    rubric judge returning a dict and imports nothing from the LME judge."""
    assert _import_sources(_BENCH / "beam_adapter.py", "judge_answer") == []
    tree = ast.parse((_BENCH / "beam_adapter.py").read_text(encoding="utf-8"))
    own = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "judge_answer"]
    assert len(own) == 1, "beam_adapter no longer defines its own judge"


def _judge_call_type_args(path: Path) -> list[str | None]:
    """The 2nd positional arg (`question_type`) of every judge call, as a
    literal where it is one and None where it is computed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str | None] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _JUDGE_ENTRYPOINTS
                and len(node.args) >= 2):
            arg = node.args[1]
            out.append(arg.value if isinstance(arg, ast.Constant) else None)
    return out


def test_msc_routes_every_question_through_the_containment_criterion():
    """MSC's exposure is TOTAL, not partial: it hard-codes `single-session-user`
    at its only judge call site, so no MSC row ever reaches the correct-looking
    `_abs` branch. The ~84.0% baseline is 100% containment-judged."""
    args = _judge_call_type_args(_BENCH / "msc_adapter.py")
    assert args, "no judge call found in msc_adapter.py"
    assert set(args) == {"single-session-user"}


def test_locomo_category_to_judge_type_mapping_is_pinned():
    """Only LoCoMo category 5 reaches the `_abs` branch; categories 1-4 are all
    containment-judged."""
    from locomo_adapter import CATEGORY_JUDGE

    assert CATEGORY_JUDGE == {
        1: "multi-session",
        2: "temporal-reasoning",
        3: "single-session-user",
        4: "single-session-user",
        5: "single-session-user_abs",
    }
    abs_types = [c for c, t in CATEGORY_JUDGE.items() if t.endswith("_abs")]
    assert abs_types == [5]


# ── 4. The audit instrument itself (benchmarks/judge_audit.py) ──────────
# `judge_audit.py` is the thing that will eventually be pointed at real runs and
# real tokens. An unvalidated classifier is the degeneracy trap in its purest
# form: a broken refusal detector returns a confident 0.0% and reads as "the
# criterion is safe". Everything below runs offline, and the end-to-end test
# exists specifically to prove the spend path is REACHABLE — an unreachable code
# path also reads as PASS.


@pytest.mark.parametrize("raw,bucket,ship,ref", [
    ("yes", "compliant_yes", True, True),
    ("No.", "compliant_no", False, False),
    ("", "empty", False, False),
    ("[LLM_ERROR: boom]", "llm_error", False, False),
    ("yes and no", "both_tokens", True, True),
    ("no\n\nyes", "both_tokens", True, False),
    # BOTH rules score this correct — "not" is not a word-boundary "no", so the
    # disagreement column is blind to negation. Hence C1 is a LOWER bound and
    # negation is counted separately as C1b.
    ("not yes", "negated_yes", True, True),
    ("The model refers to yesterday, so no.", "yes_substring_only", True, False),
    ("the model just describes eyes", "yes_substring_only", True, False),
    ("Let me analyse the model response against", "no_verdict_token", False, False),
])
def test_reply_classifier_buckets_and_rules(raw, bucket, ship, ref):
    """The classifier must reproduce `judge_answer`'s verdict EXACTLY (`ship`),
    because the audit's headline number is the disagreement between that and the
    reference rule. If `ship` ever drifts from the real function the audit
    measures its own bug."""
    c = JA.classify_reply(raw)
    assert c["bucket"] == bucket
    assert c["shipping"] is ship
    assert c["reference"] is ref
    assert c["disagrees"] is (ship != ref)


_ALL_SHAPES = ["yes", "No.", "", "[LLM_ERROR: boom]",
               "yes and no", "not yes", "no\n\nyes",
               "the model just describes eyes",
               "Let me analyse the model response against"]


@pytest.mark.parametrize("raw", _ALL_SHAPES)
def test_audit_landed_rule_matches_judge_answer_exactly(raw):
    """Cross-check against the REAL function rather than against a restatement
    of it. This is the join that stops the audit drifting from its subject.

    It moved from `shipping_verdict` to `landed_verdict` on 2026-08-25: the
    audit now carries TWO rules, and only one of them is supposed to track
    production."""
    assert JA.landed_verdict(raw) is _judge(raw)


@pytest.mark.parametrize("raw,legacy", [
    ("yes", True), ("No.", False), ("", False),
    # Every shape the legacy rule got wrong. If any of these stops reading True
    # the frozen baseline has been re-synced to the live parse.
    ("[LLM_ERROR: boom yes]", True), ("not yes", True), ("no\n\nyes", True),
    ("the model just describes eyes", True),
    ("The model refers to yesterday, so no.", True),
])
def test_the_legacy_rule_stays_frozen_at_the_pre_fix_behaviour(raw, legacy):
    """`shipping_verdict` is the historical baseline: it is what actually scored
    LoCoMo 68.2%, LME 68.4% and MSC ~84.0%, and `--verify-parse` diffs the live
    rule against it over stored replies.

    Re-syncing it to `parse_judge_verdict` "so the audit matches production"
    would make that diff a constant zero BY CONSTRUCTION — a ceiling instrument
    reporting a clean no-op it can no longer fail to report. This test is the
    only thing standing between a well-meant tidy-up and a silent re-baseline."""
    assert JA.shipping_verdict(raw) is legacy


def test_the_two_audit_rules_actually_differ():
    """Anti-vacuity for the pair above: if legacy and landed agreed everywhere,
    both tests would pass on a single rule and `--verify-parse` would be
    measuring nothing."""
    differ = [r for r in _ALL_SHAPES
              if JA.shipping_verdict(r) is not JA.landed_verdict(r)]
    assert differ, "legacy and landed rules agree on every shape — one of them " \
                   "has been re-synced to the other"


def test_reference_rule_disagrees_on_exactly_the_defect_shapes():
    """Anti-vacuity: a reference rule identical to the shipping rule would make
    the audit's headline a constant 0.0%. Prove it separates, and prove it does
    NOT separate on compliant replies (which would inflate the rate instead)."""
    defective = ["no\n\nyes", "The model refers to yesterday, so no.",
                 "the model just describes eyes"]
    benign = ["yes", "Yes.", "no", "No.", "", "yes and no", "**Yes**"]
    assert all(JA.classify_reply(r)["disagrees"] for r in defective)
    assert not any(JA.classify_reply(r)["disagrees"] for r in benign)

    # The documented BLIND SPOT, pinned so it cannot be forgotten: a negated
    # affirmative is misscored by the shipping rule and the reference rule does
    # not catch it either. It is reported as C1b, additive to C1.
    blind = JA.classify_reply("not yes")
    assert blind["disagrees"] is False
    assert blind["negated"] is True


def test_refusal_classifier_binds_on_the_phrase_the_reader_is_told_to_emit():
    """The reader prompts instruct "I don't have enough information..."
    (longmemeval_adapter.py:275/282), so that family must be caught. The
    committed answers are the correct-answer control: a classifier that flagged
    them too would report a huge, meaningless rate (guard G3)."""
    refusals = [
        "I don't have enough information to answer this question.",
        "I can't tell which is her personal best, though the context mentions 3:42.",
        "The context does not specify a date.",
    ]
    committed = [
        "Her personal best is 3:42.",
        "She ran the race in March 2023.",
        "Three: the marathon, the half and the 10k.",
    ]
    assert all(JA.classify_refusal(a)["refusal"] for a in refusals)
    assert not any(JA.classify_refusal(a)["refusal"] for a in committed)
    assert JA.classify_refusal(refusals[0])["canonical"] is True
    assert JA.classify_refusal(refusals[1])["loose_only"] is True


def test_gold_recitation_check_is_strict_in_the_safe_direction():
    """Biased DOWN on purpose: a loose similarity check on this exact shape
    false-positived at 55% against an 11% control earlier in this project and
    the number was retracted."""
    refusal = ("I can't tell which is her personal best, though the context "
               "mentions 3:42.")
    assert JA.recites_gold(refusal, "3:42") is True
    assert JA.recites_gold(refusal, "2:58") is False
    assert JA.recites_gold("", "3:42") is False
    assert JA.recites_gold(refusal, "") is False


# ── recites_gold v1 -> v2: the pre-registered token-rule gate (R1-R4) ────
#
# The 2026-08-25 audit hand-read all 10 rows of C3's numerator: zero genuine
# judge errors, and 5 of 10 were v1 FALSE NEGATIVES. These fixtures reconstruct
# the recorded MECHANISMS; the original rows live in judge_audit.json on the
# box, and whether v2 recovers all five of THEM is what --verify-recitation
# measures. A fixture pass is not that measurement, and must not be reported as
# one.

# (label, answer, gold) — every one of these is a v1 False that v2 must recover.
#
# NOTE THE DIRECTION. The banked note records two v1 defects in one breath, and
# they point OPPOSITE WAYS: discarding the numerals makes v1 more PERMISSIVE
# (it fires without the number), while mandating the gloss makes it more
# RESTRICTIVE. Only the second can produce a false negative, so only the second
# belongs here. v1's numeral blindness is a precision defect and is pinned by
# `test_v2_holds_numerals_at_zero_tolerance` and the not-nested test instead.
_R1_FIXTURES = [
    ("numerals deleted by len(t) > 2, and the gloss made mandatory",
     "I can't determine that from the context, though it mentions 22 days "
     "and 21 days.",
     "22 days (21 days is also acceptable)"),
    ("the same widening written as a trailing clause, not a parenthetical",
     "I'm not able to say. The notes put the gap at 22 days.",
     "22 days, 21 days is also acceptable"),
    ("one prose word missed in an otherwise complete recitation",
     "I'm not able to say, but the context mentions she started the "
     "medication in March.",
     "she started taking the medication in March"),
]


@pytest.mark.parametrize("label,answer,gold", _R1_FIXTURES,
                         ids=[f[0][:40] for f in _R1_FIXTURES])
def test_R1_v2_recovers_each_recorded_false_negative_mechanism(label, answer, gold):
    """R1 recall. Each fixture must be a v1 False AND a v2 True.

    Asserting BOTH halves is what makes this a recall test rather than a
    tautology: a fixture v1 already passes proves nothing about the change."""
    assert JA.recites_gold_v1(answer, gold) is False, \
        f"fixture does not reproduce the defect ({label}) — v1 already passes it"
    assert JA.recites_gold_v2(answer, gold) is True, \
        f"v2 fails to recover the recorded mechanism: {label}"


def test_R1_is_NOT_the_gate_and_the_docstring_says_so():
    """The banked gate must keep saying R1 alone cannot license the flip.

    R1 re-finds the rows the rule was written from. If a later edit quietly
    promotes it to the pass criterion, the gate stops being a gate — this is
    the same post-hoc-bar-moving the C3 ledger refuses by name."""
    doc = JA.__doc__
    assert "R1 ALONE CANNOT BE THE GATE" in doc
    assert "R2 and R3 are the gate" in doc


def test_v2_holds_numerals_at_zero_tolerance():
    """The asymmetry that pays for v2's prose slack.

    v2 is LOOSER than v1 on gloss and on one missing prose word, so it must be
    STRICTER somewhere or it is simply a loosening. Numerals are that
    somewhere: a wrong or absent number is not a recitation, and it is the
    numerals v1 was deleting in the first place."""
    gold = "the meeting ran 45 minutes"
    assert JA.recites_gold_v2("I don't know; the meeting ran 45 minutes.", gold) is True
    # Same prose, wrong number: v2 refuses where a pure loosening would not.
    assert JA.recites_gold_v2("I don't know; the meeting ran 40 minutes.", gold) is False
    # And the number missing entirely.
    assert JA.recites_gold_v2("I don't know; the meeting ran a while.", gold) is False


def test_the_two_recitation_rules_are_NOT_nested():
    """Both directions of change are real, which is why both are hand-checked.

    v2 flags rows v1 missed (numerals, gloss) AND un-flags rows v1 caught,
    because v1's `len(t) > 2` filter deleted the very token v2 now requires.
    A verifier reporting only `newly_flagged` would under-report the change."""
    newly_gold, newly_ans = ("22 days (21 days is also acceptable)",
                             "I can't say — the context mentions 22 days.")
    lost_gold, lost_ans = ("5 kilometres apart",
                           "I don't have that; the notes say the towns are "
                           "kilometres apart.")
    assert (JA.recites_gold_v1(newly_ans, newly_gold),
            JA.recites_gold_v2(newly_ans, newly_gold)) == (False, True)
    assert (JA.recites_gold_v1(lost_ans, lost_gold),
            JA.recites_gold_v2(lost_ans, lost_gold)) == (True, False)


def test_v2_keeps_every_guarantee_v1_had_in_the_safe_direction():
    """The pinned v1 behaviours that must survive the loosening."""
    refusal = ("I can't tell which is her personal best, though the context "
               "mentions 3:42.")
    assert JA.recites_gold_v2(refusal, "3:42") is True       # verbatim fast path
    assert JA.recites_gold_v2(refusal, "2:58") is False      # mismatched gold
    assert JA.recites_gold_v2("", "3:42") is False
    assert JA.recites_gold_v2(refusal, "") is False


def test_a_gold_that_is_entirely_gloss_does_not_match_everything():
    """The degenerate case the gloss-stripper has to refuse.

    Stripping the gloss from a gold that is ONLY gloss leaves no content
    tokens. Returning True there would make the rule fire on every answer —
    a ceiling instrument, not a check."""
    assert JA._gold_primary_clause("(also acceptable)").strip() == "(also acceptable)"
    assert JA.recites_gold_v2("nothing relevant here at all", "(also acceptable)") is False


def test_the_shipping_alias_stays_v1_because_R2_FAILED():
    """The gate RAN (A5, 2026-08-26) and FAILED R2. The alias is the flip.

    Before A5 this test read "until the gate runs" and pinned STATUS: UNRUN.
    That is no longer the reason: R2's bar was FP <= 1 of the newly-flagged
    sample and the hand-check found 2, failing on both readings of "sample"
    (2 of 12 newly-flagged, 2 of the 3 refusal-arm rows). So a flip is now a
    documented violation of a resolved bar, not a premature act — and this
    test is what makes that distinction cost something.

    The waiver argued at the time — that the 2 FPs "only inflate the upper
    bound" — is the hazard R2 exists to catch. The ceiling licenses the spend;
    inflating it wrongly authorises money. Re-tuning RECITE_ALPHA_COVERAGE
    until R2 clears is the move the banked verdict language forbids by name."""
    assert JA.recites_gold is JA.recites_gold_v1
    assert "R2 FAIL" in JA.__doc__
    assert "STATUS: UNRUN" not in JA.__doc__


def test_recites_gold_v1_stays_frozen_at_the_pre_fix_behaviour():
    """Exists solely to FAIL on a well-meant tidy-up, like `shipping_verdict`.

    v1 produced C3 10/470, the 5-of-10 decomposition and the 16/470 = 3.40%
    ceiling that licensed the 2026-08-25 spend. Re-syncing it to v2 would make
    --verify-recitation's before/after diff a constant zero BY CONSTRUCTION."""
    # The exact defect v1 must keep exhibiting: numerals below the length filter.
    assert JA.recites_gold_v1(
        "I can't say — the context mentions 22 days.",
        "22 days (21 days is also acceptable)") is False
    assert JA.recites_gold_v1 is not JA.recites_gold_v2


# ── --verify-recitation: the free before/after check, and its vacuity split ──

def _rrec(rid, gold, answer, *, is_abs=False, refusal=True, verdict=True):
    return {"id": rid, "is_abs": is_abs, "answer_refusal": refusal,
            "verdict": verdict, "_gold": gold, "_answer": answer}


def test_verify_recitation_counts_both_directions_of_change():
    recs = [
        _rrec("n1", "22 days (21 days is also acceptable)",
              "I can't say — the context mentions 22 days."),
        _rrec("l1", "5 kilometres apart",
              "I don't have that; the notes say the towns are kilometres apart."),
    ]
    res = JA.verify_recitation(recs)
    assert res["newly_flagged"] == 1 and res["newly_flagged_ids"] == ["n1"]
    assert res["no_longer_flagged"] == 1 and res["no_longer_flagged_ids"] == ["l1"]


def test_a_verbatim_only_corpus_is_reported_as_VACUOUS():
    """The E3 trap, reappearing inside the verification of a fix for it.

    A row whose gold appears verbatim in the answer is decided by the fast path
    BOTH rules share, so no token-rule change can move it. On such a corpus
    `newly_flagged == 0` cannot fail to be 0, and reporting that as "the change
    is inert" would be a certificate signed by an instrument that never met the
    surface it certifies. Mirrors
    `test_a_zero_flip_result_on_a_compliant_corpus_is_reported_as_VACUOUS`."""
    recs = [_rrec("v1", "3:42", "I can't tell, though the context mentions 3:42."),
            _rrec("v2", "blue", "I don't know; she did mention blue.")]
    res = JA.verify_recitation(recs)
    assert res["newly_flagged"] == 0 and res["no_longer_flagged"] == 0
    assert res["token_rule_consulted"] == 0
    assert res["verbatim_decided"] == 2
    assert "VACUOUS" in res["verdict"]


def test_the_vacuity_split_clears_when_the_token_rule_is_actually_consulted():
    """The other half of the guard: it must be able to NOT trip."""
    recs = [_rrec("v1", "3:42", "I can't tell, though the context mentions 3:42."),
            _rrec("t1", "22 days (21 days is also acceptable)",
                  "I can't say — the context mentions 22 days.")]
    res = JA.verify_recitation(recs)
    assert res["token_rule_consulted"] == 1
    assert "VACUOUS" not in res["verdict"]


def test_R3_control_pairs_each_answer_with_a_DIFFERENT_gold():
    """R3's negative control, and the reason it is not a formality.

    A loose similarity check on this exact shape false-positived at 55% against
    an 11% correct-answer control earlier in this project and the number was
    retracted. If v2 fires as often on mismatched pairs as on true ones it is
    measuring text volume; the shuffled arm is what says so for free."""
    recs = [_rrec("a", "22 days", "I can't say — the context mentions 22 days."),
            _rrec("b", "blue", "I don't know; she did mention blue."),
            _rrec("c", "17 March", "No idea, though 17 March comes up.")]
    res = JA.verify_recitation(recs)
    assert res["control_n"] == 3
    assert res["control_rate_v2"] == 0.0
    assert res["true_pair_rate_v2"] == 100.0
    assert res["r3_pass"] is True


def test_R3_FAILS_a_rule_that_fires_on_mismatched_pairs():
    """The guard asserted to TRIP, not only to clear.

    Every gold's tokens appear in every answer here, so a recitation rule
    cannot distinguish true pairs from shuffled ones — it fires at the same
    rate on both. R3 must call that out rather than report a healthy
    `newly_flagged` count."""
    # Every gold's tokens appear in every answer, but no gold appears
    # VERBATIM — otherwise the vacuity branch fires before R3 is reached.
    shared = ("I can't say; the notes mention a shade of blue, a shade of "
              "green and a shade of red.")
    recs = [_rrec("a", "blue shade", shared),
            _rrec("b", "green shade", shared),
            _rrec("c", "red shade", shared)]
    res = JA.verify_recitation(recs)
    assert res["control_rate_v2"] > 0.0
    assert res["r3_pass"] is False
    assert "R3 FAIL" in res["verdict"]


def test_R3_control_drops_pairs_whose_shuffled_gold_equals_their_own():
    """A duplicate gold would smuggle a TRUE pair into the control arm and
    inflate the floor the discriminability ratio is measured against."""
    # Non-verbatim on purpose: a verbatim pair would trip the VACUOUS branch
    # first and this guard would never be reached.
    recs = [_rrec("a", "22 elapsed days", "I can't say; 22 days elapsed."),
            _rrec("b", "22 elapsed days", "No idea; 22 days had elapsed.")]
    res = JA.verify_recitation(recs)
    assert res["control_n"] == 0
    assert "R3 UNMEASURED" in res["verdict"]


def test_R4_ceiling_is_reported_under_BOTH_rules_on_the_same_denominator():
    """R4. The licence numerator is `recites_gold`, so a token-rule change
    changes the spend arithmetic and not merely the C3 footnote. Both ceilings
    are printed side by side, over the non-`_abs` denominator `build_report`
    divides by — an `_abs` row that refuses AND recites is the INTENDED
    behaviour, never the defect."""
    recs = [
        _rrec("n1", "22 days (21 days is also acceptable)",
              "I can't say — the context mentions 22 days."),
        _rrec("a1", "17 March", "I don't know, the context mentions 17 March.",
              is_abs=True),
        _rrec("c1", "blue", "Her favourite colour is blue.", refusal=False),
    ]
    res = JA.verify_recitation(recs)
    assert res["ceiling_v1"]["den"] == res["ceiling_v2"]["den"] == 2  # _abs excluded
    assert res["ceiling_v1"]["num"] == 0
    assert res["ceiling_v2"]["num"] == 1


def test_R2_hand_check_sample_labels_the_two_directions_separately(tmp_path):
    """R2 is the arm that cannot be automated — a token rule cannot tell you
    whether a refusal genuinely recited the gold. The two directions are
    different errors and are scored separately: `newly` risks a wrongly
    LICENSED spend, `lost` is a numeral v1 never checked."""
    recs = [
        _rrec("n1", "22 days (21 days is also acceptable)",
              "I can't say — the context mentions 22 days."),
        _rrec("l1", "5 kilometres apart",
              "I don't have that; the notes say the towns are kilometres apart."),
    ]
    path = JA.write_recitation_sample(recs, str(tmp_path / "aud.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [r["id"] for r in payload["newly"]] == ["n1"]
    assert [r["id"] for r in payload["lost"]] == ["l1"]
    assert payload["newly"][0]["_gold"] and payload["newly"][0]["_answer"]


def test_recitation_sample_writes_nothing_when_the_rules_agree(tmp_path):
    recs = [_rrec("v1", "3:42", "I can't tell, though the context mentions 3:42.")]
    assert JA.write_recitation_sample(recs, str(tmp_path / "aud.json")) is None


def test_verify_recitation_cli_path_is_reachable_and_free(tmp_path, capsys):
    """The CLI branch, exercised. An unreachable code path reads as PASS."""
    aud = tmp_path / "aud.json"
    aud.write_text(json.dumps({"records": [
        _rrec("n1", "22 days (21 days is also acceptable)",
              "I can't say — the context mentions 22 days."),
        _rrec("b", "blue", "I don't know; she did mention blue."),
    ]}), encoding="utf-8")
    assert JA.main(["--verify-recitation", str(aud)]) == 0
    out = capsys.readouterr().out
    assert "RECITATION v1 -> v2" in out and "FREE, no LLM" in out
    assert "token rule actually consulted on: 1" in out
    assert "R3 control" in out and "R4 licence" in out


# -- Degeneracy guards. Each is asserted to TRIP and, separately, to CLEAR, so
# -- none of them can be a guard that is always on (vacuously safe) or always
# -- off (vacuously absent).

def _rec(**kw):
    base = {"bucket": "compliant_yes", "compliant": True, "shipping": True,
            "reference": True, "disagrees": False, "chars": 3, "id": "x",
            "question_type": "single-session-user", "is_abs": False,
            "verdict": True, "verdict_original": True, "raw": "yes",
            "answer_refusal": False, "answer_canonical": False,
            "answer_loose_only": False, "recites_gold": False,
            "negated": False}
    base.update(kw)
    return base


def _pre(**kw):
    base = {"n": 10, "judgeable": 10, "abs_rows": 1, "refusals": 1,
            "refusals_canonical": 1, "refusals_reciting_gold": 0,
            "refusal_rate": 10.0}
    base.update(kw)
    return base


def test_G1_zero_refusals_reports_unmeasured_not_a_clean_zero():
    """The guard the brief demanded. With no refusals present, C3 measured
    nothing; reporting 0.0% would be the ceiling-instrument failure."""
    rep = JA.build_report([_rec() for _ in range(10)], _pre(), handcheck=(0, 0))
    assert rep["C3_refusal_scored_correct_rate"] == 0.0
    assert rep["verdict"] == "INCOMPLETE"
    assert any(b.startswith("G1 UNMEASURED") for b in rep["blocked"])


def test_G1_clears_when_refusals_are_present():
    """Negative control for G1: it must not be permanently on."""
    recs = [_rec() for _ in range(9)] + [_rec(answer_refusal=True, verdict=False)]
    rep = JA.build_report(recs, _pre(), handcheck=(0, 0))
    assert not any(b.startswith("G1") for b in rep["blocked"])


def test_G2_no_abs_rows_is_reported_as_unmeasured():
    recs = [_rec(answer_refusal=True, verdict=False)] + [_rec() for _ in range(9)]
    rep = JA.build_report(recs, _pre(abs_rows=0), handcheck=(0, 0))
    assert any(b.startswith("G2 UNMEASURED") for b in rep["blocked"])
    rep_ok = JA.build_report(recs, _pre(abs_rows=3), handcheck=(0, 0))
    assert not any(b.startswith("G2") for b in rep_ok["blocked"])


def test_G3_classifier_ceiling_declares_broken_instead_of_spectacular():
    recs = [_rec(answer_refusal=True, verdict=True) for _ in range(9)] + [_rec()]
    rep = JA.build_report(recs, _pre(), handcheck=(0, 0))
    assert any(b.startswith("G3 BROKEN") for b in rep["blocked"])
    assert rep["verdict"] == "INCOMPLETE", \
        "a 90% refusal-scored-correct rate must NOT be reported as MATERIAL"


def test_G4_missing_handcheck_forces_INCOMPLETE_never_MATERIAL():
    """Three of four criteria plus a missing hand-score must read INCOMPLETE —
    the same gate arithmetic fact_probe already pins."""
    recs = ([_rec(answer_refusal=True, verdict=True)]
            + [_rec(answer_refusal=True, verdict=False) for _ in range(9)]
            + [_rec() for _ in range(90)])
    rep = JA.build_report(recs, _pre(), handcheck=None)
    assert rep["C3_band"] == "WATCH"
    assert rep["verdict"] == "INCOMPLETE"
    assert any(b.startswith("G4 INCOMPLETE") for b in rep["blocked"])


def test_bands_bind_at_the_preregistered_thresholds():
    """The thresholds are load-bearing, so pin them. If someone edits the
    constants, this fails and the pre-registration is visibly broken rather than
    silently relaxed."""
    assert (JA.C1_MATERIAL, JA.C1_WATCH) == (1.0, 0.2)
    assert (JA.C3_MATERIAL, JA.C3_WATCH) == (2.0, 0.5)
    assert JA.C4_MATERIAL == 1.0

    # C3 at exactly 2.0% (2 of 100 non-_abs rows) must read MATERIAL.
    recs = [_rec(answer_refusal=True, verdict=True) for _ in range(2)] + \
           [_rec(answer_refusal=True, verdict=False) for _ in range(3)] + \
           [_rec() for _ in range(95)]
    rep = JA.build_report(recs, _pre(), handcheck=(0, 0))
    assert rep["C3_refusal_scored_correct_rate"] == pytest.approx(2.0)
    assert rep["C3_band"] == "MATERIAL"
    assert rep["verdict"] == "MATERIAL"

    # One row fewer and it drops to WATCH — the band edge actually binds.
    recs2 = recs[1:] + [_rec()]
    rep2 = JA.build_report(recs2, _pre(), handcheck=(0, 0))
    assert rep2["C3_band"] == "WATCH"
    assert rep2["verdict"] == "WATCH"


def test_C4_pair_precheck_is_free_and_flags_arm_asymmetry():
    """C4 costs nothing and must be run before any spend. Pinned in both
    directions so it is neither always-VOID nor always-OK."""
    def rows(n_refusals, n):
        out = []
        for i in range(n):
            ans = ("I don't have enough information to answer."
                   if i < n_refusals else "Her personal best is 3:42.")
            out.append({"id": f"q{i}", "category": 1, "question": "q",
                        "answer": "3:42", "ai_answer": ans})
        return out

    sym = JA.pair_precheck(rows(10, 100), rows(10, 100), "locomo")
    assert sym["diff_pp"] == pytest.approx(0.0)
    assert sym["verdict"].startswith("OK")

    asym = JA.pair_precheck(rows(10, 100), rows(25, 100), "locomo")
    assert asym["diff_pp"] == pytest.approx(15.0)
    assert asym["verdict"].startswith("VOID")


def test_judge_input_reconstruction_matches_each_adapter():
    """A re-judge that rebuilds the judge input differently measures prompt
    drift, not the judge. LoCoMo cat-5 is the case that actually varies: the
    stored `answer` is a `[unanswerable; trap: ...]` string that must be
    expanded back through `_gold_for_judge`."""
    from locomo_adapter import _gold_for_judge

    cat1 = {"category": 1, "question": "q", "answer": "3:42", "ai_answer": "a"}
    assert JA.judge_inputs(cat1, "locomo") == ("multi-session", "q", "3:42", "a")

    cat5 = {"category": 5, "question": "q",
            "answer": "[unanswerable; trap: 2:58]", "ai_answer": "a"}
    qtype, _q, gold, _a = JA.judge_inputs(cat5, "locomo")
    assert qtype == "single-session-user_abs"
    assert gold == _gold_for_judge(5, None, "2:58")
    assert "2:58" in gold and gold != "[unanswerable; trap: 2:58]"

    lme = {"question_type": "knowledge-update", "question": "q",
           "answer": "g", "hypothesis": "h"}
    assert JA.judge_inputs(lme, "lme") == ("knowledge-update", "q", "g", "h")

    msc = {"question": "q", "answer": "g", "ai_answer": "h"}
    assert JA.judge_inputs(msc, "msc") == ("single-session-user", "q", "g", "h")


def test_stable_sample_is_deterministic_and_content_addressed():
    """Re-running the audit must re-judge the SAME rows; otherwise the delta
    between two audit passes is pure sampling noise."""
    rows = [{"id": f"q{i}"} for i in range(100)]
    a = JA.stable_sample(rows, 20)
    b = JA.stable_sample(list(reversed(rows)), 20)
    assert len(a) == 20
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert JA.stable_sample(rows, None) == rows
    assert len(set(r["id"] for r in a)) == 20


def test_recording_wrapper_captures_the_raw_reply_judge_answer_discards():
    """The instrument's whole reason to exist, asserted directly."""
    inner = StubJudgeClient("Yes and no, on balance yes")
    rec = JA.RecordingJudge(inner)
    verdict = judge_answer(rec, "single-session-user", "q", "a", "r")
    assert verdict is True
    assert rec.replies == ["Yes and no, on balance yes"]


def test_end_to_end_spend_path_is_reachable_with_an_injected_client(tmp_path,
                                                                    capsys):
    """The degeneracy trap in person: an unreachable spend path would let every
    test above pass while the script could never actually run. This drives
    `main()` through the real `--spend` branch with an INJECTED stub client, so
    no live client is ever constructed and no token is spent."""
    rows = [
        # refusal that recites the gold, and the stub judge says yes -> C3 hit
        {"id": "q1", "category": 1, "question": "PB?", "answer": "3:42",
         "ai_answer": "I can't tell which is her PB, though the context mentions 3:42.",
         "correct": True},
        # committed answer
        {"id": "q2", "category": 1, "question": "PB?", "answer": "3:42",
         "ai_answer": "Her personal best is 3:42.", "correct": True},
        # unjudgeable — must be excluded, not judged
        {"id": "q3", "category": 1, "question": "PB?", "answer": "3:42",
         "ai_answer": "[LLM_ERROR: boom]", "correct": False},
        # an _abs row so guard G2 clears
        {"id": "q4", "category": 5, "question": "PB?",
         "answer": "[unanswerable; trap: 2:58]",
         "ai_answer": "I don't have enough information to answer.", "correct": True},
    ]
    src = tmp_path / "run.json"
    src.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "audit.json"

    client = StubJudgeClient("yes")
    rc = JA.main(["--run", str(src), "--bench", "locomo", "--spend",
                  "--out", str(out)], client=client)
    assert rc == 0

    saved = json.loads(out.read_text(encoding="utf-8"))
    # 3 judgeable rows, exactly one judge call each; the LLM_ERROR row excluded.
    assert saved["report"]["n_judged"] == 3
    assert len(client.calls) == 3, \
        "one judge call per judgeable row (and the unjudgeable row was skipped)"
    assert all(c["max_tokens"] == 10 for c in client.calls)

    # The raw reply reached the record — the thing no run has ever stored.
    assert all(r["raw"] == "yes" for r in saved["records"])

    # C3 counts the reciting refusal on the non-_abs row and NOT the _abs row.
    assert saved["report"]["C3_n"] == 1
    assert saved["report"]["C3_denominator_non_abs"] == 2
    assert saved["report"]["C3_of_which_recite_gold"] == 1

    # G4 keeps it INCOMPLETE with no hand-check supplied.
    assert saved["report"]["verdict"] == "INCOMPLETE"

    # The hand-check file carries BOTH classes, so an FN count is possible.
    hc = json.loads(out.with_suffix(".handcheck.json").read_text(encoding="utf-8"))
    assert hc["classified_refusal"] and hc["classified_committed_CONTROL"]


def test_default_run_spends_nothing_and_prints_the_cost(tmp_path, capsys):
    """Cost discipline, asserted rather than promised: without --spend, no call
    reaches the client at all."""
    rows = [{"id": "q1", "category": 1, "question": "PB?", "answer": "3:42",
             "ai_answer": "Her personal best is 3:42.", "correct": True}]
    src = tmp_path / "run.json"
    src.write_text(json.dumps(rows), encoding="utf-8")

    client = StubJudgeClient("yes")
    rc = JA.main(["--run", str(src), "--bench", "locomo"], client=client)
    assert rc == 0
    assert client.calls == [], "the default path made an LLM call"
    assert "would cost 1 judge calls" in capsys.readouterr().out


def test_module_constructs_no_live_client_at_import():
    """House rule: never ship a real LLM backend. The only construction site is
    inside `build_judge_client`, called only under --spend."""
    tree = ast.parse((_BENCH / "judge_audit.py").read_text(encoding="utf-8"))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build_judge_client" in fns

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "LLMClient"]
    assert len(calls) == 1, "expected exactly one LLMClient construction site"
    inside = [n for n in ast.walk(fns["build_judge_client"])
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "LLMClient"]
    assert len(inside) == 1, "the LLMClient construction escaped build_judge_client"

    # And it is not at module scope: no top-level statement constructs one.
    for node in tree.body:
        assert not any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                       and c.func.id == "LLMClient" for c in ast.walk(node)) \
            or isinstance(node, ast.FunctionDef)


def test_handcheck_corrects_C3_rather_than_merely_unlocking_the_gate():
    """A control that only lifted G4 would be ceremonial: an operator could
    report a badly wrong classifier and the verdict would not move. The measured
    false-positive rate must scale C3, and the band must apply to the CORRECTED
    value."""
    # 3 refusal-scored-correct in 100 non-_abs rows = 3.0% raw -> MATERIAL.
    recs = ([_rec(answer_refusal=True, verdict=True) for _ in range(3)]
            + [_rec(answer_refusal=True, verdict=False) for _ in range(22)]
            + [_rec() for _ in range(75)])

    clean = JA.build_report(recs, _pre(), handcheck=(0, 0))
    assert clean["C3_refusal_scored_correct_rate"] == pytest.approx(3.0)
    assert clean["C3_rate_handcheck_adjusted"] == pytest.approx(3.0)
    assert clean["verdict"] == "MATERIAL"

    # Hand-check finds 15 of the 25 sampled refusals were misclassified (60% FP):
    # above the void threshold, so C3 must not be read at all.
    broken = JA.build_report(recs, _pre(), handcheck=(15, 0))
    assert broken["handcheck"]["fp_rate"] == pytest.approx(0.6)
    assert any(b.startswith("HAND-CHECK BROKEN") for b in broken["blocked"])
    assert broken["verdict"] == "INCOMPLETE"

    # A 40% FP rate is below the void threshold but pulls 3.0% down to 1.8%,
    # which crosses out of MATERIAL. This is the case that proves the
    # correction changes the DECISION, not just a printed figure.
    half = JA.build_report(recs, _pre(), handcheck=(10, 0))
    assert half["C3_rate_handcheck_adjusted"] == pytest.approx(3.0 * (1 - 10 / 25))
    assert half["C3_band"] == "WATCH", \
        "the correction must be able to DOWNGRADE the verdict, not just annotate it"
    assert half["verdict"] == "WATCH"


def test_handcheck_false_negatives_mark_C3_a_floor():
    """FN = a real refusal the classifier called committed. Those rows never
    entered C3's numerator, so a high FN rate means C3 is a floor. The report
    says so instead of silently under-reporting — and no adjustment is invented,
    because correcting it would need verdicts the sample does not carry."""
    recs = ([_rec(answer_refusal=True, verdict=True)]
            + [_rec(answer_refusal=True, verdict=False) for _ in range(24)]
            + [_rec() for _ in range(75)])  # 25 flagged, 75 control
    rep = JA.build_report(recs, _pre(), handcheck=(0, 10))
    assert rep["handcheck"]["fn_rate"] == pytest.approx(0.4)
    assert any(b.startswith("HAND-CHECK UNDER-COUNT") for b in rep["blocked"])
    assert rep["verdict"] == "INCOMPLETE"

    ok = JA.build_report(recs, _pre(), handcheck=(0, 1))
    assert not any(b.startswith("HAND-CHECK") for b in ok["blocked"])


def test_handcheck_sample_size_and_correction_divisor_cannot_drift_apart():
    """The FP rate is `fp / min(HANDCHECK_K, n_refusals)`, and HANDCHECK_K must
    be the SAME constant that sizes the sample the operator actually hand-checks.
    If the writer's `k` ever drifted from the divisor, every corrected rate would
    be silently wrong by that ratio — a defect that reads as a clean number."""
    import inspect
    sig = inspect.signature(JA.write_handcheck_sample)
    assert sig.parameters["k"].default == JA.HANDCHECK_K


# ── C3 ceiling: the _abs cross-tab that licenses (or refuses) the spend ──

def _lme(qtype: str, answer: str, gold: str = "marathon") -> dict:
    return {"question_type": qtype, "question": "q",
            "answer": gold, "hypothesis": answer}


_RECITING_REFUSAL = "I don't have enough information; the context says marathon."


def test_abs_reciting_refusals_are_excluded_from_the_C3_ceiling():
    """C3 is denominated on non-_abs rows only. An _abs row that refuses AND
    recites gold is the INTENDED behaviour, so counting it toward the ceiling
    inflates the licence to spend by exactly the rows the criterion throws away.

    Carrier: the two _abs rows here recite, the non-_abs rows do not, so the
    pooled count is 2 while the ceiling that matters is 0."""
    rows = [_lme("single-session-user_abs", _RECITING_REFUSAL) for _ in range(2)]
    rows += [_lme("knowledge-update", "she ran a 10k") for _ in range(8)]
    pre = JA.free_precheck(rows, "lme")

    assert pre["refusals_reciting_gold"] == 2          # pooled, as before
    assert pre["refusals_reciting_gold_abs"] == 2
    assert pre["refusals_reciting_gold_non_abs"] == 0
    assert pre["c3_ceiling_non_abs"] == 0.0


def test_a_non_abs_reciting_refusal_does_raise_the_ceiling():
    """Negative control for the test above. If the ceiling read 0 for BOTH
    populations the exclusion would be untested — a filter that drops every row
    passes the carrier just as well as a correct one."""
    rows = [_lme("knowledge-update", _RECITING_REFUSAL) for _ in range(2)]
    rows += [_lme("knowledge-update", "she ran a 10k") for _ in range(8)]
    pre = JA.free_precheck(rows, "lme")

    assert pre["refusals_reciting_gold_non_abs"] == 2
    assert pre["refusals_reciting_gold_abs"] == 0
    assert pre["c3_ceiling_non_abs"] == pytest.approx(20.0)


def test_the_ceiling_denominator_matches_the_C3_denominator_exactly():
    """The ceiling is only comparable to the C3 bar if it is denominated the
    same way `build_report` denominates C3 (`len(non_abs)`, i.e. JUDGED non-_abs
    rows). An LLM-error row is not judgeable and must leave BOTH denominators."""
    rows = [_lme("knowledge-update", _RECITING_REFUSAL)]
    rows += [_lme("knowledge-update", "she ran a 10k") for _ in range(3)]
    rows += [_lme("single-session-user_abs", "no idea")]
    rows += [_lme("knowledge-update", "[LLM_ERROR: upstream 503]")]
    pre = JA.free_precheck(rows, "lme")

    assert pre["n"] == 6
    assert pre["judgeable"] == 5           # the LLM_ERROR row drops out
    assert pre["judgeable_non_abs"] == 4   # ... of the non-_abs denominator too
    assert pre["c3_ceiling_non_abs"] == pytest.approx(25.0)


def test_an_all_abs_run_is_unmeasurable_and_not_a_clean_zero():
    """Zero non-_abs rows gives 0/0, which renders as 0.00% — numerically
    identical to a genuinely clean run. If the degenerate check did not precede
    the numeric comparison, an all-_abs run would report UNREACHABLE: a
    confident 'nothing to find here' for a run that measured nothing. That is
    the E3 trap in miniature."""
    rows = [_lme("single-session-user_abs", _RECITING_REFUSAL) for _ in range(4)]
    pre = JA.free_precheck(rows, "lme")
    assert pre["judgeable_non_abs"] == 0
    assert pre["c3_ceiling_non_abs"] == 0.0     # indistinguishable from clean

    code, msg = JA.c3_spend_licence(pre)
    assert code == JA.LICENCE_NO_DENOMINATOR
    assert "0/0" in msg and "NOT a clean result" in msg


def test_a_genuinely_clean_run_is_unreachable_not_unmeasurable():
    """Negative control for the test above: same 0.00% ceiling, real
    denominator. The two must NOT collapse to one code, or the degenerate guard
    would be indistinguishable from the finding it is meant to protect."""
    rows = [_lme("knowledge-update", "I don't know") for _ in range(4)]
    pre = JA.free_precheck(rows, "lme")
    assert pre["judgeable_non_abs"] == 4
    assert pre["c3_ceiling_non_abs"] == 0.0

    code, _ = JA.c3_spend_licence(pre)
    assert code == JA.LICENCE_UNREACHABLE


def test_zero_refusals_outranks_every_other_licence_branch():
    """G1 is first: with no refusals at all there is nothing for C3 to measure,
    and that is a stronger statement than 'below the bar'."""
    rows = [_lme("knowledge-update", "she ran a 10k") for _ in range(4)]
    pre = JA.free_precheck(rows, "lme")
    assert JA.c3_spend_licence(pre)[0] == JA.LICENCE_NO_REFUSALS


def test_the_licence_keys_on_the_banked_bar_not_a_hard_coded_literal():
    """If the threshold were inlined, re-banking C3_MATERIAL would silently
    leave the licence keyed to the OLD bar — authorising spends the gate no
    longer justifies, with no test failing."""
    rows = [_lme("knowledge-update", _RECITING_REFUSAL)]
    rows += [_lme("knowledge-update", "she ran a 10k") for _ in range(99)]
    pre = JA.free_precheck(rows, "lme")
    assert pre["c3_ceiling_non_abs"] == pytest.approx(1.0)

    old = JA.C3_MATERIAL
    try:
        JA.C3_MATERIAL = 5.0
        assert JA.c3_spend_licence(pre)[0] == JA.LICENCE_UNREACHABLE
        JA.C3_MATERIAL = 0.5
        assert JA.c3_spend_licence(pre)[0] == JA.LICENCE_REACHABLE
    finally:
        JA.C3_MATERIAL = old


# ── Hand-check sample / divisor parity ──────────────────────────────────

def _hc_rec(*, is_abs: bool, refusal: bool) -> dict:
    r = _rec(answer_refusal=refusal, verdict=False)
    r["is_abs"] = is_abs
    return r


def test_handcheck_arm_sizes_equal_the_divisors_they_are_scored_against(tmp_path):
    """The FP rate is `fp / n_flag`, and `n_flag` is sized off NON-_abs refusals
    (`build_report`) while the sampler drew from ALL rows. Every _abs row in the
    written arm therefore consumed a divisor slot for a row the criterion never
    counts, understating the FP rate by that fraction — silently, as a clean
    number. Pin the two populations together so they cannot drift again."""
    recs = ([_hc_rec(is_abs=True, refusal=True) for _ in range(20)]
            + [_hc_rec(is_abs=False, refusal=True) for _ in range(8)]
            + [_hc_rec(is_abs=False, refusal=False) for _ in range(9)])

    dest = JA.write_handcheck_sample(recs, str(tmp_path / "a.json"))
    written = json.loads(dest.read_text())
    rep = JA.build_report(recs, _pre(), handcheck=(0, 0))

    assert len(written["classified_refusal"]) == rep["handcheck"]["n_flag"] == 8
    assert len(written["classified_committed_CONTROL"]) == rep["handcheck"]["n_ctrl"] == 9


def test_no_abs_row_reaches_either_handcheck_arm(tmp_path):
    """Direct statement of the population rule. The parity test above would also
    pass if BOTH sides wrongly included _abs rows; this one pins the side that
    has to be non-_abs on its own."""
    recs = ([_hc_rec(is_abs=True, refusal=True) for _ in range(5)]
            + [_hc_rec(is_abs=True, refusal=False) for _ in range(5)]
            + [_hc_rec(is_abs=False, refusal=True) for _ in range(3)]
            + [_hc_rec(is_abs=False, refusal=False) for _ in range(3)])

    dest = JA.write_handcheck_sample(recs, str(tmp_path / "b.json"))
    w = json.loads(dest.read_text())
    assert len(w["classified_refusal"]) == 3
    assert len(w["classified_committed_CONTROL"]) == 3
    assert not any(r["is_abs"] for r in w["classified_refusal"])
    assert not any(r["is_abs"] for r in w["classified_committed_CONTROL"])


def _lme_row(*, qtype="single-session-user", answer="yes",
             gold="g", ident="r0"):
    return {"id": ident, "question_type": qtype, "question": "q",
            "answer": gold, "hypothesis": answer}


def test_pre_spend_writer_selects_the_same_arms_as_the_paid_writer(tmp_path):
    """write_handcheck_sample_pre must reconstruct exactly what rejudge_row
    would have put in the records, so the free-path writer and the paid-path
    writer pick the same rows. It delegates selection to write_handcheck_sample
    itself; this test pins the record construction it feeds in, including the
    _abs population rule — the drift class that shipped the 6-slot bug."""
    rows = ([_lme_row(qtype="single-session-user_abs",
                      answer="I don't have enough information", ident=f"a{i}")
             for i in range(4)]
            + [_lme_row(answer="I don't have enough information", ident=f"f{i}")
               for i in range(6)]
            + [_lme_row(answer="yes", ident=f"c{i}") for i in range(6)])
    # Paid-equivalent records: the same classifiers, applied by rejudge_row.
    paid_recs = []
    for r in rows:
        qtype, _q, gold, answer = JA.judge_inputs(r, "lme")
        cls = JA.classify_refusal(answer)
        paid_recs.append({"id": r["id"], "question_type": qtype,
                          "is_abs": "_abs" in qtype,
                          "answer_refusal": cls["refusal"],
                          "answer_canonical": cls["canonical"],
                          "answer_loose_only": cls["loose_only"],
                          "recites_gold": JA.recites_gold(answer, gold),
                          "_answer": answer, "_gold": gold})

    free = JA.write_handcheck_sample_pre(rows, "lme", str(tmp_path / "f.json"), k=3)
    paid = JA.write_handcheck_sample(paid_recs, str(tmp_path / "p.json"), k=3)
    fw, pw = json.loads(free.read_text()), json.loads(paid.read_text())

    for arm in ("classified_refusal", "classified_committed_CONTROL"):
        assert [r["id"] for r in fw[arm]] == [r["id"] for r in pw[arm]] == \
            [r["id"] for r in fw[arm]]
    # The population rule holds on the free side too.
    assert [r["id"] for r in fw["classified_refusal"]] == ["f0", "f1", "f2"]
    assert len(fw["classified_committed_CONTROL"]) == 3


def test_the_handcheck_arms_are_still_capped_at_HANDCHECK_K(tmp_path):
    """Negative control for the population fix: restricting to non-_abs must not
    disturb the cap. If the fix had been written as a filter applied AFTER the
    `[:k]` slice, arms would silently come back short of K on an _abs-heavy run
    and every rate would be divided by the wrong number in the other direction."""
    n = JA.HANDCHECK_K + 10
    # _abs rows FIRST. With them last, `[:k]` never reaches one and a
    # filter-after-slice bug is invisible — the fixture, not the code, would be
    # deciding the result.
    recs = ([_hc_rec(is_abs=True, refusal=True) for _ in range(n)]
            + [_hc_rec(is_abs=True, refusal=False) for _ in range(n)]
            + [_hc_rec(is_abs=False, refusal=True) for _ in range(n)]
            + [_hc_rec(is_abs=False, refusal=False) for _ in range(n)])

    w = json.loads(JA.write_handcheck_sample(recs, str(tmp_path / "c.json")).read_text())
    assert len(w["classified_refusal"]) == JA.HANDCHECK_K
    assert len(w["classified_committed_CONTROL"]) == JA.HANDCHECK_K


# ── The parse fix itself: the rule, and the free before/after check ─────
# `parse_judge_verdict` is tested directly here, without a client, because it is
# the piece the audit imports and the piece a future judge migration will meet
# first. Section 1 tests it through `judge_answer`; this section tests it as the
# pure function it was extracted to be.

@pytest.mark.parametrize("raw,expected", [
    ("yes", True), ("Yes.", True), ("**YES**", True), ('"yes"', True),
    ("no", False), ("No.", False), ("", False), (None, False),
    # word boundary
    ("no, she mentioned yesterday", False), ("the model describes eyes", False),
    # first verdict token wins, in BOTH directions
    ("yes then no", True), ("no then yes", False),
    # negation
    ("not yes", False), ("never a yes", False), ("isn't yes", False),
    # sentinel, including one carrying the word
    ("[LLM_ERROR: boom]", False), ("[LLM_ERROR: got 'yes' unexpectedly]", False),
    # no verdict token at all fails closed, same direction as the empty reply
    ("Let me think about this", False),
])
def test_parse_judge_verdict_rules(raw, expected):
    assert parse_judge_verdict(raw) is expected


def test_parse_judge_verdict_is_what_judge_answer_runs():
    """The extraction is only worth having if production actually calls it. A
    copy-pasted second implementation would pass every test above while
    `judge_answer` kept the old rule."""
    assert _judge("no, she mentioned yesterday") is parse_judge_verdict(
        "no, she mentioned yesterday")
    assert _judge("not yes") is parse_judge_verdict("not yes")


def _vrec(raw, bucket="compliant_yes", rid="r"):
    return {"raw": raw, "bucket": bucket, "id": rid}


def test_verify_parse_reports_zero_flips_on_a_compliant_corpus():
    """The 2026-08-25 LME shape: 500 bare yes/no replies, C2 = 0.00%."""
    res = JA.verify_parse([_vrec("yes"), _vrec("no", "compliant_no")] * 10)
    assert res["flips"] == 0
    assert res["rows_that_could_flip"] == 0


def test_a_zero_flip_result_on_a_compliant_corpus_is_reported_as_VACUOUS():
    """THE E3 TRAP, in its exact shape. "0 flips" on an all-compliant corpus and
    "0 flips" on a corpus full of non-compliant replies are the same number and
    opposite evidence: the first cannot fail, the second is a real no-op.

    If the verdict text ever collapses them, `--verify-parse` becomes a ceiling
    instrument certifying a fix it never had the data to test."""
    vacuous = JA.verify_parse([_vrec("yes"), _vrec("no", "compliant_no")])
    real = JA.verify_parse([_vrec("yes and no", "both_tokens"),
                            _vrec("Yes, clearly.", "verbose_yes")])
    assert vacuous["flips"] == real["flips"] == 0
    assert vacuous["rows_that_could_flip"] == 0
    assert real["rows_that_could_flip"] == 2
    assert "VACUOUSLY" in vacuous["verdict"]
    assert "VACUOUSLY" not in real["verdict"]


def test_verify_parse_catches_a_verdict_that_actually_changes():
    """Anti-vacuity: a checker that cannot report a flip proves nothing by
    reporting none. "yesterday" is the shape the fix exists for."""
    recs = [_vrec("no, she mentioned yesterday", "yes_substring_only", "flip"),
            _vrec("yes", "compliant_yes", "same")]
    res = JA.verify_parse(recs)
    assert res["flips"] == 1
    assert res["flip_ids"] == ["flip"]
    assert "NOT A NO-OP" in res["verdict"]


def test_verify_parse_rescores_raw_and_ignores_the_recorded_verdict():
    """The check has to RE-SCORE the stored reply under the frozen legacy rule,
    not read back the `verdict` the run happened to record.

    The two agree on every well-formed record, which is why the naive version is
    easy to write and impossible to catch with a realistic fixture. So the
    fixture here is deliberately CORRUPT: `verdict` is set to the value the
    legacy rule does NOT produce for this reply. A checker that re-scores sees a
    flip; a checker that trusts the field sees none.

    (An earlier draft of this test set `verdict` to the value the legacy rule
    DOES produce. It passed under both implementations — a test named for a
    defect that could not detect it.)"""
    raw = "no, she mentioned yesterday"
    assert JA.shipping_verdict(raw) is True and JA.landed_verdict(raw) is False

    rec = _vrec(raw, "yes_substring_only", "x")
    rec["verdict"] = False          # corrupt: NOT what the legacy rule scores
    assert JA.verify_parse([rec])["flips"] == 1, \
        "verify_parse trusted the recorded verdict instead of re-scoring `raw`"


def test_verify_parse_cli_path_is_reachable_and_free(tmp_path, capsys):
    """The same degeneracy trap as the spend path: a `--verify-parse` branch
    nobody can reach would let every unit test above pass while the standing
    check could never actually be run on the box.

    Drives `main()` through the real branch on a saved-audit-shaped file, with a
    client injected that must never be touched."""
    saved = {"precheck": {}, "report": {},
             "records": [_vrec("yes", "compliant_yes", "a"),
                         _vrec("no", "compliant_no", "b"),
                         _vrec("no, she mentioned yesterday",
                               "yes_substring_only", "c")]}
    f = tmp_path / "audit.json"
    f.write_text(json.dumps(saved), encoding="utf-8")

    client = StubJudgeClient("yes")
    rc = JA.main(["--verify-parse", str(f)], client=client)
    assert rc == 0
    assert client.calls == [], "the verification path made an LLM call"

    out = capsys.readouterr().out
    assert "stored replies: 3" in out
    assert "verdicts that CHANGE: 1" in out
    assert "NOT A NO-OP" in out


# ── D3: an outage is UNSCORED, never wrong (the policy, and its inertness) ──

def _row(qtype="single-session-user", correct=True, **kw):
    r = {"question_id": kw.pop("qid", "q"), "question_type": qtype,
         "correct": correct}
    r.update(kw)
    return r


def test_a_judge_failure_stays_in_strict_accuracy_and_conditional_is_named():
    """Headline accuracy keeps the expected denominator; judged-only is explicit."""
    from longmemeval_adapter import accuracy, scored

    rows = [_row(qid="a"), _row(qid="b"),
            _row(qid="c", correct=None, judge_error=True)]
    assert len(scored(rows)) == 2
    assert accuracy(rows) == 2 / 3
    assert accuracy(scored(rows)) == 1.0


def test_the_accuracy_helper_does_not_TypeError_on_an_unscored_row():
    """The shape that made this ~15 edits rather than 6.

    `sum(r["correct"] for r in rows) / len(rows)` — repeated across the three
    adapters — raises the moment one row is unscored. The helper exists so that
    is ONE decision rather than fifteen slightly different guards, and this
    test asserts both halves: the raw shape still raises, and the helper does
    not."""
    from longmemeval_adapter import accuracy

    rows = [_row(qid="a"), _row(qid="c", correct=None, judge_error=True)]
    with pytest.raises(TypeError):
        sum(r["correct"] for r in rows) / len(rows)
    assert accuracy(rows) == 0.5


def test_compute_scores_is_INERT_on_a_run_with_no_judge_errors():
    """The inertness claim, asserted rather than assumed.

    The 2026-08-25 audit recorded 0 judge-side sentinels over 500 replies, so
    on a clean run the filter is the identity and every canonical number is
    untouched. LoCoMo 68.2 / LME 68.4 / MSC ~84.0 are NOT re-baselined by this
    change — but "no rows were dropped" has to be a test, not a paragraph."""
    from longmemeval_adapter import compute_scores, scored

    rows = [_row(qid="a"), _row(qid="b", correct=False),
            _row(qid="c", qtype="temporal-reasoning")]
    assert scored(rows) == rows            # identity: nothing dropped
    scores = compute_scores(rows)
    assert scores["OVERALL"] == {"accuracy": 2 / 3, "count": 3}
    assert scores["single-session-user"]["count"] == 2


def test_a_judge_failure_changes_every_strict_denominator_consistently():
    """Category and overall headline scores use the same complete denominator."""
    from longmemeval_adapter import compute_scores

    clean = [_row(qid="a"), _row(qid="b", correct=False)]
    with_outage = clean + [_row(qid="c", correct=None, judge_error=True)]
    assert compute_scores(clean)["OVERALL"] == {"accuracy": 0.5, "count": 2}
    assert compute_scores(with_outage)["OVERALL"] == {
        "accuracy": 1 / 3, "count": 3
    }
    assert compute_scores(with_outage)["single-session-user"] == {
        "accuracy": 1 / 3, "count": 3
    }


def test_judge_error_note_states_the_denominator_when_the_count_is_zero():
    """"0 judge errors" over a run that made no judge calls is not reassurance.

    Same reason `--verify-parse` reports `rows_that_could_flip`: a count with no
    denominator is a certificate signed by an instrument that never met the
    surface it certifies. The zero branch names what could have errored; the
    non-zero branch names the surviving denominator so an accuracy over n-k is
    never mistaken for one over n."""
    from longmemeval_adapter import judge_error_note

    clean = judge_error_note([_row(qid="a"), _row(qid="b")])
    assert "0 of 2" in clean and "could have errored" in clean

    dirty = judge_error_note([_row(qid="a"),
                              _row(qid="c", correct=None, judge_error=True)])
    assert "counts WRONG" in dirty
    assert "strict 2-row denominator" in dirty
    assert "Conditional judged-only n=1" in dirty


def test_diag_only_rows_are_unscored_but_are_NOT_judge_errors():
    """The two reasons for `correct=None` must not be conflated.

    `--diag-only` writes `correct=None` because no reader and no judge ever
    ran; the existing branch already refuses to print 0.0% for it, on the
    grounds that a run which measured nothing must not look like a run that
    scored zero. A judge outage is the OTHER None, and only it is a defect.
    `judge_error` is the field that separates them — without it, a diagnostics
    pass would be reported as a 100% outage."""
    from longmemeval_adapter import accuracy, judge_error_rows, scored

    diag = [_row(qid="d1", correct=None), _row(qid="d2", correct=None)]
    assert scored(diag) == [] and accuracy(diag) == 0.0
    assert judge_error_rows(diag) == []          # unscored, but no judge failed
    assert "0 of 2" in __import__(
        "longmemeval_adapter").judge_error_note(diag)


@pytest.mark.parametrize("adapter,fn", [
    ("locomo_adapter.py", "evaluate_qa"),
    ("msc_adapter.py", None),
])
def test_every_adapter_record_carries_the_judge_error_field(adapter, fn):
    """Asserted from the SOURCE, because the alternative is running three
    benchmarks. A record without `judge_error` cannot be filtered downstream,
    and the filter is the whole fix."""
    src = (_BENCH / adapter).read_text(encoding="utf-8")
    assert '"judge_error"' in src, f"{adapter} records no judge_error field"
    assert '"judge_raw"' in src, f"{adapter} persists no judge_raw field"


def test_msc_no_longer_coerces_a_judge_error_into_a_wrong_answer():
    """MSC's record built `"correct": bool(correct)`, which turned an outage
    into a wrong answer at the point of writing — the deflation D3 names, in
    one call. LoCoMo's record already admitted None; MSC's did not."""
    src = (_BENCH / "msc_adapter.py").read_text(encoding="utf-8")
    assert '"correct": bool(correct)' not in src
    assert '"correct": (None if correct is None else bool(correct))' in src


def test_every_record_that_stores_a_verdict_carries_the_outage_channel():
    """Asserted from the SOURCE across all three adapters, via ast.

    The rule: a dict that records `correct` ALONGSIDE the question it scored (or
    alongside a prior verdict) must also carry `judge_raw` and `judge_error`. A
    record missing them cannot be filtered downstream, and the filter is the
    whole fix — a substring check on the file would pass while one of two record
    builders in the same module had lost the field.

    Deliberately excluded, and why: the exception records (`correct: False` next
    to `error`) are READER-side failures, which D3 does not cover, and the
    `{**r, "correct": ...}` re-score in the drift report is a projection of an
    existing record, not a new one.

    The count assertion is not decoration: an ast walk that matches nothing
    otherwise reads as a clean pass, and this file has already shipped one
    import test with exactly that hole."""
    found = 0
    for adapter in ("longmemeval_adapter.py", "locomo_adapter.py",
                    "msc_adapter.py"):
        tree = ast.parse((_BENCH / adapter).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if "correct" not in keys:
                continue
            if not ({"question", "correct_original"} & keys):
                continue
            found += 1
            assert "judge_raw" in keys, (
                f"{adapter}:{node.lineno} records a verdict without judge_raw")
            assert "judge_error" in keys, (
                f"{adapter}:{node.lineno} records a verdict without judge_error")
    assert found >= 4, (
        f"only {found} verdict-recording dicts found across the three adapters "
        "— the ast matcher has drifted and this test is certifying nothing")


def test_recall_diagnostics_does_not_count_an_unscored_row_as_a_MISS():
    """The miss decomposition is the actionable signal — it decides whether the
    next lever is retrieval, ranking or the reader. An unscored row entering the
    miss bucket would be attributed to whichever of those its `recall_ceiling`
    happens to say, so a judge outage would arrive dressed as a retrieval
    finding. That is the diagnostic-controls lesson exactly: a broken device
    returns a confident constant."""
    from longmemeval_adapter import compute_recall_diagnostics

    clean = [_row(qid="a", recall_ceiling=True, recall_tier="fts"),
             _row(qid="b", correct=False, recall_ceiling=False,
                  recall_tier="none")]
    with_outage = clean + [_row(qid="c", correct=None, judge_error=True,
                                recall_ceiling=False, recall_tier="none")]
    d_clean = compute_recall_diagnostics(clean)
    d_outage = compute_recall_diagnostics(with_outage)
    clean_cat = d_clean["single-session-user"]
    outage_cat = d_outage["single-session-user"]
    for key in (
        "known", "unknown", "ceiling_rate", "misses", "miss_retrieval",
        "miss_ranking", "miss_unknown",
    ):
        assert outage_cat[key] == clean_cat[key]
    assert outage_cat["benchmark_failures_excluded"] == 1
    assert d_outage["_tiers"] == d_clean["_tiers"]


def test_abstention_scores_do_not_count_an_unscored_row_as_a_FAILED_abstention():
    """The abstention split is where an outage would do the most damage.

    `compute_abstention_scores` coerces with `bool(r.get("correct"))`, so an
    unscored row lands as a FALSE in whichever arm it belongs to. On the
    abstention arm that reads as "the reader answered a question it should have
    refused" — a hallucination finding manufactured by a judge timeout. The
    answerable/abstention trade-off is exactly the comparison this benchmark
    exists to make, so a defect that biases only one arm does not cancel."""
    from longmemeval_adapter import compute_abstention_scores

    # The pinned evaluator derives abstention from substring membership in the
    # question ID.  The source question_type remains one of the six base types.
    clean = [_row(qid="a"),
             _row(qid="x_abs", qtype="single-session-user"),
             _row(qid="y_abs", qtype="single-session-user")]
    with_outage = clean + [_row(qid="z_abs", qtype="single-session-user",
                                correct=None, judge_error=True)]
    clean_abs = compute_abstention_scores(clean)["abstention"]
    outage_abs = compute_abstention_scores(with_outage)["abstention"]
    assert clean_abs["accuracy"] == 1.0 and clean_abs["count"] == 2
    assert clean_abs["benchmark_failures"] == 0
    assert outage_abs["accuracy"] == 2 / 3
    assert outage_abs["count"] == 3
    assert outage_abs["benchmark_failures"] == 1
    assert outage_abs["conditional_valid_accuracy"] == 1.0
    assert outage_abs["conditional_valid_count"] == 2


def test_the_run_file_instruments_also_refuse_to_score_an_unscored_row():
    """The adapters are not the only consumers of these records.

    `locomo_flip.py` and `locomo_audit.py` read the same `--out` files and are
    the LoCoMo gate instruments. Before this change `locomo_flip` would
    TypeError on a run containing one unscored row, and `locomo_audit` would
    hand it to the synthesis-bucket hand-check as a reader failure. Asserted
    from source, because both are argv-driven scripts.

    The flip case is the sharper one: an unscored row must be dropped from BOTH
    arms or from neither. Dropping it from one leaves the comparison unpaired on
    exactly the rows an outage touched, which is the C4 arm-asymmetry void
    condition arriving through the back door."""
    flip = (_BENCH / "locomo_flip.py").read_text(encoding="utf-8")
    assert 'a_rows[i].get("correct") is None' in flip
    assert 'or b_rows[i].get("correct") is None' in flip
    assert "dropped from BOTH arms" in flip

    audit = (_BENCH / "locomo_audit.py").read_text(encoding="utf-8")
    assert 'r.get("correct") is not None' in audit


def test_facts_ab_already_dropped_unpaired_rows_and_still_does():
    """A pre-existing guard, pinned rather than rewritten.

    `facts_ab.py` already skipped a pair where either arm had no verdict
    ("abstention/unjudged: no verdict to pair"), which is exactly the D3 rule
    reached independently. It needs no change — but it does need a test, or a
    later tidy-up removes it on the grounds that `correct` is always a bool."""
    src = (_BENCH / "facts_ab.py").read_text(encoding="utf-8")
    assert "if a is None or b is None:" in src
