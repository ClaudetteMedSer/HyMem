"""Characterisation tests for `longmemeval_adapter.judge_answer` — the SINGLE
function that produces every canonical benchmark number in this repo.

Why this file exists
--------------------
`benchmarks/locomo_adapter.py:421` and `benchmarks/msc_adapter.py:502` both
import `judge_answer` from `longmemeval_adapter`. One function therefore scores
LoCoMo (68.2%), LME (68.4%) and MSC (~84.0%). It is two lines::

    raw = llm.chat(messages, temperature=0.0, max_tokens=10)
    return "yes" in raw.lower()

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
1. The DECISION RULE is `"yes" in raw.lower()` — an unanchored substring test.
   It is correct on compliant replies and on the empty reply, and wrong on
   "yes and no", "not yes", and — the amplifier nobody expected — any reply
   containing "yes" INSIDE another word ("yesterday", "eyes"). "yesterday" is
   not a hypothetical: temporal-reasoning is a scored category on both LME and
   LoCoMo, and the judge's own reasoning about it is exactly where that word
   appears. The "no" half of the reply is never consulted, so every conflict
   resolves to correct.
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
3. A judge-side `[LLM_ERROR: ...]` sentinel is scored, silently, as "wrong
   answer" — indistinguishable from a genuine "no". No caller checks it; the
   two re-judge paths check only the ANSWER for that sentinel.
4. `judge_answer` returns a bool and DISCARDS `raw`. That is precisely why
   points 1 and 3 have never been counted on any real run.
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
)

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


# ── 1. The decision rule: `"yes" in raw.lower()` ────────────────────────

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
    """CORRECT, pinned as correct. Note the last case: a verbose "no" survives
    only because it happens to contain no "yes" anywhere. It is correct by luck,
    not by construction — see the DEFECT tests below for the same shape losing."""
    assert _judge(reply) is False


def test_empty_reply_scores_incorrect():
    """CORRECT, pinned as correct — and the safe direction. An empty judge reply
    fails closed (scores the answer wrong) rather than open."""
    assert _judge("") is False


def test_DEFECT_ambiguous_yes_and_no_scores_correct():
    """The judge explicitly declining to commit is scored as a correct answer."""
    assert _judge("yes and no") is True
    assert _judge("Yes and no — it depends.") is True


def test_DEFECT_negated_yes_scores_correct():
    """A NEGATED affirmative scores correct. The rule reads the token, never the
    polarity, so the judge's meaning is inverted."""
    assert _judge("not yes") is True
    assert _judge("NOT YES") is True
    assert _judge("The correct answer here is not yes.") is True


@pytest.mark.parametrize("reply", [
    "The model refers to yesterday, so no.",
    "No — the response only mentions what she did yesterday.",
    "no, the model just describes eyes",
])
def test_DEFECT_substring_rule_has_no_word_boundary(reply):
    """The highest-exposure defect, and the one that is NOT hypothetical.

    `"yes" in raw.lower()` is unanchored, so "yesterday" and "eyes" fire it. Each
    reply here is an unambiguous NO from the judge and each scores CORRECT.

    Why this is not a curiosity: temporal-reasoning is a scored category on both
    LME (`temporal-reasoning`) and LoCoMo (`category 2`), and its judge prompt
    invites day-arithmetic reasoning ("do not penalize off-by-one errors for the
    number of days"). "yesterday" is the single likeliest word in a
    non-compliant temporal judge reply, and it is a false CORRECT every time.
    A word-boundary match (`\\byes\\b`) would reject all three.
    """
    assert _judge(reply) is True


def test_DEFECT_the_no_half_of_a_reply_is_never_consulted():
    """There is no "no" test at all, so no conflict can ever be detected and
    every conflict resolves to CORRECT regardless of order. `raw.strip()` alone —
    reading only the FIRST token — would score both of these as the judge meant."""
    assert _judge("no\n\nyes") is True
    assert _judge("yes\n\nno") is True
    assert _judge("No. Well, actually, yes.") is True


@pytest.mark.parametrize("reply,expected", [
    # A reasoning preamble cut off at ~10 tokens, carrying no verdict at all.
    ("Let me analyse the model response against the correct", False),
    # The same shape, cut off just after a "yes" that is not the verdict.
    ("The question is whether a yes would be", True),
])
def test_DEFECT_truncated_reply_is_scored_as_a_verdict(reply, expected):
    """`max_tokens=10` truncates any non-compliant reply mid-sentence, and the
    fragment is then scored AS IF it were a verdict — silently, in whichever
    direction the fragment happens to fall.

    This is the failure mode the deepseek-v4-flash migration already hit from
    the other side (a reasoning preamble corrupting the yes/no parse, worked
    around with `thinking: disabled` rather than by hardening the parse), so the
    exposure is documented as real for this judge family, not theoretical."""
    assert _judge(reply) is expected


def test_DEFECT_judge_side_llm_error_is_scored_as_a_wrong_answer():
    """`LLMClient.chat` returns `f"[LLM_ERROR: {...}]"` after exhausting its
    retries (longmemeval_adapter.py:349). `judge_answer` scores that sentinel
    exactly like a genuine "no": the question is recorded WRONG and the outage
    is invisible in the per-question row.

    The two re-judge paths (`_rejudge_run`, `_rejudge_file`) DO test for this
    sentinel — but only on the ANSWER (`hypothesis` / `ai_answer`). Neither
    inspects the judge's reply, and `judge_answer` has already discarded it.
    A judge-side outage streak therefore deflates the score of the arm it hit."""
    error_verdict = _judge("[LLM_ERROR: Connection reset by peer]")
    genuine_no = _judge("no")
    assert error_verdict is False
    assert error_verdict == genuine_no, (
        "an outage sentinel and a genuine 'no' are indistinguishable at the "
        "call site — that indistinguishability IS the defect"
    )


def test_DEFECT_raw_reply_is_discarded():
    """The reason `benchmarks/judge_audit.py` has to exist.

    `judge_answer` returns a bare bool, so the rate of non-compliant replies is
    not merely unmeasured — it is unmeasurABLE from any stored run.

    The three replies here are deliberately all COMPLIANT affirmatives that
    differ only in formatting. That isolates the claim: this test is about the
    return type erasing `raw`, not about the substring rule (which the DEFECT
    tests above own). Mixing a defective reply in here would make this test fail
    for two different reasons and blur the negative control."""
    replies = ["yes", "Yes.", "  YES  \n"]
    assert len(set(replies)) == 3
    verdicts = [_judge(r) for r in replies]
    assert verdicts == [True, True, True]
    assert len(set(verdicts)) == 1, "raw reply survived into the return value"


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

def _import_sources(path: Path, name: str) -> list[ast.ImportFrom]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and n.module == "longmemeval_adapter"
            and any(a.name == name for a in n.names)]


@pytest.mark.parametrize("adapter", ["locomo_adapter.py", "msc_adapter.py"])
def test_locomo_and_msc_import_the_lme_judge(adapter):
    """One function, three canonical baselines."""
    hits = _import_sources(_BENCH / adapter, "judge_answer")
    assert len(hits) >= 1, (
        f"{adapter} no longer imports judge_answer from longmemeval_adapter — "
        "the single-judge premise of this audit has changed"
    )


def test_beam_is_NOT_in_the_blast_radius():
    """Scope guard, so a re-baseline is not over-scoped. BEAM defines its own
    rubric judge returning a dict and imports nothing from the LME judge."""
    assert _import_sources(_BENCH / "beam_adapter.py", "judge_answer") == []
    tree = ast.parse((_BENCH / "beam_adapter.py").read_text(encoding="utf-8"))
    own = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "judge_answer"]
    assert len(own) == 1, "beam_adapter no longer defines its own judge"


def _judge_call_type_args(path: Path) -> list[str | None]:
    """The 2nd positional arg (`question_type`) of every `judge_answer(...)` call,
    as a literal where it is one and None where it is computed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str | None] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "judge_answer"
                and len(node.args) >= 2):
            arg = node.args[1]
            out.append(arg.value if isinstance(arg, ast.Constant) else None)
    return out


def test_msc_routes_every_question_through_the_containment_criterion():
    """MSC's exposure is TOTAL, not partial: it hard-codes `single-session-user`
    at its only judge call site, so no MSC row ever reaches the correct-looking
    `_abs` branch. The ~84.0% baseline is 100% containment-judged."""
    args = _judge_call_type_args(_BENCH / "msc_adapter.py")
    assert args, "no judge_answer call found in msc_adapter.py"
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

import judge_audit as JA  # noqa: E402


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


@pytest.mark.parametrize("raw", ["yes", "No.", "", "[LLM_ERROR: boom]",
                                 "yes and no", "not yes", "no\n\nyes",
                                 "the model just describes eyes",
                                 "Let me analyse the model response against"])
def test_audit_shipping_rule_matches_judge_answer_exactly(raw):
    """Cross-check against the REAL function rather than against a restatement
    of it. This is the join that stops the audit drifting from its subject."""
    assert JA.shipping_verdict(raw) is _judge(raw)


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
