"""Phase 2 model pin: extra_body plumbing, the empty-content raise, the guards.

The defect these pin down is not a crash. A DeepSeek reasoning model answers in
`reasoning_content` and returns `content: ""` with HTTP 200, so before this
change the adapter read an empty string, judge_answer's regex missed, the
except path returned {"score": 0.0, "scores": []}, and the run reported a
capability number that was really a plumbing failure — indistinguishable in the
score column from a real 0.0 (lme_runs.db id=53 0.6% vs id=54 69.8%, same day,
same model, extra_body the only difference).

So: the request must carry the flag, an empty completion must raise instead of
scoring, and a pin that would reproduce the trap must be refused BEFORE the run
spends anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import beam_adapter as ba  # noqa: E402

THINKING = {"thinking": {"type": "disabled"}}


class _Resp:
    """Minimal stand-in for the requests.Response the client actually reads."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _message(**msg):
    return {"choices": [{"message": msg, "finish_reason": "stop"}]}


@pytest.fixture
def captured(monkeypatch):
    """Capture the JSON body the client posts, without touching the network."""
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["url"] = url
        seen["body"] = json
        return _Resp(seen.get("reply", _message(content="ok")))

    monkeypatch.setattr(ba.http, "post", fake_post)
    return seen


# ── extra_body plumbing ───────────────────────────────────────────────────

def test_extra_body_is_merged_into_the_request(captured):
    ba.LLMClient("deepseek-v4-flash", "k", extra_body=THINKING)._call([], 0.0, 512)
    assert captured["body"]["thinking"] == {"type": "disabled"}


def test_unflagged_client_sends_exactly_what_it_sent_before(captured):
    """Comparability guard: the default must not change the request bytes.

    Every artifact before this commit was produced by a body with these four
    keys and no others. If the plumbing defaulted to injecting the thinking
    flag — the way the library client's `auto` mode does — those artifacts
    would stop being comparators without anyone deciding that.
    """
    ba.LLMClient("deepseek-chat", "k")._call([{"role": "user", "content": "q"}], 0.1, 99)
    assert captured["body"] == {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0.1,
        "max_tokens": 99,
    }


def test_official_judge_request_omits_unset_max_tokens(captured):
    ba.LLMClient("gpt-4.1-mini", "k")._call([], 0.0, None)
    assert captured["body"] == {
        "model": "gpt-4.1-mini", "messages": [], "temperature": 0.0,
    }


@pytest.mark.parametrize("key", ["model", "messages", "temperature", "max_tokens"])
def test_extra_body_cannot_override_manifested_request_identity(captured, key):
    with pytest.raises(ba.BenchmarkIntegrityError, match="core field"):
        ba.LLMClient("m", "k", extra_body={key: "forged"})
    assert "body" not in captured


# ── the empty-content raise ───────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    {"content": ""},                                   # the v4-flash trap shape
    {"content": "   "},                                # whitespace-only
    {"content": None},                                 # the LME null shape
    {},                                                # key absent entirely
    {"content": "", "reasoning_content": "long..."},   # trap, text misfiled
])
def test_empty_content_raises_instead_of_scoring_zero(captured, msg):
    captured["reply"] = _message(**msg)
    with pytest.raises(RuntimeError, match="empty content"):
        ba.LLMClient("deepseek-v4-flash", "k")._call([], 0.0, 512)


def test_chat_surfaces_the_empty_read_as_a_named_error(captured, monkeypatch):
    """chat() retries then returns a string; it must SAY what went wrong.

    An unnamed "[LLM_ERROR: ...]" would be scored as an explicit error and
    excluded — correct handling, but it would not tell anyone the pin was
    wrong. The raise text has to survive into the string a human reads.
    """
    monkeypatch.setattr(ba.time, "sleep", lambda *_: None)
    captured["reply"] = _message(content="", reasoning_content="x" * 40)
    out = ba.LLMClient("deepseek-v4-flash", "k").chat([])
    assert out.startswith("[LLM_ERROR")
    assert "empty content" in out
    assert "reasoning=40" in out


def test_real_content_passes_through_unchanged(captured):
    captured["reply"] = _message(content="  a real answer  ")
    assert ba.LLMClient("m", "k")._call([], 0.0, 512) == "  a real answer  "


# ── check_model_pin ───────────────────────────────────────────────────────

def test_v4_flash_without_thinking_disabled_is_refused():
    """The hole this phase closes: the main path had no guard at all."""
    with pytest.raises(SystemExit) as e:
        ba.check_model_pin("answer", "deepseek-v4-flash", "deepseek", {})
    assert e.value.code == 2


def test_v4_flash_with_thinking_disabled_is_allowed():
    ba.check_model_pin("judge", "deepseek-v4-flash", "deepseek", THINKING)


def test_v4_flash_with_thinking_enabled_is_still_refused():
    """Present-but-wrong is the same trap as absent, and easier to miss."""
    with pytest.raises(SystemExit):
        ba.check_model_pin("answer", "deepseek-v4-flash", "deepseek",
                           {"thinking": {"type": "enabled"}})


def test_the_working_alias_is_not_gated():
    """deepseek-chat is the currently-working path; it must stay unflagged."""
    ba.check_model_pin("judge", "deepseek-chat", "deepseek", {})


def test_thinking_key_is_refused_for_non_deepseek_providers():
    """`thinking` is a 400 on OpenAI/Gemini, and the answerer is swappable."""
    for provider in ("openai", "gemini"):
        with pytest.raises(SystemExit) as e:
            ba.check_model_pin("answer", "gpt-4o", provider, THINKING)
        assert e.value.code == 2


def test_non_deepseek_provider_without_thinking_is_fine():
    ba.check_model_pin("answer", "gemini-2.5-flash", "gemini", {})


def test_a_non_deepseek_model_named_v4_flash_is_not_gated_by_the_wrong_rule():
    """The v4-flash rule is DeepSeek-specific; it must not fire on a lookalike
    hosted elsewhere, where the fix it demands would itself be a 400."""
    ba.check_model_pin("answer", "vendor/v4-flash", "openai", {})


# ── run_canary ────────────────────────────────────────────────────────────

class _Stub:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def chat(self, messages, temperature=0.1, max_tokens=1024):
        self.seen = (messages, temperature, max_tokens)
        return self.reply


def test_canary_aborts_on_an_llm_error():
    with pytest.raises(SystemExit) as e:
        ba.run_canary("answer", _Stub("[LLM_ERROR: empty content (finish=length)]"), [], 1024)
    assert e.value.code == 1


def test_canary_aborts_on_empty_content():
    with pytest.raises(SystemExit):
        ba.run_canary("judge", _Stub("   "), [], 512)


def test_canary_returns_the_reply_and_uses_the_paths_own_ceiling():
    """max_tokens must be the real ceiling: the trap only reproduces when the
    model can burn the whole budget on reasoning, so a canary run at a smaller
    budget would be a different test that happens to pass."""
    stub = _Stub("a real judgement")
    assert ba.run_canary("judge", stub, [{"role": "user", "content": "x"}], 512) == "a real judgement"
    assert stub.seen[1] == 0.0 and stub.seen[2] == 512


# ── canary prompt ASSEMBLY ────────────────────────────────────────────────
# The tests above stub chat(), so they prove the canary sends and judges its
# reply but never that the messages are constructible. A wrong constant name
# or a question key that does not exist would pass every one of them and then
# kill a real run at that line -- after ingestion had been paid for. These
# exercise the real construction against the real module constants.

def _q(ability, **over):
    q = {"ability_short": ability, "question": f"a {ability} question?",
         "ideal_answer": f"legacy ideal for {ability}", "gold_text": f"real gold for {ability}",
         "rubric": ["states the fact", "gets the order right"]}
    q.update(over)
    return q


def _convs(*abilities):
    return {"100K": [{"id": "c1", "questions": [_q(a) for a in abilities]}]}


def test_canary_prefers_a_reasoning_heavy_ability_over_whichever_came_first():
    """The trap only reproduces when the model can burn its whole budget
    reasoning. An easy first question yields a canary that passes for the
    wrong reason and clears a pin that would have trapped on TR."""
    picked = ba.pick_canary_question(_convs("PF", "IF", "TR"), ["100K"])
    assert picked["ability_short"] == "TR"


def test_canary_ability_preference_is_ordered_not_merely_membership():
    assert ba.pick_canary_question(_convs("SUM", "MR"), ["100K"])["ability_short"] == "MR"
    assert ba.pick_canary_question(_convs("EO", "TR"), ["100K"])["ability_short"] == "TR"


def test_canary_falls_back_when_no_reasoning_ability_is_present():
    picked = ba.pick_canary_question(_convs("PF", "IF"), ["100K"])
    assert picked["ability_short"] == "PF"


def test_canary_refuses_an_empty_sample_rather_than_skipping_itself():
    """A canary that quietly no-ops on empty input is worse than none: the run
    would print no failure and proceed unprotected."""
    with pytest.raises(SystemExit) as e:
        ba.pick_canary_question({"100K": []}, ["100K"])
    assert e.value.code == 2


def test_both_canary_prompts_assemble_against_the_real_constants():
    msgs = ba.build_canary_messages(_convs("PF", "TR"), ["100K"], judge_gold=False)
    assert msgs["ability"] == "TR"
    system, user = msgs["answer"]
    assert system["content"] == ba.ANSWERING_PROMPTS.get("TR", ba.ANSWERING_SYSTEM_PROMPT)
    assert system["content"]
    assert "a TR question?" in user["content"] and user["content"].endswith("ANSWER:")
    judge = msgs["judge"]("some model output")
    assert judge and all("content" in m for m in judge)


def test_canary_judge_prompt_is_byte_identical_to_the_scoring_path():
    """The canary is only evidence about the judge if it is the SAME prompt
    the run will send; a divergent construction tests a path nothing uses."""
    q = _q("TR")
    msgs = ba.build_canary_messages(_convs("TR"), ["100K"], judge_gold=False)
    assert msgs["judge"]("out") == ba._judge_messages(
        q["question"], q["ideal_answer"], q["rubric"], "out")


def test_canary_judge_prompt_follows_the_runs_own_gold_choice():
    """--judge-gold changes the IDEAL field, which is exactly the variable the
    gold-delta phase measured. A canary on the other gold prompts differently
    than the run it is clearing."""
    gold_on = ba.build_canary_messages(_convs("TR"), ["100K"], judge_gold=True)["judge"]("out")
    gold_off = ba.build_canary_messages(_convs("TR"), ["100K"], judge_gold=False)["judge"]("out")
    assert "real gold for TR" in json.dumps(gold_on)
    assert "legacy ideal for TR" in json.dumps(gold_off)
    assert gold_on != gold_off


# ── finish_reason as structural evidence ──────────────────────────────────
# B2 (2026-09-01) voided on a silent-0 whose cause -- the judge scoring the row
# 1.0 and then running out of tokens mid-explanation -- was recoverable only by
# reading the raw text. A gate that must separate "the plumbing broke" from
# "the judge ran long" cannot rest on prose inspection, so the reason the
# response ended is recorded alongside it.

def _reply(content, finish):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def test_finish_reason_is_recorded_from_the_response(captured):
    captured["reply"] = _reply("ok", "stop")
    c = ba.LLMClient("m", "k")
    c._call([], 0.0, 512)
    assert c.last_finish_reason == "stop"


def test_finish_reason_survives_the_empty_content_raise(captured, monkeypatch):
    """The trap's signature is content == "" AND finish_reason == "length", so
    the field has to outlive the exception the empty content triggers."""
    monkeypatch.setattr(ba.time, "sleep", lambda *_: None)
    captured["reply"] = _reply("", "length")
    c = ba.LLMClient("deepseek-v4-flash", "k")
    assert c.chat([]).startswith("[LLM_ERROR")
    assert c.last_finish_reason == "length"


def test_finish_reason_is_cleared_per_call_so_it_cannot_go_stale(captured):
    """A value left over from an earlier row read as this row's would be worse
    than no value at all: absence must look like absence."""
    c = ba.LLMClient("m", "k")
    captured["reply"] = _reply("ok", "length")
    c.chat([])
    assert c.last_finish_reason == "length"

    def boom(*a, **k):
        raise OSError("network down")

    captured["reply"] = _reply("ok", "stop")
    import beam_adapter
    orig = beam_adapter.http.post
    beam_adapter.http.post = boom
    try:
        c.chat([])
    finally:
        beam_adapter.http.post = orig
    assert c.last_finish_reason is None


def test_a_truncated_judge_reply_scores_zero_but_says_why(captured):
    """The B2 row, reproduced. The judge scored it 1.0; the regex needs a
    complete {...} and the reply has no closing brace, so the score is 0.0 --
    indistinguishable from a real 0.0 in the score column alone. The recorded
    finish_reason is what makes it distinguishable."""
    truncated = ('{"scores": [1], "total_score": 1.0, "explanation": "The response '
                 'includes numeric error status codes')
    captured["reply"] = _reply(truncated, "length")
    llm = ba.LLMClient("deepseek-chat", "k")
    out = ba.judge_answer(llm, "q", "ideal", ["states the code"], "an answer",
                          return_raw=True)
    assert out["score"] == 0.0 and out["scores"] == []
    assert out["judge_finish_reason"] == "length"
    assert '"scores": [1]' in out["judge_raw"]


def test_a_complete_judge_reply_is_parsed_and_also_carries_its_finish(captured):
    captured["reply"] = _reply('{"scores": [1, 0], "total_score": 0.5}', "stop")
    llm = ba.LLMClient("deepseek-chat", "k")
    out = ba.judge_answer(llm, "q", "ideal", ["a", "b"], "ans", return_raw=True)
    assert out["score"] == 0.5 and out["scores"] == [1, 0]
    assert out["judge_finish_reason"] == "stop"


def test_judge_answer_still_accepts_a_bare_stub():
    """judge_answer is duck-typed: callers and tests pass objects exposing only
    chat(). Recording finish_reason must not narrow that contract from
    "anything with chat()" to "an LLMClient"."""
    class Stub:
        def chat(self, messages, temperature=None, max_tokens=None):
            return '{"scores": [1], "total_score": 1.0}'

    out = ba.judge_answer(Stub(), "q", "i", ["r"], "a", return_raw=True)
    assert out["score"] == 1.0
    assert out["judge_finish_reason"] is None


def test_usage_is_unavailable_after_failure_then_success(monkeypatch):
    replies = [OSError("network down"), _Resp({
        **_message(content="ok"),
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })]

    def post(*_args, **_kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(ba.http, "post", post)
    monkeypatch.setattr(ba.time, "sleep", lambda *_args: None)
    client = ba.LLMClient("m", "k")
    assert client.chat([]) == "ok"
    usage = ba.usage_snapshot(client)
    assert usage["request_attempts"] == 2
    assert usage["successful_responses"] == 1
    assert usage["calls"] == 1
    assert usage["total_tokens"] is None
    assert usage["token_usage_available"] is False


def test_usage_is_unavailable_when_every_attempt_fails(monkeypatch):
    monkeypatch.setattr(
        ba.http, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("network down")
        )
    )
    monkeypatch.setattr(ba.time, "sleep", lambda *_args: None)
    client = ba.LLMClient("m", "k")
    assert client.chat([]).startswith("[LLM_ERROR")
    usage = ba.usage_snapshot(client)
    assert usage["request_attempts"] == 3
    assert usage["successful_responses"] == 0
    assert usage["calls"] == 0
    assert usage["total_tokens"] is None
    assert usage["token_usage_available"] is False
