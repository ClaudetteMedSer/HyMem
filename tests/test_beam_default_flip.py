"""Model-pin pre-reg §6: the default flip, and the two ways it could go wrong.

The flip is not the plumbing. The plumbing (e3313c0) is inert by design; this
is the change that decides what an UNFLAGGED run means, which is why the
pre-registration insisted it land as its own commit after Step 1 passed.

Two constraints in §6 are load-bearing, and each has a test here that fails if
it is violated:

1. **Scope.** The match term is `"v4-flash" in model`, not `"deepseek" in
   model`. The wide term is what the library client uses, and here it would
   silently add `thinking` to the `deepseek-chat` path -- changing A/B
   byte-identity and retiring the comparator that the whole gold-delta series
   is measured against.
2. **Absent is not empty.** `--judge-extra-body ''` is the operator saying "no
   extra body". A convenience that overrides an explicit statement is a bug,
   even when the statement is one the guard will refuse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import beam_adapter as ba  # noqa: E402

DISABLED = {"thinking": {"type": "disabled"}}


def test_the_defaults_are_now_the_pin():
    """§6's headline. Step 1 passed, so an unflagged run means the pin."""
    assert ba.ANSWER_MODEL == "deepseek-v4-flash"
    assert ba.JUDGE_MODEL == "deepseek-v4-flash"


def test_absent_flag_on_v4_flash_defaults_to_thinking_disabled():
    obj, fired = ba.apply_thinking_default("judge", "deepseek-v4-flash", "deepseek",
                                           True, {})
    assert fired is True
    assert obj == DISABLED


def test_the_default_is_a_copy_not_the_shared_constant():
    """A caller mutating one run's extra_body must not edit every later run's."""
    obj, _ = ba.apply_thinking_default("judge", "deepseek-v4-flash", "deepseek",
                                       True, {})
    obj["thinking"]["type"] = "enabled"
    assert ba.THINKING_DISABLED == DISABLED


# --- constraint 1: scope ----------------------------------------------------

def test_it_never_fires_on_the_alias():
    """THE test for the scope rule. `deepseek-chat` is the comparator A and B
    were produced on; adding a body to it would change the bytes and retire
    them, which is precisely what the wide `"deepseek" in model` term does."""
    obj, fired = ba.apply_thinking_default("judge", "deepseek-chat", "deepseek",
                                           True, {})
    assert fired is False
    assert obj == {}


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner",
                                   "deepseek-v3", "deepseek-chat-v4"])
def test_it_never_fires_on_any_non_v4_flash_deepseek(model):
    _, fired = ba.apply_thinking_default("judge", model, "deepseek", True, {})
    assert fired is False


@pytest.mark.parametrize("provider", ["openai", "gemini", "openrouter"])
def test_it_never_fires_off_deepseek(provider):
    """`thinking` aimed at another provider 400s, and check_model_pin refuses
    it. The default must not manufacture the thing the guard exists to catch."""
    _, fired = ba.apply_thinking_default("answer", "some-v4-flash-clone",
                                         provider, True, {})
    assert fired is False


# --- constraint 2: absent is not empty --------------------------------------

@pytest.mark.parametrize("obj", [{}, {"temperature": 0}])
def test_an_explicit_flag_is_never_overridden(obj):
    """`raw` was `''` or `'{}'` or real JSON -- the operator spoke, so §6 stays
    out of it, even when what they asked for cannot work."""
    got, fired = ba.apply_thinking_default("judge", "deepseek-v4-flash", "deepseek",
                                           False, dict(obj))
    assert fired is False
    assert got == obj


def test_an_explicitly_empty_body_still_reaches_the_guard_and_is_refused():
    """The pair that makes constraint 2 safe rather than merely principled:
    §6 declines to fire, and check_model_pin then refuses the run. The operator
    gets an error, not a silently rewritten request."""
    obj, fired = ba.apply_thinking_default("judge", "deepseek-v4-flash", "deepseek",
                                           False, {})
    assert fired is False
    with pytest.raises(SystemExit) as e:
        ba.check_model_pin("judge", "deepseek-v4-flash", "deepseek", obj)
    assert e.value.code == 2


# --- constraint 1 (ordering): the default must precede the guard ------------

def test_the_defaulted_body_passes_the_guard():
    """§6's ordering constraint, stated as its consequence: after defaulting, a
    bare v4-flash run is legal. If the default were applied AFTER the guard the
    run would already have exited 2 and §6 would silently do nothing."""
    obj, fired = ba.apply_thinking_default("answer", "deepseek-v4-flash", "deepseek",
                                           True, {})
    assert fired is True
    ba.check_model_pin("answer", "deepseek-v4-flash", "deepseek", obj)


def test_the_default_fires_in_main_before_check_model_pin(monkeypatch, capsys):
    """The ordering itself, read off the source rather than argued for: the
    §6 call site sits between the parse loop and the guard."""
    src = (_BENCH / "beam_adapter.py").read_text()
    loop = src.index('setattr(args, f"{role}_extra_body_absent"')
    judge_default = src.index("args.judge_extra_body_obj, _judge_defaulted")
    answer_default = src.index("args.answer_extra_body_obj, _answer_defaulted")
    answer_guard = src.index('check_model_pin("answer", ans_model, ans_provider')
    assert loop < judge_default < answer_default < answer_guard
