"""Gate 5's scorer, checked before it is allowed to score anything.

The criteria live in `dream_cost_watch.evaluate` and were committed before the
run. What these tests defend is the property gate 4 turned out not to have: a
leg cannot claim an arm it did not send. `evaluate` gates on the digest prompt
the calls actually carried, never on the `granularity` field the runner wrote
into the artifact -- so a leg mislabelled by the operator, or by a bug in the
runner's own flag plumbing, reads FAIL rather than passing on its own say-so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import dream_cost_watch as dcw  # noqa: E402


def leg(name, *, granular, elapsed=10.0, blob=0, gran=0, other=2,
        sessions=110, stamped=None, stamped_before=0, episodes=1218,
        digest_failures=0, fusion_failures=0):
    if stamped is None:
        stamped = gran

    def _census(n):
        stamps = {}
        if n:
            stamps["episodes.granular.v1"] = n
        if sessions - n:
            stamps["NULL"] = sessions - n
        return {"sessions": sessions, "digested": sessions,
                "episodes": episodes, "stamps": stamps}

    census = _census(stamped)
    return {
        "leg": name, "granularity": granular, "elapsed_s": elapsed,
        "llm": {"calls": blob + gran + other, "digest_blob_calls": blob,
                "digest_granular_calls": gran, "digest_calls": blob + gran,
                "other_calls": other, "prompt_chars": 1000},
        "report": {"digest_failures": digest_failures,
                   "aggregation_fusion_failures": fusion_failures},
        "census_before": _census(stamped_before), "census_after": census,
    }


def passing_legs(**over):
    legs = {
        "settle": leg("settle", granular=False, blob=9),
        "before": leg("before", granular=False, blob=0, elapsed=8.0),
        "migrate": leg("migrate", granular=True, gran=110, elapsed=300.0,
                       episodes=1600),
        "after": leg("after", granular=True, gran=0, elapsed=9.0,
                     episodes=1600),
    }
    legs.update(over)
    return legs


def verdict(legs):
    return dcw.evaluate(legs)[0]


def failed(legs):
    return [n for n, ok, _ in dcw.evaluate(legs)[1] if not ok]


# ------------------------------------------------------------------ baseline

def test_a_clean_watch_passes():
    assert verdict(passing_legs()) == "PASS"


def test_a_missing_leg_is_INCOMPLETE_not_FAIL():
    """A leg that did not run is a gap in the measurement, not evidence
    against the feature -- the G-F1b ceiling rule, same reasoning."""
    legs = passing_legs()
    del legs["after"]
    v, checks = dcw.evaluate(legs)
    assert v == "INCOMPLETE"
    assert "after" in checks[0][2]


# ------------------------------------------------- the self-evidence criteria

def test_a_migrate_leg_that_sent_the_BLOB_prompt_fails():
    """The whole point. `granularity: True` is the claim under test; the
    prompt the calls carried is the evidence. A leg where the flag never
    reached the digest would otherwise be priced as a granular migration."""
    legs = passing_legs(migrate=leg("migrate", granular=True, blob=110, gran=0,
                                    elapsed=300.0))
    assert verdict(legs) == "FAIL"
    assert "migrate sent the GRANULAR digest prompt" in failed(legs)


def test_a_mixed_migrate_leg_fails():
    """Half the sessions under each prompt cannot be priced as either."""
    legs = passing_legs(migrate=leg("migrate", granular=True, blob=55, gran=55,
                                    elapsed=300.0))
    assert verdict(legs) == "FAIL"


def test_a_migrate_leg_that_made_no_digest_calls_at_all_fails():
    legs = passing_legs(migrate=leg("migrate", granular=True, gran=0, blob=0))
    assert verdict(legs) == "FAIL"


def test_a_migrate_leg_on_a_PARTLY_migrated_store_is_scored_on_the_delta():
    """40 sessions already stamped by an earlier attempt, 5 re-read now: the
    leg did 5 sessions' work and must be scored on 5, not on the 45 stamps
    that happen to be in the store. Reading the absolute count instead makes
    a correct resumption look like 45 stamps on 5 calls -- a phantom-stamp
    FAIL. Not hypothetical: the dress rehearsal left the snapshot with 40
    stamps on it."""
    legs = passing_legs(migrate=leg("migrate", granular=True, gran=5,
                                    stamped=45, stamped_before=40,
                                    elapsed=60.0))
    assert verdict(legs) == "PASS"


def test_empty_stub_sessions_do_not_sink_the_gate():
    """The production snapshot has 110 sessions of which only ~40 are
    digestible: the rest are empty stubs the dream skips before the digest,
    so they can never carry a stamp whatever the lever does. A criterion
    keyed on the session count read FAIL at 36% here -- charging the flip
    for the store's shape. The denominator is granular calls SENT."""
    legs = passing_legs(migrate=leg("migrate", granular=True, gran=40,
                                    stamped=40, sessions=110, elapsed=300.0))
    assert verdict(legs) == "PASS"


# ---------------------------------------------------------- the cost criteria

def test_a_migration_that_loops_over_sessions_fails():
    """400 calls against 110 stamped sessions is a loop, not a migration."""
    legs = passing_legs(migrate=leg("migrate", granular=True, gran=400,
                                    stamped=110, elapsed=300.0))
    assert verdict(legs) == "FAIL"
    assert any("stamps landed" in n for n in failed(legs))


def test_calls_that_land_no_stamp_fail_even_without_a_loop():
    """85 stamps for 100 calls: 15 sessions were re-read and not recorded,
    so they re-digest on every future cycle. Distinguishes the stamp
    criterion from the phantom-stamp one below -- an earlier cut of these
    two was one inequality written twice, and no test could tell them
    apart."""
    legs = passing_legs(migrate=leg("migrate", granular=True, gran=100,
                                    stamped=85, elapsed=300.0))
    assert verdict(legs) == "FAIL"
    assert any("stamps landed" in n for n in failed(legs))


def test_a_stamp_with_no_call_behind_it_fails():
    """The other side of the bound: 60 sessions marked migrated on 40 calls
    means 20 were stamped without being re-read."""
    legs = passing_legs(migrate=leg("migrate", granular=True, gran=40,
                                    stamped=60, elapsed=300.0))
    assert verdict(legs) == "FAIL"
    assert any("not re-read" in n for n in failed(legs))


def test_a_steady_state_that_never_settles_fails():
    """`after` still re-digesting every session is the failure mode gate 5
    exists to catch: a permanent per-cycle tax, not a one-time price."""
    legs = passing_legs(after=leg("after", granular=True, gran=110,
                                  elapsed=300.0))
    assert verdict(legs) == "FAIL"
    assert any("after digest calls" in n for n in failed(legs))


def test_a_small_steady_state_increase_is_tolerated():
    """Bounded, not zero: the store gains sessions between cycles."""
    legs = passing_legs(after=leg("after", granular=True, gran=3, elapsed=9.0))
    assert verdict(legs) == "PASS"


def test_the_wall_clock_floor_keeps_a_fast_baseline_fair():
    """before=3s, after=40s is API jitter on a near-empty cycle, not a
    regression; without the floor a 2x factor would call it one."""
    legs = passing_legs(before=leg("before", granular=False, elapsed=3.0),
                        after=leg("after", granular=True, gran=0, elapsed=40.0))
    assert verdict(legs) == "PASS"


def test_a_real_wall_clock_regression_still_fails():
    legs = passing_legs(before=leg("before", granular=False, elapsed=8.0),
                        after=leg("after", granular=True, gran=0, elapsed=500.0))
    assert verdict(legs) == "FAIL"


@pytest.mark.parametrize("kw", [{"digest_failures": 3}, {"fusion_failures": 2}])
def test_new_failure_modes_on_a_granular_leg_fail(kw):
    legs = passing_legs(after=leg("after", granular=True, gran=0, elapsed=9.0,
                                  **kw))
    assert verdict(legs) == "FAIL"


def test_failures_on_the_BLOB_legs_are_not_charged_to_the_flip():
    """`settle` and `before` are the store's own pre-existing state."""
    legs = passing_legs(before=leg("before", granular=False, elapsed=8.0,
                                   digest_failures=4))
    assert verdict(legs) == "PASS"


# ------------------------------------------------------------- the call counter

class _Req:
    def __init__(self, system, user=""):
        self.system, self.user = system, user


class _Inner:
    def __init__(self):
        self.seen = []

    def complete(self, request):
        self.seen.append(request.system)
        return "ok"


def test_the_counter_classifies_by_identity_not_by_substring():
    """The granular prompt must not be counted as the blob one because they
    share an opening clause -- and vice versa."""
    inner = _Inner()
    c = dcw.CountingLLM(inner, "BLOB PROMPT TEXT", "BLOB PROMPT TEXT plus more")
    c.complete(_Req("BLOB PROMPT TEXT plus more"))
    c.complete(_Req("BLOB PROMPT TEXT"))
    c.complete(_Req("something else entirely"))
    assert c.counts() == {
        "calls": 3, "digest_blob_calls": 1, "digest_granular_calls": 1,
        "digest_calls": 2, "other_calls": 1, "prompt_chars": 
            len("BLOB PROMPT TEXT plus more") + len("BLOB PROMPT TEXT")
            + len("something else entirely")}


def test_the_counter_passes_the_call_through():
    inner = _Inner()
    c = dcw.CountingLLM(inner, "B", "G")
    assert c.complete(_Req("G", "u")) == "ok"
    assert inner.seen == ["G"]
