"""Root regressions for credit proof reaching the shared admission boundary."""
import ast
from copy import copy, deepcopy
from pathlib import Path

import pytest

from benchmarks import codex_subscription_warm_v8 as old
from benchmarks import codex_subscription_warm_v9 as fixed


def snapshot(remaining=1, credits=None):
    return {"planType": "pro", "credits": credits,
            "primary": {"usedPercent": 100 - remaining,
                        "windowDurationMins": 300, "resetsAt": 1790784000}}


POSITIVE = {"hasCredits": True, "unlimited": False, "balance": "2.0"}


def admission(module, windows):
    return {"auth": "chatgpt", "model": module.base.MODEL,
            "config_isolation_admitted": True, "inference_enabled": False,
            "quota_windows": windows}


def reserved(module, turns=16):
    budget = module.SharedBudget(module.BudgetLimits(turns, 160000, 600), max_in_flight=4)
    budget.register("root", module.BudgetLimits(4, 160000, 600))
    budget.reserve("root")
    return budget


def test_root_actual_old_failure_and_new_parser_to_ledger_success():
    raw = {"rateLimits": snapshot(credits=POSITIVE)}
    previous = old.base.quota_metadata(deepcopy(raw))
    budget = reserved(old)
    with pytest.raises(old.ConcurrentStop):
        budget.before_turn("root", admission(old, previous))
    assert budget.snapshot()["stop_code"] == "quota_unverified"
    windows = fixed.base.quota_metadata(deepcopy(raw))
    assert windows[0]["remaining_percent"] == 1
    budget = reserved(fixed)
    assert budget.before_turn("root", admission(fixed, windows)) > 0
    assert budget.snapshot()["turns"] == 1
    budget.settle("root", used=19, turn_started=True)
    state = budget.snapshot()
    assert state["known_tokens"] == 19 and state["usage_complete"] is True
    assert state["reserved"] == state["in_flight"] == 0
    assert not state["stopped"]
    # Loading the fix must not retrofit the frozen old module.
    old_again = reserved(old)
    with pytest.raises(old.ConcurrentStop):
        old_again.before_turn("root", admission(old, previous))


@pytest.mark.parametrize("credits", [None, {**POSITIVE, "balance": "0"},
    {**POSITIVE, "balance": "-1"}, {**POSITIVE, "unlimited": True},
    {**POSITIVE, "balance": "NaN"}, {**POSITIVE, "balance": "Infinity"},
    {**POSITIVE, "balance": "1e9999"}, {**POSITIVE, "balance": 2},
    {**POSITIVE, "hasCredits": 1}, {**POSITIVE, "hasCredits": False}])
def test_root_low_allowance_requires_valid_finite_positive_credit(credits):
    with pytest.raises(fixed.base.SubscriptionTransportError, match="invalid_quota|quota_floor"):
        fixed.base.quota_metadata({"rateLimits": snapshot(credits=credits)})


@pytest.mark.parametrize("denial", [{"spendControlReached": True}, {"rateLimitReachedType": "credits"}])
def test_root_provider_denial_overrides_positive_credits(denial):
    with pytest.raises(fixed.base.SubscriptionTransportError, match="quota_exhausted"):
        fixed.base.quota_metadata({"rateLimits": {**snapshot(credits=POSITIVE), **denial}})


def test_root_credit_authorization_does_not_leak_to_another_bucket():
    raw = {"rateLimitsByLimitId": {"a": snapshot(credits=POSITIVE), "b": snapshot()}}
    with pytest.raises(fixed.base.SubscriptionTransportError, match="quota_floor"):
        fixed.base.quota_metadata(raw)
    raw["rateLimitsByLimitId"]["b"] = snapshot(remaining=90)
    windows = fixed.base.quota_metadata(raw)
    assert sorted(w["remaining_percent"] for w in windows) == [1, 90]
    budget = reserved(fixed)
    assert budget.before_turn("root", admission(fixed, windows)) > 0


def test_root_unverified_low_window_cannot_grant_credit():
    budget = reserved(fixed)
    with pytest.raises(fixed.ConcurrentStop):
        budget.before_turn("root", admission(fixed, [{"remaining_percent": 1,
            "existing_credits_verified": True, "credit_eligible": True}]))
    assert budget.snapshot()["turns"] == 0


def test_root_credit_proof_cannot_be_copied_into_an_unissued_window():
    windows = fixed.base.quota_metadata({"rateLimits": snapshot(credits=POSITIVE)})
    forged = dict(windows[0])
    budget = reserved(fixed)
    with pytest.raises(fixed.ConcurrentStop):
        budget.before_turn("root", admission(fixed, [forged]))
    assert budget.snapshot()["turns"] == 0


def test_root_shallow_copy_cannot_duplicate_credit_authority():
    windows = fixed.base.quota_metadata({"rateLimits": snapshot(credits=POSITIVE)})
    budget = reserved(fixed)
    with pytest.raises(fixed.ConcurrentStop):
        budget.before_turn("root", admission(fixed, [copy(windows[0])]))
    assert budget.snapshot()["turns"] == 0


@pytest.mark.parametrize("remaining", [2, 30])
def test_root_mutating_verified_window_cannot_preserve_credit_authority(remaining):
    windows = fixed.base.quota_metadata({"rateLimits": snapshot(credits=POSITIVE)})
    budget = reserved(fixed)
    try:
        windows[0]["remaining_percent"] = remaining
    except (TypeError, AttributeError):
        return  # Immutable verified windows also preserve the boundary.
    with pytest.raises(fixed.ConcurrentStop):
        budget.before_turn("root", admission(fixed, windows))
    assert budget.snapshot()["turns"] == 0


@pytest.mark.parametrize("remaining", [25, 80, 100])
def test_root_included_allowance_needs_no_credit(remaining):
    windows = fixed.base.quota_metadata({"rateLimits": snapshot(remaining=remaining)})
    budget = reserved(fixed)
    assert budget.before_turn("root", admission(fixed, windows)) > 0


def test_root_credit_does_not_override_shared_stop_or_local_cap():
    windows = fixed.base.quota_metadata({"rateLimits": snapshot(credits=POSITIVE)})
    budget = reserved(fixed, turns=1)
    budget.before_turn("root", admission(fixed, windows))
    budget.settle("root", used=19, turn_started=True)
    with pytest.raises(fixed.ConcurrentStop):
        budget.reserve("root")
    another = reserved(fixed)
    another.halt("campaign_wall_limit")
    with pytest.raises(fixed.ConcurrentStop):
        another.before_turn("root", admission(fixed, windows))
    assert another.snapshot()["turns"] == 0


def test_root_all_nonquota_before_turn_guards_are_unchanged():
    def method(path):
        tree = ast.parse(Path(path).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SharedBudget")
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "before_turn")
        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body.pop(0)
        return node
    previous = method(Path(__file__).resolve().parents[1] / "benchmarks/codex_subscription_concurrent_v2.py")
    current = method(fixed.__file__)
    body = previous.body[0].body
    position = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "windows")
    del body[position]
    body[position].test = ast.parse("not _quota_windows_admissible(admission.get('quota_windows'))", mode="eval").body
    class QualifyStop(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "_stop":
                return ast.Attribute(value=ast.Name(id="concurrent", ctx=ast.Load()), attr="_stop", ctx=node.ctx)
            return node
    previous = QualifyStop().visit(previous)
    assert ast.dump(previous, include_attributes=False) == ast.dump(current, include_attributes=False)
