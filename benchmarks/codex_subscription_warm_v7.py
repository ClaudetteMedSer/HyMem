"""Versioned warm transport admitting existing finite credits or included quota.

Only the billing gate of a private, hash-pinned v6 lineage changes. No RPC,
model, routing, turn parser, failure serializer, or prior source is changed.
"""
from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation
import hashlib
import math
from pathlib import Path
import re
import sys
import types
from typing import Any


PINNED_WARM_V6_SHA256 = "98422aa251ca9482a79be5851d48ae54decc17f17b6aa1149bd6b93b618b784b"
PINNED_BASE_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
BILLING_POLICY = "included_allowance_or_existing_finite_positive_credits_v1"

_v6_path = Path(__file__).resolve().with_name("codex_subscription_warm_v6.py")
_v6_source = _v6_path.read_bytes()
if hashlib.sha256(_v6_source).hexdigest() != PINNED_WARM_V6_SHA256:
    raise RuntimeError("pinned_warm_v6_source_mismatch")
v6 = types.ModuleType("pinned_codex_subscription_warm_v7_base")
v6.__file__ = str(_v6_path)
sys.modules[v6.__name__] = v6
exec(compile(_v6_source, str(_v6_path), "exec"), v6.__dict__)

base = v6.base
concurrent = v6.concurrent
BudgetLimits = v6.BudgetLimits
ConcurrentStop = v6.ConcurrentStop
SharedBudget = v6.SharedBudget
WarmSession = v6.WarmSession
WarmSubscriptionClient = v6.WarmSubscriptionClient
PrivateFailureSink = v6.PrivateFailureSink
serialize_failure = v6.serialize_failure

_BALANCE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z", re.ASCII)


def _existing_credits_available_v7(credits: Any) -> bool:
    """Accept only a provider-reported finite positive existing balance."""
    if credits is None:
        return False
    if (type(credits) is not dict or type(credits.get("hasCredits")) is not bool
            or type(credits.get("unlimited")) is not bool):
        base._fail("invalid_quota")
    balance = credits.get("balance")
    if balance is None:
        return False
    if type(balance) is not str or len(balance) > 128 or _BALANCE.fullmatch(balance) is None:
        base._fail("invalid_quota")
    try:
        amount = Decimal(balance)
        finite_float = float(amount)
    except (InvalidOperation, OverflowError, ValueError):
        base._fail("invalid_quota")
    if not amount.is_finite() or not math.isfinite(finite_float):
        base._fail("invalid_quota")
    return credits["hasCredits"] and not credits["unlimited"] and amount > 0 and finite_float > 0


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("pinned_quota_parser_shape_mismatch")
    return source.replace(old, new, 1)


def _install_private_billing_gate() -> None:
    source = Path(base.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != PINNED_BASE_SHA256:
        raise RuntimeError("pinned_quota_base_source_mismatch")
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=str(base.__file__))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "quota_metadata"]
    if len(functions) != 1:
        raise RuntimeError("pinned_quota_parser_shape_mismatch")
    quota_source = ast.get_source_segment(text, functions[0])
    if quota_source is None:
        raise RuntimeError("pinned_quota_parser_shape_mismatch")
    quota_source = _replace_once(quota_source, "    windows = []\n", "    windows = []\n    credit_eligible = []\n")
    quota_source = _replace_once(quota_source,
        '        if isinstance(credits, dict) and credits.get("hasCredits") is True:\n            _fail("credit_balance_present")',
        "        credit_available = _existing_credits_available_v7(credits)")
    quota_source = _replace_once(quota_source,
        '                            "resets_at": None if window.get("resetsAt") is None else _number(window["resetsAt"])})',
        '                            "resets_at": None if window.get("resetsAt") is None else _number(window["resetsAt"])})\n            credit_eligible.append(credit_available)')
    quota_source = _replace_once(quota_source,
        '                            "resets_at": _number(individual.get("resetsAt"))})',
        '                            "resets_at": _number(individual.get("resetsAt"))})\n            credit_eligible.append(credit_available)')
    quota_source = _replace_once(quota_source,
        '    if any(w["remaining_percent"] < 25 for w in windows):',
        '    if any(w["remaining_percent"] < 25 and not eligible\n'
        '           for w, eligible in zip(windows, credit_eligible)):\n')
    base.__dict__["_existing_credits_available_v7"] = _existing_credits_available_v7
    exec(compile(quota_source, str(base.__file__), "exec"), base.__dict__)


_install_private_billing_gate()
