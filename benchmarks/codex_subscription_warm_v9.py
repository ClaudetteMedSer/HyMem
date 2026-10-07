"""Source-pinned warm transport with consistent existing-credit admission.

The v8 parser already validates finite existing credits per quota snapshot. This
version carries that result to the shared ledger for the corresponding windows.
No balance, substituted percentage, or provider response is recorded in the
admission result.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import types
from typing import Any
import weakref


PINNED_WARM_V8_SHA256 = "0a55d44053349eb90511a597dae20f295c19bb53c734a78b5db3c41197343c21"
BILLING_POLICY = "included_allowance_or_existing_finite_positive_credits_per_window_v2"

_v8_path = Path(__file__).resolve().with_name("codex_subscription_warm_v8.py")
_v8_source = _v8_path.read_bytes()
if hashlib.sha256(_v8_source).hexdigest() != PINNED_WARM_V8_SHA256:
    raise RuntimeError("pinned_warm_v8_source_mismatch")
v8 = types.ModuleType("pinned_codex_subscription_warm_v9_base")
v8.__file__ = str(_v8_path)
sys.modules[v8.__name__] = v8
exec(compile(_v8_source, str(_v8_path), "exec"), v8.__dict__)

base = v8.base
concurrent = v8.concurrent
BudgetLimits = v8.BudgetLimits
ConcurrentStop = v8.ConcurrentStop
PrivateFailureSink = v8.PrivateFailureSink
WarmSession = v8.WarmSession
serialize_failure = v8.serialize_failure
NOTIFICATION_POLICY = v8.NOTIFICATION_POLICY
OPT_OUT_NOTIFICATION_METHODS = v8.OPT_OUT_NOTIFICATION_METHODS
OPT_OUT_FAILURE_CODE = v8.OPT_OUT_FAILURE_CODE

_validated_quota_metadata = base.quota_metadata
_credit_proof = object()


class _QuotaWindow(dict):
    """A JSON-safe quota window with an opaque, parser-issued credit proof."""

    __slots__ = ("_proof", "_source_values", "__weakref__")

    def __init__(self, window: dict[str, Any]):
        super().__init__(window)
        self._proof = (_credit_proof, weakref.ref(self))
        self._source_values = tuple(window.items())


def _snapshot_windows(snapshot: dict[str, Any]) -> int:
    """Count windows with the same presence rule as the pinned quota parser."""
    return sum(snapshot.get(key) is not None for key in
               ("primary", "secondary", "individualLimit"))


def quota_metadata(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain v8 validation and mark only verified windows in their own bucket."""
    windows = _validated_quota_metadata(response)
    snapshots = response.get("rateLimitsByLimitId")
    if isinstance(snapshots, dict) and snapshots:
        snapshots = list(snapshots.values())
    else:
        snapshots = [response.get("rateLimits")]
    marked: list[dict[str, Any]] = []
    offset = 0
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            base._fail("invalid_quota")
        count = _snapshot_windows(snapshot)
        eligible = v8.v7._existing_credits_available_v7(snapshot.get("credits"))
        section = windows[offset:offset + count]
        if len(section) != count:
            base._fail("invalid_quota")
        marked.extend(_QuotaWindow(window) if eligible else window for window in section)
        offset += count
    if offset != len(windows):
        base._fail("invalid_quota")
    return marked


# These are the private globals loaded from a hash-pinned v8 source. Both
# preflight and account/rate-limit notifications resolve this same parser.
base.quota_metadata = quota_metadata


def _quota_windows_admissible(windows: Any) -> bool:
    if not isinstance(windows, list) or not windows:
        return False
    for window in windows:
        if not isinstance(window, dict):
            return False
        remaining = window.get("remaining_percent")
        if (isinstance(remaining, bool) or not isinstance(remaining, (int, float))
                or not math.isfinite(remaining) or not 0 <= remaining <= 100):
            return False
        if type(window) is _QuotaWindow:
            if (type(window._proof) is not tuple or len(window._proof) != 2
                    or window._proof[0] is not _credit_proof
                    or window._proof[1]() is not window
                    or tuple(window.items()) != window._source_values):
                return False
        elif remaining < 25:
            return False
    return True


class SharedBudget(v8.SharedBudget):
    """V8 ledger with its original limits and a consistent quota predicate."""

    def before_turn(self, question_id: str, admission: dict[str, Any]) -> float:
        with self._lock:
            q = self._questions[question_id]
            if q.in_flight != 1 or q.reserved != 1 or self.reserved < 1:
                self.halt("ledger_protocol_violation")
                concurrent._stop("ledger_protocol_violation")
            if self.stopped or q.stopped:
                concurrent._stop("budget_stopped_before_turn")
            if self._remaining(q) <= 0:
                if self.clock() - self.started_at >= self.limits.seconds:
                    self.halt("campaign_wall_limit")
                else:
                    q.stopped = True
                concurrent._stop("wall_limit")
            if (self.known_tokens >= self.limits.known_tokens
                    or q.known_tokens >= q.limits.known_tokens):
                if self.known_tokens >= self.limits.known_tokens:
                    self.halt("campaign_budget_exhausted")
                else:
                    q.stopped = True
                concurrent._stop("budget_exhausted_before_turn")
            if (admission.get("auth") != "chatgpt" or admission.get("model") != base.MODEL
                    or admission.get("config_isolation_admitted") is not True
                    or admission.get("inference_enabled") is not False):
                self.halt("admission_rejected")
                concurrent._stop("admission_rejected")
            if not _quota_windows_admissible(admission.get("quota_windows")):
                self.halt("quota_unverified")
                concurrent._stop("quota_unverified")
            self.reserved -= 1
            q.reserved -= 1
            self.turns += 1
            q.turns += 1
            return min(120.0, self._remaining(q))


concurrent.SharedBudget = SharedBudget


class WarmSubscriptionClient(v8.WarmSubscriptionClient):
    def __init__(self, *args: Any, session_factory: Any = WarmSession, **kwargs: Any):
        super().__init__(*args, session_factory=session_factory, **kwargs)
