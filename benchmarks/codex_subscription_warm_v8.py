"""Source-pinned warm transport requesting the documented delta opt-out.

The authoritative completed final item and usage remain subject to the frozen
turn parser. A server that ignores the opt-out fails before a delta can consume
the parser's finite event allowance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types
from typing import Any


PINNED_WARM_V7_SHA256 = "94234eca1daeb542a8d8f92a5b7178ba8b91060f33415f76343bd13e6d5953d9"
NOTIFICATION_POLICY = "agent_message_delta_optout_v1"
OPT_OUT_NOTIFICATION_METHODS = ("item/agentMessage/delta",)
OPT_OUT_FAILURE_CODE = "notification_optout_unverified"

_v7_path = Path(__file__).resolve().with_name("codex_subscription_warm_v7.py")
_v7_source = _v7_path.read_bytes()
if hashlib.sha256(_v7_source).hexdigest() != PINNED_WARM_V7_SHA256:
    raise RuntimeError("pinned_warm_v7_source_mismatch")
v7 = types.ModuleType("pinned_codex_subscription_warm_v8_base")
v7.__file__ = str(_v7_path)
sys.modules[v7.__name__] = v7
exec(compile(_v7_source, str(_v7_path), "exec"), v7.__dict__)

base = v7.base
concurrent = v7.concurrent
BudgetLimits = v7.BudgetLimits
ConcurrentStop = v7.ConcurrentStop
SharedBudget = v7.SharedBudget
PrivateFailureSink = v7.PrivateFailureSink
BILLING_POLICY = v7.BILLING_POLICY

# This is the cloned, hash-pinned v2 diagnostic vocabulary. The original import
# and source files remain untouched; the new code survives the existing public
# serializer and budget failure path without exposing event content.
v7.v6.v5.v4.v3.v2._FIXED_CODES = (
    v7.v6.v5.v4.v3.v2._FIXED_CODES | {OPT_OUT_FAILURE_CODE}
)
serialize_failure = v7.serialize_failure


class WarmSession(v7.WarmSession):
    def send(self, method: str, params: dict[str, Any], *, notification: bool = False) -> int:
        if method == "initialize":
            if type(params) is not dict or type(params.get("capabilities")) is not dict:
                base._fail(OPT_OUT_FAILURE_CODE)
            capabilities = dict(params["capabilities"])
            capabilities["optOutNotificationMethods"] = list(OPT_OUT_NOTIFICATION_METHODS)
            params = {**params, "capabilities": capabilities}
        return super().send(method, params, notification=notification)

    def receive(self) -> dict[str, Any]:
        event = super().receive()
        if event.get("method") in OPT_OUT_NOTIFICATION_METHODS:
            base._fail(OPT_OUT_FAILURE_CODE)
        return event

    def next_event(self) -> dict[str, Any]:
        event = super().next_event()
        # Also cover a notification already held in the turn/start queue.
        if event.get("method") in OPT_OUT_NOTIFICATION_METHODS:
            base._fail(OPT_OUT_FAILURE_CODE)
        return event

    def retire_pending(self) -> None:
        # The frozen cleanup router can consume pending notifications directly.
        if any(event.get("method") in OPT_OUT_NOTIFICATION_METHODS for event in self.pending):
            base._fail(OPT_OUT_FAILURE_CODE)
        super().retire_pending()


class WarmSubscriptionClient(v7.WarmSubscriptionClient):
    def __init__(self, *args: Any, session_factory: Any = WarmSession, **kwargs: Any):
        super().__init__(*args, session_factory=session_factory, **kwargs)
