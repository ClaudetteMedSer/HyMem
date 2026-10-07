"""Offline privacy, gate, and protocol controls for the credit metadata reader."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "luna_credit_metadata_v1.py"
SPEC = importlib.util.spec_from_file_location("luna_credit_metadata_v1", SOURCE)
assert SPEC and SPEC.loader
credit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(credit)
BASE = Path(__file__).resolve().parents[3] / "benchmarks/codex_subscription.py"
HELPER = Path(__file__).resolve().parents[1] / "luna_subscription_access_metadata_v1.py"
helper = credit._module(HELPER.read_bytes(), credit.HELPER_SHA256, "_pinned_helper_test")
SESSION, QUOTA, GATE_ERROR = helper["_load_transport"](BASE.read_bytes())


def response(credits: object = None) -> dict:
    return {"rateLimitsByLimitId": {"SECRET_ID": {"credits": credits,
        "primary": {"usedPercent": 80, "windowDurationMins": 300,
                    "resetsAt": 1_800_000_000},
        "secondary": {"usedPercent": 40},
        "rateLimitReachedType": None, "spendControlReached": None,
        "limitName": "SECRET_NAME"}}}


def stopped() -> dict:
    return {"schema": "luna-subscription-access-check-v1", "status": "access_failed",
        "runtime": "failed_exit", "recursive_cleanup_verified": True,
        "result": {"turns": 0, "usage_complete": True,
                   "first_failure": {"code": "credit_balance_present"}}}


class CreditMetadataTests(unittest.TestCase):
    def test_projection_credit_and_quota_are_independent(self):
        for balance, expected in (("0", "zero"), ("+1.25", "positive"),
                                  ("-.5", "negative"), (None, "null"),
                                  ("SECRET", "invalid")):
            gate, snapshots = credit._project(response({"hasCredits": True,
                "unlimited": False, "balance": balance}), QUOTA, GATE_ERROR)
            self.assertEqual(gate, "credit_balance_present")
            self.assertEqual(snapshots[0]["balance"], expected)
            self.assertEqual(snapshots[0]["windows"][0]["remaining_percent"], 20)
            self.assertEqual(snapshots[0]["windows"][1]["remaining_percent"], 60)
            self.assertNotIn("SECRET", repr(snapshots))
        gate, snapshots = credit._project(response({"hasCredits": False,
            "unlimited": True}), QUOTA, GATE_ERROR)
        self.assertEqual(gate, "quota_floor")
        self.assertTrue(snapshots[0]["unlimited"])
        self.assertEqual(snapshots[0]["balance"], "missing")

    def test_malformed_credit_fields_fail_closed(self):
        for credits in ({}, {"hasCredits": True},
                        {"hasCredits": None, "unlimited": False},
                        {"hasCredits": 1, "unlimited": False},
                        {"hasCredits": False, "unlimited": "false"}):
            with self.subTest(credits=credits), self.assertRaises(ValueError):
                credit._project(response(credits), QUOTA, GATE_ERROR)
        for extra in ({"spendControlReached": "false"},
                      {"rateLimitReachedType": "SECRET"},
                      {"primary": {"usedPercent": float("nan")}}):
            value = response({"hasCredits": False, "unlimited": False})
            value["rateLimitsByLimitId"]["SECRET_ID"].update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                credit._project(value, QUOTA, GATE_ERROR)

    def test_stopped_binding(self):
        credit._bound_access(stopped())
        for change in ({"runtime": "running"}, {"recursive_cleanup_verified": False},
                       {"status": "access_verified"}):
            value = stopped()
            value.update(change)
            with self.assertRaises(ValueError):
                credit._bound_access(value)
        value = stopped()
        value["result"]["turns"] = 1
        with self.assertRaises(ValueError):
            credit._bound_access(value)

    def test_public_report_rejects_extra_or_untyped_fields(self):
        report = {"schema": credit.SCHEMA, "status": "metadata_read",
            "owned_process_cleanup_verified": True, "auth": "chatgpt", "plan": "plus",
            "gate_status": "credit_balance_present", "snapshots": [{"index": 0,
                "has_credits": True, "unlimited": False, "balance": "positive",
                "spend_control_reached": None, "rate_limit_reached_type": None,
                "windows": [{"kind": "primary", "remaining_percent": 20,
                             "window_minutes": 300, "resets_at": 1_800_000_000}]}]}
        self.assertEqual(credit._safe(report), report)
        for key, invalid in (("status", []), ("auth", {}), ("plan", []),
                             ("gate_status", {})):
            bad = {**report, key: invalid}
            self.assertEqual(credit._safe(bad)["status"], "ssh_unverified")
        bad = {**report, "email": "SECRET_EMAIL"}
        self.assertEqual(credit._safe(bad)["status"], "ssh_unverified")
        bad = {**report, "snapshots": [{**report["snapshots"][0], "balance": []}]}
        self.assertEqual(credit._safe(bad)["status"], "ssh_unverified")

    def test_allowlist_and_cleanup(self):
        calls = []
        class Session:
            def rpc(self, method):
                calls.append(method)
                if method == "account/read":
                    return {"account": {"type": "chatgpt", "planType": "plus",
                                        "email": "SECRET_EMAIL"}}
                if method == "account/rateLimits/read":
                    return response({"hasCredits": True, "unlimited": False,
                                     "balance": "1.25"})
                return {}
            def close(self):
                calls.append("close")
                return True
        access = {"inspect": lambda *_: stopped()}
        injected = dict(helper)
        injected["AllowlistedSession"] = lambda *_: Session()
        with patch.dict(injected, {"_pinned_file": lambda *_args, **_kwargs: BASE.read_bytes()}):
            report = credit._inspect(injected, access)
        self.assertEqual(calls, ["initialize", "initialized", "account/read",
                                 "account/rateLimits/read", "close"])
        self.assertEqual(report["status"], "metadata_read")
        self.assertTrue(report["owned_process_cleanup_verified"])
        self.assertNotIn("SECRET", repr(report))
        self.assertEqual(set(helper["RPC_ALLOWLIST"]), {"initialize", "initialized",
            "account/read", "account/rateLimits/read"})
        self.assertIs(helper["RPC_ALLOWLIST"]["account/read"]["refreshToken"], False)


if __name__ == "__main__":
    unittest.main()
