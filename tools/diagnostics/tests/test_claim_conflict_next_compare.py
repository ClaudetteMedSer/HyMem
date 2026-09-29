"""Network-free checks for the next private offline comparison."""
from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("claim_conflict_next_compare", "claim_conflict_next_compare.py")
replay = load("claim_conflict_next_replay", "claim_conflict_next_replay.py")


class NextCompareTests(unittest.TestCase):
    def test_pins_and_offline_mounts(self):
        self.assertEqual(compare.sha(compare.LOCAL_FIX2), compare.FIX2_SHA)
        self.assertEqual(replay.CAPTURE_HELPER_SHA256, compare.CAPTURE_HELPER_SHA)
        self.assertEqual(replay.SOURCE_SHA256, compare.REFERENCE_SHA)
        host = types.SimpleNamespace(RUNTIME=Path("/frozen/runtime"), IMAGE="sha256:pin")
        for arm in ("fix1", "fix2"):
            command, mounts = compare.configure(host, arm)
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertNotIn("hermes-net", command)
            self.assertFalse(any("runtime-env.json" in value for value in command))
            self.assertIn((str(compare.REFERENCE), "/reference/source.sqlite", False),
                          mounts)
            self.assertIn((str(compare.CAPTURE_HELPER),
                           "/diag/claim_conflict_next_capture.py", False), mounts)

    def test_known_codes_do_not_emit_exception_text(self):
        expected = {
            "duplicate claim citation in validated extraction":
                "duplicate_validated_claim_citation",
            "same prompt generation claim extraction outcomes disagree":
                "same_generation_extraction_outcome_disagreement",
        }
        for message, code in expected.items():
            self.assertEqual(replay.reason(ValueError(message)), code)
        self.assertEqual(replay.reason(ValueError("private source text")),
                         "unclassified_exception")

    def test_only_proven_terminal_arms_are_conclusive(self):
        good = {"dedup_on": {"status": "persisted", "rollback_preserved": None},
                "dedup_off": {"status": "rejected", "rollback_preserved": True}}
        self.assertTrue(compare.conclusive(good))
        good["dedup_off"] = {"status": "rejected", "rollback_preserved": False}
        self.assertFalse(compare.conclusive(good))
        good["dedup_off"] = {"status": "execution_failure", "rollback_preserved": True}
        self.assertFalse(compare.conclusive(good))

    def test_replay_distinguishes_guard_from_execution_failure(self):
        def reject(**_kwargs):
            raise ValueError("duplicate claim citation in validated extraction")

        helper = types.SimpleNamespace(replay=reject, _safe_frames=lambda _exc: [])
        with patch.object(replay, "clone_digest", return_value="a" * 64):
            result = replay.run_arm(helper, dedup_enabled=True,
                                    name="test.sqlite", baseline_digest="a" * 64)
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["rollback_preserved"])

        helper.replay = lambda **_kwargs: (_ for _ in ()).throw(ValueError("private"))
        with patch.object(replay, "clone_digest", return_value="a" * 64):
            result = replay.run_arm(helper, dedup_enabled=True,
                                    name="test.sqlite", baseline_digest="a" * 64)
        self.assertEqual(result["status"], "execution_failure")
        self.assertNotIn("private", repr(result))

        with patch.object(replay, "clone_digest", side_effect=RuntimeError("secondary")):
            result = replay.run_arm(helper, dedup_enabled=True,
                                    name="test.sqlite", baseline_digest="a" * 64)
        self.assertEqual(result["status"], "setup_failure")
        self.assertEqual(result["diagnostic_error_type"], "RuntimeError")
        self.assertNotIn("secondary", repr(result))


if __name__ == "__main__":
    unittest.main()
