"""Network-free one-shot alias-dream host gates and command tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.diagnostics import claim_conflict_alias_dream_host as host


class FakeHelper:
    RUNTIME = Path("/approved/runtime")
    RUNTIME_ENV = Path("/approved/runtime-env.json")
    IMAGE = "sha256:" + "a" * 64


class AliasDreamHostTests(unittest.TestCase):
    def test_offline_and_live_bounds_and_private_mounts(self):
        offline, mounts = host.configure(FakeHelper, "offline")
        live, live_mounts = host.configure(FakeHelper, "live")
        self.assertEqual(mounts, live_mounts)
        self.assertEqual(offline[offline.index("--network") + 1], "none")
        self.assertEqual(live[live.index("--network") + 1], "hermes-net")
        self.assertEqual(live[-13:], [
            "live", "--env", "/run/runtime-env.json",
            "--phase1-sha256", host.PHASE1_SHA,
            "--max-http-attempts", "704",
            "--max-llm-http-attempts", "192",
            "--max-embedding-http-attempts", "512",
            "--deadline-seconds", "1800",
        ])
        self.assertEqual([dst for _, dst, rw in mounts if rw], ["/work"])
        self.assertNotIn(str(host.SHARED / "work/live/hymem.sqlite"),
                         [src for src, _, _ in mounts])

    def test_replay_gate_requires_both_terminal_verdicts(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            (stage / "result.json").write_text(json.dumps({
                "status": "completed", "networked_runs_started": 0,
                "stages": {
                    arm: {"container_id": ("a" if arm == "baseline" else "b") * 64,
                          "status": "exited", "pid": 0, "exit_code": 0,
                          "oom_killed": False, "configuration_verified": True,
                          "metadata": {"arm": arm}}
                    for arm in ("baseline", "fixed")
                },
            }))
            calls = []
            class Alias:
                @staticmethod
                def installed(_shared, _helper):
                    return {"sealed": True,
                            "canonicalize_sha256": host.CANONICALIZE_SHA,
                            "phase1_sha256": host.PHASE1_SHA,
                            "source_files": 480}
                @staticmethod
                def configure(_helper, mode):
                    return [], [(mode, "/candidate", False)]
                @staticmethod
                def inspect(_helper, _cid, _mode, _mounts):
                    return {"status": "exited", "pid": 0, "exit_code": 0,
                            "oom_killed": False}
                @staticmethod
                def verdict(mode, metadata, receipt):
                    calls.append((mode, metadata["arm"], receipt["sealed"]))
            class Helper:
                @staticmethod
                def read_json(path):
                    return json.loads(path.read_text())
            with patch.object(host, "ALIAS_REPLAY", stage):
                receipt_hash = host.replay_gate(object(), Alias, Helper)
            self.assertEqual(len(receipt_hash), 64)
            self.assertEqual(calls, [("baseline", "baseline", True),
                                     ("fixed", "fixed", True)])
            payload = json.loads((stage / "result.json").read_text())
            payload["stages"]["fixed"]["exit_code"] = 1
            (stage / "result.json").write_text(json.dumps(payload))
            with patch.object(host, "ALIAS_REPLAY", stage):
                with self.assertRaises(RuntimeError):
                    host.replay_gate(object(), Alias, Helper)

    def test_reviewed_worker_and_override_local_pins(self):
        local_worker = Path(__file__).resolve().parents[1] / "claim_conflict_instrumented_dream.py"
        self.assertEqual(host.sha(local_worker), host.WORKER_SHA)
        self.assertEqual(host.sha(host.LOCAL_CANONICALIZE),
                         host.CANONICALIZE_SHA)


if __name__ == "__main__":
    unittest.main()
