"""Offline controls for the versioned host boundary; no provider or host calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


DIAGNOSTICS = Path(__file__).resolve().parents[1]


def load(name: str):
    path = DIAGNOSTICS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HostBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = load("luna_application_fault_host_v1.py")
        cls.launch = load("luna_application_fault_launch_v1.py")
        cls.progress = load("luna_application_fault_progress_v1.py")
        cls.install = load("luna_application_fault_install_v1.py")

    def test_helper_pin_closure(self):
        digest = self.host._sha(DIAGNOSTICS / "luna_application_fault_host_v1.py")
        self.assertEqual(self.launch.HOST_SHA256, digest)
        self.assertEqual(self.progress.HOST_SHA256, digest)
        self.assertEqual(self.host.RUNNER_SHA, self.install.RUNNER_SHA)
        self.assertEqual(self.host.CAPTURE_SHA, self.install.CAPTURE_SHA)
        self.assertEqual(self.host.PROBE_SHA, self.install.PROBE_SHA)

    def test_policy_and_limits_are_exact(self):
        self.assertEqual(self.host.LIMITS["turns"], 12)
        self.assertEqual(self.host.LIMITS["known_tokens"], 160_000)
        self.assertEqual(self.host.LIMITS["workers"], 1)
        self.assertEqual(self.host.LIMITS["index_seconds"], 540)
        self.assertEqual(self.host.POLICY["memory_max"], 4_294_967_296)
        self.assertEqual(self.host.POLICY["tasks_max"], 256)
        self.assertEqual(self.host.POLICY["runtime_seconds"], 730)

    def test_counter_type_and_bounds(self):
        valid = {"pids_denials": 0, "memory_oom": 0, "memory_oom_kill": 0,
                 "pids_current": 1, "memory_current": 1, "memory_peak": 2}
        self.assertTrue(self.host._valid_counters(valid))
        for key in valid:
            bad = dict(valid, **{key: True})
            self.assertFalse(self.host._valid_counters(bad))
        self.assertFalse(self.host._valid_counters(dict(valid, pids_current=257)))
        self.assertFalse(self.host._valid_counters(dict(valid, memory_peak=0)))

    def test_recursive_cleanup_rejects_live_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "child"
            child.mkdir()
            for path in (root, child):
                (path / "cgroup.procs").write_text("")
                (path / "cgroup.threads").write_text("")
                (path / "cgroup.events").write_text("populated 0\n")
            self.assertTrue(self.host.recursive_empty(root))
            (child / "cgroup.procs").write_text("123\n")
            self.assertFalse(self.host.recursive_empty(root))
            (child / "cgroup.procs").write_text("")
            (child / "cgroup.events").write_text("populated 1\n")
            self.assertFalse(self.host.recursive_empty(root))

    def test_one_shot_command_policy(self):
        root = Path("/home/atta/.hymem-luna-application-fault-v1-abcdefgh")
        receipt = {"unit": "hymem-luna-application-fault-v1-probe-abcdefgh.service"}
        command = self.launch.command(root, receipt, "0" * 64, "probe")
        self.assertIn("--property=Restart=no", command)
        self.assertIn("--property=KillMode=control-group", command)
        self.assertIn("--property=OOMPolicy=kill", command)
        self.assertIn("--property=TasksMax=256", command)
        self.assertIn("--property=MemoryMax=4294967296", command)
        self.assertIn("-i", command)
        self.assertIn("--run-once", command)
        self.assertNotIn("--containment-only", command)


if __name__ == "__main__":
    unittest.main()
