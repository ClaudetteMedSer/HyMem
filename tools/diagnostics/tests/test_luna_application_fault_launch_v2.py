"""Offline regression for retained, exited historical user services."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


DIAGNOSTICS = Path(__file__).resolve().parents[1]
UNIT = "hymem-luna-old_A.service"
CGROUP = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + UNIT
EXITED = {"ActiveState": "active", "SubState": "exited", "MainPID": "0", "ControlGroup": ""}


def load(filename):
    path = DIAGNOSTICS / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetainedExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = load("luna_application_fault_launch_v1.py")
        cls.v2 = load("luna_application_fault_launch_v2.py")
        cls.host = load("luna_application_fault_host_v1.py")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cgroup_root = Path(self.temporary.name)
        self.group = self.cgroup_root / CGROUP.lstrip("/")
        self.group.mkdir(parents=True)
        self._empty(self.group)
        self.fake_host = SimpleNamespace(
            BINARY="/pinned/codex", BINARY_SHA256="0" * 64,
            CGROUP_ROOT=self.cgroup_root,
            recursive_empty=self.host.recursive_empty,
            unit_for=lambda root, mode: "hymem-luna-application-fault-v1-containment-new.service")

    @staticmethod
    def _empty(group):
        (group / "cgroup.procs").write_text("")
        (group / "cgroup.threads").write_text("")
        (group / "cgroup.events").write_text("populated 0\n")

    def test_v1_rejects_and_v2_accepts_empty_exited(self):
        original_read_text = Path.read_text

        def read_text(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return "MemAvailable: 8000000 kB\n"
            return original_read_text(path, *args, **kwargs)

        def run(command, **_):
            if "list-units" in command:
                return SimpleNamespace(stdout=f"{UNIT} loaded active exited old\n", returncode=0)
            return SimpleNamespace(stdout="false\n", returncode=0)

        for module, accepted in ((self.v1, False), (self.v2, True)):
            with self.subTest(version=module.SCHEMA):
                with patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=30 * 1024**3)), \
                     patch.object(module.Path, "read_text", read_text), \
                     patch.object(module, "_regular", return_value=True), \
                     patch.object(module, "_sha", return_value="0" * 64), \
                     patch.object(module, "_bus_env", return_value={}), \
                     patch.object(module, "_unit_state", return_value=dict(EXITED)), \
                     patch.object(module.subprocess, "run", side_effect=run):
                    if accepted:
                        module.host_admission(self.fake_host, Path("/home/atta/fresh"))
                    else:
                        with self.assertRaisesRegex(ValueError, "prior_benchmark_unit_running"):
                            module.host_admission(self.fake_host, Path("/home/atta/fresh"))

    def test_exact_group_or_blank_only(self):
        self.assertTrue(self.v2._prior_unit_stopped(self.fake_host, UNIT, EXITED))
        self.assertTrue(self.v2._prior_unit_stopped(self.fake_host, UNIT,
            dict(EXITED, ControlGroup=CGROUP)))
        self.assertFalse(self.v2._prior_unit_stopped(self.fake_host, UNIT,
            dict(EXITED, ControlGroup=CGROUP + "/child")))
        self.assertFalse(self.v2._prior_unit_stopped(self.fake_host, UNIT,
            dict(EXITED, MainPID="123")))
        self.assertFalse(self.v2._prior_unit_stopped(self.fake_host, UNIT,
            dict(EXITED, SubState="running")))

    def test_malformed_names_and_recursive_liveness(self):
        for name in ("hymem-../bad.service", "hymem-bad/child.service", "other.service",
                     "hymem-bad.service;evil", "hymem-bad..service"):
            self.assertFalse(self.v2._prior_unit_stopped(self.fake_host, name, EXITED))
        child = self.group / "child"
        child.mkdir()
        self._empty(child)
        self.assertTrue(self.v2._prior_unit_stopped(self.fake_host, UNIT, EXITED))
        (child / "cgroup.threads").write_text("42\n")
        self.assertFalse(self.v2._prior_unit_stopped(self.fake_host, UNIT, EXITED))
        (child / "cgroup.threads").write_text("")
        (child / "cgroup.events").write_text("populated 1\n")
        self.assertFalse(self.v2._prior_unit_stopped(self.fake_host, UNIT, EXITED))

    def test_symlink_rejected(self):
        link = self.group / "shortcut"
        link.symlink_to(self.cgroup_root, target_is_directory=True)
        self.assertFalse(self.v2._prior_unit_stopped(self.fake_host, UNIT, EXITED))
        link.unlink()
        root_link = self.cgroup_root.parent / (self.cgroup_root.name + "-link")
        root_link.symlink_to(self.cgroup_root, target_is_directory=True)
        self.addCleanup(root_link.unlink)
        self.fake_host.CGROUP_ROOT = root_link
        self.assertFalse(self.v2._prior_unit_stopped(self.fake_host, UNIT, EXITED))


if __name__ == "__main__":
    unittest.main()
