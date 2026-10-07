"""Offline controls for the timeout host boundary; no network or model calls."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


HOST_FILE = Path(__file__).resolve().parents[1] / "tools/diagnostics/luna_timeout_host_v3.py"
spec = importlib.util.spec_from_file_location("tested_timeout_host", HOST_FILE)
assert spec and spec.loader
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


def write_json(path: Path, value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    path.write_bytes(data)
    path.chmod(0o600)
    return hashlib.sha256(data).hexdigest()


class HostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.root = self.home / ".hymem-luna-timeout-v3-abcdefgh"
        self.root.mkdir(mode=0o700)
        installed = self.root / "timeout-host-v3.py"
        shutil.copyfile(HOST_FILE, installed)
        installed.chmod(0o600)
        self.patches = (mock.patch.object(host, "HOST_HOME", self.home),
                        mock.patch.object(host, "HOST_UID", os.getuid()),
                        mock.patch.object(host, "__file__", str(installed)),
                        mock.patch.object(host.sys, "platform", "linux"))
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def receipt(self, mode: str) -> str:
        value = host.receipt_for(self.root, host._sha(HOST_FILE), mode)
        return write_json(self.root / ("launch-receipt.json" if mode == "probe"
                           else "containment-receipt.json"), value)

    def attempt(self, mode: str, digest: str) -> None:
        write_json(self.root / ("launch-attempt.json" if mode == "probe" else
                                "containment-attempt.json"),
                   {"receipt_sha256": digest, "one_shot": True})

    def test_receipt_has_private_root_and_exact_pins(self) -> None:
        digest = self.receipt("probe")
        receipt = host.verify_receipt(self.root, digest, "probe")
        self.assertEqual(receipt["preparation_sha256"], host.PREPARATION_SHA256)
        self.assertEqual(receipt["probe_sha256"], host.PROBE_SHA256)
        self.assertEqual(receipt["observer_sha256"], host.OBSERVER_SHA256)
        self.assertEqual(receipt["expected_cgroup"].split("/")[-1], receipt["unit"])
        self.assertEqual(receipt["root"], str(self.root))
        self.assertEqual((receipt["turns"], receipt["workers"],
                          receipt["known_tokens"], receipt["campaign_seconds"],
                          receipt["invocation_seconds"]), (16, 1, 160_000, 600, 120))

    def test_receipt_drift_and_wrong_mode_denied(self) -> None:
        digest = self.receipt("probe")
        with self.assertRaises(ValueError):
            host.verify_receipt(self.root, digest, "containment")
        path = self.root / "launch-receipt.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            host.verify_receipt(self.root, digest, "probe")

    def test_root_permission_denied(self) -> None:
        self.root.chmod(0o755)
        with self.assertRaises(ValueError):
            host.unit_for(self.root, "probe")

    def test_tempfile_underscore_suffix_has_matching_v3_unit(self) -> None:
        root = self.home / ".hymem-luna-timeout-v3-abc_defg"
        root.mkdir(mode=0o700)
        self.assertEqual(host.unit_for(root, "probe"),
                         "hymem-luna-timeout-v3-probe-abc_defg.service")

    def test_policy_exact(self) -> None:
        fields = {name: "" for name in host.FIELDS}
        fields.update(NRestarts="0", Type="exec", MemoryMax="4294967296",
            TasksMax="256", CPUQuotaPerSecUSec="2s", KillMode="control-group",
            Restart="no", RemainAfterExit="yes", OOMPolicy="kill",
            RuntimeMaxUSec="12min 10s", TimeoutStopUSec="10s", UMask="0077")
        self.assertTrue(host._policy_ok(fields))
        for key, wrong in (("TasksMax", "255"), ("UMask", "0007"),
                           ("RemainAfterExit", "no"), ("RuntimeMaxUSec", "731s")):
            changed = {**fields, key: wrong}
            self.assertFalse(host._policy_ok(changed), key)

    def test_counter_rejects_duplicate_missing_malformed(self) -> None:
        path = self.root / "events"
        for data in ("max 0\nmax 1\n", "oom 0\n", "max -1\n", "max 0 1\n"):
            path.write_text(data)
            with self.assertRaises((ValueError, KeyError)):
                host._counter(path, "max")

    def test_recursive_nested_cleanup(self) -> None:
        group = self.root / "group"
        nested = group / "child"
        nested.mkdir(parents=True)
        for node in (group, nested):
            (node / "cgroup.procs").write_text("")
            (node / "cgroup.threads").write_text("")
            (node / "cgroup.events").write_text("populated 0\n")
        self.assertTrue(host.recursive_empty(group))
        (nested / "cgroup.procs").write_text("123\n")
        self.assertFalse(host.recursive_empty(group))

    def test_dangling_group_symlink_denied(self) -> None:
        link = self.root / "missing-link"
        link.symlink_to(self.root / "not-there")
        self.assertFalse(host.recursive_empty(link))

    def test_live_wrong_pid_and_group_denied(self) -> None:
        receipt = host.receipt_for(self.root, host._sha(HOST_FILE), "probe")
        values = {name: "" for name in host.FIELDS}
        values.update(ActiveState="active", SubState="running", MainPID="999999",
                      ControlGroup=receipt["expected_cgroup"])
        with mock.patch.object(host, "_unit_values", return_value=values), mock.patch.object(
                host, "_group_policy", return_value=True):
            gate, counters = host.live_attestation(receipt, "initial", None)
        self.assertFalse(gate["containment"])
        self.assertIsNone(counters)
        values["MainPID"] = str(os.getpid())
        values["ControlGroup"] = "/wrong.service"
        with mock.patch.object(host, "_unit_values", return_value=values):
            gate, _ = host.live_attestation(receipt, "initial", None)
        self.assertFalse(gate["containment"])

    def test_containment_only_never_loads_probe_and_is_one_shot(self) -> None:
        digest = self.receipt("containment")
        self.attempt("containment", digest)
        counters = {key: 0 for key in ("pids_denials", "memory_oom", "memory_oom_kill",
            "pids_current", "memory_current", "memory_peak")}
        with mock.patch.object(host, "verify_sources", side_effect=AssertionError("imported")), \
             mock.patch.object(host, "live_attestation", return_value=(
                 {"containment": True, "denials": 0, "oom": 0}, counters)):
            result = host.run_containment_only(self.root, digest)
            self.assertTrue(result["verified"])
            self.assertEqual(result["model_calls"], 0)
            with self.assertRaises(FileExistsError):
                host.run_containment_only(self.root, digest)

    def test_probe_denied_before_source_load(self) -> None:
        digest = self.receipt("probe")
        self.attempt("probe", digest)
        with mock.patch.object(host, "verify_sources", side_effect=AssertionError("imported")), \
             mock.patch.object(host, "live_attestation", return_value=(
                 {"containment": False, "denials": None, "oom": None}, None)):
            result = host.run_once(self.root, digest)
        self.assertEqual(result["failure_code"], "initial_containment_invalid")
        self.assertIsNone(result["probe_result"])
        self.assertEqual(result["status"], "incomplete_or_failed")

    def test_host_result_rejects_raw_or_invalid_sample(self) -> None:
        value = {"schema": host.SCHEMA, "mode": "probe", "status": "incomplete_or_failed",
                 "failure_code": "source_invalid", "probe_result": None,
                 "samples": [], "independent_recursive_cleanup_verified": None}
        self.assertTrue(host.validate_host_result(value))
        self.assertFalse(host.validate_host_result({**value, "raw_error": "secret"}))
        self.assertFalse(host.validate_host_result({**value, "status": "observed_success"}))
        self.assertFalse(host.validate_host_result({**value, "samples": [
            {"phase": "initial", "index": None, "gate": {"containment": True,
             "denials": 0, "oom": 0}, "resources": None}]}))

    def test_verified_normal_rotation_status_is_finite_and_distinct(self) -> None:
        candidate = {"finite": True, "status": "coverage_not_reached"}
        probe = SimpleNamespace(validate_result=lambda result, observer: result == candidate)
        value = {"schema": host.SCHEMA, "mode": "probe", "status": "coverage_not_reached",
                 "failure_code": None, "probe_result": candidate,
                 "samples": [], "independent_recursive_cleanup_verified": None}
        self.assertTrue(host.validate_host_result(value, probe=probe, observer=object()))
        self.assertFalse(host.validate_host_result({**value, "status": "unknown"},
                                              probe=probe, observer=object()))
        self.assertFalse(host.validate_host_result({**value, "raw_log": "secret"},
                                              probe=probe, observer=object()))


if __name__ == "__main__":
    unittest.main()
