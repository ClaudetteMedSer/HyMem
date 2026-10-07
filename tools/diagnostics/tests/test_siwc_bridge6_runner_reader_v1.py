"""Offline boundary tests for the bridge6 source-bound six-hour SIWC policy."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools.diagnostics import siwc_lme_diagnostic_bundle_v7 as prior_bundle


DIRECTORY = Path(__file__).parents[1]


def load(name: str):
    source = DIRECTORY / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("siwc_lme_diagnostic_v9.py")
reader = load("siwc_lme_diagnostic_progress_v11.py")


class SixHourPolicyTests(unittest.TestCase):
    def test_actual_isolated_source_import_rejects_old_bridge(self) -> None:
        frozen = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle")
        accepted = Path("/private/tmp/hymem-repaired-four-root-dtBv2p/bundle/code")
        if not frozen.is_dir() or not accepted.is_dir():
            self.skipTest("frozen local source unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / ".hymem-siwc-lme-diagnostic-invented01"
            prior_bundle.assemble(
                repo=DIRECTORY.parents[1], accepted_code=accepted,
                candidate=frozen / "candidate", map_path=frozen / "source-map.json",
                output=root)
            old_bridge = root / "code/benchmarks/chatgpt_plan_lme_v5.py"
            old_runner = root / "code/tools/diagnostics/siwc_lme_diagnostic_v8.py"
            old_bridge_bytes = old_bridge.read_bytes()
            old_bridge.unlink()
            old_runner.unlink()
            (root / "code/benchmarks/chatgpt_plan_lme_v6.py").write_bytes(
                (DIRECTORY.parents[1] / "benchmarks/chatgpt_plan_lme_v6.py").read_bytes())
            (root / "code/tools/diagnostics/siwc_lme_diagnostic_v9.py").write_bytes(
                (DIRECTORY / "siwc_lme_diagnostic_v9.py").read_bytes())
            script = r"""
import importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1])
source=root/'code/tools/diagnostics/siwc_lme_diagnostic_v9.py'
spec=importlib.util.spec_from_file_location('isolated_runner9',source)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
assert pathlib.Path(loaded['siwc'].__file__)==root/'code/benchmarks/chatgpt_plan_lme_v6.py'
assert loaded['siwc'].transport.complete is loaded['siwc'].SIWCLMEClient.__init__.__kwdefaults__['response_call']
print('source_only_ok')
"""
            done = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script, str(root)],
                capture_output=True, text=True, timeout=40)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.strip(), "source_only_ok")
            bridge = root / "code/benchmarks/chatgpt_plan_lme_v6.py"
            bridge.write_bytes(old_bridge_bytes)
            rejected = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script, str(root)],
                capture_output=True, text=True, timeout=40)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("transport_or_helper_drift", rejected.stderr)

    def test_only_wall_caps_change_from_accepted_sources(self) -> None:
        old_runner = load("siwc_lme_diagnostic_v8.py")
        old_reader = load("siwc_lme_diagnostic_progress_v10.py")
        self.assertEqual(runner.MAX_LIMITS, {
            "campaign": (8012, 48_160_000, 25_200),
            "question": (2000, 12_000_000, 23_400),
            "canary": (12, 160_000, 600)})
        self.assertEqual(reader.LIMITS, {key: list(value)
                         for key, value in runner.MAX_LIMITS.items()})
        for key in runner.MAX_LIMITS:
            self.assertEqual(runner.MAX_LIMITS[key][:2], old_runner.MAX_LIMITS[key][:2])
        self.assertEqual(runner.MAX_LIMITS["canary"], old_runner.MAX_LIMITS["canary"])
        self.assertEqual(
            {k: v for k, v in runner.SIWC_PINS.items()
             if k != "benchmarks/chatgpt_plan_lme_v6.py"},
            {k: v for k, v in old_runner.SIWC_PINS.items()
             if k != "benchmarks/chatgpt_plan_lme_v5.py"})
        self.assertNotIn("benchmarks/chatgpt_plan_lme_v5.py", runner.SIWC_PINS)
        self.assertIn("benchmarks/chatgpt_plan_lme_v6.py", runner.SIWC_PINS)
        self.assertEqual(runner.PINS, old_runner.PINS)
        self.assertEqual(reader.INVENTORY_SHA256, old_reader.INVENTORY_SHA256)
        self.assertEqual(reader.MAP_SHA256, old_reader.MAP_SHA256)
        self.assertEqual(reader.DATASET_SHA256, old_reader.DATASET_SHA256)
        self.assertEqual(reader.LAUNCHER_NAME, "siwc_lme_diagnostic_launch_v8.py")
        self.assertEqual(reader.LAUNCHER_SHA256, hashlib.sha256(
            (DIRECTORY / reader.LAUNCHER_NAME).read_bytes()).hexdigest())
        self.assertEqual(reader.RUNNER_SHA256, hashlib.sha256(
            (DIRECTORY / runner.RUNNER_RELATIVE.split("/")[-1]).read_bytes()).hexdigest())

    def test_live_entry_passes_exact_deadlines_without_external_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / ".hymem-siwc-lme-diagnostic-test1234"
            root.mkdir()
            loaded = {"questions": [{"question_id": str(i)} for i in range(4)],
                      "warm": SimpleNamespace(BudgetLimits=lambda *args: args)}
            received = {}
            def campaign(_loaded, **kwargs):
                received.update(kwargs)
                return {"run_id": "invented", "selected_denominator": 4,
                        "scored_count": 4, "campaign_stop": None,
                        "diagnostic_complete": True}
            args = ["--root", str(root), "--inventory", str(root / "source-map.json"),
                    "--inventory-sha256", "0" * 64, "--dataset", str(root / "dataset"),
                    "--receipt-sha256", "1" * 64, "--output-dir", str(root / "run"),
                    "--run"]
            with (mock.patch.object(runner, "load_verified", return_value=loaded),
                  mock.patch.object(runner, "verify_launch_receipt",
                      return_value={"expected_cgroup": "invented"}),
                  mock.patch.object(runner, "verify_live_containment"),
                  mock.patch.object(runner, "mark_execution_started"),
                  mock.patch.object(runner, "run_campaign", side_effect=campaign),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(args), 0)
            self.assertEqual(received["indexing_seconds"], 21_600)
            self.assertEqual(received["campaign_limits"], runner.MAX_LIMITS["campaign"])
            self.assertEqual(received["question_limits"], runner.MAX_LIMITS["question"])
            self.assertEqual(received["canary_limits"], runner.MAX_LIMITS["canary"])
            self.assertEqual(received["workers"], 4)

    def test_runner_receipt_accepts_new_and_rejects_old_wall_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / ".hymem-siwc-lme-diagnostic-test1234"
            root.mkdir()
            code_runner = root / "code" / runner.RUNNER_RELATIVE
            code_runner.parent.mkdir(parents=True)
            code_runner.write_bytes((DIRECTORY / "siwc_lme_diagnostic_v9.py").read_bytes())
            questions = [{"question_id": f"q-{i}"} for i in range(4)]
            loaded = {"source_only": False, "root": root,
                      "questions": questions, "dataset": root / "dataset",
                      "prior": SimpleNamespace(SelectedQuestions=lambda *_: questions),
                      "protocol": object(),
                      "siwc": SimpleNamespace(MAX_INVOCATION=300.0)}
            pins = {**runner.PINS, **runner.SIWC_PINS,
                    "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256,
                    runner.RUNNER_RELATIVE: hashlib.sha256(code_runner.read_bytes()).hexdigest()}
            def hashed(path):
                if path == loaded["dataset"]:
                    return runner.DATASET_SHA256
                if path == root / "launch-receipt.json":
                    return hashlib.sha256(path.read_bytes()).hexdigest()
                return pins[path.relative_to(root / "code").as_posix()]
            with (mock.patch.object(runner, "__file__", str(code_runner)),
                  mock.patch.object(runner, "_sha", side_effect=hashed),
                  mock.patch.object(runner, "_regular", return_value=True)):
                receipt = runner.receipt_for(root, loaded)
                self.assertEqual(receipt["indexing_seconds"], 21_600)
                self.assertEqual(receipt["limits"]["campaign"][2], 25_200)
                self.assertEqual(receipt["limits"]["question"][2], 23_400)
                self.assertEqual(receipt["limits"]["canary"], [12, 160_000, 600])
                path = root / "launch-receipt.json"
                path.write_bytes(runner._canonical(receipt))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                (root / "launch-attempt.json").write_bytes(runner._canonical(
                    {"receipt_sha256": digest, "one_shot": True}))
                self.assertEqual(runner.verify_launch_receipt(root, digest, loaded), receipt)
                old = json.loads(path.read_text())
                old["indexing_seconds"] = 10_800
                old["limits"]["campaign"][2] = 14_400
                old["limits"]["question"][2] = 12_600
                path.write_bytes(runner._canonical(old))
                old_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                (root / "launch-attempt.json").write_bytes(runner._canonical(
                    {"receipt_sha256": old_digest, "one_shot": True}))
                with self.assertRaisesRegex(ValueError, "launch_receipt_identity_invalid"):
                    runner.verify_launch_receipt(root, old_digest, loaded)
                old_bridge = json.loads(json.dumps(receipt))
                old_bridge["source_sha256"]["benchmarks/chatgpt_plan_lme_v6.py"] = (
                    "34b51c18a28bd6a09b64a1592cb13edae50716f7694f5a271ec54a27a65010e4")
                path.write_bytes(runner._canonical(old_bridge))
                old_bridge_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                (root / "launch-attempt.json").write_bytes(runner._canonical(
                    {"receipt_sha256": old_bridge_digest, "one_shot": True}))
                with self.assertRaisesRegex(ValueError, "launch_receipt_identity_invalid"):
                    runner.verify_launch_receipt(root, old_bridge_digest, loaded)

    def test_reader_checkpoint_accepts_new_and_rejects_old_wall_policy(self) -> None:
        ids = [f"q-{i}" for i in range(4)]
        receipt = {"selected_row_sha256": ["0" * 64] * 4,
                   "source_sha256": {
                       "benchmarks/chatgpt_plan_responses_v10.py": runner.SIWC_PINS[
                           "benchmarks/chatgpt_plan_responses_v10.py"],
                       "benchmarks/chatgpt_plan_responses_v6.py": runner.SIWC_PINS[
                           "benchmarks/chatgpt_plan_responses_v6.py"],
                       "benchmarks/chatgpt_plan_lme_v6.py": runner.SIWC_PINS[
                           "benchmarks/chatgpt_plan_lme_v6.py"]}}
        manifest = {"schema": reader.RUN_SCHEMA, "mode": reader.MODE,
            "canonical_r9_artifact": False, "official_model_score": False,
            "candidate_map_sha256": reader.MAP_SHA256,
            "dataset_sha256": reader.DATASET_SHA256,
            "selected_row_sha256": receipt["selected_row_sha256"],
            "selected_source_order": "first_n", "expected_count": 4,
            "expected_ids_hash": reader.canonical_hash(ids), "scored_run": True,
            "diagnostic_helper_sha256": reader.HELPER_SHA256,
            "billing_policy": reader.BILLING_POLICY,
            "runner_sha256": reader.RUNNER_SHA256,
            "transport_sha256": receipt["source_sha256"]["benchmarks/chatgpt_plan_responses_v10.py"],
            "transport_v6_sha256": receipt["source_sha256"]["benchmarks/chatgpt_plan_responses_v6.py"],
            "bridge_sha256": receipt["source_sha256"]["benchmarks/chatgpt_plan_lme_v6.py"],
            "grant_identity_sha256": reader.GRANT_IDENTITY_SHA256,
            "rerolls": 0,
            "limits": {key: dict(zip(("turns", "known_tokens", "seconds"), values))
                       for key, values in reader.LIMITS.items()} |
                      {"indexing_seconds": 21_600, "workers": 4}}
        manifest["run_id"] = reader.canonical_hash(manifest)
        entries = {qid: {"status": "failed", "attempts": 1,
                         "row": {"question_id": qid, "benchmark_failure": "indexing_failure:timeout_during_cycle",
                                 "correct": False}, "failure": "indexing_failure:timeout_during_cycle"}
                   for qid in ids}
        checkpoint = {"schema": reader.CHECKPOINT_SCHEMA, "status": "complete",
            "scored": True, "verdict_key": "correct", "manifest": manifest,
            "expected_ids": ids, "entries": entries, "run_id": manifest["run_id"],
            "counts": {"expected": 4, "attempted": 4, "unique_attempted": 4,
                       "total_attempts": 4, "completed": 0, "failed": 4, "missing": 0}}
        self.assertEqual(reader.checkpoint(checkpoint, receipt)[0]["failed"], 4)
        old = json.loads(json.dumps(checkpoint))
        old["manifest"]["limits"]["campaign"]["seconds"] = 14_400
        old["manifest"]["limits"]["question"]["seconds"] = 12_600
        old["manifest"]["limits"]["indexing_seconds"] = 10_800
        old["manifest"]["run_id"] = reader.canonical_hash({
            k: v for k, v in old["manifest"].items() if k != "run_id"})
        old["run_id"] = old["manifest"]["run_id"]
        with self.assertRaisesRegex(ValueError, "checkpoint_limits_invalid"):
            reader.checkpoint(old, receipt)
        old_bridge = json.loads(json.dumps(checkpoint))
        old_bridge["manifest"]["bridge_sha256"] = (
            "34b51c18a28bd6a09b64a1592cb13edae50716f7694f5a271ec54a27a65010e4")
        old_bridge["manifest"]["run_id"] = reader.canonical_hash({
            k: v for k, v in old_bridge["manifest"].items() if k != "run_id"})
        old_bridge["run_id"] = old_bridge["manifest"]["run_id"]
        with self.assertRaisesRegex(ValueError, "checkpoint_identity_invalid"):
            reader.checkpoint(old_bridge, receipt)

    def test_reader_runtime_accepts_new_and_rejects_old_service_wall(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            runtime.mkdir(mode=0o700)
            cgroups = Path(temp) / "cgroups"
            cgroups.mkdir(mode=0o700)
            actual_lstat = Path.lstat
            def local_lstat(path, *args, **kwargs):
                if path == runtime / "bus":
                    return SimpleNamespace(st_mode=stat.S_IFSOCK | 0o600,
                                           st_uid=os.getuid())
                return actual_lstat(path, *args, **kwargs)
            receipt = {"unit": "invented.service",
                       "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/invented.service"}
            values = {"ActiveState": "inactive", "SubState": "dead", "MainPID": "0",
                      "ControlGroup": "", "NRestarts": "0", "Result": "success",
                      "ExecMainStatus": "0", "MemoryMax": "4294967296",
                      "TasksMax": "256", "CPUQuotaPerSecUSec": "2s",
                      "KillMode": "control-group", "Restart": "no",
                      "RemainAfterExit": "yes", "OOMPolicy": "kill",
                      "RuntimeMaxUSec": "25330s", "TimeoutStopUSec": "10s"}
            calls = []
            def service(_args, **_kwargs):
                calls.append(True)
                return SimpleNamespace(stdout="\n".join(
                    f"{key}={value}" for key, value in values.items()))
            with (mock.patch.object(reader.sys, "platform", "linux"),
                  mock.patch.object(reader.os, "getuid", return_value=os.getuid()),
                  mock.patch.object(reader, "ROOT_UID", os.getuid()),
                  mock.patch.object(reader, "USER_RUNTIME", runtime),
                  mock.patch.object(reader, "CGROUP_ROOT", cgroups),
                  mock.patch.object(Path, "lstat", local_lstat),
                  mock.patch.object(reader.subprocess, "run", side_effect=service)):
                outcome = reader.runtime(receipt)
                self.assertTrue(calls)
                self.assertEqual(outcome, ("clean_exit", True))
                values["RuntimeMaxUSec"] = "14530s"
                self.assertEqual(reader.runtime(receipt), ("unverified", False))


if __name__ == "__main__":
    unittest.main()
