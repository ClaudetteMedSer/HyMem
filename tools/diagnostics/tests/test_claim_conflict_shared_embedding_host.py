"""Network-free checks for sealed shared-batch diagnostic controller."""
from __future__ import annotations

from pathlib import Path
import json
import unittest

from tools.diagnostics import claim_conflict_shared_embedding_host as host


class FakeHelper:
    RUNTIME = Path("/approved/runtime")
    RUNTIME_ENV = Path("/approved/runtime-env.json")
    IMAGE = "sha256:" + "a" * 64


class SharedEmbeddingHostTests(unittest.TestCase):
    def test_allowlisted_override_hashes_match_reviewed_freeze(self):
        host.reviewed_override_pins()
        self.assertEqual(len(host.OVERRIDE_SHAS), 8)
        for relative, expected in host.OVERRIDE_SHAS.items():
            with self.subTest(relative=relative):
                self.assertEqual(host.sha(host.LOCAL_FROZEN / relative), expected)

    def test_exact_offline_and_live_container_arguments(self):
        phase1_sha = host.OVERRIDE_SHAS["hymem/dreaming/phase1.py"]
        offline, mounts = host.configure(FakeHelper, "offline", phase1_sha)
        live, live_mounts = host.configure(FakeHelper, "live", phase1_sha)
        self.assertEqual(mounts, live_mounts)
        self.assertEqual(offline[offline.index("--network") + 1], "none")
        self.assertEqual(live[live.index("--network") + 1], "hermes-net")
        self.assertEqual(live[-13:], [
            "live", "--env", "/run/runtime-env.json",
            "--phase1-sha256", phase1_sha,
            "--max-http-attempts", "704",
            "--max-llm-http-attempts", "192",
            "--max-embedding-http-attempts", "512",
            "--deadline-seconds", "1800",
        ])
        self.assertEqual({dst for _, dst, _ in mounts},
                         {"/candidate", "/diag/claim_conflict_instrumented_dream.py",
                          "/reference/source.sqlite", "/work",
                          "/home/node/hymem-env", "/run/runtime-env.json"})
        self.assertEqual([dst for _, dst, rw in mounts if rw], ["/work"])

    def test_metadata_projection_rejects_private_codes_and_overbudget(self):
        for error_type in ("AssertionError", "TimeoutError", "ConnectionError"):
            self.assertEqual(host.project({"status": "captured_failure",
                                           "error_type": error_type})["error_type"],
                             error_type)
        for value in (
            {"status": "error", "budget_reason": "private_source_text"},
            {"status": "ready", "embedding_http_attempts": 513},
            {"status": "error", "candidate_frames": [
                {"path": "/private/source.py", "function": "f", "line": 1}]},
        ):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    host.project(value)

    def test_inspector_checks_full_command_not_only_six_arg_suffix(self):
        phase1_sha = host.OVERRIDE_SHAS["hymem/dreaming/phase1.py"]
        command, mounts = host.configure(FakeHelper, "live", phase1_sha)
        cmd = command[command.index(FakeHelper.IMAGE) + 1:]
        item = {
            "Image": FakeHelper.IMAGE,
            "Config": {"Image": FakeHelper.IMAGE, "User": "1000:1000",
                       "Entrypoint": ["/home/node/hymem-env/bin/python3"],
                       "WorkingDir": "/candidate", "Cmd": cmd,
                       "Env": ["HOME=/tmp", "TMPDIR=/tmp"]},
            "HostConfig": {"NetworkMode": "hermes-net", "ReadonlyRootfs": True,
                           "Privileged": False, "CapDrop": ["ALL"],
                           "SecurityOpt": ["no-new-privileges"], "Init": True,
                           "Memory": 2147483648, "NanoCpus": 2000000000,
                           "PidsLimit": 128,
                           "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
                           "RestartPolicy": {"Name": "no"}},
            "Mounts": [{"Destination": dst, "Source": src, "RW": rw,
                        "Type": "bind"} for src, dst, rw in mounts],
            "State": {"Status": "created", "ExitCode": 0,
                      "OOMKilled": False, "Pid": 0},
        }
        class InspectionHelper(FakeHelper):
            @staticmethod
            def run(_command, _timeout, _code):
                return json.dumps([item]).encode()
        cid = "c" * 64
        self.assertEqual(host.inspect(InspectionHelper, cid, "live", mounts,
                                      phase1_sha)["status"], "created")
        item["Config"]["Cmd"] = cmd[:-2] + ["--deadline-seconds", "900"]
        with self.assertRaises(RuntimeError):
            host.inspect(InspectionHelper, cid, "live", mounts, phase1_sha)


if __name__ == "__main__":
    unittest.main()
