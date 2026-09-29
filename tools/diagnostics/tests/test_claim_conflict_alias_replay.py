"""Network-free controller verdict and isolation checks."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.diagnostics import claim_conflict_alias_replay_host as host


class FakeHelper:
    RUNTIME = Path("/approved/runtime")
    IMAGE = "sha256:" + "a" * 64


def clean_audit():
    return {"integrity_ok": True, "foreign_key_findings": 0,
            "canonical_drift_findings": 0, "ledger_count_mismatches": 0,
            "same_generation_disagreeing_groups": 0}


def arm(mode, dedup):
    original = "1" * 64
    changed = "2" * 64
    first = ({"status": "rejected", "reason_code": "alias_owned_state_guard",
              "rollback_preserved": True, "failure_captured": True,
              "logical_digest_before": original, "logical_digest_after": original}
             if mode == "baseline" else
             {"status": "persisted", "reason_code": None,
              "logical_digest_before": original, "logical_digest_after": changed})
    repeat = None if mode == "baseline" else {
        "status": "persisted", "reason_code": None,
        "logical_digest_before": changed, "logical_digest_after": changed}
    return {"status": "completed", "dedup_enabled": dedup,
            "first": first, "exact_repeat": repeat,
            "published_before": 0,
            "published_after_first": 0 if mode == "baseline" else 1,
            "published_after_repeat": None if mode == "baseline" else 1,
            "exact_repeat_unchanged": None if mode == "baseline" else True,
            "integrity_before": clean_audit(), "integrity_after": clean_audit()}


class AliasReplayHostTests(unittest.TestCase):
    def test_artifact_digest_matches_capture_worker_commitment(self):
        from tools.diagnostics import claim_conflict_instrumented_dream as dream
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("failure.json", "llm-events.jsonl",
                         "prepersist-001.json", "prepersist-001.sqlite"):
                path = root / name
                path.write_bytes(b"private synthetic evidence")
                path.chmod(0o600)
            (root / "hymem.sqlite").write_bytes(b"ignored live database")
            class Inspector:
                @staticmethod
                def regular(path, *, mode):
                    if path.stat().st_mode & 0o777 != mode:
                        raise RuntimeError("mode drift")
            self.assertEqual(host.artifact_digest(root, Inspector),
                             dream.artifact_digest(root))

    def test_two_networkless_credentialless_containers(self):
        for mode in ("baseline", "fixed"):
            command, mounts = host.configure(FakeHelper, mode)
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertEqual(command[-5:], ["-I", "-B",
                                             "/diag/claim_conflict_instrumented_replay.py",
                                             "--capture-index", "1"])
            self.assertEqual([dst for _, dst, rw in mounts if rw], ["/work"])
            self.assertNotIn("/run/runtime-env.json", [dst for _, dst, _ in mounts])
            self.assertEqual([dst for _, dst, _ in mounts].count("/capture"), 1)

    def test_verdict_requires_alias_rollback_and_fixed_publication_repeat(self):
        receipt = {"capture_sha256": "a" * 64, "snapshot_sha256": "b" * 64}
        for mode in ("baseline", "fixed"):
            metadata = {"capture_sha256": receipt["capture_sha256"],
                        "snapshot_sha256": receipt["snapshot_sha256"],
                        "phase1_sha256": host.PHASE1_SHA,
                        "dedup_on": arm(mode, True),
                        "dedup_off": arm(mode, False)}
            host.verdict(mode, metadata, receipt)
            metadata["dedup_on"]["published_after_first"] = 1 if mode == "baseline" else 0
            with self.assertRaises(RuntimeError):
                host.verdict(mode, metadata, receipt)

    def test_verdict_rejects_missing_digests_and_boolean_capture_index(self):
        receipt = {"capture_sha256": "a" * 64, "snapshot_sha256": "b" * 64}
        metadata = {"capture_sha256": receipt["capture_sha256"],
                    "snapshot_sha256": receipt["snapshot_sha256"],
                    "phase1_sha256": host.PHASE1_SHA,
                    "dedup_on": arm("baseline", True),
                    "dedup_off": arm("baseline", False)}
        metadata["dedup_on"]["first"]["logical_digest_before"] = None
        metadata["dedup_on"]["first"]["logical_digest_after"] = None
        with self.assertRaises(RuntimeError):
            host.verdict("baseline", metadata, receipt)
        with self.assertRaises(RuntimeError):
            host.project({"status": "replayed", "capture_index": True})

    def test_audit_gate_rejects_all_finding_categories(self):
        pristine = arm("fixed", True)
        self.assertTrue(host.clean(pristine))
        for field in clean_audit():
            modified = arm("fixed", True)
            modified["integrity_after"][field] = (
                False if field == "integrity_ok" else 1)
            self.assertFalse(host.clean(modified), field)

    def test_projection_retains_inconclusive_and_safe_setup_failure(self):
        inconclusive = arm("fixed", True)
        inconclusive["status"] = "inconclusive"
        inconclusive["database_sha256"] = "c" * 64
        projected = host.project_arm(inconclusive)
        self.assertEqual(projected["status"], "inconclusive")
        setup = host.project({"status": "error", "reason_code": "replay_setup_failed",
                              "error_type": "ValueError", "failure_captured": True})
        self.assertEqual(setup["status"], "error")
        with self.assertRaises(RuntimeError):
            host.project({"status": "error", "reason_code": "private source text",
                          "error_type": "ValueError", "failure_captured": True})


if __name__ == "__main__":
    unittest.main()
