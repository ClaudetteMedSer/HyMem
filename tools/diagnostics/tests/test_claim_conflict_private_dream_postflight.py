"""Synthetic, network-free checks of private dream postflight gates."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

from tools.diagnostics import claim_conflict_private_dream_postflight as postflight
from hymem.dreaming.runner import DreamReport
from dataclasses import asdict


DEGRADED = ("coverage_integrity_failures", "digest_quarantined")


def clean_case():
    return {
        "audit_clean": True, "source_unchanged": True,
        "baseline_sha256": "a" * 64, "source_sha256": "b" * 64,
        "latest_dream_advanced": True, "dream_completed": True,
        "campaign": {"worker_status": "completed"},
        "baseline": {"counts": {}, "integrity": {"integrity_ok": True}},
        "dream": {"counts": {}, "integrity": {"integrity_ok": True}, "latest_dream": {"counters": {
            "coverage_integrity_failures": 0, "digest_quarantined": 0}}},
    }


class PrivateDreamPostflightTests(unittest.TestCase):
    def setUp(self):
        self.summary = clean_case()
        self.before = {"target_current": 0, "target_pinned": 0,
                       "target_pinned_with_proof": 0}
        self.after = {"target_current": 1, "target_pinned": 1,
                      "target_pinned_with_proof": 1}
        self.before_q = {"retry_limit": set(), "terminal_loss": set()}
        self.after_q = {"retry_limit": set(), "terminal_loss": set()}
        self.worker = {key: value for key, value in asdict(DreamReport()).items()
                       if not isinstance(value, str)}
        self.worker["chunks_processed"] = 1
        self.stored_match = True
        self.aggregation_blocking = False

    def assess(self):
        return postflight.assess(self.summary, 63, 64, self.before,
                                 self.after, self.before_q, self.after_q,
                                 DEGRADED, self.worker, self.stored_match,
                                 self.aggregation_blocking)

    def test_exact_completed_publication_passes(self):
        self.assertEqual(self.assess()["status"], "pass")

    def test_zero_worker_completions_cannot_pass_publication(self):
        self.worker["chunks_processed"] = 0
        self.assertEqual(self.assess()["status"], "fail")

    def test_wrong_schema_and_outcomes_fail(self):
        for before, after in ((64, 64), (63, 63), (63, 65)):
            report = postflight.assess(self.summary, before, after, self.before,
                                       self.after, self.before_q, self.after_q,
                                       DEGRADED, self.worker, self.stored_match,
                                       self.aggregation_blocking)
            self.assertEqual(report["status"], "fail")
        for status in ("captured_failure", "budget_stopped", "error"):
            self.summary["campaign"]["worker_status"] = status
            self.assertEqual(self.assess()["status"], "fail")
        self.summary["campaign"]["worker_status"] = "completed"
        self.summary["dream_completed"] = False
        self.assertEqual(self.assess()["status"], "fail")

    def test_missing_stale_or_unproved_target_fails(self):
        for field in self.after:
            case = copy.deepcopy(self.after)
            case[field] = 0
            self.after = case
            self.assertEqual(self.assess()["status"], "fail", field)
            self.after = {"target_current": 1, "target_pinned": 1,
                          "target_pinned_with_proof": 1}
        self.before["target_current"] = 1
        self.assertEqual(self.assess()["status"], "fail")

    def test_bad_integrity_source_mutation_and_new_failure_counter_fail(self):
        for field in ("audit_clean", "source_unchanged"):
            self.summary[field] = False
            self.assertEqual(self.assess()["status"], "fail")
            self.summary[field] = True
        self.summary["dream"]["latest_dream"]["counters"]["digest_quarantined"] = 1
        self.assertEqual(self.assess()["status"], "fail")

    def test_same_count_different_chunk_is_new_quarantine(self):
        self.before_q["retry_limit"] = {"private-old-chunk-id"}
        self.after_q["retry_limit"] = {"private-new-chunk-id"}
        report = self.assess()
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["quarantines"]["retry_limit"], {
            "baseline": 1, "dream": 1, "introduced": 1})
        self.assertNotIn("private-old-chunk-id", str(report))
        self.assertNotIn("private-new-chunk-id", str(report))
        self.after_q["retry_limit"] = {"private-old-chunk-id"}
        self.assertEqual(self.assess()["status"], "pass")
        self.assertTrue(self.assess()["baseline_quarantines_present"])

    def test_unknown_publication_or_quarantine_fields_rejected(self):
        self.after["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "publication_fields_invalid"):
            self.assess()
        del self.after["unexpected"]
        self.after_q["unexpected"] = set()
        with self.assertRaisesRegex(ValueError, "quarantine_fields_invalid"):
            self.assess()

    def test_unpersisted_extraction_failure_and_budget_gate_fail(self):
        self.worker["chunk_extraction_failures"] = 1
        self.assertEqual(self.assess()["status"], "fail")
        self.worker["chunk_extraction_failures"] = 0
        self.worker["budget_exhausted"] = True
        self.assertEqual(self.assess()["status"], "fail")

    def test_missing_unknown_and_malformed_worker_fields_rejected(self):
        del self.worker["chunk_extraction_failures"]
        with self.assertRaisesRegex(ValueError, "worker_report_fields_invalid"):
            self.assess()
        self.worker["chunk_extraction_failures"] = 0
        self.worker["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "worker_report_fields_invalid"):
            self.assess()
        del self.worker["unexpected"]
        for bad in (-1, True, "1", 1.0):
            self.worker["chunk_extraction_failures"] = bad
            with self.assertRaisesRegex(ValueError, "worker_report_value_invalid"):
                self.assess()
        self.worker["chunk_extraction_failures"] = 0

    def test_aggregation_blocking_and_inconsistent_stored_counters_fail(self):
        self.aggregation_blocking = True
        self.assertEqual(self.assess()["status"], "fail")
        self.aggregation_blocking = False
        self.stored_match = False
        self.assertEqual(self.assess()["status"], "fail")

    def test_sql_counts_only_pinned_current_proof(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE current_phase1_publications ("
                     "chunk_id TEXT, phase1_generation_key TEXT, prompt_version TEXT)")
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes ("
                     "chunk_id TEXT, phase1_generation_key TEXT, "
                     "prompt_version TEXT, local_replay_proof TEXT)")
        conn.execute("INSERT INTO current_phase1_publications VALUES(?,?,?)",
                     (postflight.TARGET_CHUNK, postflight.TARGET_GENERATION, "v"))
        conn.execute("INSERT INTO kg_claim_extraction_outcomes VALUES(?,?,?,?)",
                     (postflight.TARGET_CHUNK, postflight.TARGET_GENERATION, "v", None))
        self.assertEqual(postflight.publication_counts(conn, has_proof_column=True)[
            "target_pinned_with_proof"], 0)
        conn.execute("UPDATE kg_claim_extraction_outcomes SET local_replay_proof='private'")
        self.assertEqual(postflight.publication_counts(conn, has_proof_column=True)[
            "target_pinned_with_proof"], 1)
        conn.close()

    def test_worker_result_identity_and_persisted_counter_cross_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker-result.json"
            db = sqlite3.connect(":memory:")
            db.row_factory = sqlite3.Row
            db.execute("CREATE TABLE dream_runs (id INTEGER, chunks_processed INTEGER, "
                       "aggregation_blocking TEXT)")
            db.execute("INSERT INTO dream_runs VALUES (7, 1, '')")
            self.summary["dream"]["latest_dream"]["id"] = 7
            self.summary["campaign"]["chunks_processed"] = 1
            worker = {"status": "completed", "source_sha256": "pin",
                      "phase1_sha256": postflight.PHASE1_SHA256,
                      "target_chunk_id": postflight.TARGET_CHUNK,
                      "generation_key": postflight.TARGET_GENERATION,
                      "runtime_generation_verified": True,
                      "chunks_processed": 1, "report": self.worker}
            audit = Mock(REFERENCE_SHA="pin")
            audit.checked_file.side_effect = lambda p, mode: self.assertEqual(mode, 0o600)

            def read():
                path.write_text(json.dumps(worker))
                return postflight.worker_evidence(path, audit, db, self.summary)

            _, consistent, blocking = read()
            self.assertTrue(consistent)
            self.assertFalse(blocking)
            self.summary["campaign"]["chunks_processed"] = 0
            self.assertFalse(read()[1])
            self.summary["campaign"]["chunks_processed"] = 1
            db.execute("UPDATE dream_runs SET chunks_processed=0")
            self.assertFalse(read()[1])
            db.execute("UPDATE dream_runs SET chunks_processed=1, aggregation_blocking='private'")
            self.assertTrue(read()[2])
            del worker["phase1_sha256"]
            with self.assertRaisesRegex(ValueError, "worker_result_identity_invalid"):
                read()
            worker["phase1_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "worker_result_identity_invalid"):
                read()
            worker["phase1_sha256"] = postflight.PHASE1_SHA256
            worker["generation_key"] = "wrong"
            with self.assertRaisesRegex(ValueError, "worker_result_identity_invalid"):
                read()
            db.close()


if __name__ == "__main__":
    unittest.main()
