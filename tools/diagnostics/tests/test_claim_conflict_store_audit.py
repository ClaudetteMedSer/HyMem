"""Network-free private-store audit fixtures using initialized SQLite+vec0."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hymem.core import db
from tools.diagnostics import claim_conflict_store_audit as audit


class StoreAuditTests(unittest.TestCase):
    def fixture(self, root: Path):
        baseline = root / "reference.sqlite"
        source = root / "closed-dream.sqlite"
        receipt = root / "result.json"
        work = root / "work"
        work.mkdir(mode=0o700)
        conn = db.connect(baseline)
        db.initialize(conn)
        db.ensure_vec_table(conn, 384, model="hymem-embedding-producer-v1:" + "a" * 64)
        conn.close()
        baseline.chmod(0o400)
        audit.backup_readonly(baseline, source)
        conn = db.connect(source)
        conn.execute("INSERT INTO dream_runs(started_at,ended_at,chunks_processed) "
                     "VALUES(CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,1)")
        conn.close()
        receipt.write_text(json.dumps({"stages": {"live": {"metadata": {
            "status": "completed", "completion_calls": 1,
            "http_attempts": 1, "llm_http_attempts": 1,
            "embedding_http_attempts": 0, "chunks_processed": 1,
        }}}}))
        receipt.chmod(0o600)
        return baseline, source, receipt, work

    def test_initialized_vec_store_audits_without_source_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, source, receipt, work = self.fixture(Path(directory))
            source_sha = audit.sha(source)
            with patch.multiple(audit, BASELINE=baseline, SOURCE=source,
                                RESULT=receipt, WORK=work,
                                REFERENCE_SHA=audit.sha(baseline)):
                result = audit.audit()
            self.assertEqual(result["status"], "audited")
            self.assertTrue(result["audit_clean"])
            self.assertTrue(result["source_unchanged"])
            self.assertTrue(result["dream_completed"])
            self.assertFalse(result["degraded_counters_present"])
            self.assertEqual(result["dream"]["latest_dream"]["counters"][
                             "chunks_processed"], 1)
            self.assertEqual(audit.sha(source), source_sha)
            self.assertEqual((work / "dream.sqlite").stat().st_mode & 0o777, 0o600)

    def test_foreign_key_violation_fails_integrity_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, source, receipt, work = self.fixture(Path(directory))
            conn = db.connect(source)
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("INSERT INTO entity_mentions(chunk_id,entity_canonical) "
                         "VALUES('missing','synthetic')")
            conn.commit()
            conn.close()
            with patch.multiple(audit, BASELINE=baseline, SOURCE=source,
                                RESULT=receipt, WORK=work,
                                REFERENCE_SHA=audit.sha(baseline)):
                result = audit.audit()
            self.assertEqual(result["status"], "integrity_failed")
            self.assertGreater(result["dream"]["integrity"]["foreign_key_findings"], 0)
            self.assertFalse(result["audit_clean"])

    def test_captured_failure_is_not_called_completed_when_store_is_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, source, receipt, work = self.fixture(Path(directory))
            conn = db.connect(source)
            conn.execute("UPDATE dream_runs SET error='private failure', "
                         "coverage_integrity_failures=1 WHERE id=(SELECT MAX(id) FROM dream_runs)")
            conn.close()
            receipt.write_text(json.dumps({"stages": {"live": {"metadata": {
                "status": "captured_failure", "completion_calls": 2,
                "http_attempts": 3, "llm_http_attempts": 2,
                "embedding_http_attempts": 1,
            }}}}))
            with patch.multiple(audit, BASELINE=baseline, SOURCE=source,
                                RESULT=receipt, WORK=work,
                                REFERENCE_SHA=audit.sha(baseline)):
                result = audit.audit()
            self.assertTrue(result["audit_clean"])
            self.assertFalse(result["dream_completed"])
            self.assertTrue(result["degraded_counters_present"])
            self.assertEqual(result["interpretation"], "not_completed")
            self.assertNotIn("private failure", json.dumps(result))

    def test_health_gate_covers_every_finding(self):
        clean = {"integrity_ok": True, "foreign_key_findings": 0,
                 "canonical_drift_findings": 0, "ledger_count_mismatches": 0,
                 "same_generation_disagreeing_groups": 0}
        self.assertTrue(audit.is_clean(clean))
        for field in clean:
            dirty = dict(clean)
            dirty[field] = False if field == "integrity_ok" else 1
            self.assertFalse(audit.is_clean(dirty), field)


if __name__ == "__main__":
    unittest.main()
