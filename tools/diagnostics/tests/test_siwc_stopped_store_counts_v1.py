"""Offline controls for the single stopped-run counts projection."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).parents[1] / "siwc_lme_stopped_store_counts_v1.py"
SPEC = importlib.util.spec_from_file_location("stopped_counts", SOURCE)
assert SPEC and SPEC.loader
counts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(counts)


def stub_reader(*, turns: int = 3838, clean: bool = True,
                current: object = 2, schema: str = "siwc-lme-diagnostic-progress-v9") -> str:
    observations = {}
    for label, ordinary, structured, tokens in (
            ("canary", 8, 3, 46706),
            ("question.0", 704, 244, 3151114),
            ("question.1", 699, 263, 3222237),
            ("question.2", 743, 227, 3259542),
            ("question.3", 720, 227, 3121979)):
        for route, calls in (("ordinary", ordinary), ("structured", structured)):
            observations[label + "." + route] = {"status": "observed", "summary": {
                "usage_complete": True, "timing_saturated": False,
                "first_failure": None, "last_failure_code": None, "failures": 0,
                "calls": calls, "successes": calls, "internal_http_attempts": calls,
                "known_tokens": tokens}}
    return f'''
import json
import os
from pathlib import Path
ROOT_UID = os.getuid()
def inspect(root, receipt):
    return {{'schema':{schema!r},
        'status':'terminal_incomplete_or_unclean',
        'runtime_cleanup_verified':{clean!r},
        'completed_diagnostic_and_clean':False,
        'selected_denominator':4,'scored_count':0,'failed_count':4,
        'correct_count':0,'canary_structural_valid':True,
        'canary_model_gold_match':False,'strict_indexing_healthy_for_all':None,
        'question_failure_codes':['indexing_failure:timeout_during_cycle']*4,
        'known_turns':{turns},'known_tokens':12801578,'usage_complete':True,
        'campaign_stop':'question_failure','budget_stop_code':'question_failure',
        'first_failure':None,'owner_failure':None,'resource_fault':None,
        'resource_observation':{{'denials':0,'current':{current!r},'peak':18,'limit':256}},
        'siwc_observations':{observations!r}}}
def receipt(root, digest):
    return {{}}
def read(path, root, cap):
    if path.stat().st_size > cap:
        raise ValueError('oversize')
    return json.loads(path.read_text())
def checkpoint(value, receipt):
    return ({{'expected':4,'failed':4,'missing':0}},0,0,0,
        {{'expected_ids':['q-0000','q-0001','q-0002','q-0003'],
          'entries':{{f'q-{{i:04d}}':{{'status':'failed',
            'failure':'indexing_failure:timeout_during_cycle'}} for i in range(4)}}}})
'''


def remote(root: Path, *, turns: int = 3838, clean: bool = True,
           current: object = 2, schema: str = "siwc-lme-diagnostic-progress-v9") -> dict:
    values = {name: getattr(counts, name) for name in (
        "SCHEMA", "RECEIPT_SHA", "INDEXING_SCHEMA", "REASONS", "PENDING",
        "MALFORMED", "QUARANTINED", "REPORT_COUNTS", "REPORT_OPTIONAL_COUNTS",
        "REPORT_FLAGS", "MAX_COUNT",
        "MAX_ROWS", "MAX_OUTPUT_BYTES")}
    values["ROOT"] = str(root)
    values["READER_SOURCE"] = stub_reader(turns=turns, clean=clean,
                                           current=current, schema=schema)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(counts.REMOTE, values)
    return json.loads(output.getvalue())


def summary(*, cycles: int = 0, status: dict | None = None) -> dict:
    report = {key: 0 for key in counts.REPORT_COUNTS}
    report.update({key: False for key in counts.REPORT_FLAGS})
    report["chunks_seen"] = 3
    report["chunk_extraction_failures"] = 1
    return {"schema": counts.INDEXING_SCHEMA, "outcome": "failure",
            "complete": False, "healthy": False,
            "failure": {"code": "timeout_during_cycle"}, "max_cycles": 100,
            "timeout_s": 10800, "elapsed_s": 10800.1,
            "cycles": cycles, "reports": [report] * cycles,
            "final_status": status, "cleanup_errors": []}


class StoppedStoreCountsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        run = self.root / "run"
        run.mkdir(mode=0o700)
        (run / "diagnostic-checkpoint.json").write_text("{}")
        self.secret = "PRIVATE-DETAIL-DO-NOT-EMIT"
        for i in range(4):
            question = run / f"q-{i:04d}"
            question.mkdir(mode=0o700)
            (question / "private-indexing.json").write_text(json.dumps(summary(cycles=i)))
            database = question / "hymem.sqlite"
            conn = sqlite3.connect(database)
            try:
                conn.execute("CREATE TABLE chunk_extraction_attempts "
                             "(attempts INTEGER, last_failure_reason TEXT, "
                             "last_failure_details TEXT)")
                conn.execute("INSERT INTO chunk_extraction_attempts VALUES (?,?,?)",
                             (i + 1, "grounding_failure", self.secret))
                conn.commit()
            finally:
                conn.close()

    def test_four_store_projection_and_no_detail_egress(self) -> None:
        result = remote(self.root)
        self.assertIsNotNone(counts._valid(result))
        self.assertEqual(set(result["questions"]), {f"q-{i:04d}" for i in range(4)})
        self.assertEqual(result["questions"]["q-0003"]["retry"]
                         ["by_attempt_and_last_reason"][">3"], {"grounding_failure": 1})
        self.assertEqual(result["questions"]["q-0002"]["indexing"]
                         ["completed_report_totals"]["chunks_seen"], 6)
        self.assertIsNone(result["questions"]["q-0000"]["indexing"]
                          ["last_completed_status"])
        self.assertNotIn(self.secret, json.dumps(result))

    def test_terminal_or_cleanup_mismatch_denies_before_database(self) -> None:
        with self.assertRaisesRegex(ValueError, "terminal_gate_invalid"):
            remote(self.root, turns=3837)
        with self.assertRaisesRegex(ValueError, "terminal_gate_invalid"):
            remote(self.root, clean=False)
        with self.assertRaisesRegex(ValueError, "terminal_gate_invalid"):
            remote(self.root, current=3)
        with self.assertRaisesRegex(ValueError, "terminal_gate_invalid"):
            remote(self.root, current=True)
        with self.assertRaisesRegex(ValueError, "terminal_gate_invalid"):
            remote(self.root, schema="other")

    def test_nonempty_sidecar_and_symlink_denied(self) -> None:
        sidecar = self.root / "run/q-0000/hymem.sqlite-wal"
        sidecar.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "sqlite_sidecar_nonempty"):
            remote(self.root)
        sidecar.unlink()
        private = self.root / "run/q-0000/private-indexing.json"
        real = self.root / "private-real.json"
        private.rename(real)
        private.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "unsafe_path"):
            remote(self.root)

    def test_malformed_indexing_and_invalid_attempts_fail_closed(self) -> None:
        private = self.root / "run/q-0000/private-indexing.json"
        private.write_text(json.dumps(summary(cycles=1) | {"reports": []}))
        with self.assertRaisesRegex(ValueError, "reports_invalid"):
            remote(self.root)
        private.write_text(json.dumps(summary()))
        database = self.root / "run/q-0000/hymem.sqlite"
        conn = sqlite3.connect(database)
        try:
            conn.execute("INSERT INTO chunk_extraction_attempts VALUES (0,'x','x')")
            conn.commit()
        finally:
            conn.close()
        with self.assertRaisesRegex(ValueError, "retry_bucket_invalid"):
            remote(self.root)

    def test_local_whitelist_rejects_extra_fields_and_bad_reconciliation(self) -> None:
        result = remote(self.root)
        result["questions"]["q-0000"]["retry"]["raw_detail"] = self.secret
        self.assertIsNone(counts._valid(result))
        del result["questions"]["q-0000"]["retry"]["raw_detail"]
        result["questions"]["q-0000"]["retry"]["surviving_held_rows"] = 2
        self.assertIsNone(counts._valid(result))

    def test_source_pin_and_no_ssh_on_mismatch(self) -> None:
        with mock.patch.object(counts, "READER_SHA", "0" * 64), \
                mock.patch.object(counts, "_invoke") as invoke:
            self.assertEqual(counts.main(), 1)
            invoke.assert_not_called()

    def test_payload_compiles_with_constant_remote_error(self) -> None:
        payload = counts._payload()
        compile(payload, "<offline-payload>", "exec")
        self.assertIn('"status":"inspection_unavailable"', payload)

    def test_private_summary_changed_during_read_is_denied(self) -> None:
        values = {name: getattr(counts, name) for name in (
            "SCHEMA", "RECEIPT_SHA", "INDEXING_SCHEMA", "REASONS", "PENDING",
            "MALFORMED", "QUARANTINED", "REPORT_COUNTS", "REPORT_OPTIONAL_COUNTS",
            "REPORT_FLAGS", "MAX_COUNT", "MAX_ROWS", "MAX_OUTPUT_BYTES")}
        values["ROOT"] = str(self.root)
        values["READER_SOURCE"] = stub_reader().replace(
            "return json.loads(path.read_text())",
            "value = json.loads(path.read_text())\n"
            "    if path.name == 'private-indexing.json':\n"
            "        path.write_text(path.read_text() + ' ')\n"
            "    return value")
        with self.assertRaisesRegex(ValueError, "private_changed"):
            exec(counts.REMOTE, values)


if __name__ == "__main__":
    unittest.main()
