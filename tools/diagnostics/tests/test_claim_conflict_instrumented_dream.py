"""Network-free checks of all-stage admission and private replay evidence."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "claim_conflict_instrumented_dream.py"
spec = importlib.util.spec_from_file_location("claim_conflict_instrumented_dream", HELPER)
assert spec is not None and spec.loader is not None
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def complete(request):
    return attempt()


def attempt():
    return "provider response"


EMBED_CALLS = []


def embed(texts):
    EMBED_CALLS.append(texts)
    return [[1.0] for _ in texts]


def extraction():
    return None


def persist():
    return None


def make_probe(journal, db_path, *, max_completions=1, max_attempts=2):
    return worker.Probe(journal, db_path,
                        completion_code=complete.__code__,
                        attempt_code=attempt.__code__,
                        embedding_code=embed.__code__,
                        extraction_code=extraction.__code__,
                        persist_code=persist.__code__,
                        max_completions=max_completions,
                        max_attempts=max_attempts)


class InstrumentedDreamTests(unittest.TestCase):
    def test_embedding_payload_cap_blocks_before_exact_transport_code(self):
        class MemoryJournal:
            root = Path("/unused")
            def __init__(self):
                self.events = []
            def append(self, _name, value):
                self.events.append(value)

        for texts, reason in (
            (["x"] * 17, "embedding_payload_limit"),
            (["x" * 128001], "embedding_payload_limit"),
            (["😀" * 128000, "x"], "embedding_payload_limit"),
            (["x", 7], "embedding_payload_invalid"),
            ("not-a-sequence-of-exact-texts", "embedding_payload_invalid"),
            (["\ud800"], "embedding_payload_invalid"),
        ):
            with self.subTest(reason=reason, size=len(texts)):
                EMBED_CALLS.clear()
                journal = MemoryJournal()
                probe = make_probe(journal, Path("/unused/db.sqlite"),
                                   max_completions=4, max_attempts=4)
                sys.setprofile(probe.profile)
                try:
                    with self.assertRaises(worker.BudgetStop):
                        embed(texts)
                finally:
                    sys.setprofile(None)
                self.assertEqual(probe.budget_reason, reason)
                self.assertEqual(probe.attempts, 0)
                self.assertEqual(probe.embedding_attempts, 0)
                self.assertEqual(EMBED_CALLS, [])
                self.assertEqual(journal.events, [])

    def test_embedding_payload_exact_boundaries_admit_and_capture(self):
        EMBED_CALLS.clear()
        with tempfile.TemporaryDirectory() as directory:
            journal = worker.PrivateJournal(Path(directory))
            try:
                probe = make_probe(journal, Path(directory) / "db.sqlite",
                                   max_completions=4, max_attempts=4)
                sys.setprofile(probe.profile)
                try:
                    embed(["x"] * 16)
                    embed(["x" * 128000])
                finally:
                    sys.setprofile(None)
                self.assertEqual(probe.embedding_attempts, 2)
                self.assertEqual(len(EMBED_CALLS), 2)
                events = [json.loads(line) for line in
                          (Path(directory) / "llm-events.jsonl").read_text().splitlines()]
                self.assertEqual(len([e for e in events
                                      if e["event"] == "embedding_request"]), 2)
            finally:
                journal.close()

    def test_embedding_utf8_byte_cap_independent_of_character_cap(self):
        class MemoryJournal:
            root = Path("/unused")
            def append(self, _name, _value):
                raise AssertionError("oversized request was journaled")

        EMBED_CALLS.clear()
        probe = make_probe(MemoryJournal(), Path("/unused/db.sqlite"))
        with patch.object(worker, "MAX_EMBEDDING_UTF8_BYTES", 10):
            sys.setprofile(probe.profile)
            try:
                with self.assertRaises(worker.BudgetStop):
                    embed(["😀😀😀"])
            finally:
                sys.setprofile(None)
        self.assertEqual(probe.budget_reason, "embedding_payload_limit")
        self.assertEqual(probe.attempts, 0)
        self.assertEqual(EMBED_CALLS, [])

    def test_separate_llm_and_embedding_http_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = worker.PrivateJournal(Path(directory))
            try:
                probe = worker.Probe(journal, Path(directory) / "db.sqlite",
                                     completion_code=complete.__code__,
                                     attempt_code=attempt.__code__,
                                     embedding_code=embed.__code__,
                                     extraction_code=extraction.__code__,
                                     persist_code=persist.__code__,
                                     max_completions=4, max_attempts=5,
                                     max_llm_attempts=1,
                                     max_embedding_attempts=2)
                EMBED_CALLS.clear()
                sys.setprofile(probe.profile)
                try:
                    self.assertEqual(complete("synthetic"), "provider response")
                    embed(["one"])
                    embed(["two"])
                    with self.assertRaises(worker.BudgetStop):
                        embed(["third"])
                finally:
                    sys.setprofile(None)
                self.assertEqual(probe.budget_reason,
                                 "embedding_http_attempt_budget")
                self.assertEqual((probe.attempts, probe.llm_attempts,
                                  probe.embedding_attempts), (3, 1, 2))
                self.assertEqual(len(EMBED_CALLS), 2)
            finally:
                journal.close()

    def test_completion_and_shared_http_caps_precede_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = worker.PrivateJournal(root)
            try:
                probe = make_probe(journal, root / "db.sqlite")
                sys.setprofile(probe.profile)
                try:
                    self.assertEqual(complete("private request"), "provider response")
                    self.assertEqual(embed(["private embedding text"]), [[1.0]])
                    with self.assertRaises(worker.BudgetStop):
                        complete("blocked request")
                finally:
                    sys.setprofile(None)
                self.assertEqual((probe.completions, probe.attempts,
                                  probe.llm_attempts, probe.embedding_attempts),
                                 (1, 2, 1, 1))
                self.assertEqual(probe.budget_reason, "completion_budget")
                events = [json.loads(line) for line in
                          (root / "llm-events.jsonl").read_text().splitlines()]
                self.assertEqual(len([event for event in events
                                      if event["event"] == "attempt_begin"]), 1)
                self.assertNotIn("blocked request", repr(events))
            finally:
                journal.close()

    def test_journal_failure_stops_before_transport(self):
        class FailingJournal:
            root = Path("/unused")
            def append(self, _name, _value):
                raise OSError("private journal failed")

        probe = make_probe(FailingJournal(), Path("/unused/db.sqlite"))
        sys.setprofile(probe.profile)
        try:
            with self.assertRaises(worker.InstrumentationStop):
                complete("private")
        finally:
            sys.setprofile(None)
        self.assertTrue(probe.capture_error)
        self.assertEqual(probe.attempts, 0)

    def test_response_capture_failure_stops_after_one_paid_attempt(self):
        class FailingResponseJournal:
            root = Path("/unused")
            def append(self, _name, value):
                if value["event"] == "attempt_response":
                    raise OSError("private response write failed")

        probe = make_probe(FailingResponseJournal(), Path("/unused/db.sqlite"),
                           max_completions=4, max_attempts=4)
        sys.setprofile(probe.profile)
        try:
            with self.assertRaises(worker.InstrumentationStop):
                complete("private")
        finally:
            sys.setprofile(None)
        self.assertEqual(probe.attempts, 1)
        self.assertTrue(probe.capture_error)

    def test_trace_record_failure_preserves_primary_exception(self):
        class FailingJournal:
            root = Path("/unused")
            def append(self, _name, _value):
                raise OSError("secondary write failed")

        probe = make_probe(FailingJournal(), Path("/unused/db.sqlite"))
        primary = ValueError("primary application failure")
        fake_frame = types.SimpleNamespace()
        callback = probe._trace_local(fake_frame, "exception",
                                      (ValueError, primary, None))
        self.assertIs(callback.__self__, probe)
        self.assertTrue(probe.capture_error)
        provider_frame = types.SimpleNamespace(f_code=attempt.__code__, f_locals={})
        with self.assertRaises(worker.InstrumentationStop):
            probe.profile(provider_frame, "call", None)

    def test_controlflow_and_caught_exception_saturation_keep_admission(self):
        class MemoryJournal:
            root = Path("/unused")
            def __init__(self):
                self.events = []
            def append(self, _name, value):
                self.events.append(value)

        journal = MemoryJournal()
        probe = make_probe(journal, Path("/unused/db.sqlite"))
        fake_frame = types.SimpleNamespace()
        for _ in range(300):
            probe._trace_local(fake_frame, "exception",
                               (GeneratorExit, GeneratorExit(), None))
        self.assertEqual(probe.exception_controlflow_skipped, 300)
        self.assertEqual(probe.exception_events, 0)
        self.assertFalse(probe.exception_capture_truncated)
        for _ in range(300):
            probe._trace_local(fake_frame, "exception",
                               (ValueError, ValueError("caught diagnostic"), None))
        self.assertEqual(probe.exception_events, worker.MAX_EXCEPTION_EVENTS)
        self.assertEqual(len(journal.events), worker.MAX_EXCEPTION_EVENTS)
        self.assertTrue(probe.exception_capture_truncated)
        self.assertFalse(probe.capture_error)
        provider_frame = types.SimpleNamespace(f_code=attempt.__code__, f_locals={})
        probe.profile(provider_frame, "call", None)
        self.assertEqual(probe.attempts, 1)

    def test_prepersist_reader_backup_excludes_uncommitted_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "live.sqlite"
            writer = sqlite3.connect(db)
            writer.execute("CREATE TABLE witness (value TEXT)")
            writer.execute("INSERT INTO witness VALUES ('committed')")
            writer.commit()
            journal = worker.PrivateJournal(root)
            try:
                probe = make_probe(journal, db)
                writer.execute("BEGIN IMMEDIATE")
                writer.execute("INSERT INTO witness VALUES ('uncommitted')")
                frame = types.SimpleNamespace(f_locals={
                    "chunk": None, "extraction": None, "dedup_vectors": {},
                    "in_cycle_edges": [], "prompt_version": "v1", "cfg": None,
                })
                probe._capture_prepersist(frame)
                snapshot = sqlite3.connect(root / "prepersist-001.sqlite")
                try:
                    self.assertEqual(snapshot.execute(
                        "SELECT value FROM witness ORDER BY rowid").fetchall(),
                        [("committed",)])
                finally:
                    snapshot.close()
                self.assertEqual(json.loads((root / "prepersist-001.json").read_text())[
                    "in_cycle_edges"], [])
            finally:
                writer.rollback()
                writer.close()
                journal.close()

    def test_public_error_projection_has_no_private_exception_text(self):
        error = ValueError("private source text and key")
        self.assertEqual(worker.safe_type(error), "ValueError")
        self.assertEqual(worker.safe_frames(error), [])
        self.assertNotIn("private", json.dumps({
            "error_type": worker.safe_type(error),
            "candidate_frames": worker.safe_frames(error),
        }))


if __name__ == "__main__":
    unittest.main()
