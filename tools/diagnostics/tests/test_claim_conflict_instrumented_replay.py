"""Network-free checks for exact private prepublication replay."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "claim_conflict_instrumented_replay.py"
spec = importlib.util.spec_from_file_location("claim_conflict_instrumented_replay", HELPER)
assert spec is not None and spec.loader is not None
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fixture(root: Path) -> dict:
    from hymem.config import HyMemConfig
    from hymem.extraction.contract import extraction_cache_key

    return {
        "source_sha256": worker.SOURCE_SHA256,
        "chunk": {"id": "chunk", "session_id": "session",
                  "start_message_id": 1, "end_message_id": 2,
                  "salience_reason": "reason", "text": "private chunk text",
                  "source_message_ids": [1, 2]},
        "extraction": {"triples": [{"subject": "private subject", "predicate": "uses",
                                    "object": "private object", "polarity": 1,
                                    "source_message_id": 1}],
                       "markers": [{"kind": "decision", "statement": "private marker"}],
                       "entity_type_hints": {}, "entity_property_hints": {},
                       "failed": False, "failure_reason": None, "failure_details": [],
                       "completion_calls": 1, "provider_attempts": 1,
                       "claim_sources": {"1": {"message_id": 1,
                                               "session_id": "session", "role": "user",
                                               "content": "private source", "chunk_id": "chunk"}},
                       "source_validated": True,
                       "phase1_generation": {"generation_key": worker.GENERATION,
                                             "extraction_cache_key": extraction_cache_key("v1")}},
        "cfg": json.loads(json.dumps(asdict(HyMemConfig(root=root)), default=str)),
        "dedup_vectors": {"private subject uses private object": [0.1, 0.2]},
        "dedup_model": "pinned", "dedup_dim": 2,
        "in_cycle_edges": [{"subject": "private subject", "predicate": "uses",
                            "object": "old object", "vector": [0.1, 0.2],
                            "edge_id": 3, "model": "pinned", "dim": 2,
                            "authoritative": True}],
        "prompt_version": "v1", "database_sha256": "0" * 64,
    }


class InstrumentedReplayTests(unittest.TestCase):
    def test_real_source_valid_publication_uses_derived_namespace(self):
        from hymem import HyMem, HyMemConfig
        from hymem.extraction.contract import extraction_cache_key
        from hymem.extraction.llm import StubLLMClient
        from tests.conftest import make_routed_llm

        with tempfile.TemporaryDirectory() as directory:
            hy = HyMem(HyMemConfig(root=Path(directory)), llm=StubLLMClient(default="[]"))
            try:
                sid = "synthetic"
                hy.open_session(sid)
                hy.log_message(sid, "user", "Our local development uses uv and system Python.")
                hy.close_session(sid)
                hy.set_llm(make_routed_llm([
                    {"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1}
                ], []))
                report = hy.dream()
                self.assertGreaterEqual(report.chunks_processed, 1)
                row = hy.conn.execute(
                    "SELECT chunk_id,phase1_generation_key FROM current_phase1_publications LIMIT 1"
                ).fetchone()
                self.assertIsNotNone(row)
                raw = {"prompt_version": hy.config.prompt_version}
                derived = worker.captured_cache_key(raw)
                self.assertEqual(derived, extraction_cache_key(hy.config.prompt_version))
                self.assertEqual(worker.publication_count(
                    hy.conn, row["chunk_id"], derived, row["phase1_generation_key"]), 1)
                self.assertEqual(worker.publication_count(
                    hy.conn, row["chunk_id"], hy.config.prompt_version,
                    row["phase1_generation_key"]), 0)
            finally:
                hy.close()

    def test_wrong_capture_generation_binding_fails_before_clone(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = fixture(Path(directory))
            raw["extraction"]["phase1_generation"]["extraction_cache_key"] = "wrong"
            with patch.object(worker, "clone") as clone:
                with self.assertRaisesRegex(ValueError, "capture_generation_binding_mismatch"):
                    worker.run_arm(raw, Path(directory) / "unused.sqlite",
                                   dedup_enabled=True, label="wrong")
            clone.assert_not_called()

    def test_unregistered_generation_namespace_fails_before_persist(self):
        from hymem.core.db import connect, initialize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite"
            conn = connect(source)
            initialize(conn)
            conn.close()
            work = root / "work"
            work.mkdir(mode=0o700)
            raw = fixture(root)
            with patch.object(worker, "WORK", work), \
                 patch.object(worker, "attempt") as persist:
                with self.assertRaisesRegex(ValueError,
                                            "generation_cache_key_not_registered_in_snapshot"):
                    worker.run_arm(raw, source, dedup_enabled=True, label="unregistered")
            persist.assert_not_called()

    def test_reconstruct_exact_types_and_counterfactual_only_config(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = fixture(Path(directory))
            with patch.object(worker, "WORK", Path(directory)):
                on = worker.reconstruct(raw, dedup_enabled=True)
                off = worker.reconstruct(raw, dedup_enabled=False)
            chunk, extraction, cfg, vectors, pool = on
            self.assertEqual(chunk.source_message_ids, (1, 2))
            self.assertEqual(extraction.claim_sources[1].content, "private source")
            self.assertEqual(extraction.triples[0].subject, "private subject")
            self.assertEqual((vectors.model, vectors.dim), ("pinned", 2))
            self.assertEqual(pool[0].edge_id, 3)
            self.assertTrue(cfg.triple_dedup_enabled)
            self.assertFalse(off[2].triple_dedup_enabled)
            self.assertEqual(dict(off[3]), dict(vectors))
            self.assertEqual(off[4][0].object, pool[0].object)

    def test_input_index_and_snapshot_hash_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(worker, "CAPTURE", root):
                self.assertEqual(worker.input_paths(1),
                                 (root / "prepersist-001.json", root / "prepersist-001.sqlite"))
                for bad in (0, 33):
                    with self.assertRaisesRegex(ValueError, "capture_index"):
                        worker.input_paths(bad)
                snapshot = root / "prepersist-001.sqlite"
                snapshot.write_bytes(b"private database")
                snapshot.chmod(0o600)
                raw = fixture(root)
                raw["database_sha256"] = "0" * 64
                manifest = root / "prepersist-001.json"
                manifest.write_text(json.dumps(raw))
                manifest.chmod(0o600)
                with self.assertRaisesRegex(ValueError, "snapshot_pin_mismatch"):
                    worker.load_input(1)

    def test_rejection_rolls_back_and_keeps_private_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conn = sqlite3.connect(":memory:", isolation_level=None)
            conn.execute("CREATE TABLE witness (value TEXT)")
            raw = fixture(root)
            def reject(connection, *_args, **_kwargs):
                connection.execute("INSERT INTO witness VALUES ('private')")
                raise ValueError("duplicate claim citation in validated extraction")

            with patch.object(worker, "WORK", root), \
                 patch("hymem.dreaming.phase1.persist_chunk_results", side_effect=reject):
                result = worker.attempt(conn, raw, dedup_enabled=True, label="test")
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["reason_code"], "duplicate_validated_claim_citation")
            self.assertTrue(result["rollback_preserved"])
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM witness").fetchone()[0], 0)
            private = root / "test-failure.json"
            self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("private", json.dumps(result))
            conn.close()

    def test_exact_repeat_has_unchanged_logical_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conn = sqlite3.connect(":memory:", isolation_level=None)
            conn.execute("CREATE TABLE witness (value TEXT UNIQUE)")
            raw = fixture(root)
            def idempotent(connection, *_args, **_kwargs):
                connection.execute("INSERT OR IGNORE INTO witness VALUES ('one')")
                return []

            with patch.object(worker, "WORK", root), \
                 patch("hymem.dreaming.phase1.persist_chunk_results", side_effect=idempotent):
                first = worker.attempt(conn, raw, dedup_enabled=True, label="first")
                second = worker.attempt(conn, raw, dedup_enabled=True, label="repeat")
            self.assertEqual(first["status"], "persisted")
            self.assertNotEqual(first["logical_digest_before"],
                                first["logical_digest_after"])
            self.assertEqual(second["status"], "persisted")
            self.assertEqual(second["logical_digest_before"],
                             second["logical_digest_after"])
            conn.close()

    def test_secondary_digest_failure_keeps_primary_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conn = sqlite3.connect(":memory:", isolation_level=None)
            raw = fixture(root)
            with patch.object(worker, "WORK", root), \
                 patch.object(worker, "logical_digest",
                              side_effect=["a" * 64, OSError("secondary private detail")]), \
                 patch("hymem.dreaming.phase1.persist_chunk_results",
                       side_effect=ValueError("duplicate claim citation in validated extraction")):
                result = worker.attempt(conn, raw, dedup_enabled=True, label="secondary")
            self.assertEqual(result["reason_code"], "duplicate_validated_claim_citation")
            self.assertEqual(result["status"], "execution_failure")
            self.assertEqual(result["diagnostic_error_type"], "OSError")
            self.assertNotIn("secondary private detail", json.dumps(result))
            conn.close()

    def test_pool_before_count_is_not_changed_by_persist_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conn = sqlite3.connect(":memory:", isolation_level=None)
            conn.execute("CREATE TABLE witness (value TEXT)")
            raw = fixture(root)
            def mutate_pool(_connection, *_args, **kwargs):
                kwargs["in_cycle_edges"].append(kwargs["in_cycle_edges"][0])
                return kwargs["in_cycle_edges"]
            with patch.object(worker, "WORK", root), \
                 patch("hymem.dreaming.phase1.persist_chunk_results",
                       side_effect=mutate_pool):
                result = worker.attempt(conn, raw, dedup_enabled=True,
                                        label="mutated-pool")
            self.assertEqual(result["pool_count_before"], 1)
            self.assertEqual(result["pool_count_after"], 2)
            conn.close()

    def test_alias_owned_state_reason_is_static_and_requires_register_alias_frame(self):
        from hymem.dreaming import canonicalize
        with tempfile.TemporaryDirectory() as directory:
            from hymem.core.db import connect, initialize
            conn = connect(Path(directory) / "store.sqlite")
            initialize(conn)
            conn.execute("INSERT INTO entity_aliases(alias,canonical) "
                         "VALUES('other_surface','private_owned_name')")
            try:
                try:
                    canonicalize.register_alias(conn, "private_owned_name", "other")
                except ValueError as exc:
                    code = worker.reason(exc)
                else:
                    self.fail("expected owned-state guard")
                self.assertEqual(code, "alias_owned_state_guard")
                self.assertNotIn("private_owned_name", code)
                self.assertEqual(worker.reason(ValueError(
                    "canonical identity 'private_owned_name' already owns state; "
                    "use merge_canonical/merge")), "unclassified_exception")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
