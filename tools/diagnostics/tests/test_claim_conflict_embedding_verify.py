"""Network-free bounds and privacy tests for embedding recovery verifier."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "claim_conflict_embedding_verify.py"
spec = importlib.util.spec_from_file_location("claim_conflict_embedding_verify", HELPER)
assert spec is not None and spec.loader is not None
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

CALLS = []


def transport(texts):
    CALLS.append(tuple(texts))
    return [[1.0] for _ in texts]


class EmbeddingVerifyTests(unittest.TestCase):
    def test_only_embedding_environment_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "runtime-env.json"
            config.write_text(json.dumps({
                "HYMEM_LLM_API_KEY": "private-llm-secret",
                "DEEPSEEK_API_KEY": "private-provider-secret",
                "HYMEM_EMBEDDING_API_KEY": "private-embedding-secret",
                "HYMEM_EMBEDDING_MODEL": "pinned-model",
            }))
            config.chmod(0o600)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "inherited-secret"}, clear=True):
                verify.load_embedding_env(config, root)
                self.assertNotIn("HYMEM_LLM_API_KEY", os.environ)
                self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
                self.assertNotIn("OPENAI_API_KEY", os.environ)
                self.assertEqual(os.environ["HYMEM_EMBEDDING_API_KEY"],
                                 "private-embedding-secret")

    def test_provider_caps_block_before_dispatch(self):
        CALLS.clear()
        admission = verify.Admission(transport.__code__)
        sys.setprofile(admission.profile)
        try:
            with self.assertRaises(verify.BudgetStop):
                transport(["x"] * 65)
        finally:
            sys.setprofile(None)
        self.assertEqual(CALLS, [])

        with patch.object(verify, "MAX_UTF8_BYTES_PER_CALL", 10):
            admission = verify.Admission(transport.__code__)
            sys.setprofile(admission.profile)
            try:
                with self.assertRaises(verify.BudgetStop):
                    transport(["🌍🌍🌍"])
            finally:
                sys.setprofile(None)
            self.assertEqual(admission.http_attempts, 0)
        self.assertEqual(CALLS, [])

        admission = verify.Admission(transport.__code__)
        sys.setprofile(admission.profile)
        try:
            with self.assertRaises(verify.BudgetStop):
                transport(["🌍" * 128001])
        finally:
            sys.setprofile(None)
        self.assertEqual(CALLS, [])
        self.assertEqual(admission.http_attempts, 0)

        admission = verify.Admission(transport.__code__)
        sys.setprofile(admission.profile)
        try:
            with self.assertRaises(verify.BudgetStop):
                transport(["x" * 128001])
        finally:
            sys.setprofile(None)
        self.assertEqual(CALLS, [])

        with patch.object(verify, "MAX_EMBEDDING_HTTP", 2):
            admission = verify.Admission(transport.__code__)
            sys.setprofile(admission.profile)
            try:
                transport(["first"])
                transport(["second"])
                with self.assertRaises(verify.BudgetStop):
                    transport(["third"])
            finally:
                sys.setprofile(None)
        self.assertEqual(len(CALLS), 2)
        self.assertEqual(admission.http_attempts, 2)
        self.assertEqual(admission.reason, "embedding_http_budget")

    def test_pure_census_counts_pending_without_provider_calls(self):
        CALLS.clear()
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE chunks (id TEXT, text TEXT, chunk_kind TEXT)")
        conn.execute("CREATE TABLE chunk_embeddings "
                     "(chunk_id TEXT, vector_json TEXT, model TEXT, dim INTEGER, text_hash TEXT)")
        conn.execute("CREATE TABLE embedding_cache "
                     "(text_hash TEXT, model TEXT, vector_json TEXT, dim INTEGER)")
        conn.executemany("INSERT INTO chunks VALUES (?,?,?)", [
            ("a", "private text one", "extraction"),
            ("b", "private text two", "extraction"),
        ])
        try:
            with patch("hymem.dreaming.embeddings._embedding_identity",
                       return_value=("pinned-model", 384)):
                result = verify.census(conn, object())
            self.assertEqual(result["pending_chunks"], 2)
            self.assertEqual(result["remote_miss_texts"], 2)
            self.assertEqual(result["planned_http_calls"], 1)
            self.assertEqual(CALLS, [])
            self.assertNotIn("private text", json.dumps(result))
        finally:
            conn.close()

    def test_oversized_legacy_chunk_only_blocks_uncached_remote_miss(self):
        from hymem.extraction.embeddings import embedding_text_hash
        from hymem.core.vectors import encode_vector
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE chunks (id TEXT, text TEXT, chunk_kind TEXT)")
        conn.execute("CREATE TABLE chunk_embeddings "
                     "(chunk_id TEXT, vector_json TEXT, model TEXT, dim INTEGER, text_hash TEXT)")
        conn.execute("CREATE TABLE embedding_cache "
                     "(text_hash TEXT, model TEXT, vector_json TEXT, dim INTEGER)")
        oversized = "x" * 128001
        conn.execute("INSERT INTO chunks VALUES (?,?,?)",
                     ("legacy", oversized, "extraction"))
        try:
            with patch("hymem.dreaming.embeddings._embedding_identity",
                       return_value=("pinned-model", 384)):
                missing = verify.census(conn, object())
                self.assertEqual(missing["pending_chunks"], 1)
                self.assertEqual(missing["planned_http_calls"], 0)
                self.assertEqual(missing["uncallable_remote_batches"], 1)
                conn.execute("INSERT INTO embedding_cache VALUES (?,?,?,?)",
                             (embedding_text_hash(oversized), "pinned-model",
                              encode_vector([1.0] + [0.0] * 383), 384))
                cached = verify.census(conn, object())
                self.assertEqual(cached["pending_chunks"], 1)
                self.assertEqual(cached["cached_pending_chunks"], 1)
                self.assertEqual(cached["uncallable_remote_batches"], 0)
                conn.execute("INSERT INTO chunk_embeddings VALUES (?,?,?,?,?)",
                             ("legacy", encode_vector([1.0] + [0.0] * 383), "pinned-model",
                              384, embedding_text_hash(oversized)))
                current = verify.census(conn, object())
                self.assertEqual(current["pending_chunks"], 0)
                self.assertEqual(current["uncallable_remote_batches"], 0)
        finally:
            conn.close()

    def test_nonembedding_audit_on_initialized_store(self):
        from hymem.core.db import connect, initialize
        with tempfile.TemporaryDirectory() as directory:
            conn = connect(Path(directory) / "private.sqlite")
            try:
                initialize(conn)
                before = verify.graph_audit(conn)
                self.assertTrue(before["integrity_ok"])
                self.assertEqual(before["foreign_key_findings"], 0)
                self.assertEqual(before["canonical_drift_findings"], 0)
                self.assertEqual(before["ledger_count_mismatches"], 0)
                self.assertEqual(before, verify.graph_audit(conn))
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
