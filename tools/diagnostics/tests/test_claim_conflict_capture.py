"""Local, network-free checks for the private capture helper."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "claim_conflict_capture.py"
spec = importlib.util.spec_from_file_location("claim_conflict_capture", HELPER)
assert spec is not None and spec.loader is not None
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureSafetyTests(unittest.TestCase):
    def test_old_generation_does_not_block_pending_chunk(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE current_phase1_publications "
                         "(chunk_id TEXT, phase1_generation_key TEXT)")
            conn.executemany("INSERT INTO current_phase1_publications VALUES (?,?)", [
                (capture.CHUNK_ID, "older-generation"),
                ("other-chunk", capture.EXPECTED_GENERATION),
            ])
            self.assertEqual(capture._current_publication_count(conn), 0)
            conn.execute("INSERT INTO current_phase1_publications VALUES (?,?)",
                         (capture.CHUNK_ID, capture.EXPECTED_GENERATION))
            self.assertEqual(capture._current_publication_count(conn), 1)
        finally:
            conn.close()

    def test_budget_matches_r7_retry_envelope(self):
        from hymem.extraction.retry import DEFAULT_RETRY_ATTEMPTS

        self.assertEqual(capture.MAX_COMPLETIONS, 16)
        self.assertEqual(capture.MAX_HTTP, 48)
        self.assertEqual(capture.MAX_HTTP,
                         capture.MAX_COMPLETIONS * DEFAULT_RETRY_ATTEMPTS)

    def test_capture_file_is_private_and_env_loader_is_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            private = folder / "capture.json"
            capture._private_write(private, b"private parsed extraction")
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
            self.assertEqual(private.read_bytes(), b"private parsed extraction")
            with self.assertRaises(FileExistsError):
                capture._private_write(private, b"different private content")
            self.assertEqual(private.read_bytes(), b"private parsed extraction")
            env_file = folder / "runtime-env.json"
            env_file.write_text(json.dumps({
                "HYMEM_LLM_API_KEY": "private-test-key",
            }))
            env_file.chmod(0o600)
            with patch.dict(os.environ, {}, clear=True):
                capture._load_runtime_env(env_file)
                self.assertEqual(os.environ["HYMEM_LLM_API_KEY"], "private-test-key")
                self.assertEqual(os.environ["HYMEM_ROOT"], str(capture.WORK))
            with patch.dict(os.environ, {"HYMEM_LLM_EXTRA_BODY": "inherited"}, clear=True):
                capture._load_runtime_env(env_file)
                self.assertNotIn("HYMEM_LLM_EXTRA_BODY", os.environ)
            env_file.write_text(json.dumps({"UNRELATED_SECRET": "rejected"}))
            with self.assertRaises(ValueError):
                capture._load_runtime_env(env_file)


if __name__ == "__main__":
    unittest.main()
