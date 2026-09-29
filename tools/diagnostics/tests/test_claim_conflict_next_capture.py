"""Network-free checks for the next private chunk capture controls."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


capture = load("claim_conflict_next_capture", "claim_conflict_next_capture.py")
host = load("claim_conflict_next_host", "claim_conflict_next_host.py")


class NextCaptureTests(unittest.TestCase):
    def test_target_chunk_contract(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE chunks (id TEXT, session_id TEXT, "
                     "start_message_id INTEGER, end_message_id INTEGER, "
                     "salience_reason TEXT, text TEXT, source_manifest_version TEXT, "
                     "source_manifest_count INTEGER, chunk_kind TEXT)")
        conn.execute("CREATE TABLE chunk_message_sources "
                     "(chunk_id TEXT, source_message_id INTEGER, ordinal INTEGER)")
        try:
            conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)", (
                capture.CHUNK_ID, "session", 1016, 1017, "reason", "x" * 1298,
                "claim-source-manifest-v1", 2, "extraction"))
            conn.executemany("INSERT INTO chunk_message_sources VALUES (?,?,?)", [
                (capture.CHUNK_ID, 1016, 0), (capture.CHUNK_ID, 1017, 1)])
            chunk = capture._chunk(conn)
            self.assertEqual(chunk.source_message_ids, (1016, 1017))
            conn.execute("UPDATE chunks SET text=?", ("x" * 1299,))
            with self.assertRaisesRegex(ValueError, "target source manifest"):
                capture._chunk(conn)
        finally:
            conn.close()

    def test_budget_and_snapshot_pin(self):
        from hymem.extraction.retry import DEFAULT_RETRY_ATTEMPTS

        self.assertEqual(capture.MAX_COMPLETIONS, 16)
        self.assertEqual(capture.MAX_HTTP, 48)
        self.assertEqual(capture.MAX_COMPLETIONS * DEFAULT_RETRY_ATTEMPTS, 48)
        self.assertEqual(capture.EXPECTED_SOURCE_SHA256,
                         host.EXPECTED_REFERENCE_SHA256)
        self.assertEqual(host.REFERENCE, host.STAGE / "reference.sqlite")

    def test_host_only_builds_offline_and_live_containers(self):
        for mode in ("offline", "live"):
            command, mounts = host.container_command(mode)
            self.assertIn("/diag/claim_conflict_next_capture.py", command)
            self.assertIn("/run/runtime-env.json", command)
            self.assertIn((str(host.REFERENCE), "/reference/source.sqlite", False),
                          mounts)
        with self.assertRaisesRegex(RuntimeError, "invalid_mode"):
            host.container_command("replay")

    def test_exception_frames_exclude_private_values_and_paths(self):
        def frame(filename, name, lineno):
            return (types.SimpleNamespace(f_code=types.SimpleNamespace(
                co_filename=filename, co_name=name)), lineno)
        fake = [frame("/candidate/hymem/dreaming/phase1.py", "persist_chunk_results", 771),
                frame("/reference/source.sqlite", "secret", 9),
                frame("/candidate/hymem/extraction/chunk.py", "extract_chunk", 42),
                frame("/candidate/hymem/../private.py", "secret", 42)]
        with patch.object(capture.traceback, "walk_tb", return_value=iter(fake)):
            frames = capture._safe_frames(ValueError("private exception string"))
        self.assertEqual(frames, [
            {"path": "hymem/dreaming/phase1.py",
             "function": "persist_chunk_results", "line": 771},
            {"path": "hymem/extraction/chunk.py",
             "function": "extract_chunk", "line": 42},
        ])
        self.assertNotIn("secret", repr(frames))

    def test_top_level_module_frame_is_accepted_by_worker_and_host(self):
        namespace = {}
        exec(compile("def failure():\n    raise ValueError('private payload')\n",
                     "/candidate/hymem/bootstrap.py", "exec"), namespace)
        try:
            namespace["failure"]()
        except ValueError as exc:
            frames = capture._safe_frames(exc)
        self.assertEqual(frames, [{"path": "hymem/bootstrap.py", "function": "failure", "line": 2}])
        self.assertIsNotNone(host.FRAME_PATH.fullmatch(frames[0]["path"]))

    def test_frame_output_is_bounded(self):
        frame = types.SimpleNamespace(f_code=types.SimpleNamespace(
            co_filename="/candidate/hymem/dreaming/phase1.py", co_name="persist_chunk_results"))
        with patch.object(capture.traceback, "walk_tb", return_value=iter([(frame, 2)] * 1000)):
            frames = capture._safe_frames(ValueError("private"))
        self.assertEqual(len(frames), 12)


if __name__ == "__main__":
    unittest.main()
