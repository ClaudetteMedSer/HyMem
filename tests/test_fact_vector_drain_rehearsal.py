"""Read-only source and private-copy guards for fact-vector drain rehearsal."""

import hashlib
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.diagnostics import fact_vector_drain_rehearsal as rehearsal


class FactVectorDrainRehearsalTests(unittest.TestCase):
    def test_private_parent_requires_runner_uid_and_gid(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            rehearsal._private_parent(parent)
            with patch.object(rehearsal.os, "geteuid", return_value=os.geteuid() + 1):
                with self.assertRaisesRegex(
                    rehearsal.drain.replay.RepairStopped, "owned by the runner",
                ):
                    rehearsal._private_parent(parent)
            with patch.object(rehearsal.os, "getegid", return_value=os.getegid() + 1):
                with self.assertRaisesRegex(
                    rehearsal.drain.replay.RepairStopped, "owned by the runner",
                ):
                    rehearsal._private_parent(parent)

    def test_clone_uses_read_snapshot_and_keeps_source_unchanged(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.sqlite"
            clone_path = root / "clone.sqlite"
            conn = sqlite3.connect(source_path)
            conn.execute("CREATE TABLE sample(value INTEGER)")
            conn.execute("INSERT INTO sample VALUES (7)")
            conn.commit()
            conn.close()
            before = hashlib.sha256(source_path.read_bytes()).hexdigest()
            with patch.object(rehearsal.core_db, "read_snapshot",
                              wraps=rehearsal.core_db.read_snapshot) as snapshots:
                rehearsal._clone_read_only(source_path, clone_path)
                snapshots.assert_called_once_with(source_path)
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), before)
            self.assertEqual(sqlite3.connect(clone_path).execute(
                "SELECT value FROM sample"
            ).fetchone(), (7,))
            self.assertEqual(os.stat(clone_path).st_mode & 0o777, 0o600)

    def test_rehearsal_refuses_an_unreviewed_source_before_copy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "not-live.sqlite"
            source_path.touch()
            with patch.object(rehearsal, "_clone_read_only",
                              side_effect=AssertionError("copy started")):
                with self.assertRaisesRegex(
                    rehearsal.drain.replay.RepairStopped,
                    "reviewed container store",
                ):
                    rehearsal.rehearse(
                        source_path, root, expected_state="state",
                        expected_model_sha256="model", expected_dim=384,
                        expected_mirror_rows=823, expected_mirror_sha256="mirror",
                    )


if __name__ == "__main__":
    unittest.main()
