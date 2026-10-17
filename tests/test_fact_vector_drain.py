"""Guards for the local-only repair of the stopped partial fact replay."""

from contextlib import redirect_stdout
import io
import logging
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hymem.core import db as core_db
from hymem.core.vectors import encode_vector
from tools.diagnostics import fact_vector_drain as drain


class _Cursor:
    def fetchone(self):
        return None


class _LockFreeConnection:
    def execute(self, _sql):
        return _Cursor()


def _partial_state(**changes):
    result = {
        "state_sha256": "pinned-state", "outcomes": 128,
        "sessions": 49, "stale": 77, "pending_fresh": 1,
        "quarantined": 0,
    }
    result.update(changes)
    return result


class FactVectorDrainTests(unittest.TestCase):
    def test_target_requires_exact_reviewed_cap32_generation(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE fact_extraction_outcomes(prompt_version TEXT)")
        conn.execute("INSERT INTO fact_extraction_outcomes VALUES (?)",
                     (drain.REVIEWED_CAP32_TARGET,))
        self.assertEqual(drain._target32(conn), drain.REVIEWED_CAP32_TARGET)
        conn.execute("UPDATE fact_extraction_outcomes SET prompt_version=?",
                     (drain.REVIEWED_CAP32_TARGET.replace("tokens=3072", "tokens=4096"),))
        with self.assertRaisesRegex(drain.replay.RepairStopped, "reviewed cap-32"):
            drain._target32(conn)

    def test_exact_logical_physical_mirror_is_required(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE narrative_fact_embeddings(fact_id INTEGER, vector_json TEXT)")
        conn.execute("CREATE TABLE vec_facts(rowid INTEGER, embedding BLOB)")
        vector = [0.25, -0.5, 0.75]
        conn.execute("INSERT INTO narrative_fact_embeddings VALUES (?,?)",
                     (17, encode_vector(vector)))
        conn.execute("INSERT INTO vec_facts VALUES (?,?)",
                     (17, core_db._pack_vector(vector)))
        self.assertEqual(drain._mirror_digest(conn)[0], 1)
        conn.execute("UPDATE vec_facts SET embedding=? WHERE rowid=17",
                     (core_db._pack_vector([0.25, 0.5, 0.75]),))
        with self.assertRaisesRegex(drain.replay.RepairStopped, "mirror differs"):
            drain._mirror_digest(conn)

    def test_partial_baseline_change_stops_before_inventory_or_lease(self):
        with TemporaryDirectory() as directory:
            backup = Path(directory) / "backup.sqlite"
            with (patch.object(drain, "BACKUP_PATH", backup),
                  patch.object(drain.core_db, "schema_version", return_value=64),
                  patch.object(drain, "_target32", return_value="target32"),
                  patch.object(drain.replay, "_state", return_value=_partial_state(stale=76)),
                  patch.object(drain, "_inventory", side_effect=AssertionError("inventory ran")),
                  patch.object(drain.runner, "_acquire_lock", side_effect=AssertionError("lease ran"))):
                with self.assertRaisesRegex(drain.replay.RepairStopped, "baseline changed"):
                    drain.apply(
                        _LockFreeConnection(), drain.replay.LIVE_CONTAINER_DB, object(),
                        target32="target32", expected_state="pinned-state",
                        expected_inventory={"gaps": 902}, backup_path=backup,
                        max_batches=32, max_texts=2048,
                    )
            self.assertFalse(backup.exists())

    def test_zero_gap_stops_before_authority_or_lease(self):
        with TemporaryDirectory() as directory:
            backup = Path(directory) / "backup.sqlite"
            complete = {"gaps": 0}
            with (patch.object(drain, "BACKUP_PATH", backup),
                  patch.object(drain.core_db, "schema_version", return_value=64),
                  patch.object(drain, "_target32", return_value="target32"),
                  patch.object(drain.replay, "_state", return_value=_partial_state()),
                  patch.object(drain, "_inventory", return_value=complete),
                  patch.object(drain.replay, "_check_authority",
                               side_effect=AssertionError("authority ran")),
                  patch.object(drain.runner, "_acquire_lock",
                               side_effect=AssertionError("lease ran"))):
                with self.assertRaisesRegex(drain.replay.RepairStopped, "inventory differs"):
                    drain.apply(
                        _LockFreeConnection(), drain.replay.LIVE_CONTAINER_DB, object(),
                        target32="target32", expected_state="pinned-state",
                        expected_inventory=complete, backup_path=backup,
                        max_batches=32, max_texts=2048,
                    )
            self.assertFalse(backup.exists())

    def test_heartbeat_setup_failure_releases_acquired_lease(self):
        with TemporaryDirectory() as directory:
            backup = Path(directory) / "backup.sqlite"
            inventory = {"gaps": 1}
            with (patch.object(drain, "BACKUP_PATH", backup),
                  patch.object(drain.core_db, "schema_version", return_value=64),
                  patch.object(drain, "_target32", return_value="target32"),
                  patch.object(drain.replay, "_state",
                               return_value=_partial_state(session_ids=[])),
                  patch.object(drain, "_inventory", return_value=inventory),
                  patch.object(drain.replay, "_check_authority"),
                  patch.object(drain, "embedding_storage_identity",
                               return_value=("pinned-model", 384)),
                  patch.object(drain.runner, "_new_lease_token", return_value="lease"),
                  patch.object(drain.runner, "_acquire_lock", return_value=True),
                  patch.object(drain.runner, "_DreamLeaseHeartbeat",
                               side_effect=RuntimeError("heartbeat setup failed")),
                  patch.object(drain.runner, "_release_lock") as release):
                with self.assertRaisesRegex(RuntimeError, "heartbeat setup failed"):
                    drain.apply(
                        _LockFreeConnection(), drain.replay.LIVE_CONTAINER_DB, object(),
                        target32="target32", expected_state="pinned-state",
                        expected_inventory=inventory, backup_path=backup,
                        max_batches=32, max_texts=2048,
                    )
                release.assert_called_once_with(unittest.mock.ANY, "lease")
            self.assertFalse(backup.exists())

    def test_embedding_env_loader_ignores_llm_settings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, command in (("101", "hymem-honcho"), ("102", "hymem-server")):
                process = root / pid
                process.mkdir()
                (process / "cmdline").write_bytes(command.encode() + b"\0")
                (process / "environ").write_bytes(
                    b"HYMEM_EMBEDDING_BASE_URL=http://embedding-server:8766/v1\0"
                    b"HYMEM_EMBEDDING_MODEL=local-model\0"
                    b"HYMEM_LLM_API_KEY=synthetic-llm-secret\0"
                )
            with patch.dict(os.environ, {}, clear=True):
                drain._load_running_embedding_env(root)
                self.assertEqual(os.environ["HYMEM_EMBEDDING_MODEL"], "local-model")
                self.assertNotIn("HYMEM_LLM_API_KEY", os.environ)

    def test_main_restores_prior_logging_disable_on_failure(self):
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.sqlite"
            db_path.touch()
            old_disable = logging.root.manager.disable
            logging.disable(logging.INFO)
            try:
                with (patch.object(sys, "argv", ["fact_vector_drain.py", "--db", str(db_path)]),
                      patch.object(drain.replay, "_is_live_store", return_value=False),
                      patch.object(drain.replay, "_embedder_from_env",
                                   side_effect=drain.replay.RepairStopped("synthetic stop")),
                      redirect_stdout(io.StringIO())):
                    self.assertEqual(drain.main(), 2)
                self.assertEqual(logging.root.manager.disable, logging.INFO)
            finally:
                logging.disable(old_disable)


if __name__ == "__main__":
    unittest.main()
