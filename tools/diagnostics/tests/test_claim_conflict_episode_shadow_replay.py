"""Offline synthetic tests; no private store, provider, environment or network."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

import sqlite_vec
from hymem.core import db
from tools.diagnostics import claim_conflict_episode_shadow_replay as worker


def connection(path):
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class EpisodeShadowReplayTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source.sqlite'
        conn = connection(source)
        conn.execute('CREATE TABLE evidence(id INTEGER PRIMARY KEY,value TEXT)')
        conn.execute("INSERT INTO evidence VALUES(1,'private fixture text')")
        conn.execute('CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding float[4])')
        conn.execute('CREATE VIRTUAL TABLE vec_other USING vec0(embedding float[4])')
        vector = struct.pack('4f', 1, 2, 3, 4)
        for key in range(1, 45):
            conn.execute('INSERT INTO vec_episodes(rowid,embedding) VALUES(?,?)', (key, vector))
        conn.execute('INSERT INTO vec_other(rowid,embedding) VALUES(1,?)', (vector,))
        conn.commit()
        conn.close()
        source.chmod(0o600)
        work = root / 'work'
        work.mkdir(mode=0o700)
        return source, work, vector

    def test_immutable_clone_preserves_source_and_rejects_reuse(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            source, work, _ = self.fixture(Path(directory))
            expected = worker.sha(source)
            identity = worker.source_guard(source, expected)
            worker.clone_immutable(source, work / 'clone.sqlite', expected)
            self.assertEqual(worker.source_guard(source, expected), identity)
            self.assertEqual((work / 'clone.sqlite').stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                worker.clone_immutable(source, work / 'clone.sqlite', expected)

    def test_guard_rejects_pin_drift_sidecars_and_symlink_ancestors(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            root = Path(directory)
            source, _, _ = self.fixture(root)
            expected = worker.sha(source)
            with self.assertRaisesRegex(RuntimeError, 'pin_drift'):
                worker.source_guard(source, '0' * 64)
            for suffix in ('-wal', '-shm', '-journal'):
                sidecar = Path(str(source) + suffix)
                sidecar.touch()
                with self.assertRaisesRegex(RuntimeError, 'sidecar'):
                    worker.source_guard(source, expected)
                sidecar.unlink()
            link = root / 'linked-parent'
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, 'symlink'):
                worker.source_guard(link / 'source.sqlite', expected)

    def test_fingerprint_excludes_only_episode_rows_and_own_shadows(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            source, _, vector = self.fixture(Path(directory))
            conn = connection(source)
            before = worker.semantic_fingerprint(conn)
            conn.execute('DELETE FROM vec_episodes WHERE rowid>36')
            self.assertEqual(worker.semantic_fingerprint(conn), before)
            conn.execute('UPDATE evidence SET id=2')
            self.assertNotEqual(worker.semantic_fingerprint(conn), before)
            conn.rollback()
            self.assertEqual(worker.semantic_fingerprint(conn), before)
            conn.execute('INSERT INTO vec_other(rowid,embedding) VALUES(2,?)', (vector,))
            self.assertNotEqual(worker.semantic_fingerprint(conn), before)
            conn.close()

    def test_orchestration_repair_repeat_reopen_and_preservation(self):
        # Synthetic snapshot supplies authority; application proof/refusal tests
        # belong to the candidate helper suite, not this worker test.
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            source, work, vector = self.fixture(Path(directory))
            expected = {key: vector for key in range(1, 37)}
            def snapshot(conn, _db):
                actual = {int(row[0]): bytes(row[1]) for row in conn.execute(
                    'SELECT rowid,embedding FROM vec_episodes')}
                worker.require(all(actual.get(k) == v for k, v in expected.items()),
                               'missing_or_different_vector')
                return expected, actual
            calls = []
            def prune(conn):
                calls.append(True)
                count = conn.execute('SELECT COUNT(*) FROM vec_episodes WHERE rowid>36').fetchone()[0]
                with db.transaction(conn):
                    conn.execute('DELETE FROM vec_episodes WHERE rowid>36')
                return bool(count)
            audit = types.SimpleNamespace(integrity=lambda conn: {'clean': True},
                                          is_clean=lambda report: report['clean'])
            source_sha = worker.sha(source)
            with patch.object(worker, 'load_audit', return_value=audit), \
                 patch.object(worker, 'vector_snapshot', side_effect=snapshot), \
                 patch.object(db, 'schema_version', return_value=64), \
                 patch.object(db, 'prune_extra_episode_vectors', side_effect=prune, create=True), \
                 patch.object(db, 'vec_episodes_aligned', return_value=True):
                report = worker.replay(source, work, source_sha)
            self.assertEqual(len(calls), 2)
            self.assertEqual(report['removed_surplus'], 8)
            self.assertTrue(report['semantic_rows_unchanged'])
            self.assertTrue(report['reopen_aligned'])
            self.assertEqual(worker.sha(source), source_sha)
            self.assertNotIn('private fixture text', json.dumps(report))

    def test_failure_stdout_is_static_and_private_evidence_is_0600(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as directory:
            work = Path(directory)
            stdout = io.StringIO()
            with patch.object(worker, 'WORK', work), \
                 patch.object(worker, 'replay', side_effect=RuntimeError('PRIVATE secret')), \
                 contextlib.redirect_stdout(stdout):
                self.assertEqual(worker.main(), 1)
            self.assertNotIn('PRIVATE', stdout.getvalue())
            failure = work / 'episode-shadow-failure.txt'
            self.assertEqual(failure.stat().st_mode & 0o777, 0o600)
            self.assertIn('PRIVATE secret', failure.read_text())


if __name__ == '__main__':
    unittest.main()
