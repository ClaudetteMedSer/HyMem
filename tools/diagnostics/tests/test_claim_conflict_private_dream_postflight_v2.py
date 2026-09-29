"""Offline sealed-main-file regressions; no provider or remote operations."""
from pathlib import Path
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools.diagnostics import claim_conflict_private_dream_postflight_v2 as v2


class SealedBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'sealed.sqlite'
        self.work = self.root / 'work'
        self.work.mkdir(mode=0o700)
        conn = sqlite3.connect(self.source)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE sample (value INTEGER)')
        conn.executemany('INSERT INTO sample VALUES (?)', [(n,) for n in range(17)])
        conn.commit()
        conn.close()
        self.source.chmod(0o400)
        self.pin = v2.digest(self.source)
        self.reader = v2.backup_reader({self.source: (self.pin, 0o400, 'copy.sqlite')}, self.work)
        self.target = self.work / 'copy.sqlite'

    def tearDown(self):
        self.temp.cleanup()

    def test_closed_wal_header_copy_preserves_counts_and_original(self):
        self.assertEqual(self.source.read_bytes()[18:20], b'\x02\x02')
        self.reader(self.source, self.target)
        with sqlite3.connect(self.target) as copy:
            self.assertEqual(copy.execute('SELECT COUNT(*), SUM(value) FROM sample').fetchone(), (17, 136))
            self.assertEqual(copy.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        self.assertEqual(v2.digest(self.source), self.pin)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o600)
        self.assertFalse(Path(str(self.source) + '-wal').exists())
        self.assertFalse(Path(str(self.source) + '-shm').exists())

    def test_reader_uri_is_readonly_and_immutable(self):
        real = sqlite3.connect
        with patch.object(v2.sqlite3, 'connect', wraps=real) as connect:
            self.reader(self.source, self.target)
        self.assertEqual(connect.call_args_list[0].args[0], self.source.as_uri() + '?mode=ro&immutable=1')
        self.assertEqual(connect.call_args_list[0].kwargs, {'uri': True})

    def test_nonempty_sidecars_and_symlinks_rejected_before_backup(self):
        for suffix in ('-wal', '-journal'):
            sidecar = Path(str(self.source) + suffix)
            sidecar.write_bytes(b'pending')
            with self.assertRaisesRegex(ValueError, 'nonempty_sidecar'):
                self.reader(self.source, self.target)
            self.assertFalse(self.target.exists())
            sidecar.unlink()
            sidecar.symlink_to(self.source)
            with self.assertRaisesRegex(ValueError, 'not_regular'):
                self.reader(self.source, self.target)
            sidecar.unlink()

    def test_source_pin_and_mode_and_paths_rejected(self):
        self.source.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'mode_invalid'):
            self.reader(self.source, self.target)
        self.source.chmod(0o400)
        with self.assertRaisesRegex(ValueError, 'input_path_invalid'):
            self.reader(self.root / 'other.sqlite', self.target)
        with self.assertRaisesRegex(ValueError, 'output_path_invalid'):
            self.reader(self.source, self.work / 'other.sqlite')
        bad = v2.backup_reader({self.source: ('0' * 64, 0o400, 'copy.sqlite')}, self.work)
        with self.assertRaisesRegex(ValueError, 'pin_drift'):
            bad(self.source, self.target)

    def test_post_backup_hash_and_sidecar_changes_rejected(self):
        real = v2.sealed
        calls = 0

        def alter(path, expected, mode):
            nonlocal calls
            calls += 1
            if calls == 2:
                Path(str(path) + '-journal').write_bytes(b'pending')
            return real(path, expected, mode)

        with patch.object(v2, 'sealed', side_effect=alter):
            with self.assertRaisesRegex(ValueError, 'nonempty_sidecar'):
                self.reader(self.source, self.target)
        Path(str(self.source) + '-journal').unlink()
        self.target.unlink()
        calls = 0

        def mutate(path, expected, mode):
            nonlocal calls
            calls += 1
            if calls == 2:
                path.chmod(0o600)
                with path.open('ab') as stream:
                    stream.write(b'changed')
                path.chmod(0o400)
            return real(path, expected, mode)

        with patch.object(v2, 'sealed', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'pin_drift'):
                self.reader(self.source, self.target)

    def test_target_existing_and_invalid_database_fail(self):
        self.target.write_bytes(b'previous-evidence')
        with self.assertRaises(FileExistsError):
            self.reader(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), b'previous-evidence')
        self.target.unlink()
        self.source.chmod(0o600)
        self.source.write_bytes(b'not a SQLite database')
        self.source.chmod(0o400)
        invalid = v2.backup_reader({self.source: (v2.digest(self.source), 0o400, 'copy.sqlite')}, self.work)
        with self.assertRaises(sqlite3.DatabaseError):
            invalid(self.source, self.target)

    def test_exact_v1_loaded_and_only_backup_callback_replaced(self):
        directory = Path(v2.__file__).parent
        prior = v2.load_v1(directory / 'claim_conflict_private_dream_postflight.py')
        audit = prior.load_audit(directory / 'claim_conflict_store_audit.py')
        self.assertEqual(prior.PHASE1_SHA256, 'bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac')
        self.assertEqual(audit.backup_readonly.__module__, v2.__name__)
        self.assertEqual(audit.audit.__module__, 'private_dream_pinned_audit')
        with self.assertRaisesRegex(ValueError, 'input_path_invalid'):
            audit.backup_readonly(self.source, self.target)

    def test_v1_pin_drift_rejected(self):
        fake = self.root / 'v1.py'
        fake.write_text('raise RuntimeError("must never execute")')
        with self.assertRaisesRegex(RuntimeError, 'postflight_v1_pin_drift'):
            v2.load_v1(fake)


if __name__ == '__main__':
    unittest.main()
