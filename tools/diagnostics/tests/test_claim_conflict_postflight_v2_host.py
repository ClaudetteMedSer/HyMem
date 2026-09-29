"""Offline configuration checks for the versioned host adapter."""
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from tools.diagnostics import claim_conflict_postflight_v2_host as v2


class HostAdapterTests(unittest.TestCase):
    def test_pinned_host_and_worker_mount_and_new_container_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = Path(v2.__file__).with_name('claim_conflict_postflight_host.py')
            host_file = root / 'postflight_host_v1.py'
            shutil.copyfile(original, host_file)
            host_file.chmod(0o400)
            host = v2.load_host(host_file)
            self.assertEqual(host.STAGE.name, 'postflight-v2')
            self.assertEqual(host.CHECKER_SHA, v2.CHECKER_SHA256)
            host.STAGE = root / 'postflight-v2'
            host.STAGE.mkdir(mode=0o700)
            prior = host.STAGE / 'postflight_v1.py'
            shutil.copyfile(original.with_name('claim_conflict_private_dream_postflight.py'), prior)
            prior.chmod(0o400)
            command, mounts = host.configure(SimpleNamespace(RUNTIME=root / 'runtime'))
            self.assertEqual(command[command.index('--name') + 1], 'hymem-private-dream-postflight-v2')
            self.assertIn((str(prior), '/diag/postflight_v1.py', False), mounts)
            self.assertIn('type=bind,src=' + str(prior) + ',dst=/diag/postflight_v1.py,readonly', command)
            self.assertEqual(command[command.index('--network') + 1], 'none')
            self.assertIn('--read-only', command)
            prior.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, 'file_mode_invalid'):
                host.configure(SimpleNamespace(RUNTIME=root / 'runtime'))

    def test_host_pin_and_mode_drift_fail_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'host.py'
            path.write_text('raise Exception("must not execute")')
            with self.assertRaisesRegex(RuntimeError, 'host_invalid'):
                v2.load_host(path)
            path.chmod(0o400)
            with self.assertRaisesRegex(RuntimeError, 'host_pin_drift'):
                v2.load_host(path)


if __name__ == '__main__':
    unittest.main()
