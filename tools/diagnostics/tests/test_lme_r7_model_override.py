"""The remote override parser must preserve every non-model byte."""
from __future__ import annotations

import hashlib
import unittest

from tools.diagnostics import lme_r7_model_override as override


namespace = {"__name__": "remote_test"}
exec(override.REMOTE, namespace)
transform = namespace["transform"]
check_pin = namespace["check_pin"]


class ModelOverrideTest(unittest.TestCase):
    def test_assignment_forms_and_other_lines(self):
        cases = (
            b'export HYMEM_LLM_MODEL="deepseek-v4-flash"\n',
            b"HYMEM_LLM_MODEL='deepseek-v4-flash'\r\n",
            b'  HYMEM_LLM_MODEL=deepseek-v4-flash # note\n',
            b'  HYMEM_LLM_MODEL="deepseek-v4-flash" \\\n',
        )
        for original in cases:
            source = b'# HYMEM_LLM_MODEL="other"\nAPI_KEY=secret\n' + original
            actual, count = transform(source, True)
            self.assertEqual(count, 1)
            self.assertEqual(actual, source.replace(b'deepseek-v4-flash',
                                                    b'deepseek-flash'))
            self.assertIn(b'API_KEY=secret\n', actual)

    def test_optional_env_without_assignment_is_unchanged(self):
        source = b'# HYMEM_LLM_MODEL=other\nAPI_KEY=secret\n'
        self.assertEqual(transform(source, False), (source, 0))
        current = source + b'HYMEM_LLM_MODEL="deepseek-flash"\n'
        self.assertEqual(transform(current, False), (current, 0))

    def test_other_or_ambiguous_assignments_refused(self):
        for line in (b'HYMEM_LLM_MODEL=other\n',
                     b'HYMEM_LLM_MODEL=${MODEL}\n',
                     b'HYMEM_LLM_MODEL="deepseek-v4-flash" command\n',
                     b'HYMEM_LLM_MODEL="deepseek-v4-flash"; echo hi\n'):
            with self.subTest(line=line), self.assertRaisesRegex(
                    RuntimeError, 'other_or_ambiguous_model_assignment'):
                transform(line, True)
        with self.assertRaisesRegex(RuntimeError,
                                    'missing_or_duplicate_model_assignment'):
            transform(b'HYMEM_LLM_MODEL=deepseek-v4-flash\n'*2, True)

    def test_exact_hash_required(self):
        before = b'HYMEM_LLM_MODEL=deepseek-v4-flash\n'
        check_pin(before, hashlib.sha256(before).hexdigest())
        with self.assertRaisesRegex(RuntimeError, 'target_hash_mismatch'):
            check_pin(before + b' ', hashlib.sha256(before).hexdigest())


if __name__ == '__main__':
    unittest.main()
