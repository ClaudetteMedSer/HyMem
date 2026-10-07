"""The shape inspector must never emit shell token values."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / 'tools/ops/afrodite/honcho_recovery_launch_structure.py'
spec = importlib.util.spec_from_file_location('honcho_recovery_launch_structure', SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_launch_shape_exposes_names_not_values():
    scope = {}
    prefix = module.REMOTE.split('\nvalidate()\n', 1)[0]
    exec(compile(prefix, '<shape-remote>', 'exec'), scope)
    value = 'super_secret_opaque_value'
    result = scope['launch_token_shape'](f'nohup env HYMEM_LLM_API_KEY={value} \\', 77)
    assert result['assignment_names'] == ['HYMEM_LLM_API_KEY']
    assert result['token_labels'] == ['nohup', 'env', 'assignment']
    assert result['continuation'] is True
    assert value not in repr(result)
