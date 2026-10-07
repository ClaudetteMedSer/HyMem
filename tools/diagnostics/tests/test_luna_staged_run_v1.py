"""Local entrypoint controls using only invented, unlaunched state."""
from pathlib import Path

import pytest

from tools.diagnostics import luna_staged_core_v1 as core
from tools.diagnostics import luna_staged_run_v1 as run


def _partial(*, attempted=0, turns=0, usage_complete=True):
    return {'schema': core.SCHEMA, 'units': [], 'completed_units': 0,
            'attempted_units': attempted, 'malformed_units': 0,
            'paid_budget': {'turns': turns, 'known_tokens': 0,
                            'usage_complete': usage_complete,
                            'in_flight': 0, 'reserved': 0},
            'first_failure': None, 'client_cleanup_ok': True,
            'stop_code': 'infrastructure_or_runtime_failure',
            'diagnostic_completed': False, 'semantic_accuracy_accepted': False,
            'full_lme_ready': False, 'completed_and_clean': False,
            'process_cleanup_verified': False}


def test_safe_projection_retains_unknown_usage_and_requires_later_reader():
    result = _partial(attempted=1, turns=1, usage_complete=False)
    safe = run._safe_result(result, 'a' * 64, 'b' * 64, 'c' * 64, False, 1.23456)
    assert safe['paid_budget'] == result['paid_budget']
    assert safe['diagnostic_completed'] is False
    assert safe['completed_and_clean'] is False
    assert safe['semantic_accuracy_accepted'] is False
    assert safe['full_lme_ready'] is False
    assert safe['process_groups_absent_at_entry_exit'] is False
    assert safe['elapsed_seconds'] == 1.235
    assert 'units' not in safe


def test_safe_projection_rejects_a_fabricated_completed_result():
    result = _partial()
    result['diagnostic_completed'] = True
    with pytest.raises(core.DiagnosticStop):
        run._safe_result(result, 'a' * 64, 'b' * 64, 'c' * 64, True, 0)


@pytest.mark.parametrize('path', [
    Path('/home/atta/.hymem-luna-semantic-probe-abcdefgh'),
    Path('/home/atta/.hymem-luna-staged-probe-short'),
    Path('/tmp/.hymem-luna-staged-probe-abcdefgh'),
])
def test_early_boundary_rejects_non_staged_or_unowned_roots(path):
    with pytest.raises(ValueError, match='root_or_receipt_invalid'):
        run._early(path, 'a' * 64)
