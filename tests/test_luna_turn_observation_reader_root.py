"""Root-only negative controls for the new source-bound observer reader."""
import ast
import copy
import hashlib
from pathlib import Path

import pytest

from tools.diagnostics import luna_lme_diagnostic_progress_v5 as reader


def state():
    return dict(basis='consumed_observed_shape', events_consumed=3,
                completed_seen=True, final_seen=True, final_count=1,
                usage_update_count=0, usage_state='absent', last_event_family='turn_completed')


def budget():
    return {'stop_code': 'incomplete_turn_or_usage', 'first_failure': {
        'code': 'incomplete_turn_or_usage', 'phase': 'run', 'rpc': 'turn/events',
        'resource_observation': {'current': 95, 'peak': 132, 'limit': 256, 'denials': 0},
        'process_index': 16, 'request_index': 7, 'queue_count': 0, 'retired_count': 6,
        'turn_admitted': True, 'known_usage': False, 'usage_complete': False,
        'turn_observation': state()}}


@pytest.mark.parametrize('patch', [
    {},
    {'events_consumed': 2, 'final_seen': False, 'final_count': 0,
     'usage_update_count': 1, 'usage_state': 'positive'},
    {'events_consumed': 4, 'usage_update_count': 1, 'usage_state': 'zero'},
    {'events_consumed': 4096, 'completed_seen': False, 'final_seen': False,
     'final_count': 0, 'last_event_family': 'thread_status_changed'},
])
def test_four_literal_observations_do_not_promote_unknown_usage(patch):
    fixture = budget()
    fixture['first_failure']['turn_observation'].update(patch)
    before = copy.deepcopy(fixture)
    result = reader._failure_projection(fixture)
    assert result['known_usage'] is False and result['usage_complete'] is False
    assert result['turn_observation'] == fixture['first_failure']['turn_observation']
    assert fixture == before


@pytest.mark.parametrize('patch', [
    {'events_consumed': True}, {'events_consumed': 0}, {'events_consumed': 4097},
    {'usage_state': []}, {'usage_state': {'private': 'secret'}},
    {'last_event_family': []}, {'last_event_family': 'INVENTED_PRIVATE_TEXT'},
    {'last_event_family': 'error'}, {'last_event_family': 'unknown'},
    {'last_event_family': 'invalid_method'}, {'last_event_family': 'turn_failed'},
    {'final_count': 2}, {'completed_seen': 1}, {'final_seen': False},
    {'usage_update_count': 1}, {'usage_update_count': 4, 'usage_state': 'positive'},
    {'events_consumed': 4, 'usage_update_count': 1, 'usage_state': 'positive'},
    {'completed_seen': False},
    {'completed_seen': False, 'events_consumed': 4096},
    {'extra_private_field': 'secret'},
])
def test_contradictory_or_private_metadata_rejected(patch):
    fixture = budget()
    fixture['first_failure']['turn_observation'].update(patch)
    with pytest.raises(ValueError):
        reader._failure_projection(fixture)


@pytest.mark.parametrize('patch', [
    {'phase': 'preflight'}, {'rpc': 'turn/start'}, {'turn_admitted': False},
    {'known_usage': True}, {'code': 'stage_accounting_failure'},
])
def test_observation_cannot_be_attached_to_another_fault(patch):
    fixture = budget()
    fixture['first_failure'].update(patch)
    with pytest.raises(ValueError):
        reader._failure_projection(fixture)


def test_new_profile_cannot_silently_drop_observation_or_fault():
    fixture = budget()
    del fixture['first_failure']['turn_observation']
    with pytest.raises(ValueError):
        reader._failure_projection(fixture)
    fixture['first_failure'] = None
    with pytest.raises(ValueError):
        reader._failure_projection(fixture)


@pytest.mark.parametrize('code,sample', [
    ('resource_observer_unverified', None),
    ('resource_task_denial', {'current': 256, 'peak': 256, 'limit': 256, 'denials': 1}),
])
def test_precleanup_resource_override_preserves_actual_underlying_observation(code, sample):
    fixture = budget()
    fixture['stop_code'] = code
    fixture['first_failure'].update(code=code, underlying_code='incomplete_turn_or_usage',
                                    resource_observation=sample)
    result = reader._failure_projection(fixture)
    assert result['code'] == code
    assert result['underlying_code'] == 'incomplete_turn_or_usage'
    assert result['turn_observation'] == state()
    assert result['known_usage'] is False
    del fixture['first_failure']['turn_observation']
    with pytest.raises(ValueError):
        reader._failure_projection(fixture)


def test_resource_override_cannot_be_forged_without_denial():
    fixture = budget()
    fixture['first_failure'].update(code='resource_task_denial',
                                    underlying_code='incomplete_turn_or_usage')
    with pytest.raises(ValueError):
        reader._failure_projection(fixture)


@pytest.mark.parametrize('kind', ['plain', 'denial', 'unreadable'])
def test_actual_nested_budget_retains_observation_before_cleanup(kind):
    from benchmarks import codex_subscription_warm_v4 as warm
    from tools.diagnostics import luna_lme_diagnostic_v4 as runner
    tree = ast.parse(Path(runner.__file__).read_text())
    campaign = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'run_campaign')
    nested = next(node for node in ast.walk(campaign) if isinstance(node, ast.ClassDef)
                  and node.name == 'ObservedBudget')
    calls = []
    def sample(group):
        calls.append(group)
        if kind == 'unreadable':
            raise OSError('synthetic unavailable counter')
        return dict(current=95, peak=132, limit=256, denials=int(kind == 'denial'))
    namespace = dict(warm=warm, campaign_limits=warm.BudgetLimits(12, 160000, 600),
                     workers=4, _resource_sample=sample, resource_cgroup='/bound.service')
    # Execute the actual nested class, not a test reimplementation of its behavior.
    exec(compile(ast.Module(body=[nested], type_ignores=[]), '<actual-observed-budget>',
                 'exec'), namespace)
    actual = namespace['ObservedBudget']()
    metadata = budget()['first_failure']
    metadata.pop('code')
    metadata.pop('resource_observation')
    actual.record_first_failure('incomplete_turn_or_usage', metadata)
    assert calls == ['/bound.service']
    projected = reader._failure_projection(actual.snapshot())
    assert projected['turn_observation'] == state()
    assert projected['known_usage'] is False
    assert projected['code'] == {'plain': 'incomplete_turn_or_usage',
        'denial': 'resource_task_denial', 'unreadable': 'resource_observer_unverified'}[kind]
    if kind != 'plain':
        assert projected['underlying_code'] == 'incomplete_turn_or_usage'


def test_pins_and_unchanged_measurement_resource_logic():
    root = Path(__file__).resolve().parents[1]
    from tools.diagnostics import luna_lme_diagnostic_v4 as runner
    assert reader.RUNNER_SHA256 == hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
    for relative, expected in runner.PINS.items():
        assert reader.PINS[relative] == expected
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    assert reader.PINS['benchmarks/codex_subscription_staged_v2.py'] == runner.PINS['benchmarks/codex_subscription_staged_v2.py']
    def defs(path):
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old = defs(root / 'tools/diagnostics/luna_lme_diagnostic_v3.py')
    new = defs(Path(runner.__file__))
    for name in ('run_campaign', 'run_live_canary', '_question_worker', '_resource_sample',
                 'make_dual', 'AccountedClient', 'validate_diagnostic_row'):
        assert old[name] == new[name], name
