"""Local, no-network checks of the pinned recovery parser and launch fences."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'tools/ops/afrodite/honcho_recovery_source_start.py'
spec = importlib.util.spec_from_file_location('honcho_recovery_source_start', SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def remote_namespace(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['remote', 'plan'])
    prefix = module.REMOTE.split("\nif ACTION=='plan':", 1)[0]
    scope = {}
    exec(compile(prefix, '<pinned-remote>', 'exec'), scope)
    return scope


def write_hook(path, names, overrides=None):
    overrides = overrides or {}
    lines = ['# placeholder'] * 76 + ['(cd /home/node/HyMem && nohup env \\']
    lines += [f'{name}={overrides.get(name, "literal")} \\' for name in names]
    lines += ['/home/node/hymem-env/bin/hymem-honcho >> /private/log 2>&1 &)']
    path.write_text('\n'.join(lines) + '\n')


def test_pinned_hook_block(monkeypatch, tmp_path):
    scope = remote_namespace(monkeypatch)
    path = tmp_path / 'hook'
    write_hook(path, scope['NAMES'])
    scope['HOOK'] = path
    parsed = scope['hook_assignments']({})
    assert tuple(parsed) == scope['NAMES']
    assert set(parsed.values()) == {'literal'}


def test_hook_whole_approved_reference(monkeypatch, tmp_path):
    scope = remote_namespace(monkeypatch)
    path = tmp_path / 'hook'
    write_hook(path, scope['NAMES'], {'HYMEM_LLM_API_KEY': '"${DEEPSEEK_API_KEY:-}"'})
    scope['HOOK'] = path
    parsed = scope['hook_assignments']({'DEEPSEEK_API_KEY': 'opaque'})
    assert parsed['HYMEM_LLM_API_KEY'] == 'opaque'


def test_hook_rejects_partial_or_unknown_reference(monkeypatch, tmp_path):
    scope = remote_namespace(monkeypatch)
    path = tmp_path / 'hook'
    scope['HOOK'] = path
    for expression in ('"prefix$DEEPSEEK_API_KEY"', '"$UNAPPROVED_KEY"',
                       '"${DEEPSEEK_API_KEY:-fallback}"', "'${DEEPSEEK_API_KEY:-}'"):
        write_hook(path, scope['NAMES'], {'HYMEM_LLM_API_KEY': expression})
        with pytest.raises(RuntimeError):
            scope['hook_assignments']({'DEEPSEEK_API_KEY': 'opaque'})


def test_env_file_with_nineteen_static_assignments(monkeypatch, tmp_path):
    scope = remote_namespace(monkeypatch)
    path = tmp_path / '.env'
    lines = ['DEEPSEEK_API_KEY=opaque']
    lines += [f'PROVIDER_{index}={"" if index % 3 == 0 else "opaque"}' for index in range(18)]
    path.write_text('\n'.join(lines) + '\n')
    scope['ENV_FILE'] = path
    assert len(scope['env_file_values']()) == 19


@pytest.mark.parametrize('bad', ['$(id)', '$KEY', '`id`', 'literal\\escaped'])
def test_rejects_dynamic_hook_values(monkeypatch, tmp_path, bad):
    scope = remote_namespace(monkeypatch)
    path = tmp_path / 'hook'
    write_hook(path, scope['NAMES'], {scope['NAMES'][0]: bad})
    scope['HOOK'] = path
    with pytest.raises(RuntimeError):
        scope['hook_assignments']({})


def test_rejects_duplicate_or_reordered_hook_keys(monkeypatch, tmp_path):
    scope = remote_namespace(monkeypatch)
    path = tmp_path / 'hook'
    names = list(scope['NAMES'])
    names[1] = names[0]
    write_hook(path, names)
    scope['HOOK'] = path
    with pytest.raises(RuntimeError, match='hook_assignment_order'):
        scope['hook_assignments']({})


def test_plan_is_read_only_and_start_requires_plan(monkeypatch):
    scope = remote_namespace(monkeypatch)
    calls = []
    scope['source_fences'] = lambda: calls.append('source')
    scope['configuration'] = lambda: ({name: 'x' for name in scope['NAMES']},
        {'DEEPSEEK_API_KEY': 'x'}, 2, {'honcho': '0', 'wrapper': '1', 'mcp': ['0','1']})
    scope['runtime_fences'] = lambda: calls.append('runtime')
    _, result = scope['plan']()
    assert result['ready'] is True
    assert calls == ['source', 'runtime']


def test_configuration_rejects_wrapper_drift(monkeypatch):
    scope = remote_namespace(monkeypatch)
    names = scope['NAMES']
    hook = {name: 'same' for name in names}
    wrapper = {name: 'same' for name in names if name != 'HYMEM_DREAM_COOLDOWN_SECONDS'}
    wrapper['HYMEM_LLM_BASE_URL'] = 'different'
    scope['env_file_values'] = lambda: {'DEEPSEEK_API_KEY': 'same'}
    scope['hook_assignments'] = lambda base: hook
    scope['wrapper_assignments'] = lambda base: wrapper
    scope['mcp_environments'] = lambda: [wrapper]
    with pytest.raises(RuntimeError, match='wrapper_hook_config_mismatch'):
        scope['configuration']()


def test_aggregation_flag_can_differ_per_launcher_only_as_binary(monkeypatch):
    scope = remote_namespace(monkeypatch)
    hook = {name: 'same' for name in scope['NAMES']}
    hook.update({'HYMEM_LLM_API_KEY': 'secret', 'HYMEM_ROOT': '/home/node/.hermes',
                 'HYMEM_EMBEDDING_DIM': '384',
                 'HYMEM_AGGREGATION_NODES_ENABLED': '1'})
    wrapper = {name: value for name, value in hook.items()
               if name != 'HYMEM_DREAM_COOLDOWN_SECONDS'}
    wrapper['HYMEM_AGGREGATION_NODES_ENABLED'] = 'true'
    mcp = dict(wrapper)
    scope['env_file_values'] = lambda: {'DEEPSEEK_API_KEY': 'secret'}
    scope['hook_assignments'] = lambda base: hook
    scope['wrapper_assignments'] = lambda base: wrapper
    scope['mcp_environments'] = lambda: [mcp]
    config, base, count, flags = scope['configuration']()
    assert config['HYMEM_AGGREGATION_NODES_ENABLED'] == '1'
    assert flags == {'honcho': '1', 'wrapper': 'true', 'mcp': ['true']}
    wrapper['HYMEM_AGGREGATION_NODES_ENABLED'] = 'on'
    with pytest.raises(RuntimeError, match='aggregation_flag_invalid'):
        scope['configuration']()
    wrapper['HYMEM_AGGREGATION_NODES_ENABLED'] = 'false'
    mcp['HYMEM_AGGREGATION_NODES_ENABLED'] = 'false'
    with pytest.raises(RuntimeError, match='aggregation_flag_mismatch'):
        scope['configuration']()


def test_start_never_spawns_if_plan_fails(monkeypatch):
    scope = remote_namespace(monkeypatch)
    calls = []
    scope['plan'] = lambda: (_ for _ in ()).throw(RuntimeError('stage_receipt_changed'))
    scope['receipt'] = lambda *args: calls.append('receipt')
    monkeypatch.setattr(scope['subprocess'], 'Popen',
        lambda *args, **kwargs: calls.append('spawn'))
    with pytest.raises(RuntimeError, match='stage_receipt_changed'):
        scope['start']()
    assert calls == []


def test_start_does_not_launch_with_prior_intent(monkeypatch, tmp_path):
    scope = remote_namespace(monkeypatch)
    scope['STAGE'] = tmp_path
    (tmp_path / 'recovery-source-failed.json').touch()
    scope['plan'] = lambda: ({name: 'literal' for name in scope['NAMES']}, {})
    calls = []
    scope['receipt'] = lambda *args: calls.append('receipt')
    monkeypatch.setattr(scope['subprocess'], 'Popen',
        lambda *args, **kwargs: calls.append('spawn'))
    with pytest.raises(RuntimeError, match='prior_launch_failure'):
        scope['start']()
    assert calls == []


def test_start_launches_exact_fourteen_over_base_env(monkeypatch, tmp_path, capsys):
    scope = remote_namespace(monkeypatch)
    scope['STAGE'] = tmp_path
    scope['LIVE'] = tmp_path
    config = {name: f'value{index}' for index, name in enumerate(scope['NAMES'])}
    scope['plan'] = lambda: (config, {'ready': True})
    receipts = []
    scope['receipt'] = lambda path, value: receipts.append((path.name, value))
    scope['process_identity'] = lambda pid: {'pid': pid, 'uid': 1000,
        'exe': scope['EXE'], 'cwd': str(tmp_path), 'start_ticks': 123}
    class Child:
        pid = 12345
        def poll(self): return 1  # Exit promptly, avoiding all real /proc checks.
    launched = []
    with monkeypatch.context() as patch:
        patch.setattr(scope['subprocess'], 'Popen',
            lambda args, **kwargs: launched.append((args, kwargs)) or Child())
        scope['start']()
    assert len(launched) == 1
    args, kwargs = launched[0]
    assert args == scope['ARGV']
    assert kwargs['executable'] == scope['EXE'].encode()
    assert kwargs['cwd'] == str(tmp_path)
    for name, value in config.items():
        assert kwargs['env'][name.encode()] == value.encode()
    assert [name for name, _ in receipts] == [
        'recovery-source-intent.json', 'recovery-source-launch.json',
        'recovery-source-result.json']
    assert 'source-launch-unverified' in capsys.readouterr().out


def test_env_file_keeps_static_base_assignments(monkeypatch, tmp_path):
    scope = remote_namespace(monkeypatch)
    path = tmp_path / '.env'
    lines = ['DEEPSEEK_API_KEY=opaque']
    lines += [f'PROVIDER_{index}={"" if index == 0 else f"value{index}"}' for index in range(13)]
    lines += ['MISTRAL_API_KEY=first', 'MISTRAL_API_KEY=second']
    path.write_text('\n'.join(lines) + '\n')
    scope['ENV_FILE'] = path
    values = scope['env_file_values']()
    assert len(values) == 15
    assert values['DEEPSEEK_API_KEY'] == 'opaque'
    assert values['PROVIDER_0'] == ''
    assert values['MISTRAL_API_KEY'] == 'second'


def test_literal_empty_quoted_values_are_static(monkeypatch):
    scope = remote_namespace(monkeypatch)
    assert scope['parse_literal']('', {}, False, 'env', 'EMPTY') == ''
    assert scope['parse_literal']("''", {}, False, 'env', 'EMPTY') == ''
    assert scope['parse_literal']('""', {}, False, 'env', 'EMPTY') == ''


def test_env_file_rejects_unparsed_shell(monkeypatch, tmp_path):
    scope = remote_namespace(monkeypatch)
    path = tmp_path / '.env'
    path.write_text('DEEPSEEK_API_KEY=opaque\n. /other/file\n')
    scope['ENV_FILE'] = path
    with pytest.raises(RuntimeError, match='env_file_nonassignment'):
        scope['env_file_values']()
