"""Fresh package, collision-resistant names and explicit launch isolation."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'lme_r8_summary_host.py'
spec = importlib.util.spec_from_file_location('r8_summary_host_test', PATH)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
PIN = 'a'*64


def test_offline_has_no_credential_or_network_and_only_clone_outputs_writable(tmp_path):
    for mode in ('offline', 'live'):
        plan = host.plan(tmp_path.resolve(), mode, PIN)
        assert {key for key, (_, writable) in plan['mounts'].items() if writable} == {'/work', '/results'}
        assert plan['mounts']['/reference'] == (host.REFERENCE, False)
        assert plan['mounts']['/previous'] == (host.PREVIOUS_ROOT+'/live-results', False)
        assert plan['mounts']['/previous-manifest.json'] == (host.PREVIOUS_ROOT+'/diag/manifest.json', False)
        assert '/run/deepseek.env' in plan['mounts'] if mode == 'live' else '/run/deepseek.env' not in plan['mounts']
        assert plan['network'] == ('bridge' if mode == 'live' else 'none')
    assert host.plan(tmp_path.resolve(), 'live', PIN)['name'] != host.plan(tmp_path.resolve()/'other', 'live', PIN)['name']
    assert host.plan(tmp_path.resolve(), 'live', PIN)['name'] != host.plan(tmp_path.resolve(), 'live', 'b'*64)['name']


def test_seal_pins_complete_inventory_and_refuses_existing_package(tmp_path):
    tree = tmp_path.resolve()/'tree'
    for relative in ('hymem/dreaming/digest.py', 'hymem/core/schema.sql', 'tests/test_case.py', 'resources/data.json'):
        path = tree/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('invented')
    package = tmp_path.resolve()/'package'
    result = host.seal(tree, package, {'hymem.sqlite': PIN})
    assert result['api_calls'] == 0 and result['source_files'] == 4
    assert host.sha(package/'diag/manifest.json') == result['manifest_sha256']
    assert (package/'candidate/resources/data.json').read_text() == 'invented'
    with pytest.raises(RuntimeError, match='new_package_required'):
        host.seal(tree, package, {'hymem.sqlite': PIN})


def test_live_without_offline_receipt_refuses_before_docker(tmp_path, monkeypatch):
    tree = tmp_path.resolve()/'tree'
    for relative in ('hymem/dreaming/digest.py', 'hymem/core/schema.sql'):
        path = tree/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('invented')
    package = tmp_path.resolve()/'package'
    receipt = host.seal(tree, package, {'hymem.sqlite': PIN})
    monkeypatch.setattr(host, 'run', lambda _: pytest.fail('launch before offline gate'))
    with pytest.raises(FileNotFoundError):
        host.action(package, 'live', receipt['manifest_sha256'])
