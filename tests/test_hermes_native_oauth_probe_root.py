"""Root-owned verification of the bounded probe, with invented offline calls."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks import hermes_lme_oauth_v1 as bridge
from hymem.extraction.llm import LLMRequest

_PATH = Path(__file__).resolve().parents[1] / 'tools/diagnostics/hermes_native_oauth_probe_v1.py'
_SPEC = importlib.util.spec_from_file_location('native_probe_root_test', _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def offline_probe(tmp_path, monkeypatch, *, failure=None):
    loaded = {'warm': bridge.warm, 'staged': bridge.staged_v6,
              'request_type': LLMRequest, 'binary': Path('/unused'),
              'candidate': Path(__file__).resolve().parents[1]}
    _, _, batch = probe.build_fixture(loaded)
    quota = bridge.warm.quota_metadata({'rateLimits': {
        'planType': 'pro', 'primary': {'usedPercent': 90},
        'credits': {'hasCredits': True, 'unlimited': False, 'balance': '1'}}})
    instances = []
    class Broker:
        def __init__(self, *args, **kwargs):
            self.closed = False
            instances.append(self)
        def admit(self, deadline):
            return bridge.native.Credentials('invented-token', 'invented-account'), {
                'auth': 'chatgpt', 'model': 'gpt-6-luna', 'quota_windows': quota,
                'config_isolation_admitted': True, 'inference_enabled': False}
        def close(self):
            self.closed = True
    calls = []
    def transport(credentials, system, user, schema, *, timeout):
        calls.append(schema)
        if failure:
            raise bridge.native.TransportError(failure)
        text = 'invented ordinary reply'
        if schema is not None:
            text = json.dumps({'schema': bridge.staged_v6.staged.ORIGINAL_SCHEMA,
                'batch_sha256': batch.batch_sha256, 'complete': True,
                'originals': [{'index': 0, 'original': {'state': 'not_established', 'support': None}}]})
        return bridge.native.Completed(text, 9, 4, 13, 0, 0)
    def client(*args, **kwargs):
        return bridge.NativeLMEClient(*args, **kwargs, transport=transport)
    fake = SimpleNamespace(AdmissionBroker=Broker, NativeLMEClient=client)
    monkeypatch.setattr(probe, '_live_containment', lambda receipt: True)
    monkeypatch.setattr(probe, '_resources', lambda receipt: (0, 0))
    result = probe.run_probe(loaded, fake, tmp_path, {})
    return result, calls, instances


def test_real_bridge_real_staged_contract_two_turns(tmp_path, monkeypatch):
    result, calls, brokers = offline_probe(tmp_path, monkeypatch)
    assert result['status'] == 'transport_verified'
    assert result['turns'] == 2 and result['known_tokens'] == 26
    assert result['usage_complete'] and result['staged_response_valid']
    assert result['native_summary']['successes'] == 2
    assert calls[0] is None and type(calls[1]) is dict
    assert all(b.closed for b in brokers)
    assert probe.validate_result(result)
    serialized = json.dumps(result)
    assert 'invented-token' not in serialized and 'invented-account' not in serialized
    assert 'invented ordinary reply' not in serialized


def test_http_access_denial_is_terminal_and_usage_unknown(tmp_path, monkeypatch):
    result, calls, brokers = offline_probe(tmp_path, monkeypatch, failure='access_failure')
    assert len(calls) == 1 and all(b.closed for b in brokers)
    assert result['status'] != 'transport_verified'
    assert result['turns'] == 1 and result['usage_complete'] is False
    assert result['first_failure']['code'] == 'access_failure'
    assert result['first_failure']['phase'] == 'http'
    assert probe.validate_result(result)


@pytest.mark.parametrize('change', ['saturated', 'summary_failure', 'bad_fault_shape', 'nonfinite'])
def test_reader_rejects_inconsistent_or_malformed_clean_result(tmp_path, monkeypatch, change):
    result, _, _ = offline_probe(tmp_path, monkeypatch)
    value = copy.deepcopy(result)
    if change == 'saturated':
        value['native_summary']['timing_saturated'] = True
    elif change == 'summary_failure':
        value['native_summary']['first_failure'] = {'code': 'timeout', 'phase': 'http'}
    elif change == 'bad_fault_shape':
        value['first_failure'] = {'private_text': 'do not export'}
    else:
        value['native_summary']['timing_seconds']['http'] = float('nan')
    assert probe.validate_result(value) is False


def test_success_exit_without_result_never_means_compatibility(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, '_root', lambda p: p)
    monkeypatch.setattr(probe, 'verify_sources', lambda p: (object(), None, {}))
    monkeypatch.setattr(probe, '_receipt', lambda *a, **k: {})
    monkeypatch.setattr(probe, '_terminal_runtime', lambda *a: ('clean_exit', True))
    result = probe.inspect(tmp_path, 'a' * 64)
    assert result['recursive_cleanup_verified'] is True
    assert result['status'] == 'unverified' and result['result'] is None


def test_actual_isolated_native_source_and_fixture_import(tmp_path):
    baseline = Path('/private/tmp/hymem-repaired-four-root-dtBv2p/bundle')
    root = tmp_path / 'source'
    shutil.copytree(baseline, root)
    repo = _PATH.parents[2]
    for relative in ('benchmarks/hermes_codex_responses_v1.py',
                     'benchmarks/hermes_lme_oauth_v1.py'):
        shutil.copyfile(repo / relative, root / 'code' / relative)
    shutil.copyfile(_PATH, root / probe.TOOL)
    source = r'''
import importlib.util, sys
from pathlib import Path
root = Path(sys.argv[1])
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
runner = load('frozen_source', root/'code/tools/diagnostics/luna_lme_diagnostic_v10.py')
loaded = runner.import_source_only(root, root/'source-map.json', runner.ACCEPTED_INVENTORY_SHA256)
from benchmarks import hermes_lme_oauth_v1 as bridge
assert bridge.warm is loaded['warm'] and bridge.staged_v6 is loaded['staged']
assert Path(bridge.__file__).resolve() == root/'code/benchmarks/hermes_lme_oauth_v1.py'
probe = load('isolated_probe', root/'hermes_native_oauth_probe_v1.py')
ordinary, request, batch = probe.build_fixture(loaded)
schema = loaded['staged'].staged.build_original_output_schema(batch)
native_request = bridge.native.build_request(request.system, request.user, schema)
assert native_request['model'] == 'gpt-6-luna'
assert native_request['text']['format']['schema'] == schema
assert loaded['source_only'] and loaded['questions'] == []
print('source-and-fixture-verified-no-inference')
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', source, str(root)],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'source-and-fixture-verified-no-inference'
