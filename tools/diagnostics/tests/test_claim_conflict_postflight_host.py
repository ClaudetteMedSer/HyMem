import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('postflight_host', Path(__file__).parents[1] / 'claim_conflict_postflight_host.py')
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)


def test_configure_is_network_none_and_only_work_writable():
    command, mounts = host.configure(SimpleNamespace(RUNTIME=Path('/runtime')))
    assert command[command.index('--network') + 1] == 'none'
    assert [dst for _, dst, rw in mounts if rw] == ['/work']
    assert '/run/runtime-env.json' not in [dst for _, dst, _ in mounts]
    assert '/private-dream' in [dst for _, dst, _ in mounts]
    assert command[command.index(host.IMAGE) + 1:] == [
        '-I', '-B', '/diag/postflight.py', '--worker-result', '/campaign/worker-result.json']
    assert '--read-only' in command and '--pull' in command


def test_seal_requires_all_reviewed_pins(tmp_path):
    path = tmp_path / 'seal.json'
    fields = {'host_sha256', 'install_sha256', 'result_sha256', 'live_container_id', 'source_sha256'}
    seal = {key: 'a' * 64 for key in fields}
    path.write_text(json.dumps(seal))
    path.chmod(0o400)
    assert host.read_seal(path, host.sha(path)) == seal
    with pytest.raises(RuntimeError, match='seal_pin_drift'):
        host.read_seal(path, 'b' * 64)
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match='file_mode_invalid'):
        host.read_seal(path, host.sha(path))


@pytest.mark.parametrize('change', [{'pid': 1}, {'exit_code': 1}, {'status': 'running'},
                                  {'oom_killed': True}, {'configuration_verified': False}])
def test_live_container_must_be_closed_successfully(change):
    state = dict(status='exited', pid=0, exit_code=0, oom_killed=False, configuration_verified=True)
    host.clean(state)
    state.update(change)
    with pytest.raises(RuntimeError, match='container_not_clean'):
        host.clean(state)


@pytest.mark.parametrize('value', [{'status': 'pass', 'payload': 'private source text'},
                                  {'status': 'pass', 'count': -1},
                                  {'status': 'pass', 'count': 100000001},
                                  {'status': 'completed'}])
def test_report_fails_closed_on_unbounded_content(value):
    with pytest.raises(RuntimeError, match='checker_output_invalid'):
        host.safe_report(value)


def test_safe_report_accepts_nested_counts_hashes_and_gates():
    report = {'status': 'pass', 'source_sha256': 'a' * 64,
              'schema': {'baseline': 63, 'dream': 64, 'valid': True},
              'campaign': {'worker_status': 'completed', 'chunks_processed': 1},
              'nullable': None}
    assert host.safe_report(report) == report


def test_no_remote_or_install_cli():
    text = Path(host.__file__).read_text()
    assert "['ssh'" not in text
    assert "['docker', 'restart'" not in text


@pytest.mark.parametrize('suffix', ['-wal', '-journal'])
def test_closed_database_rejects_nonempty_sidecars(tmp_path, suffix):
    source = tmp_path / 'hymem.sqlite'
    source.write_bytes(b'database')
    sidecar = Path(str(source) + suffix)
    sidecar.write_bytes(b'unsealed writes')
    with pytest.raises(RuntimeError, match='closed_source_unsealed_sidecar'):
        host.closed_source(source, host.sha(source))
    sidecar.write_bytes(b'')
    host.closed_source(source, host.sha(source))


def test_cleanup_stops_only_exact_validated_id_and_verifies_terminal():
    calls = []
    cid = 'a' * 64
    def run(command, *_):
        calls.append(command)
        if command[1] == 'inspect':
            return json.dumps([{'Id': cid, 'State': {'Status': 'exited', 'Pid': 0, 'Running': False}}]).encode()
        return b''
    helper = SimpleNamespace(run=run)
    host.cleanup(helper, cid)
    assert calls == [['docker', 'stop', '--time', '10', cid], ['docker', 'inspect', cid]]
    calls.clear()
    with pytest.raises(RuntimeError, match='cleanup_container_id_invalid'):
        host.cleanup(helper, 'container-name')
    assert calls == []


@pytest.mark.parametrize('status', ['pass', 'fail'])
def test_checker_success_and_failure_shapes_are_safe(status):
    report = dict(status=status,
                  schema={'baseline': 63, 'dream': 64, 'expected_transition': True},
                  publication={'baseline': {'target_current': 0, 'target_pinned': 0, 'target_pinned_with_proof': 0},
                               'dream': {'target_current': 1, 'target_pinned': 1, 'target_pinned_with_proof': 1},
                               'valid': True},
                  quarantines={'retry_limit': {'baseline': 1, 'dream': 1, 'introduced': 0},
                               'terminal_loss': {'baseline': 0, 'dream': 0, 'introduced': 0}},
                  baseline_quarantines_present=True, new_degradation_detected=status == 'fail',
                  worker_report_clean=status == 'pass', worker_store_campaign_consistent=True,
                  aggregation_blocking_present=False, audit_clean=True, source_unchanged=True,
                  baseline_sha256='a' * 64, source_sha256='b' * 64,
                  latest_dream_advanced=True, dream_completed=True,
                  campaign={'worker_status': 'completed', 'chunks_processed': 1, 'completion_calls': 1},
                  baseline_counts={'chunks': 1}, dream_counts={'chunks': 1},
                  dream_run_counters={'chunks_processed': 1, 'digest_failures': int(status == 'fail')},
                  baseline_integrity={'integrity_ok': True, 'foreign_key_violations': 0},
                  dream_integrity={'integrity_ok': True, 'foreign_key_violations': 0})
    assert host.safe_report(report)['status'] == status


def test_cleanup_verifies_even_if_stop_times_out():
    calls = []
    cid = 'a' * 64
    def run(command, *_):
        calls.append(command)
        if command[1] == 'stop':
            raise TimeoutError()
        return json.dumps([{'Id': cid, 'State': {'Status': 'running', 'Pid': 123, 'Running': True}}]).encode()
    with pytest.raises(RuntimeError, match='cleanup_not_terminal'):
        host.cleanup(SimpleNamespace(run=run), cid)
    assert calls[-1] == ['docker', 'inspect', cid]


@pytest.mark.parametrize('failure', ['postflight_start', 'postflight_wait', 'postflight_logs', 'inspect'])
def test_run_failure_cleans_up_created_container_and_preserves_one_shot(tmp_path, monkeypatch, failure):
    root = tmp_path / 'campaign'
    stage = root / 'postflight-v1'
    stage.mkdir(parents=True, mode=0o700)
    live = root / 'work/live'
    live.mkdir(parents=True, mode=0o700)
    (root / 'work').chmod(0o700)
    (live / 'hymem.sqlite').write_bytes(b'db')
    cid = 'a' * 64
    metadata = {'status': 'completed'}
    campaign = {'status': 'completed', 'paid_live_runs_started': 1,
                'stages': {'live': {'container_id': cid, 'metadata': metadata}}}
    seal = dict(host_sha256='b' * 64, install_sha256='b' * 64,
                result_sha256='b' * 64, source_sha256='b' * 64, live_container_id=cid)
    receipt = {'candidate_sha256': 'ed889d8970c6d7827315de34342996ad55182773c238922fbbdf8510d73c43a4',
               'phase1_sha256': 'bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac',
               'source_files': 481}
    calls = []
    def put_json(path, value):
        # Mirror the pinned helper's exclusive creation semantics.
        with path.open('x') as stream:
            json.dump(value, stream)
    def run(command, timeout, label):
        calls.append(label)
        if label == failure:
            raise TimeoutError()
        if label == 'private_worker_logs':
            return json.dumps(metadata).encode()
        if label in {'postflight_create', 'postflight_start'}:
            return cid.encode()
        if label == 'postflight_wait':
            return b'0'
        if label == 'postflight_cleanup_inspect':
            return json.dumps([{'Id': cid, 'State': {'Status': 'exited', 'Pid': 0, 'Running': False}}]).encode()
        if label == 'postflight_logs':
            return b'not json'
        return b''
    helper = SimpleNamespace(IMAGE=host.IMAGE, RUNTIME=Path('/runtime'), put_json=put_json,
                             read_json=lambda _: campaign, run=run)
    original_configure = lambda *_: None
    inspections = []
    def inspect(*_):
        inspections.append(1)
        if failure == 'inspect' and len(inspections) == 2:
            raise RuntimeError('inspection failed after start')
        return dict(status='created' if len(inspections) == 1 else 'exited', pid=0,
                    exit_code=0, oom_killed=False, configuration_verified=True)
    shared = SimpleNamespace(project=lambda raw: raw, configure=original_configure, inspect=inspect)
    controller = SimpleNamespace(installed=lambda *_: receipt, configure=lambda *_: ([], []),
                                 inspect=lambda *_: dict(status='exited', pid=0, exit_code=0,
                                                        oom_killed=False, configuration_verified=True))
    campaign_host = SimpleNamespace(dependencies=lambda: (shared, None, helper), controller=lambda: controller,
                                   PHASE1_SHA=receipt['phase1_sha256'])
    monkeypatch.setattr(host, 'ROOT', root)
    monkeypatch.setattr(host, 'STAGE', stage)
    monkeypatch.setattr(host.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(host, 'regular', lambda *_: None)
    monkeypatch.setattr(host, 'sha', lambda path: {'claim_conflict_private_dream_postflight.py': host.CHECKER_SHA,
                                                 'claim_conflict_store_audit.py': host.AUDIT_SHA}.get(path.name, 'b' * 64))
    monkeypatch.setattr(host, 'load_host', lambda _: campaign_host)
    with pytest.raises((TimeoutError, RuntimeError, ValueError)):
        host.run(seal)
    assert calls[-2:] == ['postflight_stop', 'postflight_cleanup_inspect']
    assert shared.configure is original_configure
    assert (stage / 'intent.json').exists()
    calls.clear()
    with pytest.raises(FileExistsError):
        host.run(seal)
    assert calls == []
