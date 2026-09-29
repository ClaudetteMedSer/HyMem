"""Offline recovery dispatcher, Docker-plan, admission, and WAL-backup controls."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'tools/diagnostics/lme_summary_recovery_v1'


def load(name):
    path = BASE / name
    module = types.ModuleType('summary_host_test_' + name.replace('.', '_'))
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('network forbidden during recovery host tests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)


@pytest.mark.parametrize('live', [False, True])
def test_actual_worker_parses_dispatcher_args_and_accepts_only_live_authority(monkeypatch, capsys, live):
    protocol, worker = load('protocol.py'), load('worker.py')
    monkeypatch.setattr(sys, 'argv', protocol.arguments(live))
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setattr(worker.logging, 'disable', lambda _: None)
    frame = worker.encoded(worker.AUTHORIZATION) + b'\n' if live else b'invalid-must-not-be-read'
    stream = io.BytesIO(frame)
    monkeypatch.setattr(sys, 'stdin', types.SimpleNamespace(buffer=stream))
    def stop(path):
        raise worker.DiagnosticFailure('stop_after_argparse_and_authority')
    monkeypatch.setattr(worker, 'regular', stop)
    saved = []
    monkeypatch.setattr(worker, 'save', lambda path, value: saved.append(dict(value)))
    assert worker.main() == 1
    assert saved[0]['error_code'] == 'stop_after_argparse_and_authority'
    assert saved[0]['stock_invocations'] == 0
    assert stream.tell() == (len(frame) if live else 0)
    assert json.loads(capsys.readouterr().out)['error_code'] == 'stop_after_argparse_and_authority'


@dataclass
class Outcome:
    status: str = 'completed'
    returncode: int = 0
    safe_to_continue: bool = True


@pytest.mark.parametrize('fault', [None, 'timeout', 'exit', 'unsafe', 'supervisor', 'postverify'])
def test_live_dispatch_uses_exact_authority_once_and_preserves_supervisor_failure(tmp_path, monkeypatch, fault):
    protocol, worker = load('protocol.py'), load('worker.py')
    clone = tmp_path / 'hymem.sqlite'
    clone.touch()
    monkeypatch.setattr(protocol, 'CLONE', clone)
    monkeypatch.setattr(protocol, 'OUTPUT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['protocol.py', 'live', '--manifest-sha256', 'a' * 64])
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setattr(os, 'umask', lambda _: None)
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(protocol.signal, 'signal', lambda *a: None)
    checks = []
    def verify(pin):
        checks.append(pin)
        if fault == 'postverify' and len(checks) == 2:
            raise RuntimeError('synthetic_identity_change')
        return {}, worker
    monkeypatch.setattr(protocol, 'verify', verify)
    calls = []
    def supervise(argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv == [sys.executable, '-I', '-B', *protocol.arguments(True)]
        worker.read_authority(io.BytesIO(kwargs['stdin_bytes']))
        assert kwargs['stdin_bytes'] == worker.encoded(worker.AUTHORIZATION) + b'\n'
        assert kwargs['timeout_seconds'] == 1860 and kwargs['cleanup_seconds'] == 10
        assert kwargs['output_limit_bytes'] == 16 * 1024 * 1024
        assert kwargs['env'] == protocol.environment()
        if fault == 'supervisor': raise RuntimeError('synthetic_supervisor_fault')
        return Outcome(status='timeout' if fault == 'timeout' else 'completed',
                       returncode=1 if fault == 'exit' else 0,
                       safe_to_continue=fault != 'unsafe')
    monkeypatch.setattr(protocol, 'load', lambda *a: types.SimpleNamespace(supervise_invocation=supervise))
    saved = []
    monkeypatch.setattr(worker, 'save', lambda path, value: saved.append((path, value)))
    assert protocol.main() == (0 if fault is None else 1)
    assert len(calls) == 1 and checks == ['a' * 64, 'a' * 64]
    assert saved[0][0] == tmp_path / 'supervisor.json'
    assert saved[0][1]['package_and_source_unchanged'] is (fault != 'postverify')


def test_preflight_dispatch_clones_once_and_never_constructs_supervisor(tmp_path, monkeypatch):
    protocol = load('protocol.py')
    monkeypatch.setattr(sys, 'argv', ['protocol.py', 'preflight', '--manifest-sha256', 'a' * 64])
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setattr(os, 'umask', lambda _: None)
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    events = []
    def main():
        assert sys.argv == protocol.arguments(False)
        assert '--credential-file' not in sys.argv and '--preflight' in sys.argv
        events.append('worker')
        return 0
    def verify(pin):
        events.append('verify')
        return {}, types.SimpleNamespace(main=main)
    monkeypatch.setattr(protocol, 'verify', verify)
    monkeypatch.setattr(protocol, 'clone_reference', lambda: events.append('clone'))
    monkeypatch.setattr(protocol, 'load', lambda *a: pytest.fail('supervisor imported for preflight'))
    assert protocol.main() == 0 and events == ['verify', 'clone', 'worker', 'verify']


@pytest.fixture
def wal_source(tmp_path):
    root = tmp_path.resolve()
    reference = root / 'reference.sqlite'
    conn = sqlite3.connect(reference, isolation_level=None)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE invented (id INTEGER PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO invented VALUES (1, 'WAL-only invented value')")
    assert Path(str(reference) + '-wal').exists()
    yield reference, conn, root / 'clone.sqlite'
    conn.close()


def test_backup_includes_wal_and_preserves_original(wal_source, monkeypatch):
    reference, writer, clone = wal_source
    protocol = load('protocol.py')
    monkeypatch.setattr(protocol, 'REFERENCE', reference)
    monkeypatch.setattr(protocol, 'CLONE', clone)
    original = reference.read_bytes()
    protocol.clone_reference()
    connection = sqlite3.connect(clone)
    try:
        assert connection.execute('SELECT * FROM invented').fetchall() == [(1, 'WAL-only invented value')]
    finally: connection.close()
    assert reference.read_bytes() == original
    assert writer.execute('SELECT COUNT(*) FROM invented').fetchone()[0] == 1
    saved = clone.read_bytes()
    with pytest.raises(FileExistsError): protocol.clone_reference()
    assert clone.read_bytes() == saved


@pytest.mark.parametrize('fault', ['samefile', 'hardlink', 'clone_symlink', 'reference_symlink'])
def test_clone_freshness_and_samefile_fences(wal_source, monkeypatch, fault):
    reference, writer, clone = wal_source
    protocol = load('protocol.py')
    if fault == 'samefile': clone = reference
    elif fault == 'hardlink': os.link(reference, clone)
    elif fault == 'clone_symlink': clone.symlink_to(reference)
    elif fault == 'reference_symlink':
        alias = reference.parent / 'reference-link.sqlite'
        alias.symlink_to(reference)
        reference = alias
    monkeypatch.setattr(protocol, 'REFERENCE', reference)
    monkeypatch.setattr(protocol, 'CLONE', clone)
    with pytest.raises((FileExistsError, RuntimeError)):
        protocol.clone_reference()
    assert writer.execute('SELECT COUNT(*) FROM invented').fetchone()[0] == 1


def test_backup_deadline_is_enforced(wal_source, monkeypatch):
    reference, writer, clone = wal_source
    protocol = load('protocol.py')
    monkeypatch.setattr(protocol, 'REFERENCE', reference)
    monkeypatch.setattr(protocol, 'CLONE', clone)
    times = iter([0.0, 121.0])
    monkeypatch.setattr(protocol.time, 'monotonic', lambda: next(times))
    with pytest.raises(RuntimeError, match='backup_deadline'):
        protocol.clone_reference()
    assert writer.execute('SELECT COUNT(*) FROM invented').fetchone()[0] == 1


@pytest.mark.parametrize('fault', [None, 'callback_failure', 'deadline'])
def test_large_wal_backup_pins_snapshot_and_rolls_back_even_on_failure(wal_source, monkeypatch, fault):
    reference, writer, clone = wal_source
    # More than four 256-page steps, with the bulk content still in the WAL.
    writer.execute('PRAGMA wal_autocheckpoint=0')
    writer.execute('BEGIN')
    writer.executemany('INSERT INTO invented(value) VALUES (?)', [('x' * 4096,)] * 1100)
    writer.execute('COMMIT')
    assert writer.execute('PRAGMA page_count').fetchone()[0] > 1024
    assert Path(str(reference) + '-wal').stat().st_size > 1024 * 4096
    initial_count = writer.execute('SELECT COUNT(*) FROM invented').fetchone()[0]
    protocol = load('protocol.py')
    monkeypatch.setattr(protocol, 'REFERENCE', reference)
    monkeypatch.setattr(protocol, 'CLONE', clone)
    real_connect = sqlite3.connect
    events, remaining_pages = [], []
    concurrent_write = []

    class TrackedSource(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            normalized = sql.strip().upper()
            if normalized == 'BEGIN': events.append('begin')
            if normalized == 'SELECT COUNT(*) FROM SQLITE_SCHEMA': events.append('pin')
            return super().execute(sql, *args, **kwargs)

        def backup(self, target, *, pages, progress):
            assert self.in_transaction is True
            assert events == ['begin', 'pin']
            assert pages == 256
            events.append('backup')

            def observed(status, remaining, total):
                assert self.in_transaction is True
                remaining_pages.append(remaining)
                if fault == 'callback_failure':
                    raise RuntimeError('synthetic_backup_callback_failure')
                progress(status, remaining, total)
                if not concurrent_write:
                    assert remaining > 0  # Prove this is genuinely multi-step.
                    writer.execute('BEGIN')
                    writer.execute("UPDATE invented SET value='later concurrent value' WHERE id=1")
                    writer.execute("INSERT INTO invented(value) VALUES ('later concurrent row')")
                    writer.execute('COMMIT')
                    concurrent_write.append(True)

            return super().backup(target, pages=pages, progress=observed)

        def rollback(self):
            assert self.in_transaction is True
            events.append('rollback')
            value = super().rollback()
            assert self.in_transaction is False
            return value

        def close(self):
            assert self.in_transaction is False
            events.append('close')
            return super().close()

    def connect(database, *args, **kwargs):
        if kwargs.get('uri'):
            assert database == reference.as_uri() + '?mode=ro'
            kwargs['factory'] = TrackedSource
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(protocol.sqlite3, 'connect', connect)
    if fault == 'deadline':
        times = iter([0.0, 121.0])
        monkeypatch.setattr(protocol.time, 'monotonic', lambda: next(times))
    if fault:
        with pytest.raises(RuntimeError, match='synthetic_backup_callback_failure|backup_deadline'):
            protocol.clone_reference()
        assert writer.execute('SELECT COUNT(*) FROM invented').fetchone()[0] == initial_count
        assert concurrent_write == []
    else:
        protocol.clone_reference()
        assert concurrent_write == [True]
        assert len(remaining_pages) > 4 and remaining_pages[-1] == 0
        assert all(left > right for left, right in zip(remaining_pages, remaining_pages[1:]))
        target = real_connect(clone)
        try:
            assert target.execute('SELECT COUNT(*) FROM invented').fetchone()[0] == initial_count
            assert target.execute('SELECT value FROM invented WHERE id=1').fetchone()[0] == 'WAL-only invented value'
            assert target.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        finally: target.close()
        assert writer.execute('SELECT COUNT(*) FROM invented').fetchone()[0] == initial_count + 1
        assert writer.execute('SELECT value FROM invented WHERE id=1').fetchone()[0] == 'later concurrent value'
    assert events == ['begin', 'pin', 'backup', 'rollback', 'close']


@pytest.mark.parametrize('mode', ['preflight', 'live'])
def test_docker_plan_keeps_entire_reference_readonly_and_limits_writes(mode):
    host = load('host.py')
    plan = host.plan(mode, 'a' * 64)
    assert plan['command'][:2] == ['docker', 'create']
    assert plan['network'] == ('bridge' if mode == 'live' else 'none')
    assert plan['mounts']['/reference'] == (host.REFERENCE, False)
    assert plan['mounts']['/work'] == (host.ROOT + '/work', True)
    assert {target for target, (_, rw) in plan['mounts'].items() if rw} == {'/work', '/results'}
    assert ('/run/deepseek.env' in plan['mounts']) is (mode == 'live')
    assert '/var/run/docker.sock' not in plan['mounts']
    assert not set(plan['command']) & {'--privileged', '--publish', '--cap-add', '--device'}


def inspect_fixture(plan):
    args = plan['command'][plan['command'].index(plan['image']) + 1:]
    return {'Image': plan['image'], 'Name': '/' + plan['name'],
            'Config': {'Image': plan['image'], 'User': '1000:1000', 'WorkingDir': '/candidate',
                       'Entrypoint': ['/home/node/hymem-env/bin/python3'], 'Cmd': args, 'Env': ['PATH=/usr/bin']},
            'HostConfig': {'NetworkMode': plan['network'], 'ReadonlyRootfs': True, 'Privileged': False,
                           'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges'], 'Init': True,
                           'PidsLimit': 128, 'Memory': 2147483648, 'NanoCpus': 2000000000,
                           'Tmpfs': {'/tmp': 'rw,noexec,nosuid,size=64m'}, 'RestartPolicy': {'Name': 'no'}},
            'Mounts': [{'Destination': dst, 'Source': src, 'RW': rw, 'Type': 'bind'}
                       for dst, (src, rw) in plan['mounts'].items()],
            'State': {'Status': 'created', 'ExitCode': 0, 'OOMKilled': False, 'Pid': 0},
            'Args': args, 'Path': '/home/node/hymem-env/bin/python3'}


@pytest.mark.parametrize('fault', [None, 'reference_rw', 'credential_rw', 'network', 'privileged',
                                  'environment', 'image', 'cmd', 'restart', 'production_mount'])
def test_inspect_rejects_docker_config_drift(monkeypatch, fault):
    host = load('host.py')
    plan = host.plan('live', 'a' * 64)
    obj = inspect_fixture(plan)
    if fault in ('reference_rw', 'credential_rw'):
        destination = '/reference' if fault == 'reference_rw' else '/run/deepseek.env'
        next(row for row in obj['Mounts'] if row['Destination'] == destination)['RW'] = True
    elif fault == 'network': obj['HostConfig']['NetworkMode'] = 'host'
    elif fault == 'privileged': obj['HostConfig']['Privileged'] = True
    elif fault == 'environment': obj['Config']['Env'].append('DEEPSEEK_API_KEY=invented')
    elif fault == 'image': obj['Image'] = 'sha256:' + '0' * 64
    elif fault == 'cmd': obj['Config']['Cmd'] = ['unapproved']
    elif fault == 'restart': obj['HostConfig']['RestartPolicy']['Name'] = 'always'
    elif fault == 'production_mount': obj['Mounts'].append({'Destination': '/extra', 'Source': '/production', 'RW': False, 'Type': 'bind'})
    monkeypatch.setattr(host.subprocess, 'check_output', lambda *a, **kw: json.dumps([obj]))
    if fault:
        with pytest.raises(AssertionError): host.inspect('b' * 64, plan, 'created')
    else: assert host.inspect('b' * 64, plan, 'created')['configuration_verified'] is True


def preflight_receipt(host):
    worker = load('worker.py')
    r5 = json.loads((ROOT / 'docs/patches/2026-09-24-lme-independent-summary-indexing-r5-manifest.json').read_text())
    return {'schema': 'summary-recovery-live-diagnostic-v1', 'status': 'preflight_passed',
            'r5_manifest_sha256': host.R5_PIN, 'model': 'deepseek-flash', 'endpoint': 'https://api.deepseek.com',
            'bounds': worker.BOUNDS, 'max_http_attempts': 300, 'stock_invocations': 0, 'provider_calls': 0,
            'client_closed': True, 'reference_store_unchanged_verified': True, 'source_unchanged_verified': True,
            'source_inventory_sha256': worker.digest(r5['source_sha256']),
            'producer_identity_sha256': 'sha256:' + 'a' * 64,
            'clone_before_sha256': 'a' * 64, 'clone_after_sha256': 'a' * 64,
            'all_non_summary_state_unchanged': True, 'original_store_modified': False,
            'benchmark_rerun': False, 'full_500_readiness_verified': False, 'semantic_quality_guaranteed': False,
            'health_before': {'summary_degraded_sessions': 10, 'malformed_summaries': 0,
                              'summary_missing_sessions': 3, 'summary_healthy': False},
            'usage': {**{field: 0 for field in ('calls', 'request_attempts', 'successful_responses')},
                      **{field + '_available': True for field in ('calls', 'request_attempts', 'successful_responses')}}}


@pytest.mark.parametrize('fault', [None, 'schema', 'model', 'endpoint', 'bounds', 'attempt_cap',
                                  'call_bool', 'provider_work', 'clone_changed', 'clone_hash',
                                  'non_summary', 'health', 'source_hash', 'closed', 'calls'])
def test_live_gate_rejects_contradictory_preflight_proof(monkeypatch, fault):
    host = load('host.py')
    receipt = preflight_receipt(host)
    if fault == 'schema': receipt['schema'] = 'unrelated'
    elif fault == 'model': receipt['model'] = 'unapproved'
    elif fault == 'endpoint': receipt['endpoint'] = 'https://example.invalid'
    elif fault == 'bounds': receipt['bounds'] = {**receipt['bounds'], 'max_calls': 101}
    elif fault == 'attempt_cap': receipt['max_http_attempts'] = 301
    elif fault == 'call_bool': receipt['stock_invocations'] = False
    elif fault == 'provider_work': receipt['provider_calls'] = 1
    elif fault == 'clone_changed': receipt['clone_after_sha256'] = 'b' * 64
    elif fault == 'clone_hash': receipt['clone_before_sha256'] = receipt['clone_after_sha256'] = 'invalid'
    elif fault == 'non_summary': receipt['all_non_summary_state_unchanged'] = False
    elif fault == 'health': receipt['health_before']['summary_healthy'] = True
    elif fault == 'source_hash': receipt['source_inventory_sha256'] = 'b' * 64
    elif fault == 'closed': receipt['client_closed'] = False
    elif fault == 'calls': receipt['usage']['calls'] = 1
    r5 = (ROOT / 'docs/patches/2026-09-24-lme-independent-summary-indexing-r5-manifest.json').read_bytes()
    def read(path):
        if path.name == 'preflight-container-id.json': return json.dumps({'container_id': 'b' * 64}).encode()
        if path.name == 'preflight.json': return json.dumps(receipt).encode()
        if path.name == 'manifest.json': return r5
        raise AssertionError('unexpected read')
    monkeypatch.setattr(host, 'PIN', 'a' * 64)
    monkeypatch.setattr(host, 'read', read)
    monkeypatch.setattr(host, 'inspect', lambda *a: {'exit_code': 0, 'oom_killed': False, 'pid': 0})
    original_lstat, original_resolve = Path.lstat, Path.resolve
    def lstat(path):
        if str(path) == host.KEY: return types.SimpleNamespace(st_mode=0o100600, st_uid=1000)
        return original_lstat(path)
    def resolve(path, *a, **kw):
        return path if str(path) == host.KEY else original_resolve(path, *a, **kw)
    monkeypatch.setattr(Path, 'lstat', lstat)
    monkeypatch.setattr(Path, 'resolve', resolve)
    if fault:
        with pytest.raises(AssertionError): host.live_gates()
    else: host.live_gates()


@pytest.fixture
def package_fixture(tmp_path, monkeypatch):
    protocol = load('protocol.py')
    root = tmp_path.resolve()
    package, source = root / 'bundle', root / 'candidate'
    package.mkdir()
    source.mkdir()
    raw = (ROOT / 'docs/patches/2026-09-24-lme-independent-summary-indexing-r5-manifest.json').read_bytes()
    r5 = root / 'r5-manifest.json'
    r5.write_bytes(raw)
    source_map = json.loads(raw)['source_sha256']
    frozen = Path(os.environ.get('HYMEM_Q1_VERIFIER_SOURCE', str(ROOT))).resolve()
    for relative, pin in source_map.items():
        content = (frozen / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == pin
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    pins = {}
    for name in ('worker.py', 'protocol.py', 'supervised_invocation.py'):
        content = (BASE / name).read_bytes()
        (package / name).write_bytes(content)
        pins[name] = hashlib.sha256(content).hexdigest()
    manifest = {'schema': 'isolated-summary-recovery-package-v1',
                'source_manifest_sha256': protocol.R5_PIN, 'helper_sha256': pins}
    manifest_raw = json.dumps(manifest).encode()
    (package / 'manifest.json').write_bytes(manifest_raw)
    monkeypatch.setattr(protocol, 'PACKAGE', package)
    monkeypatch.setattr(protocol, 'SOURCE', source)
    monkeypatch.setattr(protocol, 'R5', r5)
    return protocol, package, source, r5, hashlib.sha256(manifest_raw).hexdigest()


@pytest.mark.parametrize('fault', [None, 'manifest_pin', 'helper_hash', 'helper_symlink',
                                  'extra_package', 'source_drift', 'extra_source', 'r5_pin'])
def test_protocol_exact_package_and_real_r5_source_before_any_dispatch(package_fixture, fault):
    protocol, package, source, r5, pin = package_fixture
    if fault == 'manifest_pin': pin = '0' * 64
    elif fault == 'helper_hash':
        path = package / 'worker.py'
        path.write_bytes(path.read_bytes() + b'\n# synthetic drift\n')
    elif fault == 'helper_symlink':
        path = package / 'worker.py'
        target = package.parent / 'original-worker.py'
        path.rename(target)
        path.symlink_to(target)
    elif fault == 'extra_package': (package / 'unlisted.txt').write_text('invented')
    elif fault == 'source_drift':
        path = source / 'hymem/__init__.py'
        path.write_bytes(path.read_bytes() + b'\n# synthetic drift\n')
    elif fault == 'extra_source': (source / 'unlisted.py').write_text('# invented')
    elif fault == 'r5_pin': r5.write_bytes(b'{}')
    if fault:
        with pytest.raises(RuntimeError): protocol.verify(pin)
    else:
        manifest, worker = protocol.verify(pin)
        assert manifest['source_manifest_sha256'] == protocol.R5_PIN
        assert worker.BOUNDS['max_calls'] == 100 and worker.BOUNDS['timeout_seconds'] == 1800
        worker.read_authority(io.BytesIO(worker.encoded(worker.AUTHORIZATION) + b'\n'))
