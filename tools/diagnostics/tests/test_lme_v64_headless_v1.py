"""Offline launcher controls, no Docker or provider calls."""
import hashlib
import json
import os
from pathlib import Path
import types
import pytest

BASE = Path(__file__).resolve().parents[1] / 'lme_v64_headless_v1'

def load(name):
    path = BASE / name
    mod = types.ModuleType(name.replace('/', '_'))
    mod.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    return mod

def test_candidate_protocol_and_support_pins():
    run = load('bundle/q1_stock_run.py')
    p = BASE.parent / 'lme_r7_headless/bundle/q1_stock_run.py'
    old = types.ModuleType('old')
    exec(compile(p.read_bytes(), str(p), 'exec'), old.__dict__)
    manifest = json.loads((BASE.parents[2] / 'docs/patches/2026-09-26-episode-shadow-manifest.json').read_bytes())
    assert len(manifest) == 481
    assert hashlib.sha256(run.canonical(manifest)).hexdigest() == run.SOURCE_MANIFEST_SHA
    assert run.stock_arguments() == old.stock_arguments()
    assert run.SOURCE_INDICES == old.SOURCE_INDICES
    assert run.CANARY_VERSION == old.CANARY_VERSION
    assert run.SPLIT_VERSION == old.SPLIT_VERSION
    assert run.TIMEOUT == 32400
    for name, pin in load('prepare.py').SUPPORT.items():
        assert hashlib.sha256((BASE / 'bundle' / name).read_bytes()).hexdigest() == pin

def test_pure_plans():
    control, host = load('host_control.py'), load('bundle/q1_stock_host.py')
    for live in (False, True):
        plan = host.command(str(control.ROOT), str(control.SOURCE), 'a' * 64, live=live)
        assert plan['starts_work'] is False
        assert plan['network'] == ('bridge' if live else 'none')
        assert ('/run/deepseek.env' in plan['mounts']) is live
        assert plan['mounts']['/candidate'][1] is False
        assert plan['mounts']['/home/node/hymem-env'][1] is False
        assert '/var/run/docker.sock' not in plan['mounts']
        assert 'r7-sample8' not in plan['name']
    plan = host.validation_command(str(control.ROOT), str(control.SOURCE), 'a' * 64)
    assert plan['network'] == 'none' and plan['new_provider_calls'] == 0
    assert all(not writable for _, writable in plan['mounts'].values())

@pytest.mark.parametrize('reviewed', [False, True])
def test_gate_rejected_before_docker(monkeypatch, reviewed):
    control = load('host_control.py')
    raw = json.dumps({'schema': 'lme-v64-root-reviewed-deployed-gate-v1', 'root_reviewed': reviewed, 'status': 'failed'}).encode()
    monkeypatch.setattr(control, 'read', lambda path: raw)
    monkeypatch.setattr(control, 'GATE_PIN', hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(control, 'command', lambda argv: pytest.fail('Docker forbidden'))
    with pytest.raises(AssertionError):
        control.live_gates({}, None)

def test_approved_gate_reaches_separate_preflight_check(monkeypatch):
    control = load('host_control.py')
    gate = {'schema': 'lme-v64-root-reviewed-deployed-gate-v1',
            'root_reviewed': True, 'status': 'passed',
            'source_manifest_sha256': control.R7_PIN, 'schema_version': 64,
            'preparation_manifest_sha256': 'a' * 64,
            'runtime_seal_sha256': 'b' * 64,
            'evidence_sha256': {name: 'c' * 64 for name in
                ('suite', 'paid_postflight', 'migration', 'deployment', 'postdeploy')}}
    for key in ('complete_suite_passed', 'bounded_paid_dream_postflight_passed',
                'migration_rehearsal_passed', 'deployed_source_verified',
                'postdeploy_checks_passed', 'production_environment_preserved'):
        gate[key] = True
    raw = json.dumps(gate).encode()
    monkeypatch.setattr(control, 'PIN', 'a' * 64)
    monkeypatch.setattr(control, 'GATE_PIN', hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(control, 'read', lambda path: raw)
    def preflight_read(path):
        assert path.name == 'preflight-container-id.json'
        raise RuntimeError('independent_preflight_required')
    monkeypatch.setattr(control, 'js', preflight_read)
    with pytest.raises(RuntimeError, match='independent_preflight_required'):
        control.live_gates({'runtime_seal_sha256': 'b' * 64}, None)

def test_finalizer_forwards_preparation_pin(monkeypatch):
    path = BASE.parent / 'lme_v64_headless_finalizer_v1.py'
    mod = types.ModuleType('new_finalizer')
    mod.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    monkeypatch.setattr(mod, 'PIN', 'a' * 64)
    assert mod.HOST_SHA == hashlib.sha256((BASE / 'host_control.py').read_bytes()).hexdigest()
    def run(argv, **kwargs):
        assert argv[-2:] == ['--manifest-sha256', 'a' * 64]
        return types.SimpleNamespace(returncode=0, stdout=b'{"configuration_verified":true}')
    monkeypatch.setattr(mod.subprocess, 'run', run)
    assert mod.controller('create', 'validation')['configuration_verified'] is True

def test_exact_runtime_inventory_records_hidden_links_modes_and_ownership(tmp_path):
    control = load('host_control.py')
    file = tmp_path / '.hidden-sdk'
    file.write_bytes(b'inert test runtime')
    file.chmod(0o400)
    (tmp_path / 'lib').mkdir()
    (tmp_path / 'python').symlink_to('/usr/local/bin/python3')
    inventory = control.runtime_inventory(tmp_path)
    control.checked_runtime_entries(inventory)
    assert set(inventory) == {'.hidden-sdk', 'lib', 'python'}
    assert inventory['python']['link'] == '/usr/local/bin/python3'
    assert inventory['.hidden-sdk']['mode'] == 0o400
    assert inventory['.hidden-sdk']['uid'] == file.stat().st_uid
    file.chmod(0o600)
    assert control.runtime_inventory(tmp_path) != inventory
    (tmp_path / 'added').write_bytes(b'new')
    assert set(control.runtime_inventory(tmp_path)) != set(inventory)

def test_runtime_special_file_rejected(tmp_path):
    control = load('host_control.py')
    os.mkfifo(tmp_path / 'fifo')
    with pytest.raises(AssertionError):
        control.runtime_inventory(tmp_path)

def test_runtime_inventory_allows_atime_only_drift(monkeypatch, tmp_path):
    control = load('host_control.py')
    file = tmp_path / 'sdk'
    file.write_bytes(b'inert runtime')
    before = file.stat()
    real_fstat = os.fstat
    calls = 0
    def access_time_only(fd):
        nonlocal calls
        calls += 1
        info = real_fstat(fd)
        # Synthetic atime change is deterministic even on noatime filesystems;
        # all identity, content and permission fields remain from real fstat.
        values = {field: getattr(info, field) for field in dir(info) if field.startswith('st_')}
        values['st_atime'] += calls
        values['st_atime_ns'] += calls * 1_000_000_000
        return types.SimpleNamespace(**values)
    monkeypatch.setattr(control.os, 'fstat', access_time_only)
    real_lstat = Path.lstat
    snapshots = 0
    def path_access_time_only(path, *args, **kwargs):
        nonlocal snapshots
        info = real_lstat(path, *args, **kwargs)
        if path != file:
            return info
        snapshots += 1
        values = {field: getattr(info, field) for field in dir(info) if field.startswith('st_')}
        values['st_atime'] += snapshots
        values['st_atime_ns'] += snapshots * 1_000_000_000
        return types.SimpleNamespace(**values)
    monkeypatch.setattr(Path, 'lstat', path_access_time_only)
    inventory = control.runtime_inventory(tmp_path)
    assert calls == 2
    assert inventory['sdk']['sha256'] == hashlib.sha256(b'inert runtime').hexdigest()
    assert file.stat().st_mtime_ns == before.st_mtime_ns

@pytest.mark.parametrize('field', ['st_dev', 'st_ino', 'st_mode', 'st_nlink',
    'st_uid', 'st_gid', 'st_size', 'st_mtime_ns', 'st_ctime_ns'])
def test_runtime_inventory_rejects_every_stable_field_drift(monkeypatch, tmp_path, field):
    control = load('host_control.py')
    (tmp_path / 'sdk').write_bytes(b'inert runtime')
    real_fstat = os.fstat
    calls = 0
    def drift(fd):
        nonlocal calls
        calls += 1
        info = real_fstat(fd)
        if calls == 1:
            return info
        values = {key: getattr(info, key) for key in dir(info) if key.startswith('st_')}
        values[field] += 1
        return types.SimpleNamespace(**values)
    monkeypatch.setattr(control.os, 'fstat', drift)
    with pytest.raises(AssertionError):
        control.runtime_inventory(tmp_path)

@pytest.mark.parametrize('mutation', ['content', 'same_size', 'replacement', 'symlink', 'mode'])
def test_runtime_inventory_rejects_drift_during_read(monkeypatch, tmp_path, mutation):
    control = load('host_control.py')
    file = tmp_path / 'sdk'
    file.write_bytes(b'original')
    initial = file.stat()
    real_read = os.read
    changed = False
    def mutate_after_read(fd, size):
        nonlocal changed
        chunk = real_read(fd, size)
        if not changed:
            changed = True
            if mutation in ('content', 'same_size'):
                file.write_bytes(b'changed!' if mutation == 'same_size' else b'changed content')
                # Even restoring mtime cannot conceal the write's ctime.
                os.utime(file, ns=(initial.st_atime_ns, initial.st_mtime_ns))
            elif mutation == 'mode':
                file.chmod(0o400)
            else:
                file.unlink()
                if mutation == 'replacement':
                    file.write_bytes(b'original')
                else:
                    file.symlink_to('/etc/passwd')
        return chunk
    monkeypatch.setattr(control.os, 'read', mutate_after_read)
    with pytest.raises(AssertionError):
        control.runtime_inventory(tmp_path)

@pytest.mark.parametrize('replacement', ['symlink', 'fifo'])
def test_runtime_inventory_rejects_nonregular_before_open(monkeypatch, tmp_path, replacement):
    control = load('host_control.py')
    file = tmp_path / 'sdk'
    file.write_bytes(b'original')
    real_open = os.open
    def replace_before_open(path, flags, *args, **kwargs):
        if Path(path) == file:
            file.unlink()
            if replacement == 'symlink':
                file.symlink_to('/etc/passwd')
            else:
                assert flags & os.O_NONBLOCK
                os.mkfifo(file)
        return real_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(control.os, 'open', replace_before_open)
    with pytest.raises(OSError if replacement == 'symlink' else AssertionError):
        control.runtime_inventory(tmp_path)

def test_sealed_testonly_package_source_and_archive(monkeypatch):
    root = os.environ.get('HYMEM_V64_TEST_PREPARATION')
    source = os.environ.get('HYMEM_V64_TEST_EXACT_SOURCE')
    if not root or not source:
        pytest.skip('end-to-end preparation paths supplied by explicit offline check')
    import tarfile
    run = load('bundle/q1_stock_run.py')
    package = Path(root) / 'bundle'
    seal = json.loads(Path(root + '-seal.json').read_bytes())
    monkeypatch.setattr(run, 'PACKAGE', package)
    monkeypatch.setattr(run, 'SOURCE', Path(source))
    manifest = run.validate_package(seal['manifest_sha256'])
    run.verify_source(manifest)
    assert manifest['runtime_seal']['entries'].keys() == {'TESTONLY_NOT_A_REAL_RUNTIME'}
    assert manifest['question_ids'] == run.QUESTION_IDS
    raw = Path(seal['archive']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == seal['archive_sha256']
    with tarfile.open(seal['archive']) as archive:
        assert {item.name for item in archive.getmembers()} == set(seal['files_sha256'])
        for item in archive.getmembers():
            assert item.isfile()
            assert hashlib.sha256(archive.extractfile(item).read()).hexdigest() == seal['files_sha256'][item.name]

def test_finalizer_rejects_completed_one_shot_before_spawn(monkeypatch, tmp_path):
    path = BASE.parent / 'lme_v64_headless_finalizer_v1.py'
    mod = types.ModuleType('completed_finalizer')
    mod.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'checked_remote', lambda: None)
    monkeypatch.setattr(mod, 'owned_live', lambda cid: None)
    monkeypatch.setattr(mod, 'controller', lambda *args: {'status': 'exited'})
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('spawn forbidden'))
    (tmp_path / 'finalizer-result.json').write_bytes(b'{}')
    with pytest.raises(RuntimeError, match='finalizer_already_finished'):
        mod.remote_launch('a' * 64)

def test_finalizer_duplicate_worker_claim_blocks_all_docker(monkeypatch, tmp_path):
    path = BASE.parent / 'lme_v64_headless_finalizer_v1.py'
    mod = types.ModuleType('duplicate_worker')
    mod.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'checked_remote', lambda: None)
    monkeypatch.setattr(mod, 'owned_live', lambda cid: None)
    monkeypatch.setattr(mod, 'docker_wait', lambda *args: pytest.fail('Docker forbidden'))
    (tmp_path / 'finalizer-worker-claim.json').write_bytes(b'{}')
    with pytest.raises(FileExistsError):
        mod.worker('a' * 64)

@pytest.mark.parametrize('size,accepted', [(1024 * 1024 + 1, True), (32 * 1024 * 1024 + 1, False)])
def test_finalizer_manifest_specific_size_limit(monkeypatch, tmp_path, size, accepted):
    path = BASE.parent / 'lme_v64_headless_finalizer_v1.py'
    mod = types.ModuleType('manifest_limits')
    mod.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    source = tmp_path / 'finalizer.py'
    source.write_bytes(b'inert finalizer fixture')
    host = tmp_path / 'controller.py'
    host.write_bytes(b'inert controller fixture')
    (tmp_path / 'bundle').mkdir()
    manifest = tmp_path / 'bundle/manifest.json'
    manifest.write_bytes(b' ' * size)
    (tmp_path / 'finalizer-install.json').write_text(json.dumps({
        'controller_sha256': hashlib.sha256(host.read_bytes()).hexdigest(),
        'finalizer_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}))
    monkeypatch.setattr(mod, '__file__', str(source))
    monkeypatch.setattr(mod, 'SELF', source)
    monkeypatch.setattr(mod, 'HOST', host)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'HOST_SHA', hashlib.sha256(host.read_bytes()).hexdigest())
    monkeypatch.setattr(mod, 'PIN', hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(mod.os, 'geteuid', lambda: 1000)
    if accepted:
        mod.checked_remote()
        with pytest.raises(RuntimeError, match='invalid_finalizer_input'):
            mod.read(manifest)  # Default limit remains1MiB for every other file.
    else:
        with pytest.raises(RuntimeError, match='invalid_finalizer_input'):
            mod.checked_remote()

@pytest.mark.parametrize('changed', [False, True])
def test_remote_launch_verifies_bytes_before_execution(monkeypatch, tmp_path, changed):
    import shlex
    import subprocess
    import sys
    path = BASE.parent / 'lme_v64_headless_finalizer_v1.py'
    mod = types.ModuleType('prehash_launch')
    mod.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    local = tmp_path / 'reviewed.py'
    remote = tmp_path / 'remote.py'
    reviewed = b'print("verified_fixture_executed")\n'
    local.write_bytes(reviewed)
    remote.write_bytes(b'print("unreviewed_fixture_executed")\n' if changed else reviewed)
    monkeypatch.setattr(mod, '__file__', str(local))
    monkeypatch.setattr(mod, 'SELF', remote)
    monkeypatch.setattr(mod, 'PIN', 'a' * 64)
    def capture(command, **kwargs):
        argv = shlex.split(command)
        assert argv[:4] == ['python3', '-I', '-B', '-c']
        result = subprocess.run([sys.executable, '-I', '-B', '-c', argv[4]], capture_output=True)
        if changed:
            assert result.returncode != 0
            assert b'unreviewed_fixture_executed' not in result.stdout
        else:
            assert result.returncode == 0 and result.stdout == b'verified_fixture_executed\n'
        return {'offline_fixture': True}
    monkeypatch.setattr(mod, 'ssh_run', capture)
    assert mod.local_launch('b' * 64) == {'offline_fixture': True}
