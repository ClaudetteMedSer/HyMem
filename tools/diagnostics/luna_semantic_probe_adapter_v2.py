"""Source-bound v2 startup adapter for a fresh semantic diagnostic attempt.

The accepted v1 bundle and receipt remain byte-for-byte unchanged. A separate
sealed sidecar pins this adapter before a one-shot dispatch. The user bus is
available only to systemctl control-plane queries, never to inference children.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

UID = 1000
RUNTIME = Path('/run/user/1000')
BUS = RUNTIME / 'bus'
SIDECAR = 'adapter-receipt-v2.json'
ADAPTER = 'adapter-run-v2.py'
HOST_SHA = '7e190561958b126593a3e652ec2de35fb4cef27d4e5863284a75636abe531b07'
RUN_SHA = '60fc7fca900ef8320a410af2f5f01415f973f01907cf7c0456c0b4936611030a'
READER_SHA = '68f02223a561dd31a1a0417d8972ddafab7cbd26fc15500a9a2efc1b0cb9d355'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(path: Path, name: str, expected_sha: str):
    if not path.is_file() or path.is_symlink() or digest(path) != expected_sha:
        raise ValueError('accepted_source_pin_invalid')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('adapter_import_invalid')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bus_environment() -> dict[str, str]:
    """Resolve only the fixed, owned systemd user bus endpoint."""
    if os.getuid() != UID:
        raise ValueError('trusted_user_bus_invalid')
    directory = RUNTIME.lstat()
    endpoint = BUS.lstat()
    if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != UID
            or directory.st_mode & 0o077 or not stat.S_ISSOCK(endpoint.st_mode)
            or endpoint.st_uid != UID):
        raise ValueError('trusted_user_bus_invalid')
    return {'HOME': '/home/atta', 'PATH': '/usr/local/bin:/usr/bin:/bin',
            'XDG_RUNTIME_DIR': str(RUNTIME),
            'DBUS_SESSION_BUS_ADDRESS': 'unix:path=' + str(BUS)}


def seal(root: Path, source: Path, host, mode: str) -> dict:
    if mode not in {'smoke', 'inference'}:
        raise ValueError('adapter_mode_invalid')
    if not host.root_valid(root) or (root / SIDECAR).exists() or (root / ADAPTER).exists():
        raise ValueError('adapter_root_invalid')
    receipt = root / 'launch-receipt.json'
    if not host.regular(receipt):
        raise ValueError('base_receipt_missing')
    fd = os.open(root / ADAPTER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    sidecar = {'schema': 'luna-semantic-probe-adapter-v2',
               'base_receipt_sha256': digest(receipt),
               'adapter_sha256': digest(root / ADAPTER),
               'entry': ADAPTER, 'entry_action': mode,
               'bus_scope': 'systemctl-only',
               'startup_failure_schema': 'luna-semantic-probe-startup-failure-v2'}
    host.write_once(root / SIDECAR, sidecar)
    sidecar['sidecar_sha256'] = digest(root / SIDECAR)
    return sidecar


def verify(root: Path, receipt_sha: str, adapter_sha: str,
           mode: str | None, sidecar_sha: str) -> dict:
    if (type(receipt_sha) is not str or type(adapter_sha) is not str or
            type(sidecar_sha) is not str or len(receipt_sha) != 64 or
            len(adapter_sha) != 64 or len(sidecar_sha) != 64):
        raise ValueError('adapter_pin_invalid')
    sidecar_path = root / SIDECAR
    entry_path = root / ADAPTER
    if not (sidecar_path.is_file() and not sidecar_path.is_symlink()
            and entry_path.is_file() and not entry_path.is_symlink()):
        raise ValueError('adapter_files_invalid')
    sidecar = json.loads(sidecar_path.read_bytes())
    if mode is None:
        mode = sidecar.get('entry_action')
    if mode not in {'smoke', 'inference'}:
        raise ValueError('adapter_mode_invalid')
    expected = {'schema': 'luna-semantic-probe-adapter-v2',
                'base_receipt_sha256': receipt_sha,
                'adapter_sha256': digest(entry_path), 'entry': ADAPTER,
                'entry_action': mode,
                'bus_scope': 'systemctl-only',
                'startup_failure_schema': 'luna-semantic-probe-startup-failure-v2'}
    if (type(sidecar) is not dict or
            json.dumps(sidecar, sort_keys=True, separators=(',', ':')) !=
            json.dumps(expected, sort_keys=True, separators=(',', ':')) or
            digest(root / 'launch-receipt.json') != receipt_sha or
            expected['adapter_sha256'] != adapter_sha or
            digest(sidecar_path) != sidecar_sha):
        raise ValueError('adapter_pin_invalid')
    return expected


def contained(root: Path, receipt: dict, run) -> None:
    """Reuse every v1 policy/kernel check with a scoped control-plane runner."""
    original_module = run.subprocess
    original = original_module.run
    def scoped(args, **kwargs):
        if (type(args) is not list or args[:3] != ['/usr/bin/systemctl', '--user', 'show']
                or len(args) != 6 or args[3] != receipt['unit']
                or not args[4].startswith('--property=') or args[5] != '--no-pager'
                or 'env' in kwargs):
            raise ValueError('containment_command_invalid')
        return original(args, env=bus_environment(), **kwargs)
    run.subprocess = SimpleNamespace(run=scoped)
    try:
        run.verify_live_containment(root, receipt)
    finally:
        run.subprocess = original_module


def entry(root: Path, receipt_sha: str, mode: str, adapter_sha: str,
          sidecar_sha: str) -> int:
    # The v1 entry validates its own receipt and full source inventory first.
    verified = False
    try:
        verify(root, receipt_sha, adapter_sha, mode, sidecar_sha)
        verified = True
        run = load_source(root / 'code/tools/diagnostics/luna_semantic_probe_run.py',
                          'sealed_semantic_entry_v1', RUN_SHA)
        if mode == 'smoke':
            receipt, _, _, _ = run.preflight(root, receipt_sha)
            contained(root, receipt, run)
            host = load_source(root / 'code/tools/diagnostics/luna_semantic_probe_host.py',
                               'sealed_semantic_host_v1_smoke', HOST_SHA)
            host.write_once(root / 'safe-terminal.json', {
                'schema': 'luna-semantic-probe-containment-smoke-v2',
                'receipt_sha256': receipt_sha,
                'adapter_sha256': digest(Path(__file__)),
                'policy_verified': True, 'model_calls': 0,
                'paid_turns': 0, 'known_tokens': 0})
            return 0
        result = run.execute(root, receipt_sha,
                             containment=lambda r, p: contained(r, p, run))
        return 0 if result['core_completed'] else 1
    except BaseException:
        # The v1 execution order reaches inference only after creating run/.
        # This terminal is valid solely when no run directory was ever created.
        if (verified and not (root / 'run').exists() and
                (root / 'launch-attempt.json').is_file()
                and not (root / 'safe-terminal.json').exists()):
            host = load_source(root / 'code/tools/diagnostics/luna_semantic_probe_host.py',
                               'sealed_semantic_host_v1', HOST_SHA)
            if host.root_valid(root):
                host.write_once(root / 'safe-terminal.json', {
                    'schema': 'luna-semantic-probe-startup-failure-v2',
                    'receipt_sha256': receipt_sha,
                    'adapter_sha256': digest(Path(__file__)),
                    'zero_admission_proof': 'run_directory_never_created',
                    'paid_turns': 0, 'known_tokens': 0, 'model_calls': 0,
                    'completed_and_clean': False,
                    'stop_code': 'pre_inference_startup_failure'})
        return 1


def launch(root: Path, receipt_sha: str, adapter_sha: str, sidecar_sha: str,
           host, mode: str) -> dict:
    verify(root, receipt_sha, adapter_sha, mode, sidecar_sha)
    def dispatch(command, **kwargs):
        old = str(root / 'code/tools/diagnostics/luna_semantic_probe_run.py')
        if command.count(old) != 1:
            raise ValueError('base_command_invalid')
        adapted = [str(root / ADAPTER) if item == old else item for item in command]
        index = adapted.index(str(root / ADAPTER))
        adapted.insert(index + 1, mode)
        adapted.extend(['--adapter-sha256', adapter_sha,
                        '--adapter-receipt-sha256', sidecar_sha])
        return subprocess.run(adapted, **kwargs)
    return host.launch(root, receipt_sha, dispatch=dispatch)


def observe(root: Path, receipt_sha: str, adapter_sha: str,
            sidecar_sha: str, progress) -> dict:
    sidecar = verify(root, receipt_sha, adapter_sha, None, sidecar_sha)
    host = load_source(root / 'code/tools/diagnostics/luna_semantic_probe_host.py',
                       'sealed_semantic_host_for_reader_v1', HOST_SHA)
    receipt = json.loads((root / 'launch-receipt.json').read_bytes())
    if (not host.root_valid(root) or not host.strict_equal(
            receipt, host.receipt_for(root, receipt['source_sha256'],
                                      receipt['binary_sha256']))):
        raise ValueError('base_receipt_invalid')
    host.verify_bundle(root, receipt, require_empty_workdirs=False)
    marker = root / 'launch-attempt.json'
    admission = root / 'launch-admission.json'
    expected_admission = {'receipt_sha256': receipt_sha,
        'old_luna_stopped': True, 'old_deepseek_stopped': True,
        'memory_floor_bytes': 6 * 1024**3, 'disk_floor_bytes': 20 * 1024**3,
        'memory_floor_met': True, 'disk_floor_met': True}
    if (not host.regular(marker) or not host.strict_equal(
            json.loads(marker.read_bytes()),
            {'receipt_sha256': receipt_sha, 'one_shot': True}) or
            not host.regular(admission) or not host.strict_equal(
            json.loads(admission.read_bytes()), expected_admission)):
        raise ValueError('launch_admission_invalid')
    original_module = progress.subprocess
    original = original_module.run
    def scoped(args, **kwargs):
        if type(args) is not list or args[:2] != ['/usr/bin/systemctl', '--user']:
            raise ValueError('observer_command_invalid')
        return original(args, env=bus_environment(), **kwargs)
    progress.subprocess = SimpleNamespace(run=scoped)
    try:
        unit = progress._systemd(receipt['unit'], receipt['expected_cgroup'])
    finally:
        progress.subprocess = original_module
    terminal_path = root / 'safe-terminal.json'
    startup = None
    if terminal_path.is_file() and not terminal_path.is_symlink():
        startup = json.loads(terminal_path.read_bytes())
    expected = {'schema': 'luna-semantic-probe-startup-failure-v2',
                'receipt_sha256': receipt_sha, 'adapter_sha256': adapter_sha,
                'zero_admission_proof': 'run_directory_never_created',
                'paid_turns': 0, 'known_tokens': 0, 'model_calls': 0,
                'completed_and_clean': False,
                'stop_code': 'pre_inference_startup_failure'}
    valid = (type(startup) is dict and
             json.dumps(startup, sort_keys=True, separators=(',', ':')) ==
             json.dumps(expected, sort_keys=True, separators=(',', ':')) and
             not (root / 'run').exists()
             and (root / 'launch-attempt.json').is_file())
    smoke = {'schema': 'luna-semantic-probe-containment-smoke-v2',
             'receipt_sha256': receipt_sha, 'adapter_sha256': adapter_sha,
             'policy_verified': True, 'model_calls': 0,
             'paid_turns': 0, 'known_tokens': 0}
    failed_cleanup = (unit.get('available') is True and unit.get('active_state') == 'failed'
                      and unit.get('main_pid_zero') is True
                      and unit.get('n_restarts_zero') is True
                      and unit.get('cgroup_empty') is True
                      and unit.get('resource_policy_verified') is True)
    if valid:
        return {'schema': 'luna-semantic-probe-progress-v2', 'phase': 'startup_failed',
                'zero_admission_verified': True, 'paid_turns': 0,
                'known_tokens': 0, 'model_calls': 0,
                'failed_unit_cleanup_verified': failed_cleanup,
                'completed_and_clean': False, 'semantic_fix_accepted': False}
    if (sidecar['entry_action'] == 'smoke' and type(startup) is dict and
            json.dumps(startup, sort_keys=True, separators=(',', ':')) ==
            json.dumps(smoke, sort_keys=True, separators=(',', ':'))
            and not (root / 'run').exists()):
        return {'schema': 'luna-semantic-probe-progress-v2',
                'phase': 'containment_smoke_passed',
                'effective_policy_verified': True,
                'zero_admission_verified': True, 'paid_turns': 0,
                'known_tokens': 0, 'model_calls': 0,
                'unit_cleanup_verified': unit.get('clean') is True,
                'completed_and_clean': False, 'semantic_fix_accepted': False}
    if sidecar['entry_action'] == 'smoke':
        raise ValueError('smoke_terminal_invalid')
    # The v1 reader performs full private-result validation for non-startup runs.
    result = progress.observe(root, receipt_sha,
        hashlib.sha256(json.dumps(receipt['source_sha256'], sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
        systemd=lambda unit_name, group: unit)
    result['schema'] = 'luna-semantic-probe-progress-v2'
    result['zero_admission_verified'] = False
    result['failed_unit_cleanup_verified'] = failed_cleanup
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'launch', 'smoke', 'inference', 'observe'))
    parser.add_argument('--mode', choices=('smoke', 'inference'))
    parser.add_argument('--staged-root')
    parser.add_argument('--adapter-source')
    parser.add_argument('--root')
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--adapter-sha256')
    parser.add_argument('--adapter-receipt-sha256')
    args = parser.parse_args(argv)
    try:
        if args.action in {'smoke', 'inference'}:
            return entry(Path(args.root), args.receipt_sha256, args.action,
                         args.adapter_sha256, args.adapter_receipt_sha256)
        source_root = Path(args.staged_root) if args.action == 'prepare' else Path(args.root)
        diagnostics = source_root / 'code/tools/diagnostics'
        host = load_source(diagnostics / 'luna_semantic_probe_host.py',
                           'sealed_semantic_host_v1', HOST_SHA)
        if args.action == 'prepare':
            out = host.prepare(Path(args.staged_root))
            root = Path(out['root'])
            sidecar = seal(root, Path(args.adapter_source or __file__), host, args.mode)
            out['adapter_sha256'] = sidecar['adapter_sha256']
            out['adapter_receipt_sha256'] = sidecar['sidecar_sha256']
        elif args.action == 'launch':
            out = launch(Path(args.root), args.receipt_sha256, args.adapter_sha256,
                         args.adapter_receipt_sha256, host, args.mode)
        else:
            progress = load_source(diagnostics / 'luna_semantic_probe_progress.py',
                                   'sealed_semantic_progress_v1', READER_SHA)
            out = observe(Path(args.root), args.receipt_sha256, args.adapter_sha256,
                          args.adapter_receipt_sha256, progress)
        print(json.dumps(out, sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps({'schema': 'luna-semantic-probe-adapter-error-v2',
                          'validated': False, 'reason': 'adapter_validation_failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
