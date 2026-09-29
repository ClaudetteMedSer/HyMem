"""One-shot supervised private-clone dream; never mounts production writable.

Reuse only the pinned transport/security primitives from the earlier controller.
All server output is projected through a strict content-free metadata schema.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky')
ROOT = BASE / 'instrumented-dream-v2'
SELF = ROOT / 'claim_conflict_instrumented_host.py'
WORKER = ROOT / 'claim_conflict_instrumented_dream.py'
HELPER = BASE / 'capture-next-v1/claim-conflict-next-host.py'
HELPER_SHA = 'd6d528fb54c662dff217f243ac3890bb9d43478e45a48f2610d7060900ac7195'
CAPTURE_HELPER = BASE / 'capture-next-v1/claim_conflict_next_capture.py'
CAPTURE_SHA = '4ad2819309f51e96eb7b7195d4e215b1d53081ba9551df89cd9c657c23d3178c'
SOURCE = BASE / 'offline-compare-v1/candidate'
REFERENCE = BASE / 'capture-next-v1/reference.sqlite'
REFERENCE_SHA = '7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0'
PHASE1_SHA = '31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136'
WORK = ROOT / 'work'
SSH = ('ssh', '-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ConnectionAttempts=1', 'afrodite')


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    hashed = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            hashed.update(block)
    return hashed.hexdigest()


def helper():
    need(not HELPER.is_symlink() and sha(HELPER) == HELPER_SHA, 'helper_pin_drift')
    spec = importlib.util.spec_from_file_location('instrumented_host_primitives', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(h):
    h.source_inventory()
    manifest = h.read_json(h.MANIFEST)
    expected = {}
    for group in ('source_sha256', 'test_sha256', 'auxiliary_sha256'):
        expected.update(manifest[group])
    expected['hymem/dreaming/phase1.py'] = PHASE1_SHA
    need(SOURCE.is_dir() and not SOURCE.is_symlink(), 'candidate_missing')
    actual = {}
    for path in SOURCE.rglob('*'):
        need(not path.is_symlink(), 'candidate_symlink')
        if path.is_file():
            actual[path.relative_to(SOURCE).as_posix()] = sha(path)
    need(len(actual) == 479 and actual == expected, 'candidate_pin_drift')
    need(os.geteuid() == 1000 and ROOT.is_dir() and not ROOT.is_symlink()
         and WORK.is_dir() and not WORK.is_symlink()
         and WORK.stat().st_mode & 0o777 == 0o700, 'stage_not_private')
    h.regular(REFERENCE, mode=0o400)
    h.regular(h.RUNTIME_ENV, mode=0o600)
    h.regular(WORKER, mode=0o400)
    need(sha(REFERENCE) == REFERENCE_SHA and sha(CAPTURE_HELPER) == CAPTURE_SHA,
         'input_pin_drift')
    return {'source_files': 479, 'phase1_sha256': PHASE1_SHA,
            'reference_sha256': REFERENCE_SHA, 'host_sha256': sha(SELF),
            'worker_sha256': sha(WORKER), 'runtime_env_sha256': sha(h.RUNTIME_ENV)}


def installed(h):
    pins = h.read_json(ROOT / 'install.json')
    need(pins == inputs(h), 'installed_pin_drift')
    return pins


def configure(h, mode):
    need(mode in ('offline', 'live'), 'invalid_mode')
    mounts = [(str(SOURCE), '/candidate', False),
              (str(WORKER), '/diag/claim_conflict_instrumented_dream.py', False),
              (str(REFERENCE), '/reference/source.sqlite', False),
              (str(WORK), '/work', True),
              (str(h.RUNTIME), '/home/node/hymem-env', False),
              (str(h.RUNTIME_ENV), '/run/runtime-env.json', False)]
    command = ['docker', 'create', '--name', 'hymem-' + ROOT.name + '-' + mode,
               '--pull', 'never', '--init', '--network', 'hermes-net' if mode == 'live' else 'none',
               '--user', '1000:1000', '--read-only', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges', '--pids-limit', '128',
               '--memory', '2g', '--cpus', '2', '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
               '--env', 'HOME=/tmp', '--env', 'TMPDIR=/tmp', '--env', 'PYTHONDONTWRITEBYTECODE=1']
    for src, dst, rw in mounts:
        command += ['--mount', 'type=bind,src=' + src + ',dst=' + dst + ('' if rw else ',readonly')]
    command += ['--workdir', '/candidate', '--entrypoint', '/home/node/hymem-env/bin/python3',
                h.IMAGE, '-I', '-B', '/diag/claim_conflict_instrumented_dream.py', mode,
                '--env', '/run/runtime-env.json']
    return command, mounts


def inspect(h, cid, mode, mounts):
    # Reuse the verified inspector with this campaign's explicit configuration.
    original = h.container_command
    h.container_command = lambda actual_mode: configure(h, actual_mode)
    try:
        return h.inspect_container(cid, mode, mounts)
    finally:
        h.container_command = original


def project(summary):
    need(isinstance(summary, dict), 'summary_shape')
    status = summary.get('status')
    need(status in ('ready', 'completed', 'captured_failure', 'budget_stopped', 'error'), 'summary_status')
    need('completion_calls' in summary and 'http_attempts' in summary, 'accounting_missing')
    value = {'status': status}
    for name in ('completion_calls', 'http_attempts', 'captures', 'chunks_processed',
                 'llm_http_attempts', 'embedding_http_attempts', 'prepersist_captured',
                 'extractions_captured', 'exception_events_captured',
                 'exception_controlflow_skipped',
                 'llm_provider_attempts_reported', 'embedding_provider_attempts_reported',
                 'prompt_tokens', 'completion_tokens', 'total_tokens'):
        if name in summary:
            item = summary[name]
            need(type(item) is int and 0 <= item <= 100000000, 'summary_count')
            value[name] = item
    need(value.get('completion_calls', 0) <= 64 and value.get('http_attempts', 0) <= 192,
         'budget_exceeded')
    for name in ('cleanup_ok', 'source_unchanged', 'runtime_generation_verified', 'failure_captured',
                 'instrumentation_capture_ok', 'token_usage_available', 'accounting_verified',
                 'exception_capture_truncated'):
        if name in summary:
            need(type(summary[name]) is bool, 'summary_boolean')
            value[name] = summary[name]
    for name in ('source_sha256', 'capture_sha256', 'session_sha256'):
        if name in summary:
            need(isinstance(summary[name], str) and re.fullmatch('[0-9a-f]{64}', summary[name]), 'summary_hash')
            value[name] = summary[name]
    if summary.get('error_type') is not None:
        allowed = {'ValueError', 'RuntimeError', 'TypeError', 'KeyError', 'IndexError',
                   'OSError', 'FileExistsError', 'FileNotFoundError', 'AssertionError',
                   'DeadlineExceeded', 'BudgetStopped', 'DiagnosticBudgetExceeded',
                   'BudgetStop', 'InstrumentationStop',
                   'Exception', 'BaseException', 'IntegrityError', 'OperationalError'}
        value['error_type'] = summary['error_type'] if summary['error_type'] in allowed else 'Exception'
    frames = summary.get('candidate_frames', [])
    need(isinstance(frames, list) and len(frames) <= 12, 'summary_frames')
    for frame in frames:
        need(isinstance(frame, dict) and set(frame) == {'path', 'function', 'line'}
             and isinstance(frame['path'], str)
             and re.fullmatch(r'hymem(?:/[A-Za-z_][A-Za-z_0-9]*)*/[A-Za-z_][A-Za-z_0-9]*\.py', frame['path'])
             and isinstance(frame['function'], str) and frame['function'].isidentifier()
             and len(frame['function']) <= 96 and type(frame['line']) is int
             and 1 <= frame['line'] <= 100000, 'summary_frame')
    value['candidate_frames'] = frames
    return value


def stop(h, cid, mode, mounts):
    subprocess.run(['docker', 'stop', '--time', '10', cid], capture_output=True, timeout=30)
    state = inspect(h, cid, mode, mounts)
    need(state['status'] in ('created', 'exited') and state['pid'] == 0, 'cleanup_unverified')


def supervise(h):
    result = {'status': 'failed', 'stages': {}, 'paid_live_runs_started': 0}
    try:
        installed(h)
        h.put_json(ROOT / 'supervisor-intent.json', {'paid_live_runs_allowed': 1})
        for mode in ('offline', 'live'):
            installed(h)
            command, mounts = configure(h, mode)
            h.put_json(ROOT / (mode + '-create-intent.json'), {'mode': mode})
            cid = h.run(command, 60, 'create').decode().strip()
            h.put_json(ROOT / (mode + '-container.json'), {'container_id': cid})
            need(inspect(h, cid, mode, mounts)['status'] == 'created', 'not_created')
            h.put_json(ROOT / (mode + '-start-intent.json'), {'container_id': cid})
            try:
                if mode == 'live':
                    result['paid_live_runs_started'] = 1
                need(h.run(['docker', 'start', cid], 60, 'start').decode().strip() == cid, 'start_identity')
                raw = h.run(['docker', 'wait', cid], 960 if mode == 'live' else 180, 'wait')
                need(re.fullmatch(rb'[0-9]{1,3}\n?', raw), 'wait_shape')
            except BaseException:
                stop(h, cid, mode, mounts)
                raise
            state = inspect(h, cid, mode, mounts)
            need(state['status'] == 'exited' and state['pid'] == 0
                 and not state['oom_killed'] and state['exit_code'] == int(raw), 'terminal_state')
            metadata = project(json.loads(h.run(['docker', 'logs', cid], 30, 'logs')))
            result['stages'][mode] = {**state, 'metadata': metadata}
            need(state['exit_code'] == 0 and metadata.get('cleanup_ok') is True
                 and metadata.get('source_unchanged') is True
                 and metadata.get('runtime_generation_verified') is True
                 and metadata.get('source_sha256') == REFERENCE_SHA, 'worker_not_clean')
            if mode == 'offline':
                need(metadata['status'] == 'ready' and metadata.get('completion_calls') == 0
                     and metadata.get('http_attempts') == 0
                     and metadata.get('runtime_generation_verified') is True, 'preflight_failed')
            else:
                need(metadata['status'] in ('completed', 'captured_failure', 'budget_stopped'), 'diagnostic_failed')
                need(metadata.get('instrumentation_capture_ok') is True
                     and metadata.get('accounting_verified') is True, 'capture_or_accounting_incomplete')
                if metadata['status'] == 'captured_failure':
                    need(metadata.get('failure_captured') is True, 'failure_evidence_missing')
            installed(h)
        result['status'] = 'completed'
    except BaseException as exc:
        # No exception message reaches the host receipt: it might contain data.
        result['error_type'] = type(exc).__name__ if type(exc).__name__ in ('RuntimeError', 'ValueError', 'OSError') else 'Exception'
    h.put_json(ROOT / 'result.json', result)


def remote(action):
    h = helper()
    if action == 'remote-install':
        WORK.mkdir(mode=0o700)
        pins = inputs(h)
        h.put_json(ROOT / 'install.json', pins)
        return {'status': 'installed', **pins}
    if action == 'remote-status':
        if (ROOT / 'result.json').exists():
            return h.read_json(ROOT / 'result.json')
        return {'status': 'running_or_requires_inspection' if (ROOT / 'launch-intent.json').exists() else 'installed'}
    if action == 'supervise':
        supervise(h)
        return {'status': 'supervisor_finished'}
    installed(h)
    h.put_json(ROOT / 'launch-intent.json', {'host_sha256': sha(SELF)})
    child = subprocess.Popen([sys.executable, '-I', '-B', str(SELF), 'supervise'],
                             cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    h.put_json(ROOT / 'launch.json', {'pid': child.pid})
    return {'status': 'detached_supervisor_started', 'pid': child.pid}


def ssh_json(command, data=b''):
    try:
        result = subprocess.run([*SSH, command], input=data, capture_output=True, timeout=180)
        need(result.returncode == 0 and 0 < len(result.stdout) <= 16384, 'remote_operation_failed')
        value = json.loads(result.stdout)
        need(isinstance(value, dict), 'remote_output_invalid')
        return value
    except (subprocess.TimeoutExpired, ValueError, RuntimeError):
        return {'status': 'unknown_inspect_before_retry'}


def install():
    bodies = {SELF.name: Path(__file__).read_bytes(), WORKER.name: Path(__file__).with_name(WORKER.name).read_bytes()}
    cfg = {'root': str(ROOT), 'files': {name: {'size': len(raw), 'sha': hashlib.sha256(raw).hexdigest()} for name, raw in bodies.items()}}
    code = '''import hashlib,json,os,pathlib,sys
root=pathlib.Path(C['root'])
assert os.geteuid()==1000 and not root.exists()
root.mkdir(mode=0o700)
for name,item in C['files'].items():
 raw=sys.stdin.buffer.read(item['size'])
 assert len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha']
 fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
assert sys.stdin.buffer.read(1)==b''
print(json.dumps({'status':'uploaded'}))
'''
    script = 'import json\nC=json.loads(' + repr(json.dumps(cfg)) + ')\n' + code
    value = ssh_json('python3 -I -B -c ' + shlex.quote(script), b''.join(bodies.values()))
    if value.get('status') != 'uploaded':
        return value
    return ssh_json('python3 -I -B ' + shlex.quote(str(SELF)) + ' remote-install')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'launch', 'status', 'remote-install', 'remote-launch', 'remote-status', 'supervise'))
    action = parser.parse_args().action
    try:
        if action == 'install':
            value = install()
        elif action in ('launch', 'status'):
            value = ssh_json('python3 -I -B ' + shlex.quote(str(SELF)) + ' remote-' + action)
        else:
            value = remote(action)
    except BaseException:
        value = {'status': 'operation_failed_inspect_before_retry'}
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
