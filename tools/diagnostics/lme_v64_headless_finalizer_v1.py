"""One-shot detached, host-side finalization of an owned R7 sample-eight run.

Local ``install`` and ``launch`` only send this helper and the pinned controller
to Afrodite. The remote worker never starts or resumes a paid live container;
it waits for that container, then creates and runs one network-disabled offline
validation container. Raw logs and benchmark content remain on the host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys

if sys.flags.optimize:
    raise RuntimeError('optimized_execution_forbidden')

ROOT = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/'
            'claim-conflict-20260925-zqkdtxky/lme-v64-sample8-headless-v1')
HOST = ROOT / 'finalizer-host-control.py'
SELF = ROOT / 'finalizer.py'
PIN = None
HOST_SHA = 'e98f9e444fafcdfe9ac8ca65ed80a52d9f7bfdf912ed03cc7d4c2411d0834dc5'
LOCAL_HOST = Path(__file__).with_name('lme_v64_headless_v1') / 'host_control.py'
SSH = ('ssh', '-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=15',
       '-o', 'ServerAliveCountMax=2', 'afrodite')
CID = re.compile(r'[0-9a-f]{64}\Z')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read(path: Path, limit: int = 1024 * 1024) -> bytes:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or not path.is_absolute()
            or path.resolve() != path or info.st_size > limit):
        raise RuntimeError('invalid_finalizer_input')
    return path.read_bytes()


def receipt(name: str, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > 8192:
        raise RuntimeError('oversized_finalizer_receipt')
    fd = os.open(ROOT / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def checked_remote() -> None:
    if os.geteuid() != 1000 or Path(__file__).resolve() != SELF:
        raise RuntimeError('remote_identity_required')
    installed = json.loads(read(ROOT / 'finalizer-install.json'))
    if (installed.get('controller_sha256') != HOST_SHA
            or installed.get('finalizer_sha256') != digest(read(SELF))
            or digest(read(HOST)) != HOST_SHA
            or digest(read(ROOT / 'bundle/manifest.json', 32 * 1024 * 1024)) != PIN):
        raise RuntimeError('finalizer_package_drift')


def owned_live(cid: str) -> None:
    if not CID.fullmatch(cid):
        raise RuntimeError('invalid_live_id')
    owner = json.loads(read(ROOT / 'live-container-id.json'))
    intent = json.loads(read(ROOT / 'live-start-intent.json'))
    if owner != {'container_id': cid} or intent != {
            'container_id': cid, 'manifest_sha256': PIN}:
        raise RuntimeError('unowned_live_container')


def controller(action: str, mode: str, cid: str | None = None) -> dict:
    allowed = {('status', 'live'), ('create', 'validation'),
               ('start', 'validation'), ('status', 'validation')}
    if (action, mode) not in allowed or (cid is None) != (action == 'create'):
        raise RuntimeError('forbidden_finalizer_action')
    if cid is not None and not CID.fullmatch(cid):
        raise RuntimeError('invalid_container_id')
    args = [sys.executable, '-I', '-B', str(HOST), action, mode]
    if cid is not None:
        args.append(cid)
    args += ['--manifest-sha256', PIN]
    result = subprocess.run(args, capture_output=True, timeout=180)
    if result.returncode or len(result.stdout) > 1024 * 1024:
        raise RuntimeError('controller_action_failed')
    value = json.loads(result.stdout)
    if value.get('configuration_verified') is not True:
        raise RuntimeError('unverified_container_configuration')
    if cid is not None and value.get('container_id') != cid:
        raise RuntimeError('container_identity_mismatch')
    return value


def docker_wait(cid: str, timeout: int) -> int:
    if not CID.fullmatch(cid):
        raise RuntimeError('invalid_wait_id')
    result = subprocess.run(['docker', 'wait', cid], capture_output=True,
                            timeout=timeout)
    if result.returncode or not re.fullmatch(rb'(?:0|[1-9][0-9]{0,2})\n?', result.stdout):
        raise RuntimeError('docker_wait_failed')
    code = int(result.stdout)
    if not 0 <= code <= 255:
        raise RuntimeError('docker_exit_code_invalid')
    return code


def remote_launch(cid: str) -> dict:
    checked_remote()
    owned_live(cid)
    live = controller('status', 'live', cid)
    if live['status'] not in ('running', 'exited'):
        raise RuntimeError('live_not_started')
    if (ROOT / 'finalizer-result.json').exists():
        raise RuntimeError('finalizer_already_finished')
    receipt('finalizer-intent.json', {'live_container_id': cid,
            'manifest_sha256': PIN, 'validation_runs_allowed': 1,
            'new_paid_runs_allowed': 0})
    # The intent is durable before the detached process exists. An ambiguous
    # spawn result must be inspected, never automatically retried.
    child = subprocess.Popen([sys.executable, '-I', '-B', str(SELF), 'worker', cid, '--manifest-sha256', PIN],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True,
                             close_fds=True, cwd=str(ROOT))
    value = {'status': 'detached_finalizer_started', 'pid': child.pid,
             'live_container_id': cid, 'manifest_sha256': PIN,
             'new_paid_runs_started': 0}
    receipt('finalizer-launch.json', value)
    return value


def worker(cid: str) -> None:
    checked_remote()
    owned_live(cid)
    if (ROOT / 'finalizer-result.json').exists():
        raise RuntimeError('finalizer_already_finished')
    # Exclusive, durable claim precedes any Docker operation. Duplicate workers
    # and ambiguous prior attempts require inspection rather than another run.
    receipt('finalizer-worker-claim.json', {'live_container_id': cid,
            'manifest_sha256': PIN, 'pid': os.getpid(), 'validation_runs_allowed': 1})
    result = {'status': 'finalizer_failed', 'live_container_id': cid,
              'manifest_sha256': PIN, 'validation_container_id': None,
              'live_exit_code': None, 'validation_exit_code': None,
              'new_paid_runs_started': 0, 'score_verified': False}
    try:
        checked_remote()
        owned_live(cid)
        intent = json.loads(read(ROOT / 'finalizer-intent.json'))
        if intent != {'live_container_id': cid, 'manifest_sha256': PIN,
                      'validation_runs_allowed': 1, 'new_paid_runs_allowed': 0}:
            raise RuntimeError('finalizer_intent_drift')
        result['live_exit_code'] = docker_wait(cid, 9 * 3600 + 60)
        live = controller('status', 'live', cid)
        if (live['status'] != 'exited' or live['pid'] != 0
                or live['exit_code'] != result['live_exit_code']):
            raise RuntimeError('live_terminal_state_unverified')
        created = controller('create', 'validation')
        vid = created['container_id']
        if created['status'] != 'created' or not CID.fullmatch(vid):
            raise RuntimeError('validation_create_unverified')
        result['validation_container_id'] = vid
        started = controller('start', 'validation', vid)
        if started['status'] not in ('running', 'exited'):
            raise RuntimeError('validation_start_unverified')
        result['validation_exit_code'] = docker_wait(vid, 3600)
        finished = controller('status', 'validation', vid)
        if (finished['status'] != 'exited' or finished['pid'] != 0
                or finished['exit_code'] != result['validation_exit_code']):
            raise RuntimeError('validation_terminal_state_unverified')
        result['status'] = 'offline_validation_finished'
    except BaseException as exc:
        # A timeout says nothing about whether remote/container work finished.
        # Never rerun, create another validation container, or echo raw output.
        result['status'] = 'outcome_requires_inspection'
        result['error_type'] = type(exc).__name__
    receipt('finalizer-result.json', result)


REMOTE_INSTALL = r'''
import hashlib,json,os,pathlib,stat,sys
if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
root=pathlib.Path(C['root'])
assert os.geteuid()==1000 and root.resolve()==root and root.is_dir()
assert hashlib.sha256((root/'bundle/manifest.json').read_bytes()).hexdigest()==C['pin']
bodies={}
for name in ('finalizer.py','finalizer-host-control.py'):
    size=C['sizes'][name]; body=sys.stdin.buffer.read(size)
    assert len(body)==size and hashlib.sha256(body).hexdigest()==C['digests'][name]
    bodies[name]=body
assert sys.stdin.buffer.read(1)==b''
for name,body in bodies.items():
    fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
value={'status':'finalizer_installed_not_launched','manifest_sha256':C['pin'],
       'finalizer_sha256':C['digests']['finalizer.py'],
       'controller_sha256':C['digests']['finalizer-host-control.py'],
       'new_paid_runs_started':0}
fd=os.open(root/'finalizer-install.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as stream:json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
print(json.dumps(value,sort_keys=True))
'''


def ssh_run(command: str, *, payload: bytes = b'', timeout: int = 120) -> dict:
    try:
        reply = subprocess.run([*SSH, command], input=payload,
                               capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'status': 'ssh_timeout', 'outcome': 'unknown',
                'requires_inspection': True, 'retry_attempted': False}
    if reply.returncode or len(reply.stdout) > 8192:
        return {'status': 'ssh_operation_failed', 'outcome': 'unknown',
                'requires_inspection': True, 'retry_attempted': False}
    return json.loads(reply.stdout)


def local_install() -> dict:
    source = read(Path(__file__).resolve())
    host = read(LOCAL_HOST)
    if digest(host) != HOST_SHA:
        raise RuntimeError('controller_pin_drift')
    payload = source + host
    names = ('finalizer.py', 'finalizer-host-control.py')
    cfg = {'root': str(ROOT), 'pin': PIN,
           'sizes': dict(zip(names, (len(source), len(host)))),
           'digests': dict(zip(names, (digest(source), digest(host))))}
    code = 'import json\nC=json.loads(' + repr(json.dumps(cfg)) + ')\n' + REMOTE_INSTALL
    return ssh_run('python3 -I -B -c ' + shlex.quote(code), payload=payload, timeout=300)


def local_launch(cid: str) -> dict:
    if not CID.fullmatch(cid):
        raise RuntimeError('invalid_live_id')
    source_pin = digest(read(Path(__file__).resolve()))
    # Execute only the bytes verified against this reviewed local helper.
    # Compiling the captured bytes also avoids reopening a changed remote file.
    code = ('import hashlib,pathlib,stat,sys\n'
            'p=pathlib.Path(' + repr(str(SELF)) + ')\n'
            's=p.lstat()\n'
            'assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_size<=1048576\n'
            'raw=p.read_bytes()\n'
            'assert hashlib.sha256(raw).hexdigest()==' + repr(source_pin) + '\n'
            'sys.argv=' + repr([str(SELF), 'remote-launch', cid, '--manifest-sha256', PIN]) + '\n'
            "exec(compile(raw,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})\n")
    return ssh_run('python3 -I -B -c ' + shlex.quote(code), timeout=120)


def main() -> int:
    global PIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'launch', 'remote-launch', 'worker'))
    parser.add_argument('container_id', nargs='?')
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    PIN = args.manifest_sha256
    if not re.fullmatch('[0-9a-f]{64}', PIN):
        parser.error('invalid preparation pin')
    if (args.action == 'install') != (args.container_id is None):
        parser.error('container ID required exactly for launch/worker')
    try:
        if args.action == 'install':
            value = local_install()
        elif args.action == 'launch':
            value = local_launch(args.container_id)
        elif args.action == 'remote-launch':
            value = remote_launch(args.container_id)
        else:
            checked_remote()
            worker(args.container_id)
            return 0
    except (KeyboardInterrupt, Exception) as exc:
        value = {'status': 'finalizer_operation_failed', 'error_type': type(exc).__name__,
                 'outcome': 'unknown', 'requires_inspection': True}
    print(json.dumps(value, sort_keys=True))
    return 0 if value['status'] in ('finalizer_installed_not_launched',
                                    'detached_finalizer_started') else 1


if __name__ == '__main__':
    raise SystemExit(main())
