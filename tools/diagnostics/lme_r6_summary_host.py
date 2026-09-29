"""Exclusive one-shot R6 summary diagnostic launch; status is read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ')
ROOT = BASE/'r6-fix3-summary-v3'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
KEY = '/opt/stacks/hermes/instance1/home/.hermes/.env'


def run(command):
    value = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if value.returncode:
        raise RuntimeError('docker_command_failed')
    return value.stdout.strip()


def read(path):
    assert path.resolve() == path and stat.S_ISREG(path.lstat().st_mode)
    return path.read_bytes()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True)


def plan(mode, pin):
    assert mode in ('offline', 'live') and re.fullmatch('[0-9a-f]{64}', pin)
    live = mode == 'live'
    mounts = {
        '/home/node/hymem-env': (RUNTIME, False),
        '/candidate': (str(ROOT/'candidate'), False),
        '/diag': (str(ROOT/'diag'), False), '/support': (str(ROOT/'support'), False),
        '/reference': (str(BASE/'q1-stock-v4/live-results/stores/hymem-lme-ktsn8xyj'), False),
        '/work': (str(ROOT/'work'), True),
        '/results': (str(ROOT/(mode+'-results')), True),
    }
    if live:
        mounts['/run/deepseek.env'] = (KEY, False)
        mounts['/preflight'] = (str(ROOT/'offline-results'), False)
    command = ['docker', 'create', '--name', 'hymem-r6-fix3-summary-v3-'+mode,
               '--pull', 'never', '--init', '--network', 'bridge' if live else 'none',
               '--user', '1000:1000', '--read-only', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges', '--pids-limit', '128',
               '--memory', '2g', '--cpus', '2', '--tmpfs', '/tmp:rw,noexec,nosuid,size=128m',
               '--env', 'PYTHONDONTWRITEBYTECODE=1']
    for target, (source, writable) in mounts.items():
        command += ['--mount', 'type=bind,src='+source+',dst='+target+('' if writable else ',readonly')]
    command += ['--workdir', '/candidate', '--entrypoint', '/home/node/hymem-env/bin/python3',
                IMAGE, '-I', '-B', '/diag/worker.py', 'supervise' if live else 'offline',
                '--manifest-sha256', pin]
    return {'command': command, 'mounts': mounts, 'network': 'bridge' if live else 'none',
            'name': 'hymem-r6-fix3-summary-v3-'+mode}


def inspect(cid, expected):
    assert re.fullmatch('[0-9a-f]{64}', cid)
    data = json.loads(run(['docker', 'inspect', cid]))[0]
    cfg, hc, state = data['Config'], data['HostConfig'], data['State']
    assert data['Id'] == cid and data['Image'] == cfg['Image'] == IMAGE
    assert data['Name'] == '/'+expected['name'] and cfg['User'] == '1000:1000'
    assert cfg['WorkingDir'] == '/candidate' and cfg['Entrypoint'] == ['/home/node/hymem-env/bin/python3']
    assert data['Path'] == '/home/node/hymem-env/bin/python3'
    assert data['Args'] == cfg['Cmd'] == expected['command'][expected['command'].index(IMAGE)+1:]
    assert hc['NetworkMode'] == expected['network'] and hc['ReadonlyRootfs'] is True
    assert hc['Privileged'] is False and hc['CapDrop'] == ['ALL']
    assert hc['SecurityOpt'] == ['no-new-privileges'] and hc['Init'] is True
    assert hc['PidsLimit'] == 128 and hc['Memory'] == 2147483648 and hc['NanoCpus'] == 2000000000
    assert hc['Tmpfs'] == {'/tmp': 'rw,noexec,nosuid,size=128m'} and hc['RestartPolicy']['Name'] == 'no'
    assert not any(v.startswith(('DEEPSEEK_', 'OPENAI_', 'HYMEM_')) for v in cfg['Env'])
    assert {m['Destination']: (m['Source'], m['RW'], m['Type']) for m in data['Mounts']} == {
        dest: (origin, rw, 'bind') for dest, (origin, rw) in expected['mounts'].items()}
    return {'container_id': cid, 'status': state['Status'], 'exit_code': state['ExitCode'],
            'oom_killed': state['OOMKilled'], 'pid': state['Pid'],
            'configuration_verified': True, 'network': expected['network'],
            'credential_mount': '/run/deepseek.env' in expected['mounts']}


def live_gate(pin):
    prior = json.loads(read(ROOT/'offline-launch.json'))
    assert prior['manifest_sha256'] == pin
    state = inspect(prior['container_id'], plan('offline', pin))
    assert state['status'] == 'exited' and state['exit_code'] == 0
    assert state['pid'] == 0 and state['oom_killed'] is False
    proof = json.loads(read(ROOT/'offline-results/offline.json'))
    assert proof['status'] == 'offline_passed' and proof['manifest_sha256'] == pin
    assert proof['reference_unchanged'] is True and proof['source_unchanged'] is True
    assert proof['client_closed'] is True and proof['threads_clean'] is True
    assert proof['provider_calls'] == proof['http_attempts'] == 0
    assert proof['retained_targets'] == 10 and proof['control_targets'] == 4
    info = Path(KEY).lstat()
    assert Path(KEY).resolve() == Path(KEY) and stat.S_ISREG(info.st_mode)
    assert info.st_uid == 1000 and stat.S_IMODE(info.st_mode) == 0o600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('offline', 'live', 'status-offline', 'status-live'))
    parser.add_argument('--pin', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{64}', args.pin) and ROOT.resolve() == ROOT
    raw = read(ROOT/'diag/manifest.json')
    assert hashlib.sha256(raw).hexdigest() == args.pin
    manifest = json.loads(raw)
    assert manifest['schema'] == 'r6-fix3-summary-recovery-package-v1'
    assert manifest['total_completion_cap'] == 124 and manifest['total_http_attempt_cap'] == 372
    mode = args.action.removeprefix('status-')
    expected = plan(mode, args.pin)
    receipt = ROOT/(mode+'-launch.json')
    results = ROOT/(mode+'-results')
    if args.action.startswith('status-'):
        value = json.loads(read(receipt))
        assert value['manifest_sha256'] == args.pin
        value.update(inspect(value['container_id'], expected))
        for name in (mode+'.json', 'supervisor.json'):
            if (results/name).exists():
                value[name] = json.loads(read(results/name))
        print(json.dumps(value, sort_keys=True))
        return
    if mode == 'live':
        live_gate(args.pin)
    else:
        (ROOT/'work').mkdir(mode=0o700)
        os.chown(ROOT/'work', 1000, 1000)
    assert not receipt.exists()
    results.mkdir(mode=0o700)
    os.chown(results, 1000, 1000)
    # Durable exclusive intent prevents a failed/uncertain create/start retry
    # from silently creating another paid invocation.
    save(ROOT/(mode+'-intent.json'), {'manifest_sha256': args.pin, 'mode': mode})
    cid = run(expected['command'])
    state = inspect(cid, expected)
    assert state['status'] == 'created'
    save(receipt, {'manifest_sha256': args.pin, 'container_id': cid, 'mode': mode})
    assert run(['docker', 'start', cid]) == cid
    print(json.dumps({'manifest_sha256': args.pin, **inspect(cid, expected)}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'host_failed': True, 'exception_type': type(exc).__name__}))
        raise SystemExit(1)
