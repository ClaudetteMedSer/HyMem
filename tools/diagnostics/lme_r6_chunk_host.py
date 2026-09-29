"""One-shot create/start and read-only status for the retained-chunk replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ')
ROOT = BASE/'r6-fix2-replay-v2'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
KEY = '/opt/stacks/hermes/instance1/home/.hermes/.env'


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('docker_command_failed')
    return result.stdout.strip()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True)


def inspect(cid):
    assert re.fullmatch('[0-9a-f]{64}', cid)
    data = json.loads(run(['docker', 'inspect', cid]))[0]
    assert data['Id'] == cid and data['Image'] == IMAGE
    assert data['Config']['User'] == '1000:1000' and data['HostConfig']['ReadonlyRootfs'] is True
    return {'container_id': cid, 'status': data['State']['Status'], 'exit_code': data['State']['ExitCode'],
            'oom_killed': data['State']['OOMKilled'], 'pid': data['State']['Pid'],
            'network': data['HostConfig']['NetworkMode'],
            'mounts': {item['Destination']: {'source': item['Source'], 'writable': item['RW']}
                       for item in data['Mounts'] if item['Type'] == 'bind'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('offline', 'live', 'status-offline', 'status-live'))
    parser.add_argument('--pin', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{64}', args.pin)
    assert ROOT.resolve() == ROOT
    assert hashlib.sha256((ROOT/'diag/manifest.json').read_bytes()).hexdigest() == args.pin
    mode = args.action.removeprefix('status-')
    receipt = ROOT/(mode+'-launch.json')
    results = ROOT/(mode+'-results')
    if args.action.startswith('status-'):
        value = json.loads(receipt.read_text())
        assert value['manifest_sha256'] == args.pin
        value.update(inspect(value['container_id']))
        for name in (mode+'.json', 'supervisor.json'):
            if (results/name).is_file():
                value[name] = json.loads((results/name).read_text())
        print(json.dumps(value, sort_keys=True))
        return
    live = mode == 'live'
    if live:
        prior = json.loads((ROOT/'offline-launch.json').read_text())
        state = inspect(prior['container_id'])
        assert prior['manifest_sha256'] == args.pin and state['status'] == 'exited'
        assert state['exit_code'] == 0 and state['pid'] == 0 and state['oom_killed'] is False
        assert state['network'] == 'none' and '/run/deepseek.env' not in state['mounts']
        proof = json.loads((ROOT/'offline-results/offline.json').read_text())
        assert proof['status'] == 'offline_passed' and proof['manifest_sha256'] == args.pin
        assert proof['exact_partition_verified'] and proof['canonical_sources_verified']
        assert proof['saved_r5_archive_validated'] and proof['old_contract_not_current']
        assert proof['reference_unchanged'] and proof['source_unchanged'] and proof['client_closed']
        assert proof['usage']['calls'] == proof['usage']['request_attempts'] == 0
    assert not receipt.exists()
    results.mkdir(mode=0o700)
    os.chown(results, 1000, 1000)
    mounts = {
        '/home/node/hymem-env': (RUNTIME, False), '/candidate': (str(ROOT/'candidate'), False),
        '/diag': (str(ROOT/'diag'), False), '/support': (str(ROOT/'support'), False),
        '/reference': (str(BASE/'sample8-stock-v1/live-results/stores/hymem-lme-3_kjdx3h'), False),
        '/legacy-results': (str(BASE/'sample8-stock-v1/live-results/benchmark'), False),
        '/results': (str(results), True),
    }
    if live:
        mounts['/run/deepseek.env'] = (KEY, False)
    command = ['docker', 'create', '--name', 'hymem-r6-fix2-replay-v2-'+mode, '--pull', 'never',
               '--init', '--network', 'bridge' if live else 'none', '--user', '1000:1000',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '128', '--memory', '2g', '--cpus', '2',
               '--tmpfs', '/tmp:rw,nosuid,size=128m', '--env', 'PYTHONDONTWRITEBYTECODE=1']
    for target, (source, writable) in mounts.items():
        command += ['--mount', 'type=bind,src='+source+',dst='+target+('' if writable else ',readonly')]
    command += ['--entrypoint', '/home/node/hymem-env/bin/python3', IMAGE, '-I', '-B',
                '/diag/worker.py', 'supervise' if live else 'offline', '--manifest-sha256', args.pin]
    cid = run(command)
    state = inspect(cid)
    assert state['status'] == 'created'
    assert state['network'] == ('bridge' if live else 'none')
    assert state['mounts'] == {target: {'source': source, 'writable': writable}
                                for target, (source, writable) in mounts.items()}
    save(receipt, {'manifest_sha256': args.pin, 'container_id': cid, 'mode': mode})
    assert run(['docker', 'start', cid]) == cid
    print(json.dumps({'manifest_sha256': args.pin, **inspect(cid)}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'host_failed': True, 'exception_type': type(exc).__name__}))
        raise SystemExit(1)
