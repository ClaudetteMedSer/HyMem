"""Explicit seal/offline/live/status actions. Never retries or launches at seal.

Seal each reviewed baseline/candidate tree separately, supplying reference file
and canonical-record hashes discovered read-only on Afrodite. Transport the whole
sealed directory to a new private Afrodite directory; compare manifest SHA256.
Run offline there, review offline.json, then explicitly run live with the same
pin. Export only offline.json/live.json/supervisor.json; private/ stays Afrodite.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REFERENCE = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-v64-sample8-headless-v1/live-results/stores/hymem-lme-2bw4e0ow'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
KEY = '/opt/stacks/hermes/instance1/home/.hermes/.env'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
CHUNKS = ('chk_aace7a800d186c4351ce09307040592724ff7239', 'chk_57346bc06395cd17a1f1bfd38b6701a3de5e5b8e')
BOUNDS = {'completion_calls': 96, 'http_attempts': 288, 'timeout_seconds': 1200}


def need(condition, code):
    if not condition:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def seal(tree, root, label, records, reference):
    need(label in ('baseline', 'candidate'), 'source_label')
    need(tree.is_absolute() and tree.resolve() == tree and tree.is_dir(), 'source_path')
    need(root.is_absolute() and root.resolve() == root and not root.exists(), 'new_package_required')
    need(not root.is_relative_to(tree), 'package_inside_source')
    need(set(records) == set(CHUNKS) and all(re.fullmatch('[0-9a-f]{64}', v) for v in records.values()), 'record_pins')
    need('hymem.sqlite' in reference and set(reference) <= {'hymem.sqlite', 'hymem.sqlite-wal', 'hymem.sqlite-shm'}
         and all(re.fullmatch('[0-9a-f]{64}', v) for v in reference.values()), 'reference_pins')
    files = {}
    for path in tree.rglob('*'):
        relative = path.relative_to(tree)
        if any(p in ('.git', '.pytest_cache', '__pycache__') for p in relative.parts) or path.suffix in ('.pyc', '.pyo'):
            continue
        need(not path.is_symlink() and path.resolve() == path, 'source_symlink')
        need(path.is_file() or path.is_dir(), 'source_special_file')
        if path.is_file():
            files['candidate/'+relative.as_posix()] = path.read_bytes()
    need(bool(files) and 'candidate/hymem/extraction/chunk.py' in files, 'source_empty')
    here = Path(__file__).resolve().parent
    if label == 'baseline':
        expected = json.loads((here.parents[1]/'docs/patches/2026-09-26-episode-shadow-manifest.json').read_text())
        actual = {p.removeprefix('candidate/'): hashlib.sha256(b).hexdigest() for p,b in files.items()}
        need(actual == expected, 'baseline_inventory_drift')
    for relative, path in {
        'diag/worker.py': here/'lme_r8_chunk_replay.py',
        'support/worker.py': here/'lme_summary_recovery_v1/worker.py',
        'support/supervised_invocation.py': here/'lme_sample8_v1/bundle/supervised_invocation.py',
        'host.py': Path(__file__).resolve(),
    }.items():
        files[relative] = path.read_bytes()
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    manifest = {'schema': 'r8-retained-chunks-v1', 'source_label': label,
                'chunks': records, 'reference_sha256': reference, 'bounds': BOUNDS,
                'source_sha256': {p.removeprefix('candidate/'): digest(b) for p,b in files.items() if p.startswith('candidate/')},
                'worker_sha256': digest(files['diag/worker.py']), 'host_sha256': digest(files['host.py']),
                'support_sha256': digest(files['support/worker.py']),
                'supervisor_sha256': digest(files['support/supervised_invocation.py']),
                'image': IMAGE, 'runtime_path': RUNTIME, 'reference_path': REFERENCE,
                'model': 'deepseek-flash', 'endpoint': 'https://api.deepseek.com'}
    files['diag/manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    root.mkdir(mode=0o755)
    for relative, body in files.items():
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        with path.open('xb') as stream:
            stream.write(body)
    return {'manifest_sha256': digest(files['diag/manifest.json']), 'source_label': label,
            'source_files': len(manifest['source_sha256']), 'api_calls': 0}


def run(command):
    value = subprocess.run(command, capture_output=True, text=True, timeout=30)
    need(value.returncode == 0, 'docker_failed')
    return value.stdout.strip()


def inspect(cid, pin, mode):
    need(re.fullmatch('[0-9a-f]{64}', cid), 'container_id')
    data = json.loads(run(['docker', 'inspect', cid]))[0]
    need(data['Id'] == cid and data['Image'] == IMAGE, 'container_identity')
    need(data['Config']['User'] == '1000:1000' and data['HostConfig']['ReadonlyRootfs']
         and not data['HostConfig']['Privileged'] and data['HostConfig']['CapDrop'] == ['ALL'], 'container_policy')
    need(data['Config']['Entrypoint'] == ['/home/node/hymem-env/bin/python3']
         and data['Config']['Cmd'] == ['-I', '-B', '/diag/worker.py',
            'supervise' if mode == 'live' else 'offline', '--manifest-sha256', pin]
         and data['HostConfig']['SecurityOpt'] == ['no-new-privileges']
         and data['HostConfig']['PidsLimit'] == 128
         and data['HostConfig']['Memory'] == 2*1024**3
         and data['HostConfig']['NanoCpus'] == 2*10**9, 'container_execution')
    return {'container_id': cid, 'status': data['State']['Status'], 'exit_code': data['State']['ExitCode'],
            'oom_killed': data['State']['OOMKilled'], 'pid': data['State']['Pid'],
            'network': data['HostConfig']['NetworkMode'],
            'mounts': {m['Destination']: {'source': m['Source'], 'writable': m['RW']}
                       for m in data['Mounts'] if m['Type'] == 'bind'}}


def action(root, mode, pin, status=False):
    need(root.is_absolute() and root.resolve() == root, 'package_path')
    need(re.fullmatch('[0-9a-f]{64}', pin) and sha(root/'diag/manifest.json') == pin, 'manifest_pin')
    manifest = json.loads((root/'diag/manifest.json').read_text())
    need(sha(Path(__file__).resolve()) == manifest['host_sha256'], 'host_drift')
    receipt, results = root/(mode+'-launch.json'), root/(mode+'-results')
    if status:
        saved = json.loads(receipt.read_text())
        need(saved['manifest_sha256'] == pin, 'receipt_pin')
        value = {**saved, **inspect(saved['container_id'], pin, mode)}
        for name in (mode+'.json', 'supervisor.json'):
            if (results/name).is_file():
                value[name] = json.loads((results/name).read_text())
        return value
    need(not receipt.exists() and not results.exists(), 'already_launched')
    live = mode == 'live'
    if live:
        prior = action(root, 'offline', pin, status=True)
        need(prior['status'] == 'exited' and prior['exit_code'] == 0 and prior['pid'] == 0
             and not prior['oom_killed'] and prior['network'] == 'none', 'offline_process')
        proof = prior['offline.json']
        need(proof['status'] == 'offline_passed' and proof['manifest_sha256'] == pin
             and proof['reference_unchanged'] and proof['source_unchanged'] and proof['threads_clean']
             and proof['client_closed'] and len(proof['chunks']) == 2
             and all(c['exact_partition_verified'] for c in proof['chunks'])
             and proof['usage']['calls'] == proof['usage']['request_attempts'] == 0, 'offline_proof')
    results.mkdir(mode=0o700)
    os.chown(results, 1000, 1000)
    mounts = {'/home/node/hymem-env': (RUNTIME, False), '/candidate': (str(root/'candidate'), False),
              '/diag': (str(root/'diag'), False), '/support': (str(root/'support'), False),
              '/reference': (REFERENCE, False), '/results': (str(results), True)}
    if live:
        mounts['/run/deepseek.env'] = (KEY, False)
    name = 'hymem-r8-'+manifest['source_label']+'-'+pin[:16]+'-'+mode
    command = ['docker', 'create', '--name', name, '--pull', 'never', '--init', '--network', 'bridge' if live else 'none',
               '--user', '1000:1000', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '128', '--memory', '2g', '--cpus', '2', '--tmpfs', '/tmp:rw,nosuid,size=128m',
               '--env', 'PYTHONDONTWRITEBYTECODE=1']
    for target, (source, writable) in mounts.items():
        command += ['--mount', 'type=bind,src='+source+',dst='+target+('' if writable else ',readonly')]
    command += ['--entrypoint', '/home/node/hymem-env/bin/python3', IMAGE, '-I', '-B',
                '/diag/worker.py', 'supervise' if live else 'offline', '--manifest-sha256', pin]
    cid = run(command)
    state = inspect(cid, pin, mode)
    need(state['status'] == 'created' and state['network'] == ('bridge' if live else 'none')
         and state['mounts'] == {t: {'source':s, 'writable':w} for t,(s,w) in mounts.items()}, 'container_mounts')
    save(receipt, {'manifest_sha256': pin, 'container_id': cid, 'mode': mode})
    need(run(['docker', 'start', cid]) == cid, 'container_start')
    return {'manifest_sha256': pin, **inspect(cid, pin, mode)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('seal', 'offline', 'live', 'status-offline', 'status-live'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pin')
    parser.add_argument('--tree', type=Path)
    parser.add_argument('--label', choices=('baseline', 'candidate'))
    parser.add_argument('--records-json', type=Path, help='JSON mapping chunk IDs to canonical-record SHA256')
    parser.add_argument('--reference-json', type=Path, help='JSON mapping reference DB filenames to exact-byte SHA256')
    args = parser.parse_args()
    if args.action == 'seal':
        need(all((args.tree, args.label, args.records_json, args.reference_json)), 'seal_arguments')
        value = seal(args.tree, args.root, args.label,
                     json.loads(args.records_json.read_text()), json.loads(args.reference_json.read_text()))
    else:
        need(args.pin, 'pin_required')
        value = action(args.root, args.action.removeprefix('status-'), args.pin, args.action.startswith('status-'))
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'host_failed': True, 'exception_type': type(exc).__name__}))
        raise SystemExit(1)
