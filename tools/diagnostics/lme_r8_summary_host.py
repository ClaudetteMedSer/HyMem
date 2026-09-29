"""Seal and explicitly launch one R8 summary diagnostic, with no rerolls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import types

REFERENCE = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-v64-sample8-headless-v1/live-results/stores/hymem-lme-2bw4e0ow'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
KEY = '/opt/stacks/hermes/instance1/home/.hermes/.env'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
BOUNDS = {'completion_calls': 192, 'http_attempts': 576, 'timeout_seconds': 2700}
NORMAL_PARAMETERS = {'max_chars': 12000, 'max_tokens': 3072, 'granular': False,
    'max_episodes': 12, 'separate_summary': True, 'prior_summary_is_stale': False}
RECOVERY_PARAMETERS = {'max_attempts': 3, 'max_chars': 8000, 'max_tokens': 3072}
SCHEMA = 'r8-summary-repair-verification-package-v2'
PREVIOUS_ROOT = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r8-summary-v2'
PREVIOUS_PINS = {
    'manifest_sha256': 'cf341386e467e7879cbabaa8380b761c3e905ec674757d3990842dbf0f81e67c',
    'capture_inventory_sha256': 'fd645967f1596857eeb1621d73bc82d4dd3349991c94a20fd6e35e6a3779b325',
    'request_file': '035-request.json',
    'request_sha256': 'ad07d82fac2f0b65c8fd2415f2bf656dfedb54177396e9ae1536d73b267afb78',
    'request_body_sha256': '807dc2b13fbcfd1fd6940149f4472144926549288a5c3172e62d8916fdc35a36',
    'returned_chars': 505,
}
SUPPORT_SHA = 'bd8cc72c9bec26e391632af0f133d226e00f367807120b7a9a395e4dc55bf7a5'
SUPERVISOR_SHA = '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'


def need(value, code):
    if not value:
        raise RuntimeError(code)


def sha(path):
    need(path.resolve() == path and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), 'regular_file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')


def seal(tree, root, reference):
    need(tree.is_absolute() and tree.resolve() == tree and tree.is_dir(), 'source_path')
    need(root.is_absolute() and root.resolve() == root and not root.exists()
         and not root.is_relative_to(tree), 'new_package_required')
    need('hymem.sqlite' in reference and set(reference) <= {'hymem.sqlite', 'hymem.sqlite-wal', 'hymem.sqlite-shm'}
         and all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) for v in reference.values()), 'reference_pins')
    files = {}
    for path in tree.rglob('*'):
        rel = path.relative_to(tree)
        if any(p in ('__pycache__', '.pytest_cache') for p in rel.parts) or path.suffix in ('.pyc', '.pyo'):
            continue
        need(not path.is_symlink() and (path.is_file() or path.is_dir()), 'source_special_file')
        if path.is_file():
            files['candidate/'+rel.as_posix()] = path.read_bytes()
    need('candidate/hymem/dreaming/digest.py' in files
         and 'candidate/hymem/core/schema.sql' in files, 'complete_source_required')
    here = Path(__file__).resolve().parent
    for relative, path in {
        'diag/worker.py': here/'lme_r8_summary_replay.py', 'host.py': Path(__file__).resolve(),
        'support/worker.py': here/'lme_summary_recovery_v1/worker.py',
        'support/supervised_invocation.py': here/'lme_sample8_v1/bundle/supervised_invocation.py',
        'support/r6_summary.py': here/'lme_r6_summary_replay.py',
        'support/r6_host.py': here/'lme_r6_summary_host.py',
    }.items():
        files[relative] = path.read_bytes()
    digest = lambda body: hashlib.sha256(body).hexdigest()
    need(digest(files['support/worker.py']) == SUPPORT_SHA
         and digest(files['support/supervised_invocation.py']) == SUPERVISOR_SHA, 'support_pin')
    manifest = {'schema': SCHEMA, 'bounds': BOUNDS, 'retained_targets': 14, 'control_targets': 4,
        'repair_control_targets': 4, 'previous_pins': PREVIOUS_PINS, 'previous_root': PREVIOUS_ROOT,
        'normal_parameters': NORMAL_PARAMETERS, 'recovery_parameters': RECOVERY_PARAMETERS,
        'source_sha256': {p.removeprefix('candidate/'): digest(b) for p, b in files.items() if p.startswith('candidate/')},
        'reference_sha256': reference, 'worker_sha256': digest(files['diag/worker.py']),
        'host_sha256': digest(files['host.py']), 'support_sha256': SUPPORT_SHA,
        'supervisor_sha256': SUPERVISOR_SHA, 'r6_summary_sha256': digest(files['support/r6_summary.py']),
        'r6_host_sha256': digest(files['support/r6_host.py']), 'reference_path': REFERENCE,
        'image': IMAGE, 'runtime_path': RUNTIME, 'model': 'deepseek-flash',
        'endpoint': 'https://api.deepseek.com', 'production_changes': False, 'rerolls_allowed': False}
    files['diag/manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    root.mkdir(mode=0o755)
    for relative, body in files.items():
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        with path.open('xb') as handle:
            handle.write(body)
    return {'manifest_sha256': digest(files['diag/manifest.json']), 'source_files': len(manifest['source_sha256']),
            'api_calls': 0}


def run(command):
    outcome = subprocess.run(command, capture_output=True, text=True, timeout=30)
    need(outcome.returncode == 0, 'docker_command_failed')
    return outcome.stdout.strip()


def plan(root, mode, pin):
    need(mode in ('offline', 'live') and re.fullmatch('[0-9a-f]{64}', pin), 'launch_scope')
    live = mode == 'live'
    mounts = {'/home/node/hymem-env': (RUNTIME, False), '/candidate': (str(root/'candidate'), False),
        '/diag': (str(root/'diag'), False), '/support': (str(root/'support'), False),
        '/reference': (REFERENCE, False), '/work': (str(root/'work'), True),
        '/previous': (PREVIOUS_ROOT+'/live-results', False),
        '/previous-manifest.json': (PREVIOUS_ROOT+'/diag/manifest.json', False),
        '/results': (str(root/(mode+'-results')), True)}
    if live:
        mounts['/run/deepseek.env'] = (KEY, False)
        mounts['/preflight'] = (str(root/'offline-results'), False)
    root_hash = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    name = 'hymem-r8-summary-'+pin[:16]+'-'+root_hash+'-'+mode
    command = ['docker', 'create', '--name', name, '--pull', 'never', '--init', '--network', 'bridge' if live else 'none',
        '--user', '1000:1000', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--pids-limit', '128', '--memory', '2g', '--cpus', '2', '--tmpfs', '/tmp:rw,noexec,nosuid,size=128m',
        '--env', 'PYTHONDONTWRITEBYTECODE=1']
    for target, (source, writable) in mounts.items():
        command += ['--mount', 'type=bind,src='+source+',dst='+target+('' if writable else ',readonly')]
    command += ['--workdir', '/candidate', '--entrypoint', '/home/node/hymem-env/bin/python3', IMAGE,
        '-I', '-B', '/diag/worker.py', 'supervise' if live else 'offline', '--manifest-sha256', pin]
    return {'command': command, 'name': name, 'mounts': mounts, 'network': 'bridge' if live else 'none'}


def inspect(root, cid, expected, manifest):
    path = root/'support/r6_host.py'
    need(sha(path) == manifest['r6_host_sha256'], 'host_support_drift')
    helper = types.ModuleType('r8_summary_host_support')
    helper.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), helper.__dict__)
    helper.run = run
    return helper.inspect(cid, expected)


def action(root, mode, pin, status=False):
    need(root.is_absolute() and root.resolve() == root and sha(root/'diag/manifest.json') == pin, 'manifest_pin')
    manifest = json.loads((root/'diag/manifest.json').read_text())
    need(manifest['schema'] == SCHEMA and manifest['bounds'] == BOUNDS
         and manifest['previous_pins'] == PREVIOUS_PINS and manifest['previous_root'] == PREVIOUS_ROOT
         and manifest['repair_control_targets'] == 4
         and manifest['normal_parameters'] == NORMAL_PARAMETERS
         and manifest['recovery_parameters'] == RECOVERY_PARAMETERS
         and sha(Path(__file__).resolve()) == manifest['host_sha256'], 'host_contract')
    expected = plan(root, mode, pin)
    receipt, results = root/(mode+'-launch.json'), root/(mode+'-results')
    if status:
        saved = json.loads(receipt.read_text())
        need(saved['manifest_sha256'] == pin, 'receipt_pin')
        output = {**saved, **inspect(root, saved['container_id'], expected, manifest)}
        for name in (mode+'.json', 'supervisor.json'):
            if (results/name).exists():
                output[name] = json.loads((results/name).read_text())
        return output
    need(not receipt.exists() and not results.exists() and not (root/(mode+'-intent.json')).exists(), 'already_launched')
    if mode == 'live':
        prior = action(root, 'offline', pin, True)
        need(prior['status'] == 'exited' and prior['exit_code'] == 0 and prior['pid'] == 0
             and not prior['oom_killed'] and prior['network'] == 'none', 'offline_process')
        proof = prior['offline.json']
        need(proof['status'] == 'offline_passed' and proof['manifest_sha256'] == pin
             and proof['previous_pins_verified'] == PREVIOUS_PINS
             and proof['retained_targets'] == 14 and proof['control_targets'] == 4
             and proof['provider_calls'] == proof['http_attempts'] == proof['completion_calls'] == 0
             and all(proof[k] is True for k in ('reference_unchanged', 'source_unchanged', 'clients_closed',
                    'connections_closed', 'threads_clean', 'offline_clones_unchanged')), 'offline_proof')
    else:
        (root/'work').mkdir(mode=0o700)
        os.chown(root/'work', 1000, 1000)
    results.mkdir(mode=0o700)
    os.chown(results, 1000, 1000)
    save(root/(mode+'-intent.json'), {'manifest_sha256': pin, 'mode': mode})
    cid = run(expected['command'])
    state = inspect(root, cid, expected, manifest)
    need(state['status'] == 'created', 'created_container')
    save(receipt, {'manifest_sha256': pin, 'mode': mode, 'container_id': cid})
    need(run(['docker', 'start', cid]) == cid, 'container_start')
    return {'manifest_sha256': pin, **inspect(root, cid, expected, manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('seal', 'offline', 'live', 'status-offline', 'status-live'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tree', type=Path)
    parser.add_argument('--reference-json', type=Path)
    parser.add_argument('--pin')
    args = parser.parse_args()
    if args.action == 'seal':
        need(args.tree and args.reference_json, 'seal_arguments')
        value = seal(args.tree, args.root, json.loads(args.reference_json.read_text()))
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
