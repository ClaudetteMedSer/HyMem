"""Seal and control the two exclusive R9 diagnostic capsules. Never reads secrets."""
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

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky')
ROOTS = {variant: BASE/('lme-r9-summary-'+variant+'-v2') for variant in ('baseline', 'candidate')}
REFERENCE = str(BASE/'lme-r8-sample8-headless-v3/live-results/stores/hymem-lme-r6e0644s')
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
KEY = '/opt/stacks/hermes/instance1/home/.hermes/.env'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
SCHEMA = 'r9-summary-output-cap-package-v1'
BOUNDS = {'completion_calls': 12, 'http_attempts': 36, 'timeout_seconds': 900, 'invocation_timeout_seconds': 120}
SESSION = 'd1fca62fab454a5a2c6a521bfc4c584219549044a3f7b24ab391ad13063482b2'
SUPPORT_SHA = 'bd8cc72c9bec26e391632af0f133d226e00f367807120b7a9a395e4dc55bf7a5'
SUPERVISOR_SHA = '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'
HEX = re.compile('[0-9a-f]{64}')

def need(value, code):
    if not value:
        raise RuntimeError(code)

def read(path):
    need(path.is_absolute() and path.resolve() == path and stat.S_ISREG(path.lstat().st_mode), 'regular_file')
    return path.read_bytes()

def sha(path):
    return hashlib.sha256(read(path)).hexdigest()

def save(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())

def inventory(tree):
    need(tree.is_absolute() and tree.resolve() == tree and tree.is_dir(), 'source_path')
    result = {}
    for path in sorted(tree.rglob('*')):
        rel = path.relative_to(tree)
        if any(p in ('__pycache__', '.pytest_cache', '.git') for p in rel.parts) or path.suffix in ('.pyc', '.pyo'):
            continue
        need(not path.is_symlink() and (path.is_file() or path.is_dir()), 'source_special_file')
        if path.is_file():
            result[rel.as_posix()] = sha(path)
    need('hymem/dreaming/digest.py' in result and 'hymem/core/schema.sql' in result, 'complete_source_required')
    return result

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def seal(tree, root, reference, variant, source_pin):
    need(variant in ROOTS and root.is_absolute() and root.resolve() == root and not root.exists()
         and not root.is_relative_to(tree), 'new_package_required')
    source = inventory(tree)
    need(type(source_pin) is str and HEX.fullmatch(source_pin) and digest(source) == source_pin, 'source_inventory_pin')
    need('hymem.sqlite' in reference and set(reference) <= {'hymem.sqlite', 'hymem.sqlite-wal', 'hymem.sqlite-shm'}
         and all(type(v) is str and HEX.fullmatch(v) for v in reference.values()), 'reference_pins')
    here = Path(__file__).resolve().parent
    paths = {'diag/worker.py': here/'worker.py', 'diag/core.py': here/'core.py', 'host.py': Path(__file__).resolve(),
             'support/worker.py': here.parent/'lme_summary_recovery_v1/worker.py',
             'support/supervised_invocation.py': here.parent/'lme_sample8_v1/bundle/supervised_invocation.py',
             'support/r6_summary.py': here.parent/'lme_r6_summary_replay.py',
             'support/r6_host.py': here.parent/'lme_r6_summary_host.py',
             'support/r8_summary.py': here.parent/'lme_r8_summary_replay.py'}
    bodies = {rel: read(path) for rel, path in paths.items()}
    body_sha = lambda body: hashlib.sha256(body).hexdigest()
    need(body_sha(bodies['support/worker.py']) == SUPPORT_SHA and
         body_sha(bodies['support/supervised_invocation.py']) == SUPERVISOR_SHA, 'support_pin')
    manifest = {'schema': SCHEMA, 'bounds': BOUNDS, 'variant': variant, 'capsule_root': str(ROOTS[variant]),
                'source_sha256': source, 'source_inventory_sha256':source_pin, 'reference_sha256': reference,
                'reference_path': REFERENCE, 'runtime_path': RUNTIME, 'image': IMAGE,
                'model': 'deepseek-flash', 'endpoint': 'https://api.deepseek.com', 'thinking': False,
                'session_hash': SESSION, 'message_start': 293, 'message_end': 300,
                'selected_messages': 8, 'selected_chars': 6996,
                'production_changes': False, 'rerolls_allowed': False}
    for key, rel in {'worker_sha256':'diag/worker.py', 'core_sha256':'diag/core.py', 'host_sha256':'host.py',
                     'support_sha256':'support/worker.py', 'supervisor_sha256':'support/supervised_invocation.py',
                     'r6_summary_sha256':'support/r6_summary.py', 'r6_host_sha256':'support/r6_host.py',
                     'r8_summary_sha256':'support/r8_summary.py'}.items():
        manifest[key] = body_sha(bodies[rel])
    bodies.update({'candidate/'+rel: read(tree/rel) for rel in source})
    bodies['diag/manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    root.mkdir(mode=0o700)
    for relative, body in bodies.items():
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        with path.open('xb') as handle:
            handle.write(body)
    need(inventory(tree) == source and inventory(root/'candidate') == source, 'seal_source_changed')
    return {'manifest_sha256': body_sha(bodies['diag/manifest.json']), 'source_files': len(source), 'api_calls': 0}

def run(command):
    outcome = subprocess.run(command, capture_output=True, text=True, timeout=30)
    need(outcome.returncode == 0, 'docker_command_failed')
    return outcome.stdout.strip()

def plan(root, mode, pin):
    need(root in ROOTS.values() and mode in ('offline', 'live') and HEX.fullmatch(pin), 'launch_scope')
    live = mode == 'live'
    mounts = {'/home/node/hymem-env': (RUNTIME, False), '/candidate': (str(root/'candidate'), False),
              '/diag': (str(root/'diag'), False), '/support': (str(root/'support'), False),
              '/reference': (REFERENCE, False), '/work': (str(root/'work'), True),
              '/results': (str(root/(mode+'-results')), True)}
    if live:
        mounts['/run/deepseek.env'] = (KEY, False)
        mounts['/preflight'] = (str(root/'offline-results'), False)
    name = root.name+'-'+pin[:16]+'-'+mode
    command = ['docker', 'create', '--name', name, '--pull', 'never', '--init', '--network', 'bridge' if live else 'none',
               '--user', '1000:1000', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '128', '--memory', '2g', '--cpus', '2', '--tmpfs', '/tmp:rw,noexec,nosuid,size=128m',
               '--env', 'PYTHONDONTWRITEBYTECODE=1']
    for target, (origin, writable) in mounts.items():
        command += ['--mount', 'type=bind,src='+origin+',dst='+target+('' if writable else ',readonly')]
    command += ['--workdir', '/candidate', '--entrypoint', '/home/node/hymem-env/bin/python3', IMAGE,
                '-I', '-B', '/diag/worker.py', 'supervise' if live else 'offline', '--manifest-sha256', pin]
    return {'command': command, 'name': name, 'mounts': mounts, 'network': 'bridge' if live else 'none'}

def verify(root, pin):
    need(root in ROOTS.values() and root.resolve() == root and HEX.fullmatch(pin)
         and sha(root/'diag/manifest.json') == pin, 'manifest_pin')
    manifest = json.loads(read(root/'diag/manifest.json'))
    required = {'schema': SCHEMA, 'bounds': BOUNDS, 'capsule_root': str(root), 'reference_path': REFERENCE,
                'runtime_path': RUNTIME, 'image': IMAGE, 'model': 'deepseek-flash',
                'endpoint': 'https://api.deepseek.com', 'thinking': False, 'session_hash': SESSION,
                'message_start':293, 'message_end':300, 'selected_messages':8, 'selected_chars':6996,
                'variant': next(k for k, v in ROOTS.items() if v == root),
                'production_changes':False, 'rerolls_allowed':False}
    need(all(manifest.get(k) == v for k,v in required.items()), 'host_contract')
    need(inventory(root/'candidate') == manifest['source_sha256'], 'source_drift')
    need(digest(manifest['source_sha256']) == manifest['source_inventory_sha256'], 'source_inventory_pin')
    for key, rel in {'worker_sha256':'diag/worker.py', 'core_sha256':'diag/core.py', 'host_sha256':'host.py',
                     'support_sha256':'support/worker.py', 'supervisor_sha256':'support/supervised_invocation.py',
                     'r6_summary_sha256':'support/r6_summary.py', 'r6_host_sha256':'support/r6_host.py',
                     'r8_summary_sha256':'support/r8_summary.py'}.items():
        need(sha(root/rel) == manifest[key], 'support_drift')
    need(sha(Path(__file__).resolve()) == manifest['host_sha256'] and manifest['support_sha256'] == SUPPORT_SHA
         and manifest['supervisor_sha256'] == SUPERVISOR_SHA, 'host_support_pin')
    ref = Path(REFERENCE)
    actual = {p.name: sha(p) for p in ref.iterdir() if p.name.startswith('hymem.sqlite')}
    need(actual == manifest['reference_sha256'], 'reference_drift')
    return manifest

def inspect(root, cid, expected, manifest):
    helper = types.ModuleType('r9_summary_host_support')
    path = root/'support/r6_host.py'
    helper.__file__ = str(path)
    exec(compile(read(path), str(path), 'exec'), helper.__dict__)
    helper.run = run
    state = helper.inspect(cid, expected)
    need(state['status'] in ('created','running','exited','dead','paused','restarting','removing'), 'container_status')
    return state

def project(value):
    """Closed allowlist: no provider bodies, arbitrary strings, paths, or logs."""
    out = {}
    for key in ('provider_calls','completion_calls','http_attempts','selected_messages','selected_chars','exit_code','pid'):
        if type(value.get(key)) is int and value[key] >= 0:
            out[key] = value[key]
    if type(value.get('returncode')) is int:
        out['returncode'] = value['returncode']
    for key in ('reference_unchanged','source_unchanged','clients_closed','connections_closed','threads_clean',
                'offline_clones_unchanged','oom_killed','configuration_verified','credential_mount',
                'safe_to_continue','terminal_receipt_written','child_reaped','group_absent_after_reap','cleanup_complete'):
        if type(value.get(key)) is bool:
            out[key] = value[key]
    for key in ('manifest_sha256','container_id'):
        if type(value.get(key)) is str and HEX.fullmatch(value[key]):
            out[key] = value[key]
    if value.get('status') in ('offline_passed','review_pending','error','completed','failed','timeout','cancelled',
                              'live_passed','live_failed','created','running','exited','dead','paused','restarting','removing'):
        out['status'] = value['status']
    if value.get('failure_phase') in ('preflight','normal','source_exact_repair','explicit_recovery',
                                      'control_0','control_1','control_2','control_3','audit'):
        out['failure_phase'] = value['failure_phase']
    return out

def action(root, mode, pin, status=False):
    manifest = verify(root, pin)
    expected = plan(root, mode, pin)
    receipt, results, intent = root/(mode+'-launch.json'), root/(mode+'-results'), root/(mode+'-intent.json')
    if status:
        saved = json.loads(read(receipt))
        need(saved['manifest_sha256'] == pin and saved['mode'] == mode, 'receipt_pin')
        out = project({**saved, **inspect(root, saved['container_id'], expected, manifest)})
        for name in (mode+'.json','supervisor.json'):
            if (results/name).exists():
                out[name] = project(json.loads(read(results/name)))
        return out
    need(not receipt.exists() and not results.exists() and not intent.exists(), 'already_launched')
    if mode == 'live':
        prior = json.loads(read(root/'offline-launch.json'))
        need(prior['manifest_sha256'] == pin and prior['mode'] == 'offline', 'offline_binding')
        state = inspect(root, prior['container_id'], plan(root,'offline',pin), manifest)
        need(state['status'] == 'exited' and state['exit_code'] == 0 and state['pid'] == 0
             and state['oom_killed'] is False, 'offline_process')
        proof = json.loads(read(root/'offline-results/offline.json'))
        need(proof['status'] == 'offline_passed' and proof['manifest_sha256'] == pin
             and proof['capsule_root'] == str(root) and proof['variant'] == manifest['variant']
             and all(proof[k] == 0 for k in ('provider_calls','completion_calls','http_attempts'))
             and all(proof[k] is True for k in ('reference_unchanged','source_unchanged','clients_closed',
                  'connections_closed','threads_clean','offline_clones_unchanged')), 'offline_proof')
        info = Path(KEY).lstat()
        need(Path(KEY).resolve() == Path(KEY) and stat.S_ISREG(info.st_mode) and info.st_uid == 1000
             and stat.S_IMODE(info.st_mode) == 0o600, 'credential_metadata')
    for origin, _ in expected['mounts'].values():
        path = Path(origin)
        if path in (root/'work', results):
            continue
        need(path.exists() and path.resolve() == path, 'mount_path')
    if mode == 'live':
        info = (root/'work').lstat()
        need(stat.S_ISDIR(info.st_mode) and info.st_uid == 1000 and stat.S_IMODE(info.st_mode) == 0o700, 'work_permissions')
    save(intent, {'manifest_sha256':pin,'mode':mode})
    if mode == 'offline':
        (root/'work').mkdir(mode=0o700)
        os.chown(root/'work',1000,1000)
    results.mkdir(mode=0o700)
    os.chown(results,1000,1000)
    cid = run(expected['command'])
    need(HEX.fullmatch(cid), 'container_id')
    state = inspect(root,cid,expected,manifest)
    need(state['status'] == 'created', 'created_container')
    save(receipt, {'manifest_sha256':pin,'mode':mode,'container_id':cid})
    verify(root,pin)
    need(run(['docker','start',cid]) == cid, 'container_start')
    return project({'manifest_sha256':pin,**inspect(root,cid,expected,manifest)})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('seal','offline','live','status-offline','status-live'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--tree',type=Path)
    parser.add_argument('--reference-json',type=Path)
    parser.add_argument('--source-inventory-sha256')
    parser.add_argument('--variant',choices=tuple(ROOTS))
    parser.add_argument('--pin')
    args = parser.parse_args()
    if args.action == 'seal':
        need(args.tree and args.reference_json and args.variant and args.source_inventory_sha256,'seal_arguments')
        value = seal(args.tree,args.root,json.loads(read(args.reference_json)),args.variant,args.source_inventory_sha256)
    else:
        need(args.pin,'pin_required')
        value = action(args.root,args.action.removeprefix('status-'),args.pin,args.action.startswith('status-'))
    print(json.dumps(value,sort_keys=True))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'host_failed':True,'exception_type':type(exc).__name__}))
        raise SystemExit(1)
