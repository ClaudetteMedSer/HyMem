"""One-shot, network-none postflight. Run only on the private benchmark host.

The reviewed seal is required: host_sha256, install_sha256, result_sha256,
live_container_id, source_sha256. This program has no remote/install action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/cold-replay-dream-v1/episode-shadow-dream-v1')
STAGE = ROOT / 'episode-shadow-postflight-v3'
CHECKER_SHA = '715b498b3a9c4d4b5bcf164eee78e6e60d901aa1527318b6932c43c3bcde6bc1'
SEMANTIC_SHA = '5f359e9b3302971f582966b6a3b016cf0d3e6c375d20ff26fa840a412c5cdbc0'
PROFILE_SHA = '1e1e69e036999106d4ea5bd7cf65aaca5c3cc416741a08ad4d3a0374059ae71a'
AUDIT_SHA = 'e2efe365c5aedbbe88d86d521dd37b821759afc5fb21c21a6662e6a8ce567f42'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
HEX = re.compile(r'[0-9a-f]{64}\Z')


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(path, mode=None):
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not path.is_symlink(), 'file_invalid')
    need(mode is None or stat.S_IMODE(info.st_mode) == mode, 'file_mode_invalid')


def read_seal(path, expected):
    regular(path, 0o400)
    need(bool(HEX.fullmatch(expected)) and sha(path) == expected, 'seal_pin_drift')
    seal = json.loads(path.read_bytes())
    fields = {'host_sha256', 'install_sha256', 'result_sha256', 'live_container_id', 'source_sha256'}
    need(isinstance(seal, dict) and set(seal) == fields, 'seal_fields_invalid')
    need(all(isinstance(v, str) and HEX.fullmatch(v) for v in seal.values()), 'seal_values_invalid')
    return seal


def load_host(expected):
    import importlib.util
    path = ROOT / 'claim_conflict_episode_shadow_dream_host.py'
    regular(path, 0o400)
    need(sha(path) == expected, 'host_pin_drift')
    spec = importlib.util.spec_from_file_location('postflight_campaign_host', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configure_parent()


def clean(state):
    need(state['status'] == 'exited' and state['pid'] == 0 and state['exit_code'] == 0
         and state['oom_killed'] is False and state['configuration_verified'] is True,
         'container_not_clean')


def closed_source(path, expected):
    regular(path)
    need(sha(path) == expected, 'closed_source_pin_drift')
    for suffix in ('-wal', '-journal'):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            regular(sidecar)
            need(sidecar.stat().st_size == 0, 'closed_source_unsealed_sidecar')


def cleanup(helper, cid):
    need(isinstance(cid, str) and HEX.fullmatch(cid), 'cleanup_container_id_invalid')
    # This exact ID came from our create; never stop a name or an unvalidated ID.
    try:
        helper.run(['docker', 'stop', '--time', '10', cid], 40, 'postflight_stop')
    finally:
        items = json.loads(helper.run(['docker', 'inspect', cid], 30, 'postflight_cleanup_inspect'))
        need(isinstance(items, list) and len(items) == 1 and items[0].get('Id') == cid,
             'cleanup_identity_invalid')
        state = items[0]['State']
        need(state['Status'] in {'exited', 'created'} and state['Pid'] == 0
             and state['Running'] is False, 'cleanup_not_terminal')


def configure(helper, source_sha256):
    need(isinstance(source_sha256, str) and HEX.fullmatch(source_sha256), 'source_seal_invalid')
    mounts = [(str(ROOT / 'candidate'), '/candidate', False),
              (str(STAGE / 'claim_conflict_episode_shadow_postflight_v3.py'), '/diag/postflight.py', False),
              (str(STAGE / 'claim_conflict_episode_shadow_semantic_public.json'), '/diag/semantic-config-hashes.json', False),
              (str(STAGE / 'claim_conflict_episode_shadow_embedding_public.json'), '/diag/embedding-public.json', False),
              (str(STAGE / 'claim_conflict_store_audit.py'), '/diag/claim_conflict_store_audit.py', False),
              (str(ROOT / 'work/live'), '/private-dream', False),
              (str(ROOT / 'reference.sqlite'), '/reference/source.sqlite', False),
              (str(STAGE / 'campaign'), '/campaign', False),
              (str(STAGE / 'work'), '/work', True),
              (str(helper.RUNTIME), '/home/node/hymem-env', False)]
    command = ['docker', 'create', '--name', 'hymem-private-dream-episode-shadow-postflight-v3',
               '--pull', 'never', '--init', '--network', 'none', '--user', '1000:1000',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '128', '--memory', '2g', '--cpus', '2',
               '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m']
    for src, dst, rw in mounts:
        command += ['--mount', 'type=bind,src=' + src + ',dst=' + dst + ('' if rw else ',readonly')]
    command += ['--workdir', '/candidate', '--entrypoint', '/home/node/hymem-env/bin/python3',
                IMAGE, '-I', '-B', '/diag/postflight.py', '--worker-result', '/campaign/worker-result.json',
                '--source-sha256', source_sha256, '--embedding-profile', '/diag/embedding-public.json',
                '--embedding-profile-sha256', PROFILE_SHA,
                '--semantic-config-hashes', '/diag/semantic-config-hashes.json',
                '--semantic-config-sha256', SEMANTIC_SHA]
    return command, mounts


def safe_report(value):
    # The pinned checker emits only status, hashes, gates and bounded counters.
    def check(item):
        if isinstance(item, dict):
            return all(isinstance(k, str) and check(v) for k, v in item.items())
        if isinstance(item, list):
            return all(check(v) for v in item)
        return (item is None or type(item) is bool or
                (type(item) is int and 0 <= item <= 100000000) or
                (isinstance(item, str) and (HEX.fullmatch(item) or item in
                 {'pass', 'fail', 'error', 'completed', 'ok', 'ValueError', 'RuntimeError',
                  'TypeError', 'OperationalError', 'DatabaseError', 'OSError', 'KeyError', 'Exception'})))
    need(isinstance(value, dict) and value.get('status') in {'pass', 'fail', 'error'} and check(value),
         'checker_output_invalid')
    return value


def ready(seal):
    need(os.geteuid() == 1000, 'wrong_host_user')
    need(STAGE.is_dir() and not STAGE.is_symlink() and stat.S_IMODE(STAGE.stat().st_mode) == 0o700,
         'stage_not_private')
    for name, expected in [('claim_conflict_episode_shadow_postflight_v3.py', CHECKER_SHA),
                           ('claim_conflict_store_audit.py', AUDIT_SHA),
                           ('claim_conflict_episode_shadow_embedding_public.json', PROFILE_SHA),
                           ('claim_conflict_episode_shadow_semantic_public.json', SEMANTIC_SHA)]:
        regular(STAGE / name, 0o400)
        need(sha(STAGE / name) == expected, 'checker_pin_drift')
    host = load_host(seal['host_sha256'])
    shared, proof, helper = host.dependencies()
    controller = host.controller()
    receipt = controller.installed(shared, proof, helper)
    need(receipt.get('candidate_sha256') == '5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576'
         and receipt.get('phase1_sha256') == 'bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac'
         and receipt.get('source_files') == 481, 'candidate_identity_drift')
    for name in ('install', 'result'):
        regular(ROOT / (name + '.json'), 0o600)
        need(sha(ROOT / (name + '.json')) == seal[name + '_sha256'], 'campaign_pin_drift')
    campaign = helper.read_json(ROOT / 'result.json')
    live = campaign.get('stages', {}).get('live', {})
    need(campaign.get('status') == 'completed' and campaign.get('paid_live_runs_started') == 1
         and live.get('container_id') == seal['live_container_id']
         and live.get('metadata', {}).get('status') == 'completed', 'campaign_not_successful')
    _, live_mounts = controller.configure(helper, 'live')
    clean(controller.inspect(shared, helper, seal['live_container_id'], 'live', live_mounts))
    source = ROOT / 'work/live/hymem.sqlite'
    for directory in (ROOT / 'work', ROOT / 'work/live'):
        need(directory.is_dir() and not directory.is_symlink()
             and stat.S_IMODE(directory.stat().st_mode) == 0o700, 'closed_source_directory_invalid')
    closed_source(source, seal['source_sha256'])
    need(helper.IMAGE == IMAGE, 'image_pin_drift')
    return host, shared, proof, helper, controller, receipt, campaign, live, source


def run(seal):
    host, shared, proof, helper, controller, receipt, campaign, live, source = ready(seal)
    # Exclusive intent precedes even log capture: interrupted attempts cannot rerun.
    helper.put_json(STAGE / 'intent.json', {'networked_runs_started': 0, 'runs_allowed': 1})
    raw = json.loads(helper.run(['docker', 'logs', seal['live_container_id']], 30, 'private_worker_logs'))
    need(shared.project(raw) == live['metadata'], 'worker_projection_drift')
    os.mkdir(STAGE / 'campaign', 0o700)
    os.mkdir(STAGE / 'work', 0o700)
    helper.put_json(STAGE / 'campaign/worker-result.json', raw)
    helper.put_json(STAGE / 'campaign/result.json', campaign)
    helper.put_json(STAGE / 'campaign/install.json', receipt)
    command, mounts = configure(helper, seal['source_sha256'])
    cid = helper.run(command, 60, 'postflight_create').decode().strip()
    need(isinstance(cid, str) and HEX.fullmatch(cid), 'created_container_id_invalid')
    helper.put_json(STAGE / 'container.json', {'container_id': cid})
    original = shared.configure
    shared.configure = lambda *_args: configure(helper, seal['source_sha256'])
    try:
        need(shared.inspect(helper, cid, 'offline', mounts, host.PHASE1_SHA)['status'] == 'created',
             'postflight_not_created')
        need(helper.run(['docker', 'start', cid], 60, 'postflight_start').decode().strip() == cid,
             'start_identity')
        exit_raw = helper.run(['docker', 'wait', cid], 600, 'postflight_wait')
        need(re.fullmatch(rb'[01]\n?', exit_raw) is not None, 'wait_invalid')
        state = shared.inspect(helper, cid, 'offline', mounts, host.PHASE1_SHA)
        need(state['status'] == 'exited' and state['pid'] == 0 and not state['oom_killed']
             and state['exit_code'] == int(exit_raw), 'postflight_terminal_invalid')
        report = safe_report(json.loads(helper.run(['docker', 'logs', cid], 30, 'postflight_logs')))
        need((report['status'] == 'pass') == (state['exit_code'] == 0), 'verdict_exit_mismatch')
        closed_source(source, seal['source_sha256'])
        for name in ('install', 'result'):
            need(sha(ROOT / (name + '.json')) == seal[name + '_sha256'], 'campaign_changed')
        controller.installed(shared, proof, helper)
        helper.put_json(STAGE / 'result.json', report)
        return report
    except BaseException:
        cleanup(helper, cid)
        raise
    finally:
        shared.configure = original


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--seal-sha256', required=True)
    parser.add_argument('--ready', action='store_true', required=True)
    parser.add_argument('--prepare-seal', action='store_true')
    args = parser.parse_args()
    if args.prepare_seal:
        value = json.load(sys.stdin)
        fields = {'host_sha256', 'install_sha256', 'result_sha256', 'live_container_id', 'source_sha256'}
        need(isinstance(value, dict) and set(value) == fields
             and all(isinstance(v, str) and HEX.fullmatch(v) for v in value.values()), 'seal_fields_invalid')
        ready(value)  # Verify worker PID zero, successful exit/configuration, campaign pins and closed source first.
        need(args.seal == STAGE / 'seal.json', 'seal_destination_invalid')
        raw = json.dumps(value, sort_keys=True).encode()
        need(hashlib.sha256(raw).hexdigest() == args.seal_sha256, 'seal_pin_drift')
        fd = os.open(args.seal, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({'status': 'sealed_not_launched', 'seal_sha256': args.seal_sha256}))
        return 0
    report = run(read_seal(args.seal, args.seal_sha256))
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print('{"status":"error"}')
        raise SystemExit(1)


