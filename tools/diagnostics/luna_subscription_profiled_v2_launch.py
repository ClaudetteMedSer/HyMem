"""One-shot, source-pinned headless launcher for four attributed Luna questions.

Preparation verifies the frozen inputs without inference. A launch attempt is
consumed before systemd dispatch; an ambiguous dispatch must be inspected, not
retried.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

CANDIDATE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate')
DATASET = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json')
INVENTORY = Path('/home/atta/.hymem-luna-lme-v2-yeds3h_l/headless-source-map.json')
BINARY = Path('/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex')
HOST_ROOT = Path('/home/atta')
HOST_UID = 1000
ROOT_PREFIX = '.hymem-luna-lme-attributed-'
UNIT_PREFIX = 'hymem-luna-lme-attributed-'
STAGED_PREFIX = '.hymem-luna-attributed-bundle-'
PINS = {
    'luna_subscription_lme_profiled_v2.py': '53628ac7e9c6107bb68d1cd4ebdf42d5a129b96c4d96360d4e7a6525be7bc739',
    'luna_subscription_lme_warm_v2.py': '3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567',
    'luna_stage_accounting.py': '800ef9baedc9d68093b3160cc324ef17528dd0b89f7a734f220217b07323fba2',
    'codex_subscription_warm_v2.py': '9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593',
    'codex_subscription_concurrent_v2.py': 'cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0',
    'luna_subscription_pilot.py': '0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0',
    'codex_subscription.py': '387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491',
    'headless-source-map.json': '852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb',
}
DATASET_SHA = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
LIMITS = {
    'campaign-turns': 8012, 'campaign-known-tokens': 48160000,
    'campaign-seconds': 14400, 'question-turns': 2000,
    'question-known-tokens': 12000000, 'question-seconds': 12600,
    'canary-turns': 12, 'canary-known-tokens': 160000,
    'canary-seconds': 600, 'indexing-seconds': 10800,
    'questions': 4, 'workers': 4,
    'warm-max-requests': 16, 'warm-max-age-seconds': 300,
}
HEX = re.compile(r'[0-9a-f]{64}\Z')
ROOT_NAME = re.compile(r'\.hymem-luna-lme-attributed-([A-Za-z0-9_-]{8,})\Z')
STAGED_NAME = re.compile(r'\.hymem-luna-attributed-bundle-([A-Za-z0-9_-]{8,})\Z')


def check(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w', encoding='ascii') as output:
        json.dump(value, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())


def unit_for(root):
    match = ROOT_NAME.fullmatch(root.name)
    check(match is not None, 'root_invalid')
    return UNIT_PREFIX + match.group(1) + '.service'


def verify_sources(root):
    for name, expected in PINS.items():
        path = root / name
        check(path.is_file() and not path.is_symlink() and sha(path) == expected,
              'source_pin_invalid')
    spec = importlib.util.spec_from_file_location('verified_profiled_launcher_runner',
                                                  root / 'luna_subscription_lme_profiled_v2.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    files, *_ = module.warm_runner.load_verified(
        candidate=CANDIDATE, inventory_stamp=root / 'headless-source-map.json',
        inventory_sha256=PINS['headless-source-map.json'], dataset=DATASET,
        dataset_sha256=DATASET_SHA, binary=BINARY,
        base_path=root / 'codex_subscription.py',
        concurrent_path=root / 'codex_subscription_concurrent_v2.py',
        warm_path=root / 'codex_subscription_warm_v2.py')
    check(files == 508, 'inventory_incomplete')
    module.collector.verify_candidate(CANDIDATE)


def host_admission():
    check(sys.platform == 'linux' and os.getuid() == HOST_UID, 'wrong_host_user')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    check(int(memory['MemAvailable'].split()[0]) * 1024 >= 6 * 1024**3, 'memory_floor')
    check(shutil.disk_usage(HOST_ROOT).free >= 20 * 1024**3, 'disk_floor')
    units = subprocess.run(['/usr/bin/systemctl', '--user', 'list-units', 'hymem-luna*',
        '--all', '--no-pager', '--plain', '--no-legend'], capture_output=True,
        text=True, timeout=10, check=True)
    for line in units.stdout.splitlines():
        fields = line.split()
        check(len(fields) >= 4 and fields[0].startswith('hymem-luna')
              and fields[0].endswith('.service'), 'unit_list_invalid')
        if fields[2] == 'active':
            check(fields[3] == 'exited', 'prior_unit_running')
            status = subprocess.run(['/usr/bin/systemctl', '--user', 'show', fields[0],
                '--property=MainPID,ControlGroup', '--no-pager'], capture_output=True,
                text=True, timeout=10, check=True)
            properties = dict(row.split('=', 1) for row in status.stdout.splitlines()
                              if '=' in row)
            check(properties.get('MainPID') == '0' and properties.get('ControlGroup') == '',
                  'prior_unit_not_empty')
        else:
            check(fields[2] in {'inactive', 'failed'} and fields[3] not in
                  {'running', 'start', 'stop'}, 'prior_unit_running')


def receipt_for(root):
    unit = unit_for(root)
    return {'schema': 'luna-profiled-launch-v2', 'root': str(root), 'unit': unit,
        'expected_cgroup': '/user.slice/user-1000.slice/user@1000.service/app.slice/' + unit,
        'output': 'run', 'candidate': str(CANDIDATE), 'dataset': str(DATASET),
        'dataset_sha256': DATASET_SHA,
        'inventory_stamp': str(root / 'headless-source-map.json'),
        'inventory_sha256': PINS['headless-source-map.json'],
        'runner_sha256': PINS['luna_subscription_lme_profiled_v2.py'],
        'source_sha256': PINS, 'binary': str(BINARY), 'binary_sha256': sha(BINARY),
        'runtime_max_seconds': 14530, 'timeout_stop_seconds': 10,
        'memory_max_bytes': 4294967296, 'cpu_quota_percent': 200, 'tasks_max': 128,
        'oom_policy': 'kill', 'kill_mode': 'control-group', 'restart': 'no',
        'remain_after_exit': True, 'model': 'gpt-6-luna',
        'subscription_only': True, 'reported_quota_floor_percent': 25,
        'limits': LIMITS}


def command(root, receipt):
    args = ['/usr/bin/systemd-run', '--user', '--quiet', '--unit', receipt['unit'],
        '--property=Type=exec', '--property=Restart=no', '--property=KillMode=control-group',
        '--property=RemainAfterExit=yes', '--property=RuntimeMaxSec=14530s',
        '--property=TimeoutStopSec=10s', '--property=MemoryMax=4294967296',
        '--property=CPUQuota=200%', '--property=TasksMax=128', '--property=OOMPolicy=kill',
        '--property=UMask=0077', '--property=WorkingDirectory=' + str(root / 'empty'),
        '--property=StandardOutput=file:' + str(root / 'safe-terminal.json'),
        '--property=StandardError=file:' + str(root / 'private-launch-stderr.log'),
        '/usr/bin/env', '-i', 'HOME=/home/atta', 'PATH=/usr/local/bin:/usr/bin:/bin',
        'TMPDIR=' + str(root / 'tmp'), '/usr/bin/python3', '-I', '-B',
        str(root / 'luna_subscription_lme_profiled_v2.py')]
    options = {'binary': str(BINARY), 'base-transport': str(root / 'codex_subscription.py'),
        'concurrent-transport': str(root / 'codex_subscription_concurrent_v2.py'),
        'warm-transport': str(root / 'codex_subscription_warm_v2.py'),
        'candidate': str(CANDIDATE), 'inventory-stamp': str(root / 'headless-source-map.json'),
        'inventory-sha256': PINS['headless-source-map.json'], 'dataset': str(DATASET),
        'dataset-sha256': DATASET_SHA, 'output-dir': str(root / 'run'), **LIMITS}
    for key, value in options.items():
        args.extend(['--' + key, str(value)])
    return args


def main(argv=None):
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--launch-root')
    parser.add_argument('--staged-root')
    parser.add_argument('--receipt-sha256')
    args = parser.parse_args(argv)
    root = None
    try:
        check((args.prepare and args.receipt_sha256 is None and args.staged_root is not None) or
              (not args.prepare and args.staged_root is None and isinstance(args.receipt_sha256, str) and
               HEX.fullmatch(args.receipt_sha256) is not None), 'arguments_invalid')
        host_admission()
        if args.prepare:
            staged = Path(args.staged_root)
            check(staged.is_absolute() and staged.parent == HOST_ROOT and
                  STAGED_NAME.fullmatch(staged.name) is not None and
                  staged.is_dir() and not staged.is_symlink() and
                  staged.stat().st_uid == HOST_UID and not staged.stat().st_mode & 0o077,
                  'staged_root_invalid')
            root = Path(tempfile.mkdtemp(prefix=ROOT_PREFIX, dir=HOST_ROOT))
            root.chmod(0o700)
            for name in PINS:
                source = INVENTORY if name == 'headless-source-map.json' else staged / name
                check(source.is_file() and not source.is_symlink() and sha(source) == PINS[name],
                      'staged_pin_invalid')
                shutil.copyfile(source, root / name)
                (root / name).chmod(0o600)
            for name in ('empty', 'tmp'):
                (root / name).mkdir(mode=0o700)
            verify_sources(root)
            receipt = receipt_for(root)
            write_once(root / 'launch-receipt.json', receipt)
            print(json.dumps({'prepared': True, 'launched': False, 'root': str(root),
                'unit': receipt['unit'], 'expected_cgroup': receipt['expected_cgroup'],
                'receipt_sha256': sha(root / 'launch-receipt.json'), 'model_calls': 0}))
            return 0
        root = Path(args.launch_root)
        check(root.parent == HOST_ROOT and ROOT_NAME.fullmatch(root.name)
              and root.is_dir() and not root.is_symlink()
              and root.stat().st_uid == HOST_UID and not root.stat().st_mode & 0o077,
              'root_invalid')
        path = root / 'launch-receipt.json'
        check(path.is_file() and not path.is_symlink() and sha(path) == args.receipt_sha256,
              'receipt_pin_invalid')
        receipt = json.loads(path.read_text(encoding='ascii'))
        check(receipt == receipt_for(root), 'receipt_invalid')
        check(not (root / 'run').exists(), 'output_already_exists')
        verify_sources(root)
        write_once(root / 'launch-attempt.json', {'receipt_sha256': args.receipt_sha256,
                                                 'one_shot': True})
        started = subprocess.run(command(root, receipt), capture_output=True, timeout=20)
        write_once(root / 'launch-command-result.json', {'returncode': started.returncode})
        print(json.dumps({'launch_command_returncode': started.returncode,
            'root': str(root), 'unit': receipt['unit'],
            'receipt_sha256': args.receipt_sha256, 'never_retry': True}))
        return 0 if started.returncode == 0 else 1
    except Exception:
        print(json.dumps({'ok': False, 'stage': 'prepare' if args.prepare else 'launch',
            'root': str(root) if root else None, 'reason': 'admission_or_launch_failed',
            'never_retry_launch': not args.prepare}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
