"""One-shot private host orchestration for the reviewed paired diagnostic."""
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

HOME_ROOT = Path('/home/atta')
HOST_UID = 1000
ORIGINAL = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate')
OLD = Path('/home/atta/.hymem-luna-lme-capacity-jzipeyhr')
BINARY = Path('/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex')
ROOT_RE = re.compile(r'\.hymem-luna-grounding-probe-[A-Za-z0-9_-]{8,}\Z')
PINS = {
    'luna_grounding_probe.py': 'a4e50e05d2a5eac8199fb6cf6dee950f7119f4f53d7859a1dbac155846932214',
    'luna_grounding_candidate.py': '51e03ff2bec29f7517db027b103163698c18b9d34603ab950a36f5d09255e96a',
    'luna_grounding_cases.py': '56147b137c71236aa16afc8b0f2f413b90b255c79cbcb370fed50ef71879103c',
    'luna_subscription_capacity_launch.py': 'b848ad37f66680c8e423876b0aed2f4f99bdd6ee896247f1faa9461a50e43983',
    'luna_subscription_lme_warm_v2.py': '3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567',
    'luna_subscription_pilot.py': '0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0',
    'codex_subscription_warm_v2.py': '9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593',
    'codex_subscription_concurrent_v2.py': 'cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0',
    'codex_subscription.py': '387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491',
    'headless-source-map.json': '852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb',
    'grounded_prompt.py': '17ee5017c54a1ba255e0220aa8e246766127cb71ced65fb2e8c812034f6e184c',
}
NEW = {'luna_grounding_probe.py', 'luna_grounding_candidate.py', 'luna_grounding_cases.py',
       'luna_subscription_capacity_launch.py'}


def require(value, code):
    if not value:
        raise RuntimeError(code)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected):
    require(path.is_file() and not path.is_symlink() and digest(path) == expected, 'source_pin_invalid')


def module(path, expected, name):
    checked(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def write_once(path, value):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def write_bytes_once(path, value):
    require(value is None or isinstance(value, bytes), 'dispatch_output_invalid')
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(value or b'')
        stream.flush()
        os.fsync(stream.fileno())


def copy_pinned_file(source, destination, expected):
    checked(source, expected)
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as target, source.open('rb') as original:
        shutil.copyfileobj(original, target)
        target.flush()
        os.fsync(target.fileno())
    checked(destination, expected)


def valid_root(root):
    require(root.parent == HOME_ROOT and ROOT_RE.fullmatch(root.name) and root.is_dir()
            and not root.is_symlink() and root.stat().st_uid == HOST_UID
            and not root.stat().st_mode & 0o077, 'root_invalid')


def valid_workdirs(root):
    for name in ('empty', 'tmp'):
        path = root / name
        require(path.is_dir() and not path.is_symlink()
                and path.stat().st_uid == HOST_UID
                and not path.stat().st_mode & 0o077
                and not any(path.iterdir()), 'workdir_invalid')


def unit_for(root):
    return root.name[1:] + '.service'


def verify(root):
    valid_root(root)
    valid_workdirs(root)
    for name, expected in PINS.items():
        checked(root / name, expected)
    builder = module(root / 'luna_grounding_candidate.py', PINS['luna_grounding_candidate.py'], 'verified_probe_builder')
    builder.pilot_module().verify_inventory(ORIGINAL, root / 'headless-source-map.json', PINS['headless-source-map.json'])
    require(builder.derive_prompt((ORIGINAL / builder.PROMPT_RELATIVE).read_bytes()) ==
            (root / 'grounded_prompt.py').read_bytes(), 'derived_prompt_invalid')
    return module(root / 'luna_subscription_capacity_launch.py', PINS['luna_subscription_capacity_launch.py'], 'verified_probe_admission')


def command(root, unit):
    argv = ['/usr/bin/systemd-run', '--user', '--quiet', '--unit', unit,
        '--property=Type=exec', '--property=Restart=no', '--property=KillMode=control-group',
        '--property=RemainAfterExit=yes', '--property=RuntimeMaxSec=1830s', '--property=TimeoutStopSec=10s',
        '--property=TasksMax=256', '--property=MemoryMax=4294967296', '--property=CPUQuota=200%',
        '--property=OOMPolicy=kill', '--property=UMask=0077',
        '--property=WorkingDirectory=' + str(root / 'empty'),
        '--property=StandardOutput=file:' + str(root / 'safe-terminal.json'),
        '--property=StandardError=file:' + str(root / 'private-launch-stderr.log'),
        '/usr/bin/env', '-i', 'HOME=/home/atta', 'PATH=/usr/local/bin:/usr/bin:/bin',
        'TMPDIR=' + str(root / 'tmp'), '/usr/bin/python3', '-I', '-B', str(root / 'luna_grounding_probe.py')]
    options = {'binary': BINARY, 'base-transport': root / 'codex_subscription.py',
        'concurrent-transport': root / 'codex_subscription_concurrent_v2.py',
        'warm-transport': root / 'codex_subscription_warm_v2.py', 'candidate': ORIGINAL,
        'inventory': root / 'headless-source-map.json', 'inventory-sha256': PINS['headless-source-map.json'],
        'grounded-prompt': root / 'grounded_prompt.py', 'grounded-prompt-sha256': PINS['grounded_prompt.py'],
        'output-dir': root / 'run'}
    for key, value in options.items():
        argv.extend(['--' + key, str(value)])
    return argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'launch'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--receipt-sha256')
    args = parser.parse_args()
    root = Path(args.root)
    valid_root(root)
    if args.action == 'prepare':
        for name, expected in PINS.items():
            if name in NEW:
                checked(root / name, expected)
            elif name != 'grounded_prompt.py':
                source = OLD / name
                copy_pinned_file(source, root / name, expected)
        builder = module(root / 'luna_grounding_candidate.py', PINS['luna_grounding_candidate.py'], 'preparation_builder')
        raw = builder.derive_prompt((ORIGINAL / builder.PROMPT_RELATIVE).read_bytes())
        with (root / 'grounded_prompt.py').open('xb') as stream:
            stream.write(raw)
        (root / 'grounded_prompt.py').chmod(0o600)
        for name in ('empty', 'tmp'):
            (root / name).mkdir(mode=0o700)
        verify(root).host_admission()
        receipt = {'schema': 'luna-grounding-probe-launch-v1', 'root': str(root), 'unit': unit_for(root),
            'source_pins': PINS, 'host_sha256': digest(Path(__file__)), 'binary_sha256': digest(BINARY),
            'turns': 192, 'known_tokens': 2000000, 'seconds': 1800, 'units': 44,
            'command': command(root, unit_for(root))}
        write_once(root / 'launch-receipt.json', receipt)
        print(json.dumps({'prepared': True, 'model_calls': 0, 'root': str(root), 'unit': unit_for(root),
            'receipt_sha256': digest(root / 'launch-receipt.json')}))
    else:
        require(args.receipt_sha256 and digest(root / 'launch-receipt.json') == args.receipt_sha256, 'receipt_pin_invalid')
        receipt = json.loads((root / 'launch-receipt.json').read_text())
        verify(root).host_admission()
        require(receipt['root'] == str(root) and receipt['unit'] == unit_for(root)
                and receipt['source_pins'] == PINS and receipt['host_sha256'] == digest(Path(__file__))
                and receipt['binary_sha256'] == digest(BINARY)
                and receipt['command'] == command(root, unit_for(root)) and not (root / 'run').exists(), 'receipt_invalid')
        write_once(root / 'launch-attempt.json', {'receipt_sha256': args.receipt_sha256, 'one_shot': True})
        try:
            result = subprocess.run(command(root, unit_for(root)), capture_output=True, timeout=20)
        except subprocess.TimeoutExpired as exc:
            write_bytes_once(root / 'private-dispatch-stdout.bin', exc.stdout)
            write_bytes_once(root / 'private-dispatch-stderr.bin', exc.stderr)
            raise
        write_bytes_once(root / 'private-dispatch-stdout.bin', result.stdout)
        write_bytes_once(root / 'private-dispatch-stderr.bin', result.stderr)
        write_once(root / 'launch-command-result.json', {'returncode': result.returncode})
        print(json.dumps({'launch_returncode': result.returncode, 'never_retry': True, 'unit': unit_for(root)}))
        return 0 if result.returncode == 0 else 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print(json.dumps({'ok': False, 'reason': 'host_preparation_or_dispatch_failed', 'never_retry_launch': True}))
        raise SystemExit(1)
