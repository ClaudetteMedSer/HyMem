"""Offline-reviewable one-shot host preparation for the semantic diagnostic.

Preparation invokes read-only host status commands but no inference. Only the
explicit ``launch`` action dispatches the previously sealed systemd command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

TASK_HOME_ROOT = Path('/home/atta')
UID = 1000
PREFIX = '.hymem-luna-semantic-probe-'
NAME = re.compile(r'\.hymem-luna-semantic-probe-([A-Za-z0-9_-]{8,})\Z')
HEX = re.compile(r'[0-9a-f]{64}\Z')
BINARY = Path('/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex')
OBSERVED = Path('/home/atta/.hymem-luna-lme-observed-4e8qycex')
OLD_CANDIDATE = OBSERVED / 'candidate'
OLD_MAP = OBSERVED / 'headless-grounding-source-map.json'
OLD_EVIDENCE = OBSERVED / 'run/private-canary-evidence.json'
OLD_RECEIPT = OBSERVED / 'launch-receipt.json'
OLD_RESULT = OBSERVED / 'run/private-result.json'
OLD_DEEPSEEK = '523469c234e48ff01d944de22a149251d53f0477aa16133b438bfee2f7030588'

ACCEPTED = {
    'tools/diagnostics/luna_semantic_probe.py': '3ccb7ec12f93f8502fe8ed1448a071ac977dd334d17821f661913d634c27b334',
    'tools/diagnostics/luna_semantic_candidate.py': 'a10ee5c5a1ba4f6a2694a88570a399c5fe081db02f0cb0ecfa5b92e5db06d7b2',
    'tools/diagnostics/luna_semantic_cases.py': '8876e94e6c5d3284d4d361f0238e0aef507288e2702b5df77ee706fd4c712d8a',
    'benchmarks/luna_semantic_canary.py': '3d132415573cb46c2b15a69a56776ea69b3410b350a8fdb8ba6a1f8cc85658af',
    'benchmarks/luna_semantic_stage_accounting.py': '4c654599a979e51aeb9f0985b091cd8ef38a639424f197c412da45e1f3270d30',
    'benchmarks/codex_subscription_warm_v3.py': '0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d',
    'benchmarks/codex_subscription_warm_v2.py': '9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593',
    'benchmarks/codex_subscription_concurrent_v2.py': 'cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0',
    'benchmarks/codex_subscription.py': '387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491',
    'hymem/extraction/grounding.py': 'dd1a49b56abf569b4476a0b735e67e72a86723bf88e3a4739998aceeabe19c18',
    'hymem/extraction/grounding_gate.py': 'bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8',
}
NEW = ('tools/diagnostics/luna_semantic_probe_host.py',
       'tools/diagnostics/luna_semantic_probe_run.py',
       'tools/diagnostics/luna_semantic_probe_progress.py')
INVENTORY_SHA = '11ca4cdbba18e4b7e4b56d444062e39b789820b2f0055a32898bbd0b3a1e4664'
RETAINED = {
    'evidence': 'a1fd2d45c4cf50bf483927b07dff5cad145ae3ce8a582f19d13dc1b11a92943c',
    'receipt': 'ec4b060b2122c742c4e0dac6d95037ae04693ee65d5866523a0f9b4a194fadfc',
    'result': '0730f970ec291473121bde497f855bfa4e49bda2b7b108a60d7efd7e55058147',
}
LIMITS = {'turns': 29, 'known_tokens': 500000, 'seconds': 1800,
          'control_turns': 2, 'control_known_tokens': 100000, 'control_seconds': 240,
          'hybrid_turns': 3, 'hybrid_known_tokens': 160000, 'hybrid_seconds': 600,
          'invocation_seconds': 120, 'workers': 1, 'units': 25,
          'ordinary_replays': 8, 'new_judgments_max': 29}
POLICY = {'runtime_max_seconds': 1930, 'timeout_stop_seconds': 10,
          'tasks_max': 256, 'memory_max_bytes': 4294967296,
          'cpu_quota_percent': 200, 'oom_policy': 'kill',
          'kill_mode': 'control-group', 'restart': 'no', 'remain_after_exit': True,
          'memory_admission_bytes': 6 * 1024**3, 'disk_admission_bytes': 20 * 1024**3}


def need(ok, code):
    if not ok:
        raise ValueError(code)


def strict_equal(left, right) -> bool:
    try:
        return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
            right, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError):
        return False


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def pinned(path: Path, expected: str):
    need(regular(path) and sha(path) == expected, 'source_pin_invalid')


def write_once(path: Path, value) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def root_valid(root: Path) -> bool:
    if not root.is_absolute() or root.parent != TASK_HOME_ROOT or NAME.fullmatch(root.name) is None:
        return False
    try:
        mode = root.lstat().st_mode
        return stat.S_ISDIR(mode) and root.lstat().st_uid == UID and not mode & 0o077
    except FileNotFoundError:
        return False


def unit_for(root: Path) -> str:
    need(root_valid(root), 'root_invalid')
    return 'hymem-luna-semantic-probe-' + NAME.fullmatch(root.name).group(1) + '.service'


def source_pins(code: Path) -> dict[str, str]:
    pins = dict(ACCEPTED)
    for relative in NEW:
        path = code / relative
        need(regular(path), 'new_source_missing')
        pins[relative] = sha(path)
    for relative, expected in pins.items():
        pinned(code / relative, expected)
    return pins


def verify_bundle(root: Path, receipt: dict, *, require_empty_workdirs: bool = True) -> None:
    need(root_valid(root), 'root_invalid')
    code = root / 'code'
    need(code.is_dir() and not code.is_symlink(), 'code_invalid')
    expected = receipt.get('source_sha256')
    need(type(expected) is dict and set(expected) == set(ACCEPTED) | set(NEW)
         and all(expected.get(k) == v for k, v in ACCEPTED.items())
         and all(type(expected.get(k)) is str and HEX.fullmatch(expected[k]) for k in NEW),
         'source_manifest_invalid')
    for path in code.rglob('*'):
        need(not path.is_symlink() and (path.is_dir() or path.is_file()), 'code_path_invalid')
        if path.is_file():
            need(path.relative_to(code).as_posix() in expected, 'code_extra_file')
    for relative, digest in expected.items():
        pinned(code / relative, digest)
    # Revalidate the retained 508-file source tree, not merely its stamp.
    import importlib.util
    builder_path = code / 'tools/diagnostics/luna_semantic_candidate.py'
    spec = importlib.util.spec_from_file_location('pinned_semantic_builder_for_host', builder_path)
    need(spec is not None and spec.loader is not None, 'builder_import_invalid')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    need(regular(OLD_MAP), 'original_map_missing')
    original_stamp = json.loads(OLD_MAP.read_bytes())
    original_mapping = original_stamp.get('source_sha256', original_stamp)
    need(type(original_mapping) is dict and len(original_mapping) == 508 and
         builder.mapping_sha(original_mapping) == builder.ORIGINAL_MAP_SHA256 and
         builder.inventory(OLD_CANDIDATE) == original_mapping,
         'original_candidate_drift')
    pinned(root / 'candidate-source-map.json', INVENTORY_SHA)
    try:
        stamp = json.loads((root / 'candidate-source-map.json').read_bytes())
        mapping = stamp['source_sha256']
        need(type(mapping) is dict and len(mapping) == 510 and
             all(type(k) is str and type(v) is str and HEX.fullmatch(v)
                 and not Path(k).is_absolute() and '..' not in Path(k).parts
                 for k, v in mapping.items()), 'candidate_inventory_invalid')
        actual = set()
        candidate = root / 'candidate'
        need(candidate.is_dir() and not candidate.is_symlink() and
             candidate.parent == root, 'candidate_invalid')
        for path in candidate.rglob('*'):
            need(not path.is_symlink() and (path.is_file() or path.is_dir()),
                 'candidate_extra_or_symlink')
            if path.is_file():
                relative = path.relative_to(candidate).as_posix()
                need(relative in mapping and sha(path) == mapping[relative],
                     'candidate_inventory_drift')
                actual.add(relative)
        need(actual == set(mapping), 'candidate_inventory_incomplete')
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise ValueError('candidate_inventory_invalid') from None
    for name, digest in RETAINED.items():
        pinned(OBSERVED / ('run/private-canary-evidence.json' if name == 'evidence' else
                           'launch-receipt.json' if name == 'receipt' else
                           'run/private-result.json'), digest)
    need(regular(BINARY), 'binary_missing')
    need(sha(BINARY) == receipt.get('binary_sha256'), 'binary_drift')
    for name in ('empty', 'tmp'):
        path = root / name
        need(path.is_dir() and not path.is_symlink() and
             path.stat().st_uid == UID and not path.stat().st_mode & 0o077 and
             (not require_empty_workdirs or not any(path.iterdir())), 'workdir_invalid')


def receipt_for(root: Path, pins: dict[str, str], binary_sha: str) -> dict:
    unit = unit_for(root)
    return {'schema': 'luna-semantic-probe-launch-v1', 'root': str(root),
            'unit': unit, 'expected_cgroup':
            '/user.slice/user-1000.slice/user@1000.service/app.slice/' + unit,
            'code': str(root / 'code'), 'candidate': str(root / 'candidate'),
            'inventory': str(root / 'candidate-source-map.json'),
            'inventory_sha256': INVENTORY_SHA, 'original_candidate': str(OLD_CANDIDATE),
            'original_map': str(OLD_MAP), 'observed_root': str(OBSERVED),
            'retained_sha256': RETAINED, 'source_sha256': pins,
            'binary': str(BINARY), 'binary_sha256': binary_sha,
            'model': 'gpt-6-luna', 'subscription_only': True,
            'quota_floor_percent': 25, 'limits': LIMITS, 'policy': POLICY,
            'schedule': {'controls': 24, 'supported': 12, 'reject': 10,
                         'correction': 2, 'hybrid_last': True},
            'fixture_label_sha256': '511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925',
            'extraction_identity': 'hymem-extraction-contract-sha256-v1:f349e2fa14d1778bc556869d346ca44183c025e779a919b3f15e82f7c3d78d46'}


def command(root: Path, receipt: dict) -> list[str]:
    return ['/usr/bin/systemd-run', '--user', '--quiet', '--unit', receipt['unit'],
            '--property=Type=exec', '--property=Restart=no',
            '--property=KillMode=control-group', '--property=RemainAfterExit=yes',
            '--property=RuntimeMaxSec=1930s', '--property=TimeoutStopSec=10s',
            '--property=TasksMax=256', '--property=MemoryMax=4294967296',
            '--property=CPUQuota=200%', '--property=OOMPolicy=kill',
            '--property=UMask=0077',
            '--property=WorkingDirectory=' + str(root / 'empty'),
            '--property=StandardOutput=null',
            '--property=StandardError=file:' + str(root / 'private-launch-stderr.log'),
            '/usr/bin/env', '-i', 'HOME=/home/atta', 'PATH=/usr/local/bin:/usr/bin:/bin',
            'TMPDIR=' + str(root / 'tmp'), '/usr/bin/python3', '-I', '-B',
            str(root / 'code/tools/diagnostics/luna_semantic_probe_run.py'),
            '--root', str(root), '--receipt-sha256', sha(root / 'launch-receipt.json')]


def host_admission() -> None:
    need(sys.platform == 'linux' and os.getuid() == UID, 'wrong_host_user')
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    need(int(mem['MemAvailable'].split()[0]) * 1024 >= POLICY['memory_admission_bytes'], 'memory_floor')
    need(shutil.disk_usage(TASK_HOME_ROOT).free >= POLICY['disk_admission_bytes'], 'disk_floor')
    unit = subprocess.run(['/usr/bin/systemctl', '--user', 'list-units', 'hymem-luna*',
        '--all', '--plain', '--no-legend', '--no-pager'], capture_output=True, text=True,
        check=True, timeout=10)
    for line in unit.stdout.splitlines():
        fields = line.split()
        need(len(fields) >= 4 and fields[0].startswith('hymem-luna')
             and fields[0].endswith('.service'), 'old_unit_list_invalid')
        if fields[2] == 'active':
            need(fields[3] == 'exited', 'old_luna_running')
            show = subprocess.run(['/usr/bin/systemctl', '--user', 'show', fields[0],
                '--property=MainPID,ControlGroup,NRestarts', '--no-pager'],
                capture_output=True, text=True, check=True, timeout=10)
            properties = dict(s.split('=', 1) for s in show.stdout.splitlines() if '=' in s)
            need(properties.get('MainPID') == '0' and properties.get('ControlGroup') == ''
                 and properties.get('NRestarts') == '0', 'old_luna_not_empty')
        else:
            need(fields[2] in {'inactive', 'failed'} and fields[3] not in
                 {'running', 'start', 'stop'}, 'old_luna_running')
    docker = subprocess.run(['/usr/bin/docker', 'inspect', '--format', '{{.State.Running}}',
        OLD_DEEPSEEK], capture_output=True, text=True, timeout=10)
    need(docker.returncode == 0 and docker.stdout.strip() == 'false', 'old_deepseek_running_or_unverified')


def prepare(staged: Path) -> dict:
    need(sys.platform == 'linux' and os.getuid() == UID, 'wrong_host_user')
    need(staged.is_absolute() and staged.parent == TASK_HOME_ROOT and staged.is_dir()
         and not staged.is_symlink() and staged.stat().st_uid == UID
         and not staged.stat().st_mode & 0o077, 'staged_invalid')
    stage_code = staged / 'code'
    need(stage_code.is_dir() and not stage_code.is_symlink(), 'staged_code_invalid')
    pins = source_pins(stage_code)
    for path in stage_code.rglob('*'):
        need(not path.is_symlink() and (path.is_dir() or
             path.relative_to(stage_code).as_posix() in pins), 'staged_extra')
    root = Path(tempfile.mkdtemp(prefix=PREFIX, dir=TASK_HOME_ROOT))
    root.chmod(0o700)
    shutil.copytree(stage_code, root / 'code')
    for name in ('empty', 'tmp'):
        (root / name).mkdir(mode=0o700)
    from importlib.util import module_from_spec, spec_from_file_location
    source = root / 'code/tools/diagnostics/luna_semantic_candidate.py'
    spec = spec_from_file_location('pinned_semantic_builder', source)
    builder = module_from_spec(spec)
    spec.loader.exec_module(builder)
    proof = builder.prepare(OLD_CANDIDATE, OLD_MAP, root / 'candidate',
                            root / 'candidate-source-map.json', root / 'code')
    need(proof['files'] == 510 and proof['derived_stamp_sha256'] == INVENTORY_SHA,
         'candidate_derivation_invalid')
    receipt = receipt_for(root, pins, sha(BINARY))
    verify_bundle(root, receipt)
    host_admission()
    write_once(root / 'launch-receipt.json', receipt)
    return {'prepared': True, 'launched': False, 'root': str(root),
            'unit': receipt['unit'], 'receipt_sha256': sha(root / 'launch-receipt.json'),
            'model_calls': 0}


def launch(root: Path, receipt_sha: str, *, dispatch=subprocess.run) -> dict:
    need(type(receipt_sha) is str and HEX.fullmatch(receipt_sha) is not None, 'receipt_hash_invalid')
    need(root_valid(root), 'root_invalid')
    path = root / 'launch-receipt.json'
    need(regular(path) and sha(path) == receipt_sha, 'receipt_pin_invalid')
    receipt = json.loads(path.read_bytes())
    need(strict_equal(receipt, receipt_for(root, receipt['source_sha256'], receipt['binary_sha256'])), 'receipt_invalid')
    verify_bundle(root, receipt)
    need(not (root / 'run').exists(), 'output_already_exists')
    host_admission()
    write_once(root / 'launch-admission.json', {
        'receipt_sha256': receipt_sha, 'old_luna_stopped': True,
        'old_deepseek_stopped': True, 'memory_floor_bytes': POLICY['memory_admission_bytes'],
        'disk_floor_bytes': POLICY['disk_admission_bytes'],
        'memory_floor_met': True, 'disk_floor_met': True})
    write_once(root / 'launch-attempt.json', {'receipt_sha256': receipt_sha, 'one_shot': True})
    try:
        result = dispatch(command(root, receipt), capture_output=True, timeout=20)
        for name, raw in [('private-dispatch-stdout.bin', result.stdout),
                          ('private-dispatch-stderr.bin', result.stderr)]:
            fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw or b'')
                stream.flush()
                os.fsync(stream.fileno())
        write_once(root / 'launch-command-result.json', {'returncode': result.returncode})
        return {'launched': result.returncode == 0, 'never_retry': True,
                'launch_returncode': result.returncode, 'unit': receipt['unit']}
    except BaseException:
        return {'launched': False, 'never_retry': True, 'dispatch_ambiguous': True,
                'unit': receipt['unit']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--launch-root')
    parser.add_argument('--staged-root')
    parser.add_argument('--receipt-sha256')
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            need(args.staged_root is not None and args.receipt_sha256 is None, 'arguments_invalid')
            out = prepare(Path(args.staged_root))
        else:
            need(args.staged_root is None, 'arguments_invalid')
            out = launch(Path(args.launch_root), args.receipt_sha256)
        print(json.dumps(out, sort_keys=True))
        return 0 if out.get('prepared') or out.get('launched') else 1
    except BaseException:
        print(json.dumps({'ok': False, 'stage': 'prepare' if args.prepare else 'launch',
                          'reason': 'host_preparation_or_dispatch_failed',
                          'never_retry_launch': not args.prepare}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
