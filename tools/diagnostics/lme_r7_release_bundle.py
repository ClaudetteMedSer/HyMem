"""Mechanically export and round-trip the approved R7 delta without staging it.

Only the reviewed release and pinned Git base are read. Experimental main is
never used as an application input; neither worktree nor Git index is modified.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[2]
RELEASE = Path('/private/tmp/hymem-r7-integration-20260925.vDa9yP/release')
BASE = '5fb5ce491b254015684be2f3115a9d6b70b0c5a3'
MANIFEST = REPO / 'docs/patches/2026-09-25-lme-r7-final-frozen-manifest.json'
MANIFEST_PIN = '1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8'
PATCH = REPO / 'docs/patches/2026-09-25-lme-r7-release-from-head.patch'
RECEIPT = REPO / 'docs/patches/2026-09-25-parent-r7-release-patch.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, cwd=RELEASE, input=None, allowed=(0,)):
    result = subprocess.run(['git', *args], cwd=cwd, input=input,
                            capture_output=True, timeout=60)
    if result.returncode not in allowed:
        raise RuntimeError('git_operation_failed: ' + args[0])
    return result.stdout


def names(raw):
    return {name.decode() for name in raw.split(b'\0') if name}


def inventory(tree):
    result = {}
    for path in tree.rglob('*'):
        if path.name == '.git':
            continue
        assert not path.is_symlink()
        if path.is_file():
            result[path.relative_to(tree).as_posix()] = sha(path.read_bytes())
    return result


def main():
    if sys.flags.optimize:
        raise RuntimeError('optimized_execution_forbidden')
    assert not PATCH.exists() and not RECEIPT.exists()
    assert git('rev-parse', 'HEAD').decode().strip() == BASE
    raw = MANIFEST.read_bytes()
    assert sha(raw) == MANIFEST_PIN
    manifest = json.loads(raw)
    expected = {}
    for group in ('source_sha256', 'test_sha256', 'auxiliary_sha256'):
        assert not set(expected) & set(manifest[group])
        expected.update(manifest[group])
    assert len(expected) == 479
    original = inventory(RELEASE)
    assert all(original[name] == pin for name, pin in expected.items())
    changed = names(git('diff', '--name-only', '-z', BASE))
    added = names(git('ls-files', '--others', '--exclude-standard', '-z'))
    assert len(changed) == 87 and len(added) == 49
    assert not changed & added and changed | added <= set(expected)
    assert not git('diff', '--name-only', '--diff-filter=DT', BASE)
    assert not git('diff', '--summary', BASE)  # no rename/mode changes
    base_paths = names(git('ls-tree', '-r', '--name-only', '-z', BASE))
    assert set(original) == base_paths | added
    patch = git('diff', '--binary', '--full-index', '--no-ext-diff', BASE,
                '--', *sorted(expected))
    for name in sorted(added):
        patch += git('diff', '--no-index', '--binary', '--full-index',
                     '--no-ext-diff', '--', '/dev/null', name, allowed=(1,))
    assert patch.count(b'diff --git ') == 136
    archive = git('archive', '--format=tar', BASE)
    with tempfile.TemporaryDirectory(prefix='hymem-r7-patch-roundtrip-') as temporary:
        tree = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as reader:
            members = reader.getmembers()
            assert all(member.isfile() or member.isdir() for member in members)
            assert all(not Path(member.name).is_absolute()
                       and '..' not in Path(member.name).parts for member in members)
            reader.extractall(tree, members=members, filter='data')
        git('apply', '--check', '--binary', '-', cwd=tree, input=patch)
        git('apply', '--binary', '-', cwd=tree, input=patch)
        rebuilt = inventory(tree)
        assert rebuilt == original and len(rebuilt) == 533
        assert all(rebuilt[name] == pin for name, pin in expected.items())
        git('apply', '--reverse', '--check', '--binary', '-', cwd=tree, input=patch)
    assert inventory(RELEASE) == original
    with PATCH.open('xb') as stream:
        stream.write(patch)
    receipt = {
        'schema': 'r7-release-patch-roundtrip-v1', 'base_commit': BASE,
        'manifest_sha256': MANIFEST_PIN, 'patch_sha256': sha(patch),
        'patch_bytes': len(patch), 'modified_paths': sorted(changed),
        'new_paths': sorted(added), 'deleted_paths': [],
        'roundtrip_repository_files': 533, 'verified_manifest_files': 479,
        'unchanged_base_extras': 54, 'forward_and_reverse_check_passed': True,
        'exact_release_bytes_reproduced': True, 'git_index_modified': False,
        'main_application_modified': False, 'production_modified': False,
        'provider_calls': 0,
    }
    with RECEIPT.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in ('modified_paths', 'new_paths')}, sort_keys=True))


if __name__ == '__main__':
    main()
