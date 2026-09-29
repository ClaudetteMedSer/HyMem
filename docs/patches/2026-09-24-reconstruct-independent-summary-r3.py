"""Rebuild the pinned R3 tree from Git and exact durable unified hunks.

No fuzzy patching, application imports, tests, remote operations or changes to
the working checkout. Only a freshly created isolated directory is written.
Seven unchanged test auxiliaries may be read from the checkout, but only after
matching their recorded hashes. README is reconstructed by its separate patch.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
MANIFEST = '2026-09-19-lme-independent-summary-indexing-r3-manifest.json'
MANIFEST_PIN = '573fd0f5d763adbbd9f26df7fa249a546e2980304d303771ea76fcec4526aa66'
BASE = 'af6a615fa7fd1cf14c4c0a27b9fb236ae264f122'
HUNK = re.compile(rb'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n\Z')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, expected=None):
    assert path.is_file() and not path.is_symlink() and path.resolve() == path
    raw = path.read_bytes()
    assert expected is None or sha(raw) == expected, ('pin', str(path))
    return raw

def apply_exact(patch, originals, targets):
    """Require every old byte and both hunk coordinate systems exactly."""
    lines = patch.splitlines(keepends=True)
    result, changed, index = dict(originals), [], 0
    while index < len(lines):
        # Some early durable patches separate file diffs with a bare newline.
        # This is outside hunks; every source/context byte remains exact.
        if lines[index] == b'\n':
            index += 1
            continue
        assert lines[index].startswith(b'--- ') and lines[index].endswith(b'\n'), ('file_header', index, lines[index][:120])
        old_path = lines[index][4:-1].decode()
        index += 1
        assert lines[index].startswith(b'+++ b/') and lines[index].endswith(b'\n')
        name = lines[index][6:-1].decode()
        path = PurePosixPath(name)
        assert str(path) == name and not path.is_absolute() and '..' not in path.parts
        assert name in targets and name not in changed
        is_new = old_path == '/dev/null'
        assert (is_new and name not in originals) or (
            not is_new and old_path == 'a/' + name and name in originals)
        original = b'' if is_new else originals[name]
        old_lines = original.splitlines(keepends=True)
        index += 1
        output, position, hunks = [], 0, 0
        while index < len(lines) and lines[index].startswith(b'@@ '):
            match = HUNK.fullmatch(lines[index])
            assert match is not None, ('hunk', name)
            a, ac, b, bc = match.groups()
            old_count = int(ac) if ac is not None else 1
            new_count = int(bc) if bc is not None else 1
            old_at, new_at = int(a) - bool(old_count), int(b) - bool(new_count)
            assert position <= old_at <= len(old_lines)
            output.extend(old_lines[position:old_at])
            position = old_at
            assert len(output) == new_at, ('new_coordinate', name)
            consumed = emitted = 0
            index += 1
            while consumed < old_count or emitted < new_count:
                assert index < len(lines)
                line = lines[index]
                kind, data = line[:1], line[1:]
                assert kind in (b' ', b'+', b'-') and line.endswith(b'\n')
                if kind in (b' ', b'-'):
                    assert consumed < old_count and position < len(old_lines)
                    assert old_lines[position] == data, ('old_byte', name, position + 1)
                    consumed += 1
                    position += 1
                if kind in (b' ', b'+'):
                    assert emitted < new_count
                    output.append(data)
                    emitted += 1
                index += 1
            assert (consumed, emitted) == (old_count, new_count)
            hunks += 1
        assert hunks > 0
        output.extend(old_lines[position:])
        result[name] = b''.join(output)
        assert result[name] != original or is_new
        changed.append(name)
    return result, changed

def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True,
                          capture_output=True).stdout

def main():
    manifest = json.loads(read(DOCS / MANIFEST, MANIFEST_PIN))
    assert manifest['base_commit'] == BASE
    pins = {**manifest['prior_patch_sha256'], manifest['patch']: manifest['patch_sha256']}
    assert len(pins) == 14
    targets = {**manifest['source_sha256'], **manifest['test_sha256']}
    assert len(targets) == 451
    tracked = set(git('ls-tree', '-r', '--name-only', BASE).decode().splitlines())
    originals = {name: git('show', BASE + ':' + name) for name in sorted(set(targets) & tracked)}
    result = originals
    for name, pin in sorted(pins.items()):
        try:
            result, _ = apply_exact(read(DOCS / name, pin), result, targets)
        except AssertionError as exc:
            raise AssertionError(name, *exc.args) from exc
    assert set(result) == set(targets)
    for name, pin in targets.items():
        assert sha(result[name]) == pin, ('reconstructed', name)
    auxiliary = manifest['auxiliary_sha256']
    assert len(auxiliary) == 8 and not set(auxiliary).intersection(result)
    aux_meta = json.loads(read(DOCS / manifest['auxiliary_correction_manifest'],
                              manifest['auxiliary_correction_manifest_sha256']))
    readme = read(REPO / 'README.md', aux_meta['original_readme_sha256'])
    aux_result, changed = apply_exact(read(DOCS / aux_meta['patch'], aux_meta['patch_sha256']),
                                      {'README.md': readme}, {'README.md'})
    assert changed == ['README.md']
    for name, pin in auxiliary.items():
        raw = aux_result[name] if name == 'README.md' else read(REPO / name, pin)
        assert sha(raw) == pin
        result[name] = raw
    assert len(result) == 459
    root = Path(tempfile.mkdtemp(prefix='hymem-reconstructed-r3-20260924.', dir='/private/tmp'))
    tree = root / 'verification-tree-r3'
    tree.mkdir(mode=0o700)
    for name, raw in sorted(result.items()):
        destination = tree / name
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with destination.open('xb') as stream:
            stream.write(raw)
        destination.chmod(0o400)
    for name, raw in result.items():
        assert read(tree / name, sha(raw)) == raw
    assert {p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()} == set(result)
    for name, pin in pins.items():
        read(DOCS / name, pin)
    read(DOCS / MANIFEST, MANIFEST_PIN)
    receipt = dict(schema='durable-r3-exact-reconstruction-20260924-v1',
        status='all459_file_hashes_verified', tree=str(tree), base_commit=BASE,
        durable_r3_manifest=MANIFEST, durable_r3_manifest_sha256=MANIFEST_PIN,
        prior_patch_sha256=pins, source_files_verified=230, test_files_verified=221,
        auxiliary_test_only_files_verified=8, combined_files_verified=459,
        collected_nodeids_from_original_manifest=len(manifest['expected_nodeids']),
        helper_sha256=sha(read(Path(__file__))), no_fuzzy_patching=True,
        source_and_test_bytes_from_git_and_patches_only=True,
        working_checkout_unchanged=True, tests_executed=False, api_calls=0,
        paid_calls=0, remote_access=False, deployment_performed=False,
        old_vanished_test_receipts_reused_as_new_evidence=False,
        scope='Exact reconstruction only; original collected node IDs are metadata, not a fresh passing test result.')
    print(json.dumps(receipt, sort_keys=True, indent=2))

if __name__ == '__main__':
    main()
