"""Copy a closed R6 source/test inventory and freeze its collected offline tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

R5_PIN = 'c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    baseline_raw = args.baseline_manifest.read_bytes()
    assert sha(baseline_raw) == R5_PIN
    baseline = json.loads(baseline_raw)
    assert args.tree.resolve() == args.tree and args.output.resolve() == args.output
    assert not args.output.is_relative_to(args.tree) and not args.manifest.is_relative_to(args.output)
    groups = {group: set(baseline[group]) for group in ('source_sha256', 'test_sha256', 'auxiliary_sha256')}
    groups['test_sha256'].update(p.relative_to(args.tree).as_posix() for p in (args.tree/'tests').rglob('*.py'))
    expected = set().union(*groups.values())
    # Only generated caches may be excluded; every other new file is reviewed.
    actual = set()
    for path in args.tree.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            relative = path.relative_to(args.tree)
            if '__pycache__' not in relative.parts and '.pytest_cache' not in relative.parts:
                actual.add(relative.as_posix())
    assert actual == expected, sorted(actual ^ expected)
    args.output.mkdir(mode=0o755)
    manifest = {'schema': 'lme-r6-sequential-fixes-frozen-inventory-v1',
                'baseline_manifest_sha256': R5_PIN,
                'expected_skip_nodeids': baseline['expected_skip_nodeids']}
    for group, files in groups.items():
        hashes = {}
        for relative in sorted(files):
            raw = (args.tree/relative).read_bytes()
            path = args.output/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)
            hashes[relative] = sha(raw)
        manifest[group] = hashes
    with tempfile.TemporaryDirectory(prefix='hymem-r6-collection-') as home:
        env = {'PATH': '/opt/anaconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': home,
               'TMPDIR': home, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
               'PYTHONHASHSEED': '0', 'LANG': 'en_US.UTF-8'}
        result = subprocess.run(['/opt/anaconda3/bin/python', '-B', '-m', 'pytest', 'tests',
                                 '--collect-only', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider'],
                                cwd=args.output, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-4000:]
    nodes = [line for line in result.stdout.splitlines() if line.startswith('tests/') and '::' in line]
    assert len(nodes) > 7247 and len(nodes) == len(set(nodes))
    assert set(manifest['expected_skip_nodeids']) <= set(nodes)
    manifest['expected_nodeids'] = nodes
    # Collection must not mutate any frozen source, test, or auxiliary file.
    for group in groups:
        for relative, pin in manifest[group].items():
            assert sha((args.output/relative).read_bytes()) == pin
    raw = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    with args.manifest.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'tree': str(args.output), 'manifest': str(args.manifest),
                      'manifest_sha256': sha(raw), 'test_cases': len(nodes),
                      'files': len(expected)}, sort_keys=True))


if __name__ == '__main__':
    main()
