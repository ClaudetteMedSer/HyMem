"""Freeze the approved R6-based release's closed executable/test inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if sys.flags.optimize:
    raise RuntimeError('optimized_execution_forbidden')

R6_PIN = "bd8d0f3a8fb40bd6b77e7ca6579c8e5e8ee78733bea2bb22df71bc4b2c12eaa2"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(tree):
    tests, runtime = set(), set()
    for directory in ("tests", "hymem", "benchmarks"):
        root = tree / directory
        assert not root.is_symlink()
        for path in root.rglob("*"):
            assert not path.is_symlink()
            suffixes = (".py",) if directory == "tests" else (".py", ".sql")
            if path.is_file() and path.suffix in suffixes:
                assert "__pycache__" not in path.parts
                target = tests if directory == "tests" else runtime
                target.add(path.relative_to(tree).as_posix())
    return tests, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    raw = args.baseline_manifest.read_bytes()
    assert sha(raw) == R6_PIN
    baseline = json.loads(raw)
    assert args.tree.resolve() == args.tree and args.output.resolve() == args.output
    assert args.manifest.is_absolute() and args.manifest.resolve() == args.manifest
    assert not args.output.exists() and not args.manifest.exists()
    assert not args.output.is_relative_to(args.tree)
    assert not args.manifest.is_relative_to(args.output)
    assert not args.manifest.is_relative_to(args.tree)
    groups = {name: set(baseline[name]) for name in (
        "source_sha256", "test_sha256", "auxiliary_sha256")}
    tests, runtime = inventory(args.tree)
    groups["test_sha256"] = tests
    # Retained docs and Git metadata are not executable test inputs. Conversely,
    # every runtime Python/SQL file must be explicitly covered by the source map.
    assert runtime == {name for name in groups["source_sha256"]
                       if name.startswith(("hymem/", "benchmarks/"))}
    expected = set().union(*groups.values())
    assert sum(map(len, groups.values())) == len(expected)
    assert set(baseline["test_sha256"]) <= groups["test_sha256"]
    manifest = {"schema": "lme-r7-release-frozen-inventory-v1",
                "baseline_manifest_sha256": R6_PIN,
                "expected_skip_nodeids": baseline["expected_skip_nodeids"]}
    args.output.mkdir(mode=0o755)
    for group, names in groups.items():
        hashes = {}
        for name in sorted(names):
            source = args.tree / name
            assert source.resolve() == source and source.is_file()
            content = source.read_bytes()
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
            hashes[name] = sha(content)
        manifest[group] = hashes
    with tempfile.TemporaryDirectory(prefix="hymem-r7-collection-") as private:
        env = {"PATH": "/opt/anaconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin",
               "HOME": private, "TMPDIR": private, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONHASHSEED": "0",
               "LANG": "en_US.UTF-8"}
        result = subprocess.run(
            ["/opt/anaconda3/bin/python", "-B", "-m", "pytest", "tests",
             "--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider"],
            cwd=args.output, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-4000:]
    nodes = [line for line in result.stdout.splitlines()
             if line.startswith("tests/") and "::" in line]
    assert nodes and len(nodes) == len(set(nodes))
    assert set(manifest["expected_skip_nodeids"]) <= set(nodes)
    manifest["expected_nodeids"] = nodes
    actual = set()
    for path in args.output.rglob("*"):
        assert not path.is_symlink()
        if path.is_file():
            actual.add(path.relative_to(args.output).as_posix())
    assert actual == expected
    for group in groups:
        for name, pin in manifest[group].items():
            assert sha((args.output / name).read_bytes()) == pin
            assert sha((args.tree / name).read_bytes()) == pin
    assert inventory(args.tree) == (tests, runtime)
    content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with args.manifest.open("xb") as stream:
        stream.write(content)
    print(json.dumps({"tree": str(args.output), "manifest": str(args.manifest),
                      "manifest_sha256": sha(content), "test_cases": len(nodes),
                      "files": len(expected)}, sort_keys=True))


if __name__ == "__main__":
    main()
