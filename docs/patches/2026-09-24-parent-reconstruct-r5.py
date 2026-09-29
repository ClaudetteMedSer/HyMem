"""Independently reconstruct/freeze R5 from accepted R4 and exact patch15."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import types

DOCS = Path(__file__).resolve().parent
PINS = {
    "2026-09-24-lme-independent-summary-indexing-r4-manifest.json": "16cf214451f90ee9c7d2a73151a0b26c5bd9fe04404cc40b7370f03777c916d0",
    "2026-09-24-lme-independent-summary-indexing-r5-manifest.json": "c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc",
    "2026-09-24-lme-15-response-boundary.patch": "4ba27727009ab6aae65e249f0f5896d3ea6570ccac7afaf48a588a56c4228256",
    "2026-09-24-reconstruct-independent-summary-r3.py": "4ba233909f3117ee4f0e9332b16ba66b13ffc074318b79feae16dbab413dadfa",
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, pin):
    assert path.resolve() == path and path.is_file() and not path.is_symlink()
    raw = path.read_bytes()
    assert sha(raw) == pin, str(path)
    return raw

def mapping(manifest):
    groups = [manifest[k] for k in ("source_sha256", "test_sha256", "auxiliary_sha256")]
    result = {name: pin for group in groups for name, pin in group.items()}
    assert len(result) == sum(map(len, groups))
    return result

def capture(tree, pins):
    assert tree.resolve() == tree and tree.is_dir()
    paths = list(tree.rglob("*"))
    assert not any(path.is_symlink() for path in paths)
    assert {path.relative_to(tree).as_posix() for path in paths if path.is_file()} == set(pins)
    return {name: read(tree / name, pin) for name, pin in pins.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r4", required=True, type=Path)
    parser.add_argument("--agent-r5", required=True, type=Path)
    args = parser.parse_args()
    files = {name: read(DOCS / name, pin) for name, pin in PINS.items()}
    r4 = json.loads(files["2026-09-24-lme-independent-summary-indexing-r4-manifest.json"])
    r5 = json.loads(files["2026-09-24-lme-independent-summary-indexing-r5-manifest.json"])
    before = capture(args.r4, mapping(r4))
    proposed = capture(args.agent_r5, mapping(r5))
    helper_name = "2026-09-24-reconstruct-independent-summary-r3.py"
    helper = types.ModuleType("parent_exact_r5")
    helper.__file__ = str(DOCS / helper_name)
    exec(compile(files[helper_name], helper.__file__, "exec"), helper.__dict__)
    # Git metadata is not part of a unified hunk. Actual hunk lines always
    # begin with space/+/-; none of their bytes are stripped or rewritten.
    patch = b"".join(line for line in files["2026-09-24-lme-15-response-boundary.patch"].splitlines(keepends=True)
                     if not line.startswith((b"diff --git ", b"index ", b"new file mode ")))
    # Git also annotates hunk headers with enclosing function names. Keep both
    # coordinate/count pairs exactly, removing only that non-source annotation.
    patch = re.sub(rb"^(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@)[^\n]*\n", rb"\1\n", patch, flags=re.MULTILINE)
    after, changed = helper.apply_exact(patch, before, mapping(r5))
    assert after == proposed and len(after) == 467
    root = Path(tempfile.mkdtemp(prefix="hymem-parent-frozen-r5-20260924.", dir="/private/tmp"))
    tree = root / "verification-tree-r5"
    tree.mkdir(mode=0o700)
    for name, raw in sorted(after.items()):
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o400)
    assert capture(tree, mapping(r5)) == after
    assert capture(args.r4, mapping(r4)) == before
    assert capture(args.agent_r5, mapping(r5)) == after
    receipt = {"schema": "parent-r5-exact-reconstruction-v1", "tree": str(tree),
               "verified_files": len(after), "changed_files": changed, "pins": PINS,
               "candidate_matched_agent_tree": True, "r4_unchanged": True,
               "provider_calls": 0, "benchmark_readiness_claimed": False}
    target = DOCS / "2026-09-24-parent-r5-exact-reconstruction.json"
    with target.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))

if __name__ == "__main__":
    main()
