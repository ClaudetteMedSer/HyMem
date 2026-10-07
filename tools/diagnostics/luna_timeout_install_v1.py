"""Source-only installer emitter for the accepted Luna timeout diagnostic.

The emitted script is intended for ``ssh ... python3 -I -B -``.  Emitting it
has no host effects; running it only installs pinned source bytes in a fresh
private root.  It never imports or executes any installed source.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


PREPARATION_SHA256 = "b2b8f6462b3592bc67a2094d59d93aabad3bcd13ce03ad3bb40d9d225372ad1b"
HOST_SHA256 = "f38c77ee73b05e1004dd9863b873665c55f568bd4e50782701aeaa48ce11d93c"
SOURCE_NAMES = frozenset({
    "benchmarks/codex_subscription.py",
    "benchmarks/codex_subscription_concurrent_v2.py",
    "benchmarks/codex_subscription_timeout_v1.py",
    *[f"benchmarks/codex_subscription_warm_v{i}.py" for i in range(2, 9)],
    "hymem/contrib/implementation_identity.py",
    "hymem/extraction/llm.py",
    "tools/diagnostics/luna_timeout_probe_v1.py",
})
HELPERS = {
    "timeout-host-v1.py": "luna_timeout_host_v1.py",
    "timeout-launch-v1.py": "luna_timeout_launch_v1.py",
    "timeout-progress-v1.py": "luna_timeout_progress_v1.py",
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
MAX_PAYLOAD = 500_000
MAX_SOURCE = 100_000


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _need(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _read_regular(path: Path) -> bytes:
    meta = path.lstat()
    _need(stat.S_ISREG(meta.st_mode) and meta.st_size <= MAX_SOURCE,
          "source_not_regular_or_too_large")
    data = path.read_bytes()
    _need(len(data) <= MAX_SOURCE, "source_too_large")
    return data


def _inventory(bundle: Path) -> set[str]:
    _need(stat.S_ISDIR(bundle.lstat().st_mode), "bundle_not_directory")
    found: set[str] = set()
    directories: set[str] = set()
    for parent, dirs, files in os.walk(bundle, followlinks=False):
        base = Path(parent)
        for name in dirs:
            _need(stat.S_ISDIR((base / name).lstat().st_mode), "bundle_symlink_or_special")
            directories.add((base / name).relative_to(bundle).as_posix())
        for name in files:
            path = base / name
            _need(stat.S_ISREG(path.lstat().st_mode), "bundle_symlink_or_special")
            found.add(path.relative_to(bundle).as_posix())
    expected_files = {"local-preparation-receipt.json"} | {
        "code/" + name for name in SOURCE_NAMES}
    expected_dirs = {parent.as_posix() for name in expected_files
                     for parent in Path(name).parents if parent != Path(".")}
    _need(directories == expected_dirs, "bundle_directory_inventory_invalid")
    return found


def build_payload(bundle: Path, helpers_dir: Path, launcher_sha256: str,
                  reader_sha256: str) -> bytes:
    """Read and bind precisely the accepted closure and three helper bytes."""
    _need(type(launcher_sha256) is str and HEX.fullmatch(launcher_sha256) is not None
          and type(reader_sha256) is str and HEX.fullmatch(reader_sha256) is not None,
          "helper_pin_invalid")
    expected = {"local-preparation-receipt.json"} | {
        "code/" + name for name in SOURCE_NAMES}
    _need(_inventory(bundle) == expected, "bundle_inventory_invalid")
    prep = _read_regular(bundle / "local-preparation-receipt.json")
    _need(_sha(prep) == PREPARATION_SHA256, "preparation_drift")
    receipt = json.loads(prep, object_pairs_hook=_unique)
    pins = receipt.get("source_sha256") if type(receipt) is dict else None
    _need(type(pins) is dict and set(pins) == SOURCE_NAMES and
          all(type(v) is str and HEX.fullmatch(v) for v in pins.values()),
          "preparation_inventory_invalid")
    items: list[dict[str, str]] = []
    for name in sorted(SOURCE_NAMES):
        data = _read_regular(bundle / "code" / name)
        _need(_sha(data) == pins[name], "source_drift")
        items.append({"path": "bundle/code/" + name,
                      "sha256": pins[name], "base64": base64.b64encode(data).decode("ascii")})
    items.append({"path": "bundle/local-preparation-receipt.json",
                  "sha256": PREPARATION_SHA256,
                  "base64": base64.b64encode(prep).decode("ascii")})
    for target, source in sorted(HELPERS.items()):
        data = _read_regular(helpers_dir / source)
        pin = {"timeout-host-v1.py": HOST_SHA256,
               "timeout-launch-v1.py": launcher_sha256,
               "timeout-progress-v1.py": reader_sha256}[target]
        _need(_sha(data) == pin, "helper_drift")
        items.append({"path": target, "sha256": pin,
                      "base64": base64.b64encode(data).decode("ascii")})
    payload = {"schema": "luna-timeout-install-v1", "preparation_sha256": PREPARATION_SHA256,
               "helper_sha256": {"host": HOST_SHA256, "launcher": launcher_sha256,
                                 "reader": reader_sha256}, "files": items}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("ascii")
    _need(len(encoded) <= MAX_PAYLOAD, "payload_too_large")
    return encoded


# The host script is standalone: no repository or installed source is imported.
REMOTE_CODE = r'''from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

PAYLOAD_B64 = __PAYLOAD_B64__
PAYLOAD_SHA256 = __PAYLOAD_SHA256__
HOST_HOME = Path("/home/atta")
UID = 1000
PREP_SHA = "b2b8f6462b3592bc67a2094d59d93aabad3bcd13ce03ad3bb40d9d225372ad1b"
HOST_SHA = "f38c77ee73b05e1004dd9863b873665c55f568bd4e50782701aeaa48ce11d93c"
SOURCE_NAMES = frozenset({
    "benchmarks/codex_subscription.py",
    "benchmarks/codex_subscription_concurrent_v2.py",
    "benchmarks/codex_subscription_timeout_v1.py",
    *[f"benchmarks/codex_subscription_warm_v{i}.py" for i in range(2, 9)],
    "hymem/contrib/implementation_identity.py",
    "hymem/extraction/llm.py",
    "tools/diagnostics/luna_timeout_probe_v1.py",
})
HELPER_NAMES = frozenset({"timeout-host-v1.py", "timeout-launch-v1.py",
                          "timeout-progress-v1.py"})
MAX_PAYLOAD = 500_000
MAX_SOURCE = 100_000

def need(ok, code):
    if not ok:
        raise ValueError(code)

def unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, "duplicate_json_key")
        value[key] = item
    return value

def sha(data):
    return hashlib.sha256(data).hexdigest()

def validate(data):
    need(type(data) is bytes and len(data) <= MAX_PAYLOAD and
         sha(data) == PAYLOAD_SHA256, "payload_drift")
    payload = json.loads(data, object_pairs_hook=unique,
                         parse_constant=lambda _: need(False, "nonfinite_json"))
    need(type(payload) is dict and set(payload) ==
         {"schema", "preparation_sha256", "helper_sha256", "files"} and
         payload["schema"] == "luna-timeout-install-v1" and
         payload["preparation_sha256"] == PREP_SHA, "payload_invalid")
    helper = payload["helper_sha256"]
    need(type(helper) is dict and set(helper) == {"host", "launcher", "reader"}
         and helper["host"] == HOST_SHA and
         all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v)
             for v in helper.values()), "helper_pins_invalid")
    rows = payload["files"]
    need(type(rows) is list and len(rows) == 17, "file_count_invalid")
    files = {}
    for row in rows:
        need(type(row) is dict and set(row) == {"path", "sha256", "base64"}
             and type(row["path"]) is str and type(row["sha256"]) is str
             and type(row["base64"]) is str, "file_row_invalid")
        name = row["path"]
        need(name not in files and (name in HELPER_NAMES or
             name == "bundle/local-preparation-receipt.json" or
             name in {"bundle/code/" + x for x in SOURCE_NAMES}),
             "file_path_invalid")
        try:
            content = base64.b64decode(row["base64"], validate=True)
        except (ValueError, base64.binascii.Error):
            raise ValueError("file_encoding_invalid") from None
        need(len(content) <= MAX_SOURCE and sha(content) == row["sha256"],
             "file_hash_invalid")
        files[name] = content
    expected = HELPER_NAMES | {"bundle/local-preparation-receipt.json"} | {
        "bundle/code/" + x for x in SOURCE_NAMES}
    need(set(files) == expected, "file_inventory_invalid")
    prep = files["bundle/local-preparation-receipt.json"]
    need(sha(prep) == PREP_SHA, "preparation_drift")
    receipt = json.loads(prep, object_pairs_hook=unique,
                         parse_constant=lambda _: need(False, "nonfinite_json"))
    pins = receipt.get("source_sha256") if type(receipt) is dict else None
    need(type(pins) is dict and set(pins) == SOURCE_NAMES, "source_pins_invalid")
    for name in SOURCE_NAMES:
        need(sha(files["bundle/code/" + name]) == pins[name], "source_drift")
    for name, key in (("timeout-host-v1.py", "host"),
                      ("timeout-launch-v1.py", "launcher"),
                      ("timeout-progress-v1.py", "reader")):
        need(sha(files[name]) == helper[key], "helper_drift")
    return files, helper

def write_once(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

def install(data, home=HOST_HOME, uid=UID, enforce_host=True):
    files, helper = validate(data)
    if enforce_host:
        need(sys.platform == "linux" and os.getuid() == UID and
             os.geteuid() == UID and home == HOST_HOME,
             "host_user_invalid")
    home_meta = home.lstat()
    need(stat.S_ISDIR(home_meta.st_mode) and home_meta.st_uid == uid and
         not home_meta.st_mode & 0o022, "home_invalid")
    need(shutil.disk_usage(home).free >= 20 * 1024**3,
         "disk_floor") if enforce_host else None
    if enforce_host:
        memory = dict(line.split(":", 1) for line in
                      Path("/proc/meminfo").read_text().splitlines())
        need(int(memory["MemAvailable"].split()[0]) * 1024 >= 6 * 1024**3,
             "memory_floor")
    old_umask = os.umask(0o077)
    try:
        root = Path(tempfile.mkdtemp(prefix=".hymem-luna-timeout-", dir=home))
        os.chmod(root, 0o700)
        need(stat.S_IMODE(root.stat().st_mode) == 0o700 and
             root.stat().st_uid == uid and root.parent == home,
             "root_invalid")
        for name in sorted(files):
            target = root / name
            need(target.is_relative_to(root) and ".." not in target.parts,
                 "file_path_invalid")
            parent = target.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            for directory in (parent, *[p for p in parent.parents if p != root
                                          and p.is_relative_to(root)]):
                need(stat.S_ISDIR(directory.lstat().st_mode) and
                     directory.stat().st_uid == uid and
                     stat.S_IMODE(directory.stat().st_mode) == 0o700,
                     "directory_invalid")
            write_once(target, files[name])
        receipt = {"schema": "luna-timeout-install-v1", "root": str(root),
                   "installed": True, "code_files": 13,
                   "preparation_receipts": 1, "bundle_files": 14,
                   "helpers": 3,
                   "preparation_sha256": PREP_SHA,
                   "helper_sha256": helper, "payload_sha256": sha(data),
                   "model_calls": 0}
        return receipt
    finally:
        os.umask(old_umask)

if __name__ == "__main__":
    try:
        raw = base64.b64decode(PAYLOAD_B64, validate=True)
        result = install(raw)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"),
                         allow_nan=False))
    except Exception:
        print(json.dumps({"schema": "luna-timeout-install-v1",
                          "installed": False, "error": "install_unverified",
                          "model_calls": 0},
                         sort_keys=True, separators=(",", ":")))
        raise SystemExit(1)
'''


def render_remote_script(payload: bytes) -> bytes:
    _need(type(payload) is bytes and len(payload) <= MAX_PAYLOAD,
          "payload_invalid")
    source = REMOTE_CODE.replace("__PAYLOAD_B64__",
                                 repr(base64.b64encode(payload).decode("ascii")))
    source = source.replace("__PAYLOAD_SHA256__", repr(_sha(payload)))
    return source.encode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["emit"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--reader-sha256", required=True)
    args = parser.parse_args(argv)
    payload = build_payload(args.bundle, Path(__file__).resolve().parent,
                            args.launcher_sha256, args.reader_sha256)
    sys.stdout.buffer.write(render_remote_script(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
