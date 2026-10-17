"""One-use, hash-guarded Hermes1 HyMem MCP keepalive config edit.

Preview is read-only. Apply inserts one YAML line, preserves the original
config in a private backup, and atomically replaces the file. It does not
signal or restart the gateway, call MCP tools, or contact a provider.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

import yaml


CONFIG = Path("/home/node/.hermes/config.yaml")
BACKUP = Path("/home/node/.hermes/backups/hymem-mcp-keepalive-20261017")
EXPECTED_OLD_SHA256 = "8e00b1c0f2a6b62256ad86aee80c8791adeb8db0c1e811d55884f96c9bc2fbd7"
INTERVAL_SECONDS = 60


class KeepaliveStopped(RuntimeError):
    pass


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _physical_parent(path: Path) -> None:
    for directory in reversed(path.parents):
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise KeepaliveStopped("path traverses a symlink or non-directory")


def _owned_file(path: Path) -> bytes:
    _physical_parent(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
            raise KeepaliveStopped("config or backup is not an owned private single-link file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def _private_directory(path: Path) -> None:
    _physical_parent(path / "item")
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        raise KeepaliveStopped("backup directory is not owned and private")


def _parsed(payload: bytes) -> dict:
    try:
        result = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError):
        raise KeepaliveStopped("config is not valid UTF-8 YAML") from None
    if not isinstance(result, dict):
        raise KeepaliveStopped("config YAML root is not a mapping")
    return result


def _edit(old: bytes) -> bytes:
    lines = old.splitlines(keepends=True)
    if any(not line.endswith(b"\n") or b"\r" in line for line in lines):
        raise KeepaliveStopped("config line endings differ from reviewed baseline")
    roots = [i for i, line in enumerate(lines) if line == b"mcp_servers:\n"]
    if len(roots) != 1:
        raise KeepaliveStopped("expected exactly one root MCP config block")
    root = roots[0]
    root_end = next((i for i in range(root + 1, len(lines))
                     if lines[i].strip() and not lines[i].lstrip().startswith(b"#")
                     and not lines[i].startswith((b" ", b"\t"))), len(lines))
    matches = [i for i in range(root + 1, root_end) if lines[i] == b"  hymem:\n"]
    if len(matches) != 1:
        raise KeepaliveStopped("expected exactly one HyMem MCP block")
    start = matches[0]
    end = next((i for i in range(start + 1, root_end)
                if lines[i].strip() and not lines[i].lstrip().startswith(b"#")
                and not lines[i].startswith(b"    ")), root_end)
    if b"keepalive_interval" in b"".join(lines[start:end]):
        raise KeepaliveStopped("HyMem MCP block already mentions keepalive")
    parsed = _parsed(old)
    servers = parsed.get("mcp_servers")
    hymem = servers.get("hymem") if isinstance(servers, dict) else None
    if not isinstance(hymem, dict):
        raise KeepaliveStopped("HyMem MCP config is not a mapping")
    if "keepalive_interval" in hymem:
        raise KeepaliveStopped("HyMem MCP already has a keepalive setting")
    lines.insert(end, f"    keepalive_interval: {INTERVAL_SECONDS}\n".encode())
    new = b"".join(lines)
    expected = copy.deepcopy(parsed)
    expected["mcp_servers"]["hymem"]["keepalive_interval"] = INTERVAL_SECONDS
    if _parsed(new) != expected:
        raise KeepaliveStopped("proposed YAML changes more than HyMem keepalive")
    return new


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _temporary(path: Path, payload: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=".hymem-keepalive-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o600)
        return Path(name)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def cutover(config: Path, backup: Path, *, expected_sha256: str,
            apply: bool = False) -> dict:
    _private_directory(backup.parent)
    if backup.exists() or backup.is_symlink():
        raise KeepaliveStopped("one-use private backup directory already exists")
    old = _owned_file(config)
    if _sha(old) != expected_sha256:
        raise KeepaliveStopped("config differs from the reviewed hash")
    new = _edit(old)
    result = {"applied": False, "interval_seconds": INTERVAL_SECONDS,
              "old_sha256": _sha(old), "new_sha256": _sha(new)}
    if not apply:
        return result

    backup.mkdir(mode=0o700, exist_ok=False)
    _private_directory(backup)
    _fsync_directory(backup.parent)
    saved = backup / "config.yaml.before"
    _exclusive(saved, old)
    _fsync_directory(backup)
    if _sha(_owned_file(saved)) != expected_sha256:
        raise KeepaliveStopped("private backup verification failed")

    temporary = _temporary(config, new)
    installed = False
    try:
        if _owned_file(config) != old:
            raise KeepaliveStopped("config changed after the private backup")
        os.replace(temporary, config)
        installed = True
        _fsync_directory(config.parent)
        if _owned_file(config) != new:
            raise KeepaliveStopped("installed config verification failed")
    except BaseException:
        if installed:
            restored = _temporary(config, old)
            os.replace(restored, config)
            _fsync_directory(config.parent)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    result["applied"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live-write", action="store_true")
    parser.add_argument("--expect-old-sha256")
    args = parser.parse_args()
    if args.apply and (not args.allow_live_write
                       or args.expect_old_sha256 != EXPECTED_OLD_SHA256):
        parser.error("apply requires --allow-live-write and the exact preview old SHA256")
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("keepalive edit must run as Hermes node user (uid/gid 1000)")
    try:
        print(json.dumps(cutover(CONFIG, BACKUP,
                                 expected_sha256=EXPECTED_OLD_SHA256,
                                 apply=args.apply), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, KeepaliveStopped)
                          else "unexpected keepalive edit failure; inspect private backup"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
