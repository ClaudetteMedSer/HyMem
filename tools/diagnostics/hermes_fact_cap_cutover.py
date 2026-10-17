"""Guarded, two-file Hermes1 fact-cap launch configuration cutover.

Preview only hashes the proposed edit. Apply requires explicit flags, writes
private backups, and atomically replaces both scripts with rollback if the
second replacement fails. A guarded rollback restores the exact backups after
a later code-cutover failure. Existing secret values are never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


MCP = Path("/home/node/.hermes/bin/hymem-server-wrapper")
HONCHO = Path("/home/node/.agent37/hooks/post-restart.sh")
BACKUP = Path("/home/node/.hermes/backups/hymem-fact-cap-20261017")
EXPECTED_OLD = {
    "mcp": "685b198a87e22c877d562e126b0f13780e500e30383b7a3f43ae2b6b73930402",
    "honcho": "5242c99e9100415a3cf8107f35ae60aa07e014ef443b736fa081e36a8d7c99d7",
}


class CapCutoverStopped(RuntimeError):
    pass


def _physical_parent(path: Path) -> None:
    for directory in (*reversed(path.parents), path.parent):
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise CapCutoverStopped("script path traverses a symlink or non-directory")


def _script(path: Path) -> tuple[bytes, int]:
    _physical_parent(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        raise CapCutoverStopped("launch script is not an owned single-link file")
    return path.read_bytes(), stat.S_IMODE(info.st_mode)


def _private_directory(path: Path) -> None:
    _physical_parent(path / "file")
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        raise CapCutoverStopped("backup directory is not owned and private")


def _saved_script(path: Path, role: str) -> bytes:
    _physical_parent(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        raise CapCutoverStopped("saved launch script is not an owned private file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_OLD[role]:
        raise CapCutoverStopped("saved launch script differs from reviewed baseline")
    return payload


def _edit(old: bytes, cap: int, role: str) -> bytes:
    if b"HYMEM_DREAM_MAX_FACTS_PER_SESSION" in old:
        raise CapCutoverStopped("launch script already contains a fact cap")
    lines = old.splitlines(keepends=True)
    if role == "mcp":
        anchor = b"export HYMEM_AGGREGATION_NODES_ENABLED="
        new = f"export HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap}\n".encode()
        offset = 1
    else:
        anchor = b"      /home/node/hymem-env/bin/hymem-honcho"
        new = f"      HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap} \\\n".encode()
        offset = 0
    hits = [index for index, line in enumerate(lines) if line.startswith(anchor)]
    if len(hits) != 1:
        raise CapCutoverStopped("launch script anchor differs from reviewed baseline")
    lines.insert(hits[0] + offset, new)
    return b"".join(lines)


def _exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _temp(path: Path, payload: bytes, mode: int) -> Path:
    fd, name = tempfile.mkstemp(prefix=".hymem-cap-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        return Path(name)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _install_pair(planned: list[tuple[str, Path, bytes, bytes, int]]) -> None:
    staged: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, bytes, int]] = []
    try:
        for _role, path, _current, target, mode in planned:
            staged.append((path, _temp(path, target, mode)))
        for _role, path, current, _target, mode in planned:
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(current).digest():
                raise CapCutoverStopped("launch script changed after preview")
            staged_path = next(tmp for destination, tmp in staged if destination == path)
            os.replace(staged_path, path)
            installed.append((path, current, mode))
            _fsync_directory(path.parent)
        for _role, path, _current, target, _mode in planned:
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(target).digest():
                raise CapCutoverStopped("launch script install verification failed")
    except BaseException:
        for path, current, mode in reversed(installed):
            restored = _temp(path, current, mode)
            os.replace(restored, path)
            _fsync_directory(path.parent)
        raise
    finally:
        for _path, temp in staged:
            temp.unlink(missing_ok=True)


def cutover(cap: int, mcp: Path, honcho: Path, backup: Path, *,
            apply: bool = False, rollback: bool = False) -> dict:
    if cap not in (16, 32):
        raise CapCutoverStopped("only reviewed caps 16 and 32 are supported")
    if apply and rollback:
        raise CapCutoverStopped("apply and rollback cannot run together")
    parent = backup.parent
    _private_directory(parent)
    if rollback:
        _private_directory(backup)
    elif backup.exists() or backup.is_symlink():
        raise CapCutoverStopped("private backup directory already exists")
    planned = []
    for role, path in (("mcp", mcp), ("honcho", honcho)):
        old, mode = _script(path)
        if mode != 0o755:
            raise CapCutoverStopped("launch script mode differs from reviewed baseline")
        if rollback:
            saved = _saved_script(backup / f"{role}.before", role)
            changed = _edit(saved, cap, role)
            if hashlib.sha256(old).digest() != hashlib.sha256(changed).digest():
                raise CapCutoverStopped("launch script differs from selected cap")
            planned.append((role, path, old, saved, mode))
        else:
            if hashlib.sha256(old).hexdigest() != EXPECTED_OLD[role]:
                raise CapCutoverStopped("launch script differs from reviewed baseline")
            planned.append((role, path, old, _edit(old, cap, role), mode))
    output = {"cap": cap, "applied": False,
              "old_sha256": {r: hashlib.sha256(o).hexdigest()
                             for r, _p, o, _n, _m in planned},
              "new_sha256": {r: hashlib.sha256(n).hexdigest()
                             for r, _p, _o, n, _m in planned}}
    if not apply and not rollback:
        return output
    if apply:
        backup.mkdir(mode=0o700, exist_ok=False)
        _fsync_directory(parent)
        for role, path, old, new, mode in planned:
            saved = backup / f"{role}.before"
            _exclusive(saved, old)
            if _saved_script(saved, role) != old:
                raise CapCutoverStopped("launch script backup verification failed")
            _fsync_directory(backup)
    _install_pair(planned)
    output["applied"] = apply
    output["rolled_back"] = rollback
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, choices=(16, 32), required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--rollback", action="store_true")
    parser.add_argument("--allow-live-write", action="store_true")
    args = parser.parse_args()
    if (args.apply or args.rollback) and not args.allow_live_write:
        parser.error("live cap edit or rollback requires --allow-live-write")
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("cap edit must run as the Hermes node user (uid/gid 1000)")
    try:
        print(json.dumps(cutover(args.cap, MCP, HONCHO, BACKUP,
                                 apply=args.apply, rollback=args.rollback),
                         sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, CapCutoverStopped)
                          else "unexpected cap edit failure; inspect private backup"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
