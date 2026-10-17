"""One-use Hermes1 cap 16 to 32 launch-script cutover and guarded rollback.

Preview reads exact hashes only. Apply saves both cap-16 scripts in a private
directory, then atomically replaces them with the reviewed cap-32 versions.
No process is signaled and no private script content is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from pathlib import Path


MCP = Path("/home/node/.hermes/bin/hymem-server-wrapper")
HONCHO = Path("/home/node/.agent37/hooks/post-restart.sh")
BACKUP = Path("/home/node/.hermes/backups/hymem-cap16-to32-20261017")
DATABASE = Path("/home/node/.hermes/hymem.sqlite")
OLD = {
    "mcp": "0a65d39216f9e3aa18f1c1760e2082066ac643640d882340f00cdf992dd9f335",
    "honcho": "f8c71749393a9d7735daa85bb413601433af0a1c4299b56771568874d6d0dbd7",
}
NEW = {
    "mcp": "98d2b20b26e2f2700d11e66013ecb960c8292e55259ca036bb484d018cb9779a",
    "honcho": "3bd9dd44611613208c8351559a28e3fdc2462cf745609e61f4b9c5dcab38204d",
}
ASSIGNMENTS = {
    "mcp": (b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=16\n",
            b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=32\n"),
    "honcho": (b"      HYMEM_DREAM_MAX_FACTS_PER_SESSION=16 \\\n",
               b"      HYMEM_DREAM_MAX_FACTS_PER_SESSION=32 \\\n"),
}


class Cap32Stopped(RuntimeError):
    pass


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _physical_parent(path: Path) -> None:
    for directory in reversed(path.parents):
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise Cap32Stopped("script or backup path traverses a symlink")


def _owned_file(path: Path, mode: int) -> bytes:
    _physical_parent(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != mode
                or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
            raise Cap32Stopped("script or backup is not an owned single-link file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def _private_dir(path: Path) -> None:
    _physical_parent(path / "item")
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        raise Cap32Stopped("backup directory is not owned and private")


def _dream_idle(db_path: Path) -> None:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("SELECT 1 FROM run_lock WHERE name='dreaming'").fetchone():
            raise Cap32Stopped("dreaming lease is held")
    finally:
        connection.close()


def _edit(payload: bytes, role: str) -> bytes:
    before, after = ASSIGNMENTS[role]
    if payload.count(before) != 1 or b"HYMEM_DREAM_MAX_FACTS_PER_SESSION=32" in payload:
        raise Cap32Stopped("script lacks exactly one reviewed cap-16 assignment")
    changed = payload.replace(before, after, 1)
    if _sha(changed) != NEW[role]:
        raise Cap32Stopped("proposed cap-32 script differs from reviewed hash")
    return changed


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _temp(path: Path, content: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=".hymem-cap32-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o755)
        return Path(name)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _install_pair(planned: list[tuple[str, Path, bytes, bytes]]) -> None:
    staged: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, bytes]] = []
    try:
        for _role, path, _current, target in planned:
            staged.append((path, _temp(path, target)))
        for _role, path, current, _target in planned:
            if _owned_file(path, 0o755) != current:
                raise Cap32Stopped("script changed after private backup")
            staged_path = next(tmp for destination, tmp in staged if destination == path)
            os.replace(staged_path, path)
            installed.append((path, current))
            _fsync_dir(path.parent)
        for _role, path, _current, target in planned:
            if _owned_file(path, 0o755) != target:
                raise Cap32Stopped("installed script verification failed")
    except BaseException:
        for path, current in reversed(installed):
            restored = _temp(path, current)
            os.replace(restored, path)
            _fsync_dir(path.parent)
        raise
    finally:
        for _path, temporary in staged:
            temporary.unlink(missing_ok=True)


def cutover(mcp: Path, honcho: Path, backup: Path, *,
            apply: bool = False, rollback: bool = False,
            db_path: Path | None = None) -> dict:
    if apply and rollback:
        raise Cap32Stopped("choose apply or rollback")
    db_path = db_path or DATABASE
    _dream_idle(db_path)
    _private_dir(backup.parent)
    if rollback:
        _private_dir(backup)
    elif backup.exists() or backup.is_symlink():
        raise Cap32Stopped("one-use backup directory already exists")
    planned = []
    for role, path in (("mcp", mcp), ("honcho", honcho)):
        current = _owned_file(path, 0o755)
        if rollback:
            if _sha(current) != NEW[role]:
                raise Cap32Stopped("script differs from reviewed cap-32 version")
            saved = _owned_file(backup / f"{role}.cap16", 0o600)
            if _sha(saved) != OLD[role] or _edit(saved, role) != current:
                raise Cap32Stopped("private cap-16 backup differs from reviewed version")
            planned.append((role, path, current, saved))
        else:
            if _sha(current) != OLD[role]:
                raise Cap32Stopped("script differs from reviewed cap-16 version")
            planned.append((role, path, current, _edit(current, role)))
    result = {"applied": False, "rolled_back": False, "old_cap": 16, "new_cap": 32,
              "current_sha256": {role: _sha(current) for role, _path, current, _target in planned},
              "target_sha256": {role: _sha(target) for role, _path, _current, target in planned}}
    if not apply and not rollback:
        return result
    if apply:
        _dream_idle(db_path)
        backup.mkdir(mode=0o700, exist_ok=False)
        _private_dir(backup)
        _fsync_dir(backup.parent)
        for role, _path, current, _target in planned:
            saved_path = backup / f"{role}.cap16"
            _exclusive(saved_path, current)
            _fsync_dir(backup)
            if _owned_file(saved_path, 0o600) != current:
                raise Cap32Stopped("private cap-16 backup verification failed")
    _dream_idle(db_path)
    _install_pair(planned)
    result["applied"] = apply
    result["rolled_back"] = rollback
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--rollback", action="store_true")
    parser.add_argument("--allow-live-write", action="store_true")
    args = parser.parse_args()
    if (args.apply or args.rollback) and not args.allow_live_write:
        parser.error("live cap update or rollback requires --allow-live-write")
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("cap update must run as Hermes node user (uid/gid 1000)")
    try:
        print(json.dumps(cutover(MCP, HONCHO, BACKUP,
                                 apply=args.apply, rollback=args.rollback), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, Cap32Stopped)
                          else "unexpected cap update failure; inspect private backups"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
