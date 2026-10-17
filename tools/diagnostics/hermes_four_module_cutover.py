"""One-use, hash-guarded Hermes1 code cutover; preview is read-only.

Stage the four reviewed source files in a private 0700 directory inside the
container. Apply requires explicit flags, creates private backups, atomically
replaces each module, and restores already replaced modules on failure. This
tool never restarts services or accesses the HyMem database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path


FILES = {
    "hymem/bootstrap.py": (
        "e0718d9de843b9bc40c8cc0eca9be5e22fe762f67af986d1bd35752d7f7a1716",
        "581568e8a4000f628e9fb6c17cfc5c0e33278016a4c287fc3ba0a6857a833cfd",
    ),
    "hymem/dreaming/facts.py": (
        "85411929079bbc135968a7e0044ffc0ed4e8c3c998cbd60dc6286373130a232a",
        "36a11d562ec199c4657bd657f8781c9e6463be2ef90169f810ba77e0c516c4fb",
    ),
    "hymem/dreaming/evidence.py": (
        "c921d035c4af5b6d0c5add8cb9eb89069b3dbef4447078470ae77dc0dbb36771",
        "3430c06d5ac8c74baf1a0bfbcf1ea0630f1488aec3ec766e40d70c9b3a44648e",
    ),
    "hymem/dreaming/phase1.py": (
        "bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac",
        "5ed0f056b7566257acaf8917d5a4bc4814d47307418eb4b0d6c265f88af6738c",
    ),
}
STAGE = Path("/tmp/hymem-cutover-20261017")
TARGET = Path("/home/node/HyMem")
BACKUP = Path("/home/node/.hermes/backups/hymem-code-20261017")


class CutoverStopped(RuntimeError):
    pass


def _physical_directory_chain(path: Path) -> None:
    for directory in (*reversed(path.parents), path):
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise CutoverStopped("module path traverses a symlink or non-directory")


def _regular_owned(path: Path) -> os.stat_result:
    _physical_directory_chain(path.parent)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        raise CutoverStopped("module is not an owned, single-link regular file")
    return info


def _private_dir(path: Path) -> None:
    _physical_directory_chain(path)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.geteuid() or info.st_gid != os.getegid()):
        raise CutoverStopped("stage or backup parent is not an owned private directory")


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _stage_replacement(path: Path, payload: bytes, mode: int) -> Path:
    fd, temporary = tempfile.mkstemp(prefix=".hymem-cutover-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        return Path(temporary)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def cutover(stage: Path, target: Path, backup: Path, *, apply: bool,
            deadline: float | None = None) -> dict:
    _private_dir(stage)
    _private_dir(backup.parent)
    if backup.exists() or backup.is_symlink():
        raise CutoverStopped("backup directory already exists")
    planned = []
    for name, (old_sha, new_sha) in FILES.items():
        old_path, new_path = target / name, stage / name
        info = _regular_owned(old_path)
        _regular_owned(new_path)
        old, new = old_path.read_bytes(), new_path.read_bytes()
        if (hashlib.sha256(old).hexdigest() != old_sha
                or hashlib.sha256(new).hexdigest() != new_sha):
            raise CutoverStopped("module hash differs from the reviewed manifest")
        planned.append((name, old_path, old, new, stat.S_IMODE(info.st_mode)))
    result = {
        "modules": len(planned),
        "old_sha256": {name: FILES[name][0] for name, *_ in planned},
        "new_sha256": {name: FILES[name][1] for name, *_ in planned},
        "applied": False,
    }
    if not apply:
        return result

    backup.mkdir(mode=0o700, exist_ok=False)
    _private_dir(backup)
    temporary: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, bytes, int]] = []
    try:
        for name, old_path, old, new, mode in planned:
            destination = backup / name
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _private_dir(destination.parent)
            _write_exclusive(destination, old)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != FILES[name][0]:
                raise CutoverStopped("private backup hash verification failed")
            temporary.append((old_path, _stage_replacement(old_path, new, mode)))
        _fsync_dir(backup)
        for old_path, staged in temporary:
            if deadline is not None and time.monotonic() >= deadline:
                raise CutoverStopped("module cutover exceeded its pause deadline")
            name = old_path.relative_to(target).as_posix()
            if hashlib.sha256(old_path.read_bytes()).hexdigest() != FILES[name][0]:
                raise CutoverStopped("live module changed after the backup")
            os.replace(staged, old_path)
            old = next(row[2] for row in planned if row[0] == name)
            mode = next(row[4] for row in planned if row[0] == name)
            installed.append((old_path, old, mode))
            _fsync_dir(old_path.parent)
        for name, old_path, *_ in planned:
            if deadline is not None and time.monotonic() >= deadline:
                raise CutoverStopped("module cutover exceeded its pause deadline")
            if hashlib.sha256(old_path.read_bytes()).hexdigest() != FILES[name][1]:
                raise CutoverStopped("installed module hash verification failed")
        result["applied"] = True
        return result
    except BaseException:
        for old_path, old, mode in reversed(installed):
            restored = _stage_replacement(old_path, old, mode)
            os.replace(restored, old_path)
            _fsync_dir(old_path.parent)
        raise
    finally:
        for _path, staged in temporary:
            staged.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live-write", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.allow_live_write:
        parser.error("live code cutover requires explicit --allow-live-write")
    if os.geteuid() != 1000 or os.getegid() != 1000:
        parser.error("cutover must run as the Hermes node user (uid/gid 1000)")
    try:
        print(json.dumps(cutover(STAGE, TARGET, BACKUP, apply=args.apply), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, CutoverStopped)
                          else "unexpected cutover failure; inspect private backup"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
