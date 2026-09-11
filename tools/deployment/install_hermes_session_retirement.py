#!/usr/bin/env python3
"""Pinned, offline, pre-start installer for the Hermes session retirement patch.

Only the three fixed plugin source files can be replaced. Run with Hermes
stopped, or in its pre-start hook. This is not a three-file atomic transaction:
an uncatchable interruption can leave a mixed state, which subsequent runs
refuse. Keep the separately reviewed deployment backup and receipt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys


PATHS = (
    "plugins/memory/honcho/client.py",
    "plugins/memory/honcho/session.py",
    "plugins/memory/honcho/config_schema.py",
)
MANIFEST_NAME = "hermes-session-key-retirement.json"
PATCH_NAME = "hermes-session-key-retirement.patch"
MAX_MANIFEST = 64 * 1024
MAX_PATCH = 128 * 1024
MAX_SOURCE = 4 * 1024 * 1024
_HASH = re.compile(r"[0-9a-f]{64}")
_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:[^\r\n]*)\n")


class InstallError(RuntimeError):
    """Only fixed, non-sensitive error codes are exposed by the CLI."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InstallError("duplicate_manifest_key")
        result[key] = value
    return result


def _open_directory(path: Path) -> int:
    """Walk an absolute path with no-follow descriptors, including ancestors."""
    if not path.is_absolute() or ".." in path.parts or path == Path("/"):
        raise InstallError("unsafe_directory_path")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read(directory: int, name: str, limit: int):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise InstallError("unsafe_or_oversized_file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(limit + 1)
        after = os.fstat(descriptor)
        if len(content) > limit or _identity(before) != _identity(after):
            raise InstallError("file_changed_during_read")
        return content, after
    finally:
        os.close(descriptor)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_uid, info.st_gid, info.st_nlink)


def _metadata(info):
    return stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid


def _pins(entries, expected_paths):
    if not isinstance(entries, list) or len(entries) != len(expected_paths):
        raise InstallError("invalid_manifest_file_set")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "before_sha256", "after_sha256"}:
            raise InstallError("invalid_manifest_entry")
        path = entry["path"]
        if not isinstance(path, str) or path not in expected_paths or path in result:
            raise InstallError("invalid_manifest_file_path")
        for field in ("before_sha256", "after_sha256"):
            if not isinstance(entry[field], str) or _HASH.fullmatch(entry[field]) is None:
                raise InstallError("invalid_manifest_digest")
        if entry["before_sha256"] == entry["after_sha256"]:
            raise InstallError("identical_manifest_states")
        result[path] = entry
    return result


def _load_artifacts(directory: Path):
    descriptor = _open_directory(directory)
    try:
        raw, _ = _read(descriptor, MANIFEST_NAME, MAX_MANIFEST)
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        required = {"format", "patch", "patch_sha256", "files"}
        optional = {"tests", "tests_sha256", "test_contract_patch",
                    "test_contract_patch_sha256", "test_contract_updates"}
        if not isinstance(manifest, dict) or not required <= set(manifest) or set(manifest) - required - optional:
            raise InstallError("invalid_manifest_fields")
        if manifest["format"] != "hermes-session-key-retirement-patch-v1" or manifest["patch"] != PATCH_NAME:
            raise InstallError("invalid_manifest_format_or_patch")
        for key in ("patch_sha256", "tests_sha256", "test_contract_patch_sha256"):
            if key in manifest and (not isinstance(manifest[key], str) or _HASH.fullmatch(manifest[key]) is None):
                raise InstallError("invalid_manifest_digest")
        for key, expected in (
            ("tests", "tests/test_hermes_session_key_retirement.py"),
            ("test_contract_patch", "hermes-session-key-retirement-test-contract.patch"),
        ):
            if key in manifest and manifest[key] != expected:
                raise InstallError("invalid_manifest_auxiliary_path")
        if "test_contract_updates" in manifest:
            _pins(manifest["test_contract_updates"], ("tests/honcho_plugin/test_client.py",))
        pins = _pins(manifest["files"], PATHS)
        patch, _ = _read(descriptor, PATCH_NAME, MAX_PATCH)
        if _sha(patch) != manifest["patch_sha256"]:
            raise InstallError("patch_digest_mismatch")
        return pins, _parse_patch(patch)
    finally:
        os.close(descriptor)


def _parse_patch(raw: bytes):
    """Accept only exact unified hunks for the three pinned existing files."""
    lines = raw.decode("utf-8").splitlines(keepends=True)
    patches, position = {}, 0
    while position < len(lines):
        header = lines[position]
        if not header.startswith("--- a/") or not header.endswith("\n"):
            raise InstallError("invalid_patch_header")
        path = header[len("--- a/"):-1]
        if path not in PATHS or path in patches:
            raise InstallError("invalid_patch_file_set")
        position += 1
        if position >= len(lines) or lines[position] != f"+++ b/{path}\n":
            raise InstallError("invalid_patch_target")
        position += 1
        hunks = []
        while position < len(lines) and not lines[position].startswith("--- a/"):
            match = _HUNK.fullmatch(lines[position])
            if match is None:
                raise InstallError("invalid_patch_hunk")
            old_start, old_count, new_start, new_count = (
                int(value) if value is not None else 1 for value in match.groups()
            )
            if old_start < 1 or new_start < 1 or old_count < 1 or new_count < 1:
                raise InstallError("unsupported_patch_hunk")
            position += 1
            removed, added = [], []
            while len(removed) < old_count or len(added) < new_count:
                if position >= len(lines):
                    raise InstallError("truncated_patch_hunk")
                line = lines[position]
                if not line.endswith("\n") or line[:1] not in (" ", "+", "-"):
                    raise InstallError("invalid_patch_line")
                if line[0] in " -":
                    removed.append(line[1:])
                if line[0] in " +":
                    added.append(line[1:])
                if len(removed) > old_count or len(added) > new_count:
                    raise InstallError("invalid_patch_counts")
                position += 1
            hunks.append((old_start - 1, new_start - 1, removed, added))
        if not hunks:
            raise InstallError("empty_patch_file")
        patches[path] = hunks
    if set(patches) != set(PATHS):
        raise InstallError("invalid_patch_file_set")
    return patches


def _apply_exact(original: bytes, hunks) -> bytes:
    lines = original.decode("utf-8").splitlines(keepends=True)
    result, consumed = [], 0
    for old_start, new_start, removed, added in hunks:
        if old_start < consumed or old_start > len(lines):
            raise InstallError("patch_offset_mismatch")
        result.extend(lines[consumed:old_start])
        if len(result) != new_start or lines[old_start:old_start + len(removed)] != removed:
            raise InstallError("patch_context_mismatch")
        result.extend(added)
        consumed = old_start + len(removed)
    result.extend(lines[consumed:])
    output = "".join(result).encode("utf-8")
    if len(output) > MAX_SOURCE:
        raise InstallError("patched_source_too_large")
    return output


def _write_stage(directory, name, content, original_info):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
        os.fchown(descriptor, original_info.st_uid, original_info.st_gid)
        os.fchmod(descriptor, stat.S_IMODE(original_info.st_mode))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_source(directory, name, expected):
    content, info = _read(directory, name, MAX_SOURCE)
    if content != expected[0] or _identity(info) != _identity(expected[1]):
        raise InstallError("source_changed_before_install")


def install(runtime_root: Path, *, artifact_dir: Path | None = None, check: bool = False) -> str:
    if not runtime_root.is_absolute() or runtime_root == Path("/") or ".." in runtime_root.parts:
        raise InstallError("unsafe_runtime_root")
    pins, patches = _load_artifacts(artifact_dir or Path(__file__).absolute().parent)
    directory = _open_directory(runtime_root / "plugins/memory/honcho")
    stage_fd, stage_name, stage_files = None, None, []
    retain_stage = False
    try:
        # Cooperative invocations lock the same directory inode without a
        # persistent lock file. Other writers still require the stopped-service
        # precondition and are detected by source revalidation before replacement.
        try:
            fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise InstallError("another_installer_holds_directory_lock") from None
        originals = {path: _read(directory, Path(path).name, MAX_SOURCE) for path in PATHS}
        states = [
            "before" if _sha(originals[path][0]) == pins[path]["before_sha256"] else
            "after" if _sha(originals[path][0]) == pins[path]["after_sha256"] else "unknown"
            for path in PATHS
        ]
        if states == ["after"] * len(PATHS):
            return "already-installed"
        if states != ["before"] * len(PATHS):
            raise InstallError("mixed_or_unknown_source_state")
        outputs = {path: _apply_exact(originals[path][0], patches[path]) for path in PATHS}
        if any(_sha(outputs[path]) != pins[path]["after_sha256"] for path in PATHS):
            raise InstallError("patched_source_digest_mismatch")
        if check:
            return "ready"

        candidate_name = ".hymem-session-retirement-" + secrets.token_hex(16)
        os.mkdir(candidate_name, 0o700, dir_fd=directory)
        stage_name = candidate_name
        stage_fd = os.open(stage_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                           dir_fd=directory)
        for path in PATHS:
            name = Path(path).name
            for prefix, content in (("before-", originals[path][0]), ("after-", outputs[path])):
                staged = prefix + name
                stage_files.append(staged)
                _write_stage(stage_fd, staged, content, originals[path][1])
        os.fsync(stage_fd)
        os.fsync(directory)
        for path in PATHS:
            _same_source(directory, Path(path).name, originals[path])

        replaced = []
        try:
            for path in PATHS:
                name = Path(path).name
                _same_source(directory, name, originals[path])
                # Record intent before rename, so a caught interruption between
                # the syscall and Python's next statement also rolls back.
                replaced.append(path)
                os.replace("after-" + name, name, src_dir_fd=stage_fd, dst_dir_fd=directory)
                os.fsync(directory)
            for path in PATHS:
                content, info = _read(directory, Path(path).name, MAX_SOURCE)
                if content != outputs[path] or _metadata(info) != _metadata(originals[path][1]):
                    raise InstallError("installed_source_verification_failed")
        except BaseException:
            rollback_failed = False
            for path in reversed(replaced):
                name = Path(path).name
                try:
                    content, info = _read(directory, name, MAX_SOURCE)
                    if content == originals[path][0] and _identity(info) == _identity(originals[path][1]):
                        continue  # The attempted replacement did not happen.
                    if content != outputs[path] or _metadata(info) != _metadata(originals[path][1]):
                        raise InstallError("rollback_target_changed")
                    os.replace("before-" + name, name, src_dir_fd=stage_fd, dst_dir_fd=directory)
                    os.fsync(directory)
                    restored, restored_info = _read(directory, name, MAX_SOURCE)
                    if restored != originals[path][0] or _metadata(restored_info) != _metadata(originals[path][1]):
                        raise InstallError("rollback_verification_failed")
                except BaseException:
                    rollback_failed = True
            retain_stage = rollback_failed
            if rollback_failed:
                raise InstallError("install_failed_rollback_incomplete_preserved_staging") from None
            raise InstallError("install_failed_rolled_back") from None
        return "installed"
    finally:
        if stage_fd is not None:
            try:
                if not retain_stage:
                    for name in stage_files:
                        try:
                            os.unlink(name, dir_fd=stage_fd)
                        except FileNotFoundError:
                            pass
            finally:
                os.close(stage_fd)
        try:
            if stage_name is not None and not retain_stage:
                os.rmdir(stage_name, dir_fd=directory)
                os.fsync(directory)
        finally:
            os.close(directory)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True,
                        help="Absolute Hermes source checkout root; service must be stopped")
    parser.add_argument("--check", action="store_true", help="Validate without staging or replacing files")
    args = parser.parse_args(argv)
    try:
        status = install(args.runtime_root, check=args.check)
    except InstallError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RecursionError):
        # Never emit file content, paths, original exception values or config.
        print('{"status":"FAIL","error":"unsafe_or_unavailable_install_inputs"}', file=sys.stderr)
        return 1
    print(json.dumps({"status": status, "runtime_files": len(PATHS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
