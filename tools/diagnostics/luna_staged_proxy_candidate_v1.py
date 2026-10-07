"""Derive a source-only staged-proxy candidate from the frozen 514-file bundle.

This module has no deployment, provider, receipt, or launch entry point.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


INPUT_INVENTORY_SHA256 = "228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf"
INPUT_MAP_SHA256 = "9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae"
INPUT_FILES = 514
RUNNER = "hymem/dreaming/runner.py"
PRODUCER = "hymem/extraction/producer.py"
INPUT_RUNNER_SHA256 = "25387efe4ef6cf5ca6d6178f96a0c7872748cdb3758bbf68836c222027691f5a"
INPUT_PRODUCER_SHA256 = "2b4fa362c8a101cfd8dd28a842ce52052839f2c7ecc6cf0859befb123a68070c"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: bytes, before: str, after: str) -> bytes:
    old, new = before.encode(), after.encode()
    if source.count(old) != 1:
        raise ValueError("recipe_anchor_drift")
    return source.replace(old, new, 1)


def transform_runner(source: bytes) -> bytes:
    if _sha(source) != INPUT_RUNNER_SHA256:
        raise ValueError("runner_source_drift")
    source = _replace_once(source, '''    def complete(self, request):
        delegate = self._verified_delegate()
        self.completion_calls += 1
        attempt_measurement = None
        try:
            with measure_provider_attempts(delegate) as attempt_measurement:
                return delegate.complete(request)
        finally:
            self.provider_attempts += (
                attempt_measurement.attempts
                if attempt_measurement is not None
                else 1
            )


class _HeartbeatLLMClient:''', '''    def complete(self, request):
        delegate = self._verified_delegate()
        self.completion_calls += 1
        attempt_measurement = None
        try:
            with measure_provider_attempts(delegate) as attempt_measurement:
                return delegate.complete(request)
        finally:
            self.provider_attempts += (
                attempt_measurement.attempts
                if attempt_measurement is not None
                else 1
            )

    def complete_stage(self, request, batch, stage, recheck):
        delegate = self._verified_delegate()
        self.completion_calls += 1
        attempt_measurement = None
        try:
            with measure_provider_attempts(delegate) as attempt_measurement:
                return delegate.complete_stage(request, batch, stage, recheck)
        finally:
            self.provider_attempts += (
                attempt_measurement.attempts
                if attempt_measurement is not None
                else 1
            )


class _HeartbeatLLMClient:''')
    source = _replace_once(source, '''    def complete(self, request):
        delegate = self._verified_delegate()
        self._heartbeat()
        try:
            return delegate.complete(request)
        finally:
            # If both the provider and ownership failed, lease loss wins: an
            # obsolete owner must not enter provider-failure publication paths.
            self._heartbeat()


_RUNNER_PROXY_DISPATCH_NAMES = (
    "embed", "complete", "model",''', '''    def complete(self, request):
        delegate = self._verified_delegate()
        self._heartbeat()
        try:
            return delegate.complete(request)
        finally:
            # If both the provider and ownership failed, lease loss wins: an
            # obsolete owner must not enter provider-failure publication paths.
            self._heartbeat()

    def complete_stage(self, request, batch, stage, recheck):
        delegate = self._verified_delegate()
        self._heartbeat()
        try:
            return delegate.complete_stage(request, batch, stage, recheck)
        finally:
            # Lease loss takes precedence over a raised provider call.
            self._heartbeat()


_RUNNER_PROXY_DISPATCH_NAMES = (
    "embed", "complete", "complete_stage", "model",''')
    return source


def transform_producer(source: bytes) -> bytes:
    if _sha(source) != INPUT_PRODUCER_SHA256:
        raise ValueError("producer_source_drift")
    return _replace_once(source, '''_PRODUCER_PROXY_DISPATCH_NAMES = (
    "embed", "complete", "model",''', '''_PRODUCER_PROXY_DISPATCH_NAMES = (
    "embed", "complete", "complete_stage", "model",''')


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def assemble(frozen_bundle: Path, output: Path) -> dict[str, object]:
    """Verify exact frozen inputs, then copy 514 files into a fresh private root."""
    frozen_bundle, output = Path(frozen_bundle), Path(output)
    if not frozen_bundle.is_absolute() or not output.is_absolute():
        raise ValueError("absolute_paths_required")
    candidate = frozen_bundle / "candidate"
    inventory = frozen_bundle / "source-map.json"
    if (frozen_bundle.is_symlink() or not frozen_bundle.is_dir()
            or candidate.is_symlink() or not candidate.is_dir()
            or not _regular(inventory) or output.exists() or output.is_symlink()
            or not output.parent.is_dir() or output.parent.is_symlink()):
        raise ValueError("bundle_input_invalid")
    raw_map = inventory.read_bytes()
    if _sha(raw_map) != INPUT_INVENTORY_SHA256:
        raise ValueError("inventory_drift")
    stamp = json.loads(raw_map)
    entries = stamp.get("source_sha256") if type(stamp) is dict else None
    if type(entries) is not dict or len(entries) != INPUT_FILES:
        raise ValueError("inventory_shape_invalid")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    if _sha(encoded) != INPUT_MAP_SHA256:
        raise ValueError("map_drift")
    if entries.get(RUNNER) != INPUT_RUNNER_SHA256 or entries.get(PRODUCER) != INPUT_PRODUCER_SHA256:
        raise ValueError("proxy_source_pin_drift")
    paths: dict[str, bytes] = {}
    for relative, expected in entries.items():
        if (type(relative) is not str or not relative or relative.startswith("/")
                or ".." in Path(relative).parts or type(expected) is not str
                or len(expected) != 64):
            raise ValueError("inventory_entry_invalid")
        path = candidate / relative
        if any(part.is_symlink() for part in (path, *path.parents) if part != candidate.parent and part != frozen_bundle.parent):
            raise ValueError("candidate_symlink")
        if not _regular(path):
            raise ValueError("candidate_file_invalid")
        data = path.read_bytes()
        if _sha(data) != expected:
            raise ValueError("candidate_file_drift")
        paths[relative] = data
    expected_dirs = {str(parent) for relative in entries for parent in Path(relative).parents if str(parent) != "."}
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for root, dirs, files in os.walk(candidate, followlinks=False):
        base = Path(root)
        for name in dirs:
            path = base / name
            if path.is_symlink():
                raise ValueError("candidate_symlink")
            actual_dirs.add(str(path.relative_to(candidate)))
        for name in files:
            path = base / name
            if not _regular(path):
                raise ValueError("candidate_file_invalid")
            actual_files.add(str(path.relative_to(candidate)))
    if actual_files != set(entries) or actual_dirs != expected_dirs:
        raise ValueError("candidate_file_set_drift")
    paths[RUNNER] = transform_runner(paths[RUNNER])
    paths[PRODUCER] = transform_producer(paths[PRODUCER])
    new_entries = {relative: _sha(paths[relative]) for relative in entries}
    new_map = json.dumps({"source_sha256": new_entries}, sort_keys=True, indent=2).encode() + b"\n"
    output.mkdir(mode=0o700)
    for relative, data in paths.items():
        target = output / "candidate" / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(data)
    (output / "source-map.json").write_bytes(new_map)
    return {"root": str(output), "candidate_files": INPUT_FILES,
            "changed": {relative: new_entries[relative] for relative in (RUNNER, PRODUCER)},
            "candidate_map_sha256": _sha(json.dumps(new_entries, sort_keys=True, separators=(",", ":")).encode()),
            "inventory_sha256": _sha(new_map), "model_calls": 0}
