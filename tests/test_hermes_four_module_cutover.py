"""Atomic staged code cutover for the Hermes1 one-use patch."""

import hashlib
import os
import stat

import pytest

from tools.diagnostics import hermes_four_module_cutover as cutover


def _fixture(tmp_path, monkeypatch):
    stage, target, backup_parent = (
        tmp_path / "stage", tmp_path / "target", tmp_path / "backups",
    )
    for directory in (stage, target, backup_parent):
        directory.mkdir(mode=0o700)
    manifest = {}
    for index, name in enumerate(cutover.FILES):
        source, destination = stage / name, target / name
        source.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        old = f"old-{index}".encode()
        new = f"new-{index}".encode()
        source.write_bytes(new)
        destination.write_bytes(old)
        destination.chmod(0o664)
        manifest[name] = (
            hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest(),
        )
    monkeypatch.setattr(cutover, "FILES", manifest)
    return stage, target, backup_parent / "one-use-backup"


def test_preview_checks_manifest_without_writing(tmp_path, monkeypatch):
    stage, target, backup = _fixture(tmp_path, monkeypatch)
    result = cutover.cutover(stage, target, backup, apply=False)
    assert result["modules"] == 4 and not result["applied"]
    assert not backup.exists()
    (stage / "hymem/bootstrap.py").write_bytes(b"changed")
    with pytest.raises(cutover.CutoverStopped, match="hash differs"):
        cutover.cutover(stage, target, backup, apply=False)
    assert not backup.exists()


def test_apply_preserves_live_mode_and_private_backups(tmp_path, monkeypatch):
    stage, target, backup = _fixture(tmp_path, monkeypatch)
    result = cutover.cutover(stage, target, backup, apply=True)
    assert result["applied"] and stat.S_IMODE(backup.stat().st_mode) == 0o700
    for name, (old_hash, new_hash) in cutover.FILES.items():
        destination, saved = target / name, backup / name
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == new_hash
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == old_hash
        assert stat.S_IMODE(destination.stat().st_mode) == 0o664
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600


def test_second_replace_failure_restores_first_module(tmp_path, monkeypatch):
    stage, target, backup = _fixture(tmp_path, monkeypatch)
    original_replace = os.replace
    second = target / list(cutover.FILES)[1]
    rejected = False

    def fail_once(source, destination):
        nonlocal rejected
        if not rejected and destination == second:
            rejected = True
            raise OSError("fixture injected replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(cutover.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected"):
        cutover.cutover(stage, target, backup, apply=True)
    assert rejected
    for name, (old_hash, _new_hash) in cutover.FILES.items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == old_hash


def test_symlinked_module_parent_is_rejected_before_backup(tmp_path, monkeypatch):
    stage, target, backup = _fixture(tmp_path, monkeypatch)
    nested = stage / "hymem" / "dreaming"
    moved = stage / "real-dreaming"
    nested.rename(moved)
    nested.symlink_to(moved, target_is_directory=True)
    with pytest.raises(cutover.CutoverStopped, match="symlink"):
        cutover.cutover(stage, target, backup, apply=True)
    assert not backup.exists()
