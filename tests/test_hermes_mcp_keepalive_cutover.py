"""Guarded one-line Hermes1 HyMem MCP keepalive edit."""

import copy
import hashlib
import json
import os
import stat
import sys

import pytest
import yaml

from tools.diagnostics import hermes_mcp_keepalive_cutover as operator


OLD = (
    b"model: example\nprivate_token: fixture-secret\n"
    b"mcp_servers:\n"
    b"  hymem:\n"
    b"    timeout: 1800\n"
    b"    command: /home/node/.hermes/bin/hymem-server-wrapper\n"
    b"    enabled: true\n"
    b"  another:\n"
    b"    command: other-mcp\n"
    b"plugins:\n"
    b"  enabled: true\n"
)
NEW = OLD.replace(b"    enabled: true\n", b"    enabled: true\n    keepalive_interval: 60\n")


def _fixture(tmp_path, payload=OLD):
    config = tmp_path / "config.yaml"
    config.write_bytes(payload)
    config.chmod(0o600)
    private = tmp_path / "backups"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    return config, private / "one-use-keepalive", hashlib.sha256(payload).hexdigest()


def test_preview_is_read_only_and_apply_changes_only_hymem_key(tmp_path):
    config, backup, old_sha = _fixture(tmp_path)
    preview = operator.cutover(config, backup, expected_sha256=old_sha)
    assert preview == {
        "applied": False, "interval_seconds": 60,
        "old_sha256": old_sha, "new_sha256": hashlib.sha256(NEW).hexdigest(),
    }
    assert config.read_bytes() == OLD and not backup.exists()

    result = operator.cutover(config, backup, expected_sha256=old_sha, apply=True)
    assert result["applied"] and config.read_bytes() == NEW
    before, after = yaml.safe_load(OLD), yaml.safe_load(config.read_bytes())
    expected = copy.deepcopy(before)
    expected["mcp_servers"]["hymem"]["keepalive_interval"] = 60
    assert after == expected
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    saved = backup / "config.yaml.before"
    assert saved.read_bytes() == OLD
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert list(backup.iterdir()) == [saved]


def test_stale_hash_and_existing_backup_refuse_before_edit(tmp_path):
    config, backup, old_sha = _fixture(tmp_path)
    config.write_bytes(OLD + b"# external change\n")
    with pytest.raises(operator.KeepaliveStopped, match="reviewed hash"):
        operator.cutover(config, backup, expected_sha256=old_sha, apply=True)
    assert not backup.exists()
    config.write_bytes(OLD)
    backup.mkdir(mode=0o700)
    with pytest.raises(operator.KeepaliveStopped, match="already exists"):
        operator.cutover(config, backup, expected_sha256=old_sha, apply=True)
    assert config.read_bytes() == OLD


@pytest.mark.parametrize("payload,reason", [
    (OLD.replace(b"    enabled: true\n", b"    keepalive_interval: 30\n    enabled: true\n"),
     "already"),
    (OLD.replace(b"  another:\n", b"  hymem:\n    command: duplicate\n  another:\n"),
     "exactly one HyMem"),
    (OLD.replace(b"plugins:\n", b"mcp_servers:\nplugins:\n"),
     "exactly one root"),
    (OLD.replace(b"    enabled: true\n", b"    enabled: [\n"),
     "valid UTF-8 YAML"),
])
def test_ambiguous_or_invalid_config_stops_before_backup(tmp_path, payload, reason):
    config, backup, old_sha = _fixture(tmp_path, payload)
    with pytest.raises(operator.KeepaliveStopped, match=reason):
        operator.cutover(config, backup, expected_sha256=old_sha, apply=True)
    assert config.read_bytes() == payload and not backup.exists()


def test_symlinked_config_parent_and_nonprivate_backup_parent_refuse(tmp_path):
    config, backup, old_sha = _fixture(tmp_path)
    backup.parent.chmod(0o755)
    with pytest.raises(operator.KeepaliveStopped, match="private"):
        operator.cutover(config, backup, expected_sha256=old_sha, apply=True)
    assert not backup.exists()
    backup.parent.chmod(0o700)
    actual = tmp_path / "actual"
    actual.mkdir()
    config.rename(actual / "config.yaml")
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(operator.KeepaliveStopped, match="symlink"):
        operator.cutover(link / "config.yaml", backup,
                         expected_sha256=old_sha, apply=True)
    assert not backup.exists()


def test_postinstall_verification_failure_restores_original_from_memory(tmp_path, monkeypatch):
    config, backup, old_sha = _fixture(tmp_path)
    replace = os.replace
    first = True

    def corrupt_once(source, destination):
        nonlocal first
        replace(source, destination)
        if first:
            first = False
            destination.write_bytes(b"fixture-corruption")

    monkeypatch.setattr(operator.os, "replace", corrupt_once)
    with pytest.raises(operator.KeepaliveStopped, match="installed config verification"):
        operator.cutover(config, backup, expected_sha256=old_sha, apply=True)
    assert config.read_bytes() == OLD
    assert (backup / "config.yaml.before").read_bytes() == OLD


def test_cli_preview_prints_only_hashes_and_apply_needs_preview_hash(tmp_path, monkeypatch, capsys):
    config, backup, old_sha = _fixture(tmp_path)
    monkeypatch.setattr(operator, "CONFIG", config)
    monkeypatch.setattr(operator, "BACKUP", backup)
    monkeypatch.setattr(operator, "EXPECTED_OLD_SHA256", old_sha)
    monkeypatch.setattr(sys, "argv", ["keepalive-cutover"])
    assert operator.main() == 0
    printed = capsys.readouterr().out
    assert "fixture-secret" not in printed
    assert json.loads(printed)["old_sha256"] == old_sha
    assert not backup.exists()
    monkeypatch.setattr(sys, "argv", ["keepalive-cutover", "--apply", "--allow-live-write"])
    with pytest.raises(SystemExit, match="2"):
        operator.main()
    assert config.read_bytes() == OLD and not backup.exists()
