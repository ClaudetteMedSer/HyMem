"""One-use cap16 to cap32 Hermes launch-script operator."""

import hashlib
import json
import os
import sqlite3
import stat
import sys

import pytest

from tools.diagnostics import hermes_fact_cap32_cutover as operator


MCP_16 = (
    b"#!/bin/bash\nexport PRIVATE_TOKEN=fixture-secret\n"
    b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=16\nexec hymem-server\n"
)
HONCHO_16 = (
    b"#!/bin/bash\nenv \\\n"
    b"      PRIVATE_TOKEN=fixture-secret \\\n"
    b"      HYMEM_DREAM_MAX_FACTS_PER_SESSION=16 \\\n"
    b"      /home/node/hymem-env/bin/hymem-honcho\n"
)
MCP_32 = MCP_16.replace(b"FACTS_PER_SESSION=16", b"FACTS_PER_SESSION=32")
HONCHO_32 = HONCHO_16.replace(b"FACTS_PER_SESSION=16", b"FACTS_PER_SESSION=32")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path, monkeypatch):
    mcp, honcho = tmp_path / "wrapper", tmp_path / "post-restart.sh"
    for path, payload in ((mcp, MCP_16), (honcho, HONCHO_16)):
        path.write_bytes(payload)
        path.chmod(0o755)
    parent = tmp_path / "backups"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    database = tmp_path / "hymem.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE run_lock(name TEXT PRIMARY KEY)")
    monkeypatch.setattr(operator, "DATABASE", database)
    monkeypatch.setattr(operator, "OLD", {"mcp": _sha(MCP_16), "honcho": _sha(HONCHO_16)})
    monkeypatch.setattr(operator, "NEW", {"mcp": _sha(MCP_32), "honcho": _sha(HONCHO_32)})
    return mcp, honcho, parent / "one-use-cap32"


def test_held_dream_lease_refuses_preview_and_apply_before_backup(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    with sqlite3.connect(operator.DATABASE) as connection:
        connection.execute("INSERT INTO run_lock(name) VALUES('dreaming')")
    with pytest.raises(operator.Cap32Stopped, match="lease is held"):
        operator.cutover(mcp, honcho, backup)
    with pytest.raises(operator.Cap32Stopped, match="lease is held"):
        operator.cutover(mcp, honcho, backup, apply=True)
    assert not backup.exists()
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_16, HONCHO_16)


def test_preview_apply_and_guarded_rollback(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    preview = operator.cutover(mcp, honcho, backup)
    assert preview["applied"] is False and preview["rolled_back"] is False
    assert preview["current_sha256"] == operator.OLD
    assert preview["target_sha256"] == operator.NEW
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_16, HONCHO_16)
    assert not backup.exists()

    result = operator.cutover(mcp, honcho, backup, apply=True)
    assert result["applied"] and (mcp.read_bytes(), honcho.read_bytes()) == (MCP_32, HONCHO_32)
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    for role, payload in (("mcp", MCP_16), ("honcho", HONCHO_16)):
        saved = backup / f"{role}.cap16"
        assert saved.read_bytes() == payload
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o755 for path in (mcp, honcho))

    result = operator.cutover(mcp, honcho, backup, rollback=True)
    assert result["rolled_back"] and (mcp.read_bytes(), honcho.read_bytes()) == (MCP_16, HONCHO_16)


def test_second_script_install_failure_restores_first(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    original_replace = os.replace
    calls = []

    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("fixture second replace failed")
        return original_replace(source, target)

    monkeypatch.setattr(operator.os, "replace", fail_second)
    with pytest.raises(OSError, match="fixture second replace"):
        operator.cutover(mcp, honcho, backup, apply=True)
    assert calls == [mcp, honcho, mcp]
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_16, HONCHO_16)
    assert (backup / "mcp.cap16").read_bytes() == MCP_16
    assert (backup / "honcho.cap16").read_bytes() == HONCHO_16


def test_stale_script_duplicate_assignment_and_backup_refuse(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    mcp.write_bytes(MCP_16 + b"# drift\n")
    with pytest.raises(operator.Cap32Stopped, match="reviewed cap-16"):
        operator.cutover(mcp, honcho, backup, apply=True)
    assert not backup.exists()

    duplicate = MCP_16.replace(
        b"exec hymem-server\n",
        b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=16\nexec hymem-server\n",
    )
    mcp.write_bytes(duplicate)
    monkeypatch.setitem(operator.OLD, "mcp", _sha(duplicate))
    with pytest.raises(operator.Cap32Stopped, match="exactly one"):
        operator.cutover(mcp, honcho, backup)
    assert not backup.exists()

    mcp.write_bytes(MCP_16)
    monkeypatch.setitem(operator.OLD, "mcp", _sha(MCP_16))
    backup.mkdir(mode=0o700)
    with pytest.raises(operator.Cap32Stopped, match="already exists"):
        operator.cutover(mcp, honcho, backup, apply=True)
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_16, HONCHO_16)


def test_corrupt_rollback_backup_and_nonprivate_parent_refuse(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    backup.parent.chmod(0o755)
    with pytest.raises(operator.Cap32Stopped, match="private"):
        operator.cutover(mcp, honcho, backup, apply=True)
    assert not backup.exists()
    backup.parent.chmod(0o700)
    operator.cutover(mcp, honcho, backup, apply=True)
    (backup / "mcp.cap16").write_bytes(b"corrupt")
    with pytest.raises(operator.Cap32Stopped, match="backup differs"):
        operator.cutover(mcp, honcho, backup, rollback=True)
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_32, HONCHO_32)


def test_cli_preview_hides_script_content_and_apply_requires_flag(tmp_path, monkeypatch, capsys):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "MCP", mcp)
    monkeypatch.setattr(operator, "HONCHO", honcho)
    monkeypatch.setattr(operator, "BACKUP", backup)
    monkeypatch.setattr(sys, "argv", ["cap32-cutover"])
    assert operator.main() == 0
    printed = capsys.readouterr().out
    assert "fixture-secret" not in printed
    assert json.loads(printed)["target_sha256"] == operator.NEW
    assert not backup.exists()
    monkeypatch.setattr(sys, "argv", ["cap32-cutover", "--apply"])
    with pytest.raises(SystemExit, match="2"):
        operator.main()
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_16, HONCHO_16)
