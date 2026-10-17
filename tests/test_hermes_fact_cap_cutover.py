"""Guarded two-script Hermes fact-cap edit and restoration."""

import hashlib
import json
import os
import stat
import sys

import pytest

from tools.diagnostics import hermes_fact_cap_cutover as operator


MCP_OLD = (
    b"#!/bin/bash\nexport PRIVATE_TOKEN=fixture-secret\n"
    b"export HYMEM_AGGREGATION_NODES_ENABLED=1\nexec hymem-server\n"
)
HONCHO_OLD = (
    b"#!/bin/bash\nenv \\\n"
    b"      PRIVATE_TOKEN=fixture-secret \\\n"
    b"      /home/node/hymem-env/bin/hymem-honcho\n"
)


def _fixture(tmp_path, monkeypatch):
    mcp, honcho = tmp_path / "wrapper", tmp_path / "post-restart.sh"
    for path, payload in ((mcp, MCP_OLD), (honcho, HONCHO_OLD)):
        path.write_bytes(payload)
        path.chmod(0o755)
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    backup = parent / "fact-cap-attempt"
    monkeypatch.setattr(operator, "EXPECTED_OLD", {
        "mcp": hashlib.sha256(MCP_OLD).hexdigest(),
        "honcho": hashlib.sha256(HONCHO_OLD).hexdigest(),
    })
    return mcp, honcho, backup


@pytest.mark.parametrize("cap", [16, 32])
def test_preview_apply_and_guarded_restore(tmp_path, monkeypatch, cap):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    preview = operator.cutover(cap, mcp, honcho, backup)
    assert not preview["applied"]
    assert not backup.exists()
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_OLD, HONCHO_OLD)
    applied = operator.cutover(cap, mcp, honcho, backup, apply=True)
    assert applied["applied"] and not applied["rolled_back"]
    assert f"export HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap}\n".encode() in mcp.read_bytes()
    assert f"      HYMEM_DREAM_MAX_FACTS_PER_SESSION={cap} \\\n".encode() in honcho.read_bytes()
    assert stat.S_IMODE(mcp.stat().st_mode) == 0o755
    assert stat.S_IMODE(honcho.stat().st_mode) == 0o755
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    for role, payload in (("mcp", MCP_OLD), ("honcho", HONCHO_OLD)):
        saved = backup / f"{role}.before"
        assert saved.read_bytes() == payload
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    restored = operator.cutover(cap, mcp, honcho, backup, rollback=True)
    assert restored["rolled_back"] and not restored["applied"]
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_OLD, HONCHO_OLD)


def test_second_apply_replacement_failure_restores_first(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    actual = operator.os.replace
    calls = []

    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("injected second replacement failure")
        return actual(source, destination)

    monkeypatch.setattr(operator.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        operator.cutover(16, mcp, honcho, backup, apply=True)
    assert calls == [mcp, honcho, mcp]
    assert (mcp.read_bytes(), honcho.read_bytes()) == (MCP_OLD, HONCHO_OLD)
    assert (backup / "mcp.before").read_bytes() == MCP_OLD
    assert (backup / "honcho.before").read_bytes() == HONCHO_OLD


def test_second_rollback_replacement_failure_restores_cap(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    operator.cutover(16, mcp, honcho, backup, apply=True)
    new_bytes = (mcp.read_bytes(), honcho.read_bytes())
    actual = operator.os.replace
    calls = []

    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("injected second rollback failure")
        return actual(source, destination)

    monkeypatch.setattr(operator.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        operator.cutover(16, mcp, honcho, backup, rollback=True)
    assert calls == [mcp, honcho, mcp]
    assert (mcp.read_bytes(), honcho.read_bytes()) == new_bytes


def test_stale_script_or_backup_refuses_without_changes(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    mcp.write_bytes(MCP_OLD + b"# drift\n")
    with pytest.raises(operator.CapCutoverStopped, match="baseline"):
        operator.cutover(16, mcp, honcho, backup, apply=True)
    assert not backup.exists()
    mcp.write_bytes(MCP_OLD)
    operator.cutover(16, mcp, honcho, backup, apply=True)
    capped = (mcp.read_bytes(), honcho.read_bytes())
    (backup / "mcp.before").write_bytes(b"corrupt")
    with pytest.raises(operator.CapCutoverStopped, match="saved launch script"):
        operator.cutover(16, mcp, honcho, backup, rollback=True)
    assert (mcp.read_bytes(), honcho.read_bytes()) == capped
    (backup / "mcp.before").write_bytes(MCP_OLD)
    with pytest.raises(operator.CapCutoverStopped, match="selected cap"):
        operator.cutover(32, mcp, honcho, backup, rollback=True)
    assert (mcp.read_bytes(), honcho.read_bytes()) == capped


def test_missing_anchor_and_duplicate_assignment_refuse(tmp_path, monkeypatch):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    altered = MCP_OLD.replace(
        b"export HYMEM_AGGREGATION_NODES_ENABLED=1\n", b""
    )
    monkeypatch.setitem(operator.EXPECTED_OLD, "mcp", hashlib.sha256(altered).hexdigest())
    mcp.write_bytes(altered)
    with pytest.raises(operator.CapCutoverStopped, match="anchor"):
        operator.cutover(16, mcp, honcho, backup)
    assert not backup.exists()
    altered = MCP_OLD + b"export HYMEM_DREAM_MAX_FACTS_PER_SESSION=8\n"
    monkeypatch.setitem(operator.EXPECTED_OLD, "mcp", hashlib.sha256(altered).hexdigest())
    mcp.write_bytes(altered)
    with pytest.raises(operator.CapCutoverStopped, match="already contains"):
        operator.cutover(16, mcp, honcho, backup)


def test_cli_emits_hashes_without_script_content(tmp_path, monkeypatch, capsys):
    mcp, honcho, backup = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(operator, "MCP", mcp)
    monkeypatch.setattr(operator, "HONCHO", honcho)
    monkeypatch.setattr(operator, "BACKUP", backup)
    monkeypatch.setattr(sys, "argv", ["cap-cutover", "--cap", "16"])
    assert operator.main() == 0
    printed = capsys.readouterr().out
    assert "fixture-secret" not in printed
    assert json.loads(printed)["applied"] is False
