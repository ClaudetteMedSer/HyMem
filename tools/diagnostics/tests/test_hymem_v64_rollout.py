"""Local-only rollout controls; no Docker, SSH, or production changes."""
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import ModuleType

import pytest

from tools.diagnostics import hymem_v64_rollout as rollout


@pytest.mark.parametrize("name", ["../hymem/a.py", "/hymem/a.py", "hymem/../a.py",
                                  "hymem/.env", "hymem//a.py", "venv/bin/python", ""])
def test_manifest_rejects_traversal_and_protected_paths(name):
    with pytest.raises(RuntimeError, match="unsafe_manifest_path"):
        rollout.safe_rel(name)


def test_candidate_pin_is_canonical_json_and_source_is_checked(tmp_path):
    root = tmp_path / "source"
    (root / "hymem").mkdir(parents=True)
    source = root / "hymem/example.py"
    source.write_text("reviewed")
    pins = {"hymem/example.py": rollout.sha(source)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(pins, indent=2))
    assert rollout.manifest(manifest, rollout.digest(pins), 1) == pins
    rollout.verify_files(root, pins)
    source.write_text("drift")
    with pytest.raises(RuntimeError, match="source_hash_drift"):
        rollout.verify_files(root, pins)


def test_source_symlink_is_rejected_even_with_matching_bytes(tmp_path):
    (tmp_path / "hymem").mkdir()
    target = tmp_path / "target"
    target.write_text("reviewed")
    (tmp_path / "hymem/a.py").symlink_to(target)
    with pytest.raises(RuntimeError, match="source_path_symlink"):
        rollout.verify_files(tmp_path, {"hymem/a.py": rollout.sha(target)})


def test_private_receipts_are_exclusive_and_reject_unresolved_seals(tmp_path):
    p = tmp_path / "receipt.json"
    rollout.exclusive_json(p, {"status": "passed"})
    assert p.stat().st_mode & 0o777 == 0o600
    assert rollout.read_sealed(p, rollout.sha(p)) == {"status": "passed"}
    with pytest.raises(FileExistsError):
        rollout.exclusive_json(p, {})
    with pytest.raises(RuntimeError, match="unresolved_receipt_pin"):
        rollout.read_sealed(p, None)


def sealed_gates(tmp_path):
    cfg = {"gates": {}}
    for role in ("fullsuite", "paid_postflight"):
        p = tmp_path / (role + ".json")
        body = {"status": "passed", "role": role, "root_reviewed": True,
                "candidate_manifest_sha256": rollout.CANDIDATE_PIN,
                "checks": dict.fromkeys(("claims", "ledger", "canonical", "same_generation", "integrity", "foreign_keys", "episode_vectors"), True),
                "aggregation_passed": True, "paid_calls": 1}
        if role == "fullsuite":
            body.update(collected=100, expected_collected=100, passed=100, failed=0,
                        errors=0, exit_code=0, cleanup_verified=True)
            body.pop("checks")
        p.write_text(json.dumps(body))
        cfg["gates"][role] = {"path": str(p), "sha256": rollout.sha(p)}
    return cfg


@pytest.mark.parametrize("field,value", [("root_reviewed", False), ("candidate_manifest_sha256", "0" * 64),
                                        ("status", "completed"), ("checks", {}), ("paid_calls", 0),
                                        ("aggregation_passed", False)])
def test_historical_or_incomplete_paid_receipt_fails_closed(tmp_path, field, value):
    cfg = sealed_gates(tmp_path)
    rollout.gates(cfg)
    gate = cfg["gates"]["paid_postflight"]
    p = Path(gate["path"])
    body = json.loads(p.read_text())
    body[field] = value
    p.write_text(json.dumps(body))
    gate["sha256"] = rollout.sha(p)
    with pytest.raises(RuntimeError):
        rollout.gates(cfg)


@pytest.mark.parametrize("field,value", [("expected_collected", 101), ("passed", 99),
                                        ("failed", 1), ("errors", 1), ("exit_code", 1),
                                        ("cleanup_verified", False), ("collected", 0)])
def test_fullsuite_gate_requires_exact_collection_and_cleanup(tmp_path, field, value):
    cfg = sealed_gates(tmp_path)
    gate = cfg["gates"]["fullsuite"]
    p = Path(gate["path"])
    body = json.loads(p.read_text())
    body[field] = value
    p.write_text(json.dumps(body))
    gate["sha256"] = rollout.sha(p)
    with pytest.raises(RuntimeError, match="fullsuite_collection_or_cleanup_missing"):
        rollout.gates(cfg)


def test_preserved_inventory_detects_permissions_links_and_config(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    cfg = root / "config"
    cfg.write_text("private")
    link = root / "python"
    link.symlink_to("/usr/bin/python3")
    initial = rollout.digest(rollout.inventory(root))
    cfg.chmod(0o600)
    assert rollout.digest(rollout.inventory(root)) != initial
    initial = rollout.digest(rollout.inventory(root))
    link.unlink()
    link.symlink_to("/usr/bin/false")
    assert rollout.digest(rollout.inventory(root)) != initial


def execute_migration(tmp_path, monkeypatch, defect=None):
    path = tmp_path / "test.sqlite"
    c = sqlite3.connect(path)
    c.executescript("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT);"
                    "INSERT INTO schema_meta VALUES('schema_version','63');"
                    "CREATE TABLE kg_claim_extraction_outcomes(id INTEGER PRIMARY KEY, value TEXT);"
                    "INSERT INTO kg_claim_extraction_outcomes VALUES(1,'durable');"
                    "CREATE TABLE vectors(id INTEGER PRIMARY KEY, bytes BLOB);"
                    "INSERT INTO vectors VALUES(1,x'0020ff');")
    c.close()
    db = ModuleType("hymem.core.db")
    db.connect = sqlite3.connect
    db.schema_version = lambda c: int(c.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0])

    def initialize(c):
        if db.schema_version(c) == 63:
            c.execute("ALTER TABLE kg_claim_extraction_outcomes ADD COLUMN local_replay_proof TEXT")
            c.execute("UPDATE schema_meta SET value='64' WHERE key='schema_version'")
            if defect == "rows":
                c.execute("UPDATE kg_claim_extraction_outcomes SET value='changed'")
            elif defect == "proof":
                c.execute("UPDATE kg_claim_extraction_outcomes SET local_replay_proof=?", ("sha256:" + "0" * 64,))
            c.commit()
        elif defect == "reopen":
            c.execute("INSERT INTO vectors VALUES(2,x'01')")
            c.commit()
    db.initialize = initialize
    package = ModuleType("hymem.core")
    package.db = db
    monkeypatch.setitem(sys.modules, "hymem.core", package)
    exec(compile("DBPATH=" + repr(str(path)) + "\n" + rollout.MIGRATION, "<offline-migration>", "exec"), {})


def test_migration_preserves_existing_columns_blobs_and_null_proofs(tmp_path, monkeypatch, capsys):
    execute_migration(tmp_path, monkeypatch)
    result = json.loads(capsys.readouterr().out)
    assert result["schema_before"] == 63 and result["schema_after"] == 64
    assert result["rows_preserved"] and result["historical_proofs_null"] and result["reopen_stable"]


@pytest.mark.parametrize("defect,error", [("rows", "durable_rows_changed"), ("proof", "historical_proof_minted"),
                                         ("reopen", "reopen_rows_changed")])
def test_migration_fails_on_loss_proof_minting_or_nonidempotence(tmp_path, monkeypatch, defect, error):
    with pytest.raises(RuntimeError, match=error):
        execute_migration(tmp_path, monkeypatch, defect)


def test_offline_container_has_no_network_credentials_or_docker_socket(tmp_path):
    r = object.__new__(rollout.Rollout)
    r.stage = tmp_path / "hymem-v64-rollout-test"
    captured = []
    result = {"status": "passed", "schema_after": 64,
              **dict.fromkeys(("rows_preserved", "historical_proofs_null", "reopen_stable", "integrity_ok", "foreign_keys_ok"), True)}
    r.run = lambda cmd, timeout: captured.append(cmd) or json.dumps(result).encode()
    r.offline(tmp_path / "test.sqlite", tmp_path / "candidate")
    cmd = captured[0]
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--pull" in cmd and "--read-only" in cmd
    assert all("docker.sock" not in arg and "--env-file" not in arg for arg in cmd)
    assert all(".hermes" not in arg for arg in cmd)


def test_intent_blocks_automatic_retry(tmp_path):
    r = object.__new__(rollout.Rollout)
    r.stage = tmp_path
    r.config_sha = "1" * 64
    r.intent("migrate")
    with pytest.raises(FileExistsError):
        r.intent("migrate")


def test_module_and_embedded_migration_compile_without_execution():
    compile(Path(rollout.__file__).read_text(), rollout.__file__, "exec")
    compile(rollout.MIGRATION, "<migration>", "exec")


@pytest.mark.parametrize("collision", ["file", "dangling_symlink"])
def test_new_candidate_paths_cannot_overwrite_untracked_files(tmp_path, monkeypatch, collision):
    monkeypatch.setattr(rollout,"LIVE",tmp_path)
    (tmp_path/"hymem").mkdir()
    old=tmp_path/"hymem/old.py";old.write_text("old")
    new=tmp_path/"hymem/new.py"
    r=object.__new__(rollout.Rollout)
    r.old={"hymem/old.py":rollout.sha(old)}
    r.files={**r.old,"hymem/new.py":"0"*64}
    r.verify_before()
    if collision=="file":new.write_text("untracked work")
    else:new.symlink_to(tmp_path/"missing")
    with pytest.raises(RuntimeError):r.verify_before()
    if collision=="file":assert new.read_text()=="untracked work"
