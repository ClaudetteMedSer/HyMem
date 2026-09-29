"""No-provider controls for the corrected-schema diagnostic adapter."""
import sqlite3
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_v64_dream as worker
from tools.diagnostics import claim_conflict_v64_dream_host as host


def test_cloned_store_migrates_before_provider_setup(tmp_path, monkeypatch):
    from hymem.core import db

    target = tmp_path / "clone.sqlite"
    source = tmp_path / "reference.sqlite"
    source.write_bytes(b"sealed")

    def clone(path):
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version=63")
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes (chunk_id TEXT)")
        conn.close()

    def initialize(conn):
        conn.execute("ALTER TABLE kg_claim_extraction_outcomes "
                     "ADD COLUMN local_replay_proof TEXT")
        conn.execute("PRAGMA user_version=64")

    monkeypatch.setattr(db, "schema_version", lambda conn: conn.execute(
        "PRAGMA user_version").fetchone()[0])
    monkeypatch.setattr(db, "initialize", initialize)
    old = SimpleNamespace(original_clone_source=clone, SOURCE=source,
                          SOURCE_SHA256="sealed", sha=lambda _path: "sealed")
    worker.upgrade_clone(old, target)
    conn = sqlite3.connect(target)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 64
        assert "local_replay_proof" in {row[1] for row in conn.execute(
            "PRAGMA table_info(kg_claim_extraction_outcomes)")}
    finally:
        conn.close()


def test_wrong_reference_schema_rejected_before_initialize(tmp_path, monkeypatch):
    from hymem.core import db

    target = tmp_path / "clone.sqlite"
    calls = []

    def clone(path):
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version=61")
        conn.close()

    monkeypatch.setattr(db, "schema_version", lambda conn: conn.execute(
        "PRAGMA user_version").fetchone()[0])
    monkeypatch.setattr(db, "initialize", lambda _conn: calls.append("initialize"))
    old = SimpleNamespace(original_clone_source=clone)
    with pytest.raises(RuntimeError, match="reference_schema_not_v63"):
        worker.upgrade_clone(old, target)
    assert calls == []


def test_migration_rejects_invented_historical_proof(tmp_path, monkeypatch):
    from hymem.core import db

    target = tmp_path / "clone.sqlite"
    source = tmp_path / "reference.sqlite"
    source.write_bytes(b"sealed")

    def clone(path):
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version=63")
        conn.execute("CREATE TABLE kg_claim_extraction_outcomes (chunk_id TEXT)")
        conn.execute("INSERT INTO kg_claim_extraction_outcomes VALUES ('opaque')")
        conn.commit()
        conn.close()

    def initialize(conn):
        conn.execute("ALTER TABLE kg_claim_extraction_outcomes "
                     "ADD COLUMN local_replay_proof TEXT")
        conn.execute("UPDATE kg_claim_extraction_outcomes SET local_replay_proof='invented'")
        conn.execute("PRAGMA user_version=64")

    monkeypatch.setattr(db, "schema_version", lambda conn: conn.execute(
        "PRAGMA user_version").fetchone()[0])
    monkeypatch.setattr(db, "initialize", initialize)
    old = SimpleNamespace(original_clone_source=clone, SOURCE=source,
                          SOURCE_SHA256="sealed", sha=lambda _path: "sealed")
    with pytest.raises(RuntimeError, match="clone_historical_proof_invented"):
        worker.upgrade_clone(old, target)


def test_old_instrumentation_pin_matches_local_reviewed_file(monkeypatch):
    diagnostics = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(worker, "OLD_WORKER", diagnostics / "claim_conflict_instrumented_dream.py")
    assert worker.load_old().__file__ == str(worker.OLD_WORKER)


def test_reviewed_adapter_and_suite_pins_match_local_files():
    diagnostics = Path(__file__).resolve().parents[1]
    assert host.sha(diagnostics / "claim_conflict_v64_dream.py") == host.WORKER_SHA
    assert host.sha(diagnostics / "claim_conflict_cold_replay_host.py") == host.PROOF_HOST_SHA
    assert host.sha(diagnostics / "claim_conflict_cold_pytest_v2.py") == host.SUITE_HOST_SHA
    assert host.PROOF_RESULT_SHA is None or host.HEX64.fullmatch(host.PROOF_RESULT_SHA)
    assert host.SUITE_RESULT_SHA is None or host.HEX64.fullmatch(host.SUITE_RESULT_SHA)
    assert type(host.PAYLOAD_TRANSFER_APPROVED) is bool


@pytest.mark.parametrize("missing", ["proof", "suite"])
def test_new_host_is_unreviewed_until_both_offline_receipts(monkeypatch, missing):
    def no_network(*_args, **_kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(host.subprocess, "run", no_network)
    monkeypatch.setattr(host, "PROOF_RESULT_SHA", "a" * 64)
    monkeypatch.setattr(host, "SUITE_RESULT_SHA", "b" * 64)
    monkeypatch.setattr(host, "PROOF_RESULT_SHA" if missing == "proof" else "SUITE_RESULT_SHA", None)
    with pytest.raises(RuntimeError, match=(
        "cold_proof_result_unreviewed" if missing == "proof"
        else "full_suite_result_unreviewed")):
        host.install()


def test_paid_launch_requires_specific_payload_approval(monkeypatch):
    monkeypatch.setattr(host, "PAYLOAD_TRANSFER_APPROVED", False)
    diagnostics = Path(__file__).resolve().parents[1]
    old = host.controller(diagnostics / "claim_conflict_alias_dream_host.py")
    with pytest.raises(RuntimeError, match="exact_payload_transfer_not_approved"):
        old.remote("remote-launch")
    with pytest.raises(RuntimeError, match="exact_payload_transfer_not_approved"):
        old.remote("supervise")


@pytest.mark.parametrize("mode,network", [("offline", "none"), ("live", "hermes-net")])
def test_inherited_launch_is_exactly_bounded_and_mounts_pinned_instrumentation(mode, network):
    diagnostics = Path(__file__).resolve().parents[1]
    old = host.controller(diagnostics / "claim_conflict_alias_dream_host.py")
    helper = SimpleNamespace(RUNTIME=Path("/readonly/runtime"),
                             RUNTIME_ENV=Path("/readonly/env.json"),
                             IMAGE="sha256:" + "a" * 64)
    command, mounts = old.configure(helper, mode)
    assert command[command.index("--network") + 1] == network
    assert command[command.index("--max-http-attempts") + 1] == "704"
    assert command[command.index("--max-llm-http-attempts") + 1] == "192"
    assert command[command.index("--max-embedding-http-attempts") + 1] == "512"
    assert command[command.index("--deadline-seconds") + 1] == "1800"
    assert command[command.index("--phase1-sha256") + 1] == host.PHASE1_SHA
    assert (str(host.OLD_WORKER),
            "/diag/claim_conflict_instrumented_dream_v1.py", False) in mounts
    assert (str(host.WORKER),
            "/diag/claim_conflict_instrumented_dream.py", False) in mounts
    assert (str(host.REFERENCE), "/reference/source.sqlite", False) in mounts
    assert (str(host.CANDIDATE), "/candidate", False) in mounts
    assert (str(host.WORK), "/work", True) in mounts
    assert command[-10:] == ["--phase1-sha256", host.PHASE1_SHA,
                             "--max-http-attempts", "704",
                             "--max-llm-http-attempts", "192",
                             "--max-embedding-http-attempts", "512",
                             "--deadline-seconds", "1800"]


def test_expected_manifest_adds_only_reviewed_v64_overrides(monkeypatch):
    base = {f"f-{index}": "a" * 64 for index in range(479)}
    old = "hymem/dreaming/phase1.py"
    base[old] = "b" * 64
    # Keep inventory size fixed when replacing the seven reviewed files.
    for name in ("hymem/core/db.py", "hymem/core/schema.sql",
                 "hymem/dreaming/evidence.py", "hymem/dreaming/canonicalize.py",
                 "hymem/portability.py"):
        base[name] = "b" * 64
        base.pop(next(iter(base)))
    # The new migration adds one file to the original 480-entry inventory.
    shared = SimpleNamespace(baseline_inventory=lambda _h: base.copy(), OVERRIDE_SHAS={})
    diagnostics = Path(__file__).resolve().parents[1]
    assert host.sha(diagnostics / "claim_conflict_cold_replay_host.py") == host.PROOF_HOST_SHA
    proof_adapter = host.import_pinned(
        diagnostics / "claim_conflict_proof_replay_v2_host.py",
        "e16d27c3cd989b2664a74de19bb524ae685757a9f8fc19e4452e3b3c1d5b6bce",
        "test_v64_overrides")
    proof = SimpleNamespace(reviewed_overrides=lambda: {
        **proof_adapter.reviewed_overrides(), old: host.PHASE1_SHA})
    expected = base.copy()
    expected.update(proof.reviewed_overrides())
    monkeypatch.setattr(host, "CANDIDATE_SHA", hashlib.sha256(json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest())
    assert host.expected_inventory(shared, proof, object())[
        "hymem/dreaming/phase1.py"] == host.PHASE1_SHA
