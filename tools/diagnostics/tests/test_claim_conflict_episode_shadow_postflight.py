"""Network-free regression checks against actual durable aggregation output."""
from dataclasses import asdict
from pathlib import Path
import sqlite3

import pytest

from hymem import HyMem, HyMemConfig
from hymem.extraction.llm import StubLLMClient
from hymem.dreaming.aggregate import AggregationResult
from hymem.dreaming.runner import DreamReport, DREAM_REPORT_ERROR_FIELDS, DREAM_REPORT_BOOLEAN_GATE_FIELDS
from tools.diagnostics import claim_conflict_episode_shadow_postflight as checker
from tools.diagnostics.tests.test_claim_conflict_private_dream_postflight import clean_case


@pytest.fixture
def healthy(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path, aggregation_nodes_enabled=True,
        aggregation_digest_enabled=False, profile_extraction_enabled=False,
        facts_extraction_enabled=False), llm=StubLLMClient(default="[]"))
    report = asdict(hy.dream())
    conn = sqlite3.connect(tmp_path / "hymem.sqlite")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM dream_runs ORDER BY id DESC LIMIT 1").fetchone()
    try:
        yield conn, row, report
    finally:
        conn.close()
        hy.close()


def test_real_healthy_strategy_and_durable_publication(healthy):
    conn, row, report = healthy
    assert row["aggregation_blocking"] == "entity_only:no_valid_episode_vectors"
    assert checker.aggregation_evidence(conn, row, report)["healthy"] is True
    default_row = dict(row, aggregation_blocking=AggregationResult(0, 0).blocking)
    assert default_row["aggregation_blocking"] == "exact"
    assert checker.aggregation_evidence(conn, default_row, report)["healthy"] is True


@pytest.mark.parametrize("strategy", ["", "blocked", "knn:disabled", "exact:unknown", "entity_only", "exact:disabled:extra"])
def test_invalid_strategy_fails(healthy, strategy):
    conn, row, report = healthy
    assert checker.aggregation_evidence(conn, dict(row, aggregation_blocking=strategy), report)["healthy"] is False


@pytest.mark.parametrize("field", ["aggregation_generation_key", "aggregation_config_version", "aggregation_material_epoch_key"])
def test_stale_or_missing_run_identity_fails(healthy, field):
    conn, row, report = healthy
    assert checker.aggregation_evidence(conn, dict(row, **{field: None}), report)["healthy"] is False


def test_pending_and_error_attempt_fail(healthy):
    from hymem.dreaming.aggregation_health import begin_aggregation_build, record_aggregation_build_failure
    from hymem.dreaming.aggregation_generation import aggregation_generation_binding
    conn, row, report = healthy
    # Reuse the actual registered generation binding rather than inventing identities.
    import json
    binding = json.loads(conn.execute("SELECT binding_json FROM aggregation_generations WHERE generation_key=?", (row["aggregation_generation_key"],)).fetchone()[0])
    token = begin_aggregation_build(conn, row["aggregation_config_version"], generation_binding=binding)
    assert checker.aggregation_evidence(conn, row, report)["healthy"] is False
    record_aggregation_build_failure(conn, row["aggregation_config_version"], row["aggregation_generation_key"], token, caught_exceptions=1, fusion_failures=1)
    assert checker.aggregation_evidence(conn, row, report)["healthy"] is False


def test_missing_publication_and_wrong_node_count_fail(healthy):
    conn, row, report = healthy
    assert checker.aggregation_evidence(conn, row, dict(report, aggregation_nodes_built=999))["healthy"] is False
    conn.execute("DELETE FROM aggregation_publication_state")
    assert checker.aggregation_evidence(conn, row, report)["healthy"] is False


def test_stale_success_attestation_fails(healthy):
    conn, row, report = healthy
    conn.execute("UPDATE aggregation_build_health SET last_success_at='2000-01-01 00:00:00'")
    assert checker.aggregation_evidence(conn, row, report)["healthy"] is False


def test_disabled_aggregation_cannot_borrow_old_success(healthy):
    conn, row, report = healthy
    assert checker.aggregation_evidence(conn, dict(row, aggregation_effective="disabled"), report)["healthy"] is False


def assess(worker, evidence):
    summary = clean_case()
    before = {"target_current": 0, "target_pinned": 0, "target_pinned_with_proof": 0}
    after = {"target_current": 1, "target_pinned": 1, "target_pinned_with_proof": 1}
    quarantines = {"retry_limit": set(), "terminal_loss": set()}
    return checker.assess(summary, 63, 64, before, after, quarantines, quarantines,
        ("coverage_integrity_failures",), worker, True, evidence)


def test_clean_assessment_accepts_strategy_health():
    report = {k: v for k, v in asdict(DreamReport()).items() if not isinstance(v, str)}
    report["chunks_processed"] = 1
    assert assess(report, {"healthy": True})["status"] == "pass"
    assert assess(report, {"healthy": False})["status"] == "fail"
    for name in DREAM_REPORT_ERROR_FIELDS + DREAM_REPORT_BOOLEAN_GATE_FIELDS:
        changed = dict(report, **{name: True if name in DREAM_REPORT_BOOLEAN_GATE_FIELDS else 1})
        assert assess(changed, {"healthy": True})["status"] == "fail", name


def test_source_seal_and_sidecars_fail_closed(tmp_path):
    source = tmp_path / "source.sqlite"
    source.write_bytes(b"sealed")
    checker.sealed(source, checker.digest(source))
    with pytest.raises(ValueError, match="pin_drift"):
        checker.sealed(source, "0" * 64)
    Path(str(source) + "-wal").write_bytes(b"pending")
    with pytest.raises(ValueError, match="nonempty_sidecar"):
        checker.sealed(source, checker.digest(source))


@pytest.fixture
def vector_shadow(monkeypatch):
    from hymem.core import db
    from hymem.dreaming import aggregate
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    assert db._load_vec_extension(conn)
    conn.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT)")
    conn.executemany("INSERT INTO schema_meta VALUES (?,?)", [
        ("vec_dim", "2"), ("vec_model", "hymem-embedding-producer-v1:" + "a" * 64)])
    conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding float[2])")
    episodes = [{"rowid": 17, "vector": [1.0, 0.0]}]
    monkeypatch.setattr(aggregate, "load_clusterable_episodes", lambda *args, **kwargs: episodes)
    conn.execute("INSERT INTO vec_episodes(rowid,embedding) VALUES (?,?)", (17, db._pack_vector([1.0, 0.0])))
    try:
        yield conn
    finally:
        conn.close()


def test_exact_vector_shadow_is_readonly_and_passes(vector_shadow):
    before = vector_shadow.total_changes
    assert checker.episode_vector_alignment(vector_shadow) is True
    assert vector_shadow.total_changes == before


@pytest.mark.parametrize("mutation", ["surplus", "missing", "different"])
def test_vector_shadow_mismatch_fails_without_repair(vector_shadow, mutation):
    from hymem.core import db
    if mutation == "surplus":
        vector_shadow.execute("INSERT INTO vec_episodes(rowid,embedding) VALUES (?,?)", (18, db._pack_vector([1.0, 0.0])))
    elif mutation == "missing":
        vector_shadow.execute("DELETE FROM vec_episodes WHERE rowid=17")
    else:
        vector_shadow.execute("UPDATE vec_episodes SET embedding=? WHERE rowid=17", (db._pack_vector([0.0, 1.0]),))
    before = vector_shadow.total_changes
    assert checker.episode_vector_alignment(vector_shadow) is False
    assert vector_shadow.total_changes == before


@pytest.mark.parametrize("failure", ["extension", "table", "dimension", "model", "query"])
def test_unverifiable_vector_shadow_fails_closed(vector_shadow, monkeypatch, failure):
    from hymem.core import db
    from hymem.dreaming import aggregate
    if failure == "extension":
        monkeypatch.setattr(db, "_load_vec_extension", lambda conn: False)
    elif failure == "table":
        vector_shadow.execute("DROP TABLE vec_episodes")
    elif failure in {"dimension", "model"}:
        key = "vec_dim" if failure == "dimension" else "vec_model"
        vector_shadow.execute("UPDATE schema_meta SET value='invalid' WHERE key=?", (key,))
    else:
        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("unverifiable")
        monkeypatch.setattr(aggregate, "load_clusterable_episodes", fail)
    assert checker.episode_vector_alignment(vector_shadow) is False
