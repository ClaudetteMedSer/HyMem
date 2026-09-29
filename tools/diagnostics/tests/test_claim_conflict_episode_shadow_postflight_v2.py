"""Real maintained remote identity is reconstructed offline, never spoofed."""
from dataclasses import asdict
import json
import socket
import sys

import pytest

from hymem import HyMem, HyMemConfig
from hymem.extraction.llm import StubLLMClient
from hymem.core import db
from tools.diagnostics import claim_conflict_episode_shadow_postflight_v2 as checker
from tools.diagnostics.tests.test_claim_conflict_private_dream_postflight import clean_case
from hymem.dreaming.runner import DreamReport, DREAM_REPORT_ERROR_FIELDS


def profile_file(tmp_path, **changes):
    value = dict(base_url="https://api.openai.com/v1", model="text-embedding-3-small",
        dim=1536, timeout=10.0, deployment_revision="public-revision-fixture-v1",
        deployment_tenant="public-tenant-fixture-v1", allow_insecure_internal_http=False)
    value.update(changes)
    path = tmp_path / ("profile-" + str(len(list(tmp_path.glob("profile-*")))) + ".json")
    path.write_text(json.dumps(value, sort_keys=True))
    path.chmod(0o400)
    return path


def test_real_remote_publication_needs_exact_live_identity_without_network(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    profile = profile_file(tmp_path, base_url="http://embedding-server:8766/v1", allow_insecure_internal_http=True)
    client = checker.offline_embedding_client(profile, checker.digest(profile))
    wrong_profile = profile_file(tmp_path, base_url="http://embedding-server:8766/v1", allow_insecure_internal_http=True, deployment_revision="different-public-revision")
    wrong = checker.offline_embedding_client(wrong_profile, checker.digest(wrong_profile))
    hy = HyMem(HyMemConfig(root=tmp_path / "store", aggregation_nodes_enabled=True,
        aggregation_digest_enabled=False, profile_extraction_enabled=False,
        facts_extraction_enabled=False), llm=StubLLMClient(default="[]"), embedding_client=client)
    try:
        report = asdict(hy.dream())
        conn = db.connect(tmp_path / "store/hymem.sqlite")
        try:
            row = conn.execute("SELECT * FROM dream_runs ORDER BY id DESC LIMIT 1").fetchone()
            from hymem.dreaming.aggregation_provenance import load_current_aggregation_publication
            assert load_current_aggregation_publication(conn) is None
            assert checker.aggregation_evidence(conn, row, report, client)["publication_matches"] is True
            assert checker.aggregation_evidence(conn, row, report, wrong)["publication_matches"] is False
        finally:
            conn.close()
    finally:
        hy.close()
        client.close()
        wrong.close()


def test_profile_seal_wrong_or_secret_field_fails(tmp_path):
    profile = profile_file(tmp_path)
    with pytest.raises(ValueError, match="pin_drift"):
        checker.offline_embedding_client(profile, "a" * 64)
    profile = profile_file(tmp_path, api_key="forbidden-field")
    with pytest.raises(ValueError, match="fields_invalid"):
        checker.offline_embedding_client(profile, checker.digest(profile))


def test_provider_execution_is_blocked_without_mutating_identity(tmp_path):
    from hymem.dreaming.aggregation_material import embedding_execution_identity
    profile = profile_file(tmp_path)
    client = checker.offline_embedding_client(profile, checker.digest(profile))
    before = embedding_execution_identity(client)
    prior = sys.getprofile()
    try:
        sys.setprofile(checker.offline_provider_guard())
        with pytest.raises(RuntimeError, match="execution_forbidden"):
            client.embed(["must never reach provider"])
    finally:
        sys.setprofile(prior)
        assert embedding_execution_identity(client) == before
        client.close()


def repair_assessment(report, progress):
    before = {"target_current": 0, "target_pinned": 0, "target_pinned_with_proof": 0}
    after = {"target_current": 1, "target_pinned": 1, "target_pinned_with_proof": 1}
    q = {"retry_limit": set(), "terminal_loss": set()}
    return checker.assess(clean_case(), 63, 64, before, after, q, q, (), report, True, {"healthy": True}, progress)


def test_bounded_progress_certifies_only_repair_not_convergence():
    report = {k: v for k, v in asdict(DreamReport()).items() if not isinstance(v, str)}
    report.update(chunks_processed=17, budget_exhausted=True)
    result = repair_assessment(report, {"valid": True, "pending_domains": 3, "progressed_domains": 3})
    assert result["status"] == "pass"
    assert result["repair_passed"] is True
    assert result["convergence_verified"] is False
    assert result["bounded_work_remaining"] is True
    assert result["worker_boolean_gates"]["budget_exhausted"] is True
    assert result["gates"]["worker_boolean_gates_clear"] is False
    for progress in (None, {"valid": False}):
        assert repair_assessment(report, progress)["status"] == "fail"
    for name in DREAM_REPORT_ERROR_FIELDS:
        assert repair_assessment(dict(report, **{name: 1}), {"valid": True})["status"] == "fail"
    for name in ("skipped_locked", "extraction_provider_attempt_budget_exhausted"):
        assert repair_assessment(dict(report, **{name: True}), {"valid": True})["status"] == "fail"


def progress_store(cursor):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    fields = ",".join(domain + suffix + " " + kind for domain in ("digest", "profile", "facts") for suffix, kind in (
        ("_cursor_message_id", "INTEGER"), ("_cursor_partial_message_id", "INTEGER"),
        ("_cursor_offset", "INTEGER"), ("_cursor_prompt_version", "TEXT"),
        ("_retry_count", "INTEGER"), ("_quarantined", "INTEGER")))
    conn.execute("CREATE TABLE sessions(id TEXT," + fields + ")")
    conn.execute("CREATE TABLE chunks(id TEXT,session_id TEXT)")
    conn.execute("CREATE TABLE messages(id INTEGER,session_id TEXT,role TEXT,content TEXT)")
    values = ["session"] + [v for domain in ("digest", "profile", "facts") for v in (cursor, None, 0, domain + "-walk", 0, 0)]
    conn.execute("INSERT INTO sessions VALUES (" + ",".join("?" for _ in values) + ")", values)
    conn.execute("INSERT INTO chunks VALUES ('target','session')")
    conn.executemany("INSERT INTO messages VALUES (?,'session','user','content')", [(1,), (2,), (3,)])
    for domain in ("digest", "profile"):
        conn.execute("CREATE TABLE " + domain + "_staging(session_id TEXT,generation TEXT,cursor_after_message_id INTEGER,cursor_after_partial_message_id INTEGER,cursor_after_offset INTEGER)")
        conn.execute("INSERT INTO " + domain + "_staging VALUES ('session',?,?,NULL,0)", (domain + "-walk", cursor))
    return conn


@pytest.mark.parametrize("mutation", [None, "source", "no_progress", "retry", "quarantine", "offset", "missing_stage"])
def test_durable_bounded_progress_requires_valid_unchanged_source_and_progress(mutation):
    before, after = progress_store(1), progress_store(2)
    try:
        if mutation == "source":
            after.execute("UPDATE messages SET content='changed' WHERE id=1")
        elif mutation == "no_progress":
            before.execute("UPDATE sessions SET digest_cursor_message_id=2,profile_cursor_message_id=2,facts_cursor_message_id=2")
        elif mutation == "retry":
            after.execute("UPDATE sessions SET digest_retry_count=1")
        elif mutation == "quarantine":
            after.execute("UPDATE sessions SET profile_quarantined=1")
        elif mutation == "offset":
            after.execute("UPDATE sessions SET digest_cursor_partial_message_id=3,digest_cursor_offset=999")
        elif mutation == "missing_stage":
            after.execute("DELETE FROM profile_staging")
        result = checker.bounded_progress_evidence(before, after, "target")
        assert result["valid"] is (mutation is None)
    finally:
        before.close()
        after.close()
