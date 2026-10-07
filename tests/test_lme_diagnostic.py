"""The opt-in diagnostic boundary keeps semantic loss separate from corruption."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from benchmarks import lme_diagnostic as diagnostic


def summary(*, chunks=0, degraded=0, missing=0, outcome=None):
    outcome = outcome or ("failure" if chunks else
                          "success_with_summary_degradation" if degraded else "success")
    return {
        "complete": True, "healthy": not chunks,
        "outcome": outcome, "summary_healthy": not degraded,
        "cleanup_errors": [],
        "failure": {"code": "quarantined_extraction"} if chunks else None,
        "final_status": {
            "pending": {"pending_chunks": 0},
            "malformed": {"malformed_summaries": 0},
            "quarantined": {"quarantined_chunks": chunks,
                            "quarantined_digests": 0},
            "terminal_loss": {"chunks": 0},
            "coverage_integrity": {"failures": 0},
            "in_progress": False,
            "phase1_generation_key": "generation",
            "summary_health": {"summary_healthy": not degraded,
                               "summary_degraded_sessions": degraded,
                               "summary_missing_sessions": missing,
                               "malformed_summaries": 0},
        },
    }


def row(reason="parse_failure", details="[]", count=1):
    return {"prompt_version": "prompt", "phase1_generation_key": "generation",
            "last_failure_reason": reason, "last_failure_details": details, "n": count}


def summary_row(reason="parse_failure", count=1, missing=False):
    return {"reason": reason, "count": count, "missing": missing}


def decide(state, rows=(), summary_rows=()):
    return diagnostic.classify_semantic_indexing(
        state, list(rows), list(summary_rows),
        cache_key="prompt", generation_key="generation")


def test_healthy_and_summary_degradation_remain_distinct():
    healthy = decide(summary())
    degraded = decide(summary(degraded=2, missing=1), summary_rows=[
        summary_row(missing=True), summary_row("summary_output_cap")])
    assert healthy["admitted"] and healthy["kind"] == "strict_healthy"
    assert degraded["admitted"] and degraded["kind"] == "summary_degradation"
    assert degraded["summary_degraded_sessions"] == 2
    assert degraded["summary_missing_sessions"] == 1
    assert degraded["summary_failure_reasons"] == {
        "parse_failure": 1, "summary_output_cap": 1}
    assert degraded["mode"] == diagnostic.MODE


def test_semantic_quarantine_never_becomes_healthy_success():
    state = summary(chunks=2, degraded=1, missing=1)
    decision = decide(state, [row("parse_failure"), row("contract_failure")],
                      [summary_row(missing=True)])
    assert decision["admitted"] and decision["kind"] == "semantic_quarantine"
    assert decision["quarantined_chunks"] == 2
    assert decision["semantic_failure_reasons"] == {
        "contract_failure": 1, "parse_failure": 1}
    assert state["outcome"] == "failure" and state["healthy"] is False


@pytest.mark.parametrize("reason", [
    "call_failure", "resource_limit", "source_coverage_failure",
    "internal_validation_failure", "input_contract_failure",
    "unspecified_failure", "branch_incomplete", None,
])
def test_hard_unknown_or_nested_branch_reason_rejected(reason):
    assert decide(summary(chunks=1), [row(reason)])["admitted"] is False


@pytest.mark.parametrize("details", [
    "not-json", "{}", '["diagnostics:truncated"]',
    '["diagnostic:invalid"]', '[42]',
])
def test_malformed_or_truncated_details_rejected(details):
    assert decide(summary(chunks=1), [row(details=details)])["admitted"] is False


@pytest.mark.parametrize("reason", [[], {}, 1, True])
def test_unhashable_or_nonstring_reason_rejected_without_error(reason):
    assert not decide(summary(chunks=1), [row(reason)]) ["admitted"]


@pytest.mark.parametrize("details", [
    '["response:call_failure"]', '["response:transport_failure"]',
    '["response:timeout"]', '["response:rate_limit"]',
    '["response:auth_error"]', '["response:unknown value"]',
])
def test_direct_semantic_reason_with_hard_or_unbounded_detail_rejected(details):
    assert not decide(summary(chunks=1), [row(details=details)])["admitted"]


def test_summary_degradation_requires_typed_record_and_exact_census():
    state = summary(degraded=1, missing=1)
    assert not decide(state)["admitted"]
    assert not decide(state, summary_rows=[summary_row(reason=None, missing=True)])["admitted"]
    assert not decide(state, summary_rows=[summary_row(missing=False)])["admitted"]
    assert not decide(state, summary_rows=[summary_row(count=True, missing=True)])["admitted"]
    assert decide(state, summary_rows=[summary_row(missing=True)])["admitted"]


@pytest.mark.parametrize("reason,details", [
    ("branch_incomplete", '["left:parse_failure","left.response:truncated_json"]'),
    ("branch_incomplete", '["left:branch_incomplete","left.right:shape_failure",'
                          '"left.right.response:not_object"]'),
    ("response_conflict", '["left:parse_failure","left.response:invalid_json",'
                          '"triples:polarity_conflict"]'),
    ("branch_incomplete", '["prepartition_leaf:2","left:parse_failure",'
                          '"left.response:truncated_json"]'),
])
def test_proven_semantic_branches_admitted(reason, details):
    assert decide(summary(chunks=1), [row(reason, details)])["admitted"]


@pytest.mark.parametrize("reason,details", [
    ("response_conflict", '["left:call_failure","triples:polarity_conflict"]'),
    ("response_conflict", '["left:unrecognized_failure","triples:polarity_conflict"]'),
    ("branch_incomplete", '["left:parse_failure","left.diagnostics:truncated"]'),
    ("branch_incomplete", '["left.response:truncated_json"]'),
    ("branch_incomplete", '["left:branch_incomplete","left.right:call_failure"]'),
    ("branch_incomplete", '["left:parse_failure","left.response:unknown_code"]'),
])
def test_unproved_or_mixed_branches_rejected(reason, details):
    assert not decide(summary(chunks=1), [row(reason, details)])["admitted"]


@pytest.mark.parametrize("details", [
    '["grounding:verdict_unsupported"]',
    '["grounding:verdict_uncertain"]',
    '["prepartition_leaf:1","grounding:verdict_unsupported"]',
])
def test_only_staged_semantic_grounding_verdicts_admitted(details):
    assert decide(summary(chunks=1), [row("grounding_failure", details)])["admitted"]


def test_semantic_grounding_verdict_in_failed_branch_admitted():
    details = '["left:grounding_failure","left.grounding:verdict_uncertain"]'
    assert decide(summary(chunks=1), [row("branch_incomplete", details)])["admitted"]


@pytest.mark.parametrize("details", [
    '[]', '["grounding:verdict_other"]', '["grounding:contract_shape"]',
    '["grounding:source_invalid"]', '["grounding:provider_call_failed"]',
    '["grounding:calls_max_exceeded"]',
    '["grounding:verdict_unsupported","grounding:verdict_uncertain"]',
])
def test_unproved_grounding_failure_rejected(details):
    assert not decide(summary(chunks=1), [row("grounding_failure", details)])["admitted"]


def test_mismatched_current_generation_or_census_rejected():
    wrong = row()
    wrong["phase1_generation_key"] = "stale"
    assert not decide(summary(chunks=1), [wrong])["admitted"]
    assert not decide(summary(chunks=2), [row()])["admitted"]


@pytest.mark.parametrize("change", [
    lambda s: s["final_status"]["pending"].update(pending_chunks=1),
    lambda s: s["final_status"]["malformed"].update(malformed_summaries=1),
    lambda s: s["final_status"]["quarantined"].update(quarantined_digests=1),
    lambda s: s["final_status"]["terminal_loss"].update(chunks=1),
    lambda s: s["final_status"]["coverage_integrity"].update(failures=1),
    lambda s: s["final_status"].update(in_progress=True),
    lambda s: s.update(cleanup_errors=[{"stage": "dream_fork_close"}]),
    lambda s: s.update(complete=False),
])
def test_nonsemantic_or_incomplete_state_rejected(change):
    state = summary(chunks=1)
    change(state)
    assert decide(state, [row()])["admitted"] is False


def test_sqlite_held_census_excludes_published_and_terminal_rows():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE chunks(id TEXT PRIMARY KEY, chunk_kind TEXT,
            salience_reason TEXT, source_manifest_version TEXT,
            source_manifest_count INTEGER);
        CREATE TABLE chunk_extraction_attempts(chunk_id TEXT, prompt_version TEXT,
            phase1_generation_key TEXT, attempts INTEGER,
            last_failure_reason TEXT, last_failure_details TEXT);
        CREATE TABLE chunk_extraction_terminal_losses(chunk_id TEXT);
        CREATE TABLE current_phase1_publications(chunk_id TEXT,
            prompt_version TEXT, phase1_generation_key TEXT);
    """)
    for name in ("held", "published", "terminal"):
        conn.execute("INSERT INTO chunks VALUES(?, 'extraction', NULL, 'claim-source-manifest-v1', 1)",
                     (name,))
        conn.execute("INSERT INTO chunk_extraction_attempts VALUES(?, 'prompt', 'generation', 3, 'parse_failure', '[]')",
                     (name,))
    conn.execute("INSERT INTO current_phase1_publications VALUES('published', 'prompt', 'generation')")
    conn.execute("INSERT INTO chunk_extraction_terminal_losses VALUES('terminal')")
    assert diagnostic._held_rows(conn, 3) == [row()]
    assert diagnostic._held_rows(conn, 0) == []
    conn.close()


def test_sqlite_summary_census_uses_typed_record_and_same_snapshot():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE sessions(id TEXT,summary_failure_reason TEXT,summary_failure_count INTEGER)")
    conn.executemany("INSERT INTO sessions VALUES(?,?,?)", [
        ("missing", "parse_failure", 2),
        ("stale", "summary_output_cap", 1),
        ("healthy", None, 0),
    ])
    def classifier(snapshot, session_id):
        assert snapshot is conn
        return {"summary_healthy": session_id == "healthy",
                "degraded": session_id != "healthy",
                "missing": session_id == "missing", "malformed": False}
    rows = diagnostic._summary_rows(conn, classifier)
    assert rows == [summary_row("parse_failure", 2, True),
                    summary_row("summary_output_cap", 1, False)]
    assert decide(summary(degraded=2, missing=1), summary_rows=rows)["admitted"]
    conn.close()


def test_adapter_always_closes_fork_and_rejects_snapshot_drift(monkeypatch):
    events = []
    state = summary()
    state["final_status"].update(extraction_cache_key="prompt")
    class IntegrityError(Exception):
        pass
    class ConvergenceError(IntegrityError):
        def __init__(self, message, evidence):
            super().__init__(message)
            self.summary = evidence
    class Memory:
        def dream(self, **_kwargs):
            return None
        def close(self):
            events.append("fork_close")
    memory = Memory()
    class Owner:
        def __init__(self):
            self.hy = SimpleNamespace(fork=lambda: memory,
                                      invalidate_query_caches=lambda: events.append("invalidate"))
            self.embedding_client = None
            self.last_indexing_summary = None
    def cleanup(actions, **_kwargs):
        for _name, action in actions:
            action()
    lme = SimpleNamespace(
        converge_indexing=lambda *_a, **_k: dict(state),
        durable_indexing_status=lambda *_a, **_k: state["final_status"],
        canonicalize_lme_indexing_summary=lambda value: dict(value),
        IndexingConvergenceError=ConvergenceError,
        BenchmarkIntegrityError=IntegrityError,
        LME_INDEXING_SUMMARY_VERSION="test", run_cleanup_actions=cleanup,
    )
    protocol = SimpleNamespace(_validate_versioned_indexing=lambda *_a, **_k: True)
    cls = diagnostic.make_diagnostic_adapter_class(
        lme, protocol, SimpleNamespace(), lambda *_a: {}, Owner)
    owner = cls()
    monkeypatch.setattr(diagnostic, "coherent_status_and_held",
                        lambda *_a: ({**state["final_status"], "pending": {"pending_chunks": 1}}, [], []))
    with pytest.raises(IntegrityError, match="snapshot changed"):
        owner.dream_and_wait(timeout=10, max_cycles=2)
    assert events == ["fork_close", "invalidate"]
