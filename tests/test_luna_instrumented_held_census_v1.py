"""Invented local stores only; these tests never contact the pilot host."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from benchmarks import lme_diagnostic as classifier
from tools.diagnostics import luna_instrumented_held_census_v1 as census


GENERATION = "hymem-phase1-generation-v1:" + "a" * 64
CACHE = "invented-cache"
SENTINEL = "PRIVATE_SENTINEL_DO_NOT_EXPORT"


def projection():
    scope = {"SCHEMA": census.SCHEMA, "REPORTED_HELD": census.REPORTED_HELD,
             "RETRY_BOUND": census.RETRY_BOUND, "CATEGORIES": census.CATEGORIES,
             "REASON_CODES": census.REASON_CODES}
    exec(compile(census.PROJECTION, "<offline-census>", "exec"), scope)
    return scope["_census"]


def fixture(tmp_path, rows):
    directory = tmp_path / "run" / "q-0001"
    directory.mkdir(parents=True)
    path = directory / "hymem.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE phase1_generations(generation_key TEXT, extraction_cache_key TEXT);
            CREATE TABLE chunks(id TEXT, chunk_kind TEXT, salience_reason TEXT,
                                source_manifest_version TEXT, source_manifest_count INTEGER);
            CREATE TABLE chunk_extraction_attempts(chunk_id TEXT, prompt_version TEXT,
                phase1_generation_key TEXT, attempts INTEGER, last_failure_reason TEXT,
                last_failure_details TEXT);
            CREATE TABLE chunk_extraction_terminal_losses(chunk_id TEXT);
        """)
        conn.execute("INSERT INTO phase1_generations VALUES (?,?)", (GENERATION, CACHE))
        for i, (reason, details) in enumerate(rows):
            chunk_id = str(i)
            conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                         (chunk_id, "extraction", None, "claim-source-manifest-v1", 1))
            conn.execute("INSERT INTO chunk_extraction_attempts VALUES (?,?,?,?,?,?)",
                         (chunk_id, CACHE, GENERATION, 3, reason, details))
        # These should not be included in the current-key census.
        conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                     ("old", "extraction", None, "claim-source-manifest-v1", 1))
        conn.execute("INSERT INTO chunk_extraction_attempts VALUES (?,?,?,?,?,?)",
                     ("old", "other-cache", GENERATION, 3, SENTINEL, SENTINEL))
    summary = {"final_status": {"phase1_generation_key": GENERATION,
                                "in_progress": False}}
    (directory / "private-indexing.json").write_text(json.dumps(summary))
    final = {"pending": {"pending_chunks": 0}, "malformed": {"malformed_chunks": 0},
             "quarantined": {"quarantined_chunks": 79, "quarantined_digests": 0},
             "terminal_loss": {"chunks": 0}, "coverage_integrity": {"failures": 0},
             "summary_health": {"summary_healthy": True}}
    base = {"questions": {"q-0001": {"checkpoint_failure_code": "unspecified_failure",
        "indexing": {"complete": True, "outcome": "failure", "healthy": False,
                     "failure_code": "quarantined_extraction", "cleanup_error_count": 0,
                     "final_status": final}}}}
    namespace = {
        "_read": lambda p, root, cap: json.loads(p.read_text()),
        "_file": lambda p, root, cap: p.is_file() and not p.is_symlink()
            and p.stat().st_size <= cap and p.resolve().is_relative_to(root),
    }
    return path, namespace, base


def run(tmp_path, rows):
    path, namespace, base = fixture(tmp_path, rows)
    before = hashlib.sha256(path.read_bytes()).digest()
    result = projection()(tmp_path, namespace, {}, base, classifier.__dict__,
                          lambda _: base["questions"]["q-0001"]["indexing"]["final_status"])
    assert hashlib.sha256(path.read_bytes()).digest() == before
    assert not path.with_name(path.name + "-shm").exists()
    assert not path.with_name(path.name + "-wal").exists()
    assert census._validated(result) == result
    assert SENTINEL not in json.dumps(result)
    return result


def test_exact_semantic_census(tmp_path):
    result = run(tmp_path, [("incomplete_response", '["provider:finish_length"]')] * 79)
    assert result["superset_equals_reported"] is True
    assert result["semantic_accepted"] == 79
    assert result["reason_counts"]["incomplete_response"] == 79
    assert result["diagnostic_decision_if_exact"] is True


def test_closed_rejection_categories(tmp_path):
    rows = [("incomplete_response", '["provider:finish_length"]')] * 76 + [
        (SENTINEL, SENTINEL), ("parse_failure", "not-json"),
        ("parse_failure", '["source:private_sentinel"]')]
    result = run(tmp_path, rows)
    assert result["semantic_accepted"] == 76
    assert result["semantic_rejected"] == 3
    assert result["rejection_categories"] == {
        "nonsemantic_reason": 1, "invalid_details": 1, "unsupported_details": 1}
    assert result["reason_counts"]["other"] == 1
    assert result["reason_counts"]["parse_failure"] == 2
    assert result["diagnostic_decision_if_exact"] is False


def test_mismatch_is_unknown(tmp_path):
    result = run(tmp_path, [("incomplete_response", '["provider:finish_length"]')] * 80)
    assert result["superset_count"] == 80
    assert result["superset_equals_reported"] is False
    assert result["diagnostic_decision_if_exact"] is None


def test_view_is_not_evaluated(tmp_path):
    path, namespace, base = fixture(tmp_path, [("incomplete_response", "[]")] * 79)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE chunk_extraction_terminal_losses")
        conn.execute("CREATE TABLE private_secret(x TEXT)")
        conn.execute("CREATE VIEW chunk_extraction_terminal_losses AS SELECT x AS chunk_id FROM private_secret")
    with pytest.raises(sqlite3.DatabaseError):
        projection()(tmp_path, namespace, {}, base, classifier.__dict__,
                     lambda _: base["questions"]["q-0001"]["indexing"]["final_status"])


def test_nonempty_wal_and_changed_summary_fail_closed(tmp_path):
    path, namespace, base = fixture(tmp_path, [("incomplete_response", "[]")] * 79)
    final = base["questions"]["q-0001"]["indexing"]["final_status"]
    with pytest.raises(ValueError, match="summary_changed"):
        projection()(tmp_path, namespace, {}, base, classifier.__dict__, lambda _: {})
    wal = path.with_name(path.name + "-wal")
    wal.write_bytes(b"not-read")
    with pytest.raises(ValueError, match="wal_unverified"):
        projection()(tmp_path, namespace, {}, base, classifier.__dict__, lambda _: final)


def test_authorizer_denies_mutating_and_unlisted_sql(tmp_path):
    path, _, _ = fixture(tmp_path, [])
    scope = {"SCHEMA": census.SCHEMA, "REPORTED_HELD": 79,
             "RETRY_BOUND": 3, "CATEGORIES": census.CATEGORIES,
             "REASON_CODES": census.REASON_CODES}
    exec(compile(census.PROJECTION, "<offline-census>", "exec"), scope)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        conn.set_authorizer(scope["_authorize"])
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE chunks SET chunk_kind='changed'")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("ATTACH DATABASE ':memory:' AS extra")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT name FROM sqlite_master")
    finally:
        conn.close()


def test_question_gate_precedes_private_read(tmp_path):
    bad = {"questions": {"q-0001": {"checkpoint_failure_code": "other",
        "indexing": {"final_status": None}}}}
    with pytest.raises(ValueError, match="q1_gate_invalid"):
        projection()(tmp_path, {"_read": lambda *_: pytest.fail("private read occurred")},
                     {}, bad, classifier.__dict__, lambda _: {})


def test_oversized_private_reason_is_not_fetched(tmp_path):
    _, namespace, base = fixture(tmp_path, [("x" * 129, "[]")])
    final = base["questions"]["q-0001"]["indexing"]["final_status"]
    with pytest.raises(ValueError, match="census_input_bound"):
        projection()(tmp_path, namespace, {}, base, classifier.__dict__, lambda _: final)


def test_validator_rejects_free_text_and_inconsistent_decision():
    result = {"schema": census.SCHEMA, "terminal_and_cleanup_verified": True,
              "source_receipt_verified": True, "question": "q-0001",
              "reported_held": 79, "superset_count": 79,
              "superset_equals_reported": True, "semantic_accepted": 79,
              "semantic_rejected": 0, "rejection_categories": dict.fromkeys(census.CATEGORIES, 0),
              "reason_counts": {**dict.fromkeys(census.REASON_CODES, 0),
                                "incomplete_response": 79},
              "diagnostic_decision_if_exact": True}
    assert census._validated(result) == result
    assert census._validated({**result, "secret": SENTINEL}) is None
    assert census._validated({**result, "diagnostic_decision_if_exact": False}) is None
    assert census._validated({**result, "superset_count": True}) is None
