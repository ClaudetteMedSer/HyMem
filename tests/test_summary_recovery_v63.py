"""Explicit recovery advances only its own proof chain, never item indexing."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import sqlite3

import pytest

from hymem.core import db
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import summary_recovery as recovery
from hymem.dreaming.lossless import covered_messages_after, materialize_message_coverage
from hymem.dreaming.summary_state import classify_summary_state
from hymem.session import append_message, open_session
from tests.test_summary_frontier_v62 import _generation


@pytest.fixture
def conn(tmp_path):
    value = db.connect(tmp_path / "summary-recovery.sqlite")
    db.initialize(value)
    yield value
    value.close()


class Replies:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = []
        self.request_attempts = 0

    def complete(self, request):
        self.calls.append(request)
        self.request_attempts += 1
        value = self.values.pop(0) if self.values else {"summary": "All supplied source material remains accurately summarized."}
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value(request)
        return json.dumps(value) if isinstance(value, (dict, list)) else value


def _seed(conn, sid="x", *, long=False):
    open_session(conn, sid)
    first = append_message(conn, sid, "user", "First prior source message.")
    last = append_message(conn, sid, "assistant", "The new source is retained. " + ("Exact long material. " * 70 if long else ""))
    with db.transaction(conn):
        materialize_message_coverage(conn, sid)
    generation = _generation()
    conn.execute(
        "UPDATE sessions SET digest_published_generation=?,digest_published_message_id=?,"
        "digest_cursor_prompt_version=?,digest_cursor_message_id=?,digested_message_id=?,"
        "digested_prompt_version='v1',auto_summary_generation=?,auto_summary_message_id=?,"
        "auto_summary='Prior accepted summary.',summary='Prior accepted summary.',summary_source='auto',"
        "summary_failure_reason='summary_output_cap',summary_failure_count=1 WHERE id=?",
        (generation, last, generation, last, last, generation, first, sid),
    )
    return first, last, generation


def _public(conn, sid="x"):
    return tuple(conn.execute("SELECT summary,summary_source,auto_summary,auto_summary_generation,"
                              "auto_summary_message_id,auto_summary_partial_message_id,auto_summary_message_offset,"
                              "summary_failure_reason,summary_failure_count FROM sessions WHERE id=?", (sid,)).fetchone())


def _items(conn):
    result = {name: [tuple(row) for row in conn.execute(f"SELECT * FROM {name} ORDER BY rowid")]
              for name in ("episodes", "procedures", "procedure_digest_publications", "knowledge_graph")}
    result["cursors"] = [tuple(row) for row in conn.execute(
        "SELECT id,digest_cursor_message_id,digest_cursor_partial_message_id,digest_cursor_offset,"
        "digest_cursor_prompt_version,digest_published_generation,digest_published_message_id,"
        "digested_message_id,digested_prompt_version,digest_retry_count,digest_quarantined FROM sessions ORDER BY id")]
    return result


def _no_lease(conn):
    assert conn.execute("SELECT 1 FROM run_lock WHERE name='dreaming'").fetchone() is None
    assert not conn.in_transaction


def _historical_source_hash(conn, sid, cursor, version):
    """Copy the v1-v4 wire commitment independently of the current helper."""
    encode = lambda value: json.dumps(value, ensure_ascii=True, allow_nan=False,
                                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encode([version, sid]))
    end = cursor[1] if cursor[1] is not None else cursor[0]
    after = None
    while end is not None:
        page = covered_messages_after(conn, sid, after, limit=128, through_message_id=end)
        assert page
        for message in page:
            data = encode(asdict(message))
            digest.update(str(len(data)).encode("ascii") + b":" + data)
        after = page[-1].message_id
        if after == end:
            break
    digest.update(encode(cursor))
    return digest.hexdigest()


def _as_historical_job(conn, version, sid="x"):
    """Simulate a committed legacy walk using its historical source-proof salt."""
    job = recovery._read_job(conn, sid)
    assert job is not None
    job["config_version"] = version + ":" + "a" * 64
    job["source_sha256"] = _historical_source_hash(conn, sid, recovery._position(job), version)
    job["target_source_sha256"] = _historical_source_hash(
        conn, sid, (job["target_message_id"], None, 0), version)
    with db.transaction(conn):
        recovery._save_job(conn, job)
    return job


def test_selective_overview_contract_and_dense_source_publication(conn):
    _seed(conn)
    dense = (
        "Mira proposed a September launch, but nobody approved it. The team decided "
        "to delay deployment until the audit closes. The estimate is perhaps 18 days, "
        "not a commitment. Forty lower-priority log entries list interface colors, "
        "meeting rooms, and repeated calendar details. " * 8
    )
    append_message(conn, "x", "user", dense)
    with db.transaction(conn):
        materialize_message_coverage(conn, "x")
    last = conn.execute("SELECT MAX(id) FROM messages WHERE session_id='x'").fetchone()[0]
    conn.execute("UPDATE sessions SET digest_published_message_id=?,digest_cursor_message_id=?,"
                 "digested_message_id=? WHERE id='x'", (last, last, last))
    before_items = _items(conn)
    overview = ("Mira's proposed September launch was not approved. Deployment is delayed "
                "until audit closure; the 18-day estimate remains uncertain.")
    llm = Replies({"summary": overview})
    assert len(overview) <= 240
    report = recovery.run_summary_recovery(conn, llm, max_chars=20000)
    assert report["published"] == report["calls"] == 1
    assert conn.execute("SELECT auto_summary FROM sessions WHERE id='x'").fetchone()[0] == overview
    assert _items(conn) == before_items
    system = llm.calls[0].system
    assert "selective overview" in system and "Omit examples, enumerations" in system
    assert "non-authoritative" in system and "not a complete inventory" in system
    assert "actor or speaker, polarity, uncertainty, qualification" in system
    assert "Drop peripheral detail before dropping attribution or qualifiers" in system
    assert "Do not invent relationships or causation" in system
    assert "never cut a claim or sentence mid-text" in system
    assert "targeting 180 to 240 code points" in system and "never more than 300" in system
    assert "one or two short, complete sentences" in system
    assert "at most two consequential" in system and "rewrite it shorter" in system
    assert "Do not drop a distinct claim" not in system
    _no_lease(conn)


def test_attributed_update_and_linked_proposal_fit_short_overview(conn):
    _seed(conn)
    last = append_message(
        conn, "x", "user", "Iris verified two receipts absent, leaving five unresolved. "
        "Stale cache is only Iris's hypothesis. The assistant proposed both "
        "refreshing the cache and comparing receipts; neither action ran. "
        "Peripheral shelf and display notes were repeated many times. " * 9,
    )
    with db.transaction(conn):
        materialize_message_coverage(conn, "x")
    conn.execute("UPDATE sessions SET digest_published_message_id=?,digest_cursor_message_id=?,"
                 "digested_message_id=? WHERE id='x'", (last, last, last))
    overview = ("Iris verified two absent receipts; five remain unresolved, and stale cache "
                "is only her hypothesis. The assistant proposed refreshing and comparing "
                "receipts; neither ran.")
    assert len(overview) <= 240
    before = _items(conn)
    llm = Replies({"summary": overview})
    report = recovery.run_summary_recovery(conn, llm, max_chars=12000)
    assert report["calls"] == report["published"] == 1
    assert _items(conn) == before
    assert conn.execute("SELECT auto_summary FROM sessions WHERE id='x'").fetchone()[0] == overview
    wire = json.loads(llm.calls[0].user)
    assert "Iris verified two receipts" in wire["new_material"]
    assert "refreshing the cache and comparing receipts" in wire["new_material"]
    system = llm.calls[0].system
    assert "If a named person made an update" in system
    assert "keep those steps together or omit the proposal" in system
    _no_lease(conn)


def test_multiwindow_overview_retains_qualified_decision_without_public_partial(conn):
    _seed(conn, long=True)
    last = append_message(conn, "x", "assistant", "Audit remains open, so deployment is delayed.")
    with db.transaction(conn):
        materialize_message_coverage(conn, "x")
    conn.execute("UPDATE sessions SET digest_published_message_id=?,digest_cursor_message_id=?,"
                 "digested_message_id=? WHERE id='x'", (last, last, last))
    prior_public = _public(conn)
    before_items = _items(conn)
    llm = Replies()
    # Every private window returns a substantive, bounded overview. The final
    # reply combines earlier qualified facts with a later decision.
    def summarize(request):
        wire = json.loads(request.user)
        if "Audit remains open" in wire["new_material"] or "Audit remains open" in wire["prior_summary"]:
            return {"summary": "The source remains retained. Audit remains open; deployment is delayed."}
        return {"summary": "The source remains retained; no completed deployment is reported."}
    llm.values = [summarize] * 40
    report = recovery.run_summary_recovery(conn, llm, max_calls=1, max_chars=300)
    assert report["advanced"] == 1 and report["published"] == 0
    assert _public(conn) == prior_public and _items(conn) == before_items
    job = recovery._read_job(conn, "x")
    assert job["draft"] and len(job["draft"]) <= 500
    assert json.loads(llm.calls[0].user)["prior_summary"] == ""
    report = recovery.run_summary_recovery(conn, llm, max_calls=40, max_chars=300)
    assert report["published"] == 1 and report["held"] == 0
    assert len(llm.calls) > 2
    assert all(len(json.loads(call.user)["prior_summary"]) <= 500 for call in llm.calls)
    assert any(json.loads(call.user)["prior_summary"] for call in llm.calls[1:])
    assert "Audit remains open; deployment is delayed" in conn.execute(
        "SELECT auto_summary FROM sessions WHERE id='x'").fetchone()[0]
    assert _items(conn) == before_items
    _no_lease(conn)


def test_unicode_cap_is_codepoints_and_never_slices_model_claim(conn):
    _seed(conn)
    old = _public(conn), _items(conn)
    rejected = "Decision remains tentative. " + "🧭" * 475
    assert len(rejected) > 500
    llm = Replies({"summary": rejected})
    report = recovery.run_summary_recovery(conn, llm)
    assert report["held"] == 1 and report["published"] == 0
    assert recovery._read_job(conn, "x")["failure_reason"] == "summary_output_cap"
    assert (_public(conn), _items(conn)) == old
    assert "🧭" not in conn.execute("SELECT auto_summary FROM sessions WHERE id='x'").fetchone()[0]
    _no_lease(conn)


def test_unicode_exact_boundary_is_accepted_without_byte_counting():
    summary = "Decision remains tentative. " + "é" * (500 - len("Decision remains tentative. "))
    assert len(summary) == 500 and len(summary.encode("utf-8")) > 500
    accepted, reason = recovery._parse_summary(json.dumps({"summary": summary}))
    assert reason is None and accepted == summary
    rejected, reason = recovery._parse_summary(json.dumps({"summary": summary + "🧭"}))
    assert rejected is None and reason == "summary_output_cap"


def test_one_call_publication_recovers_summary_without_item_mutation(conn):
    _seed(conn)
    before = _items(conn)
    llm = Replies({"summary": "The first and second messages remain retained."})
    result = recovery.run_summary_recovery(conn, llm)
    assert result == dict(calls=1, provider_attempts=1, provider_attempts_exact=False,
                         advanced=1, published=1, held=0, exhausted=0, remaining=0)
    assert _items(conn) == before
    assert classify_summary_state(conn, "x")["summary_healthy"]
    assert conn.execute("SELECT COUNT(*) FROM summary_recovery").fetchone()[0] == 0
    wire = json.loads(llm.calls[0].user)
    assert wire["prior_summary"] == "" and "First prior source" in wire["new_material"]
    assert "Prior accepted summary." not in llm.calls[0].user
    _no_lease(conn)


def test_private_partial_walk_survives_each_restart(conn, tmp_path):
    _seed(conn, long=True)
    llm = Replies()
    public, items = _public(conn), _items(conn)
    cfg = dict(max_calls=1, max_chars=300)
    result = recovery.run_summary_recovery(conn, llm, **cfg)
    assert result["advanced"] == 1 and result["published"] == 0
    assert _public(conn) == public and _items(conn) == items
    previous = conn.execute("SELECT cursor_message_id,cursor_partial_message_id,cursor_offset FROM summary_recovery").fetchone()
    assert previous[1] is not None
    for _ in range(30):
        # Reopen a genuine independent connection after every durable slice.
        alternate = db.connect(tmp_path / "summary-recovery.sqlite")
        try:
            db.initialize(alternate)
            result = recovery.run_summary_recovery(alternate, llm, **cfg)
            _no_lease(alternate)
        finally:
            alternate.close()
        if result["published"]:
            break
        assert _public(conn) == public and _items(conn) == items
    else:
        pytest.fail("bounded source replay did not converge")
    assert len(llm.calls) > 2
    assert classify_summary_state(conn, "x")["summary_healthy"] and _items(conn) == items


@pytest.mark.parametrize("reply,reason", [
    ("not-json", "parse_failure"), ('{"summary":"cut', "output_truncated"),
    ({"summary": "Valid but extra keys.", "episodes": []}, "shape_failure"),
    ({"summary": None}, "summary_shape_failure"), ({"summary": 3}, "summary_shape_failure"),
    ({"summary": ""}, "summary_validation_failure"), ({"summary": " \n\t "}, "summary_validation_failure"),
    ({"summary": "x"}, "summary_validation_failure"),
    ({"summary": "123456789"}, "summary_validation_failure"),
    ({"summary": '\"' * 12}, "summary_validation_failure"),
    ({"summary": '\"    123456789    \"'}, "summary_validation_failure"),
    ('{"summary":"one","summary":"two"}', "parse_failure"),
    ('Example {"summary":"Looks valid."}', "parse_failure"),
])
def test_model_rejection_holds_exact_cursor_and_public_state(conn, reply, reason):
    _seed(conn, long=True)
    llm = Replies(reply)
    old = _public(conn), _items(conn)
    result = recovery.run_summary_recovery(conn, llm, max_calls=10)
    assert result["calls"] == result["held"] == 1 and result["advanced"] == result["published"] == 0
    job = conn.execute("SELECT * FROM summary_recovery").fetchone()
    assert (job["cursor_message_id"], job["cursor_partial_message_id"], job["cursor_offset"], job["draft"]) == (None, None, 0, "")
    assert job["attempts"] == 1 and job["failure_reason"] == reason
    assert (_public(conn), _items(conn)) == old
    _no_lease(conn)


def test_exhaustion_is_durable_and_limit_change_does_not_reroll(conn):
    _seed(conn)
    llm = Replies("not-json", "not-json", "not-json")
    first = recovery.run_summary_recovery(conn, llm, max_attempts=2)
    second = recovery.run_summary_recovery(conn, llm, max_attempts=2)
    third = recovery.run_summary_recovery(conn, llm, max_attempts=100, max_chars=16000)
    assert first["exhausted"] == 0 and second["exhausted"] == third["exhausted"] == 1
    assert third["calls"] == 0 and len(llm.calls) == 2
    assert third["remaining"] == 1
    _no_lease(conn)


def test_cap_repair_regenerates_original_source_with_numeric_feedback(conn):
    _seed(conn)
    llm = Replies({"summary": "z" * 509}, {"alternatives": ["The original source remains retained."] * 3})
    items = _items(conn)
    report = recovery.run_summary_recovery(conn, llm, max_calls=3)
    assert report["calls"] == report["provider_attempts"] == 2
    assert report["published"] == 1 and report["held"] == 0
    assert json.loads(llm.calls[1].user) == {"original_generation_input": llm.calls[0].user}
    assert "509" in llm.calls[1].system and "by 9" in llm.calls[1].system
    assert "z" * 509 not in llm.calls[1].user + llm.calls[1].system
    assert _items(conn) == items
    _no_lease(conn)


def test_one_call_invocations_use_distinct_reason_only_cap_repair(conn):
    _seed(conn)
    llm = Replies({"summary": "z" * 509}, {"alternatives": ["The original source remains retained."] * 3})
    first = recovery.run_summary_recovery(conn, llm, max_calls=1)
    assert first["calls"] == first["held"] == 1
    job = recovery._read_job(conn, "x")
    assert job["attempts"] == 1 and job["draft"] == ""
    second = recovery.run_summary_recovery(conn, llm, max_calls=1)
    assert second["calls"] == second["published"] == 1
    assert llm.calls[1].system != llm.calls[0].system
    assert "prior response exceeded the output limit" in llm.calls[1].system
    assert "509" not in llm.calls[1].system
    assert json.loads(llm.calls[1].user) == {"original_generation_input": llm.calls[0].user}


@pytest.mark.parametrize("reply,reason", [
    ({"alternatives": ["z" * 501] * 3}, "summary_output_cap"),
    ({"alternatives": ["", "Valid source statement.", "Another valid statement."]}, "summary_validation_failure"),
    ({"alternatives": ["Valid source statement."] * 3, "extra": 1}, "shape_failure"),
])
def test_repaired_rejection_holds_without_third_call(conn, reply, reason):
    _seed(conn)
    llm = Replies({"summary": "z" * 509}, reply)
    old = _public(conn), _items(conn)
    report = recovery.run_summary_recovery(conn, llm, max_calls=10)
    assert report["calls"] == 2 and report["held"] == 1
    assert len(llm.calls) == 2
    job = recovery._read_job(conn, "x")
    assert job["attempts"] == 2 and job["failure_reason"] == reason
    assert job["draft"] == "" and recovery._position(job) == (None, None, 0)
    assert (_public(conn), _items(conn)) == old


def test_cap_attempt_exhaustion_blocks_repair_and_later_budget_increase(conn):
    _seed(conn)
    llm = Replies({"summary": "z" * 509})
    first = recovery.run_summary_recovery(conn, llm, max_calls=10, max_attempts=1)
    second = recovery.run_summary_recovery(conn, llm, max_calls=10, max_attempts=100)
    assert first["calls"] == first["held"] == first["exhausted"] == 1
    assert second["calls"] == 0 and second["exhausted"] == 1
    assert len(llm.calls) == 1


def test_persisted_cap_repair_is_only_call_even_with_larger_invocation_budget(conn):
    _seed(conn)
    llm = Replies({"summary": "z" * 509}, {"alternatives": ["z" * 501] * 3})
    recovery.run_summary_recovery(conn, llm, max_calls=1)
    second = recovery.run_summary_recovery(conn, llm, max_calls=10)
    assert second["calls"] == second["held"] == 1 and len(llm.calls) == 2


def test_oversized_raw_output_gets_no_numeric_repair(conn):
    _seed(conn)
    llm = Replies("z" * 65537)
    report = recovery.run_summary_recovery(conn, llm, max_calls=10)
    assert report["calls"] == report["held"] == 1


def test_cap_repair_partial_window_keeps_original_frontier(conn):
    _seed(conn, long=True)
    llm = Replies({"summary": "z" * 509}, {"alternatives": ["The first source slice is retained."] * 3})
    public, items = _public(conn), _items(conn)
    report = recovery.run_summary_recovery(conn, llm, max_calls=2, max_chars=300)
    assert report["calls"] == 2 and report["advanced"] == 1 and report["published"] == 0
    job = recovery._read_job(conn, "x")
    assert job["cursor_partial_message_id"] is not None and job["cursor_offset"] > 0
    assert job["attempts"] == 0 and job["failure_reason"] is None
    assert job["draft"] == "The first source slice is retained."
    assert (_public(conn), _items(conn)) == (public, items)
    recovery._validate_job(conn, job)


def test_raised_repair_consumes_budget_and_reports_both_calls(conn):
    _seed(conn)
    llm = Replies({"summary": "z" * 509}, RuntimeError("provider uncertainty"))
    with pytest.raises(RuntimeError) as caught:
        recovery.run_summary_recovery(conn, llm, max_calls=10, max_attempts=2)
    assert caught.value.summary_recovery_report["calls"] == 2
    assert recovery._read_job(conn, "x")["attempts"] == 2
    report = recovery.run_summary_recovery(conn, llm, max_calls=10, max_attempts=100)
    assert report["calls"] == 0 and report["exhausted"] == 1
    _no_lease(conn)


def test_raised_repair_preserves_known_cap_reason_for_next_invocation(conn):
    _seed(conn)
    llm = Replies({"summary": "z" * 509}, RuntimeError("provider uncertainty"),
                  {"alternatives": ["The original source remains retained."] * 3})
    before = _public(conn), _items(conn)
    with pytest.raises(RuntimeError) as caught:
        recovery.run_summary_recovery(conn, llm, max_calls=10, max_attempts=3)
    assert caught.value.summary_recovery_report["calls"] == 2
    job = recovery._read_job(conn, "x")
    assert job["attempts"] == 2 and job["failure_reason"] == "summary_output_cap"
    assert job["draft"] == "" and recovery._position(job) == (None, None, 0)
    assert (_public(conn), _items(conn)) == before
    _no_lease(conn)
    report = recovery.run_summary_recovery(conn, llm, max_calls=1, max_attempts=3)
    assert report["calls"] == report["published"] == 1
    assert "prior response exceeded the output limit" in llm.calls[-1].system
    assert json.loads(llm.calls[-1].user) == {"original_generation_input": llm.calls[0].user}


@pytest.mark.parametrize("version", ["summary-recovery-v1", "summary-recovery-v2", "summary-recovery-v3", "summary-recovery-v4", "summary-recovery-v5", "summary-recovery-v6"])
def test_valid_historical_partial_replay_restarts_under_v7_without_mixing_drafts(conn, version):
    _seed(conn, long=True)
    llm = Replies()
    recovery.run_summary_recovery(conn, llm, max_chars=300)
    old = _as_historical_job(conn, version)
    assert old["draft"] and old["cursor_partial_message_id"] is not None
    public, items = _public(conn), _items(conn)
    report = recovery.run_summary_recovery(conn, llm, max_chars=300)
    new = recovery._read_job(conn, "x")
    assert report["advanced"] == 1 and report["published"] == 0
    assert new["config_version"].startswith("summary-recovery-v7:")
    assert new["walk_id"] != old["walk_id"]
    assert json.loads(llm.calls[-1].user)["prior_summary"] == ""
    assert "First prior source message." in json.loads(llm.calls[-1].user)["new_material"]
    assert (_public(conn), _items(conn)) == (public, items)
    _no_lease(conn)


@pytest.mark.parametrize("version", ["summary-recovery-v1", "summary-recovery-v2", "summary-recovery-v3", "summary-recovery-v4", "summary-recovery-v5", "summary-recovery-v6"])
def test_exhausted_historical_same_target_stays_exhausted_on_v7(conn, version):
    _seed(conn)
    llm = Replies("not-json")
    recovery.run_summary_recovery(conn, llm, max_attempts=1)
    old = _as_historical_job(conn, version)
    result = recovery.run_summary_recovery(conn, llm, max_attempts=100, max_chars=12000)
    assert result["calls"] == 0 and result["exhausted"] == result["remaining"] == 1
    assert recovery._read_job(conn, "x") == old and len(llm.calls) == 1
    _no_lease(conn)


@pytest.mark.parametrize("version", ["summary-recovery-v1", "summary-recovery-v2", "summary-recovery-v3", "summary-recovery-v4", "summary-recovery-v5", "summary-recovery-v6"])
def test_exhausted_historical_changed_target_restarts_its_proven_private_walk(conn, version):
    _seed(conn)
    llm = Replies("not-json", {"summary": "A fresh indexed target has a separate recovery walk."})
    recovery.run_summary_recovery(conn, llm, max_attempts=1)
    old = _as_historical_job(conn, version)
    conn.execute("UPDATE sessions SET digest_published_generation=? WHERE id='x'", (_generation("b"),))
    report = recovery.run_summary_recovery(conn, llm, max_attempts=1)
    assert report["calls"] == report["published"] == 1
    assert len(llm.calls) == 2 and recovery._read_job(conn, "x") is None
    assert old["walk_id"]
    _no_lease(conn)


@pytest.mark.parametrize("version,proof", [
    ("summary-recovery-v8", "v1"),
    ("summary-recovery-v1", "v5"),
    ("summary-recovery-v2", "v5"),
    ("summary-recovery-v3", "v5"),
    ("summary-recovery-v4", "v5"),
    ("summary-recovery-v5", "v6"),
    ("summary-recovery-v6", "v7"),
    ("summary-recovery-v1:forged", "v1"),
])
def test_forged_version_or_mismatched_historical_proof_fails_closed(conn, version, proof):
    _seed(conn, long=True)
    llm = Replies()
    recovery.run_summary_recovery(conn, llm, max_chars=300)
    job = recovery._read_job(conn, "x")
    if version in ("summary-recovery-v1", "summary-recovery-v2", "summary-recovery-v3", "summary-recovery-v4", "summary-recovery-v5") and proof in ("v5", "v6"):
        job["config_version"] = version + ":" + "a" * 64
    elif version.endswith(":forged"):
        job["config_version"] = version
    else:
        job["config_version"] = version + ":" + "a" * 64
    with db.transaction(conn):
        recovery._save_job(conn, job)
    before = len(llm.calls), _public(conn), _items(conn)
    with pytest.raises(RuntimeError, match="summary recovery"):
        recovery.run_summary_recovery(conn, llm, max_chars=300)
    assert (len(llm.calls), _public(conn), _items(conn)) == before
    _no_lease(conn)


@pytest.mark.parametrize("version", ["summary-recovery-v1", "summary-recovery-v2", "summary-recovery-v3", "summary-recovery-v4", "summary-recovery-v5"])
def test_historical_recomputed_checksum_does_not_hide_source_proof_change(conn, version):
    _seed(conn, long=True)
    llm = Replies()
    recovery.run_summary_recovery(conn, llm, max_chars=300)
    job = _as_historical_job(conn, version)
    job["source_sha256"] = "0" * 64
    with db.transaction(conn):
        recovery._save_job(conn, job)
    before = len(llm.calls)
    with pytest.raises(RuntimeError, match="retained source changed"):
        recovery.run_summary_recovery(conn, llm, max_chars=300)
    assert len(llm.calls) == before
    _no_lease(conn)


@pytest.mark.parametrize("summary", [
    "1234567890", '\"   1234567890   \"', "  '1234567890'  ",
    "\n  The complete original source remains represented. \t",
])
def test_accepted_summary_uses_digest_normalization(conn, summary):
    from hymem.dreaming.summary import clean_summary
    _seed(conn)
    result = recovery.run_summary_recovery(conn, Replies({"summary": summary}))
    assert result["published"] == 1 and result["held"] == 0
    assert conn.execute("SELECT auto_summary FROM sessions WHERE id='x'").fetchone()[0] == clean_summary(summary)
    assert classify_summary_state(conn, "x")["summary_healthy"]
    _no_lease(conn)


def test_concurrent_raw_append_with_two_session_census_does_not_hold_read_snapshot(conn, tmp_path):
    _seed(conn, "a")
    _seed(conn, "b")
    item_snapshot = _items(conn)
    writer = db.connect(tmp_path / "summary-recovery.sqlite")
    try:
        def append_during_completion(request):
            append_message(writer, "a", "user", "New raw capture beyond the recovery target.")
            return {"summary": "The already indexed source remains represented."}

        result = recovery.run_summary_recovery(conn, Replies(append_during_completion), max_calls=2)
    finally:
        writer.close()
    assert result["calls"] == result["advanced"] == result["published"] == 2
    assert result["remaining"] == 1 and result["held"] == 0
    assert classify_summary_state(conn, "a")["degraded"]
    assert not classify_summary_state(conn, "a")["summary_healthy"]
    assert classify_summary_state(conn, "b")["summary_healthy"]
    assert _items(conn) == item_snapshot
    _no_lease(conn)


def test_exhausted_first_session_does_not_starve_later_work(conn):
    _seed(conn, "a")
    _seed(conn, "b")
    llm = Replies("not-json")
    recovery.run_summary_recovery(conn, llm, max_attempts=1, session_id="a")
    result = recovery.run_summary_recovery(conn, llm, max_attempts=1, max_calls=1)
    assert result["published"] == 1 and result["exhausted"] == result["remaining"] == 1
    assert classify_summary_state(conn, "b")["summary_healthy"]
    assert classify_summary_state(conn, "a")["degraded"]


@pytest.mark.parametrize("field,value", [
    ("draft", "Tampered private summary."), ("source_sha256", "0" * 64),
    ("target_source_sha256", "0" * 64), ("base_sha256", "0" * 64),
    ("cursor_offset", 1), ("attempts", 2), ("state_sha256", "0" * 64),
])
def test_corrupt_private_state_fails_before_paid_work(conn, field, value):
    _seed(conn, long=True)
    llm = Replies()
    recovery.run_summary_recovery(conn, llm, max_chars=300)
    conn.execute(f"UPDATE summary_recovery SET {field}=?", (value,))
    old_calls, old_public = len(llm.calls), _public(conn)
    with pytest.raises(RuntimeError, match="summary recovery"):
        recovery.run_summary_recovery(conn, llm, max_chars=300)
    assert len(llm.calls) == old_calls and _public(conn) == old_public
    _no_lease(conn)


@pytest.mark.parametrize("kind", ["target", "base", "config"])
def test_valid_context_change_restarts_only_private_prefix(conn, kind):
    first, last, generation = _seed(conn, long=True)
    llm = Replies()
    recovery.run_summary_recovery(conn, llm, max_chars=300)
    old_walk = conn.execute("SELECT walk_id FROM summary_recovery").fetchone()[0]
    if kind == "target":
        conn.execute("UPDATE sessions SET digest_published_generation=?", (_generation("b"),))
    elif kind == "base":
        conn.execute("UPDATE sessions SET summary='New exact operator text.',summary_source='operator'")
    item_snapshot = _items(conn)
    recovery.run_summary_recovery(conn, llm, max_chars=301 if kind == "config" else 300)
    assert json.loads(llm.calls[-1].user)["prior_summary"] == ""
    assert "First prior source message." in json.loads(llm.calls[-1].user)["new_material"]
    assert conn.execute("SELECT walk_id FROM summary_recovery").fetchone()[0] != old_walk
    assert _items(conn) == item_snapshot


@pytest.mark.parametrize("failure", [RuntimeError("PRIVATE_PROVIDER_TEXT"), KeyboardInterrupt(), SystemExit(7)])
def test_unknown_provider_failure_is_fatal_accounted_and_cleaned(conn, failure):
    _seed(conn)
    llm = Replies(failure)
    old = _public(conn), _items(conn)
    with pytest.raises(type(failure)) as caught:
        recovery.run_summary_recovery(conn, llm)
    assert caught.value is failure
    assert failure.summary_recovery_report["calls"] == 1
    assert failure.summary_recovery_report["provider_attempts"] == 1
    assert failure.summary_recovery_report["advanced"] == 0
    assert (_public(conn), _items(conn)) == old
    assert conn.execute("SELECT attempts,failure_reason FROM summary_recovery").fetchone()[:] == (1, None)
    _no_lease(conn)


def test_deadline_rejects_late_custom_result_without_publication(conn, monkeypatch):
    _seed(conn)
    clock = [0.0]
    real_deadline = recovery.MonotonicDeadline
    monkeypatch.setattr(recovery, "MonotonicDeadline", type("Clocked", (), {
        "after": staticmethod(lambda seconds: real_deadline(seconds, clock=lambda: clock[0]))}))

    def late(request):
        clock[0] = 20.0
        return {"summary": "This late result must never be published."}

    old = _public(conn)
    with pytest.raises(DeadlineExceeded) as caught:
        recovery.run_summary_recovery(conn, Replies(late), timeout_seconds=10)
    assert caught.value.summary_recovery_report["calls"] == 1
    assert _public(conn) == old
    _no_lease(conn)


def test_busy_lease_is_not_stolen_or_used_for_provider_calls(conn):
    _seed(conn)
    conn.execute("INSERT INTO run_lock(name,acquired_at,holder) VALUES ('dreaming',CURRENT_TIMESTAMP,'other')")
    llm = Replies()
    with pytest.raises(RuntimeError, match="summary_recovery_lease_busy"):
        recovery.run_summary_recovery(conn, llm)
    assert not llm.calls
    assert conn.execute("SELECT holder FROM run_lock").fetchone()[0] == "other"


def test_changed_target_during_completion_cannot_publish_stale_response(conn):
    _seed(conn)
    old = _public(conn)

    def change(request):
        conn.execute("UPDATE sessions SET digest_published_generation=?", (_generation("b"),))
        return {"summary": "The stale result must not replace the old summary."}

    with pytest.raises(RuntimeError, match="target changed"):
        recovery.run_summary_recovery(conn, Replies(change))
    assert _public(conn) == old
    _no_lease(conn)


@pytest.mark.parametrize("key,value", [
    ("max_calls", True), ("max_calls", 0), ("max_calls", 101), ("max_calls", 1.0),
    ("max_attempts", False), ("max_attempts", 0), ("max_attempts", 101),
    ("max_chars", "8000"), ("max_chars", 0), ("max_chars", 1_000_001),
    ("max_tokens", True), ("max_tokens", 32769),
    ("timeout_seconds", True), ("timeout_seconds", float("nan")),
    ("timeout_seconds", float("inf")), ("timeout_seconds", 0), ("timeout_seconds", 3601),
    ("session_id", ""), ("session_id", 1), ("session_id", "a\x00b"),
])
def test_invalid_limits_never_call_provider_or_create_private_state(conn, key, value):
    _seed(conn)
    llm = Replies()
    with pytest.raises(ValueError, match="invalid_bounds"):
        recovery.run_summary_recovery(conn, llm, **{key: value})
    assert not llm.calls
    assert conn.execute("SELECT COUNT(*) FROM summary_recovery").fetchone()[0] == 0
    _no_lease(conn)


def test_v63_private_storage_is_fail_closed_and_guard_heals(conn):
    assert db.schema_version(conn) == db.EXPECTED_SCHEMA_VERSION
    conn.execute("DROP TRIGGER summary_recovery_workspace_guard")
    db.initialize(conn)
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='summary_recovery_workspace_guard'").fetchone()
    conn.execute("DROP TABLE summary_recovery")
    with pytest.raises(RuntimeError, match="v63"):
        db.initialize(conn)
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='summary_recovery'").fetchone() is None


@pytest.mark.parametrize("module,name", [
    ("digest", "_render_message_part"), ("digest", "_largest_part_end"),
    ("digest", "_render_digest_leading_context"), ("summary", "persist_auto_session_summary"),
    ("digest", "_digest_summary_clause_assembly_is_meaningful"), ("summary", "clean_summary"),
])
def test_transitive_window_and_writer_helpers_are_pinned(monkeypatch, module, name):
    from hymem.dreaming import digest, summary
    owner = {"digest": digest, "summary": summary}[module]
    llm = Replies()
    before = recovery._config(llm, 8000, 3072, 3)
    original = getattr(owner, name)

    def changed(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, changed)
    assert recovery._config(llm, 8000, 3072, 3) != before


def test_recovery_only_code_does_not_change_item_generation(monkeypatch):
    from hymem.dreaming.semantic_generation import semantic_generation_suffix
    llm = Replies()
    before = semantic_generation_suffix("digest", llm)
    config = recovery._config(llm, 8000, 3072, 3)
    monkeypatch.setattr(recovery, "SUMMARY_RECOVERY_SYSTEM", recovery.SUMMARY_RECOVERY_SYSTEM + " Independent repair revision.")
    assert recovery._config(llm, 8000, 3072, 3) != config
    assert semantic_generation_suffix("digest", llm) == before


def test_exact_declared_producer_change_during_call_is_fatal(conn):
    from hymem.extraction.llm import StubLLMClient
    _seed(conn)

    class Routed(Replies):
        delegate = StubLLMClient(default="First exact route.")

        def memory_producer_declaration(self):
            return self.delegate.memory_producer_declaration()

        def complete(self, request):
            value = super().complete(request)
            self.delegate = StubLLMClient(default="Changed exact route.")
            return value

    old = _public(conn)
    with pytest.raises(RuntimeError, match="producer_or_implementation_changed") as caught:
        recovery.run_summary_recovery(conn, Routed())
    assert caught.value.summary_recovery_report["calls"] == 1
    assert _public(conn) == old
    _no_lease(conn)


def test_recomputed_private_checksum_does_not_hide_changed_source_commitment(conn):
    _seed(conn, long=True)
    llm = Replies()
    recovery.run_summary_recovery(conn, llm, max_chars=300)
    job = recovery._read_job(conn, "x")
    job["source_sha256"] = "0" * 64
    with db.transaction(conn):
        recovery._save_job(conn, job)
    before = len(llm.calls)
    with pytest.raises(RuntimeError, match="retained source changed"):
        recovery.run_summary_recovery(conn, llm, max_chars=300)
    assert len(llm.calls) == before
    _no_lease(conn)


def test_cleanup_failure_preserves_paid_counters_and_releases_lease(conn, monkeypatch):
    from hymem.dreaming import runner
    _seed(conn)
    original = runner._release_lock

    def fail_after_release(*args):
        original(*args)
        return False

    monkeypatch.setattr(runner, "_release_lock", fail_after_release)
    with pytest.raises(RuntimeError, match="cleanup_failed") as caught:
        recovery.run_summary_recovery(conn, Replies())
    assert caught.value.summary_recovery_report["calls"] == 1
    assert caught.value.summary_recovery_report["provider_attempts"] == 1
    _no_lease(conn)


def test_cleanup_failure_keeps_original_provider_exception(conn, monkeypatch):
    from hymem.dreaming import runner
    _seed(conn)
    original = runner._DreamLeaseHeartbeat.stop

    def fail_after_stop(self):
        original(self)
        raise RuntimeError("PRIVATE_CLEANUP_TEXT")

    monkeypatch.setattr(runner._DreamLeaseHeartbeat, "stop", fail_after_stop)
    failure = RuntimeError("PRIVATE_PROVIDER_TEXT")
    with pytest.raises(RuntimeError) as caught:
        recovery.run_summary_recovery(conn, Replies(failure))
    assert caught.value is failure and failure.summary_recovery_report["calls"] == 1
    assert all("PRIVATE_CLEANUP_TEXT" not in note for note in failure.__notes__)
    _no_lease(conn)


def test_safe_raw_retention_and_curated_summary_are_preserved(conn):
    _seed(conn)
    exact = "Unmodified operator or legacy memory. " * 35
    conn.execute("UPDATE sessions SET summary=?,summary_source='legacy'", (exact,))
    conn.execute("DELETE FROM messages WHERE session_id='x'")
    result = recovery.run_summary_recovery(conn, Replies())
    assert result["published"] == 1
    assert conn.execute("SELECT summary FROM sessions").fetchone()[0] == exact
    assert classify_summary_state(conn, "x")["summary_healthy"]


def test_raw_only_work_does_not_make_provider_calls_or_invent_item_frontier(conn):
    open_session(conn, "raw")
    append_message(conn, "raw", "user", "Not indexed yet.")
    llm = Replies()
    report = recovery.run_summary_recovery(conn, llm)
    assert report["remaining"] == 1 and report["calls"] == 0 and not llm.calls
    assert conn.execute("SELECT digest_published_message_id FROM sessions").fetchone()[0] is None
    _no_lease(conn)


def test_private_recovery_prevents_session_adoption_and_cascades(conn):
    from hymem.session import _session_is_pristine
    _seed(conn, long=True)
    recovery.run_summary_recovery(conn, Replies(), max_chars=300)
    assert not _session_is_pristine(conn, "x")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE sessions SET source_workspace_id='different' WHERE id='x'")
    # Source-proof FKs intentionally prevent deleting real covered history.
    # An otherwise empty placeholder isolates the private-row cascade/guard.
    open_session(conn, "empty")
    job = recovery._read_job(conn, "x")
    job["session_id"] = "empty"
    with db.transaction(conn):
        recovery._save_job(conn, job)
    assert not _session_is_pristine(conn, "empty")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE sessions SET source_workspace_id='different' WHERE id='empty'")
    conn.execute("DELETE FROM sessions WHERE id='empty'")
    assert conn.execute("SELECT COUNT(*) FROM summary_recovery WHERE session_id='empty'").fetchone()[0] == 0


def test_v63_forward_migration_failure_rolls_back_private_table_guard_and_stamp(conn, monkeypatch):
    _seed(conn)
    old_public, old_items = _public(conn), _items(conn)
    conn.execute("DROP TRIGGER summary_recovery_workspace_guard")
    conn.execute("DROP TABLE summary_recovery")
    conn.execute("DELETE FROM schema_meta WHERE key='summary_recovery_schema'")
    conn.execute("UPDATE schema_meta SET value='62' WHERE key='schema_version'")
    original = db._install_summary_recovery_guards

    def interrupted(value):
        original(value)
        raise RuntimeError("injected v63 interruption")

    monkeypatch.setattr(db, "_install_summary_recovery_guards", interrupted)
    with pytest.raises(RuntimeError, match="injected v63"):
        db._run_migrations(conn)
    assert db.schema_version(conn) == 62 and not conn.in_transaction
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='summary_recovery'").fetchone() is None
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='summary_recovery_workspace_guard'").fetchone() is None
    assert (_public(conn), _items(conn)) == (old_public, old_items)
    monkeypatch.setattr(db, "_install_summary_recovery_guards", original)
    db._run_migrations(conn)
    assert db.schema_version(conn) == db.EXPECTED_SCHEMA_VERSION
    assert (_public(conn), _items(conn)) == (old_public, old_items)
