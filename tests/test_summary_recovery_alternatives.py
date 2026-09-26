"""Repair-only whole-option selection; scripted replies do not prove model fidelity."""
import json

import pytest

from hymem.dreaming import summary_recovery as recovery
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import StubLLMClient
from hymem.core import db
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.session import append_message
from tests.test_summary_recovery_v63 import (
    conn, Replies, _seed, _public, _items, _no_lease, _as_historical_job,
)


GOOD = "Iris's cache explanation remains a hypothesis; five receipts are unresolved."


def options(*values):
    return {"alternatives": list(values or [GOOD] * 3)}


@pytest.mark.parametrize("selected", [0, 1, 2])
def test_first_fitting_whole_option_is_returned_without_joining(selected):
    values = ["a" * 501, "b" * 519, "c" * 505]
    values[selected] = GOOD
    if selected < 2:
        values[2] = "A distinct complete lower-priority statement."
    assert recovery._parse_repair_alternatives(json.dumps(options(*values))) == (GOOD, None)


@pytest.mark.parametrize("unit", ["é", "🧭", "e\u0301"])
def test_exact_unicode_codepoint_boundary_never_slices(unit):
    text = (unit * 500)[:500]
    accepted, reason = recovery._parse_repair_alternatives(
        json.dumps(options(text + unit, " \n" + text + "\t", GOOD)))
    assert accepted == text and len(accepted) == 500 and reason is None


@pytest.mark.parametrize("bad,reason", [
    (None, "summary_shape_failure"), (42, "summary_shape_failure"),
    ([], "summary_shape_failure"), ("", "summary_validation_failure"),
    (" \n\t", "summary_validation_failure"), ("tiny", "summary_validation_failure"),
    ('"         tiny         "', "summary_validation_failure"),
    ("'              '", "summary_validation_failure"),
])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_invalid_unused_member_rejects_the_whole_response(bad, reason, position):
    values = [GOOD] * 3
    values[position] = bad
    assert recovery._parse_repair_alternatives(json.dumps(options(*values))) == (None, reason)


@pytest.mark.parametrize("data", [
    {}, [], {"summary": GOOD}, {"alternatives": GOOD},
    {"alternatives": [GOOD] * 2}, {"alternatives": [GOOD] * 4},
    {"alternatives": [GOOD] * 3, "summary": GOOD},
    {"alternatives": [GOOD] * 3, "episodes": []},
])
def test_repair_envelope_is_exact(data):
    assert recovery._parse_repair_alternatives(json.dumps(data)) == (None, "shape_failure")


@pytest.mark.parametrize("raw,reason", [
    ('{"alternatives":["cut', "output_truncated"),
    ('prose {"alternatives":[]}', "parse_failure"),
    ('{"alternatives":[],"alternatives":[]}', "parse_failure"),
    ("x" * 65537, "summary_output_cap"),
], ids=["truncated", "surrounding-prose", "duplicate-key", "raw-ceiling"])
def test_parse_and_raw_ceiling_failures_remain_honest(raw, reason):
    assert recovery._parse_repair_alternatives(raw) == (None, reason)


def test_no_fit_is_not_clipped_or_laundered():
    assert recovery._parse_repair_alternatives(json.dumps(
        options("a" * 501, "b" * 505, "c" * 519))) == (None, "summary_output_cap")


def test_primary_does_not_accept_repair_shape():
    assert recovery._parse_summary(json.dumps(options())) == (None, "shape_failure")


def test_repair_publication_preserves_source_input_and_item_state(conn):
    _seed(conn)
    last = append_message(conn, "x", "user", "Iris verified two absent receipts; five remain unresolved. "
                          "Stale cache is only Iris's hypothesis.")
    with db.transaction(conn):
        materialize_message_coverage(conn, "x")
    conn.execute("UPDATE sessions SET digest_published_message_id=?,digest_cursor_message_id=?,"
                 "digested_message_id=? WHERE id='x'", (last, last, last))
    items = _items(conn)
    llm = Replies({"summary": "x" * 505}, options("a" * 519, GOOD, "Five receipts remain unresolved."))
    report = recovery.run_summary_recovery(conn, llm, max_calls=10)
    assert report["calls"] == report["provider_attempts"] == 2
    assert report["published"] == 1 and report["held"] == 0
    assert conn.execute("SELECT auto_summary FROM sessions WHERE id='x'").fetchone()[0] == GOOD
    assert _items(conn) == items
    assert json.loads(llm.calls[1].user) == {"original_generation_input": llm.calls[0].user}
    assert "x" * 505 not in llm.calls[1].system + llm.calls[1].user
    assert "Stale cache is only Iris's hypothesis." in json.loads(llm.calls[0].user)["new_material"]
    assert "505" in llm.calls[1].system and "by 5" in llm.calls[1].system
    assert "exactly three" in llm.calls[1].system
    assert "descending detail" in llm.calls[1].system
    assert "omit a secondary proposition entirely" in llm.calls[1].system
    assert llm.calls[0].max_tokens == llm.calls[1].max_tokens == 3072
    _no_lease(conn)


def test_all_over_cap_consumes_only_one_repair_and_exhausts_durably(conn):
    _seed(conn)
    old = _public(conn), _items(conn)
    llm = Replies({"summary": "x" * 505}, options(*["y" * 519] * 3), options(*["z" * 501] * 3))
    first = recovery.run_summary_recovery(conn, llm, max_calls=10)
    second = recovery.run_summary_recovery(conn, llm, max_calls=10)
    third = recovery.run_summary_recovery(conn, llm, max_calls=100, max_attempts=100)
    assert (first["calls"], second["calls"], third["calls"]) == (2, 1, 0)
    assert first["held"] == second["exhausted"] == third["exhausted"] == 1
    assert (_public(conn), _items(conn)) == old
    assert recovery._position(recovery._read_job(conn, "x")) == (None, None, 0)
    _no_lease(conn)


def test_v6_exhausted_job_remains_readable_without_reset(conn):
    _seed(conn)
    llm = Replies("not-json")
    recovery.run_summary_recovery(conn, llm, max_attempts=1)
    old = _as_historical_job(conn, "summary-recovery-v6")
    report = recovery.run_summary_recovery(conn, llm, max_attempts=100)
    assert report["calls"] == 0 and report["exhausted"] == 1
    assert recovery._read_job(conn, "x") == old


def test_repair_parser_change_leaves_normal_generation_identity_unchanged(monkeypatch):
    llm = StubLLMClient(default="[]")
    normal = semantic_generation_suffix("digest", llm)
    config = recovery._config(llm, 8000, 3072, 3)
    original = recovery._parse_repair_alternatives
    def revised(raw):
        return original(raw)
    monkeypatch.setattr(recovery, "_parse_repair_alternatives", revised)
    assert semantic_generation_suffix("digest", llm) == normal
    assert recovery._config(llm, 8000, 3072, 3) != config


def test_complete_shorter_options_keep_attribution_qualification_and_linked_steps():
    detailed = ("Iris verified two absent receipts; five remain unresolved, and stale cache "
                "is only her hypothesis. The assistant proposed refreshing the cache and "
                "comparing receipts; neither action ran.")
    compact = ("Iris's stale-cache explanation is a hypothesis. The assistant proposed both "
               "refreshing the cache and comparing receipts; neither ran.")
    shortest = "Iris's stale-cache explanation remains a hypothesis."
    values = [detailed, compact, shortest]
    assert all(len(value) <= 500 for value in values)
    assert recovery._parse_repair_alternatives(json.dumps(options(*values))) == (detailed, None)
    request = recovery._cap_recovery_request(recovery.LLMRequest(system="", user="source"), 505)
    for obligation in ("actor or speaker, polarity, uncertainty, qualification",
                       "keep those steps together or omit", "never cut a claim or sentence"):
        assert obligation in request.system
