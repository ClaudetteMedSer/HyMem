from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp")

import hymem.server as srv
from tests.conftest import make_routed_llm


def test_hymem_log_writes_to_session(hy):
    srv.set_hy(hy)
    result = srv._do_log("test-session", "user", "hello world")
    assert result == "logged"
    rows = hy.conn.execute(
        "SELECT role, content FROM messages WHERE session_id='test-session'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "hello world"


def test_hymem_capture_logs_full_conversation(hy):
    triples = [{"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1}]
    hy.set_llm(make_routed_llm(triples, []))
    srv.set_hy(hy)

    messages = json.dumps([
        {"role": "user", "content": "We use uv for python tooling now."},
        {"role": "assistant", "content": "Got it, switching to uv from pip."},
        {"role": "system", "content": "noise"},
        {"role": "weird", "content": "ignored"},
        {"role": "user", "content": ""},
    ])
    result = srv._do_capture("cap-session", messages, dream=True)

    assert "logged 3 turns" in result
    assert "cap-session" in result
    rows = hy.conn.execute(
        "SELECT role FROM messages WHERE session_id='cap-session' ORDER BY id"
    ).fetchall()
    assert [r["role"] for r in rows] == ["user", "assistant", "system"]


def test_hymem_capture_invalid_json_returns_error(hy):
    srv.set_hy(hy)
    result = srv._do_capture("bad-session", "not json", dream=False)
    assert result.startswith("error:")

    result2 = srv._do_capture("bad-session", '{"not": "an array"}', dream=False)
    assert result2.startswith("error:")


def test_hymem_capture_skips_dream_when_false(hy):
    srv.set_hy(hy)
    messages = json.dumps([{"role": "user", "content": "hello"}])
    result = srv._do_capture("nodream", messages, dream=False)
    assert "logged 1 turns" in result
    assert "dreaming" not in result


def test_hymem_dream_returns_summary(hy):
    sid = "s-dream"
    hy.open_session(sid)
    hy.log_message(sid, "assistant", "I'll set up Docker for the local dev environment.")
    hy.log_message(
        sid, "user",
        "No, we use uv and system Python. Don't suggest Docker again.",
    )
    hy.close_session(sid)
    triples = [{"subject": "local_dev", "predicate": "uses", "object": "uv", "polarity": 1}]
    hy.set_llm(make_routed_llm(triples, []))

    srv.set_hy(hy)
    result = srv._do_dream()
    assert "dreaming complete" in result
    assert "sessions" in result and "chunks" in result


def test_hymem_augment_returns_context_string(hy):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('local_dev', 'uses', 'uv', 3, 0, CURRENT_TIMESTAMP)"
    )
    srv.set_hy(hy)
    result = srv._do_augment("tell me about uv tooling")
    assert "uv" in result
    assert "Structured knowledge" in result


def test_hymem_augment_returns_empty_when_no_context(hy):
    srv.set_hy(hy)
    result = srv._do_augment("totally unknown query about nothing")
    assert result == ""


def test_hymem_profile_returns_user_and_memory_md(hy):
    hy.config.user_md_path.write_text("# Behavioral Profile\nPrefers terse code.", encoding="utf-8")
    hy.config.memory_md_path.write_text("# Project Insights\nUses uv for tooling.", encoding="utf-8")
    srv.set_hy(hy)
    result = srv._do_profile()
    assert "USER PROFILE" in result
    assert "PROJECT INSIGHTS" in result
    assert "Prefers terse code." in result
    assert "Uses uv for tooling." in result


def test_hymem_profile_handles_missing_files(hy):
    srv.set_hy(hy)
    assert not hy.config.user_md_path.exists()
    assert not hy.config.memory_md_path.exists()
    result = srv._do_profile()
    assert result == "No profile or insights available yet."


def test_hymem_profile_handles_only_user_md(hy):
    hy.config.user_md_path.write_text("# Behavioral Profile\nPrefers terse code.", encoding="utf-8")
    srv.set_hy(hy)
    result = srv._do_profile()
    assert "USER PROFILE" in result
    assert "PROJECT INSIGHTS" not in result


def test_hymem_alias_registers_mapping(hy):
    srv.set_hy(hy)
    result = srv._do_alias("MedFlow", "med_flow")
    assert "alias registered" in result
    row = hy.conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias='med_flow'"
    ).fetchone()
    assert row["canonical"] == "med_flow"


def test_hymem_retract_returns_no_match_for_missing_edge(hy):
    srv.set_hy(hy)
    result = srv._do_retract("ghost", "uses", "nothing")
    assert result == "no matching active edge found"


def test_hymem_retract_succeeds_for_existing_edge(hy):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence, last_reinforced) "
        "VALUES ('med_flow', 'depends_on', 'redis', 3, 0, CURRENT_TIMESTAMP)"
    )
    srv.set_hy(hy)
    result = srv._do_retract("med_flow", "depends_on", "redis")
    assert result == "retracted"
