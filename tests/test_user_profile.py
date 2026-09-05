"""Offline (StubLLM, no box) tests for the typed user-profile tier
(Stage 1 / P4, schema v18): hymem/dreaming/user_profile.py plus its three
additive consumers — the VERIFIED FACTS anchor in aggregate.py, the
`ctx.user_profile` tier in augment(), and `HyMem.profile()`.

Covers: fixture-driven extraction through a real dream (USER turns only in
the prompt), strict validation (closed slot vocabulary, evidence ids,
confidence range), bi-temporal supersession (single-valued slots and
relationship-per-person supersede; other multi-valued slots accumulate;
re-assertion never duplicates), anchor priority order + combined cap,
the additive augment tier (no slot competition with existing tiers),
redaction of sensitive values in every consumer, the empty-before-dream
contract, the off-by-default gate (profile.v1 failed the on-box precision
gate, so profile_extraction_enabled defaults False until profile.v2
re-passes), the decoupled per-session profile_prompt_version skip-guard
(schema v19), and the v18/v19 migrations on fresh and existing stores.

NOTE: extraction-path tests opt in with profile_extraction_enabled=True —
the shipped default is False.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming.aggregate import _anchor_facts, build_aggregation_nodes, load_digest
from hymem.dreaming import runner as dreaming_runner
from hymem.dreaming.user_profile import (
    PROFILE_PROMPT_VERSION,
    ProfileExtraction,
    build_profile_user_prompt,
    extract_user_profile,
    load_profile,
    persist_user_profile,
    profile_config_version,
    profile_generation_matches_config,
    publish_profile_generation,
    render_profile_fact,
    stage_profile_extraction,
    validate_profile_items,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient
from tests.conftest import seed_edge

# Routing substring unique to USER_PROFILE_SYSTEM (the digest/triple stubs key
# on their own closers, so the fixtures never collide).
_NEEDLE = "typed user-profile facts"


def _on(cfg):
    """Config with profile extraction opted in (the shipped default is False
    until the profile.v2 prompt re-passes the on-box precision gate)."""
    return replace(cfg, profile_extraction_enabled=True)


def _profile_calls(stub_llm) -> list:
    return [c for c in stub_llm.calls if _NEEDLE in c.system]


def _stamp(conn, sid) -> str | None:
    row = conn.execute(
        "SELECT profile_prompt_version FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    return row["profile_prompt_version"] if row else None


def _item(slot, value, mid, *, key=None, conf=0.9) -> dict:
    out = {"slot": slot, "value": value, "evidence_message_id": mid, "confidence": conf}
    if key is not None:
        out["slot_key"] = key
    return out


def _answer(items) -> str:
    """Strict production profile response envelope."""
    return json.dumps({"items": items})


def _profile_llm(answer: str) -> StubLLMClient:
    return StubLLMClient(
        fixtures={
            _NEEDLE: answer,
            "single pass": json.dumps({"triples": [], "markers": []}),
            "Return the JSON object now": json.dumps({
                "episodes": [], "summary": "", "procedures": [],
            }),
        },
        default="[]",
    )


def _quiet_profile_cfg(cfg, **changes):
    return replace(
        cfg,
        aggregation_nodes_enabled=False,
        facts_extraction_enabled=False,
        **changes,
    )


class _CapturingProfileLLM:
    def __init__(self, item_factory=None):
        self.calls = []
        self.fragments: list[str] = []
        self.item_factory = item_factory

    def complete(self, request):
        self.calls.append(request)
        if _NEEDLE in request.system:
            match = re.search(
                r'(?:"""\n|\n)\[msg \d+\] ([\s\S]*?)\n"""',
                request.user,
            )
            assert match is not None
            self.fragments.append(match.group(1))
            items = self.item_factory(len(self.fragments)) if self.item_factory else []
            return _answer(items)
        if "single pass" in request.system:
            return json.dumps({"triples": [], "markers": []})
        if request.system.startswith((
            "You analyze one conversation session",
            "You re-read one conversation session",
        )):
            return json.dumps({"episodes": [], "summary": "", "procedures": []})
        return "[]"


def _persist(conn, items, *, redact=True) -> int:
    # Direct persistence is allowed only for claims whose source has a
    # producer-bounded durable USER artifact, just like the real runner.
    sessions = {
        row["session_id"]
        for item in items
        for row in [conn.execute(
            "SELECT session_id FROM messages WHERE id = ? AND role = 'user'",
            (item["evidence_message_id"],),
        ).fetchone()]
        if row is not None
    }
    with core_db.transaction(conn):
        for session_id in sessions:
            materialize_message_coverage(conn, session_id)
    with core_db.transaction(conn):
        return persist_user_profile(
            conn, ProfileExtraction(items=items), redact_values=redact
        )


def _msg(conn, sid, content, *, role="user", created_at=None) -> int:
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (sid,))
    if created_at is not None:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (sid, role, content, created_at),
        )
    else:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
            (sid, role, content),
        )
    return cur.lastrowid


def _rows(conn) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM user_profile ORDER BY id"
    ).fetchall()


def _cover(conn, sid: str) -> None:
    with core_db.transaction(conn):
        materialize_message_coverage(conn, sid)


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    yield hy.conn
    hy.close()


# ── extraction via a real dream (USER turns only) ────────────────────────────


def test_dream_extracts_profile_from_user_turns_only(cfg, stub_llm):
    hy = HyMem(_on(cfg), llm=stub_llm)
    try:
        sid = "s_profile"
        m_user = hy.log_message(
            sid, "user",
            "Tegenwoordig werk ik als bedrijfsarts bij MedFlow in Amsterdam.",
        )
        hy.log_message(
            sid, "assistant",
            "Mooi! Occupational health at MedFlow sounds like rewarding work.",
        )
        stub_llm.fixtures[_NEEDLE] = _answer([
            _item("role", "bedrijfsarts", m_user),
            _item("employer", "MedFlow", m_user),
        ])

        report = hy.dream()
        assert report.profile_items_extracted == 2

        rows = _rows(hy.conn)
        assert {(r["slot"], r["value"]) for r in rows} == {
            ("role", "bedrijfsarts"), ("employer", "MedFlow"),
        }
        assert all(r["invalid_at"] is None for r in rows)
        assert all(r["evidence_message_id"] == m_user for r in rows)

        # The extraction prompt carried ONLY user turns, tagged with real ids.
        profile_calls = [c for c in stub_llm.calls if _NEEDLE in c.system]
        assert len(profile_calls) == 1
        prompt = profile_calls[0].user
        assert f"[msg {m_user}]" in prompt
        assert "bedrijfsarts bij MedFlow" in prompt
        assert "rewarding work" not in prompt, "assistant turns must not be in the prompt"
        # Prompt version string exists and follows the salt convention.
        assert PROFILE_PROMPT_VERSION == "profile.v4"
    finally:
        hy.close()


def test_redream_unchanged_session_skips_profile_call(cfg, stub_llm):
    """The profile call has its own stamp-based skip-guard
    (sessions.profile_prompt_version): an unchanged session already stamped
    with the current PROFILE_PROMPT_VERSION makes zero profile calls on a
    re-dream."""
    hy = HyMem(_on(cfg), llm=stub_llm)
    try:
        sid = "s_skip"
        mid = hy.log_message(
            sid, "user", "I live in Amsterdam and work as a backend engineer.",
        )
        stub_llm.fixtures[_NEEDLE] = _answer([_item("location", "Amsterdam", mid)])
        hy.dream()
        assert len(_profile_calls(stub_llm)) == 1
        assert _stamp(hy.conn, sid) == PROFILE_PROMPT_VERSION
        hy.dream()  # nothing changed
        assert len(_profile_calls(stub_llm)) == 1
        # And the re-asserted row was not duplicated either way.
        assert len(_rows(hy.conn)) == 1
    finally:
        hy.close()


@pytest.mark.parametrize(
    "corrupt_column",
    ["profile_cursor_prompt_version", "profile_published_generation"],
)
def test_malformed_profile_generation_cannot_claim_steady_state(
    cfg, corrupt_column,
):
    llm = _profile_llm(_answer([]))
    hy = HyMem(_quiet_profile_cfg(cfg), llm=llm)
    try:
        sid = f"malformed-profile-generation-{corrupt_column}"
        hy.log_message(sid, "user", "I live in Utrecht.")
        hy.close_session(sid)
        hy.dream()
        assert len(_profile_calls(llm)) == 1
        row = hy.conn.execute(
            "SELECT profile_cursor_prompt_version, "
            "profile_published_generation FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        valid = row[corrupt_column]
        assert valid is not None
        forged = valid + "-untrusted-suffix"
        hy.conn.execute(
            f"UPDATE sessions SET {corrupt_column} = ? WHERE id = ?",
            (forged, sid),
        )

        hy.dream()
        assert len(_profile_calls(llm)) == 2
        repaired = hy.conn.execute(
            "SELECT profile_cursor_prompt_version, "
            "profile_published_generation FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        config = profile_config_version(
            max_chars=hy.config.dream_digest_max_chars,
            max_items=hy.config.profile_max_items_per_session,
        )
        assert profile_generation_matches_config(
            repaired["profile_cursor_prompt_version"], config
        )
        assert profile_generation_matches_config(
            repaired["profile_published_generation"], config
        )
        assert forged not in tuple(repaired)
    finally:
        hy.close()


def test_non_user_appends_cost_zero_calls_but_short_user_reopens_profile(cfg):
    llm = _profile_llm(_answer([]))
    hy = HyMem(_quiet_profile_cfg(cfg), llm=llm)
    try:
        sid = "profile-role-cursor"
        hy.log_message(sid, "assistant", "I could suggest a city.")
        hy.close_session(sid)
        hy.dream()
        assert _profile_calls(llm) == []

        hy.log_messages(sid, [
            ("system", "internal routing note"),
            ("tool", "tool-only result"),
            ("assistant", "You might live in Utrecht."),
        ])
        hy.close_session(sid)
        hy.dream()
        assert _profile_calls(llm) == []

        short_mid = hy.log_message(sid, "user", "hi")
        hy.close_session(sid)
        hy.dream()
        assert len(_profile_calls(llm)) == 1
        assert f"[msg {short_mid}] hi" in _profile_calls(llm)[0].user

        hy.log_message(sid, "assistant", "hello")
        hy.close_session(sid)
        hy.dream()
        assert len(_profile_calls(llm)) == 1
    finally:
        hy.close()


def test_parse_failure_holds_profile_cursor_and_healing_retry_publishes(cfg):
    llm = _profile_llm("I cannot comply.")
    hy = HyMem(_quiet_profile_cfg(cfg), llm=llm)
    try:
        sid = "profile-heal"
        mid = hy.log_message(sid, "user", "I live in Utrecht.")
        hy.close_session(sid)
        before = dict(hy.conn.execute(
            "SELECT profile_prompt_version, profile_cursor_message_id, "
            "profile_cursor_partial_message_id, profile_cursor_offset, "
            "profile_cursor_prompt_version, profile_published_generation "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone())

        failed = hy.dream()
        after_failure = dict(hy.conn.execute(
            "SELECT profile_prompt_version, profile_cursor_message_id, "
            "profile_cursor_partial_message_id, profile_cursor_offset, "
            "profile_cursor_prompt_version, profile_published_generation "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone())
        assert failed.profile_failures == 1
        assert after_failure == before
        assert _rows(hy.conn) == []

        llm.fixtures[_NEEDLE] = _answer([_item("location", "Utrecht", mid)])
        healed = hy.dream()
        state = hy.conn.execute(
            "SELECT profile_prompt_version, profile_cursor_message_id, "
            "profile_cursor_offset, profile_quarantined "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert healed.profile_failures == 0
        assert state["profile_prompt_version"] == PROFILE_PROMPT_VERSION
        assert state["profile_cursor_message_id"] == mid
        assert state["profile_cursor_offset"] == 0
        assert state["profile_quarantined"] == 0
        assert [(row["slot"], row["value"]) for row in _rows(hy.conn)] == [
            ("location", "Utrecht")
        ]
    finally:
        hy.close()


def test_refusal_containing_named_empty_object_is_not_salvaged(conn):
    sid = "profile-refusal-object"
    _msg(conn, sid, "I live in Utrecht.")
    _cover(conn, sid)
    extraction = extract_user_profile(
        conn,
        sid,
        StubLLMClient(
            fixtures={_NEEDLE: 'I cannot comply. {"items": []}'},
            default="[]",
        ),
        max_chars=4000,
        max_items=10,
    )
    assert extraction is not None
    assert extraction.failed is True
    assert extraction.failure_reason == "parse_failure"


def test_bare_array_and_unrelated_array_envelope_cannot_advance(conn):
    sid = "profile-wrong-empty-shapes"
    _msg(conn, sid, "I live in Utrecht.")
    _cover(conn, sid)
    for raw in ("[]", '{"errors": []}'):
        extraction = extract_user_profile(
            conn,
            sid,
            StubLLMClient(fixtures={_NEEDLE: raw}, default="[]"),
            max_chars=4000,
            max_items=10,
        )
        assert extraction is not None
        assert extraction.failed is True
        assert extraction.failure_reason == "shape_failure"


@pytest.mark.parametrize("item", [
    {
        "slot": "location", "value": "Utrecht", "evidence_message_id": 1,
        "confidence": 0.9, "slot_key": None,
    },
    {
        "slot": "location", "value": "Utrecht", "evidence_message_id": 1,
        "confidence": 0.9, "explanation": "extra",
    },
])
def test_profile_item_exact_wire_shape_failure_holds_cursor(conn, item):
    sid = "profile-exact-item-shape"
    mid = _msg(conn, sid, "I live in Utrecht.")
    _cover(conn, sid)
    item["evidence_message_id"] = mid
    extraction = extract_user_profile(
        conn, sid,
        StubLLMClient(fixtures={_NEEDLE: _answer([item])}, default="[]"),
        max_chars=4000, max_items=10,
    )
    assert extraction is not None and extraction.failed is True
    assert extraction.failure_reason == "validation_failure"
    assert extraction.covered_message_id is None


def test_profile_extraction_enabled_by_default(cfg):
    """profile_extraction_enabled defaults to TRUE: profile.v2 passed the
    on-box precision gate (~95% adjusted, 2026-06-12) after profile.v1 had
    failed it at ~8% and kept the default False in between."""
    assert cfg.profile_extraction_enabled is True


def test_profile_extraction_disabled_flag_makes_no_call(cfg, stub_llm):
    """With profile_extraction_enabled explicitly False, a dream must make
    zero profile LLM calls, persist nothing, and leave the stamp unset."""
    cfg = replace(cfg, profile_extraction_enabled=False)
    hy = HyMem(cfg, llm=stub_llm)
    try:
        sid = "s_off"
        mid = hy.log_message(
            sid, "user", "I work as a bedrijfsarts at MedFlow nowadays.",
        )
        stub_llm.fixtures[_NEEDLE] = _answer([_item("role", "bedrijfsarts", mid)])
        hy.dream()
        assert _profile_calls(stub_llm) == []
        assert _rows(hy.conn) == []
        assert _stamp(hy.conn, sid) is None
    finally:
        hy.close()


def test_profile_stamp_set_after_successful_extraction(cfg, stub_llm):
    """A successful extraction stamps sessions.profile_prompt_version —
    including a valid extraction with ZERO items (a legitimate 'nothing
    here'), which must also suppress the call on the next dream."""
    hy = HyMem(_on(cfg), llm=stub_llm)
    try:
        sid = "s_stamp"
        hy.log_message(
            sid, "user", "Nothing personal here, just postgres tuning notes.",
        )
        stub_llm.fixtures[_NEEDLE] = _answer([])  # valid extraction, zero items
        hy.dream()
        assert len(_profile_calls(stub_llm)) == 1
        assert _rows(hy.conn) == []
        assert _stamp(hy.conn, sid) == PROFILE_PROMPT_VERSION
        hy.dream()  # stamped + no new chunk work → no second call
        assert len(_profile_calls(stub_llm)) == 1
    finally:
        hy.close()


def test_old_profile_stamp_re_extracts_despite_current_digest(cfg, stub_llm):
    """THE DECOUPLING TEST: the profile guard keys on PROFILE_PROMPT_VERSION,
    not on the digest's prompt_version. With digested_prompt_version current
    but an old profile stamp (as after a profile-prompt bump), a re-dream
    must run profile extraction again."""
    hy = HyMem(_on(cfg), llm=stub_llm)
    try:
        sid = "s_bump"
        mid = hy.log_message(
            sid, "user", "Ik werk tegenwoordig als bedrijfsarts.",
        )
        stub_llm.fixtures[_NEEDLE] = _answer([_item("role", "bedrijfsarts", mid)])
        stub_llm.fixtures["Return the JSON object now"] = json.dumps({
            "episodes": [], "summary": "", "procedures": [],
        })
        hy.dream()
        assert len(_profile_calls(stub_llm)) == 1

        # Simulate a session last extracted under the failed v1 prompt.
        hy.conn.execute(
            "UPDATE sessions SET profile_prompt_version = 'profile.v1' "
            "WHERE id = ?",
            (sid,),
        )
        # Precondition: the DIGEST guard alone would skip this session.
        digested = hy.conn.execute(
            "SELECT digested_prompt_version FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert digested["digested_prompt_version"] == hy.config.prompt_version

        hy.dream()
        assert len(_profile_calls(stub_llm)) == 2, (
            "an outdated profile stamp must re-run extraction even though "
            "the session digest is current"
        )
        assert _stamp(hy.conn, sid) == PROFILE_PROMPT_VERSION  # re-stamped
        assert len(_rows(hy.conn)) == 1  # re-assertion did not duplicate
    finally:
        hy.close()


# ── strict validation: the LLM can never invent a slot ──────────────────────


def test_validate_rejects_unknown_slot_bad_evidence_and_bad_confidence():
    valid_ids = {10, 11}
    items = validate_profile_items(
        [
            _item("role", "bedrijfsarts", 10),                      # kept
            _item("favorite_color", "blue", 10),                    # unknown slot
            _item("employer", "MedFlow", 999),                      # hallucinated id
            {"slot": "name", "value": "Atta", "confidence": 1.0},   # missing evidence
            _item("location", "Amsterdam", 11, conf=1.7),           # conf > 1
            _item("location", "Amsterdam", 11, conf=-0.1),          # conf < 0
            {"slot": "name", "value": "Atta", "evidence_message_id": 11},  # no conf
            _item("name", "   ", 11),                               # empty value
            _item("relationship", "sister", 11),                    # keyed slot, no key
            "not a dict",
        ],
        valid_ids, max_items=16,
    )
    assert items == [{
        "slot": "role", "slot_key": None, "value": "bedrijfsarts",
        "evidence_message_id": 10, "confidence": 0.9,
    }]


def test_validate_caps_items_and_rejects_stray_keys():
    valid_ids = {1}
    raw = [_item("language", f"lang{i}", 1) for i in range(10)]
    assert len(validate_profile_items(raw, valid_ids, max_items=3)) == 3
    # The cursor-authoritative contract is exact: a slot_key on a
    # non-parameterized slot is malformed rather than silently normalized.
    assert validate_profile_items(
        [_item("employer", "MedFlow", 1, key="acme")], valid_ids, max_items=16
    ) == []
    # relationship keys are casefolded so "Anna"/"anna" collide.
    [rel] = validate_profile_items(
        [_item("relationship", "sister", 1, key="  Anna ")], valid_ids, max_items=16
    )
    assert rel["slot_key"] == "anna"


def test_db_check_constraint_rejects_unknown_slot(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO user_profile(slot, value) VALUES ('favorite_color', 'blue')"
        )


def test_unknown_slot_from_llm_never_persists(cfg, stub_llm):
    hy = HyMem(_on(cfg), llm=stub_llm)
    try:
        mid = hy.log_message(
            "s_bad", "user", "My favorite color is blue and I am a teacher.",
        )
        stub_llm.fixtures[_NEEDLE] = _answer([
            _item("favorite_color", "blue", mid),   # invented slot → dropped
            _item("role", "teacher", mid),
        ])
        first = hy.dream()
        assert first.profile_failures == 1
        assert _rows(hy.conn) == [], "mixed valid/invalid output is not partial success"

        stub_llm.fixtures[_NEEDLE] = _answer([_item("role", "teacher", mid)])
        second = hy.dream()
        assert second.profile_failures == 0
        assert [(r["slot"], r["value"]) for r in _rows(hy.conn)] == [
            ("role", "teacher")
        ]
    finally:
        hy.close()


def test_output_cap_adapts_then_quarantines_and_policy_change_reopens(cfg):
    config = _quiet_profile_cfg(
        cfg,
        dream_digest_max_chars=600,
        profile_max_items_per_session=1,
        profile_extraction_max_attempts=2,
        salience_min_chars=10_000,
    )
    llm = _profile_llm(_answer([]))
    hy = HyMem(config, llm=llm)
    try:
        sid = "profile-cap-quarantine"
        mid = hy.log_message(sid, "user", "🙂e\u0301界" * 600)
        hy.close_session(sid)
        llm.fixtures[_NEEDLE] = _answer([
            _item("location", "Utrecht", mid),
            _item("role", "doctor", mid),
        ])

        first = hy.dream()
        second = hy.dream()
        assert first.profile_failures == second.profile_failures == 1
        calls = _profile_calls(llm)
        assert len(calls) == 2
        assert len(calls[1].user) < len(calls[0].user), (
            "the held cursor must retry a smaller exact input slice"
        )
        state = hy.conn.execute(
            "SELECT profile_retry_count, profile_quarantined, "
            "profile_cursor_message_id, profile_cursor_offset, "
            "profile_prompt_version FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert state["profile_retry_count"] == 2
        assert state["profile_quarantined"] == 1
        assert state["profile_cursor_message_id"] is None
        assert state["profile_cursor_offset"] == 0
        assert state["profile_prompt_version"] is None
        assert _rows(hy.conn) == []

        hy.dream()
        assert len(_profile_calls(llm)) == 2
    finally:
        hy.close()

    # Changing only retry policy must unquarantine the exact same source
    # position; it must not require a prompt/config rewind.
    retry_llm = _profile_llm(_answer([_item("location", "Utrecht", mid)]))
    reopened = HyMem(
        replace(config, profile_extraction_max_attempts=0),
        llm=retry_llm,
    )
    try:
        report = reopened.dream()
        state = reopened.conn.execute(
            "SELECT profile_retry_count, profile_quarantined, "
            "profile_cursor_offset FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert report.profile_failures == 0
        assert len(_profile_calls(retry_llm)) == 1
        assert state["profile_retry_count"] == 0
        assert state["profile_quarantined"] == 0
        assert state["profile_cursor_offset"] > 0
    finally:
        reopened.close()


def test_profile_derived_write_failure_rolls_back_stage_and_cursor(
    cfg, monkeypatch,
):
    llm = _profile_llm(_answer([]))
    hy = HyMem(_quiet_profile_cfg(cfg), llm=llm)
    original_stage = dreaming_runner.stage_profile_extraction
    try:
        sid = "profile-stage-rollback"
        mid = hy.log_message(sid, "user", "I live in Utrecht.")
        hy.close_session(sid)
        llm.fixtures[_NEEDLE] = _answer([_item("location", "Utrecht", mid)])

        def stage_then_raise(*args, **kwargs):
            original_stage(*args, **kwargs)
            raise RuntimeError("injected derived write failure")

        monkeypatch.setattr(
            dreaming_runner, "stage_profile_extraction", stage_then_raise
        )
        report = hy.dream()
        state = hy.conn.execute(
            "SELECT profile_prompt_version, profile_cursor_message_id, "
            "profile_cursor_prompt_version, profile_published_generation "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert report.profile_failures == 1
        assert all(value is None for value in tuple(state))
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM profile_staging WHERE session_id = ?", (sid,)
        ).fetchone()[0] == 0
        assert _rows(hy.conn) == []
    finally:
        hy.close()


# ── bi-temporal supersession (P2 semantics) ──────────────────────────────────


def test_single_valued_slot_supersedes_on_conflict(conn):
    m1 = _msg(conn, "s", "I work at Acme.", created_at="2026-01-01 10:00:00")
    m2 = _msg(conn, "s", "I now work at Globex.", created_at="2026-03-01 10:00:00")
    _persist(conn, [_item("employer", "Acme", m1)])
    _persist(conn, [_item("employer", "Globex", m2)])

    old, new = _rows(conn)
    assert old["value"] == "Acme"
    assert old["valid_at"] == "2026-01-01 10:00:00"
    # invalid_at = the superseding evidence's world date (stamp_invalidation
    # mirror: newest contradicting evidence, fallback flip time).
    assert old["invalid_at"] == "2026-03-01 10:00:00"
    assert new["value"] == "Globex"
    assert new["valid_at"] == "2026-03-01 10:00:00"
    assert new["invalid_at"] is None

    # Only the new value is the ACTIVE profile.
    assert [e.value for e in load_profile(conn)] == ["Globex"]


def test_multi_valued_slots_accumulate(conn):
    mid = _msg(conn, "s", "Ik spreek Nederlands en Engels en ik hardloop elke zondag.")
    _persist(conn, [
        _item("language", "Dutch", mid),
        _item("language", "English", mid),
        _item("recurring_activity", "runs every sunday", mid),
    ])
    active = load_profile(conn)
    assert {(e.slot, e.value) for e in active} == {
        ("language", "Dutch"), ("language", "English"),
        ("recurring_activity", "runs every sunday"),
    }
    assert all(e.valid_at for e in active)


def test_relationship_supersedes_per_person_only(conn):
    m1 = _msg(conn, "s", "Anna is my girlfriend.", created_at="2026-01-01 09:00:00")
    m2 = _msg(conn, "s", "Anna is now my wife!", created_at="2026-05-01 09:00:00")
    _persist(conn, [
        _item("relationship", "girlfriend", m1, key="anna"),
        _item("relationship", "friend", m1, key="bob"),
    ])
    _persist(conn, [_item("relationship", "wife", m2, key="anna")])

    active = load_profile(conn)
    assert {(e.slot_key, e.value) for e in active} == {("anna", "wife"), ("bob", "friend")}
    superseded = conn.execute(
        "SELECT invalid_at FROM user_profile WHERE value = 'girlfriend'"
    ).fetchone()
    assert superseded["invalid_at"] == "2026-05-01 09:00:00"


def test_reassertion_reinforces_without_duplicating(conn):
    mid = _msg(conn, "s", "I work at MedFlow.")
    _persist(conn, [_item("employer", "MedFlow", mid, conf=0.5)])
    _persist(conn, [_item("employer", "medflow", mid, conf=0.8)])  # case-insensitive
    _persist(conn, [_item("employer", "MedFlow", mid, conf=0.3)])  # never lowers

    rows = _rows(conn)
    assert len(rows) == 1
    assert rows[0]["invalid_at"] is None
    assert rows[0]["confidence"] == pytest.approx(0.8)


def test_newer_same_value_support_prevents_older_conflict_replay(conn):
    old_a = _msg(conn, "s-a-old", "I work at Acme.", created_at="2024-01-01T00:00:00Z")
    newer_a = _msg(conn, "s-a-new", "I still work at Acme.", created_at="2026-01-01T00:00:00Z")
    middle_b = _msg(conn, "s-b", "I work at Globex.", created_at="2025-01-01T00:00:00Z")

    _persist(conn, [_item("employer", "Acme", old_a)])
    _persist(conn, [_item("employer", "Acme", newer_a)])
    _persist(conn, [_item("employer", "Globex", middle_b)])

    [active] = load_profile(conn)
    assert active.value == "Acme"
    assert active.evidence_message_id == newer_a
    globex = conn.execute(
        "SELECT valid_at, invalid_at FROM user_profile WHERE value = 'Globex'"
    ).fetchone()
    assert globex["invalid_at"] is not None
    assert datetime.fromisoformat(globex["invalid_at"].replace("Z", "+00:00")) >= (
        datetime.fromisoformat(globex["valid_at"].replace("Z", "+00:00"))
    )


@pytest.mark.parametrize(
    "insertion_order",
    [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ],
)
def test_singleton_a_b_a_history_converges_for_every_insertion_order(
    conn, insertion_order,
):
    """Every source assertion survives, and intervals depend only on source time."""
    assertions = [
        (
            _msg(
                conn,
                "aba-history",
                "I work at Acme.",
                created_at="2026-01-01T00:00:00Z",
            ),
            "Acme",
        ),
        (
            _msg(
                conn,
                "aba-history",
                "I work at Globex now.",
                created_at="2026-02-01T00:00:00Z",
            ),
            "Globex",
        ),
        (
            _msg(
                conn,
                "aba-history",
                "I am back at Acme.",
                created_at="2026-03-01T00:00:00Z",
            ),
            "Acme",
        ),
    ]

    for index in insertion_order:
        mid, value = assertions[index]
        _persist(conn, [_item("employer", value, mid)])

    rows = conn.execute(
        "SELECT value, source_message_id, source_created_at, valid_at, invalid_at "
        "FROM user_profile WHERE slot = 'employer' ORDER BY source_created_at"
    ).fetchall()
    assert len(rows) == 3, "A@t1 and A@t3 are distinct source assertions"
    assert [
        (row["value"], row["source_message_id"], row["source_created_at"])
        for row in rows
    ] == [
        ("Acme", assertions[0][0], "2026-01-01T00:00:00Z"),
        ("Globex", assertions[1][0], "2026-02-01T00:00:00Z"),
        ("Acme", assertions[2][0], "2026-03-01T00:00:00Z"),
    ]
    assert [row["valid_at"] for row in rows] == [
        "2026-01-01T00:00:00.000000+00:00",
        "2026-02-01T00:00:00.000000+00:00",
        "2026-03-01T00:00:00.000000+00:00",
    ]
    assert [row["invalid_at"] for row in rows] == [
        "2026-02-01T00:00:00.000000+00:00",
        "2026-03-01T00:00:00.000000+00:00",
        None,
    ]
    assert [
        (entry.value, entry.evidence_message_id)
        for entry in load_profile(conn)
        if entry.slot == "employer"
    ] == [("Acme", assertions[2][0])]


def test_profile_chronology_normalizes_offsets_and_ties_by_session(conn):
    # Lexically 2026-01-02 looks newer, but +10:00 makes it 14:00Z; the second
    # assertion at 15:00Z must win.
    offset_old = _msg(
        conn,
        "offset-a",
        "I live in Utrecht.",
        created_at="2026-01-02T00:00:00+10:00",
    )
    utc_new = _msg(
        conn,
        "offset-b",
        "I live in Amsterdam.",
        created_at="2026-01-01T15:00:00Z",
    )
    _persist(conn, [_item("location", "Utrecht", offset_old)])
    _persist(conn, [_item("location", "Amsterdam", utc_new)])
    assert [entry.value for entry in load_profile(conn)] == ["Amsterdam"]

    # Equal instants use durable session id then message id, independent of
    # whichever session the runner happens to enumerate first.
    tie_z = _msg(conn, "z-session", "My role is Z.", created_at="2026-02-01T00:00:00Z")
    tie_a = _msg(conn, "a-session", "My role is A.", created_at="2026-02-01T00:00:00Z")
    _persist(conn, [_item("role", "Z", tie_z)])
    _persist(conn, [_item("role", "A", tie_a)])
    roles = [entry for entry in load_profile(conn) if entry.slot == "role"]
    assert [(entry.value, entry.evidence_message_id) for entry in roles] == [
        ("Z", tie_z)
    ]


@pytest.mark.parametrize("unknown_timestamp", [None, "not-a-date"])
def test_unknown_source_time_never_supersedes_known_active_fact(
    conn, unknown_timestamp,
):
    known = _msg(
        conn,
        "known-time",
        "I live in Amsterdam.",
        created_at="2026-01-01T00:00:00Z",
    )
    unknown = _msg(conn, "unknown-time", "I live in Paris.", created_at=None)
    # Force a genuine legacy NULL rather than the column default.
    conn.execute(
        "UPDATE messages SET created_at = ? WHERE id = ?", (unknown_timestamp, unknown)
    )
    _persist(conn, [_item("location", "Amsterdam", known)])
    _persist(conn, [_item("location", "Paris", unknown)])
    assert [entry.value for entry in load_profile(conn) if entry.slot == "location"] == [
        "Amsterdam"
    ]
    historical = conn.execute(
        "SELECT source_created_at, valid_at, invalid_at FROM user_profile "
        "WHERE value = 'Paris'"
    ).fetchone()
    assert historical["source_created_at"] == unknown_timestamp
    assert historical["invalid_at"] is not None
    assert datetime.fromisoformat(historical["invalid_at"].replace("Z", "+00:00")) >= (
        datetime.fromisoformat(historical["valid_at"].replace("Z", "+00:00"))
    )


def test_profile_persistence_rejects_assistant_or_uncovered_provenance(conn):
    assistant_mid = _msg(conn, "assistant-source", "You live in Paris.", role="assistant")
    _cover(conn, "assistant-source")
    with pytest.raises(ValueError, match="unavailable"):
        _persist(conn, [_item("location", "Paris", assistant_mid)])

    user_mid = _msg(conn, "uncovered-source", "I live in Utrecht.")
    with pytest.raises(ValueError, match="unavailable"):
        persist_user_profile(
            conn,
            ProfileExtraction(items=[_item("location", "Utrecht", user_mid)]),
        )


# ── consumer 1: the VERIFIED FACTS anchor ────────────────────────────────────


def test_anchor_profile_rows_precede_graph_edges_with_combined_cap(conn):
    with core_db.transaction(conn):
        seed_edge(conn, "atta", "part_of", "medflow", pos=9)
        seed_edge(conn, "medflow", "uses", "postgres", pos=5)
    mid = _msg(conn, "s", "Ik ben Atta, bedrijfsarts.")
    _persist(conn, [
        _item("role", "bedrijfsarts", mid),
        _item("name", "Atta", mid),
    ])

    facts = _anchor_facts(conn, 4)
    # Profile first (identity-first slot order: name before role), then edges.
    assert facts[:2] == ["user name Atta", "user role bedrijfsarts"]
    assert facts[2:] == ["atta part_of medflow", "medflow uses postgres"]

    # The cap bounds the COMBINED list; profile rows win the contested slots.
    assert _anchor_facts(conn, 3) == [
        "user name Atta", "user role bedrijfsarts", "atta part_of medflow",
    ]
    assert _anchor_facts(conn, 1) == ["user name Atta"]
    assert _anchor_facts(conn, 0) == []


def test_profile_change_regenerates_digest(cfg, conn):
    """Profile rows join the anchor block that is hashed into the root's cache
    id, so a profile change must force a fresh root fusion (same mechanism as
    a graph change — a digest pinned to a stale identity is the P4 failure)."""
    acfg = replace(cfg, aggregation_nodes_enabled=True, aggregation_digest_enabled=True)
    with core_db.transaction(conn):
        conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s1')")
        conn.execute(
            """INSERT INTO episodes(id, session_id, title, summary, participants,
                                    start_message_id, end_message_id, outcome, key_entities)
               VALUES ('e1', 's1', 'Weekend cycling', 'Started cycling.', '[]',
                       1, 2, NULL, '["cycling"]')""",
        )
    first = StubLLMClient(
        fixtures={"standing digest of everything known": json.dumps(
            {"title": "First", "summary": "No identity yet."})},
        default="[]",
    )
    build_aggregation_nodes(conn, acfg, first, None)
    assert load_digest(conn).title == "First"

    mid = _msg(conn, "s1", "Ik ben bedrijfsarts.")
    _persist(conn, [_item("role", "bedrijfsarts", mid)])
    fresh = StubLLMClient(
        fixtures={"standing digest of everything known": json.dumps(
            {"title": "Regrounded", "summary": "Knows the role now."})},
        default="[]",
    )
    build_aggregation_nodes(conn, acfg, fresh, None)
    digest_calls = [c for c in fresh.calls
                    if "standing digest of everything known" in c.system]
    assert len(digest_calls) == 1, "profile change must evict the cached root"
    assert "user role bedrijfsarts" in digest_calls[0].user
    assert load_digest(conn).title == "Regrounded"


# ── consumer 2: the additive augment tier ────────────────────────────────────


def test_augment_user_profile_tier_is_additive(cfg, stub_llm):
    hy = HyMem(_on(cfg), llm=stub_llm)
    try:
        sid = "s_aug"
        mid = hy.log_message(
            sid, "user", "The billing service runs on postgres with pgbouncer.",
        )

        before = hy.augment("what database does billing use?", session_id=sid)
        assert before.user_profile == []

        _persist(hy.conn, [
            _item("role", "bedrijfsarts", mid),
            _item("relationship", "sister", mid, key="anna"),
        ])
        after = hy.augment("what database does billing use?", session_id=sid)

        # Tier present and typed.
        assert [(e.slot, e.slot_key, e.value) for e in after.user_profile] == [
            ("role", None, "bedrijfsarts"), ("relationship", "anna", "sister"),
        ]
        # NO slot competition: every pre-existing tier is byte-identical.
        assert after.fts_hits == before.fts_hits
        assert after.message_hits == before.message_hits
        assert after.episodes == before.episodes
        assert after.procedures == before.procedures
        assert after.graph_facts == before.graph_facts
        assert after.recent_turns == before.recent_turns
        assert after.aggregation_nodes == before.aggregation_nodes
    finally:
        hy.close()


def test_augment_tier_respects_flag_and_cap(cfg, stub_llm):
    hy = HyMem(replace(cfg, profile_extraction_enabled=True, profile_context_cap=1),
               llm=stub_llm)
    try:
        mid = hy.log_message("s_cap", "user", "Ik ben Atta en ik werk als arts.")
        _persist(hy.conn, [
            _item("role", "arts", mid),
            _item("name", "Atta", mid),
        ])
        ctx = hy.augment("hello")
        assert [(e.slot, e.value) for e in ctx.user_profile] == [("name", "Atta")]
    finally:
        hy.close()

    hy_off = HyMem(replace(cfg, profile_extraction_enabled=False), llm=stub_llm)
    try:
        assert hy_off.augment("hello").user_profile == []
    finally:
        hy_off.close()


# ── consumer 3: HyMem.profile() ──────────────────────────────────────────────


def test_profile_empty_without_dream(hy):
    assert hy.profile() == []


def test_profile_returns_active_typed_entries(cfg, stub_llm):
    hy = HyMem(cfg, llm=stub_llm)
    try:
        mid = hy.log_message("s_p", "user", "I moved from Utrecht to Amsterdam.")
        _persist(hy.conn, [_item("location", "Utrecht", mid)])
        _persist(hy.conn, [_item("location", "Amsterdam", mid)])
        entries = hy.profile()
        assert [(e.slot, e.value, e.evidence_message_id) for e in entries] == [
            ("location", "Amsterdam", mid),
        ]
        assert entries[0].confidence == pytest.approx(0.9)
        assert entries[0].valid_at
    finally:
        hy.close()


# ── redaction in every consumer ──────────────────────────────────────────────


def test_health_condition_value_is_redacted_in_all_consumers(cfg, stub_llm):
    hy = HyMem(_on(cfg), llm=stub_llm)
    try:
        mid = hy.log_message("s_red", "user", "I track my asthma in an app.")
        _persist(hy.conn, [_item(
            "health_condition",
            "asthma, plan shared via atta@example.com",
            mid,
        )])

        expected = "asthma, plan shared via [REDACTED-EMAIL]"
        # persisted value is scrubbed at the chokepoint…
        assert _rows(hy.conn)[0]["value"] == expected
        # …so every consumer inherits it.
        assert hy.profile()[0].value == expected
        assert hy.augment("how is my health?").user_profile[0].value == expected
        anchor = _anchor_facts(hy.conn, 5)
        assert anchor == [f"user health_condition {expected}"]
        assert all("atta@example.com" not in line for line in anchor)
    finally:
        hy.close()


def test_redaction_follows_config_flag(conn):
    mid = _msg(conn, "s", "secret holder")
    _persist(conn, [_item("possession", "key sk-ABCD1234efgh5678ijkl", mid)],
             redact=False)
    assert "sk-ABCD1234efgh5678ijkl" in _rows(conn)[0]["value"]


def test_enabling_redaction_on_reopen_scrubs_all_profile_rows_and_drops_unsafe_stage(
    tmp_path,
):
    raw = HyMem(HyMemConfig(root=tmp_path, redact_secrets=False))
    try:
        first = raw.log_message(
            "privacy-transition",
            "user",
            "Alice is my friend; contact first.secret@example.com.",
            created_at="2026-01-01T00:00:00Z",
        )
        second = raw.log_message(
            "privacy-transition",
            "user",
            "Alice is now a colleague; token sk-ABCD1234efgh5678ijkl.",
            created_at="2026-02-01T00:00:00Z",
        )
        third = raw.log_message(
            "privacy-transition",
            "user",
            "Bob is my friend; contact second.secret@example.net.",
            created_at="2026-03-01T00:00:00Z",
        )
        with core_db.transaction(raw.conn):
            persist_user_profile(
                raw.conn,
                ProfileExtraction(items=[
                    _item(
                        "relationship",
                        "friend first.secret@example.com",
                        first,
                        key="alice.private@example.com",
                    ),
                    _item(
                        "relationship",
                        "colleague sk-ABCD1234efgh5678ijkl",
                        second,
                        key="alice.private@example.com",
                    ),
                    _item(
                        "relationship",
                        "friend second.secret@example.net",
                        third,
                        key="bob.private@example.com",
                    ),
                ]),
                redact_values=False,
            )
            legacy_generation = (
                profile_config_version(
                    max_chars=12000,
                    max_items=16,
                    redact_values=False,
                )
                + "|walk="
                + ("a" * 32)
            )
            raw.conn.executemany(
                "INSERT INTO profile_staging("
                "session_id, generation, slice_key, items_json"
                ") VALUES ('privacy-transition', ?, ?, ?)",
                [
                    (
                        legacy_generation,
                        "legacy-unredacted",
                        json.dumps([{"value": "stage.secret@example.com"}]),
                    ),
                    (
                        "malformed-legacy-generation",
                        "malformed",
                        json.dumps([{"value": "sk-ZYXW9876vuts5432rqpo"}]),
                    ),
                ],
            )

        before = raw.conn.execute(
            "SELECT value, slot_key, invalid_at FROM user_profile ORDER BY id"
        ).fetchall()
        assert any(row["invalid_at"] is not None for row in before)
        assert any("example" in row["value"] for row in before)
        assert raw.conn.execute(
            "SELECT COUNT(*) FROM profile_staging"
        ).fetchone()[0] == 2
    finally:
        raw.close()

    strict = HyMem(HyMemConfig(root=tmp_path, redact_secrets=True))
    try:
        rows = strict.conn.execute(
            "SELECT value, slot_key, invalid_at FROM user_profile ORDER BY id"
        ).fetchall()
        serialized = json.dumps([dict(row) for row in rows])
        for secret in (
            "first.secret@example.com",
            "second.secret@example.net",
            "sk-ABCD1234efgh5678ijkl",
            "alice.private@example.com",
            "bob.private@example.com",
        ):
            assert secret not in serialized
        assert "[REDACTED-EMAIL]" in serialized
        assert "[REDACTED-API-KEY]" in serialized
        assert any(row["invalid_at"] is not None for row in rows)
        assert any(row["invalid_at"] is None for row in rows)

        # Slot keys are identity as well as content: scrubbing must not merge
        # two different email-address people into one relationship key.
        keys = {row["slot_key"] for row in rows}
        assert len(keys) == 2
        assert all(key.startswith("[redacted-email]#") for key in keys)
        assert strict.conn.execute(
            "SELECT COUNT(*) FROM profile_staging"
        ).fetchone()[0] == 0
    finally:
        strict.close()


def test_unicode_oversized_user_turn_is_sliced_without_dropping_tail(cfg):
    llm = _CapturingProfileLLM()
    config = _quiet_profile_cfg(
        cfg,
        dream_digest_max_chars=256,
        salience_min_chars=10_000,
    )
    hy = HyMem(config, llm=llm)
    try:
        sid = "profile-unicode"
        content = "HEAD🙂e\u0301界\u2028" + ("ab🙂界e\u0301" * 120) + "TAIL終"
        mid = hy.log_message(sid, "user", content)
        hy.close_session(sid)

        first_report = hy.dream()
        first_state = hy.conn.execute(
            "SELECT profile_cursor_message_id, "
            "profile_cursor_partial_message_id, profile_cursor_offset "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert first_state["profile_cursor_offset"] > 0
        assert first_report.budget_exhausted is True
        # Raw retention between slices must not strand the persisted offset or
        # lose the remainder: every subsequent byte comes from v38 coverage.
        hy.conn.execute("DELETE FROM messages WHERE id = ?", (mid,))

        saw_unfinished = True
        for _ in range(29):
            report = hy.dream()
            state = hy.conn.execute(
                "SELECT profile_cursor_message_id, "
                "profile_cursor_partial_message_id, profile_cursor_offset "
                "FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            if state["profile_cursor_offset"] > 0:
                saw_unfinished = saw_unfinished or report.budget_exhausted
            if state["profile_cursor_message_id"] == mid and state[
                "profile_cursor_offset"
            ] == 0:
                break
        else:
            pytest.fail("profile cursor did not reach the oversized USER tail")

        assert "".join(llm.fragments) == content
        assert llm.fragments[0].startswith("HEAD")
        assert llm.fragments[-1].endswith("TAIL終")
        assert saw_unfinished is True
        before = len(llm.fragments)
        hy.dream()
        assert len(llm.fragments) == before, "caught-up re-dream must cost zero calls"
    finally:
        hy.close()


def test_partial_profile_stage_is_redacted_and_not_published(cfg):
    raw_email = "atta@example.com"
    raw_key = "sk-ABCD1234efgh5678ijkl"
    llm = _CapturingProfileLLM()
    config = _quiet_profile_cfg(
        cfg,
        dream_digest_max_chars=256,
        salience_min_chars=10_000,
    )
    hy = HyMem(config, llm=llm)
    try:
        sid = "profile-redacted-stage"
        old_mid = hy.log_message(sid, "user", "I live in Utrecht.")
        _persist(hy.conn, [_item("location", "Utrecht", old_mid)])
        new_mid = hy.log_message(sid, "user", "new profile evidence " + "x" * 800)
        hy.close_session(sid)
        llm.item_factory = lambda _n: [_item(
            "location",
            f"Amsterdam contact {raw_email} token {raw_key}",
            new_mid,
        )]

        report = hy.dream()
        staged = hy.conn.execute(
            "SELECT items_json FROM profile_staging WHERE session_id = ?",
            (sid,),
        ).fetchone()["items_json"]
        assert report.budget_exhausted is True
        assert raw_email not in staged and raw_key not in staged
        assert "[REDACTED-EMAIL]" in staged and "[REDACTED-API-KEY]" in staged
        assert [(entry.slot, entry.value) for entry in hy.profile()] == [
            ("location", "Utrecht")
        ], "partial rebuild output must remain invisible"
    finally:
        hy.close()


def test_prompt_rewind_after_raw_prune_preserves_old_profile_on_failure(cfg):
    config = _quiet_profile_cfg(cfg, salience_min_chars=10_000)
    llm = _profile_llm(_answer([]))
    hy = HyMem(config, llm=llm)
    try:
        sid = "profile-pruned-rewind"
        mid = hy.log_message(sid, "user", "I live in Utrecht.")
        hy.close_session(sid)
        llm.fixtures[_NEEDLE] = _answer([_item("location", "Utrecht", mid)])
        hy.dream()
        original = hy.profile()[0]
        assert original.evidence_message_id == mid

        hy.conn.execute("DELETE FROM messages WHERE id = ?", (mid,))
        assert hy.conn.execute(
            "SELECT evidence_message_id, source_message_id, source_created_at "
            "FROM user_profile"
        ).fetchone()["evidence_message_id"] is None
        retained = hy.profile()[0]
        assert retained.evidence_message_id == mid
        assert retained.valid_at == original.valid_at

        hy.conn.execute(
            "UPDATE sessions SET profile_prompt_version = 'profile.v2' WHERE id = ?",
            (sid,),
        )
        cursor_before = tuple(hy.conn.execute(
            "SELECT profile_cursor_message_id, "
            "profile_cursor_partial_message_id, profile_cursor_offset, "
            "profile_cursor_prompt_version, profile_published_generation "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone())
        llm.fixtures[_NEEDLE] = "refusal after pruning"
        failed = hy.dream()
        assert failed.profile_failures == 1
        assert [(entry.value, entry.evidence_message_id) for entry in hy.profile()] == [
            ("Utrecht", mid)
        ]
        cursor_after = tuple(hy.conn.execute(
            "SELECT profile_cursor_message_id, "
            "profile_cursor_partial_message_id, profile_cursor_offset, "
            "profile_cursor_prompt_version, profile_published_generation "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone())
        assert cursor_after == cursor_before

        llm.fixtures[_NEEDLE] = _answer([_item("location", "Amsterdam", mid)])
        healed = hy.dream()
        assert healed.profile_failures == 0
        assert [(entry.value, entry.evidence_message_id) for entry in hy.profile()] == [
            ("Amsterdam", mid)
        ]
        source = hy.conn.execute(
            "SELECT source_created_at FROM user_profile "
            "WHERE invalid_at IS NULL"
        ).fetchone()["source_created_at"]
        assert source is not None
    finally:
        hy.close()


# ── prompt rendering (shared with the front-run gate script) ─────────────────


def test_build_profile_user_prompt_tags_and_truncates():
    turns = [(7, "I am a nurse."), (9, "x" * 50)]
    prompt = build_profile_user_prompt(turns, max_chars=10_000)
    assert "[msg 7] I am a nurse." in prompt
    assert "[msg 9]" in prompt
    assert prompt.rstrip().endswith("Return the profile JSON object now.")
    # The body honors the char cap the dream phase applies.
    capped = build_profile_user_prompt(turns, max_chars=20)
    assert "x" * 50 not in capped


def test_profile_publication_orders_numeric_offsets_within_one_message(conn):
    sid = "profile-stage-order"
    mid = _msg(conn, sid, "x" * 180)
    _cover(conn, sid)
    max_chars = 30
    generation = (
        profile_config_version(max_chars=max_chars, max_items=16)
        + "|walk="
        + ("0" * 32)
    )
    cursor = None
    partial = None
    offset = 0
    starts: list[int] = []
    final_value = None
    call_no = 0
    while True:
        call_no += 1
        value = "Place-stable"
        final_value = value
        extraction = extract_user_profile(
            conn,
            sid,
            StubLLMClient(
                fixtures={_NEEDLE: _answer([_item("location", value, mid)])},
                default="[]",
            ),
            max_chars=max_chars,
            max_items=16,
            since_message_id=cursor,
            partial_message_id=partial,
            since_message_offset=offset,
        )
        assert extraction is not None and not extraction.failed
        starts.append(extraction.start_message_offset)
        with core_db.transaction(conn):
            stage_profile_extraction(conn, sid, generation, extraction)
            conn.execute(
                "UPDATE sessions SET profile_cursor_message_id = ?, "
                "profile_cursor_partial_message_id = ?, profile_cursor_offset = ?, "
                "profile_cursor_prompt_version = ? WHERE id = ?",
                (
                    extraction.covered_message_id,
                    extraction.partial_message_id,
                    extraction.next_message_offset,
                    generation,
                    sid,
                ),
            )
            if extraction.caught_up:
                publish_profile_generation(conn, sid, generation)
        cursor = extraction.covered_message_id
        partial = extraction.partial_message_id
        offset = extraction.next_message_offset
        if extraction.caught_up:
            break

    assert any(0 < start < 100 for start in starts)
    assert any(start >= 100 for start in starts)
    assert len(starts) >= 3
    assert [(entry.slot, entry.value) for entry in load_profile(conn)] == [
        ("location", final_value)
    ]


def test_profile_publication_rejects_missing_stage_wrong_generation_and_cursor(conn):
    sid = "profile-publish-guards"
    mid = _msg(conn, sid, "I live in Utrecht.")
    _cover(conn, sid)
    generation = (
        profile_config_version(max_chars=100, max_items=16)
        + "|walk="
        + ("0" * 32)
    )
    conn.execute(
        "UPDATE sessions SET profile_cursor_message_id = ?, "
        "profile_cursor_prompt_version = ? WHERE id = ?",
        (mid, generation, sid),
    )
    with pytest.raises(RuntimeError, match="no staged slices"):
        publish_profile_generation(conn, sid, generation)
    with pytest.raises(RuntimeError, match="active cursor"):
        publish_profile_generation(conn, sid, generation + "-wrong")

    extraction = ProfileExtraction(
        items=[],
        covered_message_id=mid,
        start_message_id=mid,
        end_message_id=mid,
        caught_up=True,
        slice_key="valid-empty",
    )
    stage_profile_extraction(conn, sid, generation, extraction)
    conn.execute(
        "UPDATE sessions SET profile_cursor_message_id = NULL WHERE id = ?", (sid,)
    )
    with pytest.raises(RuntimeError, match="has not reached"):
        publish_profile_generation(conn, sid, generation)

    conn.execute(
        "UPDATE sessions SET profile_cursor_message_id = ?, "
        "profile_cursor_prompt_version = 'arbitrary|walk=x' WHERE id = ?",
        (mid, sid),
    )
    with pytest.raises(RuntimeError, match="active cursor"):
        publish_profile_generation(conn, sid, "arbitrary|walk=x")


def test_profile_publication_rejects_a_gap_in_the_staged_cursor_chain(conn):
    sid = "profile-publish-chain-gap"
    mid = _msg(conn, sid, "I live in Utrecht.")
    _cover(conn, sid)
    generation = (
        profile_config_version(max_chars=100, max_items=16)
        + "|walk="
        + ("1" * 32)
    )
    first = ProfileExtraction(
        items=[],
        covered_message_id=None,
        partial_message_id=mid,
        next_message_offset=5,
        start_message_id=mid,
        start_message_offset=0,
        end_message_id=mid,
        cursor_before_message_id=None,
        cursor_before_partial_message_id=None,
        cursor_before_offset=0,
        caught_up=False,
        slice_key="slice-0-5",
    )
    # Offset 5 is deliberately absent: this second staged slice begins at 6.
    second = ProfileExtraction(
        items=[],
        covered_message_id=mid,
        partial_message_id=None,
        next_message_offset=0,
        start_message_id=mid,
        start_message_offset=6,
        end_message_id=mid,
        cursor_before_message_id=None,
        cursor_before_partial_message_id=mid,
        cursor_before_offset=6,
        caught_up=True,
        slice_key="slice-6-tail",
    )
    stage_profile_extraction(conn, sid, generation, first)
    stage_profile_extraction(conn, sid, generation, second)
    conn.execute(
        "UPDATE sessions SET profile_cursor_message_id = ?, "
        "profile_cursor_partial_message_id = NULL, profile_cursor_offset = 0, "
        "profile_cursor_prompt_version = ? WHERE id = ?",
        (mid, generation, sid),
    )
    with pytest.raises(RuntimeError, match="slice chain is incomplete"):
        publish_profile_generation(conn, sid, generation)
    assert conn.execute(
        "SELECT COUNT(*) FROM profile_staging WHERE session_id = ?", (sid,)
    ).fetchone()[0] == 2


@pytest.mark.parametrize("tampered_field", ["extra", "source_message_id"])
def test_profile_publication_rejects_tampered_internal_stage_schema(
    conn, tampered_field,
):
    sid = f"profile-stage-tamper-{tampered_field}"
    mid = _msg(conn, sid, "I live in Utrecht.")
    _cover(conn, sid)
    generation = (
        profile_config_version(max_chars=100, max_items=16)
        + "|walk=" + ("2" * 32)
    )
    created = conn.execute(
        "SELECT created_at FROM messages WHERE id = ?", (mid,)
    ).fetchone()[0]
    extraction = ProfileExtraction(
        items=[{
            **_item("location", "Utrecht", mid),
            "source_session_id": sid,
            "source_created_at": created,
        }],
        covered_message_id=mid,
        start_message_id=mid,
        end_message_id=mid,
        caught_up=True,
        slice_key="tampered",
    )
    stage_profile_extraction(conn, sid, generation, extraction)
    staged = conn.execute(
        "SELECT items_json FROM profile_staging WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    [item] = json.loads(staged)
    item[tampered_field] = mid if tampered_field == "source_message_id" else "x"
    conn.execute(
        "UPDATE profile_staging SET items_json = ? WHERE session_id = ?",
        (json.dumps([item]), sid),
    )
    conn.execute(
        "UPDATE sessions SET profile_cursor_message_id = ?, "
        "profile_cursor_prompt_version = ? WHERE id = ?",
        (mid, generation, sid),
    )
    with pytest.raises(RuntimeError, match="unexpected fields"):
        publish_profile_generation(conn, sid, generation)
    assert load_profile(conn) == []


def test_profile_publication_rejects_duplicate_keys_in_staged_json(conn):
    sid = "profile-stage-duplicate-key"
    mid = _msg(conn, sid, "I live in Utrecht.")
    _cover(conn, sid)
    generation = profile_config_version(max_chars=100, max_items=16) + "|walk=" + ("3" * 32)
    created = conn.execute(
        "SELECT created_at FROM messages WHERE id = ?", (mid,)
    ).fetchone()[0]
    extraction = ProfileExtraction(
        items=[{
            **_item("location", "Utrecht", mid),
            "source_session_id": sid,
            "source_created_at": created,
        }],
        covered_message_id=mid, start_message_id=mid, end_message_id=mid,
        caught_up=True, slice_key="duplicate-key",
    )
    stage_profile_extraction(conn, sid, generation, extraction)
    ambiguous = (
        '[{"slot":"location","slot":"employer","value":"Utrecht",'
        f'"evidence_message_id":{mid},"confidence":0.9,'
        f'"source_session_id":"{sid}","source_created_at":"{created}"}}]'
    )
    conn.execute(
        "UPDATE profile_staging SET items_json = ? WHERE session_id = ?",
        (ambiguous, sid),
    )
    conn.execute(
        "UPDATE sessions SET profile_cursor_message_id = ?, "
        "profile_cursor_prompt_version = ? WHERE id = ?",
        (mid, generation, sid),
    )
    with pytest.raises(RuntimeError, match="payload is not an array"):
        publish_profile_generation(conn, sid, generation)
    assert load_profile(conn) == []


def test_render_profile_fact_shapes():
    from hymem.dreaming.user_profile import ProfileEntry
    plain = ProfileEntry("role", None, "bedrijfsarts", 1.0, 1, "2026-01-01")
    keyed = ProfileEntry("relationship", "anna", "sister", 1.0, 1, "2026-01-01")
    assert render_profile_fact(plain) == "user role bedrijfsarts"
    assert render_profile_fact(keyed) == "user relationship(anna) sister"


# ── migrations v18 / v19 ─────────────────────────────────────────────────────


def _has_table(conn, name) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def test_fresh_store_lands_at_current_version_with_user_profile(tmp_path: Path):
    conn = core_db.connect(tmp_path / "fresh.sqlite")
    core_db.initialize(conn)
    assert core_db.schema_version(conn) == 46 == core_db.EXPECTED_SCHEMA_VERSION
    assert _has_table(conn, "user_profile")
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(user_profile)")}
    assert {"slot", "slot_key", "value", "evidence_message_id", "confidence",
            "valid_at", "invalid_at", "created_at"} <= cols
    # v19: the per-session profile stamp exists on a fresh store too.
    session_cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    assert "profile_prompt_version" in session_cols
    conn.close()


def test_migration_018_applies_on_existing_v17_store(tmp_path: Path):
    """A v17 store (pre-profile) is walked forward: the table + active index
    appear and the version advances, all without touching existing rows."""
    conn = core_db.connect(tmp_path / "v17.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '17');
        CREATE TABLE sessions(id TEXT PRIMARY KEY);
        CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE dream_runs(id INTEGER PRIMARY KEY AUTOINCREMENT);
        CREATE TABLE knowledge_graph(id INTEGER PRIMARY KEY,
            subject_canonical TEXT, predicate TEXT, object_canonical TEXT,
            first_seen TIMESTAMP, last_seen TIMESTAMP,
            valid_at TIMESTAMP, invalid_at TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active');
        INSERT INTO sessions(id) VALUES ('s');
        INSERT INTO messages(session_id, role, content)
            VALUES ('s', 'user', 'I am a bedrijfsarts.');
        """
    )
    assert not _has_table(conn, "user_profile")

    core_db._run_migrations(conn)  # from v17: migrations 018-020 apply

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
    assert _has_table(conn, "user_profile")
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='index' AND name='idx_user_profile_active'"
    ).fetchone()
    assert idx is not None, "migration 018 must create the active-rows index"
    # Existing data untouched; new writes require canonical USER provenance.
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "s")
    source = conn.execute(
        "SELECT created_at FROM messages WHERE id = 1"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO user_profile(slot, value, evidence_message_id, "
        "source_message_id, source_session_id, source_created_at) "
        "VALUES ('role', 'bedrijfsarts', 1, 1, 's', ?)",
        (source,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO user_profile(slot, value) VALUES ('made_up', 'x')")
    conn.close()


def test_migration_019_purges_v1_rows_and_adds_session_stamp(tmp_path: Path):
    """A v18 store carrying profile.v1 rows is walked to v19: the failed-gate
    rows are PURGED (regenerable by re-dreaming; v18 never shipped), the
    sessions.profile_prompt_version column appears, and unrelated data
    survives."""
    conn = core_db.connect(tmp_path / "v18.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '18');
        CREATE TABLE sessions(id TEXT PRIMARY KEY, summary TEXT,
            digested_prompt_version TEXT);
        CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot TEXT NOT NULL,
            slot_key TEXT,
            value TEXT NOT NULL,
            evidence_message_id INTEGER,
            confidence REAL NOT NULL DEFAULT 1.0,
            valid_at TIMESTAMP,
            invalid_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE dream_runs(id INTEGER PRIMARY KEY AUTOINCREMENT);
        CREATE TABLE knowledge_graph(id INTEGER PRIMARY KEY,
            subject_canonical TEXT, predicate TEXT, object_canonical TEXT,
            first_seen TIMESTAMP, last_seen TIMESTAMP,
            valid_at TIMESTAMP, invalid_at TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active');
        INSERT INTO sessions(id) VALUES ('s');
        INSERT INTO messages(session_id, role, content)
            VALUES ('s', 'user', 'Repos: ClaudetteMedSer/HyMem');
        INSERT INTO user_profile(slot, value, evidence_message_id) VALUES
            ('employer', 'ClaudetteMedSer', 1),
            ('possession', 'HyMem repository', 1);
        """
    )

    core_db._run_migrations(conn)  # from v18: migrations 019-020 apply

    assert core_db.schema_version(conn) == 46 == core_db.EXPECTED_SCHEMA_VERSION
    # The ~8%-precision profile.v1 rows are gone…
    assert conn.execute("SELECT COUNT(*) AS c FROM user_profile").fetchone()["c"] == 0
    # …the per-session stamp column exists (and starts NULL)…
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    assert "profile_prompt_version" in cols
    row = conn.execute(
        "SELECT profile_prompt_version FROM sessions WHERE id = 's'"
    ).fetchone()
    assert row["profile_prompt_version"] is None
    # …and unrelated data is untouched.
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1
    conn.close()


# ── fenced replies (dream 1013) ──────────────────────────────────────────────


def test_extract_user_profile_parses_fenced_reply(conn):
    """The profile call sets response_format="json"; dream 1013 proved that is
    a request, not a contract. A fenced reply used to drop the whole session's
    profile with no log at all."""
    sid = "s_fenced_profile"
    mid = _msg(conn, sid, "Tegenwoordig werk ik als bedrijfsarts bij MedFlow.")
    _cover(conn, sid)
    fenced = "```json\n" + _answer([_item("role", "bedrijfsarts", mid)]) + "\n```"
    llm = StubLLMClient(fixtures={_NEEDLE: fenced}, default="[]")
    extraction = extract_user_profile(
        conn, sid, llm, max_chars=4000, max_items=10
    )
    assert extraction is not None
    assert [(i["slot"], i["value"]) for i in extraction.items] == [
        ("role", "bedrijfsarts")
    ]


def test_extract_user_profile_refusal_yields_empty_extraction(conn, caplog):
    """An unparseable reply keeps the documented empty ProfileExtraction —
    a refusal must never be laundered into profile facts — and now logs."""
    sid = "s_refusal_profile"
    _msg(conn, sid, "Tegenwoordig werk ik als bedrijfsarts bij MedFlow.")
    _cover(conn, sid)
    llm = StubLLMClient(
        fixtures={_NEEDLE: "I'm sorry, I can't help with that."}, default="[]"
    )
    with caplog.at_level("WARNING"):
        extraction = extract_user_profile(
            conn, sid, llm, max_chars=4000, max_items=10
        )
    assert extraction is not None and extraction.failed is True
    assert extraction.failure_reason == "parse_failure"
    assert any(
        "user_profile.parse_failure" in r.message and sid in r.getMessage()
        for r in caplog.records
    )


def test_extract_user_profile_wrong_shape_yields_empty_extraction(conn, caplog):
    """Valid JSON, wrong shape. validate_profile_items() already returned []
    for it; without a log that is indistinguishable from "nothing about this
    user was profile-worthy"."""
    sid = "s_shape_profile"
    _msg(conn, sid, "Tegenwoordig werk ik als bedrijfsarts bij MedFlow.")
    _cover(conn, sid)
    llm = StubLLMClient(
        fixtures={_NEEDLE: '{"profile": "nothing to report"}'}, default="[]"
    )
    with caplog.at_level("WARNING"):
        extraction = extract_user_profile(
            conn, sid, llm, max_chars=4000, max_items=10
        )
    assert extraction is not None and extraction.failed is True
    assert extraction.failure_reason == "shape_failure"
    assert any(
        "user_profile.shape_failure" in r.message and sid in r.getMessage()
        for r in caplog.records
    )
