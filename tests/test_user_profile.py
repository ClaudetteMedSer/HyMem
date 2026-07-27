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
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming.aggregate import _anchor_facts, build_aggregation_nodes, load_digest
from hymem.dreaming.user_profile import (
    PROFILE_PROMPT_VERSION,
    ProfileExtraction,
    build_profile_user_prompt,
    load_profile,
    persist_user_profile,
    render_profile_fact,
    validate_profile_items,
)
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


def _persist(conn, items, *, redact=True) -> int:
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
        stub_llm.fixtures[_NEEDLE] = json.dumps([
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
        assert PROFILE_PROMPT_VERSION == "profile.v2"
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
        stub_llm.fixtures[_NEEDLE] = json.dumps([_item("location", "Amsterdam", mid)])
        hy.dream()
        assert len(_profile_calls(stub_llm)) == 1
        assert _stamp(hy.conn, sid) == PROFILE_PROMPT_VERSION
        hy.dream()  # nothing changed
        assert len(_profile_calls(stub_llm)) == 1
        # And the re-asserted row was not duplicated either way.
        assert len(_rows(hy.conn)) == 1
    finally:
        hy.close()


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
        stub_llm.fixtures[_NEEDLE] = json.dumps([_item("role", "bedrijfsarts", mid)])
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
        stub_llm.fixtures[_NEEDLE] = "[]"  # valid extraction, zero items
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
        stub_llm.fixtures[_NEEDLE] = json.dumps([_item("role", "bedrijfsarts", mid)])
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


def test_validate_caps_items_and_strips_stray_keys():
    valid_ids = {1}
    raw = [_item("language", f"lang{i}", 1) for i in range(10)]
    assert len(validate_profile_items(raw, valid_ids, max_items=3)) == 3
    # A slot_key on a non-parameterized slot is dropped, not persisted.
    [item] = validate_profile_items(
        [_item("employer", "MedFlow", 1, key="acme")], valid_ids, max_items=16
    )
    assert item["slot_key"] is None
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
        stub_llm.fixtures[_NEEDLE] = json.dumps([
            _item("favorite_color", "blue", mid),   # invented slot → dropped
            _item("role", "teacher", mid),
        ])
        hy.dream()
        rows = _rows(hy.conn)
        assert [(r["slot"], r["value"]) for r in rows] == [("role", "teacher")]
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


# ── prompt rendering (shared with the front-run gate script) ─────────────────


def test_build_profile_user_prompt_tags_and_truncates():
    turns = [(7, "I am a nurse."), (9, "x" * 50)]
    prompt = build_profile_user_prompt(turns, max_chars=10_000)
    assert "[msg 7] I am a nurse." in prompt
    assert "[msg 9]" in prompt
    assert prompt.rstrip().endswith("Return the profile JSON array now.")
    # The body honors the char cap the dream phase applies.
    capped = build_profile_user_prompt(turns, max_chars=20)
    assert "x" * 50 not in capped


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
    assert core_db.schema_version(conn) == 23 == core_db.EXPECTED_SCHEMA_VERSION
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
    # Existing data untouched; new table usable with the closed vocabulary.
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1
    conn.execute(
        "INSERT INTO user_profile(slot, value, evidence_message_id) "
        "VALUES ('role', 'bedrijfsarts', 1)"
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

    assert core_db.schema_version(conn) == 23 == core_db.EXPECTED_SCHEMA_VERSION
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
