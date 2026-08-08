"""Tests for the E1 narrative-facts tier (schema v26).

Facts are a THIRD granularity next to graph triples and episode summaries: one
self-contained sentence per thing that happened, extracted at dream time and
served as an additive retrieval tier plus the lead evidence block in `ask()`.

The properties that make the tier safe to ship while other Campaign E items are
still gated are the ones under test here:

  * **Append-only.** Fact text is immutable; a re-dream never rewrites a stored
    fact, and a prompt bump extracts FORWARD ONLY, so every row stays
    attributable to the prompt version that produced it.
  * **Watermarked coverage** (`sessions.facts_message_id`, the v24 pattern in
    its own column): a quiescent store costs zero extraction calls, a parse
    failure holds the watermark so the slice retries, and coverage never goes
    backwards.
  * **Additive retrieval.** Turning the tier on cannot change what the
    message/chunk/episode/graph tiers return — the invariant that lets the tier
    default ON without a benchmark rebaseline.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import facts as facts_mod
from hymem.extraction.llm import StubLLMClient
from hymem.query.ask import render_context
from hymem.query.augment import AugmentedContext, FactHit, MessageHit
from hymem.rules import Rule

# The unique closer of FACTS_USER_TEMPLATE — routes stubs and counts calls
# without colliding with the digest/triple/profile prompts.
_FACTS_CLOSER = "Return the JSON array of narrative facts now"


# --- helpers ---------------------------------------------------------------


_FACTS_A = [
    {
        "text": "Atta moved the MedFlow deploy to fly.io on 2026-03-04.",
        "date": "2026-03-04",
        "entities": ["MedFlow", "fly.io"],
    },
    {
        "text": "Atta chose uv over pip for the HyMem toolchain.",
        "date": None,
        "entities": ["uv"],
    },
]

_FACTS_B = [
    {
        "text": "Atta rolled the staging release back with helm rollback.",
        "date": None,
        "entities": ["helm"],
    },
]

_TURNS = [
    ("assistant", "where does MedFlow deploy?"),
    ("user", "We moved the MedFlow deploy to fly.io, and switched to uv instead of pip."),
]

_MORE_TURNS = [
    ("assistant", "and how do we undo a bad staging release?"),
    ("user", "Run helm rollback staging to the previous revision; that reverts it."),
]


def _facts_llm(
    payload: list[dict] | str = _FACTS_A,
    *,
    extra: dict[str, str] | None = None,
) -> StubLLMClient:
    """Stub returning `payload` for the facts call and "[]" everywhere else.

    `extra` maps a routing substring to a raw response and is inserted FIRST,
    so a fixture keyed on text unique to a later tail (e.g. "helm rollback")
    wins over the catch-all closer — the way a second dream over new turns
    yields different facts.
    """
    body = payload if isinstance(payload, str) else json.dumps(payload)
    fixtures = {**(extra or {}), _FACTS_CLOSER: body}
    return StubLLMClient(fixtures=fixtures, default="[]")


def _fact_calls(llm: StubLLMClient) -> list:
    return [c for c in llm.calls if _FACTS_CLOSER in c.user]


def _seed_session(hy: HyMem, sid: str, turns: list[tuple[str, str]]) -> None:
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)


def _rows(hy: HyMem) -> list[dict]:
    return [
        dict(r)
        for r in hy.conn.execute(
            "SELECT id, session_id, start_message_id, end_message_id, text, "
            "fact_date, entities, prompt_version, invalid_at "
            "FROM narrative_facts ORDER BY id"
        ).fetchall()
    ]


def _watermark(hy: HyMem, sid: str) -> int | None:
    return hy.conn.execute(
        "SELECT facts_message_id FROM sessions WHERE id = ?", (sid,)
    ).fetchone()["facts_message_id"]


# --- 1. extraction -> persist round-trip ------------------------------------


def test_extraction_persists_facts_with_canonical_entities_and_watermark(cfg):
    """One dream writes the validated facts, canonicalizes their entities, and
    advances the watermark to the last message the extractor actually read."""
    llm = _facts_llm()
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_roundtrip"
        _seed_session(hy, sid, _TURNS)
        report = hy.dream()

        assert report.facts_extracted == 2
        assert report.fact_failures == 0

        rows = _rows(hy)
        assert [r["text"] for r in rows] == [f["text"] for f in _FACTS_A]
        # Entities are stored canonical (the store speaks canonical ids), not
        # as the verbatim surface forms the extractor emitted.
        assert json.loads(rows[0]["entities"]) == ["med_flow", "fly_io"]
        assert all(r["prompt_version"] == facts_mod.FACTS_PROMPT_VERSION for r in rows)

        # An explicit date is kept; a null date stays null (never a
        # session-date fallback — the G-F1 date lesson).
        assert rows[0]["fact_date"] == "2026-03-04"
        assert rows[1]["fact_date"] is None

        newest = hy.conn.execute(
            "SELECT MAX(id) AS m FROM messages WHERE session_id = ?", (sid,)
        ).fetchone()["m"]
        assert _watermark(hy, sid) == newest
        assert all(r["end_message_id"] == newest for r in rows)

        # dream_runs carries the attribution (the v25 counters' pattern).
        run = hy.conn.execute(
            "SELECT facts_extracted, fact_failures FROM dream_runs "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run["facts_extracted"] == 2
        assert run["fact_failures"] == 0
    finally:
        hy.close()


# --- 2. append-only ---------------------------------------------------------


def test_new_traffic_appends_without_touching_stored_facts(cfg):
    """New messages produce facts for the NEW range only; the rows from the
    first dream are byte-identical afterwards. Immutability is what makes a
    later prompt bump safe."""
    llm = _facts_llm(extra={"helm rollback": json.dumps(_FACTS_B)})
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_append"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        before = _rows(hy)
        first_covered = _watermark(hy, sid)

        _seed_session(hy, sid, _MORE_TURNS)
        hy.dream()
        after = _rows(hy)

        assert after[: len(before)] == before, "stored facts must never be rewritten"
        new = after[len(before):]
        assert [r["text"] for r in new] == [f["text"] for f in _FACTS_B]
        assert new[0]["start_message_id"] > first_covered, (
            "the new fact must be anchored to the new range, not the old one"
        )
        assert _watermark(hy, sid) > first_covered

        # The second call saw only the tail: the head turns are not re-sent.
        last = _fact_calls(llm)[-1].user
        assert "helm rollback" in last
        assert "MedFlow deploy to fly.io" not in last
    finally:
        hy.close()


# --- 3. idempotent re-dream -------------------------------------------------


def test_redream_of_quiescent_session_costs_nothing(cfg):
    """A caught-up session makes zero extraction calls and writes zero rows;
    the watermark is stable. This is the steady-state cost contract."""
    llm = _facts_llm()
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_quiet"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        calls = len(_fact_calls(llm))
        rows = _rows(hy)
        mark = _watermark(hy, sid)
        assert calls == 1

        report = hy.dream()

        assert len(_fact_calls(llm)) == calls, "a quiescent tail must cost no call"
        assert report.facts_extracted == 0
        assert _rows(hy) == rows
        assert _watermark(hy, sid) == mark
    finally:
        hy.close()


# --- 4. forward-only prompt versioning --------------------------------------


def test_prompt_version_bump_extracts_forward_only(cfg, monkeypatch):
    """A FACTS_PROMPT_VERSION bump tags NEW ranges with the new version and
    leaves covered ranges untouched — no re-extraction, no rewrite. This is
    what lets the prompt evolve without disturbing stored evidence."""
    llm = _facts_llm(extra={"helm rollback": json.dumps(_FACTS_B)})
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_bump"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        before = _rows(hy)
        calls = len(_fact_calls(llm))

        monkeypatch.setattr(facts_mod, "FACTS_PROMPT_VERSION", "facts.v3")

        # A bump ALONE re-extracts nothing: the watermark, not a version stamp,
        # is the facts skip-guard (deliberately unlike digest/profile).
        hy.dream()
        assert len(_fact_calls(llm)) == calls
        assert _rows(hy) == before

        _seed_session(hy, sid, _MORE_TURNS)
        hy.dream()
        after = _rows(hy)

        assert after[: len(before)] == before
        assert [r["prompt_version"] for r in after[len(before):]] == ["facts.v3"]
    finally:
        hy.close()


# --- 5. parse failure -------------------------------------------------------


def test_parse_failure_holds_the_watermark_and_retries(cfg):
    """An unparseable reply counts into fact_failures, claims no coverage, and
    leaves the slice to be retried — advancing here would silently skip it
    forever (the starvation class migration 024 exists to prevent)."""
    sid = "s_badjson"
    bad = _facts_llm("not json at all")
    hy = HyMem(cfg, llm=bad)
    try:
        _seed_session(hy, sid, _TURNS)
        report = hy.dream()
        assert report.fact_failures == 1
        assert report.facts_extracted == 0
        assert _rows(hy) == []
        assert _watermark(hy, sid) is None, "a parse failure must not claim coverage"
    finally:
        hy.close()

    good = _facts_llm()
    hy2 = HyMem(cfg, llm=good)
    try:
        report = hy2.dream()
        assert len(_fact_calls(good)) == 1, "the uncovered slice must be retried"
        assert report.facts_extracted == 2
        assert _watermark(hy2, sid) is not None
    finally:
        hy2.close()


def test_an_oversized_turn_is_covered_rather_than_re_read_forever(cfg):
    """A single turn longer than the char cap is truncated, stored, and
    COVERED. Truncation keeps the head, so no later dream could read more of
    it — holding the watermark would re-spend the call every dream forever and
    store nothing."""
    llm = _facts_llm()
    small = dataclasses.replace(cfg, dream_digest_max_chars=200)
    hy = HyMem(small, llm=llm)
    try:
        sid = "s_huge"
        _seed_session(hy, sid, [("user", "I shipped it. " + "filler words " * 200)])
        report = hy.dream()

        assert report.facts_extracted == 2, "the truncated turn still yields facts"
        assert _watermark(hy, sid) is not None, "coverage must advance"

        calls = len(_fact_calls(llm))
        hy.dream()
        assert len(_fact_calls(llm)) == calls, "the oversized turn is not re-read"
    finally:
        hy.close()


# --- 6. UNIQUE dedup --------------------------------------------------------


def test_resubmitting_the_same_range_inserts_nothing(cfg):
    """The UNIQUE (session_id, start_message_id, text) key is the idempotency
    backstop under the watermark: a replayed extraction no-ops row by row."""
    llm = _facts_llm()
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_dedup"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        rows = _rows(hy)

        extraction = facts_mod.extract_facts(
            hy.conn, sid, llm, hy.config, since_message_id=None
        )
        assert extraction is not None
        with core_db.transaction(hy.conn):
            inserted = facts_mod.persist_facts(hy.conn, sid, extraction)

        assert inserted == 0
        assert _rows(hy) == rows
    finally:
        hy.close()


# --- 7. the retrieval tier --------------------------------------------------


def test_tier_surfaces_matches_skips_non_matches_and_hides_superseded(cfg):
    """The tier returns matching facts, stays empty for an unrelated query, and
    drops a superseded fact (invalid_at set) while KEEPING its row for audit."""
    llm = _facts_llm()
    hy = HyMem(dataclasses.replace(cfg, facts_enabled=True), llm=llm)  # default OFF post-E1; pin on
    try:
        sid = "s_tier"
        _seed_session(hy, sid, _TURNS)
        hy.dream()

        hits = hy.augment("what happened with the MedFlow deploy?").facts
        assert [h.text for h in hits][:1] == [_FACTS_A[0]["text"]]
        assert hits[0].fact_date == "2026-03-04"
        assert hits[0].entities == ["med_flow", "fly_io"]
        assert hits[0].session_id == sid
        assert any("fact_fts" in chip for chip in hits[0].why_retrieved)

        assert hy.augment("kayaking in Patagonia").facts == []

        # E6's write, simulated: close the validity interval on the top fact.
        fact_id = hits[0].fact_id
        with core_db.transaction(hy.conn):
            hy.conn.execute(
                "UPDATE narrative_facts SET invalid_at = CURRENT_TIMESTAMP WHERE id = ?",
                (fact_id,),
            )

        after = hy.augment("what happened with the MedFlow deploy?").facts
        assert fact_id not in [h.fact_id for h in after], "superseded facts leave the tier"
        assert hy.conn.execute(
            "SELECT 1 FROM narrative_facts WHERE id = ?", (fact_id,)
        ).fetchone() is not None, "...but the row stays for audit"
    finally:
        hy.close()


def test_facts_are_embedded_once_and_reach_the_vec_arm(cfg, embed_stub):
    """Facts get vectors (JSON mirror + vec_facts) on the dream that writes
    them, and a re-dream re-embeds nothing — fact text is immutable, so a
    stored vector can never be stale."""
    llm = _facts_llm()
    hy = HyMem(dataclasses.replace(cfg, facts_enabled=True), llm=llm,
               embedding_client=embed_stub)  # default OFF post-E1; pin on
    try:
        sid = "s_embed"
        _seed_session(hy, sid, _TURNS)
        report = hy.dream()

        assert report.facts_embedded == 2
        stored = hy.conn.execute(
            "SELECT COUNT(*) AS c FROM narrative_fact_embeddings"
        ).fetchone()["c"]
        assert stored == 2
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM vec_facts"
        ).fetchone()["c"] == 2

        # An embedded fact still retrieves, now through the fused path.
        hits = hy.augment("what happened with the MedFlow deploy?").facts
        assert hits, "retrieval must survive having an embedder wired"

        report2 = hy.dream()
        assert report2.facts_embedded == 0, "immutable text is embedded exactly once"
    finally:
        hy.close()


# --- 8. the additive invariant ---------------------------------------------


def test_facts_tier_does_not_disturb_any_other_tier(cfg):
    """The control that lets this tier default ON: with facts present, every
    other tier returns exactly what it returns with `facts_enabled=False`."""
    llm = _facts_llm()
    on_cfg = dataclasses.replace(cfg, facts_enabled=True)  # read-side default is OFF post-E1; pin on for the invariant test
    hy = HyMem(on_cfg, llm=llm)
    query = "what happened with the MedFlow deploy?"
    try:
        sid = "s_additive"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        on = hy.augment(query)
    finally:
        hy.close()

    off_cfg = dataclasses.replace(cfg, facts_enabled=False)
    hy_off = HyMem(off_cfg, llm=_facts_llm())
    try:
        off = hy_off.augment(query)
    finally:
        hy_off.close()

    assert on.facts, "precondition: the tier actually fired"
    assert off.facts == []
    assert on.message_hits == off.message_hits
    assert on.fts_hits == off.fts_hits
    assert on.episodes == off.episodes
    assert on.graph_facts == off.graph_facts
    assert on.matched_entities == off.matched_entities


# --- 9. pre-v26 degradation -------------------------------------------------


def test_pre_v26_store_degrades_to_an_empty_tier_and_skips_extraction(cfg):
    """A store that never got migration 026 must not crash: the tier is empty
    and extraction is skipped, both on the missing-object path."""
    llm = _facts_llm()
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_pre26"
        _seed_session(hy, sid, _TURNS)
        # Strip the v26 objects to simulate a store the migration never
        # reached. The watermark column is RENAMED rather than dropped: SQLite
        # rewrites the stored CREATE TABLE text on DROP COLUMN, and this
        # table's inline comment block would leave a dangling comma behind.
        # Either way the runtime sees "no such column: facts_message_id".
        with core_db.transaction(hy.conn):
            hy.conn.execute("DROP TABLE IF EXISTS narrative_facts_fts")
            hy.conn.execute("DROP TABLE IF EXISTS narrative_fact_embeddings")
            hy.conn.execute("DROP TABLE IF EXISTS narrative_facts")
            hy.conn.execute(
                "ALTER TABLE sessions RENAME COLUMN facts_message_id TO _gone"
            )

        assert hy.augment("MedFlow deploy").facts == []

        report = hy.dream()
        assert report.facts_extracted == 0
        assert report.fact_failures == 0
        assert _fact_calls(llm) == [], "no extraction call without a watermark column"
    finally:
        hy.close()


# --- 10. rendering ----------------------------------------------------------


def _ctx_for_render() -> AugmentedContext:
    ctx = AugmentedContext(
        rules=[Rule(id=1, text="always run the tests before pushing", scope="always_on")],
        facts=[
            FactHit(
                fact_id=1,
                text="Atta moved the MedFlow deploy to fly.io.",
                fact_date="2026-03-04",
                entities=["med_flow"],
                session_id="s",
                score=1.0,
            ),
            FactHit(
                fact_id=2,
                text="Atta chose uv over pip." + " padding" * 80,
                fact_date=None,
                entities=[],
                session_id="s",
                score=0.5,
            ),
        ],
        message_hits=[
            MessageHit(
                message_id=7,
                session_id="s",
                role="user",
                text="We moved the MedFlow deploy to fly.io.",
                score=-1.0,
                created_at="2026-03-04 10:00:00",
            )
        ],
    )
    ctx.graph_facts = []
    return ctx


def test_render_places_facts_above_conversation_evidence_and_caps_snippets():
    """Facts lead the evidence; the raw turns stay BELOW as the verification
    backup (the Acme lesson: a summary is never the only copy). Dates render,
    undated facts say so, and a long fact is snippet-capped like every other
    item."""
    block = render_context(_ctx_for_render(), max_chars=0)

    facts_at = block.index("=== FACTS (verified past events) ===")
    turns_at = block.index("=== CONVERSATION EVIDENCE")
    assert facts_at < turns_at

    assert "- [2026-03-04] Atta moved the MedFlow deploy to fly.io." in block
    assert "- [undated] Atta chose uv over pip." in block
    assert "..." in block, "the long fact is snippet-capped"
    assert " padding" * 80 not in block


def test_truncation_sheds_facts_before_standing_rules():
    """Tail-truncation order: a tight budget cuts the facts block while the
    STANDING RULES the model must obey survive."""
    ctx = _ctx_for_render()
    tight = render_context(ctx, max_chars=120)

    assert "=== STANDING RULES (always follow) ===" in tight
    assert "always run the tests before pushing" in tight
    assert "Atta moved the MedFlow deploy to fly.io." not in tight
    assert "[... context truncated]" in tight


# --- 11. ask() end-to-end ---------------------------------------------------


def test_ask_sends_the_facts_block_to_the_synthesis_call(cfg):
    """The one-call endpoint really carries facts into the prompt."""
    cfg = dataclasses.replace(cfg, facts_enabled=True)  # read-side default is OFF post-E1; pin on for the render-path test
    llm = _facts_llm()
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_ask"
        _seed_session(hy, sid, _TURNS)
        hy.dream()

        answer = hy.ask("what happened with the MedFlow deploy?")

        assert answer.context.facts, "retrieval fed the tier into ask()"
        sent = llm.calls[-1].user
        assert "=== FACTS (verified past events) ===" in sent
        assert "Atta moved the MedFlow deploy to fly.io on 2026-03-04." in sent
    finally:
        hy.close()


# --- 12. both switches off --------------------------------------------------


def test_both_flags_off_means_no_extraction_and_no_tier(cfg):
    llm = _facts_llm()
    off = dataclasses.replace(cfg, facts_enabled=False, facts_extraction_enabled=False)
    hy = HyMem(off, llm=llm)
    try:
        sid = "s_off"
        _seed_session(hy, sid, _TURNS)
        report = hy.dream()

        assert _fact_calls(llm) == []
        assert report.facts_extracted == 0
        assert _rows(hy) == []
        assert hy.augment("MedFlow deploy").facts == []
    finally:
        hy.close()


def test_facts_top_k_zero_disables_retrieval_but_not_extraction(cfg):
    """The two switches are independent: a host can keep building the fact
    store while the tier is out of the context budget."""
    llm = _facts_llm()
    hy = HyMem(dataclasses.replace(cfg, facts_top_k=0), llm=llm)
    try:
        sid = "s_topk0"
        _seed_session(hy, sid, _TURNS)
        report = hy.dream()

        assert report.facts_extracted == 2
        assert hy.augment("MedFlow deploy").facts == []
    finally:
        hy.close()


# --- 13. migration 026 ------------------------------------------------------


def _cols(conn, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_v26_adds_facts_table_watermark_and_dream_run_columns(tmp_path: Path):
    """Migration 026 against a v25-shaped DB: the new objects appear and the
    existing rows are untouched. `CREATE TABLE IF NOT EXISTS` in schema.sql
    no-ops on the pre-existing tables, so the columns can only come from the
    migration."""
    db = tmp_path / "v25.sqlite"
    conn = core_db.connect(db)
    conn.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '25');
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            summary TEXT,
            digested_prompt_version TEXT,
            profile_prompt_version TEXT,
            digested_message_id INTEGER
        );
        INSERT INTO sessions(id, summary) VALUES ('old', 'a pre-v26 session');
        CREATE TABLE dream_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP NOT NULL,
            digest_failures INTEGER NOT NULL DEFAULT 0,
            episodes_created INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO dream_runs(started_at, episodes_created) VALUES ('2026-07-30', 3);
        """
    )

    core_db.initialize(conn)

    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION == 29
    assert "facts_message_id" in _cols(conn, "sessions")
    assert {"facts_extracted", "fact_failures"} <= _cols(conn, "dream_runs")
    assert _cols(conn, "narrative_facts") >= {
        "id", "session_id", "start_message_id", "end_message_id", "text",
        "fact_date", "entities", "prompt_version", "valid_at", "invalid_at",
    }
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='narrative_facts_fts'"
    ).fetchone() is not None

    # Pre-existing data survives, and the new columns default rather than NULL
    # out an old row's counters.
    row = conn.execute("SELECT summary, facts_message_id FROM sessions").fetchone()
    assert row["summary"] == "a pre-v26 session"
    assert row["facts_message_id"] is None
    run = conn.execute(
        "SELECT episodes_created, facts_extracted, fact_failures FROM dream_runs"
    ).fetchone()
    assert (run["episodes_created"], run["facts_extracted"], run["fact_failures"]) == (3, 0, 0)
    conn.close()


def test_schema_version_guard_still_rejects_a_newer_db(tmp_path: Path):
    """The version-guard error path is intact after the bump: a DB written by a
    future release refuses to open rather than being silently downgraded."""
    db = tmp_path / "future.sqlite"
    conn = core_db.connect(db)
    core_db.initialize(conn)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(core_db.EXPECTED_SCHEMA_VERSION + 1),),
    )
    with pytest.raises(RuntimeError, match="newer than code expects"):
        core_db.initialize(conn)
    conn.close()


# --- validator unit coverage ------------------------------------------------


def test_validator_distinguishes_unparseable_from_empty():
    """None means "the reply was garbage" (hold the watermark, retry); [] means
    "this slice genuinely establishes nothing" (valid coverage). Collapsing the
    two would either lose slices or retry forever."""
    assert facts_mod.validate_fact_items("not json", max_items=8) is None
    assert facts_mod.validate_fact_items("{}", max_items=8) is None
    assert facts_mod.validate_fact_items("[]", max_items=8) == []


def test_validator_drops_a_malformed_date_but_keeps_the_fact():
    """The date is metadata, the text is the evidence."""
    items = facts_mod.validate_fact_items(
        json.dumps([{"text": "Atta shipped it.", "date": "last tuesday", "entities": []}]),
        max_items=8,
    )
    assert items == [{"text": "Atta shipped it.", "date": None, "entities": []}]


def test_validator_caps_facts_per_session():
    raw = json.dumps([{"text": f"fact {i}", "date": None, "entities": []} for i in range(20)])
    assert len(facts_mod.validate_fact_items(raw, max_items=8)) == 8


def test_validator_tolerates_a_fenced_array():
    items = facts_mod.validate_fact_items(
        '```json\n[{"text": "Atta shipped it.", "entities": ["MedFlow"]}]\n```',
        max_items=8,
    )
    assert items == [{"text": "Atta shipped it.", "date": None, "entities": ["med_flow"]}]


_ONE_FACT = '{"text": "Atta shipped it.", "entities": []}'


@pytest.mark.parametrize("raw, expected", [
    (f"[{_ONE_FACT}]", 1),                                    # bare array
    (f"```json\n[{_ONE_FACT}]\n```", 1),                      # fenced array
    (f"Here:\n```json\n[{_ONE_FACT}]\n```\nDone.", 1),        # prose + fence
    ('{"facts": [%s]}' % _ONE_FACT, 1),                       # bare envelope
    ('```json\n{"facts": [%s]}\n```' % _ONE_FACT, 1),         # fenced envelope
    ("I could not extract any facts.", None),                 # refusal
])
def test_validator_shape_table(raw, expected):
    # The envelope rows are the point: this call sets response_format="json"
    # (-> json_object) while FACTS_SYSTEM asks for a bare array, so
    # {"facts": [...]} is a shape the provider will genuinely emit. Rejecting
    # it returns None -> parse_failed -> the watermark holds -> the SAME slice
    # is re-extracted on every subsequent dream: an unbounded paid-for loop
    # that stores nothing. Bare and fenced must also agree exactly.
    items = facts_mod.validate_fact_items(raw, max_items=8)
    assert (None if items is None else len(items)) == expected


def test_validator_rejects_an_ambiguous_envelope():
    # Two lists = no single candidate. A wrong guess would silently store the
    # wrong half, which is worse than the retry.
    assert facts_mod.validate_fact_items(
        '{"facts": [%s], "notes": []}' % _ONE_FACT, max_items=8) is None
