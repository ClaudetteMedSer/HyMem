"""Tests for the temporal-reasoning (TR) retrieval path: stdlib date parsing
(dreaming/dates.py), the dream-cycle temporal_mentions index (dreaming/
temporal.py), and augment(ability="TR") returning a date-ordered event list.
"""

from __future__ import annotations

from tests.conftest import make_routed_llm, seed_edge


# --- date parsing ----------------------------------------------------------


from hymem.dreaming.dates import extract_dates


def _norm(text: str) -> list[str | None]:
    return [m.normalized_date for m in extract_dates(text)]


def test_parse_iso_date():
    assert _norm("we shipped on 2024-02-15 finally") == ["2024-02-15"]


def test_parse_english_month_name_with_year():
    assert _norm("released March 1st, 2023") == ["2023-03-01"]
    assert _norm("the Feb 15 2022 meeting") == ["2022-02-15"]


def test_parse_dutch_month_name():
    # Year-less Dutch month form: month resolved, no full ISO (None) but captured.
    mentions = extract_dates("de release was 15 maart")
    assert len(mentions) == 1
    assert mentions[0].normalized_date is None
    assert mentions[0].raw_text == "15 maart"
    # With a year it normalizes fully.
    assert _norm("op 3 mrt 2024") == ["2024-03-03"]


def test_parse_numeric_forms_and_ambiguity():
    # Day > 12 disambiguates regardless of separator.
    assert _norm("op 15-02-2024 gepland") == ["2024-02-15"]
    # Slash form assumes month-first; with year it fully resolves.
    assert _norm("release 02/15/2024") == ["2024-02-15"]
    # Year-less numeric → captured but not normalized.
    mentions = extract_dates("around 2/15")
    assert mentions and mentions[0].normalized_date is None


def test_parse_ignores_text_without_dates():
    assert extract_dates("no dates at all here") == []
    assert extract_dates("") == []


def test_parse_iso_not_double_matched_by_numeric():
    # The 02-15 inside an ISO date must not also fire the numeric matcher.
    mentions = extract_dates("2024-02-15")
    assert [m.raw_text for m in mentions] == ["2024-02-15"]


# --- dreaming index population --------------------------------------------


def _dream_with_dates(hy, sid, turns):
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)
    # A triple keeps phase1 happy; the temporal pass runs on chunk creation
    # regardless of extraction output.
    hy.set_llm(make_routed_llm([], []))
    hy.dream()


def test_dream_populates_temporal_mentions(hy):
    sid = "tr-dream"
    _dream_with_dates(
        hy,
        sid,
        [
            ("user", "We migrated to postgres on 2024-02-15 after testing."),
            ("user", "Then we added redis on 2024-05-01 for caching."),
        ],
    )

    rows = hy.conn.execute(
        "SELECT normalized_date, session_id FROM temporal_mentions "
        "ORDER BY normalized_date"
    ).fetchall()
    dates = [r["normalized_date"] for r in rows]
    assert "2024-02-15" in dates
    assert "2024-05-01" in dates
    assert all(r["session_id"] == sid for r in rows)


def test_temporal_index_is_idempotent_across_redreams(hy):
    sid = "tr-idem"
    _dream_with_dates(
        hy, sid, [("user", "shipped v2 on 2024-03-10 and told the team.")]
    )
    before = hy.conn.execute(
        "SELECT COUNT(*) AS c FROM temporal_mentions"
    ).fetchone()["c"]
    assert before >= 1

    hy.dream()  # re-dream: UNIQUE(message_id, raw_text) must prevent duplicates
    after = hy.conn.execute(
        "SELECT COUNT(*) AS c FROM temporal_mentions"
    ).fetchone()["c"]
    assert after == before


# --- augment(ability="TR") -------------------------------------------------


def test_tr_returns_events_in_chronological_order(hy):
    sid = "tr-order"
    _dream_with_dates(
        hy,
        sid,
        [
            ("user", "We deployed the api on 2024-05-01 to production."),
            ("user", "But we first set up the api on 2024-01-10 in staging."),
        ],
    )

    ctx = hy.augment("when did we work on the api?", ability="TR")
    dates = [e.date for e in ctx.temporal_events]
    assert dates == sorted(dates)  # ascending
    assert "2024-01-10" in dates and "2024-05-01" in dates
    assert all(e.source == "message" for e in ctx.temporal_events)


def test_tr_merges_dated_graph_edges(hy):
    sid = "tr-graph"
    _dream_with_dates(
        hy, sid, [("user", "we adopted kafka on 2024-06-01 for events.")]
    )
    # A dated graph edge for a matched entity. kafka is registered as an alias
    # so match_known_entities resolves it (a single object-only edge would fail
    # the entity-shape filter), and it sits in subject position here.
    hy.register_alias("kafka", "kafka")
    seed_edge(hy.conn, "kafka", "deploys_to", "production", days_ago=100)

    ctx = hy.augment("timeline for kafka", ability="TR")
    sources = {e.source for e in ctx.temporal_events}
    assert "graph" in sources  # the dated edge surfaced
    assert "message" in sources  # the message mention surfaced
    # Still globally sorted ascending across both sources.
    dates = [e.date for e in ctx.temporal_events]
    assert dates == sorted(dates)


def test_tr_empty_when_no_dates(hy):
    sid = "tr-none"
    _dream_with_dates(hy, sid, [("user", "we use postgres for everything.")])
    ctx = hy.augment("what database do we use?", ability="TR")
    assert ctx.temporal_events == []


def test_non_tr_ability_leaves_temporal_events_empty(hy):
    sid = "tr-off"
    _dream_with_dates(
        hy, sid, [("user", "we shipped on 2024-02-15 to prod.")]
    )
    # Default path: temporal_events must stay empty (no extra work, no leakage).
    ctx = hy.augment("when did we ship?")
    assert ctx.temporal_events == []


def test_tr_degrades_gracefully_without_table(hy):
    # Drop the table to simulate a pre-v14 DB; the TR path must return [] not raise.
    hy.conn.execute("DROP TABLE temporal_mentions")
    ctx = hy.augment("what happened first?", ability="TR")
    assert ctx.temporal_events == []
