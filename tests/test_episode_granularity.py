"""Plan C stage 1 — decision-grained episodes (schema v35, default OFF).

This is Plan C's validation gate 1 ("mechanical pytest, StubLLMClient"): the
box checks that must hold before the granularity question is even measurable.
Five things are pinned, and each one is a way the feature could look like it
works while doing nothing:

  * **Default-OFF invariance.** With `episode_granularity_enabled=False` the
    digest sends the shipping prompt, the cap does not apply, ids keep their
    bare-range shape and the new session stamp stays NULL. A store that never
    turns this on must not be able to tell it shipped.
  * **Multi-episode persistence.** At decision granularity several episodes of
    one session legitimately cite the SAME chunk, so they resolve to the same
    message range. Under the pre-Plan-C id that is ONE id, and the second UPSERT
    silently overwrites the first — a 4-episode session persists as 1 row and
    every downstream reading says the re-cut did nothing.
  * **Cap enforcement**, on the granular arm only.
  * **Stable-id UPSERT semantics** on a re-dream: same range + same title = same
    row, refreshed in place (rowid churn breaks the FTS/vec shadows).
  * **The prompt-version guard**: a granularity flip re-extracts every session
    exactly once, and an unchanged session then costs zero tail calls. Without
    the guard, already-digested sessions keep their blob episodes forever,
    because the digest guard keys on `cfg.prompt_version`, which a granularity
    flip does not move.

No LLM: StubLLMClient throughout, routed on the granular prompt's deliberately
unique closer.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming.digest import (
    EPISODE_GRANULAR_PROMPT_VERSION,
    active_episode_prompt_version,
    extract_session_digest,
)
from hymem.dreaming.episodes import (
    EpisodesExtraction,
    _episode_id,
    persist_episodes,
    validate_episode_items,
)
from hymem.extraction.llm import StubLLMClient

_BLOB_CLOSER = "Return the JSON object now"
_GRANULAR_CLOSER = "Return the granular digest JSON object now"


# --- helpers ---------------------------------------------------------------


def _digest_llm(episodes: list[dict] | None = None, summary: str = "") -> StubLLMClient:
    """Stub answering BOTH digest arms with the same payload, so a test that
    compares the arms is comparing prompts and never fixtures. Keyed on the two
    closers; `[]` for the chunk-extraction calls."""
    payload = json.dumps({
        "episodes": episodes or [],
        "summary": summary,
        "procedures": [],
    })
    return StubLLMClient(
        fixtures={_BLOB_CLOSER: payload, _GRANULAR_CLOSER: payload},
        default="[]",
    )


def _calls(llm: StubLLMClient, closer: str) -> list:
    """Digest calls of ONE arm. Discriminated on the closer, which is unique per
    arm — the RAPTOR root-digest fusion shares the blob closer, so the system
    prompt is checked too (the `_digest_calls` lesson in test_digest.py)."""
    return [
        c for c in llm.calls
        if closer in c.user and c.system.startswith(
            "You analyze one conversation session" if closer == _BLOB_CLOSER
            else "You re-read one conversation session"
        )
    ]


_TURNS = [
    ("assistant", "how do we deploy to staging?"),
    ("user", "Build the docker image then kubectl apply the staging manifests; "
             "that ships it to the cluster and takes about 4 minutes."),
    ("assistant", "and the database migration?"),
    ("user", "Run alembic upgrade head against staging first; version 2.1.4 "
             "adds the index that the deploy needs."),
]


def _seed(hy: HyMem, sid: str, turns=_TURNS) -> None:
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)


def _chunk_ids(hy: HyMem, sid: str) -> list[str]:
    return [
        r["id"] for r in hy.conn.execute(
            "SELECT id FROM chunks WHERE session_id = ? "
            "AND chunk_kind = 'coverage' ORDER BY start_message_id",
            (sid,),
        ).fetchall()
    ]


def _episode(title: str, chunk_ids: list[str], summary: str = "") -> dict:
    return {
        "title": title,
        "summary": summary or f"{title} — pinned at 2.1.4 on 2026-08-30.",
        "outcome": "resolved",
        "key_entities": ["alembic"],
        "chunk_ids": chunk_ids,
    }


@pytest.fixture
def granular_cfg(cfg):
    return dataclasses.replace(cfg, episode_granularity_enabled=True)


# --- default OFF is byte-identical ----------------------------------------


def test_flag_off_sends_the_shipping_prompt_and_stamps_nothing(cfg):
    """The whole default-OFF contract in one test: the granular prompt is never
    sent, and the v35 column stays NULL — which is exactly what a pre-v35 store
    reads, so the skip-guard below can never fire on a store that never opted
    in."""
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        _seed(hy, "s_off")
        hy.dream()
        assert len(_calls(llm, _BLOB_CLOSER)) == 1
        assert _calls(llm, _GRANULAR_CLOSER) == []
        assert hy.conn.execute(
            "SELECT episodes_prompt_version FROM sessions WHERE id = 's_off'"
        ).fetchone()["episodes_prompt_version"] is None
    finally:
        hy.close()


def test_flag_off_keeps_the_bare_range_episode_id(cfg):
    """Episode ids are the store's identity for re-dream UPSERTs, so the OFF arm
    must keep producing the pre-Plan-C shape. Asserted at a NON-NULL range (the
    seeded turns produce a real chunk), because the hash-id fallback would
    satisfy "no '#' in the id" for the wrong reason."""
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        _seed(hy, "s_ids")
        hy.dream()
        cids = _chunk_ids(hy, "s_ids")
        assert cids, "precondition: the session must have produced chunks"

        llm2 = _digest_llm(episodes=[_episode("Staging deploy", cids[:1])])
        digest = extract_session_digest(
            hy.conn, "s_ids", llm2,
            max_tokens=1024, max_chars=12000, granular=False,
        )
        with core_db.transaction(hy.conn):
            persist_episodes(hy.conn, "s_ids", digest.episodes, granular=False)

        row = hy.conn.execute(
            "SELECT id, start_message_id, end_message_id FROM episodes "
            "WHERE session_id = 's_ids'"
        ).fetchone()
        assert row["start_message_id"] is not None, (
            "precondition: the episode must resolve to a real range, or this "
            "test passes on the hash-id fallback instead of the range id"
        )
        assert row["id"] == f"s_ids@{row['start_message_id']}-{row['end_message_id']}"
        assert "#" not in row["id"]
    finally:
        hy.close()


def test_cap_does_not_apply_to_the_blob_arm(cfg):
    """`dream_max_episodes_per_session` bounds the granular arm only. The blob
    prompt has never been capped, and quietly trimming it would be a default
    change wearing a Plan C label. Exercised ABOVE the cap (20 > 12) so the
    assertion cannot pass by the reply being short."""
    items = [_episode(f"Episode {i}", []) for i in range(20)]
    assert len(items) > cfg.dream_max_episodes_per_session  # precondition
    assert len(validate_episode_items(items, set(), max_items=None)) == 20


# --- multi-episode persistence (the id-collision case) ---------------------


def test_several_granular_episodes_on_the_same_chunk_all_persist(granular_cfg):
    """The defect this feature would otherwise ship with.

    Three episodes, TWO of which cite the same chunk — the normal case at
    decision granularity ("chose fly.io" and "hit the 512MB limit" are one
    exchange). They resolve to one message range, so under the bare-range id
    they collide on one row and the session persists as fewer episodes than were
    extracted. The collision is asserted explicitly first: without that, this
    test would also pass against a store where the ranges happened to differ.
    """
    llm = _digest_llm()
    hy = HyMem(granular_cfg, llm=llm)
    try:
        _seed(hy, "s_multi")
        hy.dream()
        cids = _chunk_ids(hy, "s_multi")
        assert len(cids) >= 2, "precondition: need at least two chunks"

        items = [
            _episode("Chose the staging deploy path", [cids[0]]),
            _episode("Timed the deploy at 4 minutes", [cids[0]]),
            _episode("Pinned alembic to 2.1.4", [cids[-1]]),
        ]
        llm2 = _digest_llm(episodes=items)
        digest = extract_session_digest(
            hy.conn, "s_multi", llm2,
            max_tokens=1024, max_chars=12000, granular=True, max_episodes=12,
        )
        assert len(digest.episodes.items) == 3

        # PRECONDITION, the reason this test exists: two of the three items
        # share a message range, so the pre-Plan-C id function maps them onto
        # ONE id. Exercised at a real collision (2 items → 1 blob id), never at
        # the vacuous zero.
        from hymem.dreaming.episodes import _resolve_message_range
        ranges = [
            _resolve_message_range(hy.conn, it["chunk_ids"])
            for it in digest.episodes.items
        ]
        blob_ids = {
            _episode_id("s_multi", s, e, it["title"])
            for (s, e), it in zip(ranges, digest.episodes.items)
        }
        assert len(blob_ids) == 2 < len(digest.episodes.items), (
            "precondition: the seeded episodes must actually collide under the "
            "bare-range id, or this test proves nothing"
        )

        with core_db.transaction(hy.conn):
            persist_episodes(hy.conn, "s_multi", digest.episodes, granular=True)

        rows = hy.conn.execute(
            "SELECT id, title FROM episodes WHERE session_id = 's_multi' "
            "ORDER BY title"
        ).fetchall()
        assert len(rows) == 3, "each decision must persist as its own episode"
        assert all("#" in r["id"] for r in rows)
    finally:
        hy.close()


def test_granular_cap_rejects_a_runaway_reply_atomically(granular_cfg):
    """A capped reply fails as a unit instead of claiming tail coverage.

    Exercised at 20 returned against a cap of 3: silently keeping the first
    three would make output truncation indistinguishable from full extraction.
    """
    llm = _digest_llm()
    hy = HyMem(granular_cfg, llm=llm)
    try:
        _seed(hy, "s_cap")
        hy.dream()
        cids = _chunk_ids(hy, "s_cap")
        assert cids

        items = [_episode(f"Decision {i}", [cids[0]]) for i in range(20)]
        llm2 = _digest_llm(episodes=items)
        digest = extract_session_digest(
            hy.conn, "s_cap", llm2,
            max_tokens=1024, max_chars=12000, granular=True, max_episodes=3,
        )
        assert len(items) > 3  # precondition: the cap must actually bite
        assert digest.parse_failed is True
        assert digest.failure_reason == "episode_output_cap"
        assert digest.episode_input_items == 20
        assert digest.episode_rejected_items == 17
        assert digest.episodes.items == []
        assert digest.covered_message_id is None

        with core_db.transaction(hy.conn):
            persist_episodes(hy.conn, "s_cap", digest.episodes, granular=True)
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE session_id = 's_cap'"
        ).fetchone()["c"] == 0
    finally:
        hy.close()


def test_redream_upserts_granular_episodes_in_place(granular_cfg):
    """Same range + same title = same id, so a re-dream REFRESHES the row
    instead of appending a near-duplicate. Asserted at 2 rows (not 0 or 1): the
    identity has to be exercised while more than one episode shares the range,
    which is the case the discriminator introduced."""
    llm = _digest_llm()
    hy = HyMem(granular_cfg, llm=llm)
    try:
        _seed(hy, "s_upsert")
        hy.dream()
        cids = _chunk_ids(hy, "s_upsert")
        assert cids

        first = EpisodesExtraction(items=[
            _episode("Chose fly.io", [cids[0]], "Deploy target set to fly.io."),
            _episode("Hit the 512MB limit", [cids[0]], "Memory capped at 512MB."),
        ])
        with core_db.transaction(hy.conn):
            persist_episodes(hy.conn, "s_upsert", first, granular=True)
        before = hy.conn.execute(
            "SELECT id, rowid FROM episodes WHERE session_id = 's_upsert' "
            "ORDER BY id"
        ).fetchall()
        assert len(before) == 2, "precondition: two distinct rows on one range"

        second = EpisodesExtraction(items=[
            _episode("Chose fly.io", [cids[0]], "Deploy target set to fly.io "
                                                "in region ams."),
            _episode("Hit the 512MB limit", [cids[0]], "Memory capped at 512MB."),
        ])
        with core_db.transaction(hy.conn):
            persist_episodes(hy.conn, "s_upsert", second, granular=True)

        after = hy.conn.execute(
            "SELECT id, rowid, summary FROM episodes WHERE session_id = 's_upsert' "
            "ORDER BY id"
        ).fetchall()
        assert [r["id"] for r in after] == [r["id"] for r in before]
        assert [r["rowid"] for r in after] == [r["rowid"] for r in before], (
            "rowids must not churn — episodes_fts and vec_episodes key on them"
        )
        assert any("region ams" in r["summary"] for r in after), (
            "the refreshed text must actually land, or this test would pass "
            "against a persist that silently no-ops"
        )
    finally:
        hy.close()


# --- supersession of the blob rows -----------------------------------------


def test_granular_recut_supersedes_the_blob_episode_in_the_window(cfg, granular_cfg):
    """Turning granularity on must not leave BOTH granularities in the store.

    The blob row and the granular rows have different ids by construction (the
    blob spans a wider range), so UPSERT alone cannot replace it — the guard
    re-extracts and the persist supersedes the window it re-read. Asserted
    against a blob row that demonstrably existed first."""
    cids: list[str] = []
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        _seed(hy, "s_supersede")
        hy.dream()
        cids = _chunk_ids(hy, "s_supersede")
        assert cids
        # One blob episode spanning every chunk, the shape the shipping prompt
        # produces and the row this feature has to displace.
        blob = EpisodesExtraction(items=[
            _episode("Deploy work", cids, "Worked on deploys and migrations.")
        ])
        with core_db.transaction(hy.conn):
            persist_episodes(hy.conn, "s_supersede", blob, granular=False)
        blob_rows = hy.conn.execute(
            "SELECT id, start_message_id, end_message_id FROM episodes "
            "WHERE session_id = 's_supersede'"
        ).fetchall()
        assert len(blob_rows) == 1 and "#" not in blob_rows[0]["id"], (
            "precondition: a blob-shaped episode must exist before the re-cut"
        )
        window = (blob_rows[0]["start_message_id"], blob_rows[0]["end_message_id"])
    finally:
        hy.close()

    hy2 = HyMem(granular_cfg, llm=_digest_llm())
    try:
        granular = EpisodesExtraction(items=[
            _episode("Chose the staging deploy path", [cids[0]]),
            _episode("Pinned alembic to 2.1.4", [cids[-1]]),
        ])
        with core_db.transaction(hy2.conn):
            persist_episodes(hy2.conn, "s_supersede", granular,
                             granular=True, supersede_window=window)

        rows = hy2.conn.execute(
            "SELECT id FROM episodes WHERE session_id = 's_supersede'"
        ).fetchall()
        assert len(rows) == 2, "the blob row must not survive beside the re-cut"
        assert all("#" in r["id"] for r in rows)
    finally:
        hy2.close()


def test_supersede_leaves_episodes_outside_the_window_alone(granular_cfg):
    """The window is the range the extractor re-READ. An older episode below it
    was covered by an earlier call and must survive — a tail re-digest that
    deleted the session's history would be the starvation bug wearing a
    supersession label."""
    llm = _digest_llm()
    hy = HyMem(granular_cfg, llm=llm)
    try:
        _seed(hy, "s_window")
        hy.dream()
        hy.conn.execute(
            "INSERT INTO episodes(id, session_id, title, summary, "
            "start_message_id, end_message_id) VALUES "
            "('old', 's_window', 'Old episode', 'Established earlier.', 1, 2)"
        )
        hy.conn.execute(
            "INSERT INTO episodes(id, session_id, title, summary, "
            "start_message_id, end_message_id) VALUES "
            "('inside', 's_window', 'Stale episode', 'Superseded.', 3, 4)"
        )
        cids = _chunk_ids(hy, "s_window")
        assert cids

        # The re-read window covers messages 3-4 only, so 'old' (1-2) is outside
        # it and 'inside' (3-4) is inside it: the test distinguishes "deletes the
        # window" from "deletes the session".
        new = EpisodesExtraction(items=[_episode("Fresh decision", [cids[0]])])
        with core_db.transaction(hy.conn):
            persist_episodes(hy.conn, "s_window", new,
                             granular=True, supersede_window=(3, 4))

        ids = {r["id"] for r in hy.conn.execute(
            "SELECT id FROM episodes WHERE session_id = 's_window'"
        ).fetchall()}
        assert "old" in ids, "an episode below the window must survive"
        assert "inside" not in ids, "a stale episode inside the window must go"
    finally:
        hy.close()


def test_supersede_is_a_noop_when_nothing_was_written(granular_cfg):
    """An empty extraction is a legitimate 'this slice held nothing'. It must
    never be able to delete a previous extraction's work — the persist path is
    only reached with items, and the helper refuses the empty case as well."""
    llm = _digest_llm()
    hy = HyMem(granular_cfg, llm=llm)
    try:
        _seed(hy, "s_empty")
        hy.dream()
        hy.conn.execute(
            "INSERT INTO episodes(id, session_id, title, summary, "
            "start_message_id, end_message_id) VALUES "
            "('keep', 's_empty', 'Kept episode', 'Established earlier.', 1, 4)"
        )
        with core_db.transaction(hy.conn):
            persist_episodes(hy.conn, "s_empty", EpisodesExtraction(items=[]),
                             granular=True, supersede_window=(1, 4))
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE id = 'keep'"
        ).fetchone()["c"] == 1
    finally:
        hy.close()


# --- the per-session prompt-version guard ---------------------------------


def test_active_episode_prompt_version_maps_the_flag():
    """NULL for the shipping prompt is the load-bearing half: it is what every
    pre-v35 row already carries, so an untouched store never sees a mismatch."""
    assert active_episode_prompt_version(False) is None
    assert active_episode_prompt_version(True) == EPISODE_GRANULAR_PROMPT_VERSION


def test_granularity_flip_forces_one_reextraction_then_settles(cfg, granular_cfg):
    """The guard, both halves. Flipping the flag re-runs the digest on an
    already-digested session (which `cfg.prompt_version` alone would never do),
    and the dream AFTER that costs zero tail calls again."""
    llm1 = _digest_llm(summary="A short but valid session summary about deploys.")
    hy1 = HyMem(cfg, llm=llm1)
    try:
        _seed(hy1, "s_flip")
        hy1.dream()
        assert len(_calls(llm1, _BLOB_CLOSER)) == 1
        hy1.dream()
        assert len(_calls(llm1, _BLOB_CLOSER)) == 1, (
            "precondition: the OFF arm must already be in the zero-call steady "
            "state, or the re-extraction below proves nothing"
        )
    finally:
        hy1.close()

    llm2 = _digest_llm(summary="A short but valid session summary about deploys.")
    hy2 = HyMem(granular_cfg, llm=llm2)
    try:
        hy2.dream()
        assert len(_calls(llm2, _GRANULAR_CLOSER)) == 1, (
            "a granularity flip must re-extract the already-digested session"
        )
        assert hy2.conn.execute(
            "SELECT episodes_prompt_version FROM sessions WHERE id = 's_flip'"
        ).fetchone()["episodes_prompt_version"] == EPISODE_GRANULAR_PROMPT_VERSION

        hy2.dream()
        assert len(_calls(llm2, _GRANULAR_CLOSER)) == 1, (
            "the dream after the flip must cost zero tail calls again"
        )
    finally:
        hy2.close()


def test_reverting_the_flag_re_extracts_under_the_blob_prompt(cfg, granular_cfg):
    """A revert is a granularity change too. The stamp goes back to NULL and the
    session is re-read under the shipping prompt, rather than keeping granular
    episodes that nothing will refresh."""
    llm1 = _digest_llm(summary="A short but valid session summary about deploys.")
    hy1 = HyMem(granular_cfg, llm=llm1)
    try:
        _seed(hy1, "s_revert")
        hy1.dream()
        assert len(_calls(llm1, _GRANULAR_CLOSER)) == 1
        assert hy1.conn.execute(
            "SELECT episodes_prompt_version FROM sessions WHERE id = 's_revert'"
        ).fetchone()["episodes_prompt_version"] == EPISODE_GRANULAR_PROMPT_VERSION
    finally:
        hy1.close()

    llm2 = _digest_llm(summary="A short but valid session summary about deploys.")
    hy2 = HyMem(cfg, llm=llm2)
    try:
        hy2.dream()
        assert len(_calls(llm2, _BLOB_CLOSER)) == 1
        assert hy2.conn.execute(
            "SELECT episodes_prompt_version FROM sessions WHERE id = 's_revert'"
        ).fetchone()["episodes_prompt_version"] is None
    finally:
        hy2.close()


def test_granular_dream_stamps_and_persists_end_to_end(granular_cfg):
    """One full dream on the granular arm: the granular prompt is what gets
    sent, episodes land, and the stamp is written in the same transaction."""
    llm = _digest_llm()
    hy = HyMem(granular_cfg, llm=llm)
    try:
        _seed(hy, "s_e2e")
        cids = _chunk_ids(hy, "s_e2e")
        hy.set_llm(_digest_llm(
            episodes=[_episode("Pinned alembic to 2.1.4", [cids[-1]])],
            summary="Deployed to staging and pinned alembic to 2.1.4.",
        ))
        llm = hy._llm
        hy.dream()
        assert len(_calls(llm, _GRANULAR_CLOSER)) == 1
        assert _calls(llm, _BLOB_CLOSER) == []
        rows = hy.conn.execute(
            "SELECT title FROM episodes WHERE session_id = 's_e2e'"
        ).fetchall()
        assert [r["title"] for r in rows] == ["Pinned alembic to 2.1.4"]
        assert hy.conn.execute(
            "SELECT episodes_prompt_version FROM sessions WHERE id = 's_e2e'"
        ).fetchone()["episodes_prompt_version"] == EPISODE_GRANULAR_PROMPT_VERSION
    finally:
        hy.close()


# --- migration 035 ---------------------------------------------------------


def test_v35_adds_the_session_stamp_to_a_pre_v35_store(tmp_path):
    """Migration 035 against a hand-built v34 store: the column appears and
    existing rows are untouched. `CREATE TABLE IF NOT EXISTS` in schema.sql
    no-ops on the pre-existing table, so the column can only come from the
    migration."""
    conn = core_db.connect(tmp_path / "v34.sqlite")
    conn.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '34');
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            summary TEXT,
            digested_prompt_version TEXT,
            profile_prompt_version TEXT,
            digested_message_id INTEGER,
            facts_message_id INTEGER
        );
        INSERT INTO sessions(id, summary) VALUES ('old', 'a pre-v35 session');
        """
    )

    core_db.initialize(conn)

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    assert "episodes_prompt_version" in cols
    assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION == 46
    row = conn.execute("SELECT * FROM sessions WHERE id = 'old'").fetchone()
    assert row["summary"] == "a pre-v35 session"
    assert row["episodes_prompt_version"] is None, (
        "an existing session must read as the shipping prompt, not as granular"
    )
    conn.close()


# --- the revert leg, at row level ------------------------------------------


def test_reverting_supersedes_the_granular_rows_it_replaces(cfg, granular_cfg):
    """The revert leg of the flip contract, asserted on ROWS.

    ``test_reverting_the_flag_re_extracts_under_the_blob_prompt`` above pins the
    prompt and the stamp, and neither can see the failure this test exists for:
    granular ids are ``range#titlehash`` while blob ids are the bare ``range``,
    so on a flip-off the two shapes cannot collide and UPSERT alone leaves BOTH
    granularities of the same conversation standing. The store then serves a
    decision-grained row and a blob row over the same turns — the mixed store
    the supersession was built to prevent, in the direction nobody looked.

    Three passes because chunking happens INSIDE the dream: the chunk ids an
    episode has to cite do not exist until one has run. Asserted at a non-NULL
    range on purpose — a NULL-range row is deliberately never superseded
    (unattributable to any window), so citing no chunks would pass this test for
    the wrong reason.
    """
    # Pass 1 — blob arm, no episodes: this exists only to create the chunks.
    llm0 = _digest_llm(summary="A short but valid session summary about deploys.")
    hy0 = HyMem(cfg, llm=llm0)
    try:
        _seed(hy0, "s_mix")
        hy0.dream()
        cids = _chunk_ids(hy0, "s_mix")
        assert cids, "no chunks — the range-id path would be untested"
    finally:
        hy0.close()

    payload = [
        _episode("Chose fly.io over render", cids),
        _episode("Hit the 512MB memory limit", cids),
    ]

    # Pass 2 — flip ON. The NULL stamp mismatches the granular version, so the
    # session is re-read and both decisions persist as separate rows.
    llm1 = _digest_llm(episodes=payload, summary="Deploy notes for staging.")
    hy1 = HyMem(granular_cfg, llm=llm1)
    try:
        hy1.dream()
        granular_ids = [
            r["id"] for r in hy1.conn.execute(
                "SELECT id FROM episodes WHERE session_id = 's_mix'"
            ).fetchall()
        ]
        assert len(granular_ids) == 2, granular_ids
        assert all("#" in i for i in granular_ids), granular_ids
        assert all(
            r["start_message_id"] is not None
            for r in hy1.conn.execute(
                "SELECT start_message_id FROM episodes WHERE session_id = 's_mix'"
            ).fetchall()
        ), "NULL range — the supersede guard skips these, test would be vacuous"
    finally:
        hy1.close()

    # Pass 3 — flip back OFF. The blob rewrite must take the window with it.
    llm2 = _digest_llm(episodes=payload, summary="Deploy notes for staging.")
    hy2 = HyMem(cfg, llm=llm2)
    try:
        hy2.dream()
        rows = [
            r["id"] for r in hy2.conn.execute(
                "SELECT id FROM episodes WHERE session_id = 's_mix'"
            ).fetchall()
        ]
        assert not [i for i in rows if i in granular_ids], (
            f"granular rows survived the revert: {rows}"
        )
        # v38 adds a stable per-slice ordinal: both valid blob episodes survive
        # even though they cite the same message range.
        assert len(rows) == 2, rows
        assert all("#i" in episode_id for episode_id in rows), rows
    finally:
        hy2.close()


def _range(hy: HyMem, sid: str) -> str:
    row = hy.conn.execute(
        "SELECT MIN(start_message_id) AS s, MAX(end_message_id) AS e "
        "FROM chunks WHERE session_id = ?",
        (sid,),
    ).fetchone()
    return f"{row['s']}-{row['e']}"


def test_generation_cleanup_replaces_destructive_window_supersession(
    cfg, granular_cfg, monkeypatch
):
    """v38 never deletes old episodes before a replacement walk completes.

    Granularity changes now use digest generations: old rows coexist during a
    bounded/retryable walk and are retired atomically only at the coverage tail.
    The legacy per-window destructive argument must therefore remain disabled
    in every state.
    """
    import hymem.dreaming.runner as runner

    seen: list = []
    real = runner.persist_episodes

    def spy(*args, **kwargs):
        seen.append(kwargs.get("supersede_window", "MISSING"))
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "persist_episodes", spy)

    # (a) blob-only store, first dream: stamp NULL -> no window.
    llm0 = _digest_llm()
    hy0 = HyMem(cfg, llm=llm0)
    try:
        _seed(hy0, "s_wire")
        eps = [_episode("Chose fly.io over render", [_chunk_ids(hy0, "s_wire")[0]])]
        hy0.set_llm(_digest_llm(episodes=eps, summary="Deploy notes for staging."))
        hy0.dream()
    finally:
        hy0.close()
    assert seen == [None], seen

    # (b) blob-only store re-extracting under a bumped prompt version: still
    # NULL, still no window. The re-dream stays additive, as it always was.
    seen.clear()
    llm1 = _digest_llm(episodes=eps, summary="Deploy notes, take two.")
    hy1 = HyMem(dataclasses.replace(cfg, prompt_version="v9-bumped"), llm=llm1)
    try:
        hy1.dream()
    finally:
        hy1.close()
    assert seen == [None], seen

    # (c) flip ON, then (d) flip OFF again: still no eager window deletion.
    seen.clear()
    llm2 = _digest_llm(episodes=eps, summary="Deploy notes for staging.")
    hy2 = HyMem(granular_cfg, llm=llm2)
    try:
        hy2.dream()
    finally:
        hy2.close()
    assert seen == [None], seen

    seen.clear()
    llm3 = _digest_llm(episodes=eps, summary="Deploy notes for staging.")
    hy3 = HyMem(cfg, llm=llm3)
    try:
        hy3.dream()
    finally:
        hy3.close()
    assert seen == [None], seen
