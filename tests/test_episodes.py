"""Tests for the Episodic Memory Layer enhancements:

1. ``persist_episodes`` populates ``participants``, ``start_message_id``,
   ``end_message_id`` from the chunks the LLM grouped.
2. Stable episode ids (``{session}@{start}-{end}``) + UPSERT so a re-dream
   updates content instead of dropping new episodes via INSERT OR IGNORE.
3. Semantic episode search via ``vec_episodes`` + RRF fusion with FTS.
"""

from __future__ import annotations

import json

from hymem import HyMem, StubEmbeddingClient
from hymem.dreaming.episodes import (
    EpisodesExtraction,
    extract_episodes_for_session,
    persist_episodes,
)
from hymem.extraction.llm import StubLLMClient


# --- shared helpers --------------------------------------------------------


def _episode_llm_response(items: list[dict]) -> StubLLMClient:
    """Stub that returns the given items as the episodes section of the batched
    session-digest response (summary/procedures empty), and an empty array for
    everything else (so triples/markers no-op). Keyed on the unique digest
    user-prompt closer ``Return the JSON object now``."""
    digest = {"episodes": items, "summary": "", "procedures": []}
    return StubLLMClient(
        fixtures={"Return the JSON object now": json.dumps(digest)},
        default="[]",
    )


def _force_redigest(hy: HyMem, sid: str) -> None:
    """Clear the per-session digest marker so the next ``dream()`` re-runs the
    batched digest. The runner otherwise skips the digest for a session already
    digested under the current prompt_version with no newly-extracted chunks;
    these tests materialize chunks in a first dream, then re-dream to drive the
    episode-returning stub, which is exactly that skip case."""
    with hy.conn:
        hy.conn.execute(
            "UPDATE sessions SET digested_prompt_version = NULL WHERE id = ?", (sid,)
        )


def _seed_session_with_chunks(hy: HyMem, sid: str, turns: list[tuple[str, str]]) -> None:
    """Open ``sid``, log every (role, content) turn, close. Each user turn
    long enough to clear ``salience_min_chars`` becomes its own chunk."""
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)


# --- Enhancement 1: populated participants / start / end ------------------


def test_persist_episodes_populates_message_range_and_participants(cfg):
    """When the LLM returns chunk_ids, persist_episodes must look up
    start_message_id (min) and end_message_id (max) from chunks, and derive
    participants from the message roles in that range."""
    sid = "s_range"
    turns = [
        ("assistant", "Let's discuss postgres connection pooling."),
        ("user", "We hit pool exhaustion on the prod cluster last night, what should we tune?"),
        ("assistant", "Bump pool_size and connect_timeout. Anything else to address?"),
        ("user", "Also let's look at the slow query log, it had a few 30 second outliers we should triage."),
    ]
    chunk_ids: list[str] = []

    def llm_factory():
        # Return one episode covering every emitted chunk, as the episodes
        # section of the batched session-digest response.
        return StubLLMClient(
            fixtures={"Return the JSON object now": json.dumps({
                "episodes": [
                    {
                        "title": "Postgres pool tuning",
                        "summary": "Diagnosed pool exhaustion and decided to bump pool_size plus inspect the slow query log.",
                        "outcome": "resolved",
                        "key_entities": ["postgres", "pool_size", "slow_query_log"],
                        "chunk_ids": chunk_ids,  # filled in below
                    }
                ],
                "summary": "",
                "procedures": [],
            })}, default="[]",
        )

    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    try:
        _seed_session_with_chunks(hy, sid, turns)
        # Dream once with a stub that returns no episodes, just so chunks
        # are written. Then collect the chunk ids and dream again with the
        # episode-returning stub.
        hy.dream()
        rows = hy.conn.execute(
            "SELECT id FROM chunks WHERE session_id = ? ORDER BY start_message_id",
            (sid,),
        ).fetchall()
        assert rows, "expected at least one chunk to be persisted"
        chunk_ids[:] = [r["id"] for r in rows]
        hy.set_llm(llm_factory())
        _force_redigest(hy, sid)
        hy.dream()

        ep = hy.conn.execute(
            "SELECT session_id, title, summary, participants, "
            "start_message_id, end_message_id, outcome, key_entities "
            "FROM episodes WHERE session_id = ?",
            (sid,),
        ).fetchone()
        assert ep is not None
        assert ep["title"] == "Postgres pool tuning"
        assert ep["outcome"] == "resolved"

        # Message range: smallest start_message_id, largest end_message_id
        # across the named chunks.
        chunk_range = hy.conn.execute(
            f"SELECT MIN(start_message_id) AS s, MAX(end_message_id) AS e "
            f"FROM chunks WHERE id IN ({','.join('?' * len(chunk_ids))})",
            tuple(chunk_ids),
        ).fetchone()
        assert ep["start_message_id"] == chunk_range["s"]
        assert ep["end_message_id"] == chunk_range["e"]

        # Participants: the two roles seen in the message range.
        participants = json.loads(ep["participants"])
        assert sorted(participants) == ["assistant", "user"]
        # Key entities preserved.
        assert "postgres" in json.loads(ep["key_entities"])
    finally:
        hy.close()


def test_persist_episodes_drops_hallucinated_chunk_ids(cfg, stub_llm):
    """If the LLM returns a chunk_id not in the input, it must be filtered
    out so message-range lookup isn't poisoned."""
    sid = "s_halluc"
    hy = HyMem(cfg, llm=stub_llm)
    try:
        _seed_session_with_chunks(hy, sid, [
            ("assistant", "anything"),
            ("user", "Long enough user turn to clear the salience minimum threshold here."),
        ])
        # Persist chunks via a no-episode dream.
        hy.dream()
        real_ids = [
            r["id"]
            for r in hy.conn.execute(
                "SELECT id FROM chunks WHERE session_id = ?", (sid,)
            ).fetchall()
        ]
        assert real_ids

        # Now run extract directly with a stub that returns both a real id
        # and a hallucinated one.
        hy.set_llm(StubLLMClient(
            fixtures={"identify distinct episodes": json.dumps([
                {
                    "title": "Mixed episode",
                    "summary": "Has one real chunk and one hallucinated chunk id.",
                    "outcome": "informational",
                    "key_entities": [],
                    "chunk_ids": real_ids + ["chk_hallucinated_does_not_exist"],
                }
            ])}, default="[]",
        ))
        extraction = extract_episodes_for_session(hy.conn, sid, hy._llm)
        assert extraction is not None
        assert len(extraction.items) == 1
        assert extraction.items[0]["chunk_ids"] == real_ids
    finally:
        hy.close()


# --- Enhancement 2: stable id + UPSERT survives re-dreams -----------------


def test_episode_id_is_stable_across_redreams(cfg):
    """Same message range → same id → UPSERT updates title/summary in place,
    so re-dreaming with new content for the same range doesn't drop or
    duplicate rows."""
    sid = "s_redream"
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    try:
        _seed_session_with_chunks(hy, sid, [
            ("assistant", "what database should we pick"),
            ("user", "I want a reliable database for production workloads with good replication."),
        ])
        hy.dream()
        chunk_ids = [
            r["id"]
            for r in hy.conn.execute(
                "SELECT id FROM chunks WHERE session_id = ?", (sid,)
            ).fetchall()
        ]

        # Dream 1: one episode titled "Database choice".
        hy.set_llm(_episode_llm_response([
            {
                "title": "Database choice",
                "summary": "Picked postgres for reliability.",
                "outcome": "resolved",
                "key_entities": ["postgres"],
                "chunk_ids": chunk_ids,
            }
        ]))
        _force_redigest(hy, sid)
        hy.dream()
        rows1 = hy.conn.execute(
            "SELECT id, title FROM episodes WHERE session_id = ?", (sid,)
        ).fetchall()
        assert len(rows1) == 1
        first_id = rows1[0]["id"]
        assert rows1[0]["title"] == "Database choice"

        # Dream 2: re-emit the same episode (same chunk range) but rewrite
        # title and summary. UPSERT keeps the row count at 1 and updates
        # the content rather than ignoring the new fields.
        hy.set_llm(_episode_llm_response([
            {
                "title": "Postgres selection rationale",
                "summary": "Refined: chose postgres because of its replication story.",
                "outcome": "resolved",
                "key_entities": ["postgres", "replication"],
                "chunk_ids": chunk_ids,
            }
        ]))
        _force_redigest(hy, sid)
        hy.dream()
        rows2 = hy.conn.execute(
            "SELECT id, title, summary FROM episodes WHERE session_id = ?", (sid,)
        ).fetchall()
        assert len(rows2) == 1
        assert rows2[0]["id"] == first_id  # rowid-stable for FTS / vec joins
        assert rows2[0]["title"] == "Postgres selection rationale"
        assert "replication" in rows2[0]["summary"]

        # FTS picks up the new content via the UPDATE trigger.
        ftshit = hy.conn.execute(
            "SELECT title FROM episodes WHERE rowid IN ("
            "  SELECT rowid FROM episodes_fts WHERE episodes_fts MATCH 'replication'"
            ")"
        ).fetchone()
        assert ftshit is not None
        assert ftshit["title"] == "Postgres selection rationale"
    finally:
        hy.close()


# --- Enhancement 3: semantic episode search -------------------------------


def test_dream_populates_episode_embeddings_and_vec(cfg):
    """After dreaming, episode_embeddings has a row per episode and
    vec_episodes mirrors them."""
    sid = "s_embed"
    embed = StubEmbeddingClient()
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"), embedding_client=embed)
    try:
        _seed_session_with_chunks(hy, sid, [
            ("assistant", "ok"),
            ("user", "We migrated the auth service to use OAuth2 with refresh tokens this morning."),
        ])
        hy.dream()
        chunk_ids = [
            r["id"]
            for r in hy.conn.execute(
                "SELECT id FROM chunks WHERE session_id = ?", (sid,)
            ).fetchall()
        ]
        hy.set_llm(_episode_llm_response([
            {
                "title": "Auth migration",
                "summary": "Moved auth service to OAuth2 with refresh tokens.",
                "outcome": "resolved",
                "key_entities": ["oauth2", "auth"],
                "chunk_ids": chunk_ids,
            }
        ]))
        _force_redigest(hy, sid)
        report = hy.dream()

        assert report.episodes_embedded >= 1
        ee_rows = hy.conn.execute(
            "SELECT episode_id, model, dim FROM episode_embeddings"
        ).fetchall()
        assert len(ee_rows) == 1
        assert ee_rows[0]["model"] == "stub"
        assert ee_rows[0]["dim"] == 16
        vec_count = hy.conn.execute(
            "SELECT COUNT(*) AS c FROM vec_episodes"
        ).fetchone()["c"]
        assert vec_count == 1
    finally:
        hy.close()


def test_episode_embedding_refreshes_after_upsert(cfg):
    """If a re-dream rewrites title/summary for the same episode id, the
    text_hash mismatches and the next embedding pass re-embeds."""
    sid = "s_refresh"
    embed = StubEmbeddingClient()
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"), embedding_client=embed)
    try:
        _seed_session_with_chunks(hy, sid, [
            ("assistant", "ok"),
            ("user", "We picked dgraph for the social-graph store after benchmarking neo4j."),
        ])
        hy.dream()
        chunk_ids = [
            r["id"]
            for r in hy.conn.execute(
                "SELECT id FROM chunks WHERE session_id = ?", (sid,)
            ).fetchall()
        ]

        hy.set_llm(_episode_llm_response([
            {"title": "Graph store pick", "summary": "Picked dgraph over neo4j.",
             "outcome": "resolved", "key_entities": ["dgraph"], "chunk_ids": chunk_ids},
        ]))
        _force_redigest(hy, sid)
        hy.dream()
        first_hash = hy.conn.execute(
            "SELECT text_hash FROM episode_embeddings"
        ).fetchone()["text_hash"]

        # Rewrite the episode content; same id, different text_hash.
        hy.set_llm(_episode_llm_response([
            {"title": "Graph store pick", "summary": "Picked dgraph for its sharding story.",
             "outcome": "resolved", "key_entities": ["dgraph", "sharding"], "chunk_ids": chunk_ids},
        ]))
        _force_redigest(hy, sid)
        report = hy.dream()
        assert report.episodes_embedded >= 1

        new_hash = hy.conn.execute(
            "SELECT text_hash FROM episode_embeddings"
        ).fetchone()["text_hash"]
        assert new_hash != first_hash
    finally:
        hy.close()


def test_episode_embedding_idempotent_when_unchanged(cfg):
    """Two dreams in a row with identical episode content → second cycle
    embeds nothing extra."""
    sid = "s_idem"
    embed = StubEmbeddingClient()
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"), embedding_client=embed)
    try:
        _seed_session_with_chunks(hy, sid, [
            ("assistant", "ok"),
            ("user", "We benchmarked redis cluster vs dragonfly for caching warm reads."),
        ])
        hy.dream()
        chunk_ids = [
            r["id"]
            for r in hy.conn.execute(
                "SELECT id FROM chunks WHERE session_id = ?", (sid,)
            ).fetchall()
        ]
        ep_payload = [{
            "title": "Cache benchmark",
            "summary": "Compared redis cluster against dragonfly.",
            "outcome": "informational",
            "key_entities": ["redis", "dragonfly"],
            "chunk_ids": chunk_ids,
        }]
        hy.set_llm(_episode_llm_response(ep_payload))
        _force_redigest(hy, sid)
        report1 = hy.dream()
        assert report1.episodes_embedded == 1

        # Same content → fetch_episode_embeddings sees matching text_hash
        # for the only episode → returns None → no embed. We force the digest
        # to re-run so this exercises embedding idempotency (matching text_hash),
        # not just the per-session digest skip-guard.
        hy.set_llm(_episode_llm_response(ep_payload))
        _force_redigest(hy, sid)
        report2 = hy.dream()
        assert report2.episodes_embedded == 0
    finally:
        hy.close()


def test_episode_semantic_search_surfaces_topic_match(cfg):
    """Semantic episode search should rank an episode whose title/summary
    exactly matches the query above unrelated FTS noise.

    Uses the StubEmbeddingClient (cosine == 1.0 for identical text). With
    one episode whose embed_text exactly matches the query, vec_search
    returns it with score 1/(1+0) = 1.0 and it must appear in
    ``ctx.episodes`` after the RRF fuse.
    """
    embed = StubEmbeddingClient()
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"), embedding_client=embed)
    try:
        sid = "s_sem"
        _seed_session_with_chunks(hy, sid, [
            ("assistant", "ok"),
            ("user", "We decided to migrate the metrics pipeline from statsd to opentelemetry collectors."),
        ])
        hy.dream()
        chunk_ids = [
            r["id"]
            for r in hy.conn.execute(
                "SELECT id FROM chunks WHERE session_id = ?", (sid,)
            ).fetchall()
        ]
        title = "Metrics pipeline migration"
        summary = "Switched from statsd to opentelemetry collectors for the metrics pipeline."
        hy.set_llm(_episode_llm_response([
            {"title": title, "summary": summary,
             "outcome": "resolved", "key_entities": ["statsd", "opentelemetry"],
             "chunk_ids": chunk_ids},
        ]))
        _force_redigest(hy, sid)
        hy.dream()

        # Query exactly matches embed_text (title + "\n" + summary).
        query = f"{title}\n{summary}"
        ctx = hy.augment(query)
        assert ctx.episodes, "expected at least one episode hit"
        top = ctx.episodes[0]
        assert top.title == title
        # If FTS also matched, score_kind="rrf"; otherwise "vec".
        assert top.score_kind in {"rrf", "vec"}
    finally:
        hy.close()


# --- fenced replies (dream 1013) -------------------------------------------


_EPISODE_TURNS = [
    ("assistant", "anything"),
    ("user", "Long enough user turn to clear the salience minimum threshold here."),
]


def _hy_with_chunks(cfg, sid: str) -> HyMem:
    """HyMem whose `sid` has real chunks, ready for a direct extract call."""
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    _seed_session_with_chunks(hy, sid, _EPISODE_TURNS)
    hy.dream()
    return hy


def test_extract_episodes_parses_fenced_reply(cfg):
    """The episode call sets response_format="json"; dream 1013 proved a
    provider will fence the reply anyway. That used to be a silent, permanent
    drop — the one-shot ingest path has nothing that retries it."""
    sid = "s_fenced_ep"
    hy = _hy_with_chunks(cfg, sid)
    try:
        fenced = (
            "Here is the JSON:\n```json\n"
            + json.dumps([{
                "title": "Salience threshold chat",
                "summary": "The user said something long enough to be chunked.",
                "outcome": "informational",
                "chunk_ids": [],
            }])
            + "\n```"
        )
        hy.set_llm(StubLLMClient(
            fixtures={"identify distinct episodes": fenced}, default="[]",
        ))
        extraction = extract_episodes_for_session(hy.conn, sid, hy._llm)
        assert extraction is not None
        assert [i["title"] for i in extraction.items] == ["Salience threshold chat"]
    finally:
        hy.close()


def test_extract_episodes_refusal_yields_empty_extraction(cfg, caplog):
    """An unparseable reply keeps the documented empty EpisodesExtraction —
    leniency must not fabricate episodes out of a refusal — but the drop is
    now audible instead of silent."""
    sid = "s_refusal_ep"
    hy = _hy_with_chunks(cfg, sid)
    try:
        hy.set_llm(StubLLMClient(
            fixtures={"identify distinct episodes": "I'm sorry, I can't help."},
            default="[]",
        ))
        with caplog.at_level("WARNING"):
            extraction = extract_episodes_for_session(hy.conn, sid, hy._llm)
        assert extraction == EpisodesExtraction()
        assert any(
            "episodes.parse_failure" in r.message and sid in r.getMessage()
            for r in caplog.records
        )
    finally:
        hy.close()


def test_extract_episodes_wrong_shape_yields_empty_extraction(cfg, caplog):
    """validate_episode_items() returns [] for a non-list, so the behavior was
    already right — but an empty extraction reads as "this session held no
    episodes", which is exactly what a dropped reply must not look like."""
    sid = "s_shape_ep"
    hy = _hy_with_chunks(cfg, sid)
    try:
        hy.set_llm(StubLLMClient(
            fixtures={"identify distinct episodes": '{"episodes": "none"}'},
            default="[]",
        ))
        with caplog.at_level("WARNING"):
            extraction = extract_episodes_for_session(hy.conn, sid, hy._llm)
        assert extraction == EpisodesExtraction()
        assert any(
            "episodes.shape_failure" in r.message and sid in r.getMessage()
            for r in caplog.records
        )
    finally:
        hy.close()
