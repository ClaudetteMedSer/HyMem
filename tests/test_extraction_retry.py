"""Ingest failures are HELD for retry, not marked done.

`processed_chunks` is a one-shot gate: a row means no dream will ever look at
that chunk again under the same prompt version. Before this, an unparseable or
wrong-shaped LLM reply produced an empty `ChunkResult` that persisted like a
successful extraction and took the mark with it — so a transient provider
hiccup became a permanent hole, indistinguishable in the DB from a chunk that
genuinely held nothing. (A ~48-chunk cohort of exactly this shape survived a
recovery pass because the re-extraction hit the same class of failure.)

Every other pipeline already has the right semantics: the digest holds its v24
watermark on failure, facts hold the v26 one, and a failed fusion retries every
dream until it heals. These tests pin ingest to the same rule, and pin the
boundary that makes it safe: a clean parse yielding nothing IS marked done,
because that is a real empty and re-reading it forever would burn budget.
"""

from __future__ import annotations

import json
from dataclasses import replace

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.chunk import extract_chunk
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.markers import Marker


# --- the failed flag itself -------------------------------------------------


def test_unparseable_reply_is_flagged_failed():
    result = extract_chunk(StubLLMClient(default="Sorry, no JSON for you."), "x")
    assert result.triples == [] and result.markers == []
    assert result.failed is True


def test_wrong_shape_reply_is_flagged_failed():
    llm = StubLLMClient(default=json.dumps(["not", "an", "object"]))
    assert extract_chunk(llm, "x").failed is True


def test_clean_empty_object_is_not_failed():
    """The floor: the model answered, and this chunk holds nothing."""
    llm = StubLLMClient(default=json.dumps({"triples": [], "markers": []}))
    result = extract_chunk(llm, "x")
    assert result.triples == [] and result.markers == []
    assert result.failed is False


def test_stub_empty_array_is_not_failed():
    """`[]` is StubLLMClient's documented default — the no-LLM configuration
    this project ships. Flagging it failed would hold every chunk forever."""
    assert extract_chunk(StubLLMClient(default="[]"), "x").failed is False


def test_missing_both_keys_is_audible_but_not_yet_failed(caplog):
    """Candidate third hole class, measure-only. A dict with neither key is
    never compliant, but the rate is unmeasured — so it is logged now and
    flipped to failed once the re-extraction queue's verdict is clear."""
    llm = StubLLMClient(default=json.dumps({"other": 1}))
    with caplog.at_level("WARNING"):
        result = extract_chunk(llm, "x")
    assert result.triples == [] and result.markers == []
    assert result.failed is False, "flip deferred: see chunk.py"
    assert any("chunk_extraction.missing_keys" in r.message for r in caplog.records)


def test_one_key_present_is_not_flagged(caplog):
    """A model that found triples and omitted an empty markers array is
    behaving normally — this must not trip the missing-keys signal."""
    llm = StubLLMClient(default=json.dumps({
        "triples": [{"subject": "app", "predicate": "uses", "object": "uv"}],
    }))
    with caplog.at_level("WARNING"):
        result = extract_chunk(llm, "x")
    assert [t.predicate for t in result.triples] == ["uses"]
    assert not any(
        "chunk_extraction.missing_keys" in r.message for r in caplog.records
    )


def test_clean_empty_object_is_not_flagged_missing_keys(caplog):
    """Both keys present but empty is the real floor, and must stay silent."""
    llm = StubLLMClient(default=json.dumps({"triples": [], "markers": []}))
    with caplog.at_level("WARNING"):
        extract_chunk(llm, "x")
    assert not any(
        "chunk_extraction.missing_keys" in r.message for r in caplog.records
    )


# --- the behavior that flag buys: retry vs. permanent hole ------------------


def _seed_chunk(hy: HyMem, chunk_id: str = "c_retry") -> Chunk:
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s_retry')")
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content) "
        "VALUES ('s_retry', 'user', 'msg')"
    )
    mid = cur.lastrowid
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES (?, 's_retry', ?, ?, 'long_user_turn', 'text')",
        (chunk_id, mid, mid),
    )
    return Chunk(
        id=chunk_id, session_id="s_retry", start_message_id=mid,
        end_message_id=mid, salience_reason="long_user_turn", text="text",
    )


def _persist(hy: HyMem, chunk: Chunk, extraction: ChunkExtraction) -> None:
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn, chunk, extraction,
            prompt_version=hy.config.prompt_version, cfg=hy.config,
        )


def _marked(hy: HyMem, chunk_id: str) -> bool:
    return hy.conn.execute(
        "SELECT 1 FROM processed_chunks WHERE chunk_id = ? AND prompt_version = ?",
        (chunk_id, hy.config.prompt_version),
    ).fetchone() is not None


def test_failed_extraction_is_not_marked_processed(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
        assert not _marked(hy, chunk.id)
    finally:
        hy.close()


def test_clean_empty_extraction_is_marked_processed(cfg):
    """The floor stays marked — otherwise every contentless chunk in the store
    is re-extracted on every dream, forever."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=False))
        assert _marked(hy, chunk.id)
    finally:
        hy.close()


def test_held_chunk_is_re_extracted_on_the_next_dream(cfg):
    """End to end: a chunk whose reply was unparseable is offered to the LLM
    again, and succeeds once the provider recovers."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)

        broken = StubLLMClient(default="not json at all")
        first = phase1.extract_chunk_results(
            hy.conn, chunk, broken, prompt_version=hy.config.prompt_version,
        )
        assert first is not None and first.failed is True
        _persist(hy, chunk, first)

        # Provider recovers. The chunk was held, so extraction runs again
        # rather than short-circuiting to None on a processed_chunks row.
        healthy = StubLLMClient(default=json.dumps({
            "triples": [{"subject": "app", "predicate": "uses", "object": "uv"}],
            "markers": [],
        }))
        second = phase1.extract_chunk_results(
            hy.conn, chunk, healthy, prompt_version=hy.config.prompt_version,
        )
        assert second is not None, "held chunk must be re-offered to the LLM"
        assert second.failed is False
        assert [t.predicate for t in second.triples] == ["uses"]

        _persist(hy, chunk, second)
        assert _marked(hy, chunk.id), "a healed chunk is marked done"
    finally:
        hy.close()


# --- the bound: held retries are finite (v28) -------------------------------


def _attempts(hy: HyMem, chunk_id: str) -> int:
    row = hy.conn.execute(
        "SELECT attempts FROM chunk_extraction_attempts "
        "WHERE chunk_id = ? AND prompt_version = ?",
        (chunk_id, hy.config.prompt_version),
    ).fetchone()
    return row[0] if row else 0


def test_failures_accrue_and_chunk_stays_held_below_the_bound(cfg):
    hy = HyMem(replace(cfg, chunk_extraction_max_attempts=3))
    try:
        chunk = _seed_chunk(hy)
        for expected in (1, 2):
            _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
            assert _attempts(hy, chunk.id) == expected
            assert not _marked(hy, chunk.id)
    finally:
        hy.close()


def test_chunk_is_abandoned_at_the_bound(cfg, caplog):
    """The budget tax has to terminate: the runner spends a slot on a chunk
    before knowing it will fail, so an unbounded hold starves new chunks."""
    hy = HyMem(replace(cfg, chunk_extraction_max_attempts=2))
    try:
        chunk = _seed_chunk(hy)
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
        assert not _marked(hy, chunk.id)

        with caplog.at_level("WARNING"):
            _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
        assert _marked(hy, chunk.id), "must stop consuming a budget slot"
        assert any(
            "phase1.extraction_abandoned" in r.message for r in caplog.records
        ), "giving up is a real content loss and must be audible"
    finally:
        hy.close()


def test_success_clears_the_attempt_count(cfg):
    """Consecutive failures, not lifetime — a chunk that heals starts fresh."""
    hy = HyMem(replace(cfg, chunk_extraction_max_attempts=3))
    try:
        chunk = _seed_chunk(hy)
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
        assert _attempts(hy, chunk.id) == 1
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=False))
        assert _attempts(hy, chunk.id) == 0
        assert _marked(hy, chunk.id)
    finally:
        hy.close()


def test_zero_max_attempts_retries_forever(cfg):
    hy = HyMem(replace(cfg, chunk_extraction_max_attempts=0))
    try:
        chunk = _seed_chunk(hy)
        for _ in range(5):
            _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
        assert _attempts(hy, chunk.id) == 5
        assert not _marked(hy, chunk.id)
    finally:
        hy.close()


# --- marker write idempotence (what makes OR-semantics safe) ----------------


def test_re_extracting_a_chunk_does_not_duplicate_its_markers(cfg):
    """kg_evidence has UNIQUE(edge_id, chunk_id, polarity); markers had no
    equivalent, and that asymmetry is the only reason a split-merge needed
    AND-semantics (one good half marks the whole chunk done, silently
    discarding the failed half). With the write idempotent, re-extracting the
    good half is free and the content-losing tradeoff is unnecessary."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        marker = Marker(kind="preference", statement="prefers short answers")
        for _ in range(3):
            _persist(hy, chunk, ChunkExtraction(triples=[], markers=[marker]))
        count = hy.conn.execute(
            "SELECT COUNT(*) FROM behavioral_markers WHERE chunk_id = ?",
            (chunk.id,),
        ).fetchone()[0]
        assert count == 1
    finally:
        hy.close()


def test_distinct_markers_on_one_chunk_all_persist(cfg):
    """The guard keys on (chunk_id, kind, statement) — it must not collapse
    genuinely different markers that happen to share a chunk."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[
            Marker(kind="preference", statement="prefers short answers"),
            Marker(kind="preference", statement="prefers python"),
            Marker(kind="rejection", statement="prefers short answers"),
        ]))
        count = hy.conn.execute(
            "SELECT COUNT(*) FROM behavioral_markers WHERE chunk_id = ?",
            (chunk.id,),
        ).fetchone()[0]
        assert count == 3
    finally:
        hy.close()


def test_marked_chunk_is_not_re_extracted(cfg):
    """The one-shot gate still works for successful extractions."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=False))
        again = phase1.extract_chunk_results(
            hy.conn, chunk, StubLLMClient(default="[]"),
            prompt_version=hy.config.prompt_version,
        )
        assert again is None
    finally:
        hy.close()
