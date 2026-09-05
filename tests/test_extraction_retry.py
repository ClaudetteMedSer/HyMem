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
import sqlite3
from dataclasses import replace

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming import runner as dreaming_runner
from hymem.dreaming.chunks import (
    Chunk,
    chunk_extraction_is_quarantined,
    load_pending_persisted_chunks,
    persist_chunks,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.chunk import extract_chunk
from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.extraction.markers import Marker
from hymem.extraction.triples import Triple


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


def test_bare_empty_array_is_failed():
    """Only the requested two-key object can authorize the one-shot mark."""
    assert extract_chunk(StubLLMClient(default="[]"), "x").failed is True


def test_missing_both_keys_is_audible_and_failed(caplog):
    llm = StubLLMClient(default=json.dumps({"other": 1}))
    with caplog.at_level("WARNING"):
        result = extract_chunk(llm, "x")
    assert result.triples == [] and result.markers == []
    assert result.failed is True
    assert any("chunk_extraction.missing_keys" in r.message for r in caplog.records)


def test_one_key_present_fails_without_partial_output(caplog):
    llm = StubLLMClient(default=json.dumps({
        "triples": [{"subject": "app", "predicate": "uses", "object": "uv"}],
    }))
    with caplog.at_level("WARNING"):
        result = extract_chunk(llm, "x")
    assert result.failed is True
    assert result.triples == []
    assert any("chunk_extraction.missing_keys" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "payload",
    [
        {"triples": None, "markers": []},
        {"triples": [], "markers": {}},
        {"triples": "[]", "markers": []},
        {"triples": [], "markers": "[]"},
    ],
)
def test_wrong_typed_arrays_fail_atomically(payload, caplog):
    with caplog.at_level("WARNING"):
        result = extract_chunk(StubLLMClient(default=json.dumps(payload)), "x")
    assert result.failed is True
    assert result.triples == [] and result.markers == []
    assert "chunk_extraction.array_shape_failure" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {
            "triples": [
                {"subject": "app", "predicate": "uses", "object": "uv"},
                {"subject": "broken"},
            ],
            "markers": [],
        },
        {
            "triples": [],
            "markers": [
                {"kind": "preference", "statement": "prefers uv"},
                {"kind": "not-a-marker"},
            ],
        },
        {
            "triples": [{
                "subject": "app",
                "predicate": "uses",
                "object": "uv",
                "polarity": True,
            }],
            "markers": [],
        },
        {
            "triples": [{
                "subject": "latency",
                "predicate": "has_value",
                "object": "low",
                "value_numeric": True,
            }],
            "markers": [],
        },
    ],
)
def test_mixed_valid_and_invalid_members_fail_without_partial_output(payload, caplog):
    with caplog.at_level("WARNING"):
        result = extract_chunk(StubLLMClient(default=json.dumps(payload)), "tiny")
    assert result.failed is True
    assert result.triples == [] and result.markers == []
    assert "chunk_extraction.item_validation_failure" in caplog.text


def test_clean_empty_object_is_not_flagged_missing_keys(caplog):
    """Both keys present but empty is the real floor, and must stay silent."""
    llm = StubLLMClient(default=json.dumps({"triples": [], "markers": []}))
    with caplog.at_level("WARNING"):
        extract_chunk(llm, "x")
    assert not any(
        "chunk_extraction.missing_keys" in r.message for r in caplog.records
    )


@pytest.mark.parametrize("mutation", [
    "extra_top_key",
    "unknown_triple_key",
    "invalid_type_hint",
    "bad_properties",
    "marker_extra",
    "nonfinite",
    "huge_integer",
])
def test_combined_contract_rejects_malformed_optional_shapes(mutation):
    triple = {
        "subject": "app", "predicate": "uses", "object": "uv", "polarity": 1,
    }
    marker = {"kind": "preference", "statement": "prefers uv"}
    payload = {"triples": [triple], "markers": [marker]}
    if mutation == "extra_top_key":
        payload["error"] = "refused"
    elif mutation == "unknown_triple_key":
        triple["ignored"] = "must not be ignored"
    elif mutation == "invalid_type_hint":
        triple["subject_type"] = "imaginary_type"
    elif mutation == "bad_properties":
        triple["subject_properties"] = {"owner": 7}
    elif mutation == "marker_extra":
        marker["explanation"] = "extra"
    elif mutation == "nonfinite":
        triple["value_numeric"] = float("nan")
    elif mutation == "huge_integer":
        triple["value_numeric"] = 10**400
    result = extract_chunk(StubLLMClient(default=json.dumps(payload)), "tiny")
    assert result.failed is True
    assert result.triples == [] and result.markers == []


@pytest.mark.parametrize("entity_type", [
    "place", "organization", "product", "vehicle", "activity", "event",
    "document", "or_other_entity",
])
def test_combined_contract_accepts_every_personal_life_type_hint(entity_type):
    payload = {
        "triples": [{
            "subject": "thing", "predicate": "uses", "object": "item",
            "polarity": 1, "subject_type": entity_type,
        }],
        "markers": [],
    }
    result = extract_chunk(StubLLMClient(default=json.dumps(payload)), "tiny")
    assert result.failed is False
    assert result.entity_type_hints == {"thing": entity_type}


class _BoundarySplitLLM:
    def __init__(self):
        self.full_calls = 0

    def complete(self, request):
        # The original/re-roll include both sentinels and are structurally cut;
        # split windows can only recover the fact when they overlap midpoint.
        if "LEFT_EDGE" in request.user and "RIGHT_EDGE" in request.user:
            self.full_calls += 1
            return '{"triples": ['
        if "uses PostgreSQL" in request.user:
            return json.dumps({
                "triples": [{
                    "subject": "app", "predicate": "uses",
                    "object": "PostgreSQL", "polarity": 1,
                }],
                "markers": [],
            })
        return '{"triples": [], "markers": []}'


def test_split_fallback_overlap_preserves_midpoint_fact():
    text = "LEFT_EDGE" + ("x" * 100) + " app uses PostgreSQL " + ("y" * 100) + "RIGHT_EDGE"
    result = extract_chunk(_BoundarySplitLLM(), text)
    assert result.failed is False
    assert any(t.object == "PostgreSQL" for t in result.triples)


class _MetadataSplitLLM:
    def __init__(self, *, conflict: bool = False):
        self.conflict = conflict

    def complete(self, request):
        if "LEFT_EDGE" in request.user and "RIGHT_EDGE" in request.user:
            return '{"triples": ['
        if self.conflict:
            props = {"owner": "alpha" if "LEFT_EDGE" in request.user else "beta"}
        else:
            props = {"left": "one"} if "LEFT_EDGE" in request.user else {"right": "two"}
        return json.dumps({
            "triples": [{
                "subject": "app", "predicate": "uses", "object": "db",
                "polarity": 1, "subject_properties": props,
            }],
            "markers": [],
        })


def test_split_overlap_deep_merges_nonconflicting_entity_properties():
    text = "LEFT_EDGE" + ("x" * 220) + "RIGHT_EDGE"
    result = extract_chunk(_MetadataSplitLLM(), text)
    assert result.failed is False
    assert result.entity_property_hints["app"] == {"left": "one", "right": "two"}


def test_split_overlap_rejects_conflicting_entity_properties():
    text = "LEFT_EDGE" + ("x" * 220) + "RIGHT_EDGE"
    result = extract_chunk(_MetadataSplitLLM(conflict=True), text)
    assert result.failed is True


# --- the behavior that flag buys: retry vs. permanent hole ------------------


def _seed_chunk(hy: HyMem, chunk_id: str = "c_retry") -> Chunk:
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s_retry')")
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content) "
        "VALUES ('s_retry', 'user', 'msg')"
    )
    mid = int(cur.lastrowid)
    chunk = Chunk(
        id=chunk_id, session_id="s_retry", start_message_id=mid,
        end_message_id=mid, salience_reason="long_user_turn", text="user: msg",
        source_message_ids=(mid,),
    )
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "s_retry")
        persist_chunks(conn, [chunk])
    return chunk


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
            "triples": [{
                "subject": "app", "predicate": "uses", "object": "uv",
                "polarity": 1, "source_message_id": chunk.start_message_id,
            }],
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


@pytest.mark.parametrize("reverse", [False, True])
def test_phase1_alias_collapsed_polarity_conflict_is_held(cfg, reverse):
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy, f"alias-conflict-{int(reverse)}")
        hy.register_alias("PostgreSQL", "postgres")
        items = [
            {
                "subject": "app", "predicate": "uses",
                "object": "Postgres", "polarity": 1,
                "source_message_id": chunk.start_message_id,
            },
            {
                "subject": "app", "predicate": "uses",
                "object": "PostgreSQL", "polarity": -1,
                "source_message_id": chunk.start_message_id,
            },
        ]
        if reverse:
            items.reverse()
        result = phase1.extract_chunk_results(
            hy.conn,
            chunk,
            StubLLMClient(default=json.dumps({"triples": items, "markers": []})),
            prompt_version=hy.config.prompt_version,
        )
        assert result is not None and result.failed is True
        _persist(hy, chunk, result)
        assert not _marked(hy, chunk.id)
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph"
        ).fetchone()[0] == 0
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


def test_chunk_is_quarantined_unprocessed_at_the_bound(cfg, caplog):
    """The budget tax terminates without lying that extraction succeeded."""
    hy = HyMem(replace(cfg, chunk_extraction_max_attempts=2))
    try:
        chunk = _seed_chunk(hy)
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
        assert not _marked(hy, chunk.id)

        with caplog.at_level("WARNING"):
            _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
        assert not _marked(hy, chunk.id), "a failure must never become success"
        assert chunk_extraction_is_quarantined(
            hy.conn,
            chunk.id,
            prompt_version=hy.config.prompt_version,
            max_attempts=2,
        )
        assert any(
            "phase1.extraction_quarantined" in r.message for r in caplog.records
        ), "quarantine is unresolved content loss and must be audible"
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


# --- the bound must be reachable from the RUNNER, not just from persist ----


def _seed_dreamable_session(hy: HyMem, sid: str = "s_runner") -> str:
    hy.open_session(sid)
    hy.log_message(hy_sid := sid, "assistant",
                   "I'll set up Docker for the local dev environment.")
    hy.log_message(hy_sid, "user",
                   "No, we don't use Docker for local dev anymore. We switched "
                   "to uv and system Python, and I'd rather keep it that way.")
    hy.close_session(sid)
    return sid


def test_runner_accrues_attempts_and_quarantines_at_the_bound(cfg, caplog):
    """Regression: the bound lived in persist_chunk_results while the runner
    short-circuited BEFORE persist on extraction.failed, so attempts never
    accrued in production — every held chunk re-attempted forever, which is the
    unbounded budget tax v28 exists to stop. The persist-level tests passed the
    whole time because they call persist directly. This one drives the real
    dream loop, which is the only path that can catch it.
    """
    hy = HyMem(replace(cfg, chunk_extraction_max_attempts=2))
    try:
        _seed_dreamable_session(hy)
        hy.set_llm(StubLLMClient(default="not json at all"))

        hy.dream()
        rows = hy.conn.execute(
            "SELECT chunk_id, attempts FROM chunk_extraction_attempts"
        ).fetchall()
        assert rows, "a failed extraction must record an attempt via the runner"
        assert all(r["attempts"] == 1 for r in rows)
        assert not hy.conn.execute("SELECT 1 FROM processed_chunks").fetchone()

        with caplog.at_level("WARNING"):
            hy.dream()
        assert any(
            "phase1.extraction_quarantined" in r.message for r in caplog.records
        ), "the bound must fire from the runner path"
        assert not hy.conn.execute("SELECT 1 FROM processed_chunks").fetchone(), \
            "quarantine must remain distinguishable from processed success"
        attempts_at_bound = {
            row["chunk_id"]: row["attempts"]
            for row in hy.conn.execute(
                "SELECT chunk_id, attempts FROM chunk_extraction_attempts"
            )
        }
        hy.dream()
        attempts_after = {
            row["chunk_id"]: row["attempts"]
            for row in hy.conn.execute(
                "SELECT chunk_id, attempts FROM chunk_extraction_attempts"
            )
        }
        for chunk_id, attempts in attempts_at_bound.items():
            assert attempts_after[chunk_id] == attempts
        assert hy.dream_status()["quarantined_chunks"] >= len(attempts_at_bound)
    finally:
        hy.close()


def test_runner_holds_a_failed_chunk_without_marking_it(cfg):
    """The other half of the same path: below the bound, nothing is marked and
    the chunk stays eligible for the next dream."""
    hy = HyMem(replace(cfg, chunk_extraction_max_attempts=5))
    try:
        _seed_dreamable_session(hy)
        hy.set_llm(StubLLMClient(default="not json at all"))
        for expected in (1, 2, 3):
            hy.dream()
            attempts = hy.conn.execute(
                "SELECT MAX(attempts) FROM chunk_extraction_attempts"
            ).fetchone()[0]
            assert attempts == expected
            assert not hy.conn.execute("SELECT 1 FROM processed_chunks").fetchone()
    finally:
        hy.close()


def test_runner_does_not_salvage_empty_contract_from_refusal_prose(cfg):
    """A valid-looking empty embedded in prose cannot burn the one-shot gate."""
    quiet = replace(
        cfg,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
        chunk_extraction_max_attempts=5,
    )
    hy = HyMem(quiet, llm=StubLLMClient(
        default='I cannot comply. {"triples": [], "markers": []}'
    ))
    try:
        _seed_dreamable_session(hy, "phase1-refusal-object")
        report = hy.dream()
        assert report.chunk_extraction_failures > 0
        assert report.chunks_processed == 0
        assert not hy.conn.execute("SELECT 1 FROM processed_chunks").fetchone()
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


class _SplitPartialLLM:
    """Whole input fails; one terminating half succeeds and one fails."""

    def __init__(self):
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if "single pass" not in request.system:
            return json.dumps({"episodes": [], "summary": "", "procedures": []})
        has_left = "LEFTTOKEN" in request.user
        has_right = "RIGHTTOKEN" in request.user
        if has_left and has_right:
            return "[]"
        if has_left:
            return json.dumps({
                "triples": [
                    {"subject": "app", "predicate": "uses", "object": "uv"}
                ],
                "markers": [
                    {"kind": "preference", "statement": "prefers uv"}
                ],
            })
        return '{"triples": [broken]'


def test_runner_drops_failed_split_partial_without_embed_or_counts(
    cfg, monkeypatch,
):
    llm = _SplitPartialLLM()
    quiet = replace(
        cfg,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
        salience_min_chars=1,
        chunk_extraction_max_attempts=2,
    )
    hy = HyMem(quiet, llm=llm)
    dedup_calls: list[object] = []
    monkeypatch.setattr(
        dreaming_runner,
        "_prepare_dedup_vectors",
        lambda *args, **kwargs: dedup_calls.append((args, kwargs)) or {},
    )
    try:
        content = "LEFTTOKEN " + ("L" * 180) + ("R" * 180) + " RIGHTTOKEN"
        hy.log_message("split-partial", "user", content)
        hy.close_session("split-partial")
        report = hy.dream()

        assert report.chunk_extraction_failures == 1
        assert report.chunks_processed == 0
        assert report.triples_extracted == 0
        assert report.markers_extracted == 0
        assert dedup_calls == []
        assert hy.conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM behavioral_markers").fetchone()[0] == 0
        assert not hy.conn.execute("SELECT 1 FROM processed_chunks").fetchone()
    finally:
        hy.close()


def test_persisted_backlog_excludes_digest_fallback_and_handles_null_bounds(cfg):
    hy = HyMem(cfg)
    try:
        hy.conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('backlog')")
        hy.conn.executemany(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text, chunk_kind) VALUES (?, 'backlog', ?, ?, ?, ?, "
            "'extraction')",
            [
                ("normal", 1, 1, "legacy", "durable extraction text"),
                (
                    "digest-fallback",
                    1,
                    1,
                    "short_session_fallback",
                    "digest-only artifact",
                ),
            ],
        )
        rows = load_pending_persisted_chunks(
            hy.conn,
            "backlog",
            prompt_version=hy.config.prompt_version,
            limit=10,
        )
        assert [row.id for row in rows] == ["normal"]
        assert hy.dream_status()["pending_chunks"] == 1
        hy.conn.execute(
            "INSERT INTO chunk_extraction_attempts("
            "chunk_id, prompt_version, attempts) VALUES (?, ?, ?)",
            (
                "digest-fallback",
                hy.config.prompt_version,
                hy.config.chunk_extraction_max_attempts,
            ),
        )
        status = hy.dream_status()
        assert status["pending_chunks"] == 1
        assert status["quarantined_chunks"] == 0
    finally:
        hy.close()

    # Some supported legacy/imported schemas allowed nullable bounds. The
    # backlog reader must stay defensive even though a fresh v39 schema does
    # not create such rows.
    legacy = sqlite3.connect(":memory:")
    legacy.row_factory = sqlite3.Row
    try:
        legacy.executescript(
            """
            CREATE TABLE chunks(
                id TEXT PRIMARY KEY, session_id TEXT, start_message_id INTEGER,
                end_message_id INTEGER, salience_reason TEXT, text TEXT,
                chunk_kind TEXT, created_at TEXT
            );
            CREATE TABLE processed_chunks(chunk_id TEXT, prompt_version TEXT);
            CREATE TABLE chunk_extraction_attempts(
                chunk_id TEXT, prompt_version TEXT, attempts INTEGER
            );
            INSERT INTO chunks VALUES(
                'legacy-null', 'legacy', NULL, NULL, 'legacy', 'text',
                'extraction', '2020-01-01'
            );
            """
        )
        rows = load_pending_persisted_chunks(
            legacy, "legacy", prompt_version="v11", limit=1
        )
        assert rows == [], "unmanifested legacy prose is not claim input authority"
    finally:
        legacy.close()


def test_v12_replays_stored_chunk_after_raw_pruning_with_original_role_weight(cfg):
    digest_empty = json.dumps({"episodes": [], "summary": "", "procedures": []})
    v11_llm = StubLLMClient(
        fixtures={
            "single pass": json.dumps({"triples": [], "markers": []}),
            "Return the JSON object now": digest_empty,
        },
        default="[]",
    )
    base = replace(
        cfg,
        prompt_version="v11",
        message_retention_days=1,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
        salience_min_chars=1,
    )
    hy = HyMem(base, llm=v11_llm)
    try:
        sid = "pruned-v12-replay"
        hy.open_session(sid)
        hy.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) "
            "VALUES (?, 'user', ?, datetime('now', '-10 days'))",
            (sid, "I use PostgreSQL for the durable production database."),
        )
        hy.close_session(sid)
        hy.dream()
        chunk_id = hy.conn.execute(
            "SELECT id FROM chunks WHERE session_id = ? "
            "AND chunk_kind = 'extraction' LIMIT 1",
            (sid,),
        ).fetchone()["id"]
        source_message_id = hy.conn.execute(
            "SELECT source_message_id FROM chunk_message_sources "
            "WHERE chunk_id=? ORDER BY ordinal LIMIT 1", (chunk_id,),
        ).fetchone()[0]
        assert hy.conn.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_id = ? "
            "AND prompt_version = 'v11'",
            (chunk_id,),
        ).fetchone()
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
        ).fetchone()[0] == 0
    finally:
        hy.close()

    v12_llm = StubLLMClient(
        fixtures={
            "single pass": json.dumps({
                "triples": [{
                    "subject": "production_database",
                    "predicate": "uses",
                    "object": "postgresql",
                    "polarity": 1,
                    "source_message_id": source_message_id,
                }],
                "markers": [],
            }),
            "Return the JSON object now": digest_empty,
        },
        default="[]",
    )
    replay = HyMem(replace(base, prompt_version="v12"), llm=v12_llm)
    try:
        report = replay.dream()
        assert report.triples_extracted == 1
        assert replay.conn.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_id = ? "
            "AND prompt_version = 'v12'",
            (chunk_id,),
        ).fetchone()
        evidence = replay.conn.execute(
            "SELECT source_role, evidence_weight, weight_source, "
            "extraction_prompt_version FROM kg_evidence WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        assert tuple(evidence) == (
            "user", 2, "configured_role:user", "v12"
        )
    finally:
        replay.close()
