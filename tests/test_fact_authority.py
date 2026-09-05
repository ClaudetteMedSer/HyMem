"""Adversarial contracts for schema-v46 narrative-fact authority."""

from __future__ import annotations

import dataclasses
import json
import sqlite3

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.core.message_records import canonical_message_record
from hymem.core.vectors import encode_vector
from hymem.dreaming import facts
from hymem.dreaming import embeddings as fact_embeddings
from hymem.dreaming.aggregation_provenance import BoundSourceOccurrence
from hymem.dreaming.embeddings import (
    fetch_fact_embeddings,
    persist_fact_embeddings,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.message_coverage import (
    LOSSLESS_COVERAGE_VERSION,
    coverage_chunk_id,
)
from hymem.extraction.embeddings import embedding_text_hash
from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.query import augment as augment_module
from hymem.query.augment import _fact_search


_CLOSER = "Return the JSON array of narrative facts now"


def _quiet_config(cfg: HyMemConfig, **changes) -> HyMemConfig:
    return dataclasses.replace(
        cfg,
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        **changes,
    )


def _covered_hy(
    cfg: HyMemConfig, session_id: str, turns: list[tuple[str, str]],
    *, llm=None,
) -> tuple[HyMem, list[int]]:
    hy = HyMem(_quiet_config(cfg), llm=llm or StubLLMClient(default="[]"))
    ids = [hy.log_message(session_id, role, text) for role, text in turns]
    with core_db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, session_id)
    return hy, ids


def _extract(
    hy: HyMem, session_id: str, payload: object, *,
    since: int | None = None, partial: int | None = None, offset: int = 0,
    max_chars: int | None = None,
) -> facts.FactsExtraction:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    result = facts.extract_facts(
        hy.conn, session_id, StubLLMClient(default=raw), hy.config,
        since_message_id=since, partial_message_id=partial,
        start_offset=offset, max_chars=max_chars,
    )
    assert result is not None
    return result


def test_blank_occurrence_advances_without_provider_call(cfg):
    llm = StubLLMClient(default='[{"text":"must not run"}]')
    hy, ids = _covered_hy(cfg, "blank-fact", [("user", "   ")], llm=llm)
    try:
        extraction = facts.extract_facts(hy.conn, "blank-fact", llm, hy.config)
        assert extraction is not None and extraction.items == []
        assert llm.calls == []
        with core_db.transaction(hy.conn):
            assert facts.persist_facts(hy.conn, "blank-fact", extraction) == 0
        state = hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset FROM sessions WHERE id='blank-fact'"
        ).fetchone()
        assert tuple(state) == (ids[0], None, 0)
        assert facts.load_fact_outcome_source_manifest(
            hy.conn, extraction.slice_key, verify_result=True
        ) is not None
    finally:
        hy.close()


def test_semantic_slicing_avoids_tiny_next_turn_and_reconstructs_exactly(cfg):
    first_text = "First complete turn."
    long_text = (
        "Alpha beta gamma delta. Epsilon zeta eta theta. "
        "Iota kappa lambda mu."
    )
    hy, ids = _covered_hy(
        cfg, "fact-slices",
        [("user", first_text), ("assistant", long_text)],
    )
    try:
        cap = 40
        first = _extract(hy, "fact-slices", [], max_chars=cap)
        assert [source.message_id for source in first.source_occurrences] == [ids[0]]
        assert first.covered_message_id == ids[0]
        assert first.partial_message_id is None
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-slices", first)

        cursor = ids[0]
        partial = None
        offset = 0
        fragments: list[str] = []
        while True:
            extraction = _extract(
                hy, "fact-slices", [], since=cursor, partial=partial,
                offset=offset, max_chars=cap,
            )
            start = offset if partial == ids[1] else 0
            end = (
                extraction.next_message_offset
                if extraction.partial_message_id == ids[1]
                else len(long_text)
            )
            fragments.append(long_text[start:end])
            if extraction.partial_message_id is not None:
                assert long_text[end - 1].isspace() or long_text[end - 1] in ".!?\n"
            with core_db.transaction(hy.conn):
                facts.persist_facts(hy.conn, "fact-slices", extraction)
            cursor = extraction.covered_message_id
            partial = extraction.partial_message_id
            offset = extraction.next_message_offset
            if extraction.caught_up:
                break
        assert "".join(fragments) == long_text
    finally:
        hy.close()


def test_publication_rejects_skipped_manifest_and_forged_input_pre_mutation(cfg):
    hy, ids = _covered_hy(
        cfg, "forged-fact", [("user", "first"), ("assistant", "second")]
    )
    try:
        skipped = _extract(hy, "forged-fact", [], since=ids[0])
        skipped.cursor_before_message_id = None
        skipped.slice_key = facts.fact_slice_key(
            "forged-fact",
            cursor_before_message_id=None,
            cursor_before_partial_message_id=None,
            cursor_before_offset=0,
            cursor_after_message_id=skipped.covered_message_id,
            cursor_after_partial_message_id=skipped.partial_message_id,
            cursor_after_offset=skipped.next_message_offset,
            occurrences=skipped.source_occurrences,
        )
        with pytest.raises(ValueError, match="skips or misframes"):
            with core_db.transaction(hy.conn):
                facts.persist_facts(hy.conn, "forged-fact", skipped)
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM fact_extraction_outcomes"
        ).fetchone()[0] == 0

        forged_hash = _extract(hy, "forged-fact", [])
        forged_hash.input_hash = "sha256:" + "0" * 64
        with pytest.raises(ValueError, match="input hash"):
            with core_db.transaction(hy.conn):
                facts.persist_facts(hy.conn, "forged-fact", forged_hash)
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM fact_extraction_outcomes"
        ).fetchone()[0] == 0
    finally:
        hy.close()


def test_empty_replay_retracts_and_corrected_same_text_versions_cleanly(cfg):
    hy, _ = _covered_hy(cfg, "fact-replay", [("user", "A dated launch happened.")])
    try:
        original_item = [{
            "text": "The launch happened.",
            "date": "2099-01-02",
            "entities": ["old target"],
        }]
        original = _extract(hy, "fact-replay", original_item)
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-replay", original)
        slice_key = original.slice_key

        empty = facts.reextract_fact_outcome(
            hy.conn, slice_key, StubLLMClient(default="[]"), hy.config
        )
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-replay", empty)
        assert tuple(hy.conn.execute(
            "SELECT outcome_status,generation FROM fact_extraction_outcomes"
        ).fetchone()) == ("empty", 2)
        assert hy.augment("launch happened").facts == []

        corrected_item = [{
            "text": "The launch happened.",
            "date": "2099-01-03",
            "entities": ["correct target"],
        }]
        corrected = facts.reextract_fact_outcome(
            hy.conn, slice_key,
            StubLLMClient(default=json.dumps(corrected_item)), hy.config,
        )
        with core_db.transaction(hy.conn):
            assert facts.persist_facts(hy.conn, "fact-replay", corrected) == 1
        rows = hy.conn.execute(
            "SELECT id,fact_date,entities,lifecycle_status,valid_at,invalid_at "
            "FROM narrative_facts ORDER BY id"
        ).fetchall()
        assert len(rows) == 2
        assert (rows[0]["lifecycle_status"], rows[0]["invalid_at"]) == (
            "retracted", "2099-01-02T00:00:00.000Z"
        )
        assert (rows[1]["fact_date"], rows[1]["lifecycle_status"]) == (
            "2099-01-03", "active"
        )
        assert json.loads(rows[1]["entities"]) == ["correct_target"]
        assert rows[1]["valid_at"] == "2099-01-03T00:00:00.000Z"
        assert [tuple(row) for row in hy.conn.execute(
            "SELECT fact_id,generation,direction,event_at FROM "
            "narrative_fact_lifecycle ORDER BY fact_id,generation"
        )] == [
            (rows[0]["id"], 1, 1, "2099-01-02T00:00:00.000Z"),
            (rows[0]["id"], 2, -1, "2099-01-02T00:00:00.000Z"),
            (rows[1]["id"], 3, 1, "2099-01-03T00:00:00.000Z"),
        ]
        with core_db.transaction(hy.conn):
            assert facts.persist_facts(hy.conn, "fact-replay", corrected) == 0
        assert hy.conn.execute(
            "SELECT generation FROM fact_extraction_outcomes"
        ).fetchone()[0] == 3
    finally:
        hy.close()


def test_normal_writer_cannot_mutate_or_erase_published_fact_history(cfg):
    hy, _ = _covered_hy(cfg, "fact-guards", [("user", "A fact exists.")])
    try:
        extraction = _extract(hy, "fact-guards", [{"text": "A fact exists."}])
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-guards", extraction)
        fact_id = hy.conn.execute("SELECT id FROM narrative_facts").fetchone()[0]
        attempts = (
            ("UPDATE fact_extraction_outcomes SET source_manifest_complete=0", ()),
            ("DELETE FROM fact_extraction_revisions", ()),
            ("DELETE FROM narrative_fact_lifecycle", ()),
            ("DELETE FROM narrative_facts WHERE id=?", (fact_id,)),
            ("DELETE FROM fact_extraction_outcomes", ()),
        )
        for sql, params in attempts:
            with pytest.raises(sqlite3.DatabaseError):
                with core_db.evidence_mutation(hy.conn):
                    hy.conn.execute(sql, params)
        assert hy.conn.execute("SELECT COUNT(*) FROM narrative_facts").fetchone()[0] == 1

        hy.conn.execute(
            "INSERT INTO narrative_facts(session_id,start_message_id,"
            "end_message_id,text,prompt_version) VALUES "
            "('fact-guards',1,1,'legacy','facts.v2')"
        )
        legacy_id = hy.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        with pytest.raises(sqlite3.DatabaseError):
            with core_db.evidence_mutation(hy.conn):
                hy.conn.execute(
                    "UPDATE narrative_facts SET source_outcome_key=?,fact_key=?,"
                    "current_generation=1,lifecycle_status='active' WHERE id=?",
                    (extraction.slice_key, "sha256:" + "1" * 64, legacy_id),
                )
    finally:
        hy.close()


def test_disconnected_chain_is_hidden_from_read_query_and_embedder(cfg):
    hy, _ = _covered_hy(cfg, "fact-orphan", [("user", "Orion uses a quasar cache.")])
    try:
        extraction = _extract(
            hy, "fact-orphan", [{"text": "Orion uses a quasar cache."}]
        )
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-orphan", extraction)
        fact_id = hy.conn.execute("SELECT id FROM narrative_facts").fetchone()[0]
        assert facts.load_fact_source_manifest(hy.conn, fact_id) is not None

        # Simulate post-publication storage corruption at the final session
        # commit coordinate. Individual slice hashes remain valid, but the unit
        # is no longer cursor-committed and must disappear everywhere.
        hy.conn.execute(
            "UPDATE sessions SET facts_message_id=NULL,"
            "facts_cursor_message_id=NULL,facts_cursor_partial_message_id=NULL,"
            "facts_cursor_offset=0,facts_cursor_prompt_version=NULL "
            "WHERE id='fact-orphan'"
        )
        assert facts.load_fact_source_manifest(hy.conn, fact_id) is None
        assert hy.augment("quasar cache").facts == []
        embedder = StubEmbeddingClient()
        pending = fetch_fact_embeddings(hy.conn, embedder)
        assert pending is not None and pending.fact_ids == []
        assert embedder.calls == []
    finally:
        hy.close()


class _AdaptiveFactLLM(StubLLMClient):
    def __init__(self, threshold: int | None):
        super().__init__(
            fixtures={
                "Return the JSON object now":
                    '{"episodes":[],"summary":"","procedures":[]}',
                "single pass": '{"triples":[],"markers":[]}',
            },
            default="[]",
        )
        self.threshold = threshold
        self.fact_sizes: list[int] = []

    def complete(self, request: LLMRequest) -> str:
        if _CLOSER in request.user:
            self.calls.append(request)
            rendered = request.user.split('"""', 2)[1].strip("\n")
            self.fact_sizes.append(len(rendered))
            if self.threshold is None or len(rendered) > self.threshold:
                return "malformed"
            return "[]"
        return super().complete(request)


def test_adaptive_retry_shrinks_to_semantic_floor_without_skipping(cfg):
    llm = _AdaptiveFactLLM(threshold=256)
    config = _quiet_config(
        cfg, dream_digest_max_chars=1024, facts_extraction_max_attempts=6,
    )
    hy = HyMem(config, llm=llm)
    try:
        hy.log_message("fact-adaptive", "user", "word " * 500)
        first = hy.dream()
        second = hy.dream()
        third = hy.dream()
        assert first.fact_failures == second.fact_failures == 1
        assert third.fact_failures == 0
        assert [facts.facts_attempt_max_chars(1024, retry) for retry in range(3)] == [
            1024, 512, 256,
        ]
        assert 512 < llm.fact_sizes[0] <= 1024
        assert 256 < llm.fact_sizes[1] <= 512
        assert llm.fact_sizes[2] <= 256
        state = hy.conn.execute(
            "SELECT facts_cursor_partial_message_id,facts_cursor_offset,"
            "facts_retry_count,facts_quarantined FROM sessions "
            "WHERE id='fact-adaptive'"
        ).fetchone()
        assert state["facts_cursor_partial_message_id"] is not None
        assert state["facts_cursor_offset"] > 0
        assert (state["facts_retry_count"], state["facts_quarantined"]) == (0, 0)
    finally:
        hy.close()


def test_repeated_failures_never_shred_below_floor_and_quarantine(cfg):
    llm = _AdaptiveFactLLM(threshold=None)
    config = _quiet_config(
        cfg, dream_digest_max_chars=1024, facts_extraction_max_attempts=5,
    )
    hy = HyMem(config, llm=llm)
    try:
        hy.log_message("fact-floor", "assistant", "meaningful context " * 150)
        reports = [hy.dream() for _ in range(5)]
        assert all(report.fact_failures == 1 for report in reports)
        assert [facts.facts_attempt_max_chars(1024, retry) for retry in range(5)] == [
            1024, 512, 256, 256, 256,
        ]
        assert len(llm.fact_sizes) == 5
        assert 512 < llm.fact_sizes[0] <= 1024
        assert 256 < llm.fact_sizes[1] <= 512
        assert all(128 <= size <= 256 for size in llm.fact_sizes[2:])
        state = hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_retry_count,facts_quarantined FROM sessions WHERE id='fact-floor'"
        ).fetchone()
        assert tuple(state) == (None, None, 5, 1)
    finally:
        hy.close()


def test_bulk_proof_validates_one_outcome_once_and_payload_limits(cfg):
    items = [
        {"text": f"Shared unit fact {index}.", "entities": [f"entity {index}"]}
        for index in range(32)
    ]
    config = _quiet_config(cfg, dream_max_facts_per_session=32)
    hy, _ = _covered_hy(
        config, "fact-proof-cache", [("user", "Many facts in one unit.")]
    )
    try:
        extraction = _extract(hy, "fact-proof-cache", items)
        with core_db.transaction(hy.conn):
            facts.persist_facts(
                hy.conn, "fact-proof-cache", extraction, max_items=32
            )
        ids = [row[0] for row in hy.conn.execute(
            "SELECT id FROM narrative_facts ORDER BY id"
        )]
        statements: list[str] = []
        hy.conn.set_trace_callback(statements.append)
        try:
            resolved = facts.load_fact_source_manifests(hy.conn, ids)
        finally:
            hy.conn.set_trace_callback(None)
        assert set(resolved) == set(ids)
        assert sum(
            "FROM fact_extraction_revisions WHERE slice_key" in statement
            for statement in statements
        ) == 1

        assert facts.validate_fact_items([{
            "text": "too many entities",
            "entities": [f"e{index}" for index in range(65)],
        }], max_items=1) is None
        assert facts.validate_fact_items([{
            "text": "oversized entity", "entities": ["x" * 201],
        }], max_items=1) is None
        with pytest.raises(ValueError, match="between 1 and 256"):
            dataclasses.replace(cfg, dream_max_facts_per_session=257)
    finally:
        hy.close()


def test_facts_valid_at_folds_retraction_resurrection_scope_and_proof(cfg):
    hy, _ = _covered_hy(
        cfg, "fact-time-a", [("user", "The Atlas launch is recorded.")]
    )
    try:
        second_id = hy.log_message(
            "fact-time-b", "user", "The Borealis launch is recorded."
        )
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "fact-time-b")
        item_a = [{"text": "Atlas launched.", "date": "2035-04-05"}]
        item_b = [{"text": "Borealis launched.", "date": "2035-04-05"}]
        first = _extract(hy, "fact-time-a", item_a)
        second = _extract(hy, "fact-time-b", item_b)
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-time-a", first)
            facts.persist_facts(hy.conn, "fact-time-b", second)

        assert facts.facts_valid_at(
            hy.conn, "2035-04-04T23:59:59.999Z"
        ) == []
        at_onset = facts.facts_valid_at(
            hy.conn, "2035-04-05T00:00:00.000Z"
        )
        assert {row["text"] for row in at_onset} == {
            "Atlas launched.", "Borealis launched.",
        }
        assert [row["text"] for row in facts.facts_valid_at(
            hy.conn, "2035-04-05T00:00:00.000Z", session_id="fact-time-a"
        )] == ["Atlas launched."]
        assert facts.facts_valid_at(
            hy.conn, "2035-04-05T00:00:00.000Z", session_id="not-that-session"
        ) == []

        empty = facts.reextract_fact_outcome(
            hy.conn, first.slice_key, StubLLMClient(default="[]"), hy.config
        )
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-time-a", empty)
        assert facts.facts_valid_at(
            hy.conn, "2035-04-05T00:00:00.000Z", session_id="fact-time-a"
        ) == []

        resurrect = facts.reextract_fact_outcome(
            hy.conn, first.slice_key,
            StubLLMClient(default=json.dumps(item_a)), hy.config,
        )
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-time-a", resurrect)
        assert [row["text"] for row in facts.facts_valid_at(
            hy.conn, "2035-04-05T00:00:00.000Z", session_id="fact-time-a"
        )] == ["Atlas launched."]

        # Break the exact committed-chain proof after native selection.  The
        # historical reader must fail closed just like current retrieval.
        hy.conn.execute(
            "UPDATE sessions SET facts_cursor_message_id=NULL,"
            "facts_cursor_partial_message_id=NULL,facts_cursor_offset=0 "
            "WHERE id='fact-time-a'"
        )
        assert facts.facts_valid_at(
            hy.conn, "2035-04-05T00:00:00.000Z", session_id="fact-time-a"
        ) == []
        assert [row["text"] for row in facts.facts_valid_at(
            hy.conn, "2035-04-05T00:00:00.000Z", session_id="fact-time-b"
        )] == ["Borealis launched."]
        assert second_id > 0
    finally:
        hy.close()


def test_config_bump_cannot_be_laundered_by_appending_a_new_tail(cfg):
    hy, ids = _covered_hy(cfg, "fact-version", [("user", "first unit")])
    try:
        first = _extract(hy, "fact-version", [])
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-version", first)
        version_a = facts.facts_config_version(hy.config)

        config_b = dataclasses.replace(
            hy.config,
            dream_max_facts_per_session=hy.config.dream_max_facts_per_session + 1,
        )
        version_b = facts.facts_config_version(config_b)
        hy.conn.execute(
            "UPDATE sessions SET facts_cursor_prompt_version=? "
            "WHERE id='fact-version'", (version_b,),
        )
        assert facts.next_fact_outcome_for_replay(
            hy.conn, "fact-version", version_b
        ) == first.slice_key
        hy.conn.execute(
            "UPDATE sessions SET facts_cursor_prompt_version=? "
            "WHERE id='fact-version'", (version_a,),
        )

        second_id = hy.log_message("fact-version", "assistant", "second unit")
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "fact-version")
        second = facts.extract_facts(
            hy.conn, "fact-version", StubLLMClient(default="[]"), config_b,
            since_message_id=ids[0],
        )
        assert second is not None
        with pytest.raises(ValueError, match="cursor successor"):
            with core_db.transaction(hy.conn):
                facts.persist_facts(hy.conn, "fact-version", second)
        state = hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_prompt_version "
            "FROM sessions WHERE id='fact-version'"
        ).fetchone()
        assert tuple(state) == (ids[0], version_a)
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM fact_extraction_outcomes"
        ).fetchone()[0] == 1
        assert second_id > ids[0]
    finally:
        hy.close()


def test_concurrent_publish_and_replay_use_cursor_and_generation_cas(cfg):
    hy, ids = _covered_hy(cfg, "fact-cas", [("user", "initial source")])
    peer = core_db.connect(hy.config.db_path)
    try:
        winner = _extract(hy, "fact-cas", [{"text": "Winner fact."}])
        loser = _extract(hy, "fact-cas", [{"text": "Divergent fact."}])
        with core_db.transaction(peer):
            assert facts.persist_facts(peer, "fact-cas", winner) == 1
        with pytest.raises(RuntimeError, match="concurrently published"):
            with core_db.transaction(hy.conn):
                facts.persist_facts(hy.conn, "fact-cas", loser)
        assert hy.conn.execute(
            "SELECT generation,result_hash FROM fact_extraction_outcomes"
        ).fetchone()["generation"] == 1

        replay_a = facts.reextract_fact_outcome(
            hy.conn, winner.slice_key,
            StubLLMClient(default='[{"text":"Replay A."}]'), hy.config,
        )
        replay_b = facts.reextract_fact_outcome(
            peer, winner.slice_key,
            StubLLMClient(default='[{"text":"Replay B."}]'), hy.config,
        )
        assert replay_a.expected_generation == replay_b.expected_generation == 1
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-cas", replay_a)
        with pytest.raises(RuntimeError, match="generation changed"):
            with core_db.transaction(peer):
                facts.persist_facts(peer, "fact-cas", replay_b)
        assert hy.conn.execute(
            "SELECT generation FROM fact_extraction_outcomes"
        ).fetchone()[0] == 2

        # A stale duplicate worker must not regress the cursor after a later
        # unit has committed; persist owns cursor advancement now.
        second_id = hy.log_message("fact-cas", "assistant", "later source")
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "fact-cas")
        later = _extract(
            hy, "fact-cas", [], since=ids[0]
        )
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-cas", later)
        with core_db.transaction(peer):
            assert facts.persist_facts(peer, "fact-cas", replay_a) == 0
        assert hy.conn.execute(
            "SELECT facts_cursor_message_id FROM sessions WHERE id='fact-cas'"
        ).fetchone()[0] == second_id

        # A failure finishing after that success also loses its CAS and cannot
        # reinstall stale retry/quarantine state.
        retry_key = facts.facts_retry_policy_version(
            hy.config,
            replay_slice_key=facts.fact_cursor_retry_unit_key(
                "fact-cas", ids[0], None, 0
            ),
        )
        with core_db.transaction(peer):
            held = facts.record_fact_failure_if_pending(
                peer, "fact-cas",
                max_attempts=1, retry_config_version=retry_key,
                expected_cursor_message_id=ids[0],
                expected_partial_message_id=None, expected_offset=0,
            )
        assert held is None
        assert tuple(hy.conn.execute(
            "SELECT facts_retry_count,facts_retry_config_version,"
            "facts_quarantined FROM sessions WHERE id='fact-cas'"
        ).fetchone()) == (0, None, 0)
    finally:
        peer.close()
        hy.close()


def test_fact_chain_indexes_and_quiescent_replay_path_scale(cfg, monkeypatch):
    hy, _ = _covered_hy(
        cfg, "fact-scale", [("user", "word " * 1800)]
    )
    try:
        cursor = partial = None
        offset = 0
        slices = 0
        while True:
            extraction = _extract(
                hy, "fact-scale", [], since=cursor, partial=partial,
                offset=offset, max_chars=256,
            )
            with core_db.transaction(hy.conn):
                facts.persist_facts(hy.conn, "fact-scale", extraction)
            slices += 1
            cursor = extraction.covered_message_id
            partial = extraction.partial_message_id
            offset = extraction.next_message_offset
            if extraction.caught_up:
                break
        assert slices >= 20

        plans = {
            "before": hy.conn.execute(
                "EXPLAIN QUERY PLAN SELECT 1 FROM fact_extraction_outcomes "
                "WHERE session_id=? AND cursor_before_message_id IS ? "
                "AND cursor_before_partial_message_id IS ? "
                "AND cursor_before_offset=? LIMIT 1",
                ("fact-scale", None, None, 0),
            ).fetchall(),
            "after": hy.conn.execute(
                "EXPLAIN QUERY PLAN SELECT slice_key FROM fact_extraction_outcomes "
                "WHERE session_id=? AND cursor_after_message_id IS ? "
                "AND cursor_after_partial_message_id IS ? "
                "AND cursor_after_offset=? LIMIT 2",
                ("fact-scale", cursor, partial, offset),
            ).fetchall(),
            "chain": hy.conn.execute(
                "EXPLAIN QUERY PLAN SELECT slice_key FROM fact_extraction_outcomes "
                "WHERE session_id=? ORDER BY "
                "COALESCE(cursor_before_partial_message_id,"
                "cursor_before_message_id,-1),"
                "CASE WHEN cursor_before_partial_message_id IS NULL "
                "THEN 1 ELSE 0 END,cursor_before_offset,slice_key",
                ("fact-scale",),
            ).fetchall(),
            "replay": hy.conn.execute(
                "EXPLAIN QUERY PLAN SELECT slice_key FROM fact_extraction_outcomes "
                "WHERE session_id=? AND source_manifest_complete=1 "
                "AND prompt_version<? LIMIT 1",
                ("fact-scale", facts.facts_config_version(hy.config)),
            ).fetchall(),
        }
        details = {
            name: " ".join(str(row["detail"]) for row in rows)
            for name, rows in plans.items()
        }
        assert "idx_fact_outcome_before_cursor" in details["before"]
        assert "idx_fact_outcome_after_cursor" in details["after"]
        assert "idx_fact_outcome_chain_order" in details["chain"]
        assert "TEMP B-TREE" not in details["chain"]
        assert "idx_fact_outcome_replay_v46" in details["replay"]

        proof_calls = 0
        original = facts.load_fact_outcome_source_manifest

        def counted(*args, **kwargs):
            nonlocal proof_calls
            proof_calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(facts, "load_fact_outcome_source_manifest", counted)
        statements: list[str] = []
        hy.conn.set_trace_callback(statements.append)
        try:
            for _ in range(20):
                assert facts.next_fact_outcome_for_replay(
                    hy.conn, "fact-scale", facts.facts_config_version(hy.config)
                ) is None
        finally:
            hy.conn.set_trace_callback(None)
        assert proof_calls == 0
        selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        assert len(selects) <= 80

        monkeypatch.setattr(facts, "load_fact_outcome_source_manifest", original)
        committed_scans = 0
        original_committed = facts._committed_fact_slice_keys

        def counted_committed(*args, **kwargs):
            nonlocal committed_scans
            committed_scans += 1
            return original_committed(*args, **kwargs)

        monkeypatch.setattr(
            facts, "_committed_fact_slice_keys", counted_committed
        )
        hy.config = dataclasses.replace(
            hy.config,
            dream_max_facts_per_session=hy.config.dream_max_facts_per_session + 1,
        )
        target_version = facts.facts_config_version(hy.config)
        replay_sql: list[str] = []
        hy.conn.set_trace_callback(replay_sql.append)
        try:
            reports = []
            for _ in range(slices + 1):
                report = hy.dream()
                reports.append(report)
                marker = hy.conn.execute(
                    "SELECT facts_cursor_prompt_version FROM sessions "
                    "WHERE id='fact-scale'"
                ).fetchone()[0]
                if marker == target_version:
                    break
        finally:
            hy.conn.set_trace_callback(None)
        assert marker == target_version
        assert len(reports) == slices
        assert all(report.budget_exhausted for report in reports[:-1])
        assert committed_scans == 1
        assert not any(
            "COUNT(*) FROM fact_extraction_outcomes" in statement
            for statement in replay_sql
        )
    finally:
        hy.close()


def test_fact_publication_clock_never_precedes_future_proof_or_revision(cfg):
    hy = HyMem(_quiet_config(cfg), llm=StubLLMClient(default="[]"))
    try:
        hy.conn.execute("INSERT INTO sessions(id) VALUES ('fact-clock')")
        message_id = hy.conn.execute(
            "INSERT INTO messages(session_id,role,content) VALUES "
            "('fact-clock','user','A source from another writer.')"
        ).lastrowid
        message = hy.conn.execute(
            "SELECT * FROM messages WHERE id=?", (message_id,)
        ).fetchone()
        record, content_hash, hash_version, record_version = canonical_message_record(
            message_id=message_id, session_id="fact-clock", role="user",
            content=message["content"], source_created_at=message["created_at"],
            source_peer_id=None, source_workspace_id=None,
        )
        chunk_id = coverage_chunk_id("fact-clock", message_id)
        future = "2098-07-06T05:04:03.210Z"
        with core_db.transaction(hy.conn):
            hy.conn.execute(
                "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
                "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?, 'coverage')",
                (chunk_id, "fact-clock", message_id, message_id,
                 "lossless_message", record),
            )
            hy.conn.execute(
                "INSERT INTO message_retention_coverage("
                "message_id,source_session_id,source_role,source_peer_id,"
                "source_workspace_id,source_created_at,chunk_id,"
                "message_content_hash,hash_version,record_version,"
                "coverage_version,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (message_id, "fact-clock", "user", None, None,
                 message["created_at"], chunk_id, content_hash, hash_version,
                 record_version, LOSSLESS_COVERAGE_VERSION, future),
            )
            hy.conn.execute(
                "UPDATE sessions SET coverage_message_id=? WHERE id='fact-clock'",
                (message_id,),
            )
        extraction = _extract(
            hy, "fact-clock", [{"text": "A remote event was recorded."}]
        )
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-clock", extraction)
        assert hy.conn.execute(
            "SELECT succeeded_at FROM fact_extraction_revisions WHERE generation=1"
        ).fetchone()[0] == future

        replay = facts.reextract_fact_outcome(
            hy.conn, extraction.slice_key,
            StubLLMClient(default='[{"text":"Corrected remote event."}]'),
            hy.config,
        )
        with core_db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "fact-clock", replay)
        times = [row[0] for row in hy.conn.execute(
            "SELECT succeeded_at FROM fact_extraction_revisions ORDER BY generation"
        )]
        assert times == [future, future]
        outcome = hy.conn.execute(
            "SELECT generation,succeeded_at FROM fact_extraction_outcomes"
        ).fetchone()
        assert tuple(outcome) == (2, future)
        assert [row[0] for row in hy.conn.execute(
            "SELECT recorded_at FROM narrative_fact_lifecycle ORDER BY generation"
        )] == [future, future, future]
        assert facts.load_fact_outcome_source_manifest(
            hy.conn, extraction.slice_key, verify_result=True
        ) is not None
    finally:
        hy.close()


def test_fact_embedding_scan_cursor_batches_and_reuses_cross_page_proof(
    cfg, monkeypatch,
):
    config = _quiet_config(cfg, dream_max_facts_per_session=40)
    hy, _ = _covered_hy(
        config, "fact-embed-pages", [("user", "One dense authority unit.")]
    )
    try:
        extraction = _extract(
            hy, "fact-embed-pages",
            [{"text": f"Bounded embedding fact {index}."} for index in range(40)],
        )
        with core_db.transaction(hy.conn):
            facts.persist_facts(
                hy.conn, "fact-embed-pages", extraction, max_items=40
            )

        initial_embedder = StubEmbeddingClient()
        initial = fetch_fact_embeddings(
            hy.conn, initial_embedder, batch_size=36
        )
        assert initial is not None and len(initial.fact_ids) == 36
        assert max(map(len, initial_embedder.calls)) == 36
        with core_db.transaction(hy.conn):
            persist_fact_embeddings(hy.conn, initial)

        # Force the bounded repair sweep to revisit the embedded prefix. It
        # must continue across it, retain one outcome/session proof across
        # pages, and expose only the bounded miss batch to the provider.
        hy.conn.execute(
            "UPDATE schema_meta SET value='0' WHERE key=?",
            (initial.scan_state_key,),
        )
        original = fact_embeddings.load_fact_source_manifests
        page_shapes: list[tuple[int, int, int]] = []

        def counted(conn, ids, *, outcome_cache=None, chain_cache=None):
            page_shapes.append((
                len(ids), len(outcome_cache or {}), len(chain_cache or {}),
            ))
            return original(
                conn, ids, outcome_cache=outcome_cache, chain_cache=chain_cache
            )

        monkeypatch.setattr(
            fact_embeddings, "load_fact_source_manifests", counted
        )
        embedder = StubEmbeddingClient()
        statements: list[str] = []
        hy.conn.set_trace_callback(statements.append)
        try:
            pending = fetch_fact_embeddings(hy.conn, embedder, batch_size=3)
        finally:
            hy.conn.set_trace_callback(None)
        assert pending is not None and len(pending.fact_ids) == 3
        assert embedder.calls and len(embedder.calls[0]) == 3
        assert [shape[0] for shape in page_shapes] == [32, 8]
        assert page_shapes[0][1:] == (0, 0)
        assert page_shapes[1][1:] == (1, 1)
        assert sum(
            "FROM fact_extraction_revisions WHERE slice_key" in statement
            for statement in statements
        ) == 1
        assert all(" OFFSET " not in statement.upper() for statement in statements)
        with core_db.transaction(hy.conn):
            persist_fact_embeddings(hy.conn, pending)
        assert hy.conn.execute(
            "SELECT value FROM schema_meta WHERE key=?",
            (pending.scan_state_key,),
        ).fetchone()[0] == pending.scan_next_value

        peer = core_db.connect(hy.config.db_path)
        try:
            stale = fetch_fact_embeddings(hy.conn, StubEmbeddingClient(), batch_size=1)
            wrapping = fetch_fact_embeddings(peer, StubEmbeddingClient(), batch_size=3)
            assert stale is not None and wrapping is not None
            assert stale.scan_start_value == wrapping.scan_start_value
            assert stale.scan_next_value != wrapping.scan_next_value
            with core_db.transaction(peer):
                persist_fact_embeddings(peer, wrapping)
            with core_db.transaction(hy.conn):
                persist_fact_embeddings(hy.conn, stale)
            # The late narrower scan loses its state CAS; vectors themselves
            # remain harmlessly idempotent.
            assert hy.conn.execute(
                "SELECT value FROM schema_meta WHERE key=?",
                (pending.scan_state_key,),
            ).fetchone()[0] == wrapping.scan_next_value
        finally:
            peer.close()

        quiet_embedder = StubEmbeddingClient()
        page_shapes.clear()
        quiet = fetch_fact_embeddings(hy.conn, quiet_embedder, batch_size=3)
        assert quiet is not None and quiet.fact_ids == []
        assert quiet_embedder.calls == []
        assert sum(shape[0] for shape in page_shapes) <= 256
        with core_db.transaction(hy.conn):
            persist_fact_embeddings(hy.conn, quiet)
    finally:
        hy.close()


def test_fact_query_keysets_past_old_cap_with_bounded_heap_and_caches(monkeypatch):
    proof = (BoundSourceOccurrence(
        message_id=1, session_id="tail-session", role="user",
        source_peer_id=None, source_workspace_id=None,
        source_created_at="2026-01-01T00:00:00.000Z",
        coverage_chunk_id="coverage", coverage_version="lossless-message-v1",
        content_hash="sha256:" + "1" * 64,
    ),)

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

    class VectorPages:
        def __init__(self, total):
            self.total = total
            self.sql: list[str] = []

        def execute(self, sql, params=()):
            self.sql.append(sql)
            assert "narrative_fact_embeddings" in sql
            last_id = int(params[2])
            take = int(params[-1])
            end = min(self.total, last_id + take)
            rows = []
            for fact_id in range(last_id + 1, end + 1):
                text = f"candidate {fact_id}"
                rows.append({
                    "id": fact_id, "session_id": "tail-session",
                    "text": text, "fact_date": None, "entities": "[]",
                    "source_outcome_key": f"outcome-{fact_id}",
                    "vector_json": encode_vector([1.0, 0.0]),
                    "text_hash": (
                        embedding_text_hash(text)
                        if fact_id == self.total else "invalid-prefix"
                    ),
                })
            return Result(rows)

    total = 65_540  # beyond the removed 65,536-candidate safety ceiling
    vector_conn = VectorPages(total)
    proof_page_sizes: list[int] = []

    def tail_only(_conn, ids, *, outcome_cache=None, chain_cache=None):
        proof_page_sizes.append(len(ids))
        assert len(outcome_cache or {}) <= 64
        assert len(chain_cache or {}) <= 256
        return {total: proof} if total in ids else {}

    monkeypatch.setattr(
        augment_module, "load_fact_source_manifests", tail_only
    )
    original_push = augment_module.heapq.heappush
    max_heap = 0

    def counted_push(heap, item):
        nonlocal max_heap
        original_push(heap, item)
        max_heap = max(max_heap, len(heap))

    monkeypatch.setattr(augment_module.heapq, "heappush", counted_push)
    hits = _fact_search(
        vector_conn, "x", top_k=1,
        embedding_client=StubEmbeddingClient(dim_value=2),
        query_vector=[1.0, 0.0],
    )
    assert [hit.fact_id for hit in hits] == [total]
    assert max(proof_page_sizes) <= 64
    assert max_heap <= 2
    assert len(vector_conn.sql) > 1024
    assert all(" OFFSET " not in sql.upper() for sql in vector_conn.sql)

    class FtsPages:
        def __init__(self, total):
            self.total = total
            self.sql: list[str] = []

        def execute(self, sql, params=()):
            self.sql.append(sql)
            assert "narrative_facts_fts" in sql
            last_id = int(params[-2]) if len(params) > 2 else 0
            take = int(params[-1])
            end = min(self.total, last_id + take)
            return Result([{
                "id": fact_id, "session_id": "tail-session",
                "text": f"needle candidate {fact_id}", "fact_date": None,
                "entities": "[]", "source_outcome_key": f"fts-{fact_id}",
                "score": float(fact_id),
            } for fact_id in range(last_id + 1, end + 1)])

    fts_total = 300
    fts_conn = FtsPages(fts_total)

    def fts_tail_only(_conn, ids, *, outcome_cache=None, chain_cache=None):
        assert len(outcome_cache or {}) <= 64
        assert len(chain_cache or {}) <= 256
        return {fts_total: proof} if fts_total in ids else {}

    monkeypatch.setattr(
        augment_module, "load_fact_source_manifests", fts_tail_only
    )
    fts_hits = _fact_search(fts_conn, "needle", top_k=1)
    assert [hit.fact_id for hit in fts_hits] == [fts_total]
    assert len(fts_conn.sql) >= 5
    assert all(" OFFSET " not in sql.upper() for sql in fts_conn.sql)
