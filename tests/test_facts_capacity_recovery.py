"""Bounded fresh-source recovery does not loosen fact authority or item caps."""

from contextlib import closing
from dataclasses import replace
import json

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline
from hymem.dreaming import facts
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient


_CLOSER = "Return the JSON object of narrative facts now"
_OVERFLOW = json.dumps({"facts": [], "complete": False})


def _config(cfg, **overrides):
    return replace(cfg, aggregation_nodes_enabled=False,
                   profile_extraction_enabled=False, **overrides)


def _seed(hy, texts):
    ids = [hy.log_message("recovery", "user", text) for text in texts]
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "recovery")
    return ids


def _source(request):
    return request.user.split('"""')[1].strip("\n")


def _state(hy):
    tables = (
        "sessions", "messages", "chunks", "chunk_message_sources",
        "fact_extraction_outcomes", "fact_extraction_revisions",
        "fact_extraction_source_occurrences", "narrative_facts",
        "narrative_fact_lifecycle",
    )
    return tuple(tuple(tuple(row) for row in hy.conn.execute(
        f"SELECT * FROM {table} ORDER BY rowid"
    )) for table in tables)


class _SourceFaithfulClient(StubLLMClient):
    def __init__(self, texts):
        super().__init__(default="[]")
        self.texts = texts
        self.fact_calls = []

    def complete(self, request):
        if _CLOSER not in request.user:
            return super().complete(request)
        self.calls.append(request)
        self.fact_calls.append(request)
        # Intentionally ignore the item cap in the response, matching the
        # live failure. Recovery must reduce source, not salvage this set.
        return json.dumps({"facts": [
            {"text": text, "date": None, "entities": ["Nora"]}
            for text in self.texts if text in _source(request)
        ]})


def test_nine_valid_facts_retry_actual_short_source_and_commit_full_tail(cfg):
    texts = [f"Nora shipped package {index}: " + "details " * 55 + "."
             for index in range(9)]
    llm = _SourceFaithfulClient(texts)
    config = _config(cfg, dream_digest_max_chars=12000, dream_max_facts_per_session=8)
    with closing(HyMem(config, llm=llm)) as hy:
        ids = _seed(hy, texts)
        original_messages = tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM messages"))
        first = hy.dream()
        assert first.fact_failures == 0 and first.facts_extracted == 4
        assert first.budget_exhausted
        assert len(llm.fact_calls) == 2
        full, prefix = [_source(request) for request in llm.fact_calls]
        assert 4000 < len(full) < 6000 < config.dream_digest_max_chars
        assert prefix == "\n".join("user: " + text for text in texts[:4])
        assert len(prefix) <= len(full) // 2
        assert all("at most 8 facts" in request.system for request in llm.fact_calls)
        assert hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset,facts_retry_count,facts_quarantined "
            "FROM sessions WHERE id='recovery'"
        ).fetchone()[:] == (ids[3], None, 0, 0, 0)
        second = hy.dream()
        assert second.fact_failures == 0 and second.facts_extracted == 5
        assert len(llm.fact_calls) == 3
        suffix = _source(llm.fact_calls[-1])
        assert prefix + "\n" + suffix == full
        assert hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_retry_count,facts_quarantined "
            "FROM sessions WHERE id='recovery'"
        ).fetchone()[:] == (ids[-1], 0, 0)
        outcomes = list(hy.conn.execute(
            "SELECT * FROM fact_extraction_outcomes ORDER BY rowid"
        ))
        assert len(outcomes) == 2
        assert [row["input_hash"] for row in outcomes] == [
            facts.fact_input_hash(prefix), facts.fact_input_hash(suffix),
        ]
        assert all(facts.load_fact_outcome_source_manifest(hy.conn, row["slice_key"])
                   for row in outcomes)
        assert {row[0] for row in hy.conn.execute(
            "SELECT text FROM narrative_facts WHERE lifecycle_status='active'"
        )} == set(texts)
        assert facts.fact_session_authority_is_valid(hy.conn, "recovery")
        assert tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM messages")) == original_messages
        assert hy.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_double_capacity_failure_is_two_calls_one_held_attempt_and_no_cursor(cfg):
    llm = StubLLMClient(fixtures={_CLOSER: _OVERFLOW}, default="[]")
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy, ["Nora shipped Cedar. " * 200])
        before = _state(hy)
        result = facts.extract_facts(hy.conn, "recovery", llm, hy.config)
        assert result.parse_failed and result.failure_reason == "output_capacity_exceeded"
        assert result.items == [] and result.covered_message_id is None
        assert _state(hy) == before
        assert len(llm.calls) == 2
        assert len(_source(llm.calls[1])) <= len(_source(llm.calls[0])) // 2
        llm.calls.clear()
        report = hy.dream()
        assert report.fact_failures == 1 and report.facts_extracted == 0
        assert len([call for call in llm.calls if _CLOSER in call.user]) == 2
        assert hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset,facts_retry_count,facts_quarantined "
            "FROM sessions WHERE id='recovery'"
        ).fetchone()[:] == (None, None, 0, 1, 0)
        assert hy.conn.execute("SELECT COUNT(*) FROM fact_extraction_outcomes").fetchone()[0] == 0


def test_recovered_partial_turn_preserves_exact_offset_and_all_source_bytes(cfg):
    texts = [f"Nora shipped package {index}: " + "details " * 55 + "."
             for index in range(9)]
    content = "\n".join(texts)
    llm = _SourceFaithfulClient(texts)
    with closing(HyMem(_config(cfg, dream_digest_max_chars=12000), llm=llm)) as hy:
        message_id = _seed(hy, [content])[0]
        first = facts.extract_facts(hy.conn, "recovery", llm, hy.config)
        assert not first.parse_failed and len(first.items) == 4
        assert first.covered_message_id is None and first.partial_message_id == message_id
        offset = first.next_message_offset
        assert 0 < offset < len(content) and not first.caught_up
        assert llm.fact_calls[-1].user == facts.build_facts_request(
            "user: " + content[:offset], hy.config,
        ).user
        with db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "recovery", first)
        second = facts.extract_facts(
            hy.conn, "recovery", llm, hy.config,
            since_message_id=first.covered_message_id,
            partial_message_id=first.partial_message_id,
            start_offset=offset,
        )
        assert not second.parse_failed and len(second.items) == 5
        assert second.cursor_before_partial_message_id == message_id
        assert second.cursor_before_offset == offset
        assert second.partial_message_id is None and second.covered_message_id == message_id
        assert second.caught_up and len(llm.fact_calls) == 3
        assert llm.fact_calls[-1].user == facts.build_facts_request(
            "user: " + content[offset:], hy.config,
        ).user
        assert first.input_hash == facts.fact_input_hash("user: " + content[:offset])
        assert second.input_hash == facts.fact_input_hash("user: " + content[offset:])
        with db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "recovery", second)
        assert facts.fact_session_authority_is_valid(hy.conn, "recovery")
        assert {row[0] for row in hy.conn.execute(
            "SELECT text FROM narrative_facts WHERE lifecycle_status='active'"
        )} == set(texts)


@pytest.mark.parametrize("size", [20, 250])
def test_capacity_at_semantic_floor_does_not_retry_identical_source(cfg, size):
    llm = StubLLMClient(default=_OVERFLOW)
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy, ["x" * size])
        result = facts.extract_facts(hy.conn, "recovery", llm, hy.config)
        assert result.parse_failed and result.failure_reason == "output_capacity_exceeded"
        assert len(llm.calls) == 1


@pytest.mark.parametrize("raw", ["invalid-json", '{"facts":false}', '{"facts":[{"text":false}]}'])
def test_noncapacity_failure_is_not_retried(cfg, raw):
    llm = StubLLMClient(default=raw)
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy, ["Nora shipped Cedar. " * 200])
        before = _state(hy)
        result = facts.extract_facts(hy.conn, "recovery", llm, hy.config)
        assert result.parse_failed and result.failure_reason != "output_capacity_exceeded"
        assert len(llm.calls) == 1 and _state(hy) == before


@pytest.mark.parametrize("raw", ['{"facts":[]}', '{"facts":[{"text":"Nora shipped Cedar."}]}'])
def test_success_is_single_call_even_for_large_source(cfg, raw):
    llm = StubLLMClient(default=raw)
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy, ["Nora shipped Cedar. " * 200])
        result = facts.extract_facts(hy.conn, "recovery", llm, hy.config)
        assert not result.parse_failed and len(llm.calls) == 1


def test_historical_capacity_failure_remains_one_exact_call(cfg):
    llm = StubLLMClient(default='{"facts":[]}')
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy, ["Nora shipped Cedar. " * 200])
        original = facts.extract_facts(hy.conn, "recovery", llm, hy.config)
        with db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "recovery", original)
        before = _state(hy)
        bad = StubLLMClient(default=_OVERFLOW)
        failed = facts.reextract_fact_outcome(hy.conn, original.slice_key, bad, hy.config)
        assert failed.parse_failed and failed.failure_reason == "output_capacity_exceeded"
        assert len(bad.calls) == 1 and bad.calls[0] == llm.calls[0]
        assert failed.slice_key == original.slice_key and failed.input_hash == original.input_hash
        assert failed.source_occurrences == original.source_occurrences
        assert _state(hy) == before


@pytest.mark.parametrize("exception", [RuntimeError("provider failure"), DeadlineExceeded("deadline"), KeyboardInterrupt()])
@pytest.mark.parametrize("on_call", [1, 2])
def test_exception_on_either_call_propagates_without_writing(cfg, exception, on_call):
    class FailingClient(StubLLMClient):
        def complete(self, request):
            self.calls.append(request)
            if len(self.calls) == on_call:
                raise exception
            return _OVERFLOW

    llm = FailingClient()
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy, ["Nora shipped Cedar. " * 200])
        before = _state(hy)
        with pytest.raises(type(exception)) as caught:
            facts.extract_facts(hy.conn, "recovery", llm, hy.config)
        assert caught.value is exception and len(llm.calls) == on_call
        assert _state(hy) == before


def test_shared_deadline_blocks_second_completion_without_writes(cfg):
    clock = [0.0]

    class ExpiringClient(StubLLMClient):
        def complete(self, request):
            self.calls.append(request)
            clock[0] = 2.0
            return _OVERFLOW

    inner = ExpiringClient()
    client = DeadlineBoundLLMClient(inner, MonotonicDeadline(1.0, clock=lambda: clock[0]))
    with closing(HyMem(_config(cfg), llm=inner)) as hy:
        _seed(hy, ["Nora shipped Cedar. " * 200])
        before = _state(hy)
        with pytest.raises(DeadlineExceeded):
            facts.extract_facts(hy.conn, "recovery", client, hy.config)
        assert len(inner.calls) == 1 and _state(hy) == before


def test_capacity_recovery_implementation_participates_in_semantic_identity(cfg, monkeypatch):
    llm = StubLLMClient(default="[]")
    original = facts._extract_facts_fresh
    version = facts.facts_config_version(cfg, client=llm)
    retry_version = facts.facts_retry_policy_version(cfg, client=llm)

    def changed_helper(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(facts, "_extract_facts_fresh", changed_helper)
    assert facts.facts_config_version(cfg, client=llm) != version
    assert facts.facts_retry_policy_version(cfg, client=llm) != retry_version
