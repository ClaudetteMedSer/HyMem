"""Output limits never authorize skipping supported facts or source bytes.

Synthetic protocol tests only: these do not attest live model faithfulness.
"""

from contextlib import closing
from dataclasses import replace
import json

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import facts
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import StubLLMClient


_TEXT = "Nora shipped the Cedar package."
_ITEM = {"text": _TEXT, "date": None, "entities": ["Nora", "Cedar"]}
_OVERFLOW = {"facts": [], "complete": False}


def _config(cfg, **overrides):
    return replace(cfg, aggregation_nodes_enabled=False,
                   profile_extraction_enabled=False, **overrides)


def _seed(hy, text=_TEXT):
    message_id = hy.log_message("capacity", "user", text)
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "capacity")
    return message_id


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


def _old_outcome(hy, llm, monkeypatch, *, max_chars=None):
    with monkeypatch.context() as old:
        old.setattr(facts, "FACTS_PROMPT_VERSION", "facts.v3")
        result = facts.extract_facts(
            hy.conn, "capacity", llm, hy.config, max_chars=max_chars,
        )
        assert result is not None and not result.parse_failed
        with db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "capacity", result)
        return result


@pytest.mark.parametrize("path", ["fresh", "historical"])
@pytest.mark.parametrize("cap", [1, 2, 8, 32, 256])
def test_actual_paths_share_configured_capacity_contract(cfg, monkeypatch, path, cap):
    config = _config(cfg, dream_max_facts_per_session=cap)
    llm = StubLLMClient(default=json.dumps({"facts": [_ITEM]}))
    with closing(HyMem(config, llm=llm)) as hy:
        _seed(hy)
        prior = _old_outcome(hy, llm, monkeypatch) if path == "historical" else None
        before = _state(hy)
        result = (
            facts.extract_facts(hy.conn, "capacity", llm, config)
            if prior is None else facts.reextract_fact_outcome(
                hy.conn, prior.slice_key, llm, config,
            )
        )
        assert not result.parse_failed
        assert _state(hy) == before
        request = llm.calls[-1]
        assert f"at most {cap} facts" in request.system
        assert request == facts.build_facts_request("user: " + _TEXT, config)
        assert result.failure_reason is None
        assert f"at most {facts._MAX_FACT_CHARS} characters" in request.system
        assert f"at most {facts.FACT_MAX_ENTITIES_PER_ITEM} entities" in request.system
        assert f"at most {facts.FACT_MAX_ENTITY_CHARS} characters" in request.system
        assert "Never omit a supported fact" in request.system
        assert json.dumps(_OVERFLOW) in request.system
        assert request.response_format == "json" and request.temperature == 0.0
        assert request.max_tokens == config.dream_digest_max_tokens


@pytest.mark.parametrize("cap", [0, -1, False, 257])
def test_disabled_or_out_of_range_item_cap_is_not_a_supported_configuration(cfg, cap):
    with pytest.raises(ValueError, match="between 1 and 256"):
        replace(cfg, dream_max_facts_per_session=cap)


@pytest.mark.parametrize("path", ["fresh", "historical"])
@pytest.mark.parametrize("payload,reason", [
    (_OVERFLOW, "output_capacity_exceeded"),
    ({"facts": [_ITEM] * 9}, "output_capacity_exceeded"),
    ([_ITEM] * 9, "output_capacity_exceeded"),
    ({"facts": [_ITEM] * 9 + [{"text": "bad", "date": "never"}]}, "invalid_items"),
    ({"facts": [], "complete": 0}, "invalid_envelope"),
    ({"facts": [], "complete": "false"}, "invalid_envelope"),
    ({"facts": [], "complete": None}, "invalid_envelope"),
    ({"facts": [], "complete": True}, "invalid_envelope"),
    ({"facts": {}, "complete": False}, "invalid_envelope"),
    ({"facts": [_ITEM], "complete": False}, "invalid_envelope"),
    ({"facts": [], "complete": False, "reason": "capacity"}, "invalid_envelope"),
    ({"complete": False}, "invalid_envelope"),
    ({"facts": [_ITEM, dict(_ITEM, text="x" * 601)]}, "invalid_items"),
    ('{"facts": [], "complete": fal', "invalid_json"),
])
def test_capacity_or_invalid_output_holds_all_authority_atomically(
    cfg, monkeypatch, caplog, path, payload, reason,
):
    llm = StubLLMClient(default=json.dumps({"facts": [_ITEM]}))
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy)
        prior = _old_outcome(hy, llm, monkeypatch) if path == "historical" else None
        bad = StubLLMClient(default=payload if isinstance(payload, str) else json.dumps(payload))
        before = _state(hy)
        result = (
            facts.extract_facts(hy.conn, "capacity", bad, hy.config)
            if prior is None else facts.reextract_fact_outcome(
                hy.conn, prior.slice_key, bad, hy.config,
            )
        )
        assert result.parse_failed and result.items == []
        assert result.failure_reason == reason
        assert f"facts.output_rejected reason={reason}" in caplog.text
        assert _state(hy) == before
        with pytest.raises(ValueError, match="cannot persist a failed"):
            with db.transaction(hy.conn):
                facts.persist_facts(hy.conn, "capacity", result)
        assert _state(hy) == before
        if prior:
            for key in (
                "slice_key", "input_hash", "source_occurrences", "covered_message_id",
                "partial_message_id", "next_message_offset", "expected_generation",
            ):
                expected = 1 if key == "expected_generation" else getattr(prior, key)
                assert getattr(result, key) == expected


@pytest.mark.parametrize("payload", [[], {"facts": []}, [_ITEM], {"facts": [_ITEM]}])
@pytest.mark.parametrize("fenced", [False, True])
def test_legacy_complete_empty_and_nonempty_success_remain_exact(payload, fenced):
    raw = json.dumps(payload)
    if fenced:
        raw = "```json\n" + raw + "\n```"
    expected = facts.validate_fact_items(payload, max_items=8)
    items, reason = facts._validate_fact_response(raw, max_items=8)
    assert items == expected and reason is None


def test_overflow_sentinel_cannot_pass_the_persistence_item_validator():
    assert facts.validate_fact_items(_OVERFLOW, max_items=8) is None


def test_diagnostic_field_preserves_all_legacy_positional_constructor_slots():
    from dataclasses import fields

    legacy_names = [
        "items", "start_message_id", "covered_message_id", "parse_failed",
        "cursor_before_message_id", "cursor_before_partial_message_id",
        "cursor_before_offset", "partial_message_id", "next_message_offset",
        "slice_key", "input_hash", "source_occurrences", "caught_up",
        "publication_version", "expected_generation",
    ]
    assert [field.name for field in fields(facts.FactsExtraction)] == (
        legacy_names + ["failure_reason"]
    )
    sentinels = [object() for _ in legacy_names]
    result = facts.FactsExtraction(*sentinels)
    assert all(getattr(result, name) is value for name, value in zip(legacy_names, sentinels))
    assert result.failure_reason is None


class _CompleteSetClient:
    """Deterministic complete-set producer, never chooses a subset to fit."""

    def __init__(self, texts):
        self.texts = texts
        self.fact_calls = []
        self.delegate = StubLLMClient(default="[]")

    def phase1_producer_declaration(self):
        return self.delegate.phase1_producer_declaration()

    def memory_producer_declaration(self):
        return self.delegate.memory_producer_declaration()

    def complete(self, request):
        if "Return the JSON object of narrative facts now" not in request.user:
            return self.delegate.complete(request)
        self.fact_calls.append(request)
        assert "at most 2 facts" in request.system
        supported = [text for text in self.texts if text in request.user]
        if len(supported) > 2:
            return json.dumps(_OVERFLOW)
        return json.dumps({"facts": [
            {"text": text, "date": None, "entities": ["Nora"]}
            for text in supported
        ]})


def test_same_dream_recovers_smaller_complete_unit_without_skipping_any_tail(cfg):
    texts = [f"Nora shipped package {letter}{'x' * 115}." for letter in "ABCD"]
    llm = _CompleteSetClient(texts)
    config = _config(cfg, dream_digest_max_chars=700, dream_max_facts_per_session=2)
    with closing(HyMem(config, llm=llm)) as hy:
        ids = [_seed(hy, text) for text in texts]
        source_before = tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM messages"))
        first = hy.dream()
        assert first.fact_failures == 0 and first.facts_extracted == 2
        state = hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset,facts_retry_count,facts_quarantined FROM sessions "
            "WHERE id='capacity'"
        ).fetchone()
        assert tuple(state) == (ids[1], None, 0, 0, 0)
        assert hy.conn.execute("SELECT COUNT(*) FROM fact_extraction_outcomes").fetchone()[0] == 1
        second = hy.dream()
        assert second.fact_failures == 0 and second.facts_extracted == 2
        assert hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_retry_count FROM sessions WHERE id='capacity'"
        ).fetchone()[:] == (ids[-1], 0)
        assert len(llm.fact_calls) == 3
        excerpts = [call.user.split('"""')[1].strip("\n") for call in llm.fact_calls]
        assert excerpts[0] == "\n".join("user: " + text for text in texts)
        assert excerpts[1] == "\n".join("user: " + text for text in texts[:2])
        assert excerpts[2] == "\n".join("user: " + text for text in texts[2:])
        assert "\n".join(excerpts[1:]) == excerpts[0]
        assert hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset,facts_retry_count,facts_quarantined FROM sessions "
            "WHERE id='capacity'"
        ).fetchone()[:] == (ids[-1], None, 0, 0, 0)
        assert {row[0] for row in hy.conn.execute(
            "SELECT text FROM narrative_facts WHERE lifecycle_status='active'"
        )} == set(texts)
        assert facts.fact_session_authority_is_valid(hy.conn, "capacity")
        assert tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM messages")) == source_before
        assert hy.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_historical_overflow_then_recovery_keeps_exact_partial_unit(cfg, monkeypatch):
    llm = StubLLMClient(default=json.dumps({"facts": [_ITEM]}))
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy, _TEXT + " More source words." * 30)
        prior = _old_outcome(hy, llm, monkeypatch, max_chars=75)
        assert prior.partial_message_id is not None
        before = _state(hy)
        failed_client = StubLLMClient(default=json.dumps(_OVERFLOW))
        for _ in range(3):
            failed = facts.reextract_fact_outcome(
                hy.conn, prior.slice_key, failed_client, hy.config,
            )
            assert failed.parse_failed and failed.failure_reason == "output_capacity_exceeded"
            assert _state(hy) == before
        recovered = facts.reextract_fact_outcome(hy.conn, prior.slice_key, llm, hy.config)
        assert not recovered.parse_failed
        for name in (
            "slice_key", "input_hash", "source_occurrences", "covered_message_id",
            "partial_message_id", "next_message_offset", "cursor_before_offset",
        ):
            assert getattr(recovered, name) == getattr(prior, name)
        expected_excerpt = llm.calls[0].user
        assert all(call.user == expected_excerpt for call in failed_client.calls)
        assert llm.calls[-1].user == expected_excerpt
        assert _state(hy) == before
        with db.transaction(hy.conn):
            facts.persist_facts(hy.conn, "capacity", recovered)
        assert facts.fact_session_authority_is_valid(hy.conn, "capacity")


def test_capacity_contract_version_and_loaded_builder_change_semantic_identity(cfg, monkeypatch):
    assert facts.FACTS_PROMPT_VERSION == "facts.v4"
    llm = StubLLMClient(default="[]")
    current = facts.facts_config_version(cfg, client=llm)
    current_retry = facts.facts_retry_policy_version(cfg, client=llm)
    with monkeypatch.context() as old:
        old.setattr(facts, "FACTS_PROMPT_VERSION", "facts.v3")
        assert facts.facts_config_version(cfg, client=llm) != current
        assert facts.facts_retry_policy_version(cfg, client=llm) != current_retry
    for name, change in (
        ("FACTS_CAPACITY_TEMPLATE", facts.FACTS_CAPACITY_TEMPLATE + " Changed."),
        ("_MAX_FACT_CHARS", facts._MAX_FACT_CHARS - 1),
        ("FACT_MAX_ENTITIES_PER_ITEM", facts.FACT_MAX_ENTITIES_PER_ITEM - 1),
        ("FACT_MAX_ENTITY_CHARS", facts.FACT_MAX_ENTITY_CHARS - 1),
    ):
        before = semantic_generation_suffix("facts", llm)
        with monkeypatch.context() as changed:
            changed.setattr(facts, name, change)
            assert semantic_generation_suffix("facts", llm) != before
    original = facts.build_facts_request

    def changed_builder(*args, **kwargs):
        return original(*args, **kwargs)

    before = semantic_generation_suffix("facts", llm)
    monkeypatch.setattr(facts, "build_facts_request", changed_builder)
    assert semantic_generation_suffix("facts", llm) != before
