"""Output limits never authorize skipping supported facts or source bytes.

Synthetic protocol tests only: these do not attest live model faithfulness.
"""

from contextlib import closing
from dataclasses import replace
import json
import re

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
        assert request.max_tokens == config.facts_extraction_max_tokens


def test_fact_token_budget_changes_only_fact_producer(cfg):
    larger = replace(cfg, facts_extraction_max_tokens=6144)
    digest_only = replace(cfg, dream_digest_max_tokens=6144)
    llm = StubLLMClient(default=json.dumps({"facts": [_ITEM]}))
    assert facts.build_facts_request("user: " + _TEXT, larger).max_tokens == 6144
    assert facts.facts_config_version(larger, client=llm) != (
        facts.facts_config_version(cfg, client=llm)
    )
    assert larger.dream_digest_max_tokens == cfg.dream_digest_max_tokens
    assert facts.facts_config_version(digest_only, client=llm) == (
        facts.facts_config_version(cfg, client=llm)
    )
    assert facts.build_facts_request("user: " + _TEXT, digest_only).max_tokens == 3072


@pytest.mark.parametrize("tokens", [0, -1, False, 32769])
def test_invalid_fact_token_budget_is_rejected(cfg, tokens):
    with pytest.raises(ValueError, match="facts_extraction_max_tokens"):
        replace(cfg, facts_extraction_max_tokens=tokens)


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


class _DenseSourceClient:
    """Return the complete fact set only when it fits the eight-item cap."""

    def __init__(self):
        self.delegate = StubLLMClient(default="[]")
        self.fact_calls = []

    def phase1_producer_declaration(self):
        return self.delegate.phase1_producer_declaration()

    def memory_producer_declaration(self):
        return self.delegate.memory_producer_declaration()

    def complete(self, request):
        if "Return the JSON object of narrative facts now" not in request.user:
            return self.delegate.complete(request)
        self.fact_calls.append(request)
        found = re.findall(r"Nora shipped parcel \d{2}\.", request.user)
        if len(found) > 8:
            return json.dumps(_OVERFLOW)
        return json.dumps({"facts": [
            {"text": text, "date": None, "entities": ["Nora"]}
            for text in found
        ]})


class _AlwaysOverflowClient:
    def __init__(self):
        self.delegate = StubLLMClient(default="[]")
        self.fact_calls = []

    def phase1_producer_declaration(self):
        return self.delegate.phase1_producer_declaration()

    def memory_producer_declaration(self):
        return self.delegate.memory_producer_declaration()

    def complete(self, request):
        if "Return the JSON object of narrative facts now" not in request.user:
            return self.delegate.complete(request)
        self.fact_calls.append(request)
        return json.dumps(_OVERFLOW)


def test_scheduled_capacity_retry_changes_held_source_and_preserves_all_facts(cfg):
    llm = _DenseSourceClient()
    config = _config(cfg, dream_digest_max_chars=12000)
    source = " ".join(
        f"Nora shipped parcel {index:02d}. " + "Context " * 12
        for index in range(20)
    )
    with closing(HyMem(config, llm=llm)) as hy:
        message_id = _seed(hy, source)
        first = hy.dream()
        assert first.fact_failures == 1
        assert len(llm.fact_calls) == 2
        first_pair = [call.user for call in llm.fact_calls]
        assert first_pair[0] != first_pair[1]
        assert hy.conn.execute(
            "SELECT facts_retry_count,facts_cursor_message_id FROM sessions "
            "WHERE id='capacity'"
        ).fetchone()[:] == (1, None)

        second = hy.dream()
        assert second.fact_failures == 0
        assert llm.fact_calls[2].user not in first_pair
        for _ in range(10):
            state = hy.conn.execute(
                "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
                "facts_retry_count FROM sessions WHERE id='capacity'"
            ).fetchone()
            if state[:] == (message_id, None, 0):
                break
            hy.dream()
        else:
            pytest.fail("capacity retry did not consume the complete source")
        expected = {f"Nora shipped parcel {index:02d}." for index in range(20)}
        actual = {row[0] for row in hy.conn.execute(
            "SELECT text FROM narrative_facts WHERE lifecycle_status='active'"
        )}
        assert actual == expected
        assert facts.fact_session_authority_is_valid(hy.conn, "capacity")
        assert hy.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_retry_budget_uses_stable_first_turn_of_many_short_turns(cfg):
    llm = StubLLMClient(default="[]")
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        texts = [f"turn {index:02d}: " + "context " * 25 for index in range(20)]
        ids = [_seed(hy, text) for text in texts]
        first_turn_chars = len("user: " + texts[0])
        assert len("\n".join("user: " + text for text in texts)) < (
            hy.config.dream_digest_max_chars
        )
        budget = facts.facts_retry_max_chars(
            hy.conn, "capacity", since_message_id=None,
            partial_message_id=None, start_offset=0,
            configured_max=hy.config.dream_digest_max_chars, retry_count=1,
        )
        assert budget == first_turn_chars
        result = facts.extract_facts(
            hy.conn, "capacity", llm, hy.config, max_chars=budget,
        )
        assert result is not None and not result.parse_failed
        assert result.covered_message_id == ids[0]
        assert result.partial_message_id is None


def test_appended_turn_cannot_expand_held_capacity_retry(cfg):
    llm = _AlwaysOverflowClient()
    with closing(HyMem(_config(cfg, dream_digest_max_chars=12000), llm=llm)) as hy:
        first_id = _seed(hy, "Context " * 120)
        assert hy.dream().fact_failures == 1
        assert len(llm.fact_calls) == 2
        before_retry = [
            len(call.user.split('"""', 2)[1].strip("\n"))
            for call in llm.fact_calls
        ]
        assert before_retry[0] > before_retry[1] >= 256
        _seed(hy, "APPENDED_ONLY_MARKER " + "new context " * 1000)

        assert hy.dream().fact_failures == 1
        assert len(llm.fact_calls) == 3
        retried = llm.fact_calls[-1].user.split('"""', 2)[1].strip("\n")
        assert len(retried) <= before_retry[1]
        assert "APPENDED_ONLY_MARKER" not in retried
        state = hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_retry_count FROM sessions WHERE id='capacity'"
        ).fetchone()
        assert state[:] == (None, None, 2)
        assert first_id is not None


def test_changed_fact_generation_resets_retry_before_source_aware_shrink(
    cfg, monkeypatch,
):
    llm = _DenseSourceClient()
    source = " ".join(
        f"Nora shipped parcel {index:02d}. " + "Context " * 12
        for index in range(20)
    )
    with closing(HyMem(_config(cfg), llm=llm)) as hy:
        _seed(hy, source)
        assert hy.dream().fact_failures == 1
        assert len(llm.fact_calls) == 2
        old_key = hy.conn.execute(
            "SELECT facts_retry_config_version FROM sessions WHERE id='capacity'"
        ).fetchone()[0]
        with monkeypatch.context() as changed:
            changed.setattr(facts, "FACTS_PROMPT_VERSION", "facts.v5")
            # The old failure cannot be attributed to the new producer. Its
            # first attempt repeats the full source, then records a new key.
            assert hy.dream().fact_failures == 1
            assert [call.user for call in llm.fact_calls[2:4]] == [
                call.user for call in llm.fact_calls[:2]
            ]
            count, new_key = hy.conn.execute(
                "SELECT facts_retry_count,facts_retry_config_version "
                "FROM sessions WHERE id='capacity'"
            ).fetchone()
            assert count == 1 and new_key != old_key
            assert hy.dream().fact_failures == 0
            assert llm.fact_calls[4].user not in [
                call.user for call in llm.fact_calls[:4]
            ]


def test_capacity_fallback_publishes_smaller_complete_units_without_skipping_tail(cfg):
    texts = [f"Nora shipped package {letter}{'x' * 115}." for letter in "ABCD"]
    llm = _CompleteSetClient(texts)
    config = _config(cfg, dream_digest_max_chars=700, dream_max_facts_per_session=2)
    with closing(HyMem(config, llm=llm)) as hy:
        ids = [_seed(hy, text) for text in texts]
        source_before = tuple(tuple(row) for row in hy.conn.execute("SELECT * FROM messages"))
        first = hy.dream()
        assert first.fact_failures == 0 and first.facts_extracted == 2
        assert hy.conn.execute(
            "SELECT facts_cursor_message_id,facts_retry_count FROM sessions WHERE id='capacity'"
        ).fetchone()[:] == (ids[1], 0)
        second = hy.dream()
        assert second.fact_failures == 0 and second.facts_extracted == 2
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
