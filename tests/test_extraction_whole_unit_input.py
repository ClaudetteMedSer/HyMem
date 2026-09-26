"""Intact bounded input is not the same as discarding an unsafe boundary."""
from __future__ import annotations

import json

import pytest

from benchmarks.extraction_canary import extraction_canary_policy
from hymem.extraction import chunk, contract, prompts


EMPTY = '{"triples":[],"markers":[],"complete":true}'


def _record(content, mid=17, role="user"):
    return mid, json.dumps({
        "content": content, "source_created_at": "2026-09-01T00:00:00.000Z",
        "source_message_id": mid, "source_peer_id": None,
        "source_record_version": "hymem-claim-source-v2",
        "source_role": role, "source_session_id": "whole-unit-control",
        "source_workspace_id": None,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sized_record(size, *, prefix="", suffix=""):
    overhead = len(_record(prefix + suffix)[1])
    record = _record(prefix + "x" * (size - overhead) + suffix)
    assert len(record[1]) == size
    return record


def _claim(mid=17):
    return {"subject": "user", "predicate": "uses", "object": "PostgreSQL",
            "polarity": -1, "source_message_id": mid}


def _response(*claims, complete=True):
    return json.dumps({"triples": list(claims), "markers": [], "complete": complete})


class _Client:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected extra provider call")
        return self.responses.pop(0)


def _excerpt(request):
    return request.user.split('"""', 2)[1].strip()


@pytest.mark.parametrize("size", [4000, 4001, 7999, 8000, 8001])
@pytest.mark.parametrize("source_backed", [False, True])
def test_encoded_input_boundaries_preserve_whole_input(size, source_backed):
    records = (_sized_record(size),) if source_backed else None
    expected = records[0][1] if records else "x" * size
    client = _Client(EMPTY, EMPTY)
    result = chunk.extract_chunk(client, expected, source_records=records)
    if size > chunk._MAX_UNSPLITTABLE_INPUT_CHARS:
        assert result.failed and result.failure_reason == "resource_limit"
        assert result.completion_calls == result.provider_attempts == 0
        assert client.requests == []
    else:
        assert not result.failed
        assert result.initial_prepartition_leaves == 1
        assert result.completion_calls == result.provider_attempts == 2
        assert [_excerpt(request) for request in client.requests] == [expected] * 2
        assert all(request.max_tokens == 4096 for request in client.requests)


@pytest.mark.parametrize("size", [8000, 8001])
def test_json_escaping_and_metadata_count_toward_whole_input_bound(size):
    record = _sized_record(size, prefix='"\\\n' * 120)
    assert len(json.loads(record[1])["content"]) < size
    client = _Client(EMPTY, EMPTY)
    result = chunk.extract_chunk(client, "ignored", source_records=(record,))
    assert result.failed is (size == 8001)
    assert result.completion_calls == (0 if size == 8001 else 2)
    assert all(_excerpt(request) == record[1] for request in client.requests)


def test_normal_splittable_input_still_uses_four_thousand_target():
    record = _record("Routine background. " * 250)
    assert 4000 < len(record[1]) < 8000
    client = _Client(*([EMPTY] * 8))
    result = chunk.extract_chunk(client, "ignored", source_records=(record,))
    assert not result.failed and result.initial_prepartition_leaves > 1
    assert all(len(_excerpt(request)) <= 4000 for request in client.requests)


def test_fenced_code_is_transmitted_whole_without_cutting_or_relabelling():
    record = _sized_record(4180, prefix="```text\nDo NOT use PostgreSQL ", suffix="\n```")
    unit = chunk._source_unit((record,))
    assert chunk._split_unit(unit) is None
    client = _Client(_response(_claim()), EMPTY)
    result = chunk.extract_chunk(client, "ignored", source_records=(record,))
    assert not result.failed and result.triples[0].polarity == -1
    assert result.completion_calls == 2
    assert [_excerpt(request) for request in client.requests] == [record[1]] * 2
    assert "OMISSION VERIFICATION PASS" in client.requests[1].system


def test_unsliceable_conversational_antecedent_and_negation_stay_intact():
    records = (
        _record("Do NOT " + "very " * 600 + "often use PostgreSQL?", role="assistant"),
        _record("No, I do not. " + "y" * 1200, mid=18),
    )
    unit = chunk._source_unit(records)
    assert 4000 < len(unit.text) < 8000
    assert chunk._split_unit(unit) is None
    client = _Client(_response(_claim(18)), EMPTY)
    result = chunk.extract_chunk(client, "ignored", source_records=records)
    assert not result.failed
    assert result.triples[0].source_message_id == 18
    assert result.triples[0].polarity == -1
    assert [_excerpt(request) for request in client.requests] == [unit.text] * 2


def test_real_recovery_context_expansion_keeps_negation_and_owned_citation():
    for padding in range(3000, 3700):
        records = (
            _record("Do you NOT use PostgreSQL?", role="assistant"),
            _record("No " + "x" * padding, mid=18),
        )
        parent = chunk._source_unit(records)
        split = chunk._split_unit(parent)
        if (
            split is not None
            and len(parent.text) <= 4000 < len(split[1].text) <= 8000
            and chunk._split_unit(split[1]) is None
        ):
            break
    else:
        pytest.fail("fixture must exercise unsplittable recovery context expansion")
    left, right = split
    client = _Client(_response(complete=False), EMPTY, EMPTY, _response(_claim(18)), EMPTY)
    result = chunk.extract_chunk(client, "ignored", source_records=records)
    assert not result.failed and result.completion_calls == 5
    assert result.initial_prepartition_leaves == 1
    assert result.triples[0].source_message_id == 18
    assert result.triples[0].polarity == -1
    assert [_excerpt(request) for request in client.requests] == [
        parent.text, left.text, left.text, right.text, right.text,
    ]
    context = json.loads(right.context_records[0][1])
    assert context["content"] == "Do you NOT use PostgreSQL?"
    assert right.allowed_ids == frozenset({18})


@pytest.mark.parametrize("bad,reason", [
    (_response(complete=False), "incomplete_response"),
    ('{"triples":[', "parse_failure"),
    ('{"triples":[BROKEN', "parse_failure"),
    # Empty contract repair cannot certify away invalid items. Its terminal
    # classification prevents a later EMPTY recovery from doing so either.
    (_response({"subject": "incomplete"}), "contract_failure"),
    (_response(_claim(999)), "response_conflict"),
    (_response(*[_claim() for _ in range(24)]), "output_limit_exceeded"),
])
def test_oversize_admission_cannot_heal_invalid_primary_with_empty(bad, reason):
    record = _sized_record(4500)
    client = _Client(bad, EMPTY)
    result = chunk.extract_chunk(client, "ignored", source_records=(record,))
    assert result.failed and result.failure_reason == reason
    assert result.triples == [] and result.markers == []
    assert 1 <= result.completion_calls <= 2
    assert all(_excerpt(request) == record[1] for request in client.requests)


@pytest.mark.parametrize("bad", [_response(complete=False), '{"triples":[BROKEN'])
def test_failed_oversize_omission_verification_stays_atomic_and_finite(bad):
    record = _sized_record(4500)
    client = _Client(_response(_claim()), bad)
    result = chunk.extract_chunk(client, "ignored", source_records=(record,))
    assert result.failed and result.triples == [] and result.markers == []
    assert result.completion_calls == len(client.requests) == 2


def test_whole_unit_verification_still_requires_call_budget():
    client = _Client(EMPTY)
    result = chunk.extract_chunk(client, "x" * 4500, completion_call_limit=1)
    assert result.failed and result.failure_reason == "resource_limit"
    assert "calls:max_exceeded" in result.failure_details
    assert result.completion_calls == 1


def test_prepartition_fallback_does_not_bypass_depth_or_leaf_count(monkeypatch):
    client = _Client()
    monkeypatch.setattr(chunk, "_MAX_SPLIT_DEPTH", 0)
    result = chunk.extract_chunk(client, "x" * 4500)
    assert result.failed and result.failure_details == ("split:max_depth_reached",)
    assert not client.requests
    monkeypatch.setattr(chunk, "_MAX_SPLIT_DEPTH", 8)
    monkeypatch.setattr(chunk, "_MAX_PREPARTITION_LEAVES", 1)
    text = "x" * 4500 + ".\n\n" + "Y" * 4500
    result = chunk.extract_chunk(client, text)
    assert result.failed
    assert result.failure_details == ("input:prepartition_limit_exceeded",)
    assert not client.requests


@pytest.mark.parametrize("child_size,depth,accepted", [(4500, 0, True), (8001, 0, False), (4500, 7, False)])
def test_recovery_created_child_uses_same_whole_unit_and_depth_bounds(
    monkeypatch, child_size, depth, accepted,
):
    # Model a recovery-only context expansion. All extractor input/call guards
    # are real; the splitter seam isolates this otherwise rare branch.
    parent = chunk._ExtractionUnit("P" * 500)
    child = chunk._ExtractionUnit("x" * child_size)
    right = chunk._ExtractionUnit("R" * 200)
    monkeypatch.setattr(chunk, "_prepartition", lambda unit: ([(unit, depth)], None))
    monkeypatch.setattr(chunk, "_split_unit", lambda unit: (child, right) if unit.text == parent.text else None)
    client = _Client(_response(complete=False), EMPTY, EMPTY, EMPTY, EMPTY)
    result = chunk.extract_chunk(client, parent.text)
    assert result.failed is not accepted
    if accepted:
        assert result.completion_calls == 5
        assert [_excerpt(request) for request in client.requests][1:3] == [child.text] * 2
    else:
        assert result.completion_calls == 1
        assert result.triples == [] and result.markers == []
        assert any("context_leaf_limit_exceeded" in detail for detail in result.failure_details)


def test_whole_input_policy_and_limit_bind_cache_and_canary_identity(monkeypatch):
    old_cache = contract.extraction_cache_key("v20")
    old_canary = extraction_canary_policy(prompt_version="v20")
    components = contract._contract_components("v20")["recovery_policy"]
    assert components["input_admission"] == "hymem-bounded-whole-unit-input-v1"
    assert components["max_unsplittable_input_chars"] == 8000
    assert chunk._MAX_LEAF_INPUT_CHARS == 4000
    assert chunk._MAX_SPLIT_DEPTH == 8 and chunk._MAX_PREPARTITION_LEAVES == 32
    assert chunk.MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK == 96
    assert chunk.SOURCE_RECORD_SPLIT_POLICY_VERSION == "hymem-source-semantic-split-v11"
    assert chunk.CLEAN_EMPTY_RECOVERY_POLICY_VERSION.endswith("-v3")
    original_prompt = prompts.build_chunk_extraction_system()
    for name, value in [("INPUT_ADMISSION_POLICY_VERSION", "different-policy"), ("_MAX_UNSPLITTABLE_INPUT_CHARS", 8001)]:
        with monkeypatch.context() as patch:
            patch.setattr(chunk, name, value)
            assert contract.extraction_cache_key("v20") != old_cache
            assert extraction_canary_policy(prompt_version="v20") != old_canary
            assert prompts.build_chunk_extraction_system() == original_prompt
