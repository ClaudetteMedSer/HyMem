"""Fragment presentation keeps exact evidence and precedes it with context."""

from __future__ import annotations

import json

import pytest

from hymem.extraction import chunk
from hymem.extraction.llm import StubLLMClient


def _encode(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record(content: str) -> tuple[int, str]:
    return 17, _encode({
        "content": content,
        "source_created_at": "2026-09-10T00:00:00.000Z",
        "source_message_id": 17,
        "source_peer_id": None,
        "source_record_version": "hymem-claim-source-v2",
        "source_role": "user",
        "source_session_id": "session-東京",
        "source_workspace_id": None,
    })


def _unit(content: str) -> chunk._ExtractionUnit:
    record = _record(content)
    return chunk._ExtractionUnit(text=record[1], source_records=(record,))


def _assert_exact_record(unit: chunk._ExtractionUnit, original: str) -> dict:
    record, = unit.source_records
    message_id, encoded = record
    payload = chunk._source_payload(record)
    assert payload is not None
    assert message_id == payload["source_message_id"] == 17
    assert payload["content"] == original[
        payload["source_content_start"]:payload["source_content_end"]
    ]
    baseline = _encode(payload)
    assert len(encoded) == len(baseline)
    assert json.loads(encoded) == json.loads(baseline)
    contexts = sorted(set(payload) & {
        "source_boundary_context", "source_fragment_context",
    })
    if not contexts:
        assert encoded == baseline
        return payload

    # The sole wire change is moving the top-level owned-content field last.
    # Nested context objects retain their canonical byte representation.
    expected_keys = sorted(set(payload) - {"content"}) + ["content"]
    assert list(payload) == expected_keys
    expected = "{" + ",".join(
        json.dumps(key, ensure_ascii=False) + ":" + _encode(payload[key])
        for key in expected_keys
    ) + "}"
    assert encoded == expected
    for name in contexts:
        context = payload[name]
        assert '"' + name + '":' + _encode(context) in encoded
        assert context["content"] == original[
            context["source_content_start"]:context["source_content_end"]
        ]
        if "prelude_content" in context:
            assert context["prelude_content"] == original[
                context["prelude_source_content_start"]:
                context["prelude_source_content_end"]
            ]
    return payload


@pytest.mark.parametrize("boundary", [". ", ".\n", ".\r\n", ":\n\n", ":\r\n\r\n"])
def test_preceding_prose_context_is_presented_before_owned_content(boundary: str):
    left = ("é" * 450) + " My preferred database is named next" + boundary
    right = "PostgreSQL " + ("東" * 450)
    original = left + right
    split = chunk._split_unit(_unit(original))
    assert split is not None
    first, second = split
    first_payload = _assert_exact_record(first, original)
    second_payload = _assert_exact_record(second, original)
    assert first_payload["content"] + second_payload["content"] == original
    context = second_payload["source_boundary_context"]
    cut = len(left)
    assert second_payload["source_content_start"] == cut
    assert context == {
        "version": chunk.SOURCE_BOUNDARY_CONTEXT_VERSION,
        "kind": "preceding_adjacent_prose",
        "content": original[cut - chunk._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS:cut],
        "source_content_start": cut - chunk._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS,
        "source_content_end": cut,
        "applies_through_source_content_end": cut + chunk._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS,
    }


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("prelude", ["", "# Résultats", "Our deployment targets are:"])
def test_table_header_and_prelude_context_precede_owned_rows(newline: str, prelude: str):
    introduction = prelude + newline + newline if prelude else ""
    original = introduction + newline.join([
        "| service | dependency |",
        "| --- | --- |",
        *(f"| service_{index} | database_東京_{index} |" for index in range(100)),
    ])
    split = chunk._split_unit(_unit(original))
    assert split is not None
    first, second = split
    first_payload = _assert_exact_record(first, original)
    second_payload = _assert_exact_record(second, original)
    assert first_payload["content"] + second_payload["content"] == original
    context = second_payload["source_fragment_context"]
    assert context["applies_through_source_content_end"] == len(original)
    if prelude:
        assert context["prelude_content"] == prelude + newline
    else:
        assert "prelude_content" not in context


def test_context_free_fragment_is_byte_identical_and_deterministic():
    original = "Before. Café 東京 after."
    payload = json.loads(_record(original)[1])
    payload["metadata"] = {"z": [3, {"b": "é", "a": True}], "a": None}
    reordered = {key: value for key, value in reversed(tuple(payload.items()))}
    first = chunk._fragment_record((17, json.dumps(payload)), start=0, end=7)
    second = chunk._fragment_record((17, json.dumps(reordered)), start=0, end=7)
    assert first == second
    expected = {
        **payload,
        "content": original[:7],
        "source_record_version": "hymem-claim-source-fragment-v2",
        "source_content_start": 0,
        "source_content_end": 7,
    }
    assert first == (17, _encode(expected))


def test_recursive_prose_fragments_preserve_context_and_exact_source():
    first = ("é" * 450) + " Original cue finishes here. "
    middle = "Owned prefix " + ("p" * 100) + ". "
    last = "Final continuation " + ("東" * 250)
    original = first + middle + last
    split = chunk._split_unit(_unit(original))
    assert split is not None
    left, right = split
    parent = _assert_exact_record(right, original)
    assert parent["source_content_start"] == len(first)
    recursive = chunk._split_unit(right)
    assert recursive is not None
    second_left, second_right = recursive
    left_payload = _assert_exact_record(left, original)
    middle_payload = _assert_exact_record(second_left, original)
    last_payload = _assert_exact_record(second_right, original)
    assert middle_payload["source_boundary_context"] == parent["source_boundary_context"]
    assert last_payload["source_boundary_context"]["source_content_end"] == len(first + middle)
    assert left_payload["content"] + middle_payload["content"] + last_payload["content"] == original


def test_recursive_table_fragments_preserve_original_context():
    original = "# Déploiements\n\n" + "\n".join([
        "| service | dependency |", "| --- | --- |",
        *(f"| service_{index} | database_東京_{index} |" for index in range(100)),
    ])
    split = chunk._split_unit(_unit(original))
    assert split is not None
    left, right = split
    parent = _assert_exact_record(right, original)
    recursive = chunk._split_unit(right)
    assert recursive is not None
    child_payloads = [_assert_exact_record(child, original) for child in recursive]
    assert all(
        payload["source_fragment_context"] == parent["source_fragment_context"]
        for payload in child_payloads
    )
    assert "".join(payload["content"] for payload in child_payloads) == parent["content"]
    _assert_exact_record(left, original)


def test_full_source_record_reaches_provider_byte_identically():
    # Whitespace and caller ordering need not be canonical: full records are
    # forwarded unchanged, and only the private fragment serializer is edited.
    record = _record("No durable claim here. 東京")
    payload = json.loads(record[1])
    record = (17, json.dumps(dict(reversed(tuple(payload.items()))), ensure_ascii=False))
    client = StubLLMClient(default='{"triples":[],"markers":[],"complete":true}')
    result = chunk.extract_chunk(client, "unused rendering", source_records=(record,))
    assert result.failed is False
    assert client.calls
    assert all(request.user.split('"""', 2)[1].strip() == record[1] for request in client.calls)
