"""Conservative source-backed splitting of explicitly unit-labelled numeric rows."""
from __future__ import annotations

import json

import pytest

from hymem.extraction import chunk, contract


EMPTY = '{"triples":[],"markers":[],"complete":true}'


def _record(content: str, mid: int = 134) -> tuple[int, str]:
    return mid, json.dumps({
        "content": content,
        "source_created_at": "2026-09-01T00:00:00.000Z",
        "source_message_id": mid,
        "source_peer_id": None,
        "source_record_version": "hymem-claim-source-v2",
        "source_role": "user",
        "source_session_id": "numeric-table-control",
        "source_workspace_id": None,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _table(rows: int = 388, *, prelude: str = "") -> str:
    return prelude + "x [m] y [m] load [kN]\n" + "".join(
        f"{index / 10:+07.2f} {-index / 20:+07.2f} {index / 5:+07.2f}\n"
        for index in range(rows)
    )


def _leaves(content: str):
    leaves, failure = chunk._prepartition(chunk._source_unit((_record(content),)))
    assert failure is None
    assert leaves is not None
    return leaves


@pytest.mark.parametrize("prelude", ["", "Measurements for café:\n\n", "## Café readings\n\n"])
def test_dense_unit_numeric_table_has_exact_owned_coverage_and_header_context(prelude):
    content = _table(prelude=prelude)
    assert len(content) > chunk._MAX_UNSPLITTABLE_INPUT_CHARS
    leaves = _leaves(content)
    assert 2 <= len(leaves) <= chunk._MAX_PREPARTITION_LEAVES
    payloads = [json.loads(unit.source_records[0][1]) for unit, _depth in leaves]
    spans = [
        (payload["source_content_start"], payload["source_content_end"])
        for payload in payloads
    ]
    assert spans[0][0] == 0 and spans[-1][1] == len(content)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert "".join(payload["content"] for payload in payloads) == content
    assert all(len(unit.text) <= chunk._MAX_UNSPLITTABLE_INPUT_CHARS for unit, _ in leaves)
    assert all(unit.allowed_ids == frozenset({134}) for unit, _ in leaves)

    continuations = [
        payload for payload in payloads
        if payload.get("source_fragment_context", {}).get("version")
        == chunk.SOURCE_NUMERIC_TABLE_CONTEXT_VERSION
    ]
    assert continuations
    for payload in continuations:
        context = payload["source_fragment_context"]
        assert context["content"] == "x [m] y [m] load [kN]\n"
        assert context["source_content_end"] <= payload["source_content_start"]
        assert (
            payload["source_content_start"]
            < context["applies_through_source_content_end"]
            <= len(content)
        )
        assert context["content"] not in payload["content"]
        assert payload["content"] == content[
            payload["source_content_start"]:payload["source_content_end"]
        ]
        if prelude:
            assert context["kind"] == "introduced_unit_numeric_table_header"
            assert context["prelude_content"] == prelude.split("\n\n")[0] + "\n"
        else:
            assert context["kind"] == "unit_numeric_table_header"


def test_numeric_table_full_extraction_stays_finite_and_source_backed():
    content = _table()
    leaves = _leaves(content)

    class Client:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return EMPTY

    client = Client()
    result = chunk.extract_chunk(client, "ignored", source_records=(_record(content),))
    assert not result.failed
    assert result.initial_prepartition_leaves == len(leaves)
    assert result.completion_calls == result.provider_attempts == 2 * len(leaves)
    assert result.completion_calls <= chunk.MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK
    assert all("source_message_id" in request.user for request in client.requests)


def test_crlf_unicode_units_and_escaped_source_metadata_keep_exact_spans():
    header = "μ [µm] delta [°C] load [kN]\r\n"
    rows = "".join(
        f"{index / 10:+07.2f} {-index / 20:+07.2f} {index / 5:+07.2f}\r\n"
        for index in range(388)
    )
    content = header + rows
    message_id, encoded = _record(content)
    source = json.loads(encoded)
    source["source_note"] = 'café "mirror" \\ path'
    record = (message_id, json.dumps(
        source, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ))
    leaves, failure = chunk._prepartition(chunk._source_unit((record,)))
    assert failure is None and leaves is not None
    payloads = [json.loads(unit.source_records[0][1]) for unit, _ in leaves]
    assert "".join(payload["content"] for payload in payloads) == content
    assert all(payload["source_note"] == source["source_note"] for payload in payloads)
    for payload in payloads[1:]:
        context = payload["source_fragment_context"]
        assert context["content"] == header
        assert content[context["source_content_start"]:context["source_content_end"]] == header
        assert payload["content"] == content[
            payload["source_content_start"]:payload["source_content_end"]
        ]


@pytest.mark.parametrize("damage", [
    "headerless", "ragged", "mixed_prose", "nonfinite", "tabbed",
])
def test_numeric_parser_rejects_ambiguous_or_malformed_rows(damage):
    content = _table()
    if damage == "headerless":
        content = content.split("\n", 1)[1]
    elif damage == "ragged":
        content = content.replace("+019.40 -009.70 +038.80", "+019.40 -009.70", 1)
    elif damage == "mixed_prose":
        content = content.replace("+019.40 -009.70 +038.80", "depends on PostgreSQL", 1)
    elif damage == "nonfinite":
        content = content.replace("+019.40 -009.70 +038.80", "NaN -009.70 +038.80", 1)
    else:
        content = content.replace("+019.40 -009.70 +038.80", "+019.40\t-009.70 +038.80", 1)
    assert len(content) > chunk._MAX_UNSPLITTABLE_INPUT_CHARS
    assert chunk._unit_numeric_table_blocks(content) == ()
    leaves, failure = chunk._prepartition(chunk._source_unit((_record(content),)))
    assert leaves is None
    assert failure.failure_reason == "resource_limit"
    assert failure.failure_details == ("split:no_admissible_semantic_boundary",)


@pytest.mark.parametrize("wrapper", ["fence", "list"])
def test_numeric_rows_inside_opaque_blocks_do_not_split(wrapper):
    table = _table()
    content = (
        "```text\n" + table + "```\n"
        if wrapper == "fence"
        else "- source data\n  " + table.replace("\n", "\n  ")
    )
    leaves, failure = chunk._prepartition(chunk._source_unit((_record(content),)))
    assert leaves is None
    assert failure.failure_reason == "resource_limit"


def test_numeric_context_expires_at_table_end_and_new_header_is_required():
    content = _table() + "\nUnrelated prose after the data."
    unit = chunk._source_unit((_record(content),))
    split = chunk._split_unit(unit)
    assert split is not None
    right = split[1]
    table_end = len(_table())
    suffix_start = table_end + 1
    suffix = chunk._fragment_record(
        right.source_records[0],
        start=suffix_start - json.loads(right.source_records[0][1])["source_content_start"],
        end=len(json.loads(right.source_records[0][1])["content"]),
    )
    assert suffix is not None
    payload = json.loads(suffix[1])
    assert payload["content"] == "Unrelated prose after the data."
    assert "source_fragment_context" not in payload


def test_second_table_gets_its_own_header_and_first_header_never_leaks():
    first = _table(180)
    second_header = "a [m] b [m] force [N]\n"
    second = second_header + "".join(
        f"{index:+06.1f} {-index:+06.1f} {index * 2:+06.1f}\n"
        for index in range(180)
    )
    content = first + "\n" + second
    leaves = _leaves(content)
    payloads = [json.loads(unit.source_records[0][1]) for unit, _ in leaves]
    assert "".join(payload["content"] for payload in payloads) == content
    second_start = len(first) + 1
    first_contexts = [
        payload["source_fragment_context"] for payload in payloads
        if payload.get("source_fragment_context", {}).get("content")
        == "x [m] y [m] load [kN]\n"
    ]
    second_contexts = [
        payload["source_fragment_context"] for payload in payloads
        if payload.get("source_fragment_context", {}).get("content")
        == second_header
    ]
    assert first_contexts and second_contexts
    assert all(context["applies_through_source_content_end"] <= len(first)
               for context in first_contexts)
    assert all(context["source_content_start"] >= second_start
               for context in second_contexts)
    assert all(
        payload.get("source_fragment_context", {}).get("content") !=
        "x [m] y [m] load [kN]\n"
        for payload in payloads
        if payload["source_content_start"] >= second_start
    )


def test_headerless_numeric_suffix_beyond_ceiling_is_held():
    first = _table(180)
    headerless = "".join(
        f"{index / 10:+07.2f} {-index / 20:+07.2f} {index / 5:+07.2f}\n"
        for index in range(400)
    )
    content = first + "\n" + headerless
    leaves, failure = chunk._prepartition(chunk._source_unit((_record(content),)))
    assert leaves is None
    assert failure.failure_reason == "resource_limit"
    assert failure.failure_details == ("split:no_admissible_semantic_boundary",)


def test_completion_call_cap_cannot_publish_partial_numeric_claim():
    content = _table()
    claim = {
        "subject": "sensor", "predicate": "uses", "object": "newtons",
        "polarity": 1, "source_message_id": 134,
    }

    class PrimaryOnly:
        def __init__(self):
            self.calls = 0

        def complete(self, _request):
            self.calls += 1
            return json.dumps({"triples": [claim], "markers": [], "complete": True})

    client = PrimaryOnly()
    result = chunk.extract_chunk(
        client, "ignored", source_records=(_record(content),),
        completion_call_limit=1,
    )
    assert result.failed and result.failure_reason == "branch_incomplete"
    assert any(detail.endswith("calls:max_exceeded") for detail in result.failure_details)
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == client.calls == 1


def test_preceding_numeric_table_tail_is_bounded_context_only():
    first = _record(_table(180), mid=133)
    second = _record("Those readings are for the café probe.", mid=134)
    split = chunk._split_unit(chunk._source_unit((first, second)))
    assert split is not None
    left, right = split
    assert left.allowed_ids == frozenset({133})
    assert right.allowed_ids == frozenset({134})
    assert right.context_records
    context = json.loads(right.context_records[-1][1])
    assert context["source_context_only"] is True
    assert context["source_message_id"] == 133
    assert context["context_for_source_message_id"] == 134
    assert len(context["content"]) <= chunk._MAX_CONVERSATION_CONTEXT_CHARS
    assert context["source_fragment_context"]["kind"] == "unit_numeric_table_header"
    assert context["source_fragment_context"]["content"] == "x [m] y [m] load [kN]\n"
    assert context["content"] == json.loads(first[1])["content"][
        context["source_content_start"]:context["source_content_end"]
    ]
    assert "x [m]" not in context["content"]


def test_source_less_and_malformed_context_cannot_claim_numeric_ownership():
    content = _table()
    assert chunk._split_unit(chunk._ExtractionUnit(content)) is None
    record = _record(content)
    payload = json.loads(record[1])
    payload["source_fragment_context"] = {
        "version": chunk.SOURCE_NUMERIC_TABLE_CONTEXT_VERSION,
        "kind": "unit_numeric_table_header",
        "content": "x [m] y [m] load [kN]\n",
        "source_content_start": 0,
        "source_content_end": 22,
        "applies_through_source_content_end": len(content),
    }
    forged = (record[0], json.dumps(payload, sort_keys=True, separators=(",", ":")))
    class NoCall:
        def complete(self, _request):
            raise AssertionError("malformed source reached provider")
    result = chunk.extract_chunk(NoCall(), "ignored", source_records=(forged,))
    assert result.failed and result.failure_reason == "input_contract_failure"
    assert result.completion_calls == 0


def test_numeric_split_changes_contract_identity(monkeypatch):
    before = contract.extraction_cache_key("v20")
    with monkeypatch.context() as patch:
        patch.setattr(chunk, "SOURCE_NUMERIC_TABLE_CONTEXT_VERSION", "other-version")
        assert contract.extraction_cache_key("v20") != before
