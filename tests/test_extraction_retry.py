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
import time
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
    record_unrecoverable_chunk_losses,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.chunk import extract_chunk
from hymem.extraction import chunk as chunk_extraction
from hymem.extraction.contract import extraction_cache_key
from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.extraction.markers import Marker
from hymem.extraction.producer import phase1_generation_binding
from hymem.extraction.triples import Triple


def _complete(payload: dict) -> str:
    return json.dumps({**payload, "complete": True})


def _source_record(message_id: int, content: str) -> tuple[int, str]:
    return message_id, json.dumps({
        "content": content,
        "source_created_at": "2026-09-05T00:00:00.000Z",
        "source_message_id": message_id,
        "source_peer_id": None,
        "source_record_version": "hymem-claim-source-v2",
        "source_role": "user",
        "source_session_id": "s",
        "source_workspace_id": None,
    }, sort_keys=True, separators=(",", ":"))


# --- the failed flag itself -------------------------------------------------


def test_unparseable_reply_is_flagged_failed():
    result = extract_chunk(StubLLMClient(default="Sorry, no JSON for you."), "x")
    assert result.triples == [] and result.markers == []
    assert result.failed is True
    assert result.completion_calls == result.provider_attempts == 1


def test_json_grammar_controls_truncation_recovery_and_call_budget():
    # The quoted closing brace used to hide the genuinely unclosed root from
    # the raw character counter.  A legal prefix receives the one bounded
    # terminal recovery call even though this tiny source cannot split.
    cut = '{"triples": [{"note": "quoted }"}'
    cut_llm = StubLLMClient(default=cut)
    cut_result = extract_chunk(cut_llm, "x")
    assert cut_result.failed is True
    assert "response:truncated_json" in cut_result.failure_details
    assert cut_result.completion_calls == cut_result.provider_attempts == 2

    # Conversely, an opening brace inside a string cannot turn an impossible
    # JSON token into truncation.  Invalid syntax stays on the one-call,
    # fail-closed parse path rather than consuming split/retry budget.
    malformed = '{"triples": [{"note": "quoted {"}], BROKEN'
    malformed_llm = StubLLMClient(default=malformed)
    malformed_result = extract_chunk(malformed_llm, "x")
    assert malformed_result.failed is True
    assert "response:invalid_json" in malformed_result.failure_details
    assert malformed_result.completion_calls == malformed_result.provider_attempts == 1


def test_provider_error_counts_the_raised_attempt_exactly():
    llm = StubLLMClient(default=None)
    result = extract_chunk(llm, "x")
    assert result.failed is True
    assert result.failure_reason == "call_failure"
    assert result.completion_calls == result.provider_attempts == 1
    assert len(llm.calls) == 1


class _MeteredRetryingLLM:
    """Expose three HTTP attempts behind each logical completion call."""

    def __init__(self, *, fail: bool = False):
        self.request_attempts = 0
        self.completion_calls = 0
        self.fail = fail

    def complete(self, _request):
        self.completion_calls += 1
        self.request_attempts += 3
        if self.fail:
            raise RuntimeError("provider retries exhausted")
        return _complete({
            "triples": [{
                "subject": "app",
                "predicate": "uses",
                "object": "PostgreSQL",
                "polarity": 1,
            }],
            "markers": [],
        })


@pytest.mark.parametrize("fail", [False, True])
def test_request_attempt_counter_distinguishes_internal_retries(fail):
    llm = _MeteredRetryingLLM(fail=fail)
    result = extract_chunk(llm, "The app uses PostgreSQL.")
    assert result.failed is fail
    expected_calls = 1 if fail else 2
    assert result.completion_calls == llm.completion_calls == expected_calls
    assert result.provider_attempts == llm.request_attempts == 3 * expected_calls


def test_wrong_shape_reply_is_flagged_failed():
    llm = StubLLMClient(default=json.dumps(["not", "an", "object"]))
    assert extract_chunk(llm, "x").failed is True


def test_clean_empty_object_is_not_failed():
    """The floor: the model answered, and this chunk holds nothing."""
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    result = extract_chunk(llm, "x")
    assert result.completion_calls == result.provider_attempts == len(llm.calls) == 2
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
        result = extract_chunk(StubLLMClient(default=_complete(payload)), "x")
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
        result = extract_chunk(StubLLMClient(default=_complete(payload)), "tiny")
    assert result.failed is True
    assert result.triples == [] and result.markers == []
    assert "chunk_extraction.item_validation_failure" in caplog.text


def test_clean_empty_object_is_not_flagged_missing_keys(caplog):
    """Both keys present but empty is the real floor, and must stay silent."""
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
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
    payload = {"triples": [triple], "markers": [marker], "complete": True}
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
    if mutation in {"invalid_type_hint", "bad_properties", "huge_integer"}:
        assert result.failed is False
        assert len(result.triples) == 1 and len(result.markers) == 1
    else:
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
        "complete": True,
    }
    result = extract_chunk(StubLLMClient(default=json.dumps(payload)), "tiny")
    assert result.failed is False
    assert result.entity_type_hints == {"thing": entity_type}


class _BoundarySplitLLM:
    def __init__(self):
        self.full_calls = 0

    def complete(self, request):
        # The parent includes both sentinels and is structurally cut; a child
        # can recover the fact only when the complete sentence stays intact.
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
                "complete": True,
            })
        return '{"triples": [], "markers": [], "complete": true}'


def test_semantic_split_preserves_midpoint_fact():
    text = (
        "LEFT_EDGE" + ("x" * 100) + "\n"
        "app uses PostgreSQL.\n" + ("y" * 100) + "RIGHT_EDGE"
    )
    result = extract_chunk(_BoundarySplitLLM(), text)
    assert result.failed is False
    assert any(t.object == "PostgreSQL" for t in result.triples)


def test_sentence_terminal_followed_by_newline_is_an_admissible_boundary():
    left = ("x" * 110) + " The first deployment is complete.\n"
    right = "The next deployment starts " + ("y" * 110)

    assert chunk_extraction._semantic_split_point(left + right) == len(left)


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
            "complete": True,
        })


def test_semantic_split_deep_merges_nonconflicting_entity_properties():
    text = "LEFT_EDGE" + ("x" * 110) + ".\n" + ("y" * 110) + "RIGHT_EDGE"
    result = extract_chunk(_MetadataSplitLLM(), text)
    assert result.failed is False
    assert result.entity_property_hints["app"] == {"left": "one", "right": "two"}


def test_semantic_split_drops_conflicting_optional_entity_properties():
    text = "LEFT_EDGE" + ("x" * 110) + ".\n" + ("y" * 110) + "RIGHT_EDGE"
    result = extract_chunk(_MetadataSplitLLM(conflict=True), text)
    assert result.failed is False
    assert len(result.triples) == 1
    assert result.entity_property_hints == {}


class _RecursiveCutLLM:
    def __init__(self):
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        excerpt = request.user.split('"""', 2)[1]
        if len(excerpt) > 800:
            return '{"triples": ['
        triples = []
        if "LEFT_FACT" in excerpt:
            triples.append({
                "subject": "app", "predicate": "uses", "object": "left_db",
                "polarity": 1,
            })
        if "RIGHT_FACT" in excerpt:
            triples.append({
                "subject": "app", "predicate": "uses", "object": "right_db",
                "polarity": 1,
            })
        return _complete({"triples": triples, "markers": []})


def test_recursive_truncation_recovery_goes_beyond_one_split():
    llm = _RecursiveCutLLM()
    text = "LEFT_FACT. " + ("padding sentence. " * 145) + "RIGHT_FACT."
    result = extract_chunk(llm, text)
    assert result.failed is False
    assert {triple.object for triple in result.triples} == {"left_db", "right_db"}
    assert len(llm.calls) >= 7, "both first-generation children must split again"
    assert result.completion_calls == result.provider_attempts == len(llm.calls)
    assert all(call.max_tokens <= 4096 for call in llm.calls)


def _left_failure_short_circuit_text() -> str:
    """One balanced root split plus an eight-leaf right recovery tree."""

    left = "I prefer LEFT_FATAL " + ("x" * 1260) + ".\n\n"
    right = "".join(
        f"RIGHT_BRANCH_SENTENCE_{index} " + ("y" * 130) + ". "
        for index in range(8)
    )
    text = left + right
    assert chunk_extraction._semantic_split_point(text) == len(left)
    return text


class _LeftFailureShortCircuitLLM:
    """Make the unneeded right branch cost exactly 23 logical calls.

    Its eight terminal sentences form seven incomplete internal nodes, then
    eight primary/omission pairs: 7 + (8 * 2) = 23.  Each provider-facing
    completion represents three underlying attempts so the test also pins the
    invocation accounting across the raised left call.
    """

    def __init__(self, split_path: str, *, left_fails: bool):
        self.split_path = split_path
        self.left_fails = left_fails
        self.calls: list[LLMRequest] = []
        self.request_attempts = 0

    @property
    def right_calls(self) -> list[LLMRequest]:
        return [call for call in self.calls if "LEFT_FATAL" not in call.user]

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        self.request_attempts += 3
        excerpt = request.user.split('"""', 2)[1]
        is_root = (
            "LEFT_FATAL" in excerpt
            and "RIGHT_BRANCH_SENTENCE_" in excerpt
        )
        if is_root:
            if self.split_path == "cue_empty":
                return _complete({"triples": [], "markers": []})
            return json.dumps({
                "triples": [], "markers": [], "complete": False,
            })

        if "LEFT_FATAL" in excerpt:
            if self.left_fails:
                raise RuntimeError("decisive left failure")
            if "OMISSION VERIFICATION PASS" in request.system:
                return _complete({"triples": [], "markers": []})
            return _complete({
                "triples": [{
                    "subject": "user", "predicate": "prefers",
                    "object": "left", "polarity": 1,
                }],
                "markers": [],
            })

        sentence_count = excerpt.count("RIGHT_BRANCH_SENTENCE_")
        if sentence_count > 1:
            return json.dumps({
                "triples": [], "markers": [], "complete": False,
            })
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete({"triples": [], "markers": []})
        index = next(
            index for index in range(8)
            if f"RIGHT_BRANCH_SENTENCE_{index}" in excerpt
        )
        return _complete({
            "triples": [{
                "subject": f"service-{index}", "predicate": "uses",
                "object": f"right-{index}", "polarity": 1,
            }],
            "markers": [],
        })


@pytest.mark.parametrize(
    "split_path", ["cue_empty", "recoverable_failure"]
)
def test_decisive_left_failure_never_evaluates_expensive_right_branch(
    split_path,
):
    llm = _LeftFailureShortCircuitLLM(split_path, left_fails=True)

    result = extract_chunk(
        llm,
        _left_failure_short_circuit_text(),
        # The former ordering consumed all 25 calls: root + failed left + the
        # otherwise healthy 23-call right subtree established below.
        completion_call_limit=25,
    )

    assert result.failed is True
    assert result.failure_reason == "branch_incomplete"
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == len(llm.calls) == 2
    assert result.provider_attempts == llm.request_attempts == 6
    assert llm.right_calls == []
    assert "left:call_failure" in result.failure_details
    assert "left.provider:call_failed" in result.failure_details
    assert not any(detail.startswith("right") for detail in result.failure_details)


def test_left_failure_short_circuit_discards_nested_partial_values():
    nested_partial = chunk_extraction.ChunkResult(
        triples=[Triple("partial", "uses", "discard-me", 1)],
        entity_type_hints={"partial": "service"},
        entity_property_hints={"partial": {"owner": "discard-me"}},
        markers=[Marker("preference", "discard me")],
        failed=True,
        failure_reason="branch_incomplete",
        failure_details=("right:call_failure", "right.provider:call_failed"),
    )

    result = chunk_extraction._failed_split_after_left(nested_partial)

    assert result.failed is True
    assert result.failure_reason == "branch_incomplete"
    assert result.triples == [] and result.markers == []
    assert result.entity_type_hints == {}
    assert result.entity_property_hints == {}
    assert result.failure_details == (
        "left:branch_incomplete",
        "left.right:call_failure",
        "left.right.provider:call_failed",
    )


@pytest.mark.parametrize(
    "split_path", ["cue_empty", "recoverable_failure"]
)
def test_successful_left_still_evaluates_and_merges_right_branch(split_path):
    llm = _LeftFailureShortCircuitLLM(split_path, left_fails=False)

    result = extract_chunk(
        llm,
        _left_failure_short_circuit_text(),
        completion_call_limit=26,
    )

    assert result.failed is False
    assert {triple.object for triple in result.triples} == {
        "left", *(f"right-{index}" for index in range(8)),
    }
    assert len(llm.right_calls) == 23
    assert result.completion_calls == len(llm.calls) == 26
    assert result.provider_attempts == llm.request_attempts == 78


class _FragmentInspectingLLM:
    def __init__(self):
        self.payloads: list[dict] = []
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        encoded = request.user.split('"""', 2)[1].strip()
        payload = json.loads(encoded)
        self.payloads.append(payload)
        triples = []
        if "I use PostgreSQL" in payload["content"]:
            triples.append({
                "subject": "user", "predicate": "uses",
                "object": "PostgreSQL", "polarity": 1,
                "source_message_id": payload["source_message_id"],
            })
        return _complete({"triples": triples, "markers": []})


def test_oversized_source_is_prefragmented_with_absolute_offsets_and_citation():
    llm = _FragmentInspectingLLM()
    content = "I use PostgreSQL. " + ("dense context. " * 600)
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )
    assert result.failed is False
    assert len(result.triples) == 1
    assert result.triples[0].source_message_id == 17
    assert len(llm.payloads) > 1
    assert all(
        payload["source_record_version"] == "hymem-claim-source-fragment-v2"
        and payload["source_message_id"] == 17
        and isinstance(payload["source_content_start"], int)
        and isinstance(payload["source_content_end"], int)
        for payload in llm.payloads
    )
    assert min(payload["source_content_start"] for payload in llm.payloads) == 0
    assert max(payload["source_content_end"] for payload in llm.payloads) == len(content)
    ordered = sorted(llm.payloads, key=lambda payload: (
        payload["source_content_start"], payload["source_content_end"]
    ))
    assert all(
        payload["content"]
        == content[payload["source_content_start"]:payload["source_content_end"]]
        for payload in ordered
    )
    unique_intervals = sorted({
        (payload["source_content_start"], payload["source_content_end"])
        for payload in ordered
    })
    assert unique_intervals[0][0] == 0
    assert unique_intervals[-1][1] == len(content)
    assert all(
        current[0] == previous[1]
        for previous, current in zip(unique_intervals, unique_intervals[1:])
    ), "semantic fragments must preserve every source offset without a gap"
    assert all(call.max_tokens < 8192 for call in llm.calls)


class _CrossBoundaryContextLLM:
    """Extract only when the labelled preceding window completes the claim."""

    def __init__(self):
        self.calls: list[LLMRequest] = []
        self.payloads: list[dict] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        self.payloads.append(payload)
        context = payload.get("source_boundary_context", {})
        applies_local_end = max(
            0,
            context.get("applies_through_source_content_end", 0)
            - payload.get("source_content_start", 0),
        )
        preceding = context.get("content", "")
        claim_crosses_boundary = (
            "My preferred database is named" in preceding
            and "PostgreSQL" in payload["content"][:applies_local_end]
        )
        triples = []
        if claim_crosses_boundary and "VERIFICATION PASS" not in request.system:
            triples.append({
                "subject": "user",
                "predicate": "prefers",
                "object": "PostgreSQL",
                "polarity": 1,
                "source_message_id": payload["source_message_id"],
            })
        return _complete({"triples": triples, "markers": []})


@pytest.mark.parametrize(
    "boundary",
    [
        " My preferred database is named in the next sentence. ",
        " My preferred database is named in the next sentence.\n",
        " My preferred database is named in the next sentence.\r\n",
        " My preferred database is named in the next paragraph:\n\n",
        " My preferred database is named in the next paragraph:\r\n\r\n",
    ],
    ids=["sentence-space", "sentence-lf", "sentence-crlf", "paragraph-lf", "paragraph-crlf"],
)
def test_source_backed_prose_split_recovers_cross_boundary_claim(boundary: str):
    llm = _CrossBoundaryContextLLM()
    content = ("x" * 2050) + boundary + "PostgreSQL " + ("y" * 2050)

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert [
        (triple.subject, triple.predicate, triple.object, triple.source_message_id)
        for triple in result.triples
    ] == [("user", "prefers", "PostgreSQL", 17)]
    assert result.initial_prepartition_leaves == 2
    assert result.completion_calls == result.provider_attempts == len(llm.calls) == 4

    contextual = [
        payload for payload in llm.payloads
        if "source_boundary_context" in payload
    ]
    assert contextual
    intervals = sorted({
        (payload["source_content_start"], payload["source_content_end"])
        for payload in llm.payloads
    })
    assert intervals[0][0] == 0 and intervals[-1][1] == len(content)
    assert all(
        right[0] == left[1]
        for left, right in zip(intervals, intervals[1:])
    )
    assert "".join(content[start:end] for start, end in intervals) == content
    for payload in contextual:
        context = payload["source_boundary_context"]
        fragment_start = payload["source_content_start"]
        context_start = max(
            0,
            fragment_start
            - chunk_extraction._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS,
        )
        assert context == {
            "version": chunk_extraction.SOURCE_BOUNDARY_CONTEXT_VERSION,
            "kind": "preceding_adjacent_prose",
            "content": content[context_start:fragment_start],
            "source_content_start": context_start,
            "source_content_end": fragment_start,
            "applies_through_source_content_end": fragment_start + min(
                chunk_extraction._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS,
                len(content) - fragment_start,
            ),
        }
        assert payload["content"] == content[
            fragment_start:payload["source_content_end"]
        ]


def test_recursive_prose_split_combines_inherited_context_with_owned_prefix():
    first_left = ("x" * 450) + " ORIGINAL_CUE finishes here. "
    second_left = "Owned prefix " + ("p" * 100) + ". "
    final = "Final continuation " + ("y" * 250)
    content = first_left + second_left + final
    record = _source_record(17, content)
    unit = chunk_extraction._ExtractionUnit(
        text=record[1], source_records=(record,),
    )

    first = chunk_extraction._split_unit(unit)
    assert first is not None
    first_left_unit, first_right_unit = first
    first_right_payload = chunk_extraction._source_payload(
        first_right_unit.source_records[0]
    )
    assert first_right_payload is not None
    assert first_right_payload["source_content_start"] == len(first_left)

    second = chunk_extraction._split_unit(first_right_unit)
    assert second is not None
    second_left_unit, second_right_unit = second
    second_left_payload = chunk_extraction._source_payload(
        second_left_unit.source_records[0]
    )
    second_right_payload = chunk_extraction._source_payload(
        second_right_unit.source_records[0]
    )
    assert second_left_payload is not None and second_right_payload is not None
    assert second_left_payload["source_boundary_context"] == (
        first_right_payload["source_boundary_context"]
    )

    second_cut = len(first_left) + len(second_left)
    expected_start = max(
        0,
        second_cut - chunk_extraction._MAX_SOURCE_BOUNDARY_CONTEXT_CHARS,
    )
    second_context = second_right_payload["source_boundary_context"]
    assert second_context["source_content_start"] == expected_start
    assert second_context["source_content_end"] == second_cut
    assert second_context["content"] == content[expected_start:second_cut]
    assert "ORIGINAL_CUE" in second_context["content"]
    assert (
        second_left_payload["content"] + second_right_payload["content"]
        == first_right_payload["content"]
    )
    assert (
        first_left_unit.source_records is not None
        and first_left_unit.source_records[0][0] == 17
    )


class _BoundaryOwnershipLLM:
    def __init__(self):
        self.calls: list[LLMRequest] = []
        self.saw_context_only_candidate = False
        self.saw_ownership_rule = False

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        context = payload.get("source_boundary_context", {})
        context_only = (
            "Avery owns Bicycle" in context.get("content", "")
            and "Bicycle" not in payload["content"]
        )
        if context_only:
            self.saw_context_only_candidate = True
            self.saw_ownership_rule = (
                "never return a triple or marker stated wholly in it"
                in request.system
            )
        triples = []
        # This adversarial test client emits the context-only item unless the
        # production request carries the explicit one-sided ownership rule.
        if context_only and not self.saw_ownership_rule:
            triples.append({
                "subject": "Avery", "predicate": "owns",
                "object": "Bicycle", "polarity": 1,
                "source_message_id": payload["source_message_id"],
            })
        return _complete({"triples": triples, "markers": []})


def test_preceding_context_cannot_independently_publish_an_item():
    llm = _BoundaryOwnershipLLM()
    content = (
        ("x" * 2050) + " Avery owns Bicycle. Next section. " + ("y" * 2050)
    )

    result = extract_chunk(
        llm,
        "ignored",
        source_records=(_source_record(17, content),),
    )

    assert llm.saw_context_only_candidate is True
    assert llm.saw_ownership_rule is True
    assert result.failed is False
    assert result.triples == [] and result.markers == []


class _BoundaryDuplicateLLM:
    def __init__(self, *, conflict: bool = False):
        self.conflict = conflict

    def complete(self, request: LLMRequest) -> str:
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        if "VERIFICATION PASS" in request.system:
            return _complete({"triples": [], "markers": []})
        context = payload.get("source_boundary_context", {})
        from_left = "App uses PostgreSQL" in payload["content"]
        from_boundary = (
            "App uses PostgreSQL" in context.get("content", "")
            and "PostgreSQL" in payload["content"]
        )
        triples = []
        if from_left or from_boundary:
            triples.append({
                "subject": "App", "predicate": "uses",
                "object": "PostgreSQL",
                "polarity": -1 if self.conflict and from_boundary else 1,
                "source_message_id": payload["source_message_id"],
            })
        return _complete({"triples": triples, "markers": []})


def test_cross_boundary_exact_duplicate_is_published_once():
    content = (
        ("x" * 2050)
        + " App uses PostgreSQL. PostgreSQL remains listed. "
        + ("y" * 2050)
    )
    result = extract_chunk(
        _BoundaryDuplicateLLM(),
        "ignored",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert len(result.triples) == 1
    assert result.triples[0].object == "PostgreSQL"
    assert result.duplicate_triples_collapsed == 1


def test_cross_boundary_polarity_conflict_fails_the_whole_source():
    content = (
        ("x" * 2050)
        + " App uses PostgreSQL. PostgreSQL remains listed. "
        + ("y" * 2050)
    )
    result = extract_chunk(
        _BoundaryDuplicateLLM(conflict=True),
        "ignored",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "response_conflict"
    assert "triples:polarity_conflict" in result.failure_details


_MIDPOINT_CROSSING_CLAIM = (
    "HyMem Canary Relay (service) deploys [NONSEMANTIC_PADDING="
    + ("a" * 293)
    + "] to Fly.io (platform)."
)


class _SemanticBoundaryClaimLLM:
    def __init__(self):
        self.calls: list[LLMRequest] = []
        self.source_payloads: list[dict] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        self.source_payloads.append(payload)
        triples = []
        if (
            _MIDPOINT_CROSSING_CLAIM in payload["content"]
            and "OMISSION VERIFICATION PASS" not in request.system
        ):
            triples.append({
                "subject": "HyMem Canary Relay",
                "subject_type": "service",
                "predicate": "deploys_to",
                "object": "Fly.io",
                "object_type": "platform",
                "polarity": 1,
                "source_message_id": payload["source_message_id"],
            })
        return _complete({"triples": triples, "markers": []})


def test_semantic_source_split_preserves_the_whole_midpoint_claim():
    """Reproduce the old +/-64 split loss with a real 374-char assertion."""

    assert len(_MIDPOINT_CROSSING_CLAIM) == 374
    content = (
        ("x" * 2100) + " " + _MIDPOINT_CROSSING_CLAIM + " " + ("y" * 2100)
    )
    legacy_midpoint = len(content) // 2
    assert _MIDPOINT_CROSSING_CLAIM not in content[:legacy_midpoint + 64]
    assert _MIDPOINT_CROSSING_CLAIM not in content[legacy_midpoint - 64:]

    llm = _SemanticBoundaryClaimLLM()
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert [(triple.predicate, triple.object) for triple in result.triples] == [
        ("deploys_to", "Fly.io")
    ]
    containing = [
        payload for payload in llm.source_payloads
        if _MIDPOINT_CROSSING_CLAIM in payload["content"]
    ]
    assert containing, "at least one provider request must contain the whole claim"
    assert all(
        payload["content"]
        == content[payload["source_content_start"]:payload["source_content_end"]]
        for payload in llm.source_payloads
    )
    intervals = sorted({
        (payload["source_content_start"], payload["source_content_end"])
        for payload in llm.source_payloads
    })
    assert intervals[0][0] == 0 and intervals[-1][1] == len(content)
    assert all(right[0] == left[1] for left, right in zip(intervals, intervals[1:]))


@pytest.mark.parametrize(
    "internal_punctuation",
    [
        " e.g. 14-",
        " under Dr. Ada-",
        " under J. Smith-",
        " at version 3. 14-",
        " is 'ready? yes-",
        " says wow! really-",
        " at /search?q=a? mode-",
    ],
    ids=[
        "dotted-abbreviation",
        "title",
        "initial",
        "version-number",
        "quoted-question",
        "lowercase-exclamation",
        "url-query",
    ],
)
def test_ambiguous_punctuation_cannot_certify_fragmented_claim_empty(
    internal_punctuation: str,
):
    """One long assertion may not be split at sentence-internal punctuation."""

    content = (
        "The Amsterdam deployment record "
        + ("x" * 2050)
        + internal_punctuation
        + ("y" * 2050)
        + " remains authoritative in Amsterdam."
    )
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.triples == [] and result.markers == []
    # Prepartition cannot prove a safe source boundary, so no provider unit is
    # allowed to observe only half of the assertion and certify it empty.
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


class _TrueBoundaryInspectingLLM:
    def __init__(self):
        self.calls: list[LLMRequest] = []
        self.contents: list[str] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        self.contents.append(payload["content"])
        return _complete({"triples": [], "markers": []})


def test_true_sentence_boundary_preserves_complete_provider_units():
    left = "Alpha narrative " + ("x" * 2050) + ". "
    right = "Bravo narrative " + ("y" * 2050) + "."
    content = left + right
    llm = _TrueBoundaryInspectingLLM()

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert result.triples == [] and result.markers == []
    assert result.initial_prepartition_leaves == 2
    assert result.completion_calls == result.provider_attempts == 4
    assert len(llm.calls) == 4
    # Each source leaf is visible once to PRIMARY and once to EMPTY verification;
    # neither provider unit begins or ends inside a sentence.
    assert llm.contents == [left, left, right, right]


def test_true_paragraph_boundary_remains_preferred_over_ambiguous_period():
    left = ("x" * 2050) + " e.g. supporting material\n\n"
    right = "New paragraph " + ("y" * 2050)

    assert chunk_extraction._semantic_split_point(left + right) == len(left)


def _dense_headerless_table(
    row_count: int = 120, *, newline: str = "\n",
) -> str:
    return newline.join(
        f"| service_{index} | depends_on | database_{index} |"
        for index in range(row_count)
    )


class _DenseTableInspectingLLM:
    def __init__(self):
        self.calls: list[LLMRequest] = []
        self.payloads: list[dict] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        self.payloads.append(payload)
        triples = []
        if (
            "service_0" in payload["content"]
            and "VERIFICATION PASS" not in request.system
        ):
            triples.append({
                "subject": "service_0",
                "predicate": "depends_on",
                "object": "database_0",
                "polarity": 1,
                "source_message_id": payload["source_message_id"],
            })
        return _complete({"triples": triples, "markers": []})


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_dense_headerless_record_grid_fails_closed_without_provider_spend(
    newline: str,
):
    """Even record-shaped rows need a header to prove their semantics."""

    content = _dense_headerless_table(newline=newline)
    assert len(content) > chunk_extraction._MAX_LEAF_INPUT_CHARS
    assert chunk_extraction._markdown_table_boundary_points(content) == []
    assert chunk_extraction._semantic_split_point(content) is None

    llm = _DenseTableInspectingLLM()
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.completion_calls == result.provider_attempts == 0
    assert result.initial_prepartition_leaves == 0
    assert llm.calls == [] and llm.payloads == []


def _ambiguous_headerless_pipe_prose(*, newline: str) -> tuple[str, int]:
    rows: list[str] = []
    for index in range(120):
        relationship = f"ordinary continuation material {index:03d}"
        if index == 59:
            relationship = "service deployment depends"
        elif index == 60:
            relationship = "on PostgreSQL for production"
        rows.append(
            f"| narrative segment {index:03d} "
            f"| {relationship:<38} "
            f"| surrounding prose remains continuous |"
        )
    content = newline.join(rows)
    boundary = len(newline.join(rows[:60])) + len(newline)
    return content, boundary


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_same_width_headerless_pipe_prose_crossing_midpoint_fails_closed(
    newline: str,
):
    content, legacy_grid_cut = _ambiguous_headerless_pipe_prose(newline=newline)
    assert len(content) > chunk_extraction._MAX_LEAF_INPUT_CHARS
    assert abs(legacy_grid_cut - len(content) // 2) <= len(newline)
    assert "deployment depends" in content[:legacy_grid_cut]
    assert content[legacy_grid_cut:].startswith("| narrative segment 060")
    assert "on PostgreSQL" in content[legacy_grid_cut:]
    assert chunk_extraction._markdown_table_boundary_points(content) == []
    assert chunk_extraction._semantic_split_point(content) is None

    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_canonical_table_split_keeps_rows_and_exact_offsets(newline: str):
    content = newline.join([
        "| service | dependency |",
        "| --- | --- |",
        *(
            f"| service_{index} | database_{index}. |"
            for index in range(320)
        ),
    ])
    header_end = content.index(newline) + len(newline)
    delimiter_end = content.index(newline, header_end) + len(newline)
    points = chunk_extraction._markdown_table_boundary_points(content)
    cut = chunk_extraction._semantic_split_point(content)

    assert points
    assert chunk_extraction._sentence_boundary_points(content) == []
    assert header_end not in points and delimiter_end not in points
    assert cut in points and cut > delimiter_end
    assert content[:cut].startswith(
        f"| service | dependency |{newline}| --- | --- |{newline}"
    )

    llm = _DenseTableInspectingLLM()
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert result.initial_prepartition_leaves >= 4
    assert result.completion_calls == result.provider_attempts == (
        2 * result.initial_prepartition_leaves
    )
    assert len(llm.calls) == result.completion_calls
    unique = sorted({
        (
            payload["source_content_start"],
            payload["source_content_end"],
            payload["content"],
        )
        for payload in llm.payloads
    })
    assert unique[0][0] == 0 and unique[-1][1] == len(content)
    assert all(right[0] == left[1] for left, right in zip(unique, unique[1:]))
    assert "".join(fragment for _start, _end, fragment in unique) == content
    assert all(fragment == content[start:end] for start, end, fragment in unique)
    assert all(fragment.endswith(newline) for _start, _end, fragment in unique[:-1])
    assert all(fragment.startswith("|") for _start, _end, fragment in unique[1:])


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize(
    ("opener", "short_or_wrong_close", "closer"),
    [
        ("````python", "```", "```````"),
        ("~~~~ details", "```", "~~~~~"),
    ],
    ids=["backtick-longer-close", "tilde-opposite-fence"],
)
def test_fenced_code_is_atomic_and_table_or_prose_shapes_cannot_split_it(
    newline: str,
    opener: str,
    short_or_wrong_close: str,
    closer: str,
):
    interior = newline.join([
        "A complete-looking sentence. Another one!",
        "",
        "| service | dependency |",
        "| --- | --- |",
        "| app | PostgreSQL |",
        "- list-looking item",
        "  continuation with punctuation. Next clause.",
        short_or_wrong_close,
    ])
    content = newline.join([opener, *(interior for _ in range(70)), closer])

    assert len(content) > chunk_extraction._MAX_LEAF_INPUT_CHARS
    assert chunk_extraction._fenced_code_spans(content) == ((0, len(content)),)
    assert chunk_extraction._markdown_table_boundary_points(content) == []
    assert chunk_extraction._semantic_split_point(content) is None


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_unmatched_fence_protects_through_eof_and_holds_without_provider_spend(
    newline: str,
):
    content = newline.join([
        "   ````python",
        *(("inside. Another sentence." + newline + newline) for _ in range(180)),
        "```",  # shorter than the opener, so it is source text, not a close
        "| a | b |",
        "| --- | --- |",
        "| x | y |",
    ])
    assert len(content) > chunk_extraction._MAX_LEAF_INPUT_CHARS
    assert chunk_extraction._fenced_code_spans(content) == ((0, len(content)),)
    assert chunk_extraction._semantic_split_point(content) is None

    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_invalid_backtick_info_and_short_close_do_not_gain_fence_authority():
    invalid_opener = "```python`invalid\nordinary prose. Next sentence.\n"
    short_close = "````\nbody\n```\nstill body\n`````\n"

    assert chunk_extraction._fenced_code_spans(invalid_opener) == ()
    assert chunk_extraction._fenced_code_spans(short_close) == (
        (0, len(short_close)),
    )


@pytest.mark.parametrize(
    "heading",
    ["# Deployment", "Deployment\n=========="],
    ids=["atx", "setext"],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_heading_stays_with_its_first_body_block(
    heading: str, newline: str,
):
    rendered_heading = heading.replace("\n", newline)
    first_body = (
        "First body " + ("x" * 500)
        + ". A second sentence remains part of that same body block."
    )
    later = "Later independent block " + ("y" * 500) + "."
    content = newline.join([rendered_heading, "", first_body, "", later])
    body_end = content.index(first_body) + len(first_body) + len(newline)

    cut = chunk_extraction._semantic_split_point(content)

    assert cut is not None and cut >= body_end
    assert rendered_heading in content[:cut]
    assert first_body in content[:cut]
    assert content[:cut] + content[cut:] == content


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_multiline_setext_heading_paragraph_cannot_split_before_underline_or_body(
    newline: str,
):
    content = (
        "Prior paragraph " + ("a" * 180)
        + newline + "Title." + newline + "---" + newline
        + "Body " + ("b" * 180)
    )
    unsafe_sentence_cut = content.index("Title.") + len("Title.") + len(newline)

    assert unsafe_sentence_cut in chunk_extraction._sentence_boundary_points(content)
    assert chunk_extraction._semantic_split_point(content) is None
    analysis = chunk_extraction._markdown_block_analysis(content)
    assert analysis.protected_spans == ((0, len(content)),)


@pytest.mark.parametrize(
    "underline", ["=", "==", "--"],
    ids=["one-equals", "two-equals", "two-hyphen"],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_short_valid_setext_underline_keeps_heading_with_first_body(
    underline: str, newline: str,
):
    first_body = "First body " + ("x" * 260) + ". Another sentence."
    later = "Later independent block " + ("y" * 260) + "."
    content = newline.join([
        "First heading line",
        "Title.",
        underline,
        first_body,
        "",
        later,
    ])
    body_end = content.index(first_body) + len(first_body) + len(newline)

    cut = chunk_extraction._semantic_split_point(content)

    assert cut is not None and cut >= body_end
    assert "First heading line" in content[:cut]
    assert "Title." in content[:cut]
    assert underline in content[:cut]
    assert first_body in content[:cut]
    assert content[:cut] + content[cut:] == content


def test_single_hyphen_is_not_promoted_from_empty_list_item_to_setext():
    content = "Title.\n-\nList continuation"
    lines = chunk_extraction._markdown_lines(content)
    fences = chunk_extraction._fenced_code_spans(content, lines)
    lists = chunk_extraction._list_blocks(lines, fences)
    opaque = chunk_extraction._normalized_protected_spans([
        *fences,
        *(span for block in lists for span in block.item_spans),
    ])

    assert chunk_extraction._heading_line_ranges(lines, opaque) == ()


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_thematic_break_after_blank_does_not_gain_setext_boundary_authority(
    newline: str,
):
    first = "Independent paragraph " + ("x" * 260) + "."
    after_rule = "Independent later paragraph " + ("y" * 260) + "."
    content = newline.join([first, "", "---", "", after_rule])

    ranges = chunk_extraction._heading_line_ranges(
        chunk_extraction._markdown_lines(content),
        chunk_extraction._normalized_protected_spans(
            list(chunk_extraction._fenced_code_spans(content))
        ),
    )
    cut = chunk_extraction._semantic_split_point(content)

    assert ranges == ()
    assert cut is not None
    assert content[:cut] + content[cut:] == content


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_safe_boundary_before_later_heading_keeps_each_section_complete(
    newline: str,
):
    first = "# First" + newline + newline + "Alpha " + ("a" * 500)
    second = "# Second" + newline + newline + "Bravo " + ("b" * 500)
    content = first + newline + newline + second

    cut = chunk_extraction._semantic_split_point(content)

    assert cut == content.index("# Second")
    assert content[:cut] + content[cut:] == content
    assert content[cut:].startswith("# Second" + newline + newline + "Bravo")


@pytest.mark.parametrize(
    ("first_marker", "second_marker", "nested_marker"),
    [("-", "-", "*"), ("1.", "2.", "1.")],
    ids=["bulleted", "ordered"],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_colon_intro_and_entire_list_are_atomic_until_next_independent_block(
    first_marker: str,
    second_marker: str,
    nested_marker: str,
    newline: str,
):
    first_item_lines = [
        f"{first_marker} First item " + ("x" * 360) + ".",
        "  lazy continuation " + ("y" * 260) + ". Another sentence.",
        "",
        f"    {nested_marker} nested item " + ("z" * 260) + ".",
        "      nested continuation " + ("n" * 160) + ".",
    ]
    second_item = f"{second_marker} Second item " + ("q" * 550) + "."
    later = "Independent conclusion " + ("r" * 550) + "."
    content = newline.join([
        "Deployment requirements:",
        "",
        *first_item_lines,
        second_item,
        "",
        later,
    ])
    second_start = content.index(second_item)
    later_start = content.index(later)

    cut = chunk_extraction._semantic_split_point(content)

    assert cut == later_start
    assert cut > second_start
    assert "Deployment requirements:" in content[:cut]
    assert nested_marker + " nested item" in content[:cut]
    assert second_item in content[:cut]
    assert content[cut:].startswith(later)
    assert content[:cut] + content[cut:] == content


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_standalone_complete_list_items_can_split_at_a_sibling(newline: str):
    first = "- First independent item " + ("x" * 500) + "."
    second = "- Second independent item " + ("y" * 500) + "."
    content = newline.join([first, second])

    cut = chunk_extraction._semantic_split_point(content)

    assert cut == content.index(second)
    assert content[:cut] + content[cut:] == content


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("kind", ["fence", "table"])
def test_colon_intro_binds_the_entire_immediate_structural_block(
    newline: str, kind: str,
):
    if kind == "fence":
        block = newline.join([
            "```toml",
            "database = 'PostgreSQL'",
            *(("comment = '" + ("x" * 40) + "'") for _ in range(12)),
            "```",
        ])
    else:
        block = newline.join([
            "| service | database |",
            "| --- | --- |",
            *(f"| app_{index} | database_{index} |" for index in range(20)),
        ])
    later = "Independent section " + ("z" * 500) + "."
    content = newline.join(["Use this configuration:", "", block, "", later])
    block_end = content.index(block) + len(block) + len(newline)

    cut = chunk_extraction._semantic_split_point(content)

    assert cut is not None and cut >= block_end
    assert "Use this configuration:" in content[:cut]
    assert block in content[:cut]
    assert content[:cut] + content[cut:] == content


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_one_oversized_list_item_with_continuations_holds_without_provider_spend(
    newline: str,
):
    content = newline.join([
        "Important claims:",
        "",
        "- first clause " + ("x" * 2100) + ". Another sentence.",
        "",
        "  continuation clause " + ("y" * 2100) + ".",
        "    - nested-looking clause " + ("z" * 300) + ".",
    ])
    assert len(content) > chunk_extraction._MAX_LEAF_INPUT_CHARS
    assert chunk_extraction._semantic_split_point(content) is None

    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_recursive_block_splits_reconstruct_exact_source_and_table_provenance(
    newline: str,
):
    table = newline.join([
        "| service | dependency |",
        "| --- | --- |",
        *(f"| service_{index} | database_{index} |" for index in range(240)),
    ])
    unit = chunk_extraction._ExtractionUnit(
        text=_source_record(17, table)[1],
        source_records=(_source_record(17, table),),
    )

    first = chunk_extraction._split_unit(unit)
    assert first is not None
    left, right = first
    left_payload = chunk_extraction._source_payload(left.source_records[0])
    right_payload = chunk_extraction._source_payload(right.source_records[0])
    assert left_payload is not None and right_payload is not None
    assert left_payload["content"] + right_payload["content"] == table
    assert "source_fragment_context" not in left_payload
    assert right_payload["source_fragment_context"][
        "applies_through_source_content_end"
    ] == len(table)
    assert left.trusted_table_boundaries or right.trusted_table_boundaries

    continuation = right if right.trusted_table_boundaries else left
    second = chunk_extraction._split_unit(continuation)
    assert second is not None
    parent_payload = chunk_extraction._source_payload(
        continuation.source_records[0]
    )
    second_left = chunk_extraction._source_payload(second[0].source_records[0])
    second_right = chunk_extraction._source_payload(second[1].source_records[0])
    assert parent_payload is not None
    assert second_left is not None and second_right is not None
    assert (
        second_left["content"] + second_right["content"]
        == parent_payload["content"]
    )
    assert second_left["source_fragment_context"] == (
        parent_payload["source_fragment_context"]
    )
    assert second_right["source_fragment_context"] == (
        parent_payload["source_fragment_context"]
    )
    assert second_left["source_content_start"] == (
        parent_payload["source_content_start"]
    )
    assert second_left["source_content_end"] == (
        second_right["source_content_start"]
    )
    assert second_right["source_content_end"] == (
        parent_payload["source_content_end"]
    )
    assert all(
        point in chunk_extraction._trusted_table_boundary_points(
            parent_payload["content"], continuation.trusted_table_boundaries
        )
        for point in continuation.trusted_table_boundaries
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize(
    ("prelude_lines", "prelude_kind"),
    [
        (("# Results",), "atx_heading"),
        (("Quarterly results", "for applications", "---"), "setext_heading"),
        (("These rows describe", "the selected databases:"),
         "colon_led_paragraph"),
    ],
    ids=["atx", "multiline-setext", "colon-paragraph"],
)
def test_recursive_introduced_table_splits_carry_exact_labelled_context(
    newline: str,
    prelude_lines: tuple[str, ...],
    prelude_kind: str,
):
    prelude = newline.join(prelude_lines) + newline
    prefix = prelude + newline
    header = newline.join([
        "| service | selected database |",
        "| --- | --- |",
        "",
    ])
    table = header + newline.join(
        f"| app_{index:03d} | database_{index:03d} |" for index in range(420)
    )
    content = prefix + table
    record = _source_record(17, content)

    analysis = chunk_extraction._markdown_block_analysis(content)
    cut = chunk_extraction._semantic_split_point(content)
    assert cut in analysis.contextual_table_boundaries

    leaves, failure = chunk_extraction._prepartition(
        chunk_extraction._ExtractionUnit(
            text=record[1],
            source_records=(record,),
        )
    )

    assert failure is None
    assert leaves is not None and len(leaves) >= 3
    payloads = sorted(
        (
            chunk_extraction._source_payload(unit.source_records[0])
            for unit, _depth in leaves
        ),
        key=lambda payload: payload["source_content_start"],
    )
    assert all(payload is not None for payload in payloads)
    assert payloads[0]["source_content_start"] == 0
    assert payloads[-1]["source_content_end"] == len(content)
    assert all(
        right["source_content_start"] == left["source_content_end"]
        for left, right in zip(payloads, payloads[1:])
    )
    assert "".join(payload["content"] for payload in payloads) == content
    assert all(
        payload["content"] == content[
            payload["source_content_start"]:payload["source_content_end"]
        ]
        for payload in payloads
    )

    continuations = [
        payload for payload in payloads
        if payload["source_content_start"] > 0
    ]
    assert continuations
    for payload in continuations:
        context = payload["source_fragment_context"]
        applies_through = context.pop("applies_through_source_content_end")
        assert context == {
            "version": chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION,
            "kind": "introduced_canonical_markdown_table_header",
            "content": header,
            "source_content_start": len(prefix),
            "source_content_end": len(prefix) + len(header),
            "prelude_kind": prelude_kind,
            "prelude_content": prelude,
            "prelude_source_content_start": 0,
            "prelude_source_content_end": len(prelude),
        }
        assert payload["source_content_end"] <= applies_through <= len(content)
        assert header not in payload["content"]
        assert prelude not in payload["content"]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_later_independent_introduced_table_gets_its_own_prelude(newline: str):
    first_prelude = "# Primary results" + newline
    first_prefix = first_prelude + newline
    first_header = newline.join([
        "| primary service | database |",
        "| --- | --- |",
        "",
    ])
    first_table = first_header + newline.join(
        f"| primary_{index:03d} | db_{index:03d} |" for index in range(240)
    )
    second_prelude = "Independent inventory:" + newline
    second_prefix = second_prelude + newline
    second_header = newline.join([
        "| later service | database |",
        "| --- | --- |",
        "",
    ])
    second_table = second_header + newline.join(
        f"| later_{index:03d} | store_{index:03d} |" for index in range(240)
    )
    separator = newline + newline
    content = (
        first_prefix + first_table + separator + second_prefix + second_table
    )
    second_prelude_start = len(first_prefix + first_table + separator)
    second_header_start = second_prelude_start + len(second_prefix)
    record = _source_record(17, content)

    leaves, failure = chunk_extraction._prepartition(
        chunk_extraction._ExtractionUnit(
            text=record[1],
            source_records=(record,),
        )
    )

    assert failure is None
    assert leaves is not None and len(leaves) >= 4
    payloads = sorted(
        (
            chunk_extraction._source_payload(unit.source_records[0])
            for unit, _depth in leaves
        ),
        key=lambda payload: payload["source_content_start"],
    )
    assert all(payload is not None for payload in payloads)
    assert "".join(payload["content"] for payload in payloads) == content
    later_continuations = [
        payload for payload in payloads
        if payload["source_content_start"] > second_header_start + len(second_header)
    ]
    assert later_continuations
    assert all(
        payload["source_fragment_context"]["prelude_content"]
        == second_prelude
        and payload["source_fragment_context"]["prelude_kind"]
        == "colon_led_paragraph"
        and payload["source_fragment_context"]["prelude_source_content_start"]
        == second_prelude_start
        and payload["source_fragment_context"]["content"] == second_header
        and payload["source_fragment_context"]["source_content_start"]
        == second_header_start
        for payload in later_continuations
    )
    assert all(
        payload["source_fragment_context"]["prelude_content"]
        != first_prelude
        for payload in later_continuations
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_recursive_table_context_cannot_be_replaced_by_header_shaped_body_rows(
    newline: str,
):
    rows = [
        f"| p{index:03d} | db{index:03d} |" for index in range(400)
    ]
    rows[198] = "| fake person | fake database |"
    rows[199] = "| --- | --- |"
    canonical_header = newline.join([
        "| person | preferred database |",
        "| --- | --- |",
        "",
    ])
    fake_header = newline.join([rows[198], rows[199], ""])
    content = canonical_header + newline.join(rows)
    record = _source_record(17, content)

    leaves, failure = chunk_extraction._prepartition(
        chunk_extraction._ExtractionUnit(
            text=record[1],
            source_records=(record,),
        )
    )

    assert failure is None
    assert leaves is not None and len(leaves) >= 3
    payloads = sorted(
        (
            chunk_extraction._source_payload(unit.source_records[0])
            for unit, _depth in leaves
        ),
        key=lambda payload: payload["source_content_start"],
    )
    assert all(payload is not None for payload in payloads)
    assert payloads[0]["source_content_start"] == 0
    assert payloads[-1]["source_content_end"] == len(content)
    assert all(
        right["source_content_start"] == left["source_content_end"]
        for left, right in zip(payloads, payloads[1:])
    )
    assert "".join(payload["content"] for payload in payloads) == content
    assert all(
        payload["content"] == content[
            payload["source_content_start"]:payload["source_content_end"]
        ]
        for payload in payloads
    )

    descendant_contexts = [
        payload["source_fragment_context"]
        for payload in payloads
        if payload["source_content_start"] > 0
    ]
    assert descendant_contexts
    assert all(
        context["content"] == canonical_header
        and context["source_content_start"] == 0
        and context["source_content_end"] == len(canonical_header)
        for context in descendant_contexts
    )
    assert all(context["content"] != fake_header for context in descendant_contexts)
    fake_delimiter_end = content.index(fake_header) + len(fake_header)
    assert any(
        payload["source_content_start"] >= fake_delimiter_end
        and payload["source_fragment_context"]["content"] == canonical_header
        for payload in payloads
        if payload["source_content_start"] > 0
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_fragment_discovers_a_new_table_after_inherited_context_ends(
    newline: str,
):
    first_header = newline.join([
        "| person | database |",
        "| --- | --- |",
        "",
    ])
    first_row = f"| legacy | SQLite |{newline}"
    second_header = newline.join([
        "| service | dependency |",
        "| --- | --- |",
        "",
    ])
    second_rows = newline.join(
        f"| svc{index:03d} | db{index:03d} |" for index in range(400)
    )
    content = first_header + first_row + newline + second_header + second_rows
    record = _source_record(17, content)
    first_context = {
        "version": chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION,
        "kind": "canonical_markdown_table_header",
        "content": first_header,
        "source_content_start": 0,
        "source_content_end": len(first_header),
        "applies_through_source_content_end": len(first_header) + len(first_row),
    }
    fragment = chunk_extraction._fragment_record(
        record,
        start=len(first_header),
        end=len(content),
        source_fragment_context=first_context,
    )
    assert fragment is not None

    leaves, failure = chunk_extraction._prepartition(
        chunk_extraction._ExtractionUnit(
            text=fragment[1],
            source_records=(fragment,),
            # Exercise suffix rediscovery rather than inherited root cuts.
            trusted_table_boundaries=(),
        )
    )

    assert failure is None
    assert leaves is not None and len(leaves) >= 2
    payloads = sorted(
        (
            chunk_extraction._source_payload(unit.source_records[0])
            for unit, _depth in leaves
        ),
        key=lambda payload: payload["source_content_start"],
    )
    assert all(payload is not None for payload in payloads)
    assert payloads[0]["source_content_start"] == len(first_header)
    assert payloads[-1]["source_content_end"] == len(content)
    assert all(
        right["source_content_start"] == left["source_content_end"]
        for left, right in zip(payloads, payloads[1:])
    )
    assert "".join(payload["content"] for payload in payloads) == content[
        len(first_header):
    ]
    second_header_end = content.index(second_header) + len(second_header)
    later_continuations = [
        payload
        for payload in payloads
        if payload["source_content_start"] >= second_header_end
    ]
    assert later_continuations
    assert all(
        payload["source_fragment_context"]["content"] == second_header
        and payload["source_fragment_context"]["source_content_start"]
        == content.index(second_header)
        and payload["source_fragment_context"][
            "applies_through_source_content_end"
        ] >= payload["source_content_end"]
        for payload in later_continuations
    )


class _HeaderDependentTableLLM:
    def __init__(self):
        self.payloads: list[dict] = []

    def complete(self, request: LLMRequest) -> str:
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        self.payloads.append(payload)
        context = payload.get("source_fragment_context", {})
        has_header = (
            "| person | preferred database |" in payload["content"]
            or "| person | preferred database |" in context.get("content", "")
        )
        triples = []
        if (
            "| Alice | PostgreSQL |" in payload["content"]
            and has_header
            and "VERIFICATION PASS" not in request.system
        ):
            triples.append({
                "subject": "Alice",
                "predicate": "prefers",
                "object": "PostgreSQL",
                "polarity": 1,
                "source_message_id": payload["source_message_id"],
            })
        return _complete({"triples": triples, "markers": []})


class _IntroducedTableLLM:
    def __init__(self, *, expected_prelude: str):
        self.expected_prelude = expected_prelude
        self.payloads: list[dict] = []

    def complete(self, request: LLMRequest) -> str:
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        self.payloads.append(payload)
        context = payload.get("source_fragment_context", {})
        has_semantics = (
            "# Results" in payload["content"]
            and "| person | selected database |" in payload["content"]
        ) or (
            context.get("kind")
            == "introduced_canonical_markdown_table_header"
            and context.get("prelude_content") == self.expected_prelude
            and "| person | selected database |" in context.get("content", "")
        )
        triples = []
        if (
            "| Alice | PostgreSQL |" in payload["content"]
            and has_semantics
            and "VERIFICATION PASS" not in request.system
        ):
            triples.append({
                "subject": "Alice",
                "predicate": "prefers",
                "object": "PostgreSQL",
                "polarity": 1,
                "source_message_id": payload["source_message_id"],
            })
        return _complete({"triples": triples, "markers": []})


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_large_heading_introduced_table_reaches_provider_with_both_contexts(
    newline: str,
):
    prelude = "# Results" + newline
    rows = [
        f"| person_{index:03d} | database_{index:03d} |"
        for index in range(300)
    ]
    rows[240] = "| Alice | PostgreSQL |"
    content = prelude + newline + newline.join([
        "| person | selected database |",
        "| --- | --- |",
        *rows,
    ])
    llm = _IntroducedTableLLM(expected_prelude=prelude)

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert result.completion_calls > 0
    assert [(item.subject, item.predicate, item.object) for item in result.triples] == [
        ("Alice", "prefers", "PostgreSQL")
    ]
    carrying = [
        payload for payload in llm.payloads
        if "| Alice | PostgreSQL |" in payload["content"]
    ]
    assert carrying
    assert all(
        payload["source_fragment_context"]["prelude_content"] == prelude
        and payload["source_fragment_context"]["content"].startswith(
            "| person | selected database |" + newline
        )
        and prelude not in payload["content"]
        for payload in carrying
    )


def test_table_cut_does_not_bypass_an_unrepresented_heading_relationship():
    content = "\n".join([
        "# Deployment inventory",
        "",
        "The selected services are:",
        "",
        "| person | selected database |",
        "| --- | --- |",
        *(f"| person_{index:03d} | database_{index:03d} |" for index in range(300)),
    ])
    analysis = chunk_extraction._markdown_block_analysis(content)

    assert analysis.protected_spans == ((0, len(content)),)
    assert analysis.contextual_table_boundaries == ()
    assert chunk_extraction._semantic_split_point(content) is None


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_later_table_fragment_gets_explicit_exact_header_context(
    newline: str,
):
    rows = [
        f"| person_{index} | database_{index} |" for index in range(260)
    ]
    rows[210] = "| Alice | PostgreSQL |"
    content = newline.join([
        "| person | preferred database |",
        "| --- | --- |",
        *rows,
    ])
    header = newline.join([
        "| person | preferred database |",
        "| --- | --- |",
        "",
    ])
    llm = _HeaderDependentTableLLM()

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert [(item.subject, item.predicate, item.object) for item in result.triples] == [
        ("Alice", "prefers", "PostgreSQL")
    ]
    carrying = [
        payload for payload in llm.payloads
        if "| Alice | PostgreSQL |" in payload["content"]
    ]
    assert carrying
    for payload in carrying:
        context = payload["source_fragment_context"]
        assert context["version"] == (
            chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION
        )
        assert context["kind"] == "canonical_markdown_table_header"
        assert context["content"] == header
        assert context["source_content_start"] == 0
        assert context["source_content_end"] == len(header)
        assert context["source_content_end"] <= payload["source_content_start"]
        assert "| person | preferred database |" not in payload["content"]
        assert payload["content"] == content[
            payload["source_content_start"]:payload["source_content_end"]
        ]


def test_source_less_table_holds_when_header_context_cannot_be_carried():
    content = "\n".join([
        "| person | preferred database |",
        "| --- | --- |",
        *(f"| person_{index} | database_{index} |" for index in range(260)),
    ])
    assert chunk_extraction._semantic_split_point(content) in (
        chunk_extraction._markdown_table_boundary_points(content)
    )
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(llm, content)

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_lone_cr_table_never_authorizes_a_headerless_fragment():
    content = "\r".join([
        "| person | preferred database |",
        "| --- | --- |",
        *(f"| person_{index} | database_{index} |" for index in range(260)),
    ])
    assert len(content) > chunk_extraction._MAX_LEAF_INPUT_CHARS
    assert chunk_extraction._markdown_table_boundary_points(content) == []
    assert chunk_extraction._semantic_split_point(content) is None
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(
        llm,
        "ignored",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_generated_table_context_validation_failure_holds_before_provider(
    monkeypatch,
):
    content = "\n".join([
        "| person | preferred database |",
        "| --- | --- |",
        *(f"| person_{index} | database_{index} |" for index in range(260)),
    ])
    monkeypatch.setattr(
        chunk_extraction,
        "_canonical_table_context_for_cut",
        lambda *_args, **_kwargs: {
            "version": chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION,
            "kind": "canonical_markdown_table_header",
            "content": "not a canonical header\n",
            "source_content_start": 0,
            "source_content_end": 23,
            "applies_through_source_content_end": len(content),
        },
    )
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(
        llm,
        "ignored",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_full_source_record_cannot_supply_fragment_context():
    message_id, encoded = _source_record(17, "ordinary source")
    payload = json.loads(encoded)
    payload["source_fragment_context"] = {
        "version": chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION,
        "kind": "canonical_markdown_table_header",
        "content": "| key | value |\n| --- | --- |\n",
        "source_content_start": 0,
        "source_content_end": 34,
        "applies_through_source_content_end": 99,
    }
    record = (
        message_id,
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(llm, "ignored", source_records=(record,))

    assert result.failed is True
    assert result.failure_reason == "input_contract_failure"
    assert result.failure_details == ("source_records:invalid",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def _external_boundary_fragment_record() -> tuple[int, str]:
    preceding = "Avery owns Bicycle. "
    fragment = "Next section continues."
    payload = json.loads(_source_record(17, fragment)[1])
    payload.update({
        "source_record_version": "hymem-claim-source-fragment-v2",
        "source_content_start": len(preceding),
        "source_content_end": len(preceding) + len(fragment),
        "source_boundary_context": {
            "version": chunk_extraction.SOURCE_BOUNDARY_CONTEXT_VERSION,
            "kind": "preceding_adjacent_prose",
            "content": preceding,
            "source_content_start": 0,
            "source_content_end": len(preceding),
            "applies_through_source_content_end": (
                len(preceding) + len(fragment)
            ),
        },
    })
    return 17, json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_public_boundary_rejects_external_prose_context_fragment():
    record = _external_boundary_fragment_record()
    assert chunk_extraction._source_payload(record) is not None
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(llm, "ignored", source_records=(record,))

    assert result.failed is True
    assert result.failure_reason == "input_contract_failure"
    assert result.failure_details == ("source_records:invalid",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_full_source_record_cannot_smuggle_prose_boundary_context():
    message_id, encoded = _source_record(17, "Next section continues.")
    payload = json.loads(encoded)
    payload["source_boundary_context"] = json.loads(
        _external_boundary_fragment_record()[1]
    )["source_boundary_context"]
    record = (
        message_id,
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(llm, "ignored", source_records=(record,))

    assert result.failed is True
    assert result.failure_reason == "input_contract_failure"
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_prose_boundary_context_must_be_exactly_adjacent():
    message_id, encoded = _external_boundary_fragment_record()
    payload = json.loads(encoded)
    payload["source_boundary_context"]["source_content_end"] -= 1
    record = (
        message_id,
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )

    assert chunk_extraction._source_payload(record) is None


def _external_fragment_record(
    *, context_content: str = "| key | value |\n| --- | --- |\n",
) -> tuple[int, str]:
    row = "| Alice | PostgreSQL |"
    start = len(context_content)
    payload = json.loads(_source_record(17, row)[1])
    payload.update({
        "source_record_version": "hymem-claim-source-fragment-v2",
        "source_content_start": start,
        "source_content_end": start + len(row),
        "source_fragment_context": {
            "version": chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION,
            "kind": "canonical_markdown_table_header",
            "content": context_content,
            "source_content_start": 0,
            "source_content_end": start,
            "applies_through_source_content_end": start + len(row),
        },
    })
    return 17, json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_public_boundary_rejects_even_a_well_formed_external_fragment():
    record = _external_fragment_record()
    assert chunk_extraction._source_payload(record) is not None
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(llm, "ignored", source_records=(record,))

    assert result.failed is True
    assert result.failure_reason == "input_contract_failure"
    assert result.failure_details == ("source_records:invalid",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_public_boundary_rejects_spoofed_introduced_table_context():
    prelude = "# Results\n"
    separator = "\n"
    header = "| person | database |\n| --- | --- |\n"
    row = "| Alice | PostgreSQL |"
    header_start = len(prelude + separator)
    fragment_start = header_start + len(header)
    payload = json.loads(_source_record(17, row)[1])
    payload.update({
        "source_record_version": "hymem-claim-source-fragment-v2",
        "source_content_start": fragment_start,
        "source_content_end": fragment_start + len(row),
        "source_fragment_context": {
            "version": chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION,
            "kind": "introduced_canonical_markdown_table_header",
            "content": header,
            "source_content_start": header_start,
            "source_content_end": fragment_start,
            "applies_through_source_content_end": fragment_start + len(row),
            "prelude_kind": "atx_heading",
            "prelude_content": prelude,
            "prelude_source_content_start": 0,
            "prelude_source_content_end": len(prelude),
        },
    })
    record = (
        17,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    assert chunk_extraction._source_payload(record) is not None
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(llm, "ignored", source_records=(record,))

    assert result.failed is True
    assert result.failure_reason == "input_contract_failure"
    assert result.failure_details == ("source_records:invalid",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


@pytest.mark.parametrize(
    "context_content",
    [
        "arbitrary non-table context\n",
        "| key | value |\n| --- |\n",
        "| key | value |\n| -- | --- |\n",
        "| key | value |\n| --- | --- |\n| body | row |\n",
        "| key | value |\n| --- | --- |",
    ],
    ids=[
        "not-table",
        "column-mismatch",
        "bad-delimiter",
        "extra-body-row",
        "missing-final-eol",
    ],
)
def test_fragment_context_requires_exact_canonical_header_and_delimiter(
    context_content: str,
):
    assert chunk_extraction._source_payload(
        _external_fragment_record(context_content=context_content)
    ) is None


def test_fragment_context_is_dropped_at_its_exact_applicability_end():
    header = "| key | value |\n| --- | --- |\n"
    table_row = "| Alice | PostgreSQL |\n"
    trailing = "Independent prose."
    start = len(header)
    payload = json.loads(_source_record(17, table_row + trailing)[1])
    payload.update({
        "source_record_version": "hymem-claim-source-fragment-v2",
        "source_content_start": start,
        "source_content_end": start + len(table_row) + len(trailing),
        "source_fragment_context": {
            "version": chunk_extraction.SOURCE_FRAGMENT_CONTEXT_VERSION,
            "kind": "canonical_markdown_table_header",
            "content": header,
            "source_content_start": 0,
            "source_content_end": len(header),
            "applies_through_source_content_end": start + len(table_row),
        },
    })
    record = (17, json.dumps(payload, sort_keys=True, separators=(",", ":")))
    assert chunk_extraction._source_payload(record) is not None

    inside = chunk_extraction._fragment_record(
        record, start=0, end=len(table_row)
    )
    after = chunk_extraction._fragment_record(
        record,
        start=len(table_row),
        end=len(table_row) + len(trailing),
    )

    assert inside is not None and after is not None
    inside_payload = chunk_extraction._source_payload(inside)
    after_payload = chunk_extraction._source_payload(after)
    assert inside_payload is not None and after_payload is not None
    assert "source_fragment_context" in inside_payload
    assert "source_fragment_context" not in after_payload
    assert after_payload["source_content_start"] == start + len(table_row)
    assert after_payload["content"] == trailing


def test_block_analysis_scales_on_a_legal_dense_message():
    blocks = []
    index = 0
    total_chars = 0
    while total_chars < 92_000:
        block = (
            "```text\n"
            f"code_{index}. Another sentence.\n"
            "```\n\n"
            f"- item_{index} " + ("x" * 28) + ".\n"
            "  continuation remains atomic.\n\n"
        )
        blocks.append(block)
        total_chars += len(block)
        index += 1
    content = "".join(blocks)
    assert 92_000 <= len(content) <= 100_000
    assert index > 500

    started = time.perf_counter()
    cut = chunk_extraction._semantic_split_point(content)
    elapsed = time.perf_counter() - started

    assert cut is not None
    assert content[:cut] + content[cut:] == content
    assert elapsed < 5.0


@pytest.mark.parametrize("terminal", [".", "?", "!"], ids=["period", "question", "bang"])
def test_sentence_punctuation_inside_table_cell_never_bisects_row(
    terminal: str,
):
    content = "\n".join([
        "| claim | context |",
        "| --- | --- |",
        *(
            f"| claim_{index} | Alpha depends on Beta{terminal} "
            f"More same-cell context {index} {'x' * 24} |"
            for index in range(100)
        ),
    ])
    raw_sentence_candidates = [
        match.start()
        for match in chunk_extraction._SENTENCE_BOUNDARY_RE.finditer(content)
    ]
    row_points = chunk_extraction._markdown_table_boundary_points(content)
    cut = chunk_extraction._semantic_split_point(content)

    assert raw_sentence_candidates, "fixture must exercise sentence detection"
    assert chunk_extraction._sentence_boundary_points(content) == []
    assert cut in row_points
    assert content[:cut] + content[cut:] == content
    assert content[:cut].endswith("|\n")
    assert content[cut:].startswith("| claim_")
    assert all(
        content.rfind("\n", 0, candidate) < candidate
        < content.find("\n", candidate)
        for candidate in raw_sentence_candidates[:-1]
    )


@pytest.mark.parametrize(
    "content",
    [
        "\n".join(
            f"service_{index} | depends_on | database_{index}"
            for index in range(120)
        ),
        "\n".join(
            f"| service_{index} |"
            for index in range(320)
        ),
        "\n".join(
            (
                f"| service_{index} | depends_on | database_{index} |"
                if index % 3 != 2
                else f"| malformed_{index} |"
            )
            for index in range(150)
        ),
        "\n".join([
            "| service | dependency |",
            "| -- | --- |",
            *(f"| service_{index} | database_{index} |" for index in range(160)),
        ]),
        "\n".join(
            f"| `service_{index}|alias` | database_{index} |"
            for index in range(130)
        ),
    ],
    ids=[
        "missing-outer-pipes",
        "one-column-pipe-prose",
        "inconsistent-columns",
        "malformed-delimiter",
        "inline-code-pipes",
    ],
)
def test_malformed_pipe_prose_does_not_enable_soft_newline_splits(content: str):
    assert len(content) > chunk_extraction._MAX_LEAF_INPUT_CHARS
    assert chunk_extraction._markdown_table_boundary_points(content) == []
    assert chunk_extraction._semantic_split_point(content) is None

    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )
    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("split:no_admissible_semantic_boundary",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


@pytest.mark.parametrize("terminal", ["3.14.", "v2.10.3."])
def test_completed_decimal_or_version_sentence_can_still_split(terminal: str):
    left = ("x" * 200) + f" release {terminal} "
    right = "Next sentence " + ("y" * 200)

    assert chunk_extraction._semantic_split_point(left + right) == len(left)


@pytest.mark.parametrize("terminal", ["?", "!"])
def test_question_or_exclamation_true_sentence_can_still_split(terminal: str):
    left = ("x" * 200) + f" complete{terminal} "
    right = "Next sentence " + ("y" * 200)

    assert chunk_extraction._semantic_split_point(left + right) == len(left)


@pytest.mark.parametrize(
    "body",
    [
        "The ratio 3.14 remains exact",
        "Runtime v2.10.3 remains pinned",
        "J. R. R. Tolkien reviewed it",
        "The U.S. East cluster remains active",
    ],
    ids=["decimal", "version", "chained-initials", "initialism"],
)
def test_internal_period_forms_are_not_sentence_boundaries(body: str):
    content = ("x" * 110) + " " + body + " " + ("y" * 110)

    assert chunk_extraction._semantic_split_point(content) is None


def test_literal_unseparated_midpoint_claim_is_held_without_provider_spend():
    """A terminal followed by token data is not a safe sentence boundary."""

    content = ("x" * 2100) + _MIDPOINT_CROSSING_CLAIM + ("y" * 2100)
    legacy_midpoint = len(content) // 2
    assert _MIDPOINT_CROSSING_CLAIM not in content[:legacy_midpoint + 64]
    assert _MIDPOINT_CROSSING_CLAIM not in content[legacy_midpoint - 64:]
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert "split:no_admissible_semantic_boundary" in result.failure_details
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_soft_newline_inside_padded_claim_is_held_without_provider_spend():
    """A display wrap cannot authorize two independently empty fragments."""

    claim = "The deployment depends\non PostgreSQL."
    content = ("x" * 2100) + claim + ("y" * 2100)
    soft_cut = content.index("\n") + 1
    assert "deployment depends" in content[:soft_cut]
    assert "on PostgreSQL" in content[soft_cut:]
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert "split:no_admissible_semantic_boundary" in result.failure_details
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_source_less_unbreakable_claim_cannot_become_authoritative_empty():
    text = "Application uses " + ("x" * 374) + " PostgreSQL."
    legacy_midpoint = len(text) // 2
    for fragment in (text[:legacy_midpoint + 64], text[legacy_midpoint - 64:]):
        assert not (
            "Application uses" in fragment and "PostgreSQL" in fragment
        )
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))

    result = extract_chunk(llm, text)

    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert "split:no_admissible_semantic_boundary" in result.failure_details
    assert result.triples == [] and result.markers == []
    # Split safety cannot be established, but the terminal primary empty still
    # receives exactly one genuine second look before the unit is held.
    assert result.completion_calls == result.provider_attempts == 2
    assert len(llm.calls) == 2
    assert "VERIFICATION PASS" not in llm.calls[0].system
    assert "EMPTY VERIFICATION PASS" in llm.calls[1].system


class _OverflowThenSplitLLM:
    def __init__(self):
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        has_left = "LEFT_OVERFLOW" in request.user
        has_right = "RIGHT_OVERFLOW" in request.user
        if has_left and has_right:
            # Even a plausible-looking partial item is not publishable.
            return json.dumps({
                "triples": [{
                    "subject": "partial", "predicate": "uses",
                    "object": "discard_me", "polarity": 1,
                }],
                "markers": [],
                "complete": False,
            })
        triples = []
        if has_left:
            triples.append({
                "subject": "app", "predicate": "uses", "object": "left",
                "polarity": 1,
            })
        if has_right:
            triples.append({
                "subject": "app", "predicate": "uses", "object": "right",
                "polarity": 1,
            })
        return _complete({"triples": triples, "markers": []})


def test_explicit_incomplete_signal_splits_and_never_publishes_partial_items():
    llm = _OverflowThenSplitLLM()
    text = (
        "LEFT_OVERFLOW " + ("x" * 125) + ".\n"
        + ("y" * 125) + " RIGHT_OVERFLOW"
    )
    result = extract_chunk(llm, text)
    assert result.failed is False
    assert {triple.object for triple in result.triples} == {"left", "right"}
    assert "discard_me" not in {triple.object for triple in result.triples}


class _SaturatedThenSplitLLM:
    def __init__(self, kind: str):
        self.kind = kind
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        has_left = "LEFT_SATURATION" in request.user
        has_right = "RIGHT_SATURATION" in request.user
        if has_left and has_right:
            if self.kind == "triples":
                return _complete({
                    "triples": [{
                        "subject": f"partial_{index}", "predicate": "uses",
                        "object": "discard_me", "polarity": 1,
                    } for index in range(24)],
                    "markers": [],
                })
            return _complete({
                "triples": [],
                "markers": [{
                    "kind": "preference", "statement": f"discard {index}",
                } for index in range(12)],
            })
        if self.kind == "triples":
            triples = []
            if has_left:
                triples.append({
                    "subject": "app", "predicate": "uses",
                    "object": "left", "polarity": 1,
                })
            if has_right:
                triples.append({
                    "subject": "app", "predicate": "uses",
                    "object": "right", "polarity": 1,
                })
            return _complete({"triples": triples, "markers": []})
        markers = []
        if has_left:
            markers.append({"kind": "preference", "statement": "keep left"})
        if has_right:
            markers.append({"kind": "preference", "statement": "keep right"})
        return _complete({"triples": [], "markers": markers})


@pytest.mark.parametrize("kind", ["triples", "markers"])
def test_exact_output_cap_is_saturation_and_cannot_publish_partial_items(kind):
    llm = _SaturatedThenSplitLLM(kind)
    text = (
        "LEFT_SATURATION " + ("x" * 125) + ".\n"
        + ("y" * 125) + " RIGHT_SATURATION"
    )
    result = extract_chunk(llm, text)
    assert result.failed is False
    # One saturated parent plus primary+omission verification for both leaves.
    assert len(llm.calls) == 5
    if kind == "triples":
        assert {triple.object for triple in result.triples} == {"left", "right"}
        assert "discard_me" not in {triple.object for triple in result.triples}
    else:
        assert {marker.statement for marker in result.markers} == {
            "keep left", "keep right",
        }
        assert not any(marker.statement.startswith("discard") for marker in result.markers)


def test_hard_call_cap_fails_atomically(monkeypatch):
    monkeypatch.setattr(
        chunk_extraction, "MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK", 3
    )
    llm = StubLLMClient(default=json.dumps({
        "triples": [], "markers": [], "complete": False,
    }))
    result = extract_chunk(llm, "LEFT. " + ("safe sentence. " * 70) + "RIGHT.")
    assert result.failed is True
    assert len(llm.calls) == 3
    assert result.triples == [] and result.markers == []
    assert any("calls:max_exceeded" in item for item in result.failure_details)


def test_default_ingest_size_fits_prepartition_envelope():
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, "bounded sentence. " * 5555),),
    )
    assert result.failed is False
    # 32 prepartition leaves, each requiring initial + empty verification.
    assert len(llm.calls) == 64
    assert result.completion_calls == result.provider_attempts == 64


def test_prepartition_envelope_fails_before_any_provider_call(monkeypatch):
    monkeypatch.setattr(chunk_extraction, "_MAX_PREPARTITION_LEAVES", 1)
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, "bounded sentence. " * 280),),
    )
    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == ("input:prepartition_limit_exceeded",)
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_unbreakable_oversized_source_fails_closed_before_provider_call():
    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, "x" * 5000),),
    )
    assert result.failed is True
    assert result.failure_reason == "resource_limit"
    assert result.failure_details == (
        "split:no_admissible_semantic_boundary",
    )
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == 0
    assert llm.calls == []


def test_malformed_optional_hints_are_removed_without_losing_core_claim():
    result = extract_chunk(StubLLMClient(default=_complete({
        "triples": [{
            "subject": " app ", "predicate": "uses", "object": " db ",
            "polarity": 1, "subject_type": "NOT_A_TYPE",
            "object_properties": {"owner": 7, " Language ": " SQL "},
            "value_numeric": "seven",
        }],
        "markers": [],
    })), "plain text")
    assert result.failed is False
    assert [(item.subject, item.object) for item in result.triples] == [("app", "db")]
    assert result.entity_type_hints == {}
    assert result.entity_property_hints == {"db": {"language": "SQL"}}
    assert result.triples[0].value_numeric is None


def test_canonical_decimal_string_source_id_is_normalized_before_membership_check():
    result = extract_chunk(
        StubLLMClient(default=_complete({
            "triples": [{
                "subject": " app ", "predicate": " USES ", "object": " db ",
                "polarity": 1, "source_message_id": "17",
            }],
            "markers": [],
        })),
        "ignored legacy rendering",
        source_records=(_source_record(17, "The app uses the db."),),
    )
    assert result.failed is False
    assert len(result.triples) == 1
    assert result.triples[0].predicate == "uses"
    assert result.triples[0].source_message_id == 17


@pytest.mark.parametrize(
    ("source_value", "expected_detail"),
    [
        (None, "triples[0].source_message_id:missing"),
        ("017", "triples[0].source_message_id:not_positive_integer"),
        ("18", "triples[0].source_message_id:not_in_input"),
        ("9" * 5000, "triples[0].source_message_id:not_positive_integer"),
    ],
)
def test_missing_noncanonical_or_foreign_source_id_remains_fail_closed(
    source_value, expected_detail,
):
    triple = {
        "subject": "app", "predicate": "uses", "object": "db", "polarity": 1,
    }
    if source_value is not None:
        triple["source_message_id"] = source_value
    result = extract_chunk(
        StubLLMClient(default=_complete({"triples": [triple], "markers": []})),
        "ignored legacy rendering",
        source_records=(_source_record(17, "The app uses the db."),),
    )
    assert result.failed is True
    assert result.triples == []
    assert expected_detail in result.failure_details


def test_invalid_core_field_fails_closed_with_exact_field_diagnostic():
    result = extract_chunk(StubLLMClient(default=_complete({
        "triples": [{
            "subject": "app", "predicate": "invented_predicate",
            "object": "db", "polarity": 1,
        }],
        "markers": [],
    })), "plain text")
    assert result.failed is True
    assert result.triples == []
    assert "triples[0].predicate:not_allowed" in result.failure_details


class _SuspiciousEmptyLLM:
    def __init__(self, *, recover: bool):
        self.recover = recover
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if self.recover and "VERIFICATION PASS" in request.system:
            return _complete({
                "triples": [{
                    "subject": "user", "predicate": "prefers",
                    "object": "PostgreSQL", "polarity": 1,
                }],
                "markers": [],
            })
        return _complete({"triples": [], "markers": []})


def test_suspicious_empty_gets_one_bounded_verification_and_can_recover():
    llm = _SuspiciousEmptyLLM(recover=True)
    result = extract_chunk(llm, "I prefer PostgreSQL.")
    assert result.failed is False
    assert [triple.object for triple in result.triples] == ["PostgreSQL"]
    assert len(llm.calls) == 3
    assert "VERIFICATION PASS" in llm.calls[1].system
    assert "OMISSION VERIFICATION PASS" in llm.calls[2].system


def test_repeated_suspicious_empty_becomes_authoritative_after_one_verification():
    llm = _SuspiciousEmptyLLM(recover=False)
    result = extract_chunk(llm, "I prefer PostgreSQL.")
    assert result.failed is False
    assert result.triples == [] and result.markers == []
    assert len(llm.calls) == 2
    assert result.completion_calls == result.provider_attempts == 2


@pytest.mark.parametrize("text", [
    "Thanks for the explanation.",
    "Module A implements Protocol B.",
    "Module A is part of Service B.",
    "Bibliothek A implementiert Protokoll B.",
])
def test_every_clean_empty_requires_one_bounded_verification(text):
    llm = _SuspiciousEmptyLLM(recover=False)
    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, text),),
    )
    assert result.failed is False
    assert result.triples == [] and result.markers == []
    assert len(llm.calls) == 2
    assert "VERIFICATION PASS" not in llm.calls[0].system
    assert "VERIFICATION PASS" in llm.calls[1].system
    assert result.completion_calls == result.provider_attempts == 2


def test_marker_kind_whitespace_and_case_are_normalized():
    result = extract_chunk(
        StubLLMClient(default=_complete({
            "triples": [],
            "markers": [{
                "kind": " Preference ",
                "statement": "  prefers concise answers  ",
            }],
        })),
        "Please keep answers concise.",
    )
    assert result.failed is False
    assert result.markers == [
        Marker(kind="preference", statement="prefers concise answers")
    ]
    assert result.completion_calls == result.provider_attempts == 2


class _OmissionSequenceLLM:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if not self.responses:
            raise AssertionError("omission verifier made an unbounded extra call")
        return self.responses.pop(0)


def _cited_claim(obj: str, *, polarity: int = 1, source_id: int = 17) -> dict:
    return {
        "subject": "user",
        "predicate": "uses",
        "object": obj,
        "polarity": polarity,
        "source_message_id": source_id,
    }


class _EmptyVerificationOnlyPreferenceLLM:
    """Reproduce a provider that notices the claim only on second look."""

    def __init__(self, *, attempts_per_call: int = 1):
        self.calls: list[LLMRequest] = []
        self.request_attempts = 0
        self.attempts_per_call = attempts_per_call

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        self.request_attempts += self.attempts_per_call
        if (
            "EMPTY VERIFICATION PASS" in request.system
            and "I prefer PostgreSQL" in request.user
        ):
            return _complete({
                "triples": [{
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "PostgreSQL",
                    "polarity": 1,
                    "source_message_id": 17,
                }],
                "markers": [],
            })
        return _complete({"triples": [], "markers": []})


def _padded_split_preference() -> str:
    # Exactly one useful root split: both 96+-character children are below the
    # 192-character subdivision floor.  Under the former verify_empty=False
    # child calls, their two empty primaries were silently authoritative.
    return (
        "Neutral filler " + ("x" * 100) + ". "
        "I prefer PostgreSQL. " + ("y" * 100)
    )


@pytest.mark.parametrize(
    ("content", "expected_calls", "expected_primary", "expected_empty"),
    [
        ("I prefer PostgreSQL.", 3, 1, 1),
        (_padded_split_preference(), 6, 3, 2),
    ],
    ids=["short", "padded-source-split"],
)
def test_empty_verification_only_claim_recovers_on_every_terminal_leaf(
    content, expected_calls, expected_primary, expected_empty,
):
    llm = _EmptyVerificationOnlyPreferenceLLM(attempts_per_call=3)

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert [(triple.predicate, triple.object) for triple in result.triples] == [
        ("prefers", "PostgreSQL")
    ]
    assert result.completion_calls == len(llm.calls) == expected_calls
    assert result.provider_attempts == llm.request_attempts == expected_calls * 3
    empty_calls = [
        call for call in llm.calls
        if "EMPTY VERIFICATION PASS" in call.system
    ]
    omission_calls = [
        call for call in llm.calls
        if "OMISSION VERIFICATION PASS" in call.system
    ]
    primary_calls = [
        call for call in llm.calls if "VERIFICATION PASS" not in call.system
    ]
    assert len(primary_calls) == expected_primary
    assert len(empty_calls) == expected_empty
    assert len(omission_calls) == 1
    assert len({call.user for call in empty_calls}) == len(empty_calls)


def test_split_empty_recovery_fails_atomically_when_omission_exceeds_call_cap():
    llm = _EmptyVerificationOnlyPreferenceLLM()

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, _padded_split_preference()),),
        completion_call_limit=5,
    )

    assert result.failed is True
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == 5
    assert len(llm.calls) == 5
    assert sum(
        "EMPTY VERIFICATION PASS" in call.system for call in llm.calls
    ) == 2
    assert not any(
        "OMISSION VERIFICATION PASS" in call.system for call in llm.calls
    )
    assert "right:branch_incomplete" in result.failure_details
    assert "right.right:resource_limit" in result.failure_details
    assert "right.right.calls:max_exceeded" in result.failure_details


def test_empty_verification_recovery_gets_its_own_omission_pass():
    source = _source_record(
        17, "I use PostgreSQL for storage and Redis for caching."
    )
    llm = _OmissionSequenceLLM(
        _complete({"triples": [], "markers": []}),
        _complete({"triples": [_cited_claim("PostgreSQL")], "markers": []}),
        _complete({"triples": [_cited_claim("Redis")], "markers": []}),
    )

    result = extract_chunk(
        llm, "ignored legacy rendering", source_records=(source,)
    )

    assert result.failed is False
    assert {triple.object for triple in result.triples} == {
        "PostgreSQL", "Redis",
    }
    assert result.completion_calls == result.provider_attempts == 3
    assert len(llm.calls) == 3
    assert "VERIFICATION PASS" in llm.calls[1].system
    assert "OMISSION VERIFICATION PASS" in llm.calls[2].system
    assert '"object":"PostgreSQL"' in llm.calls[2].user
    sources = [call.user.split('"""', 2)[1].strip() for call in llm.calls]
    assert sources == [source[1], source[1], source[1]]


def test_terminal_retry_recovery_gets_its_own_omission_pass():
    source = _source_record(
        17, "I use PostgreSQL for storage and Redis for caching."
    )
    llm = _OmissionSequenceLLM(
        '{"triples": [',
        _complete({"triples": [_cited_claim("PostgreSQL")], "markers": []}),
        _complete({"triples": [_cited_claim("Redis")], "markers": []}),
    )

    result = extract_chunk(
        llm, "ignored legacy rendering", source_records=(source,)
    )

    assert result.failed is False
    assert {triple.object for triple in result.triples} == {
        "PostgreSQL", "Redis",
    }
    assert result.completion_calls == result.provider_attempts == 3
    assert len(llm.calls) == 3
    assert "VERIFICATION PASS" in llm.calls[1].system
    assert "OMISSION VERIFICATION PASS" in llm.calls[2].system
    assert '"object":"PostgreSQL"' in llm.calls[2].user


@pytest.mark.parametrize(
    "first_response",
    [
        _complete({"triples": [], "markers": []}),
        '{"triples": [',
    ],
    ids=["empty-verification", "terminal-retry"],
)
def test_recovered_nonempty_fails_atomically_without_omission_call_budget(
    monkeypatch, first_response,
):
    monkeypatch.setattr(
        chunk_extraction, "MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK", 2
    )
    llm = _OmissionSequenceLLM(
        first_response,
        _complete({"triples": [_cited_claim("PostgreSQL")], "markers": []}),
    )

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, "I use PostgreSQL."),),
    )

    assert result.failed is True
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == 2
    assert len(llm.calls) == 2
    assert "right:resource_limit" in result.failure_details
    assert "right.calls:max_exceeded" in result.failure_details


def test_nonempty_omission_verifier_recovers_second_explicit_claim_once():
    source = _source_record(
        17, "I use PostgreSQL for storage and Redis for caching."
    )
    llm = _OmissionSequenceLLM(
        _complete({"triples": [_cited_claim("PostgreSQL")], "markers": []}),
        _complete({"triples": [_cited_claim("Redis")], "markers": []}),
    )

    result = extract_chunk(
        llm, "ignored legacy rendering", source_records=(source,)
    )

    assert result.failed is False
    assert {triple.object for triple in result.triples} == {
        "PostgreSQL", "Redis",
    }
    assert result.completion_calls == result.provider_attempts == 2
    assert len(llm.calls) == 2
    assert "OMISSION VERIFICATION PASS" in llm.calls[1].system
    assert "Return ONLY missed supported triples and markers" in llm.calls[1].system
    assert '"object":"PostgreSQL"' in llm.calls[1].user
    primary_source = llm.calls[0].user.split('"""', 2)[1].strip()
    verifier_source = llm.calls[1].user.split('"""', 2)[1].strip()
    assert primary_source == verifier_source == source[1]


def test_nonempty_omission_verifier_clean_empty_keeps_primary():
    llm = _OmissionSequenceLLM(
        _complete({"triples": [_cited_claim("PostgreSQL")], "markers": []}),
        _complete({"triples": [], "markers": []}),
    )

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, "I use PostgreSQL."),),
    )

    assert result.failed is False
    assert [triple.object for triple in result.triples] == ["PostgreSQL"]
    assert len(llm.calls) == 2


def test_nonempty_omission_verifier_dedupes_repeated_primary_claim():
    response = _complete({
        "triples": [_cited_claim("PostgreSQL")], "markers": [],
    })
    llm = _OmissionSequenceLLM(response, response)

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, "I use PostgreSQL."),),
    )

    assert result.failed is False
    assert len(result.triples) == 1
    assert result.completion_calls == 2


@pytest.mark.parametrize(
    ("verification", "expected_detail"),
    [
        ("not-json", "right:parse_failure"),
        (
            json.dumps({"triples": [], "markers": [], "complete": False}),
            "right:incomplete_response",
        ),
        (
            _complete({
                "triples": [_cited_claim("PostgreSQL", polarity=-1)],
                "markers": [],
            }),
            "triples:polarity_conflict",
        ),
    ],
)
def test_nonempty_omission_verifier_failure_is_atomic(
    verification, expected_detail,
):
    llm = _OmissionSequenceLLM(
        _complete({"triples": [_cited_claim("PostgreSQL")], "markers": []}),
        verification,
    )

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, "I use PostgreSQL."),),
    )

    assert result.failed is True
    assert result.triples == [] and result.markers == []
    assert len(llm.calls) == 2
    assert expected_detail in result.failure_details


def test_nonempty_omission_verifier_rejects_foreign_source_id_atomically():
    llm = _OmissionSequenceLLM(
        _complete({"triples": [_cited_claim("PostgreSQL")], "markers": []}),
        _complete({
            "triples": [_cited_claim("Redis", source_id=18)], "markers": [],
        }),
    )

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, "I use PostgreSQL and Redis."),),
    )

    assert result.failed is True
    assert result.triples == []
    assert "right:response_conflict" in result.failure_details
    assert "right.triples[0].source_message_id:not_in_input" in result.failure_details


def test_nonempty_omission_verifier_respects_global_call_cap(monkeypatch):
    monkeypatch.setattr(
        chunk_extraction, "MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK", 1
    )
    llm = StubLLMClient(default=_complete({
        "triples": [{
            "subject": "app", "predicate": "uses", "object": "db",
            "polarity": 1,
        }],
        "markers": [],
    }))

    result = extract_chunk(llm, "The app uses the db.")

    assert result.failed is True
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == 1
    assert "right:resource_limit" in result.failure_details
    assert "right.calls:max_exceeded" in result.failure_details


class _VerifierSaturationLLM:
    def __init__(self):
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        payload = json.loads(request.user.split('"""', 2)[1].strip())
        # Labelled boundary context is interpretation-only; this recovery probe
        # deliberately routes on authoritative fragment bytes.
        has_left = "LEFT_VERIFY_SATURATION" in payload["content"]
        has_right = "RIGHT_VERIFY_SATURATION" in payload["content"]
        omission = "OMISSION VERIFICATION PASS" in request.system
        if omission and has_left and has_right:
            return _complete({
                "triples": [{
                    "subject": f"omitted_{index}", "predicate": "uses",
                    "object": "discard_saturated_verifier", "polarity": 1,
                } for index in range(24)],
                "markers": [],
            })
        if omission:
            return _complete({"triples": [], "markers": []})
        triples = []
        if has_left:
            triples.append(_cited_claim("left", source_id=17))
        if has_right:
            triples.append(_cited_claim("right", source_id=17))
        return _complete({"triples": triples, "markers": []})


def test_saturated_omission_verifier_subdivides_source_before_publication():
    llm = _VerifierSaturationLLM()
    content = (
        "LEFT_VERIFY_SATURATION " + ("x" * 125) + ".\n"
        + ("y" * 125) + " RIGHT_VERIFY_SATURATION"
    )

    result = extract_chunk(
        llm,
        "ignored legacy rendering",
        source_records=(_source_record(17, content),),
    )

    assert result.failed is False
    assert {triple.object for triple in result.triples} == {"left", "right"}
    assert "discard_saturated_verifier" not in {
        triple.object for triple in result.triples
    }
    # Full primary + saturated verifier, then primary+verifier for each child.
    assert result.completion_calls == len(llm.calls) == 6


class _FullThirtyTwoLeafRecoveryLLM:
    """Saturate a single binary recovery tree down to exactly 32 leaves."""

    def __init__(self):
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        excerpt = request.user.split('"""', 2)[1]
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete({"triples": [], "markers": []})
        if len(excerpt) > 230:
            return _complete({
                "triples": [{
                    "subject": f"s{index}", "predicate": "uses",
                    "object": "saturation", "polarity": 1,
                } for index in range(24)],
                "markers": [],
            })
        return _complete({
            "triples": [{
                "subject": "app", "predicate": "uses", "object": "db",
                "polarity": 1,
            }],
            "markers": [],
        })


def test_dense_32_leaf_recovery_with_omission_checks_fits_96_call_cap():
    llm = _FullThirtyTwoLeafRecoveryLLM()

    result = extract_chunk(llm, "DENSE. " + ("padding sentence. " * 211))

    # 31 saturated internal primaries + 32 terminal primaries + 32 omission
    # verifiers = 95. The global counter remains authoritative for less regular
    # prepartition+recovery shapes and fails them atomically at call 96.
    assert result.failed is False
    assert len(result.triples) == 1
    assert result.completion_calls == result.provider_attempts == 95
    assert len(llm.calls) == 95
    assert result.completion_calls < (
        chunk_extraction.MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK
    )


def test_unknown_marker_kind_remains_fail_closed():
    result = extract_chunk(
        StubLLMClient(default=_complete({
            "triples": [],
            "markers": [{"kind": " Maybe ", "statement": "uncertain"}],
        })),
        "Maybe.",
    )
    assert result.failed is True
    assert result.failure_reason == "item_validation_failure"
    assert "markers[0].kind:not_allowed" in result.failure_details
    assert result.markers == []


# --- the behavior that flag buys: retry vs. permanent hole ------------------


def _seed_chunk(
    hy: HyMem, chunk_id: str = "c_retry", *, content: str = "msg"
) -> Chunk:
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s_retry')")
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content) "
        "VALUES ('s_retry', 'user', ?)",
        (content,),
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
        (chunk_id, extraction_cache_key(hy.config.prompt_version)),
    ).fetchone() is not None


def test_failed_extraction_is_not_marked_processed(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        _persist(hy, chunk, ChunkExtraction(triples=[], markers=[], failed=True))
        assert not _marked(hy, chunk.id)
    finally:
        hy.close()


def test_v19_processed_marker_is_eligible_under_v20_default(cfg):
    hy = HyMem(cfg)
    try:
        assert hy.config.prompt_version == "v20"
        chunk = _seed_chunk(hy, "v19-ambiguous-period-replay")
        hy.conn.execute(
            "INSERT INTO processed_chunks(chunk_id,prompt_version) VALUES (?,?)",
            (chunk.id, "v19"),
        )

        extraction = phase1.extract_chunk_results(
            hy.conn,
            chunk,
            StubLLMClient(default=_complete({"triples": [], "markers": []})),
            prompt_version=hy.config.prompt_version,
        )

        assert extraction is not None
        assert extraction.failed is False
        assert not hy.conn.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_id=? "
            "AND prompt_version=?",
            (chunk.id, extraction_cache_key(hy.config.prompt_version)),
        ).fetchone()
    finally:
        hy.close()


def test_phase1_runner_holds_unsafe_unsplittable_source_without_processed_row(cfg):
    """The real runner must not translate a split-safety hold into success."""

    llm = StubLLMClient(default=_complete({"triples": [], "markers": []}))
    hy = HyMem(replace(
        cfg,
        dream_budget=1,
        dream_baseline_budget=0,
        salience_min_chars=1,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
    ), llm=llm)
    try:
        unsafe = ("x" * 2100) + _MIDPOINT_CROSSING_CLAIM + ("y" * 2100)
        chunk = _seed_chunk(hy, "unsafe-semantic-unit", content=unsafe)

        report = hy.dream(session_ids=[chunk.session_id])

        assert report.chunk_extraction_failures == 1
        assert not hy.conn.execute("SELECT 1 FROM processed_chunks").fetchone()
        attempt = hy.conn.execute(
            "SELECT last_failure_reason,last_failure_details "
            "FROM chunk_extraction_attempts ORDER BY chunk_id LIMIT 1"
        ).fetchone()
        assert attempt is not None
        assert attempt["last_failure_reason"] == "resource_limit"
        assert json.loads(attempt["last_failure_details"]) == [
            "split:no_admissible_semantic_boundary"
        ]
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
            "complete": True,
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


def test_failed_attempt_persists_exact_safe_validation_diagnostics(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy, "diagnostic-fields")
        llm = StubLLMClient(default=_complete({
            "triples": [{
                "subject": "app", "predicate": "invented_predicate",
                "object": "db", "polarity": 1,
                "source_message_id": chunk.start_message_id,
            }],
            "markers": [],
        }))
        extraction = phase1.extract_chunk_results(
            hy.conn,
            chunk,
            llm,
            prompt_version=hy.config.prompt_version,
        )
        assert extraction is not None and extraction.failed is True
        _persist(hy, chunk, extraction)
        row = hy.conn.execute(
            "SELECT last_failure_reason,last_failure_details "
            "FROM chunk_extraction_attempts WHERE chunk_id=? AND prompt_version=?",
            (chunk.id, extraction_cache_key(hy.config.prompt_version)),
        ).fetchone()
        assert row["last_failure_reason"] == "item_validation_failure"
        details = json.loads(row["last_failure_details"])
        assert "triples[0].predicate:not_allowed" in details
        assert "invented_predicate" not in row["last_failure_details"]
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
            StubLLMClient(default=_complete({"triples": items, "markers": []})),
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
        (chunk_id, extraction_cache_key(hy.config.prompt_version)),
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
        binding = phase1_generation_binding(
            hy.config.prompt_version, StubLLMClient(default="failed")
        )
        failed = ChunkExtraction(
            triples=[], markers=[], failed=True, phase1_generation=binding
        )
        _persist(hy, chunk, failed)
        assert not _marked(hy, chunk.id)

        with caplog.at_level("WARNING"):
            _persist(hy, chunk, failed)
        assert not _marked(hy, chunk.id), "a failure must never become success"
        generation_key = hy.conn.execute(
            "SELECT phase1_generation_key FROM chunk_extraction_attempts "
            "WHERE chunk_id=? AND prompt_version=?",
            (chunk.id, extraction_cache_key(hy.config.prompt_version)),
        ).fetchone()[0]
        assert chunk_extraction_is_quarantined(
            hy.conn,
            chunk.id,
            prompt_version=hy.config.prompt_version,
            max_attempts=2,
            phase1_generation_key=generation_key,
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
        llm = StubLLMClient(default="[]")
        binding = phase1_generation_binding(hy.config.prompt_version, llm)
        _persist(hy, chunk, ChunkExtraction(
            triples=[], markers=[], failed=False, source_validated=True,
            phase1_generation=binding,
        ))
        again = phase1.extract_chunk_results(
            hy.conn, chunk, llm,
            prompt_version=hy.config.prompt_version,
        )
        assert again is None
        assert llm.calls == []
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
                "complete": True,
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
        assert rows == [], "unmanifested prose is never extraction input"
        with core_db.transaction(hy.conn):
            assert record_unrecoverable_chunk_losses(hy.conn, "backlog") == 1
        assert hy.dream_status()["pending_chunks"] == 0
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
        assert status["pending_chunks"] == 0
        assert status["quarantined_chunks"] == 0
        assert status["terminal_loss_chunks"] == 1
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
            "single pass": _complete({"triples": [], "markers": []}),
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
            "AND prompt_version = ?",
            (chunk_id, extraction_cache_key("v11")),
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
                "complete": True,
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
            "AND prompt_version = ?",
            (chunk_id, extraction_cache_key("v12")),
        ).fetchone()
        evidence = replay.conn.execute(
            "SELECT source_role, evidence_weight, weight_source, "
            "extraction_prompt_version FROM kg_evidence WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        assert tuple(evidence) == (
            "user", 2, "configured_role:user", extraction_cache_key("v12")
        )
    finally:
        replay.close()
