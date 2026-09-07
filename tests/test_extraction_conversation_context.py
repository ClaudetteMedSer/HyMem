"""Cross-record context is interpretation, never additional citation ownership."""
from __future__ import annotations

import json

import pytest

from hymem.extraction import chunk
from hymem.extraction import prompts
from hymem.extraction.contract import extraction_contract_identity


def _record(mid, role, content, *, session="s", workspace=None):
    return mid, json.dumps({
        "content": content,
        "source_created_at": "2026-09-07T00:00:00.000Z",
        "source_message_id": mid,
        "source_peer_id": "assistant" if role == "assistant" else "user",
        "source_record_version": "hymem-claim-source-v2",
        "source_role": role,
        "source_session_id": session,
        "source_workspace_id": workspace,
    }, sort_keys=True, separators=(",", ":"))


def _payloads(request):
    return [json.loads(line) for line in request.user.split('"""', 2)[1].strip().splitlines()]


def _response(triples=(), markers=()):
    return json.dumps({"triples": list(triples), "markers": list(markers), "complete": True})


class _ConversationOracle:
    def __init__(self, *, force_recovery=False):
        self.requests = []
        self.force_recovery = force_recovery

    def complete(self, request):
        self.requests.append(request)
        records = _payloads(request)
        owned = [p for p in records if not p.get("source_context_only")]
        if self.force_recovery and len(owned) > 1:
            return '{"triples": ['
        if "OMISSION VERIFICATION PASS" in request.system:
            return _response()
        question = any(
            p["source_role"] == "assistant" and "Do you use PostgreSQL?" in p["content"]
            for p in records
        )
        answer = next((p for p in owned if p["source_message_id"] == 2), None)
        if question and answer and answer["content"] in ("Yes.", "No, SQLite instead."):
            return _response([{
                "subject": "user", "predicate": "uses",
                "object": "PostgreSQL" if answer["content"] == "Yes." else "SQLite",
                "polarity": 1, "source_message_id": 2,
            }])
        return _response()


@pytest.mark.parametrize("answer,expected", [("Yes.", "PostgreSQL"), ("No, SQLite instead.", "SQLite")])
@pytest.mark.parametrize("path", ["prepartition", "recovery"])
def test_confirmation_and_correction_survive_both_split_paths(answer, expected, path):
    padding = "Routine background. " * 180 if path == "prepartition" else ""
    records = (
        _record(1, "assistant", padding + "Do you use PostgreSQL?"),
        _record(2, "user", answer),
    )
    oracle = _ConversationOracle(force_recovery=path == "recovery")

    result = chunk.extract_chunk(oracle, "ignored", source_records=records)

    assert not result.failed
    assert [(t.object, t.source_message_id) for t in result.triples] == [(expected, 2)]
    assert result.initial_prepartition_leaves == (2 if path == "prepartition" else 1)
    assert result.completion_calls == len(oracle.requests) == (4 if path == "prepartition" else 5)
    contextual = [request for request in oracle.requests if any(p.get("source_context_only") for p in _payloads(request))]
    assert len(contextual) == 2
    assert _payloads(contextual[0]) == _payloads(contextual[1]), "omission verifier must see identical context"
    context = _payloads(contextual[0])[0]
    assert context["source_role"] == "assistant"
    assert context["source_peer_id"] == "assistant"
    assert context["context_for_source_message_id"] == 2
    original = json.loads(records[0][1])["content"]
    assert context["content"] == original[context["source_content_start"]:context["source_content_end"]]
    assert all(len(request.user.split('"""', 2)[1].strip()) <= chunk._MAX_LEAF_INPUT_CHARS for request in oracle.requests)


def test_context_does_not_authorize_citations_to_preceding_record():
    records = (_record(1, "assistant", "Do you use PostgreSQL?"), _record(2, "user", "Yes."))
    split = chunk._split_unit(chunk._source_unit(records))
    assert split is not None
    assert split[1].allowed_ids == frozenset({2})
    assert [mid for mid, _ in split[1].context_records] == [1]

    class WrongCitation(_ConversationOracle):
        def complete(self, request):
            result = super().complete(request)
            if any(p.get("source_context_only") for p in _payloads(request)):
                return _response([{"subject": "user", "predicate": "uses", "object": "PostgreSQL", "polarity": 1, "source_message_id": 1}])
            return result

    result = chunk.extract_chunk(WrongCitation(force_recovery=True), "ignored", source_records=records)
    assert result.failed
    assert not result.triples and not result.markers
    assert any("source_message_id:not_in_input" in detail for detail in result.failure_details)


@pytest.mark.parametrize("builder", [prompts.build_chunk_extraction_system, prompts.build_chunk_empty_verification_system, prompts.build_chunk_omission_verification_system])
def test_all_prompt_passes_forbid_relabelled_context_only_claims(builder):
    system = " ".join(builder().split())
    assert "Never return a triple or marker stated wholly in a context-only record, even relabelled with an owned ID" in system
    assert "Never cite a context record's source_message_id" in system
    assert "That owned record must contribute indispensable support" in system


def test_context_only_claim_cannot_be_relabelled_as_owned_confirmation():
    records = (_record(1, "assistant", "Routine background. " * 180 + "Avery owns Bicycle."), _record(2, "user", "Thanks."))

    class OwnershipOracle:
        def __init__(self):
            self.checked = 0

        def complete(self, request):
            if any(p.get("source_context_only") for p in _payloads(request)):
                self.checked += 1
                if "even relabelled with an owned ID" not in request.system:
                    return _response([{"subject": "Avery", "predicate": "owns", "object": "Bicycle", "polarity": 1, "source_message_id": 2}], [{"kind": "preference", "statement": "Avery owns Bicycle."}])
            return _response()

    oracle = OwnershipOracle()
    result = chunk.extract_chunk(oracle, "ignored", source_records=records)
    assert oracle.checked == 2
    assert not result.failed
    assert not result.triples and not result.markers


@pytest.mark.parametrize("different_scope", [{"session": "other"}, {"workspace": "other"}])
def test_conversation_context_never_crosses_session_or_workspace(different_scope):
    records = (_record(1, "assistant", "Do you use PostgreSQL?"), _record(2, "user", "Yes.", **different_scope))
    split = chunk._split_unit(chunk._source_unit(records))
    assert split is not None
    assert split[1].context_records == ()


def test_conversation_context_never_adopts_a_different_users_claim():
    first = _record(1, "user", "I use PostgreSQL.")
    mid, encoded = _record(2, "user", "Yes.")
    payload = json.loads(encoded)
    payload["source_peer_id"] = "different-user"
    split = chunk._split_unit(chunk._source_unit((first, (mid, json.dumps(payload)))))
    assert split is not None
    assert split[1].context_records == ()


def test_context_offsets_for_full_records_come_from_actual_content():
    records = []
    for record in (_record(1, "assistant", "Question?"), _record(2, "user", "Yes.")):
        payload = json.loads(record[1])
        # Full-v2 records historically allow unrelated extra metadata. It must
        # not become fragment provenance merely because context was added.
        payload["source_content_start"] = "not-an-offset"
        payload["source_content_end"] = "not-an-offset"
        records.append((record[0], json.dumps(payload)))
    split = chunk._split_unit(chunk._source_unit(tuple(records)))
    assert split is not None
    context = json.loads(split[1].context_records[0][1])
    assert context["source_content_start"] == 0
    assert context["source_content_end"] == len("Question?")
    assert context["applies_through_source_content_end"] == len("Yes.")


def test_same_peer_dense_table_and_independent_prose_remain_extractable():
    from benchmarks import extraction_canary

    records = []
    for mid, encoded in extraction_canary._source_records():
        payload = json.loads(encoded)
        payload["source_peer_id"] = "same-user"
        records.append((mid, json.dumps(payload, sort_keys=True, separators=(",", ":"))))

    class DenseContextOracle:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            if "OMISSION VERIFICATION PASS" in request.system:
                return _response()
            triples = []
            for payload in _payloads(request):
                if payload.get("source_context_only"):
                    continue
                if extraction_canary._TABLE_CLAIM_ROW in payload["content"]:
                    assert payload.get("source_fragment_context") == extraction_canary._EXPECTED_TABLE_CONTEXT
                    triples.append({"subject": "HyMem Canary Relay", "predicate": "deploys_to", "object": "Fly.io", "polarity": 1, "source_message_id": payload["source_message_id"]})
                if extraction_canary._PROSE_BOUNDARY_RIGHT in payload["content"]:
                    assert payload.get("source_boundary_context") == extraction_canary._EXPECTED_PROSE_BOUNDARY_CONTEXT
                    triples.append({"subject": "Avery Boundary Canary", "predicate": "prefers", "object": "PostgreSQL", "polarity": 1, "source_message_id": payload["source_message_id"]})
            return _response(triples)

    oracle = DenseContextOracle()
    result = chunk.extract_chunk(oracle, "ignored", source_records=tuple(records))
    assert not result.failed
    assert {t.object for t in result.triples} == {"Fly.io", "PostgreSQL"}
    contexts = [p for request in oracle.requests for p in _payloads(request) if p.get("source_context_only")]
    assert contexts
    for context in contexts:
        assert context["source_fragment_context"] == extraction_canary._EXPECTED_TABLE_CONTEXT
        start, end = context["source_content_start"], context["source_content_end"]
        assert context["content"] == extraction_canary._TABLE_CONTENT[start:end]
        assert len(json.dumps(context, sort_keys=True, separators=(",", ":"))) + 1 <= chunk._MAX_CONVERSATION_CONTEXT_ENCODED_CHARS
    assert all(len(request.user.split('"""', 2)[1].strip()) <= chunk._MAX_LEAF_INPUT_CHARS for request in oracle.requests)


def test_unrepresentable_empty_context_metadata_terminates_without_calls():
    record = _record(1, "assistant", "")
    payload = json.loads(record[1])
    payload["source_peer_id"] = "x" * 5000
    oracle = _ConversationOracle()
    result = chunk.extract_chunk(oracle, "ignored", source_records=((1, json.dumps(payload)), _record(2, "user", "Yes.")))
    assert result.failed
    assert not oracle.requests


def test_recursive_single_record_split_expires_conversation_context():
    content = "Yes. " + "First paragraph background. " * 30 + "\n\n" + "Later unrelated paragraph. " * 30
    split = chunk._split_unit(chunk._source_unit((_record(1, "assistant", "Do you use PostgreSQL?"), _record(2, "user", content))))
    assert split is not None and split[1].context_records
    children = chunk._split_unit(split[1])
    assert children is not None
    assert children[0].context_records == split[1].context_records
    assert children[1].context_records == (), "a distant fragment cannot reuse a conversational antecedent"


def test_recursive_multirecord_split_preserves_bounded_adjacent_turns():
    records = tuple(_record(mid, "assistant" if mid < 4 else "user", "Question?" if mid == 2 else "Yes.") for mid in range(1, 5))
    split = chunk._split_unit(chunk._source_unit(records))
    assert split is not None
    children = chunk._split_unit(split[1])
    assert children is not None
    context = [json.loads(encoded) for _, encoded in children[1].context_records]
    assert [p["source_message_id"] for p in context] == [2, 3]
    assert all(p["context_for_source_message_id"] == 4 for p in context)
    assert children[1].allowed_ids == frozenset({4})
    assert sum(len(p["content"]) for p in context) <= chunk._MAX_CONVERSATION_CONTEXT_CHARS
    assert sum(len(encoded) + 1 for _, encoded in children[1].context_records) <= chunk._MAX_CONVERSATION_CONTEXT_ENCODED_CHARS


def test_unsafe_context_truncation_holds_instead_of_dropping_negation():
    first = "Do you NOT " + "very " * 800 + "often use PostgreSQL?"
    oracle = _ConversationOracle()
    result = chunk.extract_chunk(oracle, "ignored", source_records=(_record(1, "assistant", first), _record(2, "user", "Yes.")))
    assert result.failed and result.failure_reason == "resource_limit"
    assert not oracle.requests and not result.triples


@pytest.mark.parametrize("key,value", [("source_context_only", True), ("context_for_source_message_id", 1), ("applies_through_source_content_end", 1)])
def test_caller_cannot_smuggle_context_metadata(key, value):
    mid, encoded = _record(1, "user", "Yes.")
    payload = json.loads(encoded)
    payload[key] = value
    oracle = _ConversationOracle()
    result = chunk.extract_chunk(oracle, "ignored", source_records=((mid, json.dumps(payload)),))
    assert result.failed and result.failure_reason == "input_contract_failure"
    assert not oracle.requests


def test_context_recovery_keeps_global_call_budget_atomic():
    oracle = _ConversationOracle(force_recovery=True)
    result = chunk.extract_chunk(oracle, "ignored", source_records=(_record(1, "assistant", "Do you use PostgreSQL?"), _record(2, "user", "Yes.")), completion_call_limit=4)
    assert result.failed
    assert not result.triples
    assert result.completion_calls == len(oracle.requests) == 4
    assert any("calls:max_exceeded" in detail for detail in result.failure_details)


def test_recovery_added_context_cannot_exceed_hard_input_ceiling():
    first = _record(1, "assistant", "Do you use PostgreSQL?")
    for repeats in range(140, 190):
        records = (first, _record(2, "user", "Yes. " + "Routine background. " * repeats))
        initial = chunk._source_unit(records)
        split = chunk._split_unit(initial)
        assert split is not None
        if len(initial.text) <= chunk._MAX_LEAF_INPUT_CHARS < len(split[1].text):
            break
    else:
        pytest.fail("fixture must exercise context expansion across the hard ceiling")

    class CeilingOracle(_ConversationOracle):
        def complete(self, request):
            assert len(request.user.split('"""', 2)[1].strip()) <= chunk._MAX_LEAF_INPUT_CHARS
            response = super().complete(request)
            owned = [p for p in _payloads(request) if not p.get("source_context_only")]
            if len(owned) == 1 and owned[0]["content"].startswith("Yes.") and "OMISSION VERIFICATION PASS" not in request.system:
                assert any(p.get("source_context_only") for p in _payloads(request))
                return _response([{"subject": "user", "predicate": "uses", "object": "PostgreSQL", "polarity": 1, "source_message_id": 2}])
            return response

    oracle = CeilingOracle(force_recovery=True)
    result = chunk.extract_chunk(oracle, "ignored", source_records=records)
    assert not result.failed
    assert result.initial_prepartition_leaves == 1
    assert [(t.object, t.source_message_id) for t in result.triples] == [("PostgreSQL", 2)]
    assert any(p.get("source_content_start", 0) > 320 for request in oracle.requests for p in _payloads(request) if not p.get("source_context_only"))


def test_context_split_tree_stops_at_default_96_call_budget():
    class DenseOracle:
        def __init__(self):
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            owned = [p for p in _payloads(request) if not p.get("source_context_only")]
            if len(owned) > 1:
                return '{"triples": ['
            return _response()

    oracle = DenseOracle()
    records = tuple(_record(mid, "user", "Thanks.") for mid in range(1, 41))
    result = chunk.extract_chunk(oracle, "ignored", source_records=records)
    assert result.failed
    assert result.completion_calls == oracle.calls == 96
    assert not result.triples and not result.markers
    assert any("calls:max_exceeded" in detail for detail in result.failure_details)


def test_conversation_context_contract_change_invalidates_generation(monkeypatch):
    baseline = extraction_contract_identity()
    monkeypatch.setattr(chunk, "SOURCE_CONVERSATION_CONTEXT_VERSION", "older-context")
    assert extraction_contract_identity() != baseline
