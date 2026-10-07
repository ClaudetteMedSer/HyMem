"""Offline gate mechanics using synthetic v3 assessments, not semantic judgments."""
from dataclasses import replace
import json

import pytest

from hymem.extraction import grounding_classification_gate_v3 as gate
from hymem.extraction import grounding_classification_v3 as contract
from hymem.extraction.triples import Triple


def _record(content: str, **extra: object) -> str:
    return json.dumps({"content": content, "source_role": "user", **extra})


def _claim(predicate: str = "uses", *, sid: int | None = 7, **extra: object) -> Triple:
    return Triple("Mira", predicate, "CairnDB", 1, source_message_id=sid, **extra)


def _assessment(state: str, triple: Triple, source: contract.GroundingSource) -> dict:
    if state != "supported":
        return {"state": state, "support": None}
    qualifiers = ("value_text", "value_numeric", "value_unit", "temporal_scope")
    names = ("attribution_and_roles", "relation_and_polarity",
             *(name for name in qualifiers if getattr(triple, name) is not None))
    quote = source.content[:min(20, len(source.content))]
    return {"state": "supported", "support": {
        "evidence": [{"source_message_id": triple.source_message_id,
                      "region": "owned", "quote": quote}],
        "checks": {name: {"state": "supported", "evidence_indices": [0]}
                   for name in names}}}


def _answer(batch: contract.ClassificationBatch, *, positive: str | None = "uses",
            ambiguous: bool = False, malformed: bool = False) -> str:
    if malformed:
        return "private-synthetic-secret"
    rows = []
    for index, triple in enumerate(batch.triples):
        source = next(source for source in batch.sources
                      if source.source_message_id == triple.source_message_id)
        if positive == triple.predicate and not ambiguous:
            original = _assessment("supported", triple, source)
            alternatives = None
        elif ambiguous:
            original = _assessment("ambiguous", triple, source)
            alternatives = None
        else:
            original = _assessment("not_established", triple, source)
            alternatives = {
                predicate: _assessment("supported" if predicate == positive else "not_established",
                                       triple, source)
                for predicate in contract.PREDICATE_ORDER if predicate != triple.predicate
            }
        rows.append({"index": index, "original": original, "alternatives": alternatives})
    return json.dumps({"schema": contract.GROUNDING_CONTRACT_VERSION,
                       "batch_sha256": batch.batch_sha256, "complete": True,
                       "classifications": rows})


def _run(triples, callback, *, content="Mira uses CairnDB.", records=None,
         contexts=(), legacy_text=""):
    if records is None and triples[0].source_message_id is not None:
        records = ((7, _record(content)),)
    return gate.ground_triples(triples, records, contexts, legacy_text, callback)


def test_original_wire_preserves_qualifiers_source_and_request_identity():
    claim = _claim(value_text="production", value_numeric=3.0, value_unit="nodes",
                   temporal_scope="2026-09")
    records = ((7, _record("Mira uses CairnDB.", source_peer_id="mira",
                           source_created_at="2026-09-29T00:00:00Z")),)
    calls = []

    def invoke(request, batch, recheck):
        contract.validate_request(request, batch)
        calls.append(recheck)
        assert batch.triples == (claim,)
        assert batch.sources[0].source_role == "user"
        assert batch.sources[0].source_peer_id == "mira"
        assert batch.sources[0].source_created_at == "2026-09-29T00:00:00Z"
        wire = json.loads(request.user)
        candidate = wire["batch"]["candidates"][0]
        assert all(candidate[name] == getattr(claim, name) for name in
                   ("predicate", "value_text", "value_numeric", "value_unit", "temporal_scope"))
        return _answer(batch)

    assert _run([claim], invoke, records=records) == [claim]
    assert calls == [False]
    assert records[0][1] == _record("Mira uses CairnDB.", source_peer_id="mira",
                                    source_created_at="2026-09-29T00:00:00Z")


def test_alternative_rechecks_entire_corrected_list_once():
    wrong = _claim("prefers", value_text="production", value_numeric=3.0,
                   value_unit="nodes", temporal_scope="2026-09")
    already = _claim("uses", sid=8)
    records = ((7, _record("Mira uses CairnDB.")),
               (8, _record("Mira uses CairnDB too.")))
    calls = []

    def invoke(request, batch, recheck):
        contract.validate_request(request, batch)
        calls.append((batch.triples, recheck))
        return _answer(batch)

    result = gate.ground_triples([wrong, already], records, (), "", invoke)
    corrected = replace(wrong, predicate="uses")
    assert result == [corrected, already]
    assert calls == [((wrong, already), False), ((corrected, already), True)]
    assert wrong.predicate == "prefers" and result[0].source_message_id == wrong.source_message_id


def test_late_batch_alternative_rechecks_all_batches_without_extra_calls():
    triples = [replace(_claim(), subject=f"Mira {index}") for index in range(8)]
    triples.append(replace(_claim("prefers"), subject="Mira last"))
    calls = []

    def invoke(request, batch, recheck):
        contract.validate_request(request, batch)
        calls.append((len(batch.triples), recheck))
        return _answer(batch)

    result = _run(triples, invoke)
    assert result[-1].predicate == "uses"
    assert calls == [(8, False), (1, False), (8, True), (1, True)]
    assert triples[-1].predicate == "prefers"


@pytest.mark.parametrize("positive,ambiguous,expected", [
    (None, False, "unsupported"), (None, True, "uncertain"),
])
def test_later_batch_rejection_is_atomic(positive, ambiguous, expected):
    triples = [replace(_claim(), subject=f"Mira {index}") for index in range(9)]
    calls = []

    def invoke(request, batch, recheck):
        calls.append(recheck)
        return _answer(batch, positive=positive, ambiguous=ambiguous) if len(calls) == 2 else _answer(batch)

    with pytest.raises(gate.GroundingGateError, match=f"^verdict:{expected}$"):
        _run(triples, invoke)
    assert calls == [False, False]
    assert all(t.predicate == "uses" for t in triples)


def test_second_pass_alternative_rejected_without_third_call():
    calls = []

    def invoke(request, batch, recheck):
        calls.append(recheck)
        return _answer(batch, positive="rejects" if recheck else "uses")

    with pytest.raises(gate.GroundingGateError, match="^contract:verdict_correction$"):
        _run([_claim("prefers")], invoke)
    assert calls == [False, True]


@pytest.mark.parametrize("other_polarity,expected", [
    (1, "collision"), (-1, "conflict"),
])
def test_cross_batch_alternative_collision_and_conflict(other_polarity, expected):
    first = _claim("prefers")
    triples = [replace(first, subject=f"Mira {index}") for index in range(8)]
    triples.append(replace(first, subject="Mira 0", predicate="uses",
                           polarity=other_polarity))
    calls = []

    def invoke(request, batch, recheck):
        calls.append(recheck)
        return _answer(batch)

    with pytest.raises(gate.GroundingGateError, match=f"^correction:{expected}$"):
        _run(triples, invoke)
    assert calls == [False, False]


def test_legacy_null_source_and_context_reconstruction():
    legacy = _claim(sid=None)

    def legacy_invoke(request, batch, recheck):
        assert batch.sources[0].source_message_id is None
        assert batch.sources[0].content == "Mira uses CairnDB."
        return _answer(batch)

    assert _run([legacy], legacy_invoke, records=None,
                legacy_text="Mira uses CairnDB.") == [legacy]
    owned = "Mira uses CairnDB. Later text."
    source = ((7, _record(owned, source_content_start=10,
                          source_fragment_context={"content": "Table header",
                              "applies_through_source_content_end": 28})),)
    contexts = ((6, _record("Earlier claim", context_for_source_message_id=7,
                            applies_through_source_content_end=28,
                            source_message_id=6)),)

    def context_invoke(request, batch, recheck):
        contract.validate_request(request, batch)
        regions = {item.region: item for item in batch.sources[0].contexts}
        assert regions["header"].content == "Table header"
        assert regions["header"].owned_prefix_chars == 18
        assert regions["conversation_0"].content == "Earlier claim"
        assert regions["conversation_0"].source_message_id == 6
        return _answer(batch)

    assert gate.ground_triples([_claim()], source, contexts, "", context_invoke) == [_claim()]


def test_nested_parent_context_fields_survive_reconstruction():
    owned = "Mira uses CairnDB and favors it."
    row = "| Mira | CairnDB |"
    source = ((7, _record(owned, source_message_id=7, source_peer_id="mira",
                          source_created_at="2026-09-29T09:00:00Z",
                          source_fragment_context={"content": "| Person | Database |",
                              "applies_through_source_content_end": len(owned)})),)
    contexts = ((6, _record(row + "\nLater unrelated text.", source_message_id=6,
                            source_role="assistant", source_peer_id="helper",
                            source_created_at="2026-09-29T08:00:00Z",
                            context_for_source_message_id=7,
                            source_content_start=0,
                            applies_through_source_content_end=len(owned),
                            source_fragment_context={"content": "| Person | Database |",
                                "prelude_content": "Table prelude",
                                "applies_through_source_content_end": len(row)})),)

    def invoke(request, batch, recheck):
        contract.validate_request(request, batch)
        assert batch.sources[0].source_role == "user"
        assert batch.sources[0].source_peer_id == "mira"
        assert batch.sources[0].source_created_at == "2026-09-29T09:00:00Z"
        regions = {item.region: item for item in batch.sources[0].contexts}
        assert regions["conversation_0_header"].applies_to_region == "conversation_0"
        assert regions["conversation_0_header"].applies_to_prefix_chars == len(row)
        assert regions["conversation_0_prelude"].applies_to_region == "conversation_0"
        assert regions["conversation_0_prelude"].content == "Table prelude"
        assert regions["conversation_0"].source_message_id == 6
        assert regions["conversation_0"].source_role == "assistant"
        assert regions["conversation_0"].source_peer_id == "helper"
        assert regions["conversation_0"].source_created_at == "2026-09-29T08:00:00Z"
        assert regions["conversation_0_header"].source_message_id == 6
        assert regions["conversation_0_prelude"].source_message_id == 6
        payload = json.loads(_answer(batch))
        support = payload["classifications"][0]["original"]["support"]
        support["evidence"].extend([
            {"source_message_id": 7, "region": "conversation_0", "quote": row},
            {"source_message_id": 7, "region": "conversation_0_header",
             "quote": "| Person | Database |"},
            {"source_message_id": 7, "region": "conversation_0_prelude",
             "quote": "Table prelude"},
        ])
        for check in support["checks"].values():
            check["evidence_indices"] = [0, 1, 2, 3]
        return json.dumps(payload)

    assert gate.ground_triples([_claim()], source, contexts, "", invoke) == [_claim()]


@pytest.mark.parametrize("fault,code", [
    ("owned_prefix", "contract:evidence_context_scope"),
    ("missing_parent", "contract:evidence_parent_required"),
])
def test_reconstructed_context_scope_and_parent_guards(fault, code):
    owned = "Mira uses CairnDB. Later unrelated text."
    parent = "| Mira | CairnDB | Later unrelated text."
    source = ((7, _record(owned, source_fragment_context={
        "content": "| Person | Database |",
        "applies_through_source_content_end": 18})),)
    contexts = ((6, _record(parent, source_message_id=6,
                            context_for_source_message_id=7,
                            source_content_start=0,
                            applies_through_source_content_end=18,
                            source_fragment_context={
                                "content": "| Person | Database |",
                                "applies_through_source_content_end": 19})),)
    calls = []

    def invoke(request, batch, recheck):
        calls.append(recheck)
        payload = json.loads(_answer(batch))
        support = payload["classifications"][0]["original"]["support"]
        if fault == "owned_prefix":
            support["evidence"][0]["quote"] = "Later unrelated text."
            support["evidence"].append({"source_message_id": 7,
                                        "region": "header",
                                        "quote": "| Person | Database |"})
            for check in support["checks"].values():
                check["evidence_indices"] = [0, 1]
        else:
            support["evidence"][0]["quote"] = "Mira uses CairnDB"
            support["evidence"].append({"source_message_id": 7,
                                        "region": "conversation_0_header",
                                        "quote": "| Person | Database |"})
            for check in support["checks"].values():
                check["evidence_indices"] = [0, 1]
        return json.dumps(payload)

    with pytest.raises(gate.GroundingGateError, match=f"^{code}$"):
        gate.ground_triples([_claim()], source, contexts, "", invoke)
    assert calls == [False]


def test_total_source_budget_splits_distinct_sources():
    first = "Mira uses CairnDB. " + "a" * 9000
    second = "Nora uses CairnDB. " + "b" * 9000
    claims = [_claim(), replace(_claim(), subject="Nora", source_message_id=8)]
    records = ((7, _record(first)), (8, _record(second)))
    sizes = []

    def invoke(request, batch, recheck):
        contract.validate_request(request, batch)
        sizes.append((len(batch.triples), len(batch.sources)))
        return _answer(batch)

    assert gate.ground_triples(claims, records, (), "", invoke) == claims
    assert sizes == [(1, 1), (1, 1)]


@pytest.mark.parametrize("fault,code", [
    ("old", "contract:response_schema"),
    ("checks", "contract:support_checks"),
    ("missing_alternative", "contract:classification_alternatives"),
])
def test_old_or_malformed_v3_wire_rejected_without_retry(fault, code):
    calls = []

    def invoke(request, batch, recheck):
        calls.append(recheck)
        payload = json.loads(_answer(batch, positive="prefers"))
        item = payload["classifications"][0]
        if fault == "old":
            payload["schema"] = "source-grounding-classification-v2"
        elif fault == "checks":
            item["alternatives"]["prefers"]["support"]["checks"].pop("attribution_and_roles")
        else:
            item["alternatives"].pop("owns")
        return json.dumps(payload)

    with pytest.raises(gate.GroundingGateError, match=f"^{code}$"):
        _run([_claim()], invoke)
    assert calls == [False]


def test_malformed_response_has_finite_private_code():
    with pytest.raises(gate.GroundingGateError, match="^contract:response_shape$") as error:
        _run([_claim()], lambda request, batch, recheck: _answer(batch, malformed=True))
    assert "private-synthetic-secret" not in str(error.value)


def test_callback_budget_failure_propagates_by_identity_without_retry():
    fault = gate.GroundingGateError("calls:max_exceeded")
    calls = []

    def invoke(request, batch, recheck):
        contract.validate_request(request, batch)
        calls.append(recheck)
        raise fault

    with pytest.raises(gate.GroundingGateError) as error:
        _run([_claim()], invoke)
    assert error.value is fault and calls == [False]


def test_recheck_callback_failure_propagates_by_identity_without_retry():
    fault = RuntimeError("typed cleanup fault")
    calls = []

    def invoke(request, batch, recheck):
        contract.validate_request(request, batch)
        calls.append(recheck)
        if recheck:
            raise fault
        return _answer(batch)

    with pytest.raises(RuntimeError) as error:
        _run([_claim("prefers")], invoke)
    assert error.value is fault and calls == [False, True]


def test_integrity_failure_prevents_callback(monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "parse_grounding_response", lambda *args, **kwargs: None)
    with pytest.raises(gate.GroundingGateError, match="^support:integrity$"):
        _run([_claim()], lambda *args: calls.append(1))
    assert calls == []


def test_empty_input_has_no_callback():
    assert gate.ground_triples([], None, (), "", lambda *args: 1) == []
