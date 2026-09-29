"""Offline mechanics for the inactive classification gate; no model judgments."""
from dataclasses import replace
import json

import pytest

from hymem.extraction import grounding_classification_gate_v1 as gate
from hymem.extraction import grounding_classification_v1 as c
from hymem.extraction.triples import Triple


def _record(content: str, **extra: object) -> str:
    return json.dumps({"content": content, "source_role": "user", **extra})


def _claim(predicate: str = "uses", *, sid: int | None = 7, **extra: object) -> Triple:
    return Triple("Mira", predicate, "CairnDB", 1, source_message_id=sid, **extra)


def _answer(batch: c.ClassificationBatch, *, positive: str | None = "uses",
            uncertain: bool = False, malformed: bool = False) -> str:
    if malformed:
        return "provider returned private-synthetic-secret"
    items = []
    for index, triple in enumerate(batch.triples):
        states = ["n"] * len(c.PREDICATE_ORDER)
        citations = [[] for _ in states]
        pool = []
        if positive is not None:
            position = c.PREDICATE_ORDER.index(positive)
            states[position] = "e"
            source = next(source for source in batch.sources
                          if source.source_message_id == triple.source_message_id)
            pool = [{"source_message_id": triple.source_message_id,
                     "region": "owned", "quote": source.content[:min(20, len(source.content))]}]
            citations[position] = [0]
        if uncertain:
            states[c.PREDICATE_ORDER.index(triple.predicate)] = "u"
        items.append({"index": index, "states": states, "evidence_pool": pool,
                      "citations": citations})
    return json.dumps({"schema": c.GROUNDING_CONTRACT_VERSION,
                       "batch_sha256": batch.batch_sha256, "complete": True,
                       "classifications": items})


def _run(triples, callback, *, content="Mira uses CairnDB.", records=None,
         contexts=(), legacy_text=""):
    if records is None and triples[0].source_message_id is not None:
        records = ((7, _record(content)),)
    return gate.ground_triples(triples, records, contexts, legacy_text, callback)


def test_supported_preserves_full_claim_and_source_input():
    claim = _claim(value_text="production", value_numeric=3.0, value_unit="nodes",
                   temporal_scope="2026-09")
    originals = [claim]
    records = ((7, _record("Mira uses CairnDB.", source_peer_id="mira")),)
    seen = []

    def invoke(request, batch, recheck):
        c.validate_request(request, batch)
        seen.append((batch.triples, batch.sources, recheck))
        assert batch.triples == (claim,)
        assert batch.sources[0].source_peer_id == "mira"
        return _answer(batch)

    assert _run(originals, invoke, records=records) == [claim]
    assert originals == [claim] and len(seen) == 1 and seen[0][2] is False
    assert records == ((7, _record("Mira uses CairnDB.", source_peer_id="mira")),)


def test_correction_rechecks_full_list_once_with_all_fields_unchanged(monkeypatch):
    wrong = _claim("prefers", value_text="production", value_numeric=3.0,
                   value_unit="nodes", temporal_scope="2026-09")
    already = _claim("uses", sid=8)
    records = ((7, _record("Mira uses CairnDB.")),
               (8, _record("Mira uses CairnDB too.")))
    calls = []
    checked = []
    original_validate = gate.validate_request

    def validate(request, batch):
        original_validate(request, batch)
        checked.append(batch.batch_sha256)

    monkeypatch.setattr(gate, "validate_request", validate)
    # The integrity pin is deliberately checked before any invocation.
    guard = gate._IMPORTED_GUARD
    monkeypatch.setattr(gate, "_IMPORTED_GUARD", tuple(
        validate if item is original_validate else item for item in guard))

    def invoke(request, batch, recheck):
        c.validate_request(request, batch)
        calls.append((batch.triples, recheck))
        return _answer(batch)

    result = gate.ground_triples([wrong, already], records, (), "", invoke)
    assert result == [replace(wrong, predicate="uses"), already]
    assert calls == [((wrong, already), False),
                     ((replace(wrong, predicate="uses"), already), True)]
    assert len(checked) == 2
    assert result[0].source_message_id == wrong.source_message_id
    assert [wrong, already][0].predicate == "prefers"


def test_multi_batch_rechecks_every_claim_even_when_only_late_one_changes():
    triples = [replace(_claim("uses"), subject=f"Mira {index}")
               for index in range(8)] + [_claim("prefers")]
    calls = []

    def invoke(request, batch, recheck):
        c.validate_request(request, batch)
        calls.append((len(batch.triples), recheck))
        return _answer(batch)

    result = _run(triples, invoke)
    assert len(result) == 9 and result[-1].predicate == "uses"
    assert calls == [(8, False), (1, False), (8, True), (1, True)]
    assert triples[-1].predicate == "prefers"


def test_later_batch_failure_rejects_whole_result_without_mutation():
    triples = [_claim("uses") for _ in range(8)] + [_claim("uses")]
    calls = []

    def invoke(request, batch, recheck):
        calls.append(1)
        return _answer(batch, positive=None) if len(calls) == 2 else _answer(batch)

    with pytest.raises(gate.GroundingGateError, match="^verdict:unsupported$"):
        _run(triples, invoke)
    assert len(calls) == 2 and all(t.predicate == "uses" for t in triples)


def test_recheck_correction_rejects_without_third_call():
    wrong = _claim("prefers")
    calls = []

    def invoke(request, batch, recheck):
        calls.append(recheck)
        if recheck:
            return _answer(batch, positive="prefers")
        return _answer(batch)

    with pytest.raises(gate.GroundingGateError, match="^contract:verdict_correction$"):
        _run([wrong], invoke)
    assert calls == [False, True]


@pytest.mark.parametrize("second", ["uses", "prefers"])
def test_correction_collision_and_conflict(second):
    first = _claim("prefers")
    other = replace(_claim(second), polarity=-1 if second == "prefers" else 1)
    # Both fixed claims classify as uses; canonical identity includes source,
    # subject and object but polarity decides collision versus conflict.
    def invoke(request, batch, recheck):
        return _answer(batch)

    expected = "conflict" if other.polarity == -1 else "collision"
    with pytest.raises(gate.GroundingGateError, match=f"^correction:{expected}$"):
        _run([first, other], invoke)


def test_legacy_source_and_context_reconstruction():
    legacy = _claim(sid=None)
    assert _run([legacy], lambda request, batch, recheck: _answer(batch),
                records=None, legacy_text="Mira uses CairnDB.") == [legacy]
    owned = "Mira uses CairnDB. Later text."
    source = ((7, _record(owned, source_content_start=10,
                          source_fragment_context={"content": "Table header",
                              "applies_through_source_content_end": 28})),)
    contexts = ((6, _record("Earlier claim", context_for_source_message_id=7,
                            applies_through_source_content_end=28,
                            source_message_id=6)),)

    def invoke(request, batch, recheck):
        c.validate_request(request, batch)
        regions = {item.region: item for item in batch.sources[0].contexts}
        assert regions["header"].content == "Table header"
        assert regions["header"].owned_prefix_chars == 18
        assert regions["conversation_0"].content == "Earlier claim"
        assert regions["conversation_0"].source_message_id == 6
        return _answer(batch)

    assert gate.ground_triples([_claim()], source, contexts, "", invoke) == [_claim()]


def test_canary_shaped_multi_claim_context_and_nested_parent_are_preserved():
    owned = "Mira uses CairnDB and favors it."
    parent = "| Mira | CairnDB |\nLater suggestion is unrelated."
    row = "| Mira | CairnDB |"
    source = ((7, _record(owned, source_message_id=7,
                          source_fragment_context={
                              "content": "| Person | Database |",
                              "applies_through_source_content_end": len(owned)})),)
    contexts = ((6, _record(parent, source_message_id=6,
                            context_for_source_message_id=7,
                            source_content_start=0,
                            applies_through_source_content_end=len(owned),
                            source_fragment_context={
                                "content": "| Person | Database |",
                                "applies_through_source_content_end": len(row)})),)
    claims = [_claim("uses"), replace(_claim("uses"), object="it")]
    seen = []

    def invoke(request, batch, recheck):
        c.validate_request(request, batch)
        seen.append(batch.sources)
        regions = {context.region: context for context in batch.sources[0].contexts}
        assert regions["header"].content == "| Person | Database |"
        assert regions["conversation_0"].source_message_id == 6
        assert regions["conversation_0_header"].applies_to_region == "conversation_0"
        assert regions["conversation_0_header"].applies_to_prefix_chars == len(row)
        return _answer(batch)

    assert gate.ground_triples(claims, source, contexts, "", invoke) == claims
    assert len(seen) == 1 and len(seen[0][0].contexts) == 3


def test_16384_total_source_bound_splits_distinct_sources():
    first = "Mira uses CairnDB. " + "a" * 9000
    second = "Nora uses CairnDB. " + "b" * 9000
    claims = [_claim(), replace(_claim(), subject="Nora", source_message_id=8)]
    records = ((7, _record(first)), (8, _record(second)))
    sizes = []

    def invoke(request, batch, recheck):
        c.validate_request(request, batch)
        sizes.append((len(batch.triples), len(batch.sources)))
        return _answer(batch)

    assert gate.ground_triples(claims, records, (), "", invoke) == claims
    assert sizes == [(1, 1), (1, 1)]


def test_malformed_provider_response_has_finite_private_code():
    with pytest.raises(gate.GroundingGateError, match="^contract:response_shape$") as error:
        _run([_claim()], lambda request, batch, recheck: _answer(batch, malformed=True))
    assert "private-synthetic-secret" not in str(error.value)



def test_typed_callback_failure_propagates_by_identity_without_retry():
    fault = gate.GroundingGateError("calls:max_exceeded")
    calls = []

    def fail(request, batch, recheck):
        c.validate_request(request, batch)
        calls.append(recheck)
        raise fault

    with pytest.raises(gate.GroundingGateError) as error:
        _run([_claim()], fail)
    assert error.value is fault
    assert calls == [False]


def test_recheck_callback_failure_propagates_without_extra_invocation():
    fault = RuntimeError("typed cleanup fault")
    calls = []

    def fail_on_recheck(request, batch, recheck):
        c.validate_request(request, batch)
        calls.append(recheck)
        if recheck:
            raise fault
        return _answer(batch)

    with pytest.raises(RuntimeError) as error:
        _run([_claim("prefers")], fail_on_recheck)
    assert error.value is fault
    assert calls == [False, True]


def test_integrity_failure_prevents_callback(monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "parse_grounding_response", lambda *args, **kwargs: None)
    with pytest.raises(gate.GroundingGateError, match="^support:integrity$"):
        _run([_claim()], lambda *args: calls.append(1))
    assert calls == []
