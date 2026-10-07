"""Offline mechanics for the inactive atomic staged gate."""
import json
from dataclasses import replace

import pytest

from hymem.extraction import grounding_staged_gate_v1 as gate
from hymem.extraction import grounding_staged_v1 as staged
from hymem.extraction import grounding_classification_v4 as v4
from hymem.extraction.triples import Triple


def claim(predicate="uses", *, sid=7, **fields):
    return Triple("Mira", predicate, "CairnDB", 1, source_message_id=sid, **fields)


def record(content="Mira uses CairnDB.", **fields):
    return json.dumps({"content": content, "source_role": "user", **fields})


def assessment(state="not_established", *, sid=7, quote="Mira uses CairnDB", qualifiers=(), entries=None):
    if state != "supported":
        return {"state": state, "support": None}
    evidence = entries if entries is not None else [
        {"source_message_id": sid, "region": "owned", "quote": quote}]
    names = ("attribution_and_roles", "relation_and_polarity", *qualifiers)
    return {"state": state, "support": {"evidence": evidence,
            "checks": {name: {"state": "supported", "evidence_indices": list(range(len(evidence)))}
                       for name in names}}}


def originals(batch, *, force=None):
    rows = []
    for index, triple in enumerate(batch.triples):
        state = force[index] if force is not None else (
            "supported" if triple.predicate == "uses" else "not_established")
        names = tuple(name for name in ("value_text", "value_numeric", "value_unit", "temporal_scope")
                      if getattr(triple, name) is not None)
        rows.append({"index": index, "original": assessment(
            state, sid=triple.source_message_id, qualifiers=names)})
    return json.dumps({"schema": staged.ORIGINAL_SCHEMA,
                       "batch_sha256": batch.batch_sha256, "complete": True,
                       "originals": rows})


def alternatives(batch, *, positive="uses", override=None):
    rows = []
    for index in batch.negative_indices:
        triple = batch.classification_batch.triples[index]
        names = tuple(name for name in ("value_text", "value_numeric", "value_unit", "temporal_scope")
                      if getattr(triple, name) is not None)
        values = {predicate: assessment("supported" if predicate == positive else "not_established",
                                        sid=triple.source_message_id, qualifiers=names)
                  for predicate in v4.PREDICATE_ORDER if predicate != triple.predicate}
        if override is not None:
            override(values)
        rows.append({"index": index, "alternatives": values})
    return json.dumps({"schema": staged.ALTERNATIVES_SCHEMA,
                       "batch_sha256": batch.classification_batch.batch_sha256,
                       "original_response_sha256": batch.original_response_sha256,
                       "complete": True, "alternatives": rows})


def run(triples, invoke, *, sources=None, contexts=(), legacy_text=""):
    if sources is None and triples[0].source_message_id is not None:
        sources = ((7, record()),)
    return gate.ground_triples(triples, sources, contexts, legacy_text, invoke)


def test_source_fields_and_qualifiers_survive_and_only_predicate_changes():
    wrong = claim("prefers", value_text="production", value_numeric=3.0,
                  value_unit="nodes", temporal_scope="2026-09")
    sources = ((7, record(source_message_id=7, source_peer_id="mira",
                          source_created_at="2026-09-29T09:00:00Z")),)
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append((stage, recheck))
        if stage == "original":
            staged.validate_original_request(request, batch)
            assert batch.triples[0] == (replace(wrong, predicate="uses") if recheck else wrong)
            assert batch.sources[0].source_peer_id == "mira"
            assert batch.sources[0].source_created_at == "2026-09-29T09:00:00Z"
            wire = json.loads(request.user)["batch"]["candidates"][0]
            assert all(wire[name] == getattr(wrong, name) for name in
                       ("value_text", "value_numeric", "value_unit", "temporal_scope"))
            return originals(batch)
        assert not recheck and isinstance(batch, staged.AlternativesBatch)
        staged.validate_alternatives_request(request, batch)
        assert batch.negative_indices == (0,)
        return alternatives(batch)

    assert run([wrong], invoke, sources=sources) == [replace(wrong, predicate="uses")]
    assert calls == [("original", False), ("alternatives", False), ("original", True)]
    assert wrong.predicate == "prefers"


def test_late_batch_correction_rechecks_entire_list_without_alternatives():
    triples = [replace(claim(), subject=f"Mira {i}") for i in range(8)]
    triples.append(replace(claim("prefers"), subject="Mira last"))
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append((len(batch.triples) if stage == "original" else len(batch.classification_batch.triples),
                      stage, recheck))
        return originals(batch) if stage == "original" else alternatives(batch)

    result = run(triples, invoke)
    assert result == [*triples[:-1], replace(triples[-1], predicate="uses")]
    assert calls == [(8, "original", False), (1, "original", False),
                     (1, "alternatives", False), (8, "original", True), (1, "original", True)]


def test_mixed_ambiguous_negative_rejects_before_alternatives():
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append(stage)
        return originals(batch, force=("ambiguous", "not_established"))

    with pytest.raises(gate.GroundingGateError, match="^verdict:uncertain$"):
        run([claim(), claim("prefers")], invoke)
    assert calls == ["original"]


@pytest.mark.parametrize("mode,expected", [
    ("none", "verdict:unsupported"),
    ("ambiguous", "verdict:uncertain"),
    ("multiple", "verdict:uncertain"),
])
def test_alternative_selection_failures_are_atomic(mode, expected):
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append(stage)
        if stage == "original":
            return originals(batch)
        if mode == "none":
            return alternatives(batch, positive=None)
        def override(values):
            values["owns"] = assessment("ambiguous" if mode == "ambiguous" else "supported")
        return alternatives(batch, override=override)

    with pytest.raises(gate.GroundingGateError, match=f"^{expected}$"):
        run([claim("prefers")], invoke)
    assert calls == ["original", "alternatives"]


def test_recheck_negative_rejects_without_alternatives_or_second_correction():
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append((stage, recheck))
        if stage == "alternatives":
            return alternatives(batch)
        return originals(batch, force=("not_established",)) if recheck else originals(batch)

    with pytest.raises(gate.GroundingGateError, match="^verdict:unsupported$"):
        run([claim("prefers")], invoke)
    assert calls == [("original", False), ("alternatives", False), ("original", True)]


@pytest.mark.parametrize("polarity,code", [(1, "collision"), (-1, "conflict")])
def test_cross_batch_correction_guard(polarity, code):
    triples = [replace(claim("prefers"), subject=f"Mira {i}") for i in range(8)]
    triples.append(replace(claim(), subject="Mira 0", polarity=polarity))
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append(stage)
        return originals(batch) if stage == "original" else alternatives(batch)

    with pytest.raises(gate.GroundingGateError, match=f"^correction:{code}$"):
        run(triples, invoke)
    assert calls == ["original", "alternatives", "original"]


def test_callback_exception_identity_and_malformed_response():
    error = RuntimeError("transport budget exhausted")

    def raises(request, batch, stage, recheck):
        raise error

    with pytest.raises(RuntimeError) as caught:
        run([claim()], raises)
    assert caught.value is error
    with pytest.raises(gate.GroundingGateError, match="^contract:response_shape$"):
        run([claim()], lambda *args: "not json")


def test_later_batch_invalid_response_rejects_whole_result():
    triples = [replace(claim(), subject=f"Mira {i}") for i in range(9)]
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append((stage, recheck))
        return originals(batch) if len(calls) == 1 else "not json"

    with pytest.raises(gate.GroundingGateError, match="^contract:response_shape$"):
        run(triples, invoke)
    assert calls == [("original", False), ("original", False)]
    assert all(triple.predicate == "uses" for triple in triples)


def test_recheck_rejects_later_batch_atomically():
    triples = [replace(claim(), subject=f"Mira {i}") for i in range(8)]
    triples.append(replace(claim("prefers"), subject="Mira last"))
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append((stage, recheck))
        if stage == "alternatives":
            return alternatives(batch)
        if recheck and len(batch.triples) == 1:
            return originals(batch, force=("ambiguous",))
        return originals(batch)

    with pytest.raises(gate.GroundingGateError, match="^verdict:uncertain$"):
        run(triples, invoke)
    assert calls == [("original", False), ("original", False),
                     ("alternatives", False), ("original", True), ("original", True)]
    assert triples[-1].predicate == "prefers"


def test_legacy_and_nested_context_preserve_roles_prefix_parent():
    legacy = claim(sid=None)
    assert gate.ground_triples([legacy], None, (), "Mira uses CairnDB.",
        lambda request, batch, stage, recheck: originals(batch)) == [legacy]
    owned = "Mira uses CairnDB. Later text."
    parent = "| Mira | CairnDB |"
    sources = ((7, record(owned, source_message_id=7, source_peer_id="mira",
                          source_fragment_context={"content": "| Person | Database |",
                                                   "applies_through_source_content_end": len(owned)})),)
    contexts = ((6, record(parent + " Later", source_message_id=6, source_role="assistant",
                           source_peer_id="helper", context_for_source_message_id=7,
                           source_content_start=0, applies_through_source_content_end=18,
                           source_fragment_context={"content": "| Person | Database |",
                                                    "prelude_content": "Table prelude",
                                                    "applies_through_source_content_end": len(parent)})),)

    def invoke(request, batch, stage, recheck):
        source = batch.sources[0]
        assert source.source_peer_id == "mira"
        regions = {context.region: context for context in source.contexts}
        assert regions["conversation_0"].source_role == "assistant"
        assert regions["conversation_0"].source_peer_id == "helper"
        assert regions["conversation_0"].owned_prefix_chars == 18
        assert regions["conversation_0_header"].applies_to_region == "conversation_0"
        assert regions["conversation_0_header"].applies_to_prefix_chars == len(parent)
        assert regions["conversation_0_prelude"].content == "Table prelude"
        return originals(batch)

    assert run([claim()], invoke, sources=sources, contexts=contexts) == [claim()]


def test_alternative_global_evidence_limit_still_applies():
    content = "Mira uses CairnDB. " + " ".join(f"token{i}" for i in range(10))
    calls = []

    def invoke(request, batch, stage, recheck):
        calls.append(stage)
        if stage == "original":
            return originals(batch)
        def override(values):
            for predicate, positions in (("uses", range(5)), ("owns", range(5, 10))):
                values[predicate] = assessment("supported", entries=[
                    {"source_message_id": 7, "region": "owned", "quote": f"token{i}"}
                    for i in positions])
        return alternatives(batch, positive=None, override=override)

    with pytest.raises(gate.GroundingGateError, match="^contract:evidence_global_bounds$"):
        run([claim("prefers")], invoke, sources=((7, record(content)),))
    assert calls == ["original", "alternatives"]


def test_support_integrity_guard_detects_replacement(monkeypatch):
    monkeypatch.setattr(staged, "parse_staged_responses", lambda *args, **kwargs: None)
    assert not gate.grounding_gate_support_integrity()
    with pytest.raises(gate.GroundingGateError, match="^support:integrity$"):
        run([claim()], lambda *args: None)
