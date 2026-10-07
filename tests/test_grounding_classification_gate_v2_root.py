"""Root checks of inactive v2 gate; fake verdicts test mechanics, not accuracy."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from hymem.extraction import grounding_classification_v2 as g
from hymem.extraction import grounding_classification_gate_v2 as gate


def inputs(count=1, predicate='uses'):
    records = tuple((i + 1, json.dumps(dict(source_message_id=i + 1, content=f'I use tool{i}.',
        source_role='user', source_peer_id='Mira', source_created_at='2026-09-29'))) for i in range(count))
    claims = [g.Triple('Mira', predicate, f'tool{i}', 1, source_message_id=i + 1) for i in range(count)]
    return claims, records


def reply(batch, predicates):
    rows = []
    for i, (t, p) in enumerate(zip(batch.triples, predicates, strict=True)):
        states = ['not_established'] * 22
        groups = []
        if p:
            states[g.PREDICATE_ORDER.index(p)] = 'supported'
            source = next(s for s in batch.sources if s.source_message_id == t.source_message_id)
            groups = [dict(predicates=[p], evidence=[dict(source_message_id=t.source_message_id,
                region='owned', quote=source.content)])]
        rows.append(dict(index=i, states=states, support_groups=groups))
    return json.dumps(dict(schema=g.GROUNDING_CONTRACT_VERSION, batch_sha256=batch.batch_sha256,
                           complete=True, classifications=rows))


def test_old_gate_and_new_accepted_contract_unchanged():
    root = Path(g.__file__).parent
    assert hashlib.sha256((root / 'grounding_classification_gate_v1.py').read_bytes()).hexdigest() == (
        '0c676c9e10f9dfb9fa005be4ec7b75c799d4e69e37d61ae0953a52e86ed59081')
    assert hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest() == (
        '2520e4825df9e4d6f403a301451bb9701e052d9a331b5e31a9e106cbb69ad3c9')


def test_nine_claims_recheck_full_list_once_with_qualifiers_unchanged():
    original, records = inputs(9)
    original[8] = replace(original[8], predicate='prefers', value_numeric=2, value_unit='hours',
                          value_text='daily', temporal_scope='during trial')
    before = tuple(original)
    calls = []
    def invoke(request, batch, recheck):
        g.validate_request(request, batch)
        calls.append((tuple(batch.triples), recheck))
        assert all(type(s) is g.GroundingSource for s in batch.sources)
        return reply(batch, ['uses'] * len(batch.triples))
    result = gate.ground_triples(original, records, (), '', invoke)
    assert [(len(b), r) for b, r in calls] == [(8, False), (1, False), (8, True), (1, True)]
    assert tuple(original) == before
    assert result == list(before[:8]) + [replace(before[8], predicate='uses')]
    assert calls[2][0] + calls[3][0] == tuple(result)


@pytest.mark.parametrize('failure_call', [2, 3, 4])
def test_late_failure_cannot_return_partial_corrections(failure_call):
    original, records = inputs(9, 'prefers')
    before = tuple(original)
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        return reply(batch, [None if len(calls) == failure_call else 'uses'] * len(batch.triples))
    with pytest.raises(gate.GroundingGateError, match='verdict:unsupported'):
        gate.ground_triples(original, records, (), '', invoke)
    assert len(calls) == failure_call and tuple(original) == before


@pytest.mark.parametrize('polarity', [1, -1])
def test_collisions_and_conflicts_reject_before_recheck(polarity):
    original, records = inputs()
    original.append(replace(original[0], predicate='prefers', polarity=polarity))
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        return reply(batch, ['uses'] * len(batch.triples))
    with pytest.raises(gate.GroundingGateError, match='correction:(collision|conflict)'):
        gate.ground_triples(original, records, (), '', invoke)
    assert calls == [False]


def test_budget_exception_identity_and_no_retry():
    class LimitReached(RuntimeError): pass
    problem = LimitReached('fixed-budget-sentinel')
    original, records = inputs(9, 'prefers')
    before = tuple(original)
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        if len(calls) == 2:
            raise problem
        return reply(batch, ['uses'] * len(batch.triples))
    with pytest.raises(LimitReached) as caught:
        gate.ground_triples(original, records, (), '', invoke)
    assert caught.value is problem and calls == [False, False]
    assert tuple(original) == before


def test_collision_spanning_batches_is_detected_globally():
    original, records = inputs(9)
    original[8] = replace(original[0], predicate='prefers')
    calls = []
    def invoke(request, batch, recheck):
        calls.append((len(batch.triples), recheck))
        return reply(batch, ['uses'] * len(batch.triples))
    with pytest.raises(gate.GroundingGateError, match='correction:collision'):
        gate.ground_triples(original, records, (), '', invoke)
    assert calls == [(8, False), (1, False)]


def test_second_correction_is_never_followed_by_third_pass():
    original, records = inputs(predicate='prefers')
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        return reply(batch, ['rejects' if recheck else 'uses'])
    with pytest.raises(gate.GroundingGateError, match='correction'):
        gate.ground_triples(original, records, (), '', invoke)
    assert calls == [False, True]


@pytest.mark.parametrize('fault', ['ambiguous', 'missing_group', 'v1_wire'])
def test_rejected_response_never_rerolled(fault):
    original, records = inputs()
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        raw = json.loads(reply(batch, ['uses']))
        if fault == 'ambiguous':
            raw['classifications'][0].update(states=['ambiguous'] * 22, support_groups=[])
        elif fault == 'missing_group':
            raw['classifications'][0]['support_groups'] = []
        else:
            raw['schema'] = 'source-grounding-classification-v1'
        return json.dumps(raw)
    with pytest.raises(gate.GroundingGateError):
        gate.ground_triples(original, records, (), '', invoke)
    assert calls == [False]


def test_legacy_null_provenance_and_no_call_for_empty_input():
    original, _ = inputs()
    original = [replace(original[0], source_message_id=None)]
    def invoke(request, batch, recheck):
        assert batch.sources[0].source_message_id is None
        return reply(batch, ['uses'])
    assert gate.ground_triples(original, None, (), 'I use tool0.', invoke) == original
    def forbidden(*args): raise AssertionError('unexpected invocation')
    assert gate.ground_triples([], None, (), '', forbidden) == []


def test_tampered_import_guard_rejects_before_callback(monkeypatch):
    original, records = inputs()
    monkeypatch.setattr(gate, 'parse_grounding_response', lambda *args: None)
    def forbidden(*args): raise AssertionError('unexpected invocation')
    with pytest.raises(gate.GroundingGateError, match='support:integrity'):
        gate.ground_triples(original, records, (), '', forbidden)
