"""Root atomicity/source controls; responses here are explicitly synthetic."""
from dataclasses import replace
import json
from pathlib import Path
import runpy

import pytest

from hymem.extraction import grounding_classification_v1 as g
from hymem.extraction import grounding_classification_gate_v1 as gate


def records(count=1):
    return tuple((i + 1, json.dumps({'source_message_id': i + 1,
        'source_role': 'user', 'source_peer_id': 'Mira', 'content': f'I use tool{i}.',
        'source_created_at': '2026-09-29T00:00:00Z'})) for i in range(count))


def triples(count=1, predicate='uses'):
    return [g.Triple('Mira', predicate, f'tool{i}', 1, source_message_id=i+1) for i in range(count)]


def reply(batch, predicates):
    rows = []
    for i, (triple, pred) in enumerate(zip(batch.triples, predicates, strict=True)):
        states = ['n'] * 22
        cites = [[] for _ in states]
        pool = []
        if pred is not None:
            states[g.PREDICATE_ORDER.index(pred)] = 'e'
            cites[g.PREDICATE_ORDER.index(pred)] = [0]
            source = next(s for s in batch.sources if s.source_message_id == triple.source_message_id)
            pool = [dict(source_message_id=triple.source_message_id, region='owned', quote=source.content)]
        rows.append(dict(index=i, states=states, citations=cites, evidence_pool=pool))
    return json.dumps(dict(schema=g.GROUNDING_CONTRACT_VERSION, batch_sha256=batch.batch_sha256,
                           complete=True, classifications=rows))


def test_whole_corrected_list_rechecked_exactly_once_and_input_immutable():
    original = triples(2)
    original[0] = replace(original[0], predicate='prefers', value_text='daily',
                          temporal_scope='during the trial')
    prior = tuple(original)
    calls = []
    def invoke(request, batch, recheck):
        g.validate_request(request, batch)
        calls.append((batch.triples, recheck))
        return reply(batch, ['uses']*len(batch.triples))
    result = gate.ground_triples(original, records(2), (), '', invoke)
    assert [r for _, r in calls] == [False, True]
    assert len(calls[1][0]) == 2
    assert tuple(original) == prior
    assert result == [replace(prior[0], predicate='uses'), prior[1]]
    assert calls[1][0] == tuple(result)
    assert result[0].value_text == 'daily' and result[0].temporal_scope == 'during the trial'


@pytest.mark.parametrize('where', ['initial', 'recheck'])
def test_negative_anywhere_prevents_partial_publication(where):
    original = triples(2, 'prefers')
    before = tuple(original)
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        fail = recheck if where == 'recheck' else not recheck
        return reply(batch, ['uses', None] if fail else ['uses', 'uses'])
    with pytest.raises(gate.GroundingGateError, match='verdict:unsupported'):
        gate.ground_triples(original, records(2), (), '', invoke)
    assert tuple(original) == before
    assert calls == ([False] if where == 'initial' else [False, True])


def test_multi_batch_global_recheck_is_not_just_changed_batch():
    original = triples(9)
    original[8] = replace(original[8], predicate='prefers')
    calls = []
    def invoke(request, batch, recheck):
        calls.append((tuple(t.source_message_id for t in batch.triples), recheck))
        return reply(batch, ['uses']*len(batch.triples))
    result = gate.ground_triples(original, records(9), (), '', invoke)
    assert calls == [(tuple(range(1, 9)), False), ((9,), False),
                     (tuple(range(1, 9)), True), ((9,), True)]
    assert all(t.predicate == 'uses' for t in result)


@pytest.mark.parametrize('polarity', [1, -1])
def test_collision_or_conflict_rejected_before_any_recheck(polarity):
    original = triples()
    original.append(replace(original[0], predicate='prefers', polarity=polarity))
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        return reply(batch, ['uses', 'uses'])
    with pytest.raises(gate.GroundingGateError, match='correction:(collision|conflict)'):
        gate.ground_triples(original, records(), (), '', invoke)
    assert calls == [False]
    assert original[1].predicate == 'prefers'


def test_further_recheck_correction_cannot_recurse():
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        return reply(batch, ['uses' if not recheck else 'rejects'])
    with pytest.raises(gate.GroundingGateError, match='correction'):
        gate.ground_triples(triples(predicate='prefers'), records(), (), '', invoke)
    assert calls == [False, True]


@pytest.mark.parametrize('raw', ['{', '{}', 'not json', '[]'])
def test_malformed_output_has_no_repair_reroll(raw):
    calls = []
    def invoke(*args): calls.append(True); return raw
    with pytest.raises(gate.GroundingGateError):
        gate.ground_triples(triples(), records(), (), '', invoke)
    assert calls == [True]


def test_provider_failure_propagates_without_retry_or_partial_mutation():
    original = triples(9, 'prefers')
    before = tuple(original)
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        if len(calls) == 2: raise RuntimeError('fixed_provider_failure')
        return reply(batch, ['uses']*len(batch.triples))
    with pytest.raises(RuntimeError, match='fixed_provider_failure'):
        gate.ground_triples(original, records(9), (), '', invoke)
    assert calls == [False, False] and tuple(original) == before


def test_legacy_null_evidence_stays_null():
    claim = replace(triples()[0], source_message_id=None)
    def invoke(request, batch, recheck):
        assert batch.triples[0].source_message_id is None
        assert batch.sources[0].source_message_id is None
        return reply(batch, ['uses'])
    assert gate.ground_triples([claim], None, (), 'I use tool0.', invoke) == [claim]


def test_empty_claims_do_not_invoke_or_create_sources():
    def forbidden(*args): raise AssertionError('unexpected call')
    assert gate.ground_triples([], None, (), '', forbidden) == []


def test_gate_and_real_transport_integrate_without_claiming_synthetic_accuracy():
    support = runpy.run_path(str(Path(__file__).with_name('test_codex_subscription_classification_root.py')))
    c, Base = support['c'], support['Protocol']
    class SyntheticProtocol(Base):
        instances = []
        next_response = None
        def rpc(self, method, params, **kwargs):
            result = super().rpc(method, params, **kwargs)
            if method == 'turn/start':
                for event in self.pending:
                    if event['method'] == 'item/completed':
                        event['params']['item']['text'] = self.next_response
            return result
    limit = c.warm.BudgetLimits(2, 1000, 120)
    item = c.ClassificationSubscriptionClient('unused', c.warm.SharedBudget(limit), 'q', limit,
                                             session_factory=SyntheticProtocol)
    calls = []
    def invoke(request, batch, recheck):
        calls.append(recheck)
        SyntheticProtocol.next_response = reply(batch, ['uses']*len(batch.triples))
        return item.complete_grounding(request, batch)
    try:
        original = triples(2, 'prefers')
        result = gate.ground_triples(original, records(2), (), '', invoke)
        assert calls == [False, True]
        assert all(t.predicate == 'uses' for t in result)
        assert item.observed_turns == 2 and item.observed_tokens == 22
        assert item.usage_complete
        assert all(x['output_schema_sent'] is True for x in item.requested_controls)
    finally:
        item.close()
    assert all(s.closed for s in SyntheticProtocol.instances)
