"""Independent staged-gate controls; invented attestations, not model accuracy."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from hymem.extraction import grounding_staged_v1 as s
from hymem.extraction import grounding_staged_gate_v1 as gate
from hymem.extraction import grounding_classification_v4 as g

H = runpy.run_path(str(Path(__file__).with_name('test_grounding_staged_v1_root.py')))


def inputs(count=1, predicate='uses'):
    records = tuple((i + 1, json.dumps(dict(source_message_id=i + 1,
        content=f'I use tool{i}. More details.', source_role='user',
        source_peer_id='Mira', source_created_at='2026-09-29'))) for i in range(count))
    claims = [g.Triple('Mira', predicate, f'tool{i}', 1, source_message_id=i + 1)
              for i in range(count)]
    return claims, records


def response(bound, stage, predicates=None, ambiguous=()):
    batch = bound.classification_batch if stage == 'alternatives' else bound
    predicates = predicates if predicates is not None else ['uses'] * len(batch.triples)
    pools = [[H['quote'](t.source_message_id, next(src.content for src in batch.sources
        if src.source_message_id == t.source_message_id))] for t in batch.triples]
    if stage == 'original':
        return json.dumps(H['original'](batch, [
            'ambiguous' if i in ambiguous else 'supported' if p == t.predicate else 'not_established'
            for i, (t, p) in enumerate(zip(batch.triples, predicates, strict=True))], pools))
    assert stage == 'alternatives'
    first = json.loads(bound.original_response_canonical_json)
    _, rebound, result = H['alternatives'](batch, first,
        positive=[(i, p) for i, p in enumerate(predicates) if p is not None], pools=pools)
    assert rebound == bound
    return json.dumps(result)


def validate(request, bound, stage, recheck):
    assert type(recheck) is bool
    if stage == 'original':
        assert type(bound) is g.ClassificationBatch
        s.validate_original_request(request, bound)
    else:
        assert stage == 'alternatives' and not recheck
        assert type(bound) is s.AlternativesBatch
        s.validate_alternatives_request(request, bound)


def test_accepted_dependencies_unchanged():
    for module, expected in ((s, '4862e4aedba5be756ea877d65a91b515a142b2fb46e2efb6f91c800e5096b3c9'),
                             (g, '37ab836c45cb306d5e67d17066aed107578a812d37ffc4f06fd47ecf5fd667d2')):
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected


def test_nine_claims_late_correction_rechecks_all_and_changes_only_predicate():
    claims, records = inputs(9)
    claims[8] = replace(claims[8], predicate='prefers', value_text='daily',
                        value_numeric=2, value_unit='hours', temporal_scope='during trial')
    before = tuple(claims)
    calls = []
    def invoke(request, bound, stage, recheck):
        validate(request, bound, stage, recheck)
        b = bound.classification_batch if stage == 'alternatives' else bound
        calls.append((b.triples, stage, recheck))
        return response(bound, stage)
    result = gate.ground_triples(claims, records, (), '', invoke)
    assert [(len(t), s, r) for t, s, r in calls] == [
        (8, 'original', False), (1, 'original', False), (1, 'alternatives', False),
        (8, 'original', True), (1, 'original', True)]
    assert tuple(claims) == before
    assert result == list(before[:8]) + [replace(before[8], predicate='uses')]
    assert calls[3][0] + calls[4][0] == tuple(result)


@pytest.mark.parametrize('failure_call', [2, 3, 4, 5, 6])
def test_failed_later_stage_is_atomic_without_retry(failure_call):
    claims, records = inputs(9, 'prefers')
    before = tuple(claims)
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append((stage, recheck))
        return 'private-invalid-sentinel' if len(calls) == failure_call else response(bound, stage)
    with pytest.raises(gate.GroundingGateError) as exc:
        gate.ground_triples(claims, records, (), '', invoke)
    assert len(calls) == failure_call and tuple(claims) == before
    assert 'private-invalid-sentinel' not in str(exc.value)


def test_ambiguous_and_negative_originals_do_not_authorize_alternatives():
    claims, records = inputs(2, 'prefers')
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append(stage)
        return response(bound, stage, ambiguous=(1,))
    with pytest.raises(gate.GroundingGateError, match='verdict:uncertain'):
        gate.ground_triples(claims, records, (), '', invoke)
    assert calls == ['original']


@pytest.mark.parametrize('ambiguous', [False, True])
def test_recheck_cannot_start_alternative_search(ambiguous):
    claims, records = inputs(predicate='prefers')
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append((stage, recheck))
        return response(bound, stage, ['avoids'] if recheck else None,
                        (0,) if recheck and ambiguous else ())
    with pytest.raises(gate.GroundingGateError):
        gate.ground_triples(claims, records, (), '', invoke)
    assert calls == [('original', False), ('alternatives', False), ('original', True)]


@pytest.mark.parametrize('polarity', [1, -1])
def test_cross_batch_collision_and_conflict_before_recheck(polarity):
    claims, records = inputs(9)
    claims[8] = replace(claims[0], predicate='prefers', polarity=polarity)
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append((stage, recheck))
        return response(bound, stage)
    with pytest.raises(gate.GroundingGateError, match='correction:(collision|conflict)'):
        gate.ground_triples(claims, records, (), '', invoke)
    assert calls == [('original', False), ('original', False), ('alternatives', False)]


@pytest.mark.parametrize('failure_call', [1, 2, 3])
def test_callback_exception_preserves_identity_and_no_extra_invocations(failure_call):
    claims, records = inputs(predicate='prefers')
    sentinel = RuntimeError('caller-budget-sentinel')
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append(stage)
        if len(calls) == failure_call:
            raise sentinel
        return response(bound, stage)
    with pytest.raises(RuntimeError) as exc:
        gate.ground_triples(claims, records, (), '', invoke)
    assert exc.value is sentinel and len(calls) == failure_call


def test_prior_response_binding_is_not_replaced_by_reconstructed_negatives():
    claims, records = inputs(2)
    claims[1] = replace(claims[1], predicate='prefers')
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append(stage)
        raw = response(bound, stage)
        if stage == 'alternatives':
            first = json.loads(bound.original_response_canonical_json)
            first['originals'][0]['original']['support']['evidence'][0]['quote'] = 'I use tool0.'
            altered = s.parse_original_response(json.dumps(first), bound.classification_batch)
            payload = json.loads(raw)
            payload['original_response_sha256'] = altered.response_sha256
            raw = json.dumps(payload)
        return raw
    with pytest.raises(gate.GroundingGateError, match='prior_binding'):
        gate.ground_triples(claims, records, (), '', invoke)
    assert calls == ['original', 'alternatives']


def test_rejected_positive_alternative_never_saved_by_another_positive():
    claims, records = inputs(predicate='prefers')
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append(stage)
        raw = json.loads(response(bound, stage))
        if stage == 'alternatives':
            entry = raw['alternatives'][0]['alternatives']['uses']
            entry['support']['evidence'][0]['region'] = 'conversation_0'
        return json.dumps(raw)
    with pytest.raises(gate.GroundingGateError, match='context_missing'):
        gate.ground_triples(claims, records, (), '', invoke)
    assert calls == ['original', 'alternatives']


def test_empty_no_call_and_legacy_null_provenance():
    def forbidden(*args):
        raise AssertionError('unexpected callback')
    assert gate.ground_triples([], None, (), '', forbidden) == []
    claims, _ = inputs()
    claims[0] = replace(claims[0], source_message_id=None)
    def invoke(request, bound, stage, recheck):
        assert bound.sources[0].source_message_id is None
        assert bound.sources[0].content == 'I use tool0.'
        return response(bound, stage)
    assert gate.ground_triples(claims, None, (), 'I use tool0.', invoke) == claims


def test_global_eight_quote_limit_across_alternatives_is_preserved():
    claims, records = inputs(predicate='prefers')
    content = ' '.join('word' + str(i) for i in range(9))
    records = ((1, json.dumps(dict(content=content))),)
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append(stage)
        raw = json.loads(response(bound, stage))
        if stage == 'alternatives':
            pool = [H['quote'](1, 'word' + str(i)) for i in range(9)]
            entries = raw['alternatives'][0]['alternatives']
            entries['uses'] = H['assessment'](claims[0], 'supported', pool[:8])
            entries['avoids'] = H['assessment'](claims[0], 'supported', pool[8:])
        return json.dumps(raw)
    with pytest.raises(gate.GroundingGateError, match='global_bounds'):
        gate.ground_triples(claims, records, (), '', invoke)
    assert calls == ['original', 'alternatives']


@pytest.mark.parametrize('phase', ['original', 'alternatives', 'recheck'])
def test_prefix_scope_rejection_survives_every_stage(phase):
    claims, _ = inputs(predicate='prefers')
    records = ((1, json.dumps(dict(content='I use tool0. Later unrelated.',
        source_boundary_context=dict(content='Earlier context.',
            applies_through_source_content_end=12)))),)
    calls = []
    def invoke(request, bound, stage, recheck):
        calls.append((stage, recheck))
        current_phase = 'recheck' if recheck else stage
        raw = json.loads(response(bound, stage,
            ['prefers'] if phase == 'original' else None))
        if current_phase == phase:
            batch = bound.classification_batch if stage == 'alternatives' else bound
            entry = H['assessment'](batch.triples[0], 'supported', [
                H['quote'](1, 'Later unrelated.'), H['quote'](1, 'Earlier context.', 'boundary')])
            if stage == 'alternatives':
                raw['alternatives'][0]['alternatives']['uses'] = entry
            else:
                raw['originals'][0]['original'] = entry
        return json.dumps(raw)
    with pytest.raises(gate.GroundingGateError, match='context_scope'):
        gate.ground_triples(claims, records, (), '', invoke)
    assert len(calls) == ['original', 'alternatives', 'recheck'].index(phase) + 1


def test_parent_and_role_metadata_carried_identically_through_three_stages():
    claims, _ = inputs(predicate='prefers')
    content = 'I use tool0. More details.'
    records = ((1, json.dumps(dict(content=content, source_role='user',
        source_peer_id='Mira', source_created_at='2026-09-29', source_content_start=10))),)
    contexts = ((11, json.dumps(dict(content='| tool0 | yes | More details.',
        source_message_id=11, source_role='assistant', source_peer_id='helper',
        source_created_at='2026-09-28', source_content_start=20,
        context_for_source_message_id=1, applies_through_source_content_end=22,
        source_fragment_context=dict(content='| Tool | Use |', prelude_content='Context header',
            applies_through_source_content_end=35)))),)
    seen = []
    def invoke(request, bound, stage, recheck):
        batch = bound.classification_batch if stage == 'alternatives' else bound
        src = batch.sources[0]
        assert (src.source_role, src.source_peer_id, src.source_created_at) == ('user', 'Mira', '2026-09-29')
        regions = {c.region: c for c in src.contexts}
        parent = regions['conversation_0']
        assert (parent.source_message_id, parent.source_role, parent.source_peer_id,
            parent.source_created_at, parent.owned_prefix_chars) == (11, 'assistant', 'helper', '2026-09-28', 12)
        for name in ('conversation_0_header', 'conversation_0_prelude'):
            child = regions[name]
            assert (child.applies_to_region, child.applies_to_prefix_chars, child.owned_prefix_chars) == ('conversation_0', 15, 12)
        seen.append(src)
        return response(bound, stage)
    assert gate.ground_triples(claims, records, contexts, '', invoke)[0].predicate == 'uses'
    assert len(seen) == 3 and seen[0] == seen[1] == seen[2]


def test_import_guard_rejects_before_any_callback(monkeypatch):
    claims, records = inputs()
    monkeypatch.setattr(gate._staged, 'parse_staged_responses', lambda *args, **kwargs: None)
    def forbidden(*args):
        raise AssertionError('unexpected callback')
    with pytest.raises(gate.GroundingGateError, match='support:integrity'):
        gate.ground_triples(claims, records, (), '', forbidden)
