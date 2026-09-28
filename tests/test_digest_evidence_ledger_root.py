"""Independent offline attacks against evidence location/authority, not LLM quality."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import socket

import pytest

from benchmarks import digest_evidence_ledger as ledger
from tests.digest_evidence_ledger_fixtures import build_cases
from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline
from hymem.extraction.llm import LLMRequest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('root ledger tests prohibit networking')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def packet():
    prior = 'Earlier. '
    visible = 'I will visit Linden, not Harbor. The gate is closed.'
    def source(cid, mid, role, text, *, peer=None):
        return {'chunk_id': cid, 'message_id': mid, 'role': role, 'source_peer_id': peer,
                'source_workspace_id': 'workspace-α', 'start': 0, 'end': len(text),
                'visible_content': text, 'interpretation_only_context': None}
    a = source('root-a', 101, 'user', visible)
    a.update(start=len(prior), end=len(prior) + len(visible), interpretation_only_context={
        'message_id': 101, 'role': 'user', 'source_peer_id': None,
        'source_workspace_id': 'workspace-α', 'start': 0, 'end': len(prior), 'content': prior})
    b = source('root-b', 102, 'assistant', 'Run door inspect, then door verify.', peer='peer-β')
    c = source('root-c', 103, 'user', 'UNCITED_OTHER_TOPIC_ONLY Ada moved to Meadow.')
    procedure = {'name': 'Door check', 'description': 'Inspect the door.',
                 'steps': [{'order': 1, 'action': 'Run door inspect', 'tool': 'door'},
                           {'order': 2, 'action': 'Run door verify', 'tool': None}],
                 'triggers': ['door check'], 'entities_involved': ['door']}
    return {'schema': 'digest-fidelity-decisions-v9', 'source_catalog': [a, b, c],
        'items': [
            {'index': 0, 'candidate_title': 'Planned Linden visit',
             'candidate_body': 'I plan to visit Linden; the gate is closed.', 'candidate_outcome': 'deferred',
             'candidate_key_entities': ['Linden', 'gate'], 'cited_source_ids': ['root-a']},
            {'index': 1, 'candidate_title': 'OTHER_CANDIDATE_ONLY Door check',
             'candidate_body': 'The assistant supplied door inspection steps.', 'candidate_outcome': 'informational',
             'candidate_key_entities': ['door'], 'cited_source_ids': ['root-b']}],
        'procedure_items': [{'index': 0, 'candidate': procedure, 'cited_source_ids': ['root-a']},
                            {'index': 1, 'candidate': deepcopy(procedure), 'cited_source_ids': ['root-b']}],
        'summary_item': {'index': 0, 'candidate_raw_summary': 'REJECTED_RAW_ONLY summary',
            'candidate_summary': 'Earlier plans remain context. The gate is closed.',
            'candidate_is_noop': False, 'new_source_ids': ['root-a', 'root-b', 'root-c'],
            'prior_derived_summary': 'PRIOR_ONLY_PERSON Earlier plans remain context.'}}


TEMPLATE = LLMRequest(system='IGNORED_SYSTEM_ONLY', user='IGNORED_USER_ONLY',
                      max_tokens=1777, temperature=.25)


def prepare(value=None, **kwargs):
    return ledger.prepare_evidence_ledger(packet() if value is None else value, TEMPLATE,
                                         max_calls=kwargs.pop('max_calls', 5), **kwargs)


def reply(scope, verdict='supported'):
    evidence = next((s for s in scope.evidence_sources
                     if s.allowed_use in {'support', 'continuity'} and s.text.strip()), None)
    return {'schema': ledger.VERSION, 'scope': {'kind': scope.kind, 'index': scope.index},
        'claims': [{'field': field.path, 'text': field.text, 'verdict': verdict,
                    'evidence': ([{'source_id': evidence.source_id, 'quote': evidence.text,
                                   'use': evidence.allowed_use}] if evidence and verdict == 'supported' else [])}
                   for field in scope.fields]}


def parsed(scope, data=None, **kwargs):
    raw = json.dumps(reply(scope) if data is None else data, ensure_ascii=False)
    return ledger.parse_evidence_ledger(raw, scope, **kwargs)


class Scripted:
    def __init__(self, plan, responses=None):
        self.plan, self.responses, self.calls = plan, responses or {}, []

    def complete(self, request):
        index = len(self.calls)
        assert request == self.plan.requests[index].request
        self.calls.append(request)
        value = self.responses.get(index, json.dumps(reply(self.plan.requests[index]), ensure_ascii=False))
        if isinstance(value, BaseException):
            raise value
        return value


def test_root_canonical_scope_order_lossless_fields_and_sampling():
    plan = prepare()
    assert [(s.kind, s.index) for s in plan.requests] == [
        ('episode', 0), ('episode', 1), ('procedure', 0), ('procedure', 1), ('summary', 0)]
    for scope in plan.requests:
        assert (scope.request.max_tokens, scope.request.temperature, scope.request.response_format) == (1777, .25, 'json')
        data = json.loads(scope.request.user)
        assert set(data) == {'schema', 'scope', 'fields', 'evidence_sources'}
        assert data['schema'] == ledger.VERSION
        assert data['scope'] == {'kind': scope.kind, 'index': scope.index}
        assert [(f['path'], f['text']) for f in data['fields']] == [(f.path, f.text) for f in scope.fields]
        assert 'REJECTED_RAW_ONLY' not in scope.request.user
        assert 'IGNORED_SYSTEM_ONLY' not in scope.request.system and 'IGNORED_USER_ONLY' not in scope.request.user
        if scope.kind != 'summary':
            assert 'PRIOR_ONLY_PERSON' not in scope.request.user
            assert 'UNCITED_OTHER_TOPIC_ONLY' not in scope.request.user
    assert [f.path for f in plan.requests[0].fields] == [
        '/candidate_title', '/candidate_body', '/candidate_outcome',
        '/candidate_key_entities/0', '/candidate_key_entities/1']
    assert [f.path for f in plan.requests[2].fields] == [
        '/candidate/name', '/candidate/description', '/candidate/steps/0/action',
        '/candidate/steps/0/tool', '/candidate/steps/1/action', '/candidate/triggers/0',
        '/candidate/entities_involved/0']


def test_root_sources_keep_coordinate_spaces_and_metadata_authority():
    plan = prepare()
    sources = plan.requests[0].evidence_sources
    assert [s.source_id for s in sources] == ['s0', 's1', 's2', 's3']
    assert [s.kind for s in sources] == ['canonical_text', 'attribution', 'attribution', 'boundary_context']
    assert [(s.field, s.allowed_use) for s in sources] == [
        ('visible_content', 'support'), ('role', 'support'),
        ('source_workspace_id', 'support'), ('content', 'interpretation')]
    assert sources[0].start == 9 and sources[0].end == 9 + len(sources[0].text)
    assert sources[-1].start == 0 and sources[-1].end == 9
    assert sources[-1].message_id == 101 and sources[-1].chunk_id == 'root-a'
    assert sources[2].start == 0 and sources[2].end == len('workspace-α')
    for scope in plan.requests[:-1]:
        assert all(s.kind != 'prior_summary' for s in scope.evidence_sources)
    prior = plan.requests[-1].evidence_sources[-1]
    assert (prior.kind, prior.allowed_use, prior.chunk_id, prior.message_id, prior.start) == (
        'prior_summary', 'continuity', None, None, 0)


def test_root_duplicate_procedures_keep_separate_evidence():
    first, second = prepare().requests[2:4]
    assert first.fields == second.fields
    assert {s.chunk_id for s in first.evidence_sources} == {'root-a'}
    assert {s.chunk_id for s in second.evidence_sources} == {'root-b'}
    assert first.binding_sha256 != second.binding_sha256


def test_root_exact_coverage_derives_offsets_not_model_supplied_numbers():
    scope = prepare().requests[0]
    data = reply(scope)
    original = data['claims'][1]
    cut = original['text'].index(';') + 1
    data['claims'][1:2] = [{**original, 'text': original['text'][:cut]},
                         {**original, 'text': original['text'][cut:]}]
    result = parsed(scope, data)
    assert result.status == 'valid_ledger'
    claims = [c for c in result.claims if c.field == '/candidate_body']
    assert [(c.start, c.end) for c in claims] == [(0, cut), (cut, len(original['text']))]
    assert ''.join(c.text for c in claims) == original['text']


@pytest.mark.parametrize('damage', [
    lambda d: d['claims'].pop(0),
    lambda d: d['claims'].pop(),
    lambda d: d['claims'].reverse(),
    lambda d: d['claims'].insert(0, deepcopy(d['claims'][0])),
    lambda d: d['claims'][0].update(text=d['claims'][0]['text'][1:]),
    lambda d: d['claims'][1].update(text=d['claims'][1]['text'].replace('closed', 'open')),
    lambda d: d['claims'][1].update(text=d['claims'][1]['text'].rstrip('.') ),
    lambda d: d['claims'][0].update(text=''),
    lambda d: d['claims'][0].update(field='/uncited_field'),
    lambda d: d['claims'][0].update(start=0),
    lambda d: d['claims'][0].update(end=1),
    lambda d: d['claims'][0].update(verdict='probably_supported'),
    lambda d: d['claims'][0].update(verdict=True),
    lambda d: d.update(approved=True),
    lambda d: d['scope'].update(index=True),
    lambda d: d['scope'].update(index=1),
    lambda d: d['scope'].update(kind='summary'),
    lambda d: d.update(schema='digest-evidence-isolation-v1'),
])
def test_root_omission_rewriting_reordering_and_contract_drift_rejected(damage):
    scope = prepare().requests[0]
    data = reply(scope)
    damage(data)
    result = parsed(scope, data)
    assert result.status == 'malformed_ledger' and result.claims == ()
    assert not result.model_all_supported and not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize('damage', [
    lambda ref: ref.update(source_id='not-in-this-scope'),
    lambda ref: ref.update(source_id=True),
    lambda ref: ref.update(quote='not present in the evidence'),
    lambda ref: ref.update(quote=''),
    lambda ref: ref.update(quote=' '),
    lambda ref: ref.update(use='continuity'),
    lambda ref: ref.update(use='interpretation'),
    lambda ref: ref.update(start=9),
    lambda ref: ref.update(extra='invented'),
])
def test_root_evidence_references_are_exact_and_scope_authorized(damage):
    scope = prepare().requests[0]
    data = reply(scope)
    damage(data['claims'][0]['evidence'][0])
    assert parsed(scope, data).status == 'malformed_ledger'


def test_root_duplicate_reference_rejected_but_reuse_across_claims_allowed():
    scope = prepare().requests[0]
    data = reply(scope)
    assert parsed(scope, data).status == 'valid_ledger'
    data['claims'][0]['evidence'] *= 2
    assert parsed(scope, data).status == 'malformed_ledger'


def test_root_boundary_context_is_not_a_primary_support_source():
    scope = prepare().requests[0]
    context = next(s for s in scope.evidence_sources if s.kind == 'boundary_context')
    data = reply(scope)
    ref = {'source_id': context.source_id, 'quote': context.text, 'use': 'interpretation'}
    data['claims'][0]['evidence'] = [ref]
    assert parsed(scope, data).status == 'malformed_ledger'
    data['claims'][0]['evidence'] = reply(scope)['claims'][0]['evidence'] + [ref]
    assert parsed(scope, data).status == 'valid_ledger'


def test_root_prior_summary_is_derived_continuity_never_canonical():
    scope = prepare().requests[-1]
    prior = next(s for s in scope.evidence_sources if s.kind == 'prior_summary')
    data = reply(scope)
    data['claims'][0]['evidence'] = [{'source_id': prior.source_id, 'quote': prior.text, 'use': 'continuity'}]
    result = parsed(scope, data)
    assert result.status == 'valid_ledger'
    assert result.claims[0].evidence[0].kind == 'prior_summary'
    assert not result.semantic_verified and not result.publication_authorized
    data['claims'][0]['evidence'][0]['use'] = 'support'
    assert parsed(scope, data).status == 'malformed_ledger'


@pytest.mark.parametrize('verdict', ['unsupported', 'uncertain'])
def test_root_negative_or_uncertain_claim_can_lack_evidence_but_cannot_launder_invalid_quotes(verdict):
    scope = prepare().requests[0]
    data = reply(scope, verdict)
    result = parsed(scope, data)
    assert result.status == 'valid_ledger' and not result.model_all_supported
    data['claims'][0]['evidence'] = [{'source_id': 'fake', 'quote': 'fake', 'use': 'support'}]
    assert parsed(scope, data).status == 'malformed_ledger'


def test_root_supported_claim_without_primary_evidence_fails():
    scope = prepare().requests[0]
    data = reply(scope)
    data['claims'][0]['evidence'] = []
    assert parsed(scope, data).status == 'malformed_ledger'


def test_root_unique_unicode_quote_resolves_original_codepoint_offsets():
    value = packet()
    source = value['source_catalog'][0]
    source['visible_content'] = 'Å😀 é gates stay closed.'
    source['end'] = source['start'] + len(source['visible_content'])
    scope = prepare(value).requests[0]
    data = reply(scope)
    data['claims'][0]['evidence'][0]['quote'] = 'é gates'
    result = parsed(scope, data)
    assert result.status == 'valid_ledger'
    ref = result.claims[0].evidence[0]
    expected = source['start'] + source['visible_content'].index('é gates')
    assert (ref.start, ref.end, ref.quote, ref.message_id) == (expected, expected + len('é gates'), 'é gates', 101)
    data['claims'][0]['evidence'][0]['quote'] = 'é gates'
    assert parsed(scope, data).status == 'malformed_ledger'


def test_root_repeated_quote_is_ambiguous_including_overlapping_occurrences():
    value = packet()
    source = value['source_catalog'][0]
    source['visible_content'] = 'banana, banana'
    source['end'] = source['start'] + len(source['visible_content'])
    scope = prepare(value).requests[0]
    for quote in ('banana', 'ana'):
        data = reply(scope)
        data['claims'][0]['evidence'][0]['quote'] = quote
        assert parsed(scope, data).status == 'malformed_ledger'


def test_root_real_but_irrelevant_quote_is_not_semantic_proof():
    scope = prepare().requests[0]
    data = reply(scope)
    for claim in data['claims']:
        claim['evidence'][0]['quote'] = 'The gate is closed.'
    result = parsed(scope, data)
    # This real quote cannot entail the visit title or Linden entity. The ledger
    # certifies location/authority, NOT the model's unsupported entailment claim.
    assert result.status == 'valid_ledger' and result.model_all_supported
    assert not result.semantic_verified and not result.publication_authorized


def test_root_empty_noop_never_becomes_semantic_approval():
    value = packet()
    value.update(items=[], procedure_items=[], source_catalog=[])
    value['summary_item'].update(candidate_summary='', prior_derived_summary='', candidate_raw_summary='',
                                 candidate_is_noop=True, new_source_ids=[])
    plan = prepare(value, max_calls=1)
    scope = plan.requests[0]
    assert scope.fields == () and scope.evidence_sources == ()
    outcome = parsed(scope)
    assert outcome.status == 'valid_ledger' and not outcome.model_all_supported
    result = ledger.execute_evidence_ledger(plan, Scripted(plan))
    assert result.complete and result.ledger_structure_valid and not result.model_all_supported
    assert not result.semantic_verified and not result.publication_authorized


def test_root_empty_summary_does_not_erase_supported_nonempty_episode_claims():
    value = packet()
    value['summary_item'].update(candidate_summary='', prior_derived_summary='', candidate_raw_summary='',
                                 candidate_is_noop=True)
    plan = prepare(value)
    result = ledger.execute_evidence_ledger(plan, Scripted(plan))
    assert result.complete and result.ledger_structure_valid
    assert result.outcomes[-1].claims == () and not result.outcomes[-1].model_all_supported
    assert result.model_all_supported  # Non-vacuous: every recorded episode/procedure claim is supported.
    assert not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize('value', [[0], [True], [[]], [[], []], {'a': []}])
def test_root_json_bound_is_not_smaller_than_actual_exact_serialization(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    assert ledger._bounded_json(value, len(encoded)) == encoded


def test_root_null_and_empty_strings_omitted_but_whitespace_fields_are_covered():
    value = packet()
    value['items'][0]['candidate_outcome'] = None
    value['procedure_items'][0]['candidate']['description'] = ''
    value['procedure_items'][1]['candidate']['description'] = ' \n '
    plan = prepare(value)
    assert '/candidate_outcome' not in [f.path for f in plan.requests[0].fields]
    assert '/candidate/description' not in [f.path for f in plan.requests[2].fields]
    field = next(f for f in plan.requests[3].fields if f.path == '/candidate/description')
    assert field.text == ' \n ' and parsed(plan.requests[3]).status == 'valid_ledger'


@pytest.mark.parametrize('raw', [None, True, 1, [], '{}', '[]', '{', 'not JSON',
    '{"schema":"x","schema":"y"}', '{"schema":NaN}', '{"schema":1e999}',
    '```json\n{}\n```', '{"schema":"x"}\nignored'])
def test_root_strict_certificate_parser_does_not_repair_or_salvage(raw):
    assert ledger.parse_evidence_ledger(raw, prepare().requests[0]).status == 'malformed_ledger'


def test_root_invalid_unicode_cannot_enter_claims_or_quotes():
    scope = prepare().requests[0]
    for part in ('claim', 'quote'):
        data = reply(scope)
        if part == 'claim':
            data['claims'][0]['text'] = '\ud800'
        else:
            data['claims'][0]['evidence'][0]['quote'] = '\ud800'
        assert parsed(scope, data).status == 'malformed_ledger'


@pytest.mark.parametrize('kwargs', [{'max_calls': 4}, {'max_calls': True}, {'max_calls': 0},
    {'max_claims': 1}, {'max_claims': True}, {'max_evidence_per_claim': 0},
    {'max_evidence_per_claim': True}, {'max_input_chars': 1}, {'max_input_chars': False},
    {'max_output_chars': 0}, {'max_output_chars': True}])
def test_root_impossible_or_invalid_reservations_fail_before_any_execution(kwargs):
    with pytest.raises(ValueError):
        prepare(**kwargs)


def test_root_parser_size_and_count_caps_reject_whole_ledger():
    scope = prepare().requests[0]
    data = reply(scope)
    for kwargs in ({'max_output_chars': 1}, {'max_claims': 1}):
        assert parsed(scope, data, **kwargs).status == 'malformed_ledger'
    context = next(s for s in scope.evidence_sources if s.kind == 'boundary_context')
    data['claims'][0]['evidence'].append({'source_id': context.source_id, 'quote': context.text, 'use': 'interpretation'})
    assert parsed(scope, data, max_evidence_per_claim=1).status == 'malformed_ledger'


def test_root_original_mutation_does_not_mutate_frozen_plan():
    value = packet()
    plan = prepare(value)
    snapshot = plan.requests[0].request.user
    value['items'][0]['candidate_body'] = 'Mutated after preparation'
    value['source_catalog'][0]['visible_content'] = 'Changed'
    assert plan.requests[0].request.user == snapshot
    with pytest.raises(FrozenInstanceError):
        plan.requests[0].fields[0].text = 'Changed'


@pytest.mark.parametrize('damage', [
    lambda p: replace(p, plan_sha256='0' * 64),
    lambda p: replace(p, input_sha256='0' * 64),
    lambda p: replace(p, requests=p.requests[:-1]),
    lambda p: replace(p, requests=p.requests[::-1]),
    lambda p: replace(p, requests=(replace(p.requests[0], index=True), *p.requests[1:])),
    lambda p: replace(p, requests=(replace(p.requests[0], fields=p.requests[0].fields[:-1]), *p.requests[1:])),
    lambda p: replace(p, requests=(replace(p.requests[0], binding_sha256='0' * 64), *p.requests[1:])),
])
def test_root_forged_plan_fails_before_client_call(damage):
    plan = prepare()
    client = Scripted(plan)
    with pytest.raises(ValueError):
        ledger.execute_evidence_ledger(damage(plan), client)
    assert not client.calls


def test_root_standalone_scope_mutation_cannot_rebind_evidence():
    scope = prepare().requests[0]
    bad_source = replace(scope.evidence_sources[0], text='Different source')
    forged = replace(scope, evidence_sources=(bad_source, *scope.evidence_sources[1:]))
    with pytest.raises(ValueError):
        parsed(forged, reply(scope))


def test_root_malformed_and_negative_ledgers_do_not_skip_later_scopes():
    plan = prepare()
    client = Scripted(plan, {0: '{}', 1: json.dumps(reply(plan.requests[1], 'unsupported'))})
    result = ledger.execute_evidence_ledger(plan, client)
    assert result.complete and result.attempted_calls == 5 and len(client.calls) == 5
    assert result.outcomes[0].status == 'malformed_ledger'
    assert result.outcomes[1].status == 'valid_ledger'
    assert not result.ledger_structure_valid and not result.model_all_supported
    assert not result.semantic_verified and not result.publication_authorized


def test_root_ordinary_exception_halts_without_exposing_exception_text():
    plan = prepare()
    client = Scripted(plan, {1: RuntimeError('SECRET_CLIENT_FAILURE_TEXT')})
    result = ledger.execute_evidence_ledger(plan, client)
    assert not result.complete and result.attempted_calls == 2 and len(client.calls) == 2
    assert result.outcomes[-1].status == 'execution_error'
    assert 'SECRET_CLIENT_FAILURE_TEXT' not in repr(result)
    assert not result.model_all_supported and not result.publication_authorized


@pytest.mark.parametrize('exception', [DeadlineExceeded('expired'), KeyboardInterrupt(), SystemExit(2)])
def test_root_deadlines_and_process_interrupts_propagate(exception):
    plan = prepare()
    client = Scripted(plan, {0: exception})
    with pytest.raises(type(exception)):
        ledger.execute_evidence_ledger(plan, client)
    assert len(client.calls) == 1


def test_root_expired_shared_deadline_prevents_underlying_call():
    plan = prepare()
    client = Scripted(plan)
    wrapped = DeadlineBoundLLMClient(client, MonotonicDeadline(0))
    with pytest.raises(DeadlineExceeded):
        ledger.execute_evidence_ledger(plan, wrapped)
    assert client.calls == []


@pytest.mark.parametrize('case', build_cases(), ids=lambda case: case['id'])
def test_root_fresh_semantic_controls_project_without_leaking_labels(case):
    value = case['payload']
    plan = ledger.prepare_evidence_ledger(value, TEMPLATE,
        max_calls=len(value['items']) + len(value['procedure_items']) + 1)
    target = case['target']
    scope = next(s for s in plan.requests if (s.kind, s.index) == (target['kind'], target['index']))
    assert target['field'] in {f.path for f in scope.fields}
    assert case['reason'] not in scope.request.user
    assert 'expected' not in json.loads(scope.request.user)
    # Scripted output exercises plumbing ONLY, never proves the gold label.
    result = parsed(scope, reply(scope, target['expected']))
    assert result.ledger_structure_valid
    assert not result.semantic_verified and not result.publication_authorized
    if case['pair'] == 'identity-authority':
        assert not any(s.kind == 'prior_summary' for s in scope.evidence_sources)
    if case['pair'] == 'citation-removal' and target['expected'] == 'unsupported':
        assert {s.chunk_id for s in scope.evidence_sources} == {'citation-a'}
        assert 'The access card was rotated.' not in [s.text for s in scope.evidence_sources]
    if case['pair'] == 'context-not-evidence':
        assert any(s.kind == 'boundary_context' and s.allowed_use == 'interpretation'
                   and 'West Club' in s.text for s in scope.evidence_sources)


def test_root_fresh_controls_are_six_paired_single_variable_interventions():
    cases = build_cases()
    assert len(cases) == 12 and len({c['id'] for c in cases}) == 12
    for faithful, defective in zip(cases[::2], cases[1::2]):
        assert faithful['pair'] == defective['pair']
        assert faithful['target']['expected'] == 'supported'
        assert defective['target']['expected'] == 'unsupported'
        before, after = deepcopy(faithful['payload']), deepcopy(defective['payload'])
        if faithful['pair'] == 'citation-removal':
            after['items'][0]['cited_source_ids'] = before['items'][0]['cited_source_ids']
        elif faithful['target']['kind'] == 'procedure':
            after['procedure_items'][0]['candidate']['steps'][1]['action'] = before['procedure_items'][0]['candidate']['steps'][1]['action']
        else:
            key = faithful['target']['field'][1:]
            after['items'][0][key] = before['items'][0][key]
        assert before == after
    cases[0]['payload']['source_catalog'][0]['visible_content'] = 'mutated'
    assert build_cases()[0]['payload']['source_catalog'][0]['visible_content'] != 'mutated'
