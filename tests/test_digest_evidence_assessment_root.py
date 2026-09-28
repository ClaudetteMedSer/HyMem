"""Independent offline contract attacks; scripted verdicts are not model evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
import socket

import pytest

from benchmarks import digest_evidence_assessment as assessment
from hymem.deadline import DeadlineExceeded
from hymem.extraction.llm import LLMRequest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('independent assessment tests forbid networking')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def packet():
    def source(cid, mid, text, role='user', preceding=None):
        start = len(preceding or '')
        return {'chunk_id': cid, 'message_id': mid, 'role': role,
                'source_peer_id': None, 'source_workspace_id': None,
                'start': start, 'end': start + len(text), 'visible_content': text,
                'interpretation_only_context': None if preceding is None else {
                    'message_id': mid, 'role': role, 'source_peer_id': None,
                    'source_workspace_id': None, 'start': 0, 'end': start,
                    'content': preceding}}
    a = source('a', 111, 'I considered the ferry but did not book it. I may decide next week.',
               preceding='A prior idea, not evidence. ')
    b = source('b', 222, 'First run latch inspect; then run latch verify. Never reset the latch.', 'assistant')
    c = source('c', 333, 'UNCITED_TOPIC_ONLY The card was rotated.')
    procedure = {'name': 'Latch check', 'description': None,
                 'steps': [{'order': 1, 'action': 'Run latch inspect', 'tool': 'latch'},
                           {'order': 2, 'action': 'Run latch verify', 'tool': 'latch'}],
                 'triggers': [], 'entities_involved': ['latch']}
    return {'schema': 'digest-fidelity-decisions-v9', 'source_catalog': [a, b, c],
        'items': [{'index': 0, 'candidate_title': 'Ferry consideration',
                   'candidate_body': 'The user considered the ferry but did not book it.',
                   'candidate_outcome': 'informational', 'candidate_key_entities': [],
                   'cited_source_ids': ['a']}],
        'procedure_items': [{'index': 0, 'candidate': procedure, 'cited_source_ids': ['b']},
                            {'index': 1, 'candidate': deepcopy(procedure), 'cited_source_ids': ['a']}],
        'summary_item': {'index': 0, 'candidate_raw_summary': 'REJECTED_RAW_ONLY',
            'candidate_summary': 'The ferry was considered but not booked. Inspection precedes verification.',
            'candidate_is_noop': False, 'new_source_ids': ['a', 'b', 'c'],
            'prior_derived_summary': 'PRIOR_NAME_ONLY Rhea discussed travel.'}}


TEMPLATE = LLMRequest(system='DISCARDED_SYSTEM_ONLY', user='DISCARDED_USER_ONLY',
                      max_tokens=3072, temperature=0.25)


def prepare(value=None, **kwargs):
    return assessment.prepare_evidence_assessment(packet() if value is None else value,
        TEMPLATE, max_calls=kwargs.pop('max_calls', 4), **kwargs)


def reply(scope, verdict='supported'):
    canonical = next((s.source_id for s in scope.evidence_sources if s.kind == 'canonical_text'), None)
    return {check.check_id: [verdict, ([] if verdict != 'supported' else
        [check.source_id if check.kind == 'retention' else canonical])]
        for check in scope.checks}


def parse(scope, data=None, **kwargs):
    return assessment.parse_evidence_assessment(json.dumps(reply(scope) if data is None else data), scope, **kwargs)


class Scripted:
    def __init__(self, values):
        self.values, self.calls = iter(values), []

    def complete(self, request):
        self.calls.append(request)
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def test_root_scope_isolation_and_parameter_preservation():
    plan = prepare()
    assert [(s.kind, s.index) for s in plan.requests] == [
        ('episode', 0), ('procedure', 0), ('procedure', 1), ('summary', 0)]
    for scope in plan.requests:
        assert (scope.request.temperature, scope.request.max_tokens, scope.request.response_format) == (.25, 3072, 'json')
        assert 'DISCARDED_SYSTEM_ONLY' not in scope.request.system
        assert 'DISCARDED_USER_ONLY' not in scope.request.user
        assert 'REJECTED_RAW_ONLY' not in scope.request.user
        assert [f.field_id for f in scope.fields] == [f'f{i}' for i in range(len(scope.fields))]
        assert [c.check_id for c in scope.checks] == [f'c{i}' for i in range(len(scope.checks))]
        assert [(f.start, f.end) for f in scope.fields] == [(0, len(f.text)) for f in scope.fields]
        if scope.kind != 'summary':
            assert 'PRIOR_NAME_ONLY' not in scope.request.user
            assert 'UNCITED_TOPIC_ONLY' not in scope.request.user
    first, second = plan.requests[1:3]
    assert first.fields == second.fields
    assert {s.chunk_id for s in first.evidence_sources} == {'b'}
    assert {s.chunk_id for s in second.evidence_sources} == {'a'}
    assert first.binding_sha256 != second.binding_sha256


def test_root_whitespace_and_unicode_are_never_roundtripped_through_model():
    value = packet()
    text = '  Café 👩🏽‍💻 e\u0301.\n and 否。  '
    value['items'][0]['candidate_body'] = text
    value['items'][0]['candidate_key_entities'] = ['latch', 'latch']
    value['procedure_items'][0]['candidate']['description'] = ' '
    plan = prepare(value)
    scope = plan.requests[0]
    field = next(f for f in scope.fields if f.path == '/candidate_body')
    assert (field.text, field.start, field.end) == (text, 0, len(text))
    assert [f.text for f in scope.fields[-2:]] == ['latch', 'latch']
    # v9 rejects blank entity names; optional description permits whitespace.
    assert next(f for f in plan.requests[1].fields if f.path == '/candidate/description').text == ' '
    response = reply(scope)
    assert all(type(row) is list and len(row) == 2 for row in response.values())
    assert text not in json.dumps(response)
    outcome = parse(scope, response)
    assert outcome.status == 'valid_assessment'
    assert len(outcome.judgments) == len(scope.checks)
    assert next(f for f in scope.fields if f.path == '/candidate_body').text == text


def test_root_repeated_quotes_need_no_arbitrary_occurrence_choice():
    scope = prepare().requests[1]
    unit = next(s for s in scope.evidence_sources if s.kind == 'canonical_text')
    assert unit.text.count('latch') == 3
    result = parse(scope)
    assert result.status == 'valid_assessment' and result.model_all_supported
    assert unit.start == 0 and unit.end == len(unit.text)
    assert not result.semantic_verified and not result.publication_authorized


def test_root_outcomes_are_classification_not_literal_source_words():
    scope = prepare().requests[0]
    field = next(f for f in scope.fields if f.path == '/candidate_outcome')
    checks = [c for c in scope.checks if field.field_id in c.field_ids]
    assert [c.kind for c in checks if c.kind != 'retention'] == ['outcome']
    assert all('informational' not in s.text for s in scope.evidence_sources)
    assert 'classifies the event' in scope.request.system.lower()
    assert 'literal' in scope.request.system.lower()
    assert parse(scope).status == 'valid_assessment'


def test_root_every_text_field_has_assertion_and_relation_obligations():
    for scope in prepare().requests:
        for field in scope.fields:
            wanted = ['outcome'] if field.path == '/candidate_outcome' else ['assertion', 'relations']
            assert [c.kind for c in scope.checks if c.kind != 'retention' and c.field_ids == (field.field_id,)] == wanted


def test_root_source_retention_is_not_candidate_text_coverage():
    scope = prepare().requests[-1]
    checks = [c for c in scope.checks if c.kind == 'retention']
    assert {c.source_id for c in checks} == {s.source_id for s in scope.evidence_sources if s.kind == 'canonical_text'}
    assert len(checks) == 3
    response = reply(scope)
    response[checks[-1].check_id] = ['unsupported', []]
    result = parse(scope, response)
    assert result.status == 'valid_assessment' and not result.model_all_supported
    # This is a supplied negative verdict, not an automatic omission detector.
    assert not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize('damage', [
    lambda r: r.pop('c0'),
    lambda r: r.update(c999=['supported', ['s0']]),
    lambda r: r.update(type='json_object'),
    lambda r: r.update(schema='invented'),
    lambda r: r.update(c0={'verdict': 'supported', 'evidence': ['s0']}),
    lambda r: r.update(c0=['supported']),
    lambda r: r.update(c0=['supported', ['s0'], 'explanation']),
    lambda r: r.update(c0=['supported', 's0']),
    lambda r: r.update(c0=['supported', []]),
    lambda r: r.update(c0=['supported', ['s0', 's0']]),
    lambda r: r.update(c0=['supported', ['s999']]),
    lambda r: r.update(c0=['supported', [True]]),
    lambda r: r.update(c0=['probably_supported', ['s0']]),
    lambda r: r.update(c0=[True, ['s0']]),
    lambda r: r.update(c0=['SUPPORTED', ['s0']]),
    lambda r: r.update(c0=['supported', [None]]),
])
def test_root_malformed_response_rejects_complete_scope(damage):
    scope = prepare().requests[0]
    response = reply(scope)
    damage(response)
    result = parse(scope, response)
    assert result.status == 'malformed_assessment' and result.judgments == ()
    assert not result.model_all_supported and not result.semantic_verified and not result.publication_authorized


@pytest.mark.parametrize('raw', ['[]', 'null', 'true', 'NaN', '{', '```json\n{}\n```',
    '{"c0":["supported",["s0"]],"c0":["unsupported",[]]}'])
def test_root_bad_json_never_repaired(raw):
    assert assessment.parse_evidence_assessment(raw, prepare().requests[0]).status == 'malformed_assessment'


def test_root_model_object_key_order_does_not_change_identity():
    scope = prepare().requests[0]
    response = reply(scope)
    assert parse(scope, dict(reversed(list(response.items())))) == parse(scope, response)


@pytest.mark.parametrize('verdict', ['unsupported', 'uncertain'])
def test_root_negative_without_citation_valid_but_unknown_reference_invalid(verdict):
    scope = prepare().requests[0]
    response = reply(scope, verdict)
    assert parse(scope, response).status == 'valid_assessment'
    response['c0'][1] = ['outside-scope']
    assert parse(scope, response).status == 'malformed_assessment'


def test_root_context_only_support_fails_and_continuity_stays_summary_only():
    scope = prepare().requests[0]
    context = next(s for s in scope.evidence_sources if s.kind == 'boundary_context')
    response = reply(scope)
    response['c0'][1] = [context.source_id]
    assert parse(scope, response).status == 'malformed_assessment'
    response['c0'][1] = ['s0', context.source_id]
    assert parse(scope, response).status == 'valid_assessment'
    scope = prepare().requests[-1]
    prior = next(s for s in scope.evidence_sources if s.kind == 'prior_summary')
    response = reply(scope)
    response['c0'][1] = [prior.source_id]
    assert parse(scope, response).status == 'valid_assessment'
    retention = next(c for c in scope.checks if c.kind == 'retention')
    response[retention.check_id][1] = [prior.source_id]
    assert parse(scope, response).status == 'malformed_assessment'


def test_root_retention_cannot_borrow_another_primary_record():
    scope = prepare().requests[-1]
    first, second, *_ = [c for c in scope.checks if c.kind == 'retention']
    response = reply(scope)
    response[first.check_id][1] = [second.source_id]
    assert parse(scope, response).status == 'malformed_assessment'


def test_root_wrong_semantics_with_authorized_ids_is_not_proven_true():
    value = packet()
    value['items'][0]['candidate_body'] = 'Rhea booked the ferry because the trip was completed.'
    scope = prepare(value).requests[0]
    result = parse(scope)
    assert result.status == 'valid_assessment' and result.model_all_supported
    assert not result.semantic_verified and not result.publication_authorized
    with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
        result.semantic_verified = True
    # Explicit relational dissent must remain visible, not hidden by other supported rows.
    response = reply(scope)
    body = next(f for f in scope.fields if f.path == '/candidate_body')
    relation = next(c for c in scope.checks if c.kind == 'relations' and c.field_ids == (body.field_id,))
    response[relation.check_id] = ['unsupported', ['s0']]
    assert not parse(scope, response).model_all_supported


def test_root_empty_summary_has_no_vacuous_success_but_new_sources_need_retention():
    value = packet()
    value.update(items=[], procedure_items=[], source_catalog=[])
    value['summary_item'].update(candidate_raw_summary='', candidate_summary='', candidate_is_noop=True,
                                 prior_derived_summary='', new_source_ids=[])
    scope = prepare(value).requests[0]
    assert scope.checks == () and scope.fields == ()
    result = parse(scope, {})
    assert result.status == 'unassessed' and not result.model_all_supported
    value = packet()
    value.update(items=[], procedure_items=[])
    value['summary_item'].update(candidate_summary='', prior_derived_summary='', candidate_is_noop=True)
    scope = prepare(value).requests[0]
    assert scope.fields == () and len(scope.checks) == 3
    assert all(c.kind == 'retention' for c in scope.checks)


@pytest.mark.parametrize('damage', [
    lambda s: replace(s, index=True),
    lambda s: replace(s, binding_sha256='0' * 64),
    lambda s: replace(s, fields=(replace(s.fields[0], text='changed'), *s.fields[1:])),
    lambda s: replace(s, fields=(replace(s.fields[0], end=s.fields[0].end + 1), *s.fields[1:])),
    lambda s: replace(s, checks=s.checks[:-1]),
    lambda s: replace(s, checks=tuple(reversed(s.checks))),
    lambda s: replace(s, evidence_sources=(replace(s.evidence_sources[0], start=999), *s.evidence_sources[1:])),
    lambda s: replace(s, request=replace(s.request, user='{}')),
    lambda s: replace(s, request=replace(s.request, system='forged')),
])
def test_root_scope_tampering_is_caller_error(damage):
    scope = prepare().requests[0]
    with pytest.raises(ValueError):
        parse(damage(scope), reply(scope))


def test_root_rehashed_derived_data_does_not_replace_original_projection():
    scope = prepare().requests[0]
    broken = replace(scope, checks=scope.checks[:-1])
    # Match the common body contract independently, excluding only its digest.
    body = {'version': assessment.VERSION, **asdict(broken)}
    del body['binding_sha256']
    import hashlib
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    broken = replace(broken, binding_sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        parse(broken, {k: v for k, v in reply(scope).items() if k != scope.checks[-1].check_id})


def test_root_all_scopes_preflight_before_any_call():
    plan = prepare()
    invalid_last = replace(plan.requests[-1], checks=())
    client = Scripted([])
    with pytest.raises(ValueError):
        assessment.execute_evidence_assessment(replace(plan, requests=(*plan.requests[:-1], invalid_last)), client)
    assert client.calls == []


def test_root_impossible_complete_output_cap_rejected_before_any_call():
    with pytest.raises(ValueError):
        prepare(max_output_chars=1)


def test_root_malformed_output_continues_without_repair_or_extra_calls():
    plan = prepare()
    client = Scripted(['{'] + [json.dumps(reply(s)) for s in plan.requests[1:]])
    result = assessment.execute_evidence_assessment(plan, client)
    assert result.complete and result.attempted_calls == len(client.calls) == 4
    assert [o.status for o in result.outcomes] == ['malformed_assessment'] + ['valid_assessment'] * 3
    assert not result.model_all_supported and not result.semantic_verified and not result.publication_authorized


def test_root_client_error_halts_and_deadline_or_interrupt_propagates():
    plan = prepare()
    client = Scripted([OSError('private client details must not surface')])
    result = assessment.execute_evidence_assessment(plan, client)
    assert not result.complete and result.attempted_calls == len(client.calls) == 1
    assert result.halted_reason == 'client_exception'
    assert 'private client' not in repr(result)
    for error in (DeadlineExceeded('stop'), KeyboardInterrupt(), SystemExit(9)):
        client = Scripted([error])
        with pytest.raises(type(error)):
            assessment.execute_evidence_assessment(plan, client)
        assert len(client.calls) == 1


@pytest.mark.parametrize('option', ['max_calls', 'max_checks', 'max_evidence_per_check', 'max_input_chars', 'max_output_chars'])
@pytest.mark.parametrize('value', [True, False, 0, -1, 1.5, '3'])
def test_root_invalid_caps_rejected_before_work(option, value):
    with pytest.raises(ValueError):
        prepare(**{option: value})
