"""Independent adversarial gates for the source-grounding contract."""
from dataclasses import replace
import json

import pytest

from hymem.extraction import grounding as g
from hymem.extraction.triples import Triple


def fixture():
    source = g.GroundingSource(7, 'Mira favors CairnDB over PineDB.', source_role='user')
    claim = Triple('Mira', 'prefers', 'CairnDB', 1, source_message_id=7)
    _, batch = g.build_grounding_request((claim,), (source,))
    payload = {'schema': g.GROUNDING_CONTRACT_VERSION, 'batch_sha256': batch.batch_sha256,
               'complete': True, 'verdicts': [{'index': 0, 'status': 'supported',
                'predicate': 'prefers', 'evidence': [{'source_message_id': 7,
                'region': 'owned', 'quote': source.content}]}]}
    return claim, source, batch, payload


def rejected(payload, batch):
    with pytest.raises(g.GroundingContractError) as error:
        g.parse_grounding_response(json.dumps(payload), batch)
    assert 'private-synthetic-secret' not in str(error.value)


@pytest.mark.parametrize('field', ['schema', 'batch_sha256', 'complete', 'verdicts'])
def test_required_top_level_fields(field):
    _, _, batch, payload = fixture()
    del payload[field]
    rejected(payload, batch)


@pytest.mark.parametrize('value', [True, False, [], {}, 7.0, '7'])
def test_claim_source_identity_is_strict(value):
    claim, source, _, _ = fixture()
    with pytest.raises(g.GroundingContractError):
        g.build_grounding_request((replace(claim, source_message_id=value),), (source,))


@pytest.mark.parametrize('value', [True, False, [], {}, 7.0, '7'])
def test_source_identity_is_strict(value):
    claim, source, _, _ = fixture()
    with pytest.raises(g.GroundingContractError):
        g.build_grounding_request((claim,), (replace(source, source_message_id=value),))


@pytest.mark.parametrize('field', ['index', 'status', 'predicate', 'evidence'])
@pytest.mark.parametrize('value', [None, [], {}, True, 0.5, 'private-synthetic-secret'])
def test_bad_verdict_field_shapes_have_finite_errors(field, value):
    _, _, batch, payload = fixture()
    payload['verdicts'][0][field] = value
    rejected(payload, batch)


def test_boundary_context_is_distinct_from_owned_evidence():
    content = 'CairnDB is that database.'
    source = g.GroundingSource(7, content,
        (g.GroundingContext('boundary', "Mira's preferred database follows.", len(content)),))
    claim = Triple('Mira', 'prefers', 'CairnDB', 1, source_message_id=7)
    _, batch = g.build_grounding_request((claim,), (source,))
    payload = fixture()[3]
    payload['batch_sha256'] = batch.batch_sha256
    payload['verdicts'][0]['evidence'] = [
        {'source_message_id': 7, 'region': 'owned', 'quote': source.content},
        {'source_message_id': 7, 'region': 'boundary', 'quote': source.contexts[0].content}]
    assert g.parse_grounding_response(json.dumps(payload), batch).all_supported
    payload['verdicts'][0]['evidence'].pop(0)
    rejected(payload, batch)


def test_context_cannot_be_laundered_with_one_in_range_owned_quote():
    content = 'Acknowledged. Later Mira chooses CairnDB.'
    source = g.GroundingSource(7, content,
        (g.GroundingContext('boundary', 'An unrelated prior statement.', 13),))
    claim = fixture()[0]
    _, batch = g.build_grounding_request((claim,), (source,))
    payload = fixture()[3]
    payload['batch_sha256'] = batch.batch_sha256
    payload['verdicts'][0]['evidence'] = [
        {'source_message_id': 7, 'region': 'owned', 'quote': 'Acknowledged.'},
        {'source_message_id': 7, 'region': 'owned', 'quote': 'Mira chooses CairnDB.'},
        {'source_message_id': 7, 'region': 'boundary', 'quote': source.contexts[0].content}]
    rejected(payload, batch)


def test_exact_quotes_not_casefolded_or_fuzzy():
    _, _, batch, payload = fixture()
    payload['verdicts'][0]['evidence'][0]['quote'] = 'mira favors cairndb over pinedb.'
    rejected(payload, batch)


def test_schema_and_source_mutation_cannot_reuse_binding():
    claim, source, batch, payload = fixture()
    other = replace(source, source_role='assistant')
    _, rebound = g.build_grounding_request((claim,), (other,))
    assert rebound.batch_sha256 != batch.batch_sha256
    rejected(payload, rebound)
    with pytest.raises(g.GroundingContractError):
        g.parse_grounding_response(json.dumps(payload), replace(batch, sources=(other,)))


def test_unsupported_and_uncertain_are_not_passes():
    _, _, batch, payload = fixture()
    for status in ('unsupported', 'uncertain'):
        payload['verdicts'][0].update(status=status, predicate=None, evidence=[])
        review = g.parse_grounding_response(json.dumps(payload), batch)
        assert review.all_supported is False


def test_correction_is_not_approval_and_cannot_recur():
    claim, source, _, _ = fixture()
    wrong = replace(claim, predicate='uses')
    _, batch = g.build_grounding_request((wrong,), (source,))
    payload = fixture()[3]
    payload['batch_sha256'] = batch.batch_sha256
    payload['verdicts'][0]['status'] = 'replace_predicate'
    review = g.parse_grounding_response(json.dumps(payload), batch)
    assert not review.all_supported
    assert review.verdicts[0].predicate == 'prefers'
    assert batch.triples == (wrong,)
    with pytest.raises(g.GroundingContractError):
        g.parse_grounding_response(json.dumps(payload), batch, allow_corrections=False)


def test_lone_surrogate_and_deep_json_have_safe_errors():
    claim, source, batch, _ = fixture()
    for field in ('subject', 'object', 'value_text'):
        with pytest.raises(g.GroundingContractError):
            g.build_grounding_request((replace(claim, **{field: '\ud800'}),), (source,))
    with pytest.raises(g.GroundingContractError):
        g.parse_grounding_response('[' * 2000 + '0' + ']' * 2000, batch)


def test_no_benchmark_literals_in_request_template():
    request, _ = g.build_grounding_request((fixture()[0],), (fixture()[1],))
    assert not any(value in request.system for value in
                   ('Canary', 'Avery', 'PostgreSQL', 'Fly.io', 'longmemeval'))


@pytest.mark.parametrize('number', [10 ** 1000, -(10 ** 1000), float('nan'), float('inf'), True])
def test_unrepresentable_numbers_have_finite_contract_errors(number):
    claim, source, _, _ = fixture()
    with pytest.raises(g.GroundingContractError, match='^triple:numeric$'):
        g.build_grounding_request((replace(claim, value_numeric=number),), (source,))


def nested_fixture():
    claim, _, _, payload = fixture()
    owned = 'Confirmed: that is my preference.'
    body = '| Mira | CairnDB |\nLater PineDB is merely a suggestion.'
    row = '| Mira | CairnDB |'
    metadata = dict(source_role='assistant', source_peer_id='helper', source_message_id=6)
    source = g.GroundingSource(7, owned, (
        g.GroundingContext('conversation_0', body, len(owned), **metadata),
        g.GroundingContext('conversation_0_header', '| Person | Preferred database |',
            len(owned), applies_to_region='conversation_0',
            applies_to_prefix_chars=len(row), **metadata)), source_role='user')
    request, batch = g.build_grounding_request((claim,), (source,))
    payload['batch_sha256'] = batch.batch_sha256
    payload['verdicts'][0]['evidence'] = [
        dict(source_message_id=7, region='owned', quote=owned),
        dict(source_message_id=7, region='conversation_0', quote=row),
        dict(source_message_id=7, region='conversation_0_header', quote='Preferred database')]
    return source, request, batch, payload


def test_nested_scope_preserves_complete_context_and_binds_parent_range():
    source, request, batch, payload = nested_fixture()
    contexts = json.loads(request.user)['batch']['sources'][0]['contexts']
    assert contexts[0]['content'] == source.contexts[0].content
    assert contexts[1]['applies_to_region'] == 'conversation_0'
    assert contexts[1]['applies_to_prefix_chars'] == len('| Mira | CairnDB |')
    assert g.parse_grounding_response(json.dumps(payload), batch).all_supported
    changed = replace(source, contexts=(source.contexts[0], replace(source.contexts[1],
        applies_to_prefix_chars=len(source.contexts[0].content))))
    assert g.build_grounding_request(batch.triples, (changed,))[1].batch_sha256 != batch.batch_sha256


@pytest.mark.parametrize('change', ['later_only', 'row_plus_later', 'missing_parent'])
def test_nested_header_cannot_support_unrelated_later_prose(change):
    _, _, batch, payload = nested_fixture()
    evidence = payload['verdicts'][0]['evidence']
    later = dict(source_message_id=7, region='conversation_0',
                 quote='Later PineDB is merely a suggestion.')
    if change == 'later_only':
        evidence[1] = later
    elif change == 'row_plus_later':
        evidence.append(later)
    else:
        evidence.pop(1)
    rejected(payload, batch)


@pytest.mark.parametrize('change', ['wrong_parent', 'missing_parent', 'wrong_id', 'wrong_role',
                                    'boolean_prefix', 'too_long', 'zero_prefix', 'half_pair'])
def test_nested_dependency_input_fails_closed(change):
    source, _, batch, _ = nested_fixture()
    parent, nested = source.contexts
    if change == 'wrong_parent':
        nested = replace(nested, applies_to_region='owned')
    elif change == 'missing_parent':
        parent = replace(parent, region='conversation_1')
    elif change == 'wrong_id':
        nested = replace(nested, source_message_id=99)
    elif change == 'wrong_role':
        nested = replace(nested, source_role='user')
    elif change == 'boolean_prefix':
        nested = replace(nested, applies_to_prefix_chars=True)
    elif change == 'too_long':
        nested = replace(nested, applies_to_prefix_chars=len(parent.content)+1)
    elif change == 'zero_prefix':
        nested = replace(nested, applies_to_prefix_chars=0)
    else:
        nested = replace(nested, applies_to_prefix_chars=None)
    with pytest.raises(g.GroundingContractError):
        g.build_grounding_request(batch.triples, (replace(source, contexts=(parent, nested)),))


def test_two_nested_records_have_representable_and_independent_witnesses():
    source, _, original_batch, payload = nested_fixture()
    first, header = source.contexts
    second = replace(first, region='conversation_1', source_message_id=5,
                     content='| CairnDB | active |\nUnrelated later comment.')
    second_header = replace(header, region='conversation_1_header', source_message_id=5,
        content='| Database | Status |', applies_to_region='conversation_1',
        applies_to_prefix_chars=len('| CairnDB | active |'))
    source = replace(source, contexts=(*source.contexts, second, second_header))
    _, batch = g.build_grounding_request(original_batch.triples, (source,))
    payload['batch_sha256'] = batch.batch_sha256
    evidence = payload['verdicts'][0]['evidence']
    evidence.extend([
        dict(source_message_id=7, region='conversation_1', quote='| CairnDB | active |'),
        dict(source_message_id=7, region='conversation_1_header', quote='Status')])
    assert len(evidence) == 5
    assert g.parse_grounding_response(json.dumps(payload), batch).all_supported
    evidence[3]['quote'] = 'Unrelated later comment.'
    rejected(payload, batch)
