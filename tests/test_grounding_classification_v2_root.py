"""Independent v2 contract checks. All verdicts are invented, never model evidence."""
from dataclasses import asdict, replace
import copy
import hashlib
import itertools
import json
from pathlib import Path

import jsonschema
import pytest

from hymem.extraction import grounding_classification_v1 as prior
from hymem.extraction import grounding_classification_v2 as g
from hymem.extraction import grounding_v2 as evidence_contract
from tools.diagnostics import luna_semantic_cases as controls


def sources(case):
    result = []
    for source in case.sources:
        data = asdict(source)
        data['contexts'] = tuple(g.GroundingContext(**c) for c in data['contexts'])
        result.append(g.GroundingSource(**data))
    return tuple(result)


def row(index, positive=(), evidence=(), ambiguous=()):
    states = ['not_established'] * 22
    for p in ambiguous:
        states[g.PREDICATE_ORDER.index(p)] = 'ambiguous'
    for p in positive:
        states[g.PREDICATE_ORDER.index(p)] = 'supported'
    return dict(index=index, states=states, support_groups=(
        [dict(predicates=list(positive), evidence=copy.deepcopy(list(evidence)))] if positive else []))


def response(batch, rows):
    return dict(schema=g.GROUNDING_CONTRACT_VERSION, batch_sha256=batch.batch_sha256,
                complete=True, classifications=rows)


def parse(raw, batch, **kwargs):
    return g.parse_grounding_response(json.dumps(raw), batch, **kwargs)


def first():
    case = controls.cases()[0]
    request, batch = g.build_grounding_request(case.triples, sources(case))
    quote = dict(source_message_id=101, region='owned', quote=batch.sources[0].content)
    return request, batch, response(batch, [row(0, ['uses'], [quote])])


def test_old_contracts_and_fixed_labels_remain_immutable():
    assert hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest() == (
        '7c62c6e58305a5b7be256825a52a8cc4b9119888011bf18fe267b18dccab2c06')
    assert hashlib.sha256(Path(evidence_contract.__file__).read_bytes()).hexdigest() == (
        '377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec')
    assert controls.suite_sha256() == '511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925'


@pytest.mark.parametrize('index', range(24))
def test_fixed_cases_mechanically_supported_without_changing_labels(index):
    case = controls.cases()[index]
    request, batch = g.build_grounding_request(case.triples, sources(case))
    old_request, old_batch = prior.build_grounding_request(case.triples, sources(case))
    assert json.loads(request.user)['batch']['sources'] == json.loads(old_request.user)['batch']['sources']
    assert batch.triples == case.triples and batch.batch_sha256 != old_batch.batch_sha256
    assert [x['predicate'] for x in json.loads(batch.canonical_json)['candidates']] == [t.predicate for t in case.triples]
    assert all('predicate' not in c for c in json.loads(request.user)['batch']['candidates'])
    assert (request.max_tokens, request.temperature, request.response_format) == (4096, 0.0, 'json')
    assert all(e.rationale not in request.user and e.rationale not in request.system for e in case.expected)
    rows = []
    for i, (triple, expected) in enumerate(zip(case.triples, case.expected, strict=True)):
        positive = [expected.predicate or triple.predicate] if case.category != 'reject' else []
        quotes = [dict(source_message_id=triple.source_message_id, region=r, quote=q)
                  for r, q in expected.evidence] if positive else []
        rows.append(row(i, positive, quotes))
    raw = response(batch, rows)
    jsonschema.Draft202012Validator(g.build_output_schema(batch)).validate(raw)
    review = parse(raw, batch)
    for actual, expected in zip(review.verdicts, case.expected):
        assert actual.status in expected.statuses
        if expected.predicate:
            assert actual.predicate == expected.predicate


@pytest.mark.parametrize('states', itertools.product(('supported', 'not_established', 'ambiguous'), repeat=3))
def test_selector_does_not_reinterpret_ambiguity(states):
    _, batch, raw = first()
    selected = ('uses', 'prefers', 'rejects')
    positive = [p for p, s in zip(selected, states) if s == 'supported']
    ambiguous = [p for p, s in zip(selected, states) if s == 'ambiguous']
    quotes = raw['classifications'][0]['support_groups'][0]['evidence']
    raw['classifications'] = [row(0, positive, quotes, ambiguous)]
    verdict = parse(raw, batch).verdicts[0]
    if states[0] == 'supported':
        assert (verdict.status, verdict.predicate) == ('supported', 'uses')
    elif states[0] == 'not_established' and len(positive) == 1 and not ambiguous:
        assert (verdict.status, verdict.predicate) == ('replace_predicate', positive[0])
    elif not positive and not ambiguous:
        assert verdict.status == 'unsupported'
    else:
        assert verdict.status == 'uncertain'


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(index=False),
    lambda r: r.update(extra='private-sentinel'),
    lambda r: r['states'].pop(),
    lambda r: r['states'].append('not_established'),
    lambda r: r['states'].__setitem__(0, 'u'),
    lambda r: r.update(states=['n'] * 22),
    lambda r: r.update(support_groups=[]),
    lambda r: r['support_groups'].append(copy.deepcopy(r['support_groups'][0])),
    lambda r: r['support_groups'][0]['predicates'].append('uses'),
    lambda r: r['support_groups'][0]['predicates'].append('prefers'),
    lambda r: r['support_groups'][0].update(predicates=[]),
    lambda r: r['support_groups'][0].update(predicates=[False]),
    lambda r: r['support_groups'][0].update(evidence=[]),
    lambda r: r['support_groups'][0]['evidence'].__imul__(2),
    lambda r: r['support_groups'][0]['evidence'][0].update(source_message_id=True),
    lambda r: r['support_groups'][0]['evidence'][0].update(quote='absent from the source'),
    lambda r: r['support_groups'][0]['evidence'][0].update(extra='private-sentinel'),
])
def test_rejects_bad_partition_evidence_and_old_states(mutation):
    _, batch, raw = first()
    mutation(raw['classifications'][0])
    with pytest.raises(g.GroundingContractError) as caught:
        parse(raw, batch)
    assert 'private-sentinel' not in str(caught.value)


@pytest.mark.parametrize('field, value', [('temperature', False), ('temperature', 0),
    ('temperature', 0.1), ('max_tokens', 8192), ('response_format', 'text'),
    ('system', 'foreign'), ('user', 'foreign')])
def test_typed_request_binding(field, value):
    request, batch, _ = first()
    with pytest.raises(g.GroundingContractError):
        g.validate_request(replace(request, **{field: value}), batch)


def test_v1_wire_and_batch_cannot_be_reinterpreted():
    request, batch, raw = first()
    old_request, old_batch = prior.build_grounding_request(batch.triples, batch.sources)
    with pytest.raises(g.GroundingContractError):
        g.validate_request(old_request, old_batch)
    raw['schema'] = prior.GROUNDING_CONTRACT_VERSION
    with pytest.raises(g.GroundingContractError):
        parse(raw, batch)
    different, changed = g.build_grounding_request((replace(batch.triples[0], predicate='prefers'),), batch.sources)
    left, right = json.loads(request.user), json.loads(different.user)
    assert left.pop('batch_sha256') != right.pop('batch_sha256') and left == right
    with pytest.raises(g.GroundingContractError):
        g.validate_request(request, changed)


def test_cross_group_quote_overlap_is_valid_but_ninth_distinct_quote_is_not():
    texts = [f'quote{i}' for i in range(9)]
    source = g.GroundingSource(7, ' '.join(texts))
    triple = g.Triple('Mira', 'uses', 'CairnDB', 1, source_message_id=7)
    _, batch = g.build_grounding_request((triple,), (source,))
    quotes = [dict(source_message_id=7, region='owned', quote=q) for q in texts]
    item = row(0, ['uses', 'prefers'], quotes[:8])
    item['support_groups'] = [dict(predicates=['uses'], evidence=quotes[:8]),
                              dict(predicates=['prefers'], evidence=quotes[:1])]
    assert parse(response(batch, [item]), batch).all_supported
    item['support_groups'][1]['evidence'] = quotes[8:]
    with pytest.raises(g.GroundingContractError):
        parse(response(batch, [item]), batch)


def test_invalid_unselected_positive_alternative_cannot_hide_behind_valid_original():
    source = g.GroundingSource(7, 'First clause. Later clause.', (
        g.GroundingContext('boundary', 'Mira and CairnDB', len('First clause.')),))
    triple = g.Triple('Mira', 'prefers', 'CairnDB', 1, source_message_id=7)
    _, batch = g.build_grounding_request((triple,), (source,))
    def quote(region, text): return dict(source_message_id=7, region=region, quote=text)
    item = row(0, ['prefers', 'uses'])
    item['support_groups'] = [dict(predicates=['prefers'], evidence=[quote('owned', 'First clause.')]),
        dict(predicates=['uses'], evidence=[quote('owned', 'Later clause.'), quote('boundary', 'Mira and CairnDB')])]
    with pytest.raises(g.GroundingContractError, match='context_scope'):
        parse(response(batch, [item]), batch)


def test_schema_is_fresh_and_does_not_replace_quote_validation():
    _, batch, raw = first()
    schema = g.build_output_schema(batch)
    schema['properties']['batch_sha256']['enum'][0] = 'foreign'
    assert g.build_output_schema(batch)['properties']['batch_sha256']['enum'] == [batch.batch_sha256]
    raw['classifications'][0]['support_groups'][0]['evidence'][0]['quote'] = 'invented quote'
    jsonschema.Draft202012Validator(g.build_output_schema(batch)).validate(raw)
    with pytest.raises(g.GroundingContractError):
        parse(raw, batch)


@pytest.mark.parametrize('change', ['missing_parent', 'parent_outside_prefix', 'missing_owned'])
def test_nested_context_guards_survive_evidence_grouping(change):
    case = controls.cases()[11]
    _, batch = g.build_grounding_request(case.triples, sources(case))
    quotes = [dict(source_message_id=112, region=r, quote=q) for r, q in case.expected[0].evidence]
    if change == 'missing_parent':
        quotes = [q for q in quotes if q['region'] != 'conversation_0']
    elif change == 'parent_outside_prefix':
        next(q for q in quotes if q['region'] == 'conversation_0')['quote'] = 'Later we discussed an unrelated router.'
    else:
        quotes = [q for q in quotes if q['region'] != 'owned']
    with pytest.raises(g.GroundingContractError):
        parse(response(batch, [row(0, ['prefers'], quotes)]), batch)


def test_schema_uses_only_prior_keyword_subset():
    _, batch, _ = first()
    schema = g.build_output_schema(batch)
    permitted = {'type', 'properties', 'required', 'additionalProperties', 'enum',
                 'anyOf', 'items', 'minItems', 'maxItems', 'minimum', 'maximum',
                 'minLength', 'maxLength', '$schema'}
    def visit(node):
        assert set(node) <= permitted
        if node.get('type') == 'object':
            assert node['additionalProperties'] is False
            assert set(node['required']) == set(node['properties'])
        for child in node.get('properties', {}).values():
            visit(child)
        if 'items' in node:
            visit(node['items'])
        for child in node.get('anyOf', []):
            visit(child)
    visit(schema)
    jsonschema.Draft202012Validator.check_schema(schema)


def test_duplicate_json_depth_and_old_uncertainty_remain_fail_closed():
    _, batch, raw = first()
    encoded = json.dumps(raw)
    for text in [encoded.replace('"complete": true', '"complete": true,"complete": true'),
                 '[' * 2000 + '0' + ']' * 2000]:
        with pytest.raises(g.GroundingContractError):
            g.parse_grounding_response(text, batch)
    raw['classifications'][0] = row(0, ['prefers'], raw['classifications'][0]['support_groups'][0]['evidence'],
                                      [p for p in g.PREDICATE_ORDER if p != 'prefers'])
    assert parse(raw, batch).verdicts[0].status == 'uncertain'


def test_recheck_disallows_correction_and_all_ambiguous_stays_uncertain():
    _, batch, raw = first()
    quotes = raw['classifications'][0]['support_groups'][0]['evidence']
    raw['classifications'] = [row(0, ['prefers'], quotes)]
    assert parse(raw, batch).verdicts[0].status == 'replace_predicate'
    with pytest.raises(g.GroundingContractError):
        parse(raw, batch, allow_corrections=False)
    raw['classifications'] = [row(0, (), (), g.PREDICATE_ORDER)]
    assert parse(raw, batch).verdicts[0].status == 'uncertain'


def test_shared_group_size_and_pathological_repetition_respect_character_limit():
    quotes = [str(i) + 'q' * 191 for i in range(8)]
    source = g.GroundingSource(7, '\n'.join(quotes))
    triples = tuple(g.Triple('Mira', 'uses', f'tool{i}', 1, source_message_id=7) for i in range(8))
    _, batch = g.build_grounding_request(triples, (source,))
    evidence = [dict(source_message_id=7, region='owned', quote=q) for q in quotes]
    raw = response(batch, [row(i, g.PREDICATE_ORDER, evidence) for i in range(8)])
    assert len(json.dumps(raw)) < g.MAX_RESPONSE_CHARS
    assert parse(raw, batch).all_supported
    for item in raw['classifications']:
        item['support_groups'] = [dict(predicates=[p], evidence=evidence) for p in g.PREDICATE_ORDER]
    assert len(json.dumps(raw)) > g.MAX_RESPONSE_CHARS
    with pytest.raises(g.GroundingContractError, match='response:bounds'):
        parse(raw, batch)
    # These checks deliberately do not claim 4096-token fit or semantic accuracy.
