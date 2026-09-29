"""Independent offline checks; invented verdicts are not semantic accuracy."""
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path

import pytest
import jsonschema

from hymem.extraction import grounding_v2 as old
from hymem.extraction import grounding_classification_v1 as g
from tools.diagnostics import luna_semantic_cases as controls


def sources(case):
    result = []
    for source in case.sources:
        data = asdict(source)
        data['contexts'] = tuple(g.GroundingContext(**c) for c in data['contexts'])
        result.append(g.GroundingSource(**data))
    return tuple(result)


def row(index, positive, evidence, uncertain=()):
    states = ['n'] * len(g.PREDICATE_ORDER)
    citations = [[] for _ in states]
    for predicate in uncertain:
        states[g.PREDICATE_ORDER.index(predicate)] = 'u'
    for predicate in positive:
        position = g.PREDICATE_ORDER.index(predicate)
        states[position] = 'e'
        citations[position] = list(range(len(evidence)))
    return dict(index=index, states=states, evidence_pool=evidence, citations=citations)


def response(batch, rows):
    return dict(schema=g.GROUNDING_CONTRACT_VERSION,
                batch_sha256=batch.batch_sha256, complete=True, classifications=rows)


def case0():
    case = controls.cases()[0]
    request, batch = g.build_grounding_request(case.triples, sources(case))
    evidence = [dict(source_message_id=101, region='owned', quote=batch.sources[0].content)]
    return request, batch, response(batch, [row(0, ['uses'], evidence)])


def parse(value, batch, **kwargs):
    return g.parse_grounding_response(json.dumps(value), batch, **kwargs)


def test_pinned_previous_sources_and_fixed_labels_unchanged():
    assert hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() == (
        '377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec')
    assert controls.suite_sha256() == (
        '511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925')
    assert len(g.PREDICATE_ORDER) == 22
    assert tuple(sorted(old.ALLOWED_PREDICATES)) == g.PREDICATE_ORDER


@pytest.mark.parametrize('index', range(24))
def test_fixed_control_mechanics_without_relabeling_or_leaking_oracle(index):
    case = controls.cases()[index]
    rebound = sources(case)
    request, batch = g.build_grounding_request(case.triples, rebound)
    old_request, _ = old.build_grounding_request(case.triples, rebound)
    original = json.loads(old_request.user)['batch']
    wire = json.loads(request.user)['batch']
    assert wire['sources'] == original['sources']
    assert wire['candidates'] == [{k: v for k, v in c.items() if k != 'predicate'}
                                  for c in original['candidates']]
    assert batch.triples == case.triples
    assert batch.sources == rebound
    assert [x['predicate'] for x in json.loads(batch.canonical_json)['candidates']] == [
        t.predicate for t in case.triples]
    assert (request.max_tokens, request.temperature, request.response_format) == (4096, 0.0, 'json')
    assert all(x.rationale not in request.user and x.rationale not in request.system for x in case.expected)
    rows = []
    for i, (triple, expected) in enumerate(zip(case.triples, case.expected, strict=True)):
        predicate = expected.predicate or triple.predicate
        positive = [predicate] if case.category != 'reject' else []
        pool = [dict(source_message_id=triple.source_message_id, region=region, quote=quote)
                for region, quote in expected.evidence] if positive else []
        rows.append(row(i, positive, pool))
    raw = response(batch, rows)
    jsonschema.Draft202012Validator(g.build_output_schema(batch)).validate(raw)
    result = parse(raw, batch)
    for verdict, expected in zip(result.verdicts, case.expected, strict=True):
        assert verdict.status in expected.statuses
        if expected.predicate:
            assert verdict.predicate == expected.predicate


@pytest.mark.parametrize('values', list(itertools.product('enu', repeat=3)))
def test_exhaustive_selector_original_and_two_alternatives(values):
    _, batch, raw = case0()
    selected = ('uses', 'prefers', 'rejects')
    positives = [p for p, v in zip(selected, values) if v == 'e']
    uncertain = [p for p, v in zip(selected, values) if v == 'u']
    pool = raw['classifications'][0]['evidence_pool'] if positives else []
    raw['classifications'] = [row(0, positives, pool, uncertain)]
    verdict = parse(raw, batch).verdicts[0]
    if values[0] == 'e':
        assert (verdict.status, verdict.predicate) == ('supported', 'uses')
    elif values[0] == 'n' and len(positives) == 1 and not uncertain:
        assert (verdict.status, verdict.predicate) == ('replace_predicate', positives[0])
    elif not positives and not uncertain:
        assert verdict.status == 'unsupported'
    else:
        assert verdict.status == 'uncertain'


@pytest.mark.parametrize('field,value', [('temperature', False), ('temperature', 0),
    ('temperature', 0.1), ('max_tokens', 8192), ('response_format', 'text'),
    ('system', 'foreign system'), ('user', 'foreign user')])
def test_exact_request_binding_includes_all_typed_fields(field, value):
    request, batch, _ = case0()
    g.validate_request(request, batch)
    with pytest.raises(g.GroundingContractError):
        g.validate_request(replace(request, **{field: value}), batch)


def test_original_predicate_affects_binding_but_not_classification_payload():
    request, batch, raw = case0()
    other_request, other = g.build_grounding_request(
        (replace(batch.triples[0], predicate='prefers'),), batch.sources)
    assert other.batch_sha256 != batch.batch_sha256
    left, right = json.loads(request.user), json.loads(other_request.user)
    left.pop('batch_sha256')
    right.pop('batch_sha256')
    assert left == right
    with pytest.raises(g.GroundingContractError):
        parse(raw, other)
    with pytest.raises(g.GroundingContractError):
        g.validate_request(request, other)
    with pytest.raises(g.GroundingContractError):
        parse(raw, replace(batch, triples=other.triples))


@pytest.mark.parametrize('field,value', [('schema', 'source-grounding-v2'),
    ('batch_sha256', '0'*64), ('complete', 1), ('classifications', [])])
def test_bad_root_is_not_laundered(field, value):
    _, batch, raw = case0()
    raw[field] = value
    with pytest.raises(g.GroundingContractError):
        parse(raw, batch)


@pytest.mark.parametrize('change', ['boolean_index', 'extra', 'short_states',
    'long_states', 'bad_state', 'tuple_states', 'short_citations', 'missing_positive',
    'negative_citation', 'boolean_reference', 'float_reference', 'negative_reference',
    'high_reference', 'duplicate_reference', 'duplicate_pool', 'unused_pool',
    'missing_owned', 'wrong_source', 'invented_quote', 'bad_pool_field'])
def test_malformed_classification_cannot_become_approval(change):
    _, batch, raw = case0()
    r = raw['classifications'][0]
    positive = g.PREDICATE_ORDER.index('uses')
    negative = g.PREDICATE_ORDER.index('prefers')
    if change == 'boolean_index': r['index'] = False
    elif change == 'extra': r['status'] = 'supported'
    elif change == 'short_states': r['states'].pop()
    elif change == 'long_states': r['states'].append('n')
    elif change == 'bad_state': r['states'][0] = True
    elif change == 'tuple_states': r['states'] = 'n'*22
    elif change == 'short_citations': r['citations'].pop()
    elif change == 'missing_positive': r['citations'][positive] = []
    elif change == 'negative_citation': r['citations'][negative] = [0]
    elif change == 'boolean_reference': r['citations'][positive] = [False]
    elif change == 'float_reference': r['citations'][positive] = [0.0]
    elif change == 'negative_reference': r['citations'][positive] = [-1]
    elif change == 'high_reference': r['citations'][positive] = [1]
    elif change == 'duplicate_reference': r['citations'][positive] = [0, 0]
    elif change == 'duplicate_pool': r['evidence_pool'] *= 2
    elif change == 'unused_pool': r['evidence_pool'].append(dict(source_message_id=101, region='owned', quote='QuillDB'))
    elif change == 'missing_owned': r['evidence_pool'][0]['region'] = 'boundary'
    elif change == 'wrong_source': r['evidence_pool'][0]['source_message_id'] = 102
    elif change == 'invented_quote': r['evidence_pool'][0]['quote'] = 'unrelated'
    elif change == 'bad_pool_field': r['evidence_pool'][0]['private'] = 'secret-sentinel'
    with pytest.raises(g.GroundingContractError) as exc:
        parse(raw, batch)
    assert 'secret-sentinel' not in str(exc.value)


def test_schema_copy_cannot_mutate_binding_and_still_parser_checks_semantics():
    _, batch, raw = case0()
    schema = g.build_output_schema(batch)
    schema['properties']['batch_sha256']['enum'][0] = 'foreign'
    assert g.build_output_schema(batch)['properties']['batch_sha256']['enum'] == [batch.batch_sha256]
    raw['classifications'][0]['evidence_pool'][0]['quote'] = 'absent from source'
    # JSON shape compliance is not evidence compliance.
    jsonschema.Draft202012Validator(g.build_output_schema(batch)).validate(raw)
    with pytest.raises(g.GroundingContractError):
        parse(raw, batch)


def test_scope_checks_apply_even_to_entailed_alternative_not_selected():
    source = g.GroundingSource(7, 'First clause. Later clause.', (
        g.GroundingContext('boundary', 'Mira and CairnDB', len('First clause.')),))
    triple = g.Triple('Mira', 'prefers', 'CairnDB', 1, source_message_id=7)
    _, batch = g.build_grounding_request((triple,), (source,))
    pool = [dict(source_message_id=7, region='owned', quote='First clause.'),
            dict(source_message_id=7, region='owned', quote='Later clause.'),
            dict(source_message_id=7, region='boundary', quote='Mira and CairnDB')]
    r = row(0, ['prefers', 'uses'], pool)
    r['citations'][g.PREDICATE_ORDER.index('prefers')] = [0]
    r['citations'][g.PREDICATE_ORDER.index('uses')] = [1, 2]
    with pytest.raises(g.GroundingContractError, match='context_scope'):
        parse(response(batch, [r]), batch)


def test_recheck_never_allows_further_correction():
    _, batch, raw = case0()
    raw['classifications'][0] = row(0, ['prefers'], raw['classifications'][0]['evidence_pool'])
    assert parse(raw, batch).verdicts[0].status == 'replace_predicate'
    with pytest.raises(g.GroundingContractError):
        parse(raw, batch, allow_corrections=False)
    assert batch.triples[0].predicate == 'uses'


def test_oversize_duplicate_json_and_depth_remain_finite_rejections():
    _, batch, raw = case0()
    encoded = json.dumps(raw)
    bad = [encoded + ' ' * g.MAX_RESPONSE_CHARS,
           encoded.replace('"complete": true', '"complete": true,"complete": true'),
           '['*2000 + '0' + ']'*2000]
    for text in bad:
        with pytest.raises(g.GroundingContractError):
            g.parse_grounding_response(text, batch)


def test_source_policy_preserves_identity_implicit_claims_and_predicate_meanings():
    request, _, _ = case0()
    meanings = old._SYSTEM.split('Predicate meanings: ', 1)[1].split('\n\n', 1)[0]
    assert meanings in request.system
    for phrase in ('Role, peer and time metadata', 'clearly entailed implicit relationship',
                   'Do not infer an unsupported qualifier',
                   'unavailable for both reasoning and citation'):
        assert phrase in request.system
    assert g.GROUNDING_CONTRACT_VERSION in request.system


def test_schema_uses_documented_keyword_subset_and_parser_owns_order():
    case = controls.cases()[10]
    _, batch = g.build_grounding_request(case.triples, sources(case))
    schema = g.build_output_schema(batch)
    supported = {'type', 'properties', 'required', 'additionalProperties',
                 'enum', 'anyOf', 'items', 'minItems', 'maxItems',
                 'minimum', 'maximum', 'minLength', 'maxLength', '$schema'}
    def visit(node):
        assert type(node) is dict
        assert set(node) <= supported
        if node.get('type') == 'object':
            assert node['additionalProperties'] is False
            assert set(node['required']) == set(node['properties'])
        for child in node.get('properties', {}).values(): visit(child)
        if 'items' in node: visit(node['items'])
        for child in node.get('anyOf', []): visit(child)
    visit(schema)
    jsonschema.Draft202012Validator.check_schema(schema)


def test_shared_pool_caps_worst_legal_serialization_not_token_cost():
    # Worst pool size/citation count under the structural limits, using actual
    # owned quotes. This is synthetic and does not assert semantic entailment.
    quotes = [str(i) + 'q'*191 for i in range(8)]
    source = g.GroundingSource(7, '\n'.join(quotes))
    triples = tuple(g.Triple('Mira', 'uses', f'tool{i}', 1, source_message_id=7)
                    for i in range(8))
    _, batch = g.build_grounding_request(triples, (source,))
    pool = [dict(source_message_id=7, region='owned', quote=q) for q in quotes]
    raw = response(batch, [row(i, g.PREDICATE_ORDER, pool) for i in range(8)])
    wire = json.dumps(raw, separators=(',', ':'))
    assert len(wire) < g.MAX_RESPONSE_CHARS
    assert parse(raw, batch).all_supported
    # No tokenizer or model invoked; this does NOT establish 4096-token fit.
