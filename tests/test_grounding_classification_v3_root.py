"""Root-authored contract counterexamples. All judgments below are synthetic."""
import copy
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path

import jsonschema
import pytest

from hymem.extraction import grounding_classification_v2 as old
from hymem.extraction import grounding_classification_v3 as g
from tools.diagnostics import luna_semantic_cases as cases


def source_objects(case):
    result = []
    for source in case.sources:
        data = asdict(source)
        data['contexts'] = tuple(g.GroundingContext(**item) for item in data['contexts'])
        result.append(g.GroundingSource(**data))
    return tuple(result)


def quote(sid, text, region='owned'):
    return dict(source_message_id=sid, region=region, quote=text)


def assessment(triple, state, evidence=()):
    support = None
    if state == 'supported':
        components = ['attribution_and_roles', 'relation_and_polarity'] + [
            key for key in ('value_text', 'value_numeric', 'value_unit', 'temporal_scope')
            if getattr(triple, key) is not None]
        support = dict(evidence=copy.deepcopy(list(evidence)), checks={
            key: dict(state='supported', evidence_indices=list(range(len(evidence))))
            for key in components})
    return dict(state=state, support=support)


def item(triple, index=0, original='supported', evidence=(), positives=(), ambiguous=()):
    alternatives = None
    if original == 'not_established':
        alternatives = {
            predicate: assessment(triple, 'supported' if predicate in positives else
                'ambiguous' if predicate in ambiguous else 'not_established', evidence)
            for predicate in g.PREDICATE_ORDER if predicate != triple.predicate}
    return dict(index=index, original=assessment(triple, original, evidence), alternatives=alternatives)


def response(batch, rows):
    return dict(schema=g.GROUNDING_CONTRACT_VERSION, batch_sha256=batch.batch_sha256,
                complete=True, classifications=rows)


def parse(batch, rows, **kwargs):
    return g.parse_grounding_response(json.dumps(response(batch, rows)), batch, **kwargs)


def simple(**overrides):
    triple = g.Triple('Iris', 'uses', 'CedarTool', 1, source_message_id=7)
    triple = replace(triple, **overrides)
    content = 'Iris uses CedarTool. The limit is 64 items, for today only. Mode is quiet.'
    request, batch = g.build_grounding_request((triple,), (g.GroundingSource(7, content),))
    evidence = [quote(7, content)]
    return request, batch, item(triple, evidence=evidence)


def test_prior_sources_and_frozen_labels_unchanged():
    assert hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() == (
        '2520e4825df9e4d6f403a301451bb9701e052d9a331b5e31a9e106cbb69ad3c9')
    assert cases.suite_sha256() == '511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925'


@pytest.mark.parametrize('index', range(24))
def test_frozen_controls_mechanical_compatibility_not_model_accuracy(index):
    case = cases.cases()[index]
    request, batch = g.build_grounding_request(case.triples, source_objects(case))
    _, before = old.build_grounding_request(case.triples, source_objects(case))
    wire = json.loads(request.user)['batch']
    assert wire['sources'] == json.loads(before.canonical_json)['sources']
    assert [c['predicate'] for c in wire['candidates']] == [t.predicate for t in case.triples]
    assert batch.batch_sha256 != before.batch_sha256
    assert (request.max_tokens, request.temperature, request.response_format) == (4096, 0.0, 'json')
    rows = []
    for i, (triple, oracle) in enumerate(zip(case.triples, case.expected, strict=True)):
        assert oracle.rationale not in request.user and oracle.rationale not in request.system
        evidence = [quote(triple.source_message_id, q, r) for r, q in oracle.evidence]
        if case.category == 'reject':
            rows.append(item(triple, i, original='not_established'))
        elif oracle.predicate and oracle.predicate != triple.predicate:
            rows.append(item(triple, i, original='not_established', evidence=evidence,
                             positives=[oracle.predicate]))
        else:
            rows.append(item(triple, i, evidence=evidence))
    raw = response(batch, rows)
    jsonschema.Draft202012Validator(g.build_output_schema(batch)).validate(raw)
    reviewed = parse(batch, rows)
    for result, oracle in zip(reviewed.verdicts, case.expected, strict=True):
        assert result.status in oracle.statuses
        if oracle.predicate:
            assert result.predicate == oracle.predicate


@pytest.mark.parametrize('states', itertools.product(
    ('supported', 'not_established', 'ambiguous'), repeat=3))
def test_selector_truth_table(states):
    _, batch, row = simple()
    evidence = row['original']['support']['evidence']
    positives = [p for p, state in zip(('prefers', 'rejects'), states[1:]) if state == 'supported']
    uncertain = [p for p, state in zip(('prefers', 'rejects'), states[1:]) if state == 'ambiguous']
    row = item(batch.triples[0], original=states[0], evidence=evidence,
               positives=positives, ambiguous=uncertain)
    verdict = parse(batch, [row]).verdicts[0]
    if states[0] == 'supported':
        assert (verdict.status, verdict.predicate) == ('supported', 'uses')
    elif states[0] == 'ambiguous':
        assert verdict.status == 'uncertain'
    elif len(positives) == 1 and not uncertain:
        assert (verdict.status, verdict.predicate) == ('replace_predicate', positives[0])
        with pytest.raises(g.GroundingContractError):
            parse(batch, [row], allow_corrections=False)
    elif not positives and not uncertain:
        assert verdict.status == 'unsupported'
    else:
        assert verdict.status == 'uncertain'


@pytest.mark.parametrize('field,value', [('value_text', 'quiet'), ('value_numeric', 64.0),
    ('value_numeric', 0.0), ('value_unit', 'items'), ('temporal_scope', 'today')])
@pytest.mark.parametrize('mutation', ['missing', 'not_established', 'ambiguous', 'uncited'])
def test_predicate_support_cannot_override_missing_or_negative_qualifier_check(field, value, mutation):
    _, batch, row = simple(**{field: value})
    checks = row['original']['support']['checks']
    if mutation == 'missing':
        del checks[field]
    elif mutation == 'uncited':
        checks[field]['evidence_indices'] = []
    else:
        checks[field]['state'] = mutation
    with pytest.raises(g.GroundingContractError):
        parse(batch, [row])


@pytest.mark.parametrize('component', ['attribution_and_roles', 'relation_and_polarity'])
def test_joint_binding_is_required_even_when_all_qualifiers_attested(component):
    _, batch, row = simple(value_text='quiet', value_numeric=64.0,
                            value_unit='items', temporal_scope='today')
    row['original']['support']['checks'][component]['state'] = 'not_established'
    with pytest.raises(g.GroundingContractError):
        parse(batch, [row])


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(index=True),
    lambda r: r.update(extra='secret-sentinel'),
    lambda r: r.update(alternatives={}),
    lambda r: r['original'].update(state='u'),
    lambda r: r['original'].update(support=None),
    lambda r: r['original']['support']['checks'].update(value_unit={'state':'supported','evidence_indices':[0]}),
    lambda r: r['original']['support']['checks']['attribution_and_roles'].update(evidence_indices=[True]),
    lambda r: r['original']['support']['checks']['attribution_and_roles'].update(evidence_indices=[0,0]),
    lambda r: r['original']['support']['checks']['attribution_and_roles'].update(evidence_indices=[-1]),
    lambda r: r['original']['support']['checks']['attribution_and_roles'].update(evidence_indices=[1]),
    lambda r: r['original']['support']['evidence'][0].update(source_message_id=True),
    lambda r: r['original']['support']['evidence'][0].update(quote='secret-sentinel'),
    lambda r: r['original']['support']['evidence'].__imul__(2),
])
def test_strict_shape_and_private_error_codes(mutate):
    _, batch, row = simple()
    mutate(row)
    with pytest.raises(g.GroundingContractError) as caught:
        parse(batch, [row])
    assert 'secret-sentinel' not in str(caught.value)


@pytest.mark.parametrize('mutation', ['absent', 'missing', 'original', 'unknown', 'extra_support'])
def test_not_established_requires_exact_alternative_coverage(mutation):
    _, batch, _ = simple()
    row = item(batch.triples[0], original='not_established')
    if mutation == 'absent':
        row['alternatives'] = None
    elif mutation == 'missing':
        row['alternatives'].pop('prefers')
    elif mutation in ('original', 'unknown'):
        row['alternatives']['uses' if mutation == 'original' else 'invented'] = assessment(batch.triples[0], 'not_established')
    else:
        row['alternatives']['prefers']['support'] = dict(evidence=[], checks={})
    with pytest.raises(g.GroundingContractError):
        parse(batch, [row])


def test_unselected_positive_must_pass_quote_scope_and_global_quote_limit():
    text = ' '.join('word' + str(i) for i in range(9))
    triple = g.Triple('Iris', 'uses', 'CedarTool', 1, source_message_id=7)
    _, batch = g.build_grounding_request((triple,), (g.GroundingSource(7, text),))
    evidence = [quote(7, 'word' + str(i)) for i in range(8)]
    row = item(triple, original='not_established', positives=['prefers','rejects'], evidence=evidence)
    assert parse(batch, [row]).verdicts[0].status == 'uncertain'
    row['alternatives']['rejects'] = assessment(triple, 'supported', [quote(7, 'word8')])
    with pytest.raises(g.GroundingContractError):
        parse(batch, [row])


@pytest.mark.parametrize('mode', ['unowned', 'parent_absent', 'parent_outside'])
def test_nested_context_protections(mode):
    case = cases.cases()[11]
    _, batch = g.build_grounding_request(case.triples, source_objects(case))
    evidence = [quote(112, text, region) for region, text in case.expected[0].evidence]
    if mode == 'unowned':
        evidence = [q for q in evidence if q['region'] != 'owned']
    elif mode == 'parent_absent':
        evidence = [q for q in evidence if q['region'] != 'conversation_0']
    else:
        next(q for q in evidence if q['region'] == 'conversation_0')['quote'] = 'Later we discussed an unrelated router.'
    with pytest.raises(g.GroundingContractError):
        parse(batch, [item(batch.triples[0], evidence=evidence)])


@pytest.mark.parametrize('field,value', [('temperature', 0), ('temperature', False),
    ('temperature', 1.0), ('max_tokens',8192), ('user','different'), ('system','different'), ('response_format','text')])
def test_exact_typed_request(field,value):
    request,batch,_ = simple()
    with pytest.raises(g.GroundingContractError):
        g.validate_request(replace(request, **{field:value}),batch)


def test_old_batch_wire_tampered_batch_deep_json_and_duplicates_reject():
    request,batch,row = simple()
    old_request,old_batch = old.build_grounding_request(batch.triples,batch.sources)
    with pytest.raises(g.GroundingContractError):
        g.validate_request(old_request,old_batch)
    with pytest.raises(g.GroundingContractError):
        g.validate_request(request,replace(batch,canonical_json='{}'))
    raw = json.dumps(response(batch,[row]))
    for malformed in [raw.replace(g.GROUNDING_CONTRACT_VERSION,old.GROUNDING_CONTRACT_VERSION),
                      raw.replace('"complete": true','"complete": true,"complete": true'),
                      '[' * 2000 + '0' + ']' * 2000, raw + 'x' * g.MAX_RESPONSE_CHARS]:
        with pytest.raises(g.GroundingContractError):
            g.parse_grounding_response(malformed,batch)


def test_schema_is_fresh_structurally_valid_and_not_an_entailment_oracle():
    _,batch,row = simple()
    schema = g.build_output_schema(batch)
    jsonschema.Draft202012Validator.check_schema(schema)
    original = copy.deepcopy(schema)
    schema['properties']['batch_sha256']['enum'][0] = 'tampered'
    assert g.build_output_schema(batch) == original
    row['original']['support']['evidence'][0]['quote'] = 'fabricated'
    jsonschema.Draft202012Validator(original).validate(response(batch,[row]))
    with pytest.raises(g.GroundingContractError):
        parse(batch,[row])


def test_schema_stays_within_documented_structured_output_subset():
    _,batch,_ = simple(value_numeric=0.0,value_unit='items')
    permitted = {'type','properties','required','additionalProperties','enum',
                 'anyOf','items','minItems','maxItems','minimum','maximum',
                 'minLength','maxLength','$schema','$defs','$ref'}
    def visit(node):
        assert type(node) is dict and set(node) <= permitted
        if node.get('type') == 'object':
            assert node['additionalProperties'] is False
            assert set(node['required']) == set(node['properties'])
        for key in ('properties','$defs'):
            for child in node.get(key,{}).values():
                visit(child)
        if 'items' in node:
            visit(node['items'])
        for child in node.get('anyOf',[]):
            visit(child)
    visit(g.build_output_schema(batch))


@pytest.mark.parametrize('source,scope,attestation', [
    ('I never use CedarTool anywhere.', None, 'supported'),
    ('I do not use CedarTool on my laptop.', None, 'not_established'),
    ('I do not use CedarTool on my laptop, but I use it on my desktop.', None, 'not_established'),
    ('I did not use CedarTool during June.', 'June', 'supported'),
])
def test_complementary_scoped_negative_attestations_do_not_relabel_fixed_control(source,scope,attestation):
    triple = g.Triple('Iris','uses','CedarTool',-1,source_message_id=7,temporal_scope=scope)
    _,batch = g.build_grounding_request((triple,), (g.GroundingSource(7,source,source_role='user',source_peer_id='Iris'),))
    verdict = parse(batch,[item(triple,original=attestation,evidence=[quote(7,source)])]).verdicts[0]
    assert verdict.status == ('supported' if attestation == 'supported' else 'unsupported')
    assert cases.suite_sha256() == '511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925'
