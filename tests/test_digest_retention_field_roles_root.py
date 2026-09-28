"""Independent field-role/fixture checks; no claims about model accuracy."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import digest_retention_inventory as inventory
from hymem.dreaming.digest import _validate_digest_procedure_items
from hymem.extraction.llm import LLMRequest
from tests.digest_retention_inventory_fixtures import build_cases as legacy_cases
from tests.digest_retention_inventory_v2_fixtures import build_cases


CASES = build_cases()


def scope(case):
    plan = inventory.prepare_retention_inventory(
        case['payload'], LLMRequest('', '', max_tokens=3072), max_calls=6)
    return next(s for s in plan.requests
                if {'kind': s.kind, 'index': s.index} == case['target_scope'])


def differences(left, right, path=''):
    if type(left) is not type(right):
        return {path}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {path}
        return set().union(*(differences(left[k], right[k], f'{path}/{k}') for k in left))
    if isinstance(left, list):
        if len(left) != len(right):
            return {path}
        return set().union(*(differences(a, b, f'{path}/{i}')
                             for i, (a, b) in enumerate(zip(left, right))))
    return set() if left == right else {path}


def test_legacy_fixture_file_is_unchanged_and_new_ids_do_not_relabel_history():
    path = Path(__file__).with_name('digest_retention_inventory_fixtures.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        'ba48456a194f465ae1a6b65b7b19a19a19174cc9a5b4e20d7a0173f903a41022')
    assert not {c['id'] for c in CASES} & {c['id'] for c in legacy_cases()}
    assert len(CASES) == len({c['id'] for c in CASES}) == 14
    old_sources = Counter(json.dumps(c['payload']['source_catalog'], sort_keys=True)
                          for c in legacy_cases())
    new_sources = Counter(json.dumps(c['payload']['source_catalog'], sort_keys=True)
                          for c in CASES)
    assert not old_sources - new_sources


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_procedure_candidates_meet_actual_runtime_shape_validation(case):
    # Diagnostic input omits chunk_ids from the candidate itself; the runtime
    # item stores the identical provenance alongside its other fields.
    for item in case['payload']['procedure_items']:
        candidate = deepcopy(item['candidate'])
        candidate['chunk_ids'] = list(item['cited_source_ids'])
        accepted, rejected = _validate_digest_procedure_items(
            [candidate], [r['chunk_id'] for r in case['payload']['source_catalog']])
        assert rejected == 0 and len(accepted) == 1
        assert [s['order'] for s in candidate['steps']] == list(range(1, len(candidate['steps']) + 1))
        # Shape acceptance is not technical-domain, grounding or semantic proof.


def test_all_pair_changes_are_declared_candidate_only_and_inventory_blind():
    pairs = defaultdict(list)
    for case in CASES:
        pairs[case['pair']].append(case)
    assert len(pairs) == 7
    for left, right in pairs.values():
        assert left['expected_source_obligations'] == right['expected_source_obligations']
        actual = differences(left['payload'], right['payload'])
        assert actual == set(left['changed_payload_paths']) == set(right['changed_payload_paths'])
        assert actual and all(p.startswith(('/items/', '/procedure_items/', '/summary_item/candidate_')) for p in actual)
        a, b = scope(left), scope(right)
        assert a.request == b.request
        assert a.source_binding_sha256 == b.source_binding_sha256
        assert a.binding_sha256 != b.binding_sha256
        if left['payload']['procedure_items']:
            assert left['payload']['procedure_items'][0]['candidate']['triggers'] == right['payload']['procedure_items'][0]['candidate']['triggers']


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_gold_uses_real_semantic_fields_and_never_enters_model_input(case):
    before = deepcopy(case)
    selected = scope(case)
    fields = {field.path: field for field in selected.record_scope.fields}
    for match in case['candidate_matches']:
        assert set(match['candidate_field_paths']) <= fields.keys()
        assert not any(p.startswith(('/candidate/triggers/', '/candidate/entities_involved/'))
                       for p in match['candidate_field_paths'])
    # Changing every annotation cannot change the source or candidate requests.
    mutated = deepcopy(case)
    for key in set(mutated) - {'payload', 'target_scope'}:
        mutated[key] = 'EXTERNAL-GOLD-NEVER-SEND'
    assert scope(mutated) == selected
    assert case == before
    assert 'EXTERNAL-GOLD-NEVER-SEND' not in selected.request.user


def test_matching_schema_and_uncertainty_gate_are_not_weakened_by_roles():
    selected = scope(next(c for c in CASES if c['target_scope']['kind'] == 'procedure'))
    reply = {'obligations': [
        {'facet': 'constraints', 'unit_ids': [u.unit_id], 'text': 'An explicit required rule.'}
        for u in selected.units], 'no_material_unit_ids': [],
        'uncertain_unit_ids': [selected.units[0].unit_id]}
    frozen = inventory.parse_inventory(json.dumps(reply), selected).inventory
    matching = inventory.prepare_matching(selected, frozen)
    field = selected.record_scope.fields[0].field_id
    verdicts = {o.obligation_id: ['retained', [field]] for o in frozen.obligations}
    result = inventory.parse_matching(json.dumps(verdicts), matching)
    assert result.structure_valid and not result.model_retention_satisfied
    assert not result.semantic_verified and not result.publication_authorized
    first = frozen.obligations[0].obligation_id
    verdicts[first] = ['not_applicable', []]
    assert not inventory.parse_matching(json.dumps(verdicts), matching).structure_valid
    body = json.loads(matching.request.user)
    assert body['fields'] == [asdict(f) for f in selected.record_scope.fields]
