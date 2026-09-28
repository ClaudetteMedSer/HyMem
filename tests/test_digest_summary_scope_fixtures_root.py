"""Root review of scope controls; scripted expected replies, not LLM accuracy."""
from collections import Counter, defaultdict
from copy import deepcopy
import json

import pytest

from benchmarks import digest_summary_retention as summary
from hymem.dreaming.digest import _validate_digest_response
from hymem.extraction.llm import LLMRequest
from tests.digest_summary_scope_fixtures import build_cases
from tests.test_digest_retention_field_roles_root import differences


CASES = build_cases()


def plan(case):
    return summary.prepare_summary_retention(case['payload'], LLMRequest('', '', max_tokens=8192))


def test_control_population_and_labels_are_prospective_not_old_rescored_results():
    assert len(CASES) == len({c['id'] for c in CASES}) == 4
    assert Counter(c['variant'] for c in CASES) == {'faithful': 2, 'defective': 2}
    assert sum(c['faithful_no_veto'] for c in CASES) == 2
    assert Counter(m['expected'] for c in CASES for m in c['candidate_matches']) == {
        'retained': 18, 'omitted': 2}
    assert all(c['target_scope'] == {'kind': 'summary', 'index': 0} for c in CASES)


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_exact_source_anchors_and_opaque_identifiers(case):
    records = {r['chunk_id']: r for r in case['payload']['source_catalog']}
    for record in records.values():
        assert record['chunk_id'] == f"unit-{record['message_id']}"
        assert record['end'] == record['start'] + len(record['visible_content'])
    targets = case['expected_source_obligations']
    assert len(targets) == len({t['id'] for t in targets})
    assert {t['id'] for t in targets} == {m['obligation_id'] for m in case['candidate_matches']}
    for target in targets:
        record = records[target['chunk_id']]
        left = target['canonical_start'] - record['start']
        right = target['canonical_end'] - record['start']
        assert 0 <= left < right <= len(record['visible_content'])
        assert record['visible_content'][left:right] == target['canonical_quote']
        assert target['minimum_components']


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_complete_candidate_passes_real_runtime_structure_only(case):
    value = case['payload']
    raw = {'summary': value['summary_item']['candidate_summary'], 'episodes': [], 'procedures': []}
    for item in value['items']:
        raw['episodes'].append({'title': item['candidate_title'], 'summary': item['candidate_body'],
            'outcome': item['candidate_outcome'], 'key_entities': item['candidate_key_entities'],
            'chunk_ids': item['cited_source_ids']})
    for item in value['procedure_items']:
        raw['procedures'].append({**item['candidate'], 'chunk_ids': item['cited_source_ids']})
    result = _validate_digest_response(json.dumps(raw), raw, 'scope-fixture-session',
        [r['chunk_id'] for r in value['source_catalog']], granular=True, max_episodes=None)
    assert not result.parse_failed
    assert result.episode_rejected_items == result.procedure_rejected_items == 0


def test_pair_changes_only_summary_not_sources_or_faithful_narrow_items():
    pairs = defaultdict(list)
    for case in CASES:
        pairs[case['pair']].append(case)
    for left, right in pairs.values():
        assert differences(left['payload'], right['payload']) == {
            '/summary_item/candidate_summary', '/summary_item/candidate_raw_summary'}
        assert left['expected_source_obligations'] == right['expected_source_obligations']
        assert left['item_scope_expectations'] == right['item_scope_expectations']
        assert plan(left).inventory_scope.request == plan(right).inventory_scope.request
        assert plan(left).binding_sha256 != plan(right).binding_sha256


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_gold_ownership_and_outcome_labels_never_enter_requests(case):
    selected = plan(case)
    annotated = deepcopy(case)
    for key in set(annotated) - {'payload', 'target_scope'}:
        annotated[key] = 'DO-NOT-SEND-GOLD-SENTINEL'
    assert plan(annotated) == selected
    assert case['id'] not in selected.inventory_scope.request.user
    wire = json.loads(selected.inventory_scope.request.user)
    assert not {'candidate', 'item_scope_expectations', 'candidate_matches'} & set(wire)
    for owner in case['item_scope_expectations']:
        assert not set(owner['owned_source_obligation_ids']) & set(owner['must_not_inherit_sibling_obligation_ids'])
        assert set(owner['cited_source_ids']) < set(case['payload']['summary_item']['new_source_ids'])


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_summary_omission_is_not_rescued_by_unchanged_sibling_output(case):
    prepared = plan(case)
    obligations = []
    for target in case['expected_source_obligations']:
        refs = [unit.unit_id for unit in prepared.inventory_scope.units
                if unit.chunk_id == target['chunk_id']
                and unit.start < target['canonical_end'] and unit.end > target['canonical_start']]
        assert refs
        obligations.append({'facet': target['facet'], 'unit_ids': refs, 'text': target['description']})
    frozen = summary.parse_summary_inventory(json.dumps({'obligations': obligations,
        'no_material_unit_ids': [], 'uncertain_unit_ids': []}), prepared)
    assert frozen.inventory is not None
    bound = summary.prepare_summary_matching(prepared, frozen.inventory)
    fields = bound.matching_scope.inventory_scope.record_scope.fields
    witness = next(field.field_id for field in fields if field.path == '/candidate_summary')
    assert all('/steps/' not in field.path and field.path != '/candidate_body' for field in fields)
    verdicts = {o.obligation_id: [label['expected'], [witness] if label['expected'] == 'retained' else []]
        for o, label in zip(frozen.inventory.obligations, case['candidate_matches'], strict=True)}
    result = summary.parse_summary_matching(json.dumps(verdicts), prepared, bound)
    assert result.structure_valid
    assert result.model_retention_satisfied == (case['variant'] == 'faithful')
    assert not result.semantic_verified and not result.publication_authorized
