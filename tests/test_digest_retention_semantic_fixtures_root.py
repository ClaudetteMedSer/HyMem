"""Root validation of prospectively labelled controls, not model measurements."""
from collections import Counter, defaultdict
from copy import deepcopy
import json

import pytest

from hymem.dreaming.digest import _validate_digest_procedure_items
from tests.digest_retention_semantic_fixtures import build_cases
from tests.test_digest_retention_field_roles_root import differences, scope


CASES = build_cases()


def test_eight_development_controls_keep_fixed_targets_and_semantic_defects():
    assert len(CASES) == len({c['id'] for c in CASES}) == 8
    assert Counter(c['variant'] for c in CASES) == {'faithful': 4, 'defective': 4}
    assert sum(c['faithful_no_veto'] for c in CASES) == 4
    assert all(c['label_scope'] == 'targeted_obligations_only' for c in CASES)
    assert Counter(m['expected'] for c in CASES for m in c['candidate_matches']) == {
        'retained': 16, 'omitted': 1, 'altered': 3}


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_exact_source_anchors_and_minimum_semantic_components(case):
    records = {r['chunk_id']: r for r in case['payload']['source_catalog']}
    targets = case['expected_source_obligations']
    assert len({t['id'] for t in targets}) == len(targets)
    assert {m['obligation_id'] for m in case['candidate_matches']} == {t['id'] for t in targets}
    for target in targets:
        record = records[target['chunk_id']]
        start = target['canonical_start'] - record['start']
        end = target['canonical_end'] - record['start']
        assert 0 <= start < end <= len(record['visible_content'])
        assert record['visible_content'][start:end] == target['canonical_quote']
        assert target['minimum_components'] and all(isinstance(c, str) and c.strip()
                                                   for c in target['minimum_components'])
    for record in records.values():
        assert record['end'] - record['start'] == len(record['visible_content'])


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_runtime_shape_and_witness_fields_do_not_conflate_queries_with_rules(case):
    for item in case['payload']['procedure_items']:
        candidate = deepcopy(item['candidate'])
        candidate['chunk_ids'] = item['cited_source_ids']
        accepted, rejected = _validate_digest_procedure_items(
            [candidate], [r['chunk_id'] for r in case['payload']['source_catalog']])
        assert rejected == 0 and len(accepted) == 1
    fields = {f.path for f in scope(case).record_scope.fields}
    for match in case['candidate_matches']:
        assert set(match['candidate_field_paths']) <= fields
        assert not any(p.startswith(('/candidate/triggers/', '/candidate/entities_involved/'))
                       for p in match['candidate_field_paths'])
        assert bool(match['candidate_field_paths']) == (match['expected'] != 'omitted')


def test_every_pair_has_exact_declared_candidate_mutations_and_identical_inventory_input():
    pairs = defaultdict(list)
    for case in CASES:
        pairs[case['pair']].append(case)
    assert len(pairs) == 4
    for left, right in pairs.values():
        assert left['payload']['source_catalog'] == right['payload']['source_catalog']
        assert left['expected_source_obligations'] == right['expected_source_obligations']
        assert differences(left['payload'], right['payload']) == set(left['changed_payload_paths']) == set(right['changed_payload_paths'])
        assert scope(left).request == scope(right).request
        assert scope(left).binding_sha256 != scope(right).binding_sha256


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_external_annotations_never_enter_source_requests_and_sources_are_lossless(case):
    selected = scope(case)
    mutated = deepcopy(case)
    for key in set(mutated) - {'payload', 'target_scope'}:
        mutated[key] = 'ROOT-GOLD-NOT-PROVIDER-INPUT'
    assert scope(mutated) == selected
    body = json.loads(selected.request.user)
    assert 'ROOT-GOLD-NOT-PROVIDER-INPUT' not in selected.request.user
    assert case['id'] not in selected.request.user
    assert not {'candidate', 'candidate_matches', 'minimum_components'} & body.keys()
    for source in selected.record_scope.canonical_sources:
        units = [u for u in selected.units if u.source_id == source.source_id]
        assert ''.join(u.text for u in units) == source.text
        assert units[0].start == source.start and units[-1].end == source.end


def test_boundary_event_is_interpretation_only_while_canonical_roles_remain_distinct():
    case = next(c for c in CASES if c['pair'] == 'contextual-completed-event-owner')
    selected = scope(case)
    records = json.loads(selected.request.user)['source_records']
    first, second = records
    assert first['canonical_source']['text'] == 'the staging cache.'
    assert first['boundary_context']['content'] == 'I flushed '
    assert first['boundary_context']['message']['message_id'] == first['current_message']['message_id']
    assert second['current_message']['role'] == 'user'
    assert second['boundary_context']['message']['role'] == 'assistant'
    assert second['current_message']['message_id'] != second['boundary_context']['message']['message_id']
    assert 'opened the metrics dashboard' in second['boundary_context']['content']
    assert all('metrics dashboard' not in s.text for s in selected.record_scope.canonical_sources)
    assert all('metrics dashboard' not in t['description'] for t in case['expected_source_obligations'])


def test_reversed_order_stays_an_alteration_not_an_omission_of_shared_readiness():
    faithful, defective = [c for c in CASES if c['pair'] == 'actual-order-versus-shared-prerequisite']
    for case in (faithful, defective):
        matches = {m['obligation_id']: m for m in case['candidate_matches']}
        assert matches['shared-database-readiness']['expected'] == 'retained'
        assert matches['both-schema-tasks-completed']['expected'] == 'retained'
    assert faithful['candidate_matches'][-1]['expected'] == 'retained'
    assert defective['candidate_matches'][-1]['expected'] == 'altered'
    # Reverse the SAME endpoints (export finish, rebuild start), rather than
    # substituting rebuild finish/export start and changing more chronology.
    assert 'cache rebuild beginning before schema export finished' in defective['payload']['summary_item']['candidate_summary']
