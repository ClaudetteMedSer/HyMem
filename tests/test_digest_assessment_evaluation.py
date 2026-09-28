"""Offline evaluator adversarial tests; no scripted verdict is live evidence."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from benchmarks import digest_assessment_evaluation as evaluation
from benchmarks import digest_evidence_assessment as assessment
from hymem.extraction.llm import LLMRequest


def scope():
    text = 'The valve was inspected. It must not be opened.'
    payload = {'schema': 'digest-fidelity-decisions-v9', 'items': [], 'procedure_items': [],
        'source_catalog': [{'chunk_id': 'one', 'message_id': 1, 'role': 'assistant',
            'source_peer_id': None, 'source_workspace_id': None, 'start': 0,
            'end': len(text), 'visible_content': text, 'interpretation_only_context': None}],
        'summary_item': {'index': 0, 'candidate_raw_summary': 'The valve was inspected.',
            'candidate_summary': 'The valve was inspected.', 'candidate_is_noop': False,
            'new_source_ids': ['one'], 'prior_derived_summary': ''}}
    return assessment.prepare_evidence_assessment(payload,
        LLMRequest(system='', user='', max_tokens=3072), max_calls=1).requests[0]


def label(check='assertion', expected='supported', view='primary', id_='target'):
    selector = {'kind': check, **({'chunk_id': 'one'} if check == 'retention'
                                 else {'field_path': '/candidate_summary'})}
    return {'id': id_, 'view': view, 'selectors': [selector], 'expected': expected,
            'rationale': 'Explicit independently supplied test gold.'}


def reply(unit, changes=None):
    primary = next(source.source_id for source in unit.evidence_sources if source.kind == 'canonical_text')
    data = {check.check_id: ['supported', [check.source_id or primary]] for check in unit.checks}
    for kind, value in (changes or {}).items():
        check_id = next(check.check_id for check in unit.checks if check.kind == kind)
        data[check_id] = [value, []] if value != 'supported' else ['supported', [primary]]
    return json.dumps(data)


@pytest.mark.parametrize('kind', ['assertion', 'relations', 'retention'])
def test_selectors_resolve_exact_known_coordinates(kind):
    unit = scope()
    selector = label(kind)['selectors'][0]
    expected = next(check.check_id for check in unit.checks if check.kind == kind)
    assert evaluation.resolve_selector(unit, selector) == expected


@pytest.mark.parametrize('selector', [None, [], {}, {'kind': True}, {'kind': 'other'},
    {'kind': 'assertion', 'field_path': '/absent'}, {'kind': 'outcome', 'field_path': '/candidate_summary'},
    {'kind': 'assertion', 'field_path': ''}, {'kind': 'assertion', 'field_path': 1},
    {'kind': 'assertion', 'field_path': '/candidate_summary', 'extra': 1},
    {'kind': 'retention', 'chunk_id': 's0'}, {'kind': 'retention', 'chunk_id': 'one', 'field_path': 'x'},
    {'kind': 'assertion', 'chunk_id': 'one'}, {'kind': 'retention', 'chunk_id': None}])
def test_bad_selectors_raise_before_scoring(selector):
    with pytest.raises(ValueError):
        evaluation.resolve_selector(scope(), selector)


@pytest.mark.parametrize('field,value', [('id', ''), ('id', True), ('id', 'x' * 257),
    ('view', 'accuracy'), ('view', []), ('expected', 'uncertain'), ('expected', True),
    ('rationale', ''), ('rationale', '  '), ('rationale', None), ('rationale', 'x' * 8193),
    ('selectors', []), ('selectors', None), ('selectors', [None])])
def test_bad_gold_is_rejected_even_for_malformed_response(field, value):
    gold = label()
    gold[field] = value
    with pytest.raises(ValueError):
        evaluation.score_scope('invalid JSON', scope(), [gold])


@pytest.mark.parametrize('gold', [None, (), [], [None], [{}]])
def test_invalid_label_container(gold):
    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), gold)


def test_duplicate_ids_and_duplicate_semantic_targets_not_extra_denominators():
    for labels in ([label(), label()], [label(), label(id_='other')]):
        with pytest.raises(ValueError):
            evaluation.bind_labels(scope(), labels)
    gold = label()
    gold['selectors'] *= 2
    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), [gold])


def test_cross_view_same_check_is_separate_and_conflicts_rejected():
    unit = scope()
    labels = [label('retention', 'unsupported', 'primary'),
              label('retention', 'unsupported', 'retention', 'source-one')]
    report = evaluation.score_scope(reply(unit, {'retention': 'unsupported'}), unit, labels)
    assert report['views']['primary']['matches'] == report['views']['retention']['matches'] == 1
    assert 'total_matches' not in report
    labels[1]['expected'] = 'supported'
    with pytest.raises(ValueError):
        evaluation.bind_labels(unit, labels)


def test_contradictory_conjunction_gold_rejected():
    group = label('assertion', 'unsupported', 'legacy', 'group')
    group['selectors'].append(label('relations')['selectors'][0])
    with pytest.raises(ValueError, match='contradictory_gold'):
        evaluation.bind_labels(scope(), [group, label(), label('relations', id_='relation')])
    # No deduction about which member is defective when only the group is negative.
    assert len(evaluation.bind_labels(scope(), [group, label()])) == 2


def test_retention_report_does_not_silently_mix_assertions():
    with pytest.raises(ValueError):
        evaluation.bind_labels(scope(), [label(view='retention')])


@pytest.mark.parametrize('verdict', ['unsupported', 'uncertain'])
def test_retention_veto_never_scores_as_assertion_detection(verdict):
    unit = scope()
    report = evaluation.score_scope(reply(unit, {'retention': verdict}), unit,
                                     [label(expected='unsupported')])
    target = report['targets'][0]
    assert target['observed'] == 'supported' and target['false_accept'] and not target['match']
    assert target['false_accept_masked_by_other_veto'] and len(target['off_target_vetoes']) == 1


def test_assertion_veto_never_scores_as_omission_detection():
    unit = scope()
    report = evaluation.score_scope(reply(unit, {'assertion': 'unsupported'}), unit,
                                     [label('retention', 'unsupported', 'retention')])
    assert report['targets'][0]['false_accept'] and not report['targets'][0]['match']


def test_omission_and_true_assertions_are_independent_positive_and_negative_targets():
    unit = scope()
    labels = [label('retention', 'unsupported', 'retention', 'omission'),
              label('assertion', 'supported', 'auxiliary')]
    report = evaluation.score_scope(reply(unit, {'retention': 'unsupported'}), unit, labels)
    assert all(row['match'] for row in report['targets'])
    assert not report['semantic_verified'] and not report['publication_authorized']


@pytest.mark.parametrize('raw', ['{}', '{"c0":["unsupported",[]]}', 'not-json', None,
    '{"type":"json_object"}', '{"c0":["unsupported",[]],"c0":["supported",["s0"]]}'])
def test_malformed_is_never_true_negative_or_partially_salvaged(raw):
    report = evaluation.score_scope(raw, scope(), [label(expected='unsupported')])
    target = report['targets'][0]
    assert report['status'] == 'malformed_assessment'
    assert target['observed'] == 'malformed' and not target['match'] and not target['false_accept']
    assert target['off_target_vetoes'] == []


@pytest.mark.parametrize('expected,observed,match,fa,fr', [
    ('supported', 'supported', True, False, False),
    ('supported', 'unsupported', False, False, True),
    ('supported', 'uncertain', False, False, False),
    ('unsupported', 'supported', False, True, False),
    ('unsupported', 'unsupported', True, False, False),
    ('unsupported', 'uncertain', False, False, False)])
def test_confusion_categories(expected, observed, match, fa, fr):
    unit = scope()
    report = evaluation.score_scope(reply(unit, {'assertion': observed}), unit, [label(expected=expected)])
    row = report['targets'][0]
    assert (row['match'], row['false_accept'], row['false_reject']) == (match, fa, fr)
    assert report['views']['primary']['observed'] == {observed: 1}


@pytest.mark.parametrize('values,wanted', [([], 'uncertain'), (['supported'], 'supported'),
    (['supported', 'unsupported'], 'unsupported'), (['supported', 'uncertain'], 'uncertain'),
    (['unsupported', 'uncertain'], 'unsupported')])
def test_aggregation_does_not_make_empty_or_uncertain_into_support(values, wanted):
    assert evaluation.aggregate(values) == wanted


@pytest.mark.parametrize('values', [(), None, ['malformed'], [True], [None]])
def test_unknown_aggregate_verdict_not_coerced(values):
    with pytest.raises(ValueError):
        evaluation.aggregate(values)


def test_scope_forgery_rejected_and_gold_immutable():
    unit, gold = scope(), [label()]
    before = deepcopy(gold)
    evaluation.score_scope(reply(unit), unit, gold)
    assert gold == before
    with pytest.raises(ValueError):
        evaluation.score_scope(reply(unit), replace(unit, binding_sha256='0' * 64), gold)


def test_valid_wrong_evidence_does_not_make_truth_claim():
    unit = scope()
    report = evaluation.score_scope(reply(unit), unit, [label('retention', 'unsupported', 'retention')])
    assert report['targets'][0]['false_accept']
    assert report['status'] == 'valid_assessment'
    assert report['semantic_verified'] is report['publication_authorized'] is False


def test_output_limit_applied_without_changing_gold_or_repair():
    unit = scope()
    raw = reply(unit)
    assert evaluation.score_scope(raw, unit, [label()], max_output_chars=len(raw))['targets'][0]['match']
    assert evaluation.score_scope(raw, unit, [label()], max_output_chars=len(raw)-1)['status'] == 'malformed_assessment'
