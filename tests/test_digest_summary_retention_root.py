"""Independent scope-boundary controls, not model-accuracy measurements."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from benchmarks import digest_retention_inventory as legacy
from benchmarks import digest_summary_retention as summary
from hymem.deadline import DeadlineExceeded
from hymem.extraction.llm import LLMRequest
from tests.digest_source_review_evaluation_fixtures import payload, procedure, source


def inputs():
    records = [
        source('unit-91001', 91001,
               'Rotate logs by stopping the writer and archiving the log. '
               'Rebuild search by exporting the schema and rebuilding the index.', 'assistant'),
        source('unit-91002', 91002, 'The deployment verification passed.', 'assistant'),
    ]
    value = payload(records,
        'Log rotation stops the writer before archiving; search rebuild exports the schema '
        'before rebuilding the index, and deployment verification passed.',
        prior='Earlier, a backup was scheduled.')
    value['procedure_items'] = [
        {'index': 0, 'candidate': procedure('Rotate logs',
            ('Stop the writer', 'Archive the log'), description='Rotate the logs.'),
         'cited_source_ids': ['unit-91001']},
        {'index': 1, 'candidate': procedure('Rebuild search',
            ('Export the schema', 'Rebuild the index'), description='Rebuild the search index.'),
         'cited_source_ids': ['unit-91001']},
    ]
    return value


def plan(value=None, **kwargs):
    return summary.prepare_summary_retention(inputs() if value is None else value,
        LLMRequest('unused system', 'unused user', temperature=0.25, max_tokens=8192), **kwargs)


def inventory_reply(prepared, *, uncertain=False):
    units = prepared.inventory_scope.units
    return json.dumps({'obligations': [
        {'facet': 'material_facts', 'unit_ids': [unit.unit_id], 'text': unit.text}
        for unit in units], 'no_material_unit_ids': [],
        'uncertain_unit_ids': [units[0].unit_id] if uncertain else []})


def matching(prepared, *, uncertain=False):
    parsed = summary.parse_summary_inventory(inventory_reply(prepared, uncertain=uncertain), prepared)
    assert parsed.inventory is not None
    return summary.prepare_summary_matching(prepared, parsed.inventory)


def matching_reply(bound, *, omit_last=False):
    scope = bound.matching_scope
    ref = next(f.field_id for f in scope.inventory_scope.record_scope.fields
               if f.path == '/candidate_summary')
    values = {o.obligation_id: ['retained', [ref]] for o in scope.inventory.obligations}
    if omit_last:
        values[scope.inventory.obligations[-1].obligation_id] = ['omitted', []]
    return json.dumps(values)


class Replies:
    def __init__(self, *values):
        self.values = list(values)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def test_old_item_source_request_cannot_determine_the_item_topic():
    value = inputs()
    swapped = deepcopy(value)
    swapped['procedure_items'][0]['candidate'], swapped['procedure_items'][1]['candidate'] = (
        swapped['procedure_items'][1]['candidate'], swapped['procedure_items'][0]['candidate'])
    left = legacy.prepare_retention_inventory(value, LLMRequest('', ''), max_calls=6)
    right = legacy.prepare_retention_inventory(swapped, LLMRequest('', ''), max_calls=6)
    # Each item's intended procedure changes, while its entire source-only wire
    # stays identical. A positional index is not a semantic coverage assignment.
    for first, second in zip(left.requests[:2], right.requests[:2], strict=True):
        assert first.kind == second.kind == 'procedure'
        assert first.request == second.request
        assert first.binding_sha256 != second.binding_sha256


def test_summary_scope_keeps_all_new_sources_not_just_item_citations():
    prepared = plan()
    assert prepared.reserved_calls == prepared.max_calls == 2
    assert prepared.inventory_scope.kind == 'summary'
    assert prepared.inventory_scope.index == 0
    assert {s.chunk_id for s in prepared.inventory_scope.record_scope.canonical_sources} == {
        'unit-91001', 'unit-91002'}
    assert len(prepared.validation_plan.requests) == 3  # Validation, not a call schedule.
    wire = json.loads(prepared.inventory_scope.request.user)
    assert wire['scope'] == {'kind': 'summary', 'index': 0}
    assert set(wire) == {'schema', 'scope', 'source_records', 'location_units'}
    assert 'candidate' not in wire
    assert 'Earlier, a backup' not in prepared.inventory_scope.request.user


def test_item_and_summary_candidate_mutations_do_not_select_source_obligations():
    original = plan()
    value = inputs()
    value['procedure_items'][0]['candidate']['description'] = 'UNTRUSTED-CANDIDATE-SENTINEL'
    value['summary_item'].update(candidate_summary='CHANGED-SUMMARY-SENTINEL',
                                 candidate_raw_summary='CHANGED-SUMMARY-SENTINEL',
                                 prior_derived_summary='CHANGED-PRIOR-SENTINEL')
    changed = plan(value)
    assert original.inventory_scope.request == changed.inventory_scope.request
    assert original.binding_sha256 != changed.binding_sha256
    for marker in ('UNTRUSTED-CANDIDATE', 'CHANGED-SUMMARY', 'CHANGED-PRIOR'):
        assert marker not in changed.inventory_scope.request.user


@pytest.mark.parametrize('mutation', [
    lambda p: p['procedure_items'][0].update(cited_source_ids=['unknown']),
    lambda p: p['procedure_items'][0]['candidate']['steps'][0].update(order=2),
    lambda p: p['procedure_items'][0].update(index=4),
    lambda p: p['source_catalog'][0].update(end=999999),
    lambda p: p['summary_item'].update(new_source_ids=['unit-91001']),
    lambda p: p['summary_item'].update(new_source_ids=['unit-91002', 'unit-91001']),
    lambda p: p['summary_item'].update(candidate_is_noop=True),
])
def test_full_payload_validated_before_ignoring_item_retention(mutation):
    value = inputs()
    mutation(value)
    with pytest.raises(ValueError):
        plan(value)


@pytest.mark.parametrize('cap', [0, 1, 3, 64, True, 2.0, '2', None])
def test_only_exact_two_call_reservation_is_accepted(cap):
    with pytest.raises(ValueError):
        plan(max_calls=cap)


def test_matching_keeps_exact_effective_candidate_and_prior_but_not_item_fields():
    prepared = plan()
    bound = matching(prepared)
    wire = json.loads(bound.matching_scope.request.user)
    expected = inputs()['summary_item']
    assert wire['candidate']['candidate_summary'] == expected['candidate_summary']
    assert [s['text'] for s in wire['prior_summary_sources']] == [expected['prior_derived_summary']]
    assert all(f['path'].startswith('/candidate_') for f in wire['fields'])
    assert not any('/steps/' in f['path'] for f in wire['fields'])


@pytest.mark.parametrize('omit_last,uncertain,affirmative', [
    (False, False, True), (True, False, False), (False, True, False), (True, True, False),
])
def test_omission_and_uncertainty_cannot_be_waived_by_scope_selection(omit_last, uncertain, affirmative):
    prepared = plan()
    bound = matching(prepared, uncertain=uncertain)
    result = summary.parse_summary_matching(matching_reply(bound, omit_last=omit_last), prepared, bound)
    assert result.structure_valid
    assert result.model_retention_satisfied is affirmative
    assert not result.semantic_verified and not result.publication_authorized


def test_result_never_claims_item_grounding_continuity_or_benchmark_readiness():
    prepared = plan()
    bound = matching(prepared)
    client = Replies(inventory_reply(prepared), matching_reply(bound))
    result = summary.execute_summary_retention(prepared, client)
    assert result.collection_complete and result.summary_structure_valid
    assert result.summary_model_retention_satisfied and result.attempted_calls == 2
    assert len(client.requests) == 2
    for value in (prepared, bound, result):
        assert value.coverage_scope == 'summary_new_sources_only'
        assert not value.prior_continuity_assessed
        assert not value.grounding_assessed
        assert not value.item_retention_assessed
        assert not value.semantic_verified and not value.publication_authorized
    for name in ('complete', 'pass', 'model_retention_satisfied', 'benchmark_ready'):
        assert not hasattr(result, name)


def test_item_scope_substitution_fails_before_first_call():
    prepared = plan()
    item_scope = legacy.prepare_retention_inventory(inputs(), LLMRequest('', ''), max_calls=6).requests[0]
    forged = replace(prepared, inventory_scope=item_scope)
    client = Replies('must not be sent')
    with pytest.raises(ValueError):
        summary.execute_summary_retention(forged, client)
    assert not client.requests


def test_other_candidate_cannot_reuse_bound_matching():
    prepared = plan()
    bound = matching(prepared)
    changed = inputs()
    changed['summary_item']['candidate_summary'] = 'The deployment verification passed.'
    other = plan(changed)
    with pytest.raises(ValueError):
        summary.parse_summary_matching(matching_reply(bound), other, bound)


@pytest.mark.parametrize('raw', ['not JSON', '[]', '{}', None])
def test_invalid_inventory_has_no_second_reply_and_no_collection_complete_claim(raw):
    prepared = plan()
    client = Replies(raw)
    result = summary.execute_summary_retention(prepared, client)
    assert result.attempted_calls == 1 and len(client.requests) == 1
    assert not result.collection_complete
    assert not result.summary_structure_valid and not result.summary_model_retention_satisfied


def test_empty_inventory_is_unassessed_not_vacuously_successful():
    prepared = plan()
    raw = json.dumps({'obligations': [], 'no_material_unit_ids': [
        u.unit_id for u in prepared.inventory_scope.units], 'uncertain_unit_ids': []})
    client = Replies(raw)
    result = summary.execute_summary_retention(prepared, client)
    assert result.attempted_calls == 1 and len(client.requests) == 1
    assert result.inventory.status == 'unassessed'
    assert result.matching.status == 'skipped_empty_inventory'
    assert not result.collection_complete and not result.summary_model_retention_satisfied


def test_fully_empty_canonical_scope_rejected_before_any_invocation():
    value = inputs()
    for record in value['source_catalog']:
        record.update(visible_content='', end=record['start'])
    with pytest.raises(ValueError):
        plan(value)


@pytest.mark.parametrize('phase', ['inventory', 'matching'])
def test_client_cannot_mutate_scope_and_return_a_success(phase):
    prepared = plan()
    bound = matching(prepared)

    class MutatingClient(Replies):
        def complete(self, request):
            raw = super().complete(request)
            if len(self.requests) == (1 if phase == 'inventory' else 2):
                object.__setattr__(prepared, 'coverage_scope', 'all_digest_items')
            return raw

    client = MutatingClient(inventory_reply(prepared), matching_reply(bound))
    with pytest.raises(ValueError):
        summary.execute_summary_retention(prepared, client)
    assert len(client.requests) == (1 if phase == 'inventory' else 2)


def test_cancellation_survives_even_an_invalidated_plan():
    prepared = plan()
    error = DeadlineExceeded('deadline wins over plan invalidation')

    class CancellingClient:
        def complete(self, request):
            object.__setattr__(prepared, 'max_calls', 99)
            raise error

    with pytest.raises(DeadlineExceeded) as caught:
        summary.execute_summary_retention(prepared, CancellingClient())
    assert caught.value is error


@pytest.mark.parametrize('stage', [0, 1])
def test_client_error_halts_without_retry_or_false_completion(stage):
    prepared = plan()
    values = ([RuntimeError('SECRET-ERROR')] if stage == 0
              else [inventory_reply(prepared), RuntimeError('SECRET-ERROR')])
    client = Replies(*values)
    result = summary.execute_summary_retention(prepared, client)
    assert result.attempted_calls == stage + 1
    assert not result.collection_complete and not result.summary_model_retention_satisfied
    assert 'SECRET-ERROR' not in repr(result)


@pytest.mark.parametrize('stage', [0, 1])
@pytest.mark.parametrize('error', [KeyboardInterrupt(), SystemExit(7), DeadlineExceeded('deadline')])
def test_cancellation_is_not_a_semantic_failure(stage, error):
    prepared = plan()
    values = [error] if stage == 0 else [inventory_reply(prepared), error]
    client = Replies(*values)
    with pytest.raises(type(error)) as caught:
        summary.execute_summary_retention(prepared, client)
    assert caught.value is error
    assert len(client.requests) == stage + 1
