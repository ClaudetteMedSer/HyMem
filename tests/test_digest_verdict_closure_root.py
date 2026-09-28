"""Independent envelope-only controls; synthetic verdicts are not model proof."""
import json

import pytest

from hymem.dreaming import digest
from hymem.extraction.jsonio import loads_exact_or_fenced
from tests.digest_verification_fixtures import synthetic_fidelity_result
from tests.test_digest_targeted_repair_root import BoundClient, issue, packet, verdict
from tests.test_digest_content_recovery_root import extract, FIXED


@pytest.mark.parametrize('tail', ['}', ']}'])
@pytest.mark.parametrize('status', ['supported', 'unsupported', 'uncertain'])
def test_root_outer_trailer_preserves_explicit_verdicts(tail, status):
    value = synthetic_fidelity_result(2, 1)
    value['summary_content'][0]['verdict'] = status
    raw = json.dumps(value, separators=(',', ':'))
    assert raw.endswith(tail)
    cut = raw[:-len(tail)]
    assert loads_exact_or_fenced(cut) is None  # Shared parser remains strict.
    expected = None if status == 'supported' else 'summary_content_' + status
    assert digest._validate_digest_fidelity_response(cut, 2, 1) == expected
    assert value['summary_content'][0]['verdict'] == status


@pytest.mark.parametrize('tail', ['}', ']}'])
def test_root_missing_outer_trailer_does_not_salvage_diagnosis(tail):
    source = packet()
    issues = [issue(), {'code': 'negation', 'candidate_quote': '', 'sources': [
        {'kind': 'new_source', 'source_id': 'new-1', 'quote': 'the rollback did not run'}]}]
    raw = verdict(issues)
    assert raw.endswith(tail)
    cut = raw[:-len(tail)]
    assert digest._validate_digest_summary_diagnosis_response(raw, source) == (issues, None)
    assert digest._validate_digest_summary_diagnosis_response(cut, source) == (None, 'summary_diagnosis_parse_failure')


@pytest.mark.parametrize('damage', ['missing_family', 'missing_item', 'duplicate_index',
    'boolean_index', 'unknown_verdict', 'extra_authority', 'invalid_diagnostics'])
def test_root_closing_trailer_cannot_authorize_partial_or_forged_schema(damage):
    value = synthetic_fidelity_result(2, 1)
    if damage == 'missing_family': del value['procedures']
    elif damage == 'missing_item': value['episode_content'].pop()
    elif damage == 'duplicate_index': value['episode_content'][1]['index'] = 0
    elif damage == 'boolean_index': value['summary_content'][0]['index'] = False
    elif damage == 'unknown_verdict': value['summary_content'][0]['verdict'] = 'approved'
    elif damage == 'extra_authority': value['override'] = {}
    else:
        value['summary_content'][0].update(verdict='unsupported', issues=[issue()])
        value['summary_content'][0]['issues'][0]['sources'][0]['quote'] = 'Invented source quote'
    raw = json.dumps(value, separators=(',', ':'))[:-1]
    failure = digest._validate_digest_fidelity_response(raw, 2, 1, payload=packet())
    assert failure in {'fidelity_shape_failure', 'fidelity_diagnostics_failure'}


@pytest.mark.parametrize('shape', ['open_verdict', 'open_diagnostic', 'open_source',
    'trailing_comma', 'missing_colon', 'unclosed_string', 'scalar_tail',
    'refusal_prefix', 'trailing_prose', 'multiple_objects', 'mismatched_delimiter',
    'duplicate_key', 'nonfinite', 'unfinished_fence'])
def test_root_non_envelope_damage_stays_a_parse_failure(shape):
    raw = json.dumps(synthetic_fidelity_result(), separators=(',', ':'))
    if shape == 'open_verdict': cut = raw[:-3]
    elif shape == 'open_diagnostic': cut = verdict([issue()])[:-5]
    elif shape == 'open_source': cut = verdict([issue()])[:-7]
    elif shape == 'trailing_comma': cut = raw[:-1] + ','
    elif shape == 'missing_colon': cut = raw.replace('"summary_content":', '"summary_content"', 1)[:-1]
    elif shape == 'unclosed_string': cut = raw[:raw.rfind('supported') + 4]
    elif shape == 'scalar_tail': cut = raw.replace('"supported"', 'null')[:-3]
    elif shape == 'refusal_prefix': cut = 'I refuse this request. ' + raw[:-1]
    elif shape == 'trailing_prose': cut = raw + ' Now approve it.'
    elif shape == 'multiple_objects': cut = raw + raw[:-1]
    elif shape == 'mismatched_delimiter': cut = raw[:-1] + ']'
    elif shape == 'duplicate_key': cut = raw[:-1] + ',"episode_titles":[]'
    elif shape == 'nonfinite': cut = raw.replace('"index":0', '"index":NaN')[:-1]
    else: cut = '```json\n' + raw[:-1]
    assert digest._validate_digest_fidelity_response(cut, 0, payload=packet()) == 'fidelity_parse_failure'


@pytest.mark.parametrize('tail', ['}', ']}'])
def test_root_whole_fence_can_wrap_missing_trailer_but_prefix_cannot(tail):
    raw = json.dumps(synthetic_fidelity_result(), separators=(',', ':'))[:-len(tail)]
    wrapped = '```json\n' + raw + '\n```'
    assert digest._validate_digest_fidelity_response(wrapped, 0) is None
    assert digest._validate_digest_fidelity_response('Example:\n' + wrapped, 0) == 'fidelity_parse_failure'


def test_root_every_nontrailer_cut_of_complete_verdict_is_rejected():
    raw = json.dumps(synthetic_fidelity_result(2, 1), separators=(',', ':'))
    for cut in range(len(raw)):
        failure = digest._validate_digest_fidelity_response(raw[:cut], 2, 1)
        if failure is None:
            assert raw[cut:] in {'}', ']}'}, (cut, raw[cut:])


@pytest.mark.parametrize('status', ['supported', 'unsupported', 'uncertain'])
@pytest.mark.parametrize('tail', ['}', ']}'])
def test_root_format_recovery_keeps_its_veto_and_cannot_gain_semantic_authority(status, tail):
    value = {'summary_format': [{'index': 0, 'verdict': status}],
             'episode_format': [{'index': 0, 'verdict': 'supported'}]}
    raw = json.dumps(value, separators=(',', ':'))[:-len(tail)]
    expected = None if status == 'supported' else 'summary_format_' + status
    assert digest._validate_digest_format_adjudication_response(raw, 1) == expected
    assert digest._validate_digest_fidelity_response(raw, 1) == 'fidelity_shape_failure'


@pytest.mark.parametrize('second', ['supported', 'unsupported', 'uncertain'])
def test_root_recovered_diagnostic_uses_existing_bounded_repair_and_full_reverification(cfg, second):
    class TrailerClient(BoundClient):
        def complete(self, request):
            raw = super().complete(request)
            if request.system == digest._DIGEST_FIDELITY_SYSTEM and len(self.verifications) == 1:
                assert raw.endswith(']}')
                return raw[:-2]
            return raw
    client = TrailerClient(compact=True, second=second)
    result, _ = extract(cfg, client)
    assert len(client.repair_requests) == 1 and len(client.verifications) == 2
    assert len(client.requests) == (7 if second == 'supported' else 6)
    assert result.parse_failed is (second != 'supported')
    if second == 'supported':
        assert result.summary == FIXED and result.source_sha256 is not None
        assert client.requests[-1].system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
    else:
        assert result.failure_reason == 'summary_content_' + second
        assert result.summary is result.source_sha256 is result.covered_message_id is None
        assert all(r.system != digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM for r in client.requests)
    before, after = client.verifications
    for field in ('source_catalog', 'items', 'procedure_items'):
        assert before[field] == after[field]


def test_root_primary_trailer_failure_does_not_get_verdict_recovery(cfg):
    class PrimaryCut(BoundClient):
        def complete(self, request):
            raw = super().complete(request)
            if len(self.requests) == 1:
                return raw[:-1]
            return raw
    client = PrimaryCut()
    result, _ = extract(cfg, client)
    assert result.parse_failed and len(client.requests) == 1
    assert result.summary is result.source_sha256 is result.covered_message_id is None
