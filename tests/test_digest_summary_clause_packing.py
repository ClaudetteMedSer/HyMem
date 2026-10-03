"""Bounded clause packing controls; synthetic data and no provider calls."""
from __future__ import annotations

import copy
from dataclasses import asdict
import itertools
import json
import random

import pytest

from hymem import HyMem
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import digest as mod
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import LLMRequest, StubLLMClient
from tests.test_digest_bounded_summary_repair import SequenceLLM, extract, payload, quiet, source
from tests.test_lossless_digest import RollingLLM


def pack(clauses):
    return mod._validate_digest_summary_repair(json.dumps({'clauses': clauses}))


def test_suffix_lookahead_selects_first_preferred_fitting_mixed_combination():
    clauses = [['a' * 300, 'b' * 200], ['c' * 300, 'd' * 180]]
    before = copy.deepcopy(clauses)
    assert pack(clauses) == ('a' * 300 + '; ' + 'd' * 180, None)
    assert clauses == before


def test_suffix_minimum_can_force_short_first_variant_without_dropping_any_unit():
    clauses = [['a' * 350, 'b' * 240], ['c' * 270, 'd' * 249]]
    assert pack(clauses) == ('b' * 240 + '; ' + 'd' * 249, None)


def test_packer_matches_exhaustive_oracle_for_deterministic_generated_matrices(monkeypatch):
    generator = random.Random(19092026)
    for _ in range(600):
        clauses = [[chr(65 + unit * 3 + variant) * generator.randint(1, 20)
                    for variant in range(generator.randint(1, 3))]
                   for unit in range(generator.randint(1, 4))]
        cap = generator.randint(1, 85)
        monkeypatch.setattr(mod, 'SESSION_SUMMARY_MAX_CHARS', cap)
        fitting = ['; '.join(selected) for selected in itertools.product(*clauses)
                   if len('; '.join(selected)) <= cap]
        expected = next((value for value in fitting if len(value) >= 10), None)
        result, failure = pack(clauses)
        if not fitting:
            assert result is None and failure == 'summary_output_cap'
        elif expected is None:
            assert result is None and failure == 'summary_validation_failure'
        else:
            assert result == expected and failure is None
        if result is not None:
            assert len(result) <= cap


@pytest.mark.parametrize('unit', ['x', 'é', '🧪', 'e\u0301'])
def test_counts_unicode_codepoints_and_every_separator_at_500_and_501(unit):
    first = (unit * 250)[:249]
    second = (unit * 251)[:249]
    assert len(first) == len(second) == 249
    assert pack([[first], [second]]) == (first + '; ' + second, None)
    assert pack([[first], [second + 'X']]) == (None, 'summary_output_cap')
    assert pack([[first], [second + 'X', second]]) == (first + '; ' + second, None)


@pytest.mark.parametrize('size,accepted', [(1, True), (16, True), (0, False), (17, False)])
def test_unit_count_bounds_are_exact(size, accepted):
    clauses = [[f'Unit {index} is supported'] for index in range(size)]
    result, failure = pack(clauses)
    assert (failure is None) is accepted
    if accepted:
        assert result == '; '.join(unit[0] for unit in clauses)
    else:
        assert result is None and failure == 'shape_failure'


@pytest.mark.parametrize('clauses', [None, {}, 'not arrays', [[]], [['one'] * 4], ['not a unit array']])
def test_malformed_container_or_variant_counts_fail_closed(clauses):
    assert pack(clauses) == (None, 'shape_failure')


@pytest.mark.parametrize('bad', [None, 42, True, [], {}, '', ' \t\n', '""', "'  '", '\" \' \t \"'])
def test_invalid_unused_variant_rejects_entire_plan(bad):
    reason = 'summary_shape_failure' if not isinstance(bad, str) else 'summary_validation_failure'
    assert pack([['A complete first clause'], ['A complete second clause', bad]]) == (None, reason)


@pytest.mark.parametrize('raw,reason', [
    ('{"clauses":[["Complete statement"]],"summary":"Mixed schema"}', 'shape_failure'),
    ('{"clauses":[["Complete statement"]],"summaries":[]}', 'shape_failure'),
    ('{"clauses":[["Complete statement"]],"extra":true}', 'shape_failure'),
    ('{"clauses":[["Complete statement"]],"clauses":[]}', 'parse_failure'),
    ('Before {"clauses":[["Complete statement"]]}', 'parse_failure'),
    ('{"clauses":[["Complete statement"],["cut', 'output_truncated'),
])
def test_mixed_extra_duplicate_truncated_or_prose_envelopes_are_rejected(raw, reason):
    assert mod._validate_digest_summary_repair(raw) == (None, reason)


def test_whole_trimmed_variants_and_order_survive_without_quote_rewriting():
    first, second = '"DNS is fixed"', "'TLS is active'"
    assert pack([[' \n' + first + '\t', 'A different complete first statement'],
                 [second + '   ']]) == (first + '; ' + second, None)
    assert pack([['DNS fixed', 'DNS fixed'], ['TLS up']]) == ('DNS fixed; TLS up', None)


def test_overlong_unused_variants_are_not_clipped_and_minimum_overflow_is_honest():
    assert pack([['a' * 501, 'DNS fixed'], ['TLS up']]) == ('DNS fixed; TLS up', None)
    assert pack([['a' * 251], ['b' * 248]]) == (None, 'summary_output_cap')
    assert pack([['a' * 501]]) == (None, 'summary_output_cap')


@pytest.mark.parametrize('granular', [False, True])
def test_full_digest_keeps_primary_items_source_cursor_and_request_settings(source, granular):
    clauses = [['Configured staging and built the service', 'Built the service'],
               ['Deployed the service and verified its health']]
    assembled = '; '.join(unit[0] for unit in clauses)
    original = payload(source, 'x' * 674)
    before = copy.deepcopy(original)
    client = SequenceLLM(original, {'alternatives': [assembled] * 3})
    writes = source[0].conn.total_changes
    result = extract(source, client, granular=granular, max_episodes=8)
    reference_client = SequenceLLM(payload(source, assembled))
    reference = extract(source, reference_client, granular=granular, max_episodes=8)
    assert not result.parse_failed and result.summary == assembled
    assert len(client.calls) == 2 and original == before
    assert asdict(client.calls[0]) == asdict(reference_client.calls[0])
    assert result.episodes == reference.episodes and result.procedures == reference.procedures
    assert result.source_sha256 == reference.source_sha256
    assert result.covered_message_id == reference.covered_message_id and result.caught_up
    assert source[0].conn.total_changes == writes
    assert json.loads(client.calls[1].user) == {
        'original_generation_input': client.calls[0].user,
    }
    assert asdict(client.calls[1]) | {'system': client.calls[0].system, 'user': client.calls[0].user} == asdict(client.calls[0])


@pytest.mark.parametrize('clauses,reason', [
    ([['a' * 300], ['b' * 200]], 'summary_output_cap'),
    ([['A valid clause'], [' ']], 'summary_validation_failure'),
    ([['A valid clause'], ['Another valid clause', None]], 'summary_shape_failure'),
    ([], 'shape_failure'),
    ([['OK']], 'summary_validation_failure'),
])
def test_packing_or_final_validation_failure_is_atomic_after_two_calls(source, clauses, reason, monkeypatch):
    # Exercise the explicitly historical parser route; current requests reject clauses.
    monkeypatch.setattr(mod, '_validate_current_digest_summary_repair', mod._validate_digest_summary_repair)
    client = SequenceLLM(payload(source, 'x' * 674), {'clauses': clauses})
    writes = source[0].conn.total_changes
    result = extract(source, client)
    assert result.parse_failed and result.failure_reason == reason
    assert result.failure_stage == 'summary_compaction' and len(client.calls) == 2
    assert result.summary is result.covered_message_id is result.source_sha256 is None
    assert result.episodes.items == result.procedures.items == []
    assert result.next_message_offset == 0 and not result.caught_up
    assert source[0].conn.total_changes == writes


def test_complete_short_clauses_are_not_subject_to_individual_ten_character_minimum(source, monkeypatch):
    monkeypatch.setattr(mod, '_validate_current_digest_summary_repair', mod._validate_digest_summary_repair)
    client = SequenceLLM(payload(source, 'x' * 674), {'clauses': [['DNS fixed'], ['TLS up']]})
    result = extract(source, client)
    assert not result.parse_failed and result.summary == 'DNS fixed; TLS up'


@pytest.mark.parametrize('text', ['OK', 'ninechars', '"          tiny          "', "'          tiny          '"])
def test_short_single_unit_fails_existing_assembled_meaningfulness(text):
    assert pack([[text]]) == (None, 'summary_validation_failure')


@pytest.mark.parametrize('clauses,expected', [
    ([['DNS down', 'DNS remains unavailable']], 'DNS remains unavailable'),
    ([['A', 'Alpha'], ['B', 'Beta']], 'Alpha; Beta'),
    ([['A', 'Alpha'], ['B'], ['C', 'Gamma']], 'A; B; Gamma'),
    ([['"    tiny    "', '"DNS remains unavailable"']], '"DNS remains unavailable"'),
])
def test_short_first_combination_falls_back_to_earliest_complete_meaningful_variant(clauses, expected):
    assert pack(clauses) == (expected, None)


def test_short_combination_fallback_enumerates_at_most_27_products(monkeypatch):
    original = itertools.product
    yielded = []
    def counted(*units):
        for variants in original(*units):
            yielded.append(variants)
            yield variants
    monkeypatch.setattr(mod.itertools, 'product', counted)
    assert pack([['A'] * 3, ['B'] * 3, ['C'] * 3]) == (None, 'summary_validation_failure')
    assert len(yielded) == 27


@pytest.mark.parametrize('count', [4, 16])
def test_four_or_more_nonblank_units_need_no_product_search(monkeypatch, count):
    def forbidden(*units):
        raise AssertionError('unnecessary combination search')
    monkeypatch.setattr(mod.itertools, 'product', forbidden)
    clauses = [['A', 'Alpha', 'A longer complete variant'] for _ in range(count)]
    assert pack(clauses) == ('; '.join(['A'] * count), None)


@pytest.mark.parametrize('reply,expected,failure', [
    ({'summary': 'A complete older-format summary.'}, 'A complete older-format summary.', None),
    ({'summary': 'x' * 594}, None, 'summary_output_cap'),
    ({'summaries': ['x' * 501, 'A complete older-format summary.', 'Another complete summary.']},
     'A complete older-format summary.', None),
    ({'summaries': ['', 'A complete older-format summary.', 'Another complete summary.']},
     'A complete older-format summary.', None),
    ({'summaries': ['x' * 501, 'y' * 502, 'z' * 503]}, None, 'summary_output_cap'),
])
def test_legacy_single_and_whole_alternative_contracts_stay_unchanged(reply, expected, failure):
    assert mod._validate_digest_summary_repair(json.dumps(reply)) == (expected, failure)
    assert mod._DIGEST_SUMMARY_ALTERNATIVE_TARGETS == (350, 220, 120)


def test_current_prompt_requests_overview_while_historical_packing_parser_remains():
    original = LLMRequest(system='unchanged primary', user='original input', max_tokens=3072)
    request = mod._build_digest_summary_repair_request(original, 'x' * 674)
    from hymem.dreaming.summary_policy import SUMMARY_OVERVIEW_POLICY
    assert request.system.startswith('Regenerate a length-feasible summary from the original generation inputs.')
    assert SUMMARY_OVERVIEW_POLICY in request.system
    assert 'only alternatives: an array of exactly three' in request.system
    assert 'one whole variant from EVERY clause' not in request.system
    assert request.max_tokens == original.max_tokens
    assert json.loads(request.user) == {'original_generation_input': original.user}
    # Historical envelopes remain strictly validated by the existing controls.
    assert pack([['Complete supported clause']]) == ('Complete supported clause', None)


@pytest.mark.parametrize('change', ['helper', 'meaningfulness', 'units', 'variants', 'separator'])
def test_packer_and_bounds_bind_semantic_identity(monkeypatch, change):
    client = StubLLMClient(default='[]')
    before = semantic_generation_suffix('digest', client)
    if change in ('helper', 'meaningfulness'):
        name = '_pack_digest_summary_clauses' if change == 'helper' else '_digest_summary_clause_assembly_is_meaningful'
        original = getattr(mod, name)
        def changed(clauses):
            return original(clauses)
        monkeypatch.setattr(mod, name, changed)
    else:
        name, value = {'units': ('_DIGEST_SUMMARY_CLAUSE_MAX_UNITS', 15),
                       'variants': ('_DIGEST_SUMMARY_CLAUSE_MAX_VARIANTS', 2),
                       'separator': ('_DIGEST_SUMMARY_CLAUSE_SEPARATOR', ' / ')}[change]
        monkeypatch.setattr(mod, name, value)
    assert semantic_generation_suffix('digest', client) != before


@pytest.mark.parametrize('exception', [RuntimeError, DeadlineExceeded, KeyboardInterrupt])
def test_cancellation_and_exception_propagation_stop_at_two_calls(source, exception):
    failure = exception('private failure details')
    client = SequenceLLM(payload(source, 'x' * 674), failure)
    expected = mod.DigestCompletionError if exception is RuntimeError else exception
    with pytest.raises(expected) as raised:
        extract(source, client)
    assert len(client.calls) == 2
    if exception is RuntimeError:
        assert raised.value.failure_stage == 'summary_compaction' and raised.value.__cause__ is failure
    else:
        assert raised.value is failure


class ClauseDreamLLM(RollingLLM):
    def __init__(self, *, succeeds):
        super().__init__(emit_slice_artifacts=True)
        self.succeeds = succeeds

    def complete(self, request):
        if request.system.startswith('You regenerate one rolling conversation summary'):
            self.calls.append(request)
            return json.dumps({'summary': 'Built alpha; Deployed the service and verified its health'
                               if self.succeeds else 'x' * 674})
        if request.system.startswith('Regenerate a length-feasible summary from the original generation inputs.'):
            self.calls.append(request)
            summary = 'Built alpha; Deployed the service and verified its health' if self.succeeds else 'x' * 501
            return json.dumps({'alternatives': [summary] * 3})
        raw = super().complete(request)
        if request.system.startswith(('You analyze one conversation session', 'You re-read one conversation session')):
            primary = json.loads(raw)
            primary['summary'] = 'x' * 674
            return json.dumps(primary)
        return raw


@pytest.mark.parametrize('fails_first', [False, True])
def test_real_dream_publishes_or_holds_then_recovers_and_reopens(cfg, fails_first):
    client = ClauseDreamLLM(succeeds=not fails_first)
    config = quiet(cfg, dream_digest_max_chars=12000)
    hy = HyMem(config, llm=client)
    sid = 'summary-clause-publication'
    try:
        mid = hy.log_message(sid, 'user', 'Built alpha, deployed the service and verified its health.')
        hy.close_session(sid)
        if fails_first:
            assert hy.dream().digest_failures == 0
            held = hy.conn.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone()
            assert held['digest_retry_count'] == held['digest_quarantined'] == 0
            assert held['auto_summary'] is held['auto_summary_generation'] is held['auto_summary_message_id'] is None
            assert held['digest_cursor_message_id'] == held['digest_published_message_id'] == mid
            assert held['digest_published_generation'] == held['digest_cursor_prompt_version']
            assert held['summary_failure_reason'] == 'summary_output_cap' and held['summary_failure_count'] == 1
            indexed = {table: [tuple(row) for row in hy.conn.execute('SELECT * FROM ' + table)]
                       for table in ('episodes', 'procedures')}
            assert all(len(rows) == 1 for rows in indexed.values())
            assert hy.conn.execute('SELECT COUNT(*) FROM digest_staging').fetchone()[0] == 0
            status = hy.dream_status()
            assert status['pending_digests'] == status['quarantined_digests'] == status['malformed_summaries'] == 0
            assert status['summary_degraded_sessions'] == status['summary_missing_sessions'] == 1
            assert status['summary_healthy'] is False
            client.succeeds = True
            before = len(client.calls)
            assert hy.dream().digest_failures == 0
            assert len(client.calls) == before and hy.dream_status()['summary_healthy'] is False
            recovered = hy.recover_summaries(max_calls=1, session_id=sid)
            assert recovered['calls'] == recovered['published'] == 1 and recovered['remaining'] == 0
            assert {table: [tuple(row) for row in hy.conn.execute('SELECT * FROM ' + table)]
                    for table in indexed} == indexed
        else:
            assert hy.dream().digest_failures == 0
        published = hy.conn.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone()
        assert published['auto_summary'] == 'Built alpha; Deployed the service and verified its health'
        assert published['digest_cursor_message_id'] == published['auto_summary_message_id'] == mid
        assert published['digest_cursor_partial_message_id'] is None
        assert published['digest_cursor_prompt_version'] == published['digest_published_generation']
        assert published['digest_retry_count'] == 0 and published['digest_quarantined'] == 0
        assert published['digest_retry_config_version'] is None
        assert published['summary_failure_reason'] is None and published['summary_failure_count'] == 0
        assert published['auto_summary_generation'] == published['digest_published_generation']
        assert hy.dream_status()['summary_healthy'] is True
        assert hy.conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0] == 1
        assert hy.conn.execute('SELECT COUNT(*) FROM procedures').fetchone()[0] == 1
        assert hy.conn.execute('SELECT COUNT(*) FROM digest_staging').fetchone()[0] == 0
        assert hy.conn.execute('PRAGMA foreign_key_check').fetchall() == []
        assert hy.conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        calls = [r for r in client.calls if r.system.startswith((
            'Regenerate a length-feasible summary from the original generation inputs.', 'You analyze one conversation session',
            'You regenerate one rolling conversation summary'))]
        assert len(calls) == (3 if fails_first else 2)
        hy.close()
        hy = HyMem(config, llm=client)
        before = len(client.calls)
        assert hy.dream().digest_failures == 0
        assert len(client.calls) == before
        assert hy.dream_status()['summary_healthy'] is True
    finally:
        hy.close()
