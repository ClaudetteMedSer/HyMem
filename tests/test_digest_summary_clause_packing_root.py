"""Root-owned deterministic selection oracle and pipeline invariants.

All fixtures are invented. A fitting structural plan is not semantic proof.
"""
import copy
from dataclasses import asdict
from itertools import product
import json
import random

import pytest

from hymem.dreaming import digest as mod
from tests.test_digest_bounded_summary_repair import SequenceLLM, extract, payload, source


def compile_plan(clauses):
    return mod._validate_digest_summary_repair(json.dumps({'clauses': clauses}))


def oracle(clauses):
    # Enumeration order is the desired lexicographic preference order.
    for indexes in product(*(range(len(group)) for group in clauses)):
        value = '; '.join(clauses[i][j].strip() for i, j in enumerate(indexes))
        if len(value) <= 500:
            return value
    return None


@pytest.mark.parametrize('seed', range(40))
def test_root_compiler_matches_exhaustive_feasibility_and_preference_oracle(seed):
    rng = random.Random(seed)
    clauses = []
    for i in range(rng.randint(1, 6)):
        group = []
        for j in range(rng.randint(1, 3)):
            size = rng.choice([12, 30, 49, 100, 167, 249, 250, 251, 333, 500, 501])
            prefix = f'unit{i}-choice{j}:'
            group.append(prefix + chr(65 + i) * max(0, size - len(prefix)))
        clauses.append(group)
    expected = oracle(clauses)
    selected, reason = compile_plan(clauses)
    assert selected == expected
    assert reason == (None if expected is not None else 'summary_output_cap')


def test_root_lookahead_avoids_greedy_dead_end_without_preferring_all_shortest():
    clauses = [['A' * 330, 'A' * 100], ['B' * 300, 'B' * 200]]
    assert compile_plan(clauses) == ('A' * 100 + '; ' + 'B' * 300, None)


def test_root_preference_is_lexicographic_not_maximum_length_or_minimum_rank_sum():
    clauses = [['A' * 290, 'A' * 100], ['B' * 300, 'B' * 200]]
    assert compile_plan(clauses) == ('A' * 290 + '; ' + 'B' * 200, None)


@pytest.mark.parametrize('unit', ['x', 'é', '🧪', 'e\u0301'])
def test_root_unicode_separator_boundary_is_measured_in_codepoints(unit):
    one = (unit * 250)[:249]
    two = 'z' * 249
    selected, reason = compile_plan([[one], [two]])
    assert selected == one + '; ' + two and reason is None
    assert len(selected) == 500
    assert compile_plan([[one], [two + 'z']]) == (None, 'summary_output_cap')


def test_root_outer_whitespace_only_is_normalized_internal_whitespace_is_preserved():
    clauses = [[' \nService  Alpha remains blocked\t'], ['\tDatabase Beta is owned by Acme \n']]
    assert compile_plan(clauses) == ('Service  Alpha remains blocked; Database Beta is owned by Acme', None)


def test_root_every_clause_is_retained_in_order_even_with_separator_inside_a_variant():
    clauses = [['Alpha failed; rollback pending'], ['Beta remains leased, not owned'], ['Gamma is unverified']]
    assert compile_plan(clauses) == ('Alpha failed; rollback pending; Beta remains leased, not owned; Gamma is unverified', None)


@pytest.mark.parametrize('clauses,expected', [
    ([['DNS down', 'DNS remains unavailable']], 'DNS remains unavailable'),
    ([['A'], ['B', 'Beta failed']], 'A; Beta failed'),
    ([['A'], ['B'], ['C', 'Gamma']], 'A; B; Gamma'),
    ([["'  tiny  '", 'A complete meaningful summary']], 'A complete meaningful summary'),
])
def test_root_valid_later_variant_can_rescue_a_globally_too_short_first_combination(clauses, expected):
    assert compile_plan(clauses) == (expected, None)


@pytest.mark.parametrize('malformed', [
    [], None, True, 42, 'not-a-list',
    [['Valid clause'], []],
    [['Valid clause'], ['A', 'B', 'C', 'D']],
    [['Valid clause'], ['Another valid clause', None]],
    [['Valid clause'], ['Another valid clause', True]],
    [['Valid clause'], ['Another valid clause', 7]],
    [['Valid clause'], ['Another valid clause', []]],
    [['Valid clause'], ['Another valid clause', {}]],
    [['Valid clause'], ['Another valid clause', '']],
    [['Valid clause'], ['Another valid clause', ' \n\t']],
    [['Valid clause'], ['Another valid clause', '\"  \"']],
    [['Valid clause'], ['Another valid clause', "'   '"]],
    [['Valid clause']] * 17,
])
def test_root_malformed_unused_variants_or_clauses_reject_whole_plan(malformed):
    selected, reason = compile_plan(malformed)
    assert selected is None and reason is not None


@pytest.mark.parametrize('raw', [
    '{"clauses":[["Valid clause"]],"summary":"Mixed schema"}',
    '{"clauses":[["Valid clause"]],"summaries":["A","B","C"]}',
    '{"clauses":[["Valid clause"]],"extra":0}',
    '{"clauses":[["Valid clause"]],"clauses":[]}',
    'Example {"clauses":[["Valid clause"]]}',
    '{"clauses":[["Unterminated',
])
def test_root_mixed_duplicate_prose_and_truncated_envelopes_fail(raw):
    selected, reason = mod._validate_digest_summary_repair(raw)
    assert selected is None and reason is not None


@pytest.mark.parametrize('granular', [False, True])
def test_root_full_digest_keeps_primary_items_sources_and_two_call_ceiling(source, granular):
    clauses = [
        ['Service Alpha failed health checks; rollback remains pending'],
        ['Database Beta is leased from Acme, not owned by Alpha'],
    ]
    primary = payload(source, 'z' * 674)
    original = copy.deepcopy(primary)
    chosen = '; '.join(group[0] for group in clauses)
    llm = SequenceLLM(primary, {'alternatives': [chosen] * 3})
    before = source[0].conn.total_changes
    result = extract(source, llm, granular=granular, max_episodes=8)
    direct_llm = SequenceLLM(payload(source, chosen))
    direct = extract(source, direct_llm, granular=granular, max_episodes=8)
    assert not result.parse_failed and result.summary == chosen
    assert len(llm.calls) == 2 and primary == original
    assert asdict(llm.calls[0]) == asdict(direct_llm.calls[0])
    assert result.episodes == direct.episodes and result.procedures == direct.procedures
    assert result.source_sha256 == direct.source_sha256
    assert result.covered_message_id == direct.covered_message_id
    assert result.caught_up and source[0].conn.total_changes == before
    assert json.loads(llm.calls[1].user) == {
        'original_generation_input': llm.calls[0].user,
    }
    assert llm.calls[0].max_tokens == llm.calls[1].max_tokens == 3072


@pytest.mark.parametrize('plan', [[], [['X' * 250], ['Y' * 249]], [['Good clause'], ['']]])
def test_root_failure_never_publishes_a_good_prefix_or_source_authority(source, plan):
    llm = SequenceLLM(payload(source, 'z' * 674), {'clauses': plan})
    before = source[0].conn.total_changes
    result = extract(source, llm)
    assert result.parse_failed and result.failure_stage == 'summary_compaction'
    assert len(llm.calls) == 2 and source[0].conn.total_changes == before
    assert result.summary is result.source_sha256 is result.covered_message_id is None
    assert result.episodes.items == result.procedures.items == []
