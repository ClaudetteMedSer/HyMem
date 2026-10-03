"""Whole-summary alternatives: synthetic controls, no live provider requests."""
from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest

from hymem import HyMem
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import digest as mod
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import StubLLMClient
from tests.test_digest_bounded_summary_repair import SequenceLLM, extract, payload, quiet, source
from tests.test_lossless_digest import RollingLLM


GOOD = "Configured staging, deployed the service and verified its health."


@pytest.mark.parametrize('candidates,selected', [
    ([GOOD, 'A shorter complete deployment summary.', 'Deployment completed successfully.'], GOOD),
    (['a' * 594, GOOD, 'A lower-priority complete alternative.'], GOOD),
    (['a' * 594, 'b' * 501, GOOD], GOOD),
    (['tiny', '"          tiny          "', GOOD], GOOD),
    (['', "'             '", GOOD], GOOD),
    (['a' * 501, '\n "  ' + GOOD + '  "\t', 'A lower-priority complete alternative.'], '\n "  ' + GOOD + '  "\t'),
    ([GOOD, GOOD, GOOD], GOOD),
    ([GOOD, ' \n' + GOOD + '\t', 'A lower-priority complete alternative.'], GOOD),
])
def test_selects_first_whole_eligible_raw_string_without_clipping_or_joining(candidates, selected):
    assert mod._validate_digest_summary_repair(json.dumps({'summaries': candidates})) == (selected, None)


@pytest.mark.parametrize('value', [None, True, 42, {}, [], 'not-an-array'])
def test_non_array_bundle_is_shape_failure(value):
    assert mod._validate_digest_summary_repair(json.dumps({'summaries': value})) == (None, 'shape_failure')


@pytest.mark.parametrize('value', [None, True, 42, {}, []])
@pytest.mark.parametrize('index', [1, 2])
def test_valid_first_alternative_never_hides_nonstring_later_member(value, index):
    candidates = [GOOD, 'A second complete summary.', 'A third complete summary.']
    candidates[index] = value
    assert mod._validate_digest_summary_repair(json.dumps({'summaries': candidates})) == (None, 'summary_shape_failure')


@pytest.mark.parametrize('candidates,reason', [
    (['a' * 594, 'b' * 501, 'c' * 900], 'summary_output_cap'),
    (['a' * 594, 'tiny', ''], 'summary_output_cap'),
    (['tiny', '', 'a' * 594], 'summary_output_cap'),
    (['', 'tiny', "'             '"], 'summary_validation_failure'),
    (['"            "', '"          tiny          "', '\n\t'], 'summary_validation_failure'),
])
def test_all_invalid_bundles_fail_with_stable_existing_reason(candidates, reason):
    assert mod._validate_digest_summary_repair(json.dumps({'summaries': candidates})) == (None, reason)


@pytest.mark.parametrize('candidate', [
    'tiny', '"          tiny          "', "'          tiny          '",
    '"            "', "'            '", 'a' * 9, 'a' * 10,
    '"  ' + GOOD + '  "', "'  " + GOOD + "  '", 'é' * 500,
])
def test_candidate_meaningfulness_agrees_with_complete_digest(source, candidate):
    direct = extract(source, SequenceLLM(payload(source, candidate)))
    alternate = extract(source, SequenceLLM(payload(source, 'x' * 674),
                        {'alternatives': [candidate, GOOD, 'A third complete summary.']}))
    assert alternate.parse_failed == direct.parse_failed
    assert alternate.summary == direct.summary
    if direct.parse_failed:
        assert alternate.failure_reason == 'summary_validation_failure'


@pytest.mark.parametrize('legacy', [GOOD, 'é' * 500, 'y' * 594, '', 'tiny'])
def test_legacy_single_response_keeps_existing_behavior(source, legacy, monkeypatch):
    # This compatibility parser remains available outside current dispatch.
    monkeypatch.setattr(mod, '_validate_current_digest_summary_repair', mod._validate_digest_summary_repair)
    client = SequenceLLM(payload(source, 'x' * 674), {'summary': legacy})
    result = extract(source, client)
    assert len(client.calls) == 2
    if legacy == GOOD or len(legacy) == 500:
        assert not result.parse_failed and result.summary == legacy
    else:
        assert result.parse_failed
        assert result.failure_reason == ('summary_output_cap' if len(legacy) > 500 else 'summary_validation_failure')
        assert result.summary is result.source_sha256 is result.covered_message_id is None


def test_primary_and_repair_data_envelopes_limits_and_selected_items_unchanged(source):
    client = SequenceLLM(payload(source, 'x' * 674), {'alternatives': ['a' * 594, GOOD, 'A terse complete alternative.']})
    result = extract(source, client)
    assert not result.parse_failed
    request = client.calls[1]
    assert asdict(request) | {'system': client.calls[0].system, 'user': client.calls[0].user} == asdict(client.calls[0])
    assert json.loads(request.user) == {'original_generation_input': client.calls[0].user}
    from hymem.dreaming.summary_policy import SUMMARY_OVERVIEW_POLICY
    assert SUMMARY_OVERVIEW_POLICY in request.system
    assert 'only alternatives' in request.system and 'exactly three' in request.system
    assert 'not a complete inventory' in request.system
    assert 'prior automatic summary and new material' in request.system
    assert request.max_tokens == 3072 and request.temperature == 0.0 and request.response_format == 'json'


@pytest.mark.parametrize('change', ['selector', 'targets'])
def test_new_alternative_selector_and_targets_bind_semantic_generation(monkeypatch, change):
    client = StubLLMClient(default='[]')
    before = semantic_generation_suffix('digest', client)
    if change == 'selector':
        original = mod._select_digest_summary_alternative
        def changed(summaries):
            return original(summaries)
        monkeypatch.setattr(mod, '_select_digest_summary_alternative', changed)
    else:
        monkeypatch.setattr(mod, '_DIGEST_SUMMARY_ALTERNATIVE_TARGETS', (349, 219, 119))
    assert semantic_generation_suffix('digest', client) != before


@pytest.mark.parametrize('exception', [RuntimeError, DeadlineExceeded, KeyboardInterrupt])
def test_alternative_repair_exception_is_not_retried_or_masked(source, exception):
    failure = exception('private diagnostic')
    client = SequenceLLM(payload(source, 'x' * 674), failure)
    expected = mod.DigestCompletionError if exception is RuntimeError else exception
    with pytest.raises(expected) as raised:
        extract(source, client)
    assert len(client.calls) == 2
    if exception is RuntimeError:
        assert raised.value.__cause__ is failure and raised.value.failure_stage == 'summary_compaction'
    else:
        assert raised.value is failure


class AlternativesDreamLLM(RollingLLM):
    def __init__(self, *, succeeds):
        super().__init__(emit_slice_artifacts=True)
        self.succeeds = succeeds

    def complete(self, request):
        if request.system.startswith('You regenerate one rolling conversation summary'):
            self.calls.append(request)
            return json.dumps({'summary': GOOD if self.succeeds else 'x' * 674})
        if request.system.startswith('Regenerate a length-feasible summary from the original generation inputs.'):
            self.calls.append(request)
            return json.dumps({'alternatives': ['x' * 594, 'y' * 501, GOOD if self.succeeds else 'z' * 674]})
        raw = super().complete(request)
        if request.system.startswith(('You analyze one conversation session', 'You re-read one conversation session')):
            primary = json.loads(raw)
            primary['summary'] = 'x' * 674
            return json.dumps(primary)
        return raw


@pytest.mark.parametrize('fails_first', [False, True])
def test_real_dream_publishes_complete_items_or_holds_then_recovers_atomically(cfg, fails_first):
    client = AlternativesDreamLLM(succeeds=not fails_first)
    config = quiet(cfg, dream_digest_max_chars=12000)
    hy = HyMem(config, llm=client)
    sid = 'summary-alternative-publication'
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
        assert published['auto_summary'] == GOOD
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
        digest_calls = [r for r in client.calls if r.system.startswith((
            'Regenerate a length-feasible summary from the original generation inputs.', 'You analyze one conversation session',
            'You regenerate one rolling conversation summary'))]
        assert len(digest_calls) == (3 if fails_first else 2)
        hy.close()
        hy = HyMem(config, llm=client)
        before = len(client.calls)
        assert hy.dream().digest_failures == 0
        assert len(client.calls) == before
        assert hy.dream_status()['summary_healthy'] is True
        assert hy.conn.execute('SELECT auto_summary FROM sessions WHERE id=?', (sid,)).fetchone()[0] == GOOD
    finally:
        hy.close()
