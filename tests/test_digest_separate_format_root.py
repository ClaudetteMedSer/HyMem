"""Independent isolation controls. Scripted approvals are not model-accuracy evidence."""
from dataclasses import replace
import json

import pytest

from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline, use_deadline
from hymem.dreaming import digest
from tests.test_digest_content_recovery_root import Client, extract


class SeparateClient(Client):
    def __init__(self, *, text='Dr. K. Moss tested v2.1; deployment remains conditional.',
                 grammar='supported', semantic='supported', malformed=None):
        super().__init__(initial=text, first='supported', diagnosis={'issues': []})
        self.grammar, self.semantic, self.malformed = grammar, semantic, malformed
        self.grammar_requests = []

    def complete(self, request):
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            self.requests.append(request)
            payload = json.loads(request.user)
            self.verifications.append(payload)
            value = {key: [{'index': i, 'verdict': 'supported'} for i in range(count)]
                     for key, count in [('episode_titles', len(payload['items'])),
                                        ('episode_content', len(payload['items'])),
                                        ('procedures', len(payload['procedure_items'])),
                                        ('summary_content', 1)]}
            value['summary_content'][0]['verdict'] = self.semantic
            return json.dumps(value)
        if request.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            self.requests.append(request)
            self.grammar_requests.append(request)
            payload = json.loads(request.user)
            value = {'summary_format': [{'index': 0, 'verdict': self.grammar}],
                     'episode_format': [{'index': i, 'verdict': 'supported'} for i in range(len(payload['items']))]}
            if self.malformed: self.malformed(value)
            return json.dumps(value)
        return super().complete(request)


@pytest.mark.parametrize('grammar', ['supported', 'unsupported', 'uncertain'])
def test_root_final_candidate_only_grammar_always_required(cfg, grammar):
    client = SeparateClient(grammar=grammar)
    result, _ = extract(cfg, client)
    assert len(client.grammar_requests) == 1 and len(client.requests) == 3
    assert result.parse_failed is (grammar != 'supported')
    request = client.grammar_requests[0]
    assert replace(request, system=client.requests[0].system, user=client.requests[0].user) == client.requests[0]
    payload = json.loads(request.user)
    assert set(payload) == {'schema', 'summary_item', 'items'}
    assert payload['summary_item'] == {'index': 0, 'candidate_summary': client.initial}
    assert payload['items'] == [{'index': 0, 'candidate_body': client.primary['episodes'][0]['summary']}]
    assert 'source_catalog' not in request.user and 'prior_derived_summary' not in request.user
    if grammar != 'supported':
        assert result.summary is result.source_sha256 is result.covered_message_id is None
        assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)


@pytest.mark.parametrize('text', [
    'Checks passed; deployment remains conditional.',
    'Dr. A. B. Moss tested v2.1 at 1.5 seconds; release remains pending.',
    '"Cache v2.1" passed tests; the team retained the fallback.',
    'The operator supplied directions via route 23/45, continuing earlier planning topics.',
])
def test_root_format_task_gets_exact_punctuation_and_quotes(cfg, text):
    client = SeparateClient(text=text)
    result, _ = extract(cfg, client)
    assert not result.parse_failed and result.summary == text
    assert json.loads(client.grammar_requests[0].user)['summary_item']['candidate_summary'] == text


@pytest.mark.parametrize('semantic', ['unsupported', 'uncertain'])
def test_root_semantic_veto_cannot_reach_a_favorable_grammar_model(cfg, semantic):
    client = SeparateClient(semantic=semantic, grammar='supported')
    result, _ = extract(cfg, client)
    assert result.failure_reason == 'summary_diagnosis_unactionable'
    assert len(client.requests) == 3 and not client.grammar_requests
    assert result.summary is result.source_sha256 is result.covered_message_id is None


@pytest.mark.parametrize('damage', ['semantic_key', 'missing_episode', 'boolean_index', 'unknown_verdict'])
def test_root_format_schema_cannot_return_semantic_authority_or_partial_approval(cfg, damage):
    def mutate(value):
        if damage == 'semantic_key': value['summary_content'] = [{'index': 0, 'verdict': 'supported'}]
        elif damage == 'missing_episode': value['episode_format'] = []
        elif damage == 'boolean_index': value['summary_format'][0]['index'] = False
        else: value['summary_format'][0]['verdict'] = 'looks_good'
    client = SeparateClient(malformed=mutate)
    result, _ = extract(cfg, client)
    assert result.failure_reason == 'format_adjudication_shape_failure' and len(client.requests) == 3
    assert result.summary is result.source_sha256 is result.covered_message_id is None


@pytest.mark.parametrize('prior', [None, 'Earlier retained summary; its punctuation stays unchanged.'])
def test_root_noop_format_task_checks_effective_prior_not_empty_sentinel(cfg, prior):
    client = SeparateClient(text='')
    result, _ = extract(cfg, client, prior=prior)
    assert not result.parse_failed and result.summary is None
    assert json.loads(client.grammar_requests[0].user)['summary_item']['candidate_summary'] == (prior or '')


def test_root_final_format_input_cap_holds_before_dispatch(cfg, monkeypatch):
    monkeypatch.setattr(digest, '_DIGEST_FORMAT_ADJUDICATION_MAX_INPUT_CHARS', 1)
    client = SeparateClient()
    result, _ = extract(cfg, client)
    assert result.failure_reason == 'format_adjudication_input_cap'
    assert len(client.requests) == 2 and not client.grammar_requests
    assert result.summary is result.source_sha256 is result.covered_message_id is None


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_root_mandatory_format_never_resets_deadline(cfg, phase):
    client = SeparateClient()
    clock = [0.0]
    complete = client.complete
    def timed(request):
        value = complete(request)
        if (phase == 'before' and request.system == digest._DIGEST_FIDELITY_SYSTEM
                or phase == 'after' and request.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM):
            clock[0] = 2.0
        return value
    client.complete = timed
    deadline = MonotonicDeadline(1.0, clock=lambda: clock[0])
    with use_deadline(deadline), pytest.raises(DeadlineExceeded):
        extract(cfg, DeadlineBoundLLMClient(client, deadline))
    assert len(client.requests) == (2 if phase == 'before' else 3)
