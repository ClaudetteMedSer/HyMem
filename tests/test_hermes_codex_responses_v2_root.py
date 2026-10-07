"""Independent offline review of native JSON handling and process containment."""
import json
import multiprocessing
import time

import pytest

from benchmarks import hermes_codex_responses_v2 as wire
from tests.test_hermes_codex_responses_root import partial_ipc_child, large_success_child, result_event
from tests.test_hermes_codex_responses_v2 import FakeResponse, invoke


def test_partial_ipc_deadline_and_large_response_against_v2():
    prior = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(wire.TransportError, match='timeout'):
        wire._run_child(partial_ipc_child, (), 0.7)
    assert time.monotonic() - started < 3
    assert {p.pid for p in multiprocessing.active_children()} == prior
    assert len(wire._run_child(large_success_child, (), 5).text) == 900000
    assert {p.pid for p in multiprocessing.active_children()} == prior


@pytest.mark.parametrize('mutation', [
    lambda value: value['usage'].update(total_tokens=True),
    lambda value: value['usage'].update(input_tokens_details={'cached_tokens': 99}),
    lambda value: value['usage'].update(output_tokens_details={'reasoning_tokens': 99}),
    lambda value: value['output'][0]['content'].append({'type': 'refusal', 'refusal': 'private'}),
    lambda value: value.update(output=[{'type': 'web_search_call'}]),
])
def test_json_cannot_bypass_semantic_completion_checks(monkeypatch, mutation):
    value = result_event()['response']
    mutation(value)
    _, call = invoke(monkeypatch, FakeResponse(json.dumps(value).encode()))
    with pytest.raises(wire.TransportError) as failure:
        call()
    assert 'private' not in str(failure.value)


@pytest.mark.parametrize('content_type,mitigation,expected', [
    ('text/html', '', 'html_response'),
    ('application/json', 'challenge', 'access_challenge'),
    ('application/x-private-format', '', 'unsupported_media_type'),
])
def test_blocked_response_does_not_read_private_body(monkeypatch, content_type, mitigation, expected):
    class Unreadable(FakeResponse):
        def read(self, limit):
            pytest.fail('Terminal media type must not read body')
        def readline(self, limit):
            pytest.fail('Terminal media type must not read body')
    _, call = invoke(monkeypatch, Unreadable(content_type=content_type, mitigation=mitigation))
    with pytest.raises(wire.TransportError, match=expected):
        call()


def test_nested_duplicate_json_is_rejected():
    response = FakeResponse(b'{"usage":{"input_tokens":3,"input_tokens":3}}')
    with pytest.raises(wire.TransportError, match='invalid_json'):
        wire._json_response(response)
