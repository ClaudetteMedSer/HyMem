"""Independent direct-OAuth transport checks, invented data and no network."""
import json
import multiprocessing
import os
import struct
import time

import pytest

from benchmarks import hermes_codex_responses_v1 as wire


def result_event():
    return {'type': 'response.completed', 'response': {
        'model': 'gpt-6-luna', 'status': 'completed',
        'usage': {'input_tokens': 9, 'output_tokens': 4, 'total_tokens': 13,
                  'input_tokens_details': {'cached_tokens': 3},
                  'output_tokens_details': {'reasoning_tokens': 2}},
        'output': [{'type': 'message', 'role': 'assistant',
                    'status': 'completed', 'channel': 'final_answer',
                    'content': [{'type': 'output_text', 'text': '  invented\n'}]}],
    }}


@pytest.mark.parametrize('mutation', [
    lambda r: r['usage'].update(input_tokens=0, output_tokens=0, total_tokens=0,
                              input_tokens_details={}, output_tokens_details={}),
    lambda r: r['usage'].update(input_tokens_details=[]),
    lambda r: r['usage'].update(output_tokens_details=False),
    lambda r: r.update(model='different-model'),
    lambda r: r['usage'].update(total_tokens=True),
])
def test_terminal_validation_rejects_zero_malformed_or_wrong_model(mutation):
    event = result_event()
    mutation(event['response'])
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events([event])


@pytest.mark.parametrize('event', [
    {'type': 'response.output_item.added', 'item': {'type': 'function_call', 'name': 'never_execute'}},
    {'type': 'response.output_item.done', 'item': {'type': 'web_search_call'}},
    {'type': 'response.unknown_error', 'private': 'invented-private'},
])
def test_nonterminal_fault_cannot_be_hidden_by_later_success(event):
    with pytest.raises(wire.TransportError) as failure:
        wire.parse_stream_events([event, result_event()])
    assert 'invented-private' not in str(failure.value)


def test_request_is_detached_from_mutable_schema_and_text_preserved():
    schema = {'type': 'object', 'properties': {'flag': {'type': 'boolean'}},
              'required': ['flag'], 'additionalProperties': False}
    payload = wire.build_request(' system\n', ' user\n', schema)
    schema['properties']['flag']['type'] = 'string'
    assert payload['text']['format']['schema']['properties']['flag']['type'] == 'boolean'
    assert payload['instructions'] == ' system\n'
    assert payload['input'][0]['content'][0]['text'] == ' user\n'
    assert not set(payload) & {'tools', 'temperature', 'max_output_tokens', 'previous_response_id'}
    reply = wire.parse_stream_events([result_event()])
    assert reply.text == '  invented\n'
    assert reply.total_tokens == 13  # Reasoning/cached tokens are subsets, not additions.
    assert 'invented' not in repr(reply)


def partial_ipc_child(send, timeout):
    # A readable pipe is not proof that recv() has a complete serialized result.
    os.write(send.fileno(), struct.pack('!i', 10000) + b'partial')
    time.sleep(10)


def large_success_child(send, timeout):
    send.send(('ok', wire.Completed('x' * 900000, 9, 4, 13, 3, 2)))
    send.close()


def test_partial_ipc_is_subject_to_absolute_deadline():
    prior = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(wire.TransportError):
        wire._run_child(partial_ipc_child, (), 0.7)
    assert time.monotonic() - started < 3
    assert {p.pid for p in multiprocessing.active_children()} == prior


def test_large_output_does_not_deadlock_join_before_read():
    reply = wire._run_child(large_success_child, (), 5)
    assert len(reply.text) == 900000


class InventedStream:
    def __init__(self, lines):
        self.lines = iter(lines)

    def readline(self, limit):
        return next(self.lines, b'')


def test_sse_partial_json_never_becomes_success():
    stream = InventedStream([b'data: {"type": "response.completed"}\n'])
    with pytest.raises(wire.TransportError):
        wire.parse_stream_events(wire._sse_events(stream))


def test_sse_large_redundant_stream_is_still_byte_bounded(monkeypatch):
    monkeypatch.setattr(wire, 'MAX_WIRE_BYTES', 200)
    fragment = json.dumps({'type': 'response.output_text.delta', 'delta': 'x' * 20}).encode()
    stream = InventedStream([item for _ in range(20) for item in (b'data: ' + fragment + b'\n', b'\n')])
    with pytest.raises(wire.TransportError, match='wire_limit'):
        wire.parse_stream_events(wire._sse_events(stream))
