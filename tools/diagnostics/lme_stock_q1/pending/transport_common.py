"""Bounded one-shot transport, derived from the reviewed confirmation transport.

Inert on import. Credentials are used only by an isolated, explicitly authorized
live worker; neither credentials nor provider reasoning text enter receipts.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import time

MAX_FILE = 8 * 1024 * 1024
ENDPOINT = 'https://api.deepseek.com'
MODEL = 'deepseek-flash'
ENV = {'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
IDENTITY = re.compile(r'[A-Za-z0-9_.:/-]{1,128}\Z')


def require(value, code='validation_failed'):
    if not value:
        raise RuntimeError(code)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def loads(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, 'duplicate_json_key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'nonfinite_json'))


def raw(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, 'noncanonical_path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE, 'invalid_file')
        value = handle.read(MAX_FILE + 1)
    require(len(value) <= MAX_FILE, 'oversized_file')
    return value


def read(path):
    return loads(raw(path))


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def save(path, value):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == path.parent, 'noncanonical_path')
    data = encoded(value)
    require(len(data) <= MAX_FILE, 'oversized_receipt')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def optional_identity(payload, name):
    if name not in payload:
        return {'state': 'missing', 'value': None}
    value = payload[name]
    if value is None:
        return {'state': 'null', 'value': None}
    require(type(value) is str and IDENTITY.fullmatch(value), 'malformed_identity')
    return {'state': 'known', 'value': value}


def normalize_response(payload):
    require(type(payload) is dict, 'malformed_provider_response')
    model = optional_identity(payload, 'model')
    require(model['state'] == 'known', 'unknown_response_model')
    fingerprint = optional_identity(payload, 'system_fingerprint')
    usage = payload.get('usage')
    require(type(usage) is dict, 'unknown_usage')
    tokens = {key: usage.get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    require(all(type(value) is int and value >= 0 for value in tokens.values()), 'unknown_usage')
    require(tokens['total_tokens'] == tokens['prompt_tokens'] + tokens['completion_tokens'], 'usage_mismatch')
    require(tokens['completion_tokens'] <= 8192, 'completion_budget_exceeded')
    reasoning = {'state': 'missing', 'value': None, 'locations': []}
    details = usage.get('completion_tokens_details')
    require(details is None or type(details) is dict, 'malformed_reasoning_usage')
    reported = {}
    if isinstance(details, dict) and 'reasoning_tokens' in details:
        reported['completion_tokens_details.reasoning_tokens'] = details['reasoning_tokens']
    if 'reasoning_tokens' in usage:
        reported['reasoning_tokens'] = usage['reasoning_tokens']
    if reported:
        value = next(iter(reported.values()))
        require(all(type(other) is type(value) and other == value for other in reported.values()),
                'conflicting_reasoning_usage')
        if value is None:
            reasoning = {'state': 'null', 'value': None, 'locations': sorted(reported)}
        else:
            require(type(value) is int and 0 <= value <= tokens['completion_tokens'], 'malformed_reasoning_usage')
            reasoning = {'state': 'known', 'value': value, 'locations': sorted(reported)}
    choices = payload.get('choices')
    require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict,
            'malformed_provider_response')
    choice = choices[0]
    require(type(choice.get('index')) is int and choice['index'] == 0, 'malformed_choice_index')
    message = choice.get('message')
    require(type(message) is dict and message.get('role') == 'assistant', 'malformed_message')
    content = message.get('content')
    require(content is None or type(content) is str, 'malformed_content')
    finish = choice.get('finish_reason')
    require(finish in ('stop', 'length', 'content_filter', 'tool_calls', 'function_call'), 'unknown_finish_reason')
    return {'requested_model': MODEL, 'response_model': model['value'],
            'system_fingerprint': fingerprint, 'usage': {**tokens, 'reasoning_tokens': reasoning},
            'response': content, 'finish_reason': finish,
            'reasoning_text_retained': False, 'model_version_pinned': False}


def key_from_text(text):
    values = []
    for line in text.splitlines():
        match = re.fullmatch(r'[ \t]*(?:export[ \t]+)?DEEPSEEK_API_KEY[ \t]*=(.*)', line)
        if match:
            tokens = shlex.split(match[1], comments=True, posix=True)
            require(len(tokens) == 1 and re.fullmatch(r'[A-Za-z0-9_-]{20,200}', tokens[0]), 'invalid_credential')
            values.append(tokens[0])
    require(len(values) == 1 and values[0] != 'dummy-loopback-only', 'invalid_credential')
    return values[0]


def read_key():
    path = Path('/run/deepseek.env')
    require(os.geteuid() == 1000 and path.resolve() == path, 'invalid_credential_mount')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == 1000
                and stat.S_IMODE(info.st_mode) == 0o600 and info.st_size <= 65536,
                'invalid_credential_permissions')
        value = handle.read(65537)
    require(len(value) <= 65536, 'oversized_credential_file')
    return key_from_text(value.decode())


def completed_response(body, credential, destination, *, authority):
    import httpx
    from openai import OpenAI
    count = 0

    def on_request(request):
        nonlocal count
        require(count == 0 and request.method == 'POST'
                and str(request.url) == ENDPOINT + '/chat/completions', 'unexpected_transport_attempt')
        require(encoded(loads(request.content)) == encoded(body), 'wire_drift')
        require(all(type(authority[k]) in (int, float) and math.isfinite(authority[k])
                    for k in ('issued_at', 'expires_at')), 'invalid_authority_time')
        require(authority['issued_at'] <= time.time() < authority['expires_at'], 'expired_authority')
        count += 1
        save(destination / 'http-attempt.json', {'number': 1, 'started_at': time.time(),
             'wire_sha256': hashlib.sha256(encoded(body)).hexdigest(), 'endpoint': ENDPOINT})

    with httpx.Client(trust_env=False, follow_redirects=False,
                      transport=httpx.HTTPTransport(retries=0), timeout=120,
                      event_hooks={'request': [on_request]}) as http:
        with OpenAI(api_key=credential, base_url=ENDPOINT, max_retries=0, http_client=http) as client:
            params = {key: value for key, value in body.items() if key != 'thinking'}
            response = client.chat.completions.with_raw_response.create(
                **params, extra_body={'thinking': body['thinking']})
            try:
                text = response.text
                require(type(text) is str and len(text.encode()) <= MAX_FILE, 'oversized_provider_response')
                result = normalize_response(loads(text))
            finally:
                response.http_response.close()
    require(count == 1, 'attempt_accounting_mismatch')
    return {**result, 'http_attempts': 1}
