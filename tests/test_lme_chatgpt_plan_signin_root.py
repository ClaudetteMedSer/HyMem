"""Root-owned OAuth controls; generated keys and invented data, no network."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from cryptography.hazmat.primitives.asymmetric import rsa
import jwt
import pytest

from tools.diagnostics import lme_chatgpt_plan_signin_v1 as signin


@pytest.fixture
def signed():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    public.update(kid='invented-key', use='sig', alg='RS256')
    flow = signin.new_flow('urn:uuid:ee2e8b63-3e81-454a-a5f1-eb9b703b1431', 1455)
    client = 'oaiapp_invented_client'
    now = int(time.time())
    claims = {'iss': signin.AUTH_ORIGIN, 'aud': client, 'sub': 'invented-subject',
              'iat': now, 'exp': now + 3600, 'nonce': flow['nonce']}
    def encode(**changes):
        return jwt.encode({**claims, **changes}, key, algorithm='RS256',
                          headers={'kid': 'invented-key'})
    return flow, client, {'keys': [public]}, encode


def test_exact_authorization_contract_and_real_signature(signed):
    flow, client, jwks, encode = signed
    url = urlsplit(signin.authorization_url(flow))
    query = parse_qs(url.query)
    assert url.scheme == 'https' and url.netloc == 'auth.openai.com'
    assert url.path == '/api/accounts/authorize'
    assert query['client_id'] == ['dynamic_agent_client']
    assert query['agent_name_hint'] == ['HyMem LME']
    assert query['redirect_uri'] == ['http://127.0.0.1:1455/auth/callback']
    assert query['resource'] == ['https://api.openai.com/v1']
    assert set(query['scope'][0].split()) == set(signin.SCOPES.split())
    assert query['code_challenge_method'] == ['S256']
    assert 'code_verifier' not in query
    claims = signin.validate_id_token(encode(), jwks, client, flow['nonce'])
    assert claims['sub'] == 'invented-subject'


@pytest.mark.parametrize('change', [
    {'iss': 'https://other.invalid'}, {'aud': 'oaiapp_another_client'},
    {'aud': ['oaiapp_invented_client', 'oaiapp_another_client']},
    {'nonce': 'another-nonce'}, {'exp': 1}, {'sub': ''},
])
def test_signed_but_wrong_claims_rejected(signed, change):
    flow, client, jwks, encode = signed
    with pytest.raises(signin.SigninError):
        signin.validate_id_token(encode(**change), jwks, client, flow['nonce'])


def test_wrong_key_cannot_validate_signature(signed):
    flow, client, jwks, encode = signed
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    wrong = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(other.public_key()))
    wrong.update(kid='invented-key', use='sig', alg='RS256')
    with pytest.raises(signin.SigninError):
        signin.validate_id_token(encode(), {'keys': [wrong]}, client, flow['nonce'])


def test_identity_without_plan_permission_is_not_inference_permission(signed):
    flow, client, jwks, encode = signed
    response = {'token_type': 'Bearer', 'expires_in': 3600,
                'access_token': 'invented-access', 'refresh_token': 'invented-refresh',
                'id_token': encode(), 'scope': 'openid profile email offline_access resource.invoke'}
    record = signin.validate_token_response(response, jwks, client, flow)
    assert record['plan_usage_enabled'] is False
    response['scope'] += ' chatgpt.tokens.use.direct'
    assert signin.validate_token_response(response, jwks, client, flow)['plan_usage_enabled']


def test_documented_callback_scope_and_duplicate_state(signed):
    flow, client, _, _ = signed
    path = '/auth/callback?' + urlencode({'state': flow['state'], 'client_id': client,
        'code': 'invented-code', 'scope': signin.SCOPES})
    parsed = signin.parse_callback(path, flow)
    assert parsed['client_id'] == client and parsed['code'] == 'invented-code'
    with pytest.raises(signin.SigninError):
        signin.parse_callback(path + '&state=duplicate', flow)
    with pytest.raises(signin.SigninError, match='state_mismatch'):
        signin.parse_callback('/auth/callback?state=wrong&error=access_denied', flow)


def test_callback_client_mismatch_cannot_exchange_code(tmp_path, monkeypatch, signed):
    flow, client, _, _ = signed
    state_dir = signin._private_dir(tmp_path / 'private-state')
    signin._write_private(state_dir / 'registration.json',
                          {'host_id': flow['host_id'], 'client_id': client})
    monkeypatch.setattr(signin, '_json_request', lambda *a, **k: pytest.fail('No token exchange'))
    with pytest.raises(signin.SigninError, match='client_id_mismatch'):
        signin.complete_flow(state_dir, flow, {'client_id': 'oaiapp_unrelated_client', 'code': 'private-code'})
    assert not (state_dir / 'credential.json').exists()


def test_symlink_credential_read_rejected(tmp_path):
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    target = tmp_path / 'unrelated'
    target.write_text('{"private":"do not read"}')
    target.chmod(0o600)
    link = private / 'credential.json'
    link.symlink_to(target)
    with pytest.raises(signin.SigninError):
        signin._read_private(link)


def test_real_cli_timeout_no_network_no_secret_output(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    start = time.monotonic()
    result = subprocess.run([sys.executable, '-I', '-B', str(Path(signin.__file__)),
        '--state-dir', str(tmp_path / 'signin'), '--port', str(port), '--wait-seconds', '1'],
        capture_output=True, text=True, timeout=5)
    assert time.monotonic() - start < 5
    assert result.returncode == 1 and result.stderr == ''
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]['status'] == 'ready'
    assert rows[-1]['status'] in {'timeout', 'failed'}
    assert all(row['model_calls'] == 0 for row in rows)
    assert not (tmp_path / 'signin' / 'credential.json').exists()


def test_completed_auth_survives_browser_disconnect(tmp_path, monkeypatch):
    flow = signin.new_flow('urn:uuid:ee2e8b63-3e81-454a-a5f1-eb9b703b1431', 1455)
    calls = []

    class MockServer:
        def __init__(self, address, handler):
            self.handler = handler
            assert address == ('127.0.0.1', 1455)

        def handle_request(self):
            handler = object.__new__(self.handler)
            handler.headers = {'Host': '127.0.0.1:1455'}
            handler.path = '/auth/callback?' + urlencode({
                'state': flow['state'], 'code': 'invented-code',
                'client_id': 'oaiapp_invented_client'})
            def disconnected(*args):
                raise BrokenPipeError('invented disconnected browser')
            handler._reply = disconnected
            try:
                handler.do_GET()
            except OSError:
                self.handle_error(None, None)

        def server_close(self):
            calls.append('closed')

    def complete(state_dir, got_flow, callback):
        calls.append('completed')
        signin._write_private(state_dir / 'credential.json', {'invented': True})
        return True

    monkeypatch.setattr(signin, 'HTTPServer', MockServer)
    monkeypatch.setattr(signin, 'new_flow', lambda *args: flow)
    monkeypatch.setattr(signin, 'complete_flow', complete)
    private = tmp_path / 'state'
    result = signin._serve_locked(private, 1455, 1)
    assert calls == ['completed', 'closed']
    assert result['status'] == 'complete' and result['plan_usage_enabled'] is True
    assert signin._read_private(private / 'status.json') == result
