"""Root-owned audience binding regression tests; invented signed tokens only."""
import json
import time

from cryptography.hazmat.primitives.asymmetric import rsa
import jwt
import pytest

from tools.diagnostics import lme_chatgpt_plan_signin_v2 as prior
from tools.diagnostics import lme_chatgpt_plan_signin_v3 as current


CLIENT = 'oaiapp_root_invented'
OTHER = 'oaiapp_other_invented'


@pytest.fixture
def issued():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid='invented-root-key', use='sig')
    flow = current.new_flow('urn:uuid:72ba4f70-c16d-4160-943c-8f13104e7085', 1455)
    claims = dict(iss=current.AUTH_ORIGIN, aud=CLIENT, sub='invented-subject',
                  nonce=flow['nonce'], exp=int(time.time())+3600, iat=int(time.time()))
    def token(**changes):
        return jwt.encode({**claims, **changes}, key, algorithm='RS256',
                          headers={'kid':'invented-root-key'})
    return token, {'keys':[jwk]}, flow


@pytest.mark.parametrize('extra', [
    {'aud':CLIENT}, {'aud':[CLIENT]}, {'aud':[CLIENT], 'azp':CLIENT},
    {'aud':[CLIENT, OTHER], 'azp':CLIENT}, {'aud':[OTHER, CLIENT], 'azp':CLIENT},
])
def test_verified_expected_recipient_valid_forms(issued, extra):
    token, jwks, flow = issued
    value = token(**extra)
    assert current.validate_id_token(value, jwks, CLIENT, flow['nonce'])['sub'] == 'invented-subject'
    if isinstance(extra['aud'], list):
        with pytest.raises(prior.SigninError, match='id_token_audience_format_invalid'):
            prior.validate_id_token(value, jwks, CLIENT, flow['nonce'])


@pytest.mark.parametrize('extra', [
    {'aud':OTHER}, {'aud':[OTHER]}, {'aud':[]}, {'aud':{}}, {'aud':3},
    {'aud':[CLIENT, 3]}, {'aud':[CLIENT, '']}, {'aud':[CLIENT, CLIENT]},
    {'aud':[CLIENT, OTHER]}, {'aud':[CLIENT, OTHER], 'azp':OTHER},
    {'aud':CLIENT, 'azp':OTHER}, {'aud':[CLIENT], 'azp':None},
    {'aud':[CLIENT], 'azp':[]}, {'aud':[CLIENT], 'azp':True},
    {'aud':[CLIENT, 'x'*513], 'azp':CLIENT},
    {'aud':[CLIENT]+[f'other-{i}' for i in range(16)], 'azp':CLIENT},
    {'aud':[CLIENT], 'iss':'https://unrelated.invalid'},
    {'aud':[CLIENT], 'nonce':'private-unmatched-nonce'},
    {'aud':[CLIENT], 'exp':1}, {'aud':[CLIENT], 'iat':4102444800},
])
def test_malformed_wrong_recipient_and_other_checks_still_reject(issued, extra):
    token, jwks, flow = issued
    with pytest.raises(current.SigninError):
        current.validate_id_token(token(**extra), jwks, CLIENT, flow['nonce'])


def test_signature_still_required_for_array(issued):
    token, jwks, flow = issued
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    wrong = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(other_key.public_key()))
    wrong.update(kid='invented-root-key', use='sig')
    with pytest.raises(current.SigninError, match='id_token_signature_invalid'):
        current.validate_id_token(token(aud=[CLIENT]), {'keys':[wrong]}, CLIENT, flow['nonce'])


@pytest.mark.parametrize('valid', [True, False])
def test_full_exchange_and_persistence_keeps_registration_binding(tmp_path, monkeypatch, issued, valid):
    token, jwks, flow = issued
    private = current._private_dir(tmp_path / 'signin')
    registration = {'host_id':flow['host_id'], 'client_id':CLIENT}
    current._write_private(private / 'registration.json', registration)
    response = dict(token_type='Bearer', expires_in=3600, access_token='invented-access',
                    refresh_token='invented-refresh', scope=current.SCOPES,
                    id_token=token(aud=[CLIENT if valid else OTHER]))
    calls = []
    def exchange(url, form=None):
        calls.append(url)
        if form is not None:
            assert form['client_id'] == CLIENT
            return response
        return jwks
    monkeypatch.setattr(current, '_json_request', exchange)
    if valid:
        assert current.complete_flow(private, flow, {'client_id':None, 'code':'invented-code'})
        saved = current._read_private(private / 'credential.json')
        assert saved['client_id'] == CLIENT and saved['plan_usage_enabled'] is True
    else:
        with pytest.raises(current.SigninError):
            current.complete_flow(private, flow, {'client_id':None, 'code':'invented-code'})
        assert not (private / 'credential.json').exists()
    assert current._read_private(private / 'registration.json') == registration
    assert calls == [current.TOKEN_URL, current.JWKS_URL]
