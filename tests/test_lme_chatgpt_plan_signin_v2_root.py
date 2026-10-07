"""Independent failure attribution and unchanged acceptance checks; no live tokens."""
import json
import time

from cryptography.hazmat.primitives.asymmetric import rsa
import jwt
import pytest

from tools.diagnostics import lme_chatgpt_plan_signin_v1 as old
from tools.diagnostics import lme_chatgpt_plan_signin_v2 as new


@pytest.fixture
def material():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid='root-invented-key', use='sig')
    now = int(time.time())
    claims = dict(iss=old.AUTH_ORIGIN, aud='oaiapp_root_invented',
                  sub='invented-subject', nonce='invented-nonce', iat=now, exp=now+3600)
    return key, {'keys': [jwk]}, claims


@pytest.mark.parametrize('case,expected', [
    ('issuer', 'id_token_issuer_invalid'),
    ('audience', 'id_token_audience_mismatch'),
    ('array', 'id_token_audience_format_invalid'),
    ('expired', 'id_token_expired'),
    ('future', 'id_token_issued_at_future'),
    ('missing_nonce', 'id_token_claim_missing_nonce'),
    ('nonce', 'nonce_mismatch'),
    ('signature', 'id_token_signature_invalid'),
    ('subject', 'id_token_subject_invalid'),
])
def test_precise_failure_without_changed_acceptance(material, case, expected):
    key, jwks, claims = material
    client, nonce = claims['aud'], claims['nonce']
    if case == 'issuer': claims['iss'] = 'https://unrelated.invalid'
    elif case == 'audience': claims['aud'] = 'oaiapp_unrelated_client'
    elif case == 'array': claims['aud'] = [client]
    elif case == 'expired': claims['exp'] = 1
    elif case == 'future': claims['iat'] += 120
    elif case == 'missing_nonce': del claims['nonce']
    elif case == 'nonce': claims['nonce'] = 'private-unmatched-nonce'
    elif case == 'subject': claims['sub'] = 10
    elif case == 'signature': key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode(claims, key, algorithm='RS256', headers={'kid':'root-invented-key'})
    with pytest.raises(old.SigninError): old.validate_id_token(token, jwks, client, nonce)
    with pytest.raises(new.SigninError) as got: new.validate_id_token(token, jwks, client, nonce)
    assert got.value.code == expected
    assert str(got.value) == expected
    assert token not in str(got.value)


def test_valid_signature_and_claims_identical(material):
    key, jwks, claims = material
    token = jwt.encode(claims, key, algorithm='RS256', headers={'kid':'root-invented-key'})
    assert new.validate_id_token(token, jwks, claims['aud'], claims['nonce']) == old.validate_id_token(
        token, jwks, claims['aud'], claims['nonce'])


def test_failed_validation_never_saves_credential(tmp_path, monkeypatch, material):
    key, jwks, claims = material
    flow = new.new_flow('urn:uuid:7b25a57e-3053-4d8d-a10a-f711ffca7b15', 1455)
    claims['nonce'] = flow['nonce']
    claims['iss'] = 'https://unrelated.invalid'
    token = jwt.encode(claims, key, algorithm='RS256', headers={'kid':'root-invented-key'})
    response = dict(token_type='Bearer', expires_in=3600, access_token='invented-access',
                    refresh_token='invented-refresh', id_token=token, scope=new.SCOPES)
    private = new._private_dir(tmp_path / 'private')
    calls = []
    def exchange(url, form=None):
        calls.append(url)
        return response if form is not None else jwks
    monkeypatch.setattr(new, '_json_request', exchange)
    with pytest.raises(new.SigninError, match='id_token_issuer_invalid'):
        new.complete_flow(private, flow, {'code':'invented-code', 'client_id':claims['aud']})
    assert calls == [new.TOKEN_URL, new.JWKS_URL]
    assert not (private / 'credential.json').exists()
    assert (private / 'registration.json').exists()
