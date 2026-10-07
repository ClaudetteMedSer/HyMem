"""Offline security controls for the dedicated SIWC LME diagnostic."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import time
from urllib import parse
from urllib import error as url_error

from cryptography.hazmat.primitives.asymmetric import rsa
import jwt
import pytest


PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/lme_chatgpt_plan_signin_v2.py"
SPEC = importlib.util.spec_from_file_location("lme_chatgpt_plan_signin_v2", PATH)
assert SPEC and SPEC.loader
signin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(signin)


def b64(number: int) -> str:
    data = number.to_bytes((number.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


@pytest.fixture
def signed():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_numbers()
    jwks = {"keys": [{"kty": "RSA", "kid": "test-key", "use": "sig", "alg": "RS256", "n": b64(public.n), "e": b64(public.e)}]}
    now = int(time.time())
    client_id = "oaiapp_1234567890"
    nonce = "nonce-123"

    def make(*, drop=(), signing_key=None, algorithm="RS256", headers=None, **changes):
        claims = {
            "iss": signin.AUTH_ORIGIN, "aud": client_id, "sub": "subject-1",
            "email": "test@example.org", "email_verified": True,
            "exp": now + 3600, "iat": now, "nonce": nonce,
        }
        claims.update(changes)
        for name in drop:
            claims.pop(name)
        return jwt.encode(claims, signing_key or key, algorithm=algorithm, headers=headers or {"kid": "test-key"})

    make.key = key

    return make, jwks, client_id, nonce, now


def test_authorization_uses_dynamic_registration_and_pkce():
    flow = signin.new_flow("urn:uuid:90ef64f5-40a3-4b44-b07e-e394464e4103", 1455)
    url = signin.authorization_url(flow)
    parsed = parse.urlsplit(url)
    query = dict(parse.parse_qsl(parsed.query))
    assert parsed.scheme == "https" and parsed.netloc == "auth.openai.com"
    assert query["client_id"] == "dynamic_agent_client"
    assert query["agent_name_hint"] == "HyMem LME"
    assert query["ext_agent_host_id"] == flow["host_id"]
    assert query["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"
    assert query["code_challenge"] == signin.pkce_challenge(flow["verifier"])
    assert query["code_challenge_method"] == "S256"
    assert "chatgpt.tokens.use.direct" in query["scope"].split()
    assert query["resource"] == signin.RESOURCE
    assert "verifier" not in url
    flow["client_id"] = "oaiapp_1234567890"
    returning = dict(parse.parse_qsl(parse.urlsplit(signin.authorization_url(flow)).query))
    assert returning["client_id"] == "oaiapp_1234567890"
    assert "agent_name_hint" not in returning


@pytest.mark.parametrize("raw", [
    "/auth/callback?state=bad&code=c&client_id=oaiapp_1234567890",
    "/auth/callback?state=s&state=s&code=c&client_id=oaiapp_1234567890",
    "/auth/callback?state=s&code=c&code=c2&client_id=oaiapp_1234567890",
    "/auth/callback?state=s&code=c&client_id=dynamic_agent_client",
    "/auth/callback?state=s&error=access_denied",
    "/elsewhere?state=s&code=c&client_id=oaiapp_1234567890",
])
def test_callback_rejects_invalid_or_replayed_inputs(raw):
    with pytest.raises(signin.SigninError):
        signin.parse_callback(raw, {"state": "s"})


def test_callback_accepts_single_code_and_issued_client():
    assert signin.parse_callback("/auth/callback?state=s&code=abc&client_id=oaiapp_1234567890", {"state": "s"}) == {
        "code": "abc", "client_id": "oaiapp_1234567890",
    }


@pytest.mark.parametrize("change", [
    {"iss": "https://other.example"},
    {"aud": "oaiapp_other12345"},
    {"nonce": "wrong"},
    {"exp": 1},
    {"email_verified": "true"},
    {"email": 1},
])
def test_id_token_rejects_wrong_claims(signed, change):
    make, jwks, client_id, nonce, now = signed
    with pytest.raises(signin.SigninError):
        signin.validate_id_token(make(**change), jwks, client_id, nonce, now)


def test_id_token_rejects_wrong_signature_and_external_key_header(signed):
    make, jwks, client_id, nonce, now = signed
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    claims = jwt.decode(make(), options={"verify_signature": False})
    forged = jwt.encode(claims, other, algorithm="RS256", headers={"kid": "test-key"})
    with pytest.raises(signin.SigninError):
        signin.validate_id_token(forged, jwks, client_id, nonce, now)
    forged = jwt.encode(claims, other, algorithm="RS256", headers={"kid": "test-key", "jku": "https://attacker.example/jwks"})
    with pytest.raises(signin.SigninError):
        signin.validate_id_token(forged, jwks, client_id, nonce, now)


def test_scope_gate_keeps_identity_but_disables_plan_usage(signed):
    make, jwks, client_id, nonce, now = signed
    flow = {"host_id": "urn:uuid:90ef64f5-40a3-4b44-b07e-e394464e4103", "nonce": nonce}
    response = {
        "token_type": "Bearer", "expires_in": 3600, "access_token": "access-test",
        "refresh_token": "refresh-test", "id_token": make(),
        "scope": "openid profile email offline_access resource.invoke",
    }
    record = signin.validate_token_response(response, jwks, client_id, flow, now)
    assert record["subject"] == "subject-1"
    assert record["plan_usage_enabled"] is False
    response["scope"] += " chatgpt.tokens.use.direct"
    assert signin.validate_token_response(response, jwks, client_id, flow, now)["plan_usage_enabled"] is True
    response["scope"] = "openid offline_access resource.invoke chatgpt.tokens.use.direct"
    assert signin.validate_token_response(response, jwks, client_id, flow, now)["plan_usage_enabled"] is True
    response["scope"] = "openid offline_access chatgpt.tokens.use.direct"
    assert signin.validate_token_response(response, jwks, client_id, flow, now)["plan_usage_enabled"] is False


@pytest.mark.parametrize("bad_type", [None, 1, [], {"value": "Bearer"}])
def test_malformed_token_type_is_bounded_denial(bad_type):
    with pytest.raises(signin.SigninError, match="token_type_invalid"):
        signin.validate_token_response({"token_type": bad_type}, {}, "oaiapp_1234567890", {})


@pytest.mark.parametrize("body", [b'{"error":[]}', b'{"error":{}}', b'[]', b'{"error":"invalid_grant"}'])
def test_http_error_metadata_is_bounded(body, monkeypatch):
    class Opener:
        def open(self, req, timeout):
            raise url_error.HTTPError(signin.TOKEN_URL, 400, "ignored", {}, io.BytesIO(body))

    monkeypatch.setattr(signin.request, "build_opener", lambda *_args: Opener())
    expected = "oauth_invalid_grant" if b"invalid_grant" in body else "upstream_rejected"
    with pytest.raises(signin.SigninError, match=expected):
        signin._json_request(signin.TOKEN_URL, {"grant_type": "authorization_code"})


def test_browser_disconnect_after_credential_save_keeps_complete(tmp_path, monkeypatch):
    state_dir = signin._private_dir(tmp_path / "private")
    flow = signin.new_flow("urn:uuid:90ef64f5-40a3-4b44-b07e-e394464e4103", 1455)
    monkeypatch.setattr(signin, "new_flow", lambda *_args: flow)

    def finish(directory, _flow, _callback):
        signin._write_private(directory / "credential.json", {"host_id": flow["host_id"], "client_id": "oaiapp_1234567890"})
        return True

    monkeypatch.setattr(signin, "complete_flow", finish)

    class FakeServer:
        def __init__(self, _address, handler_type):
            self.handler_type = handler_type

        def handle_request(self):
            handler = object.__new__(self.handler_type)
            handler.headers = {"Host": "127.0.0.1:1455"}
            handler.path = "/auth/callback?" + parse.urlencode({
                "state": flow["state"], "code": "invented-code", "client_id": "oaiapp_1234567890",
            })
            def broken_reply(status, _body):
                assert status == 200
                raise BrokenPipeError()
            handler._reply = broken_reply
            try:
                handler.do_GET()
            except BrokenPipeError:
                self.handle_error(None, None)

        def server_close(self):
            pass

    monkeypatch.setattr(signin, "HTTPServer", FakeServer)
    result = signin._serve_locked(state_dir, 1455, 1)
    assert result == {"status": "complete", "plan_usage_enabled": True, "model_calls": 0}
    assert signin._read_private(state_dir / "status.json") == result
    assert signin._read_private(state_dir / "credential.json")["client_id"] == "oaiapp_1234567890"


def test_private_state_and_identity_conflict(tmp_path):
    directory = signin._private_dir(tmp_path / "private")
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    first = signin.load_host_id(directory)
    assert first == signin.load_host_id(directory)
    assert stat.S_IMODE((directory / "host.json").stat().st_mode) == 0o600
    signin._write_private(directory / "registration.json", {"host_id": first, "client_id": "oaiapp_1234567890"})
    signin._write_private(directory / "credential.json", {"host_id": first, "client_id": "oaiapp_1234567890"})
    flow = {"host_id": first, "port": 1455, "verifier": "v"}
    with pytest.raises(signin.SigninError, match="credential_already_exists"):
        signin.complete_flow(directory, flow, {"code": "c", "client_id": None})
    signin._write_private(directory / "credential.json", {"host_id": first, "client_id": "oaiapp_other12345"}, replace=True)
    with pytest.raises(signin.SigninError, match="credential_identity_conflict"):
        signin.complete_flow(directory, flow, {"code": "c", "client_id": None})


def test_symlink_and_public_file_rejected(tmp_path):
    private = signin._private_dir(tmp_path / "private")
    target = tmp_path / "target.json"
    target.write_text("{}")
    (private / "host.json").symlink_to(target)
    with pytest.raises(signin.SigninError, match="state_file_not_private"):
        signin.load_host_id(private)
    link = tmp_path / "link"
    link.symlink_to(private, target_is_directory=True)
    with pytest.raises(signin.SigninError, match="state_dir_symlink"):
        signin._private_dir(link)


def test_no_network_before_valid_client_registration(tmp_path, monkeypatch):
    directory = signin._private_dir(tmp_path / "private")
    host_id = signin.load_host_id(directory)
    flow = {"host_id": host_id, "port": 1455, "verifier": "v", "nonce": "n"}
    monkeypatch.setattr(signin, "_json_request", lambda *_args, **_kwargs: pytest.fail("network reached"))
    with pytest.raises(signin.SigninError, match="client_id_missing"):
        signin.complete_flow(directory, flow, {"code": "c", "client_id": None})
    assert not (directory / "registration.json").exists()


def test_issued_client_persisted_before_exchange(tmp_path, monkeypatch):
    directory = signin._private_dir(tmp_path / "private")
    host_id = signin.load_host_id(directory)
    flow = {"host_id": host_id, "port": 1455, "verifier": "v", "nonce": "n"}

    def fail_exchange(url, form):
        registration = signin._read_private(directory / "registration.json")
        assert registration == {"host_id": host_id, "client_id": "oaiapp_1234567890"}
        assert form["client_id"] == "oaiapp_1234567890"
        raise signin.SigninError("upstream_unavailable")

    monkeypatch.setattr(signin, "_json_request", fail_exchange)
    with pytest.raises(signin.SigninError, match="upstream_unavailable"):
        signin.complete_flow(directory, flow, {"code": "c", "client_id": "oaiapp_1234567890"})


def test_existing_registration_rejects_other_client_before_network(tmp_path, monkeypatch):
    directory = signin._private_dir(tmp_path / "private")
    host_id = signin.load_host_id(directory)
    signin._write_private(directory / "registration.json", {"host_id": host_id, "client_id": "oaiapp_1234567890"})
    monkeypatch.setattr(signin, "_json_request", lambda *_args, **_kwargs: pytest.fail("network reached"))
    with pytest.raises(signin.SigninError, match="client_id_mismatch"):
        signin.complete_flow(directory, {"host_id": host_id}, {"code": "c", "client_id": "oaiapp_other12345"})


@pytest.mark.parametrize(("changes", "drop", "expected"), [
    ({"iss": "https://other.example"}, (), "id_token_issuer_invalid"),
    ({"aud": "oaiapp_other12345"}, (), "id_token_audience_mismatch"),
    ({"aud": ["oaiapp_1234567890"]}, (), "id_token_audience_format_invalid"),
    ({"exp": 1}, (), "id_token_expired"),
    ({"nbf": 4102444800}, (), "id_token_not_before_future"),
    ({"iat": 4102444800}, (), "id_token_issued_at_future"),
    ({"iat": "not-an-integer"}, (), "id_token_issued_at_invalid"),
    ({"sub": 123}, (), "id_token_subject_invalid"),
    ({"jti": 123}, (), "id_token_jti_invalid"),
    *[({}, (claim,), f"id_token_claim_missing_{claim}") for claim in ("exp", "iat", "iss", "aud", "sub", "nonce")],
])
def test_signed_jwt_failure_categories(signed, changes, drop, expected):
    make, jwks, client_id, nonce, now = signed
    token = make(drop=drop, **changes)
    with pytest.raises(signin.SigninError) as failure:
        signin.validate_id_token(token, jwks, client_id, nonce, now)
    assert failure.value.code == expected


def test_signed_jwt_signature_and_decode_categories(signed):
    make, jwks, client_id, nonce, now = signed
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = make(signing_key=other)
    with pytest.raises(signin.SigninError) as failure:
        signin.validate_id_token(forged, jwks, client_id, nonce, now)
    assert failure.value.code == "id_token_signature_invalid"

    # Construct a correctly signed JWT whose payload is not JSON.
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    header = base64.urlsafe_b64encode(b'{"alg":"RS256","kid":"test-key"}').rstrip(b"=")
    payload = base64.urlsafe_b64encode(b"not-json").rstrip(b"=")
    signing_input = header + b"." + payload
    signature = make.key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    malformed = (signing_input + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")).decode()
    with pytest.raises(signin.SigninError) as failure:
        signin.validate_id_token(malformed, jwks, client_id, nonce, now)
    assert failure.value.code == "id_token_decode_invalid"


def test_bounded_other_jwt_and_internal_categories(signed, monkeypatch):
    make, jwks, client_id, nonce, now = signed
    token = make()
    # These PyJWT errors depend on library internals or malformed key material.
    # Keep a real signed input while forcing each remaining exception seam.
    for error, expected in (
        (jwt.exceptions.InvalidAlgorithmError("secret algorithm detail"), "id_token_algorithm_invalid"),
        (jwt.exceptions.InvalidKeyError("secret key detail"), "id_token_key_invalid"),
        (jwt.exceptions.InvalidAudienceError("secret audience detail"), "id_token_audience_invalid"),
        (jwt.exceptions.ImmatureSignatureError("secret time detail"), "id_token_not_yet_valid"),
        (jwt.exceptions.MissingRequiredClaimError("untrusted_claim"), "id_token_claim_missing_other"),
        (RuntimeError("secret internal detail"), "id_token_internal_error"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(signin.jwt, "decode", lambda *_args, _error=error, **_kwargs: (_ for _ in ()).throw(_error))
            with pytest.raises(signin.SigninError) as failure:
                signin.validate_id_token(token, jwks, client_id, nonce, now)
        assert failure.value.code == expected
        assert "secret" not in str(failure.value)
        assert "untrusted_claim" not in str(failure.value)


def test_v1_v2_signed_token_acceptance_is_identical(signed):
    spec = importlib.util.spec_from_file_location("lme_chatgpt_plan_signin_v1_for_comparison", PATH.with_name("lme_chatgpt_plan_signin_v1.py"))
    assert spec and spec.loader
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    make, jwks, client_id, nonce, now = signed
    tokens = [
        make(), make(iss="https://other.example"), make(aud="oaiapp_other12345"),
        make(aud=[client_id]), make(exp=1), make(nbf=4102444800),
        make(iat="bad"), make(sub=123), make(jti=123),
        make(nonce="wrong"), make(email=7), make(email_verified="yes"),
        *(make(drop=(claim,)) for claim in ("exp", "iat", "iss", "aud", "sub", "nonce")),
    ]
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    tokens.append(make(signing_key=other))
    for token in tokens:
        decisions = []
        for module in (original, signin):
            try:
                module.validate_id_token(token, jwks, client_id, nonce, now)
                decisions.append(True)
            except module.SigninError:
                decisions.append(False)
        assert decisions[0] == decisions[1]
    assert decisions == [False, False]


def test_invalid_signed_token_keeps_existing_registration_without_credentials(tmp_path, signed, monkeypatch):
    make, jwks, client_id, nonce, now = signed
    directory = signin._private_dir(tmp_path / "private")
    host_id = signin.load_host_id(directory)
    registration = {"host_id": host_id, "client_id": client_id}
    signin._write_private(directory / "registration.json", registration)
    flow = {"host_id": host_id, "port": 1455, "verifier": "verifier", "nonce": nonce}
    token = make(iss="https://other.example")
    response = {"token_type": "Bearer", "expires_in": 3600, "access_token": "a", "refresh_token": "r", "id_token": token,
                "scope": "openid offline_access resource.invoke chatgpt.tokens.use.direct"}
    monkeypatch.setattr(signin, "_json_request", lambda url, *_args: response if url == signin.TOKEN_URL else jwks)
    with pytest.raises(signin.SigninError) as failure:
        signin.complete_flow(directory, flow, {"code": "invented-code", "client_id": None})
    assert failure.value.code == "id_token_issuer_invalid"
    assert signin._read_private(directory / "registration.json") == registration
    assert not (directory / "credential.json").exists()
    response["id_token"] = make()
    assert signin.complete_flow(directory, flow, {"code": "invented-code", "client_id": None}) is True
    assert signin._read_private(directory / "credential.json")["plan_usage_enabled"] is True
