"""Offline, invented-state controls for serialized SIWC renewal."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa


SOURCE = Path(__file__).resolve().parents[1] / "tools/diagnostics/lme_chatgpt_plan_refresh_v1.py"
spec = importlib.util.spec_from_file_location("lme_refresh_under_test", SOURCE)
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name).resolve() / "private"
        self.state.mkdir(mode=0o700)
        self.now = int(time.time())
        self.host = "urn:uuid:88aaa7c0-4f3b-4c62-843a-925831874f82"
        self.client = "oaiapp_abcdefgh12345678"
        self.subject = "invented-subject"
        self.scope = ["openid", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"]
        self.files = {
            "status.json": {"status": "complete", "plan_usage_enabled": True, "model_calls": 0},
            "host.json": {"ext_agent_host_id": self.host},
            "registration.json": {"host_id": self.host, "client_id": self.client},
            "credential.json": {"version": 1, "issuer": "https://auth.openai.com", "host_id": self.host,
                                "client_id": self.client, "subject": self.subject,
                                "access_token": "invented-access-old", "refresh_token": "invented-refresh-old",
                                "id_token": "invented-id-old", "token_type": "Bearer", "scope": self.scope,
                                "saved_at": self.now - 3500, "expires_at": self.now - 1,
                                "plan_usage_enabled": True},
        }
        self.save()

    def save(self):
        for name, data in self.files.items():
            path = self.state / name
            path.write_text(json.dumps(data), encoding="utf-8")
            path.chmod(0o600)

    def response(self, **updates):
        value = {"token_type": "Bearer", "expires_in": 3600,
                 "access_token": "invented-access-new", "refresh_token": "invented-refresh-new"}
        value.update(updates)
        return value

    def test_source_hashes_check_only_and_not_due(self):
        self.assertEqual(hashlib.sha256(SOURCE.with_name("lme_chatgpt_plan_signin_v3.py").read_bytes()).hexdigest(),
                         refresh.SIGNIN_SHA256)
        self.assertEqual(hashlib.sha256(SOURCE.with_name("lme_chatgpt_plan_catalog_v1.py").read_bytes()).hexdigest(),
                         refresh.CATALOG_SHA256)
        with mock.patch.object(refresh, "_http_json") as network:
            due = refresh.run(self.state, check_only=True)
            self.assertEqual(due["status"], "due")
            self.assertEqual(due["refresh_requests"], 0)
            self.assertEqual(sorted(p.name for p in self.state.iterdir()), sorted(self.files))
            network.assert_not_called()
            self.files["credential.json"]["expires_at"] = self.now + 700
            self.files["credential.json"]["saved_at"] = self.now - 100
            self.save()
            result = refresh.run(self.state)
            self.assertEqual(result["status"], "not_due")
            network.assert_not_called()

    def test_refresh_exact_form_rotation_and_generation_replay(self):
        before = (self.state / "credential.json").read_bytes()
        def exchange(url, form=None):
            self.assertEqual(url, "https://auth.openai.com/api/accounts/oauth/token")
            self.assertEqual(form, {"grant_type": "refresh_token", "client_id": self.client,
                                    "refresh_token": "invented-refresh-old",
                                    "resource": "https://api.openai.com/v1"})
            self.assertNotIn("scope", form)
            self.assertEqual((self.state / "credential.json").read_bytes(), before)
            self.assertEqual(len(list(self.state.glob(".refresh-attempt-*.json"))), 1)
            return self.response(earliest_refresh_at="opaque privately retained")
        with mock.patch.object(refresh, "_http_json", side_effect=exchange) as network:
            result = refresh.run(self.state)
        self.assertEqual(result["status"], "refreshed")
        self.assertEqual(result["refresh_requests"], 1)
        self.assertEqual(network.call_count, 1)
        updated = json.loads((self.state / "credential.json").read_text())
        self.assertEqual(updated["subject"], self.subject)
        self.assertEqual(updated["id_token"], "invented-id-old")
        self.assertEqual(updated["refresh_token"], "invented-refresh-new")
        self.assertEqual(updated["earliest_refresh_at"], "opaque privately retained")
        self.assertEqual((self.state / "credential.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads((self.state / "status.json").read_text()), self.files["status.json"])
        # Restoring the old generation still cannot send its rotating token again.
        (self.state / "credential.json").write_bytes(before)
        with mock.patch.object(refresh, "_http_json") as network:
            second = refresh.run(self.state)
        self.assertEqual(second["error"], "generation_consumed")
        network.assert_not_called()

    def test_failure_is_atomic_and_generation_consumed(self):
        before = (self.state / "credential.json").read_bytes()
        with mock.patch.object(refresh, "_http_json", return_value=self.response(scope="openid offline_access")):
            result = refresh.run(self.state)
        self.assertEqual(result["error"], "scope_changed")
        self.assertEqual(result["refresh_requests"], 1)
        self.assertEqual((self.state / "credential.json").read_bytes(), before)
        with mock.patch.object(refresh, "_http_json") as network:
            second = refresh.run(self.state)
        self.assertEqual(second["error"], "generation_consumed")
        network.assert_not_called()

    def test_private_binding_and_expiry_rejections_without_network(self):
        cases = [
            ("status.json", "status", "failed"), ("host.json", "ext_agent_host_id", "urn:uuid:wrong"),
            ("registration.json", "client_id", "oaiapp_wrong"),
            ("credential.json", "client_id", "oaiapp_wrong"),
            ("credential.json", "issuer", "https://elsewhere.invalid"),
            ("credential.json", "subject", ""), ("credential.json", "refresh_token", ""),
            ("credential.json", "access_token", ""), ("credential.json", "id_token", ""),
            ("credential.json", "scope", ["openid", "offline_access"]),
            ("credential.json", "expires_at", self.now - 3500),
            ("credential.json", "saved_at", self.now + 100),
        ]
        for filename, key, bad in cases:
            with self.subTest(filename=filename, key=key):
                old = self.files[filename][key]
                self.files[filename][key] = bad
                self.save()
                with mock.patch.object(refresh, "_http_json") as network:
                    result = refresh.run(self.state)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["refresh_requests"], 0)
                network.assert_not_called()
                self.files[filename][key] = old
                self.save()

    def test_nofollow_private_source_and_lock(self):
        with mock.patch.object(refresh, "SIGNIN_SHA256", "0" * 64), mock.patch.object(refresh, "_http_json") as network:
            self.assertEqual(refresh.run(self.state)["error"], "source_mismatch")
            network.assert_not_called()
        path = self.state / "credential.json"
        path.chmod(0o644)
        self.assertEqual(refresh.run(self.state, check_only=True)["error"], "state_file_not_private")
        path.chmod(0o600)
        path.unlink()
        path.symlink_to(self.state / "host.json")
        self.assertEqual(refresh.run(self.state, check_only=True)["error"], "state_file_unavailable")
        path.unlink()
        self.save()
        import fcntl
        lock = os.open(self.state / ".flow.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(refresh, "_http_json") as network:
                self.assertEqual(refresh.run(self.state)["error"], "flow_already_running")
                network.assert_not_called()
        finally:
            os.close(lock)

    def test_strict_response_and_http_transport(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]', b'{'):
            with self.subTest(raw=raw), self.assertRaises(refresh.RefreshError):
                refresh._json(raw)
        class Reply:
            status = 200
            def __init__(self, raw): self.raw = raw
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, limit): return self.raw[:limit]
        opener = mock.Mock()
        opener.open.return_value = Reply(json.dumps(self.response()).encode())
        with mock.patch.object(refresh.request, "build_opener", return_value=opener):
            value = refresh._http_json("https://auth.openai.com/api/accounts/oauth/token", {
                "grant_type": "refresh_token", "client_id": self.client,
                "refresh_token": "secret", "resource": "https://api.openai.com/v1"})
        self.assertEqual(value["access_token"], "invented-access-new")
        req = opener.open.call_args.args[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 15)
        self.assertNotIn(b"scope=", req.data)
        self.assertEqual(req.get_header("Content-type"), "application/x-www-form-urlencoded")
        redirect = refresh._NoRedirect()
        with self.assertRaises(refresh.RefreshError) as caught:
            redirect.redirect_request(None, None, 302, "secret", {}, "https://evil.invalid")
        self.assertEqual(caught.exception.code, "unexpected_redirect")

    def test_error_export_boundary_rejects_mutated_codes(self):
        for code in (["invented-secret"], "invented-secret"):
            with self.subTest(code=type(code).__name__), mock.patch.object(
                    refresh, "_sources", side_effect=refresh.RefreshError(code)):
                result = refresh.run(self.state, check_only=True)
            self.assertEqual(result["error"], "internal_error")
            self.assertNotIn("invented-secret", json.dumps(result))

    def test_signed_new_id_token_exact_identity(self):
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
        jwk.update(kid="test-key", use="sig")
        jwks = {"keys": [jwk]}
        def token(**changes):
            claims = {"iss": "https://auth.openai.com", "aud": [self.client, "other"],
                      "azp": self.client, "sub": self.subject, "iat": self.now,
                      "exp": self.now + 3600}
            claims.update(changes)
            return jwt.encode(claims, private, algorithm="RS256", headers={"kid": "test-key"})
        refresh._new_id_token(token(), jwks, self.client, self.subject, self.now)
        for changes in ({"sub": "different"}, {"aud": "different"},
                        {"azp": "different"}, {"iss": "https://elsewhere.invalid"},
                        {"exp": self.now - 1}):
            with self.subTest(changes=changes), self.assertRaises(refresh.RefreshError):
                refresh._new_id_token(token(**changes), jwks, self.client, self.subject, self.now)
        with mock.patch.object(refresh, "_http_json", return_value=jwks) as network:
            signin, _ = refresh._sources()
            signed = token()
            updated = refresh._validated_new(self.response(id_token=signed), self.files["credential.json"],
                                             signin, self.now)
        self.assertEqual(updated["id_token"], signed)
        network.assert_called_once_with("https://auth.openai.com/.well-known/jwks.json")
        calls = []
        def exchange(url, form=None):
            calls.append((url, form))
            return self.response(id_token=signed) if form is not None else jwks
        with mock.patch.object(refresh, "_http_json", side_effect=exchange):
            result = refresh.run(self.state)
        self.assertEqual(result["status"], "refreshed")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "https://auth.openai.com/api/accounts/oauth/token")
        self.assertEqual(calls[1], ("https://auth.openai.com/.well-known/jwks.json", None))
        stored = json.loads((self.state / "credential.json").read_text())
        self.assertEqual(stored["id_token"], signed)
        self.assertEqual(stored["subject"], self.subject)


if __name__ == "__main__":
    unittest.main()
