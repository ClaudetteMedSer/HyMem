"""Offline controls for the one-shot SIWC catalog checker."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "tools/diagnostics/lme_chatgpt_plan_catalog_v1.py"
spec = importlib.util.spec_from_file_location("lme_catalog_under_test", SOURCE)
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name).resolve() / "private"
        self.state.mkdir(mode=0o700)
        self.host_id = "urn:uuid:88aaa7c0-4f3b-4c62-843a-925831874f82"
        self.client_id = "oaiapp_abcdefgh12345678"
        self.token = "invented-access-token"
        self.files = {
            "status.json": {"status": "complete", "plan_usage_enabled": True, "model_calls": 0},
            "host.json": {"ext_agent_host_id": self.host_id},
            "registration.json": {"host_id": self.host_id, "client_id": self.client_id},
            "credential.json": {"version": 1, "issuer": "https://auth.openai.com", "host_id": self.host_id,
                                "client_id": self.client_id, "subject": "invented-subject", "access_token": self.token,
                                "token_type": "Bearer", "scope": ["openid", "offline_access", "resource.invoke",
                                                                 "chatgpt.tokens.use.direct"], "saved_at": 1000,
                                "expires_at": 2000, "plan_usage_enabled": True},
        }
        self._save()

    def _save(self):
        for name, value in self.files.items():
            path = self.state / name
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o600)

    def _check(self):
        with mock.patch.object(catalog.time, "time", return_value=1200), mock.patch.object(
                catalog, "_catalog_get", return_value=(True, 2)) as network:
            result = catalog.check(self.state)
        return result, network

    def test_frozen_source_and_success(self):
        signin = SOURCE.with_name("lme_chatgpt_plan_signin_v3.py")
        self.assertEqual(hashlib.sha256(signin.read_bytes()).hexdigest(), catalog.SIGNIN_SHA256)
        result, network = self._check()
        self.assertEqual(result, {"status": "available", "available": True, "http_status": 200,
                                  "catalog_count": 2, "model_slug": "gpt-6-luna", "model_calls": 0,
                                  "catalog_requests": 1})
        network.assert_called_once_with(self.token)

    def test_local_rejections_never_network(self):
        cases = [
            ("status.json", "status", "failed"), ("status.json", "plan_usage_enabled", False),
            ("status.json", "model_calls", 1), ("registration.json", "host_id", "urn:uuid:wrong"),
            ("registration.json", "client_id", "oaiapp_wrong"),
            ("credential.json", "client_id", "oaiapp_wrong"),
            ("credential.json", "host_id", "urn:uuid:wrong"),
            ("credential.json", "version", 2), ("credential.json", "issuer", "https://elsewhere.invalid"),
            ("credential.json", "token_type", "Basic"), ("credential.json", "plan_usage_enabled", False),
            ("credential.json", "expires_at", 1260), ("credential.json", "access_token", ""),
            ("credential.json", "subject", ""), ("credential.json", "scope", ["openid", "offline_access"]),
        ]
        for filename, key, bad in cases:
            with self.subTest(filename=filename, key=key):
                old = self.files[filename][key]
                self.files[filename][key] = bad
                self._save()
                result, network = self._check()
                self.assertFalse(result["available"])
                self.assertEqual(result["catalog_requests"], 0)
                network.assert_not_called()
                self.files[filename][key] = old
                self._save()

    def test_permissions_symlink_and_duplicate_json(self):
        target = self.state / "credential.json"
        target.chmod(0o644)
        result, network = self._check()
        self.assertEqual(result["error"], "state_file_not_private")
        network.assert_not_called()
        target.chmod(0o600)
        target.unlink()
        target.symlink_to(self.state / "host.json")
        result, network = self._check()
        self.assertEqual(result["error"], "state_file_unavailable")
        network.assert_not_called()
        target.unlink()
        target.write_text('{"version":1,"version":1}')
        target.chmod(0o600)
        result, network = self._check()
        self.assertEqual(result["error"], "state_file_invalid")
        network.assert_not_called()

    def test_private_directory_and_source_pin(self):
        self.state.chmod(0o755)
        result, network = self._check()
        self.assertEqual(result["error"], "state_dir_not_private")
        network.assert_not_called()
        self.state.chmod(0o700)
        with mock.patch.object(catalog, "SIGNIN_SHA256", "0" * 64):
            result, network = self._check()
        self.assertEqual(result["error"], "source_mismatch")
        network.assert_not_called()

    def test_catalog_exact_singleton_and_malformed_fail_closed(self):
        def fetch(payload, status=200):
            class Reply:
                def __enter__(self): return self
                def __exit__(self, *_): return False
                def read(self, _size): return payload
            reply = Reply()
            reply.status = status
            opener = mock.Mock()
            opener.open.return_value = reply
            with mock.patch.object(catalog.request, "build_opener", return_value=opener):
                return catalog._catalog_get(self.token), opener

        value, opener = fetch(b'{"models":[{"slug":"gpt-6-luna","visibility":"list"}]}')
        self.assertEqual(value, (True, 1))
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, catalog.CATALOG_URL)
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(req.get_header("Authorization"), "Bearer " + self.token)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 15)
        for payload in (b'{"models":[]}', b'{"models":[{"slug":"gpt-6-luna-preview","visibility":"list"}]}'):
            self.assertEqual(fetch(payload)[0][0], False)
        for payload in (
            b'{"data":[]}', b'{"models":{}}', b'{"models":[{}]}',
            b'{"models":[{"slug":"gpt-6-luna","visibility":"list"},{"slug":"gpt-6-luna","visibility":"list"}]}',
            b'{"models":[{"slug":"gpt-6-luna","visibility":"hidden"}]}',
            b'{"models":[],"models":[]}', b'{"models":[]', b'not json',
        ):
            with self.subTest(payload=payload[:50]), self.assertRaises(catalog.CheckError):
                fetch(payload)

    def test_redirect_http_and_privacy(self):
        redirect = catalog._NoRedirect()
        with self.assertRaises(catalog.CheckError) as caught:
            redirect.redirect_request(None, None, 302, "secret", {}, "https://evil.invalid/token")
        self.assertEqual(caught.exception.code, "unexpected_redirect")
        with mock.patch.object(catalog, "_catalog_get", side_effect=catalog.CheckError("catalog_http_error", 401)), \
                mock.patch.object(catalog.time, "time", return_value=1200):
            result = catalog.check(self.state)
        self.assertEqual(result["http_status"], 401)
        self.assertEqual(result["catalog_requests"], 1)
        encoded = json.dumps(result)
        for secret in (self.token, self.client_id, self.host_id, "invented-subject"):
            self.assertNotIn(secret, encoded)


if __name__ == "__main__":
    unittest.main()
