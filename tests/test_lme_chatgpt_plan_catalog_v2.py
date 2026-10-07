"""Offline controls for the read-only SIWC model listing."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "tools/diagnostics/lme_chatgpt_plan_catalog_v2.py"
spec = importlib.util.spec_from_file_location("lme_catalog_v2_under_test", SOURCE)
listing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(listing)


class ListingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name).resolve() / "private"
        self.state.mkdir(mode=0o700)
        host_id = "urn:uuid:88aaa7c0-4f3b-4c62-843a-925831874f82"
        client_id = "oaiapp_abcdefgh12345678"
        self.token = "invented-access-token"
        files = {
            "status.json": {"status": "complete", "plan_usage_enabled": True, "model_calls": 0},
            "host.json": {"ext_agent_host_id": host_id},
            "registration.json": {"host_id": host_id, "client_id": client_id},
            "credential.json": {"version": 1, "issuer": "https://auth.openai.com", "host_id": host_id,
                                "client_id": client_id, "subject": "invented-subject", "access_token": self.token,
                                "token_type": "Bearer", "scope": ["openid", "offline_access", "resource.invoke",
                                                                 "chatgpt.tokens.use.direct"], "saved_at": 1000,
                                "expires_at": 2000, "plan_usage_enabled": True},
        }
        for name, value in files.items():
            path = self.state / name
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o600)

    def _fetch(self, payload, status=200):
        class Reply:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, _size): return payload
        reply = Reply()
        reply.status = status
        opener = mock.Mock()
        opener.open.return_value = reply
        with mock.patch.object(listing.request, "build_opener", return_value=opener):
            result = listing._catalog_get(self.token, listing._load_catalog_module())
        return result, opener

    def test_frozen_sources_and_local_validation(self):
        v1 = SOURCE.with_name("lme_chatgpt_plan_catalog_v1.py")
        self.assertEqual(hashlib.sha256(v1.read_bytes()).hexdigest(), listing.CATALOG_SHA256)
        signin = SOURCE.with_name("lme_chatgpt_plan_signin_v3.py")
        self.assertEqual(hashlib.sha256(signin.read_bytes()).hexdigest(),
                         listing._load_catalog_module().SIGNIN_SHA256)
        rows = [{"slug": "gpt-6-luna", "display_name": "GPT-6 Luna"}]
        with mock.patch.object(listing.time, "time", return_value=1200), mock.patch.object(
                listing, "_catalog_get", return_value=rows) as network:
            result = listing.check(self.state)
        self.assertEqual(result, {"models": rows, "model_calls": 0, "catalog_requests": 1})
        self.assertEqual(network.call_args.args[0], self.token)
        bad = self.state / "credential.json"
        bad.chmod(0o644)
        with mock.patch.object(listing, "_catalog_get") as network:
            result = listing.check(self.state)
        self.assertEqual(result, {"error": "state_file_not_private", "models": [],
                                  "model_calls": 0, "catalog_requests": 0})
        network.assert_not_called()

    def test_server_order_visibility_and_field_allowlist(self):
        payload = json.dumps({"models": [
            {"slug": "z-model", "display_name": "Z Model", "visibility": "list", "secret": "never"},
            {"slug": "hidden-model", "display_name": "Hidden", "visibility": "hidden"},
            {"slug": "a-model", "display_name": "A Model", "visibility": "list", "other": {"x": 1}},
        ], "other": self.token}).encode()
        rows, opener = self._fetch(payload)
        self.assertEqual(rows, [{"slug": "z-model", "display_name": "Z Model"},
                                {"slug": "a-model", "display_name": "A Model"}])
        self.assertNotIn(self.token, json.dumps(rows))
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, listing.CATALOG_URL)
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(req.get_header("Authorization"), "Bearer " + self.token)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 15)

    def test_malformed_entries_and_duplicates_fail_closed(self):
        base = {"slug": "valid-1", "display_name": "Valid", "visibility": "list"}
        invalid = [
            {}, {**base, "slug": "bad slug"}, {**base, "slug": "x" * 129},
            {**base, "display_name": "bad\nname"}, {**base, "display_name": "x" * 129},
            {**base, "display_name": ""}, {**base, "visibility": 1},
            {**base, "visibility": "hidden\n"},
        ]
        for entry in invalid:
            with self.subTest(entry=entry), self.assertRaises(listing.ListError):
                self._fetch(json.dumps({"models": [entry]}).encode())
        for payload in [b'{"models":[],"models":[]}',
                        b'{"models":[{"slug":"x","slug":"y","display_name":"X","visibility":"list"}]}',
                        b'{"models":[]', b'not json', b'{"data":[]}',
                        b'{"models":[],"extra":NaN}']:
            with self.subTest(payload=payload), self.assertRaises(listing.ListError):
                self._fetch(payload)
        with self.assertRaises(listing.ListError):
            self._fetch(json.dumps({"models": [base, {**base, "visibility": "hidden"}]}).encode())
        with self.assertRaises(listing.ListError):
            self._fetch(json.dumps({"models": [{**base, "slug": str(i)} for i in range(129)]}).encode())

    def test_failure_never_emits_token(self):
        with mock.patch.object(listing.time, "time", return_value=1200), mock.patch.object(
                listing, "_catalog_get", side_effect=RuntimeError(self.token)):
            result = listing.check(self.state)
        self.assertEqual(result, {"error": "internal_error", "models": [],
                                  "model_calls": 0, "catalog_requests": 1})
        self.assertNotIn(self.token, json.dumps(result))
        with mock.patch.object(listing, "CATALOG_SHA256", "0" * 64), mock.patch.object(
                listing, "_catalog_get") as network:
            result = listing.check(self.state)
        self.assertEqual(result["error"], "source_mismatch")
        self.assertEqual(result["catalog_requests"], 0)
        network.assert_not_called()

    def test_expired_and_symlink_state_rejected_without_request(self):
        with mock.patch.object(listing.time, "time", return_value=2000), mock.patch.object(
                listing, "_catalog_get") as network:
            result = listing.check(self.state)
        self.assertEqual(result["error"], "credential_invalid")
        self.assertEqual(result["catalog_requests"], 0)
        network.assert_not_called()
        alias = self.state.parent / "alias"
        alias.symlink_to(self.state, target_is_directory=True)
        with mock.patch.object(listing, "_catalog_get") as network:
            result = listing.check(alias)
        self.assertEqual(result["error"], "state_dir_unavailable")
        network.assert_not_called()

    def test_response_size_and_redirect_rejected(self):
        with self.assertRaises(listing.ListError) as caught:
            self._fetch(b" " * (listing.MAX_RESPONSE + 1))
        self.assertEqual(caught.exception.code, "catalog_too_large")
        catalog = listing._load_catalog_module()
        with self.assertRaises(catalog.CheckError) as caught:
            catalog._NoRedirect().redirect_request(None, None, 302, "redirect", {},
                                                    "https://elsewhere.invalid")
        self.assertEqual(caught.exception.code, "unexpected_redirect")


if __name__ == "__main__":
    unittest.main()
