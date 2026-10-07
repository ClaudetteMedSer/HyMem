"""Independent root controls; invented credentials and mocked network only."""
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location(
    "catalog_fixtures", Path(__file__).with_name("test_lme_chatgpt_plan_catalog_v1.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
c = fixtures.catalog


class RootControls(fixtures.CatalogTests):
    def test_parent_symlink_rejected_without_request(self):
        alias = self.state.parent / "alias"
        alias.symlink_to(self.state, target_is_directory=True)
        with mock.patch.object(c, "_catalog_get") as network:
            result = c.check(alias)
        self.assertEqual(result["error"], "state_dir_unavailable")
        network.assert_not_called()

    def test_missing_directory_not_created(self):
        missing = self.state / "absent"
        with mock.patch.object(c, "_catalog_get") as network:
            result = c.check(missing)
        self.assertFalse(missing.exists())
        self.assertEqual(result["catalog_requests"], 0)
        network.assert_not_called()

    def test_oversized_private_file(self):
        with (self.state / "credential.json").open("wb") as stream:
            stream.write(b" " * (c.MAX_FILE + 1))
        result, network = self._check()
        self.assertEqual(result["error"], "state_file_too_large")
        network.assert_not_called()

    def test_bad_credentials_never_admitted(self):
        credential = self.files["credential.json"]
        changes = [("saved_at", 1300), ("expires_at", True), ("version", True),
                   ("scope", ["openid"] * 65), ("access_token", "x\r\nInjected: bad"),
                   ("access_token", "x" * 8193), ("scope", list(c.REQUIRED_SCOPES) + ["openid"])]
        for field, value in changes:
            with self.subTest(field=field):
                original = credential[field]
                credential[field] = value
                self._save()
                result, network = self._check()
                self.assertFalse(result["available"])
                network.assert_not_called()
                credential[field] = original

    def test_response_bound_and_no_secret_exception_output(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b" " * (c.MAX_RESPONSE + 1)
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(c.request, "build_opener", return_value=opener):
            with self.assertRaises(c.CheckError) as caught:
                c._catalog_get("invented")
        self.assertEqual(caught.exception.code, "catalog_too_large")
        response.read.assert_called_once_with(c.MAX_RESPONSE + 1)
        with mock.patch.object(c, "_catalog_get", side_effect=RuntimeError(self.token)), \
                mock.patch.object(c.time, "time", return_value=1200):
            result = c.check(self.state)
        self.assertEqual(result["error"], "internal_error")
        self.assertNotIn(self.token, json.dumps(result))

    def test_deadline_is_terminal_and_restores_handler(self):
        previous = c.signal.getsignal(c.signal.SIGALRM)
        with mock.patch.object(c, "_catalog_get", side_effect=c._AbsoluteTimeout()), \
                mock.patch.object(c.time, "time", return_value=1200):
            result = c.check(self.state)
        self.assertEqual(result["error"], "deadline_exceeded")
        self.assertEqual(c.signal.getsignal(c.signal.SIGALRM), previous)
        self.assertEqual(c.signal.getitimer(c.signal.ITIMER_REAL), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
