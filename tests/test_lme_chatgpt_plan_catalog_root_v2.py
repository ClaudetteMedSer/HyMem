"""Root-owned offline checks for the model-listing helper."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("catalog_listing_root", ROOT / "tools/diagnostics/lme_chatgpt_plan_catalog_v2.py")
listing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(listing)


class RootListingChecks(unittest.TestCase):
    def fetch(self, payload):
        base = listing._load_catalog_module()
        reply = mock.MagicMock()
        reply.__enter__.return_value = reply
        reply.status = 200
        reply.read.return_value = payload
        opener = mock.Mock()
        opener.open.return_value = reply
        with mock.patch.object(listing.request, "build_opener", return_value=opener):
            result = listing._catalog_get("invented-bearer", base)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, "https://api.openai.com/v1/models")
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(opener.open.call_args.kwargs, {"timeout": 15})
        reply.read.assert_called_once_with(1024 * 1024 + 1)
        return result

    def test_only_selected_fields_in_order(self):
        entries = [
            {"slug": "model-b", "display_name": "Model B", "visibility": "list", "private": "must-not-export"},
            {"slug": "hidden-model", "display_name": "Hidden", "visibility": "hidden"},
            {"slug": "model-a", "display_name": "Model A", "visibility": "list", "extra": {"credential": "secret"}},
        ]
        result = self.fetch(json.dumps({"models": entries, "private": "secret"}).encode())
        self.assertEqual(result, [{"slug": "model-b", "display_name": "Model B"}, {"slug": "model-a", "display_name": "Model A"}])

    def test_all_or_nothing_on_late_malformed_entry(self):
        payload = {"models": [
            {"slug": "model-a", "display_name": "Model A", "visibility": "list"},
            {"slug": "bad\nslug", "display_name": "Bad", "visibility": "list"}]}
        with self.assertRaises(listing.ListError):
            self.fetch(json.dumps(payload).encode())

    def test_duplicate_nonvisible_model_rejected(self):
        entry = {"slug": "model-a", "display_name": "A", "visibility": "hidden"}
        with self.assertRaises(listing.ListError):
            self.fetch(json.dumps({"models": [entry, entry]}).encode())

    def test_display_controls_and_limits(self):
        for name in ["", "x" * 129, "x\nsecret", "x\x1b[31m", "x\u202ey"]:
            with self.subTest(name=repr(name)), self.assertRaises(listing.ListError):
                self.fetch(json.dumps({"models": [{"slug": "model-a", "display_name": name, "visibility": "list"}]}).encode())

    def test_boundaries(self):
        entries = [{"slug": f"model-{i}", "display_name": "Model", "visibility": "list"} for i in range(129)]
        self.assertEqual(len(self.fetch(json.dumps({"models": entries[:128]}).encode())), 128)
        with self.assertRaises(listing.ListError):
            self.fetch(json.dumps({"models": entries}).encode())
        with self.assertRaises(listing.ListError):
            self.fetch(b" " * (1024 * 1024 + 1))

    def test_pinned_source_mismatch_stops_before_network(self):
        with mock.patch.object(listing, "CATALOG_SHA256", "0" * 64), mock.patch.object(listing, "_catalog_get") as network:
            result = listing.check(Path("/does-not-exist"))
        self.assertEqual(result["error"], "source_mismatch")
        self.assertEqual(result["catalog_requests"], 0)
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
