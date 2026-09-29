"""Network-free controller tests: container intent and metadata projection."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

HOST = Path(__file__).resolve().parents[1] / "claim_conflict_embedding_host.py"
spec = importlib.util.spec_from_file_location("claim_conflict_embedding_host", HOST)
assert spec is not None and spec.loader is not None
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class FakeHelper:
    RUNTIME = Path("/approved/venv")
    RUNTIME_ENV = Path("/approved/private-env.json")
    IMAGE = "sha256:" + "a" * 64


class EmbeddingHostTests(unittest.TestCase):
    def test_offline_is_networkless_and_live_only_embedding_worker(self):
        source_sha = "b" * 64
        offline, mounts = host.configure(FakeHelper, "offline", source_sha)
        live, live_mounts = host.configure(FakeHelper, "live", source_sha)
        self.assertEqual(mounts, live_mounts)
        self.assertEqual(offline[offline.index("--network") + 1], "none")
        self.assertEqual(live[live.index("--network") + 1], "hermes-net")
        self.assertEqual(live[-6:], ["-I", "-B",
                                     "/diag/claim_conflict_embedding_verify.py",
                                     "live", "--env", "/run/runtime-env.json"])
        self.assertIn("CLAIM_SOURCE_SHA256=" + source_sha, live)
        self.assertNotIn("/diag/claim_conflict_instrumented_dream.py", live)
        self.assertTrue(all(not rw for src, dst, rw in mounts
                            if src != str(host.WORK)))

    def test_projection_rejects_private_static_code(self):
        with self.assertRaises(RuntimeError):
            host.projection({"status": "error",
                             "budget_reason": "private_source_identifier"})
        with self.assertRaises(RuntimeError):
            host.projection({"status": "error",
                             "candidate_frames": [{"path": "/private/path.py",
                                                   "function": "f", "line": 1}]})


if __name__ == "__main__":
    unittest.main()
