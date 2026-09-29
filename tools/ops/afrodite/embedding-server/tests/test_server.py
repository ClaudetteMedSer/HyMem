"""Offline tests: no FastEmbed imports, model downloads, or inference."""
import asyncio
import importlib.util
import inspect
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

import httpx

spec = importlib.util.spec_from_file_location("embedding_server", Path(__file__).parents[1] / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts, *, batch_size):
        self.calls.append((list(texts), batch_size))
        assert len(texts) <= 16 and batch_size <= 16
        return ([float(text)] * server.DIM for text in texts)


class ServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = FakeEmbedder()
        self.saved = (server.embed_model, server._rerank_model, server._rerank_load_error)
        server.embed_model = self.model
        server._rerank_model = None
        server._rerank_load_error = None

    def tearDown(self):
        server.embed_model, server._rerank_model, server._rerank_load_error = self.saved

    async def request(self, path, payload):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            return await client.post(path, json=payload)

    async def test_large_request_preserves_order_and_response(self):
        response = await self.request("/v1/embeddings", {"input": [str(i) for i in range(35)], "model": "ignored"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([len(call[0]) for call in self.model.calls], [16, 16, 3])
        self.assertEqual([call[1] for call in self.model.calls], [16, 16, 16])
        self.assertEqual([row["index"] for row in body["data"]], list(range(35)))
        self.assertEqual([row["embedding"][0] for row in body["data"]], list(range(35)))
        self.assertTrue(all(len(row["embedding"]) == server.DIM for row in body["data"]))
        self.assertEqual(body["model"], server.MODEL_NAME)
        self.assertEqual(body["usage"], {"prompt_tokens": 0, "total_tokens": 0})
        self.assertEqual(body["object"], "list")

    async def test_string_empty_and_unloaded(self):
        response = await self.request("/v1/embeddings", {"input": "4"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["index"], 0)
        response = await self.request("/v1/embeddings", {"input": []})
        self.assertEqual((response.status_code, response.json()["detail"]), (400, "Empty input"))
        server.embed_model = None
        response = await self.request("/v1/embeddings", {"input": "4"})
        self.assertEqual(response.status_code, 503)

    async def test_bad_vectors_return_static_503(self):
        valid = [0.0] * server.DIM
        cases = [[], [valid, valid], [[0.0]], [[float("nan")] * server.DIM], [[float("inf")] * server.DIM], [["secret"] * server.DIM]]
        for output in cases:
            with self.subTest(output=str(output)[:30]):
                with patch.object(self.model, "embed", return_value=iter(output)):
                    response = await self.request("/v1/embeddings", {"input": "0"})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json(), {"detail": "Embedding inference failed"})
        with patch.object(self.model, "embed", side_effect=RuntimeError("secret input")):
            response = await self.request("/v1/embeddings", {"input": "0"})
        self.assertEqual(response.json(), {"detail": "Embedding inference failed"})

    async def test_health_responsive_and_inference_serialized(self):
        entered, release, reranked = threading.Event(), threading.Event(), threading.Event()

        def blocking_embed(texts, *, batch_size):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test timed out")
            return [[0.0] * server.DIM for _ in texts]

        class Reranker:
            def rerank(self, query, docs, *, batch_size):
                reranked.set()
                self.args = (query, docs, batch_size)
                return list(range(len(docs)))

        ranker = Reranker()
        server._rerank_model = ranker
        with patch.object(self.model, "embed", side_effect=blocking_embed):
            embedding = asyncio.create_task(self.request("/v1/embeddings", {"input": "0"}))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                reranking = asyncio.create_task(self.request("/v1/rerank", {"query": "q", "documents": ["x" * 2000, "y"], "top_n": 1}))
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                    health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
                self.assertEqual(health.status_code, 200)
                await asyncio.sleep(0.05)
                self.assertFalse(reranked.is_set())
            finally:
                release.set()
            responses = await asyncio.gather(embedding, reranking)
        self.assertEqual([r.status_code for r in responses], [200, 200])
        self.assertEqual(ranker.args, ("q", ["x" * server.RERANK_MAX_CHARS, "y"], server.RERANK_BATCH_SIZE))
        self.assertEqual(responses[1].json()["results"], [{"index": 1, "relevance_score": 1.0}])
        self.assertFalse(inspect.iscoroutinefunction(server.embeddings))

    async def test_startup_model_configuration_unchanged(self):
        calls = []
        fake = types.ModuleType("fastembed")
        fake.TextEmbedding = lambda **kwargs: calls.append(kwargs) or self.model
        with patch.dict(sys.modules, {"fastembed": fake}), patch.object(server.os, "makedirs"), patch.object(server, "RERANK_PRELOAD", False):
            async with server.lifespan(server.app):
                pass
        self.assertEqual(calls, [{"model_name": server.MODEL_NAME, "max_length": server.MAX_TOKENS, "cache_dir": server.CACHE_DIR}])


if __name__ == "__main__":
    unittest.main()
