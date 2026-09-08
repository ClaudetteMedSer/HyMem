"""Copyable deployment configuration and real SDK route construction, offline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

import httpx
import pytest

from hymem.bootstrap import resolve_env
from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient


_README = Path(__file__).resolve().parents[1] / "README.md"
_REQUIRED_PRODUCER = (
    "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM",
    "HYMEM_EMBEDDING_DEPLOYMENT_REVISION", "HYMEM_EMBEDDING_DEPLOYMENT_TENANT",
)
_OPERATOR_VALUES = {
    "HYMEM_EMBEDDING_BASE_URL": "https://embeddings.example/proxy/openai",
    "HYMEM_EMBEDDING_API_KEY": "embedding-test-credential",
    "HYMEM_EMBEDDING_MODEL": "operator-confirmed-model",
    "HYMEM_EMBEDDING_DIM": "7",
    "HYMEM_EMBEDDING_DEPLOYMENT_REVISION": "operator-release-2026-09",
    "HYMEM_EMBEDDING_DEPLOYMENT_TENANT": "operator-workspace",
}


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith("HYMEM_") or key in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}:
            monkeypatch.delenv(key)


def _remote_blocks():
    blocks = [
        block for block in re.findall(r"```bash\n(.*?)```", _README.read_text(), re.S)
        if re.search(r"\bexport\s+HYMEM_EMBEDDING_(?:BASE_URL|API_KEY)\b", block)
    ]
    assert len(blocks) == 2, "Keep complete HTTPS and internal embedding recipes"
    return blocks


def _run_env_block(block, values):
    # Never run installation, doctor, or server commands from a doc fixture.
    # The two copyable environment blocks must be standalone and inert.
    assert all(
        not line.strip() or line.lstrip().startswith(("#", "export ", ": "))
        for line in block.splitlines()
    ), "Remote setup must be a standalone environment-only recipe"
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", block + "\n/usr/bin/env -0"],
        env={"PATH": os.defpath, **values}, text=True, capture_output=True,
        timeout=5, check=False,
    )


def _decoded_env(result):
    assert result.returncode == 0, result.stderr
    return dict(item.split("=", 1) for item in result.stdout.split("\0") if item)


def _client(cfg):
    return OpenAICompatibleEmbeddingClient(
        api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url,
        model=cfg.embedding_model, dim=cfg.embedding_dim,
        pin_dimension=cfg.embedding_pin_dimension,
        deployment_revision=cfg.embedding_deployment_revision,
        deployment_tenant=cfg.embedding_deployment_tenant,
    )


def _mock_http(monkeypatch):
    requests = []

    def respond(_transport, request):
        requests.append(request)
        return httpx.Response(200, request=request, json={
            "object": "list", "model": _OPERATOR_VALUES["HYMEM_EMBEDDING_MODEL"],
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0] + [0.0] * 6}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    # Keep the real OpenAI SDK request builder and owned transport object.
    # Replace the bottom HTTP boundary before the client seals its policy.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    return requests


@pytest.mark.parametrize("block_index", [0, 1])
def test_remote_readme_recipes_resolve_and_use_actual_sdk_route(monkeypatch, block_index):
    block = _remote_blocks()[block_index]
    values = dict(_OPERATOR_VALUES)
    internal = "http://embedding-server:8766" in block
    if internal:
        values.pop("HYMEM_EMBEDDING_API_KEY")
    resolved = _decoded_env(_run_env_block(block, values))
    for key, value in resolved.items():
        if key.startswith("HYMEM_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("OPENAI_API_KEY", "cloud-test-credential")
    monkeypatch.setenv("HYMEM_LLM_API_KEY", "llm-test-credential")
    cfg = resolve_env()
    assert cfg.embedding_backend == "openai_compatible"
    assert cfg.embedding_fallback_reason is None
    assert cfg.embedding_pin_dimension is True
    assert cfg.embedding_dim == 7
    assert cfg.embedding_model == values["HYMEM_EMBEDDING_MODEL"]
    assert cfg.embedding_deployment_revision == values["HYMEM_EMBEDDING_DEPLOYMENT_REVISION"]
    assert cfg.embedding_deployment_tenant == values["HYMEM_EMBEDDING_DEPLOYMENT_TENANT"]
    expected_base = "http://embedding-server:8766/v1" if internal else values["HYMEM_EMBEDDING_BASE_URL"]
    assert cfg.embedding_base_url == expected_base
    requests = _mock_http(monkeypatch)
    client = _client(cfg)
    try:
        assert client.embed(["deployment probe"]) == [[1.0] + [0.0] * 6]
        assert len(requests) == client.request_attempts == 1
        assert str(requests[0].url) == expected_base + "/embeddings"
        assert requests[0].headers["authorization"] == (
            "Bearer local" if internal else "Bearer embedding-test-credential"
        )
        payload = json.loads(requests[0].content)
        assert payload["model"] == values["HYMEM_EMBEDDING_MODEL"]
        assert "dimensions" not in payload  # Pin validates output; it does not resize it.
    finally:
        client.close()


@pytest.mark.parametrize("block_index", [0, 1])
def test_remote_readme_recipes_refuse_missing_operator_values(block_index):
    block = _remote_blocks()[block_index]
    required = _REQUIRED_PRODUCER
    if "http://embedding-server:8766" not in block:
        required += ("HYMEM_EMBEDDING_BASE_URL", "HYMEM_EMBEDDING_API_KEY")
    for missing in required:
        for empty in (False, True):
            values = dict(_OPERATOR_VALUES)
            if empty:
                values[missing] = ""
            else:
                values.pop(missing)
            result = _run_env_block(block, values)
            assert result.returncode != 0, f"Recipe silently accepted missing {missing}"
            assert missing in result.stderr
            assert result.stdout == ""


@pytest.mark.parametrize(("base", "path"), [
    ("http://embedding-server:8766", "/embeddings"),
    ("http://embedding-server:8766/v1", "/v1/embeddings"),
    ("http://embedding-server:8766/v1/", "/v1/embeddings"),
    ("http://embedding-server:8766/proxy/api", "/proxy/api/embeddings"),
])
def test_real_sdk_preserves_base_route_without_automatic_v1(monkeypatch, base, path):
    for key, value in _OPERATOR_VALUES.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", base)
    monkeypatch.setenv("HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP", "1")
    monkeypatch.setenv("HYMEM_EMBEDDING_PIN_DIMENSION", "1")
    requests = _mock_http(monkeypatch)
    client = _client(resolve_env())
    try:
        client.embed(["route probe"])
        assert len(requests) == 1
        assert requests[0].url.path == path
    finally:
        client.close()


def test_readme_distinguishes_preflight_from_recovery_and_periodic_work():
    text = _README.read_text()
    assert "event-driven, not periodic" in text
    assert "single dream" in text and "does not guarantee" in text
    assert "not a per-row vector recovery audit" in text
    assert "Uses FastAPI `BackgroundTasks`" not in text
    first_block = re.search(r"```bash\n(.*?)```", text, re.S).group(1)
    assert "export HYMEM_EMBEDDING_API_KEY=" not in first_block
