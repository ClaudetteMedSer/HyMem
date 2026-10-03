"""LME embedding setup guidance agrees with the strict, network-free resolver."""

from __future__ import annotations

import argparse
import os
import socket
import sys

import pytest

from benchmarks import longmemeval_adapter as lme
from benchmarks.strictness import BenchmarkIntegrityError
from hymem.contrib.endpoint_policy import (
    EMBEDDING_INTERNAL_HTTP_ENV,
    TRANSPORT_SECURITY_INTERNAL_HTTP,
    TRANSPORT_SECURITY_LOOPBACK_HTTP,
)
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient


@pytest.fixture(autouse=True)
def isolated_embedding_setup(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("HYMEM_", "OPENAI_", "DEEPSEEK_")):
            monkeypatch.delenv(name)

    def forbidden(*args, **kwargs):
        pytest.fail("Setup guidance must not construct a provider or send traffic")

    monkeypatch.setattr(OpenAICompatibleClient, "__init__", forbidden)
    monkeypatch.setattr(OpenAICompatibleEmbeddingClient, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "sendto", forbidden)


def _cli_args(monkeypatch, *argv):
    """Parse the real CLI, stopping immediately after its parser returns."""
    original = argparse.ArgumentParser.parse_args
    captured = {}

    class Parsed(Exception):
        pass

    def parse_and_stop(parser, *args, **kwargs):
        captured["args"] = original(parser, *args, **kwargs)
        raise Parsed

    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "argv", ["longmemeval_adapter.py", *argv])
        scoped.setattr(argparse.ArgumentParser, "parse_args", parse_and_stop)
        with pytest.raises(Parsed):
            lme.main()
    return captured["args"]


def _attested_args(monkeypatch, *argv):
    return _cli_args(
        monkeypatch, "--embeddings",
        "--embedding-deployment-revision", "fixture-release-2026-09",
        "--embedding-deployment-tenant", "fixture-network", *argv,
    )


def test_cli_help_explains_required_setup_and_preserves_lexical_default(
    monkeypatch, capsys,
):
    monkeypatch.setattr(sys, "argv", ["longmemeval_adapter.py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        lme.main()
    assert stopped.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "Works with NO env setup" not in output
    assert "Requires a running embedding server" in output
    assert "--embedding-deployment-revision and --embedding-deployment-tenant" in output
    assert "HYMEM_EMBEDDING_DEPLOYMENT_REVISION and HYMEM_EMBEDDING_DEPLOYMENT_TENANT" in output
    assert "adapter pins the dimension automatically" in output
    assert "HYMEM_EMBEDDING_PIN_DIMENSION is not required here" in output
    assert "reachable --embedding-base-url (including /v1" in output
    assert "HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP=1" in output
    assert "Endpoint and identity checks remain strict" in output
    assert "DEFAULT OFF (lexical-only baseline)" in output
    for name in ("BASE_URL", "MODEL", "DIM", "API_KEY"):
        assert f"HYMEM_EMBEDDING_{name}" in output


def test_default_cli_stays_lexical_without_server_or_attestations(monkeypatch):
    args = _cli_args(monkeypatch)
    assert args.embeddings is False
    identity = lme.resolve_embedding_identity(args)
    assert identity["backend"] == "none"
    assert identity["configured"] is False
    assert identity["network_free"] is True
    assert identity["dimension"] is None


@pytest.mark.parametrize(
    "options",
    [
        (),
        ("--embedding-deployment-revision", "fixture-release"),
        ("--embedding-deployment-tenant", "fixture-network"),
    ],
)
def test_enabled_embeddings_require_both_attestations(monkeypatch, options):
    args = _cli_args(monkeypatch, "--embeddings", *options)
    with pytest.raises(BenchmarkIntegrityError, match="explicit deployment revision, tenant"):
        lme.resolve_embedding_identity(args)


@pytest.mark.parametrize("pin_setting", [None, "0", "1"])
def test_cli_attestations_work_without_requiring_pin_env(monkeypatch, pin_setting):
    if pin_setting is not None:
        monkeypatch.setenv("HYMEM_EMBEDDING_PIN_DIMENSION", pin_setting)
    args = _attested_args(monkeypatch)
    identity = lme.resolve_embedding_identity(args)
    assert identity["backend"] == "openai_compatible"
    assert identity["dimension"] == lme.LOCAL_EMBED_DIM == 384
    assert identity["identity_exact"] is True
    assert identity["fallback_policy"] == "fail-closed"
    assert identity["transport_security"] == TRANSPORT_SECURITY_LOOPBACK_HTTP


def test_full_named_env_configuration_matches_cli_configuration(monkeypatch):
    expected = lme.resolve_embedding_identity(_attested_args(
        monkeypatch, "--embedding-base-url", "http://localhost:8766/v1",
        "--embedding-model", "fixture-embed", "--embedding-dim", "768",
    ))
    for name, value in {
        "HYMEM_EMBEDDING_BASE_URL": "http://localhost:8766/v1",
        "HYMEM_EMBEDDING_MODEL": "fixture-embed",
        "HYMEM_EMBEDDING_DIM": "768",
        "HYMEM_EMBEDDING_DEPLOYMENT_REVISION": "fixture-release-2026-09",
        "HYMEM_EMBEDDING_DEPLOYMENT_TENANT": "fixture-network",
    }.items():
        monkeypatch.setenv(name, value)
    assert "HYMEM_EMBEDDING_PIN_DIMENSION" not in os.environ
    actual = lme.resolve_embedding_identity(_cli_args(monkeypatch, "--embeddings"))
    assert actual == expected


@pytest.mark.parametrize(
    "url", ["http://embedding-server:8766/v1", "http://192.168.1.20:8766/v1"],
)
def test_internal_http_requires_explicit_opt_in_even_with_attestations(monkeypatch, url):
    args = _attested_args(monkeypatch, "--embedding-base-url", url)
    with pytest.raises(BenchmarkIntegrityError, match="endpoint is unsafe or ambiguous"):
        lme.resolve_embedding_identity(args)
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, "1")
    identity = lme.resolve_embedding_identity(args)
    assert identity["transport_security"] == TRANSPORT_SECURITY_INTERNAL_HTTP
    assert identity["fallback_policy"] == "fail-closed"
    assert identity["backend"] == "openai_compatible"


def test_internal_http_opt_in_does_not_replace_attestations(monkeypatch):
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, "1")
    args = _cli_args(
        monkeypatch, "--embeddings",
        "--embedding-base-url", "http://embedding-server:8766/v1",
    )
    with pytest.raises(BenchmarkIntegrityError, match="explicit deployment revision, tenant"):
        lme.resolve_embedding_identity(args)


@pytest.mark.parametrize(
    "url",
    ["http://example.com/v1", "http://embedding-server:8766/v1?route=other"],
)
def test_internal_http_opt_in_does_not_relax_other_endpoint_checks(monkeypatch, url):
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, "1")
    args = _attested_args(monkeypatch, "--embedding-base-url", url)
    with pytest.raises(BenchmarkIntegrityError, match="endpoint is unsafe or ambiguous"):
        lme.resolve_embedding_identity(args)
