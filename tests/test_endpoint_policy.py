"""Adversarial endpoint/key-isolation tests for every active provider client."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hymem.bootstrap import resolve_env
from hymem.contrib.endpoint_policy import (
    EMBEDDING_INTERNAL_HTTP_ENV,
    EndpointPolicyError,
    resolve_embedding_api_key,
    resolve_llm_api_key,
    safe_endpoint_label,
    secret_free_endpoint_identity,
    validate_public_attestation,
    validate_http_endpoint,
)
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.contrib.openai_embedding_client import (
    OpenAICompatibleEmbeddingClient,
    openai_compatible_embedding_identity,
    safe_embedding_base_url,
)


_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(_BENCH))
import beam_adapter as beam  # noqa: E402
import beam_registry  # noqa: E402
import lme_protocol  # noqa: E402
import longmemeval_adapter as lme  # noqa: E402


@pytest.mark.parametrize("url", [
    "ftp://api.openai.com/v1",
    "https://user:secret@api.openai.com/v1",
    "https://api.openai.com/v1?opaque=secret",
    "https://api.openai.com/v1#secret",
    "https://api.openai.com:not-a-port/v1",
    "https://api.openai.com:0/v1",
    "https://[broken/v1",
    "https://api.openai.com\\@evil.example/v1",
])
def test_endpoint_shape_is_fail_closed_without_echoing_input(url):
    with pytest.raises(EndpointPolicyError) as exc_info:
        validate_http_endpoint(url, label="test")
    assert "secret" not in str(exc_info.value)
    assert "opaque" not in str(exc_info.value)


@pytest.mark.parametrize("url", [
    "https://api.deepseek.com",
    "https://api.deepseek.com/v1/",
    "HTTPS://API.DEEPSEEK.COM:443/v1",
])
def test_exact_deepseek_https_origin_accepts_only_its_provider_key(url):
    endpoint, key = resolve_llm_api_key(
        url, environ={"DEEPSEEK_API_KEY": "deepseek-only"}
    )
    assert endpoint.official_provider == "deepseek"
    assert key == "deepseek-only"
    with pytest.raises(EnvironmentError):
        resolve_llm_api_key(url, environ={"OPENAI_API_KEY": "must-not-cross"})


@pytest.mark.parametrize("url", [
    "https://api.openai.com",
    "https://api.openai.com/v1/",
    "HTTPS://API.OPENAI.COM:443/v1",
])
def test_exact_openai_https_origin_accepts_only_its_provider_key(url):
    endpoint, key = resolve_llm_api_key(
        url, environ={"OPENAI_API_KEY": "openai-only"}
    )
    assert endpoint.official_provider == "openai"
    assert key == "openai-only"
    with pytest.raises(EnvironmentError):
        resolve_llm_api_key(url, environ={"DEEPSEEK_API_KEY": "must-not-cross"})


@pytest.mark.parametrize("url", [
    "https://api.openai.com.evil.example/v1",
    "https://openai.com/v1",
    "https://api-eu.deepseek.com/v1",
    "https://deepseek.example/v1",
    "https://api.openai.com.:443/v1",
])
def test_provider_keys_do_not_flow_to_lookalikes_or_subdomains(url):
    with pytest.raises(EnvironmentError):
        resolve_llm_api_key(
            url,
            environ={
                "OPENAI_API_KEY": "openai-secret",
                "DEEPSEEK_API_KEY": "deepseek-secret",
            },
        )


def test_purpose_bound_and_constructor_keys_are_allowed_for_custom_https():
    _, from_env = resolve_llm_api_key(
        "https://gateway.example/v1",
        environ={"HYMEM_LLM_API_KEY": "purpose-key"},
    )
    _, explicit = resolve_llm_api_key(
        "https://gateway.example/v1",
        explicit_key="constructor-key",
        environ={"OPENAI_API_KEY": "must-not-win"},
    )
    assert from_env == "purpose-key"
    assert explicit == "constructor-key"


def test_embedding_keys_are_isolated_from_llm_and_wrong_provider_keys():
    with pytest.raises(EnvironmentError):
        resolve_embedding_api_key(
            "https://embed.example/v1",
            environ={
                "OPENAI_API_KEY": "must-not-cross-to-custom",
                "DEEPSEEK_API_KEY": "must-not-cross",
                "HYMEM_LLM_API_KEY": "llm-purpose-must-not-cross",
            },
        )
    with pytest.raises(EnvironmentError):
        resolve_embedding_api_key(
            "https://api.openai.com/v1",
            environ={"DEEPSEEK_API_KEY": "wrong-provider"},
        )
    assert resolve_embedding_api_key(
        "https://api.openai.com/v1",
        environ={"OPENAI_API_KEY": "openai-embedding-key"},
    )[1] == "openai-embedding-key"
    assert resolve_embedding_api_key(
        "https://embed.example/v1",
        environ={"HYMEM_EMBEDDING_API_KEY": "embedding-purpose-key"},
    )[1] == "embedding-purpose-key"


@pytest.mark.parametrize("url", [
    "http://localhost:8000/v1",
    "http://127.0.0.1:8000/v1",
    "http://[::1]:8000/v1",
])
def test_loopback_http_remains_available(url):
    endpoint = validate_http_endpoint(url, label="local")
    assert endpoint.is_loopback is True
    _, key = resolve_embedding_api_key(url, environ={})
    assert key == "local"


@pytest.mark.parametrize("url", [
    "http://embedding-server:8766/v1",
    "http://embeddings.svc:8766/v1",
    "http://10.23.4.5:8766/v1",
    "http://169.254.2.3:8766/v1",
    "http://[fd00::5]:8766/v1",
    "http://[fe80::5]:8766/v1",
])
def test_internal_embedding_http_requires_narrow_explicit_opt_in(url):
    with pytest.raises(EndpointPolicyError, match=EMBEDDING_INTERNAL_HTTP_ENV):
        resolve_embedding_api_key(url, environ={})
    endpoint, key = resolve_embedding_api_key(
        url,
        environ={
            EMBEDDING_INTERNAL_HTTP_ENV: "true",
            "OPENAI_API_KEY": "must-not-cross",
            "DEEPSEEK_API_KEY": "must-not-cross-either",
            "HYMEM_LLM_API_KEY": "llm-purpose-must-not-cross",
        },
    )
    assert endpoint.is_internal_service is True
    assert key == "local"


def test_internal_embedding_key_wins_over_dummy_but_not_without_opt_in():
    url = "http://embedding-server:8766/v1"
    env = {
        EMBEDDING_INTERNAL_HTTP_ENV: "1",
        "HYMEM_EMBEDDING_API_KEY": "embedding-purpose-key",
    }
    assert resolve_embedding_api_key(url, environ=env)[1] == "embedding-purpose-key"
    with pytest.raises(EndpointPolicyError):
        resolve_embedding_api_key(
            url, explicit_key="explicit-key", environ={}
        )


@pytest.mark.parametrize("url", [
    "http://8.8.8.8/v1",
    "http://example.com/v1",
    "http://api.openai.com/v1",
    "http://0x08080808/v1",
])
def test_internal_opt_in_never_allows_public_http(url):
    with pytest.raises(EndpointPolicyError, match="public HTTP"):
        resolve_embedding_api_key(
            url, environ={EMBEDDING_INTERNAL_HTTP_ENV: "yes"}
        )


def test_loopback_hostname_with_dns_root_dot_remains_loopback():
    endpoint = validate_http_endpoint("http://localhost.:8766/v1")
    assert endpoint.is_loopback is True


def test_embedding_identity_uses_the_same_strict_url_contract_as_client():
    with pytest.raises(ValueError):
        openai_compatible_embedding_identity(
            " https://embed.example/v1", "model"
        )


def test_internal_opt_in_truth_value_is_strict():
    with pytest.raises(EndpointPolicyError, match=EMBEDDING_INTERNAL_HTTP_ENV):
        resolve_embedding_api_key(
            "http://embedding-server:8766/v1",
            environ={EMBEDDING_INTERNAL_HTTP_ENV: "truthy-ish"},
        )


def test_safe_labels_and_identities_never_emit_url_credentials_or_queries():
    secret = "arbitrary-opaque-secret"
    raw = f"https://operator:{secret}@embed.example/v1?opaque={secret}#fragment"
    for value in (
        safe_endpoint_label(raw, label="provider"),
        safe_embedding_base_url(raw),
    ):
        assert secret not in value
        assert "operator" not in value
        assert "opaque" not in value
    with pytest.raises(ValueError) as exc_info:
        openai_compatible_embedding_identity(raw, "model")
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize("value", [
    "Authorization: Basic dXNlcjpwYXNz",
    "Basic dXNlcjpwYXNz",
    "https://proxy.example/v1/sk-live-secret-abcdefgh",
    "sk%255Flive%255F1234567890abcdef",
    "api\u200bkey=opaque-credential",
    "api／key＝opaque-credential",
])
def test_public_attestation_rejects_encoded_credentials_without_echo(value):
    with pytest.raises(ValueError) as exc_info:
        validate_public_attestation(value, label="public revision")
    assert value not in str(exc_info.value)


def test_public_attestation_preserves_normal_release_labels():
    for value in ("secret-prod-2026", "releaseabcdefghijklmnopqrstuvwx"):
        assert validate_public_attestation(value, label="public revision") == value


def test_credential_shaped_host_is_rejected_from_identity_and_diagnostics():
    marker = "sk-live-secret-abcdefgh"
    endpoint = f"https://{marker}.proxy.example/v1"
    with pytest.raises(ValueError) as exc_info:
        secret_free_endpoint_identity(endpoint, label="provider")
    assert marker not in str(exc_info.value)
    assert safe_endpoint_label(endpoint, label="provider") == "<invalid provider URL>"


def test_contrib_clients_reject_cross_host_provider_fallback_before_sdk(
    monkeypatch,
):
    constructions: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructions.append(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    for name in (
        "HYMEM_LLM_API_KEY", "HYMEM_EMBEDDING_API_KEY",
        "HYMEM_LLM_BASE_URL", "HYMEM_EMBEDDING_BASE_URL",
        EMBEDDING_INTERNAL_HTTP_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    with pytest.raises(EnvironmentError):
        OpenAICompatibleClient(base_url="https://gateway.example/v1")
    with pytest.raises(EnvironmentError):
        OpenAICompatibleEmbeddingClient(base_url="https://embed.example/v1")
    assert constructions == []


def test_raw_benchmark_clients_apply_the_same_host_and_transport_policy(
    monkeypatch,
):
    for name in ("HYMEM_LLM_API_KEY", "HYMEM_EMBEDDING_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    for client_type in (lme.LLMClient, beam.LLMClient):
        with pytest.raises((lme.BenchmarkIntegrityError, beam.BenchmarkIntegrityError)):
            client_type("model", "", base_url="https://gateway.example/v1")
        with pytest.raises((lme.BenchmarkIntegrityError, beam.BenchmarkIntegrityError)):
            client_type("model", "explicit", base_url="http://example.com/v1")
        assert client_type(
            "model", "", base_url="https://api.openai.com/v1"
        ).api_key == "openai-secret"
        assert client_type(
            "model", "", base_url="https://api.deepseek.com/v1"
        ).api_key == "deepseek-secret"


def test_bootstrap_does_not_relabel_provider_key_as_explicit(monkeypatch):
    for name in (
        "HYMEM_LLM_API_KEY", "HYMEM_EMBEDDING_API_KEY",
        "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
        "HYMEM_EMBEDDING_BASE_URL", EMBEDDING_INTERNAL_HTTP_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HYMEM_LLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    assert resolve_env().llm_api_key is None

    monkeypatch.setenv("HYMEM_LLM_API_KEY", "purpose-key")
    assert resolve_env().llm_api_key == "purpose-key"


def test_bootstrap_internal_embedding_deployment_requires_opt_in(monkeypatch):
    for name in (
        "HYMEM_EMBEDDING_API_KEY", "OPENAI_API_KEY",
        EMBEDDING_INTERNAL_HTTP_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "HYMEM_EMBEDDING_BASE_URL", "http://embedding-server:8766/v1"
    )
    rejected = resolve_env()
    assert rejected.embedding_backend == "local_feature_hash"
    assert rejected.embedding_fallback_reason == "remote_embedding_endpoint_rejected"

    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, "1")
    accepted = resolve_env()
    assert accepted.embedding_backend == "openai_compatible"
    assert accepted.embedding_api_key == "local"


def test_benchmark_embedding_artifacts_are_self_attesting(monkeypatch):
    base_url = "http://embedding-server:8766/v1"
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, "1")
    beam_identity = beam.public_embedding_config(beam.resolve_embedding_config(
        "openai-compatible",
        base_url=base_url,
        model="embedding-model",
        dimension=384,
        deployment_revision="fixture-v1",
        deployment_tenant="fixture-tenant",
    ))
    lme_identity = lme.resolve_embedding_identity(SimpleNamespace(
        embeddings=True,
        embedding_base_url=base_url,
        embedding_model="embedding-model",
        embedding_dim=384,
        embedding_deployment_revision="fixture-v1",
        embedding_deployment_tenant="fixture-tenant",
    ))
    assert beam_identity["transport_security"] == "explicit-internal-http"
    assert lme_identity["transport_security"] == "explicit-internal-http"

    # Offline validation is deterministic after the runtime authorization that
    # created the identity is no longer present in this process.
    monkeypatch.delenv(EMBEDDING_INTERNAL_HTTP_ENV, raising=False)
    beam_registry._validate_embedding_config(beam_identity)
    assert lme_protocol._validate_embedding_identity(lme_identity) == lme_identity

    forged_beam = dict(beam_identity, transport_security="https")
    forged_lme = dict(lme_identity, transport_security="https")
    with pytest.raises(ValueError, match="malformed|transport posture"):
        beam_registry._validate_embedding_config(forged_beam)
    with pytest.raises(
        lme_protocol.BenchmarkIntegrityError, match="malformed|transport posture"
    ):
        lme_protocol._validate_embedding_identity(forged_lme)

    forged_public = dict(
        beam_identity,
        base_url="http://example.com/v1",
        transport_security="explicit-internal-http",
    )
    with pytest.raises(ValueError, match="malformed|transport posture"):
        beam_registry._validate_embedding_config(forged_public)

    # The ambient opt-in cannot rescue an internal-HTTP artifact whose
    # attestation was removed after publication.
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, "1")
    unattested_beam = dict(beam_identity)
    unattested_lme = dict(lme_identity)
    unattested_beam.pop("transport_security")
    unattested_lme.pop("transport_security")
    with pytest.raises(ValueError):
        beam_registry._validate_embedding_config(unattested_beam)
    with pytest.raises(lme_protocol.BenchmarkIntegrityError):
        lme_protocol._validate_embedding_identity(unattested_lme)
