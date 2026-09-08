"""Actionable endpoint rejection without authorizing HTTP or exposing secrets."""

from __future__ import annotations

import os
from dataclasses import asdict

import pytest

from hymem import bootstrap, doctor
from hymem.bootstrap import EnvConfig, resolve_env
from hymem.contrib.endpoint_policy import EMBEDDING_INTERNAL_HTTP_ENV
from hymem.doctor import FAIL, OK, WARN, _check_embedding
from hymem.extraction.llm import StubLLMClient


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("HYMEM_") or name in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}:
            monkeypatch.delenv(name)
    monkeypatch.setenv("HYMEM_ROOT", str(tmp_path))


@pytest.mark.parametrize("flag", [None, "0", "false"])
def test_internal_http_rejection_retains_actionable_policy_detail(monkeypatch, flag):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "http://embedding-server:8766")
    if flag is not None:
        monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, flag)
    cfg = resolve_env()
    result, _, _ = _check_embedding(cfg)
    assert cfg.embedding_backend == "local_feature_hash"
    assert cfg.embedding_fallback_reason == "remote_embedding_endpoint_rejected"
    assert result.status == FAIL
    assert f"set {EMBEDDING_INTERNAL_HTTP_ENV}=1" in result.detail
    assert "isolated internal service network" in result.detail
    assert cfg.embedding_fallback_detail in result.detail
    assert len(cfg.embedding_fallback_detail) <= 512


@pytest.mark.parametrize("url", [
    "http://embedding-server:8766/v1/tenant/opaque-route-sentinel",
    "http://10.23.4.5:8766/v1",
    "http://embeddings.svc.cluster.local:8766/v1",
])
def test_valid_internal_hint_never_reproduces_the_route(monkeypatch, url):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", url)
    cfg = resolve_env()
    result, _, _ = _check_embedding(cfg)
    assert result.status == FAIL
    assert f"set {EMBEDDING_INTERNAL_HTTP_ENV}=1" in result.detail
    assert url not in result.detail
    assert "opaque-route-sentinel" not in result.detail
    assert EMBEDDING_INTERNAL_HTTP_ENV not in os.environ


@pytest.mark.parametrize("url", [
    "http://operator:private-sentinel@embedding-server:8766/v1",
    "http://embedding-server:8766/v1?key=private-sentinel",
    "http://embedding-server:8766/v1#private-sentinel",
    "http://embedding-server:private-sentinel/v1",
    "http://[private-sentinel/v1",
    "http://embedding-server:8766/ private-sentinel",
    "http://embedding-server:8766\\private-sentinel/v1",
    "ftp://embedding-server/private-sentinel",
    "http://embedding-server:8766/token/private-sentinel",
    "http://embedding-server:8766/api%252dkey/private-sentinel",
    "http://api-key-private-sentinel:8766/v1",
])
def test_unsafe_url_rejection_is_secret_free_without_internal_opt_in_advice(
    monkeypatch, url,
):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", url)
    cfg = resolve_env()
    result, _, _ = _check_embedding(cfg)
    assert cfg.embedding_fallback_reason == "remote_embedding_endpoint_rejected"
    assert result.status == FAIL
    assert cfg.embedding_fallback_detail
    assert len(cfg.embedding_fallback_detail) <= 512
    assert "private-sentinel" not in result.render()
    assert url not in result.render()
    assert "operator" not in result.render()
    assert f"set {EMBEDDING_INTERNAL_HTTP_ENV}=1" not in result.detail
    assert EMBEDDING_INTERNAL_HTTP_ENV not in os.environ


@pytest.mark.parametrize("url", ["http://example.com/v1", "http://8.8.8.8/v1"])
@pytest.mark.parametrize("flag", [None, "1", "private-sentinel"])
def test_public_http_never_gets_opt_in_advice_or_authorization(monkeypatch, url, flag):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", url)
    if flag is not None:
        monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, flag)
    cfg = resolve_env()
    result, _, _ = _check_embedding(cfg)
    assert cfg.embedding_backend == "local_feature_hash"
    assert result.status == FAIL
    assert "private-sentinel" not in result.detail
    assert f"set {EMBEDDING_INTERNAL_HTTP_ENV}=1" not in result.detail


@pytest.mark.parametrize("flag", ["private-sentinel", "private-sentinel\n" * 500])
def test_invalid_flag_reports_allowed_values_without_echo(monkeypatch, flag):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "http://embedding-server:8766")
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, flag)
    cfg = resolve_env()
    result, _, _ = _check_embedding(cfg)
    assert result.status == FAIL
    assert "must be one of 1/true/yes/on or 0/false/no/off" in result.detail
    assert "private-sentinel" not in result.detail
    assert len(cfg.embedding_fallback_detail) <= 512


def _configured_remote(monkeypatch, url="http://embedding-server:8766"):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", url)
    monkeypatch.setenv("HYMEM_EMBEDDING_MODEL", "operator-embedding-release")
    monkeypatch.setenv("HYMEM_EMBEDDING_DIM", "7")
    monkeypatch.setenv("HYMEM_EMBEDDING_PIN_DIMENSION", "1")
    monkeypatch.setenv("HYMEM_EMBEDDING_DEPLOYMENT_REVISION", "operator-release-2026-09")
    monkeypatch.setenv("HYMEM_EMBEDDING_DEPLOYMENT_TENANT", "operator-workspace")
    monkeypatch.setenv("HYMEM_LLM_API_KEY", "llm-private-sentinel")
    monkeypatch.setenv("OPENAI_API_KEY", "cloud-private-sentinel")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-private-sentinel")


@pytest.mark.parametrize("flag", ["1", "true", "yes", "on"])
def test_explicit_internal_opt_in_preserves_config_and_never_inherits_cloud_keys(
    monkeypatch, flag,
):
    _configured_remote(monkeypatch)
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, flag)
    cfg = resolve_env()
    assert cfg.embedding_backend == "openai_compatible"
    assert cfg.embedding_api_key == "local"
    assert cfg.embedding_base_url == "http://embedding-server:8766"
    assert cfg.embedding_model == "operator-embedding-release"
    assert cfg.embedding_dim == 7
    assert cfg.embedding_pin_dimension is True
    assert cfg.embedding_deployment_revision == "operator-release-2026-09"
    assert cfg.embedding_deployment_tenant == "operator-workspace"
    assert cfg.embedding_fallback_reason is None
    assert cfg.embedding_fallback_detail is None


@pytest.mark.parametrize(("url", "key", "flag"), [
    ("http://embedding-server:8766", None, "1"),
    ("http://127.0.0.1:8766/v1", None, None),
    ("http://localhost:8766/v1", None, None),
    ("https://embeddings.example/v1", "embedding-purpose-key", None),
    ("https://api.openai.com/v1", None, None),
])
def test_allowed_endpoint_constructs_unchanged_real_transport_without_requests(
    monkeypatch, url, key, flag,
):
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient

    _configured_remote(monkeypatch, url)
    if key is not None:
        monkeypatch.setenv("HYMEM_EMBEDDING_API_KEY", key)
    if flag is not None:
        monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, flag)
    cfg = resolve_env()
    assert cfg.embedding_backend == "openai_compatible"
    assert cfg.embedding_fallback_detail is None
    expected_key = (
        "cloud-private-sentinel" if url == "https://api.openai.com/v1"
        else key or "local"
    )
    assert cfg.embedding_api_key == expected_key
    client = OpenAICompatibleEmbeddingClient(
        api_key=cfg.embedding_api_key, base_url=cfg.embedding_base_url,
        model=cfg.embedding_model, dim=cfg.embedding_dim,
        pin_dimension=cfg.embedding_pin_dimension,
        deployment_revision=cfg.embedding_deployment_revision,
        deployment_tenant=cfg.embedding_deployment_tenant,
    )
    try:
        assert client.request_model == cfg.embedding_model
        assert client.configured_dim == 7
        assert client.dimension_policy == "pinned"
        assert client.request_attempts == 0
    finally:
        client.close()


@pytest.mark.parametrize("missing", [
    "HYMEM_EMBEDDING_PIN_DIMENSION",
    "HYMEM_EMBEDDING_DEPLOYMENT_REVISION",
    "HYMEM_EMBEDDING_DEPLOYMENT_TENANT",
])
def test_internal_opt_in_does_not_waive_producer_authority(monkeypatch, missing):
    _configured_remote(monkeypatch)
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, "1")
    monkeypatch.delenv(missing)
    cfg = resolve_env()
    result, _, _ = _check_embedding(cfg)
    assert result.status == FAIL
    assert "pinned dimension/deployment/tenant authority" in result.detail
    with pytest.raises(RuntimeError, match="remote embeddings require"):
        bootstrap.build_from_env()


def test_rejected_diagnostic_is_a_snapshot_not_recomputed_from_later_env(monkeypatch):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "http://embedding-server:8766")
    rejected = resolve_env()
    monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, "1")
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://example.com?key=private-sentinel")
    result, _, _ = _check_embedding(rejected)
    assert result.status == FAIL
    assert rejected.embedding_fallback_detail in result.detail
    assert f"set {EMBEDDING_INTERNAL_HTTP_ENV}=1" in result.detail
    assert "private-sentinel" not in result.detail


def test_intentionally_local_backend_remains_ok_and_envconfig_field_is_optional():
    cfg = resolve_env()
    assert cfg.embedding_backend == "local_feature_hash"
    assert cfg.embedding_fallback_reason is None
    assert cfg.embedding_fallback_detail is None
    result, _, _ = _check_embedding(cfg)
    assert result.status == OK
    legacy_fields = asdict(cfg)
    legacy_fields.pop("embedding_fallback_detail")
    assert EnvConfig(**legacy_fields).embedding_fallback_detail is None


def test_custom_https_missing_credentials_keeps_existing_warning(monkeypatch):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://embeddings.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "cloud-private-sentinel")
    cfg = resolve_env()
    result, _, _ = _check_embedding(cfg)
    assert result.status == WARN
    assert cfg.embedding_fallback_reason == "remote_embedding_credentials_missing"
    assert cfg.embedding_fallback_detail is None
    assert "private-sentinel" not in result.detail


@pytest.mark.parametrize(("url", "flag", "expect_hint"), [
    ("http://embedding-server:8766", None, True),
    ("http://operator:private-sentinel@embedding-server:8766", None, False),
    ("http://embedding-server:8766?key=private-sentinel", None, False),
    ("http://embedding-server:8766/token/private-sentinel", None, False),
    ("http://embedding-server:8766", "private-sentinel", False),
])
def test_startup_logs_safe_rejection_without_constructing_remote_embedder(
    monkeypatch, caplog, url, flag, expect_hint,
):
    import hymem.contrib.openai_client as llm_module
    import hymem.contrib.openai_embedding_client as embedding_module

    _configured_remote(monkeypatch, url)
    if flag is not None:
        monkeypatch.setenv(EMBEDDING_INTERNAL_HTTP_ENV, flag)
    monkeypatch.setattr(llm_module, "OpenAICompatibleClient", lambda **_: StubLLMClient())

    def forbidden_remote(**_):
        raise AssertionError("rejected endpoint constructed a remote client")

    monkeypatch.setattr(embedding_module, "OpenAICompatibleEmbeddingClient", forbidden_remote)
    with caplog.at_level("WARNING", logger="hymem.bootstrap"):
        memory = bootstrap.build_from_env()
    try:
        assert memory.embedding_status["backend"] == "local_feature_hash"
        assert memory.embedding_status["fallback_reason"] == "remote_embedding_endpoint_rejected"
        assert (f"set {EMBEDDING_INTERNAL_HTTP_ENV}=1" in caplog.text) is expect_hint
        assert "run hymem-doctor" in caplog.text
        assert "private-sentinel" not in caplog.text
    finally:
        bootstrap.shutdown_instance(memory)


def test_doctor_cli_remains_failed_and_renders_actionable_policy(monkeypatch, capsys):
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "http://embedding-server:8766")
    monkeypatch.setattr(doctor, "_check_llm", lambda _: doctor._Result(OK, "LLM", "offline seam"))
    assert doctor.run_doctor() == 1
    output = capsys.readouterr().out
    assert "[FAIL] embeddings:" in output
    assert f"set {EMBEDDING_INTERNAL_HTTP_ENV}=1" in output
    assert "All checks passed" not in output
