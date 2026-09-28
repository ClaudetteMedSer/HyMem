"""Current requested-service admission is distinct from historical identity."""

from copy import deepcopy

import pytest

from benchmarks import beam_adapter as beam, lme_protocol, longmemeval_adapter as lme
from hymem import bootstrap
from hymem.contrib import model_policy, openai_client
from hymem.contrib.endpoint_policy import secret_free_endpoint_identity
from hymem.extraction.producer import producer_binding_from_typed_declaration
from tests.test_lme_protocol_hardening import make_artifact, _refresh_manifest


CURRENT = "deepseek-flash"
DISABLED = {"thinking": {"type": "disabled"}}
CONTRACT = "official-deepseek-flash-request-service-at-api.deepseek.com-v1"
RETIRED = ("deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash",
           "deepseek-v4-flash-vision-exp")


@pytest.fixture(autouse=True)
def no_ambient_llm_config(monkeypatch):
    for name in (
        "HYMEM_LLM_MODEL", "HYMEM_LLM_BASE_URL", "HYMEM_LLM_THINKING",
        "HYMEM_LLM_DEPLOYMENT_REVISION", "HYMEM_LLM_DEPLOYMENT_TENANT",
        "HYMEM_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def declaration(**overrides):
    kwargs = dict(
        model=CURRENT, endpoint="https://api.deepseek.com", thinking_mode="auto",
        effective_extra_body=DISABLED, transport_package_version="fixture-sdk",
        request_timeout_seconds=120.0,
    )
    kwargs.update(overrides)
    return openai_client.openai_compatible_producer_declaration(**kwargs)


@pytest.mark.parametrize("model", RETIRED)
@pytest.mark.parametrize("form", ("{}", "  {}  ", "openai:{}", "openrouter:deepseek/{}"))
def test_retired_requests_fail_before_any_credential_or_bootstrap_resolution(
    monkeypatch, model, form,
):
    selected = form.format(model.upper())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("retired selection reached setup")

    monkeypatch.setattr(openai_client, "resolve_llm_api_key", forbidden)
    monkeypatch.setattr(bootstrap, "resolve_env", forbidden)
    with pytest.raises(model_policy.DeprecatedModelAliasError):
        openai_client.OpenAICompatibleClient(model=selected)
    # Constructor and standalone producer admit the identical selected model.
    with pytest.raises(model_policy.DeprecatedModelAliasError):
        declaration(model=selected)
    monkeypatch.setenv("HYMEM_LLM_MODEL", selected)
    with pytest.raises(model_policy.DeprecatedModelAliasError):
        openai_client.OpenAICompatibleClient()
    with pytest.raises(model_policy.DeprecatedModelAliasError):
        bootstrap.build_from_env()


@pytest.mark.parametrize("thinking,body", (("auto", DISABLED), ("disabled", DISABLED),
                                           ("off", {}), ("enabled", {})))
def test_current_client_and_standalone_declarations_are_identical(thinking, body):
    client = openai_client.OpenAICompatibleClient(
        api_key="synthetic-fixture-key", model=f"  {CURRENT}  ", thinking=thinking,
    )
    try:
        actual = client.phase1_producer_declaration()
        expected = declaration(
            model=f"  {CURRENT}  ", thinking_mode=thinking,
            effective_extra_body=body,
            transport_package_version=client.transport_package_version,
        )
        assert actual == expected == client.aggregation_producer_declaration()
        assert actual.model == client.model == CURRENT
        assert actual.effective_request["official_deployment_contract"] == CONTRACT
        assert client.effective_extra_body == body
        binding = producer_binding_from_typed_declaration(
            actual, declaration_hook="phase1_producer_declaration",
        )
        assert binding["identity_exact"] is True
        assert binding["reuse_scope"] == "durable"
    finally:
        client.close()


@pytest.mark.parametrize("endpoint", (
    "https://api.deepseek.com", "https://api.deepseek.com/v1",
    "https://gateway.example/v1",
))
@pytest.mark.parametrize("missing", ("revision", "tenant"))
def test_partial_attestation_never_uses_official_exception(endpoint, missing):
    fields = dict(deployment_revision_sha256="sha256:" + "a" * 64,
                  deployment_tenant_sha256="sha256:" + "b" * 64)
    fields[f"deployment_{missing}_sha256"] = None
    with pytest.raises(ValueError, match="revision and tenant"):
        declaration(endpoint=endpoint, **fields)


def test_custom_endpoint_still_needs_both_attestations():
    with pytest.raises(ValueError, match="revision and tenant"):
        declaration(endpoint="https://gateway.example/v1")
    result = declaration(
        endpoint="https://gateway.example/v1",
        deployment_revision_sha256="sha256:" + "a" * 64,
        deployment_tenant_sha256="sha256:" + "b" * 64,
    )
    assert result.effective_request["official_deployment_contract"] is None


def test_old_official_contract_cannot_be_asserted_for_current_service():
    with pytest.raises(ValueError, match="revision and tenant"):
        declaration(official_deployment_contract=
                    "official-deepseek-v4-flash-at-api.deepseek.com-v1")


@pytest.mark.parametrize("model", (CURRENT, "deepseek-v4-flash",
                                   "deepseek-v4-flash-vision-exp"))
def test_pure_historical_body_transforms_retain_old_and_current_flash(model):
    body, defaulted = lme.resolve_model_extra_body(model, "https://api.deepseek.com", None)
    assert (body, defaulted) == (DISABLED, True)
    body["thinking"]["type"] = "mutated"
    assert lme.resolve_model_extra_body(model, "https://api.deepseek.com", None) == (DISABLED, True)
    assert lme.resolve_model_extra_body(model, "https://gateway.example/v1", None) == ({}, False)
    for incompatible in ({}, {"thinking": {"type": "enabled"}}):
        with pytest.raises(lme.BenchmarkIntegrityError):
            lme.resolve_model_extra_body(model, "https://api.deepseek.com", incompatible)
        assert beam.apply_thinking_default("answer", model, "deepseek", False, incompatible) == (incompatible, False)
        with pytest.raises(SystemExit):
            beam.check_model_pin("answer", model, "deepseek", incompatible)
    assert beam.apply_thinking_default("answer", model, "deepseek", True, {}) == (DISABLED, True)
    beam.check_model_pin("answer", model, "deepseek", DISABLED)


def test_current_official_service_artifact_remains_archivable_after_retirement(monkeypatch):
    artifact = make_artifact()
    config, pipeline = artifact["config"], artifact["models"]["memory_pipeline"]
    identity = secret_free_endpoint_identity("https://api.deepseek.com", label="fixture")
    config.update(hymem_model=CURRENT, hymem_thinking="auto")
    config.update({f"hymem_{key}": value for key, value in identity.items()})
    pipeline.update(provider="deepseek", model=CURRENT, thinking_mode="auto", effective_extra_body=deepcopy(DISABLED),
                    deployment_revision_sha256=None, deployment_tenant_sha256=None, **identity)
    pipeline["aggregation_producer"] = producer_binding_from_typed_declaration(
        declaration(transport_package_version=pipeline["transport_package_version"]),
        declaration_hook="aggregation_producer_declaration",
    )
    _refresh_manifest(artifact)
    lme_protocol.validate_strict_artifact(artifact)
    original = deepcopy(artifact)
    monkeypatch.setattr(model_policy, "DEPRECATED_DEEPSEEK_ALIASES",
                        model_policy.DEPRECATED_DEEPSEEK_ALIASES | {CURRENT})
    with pytest.raises(lme.BenchmarkIntegrityError, match="aggregation identity"):
        lme_protocol.validate_strict_artifact(artifact)
    assert lme_protocol.validate_archived_artifact(artifact)["live_execution_eligible"] is False
    assert artifact == original


def test_current_defaults_are_one_shared_selection():
    assert model_policy.RECOMMENDED_DEEPSEEK_MODEL == bootstrap.DEFAULT_LLM_MODEL == CURRENT
    assert lme.PINNED_DEEPSEEK_MODEL == beam.HYMEM_MODEL == CURRENT
