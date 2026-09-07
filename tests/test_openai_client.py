"""Tests for the OpenAI-compatible LLM client's vendor-specific body gating.

None of these hit the network: `openai.OpenAI` is replaced with a recorder that
captures the kwargs handed to `chat.completions.create`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import threading
from typing import Any

import pytest

from hymem.contrib.openai_client import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    OpenAICompatibleClient,
)
from hymem.dreaming.runner import _CountingPhase1LLM
from hymem.extraction.chunk import extract_chunk
from hymem.extraction.llm import LLMRequest
from hymem.extraction.producer import phase1_generation_binding

_LLM_ENV = (
    "HYMEM_LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "HYMEM_LLM_BASE_URL",
    "HYMEM_LLM_MODEL",
    "HYMEM_LLM_THINKING",
    "HYMEM_LLM_DEPLOYMENT_REVISION",
    "HYMEM_LLM_DEPLOYMENT_TENANT",
)


class _RecordingCompletions:
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    def create(self, **kwargs: Any) -> Any:
        self._sink.append(kwargs)

        class _Message:
            content = "ok"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _RecordingOpenAI:
    """Stand-in for openai.OpenAI; records construction and call kwargs."""

    def __init__(self, sink: list[dict[str, Any]], init_sink: list[dict[str, Any]]):
        self._sink = sink
        self._init_sink = init_sink

    def __call__(self, **kwargs: Any) -> Any:
        self._init_sink.append(kwargs)
        sink = self._sink
        http_client = kwargs.get("http_client")

        class _Client:
            _client = http_client

            class chat:  # noqa: N801 - mirrors the openai SDK's attribute shape
                completions = _RecordingCompletions(sink)

            @staticmethod
            def close() -> None:
                if http_client is not None:
                    http_client.close()

        return _Client()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Isolate the LLM env, stub out openai.OpenAI, and expose captured calls."""
    import openai

    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HYMEM_LLM_API_KEY", "test-key")

    sink: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, "OpenAI", _RecordingOpenAI(sink, []))
    return sink


def _complete(client: OpenAICompatibleClient, response_format: str = "text") -> None:
    client.complete(
        LLMRequest(system="sys", user="usr", response_format=response_format)
    )


def test_default_construction_sends_thinking_disabled(calls) -> None:
    # Defaults resolve to the DeepSeek base URL + deepseek-v4-flash, which is
    # exactly the configuration the body key exists for.
    _complete(OpenAICompatibleClient())
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_request_identity_fields_are_immutable_after_construction(calls) -> None:
    client = OpenAICompatibleClient()
    baseline = client.aggregation_producer_declaration()
    for field, value in (
        ("model", "deepseek-chat"),
        ("base_url", "https://other.example/v1"),
        ("thinking_mode", "off"),
        ("transport_package_version", "999-forged"),
    ):
        with pytest.raises(AttributeError):
            setattr(client, field, value)
    assert client.aggregation_producer_declaration() == baseline
    _complete(client)
    assert calls[0]["model"] == "deepseek-v4-flash"


def test_retry_default_change_rotates_aggregation_producer_identity(
    calls, monkeypatch,
) -> None:
    from hymem.contrib import openai_client as client_module
    from hymem.extraction.producer import producer_binding_from_typed_declaration

    client = OpenAICompatibleClient()
    before = producer_binding_from_typed_declaration(
        client.aggregation_producer_declaration(),
        declaration_hook="aggregation_producer_declaration",
    )
    monkeypatch.setattr(
        client_module, "DEFAULT_RETRY_ATTEMPTS",
        client_module.DEFAULT_RETRY_ATTEMPTS + 1,
    )
    after = producer_binding_from_typed_declaration(
        client.aggregation_producer_declaration(),
        declaration_hook="aggregation_producer_declaration",
    )
    assert before["identity_sha256"] != after["identity_sha256"]


def test_exact_openai_llm_binding_revokes_on_dispatch_transport_and_close(calls):
    client = OpenAICompatibleClient()
    baseline = phase1_generation_binding("fixture-prompt-v1", client)
    assert baseline["producer"]["identity_exact"] is True
    assert client.transport_integrity_ok is True

    original_client = client._client
    client._client = object()
    with pytest.raises(ValueError, match="integrity"):
        phase1_generation_binding("fixture-prompt-v1", client)
    client._client = original_client
    assert phase1_generation_binding("fixture-prompt-v1", client) == baseline

    client.complete = lambda _request: "forced"
    with pytest.raises(ValueError, match="integrity"):
        phase1_generation_binding("fixture-prompt-v1", client)
    del client.complete

    client.close()
    with pytest.raises(ValueError, match="integrity"):
        phase1_generation_binding("fixture-prompt-v1", client)
    with pytest.raises(RuntimeError, match="identity changed"):
        _complete(client)


def test_completion_resource_instance_shadow_revokes_authority(calls):
    client = OpenAICompatibleClient()
    baseline = phase1_generation_binding("fixture-prompt-v1", client)
    resource = client._client.chat.completions
    resource.create = lambda **_kwargs: _provider_response("forced")
    try:
        with pytest.raises(ValueError, match="integrity"):
            phase1_generation_binding("fixture-prompt-v1", client)
        with pytest.raises(RuntimeError, match="identity changed"):
            _complete(client)
    finally:
        del resource.create
    assert phase1_generation_binding("fixture-prompt-v1", client) == baseline


@pytest.mark.parametrize("failure_stage", ["resource", "transport_state"])
def test_constructor_seal_failure_closes_owned_http_client(
    monkeypatch, failure_stage,
) -> None:
    import openai
    from hymem.contrib import openai_client as client_module

    closed: list[str] = []

    class HTTP:
        def close(self):
            closed.append("closed")

    monkeypatch.setattr(openai, "DefaultHttpxClient", lambda **_kwargs: HTTP())

    if failure_stage == "resource":
        class SDK:
            _client = None

            @property
            def chat(self):
                raise LookupError("resource capture failed")

        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SDK())
    else:
        class SDK:
            def __init__(self, **kwargs):
                self._client = kwargs["http_client"]
                self.chat = SimpleNamespace(completions=_RecordingCompletions([]))

        from types import SimpleNamespace
        monkeypatch.setattr(openai, "OpenAI", SDK)
        monkeypatch.setattr(
            client_module.OpenAICompatibleClient,
            "_transport_state",
            lambda _self: (_ for _ in ()).throw(
                LookupError("transport seal failed")
            ),
        )

    with pytest.raises(LookupError, match="failed"):
        OpenAICompatibleClient(api_key="wire-key")
    assert closed == ["closed"]


def test_custom_llm_requires_public_deployment_attestations_for_durable_reuse(
    calls,
) -> None:
    first = OpenAICompatibleClient(
        api_key="route-a", base_url="https://gateway.example/v1", model="m",
    )
    second = OpenAICompatibleClient(
        api_key="route-b", base_url="https://gateway.example/v1", model="m",
    )
    first_binding = phase1_generation_binding("fixture-prompt-v1", first)
    second_binding = phase1_generation_binding("fixture-prompt-v1", second)
    assert first_binding["producer"]["identity_exact"] is False
    assert second_binding["producer"]["identity_exact"] is False
    assert first_binding["generation_key"] != second_binding["generation_key"]

    exact_a = OpenAICompatibleClient(
        api_key="route-a", base_url="https://gateway.example/v1", model="m",
        deployment_revision="public-release-a",
        deployment_tenant="public-tenant",
    )
    exact_b = OpenAICompatibleClient(
        api_key="route-b", base_url="https://gateway.example/v1", model="m",
        deployment_revision="public-release-b",
        deployment_tenant="public-tenant",
    )
    binding_a = phase1_generation_binding("fixture-prompt-v1", exact_a)
    binding_b = phase1_generation_binding("fixture-prompt-v1", exact_b)
    assert binding_a["producer"]["identity_exact"] is True
    assert binding_b["producer"]["identity_exact"] is True
    assert binding_a["generation_key"] != binding_b["generation_key"]
    encoded = json.dumps(binding_a, sort_keys=True)
    assert "public-release-a" not in encoded
    assert "route-a" not in encoded


@pytest.mark.parametrize("credential", [
    "sk_live_abcdefghijk",
    "Authorization: Basic dXNlcjpwYXNz",
    "https://proxy.example/v1/sk-live-secret-abcdefgh",
])
def test_llm_public_attestations_reject_credentials_without_echo(calls, credential):
    with pytest.raises(ValueError) as exc_info:
        OpenAICompatibleClient(
            api_key="wire-key",
            base_url="https://gateway.example/v1",
            model="m",
            deployment_revision=credential,
            deployment_tenant="public-tenant",
        )
    assert credential not in str(exc_info.value)


def test_openai_llm_nested_pool_policy_mutation_revokes_exact_binding(
    monkeypatch,
):
    import openai

    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HYMEM_LLM_API_KEY", "test-key")
    client = OpenAICompatibleClient()
    baseline = phase1_generation_binding("fixture-prompt-v1", client)
    pool = client._client._client._transport._pool
    original = pool._http2
    pool._http2 = not original
    try:
        with pytest.raises(ValueError, match="integrity"):
            phase1_generation_binding("fixture-prompt-v1", client)
    finally:
        pool._http2 = original
        client.close()
    assert baseline["producer"]["identity_exact"] is True


def test_sdk_construction_disables_hidden_retries_and_sets_finite_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, "OpenAI", _RecordingOpenAI([], init_calls))
    client = OpenAICompatibleClient(api_key="test-key")

    assert client.request_attempts == 0
    assert len(init_calls) == 1
    owned_http = init_calls[0].pop("http_client")
    assert getattr(owned_http, "trust_env", False) is False
    assert init_calls == [{
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "organization": "",
        "project": "",
        "timeout": DEFAULT_LLM_TIMEOUT_SECONDS,
        "max_retries": 0,
    }]
    assert math.isfinite(DEFAULT_LLM_TIMEOUT_SECONDS)
    assert DEFAULT_LLM_TIMEOUT_SECONDS > 0


def test_counted_retries_equal_sdk_http_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SDK retry may hide behind one increment of request_attempts."""
    import openai

    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    wire_attempts = 0
    init_calls: list[dict[str, Any]] = []

    class _FailingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            init_calls.append(kwargs)
            self._client = kwargs.get("http_client")
            max_retries = kwargs.get("max_retries", 2)

            class _Completions:
                @staticmethod
                def create(**_request: Any) -> Any:
                    nonlocal wire_attempts
                    # Model the SDK's retry multiplicity. With max_retries=0,
                    # one create invocation corresponds to one wire attempt.
                    wire_attempts += 1 + int(max_retries)
                    raise TimeoutError("simulated provider timeout")

            class _Chat:
                completions = _Completions()

            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", _FailingOpenAI)
    monkeypatch.setattr("hymem.extraction.retry.time.sleep", lambda _delay: None)
    client = OpenAICompatibleClient(api_key="test-key")

    result = extract_chunk(client, "The app uses PostgreSQL.")

    assert init_calls[0]["max_retries"] == 0
    assert result.failed is True
    assert result.completion_calls == 1
    assert result.provider_attempts == 3
    assert client.request_attempts == wire_attempts == 3
    assert client.call_count == client.successful_responses == 0


def _provider_response(content: str) -> Any:
    class _Message:
        pass

    class _Choice:
        pass

    class _Response:
        pass

    message = _Message()
    message.content = content
    choice = _Choice()
    choice.message = message
    response = _Response()
    response.choices = [choice]
    return response


def test_chunk_attempt_meter_excludes_overlapping_shared_client_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent ask/rerank call cannot be charged to Phase-1."""
    import openai

    phase_started = threading.Event()
    release_phase = threading.Event()
    first_phase_lock = threading.Lock()
    first_phase = True

    class _ConcurrentOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self._client = _kwargs.get("http_client")

            class _Completions:
                @staticmethod
                def create(**request: Any) -> Any:
                    nonlocal first_phase
                    user = request["messages"][1]["content"]
                    if user == "DIRECT SHARED CALL":
                        return _provider_response("direct response")
                    with first_phase_lock:
                        should_block = first_phase
                        first_phase = False
                    if should_block:
                        phase_started.set()
                        if not release_phase.wait(timeout=5):
                            raise TimeoutError("test did not release Phase-1")
                    return _provider_response(json.dumps({
                        "triples": [], "markers": [], "complete": True,
                    }))

            class _Chat:
                completions = _Completions()

            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", _ConcurrentOpenAI)
    client = OpenAICompatibleClient(api_key="test-key")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            extract_chunk, client, "The application uses PostgreSQL."
        )
        assert phase_started.wait(timeout=5)
        try:
            assert client.complete(LLMRequest(
                system="direct", user="DIRECT SHARED CALL",
                response_format="text",
            )) == "direct response"
        finally:
            release_phase.set()
        result = future.result(timeout=5)

    # Clean-empty verification makes two Phase-1 completions. The overlapping
    # direct completion is visible in the global benchmark total only.
    assert result.completion_calls == result.provider_attempts == 2
    assert client.request_attempts == 3
    assert client.call_count == client.successful_responses == 3


def test_nested_runner_and_chunk_meters_isolate_retrying_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both Phase-1 meters retain three own failed attempts, not the overlap."""
    import openai

    phase_started = threading.Event()
    release_phase = threading.Event()
    phase_attempt_lock = threading.Lock()
    phase_attempts = 0

    class _ConcurrentFailingOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self._client = _kwargs.get("http_client")

            class _Completions:
                @staticmethod
                def create(**request: Any) -> Any:
                    nonlocal phase_attempts
                    user = request["messages"][1]["content"]
                    if user == "DIRECT SHARED CALL":
                        return _provider_response("direct response")
                    with phase_attempt_lock:
                        phase_attempts += 1
                        attempt = phase_attempts
                    if attempt == 1:
                        phase_started.set()
                        if not release_phase.wait(timeout=5):
                            raise TimeoutError("test did not release Phase-1")
                    raise TimeoutError(f"phase provider failure {attempt}")

            class _Chat:
                completions = _Completions()

            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", _ConcurrentFailingOpenAI)
    monkeypatch.setattr("hymem.extraction.retry.time.sleep", lambda _delay: None)
    client = OpenAICompatibleClient(api_key="test-key")
    counting = _CountingPhase1LLM(client)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            extract_chunk, counting, "The Phase-1 request will fail."
        )
        assert phase_started.wait(timeout=5)
        try:
            assert client.complete(LLMRequest(
                system="direct", user="DIRECT SHARED CALL",
                response_format="text",
            )) == "direct response"
        finally:
            release_phase.set()
        result = future.result(timeout=5)

    assert result.failed is True
    assert result.completion_calls == counting.completion_calls == 1
    assert result.provider_attempts == counting.provider_attempts == 3
    assert phase_attempts == 3
    assert client.request_attempts == 4
    assert client.call_count == client.successful_responses == 1


def test_deepseek_regional_host_still_sends_it(calls) -> None:
    _complete(
        OpenAICompatibleClient(
            base_url="https://api-eu.deepseek.com/v1/proxy", model="some-model"
        )
    )
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_model_behind_a_gateway_still_sends_it(calls) -> None:
    _complete(
        OpenAICompatibleClient(
            base_url="https://gateway.internal/v1", model="deepseek-v4-flash"
        )
    )
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openai_endpoint_omits_extra_body(calls) -> None:
    _complete(
        OpenAICompatibleClient(base_url="https://api.openai.com/v1", model="gpt-4o-mini")
    )
    assert "extra_body" not in calls[0]


def test_local_vllm_endpoint_omits_extra_body(calls) -> None:
    _complete(
        OpenAICompatibleClient(
            base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct"
        )
    )
    assert "extra_body" not in calls[0]


def test_env_override_forces_it_on_for_non_deepseek(
    calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYMEM_LLM_THINKING", "disabled")
    _complete(
        OpenAICompatibleClient(base_url="http://localhost:8000/v1", model="local-model")
    )
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_env_override_forces_it_off_for_deepseek(
    calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYMEM_LLM_THINKING", "off")
    _complete(OpenAICompatibleClient())
    assert "extra_body" not in calls[0]


def test_constructor_argument_forces_it_on(calls) -> None:
    _complete(
        OpenAICompatibleClient(
            base_url="https://api.openai.com/v1", model="gpt-4o-mini", thinking="disabled"
        )
    )
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_constructor_argument_forces_it_off(calls) -> None:
    _complete(OpenAICompatibleClient(thinking="enabled"))
    assert "extra_body" not in calls[0]


def test_constructor_argument_beats_env(calls, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYMEM_LLM_THINKING", "disabled")
    _complete(
        OpenAICompatibleClient(base_url="https://api.openai.com/v1", thinking="off")
    )
    assert "extra_body" not in calls[0]


def test_private_thinking_switch_drift_fails_closed(calls) -> None:
    client = OpenAICompatibleClient()
    enabled = phase1_generation_binding("v20", client)
    _complete(client)
    assert calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}

    client._send_thinking = False
    with pytest.raises(ValueError, match="integrity"):
        phase1_generation_binding("v20", client)
    with pytest.raises(RuntimeError, match="identity changed"):
        _complete(client)
    client._send_thinking = True
    assert phase1_generation_binding("v20", client) == enabled


def test_invalid_thinking_mode_raises(calls) -> None:
    with pytest.raises(ValueError, match="HYMEM_LLM_THINKING"):
        OpenAICompatibleClient(thinking="maybe")


def test_json_response_format_still_set(calls) -> None:
    _complete(OpenAICompatibleClient(), response_format="json")
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_text_response_format_omits_the_key(calls) -> None:
    _complete(OpenAICompatibleClient(), response_format="text")
    assert "response_format" not in calls[0]


def test_explicit_trusted_token_counter_is_exposed(calls) -> None:
    counter = lambda text: len(text.split())
    client = OpenAICompatibleClient(token_counter=counter)
    assert client.count_tokens is counter


def test_non_callable_token_counter_is_rejected(calls) -> None:
    with pytest.raises(TypeError, match="token_counter"):
        OpenAICompatibleClient(token_counter=3)


def test_missing_api_key_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(openai, "OpenAI", _RecordingOpenAI([], []))
    with pytest.raises(EnvironmentError):
        OpenAICompatibleClient()
