"""Tests for the OpenAI-compatible LLM client's vendor-specific body gating.

None of these hit the network: `openai.OpenAI` is replaced with a recorder that
captures the kwargs handed to `chat.completions.create`.
"""

from __future__ import annotations

from typing import Any

import pytest

from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.extraction.llm import LLMRequest

_LLM_ENV = (
    "HYMEM_LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "HYMEM_LLM_BASE_URL",
    "HYMEM_LLM_MODEL",
    "HYMEM_LLM_THINKING",
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

        class _Client:
            class chat:  # noqa: N801 - mirrors the openai SDK's attribute shape
                completions = _RecordingCompletions(sink)

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


def test_invalid_thinking_mode_raises(calls) -> None:
    with pytest.raises(ValueError, match="HYMEM_LLM_THINKING"):
        OpenAICompatibleClient(thinking="maybe")


def test_json_response_format_still_set(calls) -> None:
    _complete(OpenAICompatibleClient(), response_format="json")
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_text_response_format_omits_the_key(calls) -> None:
    _complete(OpenAICompatibleClient(), response_format="text")
    assert "response_format" not in calls[0]


def test_missing_api_key_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(openai, "OpenAI", _RecordingOpenAI([], []))
    with pytest.raises(EnvironmentError):
        OpenAICompatibleClient()
