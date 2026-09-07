"""Retired model aliases fail before any active LLM or benchmark work."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hymem.contrib.model_policy import (
    DeprecatedModelAliasError,
    deprecated_deepseek_alias,
    require_active_model,
)

_BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

import beam_adapter as beam  # noqa: E402
import fact_probe  # noqa: E402
import locomo_adapter as locomo  # noqa: E402
import longmemeval_adapter as lme  # noqa: E402
import msc_adapter as msc  # noqa: E402
import multihop_miner  # noqa: E402
import rules_compliance  # noqa: E402


@pytest.mark.parametrize(
    ("model", "alias"),
    [
        ("deepseek-chat", "deepseek-chat"),
        (" DEEPSEEK-CHAT ", "deepseek-chat"),
        ("deepseek-reasoner", "deepseek-reasoner"),
        ("DeepSeek: DeepSeek-Reasoner", "deepseek-reasoner"),
        ("openai:deepseek-chat", "deepseek-chat"),
        ("deepseek/deepseek-chat", "deepseek-chat"),
        ("openrouter: DeepSeek / DeepSeek-Chat", "deepseek-chat"),
    ],
)
def test_policy_recognizes_only_exact_normalized_alias_slots(model, alias):
    assert deprecated_deepseek_alias(model) == alias
    with pytest.raises(DeprecatedModelAliasError) as caught:
        require_active_model(model, role="test role")
    message = str(caught.value)
    assert alias in message
    assert "deepseek-v4-flash" in message
    assert "auto" in message and "disabled" in message


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-chat-v4",
        "deepseek-reasoner-v2",
        "my-deepseek-chat-model",
        "org/deepseek-chat",
        "deepseek-v4-flash",
        "openai:gpt-4.1-mini",
        None,
    ],
)
def test_policy_has_no_substring_false_positives(model):
    assert deprecated_deepseek_alias(model) is None
    require_active_model(model)


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
def test_library_constructor_rejects_explicit_alias_before_secret_resolution(
    monkeypatch, model,
):
    import hymem.contrib.openai_client as client_module

    resolved = []
    monkeypatch.setattr(
        client_module,
        "resolve_llm_api_key",
        lambda *_args, **_kwargs: resolved.append(True),
    )
    with pytest.raises(DeprecatedModelAliasError, match="deepseek-v4-flash"):
        client_module.OpenAICompatibleClient(model=model)
    assert resolved == []


def test_library_constructor_rejects_environment_fallback_before_secret_resolution(
    monkeypatch,
):
    import hymem.contrib.openai_client as client_module

    resolved = []
    monkeypatch.setenv("HYMEM_LLM_MODEL", "  DeepSeek-Chat  ")
    monkeypatch.setattr(
        client_module,
        "resolve_llm_api_key",
        lambda *_args, **_kwargs: resolved.append(True),
    )
    with pytest.raises(DeprecatedModelAliasError, match="deepseek-chat"):
        client_module.OpenAICompatibleClient()
    assert resolved == []


def test_library_constructor_uses_one_trimmed_active_model_identity():
    from hymem.contrib.openai_client import OpenAICompatibleClient

    client = OpenAICompatibleClient(
        api_key="fixture-key",
        model="  GPT-4o-Mini  ",
        deployment_revision="fixture-revision-v1",
        deployment_tenant="fixture-tenant-v1",
    )
    try:
        assert client.model == "GPT-4o-Mini"
        assert client.phase1_producer_declaration().model == "GPT-4o-Mini"
    finally:
        client.close()


def test_server_bootstrap_rejects_deprecated_env_before_key_or_store(monkeypatch):
    from hymem.bootstrap import build_from_env

    monkeypatch.setenv("HYMEM_LLM_MODEL", "deepseek-reasoner")
    monkeypatch.delenv("HYMEM_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(DeprecatedModelAliasError, match="deepseek-v4-flash"):
        build_from_env()


def test_doctor_reports_actionable_deprecated_env_error(monkeypatch):
    from hymem.bootstrap import resolve_env
    from hymem.doctor import FAIL, _check_llm
    import hymem.contrib.openai_client as client_module

    constructed = []
    monkeypatch.setattr(
        client_module,
        "OpenAICompatibleClient",
        lambda *_args, **_kwargs: constructed.append(True),
    )
    monkeypatch.setenv("HYMEM_LLM_MODEL", "deepseek-chat")
    cfg = resolve_env()
    result = _check_llm(cfg)
    assert result.status == FAIL
    assert "deepseek-chat" in result.detail
    assert "deepseek-v4-flash" in result.detail
    assert constructed == []


@pytest.mark.parametrize("client_module", [beam, lme])
def test_raw_benchmark_constructor_rejects_before_endpoint_or_http(
    monkeypatch, client_module,
):
    resolved = []
    posted = []
    monkeypatch.setattr(
        client_module,
        "resolve_llm_api_key",
        lambda *_args, **_kwargs: resolved.append(True),
    )
    monkeypatch.setattr(
        client_module.http,
        "post",
        lambda *_args, **_kwargs: posted.append(True),
    )
    with pytest.raises(DeprecatedModelAliasError, match="deepseek-chat"):
        client_module.LLMClient("deepseek-chat", "credential")
    assert resolved == []
    assert posted == []


@pytest.mark.parametrize(
    ("builder", "args"),
    [
        (
            multihop_miner._build_dream_llm,
            ("deepseek-chat", "https://api.deepseek.com", "credential"),
        ),
        (
            rules_compliance._build_llm,
            ("deepseek-reasoner", "https://api.deepseek.com", "credential"),
        ),
    ],
)
def test_former_warning_only_helpers_now_fail_before_sdk_construction(
    monkeypatch, builder, args,
):
    import hymem.contrib.openai_client as client_module

    constructed = []
    monkeypatch.setattr(
        client_module,
        "OpenAICompatibleClient",
        lambda *_args, **_kwargs: constructed.append(True),
    )
    kwargs = {"stub_reply": "stub"} if builder is rules_compliance._build_llm else {}
    with pytest.raises(DeprecatedModelAliasError, match="deepseek-v4-flash"):
        builder(*args, **kwargs)
    assert constructed == []


def test_beam_cli_rejects_before_dataset_or_credentials(monkeypatch):
    touched = []
    monkeypatch.setattr(beam, "resolve_dataset_revisions", lambda *_a, **_k: touched.append("dataset"))
    monkeypatch.setattr(beam, "resolve_llm_api_key", lambda *_a, **_k: touched.append("secret"))
    monkeypatch.setattr(sys, "argv", [
        "beam_adapter.py", "--answer-model", "deepseek:deepseek-chat",
        "--no-prereg",
    ])
    with pytest.raises(SystemExit) as caught:
        beam.main()
    assert caught.value.code == 2
    assert touched == []


def test_lme_cli_rejects_before_dataset_or_credentials(monkeypatch):
    touched = []
    monkeypatch.setattr(lme, "load_longmemeval_data", lambda *_a, **_k: touched.append("dataset"))
    monkeypatch.setattr(lme, "resolve_llm_api_key", lambda *_a, **_k: touched.append("secret"))
    monkeypatch.setattr(sys, "argv", [
        "longmemeval_adapter.py", "--judge-model", "DEEPSEEK-REASONER",
        "--no-prereg",
    ])
    with pytest.raises(SystemExit) as caught:
        lme.main()
    assert caught.value.code == 2
    assert touched == []


def test_paid_fact_probe_rejects_before_source_or_dataset_load(monkeypatch, capsys):
    touched = []
    monkeypatch.setattr(
        fact_probe,
        "load_longmemeval_data",
        lambda *_args, **_kwargs: touched.append("dataset"),
    )
    monkeypatch.setattr(sys, "argv", [
        "fact_probe.py", "--source", "/does/not/exist.json",
        "--dataset", "/also/missing.json", "--model", "deepseek-chat",
    ])
    with pytest.raises(SystemExit) as caught:
        fact_probe.main()
    assert caught.value.code == 2
    assert touched == []
    assert "deepseek-v4-flash" in capsys.readouterr().err


def test_fact_probe_cost_only_keeps_historical_model_label_readable(
    monkeypatch, tmp_path,
):
    source = tmp_path / "historical.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        fact_probe,
        "select_probe_sets",
        lambda *_args, **_kwargs: (["qid"], [], {}),
    )
    monkeypatch.setattr(
        fact_probe,
        "load_longmemeval_data",
        lambda *_args, **_kwargs: [{
            "question_id": "qid", "haystack_sessions": [],
        }],
    )
    monkeypatch.setattr(sys, "argv", [
        "fact_probe.py", "--source", str(source),
        "--dataset", str(tmp_path / "historical-dataset.json"),
        "--model", "deepseek-chat", "--cost",
    ])
    # --cost is explicitly provider-free. The historical model label is only
    # printed as provenance and must not be treated as an execution choice.
    assert fact_probe.main() is None


@pytest.mark.parametrize(
    ("adapter", "loader", "argv"),
    [
        (
            msc,
            "load_msc_data",
            ["msc_adapter.py", "--hymem-model", "deepseek-chat"],
        ),
        (
            locomo,
            "load_locomo_data",
            ["locomo_adapter.py", "--hymem-model", "deepseek-reasoner"],
        ),
    ],
)
def test_msc_and_locomo_reject_before_dataset_loading(
    monkeypatch, adapter, loader, argv,
):
    touched = []
    monkeypatch.setattr(adapter, loader, lambda *_a, **_k: touched.append("dataset"))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        adapter.main()
    assert caught.value.code == 2
    assert touched == []


@pytest.mark.parametrize(
    ("adapter", "loader", "argv"),
    [
        (
            msc,
            "load_msc_data",
            ["msc_adapter.py", "--sim", "--answer-model", "deepseek-chat"],
        ),
        (
            locomo,
            "load_locomo_data",
            ["locomo_adapter.py", "--sim", "--answer-model", "deepseek-chat"],
        ),
    ],
)
def test_zero_call_simulation_does_not_apply_active_model_policy(
    monkeypatch, adapter, loader, argv,
):
    touched = []

    def empty_loader(*_args, **_kwargs):
        touched.append("dataset")
        return []

    monkeypatch.setattr(adapter, loader, empty_loader)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        adapter.main()
    assert caught.value.code == 1
    assert touched == ["dataset"]


def test_historical_provider_parsing_remains_a_pure_transformation():
    # Reading/re-indexing old artifacts needs this parser; only construction of
    # an active client is prohibited.
    assert beam.parse_provider_spec("deepseek-chat") == (
        "deepseek", "deepseek-chat", beam.DEEPSEEK_BASE_URL,
    )
    assert lme.resolve_model_extra_body(
        "deepseek-chat", "https://gateway.example/v1", None
    ) == ({}, False)
