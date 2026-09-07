"""Regression coverage for current benchmark model/request defaults.

Historical artifacts and compatibility tests may still spell the retired
DeepSeek alias.  These tests cover only live constructors, CLI resolution, and
wire bodies so provenance strings can remain truthful without becoming a
runtime default again.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("requests")
_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(_BENCH))

import beam_adapter as beam  # noqa: E402
import locomo_adapter as locomo  # noqa: E402
import longmemeval_adapter as lme  # noqa: E402
import msc_adapter as msc  # noqa: E402


PIN = "deepseek-v4-flash"
DISABLED = {"thinking": {"type": "disabled"}}


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [{
                "message": {"content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }


def test_all_live_benchmark_and_pipeline_defaults_use_the_pin():
    assert lme.PINNED_DEEPSEEK_MODEL == PIN
    assert (lme.ANSWER_MODEL, lme.JUDGE_MODEL) == (PIN, PIN)
    assert (beam.ANSWER_MODEL, beam.JUDGE_MODEL, beam.HYMEM_MODEL) == (
        PIN, PIN, PIN,
    )
    assert (msc._ANSWER_MODEL, msc._JUDGE_MODEL, msc._HYMEM_MODEL) == (
        PIN, PIN, PIN,
    )
    assert (locomo._ANSWER_MODEL, locomo._JUDGE_MODEL, locomo._HYMEM_MODEL) == (
        PIN, PIN, PIN,
    )
    assert inspect.signature(lme.HyMemAdapter).parameters[
        "pipeline_model"
    ].default == PIN
    assert inspect.signature(beam.HyMemAdapter).parameters[
        "pipeline_model"
    ].default == PIN
    assert inspect.signature(beam.HyMemAdapter).parameters[
        "pipeline_thinking"
    ].default == "auto"
    assert inspect.signature(msc.MSCAdapter).parameters["hymem_model"].default == PIN
    assert inspect.signature(msc.MSCAdapter).parameters[
        "hymem_thinking"
    ].default == "auto"


def test_lme_default_raw_client_sends_thinking_disabled(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return _Response()

    monkeypatch.setattr(lme.http, "post", post)
    client = lme.LLMClient(lme.ANSWER_MODEL, "key")
    assert client.extra_body_defaulted is True
    client._call([{"role": "user", "content": "q"}], 0.0, 32)
    assert calls[0][1]["thinking"] == {"type": "disabled"}


def test_lme_custom_endpoint_gets_no_implicit_deepseek_body(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return _Response()

    monkeypatch.setattr(lme.http, "post", post)
    client = lme.LLMClient(PIN, "key", base_url="https://api.openai.com/v1")
    assert client.extra_body == {}
    assert client.extra_body_defaulted is False
    client._call([], 0.0, 16)
    assert "thinking" not in calls[0]


def test_lme_explicit_custom_gateway_body_is_preserved():
    client = lme.LLMClient(
        PIN, "key", base_url="https://gateway.example/v1",
        extra_body=DISABLED,
    )
    assert client.extra_body == DISABLED
    assert client.extra_body_defaulted is False


def test_lme_explicit_empty_deepseek_v4_body_fails_closed():
    with pytest.raises(lme.BenchmarkIntegrityError, match="requires thinking"):
        lme.LLMClient(PIN, "key", extra_body={})


def test_lme_default_rejudge_cli_resolves_effective_body(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        lme, "_rejudge_run",
        lambda args, key: captured.update(args=args, key=key),
    )
    monkeypatch.setattr(sys, "argv", [
        "longmemeval_adapter.py", "--rejudge", "historical.json",
        "--api-key", "test-key",
    ])
    lme._main()
    assert captured["args"].judge_model == PIN
    assert captured["args"].judge_extra_body_obj == DISABLED
    assert captured["args"].extra_body_defaulted == ["judge"]


def test_lme_normal_manifest_records_effective_default_provenance(
    monkeypatch, tmp_path,
):
    rows = [{
        "question_id": "qid-0",
        "question_type": "multi-session",
        "question": "question",
        "answer": "answer",
        "question_date": "2025-01-03",
        "answer_session_ids": ["s-0"],
        "haystack_session_ids": ["s-0"],
        "haystack_dates": ["2025-01-01"],
        "haystack_sessions": [[{
            "role": "user", "content": "answer", "has_answer": True,
        }]],
    }]
    (tmp_path / "longmemeval_s_cleaned.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    captured = {}

    def fake_freeze(_path, **kwargs):
        captured.update(kwargs)
        return {"dev_ids": [], "holdout_ids": ["qid-0"]}

    monkeypatch.setattr(lme, "freeze_calibration", fake_freeze)
    monkeypatch.setattr(sys, "argv", [
        "longmemeval_adapter.py", "--data-dir", str(tmp_path),
        "--results-dir", str(tmp_path), "--sample", "0", "--no-prereg",
        "--freeze-calibration", str(tmp_path / "receipt.json"),
    ])
    lme._main()
    assert captured["config"]["extra_body_defaulted"] == ["answer", "judge"]
    assert captured["models"]["reader"]["extra_body"] == DISABLED
    assert captured["models"]["judge"]["extra_body"] == DISABLED


@pytest.mark.parametrize("adapter", [msc, locomo])
def test_msc_and_locomo_raw_default_clients_resolve_effective_body(adapter):
    client = adapter._build_llm(PIN, lme.DEEPSEEK_BASE_URL, "key", None)
    assert client.extra_body == DISABLED
    assert client.extra_body_defaulted is True


def test_msc_cli_body_parser_preserves_absent_vs_explicit_empty():
    assert msc.parse_extra_body_arg(None, "answer") is None
    assert msc.parse_extra_body_arg("{}", "answer") == {}
    with pytest.raises(ValueError, match="JSON object"):
        msc.parse_extra_body_arg("null", "answer")


def test_msc_pipeline_default_overrides_ambient_thinking_off(monkeypatch, tmp_path):
    pytest.importorskip("openai")
    monkeypatch.setenv("HYMEM_LLM_THINKING", "off")
    adapter = msc.MSCAdapter(tmp_path / "msc.sqlite", api_key="test-key").open()
    try:
        assert adapter.pipeline_llm.thinking_mode == "auto"
        assert adapter.pipeline_llm.effective_extra_body == DISABLED
    finally:
        adapter.close()


def test_msc_pipeline_auto_omits_vendor_body_for_unrelated_provider(tmp_path):
    pytest.importorskip("openai")
    adapter = msc.MSCAdapter(
        tmp_path / "msc.sqlite", api_key="test-key",
        hymem_model="gpt-4o-mini",
        hymem_base_url="https://api.openai.com/v1",
    ).open()
    try:
        assert adapter.pipeline_llm.thinking_mode == "auto"
        assert adapter.pipeline_llm.effective_extra_body == {}
    finally:
        adapter.close()


def test_locomo_default_rejudge_path_uses_effective_body(monkeypatch):
    captured = {}

    def fake_rejudge(args, judge_llm, owned_clients=None):
        captured["model"] = args.judge_model
        captured["body"] = judge_llm.extra_body
        captured["key"] = judge_llm.api_key
        assert owned_clients is not None

    monkeypatch.setattr(locomo, "_rejudge_file", fake_rejudge)
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--rejudge", "historical.json",
        "--answer-api-key", "reader-only-key",
        "--judge-api-key", "test-key",
    ])
    locomo.main()
    assert captured == {"model": PIN, "body": DISABLED, "key": "test-key"}


def test_locomo_rejudge_artifact_records_effective_body(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "rejudged.json"
    source.write_text(json.dumps([{
        "id": "q1", "question_id": "q1", "category": 1,
        "question": "What?", "answer": "Gold", "ai_answer": "Gold",
        "correct": True,
    }]), encoding="utf-8")

    class Judge:
        extra_body = DISABLED

        def chat(self, *_args, **_kwargs):
            return "yes"

    args = SimpleNamespace(
        rejudge=str(source), out=str(destination), workers=1, judge_model=PIN,
    )
    locomo._rejudge_file(args, Judge())
    row = json.loads(destination.read_text(encoding="utf-8"))[0]
    assert row["judge_model"] == PIN
    assert row["judge_base_url"] == lme.DEEPSEEK_BASE_URL
    assert row["judge_extra_body"] == DISABLED


def test_bare_list_identity_records_effective_not_raw_body():
    args = SimpleNamespace(
        answer_model=PIN,
        answer_base_url=lme.DEEPSEEK_BASE_URL,
        judge_model=PIN,
        hymem_model=PIN,
        hymem_base_url=lme.DEEPSEEK_BASE_URL,
    )
    answer = msc._build_llm(PIN, lme.DEEPSEEK_BASE_URL, "key", None)
    judge = locomo._build_llm(PIN, lme.DEEPSEEK_BASE_URL, "key", None)
    pipeline = SimpleNamespace(
        thinking_mode="auto", effective_extra_body=DISABLED,
    )
    identity = msc.model_identity_fields(args, answer, judge, pipeline)
    assert identity["answer_extra_body"] == DISABLED
    assert identity["judge_extra_body"] == DISABLED
    assert identity["hymem_extra_body"] == DISABLED
    # The receipt owns copies, not the mutable client dictionaries.
    identity["answer_extra_body"]["thinking"]["type"] = "changed"
    assert answer.extra_body == DISABLED


def test_beam_default_paths_disable_pipeline_and_raw_client_thinking():
    body, defaulted = beam.apply_thinking_default(
        "answer", beam.ANSWER_MODEL, "deepseek", True, {}
    )
    assert defaulted is True and body == DISABLED
    adapter = beam.HyMemAdapter(Path("/tmp/identity-only.sqlite"))
    assert adapter.pipeline_model == PIN
    assert adapter.pipeline_thinking == "auto"


def test_live_source_has_no_deprecated_alias_default():
    for source in (
        _BENCH / "longmemeval_adapter.py",
        _BENCH / "beam_adapter.py",
        _BENCH / "msc_adapter.py",
        _BENCH / "locomo_adapter.py",
    ):
        for line in source.read_text(encoding="utf-8").splitlines():
            if "default=" in line or "pipeline_model:" in line:
                assert "deepseek-chat" not in line, f"live default in {source}: {line}"
