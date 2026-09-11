"""LoCoMo's configurable judge endpoint must match requests and receipts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


pytest.importorskip("requests")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import locomo_adapter as locomo  # noqa: E402
import longmemeval_adapter as lme  # noqa: E402


CUSTOM = "https://judge.example/v1"
DISABLED = {"thinking": {"type": "disabled"}}


def _source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{
        "id": "q1", "question_id": "q1", "category": 1,
        "question": "What?", "answer": "Gold", "ai_answer": "Gold",
        "correct": True,
    }]), encoding="utf-8")
    return source


@pytest.mark.parametrize("body", [None, DISABLED])
def test_custom_rejudge_cli_request_artifact_and_cleanup_agree(
    monkeypatch, tmp_path, body,
):
    source = _source(tmp_path)
    destination = tmp_path / "rejudged.json"
    calls, clients = [], []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "yes"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    build = locomo._build_llm

    def tracked_build(*args, **kwargs):
        client = build(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(lme.http, "post", post)
    monkeypatch.setattr(locomo, "_build_llm", tracked_build)
    argv = [
        "locomo_adapter.py", "--rejudge", str(source), "--out", str(destination),
        "--judge-base-url", CUSTOM + "/", "--judge-api-key", "judge-only-key",
        "--answer-api-key", "reader-only-key",
    ]
    if body is not None:
        argv.extend(["--judge-extra-body", json.dumps(body)])
    monkeypatch.setattr(sys, "argv", argv)
    locomo.main()

    assert len(calls) == len(clients) == 1
    url, request = calls[0]
    assert url == CUSTOM + "/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer judge-only-key"
    assert request["json"]["model"] == locomo._JUDGE_MODEL
    assert request["json"]["temperature"] == 0.0
    assert request["json"]["max_tokens"] == 10
    effective_body = body or {}
    assert clients[0].extra_body == effective_body
    assert request["json"].get("thinking") == effective_body.get("thinking")
    row = json.loads(destination.read_text(encoding="utf-8"))[0]
    assert row["judge_base_url"] == clients[0].base_url == CUSTOM
    assert row["judge_extra_body"] == effective_body
    assert row["_rejudged"] is True and row["correct"] is True
    assert clients[0]._closed is True


@pytest.mark.parametrize("configured", [False, True])
def test_direct_rejudge_uses_client_endpoint_with_legacy_namespace(
    tmp_path, configured,
):
    destination = tmp_path / "rejudged.json"

    class Judge:
        base_url = CUSTOM
        extra_body = {}

        def chat(self, *_args, **_kwargs):
            return "yes"

    args = SimpleNamespace(
        rejudge=str(_source(tmp_path)), out=str(destination), workers=1,
        judge_model=locomo._JUDGE_MODEL,
    )
    if configured:
        args.judge_base_url = CUSTOM + "/"
    locomo._rejudge_file(args, Judge())
    row = json.loads(destination.read_text(encoding="utf-8"))[0]
    assert row["judge_base_url"] == CUSTOM
    assert row["judge_extra_body"] == {}


def test_direct_rejudge_refuses_endpoint_identity_mismatch_before_call(tmp_path):
    destination = tmp_path / "rejudged.json"

    class Judge:
        base_url = CUSTOM
        extra_body = {}

        def chat(self, *_args, **_kwargs):
            pytest.fail("mismatched judge must never be called")

    args = SimpleNamespace(
        rejudge=str(_source(tmp_path)), out=str(destination), workers=1,
        judge_model=locomo._JUDGE_MODEL, judge_base_url=locomo._DEEPSEEK_BASE_URL,
    )
    with pytest.raises(locomo.BenchmarkIntegrityError, match="judge endpoint"):
        locomo._rejudge_file(args, Judge())
    assert not destination.exists()


@pytest.mark.parametrize("url", [
    "http://judge.example/v1", "https://credential@judge.example/v1",
    "https://judge.example/v1?api_key=credential",
])
def test_judge_cli_rejects_unsafe_endpoint_before_client_creation(monkeypatch, url):
    monkeypatch.setattr(
        locomo, "_build_llm",
        lambda *_a, **_k: pytest.fail("unsafe endpoint reached client creation"),
    )
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--rejudge", "unused.json", "--judge-base-url", url,
    ])
    with pytest.raises(SystemExit) as exc:
        locomo.main()
    assert exc.value.code == 2


def test_custom_judge_does_not_inherit_reader_or_provider_key(monkeypatch):
    monkeypatch.delenv("HYMEM_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-provider-only")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-provider-only")
    monkeypatch.setattr(
        locomo, "_rejudge_file",
        lambda *_a, **_k: pytest.fail("credential-less judge was constructed"),
    )
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--rejudge", "unused.json",
        "--judge-base-url", CUSTOM, "--answer-api-key", "reader-only-key",
    ])
    with pytest.raises(locomo.BenchmarkIntegrityError, match="API key"):
        locomo.main()


def test_normal_bare_row_identity_records_actual_custom_judge():
    args = SimpleNamespace(
        answer_model=locomo._ANSWER_MODEL, answer_base_url=locomo._DEEPSEEK_BASE_URL,
        judge_model=locomo._JUDGE_MODEL, judge_base_url=CUSTOM,
        hymem_model=locomo._HYMEM_MODEL, hymem_base_url=locomo._DEEPSEEK_BASE_URL,
    )
    judge = SimpleNamespace(base_url=CUSTOM, extra_body=DISABLED)
    identity = locomo.model_identity_fields(args, None, judge, None)
    assert identity["judge_base_url"] == CUSTOM
    assert identity["judge_extra_body"] == DISABLED
    identity["judge_extra_body"]["thinking"]["type"] = "mutated"
    assert judge.extra_body == DISABLED


def test_normal_strict_identity_binds_custom_judge_endpoint_and_body(
    monkeypatch, tmp_path,
):
    captured = {}

    def fake_freeze(_path, **kwargs):
        captured.update(kwargs)
        return {"dev_ids": [], "holdout_ids": kwargs["ids"]}

    monkeypatch.setattr(locomo, "freeze_calibration", fake_freeze)
    monkeypatch.setattr(
        locomo, "_build_llm",
        lambda *_a, **_k: pytest.fail("calibration identity must be pure"),
    )
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--sample", "0", "--results-dir", str(tmp_path),
        "--freeze-calibration", str(tmp_path / "receipt.json"),
        "--judge-base-url", CUSTOM,
    ])
    locomo.main()
    assert captured["models"]["judge"]["base_url"] == CUSTOM
    assert captured["models"]["judge"]["provider"] == "openai-compatible"
    assert captured["models"]["judge"]["extra_body"] == {}


def test_ordinary_cli_constructs_custom_judge_and_closes_on_abort(
    monkeypatch, tmp_path,
):
    clients = []
    build = locomo._build_llm

    def tracked_build(*args, **kwargs):
        client = build(*args, **kwargs)
        clients.append(client)
        return client

    class StopAfterConstruction(BaseException):
        pass

    def evaluate(_conv, args, answer, judge, **_kwargs):
        assert answer.api_key == "reader-only-key"
        assert judge.api_key == "judge-only-key"
        assert judge.base_url == args.judge_base_url == CUSTOM
        assert judge.extra_body == args.judge_extra_body_obj == {}
        _config, models = locomo._strict_identity(args)
        assert models["judge"]["base_url"] == judge.base_url
        assert models["judge"]["extra_body"] == judge.extra_body
        assert locomo.model_identity_fields(
            args, answer, judge, None
        )["judge_base_url"] == judge.base_url
        raise StopAfterConstruction("stop before any store or provider work")

    monkeypatch.setattr(locomo, "_build_llm", tracked_build)
    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(
        lme.http, "post", lambda *_a, **_k: pytest.fail("provider call not allowed"),
    )
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--sample", "1", "--no-dream",
        "--results-dir", str(tmp_path), "--judge-base-url", CUSTOM,
        "--answer-api-key", "reader-only-key", "--judge-api-key", "judge-only-key",
    ])
    with pytest.raises(StopAfterConstruction):
        locomo.main()
    assert len(clients) == 2
    assert all(client._closed for client in clients)
    assert not list(tmp_path.glob("locomo-*strict-*.json"))
