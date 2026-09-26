"""Archive admission checks recorded evidence, not today's model availability."""

from __future__ import annotations

import copy
import json
import sqlite3
import sys

import pytest

from benchmarks import locomo_adapter, locomo_registry, msc_adapter, msc_registry
from benchmarks.strictness import content_hash
from hymem.contrib import model_policy
from tests.archive_evidence_fixtures import bind_checkpoint, scoped_indexing
from tests.test_locomo_checkpoint_resume import _conversation, _row, _runtime
from tests.test_msc_checkpoint_resume import (
    _UsageClient,
    _example,
    _passed_canary,
    _scored_success,
)


def _make_archive(monkeypatch, tmp_path, benchmark, model):
    """Publish synthetic scored evidence while the model is fixture-active.

    No provider, credential resolver, dataset loader, or memory store is used.
    The real adapter writes the manifest, canary bindings, checkpoint, and
    immutable archive. Retirement policy is restored before the reader runs.
    """
    adapter = msc_adapter if benchmark == "msc" else locomo_adapter
    with monkeypatch.context() as fixture:
        fixture.setattr(model_policy, "DEPRECATED_DEEPSEEK_ALIASES", frozenset())
        fixture.setattr(adapter, f"{benchmark}_code_hash", lambda: "sha256:" + "a" * 64)
        fixture.setattr(adapter, "_build_llm", lambda *_a, **_k: _UsageClient())
        canary = _passed_canary()
        canary["client"]["model"] = model
        fixture.setattr(
            adapter, "run_configured_extraction_canary",
            lambda **_k: copy.deepcopy(canary),
        )
        if benchmark == "msc":
            fixture.setattr(adapter, "load_msc_data", lambda *_a, **_k: [_example("q1")])
            fixture.setattr(adapter, "run_recall", _scored_success)
        else:
            fixture.setattr(
                adapter, "load_locomo_data",
                lambda *_a, **_k: [_conversation("conv-one", "q1")],
            )

            def evaluate(conversation, args, answer, judge, **kwargs):
                for client in (answer, judge):
                    client.call_count += 1
                    client.request_attempts += 1
                    client.successful_responses += 1
                row = _row(conversation, conversation["qa"][0])
                row.update(gold_in_context=False, gold_in_pool=False)
                runtime = _runtime(conversation)
                runtime["indexing"] = scoped_indexing(runtime["scope_id"], args)
                kwargs["on_checkpoint"](row, runtime)
                return [row]

            fixture.setattr(adapter, "evaluate_conversation", evaluate)
        fixture.setattr(sys, "argv", [
            f"{benchmark}_adapter.py",
            "--answer-model", model, "--judge-model", model,
            "--hymem-model", model,
            "--checkpoint", str(tmp_path / "run.checkpoint.json"),
            "--results-dir", str(tmp_path / "results"),
        ])
        adapter.main()
        archive = next((tmp_path / "results").glob(f"{benchmark}-*strict-*.json"))
        # Validate under fixture-time policy to establish that the fixture is
        # structurally admissible before simulating later model retirement.
        _read_archive(benchmark, archive)
    return archive


def _read_archive(benchmark, path):
    if benchmark == "msc":
        return msc_registry.load_msc_artifact(path)
    return locomo_registry._locomo_row(json.loads(path.read_text()), path)


def _write_rebound(path, artifact):
    """Recompute hashes so negative cases exercise semantic validation."""
    manifest = artifact["manifest"]
    manifest["models"] = copy.deepcopy(artifact["models"])
    manifest["model_hash"] = content_hash(artifact["models"])
    manifest["run_id"] = content_hash({
        key: value for key, value in manifest.items() if key != "run_id"
    })
    for segment in artifact["execution"]["segments"]:
        segment["model_identities"] = copy.deepcopy(artifact["models"])
    artifact["result_digest"] = content_hash(artifact["per_question"])
    bind_checkpoint(artifact)
    path.write_text(json.dumps(artifact))


@pytest.mark.parametrize("benchmark", ["msc", "locomo"])
@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash"])
def test_archive_read_and_discovery_preserve_historical_model_bytes(
    monkeypatch, tmp_path, benchmark, model,
):
    path = _make_archive(monkeypatch, tmp_path, benchmark, model)
    original = path.read_bytes()
    _read_archive(benchmark, path)
    if benchmark == "msc":
        artifacts = msc_registry.scan_msc_archives(path.parent)
        assert len(artifacts) == 1
        assert artifacts[0]["models"]["memory_pipeline"]["model"] == model
        assert msc_registry.load_msc_artifact(path.parent / "msc-latest.json")["models"] == artifacts[0]["models"]
    else:
        database = tmp_path / "runs.sqlite"
        locomo_registry._ingest([path], db_path=database)
        with sqlite3.connect(database) as conn:
            row = conn.execute("SELECT answer_model, judge_model FROM runs").fetchone()
        assert row == (model, model)
    assert path.read_bytes() == original
    assert all(
        json.loads(original)["models"][role]["model"] == model
        for role in ("reader", "judge", "memory_pipeline")
    )


@pytest.mark.parametrize("benchmark", ["msc", "locomo"])
def test_later_model_retirement_does_not_revoke_archived_results(
    monkeypatch, tmp_path, benchmark,
):
    model = "deepseek-v4-flash"
    path = _make_archive(monkeypatch, tmp_path, benchmark, model)
    original = path.read_bytes()
    monkeypatch.setattr(
        model_policy, "DEPRECATED_DEEPSEEK_ALIASES",
        model_policy.DEPRECATED_DEEPSEEK_ALIASES | {model},
    )
    with pytest.raises(model_policy.DeprecatedModelAliasError):
        model_policy.require_active_model(model)
    _read_archive(benchmark, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("benchmark", ["msc", "locomo"])
@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
@pytest.mark.parametrize("flag", ["--answer-model", "--judge-model", "--hymem-model"])
def test_historical_compatibility_does_not_allow_new_retired_model_execution(
    monkeypatch, tmp_path, benchmark, model, flag,
):
    adapter = msc_adapter if benchmark == "msc" else locomo_adapter
    touched = []

    def forbidden(*_args, **_kwargs):
        touched.append(True)
        raise AssertionError("retired model reached execution setup")

    monkeypatch.setattr(adapter, f"load_{benchmark}_data", forbidden)
    monkeypatch.setattr(adapter, "_build_llm", forbidden)
    monkeypatch.setattr(adapter, "run_configured_extraction_canary", forbidden)
    monkeypatch.setattr(sys, "argv", [
        f"{benchmark}_adapter.py", flag, model,
        "--checkpoint", str(tmp_path / "run.checkpoint.json"),
        "--results-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(SystemExit) as caught:
        adapter.main()
    assert caught.value.code == 2
    assert touched == []
    assert not (tmp_path / "run.checkpoint.json").exists()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("benchmark", ["msc", "locomo"])
@pytest.mark.parametrize(("field", "value", "message"), [
    ("model", "", "model identity is absent"),
    ("model", " deepseek-chat", "model identity is absent"),
    ("model", ["deepseek-chat"], "model identity is absent"),
    ("base_url", "http://public.example/v1", "provider identity is unsafe"),
    ("base_url", "https://user:secret@api.deepseek.com", "provider identity is unsafe"),
    ("provider", "forged", "endpoint identity is inconsistent"),
    ("client_class", "forged.Client", "reader request identity"),
    ("extra_body", {"model": "override"}, "request body"),
    ("max_tokens", 2048, "reader request identity"),
])
def test_retired_model_does_not_bypass_metadata_validation(
    monkeypatch, tmp_path, benchmark, field, value, message,
):
    path = _make_archive(monkeypatch, tmp_path, benchmark, "deepseek-chat")
    artifact = json.loads(path.read_text())
    artifact["models"]["reader"][field] = value
    _write_rebound(path, artifact)
    with pytest.raises(ValueError, match=message):
        _read_archive(benchmark, path)


@pytest.mark.parametrize("benchmark", ["msc", "locomo"])
@pytest.mark.parametrize("tamper", ["model_hash", "canary_client", "canary_claims"])
def test_retired_model_does_not_bypass_hash_or_canary_binding(
    monkeypatch, tmp_path, benchmark, tamper,
):
    path = _make_archive(monkeypatch, tmp_path, benchmark, "deepseek-reasoner")
    artifact = json.loads(path.read_text())
    if tamper == "model_hash":
        artifact["models"]["reader"]["model"] = "different-model"
        path.write_text(json.dumps(artifact))
    else:
        reports = [
            artifact["execution"]["segments"][0]["extraction_canary"],
            artifact["per_question"][0]["extraction_canary"],
        ]
        for report in reports:
            if tamper == "canary_client":
                report["client"]["model"] = "different-model"
            else:
                report["claim_evidence"] = []
        _write_rebound(path, artifact)
    with pytest.raises(ValueError, match="identity|hash|canary"):
        _read_archive(benchmark, path)
