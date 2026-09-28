"""Explicit historical LME admission cannot authorize current execution."""

from copy import deepcopy
import json
import sqlite3
import sys

import pytest

from benchmarks import lme_protocol as protocol, lme_registry as registry
from benchmarks.strictness import BenchmarkIntegrityError, content_hash
from hymem.contrib import model_policy, openai_client
from hymem.extraction.producer import producer_binding_from_typed_declaration
from tests.archive_evidence_fixtures import bind_checkpoint
from tests.test_lme_protocol_hardening import (
    _current_exact_aggregation_status,
    _indexing_failure_artifact,
    _refresh_manifest,
    make_artifact,
)


def _drift(monkeypatch, kind):
    if kind == "policy":
        monkeypatch.setattr(
            model_policy, "DEPRECATED_DEEPSEEK_ALIASES",
            model_policy.DEPRECATED_DEEPSEEK_ALIASES | {"pipeline"},
        )
    elif kind == "runtime":
        monkeypatch.setattr(
            openai_client, "OPENAI_TRANSPORT_RUNTIME_VERSIONS",
            {"python": "different-from-fixture-runtime"},
        )
    elif kind == "implementation":
        monkeypatch.setattr(
            openai_client, "OPENAI_LLM_IMPLEMENTATION_SHA256", "sha256:" + "f" * 64,
        )
    else:
        raise AssertionError(kind)


def _reseal_producer(artifact):
    producer = artifact["models"]["memory_pipeline"]["aggregation_producer"]
    producer["identity_sha256"] = content_hash({
        key: value for key, value in producer["declaration"].items()
        if key != "schema"
    })
    _refresh_manifest(artifact)


@pytest.mark.parametrize("kind", ["policy", "runtime", "implementation"])
@pytest.mark.parametrize("indexed", [False, True])
def test_explicit_historical_reader_survives_drift_without_changing_evidence(
    monkeypatch, kind, indexed,
):
    artifact = _indexing_failure_artifact() if indexed else make_artifact()
    strict = protocol.validate_strict_artifact(artifact)
    before = deepcopy(artifact)
    _drift(monkeypatch, kind)
    with pytest.raises(BenchmarkIntegrityError, match="aggregation identity"):
        protocol.validate_strict_artifact(artifact)
    historical = protocol.validate_archived_artifact(artifact)
    assert historical == {
        **strict, "validation_assurance": "historical_commitment_only",
        "live_execution_eligible": False,
    }
    assert artifact == before


@pytest.mark.parametrize("kind", ["policy", "runtime", "implementation"])
def test_registry_historical_scope_is_explicit_persisted_and_still_listed(
    monkeypatch, tmp_path, capsys, kind,
):
    artifact = _indexing_failure_artifact()
    path = tmp_path / "longmemeval-v2-hymem-20260904T120000Z-seed0-strict.json"
    path.write_text(json.dumps(artifact))
    original = path.read_bytes()
    monkeypatch.setattr(registry, "DB", tmp_path / "registry.sqlite")
    _drift(monkeypatch, kind)
    conn = registry.connect()
    assert registry.ingest_file(conn, path).startswith("error: strict")
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    assert registry.ingest_file(conn, path, validation_scope="historical") == "inserted"
    conn.commit()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs").fetchone()
    assert row["strict_validated"] == 0
    assert row["archive_validated"] == 1
    assert row["validation_assurance"] == "historical_commitment_only"
    assert row["live_execution_eligible"] == 0
    assert row["pipeline_model"] == "pipeline"
    extras = json.loads(row["extras"])
    assert extras["strict_validation"] is None
    assert extras["archive_validation"]["live_execution_eligible"] is False
    assert extras["archive_validation"]["validation_assurance"] == "historical_commitment_only"
    assert extras["models"] == artifact["models"]
    assert registry.ingest_file(conn, path, validation_scope="historical") == "skipped"
    registry.cmd_list()
    assert path.name in capsys.readouterr().out
    assert path.read_bytes() == original
    conn.close()


def test_current_registry_default_and_cli_historical_selection(monkeypatch, tmp_path, capsys):
    artifact = make_artifact()
    path = tmp_path / "longmemeval-v2-hymem-20260904T120000Z-seed0-strict.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(registry, "DB", tmp_path / "current.sqlite")
    conn = registry.connect()
    assert registry.ingest_file(conn, path) == "inserted"
    assert conn.execute(
        "SELECT strict_validated, archive_validated, validation_assurance, "
        "live_execution_eligible FROM runs"
    ).fetchone() == (1, 0, "current_producer_reconstructed", None)
    conn.commit()
    registry.cmd_list()
    assert path.name in capsys.readouterr().out
    conn.close()
    monkeypatch.setattr(registry, "DB", tmp_path / "historical.sqlite")
    _drift(monkeypatch, "runtime")
    monkeypatch.setattr(sys, "argv", [
        "lme_registry.py", "ingest", str(path), "--historical-commitments",
    ])
    registry.main()
    with sqlite3.connect(registry.DB) as con:
        assert con.execute("SELECT archive_validated FROM runs").fetchone() == (1,)


def test_archive_validation_never_calls_current_producer_builder(monkeypatch):
    artifact = make_artifact()

    def forbidden(**_kwargs):
        raise AssertionError("archive reader reached current producer builder")

    monkeypatch.setattr(openai_client, "openai_compatible_producer_declaration", forbidden)
    protocol.validate_archived_artifact(artifact)
    with pytest.raises(AssertionError, match="current producer"):
        protocol.validate_strict_artifact(artifact)


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash"])
def test_recorded_retired_model_is_not_rewritten_or_authorized(monkeypatch, model):
    artifact = make_artifact()
    pipeline = artifact["models"]["memory_pipeline"]
    artifact["config"]["hymem_model"] = pipeline["model"] = model
    with monkeypatch.context() as before_retirement:
        before_retirement.setattr(model_policy, "DEPRECATED_DEEPSEEK_ALIASES", frozenset())
        pipeline["aggregation_producer"] = producer_binding_from_typed_declaration(
            openai_client.openai_compatible_producer_declaration(
                model=model, endpoint="https://pipeline.example/v1",
                thinking_mode=pipeline["thinking_mode"],
                effective_extra_body=pipeline["effective_extra_body"],
                transport_package_version=pipeline["transport_package_version"],
                request_timeout_seconds=pipeline["request_timeout_seconds"],
                deployment_revision_sha256=pipeline["deployment_revision_sha256"],
                deployment_tenant_sha256=pipeline["deployment_tenant_sha256"],
            ), declaration_hook="aggregation_producer_declaration",
        )
        _refresh_manifest(artifact)
        protocol.validate_strict_artifact(artifact)
    original = json.dumps(artifact, sort_keys=True)
    monkeypatch.setattr(
        model_policy, "DEPRECATED_DEEPSEEK_ALIASES",
        model_policy.DEPRECATED_DEEPSEEK_ALIASES | {model},
    )
    assert protocol.validate_archived_artifact(artifact)["live_execution_eligible"] is False
    with pytest.raises(BenchmarkIntegrityError, match="aggregation identity"):
        protocol.validate_strict_artifact(artifact)
    with pytest.raises(model_policy.DeprecatedModelAliasError):
        model_policy.require_active_model(model)
    assert json.dumps(artifact, sort_keys=True) == original


@pytest.mark.parametrize("kind", ["policy", "runtime", "implementation"])
def test_historical_acceptance_does_not_enable_official_export(monkeypatch, tmp_path, kind):
    artifact = make_artifact()
    _drift(monkeypatch, kind)
    protocol.validate_archived_artifact(artifact)
    destination = tmp_path / "predictions.jsonl"
    with pytest.raises(BenchmarkIntegrityError, match="aggregation identity"):
        protocol.export_official_predictions(artifact, destination)
    assert not destination.exists()


@pytest.mark.parametrize("validator", [
    protocol.validate_strict_artifact, protocol.validate_archived_artifact,
])
@pytest.mark.parametrize(("field", "value"), [
    ("client_id", "other.Client"),
    ("implementation", "not-a-digest"),
    ("model", "other-model"),
    ("endpoint_origin", "https://other.example"),
    ("endpoint_sha256", "sha256:" + "1" * 64),
    ("effective_request", {"sha256": "invalid"}),
    ("retry_policy", {"sha256": "invalid"}),
])
def test_resealed_producer_tampering_is_rejected_in_both_modes(validator, field, value):
    artifact = make_artifact()
    artifact["models"]["memory_pipeline"]["aggregation_producer"]["declaration"][field] = value
    _reseal_producer(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="aggregation identity"):
        validator(artifact)


@pytest.mark.parametrize("validator", [
    protocol.validate_strict_artifact, protocol.validate_archived_artifact,
])
@pytest.mark.parametrize(("field", "value"), [
    ("transport_package_version", ""),
    ("transport_package_version", []),
    ("request_timeout_seconds", True),
    ("request_timeout_seconds", 0),
    ("request_timeout_seconds", float("inf")),
    ("deployment_revision_sha256", None),
    ("deployment_tenant_sha256", "invalid"),
    ("thinking_mode", "invented"),
    ("effective_extra_body", {"model": "override"}),
])
def test_recorded_pipeline_shapes_remain_required(validator, field, value):
    artifact = make_artifact()
    artifact["models"]["memory_pipeline"][field] = value
    if field == "request_timeout_seconds" and value == float("inf"):
        # Non-finite JSON is invalid before any identity reconstruction.
        with pytest.raises(ValueError):
            _refresh_manifest(artifact)
        return
    _refresh_manifest(artifact)
    with pytest.raises((BenchmarkIntegrityError, ValueError)):
        validator(artifact)


@pytest.mark.parametrize("validator", [
    protocol.validate_strict_artifact, protocol.validate_archived_artifact,
])
@pytest.mark.parametrize("kind", ["score", "count", "usage", "checkpoint", "result", "producer"])
def test_historical_scope_keeps_non_model_integrity_checks(validator, kind):
    artifact = make_artifact()
    if kind == "score":
        artifact["scores"]["OVERALL"]["accuracy"] = 99.0
    elif kind == "count":
        artifact["execution"]["counts"]["expected"] += 1
        bind_checkpoint(artifact)
    elif kind == "usage":
        artifact["execution"]["segments"][0]["reader_usage"]["total_tokens"] += 1
        bind_checkpoint(artifact)
    elif kind == "checkpoint":
        artifact["execution"].pop("checkpoint")
    elif kind == "result":
        artifact["result_digest"] = "sha256:" + "e" * 64
    else:
        artifact["models"]["memory_pipeline"]["aggregation_producer"]["identity_sha256"] = "sha256:" + "e" * 64
        _refresh_manifest(artifact)
    with pytest.raises((BenchmarkIntegrityError, ValueError)):
        validator(artifact)


@pytest.mark.parametrize("archive_only", [False, True])
def test_aggregation_certificate_must_match_same_recorded_producer(cfg, archive_only):
    artifact = make_artifact()
    pipeline = artifact["models"]["memory_pipeline"]
    matching = protocol._canonical_final_indexing_status(_current_exact_aggregation_status(cfg))
    protocol._validate_aggregation_pipeline_binding(matching, pipeline, archive_only=archive_only)
    different = protocol._canonical_final_indexing_status(
        _current_exact_aggregation_status(cfg, model="different-pipeline")
    )
    with pytest.raises(BenchmarkIntegrityError, match="differs from memory pipeline"):
        protocol._validate_aggregation_pipeline_binding(different, pipeline, archive_only=archive_only)


@pytest.mark.parametrize("different", [False, True])
def test_full_indexed_archive_preserves_aggregation_certificate_crosslink(
    cfg, monkeypatch, tmp_path, different,
):
    artifact = _indexing_failure_artifact()
    artifact["config"]["aggregation_nodes"] = True
    artifact["config"]["effective_hymem_config"]["aggregation_nodes_enabled"] = True
    status = protocol._canonical_final_indexing_status(
        _current_exact_aggregation_status(
            cfg, model="other-pipeline" if different else "pipeline",
        )
    )
    summary = artifact["per_question"][0]["indexing"]
    for field in ("aggregation_generation", "aggregation_material"):
        summary["final_status"][field] = status[field]
    segment = artifact["execution"]["segments"][0]
    record = {"question_id": "qid", "summary": deepcopy(summary)}
    segment["indexing_runs"] = [record]
    segment["latest_indexing"] = deepcopy(record)
    artifact["result_digest"] = content_hash(artifact["per_question"])
    _refresh_manifest(artifact)
    if not different:
        protocol.validate_strict_artifact(artifact)
    _drift(monkeypatch, "runtime")
    path = tmp_path / "longmemeval-v2-hymem-20260904T120000Z-seed0-strict.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(registry, "DB", tmp_path / "registry.sqlite")
    with registry.connect() as conn:
        result = registry.ingest_file(conn, path, validation_scope="historical")
        if different:
            assert result.startswith("error: strict")
            assert "differs from memory pipeline" in result
            assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
        else:
            assert result == "inserted"


def test_opaque_historical_hashes_are_not_misrepresented_as_reconstructed():
    artifact = make_artifact()
    declaration = artifact["models"]["memory_pipeline"]["aggregation_producer"]["declaration"]
    declaration["implementation"] = "sha256:" + "7" * 64
    for field in ("effective_request", "retry_policy"):
        declaration[field]["sha256"] = "sha256:" + "8" * 64
    _reseal_producer(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="aggregation identity"):
        protocol.validate_strict_artifact(artifact)
    result = protocol.validate_archived_artifact(artifact)
    assert result["validation_assurance"] == "historical_commitment_only"
    assert result["live_execution_eligible"] is False
