"""Current response accounting and genuine cross-version archived evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import extraction_canary as canary, lme_protocol as protocol, lme_registry
from benchmarks.extraction_canary_archive import validate_archived_canary_config, validate_archived_canary_report
from benchmarks.strictness import AtomicCheckpoint, BenchmarkIntegrityError
from hymem.extraction.contract import extraction_contract_binding
from tests.test_benchmark_extraction_canary import _passed_report
from tests.test_lme_protocol_hardening import (
    _indexing_failure_artifact, _refresh_manifest,
    generated_indexing_failure_archive,
)


FIXTURE = Path(__file__).parent / "data/synthetic_r3_v17_indexing_failure.json"
PIN = "d9fd6ededba452c29ef039f785c3fb377c6400e18ea3023ebf4f8a328779e6aa"


def old_artifact():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PIN
    return json.loads(raw)


def old_v18_artifact():
    artifact = old_artifact()
    policy = artifact["config"]["extraction_canary"]
    report = artifact["execution"]["segments"][0]["extraction_canary"]
    for value in (policy, report):
        value["version"] = "hymem-phase1-extraction-canary-v18"
        value["normal_execution_path"]["provider_output_truncations"] = 0
    report["execution_path"]["provider_output_truncations"] = 0
    _refresh_manifest(artifact)
    return artifact


def archive_report(artifact):
    return validate_archived_canary_report(
        artifact["execution"]["segments"][0]["extraction_canary"],
        policy=artifact["config"]["extraction_canary"],
        effective_config=artifact["config"]["effective_hymem_config"],
        expected_mode="required", expected_client=artifact["models"]["memory_pipeline"],
        require_client_closed=True,
    )


def test_genuine_r3_v17_survives_contract_change_only_as_unchanged_history(tmp_path):
    artifact = old_artifact()
    before = deepcopy(artifact)
    old_binding = artifact["config"]["extraction_canary"]["extraction_contract"]
    assert old_binding != extraction_contract_binding(old_binding["prompt_version"])
    result = protocol.validate_archived_artifact(artifact)
    assert result["validation_assurance"] == "historical_commitment_only"
    assert result["live_execution_eligible"] is False
    assert archive_report(artifact)["version"] == "hymem-phase1-extraction-canary-v17"
    with pytest.raises(BenchmarkIntegrityError):
        protocol.validate_strict_artifact(artifact)
    with pytest.raises(BenchmarkIntegrityError):
        canary.validate_extraction_canary_report(
            artifact["execution"]["segments"][0]["extraction_canary"], expected_mode="required")
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(BenchmarkIntegrityError):
        protocol.export_official_predictions(artifact, output)
    assert not output.exists() and artifact == before
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == PIN


def test_v18_history_keeps_v10_split_pin_and_rejects_current_live_admission():
    artifact = old_v18_artifact()
    before = deepcopy(artifact)
    policy = artifact["config"]["extraction_canary"]
    assert policy["source_split_policy_version"] == "hymem-source-semantic-split-v10"
    assert protocol.validate_archived_artifact(artifact)["validation_assurance"] == (
        "historical_commitment_only"
    )
    with pytest.raises(BenchmarkIntegrityError):
        protocol.validate_strict_artifact(artifact)
    assert artifact == before


def test_current_writer_archive_is_admitted_under_v20_repair_policy(
    generated_indexing_failure_archive,
):
    artifact = generated_indexing_failure_archive
    policy = artifact["config"]["extraction_canary"]
    assert policy["version"] == "hymem-phase1-extraction-canary-v20"
    assert policy["contract_repair_policy_version"] == "hymem-source-only-contract-repair-v1"
    assert policy["source_split_policy_version"] == "hymem-source-semantic-split-v11"
    assert protocol.validate_strict_artifact(artifact)["counts"]["failed"] == 1
    assert protocol.validate_archived_artifact(artifact)["counts"]["failed"] == 1


@pytest.mark.parametrize("tamper", [
    "v18_with_v11", "v19_with_v10", "policy_only", "report_only",
])
def test_archive_rejects_resealed_canary_version_split_crosslinks(
    generated_indexing_failure_archive, tamper,
):
    artifact = deepcopy(generated_indexing_failure_archive)
    policy = artifact["config"]["extraction_canary"]
    report = artifact["execution"]["segments"][0]["extraction_canary"]
    if tamper == "v18_with_v11":
        policy["version"] = report["version"] = "hymem-phase1-extraction-canary-v18"
    elif tamper == "v19_with_v10":
        policy["source_split_policy_version"] = (
            report["source_split_policy_version"]
        ) = "hymem-source-semantic-split-v10"
    elif tamper == "policy_only":
        policy["version"] = "hymem-phase1-extraction-canary-v18"
    else:
        report["version"] = "hymem-phase1-extraction-canary-v18"
    _refresh_manifest(artifact)
    with pytest.raises(BenchmarkIntegrityError):
        protocol.validate_archived_artifact(artifact)


def test_genuine_old_manifest_cannot_resume_current_execution(tmp_path):
    artifact = old_artifact()
    path = tmp_path / "old-checkpoint.json"
    with AtomicCheckpoint(path, manifest=artifact["manifest"], expected_ids=["qid"]):
        pass
    before = path.read_bytes()
    current = _indexing_failure_artifact()
    with pytest.raises(BenchmarkIntegrityError):
        AtomicCheckpoint(path, manifest=current["manifest"], expected_ids=["qid"], resume=True)
    assert path.read_bytes() == before


def test_registry_requires_explicit_history_for_genuine_old_artifact(tmp_path, monkeypatch):
    artifact = old_artifact()
    path = tmp_path / "longmemeval-v2-hymem-20260904T120000Z-seed0-strict.json"
    path.write_bytes(FIXTURE.read_bytes())
    monkeypatch.setattr(lme_registry, "DB", tmp_path / "registry.sqlite")
    conn = lme_registry.connect()
    try:
        assert lme_registry.ingest_file(conn, path).startswith("error: strict")
        assert lme_registry.ingest_file(conn, path, validation_scope="historical") == "inserted"
        assert conn.execute("SELECT strict_validated,archive_validated,live_execution_eligible FROM runs").fetchone() == (0, 1, 0)
    finally:
        conn.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PIN


@pytest.mark.parametrize("version", ["v16", "v20", "unknown"])
def test_archive_rejects_unknown_versions_even_when_report_and_policy_agree(version):
    artifact = old_artifact()
    for policy in (artifact["config"]["extraction_canary"], artifact["execution"]["segments"][0]["extraction_canary"]):
        policy["version"] = "hymem-phase1-extraction-canary-" + version
    _refresh_manifest(artifact)
    with pytest.raises(BenchmarkIntegrityError):
        protocol.validate_archived_artifact(artifact)


@pytest.mark.parametrize("site", ["policy", "report"])
def test_archive_rejects_mixed_v17_v18_and_one_sided_contract_changes(site):
    artifact = old_artifact()
    obj = artifact["config"]["extraction_canary"] if site == "policy" else artifact["execution"]["segments"][0]["extraction_canary"]
    obj.update(canary.extraction_canary_policy())
    _refresh_manifest(artifact)
    with pytest.raises(BenchmarkIntegrityError):
        protocol.validate_archived_artifact(artifact)


@pytest.mark.parametrize("fault", ["extra_truncation", "normal_path", "cap", "fixture_hash", "claim", "context", "contract", "call_deficit"])
def test_archive_rejects_resealed_v17_canary_forgery(fault):
    artifact = old_artifact()
    report = artifact["execution"]["segments"][0]["extraction_canary"]
    policy = artifact["config"]["extraction_canary"]
    if fault == "extra_truncation":
        report["execution_path"]["provider_output_truncations"] = 0
    elif fault == "normal_path":
        for item in (report, policy):
            item["normal_execution_path"]["primary_requests"] += 1
    elif fault == "cap":
        report["max_provider_attempts"] = policy["max_provider_attempts"] = 73
    elif fault == "fixture_hash":
        report["fixture_sha256"] = policy["fixture_sha256"] = "sha256:" + "0" * 64
    elif fault == "claim":
        report["claim_evidence"][0]["object"] = "Invented"
    elif fault == "context":
        report["execution_path"]["table_claim_exact_context_requests"] = 0
    elif fault == "contract":
        report["extraction_contract"]["identity"] = "hymem-extraction-contract-sha256-v1:" + "0" * 64
    else:
        report["usage"]["calls"] = report["usage"]["successful_responses"] = 7
    _refresh_manifest(artifact)
    with pytest.raises(BenchmarkIntegrityError):
        protocol.validate_archived_artifact(artifact)


def test_archive_hash_without_preimage_is_only_a_commitment_not_current_proof(monkeypatch):
    artifact = old_artifact()
    old = artifact["config"]["extraction_canary"]["extraction_contract"]["identity"]
    missing = "hymem-extraction-contract-sha256-v1:" + "e" * 64
    artifact = json.loads(json.dumps(artifact).replace(old, missing))
    _refresh_manifest(artifact)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("historical validation reconstructed live contract")
    monkeypatch.setattr(canary, "extraction_contract_binding", forbidden)
    result = protocol.validate_archived_artifact(artifact)
    assert result["validation_assurance"] == "historical_commitment_only"
    assert result["live_execution_eligible"] is False
    assert artifact["config"]["extraction_canary"]["extraction_contract"]["identity"] == missing


@pytest.mark.parametrize("prompt", ["", " v20", "v20\n", "x" * 129])
def test_archive_prompt_retains_historical_bound(prompt):
    artifact = old_artifact()
    policy = artifact["config"]["extraction_canary"]
    config = artifact["config"]["effective_hymem_config"]
    policy["prompt_version"] = config["prompt_version"] = prompt
    policy["extraction_contract"]["prompt_version"] = prompt
    config["extraction_contract"]["prompt_version"] = prompt
    with pytest.raises(BenchmarkIntegrityError):
        validate_archived_canary_config(policy, config)


@pytest.mark.parametrize("count", [None, True, -1, 25, "1"])
def test_v18_rejects_invalid_or_missing_typed_count(count):
    report = _passed_report()
    if count is None:
        del report["execution_path"]["provider_output_truncations"]
    else:
        report["execution_path"]["provider_output_truncations"] = count
    with pytest.raises(BenchmarkIntegrityError):
        canary.validate_extraction_canary_report(report, expected_mode="required")


@pytest.mark.parametrize("admitted", [0, 1, 2, 7, 9])
def test_v18_rejects_typed_deficits_without_sufficient_admitted_verifications(admitted):
    report = _passed_report()
    report["completion_calls"] = report["provider_attempts"] = 9
    report["usage"].update(calls=admitted, successful_responses=admitted, request_attempts=9)
    report["execution_path"].update(primary_requests=5, parsed_source_records=9,
                                    provider_output_truncations=9 - admitted)
    if admitted == 9:
        # A nonzero count must not be added when all calls were admitted.
        report["execution_path"]["provider_output_truncations"] = 1
    with pytest.raises(BenchmarkIntegrityError):
        canary.validate_extraction_canary_report(report, expected_mode="required")


def test_v18_normal_eight_call_path_cannot_claim_a_typed_rejection():
    report = _passed_report()
    report["execution_path"]["provider_output_truncations"] = 1
    report["usage"]["calls"] = report["usage"]["successful_responses"] = 7
    with pytest.raises(BenchmarkIntegrityError):
        canary.validate_extraction_canary_report(report, expected_mode="required")


def test_v18_emission_counts_cannot_exceed_admitted_texts():
    report = _passed_report()
    report["completion_calls"] = report["provider_attempts"] = 9
    report["usage"]["request_attempts"] = 9
    report["execution_path"].update(primary_requests=5, parsed_source_records=9,
                                    provider_output_truncations=1,
                                    table_claim_exact_context_emissions=8,
                                    prose_claim_exact_context_emissions=1)
    with pytest.raises(BenchmarkIntegrityError, match="emissions"):
        canary.validate_extraction_canary_report(report, expected_mode="required")


def test_v18_failed_report_cannot_count_one_claim_more_than_admitted_texts():
    report = _passed_report()
    report.update(status="failed", completion_calls=9, provider_attempts=9,
                  failure_reason="parse_failure", failure_details=[])
    report["usage"].update(calls=2, successful_responses=2, request_attempts=9)
    report["execution_path"].update(primary_requests=5, parsed_source_records=9,
                                    provider_output_truncations=7,
                                    table_claim_exact_context_emissions=2,
                                    table_claim_wrong_context_emissions=1)
    with pytest.raises(BenchmarkIntegrityError, match="emissions exceed admitted"):
        canary.validate_extraction_canary_report(report, expected_mode="failed")


@pytest.mark.parametrize("claim", ["table", "prose"])
def test_v18_exact_emissions_cannot_exceed_matching_requests(claim):
    report = _passed_report()
    report["completion_calls"] = report["provider_attempts"] = 9
    report["usage"].update(calls=9, successful_responses=9, request_attempts=9)
    report["execution_path"].update(primary_requests=5, parsed_source_records=9)
    report["execution_path"][f"{claim}_claim_exact_context_emissions"] = 3
    with pytest.raises(BenchmarkIntegrityError, match="emissions exceed exact-context"):
        canary.validate_extraction_canary_report(report, expected_mode="required")
