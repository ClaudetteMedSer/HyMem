"""Offline source-scope and identity controls for the versioned SIWC prompt pilot."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
FROZEN = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle")
PROMPT = "hymem/extraction/prompts/__init__.py"
NAMES = (
    ("siwc_lme_diagnostic_v2.py", "siwc_lme_diagnostic_v3.py"),
    ("siwc_lme_diagnostic_progress_v4.py", "siwc_lme_diagnostic_progress_v5.py"),
    ("siwc_lme_diagnostic_launch_v1.py", "siwc_lme_diagnostic_launch_v2.py"),
    ("siwc_lme_diagnostic_host_preflight_v1.py", "siwc_lme_diagnostic_host_preflight_v2.py"),
    ("siwc_lme_diagnostic_source_install_v1.py", "siwc_lme_diagnostic_source_install_v2.py"),
)


def load(name: str):
    path = REPO / "tools/diagnostics" / name
    spec = importlib.util.spec_from_file_location("isolated_" + name[:-3], path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen():
    if not FROZEN.is_dir():
        pytest.skip("local frozen pilot source is unavailable")
    return FROZEN


def source_functions(name: str) -> dict[str, str]:
    source = (REPO / "tools/diagnostics" / name).read_text()
    return {node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_function_parity():
    for old, new in NAMES:
        before, after = source_functions(old), source_functions(new)
        assert before.keys() == after.keys()
        expected = ({"_load_verified"} if old == "siwc_lme_diagnostic_v2.py"
            else {"_success_projection"} if old == "siwc_lme_diagnostic_host_preflight_v1.py"
            else set())
        assert {name for name in before if before[name] != after[name]} == expected


def test_derived_inventory_and_exact_candidate_diff(tmp_path: Path):
    bundle = load("siwc_lme_diagnostic_bundle_v2.py")
    old = frozen()
    output = tmp_path / "bundle"
    result = bundle.assemble(repo=REPO, accepted_code=old / "code",
        candidate=old / "candidate", map_path=old / "source-map.json", output=output)
    assert result["candidate_files"] == 514
    assert result["dataset_present"] is False
    assert result["credential_present"] is False
    assert result["model_calls"] == 0
    assert bundle.sha(output / "source-map.json") == bundle.INVENTORY_SHA256
    before = json.loads((old / "source-map.json").read_bytes())["source_sha256"]
    after = json.loads((output / "source-map.json").read_bytes())["source_sha256"]
    assert len(before) == len(after) == 514
    assert set(before) == set(after)
    assert {key for key in before if before[key] != after[key]} == {PROMPT}
    assert before[PROMPT] == bundle.PROMPT_SOURCE_SHA256
    assert after[PROMPT] == bundle.PROMPT_RESULT_SHA256
    assert hashlib.sha256(json.dumps(after, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest() == bundle.runner_from(REPO).ACCEPTED_MAP_SHA256
    original_files = {str(path.relative_to(old / "candidate")) for path in (old / "candidate").rglob("*") if path.is_file()}
    revised_files = {str(path.relative_to(output / "candidate")) for path in (output / "candidate").rglob("*") if path.is_file()}
    assert original_files == revised_files == set(before)
    for relative in before:
        source, target = old / "candidate" / relative, output / "candidate" / relative
        assert bundle.sha(target) == after[relative]
        if relative != PROMPT:
            assert source.read_bytes() == target.read_bytes()
    transformer = load("siwc_extraction_prompt_repair_v1.py")
    assert (output / "candidate" / PROMPT).read_bytes() == transformer.transform_prompt_source(
        (old / "candidate" / PROMPT).read_bytes())
    with pytest.raises(ValueError, match="bundle_input_invalid"):
        bundle.assemble(repo=REPO, accepted_code=old / "code", candidate=old / "candidate",
            map_path=old / "source-map.json", output=output)


def test_revised_map_rejects_drift():
    bundle = load("siwc_lme_diagnostic_bundle_v2.py")
    old = frozen()
    runner = bundle.runner_from(REPO)
    entries = bundle.candidate_map(old / "candidate", old / "source-map.json", runner)
    assert bundle.revised_inventory(entries, runner).startswith(b'{\n  "source_sha256": {\n')
    altered = {**entries, "README.md": "0" * 64}
    with pytest.raises(ValueError, match="candidate_map_pin_invalid"):
        bundle.revised_inventory(altered, runner)
    altered = {**entries, PROMPT: bundle.PROMPT_RESULT_SHA256}
    with pytest.raises(ValueError, match="candidate_map_pin_invalid"):
        bundle.revised_inventory(altered, runner)


def test_transformer_source_pin_rejects_before_execution(tmp_path: Path):
    bundle = load("siwc_lme_diagnostic_bundle_v2.py")
    path = tmp_path / bundle.TRANSFORMER_RELATIVE
    path.parent.mkdir(parents=True)
    marker = tmp_path / "executed"
    path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n")
    with pytest.raises(ValueError, match="transformer_drift"):
        bundle.transformer_from(tmp_path)
    assert not marker.exists()


def test_host_manifest_and_archive_match_revised_bundle(tmp_path: Path):
    bundle, host = load("siwc_lme_diagnostic_bundle_v2.py"), load("siwc_lme_diagnostic_host_preflight_v2.py")
    old = frozen()
    output = tmp_path / "bundle"
    bundle.assemble(repo=REPO, accepted_code=old / "code", candidate=old / "candidate",
        map_path=old / "source-map.json", output=output)
    manifest = host.source_manifest(output)
    assert len(manifest) == 540
    assert sum(name.startswith("candidate/") for name in manifest) == 514
    assert sum(name.startswith("code/") for name in manifest) == 25
    assert manifest["candidate/" + PROMPT] == bundle.PROMPT_RESULT_SHA256
    assert manifest["source-map.json"] == bundle.INVENTORY_SHA256
    assert manifest[host.RUNNER_REL] == host.RUNNER_SHA
    assert len(host.archive_bytes(output)) < 64 * 1024 * 1024
    (output / "candidate" / PROMPT).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source_file_drift"):
        host.source_manifest(output)


def test_reader_runner_launcher_and_installer_pins():
    runner = load("siwc_lme_diagnostic_v3.py")
    reader = load("siwc_lme_diagnostic_progress_v5.py")
    launcher = load("siwc_lme_diagnostic_launch_v2.py")
    installer = load("siwc_lme_diagnostic_source_install_v2.py")
    host = load("siwc_lme_diagnostic_host_preflight_v2.py")
    runner_sha = hashlib.sha256((REPO / "tools/diagnostics" / runner.RUNNER_RELATIVE.split("/")[-1]).read_bytes()).hexdigest()
    assert runner_sha == reader.RUNNER_SHA256 == launcher.RUNNER_SHA256 == host.RUNNER_SHA
    assert reader.RUNNER_RELATIVE == launcher.RUNNER_RELATIVE == runner.RUNNER_RELATIVE
    assert reader.INVENTORY_SHA256 == launcher.INVENTORY_SHA256 == host.MAP_SHA == runner.ACCEPTED_INVENTORY_SHA256
    assert reader.MAP_SHA256 == runner.ACCEPTED_MAP_SHA256
    assert reader.RUN_SCHEMA == runner.SCHEMA
    assert installer.SOURCE_SHA256 == hashlib.sha256((REPO / installer.SOURCE).read_bytes()).hexdigest()
    command = launcher.command(Path("/home/atta/.hymem-siwc-lme-diagnostic-abcdefgh"),
        {"unit": "hymem-siwc-lme-diagnostic-abcdefgh.service",
         "runtime_path": str(runner.RUNTIME_PATH)}, "0" * 64)
    assert command[-9:] == ["--questions", "4", "--workers", "4",
        "--receipt-sha256", "0" * 64, "--output-dir",
        "/home/atta/.hymem-siwc-lme-diagnostic-abcdefgh/run", "--run"]


def test_source_install_projection_rejects_failure_and_extra_fields():
    installer = load("siwc_lme_diagnostic_source_install_v2.py")
    root = "/home/atta/.hymem-siwc-lme-diagnostic-preflight-abcdefgh"
    value = {"schema": installer.SCHEMA, "prepared": True, "model_calls": 0,
        "root": root, "unit": "hymem-siwc-lme-diagnostic-preflight-abcdefgh.service",
        "receipt_sha256": "a" * 64}
    assert installer.project(value, root=root, returncode=0) == value
    assert installer.project(value, root=root, returncode=1) is None
    assert installer.project({**value, "model_calls": 1}, root=root, returncode=0) is None
    assert installer.project({**value, "private": "x"}, root=root, returncode=0) is None


def test_reader_reconciles_five_ledgers_and_four_question_rows():
    reader = load("siwc_lme_diagnostic_progress_v5.py")
    observations, rows = {}, []
    budget = {"turns": 10, "known_tokens": 90, "questions": {}}
    for index in range(5):
        qid = "canary" if index == 0 else f"q-{index-1:04d}"
        prefix = "canary" if index == 0 else f"question.{index-1}"
        summary = {"schema": "siwc_lme_summary_v1", "calls": 1,
            "successes": 1, "failures": 0, "internal_http_attempts": 1,
            "provider_internal_retries_known": False, "admitted_turns": 2,
            "known_tokens": 18, "usage_complete": True,
            "timing_seconds": {"total": 1.0, "admission": 0.25, "http": 0.75},
            "timing_saturated": False, "first_failure": None,
            "last_failure_code": None}
        budget["questions"][qid] = {"turns": 2, "known_tokens": 18}
        observations[prefix + ".ordinary"] = {"status": "observed", "summary": summary}
        observations[prefix + ".structured"] = {"status": "observed", "summary": summary}
        rows.append({"question_id": qid, "ordinary": summary, "structured": summary,
            "ledger": {"admitted_turns": 2, "known_tokens": 18, "usage_complete": True}})
    value = {"schema": "siwc_lme_pilot_projection_v1", "canary": rows[0],
        "questions": rows[1:], "aggregate": {"calls": 10, "successes": 10,
            "failures": 0, "internal_http_attempts": 10,
            "admitted_turns": 10, "known_tokens": 90}}
    assert reader.pilot(value, observations, budget) == value
    overcounted = json.loads(json.dumps(value))
    overcounted["aggregate"]["known_tokens"] = 180
    with pytest.raises(ValueError, match="siwc_pilot_aggregate_invalid"):
        reader.pilot(overcounted, observations, budget)
    missing = json.loads(json.dumps(value))
    missing["questions"].pop()
    with pytest.raises(ValueError, match="siwc_pilot_invalid"):
        reader.pilot(missing, observations, budget)
    assert "indexing_failure:timeout_during_cycle" in reader.QUESTION_FAILURE_CODES
    assert "indexing_failure:private" not in reader.QUESTION_FAILURE_CODES
