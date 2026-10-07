"""Root-owned metadata privacy/completion controls; no external processes."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_timeout_host_v1 as host
from tools.diagnostics import luna_timeout_probe_v1 as probe
from tools.diagnostics import luna_timeout_progress_v1 as progress


def canonical(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    path.chmod(0o600)


def configure(tmp_path, monkeypatch, *, clean=True):
    spec = importlib.util.spec_from_file_location("root_timeout_fixture", Path(__file__).with_name("test_luna_timeout_probe_v1.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    observer = fixture.observer_module()
    result, _ = fixture.run_fake()
    counters = dict(pids_denials=0, memory_oom=0, memory_oom_kill=0,
                    pids_current=1, memory_current=100, memory_peak=100)
    def sample(phase, index=None):
        return {"phase": phase, "index": index, "resources": dict(counters),
                "gate": {"containment": True, "denials": 0, "oom": 0}}
    outer = {"schema": host.SCHEMA, "mode": "probe", "status": "observed_success",
        "failure_code": None, "probe_result": result,
        "samples": [sample("initial"), sample("initial")] +
                   [sample("before_admission", i) for i in range(16)] + [sample("terminal")],
        "independent_recursive_cleanup_verified": None}
    assert host.validate_host_result(outer, probe=probe, observer=observer)
    fake = SimpleNamespace(verify_receipt=lambda *args: {"unit": "test.service"},
        verify_sources=lambda root: (probe, (observer, object())),
        terminal_runtime=lambda receipt: dict(policy_verified=True, unit_stopped=True,
            group_matched=True, recursive_cleanup_verified=clean, runtime_exit="success"),
        PROBE_SHA256=host.PROBE_SHA256, OBSERVER_SHA256=host.OBSERVER_SHA256,
        PREPARATION_SHA256=host.PREPARATION_SHA256, validate_host_result=host.validate_host_result)
    monkeypatch.setattr(progress, "_host", lambda root: fake)
    monkeypatch.setattr(progress, "HOST_UID", os.getuid())
    digest = "a" * 64
    canonical(tmp_path / "launch-attempt.json", {"receipt_sha256": digest, "one_shot": True})
    canonical(tmp_path / "probe-execution-marker.json", {"receipt_sha256": digest, "execution_started": True})
    return outer, digest


def test_root_missing_result_is_unknown_even_after_exit_zero(tmp_path, monkeypatch):
    _, digest = configure(tmp_path, monkeypatch)
    result = progress.inspect(tmp_path, digest, "probe")
    assert result["model_calls"] is None
    assert result["result_verified"] is False
    assert result["completed_and_clean"] is False
    assert result["status"] == "result_missing"


def test_root_result_without_execution_marker_is_rejected(tmp_path, monkeypatch):
    outer, digest = configure(tmp_path, monkeypatch)
    canonical(tmp_path / "probe-result.json", outer)
    (tmp_path / "probe-execution-marker.json").unlink()
    with pytest.raises(ValueError, match="result_without_execution"):
        progress.inspect(tmp_path, digest, "probe")


@pytest.mark.parametrize("clean", [False, True])
def test_root_reader_needs_independent_cleanup_not_only_complete_result(tmp_path, monkeypatch, clean):
    outer, digest = configure(tmp_path, monkeypatch, clean=clean)
    canonical(tmp_path / "probe-result.json", outer)
    result = progress.inspect(tmp_path, digest, "probe")
    assert result["result_verified"] is True
    assert result["completed_and_clean"] is clean
    assert result["probe"]["counts"]["returned"] == 16
    assert result["lme_readiness_proved"] is False
    assert result["historical_timeout_cause_proved"] is False


@pytest.mark.parametrize("mutation", ["private_field", "duplicate", "truncated"])
def test_root_reader_never_prints_malformed_private_result(tmp_path, monkeypatch, capsys, mutation):
    outer, digest = configure(tmp_path, monkeypatch)
    path = tmp_path / "probe-result.json"
    if mutation == "private_field":
        outer["private"] = "NEVER-EXPORT-PRIVATE"
        canonical(path, outer)
    elif mutation == "duplicate":
        path.write_text('{"private":"NEVER-EXPORT-PRIVATE","private":"other"}')
    else:
        path.write_text('{"private":"NEVER-EXPORT-PRIVATE')
    path.chmod(0o600)
    assert progress.main(["--root", str(tmp_path), "--receipt-sha256", digest, "--mode", "probe"]) == 1
    output = capsys.readouterr().out
    assert "PRIVATE" not in output
    assert json.loads(output)["status"] == "unverified"
