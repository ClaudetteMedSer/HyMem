"""Offline metadata-reader controls; no service or model invocation."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.diagnostics import luna_lme_diagnostic_progress_v1 as reader


ASSEMBLY = Path("/private/tmp/hymem-lme-diagnostic-offline-assembly-v2")
CANDIDATE = ASSEMBLY / "candidate"


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not ASSEMBLY.is_dir():
        pytest.skip("accepted offline assembly unavailable")
    root = tmp_path / ".hymem-lme-diagnostic-reader-test001"
    root.mkdir(mode=0o700)
    shutil.copytree(CANDIDATE, root / "candidate")
    shutil.copytree(ASSEMBLY / "code", root / "code")
    shutil.copyfile(ASSEMBLY / "source-map.json", root / "source-map.json")
    monkeypatch.setattr(reader, "ROOT_PARENT", tmp_path)
    monkeypatch.setattr(reader, "ROOT_UID", os.getuid())
    unit = "hymem-luna-lme-diagnostic-reader-test001.service"
    receipt = {"schema": "luna-lme-diagnostic-launch-v1", "root": str(root),
        "unit": unit,
        "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "source_sha256": reader.PINS, "candidate_map_sha256": reader.MAP_SHA256,
        "inventory_sha256": reader.INVENTORY_SHA256,
        "dataset_sha256": reader.DATASET_SHA256,
        "binary_sha256": reader.BINARY_SHA256, "selected_count": 4,
        "workers": 4, "indexing_seconds": 10_800,
        "output_dir": str(root / "run"), "limits": reader.LIMITS}
    _json(root / "launch-receipt.json", receipt)
    digest = reader._sha(root / "launch-receipt.json")
    return root, receipt, digest


def _manifest(ids: list[str]) -> dict:
    manifest = {"schema": reader.RUN_SCHEMA, "mode": reader.MODE,
        "canonical_r9_artifact": False, "official_model_score": False,
        "candidate_map_sha256": reader.MAP_SHA256,
        "dataset_sha256": reader.DATASET_SHA256,
        "selected_row_sha256": ["1" * 64 for _ in ids],
        "selected_source_order": "first_n", "expected_count": len(ids),
        "expected_ids_hash": reader._canonical_hash(ids), "scored_run": True,
        "diagnostic_helper_sha256": reader.PINS["benchmarks/lme_diagnostic.py"],
        "runner_sha256": reader.RUNNER_SHA256,
        "transport_sha256": reader.PINS["benchmarks/codex_subscription_staged_v1.py"],
        "ordinary_transport_sha256": reader.PINS["benchmarks/codex_subscription_warm_v3.py"],
        "limits": {"campaign": dict(zip(("turns", "known_tokens", "seconds"),
                                          reader.LIMITS["campaign"])),
                   "canary": dict(zip(("turns", "known_tokens", "seconds"),
                                        reader.LIMITS["canary"])),
                   "question": dict(zip(("turns", "known_tokens", "seconds"),
                                          reader.LIMITS["question"])),
                   "indexing_seconds": 10_800, "workers": 4}, "rerolls": 0}
    manifest["run_id"] = reader._canonical_hash(manifest)
    return manifest


def _frozen_checkpoint(root: Path, *, semantic: bool = True,
                       failed: bool = False) -> dict:
    """Use the candidate's actual AtomicCheckpoint writer in an isolated process."""
    ids = [f"q{i}" for i in range(4)]
    manifest = _manifest(ids)
    payload = json.dumps({"path": str(root / "run/diagnostic-checkpoint.json"),
                          "manifest": manifest, "ids": ids, "semantic": semantic,
                          "failed": failed})
    script = r'''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from benchmarks.strictness import AtomicCheckpoint
p=json.loads(sys.argv[2]);path=Path(p['path']);path.parent.mkdir(parents=True,exist_ok=True)
with AtomicCheckpoint(path,manifest=p['manifest'],expected_ids=p['ids'],scored=True,
                      resume=False,retry_failures=False) as checkpoint:
    for i,qid in enumerate(p['ids']):
        if p['failed'] and i == len(p['ids'])-1:
            checkpoint.record(qid,row=None,failure='question_failure')
            continue
        checkpoint.record(qid,row={'question_id':qid,'correct':i%2==0,
            'benchmark_failure':None,
            'diagnostic_kind':'semantic_quarantine' if p['semantic'] and i==0 else 'strict_healthy',
            'strict_indexing_healthy':not(p['semantic'] and i==0),
            'quarantined_chunks':1 if p['semantic'] and i==0 else 0,
            'summary_degraded_sessions':1 if p['semantic'] and i==0 else 0,
            'context_sha':'a'*64})
    checkpoint.finalize()
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(CANDIDATE), payload],
                   check=True, capture_output=True, text=True, timeout=20)
    return json.loads((root / "run/diagnostic-checkpoint.json").read_text())


def _terminal(checkpoint: dict, *, usage_complete=True, stopped=False) -> dict:
    counts = checkpoint["counts"]
    return {"schema": reader.RUN_SCHEMA, "run_id": checkpoint["run_id"],
        "canonical_r9_artifact": False, "official_model_score": False,
        "selected_denominator": 4, "scored_count": 4,
        "correct_count": 2, "incorrect_count": 2,
        "quality_accuracy_full_selected": 0.5,
        "failed_or_unscored_count": 0, "strict_unhealthy_count": 1,
        "canary": {"structural_valid": True, "model_gold_match": False},
        "campaign_stop": "usage_unknown" if stopped else None,
        "checkpoint_counts": counts,
        "budget": {"turns": 100, "known_tokens": 5000, "reserved": 0,
                   "in_flight": 0, "usage_complete": usage_complete,
                   "stopped": stopped, "stop_code": "usage_unknown" if stopped else None}}


def test_prepared_root_is_reported_without_service_query(prepared, monkeypatch):
    root, _, digest = prepared
    monkeypatch.setattr(reader, "_runtime", lambda _receipt: pytest.fail(
        "prepared status queried nonexistent unit"))
    report = reader.inspect(root, digest)
    assert report["status"] == "prepared_not_launched"
    assert report["completed_diagnostic_and_clean"] is False
    assert report["known_turns"] is None


@pytest.mark.parametrize("change", [
    lambda r: r.update(selected_count=True),
    lambda r: r.update(extra="value"),
    lambda r: r["source_sha256"].update({"extra.py": "0" * 64}),
])
def test_receipt_malformed_or_extra_source_fails(prepared, change):
    root, receipt, _ = prepared
    modified = copy.deepcopy(receipt)
    change(modified)
    _json(root / "launch-receipt.json", modified)
    with pytest.raises(ValueError):
        reader.inspect(root, reader._sha(root / "launch-receipt.json"))


def test_candidate_file_drift_fails_before_status(prepared):
    root, _, digest = prepared
    target = root / "candidate/benchmarks/extraction_canary.py"
    target.write_text(target.read_text() + "\n# drift\n")
    with pytest.raises(ValueError, match="candidate_source_drift"):
        reader.inspect(root, digest)


def test_semantic_loss_can_complete_clean_with_separate_health_and_quality(prepared, monkeypatch):
    root, _, digest = prepared
    _json(root / "launch-attempt.json", {"receipt_sha256": digest, "one_shot": True})
    _json(root / "launch-command-result.json", {"returncode": 0})
    checkpoint = _frozen_checkpoint(root)
    _json(root / "run/diagnostic-result.json", _terminal(checkpoint))
    monkeypatch.setattr(reader, "_runtime", lambda _receipt: "clean_exit")
    result = reader.inspect(root, digest)
    assert result["completed_diagnostic_and_clean"] is True
    assert result["strict_indexing_healthy_for_all"] is False
    assert result["summary_degraded_sessions_total"] == 1
    assert result["canary_model_gold_match"] is False
    assert result["known_turns"] == 100 and result["known_tokens"] == 5000


@pytest.mark.parametrize("mutation", [
    lambda terminal: terminal.update(correct_count=3, incorrect_count=1),
    lambda terminal: terminal.update(strict_unhealthy_count=0),
    lambda terminal: terminal["budget"].update(turns=True),
    lambda terminal: terminal["canary"].update(model_gold_match=1),
])
def test_terminal_metadata_drift_rejected(prepared, monkeypatch, mutation):
    root, _, digest = prepared
    _json(root / "launch-attempt.json", {"receipt_sha256": digest, "one_shot": True})
    checkpoint = _frozen_checkpoint(root)
    terminal = _terminal(checkpoint)
    mutation(terminal)
    _json(root / "run/diagnostic-result.json", terminal)
    monkeypatch.setattr(reader, "_runtime", lambda _receipt: "clean_exit")
    with pytest.raises(ValueError):
        reader.inspect(root, digest)


def test_unknown_usage_or_wrong_runtime_never_clean(prepared, monkeypatch):
    root, _, digest = prepared
    _json(root / "launch-attempt.json", {"receipt_sha256": digest, "one_shot": True})
    checkpoint = _frozen_checkpoint(root)
    _json(root / "run/diagnostic-result.json", _terminal(checkpoint, usage_complete=False))
    monkeypatch.setattr(reader, "_runtime", lambda _receipt: "clean_exit")
    assert reader.inspect(root, digest)["completed_diagnostic_and_clean"] is False
    _json(root / "run/diagnostic-result.json", _terminal(checkpoint))
    monkeypatch.setattr(reader, "_runtime", lambda _receipt: "unverified")
    assert reader.inspect(root, digest)["completed_diagnostic_and_clean"] is False


def test_failed_question_retains_denominator_without_accuracy(prepared, monkeypatch):
    root, _, digest = prepared
    _json(root / "launch-attempt.json", {"receipt_sha256": digest, "one_shot": True})
    checkpoint = _frozen_checkpoint(root, failed=True)
    terminal = _terminal(checkpoint, stopped=True)
    terminal.update(scored_count=3, correct_count=2, incorrect_count=1,
                    quality_accuracy_full_selected=None,
                    failed_or_unscored_count=1)
    _json(root / "run/diagnostic-result.json", terminal)
    monkeypatch.setattr(reader, "_runtime", lambda _receipt: "clean_exit")
    result = reader.inspect(root, digest)
    assert result["status"] == "terminal_incomplete_or_unclean"
    assert result["completed_diagnostic_and_clean"] is False
    assert result["selected_denominator"] == 4 and result["scored_count"] == 3


def test_runtime_requires_actual_cgroup_policy_and_empty_descendant_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    bus = runtime / "bus"
    bus.touch()
    group_root = tmp_path / "cgroup"
    name = "/user.slice/test.service"
    group = group_root / name.lstrip("/")
    group.mkdir(parents=True)
    for item, content in {
        "memory.max": "4294967296\n", "pids.max": "128\n",
        "cpu.max": "200000 100000\n", "cgroup.procs": "123\n",
        "cgroup.threads": "123\n", "cgroup.events": "populated 1\n",
    }.items():
        (group / item).write_text(content)
    values = {"ActiveState": "active", "SubState": "running", "MainPID": "123",
        "ControlGroup": name, "NRestarts": "0", "Result": "success",
        "ExecMainStatus": "0", "MemoryMax": "4294967296", "TasksMax": "128",
        "CPUQuotaPerSecUSec": "2s", "KillMode": "control-group",
        "Restart": "no", "RemainAfterExit": "yes", "OOMPolicy": "kill",
        "RuntimeMaxUSec": "14530s", "TimeoutStopUSec": "10s"}
    monkeypatch.setattr(reader, "RUNTIME", runtime)
    monkeypatch.setattr(reader, "CGROUP_ROOT", group_root)
    monkeypatch.setattr(reader, "ROOT_UID", os.getuid())
    monkeypatch.setattr(reader, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(reader.stat, "S_ISSOCK", lambda _mode: True)
    monkeypatch.setattr(reader.subprocess, "run", lambda *_args, **_kwargs:
        SimpleNamespace(stdout="\n".join(f"{k}={v}" for k, v in values.items())))
    receipt = {"unit": "test.service", "expected_cgroup": name}
    assert reader._runtime(receipt) == "running_verified"
    (group / "pids.max").write_text("512\n")
    assert reader._runtime(receipt) == "unverified"
    (group / "pids.max").write_text("128\n")
    values.update(ActiveState="active", SubState="exited", MainPID="0",
                  ControlGroup="")
    (group / "cgroup.procs").write_text("")
    (group / "cgroup.threads").write_text("")
    assert reader._runtime(receipt) == "unverified"  # child cgroup populated
    (group / "cgroup.events").write_text("populated 0\n")
    assert reader._runtime(receipt) == "clean_exit"
