"""Offline controls for finite failed-run reporting; no provider or service calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.diagnostics import luna_lme_diagnostic_progress_v2 as reader
from tools.diagnostics.tests import test_luna_lme_diagnostic_progress_v1 as prior


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, receipt, digest = prior.prepared.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(reader, "ROOT_PARENT", tmp_path)
    monkeypatch.setattr(reader, "ROOT_UID", os.getuid())
    return root, receipt, digest


def _failed_writer(root: Path) -> dict:
    ids = [f"q{i}" for i in range(4)]
    manifest = prior._manifest(ids)
    payload = json.dumps({"path": str(root / "run/diagnostic-checkpoint.json"),
                          "manifest": manifest, "ids": ids})
    script = r'''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from benchmarks.strictness import AtomicCheckpoint
p=json.loads(sys.argv[2]);path=Path(p['path']);path.parent.mkdir(parents=True,exist_ok=True)
with AtomicCheckpoint(path,manifest=p['manifest'],expected_ids=p['ids'],scored=True,
                      resume=False,retry_failures=False) as checkpoint:
    for qid in p['ids']:
        checkpoint.record(qid,row=None,failure='question_failure')
    checkpoint.finalize()
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", script,
                    str(prior.CANDIDATE), payload], check=True,
                   capture_output=True, text=True, timeout=20)
    return json.loads((root / "run/diagnostic-checkpoint.json").read_text())


def _budget() -> dict:
    script = r'''
import json,sys
sys.path.insert(0,sys.argv[1])
from benchmarks.codex_subscription_warm_v3 import SharedBudget,BudgetLimits
b=SharedBudget(BudgetLimits(8012,48160000,14400),max_in_flight=4)
b.turns=14;b.known_tokens=113065
b.record_first_failure('rpc_failure:thread/start',{'phase':'preflight',
    'rpc':'thread/start','process_index':1,'request_index':1,'retired_count':0,
    'queue_count':0,'turn_admitted':False,'known_usage':True,
    'known_tokens':113065,'usage_complete':False,
    'rpc_error':{'category':'internal_error','code':-32603}})
print(json.dumps(b.snapshot()))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                             str(prior.ASSEMBLY / "code")], check=True,
                            capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout)


def _failed_terminal(checkpoint: dict, budget: dict) -> dict:
    return {"schema": reader.RUN_SCHEMA, "run_id": checkpoint["run_id"],
        "canonical_r9_artifact": False, "official_model_score": False,
        "selected_denominator": 4, "scored_count": 0,
        "correct_count": 0, "incorrect_count": 0,
        "quality_accuracy_full_selected": None,
        "failed_or_unscored_count": 4, "strict_unhealthy_count": 0,
        "canary": {"structural_valid": True, "model_gold_match": False},
        "campaign_stop": "rpc_failure:thread/start",
        "checkpoint_counts": checkpoint["counts"], "budget": budget}


def test_exact_transport_failure_reports_bounded_phase_and_cleanup(prepared, monkeypatch):
    root, _, digest = prepared
    prior._json(root / "launch-attempt.json", {"receipt_sha256": digest,
                                              "one_shot": True})
    checkpoint = _failed_writer(root)
    budget = _budget()
    prior._json(root / "run/diagnostic-result.json", _failed_terminal(checkpoint, budget))
    monkeypatch.setattr(reader, "_runtime", lambda _: "unverified")
    monkeypatch.setattr(reader, "_failed_exit_cleanup", lambda _: True)
    report = reader.inspect(root, digest)
    assert report["status"] == "terminal_incomplete_or_unclean"
    assert report["completed_diagnostic_and_clean"] is False
    assert report["runtime_cleanup_verified"] is True
    assert (report["selected_denominator"], report["scored_count"],
            report["failed_count"]) == (4, 0, 4)
    assert report["known_turns"] == 14 and report["known_tokens"] == 113065
    assert report["usage_complete"] is True
    assert report["first_failure"]["code"] == "rpc_failure:thread/start"
    assert report["first_failure"]["phase"] == "preflight"
    assert report["first_failure"]["rpc_error_category"] == "internal_error"
    assert report["first_failure"]["rpc_error_code"] == -32603


@pytest.mark.parametrize("code", ["rpc_failure:arbitrary", "rpc_failure:thread/start:secret",
                                  "arbitrary_provider_text"])
def test_unknown_stop_codes_fail_closed(code):
    assert reader._safe_stop(code) is False


def test_usage_unknown_and_invalid_rpc_error_are_finite():
    assert reader._safe_stop("usage_unknown") is True
    fault = {"first_failure": {"code": "rpc_failure:thread/start",
        "phase": "preflight", "rpc": "thread/start", "process_index": 1,
        "request_index": 1, "retired_count": 0, "queue_count": 0,
        "turn_admitted": False, "known_usage": False, "usage_complete": False,
        "rpc_error": {"category": "invalid"},
        "app_server_error": {"identity": "matched", "will_retry": False,
            "error_class": "rateLimitExceeded", "http_status_code": 429}}}
    projection = reader._failure_projection(fault)
    assert projection["rpc_error_category"] == "invalid"
    assert "rpc_error_code" not in projection
    assert projection["app_server_error"] == {"identity": "matched",
        "will_retry": False, "error_class": "rateLimitExceeded",
        "http_status_code": 429}


def test_prepared_state_remains_readable(prepared, monkeypatch):
    root, _, digest = prepared
    monkeypatch.setattr(reader, "_runtime", lambda _: pytest.fail("unexpected runtime query"))
    assert reader.inspect(root, digest)["status"] == "prepared_not_launched"


def test_failed_exit_cleanup_requires_empty_cgroup_and_exact_policy(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    (runtime / "bus").touch()
    cgroup_root = tmp_path / "cgroup"
    name = "/user.slice/test.service"
    group = cgroup_root / name.lstrip("/")
    group.mkdir(parents=True)
    for file, content in {"memory.max": "4294967296\n", "pids.max": "128\n",
                          "cpu.max": "200000 100000\n", "cgroup.procs": "",
                          "cgroup.threads": "", "cgroup.events": "populated 0\n"}.items():
        (group / file).write_text(content)
    values = {"ActiveState": "failed", "SubState": "failed", "MainPID": "0",
        "ControlGroup": name, "NRestarts": "0", "Result": "exit-code",
        "ExecMainStatus": "1", "MemoryMax": "4294967296", "TasksMax": "128",
        "CPUQuotaPerSecUSec": "2s", "KillMode": "control-group", "Restart": "no",
        "RemainAfterExit": "yes", "OOMPolicy": "kill", "RuntimeMaxUSec": "14530s",
        "TimeoutStopUSec": "10s"}
    monkeypatch.setattr(reader, "RUNTIME", runtime)
    monkeypatch.setattr(reader, "CGROUP_ROOT", cgroup_root)
    monkeypatch.setattr(reader, "ROOT_UID", os.getuid())
    monkeypatch.setattr(reader, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(reader.stat, "S_ISSOCK", lambda _: True)
    monkeypatch.setattr(reader.subprocess, "run", lambda *_args, **_kwargs:
        SimpleNamespace(stdout="\n".join(f"{k}={v}" for k, v in values.items())))
    receipt = {"unit": "test.service", "expected_cgroup": name}
    assert reader._failed_exit_cleanup(receipt) is True
    (group / "cgroup.events").write_text("populated 1\n")
    assert reader._failed_exit_cleanup(receipt) is False
    (group / "cgroup.events").write_text("populated 0\n")
    values["TasksMax"] = "512"
    assert reader._failed_exit_cleanup(receipt) is False
    values["TasksMax"] = "128"
    values["MainPID"] = "123"
    assert reader._failed_exit_cleanup(receipt) is False
