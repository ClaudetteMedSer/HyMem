"""Independent root controls for diagnostic capacity integration, no inference."""
from __future__ import annotations

import ast
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_lme_diagnostic_v1 as old


def _candidate():
    # Import lazily so the root's controls can be prepared while Sol implements.
    from tools.diagnostics import luna_lme_diagnostic_v2
    return luna_lme_diagnostic_v2


def _function_ast(path: str, name: str) -> str:
    tree = ast.parse(Path(path).read_text())
    function = next(node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == name)
    return ast.dump(function, include_attributes=False)


def test_capacity_repair_does_not_change_candidate_model_or_measurement():
    new = _candidate()
    for name in ("MAX_LIMITS", "ACCEPTED_FILES", "ACCEPTED_MAP_SHA256",
                 "DATASET_SHA256", "PINS", "DIAGNOSTIC_HELPER_SHA256"):
        assert getattr(new, name) == getattr(old, name), name
    for name in ("run_live_canary", "_canary_gold", "validate_diagnostic_row",
                 "_identity_manifest", "_question_worker", "make_dual"):
        assert _function_ast(new.__file__, name) == _function_ast(old.__file__, name), name


def _containment(tmp_path, monkeypatch):
    runner = _candidate()
    actual_path = Path
    root = tmp_path / "cgroup"
    relative = "/user.slice/user-1000.slice/user@1000.service/app.slice/test.service"
    group = root / relative.lstrip("/")
    group.mkdir(parents=True)
    for name, value in {
        "memory.max": "4294967296", "pids.max": "256", "cpu.max": "200000 100000",
        "pids.current": "17", "pids.peak": "17", "pids.events": "max 0",
        "cgroup.procs": str(os.getpid()), "cgroup.threads": str(os.getpid()),
        "cgroup.events": "populated 1",
    }.items():
        (group / name).write_text(value + "\n")
    membership = tmp_path / "self-cgroup"
    membership.write_text("0::" + relative + "\n")

    def mapped_path(value):
        value = os.fspath(value)
        if value == "/proc/self/cgroup":
            return membership
        if value == "/sys/fs/cgroup" or value.startswith("/sys/fs/cgroup/"):
            return root / value.removeprefix("/sys/fs/cgroup").lstrip("/")
        return actual_path(value)

    props = {"ActiveState": "active", "SubState": "running", "MainPID": str(os.getpid()),
        "ControlGroup": relative, "NRestarts": "0", "MemoryMax": "4294967296",
        "TasksMax": "256", "CPUQuotaPerSecUSec": "2s", "KillMode": "control-group",
        "Restart": "no", "RemainAfterExit": "yes", "OOMPolicy": "kill",
        "RuntimeMaxUSec": "14530s", "TimeoutStopUSec": "10s"}
    monkeypatch.setattr(runner, "Path", mapped_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="\n".join(f"{k}={v}" for k, v in props.items())))
    receipt = {"unit": "test.service", "expected_cgroup": relative}
    return runner, props, group, receipt


def test_measured_task_bound_is_exact_and_other_limits_stay_fixed(tmp_path, monkeypatch):
    runner, props, group, receipt = _containment(tmp_path, monkeypatch)
    assert runner.verify_live_containment(tmp_path, receipt) is True
    for bad in ("128", "257", "512", "max"):
        props["TasksMax"] = bad
        with pytest.raises(ValueError):
            runner.verify_live_containment(tmp_path, receipt)
    props["TasksMax"] = "256"
    for bad in ("128", "257", "max"):
        (group / "pids.max").write_text(bad)
        with pytest.raises(ValueError):
            runner.verify_live_containment(tmp_path, receipt)
    (group / "pids.max").write_text("256")
    for key, bad in (("MemoryMax", "8589934592"), ("CPUQuotaPerSecUSec", "4s"),
                     ("Restart", "on-failure"), ("MainPID", "0"),
                     ("RuntimeMaxUSec", "29060s"), ("KillMode", "process")):
        previous = props[key]
        props[key] = bad
        with pytest.raises(ValueError):
            runner.verify_live_containment(tmp_path, receipt)
        props[key] = previous


@pytest.mark.parametrize("fault", ["rpc", "denial", "unreadable"])
def test_actual_campaign_observer_records_before_cleanup_and_stops(
    tmp_path, monkeypatch, fault,
):
    """Exercise the new nested ledger, not a copy of its implementation."""
    runner = _candidate()
    events = []
    state = {"phase": "start", "fault": False}

    class Stop(Exception):
        pass

    class Budget:
        def __init__(self, *_args, **_kwargs):
            self._lock = threading.RLock()
            self.stop_code = None
            self.first_failure = None

        def halt(self, code):
            self.stop_code = self.stop_code or code

        def reserve(self, key):
            events.append("base_reserve")
            assert events[-2] == "sample:worker"
            return key

        def before_turn(self, key, admission):
            events.append("base_before_turn")
            assert events[-2] == "sample:worker"
            assert admission == "admission"
            return key

        def record_first_failure(self, code, metadata):
            events.append("record")
            assert "cleanup" not in events
            self.first_failure = {"code": code, **metadata}
            self.halt(code)

        def snapshot(self):
            return {"turns": 0, "known_tokens": 0, "reserved": 0, "in_flight": 0,
                    "usage_complete": True, "stopped": self.stop_code is not None,
                    "stop_code": self.stop_code, "first_failure": self.first_failure}

    class Checkpoint:
        def __init__(self, *_args, **_kwargs):
            self.rows = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def record(self, key, *, row, failure=None):
            self.rows[key] = row

        def finalize(self):
            return {"counts": {"completed": 0, "failed": len(self.rows)}}

    def sample(_group):
        events.append("sample:" + state["phase"])
        if state["fault"] and fault == "unreadable":
            raise OSError("unavailable")
        return {"current": 40 if state["phase"] != "cleaned" else 1,
                "peak": 100, "limit": 256,
                "denials": 3 if state["fault"] and fault == "denial" else 0}

    def worker(_loaded, budget, *_args):
        state["phase"] = "worker"
        assert budget.reserve("q1") == "q1"
        assert budget.before_turn("q1", "admission") == "q1"
        state["fault"] = True
        budget.record_first_failure("rpc_failure:thread/start", {"phase": "preflight"})
        observed = budget.snapshot()["first_failure"]
        assert "cleanup" not in events
        if fault == "unreadable":
            assert observed["resource_observation"] is None
        else:
            assert observed["resource_observation"]["current"] == 40
        if fault != "rpc":
            dispatched = events.count("base_reserve") + events.count("base_before_turn")
            with pytest.raises(Stop):
                budget.reserve("q1")
            with pytest.raises(Stop):
                budget.before_turn("q1", "admission")
            assert events.count("base_reserve") + events.count("base_before_turn") == dispatched
        events.append("cleanup")
        state["phase"] = "cleaned"
        return {"projection": None, "accounting": None,
                "stop_code": budget.stop_code}

    loaded = {"questions": [{"question_id": "q1"}], "dataset": tmp_path / "dataset",
              "warm": SimpleNamespace(SharedBudget=Budget, ConcurrentStop=Stop),
              "strictness": SimpleNamespace(AtomicCheckpoint=Checkpoint),
              "prior": SimpleNamespace(atomic_private=lambda *_args: None)}
    monkeypatch.setattr(runner, "_resource_sample", sample)
    monkeypatch.setattr(runner, "_sha", lambda _path: runner.DATASET_SHA256)
    monkeypatch.setattr(runner, "_identity_manifest", lambda *_args: {"run_id": "frozen"})
    monkeypatch.setattr(runner, "run_live_canary", lambda *_args: {
        "structural_valid": True, "model_gold_match": False,
        "quality_failure_reason": "branch_incomplete", "semantic_failure_proved": True})
    monkeypatch.setattr(runner, "_question_worker", worker)
    cap = lambda n, t, s: SimpleNamespace(turns=n, known_tokens=t, seconds=s)
    result = runner.run_campaign(loaded, output=tmp_path / "out",
        campaign_limits=cap(100, 1000, 100), canary_limits=cap(10, 100, 20),
        question_limits=cap(50, 500, 50), indexing_seconds=10, workers=1,
        helper_sha256="0" * 64, containment=lambda _loaded: True,
        resource_cgroup="/bound.service")
    expected = {"rpc": "rpc_failure:thread/start", "denial": "resource_task_denial",
                "unreadable": "resource_observer_unverified"}[fault]
    assert result["budget"]["stop_code"] == result["campaign_stop"] == expected
    assert result["budget"]["first_failure"]["code"] == expected
    assert result["selected_denominator"] == result["failed_or_unscored_count"] == 1
    assert result["scored_count"] == 0
    assert result["quality_accuracy_full_selected"] is None
    assert events.index("record") < events.index("cleanup")
    assert events[-1] == "sample:cleaned"
