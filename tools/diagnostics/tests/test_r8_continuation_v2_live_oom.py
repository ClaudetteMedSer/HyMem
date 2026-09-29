"""Offline checks: a reaped live OOM is failed evidence, not ambiguity."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "continuation_v2_oom_fixture",
    Path(__file__).with_name("test_r8_headless_continuation_v2.py"),
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)
M = FIXTURE.M


def run_with_live_state(monkeypatch, change, *, live_code=137):
    fixture = FIXTURE.setup(monkeypatch, live_code=live_code)

    def sending(*args, **kwargs):
        state = fixture[5](*args, **kwargs)
        if args[2:4] == ("status", "live"):
            state.update(change)
        return state

    result = M.pipeline(*fixture[:4], waiting=fixture[4], sending=sending)
    return fixture, result


@pytest.mark.parametrize("live_code", [137, 0])
def test_reaped_live_oom_gets_exactly_one_validation_and_is_never_success(
    monkeypatch, live_code,
):
    fixture, result = run_with_live_state(
        monkeypatch, {"oom_killed": True}, live_code=live_code,
    )
    assert result["status"] == "offline_validation_finished"
    assert result["live_exit_code"] == live_code
    assert result["live_success"] is False
    assert result["validation_success"] is True
    assert result["paid_runs_started"] == result["validation_runs_started"] == 1
    assert M.exit_status(result) == 1
    assert [(c[0], c[1]) for c in fixture[6] if c[0] != "wait"] == [
        ("create", "live"), ("start", "live"), ("status", "live"),
        ("create", "validation"), ("start", "validation"), ("status", "validation"),
    ]
    assert [c[2] for c in fixture[6] if c[0] == "wait"] == [10860, 32460, 3600]
    assert fixture[7]["intent.json"]["validation_runs_allowed"] == 1
    before = list(fixture[6])
    with pytest.raises(FileExistsError):
        M.pipeline(*fixture[:4], waiting=fixture[4], sending=fixture[5])
    assert fixture[6] == before


@pytest.mark.parametrize("change", [
    {"status": "running", "oom_killed": True},
    {"status": "dead", "oom_killed": True},
    {"pid": 42, "oom_killed": True},
    {"exit_code": 0, "oom_killed": True},
    *[{"oom_killed": value} for value in (None, 0, 1, "true", [], {})],
])
def test_ambiguous_or_malformed_live_state_never_dispatches_validation(monkeypatch, change):
    fixture, result = run_with_live_state(monkeypatch, change)
    assert result["status"] == "operator_pending"
    assert result["operator_inspection_required"] is True
    assert result["automatic_retry"] is False
    assert result["validation_runs_started"] == 0
    assert not any(c[0] != "wait" and c[1] == "validation" for c in fixture[6])
    assert M.exit_status(result) == 1


def test_live_status_dispatch_ambiguity_never_dispatches_validation(monkeypatch):
    fixture = FIXTURE.setup(monkeypatch)

    def sending(*args, **kwargs):
        if args[2:4] == ("status", "live"):
            raise RuntimeError("dispatch_ambiguous")
        return fixture[5](*args, **kwargs)

    result = M.pipeline(*fixture[:4], waiting=fixture[4], sending=sending)
    assert result["status"] == "operator_pending"
    assert result["validation_runs_started"] == 0
    assert not any(c[0] != "wait" and c[1] == "validation" for c in fixture[6])


@pytest.mark.parametrize("phase", ["suite", "validation"])
def test_other_terminal_phases_still_reject_oom(monkeypatch, phase):
    fixture = FIXTURE.setup(monkeypatch)
    if phase == "suite":
        fixture[2].inspect = lambda *args: {
            "status": "exited", "pid": 0, "exit_code": 0, "oom_killed": True,
        }

    def sending(*args, **kwargs):
        state = fixture[5](*args, **kwargs)
        if phase == "validation" and args[2:4] == ("status", "validation"):
            state["oom_killed"] = True
        return state

    result = M.pipeline(*fixture[:4], waiting=fixture[4], sending=sending)
    assert result["status"] == "operator_pending"
    assert M.exit_status(result) == 1
    if phase == "suite":
        assert not any(c[0] != "wait" for c in fixture[6])


def test_default_terminal_gate_still_requires_no_oom():
    state = {"status": "exited", "pid": 0, "exit_code": 0, "oom_killed": True}
    with pytest.raises(RuntimeError):
        M.terminal(state, 0)
