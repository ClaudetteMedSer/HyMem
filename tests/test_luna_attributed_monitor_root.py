"""Independent root checks of the attributed observer's privacy boundary."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "root_attributed_monitor", ROOT / "tools/diagnostics/luna_subscription_profiled_v2_progress.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_all_declared_transport_failure_atoms_survive_reader():
    # Importing this source is model-free; the pinned base is loaded in isolation.
    spec = importlib.util.spec_from_file_location(
        "root_attributed_warm", ROOT / "benchmarks/codex_subscription_warm_v2.py")
    warm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warm)
    assert warm._FIXED_CODES | warm._OWN_BUDGET_CODES <= monitor.FIXED_FAILURE_CODES
    for prefix in ("process_exit", "protocol_failure", "rpc_failure"):
        for method in warm._RPC_METHODS:
            assert monitor.fixed_code(prefix + ":" + method) == prefix + ":" + method
    for method in warm._EVENT_METHODS:
        assert monitor.fixed_code("unexpected_notification:" + method) is not None


@pytest.mark.parametrize("field,value", [
    ("process_index", True), ("request_index", -1),
    ("process_age_seconds", float("inf")), ("queue_count", "private text"),
    ("known_tokens", 10**13), ("known_usage", 1),
    ("rpc", "private/method"), ("phase", "private_phase"),
])
def test_first_failure_invalid_fields_are_not_exported(field, value):
    failure = {"code": "timeout", "phase": "run", "rpc": "turn/start",
        "process_index": 1, "request_index": 2, "retired_count": 0,
        "queue_count": 0, "known_tokens": 100, "process_age_seconds": 10,
        "turn_admitted": True, "known_usage": False, "usage_complete": False}
    assert monitor.first_failure_summary(failure) == failure
    assert monitor.first_failure_summary({**failure, field: value}) is None


def test_inflight_progress_is_not_complete_or_healthy():
    result = monitor.progress_summary({"budget": {"turns": 100,
        "known_tokens": 10000, "in_flight": 4}, "active_invocations": 4,
        "usage_complete_now": True, "questions": [{"question_started": True}] * 4})
    assert result["usage_complete"] is False
    assert all(q["correct"] is None and q["indexing_healthy"] is None
               and q["summary_healthy"] is None for q in result["questions"])


def test_validated_terminal_still_requires_independent_cleanup():
    unit = {"available": True, "active_state": "active", "sub_state": "exited",
        "result": "success", "main_pid": 0, "restarts": 0, "exit_status": 0,
        "exit_code_kind": 1, "policy_ok": True, "cgroup_header_matches": False,
        "cgroup_header_empty": True, "expected_cgroup_processes": 0}
    assert monitor.completed_and_clean({"validated": True}, unit)
    for changes in ({"expected_cgroup_processes": 1}, {"expected_cgroup_processes": None},
                    {"policy_ok": False}, {"main_pid": 42}, {"exit_status": 1}):
        assert not monitor.completed_and_clean({"validated": True}, {**unit, **changes})
