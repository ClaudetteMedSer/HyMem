"""One-question, source-bound application-fault attribution probe.

The host owns containment and launch. This module has no paid-run entry point.
Only fixed finite metadata may leave its private output directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Callable

SCHEMA = "luna-application-fault-probe-v1"
RUNNER_SHA = "7f96f2ac53039805d8324055edcc0902d7210195e075300eca1b0fb961764f82"
CAPTURE_SHA = "a4436c0187852b8f342d166bd6fcb245092c29ead127891753ce263f02ed1e18"
KEY = "q-0001"
PHASES = frozenset({"client_setup", "adapter_open", "ingest", "dream", "result_validation"})
STATUSES = frozenset({"application_fault", "transport_stop", "budget_stop", "provider_denial",
                      "resource_stop", "inconclusive", "unverified"})
STOPS = frozenset({"none", "application_fault", "capture_checkpoint_failure", "capture_origin_invalid",
                   "resource_task_denial", "resource_observer_unverified", "wall_limit",
                   "campaign_wall_limit", "campaign_budget_exhausted", "question_budget_exhausted",
                   "budget_exhausted_before_turn", "pilot_budget_exhausted", "quota_exhausted",
                   "quota_floor", "subscription_auth_required", "subscription_plan_unverified",
                   "admission_rejected", "quota_unverified", "transport_failure", "timeout",
                   "turn_failed", "usage_unknown", "stage_accounting_failure", "cleanup_failure",
                   "unknown_stop", "unverified_result"})
DENIALS = frozenset({"quota_exhausted", "quota_floor", "subscription_auth_required",
                     "subscription_plan_unverified", "admission_rejected", "quota_unverified"})
BUDGET_STOPS = frozenset({"wall_limit", "campaign_wall_limit", "campaign_budget_exhausted",
                          "question_budget_exhausted", "budget_exhausted_before_turn",
                          "pilot_budget_exhausted"})
RESOURCE_STOPS = frozenset({"resource_task_denial", "resource_observer_unverified"})
STAGES = ("extraction", "grounding_original_initial", "grounding_original_recheck",
          "grounding_alternatives_initial", "grounding_alternatives_recheck", "digest", "profile", "facts")
COUNTS = ("attempts", "returned", "turns", "known_tokens")


def _pinned(module: Any, path: Path, sha: str) -> bool:
    try:
        return (type(getattr(module, "__file__", None)) is str
                and Path(module.__file__).resolve() == path.resolve()
                and stat.S_ISREG(path.lstat().st_mode)
                and hashlib.sha256(path.read_bytes()).hexdigest() == sha)
    except (OSError, ValueError):
        return False


def verify_sources(loaded: dict, runner: Any, capture: Any) -> None:
    if type(loaded) is not dict or type(loaded.get("root")) is not Path:
        raise ValueError("probe_source_invalid")
    root = loaded["root"]
    if (not root.is_absolute() or root.is_symlink()
            or loaded.get("candidate") != root / "candidate"
            or loaded.get("code") != root / "code"
            or not _pinned(runner, root / "code/tools/diagnostics/luna_lme_diagnostic_v8.py", RUNNER_SHA)
            or not _pinned(capture, root / "code/tools/diagnostics/luna_application_fault_capture_v1.py", CAPTURE_SHA)
            or not callable(getattr(runner, "make_dual", None))
            or not callable(getattr(capture, "validate_snapshot", None))):
        raise ValueError("probe_source_invalid")


def select_question(loaded: dict) -> dict:
    questions = loaded.get("questions")
    if type(questions) is not list or len(questions) != 4 or any(type(q) is not dict for q in questions):
        raise ValueError("probe_selection_invalid")
    selected = questions[1]
    for key in ("question", "haystack_sessions", "haystack_session_ids", "haystack_dates"):
        if key not in selected:
            raise ValueError("probe_selection_invalid")
    if (type(selected["question"]) is not str or not selected["question"]
            or type(selected["haystack_sessions"]) is not list
            or type(selected["haystack_session_ids"]) is not list
            or type(selected["haystack_dates"]) is not list
            or not (len(selected["haystack_sessions"]) == len(selected["haystack_session_ids"])
                    == len(selected["haystack_dates"]))):
        raise ValueError("probe_selection_invalid")
    return selected


def _uint(value: Any, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _resource(value: Any) -> dict[str, int]:
    if (type(value) is not dict or set(value) != {"current", "peak", "limit", "denials"}
            or not all(_uint(value[k], 1_000_000) for k in value)
            or value["limit"] != 256 or value["current"] > value["peak"]
            or value["peak"] > 256):
        raise ValueError("resource_observer_unverified")
    return dict(value)


def _write_once(path: Path, value: dict) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > 16_384:
        raise ValueError("probe_size_invalid")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_result(value: Any, capture: Any) -> dict:
    fields = {"schema", "status", "stop", "phase", "first_snapshot", "turns", "known_tokens",
              "usage_complete", "in_flight", "reserved", "stages", "resource", "checkpoint_durable",
              "adapter_cleanup_ok", "client_cleanup_ok", "accounting_reconciled"}
    if type(value) is not dict or set(value) != fields or value["schema"] != SCHEMA:
        raise ValueError("probe_result_invalid")
    if (type(value["status"]) is not str or value["status"] not in STATUSES
            or type(value["stop"]) is not str or value["stop"] not in STOPS
            or (value["phase"] is not None and (type(value["phase"]) is not str
                                                   or value["phase"] not in PHASES))
            or any(type(value[k]) is not bool for k in ("usage_complete", "checkpoint_durable",
                "adapter_cleanup_ok", "client_cleanup_ok", "accounting_reconciled"))
            or not _uint(value["turns"], 12) or not _uint(value["known_tokens"], 10**12)
            or not _uint(value["in_flight"], 1) or not _uint(value["reserved"], 1)):
        raise ValueError("probe_result_invalid")
    snap = capture.validate_snapshot(value["first_snapshot"])
    if snap is None:
        raise ValueError("probe_snapshot_invalid")
    stages = value["stages"]
    if type(stages) is not dict or set(stages) != set(STAGES):
        raise ValueError("probe_stages_invalid")
    for item in stages.values():
        if type(item) is not dict or set(item) != set(COUNTS) or any(not _uint(item[k], 10**12) for k in COUNTS):
            raise ValueError("probe_stages_invalid")
        if item["returned"] > item["attempts"] or item["turns"] > item["attempts"]:
            raise ValueError("probe_stages_invalid")
    resource = value["resource"]
    if resource is not None:
        _resource(resource)
    if value["status"] == "application_fault" and snap["first"] is None:
        raise ValueError("probe_result_invalid")
    if value["checkpoint_durable"] and snap["first"] is None:
        raise ValueError("probe_result_invalid")
    projected = dict(value, first_snapshot=snap,
                     stages={k: dict(stages[k]) for k in STAGES},
                     resource=dict(resource) if resource is not None else None)
    if len(json.dumps(projected, sort_keys=True, allow_nan=False).encode()) > 16_384:
        raise ValueError("probe_size_invalid")
    return projected


def run_probe(loaded: dict, runner: Any, capture: Any, *, output: Path,
              containment_verified: bool, resource_check: Callable[[], dict[str, int]]) -> dict:
    """Run once after the host verifies containment and a fresh receipt.

    The supplied resource callback must perform a live verified cgroup observation.
    It is called before every admission and once at the terminal boundary.
    """
    verify_sources(loaded, runner, capture)
    question = select_question(loaded)
    if (containment_verified is not True or not callable(resource_check)
            or type(output) is not Path or not output.is_absolute() or output.exists()
            or output.parent.is_symlink()):
        raise ValueError("probe_gate_invalid")
    warm = loaded["warm"]
    limits = warm.BudgetLimits(turns=12, known_tokens=160_000, seconds=600)
    class ObservedBudget(warm.SharedBudget):
        def __init__(self):
            super().__init__(limits, max_in_flight=1)
            self.resource_last = None
            self.resource_fault = None
        def observe(self):
            try:
                sample = _resource(resource_check())
            except BaseException:
                self.resource_fault = "resource_observer_unverified"
                self.halt(self.resource_fault)
                raise warm.ConcurrentStop(self.resource_fault) from None
            self.resource_last = sample
            if sample["denials"]:
                self.resource_fault = "resource_task_denial"
                self.halt(self.resource_fault)
                raise warm.ConcurrentStop(self.resource_fault)
            return sample
        def reserve(self, key):
            self.observe()
            return super().reserve(key)
        def before_turn(self, key, admission):
            self.observe()
            return super().before_turn(key, admission)
    budget = ObservedBudget()
    budget.register(KEY, limits)
    budget.observe()
    output.mkdir(mode=0o700)
    durable = False
    def first_checkpoint(snapshot):
        nonlocal durable
        projected = capture.validate_snapshot(snapshot)
        if projected is None or projected["first"] is None:
            raise ValueError("probe_checkpoint_invalid")
        _write_once(output / "first-fault.json", projected)
        durable = True
    probe = capture.FirstApplicationFaultCapture(
        {**loaded, "runner": runner}, budget,
        intentional_types=(warm.ConcurrentStop, capture.FirstApplicationFaultStop),
        on_first=first_checkpoint)
    phase = "client_setup"
    client = adapter = accounted = None
    adapter_cleanup_ok = client_cleanup_ok = True
    terminal_error = False
    stopped = False
    try:
        with probe:
            client = runner.make_dual(loaded, budget, KEY, limits, output)
            accounted = runner.AccountedClient(client, loaded["candidate"])
            memory = runner._memory_client(loaded, accounted)
            base = loaded["prior"].old.make_adapter_class(loaded["lme"], memory)
            diagnostic = loaded["diagnostic"].make_diagnostic_adapter_class(
                loaded["lme"], loaded["protocol"], loaded["strictness"],
                loaded["summary_classifier"], base)
            adapter = diagnostic(output / "hymem.sqlite", embeddings=False,
                aggregation_nodes=False, episode_granularity=False, pipeline_model="gpt-6-luna")
            phase = "adapter_open"
            adapter.open()
            phase = "ingest"
            adapter.ingest_sessions(question["haystack_sessions"], question["haystack_session_ids"],
                question["haystack_dates"], namespace=question["question"])
            phase = "dream"
            adapter.dream_and_wait(timeout=540, max_cycles=100, require_healthy=True)
            phase = "result_validation"
            if not accounted.reconcile():
                raise ValueError("probe_accounting_invalid")
    except BaseException as exc:
        stopped = True
        if type(exc) is not capture.FirstApplicationFaultStop and type(exc) is not warm.ConcurrentStop:
            try:
                probe.record_top_level(exc, phase=phase)
            except capture.FirstApplicationFaultStop:
                pass
            if probe.snapshot()["first"] is None:
                terminal_error = True
        elif type(exc) is capture.FirstApplicationFaultStop and probe.snapshot()["first"] is None:
            terminal_error = True
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except BaseException as exc:
                adapter_cleanup_ok = False
                try:
                    probe.record_cleanup(exc)
                except BaseException:
                    terminal_error = True
        if client is not None:
            try:
                client.close()
            except BaseException as exc:
                client_cleanup_ok = False
                try:
                    probe.record_cleanup(exc)
                except BaseException:
                    terminal_error = True
    try:
        budget.observe()
    except BaseException:
        terminal_error = True
    state = budget.snapshot()
    first = probe.snapshot()
    raw_stop = state["stop_code"]
    stop = raw_stop if raw_stop in STOPS else ("none" if raw_stop is None else "unknown_stop")
    if first["first"] is not None and durable:
        status = "application_fault"
    elif stop in RESOURCE_STOPS:
        status = "resource_stop"
    elif stop in DENIALS:
        status = "provider_denial"
    elif stop in BUDGET_STOPS:
        status = "budget_stop"
    elif stop not in {"none", "unknown_stop"}:
        status = "transport_stop"
    elif not stopped or (state["turns"] == 12 or state["known_tokens"] >= 160_000):
        status = "inconclusive"
    else:
        status = "unverified"
    if (terminal_error or not adapter_cleanup_ok or not client_cleanup_ok
            or state["in_flight"] or state["reserved"] or not state["usage_complete"]
            or stop == "unknown_stop" or (first["first"] is not None and not durable)):
        status = "unverified"
    stages = {key: {field: 0 for field in COUNTS} for key in STAGES}
    if accounted is not None:
        for key, values in accounted.counts.items():
            if key not in stages or type(values) is not dict or set(values) != set(COUNTS):
                status = "unverified"
                continue
            stages[key] = dict(values)
    result = {"schema": SCHEMA, "status": status, "stop": stop,
              "phase": phase if stopped else None, "first_snapshot": first,
              "turns": state["turns"], "known_tokens": state["known_tokens"],
              "usage_complete": state["usage_complete"], "in_flight": state["in_flight"],
              "reserved": state["reserved"], "stages": stages,
              "resource": budget.resource_last, "checkpoint_durable": durable,
              "adapter_cleanup_ok": adapter_cleanup_ok, "client_cleanup_ok": client_cleanup_ok,
              "accounting_reconciled": accounted.reconcile() if accounted is not None else False}
    projected = validate_result(result, capture)
    _write_once(output / "probe-result.json", projected)
    return projected
