"""One-question, source-bound application-fault attribution probe.

The host owns containment and launch. This module has no paid-run entry point.
Only fixed finite metadata may leave its private output directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Callable

SCHEMA = "luna-application-fault-probe-v2"
RUNNER_SHA = "b3e1135893a715dec4e138c25f6bf3c70f3912dbd5df81014ee7c8f2767fd278"
CAPTURE_SHA = "e865366dc7ae1fc5cb72367c7f3d59c3c3d3227486728e7035b6e61242a44977"
KEY = "q-0001"
PHASES = frozenset({"client_setup", "adapter_open", "ingest", "dream", "result_validation"})
STATUSES = frozenset({"application_fault", "transport_stop", "provider_denial",
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


def verify_sources(loaded: dict, runner: Any, capture: Any, binary_sha256: str) -> None:
    if type(loaded) is not dict or not isinstance(loaded.get("root"), Path):
        raise ValueError("probe_source_invalid")
    root = loaded["root"]
    if (not root.is_absolute() or root.is_symlink()
            or loaded.get("source_only") is not False
            or loaded.get("candidate") != root / "candidate"
            or loaded.get("code") != root / "code"
            or not _pinned(runner, root / "code/tools/diagnostics/luna_lme_diagnostic_v9.py", RUNNER_SHA)
            or not _pinned(capture, root / "code/tools/diagnostics/luna_application_fault_capture_v2.py", CAPTURE_SHA)
            or not callable(getattr(runner, "make_dual", None))
            or not callable(getattr(capture, "validate_snapshot", None))):
        raise ValueError("probe_source_invalid")
    dataset, binary = loaded.get("dataset"), loaded.get("binary")
    if (type(binary_sha256) is not str or len(binary_sha256) != 64
            or any(char not in "0123456789abcdef" for char in binary_sha256)
            or not isinstance(dataset, Path) or not isinstance(binary, Path)
            or not dataset.is_absolute() or dataset.is_symlink() or not dataset.is_file()
            or hashlib.sha256(dataset.read_bytes()).hexdigest() != runner.DATASET_SHA256
            or not binary.is_absolute() or binary.is_symlink() or not binary.is_file()
            or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_sha256
            or not _pinned(loaded.get("lme"), root / "candidate/benchmarks/longmemeval_adapter.py",
                           runner.CANDIDATE_PINS["benchmarks/longmemeval_adapter.py"])):
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
    selected_again = list(loaded["prior"].SelectedQuestions(
        loaded["dataset"], 4, loaded["protocol"]))[1]
    if selected != selected_again:
        raise ValueError("probe_selection_drift")
    return selected


def selected_row_sha256(question: dict) -> str:
    """Digest the verified source row without exporting its identifiers or text."""
    raw = json.dumps(question, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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
    temporary = path.with_name(path.name + ".pending")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    os.link(temporary, path, follow_symlinks=False)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    os.unlink(temporary)


def validate_result(value: Any, capture: Any) -> dict:
    fields = {"schema", "status", "stop", "phase", "selected_row_sha256", "first_snapshot", "turns", "known_tokens",
              "usage_complete", "in_flight", "reserved", "stages", "resource", "checkpoint_durable",
              "adapter_cleanup_ok", "client_cleanup_ok", "accounting_reconciled"}
    if type(value) is not dict or set(value) != fields or value["schema"] != SCHEMA:
        raise ValueError("probe_result_invalid")
    if (type(value["selected_row_sha256"]) is not str
            or len(value["selected_row_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in value["selected_row_sha256"])
            or type(value["status"]) is not str or value["status"] not in STATUSES
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
        if not item["attempts"] >= item["turns"] >= item["returned"]:
            raise ValueError("probe_stages_invalid")
    stage_reconciled = (sum(item["turns"] for item in stages.values()) == value["turns"]
                        and sum(item["known_tokens"] for item in stages.values()) == value["known_tokens"])
    if value["accounting_reconciled"] and not stage_reconciled:
        raise ValueError("probe_accounting_invalid")
    if (value["turns"] == 0 and value["known_tokens"] != 0) or (
            value["turns"] > 0 and value["usage_complete"] and
            value["known_tokens"] < value["turns"]):
        raise ValueError("probe_usage_invalid")
    resource = value["resource"]
    if resource is not None:
        _resource(resource)
    if value["status"] != "unverified":
        if (not value["usage_complete"] or value["in_flight"] or value["reserved"]
                or not value["adapter_cleanup_ok"] or not value["client_cleanup_ok"]
                or not value["accounting_reconciled"] or resource is None
                or resource["denials"] or value["stop"] == "unknown_stop"):
            raise ValueError("probe_result_invalid")
    if value["status"] == "application_fault" and snap["first"] is None:
        raise ValueError("probe_result_invalid")
    if value["status"] == "application_fault" and value["stop"] != "application_fault":
        raise ValueError("probe_result_invalid")
    if value["checkpoint_durable"] and snap["first"] is None:
        raise ValueError("probe_result_invalid")
    if (snap["first"] is not None and not value["checkpoint_durable"]
            and value["status"] != "unverified"):
        raise ValueError("probe_result_invalid")
    if (value["status"] == "inconclusive" and (snap["first"] is not None
            or value["stop"] not in BUDGET_STOPS | {"none"})):
        raise ValueError("probe_result_invalid")
    if (value["status"] == "provider_denial" and
            (value["stop"] not in DENIALS or snap["first"] is not None)):
        raise ValueError("probe_result_invalid")
    if (value["status"] == "resource_stop" and
            (value["stop"] not in RESOURCE_STOPS or snap["first"] is not None)):
        raise ValueError("probe_result_invalid")
    if (value["status"] == "transport_stop" and
            (value["stop"] in DENIALS | RESOURCE_STOPS | BUDGET_STOPS |
             {"none", "application_fault", "unknown_stop"} or snap["first"] is not None)):
        raise ValueError("probe_result_invalid")
    if (value["status"] == "application_fault" and
            (not value["checkpoint_durable"] or not value["adapter_cleanup_ok"]
             or not value["client_cleanup_ok"] or not value["usage_complete"]
             or not value["accounting_reconciled"])):
        raise ValueError("probe_result_invalid")
    projected = dict(value, first_snapshot=snap,
                     stages={k: dict(stages[k]) for k in STAGES},
                     resource=dict(resource) if resource is not None else None)
    if len(json.dumps(projected, sort_keys=True, allow_nan=False).encode()) > 16_384:
        raise ValueError("probe_size_invalid")
    return projected


def run_probe(loaded: dict, runner: Any, capture: Any, *, output: Path,
              containment_verified: bool, binary_sha256: str,
              resource_check: Callable[[], dict[str, int]]) -> dict:
    """Run once after the host verifies containment and a fresh receipt.

    The supplied resource callback must perform a live verified cgroup observation.
    It is called before every admission and once at the terminal boundary.
    """
    verify_sources(loaded, runner, capture, binary_sha256)
    question = select_question(loaded)
    row_sha = selected_row_sha256(question)
    if (containment_verified is not True or not callable(resource_check)
            or not isinstance(output, Path) or not output.is_absolute() or output.exists()
            or any(parent.is_symlink() for parent in output.parents)
            or not output.parent.is_dir()):
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
                prior = self.resource_last
                if prior is not None and (sample["peak"] < prior["peak"]
                                          or sample["denials"] < prior["denials"]):
                    raise ValueError("resource_observer_unverified")
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
    budget.observe()
    output.mkdir(mode=0o700)
    deadline = time.monotonic() + 600
    durable = False
    def first_checkpoint(snapshot):
        nonlocal durable
        projected = capture.validate_snapshot(snapshot)
        if projected is None or projected["first"] is None:
            raise ValueError("probe_checkpoint_invalid")
        _write_once(output / "first-fault.json", projected)
        durable = True
        try:
            budget.observe()
        except BaseException:
            pass  # The durable first fault survives a later resource observation fault.
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
            if time.monotonic() >= deadline:
                budget.halt("campaign_wall_limit")
                raise warm.ConcurrentStop("campaign_wall_limit")
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
            if time.monotonic() >= deadline:
                budget.halt("campaign_wall_limit")
                raise warm.ConcurrentStop("campaign_wall_limit")
            adapter.open()
            phase = "ingest"
            if time.monotonic() >= deadline:
                budget.halt("campaign_wall_limit")
                raise warm.ConcurrentStop("campaign_wall_limit")
            adapter.ingest_sessions(question["haystack_sessions"], question["haystack_session_ids"],
                question["haystack_dates"], namespace=question["question"])
            phase = "dream"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                budget.halt("campaign_wall_limit")
                raise warm.ConcurrentStop("campaign_wall_limit")
            adapter.dream_and_wait(timeout=min(540, remaining), max_cycles=100, require_healthy=True)
            phase = "result_validation"
            if not accounted.reconcile():
                raise ValueError("probe_accounting_invalid")
    except BaseException as exc:
        stopped = True
        if (type(exc) is not capture.FirstApplicationFaultStop
                and type(exc) is not warm.ConcurrentStop and not budget.snapshot()["stopped"]):
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
        status = "inconclusive"
    elif stop not in {"none", "unknown_stop"}:
        status = "transport_stop"
    elif not stopped or (state["turns"] == 12 or state["known_tokens"] >= 160_000):
        status = "inconclusive"
    else:
        status = "unverified"
    if (terminal_error or budget.resource_fault is not None
            or not adapter_cleanup_ok or not client_cleanup_ok
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
    stage_reconciled = (sum(item["turns"] for item in stages.values()) == state["turns"]
                        and sum(item["known_tokens"] for item in stages.values()) == state["known_tokens"])
    if not stage_reconciled:
        status = "unverified"
    result = {"schema": SCHEMA, "status": status, "stop": stop,
              "selected_row_sha256": row_sha,
              "phase": phase if stopped else None, "first_snapshot": first,
              "turns": state["turns"], "known_tokens": state["known_tokens"],
              "usage_complete": state["usage_complete"], "in_flight": state["in_flight"],
              "reserved": state["reserved"], "stages": stages,
              "resource": budget.resource_last, "checkpoint_durable": durable,
              "adapter_cleanup_ok": adapter_cleanup_ok, "client_cleanup_ok": client_cleanup_ok,
              "accounting_reconciled": (accounted.reconcile() and stage_reconciled
                  if accounted is not None else stage_reconciled)}
    projected = validate_result(result, capture)
    _write_once(output / "probe-result.json", projected)
    return projected
