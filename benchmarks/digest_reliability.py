"""Offline planner and pure accounting for bounded digest reliability campaigns.

There is deliberately no provider client, live CLI, credential reader, process
launcher, persistence/resume mechanism, or retry loop here. ``execute`` needs a
TRUSTED worker that independently authenticates the supplied bindings, enforces
the absolute 120-second execution deadline, kills/reaps within two more seconds,
and reports usage. A Python callback is NOT a sandbox or a process supervisor;
this module cannot interrupt one that hangs or stop it spending outside its cap.

Model/parameter, source/config and case input digests are preregistered. Cases
contain no evidence text. Reports contain only fixed labels, counts and hashes;
exceptions and arbitrary worker data are never reflected into reports. Passing
this instrument's contracts is not evidence of semantic quality or LME readiness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable


VERSION = "digest-reliability-campaign-v2"
PIPELINE_CONTRACTS = frozenset({"inline-diagnostics-v1", "separate-diagnosis-v1"})
ORIGINS = frozenset({"retained", "synthetic", "control"})
STRATA = frozenset({
    "general", "retained_incident", "short_prose", "long_prose", "table",
    "boundary_context", "near_limit_summary", "dense_items", "sparse_items",
    "rolling_summary", "verifier_positive", "verifier_negative",
})
STAGES = (
    "primary", "summary_compaction", "fidelity_verification",
    "summary_diagnosis", "summary_content_recovery", "fidelity_reverification", "format_adjudication",
)
INLINE_STAGES = tuple(stage for stage in STAGES if stage != "summary_diagnosis")
REQUIRED_PIPELINE_STAGES = frozenset({"primary", "fidelity_verification", "format_adjudication"})
STATUSES = frozenset({
    "passed", "semantic_rejection", "contract_rejection", "infrastructure_error",
})
REJECTIONS = frozenset({"semantic_rejection", "contract_rejection"})
REASONS = frozenset({
    "length_violation", "malformed_json", "schema_violation", "unsupported_claim",
    "omitted_content", "verifier_rejection", "control_mismatch", "transport_failure",
    "timeout", "worker_failure", "unknown_usage", "cleanup_unsafe", "input_changed",
    "budget_violation", "invalid_outcome", "execution_deadline", "cleanup_deadline",
    "unknown",
    "false_accept", "false_reject", "uncertain_verdict", "collateral_mismatch",
})
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MODEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,127}\Z")


def _enum(value: object, choices: object, field: str) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError(f"invalid {field}")


def _sha(value: object, field: str) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"invalid {field}")


def _integer(value: object, field: str, minimum: int = 0, maximum: int = 100_000) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {field}")


def _seconds(value: object, field: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {field}")


def _labels(value: object, choices: object, field: str) -> None:
    if type(value) is not tuple:
        raise ValueError(f"invalid {field}")
    for label in value:
        _enum(label, choices, field)
    if len(set(value)) != len(value):
        raise ValueError(f"duplicate {field}")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    source_sha256: str
    config_sha256: str
    # Offline constructors retain the old six-call contract unless explicitly
    # changed. Serialized plans MUST name both v2 and each arm's contract.
    pipeline_contract: str = "inline-diagnostics-v1"

    def __post_init__(self) -> None:
        _enum(self.name, {"baseline", "candidate"}, "arm")
        _sha(self.source_sha256, "source digest")
        _sha(self.config_sha256, "configuration digest")
        _enum(self.pipeline_contract, PIPELINE_CONTRACTS, "pipeline contract")


@dataclass(frozen=True, slots=True)
class Case:
    case_sha256: str
    input_sha256: str
    origin: str
    stratum: str
    expected_verdict: str = "accept"
    control_stage: str | None = None

    def __post_init__(self) -> None:
        _sha(self.case_sha256, "case digest")
        _sha(self.input_sha256, "input digest")
        _enum(self.origin, ORIGINS, "origin")
        _enum(self.stratum, STRATA, "stratum")
        _enum(self.expected_verdict, {"accept", "reject"}, "expected verdict")
        if self.origin != "control" and self.expected_verdict != "accept":
            raise ValueError("pipeline cases must expect acceptance")
        if self.origin == "control":
            _enum(self.control_stage, {"fidelity_verification", "format_adjudication"},
                  "control stage")
        elif self.control_stage is not None:
            raise ValueError("pipeline cannot specify control stage")

    @property
    def kind(self) -> str:
        return "control" if self.origin == "control" else "pipeline"

@dataclass(frozen=True, slots=True)
class Plan:
    purpose: str
    arms: tuple[Arm, ...]
    cases: tuple[Case, ...]
    repetitions: int
    model: str
    parameters_sha256: str
    max_completions: int
    max_http_attempts: int
    execution_seconds: int = 120
    cleanup_seconds: int = 2
    version: str = VERSION

    def __post_init__(self) -> None:
        _enum(self.version, {VERSION}, "plan version")
        _enum(self.purpose, {"diagnostic_mapping", "paired_comparison"}, "purpose")
        if type(self.arms) is not tuple or type(self.cases) is not tuple:
            raise ValueError("arms and cases must be immutable tuples")
        if not self.cases or len(self.cases) > 128:
            raise ValueError("invalid case count")
        for arm in self.arms:
            if type(arm) is not Arm:
                raise ValueError("invalid arm schema")
            arm.__post_init__()
        for case in self.cases:
            if type(case) is not Case:
                raise ValueError("invalid case schema")
            case.__post_init__()
        if len({case.case_sha256 for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate case")
        if len({case.input_sha256 for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate case input")
        names = tuple(arm.name for arm in self.arms)
        expected = ("baseline",) if self.purpose == "diagnostic_mapping" else ("baseline", "candidate")
        if names != expected:
            raise ValueError("arm order/count does not match purpose")
        if len({arm.config_sha256 for arm in self.arms}) != 1:
            raise ValueError("paired arms must use identical configuration")
        if len({arm.source_sha256 for arm in self.arms}) != len(self.arms):
            raise ValueError("paired arms must use distinct source identities")
        _integer(self.repetitions, "repetitions", 1, 64)
        if self.purpose == "paired_comparison" and self.repetitions % 2:
            raise ValueError("paired comparisons require even repetitions")
        if len(self.cases) * len(self.arms) * self.repetitions > 4096:
            raise ValueError("too many scheduled tasks")
        if type(self.model) is not str or _MODEL.fullmatch(self.model) is None:
            raise ValueError("invalid model")
        _sha(self.parameters_sha256, "parameters digest")
        _integer(self.max_completions, "completion budget", 1, 28_672)
        _integer(self.max_http_attempts, "HTTP budget", 1, 86_016)
        if type(self.execution_seconds) is not int or self.execution_seconds != 120:
            raise ValueError("execution deadline must be 120 seconds")
        if type(self.cleanup_seconds) is not int or self.cleanup_seconds != 2:
            raise ValueError("cleanup deadline must be 2 seconds")

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))

    @property
    def model_sha256(self) -> str:
        return _digest(self.model)

    def schedule(self) -> tuple[Task, ...]:
        """Adjacent matched pairs, equal AB/BA per case; never select best draws."""
        self.__post_init__()
        result: list[Task] = []
        plan_sha256 = self.sha256
        for repetition in range(self.repetitions):
            # Rotate the preregistered case order to distribute temporal position.
            offset = repetition % len(self.cases)
            for index in (*range(offset, len(self.cases)), *range(offset)):
                case = self.cases[index]
                arms = self.arms if (repetition + index) % 2 == 0 else self.arms[::-1]
                for arm in arms:
                    result.append(Task(
                        len(result), repetition, arm, case, self.model_sha256,
                        self.parameters_sha256, plan_sha256,
                    ))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class Task:
    index: int
    repetition: int
    arm: Arm
    case: Case
    model_sha256: str
    parameters_sha256: str
    plan_sha256: str
    execution_seconds: int = 120
    cleanup_seconds: int = 2

    def __post_init__(self) -> None:
        _integer(self.index, "task index", 0, 4095)
        _integer(self.repetition, "task repetition", 0, 63)
        if type(self.arm) is not Arm or type(self.case) is not Case:
            raise ValueError("invalid task arm or case")
        self.arm.__post_init__()
        self.case.__post_init__()
        for name in ("model_sha256", "parameters_sha256", "plan_sha256"):
            _sha(getattr(self, name), name)
        if (type(self.execution_seconds) is not int or self.execution_seconds != 120
                or type(self.cleanup_seconds) is not int or self.cleanup_seconds != 2):
            raise ValueError("task deadline contract changed")

    @property
    def binding_sha256(self) -> str:
        """Worker must attest independently observed identities, not echo blindly."""
        return _digest(asdict(self))

    @property
    def completion_cap(self) -> int:
        if self.case.kind == "control":
            return 1
        return 7 if self.arm.pipeline_contract == "separate-diagnosis-v1" else 6

    @property
    def http_cap(self) -> int:
        return self.completion_cap * 3

    @property
    def stages(self) -> tuple[str, ...]:
        if self.case.kind == "control":
            return (self.case.control_stage,)
        return STAGES if self.arm.pipeline_contract == "separate-diagnosis-v1" else INLINE_STAGES


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """An authenticated worker observation, not inferred runtime control flow.

    ``stages_reached`` lists dispatched logical completions in their actual
    order, once per stage. Local payload construction, validation or input-cap
    checks are not dispatched stages and cannot increase this list or usage.
    A rejection before the next dispatch therefore has ``failure_stage=None``;
    the uncalled stage remains censored, not reached or optionally skipped.
    When present on a model rejection, ``failure_stage`` names the terminal
    (last dispatched) completion, never an earlier call in the trajectory.
    """

    observed_binding_sha256: str
    status: str
    completions: int | None
    http_attempts: int | None
    execution_elapsed: float
    cleanup_elapsed: float
    cleanup_safe: bool
    inputs_unchanged: bool
    failure_stage: str | None = None
    failure_reasons: tuple[str, ...] = ()
    stages_reached: tuple[str, ...] | None = None
    stages_skipped: tuple[str, ...] | None = None
    observed_verdict: str | None = None
    control_expectation_met: bool | None = None

    def __post_init__(self) -> None:
        _sha(self.observed_binding_sha256, "observed binding")
        _enum(self.status, STATUSES, "worker status")
        for name in ("completions", "http_attempts"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name)
        _seconds(self.execution_elapsed, "execution elapsed")
        _seconds(self.cleanup_elapsed, "cleanup elapsed")
        if type(self.cleanup_safe) is not bool or type(self.inputs_unchanged) is not bool:
            raise ValueError("invalid worker safety flags")
        if self.observed_verdict is not None:
            _enum(self.observed_verdict, {"accept", "reject", "uncertain", "malformed"},
                  "observed verdict")
        if self.control_expectation_met is not None and type(self.control_expectation_met) is not bool:
            raise ValueError("invalid control expectation flag")
        if self.failure_stage is not None:
            _enum(self.failure_stage, STAGES, "failure stage")
        _labels(self.failure_reasons, REASONS, "failure reasons")
        if self.status == "passed":
            if self.failure_stage is not None or self.failure_reasons:
                raise ValueError("passing outcome cannot include failure findings")
        elif not self.failure_reasons:
            raise ValueError("nonpassing outcome requires a failure reason")
        if self.stages_reached is None:
            if self.stages_skipped is not None:
                raise ValueError("stage coverage must be supplied together")
        else:
            _labels(self.stages_reached, STAGES, "stages reached")
            _labels(self.stages_skipped, STAGES, "stages skipped")
            if set(self.stages_reached) & set(self.stages_skipped):
                raise ValueError("stage both reached and skipped")
            if self.stages_reached != tuple(s for s in STAGES if s in self.stages_reached):
                raise ValueError("impossible stage order")
            if self.failure_stage is not None and self.failure_stage not in self.stages_reached:
                raise ValueError("failure stage was not reached")


@dataclass(frozen=True, slots=True)
class Record:
    task: Task
    outcome: WorkerOutcome | None
    status: str
    safety_reasons: tuple[str, ...]
    reserved_completions: int
    reserved_http_attempts: int


@dataclass(frozen=True, slots=True)
class CampaignResult:
    plan: Plan
    records: tuple[Record, ...]
    halt_reason: str | None


def _safety_findings(task: Task, outcome: WorkerOutcome) -> tuple[str, ...]:
    findings: list[str] = []
    if outcome.observed_binding_sha256 != task.binding_sha256 or not outcome.inputs_unchanged:
        findings.append("input_changed")
    if not outcome.cleanup_safe:
        findings.append("cleanup_unsafe")
    if outcome.execution_elapsed > task.execution_seconds:
        findings.append("execution_deadline")
    if outcome.cleanup_elapsed > task.cleanup_seconds:
        findings.append("cleanup_deadline")
    if outcome.completions is None or outcome.http_attempts is None:
        findings.append("unknown_usage")
    elif (
        outcome.completions > task.completion_cap
        or outcome.http_attempts > task.http_cap
        or outcome.http_attempts > outcome.completions * 3
        or outcome.http_attempts < outcome.completions
        or (outcome.status != "infrastructure_error" and outcome.completions == 0)
    ):
        findings.append("budget_violation")
    if outcome.stages_reached is not None and (
        not set(outcome.stages_reached).issubset(task.stages)
        or not set(outcome.stages_skipped).issubset(task.stages)
        or (outcome.status != "infrastructure_error"
            and outcome.completions != len(outcome.stages_reached))
    ):
        findings.append("invalid_outcome")
    if outcome.stages_reached is not None and task.case.kind == "pipeline":
        reached = set(outcome.stages_reached)
        if (set(outcome.stages_skipped) & REQUIRED_PIPELINE_STAGES
                or (outcome.status == "passed" and not REQUIRED_PIPELINE_STAGES <= reached)
                or (reached - {"primary"} and "primary" not in reached)
                or ({"summary_diagnosis", "summary_content_recovery", "fidelity_reverification", "format_adjudication"} & reached
                    and "fidelity_verification" not in reached)
                or ("fidelity_reverification" in reached and "summary_content_recovery" not in reached)
                or ("summary_content_recovery" in reached
                    and (outcome.status == "passed" or "format_adjudication" in reached)
                    and "fidelity_reverification" not in reached)):
            findings.append("invalid_outcome")
        if task.arm.pipeline_contract == "separate-diagnosis-v1" and (
            ("summary_content_recovery" in reached and "summary_diagnosis" not in reached)
            or ("summary_diagnosis" in reached
                and (outcome.status == "passed" or "format_adjudication" in reached)
                and not {"summary_content_recovery", "fidelity_reverification"} <= reached)
        ):
            findings.append("invalid_outcome")
    if (task.case.kind == "pipeline" and task.arm.pipeline_contract == "separate-diagnosis-v1"
            and outcome.status != "infrastructure_error" and outcome.stages_reached is None):
        # Unknown coverage cannot establish that a candidate obeyed its new
        # diagnosis gate; legacy reports may still explicitly remain unobserved.
        findings.append("invalid_outcome")
    if outcome.failure_stage is not None and outcome.failure_stage not in task.stages:
        findings.append("invalid_outcome")
    if (outcome.status in REJECTIONS and outcome.failure_stage is not None
            and (not outcome.stages_reached or outcome.failure_stage != outcome.stages_reached[-1])):
        findings.append("invalid_outcome")
    if task.case.kind == "control" and outcome.status != "infrastructure_error":
        if (outcome.observed_verdict is None or outcome.control_expectation_met is None
                or (outcome.status == "passed") != outcome.control_expectation_met
                or (outcome.control_expectation_met
                    and outcome.observed_verdict != task.case.expected_verdict)):
            findings.append("invalid_outcome")
    elif task.case.kind == "pipeline" and (
        outcome.observed_verdict is not None or outcome.control_expectation_met is not None
    ):
        findings.append("invalid_outcome")
    return tuple(dict.fromkeys(findings))


def execute(plan: Plan, worker: Callable[[Task], WorkerOutcome]) -> CampaignResult:
    """Run each scheduled task once; a semantic rejection does not censor others.

    Usage reservation happens BEFORE dispatch. Unspent reservation is released
    only after known, valid accounting. Unknown/invalid receipts or callback
    exceptions halt, without retry; their actual cost is explicitly unknown.
    Independent task completion does not authorize continued work after a worker
    reports infrastructure, isolation, deadline, cleanup or accounting failures.
    """
    if type(plan) is not Plan:
        raise ValueError("invalid plan schema")
    plan.__post_init__()
    supplied_plan = plan
    plan = deepcopy(plan)
    tasks = plan.schedule()
    frozen_plan_sha256 = plan.sha256
    records: list[Record] = []
    used_completions = used_http = 0
    for task in tasks:
        if (
            used_completions + task.completion_cap > plan.max_completions
            or used_http + task.http_cap > plan.max_http_attempts
        ):
            return CampaignResult(plan, tuple(records), "budget_exhausted")
        try:
            worker_task = deepcopy(task)
            outcome = worker(worker_task)
            if type(outcome) is not WorkerOutcome:
                raise ValueError("invalid worker outcome")
            outcome.__post_init__()
        except Exception:
            # Never serialize an exception message: it may contain evidence or keys.
            records.append(Record(task, None, "infrastructure_error", ("unknown_usage",),
                                  task.completion_cap, task.http_cap))
            return CampaignResult(plan, tuple(records), "worker_failure")
        findings = _safety_findings(task, outcome)
        try:
            supplied_plan.__post_init__()
            unchanged = (supplied_plan.sha256 == frozen_plan_sha256
                         and worker_task.binding_sha256 == task.binding_sha256)
        except (ValueError, TypeError):
            unchanged = False
        if not unchanged:
            findings = tuple(dict.fromkeys((*findings, "input_changed")))
        status = "infrastructure_error" if findings else outcome.status
        records.append(Record(task, outcome, status, findings, task.completion_cap, task.http_cap))
        if status == "infrastructure_error":
            halt = findings[0] if findings else "infrastructure_error"
            return CampaignResult(plan, tuple(records), halt)
        # These are necessarily integers after the safety checks above.
        used_completions += outcome.completions
        used_http += outcome.http_attempts
    return CampaignResult(plan, tuple(records), None)


def _counts(tasks: tuple[Task, ...], records: tuple[Record, ...]) -> dict:
    passed = sum(record.status == "passed" for record in records)
    rejected = sum(record.status in REJECTIONS for record in records)
    errors = sum(record.status == "infrastructure_error" for record in records)
    return {
        "planned": len(tasks), "attempted": len(records), "passed": passed,
        "rejected": rejected, "errors": errors, "unattempted": len(tasks) - len(records),
        "attempted_pass_rate": passed / len(records) if records else None,
    }


def _stage_report(tasks: tuple[Task, ...], records: tuple[Record, ...]) -> dict:
    result = {}
    for stage in STAGES:
        applicable = tuple(task for task in tasks if stage in task.stages)
        relevant = tuple(record for record in records if stage in record.task.stages)
        reached = skipped = censored = unobserved = failed = 0
        for record in relevant:
            outcome = record.outcome
            if record.status == "infrastructure_error":
                # An unsafe receipt cannot establish stage observations or blame.
                unobserved += 1
                continue
            if outcome is None or outcome.stages_reached is None:
                unobserved += 1
            elif stage in outcome.stages_reached:
                reached += 1
            elif stage in outcome.stages_skipped:
                skipped += 1
            else:
                # Unreached is NOT a failure or proof that a conditional stage was skipped.
                censored += 1
            failed += outcome is not None and outcome.failure_stage == stage
        result[stage] = {
            "planned": len(applicable), "reached": reached, "optional_skipped": skipped,
            "censored": censored, "unobserved": unobserved, "failed": failed,
            "unattempted": len(applicable) - len(relevant),
        }
    return result


def _pairs(plan: Plan, records: tuple[Record, ...], *, kind: str) -> dict:
    if plan.purpose != "paired_comparison":
        return {"comparison": "not_applicable"}
    indexed = {(r.task.case.case_sha256, r.task.repetition, r.task.arm.name): r for r in records}
    counts = Counter({"candidate_wins": 0, "baseline_wins": 0,
                      "ties_passed": 0, "ties_rejected": 0, "missing_or_unsafe": 0})
    for case in plan.cases:
        if case.kind != kind:
            continue
        for repetition in range(plan.repetitions):
            base = indexed.get((case.case_sha256, repetition, "baseline"))
            candidate = indexed.get((case.case_sha256, repetition, "candidate"))
            if (base is None or candidate is None
                    or base.status == "infrastructure_error"
                    or candidate.status == "infrastructure_error"):
                counts["missing_or_unsafe"] += 1
            elif base.status == candidate.status == "passed":
                counts["ties_passed"] += 1
            elif base.status != "passed" and candidate.status != "passed":
                counts["ties_rejected"] += 1
            elif candidate.status == "passed":
                counts["candidate_wins"] += 1
            else:
                counts["baseline_wins"] += 1
    return {"comparison": "matched_pairs", **counts}


def _control_observation(record: Record) -> str | None:
    outcome = record.outcome
    if (record.status == "infrastructure_error" or record.task.case.kind != "control"
            or outcome is None or outcome.observed_verdict is None):
        return None
    if outcome.observed_verdict in {"uncertain", "malformed"}:
        return outcome.observed_verdict
    if outcome.observed_verdict != record.task.case.expected_verdict:
        return "false_accept" if outcome.observed_verdict == "accept" else "false_reject"
    return "matched" if outcome.control_expectation_met else "collateral_mismatch"


def report(result: CampaignResult) -> dict:
    """Contract pass rates only; controls are NEVER pooled with pipeline rates."""
    if type(result) is not CampaignResult or type(result.plan) is not Plan or type(result.records) is not tuple:
        raise ValueError("invalid result schema")
    plan, records = result.plan, result.records
    tasks = plan.schedule()
    # Results are normally produced by execute; refuse forged/reordered receipts.
    if (len(records) > len(tasks) or any(type(record) is not Record for record in records)
            or any(record.task != tasks[i] for i, record in enumerate(records))):
        raise ValueError("results do not follow the frozen schedule")
    if result.halt_reason is not None:
        _enum(result.halt_reason, REASONS | {"budget_exhausted", "infrastructure_error"}, "halt reason")
    if len(records) < len(tasks) and result.halt_reason is None:
        raise ValueError("incomplete campaign must have a halt reason")
    accounted_completions = accounted_http = 0
    for index, record in enumerate(records):
        if (accounted_completions + record.task.completion_cap > plan.max_completions
                or accounted_http + record.task.http_cap > plan.max_http_attempts):
            raise ValueError("task dispatched without worst-case budget reservation")
        _enum(record.status, STATUSES, "record status")
        _labels(record.safety_reasons, REASONS, "safety reasons")
        if (type(record.reserved_completions) is not int or type(record.reserved_http_attempts) is not int
                or record.reserved_completions != record.task.completion_cap
                or record.reserved_http_attempts != record.task.http_cap):
            raise ValueError("invalid reservation")
        if record.outcome is None:
            if record.status != "infrastructure_error":
                raise ValueError("missing worker outcome")
        else:
            if type(record.outcome) is not WorkerOutcome:
                raise ValueError("invalid outcome schema")
            record.outcome.__post_init__()
            findings = _safety_findings(record.task, record.outcome)
            expected_status = "infrastructure_error" if findings or record.safety_reasons else record.outcome.status
            if record.status != expected_status:
                raise ValueError("record contradicts worker outcome")
            accounted_completions += record.outcome.completions or 0
            accounted_http += record.outcome.http_attempts or 0
        if record.status == "infrastructure_error" and (
            index != len(records) - 1 or result.halt_reason is None
        ):
            raise ValueError("campaign continued after unsafe outcome")
    if result.halt_reason == "budget_exhausted":
        if len(records) == len(tasks):
            raise ValueError("budget exhaustion after complete schedule")
        next_task = tasks[len(records)]
        if (accounted_completions + next_task.completion_cap <= plan.max_completions
                and accounted_http + next_task.http_cap <= plan.max_http_attempts):
            raise ValueError("unsubstantiated budget exhaustion")
    sections = {}
    for kind in ("pipeline", "control"):
        kind_tasks = tuple(task for task in tasks if task.case.kind == kind)
        kind_records = tuple(record for record in records if record.task.case.kind == kind)
        groups = []
        for arm in plan.arms:
            for origin, stratum in sorted({(t.case.origin, t.case.stratum) for t in kind_tasks}):
                group_tasks = tuple(t for t in kind_tasks if (t.arm.name, t.case.origin, t.case.stratum)
                                    == (arm.name, origin, stratum))
                group_records = tuple(r for r in kind_records if r.task in group_tasks)
                groups.append({"arm": arm.name, "pipeline_contract": arm.pipeline_contract,
                               "origin": origin, "stratum": stratum,
                               **_counts(group_tasks, group_records)})
        reasons = Counter(reason for record in kind_records
                          for reason in set(record.safety_reasons) | set(
                              record.outcome.failure_reasons if record.outcome else ()))
        sections[kind] = {
            **_counts(kind_tasks, kind_records), "groups": groups,
            "failure_reasons": dict(sorted(reasons.items())),
            "stages": _stage_report(kind_tasks, kind_records),
            "stages_by_arm": {arm.name: _stage_report(
                tuple(t for t in kind_tasks if t.arm == arm),
                tuple(r for r in kind_records if r.task.arm == arm),
            ) for arm in plan.arms},
            "pairs": _pairs(plan, kind_records, kind=kind),
        }
        if kind == "control":
            sections[kind]["observations"] = dict(sorted(Counter(
                label for record in kind_records if (label := _control_observation(record))
            ).items()))
    complete = len(records) == len(tasks) and result.halt_reason is None
    all_passed = complete and all(record.status == "passed" for record in records)
    usage_known = all(record.outcome is not None
                      and record.outcome.completions is not None
                      and record.outcome.http_attempts is not None for record in records)
    return {
        "version": VERSION, "plan_sha256": plan.sha256, "purpose": plan.purpose,
        "assessment": "contract_reliability_only", "readiness": "not_assessed",
        "semantic_quality": "not_assessed", "complete": complete,
        "all_tasks_passed": all_passed, "halt_reason": result.halt_reason,
        "planned": len(tasks), "attempted": len(records), "unattempted": len(tasks) - len(records),
        "usage_known": usage_known,
        "known_completions": sum(r.outcome.completions or 0 for r in records if r.outcome),
        "known_http_attempts": sum(r.outcome.http_attempts or 0 for r in records if r.outcome),
        "max_completions": plan.max_completions, "max_http_attempts": plan.max_http_attempts,
        "pipeline": sections["pipeline"], "controls": sections["control"],
        "outcomes": [{
            "task_sha256": r.task.binding_sha256, "case_sha256": r.task.case.case_sha256,
            "arm": r.task.arm.name, "repetition": r.task.repetition,
            "pipeline_contract": r.task.arm.pipeline_contract,
            "status": r.status, "reported_status": r.outcome.status if r.outcome else None,
            "safety_reasons": list(r.safety_reasons),
            "failure_stage": r.outcome.failure_stage if r.outcome and r.status != "infrastructure_error" else None,
            "reported_failure_stage": r.outcome.failure_stage if r.outcome else None,
            "failure_reasons": list(r.outcome.failure_reasons) if r.outcome else [],
            "control_observation": _control_observation(r),
            "completions": r.outcome.completions if r.outcome else None,
            "http_attempts": r.outcome.http_attempts if r.outcome else None,
            "reserved_completions": r.reserved_completions,
            "reserved_http_attempts": r.reserved_http_attempts,
        } for r in records],
    }


def plan_report(plan: Plan) -> dict:
    tasks = plan.schedule()
    worst = sum(task.completion_cap for task in tasks)
    return {
        "version": VERSION, "mode": "plan_only", "plan_sha256": plan.sha256,
        "purpose": plan.purpose, "model_sha256": plan.model_sha256,
        "parameters_sha256": plan.parameters_sha256, "planned": len(tasks),
        "pipeline_tasks": sum(task.case.kind == "pipeline" for task in tasks),
        "control_tasks": sum(task.case.kind == "control" for task in tasks),
        "worst_case_completions": worst, "worst_case_http_attempts": 3 * worst,
        "max_completions": plan.max_completions, "max_http_attempts": plan.max_http_attempts,
        "fully_reserved": worst <= plan.max_completions and 3 * worst <= plan.max_http_attempts,
        "execution_seconds": 120, "cleanup_seconds": 2,
        "schedule": [{"task_sha256": t.binding_sha256, "case_sha256": t.case.case_sha256,
                      "arm": t.arm.name, "repetition": t.repetition,
                      "pipeline_contract": t.arm.pipeline_contract,
                      "applicable_stages": list(t.stages),
                      "completion_cap": t.completion_cap, "http_cap": t.http_cap}
                     for t in tasks],
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON value")


def _fields(value: object, required: set[str], optional: set[str] = frozenset()) -> dict:
    if type(value) is not dict or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("unexpected schema fields")
    return value


def load_plan(serialized: str) -> Plan:
    """Exact input schema; no permissive coercion or duplicate-key JSON parsing."""
    if type(serialized) is not str or len(serialized) > 1_000_000:
        raise ValueError("invalid plan encoding")
    obj = json.loads(serialized, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _fields(obj, {"version", "purpose", "arms", "cases", "repetitions", "model", "parameters_sha256",
                  "max_completions", "max_http_attempts"},
            {"execution_seconds", "cleanup_seconds"})
    if type(obj["arms"]) is not list or type(obj["cases"]) is not list:
        raise ValueError("invalid plan arrays")
    arms = tuple(Arm(**_fields(a, {"name", "source_sha256", "config_sha256", "pipeline_contract"}))
                 for a in obj["arms"])
    cases = tuple(Case(**_fields(c, {"case_sha256", "input_sha256", "origin", "stratum"},
                                {"expected_verdict", "control_stage"})) for c in obj["cases"])
    return Plan(**{**obj, "arms": arms, "cases": cases})


def _simulate(task: Task, mode: str) -> WorkerOutcome:
    status = "passed"
    reasons = ()
    if task.index == 0 and mode != "pass":
        status = "contract_rejection" if mode == "reject_first" else "infrastructure_error"
        reasons = ("schema_violation",) if mode == "reject_first" else ("worker_failure",)
    stage = task.case.control_stage if task.case.kind == "control" else "primary"
    reached = ((stage,) if task.case.kind == "control" or reasons else
               tuple(s for s in STAGES if s in REQUIRED_PIPELINE_STAGES))
    skipped = (() if reasons else tuple(s for s in task.stages if s not in reached))
    return WorkerOutcome(task.binding_sha256, status, len(reached), len(reached), 0.0, 0.0, True, True,
                         stage if reasons else None, reasons, reached, skipped,
                         task.case.expected_verdict if task.case.kind == "control" else None,
                         status == "passed" if task.case.kind == "control" else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--simulate", choices=("pass", "reject_first", "infrastructure_first"))
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan.read_text(encoding="utf-8"))
        if args.simulate:
            output = report(execute(plan, lambda task: _simulate(task, args.simulate)))
            output["mode"] = "synthetic_simulation_only"
        else:
            output = plan_report(plan)
    except (ValueError, TypeError, OSError, RecursionError):
        print(json.dumps({"mode": "offline", "error": "invalid_plan"}))
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
