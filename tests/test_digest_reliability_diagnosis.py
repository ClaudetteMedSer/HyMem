"""Arm-bound six/seven-call accounting; entirely synthetic and offline."""
from dataclasses import asdict, replace
import json

import pytest

from benchmarks.digest_reliability import (
    Arm, CampaignResult, INLINE_STAGES, Record, STAGES, VERSION,
    execute, load_plan, plan_report, report,
)
from tests.test_digest_reliability import case, outcome, plan, sha


def paired(**kwargs):
    original = plan(paired=True, **kwargs)
    return replace(original, arms=(original.arms[0], replace(
        original.arms[1], pipeline_contract="separate-diagnosis-v1",
    )))


def full_path(task):
    return outcome(task, completions=len(task.stages), http_attempts=3 * len(task.stages),
                   stages_reached=task.stages, stages_skipped=())


def test_explicit_contract_controls_caps_not_arm_name_or_evidence():
    p = paired(cases=(case(),))
    base, candidate = p.schedule()[:2]
    assert (base.completion_cap, base.http_cap, base.stages) == (6, 18, INLINE_STAGES)
    assert (candidate.completion_cap, candidate.http_cap, candidate.stages) == (7, 21, STAGES)
    assert base.case == candidate.case
    assert base.arm.config_sha256 == candidate.arm.config_sha256
    assert base.model_sha256 == candidate.model_sha256
    assert base.parameters_sha256 == candidate.parameters_sha256
    assert base.arm.source_sha256 != candidate.arm.source_sha256
    assert not hasattr(base.case, "completion_cap") and not hasattr(base.case, "stages")
    legacy = replace(candidate, arm=replace(candidate.arm, pipeline_contract="inline-diagnostics-v1"))
    assert legacy.completion_cap == 6 and legacy.http_cap == 18
    assert legacy.binding_sha256 != candidate.binding_sha256


def test_full_comparison_reservations_include_candidate_diagnosis_not_controls():
    cases = tuple(case(i, origin="retained" if i < 4 else "synthetic") for i in range(8))
    cases += tuple(case(i, origin="control", stage="format_adjudication" if i % 2 else "fidelity_verification")
                   for i in range(8, 16))
    p = paired(cases=cases, max_completions=240, max_http_attempts=720)
    preview = plan_report(p)
    assert preview["version"] == VERSION == "digest-reliability-campaign-v2"
    assert preview["planned"] == 64
    assert preview["pipeline_tasks"] == preview["control_tasks"] == 32
    assert preview["worst_case_completions"] == 240  # 16*6 + 16*7 + 32*1.
    assert preview["worst_case_http_attempts"] == 720
    assert preview["fully_reserved"]
    for task, entry in zip(p.schedule(), preview["schedule"]):
        assert entry["pipeline_contract"] == task.arm.pipeline_contract
        assert entry["applicable_stages"] == list(task.stages)
    result = report(execute(p, full_path))
    assert result["complete"] and result["all_tasks_passed"]
    assert result["known_completions"] == 240 and result["known_http_attempts"] == 720
    assert result["pipeline"]["stages_by_arm"]["baseline"]["summary_diagnosis"] == {
        "planned": 0, "reached": 0, "optional_skipped": 0, "censored": 0,
        "unobserved": 0, "failed": 0, "unattempted": 0,
    }
    assert result["pipeline"]["stages_by_arm"]["candidate"]["summary_diagnosis"]["reached"] == 16
    assert result["controls"]["stages"]["summary_diagnosis"]["planned"] == 0
    assert result["readiness"] == result["semantic_quality"] == "not_assessed"


@pytest.mark.parametrize("damage", ["old_version", "missing_version", "missing_contract", "unknown_contract",
                                   "bool_contract", "case_cap", "root_cap", "arm_cap"])
def test_serialized_old_or_implicit_six_call_plan_cannot_upgrade(damage):
    obj = asdict(paired())
    if damage == "old_version":
        obj["version"] = "digest-reliability-campaign-v1"
    elif damage == "missing_version":
        del obj["version"]
    elif damage == "missing_contract":
        del obj["arms"][1]["pipeline_contract"]
    elif damage in {"unknown_contract", "bool_contract"}:
        obj["arms"][1]["pipeline_contract"] = "future" if damage == "unknown_contract" else True
    else:
        target = obj["cases"][0] if damage == "case_cap" else obj["arms"][1] if damage == "arm_cap" else obj
        target["completion_cap"] = 7
    with pytest.raises(ValueError):
        load_plan(json.dumps(obj))


def test_constructor_legacy_default_is_explicit_when_serialized():
    arm = Arm("baseline", sha("source"), sha("config"))
    assert arm.pipeline_contract == "inline-diagnostics-v1"
    original = plan()
    assert load_plan(json.dumps(asdict(original))) == original
    with pytest.raises(ValueError):
        replace(original, version="digest-reliability-campaign-v1")


def test_paired_source_identity_cannot_be_same_under_different_contract_labels():
    p = paired()
    with pytest.raises(ValueError, match="distinct source"):
        replace(p, arms=(p.arms[0], replace(p.arms[1], source_sha256=p.arms[0].source_sha256)))


@pytest.mark.parametrize("completion_budget,http_budget", [(12, 39), (13, 38), (6, 18)])
def test_mixed_arm_worst_case_reservation_precedes_candidate_dispatch(completion_budget, http_budget):
    p = paired(cases=(case(),), max_completions=completion_budget, max_http_attempts=http_budget)
    called = []
    def worker(task):
        called.append(task.arm.name)
        return full_path(task)
    result = report(execute(p, worker))
    assert called == ["baseline"]
    assert result["halt_reason"] == "budget_exhausted"
    assert result["known_completions"] == 6 and result["known_http_attempts"] == 18
    assert result["unattempted"] == 3


@pytest.mark.parametrize("reached", [
    ("primary", "fidelity_verification", "summary_content_recovery", "fidelity_reverification", "format_adjudication"),
    ("primary", "fidelity_verification", "summary_diagnosis", "fidelity_reverification", "format_adjudication"),
    ("primary", "fidelity_verification", "summary_diagnosis", "summary_content_recovery", "format_adjudication"),
    ("primary", "fidelity_verification", "summary_diagnosis", "format_adjudication"),
    ("primary", "summary_diagnosis", "summary_content_recovery", "fidelity_reverification", "format_adjudication"),
    ("primary", "fidelity_verification", "summary_diagnosis"),
])
def test_candidate_cannot_successfully_bypass_diagnosis_repair_or_reverification(reached):
    p = paired(cases=(case(),))
    def worker(task):
        if task.arm.name == "baseline":
            return outcome(task)
        return outcome(task, completions=len(reached), http_attempts=len(reached),
                       stages_reached=reached, stages_skipped=())
    result = execute(p, worker)
    assert result.halt_reason == "invalid_outcome" and len(result.records) == 2
    assert report(result)["pipeline"]["errors"] == 1


def test_candidate_unobserved_stage_coverage_cannot_claim_success():
    result = execute(paired(cases=(case(),)), lambda task: outcome(task) if task.arm.name == "baseline"
                     else outcome(task, stages_reached=None, stages_skipped=None))
    assert result.halt_reason == "invalid_outcome"


def test_candidate_without_veto_can_skip_all_repair_stages_and_format_normally():
    result = report(execute(paired(cases=(case(),)), outcome))
    assert result["all_tasks_passed"]
    candidate = result["pipeline"]["stages_by_arm"]["candidate"]
    assert candidate["summary_diagnosis"]["optional_skipped"] == 2
    assert candidate["summary_diagnosis"]["censored"] == 0


def test_diagnosis_failure_is_model_result_and_downstream_stages_are_censored():
    p = paired(cases=(case(),))
    reached = ("primary", "fidelity_verification", "summary_diagnosis")
    def worker(task):
        return outcome(task) if task.arm.name == "baseline" else outcome(
            task, status="contract_rejection", failure_stage="summary_diagnosis",
            completions=3, http_attempts=3, stages_reached=reached, stages_skipped=("summary_compaction",))
    result = report(execute(p, worker))
    assert result["complete"] and result["pipeline"]["rejected"] == 2
    candidate = result["pipeline"]["stages_by_arm"]["candidate"]
    assert candidate["summary_diagnosis"]["failed"] == 2
    assert candidate["summary_content_recovery"]["censored"] == 2
    assert candidate["format_adjudication"]["censored"] == 2


def test_predispatch_diagnosis_input_cap_is_rejection_not_virtual_dispatch_or_safety_halt():
    p = paired(cases=(case(),))
    def worker(task):
        if task.arm.name == "baseline":
            return outcome(task)
        return replace(outcome(
            task, status="contract_rejection", reason="length_violation",
            completions=2, http_attempts=2,
            stages_reached=("primary", "fidelity_verification"),
            stages_skipped=("summary_compaction",),
        ), failure_stage=None)
    result = report(execute(p, worker))
    assert result["complete"] and result["pipeline"]["rejected"] == 2
    assert result["pipeline"]["errors"] == 0 and result["halt_reason"] is None
    candidate = result["pipeline"]["stages_by_arm"]["candidate"]
    assert candidate["summary_diagnosis"] == {
        "planned": 2, "reached": 0, "optional_skipped": 0, "censored": 2,
        "unobserved": 0, "failed": 0, "unattempted": 0,
    }
    assert all(row["failure_stage"] is None for row in result["outcomes"])


@pytest.mark.parametrize("status", ["contract_rejection", "semantic_rejection"])
@pytest.mark.parametrize("failure_stage", STAGES[:-1])
def test_model_rejection_cannot_blame_earlier_stage_after_full_candidate_dispatch(status, failure_stage):
    p = paired(cases=(case(),))
    def worker(task):
        if task.arm.name == "baseline":
            return outcome(task)
        return outcome(task, status=status, failure_stage=failure_stage,
                       completions=7, http_attempts=7, stages_reached=STAGES, stages_skipped=())
    result = execute(p, worker)
    assert result.halt_reason == "invalid_outcome" and len(result.records) == 2
    reviewed = report(result)
    assert reviewed["pipeline"]["errors"] == 1
    assert reviewed["outcomes"][-1]["failure_stage"] is None
    assert reviewed["outcomes"][-1]["reported_failure_stage"] == failure_stage


def test_rejection_with_last_dispatched_stage_remains_ordinary_model_result():
    p = paired(cases=(case(),))
    result = report(execute(p, lambda task: outcome(
        task, status="contract_rejection", failure_stage="format_adjudication",
        completions=len(task.stages), http_attempts=len(task.stages),
        stages_reached=task.stages, stages_skipped=(),
    )))
    assert result["complete"] and result["pipeline"]["rejected"] == 4
    assert result["pipeline"]["errors"] == 0


def test_legacy_rejection_cannot_claim_terminal_stage_without_dispatch_coverage():
    result = execute(plan(cases=(case(),)), lambda task: outcome(
        task, status="contract_rejection", stages_reached=None, stages_skipped=None,
    ))
    assert result.halt_reason == "invalid_outcome"


@pytest.mark.parametrize("changes", [
    {"stages_reached": STAGES, "completions": 6, "http_attempts": 6},
    {"stages_reached": INLINE_STAGES, "completions": 7, "http_attempts": 7},
])
def test_baseline_cannot_spend_or_claim_candidate_only_stage(changes):
    result = execute(plan(cases=(case(),)), lambda task: outcome(task, stages_skipped=(), **changes))
    assert result.halt_reason in {"invalid_outcome", "budget_violation"}


@pytest.mark.parametrize("stage", ["fidelity_verification", "format_adjudication"])
def test_controls_remain_single_call_for_both_contracts(stage):
    p = paired(cases=(case(1, origin="control", stage=stage),))
    assert all(task.completion_cap == 1 and task.http_cap == 3 and task.stages == (stage,)
               for task in p.schedule())
    assert report(execute(p, outcome))["controls"]["passed"] == 4
    result = execute(p, lambda task: outcome(task) if task.arm.name == "baseline" else
                     outcome(task, completions=2, http_attempts=2))
    assert result.halt_reason == "budget_violation"


def test_old_contract_binding_or_mutation_cannot_authorize_new_candidate():
    p = paired(cases=(case(),))
    def worker(task):
        if task.arm.name == "baseline":
            return outcome(task)
        old = replace(task, arm=replace(task.arm, pipeline_contract="inline-diagnostics-v1"))
        return replace(outcome(task), observed_binding_sha256=old.binding_sha256)
    assert execute(p, worker).halt_reason == "input_changed"
    def mutate(task):
        good = outcome(task)
        object.__setattr__(task.arm, "pipeline_contract", "separate-diagnosis-v1")
        return good
    assert execute(plan(cases=(case(),)), mutate).halt_reason == "input_changed"


def test_report_rejects_forged_candidate_six_call_reservation_or_task():
    p = paired(cases=(case(),))
    valid = execute(p, full_path)
    candidate = valid.records[1]
    for forged in (
        replace(candidate, reserved_completions=6, reserved_http_attempts=18),
        replace(candidate, task=replace(candidate.task, arm=replace(
            candidate.task.arm, pipeline_contract="inline-diagnostics-v1"))),
        replace(candidate, outcome=replace(candidate.outcome, completions=6, http_attempts=18)),
    ):
        with pytest.raises(ValueError):
            report(replace(valid, records=(valid.records[0], forged, *valid.records[2:])))


def test_report_cannot_forge_global_candidate_reservation_from_six_call_budget():
    p = paired(cases=(case(),), max_completions=12, max_http_attempts=39)
    records = tuple(Record(task, full_path(task), "passed", (), task.completion_cap, task.http_cap)
                    for task in p.schedule()[:2])
    with pytest.raises(ValueError, match="worst-case budget reservation"):
        report(CampaignResult(p, records, "budget_exhausted"))


@pytest.mark.parametrize("field,value", [("execution_seconds", 121), ("cleanup_seconds", 3),
                                        ("index", True), ("repetition", -1)])
def test_task_cannot_override_deadline_or_identity_schema(field, value):
    with pytest.raises(ValueError):
        replace(paired().schedule()[1], **{field: value})


def test_report_detects_forged_seven_stages_with_only_three_completed_calls():
    p = paired(cases=(case(),))
    result = execute(p, lambda task: outcome(task) if task.arm.name == "baseline" else
                     outcome(task, stages_reached=STAGES, stages_skipped=()))
    assert result.halt_reason == "invalid_outcome"


@pytest.mark.parametrize("contract", ["inline-diagnostics-v1", "separate-diagnosis-v1"])
def test_every_successful_stage_subset_matches_exact_contract_state_machine(contract):
    """Enumerate all 128 subsets, rather than only known omission examples."""
    original = plan(cases=(case(),), repetitions=1)
    p = replace(original, arms=(replace(original.arms[0], pipeline_contract=contract),))
    legal = set()
    for compact in (False, True):
        for repair in (False, True):
            path = ["primary"]
            if compact:
                path.append("summary_compaction")
            path.append("fidelity_verification")
            if repair:
                if contract == "separate-diagnosis-v1":
                    path.append("summary_diagnosis")
                path.extend(("summary_content_recovery", "fidelity_reverification"))
            path.append("format_adjudication")
            legal.add(tuple(path))
    accepted = set()
    for mask in range(1 << len(STAGES)):
        reached = tuple(stage for index, stage in enumerate(STAGES) if mask & (1 << index))
        result = execute(p, lambda task: outcome(
            task, completions=len(reached), http_attempts=len(reached),
            stages_reached=reached, stages_skipped=(),
        ))
        if result.halt_reason is None:
            accepted.add(reached)
        else:
            assert result.halt_reason in {"invalid_outcome", "budget_violation"}
    assert accepted == legal
