"""Synthetic/fault-injection checks for the offline reliability instrument."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
import socket

import pytest

from benchmarks.digest_reliability import (
    Arm, CampaignResult, Case, INLINE_STAGES, Plan, Record, REQUIRED_PIPELINE_STAGES, STAGES,
    WorkerOutcome, execute, load_plan, main, plan_report, report,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def case(index: int = 0, *, origin: str = "retained", expected: str = "accept",
         stage: str = "fidelity_verification", stratum: str | None = None) -> Case:
    return Case(sha(f"case-{index}"), sha(f"input-{index}"), origin,
                stratum or ("verifier_positive" if origin == "control" else "general"),
                expected, stage if origin == "control" else None)


def plan(*, paired: bool = False, cases: tuple[Case, ...] | None = None,
         repetitions: int = 2, max_completions: int = 112,
         max_http_attempts: int = 336) -> Plan:
    arms = (Arm("baseline", sha("base"), sha("config")),)
    if paired:
        arms += (Arm("candidate", sha("candidate"), sha("config")),)
    return Plan("paired_comparison" if paired else "diagnostic_mapping", arms,
                cases or (case(), case(1, origin="control")), repetitions,
                "deepseek-v4-flash", sha("parameters"), max_completions, max_http_attempts)


def outcome(task, *, status: str = "passed", reason: str = "schema_violation",
            failure_stage: str | None = None, **changes) -> WorkerOutcome:
    if task.case.kind == "control":
        reached = (task.case.control_stage,)
        skipped = ()
    elif status == "passed":
        reached = tuple(s for s in STAGES if s in REQUIRED_PIPELINE_STAGES)
        skipped = tuple(s for s in task.stages if s not in reached)
    else:
        reached = ("primary",)
        skipped = ()
    result = WorkerOutcome(
        task.binding_sha256, status, len(reached), len(reached), 1.0, 0.1, True, True,
        reached[-1] if status != "passed" else None,
        () if status == "passed" else (reason,), reached, skipped,
        task.case.expected_verdict if task.case.kind == "control" else None,
        status == "passed" if task.case.kind == "control" else None,
    )
    if failure_stage is not None:
        changes["failure_stage"] = failure_stage
    return replace(result, **changes)


def test_baseline_only_approved_map_has_exact_fixed_reservation():
    cases = tuple(case(i, origin="retained" if i < 4 else "synthetic") for i in range(8))
    cases += tuple(case(i, origin="control", expected="reject" if i % 2 else "accept",
                        stage="format_adjudication" if i >= 12 else "fidelity_verification")
                   for i in range(8, 16))
    p = plan(cases=cases)
    preview = plan_report(p)
    assert preview["planned"] == 32
    assert preview["pipeline_tasks"] == preview["control_tasks"] == 16
    assert preview["worst_case_completions"] == 112
    assert preview["worst_case_http_attempts"] == 336
    assert preview["fully_reserved"] is True
    result = report(execute(p, outcome))
    assert result["complete"] and result["all_tasks_passed"]
    assert result["pipeline"]["pairs"] == {"comparison": "not_applicable"}
    assert result["pipeline"]["passed"] == result["controls"]["passed"] == 16
    assert result["readiness"] == result["semantic_quality"] == "not_assessed"


def test_paired_schedule_is_deterministic_adjacent_and_balanced_per_case():
    p = plan(paired=True, repetitions=4)
    tasks = p.schedule()
    assert tasks == p.schedule()
    assert len({t.binding_sha256 for t in tasks}) == len(tasks)
    assert tuple(t.index for t in tasks) == tuple(range(len(tasks)))
    for c in p.cases:
        pairs = []
        for repetition in range(4):
            pair = [t for t in tasks if t.case == c and t.repetition == repetition]
            assert pair[1].index == pair[0].index + 1
            assert pair[0].case.input_sha256 == pair[1].case.input_sha256
            assert pair[0].model_sha256 == pair[1].model_sha256
            assert pair[0].parameters_sha256 == pair[1].parameters_sha256
            pairs.append(tuple(t.arm.name for t in pair))
        assert pairs.count(("baseline", "candidate")) == 2
        assert pairs.count(("candidate", "baseline")) == 2


@pytest.mark.parametrize("field,value", [
    ("repetitions", True), ("repetitions", 0), ("repetitions", 1.0),
    ("repetitions", 65), ("max_completions", False), ("max_completions", -1),
    ("max_http_attempts", float("inf")), ("execution_seconds", 121),
    ("cleanup_seconds", True), ("purpose", "reroll"), ("model", "secret model\n"),
    ("parameters_sha256", "z" * 64), ("version", "future"),
])
def test_plan_rejects_invalid_scalar_schema(field, value):
    with pytest.raises(ValueError):
        replace(plan(), **{field: value})


def test_paired_constraints_and_immutable_inputs():
    p = plan(paired=True)
    for changes in (
        {"repetitions": 3}, {"arms": p.arms[::-1]}, {"arms": p.arms[:1]},
        {"arms": (p.arms[0], replace(p.arms[1], config_sha256=sha("different")))},
        {"cases": [case()]}, {"cases": (case(), case())},
        {"cases": (case(), replace(case(1), input_sha256=case().input_sha256))},
    ):
        with pytest.raises(ValueError):
            replace(p, **changes)
    with pytest.raises(FrozenInstanceError):
        p.model = "different"
    with pytest.raises(FrozenInstanceError):
        p.arms[0].source_sha256 = sha("different")


@pytest.mark.parametrize("changes", [
    {"origin": "user-private"}, {"stratum": "raw secret"}, {"case_sha256": True},
    {"expected_verdict": "reject"}, {"control_stage": "fidelity_verification"},
])
def test_pipeline_case_exact_schema(changes):
    with pytest.raises(ValueError):
        replace(case(), **changes)


def test_control_stage_and_verdict_are_explicit_bound_fields():
    control = case(1, origin="control")
    with pytest.raises(ValueError):
        replace(control, control_stage=None)
    with pytest.raises(ValueError):
        replace(control, control_stage="primary")
    first = plan(cases=(control,)).schedule()[0]
    second = plan(cases=(replace(control, control_stage="format_adjudication"),)).schedule()[0]
    third = plan(cases=(replace(control, expected_verdict="reject"),)).schedule()[0]
    assert len({first.binding_sha256, second.binding_sha256, third.binding_sha256}) == 3


def test_model_parameters_source_configuration_and_evidence_are_bound():
    p = plan()
    variations = [p, replace(p, model="different-model"),
                  replace(p, parameters_sha256=sha("other-parameters")),
                  replace(p, arms=(replace(p.arms[0], source_sha256=sha("other-source")),)),
                  replace(p, arms=(replace(p.arms[0], config_sha256=sha("other-config")),)),
                  replace(p, cases=(replace(p.cases[0], input_sha256=sha("other-input")), p.cases[1]))]
    assert len({v.sha256 for v in variations}) == len(variations)
    assert len({v.schedule()[0].binding_sha256 for v in variations}) == len(variations)


def test_load_plan_roundtrip_and_order_are_exact():
    p = plan(paired=True)
    assert load_plan(json.dumps(asdict(p))) == p
    reordered = replace(p, cases=p.cases[::-1])
    assert reordered.sha256 != p.sha256


@pytest.mark.parametrize("malformed", [
    '{"purpose":"diagnostic_mapping","purpose":"paired_comparison"}',
    '{"max_completions":NaN}', '{"max_completions":Infinity}',
    '[]', 'null', '{"raw_evidence":"never accepted"}',
])
def test_load_plan_rejects_duplicate_nonfinite_and_unknown_schema(malformed):
    with pytest.raises(ValueError):
        load_plan(malformed)


def test_nested_schema_unknown_fields_and_bool_rejected():
    for where, field, value in (("root", "repetitions", True),
                                ("arm", "extra", "no"),
                                ("case", "text", "private evidence")):
        obj = asdict(plan())
        target = obj if where == "root" else obj["arms" if where == "arm" else "cases"][0]
        target[field] = value
        with pytest.raises(ValueError):
            load_plan(json.dumps(obj))


@pytest.mark.parametrize("status", ["semantic_rejection", "contract_rejection"])
def test_rejections_continue_fixed_schedule_without_hidden_retries(status):
    p = plan(paired=True)
    called = []
    def worker(task):
        called.append(task.binding_sha256)
        return outcome(task, status=status) if task.index == 0 else outcome(task)
    result = execute(p, worker)
    summary = report(result)
    assert called == [t.binding_sha256 for t in p.schedule()]
    assert len(called) == len(set(called))
    assert summary["complete"] is True
    assert summary["all_tasks_passed"] is False
    assert summary["pipeline"]["rejected"] == 1
    assert summary["pipeline"]["pairs"]["candidate_wins"] == 1


def test_wins_losses_and_two_tie_types_not_best_of_repetitions():
    p = plan(paired=True, cases=(case(), case(1)), repetitions=2)
    def worker(task):
        pass_it = ((task.case == p.cases[0] and task.repetition == 0 and task.arm.name == "candidate")
                   or (task.case == p.cases[0] and task.repetition == 1 and task.arm.name == "baseline")
                   or (task.case == p.cases[1] and task.repetition == 0))
        return outcome(task, status="passed" if pass_it else "semantic_rejection")
    pairs = report(execute(p, worker))["pipeline"]["pairs"]
    assert pairs == {"comparison": "matched_pairs", "candidate_wins": 1,
                     "baseline_wins": 1, "ties_passed": 1, "ties_rejected": 1,
                     "missing_or_unsafe": 0}


@pytest.mark.parametrize("changes,halt", [
    ({"completions": None}, "unknown_usage"),
    ({"http_attempts": None}, "unknown_usage"),
    ({"cleanup_safe": False}, "cleanup_unsafe"),
    ({"inputs_unchanged": False}, "input_changed"),
    ({"observed_binding_sha256": sha("not-the-task")}, "input_changed"),
    ({"execution_elapsed": 120.001}, "execution_deadline"),
    ({"cleanup_elapsed": 2.001}, "cleanup_deadline"),
    ({"completions": 7, "http_attempts": 7}, "budget_violation"),
    ({"completions": 2, "http_attempts": 7}, "budget_violation"),
    ({"completions": 2, "http_attempts": 1}, "budget_violation"),
    ({"completions": 0, "http_attempts": 0}, "budget_violation"),
])
def test_unsafe_worker_receipts_halt_even_when_worker_claims_pass(changes, halt):
    p = plan(paired=True)
    called = []
    def worker(task):
        called.append(task.index)
        return outcome(task, **changes)
    result = execute(p, worker)
    summary = report(result)
    assert called == [0]
    assert result.halt_reason == halt
    assert summary["complete"] is summary["all_tasks_passed"] is False
    assert summary["pipeline"]["errors"] == 1
    assert summary["pipeline"]["pairs"]["missing_or_unsafe"] == 2


def test_unknown_usage_does_not_become_zero_actual_usage():
    summary = report(execute(plan(), lambda task: outcome(task, completions=None)))
    assert summary["usage_known"] is False
    assert summary["known_completions"] == 0
    assert summary["outcomes"][0]["completions"] is None
    assert summary["outcomes"][0]["reserved_completions"] == 6


def test_exceptions_unknown_schema_and_unknown_status_halt_sanitize_and_never_retry():
    for mode in ("exception", "dict", "status"):
        calls = []
        def worker(task):
            calls.append(task.index)
            if mode == "exception":
                raise RuntimeError("credential-and-raw-evidence-must-never-be-printed")
            if mode == "dict":
                return {"status": "passed", "raw": "credential-and-raw-evidence-must-never-be-printed"}
            return outcome(task, status="credential-and-raw-evidence-must-never-be-printed")
        summary = report(execute(plan(), worker))
        assert calls == [0]
        assert summary["halt_reason"] == "worker_failure"
        assert summary["usage_known"] is False
        assert "credential-and-raw-evidence-must-never-be-printed" not in json.dumps(summary)


@pytest.mark.parametrize("changes", [
    {"completions": True}, {"http_attempts": -1}, {"execution_elapsed": float("nan")},
    {"cleanup_elapsed": float("inf")}, {"cleanup_safe": 1}, {"inputs_unchanged": 1},
    {"stages_reached": ["primary"]}, {"failure_reasons": ("unknown", "unknown")},
    {"failure_stage": "raw text"}, {"control_expectation_met": 1},
])
def test_outcome_schema_rejects_ambiguous_types_and_nonfinite_values(changes):
    with pytest.raises(ValueError):
        outcome(plan().schedule()[0], **changes)


def test_budget_reserves_full_worst_case_before_dispatch():
    p = plan(cases=(case(),), max_completions=6, max_http_attempts=18)
    calls = []
    def worker(task):
        calls.append(task.index)
        return outcome(task)
    summary = report(execute(p, worker))
    assert calls == [0]  # Three actual completions leave three, not enough to reserve six.
    assert summary["halt_reason"] == "budget_exhausted"
    assert summary["known_completions"] == 3
    assert summary["unattempted"] == 1
    assert not summary["all_tasks_passed"]


def test_http_budget_can_prevent_dispatch_before_any_call():
    p = plan(cases=(case(),), max_http_attempts=17)
    summary = report(execute(p, lambda task: pytest.fail("must not dispatch")))
    assert summary["attempted"] == 0
    assert summary["halt_reason"] == "budget_exhausted"
    assert plan_report(p)["fully_reserved"] is False


def test_infrastructure_rejection_stops_otherwise_healthy_tasks():
    result = execute(plan(), lambda task: outcome(task, status="infrastructure_error",
                                                reason="transport_failure"))
    assert result.halt_reason == "infrastructure_error"
    assert len(result.records) == 1
    assert report(result)["pipeline"]["failure_reasons"] == {"transport_failure": 1}


def test_stage_reached_censored_optional_skip_and_unobserved_are_distinct():
    p = plan(cases=(case(), case(1), case(2)), repetitions=1)
    def worker(task):
        if task.index == 0:
            return outcome(task, status="contract_rejection", reason="length_violation",
                           failure_stage="summary_compaction", completions=2, http_attempts=2,
                           stages_reached=("primary", "summary_compaction"))
        if task.index == 1:
            return outcome(task)
        return outcome(task, stages_reached=None, stages_skipped=None)
    stages = report(execute(p, worker))["pipeline"]["stages"]
    assert stages["summary_compaction"] == {"planned": 3, "reached": 1,
        "optional_skipped": 1, "censored": 0, "unobserved": 1, "failed": 1, "unattempted": 0}
    assert stages["fidelity_verification"] == {"planned": 3, "reached": 1,
        "optional_skipped": 0, "censored": 1, "unobserved": 1, "failed": 0, "unattempted": 0}


def test_multiple_findings_retained_not_claimed_as_single_root_cause():
    p = plan(cases=(case(),), repetitions=1)
    result = execute(p, lambda task: outcome(task, status="semantic_rejection",
        failure_reasons=("unsupported_claim", "omitted_content")))
    assert report(result)["pipeline"]["failure_reasons"] == {"omitted_content": 1, "unsupported_claim": 1}


@pytest.mark.parametrize("reached,skipped", [
    (("primary",), ("fidelity_verification", "format_adjudication")),
    (("fidelity_verification", "format_adjudication"), ()),
    (("primary", "format_adjudication"), ()),
    (("primary", "fidelity_verification", "summary_content_recovery", "format_adjudication"), ()),
    (("primary", "fidelity_verification", "fidelity_reverification", "format_adjudication"), ()),
])
def test_contradictory_stage_coverage_is_unsafe_not_a_semantic_result(reached, skipped):
    result = execute(plan(), lambda task: outcome(task, stages_reached=reached, stages_skipped=skipped))
    assert result.halt_reason == "invalid_outcome"
    assert report(result)["pipeline"]["errors"] == 1


def test_stage_order_duplicate_and_unreached_failure_rejected():
    task = plan().schedule()[0]
    for changes in (
        {"stages_reached": ("format_adjudication", "primary"), "stages_skipped": ()},
        {"stages_reached": ("primary", "primary"), "stages_skipped": ()},
        {"stages_reached": ("primary",), "stages_skipped": ("primary",)},
        {"status": "contract_rejection", "failure_stage": "format_adjudication"},
    ):
        with pytest.raises(ValueError):
            outcome(task, **changes)


def test_full_six_stage_path_is_allowed():
    summary = report(execute(plan(cases=(case(),)), lambda task: outcome(
        task, completions=6, http_attempts=18, stages_reached=INLINE_STAGES, stages_skipped=())))
    assert summary["all_tasks_passed"] is True


@pytest.mark.parametrize("stage", ["fidelity_verification", "format_adjudication"])
def test_negative_controls_count_correct_rejection_as_pass(stage):
    p = plan(cases=(case(1, origin="control", expected="reject", stage=stage),))
    summary = report(execute(p, outcome))
    assert summary["pipeline"]["attempted"] == 0
    assert summary["controls"]["passed"] == 2
    assert summary["controls"]["observations"] == {"matched": 2}
    assert summary["controls"]["stages"][stage]["reached"] == 2


@pytest.mark.parametrize("expected,observed,classification", [
    ("reject", "accept", "false_accept"), ("accept", "reject", "false_reject"),
    ("reject", "reject", "collateral_mismatch"), ("accept", "uncertain", "uncertain"),
    ("accept", "malformed", "malformed"),
])
def test_control_rejections_are_classified_and_continue(expected, observed, classification):
    p = plan(cases=(case(1, origin="control", expected=expected),))
    result = execute(p, lambda task: outcome(task, status="contract_rejection",
        observed_verdict=observed, control_expectation_met=False))
    summary = report(result)
    assert summary["complete"]
    assert summary["controls"]["rejected"] == 2
    assert summary["controls"]["observations"] == {classification: 2}


def test_worker_cannot_claim_control_pass_without_matching_full_expectation():
    p = plan(cases=(case(1, origin="control", expected="reject"),))
    for changes in ({"observed_verdict": "accept"}, {"control_expectation_met": False},
                    {"observed_verdict": None}, {"control_expectation_met": None}):
        result = execute(p, lambda task: outcome(task, **changes))
        assert result.halt_reason == "invalid_outcome"
        assert report(result)["controls"]["errors"] == 1


def test_control_completion_cap_is_one_and_other_stage_is_unsafe():
    p = plan(cases=(case(1, origin="control"),))
    result = execute(p, lambda task: outcome(task, completions=2, http_attempts=2))
    assert result.halt_reason == "budget_violation"
    result = execute(p, lambda task: outcome(task, stages_reached=("primary",)))
    assert result.halt_reason == "invalid_outcome"


def test_unsafe_receipt_does_not_establish_control_or_stage_observations():
    p = plan(cases=(case(1, origin="control"),))
    result = execute(p, lambda task: outcome(task, status="contract_rejection", cleanup_safe=False))
    summary = report(result)
    assert summary["controls"]["observations"] == {}
    assert summary["controls"]["stages"]["fidelity_verification"]["failed"] == 0
    assert summary["controls"]["stages"]["fidelity_verification"]["unobserved"] == 1
    assert summary["outcomes"][0]["failure_stage"] is None
    assert summary["outcomes"][0]["reported_failure_stage"] == "fidelity_verification"


def test_origin_stratum_and_arm_rates_are_separate_from_controls():
    p = plan(cases=(case(0, origin="retained", stratum="rolling_summary"),
                    case(1, origin="synthetic", stratum="near_limit_summary"),
                    case(2, origin="control")), repetitions=1)
    summary = report(execute(p, lambda task: outcome(task, status="contract_rejection")
                            if task.case.origin == "retained" else outcome(task)))
    assert summary["pipeline"]["attempted_pass_rate"] == 0.5
    assert summary["controls"]["attempted_pass_rate"] == 1.0
    assert {g["origin"]: g["attempted_pass_rate"] for g in summary["pipeline"]["groups"]} == {
        "retained": 0.0, "synthetic": 1.0}


def test_supplied_plan_or_worker_task_mutation_detected_without_corrupting_receipt():
    for target in ("plan", "task"):
        p = plan()
        original = p.sha256
        def worker(task):
            good = outcome(task)
            if target == "plan":
                object.__setattr__(p, "parameters_sha256", sha("mutated"))
            else:
                object.__setattr__(task.case, "input_sha256", sha("mutated"))
            return good
        result = execute(p, worker)
        assert result.halt_reason == "input_changed"
        assert result.plan.sha256 == original
        assert report(result)["all_tasks_passed"] is False


def test_report_rejects_forged_pass_reordered_receipts_and_unknown_halt():
    p = plan()
    valid = execute(p, outcome)
    record = valid.records[0]
    for forged in (
        replace(valid, records=(replace(record, outcome=None),), halt_reason="worker_failure"),
        replace(valid, records=valid.records[::-1]),
        replace(valid, halt_reason="private error text"),
        replace(valid, records=valid.records[:-1]),
        replace(valid, records=(replace(record, status="unknown"),), halt_reason="worker_failure"),
        replace(valid, records=(replace(record, reserved_completions=1),), halt_reason="worker_failure"),
    ):
        with pytest.raises(ValueError):
            report(forged)


def test_report_cannot_promote_unsafe_receipt_or_continue_after_it():
    p = plan()
    unsafe = execute(p, lambda task: outcome(task, cleanup_safe=False))
    with pytest.raises(ValueError):
        report(replace(unsafe, records=(replace(unsafe.records[0], status="passed"),)))
    valid = execute(p, outcome)
    with pytest.raises(ValueError):
        report(replace(valid, records=(unsafe.records[0], *valid.records[1:]), halt_reason="cleanup_unsafe"))


def test_report_rejects_forged_globally_overbudget_schedule():
    p = plan(cases=(case(),), max_completions=6)
    records = tuple(Record(task, outcome(task), "passed", (), 6, 18) for task in p.schedule())
    with pytest.raises(ValueError, match="worst-case budget reservation"):
        report(CampaignResult(p, records, None))


def test_report_rejects_unsubstantiated_budget_halt_and_bad_record_types():
    p = plan()
    with pytest.raises(ValueError, match="unsubstantiated"):
        report(CampaignResult(p, (), "budget_exhausted"))
    with pytest.raises(ValueError):
        report(CampaignResult(p, (None,), "worker_failure"))


@pytest.mark.parametrize("mode,attempted", [(None, None), ("pass", 4),
                                          ("reject_first", 4), ("infrastructure_first", 1)])
def test_cli_only_plans_or_runs_explicit_synthetic_simulation(tmp_path, capsys, monkeypatch, mode, attempted):
    def no_network(*args, **kwargs):
        pytest.fail("offline CLI attempted network")
    monkeypatch.setattr(socket, "socket", no_network)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(asdict(plan())))
    args = [str(path)] + (["--simulate", mode] if mode else [])
    assert main(args) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert "deepseek-v4-flash" not in json.dumps(emitted)
    if mode:
        assert emitted["mode"] == "synthetic_simulation_only"
        assert emitted["attempted"] == attempted
        assert emitted["readiness"] == "not_assessed"
    else:
        assert emitted["mode"] == "plan_only"


def test_cli_invalid_content_not_reflected(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"private":"sensitive-evidence"}')
    assert main([str(path)]) == 2
    assert json.loads(capsys.readouterr().out) == {"mode": "offline", "error": "invalid_plan"}
