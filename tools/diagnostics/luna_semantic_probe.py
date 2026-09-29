"""Offline-reviewable core for the bounded semantic diagnostic.

This module has no launcher. A host must supply pinned imports, a private
evidence writer, the retained ordinary canary evidence, and a budgeted v3
client factory. Nothing here contacts a provider on import or at preflight.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

SCHEMA = "luna-semantic-probe-v1"
CASE_SHA256 = "511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925"
CASE_SOURCE_SHA256 = "8876e94e6c5d3284d4d361f0238e0aef507288e2702b5df77ee706fd4c712d8a"
CANARY_SHA256 = "3d132415573cb46c2b15a69a56776ea69b3410b350a8fdb8ba6a1f8cc85658af"
STAGE_SHA256 = "4c654599a979e51aeb9f0985b091cd8ef38a639424f197c412da45e1f3270d30"
INVENTORY_SHA256 = "11ca4cdbba18e4b7e4b56d444062e39b789820b2f0055a32898bbd0b3a1e4664"
RETAINED_EVIDENCE_SHA256 = "a1fd2d45c4cf50bf483927b07dff5cad145ae3ce8a582f19d13dc1b11a92943c"
RETAINED_RECEIPT_SHA256 = "ec4b060b2122c742c4e0dac6d95037ae04693ee65d5866523a0f9b4a194fadfc"
RETAINED_RESULT_SHA256 = "0730f970ec291473121bde497f855bfa4e49bda2b7b108a60d7efd7e55058147"
WARM_V3_SHA256 = "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d"
MAX_NEW_TURNS = 29
MAX_KNOWN_TOKENS = 500_000
MAX_SECONDS = 1800
CONTROL_CAP = (2, 100_000, 240)
HYBRID_CAP = (3, 160_000, 600)


class ProbeStop(BaseException):
    """Finite infrastructure or integrity failure; never expose raw exceptions."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ProbeStop(code)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def verify_local(candidate: Path, inventory: Path, *, cases_module,
                 grounding_module, canary_module, stage_module) -> dict:
    """Read-only source and import preflight; host pins transport separately."""
    repo = Path(__file__).resolve().parents[2]
    require(candidate.is_absolute() and inventory.is_absolute() and
            candidate.is_dir() and not candidate.is_symlink() and
            inventory.is_file() and not inventory.is_symlink(), "input_path_invalid")
    require(digest(inventory.read_bytes()) == INVENTORY_SHA256,
            "inventory_drift")
    try:
        payload = json.loads(inventory.read_bytes())
    except (ValueError, UnicodeError):
        raise ProbeStop("inventory_parse_invalid") from None
    require(type(payload) is dict, "inventory_shape_invalid")
    mapping = payload.get("source_sha256")
    require(type(mapping) is dict and len(mapping) == 510, "inventory_shape_invalid")
    observed = set()
    for path in candidate.rglob("*"):
        relative = path.relative_to(candidate)
        require(not path.is_symlink(), "candidate_extra_or_symlink")
        if any(part in {"__pycache__", ".pytest_cache", ".git"}
               for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        require(path.is_dir() or path.is_file(),
                "candidate_extra_or_symlink")
        if path.is_file():
            observed.add(relative.as_posix())
    require(observed == set(mapping), "candidate_extra_or_missing_file")
    for name, expected in mapping.items():
        require(type(name) is str and type(expected) is str and
                len(expected) == 64 and not Path(name).is_absolute() and
                ".." not in Path(name).parts, "inventory_entry_invalid")
        path = candidate / name
        require(path.is_file() and not path.is_symlink() and
                digest(path.read_bytes()) == expected, "candidate_inventory_drift")
    require(Path(cases_module.__file__).resolve() ==
            repo / "tools/diagnostics/luna_semantic_cases.py" and
            digest(Path(cases_module.__file__).read_bytes()) == CASE_SOURCE_SHA256 and
            cases_module.suite_sha256() == CASE_SHA256, "fixture_drift")
    for module, expected, filename in ((canary_module, CANARY_SHA256,
                                        "luna_semantic_canary.py"),
                                       (stage_module, STAGE_SHA256,
                                        "luna_semantic_stage_accounting.py")):
        path = Path(module.__file__)
        require(path.resolve() == repo / "benchmarks" / filename and
                digest(path.read_bytes()) == expected,
                "helper_drift")
    for module in (grounding_module,):
        path = Path(module.__file__).resolve()
        require(path == candidate.resolve() / "hymem/extraction/grounding.py",
                "candidate_import_drift")
    fixture = cases_module.cases()
    require(all(type(source) is grounding_module.GroundingSource for case in fixture
                for source in case.sources) and
            all(type(triple) is grounding_module.Triple for case in fixture
                for triple in case.triples), "fixture_import_drift")
    stage_module.verify_candidate(candidate)
    return {"candidate_files": len(mapping), "inventory_sha256": INVENTORY_SHA256,
            "fixture_label_sha256": CASE_SHA256}


def _result(case, statuses: list[str], verdicts, *, malformed: str | None,
            recheck_status: str | None) -> dict:
    false_support = []
    false_rejection = []
    missed_recovery = []
    for i, expected in enumerate(case.expected):
        status = statuses[i] if i < len(statuses) else "malformed"
        if case.category == "reject" and status in {"supported", "replace_predicate"}:
            false_support.append(i)
        elif case.category == "supported" and status not in expected.statuses:
            false_rejection.append(i)
        elif case.category == "correction" and (status != "replace_predicate" or
              (i < len(verdicts) and verdicts[i].predicate != expected.predicate)):
            missed_recovery.append(i)
    recheck_failure = (case.category == "correction" and
                       statuses == ["replace_predicate"] and
                       not missed_recovery and recheck_status != "supported")
    if recheck_failure:
        missed_recovery.extend(range(len(case.expected)))
    if malformed is not None:
        outcome = "malformed"
    elif false_support:
        outcome = "false_support"
    elif false_rejection:
        outcome = "false_rejection"
    elif missed_recovery:
        outcome = "missed_recovery"
    elif case.category == "correction" and recheck_status != "supported":
        outcome = "recheck_failed"
    else:
        outcome = "passed"
    return {"case_id": case.case_id, "category": case.category,
            "candidate_count": len(case.triples), "initial_statuses": statuses,
            "false_support_indexes": false_support,
            "false_rejection_indexes": false_rejection,
            "missed_recovery_indexes": missed_recovery,
            "malformed_code": malformed, "recheck_status": recheck_status,
            "recheck_failed": recheck_failure,
            "outcome": outcome, "passed": outcome == "passed"}


def run_control(case, client, grounding, *, record: Callable[[dict], None]) -> dict:
    """Exactly one initial verdict; recheck only a correctly proposed correction."""
    request, batch = grounding.build_grounding_request(case.triples, case.sources)
    require(type(request.user) is str, "request_invalid")
    record({"phase": "initial_before_dispatch", "request": asdict(request),
            "batch_sha256": batch.batch_sha256})
    raw = client.complete(request)
    record({"phase": "initial_returned", "response": raw})
    try:
        review = grounding.parse_grounding_response(raw, batch)
    except grounding.GroundingContractError as exc:
        result = _result(case, [], (), malformed=exc.code, recheck_status=None)
        result["new_calls"] = 1
        return result
    statuses = [v.status for v in review.verdicts]
    exact_correction = (case.category == "correction" and
        all(v.status == "replace_predicate" and v.predicate == e.predicate
            for v, e in zip(review.verdicts, case.expected)))
    recheck_status = None
    malformed = None
    new_calls = 1
    if exact_correction:
        corrected = tuple(replace(triple, predicate=verdict.predicate)
                          for triple, verdict in zip(case.triples, review.verdicts))
        request2, batch2 = grounding.build_grounding_request(corrected, case.sources)
        require(batch2.batch_sha256 != batch.batch_sha256, "correction_binding_invalid")
        record({"phase": "recheck_before_dispatch", "request": asdict(request2),
                "batch_sha256": batch2.batch_sha256})
        raw2 = client.complete(request2)
        new_calls = 2
        record({"phase": "recheck_returned", "response": raw2})
        try:
            rechecked = grounding.parse_grounding_response(raw2, batch2,
                                                             allow_corrections=False)
            recheck_status = ("supported" if rechecked.all_supported else
                              ",".join(v.status for v in rechecked.verdicts))
        except grounding.GroundingContractError as exc:
            malformed = exc.code
    result = _result(case, statuses, review.verdicts, malformed=malformed,
                     recheck_status=recheck_status)
    result["new_calls"] = new_calls
    return result


def verify_retained_evidence(raw: bytes) -> tuple[tuple[dict, str], ...]:
    require(digest(raw) == RETAINED_EVIDENCE_SHA256, "retained_evidence_drift")
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ProbeStop("retained_evidence_shape_invalid") from None
    require(type(obj) is dict and type(obj.get("requests")) is list and
            type(obj.get("responses")) is list and
            len(obj["requests"]) == len(obj["responses"]) == 8 and
            obj.get("provider_output_truncations") == 0 and
            all(type(r) is dict and type(a) is str for r, a in
                zip(obj["requests"], obj["responses"])),
            "retained_evidence_shape_invalid")
    return tuple(zip(obj["requests"], obj["responses"]))


def verify_retained_bundle(*, evidence: bytes, original_receipt: bytes,
                           original_result: bytes,
                           evidence_sha256_from_new_receipt: str):
    require(digest(original_receipt) == RETAINED_RECEIPT_SHA256 and
            digest(original_result) == RETAINED_RESULT_SHA256 and
            evidence_sha256_from_new_receipt == RETAINED_EVIDENCE_SHA256,
            "retained_provenance_drift")
    return verify_retained_evidence(evidence)


class HybridReplayClient:
    """Eight exact ordinary replays; at most three paid grounding completions."""

    def __init__(self, paid, retained: tuple[tuple[dict, str], ...], *, record,
                 source_ids: tuple[int, int]):
        require(len(retained) == 8, "retained_count_invalid")
        self.paid, self.retained, self.record = paid, retained, record
        self.source_ids = source_ids
        self.replayed = 0
        self.new_calls = 0
        self.grounding_phases: list[str] = []
        self.internal_http_attempts = None

    @property
    def observed_turns(self):
        # A failed but admitted grounding turn still consumed the paid budget.
        return self.replayed + self.paid.observed_turns

    @property
    def observed_tokens(self):
        return self.paid.observed_tokens

    @property
    def usage_complete(self):
        return self.paid.usage_complete

    @property
    def budget(self):
        # The stage wrapper must halt the owned paid ledger if its own
        # classification/accounting fails, including during hybrid replay.
        return self.paid.budget

    def complete(self, request):
        try:
            wire = json.loads(request.user)
        except (TypeError, ValueError):
            wire = None
        is_grounding = type(wire) is dict and set(wire) == {"batch", "batch_sha256"}
        if not is_grounding:
            require(self.replayed < 8, "unexpected_extra_ordinary")
            expected, response = self.retained[self.replayed]
            require(canonical(asdict(request)) == canonical(expected),
                    "ordinary_request_mismatch")
            self.record({"phase": "ordinary_replay", "index": self.replayed,
                         "request_sha256": digest(canonical(expected)),
                         "response_sha256": digest(response.encode("utf-8"))})
            self.replayed += 1
            return response
        require(self.new_calls < 3, "grounding_call_limit")
        require(type(wire["batch"]) is dict and
                type(wire["batch"].get("candidates")) is list,
                "grounding_shape_invalid")
        candidate = wire["batch"]["candidates"]
        require(type(candidate) is list and len(candidate) == 1 and
                type(candidate[0]) is dict, "grounding_shape_invalid")
        wanted_id = self.source_ids[0] if self.new_calls == 0 else self.source_ids[1]
        require(candidate[0].get("source_message_id") == wanted_id,
                "grounding_order_invalid")
        phase = ("table_initial" if self.new_calls == 0 else
                 "prose_initial" if self.new_calls == 1 else "prose_recheck")
        expected_replay = 4 if self.new_calls == 0 else 8
        require(self.replayed == expected_replay, "hybrid_sequence_invalid")
        self.record({"phase": phase + "_before_dispatch", "request": asdict(request),
                     "request_sha256": digest(canonical(asdict(request)))})
        answer = self.paid.complete(request)
        self.record({"phase": phase + "_returned", "response": answer})
        self.new_calls += 1
        self.grounding_phases.append(phase)
        return answer


def run_hybrid(*, canary_module, chunk_module, semantic_module, candidate: Path,
               paid, retained: tuple[tuple[dict, str], ...], record,
               stage_module=None) -> dict:
    expected = canary_module._CANARY_EXPECTED_CLAIMS
    require(len(expected) == 2, "canary_expected_invalid")
    hybrid = HybridReplayClient(paid, retained, record=record,
                                source_ids=(expected[0][6], expected[1][6]))
    ledger = stage_module.StageLedger(candidate) if stage_module is not None else None
    logical = ledger.wrap(hybrid, "canary") if ledger is not None else hybrid
    report = semantic_module.run_canary(canary_module, chunk_module, logical,
                                        candidate=candidate, evidence=record)
    require(hybrid.new_calls <= 3 and report["ordinary_calls"] == hybrid.replayed and
            report["grounding_initial_calls"] <= 2 and
            report["grounding_recheck_calls"] <= 1 and
            report["completion_calls"] == hybrid.replayed + hybrid.new_calls and
            report["observed_turn_delta"] == hybrid.replayed + paid.observed_turns and
            paid.observed_turns == hybrid.new_calls,
            "hybrid_accounting_invalid")
    stage_ok = ledger.reconcile_canary(report) if ledger is not None else None
    require(stage_ok is not False or not report["passed"],
            "hybrid_stage_accounting_invalid")
    complete = hybrid.replayed == 8 and hybrid.new_calls == 3
    return {"canary": report, "replayed_ordinary_calls": hybrid.replayed,
            "new_paid_grounding_calls": hybrid.new_calls,
            "paid_known_tokens": paid.observed_tokens,
            "usage_complete": paid.usage_complete,
            "grounding_phases": hybrid.grounding_phases,
            "logical_stage_reconciled": stage_ok,
            "hybrid_schedule_complete": complete,
            "passed": report["passed"] and complete and
                      report["corrected_claim_indexes"] == [1] and
                      paid.usage_complete}


class PrivateJournal:
    """Append immutable private records, with the request durable before dispatch."""

    def __init__(self, directory: Path):
        require(directory.is_absolute() and directory.is_dir() and
                not directory.is_symlink() and (directory.stat().st_mode & 0o077) == 0 and
                not any(directory.iterdir()), "private_directory_invalid")
        self.directory = directory
        self.sequence = 0

    def record(self, unit: str, value: dict) -> None:
        require(type(unit) is str and unit.isascii() and
                all(c.isalnum() or c in "-_" for c in unit) and
                type(value) is dict, "private_record_invalid")
        try:
            self.sequence += 1
            path = self.directory / f"{self.sequence:04d}-{unit}.json"
            raw = canonical(value)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            raise ProbeStop("private_evidence_write_failure") from None


def run_campaign(*, concurrent, warm, binary: str, cases_module,
                 grounding_module, journal: PrivateJournal, retained,
                 canary_module, chunk_module, semantic_module, stage_module,
                 candidate: Path, client_factory=None) -> dict:
    """Callable one-worker core. The host owns preflight, launch and export."""
    cases = cases_module.cases()
    require(cases_module.suite_sha256() == CASE_SHA256 and len(cases) == 24 and
            len({c.case_id for c in cases}) == 24 and len(retained) == 8,
            "schedule_invalid")
    if client_factory is None:
        require(Path(warm.__file__).name == "codex_subscription_warm_v3.py" and
                digest(Path(warm.__file__).read_bytes()) == WARM_V3_SHA256 and
                concurrent is warm.concurrent,
                "warm_transport_drift")
    limits_type = warm.BudgetLimits if client_factory is None else concurrent.BudgetLimits
    budget_type = warm.SharedBudget if client_factory is None else concurrent.SharedBudget
    budget = budget_type(
        limits_type(MAX_NEW_TURNS, MAX_KNOWN_TOKENS, MAX_SECONDS),
        max_in_flight=1)
    outcomes = []
    hybrid_result = None
    cleanup_ok = True
    stop_code = None
    factory = client_factory or (lambda key, cap, budget: warm.WarmSubscriptionClient(
        binary, budget, key, limits_type(*cap),
        max_requests=16, max_age_seconds=300))
    for index in range(25):
        if stop_code is not None:
            break
        is_hybrid = index == 24
        case = None if is_hybrid else cases[index]
        key = "canary" if is_hybrid else f"control-{index:02d}"
        cap = HYBRID_CAP if is_hybrid else (CONTROL_CAP if case.category == "correction"
                                             else (1, CONTROL_CAP[1], CONTROL_CAP[2]))
        client = None
        before = budget.snapshot()
        unit_started = time.monotonic()
        item = None
        try:
            require(not before["stopped"] and before["usage_complete"] and
                    before["reserved"] == before["in_flight"] == 0,
                    "preunit_accounting_invalid")
            client = factory(key, cap, budget)
            recorder = lambda value: journal.record(key, value)
            if is_hybrid:
                hybrid_result = run_hybrid(canary_module=canary_module,
                    chunk_module=chunk_module, semantic_module=semantic_module,
                    candidate=candidate, paid=client, retained=retained,
                    record=recorder, stage_module=stage_module)
            else:
                item = run_control(case, client, grounding_module, record=recorder)
                outcomes.append(item)
            state = budget.snapshot()
            expected_turns = (hybrid_result["new_paid_grounding_calls"] if is_hybrid
                              else item["new_calls"])
            require(state["questions"][key]["turns"] == expected_turns and
                    client.observed_turns == expected_turns and
                    state["reserved"] == state["in_flight"] == 0 and
                    state["usage_complete"] and client.usage_complete and
                    not state["stopped"], "postunit_accounting_invalid")
        except ProbeStop as exc:
            stop_code = exc.code
        except BaseException:
            stop_code = "infrastructure_or_runtime_failure"
        finally:
            if client is not None:
                try:
                    client.close()
                except BaseException:
                    cleanup_ok = False
                    stop_code = "cleanup_failure"
            after = budget.snapshot()
            target = hybrid_result if is_hybrid else item
            if target is not None:
                unit_state = after["questions"].get(key, {})
                target.update({
                    "new_admitted_turns": unit_state.get("turns"),
                    "known_tokens": unit_state.get("known_tokens"),
                    "usage_complete": unit_state.get("usage_complete"),
                    "elapsed_seconds": round(time.monotonic() - unit_started, 3),
                    "client_cleanup_ok": client is not None and stop_code != "cleanup_failure",
                })
            if stop_code is not None:
                budget.halt(stop_code)
            try:
                journal.record(key, {"phase": "unit_finished", "stop_code": stop_code,
                                     "budget": budget.snapshot()})
            except BaseException:
                stop_code = "private_evidence_write_failure"
                budget.halt(stop_code)
    state = budget.snapshot()
    first_failure = (warm.serialize_failure(state.get("first_failure"))
                     if callable(getattr(warm, "serialize_failure", None)) else None)
    public_budget = {name: state[name] for name in
                     ("turns", "known_tokens", "usage_complete", "in_flight", "reserved")}
    return {"schema": SCHEMA, "control_results": outcomes,
            "hybrid": hybrid_result,
            "false_support_claims": sum(len(x["false_support_indexes"]) for x in outcomes),
            "false_rejections": sum(len(x["false_rejection_indexes"]) for x in outcomes),
            "missed_recoveries": sum(len(x["missed_recovery_indexes"]) for x in outcomes),
            "recheck_failures": sum(x["recheck_failed"] for x in outcomes),
            "malformed_units": sum(x["malformed_code"] is not None for x in outcomes),
            "completed_units": len(outcomes) + (hybrid_result is not None),
            "paid_budget": public_budget, "first_failure": first_failure,
            "client_cleanup_ok": cleanup_ok,
            "stop_code": stop_code or ("transport_or_budget_stop" if state["stopped"] else None),
            "core_completed": len(outcomes) == 24 and hybrid_result is not None and
                hybrid_result["hybrid_schedule_complete"] and
                cleanup_ok and stop_code is None and not state["stopped"] and
                state["usage_complete"] and state["reserved"] == state["in_flight"] == 0,
            "completed_and_clean": False, "process_cleanup_verified": False,
            "all_semantic_checks_passed": len(outcomes) == 24 and
                all(x["passed"] for x in outcomes) and hybrid_result is not None and
                hybrid_result["passed"]}
