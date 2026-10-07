"""Inactive, finite staged diagnostic. A host owns provenance and export.

Model text and provider exceptions are private journal data. This module has no
launcher, network access, or production extraction hook.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Callable

from benchmarks import codex_subscription_staged_v1 as transport

staged = transport.staged
v4 = transport.classification
gate = transport.gate
SCHEMA = "luna-staged-diagnostic-v1"
CONTROL_INDICES = (9, 12, 13, 17, 19, 21)
MAX_NEW_TURNS = 29
MAX_KNOWN_TOKENS = 500_000
MAX_SECONDS = 1800
UNIT_CAP = (3, 100_000, 240)
CASE_SHA256 = "511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925"
_STATES = frozenset(("supported", "not_established", "ambiguous"))
_VERDICTS = frozenset(("supported", "unsupported", "uncertain", "replace_predicate"))
_OUTCOMES = frozenset(("accepted", "rejected", "malformed"))
_STOP_CODES = frozenset(("private_evidence_write_failure", "preunit_accounting_invalid",
    "postunit_accounting_invalid", "cleanup_failure", "infrastructure_or_runtime_failure",
    "transport_or_budget_stop", "canary_batch_binding_invalid", "schedule_invalid",
    "stage_limit", "correction_integrity_failure"))
_CONTRACT_CODES = frozenset("""alternatives:count alternatives:coverage alternatives:index
alternatives:none_required alternatives:prior_binding alternatives:required
alternatives:shape alternatives:unexpected alternatives_batch:binding alternatives_batch:type
assessment:negative_support assessment:shape assessment:state batch:binding
batch:serialization batch:type batch:unicode check:indices check:shape check:state
classification:alternatives classification:index classification:shape context:content
context:id context:parent_metadata context:parent_missing context:parent_prefix
context:parent_region context:parent_unexpected context:prefix context:region context:type
evidence:context_missing evidence:context_scope evidence:duplicate evidence:global_bounds
evidence:owned_required evidence:parent_required evidence:parent_scope evidence:quote
evidence:quote_missing evidence:region evidence:shape evidence:source original:index
original:shape request:binding response:binding response:bounds response:correction_flag
response:count response:depth response:incomplete response:schema response:shape
source:content source:contexts source:duplicate_id source:id source:legacy_scope
source:type sources:bounds sources:total_bounds support:checks support:evidence_bounds
support:shape support:unreferenced_evidence triple:numeric triple:polarity
triple:predicate triple:qualifier triple:source triple:text triple:type triples:bounds
verdict:correction verdict:evidence verdict:evidence_count verdict:index
verdict:negative_shape verdict:predicate verdict:shape verdict:status""".split())


def _contract_code(exc):
    code = getattr(exc, "code", None)
    return code if type(code) is str and code in _CONTRACT_CODES else "contract_other"


class DiagnosticStop(BaseException):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise DiagnosticStop(code)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class PrivateJournal:
    """Immutable fsynced records; the host supplies a fresh private directory."""

    def __init__(self, directory: Path):
        _require(directory.is_absolute() and directory.is_dir() and
                 not directory.is_symlink() and (directory.stat().st_mode & 0o077) == 0 and
                 not any(directory.iterdir()), "private_evidence_write_failure")
        self.directory = directory
        self.sequence = 0

    def record(self, unit: str, value: dict) -> None:
        _require(type(unit) is str and unit.isascii() and
                 all(c.isalnum() or c in "-_" for c in unit) and
                 type(value) is dict, "private_evidence_write_failure")
        try:
            self.sequence += 1
            path = self.directory / f"{self.sequence:04d}-{unit}.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(_canonical(value))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            raise DiagnosticStop("private_evidence_write_failure") from None


@dataclass(frozen=True)
class ScheduledUnit:
    kind: str
    index: int


def schedule() -> tuple[ScheduledUnit, ...]:
    return tuple(ScheduledUnit("control", i) for i in CONTROL_INDICES) + (
        ScheduledUnit("table_canary", 0), ScheduledUnit("prose_canary", 1))


def _one_source_batch(batch, source_id: int, expected: tuple, *, prose: bool) -> None:
    _require(len(batch.triples) == len(batch.sources) == 1, "canary_batch_binding_invalid")
    triple, source = batch.triples[0], batch.sources[0]
    _require(triple.source_message_id == source.source_message_id == source_id and
             (triple.subject, triple.predicate, triple.object, triple.polarity) ==
             (expected[0], "uses" if prose else expected[2], expected[3], expected[5]) and
             all(getattr(triple, field) is None for field in
                 ("value_text", "value_numeric", "value_unit", "temporal_scope")),
             "canary_batch_binding_invalid")
    v4._checked(batch)


def derive_canary_batches(*, canary_module, chunk_module, semantic_probe_module,
                          candidate: Path, retained: tuple[tuple[dict, str], ...],
                          record: Callable[[dict], None]):
    """Replay eight retained ordinary turns and capture both initial stage batches."""
    expected = canary_module._CANARY_EXPECTED_CLAIMS
    _require(len(retained) == 8 and len(expected) == 2 and
             canary_module.EXTRACTION_CANARY_MAX_COMPLETION_CALLS == 24 and
             Path(canary_module.__file__).resolve() == candidate.resolve() / "benchmarks/extraction_canary.py" and
             Path(chunk_module.__file__).resolve() == candidate.resolve() / "hymem/extraction/chunk.py",
             "canary_batch_binding_invalid")
    source_records = canary_module._source_records()
    _require(len(source_records) == 2 and
             tuple(sid for sid, _ in source_records) == (expected[0][6], expected[1][6]),
             "canary_batch_binding_invalid")

    class CapturedProse(BaseException):
        pass

    class FakePaid:
        observed_turns = 0
        observed_tokens = 0
        usage_complete = True

        def __init__(self):
            self.batches = []

        def complete_stage(self, request, batch, stage, recheck):
            _require(stage == "original" and recheck is False, "canary_batch_binding_invalid")
            staged.validate_original_request(request, batch)
            self.batches.append(batch)
            if len(self.batches) == 2:
                raise CapturedProse()
            _require(len(self.batches) == 1, "canary_batch_binding_invalid")
            quote = canary_module._TABLE_CLAIM_ROW
            _require(quote in batch.sources[0].content and len(quote) <= 192,
                     "canary_batch_binding_invalid")
            evidence = [{"source_message_id": batch.triples[0].source_message_id,
                         "region": "owned", "quote": quote}]
            check = {"state": "supported", "evidence_indices": [0]}
            raw = _canonical({"schema": staged.ORIGINAL_SCHEMA,
                "batch_sha256": batch.batch_sha256, "complete": True,
                "originals": [{"index": 0, "original": {"state": "supported",
                    "support": {"evidence": evidence, "checks": {
                        "attribution_and_roles": check, "relation_and_polarity": check}}}}]}).decode()
            staged.parse_original_response(raw, batch)
            return raw

    fake = FakePaid()
    class Hybrid(semantic_probe_module.HybridReplayClient):
        def complete_stage(self, request, batch, stage, recheck):
            wire = json.loads(request.user)
            _require(type(wire) is dict and type(wire.get("batch")) is dict and
                     type(wire["batch"].get("candidates")) is list and
                     len(wire["batch"]["candidates"]) == 1,
                     "canary_batch_binding_invalid")
            wanted = expected[len(self.grounding_phases)][6]
            phase = "table_initial" if not self.grounding_phases else "prose_initial"
            _require(len(self.grounding_phases) < 2 and
                     self.replayed == (4 if not self.grounding_phases else 8) and
                     wire["batch"]["candidates"][0].get("source_message_id") == wanted,
                     "canary_batch_binding_invalid")
            self.record({"phase": phase + "_before_dispatch", "request": asdict(request),
                         "batch_sha256": batch.batch_sha256})
            self.new_calls += 1
            self.grounding_phases.append(phase)
            return self.paid.complete_stage(request, batch, stage, recheck)

    replay = Hybrid(fake, retained, record=record,
                    source_ids=(expected[0][6], expected[1][6]))
    try:
        chunk_module.extract_chunk(replay, canary_module._CANARY_CONTENT,
            source_records=source_records,
            completion_call_limit=canary_module.EXTRACTION_CANARY_MAX_COMPLETION_CALLS)
    except CapturedProse:
        pass
    else:
        raise DiagnosticStop("canary_batch_binding_invalid")
    _require(replay.replayed == 8 and replay.new_calls == 2 and
             replay.grounding_phases == ["table_initial", "prose_initial"] and
             len(fake.batches) == 2, "canary_batch_binding_invalid")
    for index, batch in enumerate(fake.batches):
        _one_source_batch(batch, expected[index][6], expected[index], prose=index == 1)
        source = json.loads(source_records[index][1])
        _require(type(source) is dict and type(source.get("content")) is str and
                 batch.sources[0].content in source["content"] and
                 batch.sources[0].source_role == source.get("source_role") and
                 batch.sources[0].source_peer_id == source.get("source_peer_id") and
                 batch.sources[0].source_created_at == source.get("source_created_at"),
                 "canary_batch_binding_invalid")
    _require(fake.batches[0].batch_sha256 != fake.batches[1].batch_sha256,
             "canary_batch_binding_invalid")
    record({"phase": "canary_batches_derived", "table_batch_sha256": fake.batches[0].batch_sha256,
            "prose_batch_sha256": fake.batches[1].batch_sha256,
            "ordinary_replays": replay.replayed, "synthetic_table_advancement": True})
    return tuple(fake.batches)


def _stage_record(stage: str, recheck: bool, batch, states: tuple[str, ...],
                  schema_hash: str, prior_hash: str | None = None) -> dict:
    _require(all(state in _STATES for state in states), "correction_integrity_failure")
    return {"stage": stage, "recheck": recheck,
            "batch_sha256": batch.batch_sha256 if stage == "original" else batch.classification_batch.batch_sha256,
            "schema_sha256": schema_hash, "prior_response_sha256": prior_hash,
            "states": list(states), "predicate_states": []}


def _evaluation(outcome, error_code, stages, verdicts, final_predicates,
                contract_code=None):
    return {"outcome": outcome, "error_code": error_code,
            "contract_code": contract_code, "stages": stages,
            "verdicts": verdicts, "verdict_counts": {
                name: verdicts.count(name) for name in sorted(_VERDICTS)},
            "final_predicates": final_predicates}


def evaluate_unit(unit: ScheduledUnit, triples, sources, invoke: Callable) -> dict:
    """Pure staged decision over supplied responses; callback owns any I/O."""
    _require(type(unit) is ScheduledUnit and unit in schedule(), "schedule_invalid")
    stages = []
    initial = tuple(triples)
    current = initial
    for recheck in (False, True):
        request, batch = staged.build_original_request(current, sources)
        staged.validate_original_request(request, batch)
        schema_hash = _sha(_canonical(staged.build_original_output_schema(batch)))
        _require(len(stages) < 3, "stage_limit")
        raw = invoke(request, batch, "original", recheck)
        try:
            original = staged.parse_original_response(raw, batch)
        except staged.GroundingContractError as exc:
            stages.append(_stage_record("original", recheck, batch, (), schema_hash))
            return _evaluation("malformed", "original_invalid", stages, [], [], _contract_code(exc))
        stages.append(_stage_record("original", recheck, batch, original.states, schema_hash))
        if "ambiguous" in original.states:
            return _evaluation("rejected", "uncertain", stages, [], [])
        if recheck and original.negative_indices:
            return _evaluation("rejected", "unsupported", stages, [], [])
        alternative_raw = None
        if original.negative_indices:
            alt_request, alt_batch = staged.build_alternatives_request(batch, raw)
            staged.validate_alternatives_request(alt_request, alt_batch)
            schema_hash = _sha(_canonical(staged.build_alternatives_output_schema(alt_batch)))
            _require(len(stages) < 3, "stage_limit")
            alternative_raw = invoke(alt_request, alt_batch, "alternatives", False)
            stages.append(_stage_record("alternatives", False, alt_batch, (),
                                        schema_hash, alt_batch.original_response_sha256))
        try:
            review = staged.parse_staged_responses(batch, raw, alternative_raw,
                                                   allow_corrections=not recheck)
        except staged.GroundingContractError as exc:
            return _evaluation("malformed", "alternatives_or_selector_invalid", stages, [], [],
                               _contract_code(exc))
        if alternative_raw is not None:
            alt_payload = staged._load(alternative_raw)
            stages[-1]["predicate_states"] = [
                {"index": row["index"], "predicate": predicate,
                 "state": row["alternatives"][predicate]["state"]}
                for row in alt_payload["alternatives"]
                for predicate in v4.PREDICATE_ORDER
                if predicate in row["alternatives"]]
        verdicts = review.verdicts
        statuses = [v.status for v in verdicts]
        _require(all(v in _VERDICTS for v in statuses), "correction_integrity_failure")
        if any(v in {"unsupported", "uncertain"} for v in statuses):
            return _evaluation("rejected", "unsupported" if "unsupported" in statuses else "uncertain",
                               stages, statuses, [])
        if recheck or "replace_predicate" not in statuses:
            return _evaluation("accepted", None, stages, statuses,
                               [t.predicate for t in current])
        proposed = tuple(replace(t, predicate=v.predicate) if v.status == "replace_predicate" else t
                         for t, v in zip(current, verdicts, strict=True))
        try:
            gate._source_gate._check_corrections(list(current), list(proposed))
        except gate.GroundingGateError:
            return _evaluation("rejected", "correction_collision", stages, statuses, [])
        current = proposed
    raise AssertionError("unreachable")


def _unit_input(unit, cases, canary_batches):
    if unit.kind != "control":
        batch = canary_batches[unit.index]
        return batch.triples, batch.sources, batch.batch_sha256
    case = cases[unit.index]
    return case.triples, tuple(gate._v2_source(source) for source in case.sources), None


def _gold(unit, cases, evaluated):
    if evaluated["outcome"] == "malformed":
        return False
    return _gold_from_public(unit, evaluated)


def _gold_from_public(unit, evaluated):
    """Compare observed mechanics with the separately frozen expectations."""
    if evaluated["outcome"] == "malformed":
        return False
    initial = evaluated["stages"][0]["states"]
    if unit.kind == "table_canary":
        return (evaluated["outcome"] == "accepted" and initial == ["supported"] and
                evaluated["final_predicates"] == ["deploys_to"])
    if unit.kind == "prose_canary":
        return (evaluated["outcome"] == "accepted" and initial == ["not_established"] and
                len(evaluated["stages"]) == 3 and
                evaluated["stages"][-1]["states"] == ["supported"] and
                evaluated["final_predicates"] == ["prefers"])
    if unit.index == 9:
        return (evaluated["outcome"] == "accepted" and initial == ["supported"] and
                evaluated["final_predicates"] == ["prefers"])
    if unit.index in (12, 13):
        return (evaluated["outcome"] == "accepted" and initial == ["not_established"] and
                len(evaluated["stages"]) == 3 and
                evaluated["stages"][-1]["states"] == ["supported"] and
                evaluated["final_predicates"] == ["prefers" if unit.index == 12 else "uses"])
    return (evaluated["outcome"] == "rejected" and
            len(initial) == (2 if unit.index == 21 else 1) and
            all(state in {"not_established", "ambiguous"} for state in initial) and
            len(evaluated["stages"]) <= 2 and
            not any(entry["state"] == "supported" for stage in evaluated["stages"]
                    for entry in stage["predicate_states"]) and
            evaluated["error_code"] in {"unsupported", "uncertain"})


def run_campaign(*, concurrent, warm, binary: str, cases_module,
                 canary_batches, journal: PrivateJournal, client_factory=None) -> dict:
    """Run exactly eight scheduled units; every unit shares its three-turn cap."""
    cases = cases_module.cases()
    _require(cases_module.suite_sha256() == CASE_SHA256 and len(cases) == 24 and
             len({case.case_id for case in cases}) == 24 and len(canary_batches) == 2,
             "schedule_invalid")
    expected = (("HyMem Canary Relay", "deploys_to", "Fly.io", 1),
                ("Avery Boundary Canary", "uses", "PostgreSQL", 1))
    for batch, fields in zip(canary_batches, expected, strict=True):
        _require(len(batch.triples) == len(batch.sources) == 1 and
                 (batch.triples[0].subject, batch.triples[0].predicate,
                  batch.triples[0].object, batch.triples[0].polarity) == fields,
                 "schedule_invalid")
        v4._checked(batch)
    limits_type = warm.BudgetLimits if client_factory is None else concurrent.BudgetLimits
    budget_type = warm.SharedBudget if client_factory is None else concurrent.SharedBudget
    if client_factory is None:
        _require(concurrent is warm.concurrent and warm is transport.warm,
                 "schedule_invalid")
    budget = budget_type(limits_type(MAX_NEW_TURNS, MAX_KNOWN_TOKENS, MAX_SECONDS), max_in_flight=1)
    factory = client_factory or (lambda key, cap, shared: transport.StagedSubscriptionClient(
        binary, shared, key, limits_type(*cap), max_requests=16, max_age_seconds=300))
    results = []
    attempted = 0
    stop_code = None
    cleanup_ok = True
    for ordinal, unit in enumerate(schedule()):
        key = f"unit-{ordinal:02d}"
        client = None
        item = None
        before = budget.snapshot()
        started = time.monotonic()
        dispatched = 0
        try:
            _require(not before["stopped"] and before["usage_complete"] and
                     before["reserved"] == before["in_flight"] == 0,
                     "preunit_accounting_invalid")
            triples, sources, expected_hash = _unit_input(unit, cases, canary_batches)
            client = factory(key, UNIT_CAP, budget)
            attempted += 1

            def invoke(request, batch, stage, recheck):
                nonlocal dispatched
                _require(dispatched < 3, "stage_limit")
                if stage == "original":
                    staged.validate_original_request(request, batch)
                    schema = staged.build_original_output_schema(batch)
                    batch_hash = batch.batch_sha256
                    batch_json = batch.canonical_json
                    prior = None
                else:
                    staged.validate_alternatives_request(request, batch)
                    schema = staged.build_alternatives_output_schema(batch)
                    batch_hash = batch.classification_batch.batch_sha256
                    batch_json = batch.classification_batch.canonical_json
                    prior = batch.original_response_sha256
                _require(expected_hash is None or (stage != "original" or recheck or
                         batch_hash == expected_hash), "canary_batch_binding_invalid")
                journal.record(key, {"phase": "before_dispatch", "stage": stage,
                    "recheck": recheck, "request": asdict(request), "batch": batch_json,
                    "batch_sha256": batch_hash, "prior_response_sha256": prior,
                    "output_schema_sha256": _sha(_canonical(schema))})
                dispatched += 1
                raw = client.complete_stage(request, batch, stage, recheck)
                journal.record(key, {"phase": "response_returned", "stage": stage,
                                     "recheck": recheck, "response": raw})
                return raw

            evaluated = evaluate_unit(unit, triples, sources, invoke)
            journal.record(key, {"phase": "evaluated", "evaluation": evaluated})
            state = budget.snapshot()
            question = state["questions"].get(key, {})
            _require(question.get("turns") == client.observed_turns == dispatched and
                     1 <= dispatched <= 3 and state["reserved"] == state["in_flight"] == 0 and
                     state["usage_complete"] and client.usage_complete and
                     not state["stopped"], "postunit_accounting_invalid")
            item = {"ordinal": ordinal, "kind": unit.kind, "index": unit.index,
                    "label": ("table_original" if unit.kind == "table_canary" else
                              "prose_original" if unit.kind == "prose_canary" else
                              cases[unit.index].category),
                    "original_predicates": [t.predicate for t in triples],
                    "initial_batch_sha256": evaluated["stages"][0]["batch_sha256"],
                    **evaluated, "expected_gold_match": _gold(unit, cases, evaluated),
                    "admitted_turns": question["turns"],
                    "known_tokens": question.get("known_tokens"),
                    "usage_complete": question.get("usage_complete"),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "client_cleanup_ok": True}
        except DiagnosticStop as exc:
            stop_code = exc.code
        except BaseException as exc:
            try:
                journal.record(key, {"phase": "private_exception", "type": type(exc).__name__,
                                     "detail": str(exc)})
            except BaseException:
                stop_code = "private_evidence_write_failure"
            else:
                stop_code = "infrastructure_or_runtime_failure"
        finally:
            if client is not None:
                try:
                    client.close()
                except BaseException:
                    cleanup_ok = False
                    stop_code = "cleanup_failure"
            after = budget.snapshot()
            if item is not None:
                item["client_cleanup_ok"] = cleanup_ok
                results.append(item)
            if stop_code is not None:
                budget.halt(stop_code)
            try:
                journal.record(key, {"phase": "unit_finished", "stop_code": stop_code,
                                     "budget": after, "result": item})
            except BaseException:
                stop_code = "private_evidence_write_failure"
                budget.halt(stop_code)
        if stop_code is not None:
            break
    state = budget.snapshot()
    public_budget = {key: state[key] for key in
                     ("turns", "known_tokens", "usage_complete", "in_flight", "reserved")}
    first_failure = (warm.serialize_failure(state.get("first_failure"))
                     if callable(getattr(warm, "serialize_failure", None)) else None)
    complete = (len(results) == len(schedule()) and attempted == len(schedule()) and
                stop_code is None and cleanup_ok and not state["stopped"] and
                state["usage_complete"] and state["reserved"] == state["in_flight"] == 0 and
                state["turns"] == sum(item["admitted_turns"] for item in results))
    result = {"schema": SCHEMA, "units": results, "completed_units": len(results),
              "attempted_units": attempted, "malformed_units": sum(
                  item["outcome"] == "malformed" for item in results),
              "paid_budget": public_budget, "first_failure": first_failure,
              "client_cleanup_ok": cleanup_ok,
              "stop_code": stop_code or ("transport_or_budget_stop" if state["stopped"] else None),
              "diagnostic_completed": complete, "semantic_accuracy_accepted": False,
              "full_lme_ready": False, "completed_and_clean": False,
              "process_cleanup_verified": False}
    validate_public_result(result)
    return result


def validate_public_result(value: object) -> None:
    """Strict finite export and usage-prefix reconciliation, including partial runs."""
    _require(type(value) is dict and set(value) == {"schema", "units", "completed_units",
        "attempted_units", "malformed_units", "paid_budget", "first_failure",
        "client_cleanup_ok", "stop_code", "diagnostic_completed", "semantic_accuracy_accepted",
        "full_lme_ready", "completed_and_clean", "process_cleanup_verified"} and
        value["schema"] == SCHEMA, "schedule_invalid")
    units = value["units"]
    _require(type(units) is list and len(units) <= 8 and
             type(value["completed_units"]) is int and value["completed_units"] == len(units) and
             type(value["attempted_units"]) is int and
             len(units) <= value["attempted_units"] <= min(8, len(units) + 1) and
             type(value["malformed_units"]) is int and
             value["malformed_units"] == sum(type(x) is dict and x.get("outcome") == "malformed" for x in units),
             "schedule_invalid")
    turns = tokens = 0
    for ordinal, item in enumerate(units):
        unit = schedule()[ordinal]
        expected_label = ("table_original" if unit.kind == "table_canary" else
                          "prose_original" if unit.kind == "prose_canary" else
                          "supported" if unit.index == 9 else
                          "correction" if unit.index in (12, 13) else "reject")
        _require(type(item) is dict and set(item) == {"ordinal", "kind", "index", "label",
            "initial_batch_sha256", "original_predicates", "outcome", "error_code",
            "contract_code", "stages", "verdicts", "verdict_counts",
            "final_predicates", "expected_gold_match", "admitted_turns", "known_tokens",
            "usage_complete", "elapsed_seconds", "client_cleanup_ok"} and
            type(item["ordinal"]) is int and type(item["index"]) is int and
            type(item["kind"]) is str and type(item["label"]) is str and
            (item["ordinal"], item["kind"], item["index"], item["label"]) ==
            (ordinal, unit.kind, unit.index, expected_label) and
            type(item["outcome"]) is str and item["outcome"] in _OUTCOMES and
            type(item["original_predicates"]) is list and
            item["original_predicates"] == ({9: ["prefers"], 12: ["uses"],
                13: ["prefers"], 17: ["configured_with"], 19: ["prefers"],
                21: ["uses", "uses"]}[unit.index] if unit.kind == "control" else
                ["deploys_to" if unit.kind == "table_canary" else "uses"]) and
            (item["error_code"] is None or type(item["error_code"]) is str) and
            item["error_code"] in {None, "original_invalid", "alternatives_or_selector_invalid",
                                   "uncertain", "unsupported", "correction_collision"} and
            (item["contract_code"] is None or type(item["contract_code"]) is str) and
            item["contract_code"] in _CONTRACT_CODES | {None, "contract_other"} and
            (item["outcome"] == "malformed") == (item["contract_code"] is not None) and
            type(item["expected_gold_match"]) is bool and
            type(item["admitted_turns"]) is int and 1 <= item["admitted_turns"] <= 3 and
            type(item["known_tokens"]) is int and item["known_tokens"] >= 0 and
            item["usage_complete"] is True and type(item["client_cleanup_ok"]) is bool and
            type(item["elapsed_seconds"]) in (int, float) and math.isfinite(item["elapsed_seconds"]) and
            0 <= item["elapsed_seconds"] <= UNIT_CAP[2] + 10 and
            type(item["stages"]) is list and len(item["stages"]) == item["admitted_turns"] and
            1 <= len(item["stages"]) <= 3 and
            type(item["verdicts"]) is list and all(type(x) is str and x in _VERDICTS for x in item["verdicts"]) and
            len(item["verdicts"]) in (0, 2 if unit.kind == "control" and unit.index == 21 else 1) and
            type(item["verdict_counts"]) is dict and set(item["verdict_counts"]) == _VERDICTS and
            all(type(item["verdict_counts"][name]) is int and
                item["verdict_counts"][name] == item["verdicts"].count(name) for name in _VERDICTS) and
            type(item["final_predicates"]) is list and all(type(x) is str and x in v4.PREDICATE_ORDER for x in item["final_predicates"]),
            "schedule_invalid")
        sequence = [(s.get("stage"), s.get("recheck")) if type(s) is dict else None
                    for s in item["stages"]]
        _require(sequence in ([('original', False)],
                              [('original', False), ('alternatives', False)],
                              [('original', False), ('alternatives', False), ('original', True)]),
                 "schedule_invalid")
        for pos, stage in enumerate(item["stages"]):
            _require(type(stage) is dict and set(stage) == {"stage", "recheck", "batch_sha256",
                "schema_sha256", "prior_response_sha256", "states", "predicate_states"} and
                type(stage["stage"]) is str and type(stage["recheck"]) is bool and
                type(stage["states"]) is list and all(type(s) is str and s in _STATES for s in stage["states"]) and
                (len(stage["states"]) == (2 if unit.kind == "control" and unit.index == 21 else 1)
                 if stage["stage"] == "original" and not
                 (item["outcome"] == "malformed" and pos == len(item["stages"]) - 1)
                 else stage["states"] == []) and
                type(stage["predicate_states"]) is list and all(type(entry) is dict and
                    set(entry) == {"index", "predicate", "state"} and
                    type(entry["index"]) is int and 0 <= entry["index"] < 2 and
                    type(entry["predicate"]) is str and entry["predicate"] in v4.PREDICATE_ORDER and
                    type(entry["state"]) is str and entry["state"] in _STATES
                    for entry in stage["predicate_states"]) and
                (stage["stage"] == "alternatives" or not stage["predicate_states"]) and
                all(type(stage[k]) is str and len(stage[k]) == 64 and
                    all(c in "0123456789abcdef" for c in stage[k]) for k in
                    ("batch_sha256", "schema_sha256")) and
                (stage["prior_response_sha256"] is None if stage["stage"] == "original" else
                 type(stage["prior_response_sha256"]) is str and
                 len(stage["prior_response_sha256"]) == 64 and
                 all(c in "0123456789abcdef" for c in stage["prior_response_sha256"])),
                "schedule_invalid")
        if len(item["stages"]) >= 2:
            _require(item["stages"][1]["batch_sha256"] == item["initial_batch_sha256"] and
                     "not_established" in item["stages"][0]["states"] and
                     "ambiguous" not in item["stages"][0]["states"], "schedule_invalid")
            negatives = [i for i, state in enumerate(item["stages"][0]["states"])
                         if state == "not_established"]
            expected_pairs = [(i, predicate) for i in negatives
                              for predicate in v4.PREDICATE_ORDER
                              if predicate != item["original_predicates"][i]]
            actual_pairs = [(entry["index"], entry["predicate"])
                            for entry in item["stages"][1]["predicate_states"]]
            _require((item["outcome"] == "malformed" and
                      len(item["stages"]) == 2 and not actual_pairs) or
                     actual_pairs == expected_pairs, "schedule_invalid")
        if len(item["stages"]) == 3:
            initial = item["stages"][0]["states"]
            alternatives = item["stages"][1]["predicate_states"]
            negatives = [i for i, state in enumerate(initial) if state == "not_established"]
            _require("ambiguous" not in initial and negatives and
                     all(sum(entry["state"] == "supported" for entry in alternatives
                             if entry["index"] == i) == 1 for i in negatives) and
                     all(entry["state"] in {"supported", "not_established"}
                         for entry in alternatives) and
                     (item["outcome"] == "malformed" and item["stages"][2]["states"] == [] or
                      len(item["stages"][2]["states"]) == len(item["original_predicates"])),
                     "schedule_invalid")
        if item["outcome"] == "accepted":
            _require((len(item["stages"]) == 1 and
                      all(state == "supported" for state in item["stages"][0]["states"])) or
                     (len(item["stages"]) == 3 and
                      all(state == "supported" for state in item["stages"][2]["states"])),
                     "schedule_invalid")
            selected = list(item["original_predicates"])
            if len(item["stages"]) == 3:
                for entry in item["stages"][1]["predicate_states"]:
                    if entry["state"] == "supported":
                        selected[entry["index"]] = entry["predicate"]
            _require(item["final_predicates"] == selected and
                     item["verdicts"] == ["supported"] * len(selected),
                     "schedule_invalid")
        if len(item["stages"]) == 3 and item["outcome"] == "rejected":
            _require(any(state != "supported" for state in item["stages"][2]["states"]),
                     "schedule_invalid")
        _require(item["initial_batch_sha256"] == item["stages"][0]["batch_sha256"] and
                 item["stages"][0]["stage"] == "original" and
                 (item["outcome"] == "accepted") == (item["error_code"] is None) and
                 (item["outcome"] != "accepted" or len(item["final_predicates"]) in (1, 2)) and
                 (item["outcome"] == "accepted" or not item["final_predicates"]) and
                 item["expected_gold_match"] == _gold_from_public(unit, item),
                 "schedule_invalid")
        turns += item["admitted_turns"]
        tokens += item["known_tokens"] or 0
    paid = value["paid_budget"]
    _require(type(paid) is dict and set(paid) == {"turns", "known_tokens", "usage_complete",
        "in_flight", "reserved"} and all(type(paid[k]) is int and paid[k] >= 0 for k in
        ("turns", "known_tokens", "in_flight", "reserved")) and
        type(paid["usage_complete"]) is bool and paid["turns"] <= MAX_NEW_TURNS and
        turns <= paid["turns"] <= turns + 3 and
        tokens <= paid["known_tokens"] and
        (len(units) == value["attempted_units"] and paid["usage_complete"] and
         paid["turns"] == turns and paid["known_tokens"] == tokens or
         len(units) + 1 == value["attempted_units"]) and
        (value["stop_code"] is None or type(value["stop_code"]) is str) and
        value["stop_code"] in _STOP_CODES | {None} and
        type(value["client_cleanup_ok"]) is bool and
        all(value[k] is False for k in ("semantic_accuracy_accepted", "full_lme_ready",
                                       "completed_and_clean", "process_cleanup_verified")) and
        type(value["diagnostic_completed"]) is bool and
        value["diagnostic_completed"] == (len(units) == 8 and value["attempted_units"] == 8 and
             value["stop_code"] is None and value["client_cleanup_ok"] and
             paid["usage_complete"] and paid["in_flight"] == paid["reserved"] == 0 and
             paid["turns"] == turns) and
        (value["first_failure"] is None or type(value["first_failure"]) is dict and
         transport.warm.serialize_failure(value["first_failure"]) == value["first_failure"]),
        "schedule_invalid")
