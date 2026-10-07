"""Inactive fixed-schedule claim-task diagnostic; the host owns provenance and export.

Raw requests, responses, evidence and exceptions belong only in the private journal.
No provider call occurs on import or during canary input derivation.
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

from benchmarks import codex_subscription_claim_task_v1 as transport

contract = transport.contract
SCHEMA = "luna-claim-task-ablation-v1"
CONTROL_INDICES = (2, 3, 5, 6, 7, 9, 11, 12, 13, 17, 19, 21)
_POSITIVE_CONTROLS = frozenset((2, 3, 5, 6, 7, 9, 11))
_CORRECTION_CONTROLS = frozenset((12, 13))
_NEGATIVE_CONTROLS = frozenset((17, 19, 21))
MAX_NEW_TURNS = 29
MAX_KNOWN_TOKENS = 500_000
MAX_SECONDS = 1800
UNIT_CAP = (1, 100_000, 240)
CASE_SHA256 = "511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925"
_STATES = frozenset(("supported", "not_established", "ambiguous"))
_FINAL = frozenset(("supported", "unsupported", "uncertain", "replace_predicate"))


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
    """One immutable fsynced record per file, including before-dispatch records."""

    def __init__(self, directory: Path):
        _require(directory.is_absolute() and directory.is_dir() and
                 not directory.is_symlink() and (directory.stat().st_mode & 0o077) == 0 and
                 not any(directory.iterdir()), "private_directory_invalid")
        self.directory = directory
        self.sequence = 0

    def record(self, unit: str, value: dict) -> None:
        _require(type(unit) is str and unit.isascii() and
                 all(c.isalnum() or c in "-_" for c in unit) and
                 type(value) is dict, "private_record_invalid")
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
    arm: str


def schedule() -> tuple[ScheduledUnit, ...]:
    units = []
    for index in CONTROL_INDICES:
        units.extend(ScheduledUnit("control", index, arm)
                     for arm in (("A", "B") if index % 2 == 0 else ("B", "A")))
        if index == 12:
            units.append(ScheduledUnit("nominated_prefers", 12, "B"))
    units.extend((ScheduledUnit("table_canary", 0, "A"),
                  ScheduledUnit("table_canary", 0, "B"),
                  ScheduledUnit("prose_canary", 1, "B"),
                  ScheduledUnit("prose_canary", 1, "A")))
    assert len(units) == MAX_NEW_TURNS
    return tuple(units)


def _one_source_batch(batch, source_id: int, expected: tuple, *, prose: bool) -> None:
    _require(len(batch.triples) == len(batch.sources) == 1,
             "canary_batch_shape_invalid")
    triple, source = batch.triples[0], batch.sources[0]
    _require(triple.source_message_id == source.source_message_id == source_id and
             (triple.subject, triple.predicate, triple.object, triple.polarity) ==
             (expected[0], "uses" if prose else expected[2], expected[3], expected[5]) and
             all(getattr(triple, field) is None for field in
                 ("value_text", "value_numeric", "value_unit", "temporal_scope")),
             "canary_claim_invalid")
    contract.v3._checked(batch)


def derive_canary_batches(*, canary_module, chunk_module, semantic_probe_module,
                          candidate: Path, retained: tuple[tuple[dict, str], ...],
                          record: Callable[[dict], None]):
    """Replay eight exact ordinary calls and capture two initial v3 batches.

    A synthetic supported table response advances extraction. Prose raises a
    private BaseException before its judgment; no corrected prose is invented.
    The host must source-pin the candidate, helper modules and retained bytes.
    """
    expected = canary_module._CANARY_EXPECTED_CLAIMS
    _require(len(retained) == 8 and len(expected) == 2 and
             canary_module.EXTRACTION_CANARY_MAX_COMPLETION_CALLS == 24 and
             Path(canary_module.__file__).resolve() == candidate.resolve() / "benchmarks/extraction_canary.py" and
             Path(chunk_module.__file__).resolve() == candidate.resolve() / "hymem/extraction/chunk.py",
             "canary_input_invalid")
    source_records = canary_module._source_records()
    _require(len(source_records) == 2 and
             tuple(sid for sid, _ in source_records) == (expected[0][6], expected[1][6]),
             "canary_sources_invalid")

    class CapturedProse(BaseException):
        pass

    class FakePaid:
        observed_turns = 0
        observed_tokens = 0
        usage_complete = True

        def __init__(self):
            self.batches = []

        def complete_grounding(self, request, batch):
            contract.v3.validate_request(request, batch)
            self.batches.append(batch)
            if len(self.batches) == 2:
                raise CapturedProse()
            _require(len(self.batches) == 1, "canary_extra_judgment")
            quote = canary_module._TABLE_CLAIM_ROW
            _require(quote in batch.sources[0].content and len(quote) <= 192,
                     "canary_table_owned_quote_invalid")
            evidence = [{"source_message_id": batch.triples[0].source_message_id,
                         "region": "owned", "quote": quote}]
            check = {"state": "supported", "evidence_indices": [0]}
            payload = {"schema": contract.v3.GROUNDING_CONTRACT_VERSION,
                       "batch_sha256": batch.batch_sha256, "complete": True,
                       "classifications": [{"index": 0, "original": {
                           "state": "supported", "support": {"evidence": evidence,
                           "checks": {"attribution_and_roles": check,
                                      "relation_and_polarity": check}}},
                           "alternatives": None}]}
            raw = _canonical(payload).decode("utf-8")
            contract.v3.parse_grounding_response(raw, batch)
            return raw

    fake = FakePaid()
    replay = semantic_probe_module.HybridReplayClient(
        fake, retained, record=record,
        source_ids=(expected[0][6], expected[1][6]))
    try:
        chunk_module.extract_chunk(replay, canary_module._CANARY_CONTENT,
            source_records=source_records,
            completion_call_limit=canary_module.EXTRACTION_CANARY_MAX_COMPLETION_CALLS)
    except CapturedProse:
        pass
    else:
        raise DiagnosticStop("canary_prose_capture_missing")
    _require(replay.replayed == 8 and replay.new_calls == 1 and
             replay.grounding_phases == ["table_initial"] and len(fake.batches) == 2,
             "canary_replay_sequence_invalid")
    for index, batch in enumerate(fake.batches):
        _one_source_batch(batch, expected[index][6], expected[index], prose=index == 1)
        retained_source = json.loads(source_records[index][1])
        _require(type(retained_source) is dict and
                 type(retained_source.get("content")) is str and
                 batch.sources[0].content in retained_source["content"] and
                 batch.sources[0].source_role == retained_source.get("source_role") and
                 batch.sources[0].source_peer_id == retained_source.get("source_peer_id") and
                 batch.sources[0].source_created_at == retained_source.get("source_created_at"),
                 "canary_source_binding_invalid")
    _require(fake.batches[0].batch_sha256 != fake.batches[1].batch_sha256,
             "canary_batch_collision")
    record({"phase": "canary_batches_derived", "table_batch_sha256": fake.batches[0].batch_sha256,
            "prose_batch_sha256": fake.batches[1].batch_sha256,
            "ordinary_replays": replay.replayed, "synthetic_table_advancement": True})
    return tuple(fake.batches)


def _projection(result: contract.ArmResult) -> dict:
    originals = []
    for item in result.original:
        _require(item.state in _STATES, "projection_state_invalid")
        originals.append({"state": item.state,
                          "evidence_count": len(item.evidence),
                          "evidence_sha256": _sha(_canonical([asdict(e) for e in item.evidence])),
                          "check_names": [name for name, _ in item.checks],
                          "check_count": len(item.checks)})
    alternatives = None
    finals = None
    if result.arm == "A":
        _require(result.alternative_states is not None and result.final_verdicts is not None,
                 "projection_a_invalid")
        alternatives = []
        for item in result.alternative_states:
            _require(item is None or all(state in _STATES for _, state in item),
                     "projection_alternative_invalid")
            alternatives.append(None if item is None else
                                [{"predicate": name, "state": state} for name, state in item])
        finals = []
        for verdict in result.final_verdicts:
            _require(verdict.status in _FINAL, "projection_final_invalid")
            finals.append({"status": verdict.status,
                           "predicate": verdict.predicate if verdict.status in
                           ("supported", "replace_predicate") else None})
    else:
        _require(result.alternative_states is None and result.final_verdicts is None,
                 "projection_b_invalid")
    return {"batch_sha256": result.batch_sha256, "original": originals,
            "alternative_states": alternatives, "final_verdicts": finals}


def _unit_input(unit: ScheduledUnit, cases, canary_batches):
    if unit.kind in ("table_canary", "prose_canary"):
        batch = canary_batches[unit.index]
        return batch.triples, batch.sources, batch.batch_sha256
    case = cases[unit.index]
    from hymem.extraction.grounding_classification_gate_v3 import _v2_source
    sources = tuple(_v2_source(source) for source in case.sources)
    triples = case.triples
    if unit.kind == "nominated_prefers":
        _require(len(triples) == 1 and triples[0].predicate == "uses", "nominated_case_drift")
        triples = (replace(triples[0], predicate="prefers"),)
    return triples, sources, None


def _expected_metadata(unit: ScheduledUnit, cases) -> dict:
    if unit.kind.endswith("canary"):
        return {"label": "table_original" if unit.index == 0 else "prose_original",
                "scope_ambiguous": False}
    if unit.kind == "nominated_prefers":
        return {"label": "nominated_positive", "scope_ambiguous": False}
    case = cases[unit.index]
    return {"label": {"supported": "positive", "correction": "original_negative_correction",
                      "reject": "negative"}[case.category],
            "scope_ambiguous": unit.index == 5}


def run_campaign(*, concurrent, warm, binary: str, cases_module,
                 canary_batches, journal: PrivateJournal, client_factory=None) -> dict:
    """Execute the predeclared 29 one-turn units; malformed semantics continue."""
    cases = cases_module.cases()
    _require(cases_module.suite_sha256() == CASE_SHA256 and len(cases) == 24 and
             len({case.case_id for case in cases}) == 24 and
             len(canary_batches) == 2, "schedule_invalid")
    expected = (("HyMem Canary Relay", "deploys_to", "Fly.io", 1),
                ("Avery Boundary Canary", "uses", "PostgreSQL", 1))
    for batch, fields in zip(canary_batches, expected, strict=True):
        _require(len(batch.triples) == len(batch.sources) == 1 and
                 (batch.triples[0].subject, batch.triples[0].predicate,
                  batch.triples[0].object, batch.triples[0].polarity) == fields,
                 "canary_batch_invalid")
        contract.v3._checked(batch)
    limits_type = warm.BudgetLimits if client_factory is None else concurrent.BudgetLimits
    budget_type = warm.SharedBudget if client_factory is None else concurrent.SharedBudget
    if client_factory is None:
        _require(concurrent is warm.concurrent and warm is transport.warm,
                 "transport_binding_invalid")
    budget = budget_type(limits_type(MAX_NEW_TURNS, MAX_KNOWN_TOKENS, MAX_SECONDS),
                         max_in_flight=1)
    factory = client_factory or (lambda key, cap, shared: transport.ClaimTaskSubscriptionClient(
        binary, shared, key, limits_type(*cap), max_requests=16, max_age_seconds=300))
    results = []
    stop_code = None
    cleanup_ok = True
    attempted_units = 0
    for ordinal, unit in enumerate(schedule()):
        key = f"unit-{ordinal:02d}"
        client = None
        item = None
        before = budget.snapshot()
        started = time.monotonic()
        try:
            _require(not before["stopped"] and before["usage_complete"] and
                     before["reserved"] == before["in_flight"] == 0,
                     "preunit_accounting_invalid")
            triples, sources, expected_batch_hash = _unit_input(unit, cases, canary_batches)
            request, batch = contract.build_arm_request(unit.arm, triples, sources)
            contract.validate_arm_request(unit.arm, request, batch)
            _require(expected_batch_hash is None or batch.batch_sha256 == expected_batch_hash,
                     "canary_batch_binding_invalid")
            schema = contract.build_arm_output_schema(unit.arm, batch)
            schema_hash = _sha(_canonical(schema))
            metadata = _expected_metadata(unit, cases)
            journal.record(key, {"phase": "before_dispatch", "ordinal": ordinal,
                       "kind": unit.kind, "index": unit.index, "arm": unit.arm,
                       "request": asdict(request), "batch": batch.canonical_json,
                       "batch_sha256": batch.batch_sha256,
                       "output_schema_sha256": schema_hash})
            attempted_units += 1
            client = factory(key, UNIT_CAP, budget)
            raw = client.complete_arm(unit.arm, request, batch)
            journal.record(key, {"phase": "response_returned", "response": raw})
            try:
                parsed = contract.parse_arm_response(unit.arm, raw, batch)
            except contract.v2.GroundingContractError:
                item = {"ordinal": ordinal, "kind": unit.kind, "index": unit.index,
                        "arm": unit.arm, **metadata, "batch_sha256": batch.batch_sha256,
                        "schema_sha256": schema_hash, "outcome": "malformed",
                        "result": None}
            else:
                journal.record(key, {"phase": "validated_result", "result": asdict(parsed)})
                item = {"ordinal": ordinal, "kind": unit.kind, "index": unit.index,
                        "arm": unit.arm, **metadata, "batch_sha256": batch.batch_sha256,
                        "schema_sha256": schema_hash, "outcome": "valid",
                        "result": _projection(parsed)}
            state = budget.snapshot()
            _require(state["questions"][key]["turns"] == client.observed_turns == 1 and
                     state["reserved"] == state["in_flight"] == 0 and
                     state["usage_complete"] and client.usage_complete and
                     not state["stopped"], "postunit_accounting_invalid")
        except DiagnosticStop as exc:
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
            if item is not None:
                question = after["questions"].get(key, {})
                item.update({"admitted_turns": question.get("turns"),
                             "known_tokens": question.get("known_tokens"),
                             "usage_complete": question.get("usage_complete"),
                             "elapsed_seconds": round(time.monotonic() - started, 3),
                             "client_cleanup_ok": cleanup_ok})
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
    completed = len(results) == MAX_NEW_TURNS and cleanup_ok and stop_code is None and not state["stopped"] and state["usage_complete"] and state["reserved"] == state["in_flight"] == 0 and state["turns"] == MAX_NEW_TURNS and attempted_units == MAX_NEW_TURNS and all(
        item["admitted_turns"] == 1 and item["usage_complete"] is True and
        item["client_cleanup_ok"] is True for item in results)
    result = {"schema": SCHEMA, "units": results,
              "completed_units": len(results),
              "attempted_units": attempted_units,
              "malformed_units": sum(item["outcome"] == "malformed" for item in results),
              "paid_budget": public_budget, "first_failure": first_failure,
              "client_cleanup_ok": cleanup_ok,
              "stop_code": stop_code or ("transport_or_budget_stop" if state["stopped"] else None),
              "diagnostic_completed": completed,
              "semantic_accuracy_accepted": False, "full_lme_ready": False,
              "completed_and_clean": False, "process_cleanup_verified": False}
    validate_public_result(result)
    return result


def validate_public_result(value: object) -> None:
    """Reject accidental raw text, evidence or provider exceptions in export."""
    _require(type(value) is dict and set(value) == {"schema", "units", "completed_units",
        "attempted_units", "malformed_units", "paid_budget", "first_failure", "client_cleanup_ok", "stop_code",
        "diagnostic_completed", "semantic_accuracy_accepted", "full_lme_ready",
        "completed_and_clean", "process_cleanup_verified"} and value["schema"] == SCHEMA,
        "public_shape_invalid")
    _require(type(value["units"]) is list and len(value["units"]) <= MAX_NEW_TURNS and
             type(value["completed_units"]) is int and
             value["completed_units"] == len(value["units"]) and
             type(value["attempted_units"]) is int and
             len(value["units"]) <= value["attempted_units"] <= min(MAX_NEW_TURNS, len(value["units"]) + 1) and
             type(value["malformed_units"]) is int and
             0 <= value["malformed_units"] <= len(value["units"]),
             "public_units_invalid")
    for ordinal, item in enumerate(value["units"]):
        planned = schedule()[ordinal]
        expected_label = ("table_original" if planned.kind == "table_canary" else
                          "prose_original" if planned.kind == "prose_canary" else
                          "nominated_positive" if planned.kind == "nominated_prefers" else
                          "positive" if planned.index in _POSITIVE_CONTROLS else
                          "original_negative_correction" if planned.index in _CORRECTION_CONTROLS else
                          "negative")
        _require(type(item) is dict and set(item) == {"ordinal", "kind", "index", "arm",
            "label", "scope_ambiguous", "batch_sha256", "schema_sha256", "outcome",
            "result", "admitted_turns", "known_tokens", "usage_complete",
            "elapsed_seconds", "client_cleanup_ok"} and
            type(item["ordinal"]) is int and type(item["index"]) is int and
            type(item["kind"]) is str and type(item["arm"]) is str and
            (item["ordinal"], item["kind"], item["index"], item["arm"]) ==
            (ordinal, planned.kind, planned.index, planned.arm) and
            type(item["label"]) is str and
            item["label"] == expected_label and
            type(item["scope_ambiguous"]) is bool and
            item["scope_ambiguous"] == (planned.kind == "control" and planned.index == 5) and
            all(type(item[key]) is str and len(item[key]) == 64 and
                all(c in "0123456789abcdef" for c in item[key])
                for key in ("batch_sha256", "schema_sha256")) and
            type(item["outcome"]) is str and item["outcome"] in {"valid", "malformed"} and
            (item["admitted_turns"] is None or
             type(item["admitted_turns"]) is int and 0 <= item["admitted_turns"] <= 1) and
            (item["known_tokens"] is None or
             type(item["known_tokens"]) is int and item["known_tokens"] >= 0) and
            (item["usage_complete"] is None or
             type(item["usage_complete"]) is bool) and
            type(item["client_cleanup_ok"]) is bool and
            type(item["elapsed_seconds"]) in (int, float) and
            math.isfinite(item["elapsed_seconds"]) and
            0 <= item["elapsed_seconds"] <= UNIT_CAP[2] + 10,
            "public_unit_invalid")
        projection = item["result"]
        if item["outcome"] == "malformed":
            _require(projection is None, "public_malformed_invalid")
            continue
        _require(type(projection) is dict and set(projection) ==
            {"batch_sha256", "original", "alternative_states", "final_verdicts"} and
            projection["batch_sha256"] == item["batch_sha256"] and
            type(projection["original"]) is list and
            len(projection["original"]) == (2 if planned.kind == "control" and
                                            planned.index == 21 else 1),
            "public_projection_invalid")
        for assessment in projection["original"]:
            _require(type(assessment) is dict and set(assessment) ==
                {"state", "evidence_count", "evidence_sha256", "check_names", "check_count"} and
                type(assessment["state"]) is str and assessment["state"] in _STATES and
                type(assessment["evidence_count"]) is int and
                0 <= assessment["evidence_count"] <= 8 and
                type(assessment["evidence_sha256"]) is str and
                len(assessment["evidence_sha256"]) == 64 and
                all(c in "0123456789abcdef" for c in assessment["evidence_sha256"]) and
                type(assessment["check_names"]) is list and
                all(type(name) is str for name in assessment["check_names"]) and
                set(assessment["check_names"]).issubset({"attribution_and_roles",
                "relation_and_polarity", "value_text", "value_numeric", "value_unit",
                "temporal_scope"}) and len(set(assessment["check_names"])) ==
                len(assessment["check_names"]) == assessment["check_count"] and
                type(assessment["check_count"]) is int,
                "public_assessment_invalid")
            _require((assessment["state"] == "supported" and
                      1 <= assessment["evidence_count"] <= 8 and
                      assessment["check_count"] >= 2) or
                     (assessment["state"] != "supported" and
                      assessment["evidence_count"] == assessment["check_count"] == 0 and
                      assessment["evidence_sha256"] == _sha(_canonical([]))),
                     "public_assessment_consistency_invalid")
        if planned.arm == "B":
            _require(projection["alternative_states"] is None and
                     projection["final_verdicts"] is None, "public_b_invalid")
        else:
            alternatives, finals = projection["alternative_states"], projection["final_verdicts"]
            _require(type(alternatives) is list and type(finals) is list and
                     len(alternatives) == len(finals) == len(projection["original"]),
                     "public_a_invalid")
            for entry in alternatives:
                _require(entry is None or type(entry) is list and all(
                    type(pair) is dict and set(pair) == {"predicate", "state"} and
                    type(pair["predicate"]) is str and
                    pair["predicate"] in contract.v3.PREDICATE_ORDER and
                    type(pair["state"]) is str and pair["state"] in _STATES
                    for pair in entry), "public_alternatives_invalid")
            for entry in finals:
                _require(type(entry) is dict and set(entry) == {"status", "predicate"} and
                         type(entry["status"]) is str and entry["status"] in _FINAL and
                         (entry["predicate"] is None or type(entry["predicate"]) is str and entry["predicate"] in
                          contract.v3.PREDICATE_ORDER), "public_final_invalid")
    _require(type(value["paid_budget"]) is dict and
             set(value["paid_budget"]) == {"turns", "known_tokens", "usage_complete",
                                           "in_flight", "reserved"} and
             all(type(value["paid_budget"][key]) is int and
                 value["paid_budget"][key] >= 0 for key in
                 ("turns", "known_tokens", "in_flight", "reserved")) and
             type(value["paid_budget"]["usage_complete"]) is bool and
             type(value["client_cleanup_ok"]) is bool and
             type(value["diagnostic_completed"]) is bool,
             "public_budget_invalid")
    clean_complete = (value["completed_units"] == value["attempted_units"] == MAX_NEW_TURNS and
        value["paid_budget"]["turns"] == MAX_NEW_TURNS and
        value["paid_budget"]["usage_complete"] is True and
        value["paid_budget"]["in_flight"] == value["paid_budget"]["reserved"] == 0 and
        value["client_cleanup_ok"] is True and value["stop_code"] is None and
        all(item["admitted_turns"] == 1 and item["usage_complete"] is True and
            item["client_cleanup_ok"] is True for item in value["units"]))
    _require(value["malformed_units"] == sum(x["outcome"] == "malformed" for x in value["units"]) and
             (value["first_failure"] is None or
              type(value["first_failure"]) is dict and
              transport.warm.serialize_failure(value["first_failure"]) == value["first_failure"]) and
             value["stop_code"] in {None, "private_evidence_write_failure",
                 "preunit_accounting_invalid", "postunit_accounting_invalid",
                 "cleanup_failure", "infrastructure_or_runtime_failure",
                 "transport_or_budget_stop", "canary_batch_binding_invalid",
                 "projection_state_invalid", "projection_a_invalid",
                 "projection_b_invalid", "projection_alternative_invalid",
                 "projection_final_invalid", "nominated_case_drift"} and
             all(value[key] is False for key in
                 ("semantic_accuracy_accepted", "full_lme_ready",
                  "completed_and_clean", "process_cleanup_verified")) and
             value["paid_budget"]["turns"] <= value["attempted_units"] and
             value["completed_units"] <= value["paid_budget"]["turns"] and
             sum(item["known_tokens"] or 0 for item in value["units"]) <=
             value["paid_budget"]["known_tokens"] and
             (not clean_complete or
              sum(item["known_tokens"] for item in value["units"]) ==
              value["paid_budget"]["known_tokens"]) and
             value["diagnostic_completed"] == clean_complete,
             "public_summary_invalid")
