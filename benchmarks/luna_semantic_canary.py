"""Offline-reviewable, opt-in Luna semantic canary for the inactive candidate.

The ordinary source fixture and final gold oracle are the frozen canary's.
This module adds a separately counted grounding path; it never alters output.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.luna_semantic_stage_accounting import SOURCE_SHA256, verify_candidate

SCHEMA = "luna-semantic-canary-v1"
CANDIDATE_CANARY_SHA256 = "861dd9562db848fa61d97f5108ba2407e06c879a5563a0a4ce388fd47d6032ee"
FIXTURE_SHA256 = "3fedadfbcdf35a013f8f61c5ea3f6ccc3e150aca0d025d8597327704551e94de"
EXTRACTION_IDENTITY = "hymem-extraction-contract-sha256-v1:f349e2fa14d1778bc556869d346ca44183c025e779a919b3f15e82f7c3d78d46"
MAX_CALLS = 12
_OPTIONAL = ("value_text", "value_numeric", "value_unit", "temporal_scope")
_CORE = ("subject", "predicate", "object", "polarity", "source_message_id")


def _is_grounding(request: Any) -> bool:
    try:
        wire = json.loads(request.user)
    except (TypeError, ValueError, AttributeError):
        return False
    return type(wire) is dict and set(wire) == {"batch", "batch_sha256"}


def _safe_grounding_trace(request: Any, response: Any, expected: tuple, *, recheck: bool,
                          source_payloads: dict[int, dict], ordinary_payloads: list[dict]) -> dict:
    """Verify exact request hash and summarize only bounded structural data."""
    from hymem.extraction.grounding import GROUNDING_CONTRACT_VERSION

    wire = json.loads(request.user)
    batch = wire["batch"]
    if type(batch) is not dict:
        raise ValueError("grounding_binding_invalid")
    canonical = json.dumps(batch, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if (batch.get("version") != GROUNDING_CONTRACT_VERSION
            or digest != wire["batch_sha256"] or type(batch.get("candidates")) is not list
            or len(batch["candidates"]) != 1 or type(batch.get("sources")) is not list
            or len(batch["sources"]) != 1):
        raise ValueError("grounding_binding_invalid")
    candidate = batch["candidates"][0]
    source = batch["sources"][0]
    if type(candidate) is not dict or type(source) is not dict:
        raise ValueError("grounding_candidate_invalid")
    index = next((i for i, e in enumerate(expected)
                  if (candidate.get("subject"), candidate.get("object"),
                      candidate.get("polarity"), candidate.get("source_message_id"))
                  == (e[0], e[3], e[5], e[6])), None)
    if (index is None or source.get("source_message_id") != expected[index][6]
            or candidate.get("source_message_id") != source.get("source_message_id")
            or type(source.get("content")) is not str
            or source["content"] not in source_payloads[expected[index][6]]["content"]
            or any(source.get(k) != source_payloads[expected[index][6]].get(k)
                   for k in ("source_role", "source_peer_id", "source_created_at"))
            or any(candidate.get(field) is not None for field in _OPTIONAL)):
        raise ValueError("grounding_candidate_invalid")
    # Reconstruct the source the pinned gate would build from the exact
    # ordinary request payload. This binds every context prefix and metadata
    # field, not merely the owned text or header words.
    from hymem.extraction.grounding_gate import _source
    matching = [payload for payload in ordinary_payloads
                if payload.get("source_message_id") == expected[index][6]
                and payload.get("content") == source["content"]]
    if not matching:
        raise ValueError("grounding_context_invalid")
    expected_sources = []
    for payload in matching:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"))
        built = asdict(_source((expected[index][6], encoded), ()))
        built["contexts"] = list(built["contexts"])
        expected_sources.append(built)
    if source not in expected_sources:
        raise ValueError("grounding_context_invalid")
    e = expected[index]
    if recheck and candidate.get("predicate") != e[2]:
        raise ValueError("recheck_candidate_invalid")
    if type(response) is not str:
        raise ValueError("grounding_response_invalid")
    parsed = json.loads(response)
    if (type(parsed) is not dict or set(parsed) != {"schema", "batch_sha256", "complete", "verdicts"}
            or parsed["schema"] != "source-grounding-v1" or parsed["batch_sha256"] != digest
            or parsed["complete"] is not True or type(parsed["verdicts"]) is not list
            or len(parsed["verdicts"]) != 1):
        raise ValueError("grounding_response_invalid")
    verdict = parsed["verdicts"][0]
    if (type(verdict) is not dict or set(verdict) != {"index", "status", "predicate", "evidence"}
            or verdict["index"] != 0 or type(verdict["status"]) is not str
            or verdict["status"] not in {"supported", "replace_predicate"}
            or type(verdict["evidence"]) is not list or not verdict["evidence"]):
        raise ValueError("grounding_verdict_invalid")
    if recheck:
        if verdict["status"] != "supported" or verdict["predicate"] != e[2]:
            raise ValueError("recheck_verdict_invalid")
    elif verdict["status"] == "replace_predicate":
        if candidate.get("predicate") == e[2] or verdict["predicate"] != e[2]:
            raise ValueError("correction_invalid")
    elif candidate.get("predicate") != e[2] or verdict["predicate"] != e[2]:
        raise ValueError("unsupported_raw_claim")
    return {"claim_index": index, "batch_sha256": digest,
            "status": verdict["status"], "recheck": recheck,
            "candidate_predicate_expected": candidate.get("predicate") == e[2],
            "owned_source_bound": True}


def validate_report(report: dict, *, fixture_sha256: str) -> None:
    """Strict finite summary validator; only the runner can witness raw calls."""
    keys = {"schema", "passed", "fixture_sha256", "source_sha256", "extraction_identity", "completion_calls",
            "ordinary_calls", "grounding_initial_calls", "grounding_recheck_calls",
            "provider_attempts", "grounding_provider_attempts", "initial_prepartition_leaves",
            "ordinary_path_valid", "raw_exact_emissions", "raw_wrong_predicate_emissions",
            "corrected_claim_indexes", "grounding_trace", "matched_core_claims",
            "type_fields_invalid", "type_fields_wrong", "final_type_hint_mismatch",
            "usage_complete", "observed_turn_delta", "observed_token_delta", "failure_code"}
    if type(report) is not dict or set(report) != keys or report["schema"] != SCHEMA:
        raise ValueError("report_shape_invalid")
    if (report["fixture_sha256"] != fixture_sha256 or report["source_sha256"] != SOURCE_SHA256
            or report["extraction_identity"] != EXTRACTION_IDENTITY):
        raise ValueError("report_source_invalid")
    for key in ("completion_calls", "ordinary_calls", "grounding_initial_calls",
                "grounding_recheck_calls", "provider_attempts", "grounding_provider_attempts",
                "initial_prepartition_leaves", "matched_core_claims", "type_fields_invalid",
                "type_fields_wrong", "observed_turn_delta"):
        if type(report[key]) is not int or report[key] < 0:
            raise ValueError("report_count_invalid")
    if report["passed"]:
        if type(report["observed_token_delta"]) is not int or report["observed_token_delta"] < 0:
            raise ValueError("report_count_invalid")
    elif report["observed_token_delta"] is not None and (
            type(report["observed_token_delta"]) is not int or report["observed_token_delta"] < 0):
        raise ValueError("report_count_invalid")
    if (type(report["passed"]) is not bool or type(report["usage_complete"]) is not bool
            or type(report["ordinary_path_valid"]) is not bool
            or type(report["final_type_hint_mismatch"]) is not bool
            or type(report["raw_exact_emissions"]) is not list
            or type(report["raw_wrong_predicate_emissions"]) is not list
            or len(report["raw_exact_emissions"]) != 2
            or len(report["raw_wrong_predicate_emissions"]) != 2
            or any(type(n) is not int or n < 0 for n in
                   report["raw_exact_emissions"] + report["raw_wrong_predicate_emissions"])
            or type(report["corrected_claim_indexes"]) is not list
            or any(type(i) is not int or i not in (0, 1) for i in report["corrected_claim_indexes"])
            or report["corrected_claim_indexes"] != sorted(set(report["corrected_claim_indexes"]))
            or type(report["grounding_trace"]) is not list):
        raise ValueError("report_field_invalid")
    traces = report["grounding_trace"]
    for trace in traces:
        if (type(trace) is not dict or set(trace) != {"claim_index", "batch_sha256", "status", "recheck",
                                              "candidate_predicate_expected", "owned_source_bound"}
                or type(trace["claim_index"]) is not int or trace["claim_index"] not in (0, 1)
                or type(trace["batch_sha256"]) is not str or len(trace["batch_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in trace["batch_sha256"])
                or type(trace["status"]) is not str
                or trace["status"] not in {"supported", "replace_predicate"}
                or type(trace["recheck"]) is not bool
                or type(trace["candidate_predicate_expected"]) is not bool
                or trace["owned_source_bound"] is not True):
            raise ValueError("report_trace_invalid")
    if report["passed"]:
        initial = [t for t in traces if not t["recheck"]]
        repeat = [t for t in traces if t["recheck"]]
        corrected = report["corrected_claim_indexes"]
        if (report["failure_code"] is not None or not report["ordinary_path_valid"]
                or not report["usage_complete"] or report["final_type_hint_mismatch"]
                or report["type_fields_invalid"] or report["type_fields_wrong"]
                or report["matched_core_claims"] != 2
                or report["ordinary_calls"] != 8 or report["grounding_initial_calls"] != 2
                or report["grounding_recheck_calls"] != len(corrected)
                or report["completion_calls"] != 10 + len(corrected)
                or report["completion_calls"] > MAX_CALLS
                or report["observed_turn_delta"] != report["completion_calls"]
                or report["provider_attempts"] < report["completion_calls"]
                or report["provider_attempts"] > 72
                or report["grounding_provider_attempts"] < 2 + len(corrected)
                or report["grounding_provider_attempts"] > report["provider_attempts"]
                or report["provider_attempts"] < report["grounding_provider_attempts"] + report["ordinary_calls"]
                or report["initial_prepartition_leaves"] != 4
                or len(initial) != 2 or sorted(t["claim_index"] for t in initial) != [0, 1]
                or len(repeat) != len(corrected)
                or sorted(t["claim_index"] for t in repeat) != corrected
                or any(t["status"] != "supported" or not t["candidate_predicate_expected"] for t in repeat)
                or any(next(j for j, x in enumerate(traces) if x is t)
                       <= next(j for j, x in enumerate(traces) if x["claim_index"] == t["claim_index"])
                       for t in repeat)
                or any(t["batch_sha256"] == next(x["batch_sha256"] for x in initial
                       if x["claim_index"] == t["claim_index"]) for t in repeat)
                or any((t["status"] == "replace_predicate") != (t["claim_index"] in corrected)
                       for t in initial)
                or any(t["status"] == "supported" and not t["candidate_predicate_expected"]
                       or t["status"] == "replace_predicate" and t["candidate_predicate_expected"]
                       for t in initial)
                or any(report["raw_exact_emissions"][i] + report["raw_wrong_predicate_emissions"][i] != 1
                       for i in (0, 1))
                or any(report["raw_wrong_predicate_emissions"][i] != (i in corrected)
                       for i in (0, 1))):
            raise ValueError("passed_report_inconsistent")
    elif report["failure_code"] != "canary_contract_failed":
        raise ValueError("failed_report_code_invalid")


def run_canary(canary, chunk, client, *, candidate: Path, evidence=None) -> dict:
    """Run direct benchmark-grade canary with the unchanged 24-call hard ceiling."""
    verify_candidate(candidate)
    candidate = candidate.resolve(strict=True)
    if hashlib.sha256((candidate / "benchmarks/extraction_canary.py").read_bytes()).hexdigest() != CANDIDATE_CANARY_SHA256:
        raise ValueError("fixture_source_drift")
    if (Path(canary.__file__).resolve() != candidate / "benchmarks/extraction_canary.py"
            or Path(chunk.__file__).resolve() != candidate / "hymem/extraction/chunk.py"):
        raise ValueError("candidate_import_drift")
    # Fixture and final oracle come only from the pinned candidate module.
    expected = canary._CANARY_EXPECTED_CLAIMS
    policy = canary.extraction_canary_policy()
    source_records = canary._source_records()
    fixture = [{"content": json.loads(encoded)["content"], "source_message_id": sid}
               for sid, encoded in source_records]
    fixture_digest = hashlib.sha256(json.dumps(fixture, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if (len(expected) != 2 or policy["normal_execution_path"]["primary_requests"] != 4
            or canary.EXTRACTION_CANARY_FIXTURE_SHA256 != FIXTURE_SHA256
            or fixture_digest != FIXTURE_SHA256
            or policy["extraction_contract"]["identity"] != EXTRACTION_IDENTITY
            or policy["max_completion_calls"] != 24 or policy["max_provider_attempts"] != 72
            or chunk.MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK != 96):
        raise ValueError("fixture_contract_drift")
    recording = canary._RecordingClient(client)
    start_turns = client.observed_turns
    start_tokens = client.observed_tokens
    try:
        result = chunk.extract_chunk(recording, canary._CANARY_CONTENT,
            source_records=source_records,
            completion_call_limit=canary.EXTRACTION_CANARY_MAX_COMPLETION_CALLS)
    finally:
        if evidence is not None:
            evidence({"requests": [vars(r) for r in recording.requests],
                      "responses": [r for _, r in recording.responses]})
    pairs = list(recording.responses)
    ordinary = [(r, v) for r, v in pairs if not _is_grounding(r)]
    ground = [(r, v) for r, v in pairs if _is_grounding(r)]
    ordinary_payloads = [payload for request, _ in ordinary
                         for payload in canary._request_source_payloads(request)[0]]
    ordinary_path = canary._request_execution_path([r for r, _ in ordinary], ordinary)
    ordinary_path["provider_output_truncations"] = recording.provider_output_truncations
    normal = policy["normal_execution_path"]
    ordinary_valid = all(ordinary_path.get(k) == v for k, v in normal.items()
                         if not k.endswith("_emissions"))
    exact, wrong = [0, 0], [0, 0]
    from hymem.extraction.jsonio import loads_exact_or_fenced
    from hymem.extraction.triples import normalize_combined_triple_item
    for request, response in ordinary:
        payloads, failures = canary._request_source_payloads(request)
        if failures:
            continue
        data = loads_exact_or_fenced(response)
        if type(data) is not dict or type(data.get("triples")) is not list:
            continue
        for raw in data["triples"]:
            item, errors, _ = normalize_combined_triple_item(raw, require_source_message_id=True)
            if item is None or errors:
                continue
            for i, e in enumerate(expected):
                if (item["subject"], item["object"], item["polarity"], item["source_message_id"]) != (e[0], e[3], e[5], e[6]):
                    continue
                payload = next((p for p in payloads if p.get("source_message_id") == e[6]), None)
                context_ok = bool(payload and (
                    (i == 0 and canary._TABLE_CLAIM_ROW in payload.get("content", "")
                     and canary._strict_equal(payload.get("source_fragment_context"), canary._EXPECTED_TABLE_CONTEXT))
                    or (i == 1 and canary._PROSE_BOUNDARY_RIGHT in payload.get("content", "")
                        and canary._strict_equal(payload.get("source_boundary_context"), canary._EXPECTED_PROSE_BOUNDARY_CONTEXT))))
                if context_ok and all(item.get(f) is None for f in _OPTIONAL):
                    (exact if item["predicate"] == e[2] else wrong)[i] += 1
    trace = []
    trace_valid = True
    source_payloads = {sid: json.loads(encoded) for sid, encoded in source_records}
    initial_seen = set()
    correction_pending = set()
    try:
        for request, response in ground:
            wire = json.loads(request.user)
            candidate = wire["batch"]["candidates"][0]
            index = next(i for i, e in enumerate(expected)
                         if candidate.get("source_message_id") == e[6])
            recheck = index in correction_pending
            if index in initial_seen and not recheck:
                raise ValueError("repeated_initial")
            item = _safe_grounding_trace(request, response, expected,
                                         recheck=recheck, source_payloads=source_payloads,
                                         ordinary_payloads=ordinary_payloads)
            trace.append(item)
            if recheck:
                correction_pending.remove(index)
            else:
                initial_seen.add(index)
                if item["status"] == "replace_predicate":
                    correction_pending.add(index)
        if correction_pending:
            raise ValueError("missing_recheck")
    except (ValueError, KeyError, TypeError, AttributeError, IndexError,
            OverflowError, StopIteration):
        trace_valid = False
    corrected = sorted(t["claim_index"] for t in trace if t["status"] == "replace_predicate")
    final_types = {entity: typ for e in expected for entity, typ in ((e[0], e[1]), (e[3], e[4]))}
    type_mismatch = any(final_types.get(k) != v for k, v in result.entity_type_hints.items())
    matched = sum(1 for e in expected if sum(
        (t.subject, t.predicate, t.object, t.polarity, t.source_message_id)
        == (e[0], e[2], e[3], e[5], e[6]) and
        all(getattr(t, f) is None for f in _OPTIONAL) for t in result.triples) == 1)
    type_invalid = type_wrong = 0
    for request, response in ordinary:
        data = loads_exact_or_fenced(response)
        if type(data) is not dict or type(data.get("triples")) is not list:
            continue
        for raw in data["triples"]:
            if type(raw) is not dict:
                continue
            for e in expected:
                if (raw.get("subject"), raw.get("object"), raw.get("source_message_id")) != (e[0], e[3], e[6]):
                    continue
                for side, typ in (("subject", e[1]), ("object", e[4])):
                    field = side + "_type"
                    if field in raw:
                        if type(raw[field]) is not str:
                            type_invalid += 1
                        elif raw[field] != typ:
                            type_wrong += 1
    turns = client.observed_turns - start_turns
    token_delta = (client.observed_tokens - start_tokens if type(start_tokens) is int
                   and type(client.observed_tokens) is int else None)
    report = {"schema": SCHEMA, "passed": False,
        "fixture_sha256": canary.EXTRACTION_CANARY_FIXTURE_SHA256,
        "source_sha256": dict(SOURCE_SHA256), "extraction_identity": EXTRACTION_IDENTITY,
        "completion_calls": result.completion_calls,
        "ordinary_calls": len(ordinary), "grounding_initial_calls": result.grounding_calls - result.grounding_recheck_calls,
        "grounding_recheck_calls": result.grounding_recheck_calls,
        "provider_attempts": result.provider_attempts,
        "grounding_provider_attempts": result.grounding_provider_attempts,
        "initial_prepartition_leaves": result.initial_prepartition_leaves,
        "ordinary_path_valid": ordinary_valid, "raw_exact_emissions": exact,
        "raw_wrong_predicate_emissions": wrong, "corrected_claim_indexes": corrected,
        "grounding_trace": trace, "matched_core_claims": matched,
        "type_fields_invalid": type_invalid, "type_fields_wrong": type_wrong,
        "final_type_hint_mismatch": type_mismatch, "usage_complete": bool(client.usage_complete),
        "observed_turn_delta": turns, "observed_token_delta": token_delta,
        "failure_code": "canary_contract_failed"}
    report["passed"] = bool(not result.failed and trace_valid and
        len(result.triples) == 2 and not result.markers and
        result.duplicate_triples_collapsed == 0 and not result.entity_property_hints and
        not type_mismatch and type_invalid == type_wrong == 0 and
        matched == 2 and turns == result.completion_calls and
        result.completion_calls == len(recording.requests) == len(pairs) and
        result.grounding_calls == len(ground) and
        client.usage_complete and type(start_tokens) is int and
        type(client.observed_tokens) is int and token_delta >= 0)
    if report["passed"]:
        report["failure_code"] = None
    try:
        validate_report(report, fixture_sha256=canary.EXTRACTION_CANARY_FIXTURE_SHA256)
    except ValueError:
        report["passed"] = False
        report["failure_code"] = "canary_contract_failed"
        validate_report(report, fixture_sha256=canary.EXTRACTION_CANARY_FIXTURE_SHA256)
    return report
