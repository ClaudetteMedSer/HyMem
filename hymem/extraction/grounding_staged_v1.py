"""Inactive two-stage claim-first classification contract.

Each stage is bound to the same trusted v4 batch. Alternative assessments are
permitted only after an actual, validated original response is negative.
"""
from __future__ import annotations

from hymem.contrib.implementation_identity import import_time_source_sha256

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)

import copy
import hashlib
import json
from dataclasses import dataclass, replace

from hymem.extraction import grounding_classification_v4 as v4
from hymem.extraction.grounding_v2 import (
    GroundingContractError, GroundingReview, MAX_RESPONSE_CHARS,
    build_grounding_request as _build_v2,
    parse_grounding_response as _parse_v2,
)
from hymem.extraction.jsonio import loads_exact_or_fenced
from hymem.extraction.llm import LLMRequest

ORIGINAL_SCHEMA = "source-grounding-staged-original-v1"
ALTERNATIVES_SCHEMA = "source-grounding-staged-alternatives-v1"
_ORIGINAL_SYSTEM = v4._SYSTEM.replace(
    "If the original is supported or ambiguous, alternatives must be null. Only if the original is not_established, assess every other allowed predicate by name, replacing only the predicate while preserving all other fields and source. An original supported claim wins even if a different predicate might also apply. A correction requires exactly one supported alternative and explicit not_established assessments of every other alternative.",
    "Assess only the original predicate for every claim. Never assess or return alternatives in this stage. A later request may ask about alternatives only for validated not_established originals."
).replace(
    "Return exactly {schema,batch_sha256,complete,classifications}; schema is source-grounding-classification-v4, copy batch_sha256, complete is true, and classifications are in index order. Each item is exactly {index,original,alternatives}.",
    "Return exactly {schema,batch_sha256,complete,originals}; schema is source-grounding-staged-original-v1, copy batch_sha256, complete is true, and originals are in index order. Each item is exactly {index,original}."
)
_ALTERNATIVES_SYSTEM = v4._SYSTEM.replace(
    "Assess the exact original complete claim first against its cited owned source.",
    "The original complete claim was assessed in a prior validated stage. Assess predicate-only alternatives only for the listed negative_indices against each cited owned source."
).replace(
    "If the original is supported or ambiguous, alternatives must be null. Only if the original is not_established, assess every other allowed predicate by name, replacing only the predicate while preserving all other fields and source. An original supported claim wins even if a different predicate might also apply. A correction requires exactly one supported alternative and explicit not_established assessments of every other alternative.",
    "For each listed index assess every other allowed predicate by name, replacing only the predicate while preserving all other fields and source. A correction requires exactly one supported alternative and explicit not_established assessments of every other alternative."
).replace(
    "Return exactly {schema,batch_sha256,complete,classifications}; schema is source-grounding-classification-v4, copy batch_sha256, complete is true, and classifications are in index order. Each item is exactly {index,original,alternatives}.",
    "Return exactly {schema,batch_sha256,original_response_sha256,complete,alternatives}; schema is source-grounding-staged-alternatives-v1; copy both hashes; complete is true; alternatives has only the listed indices in order. Each item is exactly {index,alternatives}."
)


def _fail(code: str) -> None:
    raise GroundingContractError(code)


def _serialize(value: object) -> str:
    return v4._serialize(value)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load(raw: object) -> dict:
    if type(raw) is not str or len(raw) > MAX_RESPONSE_CHARS:
        _fail("response:bounds")
    try:
        payload = loads_exact_or_fenced(raw)
    except (RecursionError, OverflowError):
        _fail("response:depth")
    if type(payload) is not dict:
        _fail("response:shape")
    return payload


def _header(payload: dict, schema: str, batch: v4.ClassificationBatch, keys: set[str]) -> None:
    if set(payload) != keys:
        _fail("response:shape")
    if type(payload["schema"]) is not str or payload["schema"] != schema:
        _fail("response:schema")
    if type(payload["batch_sha256"]) is not str or payload["batch_sha256"] != batch.batch_sha256:
        _fail("response:binding")
    if payload["complete"] is not True:
        _fail("response:incomplete")


def _validate_supported(assessment: dict, batch: v4.ClassificationBatch, index: int, predicate: str) -> None:
    state, evidence = v4._validate_assessment(assessment, batch.triples[index], predicate)
    if state != "supported":
        return
    trial = replace(batch.triples[index], predicate=predicate)
    _, trial_batch = _build_v2((trial,), batch.sources)
    raw = _serialize({"schema": "source-grounding-v2", "batch_sha256": trial_batch.batch_sha256,
                      "complete": True, "verdicts": [{"index": 0, "status": "supported",
                      "predicate": predicate, "evidence": evidence}]})
    _parse_v2(raw, trial_batch)


def _original(raw: object, batch: v4.ClassificationBatch) -> tuple[dict, tuple[int, ...], str]:
    v4._checked(batch)
    payload = _load(raw)
    _header(payload, ORIGINAL_SCHEMA, batch, {"schema", "batch_sha256", "complete", "originals"})
    rows = payload["originals"]
    if type(rows) is not list or len(rows) != len(batch.triples):
        _fail("response:count")
    negatives = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"index", "original"}:
            _fail("original:shape")
        if type(row["index"]) is not int or row["index"] != index:
            _fail("original:index")
        assessment = row["original"]
        state, _ = v4._validate_assessment(assessment, batch.triples[index], batch.triples[index].predicate)
        if state == "supported":
            _validate_supported(assessment, batch, index, batch.triples[index].predicate)
        elif state == "not_established":
            negatives.append(index)
    canonical = _serialize(payload)
    return payload, tuple(negatives), _digest(canonical)


@dataclass(frozen=True)
class OriginalReview:
    batch_sha256: str
    states: tuple[str, ...]
    negative_indices: tuple[int, ...]
    canonical_json: str
    response_sha256: str


def parse_original_response(raw: str, batch: v4.ClassificationBatch) -> OriginalReview:
    """Validate actual original assessments, including every supported ledger."""
    payload, negatives, digest = _original(raw, batch)
    return OriginalReview(batch.batch_sha256,
                          tuple(row["original"]["state"] for row in payload["originals"]),
                          negatives, _serialize(payload), digest)


def build_original_request(triples, sources) -> tuple[LLMRequest, v4.ClassificationBatch]:
    request, batch = v4.build_grounding_request(triples, sources)
    return replace(request, system=_ORIGINAL_SYSTEM), batch


def validate_original_request(request: LLMRequest, batch: v4.ClassificationBatch) -> None:
    expected = replace(v4._checked(batch), system=_ORIGINAL_SYSTEM)
    _validate_request(request, expected)


def _validate_request(request: LLMRequest, expected: LLMRequest) -> None:
    fields = ("system", "user", "response_format", "max_tokens", "temperature")
    if type(request) is not LLMRequest or any(
        type(getattr(request, field)) is not type(getattr(expected, field))
        or getattr(request, field) != getattr(expected, field) for field in fields
    ):
        _fail("request:binding")


def build_original_output_schema(batch: v4.ClassificationBatch) -> dict:
    schema = v4.build_output_schema(batch)
    refs = schema["properties"]["classifications"]["items"]["anyOf"]
    items = [{"type": "object", "additionalProperties": False,
              "required": ["index", "original"],
              "properties": {key: copy.deepcopy(ref["properties"][key]) for key in ("index", "original")}}
             for ref in refs]
    schema["$defs"] = {key: value for key, value in schema["$defs"].items() if key.startswith("assessment_")}
    schema["required"] = ["schema", "batch_sha256", "complete", "originals"]
    schema["properties"]["schema"]["enum"] = [ORIGINAL_SCHEMA]
    schema["properties"]["originals"] = {"type": "array", "minItems": len(items),
                                            "maxItems": len(items), "items": {"anyOf": items}}
    del schema["properties"]["classifications"]
    return schema


@dataclass(frozen=True)
class AlternativesBatch:
    classification_batch: v4.ClassificationBatch
    original_response_canonical_json: str
    original_response_sha256: str
    negative_indices: tuple[int, ...]


def _checked_alternatives(batch: AlternativesBatch) -> None:
    if type(batch) is not AlternativesBatch:
        _fail("alternatives_batch:type")
    payload, negatives, digest = _original(batch.original_response_canonical_json, batch.classification_batch)
    if (_serialize(payload) != batch.original_response_canonical_json
            or digest != batch.original_response_sha256
            or type(batch.negative_indices) is not tuple
            or any(type(index) is not int for index in batch.negative_indices)
            or batch.negative_indices != negatives or not negatives):
        _fail("alternatives_batch:binding")


def _alternatives_request(batch: AlternativesBatch) -> LLMRequest:
    base = v4._checked(batch.classification_batch)
    user = json.loads(base.user)
    user["negative_indices"] = list(batch.negative_indices)
    user["original_response_sha256"] = batch.original_response_sha256
    return replace(base, system=_ALTERNATIVES_SYSTEM, user=_serialize(user))


def build_alternatives_request(classification_batch: v4.ClassificationBatch, actual_original_raw: str) -> tuple[LLMRequest, AlternativesBatch]:
    payload, negatives, digest = _original(actual_original_raw, classification_batch)
    if not negatives:
        _fail("alternatives:none_required")
    batch = AlternativesBatch(classification_batch, _serialize(payload), digest, negatives)
    return _alternatives_request(batch), batch


def validate_alternatives_request(request: LLMRequest, batch: AlternativesBatch) -> None:
    _checked_alternatives(batch)
    _validate_request(request, _alternatives_request(batch))


def build_alternatives_output_schema(batch: AlternativesBatch) -> dict:
    _checked_alternatives(batch)
    full = v4.build_output_schema(batch.classification_batch)
    refs = full["properties"]["classifications"]["items"]["anyOf"]
    items = [{"type": "object", "additionalProperties": False,
              "required": ["index", "alternatives"],
              "properties": {"index": copy.deepcopy(refs[i]["properties"]["index"]),
                             "alternatives": {"$ref": f"#/$defs/alternatives_{i}"}}}
             for i in batch.negative_indices]
    defs = {key: value for key, value in full["$defs"].items()
            if any(key == f"assessment_{i}" or key == f"alternatives_{i}" for i in batch.negative_indices)}
    return {"$schema": full["$schema"], "$defs": defs, "type": "object",
            "additionalProperties": False,
            "required": ["schema", "batch_sha256", "original_response_sha256", "complete", "alternatives"],
            "properties": {"schema": {"type": "string", "enum": [ALTERNATIVES_SCHEMA]},
                           "batch_sha256": {"type": "string", "enum": [batch.classification_batch.batch_sha256]},
                           "original_response_sha256": {"type": "string", "enum": [batch.original_response_sha256]},
                           "complete": {"type": "boolean", "enum": [True]},
                           "alternatives": {"type": "array", "minItems": len(items),
                                            "maxItems": len(items), "items": {"anyOf": items}}}}


def parse_staged_responses(classification_batch: v4.ClassificationBatch, original_raw: str,
                           alternatives_raw: str | None, *, allow_corrections: bool = True) -> GroundingReview:
    original, negatives, digest = _original(original_raw, classification_batch)
    if type(allow_corrections) is not bool:
        _fail("response:correction_flag")
    alternative_rows = {}
    if not negatives:
        if alternatives_raw is not None:
            _fail("alternatives:unexpected")
    else:
        if alternatives_raw is None:
            _fail("alternatives:required")
        payload = _load(alternatives_raw)
        _header(payload, ALTERNATIVES_SCHEMA, classification_batch,
                {"schema", "batch_sha256", "original_response_sha256", "complete", "alternatives"})
        if type(payload["original_response_sha256"]) is not str or payload["original_response_sha256"] != digest:
            _fail("alternatives:prior_binding")
        rows = payload["alternatives"]
        if type(rows) is not list or len(rows) != len(negatives):
            _fail("alternatives:count")
        for position, index in enumerate(negatives):
            row = rows[position]
            if type(row) is not dict or set(row) != {"index", "alternatives"}:
                _fail("alternatives:shape")
            if type(row["index"]) is not int or row["index"] != index:
                _fail("alternatives:index")
            assessments = row["alternatives"]
            expected = set(v4.PREDICATE_ORDER) - {classification_batch.triples[index].predicate}
            if type(assessments) is not dict or set(assessments) != expected:
                _fail("alternatives:coverage")
            for predicate, assessment in assessments.items():
                v4._validate_assessment(assessment, classification_batch.triples[index], predicate)
            alternative_rows[index] = assessments
    combined = {"schema": v4.GROUNDING_CONTRACT_VERSION,
                "batch_sha256": classification_batch.batch_sha256, "complete": True,
                "classifications": [
                    {"index": index, "original": row["original"],
                     "alternatives": alternative_rows.get(index)}
                    for index, row in enumerate(original["originals"])]}
    return v4.parse_grounding_response(_serialize(combined), classification_batch,
                                       allow_corrections=allow_corrections)
