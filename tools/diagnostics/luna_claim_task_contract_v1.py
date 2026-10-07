"""Inactive, source-bound claim-task ablation contract.

Arm A is the exact classification-v3 wire contract. Arm B asks only about the
unchanged original claim. Neither arm changes production extraction behavior.
Model states are attestations; quote validation establishes provenance, not
semantic entailment. Callers must keep raw responses and ledgers private.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

from hymem.extraction import grounding_classification_v3 as v3
from hymem.extraction import grounding_v2 as v2
from hymem.extraction.jsonio import loads_exact_or_fenced
from hymem.extraction.llm import LLMRequest
from hymem.extraction.triples import Triple


B_SCHEMA = "source-grounding-claim-task-original-v1"
_ROOT_KEYS = frozenset(("schema", "batch_sha256", "complete", "classifications"))
_ITEM_KEYS = frozenset(("index", "original"))


def _replace_once(value: str, before: str, after: str) -> str:
    if value.count(before) != 1:
        raise RuntimeError("pinned_v3_prompt_drift")
    return value.replace(before, after)


def _original_only_system() -> str:
    """Keep v3 claim semantics and vocabulary; remove only alternative work."""
    system = v3._SYSTEM
    system = _replace_once(system,
        "If the original is supported or ambiguous, alternatives must be null. Only if the original is not_established, assess every other allowed predicate by name, replacing only the predicate while preserving all other fields and source. An original supported claim wins even if a different predicate might also apply. A correction requires exactly one supported alternative and explicit not_established assessments of every other alternative.",
        "Assess only the original claim. Do not assess or return alternative predicates or a correction.")
    system = _replace_once(system,
        "Return exactly {schema,batch_sha256,complete,classifications}; schema is source-grounding-classification-v3, copy batch_sha256, complete is true, and classifications are in index order. Each item is exactly {index,original,alternatives}.",
        f"Return exactly {{schema,batch_sha256,complete,classifications}}; schema is {B_SCHEMA}, copy batch_sha256, complete is true, and classifications are in index order. Each item is exactly {{index,original}}.")
    system = _replace_once(system,
        "Across all supported assessments for one claim use at most 8 distinct evidence items.",
        "For each supported original assessment use at most 8 distinct evidence items.")
    return system


B_SYSTEM = _original_only_system()


@dataclass(frozen=True)
class AssessmentResult:
    """Finite state plus immutable validated evidence and check-index ledger."""

    state: str
    evidence: tuple[v2.GroundingEvidence, ...]
    checks: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class ArmResult:
    """B has no alternative states or final correction verdicts."""

    arm: str
    batch_sha256: str
    original: tuple[AssessmentResult, ...]
    alternative_states: tuple[tuple[tuple[str, str], ...] | None, ...] | None
    final_verdicts: tuple[v2.GroundingVerdict, ...] | None


def _arm(arm: object) -> str:
    if type(arm) is not str or arm not in ("A", "B"):
        raise v2.GroundingContractError("arm:invalid")
    return arm


def build_arm_request(
    arm: str, triples: tuple[Triple, ...] | list[Triple],
    sources: tuple[v2.GroundingSource, ...] | list[v2.GroundingSource],
) -> tuple[LLMRequest, v3.ClassificationBatch]:
    """Build a trusted arm request. A's request is byte-for-byte v3."""
    arm = _arm(arm)
    request, batch = v3.build_grounding_request(triples, sources)
    if arm == "A":
        return request, batch
    return replace(request, system=B_SYSTEM), batch


def validate_arm_request(arm: str, request: LLMRequest, batch: v3.ClassificationBatch) -> None:
    """Bind arm, full canonical batch, hash, user bytes and generation settings."""
    arm = _arm(arm)
    expected = v3._checked(batch)
    if arm == "B":
        expected = replace(expected, system=B_SYSTEM)
    fields = ("system", "user", "response_format", "max_tokens", "temperature")
    if type(request) is not LLMRequest or any(
        type(getattr(request, field)) is not type(getattr(expected, field))
        or getattr(request, field) != getattr(expected, field) for field in fields
    ):
        raise v2.GroundingContractError("request:arm_binding")


def build_arm_output_schema(arm: str, batch: v3.ClassificationBatch) -> dict:
    """Give B a distinct bounded schema using only v3-supported JSON Schema forms."""
    arm = _arm(arm)
    schema = v3.build_output_schema(batch)
    if arm == "A":
        return schema
    schema = deepcopy(schema)
    schema["$defs"] = {key: value for key, value in schema["$defs"].items()
                       if key.startswith("assessment_")}
    schema["properties"]["schema"]["enum"] = [B_SCHEMA]
    for item in schema["properties"]["classifications"]["items"]["anyOf"]:
        item["required"] = ["index", "original"]
        del item["properties"]["alternatives"]
    return schema


def _assessment(value: dict) -> AssessmentResult:
    support = value["support"]
    if support is None:
        return AssessmentResult(value["state"], (), ())
    return AssessmentResult(
        value["state"],
        tuple(v2.GroundingEvidence(entry["source_message_id"], entry["region"], entry["quote"])
              for entry in support["evidence"]),
        tuple((name, tuple(support["checks"][name]["evidence_indices"]))
              for name in sorted(support["checks"])),
    )


def _payload(raw: object, batch: v3.ClassificationBatch, expected_schema: str) -> list[dict]:
    v3._checked(batch)
    if type(raw) is not str or len(raw) > v2.MAX_RESPONSE_CHARS:
        raise v2.GroundingContractError("response:bounds")
    try:
        payload = loads_exact_or_fenced(raw)
    except (RecursionError, OverflowError):
        raise v2.GroundingContractError("response:depth") from None
    if type(payload) is not dict or set(payload) != _ROOT_KEYS:
        raise v2.GroundingContractError("response:shape")
    if type(payload["schema"]) is not str or payload["schema"] != expected_schema:
        raise v2.GroundingContractError("response:schema")
    if type(payload["batch_sha256"]) is not str or payload["batch_sha256"] != batch.batch_sha256:
        raise v2.GroundingContractError("response:binding")
    if payload["complete"] is not True:
        raise v2.GroundingContractError("response:incomplete")
    items = payload["classifications"]
    if type(items) is not list or len(items) != len(batch.triples):
        raise v2.GroundingContractError("response:count")
    return items


def parse_arm_response(arm: str, raw: object, batch: v3.ClassificationBatch) -> ArmResult:
    """Validate response; B never selects or manufactures a correction."""
    arm = _arm(arm)
    if arm == "A":
        reviewed = v3.parse_grounding_response(raw, batch)
        items = _payload(raw, batch, v3.GROUNDING_CONTRACT_VERSION)
        original = tuple(_assessment(item["original"]) for item in items)
        alternative_states = tuple(
            None if item["alternatives"] is None else
            tuple((name, item["alternatives"][name]["state"])
                  for name in v3.PREDICATE_ORDER if name != batch.triples[index].predicate)
            for index, item in enumerate(items)
        )
        return ArmResult("A", batch.batch_sha256, original, alternative_states, reviewed.verdicts)

    items = _payload(raw, batch, B_SCHEMA)
    verdicts = []
    originals = []
    for index, item in enumerate(items):
        if type(item) is not dict or set(item) != _ITEM_KEYS:
            raise v2.GroundingContractError("classification:shape")
        if type(item["index"]) is not int or item["index"] != index:
            raise v2.GroundingContractError("classification:index")
        triple = batch.triples[index]
        state, evidence = v3._validate_assessment(item["original"], triple, triple.predicate)
        originals.append(_assessment(item["original"]))
        verdicts.append({
            "index": index,
            "status": {"supported": "supported", "not_established": "unsupported",
                       "ambiguous": "uncertain"}[state],
            "predicate": triple.predicate if state == "supported" else None,
            "evidence": evidence if evidence is not None else [],
        })
    _, v2_batch = v2.build_grounding_request(batch.triples, batch.sources)
    v2_raw = v3._serialize({"schema": v2.GROUNDING_CONTRACT_VERSION,
                            "batch_sha256": v2_batch.batch_sha256, "complete": True,
                            "verdicts": verdicts})
    v2.parse_grounding_response(v2_raw, v2_batch, allow_corrections=False)
    return ArmResult("B", batch.batch_sha256, tuple(originals), None, None)
