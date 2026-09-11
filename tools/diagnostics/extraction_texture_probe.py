"""Offline, synthetic-only controls for extraction texture experiments.

This module cannot call a model. Returned request copies belong to private
diagnostics, never canonical benchmark runs or a real source/proof store.
The CLI runs synthetic instrument checks and prints only aggregate metadata.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import random
import sys
from typing import Any, Mapping, Sequence


HEX = "0123456789abcdef"
MODES = ("baseline", "digits_to_hash", "letters_to_a", "digits_to_a",
         "flat_a", "fresh_hex")
ARMS = (*MODES, "table")
MAX_CHARS = 1_000_000
MAX_SEED_CHARS = 4_096
MAX_SCHEDULE = 10_000


class ProbeInputError(ValueError):
    """Messages are fixed codes, never supplied source or response text."""


def _integer(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def fresh_hex(length: int, *, seed: str) -> str:
    """Deterministic hex alphabet, not matched tokenization or class counts.

    Consume SHA-256 *hex characters themselves*. ord(hexdigest[i]) % 16 is
    not a nibble conversion: for a-f it returns 1-6, yielding digits only.
    """
    if not _integer(length) or length > MAX_CHARS or not isinstance(seed, str):
        raise ProbeInputError("invalid_hex_parameters")
    if len(seed) > MAX_SEED_CHARS:
        raise ProbeInputError("seed_limit")
    return "".join(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()
                   for i in range((length + 63) // 64))[:length]


def mutate_padding(text: str, *, padding_span: tuple[int, int],
                   protected_claim: str, mode: str, seed: str = "0") -> str:
    """Change only an explicitly bounded lowercase-hex padding span.

    The claim must occur exactly once, with no overlap with the padding.
    Every byte outside the span is kept, including the claim and separators.
    Length here means characters/UTF-8 bytes, never tokens.
    """
    if not isinstance(text, str) or len(text) > MAX_CHARS:
        raise ProbeInputError("invalid_text")
    if mode not in MODES:
        raise ProbeInputError("invalid_mode")
    if (not isinstance(protected_claim, str) or not protected_claim
            or text.find(protected_claim) < 0
            or text.find(protected_claim, text.find(protected_claim) + 1) >= 0):
        raise ProbeInputError("claim_not_unique")
    if (not isinstance(padding_span, tuple) or len(padding_span) != 2
            or not all(_integer(x) for x in padding_span)):
        raise ProbeInputError("invalid_padding_span")
    start, end = padding_span
    claim_start = text.index(protected_claim)
    if not (start < end <= len(text)):
        raise ProbeInputError("invalid_padding_span")
    if start < claim_start + len(protected_claim) and end > claim_start:
        raise ProbeInputError("padding_overlaps_claim")
    padding = text[start:end]
    if any(char not in HEX for char in padding):
        raise ProbeInputError("padding_not_lowercase_hex")
    if mode == "fresh_hex":
        changed = fresh_hex(len(padding), seed=seed)
    else:
        changed = "".join(
            "#" if mode == "digits_to_hash" and char.isdigit() else
            "a" if mode == "letters_to_a" and char in "abcdef" else
            "a" if mode == "digits_to_a" and char.isdigit() else
            "a" if mode == "flat_a" else char for char in padding
        )
    result = text[:start] + changed + text[end:]
    # ASCII-for-ASCII padding substitution also preserves byte offsets.
    if len(result.encode("utf-8")) != len(text.encode("utf-8")):
        raise ProbeInputError("mutation_length_mismatch")
    return result


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProbeInputError("duplicate_json_key")
        value[key] = item
    return value


def _reject_constant(_: str) -> None:
    raise ProbeInputError("nonfinite_json")


def _encode(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def mutate_request(request: Mapping[str, Any], *, record_index: int,
                   field_path: tuple[str, ...], padding_span: tuple[int, int],
                   protected_claim: str, mode: str, seed: str = "0") -> dict[str, Any]:
    """Copy every captured request field and change one explicit padding span.

    Accept only exact compact/sorted JSON-line source records in the primary
    Excerpt block. Reject serialization drift, duplicate keys and ambiguous
    delimiters; do not silently rebuild arbitrary prompts. This does not turn
    synthetic variants into valid canonical source/provenance records.
    """
    if not isinstance(request, Mapping) or not isinstance(request.get("user"), str):
        raise ProbeInputError("invalid_request")
    user = request["user"]
    opener, closer = 'Excerpt:\n"""\n', '\n"""'
    if (len(user) > MAX_CHARS or not user.startswith(opener)
            or user.count(opener) != 1 or user[len(opener):].count(closer) != 1):
        raise ProbeInputError("unsupported_excerpt")
    body, suffix = user[len(opener):].split(closer)
    lines = body.split("\n")
    if not _integer(record_index) or record_index >= len(lines):
        raise ProbeInputError("invalid_record_index")
    if field_path not in (("content",), ("source_boundary_context", "content")):
        raise ProbeInputError("unsupported_field_path")
    try:
        records = [json.loads(line, object_pairs_hook=_pairs,
                              parse_constant=_reject_constant) for line in lines]
        if (any(not isinstance(record, dict) for record in records)
                or [_encode(record) for record in records] != lines):
            raise ProbeInputError("source_roundtrip_mismatch")
    except (ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, ProbeInputError):
            raise
        raise ProbeInputError("invalid_source_json") from None
    node = records[record_index]
    for field in field_path[:-1]:
        if not isinstance(node.get(field), dict):
            raise ProbeInputError("missing_field")
        node = node[field]
    if field_path[-1] not in node:
        raise ProbeInputError("missing_field")
    node[field_path[-1]] = mutate_padding(
        node[field_path[-1]], padding_span=padding_span,
        protected_claim=protected_claim, mode=mode, seed=seed,
    )
    new_lines = list(lines)
    new_lines[record_index] = _encode(records[record_index])
    result = deepcopy(dict(request))
    result["user"] = opener + "\n".join(new_lines) + closer + suffix
    if len(result["user"].encode()) != len(user.encode()):
        raise ProbeInputError("request_length_mismatch")
    return result


@dataclass(frozen=True)
class ScheduledArm:
    repetition: int
    arm: str
    planned_delay_before_seconds: float


def build_schedule(arms: Sequence[str], *, repetitions: int, seed: int,
                   spacing_seconds: float) -> tuple[ScheduledArm, ...]:
    """Plan randomized interleaved repetitions; DOES NOT execute or sleep.

    Any future external executor must actually enforce these minimum delays,
    record monotonic start/end times and close clients on every exit path.
    A schedule alone is not evidence that spacing occurred.
    """
    if (not arms or len(arms) > len(ARMS) or any(arm not in ARMS for arm in arms)
            or len(set(arms)) != len(arms) or not _integer(repetitions, minimum=1)
            or len(arms) * repetitions > MAX_SCHEDULE or type(seed) is not int
            or seed.bit_length() > 256
            or isinstance(spacing_seconds, bool)
            or not isinstance(spacing_seconds, (int, float))
            or spacing_seconds < 0 or spacing_seconds > 3_600
            or not math.isfinite(spacing_seconds)):
        raise ProbeInputError("invalid_schedule")
    rng = random.Random(seed)
    result = []
    for repetition in range(repetitions):
        order = list(arms)
        rng.shuffle(order)
        for arm in order:
            result.append(ScheduledArm(repetition, arm,
                                       float(spacing_seconds) if result else 0.0))
    return tuple(result)


@dataclass(frozen=True)
class Observation:
    classification: str
    expected_claim_present: bool


def classify_observation(*, contract_valid: bool, complete: bool,
                         triple_count: int, marker_count: int,
                         matched_claim_indexes: Sequence[int],
                         expected_claim_index: int) -> Observation:
    """Summarize an EXISTING strict validator's output, never raw responses.

    Emission is not correctness. Supply the expected index for each arm: the
    canonical table control is 0 and the prose control is 1, not one global 1.
    This is deliberately not another parser or replacement contract validator.
    """
    if (type(contract_valid) is not bool or type(complete) is not bool
            or not _integer(triple_count) or not _integer(marker_count)
            or not _integer(expected_claim_index)
            or not isinstance(matched_claim_indexes, (list, tuple))
            or any(not _integer(x) for x in matched_claim_indexes)
            or len(matched_claim_indexes) > triple_count):
        raise ProbeInputError("invalid_observation")
    if not contract_valid:
        return Observation("contract_failure", False)
    if not complete:
        return Observation("incomplete", False)
    expected = expected_claim_index in matched_claim_indexes
    if not triple_count and not marker_count:
        return Observation("empty", False)
    return Observation("emitted_expected" if expected else "emitted_other", expected)


def aggregate_observations(observations: Sequence[Observation]) -> dict[str, int]:
    """Fixed keys/counts only; no raw text, identifiers or exception payloads."""
    if len(observations) > MAX_SCHEDULE:
        raise ProbeInputError("observation_limit")
    counts = Counter()
    allowed = {"contract_failure", "incomplete", "empty", "emitted_expected", "emitted_other"}
    for item in observations:
        if (not isinstance(item, Observation) or item.classification not in allowed
                or type(item.expected_claim_present) is not bool
                or item.expected_claim_present != (item.classification == "emitted_expected")):
            raise ProbeInputError("invalid_observation")
        counts[item.classification] += 1
    return {"observations": len(observations), **{key: counts[key] for key in sorted(allowed)}}


def main(argv: Sequence[str] | None = None) -> int:
    """No arguments, input files, network imports or model calls are supported."""
    if list(sys.argv[1:] if argv is None else argv):
        print('{"status":"error","code":"no_arguments_supported","model_calls":0}')
        return 2
    actual = fresh_hex(256, seed="instrument-control-v1")
    passed = set(actual) == set(HEX)
    print(json.dumps({"status": "pass" if passed else "fail",
                      "mode": "offline_instrument_check", "model_calls": 0,
                      "hex_alphabet_symbols": len(set(actual))}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
