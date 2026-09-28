"""Pure offline scoring of pre-labelled compact-assessment checks.

No provider, retry, repair, file access or automatic semantic labels. Callers
must freeze and independently review gold before observing model responses.
Different views can overlap; their denominators must never be pooled.
"""
from __future__ import annotations

from collections import Counter

from benchmarks import digest_evidence_assessment as assessment


VERSION = "digest-assessment-check-evaluation-v1"
VIEWS = frozenset({"primary", "auxiliary", "retention", "legacy", "witness"})
VERDICTS = frozenset({"supported", "unsupported", "uncertain"})


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _resolve(scope: assessment.AssessmentRequest, selector: dict) -> str:
    _require(type(selector) is dict, "invalid_selector")
    kind = selector.get("kind")
    _require(type(kind) is str and kind in {"assertion", "relations", "outcome", "retention"},
             "invalid_check_kind")
    key = "chunk_id" if kind == "retention" else "field_path"
    _require(set(selector) == {"kind", key} and type(selector[key]) is str and bool(selector[key]),
             "invalid_selector_shape")
    if kind == "retention":
        sources = {source.source_id for source in scope.evidence_sources
                   if source.kind == "canonical_text" and source.chunk_id == selector[key]}
        matches = [check.check_id for check in scope.checks
                   if check.kind == kind and check.source_id in sources]
    else:
        fields = {field.field_id for field in scope.fields if field.path == selector[key]}
        matches = [check.check_id for check in scope.checks
                   if check.kind == kind and len(check.field_ids) == 1
                   and check.field_ids[0] in fields]
    _require(len(matches) == 1, "selector_must_resolve_exactly_once")
    return matches[0]


def resolve_selector(scope: assessment.AssessmentRequest, selector: dict) -> str:
    """Resolve semantic coordinates, not unstable positional cN/sN identifiers."""
    return _resolve(assessment._validate_scope(scope), selector)


def bind_labels(scope: assessment.AssessmentRequest, labels: list[dict]) -> tuple[dict, ...]:
    """Validate independently supplied gold; no inference from case names/text.

    A target is an explicitly reviewed conjunction of checks. A negative
    witness can name only the relevant defective field(s); an unrelated veto
    must not count as detection. Single-check labels use one selector. The
    same target/check may appear in distinct reporting views, not twice in one.
    """
    scope = assessment._validate_scope(scope)
    _require(type(labels) is list and 0 < len(labels) <= assessment.MAX_CHECKS,
             "invalid_label_count")
    seen_ids, seen_groups, seen_checks, single_gold, result = set(), set(), {}, {}, []
    for label in labels:
        _require(type(label) is dict and set(label) == {
            "id", "view", "selectors", "expected", "rationale"}, "invalid_label_shape")
        _require(type(label["id"]) is str and 0 < len(label["id"]) <= 256,
                 "invalid_label_id")
        _require(label["id"] not in seen_ids, "duplicate_label_id")
        seen_ids.add(label["id"])
        _require(type(label["view"]) is str and label["view"] in VIEWS, "invalid_label_view")
        _require(type(label["expected"]) is str and label["expected"] in {"supported", "unsupported"},
                 "invalid_gold_verdict")
        _require(type(label["rationale"]) is str and bool(label["rationale"].strip())
                 and len(label["rationale"]) <= 8192, "missing_gold_rationale")
        selectors = label["selectors"]
        _require(type(selectors) is list and 0 < len(selectors) <= len(scope.checks),
                 "invalid_selector_count")
        checks = tuple(_resolve(scope, selector) for selector in selectors)
        _require(len(set(checks)) == len(checks), "duplicate_target_check")
        group = (label["view"], frozenset(checks))
        _require(group not in seen_groups, "duplicate_target_in_view")
        seen_groups.add(group)
        used = seen_checks.setdefault(label["view"], set())
        _require(not used.intersection(checks), "overlapping_targets_in_view")
        used.update(checks)
        if label["view"] == "retention":
            _require(all(selector["kind"] == "retention" for selector in selectors),
                     "retention_view_requires_retention_checks")
        if len(checks) == 1:
            previous = single_gold.setdefault(checks[0], label["expected"])
            _require(previous == label["expected"], "conflicting_single_check_gold")
        result.append({"id": label["id"], "view": label["view"],
                       "check_ids": checks, "expected": label["expected"]})
    # A supported conjunction makes every member supported. It cannot coexist
    # with a negative singleton or a negative conjunction wholly inside it.
    supported = {check for row in result if row["expected"] == "supported"
                 for check in row["check_ids"]}
    _require(not any(row["expected"] == "unsupported" and set(row["check_ids"]) <= supported
                     for row in result), "contradictory_gold_conjunctions")
    return tuple(result)


def aggregate(verdicts: list[str]) -> str:
    """Conjunction with explicit abstention, never vacuous support."""
    _require(type(verdicts) is list and all(type(value) is str and value in VERDICTS
                                          for value in verdicts), "invalid_verdicts")
    if "unsupported" in verdicts:
        return "unsupported"
    return "uncertain" if not verdicts or "uncertain" in verdicts else "supported"


def score_scope(raw: object, scope: assessment.AssessmentRequest, labels: list[dict], *,
                max_output_chars: int = assessment.MAX_OUTPUT_CHARS) -> dict:
    """Parse the complete unmodified response; report targets without salvage.

    Malformed is a missed target, not a correct rejection. Untargeted negative
    or uncertain checks are metadata only. This score is neither an audited
    provider receipt nor publication authority; invocation/accounting audits
    belong to a separately frozen runner.
    """
    bound = bind_labels(scope, labels)  # Validate gold even if output is bad.
    outcome = assessment.parse_evidence_assessment(raw, scope, max_output_chars=max_output_chars)
    observed = {item.check_id: item.verdict for item in outcome.judgments}
    malformed = outcome.status == "malformed_assessment"
    rows = []
    for target in bound:
        value = "malformed" if malformed else aggregate([observed[c] for c in target["check_ids"]])
        vetoes = sorted(check for check, verdict in observed.items()
                        if check not in target["check_ids"] and verdict != "supported")
        rows.append({**target, "observed": value, "match": value == target["expected"],
                     "false_accept": target["expected"] == "unsupported" and value == "supported",
                     "false_reject": target["expected"] == "supported" and value == "unsupported",
                     "off_target_vetoes": vetoes,
                     "false_accept_masked_by_other_veto": target["expected"] == "unsupported"
                         and value == "supported" and bool(vetoes)})
    summaries = {}
    for view in sorted({row["view"] for row in rows}):
        selected = [row for row in rows if row["view"] == view]
        summaries[view] = {"targets": len(selected),
            "expected_supported": sum(row["expected"] == "supported" for row in selected),
            "expected_unsupported": sum(row["expected"] == "unsupported" for row in selected),
            "matches": sum(row["match"] for row in selected),
            "false_accepts": sum(row["false_accept"] for row in selected),
            "false_rejects": sum(row["false_reject"] for row in selected),
            "observed": dict(sorted(Counter(row["observed"] for row in selected).items()))}
    return {"version": VERSION, "scope_binding_sha256": scope.binding_sha256,
            "status": outcome.status, "targets": rows, "views": summaries,
            "semantic_verified": False, "publication_authorized": False}
