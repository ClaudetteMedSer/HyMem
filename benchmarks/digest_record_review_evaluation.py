"""Pure scoring of independently labelled record-review-v3 checks.

This diagnostic requires fresh explicit facet labels, never converts legacy
relations gold, and never infers semantic truth. Overlapping reporting views
retain separate denominators. It performs no provider, file, network, repair,
retry, runtime or publication work.
"""
from __future__ import annotations

from benchmarks import digest_record_review as review
from benchmarks import digest_source_review_evaluation as source_evaluation


VERSION = "digest-record-review-evaluation-v1"
VIEWS = source_evaluation.VIEWS
GROUNDING_VERDICTS = source_evaluation.GROUNDING_VERDICTS
RETENTION_GOLD = source_evaluation.RETENTION_GOLD
_GROUNDING_KINDS = frozenset({"assertion", "outcome", "actor_attribution", "identity",
                              "quantified_scope", "residual_relations"})
_RETENTION_ACCEPTS = frozenset({"retained", "not_applicable"})
_RETENTION_DEFECTS = frozenset({"omitted", "altered"})


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _resolve(scope: review.RecordReviewScope, selector: dict) -> str:
    _require(type(selector) is dict, "invalid_selector")
    kind = selector.get("kind")
    _require(type(kind) is str and kind in _GROUNDING_KINDS | {"retention"},
             "invalid_check_kind")
    coordinate = "chunk_id" if kind == "retention" else "field_path"
    keys = {"kind", coordinate, "facet"} if kind == "retention" else {"kind", coordinate}
    _require(set(selector) == keys and type(selector[coordinate]) is str
             and 0 < len(selector[coordinate]) <= review.MAX_INPUT_CHARS,
             "invalid_selector_shape")
    if kind == "retention":
        _require(type(selector["facet"]) is str and selector["facet"] in review.RETENTION_FACETS,
                 "invalid_retention_facet")
        sources = {source.source_id for source in scope.canonical_sources
                   if source.chunk_id == selector[coordinate]}
        matches = [check.check_id for check in scope.checks
                   if check.kind == "retention" and check.source_id in sources
                   and check.facet == selector["facet"]]
    else:
        fields = {field.field_id for field in scope.fields if field.path == selector[coordinate]}
        matches = [check.check_id for check in scope.checks
                   if check.kind == kind and len(check.field_ids) == 1
                   and check.field_ids[0] in fields]
    _require(len(matches) == 1, "selector_must_resolve_exactly_once")
    return matches[0]


def resolve_selector(scope: review.RecordReviewScope, selector: dict) -> str:
    """Bind an exact explicit facet coordinate in a fully validated v3 scope."""
    return _resolve(review._validate_scope(scope), selector)


def bind_labels(scope: review.RecordReviewScope, labels: list[dict]) -> tuple[dict, ...]:
    """Validate caller-reviewed gold before observing any output.

    Grounding selectors form a conjunction; retention labels require exactly
    one canonical-source facet. Consistent overlap across views is allowed,
    but within-view duplicate denominators and contradictory gold are not.
    Rationale is mandatory, never generated or inferred by this module.
    """
    scope = review._validate_scope(scope)
    _require(type(labels) is list and 0 < len(labels) <= review.MAX_CHECKS,
             "invalid_label_count")
    seen_ids, used_by_view, singleton_gold, result = set(), {}, {}, []
    for label in labels:
        _require(type(label) is dict and set(label) == {
            "id", "view", "selectors", "expected", "rationale"}, "invalid_label_shape")
        _require(type(label["id"]) is str and bool(label["id"].strip())
                 and len(label["id"]) <= 256, "invalid_label_id")
        _require(label["id"] not in seen_ids, "duplicate_label_id")
        seen_ids.add(label["id"])
        _require(type(label["view"]) is str and label["view"] in VIEWS, "invalid_label_view")
        _require(type(label["rationale"]) is str and bool(label["rationale"].strip())
                 and len(label["rationale"]) <= 8192, "missing_gold_rationale")
        selectors = label["selectors"]
        _require(type(selectors) is list and 0 < len(selectors) <= len(scope.checks),
                 "invalid_selector_count")
        checks = tuple(_resolve(scope, selector) for selector in selectors)
        _require(len(set(checks)) == len(checks), "duplicate_target_check")
        domains = {"retention" if selector["kind"] == "retention" else "grounding"
                   for selector in selectors}
        _require(len(domains) == 1, "mixed_target_domains")
        domain = next(iter(domains))
        _require(domain != "retention" or len(checks) == 1,
                 "retention_target_requires_one_facet")
        allowed_gold = RETENTION_GOLD if domain == "retention" else {"supported", "unsupported"}
        _require(type(label["expected"]) is str and label["expected"] in allowed_gold,
                 "invalid_gold_verdict")
        _require(label["view"] != "retention" or domain == "retention",
                 "retention_view_requires_retention_checks")
        _require(label["view"] != "grounding" or domain == "grounding",
                 "grounding_view_requires_grounding_checks")
        used = used_by_view.setdefault(label["view"], set())
        _require(not used.intersection(checks), "overlapping_targets_in_view")
        used.update(checks)
        if len(checks) == 1:
            previous = singleton_gold.setdefault(checks[0], label["expected"])
            _require(previous == label["expected"], "conflicting_single_check_gold")
        result.append({"id": label["id"], "view": label["view"], "domain": domain,
                       "check_ids": checks, "expected": label["expected"]})
    # Supported conjunctions entail every member. Their union cannot cover an
    # explicitly unsupported conjunction without contradicting its gold.
    positive = {check for row in result
                if row["domain"] == "grounding" and row["expected"] == "supported"
                for check in row["check_ids"]}
    _require(not any(row["domain"] == "grounding" and row["expected"] == "unsupported"
                     and set(row["check_ids"]) <= positive for row in result),
             "contradictory_gold_conjunctions")
    return tuple(result)


def aggregate_grounding(verdicts: list[str]) -> str:
    """Conjoin verdicts without vacuous support or credit for abstention."""
    return source_evaluation.aggregate_grounding(verdicts)


def score_scope(raw: object, scope: review.RecordReviewScope, labels: list[dict], *,
                max_output_chars: int = review.MAX_OUTPUT_CHARS) -> dict:
    """Score a complete untouched response, without partial-scope salvage.

    Malformed replies make every target malformed, not a detected defect.
    Uncertainty is an exact-match miss. Off-target vetoes never rescue wrong
    target acceptance. Execution/provider accounting remains a separate task.
    """
    bound = bind_labels(scope, labels)  # Invalid output never excuses bad gold.
    outcome = review.parse_record_review(raw, scope, max_output_chars=max_output_chars)
    observed = {item.check_id: item.verdict for item in outcome.judgments}
    vetoes = {item.check_id for item in outcome.judgments
              if (item.verdict not in _RETENTION_ACCEPTS if item.kind == "retention"
                  else item.verdict != "supported")}
    malformed = outcome.status == "malformed_review"
    rows = []
    for target in bound:
        domain, expected = target["domain"], target["expected"]
        value = ("malformed" if malformed else observed[target["check_ids"][0]]
                 if domain == "retention" else
                 aggregate_grounding([observed[check] for check in target["check_ids"]]))
        if domain == "retention":
            false_accept = expected in _RETENTION_DEFECTS and value in _RETENTION_ACCEPTS
            false_reject = expected in _RETENTION_ACCEPTS and value in _RETENTION_DEFECTS
        else:
            false_accept = expected == "unsupported" and value == "supported"
            false_reject = expected == "supported" and value == "unsupported"
        other_vetoes = sorted(vetoes.difference(target["check_ids"]))
        rows.append({**target, "observed": value, "match": value == expected,
            "false_accept": false_accept, "false_reject": false_reject,
            "false_not_applicable": domain == "retention" and expected != "not_applicable"
                and value == "not_applicable",
            "false_applicable": domain == "retention" and expected == "not_applicable"
                and value in {"retained", "omitted", "altered"},
            "defect_kind_confusion": domain == "retention" and expected in _RETENTION_DEFECTS
                and value in _RETENTION_DEFECTS and value != expected,
            "off_target_vetoes": other_vetoes,
            "false_accept_masked_by_other_veto": false_accept and bool(other_vetoes)})
    views = {}
    for view in sorted({row["view"] for row in rows}):
        views[view] = {}
        for domain in sorted({row["domain"] for row in rows if row["view"] == view}):
            views[view][domain] = source_evaluation._counts(
                [row for row in rows if row["view"] == view and row["domain"] == domain])
    return {"version": VERSION, "scope_binding_sha256": scope.binding_sha256,
            "status": outcome.status, "targets": rows, "views": views,
            "semantic_verified": False, "publication_authorized": False}
