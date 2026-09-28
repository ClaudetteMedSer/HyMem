"""Offline retention of summary-owned NEW sources only; never digest acceptance.

Episode/procedure citations establish support, not exhaustive item ownership.
This boundary therefore selects only the original summary projection, without
inventing item-level retention scopes or a union of their citations. The complete
input and all its projections are still validated before selection.

The validation plan's call cap is a structural planning limit, NOT authorization
or reservation for item LLM calls. This API reserves exactly two caller-owned
invocations for one summary. HTTP accounting, supervision and absolute deadlines
remain the caller's responsibility. No provider, retry, repair or runtime wiring
is implemented here. Prior-summary continuity and grounding are NOT assessed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from benchmarks import digest_retention_inventory as retention
from hymem.deadline import DeadlineExceeded
from hymem.extraction.llm import LLMClient, LLMRequest


VERSION = "digest-summary-new-source-retention-v1"
COVERAGE_SCOPE = "summary_new_sources_only"
MAX_CALLS = 2
MAX_INPUT_CHARS = retention.MAX_INPUT_CHARS
MAX_OUTPUT_CHARS = retention.MAX_OUTPUT_CHARS


@dataclass(frozen=True, slots=True)
class _SummaryOnly:
    version: str = field(default=VERSION, init=False)
    coverage_scope: str = field(default=COVERAGE_SCOPE, init=False)
    prior_continuity_assessed: bool = field(default=False, init=False)
    grounding_assessed: bool = field(default=False, init=False)
    item_retention_assessed: bool = field(default=False, init=False)

    @property
    def semantic_verified(self) -> bool:
        return False

    @property
    def publication_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SummaryRetentionPlan(_SummaryOnly):
    binding_sha256: str
    max_calls: int
    reserved_calls: int
    inventory_scope: retention.InventoryScope
    # Reconstruction evidence only; its item requests are never executed here.
    validation_plan: retention.record_review.RecordReviewPlan


@dataclass(frozen=True, slots=True)
class SummaryMatchingScope(_SummaryOnly):
    binding_sha256: str
    plan_binding_sha256: str
    matching_scope: retention.MatchingScope


@dataclass(frozen=True, slots=True)
class SummaryRetentionResult(_SummaryOnly):
    plan_binding_sha256: str
    inventory: retention.InventoryOutcome
    matching: retention.MatchingOutcome
    reserved_calls: int
    attempted_calls: int
    # Both required replies returned; validity and retention are separate.
    collection_complete: bool
    summary_structure_valid: bool
    summary_model_retention_satisfied: bool
    halted_reason: str | None


def _body(value: SummaryRetentionPlan | SummaryMatchingScope) -> dict:
    body = asdict(value)
    del body["binding_sha256"]
    return body


def _from_record(validation_plan: retention.record_review.RecordReviewPlan) -> SummaryRetentionPlan:
    summaries = tuple(scope for scope in validation_plan.requests if scope.kind == "summary")
    if len(summaries) != 1:
        raise ValueError("exactly one original summary scope is required")
    scope = retention._make_scope(summaries[0], validation_plan.max_input_chars,
                                  validation_plan.max_output_chars)
    if not scope.units:
        raise ValueError("summary new sources must contain nonempty canonical text")
    plan = SummaryRetentionPlan("", MAX_CALLS, MAX_CALLS, scope, validation_plan)
    return replace(plan, binding_sha256=retention._sha(_body(plan)))


def prepare_summary_retention(
    payload: dict, request_template: LLMRequest, *, max_calls: int = MAX_CALLS,
    max_input_chars: int = MAX_INPUT_CHARS, max_output_chars: int = MAX_OUTPUT_CHARS,
) -> SummaryRetentionPlan:
    """Validate every input projection, then reserve two summary-only calls.

    The exact original summary.new_source_ids determine source authority; item
    citations are never promoted to exhaustive coverage. The structural record
    plan is not a reservation to call any of its item requests.
    """
    retention._integer(max_calls, "summary retention call cap", MAX_CALLS, MAX_CALLS)
    retention._integer(max_input_chars, "input cap", 1, MAX_INPUT_CHARS)
    retention._integer(max_output_chars, "output cap", 1, MAX_OUTPUT_CHARS)
    validation_plan = retention.record_review.prepare_record_review(
        payload, request_template, max_calls=retention.record_review.MAX_CALLS,
        max_input_chars=max_input_chars, max_output_chars=max_output_chars)
    return _from_record(validation_plan)


def _preflight(plan: object) -> SummaryRetentionPlan:
    if type(plan) is not SummaryRetentionPlan:
        raise ValueError("a summary-only retention plan is required")
    retention._digest(plan.binding_sha256)
    retention._integer(plan.max_calls, "summary call cap", MAX_CALLS, MAX_CALLS)
    retention._integer(plan.reserved_calls, "summary reservation", MAX_CALLS, MAX_CALLS)
    try:
        # Rebuild from the full original payload before trusting the selected
        # scope. Hashing an attacker-rewritten subset cannot authorize it.
        validated = retention.record_review._preflight(plan.validation_plan)
        retention._validate_scope(plan.inventory_scope)
        rebuilt = _from_record(validated)
        if not retention._equal(rebuilt, plan):
            raise ValueError("summary retention plan binding mismatch")
    except (KeyError, IndexError, TypeError, AttributeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid summary retention plan") from exc
    return plan


def parse_summary_inventory(raw: object, plan: SummaryRetentionPlan) -> retention.InventoryOutcome:
    """Parse source-only model inventory; no item or semantic acceptance."""
    return retention.parse_inventory(raw, _preflight(plan).inventory_scope)


def _wrap_matching(plan: SummaryRetentionPlan,
                   matching: retention.MatchingScope) -> SummaryMatchingScope:
    scope = SummaryMatchingScope("", plan.binding_sha256, matching)
    return replace(scope, binding_sha256=retention._sha(_body(scope)))


def prepare_summary_matching(plan: SummaryRetentionPlan,
                             inventory: retention.FrozenInventory) -> SummaryMatchingScope:
    """Preserve v4 effective-summary and prior fields, without assessing the prior."""
    plan = _preflight(plan)
    return _wrap_matching(plan, retention.prepare_matching(plan.inventory_scope, inventory))


def _validate_matching(plan: SummaryRetentionPlan, scope: object) -> SummaryMatchingScope:
    if type(scope) is not SummaryMatchingScope:
        raise ValueError("a summary-only matching scope is required")
    retention._digest(scope.binding_sha256)
    retention._digest(scope.plan_binding_sha256)
    try:
        matching = retention._validate_matching(scope.matching_scope)
        expected = retention.prepare_matching(plan.inventory_scope, matching.inventory)
        if not retention._equal(_wrap_matching(plan, expected), scope):
            raise ValueError("summary matching scope binding mismatch")
    except (KeyError, IndexError, TypeError, AttributeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid summary matching scope") from exc
    return scope


def parse_summary_matching(raw: object, plan: SummaryRetentionPlan,
                           matching: SummaryMatchingScope) -> retention.MatchingOutcome:
    """Apply unchanged v4 whole-response, field and uncertainty rules."""
    plan = _preflight(plan)
    scope = _validate_matching(plan, matching)
    return retention.parse_matching(raw, scope.matching_scope)


def _result(plan: SummaryRetentionPlan, inventory: retention.InventoryOutcome,
            matching: retention.MatchingOutcome, calls: int,
            halted: str | None = None) -> SummaryRetentionResult:
    structural = (halted is None and inventory.structure_valid
                  and (matching.structure_valid or matching.status == "skipped_empty_inventory"))
    return SummaryRetentionResult(plan.binding_sha256, inventory, matching,
        MAX_CALLS, calls, calls == MAX_CALLS and halted is None, structural,
        structural and matching.model_retention_satisfied, halted)


def execute_summary_retention(plan: SummaryRetentionPlan, llm: LLMClient) -> SummaryRetentionResult:
    """One shot, at most two caller-owned invocations; no retries or repairs.

    Collection/structure and a model's summary-retention claim are distinct from
    semantic verification. Nothing here certifies grounding, prior continuity,
    any item, the whole digest, publication, or LME readiness.
    """
    plan = _preflight(plan)
    try:
        raw = llm.complete(plan.inventory_scope.request)
    except DeadlineExceeded:
        raise
    except Exception:
        _preflight(plan)
        return _result(plan, retention.InventoryOutcome("execution_error", None, "client_exception"),
            retention.MatchingOutcome("skipped_inventory_error", (), False, "client_exception"),
            1, "client_exception")
    _preflight(plan)
    inventory = parse_summary_inventory(raw, plan)
    if inventory.inventory is None:
        return _result(plan, inventory,
            retention.MatchingOutcome("skipped_invalid_inventory", (), False), 1)
    if not inventory.inventory.obligations:
        return _result(plan, inventory,
            retention.MatchingOutcome("skipped_empty_inventory", (), False), 1)
    try:
        matching = prepare_summary_matching(plan, inventory.inventory)
    except ValueError:
        return _result(plan, inventory,
            retention.MatchingOutcome("skipped_matching_bounds", (), False,
                                      "matching_preparation_failed"), 1)
    _preflight(plan)
    _validate_matching(plan, matching)
    try:
        raw = llm.complete(matching.matching_scope.request)
    except DeadlineExceeded:
        raise
    except Exception:
        _preflight(plan)
        _validate_matching(plan, matching)
        return _result(plan, inventory,
            retention.MatchingOutcome("execution_error", (), False, "client_exception"),
            2, "client_exception")
    _preflight(plan)
    outcome = parse_summary_matching(raw, plan, matching)
    return _result(plan, inventory, outcome, 2)
