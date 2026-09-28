"""Offline record-oriented review; never semantic or publication authorization.

This additive experiment preserves the source-review projection, source IDs,
authority parser and one-call-per-original-scope schedule. It only colocates
attribution with text and separates actor, identity and quantified-scope checks
from other relations. Opaque metadata is typed, never a naming authority.
No provider, store, credentials, network, retries or runtime integration live here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json

from benchmarks import digest_source_review as source_review
from hymem.deadline import DeadlineExceeded
from hymem.extraction.llm import LLMClient, LLMRequest


VERSION = "digest-record-review-v3"
MAX_CALLS = source_review.MAX_CALLS
MAX_INPUT_CHARS = source_review.MAX_INPUT_CHARS
MAX_OUTPUT_CHARS = source_review.MAX_OUTPUT_CHARS
MAX_PAYLOAD_CHARS = source_review.MAX_PAYLOAD_CHARS
MAX_CHECKS = source_review.MAX_CHECKS
MAX_EVIDENCE_PER_CHECK = source_review.MAX_EVIDENCE_PER_CHECK
RETENTION_FACETS = source_review.RETENTION_FACETS
CandidateField = source_review.CandidateField
EvidenceSource = source_review.EvidenceSource
RecordReviewCheck = source_review.SourceReviewCheck
RecordReviewOutcome = source_review.SourceReviewOutcome
RecordReviewResult = source_review.SourceReviewResult
_canonical = source_review._canonical
_sha = source_review._sha
_integer = source_review._integer

# IDs derive from the original relation ID, never from content or model output.
RELATION_FACETS = ("actor_attribution", "identity", "quantified_scope", "residual_relations")
QUANTIFIED_OBLIGATION = "quantified_scope_v1"


@dataclass(frozen=True, slots=True)
class OpaqueIdentifier:
    """Exact attribution handle; not a person name, alias or evidence source."""
    kind: str
    value: str | None
    allowed_use: str = "interpretation"
    display_name_authority: bool = False


@dataclass(frozen=True, slots=True)
class MessageAttribution:
    message_id: int
    role: str
    source_peer_identifier: OpaqueIdentifier
    source_workspace_identifier: OpaqueIdentifier

    @property
    def source_peer_id(self) -> str | None:
        """Compatibility view of the exact raw value, never an inferred name."""
        return self.source_peer_identifier.value

    @property
    def source_workspace_id(self) -> str | None:
        return self.source_workspace_identifier.value


@dataclass(frozen=True, slots=True)
class EmptyCanonicalSpan:
    """Preserve an empty projected span without inventing an evidence ID."""
    start: int
    end: int
    content: str = ""


@dataclass(frozen=True, slots=True)
class BoundaryContextRecord:
    source: EvidenceSource | None
    message: MessageAttribution
    start: int
    end: int
    content: str


@dataclass(frozen=True, slots=True)
class SourceRecord:
    chunk_id: str
    canonical_source: EvidenceSource | None
    empty_canonical_span: EmptyCanonicalSpan | None
    current_message: MessageAttribution
    attribution_sources: tuple[EvidenceSource, ...]
    boundary_context: BoundaryContextRecord | None


def _message(record: dict) -> MessageAttribution:
    return MessageAttribution(
        record["message_id"], record["role"],
        OpaqueIdentifier("opaque_peer_id", record["source_peer_id"]),
        OpaqueIdentifier("opaque_workspace_id", record["source_workspace_id"]))


@dataclass(frozen=True, slots=True)
class RecordReviewScope:
    kind: str
    index: int
    request: LLMRequest
    binding_sha256: str
    # Exact immutable original scope; not an additional wire payload.
    source_scope: source_review.SourceReviewScope

    @property
    def fields(self) -> tuple[CandidateField, ...]:
        return self.source_scope.fields

    @property
    def checks(self) -> tuple[RecordReviewCheck, ...]:
        result = []
        for check in self.source_scope.checks:
            if check.kind == "relations":
                result.extend(replace(check, check_id=f"{check.check_id}:{facet}", kind=facet)
                              for facet in RELATION_FACETS)
            else:
                result.append(check)
        return tuple(result)

    @property
    def canonical_sources(self) -> tuple[EvidenceSource, ...]:
        return self.source_scope.canonical_sources

    @property
    def context_sources(self) -> tuple[EvidenceSource, ...]:
        return self.source_scope.context_sources

    @property
    def prior_summary_sources(self) -> tuple[EvidenceSource, ...]:
        return self.source_scope.prior_summary_sources

    @property
    def source_records(self) -> tuple[SourceRecord, ...]:
        canonical = {source.chunk_id: source for source in self.canonical_sources}
        boundary = {source.chunk_id: source for source in self.context_sources
                    if source.kind == "boundary_context"}
        records = []
        projection = json.loads(self.source_scope.base_scope.source_projection_json)
        for value in projection["source_catalog"]:
            chunk_id = value["chunk_id"]
            context = value["interpretation_only_context"]
            records.append(SourceRecord(
                chunk_id, canonical.get(chunk_id),
                (EmptyCanonicalSpan(value["start"], value["end"])
                 if not value["visible_content"] else None),
                _message(value), tuple(source for source in self.context_sources
                                       if source.chunk_id == chunk_id
                                       and source.kind == "attribution"),
                (BoundaryContextRecord(boundary.get(chunk_id), _message(context),
                                       context["start"], context["end"], context["content"])
                 if context is not None else None)))
        return tuple(records)

    @property
    def max_checks(self) -> int:
        return self.source_scope.max_checks

    @property
    def max_evidence_per_check(self) -> int:
        return self.source_scope.max_evidence_per_check


@dataclass(frozen=True, slots=True)
class RecordReviewPlan:
    version: str
    input_sha256: str
    plan_sha256: str
    max_calls: int
    max_input_chars: int
    max_output_chars: int
    max_checks: int
    max_evidence_per_check: int
    requests: tuple[RecordReviewScope, ...]
    source_payload_json: str
    source_plan: source_review.SourceReviewPlan


def _replace_once(text: str, old: str, new: str) -> str:
    """Fail closed if the inherited contract changes, rather than mix policies."""
    if text.count(old) != 1:
        raise ValueError("inherited source-review contract changed")
    return text.replace(old, new, 1)


def _relation_rules() -> str:
    return (
        "An actor_attribution check asks whether every action, statement, request, "
        "recommendation and experience attributed in its candidate field belongs "
        "to the correct actor. Resolve first-person language using the message "
        "containing it. current_message owns canonical_source; boundary_context.message "
        "owns the boundary text. A preceding different message's speaker MUST NOT "
        "replace the current speaker. Same-message continuation preserves that one "
        "speaker. Resolve other pronouns only as justified by the exact scoped text; "
        "if ambiguous, use uncertain. Distinguish the speaker quoting or reporting "
        "someone from the person quoted: quoted first-person language belongs to "
        "the quoted speaker when established by the canonical text. Do not convert "
        "a speaker's account of another actor into that speaker's own action. "
        "An identity check asks whether each entity name, alias, identifier-to-person "
        "mapping and claimed entity equivalence in its candidate field is supported. "
        "Explicit names and aliases established by canonical text remain eligible "
        "evidence: distinguish who is being named from who is speaking. A quoted or "
        "reported other person's name MUST NOT become the current speaker's name. "
        "A peer identifier, workspace identifier or message role is a metadata label, "
        "not a personal name or alias. Opaque metadata alone never licenses a "
        "personal name, even when its exact value looks like a name. String equality "
        "alone does not establish an identity mapping. Preserve an opaque identifier "
        "as an identifier when justified, without converting it into a display name. "
        "Do not infer, normalize or construct a display name from metadata. Conversely, "
        "do not reject a genuine canonical naming statement merely because that name "
        "also appears in metadata. Canonical evidence must establish the particular "
        "identity relation asserted, not merely mention the same name. Apply the "
        "same source-owner and interpretation-only limits to identity as to every "
        "other check; boundary context cannot independently establish a named-person "
        "fact. Use uncertain when the identity mapping is ambiguous. "
        "A quantified_scope check asks whether every quantifier, cardinality, "
        "domain restriction and exclusivity claim in its candidate field is "
        "entailed by the authorized evidence. Its obligation names the fixed "
        "rubric in obligation_definitions; that definition is a "
        "review rubric, NOT extracted facts, a domain classification or a proof. "
        "Apply it semantically even without words such as all or only. Return the "
        "ordinary grounding verdict and IDs, not the rubric or intermediate answers. "
        "A residual_relations check asks whether negation, modality, time, "
        "causality and ordering are preserved. Quantification and exclusivity "
        "belong to quantified_scope, not this residual facet. "
        "Judge each scheduled facet on its own obligations, interpreting the field "
        "within the full candidate; a supported citation is not proof of meaning. "
    )


def _quantified_obligation() -> dict:
    """Fixed per-check questions, never facts extracted or guessed by code."""
    return {
        "claimed_scope": (
            "Identify the candidate's domain, predicate, polarity, quantifier and "
            "cardinality, including implicit restrictions and nested scopes. "
            "Resolve implicit or anaphoric scope from unambiguous full-candidate "
            "context, but underlying facts still require authorized sources. "
            "Never rescue an explicitly wider, global or contradictory claim "
            "by silently narrowing it using another candidate field."),
        "source_scope": (
            "Identify what the authorized sources establish about that same "
            "domain, positive members, exclusions and completeness. A named or "
            "sampled subset may be closed within itself without exhausting a "
            "broader population; unmentioned members are unknown, not absent."),
        "bounded_entailment": (
            "A complete set within the stated domain, a positive member and "
            "negative evidence for every other member support exclusivity within "
            "that domain. Apply the same reasoning to a negative predicate. "
            "Enumeration is not required when canonical text directly establishes "
            "the claimed universal, exclusion or exact count. A genuinely global "
            "canonical claim can support the same global candidate."),
        "invalid_strengthening": (
            "Do not generalize a subset to a wider domain, turn existence into "
            "universality, replace an upper or lower bound with an exact count, "
            "or invent exclusions. A universal statement does not by itself "
            "establish a nonempty domain or a particular member. An unsupported "
            "generalization is unsupported even if it remains possible."),
        "decision": (
            "Supported requires every claimed scoped relation to follow from "
            "the authorized evidence. Preserve conditions and uncertainty. Use "
            "uncertain when the intended domain, closure, reference or relation "
            "cannot be determined; do not count ambiguity as support. Missing "
            "exclusions cannot establish exclusivity. If there is no quantified "
            "claim, assess no extra quantifier obligation, but do not waive "
            "the other scheduled checks or the primary-evidence requirement."),
    }


def _wire_check(check: RecordReviewCheck) -> dict:
    value = {**asdict(check), "field_ids": list(check.field_ids)}
    if check.kind == "quantified_scope":
        value["obligation"] = QUANTIFIED_OBLIGATION
    return value


def _system(kind: str, max_evidence_per_check: int) -> str:
    # Keep the strict response, source-owned retention and evidence authority.
    # Replace the layout and relation split; clarify bounded entailment in both
    # assertion policy and inherited exclusivity warnings without easing scope.
    original = source_review._system(kind, max_evidence_per_check)
    authority_start = original.index("AUTHORITY: ")
    authority_end = original.index("Code assigns exact whole fields", authority_start)
    authority = (
        "AUTHORITY: source_records colocate each exact canonical_source primary "
        "EvidenceSource with current_message attribution and attribution_sources. "
        "Its boundary_context pairs the existing interpretation-only source with "
        "that context's actual message attribution. The record's chunk_id owns the "
        "window; boundary_context.message may describe a different, preceding "
        "message and speaker. current_message and boundary_context.message are "
        "interpretation-only, NOT independent primary evidence or new source IDs. "
        "Each message's source_peer_identifier and source_workspace_identifier are "
        "typed opaque metadata: kind identifies the handle's namespace, value is "
        "the exact original string or null, allowed_use is interpretation, and "
        "display_name_authority is false. They are NOT naming statements, even for "
        "name-like values. role is only the exact message-role label, not a personal "
        "name. Existing attribution_sources retain their original source IDs and "
        "text, but their role, source_peer_id and source_workspace_id fields record "
        "these same interpretation-only labels; their EvidenceSource wrapper does "
        "not grant display-name authority. Preserve null, empty and Unicode metadata "
        "exactly; never invent missing identities. Canonical text and genuine names "
        "remain unchanged. A null canonical_source with empty_canonical_span denotes exact "
        "empty text, not a new evidence source; a boundary with null source likewise "
        "has empty text and no returnable ID. A null boundary_context means no "
        "context window. In summary scopes only, prior_summary_sources supply "
        "fallible continuity, not new evidence. Only the existing source_id values "
        "inside EvidenceSource objects are returnable. Put canonical or summary-prior "
        "IDs only in primary_ids and attribution or boundary IDs only in context_ids. "
        "Every supported grounding check needs at least one primary ID. Every "
        "context ID in a supported check must accompany a canonical primary source "
        "with the same chunk_id; another record or prior summary cannot sponsor it. "
        "Even then, context may only interpret attribution or complete a phrase "
        "continuing into that canonical text; it cannot establish an independent "
        "context-only fact. Metadata alone cannot establish a content assertion. "
        "Negative and uncertain checks may cite no IDs, but every cited ID must "
        "remain in its authorized plane and scope. An authorized ID does not prove "
        "entailment.\n\n"
    )
    value = original[:authority_start] + authority + original[authority_end:]
    value = _replace_once(value, "Offline source review,", "Offline record review,")
    value = _replace_once(value,
        "A relations check asks whether attribution, identity, "
        "negation, modality, time, causality, ordering, quantification and exclusivity "
        "are preserved in the full candidate. ", _relation_rules())
    value = _replace_once(value,
        "An assertion check asks whether every assertion in the field is supported. ",
        "An assertion check asks whether every assertion in the field is supported. "
        "This includes the exact domain and strength of a title's assertions. "
        "Apply valid bounded entailment to assertions as well as quantified_scope: "
        "explicit closure of the stated set with one positive member and exclusions "
        "for all other members can support only within that set. Direct canonical "
        "support for a global assertion remains eligible. Do not reject a claim "
        "merely because it says exclusive, nor accept it merely because another "
        "facet is supported; inspect its actual scope and evidence. ")
    value = _replace_once(value,
        "Availability does not establish global exclusivity; identifiers are not "
        "display names; discussing a cause does not establish it. ",
        "Availability alone does not establish global exclusivity. Explicit "
        "exclusions over a complete stated domain can establish bounded exclusivity; "
        "a broader claim still needs evidence covering its broader domain. "
        "Identifiers are not display names; discussing a cause does not establish it. ")
    if kind in {"episode", "procedure"}:
        value = _replace_once(value,
            "Availability on one platform and absence from another does not "
            "establish exclusivity across all platforms. ",
            "Availability on one platform and absence from another establishes "
            "exclusivity within that stated two-platform set, but not across a "
            "broader platform domain unless its completeness and exclusions or "
            "a direct canonical global claim support that broader scope. ")
    return _replace_once(value, "Only the separately listed sources are evidence.",
                         "Only EvidenceSource objects in source_records and "
                         "prior_summary_sources are evidence.")


def _scope_body(scope: RecordReviewScope) -> dict:
    value = asdict(scope)
    del value["binding_sha256"]
    return {"version": VERSION, **value}


def _plan_body(plan: RecordReviewPlan) -> dict:
    value = asdict(plan)
    del value["plan_sha256"]
    return value


def _make_scope(source_scope: source_review.SourceReviewScope,
                max_input_chars: int) -> RecordReviewScope:
    scope = RecordReviewScope(source_scope.kind, source_scope.index,
                              source_scope.request, "", source_scope)
    if len(scope.checks) > scope.max_checks:
        raise ValueError("complete record review exceeds check cap")
    original = json.loads(source_scope.request.user)
    body = {
        "schema": VERSION, "scope": {"kind": scope.kind, "index": scope.index},
        "candidate": original["candidate"],
        "fields": [asdict(field) for field in scope.fields],
        "source_records": [{**asdict(record), "attribution_sources": [
            asdict(source) for source in record.attribution_sources]}
            for record in scope.source_records],
        "prior_summary_sources": [asdict(source) for source in scope.prior_summary_sources],
        "obligation_definitions": {QUANTIFIED_OBLIGATION: _quantified_obligation()},
        "checks": [_wire_check(check) for check in scope.checks],
    }
    request = replace(source_scope.request,
                      system=_system(scope.kind, scope.max_evidence_per_check),
                      user=source_review.ledger._bounded_json(body, max_input_chars))
    if len(request.system) + len(request.user) > max_input_chars:
        raise ValueError("record review request exceeds complete input cap")
    scope = replace(scope, request=request)
    return replace(scope, binding_sha256=_sha(_scope_body(scope)))


def _minimum_response(scope: RecordReviewScope) -> dict:
    return source_review._minimum_response(scope)


def _from_source(source_plan: source_review.SourceReviewPlan) -> RecordReviewPlan:
    scopes = tuple(_make_scope(scope, source_plan.max_input_chars)
                   for scope in source_plan.requests)
    if any(len(_canonical(_minimum_response(scope))) > source_plan.max_output_chars
           for scope in scopes):
        raise ValueError("complete record review cannot fit output cap")
    plan = RecordReviewPlan(
        VERSION, source_plan.input_sha256, "", source_plan.max_calls,
        source_plan.max_input_chars, source_plan.max_output_chars,
        source_plan.max_checks, source_plan.max_evidence_per_check,
        scopes, source_plan.source_payload_json, source_plan)
    return replace(plan, plan_sha256=_sha(_plan_body(plan)))


def prepare_record_review(
    payload: dict, request_template: LLMRequest, *, max_calls: int,
    max_input_chars: int = MAX_INPUT_CHARS, max_output_chars: int = MAX_OUTPUT_CHARS,
    max_checks: int = MAX_CHECKS, max_evidence_per_check: int = MAX_EVIDENCE_PER_CHECK,
) -> RecordReviewPlan:
    """Strictly prepare every original scope, then bind the complete new layout."""
    return _from_source(source_review.prepare_source_review(
        payload, request_template, max_calls=max_calls, max_input_chars=max_input_chars,
        max_output_chars=max_output_chars, max_checks=max_checks,
        max_evidence_per_check=max_evidence_per_check))


def _validate_scope(scope: object) -> RecordReviewScope:
    if (type(scope) is not RecordReviewScope or type(scope.kind) is not str
            or scope.kind not in {"episode", "procedure", "summary"}
            or type(scope.binding_sha256) is not str or len(scope.binding_sha256) != 64):
        raise ValueError("invalid record review scope")
    try:
        _integer(scope.index, "scope index", 0, MAX_CALLS - 1)
        source_review.isolation._template(scope.request)
        if len(scope.request.system) + len(scope.request.user) > MAX_INPUT_CHARS:
            raise ValueError("record review request exceeds input cap")
        source_scope = source_review._validate_scope(scope.source_scope)
        rebuilt = _make_scope(source_scope, MAX_INPUT_CHARS)
        if rebuilt != scope or _canonical(asdict(rebuilt)) != _canonical(asdict(scope)):
            raise ValueError("record review scope binding mismatch")
    except (KeyError, IndexError, TypeError, AttributeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid record review scope") from exc
    return scope


def _preflight(plan: object) -> RecordReviewPlan:
    if (type(plan) is not RecordReviewPlan or plan.version != VERSION
            or type(plan.requests) is not tuple or not plan.requests
            or len(plan.requests) > MAX_CALLS
            or any(type(scope) is not RecordReviewScope for scope in plan.requests)):
        raise ValueError("invalid record review plan")
    try:
        for value, label, maximum in (
            (plan.max_calls, "call cap", MAX_CALLS),
            (plan.max_input_chars, "input cap", MAX_INPUT_CHARS),
            (plan.max_output_chars, "output cap", MAX_OUTPUT_CHARS),
            (plan.max_checks, "check cap", MAX_CHECKS),
            (plan.max_evidence_per_check, "evidence cap", MAX_EVIDENCE_PER_CHECK),
        ):
            _integer(value, label, 1, maximum)
        if any(type(value) is not str or len(value) != 64
               for value in (plan.input_sha256, plan.plan_sha256)):
            raise ValueError("invalid record review plan digest")
        source_review.ledger._loads(plan.source_payload_json, MAX_PAYLOAD_CHARS)
        # The original whole-plan validator binds every scope, exact projection,
        # cap and sampling field before rebuilding any new request.
        source_plan = source_review._preflight(plan.source_plan)
        for scope in plan.requests:
            _validate_scope(scope)
        rebuilt = _from_source(source_plan)
        if rebuilt != plan or _canonical(asdict(rebuilt)) != _canonical(asdict(plan)):
            raise ValueError("record review plan binding mismatch")
    except (KeyError, IndexError, TypeError, AttributeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid record review plan") from exc
    return plan


def parse_record_review(
    raw: object, scope: RecordReviewScope, *, max_output_chars: int = MAX_OUTPUT_CHARS,
    max_checks: int = MAX_CHECKS, max_evidence_per_check: int = MAX_EVIDENCE_PER_CHECK,
) -> RecordReviewOutcome:
    """Reuse the strict whole-reply authority parser; no semantic inference."""
    _integer(max_output_chars, "output cap", 1, MAX_OUTPUT_CHARS)
    _integer(max_checks, "check cap", 1, MAX_CHECKS)
    _integer(max_evidence_per_check, "evidence cap", 1, MAX_EVIDENCE_PER_CHECK)
    scope = _validate_scope(scope)
    if len(scope.checks) > max_checks:
        raise ValueError("scope exceeds parsing check cap")
    if len(_canonical(_minimum_response(scope))) > max_output_chars:
        raise ValueError("complete record review cannot fit parsing output cap")
    return source_review._parse(raw, scope, max_output_chars,
                               min(max_evidence_per_check, scope.max_evidence_per_check))


def execute_record_review(plan: RecordReviewPlan, llm: LLMClient) -> RecordReviewResult:
    """Exactly one caller-owned invocation per scope after complete preflight.

    Malformed model replies continue without salvage; client errors halt with
    sanitized metadata. Deadlines and process interrupts propagate unchanged.
    The caller retains ownership of HTTP attempts, endpoint and absolute deadline.
    """
    plan = _preflight(plan)
    outcomes, halted = [], None
    for scope in plan.requests:
        try:
            raw = llm.complete(scope.request)
        except DeadlineExceeded:
            raise
        except Exception:
            outcomes.append(RecordReviewOutcome(
                scope.kind, scope.index, scope.binding_sha256,
                "execution_error", (), False, "client_exception"))
            halted = "client_exception"
            break
        outcomes.append(source_review._parse(raw, scope, plan.max_output_chars,
                                             plan.max_evidence_per_check))
    complete = halted is None and len(outcomes) == len(plan.requests)
    structural = complete and all(outcome.review_structure_valid for outcome in outcomes)
    return RecordReviewResult(
        VERSION, plan.plan_sha256, tuple(outcomes), len(outcomes), complete, structural,
        structural and all(outcome.model_no_defect for outcome in outcomes), halted)
