"""Eight AI-authored technical-domain development controls, not a holdout.

Only ``payload`` is model input. Source anchors, semantic components, mutation
descriptions and candidate judgments are external gold. These controls are new
prospective examples, not revisions of the preserved live pilot. They carry no
runner, provider calls, collection authority or representative-accuracy claim.

Procedure prerequisites/rules live in descriptions or actions. ``triggers``
contain query hints only; their words never establish execution constraints.
"""
from copy import deepcopy

from tests.digest_source_review_evaluation_fixtures import (
    context, payload, procedure, source,
)


FIXTURE_VERSION = "retention-semantic-development-v1"


def _obligation(identifier, record, quote, description, components, *,
                facet="constraints", interpretation_notes=()):
    text = record["visible_content"]
    assert text.count(quote) == 1
    start = record["start"] + text.index(quote)
    return {
        "id": identifier, "chunk_id": record["chunk_id"], "facet": facet,
        "description": description, "canonical_quote": quote,
        "canonical_start": start, "canonical_end": start + len(quote),
        "minimum_components": list(components),
        "interpretation_notes": list(interpretation_notes),
    }


def _match(identifier, expected, paths, rationale):
    return {"obligation_id": identifier, "expected": expected,
            "candidate_field_paths": list(paths), "rationale": rationale}


def build_cases():
    """Return four independent source-matched faithful/one-defect pairs."""
    cases = []

    def pair(name, kind, faithful, defective, obligations, matches, paths,
             change, *, faithful_scope_rationale, optional_details=(),
             excluded_context_only_facts=()):
        for position, variant in enumerate(("faithful", "defective")):
            cases.append({
                "id": f"retention-semantic-{name}-{variant}", "pair": name,
                "variant": variant, "fixture_version": FIXTURE_VERSION,
                "target_scope": {"kind": kind, "index": 0},
                "payload": deepcopy((faithful, defective)[position]),
                "unit_chars": 256,
                "expected_source_obligations": deepcopy(obligations),
                "candidate_matches": deepcopy(matches[position]),
                "optional_details": deepcopy(list(optional_details)),
                "excluded_context_only_facts": deepcopy(list(excluded_context_only_facts)),
                "changed_payload_paths": list(paths), "change": change,
                "source_mutations": [],
                "source_and_attribution_unchanged_within_pair": True,
                "label_scope": "targeted_obligations_only",
                "whole_scope_correctness_claim": False,
                "intended_faithful_no_veto": variant == "faithful",
                "faithful_no_veto": position == 0,
                "full_scope_gold_rationale": (faithful_scope_rationale if position == 0
                    else f"Known material defect in the selected scope: {change}"),
                "provenance": "AI-authored synthetic development control; not real LME data or a population sample.",
            })

    # The word "background" occurs both in genuinely irrelevant decoration
    # and in the actor of a mandatory technical rule. It is not an exemption.
    text = (
        "For audit-log rotation, write a new rotated log file. "
        "The dashboard border is blue; that color is background decoration, "
        "not part of the rotation procedure. "
        "Background log workers must remove bearer tokens before writing rotated audit logs.")
    record = source("unit-48001", 48001, text, "assistant")
    candidate = procedure(
        "Rotate audit logs", ("Write a new rotated audit log file",),
        description="Before writing rotated audit logs, background log workers must strip bearer tokens.",
        triggers=("audit log rotation", "rotate logs"))
    faithful = payload([record], text, procedure=candidate)
    defective = deepcopy(faithful)
    defective["procedure_items"][0]["candidate"]["description"] = (
        "Background log workers write rotated audit logs.")
    obligations = [
        _obligation("write-rotated-log", record,
            "For audit-log rotation, write a new rotated log file.",
            "The rotation procedure writes a new rotated log file.",
            ("write a new log file", "the file is the rotated audit log"),
            facet="material_facts"),
        _obligation("background-worker-redaction", record,
            "Background log workers must remove bearer tokens before writing rotated audit logs.",
            "Background log workers must redact bearer tokens before writing rotated logs.",
            ("background log workers", "bearer tokens must be removed",
             "removal precedes writing rotated audit logs")),
    ]
    fixed = _match("write-rotated-log", "retained", ("/candidate/steps/0/action",),
        "Both variants explicitly write a new rotated audit log; optional dashboard decoration is immaterial.")
    pair("material-rule-despite-background-word", "procedure", faithful, defective,
        obligations,
        ([fixed, _match("background-worker-redaction", "retained", ("/candidate/description",),
            "Strip faithfully paraphrases remove; the description preserves the actor, mandatory redaction and before-writing condition.")],
         [fixed, _match("background-worker-redaction", "omitted", (),
            "The neutral description keeps the worker and writing task but no semantic field requires token removal before writing.")]),
        ("/procedure_items/0/candidate/description",),
        "Remove only mandatory pre-write token redaction from the description; both candidates omit the genuinely optional color.",
        faithful_scope_rationale=(
            "The selected procedure retains new rotated-log creation plus the background workers' "
            "mandatory token removal before writing; the blue dashboard border is explicitly outside "
            "the procedure. No execution rule is inferred from query-hint triggers."),
        optional_details=({"chunk_id": record["chunk_id"],
            "canonical_quote": "The dashboard border is blue",
            "reason": "The source explicitly calls this background decoration outside the procedure; its omission is legitimate, unlike the worker's mandatory redaction rule."},))

    # A completed event continues the SAME message; the second record's
    # first-person completed event belongs to its CURRENT user, not to the
    # preceding assistant. The assistant's rollback is also canonical in
    # that second record via the current user's explicit "You" assertion.
    cache = source("unit-48011", 48011, "the staging cache.",
        context=context(48011, "I flushed "))
    checks = source("unit-48013", 48013,
        "You completed the rollback; I verified the health checks afterward.",
        context=context(48012,
            "I rolled back; I opened the metrics dashboard.", "assistant"))
    body = ("The user flushed the staging cache. The assistant completed the rollback; "
            "the user then verified the health checks.")
    faithful = payload([cache, checks], body, title="Rollback verification", body=body)
    defective = deepcopy(faithful)
    defective["items"][0]["candidate_body"] = (
        "The user flushed the staging cache. The assistant completed the rollback; "
        "the assistant then verified the health checks.")
    obligations = [
        _obligation("cache-flush-owner", cache, "the staging cache.",
            "The user completed flushing the staging cache.",
            ("user is actor", "flushing the staging cache", "completed event"),
            facet="material_facts", interpretation_notes=(
                "The same-message prefix interprets the canonical continuation; no separate context-only fact is needed.",)),
        _obligation("rollback-owner", checks, "You completed the rollback",
            "The assistant completed the rollback.",
            ("assistant is current user's addressee", "rollback", "completed event"),
            facet="material_facts"),
        _obligation("post-rollback-check-owner", checks,
            "I verified the health checks afterward.",
            "The current user verified the health checks after the assistant's rollback.",
            ("current user is actor", "health checks verified", "completed event",
             "verification follows rollback"), facet="material_facts",
            interpretation_notes=(
                "The preceding assistant message cannot replace the current user as the owner of I; the temporal referent is the rollback in current canonical text.",)),
    ]
    fixed = [_match(identifier, "retained", ("/candidate_body",),
        "Both variants preserve this completed event and its actor.")
        for identifier in ("cache-flush-owner", "rollback-owner")]
    pair("contextual-completed-event-owner", "episode", faithful, defective,
        obligations,
        (fixed + [_match("post-rollback-check-owner", "retained", ("/candidate_body",),
            "The current user remains the health-check actor after the assistant's rollback.")],
         fixed + [_match("post-rollback-check-owner", "altered", ("/candidate_body",),
            "Only the health-check actor changes to the preceding assistant; event completion and timing remain intact.")]),
        ("/items/0/candidate_body",),
        "Change only the completed health-check event's actor from current user to preceding assistant; preserve cache/rollback ownership and timing.",
        faithful_scope_rationale=(
            "The selected episode retains the user's completed cache flush, the assistant's completed "
            "rollback and the current user's subsequent health-check verification. Same-message prefix "
            "only interprets the cache continuation; the independent context-only dashboard opening is "
            "not asserted. The current user's You claim independently supports the assistant's rollback."),
        excluded_context_only_facts=({"chunk_id": checks["chunk_id"],
            "context_message_id": 48012, "context_role": "assistant",
            "context_quote": "opened the metrics dashboard.",
            "reason": "This independent event occurs only in preceding-message interpretation context, not current canonical text; it cannot be added as a supported episode fact."},))

    text = ("Run a fifteen-minute packet capture. Keep the debug console connected "
            "from the start of that capture until it finishes.")
    record = source("unit-48021", 48021, text, "assistant")
    candidate = procedure("Capture diagnostic packets", ("Run a fifteen-minute packet capture",),
        description="Maintain the debug-console connection for all fifteen minutes of packet capture.",
        triggers=("packet capture", "debug console"))
    faithful = payload([record], text, procedure=candidate)
    defective = deepcopy(faithful)
    defective["procedure_items"][0]["candidate"]["description"] = (
        "Maintain the debug-console connection for the first fourteen minutes of packet capture; "
        "it may disconnect for the last minute.")
    obligations = [
        _obligation("capture-duration", record, "Run a fifteen-minute packet capture.",
            "Run packet capture for fifteen minutes.",
            ("run packet capture", "fifteen-minute duration")),
        _obligation("connection-through-completion", record,
            "Keep the debug console connected from the start of that capture until it finishes.",
            "The debug console must remain connected throughout the entire capture.",
            ("debug console remains connected", "from capture start",
             "through capture completion", "no last-minute disconnection permission")),
    ]
    fixed = _match("capture-duration", "retained", ("/candidate/steps/0/action",),
        "The action in both candidates retains the fifteen-minute capture.")
    pair("duration-paraphrase-near-miss", "procedure", faithful, defective,
        obligations,
        ([fixed, _match("connection-through-completion", "retained", ("/candidate/description",),
            "All fifteen minutes faithfully paraphrases from capture start until capture completion.")],
         [fixed, _match("connection-through-completion", "altered", ("/candidate/description",),
            "The explicit last-minute disconnection permission contradicts continuous connection through completion; it is not merely an omitted qualifier.")]),
        ("/procedure_items/0/candidate/description",),
        "Relax only the required connection interval from the whole capture to fourteen minutes, explicitly permitting disconnection during the last minute.",
        faithful_scope_rationale=(
            "The action preserves the fifteen-minute packet capture and the description preserves "
            "connection throughout that entire capture by paraphrase. Those are all canonical "
            "instructions; triggers contain only topic queries, not conditions."))

    # Both tasks share readiness as a prerequisite, but that does not entail
    # the separately stated export-before-rebuild chronology.
    text = ("The staging database was ready before schema export and cache rebuild began. "
            "Both tasks finished, and schema export finished before cache rebuild began.")
    record = source("unit-48031", 48031, text)
    summary = ("The staging database was ready before schema export and cache rebuild began. "
               "Both tasks finished, with schema export finishing before cache rebuild began.")
    faithful = payload([record], summary)
    defective = deepcopy(faithful)
    for key in ("candidate_summary", "candidate_raw_summary"):
        defective["summary_item"][key] = (
            "The staging database was ready before schema export and cache rebuild began. "
            "Both tasks finished, with cache rebuild beginning before schema export finished.")
    obligations = [
        _obligation("shared-database-readiness", record,
            "The staging database was ready before schema export and cache rebuild began.",
            "Database readiness preceded the start of both schema tasks.",
            ("staging database ready", "readiness before schema export begins",
             "readiness before cache rebuild begins"), facet="ordering"),
        _obligation("both-schema-tasks-completed", record, "Both tasks finished",
            "Both schema export and cache rebuild completed.",
            ("schema export completed", "cache rebuild completed"), facet="material_facts"),
        _obligation("export-before-rebuild", record,
            "schema export finished before cache rebuild began.",
            "Schema export finished before cache rebuild began, not merely after a shared prerequisite.",
            ("schema export completion", "cache rebuild start",
             "export completion precedes rebuild start"), facet="ordering"),
    ]
    fixed = [_match(identifier, "retained", ("/candidate_summary",),
        "Both summaries preserve this source statement without inferring order between the two tasks.")
        for identifier in ("shared-database-readiness", "both-schema-tasks-completed")]
    pair("actual-order-versus-shared-prerequisite", "summary", faithful, defective,
        obligations,
        (fixed + [_match("export-before-rebuild", "retained", ("/candidate_summary",),
            "The final clause expressly retains export completion before rebuild start.")],
         fixed + [_match("export-before-rebuild", "altered", ("/candidate_summary",),
            "Cache rebuild now begins before schema export finishes, reversing the explicit source relation; shared readiness does not excuse the reversed relation.")]),
        ("/summary_item/candidate_summary", "/summary_item/candidate_raw_summary"),
        "Reverse only export-finish-before-rebuild-start to rebuild-start-before-export-finish; retain shared readiness, both actions and both completed outcomes.",
        faithful_scope_rationale=(
            "The selected summary preserves readiness before both task starts, both completed outcomes "
            "and the distinct export-finish-before-rebuild-start relation. The latter is explicit, "
            "not inferred from a common prerequisite. This pair does not test a source that leaves "
            "the inter-task relation unspecified; no source-changing unordered counterpart is claimed."))
    return cases
