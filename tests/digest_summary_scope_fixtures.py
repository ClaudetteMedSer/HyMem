"""Four AI-authored summary-scope development controls, not an LME holdout.

Only ``payload`` is model input. Gold, semantic ownership, source anchors and
declared mutations remain outside it. Both pairs keep faithful narrow items
unchanged while deleting one material outcome only from the summary. An item
that still contains that outcome cannot repair the summary's omission.

Citation scope supplies support, not per-item ownership of every statement in
a multi-topic message. No provider calls, evaluator or execution authority are
included. Existing controls and historical pilot artifacts are not modified.
"""
from copy import deepcopy

from tests.digest_source_review_evaluation_fixtures import payload, procedure, source


FIXTURE_VERSION = "digest-summary-scope-development-v1"


def _obligation(identifier, record, quote, description, components, *, facet):
    text = record["visible_content"]
    assert text.count(quote) == 1
    start = record["start"] + text.index(quote)
    return {
        "id": identifier, "chunk_id": record["chunk_id"], "facet": facet,
        "description": description, "canonical_quote": quote,
        "canonical_start": start, "canonical_end": start + len(quote),
        "minimum_components": list(components), "interpretation_notes": [],
    }


def _episode(index, title, body, citation, *, outcome):
    return {
        "index": index, "candidate_title": title, "candidate_body": body,
        "candidate_outcome": outcome, "candidate_key_entities": [],
        "cited_source_ids": [citation],
    }


def _pair(name, faithful, shorter_summary, obligations, omitted_id,
          item_scope_expectations, rationale):
    defective = deepcopy(faithful)
    for key in ("candidate_summary", "candidate_raw_summary"):
        defective["summary_item"][key] = shorter_summary
    cases = []
    for variant, value in (("faithful", faithful), ("defective", defective)):
        matches = []
        for obligation in obligations:
            omitted = variant == "defective" and obligation["id"] == omitted_id
            matches.append({
                "obligation_id": obligation["id"],
                "expected": "omitted" if omitted else "retained",
                "candidate_field_paths": [] if omitted else ["/candidate_summary"],
                "rationale": (
                    "The summary deletes this material result; an unchanged episode "
                    "containing it is a different output scope and cannot supply retention."
                    if omitted else
                    "The summary itself preserves this exact source meaning, independently "
                    "of what any narrower episode or procedure contains."
                ),
            })
        cases.append({
            "id": f"digest-summary-scope-{name}-{variant}", "pair": name,
            "variant": variant, "fixture_version": FIXTURE_VERSION,
            "target_scope": {"kind": "summary", "index": 0},
            "payload": deepcopy(value), "unit_chars": 256,
            "expected_source_obligations": deepcopy(obligations),
            "candidate_matches": matches,
            "optional_details": [], "excluded_context_only_facts": [],
            "item_scope_expectations": deepcopy(item_scope_expectations),
            "changed_payload_paths": [
                "/summary_item/candidate_raw_summary",
                "/summary_item/candidate_summary",
            ],
            "change": (
                f"Delete only the summary's {omitted_id} result; preserve all sources, "
                "attribution, narrow item citations, item assertions and remaining summary meaning."
            ),
            "source_mutations": [],
            "source_and_attribution_unchanged_within_pair": True,
            "unselected_items_unchanged_within_pair": True,
            "unselected_items_intended_faithful": True,
            "label_scope": "targeted_obligations_only",
            "whole_scope_correctness_claim": False,
            "intended_faithful_no_veto": variant == "faithful",
            "faithful_no_veto": variant == "faithful",
            "full_scope_gold_rationale": (
                rationale if variant == "faithful" else
                f"The selected summary loses the material {omitted_id} outcome while "
                "its other statements remain supported. Complete sibling items do not cure that loss."
            ),
            "provenance": (
                "AI-authored synthetic development control for coverage-scope alignment; "
                "not real LME data, a held-out sample or a population accuracy estimate."
            ),
        })
    return cases


def build_cases():
    """Return two source-matched faithful/summary-omission pairs."""
    cases = []

    # Two independent technical procedures have the same supporting message,
    # but neither owns the other procedure's requirements. A separate outcome
    # is an episode with a narrower, different message citation.
    instructions = source("unit-49101", 49101,
        "To rotate application logs, stop the log writer before archiving the log file; "
        "do not delete the archive. Separately, to refresh the search index, build a new "
        "index before switching the search alias; keep the old index until the switch succeeds.",
        "assistant")
    result = source("unit-49102", 49102,
        "I ran the upload integrity check. The check passed.")
    rotate = procedure("Rotate application logs", (
        "Stop the log writer", "Archive the log file"),
        description="Rotate application logs without deleting the archive.",
        triggers=("log rotation",))
    refresh = procedure("Refresh the search index", (
        "Build a new index", "Switch the search alias"),
        description="Refresh the search index while keeping the old index until the switch succeeds.",
        triggers=("search index refresh",))
    summary_without_result = (
        "Log rotation requires stopping the writer before archiving without deleting the archive; "
        "search-index refresh requires building a new index before switching the alias and "
        "keeping the old index until the switch succeeds; the user ran the upload integrity check.")
    faithful_summary = summary_without_result[:-1] + " and it passed."
    faithful = payload([instructions, result], faithful_summary)
    faithful["procedure_items"] = [
        {"index": index, "candidate": candidate,
         "cited_source_ids": [instructions["chunk_id"]]}
        for index, candidate in enumerate((rotate, refresh))
    ]
    faithful["items"] = [_episode(0, "Upload integrity check passed",
        "The user ran the upload integrity check and it passed.", result["chunk_id"],
        outcome="resolved")]
    obligations = [
        _obligation("log-stop-before-archive", instructions,
            "stop the log writer before archiving the log file",
            "Log rotation requires stopping the log writer before archiving the file.",
            ("log rotation", "stop log writer", "archive log file", "stopping precedes archiving"),
            facet="ordering"),
        _obligation("archive-not-deleted", instructions, "do not delete the archive.",
            "The log archive must not be deleted.",
            ("log archive", "deletion prohibited"), facet="constraints"),
        _obligation("build-before-alias-switch", instructions,
            "build a new index before switching the search alias",
            "Search refresh requires a new index before the search-alias switch.",
            ("build new search index", "switch search alias", "building precedes switch"),
            facet="ordering"),
        _obligation("old-index-until-switch-success", instructions,
            "keep the old index until the switch succeeds.",
            "The old search index must remain until the alias switch succeeds.",
            ("keep old search index", "until successful alias switch"), facet="constraints"),
        _obligation("upload-check-ran", result, "I ran the upload integrity check.",
            "The user ran the upload integrity check.",
            ("current user actor", "upload integrity check", "completed execution"),
            facet="material_facts"),
        _obligation("upload-check-passed", result, "The check passed.",
            "The upload integrity check passed.",
            ("upload integrity check", "passed result"), facet="material_facts"),
    ]
    owners = [
        {"kind": "procedure", "index": 0,
         "cited_source_ids": [instructions["chunk_id"]],
         "owned_source_obligation_ids": ["log-stop-before-archive", "archive-not-deleted"],
         "must_not_inherit_sibling_obligation_ids": [
             "build-before-alias-switch", "old-index-until-switch-success"],
         "rationale": "Log rotation is a complete distinct technical task; the same message also supports an unrelated search-refresh procedure."},
        {"kind": "procedure", "index": 1,
         "cited_source_ids": [instructions["chunk_id"]],
         "owned_source_obligation_ids": ["build-before-alias-switch", "old-index-until-switch-success"],
         "must_not_inherit_sibling_obligation_ids": ["log-stop-before-archive", "archive-not-deleted"],
         "rationale": "Search refresh is complete without importing log-writer or archive instructions."},
        {"kind": "episode", "index": 0,
         "cited_source_ids": [result["chunk_id"]],
         "owned_source_obligation_ids": ["upload-check-ran", "upload-check-passed"],
         "must_not_inherit_sibling_obligation_ids": [],
         "rationale": "The check result cites only its own user message, never the runbooks; its unchanged pass cannot rescue summary omission."},
    ]
    cases.extend(_pair("two-procedures-shared-source", faithful,
        summary_without_result, obligations, "upload-check-passed", owners,
        "The selected summary retains both independent runbooks' ordered actions and "
        "prohibitions/conditions, plus the user's completed check and passed outcome; "
        "its clauses describe instructions rather than falsely completed procedure execution. "
        "Each item cites only its supporting message, and the two procedures correctly "
        "share one multi-topic source without duplicating each other's requirements."))

    # Two granular episodes own different completed events from one message.
    # A third episode cites only the separate test-result message. The summary
    # must still retain the complete meaningful result across both messages.
    changes = source("unit-49201", 49201,
        "I switched the build cache to local-disk mode. Separately, I raised the API "
        "rate limit to 240 requests per minute.")
    restore = source("unit-49202", 49202,
        "The backup restore test ran on staging. A checksum mismatch blocked restore validation.",
        "assistant")
    summary_without_result = (
        "The user switched the build cache to local-disk mode and raised the API rate "
        "limit to 240 requests per minute; the backup restore test ran on staging.")
    faithful_summary = (
        summary_without_result[:-1] + " and a checksum mismatch blocked restore validation.")
    faithful = payload([changes, restore], faithful_summary)
    faithful["items"] = [
        _episode(0, "Build cache mode changed",
            "The user switched the build cache to local-disk mode.", changes["chunk_id"],
            outcome="resolved"),
        _episode(1, "API rate limit raised",
            "The user raised the API rate limit to 240 requests per minute.", changes["chunk_id"],
            outcome="resolved"),
        _episode(2, "Restore validation blocked",
            "The backup restore test ran on staging and a checksum mismatch blocked restore validation.",
            restore["chunk_id"], outcome="blocked"),
    ]
    obligations = [
        _obligation("cache-mode-changed", changes,
            "I switched the build cache to local-disk mode.",
            "The user completed changing the build cache to local-disk mode.",
            ("current user actor", "build cache", "local-disk mode", "completed change"),
            facet="material_facts"),
        _obligation("api-rate-raised", changes,
            "I raised the API rate limit to 240 requests per minute.",
            "The user raised the API rate limit to 240 requests per minute.",
            ("current user actor", "API rate limit", "raised to 240 requests per minute",
             "completed change"), facet="material_facts"),
        _obligation("restore-test-ran-on-staging", restore,
            "The backup restore test ran on staging.",
            "The backup restore test executed on staging.",
            ("backup restore test", "completed execution", "staging environment"),
            facet="material_facts"),
        _obligation("restore-validation-blocked", restore,
            "A checksum mismatch blocked restore validation.",
            "A checksum mismatch blocked restore validation.",
            ("checksum mismatch", "restore validation blocked", "mismatch caused the block"),
            facet="material_facts"),
    ]
    owners = [
        {"kind": "episode", "index": 0,
         "cited_source_ids": [changes["chunk_id"]],
         "owned_source_obligation_ids": ["cache-mode-changed"],
         "must_not_inherit_sibling_obligation_ids": ["api-rate-raised"],
         "rationale": "One granular episode concerns the cache-mode change; the second independent change belongs in its sibling, despite the shared supporting message."},
        {"kind": "episode", "index": 1,
         "cited_source_ids": [changes["chunk_id"]],
         "owned_source_obligation_ids": ["api-rate-raised"],
         "must_not_inherit_sibling_obligation_ids": ["cache-mode-changed"],
         "rationale": "One granular episode concerns the API-rate change and need not repeat the cache-mode event."},
        {"kind": "episode", "index": 2,
         "cited_source_ids": [restore["chunk_id"]],
         "owned_source_obligation_ids": ["restore-test-ran-on-staging", "restore-validation-blocked"],
         "must_not_inherit_sibling_obligation_ids": [],
         "rationale": "The restore-result episode cites only the result message; it stays faithful in both variants but is not summary evidence."},
    ]
    cases.extend(_pair("two-granular-episodes-shared-source", faithful,
        summary_without_result, obligations, "restore-validation-blocked", owners,
        "The selected summary preserves both distinct user-completed configuration changes "
        "with the exact mode and rate, plus the staging restore-test execution and its "
        "explicit checksum-mismatch block of restore validation. It invents no ordering between the independent configuration "
        "changes. The two granular episodes share one source while retaining different events; "
        "the separate result episode has only its own message as authority."))
    return cases
