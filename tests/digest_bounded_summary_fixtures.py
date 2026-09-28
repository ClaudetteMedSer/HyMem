"""Prospective bounded-highlights controls, not measured model performance.

Only ``payload`` may enter a verifier request. All policy expectations, source
anchors, candidate selection notes and mutations are external gold. The input
shape is the existing digest-fidelity-decisions-v9 packet: canonical catalog,
empty item arrays, and the exact effective summary plus prior continuity.
The policy labels below are semantic comparison arms, not runtime config names.

These twelve invented cases compare legacy material-retention semantics with
the proposed bounded-highlights treatment. They do not establish underlying
episode/procedure coverage, authorize publication, or claim LME readiness.
Permitting selection does not permit a misleading partial selected event.
Every effective candidate and prior has the runtime one-sentence/500-codepoint
shape without a ``The user`` / ``The assistant`` prefix. Explicit empty raw
no-ops retain the nonempty prior; an empty raw string is not an empty published
summary. No provider response or production memory was used to author these.

Frozen-label boundary: a bounded summary may omit an entire unrelated durable
topic without first filling all 500 characters. If the implemented policy
instead requires proving capacity exhaustion, the two selection-family labels
need prospective adjudication, not silent relabeling after provider output.
Likewise, no label here requires a no-op to reject every unrelated new topic:
the substantive no-op is an explicit correction of the retained prior itself.
"""

from copy import deepcopy

from tests.digest_source_review_evaluation_fixtures import payload, source


FIXTURE_VERSION = "digest-bounded-summary-development-v1"
POLICY_ARMS = ("legacy_material_retention", "bounded_highlights")


def _anchor(record, quote, meaning):
    """External evidence coordinates; never a model-generated semantic ID."""
    text = record["visible_content"]
    assert text.count(quote) == 1
    start = record["start"] + text.index(quote)
    return {
        "chunk_id": record["chunk_id"],
        "message_id": record["message_id"],
        "role": record["role"],
        "canonical_start": start,
        "canonical_end": start + len(quote),
        "canonical_quote": quote,
        "meaning": meaning,
    }


def _case(family, variant, records, summary, prior, *, legacy, bounded,
          rationale, anchors, selected_meaning, omitted_topics=(),
          mutation=None, noop=False, source_matched=True):
    packet = payload(deepcopy(records), summary, prior=prior)
    if noop:
        # Match actual runtime transport: empty primary keeps the prior, and
        # fidelity must assess that unchanged effective value against new data.
        assert summary == prior
        packet["summary_item"]["candidate_raw_summary"] = ""
        packet["summary_item"]["candidate_is_noop"] = True
    return {
        "case_id": f"bounded-summary-{family}-{variant}",
        "fixture_version": FIXTURE_VERSION,
        "family": family,
        "variant": variant,
        "target_scope": {"kind": "summary", "index": 0},
        "payload": packet,
        "gold": {
            "expected_policy_verdicts": {
                "legacy_material_retention": legacy,
                "bounded_highlights": bounded,
            },
            "rationale": rationale,
            "source_anchors": deepcopy(anchors),
            "selected_meaning": selected_meaning,
            "omitted_entire_topics": list(omitted_topics),
            "declared_mutation": mutation,
            "source_and_prior_unchanged_within_family": source_matched,
            "label_scope": "summary_fidelity_under_explicit_policy_only",
            "underlying_coverage_assessed": False,
            "semantic_accuracy_observed": False,
            "review_status": "prospective_authored_gold_requires_independent_review",
        },
    }


def build_cases():
    """Return twelve fresh packets with external, prospectively authored gold."""
    cases = []

    # Dense continuity and three unrelated durable new topics. The selected
    # migration is not complete merely because the attempt itself is mentioned.
    prior = (
        "Append-only audit logging is enabled; backup checks run every Sunday; "
        "support tickets are routed to the service desk; onboarding uses a shared "
        "checklist; invoices require finance approval; meeting notes stay in the "
        "team wiki; access reviews run monthly; release notes accompany deployments; "
        "training sessions use recorded demonstrations; equipment requests go to "
        "operations; incident reviews use a written timeline; shared documents "
        "retain version history.")
    migration = source("unit-61001", 61001,
        "I attempted the staging migration, but a checksum mismatch blocked it; "
        "the migration did not complete.", peer="speaker-71")
    refund = source("unit-61002", 61002,
        "Separately, my refund appeal was approved and the money arrived.",
        peer="speaker-71")
    certificate = source("unit-61003", 61003,
        "On an unrelated topic, I passed the language certification exam.",
        peer="speaker-71")
    records = [migration, refund, certificate]
    anchors = [
        _anchor(migration, migration["visible_content"],
            "The selected migration was attempted but blocked, not completed."),
        _anchor(refund, refund["visible_content"],
            "A distinct refund appeal succeeded and payment arrived."),
        _anchor(certificate, certificate["visible_content"],
            "A distinct certification exam was passed."),
    ]
    for variant, candidate, bounded in (
        ("selected-complete",
         "The staging migration was attempted but remained incomplete because of "
         "a checksum mismatch; append-only audit logging remains enabled.",
         "supported"),
        ("selected-outcome-omitted",
         "The staging migration was attempted; append-only audit logging remains enabled.",
         "unsupported"),
    ):
        cases.append(_case("selected-topic-dense-prior", variant,
            records, candidate, prior, legacy="unsupported", bounded=bounded,
            rationale=(
                "Both candidates omit entire unrelated durable refund/exam topics and "
                "several distinct prior topics, while retaining true prior continuity; "
                "the legacy coverage-oriented policy rejects that loss. Bounded highlights "
                "permit whole-topic selection, but only the selected-complete variant "
                "preserves the migration's blocked, incomplete outcome. The defective "
                "variant chooses that same event and withholds its decisive result; "
                "being literally true about an attempt does not cure selected-event loss."),
            anchors=anchors,
            selected_meaning="Attempted staging migration with its blocked outcome, plus audit-logging continuity.",
            omitted_topics=("refund appeal and payment", "certification exam",
                            "unselected prior administrative topics"),
            mutation="Remove only the selected migration's checksum-blocked and incomplete result."))

    # A recommendation is not a completed action. Condition and negation are
    # part of the selected recommendation, not optional neighboring topics.
    prior = "Audit logging remains enabled."
    rule = source("unit-61011", 61011,
        "I recommend restarting the staging worker only after change-officer "
        "approval; the production worker must not be restarted.",
        "assistant", peer="speaker-72")
    anchors = [_anchor(rule, rule["visible_content"],
        "Staging restart is conditional advice; production restart is prohibited.")]
    faithful = (
        "Restarting the staging worker was recommended only after change-officer "
        "approval, and the production worker must not be restarted; audit logging remains enabled.")
    for variant, candidate in (
        ("qualified", faithful),
        ("condition-omitted", faithful.replace(" only after change-officer approval", "")),
        ("negation-omitted", faithful.replace("must not be restarted", "must be restarted")),
    ):
        verdict = "supported" if variant == "qualified" else "unsupported"
        cases.append(_case("selected-condition-and-negation", variant,
            [rule], candidate, prior, legacy=verdict, bounded=verdict,
            rationale=(
                "The selected advice must remain advice, contingent on change-officer "
                "approval, and retain the production prohibition. Removing the approval "
                "condition broadens applicability; removing 'not' reverses the prohibited "
                "action. Neither change is whole-topic omission or harmless compression."),
            anchors=anchors,
            selected_meaning="Conditional staging advice and the explicitly selected production prohibition.",
            mutation="Delete only the approval qualifier or the production negation, respectively."))

    # Same exact source/prior and same candidate word multiset. Retention must
    # bind outcomes to the correct recurrence, not merely count their words.
    prior = "Search alerts remain enabled."
    monday = source("unit-61021", 61021,
        "On Monday I ran the archive-restore check and it failed.", peer="speaker-73")
    tuesday = source("unit-61022", 61022,
        "On Tuesday I reran the same archive-restore check and it passed.", peer="speaker-73")
    anchors = [
        _anchor(monday, monday["visible_content"], "Monday's check failed."),
        _anchor(tuesday, tuesday["visible_content"], "Tuesday's recurrence passed."),
    ]
    for variant, candidate in (
        ("bound-correctly",
         "The archive-restore check failed on Monday and passed on Tuesday; search alerts remain enabled."),
        ("bindings-swapped",
         "The archive-restore check passed on Monday and failed on Tuesday; search alerts remain enabled."),
    ):
        verdict = "supported" if variant == "bound-correctly" else "unsupported"
        cases.append(_case("recurring-outcome-bindings", variant,
            [monday, tuesday], candidate, prior, legacy=verdict, bounded=verdict,
            rationale=(
                "The completed check recurs on two days with opposite results. Both "
                "words 'failed' and 'passed' occur in both candidates, but each result "
                "must remain bound to its own day. The opaque speaker identifier is "
                "not a display name and is not itself a candidate assertion."),
            anchors=anchors,
            selected_meaning="Two distinct dated occurrences and their respective outcomes.",
            mutation="Swap only 'failed' and 'passed'; sources, prior and candidate word multiset stay unchanged."))

    # Prior is derived continuity, not authority to overrule a new correction.
    prior = "The service rollout is complete and monitoring remains enabled."
    correction = source("unit-61031", 61031,
        "Correction: the service rollout is not complete; only the staging "
        "trial finished, and production is still pending.", peer="speaker-74")
    anchors = [_anchor(correction, correction["visible_content"],
        "New canonical correction supersedes the prior completed-rollout claim.")]
    for variant, candidate in (
        ("updated",
         "Only the staging trial is complete and the production rollout is still pending; monitoring remains enabled."),
        ("stale-prior-asserted",
         "The service rollout is complete and monitoring remains enabled."),
    ):
        verdict = "supported" if variant == "updated" else "unsupported"
        cases.append(_case("explicit-prior-correction", variant,
            [correction], candidate, prior, legacy=verdict, bounded=verdict,
            rationale=(
                "The new statement explicitly corrects the prior. The candidate may "
                "retain monitoring continuity, but may not keep asserting full rollout "
                "completion as current. Treating the correction as an unselected topic "
                "cannot rescue a retained claim it directly contradicts."),
            anchors=anchors,
            selected_meaning="Current rollout status and still-relevant monitoring continuity.",
            mutation="Replace the updated rollout status with the prior's now-superseded completion assertion."))

    # Raw empty no-op semantics, not a blanket rule that every unrelated new
    # topic must be selected. The adverse input changes a retained state itself.
    prior = "The nightly backup is enabled and restores are checked weekly."
    acknowledgement = source("unit-61041", 61041, "Thanks, understood.", peer="speaker-75")
    substantive = source("unit-61041", 61041,
        "Correction: the nightly backup is disabled now; restores are still checked weekly.",
        peer="speaker-75")
    for variant, record, verdict in (
        ("inconsequential", acknowledgement, "supported"),
        ("retained-state-corrected", substantive, "unsupported"),
    ):
        cases.append(_case("effective-noop", variant, [record], prior, prior,
            legacy=verdict, bounded=verdict, noop=True, source_matched=False,
            rationale=(
                "An empty raw summary keeps the effective prior unchanged. A plain "
                "acknowledgement adds no durable state, so that is faithful; a new "
                "explicit correction makes the retained 'enabled' assertion false. "
                "The latter fails because of that contradiction, not because bounded "
                "highlights must represent every unrelated new topic."),
            anchors=[_anchor(record, record["visible_content"],
                "Acknowledgement only." if variant == "inconsequential" else
                "Nightly backup is now disabled; weekly restore checks continue.")],
            selected_meaning="Unchanged prior backup and restore-check assertions.",
            mutation="Change only the new source text; raw empty candidate, effective prior and attribution stay fixed."))

    technician = source("unit-61051", 61051,
        "I am the on-call technician; I completed the rollback after two hours of work.",
        peer="speaker-76")
    observer = source("unit-61052", 61052,
        "As the observer, I intend to inspect the logs, but I have not done that yet.",
        "assistant", peer="speaker-77")
    cases.append(_case("role-and-duration-paraphrase", "faithful",
        [technician, observer],
        "Rollback was finished by the on-call technician after 120 minutes of work, "
        "while the observer's log inspection remained planned and had not happened.",
        "", legacy="supported", bounded="supported",
        rationale=(
            "The source explicitly supplies the two speaker roles, so the candidate "
            "does not invent names from opaque peer IDs. Finished paraphrases completed; "
            "120 minutes preserves two hours of work, not an unsupported clock time "
            "or onset. The observer's intended inspection remains uncompleted."),
        anchors=[
            _anchor(technician, technician["visible_content"],
                "The current user's explicitly named technician role completed the rollback after two hours of work."),
            _anchor(observer, observer["visible_content"],
                "The assistant's explicitly named observer role intends a log inspection that has not occurred."),
        ],
        selected_meaning="Distinct role-bound completed and planned events with exact duration equivalence.",
        source_matched=False))
    return cases
