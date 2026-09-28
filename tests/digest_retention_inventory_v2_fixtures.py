"""Schema-correct, versioned development controls for procedure field roles.

The twelve earlier controls are copied, never rewritten in place. Their source
bytes, ownership, codepoint coordinates and targeted source obligations remain
unchanged. Procedure assertions now use description/steps; retrieval triggers
are query phrases, not mandatory preconditions. Gold is external to payload.
An additional pair isolates a condition mentioned only in a retrieval phrase.
These are development controls, not held-out or measured model accuracy.
"""
from copy import deepcopy

from tests.digest_retention_inventory_fixtures import build_cases as _old_cases


FIXTURE_VERSION = "retention-inventory-development-v2"
TRIGGER_ONLY_PAIR = "retrieval-trigger-only-condition"


def _candidate(case):
    return case["payload"]["procedure_items"][0]["candidate"]


def _match(case, identifier):
    return next(match for match in case["candidate_matches"]
                if match["obligation_id"] == identifier)


def build_cases():
    """Return twelve corrected copies plus a two-case retrieval-role control."""
    cases = deepcopy(_old_cases())
    for case in cases:
        case["fixture_version"] = FIXTURE_VERSION
        case["revision_of"] = case["id"]
        case["id"] = case["id"].replace("retention-inventory-", "retention-inventory-v2-", 1)
        if case["target_scope"]["kind"] != "procedure":
            continue
        candidate = _candidate(case)
        faithful = case["variant"] == "faithful"
        if case["pair"] == "same-unit-two-constraints":
            candidate["steps"][0]["action"] = "Scan only approved sheets"
            candidate["triggers"] = ["scan a sheet", "sheet scanning"]
            candidate["description"] = (
                "Scan approved sheets while remaining beside the scanner throughout the scan."
                if faithful else "Scan approved sheets.")
            fixed = _match(case, "approved-sheet")
            fixed["candidate_field_paths"] = ["/candidate/steps/0/action"]
            fixed["rationale"] = "The action explicitly limits scanning to approved sheets in both variants."
            target = _match(case, "supervised-scan")
            target["rationale"] = (
                "Remaining beside the scanner throughout scanning preserves mandatory supervision by paraphrase."
                if faithful else "Neither the neutral description nor the approved-sheet action retains supervision.")
            case["change"] = (
                "Replace only the supervision description with a valid neutral description; "
                "both source constraints remain in one location unit and approval stays in the action.")
        elif case["pair"] == "cross-unit-prerequisite":
            candidate["description"] = "Collect the labels from the supply cabinet."
            candidate["triggers"] = ["collect labels", "open supply cabinet"]
            candidate["steps"][0]["action"] = (
                "Open the supply cabinet only after the coordinator confirms that the room is empty"
                if faithful else "Open the supply cabinet")
            match = _match(case, "room-clearance")
            match["candidate_field_paths"] = ["/candidate/steps/0/action"] if faithful else []
            match["rationale"] = (
                "The opening action requires the coordinator's prior confirmation that the room is empty."
                if faithful else "The actions and neutral description omit the necessary prior confirmation.")
            case["changed_payload_paths"] = ["/procedure_items/0/candidate/steps/0/action"]
            case["change"] = "Remove only the opening action's prerequisite, preserving its action and all retrieval metadata."
        elif case["pair"] == "paraphrase-versus-altered-condition":
            candidate["description"] = "Dispatch a replacement parcel."
            candidate["triggers"] = ["replacement parcel", "dispatch a replacement"]
            candidate["steps"][0]["action"] = (
                "Dispatch the replacement only after the customer confirms "
                + ("nonreceipt" if faithful else "receipt") + " of the original parcel")
            match = _match(case, "nonreceipt-confirmation")
            match["candidate_field_paths"] = ["/candidate/steps/0/action"]
            match["rationale"] = (
                "The action's required nonreceipt confirmation paraphrases not having received the original parcel."
                if faithful else "The action requires receipt, reversing the necessary nonreceipt condition.")
            case["changed_payload_paths"] = ["/procedure_items/0/candidate/steps/0/action"]
            case["change"] = "Change only nonreceipt to receipt in the action's mandatory prerequisite."
        elif case["pair"] == "explicit-prohibition":
            candidate["triggers"] = ["archive attendance form", "attendance archiving"]
            if not faithful:
                candidate["description"] = "Archive the signed attendance form."
            case["change"] = "Replace only the address prohibition with a valid neutral description, preserving the affirmative action."

    # The same condition-bearing retrieval phrase occurs in BOTH candidates.
    # Only the assertion in the first action differs. Treating a query phrase as
    # an execution rule would therefore falsely accept the negative control.
    for original in [case for case in cases if case["pair"] == "cross-unit-prerequisite"]:
        case = deepcopy(original)
        case["revision_of"] = original["id"]
        case["pair"] = TRIGGER_ONLY_PAIR
        case["id"] = f"retention-inventory-v2-{TRIGGER_ONLY_PAIR}-{case['variant']}"
        _candidate(case)["triggers"] = [
            "opening the supply cabinet after coordinator confirmation of an empty room"]
        case["change"] = (
            "Remove only the mandatory condition from the opening action; the identical "
            "condition-bearing retrieval phrase is not an execution prerequisite.")
        match = _match(case, "room-clearance")
        match["rationale"] = (
            "The opening action explicitly preserves the mandatory prerequisite, independently of retrieval wording."
            if case["variant"] == "faithful" else
            "The required condition appears only in a retrieval phrase; neither assertion field makes it mandatory.")
        cases.append(case)
    return cases
