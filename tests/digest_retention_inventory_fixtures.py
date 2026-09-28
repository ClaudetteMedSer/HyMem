"""Invented development controls for candidate-blind retention inventories.

These are mechanism controls, not held-out LME data or model-accuracy evidence.
Only ``payload`` is a request input. Expected obligations, matching judgments,
unit-coordinate annotations, and incidental-detail judgments are external gold.
Their list is targeted, not a claim to exhaust every obligation in the scope.
"""
from copy import deepcopy

from tests.digest_source_review_evaluation_fixtures import (
    context, payload, procedure, source,
)


def _obligation(identifier, record, quote, description, *, facet="constraints"):
    start = record["visible_content"].index(quote)
    assert record["visible_content"].count(quote) == 1
    return {"id": identifier, "chunk_id": record["chunk_id"], "facet": facet,
            "description": description, "canonical_quote": quote,
            "canonical_start": record["start"] + start,
            "canonical_end": record["start"] + start + len(quote)}


def _match(identifier, expected, paths, rationale):
    return {"obligation_id": identifier, "expected": expected,
            "candidate_field_paths": list(paths), "rationale": rationale}


def build_cases():
    """Return independent v9 payloads plus deliberately separate targeted gold."""
    cases = []

    def pair(name, kind, left, right, obligations, matches, paths, change,
             *, unit_chars=256, variants=("faithful", "defective"),
             optional_details=()):
        for position, variant in enumerate(variants):
            cases.append({
                "id": f"retention-inventory-{name}-{variant}", "pair": name,
                "variant": variant, "target_scope": {"kind": kind, "index": 0},
                "payload": deepcopy((left, right)[position]),
                "unit_chars": unit_chars,
                "expected_source_obligations": deepcopy(obligations),
                "candidate_matches": deepcopy(matches[position]),
                "optional_details": deepcopy(list(optional_details)),
                "changed_payload_paths": list(paths), "change": change,
                "label_scope": "targeted_obligations_only",
                "whole_scope_correctness_claim": False,
            })

    record = source("unit-3401", 3401,
        "Scan only approved sheets. Never leave the scanner unattended during a scan.",
        "assistant")
    candidate = procedure("Sheet scanning", ("Scan the sheet",),
        triggers=("The sheet has approval",),
        description="Remain beside the scanner throughout the scan.")
    faithful = payload([record], record["visible_content"], procedure=candidate)
    defective = deepcopy(faithful)
    defective["procedure_items"][0]["candidate"]["description"] = None
    obligations = [
        _obligation("approved-sheet", record, "Scan only approved sheets.",
                    "Scanning is permitted only for approved sheets."),
        _obligation("supervised-scan", record,
                    "Never leave the scanner unattended during a scan.",
                    "The scanner must remain attended throughout scanning."),
    ]
    fixed = _match("approved-sheet", "retained", ("/candidate/triggers/0",),
                   "Both procedures keep the approval prerequisite.")
    pair("same-unit-two-constraints", "procedure", faithful, defective,
         obligations,
         ([fixed, _match("supervised-scan", "retained", ("/candidate/description",),
                         "Remaining beside the scanner preserves the prohibition by paraphrase.")],
          [fixed, _match("supervised-scan", "omitted", (),
                         "The remaining action and trigger say nothing about supervision.")]),
         ("/procedure_items/0/candidate/description",),
         "Delete only the supervision description; both source constraints occupy one location unit.")

    # Natural, meaningful prose puts the prerequisite across the 256-codepoint
    # boundary. The accent also makes byte offsets differ from codepoints.
    text = ("This note describes the label collection procedure for the café room. "
            "The cabinet stands beside the noticeboard, and the labels are stored on its middle shelf. "
            "This location description is background information. "
            "Open the supply cabinet only after the coordinator confirms the room is empty. "
            "Then collect the labels.")
    record = source("unit-3402", 3402, text, "assistant")
    candidate = procedure("Label collection", ("Open the supply cabinet", "Collect the labels"),
        triggers=("The coordinator has confirmed that the room is empty",))
    faithful = payload([record], text, procedure=candidate)
    defective = deepcopy(faithful)
    defective["procedure_items"][0]["candidate"]["triggers"] = []
    obligation = _obligation("room-clearance", record,
        "Open the supply cabinet only after the coordinator confirms the room is empty.",
        "Opening the cabinet requires the coordinator's prior confirmation that the room is empty.")
    pair("cross-unit-prerequisite", "procedure", faithful, defective, [obligation],
         ([_match("room-clearance", "retained", ("/candidate/triggers/0",),
                  "The trigger retains both the confirming actor and the empty-room condition.")],
          [_match("room-clearance", "omitted", (),
                  "The two actions remain but neither contains the prerequisite.")]),
         ("/procedure_items/0/candidate/triggers",),
         "Delete only the prerequisite trigger, leaving both actions and source prose unchanged.")

    text = "Dispatch a replacement only after the customer confirms that they have not received the original parcel."
    record = source("unit-3403", 3403, text, "assistant")
    candidate = procedure("Replacement dispatch", ("Dispatch the replacement",),
        triggers=("The customer has confirmed nonreceipt of the original parcel",))
    faithful = payload([record], text, procedure=candidate)
    defective = deepcopy(faithful)
    defective["procedure_items"][0]["candidate"]["triggers"][0] = (
        "The customer has confirmed receipt of the original parcel")
    obligation = _obligation("nonreceipt-confirmation", record, text,
        "Dispatch requires customer confirmation of nonreceipt of the original parcel.")
    pair("paraphrase-versus-altered-condition", "procedure", faithful, defective,
         [obligation],
         ([_match("nonreceipt-confirmation", "retained", ("/candidate/triggers/0",),
                  "Nonreceipt faithfully paraphrases not having received the original parcel.")],
          [_match("nonreceipt-confirmation", "altered", ("/candidate/triggers/0",),
                  "Receipt reverses the required nonreceipt condition.")]),
         ("/procedure_items/0/candidate/triggers/0",),
         "Change only nonreceipt to receipt in a paraphrased prerequisite.")

    text = "Archive the signed attendance form. Never include participants' home addresses in that archive."
    record = source("unit-3404", 3404, text, "assistant")
    candidate = procedure("Attendance archiving", ("Archive the signed attendance form",),
        description="Do not put participants' home addresses in the archive.")
    faithful = payload([record], text, procedure=candidate)
    defective = deepcopy(faithful)
    defective["procedure_items"][0]["candidate"]["description"] = None
    obligation = _obligation("address-exclusion", record,
        "Never include participants' home addresses in that archive.",
        "Participants' home addresses must be excluded from the attendance archive.")
    pair("explicit-prohibition", "procedure", faithful, defective, [obligation],
         ([_match("address-exclusion", "retained", ("/candidate/description",),
                  "The description retains the applicable address prohibition.")],
          [_match("address-exclusion", "omitted", (),
                  "The archiving step alone does not preserve the address prohibition.")]),
         ("/procedure_items/0/candidate/description",),
         "Delete only the prohibition, preserving the affirmative action.")

    same_message = source("unit-3406", 3406, "mosaic tray.",
        context=context(3406, "I will carry the "))
    different_message = source("unit-3408", 3408,
        "I will carry the paper roll; you will carry the frame.",
        context=context(3407, "I will carry the frame.", "assistant"))
    faithful_body = ("The user will carry the mosaic tray and paper roll. "
                     "The assistant will carry the frame.")
    faithful = payload([same_message, different_message], faithful_body,
        title="Carrying assignments", body=faithful_body)
    defective = deepcopy(faithful)
    defective["items"][0]["candidate_body"] = (
        "The user will carry the mosaic tray, paper roll and frame.")
    obligations = [
        _obligation("tray-carrier", same_message, "mosaic tray.",
                    "The user will carry the mosaic tray, completing the same-message first-person clause.",
                    facet="material_facts"),
        _obligation("roll-carrier", different_message, "I will carry the paper roll",
                    "The current user will carry the paper roll.", facet="material_facts"),
        _obligation("frame-carrier", different_message, "you will carry the frame.",
                    "The assistant, addressed by the current user, will carry the frame.",
                    facet="material_facts"),
    ]
    fixed = [_match(identifier, "retained", ("/candidate_body",),
                    "The candidate retains this user assignment in both variants.")
             for identifier in ("tray-carrier", "roll-carrier")]
    pair("boundary-message-attribution", "episode", faithful, defective, obligations,
         (fixed + [_match("frame-carrier", "retained", ("/candidate_body",),
                         "The assistant remains the frame carrier.")],
          fixed + [_match("frame-carrier", "altered", ("/candidate_body",),
                         "Only the frame's carrier changes from assistant to user.")]),
         ("/items/0/candidate_body",),
         "Change only the frame assignment; same-message and preceding-message authority remain intact.")

    text = ("The scratch paper is yellow; its color is unrelated to the decision. "
            "The booking request for room C-7 was approved without conditions.")
    record = source("unit-3409", 3409, text)
    decision = "The booking request for room C-7 was approved without conditions."
    detailed = payload([record], decision + " The scratch paper is yellow.")
    concise = deepcopy(detailed)
    for key in ("candidate_summary", "candidate_raw_summary"):
        concise["summary_item"][key] = decision
    obligation = _obligation("booking-decision", record, decision,
        "Room C-7's booking request was approved without conditions.", facet="material_facts")
    match = _match("booking-decision", "retained", ("/candidate_summary",),
        "Both summaries preserve the approval and lack of conditions; scratch-paper color is incidental.")
    pair("incidental-detail-omission", "summary", detailed, concise, [obligation],
         ([match], [match]),
         ("/summary_item/candidate_summary", "/summary_item/candidate_raw_summary"),
         "Omit only an explicitly irrelevant color detail; both candidates are faithful for the labelled target.",
         variants=("faithful_detailed", "faithful_concise"),
         optional_details=({"chunk_id": record["chunk_id"],
                            "canonical_quote": "The scratch paper is yellow",
                            "reason": "The source explicitly says the color is unrelated to the decision; its omission is legitimate."},))
    return cases
