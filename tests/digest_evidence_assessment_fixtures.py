"""New invented compact-assessment controls with separately reviewed obligations.

These are diagnostic development controls, not representative or held-out LME
samples. Gold is outside ``payload`` and must never enter a model request. An
unsupported primary check is not interchangeable with an unsupported retention
check; omission-only controls deliberately keep selected assertions supported.
No live responses or production records are used to construct these cases.
"""
from copy import deepcopy


def _source(chunk_id, message_id, text, role="user"):
    return {
        "chunk_id": chunk_id, "message_id": message_id, "role": role,
        "source_peer_id": None, "source_workspace_id": None,
        "start": 0, "end": len(text), "visible_content": text,
        "interpretation_only_context": None,
    }


def _payload(sources, summary, *, title=None, body=None, outcome="informational",
             prior="", procedure=None):
    ids = [source["chunk_id"] for source in sources]
    return {
        "schema": "digest-fidelity-decisions-v9",
        "source_catalog": sources,
        "items": [] if title is None else [{
            "index": 0, "candidate_title": title, "candidate_body": body,
            "candidate_outcome": outcome, "candidate_key_entities": [],
            "cited_source_ids": ids,
        }],
        "procedure_items": [] if procedure is None else [{
            "index": 0, "cited_source_ids": ids, "candidate": procedure,
        }],
        "summary_item": {
            "index": 0, "candidate_raw_summary": summary,
            "candidate_summary": summary, "candidate_is_noop": False,
            "new_source_ids": ids, "prior_derived_summary": prior,
        },
    }


def _field(kind, path, expected, rationale):
    return {"selector": {"kind": kind, "field_path": path},
            "expected": expected, "rationale": rationale}


def _retention(chunk_id, expected, rationale):
    return {"selector": {"kind": "retention", "chunk_id": chunk_id},
            "expected": expected, "rationale": rationale}


def _summary_assertions(rationale):
    return [_field("assertion", "/candidate_summary", "supported", rationale),
            _field("relations", "/candidate_summary", "supported", rationale)]


def build_cases():
    """Return independent mutable copies of 8 explicit faithful/defective pairs.

    ``primary`` supplies one preregistered mechanism-specific decision per case.
    ``retention_labels`` independently covers every canonical-text unit in the
    target scope, including a primary retention decision where applicable.
    ``auxiliary_labels`` is a separate denominator, never extra primary credit.
    Three explicitly identified retention labels have ``expected=None`` because
    full-candidate preservation under an extra incompatible claim is ambiguous.
    They still document canonical coverage, but must be excluded from scoring,
    not treated as uncertain gold or credited as either success or failure. This
    exclusion was made before model outputs; variant names never determine gold.
    ``changed_payload_paths`` documents the complete pair mutation, using JSON
    pointers into the input payload, not model-assigned IDs.
    """
    cases = []

    def pair(name, kind, faithful, defective, primaries, retention,
             changed_paths, change, auxiliary=((), ())):
        for index, variant in enumerate(("faithful", "defective")):
            cases.append({
                "id": f"assessment-{name}-{variant}", "pair": name,
                "variant": variant, "target_scope": {"kind": kind, "index": 0},
                "payload": deepcopy((faithful, defective)[index]),
                "primary": deepcopy(primaries[index]),
                "retention_labels": deepcopy(retention[index]),
                "auxiliary_labels": deepcopy(list(auxiliary[index])),
                "changed_payload_paths": list(changed_paths), "change": change,
            })

    source = _source("causal-nerys", 901,
        "The seminar ended at noon. Nerys left at noon. "
        "The record does not connect these two events causally.")
    body = "The seminar ended at noon. Nerys left at noon."
    faithful = _payload([source], body, title="Seminar and departure", body=body)
    defective = deepcopy(faithful)
    defective["items"][0]["candidate_body"] += " The seminar ending caused Nerys to leave."
    retained = [_retention("causal-nerys", "supported",
        "Both material event outcomes and their noon timing remain explicitly present. "
        "The extra causal assertion is evaluated by the primary relations check, "
        "not counted as an omitted event.")]
    pair("causal-suffix", "episode", faithful, defective,
         (_field("relations", "/candidate_body", "supported",
                 "Two noon events are reported without inferring a causal link."),
          _field("relations", "/candidate_body", "unsupported",
                 "The added causal relation has no source support; the record explicitly "
                 "does not connect the events causally.")),
         (retained, [_retention("causal-nerys", None,
             "Unscored before observing model outputs: the original event sentences remain, "
             "but the added causal claim may impair full-candidate preservation. The interaction "
             "between assertion fidelity and retention is not sufficiently specified to force "
             "either label. Exclusion is neither a credited success nor a credited failure.")]),
         ["/items/0/candidate_body"],
         "Append one unsupported causal sentence; retain both original event sentences.")

    source = _source("identity-zevin", 902, "I completed the bronze-casting course.")
    body = "The user completed the bronze-casting course."
    faithful = _payload([source], body, title="Course completion", body=body,
        outcome="resolved", prior="Zevin previously asked about sculpture exhibitions.")
    defective = deepcopy(faithful)
    defective["items"][0]["candidate_body"] += " The user's name is Zevin."
    retained = [_retention("identity-zevin", "supported",
        "The complete own-source course-completion outcome remains in the unchanged "
        "first sentence; the invented name is a separate identity-support defect.")]
    pair("prior-only-identity", "episode", faithful, defective,
         (_field("relations", "/candidate_body", "supported",
                 "The user role authorizes this first-person attribution without a display name."),
          _field("relations", "/candidate_body", "unsupported",
                 "Neither the own source nor null peer/workspace metadata establishes Zevin. "
                 "The prior summary is not episode authority and does not name this speaker.")),
         (retained, [_retention("identity-zevin", None,
             "Unscored before observing model outputs: the original user-completion sentence "
             "remains, but the invented identity may impair full-candidate preservation of its "
             "actor. This unresolved retention-versus-attribution interaction does not justify "
             "either label. Exclusion is neither a credited success nor a credited failure.")]),
         ["/items/0/candidate_body"],
         "Append an identity claim obtainable only by an unauthorized prior-summary inference.")

    text = ("The user deliberately postponed antenna calibration until next Thursday. "
            "No calibration has been performed.")
    faithful = _payload([_source("deferred-antenna", 903, text, "assistant")], text,
        title="Antenna calibration postponed", body=text, outcome="deferred")
    defective = deepcopy(faithful)
    defective["items"][0]["candidate_outcome"] = "resolved"
    retained = [_retention("deferred-antenna", "supported",
        "The unchanged body preserves postponement, next-Thursday timing and noncompletion. "
        "The conflicting category is measured independently by the outcome check.")]
    pair("categorical-outcome", "episode", faithful, defective,
         (_field("outcome", "/candidate_outcome", "supported",
                 "Deliberate postponement with no performed calibration is a deferred event."),
          _field("outcome", "/candidate_outcome", "unsupported",
                 "Explicit noncompletion cannot establish the actual completion required by resolved.")),
         (retained, [_retention("deferred-antenna", None,
             "Unscored before observing model outputs: the body preserves postponement and "
             "noncompletion, but the contradictory resolved category may impair full-candidate "
             "preservation. The retention-versus-classification interaction is unresolved, so "
             "neither label is forced. Exclusion is neither a credited success nor a credited failure.")]),
         ["/items/0/candidate_outcome"],
         "Change only the categorical outcome from deferred to resolved.")

    sources = [
        _source("audit-request", 904, "The user requested a checksum audit of packet K-42."),
        _source("audit-answer", 905,
            "Final audit result for packet K-42: the audit failed. "
            "A retry was scheduled for Tuesday.", "assistant"),
    ]
    faithful = _payload(sources,
        "The user requested a checksum audit of packet K-42. The audit failed; "
        "a retry was scheduled for Tuesday.")
    defective = deepcopy(faithful)
    defective["summary_item"]["candidate_summary"] = (
        "The user requested a checksum audit of packet K-42.")
    defective["summary_item"]["candidate_raw_summary"] = defective["summary_item"]["candidate_summary"]
    request = _retention("audit-request", "supported", "The audit request is explicitly preserved.")
    answer_yes = _retention("audit-answer", "supported",
        "The final failed result and scheduled Tuesday retry are both preserved.")
    answer_no = _retention("audit-answer", "unsupported",
        "The summary preserves only the request, omitting the material final failure "
        "and retry decision from the answer.")
    aux = _summary_assertions("Every assertion actually present is supported. Missing final "
        "outcomes are measured by retention, not by treating true request text as false.")
    pair("answered-outcome-omission", "summary", faithful, defective,
         (answer_yes, answer_no), ([request, answer_yes], [request, answer_no]),
         ["/summary_item/candidate_raw_summary", "/summary_item/candidate_summary"],
         "Remove only the answered outcome and retry sentence from the effective/raw summary.",
         (aux, aux))

    text = ("Baffle check procedure: first close the sample valve, then read the gauge. "
            "Never release the vent during this procedure.")
    procedure = {"name": "Baffle check", "description": "Never release the vent during this procedure.",
        "steps": [{"order": 1, "action": "Close the sample valve", "tool": None},
                  {"order": 2, "action": "Read the gauge", "tool": None}],
        "triggers": [], "entities_involved": []}
    faithful = _payload([_source("baffle-prohibition", 906, text, "assistant")], text,
                        procedure=procedure)
    defective = deepcopy(faithful)
    defective["procedure_items"][0]["candidate"]["description"] = None
    yes = _retention("baffle-prohibition", "supported",
        "The candidate preserves both ordered steps and the applicable explicit vent prohibition.")
    no = _retention("baffle-prohibition", "unsupported",
        "Both steps remain correct, but the only applicable explicit prohibition is omitted "
        "from the entire procedure candidate; another scope's summary cannot supply it.")
    aux = [_field("assertion", f"/candidate/steps/{index}/action", "supported",
            "This unchanged action is explicitly instructed by the procedure source.") for index in (0, 1)]
    pair("prohibition-omission", "procedure", faithful, defective,
         (yes, no), ([yes], [no]), ["/procedure_items/0/candidate/description"],
         "Remove only the prohibition description; preserve every action and its order.", (aux, aux))

    text = ("Isolation sequence: unplug the pump before opening its cover. "
            "The required sequence is first unplug the pump, then open its cover.")
    procedure = {"name": "Isolation sequence", "description": None,
        "steps": [{"order": 1, "action": "Unplug the pump", "tool": None},
                  {"order": 2, "action": "Open the pump cover", "tool": None}],
        "triggers": [], "entities_involved": []}
    faithful = _payload([_source("pump-order", 907, text, "assistant")], text, procedure=procedure)
    defective = deepcopy(faithful)
    steps = defective["procedure_items"][0]["candidate"]["steps"]
    steps[0]["action"], steps[1]["action"] = steps[1]["action"], steps[0]["action"]
    pair("cross-field-order", "procedure", faithful, defective,
         (_field("relations", "/candidate/steps/0/action", "supported",
                 "The full candidate orders unplugging before opening, exactly as instructed."),
          _field("relations", "/candidate/steps/0/action", "unsupported",
                 "The full candidate makes opening the first action while unplugging is second; "
                 "each verb's presence does not preserve the required cross-field order.")),
         ([_retention("pump-order", "supported", "Both actions and their required order are preserved.")],
          [_retention("pump-order", "unsupported", "The mandatory before/after relation is reversed.")]),
         ["/procedure_items/0/candidate/steps/0/action", "/procedure_items/0/candidate/steps/1/action"],
         "Swap only the two action strings; numeric step positions remain one then two.")

    text = ("The archivist's pencil was yellow. The restoration review concluded that "
            "folio V-83 was accepted without further work. This acceptance is the final review decision.")
    faithful = _payload([_source("folio-review", 908, text)],
        "The restoration review accepted folio V-83 without further work.")
    defective = deepcopy(faithful)
    defective["summary_item"]["candidate_summary"] = "The archivist's pencil was yellow."
    defective["summary_item"]["candidate_raw_summary"] = defective["summary_item"]["candidate_summary"]
    yes = _retention("folio-review", "supported",
        "The final acceptance and no-further-work decision are preserved; pencil color is "
        "incidental and need not be repeated.")
    no = _retention("folio-review", "unsupported",
        "The summary retains only incidental pencil color and omits the explicitly final review decision.")
    aux = _summary_assertions("The summary's actual assertion is a faithful source fact in both "
        "variants; retaining an incidental fact instead of the decision is an omission, not fabrication.")
    pair("incidental-versus-material", "summary", faithful, defective,
         (yes, no), ([yes], [no]),
         ["/summary_item/candidate_raw_summary", "/summary_item/candidate_summary"],
         "Replace the material decision summary with a true but incidental detail; neither invents a fact.",
         (aux, aux))

    sources = [
        _source("unicode-closed", 909,
            "Ticket Ω-17 status: closed. Café check complete. Café check complete."),
        _source("unicode-pending", 910,
            "Ticket Ω-18 status: pending. Café check complete. Café check complete."),
    ]
    body = "Ticket Ω-17 is closed; ticket Ω-18 is pending. Café check complete."
    faithful = _payload(sources, body, title="Ticket status report", body=body)
    defective = deepcopy(faithful)
    defective["items"][0]["candidate_body"] = (
        "Ticket Ω-17 is closed; ticket Ω-18 is closed. Café check complete.")
    closed = _retention("unicode-closed", "supported",
        "Ticket Ω-17's closed status and check completion are preserved in both variants.")
    pair("repeated-unicode-units", "episode", faithful, defective,
         (_field("assertion", "/candidate_body", "supported",
                 "Both ticket-specific statuses and the repeated check-completion statement are supported."),
          _field("assertion", "/candidate_body", "unsupported",
                 "The Ω-18 source says pending, not closed. Identical Café sentences in distinct "
                 "units cannot transfer Ω-17's closed status to Ω-18.")),
         ([closed, _retention("unicode-pending", "supported",
                    "Ticket Ω-18's pending status and check completion are preserved.")],
          [closed, _retention("unicode-pending", "unsupported",
                    "Ticket Ω-18's material pending outcome is replaced by closed, not preserved.")]),
         ["/items/0/candidate_body"],
         "Replace only Ω-18's pending status with closed in the body; preserve Unicode and repeated evidence.")
    return cases
