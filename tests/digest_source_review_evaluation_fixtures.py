"""Invented, pre-labelled mechanism controls; not representative LME samples.

Gold stays outside payload and never enters a request. These fixtures were made
after source-review-v2 froze. No provider responses or production records were
used. Unlabelled checks are descriptive, never silently successful targets.
"""
from copy import deepcopy


def source(cid, mid, text, role="user", *, context=None, peer=None):
    start = context["end"] if context is not None and context["message_id"] == mid else 0
    return {"chunk_id": cid, "message_id": mid, "role": role,
            "source_peer_id": peer, "source_workspace_id": None,
            "start": start, "end": start + len(text), "visible_content": text,
            "interpretation_only_context": context}


def context(mid, text, role="user", *, peer=None):
    return {"message_id": mid, "role": role, "source_peer_id": peer,
            "source_workspace_id": None, "start": 0, "end": len(text), "content": text}


def payload(records, summary, *, title=None, body=None, entities=(), procedure=None, prior=""):
    ids = [r["chunk_id"] for r in records]
    return {"schema": "digest-fidelity-decisions-v9", "source_catalog": records,
        "items": [] if title is None else [{"index": 0, "candidate_title": title,
            "candidate_body": body, "candidate_outcome": "informational",
            "candidate_key_entities": list(entities), "cited_source_ids": ids}],
        "procedure_items": [] if procedure is None else [{"index": 0,
            "candidate": procedure, "cited_source_ids": ids}],
        "summary_item": {"index": 0, "candidate_raw_summary": summary,
            "candidate_summary": summary, "candidate_is_noop": False,
            "new_source_ids": ids, "prior_derived_summary": prior}}


def ground(id_, path, expected, rationale, *, kind="relations", view="primary"):
    return {"id": id_, "view": view, "selectors": [{"kind": kind, "field_path": path}],
            "expected": expected, "rationale": rationale}


def retain(id_, cid, facet, expected, rationale, *, view="retention"):
    return {"id": id_, "view": view,
            "selectors": [{"kind": "retention", "chunk_id": cid, "facet": facet}],
            "expected": expected, "rationale": rationale}


def excluded(cid, facet, reason):
    return {"selector": {"kind": "retention", "chunk_id": cid, "facet": facet},
            "reason": reason}


def procedure(name, actions, *, description=None, triggers=()):
    return {"name": name, "description": description,
            "steps": [{"order": i, "action": action, "tool": None}
                      for i, action in enumerate(actions, 1)],
            "triggers": list(triggers), "entities_involved": []}


def build_cases():
    cases = []

    def pair(name, kind, left, right, labels, paths, change, exclusions=((), ())):
        for i, variant in enumerate(("faithful", "defective")):
            cases.append({"id": f"source-review-{name}-{variant}", "pair": name,
                "variant": variant, "target_scope": {"kind": kind, "index": 0},
                "payload": deepcopy((left, right)[i]), "labels": deepcopy(labels[i]),
                "excluded_checks": deepcopy(list(exclusions[i])),
                "changed_payload_paths": list(paths), "change": change})

    # Cross-message metadata must retain the actual preceding speaker. The same
    # words have a different actor when the two message roles are exchanged.
    text = "observatory? That is your plan, not mine."
    a = payload([source("speaker-plan", 1202, text,
        context=context(1201, "I plan to visit the", "assistant"))],
        "An observatory plan was discussed.", title="Observatory plan attribution",
        body="The user attributes the observatory plan to the assistant and denies making that plan themselves.")
    b = deepcopy(a)
    b["source_catalog"][0]["role"] = "assistant"
    b["source_catalog"][0]["interpretation_only_context"]["role"] = "user"
    excluded_actor = [excluded("speaker-plan", f,
        "Actor changes affect full-candidate retention; this pair isolates attribution grounding, "
        "not the boundary between altered material facts and altered qualifiers.")
        for f in ("material_facts", "constraints")]
    pair("boundary-speaker", "episode", a, b,
        ([ground("speaker", "/candidate_body", "supported", "The current user says it is the preceding assistant's plan, not their own.")],
         [ground("speaker", "/candidate_body", "unsupported", "With roles exchanged, the current assistant denies the plan and attributes it to the preceding user; the unchanged candidate reverses these actors.")]),
        ["/source_catalog/0/role", "/source_catalog/0/interpretation_only_context/role"],
        "Exchange only the roles of the two messages; keep source words and candidate identical.",
        (excluded_actor, excluded_actor))

    ctx = context(1203, "I sailed to Lysfjord, but I wa")
    text = "nt to learn glassblowing."
    a = payload([source("boundary-name", 1203, text, context=ctx)],
        "The user wants to learn glassblowing.", title="Glassblowing interest",
        body="The user wants to learn glassblowing.", entities=("glassblowing",))
    b = deepcopy(a)
    b["items"][0]["candidate_key_entities"] = ["Lysfjord"]
    keep = retain("wish", "boundary-name", "material_facts", "retained",
        "The continuing wish is preserved in the body. A separate context-only trip does not change the wording of that wish.")
    pair("boundary-independent-entity", "episode", a, b,
        ([ground("entity", "/candidate_key_entities/0", "supported", "Glassblowing occurs in the canonical continuation.", kind="assertion"), keep],
         [ground("entity", "/candidate_key_entities/0", "unsupported", "Lysfjord belongs only to an independent, already-consumed trip in boundary context, not the continuing canonical wish.", kind="assertion"), keep]),
        ["/items/0/candidate_key_entities/0"], "Replace only the candidate entity with an independent context-only place.")

    text = "Dome inspection: lock the ring, then read the dial. Never lift the shield during this inspection."
    proc = procedure("Dome inspection", ("Lock the ring", "Read the dial"),
                     description="Never lift the shield during this inspection.")
    a = payload([source("dome", 1204, text, "assistant")], text, procedure=proc)
    b = deepcopy(a)
    b["procedure_items"][0]["candidate"]["description"] = None
    labels = []
    for state in ("retained", "omitted"):
        labels.append([retain("prohibition", "dome", "constraints", state,
            "The explicit shield prohibition is present only in the faithful procedure description; the defective procedure preserves its two steps but says nothing about the shield.", view="primary"),
            ground("remaining-action", "/candidate/steps/1/action", "supported",
                "Reading the dial is an unchanged, explicitly instructed action.", kind="assertion", view="auxiliary")])
    pair("prohibition-omission", "procedure", a, b, labels,
        ["/procedure_items/0/candidate/description"], "Remove only the applicable prohibition.")

    text = "Thermal scan procedure. Start this procedure only when the enclosure is sealed. Read the thermal sensor."
    proc = procedure("Thermal scan", ("Read the thermal sensor",), triggers=("The enclosure is sealed",))
    a = payload([source("enclosure", 1205, text, "assistant")], text, procedure=proc)
    b = deepcopy(a)
    b["procedure_items"][0]["candidate"]["triggers"] = []
    labels = [[retain("start-condition", "enclosure", "constraints", state,
        "The only start condition is the sealed enclosure. It is present as a trigger in one candidate and entirely absent in the other; no contradictory trigger is substituted.", view="primary")]
        for state in ("retained", "omitted")]
    pair("condition-omission", "procedure", a, b, labels,
        ["/procedure_items/0/candidate/triggers"], "Remove only the explicit start-condition trigger.")

    records = [source("assay-request", 1206, "The user requested the final alloy-assay result for sample J-24."),
               source("assay-answer", 1207, "The alloy assay for sample J-24 passed. This is the final result.", "assistant")]
    a = payload(records, "The user requested sample J-24's final alloy-assay result. The final alloy-assay result was a pass.")
    b = deepcopy(a)
    b["summary_item"]["candidate_summary"] = "The user requested sample J-24's final alloy-assay result."
    b["summary_item"]["candidate_raw_summary"] = b["summary_item"]["candidate_summary"]
    labels = []
    for state in ("retained", "omitted"):
        labels.append([retain("answer", "assay-answer", "material_facts", state,
            "The final passed result is material and appears only in the faithful summary; the true request alone does not retain the answer.", view="primary"),
            retain("request", "assay-request", "material_facts", "retained", "Both summaries explicitly retain the request."),
            ground("stated-claims", "/candidate_summary", "supported", "Every assertion actually present is true; completeness is a separate source-retention obligation.", kind="assertion", view="auxiliary")])
    pair("answered-outcome", "summary", a, b, labels,
        ["/summary_item/candidate_summary", "/summary_item/candidate_raw_summary"], "Remove only the answer, retaining the true request.")

    records = [source("seal-first", 1208, "Batch R-62 is sealed. Label band is blue."),
               source("seal-second", 1209, "Batch R-63 is open. Label band is blue.")]
    body = "Batch R-62 is sealed; batch R-63 is open. Both label bands are blue."
    a = payload(records, body, title="Batch seal records", body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_body"] = "Batch R-62 is sealed; batch R-63 is sealed. Both label bands are blue."
    labels = []
    for state in ("retained", "altered"):
        labels.append([retain("second-batch", "seal-second", "material_facts", state,
            "The R-63 open state is preserved in one candidate and changed to sealed in the other.", view="primary"),
            retain("first-batch", "seal-first", "material_facts", "retained",
                "R-62's sealed state and blue label band are unchanged; an error about R-63 cannot change this source-owned judgment.")])
    pair("source-local-status", "episode", a, b, labels,
        ["/items/0/candidate_body"], "Change only the second batch status inside a field containing both sources.")

    text = "Service the optical housing in this required order: disable the emitter before removing the cover."
    proc = procedure("Optical housing service", ("Disable the emitter", "Remove the cover"))
    a = payload([source("emitter-order", 1210, text, "assistant")], text, procedure=proc)
    b = deepcopy(a)
    steps = b["procedure_items"][0]["candidate"]["steps"]
    steps[0]["action"], steps[1]["action"] = steps[1]["action"], steps[0]["action"]
    pair("required-order", "procedure", a, b,
        ([retain("order", "emitter-order", "ordering", "retained", "The first action disables the emitter before the cover is removed.", view="primary")],
         [retain("order", "emitter-order", "ordering", "altered", "Both actions exist, but their required before/after relation is reversed.", view="primary")]),
        ["/procedure_items/0/candidate/steps/0/action", "/procedure_items/0/candidate/steps/1/action"],
        "Swap action strings without altering the normalized numeric step positions.")

    text = "The courier completed the Westdock pickup before delivering the crate to Eastgate. Both actions were completed."
    a = payload([source("courier-order", 1211, text)],
        "The courier completed the Westdock pickup, then delivered the crate to Eastgate.")
    b = deepcopy(a)
    b["summary_item"]["candidate_summary"] = "The courier delivered the crate to Eastgate, then completed the Westdock pickup."
    b["summary_item"]["candidate_raw_summary"] = b["summary_item"]["candidate_summary"]
    pair("material-chronology", "summary", a, b,
        ([retain("chronology", "courier-order", "ordering", "retained", "The two completed actions retain the explicit pickup-before-delivery relation.", view="primary")],
         [retain("chronology", "courier-order", "ordering", "altered", "The same completed actions are placed in the opposite temporal order.", view="primary")]),
        ["/summary_item/candidate_summary", "/summary_item/candidate_raw_summary"], "Reverse only the material chronology.")

    text = "The only approved services are P and Q. Nimbus is available on P and unavailable on Q. This list does not cover other services."
    body = "Nimbus is available on P and unavailable on Q; the approved-service list contains only P and Q."
    a = payload([source("service-scope", 1212, text)], body,
        title="Nimbus is exclusive to P among the approved services", body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_title"] = "Nimbus is exclusive to P across all services"
    pair("exclusivity-scope", "episode", a, b,
        ([ground("scope", "/candidate_title", "supported", "Only P and Q are approved, and Nimbus is present only on P within that explicitly bounded set.")],
         [ground("scope", "/candidate_title", "unsupported", "The record explicitly does not cover other services, so it cannot establish global exclusivity.")]),
        ["/items/0/candidate_title"], "Widen a justified bounded exclusivity claim to an unjustified global claim.")

    text = "The relay restarted at 09:00. The alarm sounded at 09:00. The log does not establish a causal relation."
    body = "The relay restarted and the alarm sounded at 09:00. The log does not establish a causal relation."
    a = payload([source("relay-events", 1213, text)], text, title="Relay and alarm events", body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_body"] = "The relay restart caused the alarm to sound at 09:00."
    pair("causal-inference", "episode", a, b,
        ([ground("causality", "/candidate_body", "supported", "The candidate reports the two recorded contemporaneous events without inventing their relationship.")],
         [ground("causality", "/candidate_body", "unsupported", "Coincidence at 09:00 does not support the added causal link; the source explicitly says causality is not established.")]),
        ["/items/0/candidate_body"], "Replace temporal co-occurrence with an unsupported causal link.")

    text = "I prefer pale glazes."
    a = payload([source("opaque-identity", 1214, text, peer="Neris")],
        "The user prefers pale glazes.", title="Pale glaze preference",
        body="The user prefers pale glazes.", prior="Neris previously discussed gardens.")
    b = deepcopy(a)
    b["items"][0]["candidate_body"] = "A person whose display name is Neris prefers pale glazes."
    pair("opaque-identity", "episode", a, b,
        ([ground("identity", "/candidate_body", "supported", "The current user role supports this first-person attribution without inventing a display name.")],
         [ground("identity", "/candidate_body", "unsupported", "An opaque peer identifier is not a display name. The source does not name its speaker, and prior summary is not episode evidence.")]),
        ["/items/0/candidate_body"], "Replace a role attribution with an unjustified display name inferred from opaque metadata.")

    text = "The inspector's notebook was amber. The final inspection decision accepted panel M-86 without further work."
    a = payload([source("panel-decision", 1215, text)], "The final inspection accepted panel M-86 without further work.")
    b = deepcopy(a)
    b["summary_item"]["candidate_summary"] = "The inspector's notebook was amber."
    b["summary_item"]["candidate_raw_summary"] = b["summary_item"]["candidate_summary"]
    labels = []
    for state in ("retained", "omitted"):
        labels.append([retain("final-decision", "panel-decision", "material_facts", state,
            "Acceptance without further work is the material final decision; notebook colour is incidental and cannot substitute for it.", view="primary"),
            retain("no-order", "panel-decision", "ordering", "not_applicable", "The source imposes no material ordering between actions; notebook colour and the decision are not an ordered workflow."),
            ground("actual-assertion", "/candidate_summary", "supported", "Each variant states a true source fact. Omitting the decision is not fabrication of the surviving assertion.", kind="assertion", view="auxiliary")])
    pair("incidental-versus-outcome", "summary", a, b, labels,
        ["/summary_item/candidate_summary", "/summary_item/candidate_raw_summary"], "Substitute an incidental true detail for the material final decision.")
    # Authoring names above clarify the human review, but must not leak the
    # mechanism through source metadata sent to the model. Pair-stable opaque
    # units derive solely from the already-present message coordinate, not from
    # a case name, variant or expected verdict. Gold remains outside payload.
    for case in cases:
        data = case["payload"]
        mapping = {record["chunk_id"]: f"unit-{record['message_id']}"
                   for record in data["source_catalog"]}
        for record in data["source_catalog"]:
            record["chunk_id"] = mapping[record["chunk_id"]]
        for item in data["items"] + data["procedure_items"]:
            item["cited_source_ids"] = [mapping[cid] for cid in item["cited_source_ids"]]
        summary = data["summary_item"]
        summary["new_source_ids"] = [mapping[cid] for cid in summary["new_source_ids"]]
        selectors = [selector for label in case["labels"] for selector in label["selectors"]]
        selectors.extend(entry["selector"] for entry in case["excluded_checks"])
        for selector in selectors:
            if selector["kind"] == "retention":
                selector["chunk_id"] = mapping[selector["chunk_id"]]
    return cases
