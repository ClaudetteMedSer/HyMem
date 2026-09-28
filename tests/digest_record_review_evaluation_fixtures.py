"""Pre-response record-v3 controls; annotations never enter model requests.

Fresh means authored after the candidate froze, not representative LME holdout.
Replay payloads are exact earlier controls with independently justified new
facet labels. No prior verdict is used to construct labels or model inputs.
"""
from copy import deepcopy

from tests.digest_source_review_evaluation_fixtures import (
    build_cases as previous_cases, context, payload, procedure, source,
)


def ground(id_, kind, path, expected, rationale, view="primary"):
    return {"id": id_, "view": view, "selectors": [{"kind": kind, "field_path": path}],
            "expected": expected, "rationale": rationale}


def retain(cid, facet, expected, rationale, view="primary"):
    return {"id": facet, "view": view,
            "selectors": [{"kind": "retention", "chunk_id": cid, "facet": facet}],
            "expected": expected, "rationale": rationale}


def build_cases():
    cases = []

    def pair(name, left, right, labels, paths, *, kind="episode", cohort="fresh",
             faithful_reason, exclusions=()):
        for i, variant in enumerate(("faithful", "defective")):
            cases.append({"id": f"record-eval-{name}-{variant}", "pair": name,
                "variant": variant, "cohort": cohort,
                "target_scope": {"kind": kind, "index": 0},
                "payload": deepcopy((left, right)[i]), "labels": deepcopy(labels[i]),
                "changed_payload_paths": list(paths),
                "faithful_scope_acceptance": variant == "faithful",
                "faithful_scope_rationale": faithful_reason if i == 0 else None,
                "excluded_checks": deepcopy(list(exclusions))})

    # Replays: explicit new labels, never a mapping of old aggregate gold.
    old = {(c["pair"], c["variant"]): c for c in previous_cases()}
    specs = (
        ("boundary-speaker", "actor_attribution", "/candidate_body",
         "The user rejects ownership and attributes the preceding plan to the assistant; the opposite role assignment reverses this ownership.",
         "The faithful title is topical, the body preserves the user's attribution and denial, and the exchange reports a plan without claiming completion."),
        ("opaque-identity", "identity", "/candidate_body",
         "The first-person preference belongs to the current user; no canonical text establishes the display name Neris. Opaque metadata and an unrelated prior summary cannot name the speaker.",
         "The faithful candidate states only the user's unchanged glaze preference, has no inferred display name, and makes no task-completion claim."),
        ("exclusivity-scope", "quantified_scope", "/candidate_title",
         "P and Q exhaust the approved domain and only P offers Nimbus there. The source explicitly does not establish exclusivity over all services.",
         "The faithful title stays within the complete approved set, the body preserves membership and availability, and no global exclusivity or completion is claimed."),
    )
    for name, facet, field, reason, faithful_reason in specs:
        a, b = (old[(name, v)] for v in ("faithful", "defective"))
        labels = [[ground("mechanism", facet, field, expected, reason)]
                  for expected in ("supported", "unsupported")]
        if facet == "quantified_scope":
            for rows, expected in zip(labels, ("supported", "unsupported")):
                rows.append(ground("title-assertion", "assertion", field, expected,
                    "The title itself asserts this domain restriction; its truth must agree with the explicit bounded/global distinction.", "auxiliary"))
        pair("replay-" + name, a["payload"], b["payload"], labels,
             a["changed_payload_paths"], cohort="replay", faithful_reason=faithful_reason,
             exclusions=a["excluded_checks"])

    text = "You intend to register the glider; I intend to reserve the cart."
    body = "The user intends to register the glider; the assistant intends to reserve the cart."
    a = payload([source("unit-9202", 9202, text, "assistant",
        context=context(9201, "I am planning next month's field trip.", "user"))],
        body, title="Field trip intentions", body=body)
    b = deepcopy(a)
    b["source_catalog"][0]["role"] = "user"
    b["source_catalog"][0]["interpretation_only_context"]["role"] = "assistant"
    pair("dialogue-owner", a, b,
        [[ground("mechanism", "actor_attribution", "/candidate_body", e,
          "In a two-speaker exchange, the current assistant's I is the assistant and you is the user. Swapping the two message roles reverses those intentions.")]
         for e in ("supported", "unsupported")],
        ["/source_catalog/0/role", "/source_catalog/0/interpretation_only_context/role"],
        faithful_reason="Both intentions are retained with the correct speakers; the topical title and informational outcome add no accomplishment.")

    text = 'Mara told me, "I plan to repair the mast." I plan only to deliver paint.'
    body = "Mara told the assistant about Mara's plan to repair the mast; the assistant plans only to deliver paint."
    a = payload([source("unit-9203", 9203, text, "assistant")], body,
                title="Mast work plans", body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_body"] = "The assistant told Mara about the assistant's plan to repair the mast; Mara plans only to deliver paint."
    pair("reported-intentions", a, b,
        [[ground("mechanism", "actor_attribution", "/candidate_body", e,
          "Quoted I belongs to the named Mara, while the reporting first person belongs to the assistant. The defective body exchanges the reported intention, report recipient and the delivery plan.")]
         for e in ("supported", "unsupported")], ["/items/0/candidate_body"],
        faithful_reason="Mara's intention remains explicitly reported to the assistant, both plans retain their actors, and the assistant's only-deliver limitation is preserved; neither plan is reported completed.")

    body = "The assistant would like to study harbour acoustics next spring."
    a = payload([source("unit-9204", 9204, "harbour acoustics next spring.", "assistant",
        context=context(9204, "I would like to study ", "assistant"))], body,
        title="Acoustics study interest", body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_body"] = "The user would like to study harbour acoustics next spring."
    pair("contiguous-speaker", a, b,
        [[ground("mechanism", "actor_attribution", "/candidate_body", e,
          "The contiguous prefix and canonical suffix belong to the same assistant message. The user is not the first-person speaker of this interest.")]
         for e in ("supported", "unsupported")], ["/items/0/candidate_body"],
        faithful_reason="Boundary context resolves only the continuing canonical interest; topic, speaker, modality and timing are preserved.")

    body = "The speaker uses the alias Rook-6 and prefers matt paper."
    a = payload([source("unit-9205", 9205,
        "I use the alias Rook-6. I prefer matt paper.", peer="Rook-6")],
        body, title="Paper preference", body=body)
    b = deepcopy(a)
    b["source_catalog"][0]["visible_content"] = "I prefer matt paper."
    b["source_catalog"][0]["end"] = len("I prefer matt paper.")
    pair("canonical-alias", a, b,
        [[ground("mechanism", "identity", "/candidate_body", e,
          "A canonical self-declared alias names this speaker. When that declaration is removed, the identical opaque peer value cannot establish the alias.")]
         for e in ("supported", "unsupported")],
        ["/source_catalog/0/visible_content", "/source_catalog/0/end"],
        faithful_reason="The candidate preserves the explicit alias and preference without adding other facts or an accomplished task.")

    body = "The user maintains station archives."
    a = payload([source("unit-9206", 9206, "I maintain station archives.")],
        body, title="Station archive work", body=body)
    a["source_catalog"][0]["source_workspace_id"] = "Juno"
    b = deepcopy(a)
    b["items"][0]["candidate_body"] = "A person named Juno maintains station archives."
    pair("workspace-not-person", a, b,
        [[ground("mechanism", "identity", "/candidate_body", e,
          "The source assigns archive maintenance to the current user but gives no personal name. A workspace identifier is not a naming statement.")]
         for e in ("supported", "unsupported")], ["/items/0/candidate_body"],
        faithful_reason="The current user's ongoing archive work is retained without converting workspace metadata into a name.")

    text = "My name is Lena. Tomas is my neighbour. I collect reed instruments."
    body = "The speaker is named Lena, has a neighbour Tomas, and collects reed instruments."
    a = payload([source("unit-9207", 9207, text, peer="Tomas")], body,
        title="Reed instrument interest", body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_body"] = "The speaker is named Tomas, has a neighbour Lena, and collects reed instruments."
    pair("separate-people", a, b,
        [[ground("mechanism", "identity", "/candidate_body", e,
          "Canonical text assigns Lena to the speaker and Tomas to the neighbour. Mentioning both names, or a matching peer handle, cannot exchange their identity roles.")]
         for e in ("supported", "unsupported")], ["/items/0/candidate_body"],
        faithful_reason="The candidate retains both distinct people and the speaker's collection interest with no invented identity relation.")

    text = "The complete set of monitored terminals is Coral, Reed and Moss. Coral and Reed are authorized; Moss is not authorized. Other terminals have not been assessed."
    body = "Coral and Reed are authorized and Moss is not; these are exactly the monitored terminals. Other terminals were not assessed."
    a = payload([source("unit-9208", 9208, text)], body,
        title="Among monitored terminals, only Moss is unauthorized", body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_title"] = "Across all terminals, only Moss is unauthorized"
    pair("negative-predicate-domain", a, b,
        [[ground("mechanism", "quantified_scope", "/candidate_title", e,
          "The explicitly closed monitored set has exactly one unauthorized member, Moss. No result is established for unassessed terminals.")]
         for e in ("supported", "unsupported")], ["/items/0/candidate_title"],
        faithful_reason="The title's negative-predicate exclusivity is bounded to the complete monitored set and the body retains statuses and the unassessed-domain caveat.")

    text = "The catalogue contains exactly three licensed routes: Aspen, Birch and Cedar. Aspen and Birch have night service; Cedar does not. Other routes have not been checked."
    body = "Within the licensed routes Aspen, Birch and Cedar, Aspen and Birch have night service and Cedar does not. Other routes have not been checked."
    a = payload([source("unit-9209", 9209, text)], body,
        title="Two of these routes have night service", body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_title"] = "Exactly two routes anywhere have night service"
    pair("anaphoric-cardinality", a, b,
        [[ground("mechanism", "quantified_scope", "/candidate_title", e,
          "These routes unambiguously refers to the three licensed routes named in the full candidate. Their two positive and one negative statuses establish two within that set, not exactly two anywhere.")]
         for e in ("supported", "unsupported")], ["/items/0/candidate_title"],
        faithful_reason="The title uses the body's explicit licensed domain, not a global domain, while the body preserves the full enumeration and outside-domain uncertainty.")

    text = "Exactly four capsules exist in the entire shipment. All four passed inspection. There are no other capsules in the shipment."
    body = "The entire shipment consists of four capsules, and all four passed inspection."
    a = payload([source("unit-9210", 9210, text)], body,
        title="Every capsule in the entire shipment passed inspection", body=body)
    # Unlike the other informational episodes, this is an explicitly completed
    # inspection result; whole-scope acceptance also checks the outcome field.
    a["items"][0]["candidate_outcome"] = "resolved"
    b = deepcopy(a)
    b["items"][0]["candidate_title"] = "Exactly five capsules in the entire shipment passed inspection"
    pair("explicit-universal-support", a, b,
        [[ground("mechanism", "quantified_scope", "/candidate_title", e,
          "Canonical evidence explicitly covers the entire shipment and establishes four successes. A broad claim over that full domain is valid; adding a fifth capsule is not.")]
         for e in ("supported", "unsupported")], ["/items/0/candidate_title"],
        faithful_reason="The full-domain title is explicitly supported, not forbidden merely for being universal; the body preserves the exact count and success, and the resolved outcome matches completed inspection.")

    text = "Valve cleaning: keep the purge vent shut throughout cleaning. Wipe the valve, then inspect its seal."
    proc = procedure("Valve cleaning", ("Wipe the valve", "Inspect its seal"),
                     description="Keep the purge vent shut throughout cleaning.")
    a = payload([source("unit-9211", 9211, text, "assistant")], text, procedure=proc)
    b = deepcopy(a)
    b["procedure_items"][0]["candidate"]["description"] = None
    pair("retained-prohibition", a, b,
        [[retain("unit-9211", "constraints", e,
          "The source's applicable throughout-cleaning closed-vent constraint appears in the faithful description and nowhere in the defective procedure.")]
         for e in ("retained", "omitted")],
        ["/procedure_items/0/candidate/description"], kind="procedure",
        faithful_reason="The procedure preserves the closed vent constraint, both instructed actions, their sequence and their cleaning scope.")

    text = "To service the resonator, first disconnect its power and only then lift its lid. This sequence is mandatory."
    proc = procedure("Resonator service", ("Disconnect its power", "Lift its lid"))
    a = payload([source("unit-9212", 9212, text, "assistant")], text, procedure=proc)
    b = deepcopy(a)
    b["procedure_items"][0]["candidate"]["steps"][0]["action"] = "Lift its lid"
    b["procedure_items"][0]["candidate"]["steps"][1]["action"] = "Disconnect its power"
    pair("mandatory-sequence", a, b,
        [[retain("unit-9212", "ordering", e,
          "The source requires disconnect-before-lift; the defective numbered steps invert this mandatory sequence while retaining both actions.")]
         for e in ("retained", "altered")],
        ["/procedure_items/0/candidate/steps/0/action", "/procedure_items/0/candidate/steps/1/action"],
        kind="procedure", faithful_reason="The two numbered instructions preserve both service actions and their mandatory dependency without adding completion.")

    text = "The meeting agenda was printed in silver ink. The final design decision selected a circular dial for instrument X-74."
    a = payload([source("unit-9213", 9213, text)],
        "The final design decision selected a circular dial for instrument X-74.")
    b = deepcopy(a)
    b["summary_item"]["candidate_summary"] = "The meeting agenda was printed in silver ink."
    b["summary_item"]["candidate_raw_summary"] = "The meeting agenda was printed in silver ink."
    pair("material-decision", a, b,
        [[retain("unit-9213", "material_facts", e,
          "The final dial-selection decision is material to this summary. The incidental ink colour of the meeting agenda alone does not preserve it."),
          ground("remaining-assertion", "assertion", "/candidate_summary", "supported",
          "Both the selected dial decision and the alternative agenda-ink detail are true source assertions; a missing decision is distinct from fabrication.", "auxiliary")]
         for e in ("retained", "omitted")],
        ["/summary_item/candidate_summary", "/summary_item/candidate_raw_summary"], kind="summary",
        faithful_reason="The summary retains the material final design choice and the correct instrument; the meeting agenda's ink colour is incidental rather than an omitted design specification, with no imposed chronology or omitted constraint.")
    return cases
