"""Fresh invented offline controls, not a frozen paid evaluation or model scores.

Targets are explicit development annotations kept outside every model payload.
The previous source-review fixtures, labels and paid replies are untouched.
Faithful/defective describes the labelled target, not whole-scope acceptance gold.
"""
from copy import deepcopy

from tests.digest_source_review_evaluation_fixtures import context, payload, source


def build_record_controls():
    cases = []

    def pair(name, kind, field, faithful, defective, why):
        for variant, value, expected in (("faithful", faithful, "supported"),
                                         ("defective", defective, "unsupported")):
            cases.append({"id":f"record-{name}-{variant}","pair":name,"variant":variant,
                "payload":deepcopy(value),"scope":{"kind":"episode","index":0},
                "target":{"kind":kind,"field_path":field},"expected":expected,
                "rationale":why})

    body = "The assistant attributes the expedition to the user and disclaims it."
    a = payload([source("u-8102",8102,"That expedition is your plan, not mine.","assistant",
        context=context(8101,"I plan an expedition to chart the inlet.","user"))],
        body,title="Inlet expedition attribution",body=body)
    b = deepcopy(a)
    b["source_catalog"][0]["role"]="user"
    b["source_catalog"][0]["interpretation_only_context"]["role"]="assistant"
    pair("cross-message-owner","actor_attribution","/candidate_body",a,b,
        "With current assistant and preceding user, the candidate assigns the actors correctly. Exchanging only their roles makes that identical candidate reverse the actors.")

    text = 'The user said, "I booked the canoe." I made no booking.'
    body = "The assistant reports the user's canoe booking and says it made no booking itself."
    a = payload([source("u-8103",8103,text,"assistant")],body,
        title="Canoe booking report",body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_body"]="The assistant booked the canoe and says the user made no booking."
    pair("quoted-first-person","actor_attribution","/candidate_body",a,b,
        "The quoted first person belongs to the reported user; the surrounding first person belongs to the current assistant. The defective body exchanges their actions.")

    body = "The user intends to restore the clock after winter."
    a = payload([source("u-8104",8104,"clock after winter.","user",
        context=context(8104,"I intend to restore the ","user"))],body,
        title="Clock restoration plan",body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_body"]="The assistant intends to restore the clock after winter."
    pair("same-message-continuation","actor_attribution","/candidate_body",a,b,
        "The same-message prefix completes a first-person plan by the user, not the assistant; its text must remain available as interpretation-only context.")

    body = "The speaker's display name is Vela-23, and they prefer amber ceramics."
    a = payload([source("u-8105",8105,"My display name is Vela-23. I prefer amber ceramics.",
        peer="Vela-23")],body,title="Amber ceramics preference",body=body)
    b = deepcopy(a)
    b["source_catalog"][0]["visible_content"]="I prefer amber ceramics."
    b["source_catalog"][0]["end"]=len(b["source_catalog"][0]["visible_content"])
    pair("canonical-name-versus-peer","identity","/candidate_body",a,b,
        "The faithful canonical text explicitly names the speaker. In the defective source the same string exists only as an opaque peer identifier, which cannot establish a display name.")

    body = "The speaker's display name is Orin, and they prefer linen covers."
    a = payload([source("u-8106",8106,"My display name is Orin. I prefer linen covers.")],
        body,title="Linen cover preference",body=body)
    a["source_catalog"][0]["source_workspace_id"]="Orin"
    b = deepcopy(a)
    b["source_catalog"][0]["visible_content"]="I prefer linen covers."
    b["source_catalog"][0]["end"]=len(b["source_catalog"][0]["visible_content"])
    pair("canonical-name-versus-workspace","identity","/candidate_body",a,b,
        "Only the faithful canonical text establishes the speaker's name. A workspace identifier is neither a personal name nor a speaker identity assertion.")

    text = "My display name is Mira. Ada is my colleague. I prefer woven baskets."
    body = "The speaker's display name is Mira, and they prefer woven baskets."
    a = payload([source("u-8107",8107,text,peer="Ada")],body,
        title="Woven basket preference",body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_body"]="The speaker's display name is Ada, and they prefer woven baskets."
    pair("name-owned-by-another-person","identity","/candidate_body",a,b,
        "Canonical text assigns Mira to the speaker and Ada to a different person. A coincident opaque peer ID cannot transfer the colleague's name to the speaker.")

    text = "The complete list of certified depots is Vale, Dune and Cape. Solace is stocked at Vale and not stocked at Dune or Cape. Other depots have not been surveyed."
    body = "Solace is stocked at Vale and not at Dune or Cape; these are the three certified depots."
    a = payload([source("u-8108",8108,text)],body,
        title="Among certified depots, Solace is stocked exclusively at Vale",body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_title"]="Across all depots, Solace is stocked exclusively at Vale"
    pair("closed-three-member-set","quantified_scope","/candidate_title",a,b,
        "The certified set is explicitly complete and every non-Vale member lacks stock. The source makes no global claim about all depots.")

    text = "We sampled the Argo and Boreal sites. Nectar was present at Argo and absent at Boreal. These sites are not an exhaustive list."
    body = "Nectar was present at Argo and absent at Boreal in the sampled sites."
    a = payload([source("u-8109",8109,text)],body,
        title="Within the sampled Argo and Boreal sites, only Argo had Nectar",body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_title"]="Argo is the only site with Nectar"
    pair("nonexhaustive-sample","quantified_scope","/candidate_title",a,b,
        "The observed two-site subset supports exclusivity only inside that explicitly named subset, not among all sites.")

    text = "Every audited instrument passed calibration. Instruments outside the audit were not assessed."
    body = "All audited instruments passed calibration; instruments outside the audit were not assessed."
    a = payload([source("u-8110",8110,text)],body,
        title="All audited instruments passed calibration",body=body)
    b = deepcopy(a)
    b["items"][0]["candidate_title"]="All instruments passed calibration"
    pair("universal-domain-restriction","quantified_scope","/candidate_title",a,b,
        "Universal quantification is supported over the audited subset only. Removing that domain restriction claims results for unassessed instruments.")
    return cases
