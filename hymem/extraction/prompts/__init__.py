from __future__ import annotations

# Prompts are kept in code so prompt_version pins them. Bump HyMemConfig.prompt_version
# whenever you change wording so already-processed chunks get reprocessed cleanly.

ALLOWED_PREDICATES = (
    "uses",
    "depends_on",
    "prefers",
    "rejects",
    "avoids",
    "replaces",
    "conflicts_with",
    "deploys_to",
    "part_of",
    "equivalent_to",
    "implements",
    "contains",
    "configured_with",
    "requires_version",
    "runs_on",
    "connects_to",
    "generates",
    "tested_by",
    # Personal-life value facts (added v9): the KU-coverage gap was that these
    # classes — possessions, residence, activities/habits, personal metrics —
    # had no expressible predicate, so the value never reached the graph.
    "owns",
    "located_in",
    "participates_in",
    "has_attribute",
)

_TRIPLE_SYSTEM_TEMPLATE = """You extract structured technical relationships and personal-life facts from conversation excerpts.

Rules:
- Output a strict JSON array. No prose, no markdown, no code fences.
- Each item has exactly: subject (string), predicate (string), object (string), polarity (1 or -1).
- Optional fields (include only when applicable):
    value_text (string): numeric value, version string, or quantity mentioned
    value_numeric (number): parsed numeric value if available
    value_unit (string): unit for numeric values ("seconds", "MB", "rps")
    temporal_scope (string): time context ("since 2024", "during migration", "temporarily")
- Optionally include subject_type and object_type (string) to classify entities:
    language, framework, database, service, tool, library, file, environment, protocol, container, package_manager, api, platform, config_file, testing_framework, ci_tool, monitoring_tool, identity_provider, message_broker, person, team, project, codebase, place, organization, product, vehicle, activity, event, document, or_other_entity
- Include these types ONLY when you are confident. Skip them otherwise.
- Optionally include subject_properties and/or object_properties (object of
  string->string pairs) to capture stable attributes of the entity, e.g.
  {{"language": "python", "category": "build_tool", "runtime": "node"}}.
  Keep keys short and lowercase (language, runtime, category, vendor, role).
  At most 4 properties per entity; skip when not clearly stated.
- predicate MUST be one of: {predicates}.
- Predicate meanings:
    uses: A employs or utilizes B
    depends_on: A requires B to function
    prefers: A favors B over alternatives
    rejects: A explicitly refuses or negates B
    avoids: A steers clear of B
    replaces: A supersedes or substitutes B
    conflicts_with: A is incompatible with B
    deploys_to: A is deployed or released to B
    part_of: A is a component or sub-part of B
    equivalent_to: A is synonymous or interchangeable with B
    implements: A realizes or fulfills interface/contract/spec B
    contains: A holds, owns, or includes B as a subcomponent
    configured_with: A is parameterized or set up using B
    requires_version: A needs a specific version of B
    runs_on: A executes or operates on platform/runtime B
    connects_to: A has a network or data-flow connection to B
    generates: A produces, outputs, or creates B
    tested_by: A is tested or verified using B
    owns: A owns or possesses B (a vehicle, home, device, or other belonging)
    located_in: A lives in, is located in, or is based in place B
    participates_in: A does, plays, practices, attends, or is enrolled in activity/event B (put frequency or schedule in temporal_scope)
    has_attribute: A has the personal attribute or measurement B (age, height, weight, salary, a rate); state the value as the object (e.g. 60_bpm) and also in value_numeric/value_unit when numeric
{negative_examples}
- polarity is -1 only when the speaker negates or retracts the relationship
  ("we don't use X anymore", "we stopped using X", "we replaced X with Y").
  Mapping for negations: "no longer uses" -> uses with polarity -1.
  Statements like "we avoid X" use predicate 'avoids' with polarity 1, NOT 'uses' with -1.
- Skip relationships you are not confident about. An empty array [] is a valid answer.
- Subject and object should be concrete named things — tools, libraries, services,
  files, modules, environments, people, teams, projects, or codebases by name, AND
  the user's personal-life things: possessions, places they live, activities they
  do, events they attend, and personal attributes.
  Do not invent abstractions like "the system".
- When a chunk names a person or team alongside a project, codebase, or artifact
  they own, work on, or belong to, extract the linking edge explicitly. This is
  high-priority: identity-to-artifact links are the most underrepresented and
  most useful triples in the graph. Strong examples (extract eagerly when the
  chunk supports them):
    "Atta is working on MedFlow"                   -> (atta, part_of, medflow)
    "I'm building HyMem"                            -> (atta, part_of, hymem)
    "We use HyMem for the memory layer"             -> (atta, uses, hymem)
    "The platform team owns the auth service"       -> (platform_team, contains, auth_service)
    "Sara maintains the ingest pipeline"            -> (sara, part_of, ingest_pipeline)
  When the speaker is the user themselves ("I'm working on X", "we shipped Y"),
  resolve the implicit subject to the user's canonical name when known from
  context; otherwise use a first-person handle and let canonicalization resolve
  it. Do NOT skip these just because the speaker is implicit.
  This makes identity-to-artifact relationships queryable as 1-hop graph edges
  rather than fuzzy text matches across sibling canonicals.
- Personal-life facts about the user are EQUALLY high-priority — possessions,
  where they live, activities and habits, and personal attributes. Extract them
  with the same eagerness as technical facts, resolving an implicit first-person
  subject ("I", "we", "my") to the user's canonical name when known:
    "I drive a Ford F-150"             -> (user, owns, ford_f_150)
    "We just moved to Austin"           -> (user, located_in, austin)
    "I play tennis every Tuesday"       -> (user, participates_in, tennis)   [temporal_scope: "every Tuesday"]
    "My resting heart rate is 60 bpm"   -> (user, has_attribute, 60_bpm)     [value_numeric: 60, value_unit: "bpm"]
  Updates to these (a new car, a move, a changed metric) are exactly the
  knowledge-update facts the graph must capture — never skip them as "not technical".
- Excerpts may be written in languages other than English (e.g. Dutch, German,
  French, Spanish). Extract relationships regardless; keep subject and object in
  the original language as they appear in the text.
"""


def build_triple_system(negative_examples: str = "") -> str:
    """Build the triple extraction system prompt with optional negative examples."""
    neg_section = ""
    if negative_examples:
        neg_section = (
            "\nCRITICAL: The following triples were previously extracted INCORRECTLY "
            "from similar conversation contexts. DO NOT extract these exact "
            "relationships or close variants:\n"
            + negative_examples
            + "\n"
        )
    return _TRIPLE_SYSTEM_TEMPLATE.format(
        predicates=", ".join(ALLOWED_PREDICATES),
        negative_examples=neg_section,
    )


TRIPLE_SYSTEM = build_triple_system()


TRIPLE_USER_TEMPLATE = """Excerpt:
\"\"\"
{text}
\"\"\"

Return the JSON array now."""


MARKER_SYSTEM = """You identify EXPLICIT behavioral signals from a user in a conversation excerpt.

Only include signals that are stated outright. Do NOT infer mood or sentiment.
Excerpts may be in languages other than English; identify signals regardless.

Allowed kinds:
- correction: user told the assistant it was wrong about something specific.
- preference: user explicitly stated they like / want / use approach X.
- rejection: user explicitly stated they dislike / refuse / will not use X.
- style: user explicitly asked for a way of communicating (verbosity, format, tone).

Output a strict JSON array. Each item: {"kind": "...", "statement": "..."}.
'statement' is a single short factual sentence, not a quote.
Empty array [] is valid."""


MARKER_USER_TEMPLATE = """Excerpt:
\"\"\"
{text}
\"\"\"

Return the JSON array now."""


# --- Combined per-chunk extraction (triples + markers in one call) ----------
# Single prompt that returns a JSON OBJECT with both "triples" and "markers",
# halving the per-chunk LLM call count. The rules below are the verbatim triple
# and marker rules from the separate prompts above so quality does not regress;
# only the output container changes (object with two array keys, mirroring
# SESSION_DIGEST_SYSTEM). The distinctive substrings "structured technical
# relationships" and "EXPLICIT behavioral signals" are preserved for prompt
# routing in tests.

_CHUNK_EXTRACTION_SYSTEM_TEMPLATE = """You extract structured technical relationships, personal-life facts, and EXPLICIT behavioral signals from a conversation excerpt in a single pass.

Output a strict JSON OBJECT (not an array). No prose, no markdown, no code fences.
The object has exactly these two keys:

"triples": a JSON array of relationship items. Each item has exactly: subject (string), predicate (string), object (string), polarity (1 or -1).
- Optional fields (include only when applicable):
    value_text (string): numeric value, version string, or quantity mentioned
    value_numeric (number): parsed numeric value if available
    value_unit (string): unit for numeric values ("seconds", "MB", "rps")
    temporal_scope (string): time context ("since 2024", "during migration", "temporarily")
- Optionally include subject_type and object_type (string) to classify entities:
    language, framework, database, service, tool, library, file, environment, protocol, container, package_manager, api, platform, config_file, testing_framework, ci_tool, monitoring_tool, identity_provider, message_broker, person, team, project, codebase, place, organization, product, vehicle, activity, event, document, or_other_entity
- Include these types ONLY when you are confident. Skip them otherwise.
- Optionally include subject_properties and/or object_properties (object of
  string->string pairs) to capture stable attributes of the entity, e.g.
  {{"language": "python", "category": "build_tool", "runtime": "node"}}.
  Keep keys short and lowercase (language, runtime, category, vendor, role).
  At most 4 properties per entity; skip when not clearly stated.
- predicate MUST be one of: {predicates}.
- Predicate meanings:
    uses: A employs or utilizes B
    depends_on: A requires B to function
    prefers: A favors B over alternatives
    rejects: A explicitly refuses or negates B
    avoids: A steers clear of B
    replaces: A supersedes or substitutes B
    conflicts_with: A is incompatible with B
    deploys_to: A is deployed or released to B
    part_of: A is a component or sub-part of B
    equivalent_to: A is synonymous or interchangeable with B
    implements: A realizes or fulfills interface/contract/spec B
    contains: A holds, owns, or includes B as a subcomponent
    configured_with: A is parameterized or set up using B
    requires_version: A needs a specific version of B
    runs_on: A executes or operates on platform/runtime B
    connects_to: A has a network or data-flow connection to B
    generates: A produces, outputs, or creates B
    tested_by: A is tested or verified using B
    owns: A owns or possesses B (a vehicle, home, device, or other belonging)
    located_in: A lives in, is located in, or is based in place B
    participates_in: A does, plays, practices, attends, or is enrolled in activity/event B (put frequency or schedule in temporal_scope)
    has_attribute: A has the personal attribute or measurement B (age, height, weight, salary, a rate); state the value as the object (e.g. 60_bpm) and also in value_numeric/value_unit when numeric
{negative_examples}
- polarity is -1 only when the speaker negates or retracts the relationship
  ("we don't use X anymore", "we stopped using X", "we replaced X with Y").
  Mapping for negations: "no longer uses" -> uses with polarity -1.
  Statements like "we avoid X" use predicate 'avoids' with polarity 1, NOT 'uses' with -1.
- Skip relationships you are not confident about. An empty array [] is a valid answer.
- Subject and object should be concrete named things — tools, libraries, services,
  files, modules, environments, people, teams, projects, or codebases by name, AND
  the user's personal-life things: possessions, places they live, activities they
  do, events they attend, and personal attributes.
  Do not invent abstractions like "the system".
- When a chunk names a person or team alongside a project, codebase, or artifact
  they own, work on, or belong to, extract the linking edge explicitly. This is
  high-priority: identity-to-artifact links are the most underrepresented and
  most useful triples in the graph. Strong examples (extract eagerly when the
  chunk supports them):
    "Atta is working on MedFlow"                   -> (atta, part_of, medflow)
    "I'm building HyMem"                            -> (atta, part_of, hymem)
    "We use HyMem for the memory layer"             -> (atta, uses, hymem)
    "The platform team owns the auth service"       -> (platform_team, contains, auth_service)
    "Sara maintains the ingest pipeline"            -> (sara, part_of, ingest_pipeline)
  When the speaker is the user themselves ("I'm working on X", "we shipped Y"),
  resolve the implicit subject to the user's canonical name when known from
  context; otherwise use a first-person handle and let canonicalization resolve
  it. Do NOT skip these just because the speaker is implicit.
  This makes identity-to-artifact relationships queryable as 1-hop graph edges
  rather than fuzzy text matches across sibling canonicals.
- Personal-life facts about the user are EQUALLY high-priority — possessions,
  where they live, activities and habits, and personal attributes. Extract them
  with the same eagerness as technical facts, resolving an implicit first-person
  subject ("I", "we", "my") to the user's canonical name when known:
    "I drive a Ford F-150"             -> (user, owns, ford_f_150)
    "We just moved to Austin"           -> (user, located_in, austin)
    "I play tennis every Tuesday"       -> (user, participates_in, tennis)   [temporal_scope: "every Tuesday"]
    "My resting heart rate is 60 bpm"   -> (user, has_attribute, 60_bpm)     [value_numeric: 60, value_unit: "bpm"]
  Updates to these (a new car, a move, a changed metric) are exactly the
  knowledge-update facts the graph must capture — never skip them as "not technical".
- Excerpts may be written in languages other than English (e.g. Dutch, German,
  French, Spanish). Extract relationships regardless; keep subject and object in
  the original language as they appear in the text.

"markers": a JSON array of EXPLICIT behavioral signals from the user. Each item: {{"kind": "...", "statement": "..."}}.
- Only include signals that are stated outright. Do NOT infer mood or sentiment.
- Excerpts may be in languages other than English; identify signals regardless.
- Allowed kinds:
    correction: user told the assistant it was wrong about something specific.
    preference: user explicitly stated they like / want / use approach X.
    rejection: user explicitly stated they dislike / refuse / will not use X.
    style: user explicitly asked for a way of communicating (verbosity, format, tone).
- 'statement' is a single short factual sentence, not a quote.

Always return both keys. An empty array [] is valid for either. Example shape:
{{"triples": [], "markers": []}}
"""


def build_chunk_extraction_system(negative_examples: str = "") -> str:
    """Build the combined triples+markers extraction system prompt with optional
    negative examples. Mirrors ``build_triple_system``."""
    neg_section = ""
    if negative_examples:
        neg_section = (
            "\nCRITICAL: The following triples were previously extracted INCORRECTLY "
            "from similar conversation contexts. DO NOT extract these exact "
            "relationships or close variants:\n"
            + negative_examples
            + "\n"
        )
    return _CHUNK_EXTRACTION_SYSTEM_TEMPLATE.format(
        predicates=", ".join(ALLOWED_PREDICATES),
        negative_examples=neg_section,
    )


CHUNK_EXTRACTION_SYSTEM = build_chunk_extraction_system()


CHUNK_EXTRACTION_USER_TEMPLATE = """Excerpt:
\"\"\"
{text}
\"\"\"

Return the JSON object with "triples" and "markers" now."""


EPISODE_SYSTEM = """You identify distinct episodes within a conversation session.

An episode is a coherent segment focused on one topic, problem, or task. A session may have multiple episodes.

Each chunk in the input is tagged like `[chunk chk_abc123]` — the chunk id appears in square brackets before its text. When you group chunks into an episode you MUST return the exact chunk ids you used; downstream code reads them to look up the message range.

Output a strict JSON array. Each item:
- title (string): Short descriptive name, max 8 words
- summary (string): 1-2 sentence narrative of what happened
- outcome (string|null): "resolved", "blocked", "deferred", "informational", or null if unclear
- key_entities (list of strings): Named tools, services, files, or concepts discussed
- chunk_ids (list of strings): The `chk_...` ids you grouped, in conversation order. Must be non-empty and contain only ids that appear in the input.

Empty array [] is valid if the conversation has no clear episodes.
"""

EPISODE_USER_TEMPLATE = """Conversation session:
\"\"\"
{text}
\"\"\"

Return the JSON array now."""


AGGREGATE_SYSTEM = """You fuse several related episodes — drawn from DIFFERENT conversation sessions but about the same ongoing thread, person, project, or topic — into one cross-session summary.

The point is synthesis: a later question may need facts that are scattered one-per-session ("which of my projects use Postgres?", "how did my budget change over the year?"). Pull the through-line together so it can be answered from this one node instead of re-reading every session.

Output a strict JSON object:
- title (string): Short name for the shared thread, max 8 words
- summary (string): 2-4 sentences fusing what these episodes collectively establish. Preserve specific named entities, numbers, dates, and changes-over-time; prefer concrete facts over generic narration. Do NOT invent anything not present in the episodes. Never state a name, employer, role, or location unless an episode literally states it — omit absent identity details, do not fill them in. Do NOT add "The user"/"The assistant" — use passive voice or implicit subject.

Return ONLY the JSON object."""

AGGREGATE_USER_TEMPLATE = """Related episodes from across sessions:
\"\"\"
{text}
\"\"\"

Return the JSON object now."""


ROLLUP_SYSTEM = """You merge several summaries of conversation threads — which may be related or COMPLETELY UNRELATED — into one combined summary that loses no thread.

This is an intermediate node of a summary tree over a user's whole conversation history; your output will be merged again at the next level, so BREADTH beats narrative. The one failure mode to avoid: picking the dominant topic and letting the others fade — a thread dropped here is gone from every level above. Do NOT force a single story line over unrelated topics.

Output a strict JSON object:
- title (string): Short label naming the main topics (not just one), max 10 words
- summary (string): 3-8 sentences. Every distinct thread in the inputs must be mentioned at least once; give recurring threads more words, but never zero. Preserve specific named entities, numbers, and dates. Do NOT invent anything not present in the inputs. Never state a name, employer, role, or location unless an input literally states it — omit absent identity details, do not fill them in.

Return ONLY the JSON object."""

ROLLUP_USER_TEMPLATE = """Thread summaries to merge (possibly unrelated):
\"\"\"
{text}
\"\"\"

Return the JSON object now."""


DIGEST_SYSTEM = """You write the standing digest of everything known about a user from their conversation history — the top of a summary tree whose inputs below are themselves summaries of conversation threads.

This digest answers "what do you know about me?" at a glance and is injected as standing context for an assistant, so it must read like a rounded profile, not a recap of the latest topic: the main recurring activities and projects with their state, preferences and habits, recurring people and places, and notable changes over time. Favor durable facts over one-off details.

Identity is strictly evidence-bound: include the user's name, role, employer, or location ONLY when the VERIFIED FACTS block or a summary literally states them. A profile with no job title is correct; a plausible-sounding invented one ("works at Acme Corp") is the worst possible failure — when an identity detail is absent from the inputs, OMIT it entirely.

The VERIFIED FACTS block holds statements extracted directly from the user's conversations into a knowledge graph. Treat it as ground truth: when a thread summary conflicts with a verified fact, the fact wins and the summary's claim is dropped. Thread summaries are themselves machine-generated and may contain compression errors — be suspicious of identity or preference claims that appear in only one summary and in no verified fact.

Output a strict JSON object:
- title (string): Short label for the digest, max 8 words
- summary (string): 6-12 sentences. Cover EVERY distinct thread in the inputs at least briefly — breadth first, then depth on the threads that recur most. Preserve specific named entities, numbers, and dates; prefer concrete facts over generic narration. Unrelated threads get their own sentence rather than a forced story. Do NOT invent anything not present in the inputs.

Return ONLY the JSON object."""

DIGEST_USER_TEMPLATE = """VERIFIED FACTS (knowledge graph — ground truth, trust over the summaries):
\"\"\"
{facts}
\"\"\"

Thread summaries to digest:
\"\"\"
{text}
\"\"\"

Return the JSON object now."""


SESSION_SUMMARY_SYSTEM = """You write a one-sentence summary of a conversation session.

Focus on: what was accomplished, decisions made, problems solved, topics covered.
Be specific about tools, technologies, and concrete outcomes mentioned.
Do NOT add "The user" or "The assistant" — use passive voice or implicit subject.
Output ONLY the summary text, no JSON, no markdown, no quotes.
"""

SESSION_SUMMARY_USER_TEMPLATE = """Conversation:
\"\"\"
{text}
\"\"\"

One-sentence summary:"""


PROCEDURE_SYSTEM = """You identify step-by-step procedures described in a conversation.

A procedure is an ordered sequence of actions needed to accomplish a specific technical task — like deploying, configuring, debugging, setting up, or testing something.

Output a strict JSON array. Each item:
- name (string): Short descriptive imperative name, max 8 words. e.g., "Deploy to staging", "Set up local dev", "Debug Postgres connection pool"
- description (string): 1 sentence describing what the procedure accomplishes
- steps (list of objects): Ordered steps, each with:
    order (integer): Step number starting at 1
    action (string): What to do, imperative form
    tool (string or null): Tool/command/CLI used, if mentioned explicitly
- triggers (list of strings): Words/phrases someone might use to ask about this procedure. e.g., ["deploy", "ship it", "release", "push to staging"]
- entities_involved (list of strings): Named tools, services, platforms, files involved

Only extract procedures that are EXPLICITLY described. Do not invent procedures from general discussion. Empty array [] is valid.
"""

PROCEDURE_USER_TEMPLATE = """Conversation:
\"\"\"
{text}
\"\"\"

Return the JSON array now."""


SESSION_DIGEST_SYSTEM = """You analyze one conversation session and produce three things in a single pass: its episodes, a one-sentence summary, and any step-by-step procedures.

Each chunk in the input is tagged like `[chunk chk_abc123]` — the chunk id appears in square brackets before its text.

Output a strict JSON OBJECT (not an array) with exactly these three keys:

"episodes": a JSON array of episodes. An episode is a coherent segment focused on one topic, problem, or task; a session may have several. Each item:
- title (string): Short descriptive name, max 8 words
- summary (string): 1-2 sentence narrative of what happened
- outcome (string|null): "resolved", "blocked", "deferred", "informational", or null if unclear
- key_entities (list of strings): Named tools, services, files, or concepts discussed
- chunk_ids (list of strings): The `chk_...` ids you grouped, in conversation order. Must be non-empty and contain only ids that appear in the input.
Empty array [] is valid if there are no clear episodes.

"summary": a single string — one sentence covering what was accomplished, decisions made, problems solved, topics covered. Be specific about tools, technologies, and concrete outcomes. Do NOT add "The user" or "The assistant"; use passive voice or implicit subject. No markdown, no quotes. Empty string "" is valid if there is nothing to summarize.

"procedures": a JSON array of procedures. A procedure is an ordered sequence of actions needed to accomplish a specific technical task — deploying, configuring, debugging, setting up, or testing something. Each item:
- name (string): Short descriptive imperative name, max 8 words. e.g., "Deploy to staging"
- description (string): 1 sentence describing what the procedure accomplishes
- steps (list of objects): Ordered steps, each with:
    order (integer): Step number starting at 1
    action (string): What to do, imperative form
    tool (string or null): Tool/command/CLI used, if mentioned explicitly
- triggers (list of strings): Words/phrases someone might use to ask about this procedure
- entities_involved (list of strings): Named tools, services, platforms, files involved
Only extract procedures that are EXPLICITLY described; do not invent them. Empty array [] is valid.

Always return all three keys. Example shape:
{"episodes": [], "summary": "", "procedures": []}
"""

SESSION_DIGEST_USER_TEMPLATE = """Conversation session:
\"\"\"
{text}
\"\"\"

Return the JSON object now."""


# --- Typed user-profile extraction (Stage 1 / P4, schema v18) ---------------
# Runs over USER turns only, once per dreamed session, alongside the batched
# session digest. The slot vocabulary is CLOSED and re-enforced downstream
# twice (validate_profile_items + the user_profile table CHECK), so a slot the
# LLM invents can never persist. Pinned by PROFILE_PROMPT_VERSION in
# hymem/dreaming/user_profile.py — bump it when this wording changes
# materially. The user-template closer is deliberately unique ("Return the
# profile JSON array now.") so test stubs keyed on the digest/triple closers
# never route here; "typed user-profile facts" is the routing substring.

USER_PROFILE_SYSTEM = """You extract typed user-profile facts from the USER's own turns in one conversation session.

Each user turn is tagged like `[msg 42]` — the message id appears in square brackets before its text.

THE ABOUTNESS TEST (apply it to every candidate fact): extract only durable facts about THE USER AS A PERSON. User turns often contain pasted material — memory files, repository lists, agent or config dumps, notes about patients, clients, or other third parties. Facts about projects, repositories, codebases, organizations-as-artifacts, or other people are NOT user facts and must be skipped. But pasted text that explicitly describes the user themselves IS extractable — the test is WHO the fact is about, not what channel it arrived through.
- "Atta is a bedrijfsarts at O3" inside a pasted memory file → extract role "bedrijfsarts" and employer "O3" (the text is explicitly about the user).
- "Patient reports chronic back pain since 2024" in the user's clinical notes → extract NOTHING (the fact is about a patient, not the user).
- "Repos: ClaudetteMedSer/HyMem, ClaudetteMedSer/Hermes" in a pasted dump → extract NOTHING (a GitHub org/username in technical context is not an employer, and repositories are not possessions).

Output a strict JSON array. Each item has exactly:
- slot (string): MUST be one of:
    role: the user's job or professional role ("bedrijfsarts", "backend engineer")
    name: the user's own name
    employer: the organization the user works FOR. Employment must be explicitly stated; GitHub orgs, usernames, and account names appearing in technical context are NOT employers.
    location: where the user lives or is based
    language: a language the user speaks or writes
    relationship: another person in the user's life — requires slot_key
    possession: a notable durable real-world possession the user owns (a car, a house, an instrument). NEVER code repositories, software artifacts, accounts, or items enumerated in pasted lists.
    age_birthday: the user's age or birthday
    health_condition: ONLY the user's OWN health — a condition, allergy, or treatment stated first-person or unambiguously about the user. The user may be a clinician (e.g. a bedrijfsarts): conditions of patients or clients they discuss are NEVER user health facts. Job or work descriptions belong in role, never here.
    recurring_activity: a habit or recurring activity of the user ("runs every sunday")
- slot_key (string): ONLY for relationship — the other person's name ("anna"). Omit for every other slot.
- value (string): the fact itself, short and concrete ("bedrijfsarts", "Amsterdam", "sister").
- evidence_message_id (integer): the exact `[msg N]` id of the user turn stating the fact. MUST be an id present in the input.
- confidence (number between 0.0 and 1.0): how explicitly the fact is stated (1.0 = stated outright; lower when hedged or implied).

Rules:
- The slot list above is CLOSED. Never output any other slot name; skip facts that fit no slot.
- Only durable facts. Skip questions, hypotheticals, one-off events, and facts about other people (except via relationship).
- Quality over quantity: prefer a few high-confidence facts over many weak ones. When two candidate values compete for the same slot, keep only the more specific and complete one (e.g. "bedrijfsarts" over "developer" for role). Do not emit near-duplicate recurring_activity items. When in doubt, omit.
- Turns may be in languages other than English (e.g. Dutch); extract regardless and keep the value in the original language.
- Skip facts you are not confident about. An empty array [] is a valid answer.
"""

USER_PROFILE_USER_TEMPLATE = """User turns from one session:
\"\"\"
{text}
\"\"\"

Return the profile JSON array now."""


# Narrative-facts extraction (E1, schema v26). This text is the VERBATIM
# `FACTS_PROMPT_V2` that cleared the G-F1b extraction-faithfulness gate
# (2026-08-02, 123/123 strict on the healed full-source sample) — moving it
# unchanged is what carries the gate's verdict over to production, so any
# rewording, however small, re-enters the gate: bump FACTS_PROMPT_VERSION in
# hymem/dreaming/facts.py, flip facts_extraction_enabled off, re-clear
# benchmarks/fact_probe.py at ≥0.90, then re-enable. The draft arms and the
# v1→v2 defect analysis stay in fact_probe.py (the instrument), not here.
# The user-template closer ("Return the JSON array of narrative facts now.")
# is unique so test stubs keyed on the digest/triple/profile closers never
# route here.

FACTS_SYSTEM = """You extract NARRATIVE FACTS from one conversation session.

A narrative fact is a single, self-contained statement of something that happened, was decided, was preferred, or was true — written so it can be read and understood WITHOUT the conversation around it.

THE ONLY RULE THAT MATTERS: every name, number, date, price, quantity and claim you write must ALREADY BE PRESENT in the turns below. If it is not there, it does not go in. Do not infer it, do not complete it, do not make it plausible. A short, dull, literal fact is correct; a rich fact containing one invented detail is a failure.

Output a strict JSON array. Each item has exactly:
- text (string): the fact, one sentence, self-contained. Name the people, things and values explicitly instead of using "he", "it", "that", "the project" — but only names that appear in the turns.
- date (string or null): use YYYY-MM-DD ONLY when the conversation itself writes that date. If the turns say "recently", "last week", "a few days ago", or say nothing about when — use null. Never derive a date from the session date. Never guess.
- entities (array of strings): the concrete people, products, places, tools or organizations the fact is about, exactly as written in the turns.

Rules:
- VERBATIM VALUES. Names, numbers, dates, versions, prices and quantities exactly as the turns state them. Never round, convert, or normalize.
- NO QUOTA. Extract only what the turns actually establish. A session that establishes one thing yields one fact. A session of small talk, greetings, or generic assistant advice yields [] — that is a GOOD answer, not a failure, and [] is always available to you.
- Never extract the assistant's suggestions, recommendations, or hypotheticals as facts about the user. "You could try X" is not "the user uses X".
- One fact per exchange, decision, event or outcome — not one per turn. Combine a question and its answer into the single fact they establish.
- Self-contained means resolvable alone: "Atta moved the MedFlow deploy to fly.io" — not "he moved it there".
- Keep the fact in the language of the conversation.

Before writing each fact, check: can I point at the exact words in the turns that state every value in it? If not, drop it.
"""

# Deliberately does NOT show the session date (the v1→v2 lesson: the surest way
# to stop a date being copied is for the model never to see it). Explicit dates
# written in the turns are still right there in {text}.
FACTS_USER_TEMPLATE = """Conversation:
\"\"\"
{text}
\"\"\"

Return the JSON array of narrative facts now."""


RERANK_SYSTEM = """You evaluate the relevance of conversation excerpts to a user query.

For each excerpt, rate its relevance on a scale of 1-5:
5 - Directly answers or discusses the query's topic
4 - Highly relevant, close to the topic
3 - Somewhat relevant, tangentially related
2 - Marginally relevant, shares keywords but different topic
1 - Not relevant

Output strict JSON: an object with one "ratings" key holding an array.
Each array item: {"index": 0, "relevance": 4}
The index field corresponds to the [0], [1], [2] markers in the input.
{"ratings": []} is valid if nothing is relevant.
"""

RERANK_USER_TEMPLATE = """Query: "{query}"

Excerpts:
{excerpts}

Return the JSON object now."""
