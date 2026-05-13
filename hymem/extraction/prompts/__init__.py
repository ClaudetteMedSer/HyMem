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
)

TRIPLE_SYSTEM = """You extract structured technical relationships from conversation excerpts.

Rules:
- Output a strict JSON array. No prose, no markdown, no code fences.
- Each item has exactly: subject (string), predicate (string), object (string), polarity (1 or -1).
- Optional fields (include only when applicable):
    value_text (string): numeric value, version string, or quantity mentioned
    value_numeric (number): parsed numeric value if available
    value_unit (string): unit for numeric values ("seconds", "MB", "rps")
    temporal_scope (string): time context ("since 2024", "during migration", "temporarily")
- Optionally include subject_type and object_type (string) to classify entities:
    language, framework, database, service, tool, library, file, environment, protocol, container, package_manager, api, platform, config_file, testing_framework, ci_tool, monitoring_tool, identity_provider, message_broker, or_other_tool
- Include these types ONLY when you are confident. Skip them otherwise.
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
- polarity is -1 only when the speaker negates or retracts the relationship
  ("we don't use X anymore", "we stopped using X", "we replaced X with Y").
  Mapping for negations: "no longer uses" -> uses with polarity -1.
  Statements like "we avoid X" use predicate 'avoids' with polarity 1, NOT 'uses' with -1.
- Skip relationships you are not confident about. An empty array [] is a valid answer.
- Subject and object should be concrete named things (tools, libraries, services,
  files, modules, environments). Do not invent abstractions like "the system".
""".format(predicates=", ".join(ALLOWED_PREDICATES))


TRIPLE_USER_TEMPLATE = """Excerpt:
\"\"\"
{text}
\"\"\"

Return the JSON array now."""


MARKER_SYSTEM = """You identify EXPLICIT behavioral signals from a user in a conversation excerpt.

Only include signals that are stated outright. Do NOT infer mood or sentiment.

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


EPISODE_SYSTEM = """You identify distinct episodes within a conversation session.

An episode is a coherent segment focused on one topic, problem, or task. A session may have multiple episodes.

Output a strict JSON array. Each item:
- title (string): Short descriptive name, max 8 words
- summary (string): 1-2 sentence narrative of what happened
- outcome (string|null): "resolved", "blocked", "deferred", "informational", or null if unclear
- key_entities (list of strings): Named tools, services, files, or concepts discussed

Empty array [] is valid if the conversation has no clear episodes.
"""

EPISODE_USER_TEMPLATE = """Conversation session:
\"\"\"
{text}
\"\"\"

Return the JSON array now."""


RERANK_SYSTEM = """You evaluate the relevance of conversation excerpts to a user query.

For each excerpt, rate its relevance on a scale of 1-5:
5 - Directly answers or discusses the query's topic
4 - Highly relevant, close to the topic
3 - Somewhat relevant, tangentially related
2 - Marginally relevant, shares keywords but different topic
1 - Not relevant

Output a strict JSON array. Each item: {"index": 0, "relevance": 4}
The index field corresponds to the [0], [1], [2] markers in the input.
Empty array [] is valid if nothing is relevant.
"""

RERANK_USER_TEMPLATE = """Query: "{query}"

Excerpts:
{excerpts}

Return the JSON array now."""
