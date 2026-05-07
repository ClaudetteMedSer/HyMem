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
)

TRIPLE_SYSTEM = """You extract structured technical relationships from conversation excerpts.

Rules:
- Output a strict JSON array. No prose, no markdown, no code fences.
- Each item has exactly: subject (string), predicate (string), object (string), polarity (1 or -1).
- predicate MUST be one of: {predicates}.
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


PROFILE_EDIT_SYSTEM = """You maintain a structured behavioral profile for a single user.

You are given the current profile (a JSON array) and a list of new behavioral
markers. Output a JSON object: {"add": [...], "remove": [int ids], "modify": [{"id": int, "text": "..."}]}.

Rules:
- Prefer modify over add when the new marker reinforces an existing entry.
- Use remove only when a new marker directly contradicts an existing entry.
- Do not edit entries that are unrelated to the new markers.
- Each added entry has fields: kind (preference|avoidance|style|context) and text (one short sentence).
- Keep the profile under {max_entries} entries; if the cap would be exceeded,
  drop the oldest contradicted or weakest entries via 'remove'.
- Never invent markers that aren't in the input."""
