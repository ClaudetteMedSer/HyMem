from __future__ import annotations

import re
import sqlite3

from hymem.dreaming.canonicalize import normalize

# First char must be a Unicode letter (so accented Latin words like "préfère"
# tokenize whole instead of being shredded at the accent); body allows letters,
# digits, underscore, hyphen, dot. normalize() then folds it consistently with
# how the entity was stored.
_TOKEN = re.compile(r"[^\W\d_][\w\-.]{1,40}")


def match_known_entities(conn: sqlite3.Connection, message: str) -> list[str]:
    """Return canonical ids that the user message references.

    Strategy: tokenize the message, normalize each token, and look it up against
    the alias table and the graph's existing canonical names. Cheap, deterministic,
    and the graph is its own dictionary — no LLM call needed at query time.
    """
    raw_tokens = {m.group(0) for m in _TOKEN.finditer(message)}
    candidates = {normalize(t) for t in raw_tokens if len(t) >= 2}

    # Also try multi-word phrases (up to 3-grams) to catch "local dev environment".
    words = [w for w in re.split(r"\s+", message.strip()) if w]
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            candidates.add(normalize(phrase))

    if not candidates:
        return []

    candidates_list = list(candidates)
    placeholders = ",".join("?" * len(candidates_list))
    # The object-canonical branch applies a shape filter: an object only counts
    # as a known entity if it looks entity-shaped — appears as a subject
    # somewhere, has an entity_types record, or shows up as an object in more
    # than one edge. This suppresses one-off LLM extractions where a gerund or
    # verb form ("working") landed as an object and would otherwise match any
    # query containing that word. The alias and subject branches pass through
    # unfiltered: an explicit alias registration or subject-position usage are
    # already strong entity-shape signals.
    rows = conn.execute(
        f"""
        SELECT DISTINCT canonical FROM entity_aliases WHERE alias IN ({placeholders})
        UNION
        SELECT DISTINCT subject_canonical FROM knowledge_graph WHERE subject_canonical IN ({placeholders})
        UNION
        SELECT DISTINCT object_canonical FROM knowledge_graph kg
        WHERE object_canonical IN ({placeholders})
          AND (
            EXISTS (
              SELECT 1 FROM knowledge_graph
              WHERE subject_canonical = kg.object_canonical
            )
            OR EXISTS (
              SELECT 1 FROM entity_types
              WHERE entity_canonical = kg.object_canonical
            )
            OR EXISTS (
              SELECT 1 FROM knowledge_graph kg2
              WHERE kg2.object_canonical = kg.object_canonical AND kg2.id != kg.id
            )
          )
        """,
        candidates_list + candidates_list + candidates_list,
    ).fetchall()
    return [r[0] for r in rows]
