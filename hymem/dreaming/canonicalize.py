from __future__ import annotations

import re
import sqlite3
import unicodedata

# Strip leading articles, trailing parentheticals like "(container)", and
# punctuation. Lowercase. ASCII-fold. Collapse whitespace and underscores.
_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_TRAILING_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(surface: str) -> str:
    """Deterministic surface -> canonical key. Pure function, no DB needed."""
    s = unicodedata.normalize("NFKD", surface).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    s = _TRAILING_PAREN.sub("", s)
    s = _LEADING_ARTICLES.sub("", s)
    s = _NON_ALNUM.sub("_", s).strip("_")
    return s


def resolve(conn: sqlite3.Connection, surface: str) -> str:
    """Return the canonical id for `surface`, consulting the alias table."""
    norm = normalize(surface)
    row = conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias = ?", (norm,)
    ).fetchone()
    return row["canonical"] if row else norm


def register_alias(conn: sqlite3.Connection, surface: str, canonical: str) -> None:
    """Map an additional surface form onto an existing canonical id."""
    conn.execute(
        "INSERT OR REPLACE INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        (normalize(surface), canonical),
    )


def merge(conn: sqlite3.Connection, keep: str, drop: str) -> None:
    """Fold all edges and aliases referencing `drop` into `keep`.

    Caller is responsible for being inside a transaction.
    """
    if keep == drop:
        return

    conn.execute(
        "UPDATE OR IGNORE entity_aliases SET canonical = ? WHERE canonical = ?",
        (keep, drop),
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        (drop, keep),
    )

    # Migrate edges. Conflicts (same subject/predicate/object after rewrite) get
    # their evidence summed into the surviving row, then the duplicate is dropped.
    for column in ("subject_canonical", "object_canonical"):
        rows = conn.execute(
            f"SELECT id FROM knowledge_graph WHERE {column} = ?", (drop,)
        ).fetchall()
        for row in rows:
            edge_id = row["id"]
            edge = conn.execute(
                "SELECT * FROM knowledge_graph WHERE id = ?", (edge_id,)
            ).fetchone()
            new_subject = keep if edge["subject_canonical"] == drop else edge["subject_canonical"]
            new_object = keep if edge["object_canonical"] == drop else edge["object_canonical"]

            existing = conn.execute(
                """
                SELECT id, pos_evidence, neg_evidence
                FROM knowledge_graph
                WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?
                """,
                (new_subject, edge["predicate"], new_object),
            ).fetchone()

            if existing and existing["id"] != edge_id:
                conn.execute(
                    """
                    UPDATE knowledge_graph
                    SET pos_evidence = pos_evidence + ?,
                        neg_evidence = neg_evidence + ?,
                        last_seen = MAX(last_seen, ?),
                        last_reinforced = MAX(COALESCE(last_reinforced, ''), COALESCE(?, ''))
                    WHERE id = ?
                    """,
                    (
                        edge["pos_evidence"],
                        edge["neg_evidence"],
                        edge["last_seen"],
                        edge["last_reinforced"],
                        existing["id"],
                    ),
                )
                conn.execute(
                    "UPDATE kg_evidence SET edge_id = ? WHERE edge_id = ?",
                    (existing["id"], edge_id),
                )
                conn.execute("DELETE FROM knowledge_graph WHERE id = ?", (edge_id,))
            else:
                conn.execute(
                    "UPDATE knowledge_graph SET subject_canonical = ?, object_canonical = ? WHERE id = ?",
                    (new_subject, new_object, edge_id),
                )
