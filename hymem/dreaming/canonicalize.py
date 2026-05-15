from __future__ import annotations

import re
import sqlite3
import unicodedata

# Strip leading articles, trailing parentheticals like "(container)", and
# punctuation. Lowercase. ASCII-fold. Collapse whitespace and underscores.
# Articles cover common Latin-script European languages; this runs after the
# string is already lowercased and accent-folded.
_LEADING_ARTICLES = re.compile(
    r"^(the|an?|"                       # English
    r"de|het|een|"                      # Dutch
    r"der|die|das|dem|den|ein|eine|"    # German
    r"le|la|les|un|une|des|"            # French
    r"el|los|las|una|unos|unas|"        # Spanish
    r"il|lo|gli|uno|"                   # Italian
    r"os|as|um|uma"                     # Portuguese
    r")\s+",
    re.IGNORECASE,
)
_TRAILING_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(surface: str) -> str:
    """Deterministic surface -> canonical key. Pure function, no DB needed."""
    s = unicodedata.normalize("NFKD", surface).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
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


def find_canonical_drift(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return values stored in canonical columns that fail `normalize(v) == v`.

    Read-only. Surfaces write-path drift — rows that got into the DB without
    flowing through normalize() (e.g. direct SQL writes, third-party tools,
    or older code paths). Each item is (location, value) where location is
    one of "entity_aliases.canonical", "entity_aliases.alias",
    "knowledge_graph.subject_canonical", "knowledge_graph.object_canonical".
    """
    findings: list[tuple[str, str]] = []
    for query, location in (
        ("SELECT DISTINCT canonical AS v FROM entity_aliases", "entity_aliases.canonical"),
        ("SELECT DISTINCT alias AS v FROM entity_aliases", "entity_aliases.alias"),
        (
            "SELECT DISTINCT subject_canonical AS v FROM knowledge_graph",
            "knowledge_graph.subject_canonical",
        ),
        (
            "SELECT DISTINCT object_canonical AS v FROM knowledge_graph",
            "knowledge_graph.object_canonical",
        ),
    ):
        for row in conn.execute(query).fetchall():
            v = row["v"]
            if v != normalize(v):
                findings.append((location, v))
    return findings


def repair_canonical_drift(conn: sqlite3.Connection) -> list[dict]:
    """Rewrite drifted canonicals to their normalized form.

    Detects every value that fails `normalize(v) == v` across the four
    canonical columns, then rewrites references in place. When the normalized
    form already exists as a different canonical, edges with the same
    (subject, predicate, object) collapse via evidence summing — the same
    semantics as `merge()`. Caller controls the transaction.

    Returns a list of `{column, from, to, collision?}` records describing
    the fixes applied.
    """
    fixes: list[dict] = []

    drifted_canonicals: set[str] = set()
    for query in (
        "SELECT DISTINCT canonical AS v FROM entity_aliases",
        "SELECT DISTINCT subject_canonical AS v FROM knowledge_graph",
        "SELECT DISTINCT object_canonical AS v FROM knowledge_graph",
    ):
        for row in conn.execute(query).fetchall():
            v = row["v"]
            if v != normalize(v):
                drifted_canonicals.add(v)

    for drift in sorted(drifted_canonicals):
        target = normalize(drift)
        merge(conn, keep=target, drop=drift)
        # merge() preserves the drifted surface form as an alias key. We don't
        # want un-normalized alias keys in the table — drop that artifact.
        conn.execute("DELETE FROM entity_aliases WHERE alias = ?", (drift,))
        fixes.append({"column": "canonical", "from": drift, "to": target})

    for row in conn.execute("SELECT alias FROM entity_aliases").fetchall():
        alias = row["alias"]
        norm = normalize(alias)
        if alias == norm:
            continue
        existing = conn.execute(
            "SELECT 1 FROM entity_aliases WHERE alias = ?", (norm,)
        ).fetchone()
        if existing is None:
            conn.execute(
                "UPDATE entity_aliases SET alias = ? WHERE alias = ?", (norm, alias)
            )
            fixes.append({"column": "alias", "from": alias, "to": norm})
        else:
            conn.execute("DELETE FROM entity_aliases WHERE alias = ?", (alias,))
            fixes.append(
                {"column": "alias", "from": alias, "to": norm, "collision": True}
            )

    return fixes


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
