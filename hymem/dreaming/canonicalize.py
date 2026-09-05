from __future__ import annotations

import re
import sqlite3
import unicodedata

from hymem.dreaming import evidence
from hymem.core.time import (
    latest_timestamp_spelling,
    normalize_iso_timestamp,
    timestamp_at_or_before,
)

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

    # Migrate edges.  On a collision, provenance is moved and deduplicated by
    # source before cached counters are rebuilt; blindly summing the two caches
    # double-counted a chunk that supported both aliases.
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
                SELECT id, pos_evidence, neg_evidence, last_seen,
                       last_reinforced
                FROM knowledge_graph
                WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ?
                """,
                (new_subject, edge["predicate"], new_object),
            ).fetchone()

            if existing and existing["id"] != edge_id:
                present_cutoff = conn.execute(
                    "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now','+300 seconds')"
                ).fetchone()[0]

                def latest_present(left, right, *, fallback):
                    usable = []
                    for value in (left, right):
                        try:
                            canonical = normalize_iso_timestamp(
                                value, context="edge merge timestamp"
                            )
                        except ValueError:
                            continue
                        if timestamp_at_or_before(canonical, present_cutoff):
                            usable.append(value)
                    if not usable:
                        # Neither branch has usable recency authority. Collapse
                        # to one conservative value so A->B and B->A converge
                        # instead of preserving whichever poison survived.
                        return fallback
                    return latest_timestamp_spelling(*usable)

                conn.execute(
                    """
                    UPDATE knowledge_graph
                    SET last_seen = ?, last_reinforced = ?
                    WHERE id = ?
                    """,
                    (
                        latest_present(
                            existing["last_seen"], edge["last_seen"],
                            fallback="0001-01-01T00:00:00.000Z",
                        ),
                        latest_present(
                            existing["last_reinforced"],
                            edge["last_reinforced"],
                            fallback=None,
                        ),
                        existing["id"],
                    ),
                )
                evidence.move_edge_provenance(conn, existing["id"], [edge_id])
                conn.execute("DELETE FROM knowledge_graph WHERE id = ?", (edge_id,))
            else:
                outcome_chunks = [
                    str(item["chunk_id"])
                    for item in conn.execute(
                        "SELECT DISTINCT chunk_id FROM kg_claim_observations "
                        "WHERE edge_id=?",
                        (edge_id,),
                    ).fetchall()
                ]
                conn.execute(
                    "UPDATE knowledge_graph SET subject_canonical = ?, object_canonical = ? WHERE id = ?",
                    (new_subject, new_object, edge_id),
                )
                evidence.recanonicalize_lifecycle_keys(conn)
                evidence.refresh_claim_extraction_outcomes(conn, outcome_chunks)
