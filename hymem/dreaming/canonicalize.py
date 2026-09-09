from __future__ import annotations

import json
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
# punctuation. Casefold and accent-fold Latin bases, but retain non-Latin
# letters and their meaningful combining marks in the SAME pipeline. A mixed
# name must never lose its Unicode identity merely because it has ASCII too.
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
_MAX_UNICODE_CANONICAL_CHARS = 512
CANONICALIZATION_POLICY_VERSION = "hymem-unicode-canonicalization-v2"
CANONICAL_UNICODE_VERSION = unicodedata.unidata_version


def _fold_latin_accents(surface: str) -> str:
    """Compatibility-decompose, dropping marks only after a Latin base.

    Unicode word regexes omit marks such as Devanagari vowels; dropping all marks
    would likewise conflate Cyrillic Й/И, Greek accents, and vocalized Arabic.
    Keep them through NFC composition and the later token filter instead.
    """
    out: list[str] = []
    latin_base = False
    for character in unicodedata.normalize("NFKD", surface):
        category = unicodedata.category(character)
        if category.startswith("M"):
            if not latin_base:
                out.append(character)
        else:
            latin_base = category.startswith("L") and (
                "LATIN" in unicodedata.name(character, "")
            )
            out.append(character)
    return "".join(out)


def normalize(surface: str) -> str:
    """Return a bounded, idempotent Unicode key, or empty for invalid names.

    Compatibility spellings and case fold together; Latin accents, leading
    articles and trailing parentheticals retain the historical policy. Other
    letters/numbers and attached marks remain meaningful. Punctuation,
    symbols, controls, and unattached marks separate tokens. No truncation is
    allowed, including for ASCII names: oversized keys fail closed.
    """
    s = _fold_latin_accents(surface)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    # Casefold can expand characters (ß -> ss, for example). Decompose again
    # before classifying marks so the result remains a normalization fixed point.
    s = _fold_latin_accents(s.casefold()).strip()
    s = _TRAILING_PAREN.sub("", s)
    s = _LEADING_ARTICLES.sub("", s)
    out: list[str] = []
    attached = False
    for character in s:
        category = unicodedata.category(character)
        if category[0] in {"L", "N"}:
            out.append(character)
            attached = True
        elif category.startswith("M") and attached:
            out.append(character)
        else:
            if out and out[-1] != "_":
                out.append("_")
            attached = False
    key = unicodedata.normalize("NFC", "".join(out).strip("_"))
    if len(key) > _MAX_UNICODE_CANONICAL_CHARS:
        return ""
    return key


def resolve(conn: sqlite3.Connection, surface: str) -> str:
    """Return the canonical id for `surface`, consulting the alias table."""
    norm = normalize(surface)
    row = conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias = ?", (norm,)
    ).fetchone()
    return row["canonical"] if row else norm


def register_alias(conn: sqlite3.Connection, surface: str, canonical: str) -> None:
    """Map a pure surface form onto an existing canonical id.

    If the normalized surface already owns canonical state, this operation
    would strand that state behind the new alias because ``resolve`` is
    intentionally one hop.  Such identity changes must use :func:`merge`,
    which rewrites and re-hashes every provenance-bearing domain.
    """
    if not isinstance(surface, str) or not isinstance(canonical, str):
        raise ValueError("entity alias and canonical must be strings")
    alias = normalize(surface)
    if not alias:
        raise ValueError("entity alias must not be empty")
    if not canonical.strip() or normalize(canonical) != canonical:
        raise ValueError("alias target must be a normalized canonical identity")
    chained = conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias=?", (canonical,),
    ).fetchone()
    if chained is not None and str(chained["canonical"]) != canonical:
        raise ValueError("alias target must not itself be an alias")
    existing_alias = conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias=?", (alias,),
    ).fetchone()
    if (
        existing_alias is not None
        and str(existing_alias["canonical"]) != canonical
    ):
        raise ValueError("entity alias already maps to another canonical identity")
    if alias != canonical:
        owned = False
        scalar_owners = (
            ("entity_aliases", "canonical"),
            ("entity_types", "entity_canonical"),
            ("entity_properties", "entity_canonical"),
            ("entity_type_observations", "entity_canonical"),
            ("entity_property_observations", "entity_canonical"),
            ("entity_mention_observations", "entity_canonical"),
            ("entity_mentions", "entity_canonical"),
            ("knowledge_graph", "subject_canonical"),
            ("knowledge_graph", "object_canonical"),
        )
        for table, column in scalar_owners:
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone() is None:
                continue
            if conn.execute(
                f"SELECT 1 FROM {table} WHERE {column}=? LIMIT 1", (alias,),
            ).fetchone() is not None:
                owned = True
                break
        if not owned and conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rules'"
        ).fetchone() is not None:
            for row in conn.execute(
                "SELECT trigger_entities FROM rules WHERE scope='contextual'"
            ).fetchall():
                try:
                    triggers = json.loads(row["trigger_entities"])
                except (TypeError, ValueError):
                    continue
                if isinstance(triggers, list) and alias in triggers:
                    owned = True
                    break
        if owned:
            raise ValueError(
                f"canonical identity {alias!r} already owns state; "
                "use merge_canonical/merge"
            )
    conn.execute(
        "INSERT OR REPLACE INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        (alias, canonical),
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
    for table, column in (
        ("entity_types", "entity_canonical"),
        ("entity_properties", "entity_canonical"),
        ("entity_type_observations", "entity_canonical"),
        ("entity_property_observations", "entity_canonical"),
        ("entity_mention_observations", "entity_canonical"),
        ("entity_mentions", "entity_canonical"),
    ):
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone() is None:
            continue
        for row in conn.execute(
            f"SELECT DISTINCT {column} AS v FROM {table}"
        ).fetchall():
            value = row["v"]
            if value != normalize(value):
                findings.append((f"{table}.{column}", value))
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rules'"
    ).fetchone() is not None:
        for row in conn.execute(
            "SELECT trigger_entities FROM rules WHERE scope='contextual'"
        ).fetchall():
            try:
                triggers = json.loads(row["trigger_entities"])
            except (TypeError, ValueError):
                continue
            if not isinstance(triggers, list):
                continue
            for value in triggers:
                if isinstance(value, str) and value != normalize(value):
                    findings.append(("rules.trigger_entities", value))
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

    # V54 auxiliary ledgers and contextual rule triggers can be the only
    # remaining owner of a canonical identity. Finder coverage without seeding
    # those values here reported drift that repair could never actually reach.
    for location, value in find_canonical_drift(conn):
        if location != "entity_aliases.alias" and value != normalize(value):
            drifted_canonicals.add(value)

    for drift in sorted(drifted_canonicals):
        target = normalize(drift)
        merge(conn, keep=target, drop=drift, _allow_legacy_drop=True)
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


def merge(
    conn: sqlite3.Connection,
    keep: str,
    drop: str,
    *,
    _allow_legacy_drop: bool = False,
) -> None:
    """Fold all edges and aliases referencing `drop` into `keep`.

    Caller is responsible for being inside a transaction.
    """
    if not isinstance(keep, str) or not isinstance(drop, str):
        raise ValueError("canonical identities must be strings")
    if not keep.strip() or normalize(keep) != keep:
        raise ValueError("merge identities must be normalized canonical values")
    if (
        not _allow_legacy_drop
        and (not drop.strip() or normalize(drop) != drop)
    ):
        raise ValueError("merge identities must be normalized canonical values")
    if keep == drop:
        return

    # Resolving a kept name through an unrelated alias after moving state to
    # that name would strand the state behind a one-hop redirect. A self-map,
    # or a redirect to the exact identity being consumed, is safe: the latter
    # becomes a self-map when references to drop are rewritten below.
    target_alias = conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias=?", (keep,),
    ).fetchone()
    if target_alias is not None and target_alias["canonical"] not in (keep, drop):
        raise ValueError("merge target must not alias an unrelated canonical identity")

    # Two explicit user assertions are equal authority.  A differing value is
    # therefore a real semantic conflict, not something the arbitrary
    # keep/drop direction may resolve.  Detect it before touching aliases,
    # observations, rules, or graph state so callers without an outer
    # transaction also fail without a partial merge.
    property_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(entity_properties)").fetchall()
    }
    if {"entity_canonical", "key", "value", "origin"}.issubset(
        property_columns
    ):
        conflict = conn.execute(
            "SELECT kept.key FROM entity_properties kept "
            "JOIN entity_properties dropped ON dropped.key=kept.key "
            "WHERE kept.entity_canonical=? AND dropped.entity_canonical=? "
            "AND kept.origin='user' AND dropped.origin='user' "
            "AND kept.value<>dropped.value LIMIT 1",
            (keep, drop),
        ).fetchone()
        if conflict is not None:
            raise ValueError(
                "cannot merge conflicting manual entity property "
                f"{conflict['key']!r}"
            )

    conn.execute(
        "UPDATE OR IGNORE entity_aliases SET canonical = ? WHERE canonical = ?",
        (keep, drop),
    )
    conn.execute(
        "INSERT OR REPLACE INTO entity_aliases(alias, canonical) VALUES (?, ?)",
        (drop, keep),
    )

    # Entity hints are source-owned publications too.  Move their observation
    # identities under the same explicit canonical merge and rehash every
    # affected auxiliary outcome below.  Older stores simply lack these tables.
    auxiliary_identities: list[tuple[str, str]] = []
    auxiliary_tables = {
        "entity_type_observations": {
            "chunk_id", "entity_canonical", "type", "confidence",
            "phase1_generation_key", "observed_at",
        },
        "entity_property_observations": {
            "chunk_id", "entity_canonical", "key", "value",
            "phase1_generation_key", "observed_at",
        },
        "entity_mention_observations": {
            "chunk_id", "entity_canonical", "phase1_generation_key",
            "observed_at",
        },
        "phase1_auxiliary_outcomes": {
            "chunk_id", "phase1_generation_key", "result_hash",
        },
    }
    auxiliary_shape = all(
        required.issubset({
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        })
        for table, required in auxiliary_tables.items()
    )
    if auxiliary_shape:
        auxiliary_identities = [
            (str(row["chunk_id"]), str(row["phase1_generation_key"]))
            for row in conn.execute(
                "SELECT DISTINCT chunk_id,phase1_generation_key "
                "FROM entity_type_observations WHERE entity_canonical=? "
                "UNION SELECT DISTINCT chunk_id,phase1_generation_key "
                "FROM entity_property_observations WHERE entity_canonical=? "
                "UNION SELECT DISTINCT chunk_id,phase1_generation_key "
                "FROM entity_mention_observations WHERE entity_canonical=?",
                (drop, drop, drop),
            ).fetchall()
        ]
        from hymem.core.db import evidence_mutation

        with evidence_mutation(conn):
            conn.execute(
                "INSERT INTO entity_type_observations("
                "chunk_id,entity_canonical,type,confidence,"
                "phase1_generation_key,observed_at) "
                "SELECT chunk_id,?,type,confidence,phase1_generation_key,"
                "observed_at FROM entity_type_observations "
                "WHERE entity_canonical=? AND 1 "
                "ON CONFLICT(chunk_id,entity_canonical,type,"
                "phase1_generation_key) DO UPDATE SET "
                "confidence=MAX(entity_type_observations.confidence,"
                "excluded.confidence)",
                (keep, drop),
            )
            conn.execute(
                "DELETE FROM entity_type_observations WHERE entity_canonical=?",
                (drop,),
            )
            property_rows = conn.execute(
                "SELECT chunk_id,key,value,phase1_generation_key,observed_at "
                "FROM entity_property_observations WHERE entity_canonical=? "
                "ORDER BY chunk_id,phase1_generation_key,key",
                (drop,),
            ).fetchall()
            for row in property_rows:
                identity = (
                    row["chunk_id"], keep, row["key"],
                    row["phase1_generation_key"],
                )
                existing = conn.execute(
                    "SELECT value,observed_at FROM "
                    "entity_property_observations WHERE chunk_id=? "
                    "AND entity_canonical=? AND key=? "
                    "AND phase1_generation_key=?",
                    identity,
                ).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO entity_property_observations("
                        "chunk_id,entity_canonical,key,value,"
                        "phase1_generation_key,observed_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            row["chunk_id"], keep, row["key"], row["value"],
                            row["phase1_generation_key"], row["observed_at"],
                        ),
                    )
                elif existing["value"] != row["value"]:
                    # A post-extraction alias merge exposed an ambiguity that a
                    # fresh extraction under the alias map would have omitted.
                    # Preserve that equivalence; never invent a lexical winner.
                    conn.execute(
                        "DELETE FROM entity_property_observations "
                        "WHERE chunk_id=? AND entity_canonical=? AND key=? "
                        "AND phase1_generation_key=?",
                        identity,
                    )
            conn.execute(
                "DELETE FROM entity_property_observations "
                "WHERE entity_canonical=?", (drop,)
            )
            conn.execute(
                "INSERT OR IGNORE INTO entity_mention_observations("
                "chunk_id,entity_canonical,phase1_generation_key,observed_at) "
                "SELECT chunk_id,?,phase1_generation_key,observed_at "
                "FROM entity_mention_observations WHERE entity_canonical=?",
                (keep, drop),
            )
            conn.execute(
                "DELETE FROM entity_mention_observations "
                "WHERE entity_canonical=?", (drop,)
            )

    # Compatibility/manual projections use explicit origin; NULL source_chunk
    # is never interpreted as manual.  A user row wins any collision.
    try:
        for row in conn.execute(
            "SELECT type,confidence,source_chunk_id,origin FROM entity_types "
            "WHERE entity_canonical=?",
            (drop,),
        ).fetchall():
            existing = conn.execute(
                "SELECT confidence,source_chunk_id,origin FROM entity_types "
                "WHERE entity_canonical=? AND type=?",
                (keep, row["type"]),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO entity_types(entity_canonical,type,confidence,"
                    "source_chunk_id,origin) VALUES (?,?,?,?,?)",
                    (keep, row["type"], row["confidence"],
                     row["source_chunk_id"], row["origin"]),
                )
            elif existing["origin"] != "user" and row["origin"] == "user":
                conn.execute(
                    "UPDATE entity_types SET confidence=?,source_chunk_id=?,"
                    "origin='user' WHERE entity_canonical=? AND type=?",
                    (row["confidence"], row["source_chunk_id"], keep, row["type"]),
                )
        conn.execute("DELETE FROM entity_types WHERE entity_canonical=?", (drop,))
        for row in conn.execute(
            "SELECT key,value,source_chunk_id,origin FROM entity_properties "
            "WHERE entity_canonical=?",
            (drop,),
        ).fetchall():
            existing = conn.execute(
                "SELECT value,source_chunk_id,origin FROM entity_properties "
                "WHERE entity_canonical=? AND key=?",
                (keep, row["key"]),
            ).fetchone()
            use_drop = existing is None or (
                existing["origin"] != "user" and row["origin"] == "user"
            )
            if existing is None:
                conn.execute(
                    "INSERT INTO entity_properties(entity_canonical,key,value,"
                    "source_chunk_id,origin) VALUES (?,?,?,?,?)",
                    (keep, row["key"], row["value"], row["source_chunk_id"],
                     row["origin"]),
                )
            elif use_drop:
                conn.execute(
                    "UPDATE entity_properties SET value=?,source_chunk_id=?,"
                    "origin=? WHERE entity_canonical=? AND key=?",
                    (row["value"], row["source_chunk_id"], row["origin"],
                     keep, row["key"]),
                )
        conn.execute(
            "DELETE FROM entity_properties WHERE entity_canonical=?", (drop,)
        )
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute(
            "INSERT OR IGNORE INTO entity_mentions(chunk_id,entity_canonical) "
            "SELECT chunk_id,? FROM entity_mentions WHERE entity_canonical=?",
            (keep, drop),
        )
        conn.execute(
            "DELETE FROM entity_mentions WHERE entity_canonical=?", (drop,)
        )
    except sqlite3.OperationalError:
        pass

    # Contextual told rules carry canonical trigger ids.  Preserve their user
    # authority while keeping a merge from silently disabling the rule.
    try:
        from hymem.core.db import evidence_mutation

        with evidence_mutation(conn):
            for row in conn.execute(
                "SELECT id,trigger_entities FROM rules WHERE scope='contextual'"
            ).fetchall():
                try:
                    triggers = json.loads(row["trigger_entities"])
                except (TypeError, ValueError):
                    continue
                if not isinstance(triggers, list) or drop not in triggers:
                    continue
                replacement = sorted({
                    keep if item == drop else item for item in triggers
                })
                conn.execute(
                    "UPDATE rules SET trigger_entities=? WHERE id=?",
                    (json.dumps(replacement), row["id"]),
                )
    except sqlite3.OperationalError:
        pass

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

    if auxiliary_identities:
        from hymem.dreaming.phase1_auxiliary import (
            refresh_phase1_auxiliary_outcomes,
        )

        refresh_phase1_auxiliary_outcomes(conn, auxiliary_identities)
