from __future__ import annotations

import logging
import hashlib
import inspect
import sqlite3
import textwrap
from pathlib import Path

from hymem.config import HyMemConfig
from hymem.core.graph import graph_clock_order_sql, live_edge_predicate
from hymem.core import db as core_db, markdown_io
from hymem.deadline import check_current_deadline

log = logging.getLogger("hymem.dreaming.phase2")

def confidence(pos: int, neg: int) -> float:
    """Laplace-smoothed positive evidence ratio."""
    return (pos + 1) / (pos + neg + 2)


def consolidate_profile(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    *,
    phase1_generation_key: str | None = None,
    allow_legacy_unscoped: bool = False,
) -> None:
    """Promote unconsolidated markers into structured profile entries.

    Deterministic, no LLM call required. Each marker statement becomes a
    profile entry keyed on its text — repeats reinforce, contradictions get
    surfaced as a separate entry rather than silently overwriting.

    Runner calls are generation-filtered so a newly selected producer cannot
    consume an older producer's fresh marker. In a caller-owned transaction
    this only changes the database: call ``publish_profile`` after commit, or
    let the next ordinary dream repair the sidecar. Standalone calls own and
    commit their database transaction before publishing USER.md. A file error
    therefore never rolls back already-published database authority.
    """
    validate_profile_materialization_policy()
    if not conn.in_transaction:
        with core_db.transaction(conn):
            consolidate_profile(
                conn, cfg, phase1_generation_key=phase1_generation_key,
                allow_legacy_unscoped=allow_legacy_unscoped,
            )
        publish_profile(conn, cfg)
        return
    if phase1_generation_key is None and not allow_legacy_unscoped:
        # A v53 neutral connection can authorize several exact historical
        # producers.  With no selected generation there is no safe way to pick
        # which producer's fresh markers may drive a new Phase-2 projection.
        # Legacy repair/tests must opt into the old unscoped behavior loudly.
        rows = []
    elif phase1_generation_key is None:
        rows = conn.execute(
            "SELECT id,kind,statement,phase1_generation_key "
            "FROM behavioral_markers "
            "WHERE consolidated_at IS NULL "
            "AND phase1_generation_key IS NULL ORDER BY id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT marker.id,marker.kind,marker.statement,"
            "marker.phase1_generation_key "
            "FROM behavioral_markers marker "
            "JOIN current_phase1_publications publication "
            "ON publication.chunk_id=marker.chunk_id "
            "AND publication.phase1_generation_key=marker.phase1_generation_key "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM profile_marker_decisions decision "
            "WHERE decision.marker_id=marker.id "
            "AND decision.profile_policy_key=?) "
            "AND marker.phase1_generation_key=? "
            "ORDER BY marker.id",
            (PROFILE_MATERIALIZATION_POLICY_KEY, phase1_generation_key),
        ).fetchall()
    if not rows:
        return

    kind_to_profile = {
        "preference": "preference",
        "rejection": "avoidance",
        "correction": "context",
        "style": "style",
    }

    from hymem.core.db import evidence_mutation

    with evidence_mutation(conn):
        for row in rows:
            profile_kind = kind_to_profile.get(row["kind"], "context")
            text = row["statement"]
            generation_key = row["phase1_generation_key"]
            existing = conn.execute(
                "SELECT id,kind,source FROM profile_entries WHERE text=?",
                (text,),
            ).fetchone()
            if generation_key is None:
                # Explicit legacy compatibility never upgrades an unattributed
                # marker into current producer authority.
                if existing is None:
                    conn.execute(
                        "INSERT INTO profile_entries(kind,text,source) "
                        "VALUES (?,?,'legacy_unattributed')",
                        (profile_kind, text),
                    )
            elif existing is not None and existing["source"] == "user":
                # A manual/told row dominates an identical inferred signal.
                # Still record a completed non-materializing decision so the
                # marker does not loop forever on every dream.
                conn.execute(
                    "DELETE FROM profile_entry_marker_evidence WHERE marker_id=?",
                    (int(row["id"]),),
                )
                conn.execute(
                    "DELETE FROM profile_marker_decisions WHERE marker_id=?",
                    (int(row["id"]),),
                )
                conn.execute(
                    "INSERT INTO profile_marker_decisions("
                    "marker_id,phase1_generation_key,profile_policy_key,"
                    "decision,profile_entry_id) "
                    "VALUES (?,?,?,'manual_authority',?)",
                    (
                        int(row["id"]), generation_key,
                        PROFILE_MATERIALIZATION_POLICY_KEY, int(existing["id"]),
                    ),
                )
            else:
                if existing is None:
                    cursor = conn.execute(
                        "INSERT INTO profile_entries(kind,text,source) "
                        "VALUES (?,?,'agent_inferred')",
                        (profile_kind, text),
                    )
                    entry_id = int(cursor.lastrowid)
                elif existing["kind"] != profile_kind:
                    # The UNIQUE text row already has another semantic kind;
                    # do not forge a link that contradicts marker lineage, but
                    # record the deterministic non-materializing outcome.
                    entry_id = -1
                else:
                    entry_id = int(existing["id"])
                    if existing["source"] == "legacy_unattributed":
                        conn.execute(
                            "UPDATE profile_entries SET source='agent_inferred',"
                            "last_updated=CURRENT_TIMESTAMP WHERE id=?",
                            (entry_id,),
                        )
                marker_id = int(row["id"])
                conn.execute(
                    "DELETE FROM profile_entry_marker_evidence WHERE marker_id=?",
                    (marker_id,),
                )
                conn.execute(
                    "DELETE FROM profile_marker_decisions WHERE marker_id=?",
                    (marker_id,),
                )
                if entry_id >= 0:
                    conn.execute(
                        "INSERT INTO profile_entry_marker_evidence("
                        "profile_entry_id,marker_id,phase1_generation_key) "
                        "VALUES (?,?,?)",
                        (entry_id, marker_id, generation_key),
                    )
                    conn.execute(
                        "INSERT INTO profile_marker_decisions("
                        "marker_id,phase1_generation_key,profile_policy_key,"
                        "decision,profile_entry_id) "
                        "VALUES (?,?,?,'materialized',?)",
                        (
                            marker_id, generation_key,
                            PROFILE_MATERIALIZATION_POLICY_KEY, entry_id,
                        ),
                    )
                    conn.execute(
                        "UPDATE profile_entries SET pos_evidence=("
                        "SELECT COUNT(*) FROM profile_entry_marker_evidence "
                        "WHERE profile_entry_id=?),last_updated=CURRENT_TIMESTAMP "
                        "WHERE id=?",
                        (entry_id, entry_id),
                    )
                else:
                    conn.execute(
                        "INSERT INTO profile_marker_decisions("
                        "marker_id,phase1_generation_key,profile_policy_key,"
                        "decision,profile_entry_id) "
                        "VALUES (?,?,?,'identity_conflict',?)",
                        (
                            marker_id, generation_key,
                            PROFILE_MATERIALIZATION_POLICY_KEY, int(existing["id"]),
                        ),
                    )
            conn.execute(
                "UPDATE behavioral_markers SET "
                "consolidated_at=COALESCE(consolidated_at,CURRENT_TIMESTAMP) "
                "WHERE id=?",
                (row["id"],),
            )

    # profile_max_entries is a read/render budget.  Deleting a supported row
    # after stamping its marker consolidated made the signal impossible to
    # replay; retain the ledger and apply the cap in `_rewrite_profile_md`.

# Bind completed marker decisions to a pinned, runtime-independent executable
# policy. A deliberate semantic edit must mint a new key. Existing decisions
# remain until their exact retained markers are replayed, then the per-marker
# decision row is replaced (this is not an append-only history). An accidental
# edit fails closed instead of treating old materializations as current.
PROFILE_MATERIALIZATION_POLICY_KEY = "marker-profile-materialization-v5"
PROFILE_MATERIALIZATION_POLICY_SHA256 = (
    "sha256:c0a7a7ffe48bc95975857e1d27f5a477602bbc6695ecdbe56ca4d931069f3fb2"
)


def profile_materialization_policy_sha256() -> str:
    """Return the source commitment captured at module import.

    A rolling deploy may replace this file while an old worker remains alive.
    Late source reads would let that worker stamp NEW source while executing
    its already-loaded OLD function.
    """

    if (
        consolidate_profile is not _PROFILE_MATERIALIZATION_FUNCTION
        or _rewrite_profile_md is not _PROFILE_REWRITE_FUNCTION
    ):
        return "sha256:" + hashlib.sha256(
            b"profile-materialization-runtime-drift"
        ).hexdigest()
    return _PROFILE_MATERIALIZATION_IMPORT_SHA256


def validate_profile_materialization_policy() -> None:
    actual = profile_materialization_policy_sha256()
    if actual != PROFILE_MATERIALIZATION_POLICY_SHA256:
        raise RuntimeError(
            "profile materialization policy changed without a new policy key "
            f"(expected {PROFILE_MATERIALIZATION_POLICY_SHA256}, got {actual})"
        )


def consolidate_insights(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Publish MEMORY.md insights from committed graph state, outside a TX."""
    _assert_sidecar_publication_allowed(conn)
    insights: list[str] = []

    # Hubs: objects depended-on by 2+ subjects with non-trivial confidence.
    hub_rows = conn.execute(
        f"""
        SELECT object_canonical AS obj,
               GROUP_CONCAT(subject_canonical, ', ') AS subjects,
               COUNT(*) AS cnt
        FROM knowledge_graph
        WHERE predicate = 'depends_on'
           AND {live_edge_predicate()}
           AND (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) > 0.6
        GROUP BY object_canonical
        HAVING cnt >= 2
        ORDER BY cnt DESC, obj ASC
        LIMIT ?
        """,
        (cfg.insights_max_entries,),
    ).fetchall()
    for r in hub_rows:
        insights.append(
            f"- `{r['obj']}` is a shared dependency of: {r['subjects']}."
        )

    # Strong tool preferences and rejections.
    pref_rows = conn.execute(
        f"""
        SELECT predicate, subject_canonical AS s, object_canonical AS o,
               pos_evidence AS pos, neg_evidence AS neg, last_reinforced
        FROM knowledge_graph
        WHERE predicate IN ('prefers','rejects','avoids')
           AND {live_edge_predicate()}
           AND (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) > 0.7
        ORDER BY pos_evidence DESC,
                 {graph_clock_order_sql('last_reinforced')}, id
        LIMIT ?
        """,
        (cfg.insights_max_entries,),
    ).fetchall()
    for r in pref_rows:
        verb = {"prefers": "prefers", "rejects": "rejects", "avoids": "avoids"}[r["predicate"]]
        insights.append(f"- `{r['s']}` {verb} `{r['o']}` (evidence {r['pos']}/{r['pos'] + r['neg']}).")

    # Contradictions: an edge with both significant pos and neg evidence.
    contradiction_rows = conn.execute(
        f"""
        SELECT subject_canonical AS s, predicate AS p, object_canonical AS o,
               pos_evidence AS pos, neg_evidence AS neg
        FROM knowledge_graph
        WHERE pos_evidence >= 1 AND neg_evidence >= 1
          AND {live_edge_predicate()}
        ORDER BY (pos_evidence + neg_evidence) DESC
        LIMIT 5
        """
    ).fetchall()
    for r in contradiction_rows:
        insights.append(
            f"- ⚠ conflicting evidence: `{r['s']}` {r['p']} `{r['o']}` "
            f"(+{r['pos']} / -{r['neg']})."
        )

    if not insights:
        body = "_No insights yet — the graph is still warming up._"
    else:
        body = "\n".join(insights[: cfg.insights_max_entries])

    markdown_io.write_section(
        cfg.memory_md_path,
        "project_insights",
        body,
        header="## Project Insights (auto)",
        before_replace=lambda: _assert_sidecar_publication_allowed(conn),
    )


def _rewrite_profile_md(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    _assert_sidecar_publication_allowed(conn)
    body = current_profile_body(conn, cfg)
    markdown_io.write_section(
        cfg.user_md_path,
        "behavioral_profile",
        body,
        header="## Behavioral Profile (auto, do not edit manually)",
        before_replace=lambda: _assert_sidecar_publication_allowed(conn),
    )


def _assert_sidecar_publication_allowed(conn: sqlite3.Connection) -> None:
    """Cooperative preflight, not an atomic SQLite/filesystem commit fence."""
    if conn.in_transaction:
        raise RuntimeError("sidecar publication requires committed state outside a transaction")
    check_current_deadline()
    core_db._assert_transaction_lease_owned(conn)


def publish_profile(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Repair USER.md from committed authority, preserving manual sections."""
    _rewrite_profile_md(conn, cfg)


def current_profile_body(conn: sqlite3.Connection, cfg: HyMemConfig) -> str:
    """Render only the producer-authorized behavioral profile projection."""

    rows = conn.execute(
        """
        SELECT kind, text, pos_evidence, neg_evidence
        FROM current_profile_entries
        ORDER BY pos_evidence DESC, last_updated DESC
        LIMIT ?
        """,
        (cfg.profile_max_entries,),
    ).fetchall()

    if not rows:
        return "_No behavioral signals collected yet._"
    lines = []
    for r in rows:
        conf = confidence(r["pos_evidence"], r["neg_evidence"])
        lines.append(f"- [{r['kind']}] {r['text']} _(confidence {conf:.2f})_")
    return "\n".join(lines)


def authoritative_user_markdown(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
) -> str:
    """Return USER.md with its managed section refreshed in memory.

    The file remains a compatibility/inspection sidecar.  A producer switch is
    connection-local and can happen before another dream rewrites that file, so
    query paths must never trust its auto section as read authority.
    """

    existing = (
        cfg.user_md_path.read_text(encoding="utf-8")
        if cfg.user_md_path.exists() else ""
    )
    return markdown_io.render_section(
        existing,
        "behavioral_profile",
        current_profile_body(conn, cfg),
        header="## Behavioral Profile (auto, do not edit manually)",
        insert_if_missing=True,
    )


_PROFILE_MATERIALIZATION_FUNCTION = consolidate_profile
_PROFILE_REWRITE_FUNCTION = _rewrite_profile_md
_PROFILE_MATERIALIZATION_SOURCE_AT_IMPORT = (
    textwrap.dedent(inspect.getsource(consolidate_profile))
    .replace("\r\n", "\n").replace("\r", "\n").strip()
)
_PROFILE_MATERIALIZATION_IMPORT_SHA256 = (
    "sha256:" + hashlib.sha256(
        _PROFILE_MATERIALIZATION_SOURCE_AT_IMPORT.encode("utf-8")
    ).hexdigest()
)
