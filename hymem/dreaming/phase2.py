from __future__ import annotations

import sqlite3
from pathlib import Path

from hymem.config import HyMemConfig
from hymem.core import markdown_io


def confidence(pos: int, neg: int) -> float:
    """Laplace-smoothed positive evidence ratio."""
    return (pos + 1) / (pos + neg + 2)


def consolidate_profile(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Promote unconsolidated markers into structured profile entries.

    Deterministic, no LLM call required. Each marker statement becomes a
    profile entry keyed on its text — repeats reinforce, contradictions get
    surfaced as a separate entry rather than silently overwriting.
    """
    rows = conn.execute(
        """
        SELECT id, kind, statement
        FROM behavioral_markers
        WHERE consolidated_at IS NULL
        ORDER BY id
        """
    ).fetchall()
    if not rows:
        _rewrite_profile_md(conn, cfg)
        return

    kind_to_profile = {
        "preference": "preference",
        "rejection": "avoidance",
        "correction": "context",
        "style": "style",
    }

    for row in rows:
        profile_kind = kind_to_profile.get(row["kind"], "context")
        text = row["statement"]
        existing = conn.execute(
            "SELECT id, pos_evidence FROM profile_entries WHERE text = ?",
            (text,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE profile_entries
                SET pos_evidence = pos_evidence + 1,
                    last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (existing["id"],),
            )
        else:
            conn.execute(
                "INSERT INTO profile_entries(kind, text) VALUES (?, ?)",
                (profile_kind, text),
            )
        conn.execute(
            "UPDATE behavioral_markers SET consolidated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (row["id"],),
        )

    # Cap profile size: drop weakest entries when over the limit.
    conn.execute(
        """
        DELETE FROM profile_entries
        WHERE id IN (
            SELECT id FROM profile_entries
            ORDER BY pos_evidence ASC, last_updated ASC
            LIMIT MAX(0, (SELECT COUNT(*) FROM profile_entries) - ?)
        )
        """,
        (cfg.profile_max_entries,),
    )

    _rewrite_profile_md(conn, cfg)


def consolidate_insights(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    """Refresh MEMORY.md's "Project Insights" section from the current graph."""
    insights: list[str] = []

    # Hubs: objects depended-on by 2+ subjects with non-trivial confidence.
    hub_rows = conn.execute(
        """
        SELECT object_canonical AS obj,
               GROUP_CONCAT(subject_canonical, ', ') AS subjects,
               COUNT(*) AS cnt
        FROM knowledge_graph
        WHERE predicate = 'depends_on'
          AND status = 'active'
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
        """
        SELECT predicate, subject_canonical AS s, object_canonical AS o,
               pos_evidence AS pos, neg_evidence AS neg, last_reinforced
        FROM knowledge_graph
        WHERE predicate IN ('prefers','rejects','avoids')
          AND status = 'active'
          AND (pos_evidence + 1.0) / (pos_evidence + neg_evidence + 2.0) > 0.7
        ORDER BY pos_evidence DESC, last_reinforced DESC
        LIMIT ?
        """,
        (cfg.insights_max_entries,),
    ).fetchall()
    for r in pref_rows:
        verb = {"prefers": "prefers", "rejects": "rejects", "avoids": "avoids"}[r["predicate"]]
        insights.append(f"- `{r['s']}` {verb} `{r['o']}` (evidence {r['pos']}/{r['pos'] + r['neg']}).")

    # Contradictions: an edge with both significant pos and neg evidence.
    contradiction_rows = conn.execute(
        """
        SELECT subject_canonical AS s, predicate AS p, object_canonical AS o,
               pos_evidence AS pos, neg_evidence AS neg
        FROM knowledge_graph
        WHERE pos_evidence >= 1 AND neg_evidence >= 1 AND status = 'active'
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
    )


def _rewrite_profile_md(conn: sqlite3.Connection, cfg: HyMemConfig) -> None:
    rows = conn.execute(
        """
        SELECT kind, text, pos_evidence, neg_evidence
        FROM profile_entries
        ORDER BY pos_evidence DESC, last_updated DESC
        LIMIT ?
        """,
        (cfg.profile_max_entries,),
    ).fetchall()

    if not rows:
        body = "_No behavioral signals collected yet._"
    else:
        lines = []
        for r in rows:
            conf = confidence(r["pos_evidence"], r["neg_evidence"])
            lines.append(f"- [{r['kind']}] {r['text']} _(confidence {conf:.2f})_")
        body = "\n".join(lines)

    markdown_io.write_section(
        cfg.user_md_path,
        "behavioral_profile",
        body,
        header="## Behavioral Profile (auto, do not edit manually)",
    )
