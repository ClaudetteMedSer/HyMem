"""MCP server for HyMem.

Exposes seven tools to the Hermes Agent platform:
  hymem_capture  — log a full conversation at once + optionally dream (preferred)
  hymem_log      — log one conversational turn (fallback for turn-by-turn use)
  hymem_dream    — run a dreaming cycle (extract, consolidate, decay)
  hymem_augment  — retrieve graph facts + FTS context for a user message
  hymem_profile  — return USER.md (behavioral profile) + MEMORY.md (project insights)
  hymem_alias    — register a surface-form alias for an entity
  hymem_retract  — retract a wrongly extracted knowledge graph edge

Run via the installed entry point:
    hymem-server

Or directly:
    python -m hymem.server

Configuration is entirely through environment variables (see README or
hymem/contrib/openai_client.py for the full list).

Key variables:
    HYMEM_LLM_API_KEY        API key for the extraction LLM (or DEEPSEEK_API_KEY)
    HYMEM_LLM_BASE_URL       Base URL (default: https://api.deepseek.com)
    HYMEM_LLM_MODEL          Model name (default: deepseek-chat)
    HYMEM_EMBEDDING_API_KEY  API key for embeddings (falls back to LLM key)
    HYMEM_EMBEDDING_BASE_URL Embedding endpoint (default: https://api.deepseek.com)
    HYMEM_EMBEDDING_MODEL    Embedding model (default: deepseek-embedding)
    HYMEM_ROOT               Directory for hymem.sqlite, MEMORY.md, USER.md
                             (default: ~/.hermes)

If the embedding client cannot be constructed (e.g. API key absent), the server
logs a warning and falls back to FTS-only retrieval — no other functionality
is affected.
"""
from __future__ import annotations

import json

# Startup, env-var resolution, and the shared singleton live in hymem.bootstrap.
# Re-exported here under the historical names used by tests and tool helpers.
from hymem.bootstrap import get_instance as _get_hy, set_instance as set_hy


def _get_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "mcp package required: pip install 'hymem[server]'"
        ) from e
    return FastMCP("hymem")


mcp = None


# ── tool implementations (callable directly in tests) ────────────────────────

def _do_capture(session_id: str, messages: str, dream: bool = True) -> str:
    try:
        turns = json.loads(messages)
    except json.JSONDecodeError as e:
        return f"error: messages must be a JSON array — {e}"

    if not isinstance(turns, list):
        return "error: messages must be a JSON array"

    hy = _get_hy()
    hy.open_session(session_id)
    logged = 0
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        if not content:
            continue
        hy.log_message(session_id, role, content)
        logged += 1
    hy.close_session(session_id)

    if not dream:
        return f"logged {logged} turns for session {session_id!r}"

    report = hy.dream(session_ids=[session_id])
    if report.skipped_locked:
        return (
            f"logged {logged} turns for session {session_id!r}; "
            "dreaming skipped (another cycle is running — will pick up via cron)"
        )
    return (
        f"logged {logged} turns for session {session_id!r}; "
        f"dreaming complete — {report.chunks_processed}/{report.chunks_seen} chunks, "
        f"{report.triples_extracted} triples, {report.markers_extracted} markers"
    )


def _do_log(session_id: str, role: str, content: str) -> str:
    _get_hy().log_message(session_id, role, content)
    return "logged"


def _do_dream() -> str:
    report = _get_hy().dream()
    if report.skipped_locked:
        return "skipped: another dreaming cycle is already running"
    return (
        f"dreaming complete — "
        f"{report.sessions_processed} sessions, "
        f"{report.chunks_processed}/{report.chunks_seen} chunks processed, "
        f"{report.triples_extracted} triples, "
        f"{report.markers_extracted} markers extracted"
    )


def _do_augment(message: str) -> str:
    ctx = _get_hy().augment(message)
    parts: list[str] = []

    if ctx.graph_facts:
        lines = [
            f"- {f.subject} {f.predicate} {f.object} (conf {f.confidence:.2f},"
            f" +{f.pos_evidence}/-{f.neg_evidence})"
            for f in ctx.graph_facts
        ]
        parts.append("**Structured knowledge (knowledge graph):**\n" + "\n".join(lines))

    if ctx.fts_hits:
        snippets = [f"[{h.session_id}] {h.text[:300]}" for h in ctx.fts_hits]
        parts.append("**Relevant past context (keyword search):**\n" + "\n".join(snippets))

    return "\n\n".join(parts) if parts else ""


def _do_profile() -> str:
    hy = _get_hy()
    cfg = hy.config
    user = cfg.user_md_path.read_text(encoding="utf-8") if cfg.user_md_path.exists() else ""
    memory = cfg.memory_md_path.read_text(encoding="utf-8") if cfg.memory_md_path.exists() else ""
    parts: list[str] = []
    if user.strip():
        parts.append("=== USER PROFILE ===\n" + user.strip())
    if memory.strip():
        parts.append("=== PROJECT INSIGHTS ===\n" + memory.strip())
    return "\n\n".join(parts) if parts else "No profile or insights available yet."


def _do_alias(surface: str, canonical: str) -> str:
    _get_hy().register_alias(surface, canonical)
    return f"alias registered: {surface!r} → {canonical!r}"


def _do_retract(subject: str, predicate: str, object: str) -> str:
    ok = _get_hy().retract_edge(subject, predicate, object)
    return "retracted" if ok else "no matching active edge found"


# ── MCP tool registration ─────────────────────────────────────────────────────

def hymem_capture(session_id: str, messages: str, dream: bool = True) -> str:
    """Log a full conversation and optionally run dreaming. Preferred over hymem_log.

    Call this ONCE at the end of every conversation instead of calling hymem_log
    after each individual turn. This is far more reliable because it requires only
    a single tool call per session rather than one per exchange.

    Arguments:
        session_id  — unique id for this conversation, e.g. "2026-05-10-db-migration"
        messages    — JSON array of {role, content} objects representing the full
                      conversation in order, e.g.:
                      '[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]'
        dream       — if true (default), run a dreaming cycle immediately after
                      logging so MEMORY.md and USER.md are updated right away.

    Returns a summary of what was logged and, if dream=true, what was extracted.
    """
    return _do_capture(session_id, messages, dream)


def hymem_log(session_id: str, role: str, content: str) -> str:
    """Log one conversational turn to HyMem.

    Call this after every user message and every assistant reply, using the same
    session_id throughout a conversation (e.g. today's date + a short topic slug).

    role must be one of: user, assistant, system, tool.
    """
    return _do_log(session_id, role, content)


def hymem_dream() -> str:
    """Run a full dreaming cycle.

    Processes all unprocessed session chunks: extracts knowledge triples and
    behavioural markers, updates ~/.hermes/MEMORY.md and ~/.hermes/USER.md,
    then decays stale graph edges. Call at the end of a session or when idle.

    Safe to call concurrently — a run-lock prevents overlapping cycles.
    Returns a short report of what was processed.
    """
    return _do_dream()


def hymem_augment(message: str) -> str:
    """Return structured knowledge and relevant past context for a user message.

    Performs a dictionary-based entity match against the knowledge graph and a
    BM25 keyword search over past conversation chunks. No LLM call is made.
    Returns an empty string if no relevant context exists yet.
    """
    return _do_augment(message)


def hymem_profile() -> str:
    """Return the user's behavioral profile and project insights.

    Read USER.md (behavioral profile, auto-generated by HyMem) and MEMORY.md
    (project insights, auto-generated by HyMem) and return their combined
    content as a single labeled string. Use this once at session start to
    understand the user's preferences and the project's known structure
    before responding. For per-message context (relevant past chunks and
    graph facts), use hymem_augment instead.
    """
    return _do_profile()


def hymem_alias(surface: str, canonical: str) -> str:
    """Register that two names refer to the same entity.

    Example: hymem_alias('Postgres', 'postgresql') ensures that future mentions
    of 'Postgres' resolve to the same graph node as 'PostgreSQL' and 'postgresql'.
    """
    return _do_alias(surface, canonical)


def hymem_retract(subject: str, predicate: str, object: str) -> str:
    """Retract a knowledge graph edge that was wrongly extracted.

    Use this when you (or the user) realize HyMem extracted a relationship
    that's incorrect — e.g., the LLM hallucinated a dependency. Predicate must
    be one of: uses, depends_on, prefers, rejects, avoids, replaces,
    conflicts_with, deploys_to, part_of, equivalent_to.
    """
    return _do_retract(subject, predicate, object)


def main() -> None:
    mcp_instance = _get_mcp()
    mcp_instance.tool()(hymem_capture)
    mcp_instance.tool()(hymem_log)
    mcp_instance.tool()(hymem_dream)
    mcp_instance.tool()(hymem_augment)
    mcp_instance.tool()(hymem_profile)
    mcp_instance.tool()(hymem_alias)
    mcp_instance.tool()(hymem_retract)
    mcp_instance.run()


if __name__ == "__main__":
    main()
