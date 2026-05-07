"""MCP server for HyMem.

Exposes four tools to the Hermes Agent platform:
  hymem_log      — log one conversational turn
  hymem_dream    — run a dreaming cycle (extract, consolidate, decay)
  hymem_augment  — retrieve graph facts + FTS context for a user message
  hymem_alias    — register a surface-form alias for an entity

Run via the installed entry point:
    hymem-server

Or directly:
    python -m hymem.server

Configuration is entirely through environment variables (see README or
hymem/contrib/openai_client.py for the full list).

Key variables:
    HYMEM_LLM_API_KEY   API key for the extraction LLM (or DEEPSEEK_API_KEY)
    HYMEM_LLM_BASE_URL  Base URL (default: https://api.deepseek.com)
    HYMEM_LLM_MODEL     Model name (default: deepseek-chat)
    HYMEM_ROOT          Directory for hymem.sqlite, MEMORY.md, USER.md
                        (default: ~/.hermes)
"""
from __future__ import annotations

import os
from pathlib import Path

from hymem import HyMem, HyMemConfig
from hymem.contrib.openai_client import OpenAICompatibleClient


def _build() -> HyMem:
    root = Path(os.environ.get("HYMEM_ROOT", Path.home() / ".hermes"))
    return HyMem(
        HyMemConfig(root=root),
        llm=OpenAICompatibleClient(),
    )


# Module-level singleton. Initialised once when the server process starts.
_hy = _build()


def _get_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "mcp package required: pip install 'hymem[server]'"
        ) from e
    return FastMCP("hymem")


mcp = _get_mcp()


@mcp.tool()
def hymem_log(session_id: str, role: str, content: str) -> str:
    """Log one conversational turn to HyMem.

    Call this after every user message and every assistant reply, using the same
    session_id throughout a conversation (e.g. today's date + a short topic slug).

    role must be one of: user, assistant, system, tool.
    """
    _hy.log_message(session_id, role, content)
    return "logged"


@mcp.tool()
def hymem_dream() -> str:
    """Run a full dreaming cycle.

    Processes all unprocessed session chunks: extracts knowledge triples and
    behavioural markers, updates ~/.hermes/MEMORY.md and ~/.hermes/USER.md,
    then decays stale graph edges. Call at the end of a session or when idle.

    Safe to call concurrently — a run-lock prevents overlapping cycles.
    Returns a short report of what was processed.
    """
    report = _hy.dream()
    if report.skipped_locked:
        return "skipped: another dreaming cycle is already running"
    return (
        f"dreaming complete — "
        f"{report.sessions_processed} sessions, "
        f"{report.chunks_processed}/{report.chunks_seen} chunks processed, "
        f"{report.triples_extracted} triples, "
        f"{report.markers_extracted} markers extracted"
    )


@mcp.tool()
def hymem_augment(message: str) -> str:
    """Return structured knowledge and relevant past context for a user message.

    Performs a dictionary-based entity match against the knowledge graph and a
    BM25 keyword search over past conversation chunks. No LLM call is made.
    Returns an empty string if no relevant context exists yet.
    """
    ctx = _hy.augment(message)
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


@mcp.tool()
def hymem_alias(surface: str, canonical: str) -> str:
    """Register that two names refer to the same entity.

    Example: hymem_alias('Postgres', 'postgresql') ensures that future mentions
    of 'Postgres' resolve to the same graph node as 'PostgreSQL' and 'postgresql'.
    """
    _hy.register_alias(surface, canonical)
    return f"alias registered: {surface!r} → {canonical!r}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
