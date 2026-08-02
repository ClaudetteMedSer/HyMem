"""MCP server for HyMem.

Exposes eleven tools to the Hermes Agent platform:
  hymem_capture    — log a full conversation at once + optionally dream (preferred)
  hymem_log        — log one conversational turn (fallback for turn-by-turn use)
  hymem_dream      — run a dreaming cycle (extract, consolidate, decay)
  hymem_augment    — retrieve graph facts + FTS context for a user message
  hymem_ask        — ask the memory a question, get one LLM-reasoned answer
  hymem_profile    — return USER.md (behavioral profile) + MEMORY.md (project insights)
  hymem_digest     — return the standing whole-store memory digest (RAPTOR root)
  hymem_alias      — register a surface-form alias for an entity
  hymem_retract    — retract a wrongly extracted knowledge graph edge
  hymem_add_rule   — record a standing behavioral rule (always_on / contextual)
  hymem_list_rules — list the active standing rules
  hymem_suggest_rules — propose inferred standing rules to review (no auto-add)

Run via the installed entry point:
    hymem-server

Or directly:
    python -m hymem.server

Configuration is entirely through environment variables (see README or
hymem/contrib/openai_client.py for the full list).

Key variables:
    HYMEM_LLM_API_KEY        API key for the extraction LLM (or DEEPSEEK_API_KEY)
    HYMEM_LLM_BASE_URL       Base URL (default: https://api.deepseek.com)
    HYMEM_LLM_MODEL          Model name (default: deepseek-v4-flash)
    HYMEM_EMBEDDING_API_KEY  API key for embeddings (falls back to LLM key)
    HYMEM_EMBEDDING_BASE_URL Embedding endpoint (default: https://api.deepseek.com)
    HYMEM_EMBEDDING_MODEL    Embedding model (default: deepseek-embedding)
    HYMEM_ROOT               Directory for hymem.sqlite, MEMORY.md, USER.md
                             (default: ~/.hermes)
    HYMEM_AGGREGATION_NODES_ENABLED
                             Turn on the RAPTOR aggregation/digest layer at dream
                             time (default: off). Set true to gather steady-state
                             nodes/reused cost data before flipping the shipped
                             default (raptor_digest_plan.md 3c).
    HYMEM_AGGREGATION_DIGEST_ENABLED
                             Override the digest sub-switch independently (default:
                             on whenever aggregation is enabled). Set false to
                             measure level-0 node-build cost in isolation.

If the embedding client cannot be constructed (e.g. API key absent), the server
logs a warning and falls back to FTS-only retrieval — no other functionality
is affected.
"""
from __future__ import annotations

import json
import logging
import os

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

    if ctx.facts:
        lines = [
            f"- [{f.fact_date or 'undated'}] {f.text[:300]}"
            for f in ctx.facts
        ]
        parts.append("**Narrative facts (dream-verified events):**\n" + "\n".join(lines))

    if ctx.fts_hits:
        snippets = [f"[{h.session_id}] {h.text[:300]}" for h in ctx.fts_hits]
        parts.append("**Relevant past context (keyword search):**\n" + "\n".join(snippets))

    if ctx.message_hits:
        snippets = [f"[{h.session_id}/{h.role}] {h.text[:300]}" for h in ctx.message_hits]
        parts.append("**Relevant raw turns (keyword search):**\n" + "\n".join(snippets))

    return "\n\n".join(parts) if parts else ""


def _do_ask(question: str) -> str:
    return _get_hy().ask(question).answer


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


def _do_digest() -> str:
    digest = _get_hy().digest()
    if digest is None:
        return (
            "No digest available yet — it is built at dream time once the "
            "aggregation layer (aggregation_nodes_enabled + "
            "aggregation_digest_enabled) has dreamed over at least one episode."
        )
    return digest.as_context_block()


def _do_alias(surface: str, canonical: str) -> str:
    _get_hy().register_alias(surface, canonical)
    return f"alias registered: {surface!r} → {canonical!r}"


def _do_retract(subject: str, predicate: str, object: str) -> str:
    ok = _get_hy().retract_edge(subject, predicate, object)
    return "retracted" if ok else "no matching active edge found"


def _do_add_rule(text: str, scope: str = "always_on", trigger_entities: str = "") -> str:
    triggers = [t.strip() for t in trigger_entities.split(",") if t.strip()] or None
    try:
        rid = _get_hy().add_rule(text, scope=scope, trigger_entities=triggers, source="user")
    except ValueError as e:
        return f"error: {e}"
    return f"rule #{rid} added (scope={scope})"


def _do_list_rules() -> str:
    active = _get_hy().rules()
    if not active:
        return "No standing rules set."
    lines = []
    for r in active:
        tag = (r.scope if r.scope == "always_on"
               else f"contextual({', '.join(r.trigger_entities)})")
        lines.append(f"#{r.id} [{tag}] {r.text}")
    return "\n".join(lines)


def _do_suggest_rules(limit: int = 10) -> str:
    try:
        cands = _get_hy().suggest_rules(limit=limit)
    except RuntimeError as e:
        return f"error: {e}"
    if not cands:
        return ("No rule candidates (no recent markers cleared the durability "
                "tagger). Suggestions read UNCONSOLIDATED markers — call after "
                "logging a session and before dreaming.")
    lines = ["Candidate standing rules — NOT added yet; adopt with hymem_add_rule:"]
    for c in cands:
        prov = f"{c.marker_count} marker(s)/{c.session_count} session(s), conf {c.confidence:.2f}"
        dup = "  [already active — would reinforce]" if c.already_active else ""
        lines.append(f"- {c.text}  ({prov}; kinds={','.join(c.kinds)}){dup}")
    return "\n".join(lines)


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


def hymem_ask(question: str) -> str:
    """Ask the memory store a question and get one reasoned answer.

    The dialectic counterpart to hymem_augment: instead of returning raw
    retrieval context for YOU to interpret, this runs the same retrieval and
    makes a single LLM call that synthesizes a grounded answer — quoting
    concrete values and dates, stating both sides of a contradiction (most
    recent statement wins), hedging low-confidence facts, and saying plainly
    when the memory does not contain the answer. Use it for direct questions
    about the user or past sessions ("what database does the user prefer?");
    use hymem_augment when you want the raw evidence tiers instead.
    """
    return _do_ask(question)


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


def hymem_digest() -> str:
    """Return the standing whole-store memory digest.

    The digest is the root of HyMem's cross-session aggregation tree: one
    summary answering "what do you know about me?" across the entire store,
    rebuilt at dream time (never per query). Inject it as standing context at
    session start — it complements hymem_profile (behavioral profile) with a
    narrative of what the user has actually been working on. The footer states
    how many sessions it covers and when it was generated, so staleness is
    visible; re-fetch after hymem_dream to pick up a refreshed digest.
    Returns an explanatory message if no digest has been built yet.
    """
    return _do_digest()


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


def hymem_add_rule(text: str, scope: str = "always_on", trigger_entities: str = "") -> str:
    """Record a STANDING RULE — a behavioral instruction to always follow.

    Rules are imperatives about HOW to behave ("always run the tests before
    pushing", "never suggest Docker"), distinct from facts. An `always_on` rule
    (default) is injected into every future context call; a `contextual` rule
    fires only when one of its trigger entities is in play.

    Use this when the user TELLS you a standing preference or prohibition — not
    for one-off facts (those are logged as messages and extracted at dream time).

    Arguments:
        text             — the rule, phrased as a directive.
        scope            — "always_on" (default) or "contextual".
        trigger_entities — for scope="contextual" only: a comma-separated list
                           of entities that activate the rule (e.g. "redis,cache").
    """
    return _do_add_rule(text, scope, trigger_entities)


def hymem_list_rules() -> str:
    """List all active standing rules (with ids), so you can see what behavioral
    rules are in force before answering. Each line is `#id [scope] text`."""
    return _do_list_rules()


def hymem_suggest_rules(limit: int = 10) -> str:
    """Propose standing rules inferred from RECENT behavior, to review and confirm
    — this does NOT add anything. Auto-adding inferred rules is deliberately off:
    a rule injects into every future call, so a human/agent confirms first.

    Each candidate shows its corroboration — how many markers over how many
    distinct sessions support it (more sessions = more durable) — its confidence,
    and its source kinds; ones matching an active rule are flagged. To adopt one,
    call `hymem_add_rule` with its text; skip the rest. Reads unconsolidated
    markers, so call it after logging a session and before dreaming."""
    return _do_suggest_rules(limit)


def main() -> None:
    # Configure logging at the application entry point (never on import — that
    # would clobber a host's handlers). Without this, the dream runner's
    # log.info() lines — including "aggregate.built nodes=/reused=" and the
    # aggregate.build_failure exception — are silently dropped, leaving the
    # server with no operational visibility. Level is env-tunable.
    logging.basicConfig(
        level=os.environ.get("HYMEM_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp_instance = _get_mcp()
    mcp_instance.tool()(hymem_capture)
    mcp_instance.tool()(hymem_log)
    mcp_instance.tool()(hymem_dream)
    mcp_instance.tool()(hymem_augment)
    mcp_instance.tool()(hymem_ask)
    mcp_instance.tool()(hymem_profile)
    mcp_instance.tool()(hymem_digest)
    mcp_instance.tool()(hymem_alias)
    mcp_instance.tool()(hymem_retract)
    mcp_instance.tool()(hymem_add_rule)
    mcp_instance.tool()(hymem_list_rules)
    mcp_instance.tool()(hymem_suggest_rules)
    mcp_instance.run()


if __name__ == "__main__":
    main()
