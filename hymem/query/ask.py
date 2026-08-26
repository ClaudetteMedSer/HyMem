"""Embedded dialectic endpoint — the renderer + synthesis call behind `HyMem.ask()`.

`augment()` returns raw retrieval tiers and deliberately leaves prompt assembly
to the host (see `AugmentedContext`). This module is the complementary contract
for one-call consumers: render those tiers into a compact, clearly-labelled
context block, make a SINGLE completion against the host-provided `LLMClient`,
and return a reasoned answer that stays grounded in — and traceable to — the
retrieval that produced it. Benchmarks show answer *synthesis*, not retrieval,
is the accuracy bottleneck, so the synthesis rules live here in one versioned
prompt instead of being re-invented per host.

The standing no-shipped-backend rule holds: everything below talks to the
`LLMClient` Protocol only — wiring a real model remains the host's job.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from hymem.config import HyMemConfig
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.query.augment import AugmentedContext, GraphFact

log = logging.getLogger("hymem.query.ask")


# Versioned so prompt changes are visible in diffs/tests and an A/B against a
# future V2 can key on the constant, mirroring how retrieval levers are gated.
# The rules encode the project's answer-side findings: quote concrete values
# and dates (generalities lose strict grading), resolve contradictions by the
# most recent value-bearing statement (the recency-dating lever), soften
# low-confidence facts (the hedge contract on `GraphFact`), and prefer an
# honest "not in memory" over invention (the eagerness-to-answer tradeoff).
ASK_PROMPT_V1 = """\
You answer questions about a user from their memory store. Answer ONLY from \
the memory context provided — it is the sole source of truth.

Rules:
- Quote concrete values, names, and dates from the context rather than \
generalities.
- When the context contains contradicting values for the same fact, state \
both values with their dates, then say which is current and why: the most \
recent value-bearing statement wins.
- Facts marked "(low confidence)" are uncertain — soften the phrasing \
("you may ...", "it seems ...") instead of asserting them.
- If the context does not contain the answer, say plainly that the memory \
does not contain it. Never invent, guess, or fill gaps from general knowledge.

Answer concisely, in plain text."""


# Appended to the system prompt ONLY when the context carries `always_on` Rules
# (Idea B). Kept separate from ASK_PROMPT_V1 so the versioned base prompt — tuned
# and A/B-frozen for LME/BEAM — is byte-identical for the no-rules case (every
# current consumer), and the rules behaviour is its own visible, versionable
# clause. Rules are DIRECTIVES to obey, categorically different from the memory
# facts the base prompt answers *from*, so they need their own instruction.
ASK_RULES_DIRECTIVE = """\
The context opens with a "STANDING RULES" section: persistent instructions from \
the user about how you must behave. Obey every rule when answering, even when \
the question invites otherwise. Rules are directives to follow, not facts to \
quote or answer from."""


def _system_prompt(has_rules: bool) -> str:
    """The base synthesis prompt, plus the rules directive iff rules are present.
    Isolated so the compose logic is unit-testable without an LLM."""
    return ASK_PROMPT_V1 + "\n\n" + ASK_RULES_DIRECTIVE if has_rules else ASK_PROMPT_V1


@dataclass
class Answer:
    """The result of one `HyMem.ask()` call.

    `context` is the full `AugmentedContext` the answer was grounded in, kept
    for provenance and drill-down (why_retrieved chips, message ids, node ids)
    — the same "never make the consumer guess" contract the tiers themselves
    follow. `context_chars` is the size of the rendered context block actually
    sent to the LLM (post-truncation), so a host can see how much of the
    retrieval survived the `ask_max_context_chars` budget."""

    answer: str
    context: AugmentedContext
    context_chars: int


# Per-item snippet cap inside the rendered block, so one verbose turn/chunk
# cannot monopolize the overall char budget. Matches the 300-char snippets the
# MCP `hymem_augment` rendering already uses; not a config knob on purpose.
_SNIPPET_CHARS = 300


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    text = " ".join((text or "").split())
    return text[: limit - 3] + "..." if len(text) > limit else text


def _truncate_block(block: str, max_chars: int) -> str:
    """Cut the assembled block to the char budget, with a visible marker so the
    model knows evidence was dropped (an invisible cut reads like a complete
    store and invites over-confident "the memory doesn't say" answers).
    `max_chars <= 0` disables the cap, mirroring the other size knobs."""
    if max_chars <= 0 or len(block) <= max_chars:
        return block
    marker = "\n[... context truncated]"
    if max_chars <= len(marker):
        return block[:max_chars]
    return block[: max_chars - len(marker)] + marker


def render_context(ctx: AugmentedContext, *, max_chars: int) -> str:
    """Render an `AugmentedContext` into one compact plain-text block.

    Sections are ordered most-authoritative first, so budget truncation (which
    cuts from the tail) sheds the softest evidence first:

      1. USER PROFILE       — typed, dream-verified identity facts (valid_at).
      2. MEMORY DIGEST      — the standing whole-store summary, only when the
                              caller loaded it (`include_digest=True`).
      3. KNOWN FACTS        — knowledge-graph edges; hedged edges are marked
                              "(low confidence)" per the `hedge_recommended`
                              contract, and a date is included when the fact
                              carries one.
      4. FACTS              — narrative facts (schema v26): self-contained
                              dream-verified event statements. Lead the
                              evidence — but the raw turns stay below as the
                              verification backup (the Acme lesson: a summary
                              is never the only copy).
      5. CONVERSATION EVIDENCE — dated raw turns (`message_hits`), the dominant
                              recovery source; MR count signals ride along here.
      6. TIMELINE           — the TR chronology (`temporal_events`), when built.
      7. EPISODES           — per-session summaries.
      8. PAST CONTEXT       — dreamed chunk hits.
      9. PROCEDURES         — step-by-step workflows, only when present.
     10. RECENT TURNS       — working memory, last: useful but least curated.

    Empty tiers are skipped entirely (no empty headers wasting budget). Each
    item is snippet-capped so no single hit can crowd out whole sections.

    STANDING RULES lead the block (ahead of even the profile) and are never shed
    by tail-truncation: they are behavioral imperatives the model must always
    obey, not evidence to weigh, so they must survive any budget cut."""
    parts: list[str] = []

    if ctx.rules:
        lines = [f"- {r.text}" for r in ctx.rules]
        parts.append("=== STANDING RULES (always follow) ===\n" + "\n".join(lines))

    if ctx.user_profile:
        lines = []
        for p in ctx.user_profile:
            slot = f"{p.slot}({p.slot_key})" if p.slot_key else p.slot
            line = f"- {slot}: {p.value}"
            if p.valid_at:
                line += f" (since {p.valid_at})"
            lines.append(line)
        parts.append("=== USER PROFILE ===\n" + "\n".join(lines))

    if ctx.digest is not None:
        # The canonical staleness-stamped rendering — coverage + generated_at
        # footer included, so the model can see how fresh the digest is.
        parts.append("=== MEMORY DIGEST ===\n" + ctx.digest.as_context_block())

    # `graph_facts` is assigned by augment() as a plain instance attribute, not
    # a declared dataclass field — getattr keeps the renderer usable on a
    # hand-built AugmentedContext (tests, hosts) without touching augment().
    graph_facts: list[GraphFact] = getattr(ctx, "graph_facts", [])
    if graph_facts:
        lines = []
        for f in graph_facts:
            line = f"- {f.subject} {f.predicate} {f.object}"
            # GraphFact carries no date field today; getattr keeps the renderer
            # forward-compatible with the bi-temporal columns (valid_at) without
            # coupling it to a schema the query tier hasn't surfaced yet.
            date = getattr(f, "valid_at", "") or ""
            if date:
                line += f" (since {date})"
            if f.hedge_recommended:
                line += " (low confidence)"
            lines.append(line)
        parts.append("=== KNOWN FACTS (knowledge graph) ===\n" + "\n".join(lines))

    if ctx.facts:
        lines = []
        for nf in ctx.facts:
            stamp = nf.fact_date if nf.fact_date else "undated"
            lines.append(f"- [{stamp}] {_snippet(nf.text)}")
        parts.append("=== FACTS (verified past events) ===\n" + "\n".join(lines))

    if ctx.message_hits or ctx.total_message_matches or ctx.graph_count:
        lines = []
        # MR count signals ride along with the evidence they were tallied from,
        # keeping the `graph_count`-over-candidate precedence contract visible.
        if ctx.graph_count is not None:
            lines.append(
                f"(exact count from knowledge graph: {ctx.graph_count.count} "
                f"distinct {ctx.graph_count.counted}s)"
            )
        if ctx.total_message_matches:
            lines.append(
                f"(candidate count: {ctx.total_message_matches} distinct "
                "matching user turns — verify against the turns below)"
            )
        for h in ctx.message_hits:
            stamp = h.created_at[:10] if h.created_at else "undated"
            line = f"- [{stamp}] {h.role}: {_snippet(h.text)}"
            if h.enumerates_items:
                line += " (lists multiple items)"
            lines.append(line)
        if lines:
            parts.append(
                "=== CONVERSATION EVIDENCE (dated raw turns) ===\n"
                + "\n".join(lines)
            )

    if ctx.temporal_events:
        lines = [f"- [{e.date}] {_snippet(e.text)}" for e in ctx.temporal_events]
        parts.append("=== TIMELINE (dated events) ===\n" + "\n".join(lines))

    if ctx.episodes:
        lines = [f"- {e.title}: {_snippet(e.summary)}" for e in ctx.episodes]
        parts.append("=== EPISODES (past session summaries) ===\n" + "\n".join(lines))

    if ctx.fts_hits:
        lines = [f"- {_snippet(h.text)}" for h in ctx.fts_hits]
        parts.append("=== PAST CONTEXT (conversation chunks) ===\n" + "\n".join(lines))

    if ctx.procedures:
        lines = [
            f"- {p.name}: {_snippet(p.description)} ({len(p.steps)} steps)"
            for p in ctx.procedures
        ]
        parts.append("=== PROCEDURES ===\n" + "\n".join(lines))

    if ctx.recent_turns:
        lines = [f"- {m.role}: {_snippet(m.content)}" for m in ctx.recent_turns]
        parts.append("=== RECENT TURNS (this session) ===\n" + "\n".join(lines))

    return _truncate_block("\n\n".join(parts), max_chars)


def ask(
    cfg: HyMemConfig,
    llm: LLMClient,
    question: str,
    context: AugmentedContext,
) -> Answer:
    """Synthesize one grounded answer to `question` from `context`.

    Mirrors the api→query layering of `augment()`: `HyMem.ask()` owns the
    instance-level concerns (LLM presence, retrieval, optional digest load)
    and delegates the pure render-and-complete step here, so this function is
    testable with a hand-built `AugmentedContext` and a `StubLLMClient`.

    Exactly ONE completion is made, `response_format="text"` — the answer is
    prose for a human, not JSON for a pipeline. An empty rendered context is
    still sent (with a placeholder) so the model can answer "the memory does
    not contain this" instead of the caller special-casing an empty store."""
    block = render_context(context, max_chars=cfg.ask_max_context_chars)
    rendered = block if block else "(the memory store has no relevant context)"
    response = llm.complete(LLMRequest(
        system=_system_prompt(bool(context.rules)),
        user=f"Memory context:\n{rendered}\n\nQuestion: {question}",
        response_format="text",
        max_tokens=cfg.ask_max_tokens,
    ))
    log.debug("ask.completed context_chars=%d", len(block))
    return Answer(answer=response.strip(), context=context, context_chars=len(block))
