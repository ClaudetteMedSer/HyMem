"""Idea B write-side — LLM durability tagger for marker → rule routing.

The deterministic lexical classifier (`rules.rule_scope_for_marker`) capped at
~14% precision on real dream markers (2026-07-27): standing-vs-one-off is a
SEMANTIC distinction a word list can't make. "User rejects the LOWER() patch"
(one-off) and "User rejects MongoDB" (standing) are lexically identical, and
one-off corrections carry incidental modals ("X was Dutch *instead* of English").

This module is the semantic instrument. It asks the LLM the one question the
regex can't: **is this marker a standing behavioral rule the assistant must obey
on EVERY future turn, or a one-off fact / event / decision?** It reuses the dream
LLM (pluggable `LLMClient`), so a durability pass is one batched call per dream —
NOT one per marker. It also returns a CANONICAL imperative form ("Never use
MongoDB"), which normalizes paraphrases ("rejects Mongo", "avoid MongoDB") to a
single `rules.text`, so `add_rule`'s UNIQUE UPSERT accumulates `pos_evidence`
across sessions — the recurrence signal that repetition-gated promotion needs.

Three routing modes (config `rules_extraction_mode`), all behind the write-side
`rules_extraction_enabled` gate and evaluated head-to-head by
`benchmarks/rule_extraction_experiment.py`:

  - ``lexical``        — the deterministic classifier only (the ~14% baseline).
  - ``llm``            — the durability tag decides; route iff standing and
                          confidence ≥ ``rules_extraction_confidence_min``.
  - ``llm_fastpath``   — a lexical imperative modal is trusted as standing
                          (no call); everything else is sent to the LLM. Cheapest
                          LLM arm — only the ambiguous markers cost tokens.

Nothing here promotes a rule on its own; it returns a routing decision that
`rules.route_markers_to_rules` persists via `add_rule`. Repetition-gated
promotion (option A) is simulated in the experiment harness and only wired to the
live schema once the data says it beats immediate promotion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.rules import rule_scope_for_marker

log = logging.getLogger("hymem.rules_extract")

# Bump when the durability prompt wording changes materially, so a re-run is
# distinguishable in experiment logs (mirrors PROFILE_PROMPT_VERSION).
DURABILITY_PROMPT_VERSION = 1

RULES_EXTRACTION_MODES = frozenset({"lexical", "llm", "llm_fastpath"})


@dataclass(frozen=True)
class DurabilityJudgment:
    """One marker's durability verdict.

    ``standing`` — is this a rule to obey on every future turn (True) or a
    one-off fact/event/decision (False)? ``confidence`` ∈ [0,1] is how clearly
    the statement is a *standing* directive; the router gates on it. ``rule`` is
    the canonical imperative to store when standing (paraphrase-normalized), or
    None. ``index`` aligns a batched verdict back to its input marker.
    """

    standing: bool
    confidence: float
    rule: str | None = None
    index: int = -1


DURABILITY_SYSTEM = """You decide whether each behavioral signal from a user is a STANDING RULE or a ONE-OFF.

A STANDING RULE is a durable instruction the assistant must follow on EVERY future turn, indefinitely:
  - "never suggest Docker", "always run the tests before pushing"
  - "write commit messages in the imperative mood", "respond concisely"
  - "avoid global mutable state", "do not add a dependency without asking first"
A standing rule generalizes to future behavior and is NOT tied to one specific artifact, moment, or decision.

A ONE-OFF is a decision, correction, or rejection about ONE specific thing, event, or moment. It must NOT become a rule, because a rule injects into every future prompt:
  - "rejects the LOWER() patch for HyMem"  (a specific code change)
  - "rejects LoCoMo as a benchmark"        (a specific project decision)
  - "the meeting is Tuesday, not Monday"   (a specific fact fix)
  - "the podcast was in Dutch instead of English"  (a specific observation)

The word "rejects"/"refuses"/"instead"/"should" does NOT make something standing — one-offs use those words too. Judge the SUBSTANCE: would obeying this on every future turn make sense, or only in the one situation it describes?

Input is a JSON array of numbered markers: [{"index": 0, "kind": "...", "statement": "..."}, ...].
Output a strict JSON array, one object per input index, no prose, no markdown:
  {"index": 0, "standing": true, "confidence": 0.0-1.0, "rule": "canonical imperative"}
- "standing": true only for durable, generalizing directives; false for one-offs.
- "confidence": how clearly the statement is a STANDING directive (1.0 = unmistakably standing; lower when specific/hedged/ambiguous).
- "rule": when standing, rewrite it as a short canonical imperative the assistant can obey ("Never use MongoDB", "Always run the tests before pushing"). Use the SAME canonical wording for equivalent statements so duplicates collapse. When not standing, use null.
Return exactly one object per input marker, in index order."""


DURABILITY_USER_TEMPLATE = """Markers:
{markers_json}

Return the JSON array of verdicts now."""


def _coerce_conf(v: object) -> float:
    try:
        c = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if c < 0 else 1.0 if c > 1 else c


def _parse_batch(raw: str, n: int) -> list[DurabilityJudgment]:
    """Parse the LLM array into n index-aligned judgments. A missing / malformed
    entry defaults to a NON-routing verdict (standing=False, conf=0) — the
    precision-safe failure mode: a dropped verdict never mints a rule."""
    out = [DurabilityJudgment(standing=False, confidence=0.0, rule=None, index=i) for i in range(n)]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return out
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < n):
            continue
        standing = bool(item.get("standing"))
        rule = item.get("rule")
        rule = rule.strip() if isinstance(rule, str) and rule.strip() else None
        out[idx] = DurabilityJudgment(
            standing=standing and rule is not None,
            confidence=_coerce_conf(item.get("confidence")),
            rule=rule if standing else None,
            index=idx,
        )
    return out


def judge_durability_batch(
    llm: LLMClient, markers: list[tuple[str, str]]
) -> list[DurabilityJudgment]:
    """One batched durability call for a list of (kind, statement) markers.

    Index-aligned to the input. Empty input → no call. Any LLM/parse failure
    degrades to all-non-standing (precision-safe: never mints on error)."""
    if not markers:
        return []
    payload = [
        {"index": i, "kind": kind, "statement": statement}
        for i, (kind, statement) in enumerate(markers)
    ]
    request = LLMRequest(
        system=DURABILITY_SYSTEM,
        user=DURABILITY_USER_TEMPLATE.format(markers_json=json.dumps(payload, ensure_ascii=False)),
        response_format="json",
        max_tokens=2048,
    )
    try:
        raw = llm.complete(request)
    except Exception as exc:  # noqa: BLE001 - a bad tag pass must never break dreaming
        log.warning("rules_extract.durability_call_failed err=%s", exc)
        return [
            DurabilityJudgment(standing=False, confidence=0.0, rule=None, index=i)
            for i in range(len(markers))
        ]
    return _parse_batch(raw, len(markers))


@dataclass(frozen=True)
class RouteDecision:
    """A routing verdict for one marker: whether to mint a rule, and its text.

    ``text`` is the canonical rule to persist (the LLM's rewrite when available,
    else the raw statement). ``scope`` is always ``always_on`` for agent-inferred
    rules (they carry no trigger entities)."""

    route: bool
    text: str
    scope: str = "always_on"
    confidence: float = 1.0
    source_mode: str = "lexical"


def route_decisions(
    markers: list[tuple[str, str]],
    *,
    mode: str,
    llm: LLMClient | None,
    confidence_min: float,
) -> list[RouteDecision]:
    """Compute per-marker routing decisions for a batch under the given mode.

    Pure/deterministic for ``lexical``; the ``llm`` / ``llm_fastpath`` arms issue
    at most ONE batched durability call. Index-aligned to ``markers``."""
    if mode not in RULES_EXTRACTION_MODES:
        raise ValueError(f"unknown rules_extraction_mode {mode!r} (expected {sorted(RULES_EXTRACTION_MODES)})")

    n = len(markers)
    lexical = [rule_scope_for_marker(kind, stmt) for kind, stmt in markers]

    if mode == "lexical":
        return [
            RouteDecision(route=scope is not None, text=stmt, scope=scope or "always_on",
                          confidence=1.0, source_mode="lexical")
            for scope, (_, stmt) in zip(lexical, markers)
        ]

    # Which markers need the LLM. For llm_fastpath, a lexical modal is trusted as
    # standing (confidence 1.0, no call) and only the rest are judged.
    need_llm = [
        i for i in range(n)
        if not (mode == "llm_fastpath" and lexical[i] is not None)
    ]
    judged: dict[int, DurabilityJudgment] = {}
    if need_llm and llm is not None:
        sub = judge_durability_batch(llm, [markers[i] for i in need_llm])
        judged = {need_llm[j]: sub[j] for j in range(len(sub))}

    out: list[RouteDecision] = []
    for i, (_, stmt) in enumerate(markers):
        if mode == "llm_fastpath" and lexical[i] is not None:
            out.append(RouteDecision(route=True, text=stmt, scope="always_on",
                                     confidence=1.0, source_mode="lexical_fastpath"))
            continue
        j = judged.get(i)
        if j is None:  # no llm available → precision-safe drop
            out.append(RouteDecision(route=False, text=stmt, source_mode=mode))
            continue
        route = j.standing and j.confidence >= confidence_min
        out.append(RouteDecision(
            route=route,
            text=j.rule or stmt,
            scope="always_on",
            confidence=j.confidence,
            source_mode=mode,
        ))
    return out
