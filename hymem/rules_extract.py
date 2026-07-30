"""Idea B write-side — LLM durability tagger for marker → rule routing.

The deterministic lexical classifier (`rules.rule_scope_for_marker`) capped at
~14% precision on real dream markers (2026-07-27): standing-vs-one-off is a
SEMANTIC distinction a word list can't make. "User rejects the LOWER() patch"
(one-off) and "User rejects MongoDB" (standing) are lexically identical, and
one-off corrections carry incidental modals ("X was Dutch *instead* of English").

This module is the semantic instrument. It asks the LLM the one question the
regex can't: **is this marker a standing behavioral rule the assistant must obey
on EVERY future turn, or a one-off fact / event / decision?** It reuses the dream
LLM (pluggable `LLMClient`) in bounded sub-batches (~20 markers/call — a single
mega-batch makes the judge collapse to all-non-standing), so a durability pass is
a handful of calls per dream, NOT one per marker. It also returns a CANONICAL
imperative form ("Never use
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
from dataclasses import dataclass, replace

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.rules import is_rule_eligible_kind, rule_scope_for_marker

log = logging.getLogger("hymem.rules_extract")

# Bump when the durability prompt wording changes materially, so a re-run is
# distinguishable in experiment logs (mirrors PROFILE_PROMPT_VERSION).
# v2 (2026-07-28): hammer the exact output schema after deepseek-v4-flash was
# observed returning {"verdict": "STANDING RULE"} instead of {"standing": true}.
DURABILITY_PROMPT_VERSION = 2

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
    rationale: str = ""


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
Output a strict JSON array with ONE object per input marker, in index order. No prose, no markdown, no code fences. Each object has EXACTLY these keys:
  {"index": <int>, "standing": <boolean>, "confidence": <number 0.0-1.0>, "rule": <string or null>}
- "standing" MUST be a JSON boolean, true or false. Do NOT return a string, and do NOT use a "verdict" key — the key is "standing" and the value is true (durable, generalizing directive) or false (one-off).
- "confidence": how clearly the statement is a STANDING directive (1.0 = unmistakable; lower when specific/hedged/ambiguous).
- "rule": when standing, a short canonical imperative to obey ("Never use MongoDB", "Always run the tests before pushing"); reuse identical wording for equivalent statements so duplicates collapse. When not standing, null.

Example input:  [{"index":0,"kind":"rejection","statement":"the user rejects MongoDB"},{"index":1,"kind":"correction","statement":"the meeting is Tuesday not Monday"}]
Example output: [{"index":0,"standing":true,"confidence":0.95,"rule":"Never use MongoDB"},{"index":1,"standing":false,"confidence":0.1,"rule":null}]"""


DURABILITY_USER_TEMPLATE = """Markers:
{markers_json}

Return the JSON array of verdicts now."""


def _coerce_conf(v: object) -> float:
    try:
        c = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if c < 0 else 1.0 if c > 1 else c


_STANDING_TRUE = frozenset({"true", "yes", "standing", "standing rule",
                            "standing_rule", "rule", "1"})
_STANDING_FALSE = frozenset({"false", "no", "one-off", "one off", "one_off",
                             "oneoff", "not standing", "not_standing", "0",
                             "none", "null"})


def _parse_standing(item: dict) -> bool | None:
    """Extract the standing verdict, tolerant of schema drift. Despite the prompt
    demanding ``"standing": true``, real judges (deepseek-v4-flash) were observed
    returning ``"verdict": "STANDING RULE"``; silently dropping a *correct*
    classification is a false-negative bug, not precision safety. Accepts a
    boolean/int ``standing`` or a string verdict under several keys. Returns None
    only when no verdict signal is present at all."""
    v = item.get("standing")
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    for key in ("standing", "verdict", "label", "class", "classification", "decision"):
        s = item.get(key)
        if not isinstance(s, str):
            continue
        t = s.strip().lower()
        if t in _STANDING_TRUE:
            return True
        if t in _STANDING_FALSE:
            return False
        if "standing" in t:
            return "not" not in t and "one" not in t
        if "one" in t and "off" in t:
            return False
    return None


def _parse_batch(raw: str, n: int) -> list[DurabilityJudgment]:
    """Parse the LLM array into n index-aligned judgments. A genuinely
    unparseable / signal-less entry keeps the NON-routing default (standing=False,
    conf=0) — precision-safe: garbage never mints a rule. But a recognizable
    verdict is honored even under schema drift (see `_parse_standing`), and a
    standing verdict with no confidence or no canonical rewrite still routes: an
    absent confidence on a crisp verdict defaults high, and the raw statement is
    the fallback rule text downstream."""
    out = [DurabilityJudgment(standing=False, confidence=0.0, rule=None, index=i) for i in range(n)]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return out
    if not isinstance(data, list):
        return out
    for pos, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        raw_idx = item.get("index")
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            idx = pos  # ordinal fallback when the model omits "index"
        if not (0 <= idx < n):
            continue
        standing = _parse_standing(item)
        if standing is None:
            continue  # no verdict signal → keep the non-routing default
        raw_rule = item.get("rule")
        rule = raw_rule.strip() if isinstance(raw_rule, str) and raw_rule.strip() else None
        conf = item.get("confidence")
        confidence = _coerce_conf(conf) if conf is not None else (1.0 if standing else 0.0)
        reason = item.get("reason") or item.get("rationale") or item.get("why")
        rationale = reason.strip() if isinstance(reason, str) else ""
        out[idx] = DurabilityJudgment(
            standing=standing,
            confidence=confidence,
            rule=rule if standing else None,
            index=idx,
            rationale=rationale,
        )
    return out


# Markers per durability call. A real judge (deepseek-v4-flash) is accurate on a
# small batch but COLLAPSES to all-non-standing on a large one (observed: 100% at
# 10 markers, 0% at 111) — a mix of attention degradation and output truncation.
# Keep each call small and size its token budget to the slice.
_DURABILITY_BATCH_SIZE = 20


def _judge_one_batch(
    llm: LLMClient, sub: list[tuple[str, str]], system: str = DURABILITY_SYSTEM
) -> list[DurabilityJudgment]:
    """Single durability call over a small slice, index-aligned to ``sub``. Any
    LLM/parse failure degrades ONLY this slice to non-standing (precision-safe).
    ``system`` lets a caller (e.g. the adjudicator) swap the prompt while keeping
    the sub-batching, token sizing, and robust parse."""
    payload = [
        {"index": i, "kind": kind, "statement": statement}
        for i, (kind, statement) in enumerate(sub)
    ]
    request = LLMRequest(
        system=system,
        user=DURABILITY_USER_TEMPLATE.format(markers_json=json.dumps(payload, ensure_ascii=False)),
        response_format="json",
        max_tokens=min(4096, 128 * len(sub) + 256),  # sized so the array never truncates
    )
    try:
        raw = llm.complete(request)
    except Exception as exc:  # noqa: BLE001 - a bad tag pass must never break dreaming
        log.warning("rules_extract.durability_call_failed n=%d err=%s", len(sub), exc)
        return [DurabilityJudgment(standing=False, confidence=0.0, rule=None, index=i)
                for i in range(len(sub))]
    return _parse_batch(raw, len(sub))


def judge_durability_batch(
    llm: LLMClient,
    markers: list[tuple[str, str]],
    *,
    batch_size: int = _DURABILITY_BATCH_SIZE,
    system: str = DURABILITY_SYSTEM,
) -> list[DurabilityJudgment]:
    """Durability verdicts for (kind, statement) markers, in input order.

    Splits into sub-batches of ``batch_size`` (default 20) — one call each —
    because a single mega-batch makes the judge collapse to all-non-standing
    (100% at 10, 0% at 111). Empty input → no call. A failed sub-batch degrades
    only its own slice. The returned ``index`` is the GLOBAL position. ``system``
    overrides the prompt (the adjudicator reuses this with its blind prompt)."""
    if not markers:
        return []
    bs = max(1, batch_size)
    out: list[DurabilityJudgment] = []
    for start in range(0, len(markers), bs):
        sub = markers[start:start + bs]
        for k, j in enumerate(_judge_one_batch(llm, sub, system)):
            out.append(replace(j, index=start + k))
    return out


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
    batch_size: int = _DURABILITY_BATCH_SIZE,
) -> list[RouteDecision]:
    """Compute per-marker routing decisions for a batch under the given mode.

    Pure/deterministic for ``lexical``; the ``llm`` / ``llm_fastpath`` arms issue
    sub-batched durability calls (``batch_size`` markers each). Index-aligned to
    ``markers``."""
    if mode not in RULES_EXTRACTION_MODES:
        raise ValueError(f"unknown rules_extraction_mode {mode!r} (expected {sorted(RULES_EXTRACTION_MODES)})")

    n = len(markers)
    # Kind eligibility is ORTHOGONAL to durability and gates BOTH paths. The
    # lexical classifier already applies it (`rule_scope_for_marker` → None for an
    # ineligible kind); the LLM path must apply it too, BEFORE the durability call,
    # or `preference` markers flood the tagger and mint hundreds of false rules
    # (box full run: 1,768 preferences → 1,476 FPs). Gating here (not inside the
    # tagger) keeps the two paths' scope identical — they can never silently
    # diverge again — and preferences never cost an API call.
    eligible = [is_rule_eligible_kind(kind) for kind, _ in markers]
    lexical = [rule_scope_for_marker(kind, stmt) for kind, stmt in markers]

    if mode == "lexical":
        return [
            RouteDecision(route=scope is not None, text=stmt, scope=scope or "always_on",
                          confidence=1.0, source_mode="lexical")
            for scope, (_, stmt) in zip(lexical, markers)
        ]

    # Which markers need the LLM: eligible kinds only, and for llm_fastpath a
    # lexical modal is trusted as standing (confidence 1.0, no call) so only the
    # ambiguous eligible rest are judged.
    need_llm = [
        i for i in range(n)
        if eligible[i] and not (mode == "llm_fastpath" and lexical[i] is not None)
    ]
    judged: dict[int, DurabilityJudgment] = {}
    if need_llm and llm is not None:
        sub = judge_durability_batch(llm, [markers[i] for i in need_llm], batch_size=batch_size)
        judged = {need_llm[j]: sub[j] for j in range(len(sub))}

    out: list[RouteDecision] = []
    for i, (_, stmt) in enumerate(markers):
        if not eligible[i]:  # ineligible kind (e.g. preference) never mints a rule
            out.append(RouteDecision(route=False, text=stmt, source_mode=mode))
            continue
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
