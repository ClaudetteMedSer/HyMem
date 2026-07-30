#!/usr/bin/env python3
"""E1 front-run probe — G-F1, the gate that decides whether narrative facts get built.

Campaign E, Step 1 (`additional_planning.md` §Campaign E). The claim under test is
NOT "facts raise the score" — that is what Step 5 measures, after a build. It is
the two things a build can no longer measure once it exists:

  (a) **Density.** Does a narrative-fact tier put the question's gold inside
      ≤5 dense items, where the shipping pipeline needs ~45 raw turn slots?
      That is the whole thesis of the middle-granularity unit: the reader fails
      at fusing 45 fragments and succeeds at fusing pre-fused facts (P0 measured
      ~20% of the Hindsight gap as reader-side; the residual is evidence
      PACKAGING).
  (b) **Faithfulness.** Are the extracted facts traceable — every value, name and
      date present in an actual turn, nothing invented? A dense tier that
      hallucinates is worse than no tier, because `ask()` renders it ABOVE the
      raw turns.

Neither number is a benchmark score, and that is deliberate: review constraint 1
("mechanism criterion is the gate") applies because an LME MS A/B at n≈70 sits
inside the ±5pp churn floor and cannot see this. Offline mechanism decides
build/no-build; scored confirmation lives on LoCoMo n=800 + MSC (Step 5).

── The gate (pre-registered; do not re-derive it after seeing the numbers) ──────
BUILD iff ALL four hold:
  1. gold reachable in ≤5 facts on **≥60%** of the banked MS synthesis misses
  2. faithfulness hand-score **≥0.9** over a ~20-session sample
  3. median facts per session **≤8**
  4. the MS-hit control shows no systematic over-extraction (median **≤12**/session)
One prompt iteration (`FACTS_PROMPT_V2`) is allowed on a VISIBLE prompt defect;
a second failure banks E1 dead. Faithfulness is a hand-read — the probe dumps the
material and reports the verdict as INCOMPLETE until `--faithfulness` supplies the
score, so the banner can never print PASS on three of four criteria.

── "Reachable" is measured by PROVENANCE (instrument fix, 2026-07-30) ──────────
The first real run returned a hard 0% against a `--sim` reading of 82%, and the
gap was the INSTRUMENT, not the extraction. Criterion 1 originally asked whether
a returned fact CONTAINED the gold turn (`_gold_in_pool`: substring or a shared
40-char prefix). But a narrative fact is a REWRITE of its turn — this prompt
demands exactly that ("self-contained without the source turn") — so containment
can essentially never fire on real output. It fired under `--sim` only because
canned "facts" are verbatim turns. The check was structurally incapable of
returning true for the thing being tested.

Three readings are now reported, on BOTH arms:
  1. **gold-session provenance** (GATED): a returned fact was extracted from a
     session that contains the answer. Exact, paraphrase-proof, and it is the
     claim E1 actually makes — the fact tier puts the reader within reach.
  2. **answer string present** (corroborating): the gold answer value survives
     into a returned fact. The prompt demands verbatim values, so a faithful
     fact about the right exchange should carry it. Answers under
     `_MIN_ANSWER_CHARS` are excluded — "40" matches something by chance.
  3. **verbatim gold turn** (diagnostic only): the original broken check, kept as
     a strict lower bound and so the lesson stays visible in the output.

The 0.60 THRESHOLD is unchanged. Repairing an instrument that cannot return true
is not the same as moving a gate after seeing the numbers — but the distinction
only holds if the repair is itself controlled, so the MS-hit control arm is now
measured and printed alongside the misses. If both arms return the same extreme,
or the control does not beat the misses, the report says the gate is UNREAD
rather than letting a confident constant pass for a result.

── Why there is no per-question store rebuild ──────────────────────────────────
The plan sketched the `--inspect-floor` pattern (rebuild a temp HyMem per
question, ingest, retrieve). That pattern exists to reproduce a RETRIEVAL state;
this probe exercises no HyMem tier — it extracts from the haystack sessions and
indexes into its own FTS5 table, which is exactly what migration 026's
`narrative_facts_fts` will be. Rebuilding + dreaming a store per question would
cost hundreds of LLM calls and measure nothing extra, so the store is skipped and
the FTS query path is imported from production (`_FTS_SAFE`, `_fold_diacritics`)
so the tokenization under test is the real one.

One consequence, stated plainly: the selection rule's last clause ("gold survived
into the sent context") is inherited from the SOURCE RUN's instrumentation
(`recall_ceiling`, `gold_turn_tiers`) rather than re-verified live the way
`--distill-dryrun` does it. Rows where gold was in the pool but below the
char-cap are therefore not excluded here. That is conservative for the gate
(a fact tier rescuing such a row is a real win, not a measurement artifact) but
it means this set can be slightly WIDER than the banked-20 synthesis set —
the probe prints the recovered n so the reconciliation is visible.

── Cost ────────────────────────────────────────────────────────────────────────
Extraction is ONE call per session, and an LME haystack carries tens of sessions
per question — so the honest cost is `questions × sessions`, not the ~40 calls
the plan's budget line implies (that figure corresponds to one call per
QUESTION). `--cost` prints the call count and exits, and every run prints what it
actually spent.

**Cut questions, not sessions.** Two knobs bound the spend and they are NOT
equivalent. `--max-questions` shrinks n: the gate is a FRACTION (≥60%), so a
smaller set still reads it honestly, just with a wider band. `--max-sessions`
caps sessions per question and is DANGEROUS as a budget lever: it is label-free
(it keeps the most recent sessions, never consulting which one holds gold), so
when it drops the gold session the question scores a guaranteed miss and G-F1
fails for a BUDGET reason wearing a mechanism reason's clothes. The probe prints
a warning whenever the cap could have cut a gold-bearing session. Use
`--max-questions` for the budget; reserve `--max-sessions` for a deliberate
"does a recent-only window suffice?" experiment.

── Usage (from benchmarks/) ────────────────────────────────────────────────────
  # plumbing, offline, no LLM, no dataset needed beyond the source run:
  python fact_probe.py --source run.json --dataset lme_s.json --sim

  # the real gate:
  python fact_probe.py --source run.json --dataset lme_s.json --cost
  python fact_probe.py --source run.json --dataset lme_s.json \
      --api-key $DEEPSEEK_API_KEY --workers 4 --out facts_dump.json
  # after hand-scoring facts_dump.json's faithfulness sample:
  python fact_probe.py ... --faithfulness 0.95
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longmemeval_adapter import (  # noqa: E402
    LLMClient,
    _extract_gold_turns,
    _gold_in_pool,
    _norm_text,
    _normalize_date,
    load_longmemeval_data,
)

# The production FTS query path, imported rather than reimplemented: the whole
# point of the density number is that it holds under the tokenizer the fact tier
# will actually use (diacritic folding + the ASCII-safe token whitelist).
from hymem.query.augment import _FTS_SAFE, _fold_diacritics  # noqa: E402

# ── Draft extraction prompt ─────────────────────────────────────────────────
# Deliberately lives HERE and not in `hymem/extraction/prompts/__init__.py`: a
# front-run gate must not land a core change before it has cleared. Step 4 moves
# this text (unchanged if the gate passes on it) into the prompts module behind
# `FACTS_PROMPT_VERSION`, and the artifacts this probe dumps become the test
# fixtures for `tests/test_facts.py` — append-only extraction means a fact
# extracted here is byte-identical to one the build would extract.
FACTS_PROMPT_VERSION_DRAFT = "facts.v1"

FACTS_PROMPT_V1 = """You extract NARRATIVE FACTS from one conversation session.

A narrative fact is a single, self-contained statement of something that happened, was decided, was preferred, or was true — written so it can be read and understood WITHOUT the conversation around it.

Output a strict JSON array. Each item has exactly:
- text (string): the fact, one sentence, self-contained. Name the people, things and values explicitly instead of using "he", "it", "that", "the project".
- date (string or null): the ISO date (YYYY-MM-DD) the fact refers to, if the session states one or the session date applies. null when no date is warranted — never guess one.
- entities (array of strings): the concrete people, products, places, tools or organizations the fact is about, as written.

Rules:
- VERBATIM VALUES. Names, numbers, dates, versions, prices and quantities must appear exactly as the turns state them. Never round, convert, or normalize a value.
- NEVER INVENT. If the turns do not state something, it does not go in. No inferred motives, no filled-in details, no outcomes that were not reached.
- One fact per exchange, decision, event or outcome — not one per turn. Combine a question and its answer into the single fact they establish.
- Self-contained means resolvable alone: "Atta moved the MedFlow deploy to fly.io" — not "he moved it there".
- Prefer specific over comprehensive. 2 to 8 facts for a substantive session; skip greetings, small talk, and turns that establish nothing.
- Keep the fact in the language of the conversation.
- An empty array [] is a valid answer for a session that establishes nothing.
"""

FACTS_USER_TEMPLATE_V1 = """Session date: {date}

Conversation:
\"\"\"
{text}
\"\"\"

Return the JSON array of narrative facts now."""

# Validation bounds, mirroring `validate_episode_items` / `validate_profile_items`
# so the probe rejects exactly what the build's validator will reject. A gate run
# on laxer validation than the build would over-report density.
_MAX_FACT_CHARS = 600
_MAX_FACTS_PER_SESSION = 8
_SESSION_CHAR_CAP = 12000  # matches `dream_digest_max_chars`
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Pre-registered G-F1 thresholds. Named constants, not CLI defaults — a knob the
# reader can turn is not a gate.
_MIN_GOLD_IN_FACTS = 0.60
_MIN_FAITHFULNESS = 0.90
_MAX_MEDIAN_FACTS = 8
_MAX_CONTROL_MEDIAN_FACTS = 12

# How many facts the tier is allowed to return per question — the density claim.
_FACT_TOP_K = 5

# Minimum normalized length for the answer-containment check to mean anything.
# LME answers are often a bare value ("40", "ruff"); a 2-character answer will
# appear inside some fact by chance, which would report as signal.
_MIN_ANSWER_CHARS = 4

# Discrimination floor for the miss-vs-control contrast. A gap of this many
# questions or fewer — or any arm smaller than `_MIN_N_FOR_DISCRIMINATION` — is
# reported as too small to read, so a one-row difference is never narrated as a
# separation between the arms.
_MIN_DISCRIMINATING_GAP_Q = 2
_MIN_N_FOR_DISCRIMINATION = 20


def gold_session_ids(q_data: dict) -> set[str]:
    """Haystack session ids that CONTAIN the answer — the provenance ground truth.

    Mirrors `_extract_gold_turns`'s two modes: prefer the turn-level
    `has_answer` flags, fall back to `answer_session_ids`. Returns an empty set
    when the dataset carries neither, so such a question is excluded from the
    rate rather than scored against a fabricated gold.

    This reads a label, and that is fine BECAUSE it is applied strictly AFTER
    retrieval, to score it. Nothing upstream — extraction, indexing, ranking —
    ever sees it.
    """
    sessions = q_data.get("haystack_sessions", []) or []
    sids = q_data.get("haystack_session_ids",
                      [str(i) for i in range(len(sessions))])
    out = {
        sid for sid, sess in zip(sids, sessions)
        if any(isinstance(m, dict) and m.get("has_answer")
               and (m.get("content") or "").strip() for m in sess)
    }
    return out or {str(s) for s in (q_data.get("answer_session_ids") or [])}


# ── Selection (pure; pinned by tests) ───────────────────────────────────────

def select_probe_sets(
    run: dict, *, seed: int = 0, category: str = "multi-session"
) -> tuple[list[str], list[str], dict]:
    """Recover the banked MS synthesis-miss set + an equal-sized MS-hit control.

    The readside §2.1 rule, JSON-side: `question_type == category` (answerable —
    the `_abs` suffix is part of question_type, so an abstention row can never
    match), `correct` false, `recall_ceiling` true (gold WAS in the pool, so this
    is not a retrieval floor), and no `"none"` in `gold_turn_tiers` (no gold turn
    was missed by every tier). The control is a seeded random sample of the same
    category's `correct=true` rows, so the two arms are paired across runs.

    Returns (miss_ids, control_ids, diagnostics). Raises ValueError when the
    source run was not produced by the instrumented adapter — an uninstrumented
    run silently yields an empty set, which would read as "no misses".
    """
    pq = run.get("per_question", [])
    if not pq:
        raise ValueError("source run has no `per_question` rows")
    if not any("gold_turn_tiers" in r for r in pq):
        raise ValueError(
            "source run has no `gold_turn_tiers` — the synthesis-miss selection "
            "needs an instrumented run (any recent longmemeval_adapter baseline)"
        )

    def _cat(r: dict) -> bool:
        return r.get("question_type") == category

    misses = [
        r["question_id"] for r in pq
        if _cat(r) and not r.get("correct")
        and r.get("recall_ceiling") is True
        and not any(t == "none" for t in (r.get("gold_turn_tiers") or []))
    ]
    hits = [r["question_id"] for r in pq if _cat(r) and r.get("correct")]
    rng = random.Random(seed)
    n_ctrl = min(len(misses), len(hits))
    control = rng.sample(hits, n_ctrl) if n_ctrl < len(hits) else list(hits)

    diag = {
        "rows": len(pq),
        "category": category,
        "category_rows": sum(1 for r in pq if _cat(r)),
        "floor_excluded": sum(
            1 for r in pq
            if _cat(r) and not r.get("correct") and r.get("recall_ceiling") is True
            and any(t == "none" for t in (r.get("gold_turn_tiers") or []))
        ),
        "retrieval_excluded": sum(
            1 for r in pq
            if _cat(r) and not r.get("correct") and r.get("recall_ceiling") is False
        ),
        "n_misses": len(misses),
        "n_control": len(control),
    }
    return misses, control, diag


# ── Extraction ──────────────────────────────────────────────────────────────

def render_session(messages: list[dict]) -> str:
    """Session turns as the extraction prompt sees them, char-capped."""
    lines = []
    for m in messages:
        role = (m.get("role") or "user") if isinstance(m, dict) else "user"
        content = (m.get("content") or "") if isinstance(m, dict) else ""
        if content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)[:_SESSION_CHAR_CAP]


def validate_facts(raw: object, *, session_date: str | None) -> list[dict]:
    """Coerce a model response into validated fact dicts; drop what fails.

    Mirrors the build's validator: non-empty text ≤600 chars, ISO-or-null date,
    entity list of strings, cap at `_MAX_FACTS_PER_SESSION` (truncate, so a
    runaway response is bounded before it can inflate the density number)."""
    if isinstance(raw, str):
        text = raw.strip()
        # Tolerate a fenced or prose-wrapped array — the same leniency the
        # dreaming parsers apply, no more.
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return []
        try:
            raw = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        date = item.get("date")
        date = str(date).strip() if date else ""
        if date and not _ISO_DATE.match(date):
            # A malformed date is dropped, the fact is kept: the date is
            # metadata, the text is the evidence.
            date = ""
        if not date and session_date:
            date = session_date[:10]
        ents = item.get("entities")
        entities = [str(e).strip() for e in ents if str(e).strip()] if isinstance(ents, list) else []
        out.append({"text": text[:_MAX_FACT_CHARS], "date": date or None,
                    "entities": entities})
        if len(out) >= _MAX_FACTS_PER_SESSION:
            break
    return out


def extract_facts(
    llm: LLMClient, messages: list[dict], *, session_date: str | None
) -> tuple[list[dict], bool]:
    """ONE extraction call for one session. Returns (facts, parse_failed)."""
    text = render_session(messages)
    if not text.strip():
        return [], False
    raw = llm.chat(
        [
            {"role": "system", "content": FACTS_PROMPT_V1},
            {"role": "user", "content": FACTS_USER_TEMPLATE_V1.format(
                date=(session_date or "unknown")[:10], text=text)},
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    if raw.startswith("[LLM_ERROR"):
        return [], True
    facts = validate_facts(raw, session_date=session_date)
    return facts, (not facts and "[" not in raw)


def sim_extract(messages: list[dict], *, session_date: str | None) -> list[dict]:
    """Canned extraction for `--sim`: one 'fact' per substantive user turn, text
    verbatim. It is NOT a quality model — its only job is to make the plumbing
    (selection → extraction → FTS index → gold containment) exercisable with no
    LLM and no network, so a broken pipeline is caught before any spend. Because
    the text is verbatim, `--sim` gold containment is an upper bound and must
    never be read as evidence about the real prompt."""
    facts: list[dict] = []
    for m in messages:
        if not isinstance(m, dict) or (m.get("role") or "") != "user":
            continue
        content = (m.get("content") or "").strip()
        if len(content) < 40:
            continue
        facts.append({
            "text": content[:_MAX_FACT_CHARS],
            "date": (session_date or "")[:10] or None,
            "entities": [],
        })
        if len(facts) >= _MAX_FACTS_PER_SESSION:
            break
    return facts


# ── Fact index (what migration 026's narrative_facts_fts will be) ───────────

_FACT_SCHEMA = """
CREATE TABLE facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    text TEXT NOT NULL,
    fact_date TEXT,
    entities TEXT NOT NULL DEFAULT '[]'
);
CREATE VIRTUAL TABLE facts_fts USING fts5(text);
"""


def open_fact_index() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_FACT_SCHEMA)
    return conn


def index_facts(conn: sqlite3.Connection, session_id: str, facts: list[dict]) -> None:
    """Insert facts and mirror them into the FTS shadow at a PINNED rowid.

    The explicit `facts_fts(rowid, text)` insert is the point: an external-content
    FTS5 table whose rowids drift from its content table returns a confident
    constant (the diagnostic-controls lesson — one of the two traps that make a
    probe lie), so the shadow is a plain fts5 table written with the id we just
    allocated."""
    for f in facts:
        cur = conn.execute(
            "INSERT INTO facts(session_id, text, fact_date, entities) VALUES (?,?,?,?)",
            (session_id, f["text"], f.get("date"), json.dumps(f.get("entities") or [])),
        )
        conn.execute("INSERT INTO facts_fts(rowid, text) VALUES (?,?)",
                     (int(cur.lastrowid), f["text"]))
    conn.commit()


def search_facts(
    conn: sqlite3.Connection, question: str, *, top_k: int = _FACT_TOP_K
) -> list[dict]:
    """BM25 over the fact index using the PRODUCTION query path."""
    cleaned = _FTS_SAFE.sub(" ", _fold_diacritics(question)).strip()
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in tokens)
    try:
        rows = conn.execute(
            """
            SELECT f.id, f.session_id, f.text, f.fact_date,
                   bm25(facts_fts) AS score
            FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid
            WHERE facts_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"id": r["id"], "session_id": r["session_id"], "text": r["text"],
             "fact_date": r["fact_date"], "score": float(r["score"])} for r in rows]


# ── Per-question run ────────────────────────────────────────────────────────

def run_question(
    q_data: dict, *, llm: LLMClient | None, sim: bool, max_sessions: int,
    top_k: int = _FACT_TOP_K,
) -> dict:
    """Extract facts over one question's haystack, index them, retrieve top-k,
    and report whether the gold turn is reachable inside those k facts."""
    qid = q_data.get("question_id")
    question = q_data.get("question", "")
    sessions = q_data.get("haystack_sessions", []) or []
    sids = q_data.get("haystack_session_ids",
                      [str(i) for i in range(len(sessions))])
    dates = q_data.get("haystack_dates", []) or []

    order = list(range(len(sessions)))
    gold_cut = False
    if max_sessions and len(order) > max_sessions:
        # Keep the most RECENT sessions. Label-free by construction — recency is
        # a property of the haystack, not of which session holds the answer.
        order = order[-max_sessions:]
        # ...which is exactly why the cap must be AUDITED after the fact: a
        # dropped gold session makes this question an automatic miss, and a gate
        # failure caused by the budget must never be read as a mechanism result.
        # Checked here (never used to choose sessions) purely so the report can
        # say how many rows were decided by the cap rather than by extraction.
        gold_texts, _ = _extract_gold_turns(q_data)
        if gold_texts:
            kept = "\n".join(render_session(sessions[i]) for i in order)
            gold_cut = not _gold_in_pool(gold_texts, [kept])

    conn = open_fact_index()
    out = {"question_id": qid, "question": question, "sessions_processed": 0,
           "calls": 0, "parse_failures": 0, "facts_total": 0,
           "facts_per_session": [], "retrieved": [],
           "gold_session_in_facts": False, "answer_in_facts": False,
           "answer_checkable": False, "gold_verbatim_in_facts": False,
           "gold_turns": 0, "gold_session_ids": [],
           "gold_cut_by_session_cap": gold_cut,
           "error": None, "dump": []}
    try:
        for idx in order:
            messages = sessions[idx]
            sid = sids[idx] if idx < len(sids) else str(idx)
            session_date = _normalize_date(dates[idx]) if idx < len(dates) else None
            if sim:
                facts = sim_extract(messages, session_date=session_date)
                failed = False
            else:
                assert llm is not None
                facts, failed = extract_facts(llm, messages, session_date=session_date)
                out["calls"] += 1
            out["parse_failures"] += int(failed)
            out["sessions_processed"] += 1
            out["facts_per_session"].append(len(facts))
            out["facts_total"] += len(facts)
            index_facts(conn, sid, facts)
            if facts:
                out["dump"].append({
                    "session_id": sid, "date": session_date,
                    "facts": facts,
                    # Source turns travel WITH the facts so the faithfulness
                    # hand-score is a self-contained read (no re-joining the
                    # dataset to check a value is verbatim).
                    "source_turns": render_session(messages)[:4000],
                })

        retrieved = search_facts(conn, question, top_k=top_k)
        out["retrieved"] = retrieved
        gold_turns, _ = _extract_gold_turns(q_data)
        out["gold_turns"] = len(gold_turns)
        gold_sids = gold_session_ids(q_data)
        out["gold_session_ids"] = sorted(gold_sids)

        # PRIMARY: provenance. A returned fact was EXTRACTED FROM a gold-bearing
        # session, so the tier put the reader within reach of the answer.
        out["gold_session_in_facts"] = bool(gold_sids) and any(
            r["session_id"] in gold_sids for r in retrieved)
        # SECONDARY: the gold ANSWER value survives into a returned fact. The
        # prompt demands verbatim values, so a faithful fact about the right
        # exchange should carry it. Short answers ("40") match spuriously, so the
        # length floor keeps this from becoming a coin flip.
        answer = str(q_data.get("answer") or "").strip()
        out["answer_checkable"] = len(_norm_text(answer)) >= _MIN_ANSWER_CHARS
        out["answer_in_facts"] = out["answer_checkable"] and any(
            _norm_text(answer) in _norm_text(r["text"]) for r in retrieved)
        # DIAGNOSTIC ONLY: verbatim gold-TURN containment. This is what the probe
        # originally gated on, and it was wrong — a narrative fact is a rewrite of
        # its turn (the prompt demands it), so containment can essentially never
        # fire on real extraction. It fires under --sim because canned "facts" are
        # verbatim turns, which is exactly why sim read 82% and the first real run
        # read 0%. Kept as a strict lower bound and to keep that lesson visible.
        out["gold_verbatim_in_facts"] = bool(gold_turns) and _gold_in_pool(
            gold_turns, [r["text"] for r in retrieved]
        )
    except Exception as e:  # a probe row must never abort the run
        out["error"] = str(e)
    finally:
        conn.close()
    return out


# ── Roll-up + gate ──────────────────────────────────────────────────────────

def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def build_faithfulness_sample(
    rows: list[dict], by_id: dict[str, dict], *, size: int, seed: int = 0
) -> list[dict]:
    """Stratified hand-score sample: GOLD-BEARING sessions first, then distractors.

    An LME haystack is mostly distractor padding — LongMemEval surrounds each
    question's gold sessions with UltraChat/ShareGPT filler, typically ~50 filler
    to ~5 gold. A uniform sample over extracted sessions is therefore ~all filler,
    and hand-scoring it measures extractor faithfulness on generic chat instead of
    on the dated, numeric, name-bearing content where a verbatim-value error
    actually costs an answer. Half the budget is reserved for gold-bearing
    sessions so the audit covers the material the gate is about; the distractor
    half is kept because over-extraction and invention on filler is a real failure
    mode too (those facts compete for the same top-5 slots).

    Each entry is tagged `stratum` so the hand-read — and any later dispute about
    what was scored — can be split by it.
    """
    gold: list[dict] = []
    filler: list[dict] = []
    for r in rows:
        q_data = by_id.get(r.get("question_id"))
        gold_sids = gold_session_ids(q_data) if q_data else set()
        for d in r.get("dump", []) or []:
            entry = {"question_id": r.get("question_id"), **d}
            if d.get("session_id") in gold_sids:
                entry["stratum"] = "gold_bearing"
                gold.append(entry)
            else:
                entry["stratum"] = "distractor"
                filler.append(entry)

    rng = random.Random(seed)
    half = max(size // 2, 1)
    take_gold = rng.sample(gold, min(half, len(gold)))
    # Any unfilled gold budget rolls over to filler rather than shrinking the
    # sample — a question set with few gold sessions still gets a full audit.
    take_filler = rng.sample(filler, min(size - len(take_gold), len(filler)))
    out = take_gold + take_filler
    rng.shuffle(out)  # so the hand-reader is not primed by ordering
    return out


def _print_sample_note(path: Path, sample: list[dict]) -> None:
    n_gold = sum(1 for e in sample if e.get("stratum") == "gold_bearing")
    print(f"\n  dump → {path}")
    print(f"  hand-score `faithfulness_sample`: {len(sample)} sessions "
          f"({n_gold} gold-bearing, {len(sample) - n_gold} distractor) — every "
          f"value/name/date must appear in that entry's `source_turns`.")
    if not n_gold:
        print("  ⚠ NO gold-bearing sessions in the sample — the hand-score would "
              "measure faithfulness on LME's UltraChat/ShareGPT padding only. "
              "Do not score it; check that the dump rows carry `dump[].session_id` "
              "matching the dataset's gold sessions.")
    print("  Note: most LME haystack sessions ARE UltraChat/ShareGPT distractors "
          "by dataset design (LongMemEval pads each haystack with them), so "
          "seeing them here is expected — that is why the sample is stratified.")


def rescore_rows(rows: list[dict], by_id: dict[str, dict]) -> list[dict]:
    """Recompute the density readings on ALREADY-EXTRACTED rows.

    The extraction is the expensive part and it is deterministic once done: a
    dump's `retrieved` list carries each returned fact's `session_id` and text,
    which is everything the provenance and answer checks need. So an instrument
    fix costs zero LLM calls — re-read the dump instead of re-extracting. This is
    the whole reason `--out` writes `per_question` in full.

    Rows are mutated copies; anything the dump lacks (an old dump predating a
    field) is recomputed from `by_id`, never assumed.
    """
    out = []
    for r in rows:
        row = dict(r)
        q_data = by_id.get(row.get("question_id"))
        retrieved = row.get("retrieved") or []
        if q_data is None:
            row["error"] = row.get("error") or "question not in --dataset"
            out.append(row)
            continue
        gold_sids = gold_session_ids(q_data)
        row["gold_session_ids"] = sorted(gold_sids)
        row["gold_session_in_facts"] = bool(gold_sids) and any(
            h.get("session_id") in gold_sids for h in retrieved)
        answer = str(q_data.get("answer") or "").strip()
        row["answer_checkable"] = len(_norm_text(answer)) >= _MIN_ANSWER_CHARS
        row["answer_in_facts"] = row["answer_checkable"] and any(
            _norm_text(answer) in _norm_text(h.get("text", "")) for h in retrieved)
        gold_turns, _ = _extract_gold_turns(q_data)
        row["gold_verbatim_in_facts"] = bool(gold_turns) and _gold_in_pool(
            gold_turns, [h.get("text", "") for h in retrieved])
        row.setdefault("gold_cut_by_session_cap", False)
        out.append(row)
    return out


def summarize(miss_rows: list[dict], ctrl_rows: list[dict],
              faithfulness: float | None) -> dict:
    ok_miss = [r for r in miss_rows if not r["error"]]
    ok_ctrl = [r for r in ctrl_rows if not r["error"]]
    n = len(ok_miss) or 1
    covered = sum(1 for r in ok_miss if r["gold_session_in_facts"])

    def _density(rows: list[dict]) -> dict:
        """The three density readings for one arm. Reported TOGETHER on both arms
        because a single number cannot be trusted on its own: the
        diagnostic-controls lesson is that a broken check returns a confident
        constant, and the only way to see that is a second population where the
        answer is known to differ (here: MS HITS, which the live pipeline did
        answer correctly)."""
        m = len(rows) or 1
        checkable = [r for r in rows if r.get("answer_checkable")]
        return {
            "n": len(rows),
            "gold_session": sum(1 for r in rows if r["gold_session_in_facts"]),
            "gold_session_rate": 100.0 * sum(
                1 for r in rows if r["gold_session_in_facts"]) / m,
            "answer": sum(1 for r in checkable if r["answer_in_facts"]),
            "answer_checkable": len(checkable),
            "answer_rate": (100.0 * sum(1 for r in checkable if r["answer_in_facts"])
                            / (len(checkable) or 1)),
            "verbatim": sum(1 for r in rows if r.get("gold_verbatim_in_facts")),
        }

    per_sess_miss = [c for r in ok_miss for c in r["facts_per_session"]]
    per_sess_ctrl = [c for r in ok_ctrl for c in r["facts_per_session"]]
    median_miss = _median(per_sess_miss)
    median_ctrl = _median(per_sess_ctrl)

    gate = {
        "density_ok": (covered / n) >= _MIN_GOLD_IN_FACTS,
        "facts_per_session_ok": median_miss <= _MAX_MEDIAN_FACTS,
        "control_ok": median_ctrl <= _MAX_CONTROL_MEDIAN_FACTS,
        "faithfulness_ok": (faithfulness is not None
                            and faithfulness >= _MIN_FAITHFULNESS),
    }
    return {
        "n_misses": len(ok_miss), "n_control": len(ok_ctrl),
        "errors": len(miss_rows) - len(ok_miss) + len(ctrl_rows) - len(ok_ctrl),
        "top_k": _FACT_TOP_K,
        "gold_in_facts": covered,
        "gold_in_facts_rate": 100.0 * covered / n,
        "density_misses": _density(ok_miss),
        "density_control": _density(ok_ctrl),
        "median_facts_per_session": median_miss,
        "median_facts_per_session_control": median_ctrl,
        "mean_facts_per_question": (sum(r["facts_total"] for r in ok_miss) / n),
        "parse_failures": sum(r["parse_failures"] for r in miss_rows + ctrl_rows),
        "calls": sum(r["calls"] for r in miss_rows + ctrl_rows),
        # Rows whose gold session `--max-sessions` threw away: guaranteed misses
        # that say nothing about extraction. Non-zero ⇒ the density number is a
        # floor, not a measurement.
        "gold_cut_by_session_cap": sum(
            1 for r in ok_miss if r.get("gold_cut_by_session_cap")),
        "faithfulness": faithfulness,
        "gate": gate,
        # A missing hand-score is INCOMPLETE, never PASS: three of four criteria
        # is not the gate.
        "verdict": ("PASS" if all(gate.values())
                    else "INCOMPLETE (faithfulness hand-score not supplied)"
                    if all(v for k, v in gate.items() if k != "faithfulness_ok")
                    and faithfulness is None
                    else "FAIL"),
    }


def report(s: dict, diag: dict, miss_rows: list[dict], verbose: bool) -> bool:
    print(f"\n{'='*72}\nE1 FRONT-RUN PROBE — G-F1")
    print(f"  source rows: {diag['rows']}   category: {diag['category']} "
          f"({diag['category_rows']} rows)")
    print(f"  selection: {diag['n_misses']} synthesis-miss candidates, "
          f"{diag['n_control']} control")
    print(f"    excluded: {diag['retrieval_excluded']} retrieval misses "
          f"(recall_ceiling=false), {diag['floor_excluded']} floor rows "
          f"(a gold turn in NO tier)")
    if diag["n_misses"] != 20:
        print(f"  ⚠ recovered set = {diag['n_misses']}, not the banked 20 — the "
              f"gate is a FRACTION (≥{_MIN_GOLD_IN_FACTS:.0%}), not a count; "
              f"reconcile against the decomposition before reading it.")
    print(f"{'='*72}")
    dm, dc = s["density_misses"], s["density_control"]
    print(f"\n  DENSITY — is the answer reachable in top-{s['top_k']} facts?")
    print(f"    {'':<34}{'MISSES':>16}{'CONTROL (hits)':>18}")
    print(f"    {'gold-session provenance ←gated':<34}"
          f"{dm['gold_session']:>7}/{dm['n']:<4}{dm['gold_session_rate']:>5.0f}%"
          f"{dc['gold_session']:>9}/{dc['n']:<4}{dc['gold_session_rate']:>5.0f}%")
    print(f"    {'answer string present':<34}"
          f"{dm['answer']:>7}/{dm['answer_checkable']:<4}{dm['answer_rate']:>5.0f}%"
          f"{dc['answer']:>9}/{dc['answer_checkable']:<4}{dc['answer_rate']:>5.0f}%")
    print(f"    {'verbatim gold turn (diagnostic)':<34}"
          f"{dm['verbatim']:>7}/{dm['n']:<4}{'':>6}{dc['verbatim']:>9}/{dc['n']:<4}")
    # The control column IS the validity check, not decoration. Read it first.
    # Discrimination is judged in QUESTIONS, never in percentage points: at n=10
    # one question IS ten points, so a 10pp "gap" between the arms is a single row
    # and means nothing. Same discipline as the LME churn floor — a delta smaller
    # than the unit of measurement is not a delta.
    if dm["n"] and dc["n"]:
        gap_q = abs(dm["gold_session"]
                    - round(dc["gold_session_rate"] * dm["n"] / 100.0))
        if dm["gold_session_rate"] == dc["gold_session_rate"] in (0.0, 100.0):
            print(f"    ⚠ both arms returned the SAME extreme "
                  f"({dm['gold_session_rate']:.0f}%) — that is the signature of a "
                  f"broken check, not a finding. Do NOT read the gate; debug the "
                  f"measure (dump `retrieved` + `gold_session_ids` for one row).")
        elif dc["gold_session_rate"] <= dm["gold_session_rate"]:
            print("    ⚠ the CONTROL (questions the live pipeline answered "
                  "correctly) scores no better than the misses — the measure is "
                  "not discriminating. Treat the gate as unread.")
        elif (gap_q <= _MIN_DISCRIMINATING_GAP_Q
                or dm["n"] < _MIN_N_FOR_DISCRIMINATION):
            print(f"    ⚠ the arms differ by ~{gap_q} question(s) at n={dm['n']} — "
                  f"inside this set's resolution, so the measure is only weakly "
                  f"discriminating. The RATE can still be read against the "
                  f"{_MIN_GOLD_IN_FACTS:.0%} threshold, but do NOT read the "
                  f"miss-vs-control CONTRAST as evidence of anything.")
        if (dm["answer_checkable"] and dc["answer_checkable"]
                and dm["answer"] == dc["answer"] == 0):
            print("    ⚠ answer-string containment is 0 on BOTH arms — including "
                  "questions the pipeline ANSWERED CORRECTLY, whose facts "
                  "demonstrably reach the answer. The check is not firing (LME "
                  "`answer` fields are prose, so exact substring never matches). "
                  "Ignore that row; it is NOT evidence that facts drop values.")
    print(f"  facts/session median: {s['median_facts_per_session']:.1f}  "
          f"(control: {s['median_facts_per_session_control']:.1f})")
    print(f"  facts/question mean:  {s['mean_facts_per_question']:.1f}")
    print(f"  extraction calls: {s['calls']}   parse failures: {s['parse_failures']}"
          + (f"   row errors: {s['errors']}" if s["errors"] else ""))
    if s.get("gold_cut_by_session_cap"):
        print(f"  ⚠ {s['gold_cut_by_session_cap']}/{s['n_misses']} misses had "
              f"their GOLD SESSION cut by --max-sessions — those are forced "
              f"misses. The density number below is a FLOOR, not a measurement; "
              f"re-run without the session cap before reading G-F1.")
    print("  faithfulness: "
          + (f"{s['faithfulness']:.2f} (hand-scored)" if s["faithfulness"] is not None
             else "NOT SCORED — hand-read the dump, then re-run with --faithfulness"))

    if verbose:
        print(f"\n{'─'*72}\nper-question (misses):")
        for r in miss_rows:
            mark = "✓" if r["gold_in_facts"] else "✗"
            print(f"  [{mark}] {r['question_id']:<28}"
                  f"facts={r['facts_total']:<4}sessions={r['sessions_processed']:<4}"
                  f"gold_turns={r['gold_turns']}")
            print(f"      Q: {r['question'][:130]}")
            for f in r["retrieved"]:
                print(f"      · [{f['fact_date'] or 'undated'}] {f['text'][:130]}")

    checks = [
        (s["gate"]["density_ok"],
         f"gold-session provenance in ≤{s['top_k']} facts ≥ "
         f"{_MIN_GOLD_IN_FACTS:.0%} ({s['gold_in_facts_rate']:.0f}%)"),
        (s["gate"]["faithfulness_ok"],
         f"faithfulness ≥ {_MIN_FAITHFULNESS:.2f} "
         + (f"({s['faithfulness']:.2f})" if s["faithfulness"] is not None
            else "(not scored)")),
        (s["gate"]["facts_per_session_ok"],
         f"median facts/session ≤ {_MAX_MEDIAN_FACTS} "
         f"({s['median_facts_per_session']:.1f})"),
        (s["gate"]["control_ok"],
         f"control median ≤ {_MAX_CONTROL_MEDIAN_FACTS}/session "
         f"({s['median_facts_per_session_control']:.1f})"),
    ]
    print(f"\n── G-F1: {s['verdict']} ──")
    for ok, label in checks:
        print(f"  [{'✓' if ok else '✗'}] {label}")
    print("  One FACTS_PROMPT_V2 iteration is allowed on a visible prompt defect;\n"
          "  a second failure banks E1 dead. Bank this block in "
          "longmemeval_roadmap.md under E1.")
    return s["verdict"] == "PASS"


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path,
                    help="instrumented run JSON (the emb-ON floor-audit baseline)")
    ap.add_argument("--dataset", required=True, type=Path,
                    help="the LongMemEval dataset the source run was scored on")
    ap.add_argument("--category", default="multi-session",
                    help="question_type to select (default: the MS synthesis bank)")
    ap.add_argument("--seed", type=int, default=0, help="control-sample seed")
    ap.add_argument("--max-questions", type=int, default=0,
                    help="cap questions per arm (0 = all) — THE budget knob. The "
                         "gate is a fraction, so a smaller n still reads it")
    ap.add_argument("--max-sessions", type=int, default=0,
                    help="cap sessions extracted per question (0 = all); keeps "
                         "the most recent — label-free, but a poor budget lever: "
                         "dropping the gold session forces a miss (see --cost "
                         "output and the gold_cut_by_session_cap counter)")
    ap.add_argument("--top-k", type=int, default=_FACT_TOP_K,
                    help=f"facts returned per question (default {_FACT_TOP_K}; "
                         f"the density claim is about this number)")
    ap.add_argument("--sim", action="store_true",
                    help="canned extraction, no LLM — plumbing only; its "
                         "containment number is an upper bound, not evidence")
    ap.add_argument("--api-key", default="", help="reader/extractor API key")
    ap.add_argument("--model", default="deepseek-v4-flash", help="extraction model")
    ap.add_argument("--extra-body", default="",
                    help='JSON merged into every request, e.g. '
                         '\'{"thinking":{"type":"disabled"}}\' for v4-flash')
    ap.add_argument("--workers", type=int, default=1, help="parallel questions")
    ap.add_argument("--faithfulness", type=float, default=None,
                    help="hand-scored faithfulness (0..1) from a previous run's dump")
    ap.add_argument("--faithfulness-sample", type=int, default=20,
                    help="sessions written to the hand-score sample (default 20)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the full fact dump + summary JSON here")
    ap.add_argument("--rescore", type=Path, default=None,
                    help="re-read a previous run's --out dump and recompute the "
                         "density readings from its stored `retrieved` facts. "
                         "ZERO LLM calls — use this after an instrument fix "
                         "instead of re-extracting. Still needs --source and "
                         "--dataset for the selection and the gold labels")
    ap.add_argument("--cost", action="store_true",
                    help="print the extraction-call count and exit, spending nothing")
    ap.add_argument("--verbose", action="store_true", help="per-question table")
    args = ap.parse_args()

    run = json.loads(args.source.read_text())
    try:
        miss_ids, ctrl_ids, diag = select_probe_sets(
            run, seed=args.seed, category=args.category)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(2)
    if not miss_ids:
        print(f"ERROR: no {args.category} synthesis-miss rows in "
              f"{args.source.name} — nothing to gate on.")
        sys.exit(2)

    questions = load_longmemeval_data(str(args.dataset), max_questions=None,
                                      seed=args.seed)
    by_id = {q.get("question_id"): q for q in questions}
    missing = [q for q in miss_ids + ctrl_ids if q not in by_id]
    miss_ids = [q for q in miss_ids if q in by_id]
    ctrl_ids = [q for q in ctrl_ids if q in by_id]
    if missing:
        print(f"⚠ {len(missing)} selected qid(s) absent from {args.dataset.name} — "
              f"is this the dataset the source run scored? {missing[:5]}")

    if args.max_questions:
        # Truncate BEFORE the control is paired down, and take the head of each
        # list rather than a fresh sample: the selection order is already
        # deterministic, so a re-run at the same --max-questions hits the same
        # questions and two budgets nest (n=10 ⊂ n=20).
        miss_ids = miss_ids[: args.max_questions]
        ctrl_ids = ctrl_ids[: args.max_questions]

    def _sessions_for(qid: str) -> int:
        n = len(by_id[qid].get("haystack_sessions", []) or [])
        return min(n, args.max_sessions) if args.max_sessions else n

    total_calls = sum(_sessions_for(q) for q in miss_ids + ctrl_ids)
    if args.rescore:
        # No extraction is about to happen — printing a call estimate here would
        # be actively misleading about what this invocation costs.
        total_calls = 0
    print(f"\n[cost] {len(miss_ids)} misses + {len(ctrl_ids)} control = "
          f"{len(miss_ids) + len(ctrl_ids)} questions, "
          f"{total_calls} extraction calls"
          + ("  (--rescore: zero LLM calls, extraction reused)" if args.rescore
             else "  (--sim: zero LLM calls)" if args.sim else f"  @ {args.model}")
          + (f"   [--max-questions {args.max_questions}]" if args.max_questions else "")
          + (f"   [--max-sessions {args.max_sessions}]" if args.max_sessions else ""),
          flush=True)
    if args.max_sessions:
        sess = [len(by_id[q].get("haystack_sessions", []) or [])
                for q in miss_ids]
        capped = sum(1 for n in sess if n > args.max_sessions)
        print(f"  ⚠ --max-sessions truncates {capped}/{len(miss_ids)} miss "
              f"haystacks. Every question whose gold session falls outside the "
              f"window is a GUARANTEED miss — a G-F1 failure would then be a "
              f"budget artifact. Prefer --max-questions to bound spend; the run "
              f"reports `gold_cut_by_session_cap` so this is auditable.")
    if args.cost:
        print("  --cost: nothing spent. Re-run without it (bound spend with "
              "--max-questions, not --max-sessions) to execute.")
        return

    if args.rescore:
        prior = json.loads(args.rescore.read_text())
        rows = prior.get("per_question", [])
        if not rows:
            print(f"ERROR: {args.rescore.name} has no `per_question` rows to "
                  f"rescore (was it written with --out?).")
            sys.exit(2)
        rescored = rescore_rows(rows, by_id)
        miss_rows = [r for r in rescored if r.get("_kind") == "miss"]
        ctrl_rows = [r for r in rescored if r.get("_kind") == "ctrl"]
        print(f"\n[rescore] {args.rescore.name} — {len(miss_rows)} misses + "
              f"{len(ctrl_rows)} control, ZERO LLM calls "
              f"(extraction reused; prompt {prior.get('prompt_version')}, "
              f"model {prior.get('model')})", flush=True)
        s = summarize(miss_rows, ctrl_rows, args.faithfulness)
        passed = report(s, diag, miss_rows, args.verbose)
        if args.out:
            # Re-sample too: a rescore is exactly when the faithfulness sample
            # needs rebuilding, since the stratification depends on the gold
            # labels this pass just computed.
            sample = build_faithfulness_sample(
                rescored, by_id, size=args.faithfulness_sample, seed=args.seed)
            args.out.write_text(json.dumps(
                {**prior, "summary": s, "per_question": rescored,
                 "faithfulness_sample": sample,
                 "rescored_from": str(args.rescore)}, indent=2))
            _print_sample_note(args.out, sample)
        sys.exit(0 if passed else 1)

    llm = None
    if not args.sim:
        if not args.api_key:
            print("ERROR: --api-key is required without --sim.")
            sys.exit(2)
        extra = json.loads(args.extra_body) if args.extra_body else None
        llm = LLMClient(args.model, args.api_key, extra_body=extra)

    tasks = [("miss", q) for q in miss_ids] + [("ctrl", q) for q in ctrl_ids]

    def _one(kind: str, qid: str) -> dict:
        r = run_question(by_id[qid], llm=llm, sim=args.sim,
                         max_sessions=args.max_sessions, top_k=args.top_k)
        r["_kind"] = kind
        return r

    results: list[dict] = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_one, k, q) for k, q in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 5 == 0:
                    print(f"  ── {i}/{len(tasks)} questions done", flush=True)
    else:
        for i, (k, q) in enumerate(tasks, 1):
            results.append(_one(k, q))
            if i % 5 == 0:
                print(f"  ── {i}/{len(tasks)} questions done", flush=True)

    miss_rows = [r for r in results if r["_kind"] == "miss"]
    ctrl_rows = [r for r in results if r["_kind"] == "ctrl"]
    s = summarize(miss_rows, ctrl_rows, args.faithfulness)
    passed = report(s, diag, miss_rows, args.verbose)

    if args.out:
        sample = build_faithfulness_sample(
            results, by_id, size=args.faithfulness_sample, seed=args.seed)
        args.out.write_text(json.dumps({
            "prompt_version": FACTS_PROMPT_VERSION_DRAFT,
            "sim": args.sim,
            "model": None if args.sim else args.model,
            "summary": s,
            "selection": diag,
            "faithfulness_sample": sample,
            "per_question": results,
        }, indent=2))
        _print_sample_note(args.out, sample)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
