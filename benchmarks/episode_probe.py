#!/usr/bin/env python3
"""Plan C front-run probe — G-EP1, the gate that decides whether decision-grained
episodes get turned on.

`additional_planning.md` §"Plan C — Episode granularity in dreaming", STATUS
2026-08-30. The claim under test is NOT "granular episodes raise the score" —
BEAM EO at sample=3 has a ±12.5pp/category noise floor and cannot see this, and
the LME full guard is a non-regression check, not a tuning signal. It is the two
things a default flip can no longer measure once it has happened:

  (a) **Traceability.** Is every re-cut episode grounded in the turns it cites —
      every name, number, date and version present in the chunks the extractor
      actually saw, and no outcome that was never reached? This prompt rewrites
      the artifact retrieval ALREADY depends on (episodes feed `augment()`, the
      digest tree and `ask()`), where E1 only added a new tier beside it. A
      hallucinated episode is therefore worse than a hallucinated fact: it
      replaces evidence instead of adding to it.
  (b) **Granularity.** Does the re-cut actually produce decision-level episodes
      carrying concrete values — the thing the BEAM EO/SUM post-mortem said was
      missing ("developed budget tracker with Flask, added auth" cannot be
      decomposed into the rubric's event sequence) — rather than the same blob
      with more words?

── Why this exists at all, given G-F1 passed ───────────────────────────────────
G-F1 re-ran on 2026-08-30 at faithfulness 1.00 (98/98, full sources) and settled
the MODEL-level worry: the 0.55-0.76 that made Plan C's warning urgent was an
instrument artifact. It did NOT clear this plan's bar, which is banked "on
episode rewrites SPECIFICALLY" and measured the narrative-facts extractor.
Facts extraction and episode re-cutting are different generative tasks over the
same turns; accepting one gate's driver for another gate's question is the trap
that has cost this project three times (MSC parity, deixis, answerability). The
facts result buys a better PRIOR — "this model is faithful when asked to produce
self-contained items from session turns" — not a score.

── The gate (pre-registered; do not re-derive it after seeing the numbers) ─────
FLIP-DISCUSSABLE iff ALL of:
  1. faithfulness hand-score **>= 0.90** over the stratified sample
  2. median episodes per SUBSTANTIVE session in **[3, 8]** on the target arm
  3. **>= 60%** of episodes carry a concrete value in the summary
  4. **>= 90%** session coverage (substantive sessions yielding >= 1 episode)
  5. the correct-answer control shows no systematic over-extraction
     (median <= `dream_max_episodes_per_session`)
Above a 2% parse-failure rate the run is INCOMPLETE, never FAIL: a truncated
session yields ZERO episodes, which makes criteria 2/4 harder and criterion 5
easier — opposite directions, so a truncation-heavy run looks like a sparse,
clean, honest failure. Same pre-registration as G-F1b's ceiling.

Criteria 2-4 are MECHANICAL and this probe computes them. Criterion 1 is a
HAND-READ: the probe dumps the material and reports INCOMPLETE until
`--faithfulness` supplies the score. `summarize()` cannot return PASS without
it — that is asserted in tests, because "four of five criteria" is not the gate.

── The instrument lessons this probe is built around ───────────────────────────
1. **Record the FULL extractor input, never a re-render.** `fact_probe.py`
   stored a `[:4000]` slice of a 12000-char input; the hand-read then scored 50
   faithful facts as inventions, produced two false failing reads, nearly killed
   E1 and coloured a model migration. Here the recorded source is the LITERAL
   user prompt the extractor sent, captured inside the LLM client, and
   `assert_full_source()` re-hashes the recorded string against a sha256 taken
   at send time. A dump that fails that assertion is refused, not warned about.
2. **The input shape is the gated shape.** Episodes are extracted from CHUNKS
   (`[chunk chk_...] assistant: ... user: ...`, joined by `---`, capped at
   `dream_digest_max_chars`), not from raw turns. So the probe builds a real
   HyMem store, runs the production chunkers over each session, and calls the
   production `extract_session_digest`. Scoring raw turns instead would measure
   a corpus the feature never sees — the mistake `dreaming/facts.py` documents
   from the other direction.
3. **A correct-answer control arm, always printed beside the target arm.** A
   broken check returns a confident constant; the only way to see that is a
   second population where the answer is known to differ. Both arms at the same
   extreme ⇒ the report says the gate is UNREAD rather than letting the constant
   pass for a result.
4. **Zero LLM calls unless asked.** `--sim` runs the whole pipeline on a canned
   extractor; `--cost` prints the call count and exits; `--rescore` recomputes
   every mechanical criterion from a previous dump. Nothing here touches the
   network at import time, and there is no default API key or endpoint.

── Usage (from benchmarks/) ────────────────────────────────────────────────────
  # plumbing, offline, no LLM:
  python episode_probe.py --source run.json --dataset lme_s.json --sim

  # the real gate:
  python episode_probe.py --source run.json --dataset lme_s.json --cost
  python episode_probe.py --source run.json --dataset lme_s.json \
      --api-key $KEY --model gpt-oss-120b --out ep_dump.json
  # hand-score ep_dump.json's `faithfulness_sample`, then, for zero spend:
  python episode_probe.py --source run.json --dataset lme_s.json \
      --rescore ep_dump.json --faithfulness 0.95

  # the granularity CONTRAST (what the blob prompt cuts on the same sessions):
  python episode_probe.py ... --prompt-arm blob --out ep_blob.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hymem.config import HyMemConfig  # noqa: E402
from hymem.core import db as core_db  # noqa: E402
from hymem.dreaming.chunks import (  # noqa: E402
    extract_baseline_chunks,
    extract_fallback_chunk,
    extract_high_salience_chunks,
    persist_chunks,
)
from hymem.dreaming.digest import (  # noqa: E402
    EPISODE_GRANULAR_PROMPT_VERSION,
    extract_session_digest,
)
from hymem.extraction.llm import LLMRequest  # noqa: E402

# NOTE on imports: everything above is pure HyMem and pulls in no network stack.
# `longmemeval_adapter` (and `fact_probe`, which imports it) needs `requests`, so
# both are imported INSIDE the functions that need them — that keeps this module,
# and its offline tests, importable on a box without the benchmark deps.

# ── Pre-registered thresholds ───────────────────────────────────────────────
# Named constants, not CLI defaults: a knob the reader can turn is not a gate.
_MIN_FAITHFULNESS = 0.90
_MIN_MEDIAN_EPISODES = 3.0
_MAX_MEDIAN_EPISODES = 8.0
_MIN_CONCRETE_SHARE = 0.60
_MIN_SESSION_COVERAGE = 0.90
# Over-extraction ceiling for the control arm = the production cap, so the gate
# and the build agree on what "runaway" means (the facts precedent, where the
# cap and criterion 3 are the same number).
_MAX_CONTROL_MEDIAN_EPISODES = HyMemConfig.dream_max_episodes_per_session
_MAX_PARSE_FAILURE_RATE = 0.02

# A session below this many characters of user/assistant text is THIN: the
# granularity and coverage criteria are about substantive working sessions, and
# LME haystacks are mostly UltraChat/ShareGPT padding. Thin sessions are still
# extracted and still hand-scored (over-extraction on filler is a real failure
# mode — those episodes compete for the same retrieval slots), they are just
# excluded from criteria 2 and 4. Reported separately so the split is visible.
_MIN_SUBSTANTIVE_CHARS = 1500

# Discrimination floor, in SESSIONS. At n=10 one session IS ten points, so a
# 10pp "gap" between the arms is a single row and means nothing — the LME
# churn-floor discipline: a delta smaller than the unit of measurement is not a
# delta.
_MIN_DISCRIMINATING_GAP = 2
_MIN_N_FOR_DISCRIMINATION = 20

# A summary "carries a concrete value" if it contains a digit (number, date,
# version, price, quantity) or a token shaped like an identifier/path/version
# (fly.io, main.py, v2.1, requirements-dev). Deliberately CONSERVATIVE: bare
# proper nouns are not counted, because a capitalized word is weak evidence of
# the thing criterion 3 is about and over-counting would let a blob-shaped cut
# pass. Under-counting biases the criterion toward FAIL, which is the safe
# direction for a gate that unlocks a default flip.
_CONCRETE_VALUE = re.compile(r"\d|\b\w+(?:[./_-]\w+)+\b")


# ── Session selection ───────────────────────────────────────────────────────

def select_session_sets(
    run: dict, questions: list[dict], *, seed: int = 0,
    category: str = "multi-session", per_arm: int = 20,
) -> tuple[list[dict], list[dict], dict]:
    """Pick the TARGET sessions and the CORRECT-ANSWER CONTROL sessions.

    The question-level selection is `fact_probe.select_probe_sets`, imported
    rather than reimplemented: it is the banked readside §2.1 rule (category +
    wrong + recall_ceiling + no "none" tier) and it is pinned by its own tests.
    Two probes disagreeing about which questions are "the synthesis misses"
    would make their numbers incomparable, and this probe's whole justification
    for a control arm is inherited from that one.

    From those questions we descend to SESSIONS, because an episode re-cut is a
    per-session artifact. Each arm is stratified: gold-bearing sessions first
    (the dated, numeric, name-bearing material where a verbatim-value error
    actually costs an answer), then distractor padding, which LongMemEval
    supplies at roughly 10:1 and where over-extraction is the failure mode.

    Returns (target, control, diagnostics); each entry is a dict carrying the
    messages, the arm, the stratum and the originating question id.
    """
    from fact_probe import gold_session_ids, select_probe_sets  # noqa: E402

    miss_ids, ctrl_ids, diag = select_probe_sets(run, seed=seed, category=category)
    by_id = {q.get("question_id"): q for q in questions}

    def _sessions_for(qids: list[str], arm: str) -> list[dict]:
        gold: list[dict] = []
        filler: list[dict] = []
        for qid in qids:
            q = by_id.get(qid)
            if q is None:
                continue
            sessions = q.get("haystack_sessions", []) or []
            sids = q.get("haystack_session_ids",
                         [str(i) for i in range(len(sessions))])
            gold_sids = gold_session_ids(q)
            for i, messages in enumerate(sessions):
                sid = sids[i] if i < len(sids) else str(i)
                entry = {
                    "session_id": f"{qid}__{sid}",
                    "haystack_session_id": sid,
                    "question_id": qid,
                    "arm": arm,
                    "messages": messages,
                    "stratum": "gold_bearing" if sid in gold_sids else "distractor",
                }
                (gold if entry["stratum"] == "gold_bearing" else filler).append(entry)
        rng = random.Random(seed)
        half = max(per_arm // 2, 1)
        take_gold = rng.sample(gold, min(half, len(gold)))
        # Unfilled gold budget rolls over to filler rather than shrinking the
        # arm — a question set with few gold sessions still gets a full audit.
        take_filler = rng.sample(filler, min(per_arm - len(take_gold), len(filler)))
        out = take_gold + take_filler
        rng.shuffle(out)  # the hand-reader must not be primed by ordering
        return out

    target = _sessions_for(miss_ids, "target")
    control = _sessions_for(ctrl_ids, "control")
    diag = dict(diag)
    diag["target_sessions"] = len(target)
    diag["control_sessions"] = len(control)
    diag["target_gold_bearing"] = sum(
        1 for e in target if e["stratum"] == "gold_bearing")
    diag["control_gold_bearing"] = sum(
        1 for e in control if e["stratum"] == "gold_bearing")
    return target, control, diag


# ── Extraction against the PRODUCTION digest path ───────────────────────────

class CapturingLLM:
    """Wraps an extractor and records the EXACT request the digest sent.

    This is the [:4000] fix, structurally: the probe never re-renders what it
    thinks the extractor saw. `sent` holds the literal `LLMRequest.user` string
    plus a sha256 taken at send time, and `assert_full_source()` re-hashes the
    recorded copy later. Anything that slices the source between here and the
    dump therefore fails an assertion instead of producing a plausible,
    unfalsifiable hand-read.

    `backend` is a callable taking (system, user) and returning the raw reply —
    a canned function under --sim, an OpenAI-compatible client otherwise. There
    is no default backend and no default endpoint: HyMem ships the LLM Protocol
    and StubLLMClient only, and a benchmark that reached for a live model on its
    own would be shipping one.
    """

    def __init__(self, backend, *, max_tokens: int = 3072) -> None:
        self._backend = backend
        self.max_tokens = max_tokens
        self.sent: list[dict] = []
        self.calls = 0
        self.last_error: str | None = None

    def complete(self, request: LLMRequest) -> str:
        self.calls += 1
        self.sent.append({
            "system": request.system,
            "user": request.user,
            "user_sha256": hashlib.sha256(request.user.encode("utf-8")).hexdigest(),
            "user_chars": len(request.user),
        })
        try:
            return self._backend(request.system, request.user)
        except Exception as exc:  # a probe row must never abort the run
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ""


def sim_backend(system: str, user: str) -> str:
    """Canned extractor for --sim: one 'episode' per chunk tag in the input.

    It is NOT a quality model. Its only job is to make the pipeline (selection →
    store build → chunking → production digest call → validation → persistence →
    criteria) exercisable with no LLM and no network, so a broken instrument is
    caught before any spend. Because it is mechanical, its granularity and
    concreteness numbers are ARTIFACTS of the chunker and must never be read as
    evidence about the prompt.
    """
    chunk_ids = re.findall(r"\[chunk (chk_[0-9a-f]+)\]", user)
    episodes = [
        {
            "title": f"Sim episode {i + 1}",
            "summary": f"Simulated decision {i + 1} recorded from chunk {cid} "
                       f"with value {i + 1}.",
            "outcome": "informational",
            "key_entities": [],
            "chunk_ids": [cid],
        }
        for i, cid in enumerate(chunk_ids)
    ]
    return json.dumps({"episodes": episodes, "summary": "", "procedures": []})


def build_store(path: Path, entries: list[dict], cfg: HyMemConfig):
    """Ingest the selected sessions into a real HyMem store and chunk them.

    Chunking is LLM-free (regex salience tier + a length-based baseline backstop
    + the short-session fallback), so the digest input this probe scores is the
    one production builds — same `[chunk chk_...]` tagging, same `---` joins,
    same `dream_digest_max_chars` truncation. The one honest approximation: the
    runner spends a per-dream chunk budget across sessions and this ingests each
    session's tiers in full, so a very long session is chunked here slightly more
    completely than one dream would manage. That biases toward MORE input, never
    less, and the digest's own char cap binds either way.
    """
    conn = core_db.connect(path)
    core_db.initialize(conn)
    for entry in entries:
        sid = entry["session_id"]
        conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (sid,))
        for m in entry["messages"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or "user"
            if role not in ("user", "assistant"):
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            conn.execute(
                "INSERT INTO messages(session_id, role, content) VALUES (?,?,?)",
                (sid, role, content),
            )
        chunks = extract_high_salience_chunks(
            conn, sid, min_chars=cfg.salience_min_chars)
        seen = {c.id for c in chunks}
        chunks += [
            c for c in extract_baseline_chunks(
                conn, sid, prompt_version=cfg.prompt_version,
                limit=cfg.dream_baseline_budget, min_chars=cfg.salience_min_chars)
            if c.id not in seen
        ]
        if not chunks:
            fallback = extract_fallback_chunk(
                conn, sid, max_chars=cfg.dream_digest_max_chars)
            chunks = [fallback] if fallback is not None else []
        if chunks:
            persist_chunks(conn, chunks)
    conn.commit()
    return conn


def extract_one(conn, entry: dict, llm: CapturingLLM, cfg: HyMemConfig,
                *, granular: bool) -> dict:
    """Run the PRODUCTION digest over one prepared session and record the result.

    Returns a dump row. `parse_failed` is the digest's own flag, so the probe
    counts exactly what a dream would count into `dream_runs.digest_failures`.
    """
    row = {
        "session_id": entry["session_id"],
        "haystack_session_id": entry["haystack_session_id"],
        "question_id": entry["question_id"],
        "arm": entry["arm"],
        "stratum": entry["stratum"],
        "session_chars": sum(
            len((m.get("content") or "")) for m in entry["messages"]
            if isinstance(m, dict)),
        "episodes": [],
        "parse_failed": False,
        "calls": 0,
        "extractor_input": None,
        "extractor_input_sha256": None,
        "extractor_input_chars": 0,
        "error": None,
    }
    before = llm.calls
    try:
        digest = extract_session_digest(
            conn, entry["session_id"], llm,
            max_tokens=cfg.dream_digest_max_tokens,
            max_chars=cfg.dream_digest_max_chars,
            granular=granular,
            max_episodes=cfg.dream_max_episodes_per_session,
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["calls"] = llm.calls - before
    if digest is None:
        # No chunks: nothing was sent, so there is no source to hand-score.
        row["error"] = "no chunks (session produced no digest input)"
        return row
    row["parse_failed"] = bool(digest.parse_failed)
    row["episodes"] = [
        {
            "title": e.get("title", ""),
            "summary": e.get("summary", ""),
            "outcome": e.get("outcome"),
            "key_entities": e.get("key_entities", []),
            "chunk_ids": e.get("chunk_ids", []),
        }
        for e in digest.episodes.items
    ]
    if llm.sent:
        # The exact string the extractor sent, with the hash taken at send time.
        sent = llm.sent[-1]
        row["extractor_input"] = sent["user"]
        row["extractor_input_sha256"] = sent["user_sha256"]
        row["extractor_input_chars"] = sent["user_chars"]
    return row


def assert_full_source(row: dict) -> None:
    """Refuse a dump row whose recorded source is not what was sent.

    Raises AssertionError. This is deliberately fatal rather than a warning: the
    G-F1b sample carried a truncated source and every downstream reading — a
    faithfulness hand-score, a "confabulates over the truncation boundary"
    finding, part of a model-migration story — was wrong in a way nobody could
    see from the artifact. A probe that cannot prove its dump is complete has no
    business printing a verdict.
    """
    if row.get("error") or row.get("extractor_input") is None:
        return  # nothing was sent; there is nothing to hand-score either
    recorded = row["extractor_input"]
    assert isinstance(recorded, str), "extractor_input must be the literal prompt"
    assert len(recorded) == row["extractor_input_chars"], (
        f"{row['session_id']}: recorded source is {len(recorded)} chars but "
        f"{row['extractor_input_chars']} were sent — the dump is TRUNCATED"
    )
    digest = hashlib.sha256(recorded.encode("utf-8")).hexdigest()
    assert digest == row["extractor_input_sha256"], (
        f"{row['session_id']}: recorded source does not hash to what was sent — "
        f"the dump was rewritten between the call and the write"
    )


# ── Mechanical criteria ─────────────────────────────────────────────────────

def carries_concrete_value(summary: str) -> bool:
    """Criterion 3's per-episode test. See `_CONCRETE_VALUE` for why it is
    deliberately conservative."""
    return bool(_CONCRETE_VALUE.search(summary or ""))


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def arm_stats(rows: list[dict]) -> dict:
    """The mechanical readings for ONE arm, reported for both arms because a
    single number cannot be trusted alone — a broken check returns a confident
    constant and only a second population where the answer differs reveals it."""
    ok = [r for r in rows if not r.get("error")]
    substantive = [r for r in ok if r["session_chars"] >= _MIN_SUBSTANTIVE_CHARS]
    counts = [len(r["episodes"]) for r in substantive]
    episodes = [e for r in ok for e in r["episodes"]]
    concrete = sum(1 for e in episodes if carries_concrete_value(e.get("summary", "")))
    covered = sum(1 for r in substantive if r["episodes"])
    outcomes = sum(1 for e in episodes if e.get("outcome"))
    return {
        "n_sessions": len(ok),
        "n_substantive": len(substantive),
        "n_thin": len(ok) - len(substantive),
        "episodes": len(episodes),
        "median_episodes": _median(counts),
        "mean_episodes": (sum(counts) / len(counts)) if counts else 0.0,
        "max_episodes": max(counts) if counts else 0,
        "coverage": (covered / len(substantive)) if substantive else 0.0,
        "covered_sessions": covered,
        "concrete": concrete,
        "concrete_share": (concrete / len(episodes)) if episodes else 0.0,
        "outcome_share": (outcomes / len(episodes)) if episodes else 0.0,
        "errors": len(rows) - len(ok),
    }


def build_faithfulness_sample(rows: list[dict], *, size: int, seed: int = 0
                              ) -> list[dict]:
    """Stratified hand-score sample: gold-bearing sessions first, then filler.

    Same reasoning as `fact_probe.build_faithfulness_sample`. Each entry carries
    its episodes AND the full extractor input, so the hand-read is
    self-contained: every value in an episode must be findable in that entry's
    `extractor_input`, with no re-joining of the dataset — and the source is the
    complete one (see `assert_full_source`).
    """
    scoreable = [r for r in rows if not r.get("error") and r.get("episodes")]
    gold = [r for r in scoreable if r["stratum"] == "gold_bearing"]
    filler = [r for r in scoreable if r["stratum"] != "gold_bearing"]
    rng = random.Random(seed)
    half = max(size // 2, 1)
    take_gold = rng.sample(gold, min(half, len(gold)))
    take_filler = rng.sample(filler, min(size - len(take_gold), len(filler)))
    out = take_gold + take_filler
    rng.shuffle(out)
    return [
        {
            "session_id": r["session_id"],
            "question_id": r["question_id"],
            "arm": r["arm"],
            "stratum": r["stratum"],
            "episodes": r["episodes"],
            "extractor_input": r["extractor_input"],
            "extractor_input_sha256": r["extractor_input_sha256"],
            "extractor_input_chars": r["extractor_input_chars"],
        }
        for r in out
    ]


def summarize(target_rows: list[dict], control_rows: list[dict],
              faithfulness: float | None) -> dict:
    """Roll the arms up and apply the pre-registered gate.

    The verdict is INCOMPLETE — never PASS — while the faithfulness hand-score
    is missing, whatever the mechanical criteria say. Four of five criteria is
    not the gate; this is the one property of this function its tests pin
    hardest.
    """
    tgt = arm_stats(target_rows)
    ctl = arm_stats(control_rows)

    calls = sum(r.get("calls", 0) for r in target_rows + control_rows)
    parse_failures = sum(
        1 for r in target_rows + control_rows if r.get("parse_failed"))
    parse_failure_rate = (parse_failures / calls) if calls else 0.0

    gate = {
        "faithfulness_ok": (faithfulness is not None
                            and faithfulness >= _MIN_FAITHFULNESS),
        "granularity_ok": (
            tgt["n_substantive"] > 0
            and _MIN_MEDIAN_EPISODES <= tgt["median_episodes"] <= _MAX_MEDIAN_EPISODES
        ),
        "concrete_ok": (tgt["episodes"] > 0
                        and tgt["concrete_share"] >= _MIN_CONCRETE_SHARE),
        "coverage_ok": (tgt["n_substantive"] > 0
                        and tgt["coverage"] >= _MIN_SESSION_COVERAGE),
        "control_ok": ctl["median_episodes"] <= _MAX_CONTROL_MEDIAN_EPISODES,
        "parse_failures_ok": parse_failure_rate <= _MAX_PARSE_FAILURE_RATE,
    }
    # Empty arms are the classic vacuous PASS: `median <= cap` and
    # `share >= floor` both hold trivially at n=0, and a probe that extracted
    # nothing would print the same banner as one that extracted well. Each
    # mechanical criterion above therefore carries its own non-zero
    # precondition, and the verdict below refuses to read at all without a
    # populated target arm.
    unread = tgt["n_substantive"] == 0 or tgt["episodes"] == 0

    mechanical = {k: v for k, v in gate.items() if k != "faithfulness_ok"}
    if unread:
        verdict = ("INCOMPLETE (target arm has no substantive sessions or no "
                   "episodes — nothing to read the criteria on)")
    elif not gate["parse_failures_ok"]:
        verdict = (
            f"INCOMPLETE (parse-failure ceiling: {parse_failures}/{calls} = "
            f"{parse_failure_rate:.1%} > {_MAX_PARSE_FAILURE_RATE:.0%} — "
            f"truncation biases the criteria in opposite directions; re-run at a "
            f"higher --max-tokens, never read as FAIL)")
    elif all(gate.values()):
        verdict = "PASS"
    elif all(mechanical.values()) and faithfulness is None:
        verdict = "INCOMPLETE (faithfulness hand-score not supplied)"
    else:
        verdict = "FAIL"

    return {
        "target": tgt,
        "control": ctl,
        "calls": calls,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failure_rate,
        "faithfulness": faithfulness,
        "gate": gate,
        "verdict": verdict,
        "thresholds": {
            "faithfulness": _MIN_FAITHFULNESS,
            "median_episodes": [_MIN_MEDIAN_EPISODES, _MAX_MEDIAN_EPISODES],
            "concrete_share": _MIN_CONCRETE_SHARE,
            "coverage": _MIN_SESSION_COVERAGE,
            "control_median": _MAX_CONTROL_MEDIAN_EPISODES,
            "substantive_chars": _MIN_SUBSTANTIVE_CHARS,
        },
    }


def report(s: dict, diag: dict, arm_label: str, verbose: bool,
           target_rows: list[dict]) -> bool:
    t, c = s["target"], s["control"]
    print(f"\n{'='*72}\nPLAN C FRONT-RUN PROBE — G-EP1   (prompt arm: {arm_label})")
    print(f"  source rows: {diag.get('rows', '?')}   category: "
          f"{diag.get('category', '?')} ({diag.get('category_rows', '?')} rows)")
    print(f"  selection: {diag.get('n_misses', '?')} miss questions → "
          f"{diag.get('target_sessions', 0)} target sessions "
          f"({diag.get('target_gold_bearing', 0)} gold-bearing);   "
          f"{diag.get('n_control', '?')} correct-answer questions → "
          f"{diag.get('control_sessions', 0)} control sessions "
          f"({diag.get('control_gold_bearing', 0)} gold-bearing)")
    print(f"{'='*72}")
    print(f"\n  {'':<36}{'TARGET':>14}{'CONTROL (hits)':>18}")
    print(f"  {'sessions (substantive / thin)':<36}"
          f"{t['n_substantive']:>7}/{t['n_thin']:<6}"
          f"{c['n_substantive']:>11}/{c['n_thin']:<6}")
    print(f"  {'episodes extracted':<36}{t['episodes']:>14}{c['episodes']:>18}")
    print(f"  {'median episodes / substantive sess':<36}"
          f"{t['median_episodes']:>14.1f}{c['median_episodes']:>18.1f}")
    print(f"  {'max episodes in one session':<36}"
          f"{t['max_episodes']:>14}{c['max_episodes']:>18}")
    print(f"  {'session coverage':<36}{t['coverage']*100:>13.0f}%"
          f"{c['coverage']*100:>17.0f}%")
    print(f"  {'episodes carrying a concrete value':<36}"
          f"{t['concrete_share']*100:>13.0f}%{c['concrete_share']*100:>17.0f}%")
    print(f"  {'episodes carrying an outcome':<36}"
          f"{t['outcome_share']*100:>13.0f}%{c['outcome_share']*100:>17.0f}%")
    print(f"\n  extraction calls: {s['calls']}   parse failures: "
          f"{s['parse_failures']}"
          + (f"   row errors: {t['errors'] + c['errors']}"
             if (t["errors"] or c["errors"]) else ""))

    # The control column IS the validity check, not decoration. Read it first.
    if t["n_sessions"] and c["n_sessions"]:
        gap = abs(t["median_episodes"] - c["median_episodes"])
        if t["episodes"] == c["episodes"] == 0:
            print("  ⚠ BOTH arms extracted zero episodes — that is the signature "
                  "of a broken pipeline, not a finding. Do NOT read the gate; "
                  "dump one row's `extractor_input` and check the chunker "
                  "produced input at all.")
        elif t["n_substantive"] == 0:
            # The medians read 0 because every session fell under the
            # substantive floor, NOT because extraction failed. Saying
            # "broken pipeline" here would send the reader after the wrong bug.
            print(f"  ⚠ every session is under the {_MIN_SUBSTANTIVE_CHARS}-char "
                  f"substantive floor, so the granularity and coverage medians "
                  f"are computed over an EMPTY set — the episode counts above are "
                  f"real, the medians are not. Widen the selection before reading "
                  f"the gate.")
        elif (t["n_substantive"] < _MIN_N_FOR_DISCRIMINATION
              and gap <= _MIN_DISCRIMINATING_GAP):
            print(f"  ⚠ the arms differ by ~{gap:.1f} episodes/session at "
                  f"n={t['n_substantive']} — inside this set's resolution. The "
                  f"RATES can still be read against their thresholds, but do NOT "
                  f"read the target-vs-control CONTRAST as evidence of anything.")
    if s.get("parse_failure_rate", 0.0) > _MAX_PARSE_FAILURE_RATE:
        print(f"  ⚠ PARSE-FAILURE CEILING EXCEEDED: {s['parse_failures']}/"
              f"{s['calls']} = {s['parse_failure_rate']:.1%} > "
              f"{_MAX_PARSE_FAILURE_RATE:.0%}. Truncation biases the criteria in "
              f"opposite directions — this run is UNREADABLE, not a FAIL. Re-run "
              f"at a higher --max-tokens.")
    print("  faithfulness: "
          + (f"{s['faithfulness']:.2f} (hand-scored)"
             if s["faithfulness"] is not None
             else "NOT SCORED — hand-read the dump, then re-run with "
                  "--rescore <dump> --faithfulness <score>"))

    if verbose:
        print(f"\n{'─'*72}\nper-session (target arm):")
        for r in target_rows:
            if r.get("error"):
                print(f"  [!] {r['session_id']:<34}{r['error']}")
                continue
            print(f"  [{r['stratum'][:4]}] {r['session_id']:<34}"
                  f"episodes={len(r['episodes']):<3}"
                  f"chars={r['session_chars']}")
            for e in r["episodes"]:
                mark = "·" if carries_concrete_value(e.get("summary", "")) else "○"
                print(f"      {mark} {e.get('title', '')[:40]:<42}"
                      f"{e.get('summary', '')[:90]}")

    th = s["thresholds"]
    checks = [
        (s["gate"]["faithfulness_ok"],
         f"faithfulness ≥ {th['faithfulness']:.2f} "
         + (f"({s['faithfulness']:.2f})" if s["faithfulness"] is not None
            else "(not scored — HAND-READ)")),
        (s["gate"]["granularity_ok"],
         f"median episodes/substantive session in "
         f"[{th['median_episodes'][0]:.0f}, {th['median_episodes'][1]:.0f}] "
         f"({t['median_episodes']:.1f})"),
        (s["gate"]["concrete_ok"],
         f"episodes carrying a concrete value ≥ {th['concrete_share']:.0%} "
         f"({t['concrete_share']:.0%})"),
        (s["gate"]["coverage_ok"],
         f"session coverage ≥ {th['coverage']:.0%} ({t['coverage']:.0%})"),
        (s["gate"]["control_ok"],
         f"control median ≤ {th['control_median']}/session "
         f"({c['median_episodes']:.1f})"),
    ]
    print(f"\n── G-EP1: {s['verdict']} ──")
    for ok, label in checks:
        print(f"  [{'✓' if ok else '✗'}] {label}")
    print("  A PASS makes the default flip DISCUSSABLE, not automatic: the LME "
          "full guard\n  (non-regression only) and the dream cost watch are "
          "separate, and this plan\n  must not overlap a RAPTOR verification "
          "dream — re-verify aggregation reuse\n  once after it lands. Bank this "
          "block in additional_planning.md under Plan C.")
    return s["verdict"] == "PASS"


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path,
                    help="instrumented LME run JSON (selection + control labels)")
    ap.add_argument("--dataset", required=True, type=Path,
                    help="the LongMemEval dataset the source run was scored on")
    ap.add_argument("--category", default="multi-session",
                    help="question_type to select (default: the MS synthesis bank)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--sessions-per-arm", type=int, default=20,
                    help="sessions extracted per arm (THE budget knob: one LLM "
                         "call per session, both arms)")
    ap.add_argument("--prompt-arm", default="granular", choices=("granular", "blob"),
                    help="which digest prompt to score. `granular` is the Plan C "
                         "arm under test; `blob` runs the SHIPPING prompt over the "
                         "same sessions, which is the only honest baseline for the "
                         "granularity contrast (run it as a separate invocation "
                         "and compare the two dumps)")
    ap.add_argument("--sim", action="store_true",
                    help="canned extraction, no LLM, no network — plumbing only; "
                         "its granularity numbers are chunker artifacts and are "
                         "NOT evidence about the prompt")
    ap.add_argument("--api-key", default="", help="extractor API key (no default)")
    ap.add_argument("--model", default="", help="extraction model (no default)")
    ap.add_argument("--base-url", default="",
                    help="OpenAI-compatible endpoint for --model (no default)")
    ap.add_argument("--extra-body", default="",
                    help='JSON merged into every request, e.g. '
                         '\'{"thinking":{"type":"disabled"}}\'')
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="output cap per digest call. Reasoning models burn this "
                         "on chain-of-thought and return content=null when it is "
                         "too small; a granular digest is also a longer reply than "
                         "a blob one, so this is deliberately above the "
                         "dream_digest_max_tokens default")
    ap.add_argument("--faithfulness", type=float, default=None,
                    help="hand-scored faithfulness (0..1) from a previous dump")
    ap.add_argument("--faithfulness-sample", type=int, default=20,
                    help="sessions written to the hand-score sample (default 20)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the full episode dump + summary JSON here")
    ap.add_argument("--rescore", type=Path, default=None,
                    help="re-read a previous --out dump and recompute every "
                         "mechanical criterion from it. ZERO LLM calls — this is "
                         "how a hand-score is applied after the fact")
    ap.add_argument("--cost", action="store_true",
                    help="print the call count and exit, spending nothing")
    ap.add_argument("--verbose", action="store_true", help="per-session table")
    args = ap.parse_args()

    if args.rescore:
        prior = json.loads(args.rescore.read_text())
        rows = prior.get("per_session", [])
        if not rows:
            print(f"ERROR: {args.rescore.name} has no `per_session` rows "
                  f"(was it written with --out?).")
            sys.exit(2)
        for r in rows:
            assert_full_source(r)  # a truncated dump is refused, not re-read
        target = [r for r in rows if r.get("arm") == "target"]
        control = [r for r in rows if r.get("arm") == "control"]
        print(f"\n[rescore] {args.rescore.name} — {len(target)} target + "
              f"{len(control)} control sessions, ZERO LLM calls "
              f"(prompt arm {prior.get('prompt_arm')}, model {prior.get('model')})",
              flush=True)
        s = summarize(target, control, args.faithfulness)
        passed = report(s, prior.get("selection", {}),
                        prior.get("prompt_arm", "?"), args.verbose, target)
        if args.out:
            args.out.write_text(json.dumps(
                {**prior, "summary": s, "rescored_from": str(args.rescore)},
                indent=2))
            print(f"\n  dump → {args.out}")
        sys.exit(0 if passed else 1)

    run = json.loads(args.source.read_text())
    from longmemeval_adapter import load_longmemeval_data  # noqa: E402
    questions = load_longmemeval_data(str(args.dataset), max_questions=None,
                                      seed=args.seed)
    try:
        target_entries, control_entries, diag = select_session_sets(
            run, questions, seed=args.seed, category=args.category,
            per_arm=args.sessions_per_arm)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(2)
    entries = target_entries + control_entries
    if not entries:
        print("ERROR: selection produced no sessions — nothing to gate on.")
        sys.exit(2)

    print(f"\n[cost] {len(target_entries)} target + {len(control_entries)} "
          f"control sessions = {len(entries)} digest calls"
          + ("  (--sim: zero LLM calls)" if args.sim else f"  @ {args.model}"),
          flush=True)
    if args.cost:
        print("  --cost: nothing spent. Re-run without it (bound spend with "
              "--sessions-per-arm) to execute.")
        return

    if args.sim:
        backend = sim_backend
    else:
        if not (args.api_key and args.model and args.base_url):
            # No default key, no default model, no default endpoint: the probe
            # must never be one flag away from spending against a live service.
            print("ERROR: --api-key, --model and --base-url are all required "
                  "without --sim.")
            sys.exit(2)
        from longmemeval_adapter import LLMClient as ApiClient  # noqa: E402
        extra = json.loads(args.extra_body) if args.extra_body else None
        client = ApiClient(args.model, args.api_key, base_url=args.base_url,
                           extra_body=extra)

        def backend(system: str, user: str) -> str:
            return client.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.0, max_tokens=args.max_tokens,
            ) or ""

    cfg = HyMemConfig(root=Path(tempfile.mkdtemp(prefix="episode_probe_")))
    conn = build_store(Path(cfg.root) / "probe.sqlite", entries, cfg)
    rows: list[dict] = []
    try:
        for i, entry in enumerate(entries, 1):
            llm = CapturingLLM(backend, max_tokens=args.max_tokens)
            row = extract_one(conn, entry, llm, cfg,
                              granular=(args.prompt_arm == "granular"))
            assert_full_source(row)
            rows.append(row)
            if i % 5 == 0:
                print(f"  ── {i}/{len(entries)} sessions done", flush=True)
    finally:
        conn.close()

    target_rows = [r for r in rows if r["arm"] == "target"]
    control_rows = [r for r in rows if r["arm"] == "control"]
    s = summarize(target_rows, control_rows, args.faithfulness)
    passed = report(s, diag, args.prompt_arm, args.verbose, target_rows)

    if args.out:
        sample = build_faithfulness_sample(
            rows, size=args.faithfulness_sample, seed=args.seed)
        args.out.write_text(json.dumps({
            "prompt_arm": args.prompt_arm,
            "prompt_version": (EPISODE_GRANULAR_PROMPT_VERSION
                               if args.prompt_arm == "granular" else None),
            "sim": args.sim,
            "model": None if args.sim else args.model,
            "summary": s,
            "selection": diag,
            "faithfulness_sample": sample,
            "per_session": rows,
        }, indent=2))
        n_gold = sum(1 for e in sample if e["stratum"] == "gold_bearing")
        print(f"\n  dump → {args.out}")
        print(f"  hand-score `faithfulness_sample`: {len(sample)} sessions "
              f"({n_gold} gold-bearing, {len(sample) - n_gold} distractor). Every "
              f"name, number, date and version in an episode must appear in that "
              f"entry's `extractor_input` — which is the COMPLETE prompt the "
              f"extractor saw, hash-verified, not a rendering of it.")
        if not n_gold:
            print("  ⚠ NO gold-bearing sessions in the sample — the hand-score "
                  "would measure faithfulness on LME's UltraChat/ShareGPT padding "
                  "only. Do not score it.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
