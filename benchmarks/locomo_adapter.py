#!/usr/bin/env python3
"""
HyMem LoCoMo Benchmark Adapter
==============================
Runs LoCoMo ("Evaluating Very Long-Term Conversational Memory of LLM Agents",
Maharana et al., ACL 2024 — snap-research/locomo, `locomo10.json`) against
HyMem's Python SDK. The third leg of the benchmark triad: LME (single-shot
star topology), MSC (short genuine multi-session), LoCoMo (LONG genuine
multi-session — 19-32 sessions / 369-689 turns per conversation, real
timestamps, and an adversarial-question class neither of the others has).

DATA (verified 2026-07-28 against the actual snap-research locomo10.json):

    conversation = {
      "sample_id": "conv-26",
      "conversation": {
        "speaker_a": str, "speaker_b": str,
        "session_<N>": [ {"speaker": str, "dia_id": "D<N>:<t>", "text": str,
                          # optional image fields on photo-share turns:
                          "img_url": [str], "blip_caption": str, "query": str}, ...],
        "session_<N>_date_time": "1:56 pm on 8 May, 2023", ... },
      "qa": [ {"question": str, "answer": str|int, "evidence": ["D1:3", ...],
               "category": 1|2|3|4},
              {"question": str, "adversarial_answer": str, "evidence": [...],
               "category": 5}, ... ],
      "observation": ..., "session_summary": ..., "event_summary": ...,   # unused
    }

  Empirical quirks the loader must absorb (all present in the real file):
  `category` is sometimes a STRING ('5'); cat-5 `evidence` is a string REPR of a
  list ("['D2:3']"); answers can be int (6 rows) or absent (cat-5); 2 rows carry
  BOTH `answer` and `adversarial_answer`. Categories (paper §3):
  1=multi-hop (282), 2=temporal (321), 3=open-domain inference (96),
  4=single-hop (841), 5=adversarial (446; the same question as a cat-4 row but
  with the speaker/premise swapped — `adversarial_answer` is the TRAP answer,
  and the CORRECT behavior is to say the information isn't there).

CONTRACT DECISIONS (the MSC lesson ×3 — feeding parity, deixis, answerability —
restated up front instead of rediscovered one 15pp regression at a time):

  * Feeding parity: retrieval goes through `MSCAdapter.search` UNCHANGED (the
    LME-parity tier collection: top_k*3 at the pipeline layer, message-first
    ordering, additive profile tier, full pre-truncation pool for diagnostics).
  * Deixis: LoCoMo questions are THIRD-PERSON BY NAME ("What did Caroline
    research?") while memories carry [user]/[assistant] tags. A per-conversation
    perspective clause states the name↔role mapping — without it the reader
    cannot attribute facts, and cat-5 exists precisely to punish attribution
    swaps.
  * Answerability: cats 1-4 are answerable by construction, but cat-5's whole
    point is that abstention is CORRECT — so unlike MSC there is NO blanket
    answerability clause. The LME base prompt's abstention permission is
    load-bearing here (it is what a cat-5 pass looks like), and cat-5 is judged
    with the LME `_abs` abstention judge. `--answerable-clause` exists as an
    opt-in A/B lever but is LABEL-LEAKY (it conditions the prompt on the very
    thing cat-5 tests) and therefore non-canonical; the report brands runs
    that use it.
  * Label-routing only where LME itself does it: cat 2 → ability "TR" (the
    time-anchor stack: TR prompt + temporal_events chronology), cat 3 →
    permissive default prompt (the D4 posture — open-domain questions require
    world-knowledge bridging by construction; abstention guard kept).

Usage:
  python locomo_adapter.py --data data/locomo10.json --sample 200
  python locomo_adapter.py --data data/locomo10.json --db-dir /tmp/locomo_dbs \
         --workers 10                     # persistent per-conversation stores
  python locomo_adapter.py --sim          # offline: loader + mechanics, no API
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling benchmark imports

# MSCAdapter is reused WHOLESALE (open/ingest/dream/search) so LoCoMo inherits
# LME feeding parity from one shared implementation instead of a third copy.
# msc_adapter's module level is stdlib-only, so this import stays --sim-safe;
# longmemeval_adapter pieces are imported lazily inside functions, MSC-style.
from msc_adapter import MSCAdapter, _lex_match

_ANSWER_MODEL = "deepseek-v4-flash"
_JUDGE_MODEL = "deepseek-v4-flash"
_HYMEM_MODEL = "deepseek-v4-flash"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# ── category contract ───────────────────────────────────────────────────────

# question_type strings for compute_scores (the _abs suffix keeps the LME
# abstention machinery — compute_abstention_scores, judge routing — working
# untouched; compute_scores strips it for the per-category table).
CATEGORY_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                 4: "single-hop", 5: "adversarial_abs"}
# Which LME judge each category maps to. Cat 2 gets the temporal judge — but
# NOTE (verified 2026-07-29 against get_judge_prompt): its off-by-one tolerance
# covers DURATIONS only ("19 days when the answer is 18"), NOT calendar dates.
# LoCoMo cat-2 golds are mostly dates, so a one-day-off date IS scored wrong;
# do not adjudicate those as judge artifacts. Cat 5
# routes to the abstention judge via the _abs suffix inside judge_answer.
CATEGORY_JUDGE = {1: "multi-session", 2: "temporal-reasoning",
                  3: "single-session-user", 4: "single-session-user",
                  5: "single-session-user_abs"}


def locomo_perspective_clause(speaker_a: str, speaker_b: str,
                              user_is_a: bool = True) -> str:
    """The deixis contract, per conversation: questions name the speakers, the
    memories tag roles. Same bug class as MSC's perspective clause (which was
    worth +14pp there) — stated up front this time, not after a miss audit."""
    user, partner = (speaker_a, speaker_b) if user_is_a else (speaker_b, speaker_a)
    return (
        f"\nThe memories are turns from past conversations between two people: "
        f"{user} ([user] turns, and the 'Known user profile' facts) and {partner} "
        f"([assistant] turns). The question refers to them BY NAME. Attribute every "
        f"fact to the speaker whose turn actually said it: what {user} said, did, or "
        f"experienced comes from [user] turns; what {partner} said, did, or "
        f"experienced comes from [assistant] turns. Never transfer one speaker's "
        f"experience, plan, or statement to the other."
    )


# Cat-3 only. Style-routing with LME precedent (oracle SS-P → preference
# prompt); the abstention guard in the permissive prompt stays intact, so this
# does NOT leak answerability the way the opt-in clause below does.
LOCOMO_OPEN_DOMAIN_CLAUSE = (
    "\nThis question may ask for a likely inference or hypothetical rather than a "
    "directly stated fact. Combine what the memories establish about the speakers "
    "with general knowledge, and commit to the single best-supported inference. "
    "Keep the answer short."
)

# OPT-IN and LABEL-LEAKY: applying "this is answerable" only to cats 1-4 uses
# the per-question category label to defeat exactly what cat-5 measures. LME
# never leaks its _abs label into the answer prompt, so neither does the
# canonical LoCoMo posture. Kept as an explicit A/B lever (--answerable-clause)
# to size the abstention-miss cost the honest posture pays on cats 1-4.
LOCOMO_ANSWERABLE_CLAUSE = (
    "\nThis question has an answer stated in the memories — never reply that you "
    "don't have enough information. If no memory states the answer outright, commit "
    "to the single best-supported answer from what the memories do say, directly "
    "and without disclaimers."
)


# ── dataset loader ──────────────────────────────────────────────────────────

_DIA_ID_RE = re.compile(r"D\d+:\d+")
_SESSION_KEY_RE = re.compile(r"session_(\d+)$")
# "1:56 pm on 8 May, 2023" — the only format observed in locomo10.json; the
# alternates are defensive (strptime %p is case-insensitive, %d/%I accept
# unpadded values; %B needs an English-month locale, the C-locale default).
_DT_FORMATS = ("%I:%M %p on %d %B, %Y", "%H:%M on %d %B, %Y", "%d %B, %Y")


def _parse_session_dt(raw: str | None) -> datetime | None:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _turn_content(turn: dict) -> str:
    """Turn text, with the BLIP caption of a shared photo appended in-line —
    the standard text-only LoCoMo treatment; image content is otherwise lost."""
    text = (turn.get("text") or "").strip()
    cap = (turn.get("blip_caption") or "").strip()
    if cap:
        text = f"{text} [shared a photo: {cap}]".strip()
    return text


def _coerce_category(q: dict) -> int | None:
    try:
        return int(str(q.get("category")).strip())
    except (TypeError, ValueError):
        return None


def _coerce_evidence(q: dict) -> list[str]:
    """Evidence dia_ids. Regex over str() absorbs both real lists and the cat-5
    string-repr quirk ("['D2:3']")."""
    return _DIA_ID_RE.findall(str(q.get("evidence") or ""))


def _coerce_answer(q: dict) -> str | None:
    a = q.get("answer")
    if a is None:
        return None
    return str(a).strip() or None


def load_locomo_data(path: str | None, *, user_speaker: str = "a",
                     categories: set[int] | None = None,
                     name_prefix: bool = False) -> list[dict]:
    """Load + normalize LoCoMo conversations. Returns:
        {id, speaker_a, speaker_b, sessions: [[{role,content}]],
         session_dates: [str], n_sessions, evidence_map: {dia_id: (sess_idx,
         content)}, qa: [{qa_id, question, answer, adversarial_answer,
         category, qtype, judge_type, evidence}]}
    `sessions`/`session_dates`/`id` intentionally match the MSC normalized
    shape so `MSCAdapter.ingest` runs verbatim. Session dates come from the
    REAL `session_N_date_time` stamps (unlike MSC there is nothing to
    synthesize); an unparseable date falls back to previous + 1 day, ordering
    preserved. `path=None` returns the built-in --sim fixture."""
    raw = _SIM_FIXTURE if not path else json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LoCoMo data must be a JSON array of conversations")

    out = []
    for ci, rec in enumerate(raw):
        conv = rec.get("conversation") or {}
        speaker_a = (conv.get("speaker_a") or "Speaker A").strip()
        speaker_b = (conv.get("speaker_b") or "Speaker B").strip()
        user_name = speaker_a if user_speaker == "a" else speaker_b

        sess_nums = sorted(int(m.group(1)) for k in conv
                           if (m := _SESSION_KEY_RE.fullmatch(k)))
        sessions, dates, evidence_map = [], [], {}
        cursor: datetime | None = None
        for n in sess_nums:
            turns_raw = conv.get(f"session_{n}") or []
            dt = _parse_session_dt(conv.get(f"session_{n}_date_time"))
            if dt is None:
                dt = (cursor + timedelta(days=1)) if cursor else datetime(2023, 5, 1)
            cursor = dt
            turns = []
            for t in turns_raw:
                content = _turn_content(t)
                if not content:
                    continue
                speaker = (t.get("speaker") or "").strip()
                role = "user" if speaker == user_name else "assistant"
                if name_prefix and speaker:
                    content = f"{speaker}: {content}"
                turns.append({"role": role, "content": content})
                dia = t.get("dia_id")
                if dia:
                    evidence_map[dia] = (len(sessions), content)
            if turns:
                sessions.append(turns)
                dates.append(dt.strftime("%Y-%m-%d %H:%M"))
        if not sessions:
            continue

        sample_id = str(rec.get("sample_id") or f"locomo_{ci}")
        qa = []
        for qi, q in enumerate(rec.get("qa") or []):
            cat = _coerce_category(q)
            question = (q.get("question") or "").strip()
            if cat not in CATEGORY_NAME or not question:
                continue
            if categories and cat not in categories:
                continue
            qa.append({
                "qa_id": f"{sample_id}_q{qi}",
                "question": question,
                "answer": _coerce_answer(q),          # None on cat-5
                "adversarial_answer": (q.get("adversarial_answer") or "").strip(),
                "category": cat,
                "qtype": CATEGORY_NAME[cat],
                "judge_type": CATEGORY_JUDGE[cat],
                "evidence": _coerce_evidence(q),
            })
        out.append({
            "id": sample_id, "speaker_a": speaker_a, "speaker_b": speaker_b,
            "sessions": sessions, "session_dates": dates,
            "n_sessions": len(sessions), "evidence_map": evidence_map,
            "qa": qa,
        })
    return out


def sample_questions(convs: list[dict], sample: int, seed: int) -> list[dict]:
    """Global seeded QA sampling. Shuffles the (conversation, question) pool,
    keeps `sample` of them (0 = all), and drops conversations left with no
    questions — those are never ingested. Random-at-n≥100 approximates the
    category mix; use --categories for a targeted slice instead."""
    rng = random.Random(seed)
    refs = [(c["id"], q) for c in convs for q in c["qa"]]
    rng.shuffle(refs)
    if sample:
        refs = refs[:sample]
    keep: dict[str, list[dict]] = defaultdict(list)
    for cid, q in refs:
        keep[cid].append(q)
    out = []
    for c in convs:
        if c["id"] in keep:
            # Restore stable in-file order inside each conversation so runs at
            # the same seed produce identical per-conversation eval order.
            c = dict(c, qa=sorted(keep[c["id"]], key=lambda q: q["qa_id"]))
            out.append(c)
    return out


# ── per-question evaluation ─────────────────────────────────────────────────

def _evidence_diagnostics(q: dict, conv: dict, context_texts: list[str],
                          pool_texts: list[str], rendered: str | None = None) -> dict:
    """Step-0 diagnostics from LoCoMo's OWN evidence annotations (stronger than
    MSC's answer-text heuristic — the gold turn text is known exactly, the
    lexical τ=0.6 only absorbs the 600-char context truncation).

    FOUR nested surfaces, because there are FOUR places evidence can die and
    conflating the last two mislabels a truncation loss as a reader failure:
      gold_in_pool     — all evidence turns retrievable pre-truncation
      gold_in_topk     — survived the `memories[:top_k]` retrieval cut
      gold_in_render   — survived `_render_answer_context`'s MAX_CONTEXT_CHARS
                         budget, i.e. ACTUALLY REACHED THE READER. The renderer
                         `break`s at the first item that overflows, so at wide
                         apertures the top_k list is much larger than the text
                         the model sees; `gold_in_topk` alone silently credits
                         evidence the reader never got.
      gold_in_context  — alias of gold_in_render when a rendered context is
                         supplied (the honest definition), else gold_in_topk.
    `gold_distance` is sessions back from the last session to the FARTHEST-BACK
    evidence turn (1 = final session; -1 = no locatable evidence).

    On cat-5 the evidence points at the TRAP-SOURCE turn, not a gold answer —
    recorded for completeness, excluded from the miss decomposition."""
    ev = [(conv["evidence_map"][e]) for e in q["evidence"] if e in conv["evidence_map"]]
    if not ev:
        return {"gold_in_context": False, "gold_in_pool": False,
                "gold_in_topk": False, "gold_in_render": False,
                "gold_distance": -1, "evidence_in_context_frac": 0.0,
                "n_evidence": 0}
    joined_topk = " ".join(context_texts)
    joined_pool = joined_topk + " " + " ".join(pool_texts)
    in_topk = [_lex_match(text, joined_topk, tau=0.6) for _, text in ev]
    in_pool = [_lex_match(text, joined_pool, tau=0.6) for _, text in ev]
    if rendered is None:
        in_render = in_topk
    else:
        in_render = [_lex_match(text, rendered, tau=0.6) for _, text in ev]
    return {
        "gold_in_context": all(in_render),
        "gold_in_render": all(in_render),
        "gold_in_topk": all(in_topk),
        "gold_in_pool": all(in_pool),
        "gold_distance": conv["n_sessions"] - min(idx for idx, _ in ev),
        "evidence_in_context_frac": sum(in_render) / len(in_render),
        "n_evidence": len(ev),
    }


# The cat-5 judge takes an EXPLANATION of unanswerability, not a gold answer.
# Factored out of evaluate_qa so --rejudge can rebuild the identical judge input
# from a stored results file — a re-judge that reconstructed this differently
# would measure prompt drift, not judge nondeterminism.
def _gold_for_judge(cat: int, answer, trap) -> str:
    if cat != 5:
        return answer if answer is not None else ""
    s = ("The conversation never establishes this — the question's "
         "premise is false or the asked detail was never mentioned. ")
    if trap:
        s += (f"A tempting but WRONG answer (it belongs to a different "
              f"speaker or event) would be: '{trap}'. "
              f"A response giving that answer is incorrect.")
    return s


def evaluate_qa(q: dict, conv: dict, adapter: MSCAdapter, args,
                answer_llm, judge_llm) -> dict:
    cat = q["category"]
    # top_k * 3 at the pipeline layer — the LME driver's multiplier, inherited
    # via MSCAdapter.search (the silently-dropped ×3 was the entire BEAM June
    # regression AND the first 23pp of the MSC arc; never again).
    memories, info = adapter.search(q["question"], top_k=args.top_k * 3)

    # Re-render the EXACT context the answerer will build (same helper, same
    # char caps — cat-2 routes to ability="TR", which doubles the budget) so
    # gold-surface is measured against what the reader actually receives, not
    # against the pre-render top_k list. Pure string work, no LLM call.
    ability = "TR" if cat == 2 else None
    rendered = None
    if not args.sim:
        from longmemeval_adapter import _render_answer_context
        rendered = _render_answer_context(
            memories, ability, info["total_matches"], info["graph_count"],
            info["temporal_events"], info["aggregation_nodes"])

    diag = _evidence_diagnostics(q, conv, [m["content"] for m in memories],
                                 info["pool"], rendered=rendered)

    extra = locomo_perspective_clause(conv["speaker_a"], conv["speaker_b"],
                                      user_is_a=(args.user_speaker == "a"))
    if cat == 3:
        extra += LOCOMO_OPEN_DOMAIN_CLAUSE
    if args.answerable_clause and cat != 5:
        extra += LOCOMO_ANSWERABLE_CLAUSE

    if args.diag_only:
        # Retrieval + render only — no reader, no judge. `correct` stays None
        # because this pass CANNOT produce accuracy; locomo_audit.py joins the
        # dumped surfaces onto a real run by question id. Retrieval is
        # deterministic given the same store, so the top_k reproduces exactly.
        ai, correct = "", None
    elif args.sim:
        # Offline: no answer/judge LLM. "correct" = retrieval surfaced every
        # evidence turn — a retrieval-surface rate, NOT benchmark accuracy
        # (on cat-5 it reports whether the trap-source turn surfaces).
        ai, correct = (memories[0]["content"] if memories else ""), diag["gold_in_context"]
    else:
        from longmemeval_adapter import answer_question, judge_answer
        question_date = conv["session_dates"][-1] if conv["session_dates"] else ""
        ai = answer_question(
            answer_llm, memories, q["question"],
            ability="TR" if cat == 2 else None,
            total_matches=info["total_matches"], graph_count=info["graph_count"],
            temporal_events=info["temporal_events"],
            aggregation_nodes=info["aggregation_nodes"],
            question_date=question_date,
            permissive_default=(cat == 3),
            extra_system=extra)
        gold_for_judge = _gold_for_judge(cat, q["answer"] or "",
                                         q["adversarial_answer"])
        correct = judge_answer(judge_llm, q["judge_type"], q["question"],
                               gold_for_judge, ai)

    rec = {"id": q["qa_id"], "conv_id": conv["id"], "question_type": q["qtype"],
           "category": cat,
           "correct": (None if correct is None else bool(correct)),
           "question": q["question"],
           "answer": q["answer"] if cat != 5 else f"[unanswerable; trap: {q['adversarial_answer']}]",
           "ai_answer": ai, "n_sessions": conv["n_sessions"],
           "evidence": q["evidence"], **diag,
           "n_memories": len(memories), "n_profile": info["n_profile"],
           # Rendered lines carry a [MEM …]/[FACT] tag; counting them measures
           # how many retrieved memories survived the char budget.
           "n_rendered": (None if rendered is None
                          else rendered.count("[MEM") + rendered.count("[FACT"))}
    if (args.dump_context or args.diag_only) and rendered is not None:
        rec["context"] = rendered
    if args.dump_topk or args.diag_only:
        # EXACTLY the string _evidence_diagnostics scores gold_in_topk against,
        # so a strict re-check in the audit runs on the identical haystack.
        # Dumping both this and `context` is the whole point: the surfaces are
        # NESTED (render ⊆ top_k), so gold_in_context=True forces
        # gold_in_topk=True and the boolean pair can never separate a
        # composition loss from a recall loss. Only re-scoring both strings at a
        # strict τ can.
        rec["topk_text"] = " ".join(m["content"] for m in memories)
    return rec


# ── per-conversation driver ─────────────────────────────────────────────────

# Set from --max-context-chars in main(); read by the report line. A list so
# the worker threads and the report share one cell without a global statement.
_MAX_CTX: list[int | None] = [None]


def _aperture(args) -> dict:
    """Lever-L6 retrieval-aperture overrides; None means 'keep the default'."""
    return {"message_fts_top_k": args.message_fts_top_k,
            "rerank_top_k": args.rerank_top_k,
            "fts_top_k": args.fts_top_k,
            "graph_top_k": args.graph_top_k}


def evaluate_conversation(conv: dict, args, answer_llm, judge_llm) -> list[dict]:
    """Ingest one conversation into its own store, then answer its questions.
    With --db-dir the store persists and is REUSED on later runs (ingest+dream
    over 19-32 sessions is the expensive step; QA/prompt iterations shouldn't
    re-pay it). A reused store is only valid for the same core/schema —
    --fresh rebuilds after core changes."""
    if args.db_dir:
        root = Path(args.db_dir) / conv["id"]
        if args.fresh and root.exists():
            shutil.rmtree(root)
        reuse = (root / "hymem.sqlite").exists()
        root.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        root = Path(tempfile.mkdtemp(prefix=f"locomo_{conv['id']}_"))
        reuse, cleanup = False, not args.keep_db

    adapter = MSCAdapter(root / "hymem.sqlite", api_key=args.api_key, sim=args.sim,
                         hymem_model=args.hymem_model, hymem_base_url=args.hymem_base_url,
                         embeddings=args.embeddings, rules_extraction=args.rules_extraction,
                         graph_multihop=args.graph_multihop,
                         aperture=_aperture(args)).open()
    try:
        if reuse:
            print(f"  [{conv['id']}] reusing store at {root}", flush=True)
        else:
            dream_each = args.dream_per_session and not args.no_dream
            adapter.ingest(conv, dream_each=dream_each)
            if not args.no_dream and not dream_each:
                adapter.dream()
        results = []
        for k, q in enumerate(conv["qa"], 1):
            results.append(evaluate_qa(q, conv, adapter, args, answer_llm, judge_llm))
            if k % 20 == 0:
                print(f"  [{conv['id']}] {k}/{len(conv['qa'])}", flush=True)
        # --diag-only writes correct=None (no reader ran), so there is no accuracy
        # to report here — summing it would TypeError, and printing 0.0% would be
        # worse: a run that measured nothing would look like a run that scored zero.
        if args.diag_only:
            surf = sum(bool(r["gold_in_render"]) for r in results) / len(results) if results else 0.0
            print(f"  [{conv['id']}] done — {len(results)} q, gold_in_render "
                  f"{surf*100:.1f}% (tau=0.6, no reader)"
                  f" ({conv['n_sessions']} sessions)", flush=True)
        else:
            acc = sum(r["correct"] for r in results) / len(results) if results else 0.0
            print(f"  [{conv['id']}] done — {len(results)} q, {acc*100:.1f}%"
                  f" ({conv['n_sessions']} sessions)", flush=True)
        return results
    finally:
        adapter.close()
        if cleanup:
            shutil.rmtree(root, ignore_errors=True)


# ── reporting ───────────────────────────────────────────────────────────────

_DIST_BUCKETS = [(1, 1, "1 back"), (2, 3, "2-3 back"), (4, 7, "4-7 back"),
                 (8, 15, "8-15 back"), (16, 10 ** 9, "16+ back")]


def _compute_scores_local(results: list[dict]) -> dict:
    """Fallback mirror of longmemeval_adapter.compute_scores (accuracy by
    question_type, _abs folded into the base type) so --sim runs with zero API
    deps — importing the LME module pulls in `requests` at module level."""
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"].replace("_abs", "")].append(r["correct"])
    scores = {t: {"accuracy": sum(c) / len(c), "count": len(c)}
              for t, c in by_type.items()}
    all_c = [c for cs in by_type.values() for c in cs]
    scores["OVERALL"] = {"accuracy": sum(all_c) / len(all_c) if all_c else 0.0,
                         "count": len(all_c)}
    return scores


def _print_report(results: list[dict], args) -> None:
    try:
        from longmemeval_adapter import (compute_scores, compute_abstention_scores,
                                         print_abstention_scores)
    except ImportError:  # offline --sim without the LME module's HTTP deps
        compute_scores = _compute_scores_local
        compute_abstention_scores = print_abstention_scores = None
    scores = compute_scores(results)
    ov = scores.pop("OVERALL")
    print(f"\n=== LoCoMo — n={ov['count']} ===")
    if args.sim:
        print("  [SIM] 'accuracy' below = retrieval-surface rate, not benchmark accuracy")
    if args.answerable_clause:
        print("  [NON-CANONICAL] --answerable-clause is label-leaky "
              "(conditions the prompt on the adversarial label)")
    print(f"  overall accuracy: {ov['accuracy']*100:.1f}%")
    # Stamp the aperture: a run is only comparable to another at the SAME one,
    # and message_fts_top_k is the hard ceiling on gold-turn surfacing.
    ap_eff = {**MSCAdapter.APERTURE,
              **{k: v for k, v in _aperture(args).items() if v is not None}}
    print(f"  aperture: msg={ap_eff['message_fts_top_k']} "
          f"rerank_pool={ap_eff.get('rerank_top_k', 20)} "
          f"chunk={ap_eff['fts_top_k']} graph={ap_eff['graph_top_k']} "
          f"cut={args.top_k * 3}  embeddings={'on' if args.embeddings else 'off'}\n")
    print(f"  {'category':<16} {'acc':>7} {'n':>6}")
    for name, s in sorted(scores.items()):
        print(f"  {name:<16} {s['accuracy']*100:>6.1f}% {s['count']:>6}")

    if print_abstention_scores and any(r["question_type"].endswith("_abs")
                                       for r in results):
        print_abstention_scores(compute_abstention_scores(results))

    # E1 analogue: accuracy vs sessions-back to the farthest evidence turn,
    # bucketed (LoCoMo distances run 1..32). Answerable cats only — cat-5
    # "evidence" is the trap source, not a gold location.
    answerable = [r for r in results if r["category"] != 5]
    if answerable:
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in answerable:
            d = r.get("gold_distance", -1)
            label = "unknown" if d < 0 else next(
                lb for lo, hi, lb in _DIST_BUCKETS if lo <= d <= hi)
            by_bucket[label].append(r)
        order = ["unknown"] + [lb for _, _, lb in _DIST_BUCKETS]
        print("\n  ── recall vs session distance (+ gold-surface diagnostics; "
              "evidence-based, lexical τ=0.6) ──")
        print(f"  {'distance':>9} {'acc':>7} {'in-ctx':>7} {'in-pool':>8} {'n':>5}")
        for label in order:
            rs = by_bucket.get(label)
            if not rs:
                continue
            acc = sum(r["correct"] for r in rs) / len(rs)
            ctx = sum(r["gold_in_context"] for r in rs) / len(rs)
            pool = sum(r["gold_in_pool"] for r in rs) / len(rs)
            print(f"  {label:>9} {acc*100:>6.1f}% {ctx*100:>6.1f}% {pool*100:>7.1f}% {len(rs):>5}")

        misses = [r for r in answerable if not r["correct"]]
        if misses:
            # Four buckets, not three: `budget` is evidence that won the
            # retrieval cut and was then dropped by MAX_CONTEXT_CHARS before the
            # reader saw it. It used to be counted as synthesis, which reads as
            # "the reader failed" when in fact the reader was never shown it —
            # and that mislabel grows with the aperture.
            retrieval = sum(not r["gold_in_pool"] for r in misses)
            ranking = sum(r["gold_in_pool"] and not r.get("gold_in_topk", r["gold_in_context"])
                          for r in misses)
            budget = sum(r.get("gold_in_topk", r["gold_in_context"])
                         and not r["gold_in_context"] for r in misses)
            synthesis = sum(r["gold_in_context"] for r in misses)
            n = len(misses)
            print(f"\n  ── miss decomposition ({n} answerable-cat misses) ──")
            print(f"  retrieval loss   (evidence in neither pool nor ctx): {retrieval:>3}  ({retrieval/n*100:.0f}%)")
            print(f"  ranking/cut loss (evidence in pool, not in top_k):   {ranking:>3}  ({ranking/n*100:.0f}%)")
            print(f"  budget loss      (in top_k, cut by context chars):   {budget:>3}  ({budget/n*100:.0f}%)")
            print(f"  synthesis/judge  (evidence REACHED reader, wrong):   {synthesis:>3}  ({synthesis/n*100:.0f}%)")

        # How much of the retrieved list actually reaches the reader. At a wide
        # aperture with an unchanged char budget this is the binding constraint,
        # and it is invisible in every other line of the report.
        rendered_frac = [r["n_rendered"] / r["n_memories"] for r in results
                         if r.get("n_rendered") is not None and r.get("n_memories")]
        if rendered_frac:
            print(f"\n  context budget: {sum(rendered_frac)/len(rendered_frac)*100:.0f}% "
                  f"of retrieved memories survive MAX_CONTEXT_CHARS "
                  f"({_MAX_CTX[0] or 'default'} chars, x2 on cat-2/TR)")

    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_conv[r["conv_id"]].append(r)
    print(f"\n  {'conversation':<12} {'acc':>7} {'n':>6}")
    for cid in sorted(by_conv):
        rs = by_conv[cid]
        print(f"  {cid:<12} {sum(r['correct'] for r in rs)/len(rs)*100:>6.1f}% {len(rs):>6}")

    n_prof = [r.get("n_profile", 0) for r in results]
    if n_prof:
        print(f"\n  profile tier: {sum(n_prof)/len(n_prof):.1f} entries/question avg "
              f"({sum(1 for p in n_prof if p == 0)} questions saw zero)")


# ── main ────────────────────────────────────────────────────────────────────

# ── Re-judge (split the churn floor into reader share vs judge share) ───────
# Identical-config reruns move ~10 of 200 questions even though answer AND judge
# both run at temperature=0.0 (spec §8). Two nondeterministic LLMs sit in that
# loop and the accuracy line cannot separate them. Re-judging ONE stored answer
# file with the SAME judge holds the reader fixed: every flip that survives is
# the judge's share, and the remainder is the reader's. If the judge dominates,
# majority-of-3 judging shrinks the floor for the whole triad at once.
#
# LME has the same facility (`longmemeval_adapter.py:_rejudge_run`) but reads
# `hypothesis` out of a {config, per_question} envelope; LoCoMo `--out` writes a
# bare list keyed on `ai_answer`, hence this shim rather than a shared call.

_TRAP_RE = re.compile(r"^\[unanswerable; trap: (.*)\]$", re.S)


def _rejudge_file(args, judge_llm) -> None:
    """Re-judge a stored `--out` file, writing a flip-compatible copy."""
    from longmemeval_adapter import judge_answer

    rows = json.loads(Path(args.rejudge).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        sys.exit(f"{args.rejudge}: expected a non-empty list of per-question results")

    print(f"\n=== LoCoMo RE-JUDGE — {Path(args.rejudge).name} ===")
    print(f"  rows: {len(rows)}   judge: {args.judge_model}"
          + (f"  +extra_body={args.judge_extra_body}" if args.judge_extra_body else ""))
    print("  reader output is held FIXED — every flip below is judge nondeterminism\n",
          flush=True)

    def _rj(r: dict) -> dict:
        cat, ai = r["category"], str(r.get("ai_answer") or "")
        if not ai or ai.startswith("[LLM_ERROR"):
            new, judged = bool(r.get("correct")), False   # nothing judgeable
        else:
            gold = r.get("answer")
            if cat == 5:
                m = _TRAP_RE.match(str(gold))
                gold = _gold_for_judge(5, None, m.group(1) if m else "")
            new = judge_answer(judge_llm, CATEGORY_JUDGE[cat], r["question"],
                               gold if gold is not None else "", ai)
            judged = True
        return {**r, "correct": new, "correct_original": r.get("correct"),
                "_rejudged": judged}

    out_rows: list[dict] = [None] * len(rows)
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_rj, r): i for i, r in enumerate(rows)}
            for done, fut in enumerate(as_completed(futs), 1):
                out_rows[futs[fut]] = fut.result()
                if done % 25 == 0:
                    print(f"  ── re-judged {done}/{len(rows)}", flush=True)
    else:
        for i, r in enumerate(rows):
            out_rows[i] = _rj(r)

    judged = [r for r in out_rows if r["_rejudged"]]
    flipped = [r for r in judged if bool(r["correct"]) != bool(r["correct_original"])]
    t_to_f = [r for r in flipped if r["correct_original"]]
    o = sum(bool(r["correct_original"]) for r in out_rows) / len(out_rows)
    n = sum(bool(r["correct"]) for r in out_rows) / len(out_rows)
    print(f"\n  original: {o*100:.1f}%   re-judged: {n*100:.1f}%   ({(n-o)*100:+.1f}pp)")
    print(f"  judge churn: {len(flipped)}/{len(judged)} judged rows flipped "
          f"({len(flipped)/max(len(judged),1)*100:.1f}%)   "
          f"[{len(t_to_f)} correct→wrong, {len(flipped)-len(t_to_f)} wrong→correct]")
    if len(judged) < len(out_rows):
        print(f"  ({len(out_rows)-len(judged)} rows unjudgeable — kept prior verdict)")
    by_cat = defaultdict(lambda: [0, 0])
    for r in judged:
        c = by_cat[CATEGORY_NAME[r["category"]].replace("_abs", "")]
        c[1] += 1
        c[0] += bool(r["correct"]) != bool(r["correct_original"])
    print(f"\n  {'category':<14} {'flipped':>8} {'n':>5}")
    for name in sorted(by_cat):
        f, tot = by_cat[name]
        print(f"  {name:<14} {f:>8} {tot:>5}")

    # Written in the SAME bare-list shape as --out, so locomo_flip.py compares
    # this against the source file directly: that flip run IS the judge share.
    dest = args.out or str(Path(args.rejudge).with_suffix(".rejudged.json"))
    Path(dest).write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    print(f"\n  re-judged results → {dest}")
    print(f"  compare: python locomo_flip.py {args.rejudge} {dest}")


def _build_llm(model, base_url, api_key, extra_body):
    import os
    from longmemeval_adapter import LLMClient
    return LLMClient(model=model, api_key=api_key or os.environ.get("HYMEM_LLM_API_KEY", ""),
                     base_url=base_url, extra_body=extra_body)


def main() -> None:
    ap = argparse.ArgumentParser(description="HyMem LoCoMo benchmark adapter.")
    ap.add_argument("--data", default=None, help="locomo10.json (snap-research/locomo shape)")
    ap.add_argument("--sample", type=int, default=0,
                    help="global QA cap after seeded shuffle; 0 = all 1986")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--categories", default=None, metavar="1,2,4",
                    help="comma-separated LoCoMo categories to keep (default all; "
                         "5 = adversarial)")
    ap.add_argument("--convs", default=None, metavar="conv-26,conv-30",
                    help="restrict to these sample_ids")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel CONVERSATIONS (each owns its store; ≤10 useful)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="base K; the pipeline searches top_k*3 like the LME driver")
    # Lever L6 — retrieval aperture. The MSC-sized defaults (15/10/10, rerank
    # pool 20) surface ~15 of a 369-689-turn LoCoMo history, and `message_hits`
    # is the ONLY tier that can carry a gold *turn* to the reader. Note
    # --rerank-top-k must stay comfortably ABOVE --message-fts-top-k or the
    # reranker has no room to lift a weak-lexical turn into the cut (at the
    # defaults it reranks 20 down to 15 — it can drop 5 items).
    ap.add_argument("--message-fts-top-k", type=int, default=None,
                    help="raw-turn slots surfaced (default 15)")
    ap.add_argument("--rerank-top-k", type=int, default=None,
                    help="BM25 candidate pool fed to the reranker (default 20)")
    ap.add_argument("--fts-top-k", type=int, default=None,
                    help="dreamed-chunk slots (default 10)")
    ap.add_argument("--graph-top-k", type=int, default=None,
                    help="graph-fact slots (default 10)")
    ap.add_argument("--max-context-chars", type=int, default=None,
                    help="reader context budget (LME default 8000; doubled for "
                         "cat-2/TR). Must scale WITH the aperture — a wider "
                         "top_k against an unchanged budget is truncated away "
                         "by _render_answer_context before the reader sees it")
    ap.add_argument("--user-speaker", choices=["a", "b"], default="a",
                    help="which speaker HyMem models as the user (default speaker_a)")
    ap.add_argument("--name-prefix", action="store_true",
                    help="prepend 'Name: ' to each ingested turn (lever L2: lets "
                         "FTS match speaker names in questions; changes extraction "
                         "input, so non-canonical until A/B'd)")
    ap.add_argument("--answerable-clause", action="store_true",
                    help="A/B lever: MSC-style answerability clause on cats 1-4. "
                         "LABEL-LEAKY (see spec) — never canonical")
    ap.add_argument("--answer-model", default=_ANSWER_MODEL)
    ap.add_argument("--answer-base-url", default=_DEEPSEEK_BASE_URL)
    ap.add_argument("--answer-api-key", default=None)
    ap.add_argument("--answer-extra-body", default=None, metavar="JSON",
                    help='e.g. \'{"thinking":{"type":"disabled"}}\' for v4-flash')
    ap.add_argument("--judge-model", default=_JUDGE_MODEL)
    ap.add_argument("--judge-extra-body", default=None, metavar="JSON")
    ap.add_argument("--hymem-model", default=_HYMEM_MODEL, help="HyMem's dream LLM")
    ap.add_argument("--hymem-base-url", default=_DEEPSEEK_BASE_URL)
    ap.add_argument("--api-key", default="", help="HyMem dream LLM key")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--rules-extraction", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--graph-multihop", action="store_true",
                    help="Track-A BFS — cat 1 (multi-hop) is the A/B target")
    ap.add_argument("--no-dream", action="store_true")
    ap.add_argument("--dream-per-session", action="store_true",
                    help="dream after EACH of the 19-32 sessions (live-store "
                         "posture; expensive — default is one dream at the end)")
    ap.add_argument("--db-dir", default=None,
                    help="persist per-conversation stores here and REUSE them on "
                         "later runs (skips ingest+dream). Clear or --fresh after "
                         "core/schema changes")
    ap.add_argument("--fresh", action="store_true",
                    help="with --db-dir: rebuild stores instead of reusing")
    ap.add_argument("--keep-db", action="store_true")
    ap.add_argument("--out", default=None, help="write per-question results JSON here")
    ap.add_argument("--dump-context", action="store_true",
                    help="include the exact rendered answer context in each result")
    ap.add_argument("--dump-topk", action="store_true",
                    help="include the joined top_k memory text (the haystack "
                         "gold_in_topk is scored against) in each result")
    ap.add_argument("--diag-only", action="store_true",
                    help="retrieval + render only: NO answering, NO judging, so "
                         "it costs no reader calls. Implies --dump-context and "
                         "--dump-topk and writes correct=null. Join it onto a "
                         "real run by question id (locomo_audit.py --topk-dump) "
                         "to re-score the gold surfaces at a strict tau")
    ap.add_argument("--rejudge", default=None, metavar="RESULTS.json",
                    help="re-judge a stored --out file with the SAME reader output "
                         "(no ingest, no answering) and report judge-only churn; "
                         "writes a flip-compatible copy to --out or *.rejudged.json")
    ap.add_argument("--sim", action="store_true", help="offline: StubLLM, no API")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.diag_only:
        # --sim leaves `rendered` None, which is the one surface this pass exists
        # to capture; a sim diag dump would silently be topk-only.
        if args.sim:
            sys.exit("--diag-only needs the real renderer; drop --sim.")
        if not args.out:
            sys.exit("--diag-only produces a sidecar to join; pass --out FILE.")
    if args.rejudge:
        if args.sim:
            sys.exit("--rejudge needs a real judge; drop --sim.")
        jb = json.loads(args.judge_extra_body) if args.judge_extra_body else None
        _rejudge_file(args, _build_llm(args.judge_model, _DEEPSEEK_BASE_URL,
                                       args.answer_api_key, jb))
        return

    categories = ({int(c) for c in args.categories.split(",")}
                  if args.categories else None)
    convs = load_locomo_data(args.data, user_speaker=args.user_speaker,
                             categories=categories, name_prefix=args.name_prefix)
    if args.convs:
        keep = {c.strip() for c in args.convs.split(",")}
        convs = [c for c in convs if c["id"] in keep]
    convs = sample_questions(convs, args.sample, args.seed)
    n_q = sum(len(c["qa"]) for c in convs)
    if not n_q:
        print("No LoCoMo questions selected.")
        sys.exit(1)
    print(f"Loaded {len(convs)} conversations, {n_q} questions "
          f"({sum(c['n_sessions'] for c in convs)} sessions total)"
          f"{'  [SIM]' if args.sim else ''}", flush=True)

    answer_llm = judge_llm = None
    if args.max_context_chars:
        _MAX_CTX[0] = args.max_context_chars
        if not args.sim:
            # MAX_CONTEXT_CHARS is read as a module global inside
            # _render_answer_context, so rebinding it here covers both the
            # answer path and the diagnostic re-render.
            import longmemeval_adapter as _lme
            _lme.MAX_CONTEXT_CHARS = args.max_context_chars
    if not args.sim and not args.diag_only:
        ab = json.loads(args.answer_extra_body) if args.answer_extra_body else None
        jb = json.loads(args.judge_extra_body) if args.judge_extra_body else None
        answer_llm = _build_llm(args.answer_model, args.answer_base_url, args.answer_api_key, ab)
        judge_llm = _build_llm(args.judge_model, _DEEPSEEK_BASE_URL, args.answer_api_key, jb)

    results: list[dict] = []
    t0 = time.time()
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(convs))) as pool:
            futs = [pool.submit(evaluate_conversation, c, args, answer_llm, judge_llm)
                    for c in convs]
            for fut in as_completed(futs):
                results.extend(fut.result())
    else:
        for c in convs:
            results.extend(evaluate_conversation(c, args, answer_llm, judge_llm))
    print(f"  done in {time.time()-t0:.0f}s")

    if args.json:
        print(json.dumps(results, indent=2))
    elif args.diag_only:
        ans = [r for r in results if r["category"] != 5]
        n = len(ans) or 1
        print(f"\n  ── diagnostics-only pass ({len(ans)} answerable-cat questions, "
              f"no reader, no judge) ──")
        for k in ("gold_in_pool", "gold_in_topk", "gold_in_render"):
            print(f"  {k:<16} {sum(bool(r[k]) for r in ans)/n*100:>5.1f}%  (tau=0.6)")
        print("  These are the LEXICAL surfaces and they are NESTED — read them "
              "only after\n  a strict re-score. Join onto a real run:\n"
              f"    python locomo_audit.py REAL_RUN.json --data {args.data} "
              f"--topk-dump {args.out}")
    else:
        _print_report(results, args)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  per-question results → {args.out}")


# A tiny in-schema fixture: 2 speakers, 3 dated sessions, one photo-share turn,
# and one QA per category — including the real file's quirks (string category,
# string-repr evidence, int answer) so --sim exercises every coercion.
_SIM_FIXTURE = [{
    "sample_id": "sim-1",
    "conversation": {
        "speaker_a": "Ada", "speaker_b": "Ben",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Ada", "dia_id": "D1:1",
             "text": "I finally signed up for the pottery class downtown!"},
            {"speaker": "Ben", "dia_id": "D1:2",
             "text": "Nice! I spent the weekend fixing my old motorbike.",
             "img_url": ["http://example.com/bike.jpg"],
             "blip_caption": "a red motorbike in a garage"},
        ],
        "session_2_date_time": "10:04 am on 19 June, 2023",
        "session_2": [
            {"speaker": "Ada", "dia_id": "D2:1",
             "text": "The pottery class is going great, I made three bowls in 2022... "
                     "no wait, I made three bowls already this month."},
            {"speaker": "Ben", "dia_id": "D2:2",
             "text": "I sold the motorbike and bought a bicycle instead."},
        ],
        "session_3_date_time": "9:30 pm on 2 July, 2023",
        "session_3": [
            {"speaker": "Ada", "dia_id": "D3:1",
             "text": "I'm thinking of selling my bowls at the summer market."},
            {"speaker": "Ben", "dia_id": "D3:2",
             "text": "Cycling to work daily now — 20 minutes each way."},
        ],
    },
    "qa": [
        {"question": "What does Ada plan to do with her bowls, and where did she learn to make them?",
         "answer": "Sell them at the summer market; she learned at the pottery class",
         "evidence": ["D3:1", "D1:1"], "category": 1},
        {"question": "When did Ada sign up for the pottery class?",
         "answer": "8 May 2023", "evidence": ["D1:1"], "category": 2},
        {"question": "Would Ben likely enjoy a cycling holiday?",
         "answer": "Yes, he cycles to work daily", "evidence": ["D3:2"], "category": 3},
        {"question": "How many bowls did Ada make?",
         "answer": 3, "evidence": ["D2:1"], "category": 4},
        {"question": "What did Ada do with her motorbike?",
         "adversarial_answer": "sold it and bought a bicycle",
         "evidence": "['D2:2']", "category": "5"},
    ],
}]


if __name__ == "__main__":
    main()
