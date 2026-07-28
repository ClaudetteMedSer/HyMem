"""Idea B write-side — the data-driven experiment engine for marker→rule routing.

The lexical classifier caps at ~14% precision on real markers because
standing-vs-one-off is semantic, not lexical (see `rule_extraction_probe.py`).
This harness scores every candidate DESIGN head-to-head against ONE labeled
corpus so the winning arm is chosen from data, not taste, and then re-validated
on the box's real markers.

Design space (the A/B options):
  MODE       lexical | llm | llm_fastpath          (the routing instrument)
  τ  (tau)   LLM standing-confidence threshold      (precision/recall trade-off)
  PROMOTION  immediate | and | recurrence | or_highconf   (repetition-gated, option A)
  N          min distinct sessions for recurrence   (how much corroboration)

Two evaluation layers, one judgment pass:
  • Per-MARKER (E1/E2): does the instrument label THIS marker correctly? Sweeps τ.
  • Per-POLICY (E3): after grouping paraphrases into policies and counting how many
    distinct sessions each recurs in, does promotion pick the right policies? Sweeps
    N × PROMOTION. Repetition can RESCUE modal-less standing policies the classifier
    misses AND filter one-offs that slipped through.

Cost model: the LLM judge is called ONCE per mode (a single batched durability
pass); every τ and every N point is derived analytically from that judgment
table — the whole sweep is one call. `--sim` swaps in a deterministic `SimJudge`
(a fake LLM that knows the gold labels with tunable noise) so the mechanics run
locally with no API; the box passes a real `--answer-model` for real numbers.

Gate: precision ≥ threshold (default 0.90). Recall is reported. The harness
prints the best arm clearing the gate at max recall, and the full grid.

Usage:
  python benchmarks/rule_extraction_experiment.py --sim                 # offline mechanics
  python benchmarks/rule_extraction_experiment.py --sim --sim-noise 0.1 # noisier judge
  python benchmarks/rule_extraction_experiment.py --labels real.json \\
      --answer-model deepseek-v4-flash --answer-base-url <url>          # box, real judge
  python benchmarks/rule_extraction_experiment.py --labels real.json --json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from hymem import rules_extract
from hymem.dreaming.canonicalize import normalize
from hymem.extraction.llm import LLMRequest
from hymem.rules import rule_scope_for_marker


# ─────────────────────────────────────────────────────────────────────────────
# Corpus. Each item: kind, statement, is_rule (gold), session_id, policy_id.
# `policy_id` groups paraphrases of ONE standing policy (recurrence is counted
# over distinct session_id within a policy). The synthetic set mirrors the real
# finding: standing policies that RECUR, standing policies stated ONCE, and
# one-off corrections/rejections with incidental modals that appear once.
# ─────────────────────────────────────────────────────────────────────────────
SYNTHETIC: list[dict] = [
    # standing policies that recur across sessions (recurrence + LLM both catch)
    {"kind": "rejection", "statement": "The user rejects using MongoDB", "is_rule": True, "session_id": "s1", "policy_id": "no_mongo"},
    {"kind": "rejection", "statement": "Avoid MongoDB, standardize on Postgres", "is_rule": True, "session_id": "s4", "policy_id": "no_mongo"},
    {"kind": "rejection", "statement": "Don't reach for MongoDB again", "is_rule": True, "session_id": "s7", "policy_id": "no_mongo"},
    {"kind": "correction", "statement": "Always run the tests before pushing", "is_rule": True, "session_id": "s2", "policy_id": "test_before_push"},
    {"kind": "correction", "statement": "Run the full test suite before every push", "is_rule": True, "session_id": "s5", "policy_id": "test_before_push"},
    {"kind": "style", "statement": "Write commit messages in the imperative mood", "is_rule": True, "session_id": "s3", "policy_id": "imperative_commits"},
    {"kind": "style", "statement": "Commit messages should be imperative mood", "is_rule": True, "session_id": "s8", "policy_id": "imperative_commits"},
    # standing policies stated ONCE (LLM catches; recurrence-only misses at N≥2)
    {"kind": "rejection", "statement": "The user refuses to use Jira", "is_rule": True, "session_id": "s6", "policy_id": "no_jira"},
    {"kind": "style", "statement": "Respond concisely, no preamble", "is_rule": True, "session_id": "s9", "policy_id": "concise"},
    {"kind": "correction", "statement": "Never force-push to a shared branch", "is_rule": True, "session_id": "s10", "policy_id": "no_force_push"},
    # one-off rejections/corrections with incidental modals — the real FPs (once each)
    {"kind": "rejection", "statement": "The user rejects the LOWER() patch for HyMem", "is_rule": False, "session_id": "s2", "policy_id": "oneoff_lower"},
    {"kind": "rejection", "statement": "The user rejects LoCoMo as a benchmark", "is_rule": False, "session_id": "s3", "policy_id": "oneoff_locomo"},
    {"kind": "rejection", "statement": "The user rejects the mega-store approach", "is_rule": False, "session_id": "s5", "policy_id": "oneoff_megastore"},
    {"kind": "correction", "statement": "The podcast was in Dutch instead of English", "is_rule": False, "session_id": "s6", "policy_id": "oneoff_podcast"},
    {"kind": "correction", "statement": "It should be automatic, not manual", "is_rule": False, "session_id": "s7", "policy_id": "oneoff_automatic"},
    {"kind": "correction", "statement": "The conclusions don't match the code state", "is_rule": False, "session_id": "s8", "policy_id": "oneoff_conclusions"},
    {"kind": "correction", "statement": "The flag silently requires four env vars", "is_rule": False, "session_id": "s9", "policy_id": "oneoff_flag"},
    {"kind": "correction", "statement": "The deadline is March 3, not February 28", "is_rule": False, "session_id": "s1", "policy_id": "oneoff_deadline"},
    {"kind": "correction", "statement": "The staging URL is stage.acme.io", "is_rule": False, "session_id": "s4", "policy_id": "oneoff_url"},
    # preferences — never a rule by design
    {"kind": "preference", "statement": "Prefers dark mode", "is_rule": False, "session_id": "s2", "policy_id": "pref_dark"},
    {"kind": "preference", "statement": "Likes working in the morning", "is_rule": False, "session_id": "s3", "policy_id": "pref_morning"},
]


def load_corpus(path: str | None) -> list[dict]:
    if not path:
        return SYNTHETIC
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("labels file must be a JSON array of marker objects")
    out = []
    for i, item in enumerate(data):
        if "kind" not in item or "statement" not in item or "is_rule" not in item:
            raise ValueError(f"item {i} needs kind/statement/is_rule")
        out.append({
            "kind": item["kind"],
            "statement": item["statement"],
            "is_rule": bool(item["is_rule"]),
            # session/policy are optional: recurrence sim degrades to one-per-item.
            "session_id": item.get("session_id", f"_sess_{i}"),
            "policy_id": item.get("policy_id") or normalize(item["statement"]),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SimJudge — a deterministic fake LLM for offline runs. It parses the batched
# durability prompt, looks each marker up in the gold map, and emits a verdict
# with tunable label noise + a confidence model (standing rules score high,
# one-offs low). This exercises the REAL rules_extract pipeline end to end.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SimJudge:
    gold: dict[str, dict]      # statement -> corpus item
    noise: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def complete(self, request: LLMRequest) -> str:
        payload = json.loads(request.user.split("Markers:", 1)[1].rsplit("Return", 1)[0].strip())
        verdicts = []
        for m in payload:
            item = self.gold.get(m["statement"], {})
            gold_rule = bool(item.get("is_rule", False))
            standing = gold_rule
            if self._rng.random() < self.noise:      # flip with prob=noise
                standing = not standing
            # confidence: crisp near the truth, fuzzier under noise.
            base = 0.92 if standing else 0.15
            conf = min(1.0, max(0.0, base + self._rng.uniform(-0.12, 0.12)))
            verdicts.append({
                "index": m["index"],
                "standing": standing,
                "confidence": round(conf, 3),
                "rule": (item.get("policy_id") or m["statement"]) if standing else None,
            })
        return json.dumps(verdicts)


# ─────────────────────────────────────────────────────────────────────────────
# One judgment pass → a per-marker table; every τ/N point derives from it.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MarkerJudgment:
    item: dict
    lexical_hit: bool
    standing: bool        # raw LLM verdict (or lexical hit in lexical mode)
    confidence: float
    canonical: str


def judge_corpus(corpus: list[dict], mode: str, llm, batch_size: int = 20) -> list[MarkerJudgment]:
    markers = [(c["kind"], c["statement"]) for c in corpus]
    lexical = [rule_scope_for_marker(k, s) is not None for k, s in markers]

    if mode == "lexical":
        return [
            MarkerJudgment(item=c, lexical_hit=lh, standing=lh,
                           confidence=1.0 if lh else 0.0, canonical=c["statement"])
            for c, lh in zip(corpus, lexical)
        ]

    # llm / llm_fastpath: judge the markers the fastpath doesn't shortcut.
    need = [i for i in range(len(markers))
            if not (mode == "llm_fastpath" and lexical[i])]
    judged: dict[int, rules_extract.DurabilityJudgment] = {}
    if need and llm is not None:
        sub = rules_extract.judge_durability_batch(
            llm, [markers[i] for i in need], batch_size=batch_size)
        judged = {need[j]: sub[j] for j in range(len(sub))}

    out: list[MarkerJudgment] = []
    for i, c in enumerate(corpus):
        if mode == "llm_fastpath" and lexical[i]:
            out.append(MarkerJudgment(item=c, lexical_hit=True, standing=True,
                                      confidence=1.0, canonical=c["statement"]))
            continue
        j = judged.get(i)
        if j is None:
            out.append(MarkerJudgment(item=c, lexical_hit=lexical[i], standing=False,
                                      confidence=0.0, canonical=c["statement"]))
            continue
        out.append(MarkerJudgment(item=c, lexical_hit=lexical[i], standing=j.standing,
                                   confidence=j.confidence, canonical=j.rule or c["statement"]))
    return out


def _routes(j: MarkerJudgment, mode: str, tau: float) -> bool:
    if mode == "lexical":
        return j.lexical_hit
    if mode == "llm_fastpath" and j.lexical_hit:
        return True
    return j.standing and j.confidence >= tau


def _metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def marker_metrics(judgments: list[MarkerJudgment], mode: str, tau: float) -> dict:
    tp = fp = fn = 0
    for j in judgments:
        routed = _routes(j, mode, tau)
        gold = bool(j.item["is_rule"])
        tp += routed and gold
        fp += routed and not gold
        fn += (not routed) and gold
    return _metrics(tp, fp, fn)


def policy_metrics(judgments: list[MarkerJudgment], mode: str, tau: float,
                   *, promotion: str, n: int, tau_high: float) -> dict:
    """Per-policy promotion metrics. A policy is promoted per `promotion`:
      immediate   — any marker routes.
      and         — any marker routes AND recurs in ≥ n distinct sessions.
      recurrence  — recurs in ≥ n distinct sessions (classifier ignored).
      or_highconf — (routes at conf ≥ tau_high) OR recurs in ≥ n sessions.
    """
    policies: dict[str, dict] = {}
    for j in judgments:
        pid = j.item["policy_id"]
        p = policies.setdefault(pid, {"gold": False, "sessions": set(),
                                      "routed": False, "high": False})
        p["gold"] = p["gold"] or bool(j.item["is_rule"])
        p["sessions"].add(j.item["session_id"])
        if _routes(j, mode, tau):
            p["routed"] = True
            if mode != "lexical" and j.confidence >= tau_high:
                p["high"] = True
            elif mode == "lexical":
                p["high"] = True   # lexical hits are treated as high-confidence

    tp = fp = fn = 0
    for p in policies.values():
        recurs = len(p["sessions"]) >= n
        if promotion == "immediate":
            promote = p["routed"]
        elif promotion == "and":
            promote = p["routed"] and recurs
        elif promotion == "recurrence":
            promote = recurs
        elif promotion == "or_highconf":
            promote = (p["routed"] and p["high"]) or recurs
        else:
            raise ValueError(f"unknown promotion {promotion!r}")
        tp += promote and p["gold"]
        fp += promote and not p["gold"]
        fn += (not promote) and p["gold"]
    m = _metrics(tp, fp, fn)
    m["promotion"] = promotion
    m["n"] = n
    m["policies"] = len(policies)
    return m


def _build_llm(model, base_url, api_key):
    if model == "stub" or model is None:
        return None
    from hymem.contrib.openai_client import OpenAICompatibleClient
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)


def main() -> None:
    ap = argparse.ArgumentParser(description="Idea B marker→rule design experiment.")
    ap.add_argument("--labels", default=None, help="JSON corpus (else built-in synthetic set)")
    ap.add_argument("--modes", default="lexical,llm,llm_fastpath",
                    help="comma list of modes to score")
    ap.add_argument("--tau-sweep", default="0.5,0.6,0.7,0.75,0.8,0.85,0.9",
                    help="LLM confidence thresholds to sweep")
    ap.add_argument("--evidence-sweep", default="1,2,3",
                    help="recurrence N (min distinct sessions) to sweep")
    ap.add_argument("--tau-high", type=float, default=0.9,
                    help="confidence bar for the 'or_highconf' single-shot promotion")
    ap.add_argument("--threshold", type=float, default=0.90, help="precision gate")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="markers per durability call (a mega-batch collapses the judge)")
    ap.add_argument("--policy-from-canonical", action="store_true",
                    help="group policies by the LLM canonical rule (paraphrase-robust) "
                         "rather than the corpus policy_id — unblocks E3 when the dump "
                         "carries session_id but no policy_id")
    ap.add_argument("--sim", action="store_true", help="use the offline SimJudge (no API)")
    ap.add_argument("--sim-noise", type=float, default=0.0, help="SimJudge label-flip prob")
    ap.add_argument("--answer-model", default=None)
    ap.add_argument("--answer-base-url", default=None)
    ap.add_argument("--answer-api-key", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(args.labels)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    taus = [float(x) for x in args.tau_sweep.split(",") if x.strip()]
    ns = [int(x) for x in args.evidence_sweep.split(",") if x.strip()]
    gold_map = {c["statement"]: c for c in corpus}

    def llm_for(mode):
        if mode == "lexical":
            return None
        if args.sim:
            return SimJudge(gold_map, noise=args.sim_noise)
        return _build_llm(args.answer_model, args.answer_base_url, args.answer_api_key)

    marker_rows: list[dict] = []
    policy_rows: list[dict] = []
    for mode in modes:
        judgments = judge_corpus(corpus, mode, llm_for(mode), batch_size=args.batch_size)
        if args.policy_from_canonical:
            # collapse paraphrases: a policy is the LLM's canonical rule text, so
            # "reject Mongo"/"avoid MongoDB" share a policy and their sessions sum.
            for j in judgments:
                j.item["policy_id"] = normalize(j.canonical)
        sweep_taus = [1.0] if mode == "lexical" else taus
        for tau in sweep_taus:
            mm = marker_metrics(judgments, mode, tau)
            mm.update({"mode": mode, "tau": tau})
            marker_rows.append(mm)
            for promotion in ("immediate", "and", "recurrence", "or_highconf"):
                sweep_ns = [1] if promotion == "immediate" else ns
                for n in sweep_ns:
                    pm = policy_metrics(judgments, mode, tau, promotion=promotion,
                                        n=n, tau_high=args.tau_high)
                    pm.update({"mode": mode, "tau": tau})
                    policy_rows.append(pm)

    # Best arm: clears the precision gate at max recall (marker layer, then policy).
    def _best(rows):
        passing = [r for r in rows if r["precision"] >= args.threshold]
        return max(passing, key=lambda r: (r["recall"], r["f1"]), default=None)

    best_marker = _best(marker_rows)
    best_policy = _best(policy_rows)

    if args.json:
        print(json.dumps({
            "threshold": args.threshold, "n_markers": len(corpus),
            "best_marker": best_marker, "best_policy": best_policy,
            "marker_rows": marker_rows, "policy_rows": policy_rows,
        }, default=str))
        sys.exit(0 if (best_marker or best_policy) else 1)

    src = f"SIM(noise={args.sim_noise})" if args.sim else (args.answer_model or "no-llm")
    print(f"\n=== Idea B — marker→rule design experiment  (judge={src}, n={len(corpus)}) ===")
    print(f"    gate: precision ≥ {args.threshold:.2f}   (recall reported)\n")

    print("── E1/E2  per-MARKER (routing instrument × τ) ─────────────────────────")
    print(f"  {'mode':<13} {'τ':>5} {'prec':>6} {'recall':>7} {'F1':>6}  {'TP/FP/FN':>10}")
    for r in marker_rows:
        flag = "  ✓" if r["precision"] >= args.threshold else ""
        print(f"  {r['mode']:<13} {r['tau']:>5.2f} {r['precision']*100:>5.1f}% "
              f"{r['recall']*100:>6.1f}% {r['f1']*100:>5.1f}%  "
              f"{r['tp']}/{r['fp']}/{r['fn']:>2}{flag}")

    print("\n── E3  per-POLICY (promotion × N), only rows clearing the gate ────────")
    print(f"  {'mode':<13} {'τ':>5} {'promo':<11} {'N':>2} {'prec':>6} {'recall':>7} {'pol':>4}")
    shown = 0
    for r in sorted(policy_rows, key=lambda x: (-x["precision"], -x["recall"])):
        if r["precision"] < args.threshold:
            continue
        shown += 1
        print(f"  {r['mode']:<13} {r['tau']:>5.2f} {r['promotion']:<11} {r['n']:>2} "
              f"{r['precision']*100:>5.1f}% {r['recall']*100:>6.1f}% {r['policies']:>4}")
    if not shown:
        print("  (no policy arm cleared the gate)")

    print("\n── WINNER ─────────────────────────────────────────────────────────────")
    if best_marker:
        print(f"  marker layer: {best_marker['mode']} @ τ={best_marker['tau']:.2f} → "
              f"precision {best_marker['precision']*100:.1f}%  recall {best_marker['recall']*100:.1f}%")
    else:
        print("  marker layer: NONE clears the gate")
    if best_policy:
        print(f"  policy layer: {best_policy['mode']} @ τ={best_policy['tau']:.2f} "
              f"{best_policy['promotion']} N={best_policy['n']} → "
              f"precision {best_policy['precision']*100:.1f}%  recall {best_policy['recall']*100:.1f}%")
    else:
        print("  policy layer: NONE clears the gate")
    print()
    sys.exit(0 if (best_marker or best_policy) else 1)


if __name__ == "__main__":
    main()
