"""Idea B write-side — adjudicate the tagger-vs-label disagreements.

The real experiment gave the LLM tagger ~59% precision vs lexical's 8% — the
instrument works, but it doesn't clear 0.90, and some of the "false positives"
are suspected LABEL errors (the tagger being right where the hand-label was
wrong). A precision number is only as good as its ground truth, so this tool
resolves the contested markers the disciplined way — no rubber-stamping the model:

  • annotator A = the original hand label.
  • annotator B = the durability tagger (the thing under test).
  • annotator C = an INDEPENDENT, BLIND judge — a DIFFERENT model family, shown
    only kind+statement (never A's label or B's verdict). C breaks every A≠B tie.

Guards against fooling ourselves:
  • Corrected ground truth = original label on agreements, C's verdict on the
    contested items. C — not B — decides, so the tagger never votes on its own
    trial (no circularity).
  • A CONTROL sample of agreements (A==B) is adjudicated too; C's accuracy on
    them estimates C's reliability. Low control accuracy ⇒ distrust the whole run.
  • C must be a different model from B, or correlated LLM errors inflate the
    correction (flagged if the models match).

Output: per-disagreement A/B/C verdicts + C's reason, the count of "FPs" that are
actually label errors, RAW vs ADJUDICATED precision/recall, C's control accuracy,
and a corrected-labels JSON (`--out`) to feed back into
`rule_extraction_experiment.py --labels`.

Usage:
  python benchmarks/rule_extraction_adjudicate.py --labels markers.json \\
      --answer-model deepseek-v4-flash --answer-base-url <url> \\
      --judge-model  openai/gpt-oss-120b --judge-base-url <openrouter-url> \\
      --out corrected.json --verbose
  python benchmarks/rule_extraction_adjudicate.py --sim        # offline mechanics
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from hymem import rules_extract
from hymem.rules_extract import route_decisions

sys.path.insert(0, str(Path(__file__).parent))  # sibling benchmark imports
from rule_extraction_experiment import SimJudge, _build_llm, load_corpus  # noqa: E402


# C's prompt: the same standing-vs-one-off question, framed as a blind third-party
# adjudication and asking for a one-line reason. It never sees A's label or B's
# verdict, so it is an independent signal.
ADJUDICATION_SYSTEM = """You are an impartial adjudicator deciding whether each user behavioral signal is a STANDING RULE or a ONE-OFF.

A STANDING RULE is a durable instruction to follow on EVERY future turn, indefinitely, and generalizes beyond one specific artifact/moment: "never suggest Docker", "always run the tests before pushing", "write commit messages in the imperative mood".

A ONE-OFF is a decision, correction, or rejection about ONE specific thing, event, or moment, and must NOT become a rule: "rejects the LOWER() patch", "the meeting is Tuesday not Monday", "the podcast was in Dutch instead of English".

Words like "rejects"/"instead"/"should" do NOT decide it — judge the SUBSTANCE: would obeying this on every future turn make sense, or only in the one situation it describes?

Input is a JSON array of numbered markers: [{"index": 0, "kind": "...", "statement": "..."}, ...].
Output a strict JSON array, one object per input marker, in index order, no prose:
  {"index": <int>, "standing": <boolean>, "reason": "<one short clause>"}
"standing" MUST be a JSON boolean true|false."""


def _metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def _precision_recall(tagger_rule: list[bool], labels: list[bool]) -> dict:
    tp = sum(t and g for t, g in zip(tagger_rule, labels))
    fp = sum(t and not g for t, g in zip(tagger_rule, labels))
    fn = sum((not t) and g for t, g in zip(tagger_rule, labels))
    return _metrics(tp, fp, fn)


def main() -> None:
    ap = argparse.ArgumentParser(description="Adjudicate tagger-vs-label disagreements.")
    ap.add_argument("--labels", default=None, help="JSON corpus (kind/statement/is_rule)")
    ap.add_argument("--tau", type=float, default=0.75, help="tagger confidence threshold")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--controls", type=int, default=12,
                    help="agreed markers to adjudicate as a reliability control")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer-model", default=None, help="tagger model (annotator B)")
    ap.add_argument("--answer-base-url", default=None)
    ap.add_argument("--answer-api-key", default=None)
    ap.add_argument("--judge-model", default=None, help="INDEPENDENT adjudicator (annotator C)")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--sim", action="store_true", help="offline: SimJudge for B and C")
    ap.add_argument("--out", default=None, help="write corrected-labels JSON here")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(args.labels)
    markers = [(c["kind"], c["statement"]) for c in corpus]
    labels = [bool(c["is_rule"]) for c in corpus]
    gold_map = {c["statement"]: c for c in corpus}
    rng = random.Random(args.seed)

    if args.sim:
        tagger = SimJudge(gold_map, noise=0.12, seed=args.seed)          # B: noisy
        adjudicator = SimJudge(gold_map, noise=0.0, seed=args.seed + 99)  # C: oracle-ish
        same_model = False
    else:
        tagger = _build_llm(args.answer_model, args.answer_base_url, args.answer_api_key)
        adjudicator = _build_llm(args.judge_model, args.judge_base_url, args.judge_api_key)
        same_model = bool(args.answer_model) and args.answer_model == args.judge_model

    # B: the tagger's verdicts.
    decisions = route_decisions(markers, mode="llm", llm=tagger,
                                confidence_min=args.tau, batch_size=args.batch_size)
    tagger_rule = [d.route for d in decisions]

    # Partition A vs B.
    disagree = [i for i in range(len(corpus)) if tagger_rule[i] != labels[i]]
    agree = [i for i in range(len(corpus)) if tagger_rule[i] == labels[i]]
    rng.shuffle(agree)
    controls = agree[: max(0, args.controls)]

    # C: adjudicate the contested items + the control sample, blind and shuffled.
    adj_idx = disagree + controls
    rng.shuffle(adj_idx)
    adj_verdicts: dict[int, rules_extract.DurabilityJudgment] = {}
    if adj_idx:
        judged = rules_extract.judge_durability_batch(
            adjudicator, [markers[i] for i in adj_idx],
            batch_size=args.batch_size, system=ADJUDICATION_SYSTEM)
        adj_verdicts = {adj_idx[j]: judged[j] for j in range(len(judged))}

    def c_says_rule(i: int) -> bool:
        j = adj_verdicts.get(i)
        return bool(j and j.standing)

    # C's reliability on the controls (where A==B is the consensus truth).
    ctrl_hits = sum(c_says_rule(i) == labels[i] for i in controls)
    control_acc = ctrl_hits / len(controls) if controls else None

    # Corrected ground truth: C decides the contested items; agreements untouched.
    corrected = list(labels)
    fp_label_wrong = fp_real = fn_label_wrong = fn_real = 0
    for i in disagree:
        corrected[i] = c_says_rule(i)
        if tagger_rule[i] and not labels[i]:            # tagger=rule, label=not (an "FP")
            if c_says_rule(i):
                fp_label_wrong += 1                     # C agrees with tagger → label wrong
            else:
                fp_real += 1                            # C agrees with label → real FP
        elif (not tagger_rule[i]) and labels[i]:        # tagger=not, label=rule (an "FN")
            if not c_says_rule(i):
                fn_label_wrong += 1
            else:
                fn_real += 1

    raw = _precision_recall(tagger_rule, labels)
    adj = _precision_recall(tagger_rule, corrected)

    if args.out:
        out_corpus = [
            {**{k: v for k, v in corpus[i].items() if k != "is_rule"}, "is_rule": corrected[i]}
            for i in range(len(corpus))
        ]
        Path(args.out).write_text(json.dumps(out_corpus, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({
            "tau": args.tau, "n": len(corpus), "n_disagree": len(disagree),
            "fp_label_wrong": fp_label_wrong, "fp_real": fp_real,
            "fn_label_wrong": fn_label_wrong, "fn_real": fn_real,
            "control_accuracy": control_acc, "same_model": same_model,
            "raw": raw, "adjudicated": adj,
        }, default=str))
        sys.exit(0)

    src = "SIM" if args.sim else f"B={args.answer_model} / C={args.judge_model}"
    print(f"\n=== Idea B — disagreement adjudication  ({src}, n={len(corpus)}, τ={args.tau}) ===\n")
    if same_model:
        print("  ⚠  tagger and adjudicator are the SAME model — correlated errors may\n"
              "     inflate the correction. Use an independent model for C.\n")
    print(f"  A≠B disagreements: {len(disagree)}   controls adjudicated: {len(controls)}")
    if control_acc is not None:
        warn = "  ⚠ LOW — distrust this run" if control_acc < 0.8 else ""
        print(f"  C control accuracy: {control_acc*100:.1f}%{warn}")
    print()
    print("  Of the tagger's false positives (tagger=rule, label=not):")
    print(f"    • {fp_label_wrong} the independent judge calls a RULE  → label likely WRONG")
    print(f"    • {fp_real} the independent judge calls a one-off → REAL false positive")
    if fn_label_wrong or fn_real:
        print("  Of the tagger's false negatives (tagger=not, label=rule):")
        print(f"    • {fn_label_wrong} label likely wrong    • {fn_real} real miss")
    print()
    print(f"  precision  RAW {raw['precision']*100:5.1f}%   →   ADJUDICATED {adj['precision']*100:5.1f}%"
          f"   (gate 90%)")
    print(f"  recall     RAW {raw['recall']*100:5.1f}%   →   ADJUDICATED {adj['recall']*100:5.1f}%")

    if args.verbose and disagree:
        print("\n  ── contested markers (A=label, B=tagger, C=independent) ──")
        for i in disagree:
            j = adj_verdicts.get(i)
            c = "rule" if c_says_rule(i) else "one-off"
            flag = " ⟵ label likely wrong" if c_says_rule(i) != labels[i] else ""
            print(f"    A={'rule' if labels[i] else 'not ':<4} B={'rule' if tagger_rule[i] else 'not ':<4}"
                  f" C={c:<7}{flag}")
            print(f"      [{corpus[i]['kind']}] {corpus[i]['statement']}")
            if j and j.rationale:
                print(f"      C: {j.rationale}")
    if args.out:
        print(f"\n  corrected labels → {args.out}  (feed to rule_extraction_experiment.py --labels)")
    print()


if __name__ == "__main__":
    main()
