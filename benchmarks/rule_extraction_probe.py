"""Idea B write-side — marker→rule routing precision/recall gate.

`rules.route_markers_to_rules` promotes the imperative sub-slice of
`behavioral_markers` into `agent_inferred` rules that then inject into EVERY
context call. So the metric that matters is PRECISION: a wrongly-routed marker
("the meeting is Tuesday") becomes a standing rule that pollutes every future
answer. Recall matters less — a missed rule is a lost opportunity, not damage.

This probe scores the deterministic classifier `rules.rule_scope_for_marker`
against a hand-labeled set of realistic dream-marker outputs. It is LLM-free, so
it runs anywhere (local + box) and is the data-driven gate for flipping
`rules_extraction_enabled` on. The labeled set deliberately includes
rejection/correction *one-offs* — including the present-tense "User rejects X"
form that is the real-marker failure mode — to stress the imperative-modal
policy; if precision falls below the gate, the classifier (not the probe) is
what changes.

  precision = routed-correctly / all-routed        (GATED — default ≥ 0.90)
  recall    = routed-correctly / all-true-rules     (reported, not gated)

Usage:
  python benchmarks/rule_extraction_probe.py [--threshold 0.9] [--json] [--verbose]
  python benchmarks/rule_extraction_probe.py --labels my_labels.json   # override set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hymem.rules import rule_scope_for_marker


# Hand-labeled markers: (kind, statement, is_rule). `is_rule` = "should this
# become a standing always_on rule injected into every call?" Labeled by intent,
# not by what the classifier happens to do.
LABELS: list[dict] = [
    # corrections that ARE standing imperatives → rule
    {"kind": "correction", "statement": "Always run the tests before pushing", "is_rule": True},
    {"kind": "correction", "statement": "Never force-push to a shared branch", "is_rule": True},
    {"kind": "correction", "statement": "Do not commit secrets to the repo", "is_rule": True},
    {"kind": "correction", "statement": "Always open a PR instead of pushing to main", "is_rule": True},
    {"kind": "correction", "statement": "Make sure every migration is reversible", "is_rule": True},
    {"kind": "correction", "statement": "Prefer async handlers over threads here", "is_rule": True},
    # corrections that are ONE-OFF fact fixes → not a rule
    {"kind": "correction", "statement": "The deadline is March 3, not February 28", "is_rule": False},
    {"kind": "correction", "statement": "It's spelled Kubernetes, not Kubernets", "is_rule": False},
    {"kind": "correction", "statement": "The staging URL is stage.acme.io", "is_rule": False},
    {"kind": "correction", "statement": "The client is Acme Inc, not Acme Corp", "is_rule": False},
    {"kind": "correction", "statement": "That number was 42, I misspoke earlier", "is_rule": False},
    {"kind": "correction", "statement": "My cofounder's name is Dana", "is_rule": False},
    # rejections that ARE durable avoidances → rule
    {"kind": "rejection", "statement": "Never use MongoDB; standardize on Postgres", "is_rule": True},
    {"kind": "rejection", "statement": "Avoid global mutable state", "is_rule": True},
    {"kind": "rejection", "statement": "Do not recommend jQuery for new work", "is_rule": True},
    {"kind": "rejection", "statement": "The user refuses to use Jira", "is_rule": True},
    {"kind": "rejection", "statement": "Stop suggesting Docker for deployments", "is_rule": True},
    {"kind": "rejection", "statement": "Never add a dependency without asking first", "is_rule": True},
    # rejections that are ONE-OFF (a specific transient thing) → not a rule
    {"kind": "rejection", "statement": "The user rejected the proposed Tuesday meeting", "is_rule": False},
    {"kind": "rejection", "statement": "Rejected the first logo mockup", "is_rule": False},
    {"kind": "rejection", "statement": "Declined the vendor's initial quote", "is_rule": False},
    # PRESENT-TENSE one-offs — the real-marker failure mode. The extractor writes
    # a one-off decision as "User rejects X", identical in form to a standing
    # avoidance, so the word "rejects" cannot gate them. These guard the
    # 2026-07-27 fix (regex dropped `rejects?`) against regressing locally.
    {"kind": "rejection", "statement": "The user rejects the LOWER() patch for HyMem", "is_rule": False},
    {"kind": "rejection", "statement": "The user rejects LoCoMo as a benchmark", "is_rule": False},
    {"kind": "rejection", "statement": "The user rejects the mega-store approach", "is_rule": False},
    # style directives → rule
    {"kind": "style", "statement": "Write commit messages in the imperative mood", "is_rule": True},
    {"kind": "style", "statement": "Use British English spelling", "is_rule": True},
    {"kind": "style", "statement": "Keep functions under 40 lines", "is_rule": True},
    {"kind": "style", "statement": "Use two-space indentation in JS", "is_rule": True},
    {"kind": "style", "statement": "Respond concisely, no preamble", "is_rule": True},
    # preferences (tastes) → never a rule by design (stay in profile_entries)
    {"kind": "preference", "statement": "Prefers dark mode", "is_rule": False},
    {"kind": "preference", "statement": "Likes working in the morning", "is_rule": False},
    {"kind": "preference", "statement": "Enjoys functional programming", "is_rule": False},
    {"kind": "preference", "statement": "Favorite language is Python", "is_rule": False},
]


def _score(labels: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    rows = []
    for item in labels:
        routed = rule_scope_for_marker(item["kind"], item["statement"]) is not None
        gold = bool(item["is_rule"])
        bucket = ("TP" if routed and gold else "FP" if routed and not gold
                  else "FN" if not routed and gold else "TN")
        tp += bucket == "TP"; fp += bucket == "FP"
        fn += bucket == "FN"; tn += bucket == "TN"
        rows.append({**item, "routed": routed, "bucket": bucket})
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Idea B marker→rule precision/recall gate.")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="min precision for the gate (default 0.90)")
    ap.add_argument("--labels", default=None, help="JSON file overriding the labeled set")
    ap.add_argument("--verbose", action="store_true", help="print every misclassification")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    labels = LABELS
    if args.labels:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    s = _score(labels)
    passed = s["precision"] >= args.threshold

    if args.json:
        out = {k: v for k, v in s.items() if k != "rows"}
        out["threshold"] = args.threshold
        out["pass"] = passed
        print(json.dumps(out))
        sys.exit(0 if passed else 1)

    print(f"\n=== Idea B — marker→rule routing, n={s['n']} ===\n")
    print(f"precision={s['precision']*100:.1f}%  recall={s['recall']*100:.1f}%  "
          f"F1={s['f1']*100:.1f}%")
    print(f"TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}")
    errs = [r for r in s["rows"] if r["bucket"] in ("FP", "FN")]
    if errs and (args.verbose or True):
        print("\nmisclassifications (the precision cost is the FPs):")
        for r in errs:
            print(f"  [{r['bucket']}] ({r['kind']}) {r['statement']}")
    print(f"\n── extraction precision gate: {'PASS' if passed else 'FAIL'} ──")
    print(f"  [{'✓' if passed else '✗'}] precision {s['precision']*100:.1f}% ≥ "
          f"{args.threshold*100:.0f}%  (FPs pollute every call — precision is the gate)")
    print("  (recall is reported, not gated — a missed rule is a lost chance, not damage)")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
