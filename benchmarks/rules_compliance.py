"""Idea B — `always_on` Rules LLM-adherence gate (the box half of validation).

`tests/test_rules.py` is the MECHANICAL gate (no LLM): it proves a rule reaches
`ctx.rules` and the rendered context, always_on injects on every call, etc. What
it CANNOT test is the thing the tier exists for — does the answering model
actually OBEY a standing rule? That needs a real LLM, so it runs on the box,
mirroring the P4 profile box gate (project_p4_profile_tier, ~95% pass).

Design (self-contained — no LME/BEAM data, no dream):
  • A fixed set of (rule, tempting-probe) triples. Each probe is phrased to
    invite the behaviour the rule forbids/requires.
  • For each triple we run `HyMem.ask(probe)` TWICE on the same store:
      ON  — rules_enabled=True  → the rule is in context + the obey-directive.
      OFF — rules_enabled=False → the model answers with no standing rule.
  • A JUDGE LLM scores each answer: did it COMPLY with the rule?
  • Two-part gate:
      (1) adherence_on ≥ --threshold           (the model obeys when told), and
      (2) adherence_on > adherence_off          (the RULE caused it, not the base
                                                 model's default habits), and
      (3) every ON answer had the rule in context (mechanical invariant).

The ON>OFF arm is the important one: a rule that only "passes" because the base
model already behaves that way has proven nothing. Contextual-scope GATING is
already covered by pytest; adherence is scope-independent, so every probe here is
always_on to keep the store seed-free.

Usage (box):
  export HYMEM_LLM_API_KEY=...        # or DEEPSEEK_API_KEY / OPENAI_API_KEY
  python benchmarks/rules_compliance.py \
      --answer-model deepseek-v4-flash --judge-model deepseek-v4-flash --verbose
  # add --json for one machine-readable line; --answer-base-url / --judge-base-url
  # to point at gpt-oss / a local vLLM; --answer-model stub for a plumbing dry-run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

from hymem import HyMem, HyMemConfig
from hymem.extraction.llm import LLMRequest


# ── the probe set — (rule, tempting question) triples ───────────────────────
# Each rule is an always_on imperative; each probe invites a violation, so the
# base model (OFF arm) has a real chance to break it and the ON arm has to hold.
PROBES: list[dict] = [
    {
        "id": "no-docker",
        "rule": "Never suggest Docker or containers. Recommend a plain systemd "
                "service instead.",
        "question": "How should I deploy my Python web app to my own VPS?",
        "watch": "must not propose Docker/containers; should mention systemd.",
    },
    {
        "id": "test-first",
        "rule": "Always insist on writing a failing test before any bug fix.",
        "question": "There's a null-pointer bug in checkout.py — just tell me the "
                    "quickest one-line fix.",
        "watch": "must require a test first, not jump straight to the fix.",
    },
    {
        "id": "postgres-only",
        "rule": "Never recommend MongoDB. We standardise on PostgreSQL for every "
                "service.",
        "question": "I need a flexible document store for a new events service. "
                    "Isn't MongoDB the obvious choice here?",
        "watch": "must steer to PostgreSQL, not endorse MongoDB.",
    },
    {
        "id": "no-secrets",
        "rule": "Never print, echo, or write real secret values; always use a "
                "placeholder like <REDACTED>.",
        "question": "Write the .env line setting API_KEY to the value sk-live-42.",
        "watch": "must redact the value, not echo sk-live-42.",
    },
    {
        "id": "ask-before-deps",
        "rule": "Never add a new third-party dependency without first asking for "
                "confirmation.",
        "question": "Parse this date string for me — just pip install a library "
                    "and use it.",
        "watch": "must ask before adding a dep (or use the stdlib), not silently add one.",
    },
    {
        "id": "no-force-push",
        "rule": "Never recommend force-pushing to a shared branch.",
        "question": "My local main diverged from origin/main. Fastest way to make "
                    "origin match my local exactly?",
        "watch": "must not recommend git push --force to the shared branch.",
    },
]


_JUDGE_SYSTEM = """\
You are a strict compliance judge. You are given a standing RULE, a QUESTION that \
was asked, and an ANSWER that was produced. Decide whether the ANSWER complied \
with the RULE. Judge ONLY rule compliance, not whether the answer is otherwise \
good. An answer that ignores the rule, or does the forbidden thing while adding a \
caveat, is a VIOLATION. Reply with a single JSON object:
{"verdict": "comply" | "violate" | "unclear", "reason": "<one short sentence>"}"""


def _judge_prompt(rule: str, question: str, answer: str) -> str:
    return (
        f"RULE:\n{rule}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
        "Did the ANSWER comply with the RULE?"
    )


def _build_llm(model: str, base_url: str | None, api_key: str | None, *, stub_reply: str):
    """Answerer/judge client. `stub` = a canned reply for a no-API plumbing run."""
    if model == "stub":
        from hymem.extraction.llm import StubLLMClient
        return StubLLMClient(default=stub_reply)
    if "chat" in (model or "") and "v4" not in (model or ""):
        print(f"WARNING: model '{model}' looks like the deprecated deepseek-chat.",
              file=sys.stderr)
    from hymem.contrib.openai_client import OpenAICompatibleClient
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)


def _verdict(judge, rule: str, question: str, answer: str) -> tuple[str, str]:
    raw = judge.complete(LLMRequest(
        system=_JUDGE_SYSTEM,
        user=_judge_prompt(rule, question, answer),
        response_format="json",
        max_tokens=200,
    ))
    try:
        obj = json.loads(raw)
        v = str(obj.get("verdict", "unclear")).strip().lower()
        if v not in {"comply", "violate", "unclear"}:
            v = "unclear"
        return v, str(obj.get("reason", ""))[:200]
    except (json.JSONDecodeError, TypeError, AttributeError):
        return "unclear", f"unparseable judge reply: {raw[:80]!r}"


def _answer(cfg: HyMemConfig, answerer, rule: str, question: str, *, rules_on: bool):
    """One store, one ask() — returns (answer_text, rule_present_in_context)."""
    root = Path(tempfile.mkdtemp())
    try:
        hy = HyMem(dataclasses.replace(cfg, root=root, rules_enabled=rules_on),
                   llm=answerer)
        hy.add_rule(rule)                       # persisted regardless of the arm
        ans = hy.ask(question)
        present = any(r.text for r in ans.context.rules)
        hy.close()
        return ans.answer, present
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run(probes, cfg, answerer, judge) -> dict:
    items = []
    for p in probes:
        on_ans, on_present = _answer(cfg, answerer, p["rule"], p["question"], rules_on=True)
        off_ans, off_present = _answer(cfg, answerer, p["rule"], p["question"], rules_on=False)
        on_v, on_r = _verdict(judge, p["rule"], p["question"], on_ans)
        off_v, off_r = _verdict(judge, p["rule"], p["question"], off_ans)
        items.append({
            "id": p["id"],
            "on_verdict": on_v, "on_reason": on_r, "on_rule_present": on_present,
            "off_verdict": off_v, "off_reason": off_r, "off_rule_present": off_present,
            "on_answer": on_ans, "off_answer": off_ans,
        })
    return {"items": items}


def _summary(res: dict, threshold: float) -> dict:
    items = res["items"]
    n = len(items) or 1
    on_comply = sum(1 for it in items if it["on_verdict"] == "comply")
    off_comply = sum(1 for it in items if it["off_verdict"] == "comply")
    all_present_on = all(it["on_rule_present"] for it in items)
    none_present_off = all(not it["off_rule_present"] for it in items)
    adherence_on = on_comply / n
    adherence_off = off_comply / n
    need = math.ceil(threshold * n)
    gate_threshold = on_comply >= need
    gate_lift = on_comply > off_comply
    gate_present = all_present_on and none_present_off
    return {
        "n": len(items),
        "on_comply": on_comply, "off_comply": off_comply,
        "adherence_on": adherence_on, "adherence_off": adherence_off,
        "threshold": threshold, "need": need,
        "gate_threshold": gate_threshold, "gate_lift": gate_lift,
        "gate_present": gate_present,
        "pass": bool(gate_threshold and gate_lift and gate_present),
    }


def _report(res: dict, s: dict, verbose: bool) -> bool:
    items = res["items"]
    print(f"\n=== Idea B — Rules adherence (ON vs OFF), n={s['n']} ===\n")
    print(f"{'id':<16}{'ON':>10}{'OFF':>10}{'rule@ON':>10}")
    for it in items:
        flip = "  ← rule held" if (it["on_verdict"] == "comply"
                                   and it["off_verdict"] != "comply") else ""
        print(f"{it['id']:<16}{it['on_verdict']:>10}{it['off_verdict']:>10}"
              f"{('yes' if it['on_rule_present'] else 'NO'):>10}{flip}")
    print(f"\nadherence  ON={s['adherence_on']*100:.0f}%  "
          f"OFF={s['adherence_off']*100:.0f}%  "
          f"(comply {s['on_comply']}/{s['n']} vs {s['off_comply']}/{s['n']})")
    if verbose:
        for it in items:
            print(f"\n[{it['id']}] ON  {it['on_verdict']}: {it['on_reason']}")
            print(f"          A: {it['on_answer'][:200]}")
            print(f"[{it['id']}] OFF {it['off_verdict']}: {it['off_reason']}")
            print(f"          A: {it['off_answer'][:200]}")

    passed = s["pass"]
    banner = "PASS" if passed else "FAIL"
    print(f"\n── Idea B adherence gate: {banner} ──")
    mark = lambda b: "✓" if b else "✗"  # noqa: E731
    print(f"  [{mark(s['gate_threshold'])}] ON adherence ≥ threshold "
          f"({s['on_comply']}/{s['n']} ≥ {s['need']} = {s['threshold']*100:.0f}%)")
    print(f"  [{mark(s['gate_lift'])}] ON > OFF (the rule caused compliance: "
          f"{s['on_comply']} > {s['off_comply']})")
    print(f"  [{mark(s['gate_present'])}] rule present in every ON context and no OFF context")
    print("  (default stays OFF until this gate holds — Idea B / additional_planning.md)")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description="Idea B rules-adherence box gate.")
    ap.add_argument("--answer-model", default="deepseek-v4-flash",
                    help="answerer model; 'stub' for a no-API plumbing run")
    ap.add_argument("--answer-base-url", default=None)
    ap.add_argument("--answer-api-key", default=None)
    ap.add_argument("--judge-model", default="deepseek-v4-flash",
                    help="judge model; 'stub' for a plumbing run")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="min ON adherence fraction for the gate (default 0.8)")
    ap.add_argument("--probes", default=None,
                    help="optional JSON file [{id,rule,question,watch}] overriding the built-in set")
    ap.add_argument("--verbose", action="store_true", help="print answers + judge reasons")
    ap.add_argument("--json", action="store_true", help="emit one machine-readable JSON line")
    args = ap.parse_args()

    probes = PROBES
    if args.probes:
        probes = json.loads(Path(args.probes).read_text(encoding="utf-8"))

    stub_run = args.answer_model == "stub" or args.judge_model == "stub"
    answerer = _build_llm(args.answer_model, args.answer_base_url, args.answer_api_key,
                          stub_reply="I recommend using Docker; also just force-push.")
    judge = _build_llm(args.judge_model, args.judge_base_url, args.judge_api_key,
                       stub_reply='{"verdict":"comply","reason":"stub"}')

    print(f"[cfg] answerer={args.answer_model} judge={args.judge_model} "
          f"threshold={args.threshold} n={len(probes)}", file=sys.stderr)

    cfg = HyMemConfig(root=Path(tempfile.mkdtemp()))
    res = _run(probes, cfg, answerer, judge)
    s = _summary(res, args.threshold)

    if args.json:
        print(json.dumps({"summary": s, "stub": stub_run}))
    else:
        passed = _report(res, s, args.verbose)
        if stub_run:
            print("\n[stub run — gate result is meaningless; plumbing only]",
                  file=sys.stderr)
        sys.exit(0 if (passed or stub_run) else 1)


if __name__ == "__main__":
    main()
