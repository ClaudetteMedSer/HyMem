"""Versioned source-free accounting for the inactive source-grounded candidate.

Classification reads only pinned code locations and frame line numbers. Neither
request text nor frame locals enter the ledger or its durable snapshot.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import threading
import time

SOURCE_SHA256 = {
    "hymem/extraction/chunk.py": "0dc650c244d3aea75b79308ada28593442382ff9f52ad88bd086982e76e6690b",
    "hymem/extraction/grounding_gate.py": "bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8",
    "hymem/dreaming/digest.py": "f0635472740bd8f1e2ba3a37fb08745917db22f110bd6f2b0eff61158f7c2cb0",
    "benchmarks/longmemeval_adapter.py": "19ef4b5fa030d14c0c4e7e2c843ac9058bb09d19e4a704dbb84114b084f70f31",
}
STAGES = frozenset({
    "extraction_primary_or_other", "extraction_contract_repair",
    "extraction_terminal_retry", "extraction_empty_verifier",
    "extraction_omission_verifier", "grounding_initial", "grounding_recheck",
    "digest_primary", "digest_summary_repair", "reader", "judge", "unclassified",
})


class StageAccountingStop(BaseException):
    """Finite failure without provider or source material."""


def verify_candidate(candidate: Path) -> None:
    root = candidate.resolve(strict=True)
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("stage_callsite_source_drift")


def classify_stack(candidate: Path, frame=None) -> str:
    root = candidate.resolve()
    current = frame if frame is not None else sys._getframe(1)
    sites = set()
    while current is not None:
        code = current.f_code
        try:
            relative = Path(code.co_filename).relative_to(root).as_posix()
        except ValueError:
            relative = ""
        if relative in SOURCE_SHA256:
            sites.add((relative, code.co_name, current.f_lineno))
        current = current.f_back
    chunk = "hymem/extraction/chunk.py"
    gate = "hymem/extraction/grounding_gate.py"
    if (chunk, "grounding_call", 3213) in sites:
        if (gate, "ground_triples", 191) in sites:
            return "grounding_recheck"
        if (gate, "ground_triples", 193) in sites:
            return "grounding_initial"
        return "unclassified"
    if (chunk, "single_attempt", 2975) in sites:
        if (chunk, "attempt", 3172) in sites:
            return "extraction_contract_repair"
        if (chunk, "recover", 3360) in sites:
            return "extraction_terminal_retry"
        if (chunk, "recover", 3305) in sites:
            return "extraction_empty_verifier"
        if (chunk, "verify_nonempty", 3239) in sites:
            return "extraction_omission_verifier"
        return "extraction_primary_or_other"
    digest = "hymem/dreaming/digest.py"
    if (digest, "extract_session_digest", 711) in sites:
        return "digest_primary"
    if (digest, "extract_session_digest", 742) in sites:
        return "digest_summary_repair"
    adapter = "benchmarks/longmemeval_adapter.py"
    if (adapter, "answer_question_raw", 2254) in sites:
        return "reader"
    if (adapter, "judge_answer_raw", 2615) in sites:
        return "judge"
    return "unclassified"


class StageLedger:
    def __init__(self, candidate: Path):
        verify_candidate(candidate)
        self.candidate = candidate.resolve()
        self._lock = threading.RLock()
        self._data = {}

    def wrap(self, delegate, question_id: str):
        return ProfiledClient(delegate, self, question_id)

    def record(self, question_id: str, stage: str, *, turns: int, tokens: int,
               elapsed: float, returned: bool, usage_complete: bool) -> None:
        if (type(question_id) is not str or not (question_id == "canary" or
            (len(question_id) == 6 and question_id.startswith("q-") and
             question_id[2:].isascii() and question_id[2:].isdigit()))
            or stage not in STAGES or type(turns) is not int or turns not in (0, 1)
            or type(tokens) is not int or tokens < 0 or type(elapsed) is not float
            or not math.isfinite(elapsed) or elapsed < 0 or type(returned) is not bool
            or type(usage_complete) is not bool):
            raise RuntimeError("stage_counter_invalid")
        with self._lock:
            slot = self._data.setdefault(question_id, {}).setdefault(stage, {
                "attempted_calls": 0, "admitted_turns": 0, "known_tokens": 0,
                "invocation_wall_seconds": 0.0, "returned_calls": 0,
                "rejected_calls": 0, "usage_complete": True,
            })
            slot["attempted_calls"] += 1
            slot["admitted_turns"] += turns
            slot["known_tokens"] += tokens
            slot["invocation_wall_seconds"] += elapsed
            slot["returned_calls" if returned else "rejected_calls"] += 1
            slot["usage_complete"] &= usage_complete

    def snapshot(self) -> dict:
        with self._lock:
            return {q: {s: dict(v) for s, v in stages.items()}
                    for q, stages in self._data.items()}

    def reconcile(self, budget: dict) -> bool:
        data = self.snapshot()
        if type(budget) is not dict or type(budget.get("questions")) is not dict:
            return False
        if set(data) != set(budget["questions"]):
            return False
        totals = {"turns": 0, "known_tokens": 0}
        for question, stages in data.items():
            if any(v["returned_calls"] + v["rejected_calls"] != v["attempted_calls"] or
                   v["admitted_turns"] > v["attempted_calls"] for v in stages.values()):
                return False
            turns = sum(v["admitted_turns"] for v in stages.values())
            tokens = sum(v["known_tokens"] for v in stages.values())
            state = budget["questions"][question]
            if type(state) is not dict:
                return False
            if turns != state.get("turns") or tokens != state.get("known_tokens"):
                return False
            totals["turns"] += turns
            totals["known_tokens"] += tokens
        return all(budget.get(k) == v for k, v in totals.items())

    def reconcile_canary(self, report: dict) -> bool:
        """Require every classified call to match the canary's actual counters."""
        if type(report) is not dict:
            return False
        stages = self.snapshot().get("canary", {})
        if set(self.snapshot()) != {"canary"} or "unclassified" in stages:
            return False
        extraction = sum(v["attempted_calls"] for name, v in stages.items()
                         if name.startswith("extraction_"))
        initial = stages.get("grounding_initial", {}).get("attempted_calls", 0)
        recheck = stages.get("grounding_recheck", {}).get("attempted_calls", 0)
        attempts = sum(v["attempted_calls"] for v in stages.values())
        admitted = sum(v["admitted_turns"] for v in stages.values())
        tokens = sum(v["known_tokens"] for v in stages.values())
        return (extraction == report.get("ordinary_calls")
                and initial == report.get("grounding_initial_calls")
                and recheck == report.get("grounding_recheck_calls")
                and attempts == admitted == report.get("completion_calls")
                and admitted == report.get("observed_turn_delta")
                and tokens == report.get("observed_token_delta")
                and all(v["usage_complete"] and v["rejected_calls"] == 0
                        for v in stages.values()))


class ProfiledClient:
    def __init__(self, delegate, ledger: StageLedger, question_id: str):
        self.delegate, self.ledger, self.question_id = delegate, ledger, question_id

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def complete(self, request):
        stage = classify_stack(self.ledger.candidate)
        before_turns, before_tokens = self.delegate.observed_turns, self.delegate.observed_tokens
        start = time.monotonic()
        returned = False
        try:
            result = self.delegate.complete(request)
            returned = True
            return result
        finally:
            try:
                self.ledger.record(self.question_id, stage,
                    turns=self.delegate.observed_turns - before_turns,
                    tokens=self.delegate.observed_tokens - before_tokens,
                    elapsed=float(time.monotonic() - start), returned=returned,
                    usage_complete=self.delegate.usage_complete)
            except BaseException:
                try:
                    self.delegate.budget.halt("stage_accounting_failure")
                except Exception:
                    pass
                raise StageAccountingStop("stage_accounting_failure") from None
