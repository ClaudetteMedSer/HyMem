"""Source-free call accounting for the opt-in Luna LME diagnostic.

Only fixed labels and numeric aggregates are retained. The frozen candidate is
never patched; its verified code locations are used solely to label calls.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import threading
import time


SOURCE_SHA256 = {
    "hymem/extraction/chunk.py": "11fe46a7d1f6fcd65a33a109104a42ebcd7af888b99d2e30460bcd90633301c9",
    "hymem/dreaming/digest.py": "f0635472740bd8f1e2ba3a37fb08745917db22f110bd6f2b0eff61158f7c2cb0",
    "benchmarks/longmemeval_adapter.py": "19ef4b5fa030d14c0c4e7e2c843ac9058bb09d19e4a704dbb84114b084f70f31",
}

STAGES = frozenset({
    "extraction_primary_or_other", "extraction_contract_repair",
    "extraction_terminal_retry", "extraction_empty_verifier",
    "extraction_omission_verifier", "digest_primary", "digest_summary_repair",
    "reader", "judge", "unclassified",
})


class StageAccountingStop(BaseException):
    """Fixed diagnostic stop code; never carries a request or provider reply."""


def verify_candidate(candidate: Path) -> None:
    """Reject an altered call-site map before running any model call."""
    candidate = candidate.resolve(strict=True)
    for relative, expected in SOURCE_SHA256.items():
        path = candidate / relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("stage_callsite_source_drift")


def classify_stack(candidate: Path, frame=None) -> str:
    """Use code identity and current source line only; never access frame locals."""
    root = candidate.resolve()
    current = frame if frame is not None else sys._getframe(1)
    sites = []
    while current is not None:
        code = current.f_code
        filename = code.co_filename
        try:
            relative = str(Path(filename).relative_to(root))
        except ValueError:
            relative = ""
        if relative in SOURCE_SHA256:
            sites.append((relative, code.co_name, current.f_lineno))
        current = current.f_back

    # The first matching frozen call site is authoritative. Branch lines are
    # read from pinned source, including the two explicit retry paths.
    for path, function, line in sites:
        if path == "hymem/extraction/chunk.py" and function == "single_attempt" and line == 2955:
            if (path, "attempt", 3152) in sites or (path, "attempt", 3153) in sites or (path, "attempt", 3154) in sites:
                return "extraction_contract_repair"
            if (path, "recover", 3309) in sites:
                return "extraction_terminal_retry"
            if (path, "recover", 3254) in sites:
                return "extraction_empty_verifier"
            if ((path, "verify_nonempty", 3200) in sites
                    or (path, "verify_nonempty", 3201) in sites):
                return "extraction_omission_verifier"
            return "extraction_primary_or_other"
        if path == "hymem/dreaming/digest.py" and function == "extract_session_digest":
            if line == 711:
                return "digest_primary"
            if line == 742:
                return "digest_summary_repair"
        if path == "benchmarks/longmemeval_adapter.py":
            if function == "answer_question_raw" and line == 2254:
                return "reader"
            if function == "judge_answer_raw" and line == 2615:
                return "judge"
    return "unclassified"


class StageLedger:
    def __init__(self, candidate: Path):
        verify_candidate(candidate)
        self.candidate = candidate.resolve()
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, dict[str, int | float | bool]]] = {}

    def wrap(self, delegate, question_id: str):
        return ProfiledClient(delegate, self, question_id)

    def record(self, question_id: str, stage: str, *, turns: int, tokens: int,
               elapsed: float, returned: bool, usage_complete: bool) -> None:
        if (type(question_id) is not str or not (question_id == "canary" or
                 (len(question_id) == 6 and question_id.startswith("q-")
                  and all("0" <= char <= "9" for char in question_id[2:])))
                or stage not in STAGES or type(turns) is not int or turns not in (0, 1)
                or type(tokens) is not int or tokens < 0
                or not isinstance(elapsed, float) or not math.isfinite(elapsed)
                or elapsed < 0 or type(returned) is not bool
                or type(usage_complete) is not bool):
            raise RuntimeError("stage_counter_invalid")
        with self._lock:
            slot = self._data.setdefault(question_id, {}).setdefault(stage, {
                "attempted_calls": 0, "admitted_turns": 0,
                "known_tokens": 0, "invocation_wall_seconds": 0.0,
                "returned_calls": 0, "rejected_calls": 0,
                "usage_complete": True,
            })
            slot["attempted_calls"] += 1
            slot["admitted_turns"] += turns
            slot["known_tokens"] += tokens
            slot["invocation_wall_seconds"] += elapsed
            slot["returned_calls" if returned else "rejected_calls"] += 1
            slot["usage_complete"] = slot["usage_complete"] and usage_complete

    def snapshot(self) -> dict:
        with self._lock:
            return {question: {stage: dict(values) for stage, values in stages.items()}
                    for question, stages in self._data.items()}

    def reconcile(self, budget: dict) -> bool:
        snapshot = self.snapshot()
        turns = sum(slot["admitted_turns"] for stages in snapshot.values()
                    for slot in stages.values())
        tokens = sum(slot["known_tokens"] for stages in snapshot.values()
                     for slot in stages.values())
        if turns != budget.get("turns") or tokens != budget.get("known_tokens"):
            return False
        for question, state in budget.get("questions", {}).items():
            stages = snapshot.get(question, {})
            if (sum(slot["admitted_turns"] for slot in stages.values()) != state.get("turns")
                    or sum(slot["known_tokens"] for slot in stages.values()) != state.get("known_tokens")):
                return False
        return True


class ProfiledClient:
    def __init__(self, delegate, ledger: StageLedger, question_id: str):
        self.delegate, self.ledger, self.question_id = delegate, ledger, question_id

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def complete(self, request):
        stage = classify_stack(self.ledger.candidate)
        before_turns = self.delegate.observed_turns
        before_tokens = self.delegate.observed_tokens
        start = time.monotonic()
        returned = False
        try:
            result = self.delegate.complete(request)
            returned = True
            return result
        finally:
            elapsed = time.monotonic() - start
            try:
                self.ledger.record(self.question_id, stage,
                    turns=self.delegate.observed_turns - before_turns,
                    tokens=self.delegate.observed_tokens - before_tokens,
                    elapsed=elapsed, returned=returned,
                    usage_complete=self.delegate.usage_complete)
            except BaseException:
                self.delegate.budget.halt("stage_accounting_failure")
                raise StageAccountingStop("stage_accounting_failure") from None
