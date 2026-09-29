"""Offline source-line, reconciliation, and error-path accounting controls."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import luna_semantic_stage_accounting as stages

CANDIDATE = Path("/private/tmp/hymem-semantic-step2-v2-candidate-20260929")


def frame(relative, name, line, back=None):
    return SimpleNamespace(
        f_code=SimpleNamespace(co_filename=str(CANDIDATE / relative), co_name=name),
        f_lineno=line, f_back=back)


@pytest.mark.parametrize("expected,frames", [
    ("grounding_initial", [("hymem/extraction/chunk.py", "grounding_call", 3213),
                           ("hymem/extraction/grounding_gate.py", "ground_triples", 193)]),
    ("grounding_recheck", [("hymem/extraction/chunk.py", "grounding_call", 3213),
                           ("hymem/extraction/grounding_gate.py", "ground_triples", 191)]),
    ("extraction_primary_or_other", [("hymem/extraction/chunk.py", "single_attempt", 2975)]),
    ("extraction_contract_repair", [("hymem/extraction/chunk.py", "single_attempt", 2975),
                                    ("hymem/extraction/chunk.py", "attempt", 3172)]),
    ("extraction_empty_verifier", [("hymem/extraction/chunk.py", "single_attempt", 2975),
                                   ("hymem/extraction/chunk.py", "recover", 3305)]),
    ("extraction_omission_verifier", [("hymem/extraction/chunk.py", "single_attempt", 2975),
                                      ("hymem/extraction/chunk.py", "verify_nonempty", 3239)]),
    ("extraction_terminal_retry", [("hymem/extraction/chunk.py", "single_attempt", 2975),
                                   ("hymem/extraction/chunk.py", "recover", 3360)]),
    ("digest_primary", [("hymem/dreaming/digest.py", "extract_session_digest", 711)]),
    ("digest_summary_repair", [("hymem/dreaming/digest.py", "extract_session_digest", 742)]),
    ("reader", [("benchmarks/longmemeval_adapter.py", "answer_question_raw", 2254)]),
    ("judge", [("benchmarks/longmemeval_adapter.py", "judge_answer_raw", 2615)]),
])
def test_pinned_callsite_classification(expected, frames):
    current = None
    for relative, name, line in reversed(frames):
        current = frame(relative, name, line, current)
    assert stages.classify_stack(CANDIDATE, current) == expected


def test_source_drift_fails_before_accounting(tmp_path):
    with pytest.raises(RuntimeError, match="stage_callsite_source_drift"):
        stages.StageLedger(tmp_path)


def test_rejected_invocation_is_counted_and_reconciled():
    ledger = stages.StageLedger(CANDIDATE)
    ledger.record("canary", "grounding_initial", turns=1, tokens=5,
                  elapsed=0.1, returned=False, usage_complete=False)
    budget = dict(turns=1, known_tokens=5, questions={"canary": dict(turns=1, known_tokens=5)})
    assert ledger.reconcile(budget)
    assert ledger.snapshot()["canary"]["grounding_initial"]["rejected_calls"] == 1
    assert not ledger.reconcile_canary(dict(ordinary_calls=0, grounding_initial_calls=1,
                                           grounding_recheck_calls=0, completion_calls=1,
                                           observed_turn_delta=1, observed_token_delta=5))
    assert not ledger.reconcile(dict(turns=2, known_tokens=5,
                                     questions={"canary": dict(turns=1, known_tokens=5)}))


def test_counter_failure_halts_with_fixed_code():
    ledger = stages.StageLedger(CANDIDATE)
    halted = []
    class Delegate:
        observed_turns = 0
        observed_tokens = 0
        usage_complete = True
        budget = SimpleNamespace(halt=lambda code: halted.append(code))
        def complete(self, request):
            self.observed_turns += 1
            self.observed_tokens += 1
            return "private response"
    ledger.record = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private diagnostic"))
    with pytest.raises(stages.StageAccountingStop, match="stage_accounting_failure"):
        ledger.wrap(Delegate(), "canary").complete(object())
    assert halted == ["stage_accounting_failure"]
