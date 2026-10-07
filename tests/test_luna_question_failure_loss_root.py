"""Root reproduction of the frozen runner's lost question-failure evidence."""
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_staged_v6 as staged
from tools.diagnostics import luna_lme_diagnostic_v8 as runner


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError, TypeError])
@pytest.mark.parametrize("phase", ["open", "evaluate"])
def test_original_class_and_phase_are_lost(monkeypatch, tmp_path, exception_type, phase):
    """Different application faults produce the same evidence without inference."""
    cap = staged.warm.BudgetLimits(10, 10000, 600)
    budget = staged.warm.SharedBudget(cap)
    captured = []
    closed = []

    class Adapter:
        def __init__(self, *args, **kwargs):
            self.last_indexing_summary = None

        def open(self):
            if phase == "open":
                raise exception_type("PRIVATE-ORIGIN")

        def close(self):
            closed.append("adapter")

    def evaluate(*args, **kwargs):
        raise exception_type("PRIVATE-ORIGIN")

    class IndexingError(Exception):
        pass

    client = SimpleNamespace(close=lambda: closed.append("client"))
    monkeypatch.setattr(runner, "make_dual", lambda *args, **kwargs: client)
    monkeypatch.setattr(runner, "AccountedClient", lambda *args: client)
    monkeypatch.setattr(runner, "_memory_client", lambda *args: client)
    loaded = {
        "candidate": tmp_path,
        "request_type": SimpleNamespace,
        "protocol": None, "strictness": None, "summary_classifier": None,
        "lme": SimpleNamespace(IndexingConvergenceError=IndexingError,
            evaluate_question=evaluate, DEFAULT_MAX_INPUT_TOKENS=100,
            DEFAULT_MAX_INPUT_BYTES=100),
        "prior": SimpleNamespace(atomic_private=lambda *args: captured.append(args),
            old=SimpleNamespace(ChatBridge=lambda *args: client,
                make_adapter_class=lambda *args: Adapter)),
        "diagnostic": SimpleNamespace(make_diagnostic_adapter_class=lambda *args: Adapter),
    }
    result = runner._question_worker(loaded, budget, cap,
        {"question_id": "invented"}, 0, tmp_path, 100)
    assert result == {"projection": None, "accounting": None,
                      "stop_code": "question_failure"}
    assert budget.snapshot()["stop_code"] == "question_failure"
    assert budget.snapshot().get("first_failure") is None
    assert budget.snapshot()["turns"] == 0
    assert captured == []
    assert closed == ["adapter", "client"]
