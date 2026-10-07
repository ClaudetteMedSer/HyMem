"""Root-owned actual client and failure-path metadata checks, offline only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_staged_v6 as staged
from tools.diagnostics import luna_lme_diagnostic_v8 as runner


def graph(tmp_path, *, turns=80):
    cap = staged.warm.BudgetLimits(turns, 100000, 600)
    budget = staged.warm.SharedBudget(cap)
    loaded = {"warm": staged.warm, "staged": staged, "observer": staged.observer,
              "binary": Path("/unused")}
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    registry = runner.ObservationRegistry(staged.observer, budget)
    return cap, budget, loaded, directory, registry


def test_real_dual_clients_share_budget_and_capture_thirty_six_calls(monkeypatch, tmp_path):
    path = Path(__file__).with_name("test_luna_lme_staged_observer_root.py")
    spec = importlib.util.spec_from_file_location("root_dual_observed_wire", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    spare, _, processes, calls, request, batch = fixture.wire(monkeypatch)
    spare.close()
    cap, budget, loaded, directory, registry = graph(tmp_path)
    dual = runner.make_dual(loaded, budget, "q-0000", cap, directory,
                           observations=registry, slot_prefix="question.0")
    ordinary = SimpleNamespace(system="invented system", user="invented text",
        temperature=0.0, max_tokens=32, response_format="json")
    try:
        for _ in range(18):
            assert dual.complete(ordinary) == "{}"
            assert dual.complete_stage(request, batch, "original", False) == "{}"
        registry.capture_pair("question.0")
        observed = registry.snapshot()
        assert len(observed) == 10
        for route in ("ordinary", "structured"):
            value = observed["question.0." + route]
            assert value["status"] == "observed"
            assert value["summary"]["successes"] == 18
            assert value["summary"]["failures"] == 0
            assert value["summary"]["last_record"]["observed"]["completed_seen"] is True
        assert len(calls) == budget.snapshot()["turns"] == 36
        assert len(processes) == 4
        assert budget.snapshot()["known_tokens"] == 36*27
        assert set(budget.snapshot()["questions"]) == {"q-0000"}
        assert dual.ordinary.private_failure_sink is dual.staged.private_failure_sink
        assert dual.ordinary.budget is dual.staged.budget._budget is budget
        assert "private-thread" not in json.dumps(observed)
        observed["question.0.ordinary"]["summary"]["calls"] = -1
        assert registry.snapshot()["question.0.ordinary"]["summary"]["calls"] == 18
    finally:
        dual.close()
    assert all(client.session is None and client.directory is None
               for client in (dual.ordinary, dual.staged))


def test_partial_factory_keeps_first_client_summary_and_closes_it(monkeypatch, tmp_path):
    cap, budget, loaded, directory, registry = graph(tmp_path)
    made = []
    original = staged.observer.TimeoutSubscriptionClient.__init__
    def init(client, *args, **kwargs):
        original(client, *args, **kwargs)
        made.append(client)
    monkeypatch.setattr(staged.observer.TimeoutSubscriptionClient, "__init__", init)
    def rejected(*args, **kwargs):
        raise RuntimeError("PRIVATE-FACTORY-DETAIL")
    monkeypatch.setattr(staged, "StagedSubscriptionClient", rejected)
    with pytest.raises(RuntimeError, match="PRIVATE-FACTORY-DETAIL"):
        runner.make_dual(loaded, budget, "q-0000", cap, directory,
                        observations=registry, slot_prefix="question.0")
    snapshot = registry.snapshot()
    assert len(made) == 1 and made[0].closed is True
    assert snapshot["question.0.ordinary"]["status"] == "observed"
    assert snapshot["question.0.ordinary"]["summary"]["calls"] == 0
    assert snapshot["question.0.structured"] == {"status": "unknown"}
    assert budget.snapshot()["turns"] == 0
    assert "PRIVATE" not in json.dumps(snapshot)


def test_malformed_capture_is_unknown_not_zero_and_preserves_prior_fault(monkeypatch, tmp_path):
    cap, budget, loaded, directory, registry = graph(tmp_path)
    dual = runner.make_dual(loaded, budget, "q-0000", cap, directory,
                           observations=registry, slot_prefix="question.0")
    try:
        registry.capture_pair("question.0")
        budget.record_first_failure("timeout", {"phase": "run"})
        before = deepcopy(budget.snapshot()["first_failure"])
        malformed = dual.ordinary.diagnostic_summary()
        malformed["raw_prompt"] = "PRIVATE"
        monkeypatch.setattr(dual.ordinary, "diagnostic_summary", lambda: malformed)
        registry.capture_pair("question.0")
        snapshot = registry.snapshot()
        assert snapshot["question.0.ordinary"] == {"status": "unknown"}
        assert snapshot["question.0.structured"]["summary"]["calls"] == 0
        assert budget.snapshot()["first_failure"] == before
        assert budget.snapshot()["stopped"] is True
        assert "PRIVATE" not in json.dumps(snapshot)
    finally:
        dual.close()


@pytest.mark.parametrize("close_fails", [False, True])
def test_canary_exception_and_close_error_still_keep_observation(monkeypatch, tmp_path, close_fails):
    path = Path(__file__).with_name("test_luna_lme_staged_observer_root.py")
    spec = importlib.util.spec_from_file_location("root_canary_observed_wire", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    spare, _, _, _, _, _ = fixture.wire(monkeypatch)
    spare.close()
    cap, budget, loaded, directory, registry = graph(tmp_path)
    def extract(recording, *args, **kwargs):
        recording.complete(SimpleNamespace(system="invented", user="invented",
            temperature=0.0, max_tokens=32, response_format="json"))
        raise RuntimeError("PRIVATE-EXTRACTION-DETAIL")
    loaded.update(canary=SimpleNamespace(_CANARY_CONTENT="invented",
        _source_records=lambda: [], EXTRACTION_CANARY_MAX_COMPLETION_CALLS=12),
        chunk=SimpleNamespace(extract_chunk=extract),
        prior=SimpleNamespace(atomic_private=lambda *args: None))
    if close_fails:
        original = runner.DualClient.close
        def close(client):
            original(client)
            raise RuntimeError("PRIVATE-CLOSE-DETAIL")
        monkeypatch.setattr(runner.DualClient, "close", close)
    with pytest.raises(RuntimeError, match="PRIVATE-"):
        runner.run_live_canary(loaded, budget, cap, directory / "canary", observations=registry)
    snapshot = registry.snapshot()
    assert snapshot["canary.ordinary"]["summary"]["successes"] == 1
    assert snapshot["canary.structured"]["summary"]["successes"] == 0
    assert all(client.closed for client in registry._clients.values())
    assert budget.snapshot()["known_tokens"] == 27
    assert "PRIVATE-" not in json.dumps(snapshot)


def test_question_adapter_open_and_close_failure_keep_zero_call_summaries(tmp_path):
    cap, budget, loaded, directory, registry = graph(tmp_path)
    class Adapter:
        def __init__(self, *args, **kwargs): pass
        def open(self): raise RuntimeError("PRIVATE-OPEN-DETAIL")
        def close(self): raise RuntimeError("PRIVATE-CLOSE-DETAIL")
    class IndexingError(Exception): pass
    loaded.update(candidate=Path("/invented-candidate"), request_type=SimpleNamespace,
        protocol=None, strictness=None, summary_classifier=None,
        lme=SimpleNamespace(IndexingConvergenceError=IndexingError),
        prior=SimpleNamespace(atomic_private=lambda *args: None,
            old=SimpleNamespace(ChatBridge=lambda *args: None, make_adapter_class=lambda *args: Adapter)),
        diagnostic=SimpleNamespace(make_diagnostic_adapter_class=lambda *args: Adapter))
    with pytest.raises(RuntimeError, match="question_cleanup_failure"):
        runner._question_worker(loaded, budget, cap, {"question_id": "invented"},
            0, directory, 100, observations=registry)
    snapshot = registry.snapshot()
    assert all(snapshot["question.0." + route]["status"] == "observed"
               for route in ("ordinary", "structured"))
    assert all(snapshot["question.0." + route]["summary"]["calls"] == 0
               for route in ("ordinary", "structured"))
    assert budget.snapshot()["stopped"] is True
    assert all(client.closed for client in registry._clients.values())
    assert "PRIVATE-" not in json.dumps(snapshot)
