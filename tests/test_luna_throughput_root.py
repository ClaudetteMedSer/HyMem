"""Independent controls of the four-worker delta and preserved v1 contracts."""
import hashlib
import importlib.util
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmarks import codex_subscription_concurrent_v2 as concurrent
from tools.diagnostics import luna_subscription_lme_multi_v2 as multi


def _previous(name, monkeypatch):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("independent_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "concurrent", concurrent)
    if hasattr(module, "multi"):
        monkeypatch.setattr(module, "multi", multi)
    return module


def test_frozen_store_and_producer_contract_survives_v2(tmp_path, monkeypatch):
    _previous("test_luna_multi_root", monkeypatch).test_actual_frozen_stores_have_no_cross_question_state(tmp_path)


@pytest.mark.parametrize("unhealthy", [False, True])
def test_v2_real_ledger_preserves_peer_and_score_semantics(tmp_path, monkeypatch, unhealthy):
    _previous("test_luna_multi_root", monkeypatch).test_three_question_campaign_with_actual_shared_budget(tmp_path, monkeypatch, unhealthy)


@pytest.mark.parametrize("failure_on", [1, 2])
def test_v2_progress_failure_never_becomes_extractor_retry(monkeypatch, failure_on):
    _previous("test_luna_multi_root", monkeypatch).test_progress_failure_cannot_trigger_real_frozen_model_retries(failure_on)


def test_v2_aborted_dream_housekeeping_preserves_memory(tmp_path, monkeypatch):
    _previous("test_luna_multi_root", monkeypatch).test_real_frozen_aborted_dream_housekeeping_changes_no_memory(tmp_path)


def test_four_already_admitted_turns_settle_honestly_after_one_unknown():
    budget = concurrent.SharedBudget(concurrent.BudgetLimits(12, 100, 60), max_in_flight=4)
    admission = {"auth": "chatgpt", "model": "gpt-6-luna", "inference_enabled": False,
                 "config_isolation_admitted": True,
                 "quota_windows": [{"remaining_percent": 80}]}
    for key in map(str, range(4)):
        budget.register(key, concurrent.BudgetLimits(3, 50, 60))
        budget.reserve(key)
        budget.before_turn(key, admission)
    rendezvous = threading.Barrier(4, timeout=3)
    def settle(index):
        rendezvous.wait()
        budget.settle(str(index), used=None if index == 0 else 9, turn_started=True)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(settle, range(4)))
    state = budget.snapshot()
    assert state["turns"] == 4 and state["known_tokens"] == 27
    assert state["in_flight"] == state["reserved"] == 0
    assert state["stopped"] and not state["usage_complete"]
    assert state["stop_code"] == "usage_unknown"
    assert all(state["questions"][str(i)]["usage_complete"] for i in (1, 2, 3))
    with pytest.raises(concurrent.ConcurrentStop, match="campaign_stopped"):
        budget.reserve("1")


def test_versioned_delta_and_original_sources_are_pinned():
    root = Path(__file__).resolve().parents[1]
    pins = {
        "benchmarks/codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
        "benchmarks/codex_subscription_concurrent.py": "e5f1eacbb02f809ee6246449b672dfcc839069da226230572913612acece069e",
        "tools/diagnostics/luna_subscription_pilot.py": "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0",
        "tools/diagnostics/luna_subscription_lme_multi.py": "83fb27a9ee86c7f1d25ab7c6775dec31e9540321347971609de47a065734eef5",
    }
    for name, digest in pins.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(Path(concurrent.__file__).read_bytes()).hexdigest() == multi.CONCURRENT_SHA256
    assert multi.SCHEMA.endswith("-v2")
