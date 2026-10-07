"""Offline source, ten-view ledger, and terminal checkpoint controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

from benchmarks import chatgpt_plan_lme_v1 as bridge
from benchmarks import strictness
from tools.diagnostics import siwc_lme_diagnostic_v1 as runner
from hymem.extraction.llm import LLMRequest


REPO = Path(__file__).resolve().parents[3]
ACCEPTED = Path("/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle")
PRIOR = Path("/private/tmp/hymem-repaired-four-root-dtBv2p/bundle")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(relative: str, expected: str) -> Path:
    for source in (REPO / relative, PRIOR / "code" / relative):
        if source.is_file() and _hash(source) == expected:
            return source
    raise AssertionError(f"pinned source unavailable: {relative}")


def test_real_candidate_map_and_complete_source_graph():
    source_map = ACCEPTED / "source-map.json"
    assert _hash(source_map) == runner.ACCEPTED_INVENTORY_SHA256
    values = json.loads(source_map.read_text())
    entries = values.get("source_sha256", values)
    assert len(entries) == runner.ACCEPTED_FILES
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == runner.ACCEPTED_MAP_SHA256
    assert set(runner.PINS).isdisjoint(runner.SIWC_PINS)
    for relative, expected in entries.items():
        assert _hash(ACCEPTED / "candidate" / relative) == expected
    for relative, expected in {**runner.PINS, **runner.SIWC_PINS}.items():
        assert _hash(_source(relative, expected)) == expected
    assert _hash(REPO / "benchmarks/lme_diagnostic.py") == runner.DIAGNOSTIC_HELPER_SHA256
    assert "tools/diagnostics/lme_chatgpt_plan_catalog_v1.py" in runner.SIWC_PINS
    assert "tools/diagnostics/lme_chatgpt_plan_catalog_v2.py" not in runner.SIWC_PINS


def test_fresh_isolated_source_only_import_never_constructs_owner(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(ACCEPTED / "candidate", bundle / "candidate")
    shutil.copy2(ACCEPTED / "source-map.json", bundle / "source-map.json")
    sources = {**runner.PINS, **runner.SIWC_PINS,
        "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256}
    for relative, digest in sources.items():
        destination = bundle / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_source(relative, digest), destination)
    destination = bundle / "code" / runner.RUNNER_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(runner.__file__), destination)
    script = """
import importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
path = root / "code/tools/diagnostics/siwc_lme_diagnostic_v1.py"
spec = importlib.util.spec_from_file_location("isolated_siwc_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
loaded = runner.import_source_only(root, root / "source-map.json",
    runner.ACCEPTED_INVENTORY_SHA256)
assert loaded["source_only"] is True
assert loaded["siwc"].owner.CredentialBroker.__module__.startswith("tools.")
assert "broker" not in loaded
print(json.dumps({"source_only": True, "candidate_files": runner.ACCEPTED_FILES}))
"""
    completed = subprocess.run(["/opt/anaconda3/bin/python3.13", "-I", "-B", "-c", script, str(bundle)],
        text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stderr[-1500:]
    assert json.loads(completed.stdout) == {"source_only": True, "candidate_files": 514}


class FakeLedger:
    def __init__(self):
        self.questions = {}
        self.halted = []

    def register(self, key, limits):
        if key in self.questions:
            raise ValueError("duplicate_question")
        self.questions[key] = {"turns": 0, "known_tokens": 0, "usage_complete": True}

    def snapshot(self):
        return {"questions": self.questions}

    def halt(self, code):
        self.halted.append(code)


class FakeClient:
    def __init__(self, broker, budget, key, limits):
        assert broker is BROKER
        budget.register(key, limits)
        self.budget, self.key = budget, key
        self.closed = False

    def diagnostic_summary(self):
        row = self.budget.snapshot()["questions"][self.key]
        return {"schema": "siwc_lme_summary_v1", "calls": 0, "successes": 0,
            "failures": 0, "internal_http_attempts": 0,
            "provider_internal_retries_known": False,
            "admitted_turns": row["turns"], "known_tokens": row["known_tokens"],
            "usage_complete": row["usage_complete"],
            "timing_seconds": {"total": 0.0, "admission": 0.0, "http": 0.0},
            "timing_saturated": False, "first_failure": None,
            "last_failure_code": None}

    def close(self):
        self.closed = True


BROKER = object()
FAKE_SIWC = SimpleNamespace(SIWCLMEClient=FakeClient,
    validate_summary_projection=bridge.validate_summary_projection,
    validate_pilot_projection=bridge.validate_pilot_projection)


def test_one_ledger_registration_and_exact_ten_slot_projection(tmp_path):
    budget = FakeLedger()
    observations = runner.ObservationRegistry(FAKE_SIWC, budget)
    loaded = {"siwc": FAKE_SIWC, "broker": BROKER}
    clients = []
    for index, key in [(None, "canary"), *[(i, f"q-{i:04d}") for i in range(4)]]:
        prefix = "canary" if index is None else f"question.{index}"
        clients.append(runner.make_dual(loaded, budget, key, object(), tmp_path,
            observations, prefix))
        observations.capture_pair(prefix)
    assert len(budget.questions) == 5
    assert len(observations.snapshot()) == 10
    projection = observations.pilot_projection()
    assert projection["aggregate"] == {"calls": 0, "successes": 0, "failures": 0,
        "internal_http_attempts": 0, "admitted_turns": 0, "known_tokens": 0}
    for client in clients:
        client.close()
        assert client.ordinary.closed and client.staged.closed
    with pytest.raises(ValueError, match="shared_question_registration_invalid"):
        runner.RegistrationAlias(budget, "canary", object()).register("different", object())


def test_missing_slot_remains_unknown_without_erasing_observed_summary():
    budget = FakeLedger()
    budget.register("canary", object())
    observations = runner.ObservationRegistry(FAKE_SIWC, budget)
    observations.register("canary.ordinary", FakeClient.__new__(FakeClient))
    client = observations._clients["canary.ordinary"]
    client.budget, client.key = budget, "canary"
    observations.capture("canary.ordinary")
    table = observations.snapshot()
    assert table["canary.ordinary"]["status"] == "observed"
    assert table["question.3.structured"] == {"status": "unknown"}
    with pytest.raises(ValueError, match="siwc_observation_incomplete"):
        observations.pilot_projection()


def test_real_bridge_mock_broker_and_transport_reconcile_five_ledgers(monkeypatch, tmp_path):
    class MockBroker:
        identity_digest = runner.GRANT_IDENTITY_SHA256
        calls = 0

        def acquire(self, *, caller_deadline):
            assert caller_deadline > time.monotonic()
            self.calls += 1
            return bridge.owner.CredentialLease("invented-access-token", int(time.time()) + 600)

    seen = []
    def fake_response(credentials, system, user, schema, *, timeout):
        assert credentials.access_token == "invented-access-token"
        assert (system, user, schema) == ("invented system", "invented user", None)
        assert 0 < timeout <= bridge.MAX_INVOCATION
        seen.append(1)
        return bridge.transport.Completed("invented answer", 3, 2, 5, 0, 0)

    original_client = bridge.SIWCLMEClient
    class MockClient(original_client):
        def __init__(self, broker, budget, question_id, question_limits):
            super().__init__(broker, budget, question_id, question_limits,
                response_call=fake_response)

    monkeypatch.setattr(bridge.owner, "CredentialBroker", MockBroker)
    monkeypatch.setattr(bridge, "SIWCLMEClient", MockClient)
    broker = MockBroker()
    budget = bridge.SharedBudget(bridge.warm.BudgetLimits(50, 100_000, 120),
        max_in_flight=4)
    observations = runner.ObservationRegistry(bridge, budget)
    loaded = {"siwc": bridge, "broker": broker}
    for index, key in [(None, "canary"), *[(i, f"q-{i:04d}") for i in range(4)]]:
        prefix = "canary" if index is None else f"question.{index}"
        dual = runner.make_dual(loaded, budget, key,
            bridge.warm.BudgetLimits(10, 10_000, 100), tmp_path,
            observations, prefix)
        assert dual.complete(LLMRequest("invented system", "invented user")) == "invented answer"
        observations.capture_pair(prefix)
        dual.close()
    projection = observations.pilot_projection()
    snapshot = budget.snapshot()
    assert broker.calls == len(seen) == 5
    assert projection["aggregate"]["admitted_turns"] == snapshot["turns"] == 5
    assert projection["aggregate"]["known_tokens"] == snapshot["known_tokens"] == 25
    assert projection["aggregate"]["internal_http_attempts"] == 5
    assert projection["aggregate"]["failures"] == 0
    assert len(observations.snapshot()) == 10


def test_owner_open_failure_writes_finite_checkpoint_and_result(tmp_path, monkeypatch):
    class OwnerError(Exception):
        def __init__(self, code):
            self.code = code
    dataset = tmp_path / "invented-dataset.json"
    dataset.write_text("[]")
    original_sha = runner._sha
    monkeypatch.setattr(runner, "_sha", lambda path:
        runner.DATASET_SHA256 if path == dataset else original_sha(path))
    class Prior:
        @staticmethod
        def atomic_private(path, value):
            path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    loaded = {"source_only": False, "dataset": dataset,
        "questions": [{"question_id": f"invented-{i}"} for i in range(4)],
        "strictness": strictness, "diagnostic": SimpleNamespace(MODE="invented"),
        "siwc": SimpleNamespace(SharedBudget=bridge.SharedBudget,
            owner=SimpleNamespace(OwnerError=OwnerError),
            SIWCLMEClient=FakeClient,
            validate_summary_projection=bridge.validate_summary_projection,
            validate_pilot_projection=bridge.validate_pilot_projection),
        "warm": bridge.warm, "prior": Prior()}
    def denied():
        raise OwnerError("binding_invalid")
    output = tmp_path / "run"
    result = runner.run_campaign(loaded, output=output,
        campaign_limits=bridge.warm.BudgetLimits(*runner.MAX_LIMITS["campaign"]),
        canary_limits=bridge.warm.BudgetLimits(*runner.MAX_LIMITS["canary"]),
        question_limits=bridge.warm.BudgetLimits(*runner.MAX_LIMITS["question"]),
        indexing_seconds=10_800, workers=4,
        helper_sha256=runner.DIAGNOSTIC_HELPER_SHA256,
        containment=lambda _: True, broker_factory=denied)
    assert result["campaign_stop"] == "binding_invalid"
    assert result["owner_failure"] == {"phase": "owner_open", "code": "binding_invalid"}
    assert result["scored_count"] == 0
    assert result["diagnostic_complete"] is False
    assert result["siwc_pilot_projection"] is None
    assert len(result["siwc_observations"]) == 10
    assert all(item["status"] == "unknown" for item in result["siwc_observations"].values())
    assert (output / "diagnostic-result.json").is_file()
    assert (output / "diagnostic-checkpoint.json").is_file()
