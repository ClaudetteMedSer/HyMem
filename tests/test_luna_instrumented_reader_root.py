"""Independent finite reader parity and real-checkpoint completion controls."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_timeout_v3 as observer
from tools.diagnostics import luna_lme_diagnostic_progress_v9 as reader
from tools.diagnostics import luna_lme_diagnostic_v8 as runner
from tools.diagnostics.tests import test_luna_lme_diagnostic_progress_v1 as prior


@pytest.fixture
def summary(monkeypatch):
    path = Path(__file__).with_name("test_luna_lme_observer_root.py")
    spec = importlib.util.spec_from_file_location("root_reader_actual_wire", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    with monkeypatch.context() as scope:
        _, client, _, _, _, request = fixture.wire(scope)
        try:
            client.complete(request)
            return client.diagnostic_summary()
        finally:
            client.close()


def accepts(projector, value):
    try:
        return projector(value) is not None
    except (ValueError, TypeError, KeyError):
        return False


def test_actual_success_and_all_finite_observation_families(summary):
    assert reader._timeout_summary(summary) == summary
    original = summary["last_record"]
    shapes = [original["observed"], dict(basis="consumed_observed_shape",
        events_consumed=0, completed_seen=False, final_seen=False, final_count=0,
        usage_update_count=0, usage_state="absent", last_event_family=None)]
    for family in observer.warm.v8.v7.v6.v5.v4._FAMILIES:
        shapes.append(dict(basis="consumed_observed_shape", events_consumed=3,
            completed_seen=False, final_seen=True, final_count=2,
            usage_update_count=1, usage_state="positive", last_event_family=family))
    for shape in shapes:
        record = deepcopy(original)
        record.update(status="failure", known_usage=False, failure_code="timeout", observed=shape)
        assert observer._project_record(record) is not None
        assert reader._timeout_record(record) == record
    with pytest.raises(ValueError):
        reader._turn_projection(original["observed"])


def test_record_projector_matches_observer_on_field_mutations(summary):
    record = summary["last_record"]
    mutations = [None, True, False, -1, 0, 1, 1.5, 4097, 1000001,
                 float("nan"), float("inf"), "PRIVATE", [], {}, {"private": "PRIVATE"}]
    for key in record:
        for value in mutations:
            changed = deepcopy(record)
            changed[key] = value
            assert accepts(reader._timeout_record, changed) == accepts(observer._project_record, changed), (key, value)
    for key in record["observed"]:
        for value in mutations:
            changed = deepcopy(record)
            changed["observed"][key] = value
            assert accepts(reader._timeout_record, changed) == accepts(observer._project_record, changed), (key, value)
    for name in (None, "observed", "phase_seconds"):
        changed = deepcopy(record)
        (changed if name is None else changed[name])["private"] = "PRIVATE"
        assert not accepts(reader._timeout_record, changed)


def test_summary_projector_parity_and_source_derived_failure_vocabulary(summary):
    for key in summary:
        for value in (None, True, False, -1, 0, 1.0, "PRIVATE", [], {}, float("nan")):
            changed = deepcopy(summary)
            changed[key] = value
            assert accepts(reader._timeout_summary, changed) == accepts(observer._project_summary, changed), (key, value)
    for event in observer.warm.v8.v7.v6.v5.v4.v3._EVENTS:
        changed = deepcopy(summary["last_record"])
        changed.update(status="failure", known_usage=False,
                       failure_code="unexpected_notification:" + event.replace("/", "_"))
        if observer._project_record(changed) is not None:
            assert reader._timeout_record(changed) == changed


@pytest.fixture
def finished(tmp_path, monkeypatch, summary):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(reader, "ROOT_UID", os.getuid())
    monkeypatch.setattr(reader, "_root", lambda value: value)
    receipt = {"selected_count": 4}
    monkeypatch.setattr(reader, "_receipt", lambda *args: receipt)
    monkeypatch.setattr(reader, "_runtime", lambda *args: "clean_exit")
    monkeypatch.setattr(reader, "_failed_exit_cleanup", lambda *args: False)
    monkeypatch.setattr(prior, "reader", reader)
    def manifest(ids):
        loaded = {"strictness": SimpleNamespace(content_hash=reader._canonical_hash),
                  "diagnostic": SimpleNamespace(MODE=reader.MODE)}
        limits = {key: dict(zip(("turns", "known_tokens", "seconds"), values))
                  for key, values in runner.MAX_LIMITS.items()}
        limits.update(indexing_seconds=10800, workers=4)
        return runner._identity_manifest(loaded, [{"question_id": q} for q in ids],
                                         limits, runner.DIAGNOSTIC_HELPER_SHA256)
    monkeypatch.setattr(prior, "_manifest", manifest)
    checkpoint = prior._frozen_checkpoint(root)
    terminal = prior._terminal(checkpoint)
    terminal["canary"]["completion_calls"] = 2
    terminal["budget"].update(turns=10, resource_fault=None, first_failure=None,
        resource_observation={"current": 4, "peak": 132, "limit": 256, "denials": 0})
    terminal["timeout_observations"] = {slot: {"status": "observed", "summary": deepcopy(summary)}
                                        for slot in reader._TIMEOUT_SLOTS}
    prior._json(root/"launch-attempt.json", {"receipt_sha256": "a"*64, "one_shot": True})
    prior._json(root/"launch-command-result.json", {"returncode": 0})
    return root, terminal


def test_real_checkpoint_clean_reconciles_usage_but_not_perfect_quality(finished):
    root, terminal = finished
    prior._json(root/"run/diagnostic-result.json", terminal)
    result = reader.inspect(root, "a"*64)
    assert result["completed_diagnostic_and_clean"] is True
    assert result["scored_count"] == 4 and result["correct_count"] == 2
    assert result["strict_indexing_healthy_for_all"] is False
    assert result["canary_model_gold_match"] is False
    assert result["summary_degraded_sessions_total"] == 1
    assert result["known_turns"] == 10


@pytest.mark.parametrize("mutation", ["missing", "unknown", "sum", "canary", "usage", "cleanup"])
def test_incomplete_observation_or_cleanup_never_clean(finished, monkeypatch, mutation):
    root, terminal = finished
    slots = terminal["timeout_observations"]
    if mutation == "missing": slots.pop("question.0.ordinary")
    elif mutation == "unknown": slots["question.0.ordinary"] = {"status": "unknown"}
    elif mutation == "sum": terminal["budget"]["turns"] += 1
    elif mutation == "canary": terminal["canary"]["completion_calls"] += 1
    elif mutation == "usage": terminal["budget"]["usage_complete"] = False
    elif mutation == "cleanup": monkeypatch.setattr(reader, "_runtime", lambda *args: "unverified")
    prior._json(root/"run/diagnostic-result.json", terminal)
    if mutation == "missing":
        with pytest.raises(ValueError): reader.inspect(root, "a"*64)
    else:
        result = reader.inspect(root, "a"*64)
        assert result["completed_diagnostic_and_clean"] is False
        if mutation == "unknown":
            assert result["timeout_observations"]["question.0.ordinary"] == {"status": "unknown"}


def test_maximum_fixed_summary_payload_fits_terminal_bound(finished):
    _, terminal = finished
    for entry in terminal["timeout_observations"].values():
        value = entry["summary"]
        value.update(calls=1000000, successes=999999, failures=1,
                     counts_saturated=False)
        failed = deepcopy(value["last_record"])
        failed.update(status="failure", known_usage=False, failure_code="timeout")
        value["last_record"] = failed
        value["first_failure"] = {"call_index": 1000000,
            "call_index_saturated": False, "record": deepcopy(failed)}
        value["timing_seconds"] = {"total": 1000000.0,
            "phases": {key: 1000000.0 for key in reader._TIMEOUT_PHASES}}
        value["timing_saturated"] = True
        assert reader._timeout_summary(value)
    assert len(json.dumps(terminal, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) < 128000
