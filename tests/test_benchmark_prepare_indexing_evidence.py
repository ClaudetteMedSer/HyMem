"""Exercise the real receipt decoration and archive serialization boundaries."""
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks import msc_adapter as msc, locomo_adapter as locomo
from benchmarks import msc_registry, locomo_registry
from benchmarks.archive_evidence import validate_scoped_indexing
from benchmarks.strictness import (
    BenchmarkIntegrityError, IndexingConvergenceError, sanitize_for_artifact,
)
from tests.archive_evidence_fixtures import scoped_indexing, skipped_indexing


def _offline_adapter(tmp_path):
    from hymem import HyMemConfig
    adapter = msc.MSCAdapter(HyMemConfig(root=tmp_path).db_path, sim=True)
    adapter.open()
    adapter.pipeline_llm.fixtures.update({
        "typed user-profile facts": '{"items":[]}',
        "You analyze one conversation session": '{"episodes":[],"summary":"","procedures":[]}',
        "single pass": '{"triples":[],"markers":[],"complete":true}',
    })
    return adapter


@pytest.mark.parametrize("module,validate", [
    (msc, msc_registry.validate_msc_artifact),
    (locomo, locomo_registry._validate_strict_locomo),
])
def test_actual_simulation_pipeline_to_registry_without_receipt_seams(
    monkeypatch, tmp_path, module, validate,
):
    # No loader, adapter, indexing, recall, evaluator, or receipt replacement.
    monkeypatch.setattr(sys, "argv", [module.__file__, "--sim", "--no-dream",
        "--sample", "1", "--results-dir", str(tmp_path / "results"),
        "--out", str(tmp_path / "legacy.json")])
    module.main()
    artifact = json.loads(next((tmp_path / "results").glob("*-strict-*.json")).read_text())
    validate(artifact)
    assert artifact["manifest"]["scored_run"] is False
    assert artifact["execution"]["counts"]["failed"] == 0
    receipts = artifact["execution"]["segments"][0]["indexing_runs"]
    assert len(receipts) == 1
    assert receipts[0]["summary"]["store_build_receipt"]["status"] == "not_published_non_comparable"


@pytest.mark.parametrize("scope", ["msc:q1", "locomo:q1"])
def test_actual_prepare_indexing_published_and_reused_store_receipts(tmp_path, scope):
    from tests.test_msc_checkpoint_resume import _example
    args = SimpleNamespace(sim=False, no_dream=False, dream_per_session=False,
                           indexing_max_cycles=10, indexing_timeout_s=30.0)
    # A real temporary memory store with its maintained offline producer. Only
    # the provider is a stub; ingest, dream, publication and attestation run.
    adapter = _offline_adapter(tmp_path)
    try:
        published = msc.prepare_indexing(adapter, _example("q1"), args, scope_id=scope)
        published = json.loads(json.dumps(sanitize_for_artifact(published)))
        assert validate_scoped_indexing(published, scope_id=scope, config=vars(args)) is True
        assert published["store_build_receipt"]["status"] == "published"
        reused = msc.prepare_indexing(adapter, _example("q1"), args, scope_id=scope, reuse=True)
        reused = json.loads(json.dumps(sanitize_for_artifact(reused)))
        assert validate_scoped_indexing(reused, scope_id=scope, config=vars(args)) is True
        assert reused["store_build_receipt"]["status"] == "validated"
        assert reused["store_build_receipt"]["indexing_sha256"] == published["store_build_receipt"]["indexing_sha256"]
        assert reused["convergence_count"] > published["convergence_count"]
    finally:
        adapter.close()


@pytest.mark.parametrize("scope", ["msc:q1", "locomo:q1"])
@pytest.mark.parametrize("failure", ["missing_store", "publication", "convergence", "skipped_reuse"])
def test_actual_prepare_indexing_failure_envelopes_remain_readable(
    monkeypatch, tmp_path, scope, failure,
):
    from tests.test_msc_checkpoint_resume import _example
    args = SimpleNamespace(sim=False, no_dream=False, dream_per_session=False,
                           indexing_max_cycles=10, indexing_timeout_s=30.0)
    adapter = _offline_adapter(tmp_path)
    try:
        if failure == "publication":
            def reject_publication(*_a, **_k):
                raise OSError("private publication detail")
            monkeypatch.setattr(adapter, "publish_store_build_receipt", reject_publication)
        elif failure == "convergence":
            # The real dream/convergence writer emits and decorates the fault.
            from hymem import HyMem
            def fail_dream(_self):
                raise RuntimeError("private convergence detail")
            monkeypatch.setattr(HyMem, "dream", fail_dream)
        elif failure == "skipped_reuse":
            args.no_dream = True
        with pytest.raises(IndexingConvergenceError) as caught:
            msc.prepare_indexing(adapter, _example("q1"), args, scope_id=scope,
                                 reuse=failure in {"missing_store", "skipped_reuse"})
        summary = json.loads(json.dumps(sanitize_for_artifact(caught.value.summary)))
        assert summary["failure_reason"] == {
            "missing_store": "missing_store_build_receipt",
            "publication": "store_build_receipt_publication_failed",
            "convergence": "cycle_exception:RuntimeError",
            "skipped_reuse": "skipped_indexing_reused_store",
        }[failure]
        assert summary["scope_id"] == scope
        assert summary["status"] == "failed_before_scoring"
        assert "private" not in json.dumps(summary)
        assert validate_scoped_indexing(summary, scope_id=scope, config=vars(args), failed=True) is False
        with pytest.raises(BenchmarkIntegrityError):
            validate_scoped_indexing(summary, scope_id=scope, config=vars(args))
        summary["scope_id"] = "msc:other"
        with pytest.raises(BenchmarkIntegrityError):
            validate_scoped_indexing(summary, scope_id=scope, config=vars(args), failed=True)
    finally:
        adapter.close()


@pytest.mark.parametrize("skipped", [False, True])
@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("store_build_receipt"),
    lambda r: r.update(store_build_receipt=[]),
    lambda r: r["store_build_receipt"].update(version="obsolete"),
    lambda r: r["store_build_receipt"].update(file="unavailable.json"),
    lambda r: r["store_build_receipt"].update(status="validated"),
    lambda r: r["store_build_receipt"].update(identity_sha256=[]),
    lambda r: r["store_build_receipt"].update(material_state_sha256=True),
    lambda r: r["store_build_receipt"].update(indexing_sha256="sha256:" + "f" * 64),
])
def test_store_pointer_shape_lifecycle_and_available_digest_binding(skipped, mutate):
    args = SimpleNamespace(sim=skipped, no_dream=skipped,
                           indexing_max_cycles=100, indexing_timeout_s=3600.0)
    summary = skipped_indexing("msc:q1") if skipped else scoped_indexing("msc:q1", args)
    assert validate_scoped_indexing(summary, scope_id="msc:q1", config=vars(args)) is not skipped
    mutate(summary)
    with pytest.raises(BenchmarkIntegrityError):
        validate_scoped_indexing(summary, scope_id="msc:q1", config=vars(args))
