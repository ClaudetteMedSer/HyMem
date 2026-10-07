"""Independent diagnostic policy controls, including the frozen runtime seam."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import lme_diagnostic as diagnostic


@pytest.mark.parametrize("details", [
    '["left:call_failure"]', '["left:unrecognized_failure"]',
    '["left:branch_incomplete","left.right:call_failure"]',
    '["left:parse_failure","left.diagnostics:truncated"]',
    '["left.response:truncated_json"]',
])
def test_semantic_outer_reason_cannot_hide_unproved_branch(details):
    assert not diagnostic._semantic_reason("response_conflict", details)


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_actual_staged_semantic_verdict_can_be_measured(verdict):
    leaf = "grounding:verdict_" + verdict
    assert diagnostic._semantic_reason("grounding_failure", json.dumps([leaf]))
    assert diagnostic._semantic_reason("branch_incomplete", json.dumps([
        "left:grounding_failure", "left." + leaf,
    ]))


@pytest.mark.parametrize("detail", ["grounding:source_invalid",
    "grounding:support_integrity", "grounding:provider_call_failed",
    "grounding:calls_max_exceeded", "grounding:contract_response_binding"])
def test_grounding_runtime_and_provenance_cannot_be_treated_as_quality(detail):
    assert not diagnostic._semantic_reason("grounding_failure", json.dumps([detail]))


_CANDIDATE = Path("/private/tmp/hymem-staged-v1-root-fZvNIRgs/candidate")
_CHILD = r'''
import copy, importlib.util, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from benchmarks import longmemeval_adapter as lme, lme_protocol as protocol, strictness
from hymem import HyMem, HyMemConfig
from hymem.extraction.llm import StubLLMClient
from hymem.dreaming.summary_state import classify_summary_state
spec = importlib.util.spec_from_file_location("root_diagnostic", sys.argv[2])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cls = mod.make_diagnostic_adapter_class(lme, protocol, strictness,
                                        classify_summary_state, lme.HyMemAdapter)
with tempfile.TemporaryDirectory(prefix="hymem-diagnostic-root-") as directory:
    owner = cls.__new__(cls)
    owner.hy = HyMem(HyMemConfig(root=Path(directory), aggregation_nodes_enabled=False,
        profile_extraction_enabled=False, facts_extraction_enabled=False,
        rules_extraction_enabled=False), llm=StubLLMClient(default="[]"))
    owner.embedding_client = None
    owner.last_indexing_summary = None
    try:
        healthy = owner.dream_and_wait(timeout=20, max_cycles=2)
        assert healthy["outcome"] == "success"
        assert owner.diagnostic_indexing["kind"] == "strict_healthy"
        observed, rows, summaries = mod.coherent_status_and_held(
            owner.hy, strictness, classify_summary_state, None)
        assert rows == summaries == []
        held = {"prompt_version": observed["extraction_cache_key"],
                "phase1_generation_key": observed["phase1_generation_key"],
                "last_failure_reason": "parse_failure",
                "last_failure_details": '[]', "n": 1}
        raw = dict(complete=True, healthy=False, reports=healthy["reports"],
                   cycles=healthy["cycles"], max_cycles=2, timeout_s=20,
                   elapsed_s=healthy["elapsed_s"], failure_reason=None,
                   final_status={**observed, "quarantined_chunks": 1})
        canonical = protocol.canonicalize_lme_indexing_summary(raw)
        assert canonical["outcome"] == "failure"
        try:
            protocol._validate_versioned_indexing(canonical, require_healthy=True,
                                                  allow_failure=False)
        except protocol.BenchmarkIntegrityError:
            pass
        else:
            raise AssertionError("strict mode accepted quarantine")
        decision = mod.classify_semantic_indexing(canonical, [held], [],
            cache_key=observed["extraction_cache_key"],
            generation_key=observed["phase1_generation_key"])
        assert decision["admitted"] and decision["kind"] == "semantic_quarantine"
        assert canonical["healthy"] is False
        # Exercise the actual adapter and canonical validator together, retaining
        # only convergence/status as controlled synthetic failure inputs.
        original_converge = lme.converge_indexing
        original_snapshot = mod.coherent_status_and_held
        try:
            lme.converge_indexing = lambda *a, **k: copy.deepcopy(raw)
            mod.coherent_status_and_held = lambda *a: (dict(raw["final_status"]), [held], [])
            returned = owner.dream_and_wait(timeout=20, max_cycles=2)
            assert returned["outcome"] == "failure"
            assert owner.diagnostic_indexing["admitted"]
            held["last_failure_reason"] = "call_failure"
            try:
                owner.dream_and_wait(timeout=20, max_cycles=2)
            except lme.IndexingConvergenceError:
                pass
            else:
                raise AssertionError("adapter accepted provider failure")
        finally:
            lme.converge_indexing = original_converge
            mod.coherent_status_and_held = original_snapshot
        assert owner.hy.conn.execute("SELECT 1").fetchone()[0] == 1
        print(json.dumps({"real_empty_store": "passed", "strict_control": "passed",
                          "semantic_continuation": "passed", "hard_failure": "passed"}))
    finally:
        owner.hy.close()
'''


def test_actual_frozen_candidate_and_protocol_boundary():
    if not _CANDIDATE.is_dir():
        pytest.skip("optional accepted frozen candidate is unavailable")
    helper = Path(diagnostic.__file__).resolve()
    result = subprocess.run([sys.executable, "-B", "-c", _CHILD,
                             str(_CANDIDATE), str(helper)],
                            capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout.splitlines()[-1]).values()) == {"passed"}
