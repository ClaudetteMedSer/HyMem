"""Independent real-adapter controls for fidelity-held mixed indexing failures."""
from contextlib import closing
from copy import deepcopy
import json

import pytest

from benchmarks import lme_protocol as protocol
from benchmarks.strictness import BenchmarkIntegrityError, IndexingConvergenceError
from hymem import HyMem, HyMemConfig
from hymem.dreaming import digest
from hymem.dreaming.facts import fact_cursor_retry_unit_key, facts_retry_policy_version
from tests.digest_verification_fixtures import synthetic_fidelity_result
from tests.test_benchmark_adapter_strictness import (
    _control_lme_indexing_budget, _empty_indexing_llm, lme,
)


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain", "invalid-verdict"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_root_fidelity_hold_and_fact_quarantine_preserve_failed_lme_receipt(tmp_path, monkeypatch, verdict, cleanup_failure):
    config = HyMemConfig(
        root=tmp_path, aggregation_nodes_enabled=False, episode_granularity_enabled=False,
        profile_extraction_enabled=False, facts_extraction_enabled=True,
        salience_min_chars=1, dream_baseline_budget=0,
    )
    llm = _empty_indexing_llm()
    response = synthetic_fidelity_result()
    response["summary_content"][0]["verdict"] = verdict
    llm.fixtures[digest._DIGEST_FIDELITY_SYSTEM] = json.dumps(response)
    budget = _control_lme_indexing_budget(monkeypatch, status_elapsed=2.0)
    with closing(HyMem(config, llm=llm)) as hy:
        hy.log_message("mixed-root", "user", "PRIVATE_SOURCE_SENTINEL: the rollout finished successfully.")
        retry = fact_cursor_retry_unit_key("mixed-root", None, None, 0)
        identity = facts_retry_policy_version(config, replay_slice_key=retry, client=llm)
        hy.conn.execute(
            "UPDATE sessions SET facts_retry_count=?,facts_retry_config_version=?,facts_quarantined=1 WHERE id=?",
            (config.facts_extraction_max_attempts, identity, "mixed-root"),
        )
        adapter = object.__new__(lme.HyMemAdapter)
        adapter.hy, adapter.embedding_client, adapter.last_indexing_summary = hy, None, None
        if cleanup_failure:
            def fail_cleanup():
                raise RuntimeError("PRIVATE_CLEANUP_SENTINEL /private/secret/path")
            monkeypatch.setattr(hy, "invalidate_query_caches", fail_cleanup)
        with pytest.raises(IndexingConvergenceError) as failed:
            adapter.dream_and_wait(timeout=10, max_cycles=1)
        summary = failed.value.summary
        assert summary is adapter.last_indexing_summary
        assert summary["outcome"] == "failure" and summary["complete"] is False
        assert summary["healthy"] is False and summary["failure"]["code"] == "quarantined_extraction"
        assert summary["cycles"] == budget.dream_calls == len(budget.statuses) == 1
        assert summary["final_status"]["pending"]["pending_digests"] == 1
        assert summary["final_status"]["quarantined"]["quarantined_facts"] == 1
        assert summary["reports"][-1]["digest_failures"] == 1
        assert summary["cleanup_errors"] == ([{
            "stage": "query_cache_invalidation", "exception_type": "RuntimeError",
        }] if cleanup_failure else [])
        assert len([call for call in llm.calls if call.system == digest._DIGEST_FIDELITY_SYSTEM]) == 1
        assert protocol._validate_indexing(summary, allow_incomplete=True) is False
        with pytest.raises(BenchmarkIntegrityError):
            protocol._validate_indexing(summary)
        encoded = json.dumps(summary)
        assert "PRIVATE_SOURCE_SENTINEL" not in encoded and "PRIVATE_CLEANUP_SENTINEL" not in encoded
        row = hy.conn.execute(
            "SELECT digest_cursor_message_id,digest_cursor_partial_message_id,auto_summary FROM sessions WHERE id='mixed-root'",
        ).fetchone()
        assert tuple(row) == (None, None, None)
        for tamper in ("complete", "healthy", "success", "missing_evidence"):
            forged = deepcopy(summary)
            if tamper in {"complete", "healthy"}:
                forged[tamper] = True
            elif tamper == "success":
                forged.update(outcome="success", complete=True, healthy=True, cleanup_errors=[])
                del forged["failure"]
            else:
                forged["final_status"]["quarantined"] = {
                    key: 0 for key in forged["final_status"]["quarantined"]
                }
            with pytest.raises(BenchmarkIntegrityError):
                protocol._validate_indexing(forged, allow_incomplete=True)
