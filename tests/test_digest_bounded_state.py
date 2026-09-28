"""State/coverage integration controls, not model-semantic accuracy tests.

Scripted primary candidates and explicit scripted verifier decisions drive the
real dream, generation, staging, publication, health and retrieval machinery.
No judge is silently treated as correct: these tests establish state behavior
given the supplied decisions, not fidelity of a real bounded-summary model.
The retrieval control exercises scoped canonical-message lexical recovery,
not semantic recall, derived-item completeness, or a full LME benchmark. The
reopen controls reuse one producer object, isolating policy identity from the
deliberately unstable identity of otherwise unknown clients across processes.
"""

from contextlib import closing
from dataclasses import replace
import json
import re

import pytest

from benchmarks.strictness import IndexingConvergenceError, converge_indexing
from hymem import HyMem
from hymem.core.message_records import canonical_message_record
from hymem.dreaming import digest
from hymem.dreaming.lossless import validate_message_coverage_artifact
from hymem.dreaming.summary_policy import (
    BOUNDED_HIGHLIGHTS_V1, DIAGNOSIS, GENERATION, LEGACY_COMPLETE_V1, VERIFICATION,
)
from hymem.query.augment import _episode_search
from tests.digest_verification_fixtures import synthetic_fidelity_result
from tests.test_lossless_digest import RollingLLM, _quiet_cfg


class PolicyStateLLM(RollingLLM):
    """Same client identity across policy switches, with explicit decisions."""

    def __init__(self, *, artifacts=False):
        super().__init__(emit_slice_artifacts=artifacts)
        self.primary_failure = False
        self.summary_rejected = False
        self.summary_override = None
        self.primary_policies = []

    def complete(self, request):
        for policy in (LEGACY_COMPLETE_V1, BOUNDED_HIGHLIGHTS_V1):
            if request.system == digest.digest_system_for_policy(
                VERIFICATION, summary_policy=policy,
            ):
                self.calls.append(request)
                packet = json.loads(request.user)
                result = synthetic_fidelity_result(
                    len(packet["items"]), len(packet["procedure_items"]),
                )
                if self.summary_rejected:
                    result["summary_content"][0]["verdict"] = "unsupported"
                return json.dumps(result)
            if request.system == digest.digest_system_for_policy(
                DIAGNOSIS, summary_policy=policy,
            ):
                self.calls.append(request)
                # No invented source-linked repair hint or silent positive.
                return '{"issues":[]}'
            for granular in (False, True):
                if request.system == digest.digest_system_for_policy(
                    GENERATION, summary_policy=policy, granular=granular,
                ):
                    self.primary_policies.append(policy)
                    if self.primary_failure:
                        self.calls.append(request)
                        return "not-json"
                    self.artifact_label = (
                        "legacypublication" if policy == LEGACY_COMPLETE_V1
                        else "boundedprivate"
                    )
                    result = json.loads(super().complete(request))
                    result["summary"] = self.summary_override or (
                        "Legacy synthetic publication remains available."
                        if policy == LEGACY_COMPLETE_V1 else
                        "Bounded synthetic publication remains available."
                    )
                    return json.dumps(result)
        return super().complete(request)


def _state(hy, sid):
    return dict(hy.conn.execute(
        "SELECT digest_cursor_message_id,digest_cursor_partial_message_id,"
        "digest_cursor_offset,digest_cursor_prompt_version,"
        "digest_published_generation,auto_summary,auto_summary_message_id,"
        "auto_summary_partial_message_id,auto_summary_message_offset,"
        "digest_retry_count,digest_quarantined,coverage_message_id "
        "FROM sessions WHERE id=?", (sid,),
    ).fetchone())


def _publication(state):
    return {key: state[key] for key in (
        "digest_published_generation", "auto_summary", "auto_summary_message_id",
        "auto_summary_partial_message_id", "auto_summary_message_offset",
    )}


def _config_key(hy, client):
    cfg = hy.config
    return digest.digest_config_version(
        prompt_version=cfg.prompt_version,
        episode_prompt_version=digest.active_episode_prompt_version(cfg.episode_granularity_enabled),
        max_chars=cfg.dream_digest_max_chars, max_tokens=cfg.dream_digest_max_tokens,
        max_episodes=cfg.dream_max_episodes_per_session if cfg.episode_granularity_enabled else None,
        summary_policy=cfg.digest_summary_policy, client=client,
    )


def _finish(hy, *, maximum=32):
    for _ in range(maximum):
        report = hy.dream()
        assert report.digest_failures == 0
        state = hy.dream_status()
        assert state["quarantined_digests"] == 0
        assert state["malformed_digests"] == 0
        if state["pending_digests"] == 0:
            return
    pytest.fail("scripted digest did not complete within the fixed local bound")


@pytest.mark.parametrize("granular", [False, True])
def test_policy_switch_rebuilds_published_generation_and_same_policy_is_noop(cfg, granular):
    client = PolicyStateLLM(artifacts=True)
    legacy_cfg = _quiet_cfg(cfg, episode_granularity_enabled=granular,
                            digest_summary_policy=LEGACY_COMPLETE_V1)
    sid = "policy-switch-complete"
    with closing(HyMem(legacy_cfg, llm=client)) as hy:
        message_id = hy.log_message(sid, "user", "alpha staging check passed")
        hy.close_session(sid)
        _finish(hy)
        before = _state(hy, sid)
        legacy_key = _config_key(hy, client)
        assert before["digest_cursor_message_id"] == message_id
        assert digest.digest_generation_matches_config(
            before["digest_published_generation"], legacy_key,
        )
    bounded_cfg = replace(legacy_cfg, digest_summary_policy=BOUNDED_HIGHLIGHTS_V1)
    with closing(HyMem(bounded_cfg, llm=client)) as hy:
        status = hy.dream_status()
        assert status["pending_digests"] == 1
        assert status["malformed_digests"] == status["quarantined_digests"] == 0
        assert _state(hy, sid) == before  # A read-only status must not rewrite it.
        assert _config_key(hy, client).replace(
            f"|summary-policy={BOUNDED_HIGHLIGHTS_V1}", "", 1,
        ) == legacy_key
        assert not digest.digest_generation_matches_config(
            before["digest_published_generation"], _config_key(hy, client),
        )
        calls = len(client.primary_policies)
        _finish(hy)
        after = _state(hy, sid)
        assert client.primary_policies[calls:] == [BOUNDED_HIGHLIGHTS_V1]
        assert after["digest_cursor_message_id"] == message_id
        assert after["digest_published_generation"] != before["digest_published_generation"]
        assert digest.digest_generation_matches_config(
            after["digest_published_generation"], _config_key(hy, client),
        )
        assert "Bounded" in after["auto_summary"]
        calls = len(client.primary_policies)
        hy.dream()
        assert _state(hy, sid) == after
        assert len(client.primary_policies) == calls
    # Reopening the same producer/policy is also a no-op, not merely an
    # in-memory optimization of one HyMem instance.
    with closing(HyMem(bounded_cfg, llm=client)) as hy:
        assert hy.dream_status()["pending_digests"] == 0
        hy.dream()
        assert len(client.primary_policies) == calls
        assert _state(hy, sid) == after


@pytest.mark.parametrize("failure", ["primary", "fidelity"])
def test_failed_policy_rebuild_keeps_old_publication_and_never_claims_current(cfg, failure):
    client = PolicyStateLLM(artifacts=True)
    config = _quiet_cfg(cfg, digest_summary_policy=LEGACY_COMPLETE_V1,
                        digest_extraction_max_attempts=3)
    sid = "policy-failed-rebuild"
    with closing(HyMem(config, llm=client)) as hy:
        hy.log_message(sid, "user", "alpha publication source")
        hy.close_session(sid)
        _finish(hy)
        before = _state(hy, sid)
        episodes = [dict(row) for row in hy.conn.execute(
            "SELECT * FROM episodes WHERE session_id=? ORDER BY id", (sid,),
        )]
    client.primary_failure = failure == "primary"
    client.summary_rejected = failure == "fidelity"
    with closing(HyMem(replace(config, digest_summary_policy=BOUNDED_HIGHLIGHTS_V1),
                       llm=client)) as hy:
        report = hy.dream()
        after = _state(hy, sid)
        assert report.digest_failures == 1
        assert _publication(after) == _publication(before)
        for key in ("digest_cursor_message_id", "digest_cursor_partial_message_id",
                    "digest_cursor_offset", "digest_cursor_prompt_version"):
            assert after[key] == before[key]
        assert after["digest_retry_count"] == 1
        assert not digest.digest_generation_matches_config(
            after["digest_cursor_prompt_version"], _config_key(hy, client),
        )
        assert hy.dream_status()["pending_digests"] == 1
        assert [dict(row) for row in hy.conn.execute(
            "SELECT * FROM episodes WHERE session_id=? ORDER BY id", (sid,),
        )] == episodes
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM digest_staging WHERE session_id=?", (sid,),
        ).fetchone()[0] == 0


def test_partial_bounded_walk_cannot_seed_or_mix_into_switch_back(cfg):
    client = PolicyStateLLM(artifacts=True)
    config = _quiet_cfg(cfg, digest_summary_policy=LEGACY_COMPLETE_V1,
                        dream_digest_max_chars=300)
    sid = "policy-mid-walk-switch"
    with closing(HyMem(config, llm=client)) as hy:
        message_id = hy.log_message(sid, "user", "alpha " + "retained source " * 65)
        hy.close_session(sid)
        _finish(hy)
        old = _state(hy, sid)
    with closing(HyMem(replace(config, digest_summary_policy=BOUNDED_HIGHLIGHTS_V1),
                       llm=client)) as hy:
        first = hy.dream()
        partial = _state(hy, sid)
        assert first.digest_failures == 0 and first.budget_exhausted
        assert partial["digest_cursor_partial_message_id"] == message_id
        assert partial["digest_cursor_offset"] > 0
        assert _publication(partial) == _publication(old)
        staged = [dict(row) for row in hy.conn.execute(
            "SELECT generation,summary FROM digest_staging WHERE session_id=?", (sid,),
        )]
        assert len(staged) == 1 and "Bounded" in staged[0]["summary"]
        bounded_generation = staged[0]["generation"]
        assert digest.digest_generation_matches_config(bounded_generation, _config_key(hy, client))
        assert _episode_search(hy.conn, "boundedprivate", top_k=20) == []
        assert _episode_search(hy.conn, "legacypublication", top_k=20)
    with closing(HyMem(config, llm=client)) as hy:
        assert hy.dream_status()["pending_digests"] == 1
        call_offset = len(client.successful_digest_calls)
        first = hy.dream()
        assert first.digest_failures == 0 and first.budget_exhausted
        request = client.successful_digest_calls[call_offset]
        assert re.search(rf"message {message_id} role=user chars=0:\d+/", request.user)
        assert old["auto_summary"] in request.user
        assert "Bounded synthetic publication" not in request.user
        restarted = _state(hy, sid)
        assert _publication(restarted) == _publication(old)
        assert restarted["digest_cursor_prompt_version"] not in {
            old["digest_published_generation"], bounded_generation,
        }
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM digest_staging WHERE session_id=? AND generation=?",
            (sid, bounded_generation),
        ).fetchone()[0] == 0
        _finish(hy)
        final = _state(hy, sid)
        assert final["digest_published_generation"] == restarted["digest_cursor_prompt_version"]
        assert digest.digest_generation_matches_config(final["digest_published_generation"], _config_key(hy, client))
        assert "Legacy" in final["auto_summary"]
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging WHERE session_id=?", (sid,)).fetchone()[0] == 0
        assert {row[0] for row in hy.conn.execute(
            "SELECT digest_generation FROM episodes WHERE session_id=?", (sid,),
        )} == {final["digest_published_generation"]}
        assert _episode_search(hy.conn, "boundedprivate", top_k=20) == []


def test_policy_rebuild_pending_and_quarantine_still_block_strict_convergence(cfg):
    client = PolicyStateLLM()
    config = _quiet_cfg(cfg, digest_summary_policy=LEGACY_COMPLETE_V1,
                        digest_extraction_max_attempts=2)
    sid = "policy-strict-failure"
    with closing(HyMem(config, llm=client)) as hy:
        hy.log_message(sid, "user", "alpha original source")
        hy.close_session(sid)
        _finish(hy)
        published = _publication(_state(hy, sid))
    client.summary_rejected = True
    with closing(HyMem(replace(config, digest_summary_policy=BOUNDED_HIGHLIGHTS_V1),
                       llm=client)) as hy:
        with pytest.raises(IndexingConvergenceError) as pending:
            converge_indexing(hy.dream, status=hy.dream_status,
                              max_cycles=1, timeout_s=30, require_healthy=True)
        assert pending.value.summary["healthy"] is False
        assert pending.value.summary["complete"] is False
        assert pending.value.summary["final_status"]["pending_digests"] == 1
        with pytest.raises(IndexingConvergenceError) as quarantined:
            converge_indexing(hy.dream, status=hy.dream_status,
                              max_cycles=1, timeout_s=30, require_healthy=True)
        assert quarantined.value.summary["healthy"] is False
        assert quarantined.value.summary["failure_reason"] == "quarantined_extraction"
        assert quarantined.value.summary["final_status"]["quarantined_digests"] == 1
        assert _publication(_state(hy, sid)) == published
        count = len(client.primary_policies)
        hy.dream()
        assert len(client.primary_policies) == count
        assert hy.dream_status()["quarantined_digests"] == 1


def test_omitted_highlight_topic_remains_exact_scoped_retrievable_after_raw_pruning(cfg):
    client = PolicyStateLLM()
    client.summary_override = "A staging health check passed."
    config = _quiet_cfg(cfg, digest_summary_policy=BOUNDED_HIGHLIGHTS_V1,
                        message_retention_days=1)
    content = "A staging health check passed; separately, my glacierglove repair kit arrived."
    sid, peer, workspace = "retained-owner", "owner-a", "space-a"
    with closing(HyMem(config, llm=client)) as hy:
        message_id = hy.log_message(sid, "user", content,
            created_at="2020-01-01T00:00:00Z", source_peer_id=peer,
            source_workspace_id=workspace)
        hy.close_session(sid)
        for other_sid, other_peer, other_workspace in (
            ("same-space-other-owner", "owner-b", workspace),
            ("other-space", peer, "space-b"),
        ):
            hy.log_message(other_sid, "user", content,
                created_at="2020-01-01T00:00:00Z", source_peer_id=other_peer,
                source_workspace_id=other_workspace)
            hy.close_session(other_sid)
        proof_before = dict(hy.conn.execute(
            "SELECT mc.*,c.text FROM message_retention_coverage mc "
            "JOIN chunks c ON c.id=mc.chunk_id WHERE mc.message_id=?", (message_id,),
        ).fetchone())
        _finish(hy)
        assert _state(hy, sid)["auto_summary"] == client.summary_override
        assert "glacierglove" not in _state(hy, sid)["auto_summary"]
        assert hy.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        proof_after = dict(hy.conn.execute(
            "SELECT mc.*,c.text FROM message_retention_coverage mc "
            "JOIN chunks c ON c.id=mc.chunk_id WHERE mc.message_id=?", (message_id,),
        ).fetchone())
        assert proof_after == proof_before
        proof = validate_message_coverage_artifact(hy.conn,
            message_id=message_id, chunk_id=proof_after["chunk_id"],
            coverage_version=proof_after["coverage_version"])
        assert (proof.content, proof.role, proof.session_id, proof.source_peer_id,
                proof.source_workspace_id) == (content, "user", sid, peer, workspace)
        canonical, fingerprint, hash_version, record_version = canonical_message_record(
            message_id=message_id, session_id=sid, role="user", content=content,
            source_created_at=proof.source_created_at, source_peer_id=peer,
            source_workspace_id=workspace)
        assert proof_after["text"] == canonical
        assert (proof_after["message_content_hash"], proof_after["hash_version"],
                proof_after["record_version"]) == (fingerprint, hash_version, record_version)
        selected = hy.augment("glacierglove", source_session_id=sid,
            source_peer_id=peer, source_workspace_id=workspace).message_hits
        assert [(hit.message_id, hit.session_id, hit.role, hit.text,
                 hit.source_peer_id, hit.source_workspace_id) for hit in selected] == [
            (message_id, sid, "user", content, peer, workspace)]
        assert selected[0].score_kind == "coverage_lexical"
        for wrong_scope in (
            {"source_session_id": sid, "source_peer_id": "owner-b", "source_workspace_id": workspace},
            {"source_session_id": sid, "source_peer_id": peer, "source_workspace_id": "space-b"},
            {"source_session_id": "missing-session", "source_peer_id": peer, "source_workspace_id": workspace},
        ):
            assert hy.augment("glacierglove", **wrong_scope).message_hits == []
