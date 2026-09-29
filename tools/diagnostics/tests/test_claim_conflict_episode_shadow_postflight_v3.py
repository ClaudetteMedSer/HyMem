"""Epoch-scoped progress using real staged digest/profile producer fixtures."""
import hashlib
import sqlite3

import pytest

from hymem import HyMem, HyMemConfig
from tests.test_lossless_digest import RollingLLM
from tools.diagnostics import claim_conflict_episode_shadow_postflight_v3 as checker


class EpochLLM(RollingLLM):
    def complete(self, request):
        from hymem.dreaming.user_profile import USER_PROFILE_SYSTEM
        if request.system == USER_PROFILE_SYSTEM:
            return '{"items":[]}'
        return super().complete(request)


@pytest.fixture
def epoch_store(tmp_path):
    cfg = HyMemConfig(root=tmp_path, aggregation_nodes_enabled=False,
        dream_digest_max_chars=300, profile_extraction_enabled=True,
        facts_extraction_enabled=True)
    first = HyMem(cfg, llm=EpochLLM())
    session = "epoch-progress"
    first.log_message(session, "user", "alpha " + "0123456789" * 200)
    first.close_session(session)
    first.dream()
    baseline = sqlite3.connect(":memory:")
    baseline.row_factory = sqlite3.Row
    first.conn.backup(baseline)
    first.close()
    second_llm = EpochLLM()
    second = HyMem(cfg, llm=second_llm)
    second.dream()
    after = second.conn
    state = after.execute("SELECT * FROM sessions WHERE id=?", (session,)).fetchone()
    prior = baseline.execute("SELECT * FROM sessions WHERE id=?", (session,)).fetchone()
    target = after.execute("SELECT id FROM chunks WHERE session_id=? LIMIT 1", (session,)).fetchone()[0]
    from hymem.dreaming.digest import digest_config_version, active_episode_prompt_version
    from hymem.dreaming.user_profile import profile_config_version
    from hymem.dreaming.facts import facts_config_version
    expected_configs = {
        "digest": digest_config_version(prompt_version=cfg.prompt_version,
            episode_prompt_version=active_episode_prompt_version(cfg.episode_granularity_enabled),
            max_chars=cfg.dream_digest_max_chars, max_tokens=cfg.dream_digest_max_tokens,
            max_episodes=cfg.dream_max_episodes_per_session if cfg.episode_granularity_enabled else None,
            client=second_llm),
        "profile": profile_config_version(max_chars=cfg.dream_digest_max_chars,
            max_items=cfg.profile_max_items_per_session, redact_values=cfg.redact_secrets, client=second_llm),
        "facts": facts_config_version(cfg, client=second_llm),
    }
    approved = {}
    for domain in ("digest", "profile", "facts"):
        generation = state[domain + "_cursor_prompt_version"]
        assert isinstance(generation, str), domain
        config = generation.rsplit("|walk=", 1)[0] if domain != "facts" else generation
        assert config == expected_configs[domain]
        approved[domain] = hashlib.sha256(expected_configs[domain].encode()).hexdigest()
    try:
        yield baseline, after, target, approved, prior, state
    finally:
        baseline.close()
        second.close()


def test_same_numeric_cursor_new_approved_epoch_has_actual_stage_progress(epoch_store):
    before, after, target, approved, prior, state = epoch_store
    for domain in ("digest", "profile"):
        assert prior[domain + "_cursor_prompt_version"] != state[domain + "_cursor_prompt_version"]
        for suffix in ("_cursor_message_id", "_cursor_partial_message_id", "_cursor_offset"):
            assert prior[domain + suffix] == state[domain + suffix]
    changes = after.total_changes
    result = checker.bounded_progress_evidence(before, after, target, approved)
    assert result["valid"] is True, result
    assert result["new_epoch_progress_domains"] == 2
    assert result["progressed_domains"] == 2
    assert after.total_changes == changes


@pytest.mark.parametrize("mutation", ["stale_config", "forged_epoch", "no_advancement", "source_hash", "missing_stage", "profile_source"])
def test_forged_stale_or_invalid_new_stage_fails(epoch_store, mutation):
    before, after, target, approved, prior, state = epoch_store
    session = state["id"]
    if mutation == "stale_config":
        approved = dict(approved, digest="0" * 64)
    elif mutation == "forged_epoch":
        # Relabeling the old active epoch with a fresh UUID under the SAME
        # approved config cannot prove the config-change reset.
        for domain in ("digest", "profile"):
            config = state[domain + "_cursor_prompt_version"].rsplit("|walk=", 1)[0]
            before.execute("UPDATE sessions SET " + domain + "_cursor_prompt_version=? WHERE id=?", (config + "|walk=" + "a" * 32, session))
    elif mutation == "no_advancement":
        for domain in ("digest", "profile"):
            after.execute("UPDATE " + domain + "_staging SET cursor_before_message_id=cursor_after_message_id,cursor_before_partial_message_id=cursor_after_partial_message_id,cursor_before_offset=cursor_after_offset")
    elif mutation == "source_hash":
        after.execute("UPDATE digest_staging SET source_sha256=?", ("0" * 64,))
    elif mutation == "missing_stage":
        after.execute("DELETE FROM digest_staging")
        after.execute("DELETE FROM profile_staging")
    else:
        after.execute("UPDATE profile_staging SET cursor_after_offset=cursor_after_offset+9999")
        after.execute("DELETE FROM digest_staging")
    assert checker.bounded_progress_evidence(before, after, target, approved)["valid"] is False


def test_actual_digest_full_helper_rejects_wrong_source_proof_with_profile_healthy(epoch_store):
    from hymem.dreaming.digest import load_completed_digest_slices
    before, after, target, approved, prior, state = epoch_store
    after.execute("UPDATE digest_staging SET source_sha256=?", ("0" * 64,))
    assert checker.profile_stage_chain_valid(after, state["id"], state) is True
    with pytest.raises(RuntimeError, match="source proof changed"):
        load_completed_digest_slices(after, state["id"], state["digest_cursor_prompt_version"], require_complete=False)
    assert checker.bounded_progress_evidence(before, after, target, approved)["valid"] is False
