"""Normal-path contract controls; scripted replies are not paid fidelity evidence."""
from __future__ import annotations

import json

import pytest

from hymem import HyMem
from hymem.dreaming import digest, summary_policy, summary_recovery
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.dreaming.summary_state import classify_summary_state
from hymem.extraction import prompts
from hymem.extraction.llm import LLMRequest, StubLLMClient
from tests.test_digest_bounded_summary_repair import SequenceLLM, quiet
from tests.test_digest_publication import PublicationLLM, _finish
from tests.test_lossless_digest import _quiet_cfg


OVERVIEW = ("Mira's proposed September launch was not approved; deployment is delayed "
            "until audit closure. Iris verified two absent receipts; five remain unresolved.")
MATERIAL = ("Mira proposed a September launch, but nobody approved it. Deployment was "
            "delayed until the audit closes. Iris verified two absent receipts; five remain "
            "unresolved. The estimate is perhaps 18 days, not a commitment. "
            "Forty peripheral entries list interface colors, room names and calendar details. ")


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("repair", [False, True])
def test_dense_normal_overview_preserves_detailed_items_and_source(cfg, granular, repair):
    hy = HyMem(quiet(cfg), llm=StubLLMClient(default="[]"))
    try:
        sid = "dense-overview"
        raw_source = MATERIAL * 8
        message = hy.log_message(sid, "user", raw_source)
        hy.close_session(sid)
        materialize_message_coverage(hy.conn, sid)
        from hymem.dreaming.lossless import coverage_chunk_id
        chunk = coverage_chunk_id(sid, message)
        # Detailed records may retain facts omitted by the presentation overview.
        items = {"episodes": [{"title": "Tentative estimate", "summary": "The estimate is perhaps 18 days, not a commitment.",
                  "outcome": "informational", "key_entities": ["audit"], "chunk_ids": [chunk]}],
                 "procedures": []}
        primary = {**items, "summary": "🧭" * 501 if repair else OVERVIEW}
        llm = SequenceLLM(primary, {"summary": OVERVIEW})
        result = digest.extract_session_digest(hy.conn, sid, llm, max_chars=20000, max_tokens=3072,
                    granular=granular, max_episodes=8, separate_summary=True)
        assert not result.parse_failed and result.summary_failure_reason is None
        assert result.summary == OVERVIEW and "18" not in result.summary
        assert result.episodes.items == items["episodes"]
        assert result.covered_message_id == message and result.caught_up
        assert len(llm.calls) == (2 if repair else 1)
        assert all(summary_policy.SUMMARY_OVERVIEW_POLICY in call.system for call in llm.calls)
        assert "18 days" in llm.calls[0].user and "interface colors" in llm.calls[0].user
        if repair:
            assert json.loads(llm.calls[1].user) == {"original_generation_input": llm.calls[0].user}
            assert "🧭" not in llm.calls[1].user
        assert hy.conn.execute("SELECT content FROM messages WHERE id=?", (message,)).fetchone()[0] == raw_source
    finally:
        hy.close()


def test_shared_contract_is_only_top_level_and_not_full_claim_retention():
    policy = summary_policy.SUMMARY_OVERVIEW_POLICY
    for system in (prompts.SESSION_DIGEST_SYSTEM, prompts.SESSION_DIGEST_GRANULAR_SYSTEM):
        assert system.count(policy) == 1
        assert "applies ONLY to this top-level summary" in system
        assert '"episodes": a JSON array' in system and '"procedures": a JSON array' in system
        assert "chunk_ids" in system and "Only extract procedures that are EXPLICITLY described" in system
        assert "Preserve earlier accomplishments" not in system
    assert "CARRYING THE CONCRETE VALUES" in prompts.SESSION_DIGEST_GRANULAR_SYSTEM
    assert policy in summary_recovery.SUMMARY_RECOVERY_SYSTEM
    repair = digest._build_digest_summary_repair_request(LLMRequest(system="", user="source"), "x" * 501)
    assert policy in repair.system
    for phrase in ("not a complete inventory", "at most two consequential propositions",
                   "actor or speaker, polarity, uncertainty, qualification",
                   "keep those steps together or omit", "never cut a claim or sentence",
                   "not evidence that omitted earlier claims never occurred"):
        assert phrase in policy
    assert digest.DIGEST_SUMMARY_POLICY_VERSION == summary_policy.SUMMARY_OVERVIEW_VERSION


@pytest.mark.parametrize("name", ["SUMMARY_OVERVIEW_POLICY", "SUMMARY_OVERVIEW_VERSION"])
def test_shared_policy_binds_both_identities(monkeypatch, name):
    llm = StubLLMClient(default="[]")
    before_normal = semantic_generation_suffix("digest", llm)
    before_recovery = summary_recovery._config(llm, 8000, 3072, 3)
    monkeypatch.setattr(summary_policy, name, getattr(summary_policy, name) + " revised")
    assert semantic_generation_suffix("digest", llm) != before_normal
    assert summary_recovery._config(llm, 8000, 3072, 3) != before_recovery


def test_recovery_dispatch_only_revision_does_not_invalidate_item_generation(monkeypatch):
    llm = StubLLMClient(default="[]")
    before_normal = semantic_generation_suffix("digest", llm)
    before_recovery = summary_recovery._config(llm, 8000, 3072, 3)
    original = summary_recovery.run_summary_recovery
    def revised(*args, **kwargs):
        return original(*args, **kwargs)
    monkeypatch.setattr(summary_recovery, "run_summary_recovery", revised)
    assert semantic_generation_suffix("digest", llm) == before_normal
    assert summary_recovery._config(llm, 8000, 3072, 3) != before_recovery


class OverviewPublicationLLM(PublicationLLM):
    def complete(self, request):
        raw = super().complete(request)
        if request.system.startswith(("You analyze one conversation session", "You re-read one conversation session")):
            value = json.loads(raw)
            value["summary"] = OVERVIEW
            return json.dumps(value)
        return raw


def test_cumulative_normal_windows_publish_overview_only_at_complete_tail(cfg):
    llm = OverviewPublicationLLM(emit_slice_artifacts=True)
    hy = HyMem(_quiet_cfg(cfg, dream_digest_max_chars=400), llm=llm)
    try:
        raw_source = "alpha " + MATERIAL * 4
        message = hy.log_message("x", "user", raw_source)
        hy.close_session("x")
        first = hy.dream()
        assert first.budget_exhausted
        row = hy.conn.execute("SELECT * FROM sessions WHERE id='x'").fetchone()
        assert row["digest_cursor_partial_message_id"] is not None
        assert row["auto_summary"] is row["auto_summary_generation"] is None
        assert row["digest_published_message_id"] is None
        staged = digest.load_digest_staged_summary_state(hy.conn, "x", row["digest_cursor_prompt_version"],
                    (row["digest_cursor_message_id"], row["digest_cursor_partial_message_id"], row["digest_cursor_offset"]))
        assert staged == (OVERVIEW, None)
        _finish(hy)
        row = hy.conn.execute("SELECT * FROM sessions WHERE id='x'").fetchone()
        assert row["auto_summary"] == OVERVIEW
        assert row["auto_summary_message_id"] == row["digest_published_message_id"] == message
        assert row["auto_summary_generation"] == row["digest_published_generation"]
        assert classify_summary_state(hy.conn, "x")["summary_healthy"]
        assert len(llm.successful_digest_calls) > 2
        assert all(summary_policy.SUMMARY_OVERVIEW_POLICY in call.system for call in llm.successful_digest_calls)
        assert any(OVERVIEW in call.user for call in llm.successful_digest_calls[1:])
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert hy.conn.execute("SELECT content FROM messages WHERE id=?", (message,)).fetchone()[0] == raw_source
    finally:
        hy.close()
