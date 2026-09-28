"""Offline S1 controls: instructions and evidence plumbing, not LLM recall.

Scripted outputs deliberately cannot prove that a model obeys the allocation
policy. That claim requires separately audited live responses; these tests
prevent the policy or its source/boundary/cap protections regressing silently.
"""
from contextlib import closing
from dataclasses import asdict, replace
import json

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import digest, summary
from hymem.dreaming.lossless import covered_messages_after, materialize_message_coverage
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction import prompts
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.producer import canonical_module_sha256
from tests.test_digest_length_recovery import ScriptedDigestLLM
from tests.test_digest_summary_contract import _config, _extract


_SOURCE = (
    "  Last month we hiked through Pine Forest, then visited Cedar Lake; "
    "we missed the observatory. I prefer quiet trails.\n"
)
_PRIOR = (
    "Earlier topics: botanical drawing; garden recommendations including "
    "Maple Gardens, Hazel Gardens, Alder Gardens and Willow Gardens.  \n"
)
_SUMMARY = (
    "Hiked Pine Forest then visited Cedar Lake last month, missed the "
    "observatory and prefers quiet trails; earlier botanical drawing and "
    "garden recommendations remain topics."
)


def _seed_personal_source(hy):
    message_id = hy.log_message(
        "bounded-summary", "user", _SOURCE, created_at="2020-01-01",
    )
    hy.close_session("bounded-summary")
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "bounded-summary")
    return message_id


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("repair", [False, True], ids=["primary", "compaction"])
def test_emitted_event_first_policy_keeps_exact_source_and_prior(cfg, granular, repair):
    replies = [json.dumps({
        "episodes": [], "summary": "overlong " * 80 if repair else _SUMMARY,
        "procedures": [],
    })]
    if repair:
        replies.append(json.dumps({"summary": _SUMMARY}))
    llm = ScriptedDigestLLM(replies)
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed_personal_source(hy)
        before = [asdict(row) for row in covered_messages_after(hy.conn, "bounded-summary", None)]
        result = _extract(hy, llm, granular=granular, prior_summary=_PRIOR)
        assert result is not None and not result.parse_failed
        assert result.summary == _SUMMARY and result.covered_message_id == last_id
        assert result.caught_up and result.source_sha256 is not None
        assert len(llm.digest_requests) == (2 if repair else 1)
        for request in llm.digest_requests:
            assert prompts.SESSION_DIGEST_SUMMARY_ALLOCATION in request.system
            assert "first capture newly stated durable personal experiences" in request.system
            assert "then retain distinct earlier topics" in request.system
            assert "before dropping a newly stated durable experience" in request.system
            assert "do not turn a missed activity into a completed one" in request.system
            assert _PRIOR in request.user and _SOURCE in request.user
        if repair:
            primary, correction = llm.digest_requests
            assert replace(correction, system=primary.system) == primary
            assert "overlong " * 80 not in correction.system + correction.user
            assert "do not extract episodes or procedures" in correction.system
        assert [asdict(row) for row in covered_messages_after(hy.conn, "bounded-summary", None)] == before
        # Extraction still does not publish a scripted semantic claim or cursor.
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert tuple(hy.conn.execute(
            "SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='bounded-summary'",
        ).fetchone()) == (None, None)


def test_modes_share_summary_and_procedure_contracts_without_event_exclusion():
    blob, granular = prompts.SESSION_DIGEST_SYSTEM, prompts.SESSION_DIGEST_GRANULAR_SYSTEM
    for system in (blob, granular):
        assert "episode material even when no task was solved or decision changed" in system
        assert "Retain the stated event or preference itself" in system
        assert "Distinguish completed actions from intentions" in system
        assert "add the new concrete outcome" not in system
        assert "Travel directions, activity preparation, lifestyle advice" in system
        assert "not technical procedures" in system
    summary_contracts = [text.split('"summary": a single string', 1)[1].split('"procedures":', 1)[0]
                         for text in (blob, granular)]
    assert summary_contracts[0] == summary_contracts[1]
    assert "ONE EPISODE PER DECISION, CHANGE OR OUTCOME" not in granular
    assert "ONE EPISODE PER DISTINCT STATED EVENT, PREFERENCE, DECISION, CHANGE OR OUTCOME" in granular
    assert "Keep connected details of one event together" in granular
    assert "Cite the narrowest set that supports it" in granular
    assert "CARRYING THE CONCRETE VALUES" in granular
    assert "Pinned pandas to 2.1.4" in granular
    assert "NO QUOTA" in granular


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("citation", ["valid", "unseen", "previous-only"])
def test_personal_event_eligibility_never_weakens_citation_validation(cfg, granular, citation):
    llm = ScriptedDigestLLM()
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        previous_id = hy.log_message(
            "bounded-summary", "assistant", "Earlier garden recommendations.",
            created_at="2020-01-01",
        )
        _seed_personal_source(hy)
        previous, current = covered_messages_after(hy.conn, "bounded-summary", None)
        chunk_id = current.chunk_id
        cited_id = {
            "valid": chunk_id, "unseen": "msgcov_unseen",
            "previous-only": previous.chunk_id,
        }[citation]
        llm.outputs = [json.dumps({
            "episodes": [{
                "title": "Forest hike and lake visit",
                "summary": "Hiked Pine Forest then visited Cedar Lake last month; missed the observatory.",
                "outcome": "informational", "key_entities": ["Pine Forest", "Cedar Lake"],
                "chunk_ids": [cited_id],
            }],
            "summary": _SUMMARY, "procedures": [],
        })]
        result = digest.extract_session_digest(
            hy.conn, "bounded-summary", llm, max_tokens=2048, max_chars=10000,
            since_message_id=previous_id, prior_summary=_PRIOR,
            granular=granular, max_episodes=8 if granular else None,
        )
        assert result is not None and len(llm.digest_requests) == 1
        request = llm.digest_requests[0]
        assert f"previous message context message {previous_id}" in request.user
        assert f"[chunk {previous.chunk_id}]" not in request.user
        assert f"[chunk {current.chunk_id}]" in request.user
        if citation == "valid":
            assert not result.parse_failed and len(result.episodes.items) == 1
            assert result.episodes.items[0]["chunk_ids"] == [chunk_id]
        else:
            assert result.parse_failed and result.failure_reason == "episode_validation_failure"
            assert result.episodes.items == [] and result.source_sha256 is None
            assert result.covered_message_id is None and not result.caught_up


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
def test_event_first_policy_cannot_publish_over_cap_repair_or_spend_third_call(cfg, granular):
    long_summary = "explicit personal experience " * 25
    llm = ScriptedDigestLLM([
        json.dumps({"episodes": [], "summary": long_summary, "procedures": []}),
        json.dumps({"summary": long_summary}),
        json.dumps({"summary": _SUMMARY}),
    ])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        _seed_personal_source(hy)
        result = _extract(hy, llm, granular=granular, prior_summary=_PRIOR)
        assert result is not None and result.parse_failed
        assert result.failure_reason == "summary_output_cap"
        assert result.summary is None and result.source_sha256 is None
        assert result.covered_message_id is None and not result.caught_up
        assert len(llm.digest_requests) == 2 and len(llm.outputs) == 1


@pytest.mark.parametrize("field", [
    "SESSION_DIGEST_SYSTEM", "SESSION_DIGEST_GRANULAR_SYSTEM",
    "_DIGEST_SUMMARY_RECOVERY_TEMPLATE",
])
def test_event_allocation_changes_invalidate_digest_only(monkeypatch, field):
    client = StubLLMClient()
    tiers = ("digest", "facts", "profile")
    before = {tier: semantic_generation_suffix(tier, client) for tier in tiers}
    phase1 = extraction_contract_identity("v20")
    standalone = canonical_module_sha256(summary)
    prompt = getattr(digest, field)
    changed = prompt.replace(prompts.SESSION_DIGEST_SUMMARY_ALLOCATION, "")
    assert changed != prompt
    monkeypatch.setattr(digest, field, changed)
    after = {tier: semantic_generation_suffix(tier, client) for tier in tiers}
    assert after["digest"] != before["digest"]
    assert after["facts"] == before["facts"] and after["profile"] == before["profile"]
    assert extraction_contract_identity("v20") == phase1
    assert canonical_module_sha256(summary) == standalone


def test_event_eligibility_does_not_enable_experimental_granularity(cfg):
    assert cfg.episode_granularity_enabled is False
