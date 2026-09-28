"""S2 instruction/evidence controls, not tests of real-model semantic recall.

Scripted responses can prove authority, request and recovery mechanics; they
cannot prove that compression retains categories or faithfully attributes them.
"""
from contextlib import closing
from dataclasses import asdict, replace
import hashlib
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
from tests.test_digest_summary_contract import _config, _extract, _seed


_PRIOR = (
    "  A seller recommended field guides for bird identification and tide "
    "charts for trip planning; audio lessons were requested for language study.\n"
)
_SOURCE = (
    "  I finished the first language lesson yesterday; I have not bought "
    "the field guide. 🧭\n"
)
_SUMMARY = (
    "Finished the first language lesson yesterday without buying the field "
    "guide; earlier recommendations covered bird-identification guides and "
    "trip-planning tide charts, while language audio lessons were requested."
)


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("repair", [False, True], ids=["primary", "compaction"])
def test_category_relations_reach_actual_requests_without_changing_evidence(
    cfg, granular, repair,
):
    replies = [json.dumps({
        "episodes": [], "summary": "rejected wording " * 40 if repair else _SUMMARY,
        "procedures": [],
    })]
    if repair:
        replies.append(json.dumps({"summary": _SUMMARY}))
    llm = ScriptedDigestLLM(replies)
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy, _SOURCE)
        before = [asdict(row) for row in covered_messages_after(hy.conn, "bounded-summary", None)]
        result = _extract(hy, llm, granular=granular, prior_summary=_PRIOR)
        assert result is not None and not result.parse_failed
        assert result.summary == _SUMMARY and result.covered_message_id == last_id
        assert result.caught_up and result.source_sha256 is not None
        assert len(llm.digest_requests) == (2 if repair else 1)
        for request in llm.digest_requests:
            assert request.system.count(prompts.SESSION_DIGEST_CATEGORY_RELATIONS) == 1
            assert prompts.SESSION_DIGEST_SUMMARY_ALLOCATION in request.system
            assert "retain who recommended, requested, preferred, owned or did what" in request.system
            assert "Omit incidental example names before category labels" in request.system
            assert "unlike categories into one similarity or recommendation relation" in request.system
            assert "Advice or a question does not establish the recipient's" in request.system
            assert "prior-only topics and their stated relations may remain as derived continuity" in request.system
            assert "not as newly established events or evidence" in request.system
            assert _PRIOR in request.user and _SOURCE in request.user
            assert request.response_format == "json" and request.max_tokens == 2048
            assert request.temperature == 0.0
        if repair:
            primary, correction = llm.digest_requests
            assert replace(correction, system=primary.system) == primary
            assert "rejected wording " * 40 not in correction.system + correction.user
            assert "do not extract episodes or procedures" in correction.system
            assert "Only the exact visible new material may establish new claims" in correction.system
        assert [asdict(row) for row in covered_messages_after(hy.conn, "bounded-summary", None)] == before
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert tuple(hy.conn.execute(
            "SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='bounded-summary'",
        ).fetchone()) == (None, None)


def test_primary_modes_keep_common_contracts_and_scope_new_item_authority():
    blob, granular = prompts.SESSION_DIGEST_SYSTEM, prompts.SESSION_DIGEST_GRANULAR_SYSTEM
    for system in (blob, granular):
        assert "prior automatic summary is derived continuity context" in system
        assert "cannot by itself authorize a new episode or procedure" in system
        assert "Cite only exact visible new material" in system
        assert "Every emitted episode and procedure must cite at least one chunk from the new material" in system
    assert "THE ONLY RULE THAT MATTERS" not in granular
    assert "claim in an episode or procedure must come from the exact visible source" in granular
    assert "Each new item must be supported by new material and cite its supporting new chunk ids" in granular
    assert "A prior automatic summary is not new-item evidence" in granular
    assert "does not forbid retaining prior-only topics in the rolling summary" in granular
    assert "Do not infer unstated details, invent an unseen suffix or make a detail plausible" in granular
    assert "Never record an outcome that was not reached" in granular
    contracts = [text.split('"summary": a single string', 1)[1].split('"procedures":', 1)[0]
                 for text in (blob, granular)]
    assert contracts[0] == contracts[1]
    procedures = [text.split('"procedures": a JSON array of procedures.', 1)[1]
                  .split('Use empty array [] when no explicit technical procedure is present.', 1)[0]
                  for text in (blob, granular)]
    assert procedures[0] == procedures[1]


@pytest.mark.parametrize("repair", [False, True], ids=["primary", "compaction"])
def test_resumed_granular_request_can_interpret_only_exact_visible_boundary(cfg, repair):
    """A scripted split-name item tests consistent permission, not LLM recall."""
    llm = ScriptedDigestLLM()
    with closing(HyMem(_config(cfg, True), llm=llm)) as hy:
        old_id = hy.log_message(
            "bounded-summary", "assistant", "Unrelated earlier context about a weather station.",
            created_at="2020-01-01",
        )
        source = (
            "Unrelated earlier sentence about a tide gauge. " + "padding " * 10
            + "Yesterday I visited Mount Rainier and returned before dusk."
        )
        current_id = hy.log_message("bounded-summary", "user", source, created_at="2020-01-01")
        hy.close_session("bounded-summary")
        with db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "bounded-summary")
        old, current = covered_messages_after(hy.conn, "bounded-summary", None)
        split = source.index("Rainier")
        first_budget = (
            len(digest._render_digest_leading_context(old))
            + len(digest._DIGEST_SEPARATOR)
            + len(digest._render_message_part(current, 0, split))
        )
        llm.outputs = [json.dumps({"episodes": [], "summary": _PRIOR, "procedures": []})]
        first = digest.extract_session_digest(
            hy.conn, "bounded-summary", llm, max_tokens=2048, max_chars=first_budget,
            since_message_id=old_id, prior_summary=_PRIOR, granular=True, max_episodes=8,
        )
        assert first is not None and not first.parse_failed and not first.caught_up
        assert (first.covered_message_id, first.partial_message_id, first.next_message_offset) == (
            old_id, current_id, split,
        )
        episode = {
            "title": "Mountain visit", "summary": "Visited Mount Rainier yesterday and returned before dusk.",
            "outcome": "informational", "key_entities": ["Mount Rainier"], "chunk_ids": [current.chunk_id],
        }
        final_summary = "Visited Mount Rainier yesterday; earlier field-guide, tide-chart and audio-lesson topics remain."
        llm.outputs = [json.dumps({
            "episodes": [episode], "summary": "rejected wording " * 40 if repair else final_summary,
            "procedures": [],
        })]
        if repair:
            llm.outputs.append(json.dumps({"summary": final_summary}))
        result = digest.extract_session_digest(
            hy.conn, "bounded-summary", llm, max_tokens=2048, max_chars=10000,
            since_message_id=first.covered_message_id, partial_message_id=first.partial_message_id,
            since_message_offset=first.next_message_offset, prior_summary=first.summary,
            granular=True, max_episodes=8,
        )
        assert result is not None and not result.parse_failed and result.caught_up
        assert result.covered_message_id == current_id and result.summary == final_summary
        assert result.episodes.items == [episode]
        assert len(llm.digest_requests) == (3 if repair else 2)
        resumed = llm.digest_requests[1]
        assert "Bounded previous context may only interpret or complete an explicit phrase" in resumed.system
        assert "continues into the visible new material" in resumed.system
        assert "it must never independently authorize a claim" in resumed.system
        expected_source = digest._render_message_part(current, split, len(source))
        assert resumed.user == prompts.SESSION_DIGEST_GRANULAR_USER_TEMPLATE.format(
            prior_summary=first.summary, text=expected_source,
        )
        assert source[split - 48:split].endswith("Yesterday I visited Mount ")
        assert source[split:] == "Rainier and returned before dusk."
        assert f"range={split - 48}:{split}/{len(source)}" in resumed.user
        assert f"chars={split}:{len(source)}/{len(source)}" in resumed.user
        assert "Unrelated earlier" not in resumed.user
        assert old.chunk_id not in resumed.user and old.content not in resumed.user
        if repair:
            correction = llm.digest_requests[2]
            assert replace(correction, system=resumed.system) == resumed
            assert "rejected wording " * 40 not in correction.system + correction.user
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("kind", ["episode", "procedure"])
def test_prior_continuity_permission_does_not_authorize_previous_only_items(cfg, granular, kind):
    llm = ScriptedDigestLLM()
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        old_id = hy.log_message(
            "bounded-summary", "assistant", "To test the index, run indexer build, then indexer verify.",
            created_at="2020-01-01",
        )
        hy.log_message("bounded-summary", "user", _SOURCE, created_at="2020-01-01")
        hy.close_session("bounded-summary")
        with db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "bounded-summary")
        old, current = covered_messages_after(hy.conn, "bounded-summary", None)
        payload = {"episodes": [], "summary": _SUMMARY, "procedures": []}
        if kind == "episode":
            payload["episodes"] = [{
                "title": "Index test advice", "summary": "An index testing procedure was recommended.",
                "outcome": "informational", "key_entities": ["indexer"], "chunk_ids": [old.chunk_id],
            }]
        else:
            payload["procedures"] = [{
                "name": "Test index", "description": "Build and verify the index.",
                "steps": [
                    {"order": 1, "action": "Run indexer build", "tool": "indexer"},
                    {"order": 2, "action": "Run indexer verify", "tool": "indexer"},
                ],
                "triggers": ["test index"], "entities_involved": ["indexer"], "chunk_ids": [old.chunk_id],
            }]
        llm.outputs = [json.dumps(payload)]
        result = digest.extract_session_digest(
            hy.conn, "bounded-summary", llm, max_tokens=2048, max_chars=10000,
            since_message_id=old_id, prior_summary=_PRIOR, granular=granular,
            max_episodes=8 if granular else None,
        )
        assert result is not None and result.parse_failed
        assert result.failure_reason == f"{kind}_validation_failure"
        assert result.source_sha256 is result.covered_message_id is None
        assert not result.caught_up and len(llm.digest_requests) == 1
        request = llm.digest_requests[0]
        assert f"previous message context message {old_id}" in request.user
        assert f"[chunk {old.chunk_id}]" not in request.user
        assert f"[chunk {current.chunk_id}]" in request.user


@pytest.mark.parametrize("field", [
    "SESSION_DIGEST_SYSTEM", "SESSION_DIGEST_GRANULAR_SYSTEM",
    "_DIGEST_SUMMARY_RECOVERY_TEMPLATE",
])
def test_relation_guidance_invalidates_digest_generation_only(monkeypatch, field):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    standalone = canonical_module_sha256(summary)
    prompt = getattr(digest, field)
    changed = prompt.replace(prompts.SESSION_DIGEST_CATEGORY_RELATIONS, "")
    assert changed != prompt
    monkeypatch.setattr(digest, field, changed)
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert after["digest"] != before["digest"]
    assert after["facts"] == before["facts"] and after["profile"] == before["profile"]
    assert extraction_contract_identity("v20") == phase1
    assert canonical_module_sha256(summary) == standalone


def test_relation_guidance_does_not_change_source_templates_or_enable_granularity(cfg):
    expected = {
        "SESSION_DIGEST_USER_TEMPLATE": "0e2f7bf0717e2dc962d24ae3c02c08fe7b011c67b20b28e3094daf356dde14af",
        "SESSION_DIGEST_GRANULAR_USER_TEMPLATE": "982ba09abda60d3b50f55ea7ed3bc0bd60ddff8da25ce0b4724e607782ebd905",
    }
    for name, sha256 in expected.items():
        assert hashlib.sha256(getattr(prompts, name).encode()).hexdigest() == sha256
    assert cfg.episode_granularity_enabled is False
    assert prompts.SESSION_SUMMARY_MAX_CHARS == 500
