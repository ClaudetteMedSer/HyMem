"""S4 sentence-policy wiring, not real-model adherence or a sentence detector.

Scripted replies verify that the same format policy reaches primary and repair
without adding a punctuation heuristic, changing evidence or rewriting items.
"""
from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import json

import pytest

from hymem import HyMem
from hymem.dreaming import digest, summary
from hymem.dreaming.lossless import covered_messages_after
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction import prompts
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.producer import canonical_module_sha256
from tests.test_digest_length_recovery import ScriptedDigestLLM
from tests.test_digest_summary_contract import _config, _extract, _seed


_PRIOR = "  Earlier recommendations covered field guides and audio lessons.\n"
_SOURCE = (
    "  Dr. Imani may use v2.1.4 if caching is enabled; offline use is not verified. "
    "To check the cache, first run cache inspect; then run cache verify. 🧭\n"
)
_SUMMARY = (
    "Dr. Imani may use v2.1.4 if caching is enabled, though offline use is not "
    "verified; cache checking uses inspect then verify, while earlier "
    "recommendations covered field guides and audio lessons."
)


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("repair", [False, True], ids=["primary", "compaction"])
def test_sentence_policy_reaches_requests_without_rewriting_items_or_source(
    cfg, granular, repair,
):
    llm = ScriptedDigestLLM()
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy, _SOURCE)
        covered = covered_messages_after(hy.conn, "bounded-summary", None)
        before = [asdict(row) for row in covered]
        chunk_id = covered[0].chunk_id
        episode = {
            "title": "Conditional cache use",
            "summary": "Dr. Imani may use v2.1.4 if caching is enabled. Offline use is not verified.",
            "outcome": "informational", "key_entities": ["Dr. Imani", "v2.1.4"],
            "chunk_ids": [chunk_id],
        }
        procedure = {
            "name": "Check cache", "description": "Inspect and verify the cache.",
            "steps": [
                {"order": 1, "action": "Run cache inspect", "tool": "cache"},
                {"order": 2, "action": "Run cache verify", "tool": "cache"},
            ],
            "triggers": ["check cache"], "entities_involved": ["cache"],
            "chunk_ids": [chunk_id],
        }
        rejected = "REJECTED_SUMMARY_NOT_EVIDENCE " * 30
        llm.outputs = [json.dumps({
            "episodes": [episode], "procedures": [procedure],
            "summary": rejected if repair else _SUMMARY,
        })]
        if repair:
            llm.outputs.append(json.dumps({"summary": _SUMMARY}))
        result = _extract(hy, llm, granular=granular, prior_summary=_PRIOR)
        assert result is not None and not result.parse_failed and result.caught_up
        assert result.summary == _SUMMARY and result.covered_message_id == last_id
        assert result.source_sha256 is not None
        # A two-sentence episode and technical procedure remain exact even
        # when the only repaired field is the rolling summary.
        assert result.episodes.items == [episode]
        assert result.procedures.items == [{
            key: value for key, value in procedure.items() if key != "chunk_ids"
        }]
        assert result.episode_input_items == result.procedure_input_items == 1
        assert len(llm.digest_requests) == (2 if repair else 1)
        template = (prompts.SESSION_DIGEST_GRANULAR_USER_TEMPLATE if granular
                    else prompts.SESSION_DIGEST_USER_TEMPLATE)
        expected_user = template.format(
            prior_summary=_PRIOR,
            text=digest._render_message_part(covered[0], 0, len(_SOURCE)),
        )
        for request in llm.digest_requests:
            assert request.system.count(prompts.SESSION_DIGEST_SUMMARY_SENTENCE) == 1
            assert "For a nonempty rolling summary, write exactly one sentence" in request.system
            assert "conjunctions or semicolons" in request.system
            assert "preserving category boundaries, qualifiers, negation and scope" in request.system
            assert "Rephrase rather than truncate a sentence" in request.system
            assert "Prefer one sentence" not in request.system
            assert prompts.SESSION_DIGEST_SUMMARY_ALLOCATION in request.system
            assert prompts.SESSION_DIGEST_CATEGORY_RELATIONS in request.system
            assert prompts.SESSION_DIGEST_CLAIM_SCOPE in request.system
            assert request.user == expected_user
            assert _PRIOR in request.user and _SOURCE in request.user
            assert request.response_format == "json" and request.max_tokens == 2048
            assert request.temperature == 0.0
        if repair:
            primary, correction = llm.digest_requests
            assert replace(correction, system=primary.system) == primary
            assert rejected not in correction.system + correction.user
            assert "do not extract episodes or procedures" in correction.system
        assert [asdict(row) for row in covered_messages_after(hy.conn, "bounded-summary", None)] == before
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        assert tuple(hy.conn.execute(
            "SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='bounded-summary'",
        ).fetchone()) == (None, None)
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("text", [
    "Dr. Imani may use v2.1.4; e.g., caching can be enabled.",
    "Caching is enabled. Offline use is not verified.",
    "",
], ids=["abbreviations-and-version", "no-runtime-sentence-heuristic", "existing-empty-contract"])
def test_sentence_policy_does_not_add_punctuation_validation(granular, text):
    # The prompt contract is stronger than structural validation. Keeping
    # this distinction avoids new false failures from punctuation guesses.
    result = digest._validate_digest_response(
        None, {"episodes": [], "summary": text, "procedures": []},
        "bounded-summary", [], granular=granular, max_episodes=8 if granular else None,
    )
    assert not result.parse_failed and result.summary == (text or None)
    if text:
        assert digest._validate_digest_summary_repair(json.dumps({"summary": text})) == (text, None)


@pytest.mark.parametrize("field", [
    "SESSION_DIGEST_SYSTEM", "SESSION_DIGEST_GRANULAR_SYSTEM",
    "_DIGEST_SUMMARY_RECOVERY_TEMPLATE",
])
def test_sentence_contract_changes_digest_generation_only(monkeypatch, field):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    standalone = canonical_module_sha256(summary)
    standalone_prompts = (prompts.SESSION_SUMMARY_SYSTEM, prompts.SESSION_SUMMARY_USER_TEMPLATE)
    prompt = getattr(digest, field)
    changed = prompt.replace(prompts.SESSION_DIGEST_SUMMARY_SENTENCE, "")
    assert changed != prompt
    monkeypatch.setattr(digest, field, changed)
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert after["digest"] != before["digest"]
    assert after["facts"] == before["facts"] and after["profile"] == before["profile"]
    assert extraction_contract_identity("v20") == phase1
    assert canonical_module_sha256(summary) == standalone
    assert (prompts.SESSION_SUMMARY_SYSTEM, prompts.SESSION_SUMMARY_USER_TEMPLATE) == standalone_prompts


def test_sentence_guidance_preserves_episode_templates_limits_and_defaults(cfg):
    expected = {
        "SESSION_DIGEST_USER_TEMPLATE": "0e2f7bf0717e2dc962d24ae3c02c08fe7b011c67b20b28e3094daf356dde14af",
        "SESSION_DIGEST_GRANULAR_USER_TEMPLATE": "982ba09abda60d3b50f55ea7ed3bc0bd60ddff8da25ce0b4724e607782ebd905",
    }
    for name, sha256 in expected.items():
        assert hashlib.sha256(getattr(prompts, name).encode()).hexdigest() == sha256
    assert cfg.episode_granularity_enabled is False
    assert prompts.SESSION_SUMMARY_MAX_CHARS == 500
    assert "1-2 sentence narrative" in prompts.SESSION_DIGEST_SYSTEM
    assert "1-2 sentences saying what happened" in prompts.SESSION_DIGEST_GRANULAR_SYSTEM
    assert "Bounded previous context may only interpret or complete an explicit phrase" in prompts.SESSION_DIGEST_GRANULAR_SYSTEM
    assert "A prior automatic summary is not new-item evidence" in prompts.SESSION_DIGEST_GRANULAR_SYSTEM
