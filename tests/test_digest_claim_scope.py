"""S3 scope-policy wiring and preservation, not real-model faithfulness proof.

Scripted outputs exercise the actual extraction/repair path, but cannot prove
that a provider preserves qualifiers or avoids unsupported strengthening.
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


_PRIOR = "  Earlier planning covered field guides and audio lessons.\n"
_PROCEDURE_SOURCE = " To verify the cache, first run cache inspect; then run cache verify."
_CASES = [
    pytest.param(
        "The Cedar service may work if caching is enabled; offline use is unverified. "
        "As of June the preview is available through Cedar, not Birch; other providers "
        "were not checked. Earlier advice that cache refresh always fixes the error "
        "was corrected to a conditional possibility.",
        "As of June, the preview is available through Cedar, not Birch; other providers "
        "are unchecked, and cache refresh may work if caching is enabled, correcting "
        "the earlier guarantee; offline use remains unverified.",
        id="qualified-correction",
    ),
    pytest.param(
        "The June preview is available exclusively through Cedar. All June offline "
        "checks passed. The earlier claim that Birch also hosts it was explicitly "
        "corrected: only Cedar hosts this preview.",
        "The June preview is exclusively available through Cedar, correcting the "
        "earlier Birch claim; all June offline checks passed.",
        id="explicitly-supported-strength",
    ),
]


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("repair", [False, True], ids=["primary", "compaction"])
@pytest.mark.parametrize("claim_source,claim_summary", _CASES)
def test_scope_contract_reaches_requests_and_preserves_source_and_primary_items(
    cfg, granular, repair, claim_source, claim_summary,
):
    """No blacklist or semantic acceptance claim: outputs are scripted inputs."""
    source = "  " + claim_source + _PROCEDURE_SOURCE + " 🧭\n"
    final_summary = claim_summary + " Earlier field-guide and audio-lesson planning remains."
    assert len(final_summary) <= prompts.SESSION_SUMMARY_MAX_CHARS
    llm = ScriptedDigestLLM()
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy, source)
        covered = covered_messages_after(hy.conn, "bounded-summary", None)
        before = [asdict(row) for row in covered]
        assert len(covered) == 1
        chunk_id = covered[0].chunk_id
        episode = {
            "title": "Preview and cache clarification", "summary": claim_summary,
            "outcome": "informational", "key_entities": ["Cedar", "Birch"],
            "chunk_ids": [chunk_id],
        }
        procedure = {
            "name": "Verify cache", "description": "Inspect and verify the cache.",
            "steps": [
                {"order": 1, "action": "Run cache inspect", "tool": "cache"},
                {"order": 2, "action": "Run cache verify", "tool": "cache"},
            ],
            "triggers": ["verify cache"], "entities_involved": ["cache"],
            "chunk_ids": [chunk_id],
        }
        rejected_summary = "REJECTED_SUMMARY_NOT_EVIDENCE " * 30
        llm.outputs = [json.dumps({
            "episodes": [episode], "procedures": [procedure],
            "summary": rejected_summary if repair else final_summary,
        })]
        if repair:
            llm.outputs.append(json.dumps({"summary": final_summary}))
        result = _extract(hy, llm, granular=granular, prior_summary=_PRIOR)
        assert result is not None and not result.parse_failed and result.caught_up
        assert result.summary == final_summary and result.covered_message_id == last_id
        assert result.source_sha256 is not None
        assert result.episodes.items == [episode]
        # Procedure citations are checked at the digest boundary; the existing
        # normalized procedure type retains only its semantic fields.
        assert result.procedures.items == [{
            key: value for key, value in procedure.items() if key != "chunk_ids"
        }]
        assert result.episode_input_items == result.procedure_input_items == 1
        assert len(llm.digest_requests) == (2 if repair else 1)
        template = (prompts.SESSION_DIGEST_GRANULAR_USER_TEMPLATE if granular
                    else prompts.SESSION_DIGEST_USER_TEMPLATE)
        expected_user = template.format(
            prior_summary=_PRIOR,
            text=digest._render_message_part(covered[0], 0, len(source)),
        )
        for request in llm.digest_requests:
            assert request.system.count(prompts.SESSION_DIGEST_CLAIM_SCOPE) == 1
            assert "stated uncertainty, conditions, negation, time and scope" in request.system
            assert "Preserve explicit corrections and their direction" in request.system
            assert "Do not infer broader exclusivity, universality or certainty" in request.system
            assert "Retain stronger wording when explicitly supported" in request.system
            assert "Apply these limits to prior-summary continuity as well as new material" in request.system
            assert prompts.SESSION_DIGEST_SUMMARY_ALLOCATION in request.system
            assert prompts.SESSION_DIGEST_CATEGORY_RELATIONS in request.system
            assert request.user == expected_user
            assert _PRIOR in request.user and source in request.user
            assert request.response_format == "json" and request.max_tokens == 2048
            assert request.temperature == 0.0
        if repair:
            primary, correction = llm.digest_requests
            assert replace(correction, system=primary.system) == primary
            assert rejected_summary not in correction.system + correction.user
            assert "do not extract episodes or procedures" in correction.system
            assert "Only the exact visible new material may establish new claims" in correction.system
        assert [asdict(row) for row in covered_messages_after(hy.conn, "bounded-summary", None)] == before
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        assert tuple(hy.conn.execute(
            "SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='bounded-summary'",
        ).fetchone()) == (None, None)


@pytest.mark.parametrize("field", [
    "SESSION_DIGEST_SYSTEM", "SESSION_DIGEST_GRANULAR_SYSTEM",
    "_DIGEST_SUMMARY_RECOVERY_TEMPLATE",
])
def test_claim_scope_contract_changes_digest_generation_only(monkeypatch, field):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    standalone = canonical_module_sha256(summary)
    standalone_prompts = (prompts.SESSION_SUMMARY_SYSTEM, prompts.SESSION_SUMMARY_USER_TEMPLATE)
    prompt = getattr(digest, field)
    changed = prompt.replace(prompts.SESSION_DIGEST_CLAIM_SCOPE, "")
    assert changed != prompt
    monkeypatch.setattr(digest, field, changed)
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert after["digest"] != before["digest"]
    assert after["facts"] == before["facts"] and after["profile"] == before["profile"]
    assert extraction_contract_identity("v20") == phase1
    assert canonical_module_sha256(summary) == standalone
    assert (prompts.SESSION_SUMMARY_SYSTEM, prompts.SESSION_SUMMARY_USER_TEMPLATE) == standalone_prompts


def test_scope_guidance_does_not_change_templates_limits_or_default_granularity(cfg):
    expected = {
        "SESSION_DIGEST_USER_TEMPLATE": "0e2f7bf0717e2dc962d24ae3c02c08fe7b011c67b20b28e3094daf356dde14af",
        "SESSION_DIGEST_GRANULAR_USER_TEMPLATE": "982ba09abda60d3b50f55ea7ed3bc0bd60ddff8da25ce0b4724e607782ebd905",
    }
    for name, sha256 in expected.items():
        assert hashlib.sha256(getattr(prompts, name).encode()).hexdigest() == sha256
    assert cfg.episode_granularity_enabled is False
    assert prompts.SESSION_SUMMARY_MAX_CHARS == 500
    assert "Bounded previous context may only interpret or complete an explicit phrase" in prompts.SESSION_DIGEST_GRANULAR_SYSTEM
    assert "continues into the visible new material" in prompts.SESSION_DIGEST_GRANULAR_SYSTEM
    assert "A prior automatic summary is not new-item evidence" in prompts.SESSION_DIGEST_GRANULAR_SYSTEM
