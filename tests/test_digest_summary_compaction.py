"""Offline compaction controls, not evidence of real-model semantic quality."""
from contextlib import closing
from dataclasses import asdict, replace
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
from tests.test_digest_length_recovery import ScriptedDigestLLM, _summary_payload
from tests.test_digest_summary_contract import _config, _extract, _seed


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("fenced_primary", [False, True])
@pytest.mark.parametrize("fenced_repair", [False, True])
def test_summary_only_compaction_keeps_validated_primary_artifacts_exact(
    cfg, granular, fenced_primary, fenced_repair,
):
    primary = {}
    repaired = "Retained Alpha deployment and Beta debugging decisions."

    def first_reply(raw):
        data = json.loads(raw)
        data["summary"] = "FAILED_OUTPUT_NOT_CANONICAL " * 30
        data["episodes"][0]["title"] = "Preserve this exact primary episode"
        data["procedures"][0]["name"] = "Preserve this exact primary procedure"
        primary.update(data)
        rendered = json.dumps(data)
        return f"```json\n{rendered}\n```" if fenced_primary else rendered

    correction = _summary_payload(repaired)
    if fenced_repair:
        correction = f"```json\n{correction}\n```"
    llm = ScriptedDigestLLM([first_reply, correction])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy, "Alpha was deployed, and Beta debugging was documented.")
        result = _extract(hy, llm, granular=granular)
        assert result is not None and not result.parse_failed
        valid_ids = [row.chunk_id for row in covered_messages_after(hy.conn, "bounded-summary", None)]
        expected = digest._validate_digest_response(
            None, {**primary, "summary": repaired}, "bounded-summary", valid_ids,
            granular=granular, max_episodes=8 if granular else None,
        )
        assert not expected.parse_failed
        assert asdict(result.episodes) == asdict(expected.episodes)
        assert asdict(result.procedures) == asdict(expected.procedures)
        assert result.episode_input_items == result.procedure_input_items == 1
        assert result.summary == repaired and result.covered_message_id == last_id
        assert result.source_sha256 is not None and result.caught_up
        assert len(llm.digest_requests) == 2
        first, second = llm.digest_requests
        assert first.user == second.user
        assert "FAILED_OUTPUT_NOT_CANONICAL" not in second.system + second.user
        assert "Preserve this exact primary" not in second.system + second.user
        assert replace(second, system=first.system) == first
        # Extraction itself still makes no staging/publication/cursor writes.
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        row = hy.conn.execute(
            "SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='bounded-summary'",
        ).fetchone()
        assert tuple(row) == (None, None)


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("character", ["s", "🧬", "\u0301"])
@pytest.mark.parametrize("length", [350, 500, 501])
def test_near_full_prior_and_tiny_source_do_not_change_compaction_cap_or_bytes(
    cfg, granular, character, length,
):
    prior = "Coastal photography, wineries, music and podcast discussion. "
    prior += "P" * (472 - len(prior))
    source = "  Homecoming is on Netflix, not YouTube.\n"
    primary = json.dumps({"episodes": [], "summary": prior + source * 2, "procedures": []})
    repaired = " \n" + character * length + "\t "
    llm = ScriptedDigestLLM([primary, _summary_payload(repaired)])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy, source)
        result = _extract(hy, llm, granular=granular, prior_summary=prior)
        first, second = llm.digest_requests
        assert len(llm.digest_requests) == 2
        assert first.user == second.user and prior in second.user and source in second.user
        assert "Aim for 350 characters" in second.system
        assert "hard maximum is 500 Unicode code points" in second.system
        if length <= 500:
            assert not result.parse_failed and result.summary == character * length
            assert result.covered_message_id == last_id and result.caught_up
        else:
            assert result.parse_failed and result.failure_reason == "summary_output_cap"
            assert result.summary is result.covered_message_id is result.source_sha256 is None
            assert not result.caught_up


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
def test_primary_and_compaction_instructions_preserve_semantic_distinctions(cfg, granular):
    """Checks producer guidance exists; scripted text cannot attest LLM recall."""
    llm = ScriptedDigestLLM([
        json.dumps({"episodes": [], "summary": "Long prior summary. " * 30, "procedures": []}),
        _summary_payload("Visited Big Sur then Monterey; missed Santa Ynez; discussed music and documentaries."),
    ])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        _seed(hy, "I visited Big Sur then Monterey and missed Santa Ynez.")
        result = _extract(hy, llm, granular=granular)
        assert result is not None and not result.parse_failed
        first, second = llm.digest_requests
        for request in (first, second):
            assert "personal experiences" in request.system
            assert "recommendation" in request.system
            assert "artists/songs, podcasts and documentaries" in request.system
            assert "derived continuity context" in request.system
            assert "unseen cut-off phrase" in request.system
            assert "correction" in request.system
        assert "Travel directions, activity preparation, lifestyle advice" in first.system
        assert "not technical procedures" in first.system
        assert "Road names and destinations are not tools/commands/CLIs" in first.system
        assert "do not extract episodes or procedures" in second.system
        assert "exactly one key: summary" in second.system
        assert "group related examples" in second.system
        assert "not every named example" in second.system


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
def test_assembled_repair_must_pass_full_validation_before_source_cursor_return(
    cfg, monkeypatch, granular,
):
    seen = []
    validate = digest._validate_digest_response

    def reject_assembled(raw, data, *args, **kwargs):
        seen.append(data)
        if len(seen) == 2:
            return digest._empty(reason="procedure_validation_failure")
        return validate(raw, data, *args, **kwargs)

    def long_primary(raw):
        data = json.loads(raw)
        data["summary"] = "Original overlong wording. " * 25
        return json.dumps(data)

    llm = ScriptedDigestLLM([long_primary, _summary_payload("A valid compact replacement summary.")])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        _seed(hy)
        monkeypatch.setattr(digest, "_validate_digest_response", reject_assembled)
        result = _extract(hy, llm, granular=granular)
        assert len(seen) == 2 and len(llm.digest_requests) == 2
        assert seen[0]["episodes"] == seen[1]["episodes"]
        assert seen[0]["procedures"] == seen[1]["procedures"]
        assert seen[0]["summary"] != seen[1]["summary"]
        assert result.parse_failed and result.failure_reason == "procedure_validation_failure"
        assert result.episodes.items == result.procedures.items == []
        assert result.source_sha256 is result.covered_message_id is result.summary is None
        assert result.next_message_offset == 0 and not result.caught_up


@pytest.mark.parametrize("field", ["policy", "summary_prompt", "primary_prompt", "validator"])
def test_compaction_semantics_change_digest_only_not_standalone_summary(monkeypatch, field):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    standalone = canonical_module_sha256(summary)
    standalone_prompt = (prompts.SESSION_SUMMARY_SYSTEM, prompts.SESSION_SUMMARY_USER_TEMPLATE)
    if field == "policy":
        monkeypatch.setattr(digest, "DIGEST_SUMMARY_RECOVERY_VERSION", "previous-policy")
    elif field == "summary_prompt":
        monkeypatch.setattr(digest, "_DIGEST_SUMMARY_RECOVERY_TEMPLATE", "previous-summary-task")
    elif field == "primary_prompt":
        monkeypatch.setattr(digest, "SESSION_DIGEST_SYSTEM", "previous-primary-task")
    else:
        original = digest._validate_digest_summary_repair

        def changed(raw):
            return original(raw)

        monkeypatch.setattr(digest, "_validate_digest_summary_repair", changed)
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert after["digest"] != before["digest"]
    assert after["facts"] == before["facts"] and after["profile"] == before["profile"]
    assert extraction_contract_identity("v20") == phase1
    assert canonical_module_sha256(summary) == standalone
    assert (prompts.SESSION_SUMMARY_SYSTEM, prompts.SESSION_SUMMARY_USER_TEMPLATE) == standalone_prompt
