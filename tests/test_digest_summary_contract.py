"""The digest producer must know the exact summary bound its validator holds."""
from contextlib import closing
from dataclasses import replace
import json

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import digest, summary
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.retention import prune_messages
from hymem.extraction import prompts
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import StubLLMClient
from tests.digest_verification_fixtures import VerificationStubLLM


_BOUND = "The summary string must be at most 500 characters"


def _payload(text):
    return json.dumps({"episodes": [], "summary": text, "procedures": []})


def _config(cfg, granular):
    return replace(
        cfg, aggregation_nodes_enabled=False,
        profile_extraction_enabled=False, facts_extraction_enabled=False,
        episode_granularity_enabled=granular,
    )


def _extract(hy, llm, *, granular, prior_summary=None):
    return digest.extract_session_digest(
        hy.conn, "bounded-summary", llm,
        max_tokens=2048, max_chars=10000,
        prior_summary=prior_summary, granular=granular,
        max_episodes=8 if granular else None,
    )


def _seed(hy, text="Documented the Alpha service rollout."):
    message_id = hy.log_message(
        "bounded-summary", "assistant", text, created_at="2020-01-01",
    )
    hy.close_session("bounded-summary")
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "bounded-summary")
    return message_id


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
def test_actual_digest_request_declares_bound_without_dropping_input(cfg, granular):
    prior = "Earlier Alpha rollout and Beta repair decisions. " * 16
    source = "New Gamma release was documented with its rollback procedure. " * 16
    llm = VerificationStubLLM(default=_payload("Documented the Alpha, Beta and Gamma work."))
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy, source)
        result = _extract(hy, llm, granular=granular, prior_summary=prior)
        assert result is not None and not result.parse_failed
        assert result.covered_message_id == last_id and result.caught_up
        assert len(llm.calls) == 3
        request = llm.calls[0]
        assert _BOUND in request.system
        assert "after trimming leading and trailing whitespace" in request.system
        assert "Unicode code points" in request.system
        assert "JSON escaping does not add characters" in request.system
        assert "covering BOTH the prior automatic summary and the new material" in request.system
        assert "Preserve earlier accomplishments, decisions, problems solved, and topics" in request.system
        assert prior in request.user and source in request.user
        assert request.system == (
            prompts.SESSION_DIGEST_GRANULAR_SYSTEM if granular
            else prompts.SESSION_DIGEST_SYSTEM
        )
        assert request.response_format == "json"
        assert request.temperature == 0.0 and request.max_tokens == 2048


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("length", [499, 500, 501])
@pytest.mark.parametrize("character", ["s", "🧬"], ids=["ascii", "unicode"])
def test_summary_ceiling_is_exact_and_never_silently_truncated(
    cfg, granular, length, character,
):
    text = character * length
    # Outer whitespace and JSON escaping count neither toward the existing
    # stripped-string ceiling nor toward the returned cleaned summary.
    llm = VerificationStubLLM(
        fixtures={"You compact one rolling conversation summary": json.dumps({"summary": text})},
        default=_payload(" \n" + text + "\t "),
    )
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy)
        result = _extract(hy, llm, granular=granular)
        assert result is not None
        if length <= 500:
            assert not result.parse_failed
            assert result.summary == text
            assert result.covered_message_id == last_id and result.caught_up
        else:
            assert result.parse_failed
            assert result.failure_reason == "summary_output_cap"
            assert result.summary is None
            assert result.covered_message_id is None and not result.caught_up
            assert result.episodes.items == [] and result.procedures.items == []


def test_summary_bound_is_shared_by_prompt_validator_and_compatibility_cleaner():
    assert prompts.SESSION_SUMMARY_MAX_CHARS == 500
    assert digest.SESSION_SUMMARY_MAX_CHARS == prompts.SESSION_SUMMARY_MAX_CHARS
    assert summary.SESSION_SUMMARY_MAX_CHARS == prompts.SESSION_SUMMARY_MAX_CHARS
    # Legacy cleaner behavior is unchanged; the digest rejects raw over-cap
    # input rather than publishing the cleaner's truncated result.
    assert summary.clean_summary("s" * 501) == "s" * 500


def test_digest_summary_contract_does_not_invalidate_chunk_extraction(monkeypatch):
    before = extraction_contract_identity("v20")
    for name in ("SESSION_DIGEST_SYSTEM", "SESSION_DIGEST_GRANULAR_SYSTEM"):
        old_prompt = getattr(prompts, name).replace(
            prompts._SESSION_DIGEST_SUMMARY_LIMIT, " ",
        )
        monkeypatch.setattr(prompts, name, old_prompt)
        monkeypatch.setattr(digest, name, old_prompt)
    assert extraction_contract_identity("v20") == before


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
def test_old_unbounded_prompt_generation_replays_retained_source(cfg, monkeypatch, granular):
    client = VerificationStubLLM(fixtures={
        "You analyze one conversation session": _payload("Documented the Alpha service rollout."),
        "You re-read one conversation session": _payload("Documented the Alpha service rollout."),
        "single pass": '{"triples":[],"markers":[],"complete":true}',
    }, default="[]")
    config = _config(cfg, granular)
    prompt_name = (
        "SESSION_DIGEST_GRANULAR_SYSTEM" if granular else "SESSION_DIGEST_SYSTEM"
    )
    current_prompt = getattr(digest, prompt_name)
    assert _BOUND in current_prompt
    # Recreate the prior producer's summary instruction exactly, not a
    # counterfeit database stamp. The loaded prompt itself issues its old
    # generation, which current health must refuse to reuse.
    old_prompt = current_prompt.replace(prompts._SESSION_DIGEST_SUMMARY_LIMIT, " ")
    assert old_prompt != current_prompt and _BOUND not in old_prompt
    with closing(HyMem(config, llm=client)) as hy:
        _seed(hy)
        with monkeypatch.context() as old:
            old.setattr(digest, prompt_name, old_prompt)
            hy.dream()
            old_generation = hy.conn.execute(
                "SELECT digest_published_generation FROM sessions WHERE id='bounded-summary'"
            ).fetchone()[0]
            assert old_generation is not None
            assert hy.dream_status()["pending_digests"] == 0
            with db.transaction(hy.conn):
                assert prune_messages(
                    hy.conn, replace(config, message_retention_days=1),
                ) == 1
        assert hy.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert hy.dream_status()["pending_digests"] == 1
        prior_calls = len(client.calls)
        hy.dream()
        new_generation = hy.conn.execute(
            "SELECT digest_published_generation FROM sessions WHERE id='bounded-summary'"
        ).fetchone()[0]
        assert new_generation and new_generation != old_generation
        new_requests = client.calls[prior_calls:]
        digest_requests = [request for request in new_requests if request.system == current_prompt]
        assert len(digest_requests) == 1
        assert "Documented the Alpha service rollout." in digest_requests[0].user
        health = hy.dream_status()
        assert health["pending_digests"] == health["malformed_digests"] == 0
        settled_calls = len(client.calls)
        hy.dream()
        assert len(client.calls) == settled_calls
