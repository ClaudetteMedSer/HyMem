"""Bounded whole-unit verification is not decided by an English cue regex.

These are synthetic inputs and scripted contract replies, not recall tests.
Two complete empty replies can share a semantic miss; the benchmark canary's
separate known-claim oracle remains necessary to detect that provider failure.
"""

from __future__ import annotations

import json

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction import chunk
from hymem.extraction.llm import LLMRequest


def _response(*claims: dict, complete: bool = True) -> str:
    return json.dumps({"triples": list(claims), "markers": [], "complete": complete})


def _claim(object_: str, source_id: int = 17) -> dict:
    return {
        "subject": "user", "predicate": "uses", "object": object_,
        "polarity": 1, "source_message_id": source_id,
    }


def _source(content: str, message_id: int = 17) -> tuple[int, str]:
    return message_id, json.dumps({
        "content": content,
        "source_created_at": "2026-09-01T00:00:00.000Z",
        "source_message_id": message_id,
        "source_peer_id": None,
        "source_record_version": "hymem-claim-source-v2",
        "source_role": "user",
        "source_session_id": "synthetic-terminal-policy",
        "source_workspace_id": None,
    }, sort_keys=True, separators=(",", ":"))


class _SequenceClient:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if not self.responses:
            raise AssertionError("unexpected extra extraction call")
        return self.responses.pop(0)


def _sentence(prefix: str, size: int, *, terminal: str = "?") -> str:
    # Punctuation-free synthetic padding makes exact 191/192/193-character
    # boundaries testable without inventing a sentence split inside a claim.
    padding = ("illustrative detail " * size)[:size - len(prefix) - 2]
    return prefix + " " + padding + terminal


@pytest.mark.parametrize("size", [191, 192, 193, 400])
@pytest.mark.parametrize("source_backed", [False, True])
@pytest.mark.parametrize("prefix", [
    'Does the quoted phrase "I prefer amber" demonstrate grammar in this',
    "Can you explain why the hypothetical oscillator uses a spring in this",
    "If a fictional service depends on a fictional database in this",
    "Is a generic technical definition of a device that uses gearing part of this",
], ids=["quoted-cue", "question", "hypothetical", "generic-technical"])
def test_bounded_unsplittable_empty_pair_is_not_vetoed_by_cue_or_length(
    prefix, source_backed, size,
):
    text = _sentence(prefix, size)
    assert len(text) == size
    assert chunk._EXPLICIT_EXTRACTION_CUE.search(text)
    assert chunk._semantic_split_point(text) is None
    sources = (_source(text),) if source_backed else None
    expected_excerpt = sources[0][1] if sources else text
    assert len(expected_excerpt) <= chunk._MAX_LEAF_INPUT_CHARS
    client = _SequenceClient(_response(), _response())

    result = chunk.extract_chunk(client, text, source_records=sources)

    assert not result.failed
    assert result.failure_reason is None and result.failure_details == ()
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == len(client.calls) == 2
    assert "VERIFICATION PASS" not in client.calls[0].system
    assert "EMPTY VERIFICATION PASS" in client.calls[1].system
    assert client.calls[0].user == client.calls[1].user
    assert client.calls[0].user.split('"""', 2)[1].strip() == expected_excerpt


def test_bounded_empty_pair_at_split_depth_limit_still_verifies_whole_unit(monkeypatch):
    text = _sentence("A hypothetical service uses a database in this", 200)
    text += " " + _sentence("A second hypothetical service uses a cache in this", 200)
    assert chunk._semantic_split_point(text) is not None
    monkeypatch.setattr(chunk, "_MAX_SPLIT_DEPTH", 0)
    client = _SequenceClient(_response(), _response())

    result = chunk.extract_chunk(client, text)

    assert not result.failed
    assert result.completion_calls == 2
    assert client.calls[0].user == client.calls[1].user
    assert client.calls[0].user.split('"""', 2)[1].strip() == text


@pytest.mark.parametrize(("first", "reason", "calls"), [
    (_response(complete=False), "incomplete_response", 2),
    ('{"triples": [', "parse_failure", 2),
    ('{"triples": [BROKEN', "parse_failure", 1),
    (_response({"subject": "missing-required-fields"}), "item_validation_failure", 2),
])
def test_actual_incomplete_or_invalid_primary_cannot_be_certified_by_empty_retry(
    first, reason, calls,
):
    text = _sentence("Can you explain why a hypothetical oscillator uses a spring", 240)
    client = _SequenceClient(first, _response())

    result = chunk.extract_chunk(client, text)

    assert result.failed and result.failure_reason == reason
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == len(client.calls) == calls
    assert "split:no_admissible_semantic_boundary" in result.failure_details


def test_actual_incompleteness_at_depth_limit_is_not_a_clean_empty_pair(monkeypatch):
    text = _sentence("A hypothetical device uses a spring in this", 200)
    text += " " + _sentence("Another hypothetical device uses gearing in this", 200)
    assert chunk._semantic_split_point(text) is not None
    monkeypatch.setattr(chunk, "_MAX_SPLIT_DEPTH", 0)
    client = _SequenceClient(_response(complete=False), _response())

    result = chunk.extract_chunk(client, text)

    assert result.failed and result.failure_reason == "incomplete_response"
    assert "split:max_depth_reached" in result.failure_details
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == 2


@pytest.mark.parametrize(("verification", "reason"), [
    (_response(complete=False), "incomplete_response"),
    ('{"triples": [', "parse_failure"),
    ('{"triples": [BROKEN', "parse_failure"),
    (_response({"subject": "missing-required-fields"}), "item_validation_failure"),
    ('{"triples": [], "markers": []}', "contract_failure"),
])
def test_empty_primary_still_requires_a_complete_valid_verification(verification, reason):
    text = _sentence("A hypothetical service depends on a database in this", 240)
    client = _SequenceClient(_response(), verification)

    result = chunk.extract_chunk(client, text)

    assert result.failed and result.failure_reason == reason
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == len(client.calls) == 2


def test_bounded_empty_is_not_published_without_verification_budget():
    client = _SequenceClient(_response())
    result = chunk.extract_chunk(
        client, _sentence("A hypothetical device uses a spring in this", 240),
        completion_call_limit=1,
    )

    assert result.failed and result.failure_reason == "resource_limit"
    assert "calls:max_exceeded" in result.failure_details
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == len(client.calls) == 1


def test_long_terminal_claim_recovered_by_empty_check_gets_exact_omission_pass():
    source = _source(_sentence(
        "I use PostgreSQL for storage and Redis for caching in this", 240,
        terminal=".",
    ))
    client = _SequenceClient(_response(), _response(_claim("PostgreSQL")), _response(_claim("Redis")))

    result = chunk.extract_chunk(client, "ignored", source_records=(source,))

    assert not result.failed
    assert {triple.object for triple in result.triples} == {"PostgreSQL", "Redis"}
    assert result.completion_calls == result.provider_attempts == len(client.calls) == 3
    assert "EMPTY VERIFICATION PASS" in client.calls[1].system
    assert "OMISSION VERIFICATION PASS" in client.calls[2].system
    assert [call.user.split('"""', 2)[1].strip() for call in client.calls] == [source[1]] * 3


@pytest.mark.parametrize("omit_budget", [False, True])
def test_recovered_long_terminal_claim_stays_atomic_without_omission_certification(omit_budget):
    source = _source(_sentence(
        "I use PostgreSQL for storage in this", 240, terminal=".",
    ))
    client = _SequenceClient(_response(), _response(_claim("PostgreSQL")), _response(complete=False))

    result = chunk.extract_chunk(
        client, "ignored", source_records=(source,),
        completion_call_limit=2 if omit_budget else 3,
    )

    assert result.failed
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == len(client.calls) == (2 if omit_budget else 3)
    assert any(
        ("resource_limit" if omit_budget else "incomplete_response") in detail
        for detail in result.failure_details
    )


def test_safe_cue_split_still_verifies_each_child_and_never_publishes_partial_sibling(cfg):
    sources = (
        _source(_sentence("I use PostgreSQL for storage in this", 200, terminal=".")),
        _source(_sentence("A hypothetical oscillator uses a spring in this", 200), 18),
    )
    client = _SequenceClient(
        _response(),  # Parent cue causes the existing source-safe second look.
        _response(_claim("PostgreSQL")), _response(),  # Left primary + omission.
        _response(), _response(complete=False),  # Right primary + failed empty check.
    )

    hy = HyMem(cfg)
    try:
        session_id = "synthetic-terminal-policy"
        hy.conn.execute("INSERT INTO sessions(id) VALUES (?)", (session_id,))
        for message_id, encoded in sources:
            hy.conn.execute(
                "INSERT INTO messages(id,session_id,role,content) VALUES (?,?,'user',?)",
                (message_id, session_id, json.loads(encoded)["content"]),
            )
        source_chunk = Chunk(
            id="synthetic-terminal-policy", session_id=session_id,
            start_message_id=17, end_message_id=18,
            salience_reason="long_user_turn", text="ignored",
            source_message_ids=(17, 18),
        )
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, session_id)
            persist_chunks(hy.conn, [source_chunk])

        result = phase1.extract_chunk_results(
            hy.conn, source_chunk, client, prompt_version=hy.config.prompt_version,
        )

        assert result is not None and result.failed
        assert result.failure_reason == "branch_incomplete"
        assert "right:incomplete_response" in result.failure_details
        assert result.completion_calls == result.provider_attempts == len(client.calls) == 5
        assert "OMISSION VERIFICATION PASS" in client.calls[2].system
        assert "EMPTY VERIFICATION PASS" in client.calls[4].system
        # A failed split can carry diagnostic sibling triples internally. The
        # real publication boundary must retain only the failed attempt.
        with core_db.transaction(hy.conn):
            phase1.persist_chunk_results(
                hy.conn, source_chunk, result,
                prompt_version=hy.config.prompt_version, cfg=hy.config,
            )
        assert hy.conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM behavioral_markers").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM processed_chunks").fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT last_failure_reason FROM chunk_extraction_attempts WHERE chunk_id=?",
            (source_chunk.id,),
        ).fetchone()[0] == "branch_incomplete"
    finally:
        hy.close()


@pytest.mark.parametrize("source_backed", [False, True])
def test_unsplittable_oversize_still_fails_before_any_provider_call(source_backed):
    text = _sentence("A hypothetical device uses a spring in this", chunk._MAX_LEAF_INPUT_CHARS + 1)
    client = _SequenceClient()

    result = chunk.extract_chunk(
        client, text, source_records=(_source(text),) if source_backed else None,
    )

    assert result.failed and result.failure_reason == "resource_limit"
    assert "split:no_admissible_semantic_boundary" in result.failure_details
    assert result.triples == [] and result.markers == []
    assert result.completion_calls == result.provider_attempts == len(client.calls) == 0
