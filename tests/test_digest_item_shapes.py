"""Malformed JSON item values must hold a digest, never escape as TypeError."""
from contextlib import closing
from copy import deepcopy
import json

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.core import db
from hymem.dreaming import digest
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient


_CHUNKS = ["msgcov_first", "msgcov_second"]


def _episode(**changes):
    return {
        "title": "Deployment advice",
        "summary": "The assistant suggested checking the build.",
        "outcome": "informational",
        "key_entities": ["build"],
        "chunk_ids": list(_CHUNKS),
        **changes,
    }


def _procedure(**changes):
    return {
        "name": "Check build",
        "description": "Check before deploying.",
        "steps": [{"order": 1, "action": "Run checks.", "tool": None}],
        "triggers": ["Before deployment"],
        "entities_involved": ["build"],
        "chunk_ids": list(_CHUNKS),
        **changes,
    }


def _assert_held(result, kind, count=1):
    assert result is not None and result.parse_failed
    assert result.failure_reason == f"{kind}_validation_failure"
    assert result.failure_stage == "primary"
    assert getattr(result, f"{kind}_input_items") == count
    assert getattr(result, f"{kind}_rejected_items") == 1
    assert result.episodes.items == []
    assert result.procedures.items == []
    assert result.summary is None
    assert result.covered_message_id is None
    assert result.start_message_id is None
    assert result.end_message_id is None
    assert result.partial_message_id is None
    assert result.next_message_offset == 0
    assert result.source_sha256 is None
    assert not result.caught_up


def _validate(episodes=(), procedures=(), *, granular=False):
    payload = {"episodes": list(episodes), "procedures": list(procedures),
               "summary": "The assistant offered deployment advice."}
    raw = json.dumps(payload)
    decoded = json.loads(raw)
    original = deepcopy(decoded)
    result = digest._validate_digest_response(
        raw, decoded, "item-shapes", _CHUNKS,
        granular=granular, max_episodes=8 if granular else None,
    )
    assert decoded == original
    return result


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("kind,factory", [("episode", _episode), ("procedure", _procedure)])
@pytest.mark.parametrize("chunk_ids", [
    None, False, 1, 1.5, "msgcov_first", {}, [],
    [[]], [{}], [None], [False], [1], [1.5],
    ["msgcov_first", []], ["msgcov_first", {}],
    ["msgcov_first", "msgcov_first"],
    ["msgcov_second", "msgcov_first"], ["unknown"],
])
def test_digest_citation_json_shapes_fail_closed(granular, kind, factory, chunk_ids):
    items = [factory(chunk_ids=chunk_ids)]
    result = _validate(**{f"{kind}s": items}, granular=granular)
    _assert_held(result, kind)


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("outcome", [[], {}, False, True, 0, 1, 1.5, "", "unknown"])
def test_digest_outcome_json_shapes_fail_closed(granular, outcome):
    result = _validate([_episode(outcome=outcome)], granular=granular)
    _assert_held(result, "episode")


@pytest.mark.parametrize("outcome", [None, "resolved", "blocked", "deferred", "informational"])
def test_digest_valid_items_keep_normalization_and_deduplication(outcome):
    episode = _episode(title=" Deployment advice ", summary=" Advice. ", outcome=outcome)
    procedure = _procedure(name=" Check build ", steps=[
        {"order": 20, "action": " Deploy. ", "tool": None},
        {"order": 10, "action": " Run checks. ", "tool": "test-runner"},
    ])
    result = _validate([episode, deepcopy(episode)], [procedure, deepcopy(procedure)])
    assert not result.parse_failed and result.failure_reason is None
    assert result.episode_input_items == result.procedure_input_items == 2
    assert result.episode_rejected_items == result.procedure_rejected_items == 0
    assert result.episodes.items == [{**episode, "title": "Deployment advice", "summary": "Advice."}]
    assert result.procedures.items == [{
        "name": "Check build", "description": "Check before deploying.",
        "steps": [{"order": 1, "action": "Run checks.", "tool": "test-runner"},
                  {"order": 2, "action": "Deploy.", "tool": None}],
        "triggers": ["Before deployment"], "entities_involved": ["build"],
    }]


@pytest.mark.parametrize("kind,changes", [
    ("episode", {"title": {}}), ("episode", {"summary": []}),
    ("episode", {"key_entities": [{}]}),
    ("procedure", {"name": {}}), ("procedure", {"description": []}),
    ("procedure", {"steps": [None]}),
    ("procedure", {"steps": [{"order": {}, "action": "Run.", "tool": None}]}),
    ("procedure", {"steps": [{"order": [], "action": "Run.", "tool": None}]}),
    ("procedure", {"steps": [{"order": 1, "action": {}, "tool": None}]}),
    ("procedure", {"steps": [{"order": 1, "action": "Run.", "tool": []}]}),
    ("procedure", {"triggers": [{}]}),
    ("procedure", {"entities_involved": [[]]}),
])
def test_adjacent_nested_item_shapes_remain_rejected(kind, changes):
    factory = _episode if kind == "episode" else _procedure
    result = _validate(**{f"{kind}s": [factory(**changes)]})
    _assert_held(result, kind)


@pytest.mark.parametrize("changes", [{"chunk_ids": [[]]}, {"chunk_ids": [{}]},
                                     {"outcome": []}, {"outcome": {}}])
def test_staged_malformed_episode_raises_contract_error_not_type_error(changes):
    with pytest.raises(RuntimeError, match="violates its extraction contract"):
        digest._validate_digest_staged_items([_episode(**changes)], [], _CHUNKS)


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("long_summary", [False, True])
@pytest.mark.parametrize("kind,changes", [
    ("episode", {"chunk_ids": [[]]}), ("episode", {"chunk_ids": [{}]}),
    ("episode", {"outcome": []}), ("episode", {"outcome": {}}),
    ("procedure", {"chunk_ids": [[]]}), ("procedure", {"chunk_ids": [{}]}),
])
def test_extract_malformed_item_stops_before_recovery_or_publication(
    tmp_path, granular, long_summary, kind, changes,
):
    # Exercise the real canonical-coverage/store path with a one-call local
    # client. Invalid item shape must win even when the summary also needs repair.
    config = HyMemConfig(root=tmp_path, aggregation_nodes_enabled=False,
                         profile_extraction_enabled=False, facts_extraction_enabled=False)
    with closing(HyMem(config, llm=StubLLMClient(default="[]"),
                       embedding_client=StubEmbeddingClient())) as hy:
        sid = "malformed-json-item"
        hy.log_message(sid, "assistant", "Run checks; deploy only if they pass.")
        hy.close_session(sid)
        with db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, sid)
        chunk_ids = [message.chunk_id for message in digest.covered_messages_after(hy.conn, sid, None)]
        episode = _episode(chunk_ids=chunk_ids)
        procedure = _procedure(chunk_ids=chunk_ids)
        factory = _episode if kind == "episode" else _procedure
        malformed = factory(**{**{"chunk_ids": chunk_ids}, **changes})
        payload = {
            "episodes": [episode, malformed] if kind == "episode" else [episode],
            "procedures": [procedure, malformed] if kind == "procedure" else [procedure],
            "summary": "long " * 110 if long_summary else "The assistant offered deployment advice.",
        }

        class OneCallClient:
            calls = 0

            def complete(self, request):
                self.calls += 1
                assert self.calls == 1, "Malformed items must not invoke repair or verification"
                return json.dumps(payload)

        client = OneCallClient()
        before = dict(hy.conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone())
        changes_before = hy.conn.total_changes
        result = digest.extract_session_digest(
            hy.conn, sid, client, max_tokens=2048, max_chars=10000,
            granular=granular, max_episodes=8 if granular else None,
        )
        _assert_held(result, kind, count=2)
        assert client.calls == 1
        assert hy.conn.total_changes == changes_before
        assert dict(hy.conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()) == before
