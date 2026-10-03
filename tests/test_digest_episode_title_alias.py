"""Lossless wire alias admission leaves canonical digest storage strict."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import pytest

from hymem import HyMem, HyMemConfig
from hymem.dreaming import digest as mod
from hymem.dreaming.lossless import coverage_chunk_id, materialize_message_coverage
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import StubLLMClient
from tests.test_lossless_digest import RollingLLM, _quiet_cfg


def episode(*, alias=False, title="Service deployment", chunk_ids=None):
    return {
        "episode_title" if alias else "title": title,
        "summary": "Deployed the service and verified its health.",
        "outcome": "resolved", "key_entities": ["service"],
        "chunk_ids": ["first", "second"] if chunk_ids is None else chunk_ids,
    }


def payload(items, summary="Deployed the service and verified its health."):
    return {"episodes": items, "summary": summary, "procedures": []}


def validate(data, *, granular=False, max_episodes=8):
    return mod._validate_digest_response(
        json.dumps(data), data, "synthetic-session-hash", ["first", "second"],
        granular=granular, max_episodes=max_episodes,
    )


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("title", ["Service deployment", "  Service deployment  ",
                                  "é e\u0301 🧪 \\\"deployment\\\"\n"])
def test_alias_is_lossless_equivalent_to_canonical_without_mutating_response(granular, title):
    original = payload([episode(), episode(alias=True, title=title + " separate")])
    before = copy.deepcopy(original)
    normalized = mod._normalize_digest_episode_response_items(original["episodes"])
    assert normalized[0] is original["episodes"][0]
    assert normalized[1] is not original["episodes"][1]
    assert normalized[1]["title"] == title + " separate"
    assert "episode_title" not in normalized[1]
    expected = payload([episode(), episode(title=title + " separate")])
    result = validate(original, granular=granular)
    assert not result.parse_failed
    assert asdict(result) == asdict(validate(expected, granular=granular))
    assert original == before


@pytest.mark.parametrize("value", [None, "", " \t\n", 0, False, [], {}, ["title"]])
def test_invalid_alias_values_are_not_normalized_or_accepted(value):
    item = episode(alias=True, title=value)
    assert mod._normalize_digest_episode_response_items([item])[0] is item
    result = validate(payload([item]))
    assert result.parse_failed and result.failure_reason == "episode_validation_failure"
    assert result.episodes.items == [] and result.episode_rejected_items == 1


@pytest.mark.parametrize("extra", [
    {"title": "Service deployment"}, {"title": "Conflicting title"},
    {"title": None}, {"title": ""}, {"unexpected": 1}, {"name": "Service deployment"},
])
def test_ambiguous_or_extended_alias_object_is_rejected(extra):
    item = episode(alias=True) | extra
    assert mod._normalize_digest_episode_response_items([item])[0] is item
    assert validate(payload([item])).failure_reason == "episode_validation_failure"


@pytest.mark.parametrize("alias", ["name", "episodeTitle", "heading", "Title", None])
def test_no_other_alias_or_missing_title_is_invented(alias):
    item = episode()
    title = item.pop("title")
    if alias is not None:
        item[alias] = title
    result = validate(payload([item]))
    assert result.failure_reason == "episode_validation_failure"


@pytest.mark.parametrize("change", [
    {"chunk_ids": ["unknown"]}, {"chunk_ids": []}, {"chunk_ids": None},
    {"chunk_ids": "first"}, {"chunk_ids": [3]},
    {"chunk_ids": ["first", "first"]}, {"chunk_ids": ["second", "first"]},
    {"summary": ""}, {"summary": None}, {"key_entities": [""]},
    {"key_entities": [1]}, {"key_entities": "service"},
    {"outcome": "invented"}, {"outcome": 1},
])
def test_alias_does_not_weaken_item_contract(change):
    result = validate(payload([episode(), episode(alias=True, title="Separate") | change]))
    assert result.parse_failed and result.failure_reason == "episode_validation_failure"
    assert result.episode_input_items == 2 and result.episode_rejected_items == 1
    assert result.episodes.items == [] and result.summary is None


@pytest.mark.parametrize("missing", ["summary", "outcome", "key_entities", "chunk_ids"])
def test_alias_requires_all_canonical_fields(missing):
    item = episode(alias=True)
    item.pop(missing)
    assert validate(payload([item])).failure_reason == "episode_validation_failure"


def test_conflicting_identity_still_rejects_entire_response_and_equal_duplicate_deduplicates():
    same = [episode(), episode(alias=True)]
    result = validate(payload(same))
    assert not result.parse_failed and result.episodes.items == [episode()]
    same[1]["summary"] = "The service was not deployed."
    result = validate(payload(same))
    assert result.failure_reason == "episode_validation_failure"
    assert result.episodes.items == [] and result.episode_rejected_items == 1


def test_alias_items_still_count_against_primary_episode_cap():
    result = validate(payload([episode(alias=True), episode(title="Separate")]),
                      granular=True, max_episodes=1)
    assert result.failure_reason == "episode_output_cap"
    assert result.episode_input_items == 2 and result.episode_rejected_items == 1


def test_staging_and_strict_item_admission_still_require_canonical_title():
    alias = episode(alias=True)
    before = copy.deepcopy(alias)
    assert mod._validate_digest_episode_items([alias], ["first", "second"]) == ([], 1)
    with pytest.raises(RuntimeError, match="violates its extraction contract"):
        mod._validate_digest_staged_items([alias], [], ["first", "second"])
    assert alias == before
    canonical = episode()
    assert mod._validate_digest_staged_items([canonical], [], ["first", "second"]) == ([canonical], [])


class SequenceLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, request):
        self.calls.append(request)
        assert self.responses, "unexpected extra completion"
        return json.dumps(self.responses.pop(0))


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    hy = HyMem(_quiet_cfg(HyMemConfig(root=tmp_path_factory.mktemp("digest-alias"))),
               llm=StubLLMClient(default="[]"))
    sid = "synthetic-title-alias"
    mid = hy.log_message(sid, "user", "Deployed the service and verified its health.")
    hy.close_session(sid)
    materialize_message_coverage(hy.conn, sid)
    yield hy, sid, mid
    hy.close()


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("repair", [False, True])
def test_alias_extraction_and_summary_repair_assemble_canonical_items(source, granular, repair):
    hy, sid, mid = source
    original = payload([episode(alias=True, chunk_ids=[coverage_chunk_id(sid, mid)])],
                       "x" * 644 if repair else "Deployed the service and verified its health.")
    before = copy.deepcopy(original)
    llm = SequenceLLM(original, {"alternatives": ["Deployed the service and verified its health."] * 3})
    changes = hy.conn.total_changes
    result = mod.extract_session_digest(hy.conn, sid, llm, max_chars=12000,
                                        max_tokens=3072, granular=granular)
    assert not result.parse_failed and len(llm.calls) == (2 if repair else 1)
    assert result.episodes.items == [episode(chunk_ids=[coverage_chunk_id(sid, mid)])]
    assert result.covered_message_id == mid and result.caught_up and result.source_sha256
    assert original == before and hy.conn.total_changes == changes
    canonical = payload([episode(chunk_ids=[coverage_chunk_id(sid, mid)])])
    control = SequenceLLM(canonical)
    expected = mod.extract_session_digest(hy.conn, sid, control, max_chars=12000,
                                          max_tokens=3072, granular=granular)
    assert asdict(llm.calls[0]) == asdict(control.calls[0])
    assert asdict(result) == asdict(expected)


def test_bad_alias_prevents_summary_repair_and_does_not_advance_source(source):
    hy, sid, mid = source
    item = episode(alias=True, chunk_ids=[coverage_chunk_id(sid, mid)]) | {"title": "ambiguous"}
    llm = SequenceLLM(payload([item], "x" * 644))
    result = mod.extract_session_digest(hy.conn, sid, llm, max_chars=12000, max_tokens=3072)
    assert result.failure_reason == "episode_validation_failure" and len(llm.calls) == 1
    assert result.covered_message_id is result.source_sha256 is result.summary is None
    assert result.episodes.items == []


class AliasDreamLLM(RollingLLM):
    def complete(self, request):
        raw = super().complete(request)
        if request.system.startswith(("You analyze one conversation session",
                                      "You re-read one conversation session")):
            data = json.loads(raw)
            for item in data["episodes"]:
                item["episode_title"] = item.pop("title")
            return json.dumps(data)
        return raw


def test_canonical_alias_result_stages_publishes_reopens_and_roundtrips(cfg, tmp_path):
    llm = AliasDreamLLM(emit_slice_artifacts=True)
    config = _quiet_cfg(cfg)
    hy = HyMem(config, llm=llm)
    try:
        # Observe actual staging without rebinding a producer callable to a
        # mutable capture closure (that correctly trips the generation fence).
        hy.conn.execute("CREATE TEMP TABLE alias_staged_receipts(payload TEXT)")
        hy.conn.execute("CREATE TEMP TRIGGER observe_alias_stage AFTER INSERT ON digest_staging "
                        "BEGIN INSERT INTO alias_staged_receipts VALUES (new.episodes_json); END")
        hy.log_message("alias-publication", "user", "alpha source deployed the service.")
        hy.close_session("alias-publication")
        report = hy.dream()
        assert report.digest_failures == 0 and report.episodes_created == 1
        staged = [item for row in hy.conn.execute("SELECT payload FROM alias_staged_receipts")
                  for item in json.loads(row[0])]
        assert len(staged) == 1 and len(llm.successful_digest_calls) == 1
        assert all("title" in item and "episode_title" not in item for item in staged)
        assert hy.conn.execute("SELECT count(*) FROM digest_staging").fetchone()[0] == 0
        rows = [tuple(row) for row in hy.conn.execute("SELECT id,title,summary,outcome FROM episodes")]
        assert rows and rows[0][1] == staged[0]["title"]
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []
        archive = tmp_path / "alias-roundtrip.jsonl"
        hy.export(archive)
        assert "episode_title" not in archive.read_text()
    finally:
        hy.close()
    reopened = HyMem(config, llm=llm)
    try:
        assert [tuple(row) for row in reopened.conn.execute("SELECT id,title,summary,outcome FROM episodes")] == rows
        assert reopened.dream_status()["malformed_digests"] == 0
    finally:
        reopened.close()
    restored = HyMem(replace(config, root=tmp_path / "restored"), llm=llm)
    try:
        restored.import_(archive)
        assert [tuple(row) for row in restored.conn.execute("SELECT id,title,summary,outcome FROM episodes")] == rows
        assert restored.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        restored.close()


def test_alias_normalizer_is_bound_to_digest_semantic_identity(monkeypatch):
    client = StubLLMClient(default="[]")
    before = semantic_generation_suffix("digest", client)
    original = mod._normalize_digest_episode_response_items

    def replacement(items):
        return original(items)

    monkeypatch.setattr(mod, "_normalize_digest_episode_response_items", replacement)
    assert semantic_generation_suffix("digest", client) != before
