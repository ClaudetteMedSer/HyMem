"""F2 source/field wiring and containment, not real-model entailment accuracy."""
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
import json
import re

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import digest
from hymem.dreaming.lossless import CoveredMessage, materialize_message_coverage
from hymem.dreaming.procedures import validate_procedure_items
from tests.digest_verification_fixtures import resolve_fidelity_sources, synthetic_format_approval, synthetic_format_result
from tests.test_digest_summary_contract import _config


def _episode(citations, **changes):
    return {
        "title": "Deployment advice", "summary": "The assistant suggested checking the build.",
        "outcome": "informational", "key_entities": ["build"], "chunk_ids": citations,
        **changes,
    }


def _procedure(citations, **changes):
    return {
        "name": " Check build ", "description": " Check before deploying. ",
        "steps": [{"order": 30, "action": " Deploy only after checks pass. ", "tool": None},
                  {"order": 5, "action": " Run checks. ", "tool": "test-runner"}],
        "triggers": ["Before deployment"], "entities_involved": ["build"],
        "chunk_ids": citations, **changes,
    }


def _response(episodes=1, procedures=1):
    return {key: [{"index": index, "verdict": "supported"} for index in range(count)]
            for key, count in (("episode_titles", episodes), ("episode_content", episodes),
                               ("procedures", procedures), ("summary_content", 1))}


class _ItemClient:
    def __init__(self, build_items, *, verdicts=None, long_summary=False):
        self.build_items = build_items
        self.verdicts = verdicts
        self.long_summary = long_summary
        self.calls = []
        self.primary = None

    def complete(self, request):
        self.calls.append(request)
        format_result = synthetic_format_approval(request)
        if format_result is not None:
            return format_result
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            payload = json.loads(request.user)
            response = _response(len(payload["items"]), len(payload["procedure_items"]))
            for family, index, verdict in self.verdicts or ():
                response[family][index]["verdict"] = verdict
            return json.dumps(response)
        if request.system.startswith("You compact one rolling conversation summary"):
            return '{"summary":"The assistant provided conditional deployment advice."}'
        assert request.system.startswith(("You analyze one conversation session",
                                          "You re-read one conversation session"))
        citations = re.findall(r"\[chunk ([^\]]+)\]", request.user)
        episodes, procedures = self.build_items(citations)
        self.primary = {"episodes": episodes, "procedures": procedures,
                        "summary": "long " * 110 if self.long_summary else
                        "The assistant provided conditional deployment advice."}
        return json.dumps(self.primary)


def _seed(hy):
    first = hy.log_message("f2-items", "assistant", "Run checks with test-runner; deploy only if they pass.")
    second = hy.log_message("f2-items", "user", "I asked about scenic riding; I have not deployed anything.")
    hy.close_session("f2-items")
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "f2-items")
    return first, second


def _extract(hy, client, granular=False):
    return digest.extract_session_digest(
        hy.conn, "f2-items", client, max_tokens=2048, max_chars=10000,
        granular=granular, max_episodes=8 if granular else None,
        prior_summary="PRIOR_DERIVED_NOT_NEW_EVIDENCE",
    )


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("field,value", [
    ("summary", "The user completed deployment successfully."),
    ("key_entities", ["invented production-cluster"]),
    ("outcome", "resolved"),
])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_episode_fields_are_checked_independently_of_supported_title(
    cfg, granular, field, value, verdict,
):
    client = _ItemClient(lambda ids: ([_episode(ids[:1], **{field: value})], []),
                         verdicts=[("episode_content", 0, verdict)])
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        _seed(hy)
        result = _extract(hy, client, granular)
        assert len(client.calls) == 2
        payload = json.loads(client.calls[-1].user)
        item = payload["items"][0]
        key = {"summary": "candidate_body", "key_entities": "candidate_key_entities",
               "outcome": "candidate_outcome"}[field]
        assert item[key] == value
        assert "PRIOR_DERIVED_NOT_NEW_EVIDENCE" not in json.dumps(payload["items"])
        assert payload["summary_item"]["prior_derived_summary"] == "PRIOR_DERIVED_NOT_NEW_EVIDENCE"
        assert result.failure_reason == "episode_content_" + verdict
        assert result.failure_stage == "fidelity_verification"
        assert result.parse_failed and result.episodes.items == []
        assert result.covered_message_id is result.source_sha256 is result.summary is None
        assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


@pytest.mark.parametrize("field,value", [
    ("name", "Guaranteed successful deployment"),
    ("description", "Deploy without checking anything."),
    ("steps", [{"order": 1, "action": "Deploy now.", "tool": "invented-cli"},
               {"order": 2, "action": "Run checks later.", "tool": None}]),
    ("triggers", ["When checks fail"]),
    ("entities_involved", ["invented production-cluster"]),
])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_procedure_only_candidates_require_the_same_gate_and_all_fields(cfg, field, value, verdict):
    client = _ItemClient(lambda ids: ([], [_procedure(ids[:1], **{field: value})]),
                         verdicts=[("procedures", 0, verdict)])
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy)
        result = _extract(hy, client)
        assert len(client.calls) == 2
        payload = json.loads(client.calls[-1].user)
        assert payload["items"] == []
        candidate = payload["procedure_items"][0]["candidate"]
        assert candidate == validate_procedure_items(client.primary["procedures"])[0]
        assert result.failure_reason == "procedure_content_" + verdict
        assert result.failure_stage == "fidelity_verification" and result.parse_failed
        assert result.procedure_input_items == result.procedure_rejected_items == 1
        assert result.covered_message_id is result.source_sha256 is result.summary is None
        assert result.episodes.items == result.procedures.items == []


@pytest.mark.parametrize("long_summary", [False, True])
def test_all_item_kinds_share_one_final_verification_with_exact_published_normalization(cfg, long_summary):
    client = _ItemClient(lambda ids: ([_episode(ids[:1]), _episode(ids[1:], title="Riding inquiry")],
                                      [_procedure(ids[:1])]), long_summary=long_summary)
    with closing(HyMem(_config(cfg, True), llm=client)) as hy:
        first, second = _seed(hy)
        result = _extract(hy, client, True)
        assert not result.parse_failed and result.covered_message_id == second
        assert len(client.calls) == (4 if long_summary else 3)
        primary, verification = client.calls[0], client.calls[-2]
        assert replace(verification, system=primary.system, user=primary.user) == primary
        assert sum(call.system == digest._DIGEST_FIDELITY_SYSTEM for call in client.calls) == 1
        payload = json.loads(verification.user)
        assert [resolve_fidelity_sources(payload, item["cited_source_ids"])[0]["message_id"]
                for item in payload["items"]] == [first, second]
        proc = payload["procedure_items"][0]
        assert [source["message_id"] for source in resolve_fidelity_sources(payload, proc["cited_source_ids"])] == [first]
        assert proc["candidate"] == result.procedures.items[0]
        assert proc["candidate"]["steps"] == [
            {"order": 1, "action": "Run checks.", "tool": "test-runner"},
            {"order": 2, "action": "Deploy only after checks pass.", "tool": None},
        ]
        assert result.episodes.items == client.primary["episodes"]


@pytest.mark.parametrize("second_verdict", ["supported", "unsupported", "uncertain"])
def test_normalized_procedure_duplicates_keep_every_original_citation_set(cfg, second_verdict):
    client = _ItemClient(lambda ids: ([], [_procedure(ids[:1]), _procedure(ids[1:])]),
                         verdicts=[("procedures", 1, second_verdict)])
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        first, second = _seed(hy)
        result = _extract(hy, client)
        payload = json.loads(next(call.user for call in client.calls if call.system == digest._DIGEST_FIDELITY_SYSTEM))
        items = payload["procedure_items"]
        assert len(items) == 2 and items[0]["candidate"] == items[1]["candidate"]
        assert [[s["message_id"] for s in resolve_fidelity_sources(payload, item["cited_source_ids"])]
                for item in items] == [[first], [second]]
        assert result.procedure_input_items == 2
        if second_verdict == "supported":
            assert not result.parse_failed and len(result.procedures.items) == 1
        else:
            assert result.parse_failed and result.procedure_rejected_items == 2
            assert result.procedures.items == [] and result.source_sha256 is None


def test_visible_span_and_boundary_context_do_not_expand_to_other_items_or_full_message():
    prefix = "Earlier independent trip. I have always wa"
    visible = "nted to ride horses."
    first = CoveredMessage(7, "s", "user", prefix + visible, "a", source_peer_id="p")
    second = CoveredMessage(8, "s", "assistant", "Run checks. UNSEEN: deploy unconditionally.", "b")
    uncited = CoveredMessage(9, "s", "tool", "UNRELATED_PRIVATE_TEXT", "c")
    episode = _episode(["a"], summary="The user always wanted to ride horses.", key_entities=[])
    procedure = _procedure(["b"])
    original = deepcopy((episode, procedure))
    payload = digest._digest_fidelity_payload(
        [episode], [first, second, uncited], ["a", "b"],
        before_cursor=(6, 7, len(prefix)), after_cursor=(7, 8, len("Run checks.")),
        leading_context=None, raw_procedures=[procedure],
    )
    episode_source = resolve_fidelity_sources(payload, payload["items"][0]["cited_source_ids"])[0]
    proc_source = resolve_fidelity_sources(payload, payload["procedure_items"][0]["cited_source_ids"])[0]
    assert episode_source["visible_content"] == visible
    assert episode_source["interpretation_only_context"]["content"] == prefix
    assert episode_source["role"] == "user" and episode_source["source_peer_id"] == "p"
    assert proc_source["visible_content"] == "Run checks." and proc_source["role"] == "assistant"
    assert "UNSEEN" not in json.dumps(payload) and "UNRELATED_PRIVATE_TEXT" not in json.dumps(payload)
    assert "Earlier independent trip" not in json.dumps(payload["procedure_items"])
    assert original == (episode, procedure)
    policy = digest._DIGEST_FIDELITY_SYSTEM
    assert "separate trip mentioned only in context is not" in policy
    assert "Another item's sources cannot support this item" in policy
    assert "relative order of steps" in policy


@pytest.mark.parametrize("family", ["episode_titles", "episode_content", "procedures", "summary_content",
                                    "summary_format", "episode_format"])
@pytest.mark.parametrize("bad", [
    None, {}, [], [{"index": True, "verdict": "supported"}],
    [{"index": 0.0, "verdict": "supported"}], [{"index": "0", "verdict": "supported"}],
    [{"index": -1, "verdict": "supported"}], [{"index": 1, "verdict": "supported"}],
    [{"index": 0, "verdict": "supported", "reason": "not allowed"}],
    [{"index": 0}], [{"index": 0, "verdict": True}],
    [{"index": 0, "verdict": "maybe"}], [{"index": 0, "verdict": "SUPPORTED"}],
    [{"index": 0, "verdict": "supported"}, {"index": 0, "verdict": "supported"}],
])
def test_all_verdict_families_require_exact_complete_unique_coverage(family, bad):
    is_format = family in {"summary_format", "episode_format"}
    response = synthetic_format_result(1) if is_format else _response()
    response[family] = bad
    if is_format:
        assert digest._validate_digest_format_adjudication_response(json.dumps(response), 1) == "format_adjudication_shape_failure"
    else:
        assert digest._validate_digest_fidelity_response(json.dumps(response), 1, 1) == "fidelity_shape_failure"


@pytest.mark.parametrize("family", ["episode_titles", "episode_content", "procedures", "summary_content",
                                    "summary_format", "episode_format"])
def test_missing_extra_or_duplicate_family_is_rejected_even_with_other_rejections(family):
    is_format = family in {"summary_format", "episode_format"}
    response = synthetic_format_result(1) if is_format else _response()
    response["summary_format" if is_format else "episode_titles"][0]["verdict"] = "unsupported"
    validator = (lambda raw: digest._validate_digest_format_adjudication_response(raw, 1)) if is_format else (
        lambda raw: digest._validate_digest_fidelity_response(raw, 1, 1))
    prefix = "format_adjudication" if is_format else "fidelity"
    complete = json.dumps(response)
    del response[family]
    assert validator(json.dumps(response)) == prefix + "_shape_failure"
    response = json.loads(complete)
    response["extra"] = []
    assert validator(json.dumps(response)) == prefix + "_shape_failure"
    raw = complete[:-1] + ',"' + family + '":[]}'
    assert validator(raw) == prefix + "_parse_failure"


def test_verdict_index_order_is_irrelevant_but_semantic_failures_are_distinct():
    response = _response(2, 2)
    for verdicts in response.values():
        verdicts.reverse()
    assert digest._validate_digest_fidelity_response(json.dumps(response), 2, 2) is None
    response["episode_content"][0]["verdict"] = "unsupported"
    assert digest._validate_digest_fidelity_response(json.dumps(response), 2, 2) == "episode_content_unsupported"
    response["procedures"] = []
    assert digest._validate_digest_fidelity_response(json.dumps(response), 2, 2) == "fidelity_shape_failure"
    response = _response(0, 1)
    response["procedures"][0]["verdict"] = "uncertain"
    assert digest._validate_digest_fidelity_response(json.dumps(response), 0, 1) == "procedure_content_uncertain"


def test_empty_families_are_mandatory_and_cannot_hide_unexpected_items():
    assert digest._validate_digest_fidelity_response(json.dumps(_response(0, 0)), 0, 0) is None
    assert digest._validate_digest_fidelity_response(json.dumps(_response(1, 0)), 0, 0) == "fidelity_shape_failure"
    assert digest._validate_digest_fidelity_response(json.dumps(_response(0, 1)), 0, 0) == "fidelity_shape_failure"
