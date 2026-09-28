"""F5 source-catalog transport controls; synthetic verdicts are not accuracy proof."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, replace
import json
import re

import pytest

from hymem import HyMem
from hymem.dreaming import digest
from hymem.dreaming.lossless import CoveredMessage
from tests.digest_verification_fixtures import synthetic_fidelity_result, synthetic_format_approval
from tests.test_digest_item_verification import _procedure
from tests.test_digest_summary_contract import _config, _seed


def _episode(source_ids, index=0):
    return {
        "title": f"Decision {index}",
        "summary": f"Decision {index}: feature_{index} is enabled.",
        "outcome": "informational", "key_entities": [], "chunk_ids": source_ids,
    }


class _TransportClient:
    def __init__(self, *, count=12, citations=None):
        self.count = count
        self.citations = citations
        self.calls = []
        self.primary = None

    def complete(self, request):
        self.calls.append(request)
        format_result = synthetic_format_approval(request)
        if format_result is not None:
            return format_result
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            payload = json.loads(request.user)
            return json.dumps(synthetic_fidelity_result(
                len(payload["items"]), len(payload["procedure_items"]),
            ))
        assert request.system.startswith("You re-read one conversation session")
        ids = re.findall(r"\[chunk ([^\]]+)\]", request.user)
        refs = ids if self.citations is None else self.citations(ids)
        self.primary = {
            "episodes": [_episode(refs, index) for index in range(self.count)],
            "summary": "Twelve feature settings were enabled.", "procedures": [],
        }
        return json.dumps(self.primary)


def _direct_payload(*, episodes=None, messages=None, ids=None, procedures=None):
    return digest._digest_fidelity_payload(
        [_episode(["a"])] if episodes is None else episodes,
        [CoveredMessage(1, "s", "user", "Decision 0: feature_0 is enabled.", "a")]
        if messages is None else messages,
        ["a"] if ids is None else ids,
        before_cursor=(None, None, 0), after_cursor=(1, None, 0),
        leading_context=None, raw_procedures=procedures,
        raw_summary="The feature was enabled.", published_summary="The feature was enabled.",
    )


def test_shipped_twelve_episode_window_fits_once_without_extra_calls_or_truncation(cfg):
    claims = [f"Decision {index}: feature_{index} is enabled." for index in range(12)]
    source = ("\n".join(claims) + "\n" + "Additional background is unchanged. " * 330)[:11400]
    assert len(source) == 11400 and all(claim in source for claim in claims)
    client = _TransportClient()
    with closing(HyMem(_config(cfg, True), llm=client)) as hy:
        last_id = _seed(hy, source)
        result = digest.extract_session_digest(
            hy.conn, "bounded-summary", client, max_tokens=3072, max_chars=12000,
            granular=True, max_episodes=12,
        )
        assert not result.parse_failed and result.caught_up
        assert result.covered_message_id == last_id and result.source_sha256 is not None
        assert result.episodes.items == client.primary["episodes"]
        assert len(client.calls) == 3
        primary, verification, final_format = client.calls
        assert final_format.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
        assert replace(verification, system=primary.system, user=primary.user) == primary
        payload = json.loads(verification.user)
        catalog = payload["source_catalog"]
        assert len(catalog) == 1
        assert catalog[0]["visible_content"] == source
        assert catalog[0]["start"] == 0 and catalog[0]["end"] == len(source)
        assert catalog[0]["message_id"] == last_id
        assert payload["summary_item"]["new_source_ids"] == [catalog[0]["chunk_id"]]
        assert all(item["cited_source_ids"] == [catalog[0]["chunk_id"]] for item in payload["items"])
        assert verification.user.count(json.dumps(source, ensure_ascii=False)[1:-1]) == 1
        assert len(verification.user) + len(verification.system) < digest._DIGEST_FIDELITY_MAX_INPUT_CHARS
        # Reconstruct only the old transport duplication, not old source authority.
        # This same valid candidate would previously be falsely held at the cap.
        old_packet = deepcopy(payload)
        del old_packet["source_catalog"]
        for item in old_packet["items"]:
            del item["cited_source_ids"]
            item["cited_sources"] = catalog
        del old_packet["summary_item"]["new_source_ids"]
        old_packet["summary_item"]["new_sources"] = catalog
        assert len(json.dumps(old_packet, separators=(",", ":"))) > digest._DIGEST_FIDELITY_MAX_INPUT_CHARS
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


def test_catalog_deduplicates_transport_without_pooling_episode_or_procedure_authority():
    prefix = "Earlier independent trip. I always wa"
    visible = "nted to ride 🐎 near Cafe\u0301.\n"
    first = CoveredMessage(7, "s", "user", prefix + visible, "a", source_peer_id="p", source_workspace_id="w")
    second = CoveredMessage(8, "s", "assistant", "Run checks. UNSEEN_SUFFIX", "b", source_peer_id="helper")
    future = CoveredMessage(9, "s", "tool", "OUTSIDE_WINDOW", "c")
    episodes = [_episode(["a"]), _episode(["a", "b"], 1)]
    procedures = [_procedure(["a"]), _procedure(["b"])]
    originals = deepcopy((episodes, procedures, [asdict(item) for item in (first, second, future)]))
    payload = digest._digest_fidelity_payload(
        episodes, [first, second, future], ["a", "b"],
        before_cursor=(6, 7, len(prefix)), after_cursor=(7, 8, len("Run checks.")),
        leading_context=None, raw_procedures=procedures,
        prior_summary="PRIOR_IS_DERIVED_ONLY",
    )
    assert [item["chunk_id"] for item in payload["source_catalog"]] == ["a", "b"]
    first_record, second_record = payload["source_catalog"]
    assert first_record == {
        "chunk_id": "a", "message_id": 7, "role": "user", "source_peer_id": "p",
        "source_workspace_id": "w", "start": len(prefix), "end": len(first.content),
        "visible_content": visible,
        "interpretation_only_context": {
            "message_id": 7, "role": "user", "source_peer_id": "p", "source_workspace_id": "w",
            "start": 0, "end": len(prefix), "content": prefix,
        },
    }
    assert second_record["visible_content"] == "Run checks."
    assert second_record["role"] == "assistant" and second_record["source_peer_id"] == "helper"
    assert second_record["interpretation_only_context"] is None
    assert [item["cited_source_ids"] for item in payload["items"]] == [["a"], ["a", "b"]]
    assert [item["cited_source_ids"] for item in payload["procedure_items"]] == [["a"], ["b"]]
    assert payload["procedure_items"][0]["candidate"] == payload["procedure_items"][1]["candidate"]
    assert payload["summary_item"]["new_source_ids"] == ["a", "b"]
    assert "PRIOR_IS_DERIVED_ONLY" not in json.dumps(payload["source_catalog"] + payload["items"] + payload["procedure_items"])
    rendered = digest._encode_digest_fidelity_payload(payload)
    assert "UNSEEN_SUFFIX" not in rendered and "OUTSIDE_WINDOW" not in rendered
    assert rendered.count(json.dumps(visible, ensure_ascii=False)[1:-1]) == 1
    assert originals == (episodes, procedures, [asdict(item) for item in (first, second, future)])
    policy = digest._DIGEST_FIDELITY_SYSTEM
    assert "Catalog presence alone never authorizes an item's claim" in policy
    assert "Do not guess, add or substitute citations" in policy
    assert "summary_item.new_source_ids" in policy


@pytest.mark.parametrize("kind", ["episode", "procedure"])
@pytest.mark.parametrize("refs", [[], None, "a", ["missing"], ["unseen"], ["a", "a"], [True], [{}]])
def test_transport_rejects_missing_duplicate_unknown_and_unseen_item_references(kind, refs):
    messages = [CoveredMessage(1, "s", "user", "Visible.", "a"),
                CoveredMessage(2, "s", "user", "Unseen future source.", "unseen")]
    kwargs = {"episodes": [_episode(refs)]} if kind == "episode" else {"procedures": [_procedure(refs)]}
    with pytest.raises(ValueError, match="invalid fidelity item source references"):
        _direct_payload(messages=messages, **kwargs)


@pytest.mark.parametrize("kind", ["episode", "procedure"])
def test_transport_rejects_absent_item_reference_field(kind):
    item = _episode(["a"]) if kind == "episode" else _procedure(["a"])
    del item["chunk_ids"]
    kwargs = {"episodes": [item]} if kind == "episode" else {"procedures": [item]}
    with pytest.raises(ValueError, match="invalid fidelity item source references"):
        _direct_payload(**kwargs)


@pytest.mark.parametrize("ids", [["a", "a"], [None], [True], [{}], [""], "a"])
def test_transport_rejects_ambiguous_window_identifiers(ids):
    with pytest.raises(ValueError, match="invalid fidelity window source identifiers"):
        _direct_payload(ids=ids)


def test_transport_rejects_missing_catalog_record():
    with pytest.raises(ValueError, match="fidelity source set differs"):
        _direct_payload(ids=["a", "missing"])


@pytest.mark.parametrize("duplicate", [
    CoveredMessage(1, "s", "user", "Original source.", "a"),
    CoveredMessage(2, "s", "assistant", "Conflicting overwrite.", "a"),
    CoveredMessage(1, "s", "user", "Original source.", "alias"),
])
def test_transport_never_overwrites_or_aliases_a_duplicate_canonical_record(duplicate):
    messages = [CoveredMessage(1, "s", "user", "Original source.", "a"), duplicate]
    ids = ["a", "alias"] if duplicate.chunk_id == "alias" else ["a"]
    with pytest.raises(ValueError, match="duplicate fidelity source record"):
        _direct_payload(messages=messages, ids=ids)


@pytest.mark.parametrize("citation_kind", ["unknown", "duplicate", "empty"])
def test_invalid_primary_citations_do_not_spend_the_verifier_call(cfg, citation_kind):
    def citations(ids):
        return {"unknown": ["outside-window"], "duplicate": ids * 2, "empty": []}[citation_kind]
    client = _TransportClient(count=1, citations=citations)
    with closing(HyMem(_config(cfg, True), llm=client)) as hy:
        _seed(hy)
        result = digest.extract_session_digest(
            hy.conn, "bounded-summary", client, max_tokens=3072, max_chars=12000,
            granular=True, max_episodes=12,
        )
        assert result.parse_failed and result.failure_stage == "primary"
        assert len(client.calls) == 1
        assert result.covered_message_id is result.source_sha256 is None
        assert result.episodes.items == []


def test_real_unique_input_overflow_still_holds_without_truncation_or_verifier_call(cfg):
    client = _TransportClient(count=1)
    with closing(HyMem(_config(cfg, True), llm=client)) as hy:
        _seed(hy)
        result = digest.extract_session_digest(
            hy.conn, "bounded-summary", client, max_tokens=3072, max_chars=12000,
            granular=True, max_episodes=12,
            prior_summary="Prior derived continuity. " * 6000,
        )
        assert result.parse_failed and result.failure_reason == "fidelity_input_cap"
        assert result.failure_stage == "fidelity_verification"
        assert len(client.calls) == 1
        assert result.covered_message_id is result.source_sha256 is result.summary is None
        assert result.episodes.items == result.procedures.items == []
        assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)
        assert digest._DIGEST_FIDELITY_MAX_INPUT_CHARS == 131_072
        assert digest._DIGEST_FIDELITY_MAX_OUTPUT_CHARS == 65_536
