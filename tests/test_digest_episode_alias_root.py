"""Independent alias admission controls; all content is invented."""
import copy
import json

import pytest

from hymem.dreaming import digest


def episode():
    return {"episode_title": "Deployment 🧪", "summary": "Deployment remains blocked.",
            "outcome": "blocked", "key_entities": ["service"], "chunk_ids": ["c1", "c2"]}


def validate(items):
    data = {"episodes": items, "summary": "Deployment remains blocked pending review.", "procedures": []}
    return digest._validate_digest_response(json.dumps(data), data, "synthetic", ["c1", "c2"],
                                            granular=False, max_episodes=None)


def test_root_alias_is_equivalent_to_canonical_without_mutating_model_data():
    original = episode()
    before = copy.deepcopy(original)
    canonical = {"title": original["episode_title"], **{k: v for k, v in original.items() if k != "episode_title"}}
    normalized = validate([original])
    assert not normalized.parse_failed
    assert normalized == validate([canonical])
    assert original == before
    assert normalized.episodes.items[0]["outcome"] == "blocked"
    assert normalized.episodes.items[0]["chunk_ids"] == ["c1", "c2"]
    assert digest._validate_digest_episode_items([original], ["c1", "c2"])[1] == 1
    with pytest.raises(RuntimeError, match="extraction contract"):
        digest._validate_digest_staged_items([original], [], ["c1", "c2"])


@pytest.mark.parametrize("change", [
    {"title": "Deployment 🧪"}, {"title": "Different title"}, {"name": "Deployment"},
    {"episode_title": ""}, {"episode_title": " \t"}, {"episode_title": 8},
    {"chunk_ids": ["c2", "c1"]}, {"chunk_ids": ["c1", "c1"]},
    {"chunk_ids": ["unknown"]}, {"outcome": "invented"}, {"key_entities": [""]},
])
def test_root_alias_does_not_relax_other_contracts(change):
    result = validate([{**episode(), **change}])
    assert result.parse_failed and result.failure_reason == "episode_validation_failure"
    assert not result.episodes.items and result.summary is None


def test_root_alias_cannot_hide_conflicting_sibling_identity():
    original = episode()
    sibling = {"title": original["episode_title"], **{k: v for k, v in original.items() if k != "episode_title"}}
    sibling["summary"] = "Deployment finished successfully."
    sibling["outcome"] = "resolved"
    result = validate([original, sibling])
    assert result.parse_failed and result.failure_reason == "episode_validation_failure"
    assert result.episodes.items == []
