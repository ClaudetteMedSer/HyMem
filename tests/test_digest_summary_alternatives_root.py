"""Independent whole-candidate selection controls; all text is invented."""
import copy
import json
from dataclasses import asdict

import pytest

from hymem.dreaming import digest as mod
from tests.test_digest_bounded_summary_repair import SequenceLLM, extract, payload, source


@pytest.mark.parametrize("selected", [0, 1, 2])
@pytest.mark.parametrize("granular", [False, True])
def test_root_first_eligible_whole_summary_is_selected_without_touching_items(source, selected, granular):
    good = "Configured staging, deployed the service and verified its health."
    candidates = ["x" * 674, "y" * 594, "z" * 501]
    candidates[selected] = good
    if selected < 2:
        candidates[2] = "A different complete but lower-priority alternative."
    original = payload(source, "d" * 674)
    saved = copy.deepcopy(original)
    client = SequenceLLM(original, {"alternatives": candidates})
    before_writes = source[0].conn.total_changes
    result = extract(source, client, granular=granular, max_episodes=8)
    reference_client = SequenceLLM(payload(source, good))
    reference = extract(source, reference_client, granular=granular, max_episodes=8)
    assert not result.parse_failed and result.summary == good
    assert len(client.calls) == 2 and original == saved
    assert asdict(client.calls[0]) == asdict(reference_client.calls[0])
    assert result.episodes == reference.episodes and result.procedures == reference.procedures
    assert result.source_sha256 == reference.source_sha256
    assert result.covered_message_id == reference.covered_message_id
    assert result.caught_up and source[0].conn.total_changes == before_writes
    assert json.loads(client.calls[1].user) == {
        "original_generation_input": client.calls[0].user,
    }
    assert client.calls[0].max_tokens == client.calls[1].max_tokens == 3072


@pytest.mark.parametrize("unit", ["x", "é", "🧪", "e\u0301"])
def test_root_unicode_count_uses_codepoints_and_never_clips(source, unit):
    candidate = (unit * 501)[:500]
    client = SequenceLLM(payload(source, "x" * 674), {
        "alternatives": [unit * 501, " \n" + candidate + "\t ", "Last valid fallback summary."],
    })
    result = extract(source, client)
    assert not result.parse_failed and result.summary == candidate
    assert len(result.summary) == 500 and len(client.calls) == 2


@pytest.mark.parametrize("unusable", ["", "tiny", '"         tiny         "', "'             '"])
def test_root_meaningless_candidate_rejects_bundle_despite_fitting_successor(source, unusable):
    good = "The deployment remains blocked pending a successful health check."
    client = SequenceLLM(payload(source, "x" * 674), {"alternatives": [unusable, good, "x" * 594]})
    result = extract(source, client)
    assert result.parse_failed and result.failure_reason == "summary_validation_failure"
    assert result.summary is result.source_sha256 is result.covered_message_id is None
    assert not result.episodes.items and not result.procedures.items
    assert len(client.calls) == 2


@pytest.mark.parametrize("bad", [
    {"alternatives": ["A valid complete alternative.", None, "A third valid alternative."]},
    {"alternatives": ["A valid complete alternative.", 42, "A third valid alternative."]},
    {"alternatives": ["A valid complete alternative."]},
    {"alternatives": ["A valid complete alternative."] * 4},
    {"alternatives": ["A valid complete alternative."] * 3, "summary": "Ambiguous mixed response."},
    {"alternatives": ["A valid complete alternative."] * 3, "episodes": []},
    'Prose {"alternatives":["A valid alternative.","A second alternative.","A third alternative."]}',
    '{"alternatives":["A valid alternative.","A second alternative.","A third alternative."],"alternatives":[]}',
    '{"alternatives":["A valid alternative.","cut',
])
def test_root_invalid_bundle_is_not_partially_accepted(source, bad):
    client = SequenceLLM(payload(source, "x" * 674), bad)
    before = source[0].conn.total_changes
    result = extract(source, client)
    assert result.parse_failed and result.failure_stage == "summary_compaction"
    assert len(client.calls) == 2 and source[0].conn.total_changes == before
    assert result.summary is result.source_sha256 is result.covered_message_id is None
    assert result.episodes.items == result.procedures.items == []


def test_root_three_overflows_remain_honest_failure_not_three_truncated_choices(source):
    client = SequenceLLM(payload(source, "x" * 674), {"alternatives": ["a" * 594, "b" * 501, "c" * 610]})
    result = extract(source, client)
    assert result.parse_failed and result.failure_reason == "summary_output_cap"
    assert result.failure_stage == "summary_compaction" and len(client.calls) == 2
    assert result.summary is result.source_sha256 is result.covered_message_id is None


def test_root_saved_failure_shape_is_not_claimed_fixed_without_new_response(source):
    client = SequenceLLM(payload(source, "x" * 674), {"summary": "y" * 594})
    result = extract(source, client)
    assert result.parse_failed and result.failure_reason == "shape_failure"
    assert len(client.calls) == 2 and result.source_sha256 is None
