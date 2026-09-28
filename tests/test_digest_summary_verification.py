"""F3 bounded summary/source wiring; scripted verdicts are not model-accuracy proof."""
from contextlib import closing
from dataclasses import replace
import json
import re

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import digest
from hymem.dreaming.lossless import CoveredMessage, materialize_message_coverage
from tests.digest_verification_fixtures import resolve_fidelity_sources, synthetic_fidelity_result, synthetic_format_approval
from tests.test_digest_summary_contract import _config


class _SummaryClient:
    def __init__(self, candidate, *, verdict="supported", compacted=None, with_items=False):
        self.candidate = candidate
        self.verdict = verdict
        self.compacted = compacted
        self.with_items = with_items
        self.calls = []
        self.primary = None

    def complete(self, request):
        self.calls.append(request)
        format_result = synthetic_format_approval(request)
        if format_result is not None:
            return format_result
        if request.system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            return '{"issues":[]}'
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            if isinstance(self.verdict, BaseException):
                raise self.verdict
            payload = json.loads(request.user)
            result = synthetic_fidelity_result(len(payload["items"]), len(payload["procedure_items"]))
            result["summary_content"][0]["verdict"] = self.verdict
            return json.dumps(result)
        if request.system.startswith("You compact one rolling conversation summary"):
            return json.dumps({"summary": self.compacted})
        if request.system.startswith("You repair one rolling conversation summary"):
            # Repeat the same scripted candidate, never auto-approve or invent
            # a successful repair in tests of the original semantic veto.
            return json.dumps({"summary": self.compacted if self.compacted is not None else self.candidate})
        assert request.system.startswith(("You analyze one conversation session",
                                          "You re-read one conversation session"))
        citations = re.findall(r"\[chunk ([^\]]+)\]", request.user)
        self.primary = {"summary": self.candidate, "episodes": [], "procedures": []}
        if self.with_items:
            self.primary["episodes"] = [{
                "title": "Checks before deployment", "summary": "The assistant supplied conditional build steps.",
                "outcome": "informational", "key_entities": ["build"], "chunk_ids": citations,
            }]
            self.primary["procedures"] = [{
                "name": "Check the build", "description": "Check before deployment.",
                "steps": [{"order": 1, "action": "Run checks.", "tool": None}],
                "triggers": ["Before deployment"], "entities_involved": ["build"], "chunk_ids": citations,
            }]
        return json.dumps(self.primary)


def _seed(hy, source):
    hy.log_message("summary-verification", "user", "What should I do next?")
    last = hy.log_message("summary-verification", "assistant", source)
    hy.close_session("summary-verification")
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "summary-verification")
    return last


def _extract(hy, client, *, granular=False, prior=None):
    return digest.extract_session_digest(
        hy.conn, "summary-verification", client, max_tokens=2048, max_chars=10000,
        prior_summary=prior, granular=granular, max_episodes=8 if granular else None,
    )


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("source,candidate,policy", [
    ("Here are riding recommendations and scenic directions.",
     "The user asked about riding recommendations and scenic directions.", "do not report only that they were requested"),
    ("Your route question remains unanswered.", "The route question was resolved.", "if unanswered, do not invent an answer"),
    ("I completed the canyon trip, then stopped at the harbor; I missed the valley.",
     "The user plans to visit the canyon, harbor and valley.", "material temporal sequence"),
    ("If you have no subscription, consider a trial; you have not signed up.",
     "The user signed up for the subscription.", "conditional subscription suggestion is not completed signup"),
    ("The deployment did not run; checks must pass before deployment.",
     "The deployment ran successfully before checks.", "negation"),
    ("The Cedar band performs the song Birch; Maple is a podcast.",
     "The Cedar podcast recommended the band Maple and its song Birch.", "entity-category relationships"),
])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_summary_relations_are_checked_before_any_cursor_authority(cfg, granular, source, candidate, policy, verdict):
    client = _SummaryClient(candidate, verdict=verdict)
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        _seed(hy, source)
        result = _extract(hy, client, granular=granular)
        assert len(client.calls) == 3
        primary, verification = client.calls[:2]
        assert replace(verification, system=primary.system, user=primary.user) == primary
        payload = json.loads(verification.user)
        item = payload["summary_item"]
        assert item["candidate_summary"] == item["candidate_raw_summary"] == candidate
        assert resolve_fidelity_sources(payload, item["new_source_ids"])[1]["visible_content"] == source
        assert policy in verification.system
        assert result.failure_reason == "summary_diagnosis_unactionable"
        assert result.failure_stage == "summary_diagnosis" and result.parse_failed
        assert result.summary is result.source_sha256 is result.covered_message_id is None
        assert result.episodes.items == result.procedures.items == []
        assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
def test_final_compaction_only_is_checked_with_prior_continuity_and_primary_items_intact(cfg, granular, compacted):
    final = "Earlier photography and winery topics continued; the assistant supplied conditional build steps."
    rejected = "REJECTED_PRIMARY_OUTPUT_IS_NOT_SOURCE " * 20
    prior = "Earlier photography at North Bridge and two winery recommendations."
    client = _SummaryClient(rejected if compacted else final, compacted=final, with_items=True)
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        last = _seed(hy, "Run checks on the build before deployment; deploy only if they pass.")
        result = _extract(hy, client, granular=granular, prior=prior)
        assert not result.parse_failed and result.covered_message_id == last
        assert result.summary == final
        assert len(client.calls) == (4 if compacted else 3)
        assert sum(call.system == digest._DIGEST_FIDELITY_SYSTEM for call in client.calls) == 1
        payload = json.loads(client.calls[-2].user)
        summary_item = payload["summary_item"]
        assert summary_item["candidate_summary"] == summary_item["candidate_raw_summary"] == final
        assert summary_item["prior_derived_summary"] == prior and not summary_item["candidate_is_noop"]
        assert "REJECTED_PRIMARY_OUTPUT_IS_NOT_SOURCE" not in client.calls[-2].user
        assert prior not in json.dumps(payload["items"] + payload["procedure_items"])
        assert result.episodes.items == client.primary["episodes"]
        assert result.procedures.items == [payload["procedure_items"][0]["candidate"]]
        if compacted:
            assert client.calls[1].user == client.calls[0].user
        assert "Allow faithful umbrella compression" in client.calls[-2].system
        assert "all still-relevant prior topics" in client.calls[-2].system


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_valid_length_compaction_cannot_publish_a_semantically_rejected_summary(cfg, verdict):
    client = _SummaryClient("overlong primary " * 60, compacted="The user asked for conditional build advice.",
                            with_items=True, verdict=verdict)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deploying; deploy only if checks pass.")
        result = _extract(hy, client)
        assert len(client.calls) == 4 and result.parse_failed
        assert result.failure_reason == "summary_diagnosis_unactionable"
        assert result.failure_stage == "summary_diagnosis"
        assert result.episode_rejected_items == result.procedure_rejected_items == 1
        assert result.source_sha256 is result.covered_message_id is result.summary is None


@pytest.mark.parametrize("prior", [None, "Earlier retained topic, byte-for-byte. "])
@pytest.mark.parametrize("candidate", ["", " \n "])
@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
def test_empty_noop_always_gets_source_aware_screening(cfg, prior, candidate, verdict):
    client = _SummaryClient(candidate, verdict=verdict)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        last = _seed(hy, "Acknowledged." if verdict == "supported" else "The build failed and deployment was cancelled.")
        result = _extract(hy, client, prior=prior)
        assert len(client.calls) == 3
        item = json.loads(client.calls[1].user)["summary_item"]
        assert item["candidate_is_noop"]
        assert item["candidate_raw_summary"] == candidate
        assert item["candidate_summary"] == item["prior_derived_summary"] == (prior or "")
        assert result.summary is None
        if verdict == "supported":
            assert not result.parse_failed and result.covered_message_id == last
        else:
            assert result.parse_failed and result.covered_message_id is None
            assert result.failure_reason == "summary_diagnosis_unactionable"


@pytest.mark.parametrize("prior_length", [500, 501])
def test_empty_noop_cannot_silently_truncate_overlong_prior(cfg, prior_length):
    prior = "🧭" * prior_length
    client = _SummaryClient("")
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        last = _seed(hy, "Acknowledged.")
        result = _extract(hy, client, prior=prior)
        if prior_length == 500:
            assert not result.parse_failed and result.covered_message_id == last
            assert json.loads(client.calls[-1].user)["summary_item"]["candidate_summary"] == prior
            assert len(client.calls) == 3
        else:
            assert len(client.calls) == 1 and result.parse_failed
            assert result.failure_reason == "summary_noop_prior_output_cap"
            assert result.failure_stage == "fidelity_verification"
            assert result.covered_message_id is result.source_sha256 is None
            assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)


def test_nonempty_replacement_may_compact_overlong_prior_without_truncating_verifier_input(cfg):
    prior = "Earlier photography and winery topics. " * 20
    client = _SummaryClient("Earlier photography and winery discussions continued with new build advice.")
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = _extract(hy, client, prior=prior)
        assert not result.parse_failed and len(client.calls) == 3
        assert json.loads(client.calls[-2].user)["summary_item"]["prior_derived_summary"] == prior


def test_summary_receives_all_and_only_visible_new_spans_without_expanding_item_citations():
    prefix = "OLD INDEPENDENT FACT; I always wa"
    first = CoveredMessage(7, "s", "user", prefix + "nted to ride.", "a")
    second = CoveredMessage(8, "s", "assistant", "Here are recommendations. UNSEEN_SUFFIX", "b")
    future = CoveredMessage(9, "s", "tool", "OUTSIDE_WINDOW", "c")
    episode = {"title": "Riding wish", "summary": "The user always wanted to ride.",
               "chunk_ids": ["a"], "outcome": "informational", "key_entities": []}
    payload = digest._digest_fidelity_payload(
        [episode], [first, second, future], ["a", "b"],
        before_cursor=(6, 7, len(prefix)), after_cursor=(7, 8, len("Here are recommendations.")),
        leading_context=None, raw_summary="A riding wish received recommendations.",
        published_summary="A riding wish received recommendations.", prior_summary="PRIOR_DERIVED",
    )
    summary_item = payload["summary_item"]
    summary_sources = resolve_fidelity_sources(payload, summary_item["new_source_ids"])
    assert [item["visible_content"] for item in summary_sources] == [
        "nted to ride.", "Here are recommendations.",
    ]
    assert summary_sources[0]["interpretation_only_context"]["content"] == prefix
    assert payload["items"][0]["cited_source_ids"] == ["a"]
    assert "PRIOR_DERIVED" not in json.dumps(payload["items"])
    assert "UNSEEN_SUFFIX" not in json.dumps(payload) and "OUTSIDE_WINDOW" not in json.dumps(payload)


def test_final_raw_wording_is_available_for_separate_format_check(cfg):
    raw = '  "Cache v2.1" was the component Dr. A. described as \'ready\'.  '
    client = _SummaryClient(raw)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, '"Cache v2.1" was the component Dr. A. described as \'ready\'.')
        result = _extract(hy, client)
        assert not result.parse_failed
        item = json.loads(client.calls[-2].user)["summary_item"]
        assert item["candidate_raw_summary"] == raw
        assert item["candidate_summary"] == result.summary == raw.strip()
        assert "not content violations" in client.calls[-2].system
        assert json.loads(client.calls[-1].user)["summary_item"] == {"index": 0, "candidate_summary": raw.strip()}


@pytest.mark.parametrize("error", [RuntimeError("private transport error"), DeadlineExceeded("expired")])
def test_summary_only_verifier_inherits_transport_and_deadline_failure_behavior(cfg, error):
    client = _SummaryClient("The build steps were supplied.", verdict=error)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        if isinstance(error, Exception):
            with pytest.raises(digest.DigestCompletionError) as caught:
                _extract(hy, client)
            assert caught.value.failure_stage == "fidelity_verification"
        else:
            with pytest.raises(DeadlineExceeded) as caught:
                _extract(hy, client)
            assert caught.value is error
        assert len(client.calls) == 2
