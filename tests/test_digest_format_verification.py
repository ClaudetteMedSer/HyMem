"""F4 format/call/publication wiring, not real-provider sentence accuracy.

Format verdicts here are deliberately scripted. Positive controls establish
that application code preserves wording and does not add punctuation heuristics.
"""
from contextlib import closing
from dataclasses import replace
import json
import re

import pytest

from hymem import HyMem
from hymem.dreaming import digest, summary
from tests.digest_verification_fixtures import synthetic_fidelity_result, synthetic_format_result
from tests.test_digest_summary_contract import _config
from tests.test_digest_summary_verification import _extract, _seed


class _FormatClient:
    def __init__(self, candidate, *, summary_verdict="supported", bodies=(),
                 episode_verdicts=(), compacted=None):
        self.candidate = candidate
        self.summary_verdict = summary_verdict
        self.bodies = bodies
        self.episode_verdicts = episode_verdicts
        self.compacted = compacted
        self.calls = []
        self.primary = None

    def complete(self, request):
        self.calls.append(request)
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            payload = json.loads(request.user)
            result = synthetic_fidelity_result(len(payload["items"]), len(payload["procedure_items"]))
            return json.dumps(result)
        if request.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            # Independent scripted rejection establishes wiring, not model
            # accuracy: the one format-only task must still hold negatives.
            payload = json.loads(request.user)
            return json.dumps({
                "summary_format": [{"index": 0, "verdict": self.summary_verdict}],
                "episode_format": [
                    {"index": index, "verdict": self.episode_verdicts[index]
                     if index < len(self.episode_verdicts) else "supported"}
                    for index in range(len(payload["items"]))
                ],
            })
        if request.system.startswith("You compact one rolling conversation summary"):
            return json.dumps({"summary": self.compacted})
        assert request.system.startswith(("You analyze one conversation session",
                                          "You re-read one conversation session"))
        citations = re.findall(r"\[chunk ([^\]]+)\]", request.user)
        self.primary = {
            "summary": self.candidate, "procedures": [],
            "episodes": [{"title": f"Build advice {index}", "summary": body,
                          "outcome": "informational", "key_entities": [], "chunk_ids": citations}
                         for index, body in enumerate(self.bodies)],
        }
        return json.dumps(self.primary)


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("candidate", [
    "Directions were supplied. Earlier topics: photography and wineries.",
    '"Directions were supplied with build advice."',
    "'Directions were supplied with build advice.'",
    "Directions were supplied. Build advice was also supplied.",
    "- Directions were supplied with build advice.",
    "**Directions were supplied with build advice.**",
])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_summary_format_rejection_holds_exact_final_candidate_after_one_adjudication(
    cfg, granular, compacted, candidate, verdict,
):
    client = _FormatClient("REJECTED_PRIMARY " * 40 if compacted else candidate,
                           compacted=candidate if compacted else None, summary_verdict=verdict,
                           bodies=("The assistant supplied build advice.",))
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        _seed(hy, "Directions and build advice were supplied.")
        result = _extract(hy, client, granular=granular)
        assert result.parse_failed and result.failure_reason == "summary_format_" + verdict
        assert result.failure_stage == "format_adjudication"
        assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)
        assert result.summary is result.source_sha256 is result.covered_message_id is None
        assert result.episodes.items == result.procedures.items == []
        assert result.episode_input_items == result.episode_rejected_items == 1
        assert len(client.calls) == (4 if compacted else 3)
        primary, verification = client.calls[0], client.calls[-2]
        assert replace(verification, system=primary.system, user=primary.user) == primary
        assert sum(call.system == digest._DIGEST_FIDELITY_SYSTEM for call in client.calls) == 1
        adjudication = client.calls[-1]
        assert adjudication.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
        assert replace(adjudication, system=primary.system, user=primary.user) == primary
        assert json.loads(adjudication.user)["summary_item"]["candidate_summary"] == candidate
        item = json.loads(verification.user)["summary_item"]
        assert item["candidate_raw_summary"] == item["candidate_summary"] == candidate
        assert "REJECTED_PRIMARY" not in verification.user
        assert tuple(hy.conn.execute(
            "SELECT auto_summary,digest_cursor_message_id FROM sessions WHERE id='summary-verification'",
        ).fetchone()) == (None, None)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("candidate", [
    '  "Cache v2.1" was the named component.  ',
    "  'Cache v2.1' was the named component.  ",
    "Dr. A. B. Chen described version 2.1.4, e.g. its 1.5-second timeout.",
    "Build advice was supplied; deployment remained conditional on passing checks.",
    'The assistant said "run checks" before deployment.',
    "The file settings.py specifies the 1.5-second timeout for v2.1.4.",
])
def test_one_sentence_positives_keep_exact_meaningful_punctuation(cfg, granular, compacted, candidate):
    client = _FormatClient("REJECTED_PRIMARY " * 40 if compacted else candidate,
                           compacted=candidate if compacted else None)
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        last = _seed(hy, candidate.strip())
        result = _extract(hy, client, granular=granular)
        assert not result.parse_failed and result.covered_message_id == last
        assert result.summary == candidate.strip()
        assert len(client.calls) == (4 if compacted else 3)
        item = json.loads(client.calls[-2].user)["summary_item"]
        assert item["candidate_raw_summary"] == candidate
        assert item["candidate_summary"] == result.summary
        assert json.loads(client.calls[-1].user)["summary_item"]["candidate_summary"] == result.summary


@pytest.mark.parametrize("body,verdict", [
    ("The assistant supplied advice.", "supported"),
    ("The assistant supplied advice. The user acknowledged it.", "supported"),
    ('"Cache v2.1" was reviewed by Dr. A. B. Chen. The timeout is 1.5 seconds.', "supported"),
    ("The assistant supplied advice. The user acknowledged it. Checks remain pending.", "unsupported"),
    ("The assistant supplied advice. Earlier topics: build checks.", "unsupported"),
    ("The sentence boundary cannot be determined.", "uncertain"),
])
def test_episode_format_has_its_own_one_to_two_sentence_contract(cfg, body, verdict):
    first = "The first episode has a supported narrative."
    client = _FormatClient("Build advice was supplied and acknowledged.", bodies=(first, body),
                           episode_verdicts=("supported", verdict))
    with closing(HyMem(_config(cfg, True), llm=client)) as hy:
        _seed(hy, first + " " + body)
        result = _extract(hy, client, granular=True)
        assert len(client.calls) == 3
        payload = json.loads(client.calls[1].user)
        assert [item["candidate_body"] for item in payload["items"]] == [first, body]
        if verdict == "supported":
            assert not result.parse_failed
            assert result.episodes.items == client.primary["episodes"]
        else:
            adjudication = json.loads(client.calls[-1].user)
            assert [item["candidate_body"] for item in adjudication["items"]] == [first, body]
            assert result.parse_failed and result.failure_reason == "episode_format_" + verdict
            assert result.episode_rejected_items == 2
            assert result.episodes.items == [] and result.covered_message_id is None


@pytest.mark.parametrize("prior,verdict", [
    (None, "supported"),
    ("Earlier build advice was supplied. ", "supported"),
    ('"Earlier build advice was supplied."', "unsupported"),
    ("Earlier advice was supplied. Earlier topics: checks.", "unsupported"),
])
def test_empty_noop_format_checks_effective_prior_or_permitted_absence(cfg, prior, verdict):
    client = _FormatClient(" \n ", summary_verdict=verdict)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        last = _seed(hy, "Acknowledged.")
        result = _extract(hy, client, prior=prior)
        assert len(client.calls) == 3
        item = json.loads(client.calls[1].user)["summary_item"]
        assert item["candidate_is_noop"] and item["candidate_raw_summary"] == " \n "
        assert item["candidate_summary"] == (prior or "")
        assert result.summary is None
        if verdict == "supported":
            assert not result.parse_failed and result.covered_message_id == last
        else:
            assert json.loads(client.calls[-1].user)["summary_item"]["candidate_summary"] == prior
            assert result.parse_failed and result.failure_reason == "summary_format_unsupported"
            assert result.covered_message_id is result.source_sha256 is None


def test_format_schema_does_not_hide_malformed_semantics_or_conflate_failures():
    result = synthetic_fidelity_result(1)
    assert digest._validate_digest_fidelity_response(json.dumps(result), 1) is None
    result["summary_content"][0]["verdict"] = "unsupported"
    assert digest._validate_digest_fidelity_response(json.dumps(result), 1) == "summary_content_unsupported"
    result.update(synthetic_format_result(1))
    assert digest._validate_digest_fidelity_response(json.dumps(result), 1) == "fidelity_shape_failure"
    assert digest._validate_digest_format_adjudication_response(json.dumps(result), 1) == "format_adjudication_shape_failure"


def test_digest_structural_normalization_preserves_quotes_without_changing_legacy_cleaner():
    candidate = '  "Cache v2.1" was the named component.  '
    result = digest._validate_digest_response(None, {
        "summary": candidate, "episodes": [], "procedures": [],
    }, "s", [], granular=False, max_episodes=None)
    assert not result.parse_failed and result.summary == candidate.strip()
    # Standalone normalization is deliberately untouched; only digest publication
    # has stopped using this destructively normalized value.
    assert summary.clean_summary(candidate) == 'Cache v2.1" was the named component.'


def test_approved_meaningful_leading_quote_survives_actual_digest_publication(cfg):
    from tests.test_summary import _seed_session, _summary_llm

    candidate = '  "Cache v2.1" was the named component reviewed by Dr. A. B. Chen.  '
    client = _summary_llm(candidate)
    with closing(HyMem(cfg, llm=client)) as hy:
        _seed_session(hy, "quoted-component", [("assistant", candidate.strip())])
        hy.dream()
        row = hy.conn.execute(
            "SELECT summary,auto_summary,digest_cursor_message_id FROM sessions WHERE id='quoted-component'",
        ).fetchone()
        assert row["summary"] == row["auto_summary"] == candidate.strip()
        assert row["digest_cursor_message_id"] is not None
        requests = [call for call in client.calls if call.system == digest._DIGEST_FIDELITY_SYSTEM]
        assert len(requests) == 1
        assert json.loads(requests[0].user)["summary_item"]["candidate_summary"] == candidate.strip()


def test_format_policy_separates_contracts_without_punctuation_repairs():
    semantic_policy = digest._DIGEST_FIDELITY_SYSTEM
    assert "exactly four keys" in semantic_policy
    assert "separate format stage and are not content violations" in semantic_policy
    assert "summary_format" not in semantic_policy and "episode_format" not in semantic_policy
    assert "shortest exact quotations" not in semantic_policy
    assert "shortest exact quotations" in digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM
    policy = digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
    for required in (
        "exactly two keys", "summary_format", "episode_format", "one complete sentence",
        "one or two complete sentences", "no Markdown", "no enclosing quotation marks",
        "semicolons", "fragment", "quoted component name", "empty effective summary",
        "never count or split on punctuation", "Abbreviations", "initials", "decimals", "version numbers",
        "Preserve all punctuation", "Do not judge factual support",
    ):
        assert required in policy
