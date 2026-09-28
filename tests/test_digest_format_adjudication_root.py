"""Root-owned adversarial controls; scripted verdicts are not provider accuracy.

These tests deliberately do not call the implementation agent's response helpers.
"""
from contextlib import closing
from dataclasses import replace
import json

import pytest

from hymem import HyMem
from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline, use_deadline
from hymem.dreaming import digest
from hymem.dreaming.lossless import covered_messages_after
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import StubLLMClient
from tests.test_digest_summary_contract import _config, _extract, _seed


TEXT = (
    "On a recent trip the user drove to Big Sur and Monterey but did not explore "
    "the Santa Ynez Valley, and now wants to explore the countryside on horseback; "
    "the assistant recommended Santa Ynez Valley stables and scenic Highway 101/246 "
    "routing from Santa Barbara to Solvang, continuing earlier topics of Big Sur "
    "and Bixby Bridge photography, Santa Barbara County wineries, and Solvang sights."
)
BODY = 'Dr. A. B. Chen reviewed "Cache v2.1"; deployment remained conditional.'


class Client:
    def __init__(self, candidate, *, compact=False, veto=None, adjudicated="supported", clock=None):
        self.candidate, self.compact, self.veto = candidate, compact, veto
        self.adjudicated, self.clock = adjudicated, clock
        self.calls = []

    def complete(self, request):
        self.calls.append(request)
        if request.system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            return '{"issues":[]}'
        if request.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            if self.clock:
                self.clock[0] = 2.0
            if isinstance(self.adjudicated, BaseException):
                raise self.adjudicated
            return json.dumps({
                "summary_format": [{"index": 0, "verdict": self.adjudicated}],
                "episode_format": [{"index": 0, "verdict": "supported"}],
            })
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            result = {
                k: [{"index": 0, "verdict": "supported"}]
                for k in ("episode_titles", "episode_content", "summary_content")
            }
            result["procedures"] = []
            if self.veto:
                family, verdict = self.veto
                result[family][0]["verdict"] = verdict
            return json.dumps(result)
        if request.system.startswith(("You compact one rolling conversation summary",
                                      "You recompose one rolling conversation summary")):
            # Unchanged content remains vetoed; this is not a success fixture.
            return json.dumps({"summary": self.candidate["summary"]})
        return json.dumps({**self.candidate, "summary": "REJECTED_PRIMARY " * 40} if self.compact else self.candidate)


def setup(hy, *, summary=TEXT):
    last = _seed(hy, "Exact new source; no source-only marker belongs in format input.")
    sources = covered_messages_after(hy.conn, "bounded-summary", None)
    return last, {
        "summary": summary, "procedures": [],
        "episodes": [{"title": "PRIVATE_TITLE_ONLY", "summary": BODY, "outcome": "informational",
                      "key_entities": ["PRIVATE_ENTITY_ONLY"], "chunk_ids": [sources[0].chunk_id]}],
    }


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_root_captured_format_rejection_recovers_without_source_or_text_changes(cfg, granular, compact):
    with closing(HyMem(_config(cfg, granular))) as hy:
        last, candidate = setup(hy)
        client = Client(candidate, compact=compact)
        result = _extract(hy, client, granular=granular, prior_summary="PRIVATE_PRIOR_ONLY")
        assert not result.parse_failed and result.summary == TEXT
        assert result.episodes.items == candidate["episodes"] and result.covered_message_id == last
        assert result.source_sha256 and result.caught_up
        assert len(client.calls) == (4 if compact else 3)
        primary, adjudication = client.calls[0], client.calls[-1]
        assert replace(adjudication, system=primary.system, user=primary.user) == primary
        payload = json.loads(adjudication.user)
        assert payload == {
            "schema": digest.DIGEST_FORMAT_ADJUDICATION_VERSION,
            "summary_item": {"index": 0, "candidate_summary": TEXT},
            "items": [{"index": 0, "candidate_body": BODY}],
        }
        for forbidden in ("PRIVATE_PRIOR_ONLY", "PRIVATE_TITLE_ONLY", "PRIVATE_ENTITY_ONLY", "source-only marker", "REJECTED_PRIMARY", "source_catalog", "unsupported"):
            assert forbidden not in adjudication.user
        assert tuple(hy.conn.execute(
            "SELECT auto_summary,digest_cursor_message_id FROM sessions WHERE id='bounded-summary'",
        ).fetchone()) == (None, None), "extraction alone cannot publish"


@pytest.mark.parametrize("family", ["episode_titles", "episode_content", "summary_content"])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_root_semantic_veto_cannot_buy_a_format_retry(cfg, family, verdict):
    with closing(HyMem(_config(cfg, False))) as hy:
        _, candidate = setup(hy)
        client = Client(candidate, veto=(family, verdict))
        result = _extract(hy, client, granular=False)
        assert result.parse_failed and len(client.calls) == (3 if family == "summary_content" else 2)
        assert not any(call.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM for call in client.calls)
        assert result.summary is result.source_sha256 is result.covered_message_id is None
        assert result.episodes.items == []


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain", "invalid", True, None])
def test_root_unapproved_adjudication_never_publishes_or_loops(cfg, verdict):
    with closing(HyMem(_config(cfg, False))) as hy:
        _, candidate = setup(hy)
        client = Client(candidate, adjudicated=verdict)
        result = _extract(hy, client, granular=False)
        assert result.parse_failed and len(client.calls) == 3
        assert result.summary is result.source_sha256 is result.covered_message_id is None
        assert result.episodes.items == []
        assert result.failure_stage == "format_adjudication"
        assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_root_adjudication_does_not_reset_invocation_deadline(cfg, phase):
    clock = [0.0]
    with closing(HyMem(_config(cfg, False))) as hy:
        _, candidate = setup(hy)
        client = Client(candidate, clock=clock if phase == "after" else None)
        original = client.complete
        def complete(request):
            result = original(request)
            if phase == "before" and request.system == digest._DIGEST_FIDELITY_SYSTEM:
                clock[0] = 2.0
            return result
        client.complete = complete
        deadline = MonotonicDeadline(1.0, clock=lambda: clock[0])
        # The public dreaming runner supplies this proxy for custom clients;
        # the raw extraction helper does not itself install a client deadline.
        with use_deadline(deadline):
            with pytest.raises(DeadlineExceeded):
                _extract(hy, DeadlineBoundLLMClient(client, deadline), granular=False)
        assert len(client.calls) == (2 if phase == "before" else 3)
        assert tuple(hy.conn.execute(
            "SELECT auto_summary,digest_cursor_message_id FROM sessions WHERE id='bounded-summary'",
        ).fetchone()) == (None, None)


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(9)])
def test_root_adjudication_propagates_control_signals(cfg, exc):
    with closing(HyMem(_config(cfg, False))) as hy:
        _, candidate = setup(hy)
        client = Client(candidate, adjudicated=exc)
        with pytest.raises(type(exc)):
            _extract(hy, client, granular=False)
        assert len(client.calls) == 3


@pytest.mark.parametrize("name,value", [
    ("DIGEST_FORMAT_ADJUDICATION_VERSION", "changed-policy"),
    ("_DIGEST_FORMAT_ADJUDICATION_SYSTEM", "changed-instructions"),
    ("_DIGEST_FORMAT_ADJUDICATION_MAX_INPUT_CHARS", 123),
    ("_DIGEST_FORMAT_ADJUDICATION_MAX_OUTPUT_CHARS", 123),
])
def test_root_new_format_policy_is_bound_to_digest_identity_only(monkeypatch, name, value):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    monkeypatch.setattr(digest, name, value)
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert after["digest"] != before["digest"]
    assert after["facts"] == before["facts"] and after["profile"] == before["profile"]
    assert extraction_contract_identity("v20") == phase1


@pytest.mark.parametrize("verdict", ["supported", "unsupported"])
def test_root_public_dream_publishes_only_after_successful_adjudication(cfg, verdict):
    client = Client({}, adjudicated=verdict)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        last, candidate = setup(hy)
        client.candidate = candidate
        report = hy.dream()
        row = hy.conn.execute(
            "SELECT auto_summary,digest_cursor_message_id,digest_retry_count FROM sessions WHERE id='bounded-summary'",
        ).fetchone()
        if verdict == "supported":
            assert report.digest_failures == 0
            assert tuple(row) == (TEXT, last, 0)
            assert hy.conn.execute("SELECT summary FROM episodes").fetchone()[0] == BODY
        else:
            assert report.digest_failures == 1
            assert tuple(row) == (None, None, 1)
            assert hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []
