"""Independent recovery invariants; scripted verdicts do not measure model accuracy."""
from contextlib import closing
from dataclasses import replace
import json
import re

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline, use_deadline
from hymem.dreaming import digest
from hymem.dreaming.lossless import materialize_message_coverage
from tests.digest_verification_fixtures import synthetic_fidelity_result, synthetic_format_result, synthetic_summary_issues
from tests.test_digest_summary_contract import _config


SOURCE = (
    "We visited Cedar Lake then Oak Hill, but missed Birch Park. "
    "To inspect the cache, first run cache inspect, then run cache verify."
)
PRIOR = "Earlier drawing and gardening topics."
INITIAL = "Visited Cedar Lake and Oak Hill but missed Birch Park; earlier drawing and gardening remain topics."
FIXED = "Visited Cedar Lake then Oak Hill but missed Birch Park; cache inspection steps were supplied, continuing drawing and gardening topics."


class Client:
    def __init__(self, *, compact=False, format_retry=False, initial=INITIAL,
                 repaired=FIXED, first="unsupported", second="supported",
                 first_mutation=None, repair_error=None, diagnosis=None):
        self.compact, self.format_retry = compact, format_retry
        self.initial, self.repaired = initial, repaired
        self.first, self.second = first, second
        self.first_mutation, self.repair_error = first_mutation, repair_error
        self.requests = []
        self.verifications = []
        self.primary = None
        self.repair_requests = []
        self.diagnosis = diagnosis

    def complete(self, request):
        self.requests.append(request)
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            payload = json.loads(request.user)
            self.verifications.append(payload)
            value = synthetic_fidelity_result(len(payload["items"]), len(payload["procedure_items"]))
            value["summary_content"][0]["verdict"] = self.first if len(self.verifications) == 1 else self.second
            if len(self.verifications) == 1 and self.first_mutation:
                self.first_mutation(value)
            return json.dumps(value)
        if request.system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            return json.dumps(self.diagnosis if self.diagnosis is not None else {
                "issues": synthetic_summary_issues(json.loads(request.user)),
            })
        if request.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            return json.dumps(synthetic_format_result(len(self.primary["episodes"])))
        if request.system.startswith("You compact one rolling conversation summary"):
            return json.dumps({"summary": self.initial})
        if self.primary is None:
            ids = re.findall(r"\[chunk ([^\]]+)\]", request.user)
            self.primary = {
                "summary": "TOO_LONG_NOT_EVIDENCE " * 40 if self.compact else self.initial,
                "episodes": [{"title": "Lake then hill visit", "summary": "Visited Cedar Lake then Oak Hill but missed Birch Park.",
                              "outcome": "informational", "key_entities": ["Cedar Lake", "Oak Hill"], "chunk_ids": ids}],
                "procedures": [{"name": "Inspect cache", "description": "Inspect and verify the cache.",
                                "steps": [{"order": 1, "action": "Run cache inspect", "tool": "cache"},
                                          {"order": 2, "action": "Run cache verify", "tool": "cache"}],
                                "triggers": ["inspect cache"], "entities_involved": ["cache"], "chunk_ids": ids}],
            }
            return json.dumps(self.primary)
        self.repair_requests.append(request)
        if self.repair_error is not None:
            raise self.repair_error
        return json.dumps({"summary": self.repaired})


def extract(cfg, client, *, granular=False, prior=PRIOR):
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        last = hy.log_message("root-content-repair", "user", SOURCE)
        hy.close_session("root-content-repair")
        with db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "root-content-repair")
        result = digest.extract_session_digest(
            hy.conn, "root-content-repair", client, max_tokens=2048, max_chars=10000,
            prior_summary=prior, granular=granular, max_episodes=8 if granular else None,
        )
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert tuple(hy.conn.execute(
            "SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='root-content-repair'"
        ).fetchone()) == (None, None)
        return result, last


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("compact,format_retry", [(False, False), (True, False), (False, True), (True, True)])
def test_root_source_linked_repair_preserves_items_and_reverifies(cfg, granular, compact, format_retry):
    client = Client(compact=compact, format_retry=format_retry)
    result, last = extract(cfg, client, granular=granular)
    assert not result.parse_failed
    assert result.summary == FIXED and result.covered_message_id == last
    assert len(client.requests) == 6 + compact
    assert len(client.repair_requests) == 1 and len(client.verifications) == 2
    primary = client.requests[0]
    repair = client.repair_requests[0]
    assert replace(repair, system=primary.system, user=primary.user) == primary
    repair_payload = json.loads(repair.user)
    assert repair_payload["original_generation_input"] == primary.user
    assert repair_payload["rejection_diagnostics"]["candidate_summary"] == INITIAL
    assert SOURCE in repair.user and PRIOR in repair.user
    assert "TOO_LONG_NOT_EVIDENCE" not in repair.user
    first, second = client.verifications
    for key in ("source_catalog", "items", "procedure_items"):
        assert first[key] == second[key]
    for key in ("new_source_ids", "prior_derived_summary"):
        assert first["summary_item"][key] == second["summary_item"][key]
    assert second["summary_item"]["candidate_summary"] == FIXED
    assert second["summary_item"]["candidate_raw_summary"] == FIXED
    assert result.episodes.items == client.primary["episodes"]
    assert result.procedures.items == [x["candidate"] for x in second["procedure_items"]]


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_root_second_content_veto_cannot_trigger_another_repair(cfg, verdict):
    client = Client(second=verdict, format_retry=True)
    result, _ = extract(cfg, client)
    assert result.parse_failed and result.failure_reason == "summary_content_" + verdict
    assert len(client.requests) == 5 and len(client.repair_requests) == 1
    assert result.source_sha256 is result.covered_message_id is result.summary is None
    assert result.episodes.items == result.procedures.items == []


@pytest.mark.parametrize("repaired", [INITIAL, "  " + INITIAL + "\n", "", "x" * 501])
def test_root_unchanged_or_invalid_rewrite_cannot_gain_authority(cfg, repaired):
    client = Client(repaired=repaired)
    result, _ = extract(cfg, client)
    assert result.parse_failed and len(client.requests) == 4
    assert len(client.verifications) == 1
    assert result.failure_stage == ("fidelity_verification" if repaired.strip() == INITIAL
                                    else "summary_content_recovery")
    assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)
    assert result.covered_message_id is result.source_sha256 is None


@pytest.mark.parametrize("family", ["episode_titles", "episode_content", "procedures"])
def test_root_other_item_veto_never_enters_summary_repair(cfg, family):
    def mutate(value):
        value[family][0]["verdict"] = "unsupported"
    client = Client(first_mutation=mutate)
    result, _ = extract(cfg, client)
    assert result.parse_failed and len(client.requests) == 2
    assert client.repair_requests == []


def test_root_malformed_later_format_group_cannot_enter_repair(cfg):
    client = Client(first_mutation=lambda value: value.update(episode_format=[]))
    result, _ = extract(cfg, client)
    assert result.failure_reason == "fidelity_shape_failure" and len(client.requests) == 2


def test_root_rejected_noop_must_recompose_from_source(cfg):
    client = Client(initial="", repaired=PRIOR)
    result, _ = extract(cfg, client)
    assert result.parse_failed and len(client.requests) == 4
    assert client.verifications[0]["summary_item"]["candidate_summary"] == PRIOR
    assert len(client.verifications) == 1


def test_root_deadline_signal_during_content_repair_is_not_swallowed(cfg):
    error = DeadlineExceeded("expired")
    client = Client(repair_error=error)
    with pytest.raises(DeadlineExceeded) as caught:
        extract(cfg, client)
    assert caught.value is error and len(client.requests) == 4


def test_root_supported_summary_keeps_two_call_path(cfg):
    client = Client(first="supported")
    result, _ = extract(cfg, client)
    assert not result.parse_failed and len(client.requests) == 3
    assert result.summary == INITIAL and client.repair_requests == []


@pytest.mark.parametrize("phase", ["before", "after"])
def test_root_content_recovery_does_not_reset_invocation_deadline(cfg, phase):
    client = Client()
    clock = [0.0]
    original = client.complete
    def complete(request):
        response = original(request)
        if (phase == "before" and request.system == digest._DIGEST_FIDELITY_SYSTEM
                or phase == "after" and client.repair_requests):
            clock[0] = 2.0
        return response
    client.complete = complete
    deadline = MonotonicDeadline(1.0, clock=lambda: clock[0])
    with use_deadline(deadline), pytest.raises(DeadlineExceeded):
        extract(cfg, DeadlineBoundLLMClient(client, deadline))
    assert len(client.requests) == (2 if phase == "before" else 4)


@pytest.mark.parametrize("verdict", ["supported", "unsupported"])
def test_root_public_dream_publication_waits_for_replacement_verification(cfg, verdict):
    class PublicClient(Client):
        def complete(self, request):
            systems = (
                digest.SESSION_DIGEST_SYSTEM, digest.SESSION_DIGEST_GRANULAR_SYSTEM,
                digest._DIGEST_FIDELITY_SYSTEM, digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM,
                digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM,
                digest._DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE.format(max_chars=digest.SESSION_SUMMARY_MAX_CHARS),
            )
            if request.system in systems or request.system.startswith("You compact one rolling conversation summary"):
                return super().complete(request)
            return '{"triples":[],"markers":[],"complete":true}'
    client = PublicClient(second=verdict)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        last = hy.log_message("root-public-content-repair", "user", SOURCE, created_at="2020-01-01")
        hy.close_session("root-public-content-repair")
        report = hy.dream()
        state = tuple(hy.conn.execute(
            "SELECT auto_summary,digest_cursor_message_id,digest_retry_count "
            "FROM sessions WHERE id='root-public-content-repair'"
        ).fetchone())
        assert len(client.requests) == (6 if verdict == "supported" else 5)
        if verdict == "supported":
            assert report.digest_failures == 0 and state == (FIXED, last, 0)
            assert hy.conn.execute("SELECT summary FROM episodes").fetchone()[0] == client.primary["episodes"][0]["summary"]
        else:
            assert report.digest_failures == 1 and state == (None, None, 1)
            assert hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []
