"""Root-owned adversarial checks of the separated verifier experiment.

All source text/verdicts here are invented. These tests establish control flow
and source binding, not the factual accuracy of a real model.
"""
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
import json
import re

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline, current_deadline
from hymem.dreaming import digest
from hymem.dreaming.lossless import materialize_message_coverage
from tests.test_digest_summary_contract import _config


SOURCE = (
    "We visited Pine Ridge then Alder Bay, but did not visit Cedar Field. "
    "Before departure we checked the latch. To inspect the gate, first run "
    "gate status, then run gate verify."
)
PRIOR = "Earlier watercolor and orchard topics."
INITIAL = "We visited Pine Ridge and Alder Bay but not Cedar Field; earlier watercolor and orchard topics continued."
FIXED = "We visited Pine Ridge then Alder Bay but not Cedar Field; latch checks preceded departure, gate inspection steps were supplied, and earlier watercolor and orchard topics continued."


def decisions(payload, summary="supported"):
    result = {family: [{"index": item["index"], "verdict": "supported"} for item in payload[key]]
              for family, key in (("episode_titles", "items"), ("episode_content", "items"),
                                  ("procedures", "procedure_items"))}
    result["summary_content"] = [{"index": 0, "verdict": summary}]
    return result


def valid_issues(payload):
    source = next(row for row in payload["source_catalog"] if "visited Pine Ridge then Alder Bay" in row["visible_content"])
    return [{"code": "temporal_order", "candidate_quote": "Pine Ridge and Alder Bay",
             "sources": [{"kind": "new_source", "source_id": source["chunk_id"],
                          "quote": "visited Pine Ridge then Alder Bay"}]}]


class RootClient:
    def __init__(self, *, compact=False, first="unsupported", second="supported", repair=FIXED,
                 decision_mutation=None, diagnosis_mutation=None, raw_diagnosis=None,
                 exception_stage=None, clock=None, format_verdict="supported"):
        self.compact, self.first, self.second, self.repair = compact, first, second, repair
        self.decision_mutation, self.diagnosis_mutation = decision_mutation, diagnosis_mutation
        self.raw_diagnosis, self.exception_stage, self.clock = raw_diagnosis, exception_stage, clock
        self.format_verdict = format_verdict
        self.requests, self.roles, self.deadlines, self.verifiers = [], [], [], []
        self.primary = None

    def complete(self, request):
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            role = "fidelity_reverification" if self.verifiers else "fidelity_verification"
        elif request.system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            role = "summary_diagnosis"
        elif request.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            role = "format_adjudication"
        elif request.system.startswith("You compact one rolling conversation summary"):
            role = "summary_compaction"
        elif request.system.startswith("You repair one rolling conversation summary"):
            role = "summary_content_recovery"
        else:
            role = "primary"
        self.requests.append(request)
        self.roles.append(role)
        self.deadlines.append(current_deadline())
        if self.clock is not None:
            self.clock[0] += 1.0
        if role == self.exception_stage:
            raise RuntimeError("synthetic provider error: do not log raw evidence")
        if role.startswith("fidelity_"):
            payload = json.loads(request.user)
            self.verifiers.append(payload)
            result = decisions(payload, self.first if len(self.verifiers) == 1 else self.second)
            if self.decision_mutation:
                self.decision_mutation(result, len(self.verifiers))
            return json.dumps(result)
        if role == "summary_diagnosis":
            if self.raw_diagnosis is not None:
                return self.raw_diagnosis
            result = {"issues": valid_issues(self.verifiers[0])}
            if self.diagnosis_mutation:
                self.diagnosis_mutation(result)
            return json.dumps(result)
        if role == "format_adjudication":
            payload = json.loads(request.user)
            return json.dumps({"summary_format": [{"index": 0, "verdict": self.format_verdict}],
                               "episode_format": [{"index": item["index"], "verdict": "supported"}
                                                  for item in payload["items"]]})
        if role == "summary_compaction":
            return json.dumps({"summary": INITIAL})
        if role == "summary_content_recovery":
            return json.dumps({"summary": self.repair})
        ids = re.findall(r"\[chunk ([^\]]+)\]", request.user)
        assert ids and self.primary is None
        self.primary = {
            "summary": "OVERSIZED_REJECTED_GENERATION " * 30 if self.compact else INITIAL,
            "episodes": [{"title": "Ridge then bay visit", "summary": "We visited Pine Ridge then Alder Bay but not Cedar Field.",
                          "outcome": "informational", "key_entities": ["Pine Ridge", "Alder Bay"], "chunk_ids": ids}],
            "procedures": [{"name": "Inspect gate", "description": "Inspect and verify the gate.",
                            "steps": [{"order": 1, "action": "Run gate status", "tool": "gate"},
                                      {"order": 2, "action": "Run gate verify", "tool": "gate"}],
                            "triggers": ["inspect the gate"], "entities_involved": ["gate"], "chunk_ids": ids}],
        }
        return json.dumps(self.primary)


def extract(cfg, client, *, granular=False, deadline=None):
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        hy.log_message("root-separated", "user", SOURCE)
        hy.close_session("root-separated")
        with db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "root-separated")
        before = tuple(hy.conn.execute("SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='root-separated'").fetchone())
        try:
            return digest.extract_session_digest(
                hy.conn, "root-separated", DeadlineBoundLLMClient(client, deadline) if deadline else client,
                max_tokens=2048, max_chars=10000, prior_summary=PRIOR, granular=granular,
                max_episodes=8 if granular else None,
            )
        finally:
            assert tuple(hy.conn.execute("SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='root-separated'").fetchone()) == before
            assert hy.conn.execute("SELECT count(*) FROM digest_staging").fetchone()[0] == 0


def assert_held(result):
    assert result.parse_failed
    assert result.summary is result.source_sha256 is result.covered_message_id is None
    assert result.episodes.items == result.procedures.items == []
    assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_root_exact_evidence_survives_diagnosis_and_one_repair(cfg, granular, compact):
    client = RootClient(compact=compact)
    result = extract(cfg, client, granular=granular)
    assert not result.parse_failed and result.summary == FIXED
    expected = ["primary", *(["summary_compaction"] if compact else []), "fidelity_verification",
                "summary_diagnosis", "summary_content_recovery", "fidelity_reverification", "format_adjudication"]
    assert client.roles == expected
    initial, final = client.verifiers
    diagnosis = json.loads(client.requests[client.roles.index("summary_diagnosis")].user)
    assert diagnosis == {**initial, "schema": digest.DIGEST_SUMMARY_DIAGNOSIS_VERSION}
    for field in ("source_catalog", "items", "procedure_items"):
        assert initial[field] == final[field]
    for field in ("new_source_ids", "prior_derived_summary"):
        assert initial["summary_item"][field] == final["summary_item"][field]
    recovery = json.loads(client.requests[client.roles.index("summary_content_recovery")].user)
    assert recovery["original_generation_input"] == client.requests[0].user
    assert recovery["rejection_diagnostics"]["candidate_summary"] == INITIAL
    assert recovery["rejection_diagnostics"]["issues"] == valid_issues(initial)
    assert "OVERSIZED_REJECTED_GENERATION" not in json.dumps(recovery)
    for request in client.requests:
        assert replace(request, user=client.requests[0].user, system=client.requests[0].system) == client.requests[0]
    assert result.episodes.items == client.primary["episodes"]
    assert result.procedures.items == [p["candidate"] for p in final["procedure_items"]]


def test_root_supported_candidate_never_pays_for_diagnosis(cfg):
    client = RootClient(first="supported")
    result = extract(cfg, client)
    assert not result.parse_failed
    assert client.roles == ["primary", "fidelity_verification", "format_adjudication"]


@pytest.mark.parametrize("family", ["episode_titles", "episode_content", "procedures"])
@pytest.mark.parametrize("veto", ["unsupported", "uncertain"])
def test_root_item_veto_is_terminal_even_with_summary_veto(cfg, family, veto):
    def mutate(result, number):
        result[family][0]["verdict"] = veto
    client = RootClient(decision_mutation=mutate)
    assert_held(extract(cfg, client))
    assert client.roles == ["primary", "fidelity_verification"]


@pytest.mark.parametrize("damage", ["extra_verdict", "extra_summary", "empty", "wrong_quote", "wrong_candidate",
                                    "unknown_source", "wrong_prior_kind", "duplicate", "context_authority"])
def test_root_bad_diagnosis_cannot_start_repair_or_publish(cfg, damage):
    def mutate(value):
        issue = value["issues"][0]
        ref = issue["sources"][0]
        if damage == "extra_verdict": value["verdict"] = "supported"
        elif damage == "extra_summary": value["summary"] = FIXED
        elif damage == "empty": value["issues"] = []
        elif damage == "wrong_quote": ref["quote"] = "visited Alder Bay then Pine Ridge"
        elif damage == "wrong_candidate": issue["candidate_quote"] = "We visited Cedar Field"
        elif damage == "unknown_source": ref["source_id"] = "uncited"
        elif damage == "wrong_prior_kind": issue["sources"] = [{"kind": "prior_derived_summary", "quote": PRIOR}]
        elif damage == "duplicate": value["issues"].append(deepcopy(issue))
        elif damage == "context_authority": ref["kind"] = "interpretation_only_context"
    client = RootClient(diagnosis_mutation=mutate)
    result = extract(cfg, client)
    assert_held(result)
    assert result.failure_reason.startswith("summary_diagnosis_")
    assert result.failure_stage == "summary_diagnosis"
    assert client.roles == ["primary", "fidelity_verification", "summary_diagnosis"]


@pytest.mark.parametrize("raw", ['{"issues":[]', '{"issues":[],"issues":[]}', '{"issues":NaN}', '{}', '[]', 'null'])
def test_root_diagnosis_does_not_inherit_verdict_salvage(cfg, raw):
    client = RootClient(raw_diagnosis=raw)
    assert_held(extract(cfg, client))
    assert len(client.requests) == 3


@pytest.mark.parametrize("field", ["issues", "rationale", "summary_format"])
def test_root_old_inline_or_unknown_decision_fields_never_gain_authority(cfg, field):
    def mutate(value, number):
        if field == "summary_format": value[field] = [{"index": 0, "verdict": "supported"}]
        else: value["summary_content"][0][field] = []
    client = RootClient(first="supported", decision_mutation=mutate)
    result = extract(cfg, client)
    assert_held(result)
    assert result.failure_reason == "fidelity_shape_failure"
    assert len(client.requests) == 2


@pytest.mark.parametrize("repair", [INITIAL, "  " + INITIAL + "\n", "", "x" * 501])
def test_root_invalid_or_unchanged_repair_never_rerolls(cfg, repair):
    client = RootClient(repair=repair)
    assert_held(extract(cfg, client))
    assert client.roles == ["primary", "fidelity_verification", "summary_diagnosis", "summary_content_recovery"]


@pytest.mark.parametrize("second", ["unsupported", "uncertain"])
def test_root_final_content_veto_cannot_trigger_new_diagnosis(cfg, second):
    client = RootClient(second=second)
    assert_held(extract(cfg, client))
    assert client.roles.count("summary_diagnosis") == client.roles.count("summary_content_recovery") == 1
    assert client.roles[-1] == "fidelity_reverification"


def test_root_final_format_veto_still_blocks_complete_pipeline(cfg):
    client = RootClient(format_verdict="unsupported")
    result = extract(cfg, client)
    assert_held(result)
    assert result.failure_reason == "summary_format_unsupported"
    assert len(client.requests) == 6


@pytest.mark.parametrize("stage", ["summary_diagnosis", "summary_content_recovery", "fidelity_reverification", "format_adjudication"])
def test_root_nested_exception_preserves_deepest_stage(cfg, stage):
    client = RootClient(exception_stage=stage)
    with pytest.raises(digest.DigestCompletionError) as caught:
        extract(cfg, client)
    assert caught.value.failure_stage == stage
    assert client.roles[-1] == stage


@pytest.mark.parametrize("expires", [1, 2, 3, 4, 5, 6, 7, 8])
def test_root_seven_calls_share_one_absolute_deadline(cfg, expires):
    clock = [0.0]
    deadline = MonotonicDeadline(float(expires), clock=lambda: clock[0])
    client = RootClient(compact=True, clock=clock)
    if expires <= 7:
        with pytest.raises(DeadlineExceeded):
            extract(cfg, client, deadline=deadline)
    else:
        assert not extract(cfg, client, deadline=deadline).parse_failed
    assert len(client.requests) == min(expires, 7)
    assert all(value is deadline for value in client.deadlines)


def test_root_diagnosis_input_cap_is_not_permission_to_shrink_source(cfg, monkeypatch):
    monkeypatch.setattr(digest, "_DIGEST_SUMMARY_DIAGNOSIS_MAX_INPUT_CHARS", 1)
    client = RootClient()
    result = extract(cfg, client)
    assert_held(result)
    assert result.failure_reason == "summary_diagnosis_input_cap"
    assert result.failure_stage == "summary_diagnosis"
    assert client.roles == ["primary", "fidelity_verification"]


@pytest.mark.parametrize("name", ["DIGEST_SUMMARY_DIAGNOSIS_VERSION", "_DIGEST_SUMMARY_DIAGNOSIS_SYSTEM",
                                  "_DIGEST_SUMMARY_DIAGNOSIS_MAX_INPUT_CHARS", "_DIGEST_SUMMARY_DIAGNOSIS_MAX_OUTPUT_CHARS"])
def test_root_new_diagnosis_policy_changes_digest_identity_not_phase1(monkeypatch, name):
    from hymem.dreaming.semantic_generation import semantic_generation_suffix
    from hymem.extraction.contract import extraction_contract_identity
    from hymem.extraction.llm import StubLLMClient
    client = StubLLMClient(default="[]")
    before = semantic_generation_suffix("digest", client)
    phase1 = extraction_contract_identity("v20")
    original = getattr(digest, name)
    monkeypatch.setattr(digest, name, original + 1 if isinstance(original, int) else original + " changed")
    assert semantic_generation_suffix("digest", client) != before
    assert extraction_contract_identity("v20") == phase1


@pytest.mark.parametrize("name", ["_digest_summary_diagnosis_payload", "_validate_digest_summary_diagnosis_response"])
def test_root_new_diagnosis_dispatch_is_bound_into_generation(monkeypatch, name):
    from hymem.dreaming.semantic_generation import semantic_generation_suffix
    from hymem.extraction.llm import StubLLMClient
    client = StubLLMClient(default="[]")
    before = semantic_generation_suffix("digest", client)
    original = getattr(digest, name)

    def altered(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(digest, name, altered)
    assert semantic_generation_suffix("digest", client) != before
