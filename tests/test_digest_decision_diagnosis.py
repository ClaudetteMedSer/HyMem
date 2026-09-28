"""Offline candidate identity/transport controls; not model accuracy evidence."""
import json

import pytest

from benchmarks import episode_probe
from hymem.dreaming import digest
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import LLMRequest, StubLLMClient
from tests.test_digest_targeted_summary_issues import _issue, _packet


@pytest.mark.parametrize("field", [
    "DIGEST_FIDELITY_VERIFICATION_VERSION", "DIGEST_SUMMARY_DIAGNOSIS_VERSION",
    "_DIGEST_SUMMARY_DIAGNOSIS_SYSTEM", "_DIGEST_SUMMARY_DIAGNOSIS_MAX_INPUT_CHARS",
    "_DIGEST_SUMMARY_DIAGNOSIS_MAX_OUTPUT_CHARS",
])
def test_new_decision_and_diagnosis_policy_invalidate_only_digest_identity(monkeypatch, field):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    extraction = extraction_contract_identity("v20")
    monkeypatch.setattr(digest, field, "synthetic-policy-change")
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert before["digest"] != after["digest"]
    assert before["facts"] == after["facts"] and before["profile"] == after["profile"]
    assert extraction_contract_identity("v20") == extraction


def test_decision_and_diagnosis_share_unchanged_evidence_rules_but_not_wire_schema():
    rules = digest._DIGEST_FIDELITY_EVIDENCE_RULES
    assert rules in digest._DIGEST_FIDELITY_SYSTEM
    assert rules in digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM
    assert "candidate_quote" not in digest._DIGEST_FIDELITY_SYSTEM
    assert "exactly one key, issues" in digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM
    packet = _packet()
    diagnosis = digest._digest_summary_diagnosis_payload(packet)
    assert diagnosis == {**packet, "schema": digest.DIGEST_SUMMARY_DIAGNOSIS_VERSION}


@pytest.mark.parametrize("envelope", ["exact", "fenced"])
def test_diagnosis_output_bound_counts_entire_envelope(monkeypatch, envelope):
    raw = json.dumps({"issues": [_issue()]})
    if envelope == "fenced":
        raw = "```json\n" + raw + "\n```"
    monkeypatch.setattr(digest, "_DIGEST_SUMMARY_DIAGNOSIS_MAX_OUTPUT_CHARS", len(raw))
    assert digest._validate_digest_summary_diagnosis_response(raw, _packet()) == ([_issue()], None)
    monkeypatch.setattr(digest, "_DIGEST_SUMMARY_DIAGNOSIS_MAX_OUTPUT_CHARS", len(raw) - 1)
    assert digest._validate_digest_summary_diagnosis_response(raw, _packet()) == (None, "summary_diagnosis_output_cap")


def test_probe_reverification_attribution_resets_at_each_new_primary():
    capture = episode_probe.CapturingLLM(lambda system, user: "{}")
    systems = [digest.SESSION_DIGEST_SYSTEM, digest._DIGEST_FIDELITY_SYSTEM,
               digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM, digest._DIGEST_FIDELITY_SYSTEM,
               digest.SESSION_DIGEST_SYSTEM, digest._DIGEST_FIDELITY_SYSTEM]
    for system in systems:
        capture.complete(LLMRequest(system=system, user="Synthetic transport-only packet"))
    assert [record["stage"] for record in capture.sent] == [
        "primary", "fidelity_verification", "summary_diagnosis", "fidelity_reverification",
        "primary", "fidelity_verification",
    ]
