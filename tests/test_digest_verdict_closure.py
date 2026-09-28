"""Synthetic closing-envelope controls; these do not prove model accuracy."""
from contextlib import closing
from copy import deepcopy
import json

import pytest

from hymem import HyMem
from hymem.dreaming import digest
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.jsonio import loads_exact_or_fenced
from hymem.extraction.llm import StubLLMClient
from tests.digest_verification_fixtures import synthetic_fidelity_result, synthetic_format_result
from tests.test_digest_summary_content_recovery import _ContentRecoveryClient, REJECTED, REPAIRED
from tests.test_digest_summary_contract import _config
from tests.test_digest_summary_verification import _extract, _seed


def _raw(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _packet():
    return {
        "source_catalog": [{"chunk_id": "source-1", "visible_content": "Checks passed. The rollback did not run."}],
        "summary_item": {
            "candidate_summary": "The checks passed.", "prior_derived_summary": "Earlier context.",
            "new_source_ids": ["source-1"],
        },
    }


def _issue():
    return {"code": "omitted_outcome", "candidate_quote": "", "sources": [{
        "kind": "new_source", "source_id": "source-1", "quote": "The rollback did not run.",
    }]}


@pytest.mark.parametrize("trailer", ["}", "]}"])
@pytest.mark.parametrize("envelope", ["exact", "fenced", "bare_fence"])
@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
def test_only_outer_closure_is_recovered_without_changing_verdict(trailer, envelope, verdict):
    value = synthetic_fidelity_result(2)
    value["summary_content"][0]["verdict"] = verdict
    full = _raw(value)
    assert full.endswith(trailer)
    raw = full[:-len(trailer)]
    if envelope != "exact":
        raw = "```" + ("json" if envelope == "fenced" else "") + "\n" + raw + "\n```"
    assert loads_exact_or_fenced(raw) is None
    reason = None if verdict == "supported" else "summary_content_" + verdict
    assert digest._validate_digest_fidelity_response(raw, 2, payload=_packet()) == reason


@pytest.mark.parametrize("family", ["summary_format", "episode_format"])
@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
@pytest.mark.parametrize("trailer", ["}", "]}"])
def test_format_verdicts_use_same_narrow_closure_gate(family, verdict, trailer):
    value = synthetic_format_result(2)
    value[family][0]["verdict"] = verdict
    raw = _raw(value)[:-len(trailer)]
    reason = None if verdict == "supported" else family + "_" + verdict
    assert digest._validate_digest_format_adjudication_response(raw, 2) == reason


@pytest.mark.parametrize("damage", [
    "missing_family", "missing_index", "duplicate_index", "extra_index", "bool_index",
    "missing_verdict", "unknown_verdict", "additional_family", "string_array",
    "forged_quote", "unknown_source", "unrecognized_issue", "duplicate_issue",
])
def test_recovered_json_still_requires_exhaustive_schema_and_bound_diagnostics(damage):
    value = synthetic_fidelity_result(2)
    value["summary_content"][0]["verdict"] = "unsupported"
    if damage in {"forged_quote", "unknown_source", "unrecognized_issue", "duplicate_issue"}:
        value["summary_content"][0]["issues"] = [_issue()]
    if damage == "missing_family": del value["procedures"]
    elif damage == "missing_index": value["episode_titles"].pop()
    elif damage == "duplicate_index": value["episode_content"][1]["index"] = 0
    elif damage == "extra_index": value["episode_titles"].append({"index": 2, "verdict": "supported"})
    elif damage == "bool_index": value["episode_titles"][0]["index"] = False
    elif damage == "missing_verdict": del value["episode_titles"][0]["verdict"]
    elif damage == "unknown_verdict": value["episode_titles"][0]["verdict"] = "approved"
    elif damage == "additional_family": value["extra"] = []
    elif damage == "string_array": value["procedures"] = "[]"
    elif damage == "forged_quote": value["summary_content"][0]["issues"][0]["sources"][0]["quote"] = "invented"
    elif damage == "unknown_source": value["summary_content"][0]["issues"][0]["sources"][0]["source_id"] = "other"
    elif damage == "unrecognized_issue": value["summary_content"][0]["issues"][0]["code"] = "grammar"
    elif damage == "duplicate_issue": value["summary_content"][0]["issues"].append(deepcopy(_issue()))
    # Removing only the object close keeps this test independent of key order.
    reason = digest._validate_digest_fidelity_response(_raw(value)[:-1], 2, payload=_packet())
    assert reason in {"fidelity_shape_failure", "fidelity_diagnostics_failure", "fidelity_parse_failure"}


@pytest.mark.parametrize("raw", [
    "", None, {}, '[{"index":0,"verdict":"supported"}',
    '{"a":[{"index":0,"verdict":"supported"',  # Missing item close, not just envelope.
    '{"a":[{"index":0,"verdict":"supported',  # Truncated quoted token.
    '{"a":[{"verdict":"supported","index":0',  # Scalar EOF is ambiguous.
    '{"a":[{"index":0,"verdict":"supported"},',
    '{"a":', '{"a":[', '{"a":[{', '{"a":tru', '{"a":1e',
    '{"a":[{"index":0,"verdict":"supported"}] garbage',
    '{"a":[{"index":0,"verdict":"supported"}]} trailing',
    'Refusal: {"a":[{"index":0,"verdict":"supported"}',
    '{"a":[{"index":0,"verdict":"supported"}] } {',
    '{"a":[{"index":0,"verdict":"supported"}] ]',
    '{"a":[{"index":0,"verdict":"supported"},]}',
    '{"a":[],"a":[]', '{"a":[NaN]', '{"a":[Infinity]', '{"a":[1e999]',
    '```json\n{"a":[{"index":0,"verdict":"supported"}',
    '```json\n{"a":[{"index":0,"verdict":"supported"}\n``',
    '```python\n{"a":[{"index":0,"verdict":"supported"}\n```',
    '```json\n{"a":[{"index":0,"verdict":"supported"}\n``` prose',
])
def test_ambiguous_tokens_items_envelopes_or_invalid_json_never_recover(raw):
    assert digest._loads_digest_verdict(raw, maximum=65_536) is None


def test_escaped_quotes_braces_and_unicode_are_data_not_container_structure():
    packet = _packet()
    quote = 'The \\ "quoted" } ] brace text: café 🧭.'
    packet["source_catalog"][0]["visible_content"] = quote
    issue = _issue()
    issue["sources"][0]["quote"] = quote
    raw = _raw({"issues": [issue]})
    assert digest._validate_digest_summary_diagnosis_response(raw, packet) == ([issue], None)
    assert digest._validate_digest_summary_diagnosis_response(raw[:-2], packet)[1] == "summary_diagnosis_parse_failure"


def test_closing_recovery_has_strict_output_and_depth_bounds(monkeypatch):
    raw = _raw(synthetic_fidelity_result())[:-2]
    assert digest._loads_digest_verdict(raw, maximum=len(raw) + 2) is not None
    assert digest._loads_digest_verdict(raw, maximum=len(raw) + 1) is None
    monkeypatch.setattr(digest, "_DIGEST_VERDICT_MAX_RECOVERY_DEPTH", 2)
    assert digest._loads_digest_verdict(raw, maximum=65_536) is None
    assert digest._loads_digest_verdict('{"a":' + '[' * 20 + '{}' + ']' * 19, maximum=65_536) is None
    def recursion_limit(_raw):
        raise RecursionError("synthetic decoder limit")
    monkeypatch.setattr(digest, "loads_exact_or_fenced", recursion_limit)
    assert digest._loads_digest_verdict(raw, maximum=65_536) is None


class _ClosureClient(_ContentRecoveryClient):
    def complete(self, request):
        raw = super().complete(request)
        if request.system in {digest._DIGEST_FIDELITY_SYSTEM, digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM}:
            assert raw.endswith("]}")
            return raw[:-2]
        return raw


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
@pytest.mark.parametrize("compact", [False, True])
def test_closed_rejection_retains_one_repair_full_reverification_and_final_format(cfg, verdict, compact):
    client = _ClosureClient(
        "oversized " * 100 if compact else REJECTED,
        compacted=REJECTED if compact else None,
        first=(("summary_content", 0, verdict),),
    )
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        last = _seed(hy, "The canyon came before the harbor; the valley was missed.")
        result = _extract(hy, client)
    assert not result.parse_failed and result.summary == REPAIRED and result.covered_message_id == last
    assert len(client.calls) == (7 if compact else 6) and client.verifications == 2
    assert sum(call.system.startswith("You repair one rolling conversation summary") for call in client.calls) == 1
    assert client.calls[-1].system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
    verification = [json.loads(call.user) for call in client.calls if call.system == digest._DIGEST_FIDELITY_SYSTEM]
    for key in ("source_catalog", "items", "procedure_items"):
        assert verification[0][key] == verification[1][key]
    repair = next(call for call in client.calls if call.system.startswith("You repair one rolling conversation summary"))
    assert json.loads(repair.user)["original_generation_input"] == client.calls[0].user


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_recovered_final_semantic_veto_cannot_repair_twice_or_reach_format(cfg, verdict):
    client = _ClosureClient(second=(("summary_content", 0, verdict),))
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "The canyon came before the harbor; the valley was missed.")
        result = _extract(hy, client)
    assert result.failure_reason == "summary_content_" + verdict and len(client.calls) == 5
    assert result.summary is result.covered_message_id is result.source_sha256 is None
    assert not any(call.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM for call in client.calls)


def test_primary_generation_and_summary_repairs_remain_strict(cfg):
    class PrimaryTail(_ContentRecoveryClient):
        def complete(self, request):
            return super().complete(request)[:-1]
    client = PrimaryTail()
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = _extract(hy, client)
    assert result.parse_failed and result.failure_stage == "primary" and len(client.calls) == 1
    assert result.summary is result.covered_message_id is result.source_sha256 is None
    assert digest._validate_digest_summary_repair('{"summary":"A complete candidate sentence."')[0] is None
    assert loads_exact_or_fenced(_raw(synthetic_fidelity_result())[:-2]) is None


@pytest.mark.parametrize("name,value", [
    ("_DIGEST_VERDICT_MAX_RECOVERY_DEPTH", 15),
    ("_DIGEST_VERDICT_MAX_CLOSING_TRAILER", 1),
])
def test_new_local_policy_changes_digest_identity_only(monkeypatch, name, value):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    monkeypatch.setattr(digest, name, value)
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert after["digest"] != before["digest"]
    assert after["facts"] == before["facts"] and after["profile"] == before["profile"]
    assert extraction_contract_identity("v20") == phase1
