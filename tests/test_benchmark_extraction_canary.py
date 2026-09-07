from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks import extraction_canary as extraction_canary_module
from benchmarks.extraction_canary import (
    EXTRACTION_CANARY_EXPECTED_PREPARTITION_LEAVES,
    EXTRACTION_CANARY_FENCED_CODE_CONTROL_SHA256,
    EXTRACTION_CANARY_FIXTURE_SHA256,
    EXTRACTION_CANARY_FIXTURE_VERSION,
    EXTRACTION_CANARY_LIST_CONTROL_SHA256,
    EXTRACTION_CANARY_MAX_COMPLETION_CALLS,
    EXTRACTION_CANARY_MAX_PROVIDER_ATTEMPTS,
    EXTRACTION_CANARY_NORMAL_PASS_COMPLETION_CALLS,
    EXTRACTION_CANARY_PROMPT_VERSION,
    EXTRACTION_CANARY_PROSE_BOUNDARY_CLAIM_SHA256,
    EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID,
    EXTRACTION_CANARY_SOURCE_CONTENT_CHARS,
    EXTRACTION_CANARY_SOURCE_MESSAGE_ID,
    EXTRACTION_CANARY_SOURCE_MESSAGE_IDS,
    EXTRACTION_CANARY_STRUCTURAL_PROBE_VERSION,
    EXTRACTION_CANARY_TABLE_CONTINUATION_CLAIM_SHA256,
    EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID,
    EXTRACTION_CANARY_USAGE_ACCOUNTING,
    EXTRACTION_CANARY_VERSION,
    ExtractionCanaryError,
    extraction_canary_client_policy,
    extraction_canary_policy,
    run_configured_extraction_canary,
    run_extraction_canary,
    skipped_extraction_canary,
    validate_extraction_canary_report,
)
from hymem import HyMemConfig
from benchmarks.strictness import (
    BenchmarkCleanupError,
    BenchmarkIntegrityError,
    content_hash,
)
from hymem.extraction.chunk import CLEAN_EMPTY_RECOVERY_POLICY_VERSION
from hymem.extraction.chunk import MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK
from hymem.extraction.chunk import SOURCE_RECORD_SPLIT_POLICY_VERSION
from hymem.extraction.contract import extraction_contract_binding
from hymem.extraction.llm import LLMRequest, ProviderAttemptTracker


@pytest.fixture(autouse=True)
def _public_llm_deployment_attestations(monkeypatch):
    """Keep custom fixture routes exact without introducing credentials."""

    monkeypatch.setenv(
        "HYMEM_LLM_DEPLOYMENT_REVISION", "fixture-release-2026-09"
    )
    monkeypatch.setenv("HYMEM_LLM_DEPLOYMENT_TENANT", "fixture-tenant")


_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(_BENCH))
import locomo_adapter as locomo  # noqa: E402
import longmemeval_adapter as lme  # noqa: E402
import msc_adapter as msc  # noqa: E402
import beam_adapter as beam  # noqa: E402


_FORGED_CONTRACT_ID = "hymem-extraction-contract-sha256-v1:" + "0" * 64


def _manifest_contract_tamper(original, *, valid_alternate: bool = False):
    """Return an internally rehashed manifest that lies about live code."""

    def build(*args, **kwargs):
        manifest = original(*args, **kwargs)
        config = manifest["config"]
        if valid_alternate:
            # This alternate binding is perfectly valid against the executing
            # code and internally coherent with its canary/config.  Producers
            # must still reject it because the actual adapter was built with
            # the previously captured v20 runtime config.
            config["effective_hymem_config"]["prompt_version"] = "v21"
            config["effective_hymem_config"]["extraction_contract"] = (
                extraction_contract_binding("v21")
            )
            config["extraction_canary"] = extraction_canary_policy(
                prompt_version="v21"
            )
        else:
            config["effective_hymem_config"]["extraction_contract"][
                "identity"
            ] = _FORGED_CONTRACT_ID
            config["extraction_canary"]["extraction_contract"][
                "identity"
            ] = _FORGED_CONTRACT_ID
        manifest["config_hash"] = content_hash(config)
        manifest["run_id"] = content_hash({
            key: value for key, value in manifest.items() if key != "run_id"
        })
        return manifest

    return build


def _complete(
    triples: list[dict] | None = None,
    *,
    markers: list[dict] | None = None,
) -> str:
    return json.dumps({
        "triples": triples or [], "markers": markers or [], "complete": True,
    })


def _preference_claim(**overrides) -> dict:
    item = {
        "subject": "Avery Boundary Canary",
        "subject_type": "person",
        "predicate": "prefers",
        "object": "PostgreSQL",
        "object_type": "database",
        "polarity": 1,
        "source_message_id": EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID,
    }
    item.update(overrides)
    return item


def _deployment_claim(**overrides) -> dict:
    item = {
        "subject": "HyMem Canary Relay",
        "subject_type": "service",
        "predicate": "deploys_to",
        "object": "Fly.io",
        "object_type": "platform",
        "polarity": 1,
        "source_message_id": EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID,
    }
    item.update(overrides)
    return item


def _claims() -> list[dict]:
    return [_deployment_claim(), _preference_claim()]


class _Client:
    def __init__(
        self, responses: list[str] | None = None, *, default: str | None = None,
        router=None,
    ):
        self.responses = list(responses or [])
        self.default = default
        self.router = router
        self.requests: list[LLMRequest] = []
        self.model = "canary-model"
        self.base_url = "https://provider.example/v1"
        self.thinking_mode = "auto"
        self.effective_extra_body = {}
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.cost_usd = None
        self.token_usage_available = True

    def complete(self, request: LLMRequest) -> str:
        self.requests.append(request)
        self.call_count += 1
        self.request_attempts += 1
        self.successful_responses += 1
        self.prompt_tokens += 10
        self.completion_tokens += 5
        self.total_tokens += 15
        if self.responses:
            return self.responses.pop(0)
        if self.router is not None:
            return self.router(request)
        if self.default is not None:
            return self.default
        raise AssertionError("unexpected canary completion")


def _payloads_in_request(request: LLMRequest) -> list[dict]:
    if "OMISSION VERIFICATION PASS" in request.system:
        prefix = (
            "Excerpt (the exact same source records as the primary pass):\n"
            '"""\n'
        )
    else:
        prefix = 'Excerpt:\n"""\n'
    suffix = '\n"""\n\n'
    assert request.user.startswith(prefix)
    end = request.user.index(suffix, len(prefix))
    return [
        json.loads(encoded)
        for encoded in request.user[len(prefix):end].splitlines()
    ]


def _claims_in_leaf(request: LLMRequest, *, variants: bool = False) -> list[dict]:
    claims = []
    for payload in _payloads_in_request(request):
        content = payload["content"]
        if (
            extraction_canary_module._TABLE_CLAIM_ROW in content
            and payload.get("source_fragment_context")
            == extraction_canary_module._EXPECTED_TABLE_CONTEXT
        ):
            claims.append(_deployment_claim(**(
                {"subject": "HYMEM_canary---RELAY", "object": "Fly IO"}
                if variants else {}
            )))
        if (
            extraction_canary_module._PROSE_BOUNDARY_RIGHT in content
            and payload.get("source_boundary_context")
            == extraction_canary_module._EXPECTED_PROSE_BOUNDARY_CONTEXT
        ):
            claims.append(_preference_claim(**(
                {"subject": "AVERY_boundary---CANARY", "object": "Postgre-SQL"}
                if variants else {}
            )))
    return claims


def _representative_response(
    request: LLMRequest, *, variants: bool = False,
) -> str:
    if "OMISSION VERIFICATION PASS" in request.system:
        return _complete()
    return _complete(_claims_in_leaf(request, variants=variants))


def _representative_client(*, variants: bool = False) -> _Client:
    return _Client(router=lambda request: _representative_response(
        request, variants=variants,
    ))


def test_canary_uses_real_phase1_contract_and_requires_exact_claim_fields():
    client = _representative_client()

    report = run_extraction_canary(client)

    assert report["status"] == "passed"
    assert report["matched_supported_claims"] == 2
    assert report["missing_expected_claim_indexes"] == []
    assert report["completion_calls"] == EXTRACTION_CANARY_NORMAL_PASS_COMPLETION_CALLS
    assert report["provider_attempts"] == 8
    assert report["initial_prepartition_leaves"] == 4
    assert report["usage_accounting"] == EXTRACTION_CANARY_USAGE_ACCOUNTING
    assert report["store_writes"] == 0
    assert report["claim_evidence"][0] == {
        "expected_claim_index": 0,
        **report["expected_claims"][0],
    }
    assert report["claim_evidence"][0]["subject"] == "HyMem Canary Relay"
    assert report["claim_evidence"][1]["subject"] == "Avery Boundary Canary"
    assert all(
        evidence[field_name] is None
        for evidence in report["claim_evidence"]
        for field_name in (
            "value_text", "value_numeric", "value_unit", "temporal_scope",
        )
    )
    assert all(
        evidence[field_name] == {}
        for evidence in report["claim_evidence"]
        for field_name in ("subject_properties", "object_properties")
    )
    table_request = client.requests[2]
    prose_request = client.requests[6]
    controls_request = client.requests[4]
    first_request = client.requests[0]
    assert "source_message_id (integer)" in first_request.system
    assert {payload["source_message_id"] for request in client.requests
            for payload in _payloads_in_request(request)} == set(
                EXTRACTION_CANARY_SOURCE_MESSAGE_IDS
            )
    table_payload = _payloads_in_request(table_request)[0]
    assert extraction_canary_module._TABLE_CLAIM_ROW in table_payload["content"]
    assert extraction_canary_module._TABLE_HEADER_CONTEXT not in table_payload["content"]
    assert extraction_canary_module._TABLE_PRELUDE not in table_payload["content"]
    assert table_payload["source_fragment_context"] == (
        extraction_canary_module._EXPECTED_TABLE_CONTEXT
    )
    prose_payload = _payloads_in_request(prose_request)[0]
    assert extraction_canary_module._PROSE_BOUNDARY_RIGHT in prose_payload["content"]
    assert extraction_canary_module._PROSE_BOUNDARY_LEFT not in prose_payload["content"]
    assert prose_payload["source_boundary_context"] == (
        extraction_canary_module._EXPECTED_PROSE_BOUNDARY_CONTEXT
    )
    controls_payload = _payloads_in_request(controls_request)[0]
    assert extraction_canary_module._LIST_CONTROL in controls_payload["content"]
    assert extraction_canary_module._FENCED_CODE_CONTROL in controls_payload["content"]
    assert report["execution_path"] == {
        "primary_requests": 4,
        "empty_verification_requests": 2,
        "omission_verification_requests": 2,
        "parsed_source_records": 8,
        "source_record_parse_failures": 0,
        "source_message_ids_seen": list(EXTRACTION_CANARY_SOURCE_MESSAGE_IDS),
        "table_claim_requests": 2,
        "table_claim_exact_context_requests": 2,
        "table_claim_self_contained_requests": 0,
        "table_claim_exact_context_emissions": 1,
        "table_claim_wrong_context_emissions": 0,
        "prose_claim_requests": 2,
        "prose_claim_exact_context_requests": 2,
        "prose_claim_self_contained_requests": 0,
        "prose_claim_exact_context_emissions": 1,
        "prose_claim_wrong_context_emissions": 0,
        "list_control_atomic_requests": 2,
        "fenced_code_control_atomic_requests": 2,
        "protected_control_split_boundaries": 0,
        "list_control_probe_atomic": True,
        "fenced_code_control_probe_atomic": True,
    }
    assert "EMPTY VERIFICATION PASS" in client.requests[1].system
    assert "OMISSION VERIFICATION PASS" in client.requests[3].system
    assert "EMPTY VERIFICATION PASS" in client.requests[5].system
    assert "OMISSION VERIFICATION PASS" in client.requests[7].system


def test_canary_rejects_punctuation_or_case_changed_claim_entities():
    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_representative_client(variants=True))

    assert caught.value.report["missing_expected_claim_indexes"] == [0, 1]
    assert caught.value.report["claim_evidence"] == []


def test_request_recorder_preserves_scoped_provider_attempt_accounting():
    class ScopedClient(_Client):
        def __init__(self):
            super().__init__(router=_representative_response)
            self.active_tracker = None

        @contextmanager
        def track_provider_attempts(self):
            tracker = ProviderAttemptTracker()
            self.active_tracker = tracker
            try:
                yield tracker
            finally:
                self.active_tracker = None

        def complete(self, request: LLMRequest) -> str:
            assert self.active_tracker is not None
            self.active_tracker._record()
            self.active_tracker._record()
            response = super().complete(request)
            # Keep cumulative usage coherent with the two scoped attempts.
            self.request_attempts += 1
            return response

    client = ScopedClient()
    report = run_extraction_canary(client)

    assert report["completion_calls"] == 8
    assert report["provider_attempts"] == 16
    assert report["usage"]["request_attempts"] == 16
    assert report["execution_path"]["parsed_source_records"] == 8


def test_canary_rejects_clean_empty_after_phase1_verification_pass():
    client = _Client(default=_complete())

    with pytest.raises(ExtractionCanaryError, match="did not return exactly") as caught:
        run_extraction_canary(client)

    assert caught.value.report["failure_reason"] == "clean_empty"
    assert 8 <= caught.value.report["completion_calls"] <= 24
    assert len(client.requests) == caught.value.report["completion_calls"]
    assert any("VERIFICATION PASS" in request.system for request in client.requests)


def test_canary_accepts_context_claim_recovered_by_empty_verification():
    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        claims = _claims_in_leaf(request)
        if (
            "EMPTY VERIFICATION PASS" not in request.system
            and claims and claims[0]["predicate"] == "deploys_to"
        ):
            return _complete()
        return _complete(claims)

    client = _Client(router=route)

    report = run_extraction_canary(client)

    assert report["status"] == "passed"
    assert report["completion_calls"] == 9
    assert any("EMPTY VERIFICATION PASS" in r.system for r in client.requests)
    assert sum("OMISSION VERIFICATION PASS" in r.system for r in client.requests) == 2
    assert report["execution_path"]["table_claim_requests"] == 3


@pytest.mark.parametrize(
    ("primary", "missing_index"),
    [
        ([_deployment_claim()], 1),
        ([_preference_claim()], 0),
    ],
)
def test_canary_rejects_when_either_expected_claim_is_missing(
    primary: list[dict], missing_index: int,
):
    selected = primary[0]

    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        available = _claims_in_leaf(request)
        return _complete([
            claim for claim in available
            if claim["predicate"] == selected["predicate"]
        ])

    client = _Client(router=route)

    with pytest.raises(ExtractionCanaryError, match="did not return exactly") as caught:
        run_extraction_canary(client)

    assert caught.value.report["failure_reason"] == "supported_claim_evidence_missing"
    assert caught.value.report["matched_supported_claims"] == 1
    assert caught.value.report["missing_expected_claim_indexes"] == [missing_index]
    assert caught.value.report["completion_calls"] >= 8


def test_canary_prepartition_requires_both_labelled_context_paths():
    client = _representative_client()

    report = run_extraction_canary(client)

    assert report["status"] == "passed"
    assert report["matched_supported_claims"] == 2
    assert report["completion_calls"] == 8
    primary_requests = [
        request for request in client.requests
        if "VERIFICATION PASS" not in request.system
    ]
    assert len(primary_requests) == 4
    assert [_claims_in_leaf(request) for request in primary_requests] == [
        [], [_deployment_claim()], [], [_preference_claim()],
    ]


def test_canary_rejects_claims_emitted_from_context_free_source_leaves():
    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        claims = []
        for payload in _payloads_in_request(request):
            content = payload["content"]
            if (
                payload["source_message_id"]
                == EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID
                and extraction_canary_module._TABLE_PRELUDE in content
                and extraction_canary_module._TABLE_CLAIM_ROW not in content
            ):
                claims.append(_deployment_claim())
            if (
                payload["source_message_id"]
                == EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID
                and extraction_canary_module._PROSE_BOUNDARY_LEFT in content
                and extraction_canary_module._PROSE_BOUNDARY_RIGHT not in content
            ):
                claims.append(_preference_claim())
        return _complete(claims)

    with pytest.raises(BenchmarkIntegrityError, match="exact context paths"):
        run_extraction_canary(_Client(router=route))


def test_representative_canary_fails_if_table_context_is_removed(monkeypatch):
    monkeypatch.setattr(
        extraction_canary_module.chunk_extraction,
        "_canonical_table_context_for_cut",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_representative_client())

    assert caught.value.report["failure_reason"] == "resource_limit"
    assert caught.value.report["completion_calls"] == 0
    assert caught.value.report["execution_path"]["table_claim_requests"] == 0


@pytest.mark.parametrize("forged_field", ["header", "prelude"])
def test_representative_canary_fails_if_table_context_is_forged(
    monkeypatch, forged_field: str,
):
    original = (
        extraction_canary_module.chunk_extraction
        ._canonical_table_context_for_cut
    )

    def forged(*args, **kwargs):
        context = original(*args, **kwargs)
        if context is None or context.get("kind") != (
            "introduced_canonical_markdown_table_header"
        ):
            return context
        context = dict(context)
        if forged_field == "header":
            context["content"] = context["content"].replace(
                "| service |", "| subject |", 1,
            )
        else:
            context["prelude_content"] = context["prelude_content"].replace(
                "Service", "Archive", 1,
            )
        return context

    monkeypatch.setattr(
        extraction_canary_module.chunk_extraction,
        "_canonical_table_context_for_cut",
        forged,
    )

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_representative_client())

    path = caught.value.report["execution_path"]
    assert path["table_claim_requests"] >= 2
    assert path["table_claim_exact_context_requests"] == 0
    assert caught.value.report["missing_expected_claim_indexes"] == [0]


def test_representative_canary_fails_if_prose_context_is_removed(monkeypatch):
    monkeypatch.setattr(
        extraction_canary_module.chunk_extraction,
        "_source_boundary_context_for_cut",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_representative_client())

    assert caught.value.report["failure_reason"] == "resource_limit"
    assert caught.value.report["completion_calls"] == 0
    assert caught.value.report["execution_path"]["prose_claim_requests"] == 0


def test_representative_canary_fails_if_prose_context_is_forged(monkeypatch):
    original = (
        extraction_canary_module.chunk_extraction
        ._source_boundary_context_for_cut
    )

    def forged(*args, **kwargs):
        context = original(*args, **kwargs)
        if context is None:
            return None
        context = dict(context)
        context["content"] = "X" + context["content"][1:]
        return context

    monkeypatch.setattr(
        extraction_canary_module.chunk_extraction,
        "_source_boundary_context_for_cut",
        forged,
    )

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_representative_client())

    path = caught.value.report["execution_path"]
    assert path["prose_claim_requests"] >= 2
    assert path["prose_claim_exact_context_requests"] == 0
    assert caught.value.report["missing_expected_claim_indexes"] == [1]


def test_structural_subprobes_exercise_exact_list_and_fence_atoms():
    report = run_extraction_canary(_representative_client())

    assert report["execution_path"]["list_control_probe_atomic"] is True
    assert report["execution_path"]["fenced_code_control_probe_atomic"] is True
    assert report["execution_path"]["protected_control_split_boundaries"] == 0
    assert report["execution_path"]["list_control_atomic_requests"] == 2
    assert report["execution_path"]["fenced_code_control_atomic_requests"] == 2


@pytest.mark.parametrize("protection", ["list", "fence"])
def test_representative_canary_rejects_removed_markdown_block_protection(
    monkeypatch, protection: str,
):
    target = "_list_blocks" if protection == "list" else "_fenced_code_spans"
    monkeypatch.setattr(
        extraction_canary_module.chunk_extraction,
        target,
        lambda *_args, **_kwargs: (),
    )

    with pytest.raises(BenchmarkIntegrityError, match="exact context paths"):
        run_extraction_canary(_representative_client())


def _valid_extra_claim() -> dict:
    return {
        "subject": "HyMem Canary Relay",
        "subject_type": "service",
        "predicate": "uses",
        "object": "Redis",
        "object_type": "database",
        "polarity": 1,
        "source_message_id": EXTRACTION_CANARY_SOURCE_MESSAGE_ID,
    }


def test_canary_rejects_expected_claims_plus_an_extra_valid_triple():
    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        claims = _claims_in_leaf(request)
        if "HyMem Canary Relay" in request.user:
            claims.append(_valid_extra_claim())
        return _complete(claims)

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_Client(router=route))

    assert caught.value.report["failure_reason"] == "unexpected_canary_output"
    assert caught.value.report["valid_triples_returned"] == 3
    assert caught.value.report["valid_markers_returned"] == 0


def test_canary_rejects_expected_claims_plus_any_valid_marker():
    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        markers = (
            [{"kind": "style", "statement": "Use compact output."}]
            if "HyMem Canary Relay" in request.user else []
        )
        return _complete(_claims_in_leaf(request), markers=markers)

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_Client(router=route))

    assert caught.value.report["failure_reason"] == "unexpected_canary_output"
    assert caught.value.report["valid_triples_returned"] == 2
    assert caught.value.report["valid_markers_returned"] == 1


@pytest.mark.parametrize(
    ("field_name", "hallucinated_value", "private_tokens"),
    [
        ("value_text", "private-version-label", ("private-version-label",)),
        ("value_numeric", 7342.125, ("7342.125",)),
        ("value_unit", "private-unit", ("private-unit",)),
        (
            "temporal_scope", "during-private-migration",
            ("during-private-migration",),
        ),
        (
            "subject_properties",
            {"private-subject-key": "private-subject-value"},
            ("private-subject-key", "private-subject-value"),
        ),
        (
            "object_properties",
            {"private-object-key": "private-object-value"},
            ("private-object-key", "private-object-value"),
        ),
    ],
    ids=[
        "value-text", "value-numeric", "value-unit", "temporal-scope",
        "subject-properties", "object-properties",
    ],
)
def test_canary_rejects_each_unsupported_persisted_optional_field(
    field_name: str, hallucinated_value: object, private_tokens: tuple[str, ...],
):
    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        claims = _claims_in_leaf(request)
        if claims and claims[0]["predicate"] == "prefers":
            claims[0][field_name] = hallucinated_value
        return _complete(claims)

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_Client(router=route))

    report = caught.value.report
    assert report["failure_reason"] == "unexpected_canary_output"
    assert report["valid_triples_returned"] == 2
    serialized = json.dumps(report, sort_keys=True)
    assert all(token not in serialized for token in private_tokens)
    assert any(field_name in detail for detail in report["failure_details"])


def test_canary_rejects_combined_optional_metadata_without_persisting_values():
    secret_values = {
        "value_text": "private-build-label",
        "value_numeric": 7342,
        "value_unit": "private-unit",
        "temporal_scope": "during-private-migration",
        "subject_properties": {"private-subject-key": "private-subject-value"},
        "object_properties": {"private-object-key": "private-object-value"},
    }

    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        claims = _claims_in_leaf(request)
        if claims and claims[0]["predicate"] == "prefers":
            claims[0].update(secret_values)
        return _complete(claims)

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_Client(router=route))

    report = caught.value.report
    assert report["failure_reason"] == "unexpected_canary_output"
    assert report["matched_supported_claims"] == 1
    assert report["missing_expected_claim_indexes"] == [1]
    assert len(report["failure_details"]) == 7
    serialized = json.dumps(report, sort_keys=True)
    for value in (
        "private-build-label", "private-unit", "during-private-migration",
        "private-subject-key", "private-subject-value", "private-object-key",
        "private-object-value", "7342",
    ):
        assert value not in serialized


def test_canary_rejects_a_schema_valid_duplicate_even_when_phase1_coalesces_it():
    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        claims = _claims_in_leaf(request)
        if any(claim["predicate"] == "prefers" for claim in claims):
            claims.append(_preference_claim())
        return _complete(claims)

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_Client(router=route))

    assert caught.value.report["failure_reason"] == "unexpected_canary_output"
    assert caught.value.report["valid_triples_returned"] == 2
    assert caught.value.report["duplicate_triples_collapsed"] == 1


@pytest.mark.parametrize(
    "first_response",
    [
        '{"triples":[',
        _complete([
            {
                "subject": f"Saturation Subject {index}",
                "predicate": "uses",
                "object": f"Saturation Object {index}",
                "polarity": 1,
                "source_message_id": EXTRACTION_CANARY_SOURCE_MESSAGE_ID,
            }
            for index in range(24)
        ]),
    ],
    ids=["truncated-json", "triple-cap"],
)
def test_dense_canary_recovers_truncation_or_saturation_without_partial_publish(
    first_response: str,
):
    first = True

    def route(request: LLMRequest) -> str:
        nonlocal first
        if first:
            first = False
            return first_response
        return _representative_response(request)

    client = _Client(router=route)
    report = run_extraction_canary(client)

    assert report["status"] == "passed"
    assert report["valid_triples_returned"] == 2
    assert report["duplicate_triples_collapsed"] == 0
    assert 8 < report["completion_calls"] <= EXTRACTION_CANARY_MAX_COMPLETION_CALLS
    assert report["provider_attempts"] <= EXTRACTION_CANARY_MAX_PROVIDER_ATTEMPTS


def test_dense_canary_failure_remains_inside_its_smaller_call_cap():
    client = _Client(default='{"triples":[')

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(client)

    report = caught.value.report
    assert report["status"] == "failed"
    assert report["valid_triples_returned"] == 0
    assert report["valid_markers_returned"] == 0
    assert report["claim_evidence"] == []
    assert 0 < report["completion_calls"] <= EXTRACTION_CANARY_MAX_COMPLETION_CALLS
    assert report["provider_attempts"] == report["completion_calls"]
    assert len(client.requests) == report["completion_calls"]
    assert EXTRACTION_CANARY_MAX_COMPLETION_CALLS < 96
    assert MAX_EXTRACTION_COMPLETION_CALLS_PER_CHUNK == 96
    assert EXTRACTION_CANARY_MAX_PROVIDER_ATTEMPTS == 3 * (
        EXTRACTION_CANARY_MAX_COMPLETION_CALLS
    )


def test_partial_branch_failure_report_is_atomic_and_cannot_be_forged_to_pass():
    def route(request: LLMRequest) -> str:
        payloads = _payloads_in_request(request)
        if any(
            extraction_canary_module._PROSE_BOUNDARY_RIGHT
            in payload["content"]
            for payload in payloads
        ):
            return '{"triples":['
        return _representative_response(request)

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_Client(router=route))

    report = caught.value.report
    assert report["status"] == "failed"
    assert report["failure_reason"] == "branch_incomplete"
    assert report["valid_triples_returned"] == 1
    assert report["claim_evidence"] == []
    assert report["missing_expected_claim_indexes"] == [0, 1]
    assert validate_extraction_canary_report(
        report, expected_mode="failed",
    )["status"] == "failed"

    forged = deepcopy(report)
    forged["status"] = "passed"
    forged.pop("failure_reason")
    forged.pop("failure_details")
    with pytest.raises(BenchmarkIntegrityError, match="extraction canary.*exact"):
        validate_extraction_canary_report(forged, expected_mode="required")


def test_v17_validator_rejects_provider_attempts_above_declared_cap():
    report = _passed_report()
    report["provider_attempts"] = EXTRACTION_CANARY_MAX_PROVIDER_ATTEMPTS + 1
    report["usage"]["request_attempts"] = report["provider_attempts"]

    with pytest.raises(BenchmarkIntegrityError, match="attempt counters"):
        _validate_passed_report(report)


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("not-json", "parse_failure"),
        (_complete([_preference_claim(predicate="invented_predicate")]),
         "item_validation_failure"),
    ],
)
def test_canary_rejects_parse_and_item_validation_failures(response: str, reason: str):
    client = _Client(default=response)

    with pytest.raises(ExtractionCanaryError, match="canary failed") as caught:
        run_extraction_canary(client)

    assert caught.value.report["failure_reason"] in {reason, "branch_incomplete"}
    assert "raw" not in caught.value.report


def test_canary_rejects_valid_but_different_claim():
    def route(request: LLMRequest) -> str:
        if "OMISSION VERIFICATION PASS" in request.system:
            return _complete()
        claims = _claims_in_leaf(request)
        return _complete([
            (
                _preference_claim(predicate="uses", object="SQLite")
                if claim["predicate"] == "prefers" else claim
            )
            for claim in claims
        ])

    client = _Client(router=route)

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(client)

    assert caught.value.report["failure_reason"] == "unexpected_canary_output"
    assert caught.value.report["valid_triples_returned"] == 2


def test_canary_rejects_expected_claims_attributed_to_wrong_source():
    foreign_source_id = EXTRACTION_CANARY_SOURCE_MESSAGE_ID + 99
    client = _Client(default=_complete([
        _preference_claim(source_message_id=foreign_source_id),
        _deployment_claim(source_message_id=foreign_source_id),
    ]))

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(client)

    assert caught.value.report["status"] == "failed"
    assert caught.value.report["failure_reason"] in {
        "response_conflict", "branch_incomplete",
    }


def test_canary_does_not_touch_an_existing_store(tmp_path: Path):
    db_path = tmp_path / "hymem.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel VALUES ('unchanged')")
    before = db_path.read_bytes()

    run_extraction_canary(_representative_client())

    assert db_path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [db_path]


def test_canary_prompt_provenance_tracks_benchmark_hymem_default(tmp_path: Path):
    assert EXTRACTION_CANARY_VERSION == "hymem-phase1-extraction-canary-v17"
    assert EXTRACTION_CANARY_PROMPT_VERSION == "v20"
    assert EXTRACTION_CANARY_FIXTURE_VERSION == (
        "hymem-phase1-context-paths-four-leaf-v7"
    )
    assert EXTRACTION_CANARY_PROMPT_VERSION == HyMemConfig(
        root=tmp_path
    ).prompt_version
    policy = extraction_canary_policy()
    assert policy["prompt_version"] == (
        EXTRACTION_CANARY_PROMPT_VERSION
    )
    assert policy["fixture_version"] == EXTRACTION_CANARY_FIXTURE_VERSION
    assert policy["fixture_sha256"] == EXTRACTION_CANARY_FIXTURE_SHA256
    assert policy["source_content_chars"] == EXTRACTION_CANARY_SOURCE_CONTENT_CHARS
    assert policy["source_split_policy_version"] == SOURCE_RECORD_SPLIT_POLICY_VERSION
    assert policy["clean_empty_recovery_policy_version"] == (
        CLEAN_EMPTY_RECOVERY_POLICY_VERSION
    )
    assert policy["source_message_ids"] == list(
        EXTRACTION_CANARY_SOURCE_MESSAGE_IDS
    )
    assert policy["table_continuation_claim_sha256"] == (
        EXTRACTION_CANARY_TABLE_CONTINUATION_CLAIM_SHA256
    )
    assert policy["prose_boundary_claim_sha256"] == (
        EXTRACTION_CANARY_PROSE_BOUNDARY_CLAIM_SHA256
    )
    assert policy["list_control_sha256"] == EXTRACTION_CANARY_LIST_CONTROL_SHA256
    assert policy["fenced_code_control_sha256"] == (
        EXTRACTION_CANARY_FENCED_CODE_CONTROL_SHA256
    )
    assert policy["structural_control_probe_version"] == (
        EXTRACTION_CANARY_STRUCTURAL_PROBE_VERSION
    )
    assert policy["source_content_chars"] > 2 * 4000
    assert extraction_canary_module._PROSE_INITIAL_CUT == (
        extraction_canary_module._PROSE_RIGHT_START
    )
    table_points = (
        extraction_canary_module.chunk_extraction
        ._markdown_table_boundary_points(extraction_canary_module._TABLE_CONTENT)
    )
    assert table_points
    assert (
        extraction_canary_module.chunk_extraction._semantic_split_point(
            extraction_canary_module._TABLE_CONTENT
        ) in table_points
    )
    assert EXTRACTION_CANARY_EXPECTED_PREPARTITION_LEAVES == 4
    assert policy["expected_prepartition_leaves"] == 4
    assert policy["normal_pass_completion_calls"] == 8
    assert policy["normal_execution_path"] == (
        extraction_canary_module._NORMAL_EXECUTION_PATH
    )
    assert policy["minimum_pass_completion_calls"] == 8
    assert policy["max_completion_calls"] == EXTRACTION_CANARY_MAX_COMPLETION_CALLS
    assert policy["max_provider_attempts"] == EXTRACTION_CANARY_MAX_PROVIDER_ATTEMPTS
    assert policy["expected_claims"] == [
        {
            "subject": "HyMem Canary Relay",
            "subject_type": "service",
            "predicate": "deploys_to",
            "object": "Fly.io",
            "object_type": "platform",
            "polarity": 1,
            "source_message_id": EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID,
            "value_text": None,
            "value_numeric": None,
            "value_unit": None,
            "temporal_scope": None,
            "subject_properties": {},
            "object_properties": {},
        },
        {
            "subject": "Avery Boundary Canary",
            "subject_type": "person",
            "predicate": "prefers",
            "object": "PostgreSQL",
            "object_type": "database",
            "polarity": 1,
            "source_message_id": EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID,
            "value_text": None,
            "value_numeric": None,
            "value_unit": None,
            "temporal_scope": None,
            "subject_properties": {},
            "object_properties": {},
        },
    ]
    assert "expected_claim" not in policy


def test_stale_v16_canary_report_is_rejected():
    stale = skipped_extraction_canary("no_dream")
    stale["version"] = "hymem-phase1-extraction-canary-v16"

    with pytest.raises(BenchmarkIntegrityError, match="policy identity"):
        validate_extraction_canary_report(stale, expected_mode="no_dream")


def test_canary_client_identity_redacts_url_credentials():
    client = _representative_client()
    client.base_url = (
        "https://canary-user:canary-password@provider.example/v1"
        "?api_key=canary-secret"
    )

    report = run_extraction_canary(client)
    encoded = json.dumps(report["client"], sort_keys=True)

    assert "canary-user" not in encoded
    assert "canary-password" not in encoded
    assert "canary-secret" not in encoded
    assert "https://provider.example/v1" in encoded
    assert "?" not in report["client"]["base_url"]


def test_configured_canary_builds_dedicated_pipeline_client(monkeypatch):
    constructed = []
    closed = []

    class Configured(_Client):
        def __init__(self, **kwargs):
            super().__init__(router=_representative_response)
            constructed.append(kwargs)
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]
            self.thinking_mode = kwargs["thinking"]
            self.effective_extra_body = (
                {"thinking": {"type": "disabled"}}
                if kwargs["thinking"] == "disabled" else {}
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        "hymem.contrib.openai_client.OpenAICompatibleClient", Configured
    )
    report = run_configured_extraction_canary(
        api_key="credential", base_url="https://memory.example/v1",
        model="pipeline-model", thinking="disabled",
    )

    assert constructed == [{
        "api_key": "credential", "base_url": "https://memory.example/v1",
        "model": "pipeline-model", "thinking": "disabled",
    }]
    assert report["status"] == "passed"
    assert report["usage_accounting"].startswith("excluded_from_scored_usage")
    assert report["client_closed"] is True
    assert closed == [True]


def test_configured_canary_success_is_fatal_when_close_fails(monkeypatch):
    close_attempts = []
    secret = "https://private.example/v1?api_key=close-secret"

    class Configured(_Client):
        def __init__(self, **kwargs):
            super().__init__(router=_representative_response)
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]
            self.thinking_mode = kwargs["thinking"]
            self.effective_extra_body = {}

        def close(self):
            close_attempts.append(True)
            raise RuntimeError(secret)

    monkeypatch.setattr(
        "hymem.contrib.openai_client.OpenAICompatibleClient", Configured
    )

    with pytest.raises(BenchmarkCleanupError) as caught:
        run_configured_extraction_canary(
            api_key="credential", base_url="https://memory.example/v1",
            model="pipeline-model", thinking="off",
        )

    assert close_attempts == [True]
    assert caught.value.cleanup_errors == ({
        "stage": "resource_close", "exception_type": "RuntimeError",
    },)
    diagnostic = str(caught.value) + json.dumps(caught.value.cleanup_errors)
    assert "close-secret" not in diagnostic
    assert "private.example" not in diagnostic


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_configured_canary_success_preserves_cleanup_control_flow(
    monkeypatch, error_type,
):
    close_error = error_type("private-close-message")
    close_attempts = []

    class Configured(_Client):
        def __init__(self, **kwargs):
            super().__init__(router=_representative_response)
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]
            self.thinking_mode = kwargs["thinking"]
            self.effective_extra_body = {}

        def close(self):
            close_attempts.append(True)
            raise close_error

    monkeypatch.setattr(
        "hymem.contrib.openai_client.OpenAICompatibleClient", Configured
    )

    with pytest.raises(error_type) as caught:
        run_configured_extraction_canary(
            api_key="credential", base_url="https://memory.example/v1",
            model="pipeline-model", thinking="off",
        )

    assert caught.value is close_error
    assert close_attempts == [True]
    notes = json.dumps(getattr(caught.value, "__notes__", []))
    assert "resource_close" in notes
    assert error_type.__name__ in notes
    assert "private-close-message" not in notes


def test_configured_canary_keeps_extraction_failure_primary_on_close_failure(
    monkeypatch, capsys,
):
    close_attempts = []
    secret = "close failed at /private/provider?key=secret"

    class Configured(_Client):
        def __init__(self, **kwargs):
            super().__init__(default=_complete())
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]
            self.thinking_mode = kwargs["thinking"]
            self.effective_extra_body = {}

        def close(self):
            close_attempts.append(True)
            raise RuntimeError(secret)

    monkeypatch.setattr(
        "hymem.contrib.openai_client.OpenAICompatibleClient", Configured
    )

    with pytest.raises(ExtractionCanaryError) as caught:
        run_configured_extraction_canary(
            api_key="credential", base_url="https://memory.example/v1",
            model="pipeline-model", thinking="off",
        )

    assert close_attempts == [True]
    assert caught.value.report["status"] == "failed"
    assert caught.value.report["client_closed"] is False
    serialized = json.dumps(caught.value.report, sort_keys=True)
    notes = json.dumps(getattr(caught.value, "__notes__", []))
    stderr = capsys.readouterr().err
    assert "resource_close" in notes and "RuntimeError" in notes
    assert "resource_close" in stderr and "RuntimeError" in stderr
    assert secret not in serialized
    assert secret not in notes
    assert secret not in stderr


@pytest.mark.parametrize("primary_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_configured_canary_preserves_unexpected_primary_during_close_failure(
    monkeypatch, primary_type,
):
    primary = primary_type("primary-control-flow")
    close_attempts = []

    class Configured(_Client):
        def __init__(self, **_kwargs):
            super().__init__(router=_representative_response)

        def close(self):
            close_attempts.append(True)
            raise RuntimeError("https://private.example/?key=cleanup-secret")

    def fail_extraction(_client, **_kwargs):
        raise primary

    monkeypatch.setattr(
        "hymem.contrib.openai_client.OpenAICompatibleClient", Configured
    )
    monkeypatch.setattr(
        "benchmarks.extraction_canary.run_extraction_canary", fail_extraction,
    )

    with pytest.raises(primary_type) as caught:
        run_configured_extraction_canary(
            api_key="credential", base_url="https://memory.example/v1",
            model="pipeline-model", thinking="off",
        )

    assert caught.value is primary
    assert close_attempts == [True]
    notes = json.dumps(getattr(caught.value, "__notes__", []))
    assert "resource_close" in notes and "RuntimeError" in notes
    assert "cleanup-secret" not in notes


def _passed_report() -> dict:
    client = _representative_client()
    client.model = "pipeline-model"
    client.base_url = "https://memory.example/v1"
    client.thinking_mode = "disabled"
    client.effective_extra_body = {"thinking": {"type": "disabled"}}
    report = run_extraction_canary(client)
    report["client_closed"] = True
    return report


def _validate_passed_report(report: dict) -> dict:
    return validate_extraction_canary_report(
        report,
        expected_mode="required",
        expected_client=extraction_canary_client_policy(
            base_url="https://memory.example/v1",
            model="pipeline-model",
            thinking="disabled",
        ),
        require_client_closed=True,
    )


def test_v17_validator_accepts_exact_live_and_zero_work_modes():
    assert _validate_passed_report(_passed_report())["status"] == "passed"
    for mode in ("simulation", "no_dream", "no_pending_work"):
        report = skipped_extraction_canary(mode)
        assert validate_extraction_canary_report(
            report, expected_mode=mode,
        )["skip_reason"] == mode
    pending = {**extraction_canary_policy(), "status": "pending"}
    assert validate_extraction_canary_report(
        pending, expected_mode="pending",
    )["status"] == "pending"
    assert skipped_extraction_canary("no_pending_work")["status"] == (
        "not_run_no_pending"
    )


def test_v17_validator_rejects_v16_passing_evidence():
    report = _passed_report()
    report["version"] = "hymem-phase1-extraction-canary-v16"

    with pytest.raises(BenchmarkIntegrityError, match="policy identity"):
        _validate_passed_report(report)


def test_v17_validator_rejects_coherent_but_forged_normal_execution_path():
    report = deepcopy(_passed_report())
    # These counters remain non-negative, bounded, mutually consistent, and
    # claim/context-complete. They are nevertheless impossible for the bound
    # eight-call deterministic path and must not replace recorded evidence.
    report["execution_path"].update(
        table_claim_requests=3,
        table_claim_exact_context_requests=3,
    )

    with pytest.raises(BenchmarkIntegrityError, match="normal execution path"):
        _validate_passed_report(report)


@pytest.mark.parametrize("closed", [False, None])
def test_v17_validator_requires_positive_configured_client_closure(closed):
    report = _passed_report()
    report["client_closed"] = closed
    with pytest.raises(BenchmarkIntegrityError, match="not closed successfully"):
        _validate_passed_report(report)

    report = _passed_report()
    report.pop("client_closed")
    with pytest.raises(BenchmarkIntegrityError, match="live report shape"):
        _validate_passed_report(report)


@pytest.mark.parametrize(
    "mutate",
    [
        # A coherent-looking one-claim partial must not become a pass.
        lambda report: report.update(
            claim_evidence=report["claim_evidence"][:1],
            matched_supported_claims=1,
            missing_expected_claim_indexes=[1],
        ),
        lambda report: report.update(
            claim_evidence=[
                report["claim_evidence"][0], report["claim_evidence"][0],
            ],
        ),
        lambda report: report["claim_evidence"][0].__setitem__(
            "source_message_id", EXTRACTION_CANARY_SOURCE_MESSAGE_ID + 1,
        ),
        lambda report: report["claim_evidence"][0].__setitem__(
            "predicate", "invented_predicate",
        ),
        lambda report: report["claim_evidence"][0].__setitem__(
            "subject_type", "invented_type",
        ),
        lambda report: report["usage"].__setitem__("calls", 1),
        lambda report: report["usage"].__setitem__("request_attempts", 3),
        lambda report: report["usage"].__setitem__("latency_s", float("nan")),
        lambda report: report.__setitem__("valid_triples_returned", 3),
        lambda report: report.__setitem__("valid_markers_returned", 1),
        lambda report: report.__setitem__("duplicate_triples_collapsed", 1),
        lambda report: report.__setitem__("initial_prepartition_leaves", 1),
        lambda report: report["execution_path"].__setitem__(
            "table_claim_exact_context_requests", 0,
        ),
        lambda report: report["execution_path"].__setitem__(
            "prose_claim_self_contained_requests", 1,
        ),
        lambda report: report["execution_path"].__setitem__(
            "protected_control_split_boundaries", 1,
        ),
        lambda report: report["execution_path"].__setitem__(
            "list_control_probe_atomic", False,
        ),
        lambda report: report["execution_path"].__setitem__(
            "fenced_code_control_probe_atomic", 1,
        ),
        lambda report: report["execution_path"].__setitem__(
            "source_message_ids_seen",
            [EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID],
        ),
        lambda report: report.__setitem__("completion_calls", True),
        lambda report: report.__setitem__("unexpected", "field"),
        lambda report: report.pop("claim_evidence"),
    ],
)
def test_v17_validator_rejects_forged_claim_shapes_and_accounting(mutate):
    report = deepcopy(_passed_report())
    mutate(report)
    with pytest.raises(BenchmarkIntegrityError, match="extraction canary"):
        _validate_passed_report(report)


@pytest.mark.parametrize("field", ["model", "effective_extra_body"])
def test_v17_validator_rejects_pipeline_model_or_body_drift_without_leaking(field):
    report = deepcopy(_passed_report())
    report["client"][field] = (
        "private-model-token" if field == "model" else {}
    )
    with pytest.raises(BenchmarkIntegrityError) as caught:
        _validate_passed_report(report)
    assert "private-model-token" not in str(caught.value)


def test_v17_validator_rejects_skip_with_work_and_failed_report_forged_to_pass():
    skipped = skipped_extraction_canary("no_dream")
    skipped["completion_calls"] = 1
    with pytest.raises(BenchmarkIntegrityError, match="zero-work"):
        validate_extraction_canary_report(skipped, expected_mode="no_dream")

    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(_Client(default=_complete()))
    forged = deepcopy(caught.value.report)
    forged["status"] = "passed"
    forged.pop("failure_reason")
    forged.pop("failure_details")
    forged["client"] = deepcopy(_passed_report()["client"])
    forged["client_closed"] = True
    with pytest.raises(BenchmarkIntegrityError, match="extraction canary.*exact"):
        _validate_passed_report(forged)


def test_live_canary_rejects_off_vocabulary_entity_type():
    client = _Client(default=_complete([
        _preference_claim(subject_type="invented_type"), _deployment_claim(),
    ]))
    with pytest.raises(ExtractionCanaryError) as caught:
        run_extraction_canary(client)
    assert caught.value.report["failure_reason"] in {
        "supported_claim_evidence_missing", "item_validation_failure",
        "branch_incomplete", "unexpected_canary_output",
    }


@pytest.mark.parametrize("valid_alternate", [False, True])
def test_msc_rejects_rehashed_manifest_contract_before_provider_or_example(
    monkeypatch, tmp_path: Path, valid_alternate: bool,
):
    events = []
    monkeypatch.setattr(msc, "load_msc_data", lambda *_a, **_k: [{
        "id": "contract-msc", "sessions": [], "session_dates": [],
        "question": "question", "answer": "answer", "persona_facts": [],
        "n_sessions": 0,
    }])
    monkeypatch.setattr(
        msc, "build_manifest", _manifest_contract_tamper(
            msc.build_manifest, valid_alternate=valid_alternate
        )
    )
    monkeypatch.setattr(
        msc, "run_configured_extraction_canary",
        lambda **_k: events.append("canary"),
    )
    monkeypatch.setattr(
        msc, "_build_llm", lambda *_a, **_k: events.append("client")
    )
    monkeypatch.setattr(
        msc, "run_recall", lambda *_a, **_k: events.append("example")
    )
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sample", "1", "--api-key", "credential",
        "--hymem-model", "pipeline-model", "--hymem-base-url",
        "https://memory.example/v1", "--results-dir",
        str(tmp_path / "results"),
    ])

    with pytest.raises(
        BenchmarkIntegrityError, match="extraction canary|manifest extraction"
    ):
        msc.main()
    assert events == []


@pytest.mark.parametrize("valid_alternate", [False, True])
def test_locomo_rejects_rehashed_manifest_contract_before_provider_or_question(
    monkeypatch, tmp_path: Path, valid_alternate: bool,
):
    events = []
    conv = {
        "id": "contract-locomo", "speaker_a": "Ada", "speaker_b": "Ben",
        "sessions": [], "session_dates": [], "n_sessions": 0,
        "evidence_map": {},
        "qa": [{
            "qa_id": "contract-q", "question_id": "contract-q",
            "question": "question", "answer": "answer",
            "adversarial_answer": "", "category": 1,
            "qtype": "multi-hop", "judge_type": "multi-session",
            "evidence": [],
        }],
    }
    monkeypatch.setattr(
        locomo, "load_locomo_data", lambda *_a, **_k: [conv]
    )
    monkeypatch.setattr(
        locomo, "build_manifest",
        _manifest_contract_tamper(
            locomo.build_manifest, valid_alternate=valid_alternate
        ),
    )
    monkeypatch.setattr(
        locomo, "run_configured_extraction_canary",
        lambda **_k: events.append("canary"),
    )
    monkeypatch.setattr(
        locomo, "_build_llm", lambda *_a, **_k: events.append("client")
    )
    monkeypatch.setattr(
        locomo, "evaluate_conversation",
        lambda *_a, **_k: events.append("question"),
    )
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--sample", "1", "--api-key", "credential",
        "--hymem-model", "pipeline-model", "--hymem-base-url",
        "https://memory.example/v1", "--results-dir",
        str(tmp_path / "results"),
    ])

    with pytest.raises(
        BenchmarkIntegrityError, match="extraction canary|manifest extraction"
    ):
        locomo.main()
    assert events == []


@pytest.mark.parametrize("valid_alternate", [False, True])
def test_lme_rejects_rehashed_manifest_contract_before_provider_or_question(
    monkeypatch, tmp_path: Path, valid_alternate: bool,
):
    dataset = [{
        "question_id": "contract-lme",
        "question_type": "multi-session",
        "question": "What database?", "answer": "PostgreSQL",
        "question_date": "2025-01-03",
        "answer_session_ids": ["source-session"],
        "haystack_session_ids": ["source-session"],
        "haystack_dates": ["2025-01-01"],
        "haystack_sessions": [[{
            "role": "user", "content": "PostgreSQL", "has_answer": True,
        }]],
    }]
    (tmp_path / "longmemeval_s_cleaned.json").write_text(
        json.dumps(dataset), encoding="utf-8"
    )
    events = []
    monkeypatch.setattr(
        lme, "build_manifest", _manifest_contract_tamper(
            lme.build_manifest, valid_alternate=valid_alternate
        )
    )
    monkeypatch.setattr(
        lme, "LLMClient", lambda *_a, **_k: events.append("client")
    )
    monkeypatch.setattr(
        lme, "run_configured_extraction_canary",
        lambda **_k: events.append("canary"),
    )
    monkeypatch.setattr(
        lme, "_evaluate_one_question",
        lambda *_a, **_k: events.append("question"),
    )
    monkeypatch.setattr(sys, "argv", [
        "longmemeval_adapter.py", "--data-dir", str(tmp_path),
        "--results-dir", str(tmp_path / "results"), "--sample", "0",
        "--no-prereg", "--api-key", "credential", "--hymem-model",
        "pipeline-model", "--hymem-base-url",
        "https://memory.example/v1", "--hymem-api-key", "credential",
    ])

    with pytest.raises(
        BenchmarkIntegrityError, match="extraction canary|manifest extraction"
    ):
        lme.main()
    assert events == []


@pytest.mark.parametrize("valid_alternate", [False, True])
def test_beam_rejects_rehashed_manifest_contract_before_provider_or_question(
    monkeypatch, tmp_path: Path, valid_alternate: bool,
):
    conversations = {"100K": [{
        "id": "contract-beam", "scale": "100K",
        "messages": [{"role": "user", "content": "source"}],
        "questions": [{
            "question_id": "contract-beam-q", "ability_short": "IE",
            "question": "question?", "ideal_answer": "",
            "gold_text": "answer", "gold_kind": "response",
            "gold_resolution": "exact", "rubric": ["criterion"],
        }],
    }]}
    events = []
    monkeypatch.setattr(
        beam, "resolve_dataset_revisions",
        lambda _scales, _pin=None: {beam.BEAM_REPO: "a" * 40},
    )
    monkeypatch.setattr(
        beam, "load_beam_conversations", lambda *_a, **_k: conversations
    )
    monkeypatch.setattr(beam, "print_gold_audit", lambda _rows: None)
    monkeypatch.setattr(
        beam, "beam_code_hash", lambda: content_hash("code")
    )
    monkeypatch.setattr(
        beam, "build_manifest", _manifest_contract_tamper(
            beam.build_manifest, valid_alternate=valid_alternate
        )
    )
    monkeypatch.setattr(
        beam, "resolve_answer_provider",
        lambda *_a, **_k: events.append("provider"),
    )
    monkeypatch.setattr(
        beam, "run_configured_extraction_canary",
        lambda **_k: events.append("canary"),
    )
    monkeypatch.setattr(
        beam, "evaluate_conversation",
        lambda *_a, **_k: events.append("question"),
    )
    monkeypatch.setattr(sys, "argv", [
        "beam_adapter.py", "--scales", "100K", "--sample", "1",
        "--no-prereg", "--api-key", "credential",
        "--embedding-backend", "none", "--results-dir",
        str(tmp_path / "results"),
    ])

    with pytest.raises(
        BenchmarkIntegrityError, match="extraction canary|manifest extraction"
    ):
        beam.main()
    assert events == []


def test_msc_invokes_canary_once_before_any_example_work(
    monkeypatch, tmp_path: Path,
):
    events = []
    output = tmp_path / "msc.json"

    class Meter:
        def __init__(self):
            self.call_count = 0
            self.request_attempts = 0
            self.successful_responses = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.total_latency_s = 0.0
            self.cost_usd = 0.0
            self.token_usage_available = True

        def close(self):
            return None

    def canary(**kwargs):
        events.append(("canary", kwargs))
        return _passed_report()

    def recall(example, _args, answer, judge, **kwargs):
        events.append(("example", example["id"]))
        for client in (answer, judge):
            client.call_count += 1
            client.request_attempts += 1
            client.successful_responses += 1
        row = {
            "id": example["id"], "question_id": example["id"],
            "question_type": "recall", "correct": True,
        }
        scope = f"msc:{example['id']}"
        from tests.archive_evidence_fixtures import scoped_indexing
        kwargs["on_checkpoint"](row, {
            "scope_id": scope,
            "indexing": scoped_indexing(scope, _args),
            "memory_pipeline_usage": msc._known_zero_pipeline_usage(),
            "embedding_usage": msc.embedding_usage_snapshot(
                None, configured=False,
            ),
        })
        return row

    monkeypatch.setattr(msc, "run_configured_extraction_canary", canary)
    monkeypatch.setattr(msc, "run_recall", recall)
    monkeypatch.setattr(msc, "_build_llm", lambda *_args, **_kwargs: Meter())
    monkeypatch.setattr(msc, "_print_recall_report", lambda _rows: None)
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sample", "1", "--api-key", "credential",
        "--hymem-model", "pipeline-model", "--hymem-base-url",
        "https://memory.example/v1", "--hymem-thinking", "disabled",
        "--out", str(output),
    ])

    msc.main()

    assert [kind for kind, _value in events] == ["canary", "example"]
    assert events[0][1] == {
        "api_key": "credential", "base_url": "https://memory.example/v1",
        "model": "pipeline-model", "thinking": "disabled",
        "prompt_version": "v20",
    }
    rows = json.loads(output.read_text())
    assert rows[0]["extraction_canary"]["status"] == "passed"


def test_msc_rejects_forged_passed_report_before_example_work(
    monkeypatch, tmp_path: Path,
):
    example_work = []
    monkeypatch.setattr(
        msc, "run_configured_extraction_canary",
        lambda **_kwargs: {
            **extraction_canary_policy(), "status": "passed",
        },
    )
    monkeypatch.setattr(
        msc, "run_recall",
        lambda *_args, **_kwargs: example_work.append(True),
    )
    monkeypatch.setattr(sys, "argv", [
        "msc_adapter.py", "--sample", "1", "--api-key", "credential",
        "--hymem-model", "pipeline-model", "--hymem-base-url",
        "https://memory.example/v1", "--hymem-thinking", "disabled",
        "--out", str(tmp_path / "must-not-exist.json"),
    ])

    with pytest.raises(BenchmarkIntegrityError, match="extraction canary"):
        msc.main()
    assert example_work == []


def test_msc_empty_recall_selection_does_not_spend_a_canary(monkeypatch):
    monkeypatch.setattr(msc, "load_msc_data", lambda *_args, **_kwargs: [{
        "id": "no-qa", "question": None, "answer": None, "n_sessions": 1,
    }])
    monkeypatch.setattr(
        msc, "run_configured_extraction_canary",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("canary must not run without recall work")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["msc_adapter.py", "--sample", "1"])

    with pytest.raises(SystemExit):
        msc.main()


def test_locomo_invokes_canary_once_before_conversation_workers(
    monkeypatch, tmp_path: Path,
):
    events = []
    output = tmp_path / "locomo.json"
    conv = {
        "id": "conv-canary", "n_sessions": 1,
        "qa": [{"qa_id": "q-canary", "question_id": "q-canary"}],
    }

    def canary(**kwargs):
        events.append(("canary", kwargs))
        return _passed_report()

    def evaluate(conversation, *_args, **_kwargs):
        events.append(("conversation", conversation["id"]))
        return [{"id": "q-canary", "question_id": "q-canary", "correct": True}]

    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_args, **_kwargs: [conv])
    monkeypatch.setattr(locomo, "run_configured_extraction_canary", canary)
    monkeypatch.setattr(locomo, "evaluate_conversation", evaluate)
    monkeypatch.setattr(locomo, "_build_llm", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(locomo, "_print_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--api-key", "credential",
        "--hymem-model", "pipeline-model", "--hymem-base-url",
        "https://memory.example/v1", "--hymem-thinking", "disabled",
        "--out", str(output),
    ])

    locomo.main()

    assert [kind for kind, _value in events] == ["canary", "conversation"]
    rows = json.loads(output.read_text())
    assert rows[0]["extraction_canary"]["status"] == "passed"


def test_locomo_does_no_benchmark_work_after_canary_close_failure(
    monkeypatch, tmp_path: Path,
):
    conversation_work = []
    close_attempts = []
    output = tmp_path / "must-not-exist.json"
    conv = {
        "id": "conv-canary", "n_sessions": 1,
        "qa": [{"qa_id": "q-canary", "question_id": "q-canary"}],
    }

    class Configured(_Client):
        def __init__(self, **kwargs):
            super().__init__(router=_representative_response)
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]
            self.thinking_mode = kwargs["thinking"]
            self.effective_extra_body = {}

        def close(self):
            close_attempts.append(True)
            raise RuntimeError("private cleanup failure")

    monkeypatch.setattr(
        "hymem.contrib.openai_client.OpenAICompatibleClient", Configured
    )
    monkeypatch.setattr(locomo, "load_locomo_data", lambda *_a, **_k: [conv])
    monkeypatch.setattr(
        locomo, "evaluate_conversation",
        lambda *_a, **_k: conversation_work.append(True),
    )
    monkeypatch.setattr(sys, "argv", [
        "locomo_adapter.py", "--api-key", "credential",
        "--hymem-model", "pipeline-model", "--hymem-base-url",
        "https://memory.example/v1", "--hymem-thinking", "off",
        "--out", str(output),
    ])

    with pytest.raises(ValueError, match="benchmark cleanup failed"):
        locomo.main()

    assert close_attempts == [True]
    assert conversation_work == []
    assert not output.exists()


def test_lme_invokes_one_run_canary_before_question_workers(
    monkeypatch, tmp_path: Path,
):
    events = []
    dataset = [{
        "question_id": "qid-canary",
        "question_type": "multi-session",
        "question": "What database?",
        "answer": "PostgreSQL",
        "question_date": "2025-01-03",
        "answer_session_ids": ["source-session"],
        "haystack_session_ids": ["source-session"],
        "haystack_dates": ["2025-01-01"],
        "haystack_sessions": [[{
            "role": "user", "content": "PostgreSQL", "has_answer": True,
        }]],
    }]
    (tmp_path / "longmemeval_s_cleaned.json").write_text(
        json.dumps(dataset), encoding="utf-8"
    )

    class Meter:
        call_count = 0
        request_attempts = 0
        successful_responses = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_latency_s = 0.0
        cost_usd = 0.0
        token_usage_available = True

        def __init__(self, *_args, **_kwargs):
            pass

    def canary(**kwargs):
        events.append(("canary", kwargs))
        return _passed_report()

    def evaluate(_qi, _total, question, *_args, **_kwargs):
        events.append(("question", question["question_id"]))
        return {
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "correct": False,
            "benchmark_failure": "fixture",
            "oracle_ability": "MR", "detected_ability": None,
            "ability_used": None, "retrieval_only": False,
            "distill_fired": False, "distill_calls": 0,
            "memory_pipeline_usage": lme.usage_snapshot(Meter()),
        }

    monkeypatch.setattr(lme, "LLMClient", Meter)
    monkeypatch.setattr(lme, "run_configured_extraction_canary", canary)
    monkeypatch.setattr(lme, "_evaluate_one_question", evaluate)
    monkeypatch.setattr(sys, "argv", [
        "longmemeval_adapter.py", "--data-dir", str(tmp_path),
        "--results-dir", str(tmp_path), "--sample", "0", "--no-prereg",
        "--api-key", "credential", "--hymem-model", "pipeline-model",
        "--hymem-base-url", "https://memory.example/v1",
        "--hymem-thinking", "disabled", "--hymem-api-key", "credential",
    ])

    lme.main()

    assert [kind for kind, _value in events] == ["canary", "question"]
    archives = list(tmp_path.glob("longmemeval-v2-hymem-*-strict-*.json"))
    assert len(archives) == 1
    artifact = json.loads(archives[0].read_text())
    segment = artifact["execution"]["segments"][0]
    assert segment["extraction_canary"]["status"] == "passed"
    assert segment["extraction_canary"]["usage_accounting"] == (
        EXTRACTION_CANARY_USAGE_ACCOUNTING
    )
    assert segment["memory_pipeline_usage"]["calls"] == 0
