"""Offline SIWC bridge controls; no credential file, socket, or model call."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from benchmarks import chatgpt_plan_lme_v2 as bridge
from hymem.extraction.llm import LLMRequest


def graph(monkeypatch, *, responses=None):
    broker = object.__new__(bridge.owner.CredentialBroker)
    broker.identity_digest = "a" * 64
    lease = bridge.owner.CredentialLease("offline-token", int(time.time()) + 1000)
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire",
                        lambda self, *, caller_deadline: lease)
    limits = bridge.warm.BudgetLimits(30, 10000, 300)
    budget = bridge.SharedBudget(limits, max_in_flight=4)
    calls = []

    def response(credentials, system, user, schema, *, timeout):
        calls.append((credentials, system, user, schema, timeout))
        if responses:
            result = responses.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return bridge.transport.Completed("offline-answer", 9, 4, 13, 0, 0)

    client = bridge.SIWCLMEClient(broker, budget, "q", limits, response_call=response)
    return broker, budget, client, calls


def test_exact_owner_lease_admits_and_records_real_request(monkeypatch):
    _, budget, client, calls = graph(monkeypatch)
    request = LLMRequest("complete system prompt", "complete user prompt", "text", 321, 0.2)
    assert client.complete(request) == "offline-answer"
    assert calls[0][1:4] == (request.system, request.user, None)
    assert 0 < calls[0][4] <= 120
    assert budget.snapshot()["turns"] == 1
    assert budget.snapshot()["known_tokens"] == 13
    summary = bridge.validate_summary_projection(client.diagnostic_summary())
    assert summary["calls"] == summary["successes"] == summary["internal_http_attempts"] == 1
    assert summary["admitted_turns"] == 1 and summary["known_tokens"] == 13
    assert summary["provider_internal_retries_known"] is False
    assert client.requested_controls == [{"temperature_requested": 0.2, "temperature_effective": None,
        "max_tokens_requested": 321, "max_tokens_effective": None,
        "response_format_requested": "text", "response_format_effective": None,
        "output_schema_sent": False, "output_schema_acknowledged": False}]


def test_no_forged_dict_or_replayed_proof_admission(monkeypatch):
    broker, budget, client, calls = graph(monkeypatch)
    budget.reserve("q")
    with pytest.raises(bridge.warm.ConcurrentStop, match="admission_rejected"):
        budget.before_turn("q", {"policy": bridge.POLICY, "identity_digest": broker.identity_digest})
    budget.settle("q", used=None, turn_started=False, failure="admission_rejected")
    assert budget.snapshot()["turns"] == 0 and not calls
    assert budget.snapshot()["stopped"]


def test_owner_denial_stops_before_http_and_preserves_fixed_code(monkeypatch):
    _, budget, client, calls = graph(monkeypatch)
    def deny(self, *, caller_deadline):
        raise bridge.owner.OwnerError("refresh_denied")
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire", deny)
    with pytest.raises(bridge.BridgeError, match="refresh_denied"):
        client.complete(LLMRequest("s", "u"))
    summary = bridge.validate_summary_projection(client.diagnostic_summary())
    assert summary["internal_http_attempts"] == 0 and summary["admitted_turns"] == 0
    assert summary["first_failure"] == {"code": "refresh_denied", "phase": "admission",
                                        "turn_admitted": False, "unknown_usage": False}
    assert budget.snapshot()["stopped"] and not calls


@pytest.mark.parametrize("mutate", ["policy", "expiry", "identity"])
def test_lease_binding_rejects_before_http(monkeypatch, mutate):
    broker, budget, client, calls = graph(monkeypatch)
    if mutate == "policy":
        lease = bridge.owner.CredentialLease("offline-token", int(time.time()) + 1000)
        lease.policy = "unrelated"
        monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire",
                            lambda self, *, caller_deadline: lease)
    elif mutate == "expiry":
        lease = bridge.owner.CredentialLease("offline-token", int(time.time()) + 10)
        monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire",
                            lambda self, *, caller_deadline: lease)
    else:
        broker.identity_digest = "wrong"
    with pytest.raises(bridge.BridgeError, match="admission_rejected"):
        client.complete(LLMRequest("s", "u"))
    assert budget.snapshot()["stopped"] and budget.snapshot()["turns"] == 0
    assert not calls and client.internal_http_attempts == 0


def test_original_grant_is_immutable_across_turns(monkeypatch):
    broker, budget, client, calls = graph(monkeypatch)
    broker.identity_digest = "b" * 64
    with pytest.raises(bridge.BridgeError, match="admission_rejected"):
        client.complete(LLMRequest("s", "u"))
    assert budget.snapshot()["turns"] == 0 and not calls


def test_source_closure_and_malformed_lease(monkeypatch):
    for module, relative, digest, symbol in bridge._PINS:
        path = Path(bridge.__file__).resolve().parents[1] / relative
        assert Path(module.__file__).resolve() == path.resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        target = getattr(module, symbol)
        code = target.acquire if symbol == "CredentialBroker" else target
        assert Path(code.__code__.co_filename).resolve() == path.resolve()
    _, budget, client, calls = graph(monkeypatch)
    lease = bridge.owner.CredentialLease("bad\nheader", int(time.time()) + 1000)
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire",
                        lambda self, *, caller_deadline: lease)
    with pytest.raises(bridge.BridgeError, match="invalid_credentials"):
        client.complete(LLMRequest("s", "u"))
    assert budget.snapshot()["turns"] == 0 and not calls


def test_owner_deadline_failure_has_no_http_attempt(monkeypatch):
    _, budget, client, calls = graph(monkeypatch)
    def timeout(self, *, caller_deadline):
        raise bridge.owner.OwnerError("deadline_exceeded")
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire", timeout)
    with pytest.raises(bridge.BridgeError, match="deadline_exceeded"):
        client.complete(LLMRequest("s", "u"))
    assert budget.snapshot()["turns"] == 0 and not calls
    assert client.diagnostic_summary()["first_failure"]["phase"] == "admission"


@pytest.mark.parametrize("kind", ["closed", "bad_type", "oversized"])
def test_pre_admission_failure_is_finite_and_unreserved(monkeypatch, kind):
    _, budget, client, calls = graph(monkeypatch)
    if kind == "closed":
        client.close()
        request = LLMRequest("s", "u")
        expected = "invalid_request_or_closed"
    elif kind == "bad_type":
        request = object()
        expected = "invalid_request_or_closed"
    else:
        request = LLMRequest("s", "x" * (bridge.transport.MAX_REQUEST_BYTES + 1))
        expected = "request_limit"
    with pytest.raises(bridge.BridgeError, match=expected):
        client.complete(request)
    snapshot = budget.snapshot()
    assert snapshot["stopped"] and snapshot["turns"] == 0
    assert snapshot["reserved"] == snapshot["in_flight"] == 0
    assert not calls
    summary = bridge.validate_summary_projection(client.diagnostic_summary())
    assert summary["calls"] == summary["failures"] == 1
    assert summary["internal_http_attempts"] == 0
    assert summary["first_failure"]["code"] == expected


def test_subclassed_resource_budget_and_alias(monkeypatch):
    broker = object.__new__(bridge.owner.CredentialBroker)
    broker.identity_digest = "a" * 64
    lease = bridge.owner.CredentialLease("offline-token", int(time.time()) + 1000)
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire",
                        lambda self, *, caller_deadline: lease)
    limits = bridge.warm.BudgetLimits(10, 1000, 120)
    class ObservedBudget(bridge.SharedBudget):
        def __init__(self):
            super().__init__(limits)
            self.reservations = 0

        def reserve(self, question_id):
            self.reservations += 1
            return super().reserve(question_id)

    budget = ObservedBudget()
    client = bridge.SIWCLMEClient(broker, budget, "q", limits,
        response_call=lambda *args, **kwargs: bridge.transport.Completed("ok", 1, 1, 2, 0, 0))
    assert client.complete(LLMRequest("s", "u")) == "ok"
    assert budget.reservations == 1
    assert budget.snapshot()["first_failure"] is None


def test_http_error_stops_with_unknown_usage_and_safe_observation(monkeypatch):
    exc = bridge.transport.TransportError("http_failure", 403, "json", "json")
    _, budget, client, calls = graph(monkeypatch, responses=[exc])
    with pytest.raises(bridge.BridgeError, match="http_failure"):
        client.complete(LLMRequest("private system", "private user"))
    summary = bridge.validate_summary_projection(client.diagnostic_summary())
    assert summary["calls"] == summary["failures"] == summary["internal_http_attempts"] == 1
    assert summary["first_failure"]["phase"] == "http"
    assert summary["first_failure"]["unknown_usage"] is True
    assert summary["first_failure"]["http_status"] == 403
    assert not summary["usage_complete"] and budget.snapshot()["stopped"]
    assert "private" not in repr(summary) and len(calls) == 1


def test_four_concurrent_questions_share_one_ledger(monkeypatch):
    broker, budget, first, calls = graph(monkeypatch)
    limits = budget.limits
    clients = [first] + [bridge.SIWCLMEClient(broker, budget, f"q{i}", limits,
               response_call=first.response_call) for i in range(1, 4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda client: client.complete(LLMRequest("s", "u")), clients)) == ["offline-answer"] * 4
    snap = budget.snapshot()
    assert snap["turns"] == 4 and snap["known_tokens"] == 52 and snap["in_flight"] == 0
    assert len(calls) == 4


def test_four_real_overlaps_and_fifth_is_rejected(monkeypatch):
    broker, budget, first, _ = graph(monkeypatch)
    limits = budget.limits
    entered = threading.Barrier(5, timeout=5)
    release = threading.Event()
    def held_response(*args, **kwargs):
        entered.wait()
        assert release.wait(5)
        return bridge.transport.Completed("ok", 1, 1, 2, 0, 0)
    first.response_call = held_response
    clients = [first] + [bridge.SIWCLMEClient(broker, budget, f"q{i}", limits,
              response_call=held_response) for i in range(1, 5)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(client.complete, LLMRequest("s", "u")) for client in clients[:4]]
        entered.wait()
        assert budget.snapshot()["in_flight"] == 4
        with pytest.raises(bridge.warm.ConcurrentStop, match="concurrency_limit"):
            clients[4].complete(LLMRequest("s", "u"))
        release.set()
        assert [future.result() for future in futures] == ["ok"] * 4
    assert budget.snapshot()["turns"] == 4 and budget.snapshot()["in_flight"] == 0


def test_proof_replay_rejected(monkeypatch):
    broker, budget, _, _ = graph(monkeypatch)
    lease = bridge.owner.CredentialLease("offline-token", int(time.time()) + 1000)
    proof = bridge._AdmissionProof(budget._siwc_proof_secret, broker, lease, "q",
                                   time.monotonic() + 100)
    budget.reserve("q")
    budget.before_turn("q", proof)
    budget.settle("q", used=2, turn_started=True)
    budget.reserve("q")
    with pytest.raises(bridge.warm.ConcurrentStop, match="admission_rejected"):
        budget.before_turn("q", proof)
    budget.settle("q", used=None, turn_started=False, failure="admission_rejected")
    assert budget.snapshot()["turns"] == 1 and budget.snapshot()["stopped"]


def test_registration_alias_shares_question_turns_without_double_count(monkeypatch):
    broker, budget, ordinary, calls = graph(monkeypatch)
    class RegistrationAlias:
        def __init__(self, source):
            self._budget = source

        def register(self, key, limits):
            assert key == "q" and limits == self._budget.limits

        def __getattr__(self, key):
            return getattr(self._budget, key)

    structured = bridge.SIWCLMEClient(broker, RegistrationAlias(budget), "q",
                                       budget.limits, response_call=ordinary.response_call)
    assert ordinary.complete(LLMRequest("s1", "u1")) == "offline-answer"
    assert structured.complete(LLMRequest("s2", "u2")) == "offline-answer"
    a = ordinary.diagnostic_summary()
    b = structured.diagnostic_summary()
    assert a["calls"] == b["calls"] == 1
    assert a["admitted_turns"] == b["admitted_turns"] == 2
    assert budget.snapshot()["turns"] == 2 and len(calls) == 2


def test_staged_exact_schema_and_controls(monkeypatch):
    _, _, client, calls = graph(monkeypatch)
    source = bridge.staged_v6.classification.GroundingSource(
        7, "Mira uses CairnDB.", source_role="user", source_peer_id="invented",
        source_created_at="2026-09-30")
    triple = bridge.staged_v6.classification.Triple(
        "Mira", "uses", "CairnDB", 1, source_message_id=7)
    request, batch = bridge.staged_v6.staged.build_original_request((triple,), (source,))
    assert client.complete_stage(request, batch, "original", False) == "offline-answer"
    assert calls[0][1:3] == (request.system, request.user)
    assert calls[0][3] == bridge.staged_v6.staged.build_original_output_schema(batch)
    assert client.requested_controls[-1]["output_schema_sent"] is True
    assert client.requested_controls[-1]["output_schema_acknowledged"] is True


def test_staged_alternatives_recheck_and_binding(monkeypatch):
    _, budget, client, calls = graph(monkeypatch)
    source = bridge.staged_v6.classification.GroundingSource(
        7, "Mira uses CairnDB.", source_role="user", source_peer_id="invented",
        source_created_at="2026-09-30")
    triple = bridge.staged_v6.classification.Triple("Mira", "uses", "CairnDB", 1,
                                                    source_message_id=7)
    original, batch = bridge.staged_v6.staged.build_original_request((triple,), (source,))
    raw = json.dumps({"schema": bridge.staged_v6.staged.ORIGINAL_SCHEMA,
        "batch_sha256": batch.batch_sha256, "complete": True,
        "originals": [{"index": 0, "original": {"state": "not_established", "support": None}}]})
    alternative, alt_batch = bridge.staged_v6.staged.build_alternatives_request(batch, raw)
    assert client.complete_stage(original, batch, "original", True) == "offline-answer"
    with pytest.raises(bridge.staged_v6.staged.GroundingContractError):
        client.complete_stage(alternative, alt_batch, "alternatives", True)
    assert budget.snapshot()["turns"] == 1 and len(calls) == 1
    assert client.complete_stage(alternative, alt_batch, "alternatives", False) == "offline-answer"
    assert calls[1][1:4] == (alternative.system, alternative.user,
                             bridge.staged_v6.staged.build_alternatives_output_schema(alt_batch))
    from dataclasses import replace
    with pytest.raises(bridge.staged_v6.staged.GroundingContractError):
        client.complete_stage(replace(original, user=original.user + " "), batch, "original", False)
    assert len(calls) == 2


def test_strict_summary_and_pilot_reconciliation(monkeypatch):
    _, _, client, _ = graph(monkeypatch)
    summary = client.diagnostic_summary()
    for mutation in (lambda x: x.update(extra="hidden"),
                     lambda x: x.update(calls=True),
                     lambda x: x.update(provider_internal_retries_known=True)):
        bad = copy.deepcopy(summary)
        mutation(bad)
        with pytest.raises(ValueError):
            bridge.validate_summary_projection(bad)
    row = {"question_id": "", "ordinary": summary, "structured": summary,
           "ledger": {"admitted_turns": 0, "known_tokens": 0, "usage_complete": True}}
    canary = {**row, "question_id": "canary"}
    questions = [{**row, "question_id": f"q-{i}"} for i in range(4)]
    pilot = {"schema": "siwc_lme_pilot_projection_v2", "canary": canary,
             "questions": questions, "aggregate": {"calls": 0, "successes": 0,
             "failures": 0, "internal_http_attempts": 0, "admitted_turns": 0, "known_tokens": 0}}
    assert bridge.validate_pilot_projection(pilot) is pilot
    bad = copy.deepcopy(pilot)
    bad["questions"][0]["ledger"]["known_tokens"] = 1
    with pytest.raises(ValueError):
        bridge.validate_pilot_projection(bad)
    bad = copy.deepcopy(pilot)
    bad["questions"][0]["question_id"] = "canary"
    with pytest.raises(ValueError):
        bridge.validate_pilot_projection(bad)
    bad = copy.deepcopy(pilot)
    bad["aggregate"]["calls"] = False
    with pytest.raises(ValueError):
        bridge.validate_pilot_projection(bad)


def test_resource_first_fault_projection_and_typed_nested_rejection():
    fault = {"code": "resource_task_denial", "phase": "admission",
             "turn_admitted": False, "unknown_usage": False,
             "resource_observation": {"current": 12, "peak": 20, "limit": 256, "denials": 1},
             "underlying_code": "refresh_denied"}
    projected = bridge.project_first_failure(fault)
    assert projected == fault and projected is not fault
    for path, mutation in (("limit", True), ("denials", -1), ("current", 257)):
        bad = copy.deepcopy(fault)
        bad["resource_observation"][path] = mutation
        with pytest.raises(ValueError):
            bridge.project_first_failure(bad)
    bad = copy.deepcopy(fault)
    bad["underlying_code"] = "secret-provider-message"
    with pytest.raises(ValueError):
        bridge.project_first_failure(bad)
    for key, value in (("wire_observation", None), ("stream_observation", None),
                       ("phase", [])):
        bad = copy.deepcopy(fault)
        bad[key] = value
        with pytest.raises(ValueError):
            bridge.project_first_failure(bad)


def timeout_observation():
    return {"child_phase": "stream_read", "parent_timeout_site": "result_wait",
            "last_progress_elapsed_ms": 20, "parent_elapsed_ms": 50,
            "elapsed_saturated": False, "snapshot_valid": True,
            "timeout_allowance_ms": 1000, "wire_bytes": 18, "event_count": 1,
            "completion_seen": False, "result_ready": False,
            "result_ipc_started": False, "child_alive_when_sampled": True}


def test_timeout_observation_propagates_without_retry_and_is_immutable(monkeypatch):
    observation = timeout_observation()
    assert bridge.transport.sanitize_timeout_observation(observation) == observation
    exc = bridge.transport.TransportError("timeout", timeout_observation=observation)
    _, budget, client, calls = graph(monkeypatch, responses=[exc])
    with pytest.raises(bridge.BridgeError, match="timeout"):
        client.complete(LLMRequest("private system", "private user"))
    summary = bridge.validate_summary_projection(client.diagnostic_summary())
    assert summary["schema"] == "siwc_lme_summary_v2"
    assert summary["calls"] == summary["internal_http_attempts"] == 1
    assert summary["first_failure"]["timeout_observation"] == observation
    assert budget.snapshot()["first_failure"]["timeout_observation"] == observation
    observation["child_phase"] = "secret content"
    exc.timeout_observation["wire_bytes"] = 999
    summary["first_failure"]["timeout_observation"]["event_count"] = 900
    assert client.diagnostic_summary()["first_failure"]["timeout_observation"]["child_phase"] == "stream_read"
    assert client.diagnostic_summary()["first_failure"]["timeout_observation"]["wire_bytes"] == 18
    assert budget.snapshot()["first_failure"]["timeout_observation"]["event_count"] == 1
    assert "private" not in repr(client.diagnostic_summary()) and len(calls) == 1


def test_mutated_exception_metadata_is_sanitized_at_capture(monkeypatch):
    exc = bridge.transport.TransportError("timeout", timeout_observation=timeout_observation())
    exc.http_status = True
    exc.body_shape = "private body"
    exc.media_type_class = "private media"
    exc.wire_observation = {"url": "private"}
    exc.stream_observation = {"text": "private"}
    exc.timeout_observation = {**timeout_observation(), "child_phase": "private phase"}
    _, budget, client, calls = graph(monkeypatch, responses=[exc])
    with pytest.raises(bridge.BridgeError, match="timeout"):
        client.complete(LLMRequest("s", "u"))
    first = bridge.validate_summary_projection(client.diagnostic_summary())["first_failure"]
    assert first == {"code": "timeout", "phase": "http", "turn_admitted": True,
                     "unknown_usage": True}
    assert budget.snapshot()["first_failure"] == first
    assert len(calls) == 1


def test_timeout_projection_rejects_invalid_and_misplaced_metadata():
    fault = {"code": "timeout", "phase": "http", "turn_admitted": True,
             "unknown_usage": True, "timeout_observation": timeout_observation()}
    assert bridge.project_first_failure(fault) == fault
    for key, value in (("child_phase", "private content"), ("wire_bytes", True),
                       ("event_count", -1), ("parent_elapsed_ms", float("inf"))):
        bad = copy.deepcopy(fault)
        bad["timeout_observation"][key] = value
        with pytest.raises(ValueError):
            bridge.project_first_failure(bad)
    for code, phase, admitted, unknown in (("http_failure", "http", True, True),
                                          ("timeout", "admission", False, False)):
        bad = copy.deepcopy(fault)
        bad.update(code=code, phase=phase, turn_admitted=admitted, unknown_usage=unknown)
        with pytest.raises(ValueError):
            bridge.project_first_failure(bad)
    wrapped = copy.deepcopy(fault)
    wrapped.update(code="resource_observer_unverified", underlying_code="timeout")
    assert bridge.project_first_failure(wrapped) == wrapped


def test_pilot_projection_carries_timeout_observation_and_rejects_mutation(monkeypatch):
    exc = bridge.transport.TransportError("timeout", timeout_observation=timeout_observation())
    _, _, client, _ = graph(monkeypatch, responses=[exc])
    empty = client.diagnostic_summary()
    with pytest.raises(bridge.BridgeError, match="timeout"):
        client.complete(LLMRequest("s", "u"))
    failed = client.diagnostic_summary()
    shared = {key: failed[key] for key in ("admitted_turns", "known_tokens", "usage_complete")}
    silent = copy.deepcopy(empty)
    silent.update(shared)
    def row(qid, ordinary, structured, ledger):
        return {"question_id": qid, "ordinary": copy.deepcopy(ordinary),
                "structured": copy.deepcopy(structured), "ledger": copy.deepcopy(ledger)}
    zero = {"admitted_turns": 0, "known_tokens": 0, "usage_complete": True}
    pilot = {"schema": "siwc_lme_pilot_projection_v2",
             "canary": row("canary", empty, empty, zero),
             "questions": [row("q0", failed, silent, shared)] +
                          [row(f"q{i}", empty, empty, zero) for i in range(1, 4)],
             "aggregate": {"calls": 1, "successes": 0, "failures": 1,
                           "internal_http_attempts": 1, "admitted_turns": 1,
                           "known_tokens": 0}}
    assert bridge.validate_pilot_projection(pilot) is pilot
    bad = copy.deepcopy(pilot)
    bad["questions"][0]["ordinary"]["first_failure"]["timeout_observation"]["child_phase"] = "secret"
    with pytest.raises(ValueError):
        bridge.validate_pilot_projection(bad)
