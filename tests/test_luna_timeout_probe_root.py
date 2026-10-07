"""Independent probe integration controls with no runtime or provider calls."""
from copy import deepcopy
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_timeout_v1 as observed
from hymem.extraction.llm import LLMRequest
from tools.diagnostics import luna_timeout_probe_v1 as probe


PRIVATE = "PRIVATE-ROOT-TEST-ONLY"


def attested(*args):
    return {"containment": True, "denials": 0, "oom": 0}


def setup_transport(monkeypatch, *, fault=False, cleanup_failure=False):
    """Use the real budget, observer, warm routing and final-output parser."""
    monkeypatch.setattr(observed.base.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="codex-cli 0.158.0"))
    processes = []
    def popen(*a, **k):
        process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(),
                                  poll=lambda: None)
        processes.append(process)
        return process
    monkeypatch.setattr(observed.base.subprocess, "Popen", popen)
    # Avoid the real stdio reader; preflight below supplies invented wire data.
    monkeypatch.setattr(observed.warm.WarmSession, "_read", lambda self: None)
    def close(session):
        session.closed = True
        if cleanup_failure:
            raise RuntimeError(PRIVATE)
    monkeypatch.setattr(observed.TimeoutSession, "close", close)
    barrier = threading.Barrier(4)
    sessions = []
    lock = threading.Lock()
    peaks = []
    clients = []
    def factory(*args, **kwargs):
        item = observed.TimeoutSubscriptionClient(*args, **kwargs)
        clients.append(item)
        return item
    def ready(session, **kwargs):
        with lock:
            fresh = session not in sessions
            if fresh:
                sessions.append(session)
            which = sessions.index(session)
        if fresh:
            barrier.wait(timeout=3)
            peaks.append(clients[0].budget.snapshot()["in_flight"])
        thread_id = "invented-thread-" + str(session.next_id)
        session.active_thread = thread_id
        def event(method, **fields):
            return {"method": method, "params": {"threadId": thread_id,
                    "turnId": "invented-turn", **fields}}
        values = [
            {"id": session.next_id + 1, "result": {
                "turn": {"id": "invented-turn", "status": "inProgress"}}},
            event("item/started", item={"id": "invented-item", "type": "agentMessage"}),
            event("item/completed", item={"id": "invented-item", "type": "agentMessage",
                  "phase": "final_answer", "text": PRIVATE}),
            event("thread/tokenUsage/updated", tokenUsage={"total": {"totalTokens": 19}}),
            event("turn/completed", turn={"id": "invented-turn", "status": "completed"}),
            {"id": session.next_id + 2, "result": {"status": "unsubscribed"}},
        ]
        session.fail_here = fault and which == 0
        for value in values:
            session.events.put(value)
        return {"auth": "chatgpt", "model": "gpt-6-luna", "inference_enabled": False,
                "config_isolation_admitted": True, "_thread_id": thread_id,
                "quota_windows": [{"remaining_percent": 80}]}
    monkeypatch.setattr(observed.base, "inspect_preflight", ready)
    original_next = observed.TimeoutSession.next_event
    def next_event(session):
        if session.fail_here:
            observed.base._fail("timeout")
        return original_next(session)
    monkeypatch.setattr(observed.TimeoutSession, "next_event", next_event)
    return factory, clients, processes, peaks


def test_root_probe_real_observer_parser_and_budget_four_warm_workers(monkeypatch):
    factory, clients, processes, peaks = setup_transport(monkeypatch)
    result = probe.run_probe(observed, LLMRequest, "unused", attest=attested,
                             client_factory=factory)
    assert result["status"] == "observed_success"
    assert result["attempted"] == result["returned"] == result["turns"] == 16
    assert result["known_tokens"] == 16 * 19
    assert result["usage_complete"] is True
    assert result["not_attempted"] == 0
    assert len(processes) == len(clients) == 4 and max(peaks) == 4
    assert all(client.requests_on_process == 4 for client in clients)
    assert all(client.session is None and client.directory is None for client in clients)
    assert result["client_cleanup_verified"] is True
    assert result["independent_recursive_cleanup_verified"] is None
    assert result["historical_timeout_cause_proved"] is False
    assert result["lme_readiness_proved"] is False
    assert PRIVATE not in json.dumps(result)
    assert all(row["observation"]["observed"]["completed_seen"] is True
               for row in result["records"])


def test_root_missing_containment_gate_cannot_construct_clients():
    def forbidden(*args, **kwargs):
        pytest.fail("client construction before containment")
    with pytest.raises(ValueError, match="containment_attestor_required"):
        probe.run_probe(observed, LLMRequest, "unused", client_factory=forbidden)


def test_root_denied_initial_gate_returns_zero_turn_failure():
    def forbidden(*args, **kwargs):
        pytest.fail("client constructed despite denied containment")
    result = probe.run_probe(observed, LLMRequest, "unused",
        attest=lambda *args: {"containment": True, "denials": 1, "oom": 0},
        client_factory=forbidden)
    assert result["status"] == "incomplete_or_failed"
    assert result["turns"] == result["attempted"] == 0
    assert result["not_attempted"] == 16
    assert result["terminal_attested_by_caller"] is False


def test_root_before_admission_denial_starts_no_transport(monkeypatch):
    factory, clients, processes, _ = setup_transport(monkeypatch)
    def gate(stage, index):
        return {"containment": stage != "before_admission", "denials": 0, "oom": 0}
    result = probe.run_probe(observed, LLMRequest, "unused", attest=gate,
                             client_factory=factory)
    assert result["status"] == "incomplete_or_failed"
    assert result["turns"] == result["attempted"] == 0
    assert len(clients) == 4 and not processes
    assert result["client_cleanup_verified"] is True


def test_root_terminal_containment_cannot_be_inferred_from_completed_calls(monkeypatch):
    factory, _, _, _ = setup_transport(monkeypatch)
    def gate(stage, index):
        return {"containment": stage != "terminal", "denials": 0, "oom": 0}
    result = probe.run_probe(observed, LLMRequest, "unused", attest=gate,
                             client_factory=factory)
    assert result["returned"] == 16
    assert result["status"] == "incomplete_or_failed"
    assert result["terminal_attested_by_caller"] is False
    assert result["independent_recursive_cleanup_verified"] is None


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_root_timeout_remains_first_fault_with_unknown_usage(monkeypatch, cleanup_failure):
    factory, clients, processes, peaks = setup_transport(monkeypatch, fault=True,
                                                        cleanup_failure=cleanup_failure)
    result = probe.run_probe(observed, LLMRequest, "unused", attest=attested,
                             client_factory=factory)
    assert result["status"] == "incomplete_or_failed"
    assert result["usage_complete"] is False
    assert result["first_failure"]["code"] == "timeout"
    assert result["attempted"] <= 16
    assert result["reserved"] == result["in_flight"] == 0
    assert PRIVATE not in json.dumps(result)
    assert result["independent_recursive_cleanup_verified"] is None


def test_root_missing_observations_do_not_erase_primary_timeout(monkeypatch):
    factory, _, _, _ = setup_transport(monkeypatch, fault=True)
    def no_records(self):
        raise ValueError(PRIVATE)
    monkeypatch.setattr(observed.TimeoutSubscriptionClient, "diagnostic_records", no_records)
    result = probe.run_probe(observed, LLMRequest, "unused", attest=attested,
                             client_factory=factory)
    assert result["status"] == "incomplete_or_failed"
    assert result["first_failure"]["code"] == "timeout"
    assert result["usage_complete"] is False
    assert result["reserved"] == result["in_flight"] == 0
    assert all(row["observation"] is None for row in result["records"])
    assert PRIVATE not in json.dumps(result)
    assert probe.validate_result(result, observed) is True


def test_root_substituted_preparer_cannot_receive_a_trusted_receipt(tmp_path):
    source = tmp_path / "alternate"
    repo = Path(__file__).resolve().parents[1]
    for relative in (*probe.SOURCE_PINS, probe.SELF_RELATIVE):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
    with (source / probe.SELF_RELATIVE).open("a") as target:
        target.write("\n# substituted source\n")
    with pytest.raises(ValueError):
        probe.prepare(tmp_path / "prepared", source_root=source)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_root_bundle_inventory_rejects_non_source_entries(tmp_path, kind):
    root = tmp_path / "prepared"
    receipt = probe.prepare(root)
    unexpected = root / "code" / "unexpected"
    if kind == "directory":
        unexpected.mkdir()
    else:
        os.mkfifo(unexpected)
    with pytest.raises(ValueError):
        probe.verify_prepared(root, receipt["receipt_sha256"])


def test_root_bundle_loader_refuses_ambient_hymem_imports(tmp_path):
    root = tmp_path / "prepared"
    receipt = probe.prepare(root)
    with pytest.raises(ValueError, match="ambient_module_present"):
        probe.load_prepared(root, receipt["receipt_sha256"])


def test_root_malformed_first_fault_retains_safe_unknown_marker():
    assert probe._project_failure(observed, {"first_failure": [PRIVATE]}) == (None, True)


def test_root_loader_rejects_changed_bytes_before_any_execution(tmp_path):
    source = tmp_path / "changing.py"
    clean = b"value = 1\n"
    expected = hashlib.sha256(clean).hexdigest()
    marker = tmp_path / "executed"
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    with pytest.raises(ValueError):
        probe._load_source("root_modified_source", source, expected)
    assert not marker.exists()


def test_root_loader_executes_the_one_verified_read_not_changed_disk(tmp_path, monkeypatch):
    source = tmp_path / "changing.py"
    clean = b"value = 7\n"
    expected = hashlib.sha256(clean).hexdigest()
    source.write_bytes(clean)
    original = Path.read_bytes
    reads = []
    def read(path):
        content = original(path)
        if path == source:
            reads.append(True)
            source.write_bytes(b"raise RuntimeError('unverified source executed')\n")
        return content
    monkeypatch.setattr(Path, "read_bytes", read)
    module = probe._load_source("root_same_verified_bytes", source, expected)
    assert module.value == 7
    assert len(reads) == 1


@pytest.mark.parametrize("field,value", [
    ("status", []), ("runner_fault", {}), ("budget_stop_code", PRIVATE),
    ("first_failure", {"code": PRIVATE}), ("known_tokens", True),
    ("independent_recursive_cleanup_verified", True),
    ("historical_timeout_cause_proved", True), ("lme_readiness_proved", True),
])
def test_root_public_result_rejects_private_or_overclaimed_state(monkeypatch, field, value):
    factory, _, _, _ = setup_transport(monkeypatch)
    result = probe.run_probe(observed, LLMRequest, "unused", attest=attested,
                             client_factory=factory)
    changed = deepcopy(result)
    changed[field] = value
    assert probe.validate_result(changed, observed) is False
