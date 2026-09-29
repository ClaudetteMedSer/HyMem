"""Offline contract checks; these never build an instance or contact providers."""
import hashlib
import importlib.util
import io
from pathlib import Path
import sqlite3
import socket
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

PATH = Path(__file__).resolve().parents[1] / "claim_conflict_v64_production_dream.py"
spec = importlib.util.spec_from_file_location("production_v64_test", PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_frozen_paid_bounds_and_target():
    assert gate.BOUNDS == {"completions": 128, "llm_http": 384,
        "embedding_http": 512, "total_http": 896, "deadline_seconds": 2700,
        "embedding_texts": 16, "embedding_chars": 128000,
        "embedding_utf8_bytes": 512000}
    assert gate.TARGET == "chk_c14182771bd8e2a7e583f7d693cee29a5fb159fe"


def test_wrong_release_cannot_verify_or_open_store(monkeypatch):
    verify = Mock()
    monkeypatch.setattr(gate, "verify", verify)
    monkeypatch.setattr(gate.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"wrong")))
    with pytest.raises(RuntimeError, match="worker_authorization_missing"):
        gate.worker(Path("/unused"), "unused")
    verify.assert_not_called()


@pytest.mark.parametrize("open_run,lease", [(True, False), (False, True)])
def test_idle_gate_refuses_open_run_or_lease(monkeypatch, open_run, lease):
    from hymem.core import db
    monkeypatch.setattr(db, "schema_version", lambda _conn: 64)
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE dream_runs(ended_at TEXT); CREATE TABLE run_lock(name TEXT);")
    if open_run:
        conn.execute("INSERT INTO dream_runs VALUES(NULL)")
    if lease:
        conn.execute("INSERT INTO run_lock VALUES('dreaming')")
    meter = SimpleNamespace(target_session=Mock())
    with pytest.raises(RuntimeError):
        gate.idle_and_session(conn, meter)
    meter.target_session.assert_not_called()
    conn.close()


def test_changed_session_has_no_unscoped_fallback(monkeypatch):
    from hymem.core import db
    monkeypatch.setattr(db, "schema_version", lambda _conn: 64)
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE dream_runs(ended_at TEXT); CREATE TABLE run_lock(name TEXT);")
    with pytest.raises(RuntimeError, match="target_session_changed"):
        gate.idle_and_session(conn, SimpleNamespace(target_session=lambda _c: "changed"))
    conn.close()


def test_receipt_is_exclusive_and_private(tmp_path):
    path = tmp_path / "result.json"
    gate.save(path, {"status": "failed"})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        gate.save(path, {"status": "completed"})


def test_unsealed_manifest_refuses_before_loading(tmp_path):
    path = tmp_path / "manifest.json"
    gate.save(path, {"version": "schema64-production-targeted-dream-v1"})
    with pytest.raises(RuntimeError, match="launch_contract_invalid"):
        gate.verify(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_journal_retains_no_provider_content():
    journal = gate.CounterJournal()
    journal.append("llm-events.jsonl", {"request": "private", "response": "private"})
    assert vars(journal) == {}


@dataclass
class Report:
    skipped_locked: bool = False
    chunks_processed: int = 1
    budget_exhausted: bool = True


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    """All app/client/deadline hooks are inert local fakes, including profile."""
    outbound = Mock(side_effect=AssertionError("offline_control_attempted_network"))
    monkeypatch.setattr(socket, "create_connection", outbound)
    monkeypatch.setattr(socket.socket, "connect", outbound)
    monkeypatch.setattr(socket.socket, "connect_ex", outbound)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE dream_runs(id INTEGER PRIMARY KEY, ended_at TEXT,
            error TEXT, skipped_locked INTEGER);
        CREATE TABLE run_lock(name TEXT);
        CREATE TABLE current_phase1_publications(chunk_id TEXT, phase1_generation_key TEXT);
        INSERT INTO dream_runs VALUES(99,'earlier-end',NULL,0);
    """)
    calls = []
    scenario = SimpleNamespace(owned_runs=1, skipped=False, shutdown=True,
                               llm_attempts=1, embedding_attempts=1)
    hy = SimpleNamespace(conn=conn, _phase1_generation={"generation_key": gate.GENERATION},
        _llm=SimpleNamespace(request_attempts=1, token_usage_available=True,
            prompt_tokens=10, completion_tokens=2, total_tokens=12),
        _embed=SimpleNamespace(request_attempts=1),
        dream_status=lambda: {"in_progress": False, "pending_digests": 1,
                              "pending_profiles": 1, "pending_facts": 1})

    def dream(*, session_ids, deadline):
        calls.append((session_ids, deadline))
        hy._llm.request_attempts = scenario.llm_attempts
        hy._embed.request_attempts = scenario.embedding_attempts
        for index in range(scenario.owned_runs):
            conn.execute("INSERT INTO dream_runs VALUES(?,'owned-end',NULL,0)", (100 + index,))
        conn.execute("INSERT INTO current_phase1_publications VALUES(?,?)", (gate.TARGET, gate.GENERATION))
        return Report(skipped_locked=scenario.skipped)

    hy.dream = dream
    bootstrap = ModuleType("hymem.bootstrap")
    bootstrap.build_from_env = Mock(return_value=hy)
    bootstrap.shutdown_instance = Mock(side_effect=lambda _hy: scenario.shutdown)
    monkeypatch.setitem(gate.sys.modules, "hymem.bootstrap", bootstrap)
    deadline_module = ModuleType("hymem.deadline")
    deadline_token = object()
    deadline_module.MonotonicDeadline = SimpleNamespace(after=Mock(return_value=deadline_token))
    monkeypatch.setitem(gate.sys.modules, "hymem.deadline", deadline_module)
    from hymem.core import db
    monkeypatch.setattr(db, "schema_version", lambda _conn: 64)
    session = "injected-offline-session"
    monkeypatch.setattr(gate, "SESSION_SHA", hashlib.sha256(session.encode()).hexdigest())
    meter = SimpleNamespace(TARGET_CHUNK=gate.TARGET, GENERATION=gate.GENERATION,
        MAX_COMPLETIONS=128, MAX_LLM_HTTP_ATTEMPTS=384,
        MAX_EMBEDDING_HTTP_ATTEMPTS=512, MAX_HTTP_ATTEMPTS=896,
        MAX_EMBEDDING_TEXTS=16, MAX_EMBEDDING_CHARS=128000,
        MAX_EMBEDDING_UTF8_BYTES=512000, target_session=lambda _conn: session,
        code_points=lambda: (None, None, None, None, None))
    probe = SimpleNamespace(profile=lambda *_args: None, completions=1,
        attempts=2, llm_attempts=1, embedding_attempts=1,
        budget_reason=None, capture_error=False)
    meter.Probe = Mock(return_value=probe)
    manifest = {"source": str(tmp_path), "stage": str(tmp_path), "meter": {"path": "/inert-meter"}}
    monkeypatch.setattr(gate, "verify", Mock(return_value=manifest))
    monkeypatch.setattr(gate, "runtime_env", Mock(return_value={}))
    monkeypatch.setattr(gate, "load_module", Mock(return_value=meter))
    monkeypatch.setattr(gate.sys, "setprofile", Mock())
    monkeypatch.setattr(gate.threading, "setprofile", Mock())
    monkeypatch.setattr(gate.logging, "disable", Mock())
    monkeypatch.setattr(gate.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(gate.AUTHORIZATION)))
    monkeypatch.setattr(gate, "save", Mock())
    original_path = gate.sys.path[:]
    try:
        yield SimpleNamespace(scenario=scenario, hy=hy, calls=calls, bootstrap=bootstrap,
                              deadline=deadline_module.MonotonicDeadline,
                              deadline_token=deadline_token, meter=meter, probe=probe)
    finally:
        outbound.assert_not_called()
        gate.sys.path[:] = original_path
        conn.close()


def receipt():
    gate.save.assert_called_once()
    path, result = gate.save.call_args.args
    assert path.name == "production-dream-result.json"
    return result


def test_worker_owned_run_accounting_and_shutdown(lifecycle):
    h = lifecycle
    assert gate.worker(Path("/inert-manifest"), "sealed") == 0
    assert h.calls == [(["injected-offline-session"], h.deadline_token)]
    h.deadline.after.assert_called_once_with(2700)
    h.bootstrap.build_from_env.assert_called_once_with()
    h.bootstrap.shutdown_instance.assert_called_once_with(h.hy)
    result = receipt()
    assert result["status"] == "completed"
    assert result["owned_run_id"] == 100
    assert result["cleanup_ok"] and result["accounting_verified"]
    assert (result["completion_calls"], result["llm_http_attempts"],
            result["embedding_http_attempts"], result["http_attempts"]) == (1, 1, 1, 2)
    assert result["total_tokens"] == 12
    assert result["report"]["budget_exhausted"] is True
    assert result["after"]["pending_digests"] == 1
    assert result["independent_repair_postflight_required"] is True
    assert result["session_convergence_verified"] is False
    # Probe and instance are injected; no production path or provider opened.
    gate.load_module.assert_called_once_with("production_reviewed_meter", "/inert-meter")


@pytest.mark.parametrize("failure", ["accounting", "shutdown", "ambiguous", "skipped"])
def test_worker_failure_is_durable_and_never_reruns(lifecycle, failure):
    h = lifecycle
    if failure == "accounting":
        h.scenario.llm_attempts = 2
    elif failure == "shutdown":
        h.scenario.shutdown = False
    elif failure == "ambiguous":
        h.scenario.owned_runs = 2
    else:
        h.scenario.skipped = True
    assert gate.worker(Path("/inert-manifest"), "sealed") == 1
    assert len(h.calls) == 1
    h.bootstrap.shutdown_instance.assert_called_once_with(h.hy)
    result = receipt()
    assert result["status"] == "failed"
    if failure == "shutdown":
        assert result["cleanup_ok"] is False
    if failure == "accounting":
        assert result["accounting_verified"] is False
    if failure in ("ambiguous", "skipped"):
        assert "owned_run_id" not in result
        assert result["error_code"] == "worker_failed_inspect_before_retry"


def test_sigterm_enters_owned_supervisor_cleanup_and_restores_handler(monkeypatch, tmp_path):
    import signal
    previous = object()
    active = {}
    restored = []
    def signal_hook(signum, handler):
        assert signum == signal.SIGTERM
        if not active:
            active["handler"] = handler
            return previous
        restored.append(handler)
        return active["handler"]
    outcome = SimpleNamespace(status="interrupted", safe_to_continue=False)
    cleanup = Mock()
    def invocation(*_args, **_kwargs):
        try:
            active["handler"](signal.SIGTERM, None)
        except KeyboardInterrupt:
            cleanup()
        return outcome
    monkeypatch.setattr(gate.signal, "signal", signal_hook)
    monkeypatch.setattr(gate, "verify", Mock(return_value={"source": str(tmp_path),
        "stage": str(tmp_path), "supervisor": {"path": "/inert-supervisor"}}))
    monkeypatch.setattr(gate, "runtime_env", Mock(return_value={}))
    monkeypatch.setattr(gate, "load_module", Mock(return_value=SimpleNamespace(supervise_invocation=invocation)))
    monkeypatch.setattr(gate, "asdict", lambda _outcome: {"status": "interrupted"})
    monkeypatch.setattr(gate, "save", Mock())
    assert gate.supervise(Path("/inert-manifest"), "sealed") == 1
    cleanup.assert_called_once_with()
    assert restored == [previous]
    gate.save.assert_called_once()
