"""No provider, network, SSH, or production credential access."""
from __future__ import annotations

import base64
import concurrent.futures
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import pytest

from benchmarks import hermes_lme_oauth_v1 as bridge
from hymem.extraction.llm import LLMRequest


def _token(account="acct-1", email="person@example.invalid", exp=None):
    payload = {"exp": exp or time.time() + 1000, "email": email,
               "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    middle = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return "header." + middle + ".signature"


def _auth(path: Path, *, account="acct-1", exp=None, mode="chatgpt", key=None):
    data = {"auth_mode": mode, "tokens": {"access_token": _token(account, exp=exp),
            "account_id": account, "refresh_token": "private-refresh"}}
    if key is not None:
        data["OPENAI_API_KEY"] = key
    path.write_text(json.dumps(data))
    path.chmod(0o600)


class FakeSession:
    created = []
    calls = []
    refresh = None
    account = {"type": "chatgpt", "planType": "plus", "email": "person@example.invalid"}
    quota = {"rateLimits": {"planType": "plus", "primary": {"usedPercent": 0}}}

    def __init__(self, binary, cwd, timeout=120):
        self.created_at = time.monotonic()
        self.closed = False
        self.deadline = None
        self.__class__.created.append(self)

    def set_deadline(self, deadline):
        self.deadline = deadline

    def send(self, method, params, notification=False):
        assert method == "initialized" and notification
        self.__class__.calls.append(method)

    def rpc(self, method, params):
        assert method in {"initialize", "account/read", "model/list", "account/rateLimits/read"}
        self.__class__.calls.append((method, params))
        if method == "initialize":
            return {"capabilities": {}}
        if method == "account/read":
            if params["refreshToken"] and self.__class__.refresh:
                self.__class__.refresh()
            return {"account": self.__class__.account}
        if method == "model/list":
            return {"data": [{"model": "gpt-6-luna", "supportedReasoningEfforts":
                     [{"reasoningEffort": "low"}]}], "nextCursor": None}
        return self.__class__.quota

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake():
    FakeSession.created = []
    FakeSession.calls = []
    FakeSession.refresh = None
    FakeSession.account = {"type": "chatgpt", "planType": "plus", "email": "person@example.invalid"}
    FakeSession.quota = {"rateLimits": {"planType": "plus", "primary": {"usedPercent": 0}}}


def _setup(tmp_path, *, campaign_turns=8):
    path = tmp_path / "auth.json"
    _auth(path)
    broker = bridge.AdmissionBroker("fake-codex", str(path), session_factory=FakeSession)
    limits = bridge.warm.BudgetLimits(campaign_turns, 1000, 120)
    budget = bridge.warm.SharedBudget(limits, max_in_flight=4)
    return path, broker, budget


def _request():
    return LLMRequest(system="sys", user="user", max_tokens=123, temperature=0.3)


def _completed(text="answer"):
    return bridge.native.Completed(text, 3, 2, 5, 0, 0)


def test_broker_lazy_serialized_and_direct_only(tmp_path):
    _, broker, _ = _setup(tmp_path)
    assert not FakeSession.created
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: broker.admit(time.monotonic() + 5), range(4)))
        assert len(FakeSession.created) == 1
        assert all(c.account_id == "acct-1" and a["isolation_basis"] ==
                   "direct_fixed_https_no_agent_thread" for c, a in results)
        assert FakeSession.calls.count("initialized") == 1
        assert len([c for c in FakeSession.calls if isinstance(c, tuple) and c[0] == "initialize"]) == 1
        assert not any(isinstance(c, tuple) and c[0].startswith(("thread/", "turn/"))
                       for c in FakeSession.calls)
    finally:
        broker.close()
    assert FakeSession.created[0].closed


def test_auth_file_security_and_refresh_continuity(tmp_path):
    path, broker, _ = _setup(tmp_path)
    _auth(path, exp=time.time() + 1)
    FakeSession.refresh = lambda: _auth(path, exp=time.time() + 1000)
    credentials, _ = broker.admit(time.monotonic() + 5)
    assert credentials.account_id == "acct-1"
    assert ("account/read", {"refreshToken": True}) in FakeSession.calls
    broker.close()

    path, broker, _ = _setup(tmp_path)
    _auth(path, exp=time.time() + 1)
    FakeSession.refresh = lambda: _auth(path, account="acct-2")
    with pytest.raises(bridge.BridgeError, match="^account_mismatch$"):
        broker.admit(time.monotonic() + 5)
    broker.close()

    path, broker, _ = _setup(tmp_path)
    path.chmod(0o644)
    with pytest.raises(bridge.BridgeError, match="^auth_file_untrusted$"):
        broker.admit(time.monotonic() + 5)
    broker.close()

    target = tmp_path / "target.json"
    _auth(target)
    symlink = tmp_path / "link.json"
    symlink.symlink_to(target)
    broker = bridge.AdmissionBroker("fake", str(symlink), session_factory=FakeSession)
    with pytest.raises(bridge.BridgeError, match="^auth_file_untrusted$"):
        broker.admit(time.monotonic() + 5)
    broker.close()


def test_nullable_api_key_is_absent_but_real_key_is_rejected(tmp_path):
    path, broker, _ = _setup(tmp_path)
    auth = json.loads(path.read_text())
    auth["OPENAI_API_KEY"] = None
    path.write_text(json.dumps(auth))
    path.chmod(0o600)
    broker.admit(time.monotonic() + 5)
    broker.close()
    auth["OPENAI_API_KEY"] = "invented-real-key"
    path.write_text(json.dumps(auth))
    path.chmod(0o600)
    broker = bridge.AdmissionBroker("fake", str(path), session_factory=FakeSession)
    with pytest.raises(bridge.BridgeError, match="^invalid_credentials$"):
        broker.admit(time.monotonic() + 5)
    broker.close()


def test_production_auth_path_must_match_managed_home(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(managed))
    with pytest.raises(ValueError, match="^managed_auth_path_mismatch$"):
        bridge.AdmissionBroker("unused", str(tmp_path / "other-auth.json"))
    # Constructor performs only path binding; no managed process or credential read.
    broker = bridge.AdmissionBroker("unused", str(managed / "auth.json"))
    broker.close()


def test_nested_profile_email_binds_managed_account(tmp_path):
    path, broker, _ = _setup(tmp_path)
    claims = {"exp": time.time() + 1000,
              "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"},
              "https://api.openai.com/profile": {"email": "person@example.invalid"}}
    token = "header." + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=") + ".signature"
    data = json.loads(path.read_text())
    data["tokens"]["access_token"] = token
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    broker.admit(time.monotonic() + 5)
    broker.close()
    del claims["https://api.openai.com/profile"]
    data["tokens"]["access_token"] = "header." + base64.urlsafe_b64encode(
        json.dumps(claims).encode()).decode().rstrip("=") + ".signature"
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    broker = bridge.AdmissionBroker("fake", str(path), session_factory=FakeSession)
    with pytest.raises(bridge.BridgeError, match="^account_unverified$"):
        broker.admit(time.monotonic() + 5)
    broker.close()


def test_managed_watchdog_kills_captured_blocked_metadata_group(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    path = managed / "auth.json"
    _auth(path)
    monkeypatch.setenv("CODEX_HOME", str(managed))
    released = threading.Event()
    class Process:
        pid = 24680
        def poll(self):
            return None
    class Blocked(FakeSession):
        process = Process()
        def rpc(self, method, params):
            released.wait(timeout=1)
            raise OSError("blocked pipe released")
    monkeypatch.setattr(bridge.warm, "WarmSession", Blocked)
    killed = []
    def killpg(pid, sig):
        killed.append((pid, sig))
        released.set()
    monkeypatch.setattr(bridge.os, "killpg", killpg)
    broker = bridge.AdmissionBroker("fake", str(path), session_factory=Blocked)
    with pytest.raises(bridge.BridgeError, match="^timeout$"):
        broker.admit(time.monotonic() + 0.04)
    assert killed == [(24680, bridge.signal.SIGKILL)]
    assert FakeSession.created[-1].closed
    broker.close()


def test_near_expired_startup_never_reaches_http(tmp_path):
    path, _, _ = _setup(tmp_path)
    class Slow(FakeSession):
        def __init__(self, binary, cwd, timeout):
            super().__init__(binary, cwd, timeout)
            time.sleep(0.04)
    broker = bridge.AdmissionBroker("fake", str(path), session_factory=Slow)
    with pytest.raises(bridge.BridgeError, match="^timeout$"):
        broker.admit(time.monotonic() + 0.02)
    assert FakeSession.created[-1].closed
    broker.close()


def test_quota_and_account_stop_before_http(tmp_path):
    _, broker, budget = _setup(tmp_path)
    client = bridge.NativeLMEClient(broker, budget, "q", bridge.warm.BudgetLimits(2, 100, 120),
                                    transport=lambda *a, **k: pytest.fail("HTTP dispatched"))
    FakeSession.quota = {"rateLimits": {"planType": "plus", "primary": {"usedPercent": 90}}}
    with pytest.raises(bridge.BridgeError):
        client.complete(_request())
    assert budget.snapshot()["turns"] == 0
    assert budget.snapshot()["in_flight"] == 0
    broker.close()

    _, broker, budget = _setup(tmp_path)
    FakeSession.account = {"type": "api", "planType": "plus"}
    client = bridge.NativeLMEClient(broker, budget, "q", bridge.warm.BudgetLimits(2, 100, 120),
                                    transport=lambda *a, **k: pytest.fail("HTTP dispatched"))
    with pytest.raises(bridge.BridgeError, match="^account_unverified$"):
        client.complete(_request())
    assert budget.snapshot()["turns"] == 0
    broker.close()


def test_four_workers_shared_budget_and_privacy(tmp_path):
    _, broker, budget = _setup(tmp_path, campaign_turns=4)
    seen = []
    lock = threading.Lock()

    def transport(credentials, system, user, schema, timeout):
        with lock:
            seen.append((schema, timeout))
        return _completed()

    clients = [bridge.NativeLMEClient(broker, budget, f"q{i}",
               bridge.warm.BudgetLimits(1, 100, 120), transport=transport) for i in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda c: c.complete(_request()), clients)) == ["answer"] * 4
    state = budget.snapshot()
    assert state["turns"] == 4 and state["known_tokens"] == 20 and state["in_flight"] == 0
    assert len(seen) == 4 and all(0 < timeout <= 120 for _, timeout in seen)
    assert all(c.successes == 1 and c.failures == 0 for c in clients)
    assert all(c.requested_controls[0]["max_tokens_effective"] is None for c in clients)
    observed = repr((state, clients[0].requested_controls, broker))
    assert all(secret not in observed for secret in ("private-refresh", "person@example.invalid", "acct-1", "header."))
    broker.close()


def test_transport_failure_settles_unknown_usage_and_halts(tmp_path):
    _, broker, budget = _setup(tmp_path)
    def fail(*args, **kwargs):
        raise bridge.native.TransportError("http_failure")
    client = bridge.NativeLMEClient(broker, budget, "q", bridge.warm.BudgetLimits(2, 100, 120), transport=fail)
    with pytest.raises(bridge.BridgeError, match="^http_failure$"):
        client.complete(_request())
    state = budget.snapshot()
    assert state["turns"] == 1 and state["known_tokens"] == 0
    assert state["usage_complete"] is False and state["stop_code"] == "http_failure"
    assert client.last_failure_code == "http_failure"
    assert client.requested_controls[0]["output_schema_acknowledged"] is False
    summary = client.diagnostic_summary()
    assert summary["schema"] == "native_oauth_summary_v1"
    assert summary["calls"] == summary["failures"] == 1 and summary["successes"] == 0
    assert summary["turns"] == 1 and summary["known_tokens"] == 0
    assert summary["first_failure"] == {"code": "http_failure", "phase": "http",
                                        "turn_admitted": True, "known_usage": False}
    assert all(type(value) is float and 0 <= value <= 1_000_000
               for value in summary["timing_seconds"].values())
    broker.close()


def test_credit_window_identity_and_broker_account_pin(tmp_path):
    assert bridge.warm is bridge.staged_v6.warm
    path, broker, budget = _setup(tmp_path)
    FakeSession.quota = {"rateLimits": {"planType": "plus", "primary": {"usedPercent": 90},
                         "credits": {"hasCredits": True, "unlimited": False,
                                     "balance": "10"}}}
    # Credit fields are interpreted only by the source-pinned quota parser.
    client = bridge.NativeLMEClient(broker, budget, "q", bridge.warm.BudgetLimits(2, 100, 120),
                                    transport=lambda *a, **k: _completed())
    assert client.complete(_request()) == "answer"
    # A new valid local token cannot silently switch the broker's account.
    _auth(path, account="acct-2")
    with pytest.raises(bridge.BridgeError, match="^account_mismatch$"):
        broker.admit(time.monotonic() + 5)
    broker.close()


def test_staged_schema_uses_exact_contract_and_one_registration(tmp_path):
    _, broker, budget = _setup(tmp_path)
    source = bridge.staged_v6.classification.GroundingSource(
        7, "Mira uses CairnDB.", source_role="user", source_peer_id="invented",
        source_created_at="2026-09-30")
    triple = bridge.staged_v6.classification.Triple(
        "Mira", "uses", "CairnDB", 1, source_message_id=7)
    request, batch = bridge.staged_v6.staged.build_original_request((triple,), (source,))
    schemas = []
    def transport(credentials, system, user, schema, timeout):
        schemas.append(schema)
        return _completed("{}")
    client = bridge.NativeLMEClient(broker, budget, "q", bridge.warm.BudgetLimits(2, 100, 120),
                                    transport=transport)
    assert client.complete_stage(request, batch, "original", False) == "{}"
    assert schemas == [bridge.staged_v6.staged.build_original_output_schema(batch)]
    assert client.requested_controls[0]["output_schema_sent"] is True
    assert client.requested_controls[0]["output_schema_acknowledged"] is True
    assert budget.snapshot()["questions"]["q"]["turns"] == 1
    raw_original = json.dumps({"schema": bridge.staged_v6.staged.ORIGINAL_SCHEMA,
        "batch_sha256": batch.batch_sha256, "complete": True,
        "originals": [{"index": 0, "original": {"state": "not_established", "support": None}}]})
    alt_request, alt_batch = bridge.staged_v6.staged.build_alternatives_request(batch, raw_original)
    assert client.complete_stage(alt_request, alt_batch, "alternatives", False) == "{}"
    assert schemas[1] == bridge.staged_v6.staged.build_alternatives_output_schema(alt_batch)
    assert budget.snapshot()["questions"]["q"]["turns"] == 2
    assert budget.snapshot()["questions"]["q"]["known_tokens"] == 10
    with pytest.raises(bridge.staged_v6.staged.GroundingContractError):
        client.complete_stage(request, batch, "alternatives", False)
    assert budget.snapshot()["questions"]["q"]["turns"] == 2
    broker.close()
