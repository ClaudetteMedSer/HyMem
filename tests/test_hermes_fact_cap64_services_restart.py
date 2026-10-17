"""No-signal fixtures for the guarded cap-64 Honcho/gateway activation."""

import hashlib
import json
import signal
import sys
from datetime import datetime, timezone

import pytest

from tools.diagnostics import hermes_fact_cap64_services_restart as operator


def _records():
    old_honcho = operator.quiesce.Process(21, "hymem-honcho", 121)
    old_mcp = operator.quiesce.Process(22, "hymem-server", 122)
    old_gateway = operator.restart.Gateway(120, 220)
    new_honcho = operator.quiesce.Process(31, "hymem-honcho", 131)
    new_mcp = operator.quiesce.Process(32, "hymem-server", 132)
    new_gateway = operator.restart.Gateway(130, 230)
    return old_honcho, old_mcp, old_gateway, new_honcho, new_mcp, new_gateway


def test_gateway_requires_pinned_entrypoint_pid_and_command(tmp_path, monkeypatch):
    gateway = operator.restart.Gateway(120, 220)
    gw = tmp_path / "120"
    parent = tmp_path / "7"
    gw.mkdir()
    parent.mkdir()
    (gw / "stat").write_text("120 (hermes) S 7 " + "0 " * 19 + "220 0\n")
    (parent / "cmdline").write_bytes(b"bash\0/usr/local/bin/entrypoint.sh\0")
    monkeypatch.setattr(operator.restart, "_gateway_identity", lambda *_a: gateway)
    operator._gateway_entrypoint(gateway, tmp_path)
    (parent / "cmdline").write_bytes(b"bash\0unreviewed\0")
    with pytest.raises(operator.Cap64RestartStopped, match="entrypoint"):
        operator._gateway_entrypoint(gateway, tmp_path)


def test_runtime_status_requires_fresh_zero_active_work(tmp_path):
    gateway = operator.restart.Gateway(120, 220)
    status = tmp_path / "gateway_state.json"
    payload = {"pid": 120, "gateway_state": "running", "active_agents": 0,
               "restart_requested": False,
               "updated_at": datetime.now(timezone.utc).isoformat()}
    status.write_text(json.dumps(payload))
    operator._status_idle(gateway, status)
    payload["active_agents"] = 1
    status.write_text(json.dumps(payload))
    with pytest.raises(operator.Cap64RestartStopped, match="freshly idle"):
        operator._status_idle(gateway, status)
    payload["active_agents"] = 0
    payload["updated_at"] = "2020-01-01T00:00:00+00:00"
    status.write_text(json.dumps(payload))
    with pytest.raises(operator.Cap64RestartStopped, match="freshly idle"):
        operator._status_idle(gateway, status)


def test_keepalive_config_requires_exact_hash_and_semantic_60(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    payload = b"mcp_servers:\n  hymem:\n    keepalive_interval: 60\nprivate: fixture-secret\n"
    config.write_bytes(payload)
    config.chmod(0o600)
    monkeypatch.setattr(operator, "EXPECTED_MCP_CONFIG_SHA256", hashlib.sha256(payload).hexdigest())
    operator._keepalive_config_ready(config)
    altered = payload.replace(b"interval: 60", b"interval: 30")
    config.write_bytes(altered)
    with pytest.raises(operator.Cap64RestartStopped, match="differs"):
        operator._keepalive_config_ready(config)
    monkeypatch.setattr(operator, "EXPECTED_MCP_CONFIG_SHA256", hashlib.sha256(altered).hexdigest())
    with pytest.raises(operator.Cap64RestartStopped, match="not 60"):
        operator._keepalive_config_ready(config)


def test_new_process_requires_one_exact_fact_token_override(tmp_path, monkeypatch):
    record = operator.quiesce.Process(31, "hymem-honcho", 131)
    directory = tmp_path / "31"
    directory.mkdir()
    environment = directory / "environ"
    monkeypatch.setattr(operator.quiesce, "_same_process", lambda *_a: True)
    environment.write_bytes(b"OTHER=private\0HYMEM_FACTS_EXTRACTION_MAX_TOKENS=6144\0")
    assert operator._token_in_process(record, 6144, tmp_path)
    environment.write_bytes(b"HYMEM_FACTS_EXTRACTION_MAX_TOKENS=6144\0"
                            b"HYMEM_FACTS_EXTRACTION_MAX_TOKENS=6144\0")
    assert not operator._token_in_process(record, 6144, tmp_path)
    environment.write_bytes(b"HYMEM_FACTS_EXTRACTION_MAX_TOKENS=3072\0")
    assert not operator._token_in_process(record, 6144, tmp_path)


def test_preview_reports_three_exact_pins_without_signaling(monkeypatch):
    honcho, mcp, gateway, *_ = _records()
    monkeypatch.setattr(operator, "_preflight", lambda **_kwargs: (b"block", honcho, mcp, gateway))
    monkeypatch.setattr(operator, "_terminate_gateway", lambda *_args: pytest.fail("signaled"))
    result = operator.preview()
    assert result == {"cap": 64, "restarted": False,
                      "honcho": {"pid": 21, "starttime": 121},
                      "mcp": {"pid": 22, "starttime": 122},
                      "gateway": {"pid": 120, "starttime": 220}}


def test_mismatched_gateway_pin_refuses_before_first_signal(monkeypatch):
    honcho, mcp, gateway, *_ = _records()
    monkeypatch.setattr(operator, "_preflight", lambda **_kwargs: (b"block", honcho, mcp, gateway))
    monkeypatch.setattr(operator.restart, "_terminate_exact", lambda *_args: pytest.fail("signaled"))
    with pytest.raises(operator.Cap64RestartStopped, match="PID/starttime"):
        operator.service_restart(honcho, mcp, operator.restart.Gateway(120, 221))


def test_exact_honcho_then_gateway_restart_without_mcp_only_signal(monkeypatch):
    honcho, mcp, gateway, new_honcho, new_mcp, new_gateway = _records()
    events = []
    gateway_signaled = False
    new_parent_attempts = 0
    monkeypatch.setattr(operator, "_preflight", lambda **_kwargs: (b"block", honcho, mcp, gateway))
    monkeypatch.setattr(operator.quiesce, "_dream_idle", lambda _db: events.append("lease"))
    monkeypatch.setattr(operator.restart, "_terminate_exact",
                        lambda record, _proc: events.append("honcho-signal") if record == honcho
                        else pytest.fail("MCP child was signaled"))
    monkeypatch.setattr(operator.restart, "_exited", lambda *_a: True)
    monkeypatch.setattr(operator.restart, "_port_closed", lambda *_a: True)
    monkeypatch.setattr(operator.restart, "_recover_honcho",
                        lambda *_a, **_k: events.append("honcho-relaunch") or new_honcho)
    def bounded_wait(predicate, _seconds, reason):
        for _ in range(2):
            if predicate():
                return
        pytest.fail(reason)
    monkeypatch.setattr(operator.restart, "_wait_until", bounded_wait)
    monkeypatch.setattr(operator, "_gateway_entrypoint", lambda *_a: None)
    monkeypatch.setattr(operator, "_gateways",
                        lambda _proc: [new_gateway] if gateway_signaled else [gateway])
    monkeypatch.setattr(operator.restart, "_role_processes",
                        lambda role, _proc: ([new_honcho] if role == "hymem-honcho" else
                                             [new_mcp] if gateway_signaled else [mcp]))
    def parent(record, _proc):
        nonlocal new_parent_attempts
        if record == new_mcp:
            new_parent_attempts += 1
            if new_parent_attempts == 1:
                events.append("transient-parent-race")
                raise operator.restart.RestartStopped("transient child startup")
            return new_gateway
        return gateway
    monkeypatch.setattr(operator.restart, "_gateway_parent", parent)
    monkeypatch.setattr(operator.restart, "_cap_in_process",
                        lambda record, cap, _proc: (record in (honcho, mcp) and cap == 32)
                        or (record in (new_honcho, new_mcp) and cap == 64))
    monkeypatch.setattr(operator, "_token_in_process",
                        lambda record, tokens, _proc: record in (new_honcho, new_mcp)
                        and tokens == 6144)
    monkeypatch.setattr(operator.files, "_no_token_override", lambda *_a: None)
    monkeypatch.setattr(operator, "_new_files_ready", lambda *_a: events.append("files"))
    monkeypatch.setattr(operator, "_status_idle", lambda *_a: events.append("status"))
    monkeypatch.setattr(operator, "_no_notice_markers", lambda *_a: events.append("markers"))
    monkeypatch.setattr(operator, "_keepalive_config_ready", lambda *_a: events.append("config"))
    monkeypatch.setattr(operator.restart, "_health", lambda *_a: True)
    monkeypatch.setattr(operator.restart, "_mcp_protocol_probe",
                        lambda: events.append("protocol-probe"))
    monkeypatch.setattr(operator.restart, "_observe_gateway_child",
                        lambda *_a: events.append("stable"))
    monkeypatch.setattr(operator, "_gateway_exited", lambda *_a: gateway_signaled)

    def signal_gateway(_gateway, _proc):
        nonlocal gateway_signaled
        events.append("gateway-signal")
        gateway_signaled = True
    monkeypatch.setattr(operator, "_terminate_gateway", signal_gateway)

    result = operator.service_restart(honcho, mcp, gateway)
    assert result["restarted"] and result["gateway"]["new_pid"] == 130
    assert result["mcp"]["new_pid"] == 32
    assert result["fact_tokens"] == 6144
    assert events.index("honcho-signal") < events.index("honcho-relaunch") < events.index("gateway-signal")
    assert events.index("gateway-signal") < events.index("protocol-probe") < events.index("stable")
    assert events.index("transient-parent-race") < events.index("protocol-probe")
    assert events.count("lease") == 3


def test_gateway_pidfd_rechecks_identity_and_sends_only_sigterm(monkeypatch):
    gateway = operator.restart.Gateway(120, 220)
    calls = []
    monkeypatch.setattr(operator, "_gateway_entrypoint",
                        lambda *_a: calls.append("identity"))
    monkeypatch.setattr(operator.os, "pidfd_open", lambda pid: calls.append(("open", pid)) or 42)
    monkeypatch.setattr(operator.signal, "pidfd_send_signal",
                        lambda fd, sig: calls.append(("signal", fd, sig)))
    monkeypatch.setattr(operator.os, "close", lambda fd: calls.append(("close", fd)))
    operator._terminate_gateway(gateway, operator.PROC)
    assert calls == ["identity", ("open", 120), "identity",
                     ("signal", 42, signal.SIGTERM), ("close", 42)]


def test_cli_refuses_nonprivate_stage_before_any_signal(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o755)
    stage.chmod(0o755)
    monkeypatch.setattr(operator, "STAGE", stage)
    monkeypatch.setattr(sys, "argv", ["cap64-restart", "--apply", "--allow-live-signals",
                                         "--expected-honcho", "21:121",
                                         "--expected-mcp", "22:122",
                                         "--expected-gateway", "120:220"])
    monkeypatch.setattr(operator, "service_restart", lambda *_a: pytest.fail("signaled"))
    with pytest.raises(SystemExit, match="2"):
        operator.main()
