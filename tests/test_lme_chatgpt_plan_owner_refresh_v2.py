"""Offline deadline/lease-horizon regression with wholly invented grants."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest import mock

import pytest

from tools.diagnostics import lme_chatgpt_plan_owner_v2 as owner
from tools.diagnostics import lme_chatgpt_plan_refresh_v2 as refresh


HOST = "urn:uuid:00000000-0000-4000-8000-000000000002"
ORIGIN = "urn:uuid:00000000-0000-4000-8000-000000000001"
CLIENT = "oaiapp_invented123"
SCOPES = ["openid", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"]
PYTHON = Path("/opt/anaconda3/bin/python3.13")


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _state(root, *, lifetime):
    state = root / "private"
    state.mkdir(mode=0o700)
    now = int(time.time())
    _write(state / "host.json", {"ext_agent_host_id": HOST})
    _write(state / "registration.json", {"host_id": HOST, "origin_host_id": ORIGIN,
                                         "client_id": CLIENT, "registration_metadata": "invented"})
    _write(state / "status.json", {"status": "complete", "plan_usage_enabled": True, "model_calls": 0})
    _write(state / "credential.json", {
        "version": 1, "issuer": "https://auth.openai.com", "host_id": HOST,
        "origin_host_id": ORIGIN, "client_id": CLIENT, "subject": "invented-subject",
        "access_token": "invented-access-old", "refresh_token": "invented-refresh-old",
        "id_token": "invented-id-old", "token_type": "Bearer", "scope": SCOPES,
        "saved_at": now - 100, "expires_at": now + lifetime,
        "plan_usage_enabled": True,
    })
    return state


def _exchange(url, form):
    assert url == "https://auth.openai.com/api/accounts/oauth/token"
    assert form == {"grant_type": "refresh_token", "client_id": CLIENT,
                    "refresh_token": "invented-refresh-old",
                    "resource": "https://api.openai.com/v1"}
    return {"token_type": "Bearer", "expires_in": 3600,
            "access_token": "invented-access-new", "refresh_token": "invented-refresh-new"}


def _local_refresh_child(state, calls):
    """Exercise the real owner response parser and real refresh.run without a child network."""
    def spawn(command, **kwargs):
        assert command[3] == str(Path(refresh.__file__))
        assert command[4:] == ["refresh", "--state-dir", str(state)]
        assert kwargs["start_new_session"] is True
        result = refresh.run(state)
        calls.append(result["status"])
        class Child:
            returncode = 0 if result["status"] != "failed" else 1
            def communicate(self, timeout):
                assert 0 < timeout <= 35
                return json.dumps(result).encode(), b""
        return Child()
    return spawn


def test_exact_source_pin_and_historical_transfer_route():
    source = Path(refresh.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == owner.REFRESH_SHA256
    assert owner._pinned_refresh().run.__code__.co_filename == str(source)
    assert owner.REMOTE_SOURCE == Path(
        "/home/atta/.hymem-siwc-owner-v1/tools/diagnostics/lme_chatgpt_plan_owner_v1.py")
    old = Path(owner.__file__).with_name("lme_chatgpt_plan_owner_v1.py").read_text()
    new = Path(owner.__file__).read_text()
    for unchanged in ("min(timeout, 35)",
                      'not 0 < timeout <= 120',
                      'deadline: float = 120',
                      'not 0 < deadline <= 120'):
        assert old.count(unchanged) == new.count(unchanged)


def test_real_broker_accepts_300_policy_but_rejects_overbound_and_expired():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = _state(Path(tmp), lifetime=3600)
        with owner.CredentialBroker(state, PYTHON) as broker:
            lease = broker.acquire(caller_deadline=time.monotonic() + 299)
            assert lease.access_token == "invented-access-old"
            assert "invented" not in repr(lease)
            for offset in (301, -1):
                with pytest.raises(owner.OwnerError) as caught:
                    broker.acquire(caller_deadline=time.monotonic() + offset)
                assert caught.value.code == "deadline_exceeded"


def test_330_second_lease_rotates_once_for_300_second_caller():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = _state(Path(tmp), lifetime=330)
        with owner.CredentialBroker(state, PYTHON) as broker:
            # The subprocess seam stays local; acquire, owner refresh-result
            # validation, and refresh.run all execute unmodified.
            rotations = []
            with mock.patch.object(owner.subprocess, "Popen", side_effect=_local_refresh_child(state, rotations)), \
                    mock.patch.object(refresh, "_http_json", side_effect=_exchange) as network:
                lease = broker.acquire(caller_deadline=time.monotonic() + 299)
                again = broker.acquire(caller_deadline=time.monotonic() + 299)
            assert lease.access_token == again.access_token == "invented-access-new"
            assert rotations == ["refreshed"] and network.call_count == 1
            saved = json.loads((state / "credential.json").read_text())
            assert saved["refresh_token"] == "invented-refresh-new"
            assert len(list(state.glob(".refresh-attempt-*.json"))) == 1


def test_refresh_horizon_360_and_not_due_above_it():
    for lifetime, expected in ((330, "due"), (360, "due"), (361, "not_due")):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            state = _state(Path(tmp), lifetime=lifetime)
            before = (state / "credential.json").read_bytes()
            with mock.patch.object(refresh, "_http_json") as network:
                result = refresh.run(state, check_only=True)
                assert result["status"] == expected
                if expected == "not_due":
                    result = refresh.run(state)
                    assert result["status"] == "not_due"
                network.assert_not_called()
            assert (state / "credential.json").read_bytes() == before


def test_failed_rotation_consumes_generation_and_stops_broker_without_retry():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = _state(Path(tmp), lifetime=330)
        before = (state / "credential.json").read_bytes()
        with owner.CredentialBroker(state, PYTHON) as broker:
            rotations = []
            with mock.patch.object(owner.subprocess, "Popen", side_effect=_local_refresh_child(state, rotations)), \
                    mock.patch.object(refresh, "_http_json", side_effect=refresh.RefreshError("oauth_invalid_grant")) as network:
                with pytest.raises(owner.OwnerError) as caught:
                    broker.acquire(caller_deadline=time.monotonic() + 299)
                assert caught.value.code == "refresh_denied"
                with pytest.raises(owner.OwnerError) as second:
                    broker.acquire(caller_deadline=time.monotonic() + 299)
                assert second.value.code == "refresh_unknown"
                assert network.call_count == 1
                assert rotations == ["failed"]
        assert (state / "credential.json").read_bytes() == before
        assert len(list(state.glob(".refresh-attempt-*.json"))) == 1
