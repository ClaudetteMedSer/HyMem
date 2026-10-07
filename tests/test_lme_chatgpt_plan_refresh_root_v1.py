"""Independent refresh controls. Invented local grant, mocked auth endpoint."""
import json
import os
from pathlib import Path
import tempfile
import time

import pytest

from tools.diagnostics import lme_chatgpt_plan_refresh_v1 as r


def old_grant():
    now = int(time.time())
    return {"version": 1, "issuer": "https://auth.openai.com", "host_id":
        "urn:uuid:00000000-0000-4000-8000-000000000001", "client_id": "oaiapp_invented123",
        "subject": "invented-subject", "access_token": "invented-old-access",
        "refresh_token": "invented-old-refresh", "id_token": "invented-old-id",
        "token_type": "Bearer", "scope": sorted(r.REQUIRED_SCOPES),
        "saved_at": now - 3601, "expires_at": now - 1, "plan_usage_enabled": True}


def response():
    return {"access_token": "invented-new-access", "refresh_token": "invented-new-refresh",
            "token_type": "Bearer", "expires_in": 3600}


@pytest.fixture
def state():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
        path = Path(name)
        old = old_grant()
        entries = {"credential.json": old,
            "host.json": {"ext_agent_host_id": old["host_id"]},
            "registration.json": {"host_id": old["host_id"], "client_id": old["client_id"]},
            "status.json": {"status": "complete", "plan_usage_enabled": True, "model_calls": 0}}
        for filename, data in entries.items():
            file = path / filename
            file.write_text(json.dumps(data))
            file.chmod(0o600)
        yield path, old


def test_readonly_check_has_no_network_or_write(state, monkeypatch):
    path, _ = state
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    monkeypatch.setattr(r, "_http_json", lambda *args: pytest.fail("network"))
    result = r.run(path, check_only=True)
    assert result["status"] == "due" and result["model_calls"] == 0
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}


def test_one_serialized_exact_refresh_preserves_binding(state, monkeypatch):
    path, old = state
    calls = []
    def fake(url, form=None):
        calls.append((url, form))
        return response()
    monkeypatch.setattr(r, "_http_json", fake)
    result = r.run(path)
    assert result["status"] == "refreshed" and result["refresh_requests"] == 1
    assert result["model_calls"] == 0
    assert calls == [("https://auth.openai.com/api/accounts/oauth/token", {
        "grant_type": "refresh_token", "client_id": old["client_id"],
        "refresh_token": old["refresh_token"], "resource": "https://api.openai.com/v1"})]
    new = json.loads((path / "credential.json").read_text())
    for field in ("host_id", "client_id", "subject", "scope", "id_token", "issuer"):
        assert new[field] == old[field]
    assert new["refresh_token"] == "invented-new-refresh"
    assert (path / "credential.json").stat().st_mode & 0o777 == 0o600
    assert r.run(path)["status"] == "not_due" and len(calls) == 1
    assert "invented" not in json.dumps(result)


def test_ambiguous_generation_not_retried_after_metadata_change(state, monkeypatch):
    path, old = state
    calls = []
    def broken(*args):
        calls.append(1)
        raise r.RefreshError("upstream_unavailable")
    monkeypatch.setattr(r, "_http_json", broken)
    result = r.run(path)
    assert result["status"] == "failed" and len(calls) == 1
    assert json.loads((path / "credential.json").read_text()) == old
    old["email"] = "invented@example.invalid"
    (path / "credential.json").write_text(json.dumps(old))
    assert r.run(path)["error"] == "generation_consumed"
    assert len(calls) == 1


def test_held_flow_lock_blocks_refresh(state, monkeypatch):
    path, _ = state
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    lock = r._lock(fd)
    try:
        monkeypatch.setattr(r, "_http_json", lambda *args: pytest.fail("network"))
        result = r.run(path)
        assert result["error"] == "flow_already_running"
        assert result["refresh_requests"] == 0
    finally:
        os.close(lock)
        os.close(fd)


@pytest.mark.parametrize("body", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'])
def test_json_nonfinite_and_duplicates_rejected(body):
    with pytest.raises(r.RefreshError):
        r._json(body)


@pytest.mark.parametrize("change", [
    {"access_token": " "}, {"refresh_token": " "},
    {"scope": "openid offline_access resource.invoke chatgpt.tokens.use.direct additional"},
    {"expires_in": True}, {"expires_in": 0}, {"token_type": "Basic"},
])
def test_bad_replacement_never_replaces_credential(state, monkeypatch, change):
    path, old = state
    reply = response()
    reply.update(change)
    monkeypatch.setattr(r, "_http_json", lambda *args: reply)
    result = r.run(path)
    assert result["status"] == "failed"
    assert json.loads((path / "credential.json").read_text()) == old
    assert r.run(path)["error"] == "generation_consumed"


def test_changed_record_during_network_not_overwritten(state, monkeypatch):
    path, old = state
    changed = dict(old, subject="different-invented-subject")
    def fake(*args):
        (path / "credential.json").write_text(json.dumps(changed))
        return response()
    monkeypatch.setattr(r, "_http_json", fake)
    result = r.run(path)
    assert result["status"] == "failed"
    assert json.loads((path / "credential.json").read_text()) == changed


def test_atomic_replace_failure_preserves_original_and_consumes_attempt(state, monkeypatch):
    path, old = state
    monkeypatch.setattr(r, "_http_json", lambda *args: response())
    def broken_replace(*args, **kwargs):
        raise OSError("invented private detail")
    monkeypatch.setattr(r.os, "replace", broken_replace)
    result = r.run(path)
    assert result["error"] == "credential_write_failed"
    assert result["rotation_outcome"] == "unknown"
    assert json.loads((path / "credential.json").read_text()) == old
    assert not list(path.glob(".refresh-credential-*"))
    assert r.run(path)["error"] == "generation_consumed"
    assert "invented" not in json.dumps(result)


def test_deadline_is_unknown_rotation_and_releases_lock(state, monkeypatch):
    path, old = state
    def timeout(*args):
        raise r.Deadline()
    monkeypatch.setattr(r, "_http_json", timeout)
    result = r.run(path)
    assert result["error"] == "deadline_exceeded"
    assert result["rotation_outcome"] == "unknown"
    assert json.loads((path / "credential.json").read_text()) == old
    assert r.run(path)["error"] == "generation_consumed"


@pytest.mark.parametrize("provider_code, expected", [
    ("invalid_grant", "oauth_invalid_grant"),
    ("invented-private-response", "upstream_rejected"),
    (["invented-private-response"], "upstream_rejected"),
])
def test_http_errors_only_export_finite_codes(monkeypatch, provider_code, expected):
    import io
    from unittest.mock import Mock
    body = json.dumps({"error": provider_code, "message": "invented-private-detail"}).encode()
    error = r.url_error.HTTPError("https://auth.openai.com/api/accounts/oauth/token", 400,
                                "invented-private-reason", {}, io.BytesIO(body))
    opener = Mock()
    opener.open.side_effect = error
    handlers = []
    def build(*args):
        handlers.extend(args)
        return opener
    monkeypatch.setattr(r.request, "build_opener", build)
    with pytest.raises(r.RefreshError) as caught:
        r._http_json("https://auth.openai.com/api/accounts/oauth/token", {"refresh_token": "invented"})
    assert caught.value.code == expected and caught.value.http_status == 400
    assert "invented" not in str(caught.value)
    assert handlers[0].proxies == {}
    assert isinstance(handlers[1], r._NoRedirect)
