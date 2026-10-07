"""Independent VM ownership and bounded refresh checks with invented grants."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import pytest
from tools.diagnostics import lme_chatgpt_plan_owner_v1 as o
from tests.test_lme_chatgpt_plan_owner_v1 import fixture_state, write


@pytest.fixture
def state():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
        base = Path(name)
        local = fixture_state(base, due=True)
        remote = base / "remote"
        host = o.prepare_vm(remote)
        body = {"version": 1, "expected_host_id": host,
            "origin_host_id": json.loads((local / "host.json").read_text())["ext_agent_host_id"],
            "credential": json.loads((local / "credential.json").read_text()),
            "registration": json.loads((local / "registration.json").read_text()),
            "status": json.loads((local / "status.json").read_text())}
        o.import_vm(o._json_bytes(body), remote)
        yield local, remote


def test_actual_refresh_child_deadline_reaped_and_no_retry(state):
    _, remote = state
    real_popen = subprocess.Popen
    created = []
    def blocked(*args, **kwargs):
        child = real_popen([sys.executable, "-I", "-B", "-c", "import time;time.sleep(30)"],
                          **kwargs)
        created.append(child)
        return child
    with o.CredentialBroker(remote, Path(sys.executable)) as broker:
        before = time.monotonic()
        with patch.object(o.subprocess, "Popen", side_effect=blocked):
            with pytest.raises(o.OwnerError) as error:
                broker.acquire(caller_deadline=time.monotonic() + 0.3)
            assert error.value.code == "refresh_unknown"
            with pytest.raises(o.OwnerError):
                broker.acquire(caller_deadline=time.monotonic() + 1)
        assert len(created) == 1 and created[0].poll() is not None
        assert time.monotonic() - before < 3


def test_short_refresh_validity_cannot_admit_full_call(state):
    _, remote = state
    with o.CredentialBroker(remote, Path(sys.executable)) as broker:
        def short(_timeout):
            value = json.loads((remote / "credential.json").read_text())
            value["expires_at"] = int(time.time()) + 100
            write(remote / "credential.json", value)
        with patch.object(broker, "_refresh", side_effect=short):
            with pytest.raises(o.OwnerError) as caught:
                broker.acquire(caller_deadline=time.monotonic() + 120)
        assert caught.value.code == "credential_invalid"


@pytest.mark.parametrize("field,new", [("subject", "different-account"),
                                      ("client_id", "oaiapp_different"),
                                      ("scope", ["openid", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct", "new-scope"])])
def test_grant_change_is_terminal(state, field, new):
    _, remote = state
    with o.CredentialBroker(remote, Path(sys.executable)) as broker:
        value = json.loads((remote / "credential.json").read_text())
        value[field] = new
        write(remote / "credential.json", value)
        if field == "client_id":
            registration = json.loads((remote / "registration.json").read_text())
            registration[field] = new
            write(remote / "registration.json", registration)
        with pytest.raises(o.OwnerError) as error:
            broker.acquire(caller_deadline=time.monotonic() + 5)
        assert error.value.code == "binding_invalid"
        assert broker._stopped


def test_missing_source_error_stays_finite(state):
    _, remote = state
    with o.CredentialBroker(remote, Path(sys.executable)) as broker:
        with patch.object(o, "_pinned_refresh", side_effect=o.OwnerError("source_mismatch")):
            with pytest.raises(o.OwnerError) as error:
                broker.acquire(caller_deadline=time.monotonic() + 5)
        assert error.value.code == "source_mismatch"


def test_ambiguous_transfer_does_not_restore_local_owner(state):
    local, _ = state
    def remote(action, payload, timeout):
        if action == "prepare-vm":
            return {"status":"ready", "host_id":"urn:uuid:00000000-0000-4000-8000-000000000009"}
        assert not (local / "credential.json").exists()
        raise o.OwnerError("remote_unavailable")
    with patch.object(o, "_ssh", side_effect=remote):
        with pytest.raises(o.OwnerError):
            o.transfer_to_afrodite(local)
    assert not (local / "credential.json").exists()
    assert (local / "credential.transferred.json").is_file()
    with patch.object(o, "_ssh", side_effect=remote) as ssh:
        with pytest.raises(o.OwnerError):
            o.transfer_to_afrodite(local)
        assert [call.args[0] for call in ssh.call_args_list] == ["prepare-vm"]


def test_private_state_rejects_overflow_and_duplicate_fields(state):
    _, remote = state
    fd = o._dir(remote)
    try:
        for data in ('{"x":1e999}', '{"x":1,"x":2}'):
            path = remote / "invalid.json"
            path.write_text(data)
            path.chmod(0o600)
            with pytest.raises(o.OwnerError):
                o._read(fd, "invalid.json")
    finally:
        os.close(fd)
