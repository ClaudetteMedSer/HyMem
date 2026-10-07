"""Offline ownership controls. Every credential and host ID here is invented."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock

import pytest

from tools.diagnostics import lme_chatgpt_plan_owner_v1 as owner


SOURCE = Path(owner.__file__)
LOCAL_HOST = "urn:uuid:00000000-0000-4000-8000-000000000001"
REMOTE_HOST = "urn:uuid:00000000-0000-4000-8000-000000000002"
CLIENT = "oaiapp_invented123"
SCOPES = ["openid", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"]


def write(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def fixture_state(base, *, due=False):
    state = base / "private"
    state.mkdir(mode=0o700)
    now = int(time.time())
    credential = {"version": 1, "issuer": "https://auth.openai.com", "host_id": LOCAL_HOST,
                  "client_id": CLIENT, "subject": "invented-subject", "access_token": "invented-access",
                  "refresh_token": "invented-refresh", "id_token": "invented-id", "token_type": "Bearer",
                  "scope": SCOPES, "saved_at": now - 100, "expires_at": now + (100 if due else 3600),
                  "plan_usage_enabled": True}
    write(state / "host.json", {"ext_agent_host_id": LOCAL_HOST})
    write(state / "registration.json", {"host_id": LOCAL_HOST, "client_id": CLIENT,
                                        "registration_metadata": "invented"})
    write(state / "status.json", {"status": "complete", "plan_usage_enabled": True, "model_calls": 0})
    write(state / "credential.json", credential)
    return state


def fixture_vm_state(base, *, due=False):
    state = fixture_state(base, due=due)
    write(state / "host.json", {"ext_agent_host_id": REMOTE_HOST})
    registration = json.loads((state / "registration.json").read_text())
    registration.update(host_id=REMOTE_HOST, origin_host_id=LOCAL_HOST)
    write(state / "registration.json", registration)
    credential = json.loads((state / "credential.json").read_text())
    credential.update(host_id=REMOTE_HOST, origin_host_id=LOCAL_HOST)
    write(state / "credential.json", credential)
    return state


def test_pinned_sources_and_private_vm_host():
    assert hashlib.sha256(SOURCE.with_name("lme_chatgpt_plan_refresh_v1.py").read_bytes()).hexdigest() == owner.REFRESH_SHA256
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = Path(tmp) / "vm"
        first = owner.prepare_vm(state)
        assert owner.prepare_vm(state) == first
        assert (state / "host.json").stat().st_mode & 0o777 == 0o600
        assert state.stat().st_mode & 0o777 == 0o700
        assert owner._host_id(first)


def test_transfer_consumes_local_before_wire_and_import_rebinds_vm():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        base = Path(tmp)
        local = fixture_state(base)
        vm = base / "vm"
        write_host = owner.prepare_vm(vm)
        calls = []

        def ssh(action, payload, _timeout):
            calls.append(action)
            if action == "prepare-vm":
                return {"status": "ready", "host_id": write_host}
            assert not (local / "credential.json").exists()
            assert (local / "credential.transferred.json").exists()
            assert (local / ".transfer-attempt.json").exists()
            assert b"invented-refresh" in payload
            return owner.import_vm(payload, vm)

        with mock.patch.object(owner, "_ssh", side_effect=ssh):
            result = owner.transfer_to_afrodite(local)
        assert result == {"status": "transferred", "local_owner": False, "remote_owner": True}
        assert calls == ["prepare-vm", "import-vm"]
        assert not (local / "credential.json").exists()
        original = json.loads((local / "credential.transferred.json").read_text())
        imported = json.loads((vm / "credential.json").read_text())
        registration = json.loads((vm / "registration.json").read_text())
        assert imported["host_id"] == write_host != LOCAL_HOST
        assert registration["host_id"] == write_host
        assert imported["origin_host_id"] == registration["origin_host_id"] == LOCAL_HOST
        for field in ("client_id", "subject", "access_token", "refresh_token", "id_token", "scope"):
            assert imported[field] == original[field]
        for filename in ("credential.json", "registration.json", "status.json", ".import-attempt.json"):
            assert (vm / filename).stat().st_mode & 0o777 == 0o600
        with mock.patch.object(owner, "_ssh", side_effect=ssh):
            with pytest.raises(owner.OwnerError) as caught:
                owner.transfer_to_afrodite(local)
        assert caught.value.code in ("remote_existing", "transfer_consumed")


def test_ambiguous_send_stays_consumed_and_recoverable():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        local = fixture_state(Path(tmp))
        def ssh(action, payload, _timeout):
            if action == "prepare-vm":
                return {"status": "ready", "host_id": REMOTE_HOST}
            assert not (local / "credential.json").exists()
            raise owner.OwnerError("remote_unavailable")
        with mock.patch.object(owner, "_ssh", side_effect=ssh):
            with pytest.raises(owner.OwnerError) as caught:
                owner.transfer_to_afrodite(local)
        assert caught.value.code == "remote_unavailable"
        assert (local / "credential.transferred.json").exists()
        with mock.patch.object(owner, "_ssh", side_effect=ssh) as attempted:
            with pytest.raises(owner.OwnerError) as caught:
                owner.transfer_to_afrodite(local)
        assert caught.value.code == "transfer_consumed"
        assert [call.args[0] for call in attempted.call_args_list] == ["prepare-vm"]


def test_existing_vm_and_same_host_never_consume_local():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        base = Path(tmp)
        local = fixture_state(base)
        vm = base / "vm"
        vm.mkdir(mode=0o700)
        write(vm / "host.json", {"ext_agent_host_id": REMOTE_HOST})
        write(vm / "credential.json", {"access_token": "existing"})
        with pytest.raises(owner.OwnerError) as caught:
            owner.prepare_vm(vm)
        assert caught.value.code == "remote_existing"
        assert json.loads((vm / "credential.json").read_text()) == {"access_token": "existing"}
        with mock.patch.object(owner, "_ssh", return_value={"status": "ready", "host_id": LOCAL_HOST}):
            with pytest.raises(owner.OwnerError) as caught:
                owner.transfer_to_afrodite(local)
        assert caught.value.code == "binding_invalid"
        assert (local / "credential.json").exists()
        assert not (local / ".transfer-attempt.json").exists()


def test_bad_import_binding_and_existing_files_fail_closed():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        base = Path(tmp)
        local = fixture_state(base)
        vm = base / "vm"
        remote_host = owner.prepare_vm(vm)
        old = json.loads((local / "credential.json").read_text())
        registration = json.loads((local / "registration.json").read_text())
        status = json.loads((local / "status.json").read_text())
        envelope = {"version": 1, "expected_host_id": remote_host, "origin_host_id": LOCAL_HOST,
                    "registration": registration, "status": status, "credential": old}
        bad = dict(envelope, expected_host_id=LOCAL_HOST)
        with pytest.raises(owner.OwnerError) as caught:
            owner.import_vm(owner._json_bytes(bad), vm)
        assert caught.value.code == "binding_invalid"
        assert not (vm / "credential.json").exists()
        assert owner.import_vm(owner._json_bytes(envelope), vm) == {"status": "imported"}
        with pytest.raises(owner.OwnerError) as caught:
            owner.import_vm(owner._json_bytes(envelope), vm)
        assert caught.value.code == "remote_existing"


def test_broker_single_owner_four_threads_one_refresh_and_hidden_repr():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = fixture_vm_state(Path(tmp), due=True)
        broker = owner.CredentialBroker(state, Path("/opt/anaconda3/bin/python3.13"))
        with pytest.raises(owner.OwnerError) as caught:
            owner.CredentialBroker(state, Path("/opt/anaconda3/bin/python3.13"))
        assert caught.value.code == "owner_running"
        calls = []
        def fake_refresh(_timeout):
            calls.append(1)
            value = json.loads((state / "credential.json").read_text())
            value.update(access_token="invented-new-access", refresh_token="invented-new-refresh",
                         expires_at=int(time.time()) + 3600)
            write(state / "credential.json", value)
            return {"status": "refreshed"}
        broker._refresh = fake_refresh
        results = []
        errors = []
        def worker():
            try:
                results.append(broker.acquire(caller_deadline=time.monotonic() + 5))
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        broker.close()
        assert not errors and len(results) == 4 and len(calls) == 1
        assert all(result.access_token == "invented-new-access" for result in results)
        assert "invented" not in repr(results)
        assert all(result.policy == owner.POLICY for result in results)


def test_broker_subprocess_failure_stops_without_retry():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = fixture_vm_state(Path(tmp), due=True)
        with owner.CredentialBroker(state, Path("/opt/anaconda3/bin/python3.13")) as broker:
            child = mock.Mock()
            child.communicate.return_value = (b'{"status":"failed","error":"oauth_invalid_grant","rotation_outcome":"unknown"}', b"")
            child.returncode = 1
            with mock.patch.object(owner.subprocess, "Popen", return_value=child) as popen:
                with pytest.raises(owner.OwnerError) as caught:
                    broker.acquire(caller_deadline=time.monotonic() + 5)
                assert caught.value.code == "refresh_unknown"
                with pytest.raises(owner.OwnerError):
                    broker.acquire(caller_deadline=time.monotonic() + 5)
            assert popen.call_count == 1
            args = popen.call_args.args[0]
            assert args[1:4] == ["-I", "-B", str(SOURCE.with_name("lme_chatgpt_plan_refresh_v1.py"))]
            assert popen.call_args.kwargs["start_new_session"] is True
            assert popen.call_args.kwargs["env"] == {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def test_broker_identity_swap_and_short_lifetime_fail_closed():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = fixture_vm_state(Path(tmp))
        with owner.CredentialBroker(state, Path("/opt/anaconda3/bin/python3.13")) as broker:
            assert len(broker.identity_digest) == 64
            old = json.loads((state / "credential.json").read_text())
            registration = json.loads((state / "registration.json").read_text())
            old.update(client_id="oaiapp_different123", subject="different-subject")
            registration["client_id"] = old["client_id"]
            write(state / "credential.json", old)
            write(state / "registration.json", registration)
            with pytest.raises(owner.OwnerError) as caught:
                broker.acquire(caller_deadline=time.monotonic() + 5)
            assert caught.value.code == "binding_invalid"
            assert "different-subject" not in str(caught.value)
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = fixture_vm_state(Path(tmp), due=True)
        with owner.CredentialBroker(state, Path("/opt/anaconda3/bin/python3.13")) as broker:
            def insufficient(_timeout):
                value = json.loads((state / "credential.json").read_text())
                value["expires_at"] = int(time.time()) + 100
                write(state / "credential.json", value)
                return {"status": "refreshed"}
            broker._refresh = insufficient
            with pytest.raises(owner.OwnerError) as caught:
                broker.acquire(caller_deadline=time.monotonic() + 120)
            assert caught.value.code == "credential_invalid"


def test_refresh_child_timeout_is_killed_and_reaped_without_network():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        state = fixture_vm_state(Path(tmp), due=True)
        with owner.CredentialBroker(state, Path("/opt/anaconda3/bin/python3.13")) as broker:
            real_popen = subprocess.Popen
            children = []
            def sleeping_child(_command, **kwargs):
                child = real_popen([sys.executable, "-I", "-B", "-c", "import time; time.sleep(5)"], **kwargs)
                children.append(child)
                return child
            with mock.patch.object(owner.subprocess, "Popen", side_effect=sleeping_child):
                with pytest.raises(owner.OwnerError) as caught:
                    broker._refresh(0.1)
            assert caught.value.code == "refresh_unknown"
            assert len(children) == 1 and children[0].poll() is not None
            assert broker._stopped is True


def test_import_rejects_bool_version_nonfinite_and_duplicate_keys():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        vm = Path(tmp) / "vm"
        owner.prepare_vm(vm)
        for wire in (b'{"version":true}', b'{"version":1,"version":1}',
                     b'{"version":1,"expected_host_id":1e999}'):
            with pytest.raises(owner.OwnerError) as caught:
                owner.import_vm(wire, vm)
            assert caught.value.code == "remote_invalid"
        assert not (vm / "credential.json").exists()


def test_file_privacy_and_source_mismatch_block_before_send():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        local = fixture_state(Path(tmp))
        (local / "credential.json").chmod(0o644)
        with mock.patch.object(owner, "_ssh", return_value={"status": "ready", "host_id": REMOTE_HOST}) as ssh:
            with pytest.raises(owner.OwnerError):
                owner.transfer_to_afrodite(local)
        assert ssh.call_count == 1 and (local / "credential.json").exists()
        (local / "credential.json").chmod(0o600)
        with mock.patch.object(owner, "REFRESH_SHA256", "0" * 64), mock.patch.object(owner, "_ssh") as ssh:
            with pytest.raises(owner.OwnerError) as caught:
                owner.transfer_to_afrodite(local)
        assert caught.value.code == "source_mismatch"
        ssh.assert_not_called()
