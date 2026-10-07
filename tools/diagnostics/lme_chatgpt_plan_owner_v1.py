"""One-way SIWC VM transfer and one-process credential broker for HyMem LME.

Import and broker entry points never print credentials. ``transfer_to_afrodite``
is intentionally an API, not a CLI command, so a caller must select the exact
private local state and reviewed remote source/runtime before consuming it.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

sys.dont_write_bytecode = True

REFRESH_SHA256 = "194e6b10ecfef44f249d4aa227e2723acf80f3d8288504f1daaeacf4d3aad0e0"
REMOTE_DIR = Path("/home/atta/.hymem-chatgpt-plan-lme")
REMOTE_RUNTIME = Path("/home/atta/.hymem-siwc-runtime-v1/bin/python")
REMOTE_SOURCE = Path("/home/atta/.hymem-siwc-owner-v1/tools/diagnostics/lme_chatgpt_plan_owner_v1.py")
SSH_HOST = "afrodite"
POLICY = "siwc_server_enforced_plan_or_existing_credits_v1"
MAX_FILE = 128 * 1024
MAX_WIRE = 4 * MAX_FILE
MAX_REPLY = 4096
ERRORS = frozenset({"source_mismatch", "source_unavailable", "state_invalid", "binding_invalid",
                    "credential_invalid", "transfer_consumed", "transfer_unavailable", "remote_invalid",
                    "remote_existing", "remote_unavailable", "owner_running", "deadline_exceeded",
                    "refresh_unknown", "refresh_denied", "refresh_failed", "internal_error"})


class OwnerError(Exception):
    def __init__(self, code: str):
        self.code = code if type(code) is str and code in ERRORS else "internal_error"
        super().__init__(self.code)


class CredentialLease:
    """In-memory bearer access for a single broker caller; repr is always safe."""

    __slots__ = ("_access_token", "expires_at", "policy")

    def __init__(self, access_token: str, expires_at: int):
        self._access_token = access_token
        self.expires_at = expires_at
        self.policy = POLICY

    @property
    def access_token(self) -> str:
        return self._access_token

    def __repr__(self) -> str:
        return "CredentialLease(<hidden>)"


def _pinned_refresh():
    source = Path(__file__).with_name("lme_chatgpt_plan_refresh_v1.py")
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 256 * 1024:
                raise OwnerError("source_mismatch")
            raw = os.read(fd, 256 * 1024 + 1)
        finally:
            os.close(fd)
        if hashlib.sha256(raw).hexdigest() != REFRESH_SHA256:
            raise OwnerError("source_mismatch")
        spec = importlib.util.spec_from_file_location("_pinned_lme_refresh_v1", source)
        if spec is None:
            raise OwnerError("source_unavailable")
        module = importlib.util.module_from_spec(spec)
        exec(compile(raw, str(source), "exec"), module.__dict__)
        return module
    except OwnerError:
        raise
    except Exception:
        raise OwnerError("source_unavailable") from None


def _dir(path: Path, *, create: bool = False) -> int:
    if not path.is_absolute() or str(path) == "/" or ".." in path.parts:
        raise OwnerError("state_invalid")
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for index, part in enumerate(path.parts[1:]):
                if part in ("", ".", ".."):
                    raise OwnerError("state_invalid")
                if create and index == len(path.parts) - 2:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = next_fd
        except BaseException:
            os.close(fd)
            raise
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            os.close(fd)
            raise OwnerError("state_invalid")
        return fd
    except OwnerError:
        raise
    except OSError:
        raise OwnerError("state_invalid") from None


def _read(fd: int, name: str) -> dict:
    try:
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
        try:
            info = os.fstat(file_fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_size > MAX_FILE):
                raise OwnerError("state_invalid")
            raw = os.read(file_fd, MAX_FILE + 1)
        finally:
            os.close(file_fd)
    except OwnerError:
        raise
    except OSError:
        raise OwnerError("state_invalid") from None
    if len(raw) > MAX_FILE:
        raise OwnerError("state_invalid")
    try:
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError()
                value[key] = item
            return value
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if type(value) is not dict:
            raise ValueError()
        _finite(value)
        return value
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise OwnerError("state_invalid") from None


def _finite(value) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is float and not math.isfinite(item):
            raise ValueError()
        if type(item) is dict:
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)


def _exclusive(fd: int, name: str, data: bytes) -> None:
    if len(data) > MAX_FILE:
        raise OwnerError("state_invalid")
    try:
        out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                      0o600, dir_fd=fd)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(out, view):]
            os.fsync(out)
        finally:
            os.close(out)
        os.fsync(fd)
    except FileExistsError:
        raise OwnerError("remote_existing") from None
    except OSError:
        raise OwnerError("state_invalid") from None


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _lock(fd: int, *, name: str = ".flow.lock", wait: float = 0) -> int:
    try:
        lockfd = os.open(name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=fd)
        info = os.fstat(lockfd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise OwnerError("state_invalid")
        until = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lockfd
            except BlockingIOError:
                if time.monotonic() >= until:
                    raise OwnerError("owner_running") from None
                time.sleep(min(0.05, until - time.monotonic()))
    except BaseException:
        if "lockfd" in locals():
            os.close(lockfd)
        raise


def _host_id(value) -> bool:
    if type(value) is not str or not value.startswith("urn:uuid:"):
        return False
    try:
        return str(uuid.UUID(value[9:])) == value[9:]
    except ValueError:
        return False


def prepare_vm(state_dir: Path = REMOTE_DIR) -> str:
    """Persist or read only the VM's own host ID; no credential is accepted."""
    fd = _dir(state_dir, create=True)
    try:
        lock = _lock(fd)
        try:
            try:
                host = _read(fd, "host.json")["ext_agent_host_id"]
            except OwnerError:
                # Only a genuinely absent host file may be initialized.
                try:
                    os.stat("host.json", dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    host = "urn:uuid:" + str(uuid.uuid4())
                    _exclusive(fd, "host.json", _json_bytes({"ext_agent_host_id": host}))
                else:
                    raise OwnerError("state_invalid") from None
            if not _host_id(host):
                raise OwnerError("binding_invalid")
            for name in ("credential.json", "registration.json", "status.json", ".import-attempt.json"):
                try:
                    os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise OwnerError("remote_existing")
            return host
        finally:
            os.close(lock)
    finally:
        os.close(fd)


def _validated_local(fd: int, refresh) -> tuple[dict, dict, dict, str]:
    signin, catalog = refresh._sources()
    old = refresh._old_credential(catalog, signin, fd, int(time.time()))
    registration = _read(fd, "registration.json")
    status = _read(fd, "status.json")
    host = _read(fd, "host.json").get("ext_agent_host_id")
    if not _host_id(host) or host != old["host_id"]:
        raise OwnerError("binding_invalid")
    return old, registration, status, host


def _ssh(action: str, payload: bytes, timeout: float) -> dict:
    if action not in ("prepare-vm", "import-vm") or not 0 < timeout <= 120:
        raise OwnerError("state_invalid")
    remote = " ".join(shlex.quote(str(x)) for x in
                      (REMOTE_RUNTIME, "-I", "-B", REMOTE_SOURCE, action))
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1", SSH_HOST, remote]
    try:
        result = subprocess.run(command, input=payload, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        if len(result.stdout) > MAX_REPLY:
            raise OwnerError("remote_invalid")
        reply = json.loads(result.stdout)
        if result.returncode or type(reply) is not dict:
            raise OwnerError("remote_unavailable")
        return reply
    except subprocess.TimeoutExpired:
        raise OwnerError("remote_unavailable") from None
    except (OSError, ValueError, UnicodeError):
        raise OwnerError("remote_unavailable") from None


def transfer_to_afrodite(local_dir: Path, *, deadline: float = 120) -> dict:
    """Consume local ownership before sending exactly one protected SSH import.

    A remote timeout/failure is ambiguous. This function never retries or
    restores ``credential.json``. ``credential.transferred.json`` is recoverable
    only through a separate manual reconciliation, outside this API.
    """
    if not 0 < deadline <= 120:
        raise OwnerError("deadline_exceeded")
    end = time.monotonic() + deadline
    refresh = _pinned_refresh()
    preflight = _ssh("prepare-vm", b"", min(end - time.monotonic(), 30))
    remote_host = preflight.get("host_id")
    if preflight.get("status") != "ready" or not _host_id(remote_host):
        raise OwnerError("remote_invalid")
    fd = _dir(local_dir)
    try:
        lock = _lock(fd)
        try:
            if time.monotonic() >= end:
                raise OwnerError("deadline_exceeded")
            try:
                os.stat(".transfer-attempt.json", dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise OwnerError("transfer_consumed")
            try:
                old, registration, status, local_host = _validated_local(fd, refresh)
            except refresh.RefreshError:
                raise OwnerError("credential_invalid") from None
            if remote_host == local_host:
                raise OwnerError("binding_invalid")
            envelope = {"version": 1, "expected_host_id": remote_host,
                        "origin_host_id": local_host, "registration": registration,
                        "status": status, "credential": old}
            wire = _json_bytes(envelope)
            if len(wire) > MAX_WIRE:
                raise OwnerError("state_invalid")
            # Durable marker and recoverable rename precede any credential send.
            _exclusive(fd, ".transfer-attempt.json", _json_bytes({"attempted": True}))
            try:
                os.stat("credential.transferred.json", dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise OwnerError("transfer_consumed")
            try:
                os.rename("credential.json", "credential.transferred.json", src_dir_fd=fd, dst_dir_fd=fd)
                os.fsync(fd)
            except OSError:
                raise OwnerError("transfer_unavailable") from None
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise OwnerError("deadline_exceeded")
            reply = _ssh("import-vm", wire, remaining)
            if reply != {"status": "imported"}:
                raise OwnerError("remote_invalid")
            return {"status": "transferred", "local_owner": False, "remote_owner": True}
        finally:
            os.close(lock)
    finally:
        os.close(fd)


def import_vm(wire: bytes, state_dir: Path = REMOTE_DIR) -> dict:
    """One-time VM import. The wire is never emitted, logged, or persisted whole."""
    if len(wire) > MAX_WIRE:
        raise OwnerError("remote_invalid")
    try:
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError()
                value[key] = item
            return value
        envelope = json.loads(wire, object_pairs_hook=unique)
        if (type(envelope) is not dict or set(envelope) != {"version", "expected_host_id",
                "origin_host_id", "registration", "status", "credential"}
                or type(envelope["version"]) is not int or envelope["version"] != 1):
            raise ValueError()
        _finite(envelope)
    except (ValueError, UnicodeError, RecursionError):
        raise OwnerError("remote_invalid") from None
    fd = _dir(state_dir)
    try:
        lock = _lock(fd)
        try:
            remote_host = _read(fd, "host.json").get("ext_agent_host_id")
            origin = envelope.get("origin_host_id")
            registration = envelope.get("registration")
            status = envelope.get("status")
            old = envelope.get("credential")
            if (not _host_id(remote_host) or envelope.get("expected_host_id") != remote_host
                    or not _host_id(origin) or origin == remote_host
                    or type(registration) is not dict or type(status) is not dict or type(old) is not dict
                    or registration.get("host_id") != origin or old.get("host_id") != origin
                    or registration.get("client_id") != old.get("client_id")):
                raise OwnerError("binding_invalid")
            # Reuse the reviewed same-grant validation on the local record by
            # validating its shape below and on the completed VM record after
            # private writes. No network is performed by these validators.
            refresh = _pinned_refresh()
            signin, _catalog = refresh._sources()
            if (status.get("status") != "complete" or status.get("plan_usage_enabled") is not True
                    or type(status.get("model_calls")) is not int or status["model_calls"] != 0
                    or type(old.get("version")) is not int or old["version"] != 1
                    or old.get("issuer") != signin.AUTH_ORIGIN or old.get("token_type") != "Bearer"
                    or old.get("plan_usage_enabled") is not True
                    or type(old.get("subject")) is not str or not 0 < len(old["subject"]) <= 512
                    or type(registration.get("client_id")) is not str
                    or not signin.CLIENT_ID_RE.fullmatch(registration["client_id"])):
                raise OwnerError("credential_invalid")
            for key in ("access_token", "refresh_token", "id_token"):
                if not refresh._safe_token(signin, old.get(key)):
                    raise OwnerError("credential_invalid")
            try:
                refresh._scope_list(old.get("scope"))
            except refresh.RefreshError:
                raise OwnerError("credential_invalid") from None
            now = int(time.time())
            if (type(old.get("saved_at")) is not int or type(old.get("expires_at")) is not int
                    or old["saved_at"] < 0 or old["saved_at"] > now + 60
                    or old["expires_at"] <= old["saved_at"]
                    or old["expires_at"] - old["saved_at"] > 86400):
                raise OwnerError("credential_invalid")
            for name in ("credential.json", "registration.json", "status.json", ".import-attempt.json"):
                try:
                    os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise OwnerError("remote_existing")
            # Preserve provenance without using the laptop host as VM identity.
            remote_registration = dict(registration, host_id=remote_host, origin_host_id=origin)
            remote_credential = dict(old, host_id=remote_host, origin_host_id=origin)
            registration_bytes = _json_bytes(remote_registration)
            status_bytes = _json_bytes(status)
            credential_bytes = _json_bytes(remote_credential)
            if any(len(value) > MAX_FILE for value in
                   (registration_bytes, status_bytes, credential_bytes)):
                raise OwnerError("credential_invalid")
            _exclusive(fd, ".import-attempt.json", _json_bytes({"attempted": True}))
            _exclusive(fd, "registration.json", registration_bytes)
            _exclusive(fd, "status.json", status_bytes)
            _exclusive(fd, "credential.json", credential_bytes)
            refresh._old_credential(_catalog, signin, fd, now)
            return {"status": "imported"}
        finally:
            os.close(lock)
    finally:
        os.close(fd)


class CredentialBroker:
    """Sole VM process owner; four worker threads share ``acquire``."""

    def __init__(self, state_dir: Path = REMOTE_DIR, runtime: Path = REMOTE_RUNTIME):
        if not runtime.is_absolute() or not runtime.is_file() or runtime.is_symlink() or not os.access(runtime, os.X_OK):
            raise OwnerError("state_invalid")
        self.state_dir = state_dir
        self.runtime = runtime
        self._fd = _dir(state_dir)
        try:
            self._owner = _lock(self._fd, name=".owner.lock")
        except BaseException:
            os.close(self._fd)
            raise
        self._thread_lock = threading.Lock()
        self._closed = False
        self._stopped = False
        try:
            refresh = _pinned_refresh()
            signin, catalog = refresh._sources()
            initial = self._current(refresh, signin, catalog, 0)
            self.identity_digest = self._identity(initial)
        except BaseException:
            self.close()
            raise

    def _identity(self, old: dict) -> str:
        origin = old.get("origin_host_id")
        registration = _read(self._fd, "registration.json")
        if (not _host_id(origin) or origin == old["host_id"]
                or registration.get("origin_host_id") != origin):
            raise OwnerError("binding_invalid")
        fields = {key: old[key] for key in ("issuer", "host_id", "client_id", "subject")}
        fields["origin_host_id"] = origin
        fields["scope"] = sorted(old["scope"])
        return hashlib.sha256(b"HyMem SIWC VM grant v1\0" + _json_bytes(fields)).hexdigest()

    def close(self) -> None:
        with self._thread_lock:
            if not self._closed:
                os.close(self._owner)
                os.close(self._fd)
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _refresh(self, timeout: float) -> dict:
        refresh = _pinned_refresh()
        source = Path(refresh.__file__)
        command = [str(self.runtime), "-I", "-B", str(source), "refresh", "--state-dir", str(self.state_dir)]
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, start_new_session=True,
                                     env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
            try:
                raw, _ = child.communicate(timeout=min(timeout, 35))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
                self._stopped = True
                raise OwnerError("refresh_unknown") from None
            if len(raw) > MAX_REPLY:
                self._stopped = True
                raise OwnerError("refresh_unknown")
            result = json.loads(raw)
            if type(result) is not dict or result.get("status") not in ("refreshed", "not_due", "failed"):
                self._stopped = True
                raise OwnerError("refresh_unknown")
            if result["status"] == "failed":
                if (not {"status", "refreshed", "refresh_requests", "model_calls", "error",
                         "rotation_outcome"} <= set(result)
                        or set(result) - {"status", "refreshed", "refresh_requests", "model_calls",
                                          "error", "rotation_outcome", "http_status"}
                        or result["refreshed"] is not False
                        or type(result["refresh_requests"]) is not int
                        or result["refresh_requests"] not in (0, 1)
                        or type(result["model_calls"]) is not int or result["model_calls"] != 0
                        or result["rotation_outcome"] not in ("unknown", "not_attempted")
                        or type(result["error"]) is not str or result["error"] not in refresh.ERROR_CODES):
                    self._stopped = True
                    raise OwnerError("refresh_unknown")
                self._stopped = True
                code = result.get("error")
                if type(code) is str and code.startswith("oauth_"):
                    raise OwnerError("refresh_denied")
                if code == "generation_consumed" or result.get("rotation_outcome") == "unknown":
                    raise OwnerError("refresh_unknown")
                raise OwnerError("refresh_failed")
            if child.returncode != 0:
                self._stopped = True
                raise OwnerError("refresh_unknown")
            if (set(result) != {"status", "refreshed", "refresh_requests", "model_calls",
                                   "expiry_remaining", "rotation_outcome"}
                    or type(result["refreshed"]) is not bool
                    or result["refreshed"] != (result["status"] == "refreshed")
                    or type(result["refresh_requests"]) is not int
                    or result["refresh_requests"] != (1 if result["refreshed"] else 0)
                    or type(result["model_calls"]) is not int or result["model_calls"] != 0
                    or type(result["expiry_remaining"]) is not int or result["expiry_remaining"] <= 0
                    or result["rotation_outcome"] != ("confirmed" if result["refreshed"] else "not_attempted")):
                self._stopped = True
                raise OwnerError("refresh_unknown")
            return result
        except OwnerError:
            raise
        except (OSError, ValueError, UnicodeError):
            self._stopped = True
            raise OwnerError("refresh_unknown") from None

    def _current(self, refresh, signin, catalog, wait: float) -> dict:
        lock = _lock(self._fd, wait=wait)
        try:
            old = refresh._old_credential(catalog, signin, self._fd, int(time.time()))
            digest = self._identity(old)
            if hasattr(self, "identity_digest") and digest != self.identity_digest:
                self._stopped = True
                raise OwnerError("binding_invalid")
            return old
        finally:
            os.close(lock)

    def acquire(self, *, caller_deadline: float) -> CredentialLease:
        """Absolute monotonic deadline, no more than 120 seconds from caller."""
        remaining = caller_deadline - time.monotonic()
        if not 0 < remaining <= 120:
            raise OwnerError("deadline_exceeded")
        if not self._thread_lock.acquire(timeout=remaining):
            raise OwnerError("deadline_exceeded")
        try:
            if self._closed or self._stopped:
                raise OwnerError("refresh_unknown")
            try:
                refresh = _pinned_refresh()
                signin, catalog = refresh._sources()
            except OwnerError:
                raise
            except Exception:
                raise OwnerError("source_unavailable") from None
            old = self._current(refresh, signin, catalog, max(0, caller_deadline - time.monotonic()))
            if old["expires_at"] - int(time.time()) <= max(300, caller_deadline - time.monotonic() + 60):
                remaining = caller_deadline - time.monotonic()
                if remaining <= 0:
                    raise OwnerError("deadline_exceeded")
                self._refresh(remaining)
                old = self._current(refresh, signin, catalog, max(0, caller_deadline - time.monotonic()))
            if time.monotonic() >= caller_deadline:
                raise OwnerError("deadline_exceeded")
            if old["expires_at"] - int(time.time()) <= caller_deadline - time.monotonic() + 60:
                raise OwnerError("credential_invalid")
            return CredentialLease(old["access_token"], old["expires_at"])
        except OwnerError:
            raise
        except Exception:
            raise OwnerError("credential_invalid") from None
        finally:
            self._thread_lock.release()


def main(argv: list[str] | None = None) -> int:
    # Remote-only protocol; no local transfer or bearer-returning CLI exists.
    if argv is None:
        argv = sys.argv[1:]
    try:
        if argv == ["prepare-vm"]:
            result = {"status": "ready", "host_id": prepare_vm()}
        elif argv == ["import-vm"]:
            result = import_vm(sys.stdin.buffer.read(MAX_WIRE + 1))
        else:
            return 2
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except OwnerError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}), flush=True)
        return 1
    except Exception:
        print('{"status":"failed","error":"internal_error"}', flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
