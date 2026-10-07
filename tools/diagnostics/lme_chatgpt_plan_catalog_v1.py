"""One-shot, read-only SIWC catalog check for exact GPT-6 Luna availability.

Run with ``python -I -B ... --state-dir ABSOLUTE_PRIVATE_DIRECTORY``. This tool
never refreshes a token, writes state, or calls an inference endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import ssl
import stat
import sys
import time
from urllib import error as url_error, request
import uuid

sys.dont_write_bytecode = True

SIGNIN_SHA256 = "4b1863600fba63f6d620c904563ea130fa462ede21cc2cb5712d17370638e49e"
CATALOG_URL = "https://api.openai.com/v1/models"
MODEL_SLUG = "gpt-6-luna"
MAX_FILE = 128 * 1024
MAX_RESPONSE = 1024 * 1024
REQUEST_TIMEOUT = 15
ABSOLUTE_TIMEOUT = 30
REQUIRED_SCOPES = frozenset({"openid", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"})


class CheckError(Exception):
    def __init__(self, code: str, http_status: int | None = None):
        self.code = code
        self.http_status = http_status
        super().__init__(code)


class _AbsoluteTimeout(BaseException):
    pass


def _no_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _load_signin_module():
    source = Path(__file__).with_name("lme_chatgpt_plan_signin_v3.py")
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 256 * 1024:
                raise CheckError("source_mismatch")
            raw = os.read(fd, 256 * 1024 + 1)
        finally:
            os.close(fd)
        if hashlib.sha256(raw).hexdigest() != SIGNIN_SHA256:
            raise CheckError("source_mismatch")
        spec = importlib.util.spec_from_file_location("_pinned_lme_signin_v3", source)
        if spec is None:
            raise CheckError("source_unavailable")
        module = importlib.util.module_from_spec(spec)
        exec(compile(raw, str(source), "exec"), module.__dict__)
        return module
    except CheckError:
        raise
    except Exception:
        raise CheckError("source_unavailable") from None


def _open_private_dir(path: Path) -> int:
    if not path.is_absolute() or str(path) == "/":
        raise CheckError("state_dir_invalid")
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in path.parts[1:]:
                if part in ("", ".", ".."):
                    raise CheckError("state_dir_invalid")
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = next_fd
        except BaseException:
            os.close(fd)
            raise
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            os.close(fd)
            raise CheckError("state_dir_not_private")
        return fd
    except CheckError:
        raise
    except OSError:
        raise CheckError("state_dir_unavailable") from None


def _read_private_json(dir_fd: int, name: str) -> dict:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise CheckError("state_file_not_private")
            if info.st_size > MAX_FILE:
                raise CheckError("state_file_too_large")
            chunks = []
            remaining = MAX_FILE + 1
            while remaining:
                chunk = os.read(fd, min(16384, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        finally:
            os.close(fd)
    except CheckError:
        raise
    except OSError:
        raise CheckError("state_file_unavailable") from None
    raw = b"".join(chunks)
    if len(raw) > MAX_FILE:
        raise CheckError("state_file_too_large")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, UnicodeError):
        raise CheckError("state_file_invalid") from None
    if type(value) is not dict:
        raise CheckError("state_file_invalid")
    return value


def _validated_access_token(dir_fd: int, signin, now: int) -> str:
    status = _read_private_json(dir_fd, "status.json")
    if (status.get("status") != "complete" or status.get("plan_usage_enabled") is not True
            or type(status.get("model_calls")) is not int or status["model_calls"] != 0):
        raise CheckError("signin_incomplete")
    host = _read_private_json(dir_fd, "host.json")
    registration = _read_private_json(dir_fd, "registration.json")
    credential = _read_private_json(dir_fd, "credential.json")
    host_id = host.get("ext_agent_host_id")
    try:
        if type(host_id) is not str or not host_id.startswith("urn:uuid:") or str(uuid.UUID(host_id[9:])) != host_id[9:]:
            raise ValueError()
    except ValueError:
        raise CheckError("identity_binding_invalid") from None
    client_id = registration.get("client_id")
    if (type(client_id) is not str or not signin.CLIENT_ID_RE.fullmatch(client_id)
            or registration.get("host_id") != host_id
            or credential.get("host_id") != host_id or credential.get("client_id") != client_id):
        raise CheckError("identity_binding_invalid")
    if (type(credential.get("version")) is not int or credential["version"] != 1
            or credential.get("issuer") != signin.AUTH_ORIGIN
            or credential.get("token_type") != "Bearer"
            or credential.get("plan_usage_enabled") is not True):
        raise CheckError("credential_invalid")
    subject = credential.get("subject")
    token = credential.get("access_token")
    expiry = credential.get("expires_at")
    saved = credential.get("saved_at")
    if (type(subject) is not str or not subject or len(subject) > 512
            or type(token) is not str or not signin.SAFE_VALUE_RE.fullmatch(token)
            or type(expiry) is not int or type(saved) is not int
            or saved > now + 60 or expiry <= now + 60 or expiry <= saved or expiry - saved > 86400):
        raise CheckError("credential_invalid")
    scopes = credential.get("scope")
    if (type(scopes) is not list or len(scopes) > 64
            or any(type(scope) is not str or not scope or len(scope) > 128 or not scope.isascii() for scope in scopes)
            or len(set(scopes)) != len(scopes) or not REQUIRED_SCOPES <= set(scopes)):
        raise CheckError("scope_missing")
    return token


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CheckError("unexpected_redirect", code)


def _catalog_get(token: str) -> tuple[bool, int]:
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect(), request.HTTPSHandler(context=ssl.create_default_context()))
    req = request.Request(CATALOG_URL, headers={"Accept": "application/json", "Authorization": "Bearer " + token}, method="GET")
    try:
        with opener.open(req, timeout=REQUEST_TIMEOUT) as response:
            status = response.status
            if status != 200:
                raise CheckError("catalog_http_error", status if type(status) is int and 100 <= status <= 599 else None)
            raw = response.read(MAX_RESPONSE + 1)
    except CheckError:
        raise
    except url_error.HTTPError as exc:
        raise CheckError("catalog_http_error", exc.code if type(exc.code) is int and 100 <= exc.code <= 599 else None) from None
    except Exception:
        raise CheckError("catalog_unavailable") from None
    if len(raw) > MAX_RESPONSE:
        raise CheckError("catalog_too_large", 200)
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, UnicodeError):
        raise CheckError("catalog_invalid", 200) from None
    if type(value) is not dict or type(value.get("models")) is not list or len(value["models"]) > 10000:
        raise CheckError("catalog_invalid", 200)
    matches = 0
    for entry in value["models"]:
        if (type(entry) is not dict or type(entry.get("slug")) is not str
                or not 1 <= len(entry["slug"]) <= 256 or type(entry.get("visibility")) is not str
                or not 1 <= len(entry["visibility"]) <= 64):
            raise CheckError("catalog_invalid", 200)
        if entry["slug"] == MODEL_SLUG:
            matches += 1
            if entry["visibility"] != "list":
                raise CheckError("model_not_listed", 200)
    if matches > 1:
        raise CheckError("catalog_invalid", 200)
    return matches == 1, len(value["models"])


def check(state_dir: Path) -> dict:
    # SIGALRM limits the complete operation, including local reads and DNS/TLS.
    prior = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame):
        raise _AbsoluteTimeout()
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, ABSOLUTE_TIMEOUT)
    requests = 0
    try:
        signin = _load_signin_module()
        fd = _open_private_dir(state_dir)
        try:
            token = _validated_access_token(fd, signin, int(time.time()))
        finally:
            os.close(fd)
        requests = 1
        available, count = _catalog_get(token)
        if not available:
            raise CheckError("model_absent", 200)
        return {"status": "available", "available": True, "http_status": 200, "catalog_count": count,
                "model_slug": MODEL_SLUG, "model_calls": 0, "catalog_requests": requests}
    except _AbsoluteTimeout:
        error = CheckError("deadline_exceeded")
    except CheckError as exc:
        error = exc
    except Exception:
        error = CheckError("internal_error")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
    result = {"status": "failed", "error": error.code, "available": False, "model_slug": MODEL_SLUG,
              "model_calls": 0, "catalog_requests": requests}
    if error.http_status is not None:
        result["http_status"] = error.http_status
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = check(args.state_dir)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["available"] else 1


if __name__ == "__main__":
    sys.exit(main())
