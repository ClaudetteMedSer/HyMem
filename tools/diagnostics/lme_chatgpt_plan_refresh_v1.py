"""Serialized local renewal of the existing HyMem SIWC credential.

Use ``refresh --state-dir ABSOLUTE_PRIVATE_DIRECTORY`` for one renewal attempt,
or ``--check-only --state-dir ...`` for a read-only expiry check. No model call.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import ssl
import stat
import sys
import time
from urllib import error as url_error, parse, request
import uuid

import jwt

sys.dont_write_bytecode = True

SIGNIN_SHA256 = "4b1863600fba63f6d620c904563ea130fa462ede21cc2cb5712d17370638e49e"
CATALOG_SHA256 = "599a74dd6c8010f37f7f304ec34a695930a461c1b0b05afa6eba5a6cfe68bc92"
MAX_REPLY = 128 * 1024
REQUIRED_SCOPES = frozenset({"openid", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"})
OAUTH_ERRORS = frozenset({"invalid_grant", "invalid_client", "invalid_refresh_token", "token_expired",
                          "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused"})
ERROR_CODES = frozenset({
    "source_mismatch", "source_unavailable", "state_dir_invalid", "state_dir_not_private",
    "state_dir_unavailable", "state_file_not_private", "state_file_too_large", "state_file_unavailable",
    "state_file_invalid", "signin_incomplete", "identity_binding_invalid", "credential_invalid",
    "scope_invalid", "lock_not_private", "lock_unavailable", "flow_already_running",
    "attempt_marker_invalid", "attempt_marker_unavailable", "generation_consumed", "deadline_exceeded",
    "unexpected_redirect", "upstream_rejected", "upstream_unavailable", "response_invalid",
    "response_too_large", "endpoint_invalid", "token_type_invalid", "token_expiry_invalid",
    "token_invalid", "scope_changed", "id_token_invalid", "credential_too_large",
    "credential_write_failed", "credential_changed", "internal_error",
} | {"oauth_" + code for code in OAUTH_ERRORS})


class RefreshError(Exception):
    def __init__(self, code: str, http_status: int | None = None):
        self.code = code
        self.http_status = http_status
        super().__init__(code)


class Deadline(BaseException):
    pass


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _bad_constant(_value):
    raise ValueError("nonfinite number")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite number")
    return result


def _json(raw: bytes) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_bad_constant,
                           parse_float=_finite_float)
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise RefreshError("response_invalid") from None
    if type(value) is not dict:
        raise RefreshError("response_invalid")
    return value


def _pinned_module(name: str, sha256: str):
    source = Path(__file__).with_name(name)
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 256 * 1024:
                raise RefreshError("source_mismatch")
            raw = os.read(fd, 256 * 1024 + 1)
        finally:
            os.close(fd)
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise RefreshError("source_mismatch")
        import types
        module = types.ModuleType("_pinned_" + name.replace(".", "_"))
        module.__file__ = str(source)
        exec(compile(raw, str(source), "exec"), module.__dict__)
        return module
    except RefreshError:
        raise
    except Exception:
        raise RefreshError("source_unavailable") from None


def _sources():
    signin = _pinned_module("lme_chatgpt_plan_signin_v3.py", SIGNIN_SHA256)
    catalog = _pinned_module("lme_chatgpt_plan_catalog_v1.py", CATALOG_SHA256)
    return signin, catalog


def _read(catalog, dir_fd: int, name: str) -> dict:
    try:
        value = catalog._read_private_json(dir_fd, name)
    except catalog.CheckError as exc:
        raise RefreshError(exc.code) from None
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is float and not math.isfinite(item):
            raise RefreshError("state_file_invalid")
        if type(item) is dict:
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
    return value


def _scope_list(value) -> list[str]:
    if (type(value) is not list or not 1 <= len(value) <= 64
            or any(type(item) is not str or not 1 <= len(item) <= 128 or not item.isascii()
                   or any(c.isspace() for c in item) for item in value)
            or len(set(value)) != len(value) or not REQUIRED_SCOPES <= set(value)):
        raise RefreshError("scope_invalid")
    return value


def _safe_token(signin, value) -> bool:
    return (type(value) is str and signin.SAFE_VALUE_RE.fullmatch(value) is not None
            and all(33 <= ord(char) <= 126 for char in value))


def _old_credential(catalog, signin, fd: int, now: int) -> dict:
    status = _read(catalog, fd, "status.json")
    host = _read(catalog, fd, "host.json")
    registration = _read(catalog, fd, "registration.json")
    old = _read(catalog, fd, "credential.json")
    if (status.get("status") != "complete" or status.get("plan_usage_enabled") is not True
            or type(status.get("model_calls")) is not int or status["model_calls"] != 0):
        raise RefreshError("signin_incomplete")
    host_id = host.get("ext_agent_host_id")
    try:
        if (type(host_id) is not str or not host_id.startswith("urn:uuid:")
                or str(uuid.UUID(host_id[9:])) != host_id[9:]):
            raise ValueError()
    except ValueError:
        raise RefreshError("identity_binding_invalid") from None
    client = registration.get("client_id")
    if (type(client) is not str or not signin.CLIENT_ID_RE.fullmatch(client)
            or registration.get("host_id") != host_id or old.get("host_id") != host_id
            or old.get("client_id") != client):
        raise RefreshError("identity_binding_invalid")
    if (type(old.get("version")) is not int or old["version"] != 1
            or old.get("issuer") != signin.AUTH_ORIGIN or old.get("token_type") != "Bearer"
            or old.get("plan_usage_enabled") is not True):
        raise RefreshError("credential_invalid")
    subject, saved, expiry = old.get("subject"), old.get("saved_at"), old.get("expires_at")
    if (type(subject) is not str or not subject or len(subject) > 512
            or type(saved) is not int or type(expiry) is not int
            or saved < 0 or saved > now + 60 or expiry <= saved or expiry - saved > 86400):
        raise RefreshError("credential_invalid")
    for field in ("access_token", "refresh_token", "id_token"):
        value = old.get(field)
        if not _safe_token(signin, value):
            raise RefreshError("credential_invalid")
    _scope_list(old.get("scope"))
    return old


def _lock(fd: int) -> int:
    try:
        lockfd = os.open(".flow.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
        info = os.fstat(lockfd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RefreshError("lock_not_private")
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RefreshError("flow_already_running") from None
        return lockfd
    except RefreshError:
        if "lockfd" in locals():
            os.close(lockfd)
        raise
    except OSError:
        if "lockfd" in locals():
            os.close(lockfd)
        raise RefreshError("lock_unavailable") from None


def _generation(old: dict) -> str:
    return hashlib.sha256(b"HyMem SIWC refresh generation v1\0" + old["refresh_token"].encode("ascii")).hexdigest()


def _attempt_name(old: dict) -> str:
    return ".refresh-attempt-" + _generation(old) + ".json"


def _attempt_exists(fd: int, old: dict) -> bool:
    name = _attempt_name(old)
    try:
        marker = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
        try:
            info = os.fstat(marker)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RefreshError("attempt_marker_invalid")
        finally:
            os.close(marker)
        return True
    except FileNotFoundError:
        return False
    except RefreshError:
        raise
    except OSError:
        raise RefreshError("attempt_marker_invalid") from None


def _mark_attempt(fd: int, old: dict) -> None:
    try:
        marker = os.open(_attempt_name(old), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=fd)
        try:
            os.write(marker, b'{"attempted":true}\n')
            os.fsync(marker)
        finally:
            os.close(marker)
        os.fsync(fd)
    except FileExistsError:
        raise RefreshError("generation_consumed") from None
    except OSError:
        raise RefreshError("attempt_marker_unavailable") from None


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RefreshError("unexpected_redirect", code if type(code) is int and 100 <= code <= 599 else None)


def _http_json(url: str, form: dict | None = None) -> dict:
    if url not in ("https://auth.openai.com/api/accounts/oauth/token",
                   "https://auth.openai.com/.well-known/jwks.json"):
        raise RefreshError("endpoint_invalid")
    body = None if form is None else parse.urlencode(form).encode("ascii")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect(),
                                  request.HTTPSHandler(context=ssl.create_default_context()))
    req = request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    try:
        with opener.open(req, timeout=15) as response:
            status = response.status
            if status != 200:
                raise RefreshError("upstream_rejected", status if type(status) is int and 100 <= status <= 599 else None)
            raw = response.read(MAX_REPLY + 1)
    except RefreshError:
        raise
    except url_error.HTTPError as exc:
        status = exc.code if type(exc.code) is int and 100 <= exc.code <= 599 else None
        try:
            raw_error = exc.read(MAX_REPLY + 1)
            if len(raw_error) <= MAX_REPLY:
                payload = _json(raw_error)
                code = payload.get("error")
                if type(code) is str and code in OAUTH_ERRORS:
                    raise RefreshError("oauth_" + code, status)
        except RefreshError as decoded:
            if decoded.code.startswith("oauth_"):
                raise
        raise RefreshError("upstream_rejected", status) from None
    except Exception:
        raise RefreshError("upstream_unavailable") from None
    if len(raw) > MAX_REPLY:
        raise RefreshError("response_too_large", 200)
    return _json(raw)


def _new_id_token(token: str, jwks: dict, client: str, subject: str, now: int) -> None:
    try:
        header = jwt.get_unverified_header(token)
        if (type(header) is not dict or header.get("alg") != "RS256" or type(header.get("kid")) is not str
                or any(k in header for k in ("jku", "jwk", "x5u", "x5c"))):
            raise RefreshError("id_token_invalid")
        keys = jwks.get("keys")
        if type(keys) is not list or len(keys) > 64:
            raise RefreshError("id_token_invalid")
        matches = [key for key in keys if type(key) is dict and key.get("kid") == header["kid"]
                   and key.get("kty") == "RSA" and key.get("use", "sig") == "sig"]
        if len(matches) != 1:
            raise RefreshError("id_token_invalid")
        key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(matches[0]))
        claims = jwt.decode(token, key, algorithms=["RS256"], audience=client,
                            issuer="https://auth.openai.com",
                            options={"require": ["exp", "iat", "iss", "aud", "sub"], "strict_aud": False},
                            leeway=0)
        audience = claims["aud"]
        audiences = [audience] if type(audience) is str else audience
        if (type(audiences) is not list or not 1 <= len(audiences) <= 16
                or any(type(a) is not str or not a.strip() or len(a) > 512 for a in audiences)
                or len(set(audiences)) != len(audiences) or client not in audiences
                or (len(audiences) > 1 and claims.get("azp") != client)
                or ("azp" in claims and claims["azp"] != client)):
            raise RefreshError("id_token_invalid")
        if (type(claims.get("sub")) is not str or claims["sub"] != subject
                or type(claims.get("iat")) is not int or claims["iat"] > now + 60
                or claims["iat"] < now - 86400 or type(claims.get("exp")) is not int
                or claims["exp"] <= now):
            raise RefreshError("id_token_invalid")
    except RefreshError:
        raise
    except Exception:
        raise RefreshError("id_token_invalid") from None


def _validated_new(response: dict, old: dict, signin, now: int) -> dict:
    token_type = response.get("token_type")
    expires = response.get("expires_in")
    if type(token_type) is not str or token_type.lower() != "bearer":
        raise RefreshError("token_type_invalid")
    if type(expires) is not int or not 1 <= expires <= 86400:
        raise RefreshError("token_expiry_invalid")
    for key in ("access_token", "refresh_token"):
        value = response.get(key)
        if not _safe_token(signin, value) or value == old[key]:
            raise RefreshError("token_invalid")
    scopes = response.get("scope")
    if scopes is not None:
        if type(scopes) is not str or len(scopes) > 8192:
            raise RefreshError("scope_invalid")
        words = scopes.split()
        if (len(words) != len(set(words)) or any(not word.isascii() or len(word) > 128 for word in words)
                or set(words) != set(old["scope"])):
            raise RefreshError("scope_changed")
    result = dict(old)
    result.update(access_token=response["access_token"], refresh_token=response["refresh_token"],
                  saved_at=now, expires_at=now + expires)
    if "earliest_refresh_at" in response:
        result["earliest_refresh_at"] = response["earliest_refresh_at"]
    if "id_token" in response:
        token = response["id_token"]
        if not _safe_token(signin, token):
            raise RefreshError("id_token_invalid")
        jwks = _http_json(signin.JWKS_URL)
        _new_id_token(token, jwks, old["client_id"], old["subject"], now)
        result["id_token"] = token
    return result


def _replace_credential(fd: int, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(data) > MAX_REPLY:
        raise RefreshError("credential_too_large")
    name = ".refresh-credential-" + uuid.uuid4().hex
    created = False
    try:
        temp = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                       0o600, dir_fd=fd)
        created = True
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(temp, view):]
            os.fsync(temp)
        finally:
            os.close(temp)
        os.replace(name, "credential.json", src_dir_fd=fd, dst_dir_fd=fd)
        created = False
        os.fsync(fd)
    except OSError:
        raise RefreshError("credential_write_failed") from None
    finally:
        if created:
            try:
                os.unlink(name, dir_fd=fd)
            except OSError:
                pass


def run(state_dir: Path, *, check_only: bool = False) -> dict:
    requests = 0
    lockfd = None
    fd = None
    prior = signal.getsignal(signal.SIGALRM)
    def expire(_signal, _frame):
        raise Deadline()
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, 30)
        signin, catalog = _sources()
        try:
            fd = catalog._open_private_dir(state_dir)
        except catalog.CheckError as exc:
            raise RefreshError(exc.code) from None
        if not check_only:
            lockfd = _lock(fd)
        now = int(time.time())
        old = _old_credential(catalog, signin, fd, now)
        remaining = old["expires_at"] - now
        if remaining > 300:
            return {"status": "not_due", "refreshed": False, "refresh_requests": 0,
                    "model_calls": 0, "expiry_remaining": remaining, "rotation_outcome": "not_attempted"}
        if _attempt_exists(fd, old):
            raise RefreshError("generation_consumed")
        if check_only:
            return {"status": "due", "refreshed": False, "refresh_requests": 0,
                    "model_calls": 0, "expiry_remaining": remaining, "rotation_outcome": "not_attempted"}
        _mark_attempt(fd, old)
        requests = 1
        response = _http_json(signin.TOKEN_URL, {
            "grant_type": "refresh_token", "client_id": old["client_id"],
            "refresh_token": old["refresh_token"], "resource": signin.RESOURCE,
        })
        renewed = _validated_new(response, old, signin, int(time.time()))
        if _read(catalog, fd, "credential.json") != old:
            raise RefreshError("credential_changed")
        _replace_credential(fd, renewed)
        return {"status": "refreshed", "refreshed": True, "refresh_requests": requests,
                "model_calls": 0, "expiry_remaining": renewed["expires_at"] - int(time.time()),
                "rotation_outcome": "confirmed"}
    except Deadline:
        error = RefreshError("deadline_exceeded")
    except RefreshError as exc:
        error = exc
    except Exception:
        error = RefreshError("internal_error")
    finally:
        if lockfd is not None:
            os.close(lockfd)
        if fd is not None:
            os.close(fd)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
    code = error.code if type(error.code) is str and error.code in ERROR_CODES else "internal_error"
    result = {"status": "failed", "refreshed": False, "refresh_requests": requests,
              "model_calls": 0, "error": code,
              "rotation_outcome": "unknown" if requests else "not_attempted"}
    if type(error.http_status) is int and 100 <= error.http_status <= 599:
        result["http_status"] = error.http_status
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["refresh"])
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if (args.action == "refresh") == args.check_only:
        parser.error("select exactly one of refresh or --check-only")
    result = run(args.state_dir, check_only=args.check_only)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
