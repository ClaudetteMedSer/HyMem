"""One-time, loopback Sign in with ChatGPT for the isolated LME diagnostic.

Only this helper performs OAuth. It never calls a model. The credential file is
private and must be consumed only by a separately reviewed direct-plan client.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import html
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib import parse, request
from urllib import error as url_error
import uuid

import jwt


AUTH_ORIGIN = "https://auth.openai.com"
AUTHORIZE_URL = AUTH_ORIGIN + "/api/accounts/authorize"
TOKEN_URL = AUTH_ORIGIN + "/api/accounts/oauth/token"
JWKS_URL = AUTH_ORIGIN + "/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
REQUIRED_SCOPES = frozenset(SCOPES.split())
CLIENT_ID_RE = re.compile(r"oaiapp_[A-Za-z0-9_-]{8,128}\Z")
SAFE_VALUE_RE = re.compile(r"[\x20-\x7e]{1,8192}\Z")
MAX_REPLY = 128 * 1024


class SigninError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class DeadlineExpired(BaseException):
    pass


_REQUIRED_ID_CLAIMS = frozenset({"exp", "iat", "iss", "aud", "sub", "nonce"})


def _id_token_error_code(exc: Exception) -> str:
    """Reduce PyJWT failures to finite codes; never include untrusted details."""
    errors = jwt.exceptions
    if isinstance(exc, errors.InvalidSignatureError):
        return "id_token_signature_invalid"
    if isinstance(exc, errors.InvalidIssuerError):
        return "id_token_issuer_invalid"
    if isinstance(exc, errors.InvalidAudienceError):
        if str(exc) == "Invalid claim format in token (strict)":
            return "id_token_audience_format_invalid"
        if str(exc) == "Audience doesn't match (strict)":
            return "id_token_audience_mismatch"
        return "id_token_audience_invalid"
    if isinstance(exc, errors.ExpiredSignatureError):
        return "id_token_expired"
    if isinstance(exc, errors.ImmatureSignatureError):
        if str(exc) == "The token is not yet valid (iat)":
            return "id_token_issued_at_future"
        if str(exc) == "The token is not yet valid (nbf)":
            return "id_token_not_before_future"
        return "id_token_not_yet_valid"
    if isinstance(exc, errors.MissingRequiredClaimError):
        claim = exc.claim
        if isinstance(claim, str) and claim in _REQUIRED_ID_CLAIMS:
            return "id_token_claim_missing_" + claim
        return "id_token_claim_missing_other"
    if isinstance(exc, errors.InvalidIssuedAtError):
        return "id_token_issued_at_invalid"
    if isinstance(exc, errors.InvalidSubjectError):
        return "id_token_subject_invalid"
    if isinstance(exc, errors.InvalidJTIError):
        return "id_token_jti_invalid"
    if isinstance(exc, errors.InvalidAlgorithmError):
        return "id_token_algorithm_invalid"
    if isinstance(exc, errors.InvalidKeyError):
        return "id_token_key_invalid"
    if isinstance(exc, errors.DecodeError):
        return "id_token_decode_invalid"
    if isinstance(exc, errors.InvalidTokenError):
        return "id_token_decode_invalid"
    return "id_token_internal_error"


def _private_dir(path: Path) -> Path:
    if not path.is_absolute():
        raise SigninError("state_dir_not_absolute")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise SigninError("state_dir_symlink")
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
    except OSError as exc:
        raise SigninError("state_dir_unavailable") from exc
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SigninError("state_dir_not_private")
    return path


def _read_private(path: Path) -> dict | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SigninError("state_file_not_private") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SigninError("state_file_not_private")
        if info.st_size > MAX_REPLY:
            raise SigninError("state_file_too_large")
        chunks = []
        remaining = MAX_REPLY + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 16384))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_REPLY:
            raise SigninError("state_file_too_large")
    finally:
        os.close(fd)
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except ValueError as exc:
        raise SigninError("state_file_invalid") from exc


@contextmanager
def _flow_lock(state_dir: Path):
    path = state_dir / ".flow.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except OSError as exc:
        raise SigninError("lock_unavailable") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SigninError("lock_not_private")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SigninError("flow_already_running") from exc
        yield
    finally:
        os.close(fd)


def _write_private(path: Path, value: dict, *, replace: bool = False) -> None:
    old = _read_private(path)
    if old is not None and not replace:
        raise SigninError("state_file_exists")
    data = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
    fd, temp_name = tempfile.mkstemp(prefix=".signin-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temp_name, path)
        else:
            try:
                os.link(temp_name, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise SigninError("state_file_exists") from exc
            os.unlink(temp_name)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_host_id(state_dir: Path) -> str:
    path = state_dir / "host.json"
    old = _read_private(path)
    if old is None:
        host_id = "urn:uuid:" + str(uuid.uuid4())
        _write_private(path, {"ext_agent_host_id": host_id})
        return host_id
    host_id = old.get("ext_agent_host_id")
    try:
        if not isinstance(host_id, str) or not host_id.startswith("urn:uuid:"):
            raise ValueError()
        uuid.UUID(host_id[9:])
    except ValueError as exc:
        raise SigninError("host_id_invalid") from exc
    return host_id


def pkce_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()


def new_flow(host_id: str, port: int) -> dict:
    if not 1024 <= port <= 65535:
        raise SigninError("port_invalid")
    return {
        "host_id": host_id,
        "port": port,
        "state": secrets.token_urlsafe(32),
        "nonce": secrets.token_urlsafe(32),
        "verifier": secrets.token_urlsafe(64),
        "start_path": "/start/" + secrets.token_urlsafe(24),
    }


def authorization_url(flow: dict) -> str:
    fields = {
        "client_id": flow.get("client_id") or "dynamic_agent_client",
        "ext_agent_host_id": flow["host_id"],
        "response_type": "code",
        "redirect_uri": f"http://127.0.0.1:{flow['port']}/auth/callback",
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": flow["state"],
        "nonce": flow["nonce"],
        "code_challenge": pkce_challenge(flow["verifier"]),
        "code_challenge_method": "S256",
    }
    if not flow.get("client_id"):
        fields["agent_name_hint"] = "HyMem LME"
    return AUTHORIZE_URL + "?" + parse.urlencode(fields)


def parse_callback(raw_path: str, flow: dict) -> dict:
    parts = parse.urlsplit(raw_path)
    if parts.path != "/auth/callback" or parts.fragment or len(parts.query) > 16384:
        raise SigninError("callback_invalid")
    try:
        fields = parse.parse_qs(parts.query, keep_blank_values=True, strict_parsing=True, max_num_fields=12)
    except ValueError as exc:
        raise SigninError("callback_invalid") from exc
    if any(len(v) != 1 for v in fields.values()) or not set(fields) <= {"code", "state", "client_id", "scope", "error", "error_description"}:
        raise SigninError("callback_invalid")
    if not hmac.compare_digest(fields.get("state", [""])[0], flow["state"]):
        raise SigninError("state_mismatch")
    if "error" in fields:
        raise SigninError("consent_denied")
    code = fields.get("code", [""])[0]
    if not SAFE_VALUE_RE.fullmatch(code):
        raise SigninError("code_invalid")
    client_id = fields.get("client_id", [None])[0]
    if client_id is not None and not CLIENT_ID_RE.fullmatch(client_id):
        raise SigninError("client_id_invalid")
    scope = fields.get("scope", [None])[0]
    if scope is not None and (len(scope) > 8192 or not scope or any(not word.isascii() for word in scope.split())):
        raise SigninError("scope_invalid")
    return {"code": code, "client_id": client_id}


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise SigninError("unexpected_redirect")


def _json_request(url: str, form: dict | None = None) -> dict:
    if url not in (TOKEN_URL, JWKS_URL):
        raise SigninError("endpoint_invalid")
    body = None if form is None else parse.urlencode(form).encode()
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    req = request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        with opener.open(req, timeout=15) as reply:
            if reply.status != 200:
                raise SigninError("upstream_rejected")
            raw = reply.read(MAX_REPLY + 1)
    except SigninError:
        raise
    except url_error.HTTPError as exc:
        allowed = {"invalid_grant", "invalid_client", "access_denied", "invalid_request"}
        if exc.code in (400, 401, 403):
            try:
                data = exc.read(MAX_REPLY + 1)
                if len(data) <= MAX_REPLY:
                    payload = json.loads(data)
                    code = payload.get("error") if isinstance(payload, dict) else None
                    if isinstance(code, str) and code in allowed:
                        raise SigninError("oauth_" + code) from None
            except (ValueError, AttributeError):
                pass
        raise SigninError("upstream_rejected") from None
    except Exception as exc:
        raise SigninError("upstream_unavailable") from exc
    if len(raw) > MAX_REPLY:
        raise SigninError("upstream_too_large")
    try:
        result = json.loads(raw)
    except ValueError as exc:
        raise SigninError("upstream_invalid") from exc
    if not isinstance(result, dict):
        raise SigninError("upstream_invalid")
    return result


def validate_id_token(id_token: str, jwks: dict, client_id: str, nonce: str, now: int | None = None) -> dict:
    try:
        header = jwt.get_unverified_header(id_token)
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
            raise SigninError("id_token_header_invalid")
        if any(key in header for key in ("jku", "jwk", "x5u", "x5c")):
            raise SigninError("id_token_header_invalid")
        keys = [key for key in jwks.get("keys", []) if key.get("kid") == header["kid"] and key.get("kty") == "RSA" and key.get("use", "sig") == "sig"]
        if len(keys) != 1:
            raise SigninError("id_token_key_invalid")
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(keys[0]))
        claims = jwt.decode(
            id_token, public_key, algorithms=["RS256"], audience=client_id,
            issuer=AUTH_ORIGIN, options={"require": ["exp", "iat", "iss", "aud", "sub", "nonce"], "strict_aud": True},
            leeway=0,
        )
        current = int(time.time()) if now is None else now
        if (not isinstance(claims["aud"], str) or claims["aud"] != client_id
                or not isinstance(claims["exp"], int) or isinstance(claims["exp"], bool)
                or not isinstance(claims["iat"], int) or isinstance(claims["iat"], bool)
                or claims["iat"] > current + 60 or claims["iat"] < current - 86400):
            raise SigninError("id_token_time_invalid")
        if not isinstance(claims["nonce"], str) or not hmac.compare_digest(claims["nonce"], nonce):
            raise SigninError("nonce_mismatch")
        if not isinstance(claims["sub"], str) or not claims["sub"] or len(claims["sub"]) > 512:
            raise SigninError("identity_invalid")
        email = claims.get("email")
        if email is not None and (not isinstance(email, str) or "@" not in email or len(email) > 512):
            raise SigninError("identity_invalid")
        verified = claims.get("email_verified")
        if verified is not None and not isinstance(verified, bool):
            raise SigninError("identity_invalid")
        return claims
    except SigninError:
        raise
    except Exception as exc:
        raise SigninError(_id_token_error_code(exc)) from None


def validate_token_response(response: dict, jwks: dict, client_id: str, flow: dict, now: int | None = None) -> dict:
    token_type = response.get("token_type")
    if not isinstance(token_type, str) or token_type.lower() != "bearer":
        raise SigninError("token_type_invalid")
    expires = response.get("expires_in")
    if not isinstance(expires, int) or isinstance(expires, bool) or not 1 <= expires <= 86400:
        raise SigninError("token_expiry_invalid")
    for name in ("access_token", "refresh_token", "id_token"):
        value = response.get(name)
        if not isinstance(value, str) or not SAFE_VALUE_RE.fullmatch(value):
            raise SigninError("token_missing")
    scopes = response.get("scope")
    if not isinstance(scopes, str) or len(scopes) > 8192:
        raise SigninError("scope_missing")
    granted = set(scopes.split())
    if not {"openid", "offline_access"} <= granted:
        raise SigninError("scope_missing")
    claims = validate_id_token(response["id_token"], jwks, client_id, flow["nonce"], now)
    current = int(time.time()) if now is None else now
    enabled = {"openid", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"} <= granted
    return {
        "version": 1, "issuer": AUTH_ORIGIN, "host_id": flow["host_id"], "client_id": client_id,
        "subject": claims["sub"], "email": claims.get("email"), "email_verified": claims.get("email_verified"),
        "access_token": response["access_token"], "refresh_token": response["refresh_token"],
        "id_token": response["id_token"], "token_type": "Bearer",
        "scope": sorted(granted), "saved_at": current, "expires_at": current + expires,
        "plan_usage_enabled": enabled,
    }


def complete_flow(state_dir: Path, flow: dict, callback: dict) -> bool:
    registration_path = state_dir / "registration.json"
    registration = _read_private(registration_path)
    client_id = callback["client_id"]
    if registration is None:
        if client_id is None:
            raise SigninError("client_id_missing")
        registration = {"host_id": flow["host_id"], "client_id": client_id}
        _write_private(registration_path, registration)
    elif registration.get("host_id") != flow["host_id"] or not CLIENT_ID_RE.fullmatch(str(registration.get("client_id", ""))):
        raise SigninError("registration_invalid")
    elif client_id is not None and client_id != registration["client_id"]:
        raise SigninError("client_id_mismatch")
    client_id = registration["client_id"]
    existing = _read_private(state_dir / "credential.json")
    if existing is not None:
        if existing.get("host_id") != flow["host_id"] or existing.get("client_id") != client_id:
            raise SigninError("credential_identity_conflict")
        raise SigninError("credential_already_exists")
    token = _json_request(TOKEN_URL, {
        "grant_type": "authorization_code", "client_id": client_id,
        "code": callback["code"], "code_verifier": flow["verifier"],
        "redirect_uri": f"http://127.0.0.1:{flow['port']}/auth/callback", "resource": RESOURCE,
    })
    credential = validate_token_response(token, _json_request(JWKS_URL), client_id, flow)
    _write_private(state_dir / "credential.json", credential)
    return credential["plan_usage_enabled"]


def _serve_locked(state_dir: Path, port: int, wait_seconds: int) -> dict:
    if not 1 <= wait_seconds <= 900:
        raise SigninError("wait_invalid")
    state_dir = _private_dir(state_dir)
    flow = new_flow(load_host_id(state_dir), port)
    registration = _read_private(state_dir / "registration.json")
    if registration is not None:
        if registration.get("host_id") != flow["host_id"] or not CLIENT_ID_RE.fullmatch(str(registration.get("client_id", ""))):
            raise SigninError("registration_invalid")
        flow["client_id"] = registration["client_id"]
    if _read_private(state_dir / "credential.json") is not None:
        raise SigninError("credential_already_exists")
    outcome = {"status": "waiting", "plan_usage_enabled": False, "model_calls": 0}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *_args):
            pass

        def _reply(self, status: int, body: str):
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{port}":
                self._reply(400, "Invalid host")
                return
            if self.path == flow["start_path"] and outcome["status"] == "waiting":
                url = html.escape(authorization_url(flow), quote=True)
                self._reply(200, f"<html><body><a href=\"{url}\">Continue with ChatGPT</a></body></html>")
                return
            if not self.path.startswith("/auth/callback"):
                self._reply(404, "Not found")
                return
            if outcome["status"] != "waiting":
                self._reply(409, "Sign-in already completed")
                return
            try:
                callback = parse_callback(self.path, flow)
            except SigninError as exc:
                if exc.code == "consent_denied":
                    outcome.update(status="failed", error=exc.code)
                    _write_private(state_dir / "status.json", outcome, replace=True)
                self._reply(400, "Sign-in could not be completed. Return to Codex.")
                return
            # Only a callback bearing the correct state and one valid code may
            # consume the flow. Bad state and malformed requests leave it open.
            outcome["status"] = "processing"
            try:
                outcome["plan_usage_enabled"] = complete_flow(state_dir, flow, callback)
                outcome["status"] = "complete"
            except SigninError as exc:
                outcome["status"] = "failed"
                outcome["error"] = exc.code
            except Exception:
                outcome["status"] = "failed"
                outcome["error"] = "internal_error"
            _write_private(state_dir / "status.json", outcome, replace=True)
            if outcome["status"] == "complete":
                self._reply(200, "Sign-in complete. You may close this tab.")
            else:
                self._reply(400, "Sign-in could not be completed. Return to Codex.")

    class QuietServer(HTTPServer):
        def handle_error(self, _request, _client_address):
            # socketserver's default writes traceback text to stderr.
            pass

    server = QuietServer(("127.0.0.1", port), Handler)
    server.timeout = 1
    _write_private(state_dir / "status.json", outcome, replace=True)
    print(json.dumps({"status": "ready", "start_url": f"http://127.0.0.1:{port}{flow['start_path']}", "wait_seconds": wait_seconds, "model_calls": 0}), flush=True)
    deadline = time.monotonic() + wait_seconds
    try:
        while outcome["status"] == "waiting" and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if outcome["status"] == "waiting":
        outcome.update(status="timeout", error="consent_timeout")
        _write_private(state_dir / "status.json", outcome, replace=True)
    return outcome


def serve(state_dir: Path, port: int, wait_seconds: int) -> dict:
    if not 1 <= wait_seconds <= 900:
        raise SigninError("wait_invalid")
    state_dir = _private_dir(state_dir)
    with _flow_lock(state_dir):
        prior = signal.getsignal(signal.SIGALRM)

        def expire(_signum, _frame):
            raise DeadlineExpired()

        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, wait_seconds)
        try:
            return _serve_locked(state_dir, port, wait_seconds)
        except DeadlineExpired:
            outcome = {"status": "timeout", "error": "consent_timeout", "plan_usage_enabled": False, "model_calls": 0}
            _write_private(state_dir / "status.json", outcome, replace=True)
            return outcome
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=1455)
    parser.add_argument("--wait-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        result = serve(args.state_dir, args.port, args.wait_seconds)
    except SigninError as exc:
        result = {"status": "failed", "error": exc.code, "plan_usage_enabled": False, "model_calls": 0}
    except OSError:
        result = {"status": "failed", "error": "listener_unavailable", "plan_usage_enabled": False, "model_calls": 0}
    except Exception:
        result = {"status": "failed", "error": "internal_error", "plan_usage_enabled": False, "model_calls": 0}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "complete" and result["plan_usage_enabled"] else 1


if __name__ == "__main__":
    sys.exit(main())
