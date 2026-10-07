"""One-call SIWC empty-terminal-output compatibility probe. Never use as an LME result.

Prepare only writes a private source-bound receipt. Run consumes it once and
may send exactly one invented ordinary prompt through the pinned public transport.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "transport_v1": (ROOT / "benchmarks/chatgpt_plan_responses_v1.py", "14dacc381e505834c2437952794a7447ccbce50636a58591a2e83f73c07278bf"),
    "transport_v2": (ROOT / "benchmarks/chatgpt_plan_responses_v2.py", "0d0d9fe6835fb0ec5fefa15f1144f632285699ac253175495fa983448feca05a"),
    "transport_v3": (ROOT / "benchmarks/chatgpt_plan_responses_v3.py", "c5169d498c8f3f2160141d5d649ec7cf27b25b4ba1645d7e607f43b3e3074659"),
    "transport_v4": (ROOT / "benchmarks/chatgpt_plan_responses_v4.py", "4cdd173ed84e17f03a0a44b749e575cbd401150873948310837503875d6387fc"),
    "transport_v5": (ROOT / "benchmarks/chatgpt_plan_responses_v5.py", "d347493a2abd70bfc5e060d50129a625790235b5987bd4cba92611eacb05aa11"),
    "transport_v6": (ROOT / "benchmarks/chatgpt_plan_responses_v6.py", "811bff13ebc4b24ebd22cad16542c3b58597dc04538a1d7deb5085df7190a28f"),
    "catalog": (ROOT / "tools/diagnostics/lme_chatgpt_plan_catalog_v1.py", "599a74dd6c8010f37f7f304ec34a695930a461c1b0b05afa6eba5a6cfe68bc92"),
    "signin": (ROOT / "tools/diagnostics/lme_chatgpt_plan_signin_v3.py", "4b1863600fba63f6d620c904563ea130fa462ede21cc2cb5712d17370638e49e"),
}
MODEL = "gpt-5.6-luna"
EFFORT = "low"
POLICY = "siwc_server_enforced_plan_or_existing_credits_v1"
PURPOSE = "empty_terminal_output_compatibility_validation_v1"
MAX_CALLS = 1
MAX_TOKENS = 160_000
MAX_MEANINGFUL_EVENTS = 4096  # Pinned transport v1/v6 bound.
MAX_SECONDS = 300
CALL_SECONDS = 120
RECEIPT_LIFETIME = 600
PLAIN_SYSTEM = "Reply with exactly this sentence and nothing else: blue paper kite is ready"
PLAIN_USER = "Please provide the specified sentence."
PROBE_CODES = frozenset({"plain_output_mismatch", "token_cap_exceeded",
                         "campaign_timeout", "cleanup_failure", "internal_error"})
BODY_SHAPES = frozenset({"empty", "error_object", "detail", "other_json", "non_json",
                         "oversized", "sse_event"})
MEDIA_CLASSES = frozenset({"missing", "sse", "json", "html", "text", "other", "invalid"})
PROVIDER_CODES = frozenset({
    "subscription_sharing_user_not_eligible", "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_usage_unavailable", "subscription_sharing_unsupported_capability",
    "subscription_sharing_route_not_supported", "subscription_sharing_invalid_user",
    "subscription_sharing_user_unavailable", "chatpass_v2_scope_not_authorized",
    "chatpass_v2_invalid_authorization_context",
})


class ProbeError(Exception):
    def __init__(self, code: str, phase: str = "local", http_status=None, body_shape=None):
        self.code, self.phase = code, phase
        self.http_status, self.body_shape = http_status, body_shape
        super().__init__(code)


class CampaignTimeout(BaseException):
    pass


def _source_hashes() -> dict:
    hashes = {}
    for name, (path, expected) in SOURCES.items():
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > 512_000:
                    raise ProbeError("source_mismatch")
                raw = os.read(fd, 512_001)
            finally:
                os.close(fd)
        except OSError:
            raise ProbeError("source_unavailable") from None
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise ProbeError("source_mismatch")
        hashes[name] = digest
    hashes["probe"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def _private_dir(path: Path) -> int:
    if not path.is_absolute() or str(path) == "/" or any(p in (".", "..") for p in path.parts):
        raise ProbeError("directory_invalid")
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in path.parts[1:]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = next_fd
        except BaseException:
            os.close(fd)
            raise
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ProbeError("directory_not_private")
        return fd
    except ProbeError:
        raise
    except OSError:
        raise ProbeError("directory_unavailable") from None


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate")
        obj[key] = value
    return obj


def _read(dirfd: int, name: str) -> tuple[dict, str]:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dirfd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 65536:
                raise ProbeError("file_not_private")
            raw = os.read(fd, 65537)
        finally:
            os.close(fd)
    except ProbeError:
        raise
    except OSError:
        raise ProbeError("file_unavailable") from None
    if len(raw) > 65536:
        raise ProbeError("file_too_large")
    try:
        value = json.loads(raw, object_pairs_hook=_unique)
    except (ValueError, UnicodeError):
        raise ProbeError("file_invalid") from None
    if type(value) is not dict:
        raise ProbeError("file_invalid")
    return value, hashlib.sha256(raw).hexdigest()


def _create(dirfd: int, name: str, value: dict) -> str:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=dirfd)
        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(dirfd)
    except FileExistsError:
        raise ProbeError("already_exists") from None
    except OSError:
        raise ProbeError("write_failed") from None
    return hashlib.sha256(raw).hexdigest()


def _modules():
    _source_hashes()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    catalog = importlib.import_module("tools.diagnostics.lme_chatgpt_plan_catalog_v1")
    transport_v1 = importlib.import_module("benchmarks.chatgpt_plan_responses_v1")
    transport_v2 = importlib.import_module("benchmarks.chatgpt_plan_responses_v2")
    transport_v3 = importlib.import_module("benchmarks.chatgpt_plan_responses_v3")
    transport_v4 = importlib.import_module("benchmarks.chatgpt_plan_responses_v4")
    transport_v5 = importlib.import_module("benchmarks.chatgpt_plan_responses_v5")
    transport = importlib.import_module("benchmarks.chatgpt_plan_responses_v6")
    if (Path(catalog.__file__).resolve() != SOURCES["catalog"][0]
            or Path(transport_v1.__file__).resolve() != SOURCES["transport_v1"][0]
            or Path(transport_v2.__file__).resolve() != SOURCES["transport_v2"][0]
            or Path(transport_v3.__file__).resolve() != SOURCES["transport_v3"][0]
            or Path(transport_v4.__file__).resolve() != SOURCES["transport_v4"][0]
            or Path(transport_v5.__file__).resolve() != SOURCES["transport_v5"][0]
            or Path(transport.__file__).resolve() != SOURCES["transport_v6"][0]
            or transport_v2._v1 is not transport_v1
            or transport_v3._v1 is not transport_v1
            or transport_v3._v2 is not transport_v2
            or transport_v4._v1 is not transport_v1
            or transport_v4._v2 is not transport_v2
            or transport_v4._v3 is not transport_v3
            or transport_v5._v1 is not transport_v1
            or transport_v5._v2 is not transport_v2
            or transport_v5._v3 is not transport_v3
            or transport._v1 is not transport_v1
            or transport._v2 is not transport_v2
            or transport._v3 is not transport_v3):
        raise ProbeError("source_mismatch")
    _source_hashes()
    return catalog, transport


def _receipt(rootfd: int, expected_sha: str | None = None) -> dict:
    receipt, digest = _read(rootfd, "receipt.json")
    if expected_sha is not None and digest != expected_sha:
        raise ProbeError("receipt_mismatch")
    expected_keys = {"version", "purpose", "sources", "model", "effort", "policy",
                     "limits", "fixtures", "state_dir", "host_id", "client_id",
                     "created_at", "expires_at"}
    if (set(receipt) != expected_keys or type(receipt.get("version")) is not int
            or receipt["version"] != 6 or receipt["purpose"] != PURPOSE
            or receipt["sources"] != _source_hashes()
            or receipt["model"] != MODEL or receipt["effort"] != EFFORT
            or receipt["policy"] != POLICY or receipt["limits"] != {
                "calls": MAX_CALLS, "observed_tokens": MAX_TOKENS,
                "campaign_seconds": MAX_SECONDS, "child_seconds": CALL_SECONDS}
            or receipt["fixtures"] != _fixtures()
            or type(receipt["state_dir"]) is not str
            or type(receipt["host_id"]) is not str
            or not receipt["host_id"].startswith("urn:uuid:")
            or type(receipt["client_id"]) is not str
            or not receipt["client_id"]
            or type(receipt["created_at"]) is not int
            or type(receipt["expires_at"]) is not int
            or receipt["expires_at"] - receipt["created_at"] != RECEIPT_LIFETIME):
        raise ProbeError("receipt_invalid")
    return receipt


def _fixtures() -> dict:
    return {"plain_system": PLAIN_SYSTEM, "plain_user": PLAIN_USER}


def _identity(statefd: int, catalog) -> tuple[str, str]:
    host = catalog._read_private_json(statefd, "host.json")
    registration = catalog._read_private_json(statefd, "registration.json")
    host_id, client_id = host.get("ext_agent_host_id"), registration.get("client_id")
    if (type(host_id) is not str or not host_id.startswith("urn:uuid:")
            or registration.get("host_id") != host_id or type(client_id) is not str):
        raise ProbeError("identity_binding_invalid")
    return host_id, client_id


def prepare(root: Path, state: Path) -> dict:
    hashes = _source_hashes()
    catalog, _ = _modules()
    rootfd = _private_dir(root)
    statefd = _private_dir(state)
    try:
        # Preparation reads only host and registration metadata, never credential.json.
        host_id, client_id = _identity(statefd, catalog)
        if os.listdir(root):
            raise ProbeError("root_not_empty")
        now = int(time.time())
        receipt = {"version": 6, "purpose": PURPOSE, "sources": hashes, "model": MODEL, "effort": EFFORT,
                   "policy": POLICY, "limits": {"calls": MAX_CALLS, "observed_tokens": MAX_TOKENS,
                   "campaign_seconds": MAX_SECONDS, "child_seconds": CALL_SECONDS},
                   "fixtures": _fixtures(), "state_dir": str(state), "host_id": host_id,
                   "client_id": client_id, "created_at": now, "expires_at": now + RECEIPT_LIFETIME}
        digest = _create(rootfd, "receipt.json", receipt)
        return {"status": "prepared", "receipt_sha256": digest, "model": MODEL,
                "policy": POLICY, "model_calls": 0}
    finally:
        os.close(statefd)
        os.close(rootfd)


def _fault(exc: BaseException, phase: str, transport) -> dict:
    if isinstance(exc, transport.TransportError):
        # Re-sanitize even a mutated exception before exporting metadata.
        observation = transport._v3._sanitize_observation(exc.wire_observation)
        if exc.wire_observation is not None and observation is None:
            raise ProbeError("result_invalid")
        stream = transport._sanitize_stream_observation(exc.stream_observation)
        if exc.stream_observation is not None and stream is None:
            raise ProbeError("result_invalid")
        safe = transport.TransportError(exc.code, exc.http_status, exc.body_shape,
                                        exc.media_type_class, observation, stream)
        return {"code": safe.code, "phase": phase, "http_status": safe.http_status,
                "body_shape": safe.body_shape, "media_type_class": safe.media_type_class,
                "wire_observation": safe.wire_observation,
                "stream_observation": safe.stream_observation}
    if isinstance(exc, ProbeError):
        code = exc.code if exc.code in PROBE_CODES else "internal_error"
        return {"code": code, "phase": exc.phase if exc.phase != "local" else phase,
                "http_status": None, "body_shape": None, "media_type_class": None,
                "wire_observation": None, "stream_observation": None}
    if isinstance(exc, CampaignTimeout):
        return {"code": "campaign_timeout", "phase": phase, "http_status": None,
                "body_shape": None, "media_type_class": None, "wire_observation": None,
                "stream_observation": None}
    return {"code": "internal_error", "phase": phase, "http_status": None,
            "body_shape": None, "media_type_class": None, "wire_observation": None,
            "stream_observation": None}


def _check_output(text: str) -> None:
    if text != "blue paper kite is ready":
        raise ProbeError("plain_output_mismatch")


def run(root: Path, state: Path, receipt_sha: str) -> dict:
    rootfd = _private_dir(root)
    try:
        receipt = _receipt(rootfd, receipt_sha)
        if receipt["state_dir"] != str(state):
            raise ProbeError("state_binding_mismatch")
        if int(time.time()) >= receipt["expires_at"]:
            raise ProbeError("receipt_expired")
        catalog, transport = _modules()
        signin = catalog._load_signin_module()
        # Local admission runs under the sign-in lock. A failed credential
        # preflight leaves the receipt unconsumed for reviewed refresh.
        with signin._flow_lock(state):
            statefd = _private_dir(state)
            try:
                if _identity(statefd, catalog) != (receipt["host_id"], receipt["client_id"]):
                    raise ProbeError("identity_binding_mismatch")
                token = catalog._validated_access_token(statefd, signin, int(time.time()))
            finally:
                os.close(statefd)
            credentials = transport.Credentials(token)
            # Exclusive marker precedes the first possible model call.
            attempt_at = int(time.time())
            if attempt_at >= receipt["expires_at"]:
                raise ProbeError("receipt_expired")
            _create(rootfd, "attempt.json", {"receipt_sha256": receipt_sha,
                                            "started_at": attempt_at})
            return _execute(rootfd, transport, credentials)
    finally:
        os.close(rootfd)


def _execute(rootfd: int, transport, credentials) -> dict:
    result = {"status": "failed", "model": MODEL, "policy": POLICY, "model_calls": 0,
              "known_tokens": 0, "known_usage": [], "usage_complete": True,
              "child_cleanup": "verified", "elapsed_seconds": 0.0,
              "first_fault": None}
    started = time.monotonic()
    phase = "before_call"
    prior = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame):
        raise CampaignTimeout()
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
        phase = "call_1"
        result["model_calls"] = 1
        result["usage_complete"] = False
        completed = transport.complete(credentials, PLAIN_SYSTEM, PLAIN_USER,
                                       None, timeout=CALL_SECONDS)
        result["known_usage"].append({"call": 1,
            "input_tokens": completed.input_tokens,
            "output_tokens": completed.output_tokens,
            "total_tokens": completed.total_tokens,
            "cached_input_tokens": completed.cached_input_tokens,
            "reasoning_output_tokens": completed.reasoning_output_tokens})
        result["known_tokens"] = completed.total_tokens
        result["usage_complete"] = True
        if result["known_tokens"] > MAX_TOKENS:
            raise ProbeError("token_cap_exceeded", phase)
        _check_output(completed.text)
        result["status"] = "passed"
    except BaseException as exc:
        result["first_fault"] = _fault(exc, phase, transport)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if multiprocessing.active_children():
            result["child_cleanup"] = "unverified"
            result["status"] = "failed"
            if result["first_fault"] is None:
                result["first_fault"] = {"code": "cleanup_failure", "phase": phase,
                                        "http_status": None, "body_shape": None,
                                        "media_type_class": None, "wire_observation": None,
                                        "stream_observation": None}
        if result["first_fault"] is not None and result["first_fault"]["code"] == "cleanup_failure":
            result["child_cleanup"] = "unverified"
    _validate_result(result)
    _create(rootfd, "result.json", result)
    return result


def status(root: Path, receipt_sha: str) -> dict:
    rootfd = _private_dir(root)
    try:
        receipt = _receipt(rootfd, receipt_sha)
        try:
            attempt, _ = _read(rootfd, "attempt.json")
        except ProbeError as exc:
            if exc.code == "file_unavailable":
                return {"status": "prepared", "model_calls": 0}
            raise
        if (set(attempt) != {"receipt_sha256", "started_at"}
                or attempt["receipt_sha256"] != receipt_sha
                or type(attempt["started_at"]) is not int
                or not receipt["created_at"] <= attempt["started_at"] < receipt["expires_at"]):
            raise ProbeError("attempt_mismatch")
        try:
            result, _ = _read(rootfd, "result.json")
        except ProbeError as exc:
            if exc.code == "file_unavailable":
                return {"status": "attempted_no_result", "usage_complete": False}
            raise
        _validate_result(result)
        return result
    finally:
        os.close(rootfd)


def _validate_result(result: dict) -> None:
    expected = {"status", "model", "policy", "model_calls", "known_tokens",
                "known_usage", "usage_complete", "child_cleanup",
                "elapsed_seconds", "first_fault"}
    if (type(result) is not dict or set(result) != expected
            or result.get("status") not in ("passed", "failed")
            or result.get("model") != MODEL or result.get("policy") != POLICY
            or type(result.get("model_calls")) is not int
            or not 0 <= result["model_calls"] <= MAX_CALLS
            or type(result.get("known_tokens")) is not int
            or not 0 <= result["known_tokens"]
            or type(result.get("known_usage")) is not list
            or len(result["known_usage"]) > result["model_calls"]
            or type(result.get("usage_complete")) is not bool
            or result.get("child_cleanup") not in ("verified", "unverified")
            or type(result.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(result["elapsed_seconds"])
            or not 0 <= result["elapsed_seconds"] <= MAX_SECONDS + 5):
        raise ProbeError("result_invalid")
    total = 0
    for index, usage in enumerate(result["known_usage"], 1):
        if (type(usage) is not dict or set(usage) != {"call", "input_tokens", "output_tokens",
            "total_tokens", "cached_input_tokens", "reasoning_output_tokens"}
                or type(usage["call"]) is not int or usage["call"] != index):
            raise ProbeError("result_invalid")
        for key in ("input_tokens", "output_tokens", "total_tokens",
                    "cached_input_tokens", "reasoning_output_tokens"):
            if type(usage[key]) is not int or usage[key] < 0:
                raise ProbeError("result_invalid")
        if (usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]
                or usage["input_tokens"] == 0 or usage["output_tokens"] == 0
                or usage["cached_input_tokens"] > usage["input_tokens"]
                or usage["reasoning_output_tokens"] > usage["output_tokens"]):
            raise ProbeError("result_invalid")
        total += usage["total_tokens"]
    if (total != result["known_tokens"]
            or result["usage_complete"] != (len(result["known_usage"]) == result["model_calls"])):
        raise ProbeError("result_invalid")
    fault = result["first_fault"]
    if result["status"] == "passed":
        if (fault is not None or result["model_calls"] != 1
                or result["usage_complete"] is not True
                or result["known_tokens"] > MAX_TOKENS
                or result["child_cleanup"] != "verified"):
            raise ProbeError("result_invalid")
    else:
        if (type(fault) is not dict
                or set(fault) != {"code", "phase", "http_status", "body_shape",
                                  "media_type_class", "wire_observation",
                                  "stream_observation"}
                or type(fault["code"]) is not str
                or fault["code"] not in PROBE_CODES | _transport_codes()
                or type(fault["phase"]) is not str
                or fault["phase"] not in ("before_call", "call_1")
                or (fault["http_status"] is not None and
                    (type(fault["http_status"]) is not int
                     or not 100 <= fault["http_status"] <= 599))
                or (fault["body_shape"] is not None and
                    (type(fault["body_shape"]) is not str
                     or fault["body_shape"] not in BODY_SHAPES))
                or (fault["media_type_class"] is not None and
                    (type(fault["media_type_class"]) is not str
                     or fault["media_type_class"] not in MEDIA_CLASSES))
                or not _valid_observation(fault["wire_observation"])
                or not _valid_stream_observation(fault["stream_observation"])):
            raise ProbeError("result_invalid")
        if ((result["model_calls"] == 0 and fault["phase"] != "before_call")
                or (result["model_calls"] == 1 and fault["phase"] != "call_1")
                or (result["model_calls"] == 0 and fault["code"] not in
                    {"campaign_timeout", "cleanup_failure", "internal_error"})
                or (fault["code"] == "plain_output_mismatch" and not result["usage_complete"])
                or (fault["code"] == "token_cap_exceeded"
                    and (not result["usage_complete"] or result["known_tokens"] <= MAX_TOKENS))
                or (fault["code"] in (_transport_codes() - {"cleanup_failure"})
                    and result["usage_complete"])
                or (fault["code"] == "cleanup_failure"
                    and result["child_cleanup"] != "unverified")
                or (fault["code"] in PROBE_CODES
                    and (fault["http_status"] is not None or fault["body_shape"] is not None
                         or fault["media_type_class"] is not None
                         or fault["wire_observation"] is not None
                         or fault["stream_observation"] is not None))
                or (fault["wire_observation"] is not None and
                    (fault["code"] not in PROVIDER_CODES | {"invalid_content_type"}
                     or fault["http_status"] != 200
                     or fault["media_type_class"] is None
                     or fault["body_shape"] in (None, "sse_event")
                     or (fault["media_type_class"] in {"sse", "missing"}
                         and fault["wire_observation"]["header_defect"] == "none"
                         and fault["wire_observation"]["content_encoding"] in {"missing", "identity"})
                     or (fault["code"] in PROVIDER_CODES and
                         fault["body_shape"] not in {"error_object", "detail", "other_json"})
                     or (fault["body_shape"] == "empty") !=
                         (fault["wire_observation"]["body_bytes"] == 0)
                     or (fault["body_shape"] == "oversized") !=
                         fault["wire_observation"]["body_truncated"]))
                or (fault["stream_observation"] is not None and
                    (fault["code"] not in _transport_codes()
                     or fault["http_status"] != 200
                     or fault["body_shape"] != "sse_event"
                     or fault["media_type_class"] not in {"sse", "missing"}
                     or fault["wire_observation"] is not None
                     or result["usage_complete"]
                     or result["known_usage"] != []
                     or result["known_tokens"] != 0))):
            raise ProbeError("result_invalid")


def _valid_observation(value) -> bool:
    if value is None:
        return True
    if type(value) is not dict or set(value) != {
            "header_defect", "parsed_header_count", "transfer_encoding",
            "content_encoding", "body_prefix", "body_bytes", "body_truncated",
            "sse_validation"}:
        return False
    count, size = value["parsed_header_count"], value["body_bytes"]
    if not (type(value["header_defect"]) is str and value["header_defect"] in {
            "none", "missing_separator", "first_continuation", "other", "unknown"}
            and (count is None or type(count) is int and 0 <= count <= 100)
            and type(value["transfer_encoding"]) is str and value["transfer_encoding"] in {
                "missing", "chunked", "identity", "other", "invalid"}
            and type(value["content_encoding"]) is str and value["content_encoding"] in {
                "missing", "identity", "gzip", "deflate", "br", "other", "invalid"}
            and type(value["body_prefix"]) is str and value["body_prefix"] in {
                "empty", "sse_prefix", "html_prefix", "http_prefix", "json_prefix",
                "text", "binary", "unknown"}
            and type(size) is int and 0 <= size <= 65537
            and type(value["body_truncated"]) is bool
            and type(value["sse_validation"]) is str and value["sse_validation"] in {
                "not_checked", "validated_completion", "invalid", "truncated"}):
        return False
    if value["body_truncated"] != (size > 65536) or (value["body_prefix"] == "empty") != (size == 0):
        return False
    if value["body_prefix"] != "sse_prefix":
        return value["sse_validation"] == "not_checked"
    if value["body_truncated"]:
        return value["sse_validation"] == "truncated"
    return value["sse_validation"] in {"validated_completion", "invalid"}


def _valid_stream_observation(value) -> bool:
    if value is None:
        return True
    if type(value) is not dict or set(value) != {
            "event_type", "terminal_status", "terminal_model_matches",
            "terminal_output_kind", "terminal_channel", "terminal_content_kind",
            "terminal_output_state", "finalized_item_count", "output_reconstructed"}:
        return False
    event = value["event_type"]
    status = value["terminal_status"]
    model = value["terminal_model_matches"]
    output = value["terminal_output_kind"]
    channel = value["terminal_channel"]
    content = value["terminal_content_kind"]
    output_state = value["terminal_output_state"]
    count = value["finalized_item_count"]
    reconstructed = value["output_reconstructed"]
    return (type(event) is str and event in {
        "response.created", "response.in_progress", "response.queued",
        "response.output_item.added", "response.output_item.done",
        "response.content_part.added", "response.content_part.done",
        "response.output_text.delta", "response.output_text.done",
        "response.reasoning_text.delta", "response.reasoning_text.done",
        "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
        "response.refusal.delta", "response.refusal.done",
        "response.function_call_arguments.delta", "response.function_call_arguments.done",
        "response.output_text.annotation.added", "response.completed",
        "response.failed", "response.incomplete", "error", "unknown"}
        and type(status) is str and status in {
            "missing", "completed", "incomplete", "failed", "other"}
        and (model is None or type(model) is bool)
        and type(output) is str and output in {
            "missing", "message", "reasoning", "mixed", "other"}
        and type(channel) is str and channel in {
            "missing", "final", "final_answer", "analysis", "commentary", "mixed", "other"}
        and type(content) is str and content in {
            "missing", "output_text", "refusal", "reasoning_text", "mixed", "other"}
        and type(output_state) is str and output_state in {
            "unseen", "missing", "null", "empty", "list", "invalid"}
        and type(count) is int and 0 <= count <= MAX_MEANINGFUL_EVENTS
        and type(reconstructed) is bool
        and (not reconstructed or (output_state in {"missing", "null", "empty"} and count > 0)))


def _transport_codes() -> frozenset[str]:
    return frozenset({
        "invalid_credentials", "invalid_request", "request_limit", "invalid_timeout",
        "invalid_event", "invalid_usage", "missing_usage", "incomplete_response",
        "invalid_output", "unsupported_output", "output_limit", "event_limit",
        "event_after_completion", "response_failure", "missing_completion", "wire_limit",
        "truncated_stream", "http_failure", "invalid_content_type", "auth_failure",
        "access_failure", "quota_failure", "model_mismatch", "transport_failure",
        "timeout", "cleanup_failure", "subscription_sharing_user_not_eligible",
        "subscription_sharing_usage_limit_exceeded", "subscription_sharing_usage_unavailable",
        "subscription_sharing_unsupported_capability", "subscription_sharing_route_not_supported",
        "subscription_sharing_invalid_user", "subscription_sharing_user_unavailable",
        "chatpass_v2_scope_not_authorized", "chatpass_v2_invalid_authorization_context",
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "status"):
        part = sub.add_parser(name)
        part.add_argument("--root", type=Path, required=True)
        if name != "status":
            part.add_argument("--state-dir", type=Path, required=True)
        if name != "prepare":
            part.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.root, args.state_dir)
        elif args.command == "run":
            result = run(args.root, args.state_dir, args.receipt_sha256)
        else:
            result = status(args.root, args.receipt_sha256)
    except ProbeError as exc:
        result = {"status": "failed", "error": exc.code}
    except Exception:
        result = {"status": "failed", "error": "internal_error"}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] in ("prepared", "passed") else 1


if __name__ == "__main__":
    sys.exit(main())
