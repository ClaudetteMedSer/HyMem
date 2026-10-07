"""One-call, structured-only SIWC access probe; never an LME result.

Preparation binds an invented fixture and immutable sources to a private receipt.
Run consumes that receipt once, after local credential preflight. No refresh or
browser action occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "tools/diagnostics/lme_chatgpt_plan_probe_v6.py"
BASE_SHA256 = "ffa34a894344e0d34ac80b2f8b1446308ad3c1f67e50336e9901e9f029d4dee7"
MODEL = "gpt-5.6-luna"
EFFORT = "low"
POLICY = "siwc_server_enforced_plan_or_existing_credits_v1"
PURPOSE = "structured_output_access_validation_v1"
MAX_CALLS = 1
MAX_TOKENS = 160_000
MAX_SECONDS = 300
CALL_SECONDS = 120
RECEIPT_LIFETIME = 600
STRUCTURED_SYSTEM = "Return a JSON object with ready set to true."
STRUCTURED_USER = "Is the invented paper kite ready?"
SCHEMA = {"type": "object", "properties": {"ready": {"type": "boolean"}},
          "required": ["ready"], "additionalProperties": False}
OUTPUT_CODES = frozenset({"structured_output_invalid", "structured_output_mismatch"})


class ProbeError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class CampaignTimeout(BaseException):
    pass


def _hash_file(path: Path, expected: str) -> str:
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
    return digest


def _base():
    _hash_file(BASE_PATH, BASE_SHA256)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    base = importlib.import_module("tools.diagnostics.lme_chatgpt_plan_probe_v6")
    if Path(base.__file__).resolve() != BASE_PATH:
        raise ProbeError("source_mismatch")
    _hash_file(BASE_PATH, BASE_SHA256)
    return base


def _source_hashes() -> dict:
    base = _base()
    hashes = base._source_hashes()
    if hashes.pop("probe") != BASE_SHA256:
        raise ProbeError("source_mismatch")
    hashes["probe_v6"] = BASE_SHA256
    hashes["probe"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def _modules():
    base = _base()
    hashes = _source_hashes()
    catalog, transport = base._modules()
    if (transport.MODEL != MODEL or transport.MAX_WALL_SECONDS != CALL_SECONDS
            or hashes["transport_v6"] != base.SOURCES["transport_v6"][1]):
        raise ProbeError("source_mismatch")
    _source_hashes()
    return catalog, transport


def _fixtures() -> dict:
    return {"structured_system": STRUCTURED_SYSTEM, "structured_user": STRUCTURED_USER,
            "schema": SCHEMA}


def _exact_json(value, expected) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(value) == set(expected) and
                all(_exact_json(value[key], expected[key]) for key in expected))
    if type(expected) is list:
        return (len(value) == len(expected) and
                all(_exact_json(left, right) for left, right in zip(value, expected)))
    return value == expected


def _receipt(rootfd: int, expected_sha: str | None = None) -> dict:
    base = _base()
    receipt, digest = base._read(rootfd, "receipt.json")
    if expected_sha is not None and digest != expected_sha:
        raise ProbeError("receipt_mismatch")
    expected = {"version", "purpose", "sources", "model", "effort", "policy",
                "limits", "fixtures", "state_dir", "host_id", "client_id",
                "created_at", "expires_at"}
    if (set(receipt) != expected or type(receipt["version"]) is not int
            or receipt["version"] != 7 or receipt["purpose"] != PURPOSE
            or receipt["sources"] != _source_hashes()
            or receipt["model"] != MODEL or receipt["effort"] != EFFORT
            or receipt["policy"] != POLICY
            or not _exact_json(receipt["limits"], {"calls": MAX_CALLS,
                "observed_tokens": MAX_TOKENS, "campaign_seconds": MAX_SECONDS,
                "child_seconds": CALL_SECONDS})
            or not _exact_json(receipt["fixtures"], _fixtures())
            or type(receipt["state_dir"]) is not str
            or type(receipt["host_id"]) is not str
            or not receipt["host_id"].startswith("urn:uuid:")
            or type(receipt["client_id"]) is not str or not receipt["client_id"]
            or type(receipt["created_at"]) is not int
            or type(receipt["expires_at"]) is not int
            or receipt["expires_at"] - receipt["created_at"] != RECEIPT_LIFETIME):
        raise ProbeError("receipt_invalid")
    return receipt


def prepare(root: Path, state: Path) -> dict:
    base = _base()
    hashes = _source_hashes()
    catalog, _ = _modules()
    rootfd = base._private_dir(root)
    statefd = base._private_dir(state)
    try:
        # Only host/registration metadata is read; access tokens are not read.
        host_id, client_id = base._identity(statefd, catalog)
        if os.listdir(root):
            raise ProbeError("root_not_empty")
        now = int(time.time())
        receipt = {"version": 7, "purpose": PURPOSE, "sources": hashes,
                   "model": MODEL, "effort": EFFORT, "policy": POLICY,
                   "limits": {"calls": MAX_CALLS, "observed_tokens": MAX_TOKENS,
                              "campaign_seconds": MAX_SECONDS, "child_seconds": CALL_SECONDS},
                   "fixtures": _fixtures(), "state_dir": str(state), "host_id": host_id,
                   "client_id": client_id, "created_at": now,
                   "expires_at": now + RECEIPT_LIFETIME}
        digest = base._create(rootfd, "receipt.json", receipt)
        return {"status": "prepared", "receipt_sha256": digest,
                "model": MODEL, "policy": POLICY, "model_calls": 0}
    finally:
        os.close(statefd)
        os.close(rootfd)


def _check_output(text: str) -> None:
    if type(text) is not str:
        raise ProbeError("structured_output_invalid")
    try:
        value = json.loads(text, object_pairs_hook=_base()._unique)
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise ProbeError("structured_output_invalid") from None
    if (type(value) is not dict or set(value) != {"ready"}
            or type(value["ready"]) is not bool or value["ready"] is not True):
        raise ProbeError("structured_output_mismatch")


def run(root: Path, state: Path, receipt_sha: str) -> dict:
    base = _base()
    rootfd = base._private_dir(root)
    try:
        receipt = _receipt(rootfd, receipt_sha)
        if receipt["state_dir"] != str(state):
            raise ProbeError("state_binding_mismatch")
        if int(time.time()) >= receipt["expires_at"]:
            raise ProbeError("receipt_expired")
        catalog, transport = _modules()
        signin = catalog._load_signin_module()
        with signin._flow_lock(state):
            statefd = base._private_dir(state)
            try:
                if base._identity(statefd, catalog) != (receipt["host_id"], receipt["client_id"]):
                    raise ProbeError("identity_binding_invalid")
                token = catalog._validated_access_token(statefd, signin, int(time.time()))
            finally:
                os.close(statefd)
            credentials = transport.Credentials(token)
            attempt_at = int(time.time())
            if attempt_at >= receipt["expires_at"]:
                raise ProbeError("receipt_expired")
            base._create(rootfd, "attempt.json", {"receipt_sha256": receipt_sha,
                                                   "started_at": attempt_at})
            return _execute(rootfd, transport, credentials)
    finally:
        os.close(rootfd)


def _fault(exc: BaseException, phase: str, transport) -> dict:
    if isinstance(exc, ProbeError) and exc.code in OUTPUT_CODES:
        return {"code": exc.code, "phase": phase, "http_status": None,
                "body_shape": None, "media_type_class": None,
                "wire_observation": None, "stream_observation": None}
    if isinstance(exc, CampaignTimeout):
        exc = _base().CampaignTimeout()
    return _base()._fault(exc, phase, transport)


def _execute(rootfd: int, transport, credentials) -> dict:
    base = _base()
    result = {"status": "failed", "model": MODEL, "policy": POLICY,
              "model_calls": 0, "known_tokens": 0, "known_usage": [],
              "usage_complete": True, "child_cleanup": "verified",
              "elapsed_seconds": 0.0, "first_fault": None}
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
        completed = transport.complete(credentials, STRUCTURED_SYSTEM, STRUCTURED_USER,
                                       SCHEMA, timeout=CALL_SECONDS)
        result["known_usage"].append({"call": 1, "input_tokens": completed.input_tokens,
            "output_tokens": completed.output_tokens, "total_tokens": completed.total_tokens,
            "cached_input_tokens": completed.cached_input_tokens,
            "reasoning_output_tokens": completed.reasoning_output_tokens})
        result["known_tokens"] = completed.total_tokens
        result["usage_complete"] = True
        if result["known_tokens"] > MAX_TOKENS:
            raise base.ProbeError("token_cap_exceeded", phase)
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
                    "http_status": None, "body_shape": None, "media_type_class": None,
                    "wire_observation": None, "stream_observation": None}
        if (result["first_fault"] is not None
                and result["first_fault"]["code"] == "cleanup_failure"):
            result["child_cleanup"] = "unverified"
    _validate_result(result)
    base._create(rootfd, "result.json", result)
    return result


def _validate_result(result: dict) -> None:
    base = _base()
    # Reuse the pinned v6 finite result and usage validator. Its one plain-text
    # mismatch code is mapped only in a temporary copy; saved results retain the
    # structured code, and the structured parser above is the success gate.
    if type(result) is not dict:
        raise ProbeError("result_invalid")
    fault = result.get("first_fault")
    if type(fault) is dict and fault.get("code") == "plain_output_mismatch":
        raise ProbeError("result_invalid")
    if (type(fault) is dict and type(fault.get("code")) is str
            and fault["code"] in OUTPUT_CODES):
        if (result.get("status") != "failed" or result.get("model_calls") != 1
                or result.get("usage_complete") is not True):
            raise ProbeError("result_invalid")
        mapped = {**result, "first_fault": {**fault, "code": "plain_output_mismatch"}}
    else:
        mapped = result
    base._validate_result(mapped)


def status(root: Path, receipt_sha: str) -> dict:
    base = _base()
    rootfd = base._private_dir(root)
    try:
        receipt = _receipt(rootfd, receipt_sha)
        try:
            attempt, _ = base._read(rootfd, "attempt.json")
        except base.ProbeError as exc:
            if exc.code == "file_unavailable":
                return {"status": "prepared", "model_calls": 0}
            raise
        if (set(attempt) != {"receipt_sha256", "started_at"}
                or attempt["receipt_sha256"] != receipt_sha
                or type(attempt["started_at"]) is not int
                or not receipt["created_at"] <= attempt["started_at"] < receipt["expires_at"]):
            raise ProbeError("attempt_mismatch")
        try:
            result, _ = base._read(rootfd, "result.json")
        except base.ProbeError as exc:
            if exc.code == "file_unavailable":
                return {"status": "attempted_no_result", "usage_complete": False}
            raise
        _validate_result(result)
        return result
    finally:
        os.close(rootfd)


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
    except Exception:
        exc = sys.exc_info()[1]
        code = getattr(exc, "code", None)
        result = {"status": "failed", "error": code if type(code) is str and
                  code in {"source_mismatch", "source_unavailable", "receipt_mismatch",
                           "receipt_invalid", "directory_invalid", "directory_unavailable",
                           "directory_not_private", "file_unavailable", "file_not_private",
                           "file_too_large", "file_invalid", "root_not_empty",
                           "identity_binding_invalid", "state_binding_mismatch",
                           "receipt_expired", "already_exists", "write_failed",
                           "attempt_mismatch", "result_invalid", "invalid_credentials"}
                  else "internal_error"}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] in ("prepared", "passed") else 1


if __name__ == "__main__":
    sys.exit(main())
