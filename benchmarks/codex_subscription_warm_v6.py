"""Opt-in, bounded private evidence for failed warm turns.

The pinned v5 parser alone decides whether a notification is retry progress and
whether a turn succeeds. This observer never changes its event stream or limits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import types
from typing import Any


_v5_path = Path(__file__).resolve().with_name("codex_subscription_warm_v5.py")
_v5_source = _v5_path.read_bytes()
if hashlib.sha256(_v5_source).hexdigest() != "2df1ead8f6f1cee1f138075aa77d78c61ed28959195a7634ce59a0df00290702":
    raise RuntimeError("pinned_warm_v5_source_mismatch")
v5 = types.ModuleType("pinned_codex_subscription_warm_v6_base")
v5.__file__ = str(_v5_path)
sys.modules[v5.__name__] = v5
exec(compile(_v5_source, str(_v5_path), "exec"), v5.__dict__)

base = v5.base
concurrent = v5.concurrent
BudgetLimits = v5.BudgetLimits
ConcurrentStop = v5.ConcurrentStop
SharedBudget = v5.SharedBudget
serialize_failure = v5.serialize_failure

MAX_PRIVATE_TEXT_BYTES = 4096
MAX_PRIVATE_RECORD_BYTES = 20_480
MAX_PRIVATE_RECORDS = 16


def _bounded_text(value: str) -> tuple[str, bool]:
    normalized = value.encode("utf-8", errors="replace").decode("utf-8")
    used = 0
    end = 0
    for char in normalized:
        # JSON escapes can expand a one-byte control character to six bytes.
        size = len(json.dumps(char, ensure_ascii=False)[1:-1].encode("utf-8"))
        if used + size > MAX_PRIVATE_TEXT_BYTES:
            break
        used += size
        end += 1
    return normalized[:end], end < len(normalized)


def _valid_misalignment(value: Any) -> bool:
    if value is None:
        return True
    if type(value) is not dict or set(value) - {"errorType", "detailedExplanation", "steer"}:
        return False
    for name in ("errorType", "detailedExplanation"):
        if name in value and value[name] is not None and type(value[name]) is not str:
            return False
    steer = value.get("steer")
    return (steer is None or (type(steer) is dict and set(steer) == {"message"}
                              and type(steer["message"]) is str))


def _private_error(event: Any, thread_id: Any, turn_id: Any) -> dict[str, Any] | None:
    """Validate the documented error shape and retain no event or identity."""
    if type(event) is not dict or set(event) != {"method", "params"} or event.get("method") != "error":
        return None
    params = event.get("params")
    if (type(params) is not dict or set(params) != {"threadId", "turnId", "willRetry", "error"}
            or type(thread_id) is not str or not thread_id or type(turn_id) is not str or not turn_id
            or type(params.get("threadId")) is not str or params["threadId"] != thread_id
            or type(params.get("turnId")) is not str or params["turnId"] != turn_id
            or type(params.get("willRetry")) is not bool):
        return None
    error = params.get("error")
    if (type(error) is not dict or "message" not in error
            or set(error) - {"message", "codexErrorInfo", "additionalDetails", "misalignment"}
            or type(error.get("message")) is not str
            or ("additionalDetails" in error and error["additionalDetails"] is not None
                and type(error["additionalDetails"]) is not str)):
        return None
    misalignment = error.get("misalignment")
    if not _valid_misalignment(misalignment):
        return None
    info = error.get("codexErrorInfo")
    if info is None:
        kind, status = "unspecified", None
    elif type(info) is str:
        if info not in v5.v4.v3._STRING_ERRORS:
            return None
        kind, status = info, None
    elif type(info) is dict and len(info) == 1:
        kind, detail = next(iter(info.items()))
        if kind in v5.v4.v3._OBJECT_ERRORS:
            if type(detail) is not dict or set(detail) - {"httpStatusCode"}:
                return None
            status = detail.get("httpStatusCode")
            if status is not None and (type(status) is not int or not 0 <= status <= 65535):
                return None
        elif kind == "activeTurnNotSteerable":
            if (type(detail) is not dict or set(detail) != {"turnKind"}
                    or type(detail["turnKind"]) is not str or detail["turnKind"] not in {"review", "compact"}):
                return None
            status = None
        else:
            return None
    else:
        return None
    message, message_truncated = _bounded_text(error["message"])
    result = {"error_class": kind, "http_status_code": status,
              "will_retry": params["willRetry"], "message": message,
              "message_truncated": message_truncated,
              "misalignment_present": misalignment is not None}
    if type(error.get("additionalDetails")) is str:
        details, truncated = _bounded_text(error["additionalDetails"])
        result["additional_details"] = details
        result["additional_details_truncated"] = truncated
    return result


class WarmSession(v5.WarmSession):
    def __init__(self, binary: str, cwd: str, timeout: float = 120):
        super().__init__(binary, cwd, timeout)
        self.reset_private_errors()

    def reset_private_errors(self) -> None:
        self.private_error_first: dict[str, Any] | None = None
        self.private_error_last: dict[str, Any] | None = None
        self.private_error_count = 0

    def rpc(self, method: str, params: dict[str, Any], *, preserve_notifications: bool = False) -> dict[str, Any]:
        if method == "turn/start":
            try:
                self.reset_private_errors()
            except Exception:
                pass
        return super().rpc(method, params, preserve_notifications=preserve_notifications)

    def next_event(self) -> dict[str, Any]:
        event = super().next_event()
        try:
            observed = _private_error(event, getattr(self, "_observation_thread", None),
                                      getattr(self, "_observation_turn", None))
            if observed is not None:
                if self.private_error_first is None:
                    self.private_error_first = observed
                self.private_error_last = observed
                self.private_error_count = min(self.private_error_count + 1, base.MAX_EVENTS)
        except Exception:
            try:
                self.reset_private_errors()
            except Exception:
                pass
        return event

    def private_failure_record(self, code: str) -> dict[str, Any] | None:
        if self.private_error_count == 0:
            return None
        safe_code = serialize_failure({"code": code, "phase": "run", "rpc": "turn/events"})
        return {"schema": "warm_private_failure_v1", "failure_code": safe_code["code"],
                "error_count": self.private_error_count,
                "first": self.private_error_first, "last": self.private_error_last}


class PrivateFailureSink:
    """Write one record to a free fixed slot in an existing owned 0700 dir."""

    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = os.fspath(directory)
        if not isinstance(self.directory, str) or not os.path.isabs(self.directory):
            raise ValueError("private_directory_invalid")

    @staticmethod
    def _valid_record(record: Any) -> bool:
        if (type(record) is not dict or set(record) != {"schema", "failure_code", "error_count", "first", "last"}
                or record["schema"] != "warm_private_failure_v1"
                or type(record["error_count"]) is not int or not 1 <= record["error_count"] <= base.MAX_EVENTS
                or type(record["failure_code"]) is not str):
            return False
        safe = serialize_failure({"code": record["failure_code"], "phase": "run", "rpc": "turn/events"})
        if safe is None or safe["code"] != record["failure_code"]:
            return False
        for key in ("first", "last"):
            item = record[key]
            required = {"error_class", "http_status_code", "will_retry", "message", "message_truncated",
                        "misalignment_present"}
            optional = {"additional_details", "additional_details_truncated"}
            if (type(item) is not dict or not required <= set(item) or set(item) - required - optional
                    or type(item["error_class"]) is not str
                    or item["error_class"] not in (v5.v4.v3._STRING_ERRORS | v5.v4.v3._OBJECT_ERRORS | {"activeTurnNotSteerable", "unspecified"})
                    or (item["http_status_code"] is not None and
                        (type(item["http_status_code"]) is not int or not 0 <= item["http_status_code"] <= 65535))
                    or type(item["will_retry"]) is not bool or type(item["message"]) is not str
                    or len(json.dumps(item["message"], ensure_ascii=False).encode("utf-8")) - 2 > MAX_PRIVATE_TEXT_BYTES
                    or type(item["message_truncated"]) is not bool
                    or type(item["misalignment_present"]) is not bool
                    or ("additional_details" in item) != ("additional_details_truncated" in item)):
                return False
            if "additional_details" in item and (
                    type(item["additional_details"]) is not str
                    or len(json.dumps(item["additional_details"], ensure_ascii=False).encode("utf-8")) - 2 > MAX_PRIVATE_TEXT_BYTES
                    or type(item["additional_details_truncated"]) is not bool):
                return False
        return True

    def _open_directory(self) -> int:
        parts = Path(self.directory).parts
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in parts[1:]:
                if part in {"", ".", ".."}:
                    raise ValueError("private_directory_invalid")
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            mode = os.fstat(fd)
            if not stat.S_ISDIR(mode.st_mode) or mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) != 0o700:
                raise ValueError("private_directory_unsafe")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def write(self, record: dict[str, Any]) -> str:
        if not self._valid_record(record):
            raise ValueError("private_record_invalid")
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_PRIVATE_RECORD_BYTES:
            raise ValueError("private_record_too_large")
        directory_fd = self._open_directory()
        temp_name = ".warm-private-" + secrets.token_hex(16)
        temp_fd = None
        created = False
        try:
            temp_fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                              0o600, dir_fd=directory_fd)
            created = True
            os.fchmod(temp_fd, 0o600)
            with os.fdopen(temp_fd, "wb", closefd=True) as stream:
                temp_fd = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            for index in range(MAX_PRIVATE_RECORDS):
                name = f"warm-private-failure-{index:02d}.json"
                try:
                    existing = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    existing = None
                if existing is not None:
                    if (not stat.S_ISREG(existing.st_mode) or existing.st_uid != os.getuid()
                            or stat.S_IMODE(existing.st_mode) != 0o600):
                        raise ValueError("private_slot_unsafe")
                    continue
                try:
                    os.link(temp_name, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                            follow_symlinks=False)
                    os.fsync(directory_fd)
                    return name
                except FileExistsError:
                    continue
            raise ValueError("private_record_cap")
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            if created:
                try:
                    os.unlink(temp_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            os.close(directory_fd)


class WarmSubscriptionClient(v5.WarmSubscriptionClient):
    def __init__(self, *args: Any, session_factory: Any = WarmSession,
                 private_failure_sink: PrivateFailureSink | None = None, **kwargs: Any):
        super().__init__(*args, session_factory=session_factory, **kwargs)
        self.private_failure_sink = private_failure_sink
        self.private_sink_status: str | None = None

    def _complete_locked(self, request: Any) -> str:
        self.private_sink_status = None
        try:
            return super()._complete_locked(request)
        finally:
            session = self.session
            reset = getattr(session, "reset_private_errors", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    self.private_sink_status = "failed"

    def _record_failure(self, code: str, phase: str, turn_admitted: bool,
                        known_usage: bool) -> None:
        session = self.session
        record = None
        try:
            if phase == "run" and turn_admitted and session is not None:
                make_record = getattr(session, "private_failure_record", None)
                if callable(make_record):
                    record = make_record(code)
        except Exception:
            self.private_sink_status = "failed"
        super()._record_failure(code, phase, turn_admitted, known_usage)
        if record is not None and self.private_failure_sink is not None:
            try:
                self.private_failure_sink.write(record)
                self.private_sink_status = "written"
            except Exception:
                self.private_sink_status = "failed"
        reset = getattr(session, "reset_private_errors", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                self.private_sink_status = "failed"
