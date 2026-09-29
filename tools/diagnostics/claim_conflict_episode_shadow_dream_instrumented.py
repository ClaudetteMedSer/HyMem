#!/usr/bin/env python3
"""Private, bounded targeted dream on a clone of the post-failure store.

Run with ``python -I -B`` in the pinned R7 container. Stdout is metadata only.
All requests, replies, traceback details, and replay inputs stay under /work.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import threading
import traceback
import types

sys.path.insert(0, "/candidate")
from hymem.deadline import DeadlineExceeded

SOURCE = Path("/reference/source.sqlite")
WORK = Path("/work")
SOURCE_SHA256 = "7da93a6ab67937079a3df192faae9b92bc4885152ea6e5cb83d17b1f2f8ec9c0"
PHASE1_SHA256 = "31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136"
TARGET_CHUNK = "chk_c14182771bd8e2a7e583f7d693cee29a5fb159fe"
GENERATION = "hymem-phase1-generation-v1:6075085e12e32e1b790e49b99b8c3bb50718b18be0f28762d1582c58ee8e35eb"
MAX_COMPLETIONS = 128
MAX_HTTP_ATTEMPTS = 896
MAX_LLM_HTTP_ATTEMPTS = 384
MAX_EMBEDDING_HTTP_ATTEMPTS = 512
DEFAULT_DEADLINE_SECONDS = 2700
MAX_EMBEDDING_TEXTS = 16
MAX_EMBEDDING_CHARS = 128_000
MAX_EMBEDDING_UTF8_BYTES = 512_000
MAX_PREPERSIST = 32
MAX_EXCEPTION_EVENTS = 256
ENV_KEYS = frozenset({
    "HYMEM_LLM_API_KEY", "HYMEM_LLM_BASE_URL", "HYMEM_LLM_MODEL",
    "HYMEM_LLM_THINKING", "HYMEM_LLM_EXTRA_BODY",
    "HYMEM_LLM_DEPLOYMENT_REVISION", "HYMEM_LLM_DEPLOYMENT_TENANT",
    "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
    "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
    "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM",
    "HYMEM_EMBEDDING_PIN_DIMENSION", "HYMEM_EMBEDDING_DEPLOYMENT_REVISION",
    "HYMEM_EMBEDDING_DEPLOYMENT_TENANT",
    "HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP",
    "HYMEM_EMBEDDING_TIMEOUT_SECONDS",
    "HYMEM_AGGREGATION_NODES_ENABLED", "HYMEM_AGGREGATION_DIGEST_ENABLED",
})
FRAME_RE = re.compile(r"hymem(?:/[A-Za-z_][A-Za-z_0-9]*)*/[A-Za-z_][A-Za-z_0-9]*\.py\Z")
STAGE = "startup"


class BudgetStop(DeadlineExceeded):
    pass


class InstrumentationStop(DeadlineExceeded):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(path, 0o700)


def private_write(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def private_json(path: Path, value: object) -> None:
    private_write(path, json_bytes(value))


def artifact_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".jsonl")
                               or path.name.startswith("prepersist-") and path.suffix == ".sqlite"):
            digest.update(path.name.encode("ascii"))
            digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def load_env(path: Path, root: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("runtime_environment_not_private_regular_file")
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in values.items()
    ) or set(values) - ENV_KEYS:
        raise ValueError("runtime_environment_invalid")
    for key in tuple(os.environ):
        if key.startswith(("HYMEM_", "DEEPSEEK_", "OPENAI_")):
            del os.environ[key]
    os.environ.update(values)
    os.environ["HYMEM_ROOT"] = str(root)
    if (os.environ.get("HYMEM_LLM_BASE_URL", "").rstrip("/") != "https://api.deepseek.com"
            or os.environ.get("HYMEM_LLM_MODEL") != "deepseek-flash"
            or not os.environ.get("HYMEM_LLM_API_KEY")):
        raise ValueError("runtime_llm_identity_invalid")


def clone_source(target: Path) -> None:
    if target.exists():
        raise FileExistsError("clone_already_exists")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    source = sqlite3.connect(f"file:{SOURCE}?mode=ro&immutable=1", uri=True)
    try:
        dest = sqlite3.connect(target)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()


def target_session(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT session_id,text,source_manifest_version,source_manifest_count "
        "FROM chunks WHERE id=? AND chunk_kind='extraction'", (TARGET_CHUNK,)
    ).fetchone()
    if row is None or len(row[1]) != 1298 or row[2] != "claim-source-manifest-v1" or row[3] != 2:
        raise ValueError("target_chunk_contract_changed")
    mids = tuple(int(item[0]) for item in conn.execute(
        "SELECT source_message_id FROM chunk_message_sources "
        "WHERE chunk_id=? ORDER BY ordinal", (TARGET_CHUNK,)
    ))
    if mids != (1016, 1017):
        raise ValueError("target_source_ids_changed")
    return str(row[0])


def safe_frames(exc: BaseException) -> list[dict]:
    frames = []
    for index, (frame, lineno) in enumerate(traceback.walk_tb(exc.__traceback__)):
        if index >= 96:
            break
        filename = frame.f_code.co_filename
        if not filename.startswith("/candidate/hymem/"):
            continue
        relative = filename[len("/candidate/"):]
        name = frame.f_code.co_name
        if FRAME_RE.fullmatch(relative) and name.isidentifier() and len(name) <= 96 and 1 <= lineno <= 100000:
            frames.append({"path": relative, "function": name, "line": lineno})
    return frames[-12:]


def safe_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if name in {"ValueError", "RuntimeError", "TypeError", "KeyError",
                            "AssertionError", "TimeoutError", "ConnectionError",
                            "OSError", "MemoryError", "BudgetStop",
                            "InstrumentationStop"} else "Exception"


class PrivateJournal:
    """Durable 0600 event logs. No private payload is sent to stdout."""

    def __init__(self, root: Path):
        self.root = root
        self._streams = {}
        for name in ("llm-events.jsonl", "exceptions.jsonl", "aggregation-exceptions.jsonl"):
            fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            self._streams[name] = os.fdopen(fd, "wb")
        self._lock = threading.RLock()

    def append(self, name: str, value: object) -> None:
        raw = json_bytes(value) + b"\n"
        with self._lock:
            stream = self._streams[name]
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


def _provider_response(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    raise ValueError("provider_response_cannot_be_captured_losslessly")


class Probe:
    """Observe frozen function code; do not replace client or producer methods."""

    def __init__(self, journal: PrivateJournal, db_path: Path, *,
                 completion_code: types.CodeType, attempt_code: types.CodeType,
                 embedding_code: types.CodeType,
                 extraction_code: types.CodeType, persist_code: types.CodeType,
                 max_completions: int = MAX_COMPLETIONS,
                 max_attempts: int = MAX_HTTP_ATTEMPTS,
                 max_llm_attempts: int = MAX_LLM_HTTP_ATTEMPTS,
                 max_embedding_attempts: int = MAX_EMBEDDING_HTTP_ATTEMPTS):
        self.journal = journal
        self.db_path = db_path
        self.completion_code = completion_code
        self.attempt_code = attempt_code
        self.embedding_code = embedding_code
        self.extraction_code = extraction_code
        self.persist_code = persist_code
        self.max_completions = max_completions
        self.max_attempts = max_attempts
        self.max_llm_attempts = max_llm_attempts
        self.max_embedding_attempts = max_embedding_attempts
        self.completions = 0
        self.attempts = 0
        self.llm_attempts = 0
        self.embedding_attempts = 0
        self.extractions = 0
        self.prepersist = 0
        self.exception_events = 0
        self.exception_controlflow_skipped = 0
        self.exception_capture_truncated = False
        self.budget_reason = None
        self.capture_error = False
        self._lock = threading.RLock()

    def profile(self, frame, event: str, value) -> None:
        try:
            self._profile(frame, event, value)
        except (BudgetStop, InstrumentationStop):
            raise
        except BaseException:
            self.capture_error = True
            raise InstrumentationStop("instrumentation_capture_failed") from None

    def _profile(self, frame, event: str, value) -> None:
        code = frame.f_code
        if event == "call":
            if code is self.completion_code:
                if self.capture_error:
                    raise InstrumentationStop("instrumentation_capture_incomplete")
                with self._lock:
                    if self.completions >= self.max_completions:
                        self.budget_reason = "completion_budget"
                        raise BudgetStop("completion_budget")
                    self.completions += 1
                    ordinal = self.completions
                request = frame.f_locals.get("request")
                self.journal.append("llm-events.jsonl", {
                    "event": "completion_request", "completion": ordinal,
                    "request": _jsonable(request),
                })
            elif code is self.attempt_code:
                if self.capture_error:
                    raise InstrumentationStop("instrumentation_capture_incomplete")
                with self._lock:
                    if self.llm_attempts >= self.max_llm_attempts:
                        self.budget_reason = "llm_http_attempt_budget"
                        raise BudgetStop("llm_http_attempt_budget")
                    if self.attempts >= self.max_attempts:
                        self.budget_reason = "http_attempt_budget"
                        raise BudgetStop("http_attempt_budget")
                    self.attempts += 1
                    self.llm_attempts += 1
                    ordinal = self.attempts
                self.journal.append("llm-events.jsonl", {
                    "event": "attempt_begin", "attempt": ordinal,
                    "completion": self.completions,
                })
            elif code is self.embedding_code:
                texts = frame.f_locals.get("texts")
                self._admit_embedding_payload(texts)
                if texts:
                    if self.capture_error:
                        raise InstrumentationStop("instrumentation_capture_incomplete")
                    with self._lock:
                        if self.embedding_attempts >= self.max_embedding_attempts:
                            self.budget_reason = "embedding_http_attempt_budget"
                            raise BudgetStop("embedding_http_attempt_budget")
                        if self.attempts >= self.max_attempts:
                            self.budget_reason = "http_attempt_budget"
                            raise BudgetStop("http_attempt_budget")
                        self.attempts += 1
                        self.embedding_attempts += 1
                        ordinal = self.attempts
                    self.journal.append("llm-events.jsonl", {
                        "event": "embedding_request", "attempt": ordinal,
                        "texts": list(texts),
                    })
            elif code is self.persist_code:
                self._capture_prepersist(frame)
        elif event == "return":
            if code is self.attempt_code and value is not None:
                self.journal.append("llm-events.jsonl", {
                    "event": "attempt_response", "attempt": self.attempts,
                    "completion": self.completions,
                    "response": _provider_response(value),
                })
            elif code is self.completion_code and isinstance(value, str):
                self.journal.append("llm-events.jsonl", {
                    "event": "completion_response", "completion": self.completions,
                    "response": value,
                })
            elif code is self.extraction_code and value is not None:
                chunk = frame.f_locals.get("chunk")
                self.extractions += 1
                private_json(self.journal.root / f"extraction-{self.extractions:03d}.json", {
                    "chunk": _jsonable(chunk), "extraction": _jsonable(value),
                })
            elif code is self.embedding_code and value is not None:
                self.journal.append("llm-events.jsonl", {
                    "event": "embedding_response", "attempt": self.attempts,
                    "vectors": value,
                })

    def _admit_embedding_payload(self, texts) -> None:
        """Reject content-free before the exact frozen transport method runs."""
        if type(texts) not in (list, tuple):
            self.budget_reason = "embedding_payload_invalid"
            raise BudgetStop("embedding_payload_invalid")
        if len(texts) > MAX_EMBEDDING_TEXTS:
            self.budget_reason = "embedding_payload_limit"
            raise BudgetStop("embedding_payload_limit")
        chars = utf8_bytes = 0
        for item in texts:
            if type(item) is not str:
                self.budget_reason = "embedding_payload_invalid"
                raise BudgetStop("embedding_payload_invalid")
            chars += len(item)
            if chars > MAX_EMBEDDING_CHARS:
                self.budget_reason = "embedding_payload_limit"
                raise BudgetStop("embedding_payload_limit")
            try:
                utf8_bytes += len(item.encode("utf-8"))
            except UnicodeError:
                self.budget_reason = "embedding_payload_invalid"
                raise BudgetStop("embedding_payload_invalid") from None
            if utf8_bytes > MAX_EMBEDDING_UTF8_BYTES:
                self.budget_reason = "embedding_payload_limit"
                raise BudgetStop("embedding_payload_limit")

    def _capture_prepersist(self, frame) -> None:
        if self.prepersist >= MAX_PREPERSIST:
            self.budget_reason = "prepersist_capture_budget"
            raise BudgetStop("prepersist_capture_budget")
        self.prepersist += 1
        ordinal = self.prepersist
        inputs = frame.f_locals
        # A distinct read-only connection observes the last committed state,
        # even though the runner has entered BEGIN IMMEDIATE on its writer.
        target = self.journal.root / f"prepersist-{ordinal:03d}.sqlite"
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        reader = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            backup = sqlite3.connect(target)
            try:
                reader.backup(backup)
            finally:
                backup.close()
        finally:
            reader.close()
        private_json(self.journal.root / f"prepersist-{ordinal:03d}.json", {
            "chunk": _jsonable(inputs.get("chunk")),
            "extraction": _jsonable(inputs.get("extraction")),
            "dedup_vectors": _jsonable(inputs.get("dedup_vectors")),
            "dedup_model": getattr(inputs.get("dedup_vectors"), "model", None),
            "dedup_dim": getattr(inputs.get("dedup_vectors"), "dim", None),
            "in_cycle_edges": _jsonable(inputs.get("in_cycle_edges")),
            "cfg": _jsonable(inputs.get("cfg")),
            "prompt_version": inputs.get("prompt_version"),
            "source_sha256": SOURCE_SHA256,
            "database_sha256": sha(target),
        })

    def trace(self, frame, event: str, _value):
        if event != "call":
            return None
        filename = frame.f_code.co_filename
        if not filename.startswith("/candidate/hymem/"):
            return None
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False
        return self._trace_local

    def _trace_local(self, frame, event: str, value):
        if event != "exception":
            return self._trace_local
        kind, exc, tb = value
        if issubclass(kind, (StopIteration, StopAsyncIteration, GeneratorExit)):
            self.exception_controlflow_skipped += 1
            return self._trace_local
        # Independent bounded critical lane survives ordinary journal saturation.
        filename = getattr(getattr(frame, "f_code", None), "co_filename", "")
        critical = (filename == "/candidate/hymem/dreaming/runner.py"
                    and 2873 <= getattr(frame, "f_lineno", 0) <= 2935)
        count = getattr(self, "critical_exception_events", 0)
        if critical and count < 32:
            self.critical_exception_events = count + 1
            try:
                self.journal.append("aggregation-exceptions.jsonl", {
                    "event": "candidate_aggregation_exception", "index": count + 1,
                    "type": kind.__name__,
                    "traceback": "".join(traceback.format_exception(kind, exc, tb)),
                })
            except BaseException:
                self.capture_error = True
        if self.exception_events >= MAX_EXCEPTION_EVENTS:
            self.exception_capture_truncated = True
            return self._trace_local
        self.exception_events += 1
        try:
            self.journal.append("exceptions.jsonl", {
                "event": "candidate_exception",
                "index": self.exception_events,
                "type": kind.__name__,
                "traceback": "".join(traceback.format_exception(kind, exc, tb)),
            })
        except BaseException:
            # Preserve the application exception. A later provider admission
            # stops if evidence writes failed; saturation alone is partial
            # caught-exception coverage, not a lost fatal traceback.
            self.capture_error = True
        return self._trace_local


def code_points():
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
    from hymem.dreaming import phase1

    root = OpenAICompatibleClient._complete_with_execution_lease.__code__
    attempts = [item for item in root.co_consts
                if isinstance(item, types.CodeType) and item.co_name == "_attempt"]
    if len(attempts) != 1:
        raise RuntimeError("provider_attempt_code_unavailable")
    return (OpenAICompatibleClient.complete.__code__, attempts[0],
            OpenAICompatibleEmbeddingClient._embed_with_locked_transport.__code__,
            phase1.extract_chunk_results.__code__,
            phase1.persist_chunk_results.__code__)


def verify_source(phase1_sha256: str = PHASE1_SHA256) -> None:
    if sha(SOURCE) != SOURCE_SHA256:
        raise ValueError("reference_snapshot_pin_mismatch")
    if sha(Path("/candidate/hymem/dreaming/phase1.py")) != phase1_sha256:
        raise ValueError("candidate_phase1_pin_mismatch")


def offline(env_path: Path, phase1_sha256: str = PHASE1_SHA256) -> dict:
    global STAGE
    STAGE = "offline_preflight"
    verify_source(phase1_sha256)
    private_dir(WORK / "offline")
    clone = WORK / "offline/hymem.sqlite"
    clone_source(clone)
    from hymem.core.db import connect, _load_vec_extension
    conn = connect(clone)
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%USING vec0%'").fetchone():
            if not _load_vec_extension(conn):
                raise RuntimeError("snapshot_vector_extension_unavailable")
        session = target_session(conn)
        current = conn.execute(
            "SELECT COUNT(*) FROM current_phase1_publications "
            "WHERE chunk_id=? AND phase1_generation_key=?", (TARGET_CHUNK, GENERATION)
        ).fetchone()[0]
        if current:
            raise ValueError("target_already_published")
    finally:
        conn.close()
    load_env(env_path, WORK / "offline")
    from hymem.bootstrap import build_from_env, shutdown_instance
    instance = build_from_env()
    try:
        if instance._phase1_generation["generation_key"] != GENERATION:
            raise ValueError("runtime_generation_mismatch")
    finally:
        if not shutdown_instance(instance):
            raise RuntimeError("offline_cleanup_failed")
    code_points()
    return {"status": "ready", "source_sha256": SOURCE_SHA256,
            "phase1_sha256": phase1_sha256, "target_chunk_id": TARGET_CHUNK,
            "session_sha256": hashlib.sha256(session.encode()).hexdigest(),
            "generation_key": GENERATION, "runtime_generation_verified": True,
            "source_unchanged": sha(SOURCE) == SOURCE_SHA256,
            "cleanup_ok": True, "failure_captured": False,
            "current_publications": current, "completion_calls": 0,
            "http_attempts": 0}


def live(env_path: Path, phase1_sha256: str = PHASE1_SHA256, *,
         max_http_attempts: int = MAX_HTTP_ATTEMPTS,
         max_llm_attempts: int = MAX_LLM_HTTP_ATTEMPTS,
         max_embedding_attempts: int = MAX_EMBEDDING_HTTP_ATTEMPTS,
         deadline_seconds: int = DEFAULT_DEADLINE_SECONDS) -> dict:
    global STAGE
    STAGE = "live_setup"
    verify_source(phase1_sha256)
    private_dir(WORK / "live")
    clone = WORK / "live/hymem.sqlite"
    clone_source(clone)
    load_env(env_path, WORK / "live")
    from hymem.bootstrap import build_from_env, shutdown_instance
    from hymem.deadline import MonotonicDeadline
    instance = None
    journal = None
    probe = None
    primary = None
    result: dict = {"status": "error", "stage": STAGE,
                    "source_sha256": SOURCE_SHA256,
                    "phase1_sha256": phase1_sha256,
                    "target_chunk_id": TARGET_CHUNK,
                    "generation_key": GENERATION,
                    "runtime_generation_verified": False,
                    "failure_captured": False}
    try:
        journal = PrivateJournal(WORK / "live")
        instance = build_from_env()
        if instance._phase1_generation["generation_key"] != GENERATION:
            raise ValueError("runtime_generation_mismatch")
        result["runtime_generation_verified"] = True
        session = target_session(instance.conn)
        result["session_sha256"] = hashlib.sha256(session.encode()).hexdigest()
        completion_code, attempt_code, embedding_code, extraction_code, persist_code = code_points()
        probe = Probe(journal, clone, completion_code=completion_code,
                      attempt_code=attempt_code, embedding_code=embedding_code,
                      extraction_code=extraction_code, persist_code=persist_code,
                      max_attempts=max_http_attempts,
                      max_llm_attempts=max_llm_attempts,
                      max_embedding_attempts=max_embedding_attempts)
        STAGE = "targeted_dream"
        sys.settrace(probe.trace)
        threading.settrace(probe.trace)
        sys.setprofile(probe.profile)
        threading.setprofile(probe.profile)
        report = instance.dream(session_ids=[session],
                                deadline=MonotonicDeadline.after(deadline_seconds))
        result["status"] = "completed"
        result["report"] = {key: value for key, value in asdict(report).items()
                            if value is None or type(value) in (int, float, bool)}
        result["chunks_processed"] = report.chunks_processed
    except BaseException as exc:
        primary = exc
        try:
            private_json(WORK / "live/failure.json", {
                "type": type(exc).__name__,
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            })
            result["failure_captured"] = True
        except BaseException:
            pass
        result["status"] = ("error" if isinstance(exc, InstrumentationStop)
                            else "budget_stopped" if isinstance(exc, DeadlineExceeded)
                            else "captured_failure" if result["failure_captured"]
                            else "error")
        if isinstance(exc, InstrumentationStop):
            result["error_code"] = "instrumentation_stopped"
        elif isinstance(exc, DeadlineExceeded) and probe is not None and probe.budget_reason is None:
            result["budget_reason"] = "cooperative_deadline"
        result["error_type"] = safe_type(exc)
        result["candidate_frames"] = safe_frames(exc)
        result["stage"] = STAGE
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        sys.settrace(None)
        threading.settrace(None)
        if probe is not None:
            result["completion_calls"] = probe.completions
            result["http_attempts"] = probe.attempts
            result["llm_http_attempts"] = probe.llm_attempts
            result["embedding_http_attempts"] = probe.embedding_attempts
            result["extractions_captured"] = probe.extractions
            result["prepersist_captured"] = probe.prepersist
            result["exception_events_captured"] = probe.exception_events
            result["exception_controlflow_skipped"] = probe.exception_controlflow_skipped
            result["exception_capture_truncated"] = probe.exception_capture_truncated
            result["instrumentation_capture_ok"] = not probe.capture_error
            if probe.budget_reason is not None:
                result["budget_reason"] = probe.budget_reason
        cleanup_ok = False
        llm_reported = None
        embedding_reported = None
        if instance is not None:
            for label, owner, field in (
                ("llm", instance._llm, "request_attempts"),
                ("embedding", instance._embed, "request_attempts"),
            ):
                try:
                    value = getattr(owner, field, None)
                    if type(value) is int and value >= 0:
                        result[label + "_provider_attempts_reported"] = value
                        if label == "llm":
                            llm_reported = value
                        else:
                            embedding_reported = value
                except BaseException:
                    pass
            try:
                available = getattr(instance._llm, "token_usage_available", None)
                result["token_usage_available"] = available is True
            except BaseException:
                result["token_usage_available"] = False
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                try:
                    value = getattr(instance._llm, field, None)
                    if result["token_usage_available"] and type(value) is int and value >= 0:
                        result[field] = value
                except BaseException:
                    pass
            result["accounting_verified"] = bool(
                probe is not None and llm_reported is not None
                and embedding_reported is not None
                and llm_reported <= probe.llm_attempts
                and embedding_reported <= probe.embedding_attempts
                and probe.attempts == probe.llm_attempts + probe.embedding_attempts
                and probe.attempts <= max_http_attempts
                and probe.llm_attempts <= max_llm_attempts
                and probe.embedding_attempts <= max_embedding_attempts
                and probe.completions <= MAX_COMPLETIONS
            )
            try:
                cleanup_ok = shutdown_instance(instance)
            except BaseException:
                cleanup_ok = False
        if journal is not None:
            try:
                journal.close()
            except BaseException:
                cleanup_ok = False
        result["cleanup_ok"] = cleanup_ok
        try:
            result["source_unchanged"] = sha(SOURCE) == SOURCE_SHA256
            if journal is not None:
                result["capture_sha256"] = artifact_digest(WORK / "live")
        except BaseException:
            result["source_unchanged"] = False
            result["status"] = "error"
            result["error_code"] = "final_evidence_digest_failed"
        if not cleanup_ok:
            result["status"] = "error"
            result["error_code"] = "cleanup_failed"
        if result.get("accounting_verified") is not True:
            result["status"] = "error"
            result["error_code"] = "provider_accounting_unverified"
        if probe is not None and probe.capture_error:
            result["status"] = "error"
            result["error_code"] = "instrumentation_capture_incomplete"
        if primary is not None and not result["failure_captured"]:
            result["status"] = "error"
            result["error_code"] = "private_failure_capture_incomplete"
    return result


def main() -> int:
    global STAGE
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("offline", "live"))
    parser.add_argument("--env", type=Path, default=Path("/run/runtime-env.json"))
    parser.add_argument("--phase1-sha256", default=PHASE1_SHA256)
    parser.add_argument("--max-http-attempts", type=int, default=MAX_HTTP_ATTEMPTS)
    parser.add_argument("--max-llm-http-attempts", type=int,
                        default=MAX_LLM_HTTP_ATTEMPTS)
    parser.add_argument("--max-embedding-http-attempts", type=int,
                        default=MAX_EMBEDDING_HTTP_ATTEMPTS)
    parser.add_argument("--deadline-seconds", type=int,
                        default=DEFAULT_DEADLINE_SECONDS)
    args = parser.parse_args()
    if (re.fullmatch(r"[0-9a-f]{64}", args.phase1_sha256) is None
            or not 1 <= args.max_http_attempts <= 896
            or not 1 <= args.max_llm_http_attempts <= 384
            or not 1 <= args.max_embedding_http_attempts <= 512
            or args.max_http_attempts > args.max_llm_http_attempts + args.max_embedding_http_attempts
            or args.deadline_seconds != 2700):
        raise ValueError("diagnostic_limits_invalid")
    STAGE = args.mode
    if not WORK.is_dir() or WORK.is_symlink() or stat.S_IMODE(WORK.stat().st_mode) != 0o700:
        raise ValueError("private_work_directory_invalid")
    result = (offline(args.env, args.phase1_sha256) if args.mode == "offline"
              else live(args.env, args.phase1_sha256,
                        max_http_attempts=args.max_http_attempts,
                        max_llm_attempts=args.max_llm_http_attempts,
                        max_embedding_attempts=args.max_embedding_http_attempts,
                        deadline_seconds=args.deadline_seconds))
    if sha(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("reference_snapshot_changed")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in ("ready", "completed", "captured_failure", "budget_stopped") else 1


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink():
                private_json(WORK / "setup-failure.json", {
                    "type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                })
                captured = True
        except BaseException:
            pass
        print(json.dumps({"status": "error", "error_type": safe_type(exc),
                          "stage": STAGE, "candidate_frames": safe_frames(exc),
                          "failure_captured": captured},
                         sort_keys=True))
        rc = 1
    raise SystemExit(rc)
