"""Strict, restart-safe primitives shared by HyMem's scored benchmarks.

The benchmark adapters are intentionally lightweight scripts, but their result
files are evidence.  This module owns the parts that must not drift between
LongMemEval, BEAM, MSC and LoCoMo:

* deterministic dev/holdout assignment and frozen calibration receipts;
* immutable, content-addressed run manifests;
* atomic, per-item checkpoints whose run identity is verified on resume; and
* strict result reconciliation.  Missing predictions and failed attempts stay
  in the denominator as wrong; duplicate/unknown ids and malformed rows are
  structural errors, never rows an adapter may silently skip.

Only stdlib modules are used so the deterministic CI smoke path does not need
benchmark datasets, model SDKs, or network access.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import argparse
import math
import sys
import tempfile
import threading
import re
import time
try:
    import fcntl
except ImportError:  # pragma: no cover - benchmark runners are POSIX today
    fcntl = None
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from hymem.deadline import DeadlineExceeded, MonotonicDeadline
from hymem.dreaming.status import (
    DREAM_STATUS_SCHEMA_VERSION,
    DREAM_STATUS_PHASE1_AUTHORITY_FIELDS,
    DURABLE_MALFORMED_FIELDS,
    DURABLE_PENDING_FIELDS,
)


STRICT_PROTOCOL_VERSION = "hymem-benchmark-strict-v1"
CHECKPOINT_VERSION = "hymem-benchmark-checkpoint-v1"
CALIBRATION_VERSION = "hymem-benchmark-calibration-v1"
CODE_IDENTITY_VERSION = "hymem-benchmark-code-v2"
BENCHMARK_INDEXING_STATUS_VERSION = "hymem-benchmark-indexing-status-v3"
_EMBEDDING_PENDING_FIELDS = (
    "pending_chunk_embeddings",
    "pending_message_embeddings",
    "pending_edge_embeddings",
    "pending_episode_embeddings",
    "pending_fact_embeddings",
)
_DURABLE_QUARANTINE_FIELDS = (
    "quarantined_chunks",
    "quarantined_digests",
    "quarantined_profiles",
    "quarantined_facts",
    "quarantined_facts_malformed",
)


def _phase1_authority_status_reason(status: Mapping[str, Any]) -> str | None:
    """Validate the current status schema's Phase-1 producer authority."""

    if not set(DREAM_STATUS_PHASE1_AUTHORITY_FIELDS).issubset(status):
        return "malformed_status_shape"
    backlog_status = status.get("phase1_backlog_status")
    authoritative = status.get("pending_chunks_authoritative")
    generation_key = status.get("phase1_generation_key")
    if not isinstance(authoritative, bool) or backlog_status not in {
        "current_producer", "producer_unavailable",
    }:
        return "malformed_status_shape"
    if authoritative:
        if (
            backlog_status != "current_producer"
            or not isinstance(generation_key, str)
            or not generation_key
        ):
            return "malformed_status_shape"
        return None
    if backlog_status != "producer_unavailable" or generation_key is not None:
        return "malformed_status_shape"
    return "phase1_producer_unavailable"
_CURRENT_DREAM_REPORT_FAILURE_FIELDS = (
    "chunk_extraction_failures",
    "coverage_integrity_failures",
    "digest_failures",
    "digest_quarantined",
    "profile_failures",
    "fact_failures",
    "aggregation_fusion_failures",
    "aggregation_build_exceptions",
)
_CURRENT_DREAM_REPORT_BOOLEAN_FIELDS = (
    "budget_exhausted",
    "extraction_provider_attempt_budget_exhausted",
    "skipped_locked",
)
# Directory expansion is deliberately limited to executable Python and SQL.
# Data/config inputs are still hashable when a caller names the exact file;
# incidental prose in a package directory is never benchmark code.
_CODE_DIRECTORY_SUFFIXES = {".py", ".sql"}
_CHECKPOINT_COMMON_ROOT_FIELDS = (
    "schema",
    "run_id",
    "manifest",
    "expected_ids",
    "scored",
    "verdict_key",
    "entries",
    "execution_segments",
    "status",
)
_CHECKPOINT_FINAL_ROOT_FIELDS = ("counts", "failure_ids")
_CHECKPOINT_COUNT_FIELDS = (
    "expected",
    "attempted",
    "unique_attempted",
    "total_attempts",
    "completed",
    "failed",
    "missing",
)


class BenchmarkIntegrityError(ValueError):
    """The benchmark evidence is incomplete, ambiguous, or inconsistent."""


class BenchmarkCleanupError(BenchmarkIntegrityError):
    """Caller-owned benchmark resources did not close cleanly."""


class IndexingConvergenceError(BenchmarkIntegrityError):
    """A benchmark memory build did not reach a complete, healthy state."""

    def __init__(self, message: str, summary: Mapping[str, Any]):
        super().__init__(message)
        self.summary = dict(summary)


def is_structural_benchmark_error(exc: BaseException) -> bool:
    """Whether an item boundary must abort instead of emitting a score row.

    Benchmark integrity and cleanup failures describe the validity of the run,
    not the quality of one model answer.  Indexing convergence is the deliberate
    exception: strict adapters retain it as an explicit, score-zero indexing row
    with its bounded health evidence.
    """

    return isinstance(exc, BenchmarkIntegrityError) and not isinstance(
        exc, IndexingConvergenceError
    )


_SAFE_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}")
_SAFE_DIAGNOSTIC_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,127}")
_SAFE_FAILURE_CODES = frozenset({
    "branch_incomplete",
    "call_failure",
    "clean_empty",
    "contract_failure",
    "coverage_integrity_failure",
    "corrupt_store_build_receipt",
    "execution_did_not_produce_a_row",
    "incomplete_response",
    "input_contract_failure",
    "internal_validation_failure",
    "incompatible_store_build_receipt_version",
    "item_validation_failure",
    "judge_or_reader_returned_no_valid_verdict",
    "malformed_store_build_receipt",
    "materialization_failure",
    "malformed_aggregation_failure_report",
    "malformed_cycle_failure_report",
    "malformed_coverage_integrity_state",
    "malformed_durable_state",
    "malformed_pending_backlog",
    "malformed_quarantine_state",
    "malformed_status_shape",
    "malformed_terminal_loss_state",
    "max_cycles_exhausted",
    "missing_prediction",
    "missing_store_build_receipt",
    "no_digest_input",
    "no_valid_prediction_verdict",
    "output_limit_exceeded",
    "oversized_store_build_receipt",
    "parse_failure",
    "phase1_producer_unavailable",
    "quarantined_extraction",
    "reader_transport_or_content_failure",
    "reader_transport_or_empty_response",
    "resource_limit",
    "response_conflict",
    "shape_failure",
    "source_coverage_failure",
    "source_stream_invalid",
    "skipped_indexing_reused_store",
    "store_build_identity_unavailable",
    "store_build_identity_mismatch",
    "store_build_receipt_changed_during_reuse",
    "store_build_receipt_publication_failed",
    "store_embedding_attestation_failed",
    "store_embedding_state_mismatch",
    "store_material_attestation_failed",
    "store_material_state_mismatch",
    "supported_claim_evidence_missing",
    "supported_claim_missing",
    "terminal_extraction_source_loss",
    "timeout",
    "timeout_after_cycle",
    "timeout_before_cycle",
    "timeout_during_cycle",
    "unexpected_canary_output",
    "unspecified_failure",
})
_EXCEPTION_FAILURE_CODES = frozenset({
    "answer_containment",
    "conversation_failure",
    "cycle_exception",
    "embedding_usage",
    "execution_failure",
    "memory_pipeline_usage",
    "probe_failure",
    "reader_usage",
    "recall_gold_turns",
    "retrieval_usage",
    "worker_failure",
})
_INDEXING_FAILURE_CODES = frozenset({
    code for code in _SAFE_FAILURE_CODES
    if code in {
        "coverage_integrity_failure",
        "malformed_aggregation_failure_report",
        "malformed_cycle_failure_report",
        "malformed_coverage_integrity_state",
        "malformed_durable_state",
        "malformed_pending_backlog",
        "malformed_quarantine_state",
        "malformed_status_shape",
        "malformed_terminal_loss_state",
        "max_cycles_exhausted",
        "quarantined_extraction",
        "terminal_extraction_source_loss",
        "timeout_after_cycle",
        "timeout_before_cycle",
        "timeout_during_cycle",
    }
} | {"cycle_exception"})
_CLEANUP_ACTIONS = frozenset({
    "adapter_close",
    "checkpoint_close",
    "dream_fork_close",
    "embedding_usage_snapshot",
    "execution_segment_snapshot",
    "runtime_usage_handoff",
    "gc_collect",
    "indexing_summary_snapshot",
    "pipeline_usage_snapshot",
    "query_cache_invalidation",
    "resource_close",
    "temporary_store_cleanup",
})


def bounded_exception_type(exc: BaseException) -> str:
    """Return a bounded class identity safe for durable benchmark evidence."""

    name = type(exc).__name__
    return name if _SAFE_EXCEPTION_TYPE.fullmatch(name) else "Exception"


def bounded_failure_text(value: object) -> str:
    """Project operational failure detail onto a small, source-free schema.

    Durable benchmark failures are machine evidence, not logs.  This accepts
    the codes emitted by the strict adapters, optionally followed by one
    bounded exception class, and deliberately discards every free-form tail.
    Question/answer/evidence fields are handled separately and remain exact.
    """

    text = str(value or "").strip()
    if ";" in text:
        parts = [part.strip() for part in text.split(";")]
        if 1 < len(parts) <= 4 and all(parts):
            normalized = [bounded_failure_text(part) for part in parts]
            if all(part != "unspecified_failure" for part in normalized):
                return ";".join(normalized)
    if text in _SAFE_FAILURE_CODES:
        return text
    if re.fullmatch(
        r"judge_(?:transport_or_parse_failure|parse_failure|missing_rubric|"
        r"malformed|unreadable|criterion_[0-9]{1,6}_"
        r"(?:transport|unreadable|invalid_score|invalid_reason))",
        text,
    ):
        return text

    # ``indexing_failure`` carries another closed reason code, never an
    # exception message.  A cycle exception's class lives in the versioned
    # indexing summary instead of being smuggled into this row-level code.
    indexing = re.fullmatch(r"indexing_failure:([a-z][a-z0-9_]{0,127})", text)
    if indexing is not None and indexing.group(1) in _INDEXING_FAILURE_CODES:
        return text

    # Read both the new compact form and legacy ``code: Type: raw message``
    # while emitting only ``code:Type``.  This makes checkpoint recovery safe
    # without retaining a credential/path-bearing historical message.
    exception = re.match(
        r"^([a-z][a-z0-9_]{0,127})\s*:\s*"
        r"([A-Za-z_][A-Za-z0-9_.]{0,127})(?:\s*:.*)?$",
        text,
        flags=re.DOTALL,
    )
    if exception is not None and exception.group(1) in _EXCEPTION_FAILURE_CODES:
        return f"{exception.group(1)}:{exception.group(2)}"

    # Legacy callers passed prose such as ``transport failed`` to the generic
    # checkpoint API.  Keeping that prose would make the checkpoint/history a
    # second exception log, so retain only the fact that the attempt failed.
    return "unspecified_failure"


def _cleanup_evidence(action: str, exc: BaseException) -> dict[str, str]:
    if action not in _CLEANUP_ACTIONS:
        raise ValueError("unknown benchmark cleanup action")
    return {
        "stage": action,
        "exception_type": bounded_exception_type(exc),
    }


def _attach_cleanup_note(
    primary: BaseException, failures: Sequence[Mapping[str, str]],
) -> None:
    """Attach bounded evidence without changing the primary exception identity."""

    detail = json.dumps(list(failures), sort_keys=True, separators=(",", ":"))
    try:
        primary.add_note(f"benchmark cleanup failures: {detail}")
    except (AttributeError, TypeError):  # pragma: no cover - pre-3.11 fallback
        pass


def run_cleanup_actions(
    actions: Sequence[tuple[str, Callable[[], object]]],
    *,
    primary_exception: BaseException | None = None,
    evidence_sink: list[dict[str, str]] | None = None,
) -> tuple[dict[str, str], ...]:
    """Attempt independent cleanup actions and preserve failure precedence.

    Every action is attempted, even when an earlier action fails.  Evidence is
    deliberately limited to a closed action enum and a bounded exception class;
    exception messages, paths, endpoints and credentials never enter artifacts.

    An already-active primary exception always remains authoritative.  With no
    primary, an ordinary cleanup failure is a benchmark-integrity failure, while
    a control-flow ``BaseException`` (including ``DeadlineExceeded``) is re-raised
    unchanged after the remaining cleanup actions have been attempted.
    """

    failures: list[dict[str, str]] = []
    failure_exceptions: list[BaseException] = []
    for action, cleanup in actions:
        if action not in _CLEANUP_ACTIONS:
            raise ValueError("unknown benchmark cleanup action")
        try:
            cleanup()
        except BaseException as exc:
            failures.append(_cleanup_evidence(action, exc))
            failure_exceptions.append(exc)

    if evidence_sink is not None:
        evidence_sink.extend(dict(item) for item in failures)
    if not failures:
        return ()

    if primary_exception is not None:
        _attach_cleanup_note(primary_exception, failures)
        print(
            "WARNING: benchmark cleanup failures: "
            + json.dumps(failures, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )
        return tuple(failures)

    # ``Exception`` failures are ordinary cleanup faults.  Anything outside
    # that hierarchy is control flow and must not be converted or swallowed.
    for exc in failure_exceptions:
        if not isinstance(exc, Exception):
            _attach_cleanup_note(exc, failures)
            raise exc

    detail = ", ".join(
        f"{item['stage']}:{item['exception_type']}" for item in failures
    )
    error = BenchmarkCleanupError(f"benchmark cleanup failed ({detail})")
    error.cleanup_errors = tuple(dict(item) for item in failures)
    raise error


class OwnedResourceScope:
    """Close caller-owned resources once without masking primary failures.

    Benchmark provider clients are often shared by several logical roles and
    worker threads.  This scope records object identity (rather than role),
    closes in reverse construction order, and is itself idempotent.  Cleanup
    failures are fatal when the benchmark otherwise succeeded; when another
    exception is already in flight they are attached and printed while the
    original exception remains authoritative.

    Resources without a callable ``close`` are deliberately ignored.  This
    lets simulation stubs and injected non-owning counters pass through without
    pretending that the scope owns a transport they do not expose.
    """

    def __init__(self, label: str = "benchmark resources") -> None:
        self.label = str(label)
        self._resources: list[tuple[Callable[[], object], str]] = []
        self._identities: set[int] = set()
        self._closed = False

    def own(self, resource: object, *, label: str | None = None):
        """Register and return one explicitly caller-owned resource."""

        close = None if resource is None else getattr(resource, "close", None)
        if not callable(close):
            return resource
        if self._closed:
            raise RuntimeError(f"{self.label} scope is already closed")
        identity = id(resource)
        if identity not in self._identities:
            self._identities.add(identity)
            self._resources.append((
                close,
                label or type(resource).__name__,
            ))
        return resource

    def close(
        self, *, primary_exception: BaseException | None = None,
    ) -> tuple[dict[str, str], ...]:
        """Close all registered resources, preserving ``primary_exception``."""

        if self._closed:
            return ()
        self._closed = True
        actions = [
            ("resource_close", close)
            for close, _resource_label in reversed(self._resources)
        ]
        return run_cleanup_actions(
            actions, primary_exception=primary_exception,
        )

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> bool:
        self.close(primary_exception=exc)
        return False


def close_preserving_primary(
    resource: object,
    *,
    label: str,
    primary_exception: BaseException | None = None,
) -> tuple[dict[str, str], ...]:
    """Close one explicitly owned resource with scope cleanup semantics."""

    scope = OwnedResourceScope(label)
    scope.own(resource, label=type(resource).__name__)
    return scope.close(primary_exception=primary_exception)


_SECRET_KEY_PARTS = (
    "api_key", "apikey", "password", "passwd", "authorization", "auth_token",
    "access_token", "refresh_token", "secret", "credential", "cookie",
    "set_cookie", "assertion",
)
_SECRET_QUERY_KEYS = frozenset({
    "api_key", "apikey", "key", "token", "access_token", "auth",
    "authorization", "password", "passwd", "secret", "signature", "sig",
    "client_assertion", "assertion", "id_token", "saml_response",
})
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_FILE_PATH_URI_IN_TEXT_RE = re.compile(
    r"(?i)\b(?:file|sqlite):(?://)?/[^\s'\"<>]+"
)
_SCHEME_URI_IN_TEXT_RE = re.compile(
    r"(?i)\b([A-Za-z][A-Za-z0-9+.-]*)://[^\s'\"<>]+"
)
_ABSOLUTE_PATH_IN_TEXT_RE = re.compile(
    # Drive-qualified paths may use either separator even when the benchmark
    # runner itself is POSIX, so ``Path.is_absolute`` is insufficient here.
    r"(?<![A-Za-z0-9_.:/\\-])[A-Za-z]:[\\/][^\s'\"<>]+"
    # Likewise accept both slash forms of UNC paths.  The negative lookbehind
    # prevents the ``//host/path`` portion of an HTTP(S) URL from matching.
    r"|(?<![A-Za-z0-9_.:/\\-])(?:\\\\|//)"
    r"[^\\/\s'\"<>]+[\\/][^\\/\s'\"<>]+"
    r"(?:[\\/][^\s'\"<>]+)*"
    r"|(?<![A-Za-z0-9_.:/\\-])(?:/[A-Za-z0-9_.~%+-]+)+"
)
_SAFE_FAILURE_DETAIL_TEXT = re.compile(
    r"^[a-z0-9_.\[\]-]+:[a-z0-9_]+$"
)
_EVIDENCE_TEXT_KEYS = frozenset({
    "answer", "content", "context", "gold", "gold_text", "hypothesis",
    "ideal_answer", "ideal_response", "prediction", "question", "response",
    "rubric", "summary", "text",
})
_OPERATIONAL_FAILURE_TEXT_KEYS = frozenset({
    "benchmark_failure",
    "error",
    "failure",
    "failure_reason",
    "instrumentation_errors",
    "probe_error",
    "recall_diagnostic_error",
})


def _is_secret_key(value: str) -> bool:
    normalized = value.casefold().replace("-", "_")
    if normalized in {"credentials_redacted", "path_redacted"}:
        # These are fixed boolean sanitizer markers, never caller-supplied
        # credential material.  Treating the marker itself as a secret makes
        # sanitization non-idempotent and breaks checkpoint manifest recovery.
        return False
    return (
        normalized in _SECRET_QUERY_KEYS
        or any(part in normalized for part in _SECRET_KEY_PARTS)
        or normalized in {
            "x_amz_signature", "x_amz_credential", "x_amz_security_token",
        }
    )


def _sanitized_url(value: str) -> str | dict[str, Any]:
    """Return a credential-free URL with no credential-derived fingerprint."""

    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.netloc:
            return value
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            # An invalid port is not a usable endpoint; retain no potentially
            # credential-bearing bytes in an evidence artifact.
            return {
                "url": "<redacted-invalid-url>",
                "credentials_redacted": True,
            }
        netloc = host + (f":{port}" if port is not None else "")
        # Endpoint query values are impossible to classify exhaustively: a
        # secret may sit under an innocuous key such as ``opaque``.  Active
        # provider endpoints reject queries entirely; incidental URLs in
        # artifacts therefore drop the complete query rather than attempting
        # a credential-name allow/deny list.
        query_redacted = bool(parsed.query)
        has_userinfo = parsed.username is not None or parsed.password is not None
        fragment_redacted = bool(parsed.fragment)
        if not has_userinfo and not query_redacted and not fragment_redacted:
            return value
        clean = urlunsplit((
            parsed.scheme,
            netloc,
            parsed.path,
            "",
            "",
        ))
        result: dict[str, Any] = {
            "url": clean,
            "credentials_redacted": True,
        }
        return result
    except (TypeError, ValueError):
        # A malformed absolute HTTP URL is still credential-shaped input.  Do
        # not preserve it just because URL parsing failed before userinfo or a
        # query could be separated safely.
        if isinstance(value, str) and re.match(r"(?i)^https?://", value):
            return {
                "url": "<redacted-invalid-url>",
                "credentials_redacted": True,
            }
        return value


def _sanitize_failure_text(value: object) -> str:
    """Return only a bounded operational code/type, never exception prose."""

    return bounded_failure_text(value)


def _scrub_sensitive_text(value: object) -> str:
    """Redact credential syntax from ordinary non-evidence lifecycle text."""

    text = str(value)

    def replace_url(match: re.Match[str]) -> str:
        sanitized = _sanitized_url(match.group(0))
        return sanitized if isinstance(sanitized, str) else str(sanitized["url"])

    text = _URL_IN_TEXT_RE.sub(replace_url, text)
    text = re.sub(
        r"(?im)\b(proxy-authorization|authorization|set-cookie|cookie)"
        r"\s*:\s*[^\r\n]*",
        lambda match: f"{match.group(1)}: <redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\b(proxy-authorization|authorization)\s*=\s*"
        r"[^\s,;]+(?:\s+[^\s,;]+)?",
        lambda match: f"{match.group(1)}=<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\b(bearer)\s*(?::|=)?\s*[^\s,;]+",
        r"\1 <redacted>", text,
    )
    text = re.sub(
        r"(?i)(\b(?:api[_-]?key|token|password|passwd|authorization|"
        r"cookie|set-cookie|secret|credential)\b\s*[:=]\s*)[^\s,;]+",
        r"\1<redacted>", text,
    )
    return text


def _sanitize_incidental_url_text(value: str) -> str | dict[str, Any]:
    """Scrub credential URLs under unknown config keys, not evidence fields."""

    direct = _sanitized_url(value)
    if direct != value:
        return direct

    def replace(match: re.Match[str]) -> str:
        sanitized = _sanitized_url(match.group(0))
        return sanitized if isinstance(sanitized, str) else str(sanitized["url"])

    return _URL_IN_TEXT_RE.sub(replace, value)


def _sanitize_incidental_path_text(value: str) -> str | dict[str, Any]:
    """Remove host filesystem locations from non-evidence lifecycle text."""

    if re.match(r"^(?:file|sqlite):", value, flags=re.IGNORECASE):
        return {"path_redacted": True}
    uri = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*)://", value)
    if uri is not None:
        scheme = uri.group(1).casefold()
        if scheme in {"http", "https"} or value == "local://feature-hash":
            return value
        # Unknown URI schemes are not part of current benchmark identity and
        # may be wrappers around host paths. Fail closed on artifact output.
        return {"path_redacted": True}
    if Path(value).is_absolute():
        return {"path_redacted": True}

    def replace_uri(match: re.Match[str]) -> str:
        candidate = match.group(0)
        scheme = match.group(1).casefold()
        if scheme in {"http", "https"} or candidate == "local://feature-hash":
            return candidate
        return "<redacted-path-uri>"

    # Unknown URI schemes can wrap host paths just as file/sqlite URIs can.
    # HTTP(S) has already had credentials/query material stripped above, and
    # the one closed local identity token is deliberately stable.
    scrubbed = _SCHEME_URI_IN_TEXT_RE.sub(replace_uri, value)
    scrubbed = _FILE_PATH_URI_IN_TEXT_RE.sub("<redacted-path-uri>", scrubbed)
    scrubbed = _ABSOLUTE_PATH_IN_TEXT_RE.sub("<redacted-path>", scrubbed)
    return scrubbed


def sanitize_for_artifact(
    value: Any,
    *,
    key_hint: str = "",
    _preserve_evidence_text: bool = True,
    _bound_unknown_text: bool = False,
    _execution_segment_scope: bool = False,
) -> Any:
    """Remove credentials recursively while preserving score-relevant identity.

    Secret values are replaced by one opaque marker. Credentials deliberately
    do not participate in run identity: hashing low-entropy passwords or tokens
    would permit offline guesses. URL userinfo is handled even when its parent
    key has an innocuous name such as ``base_url``.
    """

    key_folded = key_hint.casefold().replace("-", "_")
    if key_hint and _is_secret_key(key_hint):
        return {"redacted": True}
    if isinstance(value, Mapping):
        sanitized_mapping: dict[str, Any] = {}
        for key, item in value.items():
            child_key = str(key)
            child_folded = child_key.casefold().replace("-", "_")
            segment_scope = (
                _execution_segment_scope
                or child_folded == "execution_segments"
                or (key_folded == "execution" and child_folded == "segments")
            )
            sanitized_mapping[child_key] = sanitize_for_artifact(
                item,
                key_hint=child_key,
                _preserve_evidence_text=(
                    False if segment_scope else _preserve_evidence_text
                ),
                _bound_unknown_text=(
                    True if segment_scope else _bound_unknown_text
                ),
                _execution_segment_scope=segment_scope,
            )
        return sanitized_mapping
    if isinstance(value, (list, tuple)):
        return [
            sanitize_for_artifact(
                item,
                key_hint=key_hint,
                _preserve_evidence_text=_preserve_evidence_text,
                _bound_unknown_text=_bound_unknown_text,
                _execution_segment_scope=_execution_segment_scope,
            )
            for item in value
        ]
    if isinstance(value, str):
        if key_folded == "exception_type":
            return value if _SAFE_EXCEPTION_TYPE.fullmatch(value) else "Exception"
        if key_folded in {"stage", "code"}:
            return value if _SAFE_DIAGNOSTIC_TOKEN.fullmatch(value) else "unknown"
        if key_folded == "failure_details":
            return (
                value if len(value) <= 256
                and _SAFE_FAILURE_DETAIL_TEXT.fullmatch(value)
                else "diagnostic:invalid"
            )
        if key_folded in _OPERATIONAL_FAILURE_TEXT_KEYS or key_folded.endswith(
            ("_error", "_errors", "_exception", "_exceptions")
        ):
            return _sanitize_failure_text(value)
        if _preserve_evidence_text and key_folded in _EVIDENCE_TEXT_KEYS:
            return value
        if _bound_unknown_text and len(value) > 4096:
            # Execution telemetry has no score-bearing prose.  A giant string
            # under an unanticipated key is therefore neither useful identity
            # nor safe diagnostics; importantly, it cannot become a durable
            # copy of an exception/provider response just because its producer
            # called the field ``summary`` or ``detail``.
            return "<redacted-oversized-text>"
        # Configuration and lifecycle strings can carry credential-shaped
        # headers even under an innocuous key (for example {"detail":
        # "Authorization: Bearer ..."}). Evidence-bearing text keys above
        # remain byte-for-byte untouched for rejudging.
        # Run all three transforms.  Returning after the first credential
        # match used to leave an absolute path later in the same diagnostic.
        # Scrubbing URLs first also preserves the historical string type of
        # endpoint identity fields required by strict canary/registry schemas.
        scrubbed = _scrub_sensitive_text(value)
        sanitized = _sanitize_incidental_url_text(scrubbed)
        if not isinstance(sanitized, str):
            return sanitized
        sanitized_path = _sanitize_incidental_path_text(sanitized)
        return sanitized_path
    return value


def add_strict_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the identical protocol/checkpoint controls to a benchmark CLI."""

    parser.add_argument(
        "--protocol-split",
        choices=("full", "dev", "holdout"),
        default="full",
        help=(
            "full is development/test-contaminated evidence; dev/holdout use "
            "a frozen internal-split calibration receipt"
        ),
    )
    parser.add_argument(
        "--calibration-receipt",
        default=None,
        help="frozen receipt binding the dev/holdout ids and exact config/models",
    )
    parser.add_argument(
        "--freeze-calibration",
        default=None,
        metavar="FILE",
        help=(
            "freeze this exact config/model and deterministic internal split, "
            "then exit without running the benchmark"
        ),
    )
    parser.add_argument(
        "--dev-fraction", type=float, default=0.5,
        help="internal dev share used only with --freeze-calibration",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="new atomic per-row checkpoint path (existing files are refused)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="CHECKPOINT",
        help="resume only when the checkpoint's immutable run identity matches",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help=(
            "on resume, explicitly retry prior failed attempts; by default "
            "failed rows remain terminal wrong answers and incur no new spend"
        ),
    )


def resolve_checkpoint_path(
    *,
    checkpoint: str | None,
    resume_from: str | None,
    base_dir: str | os.PathLike[str],
    benchmark: str,
    run_id: str,
) -> tuple[Path, bool]:
    """Resolve a new timestamped checkpoint or an explicit resume target."""

    if checkpoint and resume_from:
        raise BenchmarkIntegrityError(
            "pass only one of --checkpoint or --resume-from"
        )
    if resume_from:
        return Path(resume_from), True
    if checkpoint:
        return Path(checkpoint), False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in benchmark)
    return (
        Path(base_dir) / "checkpoints"
        / f"{safe}-{stamp}-{run_id.removeprefix('sha256:')[:12]}.json",
        False,
    )


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BenchmarkIntegrityError(
            f"value is not canonical JSON: {exc}"
        ) from exc


def content_hash(value: object) -> str:
    """SHA-256 of canonical JSON, with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def dataclass_identity(value: object, *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """JSON-safe effective dataclass settings, excluding runtime-only fields."""

    if not is_dataclass(value) or isinstance(value, type):
        raise BenchmarkIntegrityError("config identity source must be a dataclass instance")
    excluded = set(exclude)

    def normalize(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            return {str(key): normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, (set, frozenset)):
            return sorted((normalize(val) for val in item), key=str)
        return item

    return {
        field.name: normalize(getattr(value, field.name))
        for field in fields(value) if field.name not in excluded
    }


def effective_hymem_config_identity(value: object) -> dict[str, Any]:
    """Score/material identity for a resolved ``HyMemConfig``.

    ``root`` is runtime placement. ``extraction_feedback_keep`` bounds a local
    retraction-audit table that neither enters prompts nor material-store
    attestation. Including either would split otherwise identical benchmark
    arms on operational state rather than memory behavior.
    """

    return dataclass_identity(
        value, exclude={"root", "extraction_feedback_keep"}
    )


def converge_indexing(
    dream,
    *,
    status=None,
    max_cycles: int,
    timeout_s: float,
    require_healthy: bool = True,
    _clock=None,
) -> dict[str, Any]:
    """Run bounded dream cycles until the durable extraction backlog is empty.

    ``dream`` returns a DreamReport-like dataclass or mapping. ``status`` is an
    optional read-only durable backlog callback. A single non-exhausted report
    is insufficient when durable pending work remains. Quarantined work,
    prompt-independent terminal source loss, and unestablished lossless
    coverage integrity, and an enabled aggregation build without a clean
    config-matched acknowledgement make
    canonical/healthy completion fail loudly rather than masquerading as a
    completed index. Coverage corruption is durable but can be healed by a
    later complete source walk, so cycle-local coverage failures consume the
    same bounded retry loop as every other transient report failure. After an
    operator resolves a cause or the next walk succeeds, the durable signal is
    cleared before completion.

    The bound is one absolute monotonic deadline, not a fresh timeout per
    cycle/attempt. Shipped network clients cap every request and retry sleep to
    its remaining time. This is necessarily cooperative: an injected provider
    that ignores its request timeout cannot safely be killed in-process; when
    it eventually returns, its result is rejected before semantic publication
    and the convergence call reports ``timeout_during_cycle``. Deadline-bound
    runs deliberately avoid the normal background embedding worker for the
    same reason. Rollback, lease release, and terminal run telemetry are the
    only writes permitted after expiry.
    """

    if isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles <= 0:
        raise BenchmarkIntegrityError("indexing max_cycles must be positive")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise BenchmarkIntegrityError("indexing timeout_s must be positive and finite")

    clock = time.monotonic if _clock is None else _clock
    if not callable(clock):
        raise BenchmarkIntegrityError("indexing monotonic clock must be callable")

    def normalize_report(report: object) -> dict[str, Any]:
        if isinstance(report, Mapping):
            return dict(report)
        if is_dataclass(report) and not isinstance(report, type):
            return dataclass_identity(report)
        raise BenchmarkIntegrityError("dream returned a malformed report")

    started = float(clock())
    if not math.isfinite(started):
        raise BenchmarkIntegrityError("indexing monotonic clock is malformed")
    deadline = MonotonicDeadline(
        started + float(timeout_s), clock=clock,
    )
    reports: list[dict[str, Any]] = []
    latest_status: dict[str, Any] = {}

    def call_dream():
        """Pass the one absolute deadline to capable dream callbacks.

        Signature inspection avoids a catch-and-retry-on-TypeError pattern,
        which could execute a side-effecting callback twice.  Legacy/custom
        callbacks with no deadline parameter remain source-compatible, but
        shipped benchmark adapters all route through ``HyMem.dream`` and thus
        receive the deadline.
        """

        try:
            signature = inspect.signature(dream)
        except (TypeError, ValueError):
            return dream()
        parameters = signature.parameters
        deadline_parameter = parameters.get("deadline")
        if (
            deadline_parameter is not None
            and deadline_parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        ):
            return dream(deadline)
        accepts_deadline = (
            (
                deadline_parameter is not None
                and deadline_parameter.kind in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            )
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        return dream(deadline=deadline) if accepts_deadline else dream()

    def summary(*, complete: bool, reason: str | None = None) -> dict[str, Any]:
        elapsed = max(0.0, float(clock()) - started)
        quarantined = {
            key: value for key, value in latest_status.items()
            if "quarantined" in key
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        }
        terminal_losses = {
            key: value for key, value in latest_status.items()
            if key.startswith("terminal_loss_")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        }
        coverage_integrity_failure = latest_status.get(
            "coverage_integrity_failures", 0
        )
        malformed = {
            key: value for key, value in latest_status.items()
            if key in DURABLE_MALFORMED_FIELDS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        }
        in_progress = latest_status.get("in_progress", False)
        return {
            "cycles": len(reports),
            "max_cycles": max_cycles,
            "timeout_s": float(timeout_s),
            "elapsed_s": elapsed,
            "complete": bool(complete),
            "healthy": bool(
                complete
                and not quarantined
                and not terminal_losses
                and not malformed
                and coverage_integrity_failure == 0
                and in_progress is False
            ),
            "failure_reason": reason,
            "reports": reports,
            "final_status": latest_status,
            "quarantined": quarantined,
        }

    for _cycle in range(max_cycles):
        if deadline.expired:
            current = summary(complete=False, reason="timeout_before_cycle")
            raise IndexingConvergenceError(
                "memory indexing did not converge before its timeout", current
            )
        dream_returned = False
        report_appended = False
        report_obj: object = None
        try:
            report_obj = call_dream()
            dream_returned = True
            # A legacy/custom callback may ignore the propagated deadline.
            # Check before normalizing its result or making the status call.
            deadline.check()
            report = normalize_report(report_obj)
            reports.append(report)
            report_appended = True
            if status is not None:
                deadline.check()
                status_obj = status()
                deadline.check()
                if not isinstance(status_obj, Mapping):
                    raise BenchmarkIntegrityError(
                        "indexing status callback returned a malformed value"
                    )
                latest_status = dict(status_obj)
        except DeadlineExceeded as exc:
            if dream_returned and not report_appended:
                # The callback completed after its bound (for example a custom
                # provider ignored its timeout).  Its DreamReport is safe,
                # bounded evidence, but no subsequent status callback runs.
                reports.append(normalize_report(report_obj))
            reason = (
                "timeout_after_cycle"
                if dream_returned else "timeout_during_cycle"
            )
            current = summary(
                complete=False, reason=reason,
            )
            raise IndexingConvergenceError(
                "memory indexing exceeded its deadline",
                current,
            ) from exc
        except Exception as exc:
            if deadline.expired:
                current = summary(
                    complete=False, reason="timeout_during_cycle",
                )
                raise IndexingConvergenceError(
                    "memory indexing exceeded its deadline during a cycle",
                    current,
                ) from exc
            current = summary(
                complete=False,
                reason=f"cycle_exception:{bounded_exception_type(exc)}",
            )
            raise IndexingConvergenceError(
                "memory indexing cycle failed", current
            ) from exc
        current_status = (
            latest_status.get("dream_status_schema")
            == DREAM_STATUS_SCHEMA_VERSION
        )
        if (
            "dream_status_schema" in latest_status
            and not current_status
        ):
            current = summary(complete=False, reason="malformed_status_shape")
            raise IndexingConvergenceError(
                "memory indexing status has an unsupported schema", current
            )
        benchmark_status = latest_status.get("benchmark_indexing_status_schema")
        if benchmark_status not in (None, BENCHMARK_INDEXING_STATUS_VERSION):
            current = summary(complete=False, reason="malformed_status_shape")
            raise IndexingConvergenceError(
                "memory indexing benchmark status has an unsupported schema",
                current,
            )
        if benchmark_status is not None and not current_status:
            current = summary(complete=False, reason="malformed_status_shape")
            raise IndexingConvergenceError(
                "benchmark indexing status lacks its durable dream schema",
                current,
            )
        if (
            benchmark_status is None
            and any(key in latest_status for key in _EMBEDDING_PENDING_FIELDS)
        ):
            current = summary(complete=False, reason="malformed_status_shape")
            raise IndexingConvergenceError(
                "embedding backlog lacks its benchmark status schema",
                current,
            )
        schema_only_fields = (
            set(DURABLE_PENDING_FIELDS) - {"pending_chunks", "pending_aggregation"}
        ) | set(DURABLE_MALFORMED_FIELDS)
        if not current_status and schema_only_fields.intersection(latest_status):
            current = summary(complete=False, reason="malformed_status_shape")
            raise IndexingConvergenceError(
                "unversioned indexing status carries current-only health fields",
                current,
            )

        if current_status:
            authority_reason = _phase1_authority_status_reason(latest_status)
            if authority_reason is not None:
                current = summary(complete=False, reason=authority_reason)
                raise IndexingConvergenceError(
                    "memory indexing lacks exact Phase-1 producer authority",
                    current,
                )
            required_pending = set(DURABLE_PENDING_FIELDS)
            if benchmark_status == BENCHMARK_INDEXING_STATUS_VERSION:
                required_pending.update(_EMBEDDING_PENDING_FIELDS)
            allowed_pending = set(DURABLE_PENDING_FIELDS) | set(
                _EMBEDDING_PENDING_FIELDS
            ) | {"pending_chunks_authoritative"}
            unknown_pending = {
                key for key in latest_status
                if isinstance(key, str)
                and key.startswith("pending_")
                and key not in allowed_pending
            }
            missing_pending = required_pending - set(latest_status)
            if unknown_pending or missing_pending:
                current = summary(
                    complete=False, reason="malformed_status_shape"
                )
                raise IndexingConvergenceError(
                    "memory indexing status has an incomplete pending schema",
                    current,
                )
            pending_values = {
                key: latest_status[key]
                for key in sorted(required_pending)
            }
        else:
            # Explicit schema-less compatibility contract: only the historical
            # pending_chunks field is authoritative. Canonical benchmark
            # receipts reject this shape; it is retained for custom callbacks.
            if status is not None and "pending_chunks" not in latest_status:
                current = summary(
                    complete=False, reason="malformed_status_shape"
                )
                raise IndexingConvergenceError(
                    "legacy memory indexing status is missing pending_chunks",
                    current,
                )
            pending_values = {
                key: value for key, value in latest_status.items()
                if isinstance(key, str) and key.startswith("pending_")
            }
        if any(
            isinstance(value, bool)
            or not isinstance(value, int if current_status else (int, float))
            or (
                not current_status
                and not math.isfinite(float(value))
            )
            or value < 0
            for value in pending_values.values()
        ):
            current = summary(complete=False, reason="malformed_pending_backlog")
            raise IndexingConvergenceError(
                "memory indexing status has malformed pending backlog", current
            )
        pending = sum(pending_values.values())
        if current_status:
            missing_quarantine = set(_DURABLE_QUARANTINE_FIELDS) - set(
                latest_status
            )
            unknown_quarantine = {
                key for key in latest_status
                if isinstance(key, str)
                and "quarantined" in key
                and key not in _DURABLE_QUARANTINE_FIELDS
            }
            if missing_quarantine or unknown_quarantine:
                current = summary(
                    complete=False, reason="malformed_status_shape"
                )
                raise IndexingConvergenceError(
                    "memory indexing status has an incomplete quarantine schema",
                    current,
                )
            quarantine_values = {
                key: latest_status[key] for key in _DURABLE_QUARANTINE_FIELDS
            }
        else:
            quarantine_values = {
                key: value for key, value in latest_status.items()
                if isinstance(key, str) and "quarantined" in key
            }
        if any(
            isinstance(value, bool)
            or not isinstance(value, int if current_status else (int, float))
            or (
                not current_status
                and not math.isfinite(float(value))
            )
            or value < 0
            for value in quarantine_values.values()
        ):
            current = summary(complete=False, reason="malformed_quarantine_state")
            raise IndexingConvergenceError(
                "memory indexing status has malformed quarantine state", current
            )
        terminal_loss_values = {
            "terminal_loss_chunks": latest_status["terminal_loss_chunks"]
        } if "terminal_loss_chunks" in latest_status else {}
        if current_status and not terminal_loss_values:
            current = summary(
                complete=False, reason="malformed_status_shape"
            )
            raise IndexingConvergenceError(
                "memory indexing status is missing terminal loss state", current
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in terminal_loss_values.values()
        ):
            current = summary(complete=False, reason="malformed_terminal_loss_state")
            raise IndexingConvergenceError(
                "memory indexing status has malformed terminal loss state", current
            )
        coverage_integrity_failure = latest_status.get(
            "coverage_integrity_failures", None if current_status else 0
        )
        if (
            isinstance(coverage_integrity_failure, bool)
            or not isinstance(coverage_integrity_failure, int)
            or coverage_integrity_failure < 0
        ):
            current = summary(
                complete=False, reason="malformed_coverage_integrity_state"
            )
            raise IndexingConvergenceError(
                "memory indexing status has malformed coverage integrity state",
                current,
            )
        if current_status:
            malformed_fields = set(DURABLE_MALFORMED_FIELDS) - set(latest_status)
            unknown_malformed = {
                key for key in latest_status
                if isinstance(key, str)
                and key.startswith("malformed_")
                and key not in DURABLE_MALFORMED_FIELDS
            }
            if malformed_fields or unknown_malformed:
                current = summary(
                    complete=False, reason="malformed_status_shape"
                )
                raise IndexingConvergenceError(
                    "memory indexing status has an incomplete malformed-state schema",
                    current,
                )
            malformed_values = {
                key: latest_status[key] for key in DURABLE_MALFORMED_FIELDS
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in malformed_values.values()
            ):
                current = summary(
                    complete=False, reason="malformed_durable_state"
                )
                raise IndexingConvergenceError(
                    "memory indexing status has malformed durable state", current
                )
            in_progress = latest_status.get("in_progress")
            if not isinstance(in_progress, bool):
                current = summary(
                    complete=False, reason="malformed_status_shape"
                )
                raise IndexingConvergenceError(
                    "memory indexing status has malformed in-progress state",
                    current,
                )
            missing_report = (
                set(_CURRENT_DREAM_REPORT_FAILURE_FIELDS)
                | set(_CURRENT_DREAM_REPORT_BOOLEAN_FIELDS)
            ) - set(report)
            if missing_report:
                current = summary(
                    complete=False, reason="malformed_cycle_failure_report"
                )
                raise IndexingConvergenceError(
                    "memory indexing report lacks current failure fields", current
                )
            report_failure_values = {
                key: report[key] for key in _CURRENT_DREAM_REPORT_FAILURE_FIELDS
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in report_failure_values.values()
            ) or any(
                not isinstance(report[key], bool)
                for key in _CURRENT_DREAM_REPORT_BOOLEAN_FIELDS
            ):
                current = summary(
                    complete=False, reason="malformed_cycle_failure_report"
                )
                raise IndexingConvergenceError(
                    "memory indexing report has malformed current failure fields",
                    current,
                )
        else:
            malformed_values = {}
            in_progress = latest_status.get("in_progress", False)
            if not isinstance(in_progress, bool):
                current = summary(
                    complete=False, reason="malformed_status_shape"
                )
                raise IndexingConvergenceError(
                    "legacy memory indexing status has malformed in-progress state",
                    current,
                )
            report_failure_values = {
                key: report[key]
                for key in (
                    "aggregation_fusion_failures",
                    "aggregation_build_exceptions",
                )
                if key in report
            }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in report_failure_values.values()
        ):
            current = summary(
                complete=False,
                reason="malformed_aggregation_failure_report",
            )
            raise IndexingConvergenceError(
                "memory indexing report has malformed aggregation failures",
                current,
            )
        report_failures = sum(report_failure_values.values())
        exhausted = report.get("budget_exhausted") is True
        provider_exhausted = (
            report.get("extraction_provider_attempt_budget_exhausted") is True
        )
        skipped_locked = report.get("skipped_locked") is True
        quarantine_total = sum(quarantine_values.values())
        terminal_loss_total = sum(terminal_loss_values.values())
        malformed_total = sum(malformed_values.values())
        complete = bool(
            not exhausted
            and not provider_exhausted
            and not skipped_locked
            and not in_progress
            and pending == 0
            and report_failures == 0
        )
        if deadline.expired:
            current = summary(
                complete=False, reason="timeout_after_cycle",
            )
            raise IndexingConvergenceError(
                "memory indexing exceeded its timeout", current
            )
        if require_healthy and malformed_total > 0:
            current = summary(
                complete=complete, reason="malformed_durable_state"
            )
            raise IndexingConvergenceError(
                "memory indexing has malformed durable cursor or authority state",
                current,
            )
        if require_healthy and terminal_loss_total > 0:
            current = summary(
                complete=complete, reason="terminal_extraction_source_loss"
            )
            raise IndexingConvergenceError(
                "memory indexing has terminal extraction source loss", current
            )
        if require_healthy and quarantine_total > 0:
            current = summary(
                complete=complete, reason="quarantined_extraction"
            )
            raise IndexingConvergenceError(
                "memory indexing has current-policy quarantined work", current
            )
        if complete:
            current = summary(complete=True)
            if require_healthy and coverage_integrity_failure > 0:
                # Coverage repair is local and the runner clears this durable
                # record after a later complete producer walk. Keep spending
                # the caller's finite cycle bound, but never certify the
                # mechanically drained snapshot as healthy in the meantime.
                continue
            if require_healthy and not current["healthy"]:
                has_terminal_loss = any(
                    key.startswith("terminal_loss_")
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value > 0
                    for key, value in latest_status.items()
                )
                current["failure_reason"] = (
                    "terminal_extraction_source_loss"
                    if has_terminal_loss else "quarantined_extraction"
                )
                raise IndexingConvergenceError(
                    "memory indexing completed with unresolved extraction loss",
                    current,
                )
            if deadline.expired:
                current["complete"] = False
                current["healthy"] = False
                current["failure_reason"] = "timeout_after_cycle"
                raise IndexingConvergenceError(
                    "memory indexing exceeded its timeout", current
                )
            return current

    if deadline.expired:
        current = summary(complete=False, reason="timeout_after_cycle")
        raise IndexingConvergenceError(
            "memory indexing exceeded its timeout", current
        )
    terminal_reason = (
        "coverage_integrity_failure"
        if isinstance(latest_status.get("coverage_integrity_failures"), int)
        and not isinstance(latest_status.get("coverage_integrity_failures"), bool)
        and latest_status["coverage_integrity_failures"] > 0
        else "max_cycles_exhausted"
    )
    current = summary(complete=False, reason=terminal_reason)
    raise IndexingConvergenceError(
        "memory indexing did not converge within max_cycles", current
    )


def embedding_backlog_status(conn, client: object | None) -> dict[str, int]:
    """Count absent, stale, wrong-identity, or invalid vector mirrors.

    All corpus work stays in SQLite and every query returns one integer.  The
    message query uses immutable lossless coverage proofs, so the audit remains
    valid after opt-in raw-message pruning.  ``client is None`` means
    embeddings are disabled, not that every source row is a backlog item.
    """

    if client is None:
        return {
            "pending_chunk_embeddings": 0,
            "pending_message_embeddings": 0,
            "pending_edge_embeddings": 0,
            "pending_episode_embeddings": 0,
            "pending_fact_embeddings": 0,
        }

    from hymem.core.graph import live_edge_predicate
    from hymem.core.message_records import message_record_proof_valid
    from hymem.core.db import register_read_authority_functions
    from hymem.core.vectors import decode_vector
    from hymem.dreaming.aggregation_material import embedding_execution_identity
    from hymem.extraction.embeddings import embedding_text_hash

    try:
        producer, model, dim = embedding_execution_identity(client)
    except Exception as exc:
        raise BenchmarkIntegrityError(
            "embedding client identity is unavailable"
        ) from exc
    if (
        producer.get("identity_exact") is not True
        or producer.get("reuse_scope") != "durable"
        or not isinstance(model, str) or not model
        or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
    ):
        raise BenchmarkIntegrityError("embedding client identity is invalid")

    conn.create_function(
        "hymem_benchmark_embedding_hash", 1,
        lambda value: embedding_text_hash(str(value)), deterministic=True,
    )
    conn.create_function(
        "hymem_message_record_proof_valid", 4,
        message_record_proof_valid, deterministic=True,
    )
    register_read_authority_functions(conn)

    def valid_vector(value: object, expected_dim: object) -> int:
        try:
            vector = decode_vector(value)
            return int(
                len(vector) == int(expected_dim)
                and all(math.isfinite(float(item)) for item in vector)
                and math.sqrt(sum(float(item) ** 2 for item in vector)) > 0.0
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return 0

    conn.create_function(
        "hymem_benchmark_vector_valid", 2, valid_vector, deterministic=True,
    )

    def invalid_vector(alias: str) -> str:
        return (
            f"({alias}.vector_json IS NULL OR {alias}.model<>? OR {alias}.dim<>? "
            f"OR hymem_benchmark_vector_valid({alias}.vector_json,?)<>1)"
        )

    def count(sql: str, params: tuple[Any, ...]) -> int:
        row = conn.execute(sql, params).fetchone()
        if row is None:
            raise BenchmarkIntegrityError("embedding backlog query returned no row")
        value = row[0]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchmarkIntegrityError(
                "embedding backlog query returned a malformed count"
            )
        return value

    vector_params = (model, dim, dim)
    pending_chunks = count(
        "SELECT COUNT(*) FROM chunks c "
        "LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id "
        "WHERE c.chunk_kind='extraction' AND (e.text_hash<>"
        "hymem_benchmark_embedding_hash(c.text) OR e.text_hash IS NULL OR "
        + invalid_vector("e") + ")",
        vector_params,
    )
    pending_messages = count(
        "SELECT COUNT(*) FROM message_retention_coverage mc "
        "JOIN sessions s ON s.id=mc.source_session_id "
        "JOIN chunks c ON c.id=mc.chunk_id "
        "LEFT JOIN message_embeddings e ON e.message_id=mc.message_id "
        "WHERE mc.coverage_version='dream-lossless-message-v1' "
        "AND mc.source_role IN ('user','assistant') "
        "AND s.coverage_message_id IS NOT NULL "
        "AND typeof(mc.message_id)='integer' "
        "AND mc.message_id<=s.coverage_message_id "
        "AND c.session_id=mc.source_session_id "
        "AND c.start_message_id=mc.message_id "
        "AND c.end_message_id=mc.message_id AND c.chunk_kind='coverage' "
        "AND hymem_message_record_proof_valid(c.text,mc.message_content_hash,"
        "mc.hash_version,mc.record_version)=1 "
        "AND (e.source_coverage_chunk_id<>mc.chunk_id "
        "OR e.source_coverage_version<>mc.coverage_version "
        "OR e.text_hash<>hymem_benchmark_embedding_hash("
        "json_extract(c.text,'$.content')) OR e.text_hash IS NULL OR "
        + invalid_vector("e") + ")",
        vector_params,
    )
    edge_text = (
        "(k.subject_canonical || ' ' || k.predicate || ' ' || "
        "k.object_canonical)"
    )
    pending_edges = count(
        "SELECT COUNT(*) FROM knowledge_graph k "
        f"LEFT JOIN edge_embeddings e ON e.edge_text={edge_text} "
        f"WHERE {live_edge_predicate('k')} AND " + invalid_vector("e"),
        vector_params,
    )
    pending_episodes = count(
        "SELECT COUNT(*) FROM episodes ep JOIN sessions s ON s.id=ep.session_id "
        "LEFT JOIN episode_embeddings e ON e.episode_id=ep.id "
        "WHERE (ep.digest_generation IS NULL OR "
        "ep.digest_generation=s.digest_published_generation) "
        "AND (e.text_hash<>hymem_benchmark_embedding_hash("
        "ep.title || char(10) || ep.summary) OR e.text_hash IS NULL OR "
        + invalid_vector("e") + ")",
        vector_params,
    )
    pending_facts = count(
        "SELECT COUNT(*) FROM narrative_facts f "
        "JOIN fact_extraction_outcomes o ON o.slice_key=f.source_outcome_key "
        "LEFT JOIN narrative_fact_embeddings e ON e.fact_id=f.id "
        "WHERE f.source_outcome_key IS NOT NULL "
        "AND f.lifecycle_status='active' AND f.invalid_at IS NULL "
        "AND o.outcome_status='success' AND o.source_manifest_complete=1 "
        "AND o.source_manifest_version='fact-source-manifest-v1' "
        "AND o.source_manifest_count>0 "
        "AND (e.text_hash<>hymem_benchmark_embedding_hash(f.text) "
        "OR e.text_hash IS NULL OR " + invalid_vector("e") + ")",
        vector_params,
    )
    return {
        "pending_chunk_embeddings": pending_chunks,
        "pending_message_embeddings": pending_messages,
        "pending_edge_embeddings": pending_edges,
        "pending_episode_embeddings": pending_episodes,
        "pending_fact_embeddings": pending_facts,
    }


def durable_indexing_status(
    memory: object, embedding_client: object | None,
) -> dict[str, Any]:
    """Compose one benchmark-safe, read-only durable completion snapshot."""

    snapshot_status = getattr(memory, "dream_status_with_snapshot", None)
    try:
        if callable(snapshot_status):
            raw_status = snapshot_status(
                lambda conn: embedding_backlog_status(conn, embedding_client)
            )
            composed_snapshot = True
        else:
            # Explicit compatibility path for legacy/custom test doubles. Real
            # HyMem instances expose dream_status_with_snapshot. A disabled
            # embedding policy needs no second database read and is therefore
            # the only safe schema-v2 fallback.
            if embedding_client is not None:
                raise BenchmarkIntegrityError(
                    "memory lacks coherent benchmark status snapshot support"
                )
            raw_status = memory.dream_status()
            composed_snapshot = False
    except Exception as exc:
        raise BenchmarkIntegrityError(
            "memory indexing status is unavailable"
        ) from exc
    if not isinstance(raw_status, Mapping):
        raise BenchmarkIntegrityError(
            "memory indexing status callback returned a malformed value"
        )
    config = getattr(memory, "config", None)
    if config is None:
        raise BenchmarkIntegrityError(
            "memory indexing status is missing its effective configuration"
        )
    if raw_status.get("dream_status_schema") != DREAM_STATUS_SCHEMA_VERSION:
        raise BenchmarkIntegrityError(
            "memory indexing status has an unsupported durable schema"
        )
    authority_reason = _phase1_authority_status_reason(raw_status)
    if authority_reason == "phase1_producer_unavailable":
        raise BenchmarkIntegrityError(
            "memory indexing status lacks an exact Phase-1 producer"
        )
    if authority_reason is not None:
        raise BenchmarkIntegrityError(
            "memory indexing status has malformed Phase-1 producer authority"
        )
    required_counts = (
        *DURABLE_PENDING_FIELDS,
        *DURABLE_MALFORMED_FIELDS,
        *_DURABLE_QUARANTINE_FIELDS,
        "terminal_loss_chunks",
        "coverage_integrity_failures",
    )
    for key in required_counts:
        reported = raw_status.get(key)
        if (
            isinstance(reported, bool)
            or not isinstance(reported, int)
            or reported < 0
        ):
            raise BenchmarkIntegrityError(
                f"memory indexing status has malformed {key}"
            )
    if not isinstance(raw_status.get("in_progress"), bool):
        raise BenchmarkIntegrityError(
            "memory indexing status has malformed in_progress"
        )

    status = dict(raw_status)
    if not composed_snapshot:
        status.update(embedding_backlog_status(None, None))
    status["benchmark_indexing_status_schema"] = (
        BENCHMARK_INDEXING_STATUS_VERSION
    )
    return status


@dataclass(frozen=True)
class PythonSourceSlice:
    """Exact transitive top-level symbols selected from a Python source file."""

    path: Path
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        raw_symbols = tuple(self.symbols)
        if not raw_symbols or any(
            not isinstance(symbol, str) or not symbol.isidentifier()
            for symbol in raw_symbols
        ):
            raise BenchmarkIntegrityError(
                "Python source slices require one or more identifier symbols"
            )
        normalized = tuple(sorted(set(raw_symbols)))
        object.__setattr__(self, "symbols", normalized)


@dataclass(frozen=True)
class BenchmarkIdentityData:
    """An explicitly classified non-code input to benchmark execution."""

    path: Path
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.kind not in {"config", "data", "evaluator", "prompt"}:
            raise BenchmarkIntegrityError(
                "benchmark identity data kind must be config/data/evaluator/prompt"
            )


@dataclass(frozen=True)
class _PythonBinding:
    names: tuple[str, ...]
    node: ast.AST
    import_spec: tuple[str, str, int, str, str | None] | None = None


def _assignment_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    else:
        return ()
    names: list[str] = []
    for target in targets:
        names.extend(
            child.id for child in ast.walk(target)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        )
    return tuple(dict.fromkeys(names))


def _target_names(target: ast.AST | None) -> tuple[str, ...]:
    if target is None:
        return ()
    return tuple(dict.fromkeys(
        child.id for child in ast.walk(target)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    ))


def _compound_binding_names(statements: Iterable[ast.stmt]) -> tuple[str, ...]:
    """Names bound by module-level control flow, excluding nested scopes."""

    names: list[str] = []
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(statement.name)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names.extend(_assignment_names(statement))
        elif isinstance(statement, ast.Import):
            names.extend(
                alias.asname or alias.name.split(".", 1)[0]
                for alias in statement.names
            )
        elif isinstance(statement, ast.ImportFrom):
            if any(alias.name == "*" for alias in statement.names):
                raise BenchmarkIntegrityError(
                    "Python source slices do not permit module-level star imports"
                )
            names.extend(alias.asname or alias.name for alias in statement.names)
        elif isinstance(statement, ast.Try):
            branches = [statement.body, statement.orelse, statement.finalbody]
            branches.extend(handler.body for handler in statement.handlers)
            for branch in branches:
                names.extend(_compound_binding_names(branch))
        elif isinstance(statement, ast.If):
            names.extend(_compound_binding_names(statement.body))
            names.extend(_compound_binding_names(statement.orelse))
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            names.extend(_target_names(statement.target))
            names.extend(_compound_binding_names(statement.body))
            names.extend(_compound_binding_names(statement.orelse))
        elif isinstance(statement, ast.While):
            names.extend(_compound_binding_names(statement.body))
            names.extend(_compound_binding_names(statement.orelse))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                names.extend(_target_names(item.optional_vars))
            names.extend(_compound_binding_names(statement.body))
    return tuple(dict.fromkeys(names))


def _module_bindings(
    tree: ast.Module,
) -> tuple[dict[str, _PythonBinding], tuple[ast.AST, ...]]:
    bindings: dict[str, _PythonBinding] = {}
    future_nodes: list[ast.AST] = []

    def register(name: str, binding: _PythonBinding) -> None:
        previous = bindings.get(name)
        if previous is not None and previous.node is not binding.node:
            raise BenchmarkIntegrityError(
                f"Python source slice has ambiguous top-level binding: {name!r}"
            )
        bindings[name] = binding

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            register(node.name, _PythonBinding((node.name,), node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names = _assignment_names(node)
            binding = _PythonBinding(names, node)
            for name in names:
                register(name, binding)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                register(
                    bound,
                    _PythonBinding(
                        (bound,), node,
                        ("import", alias.name, 0, alias.name, alias.asname),
                    ),
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                future_nodes.append(node)
                continue
            for alias in node.names:
                if alias.name == "*":
                    raise BenchmarkIntegrityError(
                        "Python source slices do not permit top-level star imports"
                    )
                bound = alias.asname or alias.name
                register(
                    bound,
                    _PythonBinding(
                        (bound,), node,
                        (
                            "from", node.module or "", node.level,
                            alias.name, alias.asname,
                        ),
                    ),
                )
        elif isinstance(
            node,
            (ast.Try, ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith),
        ):
            # Package/direct-script fallback imports and optional-platform
            # assignments (for example ``fcntl`` or ``None``) are one logical
            # executable binding surface. Select the complete control group.
            normalized = _compound_binding_names((node,))
            if normalized:
                binding = _PythonBinding(normalized, node)
                for name in normalized:
                    register(name, binding)
    return bindings, tuple(future_nodes)


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    values = [test.left, *test.comparators]
    return (
        isinstance(test.ops[0], ast.Eq)
        and any(isinstance(value, ast.Name) and value.id == "__name__" for value in values)
        and any(isinstance(value, ast.Constant) and value.value == "__main__" for value in values)
    )


class _ImportTimeLoadVisitor(ast.NodeVisitor):
    """Collect names evaluated at module import without entering function bodies."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast API
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def _visit_definition_header(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        for argument in (
            *node.args.posonlyargs, *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_definition_header(node)

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef,
    ) -> None:
        self._visit_definition_header(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        # A class body executes at import time. Its methods are skipped by the
        # definition visitors, while assignments/invariants remain visible.
        for statement in node.body:
            self.visit(statement)

    def visit_Lambda(self, _node: ast.Lambda) -> None:  # noqa: N802
        return


def _import_time_load_names(node: ast.AST) -> set[str]:
    visitor = _ImportTimeLoadVisitor()
    visitor.visit(node)
    return visitor.names


def _python_slice_selection(
    source_slice: PythonSourceSlice,
) -> tuple[str, tuple[_PythonBinding, ...], tuple[ast.AST, ...]]:
    path = source_slice.path.resolve()
    if not path.is_file():
        raise BenchmarkIntegrityError(f"code file does not exist: {path}")
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path.name)
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise BenchmarkIntegrityError(
            f"cannot parse Python source slice: {path.name}"
        ) from exc
    bindings, future_nodes = _module_bindings(tree)
    missing = sorted(
        symbol for symbol in source_slice.symbols if symbol not in bindings
    )
    if missing:
        raise BenchmarkIntegrityError(
            f"Python source slice lacks requested symbols: {', '.join(missing)}"
        )

    pending = list(source_slice.symbols)
    seen_names: set[str] = set()
    selected: dict[int, _PythonBinding] = {}
    selected_nodes: set[int] = set()
    side_effect_candidates = tuple(
        node for node in tree.body
        if not isinstance(
            node,
            (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                ast.Assign, ast.AnnAssign, ast.AugAssign,
                ast.Import, ast.ImportFrom,
            ),
        )
        and not _is_main_guard(node)
    )
    while True:
        while pending:
            name = pending.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            binding = bindings.get(name)
            if binding is None:
                continue
            selected[id(binding)] = binding
            selected_nodes.add(id(binding.node))
            if binding.import_spec is not None:
                continue
            pending.extend(
                child.id for child in ast.walk(binding.node)
                if (
                    isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id in bindings
                    and child.id not in seen_names
                )
            )
        added_side_effect = False
        for node in side_effect_candidates:
            if id(node) in selected_nodes:
                continue
            if not (_import_time_load_names(node) & seen_names):
                continue
            binding = _PythonBinding((), node)
            selected[id(binding)] = binding
            selected_nodes.add(id(node))
            pending.extend(
                name for name in _import_time_load_names(node)
                if name in bindings and name not in seen_names
            )
            added_side_effect = True
        if not pending and not added_side_effect:
            break
    ordered = tuple(sorted(
        selected.values(),
        key=lambda binding: (
            getattr(binding.node, "lineno", -1),
            getattr(binding.node, "col_offset", -1),
            binding.names,
        ),
    ))
    return source, ordered, future_nodes


def _python_source_slice_bytes(source_slice: PythonSourceSlice) -> bytes:
    _source, selected, future_nodes = _python_slice_selection(source_slice)
    chunks: list[dict[str, Any]] = []
    for node in future_nodes:
        chunks.append({
            "kind": "future",
            "ast": ast.dump(node, annotate_fields=True, include_attributes=False),
        })
    for binding in selected:
        if binding.import_spec is not None:
            chunks.append({
                "kind": "import",
                "names": binding.names,
                "spec": binding.import_spec,
            })
            continue
        chunks.append({
            "kind": "definition",
            "names": binding.names,
            # Normalized AST includes decorators/defaults/annotations while
            # excluding comments and source-location/checkout noise.
            "ast": ast.dump(
                binding.node, annotate_fields=True, include_attributes=False
            ),
        })
    return json.dumps(
        {
            "schema": "python-source-slice-v1",
            "roots": source_slice.symbols,
            "chunks": chunks,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _symbols_imported_from_nodes(
    nodes: Iterable[ast.AST], *, module_names: Iterable[str],
) -> tuple[str, ...]:
    accepted = {module.lstrip(".") for module in module_names if module}
    imported: set[str] = set()
    module_aliases: set[str] = set()
    node_list = tuple(nodes)
    for node in node_list:
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                if (child.module or "").lstrip(".") not in accepted:
                    continue
                for alias in child.names:
                    if alias.name == "*":
                        raise BenchmarkIntegrityError(
                            "benchmark dependencies cannot use star imports"
                        )
                    imported.add(alias.name)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name.lstrip(".") in accepted:
                        module_aliases.add(
                            alias.asname or alias.name.split(".", 1)[0]
                        )
    for node in node_list:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id in module_aliases
            ):
                imported.add(child.attr)
    return tuple(sorted(imported))


def python_file_imported_symbols(
    path: str | os.PathLike[str], *, module_names: Iterable[str],
) -> tuple[str, ...]:
    """Return exact attributes imported from target modules anywhere in a file."""

    source = Path(path)
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=source.name)
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise BenchmarkIntegrityError(
            f"cannot inspect Python benchmark dependencies: {source.name}"
        ) from exc
    return _symbols_imported_from_nodes(tree.body, module_names=module_names)


def python_slice_imported_symbols(
    source_slice: PythonSourceSlice, *, module_names: Iterable[str],
) -> tuple[str, ...]:
    """Return dependency symbols reachable from a selected Python source slice."""

    _source, selected, _future = _python_slice_selection(source_slice)
    accepted = {module.lstrip(".") for module in module_names if module}
    imported: set[str] = set()
    module_aliases: set[str] = set()
    definitions: list[ast.AST] = []
    for binding in selected:
        spec = binding.import_spec
        if spec is None:
            definitions.append(binding.node)
            continue
        kind, module, _level, name, asname = spec
        if module.lstrip(".") not in accepted:
            continue
        if kind == "from":
            imported.add(name)
        else:
            module_aliases.add(asname or module.split(".", 1)[0])
    imported.update(
        _symbols_imported_from_nodes(definitions, module_names=module_names)
    )
    for node in definitions:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id in module_aliases
            ):
                imported.add(child.attr)
    return tuple(sorted(imported))


def _imported_modules_from_nodes(nodes: Iterable[ast.AST]) -> tuple[str, ...]:
    modules: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                modules.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    continue
                base = child.module or ""
                if base:
                    modules.add(base)
                    modules.update(
                        f"{base}.{alias.name}"
                        for alias in child.names if alias.name != "*"
                    )
    return tuple(sorted(modules))


def python_file_imported_modules(
    path: str | os.PathLike[str],
) -> tuple[str, ...]:
    """Return absolute module candidates imported anywhere in a Python file."""

    source = Path(path)
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=source.name)
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise BenchmarkIntegrityError(
            f"cannot inspect Python benchmark dependencies: {source.name}"
        ) from exc
    return _imported_modules_from_nodes(tree.body)


def python_slice_imported_modules(
    source_slice: PythonSourceSlice,
) -> tuple[str, ...]:
    """Return absolute module candidates reachable from a source slice."""

    _source, selected, _future = _python_slice_selection(source_slice)
    return _imported_modules_from_nodes(binding.node for binding in selected)


def _resolve_local_python_module(root: Path, module: str) -> tuple[Path, ...]:
    parts = module.split(".")
    relative = Path(*parts)
    candidates = (root / f"{relative}.py", root / relative / "__init__.py")
    resolved = [path.resolve() for path in candidates if path.is_file()]
    if not resolved:
        return ()
    # Importing ``pkg.child.module`` executes both package initializers before
    # the leaf. They are executable dependencies even if the leaf never names
    # them explicitly.
    for depth in range(1, len(parts)):
        initializer = root.joinpath(*parts[:depth], "__init__.py")
        if initializer.is_file():
            resolved.append(initializer.resolve())
    return tuple(dict.fromkeys(resolved))


def _absolute_import_module(
    node: ast.ImportFrom, *, current_package: str,
) -> str:
    if node.level == 0:
        return node.module or ""
    base = current_package.split(".") if current_package else []
    climb = node.level - 1
    if climb > len(base):
        return ""
    base = base[: len(base) - climb]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def local_python_dependency_paths(
    entry_paths: Iterable[str | os.PathLike[str]],
    *,
    root: Path,
    package_prefixes: Iterable[str],
) -> tuple[Path, ...]:
    """Resolve a deterministic local import closure for selected packages."""

    root = root.resolve()
    prefixes = tuple(sorted(set(package_prefixes)))
    pending = [Path(path).resolve() for path in entry_paths]
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file():
            raise BenchmarkIntegrityError(f"code file does not exist: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise BenchmarkIntegrityError(
                "local Python dependency lies outside the identity root"
            ) from exc
        module_parts = list(relative.with_suffix("").parts)
        is_package = bool(module_parts and module_parts[-1] == "__init__")
        if is_package:
            module_parts.pop()
        current_module = ".".join(module_parts)
        current_package = (
            current_module if is_package
            else current_module.rpartition(".")[0]
        )
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise BenchmarkIntegrityError(
                f"cannot inspect local Python dependency: {relative.as_posix()}"
            ) from exc
        found.add(path)
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _absolute_import_module(
                    node, current_package=current_package
                )
                if base:
                    modules.add(base)
                    modules.update(
                        f"{base}.{alias.name}"
                        for alias in node.names if alias.name != "*"
                    )
        for module in modules:
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in prefixes
            ):
                pending.extend(_resolve_local_python_module(root, module))
    return tuple(sorted(
        found, key=lambda item: item.relative_to(root).as_posix()
    ))


def benchmark_hymem_source_paths(
    package_path: Path,
    *,
    root: Path,
    dependency_sources: Iterable[
        str | os.PathLike[str] | PythonSourceSlice
    ],
) -> tuple[Path, ...]:
    """Exact HyMem import closure plus runtime-loaded database resources.

    ``local_python_dependency_paths`` follows ordinary, aliased, relative and
    function-local imports (including imports under ``try``/``if`` blocks).
    The database schema and migrations are discovered at runtime through
    :mod:`importlib.resources`, so they have no import edge for the AST walker
    to follow and are attached explicitly whenever ``hymem.core.db`` is in the
    executable closure.  Importing the migrations resource package executes
    its ``__init__.py`` too, making that initializer part of the identity.
    """

    package = Path(package_path).resolve()
    root = Path(root).resolve()
    if not package.is_dir():
        raise BenchmarkIntegrityError(f"HyMem package does not exist: {package}")
    imported_modules: set[str] = set()
    for source in dependency_sources:
        if isinstance(source, PythonSourceSlice):
            imported_modules.update(python_slice_imported_modules(source))
        else:
            imported_modules.update(python_file_imported_modules(source))
    prefix = package.name
    seeds: set[Path] = set()
    for module in imported_modules:
        if module == prefix or module.startswith(f"{prefix}."):
            seeds.update(_resolve_local_python_module(root, module))
    if not seeds:
        raise BenchmarkIntegrityError(
            "benchmark code identity found no imported HyMem implementation"
        )
    python_paths = local_python_dependency_paths(
        seeds, root=root, package_prefixes=(prefix,)
    )
    resources: list[Path] = []
    core_db = package / "core/db.py"
    if core_db.resolve() in python_paths:
        schema = package / "core/schema.sql"
        if schema.is_file():
            resources.append(schema.resolve())
        migrations = package / "core/migrations"
        if migrations.is_dir():
            initializer = migrations / "__init__.py"
            if initializer.is_file():
                resources.append(initializer.resolve())
            resources.extend(path.resolve() for path in migrations.glob("*.sql"))
    return tuple(sorted(
        {*python_paths, *resources},
        key=lambda item: item.relative_to(root).as_posix(),
    ))


def file_hash(path: str | os.PathLike[str]) -> str:
    """Stream a file into SHA-256; missing/non-files fail loudly."""

    source = Path(path)
    if not source.is_file():
        raise BenchmarkIntegrityError(f"data/code file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def code_hash(
    paths: Iterable[
        str | os.PathLike[str] | PythonSourceSlice | BenchmarkIdentityData
    ],
    *,
    root: Path,
    identity_version: str | None = None,
) -> str:
    """Hash executable files/symbol slices under stable repo-relative names.

    Directories expand only Python/SQL and exclude generated/cache/VCS files.
    Non-code inputs require :class:`BenchmarkIdentityData`, preventing prose
    from becoming code accidentally. Paths outside ``root`` fail closed so
    checkout location can never enter a digest. ``identity_version`` lets a
    narrower artifact (such as a material store) share this framing without
    becoming coupled to the full benchmark-code identity protocol.
    """

    resolved_identity_version = (
        CODE_IDENTITY_VERSION if identity_version is None else identity_version
    )
    if not isinstance(resolved_identity_version, str) or re.fullmatch(
        r"[A-Za-z0-9_.-]{1,96}", resolved_identity_version
    ) is None:
        raise BenchmarkIntegrityError("code identity version is invalid")
    root = root.resolve()
    files: set[Path] = set()
    slices: set[PythonSourceSlice] = set()
    data_inputs: set[BenchmarkIdentityData] = set()
    for raw in paths:
        if isinstance(raw, PythonSourceSlice):
            slices.add(PythonSourceSlice(raw.path.resolve(), raw.symbols))
            continue
        if isinstance(raw, BenchmarkIdentityData):
            data_inputs.add(BenchmarkIdentityData(raw.path.resolve(), raw.kind))
            continue
        path = Path(raw).resolve()
        if path.is_dir():
            for candidate in path.rglob("*"):
                if (
                    not candidate.is_file()
                    or "__pycache__" in candidate.parts
                    or ".git" in candidate.parts
                    or candidate.suffix.casefold() not in _CODE_DIRECTORY_SUFFIXES
                ):
                    continue
                if candidate.is_symlink():
                    raise BenchmarkIntegrityError(
                        "code directories must not contain executable symlinks"
                    )
                files.add(candidate.resolve())
        elif path.is_file() and path.suffix.casefold() in _CODE_DIRECTORY_SUFFIXES:
            files.add(path)
        elif path.is_file():
            raise BenchmarkIntegrityError(
                "non-code benchmark identity inputs require explicit data classification"
            )
        else:
            raise BenchmarkIntegrityError(f"code path does not exist: {path}")
    digest = hashlib.sha256()
    # v2 changes both the directory inventory and cross-adapter slicing. The
    # explicit salt guarantees old checkpoints fail closed even in the
    # contrived event that the component bytes otherwise collide.
    digest.update(resolved_identity_version.encode("ascii"))
    entries: list[tuple[str, bytes]] = []
    for path in files:
        try:
            name = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise BenchmarkIntegrityError(
                "code path lies outside the identity root"
            ) from exc
        entries.append((f"file:{name}", path.read_bytes()))
    for source_slice in slices:
        try:
            name = source_slice.path.relative_to(root).as_posix()
        except ValueError as exc:
            raise BenchmarkIntegrityError(
                "Python source slice lies outside the identity root"
            ) from exc
        entries.append((
            f"python-slice:{name}:{','.join(source_slice.symbols)}",
            _python_source_slice_bytes(source_slice),
        ))
    for data_input in data_inputs:
        if not data_input.path.is_file():
            raise BenchmarkIntegrityError(
                f"benchmark identity data file does not exist: {data_input.path}"
            )
        try:
            name = data_input.path.relative_to(root).as_posix()
        except ValueError as exc:
            raise BenchmarkIntegrityError(
                "benchmark identity data lies outside the identity root"
            ) from exc
        entries.append((
            f"{data_input.kind}:{name}", data_input.path.read_bytes()
        ))
    for name, payload in sorted(entries, key=lambda item: item[0]):
        digest.update(len(name.encode("utf-8")).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def validate_ids(ids: Iterable[object], *, label: str = "expected") -> tuple[str, ...]:
    """Return non-empty string ids in input order, rejecting duplicates."""

    normalized: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(ids):
        if not isinstance(raw, str) or not raw.strip():
            raise BenchmarkIntegrityError(
                f"{label} id at index {index} must be a non-empty string"
            )
        value = raw.strip()
        if value in seen:
            raise BenchmarkIntegrityError(f"duplicate {label} id: {value!r}")
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise BenchmarkIntegrityError(f"{label} ids must not be empty")
    return tuple(normalized)


def deterministic_split(
    ids: Iterable[object], *, seed: int, dev_fraction: float = 0.5
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Stable hash split independent of dataset input order.

    This is an *internal* split, never represented as an official benchmark
    split.  Both sides are guaranteed non-empty when at least two ids exist.
    """

    ordered = validate_ids(ids)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BenchmarkIntegrityError("split seed must be an integer")
    if not isinstance(dev_fraction, (int, float)) or isinstance(dev_fraction, bool):
        raise BenchmarkIntegrityError("dev_fraction must be numeric")
    if not 0.0 < float(dev_fraction) < 1.0:
        raise BenchmarkIntegrityError("dev_fraction must be strictly between 0 and 1")
    if len(ordered) < 2:
        raise BenchmarkIntegrityError("an internal split needs at least two ids")

    ranked = sorted(
        ordered,
        key=lambda item: hashlib.sha256(
            f"{seed}\0{item}".encode("utf-8")
        ).digest(),
    )
    cut = min(len(ranked) - 1, max(1, round(len(ranked) * float(dev_fraction))))
    dev = set(ranked[:cut])
    # Preserve the dataset's deterministic input order for execution/reporting.
    return (
        tuple(item for item in ordered if item in dev),
        tuple(item for item in ordered if item not in dev),
    )


def _receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_hash"}


def freeze_calibration(
    path: str | os.PathLike[str],
    *,
    benchmark: str,
    dataset_hash: str,
    ids: Iterable[object],
    config: Mapping[str, Any],
    models: Mapping[str, Any],
    seed: int,
    dev_fraction: float = 0.5,
) -> dict[str, Any]:
    """Create an exclusive, auditable dev/holdout calibration receipt.

    The receipt freezes the exact configuration *before* a holdout run.  It
    cannot retroactively make earlier full-set tuning a clean evaluation.
    Existing paths are never overwritten.
    """

    all_ids = validate_ids(ids)
    dev_ids, holdout_ids = deterministic_split(
        all_ids, seed=seed, dev_fraction=dev_fraction
    )
    config_obj = sanitize_for_artifact(
        dict(config), _preserve_evidence_text=False
    )
    model_obj = sanitize_for_artifact(
        dict(models), _preserve_evidence_text=False
    )
    receipt: dict[str, Any] = {
        "schema": CALIBRATION_VERSION,
        "benchmark": str(benchmark),
        "dataset_hash": str(dataset_hash),
        "seed": seed,
        "dev_fraction": float(dev_fraction),
        "dev_ids": list(dev_ids),
        "holdout_ids": list(holdout_ids),
        "all_ids_hash": content_hash(list(all_ids)),
        "config_hash": content_hash(config_obj),
        "model_hash": content_hash(model_obj),
        "config": config_obj,
        "models": model_obj,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "claim": "internal deterministic split; not an official benchmark split",
    }
    receipt["receipt_hash"] = content_hash(_receipt_payload(receipt))
    _write_new_json(Path(path), receipt)
    return receipt


def load_calibration(
    path: str | os.PathLike[str],
    *,
    benchmark: str,
    dataset_hash: str,
    config: Mapping[str, Any],
    models: Mapping[str, Any],
    ids: Iterable[object],
) -> dict[str, Any]:
    """Validate a frozen receipt against this exact test-run identity."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkIntegrityError(
            f"cannot read calibration receipt {source}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema") != CALIBRATION_VERSION:
        raise BenchmarkIntegrityError("unsupported calibration receipt")
    expected_hash = content_hash(_receipt_payload(raw))
    if raw.get("receipt_hash") != expected_hash:
        raise BenchmarkIntegrityError("calibration receipt hash mismatch")
    all_ids = validate_ids(ids)
    stored_config = raw.get("config")
    stored_models = raw.get("models")
    if not isinstance(stored_config, dict) or not isinstance(stored_models, dict):
        raise BenchmarkIntegrityError("calibration config/models must be objects")
    if raw.get("config_hash") != content_hash(stored_config):
        raise BenchmarkIntegrityError("calibration stored config hash mismatch")
    if raw.get("model_hash") != content_hash(stored_models):
        raise BenchmarkIntegrityError("calibration stored model hash mismatch")
    checks = {
        "benchmark": str(benchmark),
        "dataset_hash": str(dataset_hash),
        "config_hash": content_hash(sanitize_for_artifact(
            dict(config), _preserve_evidence_text=False
        )),
        "model_hash": content_hash(sanitize_for_artifact(
            dict(models), _preserve_evidence_text=False
        )),
        "all_ids_hash": content_hash(list(all_ids)),
    }
    for key, expected in checks.items():
        if raw.get(key) != expected:
            raise BenchmarkIntegrityError(
                f"calibration receipt {key} mismatch: "
                f"{raw.get(key)!r} != {expected!r}"
            )
    dev = validate_ids(raw.get("dev_ids", []), label="calibration dev")
    holdout = validate_ids(
        raw.get("holdout_ids", []), label="calibration holdout"
    )
    overlap = set(dev) & set(holdout)
    if overlap:
        raise BenchmarkIntegrityError(
            f"calibration dev/holdout overlap: {sorted(overlap)[:5]}"
        )
    if set(dev) | set(holdout) != set(all_ids):
        missing = set(all_ids) - (set(dev) | set(holdout))
        unknown = (set(dev) | set(holdout)) - set(all_ids)
        raise BenchmarkIntegrityError(
            "calibration split does not partition current ids: "
            f"missing={sorted(missing)[:5]}, unknown={sorted(unknown)[:5]}"
        )
    split_seed = raw.get("seed")
    fraction = raw.get("dev_fraction")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise BenchmarkIntegrityError("calibration seed must be an integer")
    recomputed_dev, recomputed_holdout = deterministic_split(
        all_ids, seed=split_seed, dev_fraction=fraction
    )
    if dev != recomputed_dev or holdout != recomputed_holdout:
        raise BenchmarkIntegrityError(
            "calibration ids do not match its deterministic seed/fraction"
        )
    return raw


def select_protocol_ids(
    ids: Iterable[object], *, split: str, receipt: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """Select ``full``/``dev``/``holdout`` ids under an explicit protocol."""

    all_ids = validate_ids(ids)
    if split == "full":
        return all_ids
    if split not in {"dev", "holdout"}:
        raise BenchmarkIntegrityError(f"unknown protocol split: {split!r}")
    if receipt is None:
        raise BenchmarkIntegrityError(
            f"--protocol-split {split} requires a frozen receipt"
        )
    selected = validate_ids(receipt[f"{split}_ids"], label=f"calibration {split}")
    unknown = set(selected) - set(all_ids)
    if unknown:
        raise BenchmarkIntegrityError(
            f"calibration {split} ids are absent from the dataset: "
            f"{sorted(unknown)[:5]}"
        )
    return selected


def build_manifest(
    *,
    benchmark: str,
    code_sha256: str,
    data_sha256: str,
    config: Mapping[str, Any],
    models: Mapping[str, Any],
    seed: int,
    expected_ids: Iterable[object],
    protocol_split: str,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable/reproducible portion of a run artifact."""

    ids = validate_ids(expected_ids)
    if protocol_split not in {"full", "dev", "holdout"}:
        raise BenchmarkIntegrityError(
            f"unknown protocol split: {protocol_split!r}"
        )
    config_obj = sanitize_for_artifact(
        dict(config), _preserve_evidence_text=False
    )
    model_obj = sanitize_for_artifact(
        dict(models), _preserve_evidence_text=False
    )
    if protocol_split in {"dev", "holdout"}:
        if not isinstance(calibration, Mapping):
            raise BenchmarkIntegrityError(
                f"{protocol_split} manifest requires a frozen calibration receipt"
            )
        if calibration.get("schema") != CALIBRATION_VERSION:
            raise BenchmarkIntegrityError("unsupported calibration receipt")
        if calibration.get("receipt_hash") != content_hash(
            _receipt_payload(calibration)
        ):
            raise BenchmarkIntegrityError("calibration receipt hash mismatch")
        checks = {
            "benchmark": str(benchmark),
            "dataset_hash": str(data_sha256),
            "config_hash": content_hash(config_obj),
            "model_hash": content_hash(model_obj),
            "seed": seed,
        }
        for key, expected in checks.items():
            if calibration.get(key) != expected:
                raise BenchmarkIntegrityError(
                    f"calibration receipt {key} mismatch for manifest"
                )
        receipt_ids = validate_ids(
            calibration.get(f"{protocol_split}_ids", ()),
            label=f"calibration {protocol_split}",
        )
        if ids != receipt_ids:
            raise BenchmarkIntegrityError(
                f"manifest expected ids do not equal calibration {protocol_split} ids"
            )
    label_free_answer_path = config_obj.get("label_free_answer_path")
    if not isinstance(label_free_answer_path, bool):
        raise BenchmarkIntegrityError(
            "manifest config must explicitly declare boolean "
            "label_free_answer_path"
        )
    exploratory_non_comparable = config_obj.get(
        "exploratory_non_comparable", False
    )
    exploratory_label_steering = config_obj.get(
        "exploratory_label_steering", False
    )
    scored_run = config_obj.get("scored_run", True)
    for field_name, field_value in (
        ("exploratory_non_comparable", exploratory_non_comparable),
        ("exploratory_label_steering", exploratory_label_steering),
        ("scored_run", scored_run),
    ):
        if not isinstance(field_value, bool):
            raise BenchmarkIntegrityError(
                f"manifest config {field_name} must be boolean when provided"
            )
    development_only = bool(
        protocol_split != "holdout"
        or not label_free_answer_path
        or exploratory_non_comparable
        or exploratory_label_steering
        or not scored_run
    )
    manifest: dict[str, Any] = {
        "schema": STRICT_PROTOCOL_VERSION,
        "benchmark": str(benchmark),
        "code_hash": str(code_sha256),
        "config_hash": content_hash(config_obj),
        "model_hash": content_hash(model_obj),
        "data_hash": str(data_sha256),
        "expected_ids_hash": content_hash(list(ids)),
        "expected_count": len(ids),
        "seed": seed,
        "protocol_split": protocol_split,
        "development_only": development_only,
        "official_split": False,
        "official_comparable": False,
        "label_free_answer_path": label_free_answer_path,
        "exploratory_label_steering": exploratory_label_steering,
        "exploratory_non_comparable": exploratory_non_comparable,
        "scored_run": scored_run,
        "protocol_limitation": (
            "internal deterministic split, not an official benchmark split"
            if protocol_split in {"dev", "holdout"}
            else "full-set development evidence; may be test-contaminated"
        ),
        "calibration_receipt_hash": (
            calibration.get("receipt_hash") if calibration else None
        ),
        "config": config_obj,
        "models": model_obj,
    }
    manifest["run_id"] = content_hash(manifest)
    return manifest


@dataclass(frozen=True)
class ReconciledResults:
    rows: tuple[dict[str, Any], ...]
    expected: int
    attempted: int
    completed: int
    failed: int
    missing: int
    failure_ids: tuple[str, ...]


def reconcile_results(
    expected_ids: Iterable[object],
    rows: Iterable[Mapping[str, Any]],
    *,
    id_key: str = "question_id",
    verdict_key: str = "correct",
) -> ReconciledResults:
    """Reconcile predictions and retain the complete expected denominator.

    A boolean verdict is a completed prediction (``False`` is an ordinary
    wrong answer).  ``None`` or an explicitly failed row is represented as
    wrong with ``benchmark_failure`` metadata.  Missing rows are synthesized as
    wrong.  Ambiguous structure (duplicates, unknown ids, non-boolean verdicts)
    invalidates the artifact rather than guessing.
    """

    expected = validate_ids(expected_ids)
    expected_set = set(expected)
    by_id: dict[str, dict[str, Any]] = {}
    for index, original in enumerate(rows):
        if not isinstance(original, Mapping):
            raise BenchmarkIntegrityError(f"result row {index} is not an object")
        row = dict(original)
        raw_id = row.get(id_key)
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise BenchmarkIntegrityError(
                f"result row {index} has no non-empty {id_key!r}"
            )
        item_id = raw_id.strip()
        if item_id not in expected_set:
            raise BenchmarkIntegrityError(f"unknown result id: {item_id!r}")
        if item_id in by_id:
            raise BenchmarkIntegrityError(f"duplicate result id: {item_id!r}")
        verdict = row.get(verdict_key)
        explicitly_failed = bool(row.get("benchmark_failure"))
        if verdict is not None and not isinstance(verdict, bool):
            raise BenchmarkIntegrityError(
                f"malformed verdict for {item_id!r}: expected bool/null, "
                f"got {type(verdict).__name__}"
            )
        if verdict is None and not explicitly_failed:
            row["benchmark_failure"] = (
                "judge_or_reader_returned_no_valid_verdict"
            )
        if verdict is None or explicitly_failed:
            row[verdict_key] = False
            row["strict_failure"] = True
        else:
            row["strict_failure"] = False
        row[id_key] = item_id
        by_id[item_id] = row

    failure_ids: list[str] = []
    ordered: list[dict[str, Any]] = []
    missing = 0
    completed = 0
    for item_id in expected:
        row = by_id.get(item_id)
        if row is None:
            missing += 1
            failure_ids.append(item_id)
            row = {
                id_key: item_id,
                verdict_key: False,
                "strict_failure": True,
                "benchmark_failure": "missing_prediction",
            }
        elif row["strict_failure"]:
            failure_ids.append(item_id)
        else:
            completed += 1
        ordered.append(row)
    return ReconciledResults(
        rows=tuple(ordered),
        expected=len(expected),
        attempted=len(by_id),
        completed=completed,
        failed=len(failure_ids),
        missing=missing,
        failure_ids=tuple(failure_ids),
    )


def strict_accuracy(rows: Sequence[Mapping[str, Any]], *, verdict_key: str = "correct") -> float:
    """Accuracy over every reconciled row; malformed/unreconciled rows fail."""

    if not rows:
        return 0.0
    verdicts: list[bool] = []
    for index, row in enumerate(rows):
        verdict = row.get(verdict_key)
        if not isinstance(verdict, bool):
            raise BenchmarkIntegrityError(
                f"row {index} is not reconciled: {verdict_key} is "
                f"{type(verdict).__name__}, not bool"
            )
        verdicts.append(verdict)
    return sum(verdicts) / len(verdicts)


def _atomic_json(path: Path, value: object) -> None:
    """Durably replace ``path`` with canonical JSON from the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _write_new_json(path: Path, value: object) -> None:
    """Atomically publish a new file and refuse to replace existing evidence.

    Bytes are first fsynced in a same-directory temporary file.  A hard link
    then publishes that complete inode under the final name with create-only
    semantics; a crash can leave a disposable temp file, never a truncated
    immutable artifact.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_name, path)
        except FileExistsError as exc:
            raise BenchmarkIntegrityError(
                f"refusing to overwrite immutable artifact: {path}"
            ) from exc
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


_LEASE_REGISTRY_LOCK = threading.Lock()
_LEASE_REGISTRY: dict[str, int] = {}


class _CheckpointLease:
    """Lifetime process ownership backed by a crash-releasing advisory lock."""

    def __init__(self, checkpoint: Path) -> None:
        if fcntl is None:
            raise BenchmarkIntegrityError(
                "atomic benchmark checkpoints require POSIX file locking"
            )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.path = checkpoint.with_name(checkpoint.name + ".lock")
        self._key = str(self.path.absolute())
        self._fd: int | None = None
        with _LEASE_REGISTRY_LOCK:
            if self._key in _LEASE_REGISTRY:
                raise BenchmarkIntegrityError(
                    f"checkpoint already has an owner in this process: {checkpoint}"
                )
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (BlockingIOError, OSError) as exc:
                    try:
                        owner = os.read(fd, 512).decode("utf-8", "replace").strip()
                    except OSError:
                        owner = ""
                    detail = f" ({owner})" if owner else ""
                    raise BenchmarkIntegrityError(
                        f"checkpoint is owned by another live process{detail}: "
                        f"{checkpoint}"
                    ) from exc
                owner_record = _canonical_bytes({
                    "pid": os.getpid(),
                    "checkpoint": checkpoint.name,
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                }) + b"\n"
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, owner_record)
                os.fsync(fd)
                _LEASE_REGISTRY[self._key] = fd
                self._fd = fd
            except Exception:
                os.close(fd)
                raise

    def close(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        with _LEASE_REGISTRY_LOCK:
            if _LEASE_REGISTRY.get(self._key) == fd:
                _LEASE_REGISTRY.pop(self._key, None)
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __del__(self):  # pragma: no cover - explicit close is the tested path
        try:
            self.close()
        except Exception:
            pass


class AtomicCheckpoint:
    """Thread-safe, atomic checkpoint with verified resume identity.

    One entry exists per id.  Retrying a failed item replaces that id's entry
    and increments its attempt count; it can never add a second score row.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        manifest: Mapping[str, Any],
        expected_ids: Iterable[object],
        resume: bool = False,
        retry_failures: bool = False,
        scored: bool = True,
        verdict_key: str = "correct",
    ) -> None:
        if type(resume) is not bool:
            raise BenchmarkIntegrityError(
                "checkpoint resume policy must be a boolean"
            )
        if type(retry_failures) is not bool:
            raise BenchmarkIntegrityError(
                "checkpoint retry_failures policy must be a boolean"
            )
        self.path = Path(path)
        self.manifest = dict(manifest)
        safe_manifest = sanitize_for_artifact(
            self.manifest, _preserve_evidence_text=False
        )
        if not isinstance(safe_manifest, dict) or safe_manifest != self.manifest:
            raise BenchmarkIntegrityError(
                "checkpoint manifest contains unsafe operational data"
            )
        self.expected_ids = validate_ids(expected_ids)
        self.retry_failures = retry_failures
        if type(scored) is not bool:
            raise BenchmarkIntegrityError(
                "checkpoint scored mode must be a boolean"
            )
        self.scored = scored
        if not isinstance(verdict_key, str) or not verdict_key.strip():
            raise BenchmarkIntegrityError("checkpoint verdict key must be non-empty")
        self.verdict_key = verdict_key.strip()
        self._lock = threading.RLock()
        self._lease: _CheckpointLease | None = None
        if self.manifest.get("run_id") != content_hash(
            {k: v for k, v in self.manifest.items() if k != "run_id"}
        ):
            raise BenchmarkIntegrityError("manifest run_id is invalid")
        manifest_expected_count = self.manifest.get("expected_count")
        if (
            type(manifest_expected_count) is not int
            or manifest_expected_count != len(self.expected_ids)
            or self.manifest.get("expected_ids_hash")
            != content_hash(list(self.expected_ids))
        ):
            raise BenchmarkIntegrityError(
                "checkpoint expected ids do not match its manifest identity"
            )
        manifest_scored = self.manifest.get("scored_run")
        if type(manifest_scored) is not bool or manifest_scored is not self.scored:
            raise BenchmarkIntegrityError(
                "checkpoint scored mode does not match its manifest identity"
            )
        self._lease = _CheckpointLease(self.path)
        try:
            if resume:
                self._state = self._load()
                self._validate_state()
                # Recovery of an older checkpoint may encounter free-form
                # failure messages or unsanitized execution instrumentation.
                # Validate immutable identity first, then rewrite only those
                # mutable runtime surfaces to the current bounded schema.
                self._state = self._sanitize_runtime_state(self._state)
                # Validation can reopen a finalized checkpoint for explicit
                # failed-row retry.  Persist that transition only together
                # with the bounded runtime rewrite, never in a prior raw write.
                _atomic_json(self.path, self._state)
            else:
                if self.path.exists():
                    raise BenchmarkIntegrityError(
                        f"checkpoint already exists; pass resume explicitly: {self.path}"
                    )
                self._state = {
                    "schema": CHECKPOINT_VERSION,
                    "run_id": self.manifest["run_id"],
                    "manifest": self.manifest,
                    "expected_ids": list(self.expected_ids),
                    "scored": self.scored,
                    "verdict_key": self.verdict_key,
                    "entries": {},
                    "execution_segments": [],
                    "status": "running",
                }
                _atomic_json(self.path, self._state)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release the process lease; the checkpoint file remains durable."""

        lease = getattr(self, "_lease", None)
        self._lease = None
        if lease is not None:
            lease.close()

    def __enter__(self) -> "AtomicCheckpoint":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - explicit close is preferred
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if self._lease is None:
            raise BenchmarkIntegrityError(
                "checkpoint lease is closed; reopen with resume explicitly"
            )

    def _load(self) -> dict[str, Any]:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkIntegrityError(
                f"cannot resume checkpoint {self.path}: {exc}"
            ) from exc
        if not isinstance(state, dict):
            raise BenchmarkIntegrityError("checkpoint root must be an object")
        return state

    @staticmethod
    def _sanitize_runtime_state(state: Mapping[str, Any]) -> dict[str, Any]:
        """Project a validated checkpoint onto its exact durable root schema.

        Runtime recovery must never turn an arbitrary legacy top-level field
        into newly written evidence. Finalization metadata is legitimate only
        while the checkpoint is terminal; retry reopening deliberately drops
        it so no stale denominator can coexist with mutable rows.
        """

        root_fields = list(_CHECKPOINT_COMMON_ROOT_FIELDS)
        if state.get("status") == "complete":
            root_fields.extend(_CHECKPOINT_FINAL_ROOT_FIELDS)
        result = {
            field_name: state[field_name]
            for field_name in root_fields
            if field_name in state
        }
        entries_raw = state.get("entries")
        if isinstance(entries_raw, Mapping):
            entries: dict[str, Any] = {}
            for item_id, original in entries_raw.items():
                if not isinstance(original, Mapping):
                    entries[str(item_id)] = original
                    continue
                # Entry and history objects are a closed checkpoint schema.
                # Project onto it rather than copying arbitrary legacy keys,
                # which could otherwise turn recovery into a durable rewrite
                # of raw provider diagnostics.
                entry: dict[str, Any] = {
                    "status": original.get("status"),
                    "attempts": original.get("attempts"),
                }
                if original.get("status") == "failed":
                    entry["failure"] = bounded_failure_text(original["failure"])
                row = original.get("row")
                if isinstance(row, Mapping):
                    entry["row"] = sanitize_for_artifact(dict(row))
                history = original.get("attempt_history")
                if isinstance(history, list):
                    bounded_history: list[dict[str, Any]] = []
                    last_index = len(history) - 1
                    verdict_key = state.get("verdict_key", "correct")
                    if not isinstance(verdict_key, str) or not verdict_key:
                        verdict_key = "correct"
                    for index, original_event in enumerate(history):
                        is_last = index == last_index
                        event_status = (
                            str(original.get("status"))
                            if is_last else "failed"
                        )
                        if event_status not in {"completed", "failed"}:
                            event_status = "failed"
                        source_failure: object | None = None
                        event_row: object = None
                        if isinstance(original_event, Mapping):
                            source_failure = original_event.get("failure")
                            event_row = original_event.get("row")
                            if (
                                source_failure is None
                                and isinstance(event_row, Mapping)
                            ):
                                source_failure = event_row.get(
                                    "benchmark_failure"
                                )
                        if source_failure is None and is_last:
                            source_failure = original.get("failure")
                        event_failure = (
                            bounded_failure_text(
                                source_failure or "unspecified_failure"
                            )
                            if event_status == "failed" else None
                        )
                        if isinstance(event_row, Mapping):
                            safe_event_row = sanitize_for_artifact(dict(event_row))
                        elif is_last and isinstance(entry.get("row"), Mapping):
                            safe_event_row = dict(entry["row"])
                        else:
                            safe_event_row = {
                                "question_id": str(item_id),
                                "correct": False,
                                verdict_key: False,
                            }
                        if not isinstance(safe_event_row, dict):
                            safe_event_row = {
                                "question_id": str(item_id),
                                "correct": False,
                                verdict_key: False,
                            }
                        safe_event_row["question_id"] = str(item_id)
                        if event_status == "failed":
                            safe_event_row[verdict_key] = False
                            safe_event_row["benchmark_failure"] = event_failure
                        else:
                            safe_event_row.pop("benchmark_failure", None)
                        bounded_history.append({
                            "attempt": index + 1,
                            "status": event_status,
                            "failure": event_failure,
                            "row": safe_event_row,
                        })
                    entry["attempt_history"] = bounded_history
                entries[str(item_id)] = entry
            result["entries"] = entries
        segments = state.get("execution_segments")
        if isinstance(segments, list):
            result["execution_segments"] = [
                sanitize_for_artifact(
                    dict(segment),
                    _preserve_evidence_text=False,
                    _bound_unknown_text=True,
                )
                if isinstance(segment, Mapping) else segment
                for segment in segments
            ]
        if state.get("status") == "complete":
            counts = state.get("counts")
            if isinstance(counts, Mapping):
                result["counts"] = {
                    field_name: counts[field_name]
                    for field_name in _CHECKPOINT_COUNT_FIELDS
                    if field_name in counts
                }
            else:
                result.pop("counts", None)
            failure_ids = state.get("failure_ids")
            if (
                isinstance(failure_ids, list)
                and all(isinstance(item_id, str) for item_id in failure_ids)
            ):
                result["failure_ids"] = list(failure_ids)
            else:
                result.pop("failure_ids", None)
        return result

    def _validate_state(self) -> None:
        state = self._state
        if state.get("schema") != CHECKPOINT_VERSION:
            raise BenchmarkIntegrityError("unsupported checkpoint schema")
        if state.get("run_id") != self.manifest.get("run_id"):
            raise BenchmarkIntegrityError("checkpoint run identity mismatch")
        embedded_manifest = state.get("manifest")
        if (
            not isinstance(embedded_manifest, dict)
            or _canonical_bytes(embedded_manifest)
            != _canonical_bytes(self.manifest)
        ):
            raise BenchmarkIntegrityError("checkpoint manifest was modified")
        if state.get("expected_ids") != list(self.expected_ids):
            raise BenchmarkIntegrityError("checkpoint expected-id set/order mismatch")
        if type(state.get("scored")) is not bool or state["scored"] is not self.scored:
            raise BenchmarkIntegrityError("checkpoint scored/diagnostic mode mismatch")
        if (
            not isinstance(state.get("verdict_key"), str)
            or state["verdict_key"] != self.verdict_key
        ):
            raise BenchmarkIntegrityError("checkpoint verdict-key mismatch")
        status = state.get("status")
        if status not in {"running", "complete"}:
            raise BenchmarkIntegrityError("checkpoint status is invalid")
        entries = state.get("entries")
        if not isinstance(entries, dict):
            raise BenchmarkIntegrityError("checkpoint entries must be an object")
        unknown = set(entries) - set(self.expected_ids)
        if unknown:
            raise BenchmarkIntegrityError(
                f"checkpoint contains unknown ids: {sorted(unknown)[:5]}"
            )
        for item_id, entry in entries.items():
            if not isinstance(entry, dict):
                raise BenchmarkIntegrityError(
                    f"checkpoint entry {item_id!r} must be an object"
                )
            attempts = entry.get("attempts")
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
                raise BenchmarkIntegrityError(
                    f"checkpoint entry {item_id!r} has invalid attempts"
                )
            if entry.get("status") not in {"completed", "failed"}:
                raise BenchmarkIntegrityError(
                    f"checkpoint entry {item_id!r} has invalid status"
                )
            row = entry.get("row")
            if not isinstance(row, dict):
                raise BenchmarkIntegrityError(
                    f"checkpoint entry {item_id!r} row must be an object"
                )
            if row.get("question_id") != item_id:
                raise BenchmarkIntegrityError(
                    f"checkpoint entry {item_id!r} row id mismatch"
                )
            verdict = row.get(self.verdict_key)
            if verdict is not None and not isinstance(verdict, bool):
                raise BenchmarkIntegrityError(
                    f"checkpoint entry {item_id!r} has malformed verdict"
                )
            failed = entry["status"] == "failed"
            if failed:
                if not isinstance(entry.get("failure"), str) or not entry["failure"]:
                    raise BenchmarkIntegrityError(
                        f"checkpoint entry {item_id!r} has no failure reason"
                    )
                if not row.get("benchmark_failure"):
                    raise BenchmarkIntegrityError(
                        f"checkpoint entry {item_id!r} failed row is unmarked"
                    )
            elif (self.scored and verdict is None) or row.get("benchmark_failure"):
                raise BenchmarkIntegrityError(
                    f"checkpoint entry {item_id!r} completed status conflicts with row"
                )
            history = entry.get("attempt_history")
            if not isinstance(history, list) or len(history) != attempts:
                raise BenchmarkIntegrityError(
                    f"checkpoint entry {item_id!r} attempt history mismatch"
                )
        segments = state.get("execution_segments")
        if not isinstance(segments, list):
            raise BenchmarkIntegrityError(
                "checkpoint execution_segments must be a list"
            )
        segment_ids: set[str] = set()
        for segment in segments:
            if not isinstance(segment, dict):
                raise BenchmarkIntegrityError("execution segment must be an object")
            segment_id = segment.get("segment_id")
            if not isinstance(segment_id, str) or not segment_id:
                raise BenchmarkIntegrityError("execution segment needs a non-empty id")
            if segment_id in segment_ids:
                raise BenchmarkIntegrityError(
                    f"duplicate execution segment id: {segment_id!r}"
                )
            segment_ids.add(segment_id)

        if status == "complete":
            canonical_counts, canonical_failure_ids = (
                self._canonical_finalization_fields()
            )
            stored_counts = state.get("counts")
            if (
                not isinstance(stored_counts, dict)
                or set(stored_counts) != set(_CHECKPOINT_COUNT_FIELDS)
                or any(type(value) is not int for value in stored_counts.values())
                or stored_counts != canonical_counts
            ):
                raise BenchmarkIntegrityError(
                    "checkpoint finalized counts are invalid"
                )
            stored_failure_ids = state.get("failure_ids")
            if (
                not isinstance(stored_failure_ids, list)
                or any(
                    not isinstance(item_id, str)
                    for item_id in stored_failure_ids
                )
                or stored_failure_ids != canonical_failure_ids
            ):
                raise BenchmarkIntegrityError(
                    "checkpoint finalized failure ids are invalid"
                )

        # A finalized checkpoint is terminal by default. Explicit retry may
        # reopen it only when failed or missing work actually remains, while
        # preserving the existing attempt history.
        if status == "complete" and self.retry_failures:
            has_retriable = any(
                item_id not in entries
                or entries[item_id].get("status") == "failed"
                for item_id in self.expected_ids
            )
            if has_retriable:
                self._state["status"] = "running"

    @property
    def pending_ids(self) -> tuple[str, ...]:
        """Ids without a valid completed result; failed ids are retryable."""

        with self._lock:
            self._require_open()
            entries = self._state["entries"]
            if self._state.get("status") == "complete":
                return ()
            return tuple(
                item_id for item_id in self.expected_ids
                if item_id not in entries
                or (
                    self.retry_failures
                    and entries[item_id].get("status") == "failed"
                )
            )

    @property
    def completed_ids(self) -> tuple[str, ...]:
        with self._lock:
            self._require_open()
            entries = self._state["entries"]
            return tuple(
                item_id for item_id in self.expected_ids
                if entries.get(item_id, {}).get("status") == "completed"
            )

    def record(
        self,
        item_id: str,
        *,
        row: Mapping[str, Any] | None,
        failure: str | None = None,
        execution_segment: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist one completed result or failed attempt before returning."""

        self._require_open()
        if item_id not in set(self.expected_ids):
            raise BenchmarkIntegrityError(f"cannot checkpoint unknown id {item_id!r}")
        if (row is None) == (failure is None):
            raise BenchmarkIntegrityError(
                "checkpoint record needs exactly one of row or failure"
            )
        copied: dict[str, Any] | None = None
        if row is not None:
            if not isinstance(row, Mapping):
                raise BenchmarkIntegrityError("checkpoint row must be an object")
            copied = sanitize_for_artifact(dict(row))
            if not isinstance(copied, dict):  # defensive: mappings stay mappings
                raise BenchmarkIntegrityError("checkpoint row sanitization failed")
            raw_row_id = copied.get("question_id")
            if raw_row_id is not None and raw_row_id != item_id:
                raise BenchmarkIntegrityError(
                    f"checkpoint row id {raw_row_id!r} does not match {item_id!r}"
                )
            copied["question_id"] = item_id
            verdict = copied.get(self.verdict_key)
            if verdict is not None and not isinstance(verdict, bool):
                raise BenchmarkIntegrityError(
                    f"checkpoint row {item_id!r} has malformed verdict"
                )
        with self._lock:
            self._require_open()
            if self._state.get("status") != "running":
                raise BenchmarkIntegrityError(
                    "cannot mutate a finalized checkpoint"
                )
            old = self._state["entries"].get(item_id, {})
            if old.get("status") == "completed":
                raise BenchmarkIntegrityError(
                    f"refusing to double-count completed id {item_id!r}"
                )
            if old.get("status") == "failed" and not self.retry_failures:
                raise BenchmarkIntegrityError(
                    f"retrying failed id {item_id!r} requires retry_failures"
                )
            attempts = int(old.get("attempts", 0)) + 1
            history = list(old.get("attempt_history", ()))
            if copied is not None:
                failed = bool(copied.get("benchmark_failure")) or (
                    self.scored and copied.get(self.verdict_key) is None
                )
                entry = {
                    "status": "failed" if failed else "completed",
                    "attempts": attempts,
                    "row": copied,
                }
                if failed:
                    entry["failure"] = bounded_failure_text(
                        copied.get("benchmark_failure")
                        or "no_valid_prediction_verdict"
                    )
            else:
                bounded_failure = bounded_failure_text(failure)
                entry = {
                    "status": "failed",
                    "attempts": attempts,
                    "failure": bounded_failure,
                    "row": {
                        "question_id": item_id,
                        "correct": False,
                        self.verdict_key: False,
                        "benchmark_failure": bounded_failure,
                    },
                }
            history.append({
                "attempt": attempts,
                "status": entry["status"],
                "failure": entry.get("failure"),
                "row": dict(entry["row"]),
            })
            entry["attempt_history"] = history
            entries = self._state["entries"]
            had_prior_entry = item_id in entries
            prior_entry = entries.get(item_id)
            prior_segments = list(self._state["execution_segments"])
            try:
                entries[item_id] = entry
                if execution_segment is not None:
                    self._upsert_execution_segment_locked(execution_segment)
                _atomic_json(self.path, self._state)
            except BaseException:
                # A failure can occur after the in-memory row was installed (or
                # even after os.replace but before directory fsync).  Restore the
                # last acknowledged image so a best-effort abort-segment write
                # cannot accidentally publish the rejected row.
                if had_prior_entry:
                    entries[item_id] = prior_entry
                else:
                    entries.pop(item_id, None)
                self._state["execution_segments"] = prior_segments
                raise

    def _upsert_execution_segment_locked(
        self, metrics: Mapping[str, Any]
    ) -> None:
        segment = sanitize_for_artifact(
            dict(metrics),
            _preserve_evidence_text=False,
            _bound_unknown_text=True,
        )
        if not isinstance(segment, dict):  # pragma: no cover - mapping invariant
            raise BenchmarkIntegrityError("execution segment sanitization failed")
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            raise BenchmarkIntegrityError(
                "execution segment needs a non-empty segment_id"
            )
        segment.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        segments = self._state["execution_segments"]
        for index, existing in enumerate(segments):
            if existing.get("segment_id") == segment_id:
                segments[index] = segment
                break
        else:
            segments.append(segment)

    def update_execution_segment(
        self, segment_id: str, metrics: Mapping[str, Any]
    ) -> None:
        """Atomically replace one process segment's cumulative usage snapshot."""

        with self._lock:
            self._require_open()
            if self._state.get("status") != "running":
                raise BenchmarkIntegrityError(
                    "cannot mutate a finalized checkpoint"
                )
            supplied_id = metrics.get("segment_id")
            if supplied_id is not None and supplied_id != segment_id:
                raise BenchmarkIntegrityError(
                    "execution segment metrics cannot override segment_id"
                )
            prior_segments = list(self._state["execution_segments"])
            try:
                self._upsert_execution_segment_locked(
                    {**dict(metrics), "segment_id": segment_id}
                )
                _atomic_json(self.path, self._state)
            except BaseException:
                self._state["execution_segments"] = prior_segments
                raise

    def add_execution_segment(self, metrics: Mapping[str, Any]) -> None:
        """Compatibility wrapper; prefer ``update_execution_segment``."""

        segment_id = f"segment-{len(self._state['execution_segments']) + 1}"
        self.update_execution_segment(segment_id, metrics)

    def reconcile(self) -> ReconciledResults:
        with self._lock:
            self._require_open()
            if not self.scored:
                entries = self._state["entries"]
                ordered: list[dict[str, Any]] = []
                failure_ids: list[str] = []
                completed = 0
                missing = 0
                for item_id in self.expected_ids:
                    entry = entries.get(item_id)
                    if entry is None:
                        missing += 1
                        failure_ids.append(item_id)
                        ordered.append({
                            "question_id": item_id,
                            "correct": None,
                            "diagnostic_missing": True,
                        })
                    else:
                        row = dict(entry["row"])
                        row["question_id"] = item_id
                        if entry["status"] == "failed":
                            row["benchmark_failure"] = entry["failure"]
                            failure_ids.append(item_id)
                        else:
                            completed += 1
                        ordered.append(row)
                return ReconciledResults(
                    rows=tuple(ordered), expected=len(self.expected_ids),
                    attempted=len(entries), completed=completed,
                    failed=len(failure_ids), missing=missing,
                    failure_ids=tuple(failure_ids),
                )
            rows = []
            for item_id, entry in self._state["entries"].items():
                row = dict(entry.get("row") or {})
                row["question_id"] = item_id
                if entry.get("status") == "failed":
                    row[self.verdict_key] = False
                    row["benchmark_failure"] = entry.get("failure")
                rows.append(row)
            return reconcile_results(
                self.expected_ids, rows, verdict_key=self.verdict_key
            )

    def _canonical_finalization_fields(
        self,
    ) -> tuple[dict[str, int], list[str]]:
        """Derive terminal metadata from durable rows and segment counters."""

        result = self.reconcile()
        durable_row_attempts = sum(
            int(entry["attempts"])
            for entry in self._state["entries"].values()
        )
        # A provider-capable attempt can finish before the atomic row commit
        # fails.  Adapters persist that otherwise orphaned spend in a completed
        # execution segment and rerun the still-pending row on resume.  Use the
        # larger independently durable counter: summing both would double-count
        # every normally committed attempt.
        segment_attempts = sum(
            int(segment.get("attempted_attempts", 0))
            for segment in self._state.get("execution_segments", [])
            if (
                isinstance(segment, Mapping)
                and type(segment.get("attempted_attempts")) is int
                and segment["attempted_attempts"] >= 0
            )
        )
        counts = {
            "expected": result.expected,
            "attempted": result.attempted,
            "unique_attempted": result.attempted,
            "total_attempts": max(durable_row_attempts, segment_attempts),
            "completed": result.completed,
            "failed": result.failed,
            "missing": result.missing,
        }
        return counts, list(result.failure_ids)

    def finalize(self) -> dict[str, Any]:
        """Mark complete and return an artifact-ready checkpoint snapshot."""

        with self._lock:
            self._require_open()
            counts, failure_ids = self._canonical_finalization_fields()
            self._state["status"] = "complete"
            self._state["counts"] = counts
            self._state["failure_ids"] = failure_ids
            _atomic_json(self.path, self._state)
            return json.loads(json.dumps(self._state))


def usage_snapshot(client: object | None) -> dict[str, Any]:
    """Measured provider usage without inventing unavailable precision."""

    if client is None:
        return {
            "calls": None,
            "calls_available": False,
            "request_attempts": None,
            "request_attempts_available": False,
            "successful_responses": None,
            "successful_responses_available": False,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency_s": None,
            "cost_usd": None,
            "token_usage_available": False,
            "latency_available": False,
            "cost_available": False,
        }

    def optional_number(name: str) -> int | float | None:
        value = getattr(client, name, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            return None
        return value

    prompt = optional_number("prompt_tokens")
    completion = optional_number("completion_tokens")
    total = optional_number("total_tokens")
    latency = optional_number("total_latency_s")
    cost = optional_number("cost_usd")
    calls = optional_number("call_count")
    if calls is None:
        calls = optional_number("calls")
    attempts = optional_number("request_attempts")
    successes = optional_number("successful_responses")
    explicit_token_availability = getattr(client, "token_usage_available", None)
    if explicit_token_availability is False:
        prompt = completion = total = None
    return {
        "calls": int(calls) if calls is not None else None,
        "calls_available": calls is not None,
        "request_attempts": int(attempts) if attempts is not None else None,
        "request_attempts_available": attempts is not None,
        "successful_responses": int(successes) if successes is not None else None,
        "successful_responses_available": successes is not None,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "latency_s": latency,
        "cost_usd": cost,
        "token_usage_available": bool(
            total is not None and explicit_token_availability is not False
        ),
        "latency_available": latency is not None,
        "cost_available": cost is not None,
    }


def aggregate_usage_snapshots(
    snapshots: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate usage only where every component is explicitly available."""

    rows = [dict(snapshot) for snapshot in snapshots]
    if not rows:
        return usage_snapshot(None)

    def aggregate(field: str, availability: str) -> tuple[int | float | None, bool]:
        available = all(row.get(availability) is True for row in rows)
        values = [row.get(field) for row in rows]
        if not available or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in values
        ):
            return None, False
        return sum(values), True

    calls, calls_ok = aggregate("calls", "calls_available")
    attempts, attempts_ok = aggregate(
        "request_attempts", "request_attempts_available"
    )
    successes, successes_ok = aggregate(
        "successful_responses", "successful_responses_available"
    )
    prompt, prompt_ok = aggregate("prompt_tokens", "token_usage_available")
    completion, completion_ok = aggregate(
        "completion_tokens", "token_usage_available"
    )
    total, total_ok = aggregate("total_tokens", "token_usage_available")
    latency, latency_ok = aggregate("latency_s", "latency_available")
    cost, cost_ok = aggregate("cost_usd", "cost_available")
    tokens_ok = prompt_ok and completion_ok and total_ok
    return {
        "calls": int(calls) if calls_ok else None,
        "calls_available": calls_ok,
        "request_attempts": int(attempts) if attempts_ok else None,
        "request_attempts_available": attempts_ok,
        "successful_responses": int(successes) if successes_ok else None,
        "successful_responses_available": successes_ok,
        "prompt_tokens": prompt if tokens_ok else None,
        "completion_tokens": completion if tokens_ok else None,
        "total_tokens": total if tokens_ok else None,
        "latency_s": latency if latency_ok else None,
        "cost_usd": cost if cost_ok else None,
        "token_usage_available": tokens_ok,
        "latency_available": latency_ok,
        "cost_available": cost_ok,
    }


def embedding_usage_snapshot(
    client: object | None, *, configured: bool,
) -> dict[str, Any]:
    """Embedding work/cost without confusing lexical and semantic backends."""

    if client is None:
        return {
            "configured": bool(configured),
            "backend": "none" if not configured else "unavailable",
            "quality": "none",
            "network_free": True if not configured else None,
            "model": None,
            "dimension": None,
            "identity_available": not configured,
            "identity_exact": True if not configured else None,
            "reuse_scope": "durable" if not configured else None,
            "calls": 0 if not configured else None,
            "calls_available": not configured,
            "request_attempts": 0 if not configured else None,
            "request_attempts_available": not configured,
            "successful_responses": 0 if not configured else None,
            "successful_responses_available": not configured,
            "input_count": 0 if not configured else None,
            "input_count_available": not configured,
            "input_characters": 0 if not configured else None,
            "input_characters_available": not configured,
            "prompt_tokens": None,
            "total_tokens": None,
            "provider_token_usage_available": False,
            "latency_s": 0.0 if not configured else None,
            "latency_available": not configured,
            "cost_usd": None,
            "cost_available": False,
        }

    def number(name: str) -> int | float | None:
        value = getattr(client, name, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            return None
        return value

    try:
        backend_raw = str(getattr(client, "backend", "configured"))
    except Exception:
        backend_raw = "configured"
    backend = backend_raw if backend_raw in {
        "configured", "local_feature_hash", "openai_compatible",
    } else "configured"
    try:
        quality_raw = str(getattr(client, "quality", "unknown"))
    except Exception:
        quality_raw = "unknown"
    quality = quality_raw if quality_raw in {"lexical", "semantic"} else "unknown"
    network_free_raw = getattr(client, "network_free", None)
    network_free = (
        network_free_raw if isinstance(network_free_raw, bool) else None
    )
    try:
        from hymem.dreaming.aggregation_material import (
            embedding_execution_identity,
        )

        binding, model, resolved_dimension = embedding_execution_identity(client)
        dimension = resolved_dimension
        identity_exact = bool(binding["identity_exact"])
        reuse_scope = str(binding["reuse_scope"])
        identity_available = bool(model and dimension)
    except Exception:
        model = None
        dimension = None
        identity_exact = None
        reuse_scope = None
        identity_available = False
    calls = number("call_count")
    attempts = number("request_attempts")
    successes = number("successful_responses")
    inputs = number("input_count")
    input_chars = number("input_characters")
    latency = number("total_latency_s")
    prompt = number("prompt_tokens")
    total = number("total_tokens")
    tokens_available = bool(
        getattr(client, "token_usage_available", False)
        and prompt is not None and total is not None
    )
    cost = number("cost_usd")
    return {
        "configured": bool(configured),
        "backend": backend,
        "quality": quality,
        "network_free": network_free,
        "model": model,
        "dimension": int(dimension) if dimension is not None else None,
        "identity_available": identity_available,
        "identity_exact": identity_exact,
        "reuse_scope": reuse_scope,
        "calls": int(calls) if calls is not None else None,
        "calls_available": calls is not None,
        "request_attempts": int(attempts) if attempts is not None else None,
        "request_attempts_available": attempts is not None,
        "successful_responses": (
            int(successes) if successes is not None else None
        ),
        "successful_responses_available": successes is not None,
        "input_count": int(inputs) if inputs is not None else None,
        "input_count_available": inputs is not None,
        "input_characters": (
            int(input_chars) if input_chars is not None else None
        ),
        "input_characters_available": input_chars is not None,
        "prompt_tokens": prompt if tokens_available else None,
        "total_tokens": total if tokens_available else None,
        "provider_token_usage_available": tokens_available,
        "latency_s": latency,
        "latency_available": latency is not None,
        "cost_usd": cost,
        "cost_available": cost is not None,
    }


def aggregate_embedding_usage_snapshots(
    snapshots: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate embedding work only where every instance measured a field."""

    rows = [dict(snapshot) for snapshot in snapshots]
    if not rows:
        return embedding_usage_snapshot(None, configured=False)

    def total(field: str, availability: str) -> tuple[int | float | None, bool]:
        values = [row.get(field) for row in rows]
        available = all(row.get(availability) is True for row in rows)
        if not available or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in values
        ):
            return None, False
        return sum(values), True

    calls, calls_ok = total("calls", "calls_available")
    attempts, attempts_ok = total("request_attempts", "request_attempts_available")
    successes, successes_ok = total(
        "successful_responses", "successful_responses_available"
    )
    inputs, inputs_ok = total("input_count", "input_count_available")
    input_chars, input_chars_ok = total(
        "input_characters", "input_characters_available"
    )
    prompt, prompt_ok = total(
        "prompt_tokens", "provider_token_usage_available"
    )
    provider_total, provider_total_ok = total(
        "total_tokens", "provider_token_usage_available"
    )
    latency, latency_ok = total("latency_s", "latency_available")
    cost, cost_ok = total("cost_usd", "cost_available")
    identities = {
        (
            row.get("backend"), row.get("quality"), row.get("network_free"),
            row.get("model"), row.get("dimension"), row.get("identity_exact"),
            row.get("reuse_scope"),
        )
        for row in rows
    }
    backend, quality, network_free, model, dimension, identity_exact, reuse_scope = (
        next(iter(identities)) if len(identities) == 1
        else ("mixed", "mixed", None, None, None, None, None)
    )
    identity_consistent = bool(
        len(identities) == 1
        and all(row.get("identity_available") is True for row in rows)
    )
    tokens_ok = prompt_ok and provider_total_ok
    return {
        "configured": all(row.get("configured") is True for row in rows),
        "backend": backend,
        "quality": quality,
        "network_free": network_free,
        "model": model,
        "dimension": dimension,
        "identity_exact": identity_exact,
        "reuse_scope": reuse_scope,
        "identity_consistent": identity_consistent,
        "instances": len(rows),
        "calls": int(calls) if calls_ok else None,
        "calls_available": calls_ok,
        "request_attempts": int(attempts) if attempts_ok else None,
        "request_attempts_available": attempts_ok,
        "successful_responses": int(successes) if successes_ok else None,
        "successful_responses_available": successes_ok,
        "input_count": int(inputs) if inputs_ok else None,
        "input_count_available": inputs_ok,
        "input_characters": int(input_chars) if input_chars_ok else None,
        "input_characters_available": input_chars_ok,
        "prompt_tokens": prompt if tokens_ok else None,
        "total_tokens": provider_total if tokens_ok else None,
        "provider_token_usage_available": tokens_ok,
        "latency_s": latency if latency_ok else None,
        "latency_available": latency_ok,
        "cost_usd": cost if cost_ok else None,
        "cost_available": cost_ok,
    }


def write_immutable_artifact(
    path: str | os.PathLike[str], value: Mapping[str, Any]
) -> None:
    """Public exclusive-create helper for final run archives."""

    sanitized = sanitize_for_artifact(dict(value))
    if not isinstance(sanitized, dict):
        raise BenchmarkIntegrityError("artifact root must remain an object")
    _write_new_json(Path(path), sanitized)


def write_latest_pointer(
    path: str | os.PathLike[str], *, archive: Path, run_id: str,
    artifact_digest: str | None = None,
) -> None:
    """Write a small mutable pointer, never a second mutable result artifact."""

    pointer = {"archive": archive.name, "run_id": run_id}
    if artifact_digest is not None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", artifact_digest) is None:
            raise BenchmarkIntegrityError("artifact pointer digest is malformed")
        pointer["artifact_digest"] = artifact_digest
    _atomic_json(Path(path), pointer)


def prepare_checkpoint_artifact(
    ledger: AtomicCheckpoint,
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated artifact while its checkpoint lease is still open.

    Reserved lifecycle fields always come from the validated checkpoint. An
    adapter may attach benchmark-specific scores/diagnostics through ``payload``
    but cannot replace the manifest, denominator, or rows.
    """

    extra = dict(payload or {})
    reserved = {"manifest", "config", "models", "execution", "per_question"}
    collision = reserved & set(extra)
    if collision:
        raise BenchmarkIntegrityError(
            f"artifact payload overrides reserved fields: {sorted(collision)}"
        )
    snapshot = ledger.finalize()
    reconciled = ledger.reconcile()
    try:
        from .archive_evidence import checkpoint_attestation
    except ImportError:
        from archive_evidence import checkpoint_attestation
    artifact: dict[str, Any] = {
        **extra,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": ledger.manifest,
        "config": ledger.manifest["config"],
        "models": ledger.manifest["models"],
        "execution": {
            "counts": snapshot["counts"],
            "segments": snapshot["execution_segments"],
            # A host path is neither portable evidence nor needed for
            # recovery (the operator already supplied the checkpoint).  Bind
            # the archive to a minimal, independently reconcilable projection
            # of the finalized ledger, not an unavailable file's opaque hash.
            "checkpoint": checkpoint_attestation(snapshot, list(reconciled.rows)),
        },
        "per_question": list(reconciled.rows),
    }
    return artifact


def publish_checkpoint_artifact(
    ledger: AtomicCheckpoint,
    path: str | os.PathLike[str],
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize durable rows and publish, leaving resource ownership to caller.

    Production runners that require cleanup as a publication precondition use
    ``prepare_checkpoint_artifact`` plus
    ``publish_prepared_artifact_after_cleanup``.  This low-level helper remains
    for callers/tests that deliberately manage the checkpoint lifecycle around
    publication themselves.
    """

    artifact = prepare_checkpoint_artifact(ledger, payload=payload)
    write_immutable_artifact(path, artifact)
    return sanitize_for_artifact(artifact)


def publish_prepared_artifact_after_cleanup(
    path: str | os.PathLike[str],
    artifact: Mapping[str, Any],
    *,
    cleanup_actions: Sequence[tuple[str, Callable[[], object]]],
) -> dict[str, Any]:
    """Publish a prepared artifact only after all run owners close cleanly.

    Callers prepare while the checkpoint lease is live, transfer ownership of
    each cleanup action to this function, and retain the checkpoint file as the
    recovery surface.  A cleanup failure happens before exclusive creation, so
    no apparently successful archive can survive a failed teardown.
    """

    run_cleanup_actions(cleanup_actions)
    write_immutable_artifact(path, artifact)
    sanitized = sanitize_for_artifact(dict(artifact))
    if not isinstance(sanitized, dict):  # pragma: no cover - guarded on write
        raise BenchmarkIntegrityError("artifact root must remain an object")
    return sanitized


def export_checkpoint_without_recompute(
    checkpoint_path: str | os.PathLike[str],
    artifact_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Publish validated durable rows using the checkpoint's embedded identity.

    This recovery route intentionally does not hash or import the current
    adapter implementation. It cannot add derived scores, but preserves the
    expensive row evidence after a presentation/post-processing bug or code
    change, and performs no model calls.
    """

    source = Path(checkpoint_path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkIntegrityError(f"cannot read checkpoint {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BenchmarkIntegrityError("checkpoint root must be an object")
    manifest = raw.get("manifest")
    expected_ids = raw.get("expected_ids")
    if not isinstance(manifest, dict) or not isinstance(expected_ids, list):
        raise BenchmarkIntegrityError("checkpoint lacks embedded run identity")
    ledger = AtomicCheckpoint(
        source,
        manifest=manifest,
        expected_ids=expected_ids,
        resume=True,
        scored=bool(raw.get("scored", True)),
        verdict_key=str(raw.get("verdict_key", "correct")),
    )
    try:
        artifact = prepare_checkpoint_artifact(
            ledger,
            payload={
                "benchmark": manifest.get("benchmark"),
                "version": "strict-checkpoint-recovery-v1",
                "recovery_disclosure": (
                    "Published from the checkpoint's embedded manifest and durable "
                    "rows with zero model calls; derived adapter diagnostics omitted."
                ),
            },
        )
    except BaseException as exc:
        run_cleanup_actions(
            [("checkpoint_close", ledger.close)],
            primary_exception=exc,
        )
        raise
    return publish_prepared_artifact_after_cleanup(
        artifact_path,
        artifact,
        cleanup_actions=[("checkpoint_close", ledger.close)],
    )


def read_artifact_or_pointer(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read an artifact, deliberately dereferencing the small latest pointer.

    Pointers may only name a sibling file (never an absolute/traversing path),
    and their run id must agree with the target manifest. This keeps legacy
    consumers convenient without confusing a mutable pointer for evidence.
    """

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkIntegrityError(f"cannot read artifact {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkIntegrityError("artifact root must be an object")
    if set(value) in (
        {"archive", "run_id"},
        {"archive", "run_id", "artifact_digest"},
    ):
        archive = value.get("archive")
        if (
            not isinstance(archive, str)
            or not archive
            or Path(archive).is_absolute()
            or Path(archive).name != archive
        ):
            raise BenchmarkIntegrityError("artifact pointer target is unsafe")
        target = source.parent / archive
        try:
            target_value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkIntegrityError(
                f"cannot dereference artifact pointer {source}: {exc}"
            ) from exc
        if not isinstance(target_value, dict):
            raise BenchmarkIntegrityError("pointer target root must be an object")
        target_manifest = target_value.get("manifest")
        if not isinstance(target_manifest, dict):
            raise BenchmarkIntegrityError("pointer target lacks a manifest")
        target_run_id = target_manifest.get("run_id")
        if (
            not isinstance(target_run_id, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", target_run_id)
            or target_run_id != content_hash({
                key: item for key, item in target_manifest.items()
                if key != "run_id"
            })
        ):
            raise BenchmarkIntegrityError("pointer target manifest identity is invalid")
        pointer_run_id = value.get("run_id")
        if not isinstance(pointer_run_id, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", pointer_run_id
        ):
            raise BenchmarkIntegrityError("artifact pointer run identity is invalid")
        if target_run_id != pointer_run_id:
            raise BenchmarkIntegrityError("artifact pointer run identity mismatch")
        pointer_digest = value.get("artifact_digest")
        if pointer_digest is not None and (
            not isinstance(pointer_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", pointer_digest) is None
            or pointer_digest != content_hash(target_value)
        ):
            raise BenchmarkIntegrityError("artifact pointer digest mismatch")
        return target_value
    return value


def deterministic_smoke(root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Run the strict artifact lifecycle with no model, dataset, or network.

    CI calls this path to exercise a crash/resume boundary, strict failure
    accounting, and deterministic run identity.  It deliberately retains one
    failed prediction in the final denominator.
    """

    owned = tempfile.TemporaryDirectory(prefix="hymem-benchmark-smoke-") \
        if root is None else None
    work = Path(owned.name if owned is not None else root)
    work.mkdir(parents=True, exist_ok=True)
    try:
        ids = ("smoke-1", "smoke-2", "smoke-3")
        manifest = build_manifest(
            benchmark="strict-smoke",
            code_sha256=content_hash("fixed smoke code"),
            data_sha256=content_hash("fixed smoke data"),
            config={"mode": "offline", "label_free_answer_path": True},
            models={"reader": None, "judge": None},
            seed=0,
            expected_ids=ids,
        protocol_split="full",
        )
        path = work / "smoke.checkpoint.json"
        ledger = AtomicCheckpoint(
            path, manifest=manifest, expected_ids=ids, resume=False
        )
        ledger.record(
            "smoke-1", row={"question_id": "smoke-1", "correct": True}
        )
        # This is the simulated crash boundary: reconstruct exclusively from
        # durable state, then finish the pending ids.
        ledger.close()
        resumed = AtomicCheckpoint(
            path, manifest=manifest, expected_ids=ids, resume=True
        )
        try:
            resumed.record(
                "smoke-2", row={"question_id": "smoke-2", "correct": False}
            )
            resumed.record("smoke-3", row=None, failure="synthetic transport failure")
            state = resumed.finalize()
            result = resumed.reconcile()
            return {
                "run_id": manifest["run_id"],
                "counts": state["counts"],
                "accuracy": strict_accuracy(result.rows),
                "failure_ids": list(result.failure_ids),
            }
        finally:
            resumed.close()
    finally:
        if owned is not None:
            owned.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HyMem benchmark-integrity deterministic smoke"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="run the offline strictness smoke"
    )
    parser.add_argument("--export-checkpoint", metavar="FILE")
    parser.add_argument("--artifact", metavar="FILE")
    args = parser.parse_args()
    if args.export_checkpoint:
        if not args.artifact:
            parser.error("--export-checkpoint requires --artifact")
        output = export_checkpoint_without_recompute(
            args.export_checkpoint, args.artifact
        )
        print(json.dumps({
            "artifact": args.artifact,
            "run_id": output["manifest"]["run_id"],
            "model_calls": 0,
        }, sort_keys=True))
    elif args.smoke:
        print(json.dumps(deterministic_smoke(), sort_keys=True))
    else:
        parser.error("pass --smoke or --export-checkpoint FILE --artifact FILE")


if __name__ == "__main__":
    main()
