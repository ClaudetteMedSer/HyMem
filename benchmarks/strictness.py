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

import hashlib
import json
import os
import argparse
import math
import tempfile
import threading
import re
import time
try:
    import fcntl
except ImportError:  # pragma: no cover - benchmark runners are POSIX today
    fcntl = None
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STRICT_PROTOCOL_VERSION = "hymem-benchmark-strict-v1"
CHECKPOINT_VERSION = "hymem-benchmark-checkpoint-v1"
CALIBRATION_VERSION = "hymem-benchmark-calibration-v1"
_CODE_SUFFIXES = {".py", ".sql", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}


class BenchmarkIntegrityError(ValueError):
    """The benchmark evidence is incomplete, ambiguous, or inconsistent."""


class IndexingConvergenceError(BenchmarkIntegrityError):
    """A benchmark memory build did not reach a complete, healthy state."""

    def __init__(self, message: str, summary: Mapping[str, Any]):
        super().__init__(message)
        self.summary = dict(summary)


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
_EVIDENCE_TEXT_KEYS = frozenset({
    "answer", "content", "context", "gold", "gold_text", "hypothesis",
    "ideal_answer", "ideal_response", "prediction", "question", "response",
    "rubric", "summary", "text",
})


def _is_secret_key(value: str) -> bool:
    normalized = value.casefold().replace("-", "_")
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
        redacted_query: list[tuple[str, str]] = []
        query_redacted = False
        for key, item in parse_qsl(parsed.query, keep_blank_values=True):
            if _is_secret_key(key):
                redacted_query.append((key, "<redacted>"))
                query_redacted = True
            else:
                redacted_query.append((key, item))
        has_userinfo = parsed.username is not None or parsed.password is not None
        fragment_redacted = bool(parsed.fragment)
        if not has_userinfo and not query_redacted and not fragment_redacted:
            return value
        clean = urlunsplit((
            parsed.scheme,
            netloc,
            parsed.path,
            urlencode(redacted_query, doseq=True),
            "",
        ))
        result: dict[str, Any] = {
            "url": clean,
            "credentials_redacted": True,
        }
        return result
    except (TypeError, ValueError):
        return value


def _sanitize_failure_text(value: object) -> str:
    """Scrub credentials from exception text while keeping it human-readable."""

    text = str(value)

    def replace_url(match: re.Match[str]) -> str:
        sanitized = _sanitized_url(match.group(0))
        return sanitized if isinstance(sanitized, str) else str(sanitized["url"])

    text = _URL_IN_TEXT_RE.sub(replace_url, text)
    # Header values may contain a scheme plus credential, or several cookie
    # pairs separated by semicolons. Redacting only the first token leaves the
    # credential/tail behind, so treat the complete header line as opaque.
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


def sanitize_for_artifact(value: Any, *, key_hint: str = "") -> Any:
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
        return {
            str(key): sanitize_for_artifact(item, key_hint=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_artifact(item, key_hint=key_hint) for item in value]
    if isinstance(value, str):
        if any(part in key_folded for part in ("error", "failure", "exception")):
            return _sanitize_failure_text(value)
        if key_folded in _EVIDENCE_TEXT_KEYS:
            return value
        # Configuration and lifecycle strings can carry credential-shaped
        # headers even under an innocuous key (for example {"detail":
        # "Authorization: Bearer ..."}). Evidence-bearing text keys above
        # remain byte-for-byte untouched for rejudging.
        scrubbed = _sanitize_failure_text(value)
        if scrubbed != value:
            return scrubbed
        sanitized = _sanitize_incidental_url_text(value)
        if sanitized != value:
            return sanitized
        return value
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


def converge_indexing(
    dream,
    *,
    status=None,
    max_cycles: int,
    timeout_s: float,
    require_healthy: bool = True,
) -> dict[str, Any]:
    """Run bounded dream cycles until the durable extraction backlog is empty.

    ``dream`` returns a DreamReport-like dataclass or mapping. ``status`` is an
    optional read-only durable backlog callback. A single non-exhausted report
    is insufficient when durable pending work remains. Quarantined work makes
    canonical/healthy completion fail loudly rather than masquerading as a
    completed index.
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

    def normalize_report(report: object) -> dict[str, Any]:
        if isinstance(report, Mapping):
            return dict(report)
        if is_dataclass(report) and not isinstance(report, type):
            return dataclass_identity(report)
        raise BenchmarkIntegrityError("dream returned a malformed report")

    started = time.monotonic()
    reports: list[dict[str, Any]] = []
    latest_status: dict[str, Any] = {}

    def summary(*, complete: bool, reason: str | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - started
        quarantined = {
            key: value for key, value in latest_status.items()
            if "quarantined" in key
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        }
        return {
            "cycles": len(reports),
            "max_cycles": max_cycles,
            "timeout_s": float(timeout_s),
            "elapsed_s": elapsed,
            "complete": bool(complete),
            "healthy": bool(complete and not quarantined),
            "failure_reason": reason,
            "reports": reports,
            "final_status": latest_status,
            "quarantined": quarantined,
        }

    for _cycle in range(max_cycles):
        if time.monotonic() - started >= timeout_s:
            current = summary(complete=False, reason="timeout_before_cycle")
            raise IndexingConvergenceError(
                "memory indexing did not converge before its timeout", current
            )
        try:
            report_obj = dream()
            report = normalize_report(report_obj)
            reports.append(report)
            if status is not None:
                status_obj = status()
                if not isinstance(status_obj, Mapping):
                    raise BenchmarkIntegrityError(
                        "indexing status callback returned a malformed value"
                    )
                latest_status = dict(status_obj)
        except Exception as exc:
            current = summary(
                complete=False,
                reason=f"cycle_exception: {type(exc).__name__}: {exc}",
            )
            raise IndexingConvergenceError(
                "memory indexing cycle failed", current
            ) from exc
        elapsed = time.monotonic() - started
        pending_values = {
            key: value for key, value in latest_status.items()
            if key.startswith("pending_")
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in pending_values.values()
        ):
            current = summary(complete=False, reason="malformed_pending_backlog")
            raise IndexingConvergenceError(
                "memory indexing status has malformed pending backlog", current
            )
        pending = sum(pending_values.values())
        exhausted = report.get("budget_exhausted") is True
        skipped_locked = report.get("skipped_locked") is True
        complete = not exhausted and not skipped_locked and pending == 0
        if complete:
            current = summary(complete=True)
            if require_healthy and not current["healthy"]:
                current["failure_reason"] = "quarantined_extraction"
                raise IndexingConvergenceError(
                    "memory indexing completed with quarantined extraction",
                    current,
                )
            if elapsed > timeout_s:
                current["complete"] = False
                current["healthy"] = False
                current["failure_reason"] = "timeout_after_cycle"
                raise IndexingConvergenceError(
                    "memory indexing exceeded its timeout", current
                )
            return current

    current = summary(complete=False, reason="max_cycles_exhausted")
    raise IndexingConvergenceError(
        "memory indexing did not converge within max_cycles", current
    )


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


def code_hash(paths: Iterable[str | os.PathLike[str]], *, root: Path) -> str:
    """Hash exact source bytes plus stable repo-relative path names.

    Directories are expanded recursively, excluding generated/cache/VCS files.
    The caller chooses the code surface; adapters pass their own file and the
    ``hymem`` package so a resume cannot cross an implementation change.
    """

    root = root.resolve()
    files: set[Path] = set()
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_dir():
            files.update(
                candidate for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and ".git" not in candidate.parts
                and candidate.suffix.casefold() in _CODE_SUFFIXES
            )
        elif path.is_file():
            files.add(path)
        else:
            raise BenchmarkIntegrityError(f"code path does not exist: {path}")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: str(item)):
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = path.as_posix()
        payload = path.read_bytes()
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
    config_obj = sanitize_for_artifact(dict(config))
    model_obj = sanitize_for_artifact(dict(models))
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
        "config_hash": content_hash(sanitize_for_artifact(dict(config))),
        "model_hash": content_hash(sanitize_for_artifact(dict(models))),
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
    config_obj = sanitize_for_artifact(dict(config))
    model_obj = sanitize_for_artifact(dict(models))
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
        self.path = Path(path)
        self.manifest = dict(manifest)
        self.expected_ids = validate_ids(expected_ids)
        self.retry_failures = bool(retry_failures)
        self.scored = bool(scored)
        if not isinstance(verdict_key, str) or not verdict_key.strip():
            raise BenchmarkIntegrityError("checkpoint verdict key must be non-empty")
        self.verdict_key = verdict_key.strip()
        self._lock = threading.RLock()
        self._lease: _CheckpointLease | None = None
        if self.manifest.get("run_id") != content_hash(
            {k: v for k, v in self.manifest.items() if k != "run_id"}
        ):
            raise BenchmarkIntegrityError("manifest run_id is invalid")
        self._lease = _CheckpointLease(self.path)
        try:
            if resume:
                self._state = self._load()
                self._validate_state()
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

    def _validate_state(self) -> None:
        state = self._state
        if state.get("schema") != CHECKPOINT_VERSION:
            raise BenchmarkIntegrityError("unsupported checkpoint schema")
        if state.get("run_id") != self.manifest.get("run_id"):
            raise BenchmarkIntegrityError("checkpoint run identity mismatch")
        if state.get("manifest") != self.manifest:
            raise BenchmarkIntegrityError("checkpoint manifest was modified")
        if state.get("expected_ids") != list(self.expected_ids):
            raise BenchmarkIntegrityError("checkpoint expected-id set/order mismatch")
        if state.get("scored", True) is not self.scored:
            raise BenchmarkIntegrityError("checkpoint scored/diagnostic mode mismatch")
        if state.get("verdict_key", "correct") != self.verdict_key:
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
                _atomic_json(self.path, self._state)

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
                    entry["failure"] = _sanitize_failure_text(
                        copied.get("benchmark_failure")
                        or "no valid prediction verdict"
                    )
            else:
                entry = {
                    "status": "failed",
                    "attempts": attempts,
                    "failure": _sanitize_failure_text(failure),
                    "row": {
                        "question_id": item_id,
                        "correct": False,
                        self.verdict_key: False,
                        "benchmark_failure": _sanitize_failure_text(failure),
                    },
                }
            history.append({
                "attempt": attempts,
                "status": entry["status"],
                "failure": entry.get("failure"),
                "row": dict(entry["row"]),
            })
            entry["attempt_history"] = history
            self._state["entries"][item_id] = entry
            if execution_segment is not None:
                self._upsert_execution_segment_locked(execution_segment)
            _atomic_json(self.path, self._state)

    def _upsert_execution_segment_locked(
        self, metrics: Mapping[str, Any]
    ) -> None:
        segment = dict(metrics)
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
            self._upsert_execution_segment_locked(
                {**dict(metrics), "segment_id": segment_id}
            )
            _atomic_json(self.path, self._state)

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

    def finalize(self) -> dict[str, Any]:
        """Mark complete and return an artifact-ready checkpoint snapshot."""

        with self._lock:
            self._require_open()
            result = self.reconcile()
            self._state["status"] = "complete"
            self._state["counts"] = {
                "expected": result.expected,
                "attempted": result.attempted,
                "unique_attempted": result.attempted,
                "total_attempts": sum(
                    int(entry["attempts"])
                    for entry in self._state["entries"].values()
                ),
                "completed": result.completed,
                "failed": result.failed,
                "missing": result.missing,
            }
            self._state["failure_ids"] = list(result.failure_ids)
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

    backend = str(getattr(client, "backend", "configured"))
    quality = str(getattr(client, "quality", "unknown"))
    network_free_raw = getattr(client, "network_free", None)
    network_free = (
        network_free_raw if isinstance(network_free_raw, bool) else None
    )
    try:
        model_raw = getattr(client, "model", None)
    except Exception:
        model_raw = None
    model = model_raw if isinstance(model_raw, str) and model_raw else None
    dimension = number("dim")
    if dimension is not None and not float(dimension).is_integer():
        dimension = None
    identity_available = bool(model is not None and dimension is not None)
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
            row.get("model"), row.get("dimension"),
        )
        for row in rows
    }
    backend, quality, network_free, model, dimension = (
        next(iter(identities)) if len(identities) == 1
        else ("mixed", "mixed", None, None, None)
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


def publish_checkpoint_artifact(
    ledger: AtomicCheckpoint,
    path: str | os.PathLike[str],
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize durable rows and publish an artifact before presentation work.

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
    artifact: dict[str, Any] = {
        **extra,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": ledger.manifest,
        "config": ledger.manifest["config"],
        "models": ledger.manifest["models"],
        "execution": {
            "counts": snapshot["counts"],
            "segments": snapshot["execution_segments"],
            "checkpoint": str(ledger.path),
        },
        "per_question": list(reconciled.rows),
    }
    write_immutable_artifact(path, artifact)
    return sanitize_for_artifact(artifact)


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
        return publish_checkpoint_artifact(
            ledger,
            artifact_path,
            payload={
                "benchmark": manifest.get("benchmark"),
                "version": "strict-checkpoint-recovery-v1",
                "recovery_disclosure": (
                    "Published from the checkpoint's embedded manifest and durable "
                    "rows with zero model calls; derived adapter diagnostics omitted."
                ),
            },
        )
    finally:
        ledger.close()


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
