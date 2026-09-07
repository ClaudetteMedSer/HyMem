"""Bounded exponential backoff for transient external-API failures.

Used only by the contrib network clients (OpenAI-compatible LLM and embedding
endpoints) — the host may wire its own clients with its own resilience policy,
and the read/augment hot path stays retry-free. Keep the attempt count low so a
genuinely-down backend fails the dream cycle promptly instead of hanging.
"""
from __future__ import annotations

from hymem.contrib.implementation_identity import import_time_source_sha256

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)

import logging
import time
from typing import Callable, TypeVar

from hymem.deadline import current_deadline

log = logging.getLogger("hymem.extraction.retry")

T = TypeVar("T")
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY_SECONDS = 0.5
DEFAULT_RETRY_MAX_DELAY_SECONDS = 8.0


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY_SECONDS,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY_SECONDS,
    label: str = "external call",
) -> T:
    """Call *fn* with exponential backoff, re-raising the last error if every
    attempt fails. Delay between attempt *i* and *i+1* is
    ``min(base_delay * 2**i, max_delay)``."""
    last_exc: Exception | None = None
    deadline = current_deadline()
    for attempt in range(attempts):
        if deadline is not None:
            deadline.check()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised after the loop
            last_exc = exc
            # Do not turn an already-expired operation into another provider
            # request (or even another complete backoff).  DeadlineExceeded is
            # a BaseException and therefore also passes straight through this
            # provider-error handler when raised by the request itself.
            if deadline is not None:
                deadline.check()
            if attempt == attempts - 1:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            log.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                label, attempt + 1, attempts, exc, delay,
            )
            if deadline is None:
                time.sleep(delay)
            else:
                deadline.sleep(delay, sleeper=time.sleep)
    assert last_exc is not None  # loop ran at least once
    raise last_exc


_RETRY_ORIGINAL_HELPER_GUARD = (current_deadline, with_retry)


def retry_support_integrity() -> bool:
    """Whether deadline/retry dispatch still matches the loaded module."""

    return (current_deadline, with_retry) == _RETRY_ORIGINAL_HELPER_GUARD
