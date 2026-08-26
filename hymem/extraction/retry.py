"""Bounded exponential backoff for transient external-API failures.

Used only by the contrib network clients (OpenAI-compatible LLM and embedding
endpoints) — the host may wire its own clients with its own resilience policy,
and the read/augment hot path stays retry-free. Keep the attempt count low so a
genuinely-down backend fails the dream cycle promptly instead of hanging.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

log = logging.getLogger("hymem.extraction.retry")

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    label: str = "external call",
) -> T:
    """Call *fn* with exponential backoff, re-raising the last error if every
    attempt fails. Delay between attempt *i* and *i+1* is
    ``min(base_delay * 2**i, max_delay)``."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised after the loop
            last_exc = exc
            if attempt == attempts - 1:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            log.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                label, attempt + 1, attempts, exc, delay,
            )
            time.sleep(delay)
    assert last_exc is not None  # loop ran at least once
    raise last_exc
