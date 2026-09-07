"""Cooperative absolute deadlines for bounded benchmark indexing work.

The deadline is deliberately an absolute monotonic timestamp.  Passing the
same object through a whole convergence wave prevents each provider retry or
dream phase from accidentally receiving a fresh timeout budget.

This is cooperative rather than pre-emptive: a third-party provider that
ignores the request timeout cannot be killed safely in-process.  Callers still
detect expiry immediately when that provider returns and must not publish its
late result.  The shipped network clients also cap every request timeout to the
remaining wave budget, so a conforming transport returns near the boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import math
import time
from typing import Callable, Iterator, TypeVar

from hymem.contrib.implementation_identity import import_time_source_sha256


class DeadlineExceeded(BaseException):
    """Internal control-flow signal for an expired cooperative deadline.

    It intentionally derives directly from :class:`BaseException`: dreaming
    has several best-effort ``except Exception`` recovery paths which persist
    retry/quarantine state.  A timeout is neither a provider-quality failure
    nor an extraction attempt to publish, and therefore must pass through
    those handlers unchanged.  Benchmark convergence catches this type at its
    outer boundary and turns it into bounded failure evidence.
    """


Clock = Callable[[], float]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class MonotonicDeadline:
    """One absolute monotonic deadline shared by an operation tree."""

    expires_at: float
    clock: Clock = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.expires_at, bool)
            or not isinstance(self.expires_at, (int, float))
            or not math.isfinite(float(self.expires_at))
        ):
            raise ValueError("deadline must be a finite monotonic timestamp")
        if not callable(self.clock):
            raise TypeError("deadline clock must be callable")

    @classmethod
    def after(
        cls, timeout_s: float, *, clock: Clock = time.monotonic,
    ) -> "MonotonicDeadline":
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError("deadline timeout must be positive and finite")
        return cls(float(clock()) + float(timeout_s), clock=clock)

    def remaining(self) -> float:
        """Return remaining seconds without inventing time after expiry."""

        return max(0.0, float(self.expires_at) - float(self.clock()))

    @property
    def expired(self) -> bool:
        return float(self.clock()) >= float(self.expires_at)

    def check(self) -> None:
        if self.expired:
            raise DeadlineExceeded("indexing deadline exceeded")

    def cap_timeout(self, configured_timeout_s: float) -> float:
        """Cap one provider attempt to the exact remaining operation budget."""

        if (
            isinstance(configured_timeout_s, bool)
            or not isinstance(configured_timeout_s, (int, float))
            or not math.isfinite(float(configured_timeout_s))
            or float(configured_timeout_s) <= 0.0
        ):
            raise ValueError("provider timeout must be positive and finite")
        # Read the clock exactly once.  A second read could cross the boundary
        # after ``check`` and accidentally hand a transport ``timeout=0`` while
        # still starting the request.
        remaining = float(self.expires_at) - float(self.clock())
        if remaining <= 0.0:
            raise DeadlineExceeded("indexing deadline exceeded")
        # Do not impose an arbitrary floor: even a sub-millisecond remainder
        # is the honest transport cap.
        return min(float(configured_timeout_s), remaining)

    def sleep(
        self, delay_s: float, *, sleeper: Sleeper = time.sleep,
    ) -> None:
        """Sleep at most the remaining budget, then re-check before retrying."""

        remaining = float(self.expires_at) - float(self.clock())
        if remaining <= 0.0:
            raise DeadlineExceeded("indexing deadline exceeded")
        requested = max(0.0, float(delay_s))
        sleeper(min(requested, remaining))
        self.check()


_CURRENT_DEADLINE: ContextVar[MonotonicDeadline | None] = ContextVar(
    "hymem_current_deadline", default=None,
)


def current_deadline() -> MonotonicDeadline | None:
    return _CURRENT_DEADLINE.get()


def check_current_deadline() -> None:
    deadline = current_deadline()
    if deadline is not None:
        deadline.check()


@contextmanager
def use_deadline(
    deadline: MonotonicDeadline | None,
) -> Iterator[MonotonicDeadline | None]:
    """Install *deadline* lexically for provider clients and nested helpers."""

    if deadline is not None and not isinstance(deadline, MonotonicDeadline):
        raise TypeError("deadline must be a MonotonicDeadline or None")
    token = _CURRENT_DEADLINE.set(deadline)
    try:
        yield deadline
    finally:
        _CURRENT_DEADLINE.reset(token)


T = TypeVar("T")


class DeadlineBoundLLMClient:
    """Transparent client proxy that rejects late custom-provider results."""

    def __init__(self, inner: T, deadline: MonotonicDeadline) -> None:
        self._inner = inner
        self._deadline = deadline
        from hymem.extraction.producer import _register_phase1_producer_proxy

        _register_phase1_producer_proxy(self, inner)

    def __getattr__(self, name: str):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        return getattr(_verified_phase1_proxy_delegate(self), name)

    def complete(self, request):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        inner = _verified_phase1_proxy_delegate(self)
        self._deadline.check()
        with use_deadline(self._deadline):
            result = inner.complete(request)
        self._deadline.check()
        return result


class DeadlineBoundEmbeddingClient:
    """Transparent embedding proxy that rejects late custom-provider results."""

    def __init__(self, inner: T, deadline: MonotonicDeadline) -> None:
        self._inner = inner
        self._deadline = deadline
        from hymem.extraction.producer import _register_phase1_producer_proxy

        _register_phase1_producer_proxy(self, inner)

    def __getattr__(self, name: str):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        return getattr(_verified_phase1_proxy_delegate(self), name)

    def embed(self, texts):
        from hymem.extraction.producer import _verified_phase1_proxy_delegate

        inner = _verified_phase1_proxy_delegate(self)
        self._deadline.check()
        with use_deadline(self._deadline):
            result = inner.embed(texts)
        self._deadline.check()
        return result


_DEADLINE_PROXY_DISPATCH_NAMES = (
    "embed", "complete", "model", "dim", "backend", "quality",
    "network_free", "close", "__getattribute__", "__getattr__",
    "_embed_serialized", "_validated_vector",
)
_DEADLINE_PROXY_ORIGINAL_GUARDS = {
    DeadlineBoundLLMClient: tuple(
        inspect.getattr_static(DeadlineBoundLLMClient, name, None)
        for name in _DEADLINE_PROXY_DISPATCH_NAMES
    ),
    DeadlineBoundEmbeddingClient: tuple(
        inspect.getattr_static(DeadlineBoundEmbeddingClient, name, None)
        for name in _DEADLINE_PROXY_DISPATCH_NAMES
    ),
}

_DEADLINE_SUPPORT_NAMES = (
    "__post_init__", "after", "remaining", "expired", "check",
    "cap_timeout", "sleep", "__getattribute__", "__setattr__",
)
_DEADLINE_SUPPORT_ORIGINAL_GUARD = (
    tuple(
        inspect.getattr_static(MonotonicDeadline, name, None)
        for name in _DEADLINE_SUPPORT_NAMES
    ),
    use_deadline,
)


def deadline_proxy_support_integrity() -> bool:
    """Verify helpers through which maintained deadline proxies dispatch."""

    return _DEADLINE_SUPPORT_ORIGINAL_GUARD == (
        tuple(
            inspect.getattr_static(MonotonicDeadline, name, None)
            for name in _DEADLINE_SUPPORT_NAMES
        ),
        use_deadline,
    )


DEADLINE_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)
