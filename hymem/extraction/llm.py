from __future__ import annotations

from hymem.contrib.implementation_identity import import_time_source_sha256

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)

from collections.abc import Iterator, MutableMapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import threading
from typing import Protocol


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    response_format: str = "json"  # "json" | "text"
    max_tokens: int = 1024
    temperature: float = 0.0


class LLMClient(Protocol):
    """Hermes wires whatever local LLM it wants behind this interface."""

    def complete(self, request: LLMRequest) -> str: ...


class ProviderAttemptTracker:
    """Exact request-attempt count for one caller-owned completion scope.

    LLM clients that perform internal retries may optionally expose
    ``track_provider_attempts()`` and return a context manager yielding this
    object. Every provider request started while the scope is active must be
    recorded, including requests that raise. The scope is local to the current
    execution context: overlapping calls on a shared client must not increment
    one another's trackers.

    ``request_attempts`` remains the cumulative, process-wide observability
    counter. This tracker is deliberately separate so attribution never relies
    on subtracting two shared global snapshots.
    """

    __slots__ = ("_attempts",)

    def __init__(self) -> None:
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    def _record(self) -> None:
        """Record one request; reserved for the client implementing the scope."""
        self._attempts += 1


class ScopedProviderAttemptClient(Protocol):
    """Optional extension for concurrency-safe provider-attempt attribution."""

    def track_provider_attempts(
        self,
    ) -> AbstractContextManager[ProviderAttemptTracker]: ...


@dataclass
class ProviderAttemptMeasurement:
    """Result populated by :func:`measure_provider_attempts` on scope exit."""

    attempts: int = 0
    exact: bool = False


def _cumulative_request_attempts(client: LLMClient) -> int | None:
    """Read a legacy non-negative cumulative attempt counter, if available."""
    try:
        value = getattr(client, "request_attempts", None)
    except Exception:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _scoped_attempts(tracker: object) -> int | None:
    try:
        value = getattr(tracker, "attempts", None)
    except Exception:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


@contextmanager
def measure_provider_attempts(
    client: LLMClient,
) -> Iterator[ProviderAttemptMeasurement]:
    """Measure provider requests made by one ``complete()`` invocation.

    A client's optional ``track_provider_attempts()`` scope is authoritative
    and safe when the client is shared by concurrent dream/query workers.
    Legacy custom clients that only expose a cumulative ``request_attempts``
    counter retain their serial delta behavior. Clients exposing neither (or
    an unchanged/invalid cumulative counter) are charged one request per
    logical call, so a raised completion is never silently counted as zero.
    """
    measurement = ProviderAttemptMeasurement()
    attempts_before = _cumulative_request_attempts(client)
    tracker: object = None
    scope: AbstractContextManager[object] | None = None

    try:
        scope_factory = getattr(client, "track_provider_attempts", None)
    except Exception:
        scope_factory = None
    if callable(scope_factory):
        try:
            scope = scope_factory()
        except Exception:
            # Attempt telemetry is optional observability. A malformed custom
            # implementation must not prevent the completion itself; the
            # cumulative/fallback path below remains conservative.
            scope = None

    entered = False
    scope_exact = False
    if scope is not None:
        try:
            tracker = scope.__enter__()
            entered = True
            scope_exact = True
        except Exception:
            # Scoped telemetry is optional. A broken ``__enter__`` must not
            # turn an otherwise valid provider invocation into a call failure.
            tracker = None

    raised = False
    try:
        yield measurement
    except BaseException as exc:
        raised = True
        if entered:
            try:
                # Attempt telemetry may neither suppress nor replace the
                # provider exception raised by the measured invocation.
                scope.__exit__(type(exc), exc, exc.__traceback__)
            except BaseException:
                scope_exact = False
        raise
    else:
        if entered:
            try:
                # Exit-time telemetry failures are observability failures, not
                # provider call failures. Fall back conservatively below.
                scope.__exit__(None, None, None)
            except Exception:
                scope_exact = False
    finally:
        # Establish the invariant before consulting any optional telemetry so
        # even a pathological getter cannot leave a completed call at zero.
        measurement.attempts = 1
        measurement.exact = False
        try:
            scoped = _scoped_attempts(tracker)
            if scoped is not None:
                measurement.attempts = scoped
                measurement.exact = scope_exact
            else:
                attempts_after = _cumulative_request_attempts(client)
                if (
                    attempts_before is not None
                    and attempts_after is not None
                    and attempts_after > attempts_before
                ):
                    measurement.attempts = attempts_after - attempts_before
        except BaseException:
            # Nothing in optional telemetry may replace the provider's
            # original exception. On a successful body, genuine process-level
            # interrupts still retain their normal propagation semantics.
            if not raised:
                raise


class _VersionedStubFixtures(MutableMapping[str, str]):
    """Mutable test-friendly routes with a monotonic semantic revision."""

    __slots__ = ("_owner", "_values")

    def __init__(self, owner: "StubLLMClient", values: dict[str, str]) -> None:
        self._owner = owner
        self._values = dict(values)

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __setitem__(self, key: str, value: str) -> None:
        if type(key) is not str or type(value) is not str:
            raise TypeError("StubLLMClient fixtures must map exact strings")
        with self._owner._routing_lock:
            self._values[key] = value
            self._owner._routing_serial += 1

    def __delitem__(self, key: str) -> None:
        with self._owner._routing_lock:
            del self._values[key]
            self._owner._routing_serial += 1

    def clear(self) -> None:
        with self._owner._routing_lock:
            self._values.clear()
            self._owner._routing_serial += 1

    def pop(self, key: str, default=...):
        with self._owner._routing_lock:
            if default is ...:
                value = self._values.pop(key)
            else:
                value = self._values.pop(key, default)
            self._owner._routing_serial += 1
            return value

    def popitem(self):
        with self._owner._routing_lock:
            value = self._values.popitem()
            self._owner._routing_serial += 1
            return value

    def setdefault(self, key: str, default: str | None = None) -> str:
        if type(key) is not str or type(default) is not str:
            raise TypeError("StubLLMClient fixtures must map exact strings")
        with self._owner._routing_lock:
            if key not in self._values:
                self._values[key] = default
                self._owner._routing_serial += 1
            return self._values[key]

    def update(self, *args, **kwargs) -> None:
        incoming = dict(*args, **kwargs)
        if any(type(key) is not str or type(value) is not str
               for key, value in incoming.items()):
            raise TypeError("StubLLMClient fixtures must map exact strings")
        with self._owner._routing_lock:
            self._values.update(incoming)
            self._owner._routing_serial += 1


class StubLLMClient:
    """Returns canned responses keyed by prompt substring.

    Fixture keys are matched against the concatenation of system + user, so tests
    can route on either. `default` is returned for any unmatched prompt;
    if `default` is None, unmatched prompts raise so missing fixtures are caught
    loudly.
    """

    def __init__(
        self,
        fixtures: dict[str, str] | None = None,
        default: str | None = None,
        calls: list[LLMRequest] | None = None,
    ) -> None:
        if fixtures is None:
            fixtures = {}
        if type(fixtures) is not dict or any(
            type(key) is not str or type(value) is not str
            for key, value in fixtures.items()
        ):
            raise TypeError("StubLLMClient fixtures must be an exact string mapping")
        if default is not None and type(default) is not str:
            raise TypeError("StubLLMClient default must be an exact string or None")
        if calls is not None and type(calls) is not list:
            raise TypeError("StubLLMClient calls must be an exact list")
        self._routing_lock = threading.RLock()
        self._routing_lock_identity = id(self._routing_lock)
        self._routing_serial = 0
        self._fixtures_view = _VersionedStubFixtures(self, fixtures)
        self._default = default
        self.calls = [] if calls is None else calls

    @property
    def fixtures(self) -> MutableMapping[str, str]:
        return self._fixtures_view

    @property
    def default(self) -> str | None:
        return self._default

    @default.setter
    def default(self, value: str | None) -> None:
        if value is not None and type(value) is not str:
            raise TypeError("StubLLMClient default must be an exact string or None")
        with self._routing_lock:
            self._default = value
            self._routing_serial += 1

    def phase1_producer_declaration(self):
        """Content-derived deterministic identity for the exact house stub.

        Subclasses may change routing/output behavior and must declare their
        own identity.  They deliberately fall back to process-instance reuse
        instead of inheriting an unsafe class-only or fixture-only claim.
        """

        if type(self) is not StubLLMClient:
            raise NotImplementedError(
                "StubLLMClient subclasses must declare their producer identity"
            )
        if (
            maintained_stub_llm_integrity is not _STUB_LLM_INTEGRITY_FUNCTION
            or not _STUB_LLM_INTEGRITY_FUNCTION(self)
        ):
            raise ValueError("maintained stub LLM integrity changed")
        from hymem.extraction.producer import Phase1ProducerDeclaration

        with self._routing_lock:
            fixture_state = {
                # First-match routing is insertion ordered.  Sorting this
                # mapping made two stubs with different overlapping-match
                # behavior share one exact producer identity.
                "fixtures": list(self._fixtures_view._values.items()),
                "default": self._default,
                "routing_serial": self._routing_serial,
            }
        fixture_payload = json.dumps(
            fixture_state,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return Phase1ProducerDeclaration(
            client_id="hymem.extraction.llm.StubLLMClient",
            implementation=EXTRACTION_IMPLEMENTATION_SHA256,
            model="deterministic-stub",
            endpoint=None,
            effective_request={
                "fixture_content_sha256": (
                    "sha256:" + hashlib.sha256(fixture_payload).hexdigest()
                ),
                "match_policy": "ordered-system-newline-user-substring-v1",
            },
            retry_policy={"owner": "none", "attempts": 1},
        )

    def aggregation_producer_declaration(self):
        """The house stub routes aggregation through the same exact backend."""

        return self.phase1_producer_declaration()

    def complete(self, request: LLMRequest) -> str:
        if type(self) is StubLLMClient and (
            maintained_stub_llm_integrity is not _STUB_LLM_INTEGRITY_FUNCTION
            or not _STUB_LLM_INTEGRITY_FUNCTION(self)
        ):
            raise RuntimeError("maintained stub LLM integrity changed")
        with self._routing_lock:
            self.calls.append(request)
            haystack = request.system + "\n" + request.user
            for needle, response in self._fixtures_view._values.items():
                if needle in haystack:
                    return response
            if self._default is not None:
                return self._default
        raise LookupError(
            f"StubLLMClient: no fixture matched. user prompt began with: "
            f"{request.user[:120]!r}"
        )


_STUB_LLM_GUARD_NAMES = (
    "complete", "phase1_producer_declaration",
    "aggregation_producer_declaration", "fixtures", "default",
    "__getattribute__", "__getattr__",
)
_STUB_LLM_ORIGINAL_GUARD = tuple(
    inspect.getattr_static(StubLLMClient, name, None)
    for name in _STUB_LLM_GUARD_NAMES
)


def maintained_stub_llm_integrity(client: object) -> bool:
    """Whether an exact house stub still has its frozen execution contract."""

    if type(client) is not StubLLMClient or tuple(
        inspect.getattr_static(StubLLMClient, name, None)
        for name in _STUB_LLM_GUARD_NAMES
    ) != _STUB_LLM_ORIGINAL_GUARD:
        return False
    try:
        state = object.__getattribute__(client, "__dict__")
    except (AttributeError, TypeError):
        return False
    if any(name in state for name in _STUB_LLM_GUARD_NAMES):
        return False
    fixtures = state.get("_fixtures_view")
    default = state.get("_default")
    calls = state.get("calls")
    return bool(
        type(fixtures) is _VersionedStubFixtures
        and fixtures._owner is client
        and all(type(key) is str and type(value) is str
                for key, value in fixtures._values.items())
        and (default is None or type(default) is str)
        and type(calls) is list
        and type(state.get("_routing_serial")) is int
        and state.get("_routing_serial", -1) >= 0
        and id(state.get("_routing_lock")) == state.get("_routing_lock_identity")
    )


_STUB_LLM_INTEGRITY_FUNCTION = maintained_stub_llm_integrity
_STUB_LLM_MAINTAINED_CLASS = StubLLMClient
