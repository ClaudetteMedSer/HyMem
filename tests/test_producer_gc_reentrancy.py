"""Producer registry cleanup must not deadlock an in-flight provider call.

The regression runs in bounded subprocesses: restoring the former Lock must
produce the diagnosed deadlock, not wedge pytest or get swallowed as an
unraisable weakref exception. No network, store, or automatic-GC policy changes
are needed to force collection at the actual guarded dispatch boundary.
"""

from __future__ import annotations

import faulthandler
import gc
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from hymem.deadline import DeadlineBoundLLMClient, MonotonicDeadline
from hymem.extraction import producer
from hymem.extraction.llm import LLMRequest, StubLLMClient


_REQUEST = LLMRequest(system="registry regression", user="complete normally")


class _UnknownLLM:
    def complete(self, request: LLMRequest) -> str:
        return "unknown-ok"


def _wrapper(client: object) -> DeadlineBoundLLMClient:
    # A static clock excludes timing/performance from this lock regression.
    return DeadlineBoundLLMClient(client, MonotonicDeadline(100, clock=lambda: 0))


def _authorize(client: object, key: str) -> dict:
    binding = producer.phase1_producer_binding(client)
    assert binding["identity_exact"] is False
    producer.authorize_inexact_producer_generation(client, key)
    assert producer.phase1_generation_runtime_authorized(key, False) == 1
    return binding


def _force_gc_during_dispatch(kind: str) -> dict:
    source = StubLLMClient(default="ok")
    active = _wrapper(source)
    before = producer.phase1_producer_binding(active)
    assert before == producer.phase1_producer_binding(source)
    assert before["identity_exact"] is True
    live_unknown = _UnknownLLM()
    live_binding = _authorize(live_unknown, "live-generation")

    if kind == "proxy":
        doomed = _wrapper(_UnknownLLM())
        dead_source = weakref.ref(doomed._inner)
        _authorize(doomed, "dead-generation")
        registry = producer._producer_proxies
    else:
        doomed = _UnknownLLM()
        dead_source = weakref.ref(doomed)
        _authorize(doomed, "dead-generation")
        registry = producer._unknown_instances
    doomed._cycle = doomed
    dead_id = id(doomed)
    dead_reference = weakref.ref(doomed)
    # Keep a strong reference until the exact guarded allocation. Automatic
    # collection remains enabled throughout; it cannot collect this early.
    retained = [doomed]
    del doomed
    assert dead_id in registry

    original_getattr_static = inspect.getattr_static
    collected = False

    def collect_at_guard_allocation(*args, **kwargs):
        nonlocal collected
        # Every patched inspect call during complete() is reached beneath
        # _phase1_proxy_source's lock, including support-integrity checks.
        if not collected:
            collected = True
            retained.clear()
            print("collecting-under-registry-lock", flush=True)
            gc.collect()
        return original_getattr_static(*args, **kwargs)

    inspect.getattr_static = collect_at_guard_allocation
    try:
        assert active.complete(_REQUEST) == "ok"
    finally:
        inspect.getattr_static = original_getattr_static

    assert collected
    assert dead_reference() is None and dead_source() is None
    assert dead_id not in registry
    assert producer.phase1_generation_runtime_authorized("dead-generation", False) == 0
    assert producer.phase1_generation_runtime_authorized("live-generation", False) == 1
    assert producer.phase1_producer_binding(live_unknown) == live_binding
    assert producer.phase1_producer_binding(active) == before

    # The reentrant lock must not confer authority on a rebound wrapper or
    # permit dispatch to the new source, even after cleanup occurred.
    other = StubLLMClient(default="other")
    active._inner = other
    with pytest.raises(RuntimeError, match="proxy identity changed"):
        active.complete(_REQUEST)
    assert other.calls == []
    assert producer.phase1_producer_binding(active)["identity_exact"] is False
    return {"kind": kind, "cleanup": True, "authority": "preserved"}


def _nonweak_eviction() -> dict:
    events = []

    class NonweakLLM:
        __slots__ = ("key",)

        def __init__(self, key):
            self.key = key

        def complete(self, request):
            return "ok"

        def __del__(self):
            # The registry is the last strong owner when eviction occurs.
            # User finalizers can call identity infrastructure again. They
            # must see revoked authority and must not corrupt a new entry.
            events.append((
                self.key,
                producer.phase1_generation_runtime_authorized(self.key, False),
            ))

    old_cap = producer._MAX_NONWEAK_UNKNOWN_CLIENTS
    producer._MAX_NONWEAK_UNKNOWN_CLIENTS = 2
    try:
        evicted = NonweakLLM("evicted")
        _authorize(evicted, evicted.key)
        evicted_id = id(evicted)
        del evicted
        held = NonweakLLM("held")
        held_binding = _authorize(held, held.key)
        replacement = NonweakLLM("replacement")
        replacement_binding = _authorize(replacement, replacement.key)
        assert events == [("evicted", 0)]
        assert evicted_id not in producer._unknown_nonweak
        assert producer.phase1_generation_runtime_authorized("evicted", False) == 0
        assert producer.phase1_producer_binding(held) == held_binding
        assert producer.phase1_producer_binding(replacement) == replacement_binding
        assert len(producer._unknown_nonweak) == 2
    finally:
        producer._MAX_NONWEAK_UNKNOWN_CLIENTS = old_cap
    return {"kind": "nonweak", "cleanup": True, "authority": "revoked-before-finalizer"}


def _stale_callbacks() -> dict:
    first = _UnknownLLM()
    replacement = _UnknownLLM()
    _authorize(first, "first")
    _authorize(replacement, "replacement")
    first_ref = producer._unknown_instances[id(first)][0]
    replacement_entry = producer._unknown_instances[id(replacement)]
    # Simulate an old callback arriving for a recycled object id. The exact
    # weakref, not merely the integer id, must match before deleting authority.
    producer._drop_unknown_reference(id(replacement), first_ref)
    assert producer._unknown_instances[id(replacement)] is replacement_entry
    assert producer.phase1_generation_runtime_authorized("replacement", False) == 1
    first_proxy = _wrapper(first)
    replacement_proxy = _wrapper(replacement)
    first_proxy_ref = producer._producer_proxies[id(first_proxy)][0]
    replacement_proxy_entry = producer._producer_proxies[id(replacement_proxy)]
    producer._drop_producer_proxy(id(replacement_proxy), first_proxy_ref)
    assert producer._producer_proxies[id(replacement_proxy)] is replacement_proxy_entry
    assert replacement_proxy.complete(_REQUEST) == "unknown-ok"
    return {"kind": "stale", "authority": "preserved"}


def _concurrent_dispatch_and_collection() -> dict:
    workers = 6
    rounds = 15
    barrier = threading.Barrier(workers)

    def exercise(worker: int) -> list[weakref.ReferenceType]:
        client = _UnknownLLM()
        key = f"worker-{worker}"
        binding = _authorize(client, key)
        active = _wrapper(client)
        references = []
        barrier.wait(timeout=10)
        for _ in range(rounds):
            doomed = _wrapper(_UnknownLLM())
            doomed._cycle = doomed
            references.append(weakref.ref(doomed))
            del doomed
            assert active.complete(_REQUEST) == "unknown-ok"
            assert producer.phase1_producer_binding(active) == binding
            assert producer.phase1_generation_runtime_authorized(key, False) == 1
            gc.collect()
        return references

    with ThreadPoolExecutor(max_workers=workers) as pool:
        references = [ref for result in pool.map(exercise, range(workers)) for ref in result]
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not producer._producer_proxies
    assert not producer._unknown_instances
    assert not producer._unknown_generation_keys
    assert not producer._runtime_authorized_inexact_generations
    return {"kind": "concurrent", "calls": workers * rounds, "retained": 0}


def _durable_identity_unchanged() -> dict:
    from hymem.config import HyMemConfig
    from hymem.dreaming.aggregation_generation import aggregation_generation_binding
    from hymem.dreaming.semantic_generation import semantic_generation_suffix

    # No HyMem instance or filesystem operations: config and exact stubs are
    # enough to compare the existing durable material-generation identities.
    cfg = HyMemConfig(root=Path("unused-producer-regression-root"))
    client = StubLLMClient(default="ok")

    def identities():
        return {
            "phase1": producer.phase1_generation_binding(cfg.prompt_version, client),
            "aggregation": aggregation_generation_binding(cfg, client),
            **{tier: semantic_generation_suffix(tier, client)
               for tier in ("digest", "profile", "facts")},
        }

    current = identities()
    lock = producer._unknown_lock
    try:
        producer._unknown_lock = threading.Lock()
        previous = identities()
    finally:
        producer._unknown_lock = lock
    assert current == previous
    assert current["phase1"]["producer"]["identity_exact"] is True
    return {"kind": "identity", "durable_generations": "unchanged"}


def _child(kind: str, old_lock: bool) -> None:
    if old_lock:
        producer._unknown_lock = threading.Lock()
    # exit=True cannot be swallowed by a weakref callback, unlike a signal
    # raising KeyboardInterrupt. The parent supplies a second hard bound.
    faulthandler.dump_traceback_later(5 if old_lock else 25, exit=True)
    try:
        if kind in {"proxy", "unknown"}:
            result = _force_gc_during_dispatch(kind)
        elif kind == "nonweak":
            result = _nonweak_eviction()
        elif kind == "stale":
            result = _stale_callbacks()
        elif kind == "identity":
            result = _durable_identity_unchanged()
        else:
            result = _concurrent_dispatch_and_collection()
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_child(kind: str, *, old_lock: bool = False) -> subprocess.CompletedProcess:
    command = [sys.executable, str(Path(__file__).resolve()), kind]
    if old_lock:
        command.append("--old-lock")
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    return subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=35, check=False,
    )


@pytest.mark.parametrize("kind", ["proxy", "unknown", "nonweak", "stale", "concurrent", "identity"])
def test_registry_collection_preserves_dispatch_and_authority(kind):
    result = _run_child(kind)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["kind"] == kind
    assert result.stderr == ""


@pytest.mark.parametrize("kind,callback", [
    ("proxy", "_drop_producer_proxy"),
    ("unknown", "_drop_unknown_reference"),
])
def test_former_lock_reproduces_cleanup_deadlock_in_bounded_subprocess(kind, callback):
    result = _run_child(kind, old_lock=True)
    assert result.returncode != 0
    assert "collecting-under-registry-lock" in result.stdout
    assert "Timeout" in result.stderr
    assert callback in result.stderr
    assert "_phase1_proxy_source" in result.stderr


if __name__ == "__main__":
    _child(sys.argv[1], "--old-lock" in sys.argv[2:])
