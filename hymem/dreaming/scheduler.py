"""Long-lived daemon thread that owns the dream cycle.

Replaces FastAPI BackgroundTasks for dreaming. The thread holds one forked
HyMem (separate SQLite connection) for its whole lifetime, so cycles don't
pay fork+close cost per kick and there's a single owner to start/stop.

Endpoints call `kick()` unconditionally; cooldown pacing lives here, not at
the call site.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hymem.api import HyMem

log = logging.getLogger("hymem.dreaming.scheduler")


class DreamScheduler:
    def __init__(self, hy: "HyMem", cooldown: float) -> None:
        self._hy = hy  # live instance; we invalidate its caches after each cycle
        self._cooldown = cooldown
        self._kick = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dream_hy: "HyMem | None" = None
        self._last_run: float = 0.0
        self._cycles_completed: int = 0

    @property
    def cycles_completed(self) -> int:
        return self._cycles_completed

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._dream_hy = self._hy.fork()
        self._thread = threading.Thread(
            target=self._loop, name="hymem-dream", daemon=True
        )
        self._thread.start()
        log.info("dream_scheduler.started cooldown=%.0fs", self._cooldown)

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._kick.set()  # wake the loop so it observes stop
        self._thread.join(timeout=timeout)
        if self._dream_hy is not None:
            self._dream_hy.close()
            self._dream_hy = None
        self._thread = None
        log.info("dream_scheduler.stopped")

    def kick(self) -> None:
        """Signal a dream may be due. Returns immediately."""
        self._kick.set()

    def wait_for_cycle(self, n: int, timeout: float = 5.0) -> bool:
        """Block until at least `n` cycles have completed since start, or
        until `timeout` elapses. Returns True if the count was reached.

        Used by tests that need to assert on post-cycle state without races.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cycles_completed >= n:
                return True
            time.sleep(0.02)
        return self._cycles_completed >= n

    def _loop(self) -> None:
        assert self._dream_hy is not None
        while not self._stop.is_set():
            self._kick.wait()
            if self._stop.is_set():
                return
            self._kick.clear()

            since = time.monotonic() - self._last_run
            if since < self._cooldown:
                # Honor cooldown but stay responsive to stop.
                self._stop.wait(timeout=self._cooldown - since)
                if self._stop.is_set():
                    return

            self._last_run = time.monotonic()
            try:
                start = time.monotonic()
                self._dream_hy.dream()
                self._hy.invalidate_query_caches()
                self._cycles_completed += 1
                log.info(
                    "dream_scheduler.cycle_complete elapsed=%.1fs",
                    time.monotonic() - start,
                )
            except Exception:
                log.exception("dream_scheduler.cycle_failed")
