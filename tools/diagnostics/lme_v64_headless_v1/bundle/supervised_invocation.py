"""Wall-clock supervision for a *trusted*, freshly executed validation worker.

The caller supplies the complete executable command and environment; this module
does not authorize provider access or discover credentials.  Workers that need
authorization must wait for the stdin payload: it is not released until the
exclusive launch/ownership receipt has been durably written.  The worker must
not daemonize, create another session/process group, or delegate work outside its
group.  There must be no competing SIGCHLD handler/reaper in the calling process.
The caller must translate its shutdown signals (for example SIGTERM) into a
Python cancellation exception; this thread-safe primitive installs no global
signal handlers. SIGKILL, parent/host crashes and escaping workers require an
external containment mechanism and cannot be guaranteed by this boundary.
Signal deferral during startup is per-thread: the supervising CLI should be
single-threaded. Multithreaded callers must arrange shutdown-signal masks in
their other threads too, or use a cooperative cancellation policy; an unblocked
sibling can otherwise queue a Python handler for the main thread during launch.

One monotonic deadline covers startup, stdin delivery and execution.  A separate
bounded cleanup allowance covers group termination and reaping.  As with any
userspace supervisor, an uninterruptible kernel/filesystem operation cannot be
given an absolute scheduling guarantee.  An uncertain cleanup is reported as
unsafe, never as success; callers MUST stop the batch unless ``safe_to_continue``
is true.  The primitive does not retry an invocation or reuse its output fence.

Supported platforms are Linux and macOS with waitid(WNOWAIT).  Holding the exited
leader unreaped until group signaling is finished prevents a recycled leader PID
from redirecting termination at another process group.  Captured output remains
in private regular files; no output, command, environment or stdin is in receipts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal


_POLL_SECONDS = 0.02
_MAX_STDIN_BYTES = 1_048_576
# Popen blocks in its constructor before it returns the owned handle.  Deferring
# catchable signals during that ownership-transfer interval avoids losing a
# launched child to KeyboardInterrupt in the supported single-threaded parent.
# Signal masks survive exec, so a fresh
# Python bootstrap MUST restore the exact caller mask before the actual exec.
# It never reads stdin, opens a provider or invokes a shell.  -I -S excludes
# environment/user-site Python startup hooks before mask restoration.
_EXEC_BOOTSTRAP = """import os, signal, sys
signal.pthread_sigmask(signal.SIG_SETMASK, [int(x) for x in sys.argv[1].split(',') if x])
for name in ('SIGPIPE', 'SIGXFZ', 'SIGXFSZ'):
    sig = getattr(signal, name, None)
    if sig is not None:
        signal.signal(sig, signal.SIG_DFL)
os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
"""


@dataclass(frozen=True)
class InvocationOutcome:
    status: Literal["completed", "timeout", "failed", "cancelled"]
    pid: int | None
    pgid: int | None
    returncode: int | None
    started_at: float
    expires_at: float
    finished_at: float
    elapsed_seconds: float
    cleanup_seconds: float
    child_reaped: bool
    group_absent_after_reap: bool
    cleanup_complete: bool
    safe_to_continue: bool
    stdin_bytes_sent: int
    stdout_bytes: int
    stderr_bytes: int
    errors: tuple[str, ...]
    cleanup_warnings: tuple[str, ...]
    output_dir: str
    terminal_receipt_written: bool = False


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} is outside its allowed range") from None
    if not math.isfinite(number) or number <= minimum:
        raise ValueError(f"{name} is outside its allowed range")
    return number


def _private_json(path: Path, value: Mapping[str, object]) -> None:
    """Exclusive, durable metadata write; intentionally has no secret arguments."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _observe(pid: int):
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def _observed_returncode(observation) -> int:
    if observation.si_code == os.CLD_EXITED:
        return int(observation.si_status)
    return -int(observation.si_status)


def _group_absent(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def _owns_group(pid: int) -> bool:
    try:
        return os.getpgid(pid) == pid
    except ProcessLookupError:
        # macOS hides an exited leader from getpgid even while WNOWAIT keeps
        # that exact child waitable (and its PID reserved).  This is not loss of
        # ownership: Popen established the fresh session before exec, and the
        # still-waitable child prevents PID/PGID reuse through final signaling.
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _cleanup_owned(
    process: subprocess.Popen[bytes], allowance: float
) -> tuple[int | None, bool, bool, float, tuple[str, ...]]:
    """Never raise or wait without a remaining-time bound after taking ownership."""
    start = time.monotonic()
    expiry = start + allowance
    errors: list[str] = []
    reaped = False
    absent = False
    returncode = None
    owned = False
    try:
        owned = _owns_group(process.pid)
    except BaseException:
        errors.append("cleanup_ownership_unconfirmed")
    if not owned:
        errors.append("cleanup_owned_group_missing")

    # No poll(), communicate(), or wait() may reap the leader before the final
    # group signal.  Even an already-exited leader reserves its PID here.
    if owned:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except BaseException:
            errors.append("cleanup_term_failed")
        grace_end = min(expiry, start + min(0.25, allowance / 4))
        while time.monotonic() < grace_end:
            try:
                time.sleep(min(_POLL_SECONDS, max(0.0, grace_end - time.monotonic())))
            except BaseException:
                if "cleanup_interrupted" not in errors:
                    errors.append("cleanup_interrupted")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException:
            errors.append("cleanup_kill_failed")

    while time.monotonic() < expiry:
        if not reaped:
            try:
                waited_pid, wait_status = os.waitpid(process.pid, os.WNOHANG)
                if waited_pid == process.pid:
                    returncode = os.waitstatus_to_exitcode(wait_status)
                    process.returncode = returncode
                    reaped = True
            except ChildProcessError:
                errors.append("cleanup_child_reaped_externally")
                break
            except BaseException:
                if "cleanup_reap_failed" not in errors:
                    errors.append("cleanup_reap_failed")
        if reaped:
            # Queries after reaping are safe; never send another signal, since
            # at that point the old PGID is no longer reserved by our leader.
            try:
                absent = _group_absent(process.pid)
            except BaseException:
                if "cleanup_group_check_failed" not in errors:
                    errors.append("cleanup_group_check_failed")
            if absent:
                break
        try:
            time.sleep(min(_POLL_SECONDS, max(0.0, expiry - time.monotonic())))
        except BaseException:
            if "cleanup_interrupted" not in errors:
                errors.append("cleanup_interrupted")
    if not reaped:
        errors.append("cleanup_child_not_reaped")
    if not absent:
        errors.append("cleanup_group_still_present_or_unknown")
    return returncode, reaped, absent, time.monotonic() - start, tuple(errors)


def supervise_invocation(
    command: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    output_dir: str | Path,
    timeout_seconds: float,
    cleanup_seconds: float = 2.0,
    stdin_bytes: bytes = b"",
    output_limit_bytes: int = 8 * 1024 * 1024,
    deadline_expires_at: float | None = None,
) -> InvocationOutcome:
    """Execute once, with no shell and one absolute execution deadline.

    ``deadline_expires_at`` lets a caller include its earlier reservation work in
    the deadline.  It can shorten but never extend ``timeout_seconds``.  Validation
    errors and an already-existing output directory raise before launch.  Other
    failures return a sanitized outcome; the actual output files are private and
    MUST NOT be treated as automatically safe to publish.
    """
    if os.name != "posix" or sys.platform not in {"linux", "darwin"} or not all(
        hasattr(os, name) for name in ("waitid", "WNOWAIT", "WEXITED", "P_PID")
    ) or not hasattr(signal, "pthread_sigmask") or not Path(sys.executable).is_absolute():
        raise RuntimeError("owned supervision requires Linux/macOS waitid(WNOWAIT)")
    timeout = _number(timeout_seconds, "timeout_seconds")
    cleanup = _number(cleanup_seconds, "cleanup_seconds", minimum=0.099999999)
    deadline = None if deadline_expires_at is None else _number(
        deadline_expires_at, "deadline_expires_at"
    )
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence) or not command:
        raise ValueError("command must be a nonempty sequence")
    argv = tuple(command)
    if any(not isinstance(item, str) or not item or "\0" in item for item in argv):
        raise ValueError("command items must be nonempty strings without NUL")
    if not isinstance(env, Mapping) or any(
        not isinstance(key, str) or not key or "=" in key or "\0" in key
        or not isinstance(value, str) or "\0" in value
        for key, value in env.items()
    ):
        raise ValueError("env must be an explicit string mapping")
    environment = dict(env)
    if not isinstance(stdin_bytes, bytes) or len(stdin_bytes) > _MAX_STDIN_BYTES:
        raise ValueError("stdin_bytes must be bytes no larger than 1 MiB")
    if isinstance(output_limit_bytes, bool) or not isinstance(output_limit_bytes, int) or output_limit_bytes < 1:
        raise ValueError("output_limit_bytes must be a positive integer")
    workdir, destination = Path(cwd), Path(output_dir)
    if not workdir.is_absolute() or not workdir.is_dir() or not destination.is_absolute():
        raise ValueError("cwd must be an existing absolute directory; output_dir must be absolute")
    if not destination.parent.is_dir():
        raise ValueError("output_dir parent must already exist")

    start = time.monotonic()
    expiry = min(start + timeout, deadline) if deadline is not None else start + timeout
    if not math.isfinite(expiry):
        raise ValueError("deadline arithmetic must remain finite")
    # Atomic mkdir is the exclusive intent/fence.  Never merge with an old run.
    destination.mkdir(mode=0o700)
    process = None
    stdout = stderr = selector = None
    status = "failed"
    errors: list[str] = []
    cleanup_warnings: list[str] = []
    sent = 0
    reaped = absent = True
    cleanup_elapsed = 0.0
    returncode = None
    pid = None
    stdout_size = stderr_size = 0

    def drain_output(key) -> bool:
        """Drain one bounded buffer; False means the strict saved-byte cap hit."""
        nonlocal stdout_size, stderr_size
        capacity = output_limit_bytes - stdout_size - stderr_size
        try:
            block = os.read(key.fd, min(65536, capacity + 1))
        except BlockingIOError:
            return True
        if not block:
            selector.unregister(key.fileobj)
            key.fileobj.close()
            return True
        saved = block[:capacity]
        stream = stdout if key.data == "stdout" else stderr
        # Regular-file write failures are caught by the enclosing parent guard.
        # A local filesystem is required; no userspace timeout can preempt an
        # uninterruptible kernel write to a malfunctioning filesystem.
        view = memoryview(saved)
        while view:
            written = stream.write(view)
            if not written:
                raise OSError("private output write made no progress")
            view = view[written:]
        if key.data == "stdout":
            stdout_size += len(saved)
        else:
            stderr_size += len(saved)
        if len(block) > capacity:
            if "output_limit_exceeded" not in errors:
                errors.append("output_limit_exceeded")
            return False
        return True

    try:
        _private_json(destination / "intent.json", {
            "version": "owned-invocation-intent-v1", "started_at": start,
            "expires_at": expiry, "cleanup_allowance_seconds": cleanup,
            "output_limit_bytes": output_limit_bytes,
        })
        stdout = open(destination / "stdout.bin", "xb", buffering=0)
        stderr = open(destination / "stderr.bin", "xb", buffering=0)
        os.chmod(destination / "stdout.bin", 0o600)
        os.chmod(destination / "stderr.bin", 0o600)
        if time.monotonic() >= expiry:
            status = "timeout"
            errors.append("deadline_expired_before_launch")
        else:
            blocked = signal.valid_signals() - {signal.SIGKILL, signal.SIGSTOP}
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
            try:
                bootstrap_argv = (
                    sys.executable, "-I", "-S", "-c", _EXEC_BOOTSTRAP,
                    ",".join(str(int(sig)) for sig in sorted(previous_mask)), *argv,
                )
                process = subprocess.Popen(
                    bootstrap_argv, cwd=workdir, env=environment,
                    stdin=subprocess.PIPE if stdin_bytes else subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, close_fds=True,
                    start_new_session=True,
                )
                pid = process.pid
                reaped = absent = False
            finally:
                # Pending cancellation is delivered only after retaining the
                # complete handle and PID.  The enclosing finally can reap it.
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            if not _owns_group(pid):
                raise RuntimeError("worker did not retain its owned process group")
            # The worker must wait for stdin authorization.  A failed durable
            # ownership write therefore cannot release authorized provider work.
            _private_json(destination / "launched.json", {
                "version": "owned-invocation-launch-v1", "pid": pid, "pgid": pid,
                "started_at": start, "expires_at": expiry,
                "ownership_recorded_at": time.monotonic(),
            })
            selector = selectors.DefaultSelector()
            for pipe, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, label)
            if process.stdin is not None:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            while True:
                if time.monotonic() >= expiry:
                    status = "timeout"
                    errors.append("execution_deadline_expired")
                    break
                observed = _observe(pid)
                if observed is not None:
                    if time.monotonic() >= expiry:
                        status = "timeout"
                        errors.append("completion_observed_at_or_after_deadline")
                    elif _observed_returncode(observed) != 0:
                        errors.append("worker_nonzero_exit")
                    elif sent != len(stdin_bytes):
                        errors.append("worker_exited_before_stdin_delivery")
                    else:
                        status = "completed"
                    break
                remaining = max(0.0, expiry - time.monotonic())
                ready = selector.select(min(_POLL_SECONDS, remaining))
                stop = False
                for key, _ in ready:
                    if time.monotonic() >= expiry:
                        break
                    if key.data != "stdin":
                        if not drain_output(key):
                            stop = True
                            break
                    else:
                        try:
                            sent += os.write(process.stdin.fileno(), stdin_bytes[sent:sent + 65536])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            errors.append("worker_closed_stdin_early")
                            stop = True
                            break
                        if sent == len(stdin_bytes):
                            selector.unregister(process.stdin)
                            process.stdin.close()
                if stop:
                    break
    except Exception:
        errors.append("parent_operation_failed")
    except BaseException:
        status = "cancelled"
        errors.append("parent_cancelled")
    finally:
        # Closing a raw, unbuffered pipe never flushes pending user-space bytes.
        # Metadata/selector/stdin failures cannot bypass owned-process cleanup.
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except BaseException:
                    errors.append("stdin_close_failed")
            cleanup_started = time.monotonic()
            returncode, reaped, absent, cleanup_elapsed, cleanup_errors = _cleanup_owned(process, cleanup)
            # macOS may return EPERM for signaling an unreaped, exited-only
            # group. Preserve the event, but it is not a failed cleanup once
            # actual exit zero, authoritative reap and group absence prove that
            # there is nothing left to terminate. Never downgrade uncertainty.
            for code in cleanup_errors:
                if reaped and absent and returncode == 0 and code in {
                    "cleanup_term_failed", "cleanup_kill_failed",
                }:
                    cleanup_warnings.append(code)
                else:
                    errors.append(code)
            if selector is not None and "output_limit_exceeded" not in errors:
                # Never wait for EOF: a descendant may still own a pipe.  Drain
                # only currently readable bounded buffers, within cleanup time.
                try:
                    while time.monotonic() < cleanup_started + cleanup:
                        readable = [
                            key for key, _ in selector.select(0) if key.data != "stdin"
                        ]
                        if not readable:
                            break
                        if not all(drain_output(key) for key in readable):
                            break
                    else:
                        errors.append("final_output_drain_deadline")
                except BaseException:
                    errors.append("final_output_drain_failed")
            cleanup_elapsed = time.monotonic() - cleanup_started
            for pipe in (process.stdout, process.stderr):
                try:
                    pipe.close()
                except BaseException:
                    errors.append("output_pipe_close_failed")
        if selector is not None:
            try:
                selector.close()
            except BaseException:
                errors.append("selector_close_failed")
        for stream, label in ((stdout, "stdout"), (stderr, "stderr")):
            if stream is not None:
                try:
                    size = os.fstat(stream.fileno()).st_size
                    if label == "stdout":
                        stdout_size = size
                    else:
                        stderr_size = size
                except BaseException:
                    errors.append(f"{label}_size_failed")
                try:
                    stream.close()
                except BaseException:
                    errors.append(f"{label}_close_failed")
    if stdout_size + stderr_size > output_limit_bytes and "output_limit_exceeded" not in errors:
        errors.append("output_limit_exceeded")
    if status == "completed" and (errors or not reaped or not absent or returncode != 0):
        status = "failed"
    finish = time.monotonic()
    outcome = InvocationOutcome(
        status=status, pid=pid, pgid=pid, returncode=returncode,
        started_at=start, expires_at=expiry, finished_at=finish,
        elapsed_seconds=finish - start, cleanup_seconds=cleanup_elapsed,
        child_reaped=reaped, group_absent_after_reap=absent,
        cleanup_complete=reaped and absent, safe_to_continue=reaped and absent,
        stdin_bytes_sent=sent, stdout_bytes=stdout_size, stderr_bytes=stderr_size,
        errors=tuple(errors), cleanup_warnings=tuple(cleanup_warnings), output_dir=str(destination),
    )
    try:
        terminal = replace(outcome, terminal_receipt_written=True)
        _private_json(destination / "terminal.json", asdict(terminal))
        outcome = terminal
    except BaseException:
        outcome = replace(
            outcome, status="failed" if status == "completed" else status,
            errors=outcome.errors + ("terminal_receipt_failed",), safe_to_continue=False,
        )
    return outcome
