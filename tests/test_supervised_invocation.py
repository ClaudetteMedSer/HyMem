"""Network-free, outer-bounded tests of the owned invocation boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

from benchmarks.supervised_invocation import supervise_invocation


REPO = Path(__file__).resolve().parents[1]
SUPPORTED = sys.platform in {"linux", "darwin"} and hasattr(os, "WNOWAIT")
pytestmark = pytest.mark.skipif(not SUPPORTED, reason="POSIX waitid(WNOWAIT) required")


def run_case(tmp_path, code, *, setup="", timeout=0.3, cleanup=0.6, stdin=b"", limit=65536):
    """Run the supervisor itself in an externally time-bounded Python process."""
    directory = tmp_path / "invocation"
    script = f"""
import json, os, sys, time
from dataclasses import asdict
from pathlib import Path
import benchmarks.supervised_invocation as module
{setup}
outcome = module.supervise_invocation(
    [sys.executable, '-c', {code!r}], cwd={str(REPO)!r},
    env={{'PATH': os.defpath}}, output_dir={str(directory)!r},
    timeout_seconds={timeout!r}, cleanup_seconds={cleanup!r},
    stdin_bytes=bytes.fromhex(sys.stdin.read()), output_limit_bytes={limit!r})
print(json.dumps(asdict(outcome)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO,
        env={"PATH": os.defpath, "PYTHONPATH": str(REPO)},
        capture_output=True, text=True, input=stdin.hex(), timeout=8,
    )
    assert completed.returncode == 0, completed.stderr
    outcome = json.loads(completed.stdout)
    return outcome, directory


def assert_gone(outcome):
    assert outcome["child_reaped"]
    assert outcome["group_absent_after_reap"]
    assert outcome["cleanup_complete"]
    with pytest.raises(ProcessLookupError):
        os.kill(outcome["pid"], 0)


def test_success_captures_private_bytes_not_receipt_secrets(tmp_path):
    secret = "synthetic-sensitive-value"
    result, directory = run_case(
        tmp_path, f"import sys; print({secret!r}); sys.stderr.write('private error')",
    )
    assert result["status"] == "completed"
    assert result["returncode"] == 0
    assert result["safe_to_continue"]
    assert_gone(result)
    assert (directory / "stdout.bin").read_text() == secret + "\n"
    assert (directory / "stderr.bin").read_text() == "private error"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in directory.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        if path.suffix == ".json":
            assert secret not in path.read_text()
            assert "private error" not in path.read_text()


def test_stdin_delivery_is_after_ownership_receipt(tmp_path):
    directory = tmp_path / "invocation"
    code = (
        "import sys,pathlib; content=sys.stdin.buffer.read(); "
        f"assert pathlib.Path({str(directory / 'launched.json')!r}).is_file(); "
        "sys.stdout.buffer.write(content)"
    )
    result, directory = run_case(tmp_path, code, stdin=b"private payload")
    assert result["status"] == "completed"
    assert result["stdin_bytes_sent"] == len(b"private payload")
    assert (directory / "stdout.bin").read_bytes() == b"private payload"
    assert_gone(result)


@pytest.mark.parametrize("code", [
    "import time; time.sleep(60)",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    "import sys,time\nwhile True:\n sys.stdout.write('heartbeat\\n'); sys.stdout.flush(); time.sleep(.01)",
])
def test_inactivity_and_keepalives_cannot_extend_deadline(tmp_path, code):
    result, _ = run_case(tmp_path, code)
    assert result["status"] == "timeout"
    assert result["returncode"] in {-9, -15}
    assert result["elapsed_seconds"] < 1.5
    assert_gone(result)


def test_worker_that_never_reads_stdin_is_bounded(tmp_path):
    result, _ = run_case(tmp_path, "import time; time.sleep(60)", stdin=b"x" * 1_048_576)
    assert result["status"] == "timeout"
    assert result["stdin_bytes_sent"] < 1_048_576
    assert result["elapsed_seconds"] < 1.5
    assert_gone(result)


def test_stdout_and_stderr_are_strictly_capped(tmp_path):
    result, directory = run_case(
        tmp_path,
        "import os\nwhile True:\n os.write(1,b'x'*65536); os.write(2,b'y'*65536)",
        limit=1001,
    )
    assert result["status"] == "failed"
    assert "output_limit_exceeded" in result["errors"]
    assert sum((directory / name).stat().st_size for name in ("stdout.bin", "stderr.bin")) == 1001
    assert result["stdout_bytes"] + result["stderr_bytes"] == 1001
    assert_gone(result)


def test_large_output_below_cap_is_drained_without_pipe_deadlock(tmp_path):
    result, directory = run_case(
        tmp_path, "import os; os.write(1,b'x'*300000); os.write(2,b'y'*200000)",
        timeout=2, limit=500000,
    )
    assert result["status"] == "completed"
    assert (directory / "stdout.bin").stat().st_size == 300000
    assert (directory / "stderr.bin").stat().st_size == 200000
    assert_gone(result)


def test_worker_crash_is_failure_with_actual_exit(tmp_path):
    result, _ = run_case(tmp_path, "import os; os._exit(17)")
    assert result["status"] == "failed"
    assert result["returncode"] == 17
    assert "worker_nonzero_exit" in result["errors"]
    assert_gone(result)


def test_completion_with_descendant_does_not_wait_for_inherited_pipe_eof(tmp_path):
    code = """
import subprocess, sys
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
print('leader result', flush=True)
"""
    result, directory = run_case(tmp_path, code, cleanup=1.5)
    assert result["elapsed_seconds"] < 2
    assert result["child_reaped"]
    assert (directory / "stdout.bin").read_text() == "leader result\n"
    # A host PID1 that never reaps orphan zombies may retain the group.  Such a
    # host must fail closed; it must not be told that cleanup has been proved.
    if result["group_absent_after_reap"]:
        assert result["status"] == "completed"
    else:
        assert result["status"] == "failed"
        assert not result["safe_to_continue"]


def test_final_drain_enforces_output_cap_after_leader_exit(tmp_path):
    setup = """
original_observe = module._observe
def await_fast_child(pid):
    for _ in range(200):
        observed = original_observe(pid)
        if observed is not None:
            return observed
        time.sleep(.001)
    return None
module._observe = await_fast_child
"""
    result, directory = run_case(tmp_path, "print('0123456789')", setup=setup, limit=4)
    assert result["status"] == "failed"
    assert "output_limit_exceeded" in result["errors"]
    assert (directory / "stdout.bin").read_bytes() == b"0123"
    assert_gone(result)


def test_real_exec_failure_is_sanitized(tmp_path):
    result = supervise_invocation(
        [str(tmp_path / "missing-synthetic-secret-executable")], cwd=REPO, env={},
        output_dir=tmp_path / "exec-failure", timeout_seconds=1,
    )
    assert result.status == "failed"
    assert result.pid is not None
    assert result.child_reaped
    assert result.group_absent_after_reap
    assert "missing-synthetic-secret" not in (tmp_path / "exec-failure" / "terminal.json").read_text()


def test_startup_delay_uses_original_deadline_before_stdin(tmp_path):
    setup = """
original_popen = module.subprocess.Popen
def slow_startup(*args, **kwargs):
    process = original_popen(*args, **kwargs)
    time.sleep(.25)
    return process
module.subprocess.Popen = slow_startup
"""
    result, _ = run_case(tmp_path, "import sys; sys.stdin.buffer.read()", setup=setup,
                         timeout=.1, stdin=b"permission")
    assert result["status"] == "timeout"
    assert result["stdin_bytes_sent"] == 0
    assert_gone(result)


def test_real_startup_sigint_is_deferred_until_owned_handle_exists(tmp_path):
    setup = """
import signal, threading
original_popen = module.subprocess.Popen
def interrupted_startup(*args, **kwargs):
    process = original_popen(*args, **kwargs)
    sender = threading.Thread(target=lambda: (time.sleep(.02), os.kill(os.getpid(), signal.SIGINT)))
    sender.start()
    time.sleep(.08)
    sender.join(timeout=.1)
    return process
module.subprocess.Popen = interrupted_startup
"""
    result, _ = run_case(tmp_path, "import time; time.sleep(60)", setup=setup)
    assert result["status"] == "cancelled"
    assert result["stdin_bytes_sent"] == 0
    assert_gone(result)


def test_bootstrap_restores_exact_signal_mask_to_worker_and_parent(tmp_path):
    setup = """
import signal
original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
expected_mask = sorted(int(sig) for sig in original_mask | {signal.SIGUSR1})
"""
    code = "import json,signal; print(json.dumps(sorted(int(s) for s in signal.pthread_sigmask(signal.SIG_BLOCK, set()))))"
    result, directory = run_case(tmp_path, code, setup=setup)
    assert result["status"] == "completed"
    import signal
    assert json.loads((directory / "stdout.bin").read_text()) == [int(signal.SIGUSR1)]
    assert_gone(result)


def test_slow_ownership_receipt_cannot_release_expired_authorization(tmp_path):
    setup = """
original_write = module._private_json
def slow_write(path, value):
    if path.name == 'launched.json':
        time.sleep(.25)
    return original_write(path, value)
module._private_json = slow_write
"""
    result, _ = run_case(tmp_path, "import sys; sys.stdin.buffer.read()", setup=setup,
                         timeout=.1, stdin=b"permission")
    assert result["status"] == "timeout"
    assert result["stdin_bytes_sent"] == 0
    assert_gone(result)


def test_child_early_stdin_close_is_failed_and_reaped(tmp_path):
    result, _ = run_case(
        tmp_path, "import os,time; os.close(0); time.sleep(.1)", stdin=b"x" * 1_048_576,
    )
    assert result["status"] == "failed"
    assert any(code in result["errors"] for code in (
        "worker_closed_stdin_early", "worker_exited_before_stdin_delivery",
    ))
    assert_gone(result)


def test_late_observed_zero_exit_is_never_success(tmp_path):
    setup = """
original_observe = module._observe
def delayed_observe(pid):
    observed = original_observe(pid)
    if observed is not None:
        time.sleep(.3)
    return observed
module._observe = delayed_observe
"""
    result, _ = run_case(tmp_path, "print('result')", setup=setup, timeout=0.15)
    assert result["status"] == "timeout"
    assert result["returncode"] == 0
    assert_gone(result)


@pytest.mark.parametrize("exception", ["KeyboardInterrupt", "SystemExit", "BaseException"])
def test_parent_cancellation_reaps_worker(tmp_path, exception):
    setup = f"""
def cancelled_observe(pid):
    raise {exception}('synthetic secret never published')
module._observe = cancelled_observe
"""
    result, directory = run_case(tmp_path, "import time; time.sleep(60)", setup=setup)
    assert result["status"] == "cancelled"
    assert "parent_cancelled" in result["errors"]
    assert "synthetic secret" not in (directory / "terminal.json").read_text()
    assert_gone(result)


def test_failed_ownership_receipt_never_releases_stdin(tmp_path):
    marker = tmp_path / "authorized-work"
    setup = """
original_write = module._private_json
def broken_launch_receipt(path, value):
    if path.name == 'launched.json':
        raise OSError('synthetic secret')
    return original_write(path, value)
module._private_json = broken_launch_receipt
"""
    code = f"import sys,pathlib; data=sys.stdin.buffer.read(); data and pathlib.Path({str(marker)!r}).touch()"
    result, _ = run_case(tmp_path, code, setup=setup, stdin=b"permission")
    assert result["status"] == "failed"
    assert result["stdin_bytes_sent"] == 0
    assert not marker.exists()
    assert_gone(result)


def test_terminal_receipt_failure_is_fail_closed_after_cleanup(tmp_path):
    setup = """
original_write = module._private_json
def broken_terminal_receipt(path, value):
    if path.name == 'terminal.json':
        raise OSError('synthetic secret')
    return original_write(path, value)
module._private_json = broken_terminal_receipt
"""
    result, directory = run_case(tmp_path, "print('finished')", setup=setup)
    assert result["status"] == "failed"
    assert not result["safe_to_continue"]
    assert not result["terminal_receipt_written"]
    assert "terminal_receipt_failed" in result["errors"]
    assert not (directory / "terminal.json").exists()
    assert_gone(result)


def test_intent_receipt_failure_launches_nothing(tmp_path):
    setup = """
def broken_intent_receipt(path, value):
    raise OSError('synthetic secret')
module._private_json = broken_intent_receipt
"""
    result, _ = run_case(tmp_path, "raise AssertionError('must not launch')", setup=setup)
    assert result["status"] == "failed"
    assert result["pid"] is None
    assert result["cleanup_complete"]


def test_cleanup_signals_only_owned_group_with_descendant(tmp_path):
    # Leader handles TERM by waiting/reaping its child; both share the fresh
    # owned group.  Reaping is needed even on hosts with slow PID1 zombie reap.
    code = """
import os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
print(child.pid, flush=True)
def terminate(signum, frame):
    child.wait(timeout=.1)
    sys.exit(0)
signal.signal(signal.SIGTERM, terminate)
time.sleep(60)
"""
    result, directory = run_case(tmp_path, code, cleanup=1)
    assert result["status"] == "timeout"
    assert_gone(result)
    descendant = int((directory / "stdout.bin").read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(descendant, 0)


@pytest.mark.parametrize("value", [0, -1, True, False, float("nan"), float("inf"), "1"])
def test_invalid_timeout_is_rejected_before_fence(tmp_path, value):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        supervise_invocation([sys.executable, "-c", "pass"], cwd=REPO, env={},
                             output_dir=output, timeout_seconds=value)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("cleanup_seconds", 0.01), ("cleanup_seconds", True),
    ("deadline_expires_at", float("nan")), ("deadline_expires_at", True),
    ("output_limit_bytes", True), ("output_limit_bytes", 0),
    ("stdin_bytes", "not bytes"), ("stdin_bytes", b"x" * 1_048_577),
    ("env", {"BAD=NAME": "value"}), ("env", {"GOOD": None}),
    ("command", "not-a-sequence"), ("command", [sys.executable, "bad\0value"]),
    ("cwd", "relative"), ("output_dir", "relative"),
], ids=[
    "cleanup-too-short", "cleanup-bool", "deadline-nan", "deadline-bool",
    "output-bool", "output-zero", "stdin-string", "stdin-oversize",
    "env-name", "env-value", "command-string", "command-nul",
    "cwd-relative", "output-relative",
])
def test_invalid_parameters_are_rejected_before_fence(tmp_path, field, value):
    arguments = dict(command=[sys.executable, "-c", "pass"], cwd=REPO, env={},
                     output_dir=tmp_path / "invalid", timeout_seconds=1)
    arguments[field] = value
    with pytest.raises(ValueError):
        supervise_invocation(**arguments)
    assert not (tmp_path / "invalid").exists()


def test_expired_absolute_deadline_does_not_launch(tmp_path):
    result = supervise_invocation(
        [sys.executable, "-c", "raise AssertionError('must not launch')"],
        cwd=REPO, env={}, output_dir=tmp_path / "expired", timeout_seconds=60,
        deadline_expires_at=time.monotonic() - 1,
    )
    assert result.status == "timeout"
    assert result.pid is None
    assert result.expires_at < result.started_at


def test_absolute_deadline_never_extends_relative_bound(tmp_path):
    result = supervise_invocation(
        [sys.executable, "-c", "pass"], cwd=REPO, env={},
        output_dir=tmp_path / "shorter", timeout_seconds=1,
        deadline_expires_at=time.monotonic() + 100,
    )
    assert result.expires_at == result.started_at + 1


def test_existing_fence_is_never_replayed(tmp_path):
    destination = tmp_path / "used"
    destination.mkdir()
    marker = destination / "untouched"
    marker.write_text("old evidence")
    with pytest.raises(FileExistsError):
        supervise_invocation([sys.executable, "-c", "pass"], cwd=REPO, env={},
                             output_dir=destination, timeout_seconds=1)
    assert list(destination.iterdir()) == [marker]


def test_concurrent_same_fence_has_only_one_owner(tmp_path):
    script = f"""
import concurrent.futures, json, os, sys
from benchmarks.supervised_invocation import supervise_invocation
def call():
    try:
        result = supervise_invocation([sys.executable, '-c', 'import time; time.sleep(.05)'],
            cwd={str(REPO)!r}, env={{'PATH': os.defpath}}, output_dir={str(tmp_path / 'shared')!r},
            timeout_seconds=1, cleanup_seconds=.3)
        return result.status
    except FileExistsError:
        return 'fenced'
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    print(json.dumps(sorted(pool.map(lambda _: call(), range(2)))))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["completed", "fenced"]


def test_concurrent_distinct_groups_do_not_cancel_each_other(tmp_path):
    script = f"""
import concurrent.futures, json, os, sys
from benchmarks.supervised_invocation import supervise_invocation
def call(index):
    code = 'import time; time.sleep(60)' if index == 0 else 'import time; time.sleep(.35)'
    result = supervise_invocation([sys.executable, '-c', code],
        cwd={str(REPO)!r}, env={{'PATH': os.defpath}},
        output_dir={str(tmp_path)!r} + '/group-' + str(index),
        timeout_seconds=.15 if index == 0 else 1, cleanup_seconds=.5)
    return [result.status, result.pid, result.cleanup_complete]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    print(json.dumps(list(pool.map(call, range(2)))))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    outcomes = json.loads(result.stdout)
    assert [item[0] for item in outcomes] == ["timeout", "completed"]
    assert outcomes[0][1] != outcomes[1][1]
    assert all(item[2] for item in outcomes)


def test_failed_kill_is_reported_unsafe_without_unbounded_wait(tmp_path):
    # Deliberately deny this test's owned signals, then restore the real function
    # and clean that same still-unreaped child in the test parent. No leaked job.
    script = f"""
import json, os, signal, sys, time
from dataclasses import asdict
import benchmarks.supervised_invocation as module
original_killpg = os.killpg
def deny_signal(pgid, sig):
    if sig:
        raise PermissionError('synthetic denial')
    return original_killpg(pgid, sig)
os.killpg = deny_signal
result = module.supervise_invocation(
    [sys.executable, '-c', 'import time; time.sleep(60)'], cwd={str(REPO)!r}, env={{}},
    output_dir={str(tmp_path / 'uncertain-cleanup')!r}, timeout_seconds=.1, cleanup_seconds=.1)
os.killpg = original_killpg
assert not result.safe_to_continue and not result.child_reaped
original_killpg(result.pid, signal.SIGKILL)
end = time.monotonic() + 1
while time.monotonic() < end:
    pid, _ = os.waitpid(result.pid, os.WNOHANG)
    if pid == result.pid:
        break
    time.sleep(.01)
else:
    raise AssertionError('test cleanup did not reap its child')
print(json.dumps(asdict(result)))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["elapsed_seconds"] < .8
    assert not outcome["cleanup_complete"]
    assert "cleanup_child_not_reaped" in outcome["errors"]


def test_unsupported_platform_fails_before_fence(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "unsupported")
    with pytest.raises(RuntimeError):
        supervise_invocation([sys.executable, "-c", "pass"], cwd=REPO, env={},
                             output_dir=tmp_path / "unsupported", timeout_seconds=1)
    assert not (tmp_path / "unsupported").exists()
