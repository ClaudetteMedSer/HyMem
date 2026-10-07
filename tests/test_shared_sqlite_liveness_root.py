"""Independent bounded reproduction of the production SQLite/GIL inversion."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


_CHILD = r'''
import json, sqlite3, sys, tempfile, threading, time
from pathlib import Path
mode, cache, access = sys.argv[1:]
with tempfile.TemporaryDirectory(prefix="hymem-sqlite-root-") as directory:
    if mode == "guarded":
        from hymem.core.db import connect
        conn = connect(Path(directory) / "test.sqlite")
    else:
        conn = sqlite3.connect(":memory:", check_same_thread=False, cached_statements=int(cache))
    if mode == "raw":
        conn.execute("CREATE TABLE numbers(x INTEGER)")
        conn.executemany("INSERT INTO numbers VALUES (?)", [(i,) for i in range(250)])
        barrier = threading.Barrier(3)
        def yielding(value):
            if value % 17 == 0:
                time.sleep(0)
            return value
        conn.create_function("yielding", 1, yielding)
        def scan(index):
            barrier.wait()
            for i in range(60):
                row = conn.execute("SELECT sum(yielding(x)) + ? FROM numbers /* %d */" % (i % 3), (index,)).fetchone()
                assert row[0] == 31125 + index
        threads = [threading.Thread(target=scan, args=(i,)) for i in range(2)]
        for thread in threads: thread.start()
        barrier.wait()
        for thread in threads: thread.join()
        print("raw-completed", flush=True)
        sys.exit(0)
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    def callback(value):
        entered.set()
        if not release.wait(2):
            raise RuntimeError("release did not run")
        return value
    conn.create_function("root_callback", 1, callback)
    rows, failures = [], []
    def query(sql, value):
        try:
            handle = conn.cursor() if access == "cursor" else conn
            rows.append(handle.execute(sql, (value,)).fetchone()[0])
        except BaseException as exc:
            failures.append(type(exc).__name__)
    first = threading.Thread(target=query, args=("SELECT root_callback(?)", 11))
    first.start()
    assert entered.wait(2)
    def second_query():
        second_started.set()
        query("SELECT root_callback(?) + 1", 20)
    second = threading.Thread(target=second_query)
    second.start()
    assert second_started.wait(2)
    print("both-dispatched", flush=True)
    # The unsafe second query enters sqlite3_limit holding the GIL, so even
    # this timer cannot run. With the Python lock, it can release the callback.
    time.sleep(0.15)
    release.set()
    first.join(2); second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert not failures, failures
    assert sorted(rows) == [11, 21], rows
    conn.close()
    print(json.dumps({"completed": True, "rows": sorted(rows)}), flush=True)
'''


def _run(mode, cache, access, *, executable=sys.executable):
    process = subprocess.Popen(
        [executable, "-B", "-c", _CHILD, mode, str(cache), access],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=3)
        return "timeout", stdout, stderr


@pytest.mark.parametrize("access", ["connection", "cursor"])
def test_guarded_udf_overlap_does_not_freeze_interpreter(access):
    code, stdout, stderr = _run("guarded", 0, access)
    assert code == 0, (code, stdout, stderr)
    assert json.loads(stdout.splitlines()[-1]) == {"completed": True, "rows": [11, 21]}


@pytest.mark.parametrize("cache", [0, 128])
def test_raw_python311_counterfactual_still_reproduces(cache):
    executable = Path("/usr/local/bin/python3.11")
    if not executable.is_file():
        pytest.skip("bounded counterfactual requires the available Python 3.11 runtime")
    code, stdout, stderr = _run("raw", cache, "connection", executable=str(executable))
    assert code == "timeout", (code, stdout, stderr)


@pytest.mark.parametrize("access", ["connection", "cursor"])
def test_guarded_python311_matches_production_runtime_family(access):
    executable = Path("/usr/local/bin/python3.11")
    if not executable.is_file():
        pytest.skip("Python 3.11 unavailable")
    code, stdout, stderr = _run("guarded", 0, access, executable=str(executable))
    assert code == 0, (code, stdout, stderr)


def test_interrupt_can_cancel_sql_while_operation_lock_is_owned():
    script = r'''
import sqlite3, threading
from hymem.core.serialized_sqlite import SerializedConnection
conn = sqlite3.connect(":memory:", factory=SerializedConnection, check_same_thread=False)
entered = threading.Event()
errors = []
def progress():
    entered.set()
    return 0
conn.set_progress_handler(progress, 100)
def run():
    try:
        conn.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000000000) SELECT sum(x) FROM n").fetchall()
    except sqlite3.OperationalError:
        errors.append("interrupted")
worker = threading.Thread(target=run)
worker.start()
assert entered.wait(2)
conn.interrupt()
worker.join(2)
assert not worker.is_alive() and errors == ["interrupted"]
assert conn.execute("SELECT 42").fetchone()[0] == 42
conn.close()
'''
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            capture_output=True, text=True, timeout=6)
    assert result.returncode == 0, result.stderr


def test_c_api_metadata_and_cancellation_coverage():
    import sqlite3
    from hymem.core.serialized_sqlite import SerializedConnection, SerializedCursor
    for name in ("getlimit", "setlimit", "getconfig", "setconfig", "__call__",
                 "close", "commit", "rollback", "create_function", "create_aggregate",
                 "enable_load_extension", "load_extension", "set_progress_handler"):
        if hasattr(sqlite3.Connection, name):
            assert getattr(SerializedConnection, name) is not getattr(sqlite3.Connection, name)
    for name in ("in_transaction", "total_changes", "autocommit", "isolation_level"):
        if hasattr(sqlite3.Connection, name):
            assert isinstance(getattr(SerializedConnection, name), property)
    assert SerializedConnection.interrupt is sqlite3.Connection.interrupt
    assert SerializedCursor.__next__ is not sqlite3.Cursor.__next__
