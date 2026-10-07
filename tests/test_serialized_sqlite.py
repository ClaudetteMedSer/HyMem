"""Regression for Python callback / shared SQLite connection GIL inversion."""
from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
import threading

import pytest

from hymem.core import db as core_db
from hymem.core.serialized_sqlite import SerializedConnection, SerializedCursor


_CONCURRENT_UDF_SCRIPT = textwrap.dedent(
    """
    import sqlite3, sys, threading, time
    from hymem.core.serialized_sqlite import SerializedConnection

    cache = int(sys.argv[1])
    guarded = sys.argv[2] == "guarded"
    args = {"factory": SerializedConnection} if guarded else {}
    conn = sqlite3.connect(
        ":memory:", check_same_thread=False, cached_statements=cache, **args
    )
    conn.execute("CREATE TABLE source(value INTEGER)")
    conn.executemany("INSERT INTO source VALUES (?)", [(i,) for i in range(200)])

    def callback(value):
        if value % 20 == 0:
            time.sleep(0)  # release the GIL while SQLite owns its mutex
        return value

    conn.create_function("callback", 1, callback)
    stop = time.monotonic() + 0.6
    start = threading.Barrier(8)
    errors = []

    def read(worker):
        try:
            start.wait()
            turn = 0
            while time.monotonic() < stop:
                row = conn.execute(
                    f"SELECT sum(callback(value)) FROM source "
                    f"WHERE value % {worker + 2} = {turn % (worker + 2)} "
                    f"/* cache miss {turn % 256} */"
                ).fetchone()
                assert row is not None
                turn += 1
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=read, args=(i,), daemon=True)
               for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1.5)
    if errors:
        raise errors[0]
    assert not any(thread.is_alive() for thread in threads)
    conn.close()
    print("completed", flush=True)
    """
)


@pytest.mark.parametrize("cached_statements", [0, 128])
def test_shared_udf_connection_counterfactual_and_fix(cached_statements):
    """Both cache modes deadlock raw Python 3.11/3.13; guarded mode finishes."""
    def run(mode):
        return subprocess.run(
            [sys.executable, "-B", "-c", _CONCURRENT_UDF_SCRIPT,
             str(cached_statements), mode],
            capture_output=True, text=True, timeout=4.0,
        )

    guarded = run("guarded")
    assert guarded.returncode == 0, guarded.stderr
    assert guarded.stdout.strip() == "completed"

    # This control reproduces the interpreter bug on the supported 3.11-3.13
    # runtimes. If a later interpreter fixes it, the guard remains valuable but
    # this historical counterfactual need no longer be asserted.
    if sys.version_info < (3, 14):
        with pytest.raises(subprocess.TimeoutExpired):
            run("raw")


def test_cursor_parameter_binding_iteration_and_udf_reentry():
    conn = sqlite3.connect(":memory:", factory=SerializedConnection,
                           check_same_thread=False)
    try:
        assert isinstance(conn, sqlite3.Connection)
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE source(value INTEGER)")
        conn.executemany("INSERT INTO source VALUES (?)", [(2,), (3,)])

        def nested(value):
            return conn.execute("SELECT ? + 1", (value,)).fetchone()[0]

        conn.create_function("nested", 1, nested)
        cursor = conn.cursor()
        assert isinstance(cursor, SerializedCursor)
        rows = list(cursor.execute("SELECT nested(value) AS result FROM source"))
        assert [row["result"] for row in rows] == [3, 4]
        cursor.execute("SELECT value FROM source ORDER BY value")
        assert cursor.fetchmany(1)[0]["value"] == 2
        assert cursor.fetchone()["value"] == 3
        assert cursor.fetchall() == []
        cursor.close()
        with pytest.raises(TypeError, match="SerializedCursor"):
            conn.cursor(factory=sqlite3.Cursor)
    finally:
        conn.close()


def test_sql_failure_releases_connection_lock():
    conn = sqlite3.connect(":memory:", factory=SerializedConnection,
                           check_same_thread=False)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("SELECT * FROM missing_table")
        result = []
        thread = threading.Thread(
            target=lambda: result.append(conn.execute("SELECT ?", (7,)).fetchone()[0]),
            daemon=True,
        )
        thread.start()
        thread.join(2.0)
        assert not thread.is_alive()
        assert result == [7]
    finally:
        conn.close()


def test_connection_context_rollback_releases_lock():
    conn = sqlite3.connect(":memory:", factory=SerializedConnection,
                           check_same_thread=False)
    try:
        conn.execute("CREATE TABLE source(value INTEGER)")
        with pytest.raises(ValueError):
            with conn:
                conn.execute("INSERT INTO source VALUES (9)")
                raise ValueError("rollback")
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                conn.execute("SELECT count(*) FROM source").fetchone()[0]
            ),
            daemon=True,
        )
        thread.start()
        thread.join(2.0)
        assert not thread.is_alive()
        assert result == [0]
    finally:
        conn.close()


def test_transaction_scope_excludes_other_threads():
    conn = sqlite3.connect(":memory:", factory=SerializedConnection,
                           check_same_thread=False, isolation_level=None)
    conn.execute("CREATE TABLE source(value INTEGER)")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def writer():
        with core_db.transaction(conn):
            conn.execute("INSERT INTO source VALUES (1)")
            entered.set()
            assert release.wait(2.0)

    def contender():
        conn.execute("SELECT count(*) FROM source").fetchone()
        finished.set()

    first = threading.Thread(target=writer, daemon=True)
    second = threading.Thread(target=contender, daemon=True)
    try:
        first.start()
        assert entered.wait(2.0)
        second.start()
        assert not finished.wait(0.05)
        release.set()
        first.join(2.0)
        second.join(2.0)
        assert not first.is_alive() and not second.is_alive()
        assert finished.is_set()
    finally:
        release.set()
        conn.close()
