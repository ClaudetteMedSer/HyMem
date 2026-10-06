"""Serialize entry into a SQLite handle before SQLite can wait with the GIL.

``check_same_thread=False`` permits sharing but does not make Python callbacks
safe to enter concurrently.  One thread can own SQLite's connection mutex and
wait for the GIL in a UDF while another owns the GIL and waits for that mutex.
The Python RLock is acquired before every operation that can enter SQLite.
Callers must use the instance methods and cursors returned by ``cursor()`` or
``execute()``. Explicit base-class calls such as ``sqlite3.Connection.execute``
and directly constructed ``sqlite3.Cursor(conn)`` bypass subclass dispatch.
``interrupt()`` deliberately remains lock-free so another thread can cancel a
running query.
"""
from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Iterator


class SerializedCursor(sqlite3.Cursor):
    """A cursor whose stepping and lifetime operations use its owner's lock."""

    def __init__(self, connection):
        self._owner = connection
        super().__init__(connection)

    def _locked(self, method, *args, **kwargs):
        with self._owner._sqlite_lock:
            return method(self, *args, **kwargs)

    def execute(self, sql, parameters=(), /):
        return self._locked(sqlite3.Cursor.execute, sql, parameters)

    def executemany(self, sql, parameters, /):
        return self._locked(sqlite3.Cursor.executemany, sql, parameters)

    def executescript(self, sql_script, /):
        return self._locked(sqlite3.Cursor.executescript, sql_script)

    def fetchone(self):
        return self._locked(sqlite3.Cursor.fetchone)

    def fetchmany(self, size=None):
        if size is None:
            return self._locked(sqlite3.Cursor.fetchmany)
        return self._locked(sqlite3.Cursor.fetchmany, size)

    def fetchall(self):
        return self._locked(sqlite3.Cursor.fetchall)

    def __next__(self):
        return self._locked(sqlite3.Cursor.__next__)

    def __iter__(self):
        return self

    def close(self):
        return self._locked(sqlite3.Cursor.close)

    def setinputsizes(self, sizes, /):
        return self._locked(sqlite3.Cursor.setinputsizes, sizes)

    def setoutputsize(self, size, column=None, /):
        if column is None:
            return self._locked(sqlite3.Cursor.setoutputsize, size)
        return self._locked(sqlite3.Cursor.setoutputsize, size, column)


class SerializedConnection(sqlite3.Connection):
    """A normal sqlite3.Connection with one Python-side operation mutex.

    Custom cursor factories must inherit SerializedCursor. A bare sqlite3.Cursor
    would bypass the protection during fetch/iteration, so reject it explicitly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sqlite_lock = threading.RLock()

    def cursor(self, factory=None):
        if factory is None:
            factory = SerializedCursor
        elif not isinstance(factory, type) or not issubclass(factory, SerializedCursor):
            raise TypeError("cursor factory must inherit SerializedCursor")
        with self._sqlite_lock:
            return sqlite3.Connection.cursor(self, factory)

    def execute(self, sql, parameters=(), /):
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql, parameters, /):
        return self.cursor().executemany(sql, parameters)

    def executescript(self, sql_script, /):
        return self.cursor().executescript(sql_script)

    def iterdump(self) -> Iterator[str]:
        # sqlite3's implementation is a generator: locking only its creation
        # does not guard later stepping. Lock each advance without holding the
        # mutex across the caller's own iteration body.
        with self._sqlite_lock:
            dump = sqlite3.Connection.iterdump(self)
        while True:
            with self._sqlite_lock:
                try:
                    line = next(dump)
                except StopIteration:
                    return
            yield line

    def backup(self, target, *, pages=-1, progress=None, name="main", sleep=0.250):
        # Backups touch two handles. A fixed order prevents backups in opposite
        # directions from acquiring their Python locks in opposite order.
        target_lock = getattr(target, "_sqlite_lock", None)
        if target_lock is None or target is self:
            with self._sqlite_lock:
                return sqlite3.Connection.backup(
                    self, target, pages=pages, progress=progress, name=name, sleep=sleep
                )
        first, second = sorted((self, target), key=id)
        with first._sqlite_lock, second._sqlite_lock:
            return sqlite3.Connection.backup(
                self, target, pages=pages, progress=progress, name=name, sleep=sleep
            )

    def blobopen(self, *args, **kwargs):
        # sqlite3.Blob is a live native handle whose methods cannot be wrapped
        # while preserving its type. No HyMem path uses incremental BLOB I/O.
        raise NotImplementedError("shared connections do not support blobopen")

    def __enter__(self):
        self._sqlite_lock.acquire()
        try:
            return sqlite3.Connection.__enter__(self)
        except BaseException:
            self._sqlite_lock.release()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return sqlite3.Connection.__exit__(self, exc_type, exc_value, traceback)
        finally:
            self._sqlite_lock.release()


def _guard_connection_method(name: str) -> None:
    method = getattr(sqlite3.Connection, name, None)
    if method is None:
        return

    def guarded(self, *args, **kwargs):
        with self._sqlite_lock:
            return method(self, *args, **kwargs)

    guarded.__name__ = name
    setattr(SerializedConnection, name, guarded)


for _method in (
    "commit", "rollback", "close", "create_function",
    "create_aggregate", "create_collation", "create_window_function",
    "set_authorizer", "set_progress_handler", "set_trace_callback",
    "enable_load_extension", "load_extension", "serialize", "deserialize",
    "getlimit", "setlimit", "getconfig", "setconfig", "__call__",
):
    _guard_connection_method(_method)


def _guard_property(owner: type, name: str, *, cursor: bool = False) -> None:
    descriptor = getattr(owner.__mro__[1], name, None)
    if descriptor is None:
        return

    def lock_for(instance):
        connection = instance._owner if cursor else instance
        return connection._sqlite_lock

    def get(instance):
        with lock_for(instance):
            return descriptor.__get__(instance, type(instance))

    def set(instance, value):
        with lock_for(instance):
            return descriptor.__set__(instance, value)

    setattr(owner, name, property(get, set if hasattr(descriptor, "__set__") else None))


for _property in (
    "in_transaction", "total_changes", "isolation_level", "row_factory",
    "text_factory", "autocommit",
):
    _guard_property(SerializedConnection, _property)

for _property in (
    "arraysize", "description", "rowcount", "lastrowid", "connection",
    "row_factory",
):
    _guard_property(SerializedCursor, _property, cursor=True)


def operation_scope(conn: sqlite3.Connection):
    """Hold the connection lock across a multi-statement transaction body."""
    return getattr(conn, "_sqlite_lock", contextlib.nullcontext())
