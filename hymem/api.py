from __future__ import annotations

import sqlite3
from typing import Iterable

from hymem import session as session_log
from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import canonicalize as canon
from hymem.dreaming.runner import DreamReport, run_dreaming
from hymem.extraction.llm import LLMClient
from hymem.query.augment import AugmentedContext, augment


class HyMem:
    """Public API for the Hermes host.

    Hermes is responsible for:
      - constructing one HyMem per agent process (or per project root),
      - calling `log_message` for every conversational turn,
      - calling `augment` before sending a user turn to its model,
      - calling `dream` during idle windows (or via its own scheduler),
      - providing an LLMClient — required only for `dream`, not for `augment`.
    """

    def __init__(self, config: HyMemConfig, *, llm: LLMClient | None = None) -> None:
        self.config = config
        self._llm = llm
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    # ---- lifecycle ---------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = core_db.connect(self.config.db_path)
        if not self._initialized:
            core_db.initialize(self._conn)
            self._initialized = True
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False

    def set_llm(self, llm: LLMClient) -> None:
        self._llm = llm

    # ---- session log -------------------------------------------------

    def open_session(self, session_id: str) -> None:
        with core_db.transaction(self.conn):
            session_log.open_session(self.conn, session_id)

    def close_session(self, session_id: str) -> None:
        with core_db.transaction(self.conn):
            session_log.close_session(self.conn, session_id)

    def log_message(self, session_id: str, role: str, content: str) -> int:
        with core_db.transaction(self.conn):
            session_log.open_session(self.conn, session_id)
            return session_log.append_message(self.conn, session_id, role, content)

    # ---- query-time --------------------------------------------------

    def augment(self, user_message: str) -> AugmentedContext:
        return augment(self.conn, self.config, user_message)

    # ---- dreaming ----------------------------------------------------

    def dream(self, *, session_ids: Iterable[str] | None = None) -> DreamReport:
        if self._llm is None:
            raise RuntimeError(
                "HyMem.dream requires an LLMClient. Pass one to the constructor "
                "or call set_llm() before dreaming."
            )
        ids = list(session_ids) if session_ids is not None else None
        return run_dreaming(self.conn, self.config, self._llm, session_ids=ids)

    # ---- maintenance -------------------------------------------------

    def register_alias(self, surface: str, canonical: str) -> None:
        with core_db.transaction(self.conn):
            canon.register_alias(self.conn, surface, canonical)

    def merge_canonical(self, keep: str, drop: str) -> None:
        with core_db.transaction(self.conn):
            canon.merge(self.conn, keep, drop)
