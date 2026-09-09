"""Explicit historical-row admission for fixtures, never production bypasses."""
from contextlib import contextmanager

from hymem.core import db


@contextmanager
def legacy_canonical_rows(conn):
    """Temporarily remove only v60 guards; restore them before exercising code."""
    for table in (
        "entity_aliases", "knowledge_graph", "entity_mentions",
    ):
        for operation in ("insert", "update"):
            conn.execute(
                f"DROP TRIGGER IF EXISTS {table}_canonical_{operation}_guard"
            )
    try:
        yield
    finally:
        db._install_canonical_write_guards(conn)
