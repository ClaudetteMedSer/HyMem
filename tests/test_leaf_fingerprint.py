"""The leftover-displacement watermark (schema v30).

`aggregation_leaf_changed` is the third term of the deficit model
`rebuilt ~ A*level0_missed + root_term + leaf_term`. It used to be derived from
a module global holding the previous dream's leaf set, which made it readable
only on the SECOND and later aggregation dreams of one process. The box starts
a fresh process per dream, so the 2026-08-08 read found 175 of 187 rows NULL:
the channel was not under-sampled, it was unobservable, and no amount of
waiting would have produced a reading.

The tests that matter here are the ones a process-local implementation FAILS —
`test_leaf_changed_survives_a_process_restart` and
`test_a_cleared_watermark_reads_unattributed_not_unchanged`. Both drive the
comparison through the store rather than through memory, so neither can pass
by accident on the old code. The rest pin the semantics around them: NULL means
unattributed and never a counterfeit fixed point, the watermark only advances
with the nodes it belongs to, and a disabled digest leaves it alone.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hymem import HyMem, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.dreaming import aggregate as agg_mod
from hymem.dreaming.aggregate import _leaf_fingerprint, build_aggregation_nodes
from hymem.extraction.llm import StubLLMClient

_NODE_JSON = json.dumps({"title": "Postgres", "summary": "Postgres everywhere."})
_ROLLUP_JSON = json.dumps({"title": "Mixed", "summary": "Several threads."})
_DIGEST_JSON = json.dumps({"title": "Digest", "summary": "Everything known."})


def _agg_llm() -> StubLLMClient:
    return StubLLMClient(
        fixtures={
            "fuse several related episodes": _NODE_JSON,
            "combined summary that loses no thread": _ROLLUP_JSON,
            "standing digest of everything known": _DIGEST_JSON,
        },
        default="[]",
    )


def _digest_cfg(cfg):
    return replace(cfg, aggregation_nodes_enabled=True,
                   aggregation_digest_enabled=True)


def _seed_episode(conn, eid, sid, entities):
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (sid,))
    conn.execute(
        """INSERT INTO episodes(id, session_id, title, summary, participants,
                                start_message_id, end_message_id, outcome, key_entities)
           VALUES (?, ?, ?, ?, '[]', 1, 2, NULL, ?)""",
        (eid, sid, f"Topic {eid}", f"Notes about {eid}.", json.dumps(entities)),
    )


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    yield hy.conn
    hy.close()


def _seed(conn, n: int = 3) -> None:
    """Disjoint singletons: `select_clusters` keeps none, so every episode
    becomes a digest LEAF — which is exactly the set this watermark tracks."""
    with core_db.transaction(conn):
        for i in range(1, n + 1):
            _seed_episode(conn, f"e{i}", f"s{i}", [f"topic{i}"])


def _dream(conn, cfg):
    return build_aggregation_nodes(conn, _digest_cfg(cfg), _agg_llm(), None)


def _stored(conn):
    return conn.execute(
        "SELECT fingerprint, n_leaves FROM aggregation_leaf_state WHERE id = 1"
    ).fetchone()


# ── the two tests a process-local implementation cannot pass ────────────────

def test_leaf_changed_survives_a_process_restart(conn, cfg):
    """THE regression. A second dream against an unchanged leaf set must read
    0, and it must read it from the STORE — so the assertion is made after
    overwriting the persisted fingerprint out from under an in-memory copy.

    With the old module global, step 3 returns 0 (memory still holds the real
    previous set and it matches). Reading the watermark from the store returns
    1, because the store says the last persisted leaf set was different. That
    divergence is the whole point: it proves the comparison crosses a process
    boundary rather than living in this one.
    """
    _seed(conn)
    assert _dream(conn, cfg).leaf_changed is None      # first ever: unattributed
    assert _dream(conn, cfg).leaf_changed == 0         # unchanged leaf set

    with core_db.transaction(conn):
        conn.execute(
            "UPDATE aggregation_leaf_state SET fingerprint = 'from-another-process' "
            "WHERE id = 1"
        )
    assert _dream(conn, cfg).leaf_changed == 1


def test_a_cleared_watermark_reads_unattributed_not_unchanged(conn, cfg):
    """No stored predecessor means NULL, never 0. `level0_missed = 0` together
    with `leaf_changed = 0` IS the fixed-point signature, so a missing reading
    that defaulted to 0 would manufacture fixed points — the counterfeit the
    029 header warns about. A process-local implementation answers 0 here.
    """
    _seed(conn)
    _dream(conn, cfg)
    with core_db.transaction(conn):
        conn.execute("DELETE FROM aggregation_leaf_state")
    assert _dream(conn, cfg).leaf_changed is None


# ── semantics around the watermark ─────────────────────────────────────────

def test_first_dream_writes_the_watermark_it_could_not_read(conn, cfg):
    _seed(conn, n=3)
    assert _stored(conn) is None
    assert _dream(conn, cfg).leaf_changed is None
    row = _stored(conn)
    assert row is not None
    assert row["n_leaves"] == 3
    assert row["fingerprint"] == _leaf_fingerprint(frozenset({"e1", "e2", "e3"}))


def test_a_moved_leaf_set_reads_changed(conn, cfg):
    _seed(conn, n=3)
    _dream(conn, cfg)
    with core_db.transaction(conn):
        _seed_episode(conn, "e4", "s4", ["topic4"])
    result = _dream(conn, cfg)
    assert result.leaf_changed == 1
    assert _stored(conn)["n_leaves"] == 4


def test_fingerprint_is_order_independent(conn, cfg):
    assert (_leaf_fingerprint(frozenset({"a", "b"}))
            == _leaf_fingerprint(frozenset({"b", "a"})))
    assert (_leaf_fingerprint(frozenset({"a", "b"}))
            != _leaf_fingerprint(frozenset({"a", "b", "c"})))


def test_disabled_digest_reports_unattributed_and_leaves_the_row_alone(conn, cfg):
    """With no digest there are no leaves and no leaf term. The result reports
    NULL (the -1 sentinel is a log-only value), and an existing watermark from
    a digest-enabled dream must not be clobbered by a run that never looked at
    a leaf set."""
    _seed(conn)
    _dream(conn, cfg)
    before = _stored(conn)["fingerprint"]

    nodigest = replace(cfg, aggregation_nodes_enabled=True,
                       aggregation_digest_enabled=False)
    assert build_aggregation_nodes(conn, nodigest, _agg_llm(), None).leaf_changed is None
    assert _stored(conn)["fingerprint"] == before


def test_watermark_is_written_inside_the_persist_transaction(conn, cfg, monkeypatch):
    """It must commit with the nodes that consumed the leaf set. A watermark
    that advanced outside the transaction would, after a dream that died before
    persisting, report the NEXT dream's genuine displacement as unchanged —
    silently zeroing the channel this table exists to measure."""
    seen: list[bool] = []
    real = agg_mod._write_leaf_fingerprint

    def spy(conn_, fingerprint, n_leaves):
        seen.append(conn_.in_transaction)
        return real(conn_, fingerprint, n_leaves)

    monkeypatch.setattr(agg_mod, "_write_leaf_fingerprint", spy)
    _seed(conn)
    _dream(conn, cfg)
    assert seen == [True]
