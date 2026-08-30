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

    def spy(conn_, fingerprint, n_leaves, leaf_ids=None):
        seen.append(conn_.in_transaction)
        return real(conn_, fingerprint, n_leaves, leaf_ids)

    monkeypatch.setattr(agg_mod, "_write_leaf_fingerprint", spy)
    _seed(conn)
    _dream(conn, cfg)
    assert seen == [True]


# ── v34: the SIZE of the shift, not just whether it moved ───────────────────
#
# `leaf_changed` is binary, and #1324 showed it is standing in for a continuous
# quantity: at constant level0_missed=3 the rollup term ranged 4 -> 8 -> 15-18
# across rows carrying the same flag. These tests pin the symmetric difference
# that measures it, and — more importantly — pin the two identities that make
# it self-checking, plus a control proving the self-check can actually fail.

def _leaf_state(conn):
    return conn.execute(
        "SELECT fingerprint, n_leaves, leaf_ids FROM aggregation_leaf_state "
        "WHERE id = 1"
    ).fetchone()


def test_first_dream_reports_the_delta_unattributed_not_zero(conn, cfg):
    """No predecessor id list means NULL, never (0, 0). A counterfeit zero
    would read as "the leaf set held still" on exactly the row where nothing is
    known — the same trap v29's NULL contract exists to prevent."""
    _seed(conn, n=3)
    result = _dream(conn, cfg)
    assert result.leaf_added is None
    assert result.leaf_removed is None
    assert json.loads(_leaf_state(conn)["leaf_ids"]) == ["e1", "e2", "e3"]


def test_an_unchanged_leaf_set_reads_zero_zero(conn, cfg):
    _seed(conn, n=3)
    _dream(conn, cfg)
    result = _dream(conn, cfg)
    assert (result.leaf_added, result.leaf_removed) == (0, 0)
    assert result.leaf_changed == 0


def test_an_added_leaf_is_counted_on_the_added_side(conn, cfg):
    _seed(conn, n=3)
    _dream(conn, cfg)
    with core_db.transaction(conn):
        _seed_episode(conn, "e4", "s4", ["topic4"])
    result = _dream(conn, cfg)
    assert (result.leaf_added, result.leaf_removed) == (1, 0)
    assert result.leaf_changed == 1


def test_a_removed_leaf_is_counted_on_the_removed_side(conn, cfg):
    _seed(conn, n=3)
    _dream(conn, cfg)
    with core_db.transaction(conn):
        conn.execute("DELETE FROM episodes WHERE id = 'e3'")
    result = _dream(conn, cfg)
    assert (result.leaf_added, result.leaf_removed) == (0, 1)
    assert result.leaf_changed == 1


def test_a_swap_is_the_case_n_leaves_cannot_see(conn, cfg):
    """THE test that justifies the column pair. One leaf out, one in: the set
    turned over, `n_leaves` is identical, and a count-delta channel would
    report 0. Only the symmetric difference registers it — and this is the
    shape the #1324 ladder is made of, since those rows differ by ~8 rebuilds
    while sitting at the same level0_missed."""
    _seed(conn, n=3)
    _dream(conn, cfg)
    before = _leaf_state(conn)["n_leaves"]
    with core_db.transaction(conn):
        conn.execute("DELETE FROM episodes WHERE id = 'e3'")
        _seed_episode(conn, "e9", "s9", ["topic9"])
    result = _dream(conn, cfg)
    assert (result.leaf_added, result.leaf_removed) == (1, 1)
    assert _leaf_state(conn)["n_leaves"] == before      # count says nothing
    assert result.leaf_changed == 1


def test_identity_1_the_flag_and_the_counts_agree(conn, cfg):
    """`leaf_changed == 1` IFF `added + removed > 0`, across a changed and an
    unchanged dream. Two independent routes to one comparison — hash equality
    and set difference — so a disagreement means one is broken."""
    _seed(conn, n=3)
    _dream(conn, cfg)
    for mutate in (False, True):
        if mutate:
            with core_db.transaction(conn):
                _seed_episode(conn, "e5", "s5", ["topic5"])
        r = _dream(conn, cfg)
        assert r.leaf_changed == int(r.leaf_added + r.leaf_removed > 0)


def test_identity_2_net_delta_tracks_n_leaves(conn, cfg):
    """`added - removed == n_leaves - previous n_leaves`, on a mixed change so
    the arms are not trivially equal (added 2, removed 1, net +1)."""
    _seed(conn, n=3)
    _dream(conn, cfg)
    before = _leaf_state(conn)["n_leaves"]
    with core_db.transaction(conn):
        conn.execute("DELETE FROM episodes WHERE id = 'e1'")
        _seed_episode(conn, "e7", "s7", ["topic7"])
        _seed_episode(conn, "e8", "s8", ["topic8"])
    result = _dream(conn, cfg)
    assert (result.leaf_added, result.leaf_removed) == (2, 1)
    assert result.leaf_added - result.leaf_removed == (
        _leaf_state(conn)["n_leaves"] - before
    )


def test_a_prev34_watermark_row_reads_unattributed(conn, cfg):
    """A store migrated from v30-v33 carries a fingerprint with no id list.
    That must read NULL, not `frozenset()` — an empty predecessor would make
    every leaf look newly added and manufacture a large delta out of a store
    that never moved."""
    _seed(conn, n=3)
    _dream(conn, cfg)
    with core_db.transaction(conn):
        conn.execute("UPDATE aggregation_leaf_state SET leaf_ids = NULL")
    result = _dream(conn, cfg)
    assert result.leaf_added is None
    assert result.leaf_removed is None
    assert result.leaf_changed == 0        # the fingerprint arm still reads


def test_the_disagreement_warning_can_actually_fire(conn, cfg, caplog):
    """Control for the self-check itself. A guard whose path is unreachable
    from the config under test reads clean regardless (the E3 lesson), so the
    identity-1 assertion above is only worth having if a violation is
    detectable. Forge one: leave the id list matching (delta 0) while
    corrupting the fingerprint (flag 1), and the two routes must disagree."""
    _seed(conn, n=3)
    _dream(conn, cfg)
    with core_db.transaction(conn):
        conn.execute("UPDATE aggregation_leaf_state SET fingerprint = 'forged'")
    with caplog.at_level("WARNING"):
        result = _dream(conn, cfg)
    assert (result.leaf_added, result.leaf_removed) == (0, 0)
    assert result.leaf_changed == 1
    assert "leafdelta_disagreement" in caplog.text


def test_the_delta_reaches_dream_runs(conn, cfg):
    """The columns are only worth adding if they land in the table the verdict
    rows are read from."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(dream_runs)")}
    assert {"aggregation_leaf_added", "aggregation_leaf_removed"} <= cols
