"""Stage-0 sizing probe for the digest profile squeeze
(`benchmarks/digest_squeeze_probe.py`).

`_anchor_facts` (`aggregate.py:801-836`) budgets profile rows and graph edges
from ONE shared cap and returns early once profile rows fill it, so a store
whose active profile has grown past `aggregation_digest_anchor_facts` injects
ZERO graph edges into the root digest. The probe's headline is therefore a
COUNTERFACTUAL DIFF -- the block as rendered today against the block the
separate-budget fix would render -- not a row count.

The design test is `test_a_squeezed_store_restores_every_active_edge`. The
four verdict branches are degeneracy guards, not a bar: the box outcome is
already known, so "does it change anything" would be a ceiling instrument.
Each branch is asserted on its VALUE (counts and lines), never on the verdict
string alone, because a verdict string can be right for the wrong reason.

`test_the_fixed_arm_renders_the_same_lines_when_the_cap_binds_nothing` is the
parity control: the CURRENT arm is production `_anchor_facts` itself (no copy),
so the only thing that can drift is the FIXED arm's rendering and ordering, and
that test pins it against production on a store where the two must agree.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from hymem import HyMem, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.dreaming.aggregate import _anchor_facts
from hymem.extraction.llm import StubLLMClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from digest_squeeze_probe import (  # noqa: E402
    SnapshotMoved,
    _data_version,
    _json_payload,
    _render,
    _verdict,
    fixed_facts,
    main,
    measure_snapshot,
    measure_squeeze,
)


# ── seeding helpers (the tests/test_recovery_probe.py raw-SQL idiom) ─────────

def _seed_profile(conn, slot: str, value: str, *, slot_key: str | None = None,
                  invalid_at: str | None = None) -> None:
    conn.execute(
        "INSERT INTO user_profile(slot, slot_key, value, confidence, invalid_at) "
        "VALUES (?, ?, ?, 1.0, ?)",
        (slot, slot_key, value, invalid_at),
    )


def _seed_edge(conn, subject: str, predicate: str = "uses",
               obj: str = "postgres", *, pos: int = 3, neg: int = 0,
               derived: int = 0, status: str = "active",
               invalid_at: str | None = None) -> None:
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, "
        "object_canonical, pos_evidence, neg_evidence, first_seen, last_seen, "
        "last_reinforced, invalid_at, status, derived) "
        "VALUES (?, ?, ?, ?, ?, '2024-01-01 00:00:00', CURRENT_TIMESTAMP, "
        "CURRENT_TIMESTAMP, ?, ?, ?)",
        (subject, predicate, obj, pos, neg, invalid_at, status, derived),
    )


def _profile_rows(conn, n: int, *, first: int = 0) -> None:
    """`n` ACTIVE profile rows. 'possession' is outside SINGLE_VALUED_SLOTS
    (user_profile.py:80), which is exactly why a real profile accumulates past
    the cap in the first place."""
    for i in range(first, first + n):
        _seed_profile(conn, "possession", f"thing{i:02d}")


def _edges(conn, n: int, *, first: int = 0) -> None:
    for i in range(first, first + n):
        _seed_edge(conn, f"svc{i:02d}")


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    yield hy.conn
    hy.close()


# ── the counterfactual diff ──────────────────────────────────────────────────

def test_a_squeezed_store_restores_every_active_edge(conn):
    """THE design test: the box's shape (profile past the cap). Today the block
    is profile-only and the edge budget is 0; the fix restores every
    anchor-eligible edge. Values, not the verdict string, carry this."""
    with core_db.transaction(conn):
        _profile_rows(conn, 21)
        _edges(conn, 5)

    report = measure_squeeze(conn, cap=20)

    assert report["n_profile_active"] == 21
    assert report["edge_budget_today"] == 0
    assert report["n_edges_active"] == 5
    assert report["n_current_facts"] == 20          # 20 profile lines, 0 edges
    assert report["edges_restored"] == 5
    assert report["verdict"] == "SQUEEZED"
    # And the current block really does hold no edge line at all.
    assert all(line.startswith("user ") for line in _anchor_facts(conn, 20))


def test_the_fixed_arm_renders_the_same_lines_when_the_cap_binds_nothing(conn):
    """PARITY CONTROL. The CURRENT arm is production `_anchor_facts` itself, so
    the only thing that can drift is the FIXED arm's own rendering and ordering.
    On a store the cap does not bind, the two must agree line for line -- and
    they are produced by different code paths, so this is not a tautology."""
    with core_db.transaction(conn):
        _seed_profile(conn, "name", "Atta")
        _seed_profile(conn, "role", "bedrijfsarts")
        _seed_edge(conn, "atta", "part_of", "medflow", pos=9)
        _seed_edge(conn, "medflow", "uses", "postgres", pos=5)

    current = _anchor_facts(conn, 20)
    assert current == [
        "user name Atta", "user role bedrijfsarts",
        "atta part_of medflow", "medflow uses postgres",
    ]
    assert fixed_facts(conn, edge_cap=20, profile_cap=20) == current
    assert measure_squeeze(conn, cap=20)["edges_restored"] == 0


def test_ineligible_edges_are_never_restored(conn):
    """The fix restores the ANCHOR-eligible population only. Derived,
    margin-negative, retracted and invalid_at-stamped edges are excluded in
    BOTH arms -- otherwise `edges_restored` reports the size of the graph."""
    with core_db.transaction(conn):
        _profile_rows(conn, 21)
        _seed_edge(conn, "good")
        _seed_edge(conn, "derived", derived=1)
        _seed_edge(conn, "margin", pos=1, neg=4)
        _seed_edge(conn, "gone", status="retracted")
        _seed_edge(conn, "stamped", invalid_at="2024-02-01 00:00:00")

    report = measure_squeeze(conn, cap=20)

    assert report["n_edges_active"] == 1
    # active + non-derived, but no margin/invalid_at filter: good, margin, stamped
    assert report["n_edges_active_total"] == 3
    assert report["edges_restored"] == 1
    assert report["restored_lines"] == ["good uses postgres"]


# ── guard branches (ordered; each asserts a value, not just a label) ─────────

def test_no_active_edges_reads_vacuous(conn):
    """GUARD 1, and it outranks SQUEEZED: a store with no anchor-eligible edge
    cannot show restoration, so a 0 diff there is arithmetic, not evidence."""
    with core_db.transaction(conn):
        _profile_rows(conn, 21)
        _seed_edge(conn, "derived", derived=1)       # active, but never eligible

    report = measure_squeeze(conn, cap=20)

    assert report["n_edges_active"] == 0
    assert report["edges_restored"] == 0
    assert report["edge_budget_today"] == 0          # also squeezed: guard order
    assert report["verdict"] == "VACUOUS"


def test_a_zero_profile_store_diffs_exactly_zero(conn):
    """GUARD 2 -- the plan's genuine zero-diff control, which exists only as a
    fixture (neither real store is one). With no profile rows the two arms must
    be byte-identical; anything else means the probe measures something other
    than the squeeze."""
    with core_db.transaction(conn):
        _edges(conn, 5)

    report = measure_squeeze(conn, cap=20)

    assert report["n_profile_active"] == 0
    assert report["edges_restored"] == 0
    assert report["restored_lines"] == []
    assert fixed_facts(conn, edge_cap=20, profile_cap=20) == _anchor_facts(conn, 20)
    assert report["verdict"] == "ZERO-PROFILE"


def test_invalidated_profile_rows_do_not_count_as_active(conn):
    """The zero-profile control must key on ACTIVE rows: a store whose profile
    is entirely invalidated is a zero-profile store."""
    with core_db.transaction(conn):
        _seed_profile(conn, "role", "old", invalid_at="2024-02-01 00:00:00")
        _edges(conn, 3)

    report = measure_squeeze(conn, cap=20)

    assert report["n_profile_active"] == 0
    assert report["verdict"] == "ZERO-PROFILE"


def test_a_store_under_the_cap_is_not_squeezed(conn):
    """GUARD 3 -- conv-26's shape (16 rows against cap 20). NOT-SQUEEZED means
    'not total exclusion', NOT 'no diff': the 4-slot budget still costs this
    store 16 edges. Asserting the non-zero value is what stops the branch being
    read as a null control."""
    with core_db.transaction(conn):
        _profile_rows(conn, 16)
        _edges(conn, 20)

    report = measure_squeeze(conn, cap=20)

    assert report["n_profile_active"] == 16
    assert report["edge_budget_today"] == 4
    assert report["edges_restored"] == 16            # NOT a null control
    assert report["verdict"] == "NOT-SQUEEZED"


def test_cap_zero_reports_a_disabled_block(conn):
    """GUARD 0. `0 disables` (config.py:197, pinned by test_aggregate.py:1031):
    with the block off there is no squeeze to size, and reporting SQUEEZED off
    an edge budget of 0 would be the degenerate-criterion trap."""
    with core_db.transaction(conn):
        _profile_rows(conn, 21)
        _edges(conn, 5)

    report = measure_squeeze(conn, cap=0, edge_cap=0, profile_cap=0)

    assert _anchor_facts(conn, 0) == []
    assert report["n_current_facts"] == 0
    assert report["n_fixed_facts"] == 0
    assert report["edges_restored"] == 0
    assert report["verdict"] == "DISABLED"


def test_every_verdict_branch_is_reachable():
    """No branch may be dead code (the unreachable-code-path trap: a path that
    never runs reads as PASS). Each verdict is produced from a report shape a
    real store can have."""
    def rep(**kw):
        base = {"cap": 20, "n_profile_active": 0, "n_edges_active": 5,
                "edge_budget_today": 20, "edges_restored": 0}
        return {**base, **kw}

    assert _verdict(rep(cap=0))[0] == "DISABLED"
    assert _verdict(rep(n_edges_active=0))[0] == "VACUOUS"
    assert _verdict(rep())[0] == "ZERO-PROFILE"
    assert _verdict(rep(n_profile_active=16, edge_budget_today=4,
                        edges_restored=16))[0] == "NOT-SQUEEZED"
    assert _verdict(rep(n_profile_active=22, edge_budget_today=0,
                        edges_restored=5))[0] == "SQUEEZED"


# ── the separate, still-open defect: the dropped profile tail ────────────────

def test_profile_tail_dropped_is_reported_and_never_restored(conn):
    """`load_profile(conn, cap=20)` truncates a 22-row profile to 20
    (`user_profile.py:343`). The separate-budget fix does NOT repair that: the
    FIXED arm still renders only `profile_cap` profile lines. So
    `profile_tail_dropped` is a REPORTING figure for a still-open defect and
    must never be folded into a restored count."""
    with core_db.transaction(conn):
        _profile_rows(conn, 22)
        _edges(conn, 5)

    report = measure_squeeze(conn, cap=20)

    assert report["n_profile_active"] == 22
    assert report["n_profile_rendered"] == 20
    assert report["profile_tail_dropped"] == 2
    # Still 20 profile lines in the fixed arm, and the tail is NOT restored.
    fixed = fixed_facts(conn, edge_cap=20, profile_cap=20)
    assert sum(1 for line in fixed if line.startswith("user ")) == 20
    assert report["edges_restored"] == 5             # edges only, tail excluded
    assert report["profile_lines_restored"] == 0


# ── snapshot integrity on a live WAL store ──────────────────────────────────

def test_a_concurrent_write_moves_data_version(cfg, conn):
    """MECHANISM test for the abort branch: prove `PRAGMA data_version` on a
    READ-ONLY connection actually moves when another connection commits. The
    branch is worthless if the signal does not fire."""
    with core_db.transaction(conn):
        _edges(conn, 1)
    ro = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        before = _data_version(ro)
        with core_db.transaction(conn):
            _edges(conn, 1, first=1)
        assert _data_version(ro) != before
    finally:
        ro.close()


def test_a_moved_snapshot_aborts_the_measurement(conn, monkeypatch):
    """And the branch consumes it: a snapshot that moved mid-read raises rather
    than returning a torn reading."""
    import digest_squeeze_probe as probe

    versions = iter([1, 2])
    monkeypatch.setattr(probe, "_data_version", lambda c: next(versions))
    with pytest.raises(SnapshotMoved):
        measure_snapshot(conn, cap=20)


def test_a_still_snapshot_measures_normally(conn):
    """Control for the abort: with nothing writing, the same call path returns
    the report (otherwise the guard could pass by always aborting)."""
    with core_db.transaction(conn):
        _profile_rows(conn, 21)
        _edges(conn, 5)

    assert measure_snapshot(conn, cap=20)["edges_restored"] == 5


# ── house rule: never write or echo raw fact text ────────────────────────────

def test_the_json_payload_carries_no_fact_text(conn, tmp_path):
    """HOUSE RULE. `recovery_probe --json` writes raw fact text to disk; this
    probe deliberately does not -- profile and graph lines are the user's real
    conversation content. The payload is counts and verdict only."""
    with core_db.transaction(conn):
        _profile_rows(conn, 21)
        _seed_edge(conn, "distinctivesubject", "uses", "distinctiveobject")

    payload = _json_payload(measure_squeeze(conn, cap=20))
    blob = json.dumps(payload)

    assert "distinctivesubject" not in blob
    assert "distinctiveobject" not in blob
    assert "thing00" not in blob
    assert "restored_lines" not in payload
    # An allow-list, so a later text field cannot slip in unnoticed.
    assert all(isinstance(v, (int, str, type(None))) for v in payload.values())


def test_restored_lines_are_withheld_from_the_render_by_default(conn):
    """Same rule for stdout: an agent-run invocation must not surface fact text.
    `--show-restored` is the human's explicit opt-in."""
    with core_db.transaction(conn):
        _profile_rows(conn, 21)
        _seed_edge(conn, "distinctivesubject", "uses", "distinctiveobject")

    report = measure_squeeze(conn, cap=20)

    quiet = _render(report, path="store.sqlite", show_restored=False)
    assert "distinctivesubject" not in quiet
    assert "1" in quiet                                  # the count still shows
    loud = _render(report, path="store.sqlite", show_restored=True)
    assert "distinctivesubject uses distinctiveobject" in loud


# ── the CLI ─────────────────────────────────────────────────────────────────

def test_main_runs_read_only_and_writes_only_counts(cfg, conn, tmp_path, capsys):
    with core_db.transaction(conn):
        _profile_rows(conn, 21)
        _seed_edge(conn, "distinctivesubject", "uses", "distinctiveobject")
    out_json = tmp_path / "squeeze.json"

    rc = main([str(cfg.db_path), "--json", str(out_json)])

    assert rc == 0
    assert capsys.readouterr().out.count("distinctivesubject") == 0
    payload = json.loads(out_json.read_text())
    assert payload["verdict"] == "SQUEEZED"
    assert payload["edges_restored"] == 1
    assert "distinctivesubject" not in out_json.read_text()


def test_main_aborts_on_a_moved_snapshot(cfg, conn, monkeypatch, capsys):
    import digest_squeeze_probe as probe

    with core_db.transaction(conn):
        _edges(conn, 3)
    versions = iter([1, 2])
    monkeypatch.setattr(probe, "_data_version", lambda c: next(versions))

    assert main([str(cfg.db_path)]) == 2
    assert "data_version" in capsys.readouterr().err
