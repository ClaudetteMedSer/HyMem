"""Stage-1 instrument for the digest profile squeeze
(`benchmarks/digest_squeeze_dump.py`).

The dump renders BOTH root-fusion prompts -- the block as `_anchor_facts`
builds it today, and the block the separate-budget fix would build -- over the
SAME tree on the SAME snapshot, so a human can paste them into the box LLM and
hand-score S1-C1..C4. It is read-only and LLM-*less*, exactly like
`benchmarks/profile_prompt_dump.py`: it prints, it never calls a model, and it
never writes a file.

The load-bearing test is
`test_the_current_arm_reproduces_the_prompt_production_actually_sent`: the
CURRENT arm is compared byte for byte against the prompt a real
`build_aggregation_nodes` dream handed the LLM on the same store. Without that,
"same tree, same snapshot" is an assertion about code the test never runs, and
the hand-scoring would be scoring a prompt production never sends.

The guard tests exist because an empty prompt, a torn snapshot and two
identical arms all read as a clean PASS to a human scorer.
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.core import db as core_db
from hymem.dreaming.aggregate import build_aggregation_nodes
from hymem.extraction.llm import StubLLMClient
from tests.test_aggregate import _agg_llm, _seed_episode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from digest_squeeze_dump import (  # noqa: E402
    DUMP_MODULE_PATH,
    NoRootDigest,
    build_arms,
    main,
    root_items,
)
from digest_squeeze_probe import SnapshotMoved  # noqa: E402

from tests.test_digest_squeeze_probe import (  # noqa: E402
    _profile_rows,
    _seed_edge,
    _seed_profile,
)


def _dreamed(cfg, conn, *, profile_rows: int = 0, edges: int = 0):
    """A store with a real root digest over a two-level tree: one level-0 node
    (two entity-linked cross-session episodes) plus one pass-through episode,
    so BOTH member-resolution paths (aggregation_nodes and episodes) are
    exercised. Returns the stub LLM so the test can read the prompt production
    actually sent."""
    acfg = replace(cfg, aggregation_nodes_enabled=True,
                   aggregation_digest_enabled=True)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Billing sync",
                      "Postgres billing sync work.", ["postgres", "billing"])
        _seed_episode(conn, "e2", "s2", "Billing retries",
                      "Retries on the billing sync.", ["postgres", "billing"])
        _seed_episode(conn, "e3", "s3", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"])
        if profile_rows:
            _profile_rows(conn, profile_rows)
        for i in range(edges):
            _seed_edge(conn, f"svc{i:02d}", "uses", "postgres")
    llm = _agg_llm()
    build_aggregation_nodes(conn, acfg, llm, None)
    return acfg, llm


@pytest.fixture
def conn(cfg):
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"),
               embedding_client=StubEmbeddingClient())
    yield hy.conn
    hy.close()


# ── fidelity: the dump must render what production renders ──────────────────

def test_the_current_arm_reproduces_the_prompt_production_actually_sent(cfg, conn):
    """THE binding test. Same tree, same snapshot is worthless as a claim: the
    CURRENT prompt is compared byte for byte against the user message a real
    dream handed the LLM. Both member-resolution paths (a level-0 node and a
    pass-through episode) are in the tree."""
    _, llm = _dreamed(cfg, conn, profile_rows=21, edges=5)
    sent = [c for c in llm.calls
            if "standing digest of everything known" in c.system]
    assert len(sent) == 1

    arms = build_arms(conn, cap=20)

    assert arms["current_prompt"] == sent[0].user


def test_the_tree_is_reconstructed_in_the_persisted_member_order(cfg, conn):
    """Member order IS fusion-input order, and `_items_text` joins in that
    order, so a reconstruction that reordered members would render a different
    prompt while still 'containing the same members'."""
    _, llm = _dreamed(cfg, conn, profile_rows=21, edges=5)
    root = conn.execute(
        "SELECT member_episode_ids FROM aggregation_nodes WHERE is_root = 1"
    ).fetchone()

    items, missing = root_items(conn)

    assert missing == []
    assert [i["id"] for i in items] == json.loads(root["member_episode_ids"])
    assert len(items) == 2                       # one level-0 node + one leftover


# ── the two arms ────────────────────────────────────────────────────────────

def test_both_arms_share_a_byte_identical_summaries_half(cfg, conn):
    """The whole design: only the VERIFIED FACTS block may differ. If the
    summaries half moved too, a scored difference could not be attributed to
    the squeeze."""
    _dreamed(cfg, conn, profile_rows=21, edges=5)

    arms = build_arms(conn, cap=20)

    assert arms["current_prompt"] != arms["fixed_prompt"]
    assert arms["text"] in arms["current_prompt"]
    assert arms["text"] in arms["fixed_prompt"]
    # Everything outside the facts block is identical, character for character.
    assert (arms["current_prompt"].replace(arms["current_facts"], "@@")
            == arms["fixed_prompt"].replace(arms["fixed_facts"], "@@"))


def test_the_fixed_arm_carries_the_restored_edges_and_the_current_one_does_not(
    cfg, conn
):
    """The dump must actually EXHIBIT the squeeze, not merely be capable of it:
    on the box's shape the current facts block holds no edge line at all."""
    _dreamed(cfg, conn, profile_rows=21, edges=5)

    arms = build_arms(conn, cap=20)

    assert "svc00 uses postgres" not in arms["current_facts"]
    assert "svc00 uses postgres" in arms["fixed_facts"]
    assert arms["edges_restored"] == 5
    assert arms["current_facts"].count("\n") + 1 == 20      # 20 profile lines


def test_a_disabled_anchor_renders_the_none_placeholder(cfg, conn):
    """`0 disables` end to end: the block production sends is the literal
    "(none)" (aggregate.py:1240), so the dump must render that, not "".
    """
    acfg = replace(cfg, aggregation_nodes_enabled=True,
                   aggregation_digest_enabled=True,
                   aggregation_digest_anchor_facts=0)
    with core_db.transaction(conn):
        _seed_episode(conn, "e1", "s1", "Weekend cycling",
                      "Started cycling on weekends.", ["cycling"])
        _profile_rows(conn, 21)
    llm = _agg_llm()
    build_aggregation_nodes(conn, acfg, llm, None)
    sent = [c for c in llm.calls
            if "standing digest of everything known" in c.system][0]

    arms = build_arms(conn, cap=0, edge_cap=0, profile_cap=0)

    assert arms["current_facts"] == "(none)"
    assert arms["current_prompt"] == sent.user


# ── guards: an empty, torn or degenerate dump reads as a clean PASS ─────────

def test_a_store_with_no_root_is_an_error_not_an_empty_prompt(cfg, conn, capsys):
    """A human handed an empty prompt would score a confident constant. The
    dump refuses instead, and prints no prompt at all."""
    # The message must name THIS branch: the sibling guard below also raises
    # NoRootDigest and also returns 1, so a bare `raises` would let either
    # branch be deleted without a test noticing.
    with pytest.raises(NoRootDigest, match="no root digest node"):
        build_arms(conn, cap=20)

    assert main([str(cfg.db_path)]) == 1
    out = capsys.readouterr()
    assert "VERIFIED FACTS" not in out.out
    assert "no root digest" in out.err.lower()


def test_a_root_whose_members_all_vanished_is_an_error(cfg, conn, capsys):
    """Same trap one level down: a root that resolves to zero members renders
    an empty summaries half. Refused, not printed."""
    _dreamed(cfg, conn, profile_rows=21, edges=5)
    with core_db.transaction(conn):
        conn.execute("UPDATE aggregation_nodes SET member_episode_ids = ? "
                     "WHERE is_root = 1", (json.dumps(["ghost1", "ghost2"]),))

    with pytest.raises(NoRootDigest, match="resolves to no members"):
        build_arms(conn, cap=20)
    assert main([str(cfg.db_path)]) == 1
    assert "VERIFIED FACTS" not in capsys.readouterr().out


def test_unresolvable_members_are_counted_not_silently_dropped(cfg, conn, capsys):
    """A root kept through a failed fusion can point at replaced nodes. The
    dump reports the shortfall in its header so the scorer knows the summaries
    half is incomplete, rather than shrinking it silently."""
    _dreamed(cfg, conn, profile_rows=21, edges=5)
    root = conn.execute(
        "SELECT member_episode_ids FROM aggregation_nodes WHERE is_root = 1"
    ).fetchone()
    members = json.loads(root["member_episode_ids"]) + ["ghost"]
    with core_db.transaction(conn):
        conn.execute("UPDATE aggregation_nodes SET member_episode_ids = ? "
                     "WHERE is_root = 1", (json.dumps(members),))

    items, missing = root_items(conn)
    assert missing == ["ghost"]

    assert main([str(cfg.db_path)]) == 0
    assert "unresolved member" in capsys.readouterr().out


def test_identical_facts_blocks_are_refused(cfg, conn, capsys):
    """DEGENERACY guard. On a zero-profile store the two arms are identical;
    a human scoring two identical prompts produces a confident constant, and
    S1-C1 would read 0 for a reason that has nothing to do with the fix."""
    _dreamed(cfg, conn, profile_rows=0, edges=5)

    arms = build_arms(conn, cap=20)
    assert arms["current_facts"] == arms["fixed_facts"]

    assert main([str(cfg.db_path)]) == 3
    out = capsys.readouterr()
    assert "VERIFIED FACTS" not in out.out
    assert "identical" in out.err.lower()


def test_a_moved_snapshot_aborts_the_dump(cfg, conn, monkeypatch, capsys):
    """The box store is live WAL: the two arms must come off ONE snapshot, or
    the difference the human scores is partly a dream landing mid-dump."""
    import digest_squeeze_dump as dump

    _dreamed(cfg, conn, profile_rows=21, edges=5)
    versions = iter([1, 2])
    monkeypatch.setattr(dump, "_data_version", lambda c: next(versions))

    assert main([str(cfg.db_path)]) == 2
    out = capsys.readouterr()
    assert "VERIFIED FACTS" not in out.out
    assert "data_version" in out.err
    assert SnapshotMoved is dump.SnapshotMoved       # the probe's, not a copy


# ── the criteria travel with the prompts ────────────────────────────────────

def test_the_banked_criteria_are_printed_with_the_prompts(cfg, conn, capsys):
    """The bars were banked before any number existed; printing them on the
    page the human scores from is what stops them being re-read afterwards."""
    _dreamed(cfg, conn, profile_rows=21, edges=5)

    assert main([str(cfg.db_path)]) == 0
    out = capsys.readouterr().out

    for criterion in ("S1-C1", "S1-C2", "S1-C3", "S1-C4"):
        assert criterion in out
    assert ">=3 of 10" in out or "3 of 10" in out
    assert "ARM A" in out and "ARM B" in out


# ── house rules ─────────────────────────────────────────────────────────────

def test_the_dump_imports_no_llm_client(cfg):
    """HOUSE RULE, by construction: Stage 1 is hand-scored on the box. A dump
    that could call a model would invite an agent to run the gate for the
    human -- and the only backends this repo may ship are the Protocol and
    StubLLMClient."""
    tree = ast.parse(Path(DUMP_MODULE_PATH).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [node.module or ""] + [a.name for a in node.names]

    assert imported, "the AST walk found no imports at all -- vacuous test"
    offenders = [name for name in imported
                 if "llm" in name.lower() or "client" in name.lower()]
    assert offenders == [], f"the Stage-1 dump must stay LLM-less: {offenders}"


def test_the_dump_writes_no_files(cfg, conn, tmp_path, monkeypatch, capsys):
    """HOUSE RULE: the rendered prompts are real conversation content. They go
    to the human's terminal and nowhere else -- no artifact file, no --json,
    no log."""
    _dreamed(cfg, conn, profile_rows=21, edges=5)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    store_dir_before = sorted(p.name for p in cfg.db_path.parent.iterdir())

    assert main([str(cfg.db_path)]) == 0
    capsys.readouterr()

    assert list(workdir.iterdir()) == []
    assert sorted(p.name for p in cfg.db_path.parent.iterdir()) == store_dir_before


def test_the_dump_opens_the_store_read_only(cfg, conn, capsys):
    """Read-only in the `recovery_probe` sense: a stray write raises instead of
    mutating the live store."""
    import digest_squeeze_dump as dump

    _dreamed(cfg, conn, profile_rows=21, edges=5)
    ro = dump.open_store_readonly(cfg.db_path)
    try:
        with pytest.raises(Exception):
            ro.execute("DELETE FROM aggregation_nodes")
    finally:
        ro.close()
