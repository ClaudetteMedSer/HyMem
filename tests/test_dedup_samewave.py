"""Same-wave (same-cycle) sibling collapse for phase-1 triple dedup.

Cross-cycle dedup (tests/test_dedup.py) only fires against edges already in
``edge_embeddings`` — i.e. minted by a PRIOR dream. Sibling phrasal variants of
the same preference that all appear in the SAME dream cycle therefore each minted
their own edge, because none was in ``edge_embeddings`` yet to dedup the others
against. Same-wave collapse fixes that by comparing a new candidate against edges
minted earlier IN THIS cycle (held in an in-memory pool the runner threads across
chunks), reusing the precomputed prepare vectors and the EXACT same three gates
as cross-cycle dedup (predicate exact, one-endpoint-shared structure, lexical
sibling, cosine >= threshold).
"""

from __future__ import annotations

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.triples import Triple


class FakeEmbedder:
    """Maps exact triple-text strings to controlled vectors so cosine is
    deterministic. Unmapped texts raise (every embedded text must be mapped)."""

    model = "fake"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self.mapping[t] for t in texts]


class TxnWatchingEmbedder(FakeEmbedder):
    """Records ``conn.in_transaction`` at every embed() call so a test can prove
    no embed ever ran under the write lock (mirrors test_dedup_delock.py)."""

    def __init__(self, conn, mapping):
        super().__init__(mapping)
        self._conn = conn
        self.in_txn_flags: list[bool] = []

    def embed(self, texts):
        self.in_txn_flags.append(self._conn.in_transaction)
        return super().embed(texts)


def _seed_chunk(hy: HyMem, chunk_id: str) -> Chunk:
    """Insert a session/message/chunk with an assistant first message (speaker
    weight 1, isolating dedup from speaker-weighting)."""
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s_sw')")
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s_sw', 'assistant', 'msg')"
    )
    mid = cur.lastrowid
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES (?, 's_sw', ?, ?, 'long_user_turn', 'text')",
        (chunk_id, mid, mid),
    )
    return Chunk(
        id=chunk_id, session_id="s_sw", start_message_id=mid,
        end_message_id=mid, salience_reason="long_user_turn", text="text",
    )


def _persist(hy, chunk, triples, embedder, in_cycle_edges):
    """Mirror the runner: prepare vectors OUTSIDE the lock, then persist with the
    shared same-wave pool threaded in."""
    ext = ChunkExtraction(triples=list(triples), markers=[])
    dedup_vectors = phase1.prepare_dedup_vectors(hy.conn, ext, hy.config, embedder)
    staged = None
    with core_db.transaction(hy.conn):
        staged = phase1.persist_chunk_results(
            hy.conn, chunk, ext,
            prompt_version=hy.config.prompt_version,
            cfg=hy.config, embedding_client=embedder,
            dedup_vectors=dedup_vectors,
            in_cycle_edges=in_cycle_edges,
        )
    if staged is not None:
        in_cycle_edges[:] = staged


def test_samewave_same_chunk_collapses(cfg):
    """Two phrasal-variant triples in the SAME chunk (no prior edge) collapse to
    one edge; their shared source chunk remains one independent proof."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy, "c1")
        # Two lexical-sibling objects with near-identical vectors (cosine 1.0).
        embedder = FakeEmbedder({
            "app prefers concise": [1.0, 0.0, 0.0, 0.0],
            "app prefers concise_mode": [1.0, 0.0, 0.0, 0.0],
        })
        pool = phase1.new_in_cycle_pool()
        _persist(
            hy, chunk,
            [Triple("app", "prefers", "concise", 1),
             Triple("app", "prefers", "concise_mode", 1)],
            embedder, pool,
        )

        # Exactly one edge for (app, prefers, *).
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='prefers'"
        ).fetchone()["c"] == 1
        # Two variants from one source must not manufacture two proofs.
        assert hy.conn.execute(
            "SELECT pos_evidence FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='prefers'"
        ).fetchone()["pos_evidence"] == 1
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM kg_evidence"
        ).fetchone()["c"] == 1
    finally:
        hy.close()


def test_samewave_cross_chunk_collapses(cfg):
    """Two variants in DIFFERENT chunks of the same cycle collapse — proves the
    in-cycle pool (with vectors) spans chunks."""
    hy = HyMem(cfg)
    try:
        c1 = _seed_chunk(hy, "c_a")
        c2 = _seed_chunk(hy, "c_b")
        pool = phase1.new_in_cycle_pool()

        e1 = FakeEmbedder({"app prefers concise": [1.0, 0.0, 0.0, 0.0]})
        _persist(hy, c1, [Triple("app", "prefers", "concise", 1)], e1, pool)

        e2 = FakeEmbedder({"app prefers concise_mode": [1.0, 0.0, 0.0, 0.0]})
        _persist(hy, c2, [Triple("app", "prefers", "concise_mode", 1)], e2, pool)

        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='prefers'"
        ).fetchone()["c"] == 1
        assert hy.conn.execute(
            "SELECT object_canonical FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='prefers'"
        ).fetchone()["object_canonical"] == "concise"
        assert hy.conn.execute(
            "SELECT pos_evidence FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='prefers'"
        ).fetchone()["pos_evidence"] == 2
    finally:
        hy.close()


def test_samewave_non_siblings_stay_separate(cfg):
    """Two NOT-sibling triples (distinct objects, low cosine) stay as TWO edges —
    no over-merging."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy, "c_neg")
        embedder = FakeEmbedder({
            "app prefers concise": [1.0, 0.0, 0.0, 0.0],
            "app prefers verbose": [0.0, 1.0, 0.0, 0.0],  # orthogonal -> cosine 0
        })
        pool = phase1.new_in_cycle_pool()
        _persist(
            hy, chunk,
            [Triple("app", "prefers", "concise", 1),
             Triple("app", "prefers", "verbose", 1)],
            embedder, pool,
        )

        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='prefers'"
        ).fetchone()["c"] == 2
    finally:
        hy.close()


def test_samewave_lexical_guard_still_applies(cfg):
    """Same-wave reuses the lexical-sibling gate: two short, distinct names
    (`redis`/`redash`) with cosine 1.0 must NOT collapse, exactly like the
    cross-cycle lexical guard."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy, "c_lex")
        embedder = FakeEmbedder({
            "app uses redis": [1.0, 0.0, 0.0, 0.0],
            "app uses redash": [1.0, 0.0, 0.0, 0.0],
        })
        pool = phase1.new_in_cycle_pool()
        _persist(
            hy, chunk,
            [Triple("app", "uses", "redis", 1),
             Triple("app", "uses", "redash", 1)],
            embedder, pool,
        )

        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='uses'"
        ).fetchone()["c"] == 2
    finally:
        hy.close()


def test_samewave_embed_never_inside_write_lock(cfg):
    """No embedding-API call occurs inside a write transaction, even with the
    broadened prepare rule that embeds every new candidate."""
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy, "c_txn")
        embedder = TxnWatchingEmbedder(hy.conn, {
            "app prefers concise": [1.0, 0.0, 0.0, 0.0],
            "app prefers concise_mode": [1.0, 0.0, 0.0, 0.0],
        })
        pool = phase1.new_in_cycle_pool()
        _persist(
            hy, chunk,
            [Triple("app", "prefers", "concise", 1),
             Triple("app", "prefers", "concise_mode", 1)],
            embedder, pool,
        )

        assert embedder.in_txn_flags, "expected at least one embed() call"
        assert all(flag is False for flag in embedder.in_txn_flags), (
            f"embed ran inside a write transaction: {embedder.in_txn_flags}"
        )
        # And the collapse still happened.
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='prefers'"
        ).fetchone()["c"] == 1
    finally:
        hy.close()
