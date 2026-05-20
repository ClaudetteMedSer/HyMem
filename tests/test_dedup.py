"""Tests for triple semantic dedup at extraction time (improv item E).

Before minting a new edge, `_upsert_triple` checks existing same-predicate
edges by cosine similarity over cached `edge_embeddings`. A near-duplicate
(`app uses uv` vs `app uses uv_pip`) gets the new evidence attached instead of
spawning a sibling canonical. The predicate is a hard gate so `uses` / `avoids`
never collapse together.
"""

from __future__ import annotations

import json

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.dreaming.chunks import Chunk
from hymem.extraction.triples import Triple


class FakeEmbedder:
    """Maps exact triple-text strings to controlled vectors so cosine
    similarity is deterministic. Unmapped texts raise (the tests map every
    text they expect to be embedded)."""

    model = "fake"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embed(self, texts):
        return [self.mapping[t] for t in texts]


def _seed_existing_edge(hy: HyMem, subj, pred, obj, vector):
    """Insert an active edge plus its cached edge_embeddings vector."""
    conn = hy.conn
    conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, neg_evidence) VALUES (?, ?, ?, 0, 0)",
        (subj, pred, obj),
    )
    conn.execute(
        "INSERT INTO edge_embeddings(edge_text, vector_json, model, dim) "
        "VALUES (?, ?, 'fake', 4)",
        (f"{subj} {pred} {obj}", json.dumps(vector)),
    )


def _seed_chunk(hy: HyMem, chunk_id="c_dedup") -> Chunk:
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s_dedup')")
    # Assistant role → speaker weight 1, so this test isolates dedup behavior
    # from the speaker-weighting feature.
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s_dedup', 'assistant', 'msg')"
    )
    mid = cur.lastrowid
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text) VALUES (?, 's_dedup', ?, ?, 'long_user_turn', 'text')",
        (chunk_id, mid, mid),
    )
    return Chunk(
        id=chunk_id, session_id="s_dedup", start_message_id=mid,
        end_message_id=mid, salience_reason="long_user_turn", text="text",
    )


def _persist(hy: HyMem, chunk: Chunk, triple: Triple, embedder):
    ext = ChunkExtraction(triples=[triple], markers=[])
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn, chunk, ext,
            prompt_version=hy.config.prompt_version,
            cfg=hy.config, embedding_client=embedder,
        )


def test_near_duplicate_attaches_to_existing_edge(cfg):
    hy = HyMem(cfg)
    try:
        _seed_existing_edge(hy, "app", "uses", "uv", [1.0, 0.0, 0.0, 0.0])
        chunk = _seed_chunk(hy)
        # Candidate "app uses uv_pip" embeds identically to "app uses uv".
        embedder = FakeEmbedder({"app uses uv_pip": [1.0, 0.0, 0.0, 0.0]})

        _persist(hy, chunk, Triple("app", "uses", "uv_pip", 1), embedder)

        # No sibling canonical was created.
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE object_canonical = 'uv_pip'"
        ).fetchone()["c"] == 0
        # Evidence landed on the existing edge, carrying the candidate surface form.
        row = hy.conn.execute(
            "SELECT kg.pos_evidence, e.surface_object "
            "FROM knowledge_graph kg JOIN kg_evidence e ON e.edge_id = kg.id "
            "WHERE kg.object_canonical = 'uv'"
        ).fetchone()
        assert row["pos_evidence"] == 1
        assert row["surface_object"] == "uv_pip"
    finally:
        hy.close()


def test_dissimilar_triple_creates_new_edge(cfg):
    hy = HyMem(cfg)
    try:
        _seed_existing_edge(hy, "app", "uses", "uv", [1.0, 0.0, 0.0, 0.0])
        chunk = _seed_chunk(hy)
        # Orthogonal vector → cosine 0 → below threshold → new edge.
        embedder = FakeEmbedder({"app uses mysql": [0.0, 1.0, 0.0, 0.0]})

        _persist(hy, chunk, Triple("app", "uses", "mysql", 1), embedder)

        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE object_canonical = 'mysql'"
        ).fetchone()["c"] == 1
    finally:
        hy.close()


def test_predicate_gate_blocks_cross_predicate_dedup(cfg):
    """An identical embedding under a different predicate must NOT dedup —
    `app avoids uv` means the opposite of `app uses uv`."""
    hy = HyMem(cfg)
    try:
        _seed_existing_edge(hy, "app", "uses", "uv", [1.0, 0.0, 0.0, 0.0])
        chunk = _seed_chunk(hy)
        embedder = FakeEmbedder({"app avoids uv": [1.0, 0.0, 0.0, 0.0]})

        _persist(hy, chunk, Triple("app", "avoids", "uv", 1), embedder)

        # A distinct (app, avoids, uv) edge exists alongside (app, uses, uv).
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph "
            "WHERE predicate = 'avoids' AND object_canonical = 'uv'"
        ).fetchone()["c"] == 1
    finally:
        hy.close()


def test_lexical_guard_blocks_false_merge(cfg):
    """Even at cosine 1.0, two short distinct names (`redis` / `redash`) must
    NOT merge — the lexical-sibling guard rejects them."""
    hy = HyMem(cfg)
    try:
        _seed_existing_edge(hy, "app", "uses", "redis", [1.0, 0.0, 0.0, 0.0])
        chunk = _seed_chunk(hy)
        embedder = FakeEmbedder({"app uses redash": [1.0, 0.0, 0.0, 0.0]})

        _persist(hy, chunk, Triple("app", "uses", "redash", 1), embedder)

        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE object_canonical = 'redash'"
        ).fetchone()["c"] == 1
    finally:
        hy.close()


def test_different_subject_not_merged(cfg):
    """Same predicate+object but a genuinely different subject is a different
    fact, not a sibling canonical — no merge even at cosine 1.0."""
    hy = HyMem(cfg)
    try:
        _seed_existing_edge(hy, "med_flow", "uses", "fastapi", [1.0, 0.0, 0.0, 0.0])
        chunk = _seed_chunk(hy)
        embedder = FakeEmbedder({"fractal uses fastapi": [1.0, 0.0, 0.0, 0.0]})

        _persist(hy, chunk, Triple("fractal", "uses", "fastapi", 1), embedder)

        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE subject_canonical = 'fractal'"
        ).fetchone()["c"] == 1
    finally:
        hy.close()


def test_dedup_disabled_by_config(cfg):
    import dataclasses

    hy = HyMem(dataclasses.replace(cfg, triple_dedup_enabled=False))
    try:
        _seed_existing_edge(hy, "app", "uses", "uv", [1.0, 0.0, 0.0, 0.0])
        chunk = _seed_chunk(hy)
        embedder = FakeEmbedder({"app uses uv_pip": [1.0, 0.0, 0.0, 0.0]})

        _persist(hy, chunk, Triple("app", "uses", "uv_pip", 1), embedder)

        # Dedup off → the sibling canonical is created.
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE object_canonical = 'uv_pip'"
        ).fetchone()["c"] == 1
    finally:
        hy.close()
