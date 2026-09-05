"""Tests proving the phase-1 dedup embed runs OUTSIDE the SQLite write lock.

The dedup candidate's triple-text embedding used to be issued by
`_find_near_duplicate_edge` while the per-chunk `BEGIN IMMEDIATE` write
transaction was held, so a (now retried/backed-off) network round-trip
inflated lock-hold time. The embed is now precomputed by
`phase1.prepare_dedup_vectors` before the transaction opens; these tests
guard that contract and confirm dedup behaviour is unchanged end-to-end.
"""

from __future__ import annotations

import dataclasses
import json

from hymem import HyMem
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.embeddings import StubEmbeddingClient
from hymem.extraction.triples import Triple

from tests.conftest import make_routed_llm


class TxnRecordingEmbedder:
    """Wraps embed() to record `conn.in_transaction` at call time.

    `sqlite3.Connection.in_transaction` is True exactly when a transaction is
    open (i.e. inside `BEGIN IMMEDIATE`). Asserting it was False on every embed
    proves no embed ran under the write lock.
    """

    model = "fake"
    dim = 4

    def __init__(self, conn, mapping: dict[str, list[float]]):
        self._conn = conn
        self._mapping = mapping
        self.in_txn_flags: list[bool] = []
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.in_txn_flags.append(self._conn.in_transaction)
        self.calls.append(list(texts))
        return [self._mapping[t] for t in texts]


def _seed_existing_edge(hy, subj, pred, obj, vector):
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


def _seed_chunk(hy, chunk_id="c_dedup") -> Chunk:
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('s_dedup')")
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


def test_prepare_runs_embed_outside_transaction(cfg):
    """prepare_dedup_vectors must embed with NO write transaction open, and the
    precomputed vector must still drive the merge once inside the lock."""
    hy = HyMem(cfg)
    try:
        _seed_existing_edge(hy, "app", "uses", "uv", [1.0, 0.0, 0.0, 0.0])
        chunk = _seed_chunk(hy)
        embedder = TxnRecordingEmbedder(hy.conn, {"app uses uv_pip": [1.0, 0.0, 0.0, 0.0]})
        ext = ChunkExtraction(triples=[Triple("app", "uses", "uv_pip", 1)], markers=[])

        # Prepare OUTSIDE any transaction (as the runner does).
        assert hy.conn.in_transaction is False
        dedup_vectors = phase1.prepare_dedup_vectors(hy.conn, ext, hy.config, embedder)
        assert dedup_vectors == {"app uses uv_pip": [1.0, 0.0, 0.0, 0.0]}

        from hymem.core import db as core_db
        with core_db.transaction(hy.conn):
            phase1.persist_chunk_results(
                hy.conn, chunk, ext, prompt_version=hy.config.prompt_version,
                cfg=hy.config, embedding_client=embedder, dedup_vectors=dedup_vectors,
            )

        # embed() was called exactly once, and never under the write lock.
        assert embedder.calls == [["app uses uv_pip"]]
        assert embedder.in_txn_flags == [False]
        # Merge still fired: no sibling canonical created.
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE object_canonical = 'uv_pip'"
        ).fetchone()["c"] == 0
    finally:
        hy.close()


def test_dream_embed_never_inside_write_lock(cfg):
    """End-to-end through hy.dream(): a wrapped StubEmbeddingClient records
    conn.in_transaction on every embed; it must be False for all of them, and
    at least one dedup embed must have occurred."""
    seen_in_txn: list[bool] = []

    class WatchingStub(StubEmbeddingClient):
        # Set after HyMem builds its conn so we can observe the live connection.
        conn = None

        def embed(self, texts):
            # Only watch the dedup-candidate embed. The runner's chunk-embedding
            # pass runs on a *background* thread, where reading the main
            # connection's in_transaction would be a cross-thread race; the
            # dedup prepare embed is synchronous on the main thread, so its
            # in_transaction reading is meaningful and is the call we de-locked.
            if self.conn is not None and texts == ["app uses uv_pip"]:
                seen_in_txn.append(self.conn.in_transaction)
            return super().embed(texts)

    embed = WatchingStub()
    # The exact source id is only known after the messages are written. Start
    # with an empty valid response, then install a source-citing response below.
    llm = make_routed_llm([], [])
    hy = HyMem(cfg, llm=llm, embedding_client=embed)
    embed.conn = hy.conn
    try:
        # Seed the existing edge + its cached embedding the candidate matches.
        cand_text = "app uses uv_pip"
        vec = embed.embed([cand_text])[0]  # outside txn; primes nothing but the vector
        seen_in_txn.clear()
        hy.conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
            "pos_evidence, neg_evidence, status) VALUES ('app', 'uses', 'uv', 0, 0, 'active')"
        )
        hy.conn.execute(
            "INSERT INTO edge_embeddings(edge_text, vector_json, model, dim) "
            "VALUES (?, ?, ?, ?)",
            ("app uses uv", json.dumps(vec), embed.model, embed.dim),
        )
        hy.conn.commit()

        hy.open_session("s1")
        user_message_ids = []
        for _ in range(2):
            hy.log_message("s1", "assistant", "anything here for context padding")
            user_message_ids.append(hy.log_message(
                "s1", "user",
                "I really prefer the uv_pip tool for the local dev environment overall.",
            ))
        hy.close_session("s1")
        hy.set_llm(make_routed_llm(
            [{
                "subject": "app", "predicate": "uses", "object": "uv_pip",
                "polarity": 1, "source_message_id": user_message_ids[-1],
            }],
            [],
        ))

        hy.dream()

        assert seen_in_txn, "expected at least one embed() during the dream"
        assert all(flag is False for flag in seen_in_txn), (
            f"embed ran inside a write transaction: {seen_in_txn}"
        )
        # Behaviour preserved end-to-end: the near-duplicate attached to the
        # existing edge rather than minting a sibling canonical.
        assert hy.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph WHERE object_canonical = 'uv_pip'"
        ).fetchone()["c"] == 0
        assert hy.conn.execute(
            "SELECT pos_evidence FROM knowledge_graph "
            "WHERE subject_canonical='app' AND predicate='uses' AND object_canonical='uv'"
        ).fetchone()["pos_evidence"] >= 1
    finally:
        hy.close()


def test_prepare_returns_empty_when_disabled_or_no_client(cfg):
    """No embed and an empty dict when dedup is off or no embedding client."""
    hy = HyMem(cfg)
    try:
        _seed_existing_edge(hy, "app", "uses", "uv", [1.0, 0.0, 0.0, 0.0])
        ext = ChunkExtraction(triples=[Triple("app", "uses", "uv_pip", 1)], markers=[])

        # No embedding client -> {}
        assert phase1.prepare_dedup_vectors(hy.conn, ext, hy.config, None) == {}

        # Dedup disabled -> {}, and the embedder is never called.
        embedder = TxnRecordingEmbedder(hy.conn, {"app uses uv_pip": [1.0, 0.0, 0.0, 0.0]})
        disabled = dataclasses.replace(hy.config, triple_dedup_enabled=False)
        assert phase1.prepare_dedup_vectors(hy.conn, ext, disabled, embedder) == {}
        assert embedder.calls == []
    finally:
        hy.close()
