"""Independent, network-free controls for shared embedding request admission.

Callsite tests retain the maintained producer implementation and its identity
checks. The helper-only adversaries deliberately do not represent producers.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db
from hymem.core.embedding_batches import (
    EmbeddingInputTooLarge,
    EmbeddingResponseInvalid,
    embed_bounded,
    plan_embedding_batches,
)
from hymem.dreaming import aggregate, embeddings, phase1, runner
from hymem.dreaming.aggregation_material import embedding_storage_identity
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.embeddings import MappedStubEmbeddingClient
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.triples import Triple
from tests.test_aggregation_provenance import _seed_native_episode


@pytest.fixture
def bounds_conn(tmp_path):
    conn = db.connect(tmp_path / "bounds.sqlite")
    db.initialize(conn)
    try:
        yield conn
    finally:
        conn.close()


def _client(conn=None, vectors=None):
    return MappedStubEmbeddingClient(
        vectors, model="root-bounds-offline-v1", dim=2,
        default=[1.0, 0.5], conn=conn,
    )


def _flatten(calls):
    return [text for call in calls for text in call]


def _assert_bounded(client):
    assert client.calls
    for call in client.calls:
        assert 1 <= len(call) <= 16
        assert sum(map(len, call)) <= 128_000
        assert sum(len(text.encode("utf-8")) for text in call) <= 512_000
    if client.transaction_states:
        assert client.transaction_states == [False] * len(client.calls)


def _messages(conn, texts):
    with db.transaction(conn):
        conn.execute("INSERT INTO sessions(id) VALUES ('bounds-messages')")
        ids = [int(conn.execute(
            "INSERT INTO messages(session_id,role,content) "
            "VALUES ('bounds-messages','user',?)", (text,),
        ).lastrowid) for text in texts]
        materialize_message_coverage(conn, "bounds-messages")
    return tuple(ids)


@pytest.mark.parametrize("texts", [
    [f"item-{i}" for i in range(35)],
    ["a" * 64_000, "b" * 64_000, "a" * 64_000],
    ["\U0001f600" * 64_000, "\U0001f680" * 64_000, "tail", "tail"],
])
def test_planner_and_dispatch_preserve_exact_order_and_duplicates(texts):
    vectors = {text: [float(i + 1), 0.5] for i, text in enumerate(texts)}
    client = _client(vectors=vectors)
    identity = embedding_storage_identity(client)
    embed_function = type(client).embed
    result = embed_bounded(
        client, texts, identity=lambda: embedding_storage_identity(client),
        expected_model=identity[0],
    )
    _assert_bounded(client)
    assert _flatten(client.calls) == texts
    assert _flatten(plan_embedding_batches(texts)) == texts
    assert result == [vectors[text] for text in texts]
    assert embedding_storage_identity(client) == identity
    assert type(client).embed is embed_function


def test_utf8_limit_is_independent_of_character_limit(monkeypatch):
    # UTF-8 needs at most four bytes/codepoint, so the deployed 4:1 byte cap
    # coincides with the character cap. Tighten ONLY this cap to exercise it.
    from hymem.core import embedding_batches

    monkeypatch.setattr(embedding_batches, "MAX_EMBEDDING_UTF8_BYTES", 12)
    assert plan_embedding_batches(["\U0001f600" * 2, "\U0001f680" * 2]) == (
        ("\U0001f600" * 2,), ("\U0001f680" * 2,),
    )
    with pytest.raises(EmbeddingInputTooLarge):
        plan_embedding_batches(["\U0001f600" * 4])


@pytest.mark.parametrize("unsafe", ["x" * 128_001, "\U0001f600" * 128_001])
def test_late_oversize_input_is_rejected_before_any_provider_call(unsafe):
    client = _client()
    with pytest.raises(EmbeddingInputTooLarge) as caught:
        embed_bounded(
            client, [f"safe-{i}" for i in range(33)] + [unsafe],
            identity=lambda: embedding_storage_identity(client),
            expected_model=embedding_storage_identity(client)[0],
        )
    assert client.calls == []
    assert unsafe not in str(caught.value)


@pytest.mark.parametrize("bad_input", ["private-surrogate-\ud800", 7, b"bytes"])
def test_invalid_input_is_preflighted_without_dispatch(bad_input):
    client = _client()
    with pytest.raises(ValueError):
        embed_bounded(
            client, ["safe"] * 32 + [bad_input],
            identity=lambda: embedding_storage_identity(client),
            expected_model=embedding_storage_identity(client)[0],
        )
    assert client.calls == []


@pytest.mark.parametrize("failure", [
    "cardinality", "dimension", "nonfinite", "zero", "wrong_model",
    "identity_dimension",
])
def test_first_invalid_response_prevents_later_provider_requests(failure):
    class AdversarialClient:
        calls = 0

        def embed(self, texts):
            self.calls += 1
            if failure == "cardinality":
                return [[1.0, 0.5]] * (len(texts) - 1)
            vector = {
                "dimension": [1.0], "nonfinite": [float("nan"), 0.5],
                "zero": [0.0, 0.0],
            }.get(failure, [1.0, 0.5])
            return [vector] * len(texts)

    client = AdversarialClient()
    identity = lambda: (
        "wrong" if failure == "wrong_model" else "fixture",
        3 if failure == "identity_dimension" else 2,
    )
    with pytest.raises(RuntimeError):
        embed_bounded(
            client, [f"item-{i}" for i in range(33)],
            identity=identity, expected_model="fixture",
        )
    assert client.calls == 1


def test_dimension_changes_after_first_valid_batch_stop_before_third():
    class ChangingClient:
        calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[1.0] * (2 if self.calls == 1 else 3) for _ in texts]

    client = ChangingClient()
    with pytest.raises(RuntimeError, match="dimension"):
        embed_bounded(
            client, [str(i) for i in range(33)],
            identity=lambda: ("fixture", 2 if client.calls == 1 else 3),
            expected_model="fixture",
        )
    assert client.calls == 2


def test_edge_fetch_bounds_request_and_preserves_text_vector_mapping(bounds_conn):
    texts = [f"service_{i} uses store_{i}" for i in range(65)]
    with db.transaction(bounds_conn):
        bounds_conn.executemany(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,neg_evidence,status) "
            "VALUES (?,'uses',?,1,0,'active')",
            [(f"service_{i}", f"store_{i}") for i in range(65)],
        )
    vectors = {text: [float(i + 1), 0.5] for i, text in enumerate(texts)}
    client = _client(bounds_conn, vectors)
    identity = embedding_storage_identity(client)
    pending = embeddings.fetch_edge_embeddings(bounds_conn, client)
    _assert_bounded(client)
    assert _flatten(client.calls) == sorted(texts)
    assert pending.new_text_vectors == vectors
    assert (pending.model, pending.dim) == identity
    assert bounds_conn.execute("SELECT count(*) FROM edge_embeddings").fetchone()[0] == 0


def test_message_fetch_preserves_duplicate_occurrences(bounds_conn):
    texts = [f"message-{i}" for i in range(40)] + ["message-3"] * 2
    ids = _messages(bounds_conn, texts)
    vectors = {text: [float(i + 1), 0.5] for i, text in enumerate(texts)}
    client = _client(bounds_conn, vectors)
    pending = embeddings.fetch_message_embeddings(bounds_conn, client, message_ids=ids)
    _assert_bounded(client)
    assert _flatten(client.calls) == list(dict.fromkeys(texts))
    assert pending.message_ids == list(ids)
    assert pending.vectors == [vectors[text] for text in texts]
    assert bounds_conn.execute("SELECT count(*) FROM message_embeddings").fetchone()[0] == 0


def test_message_oversize_has_no_network_bisection_or_partial_publication(bounds_conn):
    texts = [f"message-{i}" for i in range(33)] + ["unsafe-source-" + "x" * 128_001]
    ids = _messages(bounds_conn, texts)
    client = _client(bounds_conn)
    persisted, cache_hits, _abort = runner._persist_message_batch_with_failure_isolation(
        bounds_conn, client, ids,
    )
    assert client.calls == []
    assert (persisted, cache_hits) == (0, 0)
    assert bounds_conn.execute("SELECT count(*) FROM message_embeddings").fetchone()[0] == 0
    assert [row[0] for row in bounds_conn.execute(
        "SELECT content FROM messages ORDER BY id"
    )] == texts


def test_message_response_admission_failure_is_not_bisected(bounds_conn, monkeypatch):
    ids = _messages(bounds_conn, [f"message-{i}" for i in range(33)])
    attempted = []

    def invalid_response(_conn, _client, *, message_ids):
        attempted.append(message_ids)
        raise EmbeddingResponseInvalid("synthetic local admission failure")

    monkeypatch.setattr(runner, "fetch_message_embeddings", invalid_response)
    persisted, cache_hits, _abort = runner._persist_message_batch_with_failure_isolation(
        bounds_conn, _client(bounds_conn), ids,
    )
    assert attempted == [ids]
    assert (persisted, cache_hits) == (0, 0)


@pytest.mark.parametrize("caller", ["episodes", "aggregate"])
def test_oversize_source_rejected_by_callsite_before_any_request(bounds_conn, caller):
    client = _client(bounds_conn)
    unsafe = "x" * 128_001
    if caller == "episodes":
        for i in range(18):
            _seed_native_episode(
                bounds_conn, f"oversize_episode_{i}", title=f"Episode {i}",
                summary=unsafe if i == 17 else f"Safe summary {i}", entity="fixture",
            )
        invoke = lambda: embeddings.fetch_episode_embeddings(bounds_conn, client)
    else:
        rows = [dict(
            id=f"node-{i:03}", title=f"Title {i}",
            summary=unsafe if i == 33 else "Safe summary", node_kind="cluster", level=0,
        ) for i in range(34)]
        invoke = lambda: aggregate._prepare_candidate_node_embeddings(bounds_conn, rows, client)
    with pytest.raises(EmbeddingInputTooLarge):
        invoke()
    assert client.calls == []


def test_dedup_fetch_bounds_request_and_deduplicates_exact_candidates(bounds_conn, tmp_path):
    triples = [Triple("new_service", "uses", f"new_object_{i}", 1) for i in range(65)]
    extraction = phase1.ChunkExtraction(triples=triples + [triples[3]], markers=[])
    texts = [f"new_service uses new_object_{i}" for i in range(65)]
    client = _client(bounds_conn)
    result = phase1.prepare_dedup_vectors(
        bounds_conn, extraction, HyMemConfig(root=tmp_path), client,
    )
    _assert_bounded(client)
    assert _flatten(client.calls) == texts
    assert list(result) == texts
    assert list(result.values()) == [[1.0, 0.5]] * len(texts)


def test_episode_fetch_bounds_request_and_preserves_id_mapping(bounds_conn):
    texts_by_id = {}
    for i in range(33):
        title, summary = f"Episode {i}", f"Summary {i} " + "x" * 4_000
        episode_id, _, _ = _seed_native_episode(
            bounds_conn, f"episode_{i}", title=title, summary=summary, entity="fixture",
        )
        texts_by_id[episode_id] = f"{title}\n{summary}"
    vectors = {text: [float(i + 1), 0.5] for i, text in enumerate(texts_by_id.values())}
    client = _client(bounds_conn, vectors)
    pending = embeddings.fetch_episode_embeddings(bounds_conn, client)
    _assert_bounded(client)
    assert _flatten(client.calls) == [texts_by_id[item] for item in pending.ids]
    assert pending.vectors == [vectors[texts_by_id[item]] for item in pending.ids]


@pytest.mark.parametrize("current_publication", [False, True])
def test_aggregate_provider_paths_are_bounded(bounds_conn, monkeypatch, current_publication):
    rows = [dict(
        id=f"node-{i:03}", title=f"Title {i}", summary="x" * 2_000,
        node_kind="cluster", level=0,
    ) for i in range(65)]
    texts = [f"{row['title']}\n{row['summary']}" for row in rows]
    vectors = {text: [float(i + 1), 0.5] for i, text in enumerate(texts)}
    client = _client(bounds_conn, vectors)
    if current_publication:
        # The source-proof validator has its own adversarial suite. This
        # control injects its typed result, not a producer/client substitute.
        monkeypatch.setattr(aggregate, "load_current_aggregation_publication", lambda *a, **k:
            SimpleNamespace(nodes={row["id"]: SimpleNamespace(row=row) for row in rows}))
        pending = aggregate.fetch_node_embeddings(bounds_conn, client)
    else:
        pending = aggregate._prepare_candidate_node_embeddings(bounds_conn, rows, client)
    _assert_bounded(client)
    assert _flatten(client.calls) == texts
    assert pending.node_ids == [row["id"] for row in rows]
    assert pending.vectors == [vectors[text] for text in texts]


def test_real_background_dispatch_bounds_before_phase1(cfg, monkeypatch):
    class StopAfterBackground(BaseException):
        pass

    executions = []

    class InlineExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, function):
            before = len(client.calls)
            vectors = function()
            executions.append((vectors, client.calls[before:]))
            raise StopAfterBackground

        def shutdown(self, *args, **kwargs):
            pass

    monkeypatch.setattr(runner, "ThreadPoolExecutor", InlineExecutor)
    client = _client()
    hy = HyMem(
        replace(cfg, dream_budget=1, dream_baseline_budget=0),
        llm=StubLLMClient(default="[]"), embedding_client=client,
    )
    try:
        hy.open_session("background")
        for i in range(33):
            hy.log_message("background", "assistant", "anything")
            hy.log_message("background", "user",
                f"I prefer choice_{i} for the local development environment because it is fast.")
        hy.close_session("background")
        with pytest.raises(StopAfterBackground):
            hy.dream(session_ids=["background"])
        _assert_bounded(client)
        assert len(executions) == 1
        assert len(executions[0][0]) == 33
        assert len(_flatten(executions[0][1])) == 33
        assert hy.conn.execute("SELECT count(*) FROM chunk_embeddings").fetchone()[0] == 0
    finally:
        hy.close()
