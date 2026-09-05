from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hymem import HyMem
from hymem.bootstrap import build_from_env, resolve_env
from hymem.config import HyMemConfig
from hymem.contrib.openai_embedding_client import (
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    OpenAICompatibleEmbeddingClient,
    openai_compatible_embedding_identity,
    safe_embedding_base_url,
)
from hymem.core import db as core_db
from hymem.core.vectors import encode_vector
from hymem.doctor import FAIL, WARN, _check_embedding, _check_schema_and_dim
from hymem.dreaming.aggregate import (
    fetch_node_embeddings,
    persist_node_embeddings,
)
from hymem.dreaming.embeddings import (
    MESSAGE_EMBEDDING_BATCH_SIZE,
    _fetch_cached_vectors,
    fetch_chunk_embeddings,
    fetch_edge_embeddings,
    fetch_episode_embeddings,
    fetch_fact_embeddings,
    fetch_message_embeddings,
    message_embedding_id_batches,
    persist_message_embeddings,
)
from hymem.dreaming import facts as narrative_facts
from hymem.dreaming.retention import prune_messages
from hymem.extraction.embeddings import (
    CachedEmbeddingClient,
    LocalHashEmbeddingClient,
    embedding_text_hash,
    normalize_text,
)
from hymem.extraction.llm import StubLLMClient
from hymem.query.augment import (
    _aggregation_search,
    _episode_search,
    _fact_search,
    _vector_search,
)


class RecordingEmbedder:
    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        *,
        model: str = "semantic-test-v1",
        dim: int = 3,
        default: list[float] | None = None,
        quality: str = "semantic",
        fail_on: str | None = None,
        conn=None,
    ) -> None:
        self._model = model
        self._dim = dim
        self.vectors = vectors or {}
        self.default = default or [0.0, 1.0, 0.0][:dim]
        self.quality = quality
        self.backend = "recording"
        self.network_free = True
        self.fail_on = fail_on
        self.conn = conn
        self.calls: list[list[str]] = []
        self.transaction_states: list[bool] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts):
        payload = list(texts)
        self.calls.append(payload)
        if self.conn is not None:
            self.transaction_states.append(bool(self.conn.in_transaction))
        if self.fail_on is not None and any(self.fail_on in text for text in payload):
            raise RuntimeError("provider unavailable")
        return [list(self.vectors.get(text, self.default)) for text in payload]


def _quiet_cfg(cfg: HyMemConfig, **overrides) -> HyMemConfig:
    return replace(
        cfg,
        working_memory_turns=0,
        aggregation_nodes_enabled=False,
        **overrides,
    )


def test_default_local_backend_is_deterministic_observable_and_collision_safe(cfg):
    embedder = LocalHashEmbeddingClient()
    first = embedder.embed(["Orchestrator configuration"])[0]
    second = embedder.embed(["Orchestrator configuration"])[0]
    assert first == second
    assert len(first) == embedder.dim
    assert math.isclose(sum(value * value for value in first), 1.0)
    assert (embedder.backend, embedder.quality, embedder.network_free) == (
        "local_feature_hash", "lexical", True,
    )

    hy = HyMem(_quiet_cfg(cfg), embedding_client=embedder)
    try:
        hy.log_messages(
            "local",
            [
                ("user", "favorite jazz musician"),
                ("user", "annual dental checkup"),
            ],
        )
        ctx = hy.augment("quantum chromodynamics unrelated terms")
        assert ctx.message_hits == []
        assert ctx.semantic_status.available is True
        assert ctx.semantic_status.quality == "lexical"
        assert hy.embedding_status["network_free"] is True
    finally:
        hy.close()


def test_deepseek_only_environment_uses_local_fallback_without_remote_embedder(
    monkeypatch, tmp_path,
):
    for name in (
        "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
        "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM", "OPENAI_API_KEY",
        "HYMEM_LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "llm-only-key")
    monkeypatch.setenv("HYMEM_ROOT", str(tmp_path))

    class DummyLLM:
        def __init__(self, **_kwargs):
            pass

    class BrokenRemoteEmbedder:
        def __init__(self, **_kwargs):
            raise AssertionError("DeepSeek-only config must not construct this")

    monkeypatch.setattr(
        "hymem.contrib.openai_client.OpenAICompatibleClient", DummyLLM
    )
    monkeypatch.setattr(
        "hymem.contrib.openai_embedding_client.OpenAICompatibleEmbeddingClient",
        BrokenRemoteEmbedder,
    )
    resolved = resolve_env()
    assert resolved.embedding_backend == "local_feature_hash"
    assert resolved.has_embedding_client is True
    hy = build_from_env()
    try:
        assert hy.embedding_status["backend"] == "local_feature_hash"
        assert hy.embedding_status["quality"] == "lexical"
    finally:
        hy.close()


def test_explicit_official_openai_embeddings_may_inherit_openai_key(
    monkeypatch, tmp_path,
):
    for name in (
        "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
        "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HYMEM_ROOT", str(tmp_path))
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("HYMEM_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "official-openai-key")

    resolved = resolve_env()
    assert resolved.embedding_backend == "openai_compatible"
    assert resolved.embedding_api_key == "official-openai-key"


def test_custom_embedding_endpoint_never_inherits_openai_key(monkeypatch, cfg):
    for name in ("HYMEM_EMBEDDING_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://attacker.example/v1")
    monkeypatch.setenv("HYMEM_EMBEDDING_MODEL", "same-label")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    constructed: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    resolved = resolve_env()
    assert resolved.embedding_backend == "local_feature_hash"
    assert resolved.embedding_fallback_reason == "remote_embedding_credentials_missing"
    doctor_result, live_dim = _check_embedding(resolved)
    assert doctor_result.status == WARN
    assert "remote_embedding_credentials_missing" in doctor_result.detail
    assert live_dim == resolved.embedding_dim
    with pytest.raises(EnvironmentError, match="HYMEM_EMBEDDING_API_KEY"):
        OpenAICompatibleEmbeddingClient()
    assert constructed == []
    hy = HyMem(
        _quiet_cfg(cfg),
        embedding_client=CachedEmbeddingClient(LocalHashEmbeddingClient(
            fallback_reason=resolved.embedding_fallback_reason,
        )),
    )
    try:
        assert hy.embedding_status["fallback_reason"] == (
            "remote_embedding_credentials_missing"
        )
        assert hy.augment("observable").semantic_status.fallback_reason == (
            "remote_embedding_credentials_missing"
        )
    finally:
        hy.close()


def test_embedding_transport_requires_https_except_loopback(monkeypatch):
    for name in ("HYMEM_EMBEDDING_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    constructed: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.embeddings = SimpleNamespace()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("HYMEM_EMBEDDING_API_KEY", "embedding-specific")
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "http://remote.example/v1")
    rejected = resolve_env()
    assert rejected.embedding_backend == "local_feature_hash"
    assert rejected.embedding_fallback_reason == "remote_embedding_endpoint_rejected"
    doctor_result, live_dim = _check_embedding(rejected)
    assert doctor_result.status == FAIL
    assert "remote_embedding_endpoint_rejected" in doctor_result.detail
    assert live_dim == rejected.embedding_dim
    with pytest.raises(ValueError, match="only on loopback"):
        OpenAICompatibleEmbeddingClient()
    assert constructed == []

    monkeypatch.delenv("HYMEM_EMBEDDING_API_KEY")
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "http://127.0.0.1:8080/v1")
    local = resolve_env()
    assert local.embedding_backend == "openai_compatible"
    assert local.embedding_api_key == "local"
    OpenAICompatibleEmbeddingClient()
    assert constructed[-1]["api_key"] == "local"

    monkeypatch.setenv("HYMEM_EMBEDDING_API_KEY", "embedding-specific")
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://remote.example/v1")
    secure = resolve_env()
    assert secure.embedding_backend == "openai_compatible"
    OpenAICompatibleEmbeddingClient()
    assert constructed[-1] == {
        "api_key": "embedding-specific",
        "base_url": "https://remote.example/v1",
        "timeout": DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        "max_retries": 0,
    }


def test_provider_identity_namespaces_endpoint_without_hashing_query_secrets():
    a = openai_compatible_embedding_identity(
        "HTTPS://Embed.Example/v1/?deployment=a&api_key=top-secret", "same-model"
    )
    b = openai_compatible_embedding_identity(
        "https://embed.example/v1?deployment=b&api_key=other-secret", "same-model"
    )
    assert a != b
    assert "same-model" in a
    assert "top-secret" not in a
    assert "other-secret" not in b
    assert "deployment=a" in a and "deployment=b" in b
    # Credentials do not alter vector-space identity.
    assert a == openai_compatible_embedding_identity(
        "https://embed.example/v1?deployment=a&api_key=different", "same-model"
    )


def test_embedding_diagnostics_redact_url_credentials_and_provider_errors(
    monkeypatch, tmp_path, capsys,
):
    from hymem import doctor as doctor_mod

    secret = "top-secret-query-credential"
    raw_url = (
        "https://operator:password@embed.example/v1"
        f"?deployment=a&api_key={secret}#private-fragment"
    )
    safe_url = safe_embedding_base_url(raw_url)
    assert safe_url.startswith(
        "https://embed.example/v1?deployment=a"
    )
    assert all(value not in safe_url for value in (
        secret, "operator", "password", "private-fragment", "api_key",
    ))

    monkeypatch.setenv("HYMEM_ROOT", str(tmp_path))
    monkeypatch.setenv("HYMEM_EMBEDDING_API_KEY", "embedding-specific")
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", raw_url)
    monkeypatch.setenv("HYMEM_EMBEDDING_MODEL", "safe-model")
    monkeypatch.delenv("HYMEM_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class LeakyFailureClient:
        dim = 3

        def __init__(self, **_kwargs):
            pass

        def embed(self, _texts):
            raise RuntimeError(f"provider failed at {raw_url}")

    monkeypatch.setattr(
        "hymem.contrib.openai_embedding_client.OpenAICompatibleEmbeddingClient",
        LeakyFailureClient,
    )
    cfg = resolve_env()
    result, live_dim = _check_embedding(cfg)
    assert live_dim is None and result.status == FAIL
    assert safe_url in result.detail
    assert secret not in result.detail and "password" not in result.detail

    assert doctor_mod.run_doctor() == 1
    output = capsys.readouterr().out
    assert safe_url in output
    assert secret not in output and "password" not in output


def test_openai_provider_has_one_explicitly_timed_attempt(monkeypatch):
    constructed: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

            def create(**request):
                calls.append(request)
                raise TimeoutError("simulated timeout")

            self.embeddings = SimpleNamespace(create=create)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    client = OpenAICompatibleEmbeddingClient(
        api_key="key",
        base_url="https://embed.example/v1",
        model="m",
        dim=3,
        timeout=2.5,
    )
    with pytest.raises(TimeoutError, match="simulated timeout"):
        client.embed(["one request"])
    assert constructed == [{
        "api_key": "key",
        "base_url": "https://embed.example/v1",
        "timeout": 2.5,
        "max_retries": 0,
    }]
    assert calls == [{"model": "m", "input": ["one request"]}]


def test_provider_identity_prevents_cross_endpoint_cache_reuse(cfg):
    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    first = openai_compatible_embedding_identity("https://a.example/v1", "m")
    second = openai_compatible_embedding_identity("https://b.example/v1", "m")
    text_hash = embedding_text_hash("same input")
    try:
        conn.execute(
            "INSERT INTO embedding_cache(text_hash,model,vector_json,dim) "
            "VALUES (?,?,?,3)",
            (text_hash, first, encode_vector([1.0, 0.0, 0.0])),
        )
        assert _fetch_cached_vectors(
            conn, [text_hash], first, expected_dim=3
        ) == {text_hash: [1.0, 0.0, 0.0]}
        assert _fetch_cached_vectors(
            conn, [text_hash], second, expected_dim=3
        ) == {}
    finally:
        conn.close()


def test_openai_provider_reorders_complete_indices_and_rejects_partial(
    monkeypatch,
):
    responses = [
        SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]),
        SimpleNamespace(data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            SimpleNamespace(index=None, embedding=[0.0, 1.0]),
        ]),
    ]

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.embeddings = SimpleNamespace(create=lambda **_kw: responses.pop(0))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    client = OpenAICompatibleEmbeddingClient(
        api_key="key", base_url="https://one.example/v1", model="m", dim=99
    )
    assert client.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert client.dim == 2
    with pytest.raises(RuntimeError, match="partial/malformed indices"):
        client.embed(["first", "second"])


@pytest.mark.parametrize(
    "data,match",
    [
        ([SimpleNamespace(index=0, embedding=[1.0, 0.0])], "1 vectors for 2"),
        ([SimpleNamespace(index=0, embedding=[float("nan"), 1.0])], "non-finite"),
        ([SimpleNamespace(index=0, embedding=[0.0, 0.0])], "non-finite/zero"),
    ],
)
def test_openai_provider_rejects_bad_cardinality_and_vectors(
    monkeypatch, data, match,
):
    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.embeddings = SimpleNamespace(
                create=lambda **_kw: SimpleNamespace(data=data)
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    client = OpenAICompatibleEmbeddingClient(
        api_key="key", base_url="https://one.example/v1", model="m", dim=2
    )
    payload = ["a", "b"] if len(data) == 1 and match.startswith("1 vectors") else ["a"]
    with pytest.raises(RuntimeError, match=match):
        client.embed(payload)


def test_cached_client_rejects_short_batch_and_protects_cached_vectors():
    class ShortClient(RecordingEmbedder):
        def embed(self, texts):
            return []

    with pytest.raises(RuntimeError, match="0 vectors for 1 inputs"):
        CachedEmbeddingClient(ShortClient()).embed(["x"])

    inner = RecordingEmbedder({"x": [1.0, 0.0, 0.0]})
    cached = CachedEmbeddingClient(inner)
    returned = cached.embed(["x"])
    returned[0][0] = float("nan")
    assert cached.embed(["x"]) == [[1.0, 0.0, 0.0]]


def test_cached_client_rejects_identity_change_between_snapshot_and_call():
    class RacingIdentityClient:
        dim = 3

        def __init__(self):
            self.model_reads = 0
            self.calls = 0

        @property
        def model(self):
            self.model_reads += 1
            return "old-space" if self.model_reads == 1 else "new-space"

        def embed(self, texts):
            self.calls += 1
            return [[1.0, 0.0, 0.0] for _ in texts]

    inner = RacingIdentityClient()
    cached = CachedEmbeddingClient(inner)
    with pytest.raises(RuntimeError, match="changed before provider call"):
        cached.embed(["x"])
    assert inner.calls == 0


def test_exact_input_hash_prevents_normalization_cache_alias(cfg):
    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    embedder = RecordingEmbedder({
        "Hello  World": [1.0, 0.0, 0.0],
        " hello world ": [0.0, 1.0, 0.0],
    })
    try:
        conn.execute("INSERT INTO sessions(id) VALUES ('s')")
        for index, text in enumerate(("Hello  World", " hello world "), start=1):
            conn.execute(
                "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
                "salience_reason,text) VALUES (?,?,?,?,'test',?)",
                (f"c{index}", "s", index, index, text),
            )
        legacy_hash = hashlib.sha256(
            normalize_text("Hello  World").encode("utf-8")
        ).hexdigest()
        conn.execute(
            "INSERT INTO embedding_cache(text_hash,model,vector_json,dim) "
            "VALUES (?,?,?,?)",
            (legacy_hash, embedder.model, encode_vector([0.0, 0.0, 1.0]), 3),
        )
        pending = fetch_chunk_embeddings(conn, embedder)
        assert pending is not None
        assert embedder.calls == [["Hello  World", " hello world "]]
        assert len(set(pending.text_hashes)) == 2
        assert all(value.startswith("sha256-exact-input-v1:") for value in pending.text_hashes)
    finally:
        conn.close()


def test_no_keyword_paraphrase_recovers_exact_message_and_keeps_lexical_winner(cfg):
    query = "Where did I leave my car?"
    semantic = "The automobile is sheltered behind the cedar garage."
    lexical = "car car car location is written in this exact lexical answer"
    noisy = "A completely unrelated record about ceramics."
    embedder = RecordingEmbedder({
        query: [1.0, 0.0, 0.0],
        semantic: [1.0, 0.0, 0.0],
        lexical: [0.0, 1.0, 0.0],
        noisy: [1.0, 0.0, 0.0],
    })
    hy = HyMem(
        _quiet_cfg(cfg, message_fts_top_k=2), embedding_client=embedder
    )
    try:
        ids = hy.log_messages(
            "s", [("user", semantic), ("user", lexical), ("user", noisy)]
        )
        embedder.calls.clear()
        ctx = hy.augment(query)
        assert any(hit.message_id == ids[0] and hit.score_kind == "semantic"
                   for hit in ctx.message_hits)
        assert embedder.calls == [[query]]

        control = hy.augment("car", source_session_id="s")
        assert control.message_hits[0].message_id == ids[1]
        assert "message_lexical_preserved" in control.message_hits[0].why_retrieved
    finally:
        hy.close()


def test_blank_query_abstains_without_embedding_or_arbitrary_message_hits(cfg):
    embedder = RecordingEmbedder(default=[1.0, 0.0, 0.0])
    hy = HyMem(_quiet_cfg(cfg), embedding_client=embedder)
    try:
        hy.log_messages(
            "s",
            [("user", "first unrelated memory"), ("assistant", "second memory")],
        )
        embedder.calls.clear()
        ctx = hy.augment(" \t\n", source_session_id="s")
        assert embedder.calls == []
        assert ctx.message_hits == []
        assert ctx.semantic_status.reason == "blank_query"
        assert ctx.semantic_status.attempted is False
    finally:
        hy.close()


def test_message_embedding_batches_are_bounded_deduplicated_and_complete(cfg):
    text = "one exact duplicate occurrence"
    count = MESSAGE_EMBEDDING_BATCH_SIZE * 2 + 5
    embedder = RecordingEmbedder({text: [1.0, 0.0, 0.0]})
    hy = HyMem(_quiet_cfg(cfg), embedding_client=embedder)
    try:
        ids = hy.log_messages("s", [("user", text)] * count)
        assert embedder.calls == [[text]]
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_embeddings WHERE message_id IN "
            f"({','.join('?' * len(ids))})",
            tuple(ids),
        ).fetchone()[0] == count
        assert all(
            len(batch) <= MESSAGE_EMBEDDING_BATCH_SIZE
            for batch in message_embedding_id_batches(
                hy.conn, message_ids=ids
            )
        )
    finally:
        hy.close()


def test_dream_isolates_one_poison_without_starving_same_batch_peers(cfg):
    count = MESSAGE_EMBEDDING_BATCH_SIZE + 1
    poison_index = 1
    texts = [f"backlog occurrence {index}" for index in range(count)]
    poison = "PERMANENTLY_REJECTED_INPUT"
    texts[poison_index] = poison
    hy = HyMem(
        _quiet_cfg(cfg),
        llm=StubLLMClient(default="[]"),
    )
    try:
        ids = hy.log_messages("s", [("user", text) for text in texts])
        failing = RecordingEmbedder(fail_on=poison, conn=hy.conn)
        hy.set_embedding_client(failing)
        hy.dream()
        embedded = {
            row["message_id"] for row in hy.conn.execute(
                "SELECT message_id FROM message_embeddings"
            ).fetchall()
        }
        assert embedded == set(ids) - {ids[poison_index]}
        assert len(failing.calls) <= 16
        assert all(state is False for state in failing.transaction_states)

        calls_before = len(failing.calls)
        hy.dream()
        embedded_again = {
            row["message_id"] for row in hy.conn.execute(
                "SELECT message_id FROM message_embeddings"
            ).fetchall()
        }
        assert embedded_again == embedded
        assert len(failing.calls) - calls_before <= 16
    finally:
        hy.close()


def test_dream_message_isolation_circuit_breaks_a_general_outage(cfg):
    hy = HyMem(_quiet_cfg(cfg), llm=StubLLMClient(default="[]"))
    try:
        hy.log_messages(
            "s",
            [("user", f"outage backlog {index}")
             for index in range(MESSAGE_EMBEDDING_BATCH_SIZE + 1)],
        )
        unavailable = RecordingEmbedder(fail_on="outage backlog", conn=hy.conn)
        hy.set_embedding_client(unavailable)
        hy.dream()
        # Parent + its two independent halves establish provider-wide failure;
        # never degrade into one full timeout per message.
        assert len(unavailable.calls) == 3
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_embeddings"
        ).fetchone()[0] == 0
        assert unavailable.transaction_states == [False, False, False]
    finally:
        hy.close()


def test_streamed_exact_message_search_keeps_a_late_scoped_winner(cfg):
    query = "opaque late semantic lookup"
    target = "the final durable occurrence contains the answer"
    distractors = [f"unrelated durable item {index}" for index in range(145)]
    embedder = RecordingEmbedder(
        {query: [1.0, 0.0, 0.0], target: [1.0, 0.0, 0.0]},
        default=[0.0, 1.0, 0.0],
    )
    hy = HyMem(
        _quiet_cfg(cfg, message_fts_top_k=1), embedding_client=embedder
    )
    try:
        ids = hy.log_messages(
            "late",
            [("user", text) for text in [*distractors, target]],
        )
        embedder.calls.clear()
        ctx = hy.augment(query, source_session_id="late")
        assert [hit.message_id for hit in ctx.message_hits] == [ids[-1]]
        assert embedder.calls == [[query]]
    finally:
        hy.close()


def test_retained_semantic_search_applies_scope_before_ranking_and_survives_prune(cfg):
    query = "Where is the vehicle kept?"
    target = "The automobile is sheltered behind the cedar garage."
    embedder = RecordingEmbedder({query: [1.0, 0.0, 0.0], target: [1.0, 0.0, 0.0]})
    scoped = _quiet_cfg(cfg, message_fts_top_k=1)
    hy = HyMem(scoped, embedding_client=embedder)
    try:
        old = "2020-01-02T03:04:05Z"
        target_id = hy.log_message(
            "target-session", "user", target, created_at=old,
            source_peer_id="alice", source_workspace_id="workspace-a",
        )
        distractors = [
            ("user", f"Unrelated retained record number {index}.", old)
            for index in range(40)
        ]
        hy.log_messages(
            "crowd-session", distractors,
            source_peer_ids=["alice"] * len(distractors),
            source_workspace_id="workspace-a",
        )
        other_id = hy.log_message(
            "other-workspace", "user", "A different automobile answer.",
            created_at=old, source_peer_id="mallory",
            source_workspace_id="workspace-b",
        )
        for session in ("target-session", "crowd-session", "other-workspace"):
            hy.close_session(session)
        with core_db.transaction(hy.conn):
            assert prune_messages(
                hy.conn, replace(scoped, message_retention_days=1)
            ) == 42

        ctx = hy.augment(
            query,
            source_session_id="target-session",
            source_peer_id="alice",
            source_workspace_id="workspace-a",
        )
        assert [hit.message_id for hit in ctx.message_hits] == [target_id]
        assert all(hit.source_workspace_id == "workspace-a" for hit in ctx.message_hits)
        assert other_id not in {hit.message_id for hit in ctx.message_hits}
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_embeddings WHERE message_id=?",
            (target_id,),
        ).fetchone()[0] == 1
        columns = {
            row["name"] for row in hy.conn.execute(
                "PRAGMA table_info(message_embeddings)"
            ).fetchall()
        }
        assert "source_session_id" not in columns
    finally:
        hy.close()


def test_message_semantic_rejects_wrong_identity_and_malformed_vectors(cfg):
    query = "opaque semantic lookup"
    embedder = RecordingEmbedder({query: [1.0, 0.0, 0.0]}, default=[1.0, 0.0, 0.0])
    hy = HyMem(_quiet_cfg(cfg, message_fts_top_k=10), embedding_client=embedder)
    try:
        texts = [f"stored occurrence {index}" for index in range(6)]
        ids = hy.log_messages("s", [("user", text) for text in texts])
        rows = [
            ("other-model", 3, encode_vector([1.0, 0.0, 0.0])),
            (embedder.model, 2, encode_vector([1.0, 0.0])),
            (embedder.model, 3, "[NaN,0,0]"),
            (embedder.model, 3, encode_vector([0.0, 0.0, 0.0])),
            (embedder.model, 3, "corrupt"),
        ]
        for message_id, (model, dim, vector) in zip(ids[1:], rows):
            hy.conn.execute(
                "UPDATE message_embeddings SET model=?,dim=?,vector_json=? "
                "WHERE message_id=?",
                (model, dim, vector, message_id),
            )
        if core_db.has_vec_table(hy.conn, table="vec_messages"):
            hy.conn.execute(
                "INSERT OR IGNORE INTO vec_messages(rowid,embedding) VALUES (?,?)",
                (999999, core_db._pack_vector([1.0, 0.0, 0.0])),
            )
        hits = hy.augment(query, source_session_id="s").message_hits
        assert [hit.message_id for hit in hits] == [ids[0]]
    finally:
        hy.close()


def test_ingest_batches_only_new_ids_redacts_before_provider_and_retries_in_dream(cfg):
    scoped = _quiet_cfg(cfg)
    failing = RecordingEmbedder(fail_on="retry backlog")
    hy = HyMem(scoped, llm=StubLLMClient(default="[]"), embedding_client=failing)
    try:
        backlog_id = hy.log_message("s", "user", "retry backlog")
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE id=?", (backlog_id,)
        ).fetchone()[0] == 1
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_embeddings WHERE message_id=?", (backlog_id,)
        ).fetchone()[0] == 0

        good = RecordingEmbedder(conn=hy.conn)
        hy.set_embedding_client(good)
        new_ids = hy.log_messages(
            "s", [
                ("user", "first new occurrence"),
                ("assistant", "secret sk-ABCD1234efgh5678ijkl"),
            ]
        )
        assert len(good.calls) == 1
        assert good.calls[0][0] == "first new occurrence"
        assert "sk-ABCD1234efgh5678ijkl" not in good.calls[0][1]
        assert "[REDACTED-API-KEY]" in good.calls[0][1]
        assert good.transaction_states == [False]
        assert backlog_id not in {
            row["message_id"] for row in hy.conn.execute(
                "SELECT message_id FROM message_embeddings"
            ).fetchall()
        }
        assert set(new_ids).issubset({
            row["message_id"] for row in hy.conn.execute(
                "SELECT message_id FROM message_embeddings"
            ).fetchall()
        })

        good.calls.clear()
        hy.dream()
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_embeddings WHERE message_id=?", (backlog_id,)
        ).fetchone()[0] == 1
    finally:
        hy.close()


def test_dream_background_first_response_dimension_change_persists_true_shape(cfg):
    scoped = replace(
        cfg,
        aggregation_nodes_enabled=False,
        salience_min_chars=1,
    )
    hy = HyMem(scoped, llm=StubLLMClient(default="[]"))

    class DynamicDimensionEmbedder(RecordingEmbedder):
        def __init__(self, conn):
            super().__init__(dim=99, conn=conn, default=[1.0, 0.0, 0.0])

        def embed(self, texts):
            payload = list(texts)
            self.calls.append(payload)
            self.transaction_states.append(bool(self.conn.in_transaction))
            self._dim = 3
            return [[1.0, 0.0, 0.0] for _ in payload]

    try:
        hy.log_messages(
            "s",
            [
                ("assistant", "We should record the deployment decision in detail."),
                ("user", "Use the cedar deployment path for every production rollout."),
            ],
        )
        hy.close_session("s")
        embedder = DynamicDimensionEmbedder(hy.conn)
        hy.set_embedding_client(embedder)
        report = hy.dream()
        rows = hy.conn.execute(
            "SELECT vector_json,dim,model FROM chunk_embeddings"
        ).fetchall()
        assert report.chunks_embedded >= 1 and rows
        assert all(row["dim"] == 3 and row["model"] == embedder.model for row in rows)
        assert embedder.transaction_states[0] is False
        assert hy.conn.execute(
            "SELECT value FROM schema_meta WHERE key='vec_dim'"
        ).fetchone()[0] == "3"
    finally:
        hy.close()


def test_all_embedding_fetch_batches_reject_provider_model_swap(cfg):
    class ModelSwapEmbedder(RecordingEmbedder):
        def embed(self, texts):
            vectors = super().embed(texts)
            self._model = "space-after-call"
            return vectors

    hy = HyMem(_quiet_cfg(cfg))
    try:
        message_id = hy.log_message("s", "user", "message vector source")
        conn = hy.conn
        conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text) VALUES ('chunk','s',?,?, 'test','chunk source')",
            (message_id, message_id),
        )
        conn.execute(
            "INSERT INTO episodes(id,session_id,title,summary) "
            "VALUES ('episode','s','episode','source')"
        )
        fact_extraction = narrative_facts.extract_facts(
            conn, "s",
            StubLLMClient(default=json.dumps([{"text": "fact source"}])),
            hy.config,
        )
        assert fact_extraction is not None
        with core_db.transaction(conn):
            narrative_facts.persist_facts(conn, "s", fact_extraction)
        conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
            "pos_evidence,neg_evidence,status) "
            "VALUES ('app','uses','sqlite',1,0,'active')"
        )
        conn.execute(
            "INSERT INTO aggregation_nodes(id,title,summary,level) "
            "VALUES ('node','node','source',0)"
        )

        fetches = (
            lambda embedder: fetch_message_embeddings(
                conn, embedder, message_ids=(message_id,)
            ),
            lambda embedder: fetch_chunk_embeddings(conn, embedder),
            lambda embedder: fetch_edge_embeddings(conn, embedder),
            lambda embedder: fetch_episode_embeddings(conn, embedder),
            lambda embedder: fetch_fact_embeddings(conn, embedder),
            lambda embedder: fetch_node_embeddings(conn, embedder),
        )
        for fetch in fetches:
            embedder = ModelSwapEmbedder(
                model="space-before-call", default=[1.0, 0.0, 0.0]
            )
            with pytest.raises(RuntimeError, match="changed model"):
                fetch(embedder)
            assert len(embedder.calls) == 1
    finally:
        hy.close()


def test_edge_fetch_accepts_first_call_dynamic_dimension_without_cache(cfg):
    class DynamicDimEmbedder(RecordingEmbedder):
        def __init__(self):
            super().__init__(model="dynamic-space", dim=99)

        def embed(self, texts):
            self.calls.append(list(texts))
            self._dim = 3
            return [[1.0, 0.0, 0.0] for _ in texts]

    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    try:
        conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
            "pos_evidence,neg_evidence,status) "
            "VALUES ('app','uses','sqlite',1,0,'active')"
        )
        embedder = DynamicDimEmbedder()
        pending = fetch_edge_embeddings(conn, embedder)
        assert pending is not None
        assert pending.dim == 3
        assert pending.model == "dynamic-space"
        assert list(pending.new_text_vectors.values()) == [[1.0, 0.0, 0.0]]
    finally:
        conn.close()


def test_invalid_ingest_never_calls_provider_and_provider_failure_keeps_source(cfg):
    embedder = RecordingEmbedder(fail_on="provider fails")
    hy = HyMem(_quiet_cfg(cfg), embedding_client=embedder)
    try:
        for kwargs in (
            {"session_id": "s", "role": "invalid", "content": "private"},
            {"session_id": "", "role": "user", "content": "private"},
            {
                "session_id": "s", "role": "user", "content": "private",
                "source_peer_id": "alice",
            },
            {
                "session_id": "s", "role": "user", "content": "private",
                "created_at": "2999-01-01T00:00:00Z",
            },
        ):
            with pytest.raises(ValueError):
                hy.log_message(**kwargs)
        assert embedder.calls == []

        message_id = hy.log_message("s", "user", "provider fails but source commits")
        assert embedder.calls == [["provider fails but source commits"]]
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE id=?", (message_id,)
        ).fetchone()[0] == 1
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_retention_coverage WHERE message_id=?",
            (message_id,),
        ).fetchone()[0] == 1
    finally:
        hy.close()


def test_all_semantic_tiers_share_one_query_embedding(cfg):
    query = "opaque semantic prompt"
    vector = [1.0, 0.0, 0.0]
    embedder = RecordingEmbedder({query: vector}, default=vector)
    acfg = replace(
        cfg,
        working_memory_turns=0,
        aggregation_nodes_enabled=True,
        aggregation_inject_abilities=(),
        facts_enabled=True,
    )
    hy = HyMem(acfg, embedding_client=embedder)
    try:
        hy.log_message("messages", "user", "durable occurrence corpus")
        conn = hy.conn
        conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('derived')")
        conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text) VALUES ('chunk','derived',1,1,'test','chunk corpus')"
        )
        conn.execute(
            "INSERT INTO chunk_embeddings(chunk_id,vector_json,model,dim,text_hash) "
            "VALUES ('chunk',?,?,?,?)",
            (encode_vector(vector), embedder.model, embedder.dim,
             embedding_text_hash("chunk corpus")),
        )
        edge_id = conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical,"
            "pos_evidence) VALUES ('alpha','uses','omega',2)"
        ).lastrowid
        conn.execute(
            "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) VALUES (?,?,?,?)",
            ("alpha uses omega", encode_vector(vector), embedder.model, embedder.dim),
        )
        conn.execute(
            "INSERT INTO episodes(id,session_id,title,summary) "
            "VALUES ('episode','derived','episode title','episode corpus')"
        )
        episode_text = "episode title\nepisode corpus"
        conn.execute(
            "INSERT INTO episode_embeddings(episode_id,vector_json,model,dim,text_hash) "
            "VALUES ('episode',?,?,?,?)",
            (encode_vector(vector), embedder.model, embedder.dim,
             embedding_text_hash(episode_text)),
        )
        fact_extraction = narrative_facts.extract_facts(
            conn, "messages",
            StubLLMClient(default=json.dumps([{"text": "fact corpus"}])),
            hy.config,
        )
        assert fact_extraction is not None
        with core_db.transaction(conn):
            narrative_facts.persist_facts(conn, "messages", fact_extraction)
        fact_id = conn.execute(
            "SELECT id FROM narrative_facts WHERE text='fact corpus'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO narrative_fact_embeddings(fact_id,vector_json,model,dim,text_hash) "
            "VALUES (?,?,?,?,?)",
            (fact_id, encode_vector(vector), embedder.model, embedder.dim,
             embedding_text_hash("fact corpus")),
        )
        conn.execute(
            "INSERT INTO aggregation_nodes(id,title,summary,level) "
            "VALUES ('node','node title','node corpus',0)"
        )
        node_text = "node title\nnode corpus"
        conn.execute(
            "INSERT INTO aggregation_node_embeddings(node_id,vector_json,model,dim,text_hash) "
            "VALUES ('node',?,?,?,?)",
            (encode_vector(vector), embedder.model, embedder.dim,
             embedding_text_hash(node_text)),
        )
        embedder.calls.clear()
        ctx = hy.augment(query)
        assert embedder.calls == [[query]]
        assert ctx.semantic_status.available is True
        assert ctx.fts_hits and ctx.message_hits and ctx.episodes and ctx.facts
        assert ctx.aggregation_nodes and any(f.edge_id == edge_id for f in ctx.graph_facts)
    finally:
        hy.close()


def test_stale_content_hashes_are_filtered_from_chunk_episode_fact_and_node(cfg):
    embedder = RecordingEmbedder(default=[1.0, 0.0, 0.0])
    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    vector = encode_vector([1.0, 0.0, 0.0])
    try:
        conn.execute("INSERT INTO sessions(id) VALUES ('s')")
        conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text) VALUES ('c','s',1,1,'test','old chunk')"
        )
        conn.execute(
            "INSERT INTO chunk_embeddings(chunk_id,vector_json,model,dim,text_hash) "
            "VALUES ('c',?,?,?,?)",
            (vector, embedder.model, 3, embedding_text_hash("old chunk")),
        )
        conn.execute(
            "INSERT INTO episodes(id,session_id,title,summary) VALUES ('e','s','old','episode')"
        )
        conn.execute(
            "INSERT INTO episode_embeddings VALUES ('e',?,?,?,?,CURRENT_TIMESTAMP)",
            (vector, embedder.model, 3, embedding_text_hash("old\nepisode")),
        )
        fact_id = conn.execute(
            "INSERT INTO narrative_facts(session_id,start_message_id,end_message_id,"
            "text,prompt_version) VALUES ('s',1,1,'old fact','test')"
        ).lastrowid
        conn.execute(
            "INSERT INTO narrative_fact_embeddings VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
            (fact_id, vector, embedder.model, 3, embedding_text_hash("old fact")),
        )
        conn.execute(
            "INSERT INTO aggregation_nodes(id,title,summary,level) VALUES ('n','old','node',0)"
        )
        conn.execute(
            "INSERT INTO aggregation_node_embeddings VALUES ('n',?,?,?,?,CURRENT_TIMESTAMP)",
            (vector, embedder.model, 3, embedding_text_hash("old\nnode")),
        )
        conn.execute("UPDATE chunks SET text='changed chunk' WHERE id='c'")
        conn.execute("UPDATE episodes SET title='changed' WHERE id='e'")
        conn.execute("UPDATE narrative_facts SET text='changed fact' WHERE id=?", (fact_id,))
        conn.execute("UPDATE aggregation_nodes SET title='changed' WHERE id='n'")
        supplied = [1.0, 0.0, 0.0]
        assert _vector_search(
            conn, embedder, "opaque", top_k=3, max_scan=10,
            query_vector=supplied,
        ) == []
        assert _episode_search(
            conn, "opaque", top_k=3, embedding_client=embedder,
            query_vector=supplied,
        ) == []
        assert _fact_search(
            conn, "opaque", top_k=3, embedding_client=embedder,
            query_vector=supplied,
        ) == []
        assert _aggregation_search(
            conn, "opaque", top_k=3, embedding_client=embedder,
            query_vector=supplied,
        ) == []
    finally:
        conn.close()


def test_aggregation_embeddings_refresh_identity_cache_and_run_without_write_lock(cfg):
    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    embedder = RecordingEmbedder(model="new-space", conn=conn, default=[1.0, 0.0, 0.0])
    try:
        text = "node title\nnode summary"
        text_hash = embedding_text_hash(text)
        conn.execute(
            "INSERT INTO aggregation_nodes(id,title,summary,level) "
            "VALUES ('n','node title','node summary',0)"
        )
        conn.execute(
            "INSERT INTO aggregation_node_embeddings(node_id,vector_json,model,dim,text_hash) "
            "VALUES ('n',?,'old-space',3,?)",
            (encode_vector([1.0, 0.0, 0.0]), text_hash),
        )
        conn.execute(
            "INSERT INTO embedding_cache(text_hash,model,vector_json,dim) "
            "VALUES (?,?,?,3)",
            (text_hash, embedder.model, encode_vector([0.0, 0.0, 0.0])),
        )
        pending = fetch_node_embeddings(conn, embedder)
        assert pending is not None
        assert embedder.calls == [[text]]
        assert embedder.transaction_states == [False]
        with core_db.transaction(conn):
            assert persist_node_embeddings(conn, pending) == 1
        row = conn.execute(
            "SELECT model,dim,vector_json FROM aggregation_node_embeddings WHERE node_id='n'"
        ).fetchone()
        assert (row["model"], row["dim"]) == (embedder.model, embedder.dim)
    finally:
        conn.close()


def test_doctor_reports_same_dim_model_swap_and_malformed_vec_metadata(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HYMEM_ROOT", str(tmp_path))
    for name in (
        "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
        "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = resolve_env()
    conn = core_db.connect(HyMemConfig(root=tmp_path).db_path)
    core_db.initialize(conn)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES ('vec_dim',?)",
            (str(cfg.embedding_dim),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES ('vec_model','wrong-model')"
        )
    finally:
        conn.close()
    results = _check_schema_and_dim(cfg, cfg.embedding_dim)
    assert any(
        result.status == FAIL and "wrong-model" in result.detail
        for result in results
    )

    conn = core_db.connect(HyMemConfig(root=tmp_path).db_path)
    try:
        conn.execute("UPDATE schema_meta SET value='not-an-int' WHERE key='vec_dim'")
        core_db.ensure_vec_table(
            conn, cfg.embedding_dim, model=cfg.embedding_identity
        )
        assert conn.execute(
            "SELECT value FROM schema_meta WHERE key='vec_dim'"
        ).fetchone()[0] == str(cfg.embedding_dim)
    finally:
        conn.close()


def test_doctor_accepts_remote_endpoint_qualified_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("HYMEM_ROOT", str(tmp_path))
    monkeypatch.setenv("HYMEM_EMBEDDING_API_KEY", "key")
    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "https://embed.example/v1/")
    monkeypatch.setenv("HYMEM_EMBEDDING_MODEL", "shared-label")
    monkeypatch.setenv("HYMEM_EMBEDDING_DIM", "3")
    cfg = resolve_env()
    conn = core_db.connect(HyMemConfig(root=tmp_path).db_path)
    core_db.initialize(conn)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES ('vec_dim','3')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES ('vec_model',?)",
            (cfg.embedding_identity,),
        )
    finally:
        conn.close()
    results = _check_schema_and_dim(cfg, 3)
    assert any(
        result.status == "OK" and "matches stored shadows" in result.detail
        for result in results
    )


def test_provider_failure_status_keeps_known_identity(cfg):
    embedder = RecordingEmbedder(model="known-model", dim=3, fail_on="query")
    hy = HyMem(_quiet_cfg(cfg), embedding_client=embedder)
    try:
        status = hy.augment("query").semantic_status
        assert status.attempted is True and status.available is False
        assert status.reason == "provider_error"
        assert (status.backend, status.model, status.dim) == (
            "recording", "known-model", 3,
        )
    finally:
        hy.close()


@pytest.mark.parametrize("broken", ["backend", "quality", "model", "dim"])
def test_raising_embedding_metadata_never_breaks_lexical_retrieval(cfg, broken):
    class RaisingMetadataEmbedder(RecordingEmbedder):
        @property
        def backend(self):
            if broken == "backend":
                raise RuntimeError("bad backend property")
            return "custom"

        @backend.setter
        def backend(self, _value):
            pass

        @property
        def quality(self):
            if broken == "quality":
                raise RuntimeError("bad quality property")
            return "semantic"

        @quality.setter
        def quality(self, _value):
            pass

        @property
        def model(self):
            if broken == "model":
                raise RuntimeError("bad model property")
            return self._model

        @property
        def dim(self):
            if broken == "dim":
                raise RuntimeError("bad dim property")
            return self._dim

        def embed(self, texts):
            raise RuntimeError("provider unavailable")

    hy = HyMem(_quiet_cfg(cfg))
    try:
        message_id = hy.log_message("s", "user", "lexicalneedle survives")
        hy.set_embedding_client(RaisingMetadataEmbedder())
        ctx = hy.augment("lexicalneedle", source_session_id="s")
        assert [hit.message_id for hit in ctx.message_hits] == [message_id]
        assert ctx.semantic_status.available is False
        assert hy.embedding_status["configured"] is True
    finally:
        hy.close()


def test_generic_dream_never_uses_env_coupled_health_probe(monkeypatch, cfg):
    from hymem import api

    monkeypatch.setenv("HYMEM_EMBEDDING_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("HYMEM_EMBEDDING_SERVER_CMD", "should-not-launch")
    monkeypatch.setattr(
        api.urllib.request,
        "urlopen",
        lambda *_a, **_k: pytest.fail("generic client must not perform a health GET"),
    )
    monkeypatch.setattr(
        api.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("generic client must not launch a service"),
    )
    hy = HyMem(
        _quiet_cfg(cfg),
        llm=StubLLMClient(default="[]"),
        embedding_client=LocalHashEmbeddingClient(),
    )
    try:
        hy.dream()
    finally:
        hy.close()


def test_vec_backfill_rejects_wrong_length_instead_of_padding(cfg):
    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    try:
        conn.execute("INSERT INTO sessions(id) VALUES ('s')")
        conn.execute(
            "INSERT INTO episodes(id,session_id,title,summary) VALUES ('e','s','t','s')"
        )
        conn.execute(
            "INSERT INTO episode_embeddings(episode_id,vector_json,model,dim,text_hash) "
            "VALUES ('e',?,'space',3,?)",
            (encode_vector([1.0, 0.0]), embedding_text_hash("t\ns")),
        )
        conn.execute("DELETE FROM schema_meta WHERE key IN ('vec_dim','vec_model')")
        core_db.ensure_vec_table(conn, 3, model="space")
        if core_db.has_vec_table(conn, table="vec_episodes"):
            assert conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0] == 0
    finally:
        conn.close()


def test_identity_matched_vec_ensure_does_not_rescan_durable_corpora(
    monkeypatch, cfg,
):
    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    try:
        for table in core_db._VEC_TABLES:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(rowid INTEGER PRIMARY KEY, embedding BLOB)"
            )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES ('vec_dim','3')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES ('vec_model','space')"
        )
        monkeypatch.setattr(core_db, "_load_vec_extension", lambda _conn: True)

        def unexpected_scan(*_args, **_kwargs):
            pytest.fail("identity-matched ensure must not run a full backfill")

        for name in (
            "_backfill_vec", "_backfill_vec_edges", "_backfill_vec_messages",
            "_backfill_vec_episodes", "_backfill_vec_facts",
        ):
            monkeypatch.setattr(core_db, name, unexpected_scan)
        core_db.ensure_vec_table(conn, 3, model="space")
    finally:
        conn.close()


def test_system_and_tool_coverage_never_surface_as_message_hits(cfg):
    query = "private system needle"
    embedder = RecordingEmbedder({query: [1.0, 0.0, 0.0]}, default=[1.0, 0.0, 0.0])
    hy = HyMem(_quiet_cfg(cfg), embedding_client=embedder)
    try:
        hy.log_messages(
            "s", [("system", query), ("tool", query)]
        )
        assert hy.augment(query, source_session_id="s").message_hits == []
    finally:
        hy.close()
