from __future__ import annotations

import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("requests")
pytest.importorskip("ijson")
_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(_BENCH))
import longmemeval_adapter as lme  # noqa: E402
import beam_adapter as beam  # noqa: E402
from benchmarks.strictness import (  # noqa: E402
    AtomicCheckpoint,
    BenchmarkCleanupError,
    BenchmarkIntegrityError,
    IndexingConvergenceError,
    build_manifest,
    content_hash,
    durable_indexing_status,
    publish_checkpoint_artifact,
)


def test_lme_sampling_is_deterministic_and_label_blind(tmp_path: Path):
    rows = [
        {
            "question_id": f"q{index}",
            "question_type": "rare" if index == 0 else "common",
            "question": f"question {index}",
            "answer": str(index),
        }
        for index in range(12)
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    selected = lme.load_longmemeval_data(str(path), max_questions=4, seed=9)
    assert [row["question_id"] for row in selected] == ["q2", "q3", "q6", "q9"]
    assert [int(row["question_id"][1:]) for row in selected] == sorted(
        int(row["question_id"][1:]) for row in selected
    ), "selection must preserve pinned source order"

    # Labels and qids (including `_abs`) can change without changing selected
    # source positions: the sampler hashes only seed + source index.
    relabelled = [
        {**row, "question_id": f"renamed_{index}_abs",
         "question_type": "other", "answer": "secret"}
        for index, row in enumerate(rows)
    ]
    assert [row["question"] for row in lme.select_label_blind_questions(
        relabelled, sample=4, seed=9
    )] == [row["question"] for row in selected]


def test_lme_default_routing_is_label_free_and_strict_score_keeps_failures():
    assert inspect.signature(lme.evaluate_question).parameters["auto_ability"].default is True
    rows = [
        {"question_id": "a", "question_type": "multi-session", "correct": True},
        {"question_id": "b", "question_type": "multi-session", "correct": None,
         "benchmark_failure": "judge_parse_failure"},
        {"question_id": "c", "question_type": "temporal-reasoning", "correct": False},
    ]
    scores = lme.compute_scores(rows)
    assert scores["multi-session"] == {"accuracy": 0.5, "count": 2}
    assert scores["OVERALL"] == {"accuracy": pytest.approx(1 / 3), "count": 3}


def test_lme_malformed_judge_output_is_a_surfaced_failure():
    class Judge:
        def chat(self, *_args, **_kwargs):
            return "The rubric is difficult to apply."

    verdict, raw = lme.judge_scored(
        Judge(), "single-session-user", "q", "gold", "prediction"
    )
    assert verdict is None
    assert raw.startswith("The rubric")


@pytest.mark.parametrize("key", ["model", "messages", "temperature", "max_tokens"])
def test_lme_extra_body_cannot_override_manifested_request_identity(key):
    with pytest.raises(lme.BenchmarkIntegrityError, match="core field"):
        lme.LLMClient("model", "key", extra_body={key: "forged"})


def test_lme_renderer_counts_headers_and_skips_oversized_early_evidence():
    memories = [
        {"type": "episode", "content": "X" * 500},
        {"type": "message_hit", "content": "compact sentinel", "created_at": "2025-01-01"},
    ]
    rendered = lme._render_answer_context(
        memories,
        None,
        0,
        None,
        None,
        ["Y" * 500],
        narrative_facts=["Z" * 500],
        max_context_chars=100,
    )
    assert "compact sentinel" in rendered
    assert "X" * 100 not in rendered
    assert len(rendered) <= 100
    assert "truncated" in rendered


def test_lme_renderer_hard_token_cap_covers_full_visible_prompt():
    counter = lambda text: len(text.encode("utf-8"))
    prefix = "system:fixed\nuser:CONTEXT:\n"
    suffix = "\n\nQUESTION: q?\n\nANSWER:"
    cap = len((prefix + suffix).encode("utf-8")) + 70
    rendered = lme._render_answer_context(
        [
            {"type": "episode", "content": "資料🙂" * 80},
            {"type": "episode", "content": "small"},
        ],
        None, 0, None, None, None,
        max_context_chars=10_000,
        max_input_tokens=cap,
        token_counter=counter,
        prompt_prefix=prefix,
        prompt_suffix=suffix,
    )
    assert "small" in rendered
    assert counter(prefix + rendered + suffix) <= cap


def test_lme_no_longer_applies_the_legacy_8k_character_cap():
    question = "Q" * 3_500
    messages = lme.build_answer_messages(
        [
            {"type": "episode", "content": "X" * 5_000},
            {"type": "message_hit", "content": "compact sentinel"},
        ],
        question,
    )
    visible = "system:" + messages[0]["content"] + "\nuser:" + messages[1]["content"]
    assert len(visible) > lme.MAX_CONTEXT_CHARS
    assert "compact sentinel" in visible
    assert "X" * 5_000 in visible


def test_lme_retrieval_exception_is_failure_even_if_reader_would_answer_gold(
    monkeypatch,
):
    secret = "LME_PRIVATE_SENTINEL_190"
    detail = (
        f"Bearer {secret} at /home/node/private/{secret}/state.sqlite via "
        f"https://user:{secret}@provider.example/v1?token={secret} "
        + "x" * 20_000
    )

    class BrokenAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def open(self):
            return self

        def close(self):
            pass

        def ingest_sessions(self, sessions, ids, dates, **_kwargs):
            return {"sessions": len(sessions), "messages": 1, "chars": 4}

        def search(self, *_args, **_kwargs):
            raise RuntimeError(detail)

    class WouldHallucinateGold:
        calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            return "gold"

    monkeypatch.setattr(lme, "HyMemAdapter", BrokenAdapter)
    args = SimpleNamespace(
        embeddings=False, rerank_top_k=None, rerank_model=None,
        rerank_message_hits=None, aggregation_nodes=None,
        aggregation_broad=False, episode_granularity=None,
        value_supersession=None, graph_multihop=False,
        graph_multihop_max_hops=None, graph_multihop_decay=None,
        graph_multihop_min_score=None, rules=None, rules_extraction=None,
        facts=None, facts_extraction=None, top_k=3, auto_ability=True,
        no_dream=True, graph_facts_first=False, permissive_default=False,
        distill=False, distill_prompt_version=lme.DEFAULT_DISTILL_PROMPT_VERSION,
        retrieval_only=False, max_input_tokens=lme.DEFAULT_MAX_INPUT_TOKENS,
        keep_db=False,
    )
    reader = WouldHallucinateGold()
    judge = WouldHallucinateGold()
    row = lme._evaluate_one_question(
        0, 1,
        {
            "question_id": "q1", "question_type": "single-session-user",
            "question": "What is it?", "answer": "gold",
            "haystack_sessions": [[{"role": "user", "content": "gold"}]],
            "haystack_session_ids": ["s"], "haystack_dates": ["2025-01-01"],
            "question_date": "2025-01-02", "answer_session_ids": ["s"],
        },
        args, reader, judge, "unused",
    )
    assert row["correct"] is False
    assert "execution_failure" in row["benchmark_failure"]
    assert row["benchmark_failure"] == "execution_failure:RuntimeError"
    wire = json.dumps(row)
    assert secret not in wire and "/home/node/private" not in wire
    assert reader.calls == judge.calls == 0


@pytest.mark.parametrize("module", [lme, beam])
def test_provider_error_sentinel_contains_only_exception_type(monkeypatch, module):
    secret = "PROVIDER_PRIVATE_SENTINEL_884"
    detail = (
        f"Bearer {secret} at /home/node/private/{secret}/key.json via "
        f"https://user:{secret}@provider.example/v1?token={secret}"
    )
    client = module.LLMClient("model", "key")

    def fail(*_args, **_kwargs):
        raise RuntimeError(detail)

    monkeypatch.setattr(client, "_call", fail)
    monkeypatch.setattr(module.time, "sleep", lambda *_args: None)
    result = client.chat([])
    assert result == "[LLM_ERROR:RuntimeError]"
    assert secret not in result
    if hasattr(client, "last_error"):
        assert client.last_error == "RuntimeError"


def test_beam_two_conversation_store_cannot_retrieve_prior_sentinel(tmp_path: Path):
    adapter = beam.HyMemAdapter(tmp_path / "beam.sqlite", api_key="unused")
    adapter.open()
    try:
        adapter.ingest("beam-100K-first", [
            {"role": "user", "content": "PRIVATE_SENTINEL_ORCHID_741"},
        ])
        adapter.ingest("beam-100K-second", [
            {"role": "user", "content": "This independent conversation is about tea."},
        ])
        memories, _count, facts = adapter.search(
            "beam-100K-second", "PRIVATE_SENTINEL_ORCHID_741", top_k=20
        )
        rendered = "\n".join(
            [str(item.get("content", "")) for item in memories] + list(facts)
        )
        assert "PRIVATE_SENTINEL_ORCHID_741" not in rendered
    finally:
        adapter.close()


def test_beam_embedding_backends_are_explicit_and_manifest_bound():
    local = beam.resolve_embedding_config("local-hash")
    disabled = beam.resolve_embedding_config("none")
    semantic = beam.resolve_embedding_config(
        "openai-compatible", model="embed-a",
        base_url="https://embeddings.example/v1", dimension=768,
        deployment_revision="public-revision-v1",
        deployment_tenant="public-tenant-v1",
    )
    assert local["configured"] is True
    assert local["backend"] == "local_feature_hash"
    assert local["dimension"] == 384
    assert local["quality"] == "lexical"
    assert local["network_free"] is True
    assert local["identity_exact"] is True
    assert local["reuse_scope"] == "durable"
    assert local["transport_security"] == "local-no-network"
    assert local["request_model"] == "hymem-local-feature-hash-v1"
    assert local["vector_space_key"].startswith(
        "hymem-embedding-producer-v1:"
    )
    assert disabled["configured"] is False and disabled["quality"] == "none"
    assert semantic["quality"] == "semantic"
    assert semantic["network_free"] is False
    public = beam.public_embedding_config(semantic)
    assert "request_base_url" not in public

    def run_id(embedding):
        return build_manifest(
            benchmark="BEAM", code_sha256=content_hash("code"),
            data_sha256=content_hash("data"),
            config={"label_free_answer_path": True, "embedding": embedding},
            models={"embedding": embedding}, seed=0,
            expected_ids=["q1"], protocol_split="full",
        )["run_id"]

    assert len({run_id(local), run_id(disabled), run_id(public)}) == 3


@pytest.mark.parametrize(
    "url",
    [
        "https://embeddings.example/v1?client_assertion=low-secret",
        "https://embeddings.example/v1?deployment=blue",
    ],
)
def test_beam_embedding_endpoint_rejects_all_query_credentials_and_ambiguity(url):
    with pytest.raises(beam.BenchmarkIntegrityError, match="query parameters"):
        beam.resolve_embedding_config(
            "openai-compatible", base_url=url, model="m", dimension=3,
        )


@pytest.mark.parametrize("bad_dimension", [2, "three", None])
def test_beam_pinned_embedding_client_rejects_provider_dimension_drift(
    bad_dimension,
):
    class Provider:
        model = "provider-space"
        dim = 3

        def embed(self, _texts):
            self.dim = bad_dimension
            return [[1.0, 0.0]]

    wrapped = beam.BenchmarkPinnedEmbeddingClient(
        Provider(), expected_dimension=3
    )
    with pytest.raises(beam.BenchmarkIntegrityError, match="dimension/identity"):
        wrapped.embed(["query"])


def test_embedding_usage_records_identity_and_marks_cross_instance_drift():
    class Meter:
        backend = "openai_compatible"
        quality = "semantic"
        network_free = False
        call_count = request_attempts = successful_responses = 1
        input_count = 1
        input_characters = 5
        prompt_tokens = total_tokens = 2
        total_latency_s = 0.1
        token_usage_available = True
        cost_usd = None

        def __init__(self, model, dim):
            self.model = model
            self.dim = dim

    first = beam.embedding_usage_snapshot(Meter("space-a", 3), configured=True)
    second = beam.embedding_usage_snapshot(Meter("space-b", 4), configured=True)
    assert first["model"].startswith("hymem-embedding-producer-v1:")
    assert (first["dimension"], first["identity_available"]) == (3, True)
    combined = beam.aggregate_embedding_usage_snapshots([first, second])
    assert combined["identity_consistent"] is False
    assert combined["model"] is None and combined["dimension"] is None
    assert combined["backend"] == "mixed"


def test_beam_embedding_backend_reaches_hymem_and_is_honestly_metered(
    tmp_path: Path,
):
    local = beam.HyMemAdapter(
        tmp_path / "local.sqlite", api_key="unused",
        embedding_backend="local-hash",
    )
    local.open()
    try:
        assert local.embedding_client is local.hy._embed
        local.embedding_client.embed(["one", "two"])
        usage = beam.embedding_usage_snapshot(
            local.embedding_client, configured=True
        )
        assert usage["backend"] == "local_feature_hash"
        assert usage["quality"] == "lexical"
        assert usage["network_free"] is True
        assert usage["calls"] == 1 and usage["input_count"] == 2
        from hymem.dreaming.aggregation_material import embedding_storage_identity

        assert usage["model"] == embedding_storage_identity(
            local.embedding_client
        )[0]
        assert usage["dimension"] == 384
        assert usage["request_attempts"] == 0
        assert usage["provider_token_usage_available"] is False
    finally:
        local.close()

    disabled = beam.HyMemAdapter(
        tmp_path / "none.sqlite", api_key="unused",
        embedding_backend="none",
    )
    disabled.open()
    try:
        assert disabled.embedding_client is None
        assert disabled.hy._embed is None
        usage = beam.embedding_usage_snapshot(None, configured=False)
        assert usage["backend"] == "none" and usage["calls"] == 0
        assert usage["provider_token_usage_available"] is False
    finally:
        disabled.close()


def test_beam_embedding_backlog_covers_retained_messages_and_rejects_zero_vector(
    tmp_path: Path,
):
    adapter = beam.HyMemAdapter(
        tmp_path / "coverage.sqlite", api_key="unused",
        embedding_backend="local-hash",
    )
    adapter.open()
    try:
        adapter.ingest("retained", [
            {"role": "user", "content": "durable sentinel"},
        ])
        assert all(
            value == 0 for value in beam.embedding_backlog_status(
                adapter.hy.read_conn, adapter.embedding_client
            ).values()
        )
        message_id = adapter.hy.conn.execute(
            "SELECT id FROM messages WHERE session_id='retained'"
        ).fetchone()[0]
        adapter.hy.conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
        # The lossless coverage proof remains the source corpus after pruning.
        assert beam.embedding_backlog_status(
            adapter.hy.read_conn, adapter.embedding_client
        )["pending_message_embeddings"] == 0
        from hymem.core import db as core_db

        with core_db.embedding_mutation(adapter.hy.conn):
            adapter.hy.conn.execute(
                "UPDATE message_embeddings SET vector_json=? WHERE message_id=?",
                (json.dumps([0.0] * adapter.embedding_client.dim), message_id),
            )
        assert beam.embedding_backlog_status(
            adapter.hy.read_conn, adapter.embedding_client
        )["pending_message_embeddings"] == 1
    finally:
        adapter.close()


def test_beam_embedding_backlog_uses_scalar_queries_not_fetchall():
    class Cursor:
        def fetchone(self):
            return (0,)

        def fetchall(self):
            raise AssertionError("corpus was bulk-materialized")

    class Connection:
        def create_function(self, *_args, **_kwargs):
            pass

        def execute(self, *_args, **_kwargs):
            return Cursor()

    from hymem import StubEmbeddingClient

    status = beam.embedding_backlog_status(
        Connection(), StubEmbeddingClient(model_name="m", dim_value=3)
    )
    assert status == {
        "pending_chunk_embeddings": 0,
        "pending_message_embeddings": 0,
        "pending_edge_embeddings": 0,
        "pending_episode_embeddings": 0,
        "pending_fact_embeddings": 0,
    }

    class DisabledConnection:
        def __getattr__(self, _name):
            raise AssertionError("disabled embeddings must not query the store")

    assert all(
        value == 0
        for value in beam.embedding_backlog_status(
            DisabledConnection(), None
        ).values()
    )
    with pytest.raises(BenchmarkIntegrityError, match="identity is unavailable"):
        beam.embedding_backlog_status(
            Connection(), SimpleNamespace(model="", dim=3)
        )


def _empty_indexing_llm():
    from hymem.extraction.llm import StubLLMClient

    return StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [], "summary": "", "procedures": [],
            }),
        },
        default=json.dumps({
            "triples": [], "markers": [], "complete": True,
        }),
    )


def _control_lme_indexing_budget(monkeypatch, *, status_elapsed: float = 0.0):
    """Control only the caller-owned indexing budget, not the dream's work.

    These are durable-failure classification tests, not ten-second performance
    gates. The real dream still receives the real convergence deadline object
    and follows its deadline-aware path; only that object's existing clock seam
    is controlled. In particular, do not patch the shared time module (which
    would also alter scoring, SQLite timing, and dream lease clocks), or remove
    the deadline (which would enable the best-effort embedding worker).
    """
    state = SimpleNamespace(now=0.0, dream_calls=0, statuses=[])
    real_converge = lme.converge_indexing

    def clock():
        return state.now

    def converge(dream, *, status, **kwargs):
        def run_real_dream(*, deadline):
            state.dream_calls += 1
            assert deadline.clock is clock
            return dream(deadline=deadline)

        def read_real_status():
            value = status()
            state.statuses.append(value)
            # Advance at an explicit semantic boundary, independent of how
            # much wall time the real dream and durable SQLite scan consumed.
            state.now = status_elapsed
            return value

        return real_converge(
            run_real_dream, status=read_real_status, _clock=clock, **kwargs,
        )

    monkeypatch.setattr(lme, "converge_indexing", converge)
    return state


@pytest.mark.parametrize(
    ("status_elapsed", "expected_failure"),
    [
        pytest.param(9.999, "quarantined_extraction", id="before-deadline"),
        pytest.param(10.0, "timeout_after_cycle", id="at-deadline"),
        pytest.param(10.001, "timeout_after_cycle", id="after-deadline"),
    ],
)
def test_lme_durable_status_fails_on_only_current_fact_quarantine(
    tmp_path: Path, monkeypatch, status_elapsed, expected_failure,
):
    from hymem import HyMem, HyMemConfig
    from hymem.dreaming.facts import (
        fact_cursor_retry_unit_key,
        facts_retry_policy_version,
    )

    cfg = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=False,
        episode_granularity_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=True,
        salience_min_chars=1,
        dream_baseline_budget=0,
    )
    budget = _control_lme_indexing_budget(
        monkeypatch, status_elapsed=status_elapsed,
    )
    hy = HyMem(cfg, llm=_empty_indexing_llm())
    try:
        hy.log_message("fact-quarantine", "user", "A current fact retry unit.")
        retry_unit = fact_cursor_retry_unit_key(
            "fact-quarantine", None, None, 0
        )
        retry_identity = facts_retry_policy_version(
            cfg, replay_slice_key=retry_unit, client=hy._llm
        )
        hy.conn.execute(
            "UPDATE sessions SET facts_retry_count=?,"
            "facts_retry_config_version=?,facts_quarantined=1 WHERE id=?",
            (
                cfg.facts_extraction_max_attempts,
                retry_identity,
                "fact-quarantine",
            ),
        )

        adapter = object.__new__(lme.HyMemAdapter)
        adapter.hy = hy
        adapter.embedding_client = None
        adapter.last_indexing_summary = None
        with pytest.raises(IndexingConvergenceError) as failed:
            adapter.dream_and_wait(timeout=10, max_cycles=1)
        summary = failed.value.summary
        assert summary["failure"]["code"] == expected_failure
        assert summary["elapsed_s"] == status_elapsed
        assert summary["cycles"] == len(summary["reports"]) == 1
        assert budget.dream_calls == len(budget.statuses) == 1
        assert budget.statuses[0]["quarantined_facts"] == 1
        assert summary["healthy"] is False
        assert summary["cleanup_errors"] == []
        if expected_failure == "quarantined_extraction":
            assert summary["final_status"]["quarantined"]["quarantined_facts"] == 1
        else:
            # Negative controls: even a genuine quarantine cannot supersede
            # the absolute deadline. A late status is not published as valid.
            assert summary["final_status"] is None
            assert summary["complete"] is False
        run = hy.conn.execute(
            "SELECT ended_at, error FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run is not None and run["ended_at"] is not None
        assert run["error"] is None

        # A recognized quarantine from an old prompt/config generation is not
        # active under the runner's current policy and must not poison a rebuild.
        stale_cfg = replace(
            cfg, dream_digest_max_chars=cfg.dream_digest_max_chars + 1
        )
        stale_identity = facts_retry_policy_version(
            stale_cfg, replay_slice_key=retry_unit, client=hy._llm
        )
        hy.conn.execute(
            "UPDATE sessions SET facts_retry_config_version=? WHERE id=?",
            (stale_identity, "fact-quarantine"),
        )
        status = durable_indexing_status(hy, None)
        assert status["quarantined_facts"] == 0
        assert status["quarantined_facts_malformed"] == 0
    finally:
        hy.close()


def test_lme_durable_status_catches_poison_singleton_message_embedding(
    tmp_path: Path, monkeypatch,
):
    from hymem import HyMem, HyMemConfig

    poison_text = "POISON_MESSAGE_EMBED_ONLY"

    from hymem.extraction.embeddings import MappedStubEmbeddingClient

    cfg = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=False,
        episode_granularity_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        salience_min_chars=1,
        dream_baseline_budget=0,
    )
    budget = _control_lme_indexing_budget(monkeypatch)
    embedding = MappedStubEmbeddingClient(
        dim=3,
        model="poison-singleton-v1",
        default=[1.0, 0.0, 0.0],
        fail_on=poison_text,
    )
    hy = HyMem(
        cfg, llm=_empty_indexing_llm(), embedding_client=embedding
    )
    try:
        message_id = hy.log_message("poison", "user", poison_text)
        assert hy.conn.execute(
            "SELECT 1 FROM message_embeddings WHERE message_id=?",
            (message_id,),
        ).fetchone() is None

        adapter = object.__new__(lme.HyMemAdapter)
        adapter.hy = hy
        adapter.embedding_client = embedding
        adapter.last_indexing_summary = None
        with pytest.raises(IndexingConvergenceError) as failed:
            adapter.dream_and_wait(timeout=10, max_cycles=1)

        summary = failed.value.summary
        assert summary["failure"]["code"] == "cycle_exception"
        assert summary["elapsed_s"] == 0.0
        assert budget.dream_calls == 1
        assert budget.statuses == []
        # An exact content-specific producer failure is now surfaced by the
        # cycle itself.  It must remain audible and cannot be treated as a
        # healthy durable mirror merely because ingestion was best-effort.
        assert summary["final_status"] is None
        assert hy.conn.execute(
            "SELECT 1 FROM message_embeddings WHERE message_id=?",
            (message_id,),
        ).fetchone() is None
    finally:
        hy.close()


def test_beam_semantic_embedding_client_is_passed_through(
    tmp_path: Path, monkeypatch,
):
    sentinel = object()
    seen = {}

    def fake_builder(config, *, api_key=""):
        seen.update(config=config, api_key=api_key)
        return sentinel

    monkeypatch.setattr(beam, "build_embedding_client", fake_builder)
    adapter = beam.HyMemAdapter(
        tmp_path / "semantic.sqlite", api_key="unused",
        embedding_backend="openai-compatible",
        embedding_model="embed-a", embedding_base_url="https://embed.example/v1",
        embedding_dim=768, embedding_api_key="do-not-publish",
        embedding_deployment_revision="fixture-release-2026-09",
        embedding_deployment_tenant="fixture-tenant",
    )
    adapter.open()
    try:
        assert adapter.hy._embed is sentinel
        assert seen["api_key"] == "do-not-publish"
        assert seen["config"]["backend"] == "openai_compatible"
        assert seen["config"]["dimension"] == 768
        assert "do-not-publish" not in json.dumps(
            beam.public_embedding_config(seen["config"])
        )
    finally:
        adapter.close()


def test_beam_search_passes_source_scope_and_fails_closed_on_leak():
    class LeakedHit:
        session_id = "prior-conversation"
        text = "PRIVATE_SENTINEL"
        role = "user"
        source_occurrences = ()

    class Context:
        total_message_matches = 0
        message_hits = [LeakedHit()]
        count_message_hits = []
        recent_turns = []
        fts_hits = []
        facts = []
        episodes = []
        aggregation_nodes = []
        graph_facts = []
        procedures = []
        temporal_events = []
        user_profile = []

    class FakeHy:
        def __init__(self):
            self.kwargs = None

        def augment(self, _query, **kwargs):
            self.kwargs = kwargs
            return Context()

    adapter = object.__new__(beam.HyMemAdapter)
    adapter.hy = FakeHy()
    adapter.embedding_config = {"configured": False}
    with pytest.raises(beam.BenchmarkIntegrityError, match="source isolation"):
        adapter.search("current-conversation", "sentinel")
    assert adapter.hy.kwargs["source_session_id"] == "current-conversation"


def test_beam_configured_embedding_cannot_silently_degrade_to_fts():
    from hymem import StubEmbeddingClient

    embedding_client = StubEmbeddingClient(
        model_name="embed-v1", dim_value=8,
    )
    embedding_config = beam.resolve_embedding_config(
        "local-hash", model="embed-v1", dimension=8,
    )
    context = SimpleNamespace(
        semantic_status=SimpleNamespace(
            configured=True, attempted=True, available=False,
            model=embedding_config["vector_space_key"], dim=8,
            reason="provider_error",
        ),
        total_message_matches=0, message_hits=[], count_message_hits=[],
        recent_turns=[], fts_hits=[], facts=[], episodes=[],
        aggregation_nodes=[], graph_facts=[], procedures=[], temporal_events=[],
        user_profile=[],
    )
    adapter = object.__new__(beam.HyMemAdapter)
    adapter.embedding_config = embedding_config
    adapter.embedding_client = embedding_client
    adapter.hy = SimpleNamespace(augment=lambda *_args, **_kwargs: context)
    with pytest.raises(
        beam.BenchmarkIntegrityError, match="embedding retrieval was unavailable"
    ):
        adapter.search("current", "question")


def test_beam_malformed_judge_reply_is_not_a_semantic_zero():
    class Judge:
        last_finish_reason = "stop"

        def chat(self, *_args, **_kwargs):
            return "not-json"

    result = beam.judge_answer(
        Judge(), "q", "gold", ["contains gold"], "answer", return_raw=True
    )
    assert result["score"] == 0.0
    assert result["judge_parse"] == "unreadable"
    assert result["judge_raw"] == "not-json"


def test_beam_official_prompt_is_pinned_and_contains_no_question_or_gold():
    assert beam.BEAM_UPSTREAM_COMMIT == (
        "b2da22eac88bb0874c64665f13457eb99835774a"
    )
    assert beam.BEAM_OFFICIAL_JUDGE_PROMPT_HASH == (
        "sha256:593373c642a288a7b590577d8a8fc92c3f9a2b70e2f64ad6e59a040a6c56b7f5"
    )
    messages = beam._official_judge_messages(
        "The response mentions ORCHID_RUBRIC.", "ORCHID_RESPONSE"
    )
    assert messages == [{"role": "user", "content": messages[0]["content"]}]
    rendered = messages[0]["content"]
    assert "ORCHID_RUBRIC" in rendered and "ORCHID_RESPONSE" in rendered
    assert "ORCHID_QUESTION" not in rendered
    assert "ORCHID_GOLD" not in rendered
    assert "<rubric_item>" not in rendered
    assert "<llm_response>" not in rendered


def test_beam_official_judge_calls_each_rubric_and_accepts_half_credit():
    class Judge:
        last_finish_reason = "stop"

        def __init__(self):
            self.calls = []
            self.replies = [
                '{"score": 1.0, "reason": "complete"}',
                '{"score": 0.5, "reason": "partial"}',
                '{"score": 0.0, "reason": "absent"}',
            ]

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return self.replies.pop(0)

    judge = Judge()
    result = beam.official_judge_answer(
        judge, ["criterion one", "criterion two", "criterion three"],
        "the model response",
    )
    assert result["judge_parse"] == "ok"
    assert result["scores"] == [1.0, 0.5, 0.0]
    assert result["llm_judge_score"] == pytest.approx(0.5)
    assert len(judge.calls) == 3
    assert all(kwargs == {"temperature": 0.0, "max_tokens": None}
               for _messages, kwargs in judge.calls)
    assert all(len(messages) == 1 and messages[0]["role"] == "user"
               for messages, _kwargs in judge.calls)


@pytest.mark.parametrize(
    "reply,parse",
    [
        ('{"score": 0.7, "reason": "not ternary"}', "invalid_score"),
        ('{"score": 1.0, "reason": ""}', "invalid_reason"),
        ("not json", "unreadable"),
        ("[LLM_ERROR: outage]", "transport"),
    ],
)
def test_beam_official_judge_fails_closed_on_malformed_criterion(reply, parse):
    class Judge:
        last_finish_reason = "stop"

        def chat(self, *_args, **_kwargs):
            return reply

    result = beam.official_judge_answer(Judge(), ["criterion"], "response")
    assert result["score"] == 0.0 and result["scores"] == []
    assert result["judge_parse"] == f"criterion_0_{parse}"


def test_beam_official_configuration_detects_model_or_protocol_override():
    baseline = dict(
        protocol="official", provider="openai", model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1", extra_body={},
    )
    assert beam.is_official_judge_configuration(**baseline) is True
    assert beam.is_official_judge_configuration(
        **{**baseline, "model": "gpt-4.1"}
    ) is False
    assert beam.is_official_judge_configuration(
        **{**baseline, "protocol": "legacy-custom"}
    ) is False
    assert beam.is_official_judge_configuration(
        **{**baseline, "extra_body": {"seed": 1}}
    ) is False


def test_beam_official_scored_path_never_sends_question_or_gold_to_judge():
    class Reader:
        def chat(self, *_args, **_kwargs):
            return "VISIBLE_RESPONSE"

    class Judge:
        last_finish_reason = "stop"

        def __init__(self):
            self.messages = []

        def chat(self, messages, **_kwargs):
            self.messages.extend(messages)
            return '{"score": 1.0, "reason": "satisfied"}'

    class Adapter:
        def search(self, *_args, **_kwargs):
            return ([{"type": "message_hit", "content": "context"}], 0, [])

    q = {
        "question_id": "q1", "ability_short": "IE",
        "question": "ORCHID_QUESTION", "ideal_answer": "ORCHID_IDEAL",
        "gold_text": "ORCHID_GOLD", "gold_kind": "response",
        "rubric": ["contains VISIBLE_RESPONSE"],
    }
    judge = Judge()
    row = beam._evaluate_beam_question(
        True, Reader(), judge, Adapter(),
        {"id": "c", "scale": "100K", "questions": [q]}, q, 0, 3,
        oracle_ability=True, judge_protocol="official",
    )
    rendered = "\n".join(message["content"] for message in judge.messages)
    assert row["score"] == 1.0 and row["result_valid"] is True
    assert row["judged_ideal"] is None
    assert "ORCHID_QUESTION" not in rendered
    assert "ORCHID_IDEAL" not in rendered
    assert "ORCHID_GOLD" not in rendered


def test_beam_partial_official_judge_failure_invalidates_entire_row():
    class Reader:
        def chat(self, *_args, **_kwargs):
            return "answer"

    class Judge:
        last_finish_reason = "stop"

        def __init__(self):
            self.replies = [
                '{"score": 1.0, "reason": "first passed"}', "broken json",
            ]

        def chat(self, *_args, **_kwargs):
            return self.replies.pop(0)

    class Adapter:
        def search(self, *_args, **_kwargs):
            return ([{"type": "message_hit", "content": "answer"}], 0, [])

    q = {
        "question_id": "q1", "ability_short": "IE", "question": "q",
        "ideal_answer": "", "gold_text": "answer", "gold_kind": "response",
        "rubric": ["first", "second"],
    }
    row = beam._evaluate_beam_question(
        True, Reader(), Judge(), Adapter(),
        {"id": "c", "scale": "100K", "questions": [q]}, q, 0, 3,
        oracle_ability=True, judge_protocol="official",
    )
    assert row["result_valid"] is False and row["score"] == 0.0
    assert row["correct"] is False
    assert row["benchmark_failure"] == "judge_criterion_1_unreadable"
    assert len(row["judge_criterion_results"]) == 2


def test_beam_retrieval_exception_cannot_be_scored_as_capability():
    secret = "BEAM_PRIVATE_SENTINEL_491"
    detail = (
        f"Bearer {secret} at /home/node/private/{secret}/state.sqlite via "
        f"https://user:{secret}@provider.example/v1?token={secret} "
        + "x" * 20_000
    )

    class BrokenAdapter:
        def ingest(self, *_args, **_kwargs):
            return {"total_msgs": 1, "total_chars": 4}

        def dream_and_wait(self):
            pass

        def search(self, *_args, **_kwargs):
            raise RuntimeError(detail)

    class WouldHallucinateGold:
        calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            return "gold"

    reader = WouldHallucinateGold()
    judge = WouldHallucinateGold()
    conv = {
        "id": "c", "scale": "100K",
        "messages": [{"role": "user", "content": "gold"}],
        "questions": [{
            "question_id": "beam:100K:c:ordinal:0:x", "ability_short": "IF",
            "question": "What is it?", "ideal_answer": "gold",
            "gold_text": "gold", "rubric": ["contains gold"],
        }],
    }
    output = beam.evaluate_conversation(
        True, reader, judge, BrokenAdapter(), conv, 3,
        oracle_ability=True,
    )
    row = output["questions"][0]
    assert row["score"] == 0.0
    assert row["result_valid"] is False
    assert row["benchmark_failure"] == "execution_failure:RuntimeError"
    wire = json.dumps(row)
    assert secret not in wire and "/home/node/private" not in wire
    assert reader.calls == judge.calls == 0


@pytest.mark.parametrize(
    "error_type", [BenchmarkIntegrityError, BenchmarkCleanupError]
)
def test_beam_question_structural_failure_escapes_item_boundary(
    monkeypatch, error_type,
):
    class Adapter:
        last_indexing_summary = None

        def ingest(self, *_args, **_kwargs):
            return {"total_msgs": 1, "total_chars": 6}

        def dream_and_wait(self, *_args, **_kwargs):
            pass

    primary = error_type("structural question failure")
    monkeypatch.setattr(
        beam, "_evaluate_beam_question",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    conv = {
        "id": "c", "scale": "100K",
        "messages": [{"role": "user", "content": "source"}],
        "questions": [{
            "question_id": "q1", "ability_short": "IE", "question": "q?",
        }],
    }

    with pytest.raises(error_type) as caught:
        beam.evaluate_conversation(
            True, object(), object(), Adapter(), conv, 3,
            oracle_ability=True,
        )

    assert caught.value is primary


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda sample: sample.update(chat={}), "chat must be"),
        (lambda sample: sample.update(chat=[{"role": "user"}]), "chat block"),
        (lambda sample: sample["chat"][0].append("bad"), "must be an object"),
        (lambda sample: sample["chat"][0][0].update(role=7), "malformed role"),
        (lambda sample: sample["chat"][0][0].update(role="moderator"), "unsupported role"),
        (lambda sample: sample["chat"][0][0].update(content=7), "non-string"),
        (lambda sample: sample["chat"][0][0].pop("time_anchor"), "exactly one"),
        (
            lambda sample: sample["chat"][0][0].update(
                time_anchor="not-a-real-date"
            ),
            "unparseable time_anchor",
        ),
        (
            lambda sample: sample.update(
                probing_questions={"future_unknown_ability": []}
            ),
            "unknown ability",
        ),
    ],
)
def test_beam_parser_fails_closed_on_schema_drift(mutate, match):
    sample = {
        "conversation_id": "c1",
        "chat": [[{
            "role": "user", "content": "hello", "time_anchor": "2025-01-02",
        }]],
        "probing_questions": {
            "event_ordering": [{
                "question": "what came first?", "answer": "A then B",
                "rubric": ["A precedes B"],
            }],
        },
    }
    mutate(sample)
    with pytest.raises(beam.BenchmarkIntegrityError, match=match):
        beam._parse_sample(sample, "100K", 0)


def test_beam_parser_marks_recovered_gold_noncanonical():
    sample = {
        "conversation_id": "c1",
        "chat": [[{
            "role": "user", "content": "hello", "time_anchor": "2025-01-02",
        }]],
        "probing_questions": {
            "event_ordering": [{
                "question": "what came first?", "ideal_summary": "A then B",
                "rubric": ["A precedes B"],
            }],
        },
    }
    question = beam._parse_sample(sample, "100K", 0)["questions"][0]
    assert question["gold_text"] == "A then B"
    assert question["gold_resolution"] == "recovered"
    assert question["gold_field"] == "ideal_summary"


def test_beam_seeded_sample_is_deterministic_and_label_blind():
    conversations = [
        {"id": f"c{i}", "questions": [{"ability_short": "IE"}]}
        for i in range(12)
    ]
    first = beam._label_blind_conversation_sample(
        conversations, 4, seed=17, scale="100K"
    )
    mutated = [
        {**conv, "questions": [{"ability_short": "ABS"}]}
        for conv in conversations
    ]
    second = beam._label_blind_conversation_sample(
        mutated, 4, seed=17, scale="100K"
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert [row["id"] for row in first] != [
        row["id"] for row in beam._label_blind_conversation_sample(
            conversations, 4, seed=18, scale="100K"
        )
    ]


def test_beam_load_binds_each_repository_to_its_resolved_revision(monkeypatch):
    calls = []

    def sample(conv_id):
        return {
            "conversation_id": conv_id,
            "chat": [[{
                "role": "user", "content": "hello", "time_anchor": "2025-01-02",
            }]],
            "probing_questions": {
                "event_ordering": [{
                    "question": "what came first?", "answer": "A then B",
                    "rubric": ["A precedes B"],
                }],
            },
        }

    def load_dataset(repo, *, streaming, revision):
        calls.append((repo, streaming, revision))
        if streaming:
            return {"10M": [sample("ten-million")]}
        return {"100K": [sample("hundred-k")]}

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    revisions = {
        beam.BEAM_REPO: "sha-small",
        beam.BEAM_REPO_10M: "sha-large",
    }
    loaded = beam.load_beam_conversations(
        ["100K", "10M"], revisions=revisions, seed=3
    )
    assert set(loaded) == {"100K", "10M"}
    assert calls == [
        (beam.BEAM_REPO, False, "sha-small"),
        (beam.BEAM_REPO_10M, True, "sha-large"),
    ]


def test_beam_loader_rejects_missing_requested_split(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: {"other": []}),
    )
    with pytest.raises(beam.BenchmarkIntegrityError, match="requested split .*500K"):
        beam.load_beam_conversations(
            ["500K"], revisions={beam.BEAM_REPO: "a" * 40}
        )
    with pytest.raises(beam.BenchmarkIntegrityError, match="requested 10M split"):
        beam.load_beam_conversations(
            ["10M"], revisions={beam.BEAM_REPO_10M: "b" * 40}
        )


def test_beam_official_denominator_validation_is_exact():
    per_conversation = [
        {"ability_short": ability}
        for ability in sorted(set(beam.ABILITY_MAP.values())) for _ in range(2)
    ]
    conversations = {
        scale: [
            {"id": f"{scale}-{index}", "questions": list(per_conversation)}
            for index in range(spec["conversations"])
        ]
        for scale, spec in beam.OFFICIAL_BEAM_DENOMINATORS.items()
    }
    beam.validate_official_denominators(
        conversations, list(beam.OFFICIAL_BEAM_DENOMINATORS)
    )
    conversations["100K"][0]["questions"].pop()
    with pytest.raises(beam.BenchmarkIntegrityError, match="denominator mismatch"):
        beam.validate_official_denominators(conversations, ["100K"])


def test_beam_official_denominator_rejects_corrupt_ability_distribution():
    questions = [
        {"ability_short": ability}
        for ability in sorted(set(beam.ABILITY_MAP.values())) for _ in range(2)
    ]
    conversations = {
        "100K": [
            {"id": f"c{index}", "questions": [dict(q) for q in questions]}
            for index in range(20)
        ],
    }
    conversations["100K"][0]["questions"][0]["ability_short"] = "CR"
    with pytest.raises(beam.BenchmarkIntegrityError, match="ability distribution"):
        beam.validate_official_denominators(conversations, ["100K"])


@pytest.mark.parametrize("native_id", ["", "   ", True, [], {}, float("nan")])
def test_beam_parser_rejects_malformed_native_question_ids(native_id):
    sample = {
        "conversation_id": "c1",
        "chat": [[{
            "role": "user", "content": "hello", "time_anchor": "2025-01-02",
        }]],
        "probing_questions": {
            "event_ordering": [{
                "question_id": native_id, "question": "what came first?",
                "answer": "A then B", "rubric": ["A precedes B"],
            }],
        },
    }
    with pytest.raises(beam.BenchmarkIntegrityError, match="native question id"):
        beam._parse_sample(sample, "100K", 0)


def test_beam_parser_normalizes_valid_whitespace_padded_roles():
    sample = {
        "conversation_id": "c1",
        "chat": [[{
            "role": " user ", "content": "hello", "time_anchor": "2025-01-02",
        }]],
        "probing_questions": {
            "event_ordering": [{
                "question_id": " q-1 ", "question": "what came first?",
                "answer": "A then B", "rubric": ["A precedes B"],
            }],
        },
    }
    parsed = beam._parse_sample(sample, "100K", 0)
    assert parsed["messages"][0]["role"] == "user"
    assert parsed["questions"][0]["source_question_id"] == "q-1"


def test_beam_unresolved_revision_is_only_allowed_as_exploratory():
    revisions = {beam.BEAM_REPO: None}
    with pytest.raises(beam.BenchmarkIntegrityError, match="requires resolved"):
        beam.validate_dataset_revision_binding(revisions, canonical=True)
    assert beam.validate_dataset_revision_binding(
        revisions, canonical=False
    ) == (beam.BEAM_REPO,)


@pytest.mark.parametrize("tier", ["fts_hits", "facts", "episodes"])
def test_beam_composite_evidence_requires_positive_complete_provenance(tier):
    class Hit:
        session_id = "current"
        text = "PRIVATE_SENTINEL"
        title = ""
        summary = "PRIVATE_SENTINEL"
        source_occurrences = ()
        source_provenance_complete = False

    context = SimpleNamespace(
        total_message_matches=0, message_hits=[], count_message_hits=[],
        recent_turns=[], fts_hits=[], facts=[], episodes=[],
        aggregation_nodes=[], graph_facts=[], procedures=[], temporal_events=[],
        user_profile=[],
    )
    setattr(context, tier, [Hit()])
    adapter = object.__new__(beam.HyMemAdapter)
    adapter.hy = SimpleNamespace(augment=lambda *_args, **_kwargs: context)
    adapter.embedding_config = {"configured": False}
    with pytest.raises(beam.BenchmarkIntegrityError, match="provenance is absent"):
        adapter.search("current", "sentinel")


def test_beam_graph_evidence_without_citations_is_rejected():
    graph = SimpleNamespace(
        subject="private", predicate="is", object="sentinel", citations=[]
    )
    context = SimpleNamespace(
        total_message_matches=0, message_hits=[], count_message_hits=[],
        recent_turns=[], fts_hits=[], facts=[], episodes=[],
        aggregation_nodes=[], graph_facts=[graph], procedures=[],
        temporal_events=[], user_profile=[],
    )
    adapter = object.__new__(beam.HyMemAdapter)
    adapter.hy = SimpleNamespace(augment=lambda *_args, **_kwargs: context)
    adapter.embedding_config = {"configured": False}
    with pytest.raises(beam.BenchmarkIntegrityError, match="citations are absent"):
        adapter.search("current", "sentinel")


def test_beam_episode_probe_failure_preserves_valid_score(monkeypatch):
    class Reader:
        def chat(self, *_args, **_kwargs):
            return "the answer"

    class Judge:
        last_finish_reason = "stop"

        def chat(self, *_args, **_kwargs):
            return '{"score": 1.0, "reason": "criterion satisfied"}'

    class Adapter:
        def search(self, *_args, **_kwargs):
            return ([{"type": "message_hit", "content": "the answer"}], 0, [])

    monkeypatch.setattr(
        beam, "episode_probe", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("diagnostic exploded")
        )
    )
    conv = {
        "id": "c", "scale": "100K", "questions": [{
            "question_id": "q1", "ability_short": "IE", "question": "q?",
            "ideal_answer": "", "gold_text": "the answer", "gold_kind": "response",
            "rubric": ["contains the answer"],
        }],
    }
    row = beam._evaluate_beam_question(
        True, Reader(), Judge(), Adapter(), conv, conv["questions"][0], 0, 3,
        oracle_ability=True,
    )
    assert row["result_valid"] is True and row["score"] == 1.0
    assert row["probe"] is None
    assert row["probe_error"] == "probe_failure:RuntimeError"
    assert "diagnostic exploded" not in json.dumps(row)


def test_beam_failing_canary_persists_usage_before_exit(tmp_path: Path):
    manifest = build_manifest(
        benchmark="BEAM", code_sha256=content_hash("code"),
        data_sha256=content_hash("data"),
        config={"label_free_answer_path": True}, models={"reader": "stub"},
        seed=0, expected_ids=["q1"], protocol_split="full",
    )
    ledger = AtomicCheckpoint(
        tmp_path / "beam.checkpoint.json", manifest=manifest,
        expected_ids=["q1"], verdict_key="result_valid",
    )

    class FailedCanary:
        call_count = 0
        request_attempts = 1
        successful_responses = 0
        total_latency_s = 0.25
        token_usage_available = False

        def chat(self, *_args, **_kwargs):
            return "[LLM_ERROR: transport down]"

    llm = FailedCanary()
    with pytest.raises(SystemExit):
        beam._run_canary_with_checkpoint(
            ledger, "process-test",
            lambda status: {
                "status": status, "reader_usage": beam.usage_snapshot(llm)
            },
            "answer", llm, [], 12,
        )
    stored = json.loads(ledger.path.read_text())
    usage = stored["execution_segments"][0]["reader_usage"]
    assert usage["request_attempts"] == 1
    assert usage["calls"] == 0
    assert usage["token_usage_available"] is False


@pytest.mark.parametrize(
    "primary_type", [RuntimeError, KeyboardInterrupt, SystemExit],
    ids=["exception", "keyboard_interrupt", "system_exit"],
)
def test_beam_canary_segment_failure_preserves_exact_primary_and_safe_note(
    capsys, primary_type,
):
    primary = primary_type("provider secret=primary-token")
    secondary = RuntimeError("checkpoint secret=secondary-token")
    primaries_seen = []

    class FailingCanary:
        def chat(self, *_args, **_kwargs):
            raise primary

    class FailingLedger:
        def update_execution_segment(self, *_args, **_kwargs):
            primaries_seen.append(sys.exc_info()[1])
            raise secondary

    with pytest.raises(BaseException) as caught:
        beam._run_canary_with_checkpoint(
            FailingLedger(), "process-test", lambda status: {"status": status},
            "answer", FailingCanary(), [], 12,
        )

    assert caught.value is primary
    assert primaries_seen == [primary]
    notes = "\n".join(getattr(primary, "__notes__", ()))
    assert '"stage":"execution_segment_snapshot"' in notes
    assert '"exception_type":"RuntimeError"' in notes
    assert "primary-token" not in notes
    assert "secondary-token" not in notes
    assert "secondary-token" not in capsys.readouterr().err


def test_beam_canary_standalone_segment_failure_raises_unchanged():
    secondary = RuntimeError("standalone segment persistence failure")
    primaries_seen = []

    class SuccessfulCanary:
        def chat(self, *_args, **_kwargs):
            return "ok"

    class FailingLedger:
        def update_execution_segment(self, *_args, **_kwargs):
            primaries_seen.append(sys.exc_info()[1])
            raise secondary

    with pytest.raises(BaseException) as caught:
        beam._run_canary_with_checkpoint(
            FailingLedger(), "process-test", lambda status: {"status": status},
            "answer", SuccessfulCanary(), [], 12,
        )

    assert caught.value is secondary
    assert primaries_seen == [None]


def test_beam_postprocessing_failure_still_publishes_durable_rows(
    tmp_path: Path, monkeypatch,
):
    manifest = build_manifest(
        benchmark="BEAM", code_sha256=content_hash("code"),
        data_sha256=content_hash("data"),
        config={"label_free_answer_path": True}, models={"reader": "stub"},
        seed=0, expected_ids=["q1"], protocol_split="full",
    )
    ledger = AtomicCheckpoint(
        tmp_path / "beam.checkpoint.json", manifest=manifest,
        expected_ids=["q1"], verdict_key="result_valid",
    )
    ledger.record("q1", row={
        "question_id": "q1", "ability": "IE", "score": 1.0,
        "result_valid": True, "correct": True,
    })
    monkeypatch.setattr(
        beam, "compute_scores", lambda _rows: (_ for _ in ()).throw(
            RuntimeError("summary exploded")
        )
    )
    payload, summary, _rows = beam._strict_beam_payload(
        ledger,
        {"100K": [{
            "id": "c", "questions": [{"question_id": "q1", "ability_short": "IE"}],
        }]},
        ["100K"], label_free=True, judge_gold=True,
    )
    assert summary == {}
    assert payload["diagnostic_errors"] == [{
        "stage": "score_summary", "exception_type": "RuntimeError",
    }]
    assert "summary" not in payload
    assert "summary_counts" not in payload
    assert "summary exploded" not in json.dumps(payload)
    archive = tmp_path / "results_20260904T120000Z-strict-deadbeef.json"
    publish_checkpoint_artifact(ledger, archive, payload=payload)
    saved = json.loads(archive.read_text())
    assert saved["execution"]["counts"]["expected"] == 1
    assert saved["per_question"][0]["score"] == 1.0
    assert "summary" not in saved
    assert "summary_counts" not in saved
    ledger.close()

    malformed_ledger = AtomicCheckpoint(
        tmp_path / "malformed-summary.checkpoint.json", manifest=manifest,
        expected_ids=["q1"], verdict_key="result_valid",
    )
    malformed_ledger.record("q1", row={
        "question_id": "q1", "ability": "IE", "score": 1.0,
        "result_valid": True, "correct": True,
    })
    monkeypatch.setattr(beam, "compute_scores", lambda _rows: {"100K": {}})
    malformed_payload, malformed_summary, _ = beam._strict_beam_payload(
        malformed_ledger,
        {"100K": [{
            "id": "c", "questions": [
                {"question_id": "q1", "ability_short": "IE"},
            ],
        }]},
        ["100K"], label_free=True, judge_gold=True,
    )
    assert malformed_summary == {}
    assert "summary" not in malformed_payload
    assert "summary_counts" not in malformed_payload
    assert malformed_payload["diagnostic_errors"] == [{
        "stage": "score_summary",
        "exception_type": "BenchmarkIntegrityError",
    }]
    malformed_ledger.close()

    reconstruction_ledger = AtomicCheckpoint(
        tmp_path / "reconstruction.checkpoint.json", manifest=manifest,
        expected_ids=["q1"], verdict_key="result_valid",
    )
    reconstruction_ledger.record("q1", row={
        "question_id": "q1", "ability": "IE", "score": 1.0,
        "result_valid": True, "correct": True,
    })
    monkeypatch.setattr(
        reconstruction_ledger, "reconcile",
        lambda: (_ for _ in ()).throw(ValueError("private reconstruction detail")),
    )
    reconstruction_payload, reconstruction_summary, reconstructed = (
        beam._strict_beam_payload(
            reconstruction_ledger,
            {"100K": [{
                "id": "c", "questions": [
                    {"question_id": "q1", "ability_short": "IE"},
                ],
            }]},
            ["100K"], label_free=True, judge_gold=True,
        )
    )
    assert reconstruction_summary == {}
    assert reconstructed == []
    assert "summary" not in reconstruction_payload
    assert "summary_counts" not in reconstruction_payload
    assert reconstruction_payload["diagnostic_errors"] == [{
        "stage": "result_reconstruction", "exception_type": "ValueError",
    }]
    assert "private reconstruction detail" not in json.dumps(
        reconstruction_payload
    )
    reconstruction_ledger.close()


def test_beam_partial_callback_failure_materializes_only_still_pending_ids(
    tmp_path: Path,
):
    question_ids = ["q1", "q2"]
    manifest = build_manifest(
        benchmark="BEAM", code_sha256=content_hash("code"),
        data_sha256=content_hash("data"),
        config={"label_free_answer_path": True}, models={"reader": "stub"},
        seed=0, expected_ids=question_ids, protocol_split="full",
    )
    ledger = AtomicCheckpoint(
        tmp_path / "beam.checkpoint.json", manifest=manifest,
        expected_ids=question_ids, verdict_key="result_valid",
    )

    class Reader:
        def chat(self, *_args, **_kwargs):
            return "answer"

    class Judge:
        last_finish_reason = "stop"

        def chat(self, *_args, **_kwargs):
            return '{"score": 1.0, "reason": "criterion satisfied"}'

    class Adapter:
        def ingest(self, *_args, **_kwargs):
            return {"total_msgs": 1, "total_chars": 5}

        def dream_and_wait(self):
            pass

        def search(self, *_args, **_kwargs):
            return ([{"type": "message_hit", "content": "answer"}], 0, [])

    questions = [{
        "question_id": item_id, "ability_short": "IE", "question": "q?",
        "ideal_answer": "", "gold_text": "answer", "gold_kind": "response",
        "rubric": ["contains answer"],
    } for item_id in question_ids]
    conv = {
        "id": "c", "scale": "100K",
        "messages": [{"role": "user", "content": "hello"}],
        "questions": questions,
    }

    def checkpoint_then_crash(row):
        ledger.record(row["question_id"], row=row)
        raise RuntimeError("callback crashed after durable write")

    with pytest.raises(RuntimeError, match="durable write"):
        beam.evaluate_conversation(
            True, Reader(), Judge(), Adapter(), conv, 3,
            oracle_ability=True, pending_ids=set(question_ids),
            on_result=checkpoint_then_crash,
        )
    remaining = set(ledger.pending_ids)
    assert remaining == {"q2"}
    for q in questions:
        if q["question_id"] in remaining:
            ledger.record(q["question_id"], row={
                "question_id": q["question_id"], "score": 0.0,
                "result_valid": False, "correct": False,
                "benchmark_failure": "conversation_failure: callback crashed",
            })
    snapshot = ledger.finalize()
    assert snapshot["counts"]["total_attempts"] == 2
    assert snapshot["entries"]["q1"]["status"] == "completed"
    assert snapshot["entries"]["q2"]["status"] == "failed"


def test_beam_main_isolated_lifecycle_archive_and_terminal_resume_no_clients(
    tmp_path: Path, monkeypatch,
):
    """Exercise the paid-run lifecycle with deterministic, network-free fakes.

    This covers the failure seams that unit helpers cannot: a callback crashes
    after one durable row, adapter cleanup blocks publication, presentation
    raises only after cleanup and publication, independent conversations receive
    independent stores, and a terminal resume republishes without constructing
    provider clients or reopening memory stores.
    """

    def question(item_id: str, ability: str = "IE") -> dict:
        return {
            "question_id": item_id,
            "ability_short": ability,
            "question": f"question {item_id}?",
            "ideal_answer": "",
            "gold_text": f"gold {item_id}",
            "gold_kind": "response",
            "gold_resolution": "exact",
            "rubric": [f"criterion {item_id}"],
        }

    conversations = {
        "100K": [
            {
                "id": "conversation-a", "scale": "100K",
                "messages": [{"role": "user", "content": "private-a"}],
                "questions": [question("q1", "TR"), question("q2")],
            },
            {
                "id": "conversation-b", "scale": "100K",
                "messages": [{"role": "user", "content": "private-b"}],
                "questions": [question("q3")],
            },
        ],
    }
    monkeypatch.setattr(
        beam, "resolve_dataset_revisions",
        lambda _scales, _pin=None: {beam.BEAM_REPO: "a" * 40},
    )
    monkeypatch.setattr(
        beam, "load_beam_conversations",
        lambda *_args, **_kwargs: conversations,
    )
    monkeypatch.setattr(beam, "print_gold_audit", lambda _rows: None)
    monkeypatch.setattr(beam, "code_hash", lambda *_args, **_kwargs: content_hash("code"))

    provider_resolutions = []

    def resolve_provider(spec, _deepseek_key, *, role="answer"):
        provider_resolutions.append((role, spec))
        provider, model, base = beam.parse_provider_spec(spec)
        return model, base, "resolved-key", provider

    monkeypatch.setattr(beam, "resolve_answer_provider", resolve_provider)

    llm_constructions = []
    fail_canary = {"enabled": False}

    class FakeLLM:
        def __init__(self, model, _key, **_kwargs):
            self.model = model
            self.call_count = 0
            self.request_attempts = 0
            self.successful_responses = 0
            self.total_latency_s = 0.0
            self.token_usage_available = False
            self.last_finish_reason = "stop"
            llm_constructions.append(model)

        def chat(self, *_args, **_kwargs):
            self.call_count += 1
            self.request_attempts += 1
            if fail_canary["enabled"]:
                raise RuntimeError("synthetic canary exception")
            self.successful_responses += 1
            return (
                '{"score": 1.0, "reason": "criterion satisfied"}'
                if self.model == "gpt-4.1-mini" else "reader answer"
            )

    monkeypatch.setattr(beam, "LLMClient", FakeLLM)

    live_adapters = []
    open_paths = []
    fail_adapter_close = {"enabled": True}

    class ZeroMeter:
        call_count = 0
        request_attempts = 0
        successful_responses = 0
        total_latency_s = 0.0
        token_usage_available = False

    class FakeAdapter:
        def __init__(self, db_path, **_kwargs):
            self.db_path = Path(db_path)
            self.pipeline_llm = None
            self.embedding_client = None
            self.last_indexing_summary = None
            self.private_state = set()
            self.is_probe = str(self.db_path).startswith("/benchmark-identity/")
            if not self.is_probe:
                live_adapters.append(self)

        def build_config(self):
            from hymem import HyMemConfig
            return HyMemConfig(
                root=self.db_path.parent,
                aggregation_nodes_enabled=False,
                episode_granularity_enabled=False,
            )

        def open(self):
            assert not self.private_state
            self.pipeline_llm = ZeroMeter()
            from tests.archive_evidence_fixtures import healthy_convergence
            self.last_indexing_summary = healthy_convergence({
                "indexing_max_cycles": 100, "indexing_timeout_s": 3600.0,
            })
            open_paths.append(self.db_path)

        def close(self):
            if fail_adapter_close["enabled"]:
                raise RuntimeError("synthetic close failure")

    monkeypatch.setattr(beam, "HyMemAdapter", FakeAdapter)

    def fake_evaluate(
        _judge_gold, _reader, _judge, adapter, conv, _top_k, *,
        pending_ids, on_result, **_kwargs,
    ):
        # If stores were reused, the second conversation would observe the
        # first id here and the assertion would expose write-side interference.
        assert adapter.private_state == set()
        adapter.private_state.add(conv["id"])
        for index, q in enumerate(conv["questions"]):
            if q["question_id"] not in pending_ids:
                continue
            on_result({
                "question_id": q["question_id"], "scale": conv["scale"],
                "conv_id": conv["id"], "ability": q["ability_short"],
                "question": q["question"], "score": 1.0,
                "llm_judge_score": 1.0, "scores": [1.0],
                "judge_protocol": "official", "result_valid": True,
                "correct": True,
            })
            if conv["id"] == "conversation-a" and index == 0:
                raise RuntimeError("synthetic crash after durable callback")

    monkeypatch.setattr(beam, "evaluate_conversation", fake_evaluate)
    monkeypatch.setattr(beam, "print_episode_probe", lambda _rows: None)
    extraction_canary_calls = []
    fail_extraction_canary = {"enabled": False}

    def fake_extraction_canary(**kwargs):
        extraction_canary_calls.append(kwargs)
        policy = beam.extraction_canary_policy()
        client_identity = beam.extraction_canary_client_policy(
            base_url=kwargs["base_url"], model=kwargs["model"],
            thinking=kwargs["thinking"],
        )
        report = {
            **policy,
            "status": "passed",
            "client": {
                "client_class": "fixture.CanaryClient", **client_identity,
            },
            "client_closed": True,
            "completion_calls": policy["normal_pass_completion_calls"],
            "provider_attempts": policy["normal_pass_completion_calls"],
            "initial_prepartition_leaves": policy["expected_prepartition_leaves"],
            "duplicate_triples_collapsed": 0,
            "usage": {
                "calls": policy["normal_pass_completion_calls"],
                "calls_available": True,
                "request_attempts": policy["normal_pass_completion_calls"],
                "request_attempts_available": True,
                "successful_responses": policy["normal_pass_completion_calls"],
                "successful_responses_available": True,
                "prompt_tokens": 20, "completion_tokens": 10,
                "total_tokens": 30, "latency_s": 0.1,
                "cost_usd": None, "token_usage_available": True,
                "latency_available": True, "cost_available": False,
            },
            "matched_supported_claims": 2,
            "missing_expected_claim_indexes": [],
            "valid_triples_returned": 2,
            "valid_markers_returned": 0,
            "execution_path": policy["normal_execution_path"],
            "claim_evidence": [
                {"expected_claim_index": index, **claim}
                for index, claim in enumerate(policy["expected_claims"])
            ],
        }
        if fail_extraction_canary["enabled"]:
            report.update(
                status="failed", failure_reason="clean_empty",
                failure_details=[], matched_supported_claims=0,
                missing_expected_claim_indexes=[0, 1],
                valid_triples_returned=0, claim_evidence=[],
            )
            raise beam.ExtractionCanaryError(
                "synthetic extraction canary failure", report
            )
        return report

    monkeypatch.setattr(
        beam, "run_configured_extraction_canary", fake_extraction_canary
    )

    # The new Phase-1 canary itself fails before reader/judge canaries and
    # before any live memory adapter/store. Its sanitized report is durable in
    # the checkpoint execution segment, and the lease is still recoverable.
    extraction_results = tmp_path / "extraction-canary"
    extraction_argv = [
        "beam_adapter.py", "--scales", "100K", "--sample", "2",
        "--no-prereg", "--api-key", "fake", "--embedding-backend", "none",
        "--results-dir", str(extraction_results),
    ]
    fail_extraction_canary["enabled"] = True
    monkeypatch.setattr(sys, "argv", extraction_argv)
    with pytest.raises(
        beam.ExtractionCanaryError, match="synthetic extraction canary failure"
    ):
        beam.main()
    fail_extraction_canary["enabled"] = False
    extraction_checkpoints = list(
        (extraction_results / "checkpoints").glob("*.json")
    )
    assert len(extraction_checkpoints) == 1
    extraction_checkpoint = json.loads(extraction_checkpoints[0].read_text())
    extraction_segment = extraction_checkpoint["execution_segments"][0]
    assert extraction_segment["extraction_canary"]["status"] == "failed"
    assert extraction_segment["extraction_canary"]["failure_reason"] == "clean_empty"
    extraction_reacquired = AtomicCheckpoint(
        extraction_checkpoints[0], manifest=extraction_checkpoint["manifest"],
        expected_ids=extraction_checkpoint["expected_ids"], resume=True,
        verdict_key="result_valid",
    )
    extraction_reacquired.close()
    assert open_paths == []

    # A canary exception occurs before any memory store opens. The public main
    # wrapper must still release the checkpoint lease even while pytest retains
    # the exception/traceback in this process.
    canary_results = tmp_path / "canary"
    canary_argv = [
        "beam_adapter.py", "--scales", "100K", "--sample", "2",
        "--no-prereg", "--api-key", "fake", "--embedding-backend", "none",
        "--results-dir", str(canary_results),
    ]
    fail_canary["enabled"] = True
    monkeypatch.setattr(sys, "argv", canary_argv)
    with pytest.raises(RuntimeError, match="canary exception"):
        beam.main()
    fail_canary["enabled"] = False
    canary_checkpoints = list((canary_results / "checkpoints").glob("*.json"))
    assert len(canary_checkpoints) == 1
    raw_checkpoint = json.loads(canary_checkpoints[0].read_text())
    reacquired = AtomicCheckpoint(
        canary_checkpoints[0], manifest=raw_checkpoint["manifest"],
        expected_ids=raw_checkpoint["expected_ids"], resume=True,
        verdict_key="result_valid",
    )
    reacquired.close()
    assert open_paths == []

    first_results = tmp_path / "first"
    argv = [
        "beam_adapter.py", "--scales", "100K", "--sample", "2",
        "--no-prereg", "--api-key", "fake", "--embedding-backend", "none",
        "--results-dir", str(first_results),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(BenchmarkIntegrityError, match="cleanup failed") as cleanup:
        beam.main()
    assert "synthetic close failure" not in str(cleanup.value)
    assert not list(first_results.glob("results_*-strict-*.json"))
    assert not (first_results / "results_latest.json").exists()

    checkpoints = list((first_results / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 1
    fail_adapter_close["enabled"] = False
    monkeypatch.setattr(
        sys, "argv", argv + ["--resume-from", str(checkpoints[0])],
    )
    monkeypatch.setattr(
        beam, "print_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic presentation failure")
        ),
    )
    with pytest.raises(RuntimeError, match="presentation failure"):
        beam.main()

    archives = list(first_results.glob("results_*-strict-*.json"))
    assert len(archives) == 1
    saved = json.loads(archives[0].read_text())
    assert saved["execution"]["counts"] == {
        "expected": 3, "attempted": 3, "unique_attempted": 3,
        "total_attempts": 3, "completed": 2, "failed": 1, "missing": 0,
    }
    assert [row["question_id"] for row in saved["per_question"]] == [
        "q1", "q2", "q3",
    ]
    assert saved["per_question"][1]["result_valid"] is False
    assert saved["per_question"][1]["benchmark_failure"] == (
        "conversation_failure:RuntimeError"
    )
    assert "synthetic crash after durable callback" not in json.dumps(saved)
    first_segment = saved["execution"]["segments"][0]
    from benchmarks.archive_evidence import validate_convergence_summary
    for segment in saved["execution"]["segments"]:
        for receipt in segment["indexing_runs"]:
            assert receipt["scale"] == "100K"
            assert receipt["conversation_id"] in {"conversation-a", "conversation-b"}
            assert validate_convergence_summary({key: value for key, value in receipt.items()
                                                if key not in {"scale", "conversation_id"}}, config=saved["config"])
        if segment.get("latest_indexing") is not None:
            assert segment["latest_indexing"]["scale"] == "100K"
            assert segment["latest_indexing"]["conversation_id"] in {"conversation-a", "conversation-b"}
    assert first_segment["extraction_canary"]["status"] == "passed"
    assert first_segment["extraction_canary"]["usage_accounting"] == (
        beam.extraction_canary_policy()["usage_accounting"]
    )
    assert first_segment["memory_pipeline_usage"]["calls"] == 0
    # This lifecycle fixture intentionally abbreviates BEAM to IE/TR instead
    # of constructing the benchmark's ten-ability denominator. The adapter
    # must therefore preserve its rows while declining to publish a strict
    # derived summary that the registry would reject.
    assert saved["diagnostic_errors"] == [{
        "stage": "score_summary",
        "exception_type": "BenchmarkIntegrityError",
    }]
    assert "summary" not in saved
    assert "summary_counts" not in saved
    assert len(open_paths) == 2 and len(set(open_paths)) == 2
    assert [adapter.private_state for adapter in live_adapters] == [
        {"conversation-a"}, {"conversation-b"},
    ]

    client_count = len(llm_constructions)
    provider_count = len(provider_resolutions)
    open_count = len(open_paths)

    # A complete checkpoint can be finalized into another results directory
    # without resolving keys, constructing provider clients, or reopening DBs.
    second_results = tmp_path / "second"
    monkeypatch.setattr(
        sys, "argv", argv[:-2] + [
            "--results-dir", str(second_results),
            "--resume-from", str(checkpoints[0]),
        ],
    )
    monkeypatch.setattr(beam, "print_report", lambda *_args, **_kwargs: None)
    beam.main()
    assert len(llm_constructions) == client_count
    assert len(provider_resolutions) == provider_count
    assert len(open_paths) == open_count
    # Once for each pending run configuration above; never per conversation,
    # terminal resume, or calibration-only invocation.
    assert len(extraction_canary_calls) == 4
    republished = list(second_results.glob("results_*-strict-*.json"))
    assert len(republished) == 1
    assert json.loads(republished[0].read_text())["execution"]["counts"] == saved[
        "execution"
    ]["counts"]
    beam.main()
    assert len(list(second_results.glob("results_*-strict-*.json"))) == 2
    assert len(llm_constructions) == client_count
    assert len(open_paths) == open_count

    # Calibration freezing also happens before API-key resolution or clients.
    receipt = tmp_path / "beam-calibration.json"
    monkeypatch.setattr(sys, "argv", [
        "beam_adapter.py", "--scales", "100K", "--sample", "0",
        "--oracle-ability", "--no-prereg", "--embedding-backend", "none",
        "--freeze-calibration", str(receipt),
        "--results-dir", str(tmp_path / "freeze"),
    ])
    beam.main()
    assert receipt.exists()
    assert len(llm_constructions) == client_count
    assert len(provider_resolutions) == provider_count
    assert len(open_paths) == open_count


@pytest.mark.parametrize(
    "failure_site",
    [
        "checkpoint",
        "evaluator",
        "inner_question",
        "checkpoint_and_segment",
        "evaluator_and_segment",
        "interrupt_and_segment",
        "segment_only",
    ],
)
def test_beam_structural_conversation_failure_aborts_before_publication(
    monkeypatch, tmp_path, failure_site,
):
    """Conversation primaries survive a failing final segment snapshot."""

    question = {
        "question_id": "q1", "ability_short": "IE", "question": "q?",
        "ideal_answer": "", "gold_text": "gold", "gold_kind": "response",
        "gold_resolution": "exact", "rubric": ["criterion"],
    }
    conversations = {"100K": [{
        "id": "conversation-a", "scale": "100K",
        "messages": [{"role": "user", "content": "source"}],
        "questions": [question],
    }]}
    monkeypatch.setattr(
        beam, "resolve_dataset_revisions",
        lambda _scales, _pin=None: {beam.BEAM_REPO: "a" * 40},
    )
    monkeypatch.setattr(
        beam, "load_beam_conversations", lambda *_a, **_k: conversations,
    )
    monkeypatch.setattr(beam, "print_gold_audit", lambda _rows: None)
    monkeypatch.setattr(beam, "print_report", lambda *_a, **_k: None)
    monkeypatch.setattr(beam, "print_episode_probe", lambda *_a, **_k: None)
    monkeypatch.setattr(
        beam, "resolve_answer_provider",
        lambda spec, _key, role="answer": (
            beam.parse_provider_spec(spec)[1],
            beam.parse_provider_spec(spec)[2],
            "resolved-key",
            beam.parse_provider_spec(spec)[0],
        ),
    )

    class MeterOnlyClient:
        call_count = 0
        request_attempts = 0
        successful_responses = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_latency_s = 0.0
        cost_usd = 0.0
        token_usage_available = True
        last_finish_reason = "stop"

        def __init__(self, model, _key, **_kwargs):
            self.model = model

        def close(self):
            pass

    class FakeAdapter:
        def __init__(self, db_path, **_kwargs):
            self.db_path = Path(db_path)
            self.pipeline_llm = None
            self.embedding_client = None
            self.last_indexing_summary = None

        def build_config(self):
            from hymem import HyMemConfig
            return HyMemConfig(root=self.db_path.parent)

        def open(self):
            self.pipeline_llm = MeterOnlyClient("pipeline", "key")
            self.last_indexing_summary = {
                "cycles": 1, "converged": True, "pending_total": 0,
            }

        def ingest(self, *_args, **_kwargs):
            return {"total_msgs": 1, "total_chars": 6}

        def dream_and_wait(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(beam, "LLMClient", MeterOnlyClient)
    monkeypatch.setattr(beam, "HyMemAdapter", FakeAdapter)
    monkeypatch.setattr(
        beam, "run_configured_extraction_canary", lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        beam, "validate_extraction_canary_report", lambda *_a, **_k: {},
    )
    monkeypatch.setattr(beam, "print_extraction_canary", lambda *_a, **_k: None)
    monkeypatch.setattr(
        beam, "_run_canary_with_checkpoint", lambda *_a, **_k: "ok",
    )

    original_record = beam.AtomicCheckpoint.record
    original_update_segment = beam.AtomicCheckpoint.update_execution_segment
    record_calls = 0
    fail_segment_update = False
    segment_primary_seen = []
    checkpoint_primary = BenchmarkIntegrityError(
        "original checkpoint failure with secret=primary-token"
    )
    evaluator_primary = BenchmarkIntegrityError(
        "original evaluator failure with secret=evaluator-token"
    )
    interrupt_primary = KeyboardInterrupt(
        "original interrupt with secret=interrupt-token"
    )
    segment_failure = RuntimeError(
        "secondary segment failure with secret=secondary-token"
    )

    def maybe_fail_record(self, *args, **kwargs):
        nonlocal fail_segment_update, record_calls
        record_calls += 1
        if failure_site in {"checkpoint", "checkpoint_and_segment"} \
                and record_calls == 1:
            if failure_site == "checkpoint_and_segment":
                fail_segment_update = True
            raise checkpoint_primary
        return original_record(self, *args, **kwargs)

    monkeypatch.setattr(beam.AtomicCheckpoint, "record", maybe_fail_record)

    def maybe_fail_segment_update(self, *args, **kwargs):
        if fail_segment_update:
            segment_primary_seen.append(sys.exc_info()[1])
            raise segment_failure
        return original_update_segment(self, *args, **kwargs)

    monkeypatch.setattr(
        beam.AtomicCheckpoint,
        "update_execution_segment",
        maybe_fail_segment_update,
    )

    def evaluate(*_args, pending_ids, on_result, **_kwargs):
        nonlocal fail_segment_update
        if failure_site == "evaluator":
            raise BenchmarkIntegrityError("synthetic evaluator integrity failure")
        if failure_site == "evaluator_and_segment":
            fail_segment_update = True
            raise evaluator_primary
        if failure_site == "interrupt_and_segment":
            fail_segment_update = True
            raise interrupt_primary
        assert pending_ids == {"q1"}
        on_result({
            "question_id": "q1", "scale": "100K",
            "conv_id": "conversation-a", "ability": "IE",
            "question": "q?", "score": 1.0, "llm_judge_score": 1.0,
            "scores": [1.0], "judge_protocol": "official",
            "result_valid": True, "correct": True,
        })
        if failure_site == "segment_only":
            fail_segment_update = True

    if failure_site == "inner_question":
        monkeypatch.setattr(
            beam, "_evaluate_beam_question",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                BenchmarkIntegrityError("synthetic inner integrity failure")
            ),
        )
    else:
        monkeypatch.setattr(beam, "evaluate_conversation", evaluate)
    results_dir = tmp_path / "results"
    checkpoint = tmp_path / f"beam-{failure_site}.checkpoint.json"
    monkeypatch.setattr(sys, "argv", [
        "beam_adapter.py", "--scales", "100K", "--sample", "1",
        "--no-prereg", "--api-key", "fake", "--embedding-backend", "none",
        "--results-dir", str(results_dir), "--checkpoint", str(checkpoint),
    ])

    with pytest.raises(BaseException) as caught:
        beam.main()

    if failure_site == "checkpoint_and_segment":
        assert segment_primary_seen == [caught.value]
        assert type(caught.value).__name__ == "_CheckpointPersistenceAbort"
        assert caught.value.__cause__ is checkpoint_primary
        notes = "\n".join(getattr(caught.value, "__notes__", ()))
        assert '"stage":"execution_segment_snapshot"' in notes
        assert '"exception_type":"RuntimeError"' in notes
        assert "secondary-token" not in notes
    elif failure_site == "evaluator_and_segment":
        assert segment_primary_seen == [caught.value]
        assert type(caught.value).__name__ == "_CheckpointPersistenceAbort"
        assert caught.value.__cause__ is evaluator_primary
        notes = "\n".join(getattr(caught.value, "__notes__", ()))
        assert '"stage":"execution_segment_snapshot"' in notes
        assert '"exception_type":"RuntimeError"' in notes
        assert "secondary-token" not in notes
    elif failure_site == "interrupt_and_segment":
        assert caught.value is interrupt_primary
        assert segment_primary_seen == [interrupt_primary]
        notes = "\n".join(getattr(caught.value, "__notes__", ()))
        assert '"stage":"execution_segment_snapshot"' in notes
        assert '"exception_type":"RuntimeError"' in notes
        assert "secondary-token" not in notes
    elif failure_site == "segment_only":
        assert caught.value is segment_failure
        assert segment_primary_seen == [None]
    else:
        assert type(caught.value).__name__ == "_CheckpointPersistenceAbort"
        assert isinstance(caught.value.__cause__, BenchmarkIntegrityError)
    assert record_calls == (
        1 if failure_site in {
            "checkpoint", "checkpoint_and_segment", "segment_only",
        } else 0
    )
    state = json.loads(checkpoint.read_text())
    assert state["status"] == "running"
    if failure_site == "segment_only":
        assert list(state["entries"]) == ["q1"]
    else:
        assert state["entries"] == {}
    assert not list(results_dir.glob("results_*-strict-*.json"))
    assert not (results_dir / "results_latest.json").exists()

    resumed = beam.AtomicCheckpoint(
        checkpoint,
        manifest=state["manifest"],
        expected_ids=state["expected_ids"],
        resume=True,
        verdict_key="result_valid",
    )
    try:
        assert resumed.pending_ids == (
            () if failure_site == "segment_only" else ("q1",)
        )
    finally:
        resumed.close()
