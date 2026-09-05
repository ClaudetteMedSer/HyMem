from __future__ import annotations

import inspect
import json
import sys
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
    build_manifest,
    content_hash,
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
            raise RuntimeError("augment exploded")

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
    assert "augment exploded" in row["benchmark_failure"]
    assert reader.calls == judge.calls == 0


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
    )
    assert local == {
        "configured": True,
        "backend": "local-hash",
        "model": "hymem-local-feature-hash-v1",
        "base_url": "local://feature-hash",
        "dimension": 384,
        "quality": "lexical-feature-hash",
        "network_free": True,
        "fallback_policy": "none",
        "fallback_reason": None,
    }
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
    assert (first["model"], first["dimension"], first["identity_available"]) == (
        "space-a", 3, True,
    )
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
        assert usage["model"] == local.embedding_client.model
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

    status = beam.embedding_backlog_status(
        Connection(), SimpleNamespace(model="m", dim=3)
    )
    assert status == {
        "pending_chunk_embeddings": 0,
        "pending_message_embeddings": 0,
        "pending_edge_embeddings": 0,
        "pending_episode_embeddings": 0,
        "pending_fact_embeddings": 0,
    }


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
    )
    adapter.open()
    try:
        assert adapter.hy._embed is sentinel
        assert seen["api_key"] == "do-not-publish"
        assert seen["config"]["backend"] == "openai-compatible"
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
    context = SimpleNamespace(
        semantic_status=SimpleNamespace(
            configured=True, attempted=True, available=False,
            model="embed-v1", dim=8, reason="provider_error",
        ),
        total_message_matches=0, message_hits=[], count_message_hits=[],
        recent_turns=[], fts_hits=[], facts=[], episodes=[],
        aggregation_nodes=[], graph_facts=[], procedures=[], temporal_events=[],
        user_profile=[],
    )
    adapter = object.__new__(beam.HyMemAdapter)
    adapter.embedding_config = {"configured": True, "dimension": 8}
    adapter.embedding_client = SimpleNamespace(model="embed-v1", dim=8)
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
    class BrokenAdapter:
        def ingest(self, *_args, **_kwargs):
            return {"total_msgs": 1, "total_chars": 4}

        def dream_and_wait(self):
            pass

        def search(self, *_args, **_kwargs):
            raise RuntimeError("augment exploded")

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
    assert "augment exploded" in row["benchmark_failure"]
    assert reader.calls == judge.calls == 0


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
    assert "diagnostic exploded" in row["probe_error"]


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
    assert "summary exploded" in payload["diagnostic_errors"][0]
    archive = tmp_path / "results_20260904T120000Z-strict-deadbeef.json"
    publish_checkpoint_artifact(ledger, archive, payload=payload)
    saved = json.loads(archive.read_text())
    assert saved["execution"]["counts"]["expected"] == 1
    assert saved["per_question"][0]["score"] == 1.0


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
    after one durable row, adapter cleanup raises, presentation raises after
    publication, independent conversations receive independent stores, and a
    terminal resume republishes without constructing provider clients or
    reopening memory stores.
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
            self.last_indexing_summary = {
                "cycles": 1, "converged": True, "pending_total": 0,
            }
            open_paths.append(self.db_path)

        def close(self):
            # Cleanup failures are diagnostics and cannot strand durable rows.
            if self is live_adapters[0]:
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
    assert any(
        "synthetic close failure" in error
        for error in saved["diagnostic_errors"]
    )
    assert len(open_paths) == 2 and len(set(open_paths)) == 2
    assert [adapter.private_state for adapter in live_adapters] == [
        {"conversation-a"}, {"conversation-b"},
    ]

    checkpoints = list((first_results / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 1
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
