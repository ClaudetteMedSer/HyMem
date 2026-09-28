"""A recoverable indexing failure must not prevent the next LME question."""
from copy import deepcopy
from types import SimpleNamespace

from benchmarks import lme_protocol as protocol
from benchmarks import longmemeval_adapter as lme
from benchmarks.strictness import strict_accuracy
from tests.test_lme_mixed_failure_envelope import _raw_failure


def test_failed_indexing_keeps_its_zero_and_next_question_can_complete(monkeypatch):
    raw = _raw_failure("quarantined_extraction", "pending:pending_digests")
    raw["reports"][0]["digest_failures"] = 1
    events = []

    class Fork:
        def dream(self, **kwargs):
            events.append("dream")
            return deepcopy(raw["reports"][0])

        def close(self):
            events.append("fork_close")

    class Parent:
        def fork(self):
            return Fork()

        def invalidate_query_caches(self):
            events.append("invalidate")

    adapters = []

    def make_adapter(*args, **kwargs):
        adapter = object.__new__(lme.HyMemAdapter)
        adapter.hy = Parent()
        adapter.pipeline_llm = None
        adapter.embedding_client = None
        adapter.last_indexing_summary = None
        adapter.open = lambda: adapter
        adapter.close = lambda: events.append("adapter_close")
        adapters.append(adapter)
        return adapter

    def evaluate(answer_llm, judge_llm, adapter, question, *args, **kwargs):
        events.append(question["question_id"])
        if question["question_id"] == "failed-first":
            adapter.dream_and_wait(timeout=10.0, max_cycles=3)
            raise AssertionError("Mixed quarantine must never reach scoring")
        return {
            "question_id": question["question_id"],
            "question_type": "multi-session", "correct": True,
        }

    monkeypatch.setattr(lme, "_adapter_for_args", make_adapter)
    monkeypatch.setattr(lme, "evaluate_question", evaluate)
    monkeypatch.setattr(lme, "durable_indexing_status", lambda *_: deepcopy(raw["final_status"]))
    args = SimpleNamespace(
        keep_db=False, embeddings=False, top_k=5, auto_ability=True,
        no_dream=False, graph_facts_first=False, permissive_default=False,
        distill=False, distill_prompt_version=lme.DEFAULT_DISTILL_PROMPT_VERSION,
        retrieval_only=False, max_input_tokens=16000, max_input_bytes=60000,
        token_counter=None, judge_protocol="legacy-custom",
        indexing_max_cycles=3, indexing_timeout_s=10.0,
        indexing_require_healthy=True,
    )
    rows = [lme._evaluate_one_question(
        index, 2, {"question_id": question_id, "question_type": "multi-session"},
        args, object(), object(), "unused",
    ) for index, question_id in enumerate(("failed-first", "healthy-second"))]
    assert rows[0]["correct"] is False
    assert rows[0]["benchmark_failure"] == "indexing_failure:quarantined_extraction"
    assert rows[0]["indexing"]["complete"] is False
    assert rows[0]["indexing"]["healthy"] is False
    assert protocol._validate_indexing(rows[0]["indexing"], allow_incomplete=True) is False
    assert rows[1]["correct"] is True
    assert events.count("adapter_close") == 2
    assert events.count("dream") == 1
    assert events.index("failed-first") < events.index("healthy-second")
    assert len(adapters) == 2
    assert strict_accuracy(rows) == 0.5
