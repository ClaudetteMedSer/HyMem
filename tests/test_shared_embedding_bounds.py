from __future__ import annotations

import pytest

from hymem.core.embedding_batches import (
    EmbeddingInputTooLarge,
    EmbeddingResponseInvalid,
    embed_bounded,
    plan_embedding_batches,
)
from hymem.extraction.embeddings import MappedStubEmbeddingClient
from hymem.query.augment import _query_embedding_with_status


class RecordingEmbedder:
    def __init__(self):
        self.calls = []
        self.dim = 2
        self.reply = None

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.reply is not None:
            return self.reply(texts, len(self.calls))
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]


def run(client, texts, *, boundary=None):
    return embed_bounded(
        client, texts, identity=lambda: ("stable", client.dim),
        expected_model="stable", boundary=boundary,
    )


def test_count_and_order_and_duplicate_preservation():
    client = RecordingEmbedder()
    texts = [f"item {i % 7}" for i in range(17)]
    vectors = run(client, texts)
    assert client.calls == [texts[:16], texts[16:]]
    assert len(vectors) == len(texts)
    assert client.calls[0][0] == client.calls[0][7]


def test_character_and_byte_boundaries():
    assert len(plan_embedding_batches(["a" * 128_000])) == 1
    assert [len(batch) for batch in plan_embedding_batches(["a" * 64_001, "b" * 64_000])] == [1, 1]
    assert len(plan_embedding_batches(["🪷" * 128_000])) == 1
    assert [len(batch) for batch in plan_embedding_batches(["🪷" * 64_001, "🪷" * 64_000])] == [1, 1]


@pytest.mark.parametrize("invalid", ["a" * 128_001, "🪷" * 128_001])
def test_late_oversize_preflights_whole_list_without_provider_call(invalid):
    client = RecordingEmbedder()
    with pytest.raises(EmbeddingInputTooLarge, match="request limit"):
        run(client, ["a"] * 17 + [invalid])
    assert client.calls == []


@pytest.mark.parametrize("invalid", [None, 123, "\ud800"])
def test_invalid_source_never_dispatches_earlier_batches(invalid):
    client = RecordingEmbedder()
    with pytest.raises(ValueError):
        run(client, ["a"] * 17 + [invalid])
    assert client.calls == []


@pytest.mark.parametrize("reply", [
    lambda texts, n: [] if n == 1 else [[1.0, 2.0] for _ in texts],
    lambda texts, n: [[float("nan"), 2.0] for _ in texts],
    lambda texts, n: [[0.0, 0.0] for _ in texts],
])
def test_bad_first_response_stops_second_request(reply):
    client = RecordingEmbedder()
    client.reply = reply
    with pytest.raises(RuntimeError):
        run(client, ["a"] * 17)
    assert len(client.calls) == 1


def test_identity_change_stops_second_request():
    client = RecordingEmbedder()
    client.reply = lambda texts, n: (
        setattr(client, "dim", 3) or [[1.0, 2.0] for _ in texts]
    )
    with pytest.raises(RuntimeError):
        run(client, ["a"] * 17)
    assert len(client.calls) == 1


def test_identity_lookup_failure_stops_second_request():
    client = RecordingEmbedder()

    def broken_identity():
        raise ValueError("tampered producer")

    with pytest.raises(EmbeddingResponseInvalid):
        embed_bounded(
            client, ["a"] * 17, identity=broken_identity,
            expected_model="stable",
        )
    assert len(client.calls) == 1


def test_boundary_checks_before_and_after_each_request():
    client = RecordingEmbedder()
    checks = []

    def boundary():
        checks.append(len(client.calls))

    run(client, ["a"] * 17, boundary=boundary)
    assert checks == [0, 1, 1, 2]

    def reject_second():
        if len(client.calls) == 1:
            raise RuntimeError("deadline")

    client = RecordingEmbedder()
    with pytest.raises(RuntimeError, match="deadline"):
        run(client, ["a"] * 17, boundary=reject_second)
    assert len(client.calls) == 1


def test_dispatch_accounting_counts_actual_requests_only():
    client = RecordingEmbedder()
    attempts = []
    embed_bounded(
        client, ["a"] * 17,
        identity=lambda: ("stable", client.dim), expected_model="stable",
        on_dispatch=lambda: attempts.append(len(client.calls)),
    )
    assert attempts == [0, 1]
    with pytest.raises(EmbeddingInputTooLarge):
        embed_bounded(
            client, ["a" * 128_001],
            identity=lambda: ("stable", client.dim), expected_model="stable",
            on_dispatch=lambda: attempts.append(len(client.calls)),
        )
    assert attempts == [0, 1]


def test_query_local_oversize_is_not_a_provider_attempt():
    client = MappedStubEmbeddingClient()
    vector, status = _query_embedding_with_status(client, "q" * 128_001)
    assert vector is None
    assert not status.attempted
    assert status.reason == "input_too_large"
    assert client.calls == []


def test_query_malformed_response_retains_malformed_status():
    client = MappedStubEmbeddingClient(default=[0.0, 0.0, 0.0])
    vector, status = _query_embedding_with_status(client, "question")
    assert vector is None
    assert status.attempted
    assert status.reason == "malformed_vector"
    assert client.calls == [["question"]]


def test_query_provider_failure_retains_provider_status():
    client = MappedStubEmbeddingClient(fail_on="question")
    vector, status = _query_embedding_with_status(client, "question")
    assert vector is None
    assert status.attempted
    assert status.reason == "provider_error"
    assert client.calls == [["question"]]
