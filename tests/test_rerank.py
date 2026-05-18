"""Rerank module: LLM backend produces a relevance-sorted top-k, cross-encoder
backend degrades to LLM/passthrough when sentence-transformers is unavailable,
and augment() wires `rerank_top_k` through as the candidate pool size."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

from hymem.extraction.llm import StubLLMClient
from hymem.query import rerank as rerank_mod
from hymem.query.rerank import cross_encoder_rerank, llm_rerank, rerank


@dataclass
class _Hit:
    text: str
    score: float = 0.0
    score_kind: str = "rrf"


def test_llm_rerank_orders_by_relevance_rating():
    candidates = [
        _Hit(text="completely irrelevant blue elephant trivia"),
        _Hit(text="how to deploy api to staging via docker"),
        _Hit(text="postgres pool tuning notes"),
    ]
    ratings = json.dumps([
        {"index": 0, "relevance": 1},
        {"index": 1, "relevance": 5},
        {"index": 2, "relevance": 3},
    ])
    llm = StubLLMClient(default=ratings)

    out = llm_rerank("how do I deploy?", candidates, llm, top_k=2)
    assert len(out) == 2
    assert "deploy" in out[0].text
    assert out[0].score_kind == "reranked"


def test_llm_rerank_returns_passthrough_on_garbage():
    candidates = [_Hit(text="a"), _Hit(text="b"), _Hit(text="c")]
    llm = StubLLMClient(default="not json")
    out = llm_rerank("q", candidates, llm, top_k=2)
    # Falls back to original ordering truncated to top_k.
    assert [h.text for h in out] == ["a", "b"]


def test_rerank_dispatch_with_llm_model():
    candidates = [_Hit(text="x"), _Hit(text="y")]
    llm = StubLLMClient(default=json.dumps([{"index": 1, "relevance": 5}]))
    out = rerank("q", candidates, top_k=1, model="llm", llm=llm)
    assert out[0].text == "y"


def test_rerank_dispatch_without_llm_returns_passthrough():
    candidates = [_Hit(text="a"), _Hit(text="b"), _Hit(text="c")]
    out = rerank("q", candidates, top_k=2, model="llm", llm=None)
    assert [h.text for h in out] == ["a", "b"]
    # No backend ran, so no "reranked" tag.
    assert out[0].score_kind == "rrf"


def test_cross_encoder_falls_back_when_unavailable(monkeypatch):
    """If sentence-transformers isn't installed, cross_encoder_rerank must
    return the candidates unchanged (truncated) rather than raising."""
    monkeypatch.setattr(rerank_mod, "_CROSS_ENCODER_CACHE", {})
    monkeypatch.setattr(rerank_mod, "_get_cross_encoder", lambda _name: None)
    candidates = [_Hit(text="a"), _Hit(text="b"), _Hit(text="c")]
    out = cross_encoder_rerank("q", candidates, top_k=2)
    assert [h.text for h in out] == ["a", "b"]
    assert out[0].score_kind == "rrf"


def test_rerank_dispatch_cross_encoder_falls_back_to_llm(monkeypatch):
    """When cross-encoder is unavailable and an LLM is wired, dispatch
    should pick up the LLM path so the user still gets a reordered list."""
    monkeypatch.setattr(rerank_mod, "_CROSS_ENCODER_CACHE", {})
    monkeypatch.setattr(rerank_mod, "_get_cross_encoder", lambda _name: None)

    candidates = [_Hit(text="irrelevant"), _Hit(text="relevant")]
    llm = StubLLMClient(default=json.dumps([
        {"index": 0, "relevance": 1},
        {"index": 1, "relevance": 5},
    ]))
    out = rerank("q", candidates, top_k=1, model="cross-encoder", llm=llm)
    assert out[0].text == "relevant"
    assert out[0].score_kind == "reranked"


def test_augment_uses_rerank_top_k_as_candidate_pool(hy_with_embed):
    """augment() should pull `rerank_top_k` candidates from FTS+vec when
    rerank is enabled, even if `fts_top_k` is smaller. We verify by stubbing
    the LLM to record the rerank request and checking how many excerpts
    were sent."""
    hy = hy_with_embed
    sid = "s_pool"
    hy.open_session(sid)
    # Seed many small chunks all containing the same keyword.
    for i in range(15):
        hy.log_message(sid, "user", f"build pipeline note number {i}: deploy task notes")
    hy.close_session(sid)

    # Stub triples LLM (empty) so dreaming runs cleanly.
    from tests.conftest import make_routed_llm
    hy.set_llm(make_routed_llm([], []))
    hy.dream()

    # Now replace the LLM with one that records rerank calls.
    rerank_llm = StubLLMClient(default="[]")
    hy.set_llm(rerank_llm)

    # Configure: small fts_top_k, larger rerank_top_k.
    from dataclasses import replace as dc_replace
    hy.config = dc_replace(
        hy.config,
        fts_top_k=3,
        rerank_top_k=10,
        rerank_ambiguity_threshold=1.0,  # always rerank when both lists present
    )

    hy.augment("build")

    # Pick the rerank request out of the recorded calls — it's the one whose
    # system prompt mentions the relevance-rating rubric.
    rerank_calls = [
        c for c in rerank_llm.calls if "evaluate the relevance" in c.system
    ]
    assert rerank_calls, "rerank was never invoked"
    # The user prompt contains "[i]" markers for each candidate. Count how
    # many were sent: should be > fts_top_k (=3), bounded by rerank_top_k (=10).
    user = rerank_calls[-1].user
    marker_count = sum(1 for line in user.splitlines() if line.startswith("[") and "]" in line)
    assert marker_count > 3
    assert marker_count <= 10
