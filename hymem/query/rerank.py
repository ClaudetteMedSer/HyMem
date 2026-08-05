"""Cross-source rerank step that runs after RRF in `augment()`.

Two backends ship in-tree:

  * **LLM**: reuses the host-provided :class:`~hymem.extraction.llm.LLMClient`
    with a tiny relevance-rating prompt. Always available when the host
    already wires an LLM for dreaming, costs one extra request per query.
  * **Cross-encoder**: lazy-imported sentence-transformers cross-encoder
    (e.g. ``mixedbread-ai/mxbai-rerank-base-v1``). Local, CPU-friendly,
    no token budget. Optional dependency — if sentence-transformers is not
    installed or the model is unavailable, ``cross_encoder_rerank`` returns
    the candidates unchanged so the pipeline degrades to the LLM path (or
    the un-reranked RRF list, whichever the caller picked).

The reranker contract is intentionally narrow:
``(query, candidates) -> ranked candidates``. Score units are not
normalized across backends — the rank order is what matters; callers should
only sort downstream consumers by ``score`` if they keep the kind tag too.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Protocol

from hymem.extraction.jsonio import loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import RERANK_SYSTEM, RERANK_USER_TEMPLATE

log = logging.getLogger("hymem.query.rerank")


class Reranker(Protocol):
    """Pluggable reranker. Implementations should preserve input order on ties
    and never raise — degrade to the input list on any backend failure."""

    def rerank(self, query: str, candidates: list, *, top_k: int) -> list: ...


def llm_rerank(
    query: str,
    candidates: list,
    llm: LLMClient,
    *,
    top_k: int,
) -> list:
    """Ask the LLM to rate each candidate 1-5 against the query, then sort.

    `candidates` is expected to be a list of objects with ``.text`` and
    ``.score`` / ``.score_kind`` attributes (matches
    :class:`~hymem.query.augment.FtsHit`). RRF scores are kept as the
    tiebreaker so two equally-rated candidates fall back to the upstream
    fused ordering instead of an arbitrary one.

    Returns up to ``top_k`` candidates with ``score`` set to the combined
    rating and ``score_kind`` set to ``"reranked"``. On any backend or parse
    failure the input is truncated to ``top_k`` and returned unchanged — the
    pipeline never fails closed on a flaky LLM. This sits on augment()'s hot
    path, so the Reranker contract (never raise) is load-bearing: an exception
    here would abort the whole augment call, not just the rerank.
    """
    if not candidates:
        return candidates

    excerpts_lines = [f"[{i}] {hit.text[:400]}" for i, hit in enumerate(candidates)]
    excerpts = "\n\n".join(excerpts_lines)

    request = LLMRequest(
        system=RERANK_SYSTEM,
        user=RERANK_USER_TEMPLATE.format(query=query, excerpts=excerpts),
        response_format="json",
    )
    try:
        raw = llm.complete(request)
    except Exception:
        log.exception("llm_rerank backend failed; returning candidates unreranked")
        return candidates[:top_k]

    # Fences/prose around the JSON are tolerated (dream 1013 — json_object mode
    # is a request, not a contract). `expect="any"` rather than "object"
    # because the shape check below deliberately accepts a bare array too:
    # narrowing recovery to objects would drop a fenced array this function is
    # otherwise happy to rerank. Pure string work, no extra request, so the
    # deliberately retry-free hot path stays that way.
    parsed = loads_lenient(raw, expect="any")
    if parsed is None:
        log.warning("rerank.parse_failure candidates=%d raw_len=%d",
                    len(candidates), len(raw) if isinstance(raw, str) else -1)
        return candidates[:top_k]
    # JSON-mode backends (response_format "json_object") are required to emit
    # an object, so the prompt asks for {"ratings": [...]}; models without
    # strict JSON mode may still return the bare array. Accept both.
    ratings = parsed.get("ratings") if isinstance(parsed, dict) else parsed
    if not isinstance(ratings, list):
        # Valid JSON of the wrong shape. Silent, this is indistinguishable
        # downstream from "the model rated nothing relevant" — the candidates
        # come back in upstream order either way — so it gets its own line.
        log.warning("rerank.shape_failure candidates=%d type=%s",
                    len(candidates), type(parsed).__name__)
        return candidates[:top_k]

    relevance: dict[int, int] = {}
    for item in ratings:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        score = item.get("relevance")
        if isinstance(idx, int) and isinstance(score, (int, float)) and 0 <= idx < len(candidates):
            relevance[idx] = int(score)

    scored = []
    for i, hit in enumerate(candidates):
        rrf_score = hit.score if hit.score_kind == "rrf" else 0.0
        llm_score = relevance.get(i, 3)
        combined = llm_score * 100 + rrf_score
        scored.append((combined, i, hit))

    # (-combined, original_index) so ties break on upstream rank.
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        replace(hit, score=float(score), score_kind="reranked")
        for score, _idx, hit in scored[:top_k]
    ]


def cross_encoder_rerank(
    query: str,
    candidates: list,
    *,
    top_k: int,
    model_name: str = "mixedbread-ai/mxbai-rerank-base-v1",
) -> list:
    """Score each candidate with a local cross-encoder, then sort.

    Lazy-imports sentence-transformers; if unavailable, returns
    ``candidates[:top_k]`` unchanged so the caller can fall through to the
    LLM backend or skip reranking. Model load is cached per-process via
    :func:`_get_cross_encoder` so repeated queries pay the load cost once.
    """
    if not candidates:
        return candidates

    model = _get_cross_encoder(model_name)
    if model is None:
        return candidates[:top_k]

    pairs = [(query, hit.text[:400]) for hit in candidates]
    try:
        scores = model.predict(pairs)
    except Exception:
        log.exception("cross_encoder.predict_failed model=%s", model_name)
        return candidates[:top_k]

    scored = [
        (float(score), i, hit)
        for i, (hit, score) in enumerate(zip(candidates, scores))
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        replace(hit, score=float(score), score_kind="reranked")
        for score, _idx, hit in scored[:top_k]
    ]


_CROSS_ENCODER_CACHE: dict[str, object] = {}


def _get_cross_encoder(model_name: str):
    """Cache the cross-encoder model per process. Returns None when
    sentence-transformers (an optional dependency) is not installed or the
    model can't be loaded — callers degrade gracefully."""
    cached = _CROSS_ENCODER_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError:
        log.info(
            "sentence-transformers not installed; cross-encoder rerank unavailable"
        )
        _CROSS_ENCODER_CACHE[model_name] = None  # type: ignore[assignment]
        return None
    try:
        model = CrossEncoder(model_name)
    except Exception as exc:
        log.info(
            "cross-encoder model %s failed to load (%s); falling back",
            model_name, exc,
        )
        _CROSS_ENCODER_CACHE[model_name] = None  # type: ignore[assignment]
        return None
    _CROSS_ENCODER_CACHE[model_name] = model
    return model


def rerank(
    query: str,
    candidates: list,
    *,
    top_k: int,
    model: str = "llm",
    llm: LLMClient | None = None,
    cross_encoder_model: str = "mixedbread-ai/mxbai-rerank-base-v1",
) -> list:
    """Dispatch to the configured rerank backend.

    ``model="llm"`` requires ``llm`` to be wired; without it the candidates
    are returned untouched (truncated to ``top_k``). ``model="cross-encoder"``
    falls through to the LLM backend if the cross-encoder model can't be
    loaded — best-effort, never raises.
    """
    if not candidates:
        return candidates
    if model == "cross-encoder":
        reranked = cross_encoder_rerank(
            query, candidates, top_k=top_k, model_name=cross_encoder_model
        )
        # If the cross-encoder backend was unavailable it returned the
        # raw candidates (no "reranked" score_kind). Try LLM as a fallback
        # when one is wired so the caller still gets a reordered list.
        if reranked and reranked[0].score_kind != "reranked" and llm is not None:
            return llm_rerank(query, candidates, llm, top_k=top_k)
        return reranked
    if model == "llm":
        if llm is None:
            return candidates[:top_k]
        return llm_rerank(query, candidates, llm, top_k=top_k)
    log.warning("unknown rerank model %r; returning candidates unchanged", model)
    return candidates[:top_k]
