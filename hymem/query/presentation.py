"""Pure, dependency-free presentation helpers shared by retrieval consumers."""

from __future__ import annotations

import re
import math


_QUERY_STOPWORDS = frozenset({
    "and", "are", "did", "for", "from", "have", "how", "the", "that",
    "this", "was", "what", "when", "where", "which", "who", "with",
    "you", "your",
})


def query_centered_excerpt(text: str, *, query: str, limit: int) -> str:
    """Return a bounded excerpt around the most discriminative query window.

    Query terms are matched as whole tokens. Repeated generic terms are
    down-weighted, so a unique tail term wins over the first occurrence of a
    common query word near the prefix. The result never exceeds ``limit`` —
    including the ellipses — even for pathological tiny limits.
    """
    text = " ".join((text or "").split())
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 6:
        return text[:limit]

    query_tokens = {
        token.casefold() for token in re.findall(r"[^\W_]+", query or "")
        if len(token) >= 3 and token.casefold() not in _QUERY_STOPWORDS
    }
    text_tokens = [
        (match.group(0).casefold(), match.start(), match.end())
        for match in re.finditer(r"[^\W_]+", text)
    ]
    frequencies: dict[str, int] = {}
    matches: list[tuple[str, int, int]] = []
    for token, start, end in text_tokens:
        if token not in query_tokens:
            continue
        frequencies[token] = frequencies.get(token, 0) + 1
        matches.append((token, start, end))

    body_limit = limit - 6

    def window(raw_start: int) -> tuple[str, int, int]:
        start = min(max(0, raw_start), max(0, len(text) - body_limit))
        end = min(len(text), start + body_limit)
        if start > 0 and text[start - 1] != " " and text[start] != " ":
            boundary = text.find(" ", start, end)
            if boundary >= 0:
                start = boundary + 1
        if end < len(text) and text[end - 1] != " " and text[end] != " ":
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        body = text[start:end]
        excerpt = ("..." if start > 0 else "") + body
        if end < len(text):
            excerpt += "..."
        # body_limit reserves both ellipses, so this is defensive rather than
        # an expected clipping path.
        return excerpt[:limit], start, end

    if matches:
        corpus_size = max(1, len(text_tokens))
        weights = {
            token: (1.0 + math.log1p(corpus_size / frequency))
                   * math.sqrt(len(token))
            for token, frequency in frequencies.items()
        }
        # Linear sliding window over matching token spans. Keeping per-token
        # multiplicity lets us score distinct query concepts without rescanning
        # thousands of repeated matches for every possible anchor.
        left = 0
        counts: dict[str, int] = {}
        score = 0.0
        distinct_count = 0
        best: tuple[float, int, int, int] | None = None
        for right, (token, token_start, token_end) in enumerate(matches):
            if token_end - token_start > body_limit:
                candidate = (
                    -weights[token], -1, token_start,
                    min(len(text), token_start + body_limit),
                )
                if best is None or candidate < best:
                    best = candidate
                left = right + 1
                counts.clear()
                score = 0.0
                distinct_count = 0
                continue
            if counts.get(token, 0) == 0:
                score += weights[token]
                distinct_count += 1
            counts[token] = counts.get(token, 0) + 1
            while (
                left <= right
                and token_end - matches[left][1] > body_limit
            ):
                old_token = matches[left][0]
                counts[old_token] -= 1
                if counts[old_token] == 0:
                    score -= weights[old_token]
                    distinct_count -= 1
                left += 1
            if left > right:
                continue
            span_start = matches[left][1]
            span_end = token_end
            candidate = (-score, -distinct_count, span_start, span_end)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        span_start, span_end = best[2], best[3]
        slack = max(0, body_limit - (span_end - span_start))
        excerpt, _start, _end = window(max(0, span_start - slack // 3))
    else:
        excerpt, _start, _end = window(0)
    return excerpt
