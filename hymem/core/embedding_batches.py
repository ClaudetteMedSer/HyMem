"""Lossless bounds and response admission for embedding provider requests."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from hymem.deadline import check_current_deadline

MAX_EMBEDDING_TEXTS = 16
MAX_EMBEDDING_CHARACTERS = 128_000
MAX_EMBEDDING_UTF8_BYTES = 512_000


class EmbeddingInputTooLarge(ValueError):
    """One source cannot fit in a provider request; contains no source text."""


class EmbeddingResponseInvalid(RuntimeError):
    """A provider response failed local admission; contains no response data."""


def plan_embedding_batches(texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Validate the entire input before dispatching any request."""
    sizes: list[tuple[int, int]] = []
    for text in texts:
        if type(text) is not str:
            raise ValueError("embedding input must be exact str")
        try:
            byte_count = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("embedding input contains invalid Unicode") from exc
        char_count = len(text)
        if char_count > MAX_EMBEDDING_CHARACTERS or byte_count > MAX_EMBEDDING_UTF8_BYTES:
            raise EmbeddingInputTooLarge("embedding input exceeds request limit")
        sizes.append((char_count, byte_count))
    batches: list[tuple[str, ...]] = []
    start = 0
    count = chars = bytes_count = 0
    for index, (item_chars, item_bytes) in enumerate(sizes):
        if count and (count == MAX_EMBEDDING_TEXTS
                      or chars + item_chars > MAX_EMBEDDING_CHARACTERS
                      or bytes_count + item_bytes > MAX_EMBEDDING_UTF8_BYTES):
            batches.append(tuple(texts[start:index]))
            start = index
            count = chars = bytes_count = 0
        count += 1
        chars += item_chars
        bytes_count += item_bytes
    if count:
        batches.append(tuple(texts[start:]))
    return tuple(batches)


def embed_bounded(
    embedder: object,
    texts: Sequence[str],
    *,
    identity: Callable[[], tuple[str, int]],
    expected_model: str,
    boundary: Callable[[], None] | None = None,
    required_dim: int | None = None,
    validate_response: Callable[[list[list[float]], int], None] | None = None,
    on_dispatch: Callable[[], None] | None = None,
) -> list[list[float]]:
    """Dispatch preflighted slices and admit each before sending the next.

    The first response may establish the true dimension. Subsequent responses
    must use that exact vector space. The returned list retains input order and
    duplicates. Persistence remains the caller's responsibility.
    """
    batches = plan_embedding_batches(texts)
    result: list[list[float]] = []
    final_dim: int | None = None
    for batch in batches:
        check_current_deadline()
        if boundary is not None:
            boundary()
        if on_dispatch is not None:
            on_dispatch()
        vectors = embedder.embed(list(batch))  # type: ignore[attr-defined]
        check_current_deadline()
        if boundary is not None:
            boundary()
        if not isinstance(vectors, (list, tuple)) or len(vectors) != len(batch):
            raise EmbeddingResponseInvalid("embedding client returned wrong vector count")
        try:
            model, dim = identity()
        except Exception as exc:
            raise EmbeddingResponseInvalid(
                "embedding client identity unavailable after response"
            ) from exc
        if model != expected_model or type(dim) is not int or dim <= 0:
            raise EmbeddingResponseInvalid("embedding client changed identity during batch")
        if required_dim is not None and dim != required_dim:
            raise EmbeddingResponseInvalid("embedding client changed dimension during batch")
        if final_dim is not None and dim != final_dim:
            raise EmbeddingResponseInvalid("embedding client changed dimension during batch")
        admitted: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, (list, tuple)) or len(vector) != dim:
                raise EmbeddingResponseInvalid("embedding client returned malformed vector")
            try:
                numeric = [float(value) for value in vector]
            except (TypeError, ValueError, OverflowError) as exc:
                raise EmbeddingResponseInvalid("embedding client returned malformed vector") from exc
            if not all(math.isfinite(value) for value in numeric):
                raise EmbeddingResponseInvalid("embedding client returned malformed vector")
            norm = math.sqrt(sum(value * value for value in numeric))
            if not math.isfinite(norm) or norm <= 0:
                raise EmbeddingResponseInvalid("embedding client returned malformed vector")
            admitted.append(numeric)
        if validate_response is not None:
            validate_response(admitted, dim)
        result.extend(admitted)
        final_dim = dim
    return result
