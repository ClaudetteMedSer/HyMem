"""Global retrieval fusion and exact-occurrence provenance.

The individual retrieval tiers deliberately keep their native score units and
public DTOs.  This module adds a second, host-facing view which can compare
those tiers without pretending that an FTS5 BM25 number, a cosine and an RRF
score are calibrated probabilities.  Fusion is rank based; raw scores are
retained solely as provenance.

Cross-tier deduplication is equally conservative: it uses exact
``(session_id, message_id)`` source occurrences.  Equal text is never an
identity key, and numeric message ranges are resolved through actual durable
coverage/live rows rather than expanded arithmetically.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

from hymem.dreaming.aggregation_provenance import (
    BoundSourceOccurrence,
    load_aggregation_source_manifest,
)
from hymem.dreaming.lossless import validate_message_coverage_artifact
from hymem.dreaming.facts import load_fact_source_manifests


@dataclass(frozen=True)
class SourceOccurrence:
    """One exact source turn, with external ownership when it exists."""

    session_id: str
    message_id: int
    source_peer_id: str | None = None
    source_workspace_id: str | None = None

    @property
    def identity(self) -> tuple[str, int, str | None, str | None]:
        """Full collapse identity, including ownership metadata."""
        return (
            self.session_id, self.message_id,
            self.source_peer_id, self.source_workspace_id,
        )

    @property
    def coverage_identity(self) -> tuple[str, int]:
        return (self.session_id, self.message_id)


@dataclass(frozen=True)
class RetrievalProvenance:
    """The native artifact/rank/score retained behind one fused unit."""

    tier: str
    artifact_id: str
    rank: int
    raw_score: float | None
    score_kind: str
    why_retrieved: tuple[str, ...] = ()


@dataclass
class FusedEvidence:
    """One globally ranked evidence unit.

    ``payload`` is the original public tier DTO, so consumers do not lose any
    source-specific metadata.  ``provenance`` grows when several summaries or
    hits resolve to the same exact occurrence; this makes dedup observable
    rather than silently deleting the losing representations.
    """

    key: str
    tier: str
    payload: Any = field(repr=False, compare=False)
    normalized_score: float
    protected: bool
    source_occurrences: tuple[SourceOccurrence, ...] = ()
    marginal_occurrences: tuple[SourceOccurrence, ...] = ()
    provenance: tuple[RetrievalProvenance, ...] = ()
    source_tiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackedContext:
    """Observable result of item-level prompt packing."""

    text: str
    items: tuple[FusedEvidence, ...]
    token_budget: int | None
    tokens_used: int
    char_budget: int | None
    chars_used: int
    truncated: bool
    dropped_items: int


# Direct source turns are the most faithful representation of an occurrence;
# dreamed summaries remain useful only for source turns not already represented.
_REPRESENTATION_PRIORITY = {
    "message": 100,
    "count_message": 99,
    "recent": 95,
    "graph": 90,
    "fact": 80,
    "chunk": 70,
    "episode": 60,
    "temporal": 55,
    "procedure": 50,
    "aggregation": 40,
    "digest": 30,
}

_AUTHORITY = {
    "rule": 1.00,
    "profile": 0.98,
    "graph_count": 0.97,
    "graph": 0.94,
    "message": 0.90,
    "count_message": 0.91,
    "recent": 0.88,
    "fact": 0.84,
    "temporal": 0.80,
    "chunk": 0.76,
    "procedure": 0.72,
    "episode": 0.68,
    "aggregation": 0.60,
    "digest": 0.55,
}

TokenCounter = Callable[[str], int]


class _ConfiguredTokenizerFailure(RuntimeError):
    """A trusted counter failed mid-pack; the whole pack must be retried."""


def stable_token_counter(counter: TokenCounter) -> TokenCounter:
    """Validate and memoize one counter for the lifetime of a packing pass.

    Mixing model-token counts for some candidates with byte fallback for later
    candidates can admit a context that violates the hard ceiling. Any failure
    therefore aborts the pass; the caller can retry wholly with fallback.
    """

    cache: dict[str, int] = {}

    def count(text: str) -> int:
        if text in cache:
            return cache[text]
        try:
            value = counter(text)
        except Exception as exc:
            raise _ConfiguredTokenizerFailure(str(exc)) from exc
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or (bool(text) and value == 0)
        ):
            raise _ConfiguredTokenizerFailure(
                "configured token counter returned an invalid count"
            )
        cache[text] = value
        return value

    return count


def estimate_tokens(text: str, token_counter: TokenCounter | None = None) -> int:
    """Dependency-free conservative token estimate.

    A caller-supplied counter is the configured model's tokenizer and therefore
    takes precedence.  It is accepted only when it returns a non-negative
    integer; exceptions and malformed results fail closed to the local path.
    Without an explicitly model-bound counter, UTF-8 byte length is used as a
    conservative upper bound for byte-level BPE token count. An ambient cached
    tokenizer is intentionally never guessed: cl100k is not a safety bound for
    DeepSeek, local, or future configured models.
    """

    if not text:
        return 0
    if token_counter is not None:
        try:
            exact = token_counter(text)
        except _ConfiguredTokenizerFailure:
            raise
        except Exception:
            exact = None
        if isinstance(exact, int) and not isinstance(exact, bool) and exact > 0:
            return exact
    # A byte-level BPE cannot emit more tokens than input bytes. This is a true
    # upper bound (including CJK, emoji and unbroken random ASCII), unlike the
    # usual chars/4 heuristic. It is conservative but makes an external hard
    # budget safe even with no tokenizer package/cache/network.
    return len(text.encode("utf-8"))


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _artifact_candidate(
    *,
    tier: str,
    artifact_id: object,
    payload: Any,
    rank: int,
    score: object = None,
    score_kind: object = "standing",
    why_retrieved: Iterable[object] = (),
    occurrences: Iterable[SourceOccurrence] = (),
    protected: bool = False,
) -> FusedEvidence:
    # Ranks, unlike native score magnitudes, are comparable between retrieval
    # systems.  A small lexical/RRF agreement bonus preserves each tier's strong
    # lexical winner without letting arbitrary BM25 magnitude dominate globally.
    rank_signal = 1.0 / max(1, rank)
    kind = str(score_kind or "unknown")
    chips = tuple(str(chip) for chip in why_retrieved)
    lexical_bonus = 0.08 if kind in {"bm25", "reranked"} else 0.0
    proves_two_channels = any(
        "fts+vec" in chip or "vec+fts" in chip for chip in chips
    )
    agreement_bonus = 0.04 if kind == "rrf" and proves_two_channels else 0.0
    fallback_penalty = 0.0
    if tier == "graph" and any(
        "fallback" in chip.casefold() or "recency" in chip.casefold()
        for chip in chips
    ):
        fallback_penalty = 0.08
    control_bonus = 0.08 if tier == "graph_count" else 0.0
    normalized = min(
        1.0,
        0.58 * _AUTHORITY.get(tier, 0.5)
        + 0.34 * rank_signal
        + lexical_bonus
        + agreement_bonus
        + control_bonus
        - fallback_penalty,
    )
    occ_by_id = {item.identity: item for item in occurrences}
    ordered_occurrences = tuple(
        occ_by_id[key] for key in sorted(occ_by_id, key=repr)
    )
    provenance = RetrievalProvenance(
        tier=tier,
        artifact_id=str(artifact_id),
        rank=rank,
        raw_score=_safe_float(score),
        score_kind=kind,
        why_retrieved=chips,
    )
    return FusedEvidence(
        key=f"{tier}:{artifact_id}",
        tier=tier,
        payload=payload,
        normalized_score=normalized,
        protected=protected,
        source_occurrences=ordered_occurrences,
        marginal_occurrences=ordered_occurrences,
        provenance=(provenance,),
        source_tiers=(tier,),
    )


def _scope_allows(
    item: FusedEvidence,
    *,
    source_session_id: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
) -> bool:
    if item.tier == "fact" and not bool(
        getattr(item.payload, "source_provenance_complete", False)
    ):
        return False
    requested = any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    )
    if not requested:
        return True
    # Standing instructions are configuration, not authored memory evidence.
    if item.tier == "rule":
        return True
    if not item.source_occurrences:
        return False
    if (
        item.tier in {"chunk", "fact", "episode", "aggregation", "procedure"}
        and not bool(getattr(item.payload, "source_provenance_complete", False))
    ):
        return False
    # A composite summary can expose every source in its text, so it is safe
    # only when *all* of its exact sources satisfy the requested boundary.
    return all(
        (source_session_id is None or occ.session_id == source_session_id)
        and (source_peer_id is None or occ.source_peer_id == source_peer_id)
        and (
            source_workspace_id is None
            or occ.source_workspace_id == source_workspace_id
        )
        for occ in item.source_occurrences
    )


def _graph_occurrences(fact: Any) -> tuple[SourceOccurrence, ...]:
    found: dict[tuple[object, ...], SourceOccurrence] = {}
    for citation in getattr(fact, "citations", ()) or ():
        session_id = getattr(citation, "source_session_id", None)
        message_id = getattr(citation, "source_message_id", None)
        if not isinstance(session_id, str) or not session_id:
            continue
        if not isinstance(message_id, int) or isinstance(message_id, bool):
            continue
        occurrence = SourceOccurrence(
            session_id=session_id,
            message_id=message_id,
            source_peer_id=getattr(citation, "source_peer_id", None),
            source_workspace_id=getattr(citation, "source_workspace_id", None),
        )
        found[occurrence.identity] = occurrence
    return tuple(found[key] for key in sorted(found, key=repr))


def _payload_occurrences(payload: Any) -> tuple[SourceOccurrence, ...]:
    raw = getattr(payload, "source_occurrences", ()) or ()
    return tuple(item for item in raw if isinstance(item, SourceOccurrence))


def _content_identity(item: FusedEvidence) -> str:
    """Exact representation content used only with an exact source set.

    Range overlap alone never collapses claims: a single turn may support two
    different facts. We intentionally do not case-fold, strip, or fuzzy-match.
    """
    payload = item.payload
    if item.tier == "message":
        return str(getattr(payload, "text", ""))
    if item.tier == "recent":
        return str(getattr(payload, "content", ""))
    if item.tier in {"chunk", "fact", "temporal"}:
        return str(getattr(payload, "text", ""))
    if item.tier in {"episode", "aggregation"}:
        return str(getattr(payload, "summary", ""))
    if item.tier == "graph":
        return "\x1f".join(str(getattr(payload, key, "")) for key in (
            "subject", "predicate", "object",
        ))
    if item.tier == "procedure":
        return repr((
            getattr(payload, "name", ""),
            getattr(payload, "description", ""),
            getattr(payload, "steps", ()),
        ))
    return repr(payload)


def _all_candidates(ctx: Any) -> list[FusedEvidence]:
    out: list[FusedEvidence] = []

    for rank, rule in enumerate(getattr(ctx, "rules", ()) or (), 1):
        out.append(_artifact_candidate(
            tier="rule", artifact_id=getattr(rule, "id", rank), payload=rule,
            rank=rank, protected=True,
        ))
    for rank, profile in enumerate(getattr(ctx, "user_profile", ()) or (), 1):
        profile_id = (
            getattr(profile, "slot", "profile"),
            getattr(profile, "slot_key", None),
            getattr(profile, "value", ""),
        )
        out.append(_artifact_candidate(
            tier="profile", artifact_id=repr(profile_id), payload=profile,
            rank=rank,
            occurrences=_payload_occurrences(profile),
        ))
    digest = getattr(ctx, "digest", None)
    if digest is not None:
        out.append(_artifact_candidate(
            tier="digest", artifact_id="root", payload=digest, rank=1,
        ))

    graph_count = getattr(ctx, "graph_count", None)
    if graph_count is not None:
        out.append(_artifact_candidate(
            tier="graph_count", artifact_id="exact", payload=graph_count,
            rank=1,
        ))
    total_matches = int(getattr(ctx, "total_message_matches", 0) or 0)
    if total_matches:
        # This is metadata about the complete scoped search, not another source
        # occurrence. Packing retains it only when at least one exact aggregate
        # turn remains beside it; the count itself is intentionally droppable.
        out.append(_artifact_candidate(
            tier="graph_count", artifact_id="candidate", payload=(
                "candidate", total_matches,
                int(getattr(ctx, "enumeration_turns", 0) or 0),
            ), rank=2,
        ))

    tier_specs = (
        ("graph", "graph_facts", "edge_id", "score", "score_kind"),
        ("message", "message_hits", "message_id", "score", "score_kind"),
        ("count_message", "count_message_hits", "message_id", "score", "score_kind"),
        ("fact", "facts", "fact_id", "score", "score_kind"),
        ("temporal", "temporal_events", "date", None, None),
        ("chunk", "fts_hits", "chunk_id", "score", "score_kind"),
        ("procedure", "procedures", "procedure_id", "score", "score_kind"),
        ("episode", "episodes", "episode_id", "score", "score_kind"),
        ("aggregation", "aggregation_nodes", "node_id", "score", "score_kind"),
        ("recent", "recent_turns", "id", None, None),
    )
    for tier, field_name, id_name, score_name, kind_name in tier_specs:
        payloads = list(getattr(ctx, field_name, ()) or ())
        # Native retrieval functions already emit deterministic rank order.
        # Preserve that ordinal exactly: even within one public DTO list score
        # kinds can use incompatible units/directions (coverage BM25 vs cosine),
        # so re-sorting by their numeric values would reverse lexical winners.
        for rank, payload in enumerate(payloads, 1):
            artifact_id = getattr(payload, id_name, rank)
            if tier == "temporal":
                artifact_id = f"{artifact_id}:{rank}"
            occurrences = (
                _graph_occurrences(payload)
                if tier == "graph"
                else _payload_occurrences(payload)
            )
            if tier in {"message", "count_message"} and not occurrences:
                session_id = getattr(payload, "session_id", None)
                message_id = getattr(payload, "message_id", None)
                if isinstance(session_id, str) and isinstance(message_id, int):
                    occurrences = (SourceOccurrence(
                        session_id, message_id,
                        getattr(payload, "source_peer_id", None),
                        getattr(payload, "source_workspace_id", None),
                    ),)
            elif tier == "recent" and not occurrences:
                session_id = getattr(payload, "session_id", None)
                message_id = getattr(payload, "id", None)
                if isinstance(session_id, str) and isinstance(message_id, int):
                    occurrences = (SourceOccurrence(
                        session_id, message_id,
                        getattr(payload, "source_peer_id", None),
                        getattr(payload, "source_workspace_id", None),
                    ),)
            out.append(_artifact_candidate(
                tier=tier,
                artifact_id=artifact_id,
                payload=payload,
                rank=rank,
                score=(getattr(payload, score_name, None) if score_name else None),
                score_kind=(
                    getattr(payload, kind_name, "standing")
                    if kind_name else "standing"
                ),
                why_retrieved=getattr(payload, "why_retrieved", ()) or (),
                occurrences=occurrences,
            ))
    return out


def fuse_context(
    ctx: Any,
    *,
    source_session_id: str | None = None,
    source_peer_id: str | None = None,
    source_workspace_id: str | None = None,
) -> list[FusedEvidence]:
    """Return a deterministic, globally comparable, occurrence-deduped view."""

    candidates = [
        item for item in _all_candidates(ctx)
        if _scope_allows(
            item,
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
        )
    ]
    # A source occurrence has one ownership tuple. Conflicting peer/workspace
    # claims for the same (session,message) key are corrupt authority, not two
    # independent occurrences; quarantine every dependent item rather than
    # choosing whichever representation ranked first.
    owners: dict[tuple[str, int], set[tuple[str | None, str | None]]] = {}
    for item in candidates:
        for occurrence in item.source_occurrences:
            owners.setdefault(occurrence.coverage_identity, set()).add((
                occurrence.source_peer_id, occurrence.source_workspace_id,
            ))
    conflicted = {key for key, values in owners.items() if len(values) > 1}
    candidates = [
        item for item in candidates
        if not any(
            occurrence.coverage_identity in conflicted
            for occurrence in item.source_occurrences
        )
    ]

    protected = [item for item in candidates if item.protected]
    evidence = [item for item in candidates if not item.protected]

    # Resolve representation identity before global ranking. Direct turns are
    # visited first, so another tier carrying the *same exact content* from the
    # same exact source set becomes provenance on that turn. Coarse range
    # overlap alone is only a diversity signal: one message can support several
    # distinct facts, and a multi-source aggregation must retain its marginal
    # synthesis value.
    evidence.sort(key=lambda item: (
        -_REPRESENTATION_PRIORITY.get(item.tier, 0),
        -item.normalized_score,
        item.key,
    ))
    deduped: list[FusedEvidence] = []
    verbatim_owner: dict[tuple[str, int], int] = {}
    exact_representation_owner: dict[
        tuple[tuple[tuple[str, int, str | None, str | None], ...], str], int
    ] = {}

    def merge_provenance(owner_idx: int, item: FusedEvidence) -> None:
        owner = deduped[owner_idx]
        owner.normalized_score = max(owner.normalized_score, item.normalized_score)
        owner.provenance = tuple(sorted(
            {*owner.provenance, *item.provenance},
            key=lambda p: (p.tier, p.artifact_id, p.rank),
        ))
        owner.source_tiers = tuple(sorted({*owner.source_tiers, item.tier}))

    for item in evidence:
        verbatim_key: tuple[str, int] | None = None
        if len(item.source_occurrences) == 1 and (
            item.tier in {"message", "count_message", "recent"}
            or (
                item.tier == "chunk"
                and bool(getattr(
                    item.payload, "source_provenance_complete", False
                ))
            )
        ):
            verbatim_key = item.source_occurrences[0].coverage_identity
        if verbatim_key is not None and verbatim_key in verbatim_owner:
            merge_provenance(verbatim_owner[verbatim_key], item)
            continue
        exact_key = None
        if item.source_occurrences:
            exact_key = (
                tuple(source.identity for source in item.source_occurrences),
                _content_identity(item),
            )
            if exact_key in exact_representation_owner:
                merge_provenance(exact_representation_owner[exact_key], item)
                continue
        idx = len(deduped)
        deduped.append(item)
        if verbatim_key is not None:
            verbatim_owner[verbatim_key] = idx
        if exact_key is not None:
            exact_representation_owner[exact_key] = idx

    # Greedy deterministic diversity: native scores never enter this stage.
    # The small penalties prevent one tier/session from filling the entire head
    # while preserving a rank-1 lexical item's substantial calibrated lead.
    ranked: list[FusedEvidence] = []
    tier_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}
    occurrence_counts: dict[tuple[str, int], int] = {}
    remaining = list(deduped)
    while remaining:
        def adjusted_score(item: FusedEvidence) -> float:
            marginal = tuple(
                occurrence for occurrence in item.source_occurrences
                if occurrence_counts.get(occurrence.coverage_identity, 0) == 0
            )
            sessions = {o.session_id for o in marginal}
            penalty = 0.025 * tier_counts.get(item.tier, 0)
            penalty += 0.015 * sum(session_counts.get(s, 0) for s in sessions)
            if item.source_occurrences:
                covered = len(item.source_occurrences) - len(marginal)
                penalty += 0.16 * (covered / len(item.source_occurrences))
            return item.normalized_score - penalty

        def selection_key(item: FusedEvidence) -> tuple[float, str]:
            return (-adjusted_score(item), item.key)

        chosen = min(remaining, key=selection_key)
        remaining.remove(chosen)
        final_score = adjusted_score(chosen)
        chosen.normalized_score = max(0.0, min(1.0, final_score))
        chosen.marginal_occurrences = tuple(
            occurrence for occurrence in chosen.source_occurrences
            if occurrence_counts.get(occurrence.coverage_identity, 0) == 0
        )
        ranked.append(chosen)
        tier_counts[chosen.tier] = tier_counts.get(chosen.tier, 0) + 1
        for session_id in {o.session_id for o in chosen.marginal_occurrences}:
            session_counts[session_id] = session_counts.get(session_id, 0) + 1
        for occurrence in chosen.source_occurrences:
            key = occurrence.coverage_identity
            occurrence_counts[key] = (
                occurrence_counts.get(key, 0) + 1
            )

    protected.sort(key=lambda item: (
        -_AUTHORITY.get(item.tier, 0.5),
        item.provenance[0].rank if item.provenance else math.inf,
        item.key,
    ))
    return [*protected, *ranked]


def enrich_context_provenance(conn: sqlite3.Connection, ctx: Any) -> None:
    """Attach exact source occurrences to selected tier DTOs, in place.

    This is one bounded read pass over already-selected artifacts. It neither
    changes native retrieval order nor performs embedding/LLM work.
    """

    chunk_ids = [hit.chunk_id for hit in getattr(ctx, "fts_hits", ())]
    chunk_occurrences: dict[str, tuple[SourceOccurrence, ...]] = {}
    complete_chunks: set[str] = set()
    chunk_meta: dict[str, tuple[object, ...]] = {}
    if chunk_ids:
        placeholders = ",".join("?" for _ in chunk_ids)
        try:
            meta_rows = conn.execute(
                "SELECT id,session_id,start_message_id,end_message_id,text,"
                "source_manifest_version,source_manifest_count "
                f"FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        except sqlite3.OperationalError:
            meta_rows = []
        chunk_meta = {
            row["id"]: (
                row["session_id"], row["start_message_id"],
                row["end_message_id"], row["text"],
                row["source_manifest_version"], row["source_manifest_count"],
            ) for row in meta_rows
        }
    if chunk_ids:
        placeholders = ",".join("?" for _ in chunk_ids)
        try:
            manifest = conn.execute(
                "SELECT cms.chunk_id,cms.ordinal,cms.source_session_id,"
                "cms.source_message_id,cms.source_coverage_chunk_id,"
                "cms.source_coverage_version "
                "FROM chunk_message_sources cms "
                f"WHERE cms.chunk_id IN ({placeholders}) ORDER BY cms.chunk_id,cms.ordinal",
                chunk_ids,
            ).fetchall()
        except sqlite3.OperationalError:
            manifest = []
        by_chunk: dict[
            str, list[tuple[int, SourceOccurrence, str]]
        ] = {}
        for row in manifest:
            try:
                proof = validate_message_coverage_artifact(
                    conn,
                    message_id=int(row["source_message_id"]),
                    chunk_id=row["source_coverage_chunk_id"],
                    coverage_version=row["source_coverage_version"],
                )
            except (RuntimeError, TypeError, ValueError, sqlite3.Error):
                continue
            if (
                proof.session_id != row["source_session_id"]
                or proof.message_id != row["source_message_id"]
            ):
                continue
            occurrence = SourceOccurrence(
                proof.session_id, proof.message_id,
                proof.source_peer_id, proof.source_workspace_id,
            )
            by_chunk.setdefault(row["chunk_id"], []).append(
                (
                    int(row["ordinal"]), occurrence,
                    f"{proof.role}: {proof.content}",
                )
            )
        for chunk_id, ordinal_occurrences in by_chunk.items():
            # Every manifest member was independently resolved through its
            # canonical coverage artifact above. Exact source membership comes
            # from this manifest only; coarse numeric ranges prove nothing.
            occurrences = [item[1] for item in ordinal_occurrences]
            (
                chunk_session, start_message_id, end_message_id, chunk_text,
                version, declared_count,
            ) = chunk_meta.get(chunk_id, (None,) * 6)
            complete = bool(
                version == "claim-source-manifest-v1"
                and isinstance(declared_count, int)
                and declared_count == len(ordinal_occurrences)
                and all(
                    item[0] == expected
                    for expected, item in enumerate(ordinal_occurrences)
                )
                and occurrences
                and all(o.session_id == chunk_session for o in occurrences)
                and occurrences[0].message_id == start_message_id
                and occurrences[-1].message_id == end_message_id
                and all(
                    left.message_id < right.message_id
                    for left, right in zip(occurrences, occurrences[1:])
                )
                and "\n".join(item[2] for item in ordinal_occurrences)
                    == chunk_text
                and chunk_text == next(
                    (
                        hit.text for hit in getattr(ctx, "fts_hits", ())
                        if hit.chunk_id == chunk_id
                    ),
                    None,
                )
            )
            if complete:
                chunk_occurrences[chunk_id] = tuple(occurrences)
                complete_chunks.add(chunk_id)
    ctx.fts_hits = [
        replace(
            hit,
            source_occurrences=chunk_occurrences.get(hit.chunk_id, ()),
            source_provenance_complete=hit.chunk_id in complete_chunks,
        )
        for hit in getattr(ctx, "fts_hits", ())
    ]

    ctx.episodes = [
        replace(
            hit,
            source_occurrences=(),
            source_provenance_complete=False,
        )
        for hit in getattr(ctx, "episodes", ())
    ]

    def public_sources(
        sources: tuple[BoundSourceOccurrence, ...] | None,
    ) -> tuple[SourceOccurrence, ...]:
        if sources is None:
            return ()
        return tuple(
            SourceOccurrence(
                item.session_id,
                item.message_id,
                item.source_peer_id,
                item.source_workspace_id,
            )
            for item in sources
        )

    native_facts = list(getattr(ctx, "facts", ()))
    fact_sources = load_fact_source_manifests(
        conn, [fact.fact_id for fact in native_facts], outcome_cache={}
    )
    enriched_facts = []
    for fact in native_facts:
        sources = fact_sources.get(fact.fact_id)
        if sources is None:
            continue
        occurrences = public_sources(sources)
        enriched_facts.append(replace(
            fact,
            source_occurrences=occurrences,
            source_provenance_complete=sources is not None,
        ))
    ctx.facts = enriched_facts

    enriched_nodes = []
    for node in getattr(ctx, "aggregation_nodes", ()):
        sources = load_aggregation_source_manifest(conn, node.node_id)
        occurrences = public_sources(sources)
        enriched_nodes.append(replace(
            node,
            source_occurrences=occurrences,
            source_provenance_complete=sources is not None,
        ))
    ctx.aggregation_nodes = enriched_nodes


def scope_context_in_place(
    ctx: Any,
    *,
    source_session_id: str | None,
    source_peer_id: str | None,
    source_workspace_id: str | None,
) -> None:
    """Fail closed for every native DTO when an ownership scope is requested.

    Composite derived text is retained only when every exact source occurrence
    is inside the boundary. This intentionally prefers omission to laundering a
    multi-author chunk/summary into a sole peer's context.
    """
    if not any(
        value is not None
        for value in (source_session_id, source_peer_id, source_workspace_id)
    ):
        return

    def occurrence_matches(occurrence: SourceOccurrence) -> bool:
        return (
            (source_session_id is None or occurrence.session_id == source_session_id)
            and (
                source_peer_id is None
                or occurrence.source_peer_id == source_peer_id
            )
            and (
                source_workspace_id is None
                or occurrence.source_workspace_id == source_workspace_id
            )
        )

    def safe_composite(payload: Any) -> bool:
        occurrences = _payload_occurrences(payload)
        return bool(
            occurrences
            and bool(getattr(payload, "source_provenance_complete", False))
            and all(occurrence_matches(o) for o in occurrences)
        )

    ctx.fts_hits = [hit for hit in getattr(ctx, "fts_hits", ()) if safe_composite(hit)]
    ctx.facts = [hit for hit in getattr(ctx, "facts", ()) if safe_composite(hit)]
    ctx.episodes = [hit for hit in getattr(ctx, "episodes", ()) if safe_composite(hit)]
    ctx.aggregation_nodes = [
        hit for hit in getattr(ctx, "aggregation_nodes", ()) if safe_composite(hit)
    ]

    safe_messages = []
    for hit in getattr(ctx, "message_hits", ()):
        occurrences = _payload_occurrences(hit)
        if not occurrences:
            occurrences = (SourceOccurrence(
                hit.session_id, hit.message_id,
                getattr(hit, "source_peer_id", None),
                getattr(hit, "source_workspace_id", None),
            ),)
        if all(occurrence_matches(o) for o in occurrences):
            safe_messages.append(hit)
    ctx.message_hits = safe_messages
    ctx.count_message_hits = [
        hit for hit in getattr(ctx, "count_message_hits", ())
        if all(occurrence_matches(o) for o in (
            _payload_occurrences(hit)
            or (SourceOccurrence(
                hit.session_id, hit.message_id,
                getattr(hit, "source_peer_id", None),
                getattr(hit, "source_workspace_id", None),
            ),)
        ))
    ]

    ctx.recent_turns = [
        message for message in getattr(ctx, "recent_turns", ())
        if occurrence_matches(SourceOccurrence(
            message.session_id, message.id,
            getattr(message, "source_peer_id", None),
            getattr(message, "source_workspace_id", None),
        ))
    ]

    safe_graph = []
    for fact in getattr(ctx, "graph_facts", ()):
        citations = []
        for citation in getattr(fact, "citations", ()):
            session_id = getattr(citation, "source_session_id", None)
            message_id = getattr(citation, "source_message_id", None)
            if not isinstance(session_id, str) or not isinstance(message_id, int):
                continue
            occurrence = SourceOccurrence(
                session_id, message_id,
                getattr(citation, "source_peer_id", None),
                getattr(citation, "source_workspace_id", None),
            )
            if occurrence_matches(occurrence):
                citations.append(citation)
        if citations:
            safe_graph.append(replace(fact, citations=citations))
    ctx.graph_facts = safe_graph

    # These tiers do not expose exact occurrence/ownership provenance today.
    # Do not guess from a session id or global file path.
    ctx.procedures = []
    ctx.temporal_events = []
    ctx.user_profile = []
    ctx.digest = None
    ctx.user_md = ""
    ctx.memory_md = ""
    ctx.graph_count = None
