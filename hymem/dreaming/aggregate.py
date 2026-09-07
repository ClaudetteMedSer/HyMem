"""Phase-2 RAPTOR cross-session aggregation nodes.

Dreaming already produces per-session *episodes*. The multi-session residual on
LongMemEval is a *synthesis* problem: the gold turns reach the answer context,
but the reader fails to fuse facts scattered one-per-session across ~45 raw
slots. This module closes that upstream — it clusters episodes ACROSS sessions
(connected components over embedding-OR-entity overlap) and fuses each
cross-session cluster into a single `aggregation_nodes` summary, so a synthesis
question can be answered from a handful of cluster summaries instead of dozens
of raw turns.

The whole layer is additive and enabled by default
(`cfg.aggregation_nodes_enabled`): set the master switch to False to skip
`build_aggregation_nodes` and leave query-time behavior unchanged. The build was
front-run gated by an offline co-location probe
(`benchmarks/raptor_cluster_probe.py`); the pure clustering core below is the
canonical home the probe re-exports, so probe, unit tests, and production all
run the *same* clusterer.

Cost discipline: only clusters spanning ≥ `aggregation_min_sessions` distinct
sessions with ≥ `aggregation_min_members` episodes are summarized — singletons
and single-session clusters add nothing over the per-session episode, so they
cost no LLM call. Nodes are rebuilt from scratch each dream (membership is a
pure function of the current episodes), so there is no stale-id UPSERT churn;
summary embeddings are cache-keyed by text and fusions by member-set hash, so
an unchanged node re-uses both its vector and its LLM summary.

On top of the flat level-0 layer sits the RAPTOR rollup (schema v17,
`cfg.aggregation_digest_enabled`): the level-0 nodes plus the episodes no
cluster absorbed are recursively clustered-and-fused into level-N nodes until
one ROOT digest remains — the standing "what do you know about me?" summary
`HyMem.digest()` returns for host system-prompt injection. The G4 LME A/Bs
showed retrieval-side injection of these summaries is at best a wash (raw
message FTS already wins wherever the query has keywords), so the tree's value
is host-facing standing context: levels ≥ 1 never enter the query-time tier.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import copy
from dataclasses import dataclass
from typing import NamedTuple
from collections.abc import Callable, Mapping

from hymem.config import HyMemConfig
from hymem.deadline import check_current_deadline
from hymem.core import db as core_db
from hymem.core.vectors import decode_vector, encode_vector
from hymem.dreaming.aggregation_provenance import (
    AGGREGATION_CLUSTER_SALT,
    AGGREGATION_INPUT_MANIFEST_VERSION,
    AGGREGATION_MAX_SUMMARY_CHARS,
    AGGREGATION_MAX_TITLE_CHARS,
    AGGREGATION_ROLLUP_SALT,
    AGGREGATION_ROOT_SALT,
    AGGREGATION_SOURCE_MANIFEST_VERSION,
    AggregationInputProof,
    aggregation_canonical_json,
    aggregation_input_manifest_hash,
    aggregation_fusion_max_tokens,
    aggregation_llm_request,
    aggregation_llm_request_hash,
    aggregation_node_authority_hash,
    aggregation_node_embedding_set_hash,
    aggregation_node_id,
    aggregation_output_hash,
    aggregation_output_is_canonical,
    aggregation_publication_id,
    aggregation_publication_node_set_hash,
    aggregation_publication_timestamp_is_canonical,
    aggregation_typed_input_fingerprint,
    combine_source_occurrences,
    episode_authority_hash,
    load_aggregation_node_proof,
    load_current_aggregation_publication,
    load_current_aggregation_node_proof,
    load_published_aggregation_root,
    load_root_anchor_inputs,
    make_aggregation_input_proof,
    load_episode_source_manifest,
    persist_aggregation_source_manifest,
    source_manifest_hash,
)
from hymem.dreaming.embeddings import (
    _embedding_identity,
    _fetch_cached_vectors,
    _finite_embedding_vector,
    _post_embed_identity,
)
from hymem.dreaming.aggregation_material import (
    aggregation_anchor_phase1_generation_keys,
    aggregation_material_binding,
    aggregation_phase1_scope_identity,
    current_aggregation_material_revision,
    disabled_aggregation_phase1_scope_identity,
    embedding_execution_identity,
    register_aggregation_material_epoch,
    validate_aggregation_material_binding,
    verify_aggregation_material_epoch,
)
from hymem.extraction.embeddings import (
    EmbeddingClient,
    embedding_text_hash,
    normalize_text,
)
from hymem.extraction.jsonio import is_ceiling_cut, loads_lenient
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import (
    AGGREGATE_SYSTEM,
    AGGREGATE_USER_TEMPLATE,
    DIGEST_SYSTEM,
    DIGEST_USER_TEMPLATE,
    ROLLUP_SYSTEM,
    ROLLUP_USER_TEMPLATE,
)

log = logging.getLogger("hymem.dreaming.aggregate")
_CANDIDATE_PAIRS_UNSET = object()


class AggregationResult(NamedTuple):
    """Outcome of a node (re)build: total nodes written, and how many fusions
    were served from cache (a content-hash id that already existed) instead of
    being recomputed. ``reused`` is the dream-cost signal the RAPTOR flip
    criteria watches — near-full reuse on an unchanged store means steady
    state. See benchmarks/raptor_digest_plan.md Stage 3c.

    ``fusion_failures``/``input_episodes``/``blocking`` are the attribution
    fields the 2026-07-12 reuse instability hunt was missing: a low-reuse run
    with failures > 0 is an LLM-flakiness event (retries next dream), a shifted
    input_episodes explains a built-count drift, and a blocking-mode change
    between runs means the two dreams clustered with different candidate
    generators (e.g. one process has sqlite-vec, the other doesn't)."""
    nodes: int
    reused: int
    fusion_failures: int = 0
    input_episodes: int = 0
    blocking: str = "exact"
    level0_missed: int = 0
    leaf_changed: int | None = None
    predicted_rebuild: int = 0
    keying_residual: int = 0
    facts_rekey: int = 0
    rebuilt_level0: int = 0
    rebuilt_rollup: int = 0
    rebuilt_root: int = 0
    leaf_added: int | None = None
    leaf_removed: int | None = None
    material_epoch_key: str | None = None


@dataclass(frozen=True)
class CapturedAggregationMaterial:
    """One coherent, exact aggregation input and candidate-selection bundle."""

    episodes: tuple[dict, ...]
    anchor_inputs: tuple[AggregationInputProof, ...]
    candidate_pairs: frozenset[tuple[str, str]] | None
    blocking: Mapping[str, object]
    binding: Mapping[str, object]

# Fusion-prompt versions, baked into the node-id salt of the level the prompt
# serves. Reuse is keyed by node id, so bumping a version when its prompt
# changes materially makes every cached fusion of that kind regenerate on the
# next dream — without touching the other levels' caches. This matters beyond
# style: a hallucination CRYSTALLIZES in a cached fusion (the "Acme Corp"
# incident lived in a persisted rollup and survived a root-only fix), so a
# prompt hardened against an artifact must invalidate the level that produced
# it, or the artifact outlives the fix.
_CLUSTER_SALT = AGGREGATION_CLUSTER_SALT
                              #     windows re-keyed the whole component on any
                              #     mid-order membership change — 2026-07-12
                              #     reuse instability); v3: recency-window split
                              #     at max_cluster_size; v2: identity evidence-bound
_ROLLUP_SALT = AGGREGATION_ROLLUP_SALT
                              #     2026-07-12 fix); v2: identity evidence-bound
_ROOT_SALT = AGGREGATION_ROOT_SALT

# Only settings that can change the materialized aggregation tree belong in
# this identity. Query-time delivery controls (top-k, ability routing, sparse
# fallback) intentionally do not: changing them does not require a rebuild.
# The contract literal must be bumped for a material algorithm change that is
# not already represented by a salt, prompt, source-manifest version, or field.
_AGGREGATION_MATERIAL_CONFIG_FIELDS = (
    # A False -> True transition scrubs persisted profile anchors before the
    # next build, so privacy policy is material even though it is not an
    # `aggregation_*` field.
    "redact_secrets",
    "aggregation_emb_threshold",
    "aggregation_ent_threshold",
    "aggregation_max_cluster_size",
    "aggregation_blocking_top_k",
    "aggregation_min_sessions",
    "aggregation_min_members",
    "aggregation_max_members",
    "aggregation_digest_enabled",
    "aggregation_digest_max_leaves",
    "aggregation_digest_anchor_facts",
)


def aggregation_config_version(cfg: HyMemConfig) -> str:
    """Return a non-secret identity for one material aggregation policy.

    The full payload is hashed rather than persisted: operator status can bind
    pending work to the exact effective build policy without retaining prompt
    text, endpoints, model credentials, or conversation content.
    """

    payload = {
        "contract": "aggregation-material-v2",
        "enabled": cfg.aggregation_nodes_enabled,
        "salts": {
            "cluster": _CLUSTER_SALT,
            "rollup": _ROLLUP_SALT,
            "root": _ROOT_SALT,
        },
        "source_manifest": AGGREGATION_SOURCE_MANIFEST_VERSION,
        "input_manifest": AGGREGATION_INPUT_MANIFEST_VERSION,
        "prompts": {
            "cluster_system": AGGREGATE_SYSTEM,
            "cluster_user": AGGREGATE_USER_TEMPLATE,
            "rollup_system": ROLLUP_SYSTEM,
            "rollup_user": ROLLUP_USER_TEMPLATE,
            "digest_system": DIGEST_SYSTEM,
            "digest_user": DIGEST_USER_TEMPLATE,
        },
        "settings": {
            field: getattr(cfg, field)
            for field in _AGGREGATION_MATERIAL_CONFIG_FIELDS
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "aggregation-build-config-v1:" + hashlib.sha256(encoded).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Pure clustering core (canonical home; benchmarks/raptor_cluster_probe.py and
# tests/test_raptor_cluster_probe.py re-export these). No DB / LLM / embedding
# dependency — operates on plain dicts {"id", "vector": list|None, "entities": set}.
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0.0 on dim mismatch or zero vectors (matches the
    behavioral_dedup._cosine contract used elsewhere in the dreaming layer)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (na * nb)


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard overlap of two entity sets; 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _linked(e1: dict, e2: dict, emb_threshold: float, ent_threshold: float) -> bool:
    """Two episodes are in the same cluster iff their embeddings are close OR they
    share enough key entities. OR (not AND): either signal is sufficient evidence
    the episodes are about the same thread — embeddings catch paraphrase, entity
    overlap catches the named-thing continuity embeddings sometimes miss."""
    if (e1.get("vector") and e2.get("vector")
            and _cosine(e1["vector"], e2["vector"]) >= emb_threshold):
        return True
    if _jaccard(e1.get("entities") or set(), e2.get("entities") or set()) >= ent_threshold:
        return True
    return False


def cluster_episodes(
    episodes: list[dict], emb_threshold: float, ent_threshold: float,
    *, max_cluster_size: int | None = None,
    candidate_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, int]:
    """Connected-components clustering over the episode link graph (union-find).

    `episodes`: list of {"id": str, "vector": list[float]|None, "entities": set[str]}.
    Returns {episode_id -> cluster_label}. Two episodes share a label iff there is a
    path of `_linked` edges between them (transitive closure). This is deliberately
    the simplest cross-session aggregation a RAPTOR layer could do; if even this
    co-locates the gold, a smarter clusterer only does better.

    `candidate_pairs` is the Stage-3b candidate-blocking hook
    (raptor_digest_plan.md; prod timing 2026-06-12: 395 episodes → 77,815
    all-pairs `_linked` tests → 4.04s per dream, past the 2s gate). When None
    (the default, and what benchmarks/cluster_size_probe.py measures), every
    pair is tested — byte-identical to the historical all-pairs behavior. When
    provided, ONLY those pairs are `_linked`-tested; each pair is a tuple of
    two episode ids normalized ascending (`(a, b) with a < b` — i.e.
    `tuple(sorted(...))`), and pairs naming ids outside `episodes` are ignored.
    The caller (`generate_candidate_pairs`) builds the set from an entity
    inverted index (EXACT for the Jaccard arm: ent_threshold >= any positive
    value requires >= 1 shared entity, so no entity link is ever lost) plus
    embedding KNN top-k over `vec_episodes` (APPROXIMATE for the cosine arm: a
    missed link means both endpoints already had >= k closer neighbors).
    Components are order-independent, so labels stay deterministic regardless
    of set iteration order. Deliberately NO node-id salt bump for blocking:
    salts version PROMPT-level staleness, while node cache ids already key on
    member-set hashes — a membership changed by blocking regenerates its
    fusion naturally, an unchanged one keeps its still-valid cache.

    `max_cluster_size` is the Stage-3a chaining guard (raptor_digest_plan.md):
    OR-links chain transitively, and on the prod store (probe, 2026-06-12) that
    snowballed into ONE component of 348 episodes spanning 61 sessions — a
    fusion of that is mush. When set, any component larger than the cap is
    split deterministically into consecutive recency-ordered windows of at most
    `max_cluster_size` members; when None (the default, and what the probes
    pass) behavior is identical to the uncapped clusterer, so
    benchmarks/cluster_size_probe.py keeps measuring RAW chaining.

    Recency signal: members are ordered by `start_message_id` ascending when
    present (messages.id is a store-wide AUTOINCREMENT, so it is a true
    cross-session ingestion-order clock — the loader already carries it),
    falling back to input position (rollup items and bare test dicts carry no
    message ids; input position is the loader's stable order). Window
    boundaries over that order are CONTENT-DEFINED (`_content_defined_groups`):
    a window closes after any member whose own id hashes to a cut, or at
    `max_cluster_size` members, whichever comes first. Boundaries are therefore
    properties of the member ids themselves, not of positions: appending at the
    newest end only grows/cuts the tail window (same append-stability the
    oldest-anchored v3 split had), and — the part v3 lacked — a MID-order
    membership change (an episode joining or leaving between two dreams, a
    bridge episode merging two components and interleaving their orders, a
    superseded/pruned episode dropping out) re-cuts only the window(s) around
    the change instead of shifting every downstream boundary. Positional
    windows turned any such change into a full component re-key — the
    2026-07-12 reuse instability (runs 725-736), the same failure class as the
    newest-end alignment fixed on 2026-07-05 (runs 685-693). The expected
    window size under content cuts is a little under `max_cluster_size` (a cut
    fires with probability 1/max_cluster_size per member, plus the forced cut
    at the cap), so the tree carries somewhat more, smaller windows — finer
    fusions, one-time refusion on deploy (salt v4). Undersized windows are
    dropped by the min-members/min-sessions policy exactly like v3's tail
    window; their episodes stay directly retrievable and still reach the
    digest as leftover pass-through leaves.
    """
    if max_cluster_size is not None and max_cluster_size < 1:
        raise ValueError(f"max_cluster_size must be >= 1, got {max_cluster_size}")
    def item_key(item: dict) -> str:
        # Rollup frontiers can contain an episode and an aggregation child
        # whose identifiers are byte-identical. Their typed identity is the
        # clustering identity; ordinary episode callers carry no override and
        # retain the historical id-keyed API.
        value = item.get("_aggregation_cluster_key", item["id"])
        if not isinstance(value, str) or not value:
            raise ValueError("cluster item identity must be non-empty text")
        return value

    keys = [item_key(item) for item in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("cluster item identities must be unique")
    parent: dict[str, str] = {key: key for key in keys}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if candidate_pairs is None:
        for i in range(len(episodes)):
            check_current_deadline()
            for j in range(i + 1, len(episodes)):
                if _linked(episodes[i], episodes[j], emb_threshold, ent_threshold):
                    union(item_key(episodes[i]), item_key(episodes[j]))
    else:
        by_id = {item_key(e): e for e in episodes}
        for a, b in candidate_pairs:
            check_current_deadline()
            if a == b or a not in by_id or b not in by_id:
                continue
            if _linked(by_id[a], by_id[b], emb_threshold, ent_threshold):
                union(a, b)

    roots = {item_key(e): find(item_key(e)) for e in episodes}
    label_of: dict[str, int] = {}
    out: dict[str, int] = {}
    for eid, root in roots.items():
        if root not in label_of:
            label_of[root] = len(label_of)
        out[eid] = label_of[root]
    if max_cluster_size is None:
        return out

    # Chaining guard: split every over-cap component into recency windows.
    pos = {item_key(e): i for i, e in enumerate(episodes)}

    def _recency_key(e: dict) -> tuple:
        sm = e.get("start_message_id")
        # Episodes carrying a message id sort by it (global ingestion order);
        # items without one (rollup nodes, plain dicts) keep input order and
        # sort after dated ones. Input position breaks all ties → deterministic.
        key = item_key(e)
        return (0, sm, pos[key]) if isinstance(sm, int) else (1, 0, pos[key])

    components: dict[int, list[dict]] = {}
    for e in episodes:
        components.setdefault(out[item_key(e)], []).append(e)

    capped: dict[str, int] = {}
    next_label = 0
    for label in sorted(components):       # original first-seen label order
        members = components[label]
        if len(members) <= max_cluster_size:
            for m in members:
                capped[item_key(m)] = next_label
            next_label += 1
            continue
        ordered = sorted(members, key=_recency_key)   # oldest → newest
        # Content-defined cuts: boundaries belong to member ids, so a
        # membership change anywhere re-cuts only its local window(s).
        for window in _content_defined_groups(ordered, max_cluster_size):
            for m in window:
                capped[item_key(m)] = next_label
            next_label += 1
    return capped


def _is_cut_id(item_id: str, avg_size: int) -> bool:
    """True when this id closes a content-defined group. Pure function of the
    id, so boundaries survive any reordering/insertion/removal around it."""
    digest = hashlib.sha1(f"cut::{item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % avg_size == 0


def _content_defined_groups(ordered: list[dict], max_size: int) -> list[list[dict]]:
    """Split `ordered` into consecutive groups whose boundaries are decided by
    each item's own id hash (content-defined chunking, the rsync trick): a
    group closes after a cut id, or at `max_size` members (the fusion-input
    cap) — so the expected group size is a little under `max_size` and no
    group ever exceeds it. Because a boundary is a property of the id at which
    it falls, inserting or removing items re-cuts only the group(s) touching
    the change; positional slicing (`seq[i:i+size]`) shifted every downstream
    boundary instead, re-keying whole chains of cached fusions."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for item in ordered:
        current.append(item)
        cut_identity = item.get("_aggregation_cluster_key", item["id"])
        if len(current) >= max_size or _is_cut_id(cut_identity, max_size):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# DB-side build (Hermes box: needs real episodes + embeddings; StubLLM in tests).
# ─────────────────────────────────────────────────────────────────────────────

def _norm_entity(x: str) -> str:
    return normalize_text(x).strip()


def load_clusterable_episodes(
    conn: sqlite3.Connection, *, max_rowid: int | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
) -> list[dict]:
    """All episodes with their summary vector + normalized entity set, ordered so
    a stable member list / id falls out of clustering. Mirrors the probe loader.

    `max_rowid` caps the set to episodes that existed at a snapshot the dream
    runner takes just before aggregation (the phase-3 boundary, after this
    dream's own episode-embedding pass). The MCP server writes episodes
    asynchronously; without the ceiling a stray landing mid-build joins the
    clustering, shifts a member set, and forces a spurious near-full refusion
    (dream runs 678/680, 2026-06-28). episodes.rowid is monotonic at insert and
    nothing is deleted after the snapshot within a dream, so `rowid <= max_rowid`
    is exactly 'present at the snapshot'; strays land above it and defer to the
    next dream, which clusters them deterministically."""
    owned_snapshot = not conn.in_transaction
    try:
        if owned_snapshot:
            conn.execute("BEGIN")
        ceiling = "AND e.rowid <= ?" if max_rowid is not None else ""
        rows = conn.execute(
            f"""
            SELECT e.rowid AS rowid, e.id, e.session_id, e.title, e.summary,
                   e.start_message_id, e.end_message_id, e.key_entities,
                   e.digest_slice_key,e.digest_generation,
                   e.source_manifest_version,e.source_manifest_count,
                   e.source_manifest_hash,e.source_manifest_complete,
                   em.vector_json,em.model AS embedding_model,
                   em.dim AS embedding_dim,em.text_hash AS embedding_text_hash,
                   em.embedding_producer_key
            FROM episodes e
            JOIN sessions s ON s.id = e.session_id
            LEFT JOIN episode_embeddings em ON em.episode_id = e.id
            WHERE (e.digest_generation IS NULL
                   OR e.digest_generation = s.digest_published_generation)
            {ceiling}
            ORDER BY e.session_id, e.start_message_id, e.id
            """,
            () if max_rowid is None else (max_rowid,),
        ).fetchall()
        vector_rowids = core_db.episode_vector_rowids(
            str(row["id"]) for row in rows
        )
        episodes: list[dict] = []
        for r in rows:
            try:
                raw_entities = json.loads(r["key_entities"] or "[]")
            except (ValueError, TypeError):
                raw_entities = []
            sources = load_episode_source_manifest(
                conn, r["id"], _episode_row=r,
            )
            # An episode without a complete exact lossless manifest is
            # historical input, never material the aggregation model may see.
            if sources is None:
                continue
            # The row and exact occurrence manifest already belong to this
            # coherent snapshot.  Construct both renderings from that one
            # proof instead of reloading the episode and revalidating every
            # coverage occurrence twice more.
            authority_hash = episode_authority_hash(dict(r))
            plain_text = f"{r['title']}\n{r['summary']}"
            plain_proof = make_aggregation_input_proof(
                kind="episode", source_ref={"id": r["id"]},
                rendered_text=plain_text, authority_hash=authority_hash,
                occurrences=sources,
            )
            cluster_proof = make_aggregation_input_proof(
                kind="episode", source_ref={"id": r["id"]},
                rendered_text=f"[{r['session_id']}] {plain_text}",
                authority_hash=authority_hash, occurrences=sources,
            )
            expected_text_hash = embedding_text_hash(
                f"{r['title']}\n{r['summary']}"
            )
            vec = None
            embedding_record: dict[str, object] = {
                "effective": False,
                "model": None,
                "dimension": None,
                "text_hash": None,
                "vector_sha256": None,
            }
            if (
                embedding_model is not None
                and embedding_dim is not None
                and r["embedding_model"] == embedding_model
                and r["embedding_producer_key"] == embedding_model
                and r["embedding_dim"] == embedding_dim
                and r["embedding_text_hash"] == expected_text_hash
                and r["vector_json"] is not None
            ):
                try:
                    decoded = decode_vector(r["vector_json"])
                except (AttributeError, UnicodeError, TypeError, ValueError):
                    decoded = None
                vec = _finite_embedding_vector(
                    decoded, expected_dim=embedding_dim,
                )
                if vec is not None:
                    embedding_record = {
                        "effective": True,
                        "model": embedding_model,
                        "dimension": embedding_dim,
                        "text_hash": expected_text_hash,
                        "vector_sha256": "sha256:" + hashlib.sha256(
                            encode_vector(vec).encode("utf-8")
                        ).hexdigest(),
                    }
            episodes.append({
                "id": r["id"],
                # sqlite-vec needs an integer rowid, but episodes uses a TEXT
                # primary key whose implicit rowid can be renumbered by
                # VACUUM. Use the canonical id-derived vector key instead.
                "rowid": vector_rowids[str(r["id"])],
                "session_id": r["session_id"],
                "title": r["title"],
                "summary": r["summary"],
                # Recency signal for the max_cluster_size window split:
                # messages.id is a store-wide AUTOINCREMENT.
                # Order derives from exact authoritative occurrences. Mutable
                # start/end range columns remain compatibility metadata only.
                "start_message_id": min(item.message_id for item in sources),
                "entities": {_norm_entity(x) for x in raw_entities if x},
                "vector": vec,
                "source_occurrences": sources,
                "source_provenance_complete": True,
                "source_manifest_hash": r["source_manifest_hash"],
                "cluster_input_proof": cluster_proof,
                "plain_input_proof": plain_proof,
                "embedding_record": embedding_record,
            })
        episodes.sort(key=lambda item: (
            item["session_id"], item["start_message_id"], item["id"],
        ))
        if owned_snapshot:
            conn.execute("COMMIT")
        return episodes
    except BaseException:
        if owned_snapshot and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def generate_candidate_pairs(
    conn: sqlite3.Connection, episodes: list[dict], *, emb_top_k: int,
) -> set[tuple[str, str]] | None:
    """Stage-3b candidate blocking: the pair set `cluster_episodes` should test
    instead of all O(n²) pairs (prod, 2026-06-12: 395 episodes → 77,815 pairs →
    4.04s per dream, past the 2s gate).

    Returns None whenever the vector arm cannot run exactly as designed while
    at least one valid vector exists, and the caller MUST then fall back to
    exact all-pairs (pass candidate_pairs=None):
      - `emb_top_k <= 0` (config: aggregation_blocking_top_k=0 disables blocking);
      - no `vec_episodes` table / sqlite_vec extension unavailable — sqlite_vec
        is an optional dependency, and embedded small stores without it must
        keep today's exact behavior unchanged.

    When no episode has a valid vector, the cosine arm is provably empty and
    the exact entity-only candidate set is returned without requiring sqlite-vec.

    Otherwise returns ascending-normalized id pairs from two arms:
      - Entity arm (EXACT): inverted index entity → episode ids over the
        already-normalized `e["entities"]` sets; every co-occurring pair under
        any entity is a candidate. Jaccard >= 0.5 requires >= 1 shared entity,
        so this arm loses nothing.
      - Cosine arm (approximate): per-episode KNN top `emb_top_k` neighbors via
        `core_db.vec_search` over `vec_episodes` (whose integer rowids are a
        deterministic projection of episode ids). Queried as k+1 so self never consumes
        a neighbor slot — hence with emb_top_k >= n-1 the arm is exact and
        small stores lose nothing. Hits whose rowid is not in `episodes`
        (retention may have pruned the row since vec ingest) and self-pairs are
        skipped. Episodes without a vector contribute no cosine candidates —
        exactly `_linked`'s behavior, whose cosine arm never fires for them;
        the entity arm still covers them.
    """
    return _aggregation_candidate_plan(
        conn, episodes, emb_top_k=emb_top_k,
    )[0]


def _aggregation_candidate_plan(
    conn: sqlite3.Connection, episodes: list[dict], *, emb_top_k: int,
) -> tuple[set[tuple[str, str]] | None, dict[str, object]]:
    """Return candidate pairs plus the actual, auditable execution mode."""

    ordered_ids = [str(item["id"]) for item in episodes]
    universe_hash = "sha256:" + hashlib.sha256(
        aggregation_canonical_json(ordered_ids).encode("utf-8")
    ).hexdigest()

    def exact(reason: str) -> tuple[None, dict[str, object]]:
        return None, {
            "contract": "hymem-aggregation-blocking-v1",
            "mode": "exact",
            "reason": reason,
            "top_k": emb_top_k,
            "candidate_count": None,
            "candidate_pairs_sha256": None,
            "episode_order_sha256": universe_hash,
        }

    if emb_top_k <= 0:
        return exact("disabled")
    pairs: set[tuple[str, str]] = set()
    inverted: dict[str, list[str]] = {}
    for episode in episodes:
        for entity in episode.get("entities") or ():
            inverted.setdefault(entity, []).append(str(episode["id"]))
    for ids in inverted.values():
        for index, left in enumerate(ids):
            for right in ids[index + 1:]:
                if left != right:
                    pairs.add((left, right) if left < right else (right, left))
    vectored = [item for item in episodes if item.get("vector") is not None]
    if not vectored:
        # With no valid vectors the cosine arm is provably empty.  Entity
        # co-occurrence is therefore the complete exact candidate relation;
        # returning it avoids an unnecessary O(n^2) all-pairs walk.
        ordered_pairs = [list(pair) for pair in sorted(pairs)]
        pairs_hash = "sha256:" + hashlib.sha256(
            aggregation_canonical_json(ordered_pairs).encode("utf-8")
        ).hexdigest()
        return pairs, {
            "contract": "hymem-aggregation-blocking-v1",
            "mode": "entity_only",
            "reason": "no_valid_episode_vectors",
            "top_k": emb_top_k,
            "candidate_count": len(pairs),
            "candidate_pairs_sha256": pairs_hash,
            "episode_order_sha256": universe_hash,
        }
    if not core_db._load_vec_extension(conn):
        return exact("vec_extension_unavailable")
    if not core_db.has_vec_table(conn, table="vec_episodes"):
        return exact("vec_table_unavailable")
    dimensions = {len(item["vector"]) for item in vectored}
    models = {
        item["embedding_record"]["model"] for item in vectored
    }
    if len(dimensions) != 1 or len(models) != 1 or None in models:
        return exact("mixed_vector_space")
    dimension = next(iter(dimensions))
    model = next(iter(models))
    dim_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='vec_dim'"
    ).fetchone()
    model_row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='vec_model'"
    ).fetchone()
    if (
        dim_row is None or model_row is None
        or str(dim_row["value"]) != str(dimension)
        or model_row["value"] != model
    ):
        return exact("vec_metadata_mismatch")

    expected_shadow = {
        int(item["rowid"]): core_db._pack_vector(item["vector"])
        for item in vectored
    }
    try:
        shadow_rows = conn.execute(
            "SELECT rowid,embedding FROM vec_episodes ORDER BY rowid"
        ).fetchall()
        actual_shadow = {
            int(row["rowid"]): bytes(row["embedding"])
            for row in shadow_rows
        }
    except (sqlite3.Error, TypeError, ValueError):
        return exact("vec_shadow_unverifiable")
    # Exact equality prevents ineligible/late rows from consuming global KNN
    # slots and proves every used rowid maps to the JSON vector we committed.
    if actual_shadow != expected_shadow:
        return exact("vec_shadow_mismatch")

    id_of_rowid = {int(item["rowid"]): str(item["id"]) for item in vectored}
    try:
        for episode in vectored:
            check_current_deadline()
            # Full-shadow equality above proves there are no out-of-snapshot
            # rows to steal a global slot, so k+1 retains the intended cost.
            hits = core_db.vec_search_strict(
                conn, episode["vector"], min(
                    len(vectored), emb_top_k + 1,
                ), table="vec_episodes",
            )
            neighbors = [
                id_of_rowid[rowid] for rowid, _distance in hits
                if rowid in id_of_rowid and rowid != int(episode["rowid"])
            ][:emb_top_k]
            for other in neighbors:
                left = str(episode["id"])
                pairs.add((left, other) if left < other else (other, left))
    except (RuntimeError, sqlite3.Error, TypeError, ValueError):
        return exact("vec_query_failed")

    ordered_pairs = [list(pair) for pair in sorted(pairs)]
    pairs_hash = "sha256:" + hashlib.sha256(
        aggregation_canonical_json(ordered_pairs).encode("utf-8")
    ).hexdigest()
    return pairs, {
        "contract": "hymem-aggregation-blocking-v1",
        "mode": "knn",
        "reason": "verified_full_shadow",
        "top_k": emb_top_k,
        "candidate_count": len(pairs),
        "candidate_pairs_sha256": pairs_hash,
        "episode_order_sha256": universe_hash,
        "vec_model": model,
        "vec_dimension": dimension,
        "vec_row_count": len(actual_shadow),
    }


def capture_aggregation_material(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    embedding_client: EmbeddingClient | None,
    *,
    episode_ceiling_rowid: int | None = None,
    pending_generation_key: str | None = None,
    pending_attempt_token: int | None = None,
) -> CapturedAggregationMaterial:
    """Capture one exact material epoch in a coherent read transaction.

    The caller's historical ceiling is advisory only. We recapture the current
    eligible high-water inside this snapshot, so an episode landing after the
    runner's earlier phase boundary cannot be silently omitted. A later insert
    advances the material clock and aborts publication.
    """

    del episode_ceiling_rowid
    if conn.in_transaction:
        raise RuntimeError("aggregation material capture requires no transaction")
    producer, embedding_model, embedding_dim = embedding_execution_identity(
        embedding_client
    )
    if embedding_client is None:
        embedding_model = None
    config_version = aggregation_config_version(cfg)
    conn.execute("BEGIN")
    try:
        revision = current_aggregation_material_revision(conn)
        # Root anchors are reselected and exact-proof-hashed at every build
        # fence and every serving publication load. A guessed MIN horizon over
        # a broader KG relation can only introduce false expiry (and cannot
        # replace that exact check), so v57 carries no independent time lease.
        fresh_until = None
        # Load the complete generation-visible, proof-valid universe.
        episodes = load_clusterable_episodes(
            conn, max_rowid=None, embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        # This transaction already owns a coherent snapshot. An allocation
        # rowid ceiling is neither needed nor portable and would make VACUUM
        # change an otherwise identical material identity.
        ceiling = None
        candidate_pairs, blocking = _aggregation_candidate_plan(
            conn, episodes, emb_top_k=cfg.aggregation_blocking_top_k,
        )
        # A rootless empty publication consumes no root prompt material.  Bind
        # the canonical inert root scope until at least one episode can feed a
        # digest root; the first eligible episode is clock-fenced and forces a
        # fresh capture.
        root_anchors_enabled = bool(cfg.aggregation_digest_enabled and episodes)
        anchors = load_root_anchor_inputs(
            conn, cfg.aggregation_digest_anchor_facts,
        ) if root_anchors_enabled else []
        selected_generation_keys = aggregation_anchor_phase1_generation_keys(
            anchors,
        )
        (
            phase1_scope,
            phase1_scope_exact,
            phase1_scope_reuse,
        ) = (
            aggregation_phase1_scope_identity(
                conn, generation_keys=selected_generation_keys,
            )
            if selected_generation_keys
            else disabled_aggregation_phase1_scope_identity()
        )
        if current_aggregation_material_revision(conn) != revision:
            raise RuntimeError("aggregation material changed during capture")
        episode_records = [
            {
                "ordinal": ordinal,
                "id": episode["id"],
                "session_id": episode["session_id"],
                "order_message_id": episode["start_message_id"],
                "cluster_input_proof_sha256": episode["cluster_input_proof"].proof_hash,
                "plain_input_proof_sha256": episode["plain_input_proof"].proof_hash,
                "entities": sorted(episode["entities"]),
                "embedding": episode["embedding_record"],
            }
            for ordinal, episode in enumerate(episodes)
        ]
        anchor_records = [
            {
                "ordinal": ordinal,
                "kind": anchor.kind,
                "source_key": anchor.source_key,
                "proof_sha256": anchor.proof_hash,
            }
            for ordinal, anchor in enumerate(anchors)
        ]
        binding = aggregation_material_binding(
            material_revision=revision,
            config_version=config_version,
            episode_ceiling_rowid=ceiling,
            episode_records=episode_records,
            anchor_records=anchor_records,
            blocking=blocking,
            embedding_binding=producer,
            embedding_dimension=embedding_dim,
            node_embedding_required=embedding_client is not None,
            root_anchors_enabled=root_anchors_enabled,
            phase1_scope_sha256=phase1_scope,
            phase1_scope_identity_exact=phase1_scope_exact,
            phase1_scope_reuse_scope=phase1_scope_reuse,
            fresh_until=fresh_until,
        )
        # Registration and the runner's durable pending attribution share the
        # exact capture transaction. A crash after this commit can therefore
        # never leave a successfully captured epoch unattributed, and source
        # invalidators can scope vector mutations to this pending producer.
        register_aggregation_material_epoch(conn, binding)
        if (pending_generation_key is None) != (pending_attempt_token is None):
            raise ValueError(
                "aggregation pending generation and attempt must be paired"
            )
        if pending_generation_key is not None:
            from hymem.dreaming.aggregation_health import (
                bind_pending_aggregation_material,
            )
            bind_pending_aggregation_material(
                conn, config_version, pending_generation_key,
                pending_attempt_token,
                str(binding["material_epoch_key"]),
            )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return CapturedAggregationMaterial(
        episodes=tuple(episodes), anchor_inputs=tuple(anchors),
        candidate_pairs=(
            None if candidate_pairs is None else frozenset(candidate_pairs)
        ),
        blocking=blocking, binding=binding,
    )


def _node_id(
    member_ids: list[str], *, salt: str = "",
    input_fingerprint: str | None = None,
) -> str:
    """Stable id for a node = content hash of its sorted member ids, so an
    unchanged cluster keeps its id (and cached embedding + fusion) across dream
    cycles. `salt` separates id spaces for nodes that could share a member set
    but carry a different KIND of fusion (the root digest uses a different
    prompt than an intermediate rollup, so they must never reuse each other)."""
    kind_by_salt = {
        _CLUSTER_SALT: "cluster", _ROLLUP_SALT: "rollup", _ROOT_SALT: "root",
    }
    kind = kind_by_salt.get(salt)
    if kind is None or input_fingerprint is None:
        # Compatibility for pure clustering probes which deliberately exercise
        # only a local synthetic id, never a publishable material node.
        payload = f"{salt}::{'|'.join(sorted(member_ids))}::input={input_fingerprint}"
        return "agg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return aggregation_node_id(
        member_ids, node_kind=kind, input_fingerprint=input_fingerprint,
    )


def _stable_sample(seq: list[dict], cap: int) -> list[dict]:
    """At most `cap` items chosen by id-hash rank, returned in input order.
    Caps the digest's pass-through leaves. Two properties matter, in this
    order:

    1. STABILITY UNDER CHURN: adding or removing one item displaces at most
       one selected leaf (its hash rank bumps exactly one other item across
       the cap line). The index-arithmetic predecessor (`_evenly_spaced`,
       round(i*(n-1)/(cap-1))) recomputed every pick from `len(seq)`, so a
       single new leftover episode swapped a large fraction of the selected
       leaves, re-keying most rollup fusions above them — the dominant
       amplifier in the 2026-07-12 reuse instability (one quiet episode →
       ~50% reuse).
    2. WHOLE-SPAN COVERAGE: the hash rank is uniform over items, so the
       selection still spans the full backlog in expectation (a recency slice
       `seq[-cap:]` would digest only the newest stretch); it is merely no
       longer perfectly evenly spaced, which the fusion never depended on.

    `cap <= 0` means uncapped, matching the old semantics."""
    if cap <= 0 or len(seq) <= cap:
        return list(seq)
    ranked = sorted(
        seq,
        key=lambda e: hashlib.sha1(f"leaf::{e['id']}".encode("utf-8")).hexdigest(),
    )
    keep_ids = {e["id"] for e in ranked[:cap]}
    return [e for e in seq if e["id"] in keep_ids]


def _centroid(vectors: list[list[float] | None]) -> list[float] | None:
    """Mean of the non-None member vectors (None if there are none) — gives a
    rollup item a clusterable vector without an embedding call; the persisted
    node embedding is computed separately from the fused text."""
    present = [v for v in vectors if v]
    if not present:
        return None
    dim = len(present[0])
    if any(len(v) != dim for v in present):
        return None
    n = len(present)
    return [sum(v[i] for v in present) / n for i in range(dim)]


def select_clusters(
    episodes: list[dict], cfg: HyMemConfig,
    conn: sqlite3.Connection | None = None,
    *, candidate_pairs: object = _CANDIDATE_PAIRS_UNSET,
) -> list[list[dict]]:
    """Cluster all episodes, then keep only the clusters worth a summary: at least
    `aggregation_min_members` episodes spanning at least `aggregation_min_sessions`
    distinct sessions. Returns each kept cluster's episodes in load order.

    Clustering runs with the `aggregation_max_cluster_size` chaining guard
    (0 in config = uncapped → None here): an over-cap component arrives as
    recency windows, and each window flows through the SAME min-members /
    min-sessions policy below — an undersized trailing window (the NEWEST,
    still-filling slice) is dropped here exactly like any other too-small
    cluster; its episodes still reach the digest as leftover leaves.

    `conn` enables Stage-3b candidate blocking (the KNN cosine arm needs the
    store's `vec_episodes` table); None — pure offline callers — means exact
    all-pairs clustering, as does any condition under which
    `generate_candidate_pairs` declines to block."""
    if not episodes:
        return []
    pairs = (
        generate_candidate_pairs(
            conn, episodes, emb_top_k=cfg.aggregation_blocking_top_k)
        if candidate_pairs is _CANDIDATE_PAIRS_UNSET and conn is not None
        else None if candidate_pairs is _CANDIDATE_PAIRS_UNSET
        else candidate_pairs
    )
    labels = cluster_episodes(
        episodes, cfg.aggregation_emb_threshold, cfg.aggregation_ent_threshold,
        max_cluster_size=cfg.aggregation_max_cluster_size or None,
        candidate_pairs=pairs,
    )
    grouped: dict[int, list[dict]] = {}
    for ep in episodes:
        grouped.setdefault(labels[ep["id"]], []).append(ep)

    kept: list[list[dict]] = []
    for members in grouped.values():
        check_current_deadline()
        # The fusion prompt must see every member a persisted node attests.  The
        # clustering cap and prompt cap are independent knobs (15 vs 12 by
        # default), so partition oversized components here instead of slicing
        # them invisibly inside `_summarize_cluster`.
        bounded = (
            _content_defined_groups(members, max(1, cfg.aggregation_max_members))
            if len(members) > max(1, cfg.aggregation_max_members)
            else [members]
        )
        for group in bounded:
            if len(group) < cfg.aggregation_min_members:
                continue
            if len({m["session_id"] for m in group}) < cfg.aggregation_min_sessions:
                continue
            kept.append(group)
    # Deterministic CHRONOLOGY-STABLE order: by oldest member (ingestion order,
    # id tiebreak). The previous larger-clusters-first sort reordered the whole
    # rollup frontier whenever any cluster's size changed relative to another,
    # recomposing downstream rollup groups and re-keying their cached fusions
    # for a membership change that touched one cluster.
    def _oldest_member(c: list[dict]) -> tuple:
        return min(
            (m["start_message_id"] if isinstance(m.get("start_message_id"), int)
             else float("inf"), m["id"])
            for m in c
        )
    kept.sort(key=_oldest_member)
    return kept


# Persist-time bounds (facts.py:211 model): a bloated fusion summary would
# propagate into every ancestor's render via _items_text and crowd out
# siblings under aggregation_max_members — cap it where it is visible (at
# persist time, with a warning) instead of at the token ceiling where it
# becomes a silent parse failure.
_MAX_FUSION_TITLE_CHARS = AGGREGATION_MAX_TITLE_CHARS
_MAX_FUSION_SUMMARY_CHARS = AGGREGATION_MAX_SUMMARY_CHARS

class _RebuildForecast(NamedTuple):
    """Structural account of one dream's rebuild. See `_forecast_rebuild`."""
    predicted: int
    actual: int
    residual: int
    facts_rekey: int
    # v33: `actual` split by tree level. level0 + rollup + root == actual by
    # construction, so the triple is self-checking.
    rebuilt_level0: int = 0
    rebuilt_rollup: int = 0
    rebuilt_root: int = 0


def _forecast_rebuild(
    rows: list[dict], prev_inputs: set[tuple],
) -> _RebuildForecast:
    """Predict this dream's rebuild from effective prompt inputs.

    The amplification model `rebuilt ~ A*level0_missed + root + leaf` was never
    fittable on this store: `level0_missed` sat at 3 for 11 of 13 dreams
    (2026-08-09), so the slope had one x-value and the intercept absorbed the
    leaf term by construction. This replaces the fit rather than waiting for a
    dispersion that is not coming.

    A node must be rebuilt when its effective input is new.  That includes the
    ordered member ids *and* the input fingerprint binding member text and
    provenance (plus root anchors).  It is computable per dream instead of
    estimated across dreams. So:

        predicted = nodes whose (level, member set, input fingerprint,
                    generation, request) is absent from the previous tree
        actual    = nodes whose id missed the fusion cache
        residual  = actual - predicted

    Residual 0 means every rebuild this dream is accounted for by an input
    change. A POSITIVE residual is the interesting signal and the reason this
    is not circular: it counts nodes that kept their exact effective inputs and
    still failed to reuse. That is the fifth-cause
    class §0.3 hunts for (salt bump, hash instability, rowid/shadow desync),
    and it is visible on a SINGLE dream with no bar to calibrate.

    A root whose membership is unchanged but effective input changes is also
    counted as ``facts_rekey`` for continuity with the instrumentation surface.

    A negative residual is not an error either: it means a node whose
    membership is new was nevertheless served from cache, which happens when
    two levels share a member set. It is reported as-is rather than clamped —
    a clamp would hide the collision.
    """
    predicted = actual = facts_rekey = 0
    lvl0 = rollup = root = 0
    # Accept the historical two/three-tuple forms in pure unit callers.
    # Persisted predecessor state uses the five-tuple form: any declared
    # generation/request rotation is a deliberate cache boundary, not
    # unexplained key drift. A key change *within* the same generation and
    # request remains the residual this instrument exists to expose.
    membership_keys = {(item[0], item[1]) for item in prev_inputs}
    input_keys = {
        (
            item[0], item[1],
            item[2] if len(item) > 2 else None,
            item[3] if len(item) > 3 else None,
            item[4] if len(item) > 4 else None,
        )
        for item in prev_inputs
    }
    for r in rows:
        membership_key = (r["level"], frozenset(r["member_ids_list"]))
        membership_is_new = membership_key not in membership_keys
        input_is_new = (
            *membership_key,
            r.get("input_fingerprint"),
            r.get("aggregation_generation_key"),
            r.get("aggregation_request_hash"),
        ) not in input_keys
        was_rebuilt = not r.get("reused", False)
        if was_rebuilt:
            actual += 1
            # v33 decomposition. Which LEVEL rebuilt is what separates the two
            # readings of a low-reuse leaf-changed row: level-0 rebuilds track
            # episode arrivals into clusters, level->=1 interior rebuilds track
            # the digest leaf set shifting and cascading up. `leaf_changed` is
            # binary and cannot tell those apart, which is why #1183/#1307/#1317
            # stayed arguable across three readings.
            if r["is_root"]:
                root += 1
            elif r["level"] == 0:
                lvl0 += 1
            else:
                rollup += 1
        if input_is_new:
            predicted += 1
            if not membership_is_new and r["is_root"]:
                facts_rekey += 1
        elif was_rebuilt and r["is_root"]:
            # Backward compatibility for callers without fingerprints: an
            # unchanged root rebuild historically represented a facts re-key.
            facts_rekey += 1
            predicted += 1
    return _RebuildForecast(
        predicted, actual, actual - predicted, facts_rekey, lvl0, rollup, root,
    )


def _leaf_fingerprint(leaf_ids: frozenset[str]) -> str:
    """Order-independent fingerprint of a digest leaf set.

    Only equality is ever tested, so the id list is not stored — see migration
    030. Sorted before hashing for the same reason `_node_id` sorts: the
    selection order must not re-key an unchanged set.
    """
    return hashlib.sha1("\x00".join(sorted(leaf_ids)).encode()).hexdigest()


def _read_leaf_fingerprint(conn: sqlite3.Connection) -> str | None:
    """The leaf set the last dream that persisted aggregation actually used.
    None means no dream ever has — unattributed, NOT an unchanged set."""
    row = conn.execute(
        "SELECT fingerprint FROM aggregation_leaf_state WHERE id = 1"
    ).fetchone()
    return row[0] if row else None


def _read_leaf_ids(conn: sqlite3.Connection) -> frozenset[str] | None:
    """The previous dream's leaf ID SET, for the v34 size-of-shift channel.

    None means "not attributable" and covers BOTH the no-predecessor case and a
    pre-v34 watermark row that only ever stored a fingerprint. Both must read
    NULL rather than an empty set: `frozenset()` would make every leaf look
    newly added, which is a counterfeit reading of exactly the kind v29's NULL
    contract exists to prevent."""
    row = conn.execute(
        "SELECT leaf_ids FROM aggregation_leaf_state WHERE id = 1"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return frozenset(json.loads(row[0]))


def _write_leaf_fingerprint(conn: sqlite3.Connection, fingerprint: str,
                            n_leaves: int,
                            leaf_ids: frozenset[str] | None = None) -> None:
    """Advance the watermark. Called INSIDE the node-persist transaction so it
    commits with the nodes that consumed this leaf set, never ahead of them."""
    conn.execute(
        """
        INSERT INTO aggregation_leaf_state(id, fingerprint, n_leaves, leaf_ids,
                                           updated_at)
        VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            fingerprint = excluded.fingerprint,
            n_leaves = excluded.n_leaves,
            leaf_ids = excluded.leaf_ids,
            updated_at = CURRENT_TIMESTAMP
        """,
        (fingerprint, n_leaves,
         json.dumps(sorted(leaf_ids)) if leaf_ids is not None else None),
    )


def _fusion_max_tokens(prompt: str) -> int:
    """Payload-sized ceiling with headroom (rules_extract.py:223 model: size
    the ceiling so the reply never truncates). Fusion output scales with the
    rendered input; the bare 1024 default cut big rollup/root prompts
    (measured cut band 3769-4619 chars, ~52% of dreams mid-drain). Capped at
    8192 — the retry ladder (re-roll, then membership-preserving shrink)
    covers anything beyond.
    """
    return aggregation_fusion_max_tokens(prompt)


def _fusion_request(user_prompt: str, *, system: str) -> LLMRequest:
    """Construct the sole request shape allowed to mint aggregation text."""

    return LLMRequest(
        system=system, user=user_prompt, response_format="json",
        temperature=0.0, max_tokens=_fusion_max_tokens(user_prompt),
    )


def _llm_fuse(
    user_prompt: str, llm: LLMClient, *, system: str, kind: str = "fusion",
    shrink: Callable[[], str] | None = None,
    verify_generation: Callable[[], None] | None = None,
    prepared_request: LLMRequest | None = None,
) -> dict | None:
    """One LLM call fusing the prepared `user_prompt` into {title, summary}.
    Returns None when the call fails or yields nothing usable (so no empty
    node is persisted). Every failure path logs at WARNING with `kind`
    (cluster/rollup/root): a failed fusion retries on every subsequent dream
    until it succeeds, and each fail→heal transition costs reuse — silent
    failures made the 2026-07-12 low-reuse runs unattributable.

    RETRY LADDER (2026-08-07): the ceiling is payload-sized (see
    `_fusion_max_tokens`). When the reply still fails to parse and the
    structural cut detector fires (opens '{', unterminated — same evidence
    finish_reason="length" would give, computed from the string in hand),
    the SAME input is re-rolled ONCE. The re-roll is licensed empirically by
    deepseek-v4-flash output variance at temperature=0.0 (measured 0.3x-4.8x
    output spread on identical input); it is NOT a Protocol guarantee — a
    deterministic backend turns it into a wasted call every time. The
    v55 deliberately does not execute ``shrink``. A reduced render is a
    different effective input and cannot be published under the full-input
    fingerprint. The parameter remains only for source compatibility with
    older callers; a same-prompt re-roll remains safe.
    """
    def attempt(prompt: str) -> tuple[dict | None, str | None, str]:
        request = (
            prepared_request
            if prepared_request is not None else _fusion_request(
                prompt, system=system
            )
        )
        if request.system != system or request.user != prompt:
            raise ValueError("prepared aggregation request disagrees with prompt")
        request_hash = aggregation_llm_request_hash(request)
        try:
            if verify_generation is not None:
                verify_generation()
            raw = llm.complete(request)
            if verify_generation is not None:
                verify_generation()
        except Exception:
            log.exception("aggregate.fusion_failure kind=%s stage=call", kind)
            return None, None, "call"
        data = loads_lenient(raw, expect="object")
        if data is None:
            log.warning("aggregate.fusion_failure kind=%s stage=parse raw_len=%d",
                        kind, len(raw) if isinstance(raw, str) else -1)
            return None, raw, "parse"
        if not isinstance(data, dict):
            log.warning("aggregate.fusion_failure kind=%s stage=shape", kind)
            return None, raw, "shape"
        title = data.get("title", "")
        summary = data.get("summary", "")
        if not isinstance(title, str) or not isinstance(summary, str):
            log.warning("aggregate.fusion_failure kind=%s stage=shape", kind)
            return None, raw, "shape"
        title, summary = title.strip(), summary.strip()
        if not title or not summary:
            log.warning("aggregate.fusion_failure kind=%s stage=empty", kind)
            return None, raw, "empty"
        if len(summary) > _MAX_FUSION_SUMMARY_CHARS:
            log.warning("aggregate.fusion_summary_capped kind=%s chars=%d->%d",
                        kind, len(summary), _MAX_FUSION_SUMMARY_CHARS)
            summary = summary[:_MAX_FUSION_SUMMARY_CHARS]
        if len(title) > _MAX_FUSION_TITLE_CHARS:
            title = title[:_MAX_FUSION_TITLE_CHARS]
        return {
            "title": title, "summary": summary,
            "_aggregation_request_hash": request_hash,
        }, raw, "ok"

    fused, raw, stage = attempt(user_prompt)
    if fused is None and stage == "parse" and raw is not None and is_ceiling_cut(raw):
        # One re-roll of the SAME input — empirical license, see docstring.
        fused, raw, stage = attempt(user_prompt)
    del shrink
    return fused


def _summarize_cluster(
    members: list[dict], cfg: HyMemConfig, llm: LLMClient,
    *, verify_generation: Callable[[], None] | None = None,
    user_prompt: str | None = None,
    prepared_request: LLMRequest | None = None,
) -> dict | None:
    """Fuse a level-0 cluster's episodes into {title, summary}."""
    def render(scale: float = 1.0) -> str:
        if scale >= 1.0:
            return "\n\n---\n\n".join(
                m["cluster_input_proof"].rendered_text for m in members
            )
        return "\n\n---\n\n".join(
            f"[{m['session_id']}] {m['title'][:int(len(m['title']) * scale)]}\n"
            f"{m['summary'][:int(len(m['summary']) * scale)]}" for m in members
        )
    return _llm_fuse(
        user_prompt if user_prompt is not None else AGGREGATE_USER_TEMPLATE.format(text=render()), llm,
        system=AGGREGATE_SYSTEM, kind="cluster",
        verify_generation=verify_generation,
        prepared_request=prepared_request,
    )


def _items_text(items: list[dict], cfg: HyMemConfig, *, char_scale: float = 1.0) -> str:
    """Render hierarchy items (level-0 nodes / rollups / pass-through episodes,
    all carrying title+summary) as one fusion input block.

    `char_scale` remains a compatibility hook for direct formatting tests.
    Published v55 fusion always uses 1.0: a shorter render is a different
    effective input and may not reuse the full-input proof identity.
    """
    if len(items) > max(2, cfg.aggregation_max_members):
        raise ValueError("aggregation fusion input exceeds its member budget")
    if char_scale >= 1.0:
        return "\n\n---\n\n".join(
            (
                item["input_proof"].rendered_text
                if isinstance(item.get("input_proof"), AggregationInputProof)
                else f"{item['title']}\n{item['summary']}"
            )
            for item in items
        )
    parts = []
    for m in items:
        t = m["title"] or ""
        s = m["summary"] or ""
        parts.append(
            f"{t[:int(len(t) * char_scale)]}\n{s[:int(len(s) * char_scale)]}"
        )
    return "\n\n---\n\n".join(parts)


def _anchor_facts(conn: sqlite3.Connection, cap: int) -> list[str]:
    """Compatibility rendering of only exact, typed, source-backed anchors."""

    return [item.rendered_text for item in load_root_anchor_inputs(conn, cap)]


@dataclass
class PendingAggregationNodeEmbeddings:
    node_ids: list[str]
    text_hashes: list[str]
    vectors: list[list[float]]
    from_cache: list[bool]
    model: str
    dim: int
    cache_hits: int = 0


def _prepare_candidate_node_embeddings(
    conn: sqlite3.Connection,
    rows: list[dict],
    embedder: EmbeddingClient,
    *, verify_material: Callable[[], None] | None = None,
) -> PendingAggregationNodeEmbeddings | None:
    """Embed an in-memory exact candidate before acquiring the write lock.

    Every candidate row was produced from typed input proofs above. Existing
    node vectors/cache entries are reusable only for the exact output-text hash
    and provider identity. The returned batch contains *all* level-0 candidate
    vectors, including reuse hits, because atomic replacement deletes the old
    node rows (and therefore their FK-owned embedding mirrors).
    """

    if conn.in_transaction:
        raise RuntimeError("candidate embedding fetch requires no transaction")
    model, initial_dim = _embedding_identity(embedder)
    candidates = sorted(
        (
            row["id"], f"{row['title']}\n{row['summary']}",
            embedding_text_hash(f"{row['title']}\n{row['summary']}"),
        )
        for row in rows
        if row["node_kind"] == "cluster" and int(row["level"]) == 0
    )
    if not candidates:
        return None

    existing = {
        str(row["node_id"]): row
        for row in conn.execute(
            "SELECT node_id,vector_json,model,dim,text_hash,"
            "embedding_producer_key "
            "FROM aggregation_node_embeddings WHERE model=? AND dim=?",
            (model, initial_dim),
        ).fetchall()
    }
    vectors: list[list[float] | None] = [None] * len(candidates)
    from_cache = [False] * len(candidates)
    for index, (node_id, _text, text_hash) in enumerate(candidates):
        stored = existing.get(str(node_id))
        if (
            stored is None or stored["text_hash"] != text_hash
            or stored["embedding_producer_key"] != model
        ):
            continue
        try:
            decoded = decode_vector(stored["vector_json"])
        except (AttributeError, UnicodeError, TypeError, ValueError):
            continue
        vector = _finite_embedding_vector(decoded, expected_dim=initial_dim)
        if vector is not None:
            vectors[index] = vector
            from_cache[index] = True

    hashes = [text_hash for _node_id, _text, text_hash in candidates]
    missing_hashes = [
        text_hash for index, text_hash in enumerate(hashes)
        if vectors[index] is None
    ]
    cached = _fetch_cached_vectors(
        conn, missing_hashes, model, expected_dim=initial_dim
    )
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    for index, (_node_id, text, text_hash) in enumerate(candidates):
        if vectors[index] is not None:
            continue
        cached_vector = cached.get(text_hash)
        if cached_vector is None:
            miss_indices.append(index)
            miss_texts.append(text)
        else:
            vectors[index] = cached_vector
            from_cache[index] = True

    if miss_texts:
        if verify_material is not None:
            verify_material()
        embedded = embedder.embed(miss_texts)
        if verify_material is not None:
            verify_material()
        if len(embedded) != len(miss_texts):
            raise RuntimeError(
                f"embedding client returned {len(embedded)} vectors for "
                f"{len(miss_texts)} aggregation nodes"
            )
        final_dim = _post_embed_identity(embedder, expected_model=model)
        if final_dim != initial_dim and any(from_cache):
            redo_indices = [index for index, hit in enumerate(from_cache) if hit]
            if verify_material is not None:
                verify_material()
            redo = embedder.embed([candidates[index][1] for index in redo_indices])
            if verify_material is not None:
                verify_material()
            if len(redo) != len(redo_indices):
                raise RuntimeError(
                    "embedding client returned the wrong number of node vectors"
                )
            redo_dim = _post_embed_identity(embedder, expected_model=model)
            if redo_dim != final_dim:
                raise RuntimeError(
                    "embedding client changed dimension during node retry"
                )
            for index, vector in zip(redo_indices, redo):
                vectors[index] = vector
                from_cache[index] = False
        for index, vector in zip(miss_indices, embedded):
            vectors[index] = vector
    else:
        final_dim = initial_dim

    validated = [
        _finite_embedding_vector(vector, expected_dim=final_dim)
        for vector in vectors
    ]
    if any(vector is None for vector in validated):
        raise RuntimeError("embedding client returned malformed node vectors")
    return PendingAggregationNodeEmbeddings(
        node_ids=[str(node_id) for node_id, _text, _hash in candidates],
        text_hashes=hashes,
        vectors=[vector for vector in validated if vector is not None],
        from_cache=from_cache,
        model=model,
        dim=final_dim,
        cache_hits=sum(from_cache),
    )


def fetch_node_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient,
) -> PendingAggregationNodeEmbeddings | None:
    """Prepare vectors only for the wholly proven current publication."""
    if conn.in_transaction:
        raise RuntimeError("current-publication embedding fetch requires no transaction")
    model, initial_dim = _embedding_identity(embedder)
    pending: list[tuple[str, str, str]] = []
    # Capture publication proof, exact output bytes, and embedding mirror in
    # one read snapshot. Provider work begins only after the snapshot is
    # released, and it renders exclusively from the validated proof row.
    conn.execute("BEGIN")
    try:
        current = load_current_aggregation_publication(
            conn, embedding_client=embedder,
        )
        stored_by_id = {
            str(row["node_id"]): row
            for row in conn.execute(
                "SELECT node_id,text_hash,model,dim,vector_json,"
                "embedding_producer_key "
                "FROM aggregation_node_embeddings"
            ).fetchall()
        }
        if current is not None:
            for node_id, proof in sorted(current.nodes.items()):
                if proof.row["node_kind"] != "cluster":
                    continue
                text = f"{proof.row['title']}\n{proof.row['summary']}"
                text_hash = embedding_text_hash(text)
                stored_row = stored_by_id.get(node_id)
                if (
                    stored_row is not None
                    and stored_row["text_hash"] == text_hash
                    and stored_row["model"] == model
                    and stored_row["dim"] == initial_dim
                    and stored_row["embedding_producer_key"] == model
                ):
                    try:
                        stored = decode_vector(stored_row["vector_json"])
                    except (AttributeError, UnicodeError, TypeError, ValueError):
                        stored = None
                    if _finite_embedding_vector(
                        stored, expected_dim=initial_dim
                    ) is not None:
                        continue
                pending.append((node_id, text, text_hash))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    if not pending:
        return None

    hashes = [h for _, _, h in pending]
    cached = _fetch_cached_vectors(
        conn, hashes, model, expected_dim=initial_dim
    )
    vectors: list[list[float] | None] = [None] * len(pending)
    from_cache = [False] * len(pending)
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    for index, (_, text, text_hash) in enumerate(pending):
        cached_vector = cached.get(text_hash)
        if cached_vector is None:
            miss_indices.append(index)
            miss_texts.append(text)
        else:
            vectors[index] = cached_vector
            from_cache[index] = True

    if miss_texts:
        embedded = embedder.embed(miss_texts)
        if len(embedded) != len(miss_texts):
            raise RuntimeError(
                f"embedding client returned {len(embedded)} vectors for "
                f"{len(miss_texts)} aggregation nodes"
            )
        final_dim = _post_embed_identity(embedder, expected_model=model)
        if final_dim != initial_dim and any(from_cache):
            redo_indices = [i for i, hit in enumerate(from_cache) if hit]
            redo = embedder.embed([pending[i][1] for i in redo_indices])
            if len(redo) != len(redo_indices):
                raise RuntimeError(
                    "embedding client returned the wrong number of node vectors"
                )
            redo_dim = _post_embed_identity(embedder, expected_model=model)
            if redo_dim != final_dim:
                raise RuntimeError(
                    "embedding client changed dimension during node retry"
                )
            for index, vector in zip(redo_indices, redo):
                vectors[index] = vector
                from_cache[index] = False
        for index, vector in zip(miss_indices, embedded):
            vectors[index] = vector
    else:
        final_dim = initial_dim

    validated = [
        _finite_embedding_vector(vector, expected_dim=final_dim)
        for vector in vectors
    ]
    if any(vector is None for vector in validated):
        raise RuntimeError("embedding client returned malformed node vectors")
    return PendingAggregationNodeEmbeddings(
        node_ids=[node_id for node_id, _, _ in pending],
        text_hashes=hashes,
        vectors=[vector for vector in validated if vector is not None],
        from_cache=from_cache,
        model=model,
        dim=final_dim,
        cache_hits=sum(from_cache),
    )


@core_db.embedding_writer
def persist_node_embeddings(
    conn: sqlite3.Connection, pending: PendingAggregationNodeEmbeddings,
    *, publication_id: str | None = None,
    material_epoch_key: str | None = None,
) -> int:
    """Persist a validated node batch in the caller's short transaction."""
    persisted = 0
    for node_id, text_hash, candidate, is_cached in zip(
        pending.node_ids,
        pending.text_hashes,
        pending.vectors,
        pending.from_cache,
    ):
        vector = _finite_embedding_vector(candidate, expected_dim=pending.dim)
        if vector is None:
            continue
        proof = (
            load_aggregation_node_proof(
                conn, node_id,
                expected_material_epoch_key=material_epoch_key,
            )
            if publication_id is not None
            else load_current_aggregation_node_proof(conn, node_id)
        )
        if (
            proof is None
            or (publication_id is not None
                and proof.row["publication_id"] != publication_id)
            or embedding_text_hash(
                f"{proof.row['title']}\n{proof.row['summary']}"
            ) != text_hash
        ):
            continue
        if not is_cached:
            conn.execute(
                """
                INSERT INTO embedding_cache(text_hash, model, vector_json, dim)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(text_hash, model) DO UPDATE SET
                    vector_json=excluded.vector_json,
                    dim=excluded.dim,
                    created_at=CURRENT_TIMESTAMP
                """,
                (text_hash, pending.model, encode_vector(vector), pending.dim),
            )
        conn.execute(
            """
            INSERT INTO aggregation_node_embeddings(
                node_id,vector_json,model,dim,text_hash,embedding_producer_key
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                vector_json = excluded.vector_json,
                model = excluded.model,
                dim = excluded.dim,
                text_hash = excluded.text_hash,
                embedding_producer_key = excluded.embedding_producer_key
            """,
            (
                node_id, encode_vector(vector), pending.model,
                pending.dim, text_hash, pending.model,
            ),
        )
        persisted += 1
    return persisted


def _reusable_fusion(
    conn: sqlite3.Connection,
    node_id: str,
    cached: dict | None,
    *,
    expected_inputs: tuple[AggregationInputProof, ...],
    expected_config_version: str,
    expected_generation_key: str,
    expected_request_hash: str,
) -> dict | None:
    """Return a cached fusion only when its exact effective input still agrees."""

    if cached is None:
        return None
    proof = load_aggregation_node_proof(
        conn, node_id, expected_generation_key=expected_generation_key,
        # Physical fusion reuse depends on the exact typed prompt, v56
        # producer/request, and canonical output—not on whether the old tree's
        # material epoch is still current.  Supplying the row's own immutable
        # key authorizes validation of a process-scoped registry row without
        # turning it into public read authority; the candidate is re-emitted
        # under the newly captured epoch before publication.
        expected_material_epoch_key=cached.get(
            "aggregation_material_epoch_key"
        ),
    )
    if (
        proof is None
        or proof.inputs != expected_inputs
        or proof.row["build_config_version"] != expected_config_version
        or proof.row["aggregation_generation_key"] != expected_generation_key
        or proof.row["aggregation_request_hash"] != expected_request_hash
    ):
        return None
    return {
        "title": proof.row["title"], "summary": proof.row["summary"],
        "_aggregation_request_hash": proof.row["aggregation_request_hash"],
    }


def _candidate_node_row(
    *, inputs: tuple[AggregationInputProof, ...], fused: dict,
    node_kind: str, level: int, config_version: str,
    generation_key: str, request_hash: str, material_epoch_key: str,
    reused: bool,
) -> dict:
    """Create the exact DB-shaped, self-verifying header for one candidate."""

    anchors_started = False
    member_ids: list[str] = []
    typed_member_keys: list[tuple[str, str]] = []
    for item in inputs:
        if item.kind in {"episode", "aggregation_node"}:
            if anchors_started:
                raise ValueError("aggregation members must precede anchors")
            member_id = item.source_ref.get("id")
            if not isinstance(member_id, str) or not member_id:
                raise ValueError("aggregation member reference is malformed")
            member_ids.append(member_id)
            typed_member_keys.append((item.kind, member_id))
        else:
            anchors_started = True
    if not member_ids or len(typed_member_keys) != len(set(typed_member_keys)):
        raise ValueError("aggregation node requires unique typed members")
    if node_kind == "rollup" and len(member_ids) < 2:
        raise ValueError("aggregation rollup requires at least two members")
    if node_kind != "root" and anchors_started:
        raise ValueError("only an aggregation root may carry anchor inputs")
    fingerprint = aggregation_typed_input_fingerprint(inputs)
    node_id = aggregation_node_id(
        member_ids, node_kind=node_kind, input_fingerprint=fingerprint,
        generation_key=generation_key, request_hash=request_hash,
    )
    occurrences = combine_source_occurrences(item.occurrences for item in inputs)
    sessions = sorted({item.session_id for item in occurrences})
    title = str(fused["title"])
    summary = str(fused["summary"])
    if fused.get("_aggregation_request_hash") != request_hash:
        raise ValueError("aggregation output/request proof disagrees")
    if not aggregation_output_is_canonical(title, summary):
        raise ValueError("aggregation fusion output is not canonical")
    return {
        "id": node_id,
        "title": title,
        "summary": summary,
        "member_ids_list": member_ids,
        "member_episode_ids": aggregation_canonical_json(member_ids),
        "session_ids_list": sessions,
        "session_ids": aggregation_canonical_json(sessions),
        "n_members": len(member_ids),
        "n_sessions": len(sessions),
        "level": level,
        "is_root": int(node_kind == "root"),
        "node_kind": node_kind,
        "output_hash": aggregation_output_hash(
            node_kind, title, summary, request_hash,
        ),
        "aggregation_request_hash": request_hash,
        "input_fingerprint": fingerprint,
        "input_manifest_version": AGGREGATION_INPUT_MANIFEST_VERSION,
        "input_manifest_count": len(inputs),
        "input_manifest_hash": aggregation_input_manifest_hash(inputs),
        "input_manifest_complete": 1,
        "aggregation_generation_key": generation_key,
        "aggregation_material_epoch_key": material_epoch_key,
        "source_manifest_version": AGGREGATION_SOURCE_MANIFEST_VERSION,
        "source_manifest_count": len(occurrences),
        "source_manifest_hash": source_manifest_hash(
            AGGREGATION_SOURCE_MANIFEST_VERSION, occurrences
        ),
        "source_manifest_complete": 1,
        "build_config_version": config_version,
        "publication_id": None,
        "input_proofs": inputs,
        "source_occurrences": occurrences,
        "reused": reused,
    }


def _node_frontier_item(
    row: dict, *, vector: list[float] | None, entities: set[str],
) -> dict:
    proof = make_aggregation_input_proof(
        kind="aggregation_node", source_ref={"id": row["id"]},
        rendered_text=f"{row['title']}\n{row['summary']}",
        authority_hash=aggregation_node_authority_hash(row),
        occurrences=row["source_occurrences"],
    )
    return {
        "id": row["id"], "title": row["title"], "summary": row["summary"],
        "_aggregation_cluster_key": f"aggregation_node:{row['id']}",
        "_aggregation_level": int(row["level"]),
        "vector": vector, "entities": entities,
        "session_ids": set(row["session_ids_list"]),
        "source_occurrences": row["source_occurrences"],
        "source_provenance_complete": True,
        "source_manifest_hash": row["source_manifest_hash"],
        "input_proof": proof,
    }


def build_aggregation_nodes(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    llm: LLMClient,
    embedding_client: EmbeddingClient | None = None,
    *,
    episode_ceiling_rowid: int | None = None,
    generation_binding: Mapping[str, object] | None = None,
    health_managed: bool = False,
    health_attempt_token: int | None = None,
) -> AggregationResult:
    """Build under one exact attempt, recording direct-call exceptions safely."""

    attempt_token_sink: list[int] = []
    try:
        return _build_aggregation_nodes_attempt(
            conn, cfg, llm, embedding_client,
            episode_ceiling_rowid=episode_ceiling_rowid,
            generation_binding=generation_binding,
            health_managed=health_managed,
            health_attempt_token=health_attempt_token,
            _attempt_token_sink=attempt_token_sink,
        )
    except Exception as exc:
        if not health_managed and attempt_token_sink:
            token = attempt_token_sink[0]
            pending = conn.execute(
                "SELECT pending_config_version,pending_generation_key "
                "FROM aggregation_build_health WHERE id=1 "
                "AND pending_attempt_token=?",
                (token,),
            ).fetchone()
            if pending is not None:
                from hymem.dreaming.aggregation_health import (
                    record_aggregation_build_failure,
                )
                try:
                    record_aggregation_build_failure(
                        conn, str(pending["pending_config_version"]),
                        str(pending["pending_generation_key"]), token,
                        caught_exceptions=1,
                    )
                except Exception as record_exc:
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            "aggregation failure attribution also failed: "
                            + type(record_exc).__name__
                        )
        raise


def _build_aggregation_nodes_attempt(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    llm: LLMClient,
    embedding_client: EmbeddingClient | None = None,
    *,
    episode_ceiling_rowid: int | None = None,
    generation_binding: Mapping[str, object] | None = None,
    health_managed: bool = False,
    health_attempt_token: int | None = None,
    _attempt_token_sink: list[int] | None = None,
) -> AggregationResult:
    """Rebuild the cross-session aggregation layer from the current episodes.

    No-op when the layer is disabled. Otherwise: cluster → keep cross-session
    multi-member clusters → fuse each NEW cluster with one LLM call (an
    unchanged member set reuses the stored fusion, no call) → full-replace
    `aggregation_nodes` → (re-)embed node summaries. Returns the node count and
    the reused-fusion count (see :class:`AggregationResult`).
    Full rebuild (DELETE then INSERT) because membership is a pure function of
    the present episodes; the content-hash id means an unchanged cluster keeps
    both its fusion and its embedding. Caller need not hold a transaction.
    `episode_ceiling_rowid` (set by the dream runner) freezes that 'present
    episodes' set at the phase-3 boundary so an async write mid-build can't
    shift membership and trigger a spurious refusion; see
    :func:`load_clusterable_episodes`.
    """
    if not cfg.aggregation_nodes_enabled:
        return AggregationResult(0, 0)

    # The configuration object is mutable and provider hooks are arbitrary
    # code.  Build exclusively from a private frozen copy, while every
    # provider/final fence below proves that the caller-visible configuration
    # has not drifted.  This prevents a callback from mixing two clustering,
    # rollup, or publication policies under one config key.
    live_cfg = cfg
    cfg = copy.deepcopy(cfg)

    # Arbitrary/process-scoped embedders have no durable mirror authority: a
    # nonce distinguishes objects but cannot detect hidden route mutation
    # inside the same object.  Aggregation therefore remains fully usable in
    # deterministic entity/exact-blocking mode while treating that producer as
    # disabled for clustering and node embeddings.
    aggregation_embedding_client = embedding_client
    if embedding_client is not None:
        producer_snapshot, _producer_key, _producer_dim = (
            embedding_execution_identity(embedding_client)
        )
        if not producer_snapshot["identity_exact"]:
            aggregation_embedding_client = None

    from hymem.dreaming.aggregation_generation import (
        aggregation_generation_binding_for_contract,
        aggregation_generation_contract,
        register_aggregation_generation,
        validate_aggregation_generation_binding,
    )

    generation_contract = aggregation_generation_contract(cfg)
    resolved_generation = aggregation_generation_binding_for_contract(
        generation_contract, llm,
    )
    if generation_binding is not None:
        supplied_generation = validate_aggregation_generation_binding(
            generation_binding
        )
        if supplied_generation["contract"] != generation_contract:
            raise RuntimeError("aggregation executable contract changed before build")
        if supplied_generation != resolved_generation:
            raise RuntimeError("aggregation producer identity changed before build")
        resolved_generation = supplied_generation
    generation_key = str(resolved_generation["generation_key"])
    config_version = aggregation_config_version(cfg)

    # Withdraw before capture and ensure even a direct low-level caller owns a
    # durable attempt.  Capture binds its material epoch to this row in the
    # same transaction, so source invalidators have a live revision fence for
    # the entire provider/candidate window (the runner normally creates it).
    if not isinstance(health_managed, bool):
        raise ValueError("health_managed must be a boolean")
    owned_pending_attempt = not health_managed
    attempt_token: int
    with core_db.transaction(conn):
        pending = conn.execute(
            "SELECT pending_config_version,pending_generation_key "
            "FROM aggregation_build_health WHERE id=1"
        ).fetchone()
        if health_managed:
            if health_attempt_token is None:
                raise ValueError(
                    "managed aggregation build requires an attempt token"
                )
            from hymem.dreaming.aggregation_health import (
                require_pending_aggregation_attempt,
            )
            if (
                pending is None
                or pending["pending_config_version"] != config_version
                or pending["pending_generation_key"] != generation_key
            ):
                raise RuntimeError(
                    "aggregation build has no matching managed health attempt"
                )
            attempt_token = health_attempt_token
            require_pending_aggregation_attempt(
                conn, config_version, generation_key, attempt_token,
            )
            conn.execute("DELETE FROM aggregation_publication_state")
        else:
            from hymem.dreaming.aggregation_health import begin_aggregation_build

            if health_attempt_token is not None:
                raise ValueError(
                    "direct aggregation build cannot borrow an attempt token"
                )

            # A direct call owns its entire health lifecycle, including a
            # retry after a contained failure. Starting a fresh attempt resets
            # the prior material key before capture rather than inferring
            # ownership from a stale same-generation pending row.
            attempt_token = begin_aggregation_build(
                conn, config_version, generation_binding=resolved_generation,
            )
        if _attempt_token_sink is not None:
            _attempt_token_sink.append(attempt_token)

    captured = capture_aggregation_material(
        conn, cfg, aggregation_embedding_client,
        episode_ceiling_rowid=episode_ceiling_rowid,
        pending_generation_key=generation_key,
        pending_attempt_token=attempt_token,
    )
    material_binding = validate_aggregation_material_binding(captured.binding)
    if material_binding["config_version"] != aggregation_config_version(cfg):
        raise RuntimeError("aggregation material config changed before build")
    verify_aggregation_material_epoch(
        conn, material_binding, embedding_client=aggregation_embedding_client,
        anchor_cap=(
            cfg.aggregation_digest_anchor_facts
            if cfg.aggregation_digest_enabled else 0
        ),
    )
    material_epoch_key = str(material_binding["material_epoch_key"])
    if aggregation_config_version(live_cfg) != config_version:
        raise RuntimeError("aggregation material config changed before build")

    def verify_generation() -> None:
        if aggregation_config_version(live_cfg) != config_version:
            raise RuntimeError("aggregation material config changed during build")
        if aggregation_generation_binding_for_contract(
            generation_contract, llm,
        ) != resolved_generation:
            raise RuntimeError("aggregation producer identity changed during build")
        verify_aggregation_material_epoch(
            conn, material_binding,
            embedding_client=aggregation_embedding_client,
            anchor_cap=(
                cfg.aggregation_digest_anchor_facts
                if cfg.aggregation_digest_enabled else 0
            ),
        )
        from hymem.dreaming.aggregation_health import (
            require_pending_aggregation_attempt,
        )
        require_pending_aggregation_attempt(
            conn, config_version, generation_key, attempt_token,
            material_epoch_key=material_epoch_key,
        )

    # The runner normally withdraws publication together with its durable
    # pending marker. Keep the lower-level build API equally fail closed for
    # direct callers: all old rows remain available to the exact physical proof
    # loader for cache reuse, but consumers see no tree until the candidate's
    # final publication statement commits.
    with core_db.transaction(conn):
        from hymem.dreaming.aggregation_health import (
            require_pending_aggregation_attempt,
        )
        require_pending_aggregation_attempt(
            conn, config_version, generation_key, attempt_token,
            material_epoch_key=material_epoch_key,
        )
        register_aggregation_generation(conn, resolved_generation)
        conn.execute("DELETE FROM aggregation_publication_state")
    episodes = list(captured.episodes)
    clusters = select_clusters(
        episodes, cfg, candidate_pairs=captured.candidate_pairs,
    )

    # Attribution: which candidate generator clustered this dream. Node ids
    # are a function of membership, and membership can differ between the KNN
    # and the exact path (the cosine arm of blocking is approximate) — so two
    # trigger paths with different environments (one missing sqlite-vec)
    # silently alternate between two self-consistent trees, re-keying on every
    # switch. Persisting the mode makes that alternation visible in dream_runs.
    blocking = f"{captured.blocking['mode']}:{captured.blocking['reason']}"
    vectorless = sum(1 for e in episodes if not e["vector"])

    # The content-hash node id makes the previous fusion reusable: an unchanged
    # member set keeps its title/summary without a new LLM call, so a dream over
    # a mostly-stable store rebuilds the whole tree only paying for memberships
    # that actually changed. (The embedding was already cache-keyed; this
    # extends the same discipline to the much more expensive fusion call.)
    existing: dict[str, dict] = {}
    # (level, member set, effective-input fingerprint) of every node in the
    # PREVIOUS tree. The structural
    # predictor below asks a different question from the fusion cache: the
    # cache asks "does this id exist", this asks "did a node with this exact
    # membership exist at this level". The two answers diverge exactly when id
    # keying is broken — a salt change, an unstable hash, a rowid/shadow
    # desync — which is the failure class the reuse watch keeps hitting.
    prev_inputs: set[
        tuple[int, frozenset[str], str | None, str | None, str | None]
    ] = set()
    for row in conn.execute("SELECT * FROM aggregation_nodes"):
        existing[row["id"]] = dict(row)
        try:
            members = json.loads(row["member_episode_ids"])
        except (TypeError, ValueError):
            continue
        prev_inputs.add(
            (
                row["level"], frozenset(members), row["input_fingerprint"],
                row["aggregation_generation_key"],
                row["aggregation_request_hash"],
            )
        )

    rows: list[dict] = []
    items: list[dict] = []          # hierarchy frontier: level-0 nodes first
    clustered_ids: set[str] = set()
    reused = 0
    failures = 0
    level0_missed = 0               # instrumentation: level-0 re-keys this dream
    for members in clusters:
        check_current_deadline()
        member_ids = [m["id"] for m in members]
        inputs = tuple(m["cluster_input_proof"] for m in members)
        input_fingerprint = aggregation_typed_input_fingerprint(inputs)
        request = aggregation_llm_request(inputs, node_kind="cluster")
        user_prompt = request.user
        request_hash = aggregation_llm_request_hash(request)
        node_id = aggregation_node_id(
            member_ids, node_kind="cluster",
            input_fingerprint=input_fingerprint,
            generation_key=generation_key, request_hash=request_hash,
        )
        cached = existing.get(node_id)
        fused = _reusable_fusion(
            conn, node_id, cached, expected_inputs=inputs,
            expected_config_version=config_version,
            expected_generation_key=generation_key,
            expected_request_hash=request_hash,
        )
        level0_reused = fused is not None
        if fused is not None:
            reused += 1
        else:
            level0_missed += 1
            fused = _summarize_cluster(
                members, cfg, llm, verify_generation=verify_generation,
                user_prompt=user_prompt, prepared_request=request,
            )
            if fused is None:
                # CONTAINMENT: the members still count as clustered so they do
                # NOT leak into the digest leftovers. Before this, one failed
                # fusion pushed its members into the leftover pool, which
                # resampled the pass-through leaves, re-keyed the rollup chain,
                # and fed the members' raw text into rollup prompts — if the
                # content itself tripped the failure, it cascaded to the root
                # (repro 2026-07-12: one poisoned cluster → built 46 → 19 and
                # a vanished digest). Now the tree just misses this one node
                # for a dream; the unchanged node id retries next dream.
                failures += 1
                clustered_ids.update(member_ids)
                continue
        row = _candidate_node_row(
            inputs=inputs, fused=fused, node_kind="cluster", level=0,
            config_version=config_version, generation_key=generation_key,
            request_hash=request_hash, material_epoch_key=material_epoch_key,
            reused=level0_reused,
        )
        rows.append(row)
        clustered_ids.update(member_ids)
        items.append(_node_frontier_item(
            row, vector=_centroid([m["vector"] for m in members]),
            entities=set().union(*(m["entities"] for m in members)),
        ))

    root_failed = False
    leaf_changed = -1               # instrumentation: -1 when digest disabled
    leaf_fingerprint: str | None = None      # None => nothing to advance
    leaf_count = 0
    leaf_set: frozenset[str] = frozenset()
    leaf_added: int | None = None            # v34: NULL until a predecessor
    leaf_removed: int | None = None          # id list exists to diff against
    if cfg.aggregation_digest_enabled:
        # Digest leaves = the level-0 nodes plus every episode no cluster
        # absorbed (capped by a churn-stable hash-rank sample), so the root
        # covers the WHOLE store — full time span, not just recent threads.
        leftovers = [e for e in episodes if e["id"] not in clustered_ids]
        leftovers = _stable_sample(leftovers, cfg.aggregation_digest_max_leaves)
        # Instrumentation for the leftover-displacement channel: whether the
        # selected leaf set moved since the last dream that PERSISTED
        # aggregation (the tunable aggregation_digest_max_leaves re-keys the
        # root's level-1 parent when a hash-rank crosses the cap line).
        #
        # The watermark lives in the store (v30), not in a module global. It
        # used to be process-local, which meant the first dream of every
        # process wrote NULL — and the box starts a fresh process per dream, so
        # 175 of 187 rows were unreadable and the channel could not be measured
        # at all. Reading it from the store makes the comparison survive the
        # restart; NULL now means "no dream has ever aggregated this store".
        leaf_set = frozenset(e["id"] for e in leftovers)
        leaf_fingerprint = _leaf_fingerprint(leaf_set)
        leaf_count = len(leaf_set)
        previous_fingerprint = _read_leaf_fingerprint(conn)
        previous_leaf_ids = _read_leaf_ids(conn)
        if previous_leaf_ids is not None:
            # v34: the SIZE of the shift, which the binary flag cannot carry.
            # Computed from a set already in memory against the watermark the
            # store already keeps — no extra pass, no new query, no threshold.
            leaf_added = len(leaf_set - previous_leaf_ids)
            leaf_removed = len(previous_leaf_ids - leaf_set)
        if previous_fingerprint is None:
            # No predecessor to compare against. Report unattributed (NULL),
            # not a counterfeit 0 — leaf_changed=0 is part of the fixed-point
            # signature, and 1162's 8 rebuilds at "leaf_changed=0" were this
            # artifact, not a model violation.
            leaf_changed = None
        else:
            leaf_changed = int(leaf_fingerprint != previous_fingerprint)
        items += [{
            "id": e["id"], "title": e["title"] or "", "summary": e["summary"] or "",
            "_aggregation_cluster_key": f"episode:{e['id']}",
            "_aggregation_level": 0,
            "vector": e["vector"], "entities": e["entities"],
            "session_ids": {e["session_id"]},
            "source_occurrences": e["source_occurrences"],
            "source_provenance_complete": e["source_provenance_complete"],
            "source_manifest_hash": e["source_manifest_hash"],
            "input_proof": e["plain_input_proof"],
        } for e in leftovers]
        anchor_inputs = list(captured.anchor_inputs)
        digest_rows, digest_reused, digest_failures, root_failed = _build_digest_levels(
            conn, items, cfg, llm, existing,
            anchor_inputs=anchor_inputs, config_version=config_version,
            generation_key=generation_key,
            material_epoch_key=material_epoch_key,
            verify_generation=verify_generation,
        )
        rows += digest_rows
        reused += digest_reused
        failures += digest_failures

    if failures:
        # A contained child/root failure is still an incomplete build. Keep the
        # prior physical rows unchanged, but leave publication withdrawn; the
        # failed candidate is never persisted and the watermark is not
        # advanced.
        log.warning(
            "aggregate.unpublished failures=%d root_failed=%s",
            failures, root_failed,
        )
    else:
        roots = [r["id"] for r in rows if r["node_kind"] == "root"]
        if len(roots) > 1:
            raise RuntimeError("aggregation candidate has multiple roots")
        root_id = roots[0] if roots else None
        node_set_hash = aggregation_publication_node_set_hash(rows)

        # Provider work must never hold SQLite's replacement write lock. The
        # candidate is already exact in memory; a provider/cache failure leaves
        # all prior physical rows untouched while the earlier publication fence
        # keeps them unserved. The short transaction below revalidates every
        # typed proof before making this prepared batch authoritative.
        pending_node_embeddings = (
            _prepare_candidate_node_embeddings(
                conn, rows, aggregation_embedding_client,
                verify_material=verify_generation,
            )
            if aggregation_embedding_client is not None and rows else None
        )
        candidate_embedding_rows = [] if pending_node_embeddings is None else [
            {
                "node_id": node_id,
                "vector_json": encode_vector(vector),
                "model": pending_node_embeddings.model,
                "dim": pending_node_embeddings.dim,
                "text_hash": text_hash,
                "embedding_producer_key": pending_node_embeddings.model,
            }
            for node_id, vector, text_hash in zip(
                pending_node_embeddings.node_ids,
                pending_node_embeddings.vectors,
                pending_node_embeddings.text_hashes,
            )
        ]
        node_embedding_count = len(candidate_embedding_rows)
        node_embedding_set_hash = aggregation_node_embedding_set_hash(
            candidate_embedding_rows
        )
        verify_generation()
        published_at = str(conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0])
        if not aggregation_publication_timestamp_is_canonical(published_at):
            raise RuntimeError("SQLite returned a non-canonical publication clock")
        publication_id = aggregation_publication_id(
            config_version=config_version, root_id=root_id,
            node_count=len(rows), node_set_hash=node_set_hash,
            published_at=published_at,
            cluster_min_members=cfg.aggregation_min_members,
            cluster_min_sessions=cfg.aggregation_min_sessions,
            anchor_fact_cap=cfg.aggregation_digest_anchor_facts,
            generation_key=generation_key,
            material_epoch_key=material_epoch_key,
            material_revision=int(material_binding["material_revision"]),
            node_embedding_count=node_embedding_count,
            node_embedding_set_hash=node_embedding_set_hash,
            request_contract_sha256=str(
                resolved_generation["contract"]["request_policy_sha256"]
            ),
        )
        for row in rows:
            row["publication_id"] = publication_id

        # Candidate replacement, optional embedding provider work, proof
        # validation, and publication are one atomic write unit. Provider work
        # was completed above; a later proof mismatch rolls this replacement
        # DELETE back, retaining the prior physical tree for audit/recovery.
        verify_generation()
        with core_db.transaction(conn):
            from hymem.dreaming.aggregation_health import (
                require_pending_aggregation_attempt,
            )
            require_pending_aggregation_attempt(
                conn, config_version, generation_key, attempt_token,
                material_epoch_key=material_epoch_key,
            )
            conn.execute("DELETE FROM aggregation_publication_state")
            conn.execute("DELETE FROM aggregation_nodes")
            for r in rows:
                check_current_deadline()
                conn.execute(
                    """
                    INSERT INTO aggregation_nodes(
                        id,title,summary,member_episode_ids,session_ids,
                        n_members,n_sessions,level,is_root,input_fingerprint,
                        node_kind,output_hash,publication_id,build_config_version,
                        aggregation_generation_key,aggregation_request_hash
                        ,aggregation_material_epoch_key
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        r["id"], r["title"], r["summary"],
                        r["member_episode_ids"], r["session_ids"],
                        r["n_members"], r["n_sessions"], r["level"],
                        r["is_root"], r["input_fingerprint"], r["node_kind"],
                        r["output_hash"], publication_id, config_version,
                        generation_key,
                        r["aggregation_request_hash"],
                        material_epoch_key,
                    ),
                )
                persist_aggregation_source_manifest(
                    conn, r["id"], inputs=r["input_proofs"],
                    node_kind=r["node_kind"], publication_id=publication_id,
                    build_config_version=config_version,
                    aggregation_generation_key=generation_key,
                    aggregation_material_epoch_key=material_epoch_key,
                )
            if leaf_fingerprint is not None:
                _write_leaf_fingerprint(
                    conn, leaf_fingerprint, leaf_count, leaf_set
                )
            if pending_node_embeddings is not None:
                embedded = persist_node_embeddings(
                    conn, pending_node_embeddings,
                    publication_id=publication_id,
                    material_epoch_key=material_epoch_key,
                )
                if embedded != len(pending_node_embeddings.node_ids):
                    raise RuntimeError(
                        "aggregation embedding proof changed before persist"
                    )

            # Publication is the final statement, after every proof and
            # optional embedding provider step succeeded.
            verify_generation()
            for r in rows:
                proof = load_aggregation_node_proof(
                    conn, r["id"], expected_generation_key=generation_key,
                    expected_material_epoch_key=material_epoch_key,
                )
                if proof is None or proof.row["publication_id"] != publication_id:
                    raise RuntimeError("aggregation candidate proof failed")
            stored_rows = [dict(row) for row in conn.execute(
                "SELECT * FROM aggregation_nodes WHERE publication_id=? ORDER BY id",
                (publication_id,),
            ).fetchall()]
            if (
                len(stored_rows) != len(rows)
                or aggregation_publication_node_set_hash(stored_rows) != node_set_hash
            ):
                raise RuntimeError("aggregation candidate set changed before publication")
            stored_embedding_rows = [dict(row) for row in conn.execute(
                "SELECT e.* FROM aggregation_node_embeddings e "
                "JOIN aggregation_nodes n ON n.id=e.node_id "
                "WHERE n.publication_id=? ORDER BY e.node_id",
                (publication_id,),
            ).fetchall()]
            if (
                len(stored_embedding_rows) != node_embedding_count
                or aggregation_node_embedding_set_hash(stored_embedding_rows)
                != node_embedding_set_hash
            ):
                raise RuntimeError("aggregation node embedding set changed before publication")
            verify_generation()
            conn.execute(
                "INSERT INTO aggregation_publication_state("
                "id,publication_id,config_version,cluster_min_members,"
                "cluster_min_sessions,anchor_fact_cap,root_node_id,node_count,"
                "node_set_hash,published_at,aggregation_generation_key,"
                "request_contract_sha256,aggregation_material_epoch_key,"
                "material_revision,node_embedding_count,node_embedding_set_hash) "
                "VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    publication_id, config_version, cfg.aggregation_min_members,
                    cfg.aggregation_min_sessions,
                    cfg.aggregation_digest_anchor_facts, root_id,
                    len(rows), node_set_hash, published_at,
                    generation_key,
                    resolved_generation["contract"]["request_policy_sha256"],
                    material_epoch_key,
                    material_binding["material_revision"],
                    node_embedding_count,
                    node_embedding_set_hash,
                ),
            )

    forecast = _forecast_rebuild(rows, prev_inputs)
    log.info(
        "aggregate.built nodes=%d reused=%d failures=%d blocking=%s "
        "vectorless=%d level0_missed=%d leaf_changed=%d (from %d episodes)",
        len(rows), reused, failures, blocking, vectorless,
        level0_missed, -1 if leaf_changed is None else leaf_changed, len(episodes),
    )
    log.info(
        "aggregate.forecast predicted=%d actual=%d residual=%d facts_rekey=%d "
        "rebuilt_level0=%d rebuilt_rollup=%d rebuilt_root=%d",
        forecast.predicted, forecast.actual, forecast.residual, forecast.facts_rekey,
        forecast.rebuilt_level0, forecast.rebuilt_rollup, forecast.rebuilt_root,
    )
    if forecast.residual > 0:
        # Membership-identical nodes that did not reuse. Nothing in the build
        # explains this; it is the signature of an id-keying defect.
        log.warning(
            "aggregate.keying_residual nodes=%d — %d node(s) kept their exact "
            "membership and still missed the fusion cache",
            forecast.residual, forecast.residual,
        )
    leaf_res: int | None = leaf_changed
    if leaf_res is not None and leaf_res < 0:
        leaf_res = None
    if leaf_added is not None and leaf_removed is not None:
        log.info(
            "aggregate.leafdelta added=%d removed=%d net=%d",
            leaf_added, leaf_removed, leaf_added - leaf_removed,
        )
        # v34 self-check, identity (1): the v29 flag and the v34 counts are two
        # independent routes to the same comparison — hash equality vs set
        # difference. They cannot disagree unless one is broken. Logged rather
        # than raised: an instrument that aborts the dream it is measuring
        # costs more than the reading is worth, and a silent disagreement is
        # the failure mode this channel exists to make visible.
        moved = int(leaf_added + leaf_removed > 0)
        if leaf_res is not None and moved != leaf_res:
            log.warning(
                "aggregate.leafdelta_disagreement leaf_changed=%d but "
                "added+removed=%d — the fingerprint and set-difference routes "
                "disagree; one of them is broken",
                leaf_res, leaf_added + leaf_removed,
            )
    result = AggregationResult(
        len(rows), reused, failures, len(episodes), blocking,
        level0_missed, leaf_res,
        forecast.predicted, forecast.residual, forecast.facts_rekey,
        forecast.rebuilt_level0, forecast.rebuilt_rollup, forecast.rebuilt_root,
        leaf_added, leaf_removed, material_epoch_key,
    )
    if owned_pending_attempt:
        from hymem.dreaming.aggregation_health import (
            complete_aggregation_build,
            record_aggregation_build_failure,
        )

        if failures:
            from hymem.dreaming.aggregation_health import (
                require_pending_aggregation_attempt,
            )
            require_pending_aggregation_attempt(
                conn, config_version, generation_key, attempt_token,
                material_epoch_key=material_epoch_key,
            )
            record_aggregation_build_failure(
                conn, config_version, generation_key, attempt_token,
                fusion_failures=failures,
                material_epoch_key=material_epoch_key,
            )
        else:
            complete_aggregation_build(
                conn, config_version, generation_key, attempt_token,
                expected_node_count=len(rows),
                material_epoch_key=material_epoch_key,
                embedding_client=aggregation_embedding_client,
            )
    return result


def _build_digest_levels(
    conn: sqlite3.Connection,
    items: list[dict], cfg: HyMemConfig, llm: LLMClient,
    existing: dict[str, dict], *,
    anchor_inputs: list[AggregationInputProof], config_version: str,
    generation_key: str,
    material_epoch_key: str,
    verify_generation: Callable[[], None] | None = None,
) -> tuple[list[dict], int, int, bool]:
    """RAPTOR rollup: recursively cluster-and-fuse the frontier `items` (each
    {"id","title","summary","vector","entities","session_ids"}) until at most
    `aggregation_max_members` remain, then fuse those into the single ROOT
    digest node. Returns (node rows for every level >= 1, reused-fusion count,
    failed-fusion count, root_failed).

    Each pass clusters with the SAME `_linked` rule as level 0 (centroid
    vectors, union entity sets); when nothing links — disjoint topics, exactly
    the case a digest must still cover — it falls back to content-defined
    groups of ~`fan_in` items (`_content_defined_groups`), which guarantees the
    loop converges and keeps group boundaries stable under frontier churn
    (positional `items[i:i+fan_in]` slabs shifted every downstream group when
    one item appeared or vanished). A failed fusion DROPS its group for this
    dream (counted, retried next dream at the same node id); the previous
    pass-through of raw members reshaped every level above AND propagated the
    very content that failed into the parent prompts — one poisoned cluster
    took out the whole chain to the root. If a whole pass makes no progress
    the loop bails out and the root fuses whatever frontier remains (capped
    inside `_items_text`).

    The root fusion is GROUNDED: `anchor_facts` (top knowledge-graph edges)
    render as a VERIFIED FACTS block the digest prompt treats as ground truth
    over the machine-generated summaries, and the block's hash joins the root's
    cache id so a changed graph regenerates the digest. `root_failed` tells the
    caller which fusion failed for diagnostics; every incomplete candidate
    remains unpublished while prior physical rows stay available for audit and
    exact cache reuse."""
    rows: list[dict] = []
    reused = 0
    failures = 0
    fan_in = max(2, cfg.aggregation_max_members)
    while len(items) > fan_in:
        check_current_deadline()
        # Same chaining guard as level 0: a transitive mega-component among the
        # rollup frontier would otherwise fuse from a `aggregation_max_members`
        # truncation of itself, silently dropping every thread past the cut.
        # Deliberately EXACT all-pairs (no candidate blocking): the frontier is
        # a few dozen items at most, and rollup items aren't in vec_episodes.
        labels = cluster_episodes(
            items, cfg.aggregation_emb_threshold, cfg.aggregation_ent_threshold,
            max_cluster_size=cfg.aggregation_max_cluster_size or None,
        )
        grouped: dict[int, list[dict]] = {}
        for it in items:
            cluster_key = it.get("_aggregation_cluster_key", it["id"])
            grouped.setdefault(labels[cluster_key], []).append(it)
        # First-seen order (dict insertion follows `items` order): stable under
        # membership churn. Sorting by size reordered the whole level whenever
        # any group's size changed, re-keying unrelated parents downstream.
        groups = list(grouped.values())
        if all(len(g) < 2 for g in groups):
            groups = _content_defined_groups(items, fan_in)
            if all(len(g) < 2 for g in groups) and len(items) > fan_in:
                # With a tiny fan-in every item can itself be a CDC cut.  That
                # shape makes no progress and would leave an over-budget root
                # frontier. Deterministically pair consecutive stable-order
                # items as the terminating fallback; membership remains exact.
                groups = [
                    items[index:index + fan_in]
                    for index in range(0, len(items), fan_in)
                ]
        else:
            # A similarity component can be larger than the fusion fan-in even
            # when the cluster chaining guard is enabled. Partition before the
            # prompt is rendered so persisted membership never overclaims items
            # that `_items_text` silently omitted.
            groups = [
                bounded
                for group in groups
                for bounded in (
                    _content_defined_groups(group, fan_in)
                    if len(group) > fan_in else [group]
                )
            ]

        next_items: list[dict] = []
        for g in groups:
            check_current_deadline()
            if len(g) < 2:
                next_items.append(g[0])
                continue
            member_ids = [m["id"] for m in g]
            inputs = tuple(m["input_proof"] for m in g)
            input_fingerprint = aggregation_typed_input_fingerprint(inputs)
            request = aggregation_llm_request(inputs, node_kind="rollup")
            user_prompt = request.user
            request_hash = aggregation_llm_request_hash(request)
            node_id = aggregation_node_id(
                member_ids, node_kind="rollup",
                input_fingerprint=input_fingerprint,
                generation_key=generation_key, request_hash=request_hash,
            )
            cached = existing.get(node_id)
            fused = _reusable_fusion(
                conn, node_id, cached, expected_inputs=inputs,
                expected_config_version=config_version,
                expected_generation_key=generation_key,
                expected_request_hash=request_hash,
            )
            rollup_reused = fused is not None
            if fused is not None:
                reused += 1
            else:
                # ROLLUP, not AGGREGATE: a rollup group (especially a forced
                # chunk) can hold UNRELATED threads, and the thread-fusion
                # prompt would narrow to the dominant one — a thread dropped
                # here is gone from every level above, which is exactly how a
                # whole-store digest degrades into a recap of one topic.
                fused = _llm_fuse(
                    user_prompt,
                    llm, system=ROLLUP_SYSTEM, kind="rollup",
                    verify_generation=verify_generation,
                    prepared_request=request,
                )
                if fused is None:
                    # CONTAINMENT (see the level-0 twin): the group sits this
                    # dream out rather than leaking raw members upward.
                    failures += 1
                    continue
            row = _candidate_node_row(
                inputs=inputs, fused=fused, node_kind="rollup",
                level=max(int(m.get("_aggregation_level", 0)) for m in g) + 1,
                config_version=config_version, generation_key=generation_key,
                request_hash=request_hash,
                material_epoch_key=material_epoch_key,
                reused=rollup_reused,
            )
            rows.append(row)
            next_items.append(_node_frontier_item(
                row, vector=_centroid([m["vector"] for m in g]),
                entities=set().union(*(m["entities"] for m in g)),
            ))
        if len(next_items) >= len(items):    # no progress (fusions all failed)
            items = next_items
            break
        items = next_items

    if not items:
        return rows, reused, failures, False
    member_ids = [m["id"] for m in items]
    # Every effective fact is now a typed authoritative input with its own
    # exact occurrence manifest. Invalid anchors were excluded before prompt
    # construction and therefore cannot become unsourced model material.
    facts_block = (
        "\n".join(f"- {item.rendered_text}" for item in anchor_inputs)
        if anchor_inputs else "(none)"
    )
    inputs = tuple(
        [item["input_proof"] for item in items] + list(anchor_inputs)
    )
    input_fingerprint = aggregation_typed_input_fingerprint(inputs)
    request = aggregation_llm_request(inputs, node_kind="root")
    user_prompt = request.user
    request_hash = aggregation_llm_request_hash(request)
    root_id = aggregation_node_id(
        member_ids, node_kind="root", input_fingerprint=input_fingerprint,
        generation_key=generation_key, request_hash=request_hash,
    )
    cached = existing.get(root_id)
    fused = _reusable_fusion(
        conn, root_id, cached, expected_inputs=inputs,
        expected_config_version=config_version,
        expected_generation_key=generation_key,
        expected_request_hash=request_hash,
    )
    root_reused = fused is not None
    if fused is not None:
        reused += 1
    else:
        fused = _llm_fuse(
            user_prompt,
            llm, system=DIGEST_SYSTEM, kind="root",
            verify_generation=verify_generation,
            prepared_request=request,
        )
    if fused is None:
        return rows, reused, failures + 1, True
    rows.append(_candidate_node_row(
        inputs=inputs, fused=fused, node_kind="root",
        level=max(int(item.get("_aggregation_level", 0)) for item in items) + 1,
        config_version=config_version, generation_key=generation_key,
        request_hash=request_hash, material_epoch_key=material_epoch_key,
        reused=root_reused,
    ))
    return rows, reused, failures, False


@dataclass
class Digest:
    """The root of the RAPTOR tree — the standing whole-store summary that
    `HyMem.digest()` exposes for system-prompt-style injection.
    `n_sessions`/`n_sessions_total` say how much of the store's history the
    digest actually condenses — a low ratio means many sessions never produced
    episodes (a dream-coverage gap upstream of the digest, not a tree problem).
    `generated_at` is the build time."""

    title: str
    summary: str
    n_sessions: int
    n_sessions_total: int
    generated_at: str
    node_id: str = ""
    """The root aggregation node's id — the entry point for the Stage-4b
    drill-down: pass it to `HyMem.expand_node()` to see which child nodes and
    episodes the digest was fused from ("why does my digest say X?")."""

    def as_context_block(self) -> str:
        """The canonical system-prompt rendering: title, summary, and one
        provenance footer. Every delivery surface (embedded host injection,
        the MCP `hymem_digest` tool, the Honcho peer representation) uses this
        so the staleness display is decided in exactly one place: the footer
        names the coverage ratio and the build time, because a digest is a
        dream-time artifact — the reader must be able to see "this reflects
        the store as of <generated_at>", not mistake it for live state."""
        footer = f"(Memory digest covering {self.n_sessions} of {self.n_sessions_total} sessions"
        if self.generated_at:
            footer += f"; generated {self.generated_at}"
        footer += ".)"
        return f"## {self.title}\n\n{self.summary}\n\n{footer}"


def load_digest(
    conn: sqlite3.Connection, *, expected_config_version: str | None = None,
    expected_cluster_min_members: int | None = None,
    expected_cluster_min_sessions: int | None = None,
    expected_anchor_fact_cap: int | None = None,
    expected_generation_key: str | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> Digest | None:
    """Return the current root digest, or None when the aggregation layer is
    disabled, has not dreamed yet, or the store has no episodes. Read-only."""
    owned_snapshot = not conn.in_transaction
    try:
        if owned_snapshot:
            conn.execute("BEGIN")
        publication = load_current_aggregation_publication(
            conn, expected_config_version=expected_config_version,
            expected_cluster_min_members=expected_cluster_min_members,
            expected_cluster_min_sessions=expected_cluster_min_sessions,
            expected_anchor_fact_cap=expected_anchor_fact_cap,
            expected_generation_key=expected_generation_key,
            embedding_client=embedding_client,
        )
        if publication is None or publication.root_node_id is None:
            result = None
        else:
            proof = publication.nodes.get(publication.root_node_id)
            if proof is None:
                result = None
            else:
                row = proof.row
                total = conn.execute(
                    "SELECT COUNT(*) AS c FROM sessions"
                ).fetchone()["c"]
                result = Digest(
                    title=row["title"], summary=row["summary"],
                    n_sessions=row["n_sessions"], n_sessions_total=int(total),
                    generated_at=publication.published_at, node_id=row["id"],
                )
        if owned_snapshot:
            conn.execute("COMMIT")
            if result is not None:
                current = load_current_aggregation_publication(
                    conn, expected_config_version=expected_config_version,
                    expected_cluster_min_members=expected_cluster_min_members,
                    expected_cluster_min_sessions=expected_cluster_min_sessions,
                    expected_anchor_fact_cap=expected_anchor_fact_cap,
                    expected_generation_key=expected_generation_key,
                    embedding_client=embedding_client,
                )
                if current is None or current.root_node_id != result.node_id:
                    result = None
        return result
    except (RuntimeError, TypeError, ValueError, sqlite3.Error):
        if owned_snapshot and conn.in_transaction:
            conn.execute("ROLLBACK")
        return None


@dataclass
class NodeChild:
    """A child aggregation node inside a `NodeExpansion` — one level-(N−1)
    fusion the expanded node rolled up. Expand it in turn to keep descending."""

    id: str
    title: str
    summary: str
    level: int
    n_members: int
    n_sessions: int


@dataclass
class NodeSourceOccurrence:
    """Public, lightweight coordinate for one exact aggregation source."""

    message_id: int
    session_id: str
    source_peer_id: str | None
    source_workspace_id: str | None


@dataclass
class NodeMemberEpisode:
    """A leaf inside a `NodeExpansion`: the per-session episode whose summary
    fed the node's fusion. `source_occurrences` / `source_message_ids` are the
    exact ordered proof coordinates. `start_message_id`/`end_message_id` are a
    compatibility envelope derived from those coordinates, never trusted from
    the mutable episode range metadata."""

    id: str
    session_id: str
    title: str
    summary: str
    start_message_id: int
    end_message_id: int
    source_message_ids: tuple[int, ...]
    source_occurrences: tuple[NodeSourceOccurrence, ...]


@dataclass
class NodeExpansion:
    """One step of the RAPTOR tree-traversal read (`HyMem.expand_node()`):
    the node itself plus its members, resolved one level down. Members of a
    level >= 1 node are a mix of child nodes and pass-through episodes (leaves
    no cluster absorbed); level-0 members are episodes only. Member order is
    the persisted fusion-input order. `missing_member_ids` is retained for API
    compatibility and is empty on every successful v55 expansion: a missing,
    malformed, or type-confused member invalidates the publication and the
    expansion fails closed instead of returning a partial tree."""

    id: str
    title: str
    summary: str
    level: int
    is_root: bool
    child_nodes: list[NodeChild]
    episodes: list[NodeMemberEpisode]
    missing_member_ids: list[str]


def expand_node(
    conn: sqlite3.Connection, node_id: str, *,
    expected_config_version: str | None = None,
    expected_cluster_min_members: int | None = None,
    expected_cluster_min_sessions: int | None = None,
    expected_anchor_fact_cap: int | None = None,
    expected_generation_key: str | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> NodeExpansion | None:
    """Resolve an aggregation node's members one level down — the Stage-4b
    drill-down behind "why does my digest say X?". Start from
    `Digest.node_id` (the root) or an `AggregationNodeHit.node_id` from the
    query tier, and recurse through `child_nodes` until everything is
    episodes. Returns None for an unknown id. Read-only; per-member point
    lookups (members are capped at fusion time, so the fan-out is small)."""
    owned_snapshot = not conn.in_transaction
    try:
        if owned_snapshot:
            conn.execute("BEGIN")
        publication = load_current_aggregation_publication(
            conn, expected_config_version=expected_config_version,
            expected_cluster_min_members=expected_cluster_min_members,
            expected_cluster_min_sessions=expected_cluster_min_sessions,
            expected_anchor_fact_cap=expected_anchor_fact_cap,
            expected_generation_key=expected_generation_key,
            embedding_client=embedding_client,
        )
        proof = publication.nodes.get(node_id) if publication is not None else None
        if proof is None:
            result = None
        else:
            row = proof.row
            child_nodes: list[NodeChild] = []
            episodes: list[NodeMemberEpisode] = []
            for item in proof.inputs:
                if item.kind == "aggregation_node":
                    child = publication.nodes.get(str(item.source_ref["id"]))
                    if child is None:
                        result = None
                        break
                    child_nodes.append(NodeChild(
                        id=str(child.row["id"]), title=str(child.row["title"]),
                        summary=str(child.row["summary"]),
                        level=int(child.row["level"]),
                        n_members=int(child.row["n_members"]),
                        n_sessions=int(child.row["n_sessions"]),
                    ))
                elif item.kind == "episode":
                    ep_row = conn.execute(
                        "SELECT e.id,e.session_id,e.title,e.summary FROM episodes e "
                        "JOIN sessions s ON s.id=e.session_id WHERE e.id=? AND "
                        "(e.digest_generation IS NULL OR "
                        "e.digest_generation=s.digest_published_generation)",
                        (item.source_ref["id"],),
                    ).fetchone()
                    if ep_row is None:
                        result = None
                        break
                    source_coordinates = tuple(
                        NodeSourceOccurrence(
                            message_id=source.message_id,
                            session_id=source.session_id,
                            source_peer_id=source.source_peer_id,
                            source_workspace_id=source.source_workspace_id,
                        )
                        for source in item.occurrences
                    )
                    source_ids = tuple(
                        source.message_id for source in source_coordinates
                    )
                    episodes.append(NodeMemberEpisode(
                        id=ep_row["id"], session_id=ep_row["session_id"],
                        title=ep_row["title"], summary=ep_row["summary"],
                        start_message_id=source_ids[0],
                        end_message_id=source_ids[-1],
                        source_message_ids=source_ids,
                        source_occurrences=source_coordinates,
                    ))
            else:
                result = NodeExpansion(
                    id=str(row["id"]), title=str(row["title"]),
                    summary=str(row["summary"]), level=int(row["level"]),
                    is_root=bool(row["is_root"]), child_nodes=child_nodes,
                    episodes=episodes, missing_member_ids=[],
                )
        if owned_snapshot:
            conn.execute("COMMIT")
            if result is not None and publication is not None:
                current = load_current_aggregation_publication(
                    conn,
                    expected_config_version=expected_config_version,
                    expected_cluster_min_members=expected_cluster_min_members,
                    expected_cluster_min_sessions=expected_cluster_min_sessions,
                    expected_anchor_fact_cap=expected_anchor_fact_cap,
                    expected_generation_key=expected_generation_key,
                    expected_material_epoch_key=publication.material_epoch_key,
                    embedding_client=embedding_client,
                )
                if (
                    current is None
                    or current.publication_id != publication.publication_id
                    or node_id not in current.nodes
                ):
                    result = None
        return result
    except (RuntimeError, TypeError, ValueError, sqlite3.Error):
        if owned_snapshot and conn.in_transaction:
            conn.execute("ROLLBACK")
        return None
