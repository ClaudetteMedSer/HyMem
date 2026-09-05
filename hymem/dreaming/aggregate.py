"""Phase-2 RAPTOR cross-session aggregation nodes.

Dreaming already produces per-session *episodes*. The multi-session residual on
LongMemEval is a *synthesis* problem: the gold turns reach the answer context,
but the reader fails to fuse facts scattered one-per-session across ~45 raw
slots. This module closes that upstream — it clusters episodes ACROSS sessions
(connected components over embedding-OR-entity overlap) and fuses each
cross-session cluster into a single `aggregation_nodes` summary, so a synthesis
question can be answered from a handful of cluster summaries instead of dozens
of raw turns.

The whole layer is additive and off by default (`cfg.aggregation_nodes_enabled`):
when disabled, `build_aggregation_nodes` is never called and query-time behavior
is unchanged. The build was front-run gated by an offline co-location probe
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
from dataclasses import dataclass
from typing import NamedTuple
from collections.abc import Callable

from hymem.config import HyMemConfig
from hymem.core.graph import graph_clock_order_sql, live_edge_predicate
from hymem.core import db as core_db
from hymem.core.vectors import decode_vector, encode_vector
from hymem.dreaming.aggregation_provenance import (
    AGGREGATION_SOURCE_MANIFEST_VERSION,
    aggregation_input_fingerprint,
    combine_source_occurrences,
    load_aggregation_source_manifest,
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
from hymem.dreaming.user_profile import load_profile, render_profile_fact
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

# Fusion-prompt versions, baked into the node-id salt of the level the prompt
# serves. Reuse is keyed by node id, so bumping a version when its prompt
# changes materially makes every cached fusion of that kind regenerate on the
# next dream — without touching the other levels' caches. This matters beyond
# style: a hallucination CRYSTALLIZES in a cached fusion (the "Acme Corp"
# incident lived in a persisted rollup and survived a root-only fix), so a
# prompt hardened against an artifact must invalidate the level that produced
# it, or the artifact outlives the fix.
_CLUSTER_SALT = "cluster.v4"  # v4: content-defined window cuts (positional
                              #     windows re-keyed the whole component on any
                              #     mid-order membership change — 2026-07-12
                              #     reuse instability); v3: recency-window split
                              #     at max_cluster_size; v2: identity evidence-bound
_ROLLUP_SALT = "rollup.v3"    # v3: content-defined fallback grouping (same
                              #     2026-07-12 fix); v2: identity evidence-bound
_ROOT_SALT = "root.v4"        # v4: VERIFIED FACTS anchor block


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
    parent: dict[str, str] = {e["id"]: e["id"] for e in episodes}

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
            for j in range(i + 1, len(episodes)):
                if _linked(episodes[i], episodes[j], emb_threshold, ent_threshold):
                    union(episodes[i]["id"], episodes[j]["id"])
    else:
        by_id = {e["id"]: e for e in episodes}
        for a, b in candidate_pairs:
            if a == b or a not in by_id or b not in by_id:
                continue
            if _linked(by_id[a], by_id[b], emb_threshold, ent_threshold):
                union(a, b)

    roots = {e["id"]: find(e["id"]) for e in episodes}
    label_of: dict[str, int] = {}
    out: dict[str, int] = {}
    for eid, root in roots.items():
        if root not in label_of:
            label_of[root] = len(label_of)
        out[eid] = label_of[root]
    if max_cluster_size is None:
        return out

    # Chaining guard: split every over-cap component into recency windows.
    pos = {e["id"]: i for i, e in enumerate(episodes)}

    def _recency_key(e: dict) -> tuple:
        sm = e.get("start_message_id")
        # Episodes carrying a message id sort by it (global ingestion order);
        # items without one (rollup nodes, plain dicts) keep input order and
        # sort after dated ones. Input position breaks all ties → deterministic.
        return (0, sm, pos[e["id"]]) if isinstance(sm, int) else (1, 0, pos[e["id"]])

    components: dict[int, list[dict]] = {}
    for e in episodes:
        components.setdefault(out[e["id"]], []).append(e)

    capped: dict[str, int] = {}
    next_label = 0
    for label in sorted(components):       # original first-seen label order
        members = components[label]
        if len(members) <= max_cluster_size:
            for m in members:
                capped[m["id"]] = next_label
            next_label += 1
            continue
        ordered = sorted(members, key=_recency_key)   # oldest → newest
        # Content-defined cuts: boundaries belong to member ids, so a
        # membership change anywhere re-cuts only its local window(s).
        for window in _content_defined_groups(ordered, max_cluster_size):
            for m in window:
                capped[m["id"]] = next_label
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
        if len(current) >= max_size or _is_cut_id(item["id"], max_size):
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
    ceiling = "AND e.rowid <= ?" if max_rowid is not None else ""
    rows = conn.execute(
        f"""
        SELECT e.rowid AS rowid, e.id, e.session_id, e.title, e.summary,
               e.start_message_id, e.end_message_id, e.key_entities,
               e.source_manifest_hash, em.vector_json
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
    episodes: list[dict] = []
    for r in rows:
        try:
            raw_entities = json.loads(r["key_entities"] or "[]")
        except (ValueError, TypeError):
            raw_entities = []
        vec = decode_vector(r["vector_json"]) if r["vector_json"] else None
        sources = load_episode_source_manifest(conn, r["id"])
        episodes.append({
            "id": r["id"],
            # episodes.rowid mirrors vec_episodes.rowid (see _backfill_vec_episodes /
            # persist_episode_embeddings), so blocking can translate KNN hits back
            # to episode ids without a per-hit SELECT.
            "rowid": r["rowid"],
            "session_id": r["session_id"],
            "title": r["title"],
            "summary": r["summary"],
            # Recency signal for the max_cluster_size window split: messages.id
            # is a store-wide AUTOINCREMENT, so this orders episodes by
            # ingestion time across sessions (session_id alone is lexicographic
            # and NOT chronological for non-date-prefixed session names).
            "start_message_id": r["start_message_id"],
            "entities": {_norm_entity(x) for x in raw_entities if x},
            "vector": vec,
            "source_occurrences": sources or (),
            "source_provenance_complete": sources is not None,
            "source_manifest_hash": (
                r["source_manifest_hash"] if sources is not None else None
            ),
        })
    return episodes


def generate_candidate_pairs(
    conn: sqlite3.Connection, episodes: list[dict], *, emb_top_k: int,
) -> set[tuple[str, str]] | None:
    """Stage-3b candidate blocking: the pair set `cluster_episodes` should test
    instead of all O(n²) pairs (prod, 2026-06-12: 395 episodes → 77,815 pairs →
    4.04s per dream, past the 2s gate).

    Returns None whenever blocking cannot run exactly as designed, and the
    caller MUST then fall back to exact all-pairs (pass candidate_pairs=None):
      - `emb_top_k <= 0` (config: aggregation_blocking_top_k=0 disables blocking);
      - no `vec_episodes` table / sqlite_vec extension unavailable — sqlite_vec
        is an optional dependency, and embedded small stores without it must
        keep today's exact behavior unchanged.

    Otherwise returns ascending-normalized id pairs from two arms:
      - Entity arm (EXACT): inverted index entity → episode ids over the
        already-normalized `e["entities"]` sets; every co-occurring pair under
        any entity is a candidate. Jaccard >= 0.5 requires >= 1 shared entity,
        so this arm loses nothing.
      - Cosine arm (approximate): per-episode KNN top `emb_top_k` neighbors via
        `core_db.vec_search` over `vec_episodes` (whose rowids mirror
        episodes.rowid). Queried as k+1 so the episode's own row never consumes
        a neighbor slot — hence with emb_top_k >= n-1 the arm is exact and
        small stores lose nothing. Hits whose rowid is not in `episodes`
        (retention may have pruned the row since vec ingest) and self-pairs are
        skipped. Episodes without a vector contribute no cosine candidates —
        exactly `_linked`'s behavior, whose cosine arm never fires for them;
        the entity arm still covers them.
    """
    if emb_top_k <= 0:
        return None
    # Mirrors the augment.py vec path guard: the virtual table can be listed in
    # sqlite_master while the extension fails to load on THIS connection — then
    # every vec_search returns [], which would silently amputate the cosine arm
    # rather than approximate it. Treat that as "cannot run as designed".
    # The declines are logged because the fallback is invisible from results
    # (identical components, just slower) — a bare core_db.connect() without
    # vec initialization once made a hand-timed box run measure the exact path.
    # WARNING, not debug: a decline is invisible from results (identical-ish
    # components, just slower) yet it changes WHICH pairs get tested — a
    # deployment where one trigger path declines and another doesn't alternates
    # between two different clusterings, re-keying cached fusions every switch.
    if not core_db._load_vec_extension(conn):
        log.warning("blocking.decline reason=vec_extension_unavailable (exact all-pairs)")
        return None
    if not core_db.has_vec_table(conn, table="vec_episodes"):
        log.warning("blocking.decline reason=no_vec_episodes_table (exact all-pairs)")
        return None

    pairs: set[tuple[str, str]] = set()

    # Entity arm (exact): invert entity → episode ids, pair all co-occurrences.
    inverted: dict[str, list[str]] = {}
    for e in episodes:
        for ent in e.get("entities") or ():
            inverted.setdefault(ent, []).append(e["id"])
    for ids in inverted.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a != b:
                    pairs.add((a, b) if a < b else (b, a))

    # Cosine arm (approximate): KNN per vectored episode, translated rowid → id.
    id_of_rowid = {
        e["rowid"]: e["id"] for e in episodes if e.get("rowid") is not None
    }
    for e in episodes:
        vec = e.get("vector")
        if not vec:
            continue
        hits = core_db.vec_search(conn, vec, emb_top_k + 1, table="vec_episodes")
        for rowid, _distance in hits:
            other = id_of_rowid.get(rowid)
            if other is None or other == e["id"]:
                continue
            a, b = e["id"], other
            pairs.add((a, b) if a < b else (b, a))
    return pairs


def _node_id(
    member_ids: list[str], *, salt: str = "",
    input_fingerprint: str | None = None,
) -> str:
    """Stable id for a node = content hash of its sorted member ids, so an
    unchanged cluster keeps its id (and cached embedding + fusion) across dream
    cycles. `salt` separates id spaces for nodes that could share a member set
    but carry a different KIND of fusion (the root digest uses a different
    prompt than an intermediate rollup, so they must never reuse each other)."""
    payload = "|".join(sorted(member_ids))
    if salt:
        payload = f"{salt}::{payload}"
    if input_fingerprint is not None:
        payload = f"{payload}::input={input_fingerprint}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"agg_{digest}"


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
        if conn is not None else None
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
_MAX_FUSION_TITLE_CHARS = 300
_MAX_FUSION_SUMMARY_CHARS = 2000

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
    rows: list[dict], prev_inputs: set[tuple]
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

        predicted = nodes whose (level, member set, input fingerprint) is
                    absent from the previous tree
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
    # Accept the historical two-tuple form in pure unit callers.  Persisted
    # predecessor state uses the three-tuple form, which is what distinguishes
    # a legitimate in-place episode rewrite from a cache-keying defect.
    membership_keys = {(item[0], item[1]) for item in prev_inputs}
    input_keys = {
        (item[0], item[1], item[2] if len(item) > 2 else None)
        for item in prev_inputs
    }
    for r in rows:
        membership_key = (r["level"], frozenset(r["member_ids"]))
        membership_is_new = membership_key not in membership_keys
        input_is_new = (*membership_key, r.get("input_fingerprint")) not in input_keys
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
    return min(8192, 2048 + len(prompt) // 2)


def _llm_fuse(
    user_prompt: str, llm: LLMClient, *, system: str, kind: str = "fusion",
    shrink: Callable[[], str] | None = None,
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
    terminating step, `shrink`, reduces the RENDERED input only (fewer chars
    per member), never the member set: node_id = sha1(sorted(member_ids)),
    so a different member set would produce a different node id and re-key
    every ancestor permanently.
    """
    def attempt(prompt: str) -> tuple[dict | None, str | None, str]:
        request = LLMRequest(
            system=system, user=prompt, response_format="json",
            max_tokens=_fusion_max_tokens(prompt),
        )
        try:
            raw = llm.complete(request)
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
        return {"title": title, "summary": summary}, raw, "ok"

    fused, raw, stage = attempt(user_prompt)
    if fused is None and stage == "parse" and raw is not None and is_ceiling_cut(raw):
        # One re-roll of the SAME input — empirical license, see docstring.
        fused, raw, stage = attempt(user_prompt)
    if fused is None and stage == "parse" and shrink is not None:
        # Terminating step: membership-PRESERVING render shrink.
        fused, _, _ = attempt(shrink())
    return fused


def _summarize_cluster(
    members: list[dict], cfg: HyMemConfig, llm: LLMClient
) -> dict | None:
    """Fuse a level-0 cluster's episodes into {title, summary}."""
    def render(scale: float = 1.0) -> str:
        if scale >= 1.0:
            return "\n\n---\n\n".join(
                f"[{m['session_id']}] {m['title']}\n{m['summary']}" for m in members
            )
        return "\n\n---\n\n".join(
            f"[{m['session_id']}] {m['title'][:int(len(m['title']) * scale)]}\n"
            f"{m['summary'][:int(len(m['summary']) * scale)]}" for m in members
        )
    return _llm_fuse(
        AGGREGATE_USER_TEMPLATE.format(text=render()), llm,
        system=AGGREGATE_SYSTEM, kind="cluster",
        shrink=lambda: AGGREGATE_USER_TEMPLATE.format(text=render(0.5)),
    )


def _items_text(items: list[dict], cfg: HyMemConfig, *, char_scale: float = 1.0) -> str:
    """Render hierarchy items (level-0 nodes / rollups / pass-through episodes,
    all carrying title+summary) as one fusion input block.

    `char_scale` shrinks each member's rendered title/summary for the retry
    ladder's terminating step — the MEMBER SET is untouched, so the node's
    content-hash id is unchanged and a fused result stays cache-compatible.
    """
    if len(items) > max(2, cfg.aggregation_max_members):
        raise ValueError("aggregation fusion input exceeds its member budget")
    if char_scale >= 1.0:
        return "\n\n---\n\n".join(f"{m['title']}\n{m['summary']}" for m in items)
    parts = []
    for m in items:
        t = m["title"] or ""
        s = m["summary"] or ""
        parts.append(
            f"{t[:int(len(t) * char_scale)]}\n{s[:int(len(s) * char_scale)]}"
        )
    return "\n\n---\n\n".join(parts)


def _anchor_facts(conn: sqlite3.Connection, cap: int) -> list[str]:
    """ACTIVE typed user-profile rows (schema v18) followed by top ACTIVE,
    non-derived, non-superseded knowledge-graph edges, rendered as one-line
    facts — the VERIFIED FACTS block grounding the root digest fusion.

    Profile rows lead the block: they hold exactly the durable identity facts
    (name, role, employer, location, ...) the tech-domain graph vocabulary can
    never mint — the Stage-0 finding that motivated P4 — so they outrank graph
    edges, and `cap` bounds the COMBINED list (graph edges fill the remainder).
    Both sources come straight from conversation evidence (unlike the
    machine-generated summaries the root fuses), so they give the model true
    identity/preference signals and the authority to drop a summary claim that
    conflicts — the countermeasure to hallucinations crystallized in cached
    rollups. Edges strongest-evidence first. Because profile rows join the
    returned list, they flow into the facts-block hash in the root's cache id,
    so a profile change regenerates the digest just like a graph change."""
    if cap <= 0:
        return []
    profile = [
        render_profile_fact(entry) for entry in load_profile(conn, cap=cap)
    ]
    remaining = cap - len(profile)
    if remaining <= 0:
        return profile
    rows = conn.execute(
        f"""
        SELECT subject_canonical AS s, predicate AS p, object_canonical AS o
        FROM knowledge_graph
        WHERE {live_edge_predicate()}
        ORDER BY pos_evidence - neg_evidence DESC,
                 {graph_clock_order_sql('last_seen')}, id
        LIMIT ?
        """,
        (remaining,),
    ).fetchall()
    return profile + [f"{r['s']} {r['p']} {r['o']}" for r in rows]


@dataclass
class PendingAggregationNodeEmbeddings:
    node_ids: list[str]
    text_hashes: list[str]
    vectors: list[list[float]]
    from_cache: list[bool]
    model: str
    dim: int
    cache_hits: int = 0


def fetch_node_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> PendingAggregationNodeEmbeddings | None:
    """Prepare current node vectors without holding a database write lock."""
    if conn.in_transaction:
        raise RuntimeError("aggregation embedding fetch requires no transaction")
    model, initial_dim = _embedding_identity(embedder)
    rows = conn.execute(
        """
        SELECT n.id, n.title, n.summary, ne.text_hash AS stored_hash,
               ne.model AS stored_model, ne.dim AS stored_dim,
               ne.vector_json AS stored_vector
        FROM aggregation_nodes n
        LEFT JOIN aggregation_node_embeddings ne ON ne.node_id = n.id
        ORDER BY n.id
        """
    ).fetchall()
    pending: list[tuple[str, str, str]] = []
    for r in rows:
        text = f"{r['title']}\n{r['summary']}"
        text_hash = embedding_text_hash(text)
        if (
            r["stored_hash"] == text_hash
            and r["stored_model"] == model
            and r["stored_dim"] == initial_dim
        ):
            try:
                stored = decode_vector(r["stored_vector"])
            except (AttributeError, UnicodeError, TypeError, ValueError):
                stored = None
            if _finite_embedding_vector(
                stored, expected_dim=initial_dim
            ) is not None:
                continue
        pending.append((r["id"], text, text_hash))
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


def persist_node_embeddings(
    conn: sqlite3.Connection, pending: PendingAggregationNodeEmbeddings
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
        source = conn.execute(
            "SELECT title, summary FROM aggregation_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if source is None or embedding_text_hash(
            f"{source['title']}\n{source['summary']}"
        ) != text_hash:
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
            INSERT INTO aggregation_node_embeddings(node_id, vector_json, model, dim, text_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                vector_json = excluded.vector_json,
                model = excluded.model,
                dim = excluded.dim,
                text_hash = excluded.text_hash
            """,
            (
                node_id, encode_vector(vector), pending.model,
                pending.dim, text_hash,
            ),
        )
        persisted += 1
    return persisted


def _reusable_fusion(
    conn: sqlite3.Connection,
    node_id: str,
    cached: dict | None,
    *,
    input_fingerprint: str,
    expected_sources: tuple | None,
) -> dict | None:
    """Return a cached fusion only when its exact effective input still agrees."""

    if cached is None or cached.get("input_fingerprint") != input_fingerprint:
        return None
    if expected_sources is None:
        source_count = conn.execute(
            "SELECT COUNT(*) FROM aggregation_node_source_occurrences "
            "WHERE node_id=?",
            (node_id,),
        ).fetchone()[0]
        if (
            cached.get("source_manifest_complete") != 0
            or cached.get("source_manifest_count") != 0
            or source_count != 0
        ):
            return None
    elif load_aggregation_source_manifest(
        conn, node_id, validate_level0_input=False
    ) != expected_sources:
        return None
    return {"title": cached["title"], "summary": cached["summary"]}


def build_aggregation_nodes(
    conn: sqlite3.Connection,
    cfg: HyMemConfig,
    llm: LLMClient,
    embedding_client: EmbeddingClient | None = None,
    *,
    episode_ceiling_rowid: int | None = None,
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

    episodes = load_clusterable_episodes(conn, max_rowid=episode_ceiling_rowid)
    clusters = select_clusters(episodes, cfg, conn)

    # Attribution: which candidate generator clustered this dream. Node ids
    # are a function of membership, and membership can differ between the KNN
    # and the exact path (the cosine arm of blocking is approximate) — so two
    # trigger paths with different environments (one missing sqlite-vec)
    # silently alternate between two self-consistent trees, re-keying on every
    # switch. Persisting the mode makes that alternation visible in dream_runs.
    if cfg.aggregation_blocking_top_k <= 0:
        blocking = "exact:disabled"
    elif not core_db._load_vec_extension(conn):
        blocking = "exact:no_vec_extension"
    elif not core_db.has_vec_table(conn, table="vec_episodes"):
        blocking = "exact:no_vec_table"
    else:
        blocking = "knn"
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
    prev_inputs: set[tuple[int, frozenset[str], str | None]] = set()
    for row in conn.execute(
        "SELECT id,title,summary,member_episode_ids,level,input_fingerprint,"
        "source_manifest_complete,source_manifest_count FROM aggregation_nodes"
    ):
        existing[row["id"]] = {
            "title": row["title"],
            "summary": row["summary"],
            "input_fingerprint": row["input_fingerprint"],
            "source_manifest_complete": row["source_manifest_complete"],
            "source_manifest_count": row["source_manifest_count"],
        }
        try:
            members = json.loads(row["member_episode_ids"])
        except (TypeError, ValueError):
            continue
        prev_inputs.add(
            (row["level"], frozenset(members), row["input_fingerprint"])
        )

    rows: list[dict] = []
    items: list[dict] = []          # hierarchy frontier: level-0 nodes first
    clustered_ids: set[str] = set()
    reused = 0
    failures = 0
    level0_missed = 0               # instrumentation: level-0 re-keys this dream
    for members in clusters:
        member_ids = [m["id"] for m in members]
        input_fingerprint = aggregation_input_fingerprint(members)
        node_id = _node_id(
            member_ids, salt=_CLUSTER_SALT,
            input_fingerprint=input_fingerprint,
        )
        sources = None
        if all(m["source_provenance_complete"] for m in members):
            sources = combine_source_occurrences(
                m["source_occurrences"] for m in members
            )
        cached = existing.get(node_id)
        fused = _reusable_fusion(
            conn, node_id, cached,
            input_fingerprint=input_fingerprint,
            expected_sources=sources,
        )
        level0_reused = fused is not None
        if fused is not None:
            reused += 1
        else:
            level0_missed += 1
            fused = _summarize_cluster(members, cfg, llm)
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
        session_ids = sorted({m["session_id"] for m in members})
        rows.append({
            "id": node_id, "title": fused["title"], "summary": fused["summary"],
            "member_ids": member_ids, "session_ids": session_ids,
            "level": 0, "is_root": 0, "reused": level0_reused,
            "source_occurrences": sources,
            "input_fingerprint": input_fingerprint,
        })
        clustered_ids.update(member_ids)
        items.append({
            "id": node_id, "title": fused["title"], "summary": fused["summary"],
            "vector": _centroid([m["vector"] for m in members]),
            "entities": set().union(*(m["entities"] for m in members)),
            "session_ids": set(session_ids),
            "source_occurrences": sources or (),
            "source_provenance_complete": sources is not None,
            "source_manifest_hash": (
                source_manifest_hash(AGGREGATION_SOURCE_MANIFEST_VERSION, sources)
                if sources is not None else None
            ),
        })

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
            "vector": e["vector"], "entities": e["entities"],
            "session_ids": {e["session_id"]},
            "source_occurrences": e["source_occurrences"],
            "source_provenance_complete": e["source_provenance_complete"],
            "source_manifest_hash": e["source_manifest_hash"],
        } for e in leftovers]
        digest_rows, digest_reused, digest_failures, root_failed = _build_digest_levels(
            conn, items, cfg, llm, existing,
            anchor_facts=_anchor_facts(conn, cfg.aggregation_digest_anchor_facts),
        )
        rows += digest_rows
        reused += digest_reused
        failures += digest_failures

    with core_db.transaction(conn):
        # Full replace: rows the new clustering no longer produces must not linger.
        # ON DELETE CASCADE clears aggregation_node_embeddings for dropped nodes.
        # Exception: when the ROOT fusion failed, the previous root survives —
        # a one-dream-stale digest (its footer already names generated_at)
        # beats HyMem.digest() returning nothing until the retry heals. Its
        # member ids may point at replaced nodes; expand_node reports those as
        # missing_member_ids rather than failing.
        if root_failed:
            log.warning("aggregate.root_fusion_failed keeping previous root")
            # Compatibility keeps the last readable digest text, but its old
            # tree members/proof no longer describe the just-built frontier.
            # Quarantine that cache entry before deleting its children so no
            # caller can mistake stale source rows for current provenance.
            conn.execute(
                "UPDATE aggregation_nodes SET source_manifest_version=?,"
                "source_manifest_count=0,source_manifest_hash=NULL,"
                "source_manifest_complete=0 WHERE is_root=1",
                (AGGREGATION_SOURCE_MANIFEST_VERSION,),
            )
            conn.execute(
                "DELETE FROM aggregation_node_source_occurrences WHERE node_id IN "
                "(SELECT id FROM aggregation_nodes WHERE is_root=1)"
            )
            conn.execute("DELETE FROM aggregation_nodes WHERE is_root = 0")
        else:
            conn.execute("DELETE FROM aggregation_nodes")
        for r in rows:
            conn.execute(
                """
                INSERT INTO aggregation_nodes(
                    id, title, summary, member_episode_ids, session_ids,
                    n_members, n_sessions, level, is_root, input_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["id"], r["title"], r["summary"],
                    json.dumps(r["member_ids"]), json.dumps(r["session_ids"]),
                    len(r["member_ids"]), len(r["session_ids"]),
                    r["level"], r["is_root"], r["input_fingerprint"],
                ),
            )
            persist_aggregation_source_manifest(
                conn,
                r["id"],
                occurrences=r["source_occurrences"],
                input_fingerprint=r["input_fingerprint"],
            )
        if leaf_fingerprint is not None:
            # Same transaction as the nodes above: a dream that dies before
            # persisting must not leave the watermark pointing at a leaf set no
            # tree was ever built from, which would report the NEXT dream's
            # genuine displacement as unchanged.
            _write_leaf_fingerprint(conn, leaf_fingerprint, leaf_count, leaf_set)

    if embedding_client is not None and rows:
        pending_node_embeddings = fetch_node_embeddings(conn, embedding_client)
        if pending_node_embeddings is not None:
            with core_db.transaction(conn):
                persist_node_embeddings(conn, pending_node_embeddings)

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
    return AggregationResult(
        len(rows), reused, failures, len(episodes), blocking,
        level0_missed, leaf_res,
        forecast.predicted, forecast.residual, forecast.facts_rekey,
        forecast.rebuilt_level0, forecast.rebuilt_rollup, forecast.rebuilt_root,
        leaf_added, leaf_removed,
    )


def _build_digest_levels(
    conn: sqlite3.Connection,
    items: list[dict], cfg: HyMemConfig, llm: LLMClient,
    existing: dict[str, dict], *, anchor_facts: list[str],
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
    caller the root specifically failed, so it can keep the previous root row
    instead of leaving the store digest-less until the retry heals."""
    rows: list[dict] = []
    reused = 0
    failures = 0
    fan_in = max(2, cfg.aggregation_max_members)
    level = 1
    while len(items) > fan_in:
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
            grouped.setdefault(labels[it["id"]], []).append(it)
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
            if len(g) < 2:
                next_items.append(g[0])
                continue
            member_ids = [m["id"] for m in g]
            input_fingerprint = aggregation_input_fingerprint(g)
            node_id = _node_id(
                member_ids, salt=_ROLLUP_SALT,
                input_fingerprint=input_fingerprint,
            )
            sources = None
            if all(m.get("source_provenance_complete", False) for m in g):
                sources = combine_source_occurrences(
                    m["source_occurrences"] for m in g
                )
            cached = existing.get(node_id)
            fused = _reusable_fusion(
                conn, node_id, cached,
                input_fingerprint=input_fingerprint,
                expected_sources=sources,
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
                    ROLLUP_USER_TEMPLATE.format(text=_items_text(g, cfg)),
                    llm, system=ROLLUP_SYSTEM, kind="rollup",
                    shrink=lambda: ROLLUP_USER_TEMPLATE.format(
                        text=_items_text(g, cfg, char_scale=0.5)),
                )
                if fused is None:
                    # CONTAINMENT (see the level-0 twin): the group sits this
                    # dream out rather than leaking raw members upward.
                    failures += 1
                    continue
            session_ids: set[str] = set().union(*(m["session_ids"] for m in g))
            rows.append({
                "id": node_id, "title": fused["title"], "summary": fused["summary"],
                "member_ids": member_ids, "session_ids": sorted(session_ids),
                "level": level, "is_root": 0, "reused": rollup_reused,
                "source_occurrences": sources,
                "input_fingerprint": input_fingerprint,
            })
            next_items.append({
                "id": node_id, "title": fused["title"], "summary": fused["summary"],
                "vector": _centroid([m["vector"] for m in g]),
                "entities": set().union(*(m["entities"] for m in g)),
                "session_ids": session_ids,
                "source_occurrences": sources or (),
                "source_provenance_complete": sources is not None,
                "source_manifest_hash": (
                    source_manifest_hash(
                        AGGREGATION_SOURCE_MANIFEST_VERSION, sources
                    ) if sources is not None else None
                ),
            })
        if len(next_items) >= len(items):    # no progress (fusions all failed)
            items = next_items
            break
        items = next_items
        level += 1

    if not items:
        return rows, reused, failures, False
    member_ids = [m["id"] for m in items]
    # The VERIFIED FACTS anchor is part of the root's INPUT, so it joins the
    # cache key: a changed graph (new fact, supersession) regenerates the
    # digest even when the tree's membership is unchanged — at most one extra
    # LLM call per dream, and the price of NOT doing it is a digest pinned to
    # stale ground truth.
    facts_block = "\n".join(f"- {f}" for f in anchor_facts) if anchor_facts else "(none)"
    facts_hash = hashlib.sha1(facts_block.encode("utf-8")).hexdigest()[:12]
    input_fingerprint = aggregation_input_fingerprint(
        items, extra_inputs=(facts_block,)
    )
    root_id = _node_id(
        member_ids, salt=f"{_ROOT_SALT}|{facts_hash}",
        input_fingerprint=input_fingerprint,
    )
    # The verified-facts block currently returns rendered strings, not a source
    # manifest. Never attest only the tree leaves when additional graph/profile
    # claims entered the effective prompt. A future typed anchor DTO can make
    # this complete; until then roots with anchors are explicitly quarantined.
    root_sources = None
    if not anchor_facts and all(
        item.get("source_provenance_complete", False) for item in items
    ):
        root_sources = combine_source_occurrences(
            item["source_occurrences"] for item in items
        )
    cached = existing.get(root_id)
    fused = _reusable_fusion(
        conn, root_id, cached,
        input_fingerprint=input_fingerprint,
        expected_sources=root_sources,
    )
    root_reused = fused is not None
    if fused is not None:
        reused += 1
    else:
        fused = _llm_fuse(
            DIGEST_USER_TEMPLATE.format(facts=facts_block,
                                        text=_items_text(items, cfg)),
            llm, system=DIGEST_SYSTEM, kind="root",
            shrink=lambda: DIGEST_USER_TEMPLATE.format(
                facts=facts_block,
                text=_items_text(items, cfg, char_scale=0.5)),
        )
    if fused is None:
        return rows, reused, failures + 1, True
    session_ids = sorted(set().union(*(m["session_ids"] for m in items)))
    rows.append({
        "id": root_id, "title": fused["title"], "summary": fused["summary"],
        "member_ids": member_ids, "session_ids": session_ids,
        "level": level, "is_root": 1, "reused": root_reused,
        "source_occurrences": root_sources,
        "input_fingerprint": input_fingerprint,
    })
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


def load_digest(conn: sqlite3.Connection) -> Digest | None:
    """Return the current root digest, or None when the aggregation layer is
    disabled, has not dreamed yet, or the store has no episodes. Read-only."""
    row = conn.execute(
        """
        SELECT id, title, summary, n_sessions, created_at
        FROM aggregation_nodes
        WHERE is_root = 1
        ORDER BY created_at DESC, id
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    total = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
    return Digest(
        title=row["title"],
        summary=row["summary"],
        n_sessions=row["n_sessions"],
        n_sessions_total=int(total),
        generated_at=row["created_at"] or "",
        node_id=row["id"],
    )


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
class NodeMemberEpisode:
    """A leaf inside a `NodeExpansion`: the per-session episode whose summary
    fed the node's fusion. `start_message_id`/`end_message_id` bound the raw
    turns it condenses, so a host can jump from any digest claim all the way
    down to the original conversation."""

    id: str
    session_id: str
    title: str
    summary: str
    start_message_id: int
    end_message_id: int


@dataclass
class NodeExpansion:
    """One step of the RAPTOR tree-traversal read (`HyMem.expand_node()`):
    the node itself plus its members, resolved one level down. Members of a
    level >= 1 node are a mix of child nodes and pass-through episodes (leaves
    no cluster absorbed); level-0 members are episodes only. Member order is
    the persisted fusion-input order. `missing_member_ids` keeps the read
    honest instead of silently shrinking: ids that resolved to neither a node
    nor an episode (should not happen — nodes are rebuilt atomically each
    dream — so anything here indicates store surgery)."""

    id: str
    title: str
    summary: str
    level: int
    is_root: bool
    child_nodes: list[NodeChild]
    episodes: list[NodeMemberEpisode]
    missing_member_ids: list[str]


def expand_node(conn: sqlite3.Connection, node_id: str) -> NodeExpansion | None:
    """Resolve an aggregation node's members one level down — the Stage-4b
    drill-down behind "why does my digest say X?". Start from
    `Digest.node_id` (the root) or an `AggregationNodeHit.node_id` from the
    query tier, and recurse through `child_nodes` until everything is
    episodes. Returns None for an unknown id. Read-only; per-member point
    lookups (members are capped at fusion time, so the fan-out is small)."""
    row = conn.execute(
        """
        SELECT id, title, summary, level, is_root, member_episode_ids
        FROM aggregation_nodes
        WHERE id = ?
        """,
        (node_id,),
    ).fetchone()
    if row is None:
        return None

    child_nodes: list[NodeChild] = []
    episodes: list[NodeMemberEpisode] = []
    missing: list[str] = []
    for member_id in json.loads(row["member_episode_ids"]):
        node_row = conn.execute(
            """
            SELECT id, title, summary, level, n_members, n_sessions
            FROM aggregation_nodes
            WHERE id = ?
            """,
            (member_id,),
        ).fetchone()
        if node_row is not None:
            child_nodes.append(NodeChild(
                id=node_row["id"],
                title=node_row["title"],
                summary=node_row["summary"],
                level=node_row["level"],
                n_members=node_row["n_members"],
                n_sessions=node_row["n_sessions"],
            ))
            continue
        ep_row = conn.execute(
            """
            SELECT e.id, e.session_id, e.title, e.summary,
                   e.start_message_id, e.end_message_id
            FROM episodes e
            JOIN sessions s ON s.id = e.session_id
            WHERE e.id = ?
              AND (e.digest_generation IS NULL
                   OR e.digest_generation = s.digest_published_generation)
            """,
            (member_id,),
        ).fetchone()
        if ep_row is not None:
            episodes.append(NodeMemberEpisode(
                id=ep_row["id"],
                session_id=ep_row["session_id"],
                title=ep_row["title"],
                summary=ep_row["summary"],
                start_message_id=ep_row["start_message_id"],
                end_message_id=ep_row["end_message_id"],
            ))
            continue
        missing.append(member_id)

    return NodeExpansion(
        id=row["id"],
        title=row["title"],
        summary=row["summary"],
        level=row["level"],
        is_root=bool(row["is_root"]),
        child_nodes=child_nodes,
        episodes=episodes,
        missing_member_ids=missing,
    )
