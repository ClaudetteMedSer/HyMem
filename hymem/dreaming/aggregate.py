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

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.core.vectors import decode_vector, encode_vector
from hymem.dreaming.user_profile import load_profile, render_profile_fact
from hymem.extraction.embeddings import EmbeddingClient, normalize_text
from hymem.extraction.jsonio import loads_lenient
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
    where = "WHERE e.rowid <= ?" if max_rowid is not None else ""
    rows = conn.execute(
        f"""
        SELECT e.rowid AS rowid, e.id, e.session_id, e.title, e.summary,
               e.start_message_id, e.end_message_id, e.key_entities,
               em.vector_json
        FROM episodes e
        LEFT JOIN episode_embeddings em ON em.episode_id = e.id
        {where}
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


def _node_id(member_ids: list[str], *, salt: str = "") -> str:
    """Stable id for a node = content hash of its sorted member ids, so an
    unchanged cluster keeps its id (and cached embedding + fusion) across dream
    cycles. `salt` separates id spaces for nodes that could share a member set
    but carry a different KIND of fusion (the root digest uses a different
    prompt than an intermediate rollup, so they must never reuse each other)."""
    payload = "|".join(sorted(member_ids))
    if salt:
        payload = f"{salt}::{payload}"
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
        if len(members) < cfg.aggregation_min_members:
            continue
        if len({m["session_id"] for m in members}) < cfg.aggregation_min_sessions:
            continue
        kept.append(members)
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


def _llm_fuse(
    user_prompt: str, llm: LLMClient, *, system: str, kind: str = "fusion"
) -> dict | None:
    """One LLM call fusing the prepared `user_prompt` into {title, summary}.
    Returns None when the call fails or yields nothing usable (so no empty
    node is persisted). Every failure path logs at WARNING with `kind`
    (cluster/rollup/root): a failed fusion retries on every subsequent dream
    until it succeeds, and each fail→heal transition costs reuse — silent
    failures made the 2026-07-12 low-reuse runs unattributable."""
    request = LLMRequest(
        system=system,
        user=user_prompt,
        response_format="json",
    )
    try:
        raw = llm.complete(request)
    except Exception:
        log.exception("aggregate.fusion_failure kind=%s stage=call", kind)
        return None
    # Fences/prose around the JSON are tolerated (dream 1013): json_object mode
    # was already set on this call and the provider fenced it anyway.
    data = loads_lenient(raw, expect="object")
    if data is None:
        log.warning("aggregate.fusion_failure kind=%s stage=parse raw_len=%d",
                    kind, len(raw) if isinstance(raw, str) else -1)
        return None
    if not isinstance(data, dict):
        log.warning("aggregate.fusion_failure kind=%s stage=shape", kind)
        return None
    title = data.get("title", "")
    summary = data.get("summary", "")
    if not isinstance(title, str) or not isinstance(summary, str):
        log.warning("aggregate.fusion_failure kind=%s stage=shape", kind)
        return None
    title, summary = title.strip(), summary.strip()
    if not title or not summary:
        log.warning("aggregate.fusion_failure kind=%s stage=empty", kind)
        return None
    return {"title": title, "summary": summary}


def _summarize_cluster(
    members: list[dict], cfg: HyMemConfig, llm: LLMClient
) -> dict | None:
    """Fuse a level-0 cluster's episodes into {title, summary}."""
    capped = members[: cfg.aggregation_max_members]
    text = "\n\n---\n\n".join(
        f"[{m['session_id']}] {m['title']}\n{m['summary']}" for m in capped
    )
    return _llm_fuse(AGGREGATE_USER_TEMPLATE.format(text=text), llm,
                     system=AGGREGATE_SYSTEM, kind="cluster")


def _items_text(items: list[dict], cfg: HyMemConfig) -> str:
    """Render hierarchy items (level-0 nodes / rollups / pass-through episodes,
    all carrying title+summary) as one fusion input block."""
    capped = items[: cfg.aggregation_max_members]
    return "\n\n---\n\n".join(f"{m['title']}\n{m['summary']}" for m in capped)


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
        """
        SELECT subject_canonical AS s, predicate AS p, object_canonical AS o
        FROM knowledge_graph
        WHERE status = 'active' AND derived = 0 AND invalid_at IS NULL
          AND pos_evidence > neg_evidence
        ORDER BY pos_evidence - neg_evidence DESC, last_seen DESC, id
        LIMIT ?
        """,
        (remaining,),
    ).fetchall()
    return profile + [f"{r['s']} {r['p']} {r['o']}" for r in rows]


def _persist_node_embeddings(
    conn: sqlite3.Connection, embedder: EmbeddingClient
) -> int:
    """Embed every aggregation node missing a current vector (cache-aware) and
    UPSERT into aggregation_node_embeddings. Caller wraps in a transaction.
    No vec0 table — retrieval scans these rows with Python cosine."""
    rows = conn.execute(
        """
        SELECT n.id, n.title, n.summary, ne.text_hash AS stored_hash
        FROM aggregation_nodes n
        LEFT JOIN aggregation_node_embeddings ne ON ne.node_id = n.id
        """
    ).fetchall()
    pending = []
    for r in rows:
        text = f"{r['title']}\n{r['summary']}"
        text_hash = hashlib.sha256(normalize_text(text).encode()).hexdigest()
        if r["stored_hash"] == text_hash:
            continue
        pending.append((r["id"], text, text_hash))
    if not pending:
        return 0

    model = embedder.model
    hashes = [h for _, _, h in pending]
    placeholders = ",".join("?" * len(hashes))
    cached = {
        row["text_hash"]: decode_vector(row["vector_json"])
        for row in conn.execute(
            f"SELECT text_hash, vector_json FROM embedding_cache "
            f"WHERE model = ? AND text_hash IN ({placeholders})",
            (model, *hashes),
        ).fetchall()
    }
    miss_idx = [i for i, (_, _, h) in enumerate(pending) if h not in cached]
    if miss_idx:
        embedded = embedder.embed([pending[i][1] for i in miss_idx])
        if len(embedded) != len(miss_idx):
            raise RuntimeError(
                f"embedding client returned {len(embedded)} vectors for "
                f"{len(miss_idx)} aggregation nodes"
            )
        for i, vec in zip(miss_idx, embedded):
            cached[pending[i][2]] = vec
            conn.execute(
                "INSERT OR IGNORE INTO embedding_cache(text_hash, model, vector_json, dim) "
                "VALUES (?, ?, ?, ?)",
                (pending[i][2], model, encode_vector(vec), len(vec)),
            )

    for node_id, _, text_hash in pending:
        vec = cached[text_hash]
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
            (node_id, encode_vector(vec), model, len(vec), text_hash),
        )
    return len(pending)


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
    existing: dict[str, dict] = {
        row["id"]: {"title": row["title"], "summary": row["summary"]}
        for row in conn.execute("SELECT id, title, summary FROM aggregation_nodes")
    }

    rows: list[dict] = []
    items: list[dict] = []          # hierarchy frontier: level-0 nodes first
    clustered_ids: set[str] = set()
    reused = 0
    failures = 0
    for members in clusters:
        member_ids = [m["id"] for m in members]
        node_id = _node_id(member_ids, salt=_CLUSTER_SALT)
        fused = existing.get(node_id)
        if fused is not None:
            reused += 1
        else:
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
            "level": 0, "is_root": 0,
        })
        clustered_ids.update(member_ids)
        items.append({
            "id": node_id, "title": fused["title"], "summary": fused["summary"],
            "vector": _centroid([m["vector"] for m in members]),
            "entities": set().union(*(m["entities"] for m in members)),
            "session_ids": set(session_ids),
        })

    root_failed = False
    if cfg.aggregation_digest_enabled:
        # Digest leaves = the level-0 nodes plus every episode no cluster
        # absorbed (capped by a churn-stable hash-rank sample), so the root
        # covers the WHOLE store — full time span, not just recent threads.
        leftovers = [e for e in episodes if e["id"] not in clustered_ids]
        leftovers = _stable_sample(leftovers, cfg.aggregation_digest_max_leaves)
        items += [{
            "id": e["id"], "title": e["title"] or "", "summary": e["summary"] or "",
            "vector": e["vector"], "entities": e["entities"],
            "session_ids": {e["session_id"]},
        } for e in leftovers]
        digest_rows, digest_reused, digest_failures, root_failed = _build_digest_levels(
            items, cfg, llm, existing,
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
            conn.execute("DELETE FROM aggregation_nodes WHERE is_root = 0")
        else:
            conn.execute("DELETE FROM aggregation_nodes")
        for r in rows:
            conn.execute(
                """
                INSERT INTO aggregation_nodes(
                    id, title, summary, member_episode_ids, session_ids,
                    n_members, n_sessions, level, is_root
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["id"], r["title"], r["summary"],
                    json.dumps(r["member_ids"]), json.dumps(r["session_ids"]),
                    len(r["member_ids"]), len(r["session_ids"]),
                    r["level"], r["is_root"],
                ),
            )

    if embedding_client is not None and rows:
        with core_db.transaction(conn):
            _persist_node_embeddings(conn, embedding_client)

    log.info(
        "aggregate.built nodes=%d reused=%d failures=%d blocking=%s "
        "vectorless=%d (from %d episodes)",
        len(rows), reused, failures, blocking, vectorless, len(episodes),
    )
    return AggregationResult(len(rows), reused, failures, len(episodes), blocking)


def _build_digest_levels(
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

        next_items: list[dict] = []
        for g in groups:
            if len(g) < 2:
                next_items.append(g[0])
                continue
            member_ids = [m["id"] for m in g]
            node_id = _node_id(member_ids, salt=_ROLLUP_SALT)
            fused = existing.get(node_id)
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
                "level": level, "is_root": 0,
            })
            next_items.append({
                "id": node_id, "title": fused["title"], "summary": fused["summary"],
                "vector": _centroid([m["vector"] for m in g]),
                "entities": set().union(*(m["entities"] for m in g)),
                "session_ids": session_ids,
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
    root_id = _node_id(member_ids, salt=f"{_ROOT_SALT}|{facts_hash}")
    fused = existing.get(root_id)
    if fused is not None:
        reused += 1
    else:
        fused = _llm_fuse(
            DIGEST_USER_TEMPLATE.format(facts=facts_block,
                                        text=_items_text(items, cfg)),
            llm, system=DIGEST_SYSTEM, kind="root",
        )
    if fused is None:
        return rows, reused, failures + 1, True
    session_ids = sorted(set().union(*(m["session_ids"] for m in items)))
    rows.append({
        "id": root_id, "title": fused["title"], "summary": fused["summary"],
        "member_ids": member_ids, "session_ids": session_ids,
        "level": level, "is_root": 1,
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
            SELECT id, session_id, title, summary,
                   start_message_id, end_message_id
            FROM episodes
            WHERE id = ?
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
