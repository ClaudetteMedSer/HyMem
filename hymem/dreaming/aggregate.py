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

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.core.vectors import decode_vector, encode_vector
from hymem.extraction.embeddings import EmbeddingClient, normalize_text
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

# Fusion-prompt versions, baked into the node-id salt of the level the prompt
# serves. Reuse is keyed by node id, so bumping a version when its prompt
# changes materially makes every cached fusion of that kind regenerate on the
# next dream — without touching the other levels' caches. This matters beyond
# style: a hallucination CRYSTALLIZES in a cached fusion (the "Acme Corp"
# incident lived in a persisted rollup and survived a root-only fix), so a
# prompt hardened against an artifact must invalidate the level that produced
# it, or the artifact outlives the fix.
_CLUSTER_SALT = "cluster.v2"  # v2: identity evidence-bound (was unsalted)
_ROLLUP_SALT = "rollup.v2"    # v2: identity evidence-bound
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
    episodes: list[dict], emb_threshold: float, ent_threshold: float
) -> dict[str, int]:
    """Connected-components clustering over the episode link graph (union-find).

    `episodes`: list of {"id": str, "vector": list[float]|None, "entities": set[str]}.
    Returns {episode_id -> cluster_label}. Two episodes share a label iff there is a
    path of `_linked` edges between them (transitive closure). This is deliberately
    the simplest cross-session aggregation a RAPTOR layer could do; if even this
    co-locates the gold, a smarter clusterer only does better.
    """
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

    for i in range(len(episodes)):
        for j in range(i + 1, len(episodes)):
            if _linked(episodes[i], episodes[j], emb_threshold, ent_threshold):
                union(episodes[i]["id"], episodes[j]["id"])

    roots = {e["id"]: find(e["id"]) for e in episodes}
    label_of: dict[str, int] = {}
    out: dict[str, int] = {}
    for eid, root in roots.items():
        if root not in label_of:
            label_of[root] = len(label_of)
        out[eid] = label_of[root]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DB-side build (Hermes box: needs real episodes + embeddings; StubLLM in tests).
# ─────────────────────────────────────────────────────────────────────────────

def _norm_entity(x: str) -> str:
    return normalize_text(x).strip()


def load_clusterable_episodes(conn: sqlite3.Connection) -> list[dict]:
    """All episodes with their summary vector + normalized entity set, ordered so
    a stable member list / id falls out of clustering. Mirrors the probe loader."""
    rows = conn.execute(
        """
        SELECT e.id, e.session_id, e.title, e.summary,
               e.start_message_id, e.end_message_id, e.key_entities,
               em.vector_json
        FROM episodes e
        LEFT JOIN episode_embeddings em ON em.episode_id = e.id
        ORDER BY e.session_id, e.start_message_id, e.id
        """
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
            "session_id": r["session_id"],
            "title": r["title"],
            "summary": r["summary"],
            "entities": {_norm_entity(x) for x in raw_entities if x},
            "vector": vec,
        })
    return episodes


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


def _evenly_spaced(seq: list, cap: int) -> list:
    """At most `cap` elements spread evenly across `seq` (first and last always
    kept). Used to cap the digest's pass-through leaves: a recency slice
    (`seq[-cap:]`) would make a first build over a long backlog digest only the
    newest stretch of history — even spacing keeps the digest spanning the
    user's whole span at the same LLM cost."""
    n = len(seq)
    if cap <= 0 or n <= cap:
        return list(seq)
    if cap == 1:
        return [seq[-1]]
    idx = sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})
    return [seq[i] for i in idx]


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
    episodes: list[dict], cfg: HyMemConfig
) -> list[list[dict]]:
    """Cluster all episodes, then keep only the clusters worth a summary: at least
    `aggregation_min_members` episodes spanning at least `aggregation_min_sessions`
    distinct sessions. Returns each kept cluster's episodes in load order."""
    if not episodes:
        return []
    labels = cluster_episodes(
        episodes, cfg.aggregation_emb_threshold, cfg.aggregation_ent_threshold
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
    # Deterministic order: larger clusters first, then by first member id.
    kept.sort(key=lambda c: (-len(c), sorted(m["id"] for m in c)[0]))
    return kept


def _llm_fuse(
    user_prompt: str, llm: LLMClient, *, system: str
) -> dict | None:
    """One LLM call fusing the prepared `user_prompt` into {title, summary}.
    Returns None when the call fails or yields nothing usable (so no empty
    node is persisted)."""
    request = LLMRequest(
        system=system,
        user=user_prompt,
        response_format="json",
    )
    try:
        raw = llm.complete(request)
    except Exception:
        log.exception("aggregate.llm_failure")
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    title = data.get("title", "")
    summary = data.get("summary", "")
    if not isinstance(title, str) or not isinstance(summary, str):
        return None
    title, summary = title.strip(), summary.strip()
    if not title or not summary:
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
                     system=AGGREGATE_SYSTEM)


def _items_text(items: list[dict], cfg: HyMemConfig) -> str:
    """Render hierarchy items (level-0 nodes / rollups / pass-through episodes,
    all carrying title+summary) as one fusion input block."""
    capped = items[: cfg.aggregation_max_members]
    return "\n\n---\n\n".join(f"{m['title']}\n{m['summary']}" for m in capped)


def _anchor_facts(conn: sqlite3.Connection, cap: int) -> list[str]:
    """Top ACTIVE, non-derived, non-superseded knowledge-graph edges rendered
    as one-line facts — the VERIFIED FACTS block grounding the root digest
    fusion. Graph edges come straight from conversation evidence (unlike the
    machine-generated summaries the root fuses), so they give the model true
    identity/preference signals and the authority to drop a summary claim that
    conflicts — the countermeasure to hallucinations crystallized in cached
    rollups. Strongest evidence first."""
    if cap <= 0:
        return []
    rows = conn.execute(
        """
        SELECT subject_canonical AS s, predicate AS p, object_canonical AS o
        FROM knowledge_graph
        WHERE status = 'active' AND derived = 0 AND invalid_at IS NULL
          AND pos_evidence > neg_evidence
        ORDER BY pos_evidence - neg_evidence DESC, last_seen DESC, id
        LIMIT ?
        """,
        (cap,),
    ).fetchall()
    return [f"{r['s']} {r['p']} {r['o']}" for r in rows]


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
) -> int:
    """Rebuild the cross-session aggregation layer from the current episodes.

    No-op when the layer is disabled. Otherwise: cluster → keep cross-session
    multi-member clusters → fuse each NEW cluster with one LLM call (an
    unchanged member set reuses the stored fusion, no call) → full-replace
    `aggregation_nodes` → (re-)embed node summaries. Returns the node count.
    Full rebuild (DELETE then INSERT) because membership is a pure function of
    the present episodes; the content-hash id means an unchanged cluster keeps
    both its fusion and its embedding. Caller need not hold a transaction.
    """
    if not cfg.aggregation_nodes_enabled:
        return 0

    episodes = load_clusterable_episodes(conn)
    clusters = select_clusters(episodes, cfg)

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
    for members in clusters:
        member_ids = [m["id"] for m in members]
        node_id = _node_id(member_ids, salt=_CLUSTER_SALT)
        fused = existing.get(node_id)
        if fused is not None:
            reused += 1
        else:
            fused = _summarize_cluster(members, cfg, llm)
            if fused is None:
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

    if cfg.aggregation_digest_enabled:
        # Digest leaves = the level-0 nodes plus every episode no cluster
        # absorbed (capped, sampled evenly across the backlog), so the root
        # covers the WHOLE store — full time span, not just recent threads.
        leftovers = [e for e in episodes if e["id"] not in clustered_ids]
        leftovers = _evenly_spaced(leftovers, cfg.aggregation_digest_max_leaves)
        items += [{
            "id": e["id"], "title": e["title"] or "", "summary": e["summary"] or "",
            "vector": e["vector"], "entities": e["entities"],
            "session_ids": {e["session_id"]},
        } for e in leftovers]
        digest_rows, digest_reused = _build_digest_levels(
            items, cfg, llm, existing,
            anchor_facts=_anchor_facts(conn, cfg.aggregation_digest_anchor_facts),
        )
        rows += digest_rows
        reused += digest_reused

    with core_db.transaction(conn):
        # Full replace: rows the new clustering no longer produces must not linger.
        # ON DELETE CASCADE clears aggregation_node_embeddings for dropped nodes.
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

    log.info("aggregate.built nodes=%d reused=%d (from %d episodes)",
             len(rows), reused, len(episodes))
    return len(rows)


def _build_digest_levels(
    items: list[dict], cfg: HyMemConfig, llm: LLMClient,
    existing: dict[str, dict], *, anchor_facts: list[str],
) -> tuple[list[dict], int]:
    """RAPTOR rollup: recursively cluster-and-fuse the frontier `items` (each
    {"id","title","summary","vector","entities","session_ids"}) until at most
    `aggregation_max_members` remain, then fuse those into the single ROOT
    digest node. Returns (node rows for every level >= 1, reused-fusion count).

    Each pass clusters with the SAME `_linked` rule as level 0 (centroid
    vectors, union entity sets); when nothing links — disjoint topics, exactly
    the case a digest must still cover — it falls back to fusing consecutive
    runs of `fan_in` items, which guarantees the loop converges. A failed
    fusion passes its members through unfused (the tree degrades, never
    blocks); if a whole pass makes no progress the loop bails out and the root
    fuses whatever frontier remains (capped inside `_items_text`).

    The root fusion is GROUNDED: `anchor_facts` (top knowledge-graph edges)
    render as a VERIFIED FACTS block the digest prompt treats as ground truth
    over the machine-generated summaries, and the block's hash joins the root's
    cache id so a changed graph regenerates the digest."""
    rows: list[dict] = []
    reused = 0
    fan_in = max(2, cfg.aggregation_max_members)
    level = 1
    while len(items) > fan_in:
        labels = cluster_episodes(
            items, cfg.aggregation_emb_threshold, cfg.aggregation_ent_threshold
        )
        grouped: dict[int, list[dict]] = {}
        for it in items:
            grouped.setdefault(labels[it["id"]], []).append(it)
        groups = sorted(grouped.values(),
                        key=lambda g: (-len(g), sorted(m["id"] for m in g)[0]))
        if all(len(g) < 2 for g in groups):
            groups = [items[i:i + fan_in] for i in range(0, len(items), fan_in)]

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
                    llm, system=ROLLUP_SYSTEM,
                )
                if fused is None:
                    next_items.extend(g)
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
        return rows, reused
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
            llm, system=DIGEST_SYSTEM,
        )
    if fused is None:
        return rows, reused
    session_ids = sorted(set().union(*(m["session_ids"] for m in items)))
    rows.append({
        "id": root_id, "title": fused["title"], "summary": fused["summary"],
        "member_ids": member_ids, "session_ids": session_ids,
        "level": level, "is_root": 1,
    })
    return rows, reused


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


def load_digest(conn: sqlite3.Connection) -> Digest | None:
    """Return the current root digest, or None when the aggregation layer is
    disabled, has not dreamed yet, or the store has no episodes. Read-only."""
    row = conn.execute(
        """
        SELECT title, summary, n_sessions, created_at
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
    )
