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
summary embeddings are cache-keyed by text so an unchanged node re-uses its
vector.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.core.vectors import decode_vector, encode_vector
from hymem.extraction.embeddings import EmbeddingClient, normalize_text
from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import AGGREGATE_SYSTEM, AGGREGATE_USER_TEMPLATE

log = logging.getLogger("hymem.dreaming.aggregate")


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


def _node_id(member_ids: list[str]) -> str:
    """Stable id for a node = content hash of its sorted member episode ids, so an
    unchanged cluster keeps its id (and cached embedding) across dream cycles."""
    digest = hashlib.sha1("|".join(sorted(member_ids)).encode("utf-8")).hexdigest()[:16]
    return f"agg_{digest}"


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


def _summarize_cluster(
    members: list[dict], cfg: HyMemConfig, llm: LLMClient
) -> dict | None:
    """One LLM call fusing a cluster's episodes into {title, summary}. Returns
    None if the cluster yields nothing usable (so no empty node is persisted)."""
    capped = members[: cfg.aggregation_max_members]
    text = "\n\n---\n\n".join(
        f"[{m['session_id']}] {m['title']}\n{m['summary']}" for m in capped
    )
    request = LLMRequest(
        system=AGGREGATE_SYSTEM,
        user=AGGREGATE_USER_TEMPLATE.format(text=text),
        response_format="json",
    )
    try:
        raw = llm.complete(request)
    except Exception:
        log.exception("aggregate.llm_failure members=%d", len(members))
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
    multi-member clusters → fuse each with one LLM call → full-replace
    `aggregation_nodes` → (re-)embed node summaries. Returns the node count.
    Full rebuild (DELETE then INSERT) because membership is a pure function of
    the present episodes; the content-hash id means an unchanged cluster keeps
    its embedding via the cache. Caller need not hold a transaction.
    """
    if not cfg.aggregation_nodes_enabled:
        return 0

    episodes = load_clusterable_episodes(conn)
    clusters = select_clusters(episodes, cfg)

    built: list[tuple[str, dict, list[dict]]] = []
    for members in clusters:
        fused = _summarize_cluster(members, cfg, llm)
        if fused is None:
            continue
        member_ids = [m["id"] for m in members]
        built.append((_node_id(member_ids), fused, members))

    with core_db.transaction(conn):
        # Full replace: rows the new clustering no longer produces must not linger.
        # ON DELETE CASCADE clears aggregation_node_embeddings for dropped nodes.
        conn.execute("DELETE FROM aggregation_nodes")
        for node_id, fused, members in built:
            member_ids = [m["id"] for m in members]
            session_ids = sorted({m["session_id"] for m in members})
            conn.execute(
                """
                INSERT INTO aggregation_nodes(
                    id, title, summary, member_episode_ids, session_ids,
                    n_members, n_sessions
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id, fused["title"], fused["summary"],
                    json.dumps(member_ids), json.dumps(session_ids),
                    len(member_ids), len(session_ids),
                ),
            )

    if embedding_client is not None and built:
        with core_db.transaction(conn):
            _persist_node_embeddings(conn, embedding_client)

    log.info("aggregate.built nodes=%d (from %d episodes)", len(built), len(episodes))
    return len(built)
