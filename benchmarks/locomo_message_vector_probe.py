#!/usr/bin/env python3
"""Could a VECTOR path over raw turns reach the gold that BM25 can't? Read-only.

The LoCoMo residual is architectural, not adapter-side: `message_hits` — the
dominant recovery tier — is BM25-only ([_message_fts_search](../hymem/query/augment.py)),
with no vector path over raw `messages` at all (`vec_chunks`/`vec_edges`/
`vec_episodes` exist; there is no `vec_messages`). The store probe found 25 of 42
gold turns are pure vocabulary gaps: indexed, reachable by their own terms, but
sharing no salient token with the question — so no aperture, top_k, or prompt can
put them in the pool. This probe asks the only question that matters before any
build: *would embeddings actually retrieve those turns?*

Method, per gold turn: locate it in the store, rank it under production BM25
(`_message_fts_search`), then rank it by cosine against a fresh embedding of
every user/assistant turn in the same store, then under RRF fusion of the two.
The gap set is gold that BM25 cannot deliver inside the production cut.

CONTROL (mandatory — the diagnostic-controls rule): the same vector ranking runs
over gold turns from CORRECT answers that BM25 *does* deliver. If the vector arm
cannot rank those either, the instrument is broken and the gap-set number means
nothing. Read the control FIRST.

Verdict thresholds, fixed here so the read can't drift after seeing the number:

  vector/hybrid recall@k on the gap set  ≥40%  -> real lever, worth building
                                        15-40% -> marginal; weigh against the
                                                  write cost of a message-vector
                                                  tier before committing
                                        <15%   -> this embedding model does not
                                                  bridge the paraphrase gap; the
                                                  architectural story closes and
                                                  the residual is the reader
  control recall@k                      <80%   -> INSTRUMENT BROKEN, do not read

Costs nothing but local embedding calls: it opens the stores read-only, runs no
LLM, and writes nothing. Embeddings come from the SAME env-driven client L1 wired
(the local FastEmbed ONNX server Hermes uses), so a positive result is a
production-faithful one.

Usage:
  python locomo_message_vector_probe.py RESULTS.json --data data/locomo10.json \
      --db-dir <same dir the run used> [--top-k 15] [--control 25] [--out probe.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from locomo_adapter import load_locomo_data

# Production BM25 over raw turns — the exact tier this probe is testing against.
from hymem.query.augment import _message_fts_search

# RRF constant: the standard 60. Only the ordering matters here, not the scale.
_RRF_K = 60
_EMBED_BATCH = 64


def _cosine(a: list[float], b: list[float]) -> float:
    """Same contract as the dreaming layer's _cosine (0.0 on dim mismatch)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _locate(con: sqlite3.Connection, turn: str) -> tuple[int, str] | None:
    """Find the gold turn's messages row by a prefix LIKE, mirroring
    locomo_index_probe.probe so the two probes agree on what "the gold turn" is."""
    key = " ".join(turn.split()[:6])
    row = con.execute(
        "SELECT id, role FROM messages WHERE content LIKE ? LIMIT 1", (f"%{key}%",)
    ).fetchone()
    return (row[0], row[1]) if row else None


def _bm25_ranks(con: sqlite3.Connection, question: str, depth: int) -> list[int]:
    """Production BM25 message ranking for a question: message_ids, best first."""
    return [h.message_id for h in _message_fts_search(con, question, top_k=depth)]


class TurnVectors:
    """Embeddings for every user/assistant turn in one store, built once and
    reused across that conversation's questions (the same store serves many)."""

    def __init__(self, con: sqlite3.Connection, embedder) -> None:
        rows = con.execute(
            "SELECT id, content FROM messages WHERE role IN ('user','assistant') "
            "ORDER BY id"
        ).fetchall()
        self.ids: list[int] = [r[0] for r in rows]
        texts: list[str] = [r[1] for r in rows]
        self.vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            self.vectors.extend(embedder.embed(texts[i:i + _EMBED_BATCH]))

    def ranks(self, query_vec: list[float]) -> list[int]:
        """message_ids ordered by descending cosine against the query."""
        scored = [
            (_cosine(query_vec, v), mid)
            for mid, v in zip(self.ids, self.vectors)
        ]
        scored.sort(key=lambda t: -t[0])
        return [mid for _, mid in scored]

    def score_of(self, query_vec: list[float], message_id: int) -> float:
        try:
            idx = self.ids.index(message_id)
        except ValueError:
            return 0.0
        return _cosine(query_vec, self.vectors[idx])


def _rank_of(ranked: list[int], message_id: int) -> int | None:
    """1-indexed rank, or None when absent."""
    try:
        return ranked.index(message_id) + 1
    except ValueError:
        return None


def _rrf(bm25: list[int], vec: list[int], depth: int) -> list[int]:
    """Reciprocal-rank fusion of the two rankings — the shape a real hybrid
    message tier would take, so the number reflects an implementable design
    rather than a vector-only fantasy."""
    scores: dict[int, float] = {}
    for ranking in (bm25[:depth], vec[:depth]):
        for i, mid in enumerate(ranking, 1):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (_RRF_K + i)
    return [mid for mid, _ in sorted(scores.items(), key=lambda t: -t[1])]


def _build_embedder():
    """The env-driven production client (L1's local FastEmbed server)."""
    try:
        from hymem.contrib.openai_embedding_client import (
            OpenAICompatibleEmbeddingClient,
        )
        client = OpenAICompatibleEmbeddingClient()
        probe = client.embed(["warmup"])
        if not probe or not probe[0]:
            raise RuntimeError("embed() returned nothing")
        return client, len(probe[0])
    except Exception as exc:
        sys.exit(
            f"error: no usable embedding client ({exc}).\n"
            "This probe needs the same endpoint L1 wired — set "
            "HYMEM_EMBEDDING_BASE_URL / HYMEM_EMBEDDING_MODEL (the local "
            "FastEmbed ONNX server, paraphrase-multilingual-MiniLM-L12-v2 on "
            ":8766) and re-run. Nothing else in the probe costs anything."
        )


def _measure(rows: list[dict], ev_map: dict, db_dir: Path, embedder,
             *, depth: int, top_k: int, want_gap: bool) -> list[dict]:
    """Measure one population. `want_gap` selects the vocabulary-gap set (gold
    BM25 cannot deliver inside `top_k`); otherwise the control set (gold BM25
    DOES deliver)."""
    out: list[dict] = []
    cache: dict[str, TurnVectors] = {}
    for r in rows:
        db = db_dir / r["conv_id"] / "hymem.sqlite"
        if not db.exists():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        # _message_fts_search reads rows by column name, so the Row factory is
        # required — with tuples it only fails once a query actually matches,
        # which a vocabulary-gap question never does.
        con.row_factory = sqlite3.Row
        try:
            bm25 = _bm25_ranks(con, r["question"], depth)
            for eid in (r.get("evidence") or []):
                hit = ev_map.get(r["conv_id"], {}).get(eid)
                if not hit:
                    continue
                _, turn = hit
                found = _locate(con, turn)
                if found is None:
                    continue  # NOT INGESTED — a defect the store probe already closed
                mid, role = found
                bm25_rank = _rank_of(bm25, mid)
                delivered = bm25_rank is not None and bm25_rank <= top_k
                if delivered == want_gap:
                    continue

                if r["conv_id"] not in cache:
                    cache[r["conv_id"]] = TurnVectors(con, embedder)
                tv = cache[r["conv_id"]]
                qvec = embedder.embed([r["question"]])[0]
                vec_ranked = tv.ranks(qvec)
                out.append({
                    "id": r["id"], "conv_id": r["conv_id"], "evidence_id": eid,
                    "question": r["question"], "turn": turn, "role": role,
                    "message_id": mid,
                    "bm25_rank": bm25_rank,
                    "vector_rank": _rank_of(vec_ranked, mid),
                    "hybrid_rank": _rank_of(_rrf(bm25, vec_ranked, depth), mid),
                    "cosine": round(tv.score_of(qvec, mid), 4),
                    "n_turns": len(tv.ids),
                })
        finally:
            con.close()
    return out


def _recall(rows: list[dict], field: str, k: int) -> tuple[int, int]:
    hit = sum(1 for r in rows if r[field] is not None and r[field] <= k)
    return hit, len(rows)


def _pct(hit: int, total: int) -> float:
    return round(100.0 * hit / total, 1) if total else 0.0


def _usable(rows: list[dict], top_k: int) -> tuple[list[dict], int]:
    """Drop rows whose whole corpus fits inside the cut. In a store with fewer
    user/assistant turns than `top_k`, EVERY present turn ranks within top_k by
    construction, so the vector arm scores 100% while measuring nothing — the
    "confident constant when broken" failure mode. Counted and reported, never
    silently averaged in."""
    keep = [r for r in rows if r["n_turns"] > top_k]
    return keep, len(rows) - len(keep)


def _report(gap: list[dict], control: list[dict], top_k: int, limit: int) -> str:
    lines: list[str] = []
    gap, gap_vacuous = _usable(gap, top_k)
    control, control_vacuous = _usable(control, top_k)

    def block(title: str, rows: list[dict], vacuous: int) -> None:
        lines.append(f"\n{'=' * 72}\n{title} — n={len(rows)}"
                     + (f" (+{vacuous} vacuous: corpus ≤ cut, excluded)" if vacuous else "")
                     + f"\n{'=' * 72}")
        if not rows:
            lines.append("  (empty — nothing to read)")
            return
        for field, label in (("bm25_rank", "BM25"),
                             ("vector_rank", "vector"),
                             ("hybrid_rank", "hybrid (RRF)")):
            cells = []
            for k in (5, top_k, 30):
                h, n = _recall(rows, field, k)
                cells.append(f"@{k}: {_pct(h, n):>5}% ({h}/{n})")
            lines.append(f"  {label:<13} " + "   ".join(cells))
        cos = [r["cosine"] for r in rows]
        lines.append(f"  cosine(question, gold): median "
                     f"{sorted(cos)[len(cos) // 2]:.3f}  "
                     f"min {min(cos):.3f}  max {max(cos):.3f}")

    block("CONTROL — gold BM25 already delivers (read this FIRST)", control,
          control_vacuous)
    ch, cn = _recall(control, "vector_rank", top_k)
    control_pct = _pct(ch, cn)
    if cn and control_pct < 80.0:
        lines.append(
            f"\n  ** INSTRUMENT BROKEN: control vector recall@{top_k} = "
            f"{control_pct}% (< 80%). The embedding path cannot rank gold it "
            "should find easily — wrong model, wrong dim, or a text mismatch. "
            "Do NOT read the gap set below. **"
        )

    block("GAP SET — gold BM25 cannot deliver inside the production cut", gap,
          gap_vacuous)

    gh, gn = _recall(gap, "vector_rank", top_k)
    hh, hn = _recall(gap, "hybrid_rank", top_k)
    best = max(_pct(gh, gn), _pct(hh, hn))
    lines.append(f"\n{'-' * 72}\nVERDICT (thresholds fixed in the docstring)")
    lines.append(f"  best gap-set recall@{top_k}: {best}% "
                 f"(vector {_pct(gh, gn)}%, hybrid {_pct(hh, hn)}%)")
    if not gap:
        lines.append("  → UNREADABLE: every gap row was vacuous (corpus ≤ cut) or the "
                     "set is empty. Re-run against stores whose turn count exceeds "
                     f"--top-k ({top_k}); a small store cannot discriminate.")
    elif cn and control_pct < 80.0:
        lines.append("  → UNREADABLE: fix the instrument (control failed) and re-run.")
    elif best >= 40.0:
        lines.append("  → REAL LEVER: a vector path over raw turns recovers gold no "
                     "aperture can reach. Build it probe-gated; expect ~nothing on "
                     "LME (its gold is already BM25-reachable) and judge on "
                     "LoCoMo/BEAM.")
    elif best >= 15.0:
        lines.append("  → MARGINAL: weigh the recovery against a message-vector "
                     "tier's write cost (embed every turn at ingest) before "
                     "committing. Consider whether the recovered turns are the ones "
                     "that actually change answers.")
    else:
        lines.append("  → CLOSED: this embedding model does not bridge the "
                     "paraphrase gap. The LoCoMo residual is not retrievable by "
                     "adding vectors over raw turns; the remaining story is the "
                     "reader (P0 measured ~80% architectural).")

    worst = sorted(gap, key=lambda r: -(r["vector_rank"] or 10**9))[:limit]
    if worst:
        lines.append(f"\n{'-' * 72}\nWORST GAP CASES (vector still can't reach them)")
        for r in worst:
            lines.append(
                f"\n  [{r['id']} {r['evidence_id']}] bm25={r['bm25_rank']} "
                f"vector={r['vector_rank']} hybrid={r['hybrid_rank']} "
                f"cos={r['cosine']} of {r['n_turns']} turns"
            )
            lines.append(f"    Q: {r['question'][:150]}")
            lines.append(f"    gold [{r['role']}]: {r['turn'][:150]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", help="the --out file whose misses were audited")
    ap.add_argument("--data", required=True)
    ap.add_argument("--db-dir", required=True, help="the SAME --db-dir the run used")
    ap.add_argument("--user-speaker", choices=["a", "b"], default="a")
    ap.add_argument("--ids", default=None,
                    help="comma-separated question ids (default: every non-cat-5 miss)")
    ap.add_argument("--top-k", type=int, default=15,
                    help="the production message cut the recall is measured at "
                         "(default 15 = message_fts_top_k)")
    ap.add_argument("--depth", type=int, default=200,
                    help="BM25 candidates pulled per question when ranking (default 200)")
    ap.add_argument("--control", type=int, default=25,
                    help="how many CORRECT-answer questions to draw the control from "
                         "(0 disables — but then the gap number is uninterpretable)")
    ap.add_argument("--limit", type=int, default=8, help="worst cases to print")
    ap.add_argument("--out", type=Path, help="write the full per-turn rows as JSON")
    args = ap.parse_args()

    all_rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if args.ids:
        keep = {s.strip() for s in args.ids.split(",")}
        misses = [r for r in all_rows if r["id"] in keep]
    else:
        misses = [r for r in all_rows if not r["correct"] and r["category"] != 5]
    hits = [r for r in all_rows if r.get("correct") and r["category"] != 5]
    if not misses:
        sys.exit("Nothing to probe.")

    ev_map = {c["id"]: c["evidence_map"]
              for c in load_locomo_data(args.data, user_speaker=args.user_speaker)}
    embedder, dim = _build_embedder()
    db_dir = Path(args.db_dir)
    print(f"[embed] {dim}-dim client ready; probing {len(misses)} misses "
          f"(+{min(args.control, len(hits))} control)", file=sys.stderr)

    gap = _measure(misses, ev_map, db_dir, embedder,
                   depth=args.depth, top_k=args.top_k, want_gap=True)
    control = _measure(hits[:args.control], ev_map, db_dir, embedder,
                       depth=args.depth, top_k=args.top_k, want_gap=False)

    report = _report(gap, control, args.top_k, args.limit)
    print(report)

    if args.out:
        args.out.write_text(json.dumps(
            {"top_k": args.top_k, "embedding_dim": dim,
             "gap": gap, "control": control}, indent=2), encoding="utf-8")
        print(f"[out] {args.out}", file=sys.stderr)

    tally = Counter(
        "reached" if (r["hybrid_rank"] or 10**9) <= args.top_k else "unreached"
        for r in gap
    )
    return 0 if tally["reached"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
