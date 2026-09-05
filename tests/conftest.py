from __future__ import annotations

import json
from pathlib import Path

import pytest

from hymem import HyMem, HyMemConfig, StubEmbeddingClient
from hymem.extraction.llm import StubLLMClient


@pytest.fixture
def cfg(tmp_path: Path) -> HyMemConfig:
    return HyMemConfig(root=tmp_path)


@pytest.fixture
def stub_llm() -> StubLLMClient:
    return StubLLMClient(default="[]")


@pytest.fixture
def embed_stub() -> StubEmbeddingClient:
    return StubEmbeddingClient()


@pytest.fixture
def hy(cfg: HyMemConfig, stub_llm: StubLLMClient):
    instance = HyMem(cfg, llm=stub_llm)
    yield instance
    instance.close()


@pytest.fixture
def hy_agg(cfg: HyMemConfig, stub_llm: StubLLMClient):
    """HyMem in MR replace-mode with a wide counting cap (200) so aggregation
    tests see the aggregate's own evidence turns (distinct deduped user turns
    with enumerates_items flags) as `message_hits`. `mr_aggregate_additive=False`
    pins the legacy path these tests assert on; the additive default (relevance
    retrieval + a layered count) is covered separately."""
    from dataclasses import replace

    instance = HyMem(
        replace(cfg, message_fts_aggregate_cap=200, mr_aggregate_additive=False),
        llm=stub_llm,
    )
    yield instance
    instance.close()


@pytest.fixture
def hy_with_embed(
    cfg: HyMemConfig, stub_llm: StubLLMClient, embed_stub: StubEmbeddingClient
):
    instance = HyMem(cfg, llm=stub_llm, embedding_client=embed_stub)
    yield instance
    instance.close()


def seed_edge(
    conn,
    subject: str,
    predicate: str,
    obj: str,
    *,
    pos: int = 1,
    neg: int = 0,
    days_ago: int = 0,
    derived: int = 0,
    status: str = "active",
) -> None:
    """Insert a knowledge_graph edge directly, with last_seen `days_ago` in the past."""
    conn.execute(
        """INSERT INTO knowledge_graph
           (subject_canonical, predicate, object_canonical, pos_evidence,
            neg_evidence, last_seen, last_reinforced, status, derived)
           VALUES (?, ?, ?, ?, ?, datetime('now', ?), datetime('now', ?), ?, ?)""",
        (
            subject, predicate, obj, pos, neg,
            f"-{days_ago} days", f"-{days_ago} days", status, derived,
        ),
    )


class PromptSourceAwareStub(StubLLMClient):
    """Fill legacy canned triples from an exact source record in the prompt.

    Most integration fixtures predate claim-level provenance and intentionally
    describe only the semantic triple.  At request time the extraction prompt
    contains the authoritative, one-line JSON source records, so choosing the
    final record keeps those fixtures useful without weakening production
    validation or inventing an id outside the exact input slice.  Tests that
    exercise mixed-source attribution provide ``source_message_id`` explicitly
    and are left untouched.
    """

    def complete(self, request):
        raw = super().complete(request)
        if "source_message_id (integer)" not in request.system:
            return raw
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw
        if not isinstance(payload, dict) or not isinstance(payload.get("triples"), list):
            return raw

        source_ids: list[int] = []
        for line in request.user.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            source_id = record.get("source_message_id")
            if isinstance(source_id, int) and not isinstance(source_id, bool):
                source_ids.append(source_id)
        if not source_ids:
            return raw

        changed = False
        triples = []
        for original in payload["triples"]:
            if not isinstance(original, dict) or "source_message_id" in original:
                triples.append(original)
                continue
            item = dict(original)
            item["source_message_id"] = source_ids[-1]
            triples.append(item)
            changed = True
        if not changed:
            return raw
        return json.dumps({**payload, "triples": triples})


def make_routed_llm(triples: list[dict], markers: list[dict]) -> StubLLMClient:
    """Stub for the merged per-chunk extraction call.

    Phase 1 now issues a SINGLE call whose prompt returns a JSON object with
    both "triples" and "markers". The combined prompt contains both distinctive
    substrings ("structured technical relationships" and "EXPLICIT behavioral
    signals"), so a single fixture keyed on either routes to the combined object.
    The separate-key fixtures are retained so any code still issuing the old
    standalone triple/marker prompts continues to route correctly.
    """
    combined = json.dumps({"triples": triples, "markers": markers})
    return PromptSourceAwareStub(
        fixtures={
            # Combined chunk-extraction prompt -> object with both keys.
            "single pass": combined,
            # Legacy standalone prompts (extract_triples / extract_markers).
            "structured technical relationships": json.dumps(triples),
            "EXPLICIT behavioral signals": json.dumps(markers),
        },
        default="[]",
    )
