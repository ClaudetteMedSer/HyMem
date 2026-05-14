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


def make_routed_llm(triples: list[dict], markers: list[dict]) -> StubLLMClient:
    """Stub that routes triple prompts to one payload and marker prompts to another.

    Distinguishes via unique strings in the respective system prompts.
    """
    return StubLLMClient(
        fixtures={
            "structured technical relationships": json.dumps(triples),
            "EXPLICIT behavioral signals": json.dumps(markers),
        },
        default="[]",
    )
