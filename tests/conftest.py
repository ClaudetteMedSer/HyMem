from __future__ import annotations

import json
from pathlib import Path

import pytest

from hymem import HyMem, HyMemConfig
from hymem.extraction.llm import StubLLMClient


@pytest.fixture
def cfg(tmp_path: Path) -> HyMemConfig:
    return HyMemConfig(root=tmp_path)


@pytest.fixture
def stub_llm() -> StubLLMClient:
    return StubLLMClient(default="[]")


@pytest.fixture
def hy(cfg: HyMemConfig, stub_llm: StubLLMClient):
    instance = HyMem(cfg, llm=stub_llm)
    yield instance
    instance.close()


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
