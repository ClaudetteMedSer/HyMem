"""Offline tests for the LongMemEval adapter's config PINS.

An adapter that only sets a lever's True leg inherits the library default on the
False leg, so the day that default flips, every run silently changes arm while
still being compared to a baseline that ran the other way. That is not
hypothetical here: `aggregation_nodes_enabled` flipped False -> True on
2026-08-26, `beam_adapter` and `msc_adapter` were pinned the same day, and this
adapter -- the one carrying the frozen 68.4 baseline -- was missed for five
days while the written record said all three were pinned.

So both levers are asserted in BOTH positions, and the aggregation test asserts
the pin is doing work rather than agreeing with the default by luck.

Skipped where `requests` is absent (the adapter imports it at module scope);
it runs on the box, which is where this suite is authoritative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from longmemeval_adapter import HyMemAdapter  # noqa: E402

from hymem.config import HyMemConfig  # noqa: E402


def _opened(tmp_path: Path, **kwargs) -> HyMemConfig:
    adapter = HyMemAdapter(tmp_path / "hymem.sqlite", api_key="unused", **kwargs)
    adapter.open()
    return adapter.hy.config


@pytest.mark.parametrize("enabled", [False, True])
def test_the_granularity_lever_reaches_the_config_both_ways(tmp_path, enabled):
    """Plan C's LME non-regression guard cannot run without this.

    Before it existed, `episode_granularity_enabled` appeared nowhere under
    benchmarks/, so the granular arm of the guard was unreachable from the CLI
    and the guard would have measured the blob prompt twice -- a clean null
    produced by an instrument that never touched the lever.
    """
    cfg = _opened(tmp_path, episode_granularity=enabled)
    assert cfg.episode_granularity_enabled is enabled


def test_the_aggregation_pin_holds_against_the_library_default(tmp_path):
    """The pin this adapter was missing, and the reason to assert it here.

    The second assertion is the one that matters: it establishes that the
    library default DISAGREES with the pin, which is what makes the first
    assertion evidence of a pin rather than of an inherited default. If the
    library default ever moves back to False this test SHOULD fail -- not
    because the pin broke, but because the check would otherwise keep reading
    green while testing nothing.
    """
    assert _opened(tmp_path, aggregation_nodes=False).aggregation_nodes_enabled is False
    assert HyMemConfig(root=tmp_path).aggregation_nodes_enabled is True
    assert _opened(tmp_path / "on", aggregation_nodes=True).aggregation_nodes_enabled is True
