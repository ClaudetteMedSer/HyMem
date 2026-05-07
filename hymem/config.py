from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HyMemConfig:
    root: Path
    """Directory holding hymem.sqlite, MEMORY.md, USER.md."""

    salience_min_chars: int = 80
    """Minimum chunk size before extraction is attempted."""

    fts_top_k: int = 5
    graph_top_k_per_entity: int = 3

    decay_window_days: int = 30
    decay_factor: float = 0.9
    retract_threshold: float = 0.15

    profile_max_entries: int = 16
    insights_max_entries: int = 12

    prompt_version: str = "v1"

    @property
    def db_path(self) -> Path:
        return self.root / "hymem.sqlite"

    @property
    def memory_md_path(self) -> Path:
        return self.root / "MEMORY.md"

    @property
    def user_md_path(self) -> Path:
        return self.root / "USER.md"
