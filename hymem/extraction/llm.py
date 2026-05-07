from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    response_format: str = "json"  # "json" | "text"
    max_tokens: int = 1024
    temperature: float = 0.0


class LLMClient(Protocol):
    """Hermes wires whatever local LLM it wants behind this interface."""

    def complete(self, request: LLMRequest) -> str: ...


@dataclass
class StubLLMClient:
    """Returns canned responses keyed by prompt substring.

    Fixture keys are matched against the concatenation of system + user, so tests
    can route on either. `default` is returned for any unmatched prompt;
    if `default` is None, unmatched prompts raise so missing fixtures are caught
    loudly.
    """

    fixtures: dict[str, str] = field(default_factory=dict)
    default: str | None = None
    calls: list[LLMRequest] = field(default_factory=list)

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        haystack = request.system + "\n" + request.user
        for needle, response in self.fixtures.items():
            if needle in haystack:
                return response
        if self.default is not None:
            return self.default
        raise LookupError(
            f"StubLLMClient: no fixture matched. user prompt began with: "
            f"{request.user[:120]!r}"
        )
