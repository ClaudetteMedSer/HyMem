"""Fail-closed policy for model identifiers used by live LLM clients.

Historical benchmark artifacts are data and must remain readable.  This module
therefore does not rewrite or scan stored provenance; callers invoke the policy
only when a model is about to become an active execution dependency.
"""

from __future__ import annotations

from hymem.contrib.implementation_identity import import_time_source_sha256

RECOMMENDED_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEPRECATED_DEEPSEEK_ALIASES = frozenset({
    "deepseek-chat",
    "deepseek-reasoner",
})


class DeprecatedModelAliasError(ValueError):
    """Raised before execution when a mutable retired model alias is selected."""


def deprecated_deepseek_alias(model: object) -> str | None:
    """Return the canonical retired alias named by *model*, if any.

    Matching is deliberately exact after case-folding and surrounding-space
    normalization.  The two provider-qualified forms used by this repository
    and common OpenAI-compatible gateways are recognized as well:
    ``deepseek:deepseek-chat`` and ``deepseek/deepseek-chat``.  Versioned model
    ids such as ``deepseek-chat-v4`` are not aliases and are never rejected by
    substring inference.
    """

    if not isinstance(model, str):
        return None
    candidate = model.strip().casefold()
    if candidate in DEPRECATED_DEEPSEEK_ALIASES:
        return candidate

    # ``provider:model`` is a routing envelope: whatever provider is selected,
    # an exact retired model id in its model slot remains an active alias.
    qualified_candidate = candidate
    if ":" in candidate:
        _provider, qualified_candidate = candidate.split(":", 1)
        qualified_candidate = qualified_candidate.strip()
        if qualified_candidate in DEPRECATED_DEEPSEEK_ALIASES:
            return qualified_candidate

    # OpenAI-compatible gateways commonly use the vendor/model form.
    if "/" in qualified_candidate:
        vendor, qualified = qualified_candidate.split("/", 1)
        if (
            vendor.strip() == "deepseek"
            and qualified.strip() in DEPRECATED_DEEPSEEK_ALIASES
        ):
            return qualified.strip()
    return None


def require_active_model(model: object, *, role: str = "LLM") -> None:
    """Reject retired DeepSeek aliases before a live execution path starts."""

    alias = deprecated_deepseek_alias(model)
    if alias is None:
        return
    raise DeprecatedModelAliasError(
        f"retired DeepSeek model alias {alias!r} is not allowed for active "
        f"{role} execution; use {RECOMMENDED_DEEPSEEK_MODEL!r} with thinking "
        "set to 'auto' or 'disabled'"
    )


MODEL_POLICY_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)
