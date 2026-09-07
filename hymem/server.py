"""MCP server for HyMem.

Exposes twelve tools to the Hermes Agent platform:
  hymem_capture    — log a full conversation at once + optionally dream (preferred)
  hymem_log        — log one conversational turn (fallback for turn-by-turn use)
  hymem_dream      — run a dreaming cycle (extract, consolidate, decay)
  hymem_augment    — retrieve graph facts + FTS context for a user message
  hymem_ask        — ask the memory a question, get one LLM-reasoned answer
  hymem_profile    — return USER.md (behavioral profile) + MEMORY.md (project insights)
  hymem_digest     — return the standing whole-store memory digest (RAPTOR root)
  hymem_alias      — register a surface-form alias for an entity
  hymem_retract    — retract a wrongly extracted knowledge graph edge
  hymem_add_rule   — record a standing behavioral rule (always_on / contextual)
  hymem_list_rules — list the active standing rules
  hymem_suggest_rules — propose inferred standing rules to review (no auto-add)

Run via the installed entry point:
    hymem-server

Or directly:
    python -m hymem.server

Configuration is entirely through environment variables (see README or
hymem/contrib/openai_client.py for the full list).

Key variables:
    HYMEM_LLM_API_KEY        API key for the extraction LLM (or DEEPSEEK_API_KEY)
    HYMEM_LLM_BASE_URL       Base URL (default: https://api.deepseek.com)
    HYMEM_LLM_MODEL          Model name (default: deepseek-v4-flash)
                             Retired deepseek-chat/deepseek-reasoner aliases
                             are rejected before the server opens its store.
    HYMEM_LLM_THINKING       Thinking-body policy (default: auto)
    HYMEM_EMBEDDING_API_KEY  API key for an explicitly configured remote embedder
                             (OPENAI_API_KEY is used only for api.openai.com)
    HYMEM_EMBEDDING_BASE_URL HTTPS OpenAI-compatible endpoint (HTTP only loopback)
    HYMEM_EMBEDDING_MODEL    Embedding model (with matching HYMEM_EMBEDDING_DIM)
    HYMEM_EMBEDDING_DIM      Declared vector dimension
    HYMEM_EMBEDDING_TIMEOUT_SECONDS
                             Remote request timeout (default: 10; SDK retries off)
    HYMEM_ROOT               Directory for hymem.sqlite, MEMORY.md, USER.md
                             (default: ~/.hermes)
    HYMEM_AGGREGATION_NODES_ENABLED
                             RAPTOR aggregation/digest master switch at dream
                             time (default: on). Set false to opt out.
    HYMEM_AGGREGATION_DIGEST_ENABLED
                             Override the digest sub-switch independently (default:
                             on whenever aggregation is enabled). Set false to
                             measure level-0 node-build cost in isolation.

With no embedding configuration, the server uses a deterministic, dependency-
free local feature-hash backend (model identity and lexical quality are exposed
in query status). An explicit localhost endpoint needs no real API key. An
incomplete/unavailable remote configuration falls back without launching a
service at import or construction time.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

# Startup, env-var resolution, and the shared singleton live in hymem.bootstrap.
# Re-exported here under the historical names used by tests and tool helpers.
from hymem.bootstrap import (
    get_instance as _get_hy,
    set_instance as set_hy,
    shutdown_instance as _shutdown_hy,
)
from hymem.query.augment import format_graph_fact_sources
from hymem.dreaming.runner import (
    DREAM_REPORT_BOOLEAN_GATE_FIELDS,
    DREAM_REPORT_COUNT_FIELDS,
    DREAM_REPORT_ERROR_FIELDS,
    DREAM_REPORT_FIELD_NAMES,
    DREAM_REPORT_NULLABLE_COUNT_FIELDS,
    DREAM_REPORT_TEXT_FIELDS,
)
from hymem.dreaming.status import (
    DREAM_STATUS_BLOCKING_COUNT_FIELDS,
    DREAM_STATUS_BOOLEAN_GATE_FIELDS,
    DREAM_STATUS_DIAGNOSTIC_BOOLEAN_FIELDS,
    DREAM_STATUS_DIAGNOSTIC_CONFIG_VERSION_FIELDS,
    DREAM_STATUS_DIAGNOSTIC_COUNT_FIELDS,
    DREAM_STATUS_DIAGNOSTIC_NULLABLE_TEXT_FIELDS,
    DREAM_STATUS_HEALTH_DETAIL_FIELDS,
    DREAM_STATUS_RECOGNIZED_HEALTH_FIELDS,
    DREAM_STATUS_SCHEMA_VERSION,
    is_health_like_dream_status_field,
)
from hymem.dreaming.lossless import COVERAGE_INTEGRITY_CONFIG_VERSION


def _get_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "mcp package required: pip install 'hymem[server]'"
        ) from e
    return FastMCP("hymem")


mcp = None

_MAX_REPORTED_COUNT = (1 << 63) - 1
_AGGREGATION_CONFIG_VERSION_PREFIX = "aggregation-build-config-v1:"
_PHASE1_GENERATION_KEY_RE = re.compile(
    r"hymem-phase1-generation-v1:[0-9a-f]{64}\Z"
)


@dataclass(frozen=True)
class _DreamCompletionAssessment:
    state: str
    blockers: tuple[str, ...] = ()


def _is_nonnegative_int(value: object) -> bool:
    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_REPORTED_COUNT
    )


def _report_skipped_hint(report) -> bool:
    try:
        return getattr(report, "skipped_locked", None) is True
    except Exception:
        return False


def _bounded_field_names(field_names: set[str]) -> str:
    """Render schema drift without allowing an unbounded MCP response."""

    ordered = sorted(field_names)
    shown = ordered[:6]
    rendered = ",".join(name[:80] for name in shown)
    if len(ordered) > len(shown):
        rendered += f",...(+{len(ordered) - len(shown)})"
    return rendered


def _bounded_blocker_text(blockers: tuple[str, ...]) -> str:
    """Keep an actionable prefix while deduplicating and bounding output."""

    unique = tuple(dict.fromkeys(blockers))
    shown = unique[:10]
    rendered = ", ".join(shown)
    if len(unique) > len(shown):
        rendered += f", ... (+{len(unique) - len(shown)} more)"
    return rendered


def _reason_counts_match(value: object, expected: int) -> bool:
    if type(value) is not dict:
        return False
    counts = list(value.values())
    return bool(
        all(isinstance(key, str) and key.strip() for key in value)
        and all(_is_nonnegative_int(count) and count > 0 for count in counts)
        and sum(counts) == expected
    )


def _is_aggregation_config_version(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.removeprefix(_AGGREGATION_CONFIG_VERSION_PREFIX)
    return bool(
        value.startswith(_AGGREGATION_CONFIG_VERSION_PREFIX)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _coverage_details_are_shaped(value: object) -> bool:
    if type(value) is not list or len(value) > 100:
        return False
    required = {
        "session_id",
        "config_version",
        "failure_reason",
        "occurrences",
        "first_detected_at",
        "last_detected_at",
    }
    for detail in value:
        if type(detail) is not dict or not required.issubset(detail):
            return False
        if not all(
            isinstance(detail[name], str) and detail[name].strip()
            for name in (
                "session_id",
                "config_version",
                "failure_reason",
                "first_detected_at",
                "last_detected_at",
            )
        ):
            return False
        if (
            not _is_nonnegative_int(detail["occurrences"])
            or detail["occurrences"] == 0
            or detail["config_version"] != COVERAGE_INTEGRITY_CONFIG_VERSION
        ):
            return False
    return True


def _assess_dream_completion(hy, report) -> _DreamCompletionAssessment:
    """Classify one run against a fresh, store-wide reported-health snapshot.

    Both objects are strict trust boundaries. Missing or malformed fields can
    never be interpreted as zero/healthy, because this text is exposed to
    agents that may otherwise stop scheduling recovery.
    """

    try:
        status = hy.dream_status()
    except Exception:
        return _DreamCompletionAssessment(
            (
                "skipped_unverified"
                if _report_skipped_hint(report)
                else "unverified"
            ),
            ("dream_status:unavailable",),
        )
    if type(status) is not dict:
        return _DreamCompletionAssessment(
            (
                "skipped_unverified"
                if _report_skipped_hint(report)
                else "unverified"
            ),
            ("dream_status:malformed",),
        )

    if status.get("dream_status_schema") != DREAM_STATUS_SCHEMA_VERSION:
        return _DreamCompletionAssessment(
            (
                "skipped_unverified"
                if _report_skipped_hint(report)
                else "unverified"
            ),
            ("status.dream_status_schema:missing_or_invalid",),
        )

    malformed: list[str] = []
    status_field_names = {
        name for name in status if isinstance(name, str)
    }
    if len(status_field_names) != len(status):
        malformed.append("status.health_schema:non_string_field")
    unknown_health_fields = {
        name for name in status_field_names
        if (
            is_health_like_dream_status_field(name)
            and name not in DREAM_STATUS_RECOGNIZED_HEALTH_FIELDS
        )
    }
    if unknown_health_fields:
        malformed.append(
            "status.health_schema:unknown="
            + _bounded_field_names(unknown_health_fields)
        )

    status_counts: dict[str, int] = {}
    for field_name in DREAM_STATUS_BLOCKING_COUNT_FIELDS:
        value = status.get(field_name)
        if not _is_nonnegative_int(value):
            malformed.append(f"status.{field_name}:missing_or_invalid")
        else:
            status_counts[field_name] = value

    status_flags: dict[str, bool] = {}
    for field_name in DREAM_STATUS_BOOLEAN_GATE_FIELDS:
        value = status.get(field_name)
        if not isinstance(value, bool):
            malformed.append(f"status.{field_name}:missing_or_invalid")
        else:
            status_flags[field_name] = value

    phase1_backlog_status = status.get("phase1_backlog_status")
    pending_chunks_authoritative = status.get(
        "pending_chunks_authoritative"
    )
    phase1_generation_key = status.get("phase1_generation_key")
    if phase1_backlog_status not in {
        "current_producer", "producer_unavailable",
    }:
        malformed.append("status.phase1_backlog_status:missing_or_invalid")
    if not isinstance(pending_chunks_authoritative, bool):
        malformed.append(
            "status.pending_chunks_authoritative:missing_or_invalid"
        )
    if phase1_generation_key is not None and (
        type(phase1_generation_key) is not str
        or _PHASE1_GENERATION_KEY_RE.fullmatch(phase1_generation_key) is None
    ):
        malformed.append("status.phase1_generation_key:missing_or_invalid")
    if (
        phase1_backlog_status == "current_producer"
        and (
            pending_chunks_authoritative is not True
            or phase1_generation_key is None
        )
    ) or (
        phase1_backlog_status == "producer_unavailable"
        and (
            pending_chunks_authoritative is not False
            or phase1_generation_key is not None
        )
    ):
        malformed.append("status.phase1_authority:inconsistent")

    diagnostic_counts: dict[str, int] = {}
    for field_name in DREAM_STATUS_DIAGNOSTIC_COUNT_FIELDS:
        value = status.get(field_name)
        if not _is_nonnegative_int(value):
            malformed.append(f"status.{field_name}:missing_or_invalid")
        else:
            diagnostic_counts[field_name] = value
    diagnostic_flags: dict[str, bool] = {}
    for field_name in DREAM_STATUS_DIAGNOSTIC_BOOLEAN_FIELDS:
        value = status.get(field_name)
        if not isinstance(value, bool):
            malformed.append(f"status.{field_name}:missing_or_invalid")
        else:
            diagnostic_flags[field_name] = value
    diagnostic_text: dict[str, str | None] = {}
    for field_name in DREAM_STATUS_DIAGNOSTIC_NULLABLE_TEXT_FIELDS:
        value = status.get(field_name)
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            malformed.append(f"status.{field_name}:missing_or_invalid")
        elif field_name not in status:
            malformed.append(f"status.{field_name}:missing_or_invalid")
        else:
            diagnostic_text[field_name] = value
    for field_name in DREAM_STATUS_DIAGNOSTIC_CONFIG_VERSION_FIELDS:
        value = diagnostic_text.get(field_name)
        if value is not None and not _is_aggregation_config_version(value):
            malformed.append(f"status.{field_name}:missing_or_invalid")

    for field_name in DREAM_STATUS_HEALTH_DETAIL_FIELDS:
        if field_name not in status:
            malformed.append(f"status.{field_name}:missing_or_invalid")
    terminal_loss_reasons = status.get("terminal_loss_reasons")
    if (
        "terminal_loss_reasons" in status
        and "terminal_loss_chunks" in status_counts
        and not _reason_counts_match(
            terminal_loss_reasons,
            status_counts["terminal_loss_chunks"],
        )
    ):
        malformed.append("status.terminal_loss_reasons:missing_or_invalid")
    coverage_failure_reasons = status.get(
        "coverage_integrity_failure_reasons"
    )
    if (
        "coverage_integrity_failure_reasons" in status
        and "coverage_integrity_failures" in status_counts
        and not _reason_counts_match(
            coverage_failure_reasons,
            status_counts["coverage_integrity_failures"],
        )
    ):
        malformed.append(
            "status.coverage_integrity_failure_reasons:missing_or_invalid"
        )
    if (
        "coverage_integrity_failure_details" in status
        and not _coverage_details_are_shaped(
            status.get("coverage_integrity_failure_details")
        )
    ):
        malformed.append(
            "status.coverage_integrity_failure_details:missing_or_invalid"
        )
    if (
        "coverage_integrity_failure_details_truncated" in status
        and not isinstance(
            status.get("coverage_integrity_failure_details_truncated"), bool
        )
    ):
        malformed.append(
            "status.coverage_integrity_failure_details_truncated:"
            "missing_or_invalid"
        )
    if (
        "coverage_integrity_config_version" in status
        and status.get("coverage_integrity_config_version")
        != COVERAGE_INTEGRITY_CONFIG_VERSION
    ):
        malformed.append(
            "status.coverage_integrity_config_version:missing_or_invalid"
        )

    coverage_count = status_counts.get("coverage_integrity_failures")
    coverage_details = status.get("coverage_integrity_failure_details")
    coverage_truncated = status.get(
        "coverage_integrity_failure_details_truncated"
    )
    if (
        coverage_count is not None
        and type(coverage_details) is list
        and isinstance(coverage_truncated, bool)
        and (
            len(coverage_details) != min(coverage_count, 100)
            or coverage_truncated is not (coverage_count > 100)
        )
    ):
        malformed.append("status.coverage_integrity_details:inconsistent")

    pending_aggregation = status_counts.get("pending_aggregation")
    active_aggregation_fields = (
        "aggregation_active_build_attempts",
        "aggregation_active_caught_exceptions",
        "aggregation_active_fusion_failures",
    )
    if (
        pending_aggregation == 0
        and all(
            field_name in diagnostic_counts
            for field_name in active_aggregation_fields
        )
        and any(
            diagnostic_counts[field_name] != 0
            for field_name in active_aggregation_fields
        )
    ):
        malformed.append("status.aggregation_active_diagnostics:inconsistent")
    if (
        "aggregation_active_caught_exceptions" in diagnostic_counts
        and "aggregation_total_caught_exceptions" in diagnostic_counts
        and diagnostic_counts["aggregation_active_caught_exceptions"]
        > diagnostic_counts["aggregation_total_caught_exceptions"]
    ):
        malformed.append("status.aggregation_exception_counts:inconsistent")
    if (
        "aggregation_active_fusion_failures" in diagnostic_counts
        and "aggregation_total_fusion_failures" in diagnostic_counts
        and diagnostic_counts["aggregation_active_fusion_failures"]
        > diagnostic_counts["aggregation_total_fusion_failures"]
    ):
        malformed.append("status.aggregation_fusion_counts:inconsistent")

    aggregation_enabled = diagnostic_flags.get("aggregation_enabled")
    aggregation_config = diagnostic_text.get("aggregation_config_version")
    if (
        aggregation_enabled is False
        and (aggregation_config is not None or pending_aggregation != 0)
    ) or (aggregation_enabled is True and aggregation_config is None):
        malformed.append("status.aggregation_configuration:inconsistent")

    success_metadata = (
        diagnostic_text.get("aggregation_last_success_config_version"),
        diagnostic_text.get("aggregation_last_success_at"),
    )
    if any(value is None for value in success_metadata) != all(
        value is None for value in success_metadata
    ):
        malformed.append("status.aggregation_success_metadata:inconsistent")
    if (
        aggregation_enabled is True
        and pending_aggregation == 0
        and success_metadata[0] != aggregation_config
    ):
        malformed.append("status.aggregation_success_metadata:stale")
    failure_metadata = (
        diagnostic_text.get("aggregation_last_failure_config_version"),
        diagnostic_text.get("aggregation_last_failure_kind"),
        diagnostic_text.get("aggregation_last_failure_at"),
    )
    if any(value is None for value in failure_metadata) != all(
        value is None for value in failure_metadata
    ):
        malformed.append("status.aggregation_failure_metadata:inconsistent")
    failure_kind = diagnostic_text.get("aggregation_last_failure_kind")
    if failure_kind not in {
        None,
        "exception",
        "fusion_failure",
        "exception_and_fusion",
    }:
        malformed.append("status.aggregation_last_failure_kind:invalid")

    report_counts: dict[str, int] = {}
    report_flags: dict[str, bool] = {}
    try:
        report_values = vars(report)
        if type(report_values) is not dict:
            raise TypeError("dream report has no attribute dictionary")
        report_field_names = set(report_values)
        required_report_fields = set(DREAM_REPORT_FIELD_NAMES)
        missing_report_fields = required_report_fields - report_field_names
        extra_report_fields = report_field_names - required_report_fields
        if missing_report_fields:
            malformed.append(
                "report.schema:missing="
                + _bounded_field_names(missing_report_fields)
            )
        if extra_report_fields:
            malformed.append(
                "report.schema:extra="
                + _bounded_field_names(extra_report_fields)
            )
        for field_name in (
            *DREAM_REPORT_COUNT_FIELDS,
            *DREAM_REPORT_ERROR_FIELDS,
        ):
            if field_name not in report_values:
                continue
            value = report_values.get(field_name)
            if not _is_nonnegative_int(value):
                malformed.append(f"report.{field_name}:missing_or_invalid")
            elif field_name in DREAM_REPORT_ERROR_FIELDS:
                report_counts[field_name] = value
        for field_name in DREAM_REPORT_NULLABLE_COUNT_FIELDS:
            if field_name not in report_values:
                continue
            value = report_values.get(field_name)
            if value is not None and not _is_nonnegative_int(value):
                malformed.append(f"report.{field_name}:missing_or_invalid")
        for field_name in DREAM_REPORT_TEXT_FIELDS:
            if field_name not in report_values:
                continue
            if not isinstance(report_values.get(field_name), str):
                malformed.append(f"report.{field_name}:missing_or_invalid")
        for field_name in DREAM_REPORT_BOOLEAN_GATE_FIELDS:
            if field_name not in report_values:
                continue
            value = report_values.get(field_name)
            if not isinstance(value, bool):
                malformed.append(f"report.{field_name}:missing_or_invalid")
            else:
                report_flags[field_name] = value
    except Exception:
        return _DreamCompletionAssessment(
            "unverified", ("dream_report:malformed",)
        )

    if malformed:
        return _DreamCompletionAssessment(
            (
                "skipped_unverified"
                if report_flags.get("skipped_locked") is True
                else "unverified"
            ),
            tuple(malformed),
        )

    blockers = [
        f"status.{field_name}={value}"
        for field_name, value in status_counts.items()
        if value > 0
    ]
    blockers.extend(
        f"report.{field_name}={value}"
        for field_name, value in report_counts.items()
        if value > 0
    )
    blockers.extend(
        f"report.{field_name}=true"
        for field_name in (
            "budget_exhausted",
            "extraction_provider_attempt_budget_exhausted",
        )
        if report_flags[field_name]
    )
    if phase1_backlog_status == "producer_unavailable":
        blockers.append("status.phase1_backlog_status=producer_unavailable")

    in_progress = status_flags["in_progress"]
    if report_flags["skipped_locked"]:
        if in_progress:
            return _DreamCompletionAssessment(
                "skipped_in_progress", tuple(blockers)
            )
        return _DreamCompletionAssessment(
            "skipped", ("report.skipped_locked=true", *blockers)
        )
    if in_progress:
        return _DreamCompletionAssessment(
            "in_progress", ("status.in_progress=true", *blockers)
        )
    if blockers:
        return _DreamCompletionAssessment("incomplete", tuple(blockers))
    return _DreamCompletionAssessment("complete")


def _safe_report_count(report, field_name: str) -> str:
    try:
        value = getattr(report, field_name, None)
    except Exception:
        return "unknown"
    return str(value) if _is_nonnegative_int(value) else "unknown"


def _dream_run_summary(report) -> str:
    return (
        f"{_safe_report_count(report, 'sessions_processed')} sessions, "
        f"{_safe_report_count(report, 'chunks_processed')} chunks newly completed "
        f"this run ({_safe_report_count(report, 'chunks_seen')} seen), "
        f"{_safe_report_count(report, 'triples_extracted')} triples, "
        f"{_safe_report_count(report, 'markers_extracted')} markers extracted"
    )


def _format_dream_completion(hy, report, *, targeted: bool) -> str:
    """Render the shared, fail-closed completion statement for MCP tools."""

    assessment = _assess_dream_completion(hy, report)
    if assessment.state == "complete":
        headline = (
            "targeted dreaming cycle finished cleanly — store-wide durable "
            "blockers were clear at the coherent post-run snapshot; later "
            "arrivals may reopen work"
            if targeted else
            "dreaming cycle finished cleanly — store-wide durable blockers "
            "were clear at the coherent post-run snapshot; later arrivals "
            "may reopen work"
        )
    elif assessment.state == "skipped_in_progress":
        headline = (
            "dreaming skipped (another cycle owns the lease) — "
            "store-wide indexing is in progress"
        )
    elif assessment.state == "in_progress":
        headline = (
            "dreaming run finished — store-wide indexing is in progress "
            "(another cycle now owns the lease)"
        )
    elif assessment.state == "skipped":
        headline = (
            "dreaming skipped (lease was busy) — "
            "store-wide reported indexing health is incomplete"
        )
    elif assessment.state in {"unverified", "skipped_unverified"}:
        headline = (
            (
                "dreaming skipped — "
                if assessment.state == "skipped_unverified"
                else "dreaming incomplete/unverified — "
            )
            + "store-wide reported indexing health is incomplete/unverified"
        )
    else:
        headline = (
            "dreaming incomplete — "
            "store-wide reported indexing health is incomplete"
        )
    if assessment.blockers:
        headline += "; blockers: " + _bounded_blocker_text(
            assessment.blockers
        )
    return f"{headline}; {_dream_run_summary(report)}"


# ── tool implementations (callable directly in tests) ────────────────────────

def _do_capture(session_id: str, messages: str, dream: bool = True) -> str:
    # Validate the complete envelope before lazy bootstrap can create a store,
    # clients, or source rows. Errors contain field/index diagnostics only.
    if not isinstance(session_id, str) or not session_id.strip():
        return "error: session_id must be a non-empty string"
    if not isinstance(dream, bool):
        return "error: dream must be a boolean"
    if not isinstance(messages, str):
        return "error: messages must be a JSON array"
    try:
        turns = json.loads(messages)
    except (json.JSONDecodeError, RecursionError):
        return "error: messages must be a valid JSON array"

    if not isinstance(turns, list):
        return "error: messages must be a JSON array"

    accepted: list[tuple[str, str]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            return f"error: messages[{index}] must be an object"
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not isinstance(role, str) or not isinstance(content, str):
            return f"error: messages[{index}].role and content must be strings"
        # Preserve the prior filtering policy for otherwise well-typed items.
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        if not content:
            continue
        accepted.append((role, content))

    hy = _get_hy()
    logged = len(hy.log_messages(session_id, accepted, close_session=True))

    if not dream:
        return f"logged {logged} turns for session {session_id!r}"

    report = hy.dream(session_ids=[session_id])
    completion = _format_dream_completion(hy, report, targeted=True)
    return f"logged {logged} turns for session {session_id!r}; {completion}"


def _do_log(session_id: str, role: str, content: str) -> str:
    _get_hy().log_message(session_id, role, content)
    return "logged"


def _do_dream() -> str:
    hy = _get_hy()
    report = hy.dream()
    return _format_dream_completion(hy, report, targeted=False)


def _do_augment(message: str) -> str:
    ctx = _get_hy().augment(message)
    parts: list[str] = []

    if ctx.graph_facts:
        lines = []
        for f in ctx.graph_facts:
            line = (
                f"- [edge {f.edge_id}] {f.subject} {f.predicate} {f.object} "
                f"(conf {f.confidence:.2f}, +{f.pos_evidence}/-{f.neg_evidence}"
            )
            if f.valid_at:
                line += f", valid since {f.valid_at}"
            line += ")"
            line += f" [sources: {format_graph_fact_sources(f)}]"
            lines.append(line)
        parts.append("**Structured knowledge (knowledge graph):**\n" + "\n".join(lines))

    if ctx.facts:
        lines = [
            f"- [{f.fact_date or 'undated'}] {f.text[:300]}"
            for f in ctx.facts
        ]
        parts.append("**Narrative facts (dream-verified events):**\n" + "\n".join(lines))

    if ctx.fts_hits:
        snippets = [f"[{h.session_id}] {h.text[:300]}" for h in ctx.fts_hits]
        parts.append("**Relevant past context (hybrid retrieval):**\n" + "\n".join(snippets))

    if ctx.message_hits:
        snippets = [f"[{h.session_id}/{h.role}] {h.text[:300]}" for h in ctx.message_hits]
        parts.append("**Relevant raw turns (hybrid retrieval):**\n" + "\n".join(snippets))

    return "\n\n".join(parts) if parts else ""


def _do_ask(question: str) -> str:
    return _get_hy().ask(question).answer


def _do_profile() -> str:
    hy = _get_hy()
    cfg = hy.config
    from hymem.dreaming.phase2 import authoritative_user_markdown

    user = authoritative_user_markdown(hy.read_conn, cfg)
    memory = cfg.memory_md_path.read_text(encoding="utf-8") if cfg.memory_md_path.exists() else ""
    parts: list[str] = []
    if user.strip():
        parts.append("=== USER PROFILE ===\n" + user.strip())
    if memory.strip():
        parts.append("=== PROJECT INSIGHTS ===\n" + memory.strip())
    return "\n\n".join(parts) if parts else "No profile or insights available yet."


def _do_digest() -> str:
    digest = _get_hy().digest()
    if digest is None:
        return (
            "No digest available yet — it is built at dream time once the "
            "aggregation layer (aggregation_nodes_enabled + "
            "aggregation_digest_enabled) has dreamed over at least one episode."
        )
    return digest.as_context_block()


def _do_alias(surface: str, canonical: str) -> str:
    _get_hy().register_alias(surface, canonical)
    return f"alias registered: {surface!r} → {canonical!r}"


def _do_retract(subject: str, predicate: str, object: str) -> str:
    ok = _get_hy().retract_edge(subject, predicate, object)
    return "retracted" if ok else "no matching active edge found"


def _do_add_rule(text: str, scope: str = "always_on", trigger_entities: str = "") -> str:
    triggers = [t.strip() for t in trigger_entities.split(",") if t.strip()] or None
    try:
        rid = _get_hy().add_rule(text, scope=scope, trigger_entities=triggers, source="user")
    except ValueError as e:
        return f"error: {e}"
    return f"rule #{rid} added (scope={scope})"


def _do_list_rules() -> str:
    active = _get_hy().rules()
    if not active:
        return "No standing rules set."
    lines = []
    for r in active:
        tag = (r.scope if r.scope == "always_on"
               else f"contextual({', '.join(r.trigger_entities)})")
        lines.append(f"#{r.id} [{tag}] {r.text}")
    return "\n".join(lines)


def _do_suggest_rules(limit: int = 10) -> str:
    try:
        cands = _get_hy().suggest_rules(limit=limit)
    except RuntimeError as e:
        return f"error: {e}"
    if not cands:
        return ("No rule candidates (no recent markers cleared the durability "
                "tagger). Suggestions read UNCONSOLIDATED markers — call after "
                "logging a session and before dreaming.")
    lines = ["Candidate standing rules — NOT added yet; adopt with hymem_add_rule:"]
    for c in cands:
        prov = f"{c.marker_count} marker(s)/{c.session_count} session(s), conf {c.confidence:.2f}"
        dup = "  [already active — would reinforce]" if c.already_active else ""
        lines.append(f"- {c.text}  ({prov}; kinds={','.join(c.kinds)}){dup}")
    return "\n".join(lines)


# ── MCP tool registration ─────────────────────────────────────────────────────

def hymem_capture(session_id: str, messages: str, dream: bool = True) -> str:
    """Log a full conversation and optionally run dreaming. Preferred over hymem_log.

    Call this ONCE at the end of every conversation instead of calling hymem_log
    after each individual turn. This is far more reliable because it requires only
    a single tool call per session rather than one per exchange.

    Arguments:
        session_id  — unique id for this conversation, e.g. "2026-05-10-db-migration"
        messages    — JSON array of {role, content} objects representing the full
                      conversation in order, e.g.:
                      '[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]'
        dream       — if true (default), run a dreaming cycle immediately after
                      logging so MEMORY.md and USER.md are updated right away.

    Returns a summary of what was logged and, if dream=true, what was extracted.
    """
    return _do_capture(session_id, messages, dream)


def hymem_log(session_id: str, role: str, content: str) -> str:
    """Log one conversational turn to HyMem.

    Call this after every user message and every assistant reply, using the same
    session_id throughout a conversation (e.g. today's date + a short topic slug).

    role must be one of: user, assistant, system, tool.
    """
    return _do_log(session_id, role, content)


def hymem_dream() -> str:
    """Run a full dreaming cycle.

    Processes all unprocessed session chunks: extracts knowledge triples and
    behavioural markers, updates ~/.hermes/MEMORY.md and ~/.hermes/USER.md,
    then decays stale graph edges. Call at the end of a session or when idle.

    Safe to call concurrently — a run-lock prevents overlapping cycles.
    Returns a short report of what was processed.
    """
    return _do_dream()


def hymem_augment(message: str) -> str:
    """Return structured knowledge and relevant past context for a user message.

    Performs entity/graph lookup plus lexical and semantic retrieval over past
    chunks and durable message occurrences. No query-time LLM call is made.
    Returns an empty string if no relevant context exists yet.
    """
    return _do_augment(message)


def hymem_ask(question: str) -> str:
    """Ask the memory store a question and get one reasoned answer.

    The dialectic counterpart to hymem_augment: instead of returning raw
    retrieval context for YOU to interpret, this runs the same retrieval and
    makes a single LLM call that synthesizes a grounded answer — quoting
    concrete values and dates, stating both sides of a contradiction (most
    recent statement wins), hedging low-confidence facts, and saying plainly
    when the memory does not contain the answer. Use it for direct questions
    about the user or past sessions ("what database does the user prefer?");
    use hymem_augment when you want the raw evidence tiers instead.
    """
    return _do_ask(question)


def hymem_profile() -> str:
    """Return the user's behavioral profile and project insights.

    Read USER.md (behavioral profile, auto-generated by HyMem) and MEMORY.md
    (project insights, auto-generated by HyMem) and return their combined
    content as a single labeled string. Use this once at session start to
    understand the user's preferences and the project's known structure
    before responding. For per-message context (relevant past chunks and
    graph facts), use hymem_augment instead.
    """
    return _do_profile()


def hymem_digest() -> str:
    """Return the standing whole-store memory digest.

    The digest is the root of HyMem's cross-session aggregation tree: one
    summary answering "what do you know about me?" across the entire store,
    rebuilt at dream time (never per query). Inject it as standing context at
    session start — it complements hymem_profile (behavioral profile) with a
    narrative of what the user has actually been working on. The footer states
    how many sessions it covers and when it was generated, so staleness is
    visible; re-fetch after hymem_dream to pick up a refreshed digest.
    Returns an explanatory message if no digest has been built yet.
    """
    return _do_digest()


def hymem_alias(surface: str, canonical: str) -> str:
    """Register that two names refer to the same entity.

    Example: hymem_alias('Postgres', 'postgresql') ensures that future mentions
    of 'Postgres' resolve to the same graph node as 'PostgreSQL' and 'postgresql'.
    """
    return _do_alias(surface, canonical)


def hymem_retract(subject: str, predicate: str, object: str) -> str:
    """Retract a knowledge graph edge that was wrongly extracted.

    Use this when you (or the user) realize HyMem extracted a relationship
    that's incorrect — e.g., the LLM hallucinated a dependency. Predicate must
    be one of: uses, depends_on, prefers, rejects, avoids, replaces,
    conflicts_with, deploys_to, part_of, equivalent_to.
    """
    return _do_retract(subject, predicate, object)


def hymem_add_rule(text: str, scope: str = "always_on", trigger_entities: str = "") -> str:
    """Record a STANDING RULE — a behavioral instruction to always follow.

    Rules are imperatives about HOW to behave ("always run the tests before
    pushing", "never suggest Docker"), distinct from facts. An `always_on` rule
    (default) is injected into every future context call; a `contextual` rule
    fires only when one of its trigger entities is in play.

    Use this when the user TELLS you a standing preference or prohibition — not
    for one-off facts (those are logged as messages and extracted at dream time).

    Arguments:
        text             — the rule, phrased as a directive.
        scope            — "always_on" (default) or "contextual".
        trigger_entities — for scope="contextual" only: a comma-separated list
                           of entities that activate the rule (e.g. "redis,cache").
    """
    return _do_add_rule(text, scope, trigger_entities)


def hymem_list_rules() -> str:
    """List all active standing rules (with ids), so you can see what behavioral
    rules are in force before answering. Each line is `#id [scope] text`."""
    return _do_list_rules()


def hymem_suggest_rules(limit: int = 10) -> str:
    """Propose standing rules inferred from RECENT behavior, to review and confirm
    — this does NOT add anything. Auto-adding inferred rules is deliberately off:
    a rule injects into every future call, so a human/agent confirms first.

    Each candidate shows its corroboration — how many markers over how many
    distinct sessions support it (more sessions = more durable) — its confidence,
    and its source kinds; ones matching an active rule are flagged. To adopt one,
    call `hymem_add_rule` with its text; skip the rest. Reads unconsolidated
    markers, so call it after logging a session and before dreaming."""
    return _do_suggest_rules(limit)


def main() -> None:
    # Configure logging at the application entry point (never on import — that
    # would clobber a host's handlers). Without this, the dream runner's
    # log.info() lines — including "aggregate.built nodes=/reused=" and the
    # aggregate.build_failure exception — are silently dropped, leaving the
    # server with no operational visibility. Level is env-tunable.
    logging.basicConfig(
        level=os.environ.get("HYMEM_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    primary: BaseException | None = None
    try:
        mcp_instance = _get_mcp()
        # Fail during process startup (with bootstrap rollback) instead of on
        # the first tool call after the MCP transport has advertised itself.
        _get_hy()
        mcp_instance.tool()(hymem_capture)
        mcp_instance.tool()(hymem_log)
        mcp_instance.tool()(hymem_dream)
        mcp_instance.tool()(hymem_augment)
        mcp_instance.tool()(hymem_ask)
        mcp_instance.tool()(hymem_profile)
        mcp_instance.tool()(hymem_digest)
        mcp_instance.tool()(hymem_alias)
        mcp_instance.tool()(hymem_retract)
        mcp_instance.tool()(hymem_add_rule)
        mcp_instance.tool()(hymem_list_rules)
        mcp_instance.tool()(hymem_suggest_rules)
        mcp_instance.run()
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            _shutdown_hy()
        except BaseException as cleanup:
            if primary is None:
                raise
            try:
                primary.add_note(
                    "HyMem MCP lifecycle cleanup failed: "
                    f"{type(cleanup).__name__}"
                )
            except (AttributeError, TypeError):  # pragma: no cover
                pass


if __name__ == "__main__":
    main()
