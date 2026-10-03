"""Opt-in, noncanonical LME indexing boundary for semantic-loss measurement.

The caller injects one source-verified LME adapter/protocol pair.  This module
does not change the benchmark's prompts, extraction writes, search, or scorer.
An unhealthy index remains unhealthy in its canonical summary; admission is a
separate diagnostic decision with a closed, source-free schema.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any


MODE = "semantic_diagnostic_v1"
_SEMANTIC_REASONS = frozenset({
    "contract_failure", "incomplete_response", "item_validation_failure",
    "output_limit_exceeded", "parse_failure", "response_conflict",
    "shape_failure", "grounding_failure",
})
_BRANCH_REASONS = frozenset({"branch_incomplete", "response_conflict"})
_MAX_HELD_ROWS = 100_000
_SUMMARY_REASONS = frozenset({
    "summary_output_cap", "summary_shape_failure", "summary_validation_failure",
    "parse_failure", "output_truncated", "shape_failure", "prior_summary_gap",
})
_DETAIL = re.compile(r"[a-z0-9_.\[\]-]{1,100}:[a-z0-9_]{1,60}\Z")
_DIRECT_DETAILS = frozenset({
    "provider:finish_length", "response:invalid_json",
    "response:truncated_json", "response:not_object",
    "top:unexpected_keys", "top.complete:not_boolean",
    "top.complete:false", "top.triples:not_array",
    "top.markers:not_array", "top.triples:item_cap_reached",
    "top.markers:item_cap_reached", "repair:failed",
    "repair:empty_after_invalid", "split:max_depth_reached",
    "split:no_admissible_semantic_boundary",
    "triples:polarity_conflict",
})
_ITEM_CODES = frozenset({
    "missing", "unexpected_keys", "not_object", "not_string", "empty",
    "not_allowed", "not_plus_or_minus_one", "not_positive_integer",
    "not_in_input", "response_conflict",
})
_ITEM_DETAIL = re.compile(
    r"(triples|markers)\[([0-9]{1,4})\]\.([a-z_]+):([a-z_]+)\Z")
_TOP_MISSING = re.compile(r"top\.(triples|markers|complete):missing\Z")
_PREPARTITION_LEAF = re.compile(r"prepartition_leaf:[0-9]{1,4}\Z")


def _nonnegative_counts(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping) or any(
        type(key) is not str or type(count) is not int or count < 0
        for key, count in value.items()
    ):
        return None
    return dict(value)


def _direct_detail(detail: str) -> bool:
    """Closed, source-derived diagnostic codes; no free-form provider text."""
    if detail in _DIRECT_DETAILS or _TOP_MISSING.fullmatch(detail):
        return True
    if detail in {"stage:primary", "stage:empty", "stage:omission"}:
        return True
    if _PREPARTITION_LEAF.fullmatch(detail):
        return True
    if detail.startswith("original:"):
        return detail[9:] in {"contract_failure", "item_validation_failure"}
    item = _ITEM_DETAIL.fullmatch(detail)
    if item is None:
        return False
    field, code = item.group(3), item.group(4)
    if code not in _ITEM_CODES:
        return False
    if item.group(1) == "markers":
        return field in {"item", "kind", "statement"}
    return field in {
        "item", "subject", "predicate", "object", "polarity",
        "source_message_id",
    }


def _semantic_tree(reason: str, details: list[str], depth: int = 0) -> bool:
    if depth > 16 or reason not in _SEMANTIC_REASONS | _BRANCH_REASONS:
        return False
    children: dict[str, dict[str, Any]] = {}
    direct: list[str] = []
    for detail in details:
        branch = next((side for side in ("left", "right")
                       if detail.startswith(side + ":")
                       or detail.startswith(side + ".")), None)
        if branch is None:
            direct.append(detail)
            continue
        child = children.setdefault(branch, {"reason": None, "details": []})
        if detail.startswith(branch + ":"):
            if child["reason"] is not None:
                return False
            child["reason"] = detail[len(branch) + 1:]
        else:
            child["details"].append(detail[len(branch) + 1:])
    if children and reason not in _BRANCH_REASONS:
        return False
    if any(child["reason"] is None or not _semantic_tree(
        child["reason"], child["details"], depth + 1,
    ) for child in children.values()):
        return False
    source_position = [item for item in direct
                       if _PREPARTITION_LEAF.fullmatch(item)]
    if len(source_position) > 1:
        return False
    direct = [item for item in direct if not _PREPARTITION_LEAF.fullmatch(item)]
    if reason == "branch_incomplete":
        return bool(children) and not direct
    if reason == "response_conflict":
        return bool(direct) and all(detail == "triples:polarity_conflict"
                                    or (item := _ITEM_DETAIL.fullmatch(detail)) is not None
                                    and item.group(1) == "triples"
                                    and (item.group(3), item.group(4)) in {
                                        ("polarity", "response_conflict"),
                                        ("source_message_id", "not_in_input"),
                                    }
                                    for detail in direct)
    if reason == "grounding_failure":
        # Only the staged gate's model verdicts are quality misses. Contract,
        # support-integrity, source, transport and budget errors remain fatal.
        return (not children and len(direct) == 1
                and direct[0] in {"grounding:verdict_unsupported",
                                  "grounding:verdict_uncertain"})
    return all(_direct_detail(detail) for detail in direct)


def _semantic_reason(reason: object, details: object) -> bool:
    """Recursively prove every reported failed leaf is semantic."""
    if type(reason) is not str or type(details) is not str:
        return False
    try:
        parsed = json.loads(details)
    except (TypeError, ValueError):
        return False
    return (type(parsed) is list and len(parsed) <= 32
            and all(type(item) is str and len(item) <= 160
                    and _DETAIL.fullmatch(item) is not None
                    and not item.endswith("diagnostics:truncated")
                    and not item.endswith("diagnostic:invalid")
                    for item in parsed)
            and _semantic_tree(reason, parsed))


def _held_rows(conn: Any, retry_bound: int) -> list[dict[str, Any]]:
    """Read held, unpublished extraction attempts in the caller's snapshot."""
    if type(retry_bound) is not int or retry_bound < 0:
        raise ValueError("diagnostic retry policy is invalid")
    if retry_bound == 0:
        return []
    rows = conn.execute(
        """SELECT a.prompt_version, a.phase1_generation_key,
                  a.last_failure_reason, a.last_failure_details, COUNT(*) AS n
           FROM chunk_extraction_attempts a JOIN chunks c ON c.id=a.chunk_id
           WHERE c.chunk_kind='extraction'
             AND COALESCE(c.salience_reason,'')<>'short_session_fallback'
             AND c.source_manifest_version='claim-source-manifest-v1'
             AND c.source_manifest_count>0 AND a.attempts>=?
             AND NOT EXISTS (SELECT 1 FROM chunk_extraction_terminal_losses loss
                             WHERE loss.chunk_id=c.id)
             AND NOT EXISTS (SELECT 1 FROM current_phase1_publications p
                             WHERE p.chunk_id=c.id
                               AND p.prompt_version=a.prompt_version
                               AND p.phase1_generation_key=a.phase1_generation_key)
           GROUP BY a.prompt_version, a.phase1_generation_key,
                    a.last_failure_reason, a.last_failure_details
           LIMIT ?""", (retry_bound, _MAX_HELD_ROWS + 1),
    ).fetchall()
    if len(rows) > _MAX_HELD_ROWS:
        raise ValueError("diagnostic held-row census exceeds bound")
    return [dict(row) for row in rows]


def _summary_rows(conn: Any, classifier: Any) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id,summary_failure_reason,summary_failure_count FROM sessions LIMIT ?",
        (_MAX_HELD_ROWS + 1,),
    ).fetchall()
    if len(rows) > _MAX_HELD_ROWS:
        raise ValueError("diagnostic summary census exceeds bound")
    result = []
    for row in rows:
        state = classifier(conn, row["id"])
        if not isinstance(state, Mapping) or set(state) != {
            "summary_healthy", "degraded", "missing", "malformed",
        } or any(type(value) is not bool for value in state.values()):
            raise ValueError("diagnostic summary state is malformed")
        if state["degraded"]:
            result.append({
                "reason": row["summary_failure_reason"],
                "count": row["summary_failure_count"],
                "missing": state["missing"],
            })
    return result


def coherent_status_and_held(
    memory: Any, status_runtime: Any, summary_classifier: Any,
    embedding_client: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Get validated benchmark status and deficit reasons in one DB snapshot."""
    snapshot = getattr(memory, "dream_status_with_snapshot", None)
    if not callable(snapshot):
        raise ValueError("coherent diagnostic snapshot unavailable")
    retry_bound = memory.config.chunk_extraction_max_attempts

    def extension(conn: Any) -> dict[str, Any]:
        return {
            **status_runtime.embedding_backlog_status(conn, embedding_client),
            "diagnostic_held_reason_rows": _held_rows(conn, retry_bound),
            "diagnostic_summary_reason_rows": _summary_rows(conn, summary_classifier),
        }

    observed = snapshot(extension)
    if not isinstance(observed, dict):
        raise ValueError("diagnostic status is malformed")
    rows = observed.pop("diagnostic_held_reason_rows", None)
    summary_rows = observed.pop("diagnostic_summary_reason_rows", None)
    if type(rows) is not list or type(summary_rows) is not list:
        raise ValueError("diagnostic deficit census is malformed")
    # Reuse the candidate's full durable-status validation without another DB
    # read. Its callback receives the immutable snapshot already captured.
    proxy = SimpleNamespace(
        config=memory.config,
        dream_status_with_snapshot=lambda _extension: dict(observed),
    )
    return (status_runtime.durable_indexing_status(
        proxy, embedding_client), rows, summary_rows)


def classify_semantic_indexing(
    summary: Mapping[str, Any], held_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]], *, cache_key: str, generation_key: str,
) -> dict[str, Any]:
    """Classify an already protocol-validated canonical summary.

    This is a second, diagnostic-only policy gate, not a replacement for the
    candidate protocol's complete schema and provenance validation.
    """
    result: dict[str, Any] = {
        "mode": MODE, "admitted": False, "kind": "rejected",
        "quarantined_chunks": 0, "summary_degraded_sessions": 0,
        "summary_missing_sessions": 0, "semantic_failure_reasons": {},
        "summary_failure_reasons": {},
    }
    if (type(cache_key) is not str or not cache_key
            or type(generation_key) is not str or not generation_key
            or not isinstance(summary, Mapping)
            or summary.get("complete") is not True
            or summary.get("cleanup_errors") != []):
        return result
    final = summary.get("final_status")
    if not isinstance(final, Mapping):
        return result
    pending = _nonnegative_counts(final.get("pending"))
    malformed = _nonnegative_counts(final.get("malformed"))
    quarantine = _nonnegative_counts(final.get("quarantined"))
    summary_health = final.get("summary_health")
    terminal = final.get("terminal_loss")
    coverage = final.get("coverage_integrity")
    if (pending is None or malformed is None or quarantine is None
            or not isinstance(summary_health, Mapping)
            or not isinstance(terminal, Mapping) or not isinstance(coverage, Mapping)
            or any(pending.values()) or any(malformed.values())
            or terminal.get("chunks") != 0 or coverage.get("failures") != 0
            or final.get("in_progress") is not False
            or final.get("phase1_generation_key") != generation_key
            or summary_health.get("malformed_summaries") != 0
            or any(value for key, value in quarantine.items() if key != "quarantined_chunks")):
        return result
    degraded = summary_health.get("summary_degraded_sessions")
    missing = summary_health.get("summary_missing_sessions")
    if (type(degraded) is not int or type(missing) is not int
            or not 0 <= missing <= degraded):
        return result
    if type(summary_rows) is not list or len(summary_rows) != degraded:
        return result
    summary_reasons: Counter[str] = Counter()
    missing_seen = 0
    for row in summary_rows:
        if not isinstance(row, dict):
            return result
        reason, count, is_missing = (row.get("reason"), row.get("count"),
                                     row.get("missing"))
        if (type(reason) is not str or reason not in _SUMMARY_REASONS
                or type(count) is not int or count < 1
                or type(is_missing) is not bool):
            return result
        summary_reasons[reason] += 1
        missing_seen += is_missing
    if missing_seen != missing:
        return result
    held_count = quarantine.get("quarantined_chunks")
    if type(held_count) is not int or held_count < 0:
        return result
    reasons: Counter[str] = Counter()
    for row in held_rows:
        if not isinstance(row, dict):
            return result
        if (row.get("prompt_version") != cache_key
                or row.get("phase1_generation_key") != generation_key):
            continue
        count = row.get("n")
        reason = row.get("last_failure_reason")
        if (type(count) is not int or count <= 0
                or not _semantic_reason(reason, row.get("last_failure_details"))):
            return result
        reasons[reason] += count
    if sum(reasons.values()) != held_count:
        return result
    outcome = summary.get("outcome")
    if held_count:
        if (outcome != "failure" or summary.get("healthy") is not False
                or (summary.get("failure") or {}).get("code") != "quarantined_extraction"):
            return result
        kind = "semantic_quarantine"
    elif degraded:
        if (outcome != "success_with_summary_degradation"
                or summary.get("healthy") is not True
                or summary.get("summary_healthy") is not False):
            return result
        kind = "summary_degradation"
    else:
        if (outcome != "success" or summary.get("healthy") is not True
                or summary.get("summary_healthy") is not True):
            return result
        kind = "strict_healthy"
    result.update(admitted=True, kind=kind, quarantined_chunks=held_count,
                  summary_degraded_sessions=degraded,
                  summary_missing_sessions=missing,
                  semantic_failure_reasons=dict(sorted(reasons.items())),
                  summary_failure_reasons=dict(sorted(summary_reasons.items())))
    return result


def make_diagnostic_adapter_class(
    lme: Any, protocol: Any, status_runtime: Any,
    summary_classifier: Any, base_class: type,
) -> type:
    """Compose with a source-verified candidate adapter; no global patching."""
    class DiagnosticAdapter(base_class):
        diagnostic_indexing: dict[str, Any] | None = None

        def dream_and_wait(self, timeout=3600, *, max_cycles=100, require_healthy=True):
            if require_healthy is not True:
                raise ValueError("diagnostic indexing requires the strict call site")
            dream_hy = self.hy.fork()
            self.diagnostic_indexing = None
            try:
                try:
                    raw = lme.converge_indexing(
                        dream_hy.dream,
                        status=lambda: lme.durable_indexing_status(
                            dream_hy, getattr(self, "embedding_client", None)),
                        max_cycles=max_cycles, timeout_s=timeout,
                        require_healthy=False,
                    )
                    canonical = lme.canonicalize_lme_indexing_summary(raw)
                    protocol._validate_versioned_indexing(
                        canonical, require_healthy=True, allow_failure=True)
                    self.last_indexing_summary = canonical
                    observed, rows, summary_rows = coherent_status_and_held(
                        dream_hy, status_runtime, summary_classifier,
                        getattr(self, "embedding_client", None))
                    comparison = lme.canonicalize_lme_indexing_summary(
                        {**raw, "final_status": observed})
                    if comparison["final_status"] != canonical["final_status"]:
                        raise lme.BenchmarkIntegrityError(
                            "diagnostic indexing snapshot changed")
                    decision = classify_semantic_indexing(
                        canonical, rows, summary_rows,
                        cache_key=observed.get("extraction_cache_key"),
                        generation_key=observed.get("phase1_generation_key"),
                    )
                    self.diagnostic_indexing = decision
                    if not decision["admitted"]:
                        raise lme.IndexingConvergenceError(
                            "diagnostic indexing deficit is not semantic", canonical)
                    return canonical
                except lme.IndexingConvergenceError as exc:
                    if exc.summary.get("schema") == lme.LME_INDEXING_SUMMARY_VERSION:
                        self.last_indexing_summary = dict(exc.summary)
                    else:
                        self.last_indexing_summary = lme.canonicalize_lme_indexing_summary(
                            exc.summary)
                    exc.summary = self.last_indexing_summary
                    raise
            finally:
                evidence = (self.last_indexing_summary.get("cleanup_errors")
                            if isinstance(self.last_indexing_summary, dict) else None)
                lme.run_cleanup_actions(
                    [("dream_fork_close", dream_hy.close),
                     ("query_cache_invalidation", self.hy.invalidate_query_caches)],
                    primary_exception=sys.exc_info()[1], evidence_sink=evidence)

    DiagnosticAdapter.__name__ = "DiagnosticAdapter"
    return DiagnosticAdapter
