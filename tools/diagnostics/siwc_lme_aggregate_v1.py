"""Offline, source-order audit of completed Luna LongMemEval-S windows.

This reads only the pinned dataset and the one-shot capsules. It never imports
the SIWC transport, opens credentials, or runs a model. A full score is emitted
only for one validated result per one of the 500 original question IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any


DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
SOURCE_IDS_HASH = "sha256:a4849b8afda6b6ed31ead4fc28d00784d2d5fef945be87642f5ce3ab710b21c4"
QUESTION_COUNT = 500
RESULT_SCHEMA = "siwc-lme-semantic-diagnostic-v12"
RECEIPT_SCHEMA = "siwc-lme-diagnostic-launch-v2"
NO_DREAM_RESULT_SCHEMA = "siwc-lme-nodream-diagnostic-v13"
NO_DREAM_RECEIPT_SCHEMA = "siwc-lme-nodream-launch-v1"
CHECKPOINT_SCHEMA = "hymem-benchmark-checkpoint-v1"
RUNNER_PATH = "tools/diagnostics/siwc_lme_diagnostic_v12.py"
NO_DREAM_RUNNER_PATH = "tools/diagnostics/siwc_lme_nodream_v13.py"
TRANSPORT_PATH = "benchmarks/chatgpt_plan_responses_v11.py"
BRIDGE_PATH = "benchmarks/chatgpt_plan_lme_v8.py"
RUNNER_SHA256 = "f473fcfc3b80e7b828604ae66f07ef0d1357f0b209059f1d29b4e85484ad8ddf"
TRANSPORT_SHA256 = "90136777330a6d478ad2682911004e77c4cc85d146a50c31a7fc58408deabcbf"
BRIDGE_SHA256 = "e8230913f158166725bc4890db23dfc2e7f10922df0b1c505f7be1b59d0ac6db"
CANDIDATE_MAP_SHA256 = "9868c633a4ddd132efc8fe5d0f9e77110281a34f15828a0bf7af488687bbe6ea"
INVENTORY_SHA256 = "022b1e60f68afd1eac10fc48b76feb02d7c2ffbf4734d84bbd526229a61a2f17"
GRANT_IDENTITY_SHA256 = "5f91fe05fb7d3b0552b7247d6a81f3fd29aae894dbcf45828b1556a0b11633dc"
BILLING_POLICY = "siwc_server_enforced_plan_or_existing_credits_v1"
DIAGNOSTIC_MODE = "semantic_diagnostic_v2"
NO_DREAM_MODE = "message_only_luna_diagnostic_v1"
NO_DREAM_RUNNER_SHA256 = "787925aed73b8e4d4a82979400d3650770d249d5e19bd2bc44b31b77198a4ad0"
PROFILES = {
    RECEIPT_SCHEMA: {"variant": "full_dream_diagnostic_v12",
                     "campaign_schema": "siwc-lme-luna-campaign-v1",
                     "result_schema": RESULT_SCHEMA,
                     "mode": DIAGNOSTIC_MODE, "dream_mode": None,
                     "runner_path": RUNNER_PATH, "runner_sha256": RUNNER_SHA256,
                     "candidate_map_sha256": CANDIDATE_MAP_SHA256,
                     "inventory_sha256": INVENTORY_SHA256},
    NO_DREAM_RECEIPT_SCHEMA: {"variant": "no_dream_diagnostic_v13",
                              "campaign_schema": "siwc-lme-luna-nodream-campaign-v1",
                              "result_schema": NO_DREAM_RESULT_SCHEMA,
                              "mode": NO_DREAM_MODE, "dream_mode": "no_dream",
                              "runner_path": NO_DREAM_RUNNER_PATH,
                              "runner_sha256": NO_DREAM_RUNNER_SHA256,
                              "candidate_map_sha256": CANDIDATE_MAP_SHA256,
                              "inventory_sha256": INVENTORY_SHA256},
}
_YES_WORD = re.compile(r"\byes\b")
_NO_WORD = re.compile(r"\bno\b")
_NEGATED_YES = re.compile(
    r"\b(?:not|never|isn'?t|wasn'?t|aren'?t|ain'?t)\s+"
    r"(?:really\s+|quite\s+|exactly\s+|an?\s+)?yes\b")


class AggregationError(ValueError):
    """A bounded evidence reason; never includes source or model text."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise AggregationError(code)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AggregationError("duplicate_json_key")
        result[key] = value
    return result


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def _read_json(path: Path, *, maximum: int) -> dict[str, Any]:
    _require(_regular(path) and path.stat().st_size <= maximum, "artifact_missing_or_oversize")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
    except (OSError, UnicodeError, ValueError) as exc:
        raise AggregationError("artifact_json_invalid") from exc
    _require(type(value) is dict, "artifact_shape_invalid")
    return value


def _sha_file(path: Path) -> str:
    _require(_regular(path), "source_file_missing")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


_dataset_cache: tuple[tuple[Any, ...], list[tuple[str, str, str, str, str]]] | None = None


def _dataset(path: Path) -> list[tuple[str, str, str, str, str]]:
    """Keep source hashes and gold bindings, never all 277 MB of rows."""
    global _dataset_cache
    _require(_regular(path), "dataset_missing")
    before = path.stat()
    stamp = (str(path.resolve()), before.st_dev, before.st_ino,
             before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if _dataset_cache is not None and _dataset_cache[0] == stamp:
        return _dataset_cache[1]
    _require(_sha_file(path) == DATASET_SHA256, "dataset_hash_mismatch")
    decoder = json.JSONDecoder(object_pairs_hook=_object_pairs)
    indexed: list[tuple[str, str, str, str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            _require(source.read(1) == "[", "dataset_shape_invalid")
            buffer = ""
            eof = False
            while True:
                buffer = buffer.lstrip()
                if not buffer and not eof:
                    block = source.read(64 * 1024)
                    eof = not block
                    buffer += block
                    continue
                _require(bool(buffer), "dataset_shape_invalid")
                if buffer[0] == "]":
                    buffer = buffer[1:] + source.read()
                    _require(not buffer.strip(), "dataset_shape_invalid")
                    break
                if indexed:
                    _require(buffer[0] == ",", "dataset_shape_invalid")
                    buffer = buffer[1:].lstrip()
                    while not buffer and not eof:
                        block = source.read(64 * 1024)
                        eof = not block
                        buffer += block
                        buffer = buffer.lstrip()
                _require(bool(buffer) and buffer[0] not in ",]",
                         "dataset_shape_invalid")
                while True:
                    try:
                        row, end = decoder.raw_decode(buffer)
                        break
                    except json.JSONDecodeError as exc:
                        _require(not eof and len(buffer) < 16_000_000,
                                 "dataset_item_invalid")
                        block = source.read(64 * 1024)
                        eof = not block
                        buffer += block
                _require(type(row) is dict and len(indexed) < QUESTION_COUNT,
                         "dataset_denominator_invalid")
                qid = row.get("question_id")
                question = row.get("question")
                answer = row.get("answer")
                qtype = row.get("question_type")
                _require(type(qid) is str and bool(qid)
                         and type(question) is str and bool(question)
                         and type(answer) in (str, int) and type(qtype) is str,
                         "dataset_ids_invalid")
                indexed.append((qid, hashlib.sha256(_canonical(row)).hexdigest(),
                    hashlib.sha256(question.encode("utf-8")).hexdigest(),
                    hashlib.sha256(str(answer).encode("utf-8")).hexdigest(), qtype))
                buffer = buffer[end:]
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, AggregationError):
            raise
        raise AggregationError("dataset_json_invalid") from exc
    ids = [item[0] for item in indexed]
    _require(len(ids) == QUESTION_COUNT and len(set(ids)) == QUESTION_COUNT
             and _content_hash(ids) == SOURCE_IDS_HASH,
             "dataset_ids_invalid")
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size,
              before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_dev, after.st_ino, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns), "dataset_changed_during_audit")
    _dataset_cache = (stamp, indexed)
    return indexed


def _verify_no_dream_row(
    root: Path, local_index: int, projection: dict[str, Any],
    source: tuple[str, str, str, str, str],
) -> None:
    path = root / "run" / f"q-{local_index:04d}" / "private-row.json"
    _require(_regular(path) and path.stat().st_size <= 1_000_000
             and type(projection.get("private_row_sha256")) is str
             and projection["private_row_sha256"] == _sha_file(path),
             "private_row_digest_mismatch")
    row = _read_json(path, maximum=1_000_000)
    question, answer = row.get("question"), row.get("answer")
    hypothesis, judge_raw = row.get("hypothesis"), row.get("judge_raw")
    _require(row.get("question_id") == source[0]
             and row.get("question_type") == source[4]
             and type(question) is str
             and hashlib.sha256(question.encode("utf-8")).hexdigest() == source[2]
             and type(answer) is str
             and hashlib.sha256(answer.encode("utf-8")).hexdigest() == source[3]
             and "indexing" in row and row["indexing"] is None
             and row.get("retrieval_only") is False
             and row.get("judge_protocol") == "legacy-custom"
             and row.get("benchmark_failure") is None
             and row.get("judge_error") is False
             and row.get("judge_parse_valid") is True
             and row.get("context_sha") == projection.get("context_sha")
             and row.get("correct") is projection.get("correct")
             and type(hypothesis) is str and bool(hypothesis.strip())
             and not hypothesis.startswith("[LLM_ERROR:")
             and type(judge_raw) is str and bool(judge_raw.strip())
             and not judge_raw.startswith("[LLM_ERROR:"),
             "private_row_content_invalid")
    parseable = judge_raw.casefold()
    yes, no = bool(_YES_WORD.search(parseable)), bool(_NO_WORD.search(parseable))
    low = judge_raw.lower()
    _require(yes != no and projection["correct"] is (
        yes and not bool(_NEGATED_YES.search(low))),
        "private_judge_verdict_mismatch")
    _require(projection.get("diagnostic_kind") == "no_dream"
             and projection.get("diagnostic_degraded") is False
             and projection.get("strict_indexing_healthy") is None
             and projection.get("grounding_rejected_claims") == 0
             and projection.get("grounding_rejection_reasons") == {}
             and type(projection.get("context_sha")) is str
             and re.fullmatch(r"[0-9a-f]{64}", projection["context_sha"]) is not None,
             "checkpoint_no_dream_projection_invalid")


def _verify_no_dream_stages(
    stages: dict[str, Any], budget: dict[str, Any], local_index: int,
) -> None:
    _require(set(stages) <= {"rerank", "reader", "judge"}
             and {"reader", "judge"} <= set(stages),
             "no_dream_stage_set_invalid")
    for label, stage in stages.items():
        _require(type(stage) is dict
                 and all(type(stage.get(field)) is int and stage[field] >= 0
                         for field in ("attempts", "returned", "turns", "known_tokens"))
                 and stage["attempts"] >= stage["turns"] >= stage["returned"],
                 "no_dream_stage_accounting_invalid")
        if label in {"reader", "judge"}:
            _require(stage["attempts"] == stage["returned"] == stage["turns"] == 1
                     and stage["known_tokens"] > 0,
                     "no_dream_reader_or_judge_missing")
        elif label == "rerank":
            _require(stage["returned"] <= 1, "no_dream_rerank_count_invalid")
    question_ledgers = budget.get("questions")
    ledger = (question_ledgers.get(f"q-{local_index:04d}")
              if type(question_ledgers) is dict else None)
    _require(type(ledger) is dict and ledger.get("usage_complete") is True
             and ledger.get("stopped") is False
             and ledger.get("in_flight") == 0
             and sum(stage["turns"] for stage in stages.values()) == ledger.get("turns")
             and sum(stage["known_tokens"] for stage in stages.values())
             == ledger.get("known_tokens"),
             "no_dream_question_ledger_mismatch")


def _receipt_limits(receipt: dict[str, Any]) -> dict[str, Any] | None:
    raw = receipt.get("limits")
    if type(raw) is not dict or set(raw) != {"campaign", "canary", "question"}:
        return None
    limits: dict[str, Any] = {}
    for name in ("campaign", "canary", "question"):
        values = raw[name]
        if type(values) is not list or len(values) != 3 or any(
            type(value) not in (int, float) for value in values
        ):
            return None
        limits[name] = dict(zip(("turns", "known_tokens", "seconds"), values))
    limits["indexing_seconds"] = receipt.get("indexing_seconds")
    limits["workers"] = receipt.get("workers")
    return limits


def _window(root: Path, dataset: list[tuple[str, str, str, str, str]]) -> dict[str, Any]:
    _require(root.is_absolute() and root.is_dir() and not root.is_symlink(),
             "capsule_root_invalid")
    receipt_path = root / "launch-receipt.json"
    receipt = _read_json(receipt_path, maximum=8192)
    attempt = _read_json(root / "launch-attempt.json", maximum=512)
    checkpoint = _read_json(root / "run" / "diagnostic-checkpoint.json", maximum=2_000_000)
    result = _read_json(root / "run" / "diagnostic-result.json", maximum=128_000)
    profile = PROFILES.get(receipt.get("schema"))
    _require(profile is not None, "receipt_schema_invalid")
    offset, count = receipt.get("source_offset"), receipt.get("selected_count")
    _require(receipt.get("root") == str(root)
             and type(offset) is int and type(count) is int
             and 0 <= offset <= QUESTION_COUNT - count and 1 <= count <= 4
             and receipt.get("dataset_sha256") == DATASET_SHA256
             and receipt.get("candidate_map_sha256") == profile["candidate_map_sha256"]
             and receipt.get("inventory_sha256") == profile["inventory_sha256"]
             and receipt.get("grant_identity_sha256") == GRANT_IDENTITY_SHA256
             and receipt.get("billing_policy") == BILLING_POLICY
             and receipt.get("selected_source_order") == "source_window"
             and receipt.get("model") == "gpt-5.6-luna"
             and receipt.get("reasoning") == "low"
             and receipt.get("auth") == "siwc_oauth"
             and receipt.get("endpoint") == "https://api.openai.com/v1/responses"
             and receipt.get("api_fallback_allowed") is False
             and receipt.get("automatic_topup_user_attested_off") is True
             and receipt.get("reload_allowed") is False
             and receipt.get("store") is False
             and receipt.get("stream") is True
             and receipt.get("dream_mode") == profile["dream_mode"]
             and (profile["dream_mode"] is None
                  or receipt.get("mode") == profile["mode"])
             and receipt.get("one_shot") is True, "receipt_policy_invalid")
    _require(attempt == {"receipt_sha256": _sha_file(receipt_path), "one_shot": True},
             "launch_attempt_mismatch")
    selected = dataset[offset:offset + count]
    ids = [item[0] for item in selected]
    row_digests = [item[1] for item in selected]
    _require(receipt.get("selected_row_sha256") == row_digests,
             "receipt_source_rows_mismatch")

    manifest = checkpoint.get("manifest")
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA
             and type(manifest) is dict
             and checkpoint.get("status") == "complete"
             and checkpoint.get("scored") is True
             and checkpoint.get("verdict_key") == "correct"
             and checkpoint.get("expected_ids") == ids,
             "checkpoint_identity_invalid")
    _require(manifest.get("schema") == profile["result_schema"]
             and manifest.get("mode") == profile["mode"]
             and manifest.get("run_id") == _content_hash({
                 key: value for key, value in manifest.items() if key != "run_id"})
             and checkpoint.get("run_id") == manifest["run_id"]
             and result.get("run_id") == manifest["run_id"]
             and manifest.get("source_offset") == offset
             and manifest.get("expected_count") == count
             and manifest.get("expected_ids_hash") == _content_hash(ids)
             and manifest.get("selected_row_sha256") == row_digests
             and manifest.get("selected_source_order") == "source_window"
             and manifest.get("dataset_sha256") == DATASET_SHA256
             and manifest.get("candidate_map_sha256") == receipt.get("candidate_map_sha256")
             and manifest.get("grant_identity_sha256") == receipt.get("grant_identity_sha256")
             and manifest.get("billing_policy") == receipt.get("billing_policy")
             and manifest.get("limits") == _receipt_limits(receipt)
             and manifest.get("dream_mode") == receipt.get("dream_mode")
             and manifest.get("scored_run") is True
             and manifest.get("canonical_r9_artifact") is False
             and manifest.get("official_model_score") is False,
             "manifest_receipt_mismatch")
    sources = receipt.get("source_sha256")
    _require(type(sources) is dict
             and sources.get(profile["runner_path"]) == profile["runner_sha256"]
             and sources.get(TRANSPORT_PATH) == TRANSPORT_SHA256
             and sources.get(BRIDGE_PATH) == BRIDGE_SHA256
             and manifest.get("runner_sha256") == profile["runner_sha256"]
             and manifest.get("transport_sha256") == TRANSPORT_SHA256
             and manifest.get("bridge_sha256") == BRIDGE_SHA256,
             "manifest_source_mismatch")
    _require(_sha_file(root / "source-map.json") == receipt.get("inventory_sha256"),
             "inventory_source_mismatch")
    for relative, digest in sources.items():
        rel = Path(relative)
        _require(type(relative) is str and rel.parts and not rel.is_absolute()
                 and all(part not in (".", "..") for part in rel.parts)
                 and type(digest) is str and len(digest) == 64
                 and _sha_file(root / "code" / rel) == digest,
                 "capsule_code_drift")
    entries = checkpoint.get("entries")
    counts = checkpoint.get("counts")
    _require(type(entries) is dict and set(entries) == set(ids)
             and type(counts) is dict
             and all(type(counts.get(key)) is int and counts[key] == expected
                     for key, expected in (("expected", count), ("completed", count),
                                           ("failed", 0), ("missing", 0)))
             and checkpoint.get("failure_ids") == [], "checkpoint_coverage_invalid")
    rows = []
    for local_index, qid in enumerate(ids):
        entry = entries[qid]
        _require(type(entry) is dict and entry.get("status") == "completed"
                 and entry.get("attempts") == 1
                 and entry.get("attempt_history") == [{
                     "attempt": 1, "status": "completed", "failure": None,
                     "row": entry.get("row")}]
                 and type(entry.get("row")) is dict,
                 "checkpoint_entry_invalid")
        row = entry["row"]
        reasons = row.get("grounding_rejection_reasons")
        _require(row.get("question_id") == qid
                 and type(row.get("correct")) is bool
                 and row.get("benchmark_failure") is None
                 and row.get("diagnostic_only") is True
                 and type(row.get("diagnostic_degraded")) is bool
                 and type(row.get("grounding_rejected_claims")) is int
                 and type(reasons) is dict
                 and all(type(code) is str and type(value) is int and value > 0
                         for code, value in reasons.items())
                 and sum(reasons.values()) == row["grounding_rejected_claims"],
                 "checkpoint_row_invalid")
        if profile["dream_mode"] == "no_dream":
            _verify_no_dream_row(root, local_index, row, selected[local_index])
        else:
            _require(row.get("diagnostic_kind") in {
                "strict_healthy", "grounding_recovery",
                "summary_degradation", "semantic_quarantine"}
                and type(row.get("strict_indexing_healthy")) is bool,
                "checkpoint_full_dream_row_invalid")
        rows.append(row)
    correct = sum(row["correct"] for row in rows)
    rejections: Counter[str] = Counter()
    for row in rows:
        rejections.update(row["grounding_rejection_reasons"])
    _require(result.get("schema") == profile["result_schema"]
             and (profile["dream_mode"] is None
                  or (result.get("dream_mode") == profile["dream_mode"]
                      and result.get("mode") == profile["mode"]))
             and result.get("diagnostic_complete") is True
             and result.get("diagnostic_only") is True
             and result.get("campaign_stop") is None
             and result.get("selected_denominator") == count
             and result.get("scored_count") == count
             and result.get("failed_or_unscored_count") == 0
             and result.get("correct_count") == correct
             and result.get("incorrect_count") == count - correct
             and result.get("grounding_rejected_claims") == sum(rejections.values())
             and result.get("grounding_rejection_reasons") == dict(sorted(rejections.items()))
             and result.get("diagnostic_degraded") is any(
                 row["diagnostic_degraded"] for row in rows)
             and result.get("strict_unhealthy_count") == sum(
                 row.get("strict_indexing_healthy") is False for row in rows)
             and result.get("stage_accounting_clean") is True
             and result.get("checkpoint_counts") == counts
             and type(result.get("canary")) is dict
             and result["canary"].get("usage_complete") is True
             and result.get("canonical_r9_artifact") is False
             and result.get("official_model_score") is False,
             "result_checkpoint_mismatch")
    _require(result["canary"].get("structural_valid") is True,
             "result_canary_invalid")
    if profile["dream_mode"] == "no_dream":
        _require(result["canary"].get("schema") == "luna-lme-nodream-route-probe-v1"
                 and result["canary"].get("ordinary_calls") == 1
                 and result["canary"].get("staged_calls") == 0
                 and result["canary"].get("completion_calls") == 1,
                 "no_dream_canary_invalid")
    accuracy = result.get("quality_accuracy_full_selected")
    _require(type(accuracy) in (int, float) and math.isfinite(accuracy)
             and math.isclose(accuracy, correct / count, rel_tol=0, abs_tol=1e-12),
             "result_accuracy_mismatch")
    budget = result.get("budget")
    _require(type(budget) is dict and budget.get("stopped") is False
             and budget.get("usage_complete") is True
             and budget.get("reserved") == budget.get("in_flight") == 0
             and type(budget.get("known_tokens")) is int
             and budget["known_tokens"] >= 0
             and type(budget.get("turns")) is int and budget["turns"] > 0,
             "result_budget_invalid")
    accounting = result.get("stage_accounting")
    _require(type(accounting) is dict and set(accounting) == set(ids)
             and all(type(accounting[qid]) is dict for qid in ids),
             "result_stage_accounting_invalid")
    if profile["dream_mode"] == "no_dream":
        for local_index, qid in enumerate(ids):
            _verify_no_dream_stages(accounting[qid], budget, local_index)
        _require(type(result["canary"].get("known_tokens")) is int
                 and result["canary"]["known_tokens"] >= 0
                 and result["canary"]["completion_calls"]
                 + sum(stage["turns"] for stages in accounting.values()
                       for stage in stages.values()) == budget["turns"]
                 and result["canary"]["known_tokens"]
                 + sum(stage["known_tokens"] for stages in accounting.values()
                       for stage in stages.values()) == budget["known_tokens"],
                 "no_dream_window_ledger_mismatch")
    return {"root": str(root), "offset": offset, "count": count,
            "ids": ids, "rows": rows, "correct": correct,
            "variant": profile["variant"], "dream_mode": profile["dream_mode"],
            "known_tokens": budget["known_tokens"], "turns": budget["turns"],
            "rejections": dict(sorted(rejections.items())),
            "identity": {key: receipt.get(key) for key in (
                "dataset_sha256", "candidate_map_sha256", "source_sha256",
                "billing_policy", "grant_identity_sha256", "model", "reasoning",
                "auth", "endpoint", "store", "stream", "api_fallback_allowed",
                "dream_mode")} | {"diagnostic_mode": manifest.get("mode")}}


def verify_capsule(dataset: Path, root: Path) -> dict[str, Any]:
    """Verify one completed window at any source offset, with no model calls."""
    _require(isinstance(dataset, Path) and isinstance(root, Path),
             "capsule_input_invalid")
    window = _window(root, _dataset(dataset))
    return {"schema": "siwc-lme-offline-capsule-v1", "root": window["root"],
            "source_offset": window["offset"], "rows_verified": window["count"],
            "variant": window["variant"], "dream_mode": window["dream_mode"],
            "correct_count": window["correct"],
            "known_tokens": window["known_tokens"], "turns": window["turns"],
            "diagnostic_only": True, "verification_model_calls": 0}


def verify_campaign(dataset: Path, capsule_roots: list[Path],
                    require_full: bool = False) -> dict[str, Any]:
    """Verify an ordered contiguous prefix; score only after all 500 rows."""
    _require(type(require_full) is bool and type(capsule_roots) is list
             and all(isinstance(root, Path) for root in capsule_roots),
             "campaign_input_invalid")
    source_rows = _dataset(dataset)
    windows = [_window(root, source_rows) for root in capsule_roots]
    _require(len({window["root"] for window in windows}) == len(windows),
             "duplicate_capsule_root")
    _require(len({_content_hash(window["identity"]) for window in windows}) <= 1,
             "mixed_campaign_identity")
    _require(len({window["variant"] for window in windows}) <= 1,
             "mixed_campaign_variant")
    ordered: list[dict[str, Any]] = []
    for window in windows:
        _require(window["offset"] == len(ordered), "campaign_gap_or_overlap")
        ordered.extend(window["rows"])
    if require_full:
        _require(len(ordered) == QUESTION_COUNT,
                 "incomplete_500_question_coverage")
    complete = len(ordered) == QUESTION_COUNT
    correct = sum(row["correct"] for row in ordered)
    rejections: Counter[str] = Counter()
    for row in ordered:
        rejections.update(row["grounding_rejection_reasons"])
    evidence = [{"question_id": source_rows[index][0],
                 "correct": row["correct"],
                 "diagnostic_degraded": row["diagnostic_degraded"],
                 "grounding_rejected_claims": row["grounding_rejected_claims"]}
                for index, row in enumerate(ordered)]
    return {"schema": "siwc-lme-offline-aggregate-v1",
            "diagnostic_only": True, "canonical_r9_artifact": False,
            "official_model_score": False,
            "variant": windows[0]["variant"] if windows else None,
            "dream_mode": windows[0]["dream_mode"] if windows else None,
            "dataset_sha256": DATASET_SHA256,
            "source_ids_hash": SOURCE_IDS_HASH,
            "selected_window_count": len(windows),
            "window_plan_sha256": _content_hash([
                {"root": window["root"], "offset": window["offset"],
                 "count": window["count"]} for window in windows]),
            "rows_verified": len(ordered),
            "covered_question_count": len(ordered),
            "total_question_count": QUESTION_COUNT,
            "first_missing_offset": len(ordered) if not complete else None,
            "full_coverage_verified": complete,
            "correct_count": correct,
            "accuracy": correct / QUESTION_COUNT if complete else None,
            "known_tokens": sum(window["known_tokens"] for window in windows),
            "turns": sum(window["turns"] for window in windows),
            "grounding_rejected_claims": sum(rejections.values()),
            "grounding_rejection_reasons": dict(sorted(rejections.items())),
            "ordered_verdicts_sha256": _content_hash(evidence),
            "verification_model_calls": 0}


def _verify_budget_chain(
    root: Path, manifest: dict[str, Any], manifest_sha256: str,
    recorded: dict[str, Any], campaign_schema: str,
) -> None:
    revision = recorded.get("budget_revision")
    _require(type(revision) is int and 0 <= revision <= 100,
             "campaign_budget_revision_invalid")
    previous = manifest_sha256
    limits = {key: manifest.get(key) for key in (
        "max_known_tokens", "max_turns", "max_wall_seconds")}
    _require(all(type(value) is int and value > 0 for value in limits.values()),
             "campaign_budget_manifest_invalid")
    for number in range(1, revision + 1):
        path = root / f"budget-{number:04d}.json"
        value = _read_json(path, maximum=2048)
        revised = {key: value.get(key) for key in limits}
        _require(value.get("schema") == campaign_schema + "-budget-v1"
                 and value.get("manifest_sha256") == manifest_sha256
                 and value.get("revision") == number
                 and value.get("previous_sha256") == previous
                 and all(type(revised[key]) is int
                         and revised[key] >= limits[key] for key in limits)
                 and any(revised[key] > limits[key] for key in limits),
                 "campaign_budget_revision_invalid")
        limits = revised
        previous = _sha_file(path)
    _require(recorded.get("budget_revision_sha256") == previous,
             "campaign_budget_revision_mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, action="append", default=[])
    parser.add_argument("--campaign-result", type=Path,
                        help="JSON with an ordered `capsules` list")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        roots = args.capsule
        if args.campaign_result is not None:
            _require(not roots, "mixed_capsule_selection")
            recorded = _read_json(args.campaign_result, maximum=2_000_000)
            paths = recorded.get("capsules")
            _require(type(paths) is list and all(type(value) is str for value in paths),
                     "campaign_capsules_invalid")
            roots = [Path(value) for value in paths]
        report = verify_campaign(args.dataset, roots,
                                 require_full=not args.allow_partial)
        if args.campaign_result is not None:
            manifest_path = Path(recorded.get("campaign_manifest", ""))
            _require(manifest_path.is_absolute()
                     and manifest_path.parent == args.campaign_result.parent
                     and type(recorded.get("manifest_sha256")) is str
                     and _sha_file(manifest_path) == recorded["manifest_sha256"],
                     "campaign_manifest_mismatch")
            manifest = _read_json(manifest_path, maximum=262_144)
            planned = manifest.get("windows")
            campaign_schema = next(
                (profile["campaign_schema"] for profile in PROFILES.values()
                 if profile["variant"] == report["variant"]), None)
            _require(manifest.get("schema") == campaign_schema
                     and manifest.get("campaign_root") == str(manifest_path.parent)
                     and manifest.get("dataset_sha256") == DATASET_SHA256
                     and manifest.get("runner_sha256") == next(
                         (profile["runner_sha256"] for profile in PROFILES.values()
                          if profile["variant"] == report["variant"]), None)
                     and manifest.get("verifier_sha256") == _sha_file(Path(__file__))
                     and manifest.get("variant") == report["variant"]
                     and type(planned) is list
                     and all(type(window) is dict for window in planned)
                     and report["window_plan_sha256"] == _content_hash([
                         {"root": window.get("root"), "offset": window.get("offset"),
                          "count": window.get("count")} for window in planned]),
                     "campaign_manifest_plan_mismatch")
            _require(recorded.get("schema") == campaign_schema
                     and recorded.get("diagnostic_complete") is True
                     and recorded.get("canonical_r9_artifact") is False
                     and recorded.get("official_model_score") is False
                     and recorded.get("model") == "gpt-5.6-luna"
                     and recorded.get("auth") == "siwc_oauth"
                     and recorded.get("variant") == report["variant"]
                     and recorded.get("billing_policy") == BILLING_POLICY
                     and recorded.get("api_fallback_allowed") is False
                     and recorded.get("rows_verified") == report["rows_verified"]
                     and recorded.get("correct_count") == report["correct_count"]
                     and recorded.get("known_tokens") == report["known_tokens"]
                     and recorded.get("turns") == report["turns"]
                     and recorded.get("diagnostic_accuracy") == report["accuracy"],
                     "campaign_result_mismatch")
            if report["dream_mode"] == "no_dream":
                _verify_budget_chain(manifest_path.parent, manifest,
                                     recorded["manifest_sha256"], recorded,
                                     campaign_schema)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except (AggregationError, OSError, UnicodeError, TypeError, ValueError) as exc:
        code = exc.args[0] if type(exc) is AggregationError else "offline_verification_failed"
        print(json.dumps({"schema": "siwc-lme-offline-aggregate-v1", "ok": False,
                          "reason": code, "verification_model_calls": 0}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
