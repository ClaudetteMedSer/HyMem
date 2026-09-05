#!/usr/bin/env python3
"""Run registry for LongMemEval benchmark executions.

Builds/updates a small SQLite DB (lme_runs.db) that records one row per
run JSON: date, scores, and the effective flags that were on/off at the
time the run was created.  The goal is a queryable answer to "which
config produced this score?" — something the per-run JSON files cannot
answer, and which would have surfaced the 2026-08-26 aggregation-default
flip had it existed.

Usage:
  lme_registry.py ingest [FILE ...]      Add runs from JSON files (default: all archives)
  lme_registry.py list [--limit N] [--flag COL]   Print table
  lme_registry.py query "SQL"            Raw query against the DB
  lme_registry.py audit [--strict]       Date-check the aggregation label
  lme_registry.py arms A.json B.json --lever K   Can this pair evidence its A/B?

Flags are recorded exactly as they appear in each run's config block.
If a key is absent (e.g. older format, or a lever that was not recorded),
the value is stored as NULL — this registry does not guess; the
date-check / provenance analysis is done on top of NULLs, not inside
them.
"""
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

try:  # package import (tests): benchmarks.run_registry
    from . import run_registry as rr
    from .strictness import (
        BenchmarkIntegrityError, content_hash, read_artifact_or_pointer,
    )
    from .lme_protocol import strict_intent, validate_strict_artifact
except (ImportError, ValueError):  # direct CLI: python benchmarks/lme_registry.py
    import run_registry as rr
    from strictness import BenchmarkIntegrityError, content_hash, read_artifact_or_pointer
    from lme_protocol import strict_intent, validate_strict_artifact

DB = Path(os.environ.get("LME_REGISTRY_DB", "/home/node/.hermes/benchmarks/lme_runs.db"))
BENCH_DIR = Path(os.environ.get("LME_BENCH_DIR", "/home/node/.hermes/benchmarks"))

# Columns that exist across the known config formats (superset).
# Each maps to config-key or is a fixed transform.
FLAG_COLUMNS = [
    "auto_ability",
    "no_dream",
    "permissive_default",
    "embeddings",
    "graph_facts_first",
    "graph_multihop",
    "distill",
    "rerank_top_k",
    "sample",
    "scale",
    "seed",
    "workers",
    "top_k",
    "answer_model",
    "judge_model",
    "aggregation_nodes_enabled",   # NULL where never recorded (pre-pin formats)
    "episode_granularity_enabled", # NULL where never recorded (pre-2247074)
    "value_supersession_enabled",  # NULL where never recorded
]
# Config keys known to be *absent* in run JSONs even though they were
# active: pre-6543ee6 runs never carried these in the config block; the guard
# arms and all earlier rows fill them via `--set` only (NULL = not recorded).
# Post-6543ee6 the adapter writes them into the config block at :2887, so
# newer runs ingest without --set. Rows predating 6543ee6 stay NULL —
# the gap is visible, not guessed.
KNOWN_ABSENT = {"aggregation_nodes_enabled", "episode_granularity_enabled",
                "value_supersession_enabled"}

SCORE_CATEGORIES = [
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    archive     TEXT UNIQUE NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'archive',  -- archive | variant | rejudge
    run_date    TEXT NOT NULL,          -- from JSON date field (UTC-ish ISO);
                                        -- rejudge rows: last stem stamp (exec)
    source_date TEXT,              -- filename timestamp stamp (first
                                        -- stamp / rejudged_from); NULL for
                                        -- analyst-named variants (no stamp)
    -- flags (all NULL if not present in the run config block)
    auto_ability               INTEGER,
    no_dream                   INTEGER,
    permissive_default         INTEGER,
    embeddings                 INTEGER,
    graph_facts_first          INTEGER,
    graph_multihop             INTEGER,
    distill                    INTEGER,
    rerank_top_k               INTEGER,
    sample                     INTEGER,
    scale                      TEXT,
    seed                       INTEGER,
    workers                    INTEGER,
    top_k                      INTEGER,
    answer_model               TEXT,
    judge_model                TEXT,
    mr_aggregate_additive      INTEGER,  -- old-format name of the aggregation lever
    aggregation_nodes_enabled  INTEGER,
    episode_granularity_enabled INTEGER,
    value_supersession_enabled INTEGER,
    -- scores
    overall                    REAL,
    multi_session              REAL,
    single_session_assistant   REAL,
    single_session_preference  REAL,
    single_session_user        REAL,
    knowledge_update           REAL,
    temporal_reasoning         REAL,
    count                      INTEGER,
    answer_calls               INTEGER,
    judge_calls                INTEGER,
    retrieval_calls            INTEGER,
    total_tokens               INTEGER,
    elapsed_s                  REAL,
    flags_provenance           TEXT,     -- how recorded + analyst-set flags were established
    extras                     TEXT,     -- JSON: full config block + scores, for anything unmodeled
    -- strict-v1 evidence (additive; genuine legacy rows remain NULL)
    run_id                     TEXT,
    protocol                   TEXT,
    protocol_split             TEXT,
    calibration_receipt_hash   TEXT,
    strict_validated           INTEGER,
    official_comparable        INTEGER,
    development_only           INTEGER,
    official_denominator_validated INTEGER,
    official_protocol_aligned  INTEGER,
    official_scoring_semantics_aligned INTEGER,
    scored_run                 INTEGER,
    retrieval_only             INTEGER,
    label_free_answer_path     INTEGER,
    exploratory_non_comparable INTEGER,
    abstention_count           INTEGER,
    answerable_count           INTEGER,
    abstention_accuracy        REAL,
    answerable_accuracy        REAL,
    pipeline_calls             INTEGER,
    usage_exact                INTEGER,
    dataset_revision           TEXT,
    dataset_sha256             TEXT,
    source_ids_hash            TEXT,
    evaluator_commit           TEXT,
    evaluator_sha256           TEXT,
    reader_provider            TEXT,
    reader_base_url            TEXT,
    judge_provider             TEXT,
    judge_base_url             TEXT,
    pipeline_provider          TEXT,
    pipeline_model             TEXT,
    pipeline_base_url          TEXT,
    embedding_backend          TEXT,
    embedding_model            TEXT,
    embedding_base_url         TEXT,
    embedding_dimension        INTEGER,
    embedding_quality          TEXT,
    embedding_network_free     INTEGER,
    preregistered              INTEGER,
    result_digest              TEXT,
    artifact_digest            TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_date ON runs(run_date);
CREATE INDEX IF NOT EXISTS idx_runs_flags ON runs(auto_ability, no_dream, permissive_default);
CREATE INDEX IF NOT EXISTS idx_runs_aggr ON runs(aggregation_nodes_enabled);
"""

ADDITIVE_COLUMNS = {
    "run_id": "TEXT", "protocol": "TEXT", "strict_validated": "INTEGER",
    "protocol_split": "TEXT", "calibration_receipt_hash": "TEXT",
    "official_comparable": "INTEGER", "development_only": "INTEGER",
    "official_denominator_validated": "INTEGER",
    "official_protocol_aligned": "INTEGER", "scored_run": "INTEGER",
    "official_scoring_semantics_aligned": "INTEGER",
    "retrieval_only": "INTEGER", "label_free_answer_path": "INTEGER",
    "exploratory_non_comparable": "INTEGER", "abstention_count": "INTEGER",
    "answerable_count": "INTEGER", "abstention_accuracy": "REAL",
    "answerable_accuracy": "REAL", "pipeline_calls": "INTEGER",
    "usage_exact": "INTEGER", "dataset_revision": "TEXT",
    "dataset_sha256": "TEXT", "source_ids_hash": "TEXT",
    "evaluator_commit": "TEXT", "evaluator_sha256": "TEXT",
    "reader_provider": "TEXT", "reader_base_url": "TEXT",
    "judge_provider": "TEXT", "judge_base_url": "TEXT",
    "pipeline_provider": "TEXT", "pipeline_model": "TEXT",
    "pipeline_base_url": "TEXT",
    "embedding_backend": "TEXT", "embedding_model": "TEXT",
    "embedding_base_url": "TEXT",
    "embedding_dimension": "INTEGER", "embedding_quality": "TEXT",
    "embedding_network_free": "INTEGER", "preregistered": "INTEGER",
    "retrieval_calls": "INTEGER", "result_digest": "TEXT",
    "artifact_digest": "TEXT",
}


def connect():
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    # §6 migration (2026-08-31): source_date became nullable — NULL is the
    # honest value where a stem carries no stamp (beam/locomo-style rows,
    # analyst-renamed LME variants).  SQLite cannot drop a NOT NULL in
    # place, so rebuild the table when the old constraint is present.
    cols = {r[1]: r for r in con.execute("PRAGMA table_info(runs)")}
    if cols.get("source_date") and cols["source_date"][3]:  # notnull flag
        con.execute("ALTER TABLE runs RENAME TO runs_prestamp")
        con.executescript(SCHEMA)
        old_names = {
            row[1] for row in con.execute("PRAGMA table_info(runs_prestamp)")
        }
        new_names = [
            row[1] for row in con.execute("PRAGMA table_info(runs)")
            if row[1] in old_names
        ]
        quoted = ", ".join(f'"{name}"' for name in new_names)
        con.execute(
            f"INSERT INTO runs ({quoted}) SELECT {quoted} FROM runs_prestamp"
        )
        con.execute("DROP TABLE runs_prestamp")
        con.executescript(SCHEMA)
        con.commit()
    existing = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
    for name, sql_type in ADDITIVE_COLUMNS.items():
        if name not in existing:
            con.execute(f'ALTER TABLE runs ADD COLUMN "{name}" {sql_type}')
    existing_digest_index = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND name='idx_runs_artifact_digest'"
    ).fetchone()
    if existing_digest_index and "UNIQUE" not in (existing_digest_index[0] or "").upper():
        con.execute("DROP INDEX idx_runs_artifact_digest")
    con.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_artifact_digest "
        "ON runs(artifact_digest) WHERE artifact_digest IS NOT NULL"
    )
    con.commit()
    return con


def _to_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


# ---- stamp policy (§6, 2026-08-31) --------------------------------------
# LME adapter-written names always carry a \\d{8}T\\d{6}Z stamp (archive +
# rejudge): policy "required" — a missing stamp RAISES via the shared
# helper.  A NULL there is a defect, not a domain fact (it would look
# identical to a legitimately stamp-less beam/locomo row).  Analyst-renamed
# variants (longmemeval-v2-hymem-additive.json, -auto-ability-fulldream.json)
# carry no stamp by construction -> policy "optional", NULL recorded.
STAMP_POLICY = {"archive": "required", "rejudge": "required", "variant": "optional"}


def _require_run_date(run_date: str | None, archive: str) -> str:
    """§6.5: this table declares `run_date TEXT NOT NULL` (see SCHEMA).

    iso_ts records an absent date as None, which beam/locomo store as NULL
    but LME's schema cannot.  Raise the §6-style defect here rather than
    let SQLite raise IntegrityError from inside the INSERT: an LME
    artifact carrying neither a date field nor an exec stamp is a defect
    in the artifact, and the error should name the file that caused it.
    """
    if run_date is None:
        raise ValueError(
            f"no usable run_date for {archive!r}: this benchmark's table "
            "declares run_date NOT NULL -- an artifact with no date field "
            "and no exec stamp is a defect, not a domain fact"
        )
    return run_date


def _stamp_fields(kind: str, archive: str, data: dict, cfg: dict):
    """(run_date, source_date, total_tokens, elapsed_s) under §6 rules.

    Rejudge: run_date = last stem stamp (rejudge exec; the artifact's own
    date field is the SOURCE's date, inherited) or the artifact date fallback;
    source_date = first stamp of rejudged_from (the source pointer); stats
    = NULL (inherited-but-wrong is worse than missing).
    Archive: run_date = artifact date; source_date = first stem stamp
    (required policy -> raises if a stamp-bearing name lacks one).
    Variant: same, optional policy (analyst labels may carry no stamp).
    """
    policy = STAMP_POLICY.get(kind, "optional")
    if kind == "rejudge":
        source_date, exec_date = rr.rejudge_dates(
            archive, cfg.get("rejudged_from") or "", policy)
        return (_require_run_date(
                    rr.iso_ts(exec_date or data.get("date")), archive),
                source_date, None, None)
    return (_require_run_date(rr.iso_ts(data.get("date")), archive),
            rr.stem_source_date(archive, policy),
            cfg.get("total_tokens"), cfg.get("elapsed_s"))


def _strict_execution(data: dict) -> tuple[dict, dict]:
    """Read cumulative strict usage only when every segment is complete."""

    execution = data.get("execution")
    if not isinstance(execution, dict):
        return {}, {"strict_execution": False}
    segments = execution.get("segments")
    disclosure = {
        "strict_execution": True,
        "segments_present": isinstance(segments, list),
        "segments_complete": False,
        "segment_count": len(segments) if isinstance(segments, list) else None,
    }
    if not isinstance(segments, list) or not segments:
        return {}, disclosure
    complete = all(
        isinstance(segment, dict) and segment.get("status") == "complete"
        for segment in segments
    )
    disclosure["segments_complete"] = complete

    def number(value, *, integer=False):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            or (integer and not float(value).is_integer())
        ):
            return None
        return int(value) if integer else float(value)

    def usage_sum(key: str, field: str, availability: str):
        if not complete:
            return None
        values = []
        for segment in segments:
            usage = segment.get(key)
            if not isinstance(usage, dict) or usage.get(availability) is not True:
                return None
            value = number(usage.get(field), integer=field != "latency_s")
            if value is None:
                return None
            values.append(value)
        return sum(values)

    elapsed = None
    if complete:
        elapsed_values = [number(segment.get("elapsed_s")) for segment in segments]
        if all(value is not None for value in elapsed_values):
            elapsed = sum(elapsed_values)

    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    required_usage = ["reader_usage", "judge_usage", "retrieval_usage"]
    if "memory_pipeline" in models:
        required_usage.append("memory_pipeline_usage")
    tokens_available = complete
    total_tokens = 0
    for segment in segments:
        if not tokens_available:
            break
        for key in required_usage:
            usage = segment.get(key)
            value = number(
                usage.get("total_tokens") if isinstance(usage, dict) else None,
                integer=True,
            )
            if (
                not isinstance(usage, dict)
                or usage.get("token_usage_available") is not True
                or value is None
            ):
                tokens_available = False
                break
            total_tokens += value
        embedding = segment.get("embedding_usage")
        embedding_cfg = (data.get("config") or {}).get("embedding_runtime")
        if (
            tokens_available and isinstance(embedding_cfg, dict)
            and embedding_cfg.get("configured")
        ):
            provider_value = number(
                embedding.get("total_tokens")
                if isinstance(embedding, dict) else None,
                integer=True,
            )
            if (
                not isinstance(embedding, dict)
                or embedding.get("provider_token_usage_available") is not True
                or provider_value is None
            ):
                tokens_available = False
            else:
                total_tokens += provider_value
    disclosure["total_tokens_available"] = tokens_available
    counts = execution.get("counts") if isinstance(execution.get("counts"), dict) else {}
    return {
        "count": number(counts.get("expected"), integer=True),
        "answer_calls": usage_sum(
            "reader_usage", "calls", "calls_available"
        ),
        "judge_calls": usage_sum(
            "judge_usage", "calls", "calls_available"
        ),
        "retrieval_calls": usage_sum(
            "retrieval_usage", "calls", "calls_available"
        ),
        "total_tokens": total_tokens if tokens_available else None,
        "elapsed_s": elapsed,
    }, disclosure


def _load_registry_artifact(path: Path) -> tuple[dict, str, str, str]:
    """Resolve a pointer to immutable evidence and return canonical identity.

    Old two-field pointers remain readable because the shared reader validates
    their target manifest/run id. New pointers additionally bind the exact
    target artifact digest. Registry identity always uses the target basename,
    never the mutable pointer's filename.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkIntegrityError(f"cannot read artifact {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BenchmarkIntegrityError("artifact root must be an object")
    pointer_keys = set(raw)
    is_pointer = pointer_keys in (
        {"archive", "run_id"},
        {"archive", "run_id", "artifact_digest"},
    )
    if {"archive", "run_id"} <= pointer_keys and not is_pointer:
        raise BenchmarkIntegrityError(
            "artifact pointer has unexpected fields"
        )
    data = read_artifact_or_pointer(path)
    archive = raw["archive"] if is_pointer else path.name
    digest = content_hash(data)
    if is_pointer and "artifact_digest" in raw:
        compatibility = "pointer-target-digest-validated"
    elif is_pointer:
        compatibility = "legacy-pointer-run-id-validated-digest-computed"
    else:
        compatibility = "direct-immutable-artifact"
    return data, archive, digest, compatibility


def ingest_file(con, path: Path, overrides: dict | None = None):
    """Insert one run JSON. Returns 'inserted' | 'skipped' | 'error'."""
    overrides = overrides or {}
    try:
        data, archive, artifact_digest, pointer_compatibility = (
            _load_registry_artifact(path)
        )
    except Exception as e:
        return f"error: {e}"
    strict = strict_intent(data, path)
    validated = None
    if strict:
        if overrides:
            return "error: strict LongMemEval artifacts reject analyst overrides"
        try:
            validated = validate_strict_artifact(data, path=path)
        except (BenchmarkIntegrityError, ValueError, TypeError) as exc:
            # Strict intent is fail-closed: malformed evidence never falls back
            # to permissive legacy parsing.
            return f"error: strict LongMemEval validation failed: {exc}"
    cfg = dict(data.get("config") or {})
    # Recorded-absent levers: the adapter never wrote these into config,
    # even when active.  Fill only from explicit analyst overrides; the
    # extras blob records that they were supplied out-of-band.
    for k, v in overrides.items():
        if k in OVERRIDE_COLUMNS:
            cfg[k] = v
    legacy_retrieval_only = bool(not strict and rr.is_retrieval_only(data))
    scores = (
        validated["scores"] if validated is not None
        else ({} if legacy_retrieval_only else (data.get("scores") or {}))
    )

    def cat(name):
        v = scores.get(name)
        return round(v["accuracy"], 3) if isinstance(v, dict) else None

    overall = scores.get("OVERALL")
    overall_acc = round(overall["accuracy"], 3) if isinstance(overall, dict) else None

    # Artifact identity, not config/run_id, governs deduplication. Legitimate
    # repeat executions can share a run_id while producing different evidence.
    same_name = con.execute(
        "SELECT artifact_digest FROM runs WHERE archive=?", (archive,)
    ).fetchone()
    if same_name:
        if same_name[0] == artifact_digest:
            return "skipped"
        return (
            "error: archive basename collision with different artifact digest"
        )
    same_digest = con.execute(
        "SELECT archive FROM runs WHERE artifact_digest=? LIMIT 1",
        (artifact_digest,),
    ).fetchone()
    if same_digest:
        return "skipped"

    if "rejudged" in archive:
        kind = "rejudge"
    elif "-seed" in archive or archive.startswith("longmemeval-v2-hymem-2"):
        kind = "archive"
    else:
        kind = "variant"

    run_date, source_date, total_tokens, elapsed_s = _stamp_fields(
        kind, archive, data, cfg)
    if strict and kind != "rejudge":
        total_tokens = validated["total_tokens"]
        elapsed_s = validated["elapsed_s"]

    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    reader_model = models.get("reader") if isinstance(models.get("reader"), dict) else {}
    judge_model = models.get("judge") if isinstance(models.get("judge"), dict) else {}
    pipeline_model = (
        models.get("memory_pipeline")
        if isinstance(models.get("memory_pipeline"), dict) else {}
    )
    manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}
    effective_hymem = (
        cfg.get("effective_hymem_config")
        if isinstance(cfg.get("effective_hymem_config"), dict) else {}
    )
    for key in (
        "aggregation_nodes_enabled", "episode_granularity_enabled",
        "value_supersession_enabled",
    ):
        if strict and key not in cfg and key in effective_hymem:
            cfg[key] = effective_hymem[key]

    embedding = models.get("embedding") if isinstance(
        models.get("embedding"), dict
    ) else {}
    strict_count = validated["counts"]["expected"] if validated else None
    abstention_count = validated.get("abstention_count") if validated else None
    scored_run = bool(validated is not None and cfg.get("scored_run") is True)
    retrieval_only = bool(
        (validated is not None and cfg.get("retrieval_only") is True)
        or legacy_retrieval_only
    )
    usage_exact = bool(
        validated is not None
        and all(validated.get(key) is not None for key in (
            "answer_calls", "judge_calls", "retrieval_calls", "pipeline_calls",
            "total_tokens", "elapsed_s",
        ))
    )

    extras = {
        "config": cfg,
        "scores": scores,
        "models": data.get("models") if strict else None,
        "execution": data.get("execution") if strict else None,
        "strict_validation": (
            {
                "run_id": validated["run_id"],
                "official_protocol_aligned": validated["official_protocol_aligned"],
                "official_scoring_semantics_aligned": validated[
                    "official_scoring_semantics_aligned"
                ],
                "official_denominator_validated": validated[
                    "official_denominator_validated"
                ],
                "usage_exact": usage_exact,
                "result_digest": data.get("result_digest"),
                "artifact_digest": artifact_digest,
            } if validated else None
        ),
        "raw_json_keys": sorted(data.keys()),
        "pointer_resolution": pointer_compatibility,
        "analyst_set": overrides,  # non-empty only when --set was used
    }
    proven = ["recorded" if k not in overrides else f"analyst:{k}={overrides[k]}" for k in overrides]
    prov = "recorded" if not overrides else "recorded + " + "; ".join(proven)
    row = {
        "archive": archive, "kind": kind, "run_date": run_date,
        "source_date": source_date,
        "auto_ability": _to_int(cfg.get("auto_ability")),
        "no_dream": _to_int(cfg.get("no_dream")),
        "permissive_default": _to_int(cfg.get("permissive_default")),
        "embeddings": _to_int(cfg.get("embeddings")),
        "graph_facts_first": _to_int(cfg.get("graph_facts_first")),
        "graph_multihop": _to_int(cfg.get("graph_multihop")),
        "distill": _to_int(cfg.get("distill")),
        "rerank_top_k": cfg.get("rerank_top_k"), "sample": cfg.get("sample"),
        "scale": cfg.get("scale") or cfg.get("scales"), "seed": cfg.get("seed"),
        "workers": cfg.get("workers"), "top_k": cfg.get("top_k"),
        "answer_model": (
            reader_model.get("model") if strict else cfg.get("answer_model")
        ),
        "judge_model": (
            judge_model.get("model") if strict else cfg.get("judge_model")
        ),
        "mr_aggregate_additive": _to_int(cfg.get("mr_aggregate_additive")),
        "aggregation_nodes_enabled": cfg.get("aggregation_nodes_enabled"),
        "episode_granularity_enabled": cfg.get("episode_granularity_enabled"),
        "value_supersession_enabled": cfg.get("value_supersession_enabled"),
        "overall": overall_acc, "multi_session": cat("multi-session"),
        "single_session_assistant": cat("single-session-assistant"),
        "single_session_preference": cat("single-session-preference"),
        "single_session_user": cat("single-session-user"),
        "knowledge_update": cat("knowledge-update"),
        "temporal_reasoning": cat("temporal-reasoning"),
        "count": (
            strict_count if strict else
            (_to_int(cfg.get("count")) or (
                overall.get("count") if isinstance(overall, dict) else None
            ))
        ),
        "answer_calls": (
            validated["answer_calls"] if validated else cfg.get("answer_calls")
        ),
        "judge_calls": (
            validated["judge_calls"] if validated else cfg.get("judge_calls")
        ),
        "retrieval_calls": (
            validated["retrieval_calls"] if validated else cfg.get("retrieval_calls")
        ),
        "total_tokens": total_tokens, "elapsed_s": elapsed_s,
        "flags_provenance": (
            "strict-v1 validated durable evidence" if strict else prov
        ),
        "extras": json.dumps(extras, default=str),
        "run_id": validated.get("run_id") if validated else None,
        "protocol": validated.get("judge_protocol") if validated else "legacy-unvalidated",
        "protocol_split": manifest.get("protocol_split") if strict else None,
        "calibration_receipt_hash": (
            manifest.get("calibration_receipt_hash") if strict else None
        ),
        "strict_validated": int(strict),
        "official_comparable": _to_int(
            validated.get("official_comparable") if validated else None
        ),
        "development_only": _to_int(
            validated.get("development_only") if validated else None
        ),
        "official_denominator_validated": _to_int(
            validated.get("official_denominator_validated") if validated else None
        ),
        "official_protocol_aligned": _to_int(
            validated.get("official_protocol_aligned") if validated else None
        ),
        "official_scoring_semantics_aligned": _to_int(
            validated.get("official_scoring_semantics_aligned")
            if validated else None
        ),
        "scored_run": int(scored_run) if strict else (0 if retrieval_only else None),
        "retrieval_only": int(retrieval_only),
        "label_free_answer_path": _to_int(
            cfg.get("label_free_answer_path") if strict else None
        ),
        "exploratory_non_comparable": _to_int(
            cfg.get("exploratory_non_comparable") if strict else None
        ),
        "abstention_count": abstention_count,
        "answerable_count": (
            strict_count - abstention_count
            if strict_count is not None and abstention_count is not None else None
        ),
        "abstention_accuracy": (
            validated.get("abstention_accuracy") if validated else None
        ),
        "answerable_accuracy": (
            validated.get("answerable_accuracy") if validated else None
        ),
        "pipeline_calls": validated.get("pipeline_calls") if validated else None,
        "usage_exact": int(usage_exact) if strict else None,
        "dataset_revision": cfg.get("dataset_revision") if strict else None,
        "dataset_sha256": cfg.get("dataset_sha256") if strict else None,
        "source_ids_hash": cfg.get("source_ids_hash") if strict else None,
        "evaluator_commit": cfg.get("evaluator_commit") if strict else None,
        "evaluator_sha256": cfg.get("evaluator_sha256") if strict else None,
        "reader_provider": reader_model.get("provider") if strict else None,
        "reader_base_url": reader_model.get("base_url") if strict else None,
        "judge_provider": judge_model.get("provider") if strict else None,
        "judge_base_url": judge_model.get("base_url") if strict else None,
        "pipeline_provider": pipeline_model.get("provider") if strict else None,
        "pipeline_model": pipeline_model.get("model") if strict else None,
        "pipeline_base_url": pipeline_model.get("base_url") if strict else None,
        "embedding_backend": embedding.get("backend") if strict else None,
        "embedding_model": embedding.get("model") if strict else None,
        "embedding_base_url": embedding.get("base_url") if strict else None,
        "embedding_dimension": embedding.get("dimension") if strict else None,
        "embedding_quality": embedding.get("quality") if strict else None,
        "embedding_network_free": _to_int(
            embedding.get("network_free") if strict else None
        ),
        "preregistered": int(bool(cfg.get("prereg"))) if strict else None,
        "result_digest": data.get("result_digest") if strict else None,
        "artifact_digest": artifact_digest,
    }
    columns = list(row)
    placeholders = ",".join("?" for _ in columns)
    quoted_columns = ",".join(f'"{column}"' for column in columns)
    try:
        con.execute(
            f"INSERT INTO runs ({quoted_columns}) VALUES ({placeholders})",
            tuple(row[column] for column in columns),
        )
        return "inserted"
    except sqlite3.IntegrityError as exc:
        same_name = con.execute(
            "SELECT artifact_digest FROM runs WHERE archive=?", (archive,)
        ).fetchone()
        if same_name:
            if same_name[0] == artifact_digest:
                return "skipped"
            return "error: archive basename collision with different artifact digest"
        same_digest = con.execute(
            "SELECT archive FROM runs WHERE artifact_digest=? LIMIT 1",
            (artifact_digest,),
        ).fetchone()
        if same_digest:
            return "skipped"
        return f"error: registry integrity failure: {exc}"


OVERRIDE_COLUMNS = {
    "aggregation_nodes_enabled", "episode_granularity_enabled",
    "value_supersession_enabled", "auto_ability", "no_dream",
    "permissive_default", "embeddings", "graph_facts_first",
    "graph_multihop", "distill", "rerank_top_k", "sample", "scale",
    "seed", "workers", "top_k", "answer_model", "judge_model",
}


def cmd_ingest(files, overrides=None):
    con = connect()
    if not files:
        files = sorted(BENCH_DIR.glob("longmemeval-v2-hymem*.json"))
        # Exclude the "latest" pointer (it duplicates an archive) and any
        # file that isn't a run record.
        files = [f for f in files if f.name != "longmemeval-v2-hymem.json"]
    stats = {}
    for f in files:
        r = ingest_file(con, Path(f), overrides)
        stats[r] = stats.get(r, 0) + 1
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    print(f"DB: {DB}  rows={total}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    missing = con.execute(
        "SELECT COUNT(*) FROM runs WHERE aggregation_nodes_enabled IS NULL AND mr_aggregate_additive IS NULL").fetchone()[0]
    if missing:
        print(f"  NOTE: {missing} rows have neither aggregation_nodes_enabled nor mr_aggregate_additive recorded")


def cmd_list(limit=30, flag=None):
    con = connect()
    cols = ["id", "run_date", "archive", "overall", "multi_session",
            "auto_ability", "no_dream", "permissive_default",
            "aggregation_nodes_enabled", "episode_granularity_enabled", "answer_model"]
    if flag:
        cols.append(flag)
    q = f"SELECT {', '.join(cols)} FROM runs ORDER BY run_date DESC LIMIT ?"
    rows = con.execute(q, (limit,)).fetchall()
    print(" | ".join(cols))
    print("-" * 120)
    for r in rows:
        print(" | ".join("-" if v is None else str(v) for v in r))


def cmd_query(sql):
    con = connect()
    cur = con.execute(sql)
    print(" | ".join(d[0] for d in cur.description))
    for r in cur.fetchall():
        print(" | ".join("-" if v is None else str(v) for v in r))


def cmd_backfill(bench_dir=None):
    """§6.4: recompute stamp-derived fields for EVERY row from its artifact
    file; UPDATE where the value differs.  The read-back is a DIFF against
    the pre-backfill values, not a spot check that new values parse — the
    interesting failure is a row that changes when it shouldn't.

    §6.5: a row whose artifact BENCH_DIR cannot supply is UNREACHABLE,
    reported separately from unreadable/recompute-failed and returned as
    a count so main() exits nonzero.  LME artifacts all live in one dir,
    so no multi-dir search is needed here (beam splits across two) — but
    the guarantee all three registries share is that a row the backfill
    could not migrate can never read as success."""
    con = connect()
    bench = Path(bench_dir or BENCH_DIR)
    rows = con.execute(
        "SELECT id, archive, kind, run_date, source_date, total_tokens, "
        "elapsed_s FROM runs ORDER BY id").fetchall()
    changed, missing, unreachable = [], [], []
    for _id, archive, kind, cur_rd, cur_sd, cur_tt, cur_es in rows:
        if kind == "doc":
            continue
        p = bench / archive
        if not p.exists():
            unreachable.append((_id, archive, f"no artifact in {bench}"))
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            missing.append((_id, archive, f"unreadable: {e}"))
            continue
        try:
            cfg = dict(data.get("config") or {})
            new_rd, new_sd, new_tt, new_es = _stamp_fields(kind, archive, data, cfg)
        except ValueError as e:
            missing.append((_id, archive, f"recompute failed: {e}"))
            continue
        diffs = []
        if cur_rd != new_rd:
            diffs.append(("run_date", cur_rd, new_rd))
        if cur_sd != new_sd:
            diffs.append(("source_date", cur_sd, new_sd))
        if cur_tt != new_tt:
            diffs.append(("total_tokens", cur_tt, new_tt))
        if cur_es != new_es:
            diffs.append(("elapsed_s", cur_es, new_es))
        if not diffs:
            continue
        changed.append((_id, archive, kind, diffs))
        con.execute(
            "UPDATE runs SET run_date=?, source_date=?, total_tokens=?, "
            "elapsed_s=? WHERE id=?",
            (new_rd, new_sd, new_tt, new_es, _id))
    con.commit()
    print(f"backfill: {len(changed)} row(s) changed, {len(missing)} row(s) "
          f"skipped (unreadable/recompute-failed), "
          f"{len(unreachable)} row(s) UNREACHABLE")
    for _id, archive, kind, diffs in changed:
        for f, old, newv in diffs:
            print(f"  id={_id} [{kind}] {archive} {f}: {old!r} -> {newv!r}")
    for _id, archive, why in missing:
        print(f"  id={_id} {archive}: SKIPPED ({why})")
    for _id, archive, why in unreachable:
        print(f"  id={_id} {archive}: UNREACHABLE ({why})")
    if unreachable:
        print("  NOTE: unreachable rows were NOT migrated -- fix the "
              "artifact path or LME_BENCH_DIR and re-run")
    return len(unreachable)



# ---------------------------------------------------------------- audit ----
#
# The date-check this module's docstring promises ("done on top of NULLs, not
# inside them"). It exists because `aggregation_nodes_enabled = 0` in this DB
# is not one claim but three, and only one of them is a measurement:
#
#   RECORDED  the run's own config block carried the key (post-6543ee6).
#   ASSERTED  an analyst supplied it with `--set` after the fact.
#   ABSENT    NULL -- nothing recorded it at all.
#
# An ASSERTED or ABSENT 0 is a statement about what the operator MEANT. Whether
# the run actually had the layer off depends on the code that ran, and for four
# days it did not follow the intent:
#
#   2026-08-26T16:26:57Z  52adfe5  library default aggregation_nodes_enabled
#                                  False -> True (G-FLIP PASS)
#   2026-08-30T20:50:00Z  2247074  longmemeval_adapter pins the lever BOTH ways
#
# Between those two commits the adapter set only the True leg, so an
# un-flagged run inherited the library's new True. Any row in that window
# labelled 0 is contradicted by the code that produced it, however the label
# got there. Outside it the label is honoured: before, the default agreed with
# it; after, the pin enforces it.
#
# `run_date` is the END of the run -- the archive stamp is written when the
# file is. A run that ENDED after the pin may have STARTED before it, so the
# window test uses `run_date - elapsed_s` where elapsed_s is known. On the two
# 2026-08-30/31 guard arms that is the difference between a 26-minute margin
# and no margin at all, and the conservative reading is the only safe one.
AGGREGATION_DEFAULT_FLIP = "2026-08-26T16:26:57Z"   # 52adfe5
AGGREGATION_ADAPTER_PIN = "2026-08-30T20:50:00Z"    # 2247074
# Before this commit `--episode-granularity` did not exist, so no run through
# this adapter could have had it on, whatever a later --set says.
EPISODE_GRANULARITY_LEVER = "2026-08-30T20:50:00Z"  # 2247074, same commit


def _iso_z(ts):
    """Normalise the registry's several stamp spellings to a sortable Z form."""
    if not ts:
        return None
    t = str(ts).strip().replace(" ", "T")
    if t.endswith("Z"):
        t = t[:-1]
    if "+" in t[10:]:
        t = t[:10] + t[10:].split("+")[0]
    return t[:19] + "Z"


def _minus_seconds(iso, seconds):
    """`iso` less `seconds`, as a Z string. Used to turn a run's END stamp into
    its START, because that is the instant the code version is decided."""
    import datetime as _dt
    if iso is None or not seconds:
        return iso
    t = _dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
    return (t - _dt.timedelta(seconds=float(seconds))).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def label_source(provenance, column, value):
    """Where this row's value for `column` came from -- not what it says."""
    if value is None:
        return "ABSENT"
    if provenance and f"analyst:{column}=" in provenance:
        return "ASSERTED"
    return "RECORDED"


def audit_row(run_date, elapsed_s, aggr, epg, provenance):
    """Verdict for one row. Returns (start, verdict, note).

    CONTRADICTED is the only failing verdict, and it is reserved for a claim
    the code of the day could not have honoured."""
    end = _iso_z(run_date)
    start = _minus_seconds(end, elapsed_s)
    src_a = label_source(provenance, "aggregation_nodes_enabled", aggr)
    src_e = label_source(provenance, "episode_granularity_enabled", epg)

    if epg == 1 and start is not None and start < EPISODE_GRANULARITY_LEVER:
        return start, "CONTRADICTED", (
            f"episode_granularity_enabled=1 ({src_e}) on a run that started "
            f"{start}, before the lever existed ({EPISODE_GRANULARITY_LEVER}, "
            f"2247074) -- no run through this adapter could have had it on")

    if start is None:
        return start, "UNKNOWN", "no run_date, so no window can be decided"
    in_window = AGGREGATION_DEFAULT_FLIP <= start < AGGREGATION_ADAPTER_PIN
    if not in_window:
        side = "pre-flip" if start < AGGREGATION_DEFAULT_FLIP else "post-pin"
        return start, "OK", f"{side}; aggregation label {src_a} and honoured"
    if aggr == 0:
        return start, "CONTRADICTED", (
            f"aggregation_nodes_enabled=0 ({src_a}) on a run that started "
            f"{start}, inside the unpinned window -- the adapter set only the "
            f"True leg, so the layer was ON regardless of the label")
    if aggr is None:
        return start, "CONTRADICTED", (
            "aggregation_nodes_enabled is ABSENT on a run inside the unpinned "
            f"window (started {start}) -- it inherited the library's True, so "
            "this row is an aggregation-ON run and must not read as unknown")
    return start, "OK", f"in-window but labelled ON ({src_a}); no conflict"


def cmd_audit(strict=False):
    con = connect()
    rows = con.execute(
        "SELECT id, archive, kind, run_date, elapsed_s, "
        "aggregation_nodes_enabled, episode_granularity_enabled, "
        "flags_provenance FROM runs ORDER BY run_date, id").fetchall()
    results = []
    for _id, archive, kind, rd, es, aggr, epg, prov in rows:
        if kind == "doc":
            continue
        start, verdict, note = audit_row(rd, es, aggr, epg, prov)
        results.append((_id, archive, start, verdict, note))

    window = [r for r in results
              if r[2] is not None
              and AGGREGATION_DEFAULT_FLIP <= r[2] < AGGREGATION_ADAPTER_PIN]
    bad = [r for r in results if r[3] == "CONTRADICTED"]
    unknown = [r for r in results if r[3] == "UNKNOWN"]

    print(f"\n=== aggregation-label audit — {len(results)} run(s) ===")
    print(f"  unpinned window: [{AGGREGATION_DEFAULT_FLIP}, "
          f"{AGGREGATION_ADAPTER_PIN})  (52adfe5 → 2247074)")
    # A window with nothing in it is the outcome that most resembles a pass, so
    # it is stated as a count rather than left to be inferred from silence. The
    # check having no work to do is a finding about the ledger, not a clean
    # bill of health for a check that ran.
    print(f"  runs whose execution overlapped that window: {len(window)}")
    if not window:
        print("  → the contamination has no victims in this ledger: every run "
              "either predates the\n    default flip or postdates the adapter "
              "pin. Nothing here was silently aggregation-ON.")
    for _id, archive, start, verdict, note in results:
        if verdict == "OK" and not strict:
            continue
        print(f"  [{verdict}] id={_id} {archive}\n      started {start}: {note}")
    print(f"\n  CONTRADICTED {len(bad)}   UNKNOWN {len(unknown)}   "
          f"OK {len(results) - len(bad) - len(unknown)}")
    if bad:
        print("  → these rows must NOT be used as an aggregation-OFF baseline "
              "for the episode-granularity flip.")
    return len(bad)



def cmd_arms(path_a, path_b, lever):
    """Ask whether a claimed A/B pair can evidence its own contrast.

    Run BEFORE reading the two scores, for the same reason a pre-registration is
    written before a run: once the numbers are in, an unevidenced pair reads as
    a result rather than as a question."""
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    verdict, note, confounds = rr.arm_evidence(
        a.get("config"), b.get("config"), lever)
    print(f"\n=== arm evidence — {lever} ===")
    print(f"  A  {Path(path_a).name}")
    print(f"  B  {Path(path_b).name}")
    print(f"  [{verdict}] {note}")
    if confounds:
        print(f"  confounded on {len(confounds)} other key(s): "
              f"{', '.join(confounds)}")
    if verdict != rr.ARM_EVIDENCED:
        print("  → this pair cannot discharge a gate on that lever. The scores "
              "may still be\n    correct; what is missing is any way to tell a "
              "real null from two runs of one arm.")
        return 1
    return 0


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        ov = {}
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--set" and i + 1 < len(args):
                k, v = args[i + 1].split("=", 1)
                ov[k] = v
                args.pop(i)
                args.pop(i)
            else:
                i += 1
        cmd_ingest(args or None, ov)  # noqa: E1123
    elif cmd == "backfill":
        if cmd_backfill():
            sys.exit(1)   # §6.5: unreachable rows were not migrated
    elif cmd == "list":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--limit", type=int, default=30)
        p.add_argument("--flag")
        a = p.parse_args(sys.argv[2:])
        cmd_list(a.limit, a.flag)
    elif cmd == "query":
        cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "arms":
        import argparse
        p = argparse.ArgumentParser(prog="lme_registry.py arms")
        p.add_argument("a")
        p.add_argument("b")
        p.add_argument("--lever", required=True)
        a = p.parse_args(sys.argv[2:])
        if cmd_arms(a.a, a.b, a.lever):
            sys.exit(1)
    elif cmd == "audit":
        if cmd_audit(strict="--strict" in sys.argv[2:]):
            sys.exit(1)   # a contradicted label is a hard failure, not a note
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
