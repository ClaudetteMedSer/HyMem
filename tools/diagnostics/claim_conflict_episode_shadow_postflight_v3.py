#!/usr/bin/env python3
"""Episode-shadow postflight: sealed source, strategy attribution, durable health.

Run only after worker exit, with the entire source directory mounted read-only,
network disabled, and no runtime environment supplied by the host.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import stat
import re
import sqlite3
import sys
import traceback

sys.path.insert(0, "/candidate")

AUDIT_FILE = Path("/diag/claim_conflict_store_audit.py")
AUDIT_SHA256 = "e2efe365c5aedbbe88d86d521dd37b821759afc5fb21c21a6662e6a8ce567f42"
TARGET_CHUNK = "chk_c14182771bd8e2a7e583f7d693cee29a5fb159fe"
TARGET_GENERATION = (
    "hymem-phase1-generation-v1:"
    "6075085e12e32e1b790e49b99b8c3bb50718b18be0f28762d1582c58ee8e35eb"
)
PHASE1_SHA256 = "bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac"
WORK = Path("/work")
EMBEDDING_CLIENT = None
SEMANTIC_CONFIG_HASHES = {}


def offline_provider_guard():
    """Reject maintained embedding execution without changing producer methods."""
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
    forbidden = {OpenAICompatibleEmbeddingClient.embed.__code__,
                 OpenAICompatibleEmbeddingClient._embed_with_locked_transport.__code__}
    def guard(frame, event, _value):
        if event == "call" and frame.f_code in forbidden:
            raise RuntimeError("offline_embedding_execution_forbidden")
    return guard


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def regular(path, mode=None):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError('sealed_input_not_regular')
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        raise ValueError('sealed_input_mode_invalid')
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError('sealed_input_parent_symlink')


def sealed(path, expected, mode=None):
    regular(path, mode)
    for suffix in ('-wal', '-journal'):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            regular(sidecar)
            if sidecar.stat().st_size:
                raise ValueError('sealed_input_nonempty_sidecar')
    if digest(path) != expected:
        raise ValueError('sealed_input_pin_drift')


def backup_reader(pins, work):
    """Build the sole replacement callback; no global SQLite changes."""
    def backup(source, target):
        if source not in pins:
            raise ValueError('sealed_input_path_invalid')
        expected, mode, name = pins[source]
        if target != work / name:
            raise ValueError('sealed_output_path_invalid')
        if (not work.is_dir() or work.is_symlink()
                or stat.S_IMODE(work.stat().st_mode) != 0o700):
            raise ValueError('private_work_directory_invalid')
        sealed(source, expected, mode)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        try:
            origin = sqlite3.connect(source.as_uri() + '?mode=ro&immutable=1', uri=True)
            try:
                copy = sqlite3.connect(target)
                try:
                    origin.backup(copy)
                finally:
                    copy.close()
            finally:
                origin.close()
        finally:
            sealed(source, expected, mode)
    return backup



def load_audit(path: Path = AUDIT_FILE):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("audit_helper_invalid")
    if hashlib.sha256(path.read_bytes()).hexdigest() != AUDIT_SHA256:
        raise RuntimeError("audit_helper_pin_drift")
    spec = importlib.util.spec_from_file_location("private_dream_pinned_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("audit_helper_unloadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quarantine_ids(conn) -> dict[str, set[str]]:
    """Keep identifiers in memory; only cardinalities leave the worker."""
    return {
        "retry_limit": {str(row[0]) for row in conn.execute(
            "SELECT DISTINCT chunk_id FROM chunk_extraction_attempts WHERE attempts>=3")},
        "terminal_loss": {str(row[0]) for row in conn.execute(
            "SELECT chunk_id FROM chunk_extraction_terminal_losses")},
    }


def publication_counts(conn, *, has_proof_column: bool,
                       target: str = TARGET_CHUNK,
                       generation: str = TARGET_GENERATION) -> dict[str, int]:
    current = int(conn.execute(
        "SELECT COUNT(*) FROM current_phase1_publications WHERE chunk_id=?",
        (target,),
    ).fetchone()[0])
    pinned = int(conn.execute(
        "SELECT COUNT(*) FROM current_phase1_publications "
        "WHERE chunk_id=? AND phase1_generation_key=?",
        (target, generation),
    ).fetchone()[0])
    with_proof = int(conn.execute(
        "SELECT COUNT(*) FROM current_phase1_publications publication "
        "JOIN kg_claim_extraction_outcomes claim "
        "ON claim.chunk_id=publication.chunk_id "
        "AND claim.phase1_generation_key=publication.phase1_generation_key "
        "AND claim.prompt_version=publication.prompt_version "
        "WHERE publication.chunk_id=? AND publication.phase1_generation_key=? "
        "AND claim.local_replay_proof IS NOT NULL",
        (target, generation),
    ).fetchone()[0]) if has_proof_column else 0
    return {"target_current": current, "target_pinned": pinned,
            "target_pinned_with_proof": with_proof}


def validate_worker_report(report: object) -> dict:
    from hymem.dreaming.runner import (
        DREAM_REPORT_COUNT_FIELDS, DREAM_REPORT_ERROR_FIELDS,
        DREAM_REPORT_NULLABLE_COUNT_FIELDS, DREAM_REPORT_TEXT_FIELDS,
        DREAM_REPORT_BOOLEAN_GATE_FIELDS,
    )

    expected = set(DREAM_REPORT_COUNT_FIELDS + DREAM_REPORT_ERROR_FIELDS
                   + DREAM_REPORT_NULLABLE_COUNT_FIELDS
                   + DREAM_REPORT_BOOLEAN_GATE_FIELDS)
    if set(DREAM_REPORT_TEXT_FIELDS) != {"aggregation_blocking"}:
        raise RuntimeError("candidate_dream_report_partition_changed")
    if not isinstance(report, dict) or set(report) != expected:
        raise ValueError("worker_report_fields_invalid")
    for name in DREAM_REPORT_COUNT_FIELDS + DREAM_REPORT_ERROR_FIELDS:
        if type(report[name]) is not int or report[name] < 0:
            raise ValueError("worker_report_value_invalid")
    for name in DREAM_REPORT_NULLABLE_COUNT_FIELDS:
        value = report[name]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("worker_report_value_invalid")
    for name in DREAM_REPORT_BOOLEAN_GATE_FIELDS:
        if type(report[name]) is not bool:
            raise ValueError("worker_report_value_invalid")
    return report


def offline_embedding_client(path: Path, expected: str):
    """Rebuild maintained identity from a sealed public profile without credentials."""
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("embedding_profile_seal_invalid")
    regular(path, 0o400)
    if digest(path) != expected:
        raise ValueError("embedding_profile_pin_drift")
    profile = json.loads(path.read_bytes())
    fields = {"base_url", "model", "dim", "timeout", "deployment_revision", "deployment_tenant", "allow_insecure_internal_http"}
    if not isinstance(profile, dict) or set(profile) != fields:
        raise ValueError("embedding_profile_fields_invalid")
    if any(type(profile[name]) is not str or not profile[name] for name in (
        "base_url", "model", "deployment_revision", "deployment_tenant",
    )) or type(profile["dim"]) is not int or profile["dim"] <= 0 or (
        type(profile["timeout"]) not in (int, float) or not 0 < profile["timeout"] <= 120
    ):
        raise ValueError("embedding_profile_values_invalid")
    if type(profile["allow_insecure_internal_http"]) is not bool:
        raise ValueError("embedding_profile_values_invalid")
    allow_internal = profile.pop("allow_insecure_internal_http")
    # Explicit inert key prevents environment credential resolution. The SDK
    # constructor creates a transport but sends no request; callers never embed.
    from hymem.contrib.endpoint_policy import EMBEDDING_INTERNAL_HTTP_ENV
    prior = os.environ.get(EMBEDDING_INTERNAL_HTTP_ENV)
    try:
        os.environ[EMBEDDING_INTERNAL_HTTP_ENV] = "1" if allow_internal else "0"
        client = OpenAICompatibleEmbeddingClient(
            api_key="offline-postflight-no-network", pin_dimension=True, **profile)
    finally:
        if prior is None:
            os.environ.pop(EMBEDDING_INTERNAL_HTTP_ENV, None)
        else:
            os.environ[EMBEDDING_INTERNAL_HTTP_ENV] = prior
    from hymem.dreaming.aggregation_material import embedding_execution_identity
    binding, _model, _dimension = embedding_execution_identity(client)
    if binding["identity_exact"] is not True or binding["reuse_scope"] != "durable":
        client.close()
        raise ValueError("embedding_profile_identity_not_exact")
    return client


VALID_STRATEGIES = {"exact"} | {
    "exact:" + reason for reason in (
        "disabled", "vec_extension_unavailable", "vec_table_unavailable",
        "mixed_vector_space", "vec_metadata_mismatch", "vec_shadow_unverifiable",
        "vec_shadow_mismatch", "vec_query_failed",
    )
} | {"entity_only:no_valid_episode_vectors", "knn:verified_full_shadow"}


def episode_vector_alignment(conn) -> bool:
    """Read the complete shadow, rejecting the repair probe's unverifiable case."""
    from hymem.core import db
    from hymem.dreaming.aggregate import load_clusterable_episodes

    try:
        if not db._load_vec_extension(conn) or not db.has_vec_table(conn, table="vec_episodes"):
            return False
        dim_row = conn.execute("SELECT value FROM schema_meta WHERE key='vec_dim'").fetchone()
        model_row = conn.execute("SELECT value FROM schema_meta WHERE key='vec_model'").fetchone()
        if dim_row is None or model_row is None:
            return False
        dimension = int(dim_row["value"])
        model = model_row["value"]
        if dimension <= 0 or not isinstance(model, str) or re.fullmatch(
            r"hymem-embedding-producer-v1:[0-9a-f]{64}", model
        ) is None:
            return False
        episodes = load_clusterable_episodes(conn, max_rowid=None,
            embedding_model=model, embedding_dim=dimension)
        expected = {}
        for episode in episodes:
            if episode["vector"] is None:
                continue
            vector = db._finite_vec(episode["vector"], dimension)
            if vector is None:
                return False
            key = int(episode["rowid"])
            if key in expected:
                return False
            expected[key] = db._pack_vector(vector)
        actual_rows = conn.execute("SELECT rowid,embedding FROM vec_episodes ORDER BY rowid").fetchall()
        actual = {int(row["rowid"]): bytes(row["embedding"]) for row in actual_rows}
        return (len(actual_rows) == len(actual) and actual == expected
                and db.vec_episodes_aligned(conn) is True)
    except (sqlite3.Error, AttributeError, UnicodeError, TypeError, ValueError, OverflowError):
        return False


def aggregation_evidence(conn, row, report, embedding_client=None) -> dict[str, bool]:
    """Strategy labels describe pair selection; durable publication proves success."""
    from hymem.dreaming.aggregation_provenance import load_current_aggregation_publication

    strategy_valid = row["aggregation_blocking"] in VALID_STRATEGIES
    health = conn.execute("SELECT * FROM aggregation_build_health WHERE id=1").fetchone()
    config = row["aggregation_config_version"]
    generation = row["aggregation_generation_key"]
    material = row["aggregation_material_epoch_key"]
    identity = all(isinstance(value, str) and value for value in (config, generation, material))
    pending_clear = health is not None and all(
        health[name] is None for name in (
            "pending_config_version", "pending_generation_key",
            "pending_material_epoch_key", "pending_attempt_token",
            "first_pending_at", "last_attempt_at",
        )
    ) and all(health[name] == 0 for name in (
        "pending_attempts", "pending_caught_exceptions", "pending_fusion_failures",
    ))
    success_matches = bool(identity and health is not None
        and health["last_success_config_version"] == config
        and health["last_success_generation_key"] == generation
        and health["last_success_material_epoch_key"] == material
        and health["last_success_at"] is not None
        and type(health["attempt_serial"]) is int and health["attempt_serial"] > 0)
    success_in_run = bool(health is not None
        and isinstance(row["started_at"], str) and isinstance(row["ended_at"], str)
        and isinstance(health["last_success_at"], str)
        and row["started_at"] <= health["last_success_at"] <= row["ended_at"])
    publication = load_current_aggregation_publication(
        conn, expected_config_version=config,
        expected_generation_key=generation, expected_material_epoch_key=material,
        embedding_client=embedding_client,
    ) if identity else None
    publication_matches = bool(publication is not None
        and publication.config_version == config
        and publication.generation_key == generation
        and publication.material_epoch_key == material
        and len(publication.nodes) == report["aggregation_nodes_built"])
    result = {
        "strategy_valid": strategy_valid, "pending_clear": pending_clear,
        "success_matches": success_matches, "publication_matches": publication_matches,
        "success_in_run": success_in_run,
        "enabled_in_run": row["aggregation_effective"] == "enabled",
    }
    result["healthy"] = all(result.values())
    return result



def semantic_config_hashes(path: Path, expected: str) -> dict[str, str]:
    regular(path, 0o400)
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None or digest(path) != expected:
        raise ValueError("semantic_config_seal_invalid")
    values = json.loads(path.read_bytes())
    if not isinstance(values, dict) or set(values) != {"digest", "profile", "facts"} or not all(
        isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in values.values()
    ):
        raise ValueError("semantic_config_fields_invalid")
    return values


def profile_stage_chain_valid(conn, session, state) -> bool:
    """Read-only equivalent of the maintained profile publication chain gates."""
    from hymem.dreaming import user_profile as profile
    from hymem.dreaming.lossless import lossless_cursor_is_valid, covered_messages_after
    from hymem.extraction.jsonio import loads_exact_or_fenced

    generation = state["profile_cursor_prompt_version"]
    rows = conn.execute(
        "SELECT * FROM profile_staging WHERE session_id=? AND generation=? "
        "ORDER BY COALESCE(start_message_id,-1),start_message_offset,COALESCE(end_message_id,-1),slice_key",
        (session, generation),
    ).fetchall()
    if not rows:
        return False
    def cursor(row, prefix):
        return (row[prefix + "_message_id"], row[prefix + "_partial_message_id"], row[prefix + "_offset"])
    expected = cursor(rows[0], "cursor_before")
    if generation != state["profile_published_generation"] and expected != (None, None, 0):
        return False
    seen = {}
    for row in rows:
        before, after = cursor(row, "cursor_before"), cursor(row, "cursor_after")
        if before != expected or before == after or not all(
            lossless_cursor_is_valid(conn, session, *position, roles=frozenset({"user"}))
            for position in (before, after)
        ):
            return False
        # Bind each stage to the exact ordered USER source interval.
        end = after[1] if after[1] is not None else after[0]
        sources = covered_messages_after(conn, session, before[0], roles=frozenset({"user"}),
                                         through_message_id=end)
        if (not sources or sources[-1].message_id != end
            or (before[1] is not None and sources[0].message_id != before[1])
            or (before[0] == after[0] and after[2] <= before[2])):
            return False
        allowed = {part.message_id for part in sources}
        items = loads_exact_or_fenced(row["items_json"])
        if not isinstance(items, list):
            return False
        for item in items:
            if not isinstance(item, dict):
                return False
            keys = {"slot", "value", "evidence_message_id", "confidence", "source_session_id", "source_created_at"}
            if item.get("slot") == "relationship":
                keys.add("slot_key")
            if set(item) != keys or item["source_session_id"] != session:
                return False
            mid = item["evidence_message_id"]
            contract = {key: item[key] for key in ("slot", "value", "evidence_message_id", "confidence")}
            if item.get("slot") == "relationship":
                contract["slot_key"] = item["slot_key"]
            clean = profile._validate_profile_item(contract, allowed)
            if type(mid) is not int or clean is None:
                return False
            source_mid, source_session, _created, _live = profile._resolve_profile_source(conn, item)
            if source_mid != mid or source_session != session:
                return False
            identity = (clean["slot"], clean.get("slot_key"), source_mid)
            if clean["slot"] in profile.SINGLE_VALUED_SLOTS or clean["slot"] == "relationship":
                previous = seen.get(identity)
                if previous is not None and not profile._same_value(previous, clean["value"]):
                    return False
                seen[identity] = clean["value"]
        expected = after
    return expected == (state["profile_cursor_message_id"], state["profile_cursor_partial_message_id"], state["profile_cursor_offset"])


def new_epoch_reset_required(conn, session, state, domain, config) -> bool:
    """Admit only the observed maintained config-change reset, not UUID relabels."""
    from hymem.dreaming import digest as digest_module, user_profile
    del conn, session
    previous = state[domain + "_cursor_prompt_version"]
    matcher = (digest_module.digest_generation_matches_config if domain == "digest"
               else user_profile.profile_generation_matches_config)
    return not matcher(previous, config)


def bounded_progress_evidence(before, after, target: str = TARGET_CHUNK,
                              approved_configs: dict[str, str] | None = None) -> dict:
    """Prove progress within the active epoch, never compare obsolete cursors."""
    from hymem.dreaming import digest as digest_module, user_profile, facts
    from hymem.dreaming.lossless import lossless_cursor_is_valid

    approved = SEMANTIC_CONFIG_HASHES if approved_configs is None else approved_configs
    result = {"valid": False, "pending_domains": 0, "progressed_domains": 0,
              "new_epoch_progress_domains": 0, "source_messages_unchanged": False,
              "current_configs_match": False, "staging_valid": False}
    if set(approved) != {"digest", "profile", "facts"}:
        return result
    try:
        found = after.execute("SELECT session_id FROM chunks WHERE id=?", (target,)).fetchone()
        if found is None:
            return result
        session = found["session_id"]
        prior = before.execute("SELECT * FROM sessions WHERE id=?", (session,)).fetchone()
        current = after.execute("SELECT * FROM sessions WHERE id=?", (session,)).fetchone()
        if prior is None or current is None:
            return result
        def messages(conn):
            return [tuple(row) for row in conn.execute(
                "SELECT id,role,content FROM messages WHERE session_id=? ORDER BY id", (session,))]
        original, latest = messages(before), messages(after)
        result["source_messages_unchanged"] = original == latest and bool(latest)
        valid = result["source_messages_unchanged"]
        configs_match = stages_valid = True
        for domain in ("digest", "profile", "facts"):
            generation = current[domain + "_cursor_prompt_version"]
            if not isinstance(generation, str):
                return result
            config = generation.rsplit("|walk=", 1)[0] if domain != "facts" else generation
            recognized = (digest_module.digest_generation_is_recognized(generation)
                          and digest_module.digest_generation_matches_config(generation, config)) if domain == "digest" else (
                user_profile.profile_generation_is_recognized(generation)
                and user_profile.profile_generation_matches_config(generation, config)) if domain == "profile" else facts.facts_generation_is_recognized(generation)
            matches = recognized and hashlib.sha256(config.encode()).hexdigest() == approved[domain]
            configs_match = configs_match and matches
            names = (domain + "_cursor_message_id", domain + "_cursor_partial_message_id", domain + "_cursor_offset")
            cursor = tuple(current[name] for name in names)
            old_cursor = tuple(prior[name] for name in names)
            roles = frozenset({"user"}) if domain == "profile" else frozenset({"user", "assistant"}) if domain == "facts" else None
            cursor_valid = lossless_cursor_is_valid(after, session, *cursor, roles=roles)
            valid = valid and cursor_valid and current[domain + "_retry_count"] == 0 and current[domain + "_quarantined"] == 0
            tail = max((row[0] for row in latest if roles is None or row[1] in roles), default=None)
            pending = tail is not None and (cursor[0] != tail or cursor[1] is not None)
            result["pending_domains"] += int(pending)
            if domain == "facts":
                # Facts may replay a stale outcome while retaining its numeric
                # cursor. Its pending state is checked but is not a progress witness.
                continue
            if not pending:
                continue
            stage_rows = after.execute("SELECT * FROM " + domain + "_staging WHERE session_id=? AND generation=?", (session, generation)).fetchall()
            if domain == "digest":
                completed = digest_module.load_completed_digest_slices(
                    after, session, generation, require_complete=False)
                summary, failure = digest_module.load_digest_staged_summary_state(after, session, generation, cursor)
                stage_valid = bool(stage_rows and completed
                                   and all(part["summary_failure_reason"] is None for part in completed)
                                   and summary is not None and failure is None
                                   and digest_module.digest_staging_cursor_is_valid(after, session))
            else:
                stage_valid = profile_stage_chain_valid(after, session, current)
            stages_valid = stages_valid and stage_valid
            def position(value):
                return (value[0] if value[0] is not None else -1, value[1] if value[1] is not None else -1, value[2])
            same_epoch = prior[domain + "_cursor_prompt_version"] == generation
            if same_epoch:
                progressed = position(cursor) > position(old_cursor)
            else:
                # A new config/walk cannot inherit numeric progress from the
                # obsolete epoch. Its newly committed source-valid chain is
                # the witness, and its epoch must not exist in the baseline.
                absent = before.execute("SELECT COUNT(*) FROM " + domain + "_staging WHERE session_id=? AND generation=?", (session, generation)).fetchone()[0] == 0
                begins_at_start = any(
                    row["cursor_before_message_id"] is None and row["cursor_before_partial_message_id"] is None
                    and row["cursor_before_offset"] == 0 for row in stage_rows)
                reset_required = new_epoch_reset_required(before, session, prior, domain, config)
                progressed = absent and begins_at_start and stage_valid and reset_required
                result["new_epoch_progress_domains"] += int(progressed)
            result["progressed_domains"] += int(progressed and stage_valid and matches)
        result["current_configs_match"] = bool(configs_match)
        result["staging_valid"] = bool(stages_valid)
        result["valid"] = bool(valid and configs_match and stages_valid
                               and result["pending_domains"] > 0 and result["progressed_domains"] > 0)
    except (sqlite3.Error, RuntimeError, AttributeError, KeyError, TypeError, ValueError, OverflowError):
        pass
    return result


def worker_evidence(path: Path, audit, dream, summary: dict) -> tuple[dict, bool, dict]:
    audit.checked_file(path, mode=0o600)
    worker = json.loads(path.read_bytes())
    if not isinstance(worker, dict) or (
        worker.get("status") != "completed"
        or worker.get("source_sha256") != audit.REFERENCE_SHA
        or worker.get("phase1_sha256") != PHASE1_SHA256
        or worker.get("target_chunk_id") != TARGET_CHUNK
        or worker.get("generation_key") != TARGET_GENERATION
        or worker.get("runtime_generation_verified") is not True
    ):
        raise ValueError("worker_result_identity_invalid")
    report = validate_worker_report(worker.get("report"))
    if type(worker.get("chunks_processed")) is not int or (
        worker["chunks_processed"] != report["chunks_processed"]
    ):
        raise ValueError("worker_result_count_invalid")
    row = dream.execute("SELECT * FROM dream_runs ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        raise ValueError("worker_dream_run_missing")
    persisted = set(row.keys()) & set(report)
    consistent = int(row["id"]) == summary["dream"]["latest_dream"]["id"]
    for name in persisted:
        value = report[name]
        if type(value) is bool:
            value = int(value)
        if row[name] != value:
            consistent = False
    campaign_processed = summary["campaign"].get("chunks_processed")
    if campaign_processed is not None and campaign_processed != report["chunks_processed"]:
        consistent = False
    blocking = row["aggregation_blocking"]
    if not isinstance(blocking, str):
        raise ValueError("dream_aggregation_blocking_invalid")
    return report, consistent, aggregation_evidence(dream, row, report, EMBEDDING_CLIENT)


def assess(summary: dict, baseline_version: int, dream_version: int,
           before_publications: dict[str, int], after_publications: dict[str, int],
           before_quarantines: dict[str, set[str]],
           after_quarantines: dict[str, set[str]],
           degraded_counters: tuple[str, ...], worker_report: dict,
           stored_counter_consistent: bool,
           aggregation: dict[str, bool], bounded_progress: dict | None = None) -> dict:
    from hymem.dreaming.runner import (
        DREAM_REPORT_ERROR_FIELDS, DREAM_REPORT_BOOLEAN_GATE_FIELDS,
    )

    worker_report = validate_worker_report(worker_report)
    required = {"target_current", "target_pinned", "target_pinned_with_proof"}
    if set(before_publications) != required or set(after_publications) != required:
        raise ValueError("publication_fields_invalid")
    if set(before_quarantines) != {"retry_limit", "terminal_loss"} or set(after_quarantines) != {"retry_limit", "terminal_loss"}:
        raise ValueError("quarantine_fields_invalid")
    schema_ok = baseline_version == 63 and dream_version == 64
    publication_ok = (before_publications["target_current"] == 0
                      and after_publications == {"target_current": 1,
                                                 "target_pinned": 1,
                                                 "target_pinned_with_proof": 1})
    new_quarantines = {
        kind: len(after_quarantines[kind] - before_quarantines[kind])
        for kind in ("retry_limit", "terminal_loss")
    }
    quarantine_counts = {
        kind: {"baseline": len(before_quarantines[kind]),
               "dream": len(after_quarantines[kind]),
               "introduced": new_quarantines[kind]}
        for kind in new_quarantines
    }
    counters = summary["dream"]["latest_dream"]["counters"]
    run_degraded = any(counters.get(name, 0) > 0 for name in degraded_counters)
    error_counters_clear = all(worker_report[name] == 0 for name in DREAM_REPORT_ERROR_FIELDS)
    hard_boolean_gates_clear = all(worker_report[name] is False for name in DREAM_REPORT_BOOLEAN_GATE_FIELDS if name != "budget_exhausted")
    bounded_slice_valid = bool(bounded_progress is not None and bounded_progress.get("valid") is True)
    budget_stop_explained = (worker_report["budget_exhausted"] is False or bounded_slice_valid)
    worker_clean = error_counters_clear and hard_boolean_gates_clear and budget_stop_explained
    no_new_degradation = (not run_degraded and worker_clean
                          and aggregation.get("healthy") is True
                          and all(n == 0 for n in new_quarantines.values()))
    passed = (schema_ok and publication_ok and summary["audit_clean"] is True
              and summary["source_unchanged"] is True
              and summary["dream_completed"] is True
              and summary["campaign"]["worker_status"] == "completed"
              and worker_report["chunks_processed"] > 0
              and stored_counter_consistent
              and no_new_degradation)
    return {
        "status": "pass" if passed else "fail",
        "repair_passed": passed,
        "convergence_verified": False,
        "bounded_work_remaining": worker_report["budget_exhausted"],
        "bounded_progress": bounded_progress,
        "gates": {
            "schema_transition": schema_ok,
            "target_publication": publication_ok,
            "audit_clean": summary["audit_clean"] is True,
            "source_unchanged": summary["source_unchanged"] is True,
            "dream_completed": summary["dream_completed"] is True,
            "campaign_completed": summary["campaign"]["worker_status"] == "completed",
            "processed_positive": worker_report["chunks_processed"] > 0,
            "stored_counters_match": stored_counter_consistent,
            "worker_error_counters_zero": all(worker_report[name] == 0 for name in DREAM_REPORT_ERROR_FIELDS),
        "worker_boolean_gates_clear": all(worker_report[name] is False for name in DREAM_REPORT_BOOLEAN_GATE_FIELDS),
            "hard_boolean_gates_clear": hard_boolean_gates_clear,
            "budget_stop_explained": budget_stop_explained,
            "aggregation_healthy": aggregation.get("healthy") is True,
            "new_quarantines_absent": all(n == 0 for n in new_quarantines.values()),
            "stored_degraded_counters_zero": not run_degraded,
        },
        "worker_error_counters": {name: worker_report[name] for name in DREAM_REPORT_ERROR_FIELDS},
        "worker_boolean_gates": {name: worker_report[name] for name in DREAM_REPORT_BOOLEAN_GATE_FIELDS},
        "schema": {"baseline": baseline_version, "dream": dream_version,
                   "expected_transition": schema_ok},
        "publication": {"baseline": before_publications,
                        "dream": after_publications, "valid": publication_ok},
        "quarantines": quarantine_counts,
        "baseline_quarantines_present": any(before_quarantines.values()),
        "new_degradation_detected": not no_new_degradation,
        "worker_report_clean": worker_clean,
        "worker_store_campaign_consistent": stored_counter_consistent,
        "aggregation": aggregation,
        "audit_clean": summary["audit_clean"],
        "source_unchanged": summary["source_unchanged"],
        "baseline_sha256": summary["baseline_sha256"],
        "source_sha256": summary["source_sha256"],
        "latest_dream_advanced": summary["latest_dream_advanced"],
        "dream_completed": summary["dream_completed"],
        "campaign": summary["campaign"],
        "baseline_counts": summary["baseline"]["counts"],
        "dream_counts": summary["dream"]["counts"],
        "dream_run_counters": counters,
        "baseline_integrity": summary["baseline"]["integrity"],
        "dream_integrity": summary["dream"]["integrity"],
    }


def postflight(audit, worker_result: Path) -> dict:
    from hymem.core.db import schema_version

    summary = audit.audit()
    baseline = audit.inspect_copy(WORK / "baseline.sqlite")
    dream = audit.inspect_copy(WORK / "dream.sqlite")
    try:
        before_version, after_version = schema_version(baseline), schema_version(dream)
        proof_columns = {str(row["name"]) for row in dream.execute(
            "PRAGMA table_info(kg_claim_extraction_outcomes)")}
        if after_version == 64 and "local_replay_proof" not in proof_columns:
            raise ValueError("dream_proof_column_missing")
        worker_report, stored_match, aggregation_blocking = worker_evidence(
            worker_result, audit, dream, summary)
        bounded_progress = bounded_progress_evidence(baseline, dream)
        report = assess(summary, before_version, after_version,
                        publication_counts(baseline, has_proof_column=False),
                        publication_counts(dream, has_proof_column="local_replay_proof" in proof_columns),
                        quarantine_ids(baseline), quarantine_ids(dream),
                        audit.DEGRADED_COUNTERS, worker_report, stored_match,
                        aggregation_blocking, bounded_progress)
        report["episode_vector_alignment"] = episode_vector_alignment(dream)
        report["gates"]["episode_vector_alignment"] = report["episode_vector_alignment"]
        if report["episode_vector_alignment"] is not True:
            report["status"] = "fail"
            report["repair_passed"] = False
    finally:
        baseline.close()
        dream.close()
    # The second inspection must not invalidate the source immutability result.
    still_unchanged = (audit.sha(audit.BASELINE) == report["baseline_sha256"]
                       and audit.sha(audit.SOURCE) == report["source_sha256"])
    report["source_unchanged"] = report["source_unchanged"] and still_unchanged
    if not report["source_unchanged"]:
        report["status"] = "fail"
        report["repair_passed"] = False
        report["gates"]["source_unchanged"] = False
    return report


def main() -> int:
    global WORK, EMBEDDING_CLIENT, SEMANTIC_CONFIG_HASHES
    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-helper", type=Path, default=AUDIT_FILE)
    parser.add_argument("--semantic-config-hashes", type=Path, required=True)
    parser.add_argument("--semantic-config-sha256", required=True)
    parser.add_argument("--embedding-profile", type=Path, required=True)
    parser.add_argument("--embedding-profile-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source", type=Path, default=Path("/private-dream/hymem.sqlite"))
    parser.add_argument("--baseline", type=Path, default=Path("/reference/source.sqlite"))
    parser.add_argument("--result", type=Path, default=Path("/campaign/result.json"))
    parser.add_argument("--worker-result", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    WORK = args.work
    if re.fullmatch(r"[0-9a-f]{64}", args.source_sha256) is None:
        raise ValueError("source_seal_invalid")
    if (not WORK.is_dir() or WORK.is_symlink()
            or stat.S_IMODE(WORK.stat().st_mode) != 0o700 or any(WORK.iterdir())):
        raise ValueError("private_work_directory_not_fresh")
    audit = load_audit(args.audit_helper)
    audit.backup_readonly = backup_reader({
        args.baseline: (audit.REFERENCE_SHA, 0o400, "baseline.sqlite"),
        args.source: (args.source_sha256, None, "dream.sqlite"),
    }, WORK)
    audit.SOURCE, audit.BASELINE, audit.RESULT, audit.WORK = (
        args.source, args.baseline, args.result, WORK)
    SEMANTIC_CONFIG_HASHES = semantic_config_hashes(args.semantic_config_hashes, args.semantic_config_sha256)
    EMBEDDING_CLIENT = offline_embedding_client(args.embedding_profile, args.embedding_profile_sha256)
    prior_profile = sys.getprofile()
    try:
        sys.setprofile(offline_provider_guard())
        report = postflight(audit, args.worker_result)
    finally:
        sys.setprofile(prior_profile)
        EMBEDDING_CLIENT.close()
    sealed(args.source, args.source_sha256)
    sealed(args.baseline, audit.REFERENCE_SHA, 0o400)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:
        captured = False
        try:
            if WORK.is_dir() and not WORK.is_symlink() and stat.S_IMODE(WORK.stat().st_mode) == 0o700:
                # Details remain in private evidence; stdout reveals only a type.
                raw = json.dumps({
                    "type": type(exc).__name__,
                    "traceback": "".join(traceback.format_exception(exc)),
                }, sort_keys=True, ensure_ascii=True).encode()
                fd = os.open(WORK / "postflight-failure.json",
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                captured = True
        except BaseException:
            pass
        safe = type(exc).__name__ if type(exc).__name__ in {
            "ValueError", "RuntimeError", "TypeError", "OperationalError",
            "DatabaseError", "OSError", "KeyError"} else "Exception"
        print(json.dumps({"status": "error", "error_type": safe,
                          "failure_captured": captured}, sort_keys=True))
        code = 1
    raise SystemExit(code)
