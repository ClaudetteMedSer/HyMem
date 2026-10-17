"""Guarded, fact-only repair for an existing HyMem store.

Preview is read-only and makes no provider request. Probe makes at most four
fact completions without writing the store. Apply requires a fresh preview
fingerprint, creates an online backup under the dreaming lease, and touches
only fact authority/cursors plus their local vector mirror. Never route an
unreviewed production store through ``apply``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import stat
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from hymem import bootstrap
from hymem.config import HyMemConfig
from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.core import db as core_db
from hymem.dreaming import facts as fact_mod
from hymem.dreaming import runner
from hymem.dreaming.embeddings import (
    fetch_fact_embeddings, persist_fact_embeddings,
)


class RepairStopped(RuntimeError):
    """A safe stop whose message contains no source or provider text."""


MAX_PROBE_LOGICAL = 4
MAX_PROBE_HTTP = 12
MAX_REPLAY_LOGICAL = 128
MAX_FRESH_LOGICAL = 8
MAX_APPLY_LOGICAL = MAX_REPLAY_LOGICAL + MAX_FRESH_LOGICAL
MAX_APPLY_HTTP = MAX_APPLY_LOGICAL * 3
LIVE_CONTAINER_DB = Path("/home/node/.hermes/hymem.sqlite")
LIVE_HOST_DB = Path("/opt/stacks/hermes/instance1/home/.hermes/hymem.sqlite")
REVIEWED_3072_FACT_VERSIONS = frozenset({
    "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=3072|items=8|"
    "semantic=sha256:629a889c5fb0052f6326425c7eef762d9e73401bb09f4224fd52017cd2741593",
    "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=3072|items=32|"
    "semantic=sha256:545261b726153f79c65448281e0f426d8320f37c814f0fa9f5c39fa49600ab9c",
})


def _is_live_store(path: Path) -> bool:
    """Include hardlink aliases; a resolved path comparison is insufficient."""
    return any(
        canonical.exists() and path.samefile(canonical)
        for canonical in (LIVE_CONTAINER_DB, LIVE_HOST_DB)
    )


def _load_server_process_env(proc_root: Path = Path("/proc")) -> None:
    """Copy only HyMem producer settings from the unique running server env."""
    allowed = {
        "HYMEM_ROOT", "HYMEM_DREAM_MAX_FACTS_PER_SESSION",
        "HYMEM_FACTS_EXTRACTION_MAX_TOKENS",
        "HYMEM_LLM_API_KEY", "HYMEM_LLM_BASE_URL", "HYMEM_LLM_MODEL",
        "HYMEM_LLM_THINKING", "HYMEM_LLM_EXTRA_BODY",
        "HYMEM_LLM_DEPLOYMENT_REVISION", "HYMEM_LLM_DEPLOYMENT_TENANT",
        "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
        "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM",
        "HYMEM_EMBEDDING_PIN_DIMENSION",
        "HYMEM_EMBEDDING_DEPLOYMENT_REVISION",
        "HYMEM_EMBEDDING_DEPLOYMENT_TENANT",
        "HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP",
        "HYMEM_EMBEDDING_TIMEOUT_SECONDS",
    }
    candidates: dict[str, list[dict[str, str]]] = {
        "hymem-honcho": [], "hymem-server": [],
    }
    for process in sorted(proc_root.iterdir()):
        if not process.name.isdecimal():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
            tokens = {argument.rsplit(b"/", 1)[-1] for argument in argv}
            role = (
                "hymem-honcho" if tokens & {b"hymem-honcho", b"hymem.honcho.app"}
                else "hymem-server" if tokens & {b"hymem-server", b"hymem.server"}
                else None
            )
            if role is None:
                continue
            entries = (process / "environ").read_bytes().split(b"\0")
            variables = dict(item.split(b"=", 1) for item in entries if b"=" in item)
            if (variables.get(b"HYMEM_EMBEDDING_BASE_URL")
                    != b"http://embedding-server:8766/v1"
                    or not variables.get(b"HYMEM_LLM_API_KEY")
                    or not variables.get(b"HYMEM_EMBEDDING_API_KEY")):
                continue
            selected = {
                key.decode("ascii"): value.decode("utf-8")
                for key, value in variables.items()
                if key.decode("ascii") in allowed
            }
        except (OSError, UnicodeError, ValueError):
            continue
        candidates[role].append(selected)
    if (any(len(values) != 1 for values in candidates.values())
            or candidates["hymem-honcho"][0] != candidates["hymem-server"][0]):
        raise RepairStopped("matching running Honcho/MCP producer environments unavailable")
    for key in tuple(os.environ):
        if key.startswith(("HYMEM_LLM_", "HYMEM_EMBEDDING_")) or key in {
            "HYMEM_DREAM_MAX_FACTS_PER_SESSION",
            "HYMEM_FACTS_EXTRACTION_MAX_TOKENS", "HYMEM_ROOT",
        }:
            del os.environ[key]
    os.environ.update(candidates["hymem-honcho"][0])


def _client_from_env() -> OpenAICompatibleClient:
    env = bootstrap.resolve_env()
    if not env.has_llm_key:
        raise RepairStopped("configured LLM credential is unavailable")
    return OpenAICompatibleClient(
        api_key=env.llm_api_key, base_url=env.llm_base_url,
        model=env.llm_model,
    )


def _embedder_from_env():
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
    from hymem.extraction.embeddings import CachedEmbeddingClient

    env = bootstrap.resolve_env()
    route = urlsplit(env.embedding_base_url)
    if (env.embedding_backend != "openai_compatible"
            or route.scheme != "http" or route.hostname != "embedding-server"
            or route.port != 8766 or route.path.rstrip("/") != "/v1"
            or route.username is not None or route.password is not None
            or route.query or route.fragment):
        raise RepairStopped("embedding route is not the pinned local 8766/v1 service")
    if not (env.embedding_pin_dimension and env.embedding_deployment_revision
            and env.embedding_deployment_tenant and env.has_embedding_key):
        raise RepairStopped("configured embedding identity is not pinned")
    transport = OpenAICompatibleEmbeddingClient(
        api_key=env.embedding_api_key, base_url=env.embedding_base_url,
        model=env.embedding_model, dim=env.embedding_dim,
        pin_dimension=env.embedding_pin_dimension,
        deployment_revision=env.embedding_deployment_revision,
        deployment_tenant=env.embedding_deployment_tenant,
    )
    return CachedEmbeddingClient(transport)


def _configuration(db_path: Path, cap: int, fact_tokens: int = 3072) -> HyMemConfig:
    if (cap, fact_tokens) not in {(16, 3072), (32, 3072), (64, 6144)}:
        raise RepairStopped("only reviewed fact caps and token budgets are supported")
    return HyMemConfig(
        root=db_path.parent, dream_max_facts_per_session=cap,
        facts_extraction_max_tokens=fact_tokens,
    )


def _allowed_v4_fields(target_version: str) -> set[str]:
    target_base = target_version.split("|semantic=", 1)[0]
    target_fields = target_base.rsplit("|items=", 1)[0]
    allowed = {target_fields}
    if target_fields.endswith("|tokens=6144"):
        allowed.add(target_fields.removesuffix("|tokens=6144") + "|tokens=3072")
    return allowed


def _v4_version_allowed(version: str, target_version: str) -> bool:
    target_fields = target_version.split("|semantic=", 1)[0].rsplit("|items=", 1)[0]
    if target_fields.endswith("|tokens=6144"):
        return version == target_version or version in REVIEWED_3072_FACT_VERSIONS
    base = version.split("|semantic=", 1)[0]
    return base.rsplit("|items=", 1)[0] in _allowed_v4_fields(target_version)


def _pending_source_seal(conn: sqlite3.Connection, sid: str,
                         cursor: int | None, tail: int,
                         fingerprint=None) -> str:
    """Bind every covered turn and coverage artifact selected after a cursor."""
    from hymem.dreaming.lossless import covered_messages_after

    source = covered_messages_after(
        conn, sid, cursor, limit=257,
        roles=frozenset({"user", "assistant"}),
    )
    if len(source) > 256 or not source or source[-1].message_id != tail:
        raise RepairStopped("held fact source exceeds or differs from reviewed tail")
    seal = hashlib.sha256()
    for message in source:
        encoded = json.dumps((
            message.message_id, message.role, message.content,
            message.chunk_id, message.source_created_at,
            message.source_peer_id, message.source_workspace_id,
        ), ensure_ascii=True, separators=(",", ":")).encode()
        seal.update(encoded)
        if fingerprint is not None:
            fingerprint.update(encoded)
    return seal.hexdigest()


def _state(conn: sqlite3.Connection, target_version: str) -> dict:
    """Fingerprint exact fact state without serializing private source text."""
    sessions = [tuple(row) for row in conn.execute(
        "SELECT id,coverage_message_id,facts_message_id,"
        "facts_cursor_message_id,facts_cursor_partial_message_id,"
        "facts_cursor_offset,facts_cursor_prompt_version,facts_retry_count,"
        "facts_retry_config_version,facts_quarantined FROM sessions "
        "WHERE id IN (SELECT session_id FROM fact_extraction_outcomes) "
        "OR facts_retry_count>0 ORDER BY id"
    )]
    outcomes = [tuple(row) for row in conn.execute(
        "SELECT slice_key,session_id,prompt_version,input_hash,"
        "cursor_before_message_id,cursor_before_partial_message_id,"
        "cursor_before_offset,cursor_after_message_id,"
        "cursor_after_partial_message_id,cursor_after_offset,generation,"
        "outcome_status,result_hash,source_manifest_hash,"
        "source_manifest_count,source_manifest_complete "
        "FROM fact_extraction_outcomes ORDER BY session_id,slice_key"
    )]
    packed = json.dumps([sessions, outcomes], ensure_ascii=True,
                        separators=(",", ":"), default=str).encode()
    fingerprint = hashlib.sha256(packed)
    for table in (
        "fact_extraction_revisions", "fact_extraction_source_occurrences",
        "narrative_facts", "narrative_fact_lifecycle",
        "narrative_fact_embeddings",
    ):
        fingerprint.update(table.encode())
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid"):
            # Source and fact text enter only the digest, never the report.
            fingerprint.update(json.dumps(
                tuple(row), ensure_ascii=True, separators=(",", ":"),
                default=str,
            ).encode())
    per_session = Counter(row[1] for row in outcomes)
    stale = sum(row[2] != target_version for row in outcomes)
    # Exact historical v2 rows are allowed. The one reviewed cap-64 cutover
    # also allows old v4 outcomes at 3072 tokens while targeting 6144.
    for row in outcomes:
        version = str(row[2])
        if "|prompt=facts.v4|" in version:
            if not _v4_version_allowed(version, target_version):
                raise RepairStopped("server fact prompt/char/token config differs from target")
    pending = []
    for row in sessions:
        sid, _, _, cursor, partial, offset, _, retries, _, quarantine = row
        tail = fact_mod.facts_tail_message_id(conn, sid)
        if tail is not None and (partial is not None or cursor != tail):
            pending.append((sid, cursor, partial, int(offset or 0),
                            int(retries or 0), int(quarantine or 0), tail))
    # A covered turn can be added or its canonical artifact replaced after
    # preview without changing a fact outcome. Bind the exact eligible tail
    # that fresh extraction will read, including private bytes only in SHA256.
    fingerprint.update(json.dumps(pending, separators=(",", ":")).encode())
    pending_source_seals = {}
    for sid, cursor, _partial, _offset, _retries, _quarantined, tail in pending:
        pending_source_seals[sid] = _pending_source_seal(
            conn, sid, cursor, tail, fingerprint,
        )
    return {
        "state_sha256": fingerprint.hexdigest(),
        "schema": core_db.schema_version(conn),
        "outcomes": len(outcomes), "sessions": len(per_session),
        "stale": stale, "max_outcomes_per_session": max(per_session.values(), default=0),
        "pending_fresh": len(pending),
        "quarantined": sum(int(row[9] or 0) for row in sessions),
        "pending_units": pending, "pending_source_seals": pending_source_seals,
        "session_ids": sorted(per_session),
    }


def _public_state(state: dict, target_version: str, locked: bool) -> dict:
    return {key: value for key, value in state.items()
            if key not in {"pending_units", "pending_source_seals", "session_ids"}} | {
        "target_version": target_version,
        "dream_lock_held": locked,
    }


def _check_authority(conn: sqlite3.Connection, session_ids: list[str]) -> None:
    for sid in session_ids:
        if not fact_mod.fact_session_authority_is_valid(conn, sid):
            raise RepairStopped("existing or resulting fact authority is invalid")


def _backup_under_lease(conn: sqlite3.Connection, path: Path,
                        expected_state: str, target: str) -> None:
    parent = path.parent
    if (not path.is_absolute() or not parent.is_dir()
            or parent.is_symlink() or parent.resolve(strict=True) != parent
            or stat.S_IMODE(parent.stat().st_mode) != 0o700):
        raise RepairStopped("backup parent must already be a private 0700 directory")
    if path.exists() or path.is_symlink():
        raise RepairStopped("backup path must not already exist or be a symlink")
    descriptor = os.open(
        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600,
    )
    os.close(descriptor)
    try:
        backup = sqlite3.connect(str(path), isolation_level=None)
        try:
            conn.backup(backup)
            # The online image necessarily contains this operator's lease.
            # A quiesced restore must never revive that transient token.
            backup.execute("DELETE FROM run_lock WHERE name='dreaming'")
        finally:
            backup.close()
        with core_db.read_snapshot(path) as snapshot:
            if _state(snapshot, target)["state_sha256"] != expected_state:
                raise RepairStopped("online backup disagrees with preflight fact state")
            if snapshot.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RepairStopped("online backup failed SQLite quick_check")
            if snapshot.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RepairStopped("online backup failed foreign-key check")
    except BaseException:
        # A partial file is never an acceptable rollback point.
        path.unlink(missing_ok=True)
        raise


def probe(conn: sqlite3.Connection, db_path: Path, llm) -> list[dict]:
    """Two held units, cap 16 then cap 32 only for capacity rejection."""
    held = conn.execute(
        "SELECT id,facts_cursor_message_id,facts_cursor_partial_message_id,"
        "facts_cursor_offset,facts_retry_count,facts_quarantined "
        "FROM sessions WHERE facts_retry_count>0 ORDER BY facts_quarantined DESC"
    ).fetchall()
    fixed = [row for row in held if int(row[5] or 0) == 1]
    fresh = [row for row in held if int(row[5] or 0) == 0]
    if len(fixed) != 1 or len(fresh) != 1:
        raise RepairStopped("expected exactly one fixed and one fresh held unit")
    fixed_key = conn.execute(
        "SELECT slice_key FROM fact_extraction_outcomes WHERE session_id=?",
        (fixed[0][0],),
    ).fetchall()
    if len(fixed_key) != 1:
        raise RepairStopped("fixed held session has an unexpected replay shape")
    results: list[dict] = []
    start_http = int(getattr(llm, "request_attempts", 0))
    pending = {"fixed", "fresh"}
    for cap in (16, 32):
        cfg = _configuration(db_path, cap)
        next_pending: set[str] = set()
        for mode in ("fixed", "fresh"):
            if mode not in pending:
                continue
            before = int(getattr(llm, "request_attempts", 0))
            if mode == "fixed":
                result = fact_mod.reextract_fact_outcome(
                    conn, fixed_key[0][0], llm, cfg,
                )
            else:
                row = fresh[0]
                result = fact_mod._extract_facts_fresh(
                    conn, row[0], llm, cfg,
                    since_message_id=row[1], partial_message_id=row[2],
                    start_offset=int(row[3] or 0),
                    max_chars=cfg.dream_digest_max_chars,
                    allow_capacity_retry=False,
                )
            if result is None:
                raise RepairStopped("held fresh unit has no selectable source")
            attempts = int(getattr(llm, "request_attempts", 0)) - before
            if (len(results) >= MAX_PROBE_LOGICAL or attempts > 3
                    or int(getattr(llm, "request_attempts", 0)) - start_http
                    > MAX_PROBE_HTTP):
                raise RepairStopped("provider exceeded three HTTP attempts per probe")
            results.append({"mode": mode, "cap": cap,
                            "accepted": not result.parse_failed,
                            "reason": result.failure_reason if result.parse_failed else None,
                            "items": len(result.items) if not result.parse_failed else None,
                            "http_attempts": attempts})
            if result.parse_failed:
                if result.failure_reason != "output_capacity_exceeded":
                    raise RepairStopped("probe failed for a non-capacity reason")
                next_pending.add(mode)
        pending = next_pending
        if not pending:
            break
    if pending:
        raise RepairStopped("cap 32 did not validate every complete held unit")
    return results


def _fact_vector_gaps(conn: sqlite3.Connection, embedder) -> int:
    from hymem.dreaming.embeddings import _embedding_identity, embedding_text_hash
    model, dim = _embedding_identity(embedder)
    missing = 0
    for row in conn.execute(
        "SELECT f.text,e.model,e.dim,e.text_hash FROM narrative_facts f "
        "LEFT JOIN narrative_fact_embeddings e ON e.fact_id=f.id "
        "WHERE f.source_outcome_key IS NOT NULL AND f.lifecycle_status='active' "
        "AND f.invalid_at IS NULL"
    ):
        if row[1] != model or row[2] != dim or row[3] != embedding_text_hash(row[0]):
            missing += 1
    return missing


def _drain_fact_vectors(conn: sqlite3.Connection, embedder, *,
                        max_batches: int, max_texts: int,
                        target_version: str) -> dict:
    if (not core_db._load_vec_extension(conn) or not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='vec_facts'"
    ).fetchone()):
        raise RepairStopped("physical fact vector mirror is unavailable")
    batches = texts = 0
    while _fact_vector_gaps(conn, embedder):
        if batches >= max_batches or texts + 64 > max_texts:
            raise RepairStopped("fact embedding batch or text ceiling reached")
        pending = fetch_fact_embeddings(conn, embedder)
        if pending is None:
            raise RepairStopped("fact embedding scan made no progress")
        with core_db.transaction(conn):
            persist_fact_embeddings(conn, pending)
        batches += 1
        texts += len(pending.fact_ids)
    from hymem.dreaming.embeddings import (
        _embedding_identity, decode_vector, embedding_text_hash,
    )
    model, dim = _embedding_identity(embedder)
    rows = conn.execute(
        "SELECT f.text,o.prompt_version,e.vector_json,e.model,e.dim,"
        "e.text_hash,v.embedding FROM narrative_facts f "
        "JOIN fact_extraction_outcomes o ON o.slice_key=f.source_outcome_key "
        "LEFT JOIN narrative_fact_embeddings e ON e.fact_id=f.id "
        "LEFT JOIN vec_facts v ON v.rowid=f.id "
        "WHERE f.source_outcome_key IS NOT NULL "
        "AND f.lifecycle_status='active' AND f.invalid_at IS NULL"
    )
    for row in rows:
        if (row[1] != target_version or row[3] != model or row[4] != dim
                or row[5] != embedding_text_hash(row[0])
                or row[6] is None):
            raise RepairStopped("active fact vector authority or mirror is missing")
        try:
            matching = bytes(row[6]) == core_db._pack_vector(
                decode_vector(row[2])
            )
        except (TypeError, ValueError, OverflowError):
            matching = False
        if not matching:
            raise RepairStopped("physical active fact vector mirror mismatch")
    return {"embedding_batches": batches, "embedding_texts": texts}


def apply(conn: sqlite3.Connection, db_path: Path, cfg: HyMemConfig, llm,
          embedder, *, expected_state: str, expected_target: str,
          expected_outcomes: int, expected_sessions: int,
          expected_stale: int, backup_path: Path,
          max_logical: int, max_http: int, max_embed_batches: int,
          max_embed_texts: int, include_fresh: bool) -> dict:
    """One bounded lease-fenced replay; no calls to other dreaming tiers."""
    target = fact_mod.facts_config_version(cfg, client=llm)
    if target != expected_target:
        raise RepairStopped("fact producer identity differs from preview")
    before = _state(conn, target)
    if (before["state_sha256"] != expected_state
            or before["schema"] != 64
            or before["outcomes"] != expected_outcomes
            or before["sessions"] != expected_sessions
            or before["stale"] != expected_stale
            or expected_stale < 0 or expected_stale > MAX_REPLAY_LOGICAL
            or (include_fresh and before["pending_fresh"] != 1)):
        raise RepairStopped("fact state differs from approved preview")
    if (max_logical < 0 or max_logical > MAX_APPLY_LOGICAL
            or max_http < 0 or max_http > MAX_APPLY_HTTP
            or max_embed_batches < 0 or max_embed_batches > 70
            or max_embed_texts < 0 or max_embed_texts > 4200):
        raise RepairStopped("operator ceiling exceeds the reviewed hard limit")
    if include_fresh:
        held = before["pending_units"][0]
        if (before["pending_fresh"] != 1 or held[5] != 0
                or len(before["pending_units"]) != 1):
            raise RepairStopped("held fresh unit selection is ambiguous")
        # A completed 12,000-char unit may span more than one message, but
        # never more units than remaining covered turns when no turn is split.
        from hymem.dreaming.lossless import covered_messages_after
        remaining_turns = covered_messages_after(
            conn, held[0], held[1], limit=256,
            roles=frozenset({"user", "assistant"}),
        )
        if not 1 <= len(remaining_turns) <= MAX_FRESH_LOGICAL:
            raise RepairStopped("held fresh tail exceeds the reviewed unit ceiling")
        if expected_stale == expected_outcomes and held[4] != 2:
            raise RepairStopped("held fresh retry count differs from the live baseline")
    required = before["stale"] + (len(remaining_turns) if include_fresh else 0)
    if max_logical < required or max_http < required * 3:
        raise RepairStopped("provider ceilings are below the planned worst case")
    if backup_path.resolve() == db_path.resolve():
        raise RepairStopped("backup path must differ from the store")
    if backup_path.exists():
        raise RepairStopped("backup path already exists")
    if conn.execute("SELECT 1 FROM run_lock WHERE name='dreaming'").fetchone():
        raise RepairStopped("dream lock is already held")
    from hymem.dreaming.aggregation_material import embedding_storage_identity
    embed_model, embed_dim = embedding_storage_identity(embedder)
    stored = dict(conn.execute(
        "SELECT key,value FROM schema_meta WHERE key IN ('vec_model','vec_dim')"
    ).fetchall())
    if stored.get("vec_model") != embed_model or stored.get("vec_dim") != str(embed_dim):
        raise RepairStopped("configured embedding identity differs from stored vectors")
    if (not core_db._load_vec_extension(conn) or not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='vec_facts'"
    ).fetchone()):
        raise RepairStopped("physical fact vector mirror is unavailable")
    _check_authority(conn, before["session_ids"])
    holder = runner._new_lease_token()
    if not runner._acquire_lock(conn, holder):
        raise RepairStopped("dream lock acquisition lost")
    heartbeat = None
    fence = None
    logical = 0
    start_http = int(getattr(llm, "request_attempts", 0))
    try:
        heartbeat = runner._DreamLeaseHeartbeat(
            conn, holder, interval_seconds=runner._LOCK_REFRESH_INTERVAL_SECONDS,
        )
        fence = core_db.activate_transaction_lease_fence(
            conn, name="dreaming", holder=holder,
        )
        heartbeat.start()
        if _state(conn, target)["state_sha256"] != expected_state:
            raise RepairStopped("fact state changed before backup")
        _backup_under_lease(conn, backup_path, expected_state, target)
        for sid in before["session_ids"]:
            # Stage one complete session before publishing any of its revised
            # outcomes. A failed provider call therefore leaves that session's
            # prior authoritative generation intact. Completed earlier
            # sessions remain independently valid and can be resumed safely.
            keys = [row[0] for row in conn.execute(
                "SELECT slice_key FROM fact_extraction_outcomes "
                "WHERE session_id=? AND prompt_version<>? ORDER BY "
                + fact_mod._fact_replay_order_sql(), (sid, target),
            )]
            if keys and fact_mod.next_fact_outcome_for_replay(
                conn, sid, target,
            ) != keys[0]:
                raise RepairStopped("stale fact replay order changed")
            staged = []
            for slice_key in keys:
                heartbeat.check()
                if fact_mod.facts_config_version(cfg, client=llm) != target:
                    raise RepairStopped("fact producer changed during replay")
                if logical + 1 > max_logical or (
                    int(getattr(llm, "request_attempts", 0)) - start_http + 3 > max_http
                ):
                    raise RepairStopped("provider call ceiling reached")
                logical += 1
                extraction = fact_mod.reextract_fact_outcome(
                    conn, slice_key, llm, cfg, _require_committed_chain=False,
                )
                heartbeat.check()
                if extraction.parse_failed:
                    raise RepairStopped("fact replay output rejected: "
                                        + str(extraction.failure_reason))
                if extraction.publication_version != target:
                    raise RepairStopped("fact replay producer changed")
                staged.append(extraction)
                if int(getattr(llm, "request_attempts", 0)) - start_http > max_http:
                    raise RepairStopped("HTTP attempt ceiling exceeded")
            if staged:
                with core_db.transaction(conn):
                    if fact_mod.facts_config_version(cfg, client=llm) != target:
                        raise RepairStopped("fact producer changed before session commit")
                    for extraction in staged:
                        if conn.execute(
                            "SELECT generation FROM fact_extraction_outcomes "
                            "WHERE slice_key=?", (extraction.slice_key,),
                        ).fetchone()[0] != extraction.expected_generation:
                            raise RepairStopped("fact outcome changed during staging")
                        fact_mod.persist_facts(
                            conn, sid, extraction,
                            max_items=cfg.dream_max_facts_per_session,
                            _defer_chain_audit=True,
                        )
                    remaining = conn.execute(
                        "SELECT 1 FROM fact_extraction_outcomes "
                        "WHERE session_id=? AND prompt_version<>? LIMIT 1",
                        (sid, target),
                    ).fetchone() is not None
                    if not remaining and not fact_mod.fact_session_authority_is_valid(conn, sid):
                        raise RepairStopped("final fact session authority audit failed")
                    conn.execute(
                        "UPDATE sessions SET facts_cursor_prompt_version="
                        "CASE WHEN ?=0 THEN ? ELSE facts_cursor_prompt_version END,"
                        "facts_retry_count=0,facts_retry_config_version=NULL,"
                        "facts_quarantined=0 WHERE id=?",
                        (int(remaining), target, sid),
                    )
        if include_fresh:
            held = before["pending_units"][0]
            sid, cursor, partial, offset, _, _, tail = held
            for unit_index in range(1, MAX_FRESH_LOGICAL + 1):
                now = conn.execute(
                    "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
                    "facts_cursor_offset FROM sessions WHERE id=?", (sid,),
                ).fetchone()
                if (now[0], now[1], int(now[2] or 0)) != (cursor, partial, offset):
                    raise RepairStopped("held fresh cursor changed during replay")
                if fact_mod.facts_tail_message_id(conn, sid) != tail:
                    raise RepairStopped("held fresh source tail changed during replay")
                if logical + 1 > max_logical or (
                    int(getattr(llm, "request_attempts", 0)) - start_http + 3 > max_http
                ):
                    raise RepairStopped("fresh fact call ceiling reached")
                logical += 1
                # Keep the provider's entire source selection on a single
                # immutable WAL snapshot. Concurrent covered appends cannot
                # enter this prompt after the approved-tail check.
                with core_db.read_snapshot(db_path) as source_conn:
                    if (fact_mod.facts_tail_message_id(source_conn, sid) != tail
                            or _pending_source_seal(
                                source_conn, sid, held[1], tail,
                            ) != before["pending_source_seals"][sid]):
                        raise RepairStopped("held fresh source changed before extraction")
                    extraction = fact_mod._extract_facts_fresh(
                        source_conn, sid, llm, cfg, since_message_id=cursor,
                        partial_message_id=partial, start_offset=offset,
                        max_chars=cfg.dream_digest_max_chars,
                        allow_capacity_retry=False,
                    )
                heartbeat.check()
                if extraction is None or extraction.parse_failed:
                    reason = extraction.failure_reason if extraction is not None else "no_source"
                    raise RepairStopped(
                        f"fresh fact unit {unit_index} rejected: {reason}"
                    )
                if (extraction.partial_message_id is not None
                        or extraction.publication_version != target
                        or extraction.covered_message_id == cursor
                        or extraction.covered_message_id is None
                        or extraction.covered_message_id > tail):
                    raise RepairStopped("fresh fact result did not cover its full unit")
                with core_db.transaction(conn):
                    current = conn.execute(
                        "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
                        "facts_cursor_offset FROM sessions WHERE id=?", (sid,),
                    ).fetchone()
                    if ((current[0], current[1], int(current[2] or 0))
                            != (cursor, partial, offset)
                            or fact_mod.facts_tail_message_id(conn, sid) != tail
                            or _pending_source_seal(
                                conn, sid, held[1], tail,
                            ) != before["pending_source_seals"][sid]):
                        raise RepairStopped("held fresh source changed before publication")
                    fact_mod.persist_facts(
                        conn, sid, extraction,
                        max_items=cfg.dream_max_facts_per_session,
                    )
                    if not fact_mod.fact_session_authority_is_valid(conn, sid):
                        raise RepairStopped("fresh fact unit authority audit failed")
                cursor = extraction.covered_message_id
                partial = extraction.partial_message_id
                offset = extraction.next_message_offset
                if extraction.caught_up:
                    break
            else:
                raise RepairStopped("fresh fact tail exceeded eight complete units")
        if int(getattr(llm, "request_attempts", 0)) - start_http > max_http:
            raise RepairStopped("HTTP attempt ceiling exceeded")
        _check_authority(conn, before["session_ids"])
        after = _state(conn, target)
        if after["stale"] or after["quarantined"] or (
            include_fresh and after["pending_fresh"]
        ):
            raise RepairStopped("fact replay postflight is incomplete")
        vectors = _drain_fact_vectors(
            conn, embedder, max_batches=max_embed_batches,
            max_texts=max_embed_texts, target_version=target,
        )
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RepairStopped("SQLite quick_check failed after replay")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RepairStopped("foreign-key check failed after replay")
        heartbeat.check()
        return {"replayed": before["stale"], "fresh_complete": include_fresh,
                "logical_slots_reserved": logical,
                "http_attempts": int(getattr(llm, "request_attempts", 0)) - start_http,
                "stale_remaining": after["stale"],
                "quarantined_remaining": after["quarantined"],
                "pending_fresh_remaining": after["pending_fresh"],
                "backup": str(backup_path), **vectors}
    finally:
        try:
            if heartbeat is not None:
                heartbeat.stop()
        finally:
            try:
                if fence is not None:
                    core_db.deactivate_transaction_lease_fence(fence)
            finally:
                if not runner._release_lock(conn, holder):
                    raise RepairStopped("dream lock release failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cap", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument("--fact-tokens", type=int, choices=(3072, 6144), default=3072)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--probe", action="store_true")
    modes.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--server-process-env", action="store_true")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--expect-state-sha256")
    parser.add_argument("--expect-target-version")
    parser.add_argument("--expect-outcomes", type=int)
    parser.add_argument("--expect-sessions", type=int)
    parser.add_argument("--expect-stale", type=int)
    parser.add_argument("--max-logical", type=int)
    parser.add_argument("--max-http", type=int)
    parser.add_argument("--max-embed-batches", type=int, default=70)
    parser.add_argument("--max-embed-texts", type=int, default=4200)
    parser.add_argument("--include-fresh", action="store_true")
    args = parser.parse_args()
    if (args.probe or args.apply) and not args.allow_provider_calls:
        parser.error("provider calls require --allow-provider-calls and operator approval")
    db_path = args.db.resolve(strict=True)
    if not db_path.is_file():
        parser.error("--db must name an existing regular SQLite store")
    live = _is_live_store(db_path)
    if live and db_path != LIVE_CONTAINER_DB:
        parser.error("live operator work must run inside the Hermes1 container")
    if live and not args.server_process_env:
        parser.error("live operator work requires --server-process-env")
    # Provider exception strings may contain reply text. The tool emits only
    # bounded categories and counters, never provider exceptions or raw memory.
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    llm = None
    try:
        if args.server_process_env:
            _load_server_process_env()
        cfg = _configuration(db_path, args.cap, args.fact_tokens)
        llm = _client_from_env()
        target = fact_mod.facts_config_version(cfg, client=llm)
        if not args.apply:
            with core_db.read_snapshot(db_path) as conn:
                state = _state(conn, target)
                locked = conn.execute(
                    "SELECT 1 FROM run_lock WHERE name='dreaming'"
                ).fetchone() is not None
                if args.probe:
                    output = {"probes": probe(conn, db_path, llm)}
                else:
                    output = _public_state(state, target, locked)
        else:
            selected_cap = bootstrap._validated_facts_cap_override(
                bootstrap.resolve_env().dream_max_facts_per_session_raw
            )
            selected_tokens = bootstrap._validated_facts_tokens_override(
                bootstrap.resolve_env().facts_extraction_max_tokens_raw
            ) or 3072
            if live and selected_cap != args.cap:
                raise RepairStopped("running Hermes fact cap differs from replay cap")
            if live and selected_tokens != args.fact_tokens:
                raise RepairStopped("running Hermes fact tokens differ from replay budget")
            required = (args.backup, args.expect_state_sha256,
                        args.expect_target_version, args.expect_outcomes,
                        args.expect_sessions, args.expect_stale,
                        args.max_logical, args.max_http)
            if any(value is None for value in required):
                parser.error("apply requires backup, exact state/target/count guards and call ceilings")
            embedder = _embedder_from_env()
            try:
                conn = core_db.connect(db_path)
                try:
                    if not core_db._load_vec_extension(conn):
                        raise RepairStopped("physical vector extension is unavailable")
                    output = apply(
                        conn, db_path, cfg, llm, embedder,
                        expected_state=args.expect_state_sha256,
                        expected_target=args.expect_target_version,
                        expected_outcomes=args.expect_outcomes,
                        expected_sessions=args.expect_sessions,
                        expected_stale=args.expect_stale,
                        backup_path=args.backup,
                        max_logical=args.max_logical, max_http=args.max_http,
                        max_embed_batches=args.max_embed_batches,
                        max_embed_texts=args.max_embed_texts,
                        include_fresh=args.include_fresh,
                    )
                finally:
                    conn.close()
            finally:
                embedder.close()
        print(json.dumps(output, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"stopped": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, RepairStopped)
                          else "unexpected failure; inspect bounded operator logs"}))
        return 2
    finally:
        logging.disable(previous_logging_disable)
        if llm is not None:
            # A transport close error must not bypass the bounded output path.
            # SDK exception text can contain request or response fragments.
            try:
                llm.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
