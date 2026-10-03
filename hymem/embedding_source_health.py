"""Bounded, read-only explanation of incompatible *stored* vector mirrors.

Raw compatibility inventory remains unchanged. Only well-formed incompatible
rows receive source classification: complete current proof, explicit historical
state, or unsafe unknown. A failed proof never implies retirement. Compatible
rows skip proof work; consequently this is not a missing-row or corpus coverage
audit, and its neutral producer scope is not a configured retrieval guarantee.

The path API owns a read-only connection, snapshot and SQLite progress handler.
It never initializes a database, repairs sources or calls an embedding/LLM
provider. Memory/work caps also apply to legacy proof helpers that fetch lists.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import time

from hymem.embedding_health import (
    EmbeddingHealth, ForeignKeyHealth, MIRROR_TABLES, _exact_key,
    _positive_dimension, _TYPED_PRODUCER_TABLES, _unavailable, _valid_vector,
    scan_embedding_health,
)

_KEYS = ("chunk_id", "message_id", "edge_text", "episode_id", "fact_id", "node_id")
_CATEGORIES = ("source_eligible", "retained_unverified", "retired", "source_withdrawn", "rebuild_required", "unsafe_unknown")


@dataclass(frozen=True)
class SourceMirrorHealth:
    table: str
    total: int | None
    source_eligible: int | None
    retained_unverified: int | None
    retired: int | None
    rebuild_required: int | None
    unsafe_unknown: int | None
    error_code: str | None = None
    # Subset of retained_unverified, not another partition or proof of age.
    terminal_source_loss: int | None = 0
    # A complete successful-empty extraction withdrew source authority. This
    # is not a negative assertion about the world or permission to erase audit.
    source_withdrawn: int | None = 0


@dataclass(frozen=True)
class EmbeddingRecoveryHealth:
    inventory: EmbeddingHealth
    tables: tuple[SourceMirrorHealth, ...]

    @property
    def status(self) -> str:
        if self.inventory.status == "unavailable" or any(item.error_code for item in self.tables):
            return "unavailable"
        if (self.inventory.foreign_keys.status != "valid"
                or any(item.malformed for item in self.inventory.tables)
                or any(item.source_eligible or item.rebuild_required or item.unsafe_unknown for item in self.tables)):
            return "action_required"
        if not self.inventory.live_identity_verified:
            return "unverified"
        if any(item.retained_unverified for item in self.tables):
            return "historical"
        return "compatible"


class _AuditLimit(BaseException):
    """Cannot be swallowed by a proof helper's best-effort failure handler."""

    def __init__(self, code: str):
        self.code = code


class _Budget:
    def __init__(self, seconds, sql_steps):
        self.expires = time.monotonic() + seconds
        self.limit = sql_steps
        self.interval = min(1000, sql_steps)
        self.steps = 0
        self.exhausted = False

    def check(self):
        if self.exhausted or time.monotonic() >= self.expires:
            self.exhausted = True
            raise _AuditLimit("diagnostic_budget_exhausted")

    def progress(self):
        self.steps += self.interval
        self.exhausted |= self.steps >= self.limit or time.monotonic() >= self.expires
        return int(self.exhausted)


class _ProofReader:
    """Cap fetched source rows/bytes without changing proof semantics."""

    def __init__(self, conn, budget):
        self.conn, self.budget = conn, budget
        self.rows, self.bytes = 0, 0

    def execute(self, *args):
        self.budget.check()
        return _ProofCursor(self, self.conn.execute(*args))

    def account(self, row):
        self.budget.check()
        if row is not None:
            self.rows += 1
            self.bytes += sum(len(value) for value in row if isinstance(value, (str, bytes)))
            if self.rows > 8192 or self.bytes > 8 * 1024 * 1024:
                raise _AuditLimit("source_proof_budget_exhausted")
        return row


class _ProofCursor:
    def __init__(self, reader, cursor):
        self.reader, self.cursor = reader, cursor

    def fetchone(self):
        return self.reader.account(self.cursor.fetchone())

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    def fetchall(self):
        return list(self)

    def close(self):
        self.cursor.close()


def _unknown(table, total, code):
    return SourceMirrorHealth(table, total, None, None, None, None, None, code, None, None)


def _failed_inventory(verified, code):
    return EmbeddingHealth(
        verified, tuple(_unavailable(table, code) for table in MIRROR_TABLES),
        ForeignKeyHealth("unavailable", None, error_code=code),
    )


def _incomplete_manifest(row, *, legacy_version=None):
    return (row["source_manifest_complete"] == 0
            and row["source_manifest_count"] == 0
            and row["source_manifest_hash"] is None
            and row["source_manifest_version"] in (None, legacy_version))


def _source_withdrawn(conn, owner, now):
    """Prove the maintained successful-empty terminal state, without repair.

    Missing observations or a retracted cache cannot establish withdrawal.
    Every final source revision needs its still-retained empty retirement
    receipt. A later same-binding winner may renew current withdrawal, but
    cannot replace the original clock witness. Missing/overwritten receipts,
    different bindings and no_current_authority remain unknown. This
    deliberately recognizes only the all-canonical direct-extraction shape.
    All queries and maintained proof helpers use the caller's bounded reader.
    """
    from hymem import reembed
    from hymem.core.time import normalize_iso_timestamp, validate_event_clock
    from hymem.dreaming import evidence
    from hymem.dreaming.bitemporal import _ordered_events
    from hymem.dreaming.lossless import validate_message_coverage_artifact
    from hymem.dreaming.message_coverage import LOSSLESS_COVERAGE_VERSION
    from hymem.extraction.producer import phase1_generation_registry_row_is_valid

    def clock(value):
        return normalize_iso_timestamp(value, context="withdrawn extraction audit")

    edge_id = owner["id"]
    if (owner["derived"] != 0 or owner["status"] != "retracted"
            or (owner["pos_evidence"], owner["neg_evidence"]) != (0, 0)
            or evidence.ledger_counts(conn, edge_id) != (0, 0)
            or _ordered_events(conn, edge_id)
            or conn.execute("SELECT 1 FROM kg_evidence_signals WHERE edge_id=? "
                            "AND counts_toward_confidence=1 LIMIT 1", (edge_id,)).fetchone()
            or conn.execute("SELECT 1 FROM kg_claim_observations WHERE edge_id=? LIMIT 1", (edge_id,)).fetchone()):
        return False
    revisions = conn.execute("SELECT * FROM kg_evidence WHERE edge_id=?", (edge_id,)).fetchall()
    if not revisions:
        return False
    by_id, final_sources, positives, all_events = {}, {}, [], []
    seen_revisions = set()
    for item in revisions:
        if (item["provenance_status"] != "canonical" or item["evidence_kind"] != "extraction"
                or item["source_coverage_version"] != LOSSLESS_COVERAGE_VERSION
                or item["is_current"] != 0 or type(item["revision"]) is not int or item["revision"] < 1
                or item["polarity"] not in (-1, 1) or not item["superseded_reason"]
                or type(item["evidence_weight"]) is not int or item["evidence_weight"] < 1
                or not isinstance(item["weight_source"], str) or not item["weight_source"]
                or (item["value_numeric"] is not None and (
                    type(item["value_numeric"]) not in (int, float) or not math.isfinite(item["value_numeric"])))
                or item["interpretation_key"] != evidence._interpretation_key(**{name: item[name] for name in (
                    "polarity", "evidence_weight", "weight_source", "source_role", "surface_subject", "surface_object",
                    "value_text", "value_numeric", "value_unit", "temporal_scope")})):
            return False
        extracted, published, superseded = map(clock, (item["extracted_at"], item["published_at"], item["superseded_at"]))
        event_at = clock(item["source_event_at"])
        try:
            source_created = clock(item["source_created_at"])
        except ValueError:
            # The maintained canonical writer explicitly uses year one for
            # retained NULL/malformed legacy source time, never a new clock.
            source_created = "0001-01-01T00:00:00.000Z"
        if item["source_event_at"] != source_created:
            return False
        if not extracted <= published <= superseded <= now or event_at > now:
            return False
        validate_event_clock(conn, item["source_event_at"], item["extracted_at"])
        proof = validate_message_coverage_artifact(
            conn, message_id=item["source_message_id"], chunk_id=item["source_coverage_chunk_id"],
            coverage_version=item["source_coverage_version"],
        )
        source_identity = (item["source_session_id"], item["source_message_id"], item["evidence_kind"])
        if ((proof.session_id, proof.message_id, proof.role, proof.source_peer_id,
             proof.source_workspace_id, proof.source_created_at)
                != tuple(item[name] for name in ("source_session_id", "source_message_id", "source_role",
                                                 "source_peer_id", "source_workspace_id", "source_created_at"))
                or conn.execute("SELECT 1 FROM coverage_integrity_failures WHERE session_id=? LIMIT 1",
                                (proof.session_id,)).fetchone()):
            return False
        revision_identity = (*source_identity, item["revision"])
        if revision_identity in seen_revisions:
            return False
        seen_revisions.add(revision_identity)
        if source_identity not in final_sources or item["revision"] > final_sources[source_identity]["revision"]:
            final_sources[source_identity] = item
        by_id[item["id"]] = item
        all_events.append(event_at)
        if item["polarity"] == 1:
            positives.append(event_at)

    if (not positives or owner["valid_at"] != max(positives)
            or owner["invalid_at"] != max(max(positives), max(all_events))):
        return False
    receipt_sources = """
        SELECT outcome.*, artifact.created_at AS chunk_created_at,
               generation.generation_key, generation.extraction_cache_key,
               generation.producer_identity_sha256, generation.identity_exact,
               generation.reuse_scope, generation.binding_json
        FROM kg_claim_extraction_outcomes outcome
        LEFT JOIN chunks artifact ON artifact.id=outcome.chunk_id
        JOIN chunk_message_sources member ON member.chunk_id=outcome.chunk_id
        JOIN phase1_generations generation
          ON generation.generation_key=outcome.phase1_generation_key
         AND generation.extraction_cache_key=outcome.prompt_version
        WHERE member.source_session_id=? AND member.source_message_id=?
          AND outcome.phase1_generation_key IS NOT NULL
          AND hymem_phase1_generation_is_authorized(generation.generation_key,generation.identity_exact)=1
          AND hymem_normalize_iso_timestamp(outcome.succeeded_at) IS NOT NULL
    """

    def complete_empty(receipt, session_id, message_id):
        succeeded = clock(receipt["succeeded_at"])
        if (succeeded > now or clock(receipt["chunk_created_at"]) > succeeded
                or receipt["prompt_generation"] != evidence.prompt_generation(receipt["prompt_version"])
                or not phase1_generation_registry_row_is_valid(*(receipt[name] for name in (
                    "generation_key", "extraction_cache_key", "producer_identity_sha256",
                    "identity_exact", "reuse_scope", "binding_json")))):
            return False
        manifest = reembed._source(conn, 0, {"chunk_id": receipt["chunk_id"]})
        return bool(manifest is not None and any(
            source.session_id == session_id and source.message_id == message_id
            for source in manifest[1][1])
            and conn.execute("SELECT 1 FROM kg_claim_observations WHERE chunk_id=? LIMIT 1",
                             (receipt["chunk_id"],)).fetchone() is None
            and receipt["result_hash"] == evidence.claim_result_hash([])
            and receipt["result_hash"] == evidence.claim_observation_result_hash(conn, receipt["chunk_id"]))

    for (session_id, message_id, _), item in final_sources.items():
        authority = evidence.claim_retirement_authority(
            conn, source_session_id=session_id, source_message_id=message_id,
        )
        # Mirror the maintained authority's exact winner ordering, then validate
        # the entire receipt. Runtime authorization alone does not validate JSON.
        winner = conn.execute(receipt_sources + """
            ORDER BY outcome.prompt_generation DESC, outcome.prompt_version DESC,
                     hymem_normalize_iso_timestamp(outcome.succeeded_at) DESC,
                     outcome.succeeded_at DESC, outcome.chunk_id DESC LIMIT 1
        """, (session_id, message_id)).fetchone()
        if winner is None:
            return False
        succeeded = clock(winner["succeeded_at"])
        reason = "successful_reextract:" + winner["prompt_version"]
        retired, published = clock(item["superseded_at"]), clock(item["published_at"])
        if (authority != (succeeded, reason) or item["superseded_reason"] != reason
                or reason == "successful_reextract:no_current_authority"
                or retired > max(succeeded, published)
                or not complete_empty(winner, session_id, message_id)):
            return False
        if retired != max(succeeded, published):
            # A later overlapping chunk can renew the same empty authority
            # without rewriting already-retired evidence. The reason string
            # contains only a prompt key, NOT a producer generation: <= alone
            # would also excuse an overwritten different-producer receipt.
            # Require a complete retained same-binding witness at the original
            # publication-floor retirement coordinate. Candidate work stays
            # under this edge's existing row/byte/time/SQL proof budgets.
            witnesses = conn.execute(receipt_sources + """
                AND outcome.phase1_generation_key=? AND outcome.prompt_version=?
                AND MAX(hymem_normalize_iso_timestamp(outcome.succeeded_at),?)=?
                AND hymem_normalize_iso_timestamp(outcome.succeeded_at)<=?
                ORDER BY outcome.chunk_id
            """, (session_id, message_id, winner["phase1_generation_key"], winner["prompt_version"],
                  published, retired, succeeded))
            if not any(complete_empty(receipt, session_id, message_id) for receipt in witnesses):
                return False

    # Empty eligible lifecycle is expected, but cannot hide corrupt retained
    # assertions. Require each positive revision's original assertion as well.
    asserted = set()
    for event in conn.execute("SELECT * FROM kg_edge_lifecycle WHERE edge_id=?", (edge_id,)):
        source = by_id.get(event["source_evidence_id"])
        if (source is None or source["polarity"] != 1 or event["event_kind"] != "claim_assertion"
                or event["direction"] != 1 or event["dependency_count"] != 0
                or conn.execute("SELECT 1 FROM kg_lifecycle_dependencies WHERE lifecycle_id=? LIMIT 1",
                                (event["id"],)).fetchone()
                or event["event_key"] != evidence.claim_assertion_event_key(
                    source["source_session_id"], source["source_message_id"], source["evidence_kind"], source["revision"])
                or event["event_at"] != clock(source["source_event_at"])
                or not clock(source["extracted_at"]) <= clock(event["created_at"]) <= clock(source["published_at"])):
            return False
        validate_event_clock(conn, event["event_at"], event["created_at"])
        asserted.add(source["id"])
    return asserted == {item["id"] for item in revisions if item["polarity"] == 1}


def _classify(conn, index, row):
    # Import lazily: reembed depends on the raw inventory, not this scanner.
    from hymem import reembed
    from hymem.core.time import normalize_iso_timestamp

    key = row[_KEYS[index]]
    if index == 0:
        source = conn.execute(
            "SELECT c.* FROM chunks c JOIN sessions s ON s.id=c.session_id WHERE c.id=?",
            (key,),
        ).fetchone()
        if source is None or source["chunk_kind"] != "extraction":
            return "unsafe_unknown"
        if conn.execute(
            "SELECT 1 FROM coverage_integrity_failures WHERE session_id=? LIMIT 1", (source["session_id"],),
        ).fetchone() is not None:
            return "unsafe_unknown"
        if source["source_manifest_version"] is None and source["source_manifest_count"] is None:
            # A partial manifest is corruption, not a legacy marker. This also
            # covers honestly unmanifested short_session_fallback artifacts;
            # builder naming alone never excuses a broken complete manifest.
            partial = conn.execute(
                "SELECT 1 FROM chunk_message_sources WHERE chunk_id=? "
                "UNION ALL SELECT 1 FROM kg_evidence WHERE chunk_id=? AND provenance_status='canonical' LIMIT 1",
                (key, key),
            ).fetchone()
            return "retained_unverified" if partial is None else "unsafe_unknown"
    elif index == 2:
        from hymem.dreaming.bitemporal import _ordered_events
        owners = conn.execute(
            "SELECT id,status,valid_at,invalid_at,derived,pos_evidence,neg_evidence FROM knowledge_graph WHERE "
            "subject_canonical || ' ' || predicate || ' ' || object_canonical=? LIMIT 65",
            (key,),
        ).fetchall()
        if not owners or len(owners) > 64:
            return "unsafe_unknown"
        # Every owner must be explicitly closed. A merely stale/derived edge,
        # insufficient evidence, or absent current proof is NOT retirement.
        now = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')").fetchone()[0]
        def closed(owner):
            if owner["status"] != "retracted" or owner["invalid_at"] is None or owner["valid_at"] is None:
                return False
            valid_at = normalize_iso_timestamp(owner["valid_at"], context="embedding source validity")
            invalid_at = normalize_iso_timestamp(owner["invalid_at"], context="embedding source retirement")
            if not valid_at <= invalid_at <= now:
                return False
            events = _ordered_events(conn, owner["id"])
            return bool(events and events[-1][2] == -1
                        and any(at == invalid_at and direction == -1 for at, _, direction in events))
        closures = [closed(owner) for owner in owners]
        if all(closures):
            return "retired"
        if all(is_closed or _source_withdrawn(conn, owner, now) for owner, is_closed in zip(owners, closures)):
            return "source_withdrawn"
    elif index == 3:
        source = conn.execute(
            "SELECT e.* FROM episodes e JOIN sessions s ON s.id=e.session_id WHERE e.id=?", (key,),
        ).fetchone()
        if source is None:
            return "unsafe_unknown"
        if _incomplete_manifest(source):
            partial = conn.execute("SELECT 1 FROM episode_source_occurrences WHERE episode_id=? LIMIT 1", (key,)).fetchone()
            return "retained_unverified" if partial is None else "unsafe_unknown"
        # An unpublished digest generation is staged, not proven retired.
    elif index == 4:
        from hymem.dreaming.facts import load_fact_outcome_source_manifest
        source = conn.execute("SELECT * FROM narrative_facts WHERE id=?", (key,)).fetchone()
        if source is None:
            return "unsafe_unknown"
        if source["lifecycle_status"] == "legacy_unproven":
            partial = conn.execute("SELECT 1 FROM narrative_fact_lifecycle WHERE fact_id=? LIMIT 1", (key,)).fetchone()
            if source["source_outcome_key"] is None and source["current_generation"] is None and partial is None:
                return "retained_unverified"
            return "unsafe_unknown"
        if source["lifecycle_status"] == "retracted":
            # The maintained outcome validator folds every lifecycle revision
            # and verifies the retracted projection, not just a status flag.
            proof = (load_fact_outcome_source_manifest(conn, source["source_outcome_key"])
                     if source["source_outcome_key"] is not None else None)
            return "retired" if proof is not None else "unsafe_unknown"
    elif index == 5:
        source = conn.execute("SELECT * FROM aggregation_nodes WHERE id=?", (key,)).fetchone()
        if source is None:
            return "unsafe_unknown"
        if (_incomplete_manifest(source, legacy_version="aggregation-source-manifest-v1")
                and source["input_manifest_complete"] == 0
                and source["input_manifest_count"] == 0 and source["input_manifest_hash"] is None
                and source["input_manifest_version"] is None and source["publication_id"] is None):
            partial = conn.execute(
                "SELECT 1 FROM aggregation_node_source_occurrences WHERE node_id=? "
                "UNION ALL SELECT 1 FROM aggregation_node_inputs WHERE node_id=? LIMIT 1", (key, key),
            ).fetchone()
            return "retained_unverified" if partial is None else "unsafe_unknown"
        # Source-proof validity alone cannot authorize embedding-only changes
        # to a publication-bound aggregate. Do not label this repairable.
        return "rebuild_required"
    return "source_eligible" if reembed._source(conn, index, row) is not None else "unsafe_unknown"


def _scan_sources(conn, inventory, model, dim, budget, max_candidates):
    results = []
    used = 0
    for index, raw in enumerate(inventory.tables):
        if raw.incompatible is None:
            results.append(_unknown(raw.table, None, "inventory_unavailable"))
            continue
        if not raw.incompatible:
            results.append(SourceMirrorHealth(raw.table, 0, 0, 0, 0, 0, 0))
            continue
        if used + raw.incompatible > max_candidates:
            results.append(_unknown(raw.table, raw.incompatible, "candidate_budget_exhausted"))
            continue
        counts = dict.fromkeys(_CATEGORIES, 0)
        terminal_source_loss = 0
        cursor = None
        try:
            budget.check()
            cursor = conn.execute(f'SELECT * FROM "{raw.table}" WHERE model<>? OR dim<>?', (model, dim))
            for row in cursor:
                budget.check()
                typed = row["embedding_producer_key"] if raw.table in _TYPED_PRODUCER_TABLES else row["model"]
                if (not _exact_key(row["model"]) or not _positive_dimension(row["dim"])
                        or typed != row["model"] or not _valid_vector(row["vector_json"], row["dim"])):
                    continue  # Already a separate unconditional malformed FAIL.
                used += 1
                if used > max_candidates:
                    raise _AuditLimit("candidate_budget_exhausted")
                try:
                    category = _classify(_ProofReader(conn, budget), index, row)
                except (RuntimeError, ValueError, TypeError, KeyError, IndexError, UnicodeError):
                    category = "unsafe_unknown"
                budget.check()
                counts[category] += 1
                if category == "retained_unverified" and index == 0:
                    terminal_source_loss += int(conn.execute(
                        "SELECT 1 FROM chunk_extraction_terminal_losses WHERE chunk_id=? LIMIT 1", (row[_KEYS[index]],),
                    ).fetchone() is not None)
            budget.check()
            if sum(counts.values()) != raw.incompatible:
                results.append(_unknown(raw.table, raw.incompatible, "classification_count_mismatch"))
            else:
                results.append(SourceMirrorHealth(raw.table, raw.incompatible, **counts,
                                                  terminal_source_loss=terminal_source_loss))
        except _AuditLimit as exc:
            results.append(_unknown(raw.table, raw.incompatible, exc.code))
        except sqlite3.Error:
            code = "diagnostic_budget_exhausted" if budget.exhausted else "source_schema_or_read_failure"
            results.append(_unknown(raw.table, raw.incompatible, code))
        finally:
            if cursor is not None:
                cursor.close()
    return tuple(results)


def scan_embedding_recovery_health(
    path: Path, *, live_model: str | None, live_dim: int | None,
    max_candidates: int = 4096, max_seconds: float = 60.0,
    max_sql_steps: int = 100_000_000,
) -> EmbeddingRecoveryHealth:
    """Explain stored mismatches within explicit work bounds, never mutate.

    Defaults accommodate the known ~10,000-vector store without proving its
    compatible rows. At most 4,096 incompatible rows receive proofs, each
    capped at 8,192 fetched rows / 8 MiB of values; SQLite values are capped at
    8 MiB. The entire inventory and classification share a 60-second cooperative
    Python deadline and 100-million SQLite-step ceiling. Exhaustion is unknown,
    not a partial green count. Bounds can be raised for larger offline audits.
    """
    verified = _exact_key(live_model) and _positive_dimension(live_dim)
    inventory = _failed_inventory(verified, "snapshot_unavailable")
    conn = None
    code = "snapshot_unavailable"
    if (type(max_candidates) is not int or not 1 <= max_candidates <= 100000
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 300
            or type(max_sql_steps) is not int or not 1 <= max_sql_steps <= 1_000_000_000):
        code = "invalid_bounds"
        return EmbeddingRecoveryHealth(inventory, tuple(_unknown(table, None, code) for table in MIRROR_TABLES))
    budget = _Budget(max_seconds, max_sql_steps)
    try:
        from hymem.core import db
        budget.check()
        conn = sqlite3.connect(Path(path).absolute().as_uri() + "?mode=ro", uri=True,
                               isolation_level=None, timeout=min(1.0, max_seconds))
        conn.row_factory = sqlite3.Row
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 8 * 1024 * 1024)
        conn.set_progress_handler(budget.progress, budget.interval)
        db.register_read_authority_functions(conn)
        conn.execute("BEGIN")
        inventory = scan_embedding_health(conn, live_model=live_model, live_dim=live_dim)
        budget.check()
        if db.schema_version(conn) != db.EXPECTED_SCHEMA_VERSION:
            return EmbeddingRecoveryHealth(inventory, tuple(
                _unknown(item.table, item.incompatible, "current_schema_required") for item in inventory.tables
            ))
        rows = _scan_sources(conn, inventory, live_model, live_dim, budget, max_candidates)
        return EmbeddingRecoveryHealth(inventory, rows)
    except _AuditLimit as exc:
        code = exc.code
    except (sqlite3.Error, OSError, ValueError, TypeError, UnicodeError):
        code = "diagnostic_budget_exhausted" if budget.exhausted else "snapshot_unavailable"
    finally:
        if conn is not None:
            conn.set_progress_handler(None, 0)
            if conn.in_transaction:
                conn.rollback()
            conn.close()
    return EmbeddingRecoveryHealth(inventory, tuple(
        _unknown(item.table, item.incompatible, code) for item in inventory.tables
    ))
