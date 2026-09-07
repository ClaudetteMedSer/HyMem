from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping
from urllib.parse import urlsplit

from hymem import portability
from hymem import redaction
from hymem import rules as rules_mod
from hymem import session as session_log
from hymem.config import HyMemConfig
from hymem.deadline import MonotonicDeadline
from hymem.core import db as core_db
from hymem.core.graph import graph_clock_order_sql, live_edge_predicate
from hymem.dreaming import canonicalize as canon
from hymem.dreaming import evidence as evidence_ledger
from hymem.dreaming.aggregate import (
    Digest,
    NodeExpansion,
    aggregation_config_version,
    expand_node as aggregate_expand_node,
    load_digest,
)
from hymem.dreaming.aggregation_health import aggregation_health_status
from hymem.dreaming.aggregation_material import embedding_execution_identity
from hymem.dreaming.aggregation_generation import (
    aggregation_generation_binding_for_contract,
    aggregation_generation_contract,
)
from hymem.dreaming.lossless import (
    COVERAGE_INTEGRITY_CONFIG_VERSION,
    materialize_message_coverage,
)
from hymem.dreaming.phase1_auxiliary import (
    add_manual_entity_property,
    add_manual_entity_type,
)
from hymem.dreaming.embeddings import (
    fetch_message_embeddings,
    message_embedding_id_batches,
    persist_message_embeddings,
)
from hymem.dreaming.status import durable_dream_work_status
from hymem.dreaming.runner import DreamLeaseLost, DreamReport, run_dreaming
from hymem.dreaming.user_profile import (
    ProfileEntry,
    enforce_profile_redaction_policy,
    load_profile,
)
from hymem.extraction.embeddings import EmbeddingClient
from hymem.extraction.llm import LLMClient
from hymem.extraction.producer import phase1_generation_binding
from hymem.query.ask import Answer, ask as query_ask
from hymem.query.augment import AugmentedContext, augment, build_token_overlap_index
from hymem.query.conflicts import Conflict, find_conflicts
from hymem.query.entities import (
    GraphCount,
    TimelineEntry,
    count_relations as query_count_relations,
    timeline as query_timeline,
)
from hymem.query.graph_state import AsOfGraphFact, facts_at as query_facts_at

log = logging.getLogger("hymem.api")


def _clean_timestamp(value: str | None) -> str | None:
    """Type-check a caller timestamp without changing its wire spelling."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("created_at must be an ISO-8601 string")
    return value


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})


def _ensure_embedding_server(timeout: float = 45.0) -> bool:
    """Best-effort: make sure a *local* embedding server is reachable, restarting
    it if it has died unexpectedly.

    Self-healing at the point of use: the code path that would otherwise crash on
    a dead embedder (``dream``) is the one that revives it. There is no watchdog
    process and no polling while idle — the check runs only when an embedder is
    actually about to be used, so the steady-state cost is one ``/health`` GET.

    Returns ``True`` when the embedding path is safe to proceed, ``False`` only
    when a local server was down and could not be brought back up.

    Decision table:
      - ``HYMEM_EMBEDDING_BASE_URL`` unset  → ``True``  (FTS-only / benchmark path)
      - remote URL (not loopback)           → ``True``  (can't restart a remote host)
      - local server answers ``/health``    → ``True``  (fast path, no work)
      - local server down, restart cmd set  → Popen it, poll ``/health`` to timeout
      - local server down, no restart cmd   → ``False`` (nothing we can do)
    """
    base_url = os.environ.get("HYMEM_EMBEDDING_BASE_URL")
    if not base_url:
        return True
    from hymem.contrib.endpoint_policy import (
        EMBEDDING_INTERNAL_HTTP_ENV,
        validate_http_endpoint,
    )
    from hymem.contrib.openai_embedding_client import safe_embedding_base_url
    display_url = safe_embedding_base_url(base_url)
    try:
        endpoint = validate_http_endpoint(
            base_url,
            label="embedding",
            allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
        )
    except ValueError:
        log.warning("embedding endpoint %s was rejected by transport policy", display_url)
        return False

    parts = urlsplit(endpoint.url)
    host = (parts.hostname or "").lower()
    if host not in _LOCAL_HOSTS:
        # Remote provider: not ours to manage. Skip rather than fail.
        return True

    # Health lives at the server root, not under the OpenAI-style path segment
    # base_url carries for /v1/embeddings (e.g. .../v1). Build it from the origin
    # only — appending to base_url would probe /v1/health, which 404s, falsely
    # reporting a healthy server as down (spurious restart + full-timeout stall).
    health_url = f"{parts.scheme}://{parts.netloc}/health"
    if _embedding_health_ok(health_url):
        return True  # already up — nothing to do

    cmd = os.environ.get("HYMEM_EMBEDDING_SERVER_CMD")
    if not cmd:
        log.warning(
            "embedding server at %s is down and HYMEM_EMBEDDING_SERVER_CMD is "
            "not set — cannot restart it automatically", display_url,
        )
        return False

    # The operator command may contain credentials or an opaque tenant route.
    # Never render it into process logs.
    log.warning("embedding server at %s is down — restart requested", display_url)
    try:
        # Detach so the embedding server outlives this restart trigger; inherit
        # the environment so HYMEM_EMBEDDING_* stay consistent with the client.
        subprocess.Popen(  # noqa: S603 - cmd is operator-supplied configuration
            shlex.split(cmd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        log.warning(
            "embedding server launch failed error_type=%s",
            type(exc).__name__,
        )
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _embedding_health_ok(health_url):
            log.info("embedding server at %s is back up", display_url)
            return True
        time.sleep(1.0)

    log.warning(
        "embedding server at %s did not become healthy within %.0fs",
        display_url, timeout,
    )
    return False


def _embedding_health_ok(health_url: str, timeout: float = 2.0) -> bool:
    """Return True iff ``GET health_url`` answers with a 2xx status."""
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


class HyMem:
    """Public API for the Hermes host.

    Hermes is responsible for:
      - constructing one HyMem per agent process (or per project root),
      - calling `log_message` for every conversational turn,
      - calling `augment` before sending a user turn to its model,
      - calling `dream` during idle windows (or via its own scheduler),
      - providing an LLMClient — required only for `dream`, not for `augment`.
    """

    def __init__(
        self,
        config: HyMemConfig,
        *,
        llm: LLMClient | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config
        # Resolve before assignment so an unsafe explicit declaration (for
        # example a credential-bearing endpoint) cannot leave the instance in
        # a half-switched state. Unknown clients receive their conservative
        # process-instance binding here and keep it for this object lifetime.
        self._phase1_generation = (
            phase1_generation_binding(config.prompt_version, llm)
            if llm is not None else None
        )
        self._aggregation_contract = (
            aggregation_generation_contract(config)
            if config.aggregation_nodes_enabled else None
        )
        self._aggregation_generation = (
            aggregation_generation_binding_for_contract(
                self._aggregation_contract, llm,
            )
            if llm is not None and self._aggregation_contract is not None else None
        )
        self._llm = llm
        self._embed = embedding_client
        self._conn: sqlite3.Connection | None = None
        self._read_conn: sqlite3.Connection | None = None
        self._initialized = False
        # Token-overlap index for entity expansion in augment(). Built lazily
        # on first augment, invalidated after dreaming since dreaming is the
        # only thing that mutates the canonical set.
        self._token_overlap_index: dict[str, list[str]] | None = None

    # ---- lifecycle ---------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = core_db.connect(self.config.db_path)
        if not self._initialized:
            core_db.initialize(self._conn)
            if self.config.redact_secrets:
                with core_db.transaction(self._conn):
                    enforce_profile_redaction_policy(self._conn)
            self._initialized = True
            self._scope_phase1_reads(self._conn)
        return self._conn

    @property
    def read_conn(self) -> sqlite3.Connection:
        if self._read_conn is None:
            self.conn  # ensure the write connection is initialized first
            self._read_conn = core_db.connect(self.config.db_path)
            self._scope_phase1_reads(self._read_conn)
            # Load optional vec0 acceleration shadows before query_only. Durable
            # identity/content-validated vectors remain retrieval authority.
            core_db._load_vec_extension(self._read_conn)
            self._read_conn.execute("PRAGMA query_only = ON")
        return self._read_conn

    def close(self) -> None:
        if self._read_conn is not None:
            self._read_conn.close()
            self._read_conn = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False

    def set_llm(self, llm: LLMClient) -> None:
        generation = phase1_generation_binding(self.config.prompt_version, llm)
        aggregation_generation = (
            aggregation_generation_binding_for_contract(
                self._aggregation_contract, llm,
            )
            if self._aggregation_contract is not None else None
        )
        self._phase1_generation = generation
        self._aggregation_generation = aggregation_generation
        self._llm = llm
        if self._conn is not None:
            self._scope_phase1_reads(self._conn)
        if self._read_conn is not None:
            self._scope_phase1_reads(self._read_conn)

    def _scope_phase1_reads(self, conn: sqlite3.Connection) -> None:
        generation_key = (
            str(self._phase1_generation["generation_key"])
            if self._phase1_generation is not None else None
        )
        core_db.register_current_phase1_generation(conn, generation_key)
        core_db.register_current_rule_routing(
            conn, rules_mod.rule_routing_key(self.config, generation_key)
        )

    def _current_aggregation_generation_key(self) -> str | None:
        """Resolve the live producer at each aggregation read boundary.

        Maintained clients expose mutable Python attributes, so the binding
        cached by ``__init__``/``set_llm`` is only lifecycle metadata. It must
        never authorize reads after an in-place model, endpoint, request, or
        retry-policy change.
        """

        if self._llm is None or self._aggregation_contract is None:
            return None
        current = aggregation_generation_binding_for_contract(
            self._aggregation_contract, self._llm,
        )
        return str(current["generation_key"])

    def set_embedding_client(self, embedding_client: EmbeddingClient) -> None:
        self._embed = embedding_client

    @property
    def embedding_status(self) -> dict[str, object]:
        """Resolved embedding backend identity without probing the network."""
        if self._embed is None:
            return {
                "configured": False,
                "backend": "none",
                "quality": "none",
                "network_free": True,
                "model": None,
                "dim": None,
                "fallback_reason": None,
            }
        try:
            _binding, model, dim = embedding_execution_identity(self._embed)
        except Exception:
            model, dim = None, None
        try:
            backend_value = str(getattr(self._embed, "backend", "configured"))
        except Exception:
            backend_value = "configured"
        backend = (
            backend_value if backend_value in {
                "configured", "local_feature_hash", "openai_compatible",
            } else "configured"
        )
        try:
            quality_value = str(getattr(self._embed, "quality", "semantic"))
        except Exception:
            quality_value = "semantic"
        quality = quality_value if quality_value in {"lexical", "semantic"} else "semantic"
        try:
            network_free = bool(getattr(self._embed, "network_free", False))
        except Exception:
            network_free = False
        try:
            fallback_reason = getattr(self._embed, "fallback_reason", None)
        except Exception:
            fallback_reason = None
        if fallback_reason not in {
            "remote_embedding_credentials_missing",
            "remote_embedding_endpoint_rejected",
            "remote_embedding_client_unavailable",
        }:
            fallback_reason = None
        return {
            "configured": True,
            "backend": backend,
            "quality": quality,
            "network_free": network_free,
            "model": model,
            "dim": dim,
            "fallback_reason": fallback_reason,
        }

    def _embed_pending_messages_best_effort(
        self, message_ids: Iterable[int]
    ) -> None:
        """Fill the durable message-vector mirror without holding a write lock.

        Public ingestion has already committed and validated/redacted the
        source before this runs. A provider failure therefore never rolls back
        accepted history and leaves a plainly retryable missing embedding for
        the next message or dream cycle.
        """
        if self._embed is None:
            return
        if self.conn.in_transaction:
            raise RuntimeError("message embedding must run outside a transaction")
        consecutive_failures = 0
        for batch in message_embedding_id_batches(
            self.conn, message_ids=tuple(message_ids)
        ):
            try:
                pending = fetch_message_embeddings(
                    self.conn, self._embed, message_ids=batch
                )
                if pending is not None:
                    with core_db.transaction(self.conn):
                        persist_message_embeddings(self.conn, pending)
            except Exception as exc:  # provider failure leaves retryable rows
                consecutive_failures += 1
                log.error(
                    "embedding.message_ingest_failure batch_size=%d error=%s",
                    len(batch), type(exc).__name__,
                )
                # Bound a genuinely unavailable provider to two timed attempts,
                # while allowing a single poison/oversized batch not to block
                # later committed groups in the same public batch.
                if consecutive_failures >= 2:
                    break
            else:
                consecutive_failures = 0

    def fork(self) -> "HyMem":
        """Return a new HyMem on the same database with its own SQLite
        connection, reusing this instance's LLM and embedding clients.

        Used to run background dreaming on a separate connection so it does
        not collide with live ingestion on the primary connection.
        """
        return HyMem(self.config, llm=self._llm, embedding_client=self._embed)

    # ---- session log -------------------------------------------------

    def open_session(
        self,
        session_id: str,
        *,
        source_workspace_id: str | None = None,
    ) -> None:
        with core_db.transaction(self.conn):
            session_log.open_session(
                self.conn, session_id,
                source_workspace_id=source_workspace_id,
            )

    def close_session(self, session_id: str) -> None:
        with core_db.transaction(self.conn):
            session_log.close_session(self.conn, session_id)

    def _prepare_content(self, content: str) -> str:
        """Apply the ingest-boundary guards (size cap, then secret redaction)
        before any message text reaches SQLite. Truncation happens first so the
        redactor never has to scan an unbounded string."""
        cap = self.config.max_message_chars
        if cap and len(content) > cap:
            log.warning(
                "log_message: content of %d chars exceeds max_message_chars=%d; truncating",
                len(content), cap,
            )
            content = content[:cap] + "\n[TRUNCATED]"
        if self.config.redact_secrets:
            content = redaction.redact(content)
        return content

    def log_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        created_at: str | None = None,
        source_peer_id: str | None = None,
        source_workspace_id: str | None = None,
    ) -> int:
        """Atomically append one source turn and its lossless coverage proof.

        ``created_at`` is the message occurrence/source-valid time, not a
        scheduled future effective date. It must be strict ISO-8601 and may
        lead HyMem's observation clock by at most 300 seconds for producer
        skew; invalid/further-future input is rejected without writing either
        the raw turn or coverage artifact. Omit it to use ingestion time.
        """
        content = self._prepare_content(content)
        with core_db.transaction(self.conn):
            session_log.open_session(
                self.conn, session_id,
                source_workspace_id=source_workspace_id,
            )
            if source_peer_id is not None:
                if source_workspace_id is None:
                    raise ValueError(
                        "source_peer_id and source_workspace_id must be provided together"
                    )
                session_log.register_session_peer(
                    self.conn, session_id, source_workspace_id,
                    source_peer_id, role,
                )
            message_id = session_log.append_message(
                self.conn, session_id, role, content,
                created_at=_clean_timestamp(created_at),
                source_peer_id=source_peer_id,
                source_workspace_id=source_workspace_id,
            )
            # Source and durable canonical artifact commit together.  Dreaming
            # still backfills legacy/direct-SQL rows, but public ingestion never
            # leaves a newly accepted message outside the lossless stream.
            materialize_message_coverage(self.conn, session_id)
        self._embed_pending_messages_best_effort((message_id,))
        return message_id

    def log_messages(
        self,
        session_id: str,
        turns: Iterable[tuple[str, str] | tuple[str, str, str | None]],
        *,
        source_peer_ids: Iterable[str | None] | None = None,
        source_workspace_id: str | None = None,
    ) -> list[int]:
        """Append a batch of turns in a single transaction.

        Each turn is `(role, content)` or `(role, content, created_at)`, where
        `created_at` is the caller-supplied occurrence/source-valid time, not a
        scheduled-effective date. It must be strict ISO-8601 and may lead the
        observation clock by at most 300 seconds; omit it (or pass the 2-tuple)
        to use ingestion time. Any invalid turn rolls back the full batch and
        its coverage artifacts. One BEGIN IMMEDIATE covers the whole batch.
        """
        prepared = [
            (
                turn[0],
                self._prepare_content(turn[1]),
                _clean_timestamp(turn[2] if len(turn) > 2 else None),
            )
            for turn in turns
        ]
        peers = (
            list(source_peer_ids)
            if source_peer_ids is not None
            else [None] * len(prepared)
        )
        if len(peers) != len(prepared):
            raise ValueError("source_peer_ids must align one-to-one with turns")
        if any(peer is not None for peer in peers) and source_workspace_id is None:
            raise ValueError(
                "source_peer_ids and source_workspace_id must be provided together"
            )
        if source_workspace_id is not None and any(peer is None for peer in peers):
            raise ValueError(
                "workspace-qualified batches require an exact peer for every turn"
            )
        with core_db.transaction(self.conn):
            session_log.open_session(
                self.conn, session_id,
                source_workspace_id=source_workspace_id,
            )
            for (role, _content, _created_at), peer_id in zip(prepared, peers):
                if peer_id is not None:
                    session_log.register_session_peer(
                        self.conn, session_id, source_workspace_id,
                        peer_id, role,
                    )
            message_ids = [
                session_log.append_message(
                    self.conn, session_id, role, content, created_at=created_at,
                    source_peer_id=peer_id,
                    source_workspace_id=source_workspace_id,
                )
                for (role, content, created_at), peer_id in zip(prepared, peers)
            ]
            materialize_message_coverage(self.conn, session_id)
        self._embed_pending_messages_best_effort(message_ids)
        return message_ids

    def add_rule(
        self,
        text: str,
        *,
        scope: str = "always_on",
        trigger_entities: list[str] | None = None,
        source: str = "user",
        supersedes: int | None = None,
    ) -> int:
        """Add (or reinforce) a standing behavioral rule; return its id.

        Rules are the imperative subset of "always loaded" context ("always run
        the tests before pushing", "never suggest Docker") — a first-class node
        type (schema v23) injected into every `augment()` call via `ctx.rules`
        when `cfg.rules_enabled` is set. `scope='always_on'` (default) injects
        unconditionally; `scope='contextual'` injects only when a
        `trigger_entities` member overlaps the call's matched entities.

        Rules are often *told*, not inferred, so this mirrors the `HyMem.ask()`
        direct-API pattern. `supersedes` closes a prior rule's validity interval
        (bi-temporal — a contradicting instruction supersedes rather than
        overwrites). Re-asserting identical text reinforces instead of
        duplicating. Text is redaction-scrubbed at persist time.
        """
        if source != "user":
            raise ValueError(
                "HyMem.add_rule accepts user/told rules only; inferred rules "
                "are created by the producer-scoped marker router"
            )
        with core_db.transaction(self.conn):
            return rules_mod.add_rule(
                self.conn,
                text,
                scope=scope,
                trigger_entities=trigger_entities,
                source="user",
                supersedes=supersedes,
            )

    def set_entity_type(
        self,
        entity: str,
        type_name: str,
        *,
        confidence: float = 1.0,
    ) -> None:
        """Assert a user-authored entity type with explicit durable authority.

        This is the supported alternative to writing ``entity_types``
        directly.  It cannot be hidden, reinforced, or overwritten by an LLM
        extraction; a later explicit call may update its confidence.
        """
        with core_db.transaction(self.conn):
            add_manual_entity_type(
                self.conn, entity, type_name, confidence=confidence,
            )
        self._token_overlap_index = None

    def set_entity_property(self, entity: str, key: str, value: str) -> None:
        """Assert a user-authored property with explicit durable authority.

        Identifiers must be non-empty.  An empty string remains a meaningful
        explicit value; when secret redaction is enabled, the value is scrubbed
        at this persistence boundary.
        """
        if not isinstance(value, str):
            raise ValueError("entity property value must be a string")
        prepared_value = (
            redaction.redact(value) if self.config.redact_secrets else value
        )
        with core_db.transaction(self.conn):
            add_manual_entity_property(self.conn, entity, key, prepared_value)
        self._token_overlap_index = None

    def retract_rule(self, rule_id: int) -> None:
        """Retire a rule by closing its validity interval (`status='retracted'`
        + `invalid_at`); it stops surfacing in `ctx.rules` but stays in the table
        as auditable history. Idempotent."""
        with core_db.transaction(self.conn):
            rules_mod.retract_rule(self.conn, rule_id)

    def rules(self) -> list[rules_mod.Rule]:
        """Every ACTIVE rule (always_on + contextual), the whole rulebook — the
        read-side counterpart to `add_rule`, mirroring `profile()`. Unlike the
        per-call `ctx.rules` (trigger-gated, capped) this is the full set, for a
        host that wants to show or audit what standing rules exist. Read-only."""
        return rules_mod.list_rules(self.read_conn)

    def suggest_rules(
        self,
        *,
        limit: int | None = None,
        mode: str = "llm",
        confidence_min: float | None = None,
    ) -> list[rules_mod.RuleCandidate]:
        """Propose standing rules the durability tagger infers from recent
        (unconsolidated) behavioral markers, ranked and de-duplicated — for a
        human or agent to confirm via `add_rule`. **Read-only: nothing is
        persisted.**

        This is the candidate-suggestion counterpart to the (default-OFF)
        write-side auto-extraction. Auto-*injecting* inferred rules didn't clear
        the precision gate on real markers — the tagger reliably FINDS standing
        directives (high recall) but over-fires on one-offs — so instead of
        silently minting rules, this surfaces candidates and lets the confirming
        human/agent be the precision gate. Each `RuleCandidate` shows its
        corroboration (markers over distinct sessions), confidence, source kinds,
        the raw supporting statements, and whether it's `already_active`.

        Typical flow: log a session → `suggest_rules()` to review → `add_rule()`
        the good ones → `dream()`. Requires an LLMClient (the tagger), like
        `ask()`/`dream()`; returns `[]` when no marker clears the tagger.
        """
        if self._llm is None:
            raise RuntimeError(
                "HyMem.suggest_rules requires an LLMClient (the durability tagger). "
                "Pass one to the constructor or call set_llm() before suggesting."
            )
        return rules_mod.suggest_rules_from_markers(
            self.read_conn, self.config, self._llm,
            limit=limit, mode=mode, confidence_min=confidence_min,
            phase1_generation_key=phase1_generation_binding(
                self.config.prompt_version, self._llm
            )["generation_key"],
        )

    # ---- query-time --------------------------------------------------

    def augment(
        self,
        user_message: str,
        *,
        session_id: str | None = None,
        source_session_id: str | None = None,
        source_peer_id: str | None = None,
        source_workspace_id: str | None = None,
        ability: str | None = None,
    ) -> AugmentedContext:
        """Build retrieval context for `user_message`.

        `ability` is an optional question-type hint (e.g. "MR") the host may pass
        to shape retrieval — only the host knows the type, HyMem does not infer
        it. `ability="MR"` switches the raw-message tier into aggregation mode
        (all matches, chronological, with `total_message_matches`) for
        "how many X across all my requests?" questions. Unknown/None hints use
        the default retrieval path.
        """
        cap = self.config.max_query_chars
        if cap and len(user_message) > cap:
            log.warning(
                "augment: query of %d chars exceeds max_query_chars=%d; truncating",
                len(user_message), cap,
            )
            user_message = user_message[:cap]
        if self._token_overlap_index is None:
            self._token_overlap_index = build_token_overlap_index(
                self.read_conn, write_conn=self.conn
            )
        return augment(
            self.read_conn, self.config, user_message,
            embedding_client=self._embed,
            llm=self._llm,
            token_overlap_index=self._token_overlap_index,
            session_id=session_id,
            source_session_id=source_session_id,
            source_peer_id=source_peer_id,
            source_workspace_id=source_workspace_id,
            ability=ability,
            aggregation_generation_key=self._current_aggregation_generation_key(),
        )

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        ability: str | None = None,
        include_digest: bool = False,
    ) -> Answer:
        """The dialectic/synthesis endpoint: one call, a reasoned answer,
        grounded in the retrieval tiers.

        Where `augment()` returns raw tiers for the HOST to assemble into its
        own prompt, `ask()` closes the loop inside HyMem: it runs the same
        retrieval (`session_id`/`ability` pass straight through), renders the
        tiers into a compact most-authoritative-first context block (capped at
        `cfg.ask_max_context_chars`), and makes ONE completion against the
        host-provided LLM under `ASK_PROMPT_V2` — answer only from the
        context, quote concrete values/dates, state contradictions with their
        dates (most recent value-bearing statement wins), soften
        low-confidence facts, and say plainly when the memory doesn't contain
        the answer. The returned `Answer` keeps the full `AugmentedContext`
        for provenance/drill-down plus the rendered block size actually sent.

        `include_digest=True` additionally loads the standing whole-store
        digest (see `digest()`) into the context, so a global "what do you
        know about me?" can draw on it; off by default because per-query
        retrieval usually answers better without dream-time standing context.

        Hosts that want the raw tiers (Hermes) keep using `augment()` — this
        endpoint exists for one-call consumers (the MCP `hymem_ask` tool,
        Honcho-style dialectic chat). Requires an LLMClient, like `dream()`.
        """
        if self._llm is None:
            raise RuntimeError(
                "HyMem.ask requires an LLMClient. Pass one to the constructor "
                "or call set_llm() before asking."
            )
        ctx = self.augment(question, session_id=session_id, ability=ability)
        # `ctx.digest` may already be populated when cfg.augment_include_digest
        # is on; only load it here when the caller asked and augment didn't.
        if include_digest and ctx.digest is None:
            ctx.digest = self.digest()
        return query_ask(self.config, self._llm, question, ctx)

    def digest(self) -> Digest | None:
        """The standing whole-store summary — the root of the RAPTOR
        aggregation tree, answering "what do you know about me?" at a glance.
        Intended as host-facing standing context (e.g. system-prompt
        injection), NOT as a retrieval tier: it is rebuilt at dream time and
        never competes with per-query retrieval. Returns None until the
        aggregation layer (`cfg.aggregation_nodes_enabled` +
        `cfg.aggregation_digest_enabled`) has dreamed over at least one
        episode. Read-only.

        Embedded-host pattern: inject `digest().as_context_block()` into the
        system prompt and re-fetch after each `dream()` — the digest only
        changes at dream time, so there is nothing to poll between dreams.
        The block's footer carries session coverage and `generated_at`, making
        staleness visible to the model and the user. Hosts that prefer a
        single call per turn can set `cfg.augment_include_digest` to receive
        the same object as `ctx.digest` from `augment()` instead.
        """
        if (
            not self.config.aggregation_nodes_enabled
            or not self.config.aggregation_digest_enabled
        ):
            return None
        return load_digest(
            self.read_conn,
            expected_config_version=aggregation_config_version(self.config),
            expected_cluster_min_members=self.config.aggregation_min_members,
            expected_cluster_min_sessions=self.config.aggregation_min_sessions,
            expected_anchor_fact_cap=self.config.aggregation_digest_anchor_facts,
            expected_generation_key=self._current_aggregation_generation_key(),
            embedding_client=self._embed,
        )

    def expand_node(self, node_id: str) -> NodeExpansion | None:
        """Drill one level down into the RAPTOR aggregation tree — the
        provenance read behind "why does my digest say X?". Resolves the
        node's persisted members into child nodes and member episodes (each
        episode carrying its exact ordered source-message coordinates and
        ownership scope), so a host can
        walk from the standing digest (`digest().node_id`) or a query-tier
        node (`AggregationNodeHit.node_id`) all the way down to the original
        turns. Returns None for an unknown id (e.g. a stale id from before
        the last dream — nodes are rebuilt from scratch each cycle, so ids
        are only stable while the underlying episode membership is).
        Read-only; never an LLM call.
        """
        if not self.config.aggregation_nodes_enabled:
            return None
        return aggregate_expand_node(
            self.read_conn, node_id,
            expected_config_version=aggregation_config_version(self.config),
            expected_cluster_min_members=self.config.aggregation_min_members,
            expected_cluster_min_sessions=self.config.aggregation_min_sessions,
            expected_anchor_fact_cap=self.config.aggregation_digest_anchor_facts,
            expected_generation_key=self._current_aggregation_generation_key(),
            embedding_client=self._embed,
        )

    def profile(self) -> list[ProfileEntry]:
        """ACTIVE typed user-profile rows (schema v18) — the durable personal
        facts (name, role, employer, location, language, relationship(person),
        possession, age_birthday, health_condition, recurring_activity)
        extracted from USER turns during dreaming under a closed slot
        vocabulary. Identity slots first; superseded rows (invalid_at set) are
        excluded, so this is the CURRENT profile. Values were
        redaction-scrubbed at persist time. Empty before the first dream (or
        with `profile_extraction_enabled=False`, nothing is ever extracted).
        Read-only.
        """
        return load_profile(self.read_conn)

    def timeline(self, entity: str) -> list["TimelineEntry"]:
        """Earliest source-valid edge per predicate for ``entity``.

        Current direct observations are returned by default and ordered by
        ``valid_at`` -- never ingestion time. ``entity`` may be a surface form;
        it is resolved through the alias table. Inferred rows are excluded
        because no derivation-time lineage is persisted. Read-only.
        """
        return query_timeline(self.read_conn, entity)

    def facts_at(
        self,
        valid_time: str,
        *,
        recorded_at: str | None = None,
        entity: str | None = None,
    ) -> list[AsOfGraphFact]:
        """Direct graph facts valid at one source-time coordinate.

        This reconstructs half-open validity intervals from the immutable edge
        lifecycle (assert -> retract -> reassert), rather than consulting the
        current materialized row. ``recorded_at`` is an independent optional
        authority cutoff: omit it for today's authoritative revisions, or
        provide it to select revisions/events that existed then. Canonical
        merges are not transaction-versioned, so either mode projects onto
        today's entity topology rather than claiming a literal old graph shape.
        This lifecycle view changes only on persisted transitions; ordinary
        current retrieval additionally applies the active/open/evidence-majority
        gate and may therefore be a conservative subset. Both inputs are
        validated ISO timestamps and normalized to UTC. ``entity`` is an
        optional alias-aware subject/object filter. Read-only.
        """
        return query_facts_at(
            self.read_conn,
            valid_time,
            recorded_at=recorded_at,
            entity=entity,
        )

    def conflicts(self) -> list[Conflict]:
        """Return detected contradictions in the knowledge graph. Read-only.

        Surfaces edges that disagree — competing objects under an exclusive
        predicate, or a subject/object pair joined by opposing predicates.
        """
        return find_conflicts(self.read_conn)

    def count_relations(
        self,
        *,
        count: Literal["subject", "object"] = "subject",
        predicates: Iterable[str] | None = None,
        subject: str | None = None,
        object: str | None = None,
        object_type: str | None = None,
        subject_type: str | None = None,
        include_derived: bool = False,
    ) -> GraphCount:
        """Exact graph-native count over active `knowledge_graph` edges, for
        in-domain "how many X …" questions. Read-only.

        Answers questions the aggregate message-FTS path can only *estimate*,
        because here the entity type vocabulary applies: "how many services
        depend on redis?" is `count="subject", predicates=["depends_on"],
        object="redis"`; "how many databases do we use?" is `count="object",
        predicates=["uses"], object_type="database"`. Only the IN-DOMAIN (tech)
        type/predicate vocabulary is countable this way — consumer categories
        ("clothing") aren't typed, so those stay on the message-FTS aggregate
        path (`augment(ability="MR")`).

        `count` is the load-bearing argument: it states whether DISTINCT subjects
        or DISTINCT objects are tallied (default `"subject"` — the canonical
        "how many subjects relate to this object" shape). HyMem never infers the
        side from the filters. `subject`/`object` surface forms are resolved
        through the alias table; `subject_type`/`object_type` filter via
        `entity_types`; `predicates` is optional (omit ⇒ all predicates). The
        Current direct observations are the default (active, open interval,
        positive evidence majority). Pass ``include_derived=True`` only when a
        closure-inclusive count is intentional. The returned `GraphCount`
        carries the exact `count`, the distinct entities
        behind it (capped, count stays exact), and the resolved filters used.
        """
        return query_count_relations(
            self.read_conn,
            count=count,
            predicates=predicates,
            subject=subject,
            object=object,
            object_type=object_type,
            subject_type=subject_type,
            include_derived=include_derived,
        )

    # ---- dreaming ----------------------------------------------------

    def dream(
        self,
        *,
        session_ids: Iterable[str] | None = None,
        deadline: MonotonicDeadline | None = None,
    ) -> DreamReport:
        if self._llm is None:
            raise RuntimeError(
                "HyMem.dream requires an LLMClient. Pass one to the constructor "
                "or call set_llm() before dreaming."
            )
        # Permit intentional in-place configuration changes only by treating
        # them as an explicit producer switch. The runner independently fences
        # against further drift during a provider request.
        self._phase1_generation = phase1_generation_binding(
            self.config.prompt_version, self._llm
        )
        if self._conn is not None:
            self._scope_phase1_reads(self._conn)
        if self._read_conn is not None:
            self._scope_phase1_reads(self._read_conn)
        ids = list(session_ids) if session_ids is not None else None
        try:
            report = run_dreaming(
                self.conn,
                self.config,
                self._llm,
                session_ids=ids,
                embedding_client=self._embed,
                deadline=deadline,
            )
        except DreamLeaseLost:
            # Earlier fenced units from this cycle, or the successor process,
            # may already have changed retrieval state before the handoff.
            self.invalidate_query_caches()
            raise
        # Dreaming may have added, retracted, or merged canonicals — invalidate
        # the token-overlap index so the next augment() rebuilds it.
        self._token_overlap_index = None
        return report

    def invalidate_query_caches(self) -> None:
        """Clear query-side caches (token-overlap index).

        Call after an *external* write to the DB — e.g. a forked HyMem instance
        completed a background dream cycle — so this instance's next augment()
        rebuilds the index from fresh state. In-process `dream()`,
        `merge_canonical()`, and `retract_edge()` already self-invalidate.
        """
        self._token_overlap_index = None

    def recent_dream_runs(self, limit: int = 20) -> list[dict]:
        """Return the last N dream_runs rows as dicts, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM dream_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def behavioral_duplicate_report(
        self, *, cosine_threshold: float | None = None
    ) -> dict:
        """Dry-run report of pre-existing behavioral edges (`prefers` / `avoids`
        / `rejects`) that would collapse if merged on semantic similarity alone.

        Read-only — writes nothing, makes no embedding-API call (it reuses cached
        `edge_embeddings` vectors). Surfaces the proliferation that predates
        same-wave collapse so an operator can decide whether a future apply step
        is worth running. `cosine_threshold` overrides
        `config.behavioral_dedup_cosine_threshold`; lower it to see more
        aggressive merges, raise it for only the closest paraphrases.

        Returns ``{cosine_threshold, clusters, edges_collapsed, merges}`` where
        each merge names the proposed survivor and the members that would fold
        into it (with each member's cosine to the survivor).
        """
        from hymem.dreaming import behavioral_dedup

        threshold = (
            cosine_threshold
            if cosine_threshold is not None
            else self.config.behavioral_dedup_cosine_threshold
        )
        proposals = behavioral_dedup.find_behavioral_duplicates(
            self.read_conn,
            cosine_threshold=threshold,
            embedding_client=self._embed,
        )
        return {
            "cosine_threshold": threshold,
            "clusters": len(proposals),
            "edges_collapsed": sum(p.collapses for p in proposals),
            "merges": [
                {
                    "subject": p.subject,
                    "predicate": p.predicate,
                    "survivor": {
                        "edge_id": p.survivor_id,
                        "object": p.survivor_object,
                        "pos_evidence": p.survivor_pos,
                        "neg_evidence": p.survivor_neg,
                    },
                    "members": [
                        {
                            "edge_id": m.edge_id,
                            "object": m.object,
                            "pos_evidence": m.pos_evidence,
                            "neg_evidence": m.neg_evidence,
                            "cosine_to_survivor": m.cosine_to_survivor,
                        }
                        for m in p.members
                    ],
                }
                for p in proposals
            ],
        }

    def apply_behavioral_merges(
        self, *, cosine_threshold: float | None = None
    ) -> dict:
        """Run the behavioral dedup dry-run report, then execute the merges.

        A convenience combining :meth:`behavioral_duplicate_report` and
        :func:`hymem.dreaming.behavioral_dedup.apply_behavioral_merges` into a
        single atomic call. Runs inside ``core_db.transaction()`` on the primary
        connection so failures roll back completely. Returns the merged summary
        dict from ``apply_behavioral_merges`` with the report appended so the
        caller can see what was done.

        After merging, the in-memory token-overlap index is invalidated (the
        graph changed) so the next :meth:`augment` rebuilds it.
        """
        from hymem.dreaming import behavioral_dedup

        threshold = (
            cosine_threshold
            if cosine_threshold is not None
            else self.config.behavioral_dedup_cosine_threshold
        )
        proposals = behavioral_dedup.find_behavioral_duplicates(
            self.read_conn,
            cosine_threshold=threshold,
            embedding_client=self._embed,
        )
        if not proposals:
            return {
                "clusters_merged": 0,
                "edges_retracted": 0,
                "survivors_updated": 0,
                "proposals_found": 0,
            }

        from hymem.core import db as core_db

        with core_db.transaction(self.conn):
            result = behavioral_dedup.apply_behavioral_merges(
                self.conn, proposals, embedding_client=self._embed,
            )

        # Invalidate so next augment rebuilds entity lookups from the slimmed graph.
        self._token_overlap_index = None

        result["cosine_threshold"] = threshold
        result["proposals_found"] = len(proposals)
        return result

    def dream_status(self) -> dict:
        """Return one coherent, read-only durable indexing-health snapshot."""
        return self.dream_status_with_snapshot(lambda _conn: {})

    def dream_status_with_snapshot(
        self,
        extension: Callable[[sqlite3.Connection], Mapping[str, object]],
    ) -> dict:
        """Extend :meth:`dream_status` inside its exact SQLite snapshot.

        Benchmark completion also audits embedding mirrors.  Letting that
        audit run here prevents a writer from advancing source state between
        two independently clean-looking status reads.  The callback is
        read-only, executes on a dedicated query-only connection, and may not
        shadow production dream-status fields.
        """
        if not callable(extension):
            raise TypeError("dream status snapshot extension must be callable")
        self.conn  # initialize/migrate the store before opening it read-only
        with core_db.read_snapshot(self.config.db_path) as conn:
            self._scope_phase1_reads(conn)
            status = self._dream_status_snapshot(conn)
            extra = extension(conn)
            if not isinstance(extra, Mapping):
                raise TypeError("dream status snapshot extension must return a mapping")
            overlap = set(status).intersection(extra)
            if overlap:
                raise ValueError("dream status snapshot extension shadows status fields")
            status.update(extra)
            return status

    def _dream_status_snapshot(self, conn: sqlite3.Connection) -> dict:
        """Operator-visibility snapshot of the dreaming/extraction backlog.

        Pure reads over a dedicated coherent SQLite snapshot — no LLM,
        embedding calls, or writes — so it works even when no LLM/embedding
        client is configured. Useful to explain
        the re-extraction surge that follows a `prompt_version` bump: when the
        version changes, every chunk is "pending" again and the next dream(s)
        reprocess the whole backlog, which can take minutes.

        Returns a dict with:
          - `pending_chunks`: actionable chunks still owed an extraction pass
            for the CURRENT prompt version (quarantined failures excluded).
          - `quarantined_chunks`: unprocessed chunks whose consecutive failure
            count reached the current retry bound. These are not completed and
            reopen when the prompt or retry policy changes.
          - `pending_source_materialization`: covered/raw session frontiers not
            yet acknowledged by the current salience+baseline chunk producers.
          - `pending_digests`, `pending_profiles`, and `pending_facts`: current
            lossless consumer cursors, rebuilds/replays, or local publication
            stamps still owed below their exact quarantine bounds.
          - `malformed_*`: structurally invalid cursor/retry/authority state.
            These counters are health blockers and never convert to success.
          - `terminal_loss_chunks`: chunks whose exact source manifest is
            irrecoverably absent. They are not successful and never reopen on
            a prompt change; `terminal_loss_reasons` provides the breakdown.
          - `coverage_integrity_failures`: sessions whose ordered lossless
            source could not be established under the current structural
            validator. This is non-pending unhealthy state, not work that a
            convergence loop should blindly spin on. Resolve a transient
            storage/validation failure or repair persistent artifact damage;
            the next complete good walk clears it. Bounded structural details
            name the affected session/config without source or exception text.
          - `quarantined_facts`: fact extraction units held under the CURRENT
            prompt/config/retry/cursor identity. Stale identities are reopened
            work and do not count. `quarantined_facts_malformed` separately
            counts flagged rows whose bounded retry, cursor, or fact authority
            state is inconsistent.
            Both are zero when fact extraction is disabled or its retry bound
            is zero.
          - `pending_aggregation`: 1 while the enabled material aggregation
            config has no clean local build acknowledgement, including after a
            partial fusion failure, total exception, crash, or portable source
            import. Disabled aggregation never reports actionable pending work;
            historical bounded counters remain visible for audit.
          - `total_chunks`: retrieval/extraction chunk count. Lossless coverage
            artifacts are durable source storage and do not enter this budget.
          - `prompt_version`: the current human `config.prompt_version`.
          - `extraction_contract`: its derived prompt/validator/recovery
            binding; `extraction_cache_key` is the exact Phase-1 DB namespace.
          - `phase1_generation`: the secret-free producer/effective-request
            binding currently owed. A configured model switch therefore makes
            old processed markers visibly pending instead of relabeling them.
          - `extraction_provider_attempt_budget`: configured soft Phase-1
            provider-request-attempt ceiling per cycle (0 means unlimited).
            Measured usage and whether it blocked pending work are exposed on
            `last_run`.
          - `in_progress`: True iff a `run_lock` row named 'dreaming' exists.
            This is intentionally coarse — a *stale* lock (e.g. from a crashed
            dream) reads as in_progress until it expires. Lock heartbeat/TTL is
            handled in the dreaming runner; this method does not reimplement it.
          - `last_run`: the most recent `dream_runs` row as a dict, or None if
            no dream has ever run.
        """
        from hymem.extraction.contract import extraction_cache_key

        pv = extraction_cache_key(self.config.prompt_version)
        generation = (
            phase1_generation_binding(self.config.prompt_version, self._llm)
            if self._llm is not None else None
        )
        generation_key = (
            str(generation["generation_key"]) if generation is not None else ""
        )

        retry_bound = int(self.config.chunk_extraction_max_attempts)
        pending_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks c "
            "WHERE c.chunk_kind = 'extraction' "
            "AND COALESCE(c.salience_reason, '') <> 'short_session_fallback' "
            "AND c.source_manifest_version='claim-source-manifest-v1' "
            "AND c.source_manifest_count>0 "
            "AND NOT EXISTS ("
            "    SELECT 1 FROM chunk_extraction_terminal_losses loss "
            "    WHERE loss.chunk_id=c.id"
            ") "
            "AND NOT EXISTS ("
            "    SELECT 1 FROM current_phase1_publications publication "
            "    WHERE publication.chunk_id=c.id "
            "      AND publication.prompt_version=? "
            "      AND publication.phase1_generation_key=?"
            ") "
            "AND (? <= 0 OR NOT EXISTS ("
            "    SELECT 1 FROM chunk_extraction_attempts a "
            "    WHERE a.chunk_id = c.id AND a.prompt_version = ? "
            "      AND a.phase1_generation_key=? "
            "      AND a.attempts >= ?"
            "))",
            (pv, generation_key, retry_bound, pv, generation_key, retry_bound),
        ).fetchone()[0]
        quarantined_chunks = (
            conn.execute(
                "SELECT COUNT(*) FROM chunks c "
                "WHERE c.chunk_kind = 'extraction' "
                "AND COALESCE(c.salience_reason, '') <> 'short_session_fallback' "
                "AND c.source_manifest_version='claim-source-manifest-v1' "
                "AND c.source_manifest_count>0 "
                "AND NOT EXISTS ("
                "    SELECT 1 FROM chunk_extraction_terminal_losses loss "
                "    WHERE loss.chunk_id=c.id"
                ") "
                "AND NOT EXISTS ("
                "    SELECT 1 FROM current_phase1_publications publication "
                "    WHERE publication.chunk_id=c.id "
                "      AND publication.prompt_version=? "
                "      AND publication.phase1_generation_key=?"
                ") "
                "AND EXISTS ("
                "    SELECT 1 FROM chunk_extraction_attempts a "
                "    WHERE a.chunk_id = c.id AND a.prompt_version = ? "
                "      AND a.phase1_generation_key=? "
                "      AND a.attempts >= ?"
                ")",
                (pv, generation_key, pv, generation_key, retry_bound),
            ).fetchone()[0]
            if retry_bound > 0
            else 0
        )
        terminal_loss_rows = conn.execute(
            "SELECT reason,COUNT(*) AS n "
            "FROM chunk_extraction_terminal_losses GROUP BY reason ORDER BY reason"
        ).fetchall()
        terminal_loss_reasons = {
            str(row["reason"]): int(row["n"]) for row in terminal_loss_rows
        }
        terminal_loss_chunks = sum(terminal_loss_reasons.values())
        coverage_failure_rows = conn.execute(
            "SELECT failure_reason,COUNT(*) AS n "
            "FROM coverage_integrity_failures "
            "GROUP BY failure_reason ORDER BY failure_reason"
        ).fetchall()
        coverage_integrity_failure_reasons = {
            str(row["failure_reason"]): int(row["n"])
            for row in coverage_failure_rows
        }
        coverage_integrity_failures = sum(
            coverage_integrity_failure_reasons.values()
        )
        coverage_detail_limit = 100
        coverage_detail_rows = conn.execute(
            "SELECT session_id,config_version,failure_reason,occurrences,"
            "first_detected_at,last_detected_at "
            "FROM coverage_integrity_failures "
            "ORDER BY last_detected_at DESC,session_id LIMIT ?",
            (coverage_detail_limit + 1,),
        ).fetchall()
        coverage_integrity_failure_details = [
            dict(row) for row in coverage_detail_rows[:coverage_detail_limit]
        ]
        total_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_kind = 'extraction'"
        ).fetchone()[0]
        in_progress = (
            conn.execute(
                "SELECT 1 FROM run_lock WHERE name = 'dreaming' LIMIT 1"
            ).fetchone()
            is not None
        )
        last_row = conn.execute(
            "SELECT * FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

        aggregation_status = aggregation_health_status(
            conn,
            enabled=self.config.aggregation_nodes_enabled,
            config_version=(
                aggregation_config_version(self.config)
                if self.config.aggregation_nodes_enabled else None
            ),
            generation_key=(
                self._current_aggregation_generation_key()
                if self.config.aggregation_nodes_enabled else None
            ),
            embedding_client=self._embed,
        )
        work_status = durable_dream_work_status(conn, self.config)

        return {
            **work_status,
            "pending_chunks": pending_chunks,
            "quarantined_chunks": quarantined_chunks,
            "terminal_loss_chunks": terminal_loss_chunks,
            "terminal_loss_reasons": terminal_loss_reasons,
            "coverage_integrity_failures": coverage_integrity_failures,
            "coverage_integrity_failure_reasons": (
                coverage_integrity_failure_reasons
            ),
            "coverage_integrity_failure_details": (
                coverage_integrity_failure_details
            ),
            "coverage_integrity_failure_details_truncated": (
                len(coverage_detail_rows) > coverage_detail_limit
            ),
            "coverage_integrity_config_version": (
                COVERAGE_INTEGRITY_CONFIG_VERSION
            ),
            "total_chunks": total_chunks,
            "prompt_version": self.config.prompt_version,
            "extraction_contract": dict(self.config.extraction_contract),
            "extraction_cache_key": pv,
            "phase1_generation": generation,
            "phase1_generation_key": (
                generation_key if generation is not None else None
            ),
            "phase1_backlog_status": (
                "current_producer" if generation is not None
                else "producer_unavailable"
            ),
            "pending_chunks_authoritative": generation is not None,
            "extraction_provider_attempt_budget": (
                self.config.dream_extraction_provider_attempt_budget
            ),
            "in_progress": in_progress,
            "last_run": dict(last_row) if last_row is not None else None,
            **aggregation_status,
        }

    # ---- portability -------------------------------------------------

    def export(self, path: str | Path) -> dict[str, int]:
        """Write the canonical state (sessions, chunks, edges, episodes,
        procedures, profile entries) to `path` as JSON Lines. Returns per-kind
        row counts. Useful for backups, project migration, and external
        inspection. Read-only.
        """
        return portability.export_jsonl(self.conn, path)

    def import_(self, path: str | Path) -> dict[str, int]:
        """Load a JSON Lines export written by `export()`.

        Disjoint identities merge additively and exact reimports are no-ops;
        a conflicting deterministic identity aborts the complete import.
        Returns per-kind inserted-row counts and invalidates query caches.
        """
        result = portability.import_jsonl(
            self.conn, path, redact_values=self.config.redact_secrets,
            config=self.config,
        )
        self._token_overlap_index = None
        return result

    # ---- maintenance -------------------------------------------------

    def register_alias(self, surface: str, canonical: str) -> None:
        with core_db.transaction(self.conn):
            canon.register_alias(self.conn, surface, canonical)

    def merge_canonical(self, keep: str, drop: str) -> None:
        with core_db.transaction(self.conn):
            canon.merge(self.conn, keep, drop)
            self.conn.execute(
                "DELETE FROM token_overlap_index WHERE canonical = ?", (drop,)
            )
        self._token_overlap_index = None

    def retract_edge(self, subject: str, predicate: str, object: str) -> bool:
        """Mark an edge as retracted. Idempotent. Returns True if an edge was found and updated, False otherwise.

        Subjects/objects are normalized through the alias table — pass the surface
        form (e.g., 'MedFlow') and HyMem resolves to the canonical id.

        Only acts on edges with status='active'; calling again on an already
        retracted edge returns False.
        """
        with core_db.transaction(self.conn):
            subj = canon.resolve(self.conn, subject)
            obj = canon.resolve(self.conn, object)
            row = self.conn.execute(
                f"SELECT id FROM knowledge_graph "
                "WHERE subject_canonical = ? AND predicate = ? AND object_canonical = ? "
                f"AND {live_edge_predicate()}",
                (subj, predicate, obj),
            ).fetchone()
            if row is None:
                return False
            evidence_ledger.record_signal(
                self.conn,
                edge_id=row["id"],
                signal_key=f"manual-retraction:{uuid.uuid4().hex}",
                signal_kind="manual_retraction",
                polarity=-1,
                evidence_weight=1,
                details=f"HyMem.retract_edge({subj}, {predicate}, {obj})",
            )
            # Retain source-linked correction evidence for bounded audit.
            # Audit values never flow back into an extraction prompt.
            evidence_rows = self.conn.execute(
                f"""SELECT chunk_id FROM kg_evidence
                   WHERE edge_id = ? AND polarity = 1 AND is_current = 1
                   ORDER BY {graph_clock_order_sql('extracted_at')}, id
                   LIMIT 5""",
                (row["id"],),
            ).fetchall()
            for er in evidence_rows:
                chunk_row = self.conn.execute(
                    "SELECT text FROM chunks WHERE id = ?", (er["chunk_id"],)
                ).fetchone()
                if chunk_row:
                    snippet = chunk_row["text"][:600]
                    self.conn.execute(
                        """INSERT OR IGNORE INTO extraction_feedback
                           (chunk_id, chunk_text_snippet, extracted_subject,
                            extracted_predicate, extracted_object, feedback_type)
                           VALUES (?, ?, ?, ?, ?, 'retracted')""",
                        (er["chunk_id"], snippet, subject, predicate, object),
                    )
            for c in (subj, obj):
                still_active = self.conn.execute(
                    f"SELECT 1 FROM knowledge_graph "
                    "WHERE (subject_canonical = ? OR object_canonical = ?) "
                    f"AND {live_edge_predicate()} LIMIT 1",
                    (c, c),
                ).fetchone()
                if still_active is None:
                    self.conn.execute(
                        "DELETE FROM token_overlap_index WHERE canonical = ?", (c,)
                    )
        self._token_overlap_index = None
        return True

    def mark_procedure_stale(self, procedure_id: str) -> bool:
        """Flag a procedure as wrong / outdated. Idempotent. Returns True if an
        active procedure was found and updated, False otherwise.

        Symmetric to `retract_edge`, but for procedural memory: when Hermes
        surfaces a procedure via `augment()` (a `ProcedureHit`) and the user
        marks it stale, the row is flipped to status='stale' — which removes it
        from future `_procedure_search` results — and its `confidence` is
        knocked down by `cfg.procedure_stale_confidence_factor` so the negative
        signal survives even if the procedure is later re-extracted.

        Only acts on procedures with status='active'; calling again on an
        already-stale procedure returns False.
        """
        with core_db.transaction(self.conn):
            row = self.conn.execute(
                "SELECT id FROM procedures WHERE id = ? AND status = 'active'",
                (procedure_id,),
            ).fetchone()
            if row is None:
                return False
            self.conn.execute(
                "UPDATE procedures "
                "SET status = 'stale', confidence = confidence * ? "
                "WHERE id = ?",
                (self.config.procedure_stale_confidence_factor, procedure_id),
            )
        return True
