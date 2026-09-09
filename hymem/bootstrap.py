"""Zero-config startup: build a HyMem instance from environment variables.

This is the single source of truth for environment-variable resolution. Both
entry points (`hymem-server`, `hymem-honcho`) and `hymem-doctor` build on it,
so configuration behaviour stays consistent across every surface.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from hymem.api import HyMem
from hymem.config import HyMemConfig
from hymem.contrib.endpoint_policy import (
    EMBEDDING_INTERNAL_HTTP_ENV,
    EndpointPolicyError,
    _reject_credential_shaped_endpoint_host,
    _reject_credential_shaped_endpoint_path,
    resolve_embedding_api_key,
    resolve_llm_api_key,
    validate_http_endpoint,
)

log = logging.getLogger("hymem.bootstrap")

DEFAULT_ROOT = Path.home() / ".hermes"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"  # deepseek-chat hard-deprecated 2026-07-24
DEFAULT_EMBEDDING_BASE_URL = "local://feature-hash"
DEFAULT_EMBEDDING_MODEL = "hymem-local-feature-hash-v1"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_REMOTE_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_REMOTE_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_REMOTE_EMBEDDING_DIM = 1536


@dataclass(frozen=True)
class EnvConfig:
    """Environment-resolved configuration. ``*_api_key`` is None when absent."""

    root: Path
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    embedding_api_key: str | None
    embedding_base_url: str
    embedding_model: str
    embedding_dim: int
    embedding_backend: str
    embedding_fallback_reason: str | None
    # None = env var unset → fall back to the HyMemConfig dataclass default
    # (don't hard-code it here, so a future default change stays authoritative).
    aggregation_nodes_enabled: bool | None
    aggregation_digest_enabled: bool | None
    # Durable remote-vector reuse is opt-in.  Defaults preserve source
    # compatibility for callers which construct EnvConfig directly while the
    # environment resolver always fills these fields explicitly.
    embedding_pin_dimension: bool = False
    embedding_deployment_revision: str | None = None
    embedding_deployment_tenant: str | None = None
    # Snapshot of the rejected endpoint's safe policy explanation. Do not
    # reconstruct it from later ambient environment or retain the rejected URL.
    embedding_fallback_detail: str | None = None

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def has_embedding_key(self) -> bool:
        return bool(self.embedding_api_key)

    @property
    def has_embedding_client(self) -> bool:
        return self.embedding_fallback_reason is None and (
            self.embedding_backend == "local_feature_hash" or self.has_embedding_key
        )

    @property
    def embedding_identity(self) -> str:
        """Exact durable vector-space identity expected from this config."""
        if self.embedding_backend != "openai_compatible":
            return self.embedding_model
        from hymem.contrib.openai_embedding_client import (
            openai_compatible_embedding_identity,
        )
        return openai_compatible_embedding_identity(
            self.embedding_base_url, self.embedding_model
        )


_OWNERSHIP_ATTR = "_hymem_bootstrap_ownership"


@dataclass
class _BootstrapOwnership:
    """Lifecycle state attached only to instances built by this module."""

    transports: tuple[object, ...]
    lock: threading.RLock = field(default_factory=threading.RLock)
    store_close_started: bool = False
    attempted_resource_ids: set[int] = field(default_factory=set)
    finished: bool = False


def _resource_identity_tree(resource: object) -> set[int]:
    """Return wrapper + transport identities closed by one close invocation."""

    from hymem.extraction.embeddings import CachedEmbeddingClient

    identities: set[int] = set()
    pending = [resource]
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in identities:
            continue
        identities.add(identity)
        if isinstance(current, CachedEmbeddingClient):
            pending.append(current._inner)
    return identities


def _attempt_transport_closes(
    resources: Iterable[tuple[str, object | None]],
    *,
    attempted_resource_ids: set[int],
) -> list[tuple[str, BaseException]]:
    """Close distinct owned transports once, continuing after failures."""

    failures: list[tuple[str, BaseException]] = []
    for label, resource in resources:
        if resource is None:
            continue
        identities = _resource_identity_tree(resource)
        if identities & attempted_resource_ids:
            continue
        # Claim the wrapper and every transport to which its close delegates
        # before invoking user/provider code. A partial close is not retryable.
        attempted_resource_ids.update(identities)
        try:
            close = getattr(resource, "close", None)
        except BaseException as exc:
            failures.append((label, exc))
            continue
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            failures.append((label, exc))
    return failures


def _add_cleanup_note(
    primary: BaseException, *, label: str, cleanup: BaseException
) -> None:
    """Annotate a primary fault without leaking provider-controlled text."""

    try:
        primary.add_note(f"{label} cleanup failed: {type(cleanup).__name__}")
    except (AttributeError, TypeError):  # pragma: no cover - pre-3.11/custom
        pass


def _cleanup_failed_build(
    primary: BaseException,
    *,
    instance: HyMem | None,
    embedder: object | None,
    embedding_transport: object | None,
    llm: object | None,
) -> None:
    """Best-effort rollback which never replaces the construction failure."""

    if instance is not None:
        try:
            instance.close()
        except BaseException as cleanup:
            _add_cleanup_note(primary, label="HyMem store", cleanup=cleanup)
    attempted: set[int] = set()
    for label, cleanup in _attempt_transport_closes(
        (
            ("embedding transport", embedder),
            ("embedding transport", embedding_transport),
            ("LLM transport", llm),
        ),
        attempted_resource_ids=attempted,
    ):
        _add_cleanup_note(primary, label=label, cleanup=cleanup)


def _env_flag(name: str) -> bool | None:
    """Parse a boolean env var, or None when unset.

    None lets the caller defer to the dataclass default instead of forcing a
    value. Truthy set: 1/true/yes/on (case-insensitive); anything else is False.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _embedding_endpoint_rejection_detail(
    base_url: str, error: EndpointPolicyError,
) -> str:
    """Retain bounded policy text, never endpoint/key/flag values.

    EndpointPolicyError messages from these maintained policy functions use
    constant labels and never interpolate rejected input. Qualify the internal
    HTTP hint against the complete route: transport rejection happens before
    credential-shaped host/path validation, and enabling HTTP must not be
    recommended as a remedy for such a route.
    """
    detail = str(error)
    if f"; set {EMBEDDING_INTERNAL_HTTP_ENV}=1" in detail:
        try:
            # Inert diagnostic validation only. This explicit mapping neither
            # changes process environment nor authorizes a key, client or call.
            endpoint = validate_http_endpoint(
                base_url,
                label="embedding",
                allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
                environ={EMBEDDING_INTERNAL_HTTP_ENV: "1"},
            )
            _reject_credential_shaped_endpoint_host(endpoint)
            _reject_credential_shaped_endpoint_path(endpoint)
        except EndpointPolicyError as unsafe_route:
            detail = str(unsafe_route)
    return detail[:512]


def resolve_env() -> EnvConfig:
    """Resolve all HyMem configuration from the environment.

    Never raises and never constructs network clients — safe for the doctor
    to report the candidate configuration and why startup would be refused.
    """
    env = os.environ.get
    llm_base_url = env("HYMEM_LLM_BASE_URL", DEFAULT_BASE_URL)
    # Resolve provider fallbacks while the endpoint is still known.  Passing a
    # generic pre-resolved OPENAI/DEEPSEEK key to the client as though it were
    # constructor-explicit would bypass the client's origin binding.
    try:
        _llm_endpoint, llm_key = resolve_llm_api_key(llm_base_url)
    except (EndpointPolicyError, EnvironmentError):
        llm_key = None
    explicit_embedding = any(
        env(name)
        for name in (
            "HYMEM_EMBEDDING_API_KEY", "HYMEM_EMBEDDING_BASE_URL",
            "HYMEM_EMBEDDING_MODEL", "HYMEM_EMBEDDING_DIM",
            "HYMEM_EMBEDDING_PIN_DIMENSION",
            "HYMEM_EMBEDDING_DEPLOYMENT_REVISION",
            "HYMEM_EMBEDDING_DEPLOYMENT_TENANT",
        )
    )
    embedding_key: str | None = None
    embedding_fallback_reason: str | None = None
    embedding_fallback_detail: str | None = None
    if explicit_embedding:
        requested_base = env(
            "HYMEM_EMBEDDING_BASE_URL", DEFAULT_REMOTE_EMBEDDING_BASE_URL
        )
        requested_model = env(
            "HYMEM_EMBEDDING_MODEL", DEFAULT_REMOTE_EMBEDDING_MODEL
        )
        requested_dim = _env_positive_int(
            "HYMEM_EMBEDDING_DIM", DEFAULT_REMOTE_EMBEDDING_DIM
        )
        try:
            _embedding_endpoint, embedding_key = resolve_embedding_api_key(
                requested_base
            )
        except EndpointPolicyError as exc:
            embedding_fallback_reason = "remote_embedding_endpoint_rejected"
            embedding_fallback_detail = _embedding_endpoint_rejection_detail(
                requested_base, exc,
            )
        except EnvironmentError:
            embedding_fallback_reason = "remote_embedding_credentials_missing"
        if embedding_key:
            embedding_base_url = requested_base
            embedding_model = requested_model
            embedding_dim = requested_dim
            embedding_backend = "openai_compatible"
        else:
            # Retain the legacy inert diagnostic fields, not the rejected URL
            # or credentials. A fallback_reason makes this configuration
            # unavailable: build_from_env must refuse it, never activate these
            # local defaults against a store configured for a remote producer.
            embedding_base_url = DEFAULT_EMBEDDING_BASE_URL
            embedding_model = DEFAULT_EMBEDDING_MODEL
            embedding_dim = DEFAULT_EMBEDDING_DIM
            embedding_backend = "local_feature_hash"
            if embedding_fallback_reason is None:
                embedding_fallback_reason = "remote_embedding_credentials_missing"
    else:
        # In particular, a DeepSeek-only LLM environment lands here. DeepSeek
        # does not expose the old fictional `deepseek-embedding` model, so its
        # key is never silently reused for embeddings.
        embedding_base_url = DEFAULT_EMBEDDING_BASE_URL
        embedding_model = DEFAULT_EMBEDDING_MODEL
        embedding_dim = DEFAULT_EMBEDDING_DIM
        embedding_backend = "local_feature_hash"
    return EnvConfig(
        root=Path(env("HYMEM_ROOT", str(DEFAULT_ROOT))),
        llm_api_key=llm_key,
        llm_base_url=llm_base_url,
        llm_model=env("HYMEM_LLM_MODEL", DEFAULT_LLM_MODEL),
        embedding_api_key=embedding_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_pin_dimension=bool(
            _env_flag("HYMEM_EMBEDDING_PIN_DIMENSION")
        ),
        embedding_deployment_revision=env(
            "HYMEM_EMBEDDING_DEPLOYMENT_REVISION"
        ),
        embedding_deployment_tenant=env(
            "HYMEM_EMBEDDING_DEPLOYMENT_TENANT"
        ),
        embedding_backend=embedding_backend,
        embedding_fallback_reason=embedding_fallback_reason,
        embedding_fallback_detail=embedding_fallback_detail,
        aggregation_nodes_enabled=_env_flag("HYMEM_AGGREGATION_NODES_ENABLED"),
        aggregation_digest_enabled=_env_flag("HYMEM_AGGREGATION_DIGEST_ENABLED"),
    )


def _default_local_store_error(cfg: EnvConfig) -> str | None:
    """Refuse implicit local startup over another or unknown durable space.

    This is a bounded, read-only *admission* check, not a migration, vector
    integrity audit or global writer lock. A committed exact local shadow
    identity also permits historical incompatible mirrors/cache entries: an
    intentional ``hymem-reembed --allow-local`` repair need not destroy them.
    Without shadow identity, populated mirrors/cache are the only surviving
    producer evidence and must all agree with the candidate local space.

    Direct ``HyMem(..., embedding_client=...)`` construction and the explicit
    re-embed workflow do not use this environment-bootstrap guard.
    """
    if cfg.embedding_backend != "local_feature_hash":
        return None
    conn: sqlite3.Connection | None = None
    conflict = False
    unavailable = False
    try:
        path = HyMemConfig(root=cfg.root).db_path.absolute()
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(mode):
            raise ValueError("existing store is not a regular file")

        from hymem.dreaming.aggregation_material import embedding_storage_identity
        from hymem.extraction.embeddings import LocalHashEmbeddingClient

        local_model, local_dim = embedding_storage_identity(LocalHashEmbeddingClient(
            dim_value=cfg.embedding_dim, model_name=cfg.embedding_model,
        ))
        # Never use core_db.connect/initialize here: even an unsuccessful
        # diagnostic must not initialize, migrate or heal an existing store.
        # mode=ro (not immutable) observes committed WAL state. SQLite owns any
        # required read locks/sidecars; read-only opening can create coordination
        # sidecars but does not mutate source rows or initialize/migrate the DB.
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1.0)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA trusted_schema=OFF")
        deadline = time.monotonic() + 2.0
        remaining_steps = 2_000_000

        def bounded_read() -> int:
            nonlocal remaining_steps
            remaining_steps -= 1_000
            return int(remaining_steps < 0 or time.monotonic() >= deadline)

        conn.set_progress_handler(bounded_read, 1_000)
        conn.execute("BEGIN")
        mirror_tables = (
            "chunk_embeddings", "message_embeddings", "edge_embeddings",
            "episode_embeddings", "narrative_fact_embeddings",
            "aggregation_node_embeddings", "embedding_cache",
        )
        names = ("schema_meta", *mirror_tables)
        objects = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE name IN ("
            + ",".join("?" for _ in names)
            + ") OR name GLOB 'vec_*' LIMIT 65", names,
        ).fetchall()
        if len(objects) > 64 or any(kind != "table" for _, kind in objects):
            raise ValueError("unrecognized vector storage layout")
        tables = {name for name, _ in objects}
        metadata = []
        if "schema_meta" in tables:
            metadata = conn.execute(
                "SELECT key, value FROM schema_meta "
                "WHERE key IN ('vec_model', 'vec_dim') LIMIT 3"
            ).fetchall()
        if metadata:
            values = dict(metadata)
            # Partial, duplicate, malformed and unknown identities cannot
            # authorize a default producer switch, even on empty shadows.
            conflict = not (
                len(metadata) == len(values) == 2
                and values.get("vec_model") == local_model
                and type(values.get("vec_dim")) in (str, int)
                and values.get("vec_dim") in (str(local_dim), local_dim)
            )
        elif any(name.startswith("vec_") for name in tables):
            conflict = True
        else:
            for table in mirror_tables:
                if table not in tables:
                    continue
                # Empty pre-vector legacy tables need not possess today's
                # columns. A populated uninspectable table fails closed.
                if conn.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone() is None:
                    continue
                typed_conflict = ""
                if table in ("episode_embeddings", "aggregation_node_embeddings"):
                    columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
                    if "embedding_producer_key" in columns:
                        typed_conflict = " OR embedding_producer_key COLLATE BINARY IS NOT model"
                if conn.execute(
                    f'SELECT 1 FROM "{table}" WHERE '
                    "typeof(model) IS NOT 'text' OR model COLLATE BINARY IS NOT ? "
                    "OR typeof(dim) IS NOT 'integer' OR dim IS NOT ?"
                    + typed_conflict + " LIMIT 1",
                    (local_model, local_dim),
                ).fetchone() is not None:
                    conflict = True
                    break
        if time.monotonic() >= deadline:
            raise TimeoutError("local embedding admission deadline")
    except Exception:  # noqa: BLE001 - do not expose stored values/paths/errors
        unavailable = True
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - a failed close cannot admit startup
                unavailable = True
    if unavailable or conflict:
        reason = (
            "existing vector producer state could not be safely verified"
            if unavailable else
            "existing vector producer state does not match default local embeddings"
        )
        return (
            f"HyMem cannot start: {reason}. Restore the service's embedding "
            "configuration before starting it. An intentional producer change "
            "requires the maintained `hymem-reembed` workflow "
            "(`--allow-local` for local apply); no vectors were changed by "
            "this check. Run `hymem-doctor` to diagnose your configuration."
        )
    return None


def _embedding_configuration_error(cfg: EnvConfig) -> str | None:
    """Safe resolution/admission failure shared by startup and the doctor."""
    if cfg.embedding_fallback_reason is None:
        return _default_local_store_error(cfg)
    if cfg.embedding_fallback_reason == "remote_embedding_endpoint_rejected":
        reason = "remote_embedding_endpoint_rejected: configured remote embedding endpoint rejected"
        if cfg.embedding_fallback_detail:
            reason += f": {cfg.embedding_fallback_detail}"
    elif cfg.embedding_fallback_reason == "remote_embedding_credentials_missing":
        reason = (
            "remote_embedding_credentials_missing: set HYMEM_EMBEDDING_API_KEY "
            "for the configured endpoint, or use a provider key only with its "
            "exact official HTTPS endpoint"
        )
    else:
        # Do not echo arbitrary caller-supplied reason strings.
        reason = "configured remote embeddings are unavailable"
    return (
        f"HyMem cannot start: {reason}. "
        "No local embedding fallback will be installed. "
        "Run `hymem-doctor` to diagnose your configuration."
    )


def build_from_env() -> HyMem:
    """Construct a HyMem instance from environment variables.

    Fails fast with a clear, actionable error if the extraction LLM key is
    missing — instead of raising deep inside the first dream cycle. Embeddings
    default to a deterministic dependency-free local feature hash only without
    explicit remote configuration and without an incompatible existing store.
    A rejected or unavailable configured remote client fails before opening
    the store; it never switches the vector space.
    """
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
    from hymem.contrib.model_policy import require_active_model

    cfg = resolve_env()

    # Surface a retired deployment override as the primary startup fault,
    # before constructing clients or opening the store.  The library client
    # repeats the same central check so direct callers cannot bypass it.
    require_active_model(cfg.llm_model, role="HyMem server LLM")

    if not cfg.has_llm_key:
        raise RuntimeError(
            "HyMem cannot start: no extraction LLM API key found.\n"
            "Set HYMEM_LLM_API_KEY, or use the provider key only with its "
            "exact official HTTPS endpoint, "
            "before launching the server.\n"
            "Run `hymem-doctor` to diagnose your configuration."
        )

    embedding_configuration_error = _embedding_configuration_error(cfg)
    if embedding_configuration_error is not None:
        raise RuntimeError(embedding_configuration_error)

    from hymem.extraction.embeddings import (
        CachedEmbeddingClient,
        LocalHashEmbeddingClient,
    )

    llm: object | None = None
    embedding_transport: object | None = None
    embedder: object | None = None
    instance: HyMem | None = None
    try:
        if cfg.embedding_backend == "openai_compatible" and (
            not cfg.embedding_pin_dimension
            or not cfg.embedding_deployment_revision
            or not cfg.embedding_deployment_tenant
        ):
            raise RuntimeError(
                "remote embeddings require HYMEM_EMBEDDING_PIN_DIMENSION=1, "
                "HYMEM_EMBEDDING_DEPLOYMENT_REVISION, and "
                "HYMEM_EMBEDDING_DEPLOYMENT_TENANT so durable vectors have "
                "one exact reusable producer identity"
            )
        llm = OpenAICompatibleClient(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
        )

        if cfg.embedding_backend == "openai_compatible":
            embedding_construction_failed = False
            try:
                embedding_transport = OpenAICompatibleEmbeddingClient(
                    api_key=cfg.embedding_api_key,
                    base_url=cfg.embedding_base_url,
                    model=cfg.embedding_model,
                    dim=cfg.embedding_dim,
                    pin_dimension=cfg.embedding_pin_dimension,
                    deployment_revision=cfg.embedding_deployment_revision,
                    deployment_tenant=cfg.embedding_deployment_tenant,
                )
            except Exception:  # noqa: BLE001 - redact provider-controlled faults
                embedding_construction_failed = True
            if embedding_construction_failed:
                # Raise outside the handler so even the implicit exception
                # context cannot retain provider-controlled credentials/URLs.
                # The outer rollback still closes the already-created LLM.
                raise RuntimeError(
                    "HyMem cannot start: configured remote embedding client "
                    "could not be initialized. Check the embedding configuration "
                    "and installed client dependencies; run `hymem-doctor`. "
                    "No local embedding fallback will be installed."
                )
        else:
            embedding_transport = LocalHashEmbeddingClient(
                dim_value=cfg.embedding_dim,
                model_name=cfg.embedding_model,
                fallback_reason=cfg.embedding_fallback_reason,
            )
            log.info(
                "embeddings backend=%s model=%s dim=%d quality=lexical network=none%s",
                cfg.embedding_backend, cfg.embedding_model, cfg.embedding_dim,
                (
                    f" fallback_reason={cfg.embedding_fallback_reason}"
                    if cfg.embedding_fallback_reason else ""
                ),
            )
        embedder = CachedEmbeddingClient(embedding_transport)

        # Only env vars that were actually set override the dataclass defaults,
        # keeping the HyMemConfig dataclass as the authoritative default.
        overrides: dict[str, object] = {}
        if cfg.aggregation_nodes_enabled is not None:
            overrides["aggregation_nodes_enabled"] = cfg.aggregation_nodes_enabled
        if cfg.aggregation_digest_enabled is not None:
            overrides["aggregation_digest_enabled"] = cfg.aggregation_digest_enabled

        mem_cfg = HyMemConfig(root=cfg.root, **overrides)
        if mem_cfg.aggregation_nodes_enabled:
            log.info(
                "aggregation layer enabled (digest=%s) — dream will report "
                "aggregation_nodes_built/reused; watch the dream.end log line",
                mem_cfg.aggregation_digest_enabled,
            )

        instance = HyMem(mem_cfg, llm=llm, embedding_client=embedder)
        # Preserve eager store initialization/migration validation without
        # mutating another process's live dreaming lease. Stale leases are
        # reclaimed atomically by the runner after their TTL.
        _ = instance.conn
        setattr(
            instance,
            _OWNERSHIP_ATTR,
            _BootstrapOwnership(transports=(embedder, llm)),
        )
        return instance
    except BaseException as primary:
        _cleanup_failed_build(
            primary,
            instance=instance,
            embedder=embedder,
            embedding_transport=embedding_transport,
            llm=llm,
        )
        raise

# ── shared singleton ─────────────────────────────────────────────────────────
# Both server entry points and the test/integration harness go through these,
# so there is exactly one HyMem instance per process unless explicitly injected.

_instance: HyMem | None = None
_instance_lock = threading.RLock()


def get_instance() -> HyMem:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = build_from_env()
        return _instance


def set_instance(instance: HyMem) -> None:
    """Inject a pre-built, caller-owned HyMem.

    An ordinary direct ``HyMem(...)`` carries no bootstrap ownership marker,
    so server lifecycle cleanup stops its own background work but does not
    close the injected store or provider clients.
    """
    global _instance
    with _instance_lock:
        if _instance is not None and _instance is not instance:
            existing_ownership = getattr(_instance, _OWNERSHIP_ATTR, None)
            if (
                isinstance(existing_ownership, _BootstrapOwnership)
                and not existing_ownership.finished
            ):
                raise RuntimeError(
                    "cannot replace a live bootstrap-owned HyMem instance; "
                    "call shutdown_instance() first"
                )
        _instance = instance


def shutdown_instance(
    instance: HyMem | None = None,
    *,
    stop_background: Callable[[], None] | None = None,
) -> bool:
    """Quiesce and close one bootstrap-owned HyMem instance.

    ``stop_background`` runs first for the process singleton (or when no other
    singleton is installed) and must return only after every worker has stopped
    using the store and providers. If it fails, including on a scheduler join
    timeout, no dependent resource is closed and a later call can retry. It is
    deliberately skipped when explicitly closing a non-singleton while a
    different process singleton is installed.

    Instances created directly by callers are intentionally not owned here.
    For those, the optional background callback still runs, but their store and
    injected clients remain untouched.  Returns ``True`` only when this call
    performed an owned-resource shutdown.
    """

    global _instance
    failures: list[tuple[str, BaseException]] = []
    performed = False
    # Keep the singleton unavailable to concurrent get/set callers throughout
    # teardown. A getter blocks, then constructs a fresh instance only after
    # the old singleton has been fully cleared.
    with _instance_lock:
        target = instance if instance is not None else _instance
        if target is None:
            if stop_background is not None:
                stop_background()
            return False
        # A callback supplied for an explicitly targeted, non-singleton build
        # must not stop background work belonging to a different singleton.
        may_stop_background = _instance is None or _instance is target

        ownership = getattr(target, _OWNERSHIP_ATTR, None)
        if not isinstance(ownership, _BootstrapOwnership):
            if stop_background is not None and may_stop_background:
                stop_background()
            return False

        with ownership.lock:
            if ownership.finished:
                if stop_background is not None and may_stop_background:
                    stop_background()
                if _instance is target:
                    _instance = None
                return False

            # Quiescence is a hard precondition. In particular
            # DreamScheduler.stop raises while its worker is alive, leaving
            # this ownership state wholly retryable and preventing
            # use-after-close.
            if stop_background is not None and may_stop_background:
                stop_background()

            if not ownership.store_close_started:
                ownership.store_close_started = True
                try:
                    target.close()
                except BaseException as exc:
                    failures.append(("HyMem store", exc))
            failures.extend(
                _attempt_transport_closes(
                    (
                        ("embedding transport", ownership.transports[0]),
                        ("LLM transport", ownership.transports[1]),
                    ),
                    attempted_resource_ids=ownership.attempted_resource_ids,
                )
            )
            ownership.finished = True
            performed = True
        if _instance is target:
            _instance = None

    if failures:
        first_label, primary = failures[0]
        for label, cleanup in failures[1:]:
            _add_cleanup_note(primary, label=label, cleanup=cleanup)
        # Give the leading phase a bounded label too when a secondary exists;
        # never include exception messages, endpoint strings, or credentials.
        if len(failures) > 1:
            try:
                primary.add_note(f"first lifecycle failure: {first_label}")
            except (AttributeError, TypeError):  # pragma: no cover
                pass
        raise primary
    return performed
