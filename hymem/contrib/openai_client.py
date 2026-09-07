from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
import json
import math
import os
import re
import threading
import time
import weakref
from typing import Callable, NamedTuple

from hymem.contrib.endpoint_policy import (
    resolve_llm_api_key,
    validate_public_attestation,
)
from hymem.contrib.openai_embedding_client import (
    OPENAI_TRANSPORT_RUNTIME_EXACT,
    OPENAI_TRANSPORT_RUNTIME_VERSIONS,
)
from hymem.contrib.model_policy import require_active_model
from hymem.deadline import DeadlineExceeded, current_deadline

from hymem.extraction.llm import (
    LLMClient,
    LLMRequest,
    ProviderAttemptTracker,
)
from hymem.extraction.retry import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BASE_DELAY_SECONDS,
    DEFAULT_RETRY_MAX_DELAY_SECONDS,
    with_retry,
)
from hymem.extraction import retry as retry_module

# Accepted values for the thinking override (env var / constructor argument).
# "disabled" force-sends the vendor body key, "off"/"enabled" force-omit it.
_THINKING_MODES = {"auto", "disabled", "off", "enabled"}

# Keep the SDK transport bounded independently of our explicit retry wrapper.
# A finite 120-second attempt matches the raw benchmark clients and is long
# enough for reasoning-capable OpenAI-compatible endpoints without inheriting
# the OpenAI SDK's multi-minute default.
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0
_OPENAI_LLM_EXECUTION_ATTRIBUTES = (
    "complete", "_complete_with_execution_lease",
    "phase1_producer_declaration",
    "aggregation_producer_declaration", "track_provider_attempts",
    "effective_extra_body", "close", "transport_integrity_ok",
    "model", "base_url", "thinking_mode", "transport_package_version",
    "count_tokens",
    "_verify_transport_integrity", "_transport_state",
    "_canonical_transport_value", "_http_transport_state",
    "__getattribute__", "__getattr__",
)
_OPENAI_LLM_ORIGINAL_IMPLEMENTATION_GUARD: tuple[object, ...] | None = None
_OPENAI_LLM_ORIGINAL_HELPER_GUARD: tuple[object, ...] | None = None
_OPENAI_LLM_CLOSE_TARGETS: weakref.WeakKeyDictionary[object, object] = (
    weakref.WeakKeyDictionary()
)
_OPENAI_LLM_CLOSE_TARGETS_LOCK = threading.Lock()
_OPENAI_LLM_EXECUTION_CONFIGS: weakref.WeakKeyDictionary[
    object, "_OpenAILLMExecutionConfig"
] = weakref.WeakKeyDictionary()
_OPENAI_LLM_EXECUTION_CONFIGS_LOCK = threading.Lock()
_OFFICIAL_DEEPSEEK_DEPLOYMENT_CONTRACT = (
    "official-deepseek-v4-flash-at-api.deepseek.com-v1"
)
_FROZEN_CURRENT_DEADLINE = current_deadline
_FROZEN_WITH_RETRY = with_retry
_FROZEN_REQUIRE_ACTIVE_MODEL = require_active_model


class _OpenAILLMExecutionConfig(NamedTuple):
    model: str
    endpoint: str
    thinking_mode: str
    send_thinking: bool
    request_timeout: float
    transport_package_version: str | None
    deployment_revision_sha256: str | None
    deployment_tenant_sha256: str | None
    official_deployment_contract: str | None
    completion_create: Callable[..., object]


def llm_attestation_sha256(value: object, *, label: str) -> str | None:
    """Hash an operator-declared public LLM deployment label."""

    if value is None:
        return None
    public = validate_public_attestation(value, label=label, max_bytes=4096)
    return "sha256:" + hashlib.sha256(public.encode("utf-8")).hexdigest()


def _registered_llm_execution_config(
    client: object,
) -> _OpenAILLMExecutionConfig | None:
    with _OPENAI_LLM_EXECUTION_CONFIGS_LOCK:
        return _OPENAI_LLM_EXECUTION_CONFIGS.get(client)


class OpenAICompatibleClient:
    """LLMClient backed by any OpenAI-compatible HTTP endpoint.

    Works with DeepSeek, OpenAI, Together, local vLLM, etc.  Provider-specific
    environment keys are origin-bound: ``DEEPSEEK_API_KEY`` and
    ``OPENAI_API_KEY`` are never sent to a custom host.  Use the purpose-bound
    ``HYMEM_LLM_API_KEY`` (or an explicit constructor key) for a custom HTTPS
    endpoint.

    Environment variables (all optional if arguments are passed directly):
        HYMEM_LLM_API_KEY   — purpose-bound API key for the configured endpoint
        HYMEM_LLM_BASE_URL  — base URL (default: https://api.deepseek.com)
        HYMEM_LLM_MODEL     — model name (default: deepseek-v4-flash)
        HYMEM_LLM_THINKING  — whether to send DeepSeek's `thinking` body key:
                              "auto" (default, send only on DeepSeek endpoints),
                              "disabled" (always send it, i.e. force reasoning
                              off), "off"/"enabled" (never send it).

    ``token_counter`` is an optional trusted tokenizer for the configured
    model. When supplied it is exposed as ``count_tokens`` so prompt packing
    and reported context usage share the exact same accounting. When omitted,
    the query layer uses its conservative offline fallback.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
        token_counter: Callable[[str], int] | None = None,
        deployment_revision: str | None = None,
        deployment_tenant: str | None = None,
    ) -> None:
        # Resolve and reject retired mutable aliases before endpoint credential
        # resolution, optional SDK import/construction, or any network-capable
        # object exists.  Environment fallback is part of the effective model
        # identity and must not bypass the same constructor-level policy.
        resolved_model = (
            model or os.environ.get("HYMEM_LLM_MODEL") or "deepseek-v4-flash"
        )
        if type(resolved_model) is str:
            resolved_model = resolved_model.strip()
        # Preserve the actionable retired-alias failure even when an operator
        # copied the legacy label with surrounding whitespace.  Active model
        # labels use this same single normalized value for attestation,
        # declaration, and wire dispatch immediately afterwards, before
        # credential resolution.
        _FROZEN_REQUIRE_ACTIVE_MODEL(resolved_model, role="HyMem LLM")
        resolved_model = validate_public_attestation(
            resolved_model, label="LLM model", max_bytes=4096,
        )
        resolved_base = (
            base_url
            or os.environ.get("HYMEM_LLM_BASE_URL")
            or "https://api.deepseek.com"
        )
        endpoint, resolved_key = resolve_llm_api_key(
            resolved_base, explicit_key=api_key
        )
        try:
            from openai import DefaultHttpxClient, OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install 'hymem[server]'"
            ) from e

        loaded_versions = dict(OPENAI_TRANSPORT_RUNTIME_VERSIONS)
        transport_package_version = loaded_versions.get("openai")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable or None")
        self._count_tokens = token_counter

        revision = (
            deployment_revision
            if deployment_revision is not None
            else os.environ.get("HYMEM_LLM_DEPLOYMENT_REVISION")
        )
        tenant = (
            deployment_tenant
            if deployment_tenant is not None
            else os.environ.get("HYMEM_LLM_DEPLOYMENT_TENANT")
        )
        revision_sha256 = llm_attestation_sha256(
            revision, label="LLM deployment revision",
        )
        tenant_sha256 = llm_attestation_sha256(
            tenant, label="LLM deployment tenant",
        )
        official_deployment_contract = (
            _OFFICIAL_DEEPSEEK_DEPLOYMENT_CONTRACT
            if (
                revision_sha256 is None
                and tenant_sha256 is None
                and endpoint.official_provider == "deepseek"
                and resolved_model == "deepseek-v4-flash"
            )
            else None
        )

        # `thinking` is a DeepSeek-specific body key: deepseek-v4-flash otherwise
        # spends its whole token budget reasoning and returns empty content. But
        # the OpenAI API rejects unknown body params with a 400, and vLLM's
        # tolerance varies by version — and because every call goes through
        # with_retry(), an unconditional send turns "wrong vendor" into three
        # doomed attempts with sleeps in between rather than one clean failure.
        # So resolve the decision once, here, from the endpoint we actually
        # resolved above, and let an operator override it either way.
        raw_mode = (
            thinking
            or os.environ.get("HYMEM_LLM_THINKING")
            or "auto"
        )
        if type(raw_mode) is not str:
            raise ValueError("HYMEM_LLM_THINKING must be public text")
        mode = raw_mode.strip().lower()
        if mode not in _THINKING_MODES:
            raise ValueError(
                f"Invalid HYMEM_LLM_THINKING value {mode!r}; "
                f"expected one of {sorted(_THINKING_MODES)}."
            )
        if mode == "auto":
            # Substring match, not equality: DeepSeek is also reached via
            # regional hosts and reverse proxies that keep the vendor name in
            # the host or the model id.
            host = endpoint.hostname
            send_thinking = (
                "deepseek" in host or "deepseek" in resolved_model.lower()
            )
        else:
            send_thinking = mode == "disabled"

        # These private mirrors are diagnostic only.  The actual request and
        # durable declaration use the module-owned immutable execution record
        # installed after the SDK transport is sealed.
        self._model = resolved_model
        self._base_url = endpoint.url
        self._thinking_mode = mode
        self._send_thinking = send_thinking
        self._transport_package_version = transport_package_version
        self._deployment_revision_sha256 = revision_sha256
        self._deployment_tenant_sha256 = tenant_sha256
        self._official_deployment_contract = official_deployment_contract

        # Provider accounting is cumulative and thread-safe because benchmark
        # workers can share this client. Any failed attempt or response without
        # a complete usage block makes token totals unavailable rather than
        # falsely exact.
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.cost_usd = None
        self.token_usage_available = False
        self._usage_complete = True
        self._usage_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._close_condition = threading.Condition(self._close_lock)
        self._active_calls = 0
        self._closed = False
        self._request_timeout = DEFAULT_LLM_TIMEOUT_SECONDS
        # Lexically scoped attempt meters solve a different problem from the
        # cumulative benchmark counters above. Each context/thread receives
        # its own immutable scope stack, while the mutable trackers on that
        # stack count only requests started under their caller-owned scope.
        # Nested wrappers are intentional: one HTTP attempt increments both
        # the inner runner meter and the outer chunk meter exactly once each.
        self._provider_attempt_scopes: ContextVar[
            tuple[ProviderAttemptTracker, ...]
        ] = ContextVar(
            f"hymem_provider_attempt_scopes_{id(self)}",
            default=(),
        )

        # ``complete()`` owns the only retry layer.  The SDK otherwise retries
        # each ``create`` call internally by default, invisibly multiplying both
        # cost and latency while ``request_attempts`` advances only once.  Pin
        # it to one HTTP attempt so every counted provider attempt is real and
        # the documented three-attempt extraction envelope is enforceable.
        owned_http_client = None
        factory = DefaultHttpxClient if callable(DefaultHttpxClient) else None
        if factory is not None:
            owned_http_client = factory(trust_env=False)
        kwargs = {
            "api_key": resolved_key,
            "base_url": endpoint.url,
            "organization": "",
            "project": "",
            "timeout": DEFAULT_LLM_TIMEOUT_SECONDS,
            "max_retries": 0,
        }
        if owned_http_client is not None:
            kwargs["http_client"] = owned_http_client
        try:
            self._client = OpenAI(**kwargs)
            self._owned_http_client = owned_http_client
            self._owned_http_client_policy = owned_http_client is not None
            self._transport_object = self._client
            self._http_transport_object = getattr(self._client, "_client", None)
            self._chat_resource = getattr(self._client, "chat", None)
            self._completion_resource = getattr(
                self._chat_resource, "completions", None,
            )
            self._completion_create_descriptor = inspect.getattr_static(
                type(self._completion_resource), "create", None,
            ) if self._completion_resource is not None else None
            self._completion_instance_create = getattr(
                self._completion_resource, "create", None,
            )
            completion_create = getattr(self._completion_resource, "create", None)
            if not callable(completion_create):
                raise RuntimeError("OpenAI completion resource is unavailable")
            self._completion_resource_has_client_backref = bool(
                self._completion_resource is not None
                and hasattr(self._completion_resource, "_client")
            )
            self._implementation_guard = _OPENAI_LLM_ORIGINAL_IMPLEMENTATION_GUARD
            self._helper_guard = _OPENAI_LLM_ORIGINAL_HELPER_GUARD
            self._lifecycle_guard = (
                id(self._usage_lock), id(self._close_lock),
                id(self._close_condition), id(self._provider_attempt_scopes),
                id(self._owned_http_client),
            )
            self._transport_guard = self._transport_state()
            with _OPENAI_LLM_CLOSE_TARGETS_LOCK:
                _OPENAI_LLM_CLOSE_TARGETS[self] = self._client
            with _OPENAI_LLM_EXECUTION_CONFIGS_LOCK:
                _OPENAI_LLM_EXECUTION_CONFIGS[self] = _OpenAILLMExecutionConfig(
                    model=resolved_model,
                    endpoint=endpoint.url,
                    thinking_mode=mode,
                    send_thinking=send_thinking,
                    request_timeout=float(DEFAULT_LLM_TIMEOUT_SECONDS),
                    transport_package_version=transport_package_version,
                    deployment_revision_sha256=revision_sha256,
                    deployment_tenant_sha256=tenant_sha256,
                    official_deployment_contract=official_deployment_contract,
                    completion_create=completion_create,
                )
        except BaseException as primary:
            if owned_http_client is not None:
                try:
                    owned_http_client.close()
                except Exception as cleanup_error:
                    try:
                        primary.add_note(
                            "owned LLM transport cleanup failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    except Exception:
                        pass
            raise

    @staticmethod
    def _canonical_transport_value(value: object) -> object:
        from hymem.contrib.openai_embedding_client import (
            OpenAICompatibleEmbeddingClient,
        )

        return OpenAICompatibleEmbeddingClient._canonical_transport_value(value)

    @classmethod
    def _http_transport_state(cls, transport: object) -> object:
        from hymem.contrib.openai_embedding_client import (
            OpenAICompatibleEmbeddingClient,
        )

        return OpenAICompatibleEmbeddingClient._http_transport_state(transport)

    def _transport_state(self) -> tuple[object, ...]:
        transport = self._client
        http_client = getattr(transport, "_client", None)
        mounts = getattr(http_client, "_mounts", None)
        mount_state = (
            tuple(sorted(
                (id(key), self._http_transport_state(value))
                for key, value in mounts.items()
            ))
            if isinstance(mounts, Mapping) else None
        )
        hooks = getattr(http_client, "_event_hooks", None)
        hook_state = (
            tuple(sorted(
                (str(key), tuple(id(callback) for callback in callbacks))
                for key, callbacks in hooks.items()
            ))
            if isinstance(hooks, Mapping) else None
        )
        timeout = getattr(http_client, "timeout", None)
        def collection_state(value: object) -> object:
            try:
                if hasattr(value, "multi_items"):
                    value = list(value.multi_items())
                elif hasattr(value, "items"):
                    value = list(value.items())
            except Exception:
                pass
            return self._canonical_transport_value(value)

        return (
            str(getattr(transport, "base_url", "")),
            getattr(transport, "api_key", None),
            getattr(transport, "organization", None),
            getattr(transport, "project", None),
            self._canonical_transport_value(
                getattr(transport, "default_headers", None)
            ),
            self._canonical_transport_value(
                getattr(transport, "default_query", None)
            ),
            self._http_transport_state(getattr(http_client, "_transport", None)),
            id(getattr(http_client, "_auth", None)),
            mount_state,
            hook_state,
            str(getattr(http_client, "base_url", "")),
            collection_state(getattr(http_client, "headers", None)),
            collection_state(getattr(http_client, "cookies", None)),
            collection_state(getattr(http_client, "params", None)),
            tuple(
                self._canonical_transport_value(getattr(timeout, field, None))
                for field in ("connect", "read", "write", "pool")
            ),
            self._canonical_transport_value(
                getattr(http_client, "follow_redirects", None)
            ),
            self._canonical_transport_value(
                getattr(http_client, "max_redirects", None)
            ),
            self._canonical_transport_value(getattr(http_client, "trust_env", None)),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def thinking_mode(self) -> str:
        return self._thinking_mode

    @property
    def transport_package_version(self) -> str | None:
        return self._transport_package_version

    @property
    def count_tokens(self) -> Callable[[str], int] | None:
        return self._count_tokens

    @property
    def transport_integrity_ok(self) -> bool:
        try:
            config = _registered_llm_execution_config(self)
            if config is None:
                return False
            resource = self._client.chat.completions
            return bool(
                type(self) is OpenAICompatibleClient
                and OPENAI_TRANSPORT_RUNTIME_EXACT
                and self._owned_http_client_policy
                and self._closed is False
                and self._implementation_guard is not None
                and self._helper_guard == _OPENAI_LLM_ORIGINAL_HELPER_GUARD
                and self._helper_guard == (
                    current_deadline, with_retry, require_active_model,
                    resolve_llm_api_key, retry_module.retry_support_integrity,
                )
                and retry_module.retry_support_integrity()
                and tuple(
                    inspect.getattr_static(type(self), name, None)
                    for name in _OPENAI_LLM_EXECUTION_ATTRIBUTES
                ) == self._implementation_guard
                and not any(
                    name in object.__getattribute__(self, "__dict__")
                    for name in _OPENAI_LLM_EXECUTION_ATTRIBUTES
                )
                and self._client is self._transport_object
                and getattr(self._client, "_client", None)
                    is self._http_transport_object
                and self._http_transport_object is self._owned_http_client
                and self._chat_resource is getattr(self._client, "chat", None)
                and resource is self._completion_resource
                and (
                    self._model,
                    self._base_url,
                    self._thinking_mode,
                    self._send_thinking,
                    self._request_timeout,
                    self._transport_package_version,
                    self._deployment_revision_sha256,
                    self._deployment_tenant_sha256,
                    self._official_deployment_contract,
                ) == config[:-1]
                and (
                    not self._completion_resource_has_client_backref
                    or getattr(resource, "_client", None) is self._client
                )
                and inspect.getattr_static(type(resource), "create", None)
                    is self._completion_create_descriptor
                and (
                    (
                        self._completion_create_descriptor is not None
                        and "create" not in getattr(resource, "__dict__", {})
                    )
                    or (
                        self._completion_create_descriptor is None
                        and getattr(resource, "create", None)
                            is self._completion_instance_create
                    )
                )
                and self._lifecycle_guard == (
                    id(self._usage_lock), id(self._close_lock),
                    id(self._close_condition), id(self._provider_attempt_scopes),
                    id(self._owned_http_client),
                )
                and self._transport_state() == self._transport_guard
            )
        except Exception:
            return False

    def _verify_transport_integrity(self) -> None:
        if not self.transport_integrity_ok:
            raise RuntimeError("LLM transport identity changed")

    @property
    def effective_extra_body(self) -> dict:
        """Body extension actually sent by :meth:`complete`.

        Deriving this from the same private switch used at request time keeps
        the producer declaration and wire request from drifting apart if a
        long-lived client is reconfigured while Phase-1 is running.
        """

        config = _registered_llm_execution_config(self)
        if config is None:
            raise RuntimeError("LLM execution identity is unavailable")
        return (
            {"thinking": {"type": "disabled"}}
            if config.send_thinking else {}
        )

    def phase1_producer_declaration(self):
        """Return the exact credential-free Phase-1 request implementation."""

        if not self.transport_integrity_ok:
            raise NotImplementedError("OpenAI LLM transport identity is not exact")
        config = _registered_llm_execution_config(self)
        if config is None:
            raise NotImplementedError("OpenAI LLM execution identity is unavailable")
        _FROZEN_REQUIRE_ACTIVE_MODEL(config.model, role="HyMem LLM")
        if type(config.transport_package_version) is not str or not (
            config.transport_package_version.strip()
        ):
            # Let the generic producer resolver assign a process-instance
            # identity rather than calling this incomplete declaration exact.
            raise NotImplementedError(
                "OpenAI transport package version is unavailable"
            )
        if (
            config.official_deployment_contract is None
            and (
                config.deployment_revision_sha256 is None
                or config.deployment_tenant_sha256 is None
            )
        ):
            raise NotImplementedError(
                "custom OpenAI-compatible LLM endpoints require public "
                "deployment revision and tenant attestations"
            )
        return openai_compatible_producer_declaration(
            model=config.model,
            endpoint=config.endpoint,
            thinking_mode=config.thinking_mode,
            effective_extra_body=self.effective_extra_body,
            transport_package_version=config.transport_package_version,
            request_timeout_seconds=config.request_timeout,
            deployment_revision_sha256=config.deployment_revision_sha256,
            deployment_tenant_sha256=config.deployment_tenant_sha256,
            official_deployment_contract=config.official_deployment_contract,
        )

    def aggregation_producer_declaration(self):
        """Aggregation uses this client's same declared chat-completion wire."""

        return self.phase1_producer_declaration()

    @contextmanager
    def track_provider_attempts(self) -> Iterator[ProviderAttemptTracker]:
        """Yield an exact attempt meter local to this completion scope.

        The scope is safe across threads and copied contexts and remains
        populated when ``complete()`` raises after exhausting retries.
        Cumulative public usage metrics continue to be updated independently.
        """
        tracker = ProviderAttemptTracker()
        active = self._provider_attempt_scopes.get()
        token = self._provider_attempt_scopes.set((*active, tracker))
        try:
            yield tracker
        finally:
            self._provider_attempt_scopes.reset(token)

    def complete(self, request: LLMRequest) -> str:
        with self._close_condition:
            if self._closed:
                raise RuntimeError("LLM transport identity changed")
            self._active_calls += 1
        try:
            return self._complete_with_execution_lease(request)
        finally:
            with self._close_condition:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._close_condition.notify_all()

    def _complete_with_execution_lease(self, request: LLMRequest) -> str:
        config = _registered_llm_execution_config(self)
        if config is None:
            raise RuntimeError("LLM execution identity is unavailable")
        self._verify_transport_integrity()
        _FROZEN_REQUIRE_ACTIVE_MODEL(config.model, role="HyMem LLM")
        kwargs: dict = dict(
            model=config.model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user",   "content": request.user},
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        # Omit the key entirely rather than passing an empty/None extra_body:
        # some servers reject a body they were not expecting at all.
        extra_body = (
            {"thinking": {"type": "disabled"}}
            if config.send_thinking else {}
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        def _attempt():
            deadline = _FROZEN_CURRENT_DEADLINE()
            request_kwargs = dict(kwargs)
            if deadline is not None:
                request_kwargs["timeout"] = deadline.cap_timeout(
                    config.request_timeout
                )
            for tracker in self._provider_attempt_scopes.get():
                tracker._record()
            with self._usage_lock:
                self.request_attempts += 1
            started = time.monotonic()
            try:
                response = config.completion_create(**request_kwargs)
                # A custom/OpenAI-compatible transport may ignore its timeout.
                # Reject a late response before parsing or publication once it
                # eventually hands control back to us.
                if deadline is not None:
                    deadline.check()
                return response
            except (Exception, DeadlineExceeded):
                with self._usage_lock:
                    self._usage_complete = False
                    self.token_usage_available = False
                raise
            finally:
                with self._usage_lock:
                    self.total_latency_s += time.monotonic() - started

        resp = _FROZEN_WITH_RETRY(
            _attempt,
            attempts=DEFAULT_RETRY_ATTEMPTS,
            base_delay=DEFAULT_RETRY_BASE_DELAY_SECONDS,
            max_delay=DEFAULT_RETRY_MAX_DELAY_SECONDS,
            label=f"LLM completion ({config.model})",
        )
        self._verify_transport_integrity()
        content = resp.choices[0].message.content
        if type(content) is not str:
            raise RuntimeError("LLM response content is not a string")
        usage = getattr(resp, "usage", None)
        values = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
            for value in values.values()
        )
        with self._usage_lock:
            self.call_count += 1
            self.successful_responses += 1
            if valid:
                self.prompt_tokens += values["prompt_tokens"]
                self.completion_tokens += values["completion_tokens"]
                self.total_tokens += values["total_tokens"]
            else:
                self._usage_complete = False
            self.token_usage_available = (
                self.successful_responses > 0 and self._usage_complete
            )
        return content

    def close(self) -> None:
        """Release the SDK's underlying HTTP connection pool exactly once."""

        with self._close_lock:
            if self._closed:
                return
            # Mark first so no new caller can acquire an execution lease, then
            # wait for already-started requests to finish before releasing the
            # shared transport. Concurrent requests remain allowed.
            self._closed = True
            while self._active_calls:
                self._close_condition.wait()
            with _OPENAI_LLM_CLOSE_TARGETS_LOCK:
                target = _OPENAI_LLM_CLOSE_TARGETS.pop(self, self._client)
            with _OPENAI_LLM_EXECUTION_CONFIGS_LOCK:
                _OPENAI_LLM_EXECUTION_CONFIGS.pop(self, None)
            close = getattr(target, "close", None)
            if callable(close):
                close()


_OPENAI_LLM_ORIGINAL_IMPLEMENTATION_GUARD = tuple(
    inspect.getattr_static(OpenAICompatibleClient, name, None)
    for name in _OPENAI_LLM_EXECUTION_ATTRIBUTES
)
_OPENAI_LLM_ORIGINAL_HELPER_GUARD = (
    current_deadline, with_retry, require_active_model, resolve_llm_api_key,
    retry_module.retry_support_integrity,
)


def maintained_openai_llm_integrity(client: object) -> bool:
    """Evaluate the frozen transport property, not mutable instance dispatch."""

    if type(client) is not OpenAICompatibleClient:
        return False
    try:
        property_index = _OPENAI_LLM_EXECUTION_ATTRIBUTES.index(
            "transport_integrity_ok"
        )
        descriptor = _OPENAI_LLM_ORIGINAL_IMPLEMENTATION_GUARD[property_index]
        return bool(
            isinstance(descriptor, property)
            and descriptor.fget is not None
            and descriptor.fget(client)
        )
    except Exception:
        return False


_OPENAI_LLM_INTEGRITY_FUNCTION = maintained_openai_llm_integrity
_OPENAI_LLM_MAINTAINED_CLASS = OpenAICompatibleClient


def openai_compatible_producer_declaration(
    *, model: str, endpoint: str, thinking_mode: str,
    effective_extra_body: dict, transport_package_version: str,
    request_timeout_seconds: float,
    deployment_revision_sha256: str | None = None,
    deployment_tenant_sha256: str | None = None,
    official_deployment_contract: str | None = None,
    require_consistent_thinking: bool = True,
):
    """Derive the same safe producer declaration without constructing I/O."""

    from hymem.contrib.endpoint_policy import ENDPOINT_POLICY_VERSION
    from hymem.extraction.producer import Phase1ProducerDeclaration

    model = validate_public_attestation(
        model, label="OpenAI LLM model", max_bytes=4096,
    )
    _FROZEN_REQUIRE_ACTIVE_MODEL(model, role="HyMem LLM")
    if thinking_mode not in _THINKING_MODES:
        raise ValueError("OpenAI thinking mode is invalid")
    from urllib.parse import urlsplit
    host = (urlsplit(endpoint).hostname or "").casefold()
    sends_thinking = thinking_mode == "disabled" or (
        thinking_mode == "auto"
        and ("deepseek" in host or "deepseek" in model.casefold())
    )
    expected_extra_body = (
        {"thinking": {"type": "disabled"}} if sends_thinking else {}
    )
    # A declaration that disagrees with the actual body can never be exact;
    # callers cannot opt out of this consistency check.
    if effective_extra_body != expected_extra_body:
        raise ValueError("OpenAI effective thinking request is inconsistent")
    if (
        type(transport_package_version) is not str
        or not transport_package_version.strip()
    ):
        raise ValueError("OpenAI transport package version is unavailable")
    validate_public_attestation(
        transport_package_version,
        label="OpenAI transport package version",
        max_bytes=256,
    )
    digest_re = re.compile(r"sha256:[0-9a-f]{64}\Z")
    exact_operator_attestation = bool(
        type(deployment_revision_sha256) is str
        and digest_re.fullmatch(deployment_revision_sha256)
        and type(deployment_tenant_sha256) is str
        and digest_re.fullmatch(deployment_tenant_sha256)
        and official_deployment_contract is None
    )
    from hymem.contrib.endpoint_policy import validate_http_endpoint
    endpoint_meta = validate_http_endpoint(endpoint, label="OpenAI LLM")
    if (
        official_deployment_contract is None
        and deployment_revision_sha256 is None
        and deployment_tenant_sha256 is None
        and endpoint_meta.official_provider == "deepseek"
        and model == "deepseek-v4-flash"
    ):
        official_deployment_contract = _OFFICIAL_DEEPSEEK_DEPLOYMENT_CONTRACT
    exact_official_contract = bool(
        official_deployment_contract == _OFFICIAL_DEEPSEEK_DEPLOYMENT_CONTRACT
        and deployment_revision_sha256 is None
        and deployment_tenant_sha256 is None
        and endpoint_meta.official_provider == "deepseek"
        and model == "deepseek-v4-flash"
    )
    if not (exact_operator_attestation or exact_official_contract):
        raise ValueError(
            "OpenAI-compatible LLM identity requires public deployment "
            "revision and tenant attestations"
        )
    if (
        isinstance(request_timeout_seconds, bool)
        or not isinstance(request_timeout_seconds, (int, float))
        or not math.isfinite(float(request_timeout_seconds))
        or request_timeout_seconds <= 0
    ):
        raise ValueError("OpenAI request timeout is invalid")
    return Phase1ProducerDeclaration(
        client_id="hymem.contrib.openai_client.OpenAICompatibleClient",
        implementation=OPENAI_LLM_IMPLEMENTATION_SHA256,
        model=model,
        endpoint=endpoint,
        effective_request={
            "endpoint_policy": ENDPOINT_POLICY_VERSION,
            "transport": "openai-python",
            "transport_package_version": transport_package_version,
            "transport_runtime_versions": dict(
                OPENAI_TRANSPORT_RUNTIME_VERSIONS
            ),
            "ambient_organization": "disabled",
            "ambient_project": "disabled",
            "messages": ["system", "user"],
            "model_source": "resolved_client_model",
            "temperature_source": "LLMRequest.temperature",
            "max_tokens_source": "LLMRequest.max_tokens",
            "json_response_format": {"type": "json_object"},
            "text_response_format": None,
            "thinking_mode": thinking_mode,
            "effective_extra_body": effective_extra_body,
            "request_timeout_seconds": request_timeout_seconds,
            "deployment_revision_sha256": deployment_revision_sha256,
            "deployment_tenant_sha256": deployment_tenant_sha256,
            "official_deployment_contract": official_deployment_contract,
        },
        retry_policy={
            "owner": "OpenAICompatibleClient.complete",
            "attempts": DEFAULT_RETRY_ATTEMPTS,
            "base_delay_seconds": DEFAULT_RETRY_BASE_DELAY_SECONDS,
            "max_delay_seconds": DEFAULT_RETRY_MAX_DELAY_SECONDS,
            "sdk_max_retries": 0,
        },
    )


# Freeze the maintained implementation at this owner module's import
# boundary.  Runtime identity must never reread mutable package files after a
# rolling deploy, and each dependency contributes its independently captured
# commitment rather than being hashed later by this consumer.
from hymem.contrib.endpoint_policy import ENDPOINT_POLICY_IMPLEMENTATION_SHA256
from hymem.contrib.implementation_identity import (
    compose_import_time_sha256,
    import_time_source_sha256,
)
from hymem.contrib.model_policy import MODEL_POLICY_IMPLEMENTATION_SHA256
from hymem.deadline import DEADLINE_IMPLEMENTATION_SHA256

OPENAI_LLM_IMPLEMENTATION_SHA256 = compose_import_time_sha256(
    import_time_source_sha256(__file__),
    ENDPOINT_POLICY_IMPLEMENTATION_SHA256,
    MODEL_POLICY_IMPLEMENTATION_SHA256,
    retry_module.EXTRACTION_IMPLEMENTATION_SHA256,
    DEADLINE_IMPLEMENTATION_SHA256,
)
