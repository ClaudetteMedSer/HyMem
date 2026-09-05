from __future__ import annotations

import ipaddress
import math
import os
import threading
import time
from typing import Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 10.0


def _embedding_url_parts(base_url: str):
    raw_url = str(base_url).strip()
    if not raw_url:
        raise ValueError("embedding base URL must be non-empty")
    parts = urlsplit(raw_url)
    if not parts.scheme or not parts.hostname:
        raise ValueError("embedding base URL must be absolute")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("embedding base URL has an invalid port") from exc
    return parts


def is_loopback_embedding_url(base_url: str) -> bool:
    """Whether an HTTP endpoint is unambiguously local to this machine."""
    try:
        hostname = (_embedding_url_parts(base_url).hostname or "").lower()
    except ValueError:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def is_official_openai_embedding_url(base_url: str) -> bool:
    """Whether it is safe to inherit the general ``OPENAI_API_KEY``."""
    try:
        parts = _embedding_url_parts(base_url)
    except ValueError:
        return False
    return (
        parts.scheme.lower() == "https"
        and (parts.hostname or "").lower() == "api.openai.com"
    )


def validate_embedding_base_url(base_url: str) -> None:
    """Reject unsupported or clear-text non-loopback embedding transports."""
    parts = _embedding_url_parts(base_url)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("embedding base URL must use http:// or https://")
    if scheme == "http" and not is_loopback_embedding_url(base_url):
        raise ValueError(
            "plaintext HTTP embedding endpoints are allowed only on loopback"
        )


def safe_embedding_base_url(base_url: str) -> str:
    """Return an endpoint label safe for diagnostics and logs.

    Userinfo and fragments are discarded. Non-secret routing query parameters
    remain exact; credential values become one opaque marker and never a
    dictionary-verifiable hash. An invalid URL is represented by a constant
    rather than echoing attacker- or operator-controlled input in an error path.
    """
    try:
        parts = _embedding_url_parts(base_url)
        scheme = parts.scheme.lower()
        hostname = (parts.hostname or "").lower()
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        credential_keys = {
            "api_key", "apikey", "key", "token", "access_token", "auth",
            "authorization", "password", "passwd", "secret", "signature",
            "sig", "credential", "x_amz_signature", "x_amz_credential",
            "x_amz_security_token",
        }
        query = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            normalized = key.casefold().replace("-", "_")
            is_secret = (
                normalized in credential_keys
                or normalized.endswith("_token")
                or normalized.endswith("_secret")
                or normalized.endswith("_password")
            )
            if not is_secret:
                query.append((key, value))
        return urlunsplit((
            scheme, host, parts.path.rstrip("/"), urlencode(query, doseq=True), "",
        ))
    except (TypeError, ValueError, OverflowError):
        return "<invalid embedding URL>"


def openai_compatible_embedding_identity(base_url: str, model: str) -> str:
    """Stable vector-space id containing provider endpoint and request model.

    Model labels are not globally unique: two gateways may expose the same
    label with unrelated weights/tokenization.  Persisting only that label can
    silently mix their vectors.  Credentials and URL fragments are excluded.
    """
    raw_url = str(base_url).strip()
    raw_model = str(model).strip()
    if not raw_url or not raw_model:
        raise ValueError("embedding endpoint and model must be non-empty")
    # Validate before using the safe form: the display helper intentionally
    # maps malformed input to a constant, while a durable identity must fail.
    _embedding_url_parts(raw_url)
    normalized = safe_embedding_base_url(raw_url)
    return f"openai-compatible:{normalized}::{raw_model}"


class OpenAICompatibleEmbeddingClient:
    """EmbeddingClient backed by any OpenAI-compatible HTTP endpoint.

    Works with OpenAI and embedding-capable OpenAI-compatible endpoints (for
    example Together or a local vLLM deployment).  A chat-compatible endpoint
    is not necessarily embedding-compatible; in particular no LLM credential
    is implicitly reused here.
    All constructor arguments fall back to environment variables so the server
    can be configured entirely via the shell environment.

    Environment variables (all optional if arguments are passed directly):
        HYMEM_EMBEDDING_API_KEY   — API key. OPENAI_API_KEY is inherited only
                                    for the official api.openai.com endpoint;
                                    loopback endpoints use a dummy local key.
        HYMEM_EMBEDDING_BASE_URL  — base URL (default: https://api.openai.com/v1)
        HYMEM_EMBEDDING_MODEL     — model name (default: text-embedding-3-small)
        HYMEM_EMBEDDING_DIM       — declared dim (default: 1536); the actual
                                    dim is read from the API response on the
                                    first call.
        HYMEM_EMBEDDING_TIMEOUT_SECONDS
                                  — per-request timeout (default: 10 seconds).
                                    SDK retries are disabled.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved_base = (
            base_url
            or os.environ.get("HYMEM_EMBEDDING_BASE_URL")
            or "https://api.openai.com/v1"
        )
        validate_embedding_base_url(resolved_base)

        resolved_key = api_key or os.environ.get("HYMEM_EMBEDDING_API_KEY")
        if not resolved_key and is_official_openai_embedding_url(resolved_base):
            resolved_key = os.environ.get("OPENAI_API_KEY")
        if not resolved_key and is_loopback_embedding_url(resolved_base):
            resolved_key = "local"
        if not resolved_key:
            raise EnvironmentError(
                "No embedding API key found. Set HYMEM_EMBEDDING_API_KEY; "
                "OPENAI_API_KEY is used only for https://api.openai.com."
            )

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install 'hymem[server]'"
            ) from e
        self._model = (
            model
            or os.environ.get("HYMEM_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )
        env_dim = os.environ.get("HYMEM_EMBEDDING_DIM")
        try:
            resolved_dim = dim if dim is not None else (int(env_dim) if env_dim else 1536)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding dimension must be a positive integer") from exc
        if (
            isinstance(resolved_dim, bool)
            or not isinstance(resolved_dim, int)
            or resolved_dim <= 0
        ):
            raise ValueError("embedding dimension must be a positive integer")
        if not isinstance(self._model, str) or not self._model.strip():
            raise ValueError("embedding model id must be non-empty")
        self._model = self._model.strip()
        env_timeout = os.environ.get("HYMEM_EMBEDDING_TIMEOUT_SECONDS")
        try:
            resolved_timeout = (
                timeout
                if timeout is not None
                else (
                    float(env_timeout)
                    if env_timeout is not None
                    else DEFAULT_EMBEDDING_TIMEOUT_SECONDS
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding timeout must be a positive finite number") from exc
        if (
            isinstance(resolved_timeout, bool)
            or not isinstance(resolved_timeout, (int, float))
            or not math.isfinite(float(resolved_timeout))
            or float(resolved_timeout) <= 0.0
        ):
            raise ValueError("embedding timeout must be a positive finite number")
        self._identity = openai_compatible_embedding_identity(
            resolved_base, self._model
        )
        self._dim = resolved_dim
        self.base_url = safe_embedding_base_url(resolved_base)
        self.call_count = 0
        self.request_attempts = 0
        self.successful_responses = 0
        self.input_count = 0
        self.input_characters = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.cost_usd = None
        self.token_usage_available = False
        self._usage_complete = True
        self._usage_lock = threading.Lock()
        # Keep exactly one retry layer on the latency-sensitive query and
        # post-commit ingestion paths.  The SDK default retries and multi-minute
        # timeout would otherwise multiply into an effectively unbounded stall.
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
            timeout=float(resolved_timeout),
            max_retries=0,
        )

    @property
    def model(self) -> str:
        """Persisted vector-space identity (provider endpoint + model)."""
        return self._identity

    @property
    def request_model(self) -> str:
        """Model label sent to the remote API."""
        return self._model

    @property
    def backend(self) -> str:
        return "openai_compatible"

    @property
    def quality(self) -> str:
        return "semantic"

    @property
    def network_free(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = list(texts)
        if not payload:
            return []
        with self._usage_lock:
            self.request_attempts += 1
            self.input_count += len(payload)
            self.input_characters += sum(len(str(text)) for text in payload)
        started = time.monotonic()
        try:
            resp = self._client.embeddings.create(model=self._model, input=payload)
        except Exception:
            with self._usage_lock:
                self._usage_complete = False
                self.token_usage_available = False
            raise
        finally:
            with self._usage_lock:
                self.total_latency_s += time.monotonic() - started
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        valid_usage = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
            for value in (prompt_tokens, total_tokens)
        )
        with self._usage_lock:
            self.call_count += 1
            self.successful_responses += 1
            if valid_usage:
                self.prompt_tokens += prompt_tokens
                self.total_tokens += total_tokens
            else:
                self._usage_complete = False
            self.token_usage_available = (
                self.successful_responses > 0 and self._usage_complete
            )
        data = list(resp.data)
        if len(data) != len(payload):
            raise RuntimeError(
                f"embedding provider returned {len(data)} vectors for "
                f"{len(payload)} inputs"
            )
        indices = [getattr(item, "index", None) for item in data]
        has_any_index = any(index is not None for index in indices)
        has_complete_indices = all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in indices
        )
        if has_any_index and not has_complete_indices:
            raise RuntimeError("embedding provider returned partial/malformed indices")
        if has_complete_indices:
            if sorted(indices) != list(range(len(payload))):
                raise RuntimeError("embedding provider returned invalid response indices")
            data.sort(key=lambda item: item.index)
        vectors: list[list[float]] = []
        resolved_dim: int | None = None
        for item in data:
            raw = getattr(item, "embedding", None)
            if not isinstance(raw, (list, tuple)) or not raw:
                raise RuntimeError("embedding provider returned a malformed vector")
            try:
                vector = [float(value) for value in raw]
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("embedding provider returned a malformed vector") from exc
            norm = math.sqrt(sum(value * value for value in vector))
            if (
                not all(math.isfinite(value) for value in vector)
                or not math.isfinite(norm) or norm <= 0.0
            ):
                raise RuntimeError("embedding provider returned a non-finite/zero vector")
            if resolved_dim is None:
                resolved_dim = len(vector)
            elif len(vector) != resolved_dim:
                raise RuntimeError("embedding provider returned mixed vector dimensions")
            vectors.append(vector)
        if resolved_dim is not None:
            self._dim = resolved_dim
        return vectors
