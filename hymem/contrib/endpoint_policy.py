"""Fail-closed endpoint and credential policy for provider transports.

Provider-compatible clients are deliberately configurable, but a configurable
URL must not turn a process-wide provider credential into a bearer token for an
unrelated host.  This module keeps URL validation and key inheritance in one
stdlib-only place so the library clients and benchmark clients cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
import base64
import binascii
from dataclasses import dataclass
import hashlib
import ipaddress
import os
import re
import string
from urllib.parse import urlsplit, urlunsplit
from urllib.parse import unquote
import unicodedata

from hymem.contrib.implementation_identity import import_time_source_sha256


DEEPSEEK_API_HOST = "api.deepseek.com"
OPENAI_API_HOST = "api.openai.com"
ENDPOINT_POLICY_VERSION = "hymem-provider-endpoint-policy-v1"
EMBEDDING_INTERNAL_HTTP_ENV = "HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP"
TRANSPORT_SECURITY_HTTPS = "https"
TRANSPORT_SECURITY_LOOPBACK_HTTP = "loopback-http"
TRANSPORT_SECURITY_INTERNAL_HTTP = "explicit-internal-http"
TRANSPORT_SECURITY_LOCAL = "local-no-network"
TRANSPORT_SECURITY_NONE = "none"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_INTERNAL_DNS_SUFFIXES = (
    ".internal",
    ".local",
    ".svc",
    ".svc.cluster.local",
)
_RFC1918_V4 = tuple(ipaddress.ip_network(value) for value in (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
))
_PRIVATE_V6 = ipaddress.ip_network("fc00::/7")
_CREDENTIAL_PATH_ROOTS = frozenset({
    "apikey", "accesstoken", "authtoken", "bearer", "bearertoken",
    "clientsecret", "credential", "credentials", "key", "password",
    "secret", "secretkey", "token",
})
_PUBLIC_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:api[^a-z0-9]*key|access[^a-z0-9]*token|"
    r"auth[^a-z0-9]*token|client[^a-z0-9]*secret|password|passwd|"
    r"secret|token)[^a-z0-9]*[:=]"
)
_PUBLIC_BEARER_RE = re.compile(r"^bearer(?:\s+|\s*:)\S+\Z", re.IGNORECASE)
_PUBLIC_BASIC_RE = re.compile(
    r"^(?:authorization\s*:\s*)?basic(?:\s+|\s*:)\S+\Z",
    re.IGNORECASE,
)
_PUBLIC_SK_RE = re.compile(
    r"^sk(?:[-_](?:live|test|prod|project|proj))?[-_][A-Za-z0-9_-]{8,}\Z",
    re.IGNORECASE,
)
_PUBLIC_JWT_RE = re.compile(
    r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\Z"
)
_PUBLIC_BASE64_RE = re.compile(r"[A-Za-z0-9+/_-]{24,}={0,2}\Z")


class EndpointPolicyError(ValueError):
    """An endpoint or its credential source violates the transport policy."""


def validate_public_attestation(
    value: object,
    *,
    label: str,
    max_bytes: int = 4096,
) -> str:
    """Validate one operator-declared public identity label.

    Digests of these values are integrity commitments, not encryption.  This
    bounded check rejects common credential encodings before a value can be
    hashed into durable metadata.  It deliberately makes no claim that an
    arbitrary label is proven non-secret; operators must provide public model,
    deployment, tenant, revision, and request-policy identifiers.

    Errors never interpolate the rejected value.
    """

    if (
        type(value) is not str
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > max_bytes
    ):
        raise ValueError(f"{label} must be non-empty public metadata")
    decoded = value
    for _ in range(4):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    if unquote(decoded) != decoded or re.search(r"%[0-9a-fA-F]{2}", decoded):
        raise ValueError(f"{label} must be public non-credential metadata")
    normalized = unicodedata.normalize("NFKC", decoded)
    normalized = "".join(
        character for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    folded = normalized.casefold()
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    decoded_credential = False
    if _PUBLIC_BASE64_RE.fullmatch(normalized) is not None:
        try:
            padding = "=" * ((-len(normalized)) % 4)
            raw_decoded = base64.urlsafe_b64decode(normalized + padding)
            rendered = raw_decoded.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
        else:
            rendered = unicodedata.normalize("NFKC", rendered).casefold()
            rendered_compact = re.sub(r"[^a-z0-9]+", "", rendered)
            decoded_credential = any(
                marker in rendered_compact for marker in (
                    "apikey", "authorization", "bearer", "credential",
                    "password", "secret", "token",
                )
            )
    if (
        "://" in folded
        or _PUBLIC_CREDENTIAL_ASSIGNMENT_RE.search(folded) is not None
        or _PUBLIC_BEARER_RE.fullmatch(normalized) is not None
        or _PUBLIC_BASIC_RE.fullmatch(normalized) is not None
        or _PUBLIC_SK_RE.fullmatch(normalized) is not None
        or _PUBLIC_JWT_RE.fullmatch(normalized) is not None
        or compact.startswith((
            "authorizationbearer", "apikey", "accesstoken", "authtoken",
            "clientsecret", "sklive", "sktest", "skprod", "skproject",
            "skproj",
        ))
        or decoded_credential
    ):
        raise ValueError(f"{label} must be public non-credential metadata")
    return value


@dataclass(frozen=True)
class ValidatedEndpoint:
    """Canonical, credential-free endpoint metadata."""

    url: str
    scheme: str
    hostname: str
    port: int | None
    is_loopback: bool
    is_internal_service: bool
    official_provider: str | None


_NETWORK_TRANSPORT_SECURITY = frozenset({
    TRANSPORT_SECURITY_HTTPS,
    TRANSPORT_SECURITY_LOOPBACK_HTTP,
    TRANSPORT_SECURITY_INTERNAL_HTTP,
})


def _strict_env_flag(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None:
        return False
    if not isinstance(raw, str):
        raise EndpointPolicyError(
            f"{name} must be one of 1/true/yes/on or 0/false/no/off"
        )
    normalized = raw.strip().casefold()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise EndpointPolicyError(
        f"{name} must be one of 1/true/yes/on or 0/false/no/off"
    )


def _canonical_host(
    hostname: str,
) -> tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address | None]:
    if not hostname or "%" in hostname:
        raise EndpointPolicyError("provider endpoint host is malformed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None:
        return address.compressed.casefold(), address

    absolute = hostname.endswith(".")
    unqualified = hostname[:-1] if absolute else hostname
    try:
        ascii_host = unqualified.encode("idna").decode("ascii").casefold()
    except UnicodeError as exc:
        raise EndpointPolicyError("provider endpoint host is malformed") from exc
    if (
        not ascii_host
        or len(ascii_host) > 253
        or any(not _DNS_LABEL.fullmatch(label) for label in ascii_host.split("."))
    ):
        raise EndpointPolicyError("provider endpoint host is malformed")
    return ascii_host + ("." if absolute else ""), None


def _host_scope(
    hostname: str,
    address: ipaddress.IPv4Address | ipaddress.IPv6Address | None,
) -> tuple[bool, bool]:
    if address is not None:
        loopback = address.is_loopback
        if isinstance(address, ipaddress.IPv4Address):
            private = any(address in network for network in _RFC1918_V4)
        else:
            private = address in _PRIVATE_V6
        return loopback, bool(loopback or private or address.is_link_local)

    folded = hostname.casefold()
    scope_host = folded[:-1] if folded.endswith(".") else folded
    loopback = scope_host == "localhost" or scope_host.endswith(".localhost")
    # A Docker/Compose/Kubernetes service such as ``embedding-server`` is a
    # single DNS label.  Numeric and hexadecimal-integer labels are excluded
    # because several URL stacks interpret them as alternative IPv4 spellings
    # (for example ``2130706433`` and ``0x7f000001``).
    alternate_numeric_host = scope_host.isdecimal() or (
        scope_host.startswith("0x")
        and len(scope_host) > 2
        and all(character in string.hexdigits for character in scope_host[2:])
    )
    single_label_service = (
        "." not in scope_host
        and any(character.isalpha() for character in scope_host)
        and scope_host != "localhost"
        and not alternate_numeric_host
    )
    internal_suffix = any(
        scope_host.endswith(suffix) and scope_host != suffix[1:]
        for suffix in _INTERNAL_DNS_SUFFIXES
    )
    return loopback, bool(loopback or single_label_service or internal_suffix)


def _official_provider(scheme: str, hostname: str, port: int | None) -> str | None:
    # Explicit :443 and an omitted default port are the same HTTPS origin.
    if scheme != "https" or port not in (None, 443):
        return None
    if hostname == DEEPSEEK_API_HOST:
        return "deepseek"
    if hostname == OPENAI_API_HOST:
        return "openai"
    return None


def validate_http_endpoint(
    base_url: object,
    *,
    label: str = "provider",
    allow_insecure_internal_env: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ValidatedEndpoint:
    """Validate and canonicalize one OpenAI-compatible base URL.

    Cleartext is always allowed on actual loopback.  A caller may name a
    dedicated opt-in environment variable for private/service-network HTTP;
    setting it never permits a public IP or public multi-label DNS name.
    Userinfo, query strings, and fragments are rejected rather than merely
    redacted, keeping request routing identical to its recorded identity.
    """

    if (
        type(base_url) is not str
        or not base_url
        or base_url != base_url.strip()
        or "\\" in base_url
        or any(character.isspace() or ord(character) < 32 for character in base_url)
    ):
        raise EndpointPolicyError(f"{label} endpoint is malformed")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise EndpointPolicyError(f"{label} endpoint is malformed") from exc
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise EndpointPolicyError(
            f"{label} endpoint must be an absolute http:// or https:// URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise EndpointPolicyError(
            f"{label} endpoint must not contain URL userinfo; pass credentials separately"
        )
    if parsed.query or parsed.fragment:
        raise EndpointPolicyError(
            f"{label} endpoint must not contain query parameters or a fragment"
        )
    if port is not None and port <= 0:
        raise EndpointPolicyError(f"{label} endpoint has an invalid port")

    try:
        hostname, address = _canonical_host(parsed.hostname)
    except EndpointPolicyError as exc:
        raise EndpointPolicyError(f"{label} endpoint host is malformed") from exc
    loopback, internal = _host_scope(hostname, address)

    if scheme == "http" and not loopback:
        opted_in = False
        if allow_insecure_internal_env is not None:
            opted_in = _strict_env_flag(
                allow_insecure_internal_env, environ=environ
            )
        if not opted_in:
            suffix = (
                f"; set {allow_insecure_internal_env}=1 only for an isolated "
                "internal service network"
                if allow_insecure_internal_env is not None and internal
                else ""
            )
            raise EndpointPolicyError(
                f"plaintext HTTP {label} endpoints are allowed only on loopback{suffix}"
            )
        if not internal:
            raise EndpointPolicyError(
                f"{allow_insecure_internal_env} does not permit public HTTP {label} endpoints"
            )

    host_for_url = (
        f"[{hostname}]"
        if address is not None and address.version == 6
        else hostname
    )
    if port is not None:
        host_for_url = f"{host_for_url}:{port}"
    path = parsed.path.rstrip("/")
    normalized = urlunsplit((scheme, host_for_url, path, "", ""))
    return ValidatedEndpoint(
        url=normalized,
        scheme=scheme,
        hostname=hostname,
        port=port,
        is_loopback=loopback,
        is_internal_service=internal,
        official_provider=_official_provider(scheme, hostname, port),
    )


def endpoint_transport_security(endpoint: ValidatedEndpoint) -> str:
    """Return the bounded transport posture of an already-validated endpoint."""

    if endpoint.scheme == "https":
        return TRANSPORT_SECURITY_HTTPS
    if endpoint.scheme == "http" and endpoint.is_loopback:
        return TRANSPORT_SECURITY_LOOPBACK_HTTP
    if endpoint.scheme == "http" and endpoint.is_internal_service:
        return TRANSPORT_SECURITY_INTERNAL_HTTP
    raise EndpointPolicyError("validated endpoint has no supported transport posture")


def _reject_credential_shaped_endpoint_path(endpoint: ValidatedEndpoint) -> None:
    """Reject routes whose commitment would make a small secret guessable.

    A digest is an integrity commitment, not encryption.  Repeated decoding is
    intentional: URL stacks and reverse proxies may decode at more than one
    layer, so ``api%252dkey`` must not evade the same policy as ``api-key``.
    Every path accepted here is treated as public routing metadata by the
    durable identity contract; credentials belong in the separate auth field.
    """

    path = urlsplit(endpoint.url).path
    for _ in range(4):
        decoded = unquote(path)
        if decoded == path:
            break
        path = decoded
    if unquote(path) != path or re.search(r"%[0-9a-fA-F]{2}", path):
        raise EndpointPolicyError("provider endpoint path is ambiguously encoded")
    normalized = unicodedata.normalize("NFKC", path).casefold()
    for segment in normalized.split("/"):
        compact = re.sub(r"[^a-z0-9]+", "", segment)
        words = re.findall(r"[a-z0-9]+", segment)
        if not compact:
            continue
        if (
            compact in _CREDENTIAL_PATH_ROOTS
            or any(word in _CREDENTIAL_PATH_ROOTS for word in words)
            or ("sk" in words and len(words) > 1)
            or any(root in compact for root in (
                "apikey", "accesstoken", "authtoken", "bearertoken",
                "clientsecret", "credential", "password", "secretkey",
            ))
            or re.fullmatch(r"sk[a-z0-9]{2,}", compact) is not None
        ):
            raise EndpointPolicyError(
                "provider endpoint path is credential-shaped; pass credentials separately"
            )


def _reject_credential_shaped_endpoint_host(endpoint: ValidatedEndpoint) -> None:
    """Reject public origins whose DNS labels look like embedded credentials."""

    # IP literals have no operator-controlled DNS label to classify.  For DNS
    # names, run each exact ASCII label through the same bounded public-label
    # policy used by producer attestations.  Ordinary service labels such as
    # ``secret-prod-2026`` remain valid, while ``sk-live-...`` and
    # ``api-key-...`` cannot be persisted as a supposedly safe origin.
    try:
        ipaddress.ip_address(endpoint.hostname.rstrip("."))
    except ValueError:
        for component in endpoint.hostname.rstrip(".").split("."):
            try:
                validate_public_attestation(
                    component, label="provider endpoint host label", max_bytes=63,
                )
            except ValueError as exc:
                raise EndpointPolicyError(
                    "provider endpoint host is credential-shaped"
                ) from exc


def secret_free_endpoint_identity(
    base_url: object,
    *,
    label: str = "provider",
    allow_insecure_internal_env: str | None = None,
) -> dict[str, str]:
    """Return an exact endpoint commitment without retaining its route.

    Provider routes may contain opaque tenant or deployment identifiers.  They
    are request material but are not safe durable metadata.  The origin is
    sufficient for diagnostics/provider classification and the digest keeps
    distinct routes cryptographically distinct.
    """

    endpoint = validate_http_endpoint(
        base_url,
        label=label,
        allow_insecure_internal_env=allow_insecure_internal_env,
    )
    _reject_credential_shaped_endpoint_host(endpoint)
    _reject_credential_shaped_endpoint_path(endpoint)
    host = endpoint.hostname
    if ":" in host:
        host = f"[{host}]"
    if endpoint.port is not None:
        host = f"{host}:{endpoint.port}"
    origin = urlunsplit((endpoint.scheme, host, "", "", ""))
    return {
        "endpoint_origin": origin,
        "endpoint_sha256": (
            "sha256:" + hashlib.sha256(endpoint.url.encode("utf-8")).hexdigest()
        ),
    }


def validate_recorded_endpoint_origin(
    origin: object, *, label: str = "embedding",
) -> tuple[ValidatedEndpoint, str]:
    """Validate one canonical, credential-free origin from durable evidence."""

    endpoint = validate_http_endpoint(
        origin,
        label=label,
        allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
        environ={EMBEDDING_INTERNAL_HTTP_ENV: "1"},
    )
    parsed = urlsplit(endpoint.url)
    if parsed.path or endpoint.url != origin:
        raise EndpointPolicyError(f"{label} origin is not canonical")
    return endpoint, endpoint_transport_security(endpoint)


def validate_recorded_embedding_endpoint(
    base_url: object,
    *,
    transport_security: object,
    label: str = "embedding",
) -> ValidatedEndpoint:
    """Validate a self-attesting artifact endpoint without ambient state.

    This function validates inert benchmark evidence; it does not authorize a
    live request.  Runtime clients independently require the real
    ``HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP`` environment opt-in.  For
    an artifact, the bounded recorded posture supplies only enough context to
    parse an internal HTTP URL, after which the URL-derived posture must match
    exactly.  A public HTTP host remains invalid for every posture.
    """

    if (
        not isinstance(transport_security, str)
        or transport_security not in _NETWORK_TRANSPORT_SECURITY
    ):
        raise EndpointPolicyError(
            f"{label} transport_security is missing or unsupported"
        )
    recorded_internal_opt_in = (
        transport_security == TRANSPORT_SECURITY_INTERNAL_HTTP
    )
    endpoint = validate_http_endpoint(
        base_url,
        label=label,
        allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
        # Deliberately self-contained: artifact validity cannot depend on the
        # validator process's environment.
        environ={
            EMBEDDING_INTERNAL_HTTP_ENV: (
                "1" if recorded_internal_opt_in else "0"
            )
        },
    )
    if endpoint_transport_security(endpoint) != transport_security:
        raise EndpointPolicyError(
            f"{label} transport_security does not match its endpoint"
        )
    return endpoint


def safe_endpoint_label(base_url: object, *, label: str = "provider") -> str:
    """Return a diagnostic label that cannot contain URL credentials/query data."""

    try:
        if not isinstance(base_url, str):
            raise ValueError
        parsed = urlsplit(base_url.strip())
        port = parsed.port
        scheme = parsed.scheme.casefold()
        if scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        hostname, address = _canonical_host(parsed.hostname)
        # Diagnostic rendering is itself a persistence surface (doctor output,
        # service logs, and benchmark failure records).  Apply the same public
        # host-label policy as durable endpoint identity before reproducing any
        # operator-controlled DNS bytes.
        _reject_credential_shaped_endpoint_host(ValidatedEndpoint(
            url=urlunsplit((scheme, hostname, "", "", "")),
            scheme=scheme,
            hostname=hostname,
            port=port,
            is_loopback=False,
            is_internal_service=False,
            official_provider=None,
        ))
        host = f"[{hostname}]" if address is not None and address.version == 6 else hostname
        if port is not None:
            host = f"{host}:{port}"
        # Userinfo, the complete route, query, and fragment are omitted even
        # when malformed input reaches diagnostics before normal validation.
        return urlunsplit((scheme, host, "", "", ""))
    except (EndpointPolicyError, TypeError, ValueError, OverflowError):
        return f"<invalid {label} URL>"


def _explicit_key(value: object, *, source: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise EndpointPolicyError(f"{source} must be a non-empty string")
    return value


def resolve_llm_api_key(
    base_url: object,
    *,
    explicit_key: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[ValidatedEndpoint, str]:
    """Resolve an LLM credential without crossing provider origins."""

    source = os.environ if environ is None else environ
    endpoint = validate_http_endpoint(base_url, label="LLM", environ=source)
    key = _explicit_key(explicit_key, source="explicit LLM API key")
    if key is None:
        key = _explicit_key(
            source.get("HYMEM_LLM_API_KEY"), source="HYMEM_LLM_API_KEY"
        )
    if key is None and endpoint.official_provider == "deepseek":
        key = _explicit_key(
            source.get("DEEPSEEK_API_KEY"), source="DEEPSEEK_API_KEY"
        )
    if key is None and endpoint.official_provider == "openai":
        key = _explicit_key(
            source.get("OPENAI_API_KEY"), source="OPENAI_API_KEY"
        )
    if key is None:
        raise EnvironmentError(
            "No LLM API key is authorized for the configured endpoint. Set "
            "HYMEM_LLM_API_KEY or pass an explicit key; provider keys are "
            "inherited only by their exact official HTTPS origin."
        )
    return endpoint, key


def resolve_embedding_api_key(
    base_url: object,
    *,
    explicit_key: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[ValidatedEndpoint, str]:
    """Resolve an embedding-only credential and internal-HTTP posture."""

    source = os.environ if environ is None else environ
    endpoint = validate_http_endpoint(
        base_url,
        label="embedding",
        allow_insecure_internal_env=EMBEDDING_INTERNAL_HTTP_ENV,
        environ=source,
    )
    key = _explicit_key(explicit_key, source="explicit embedding API key")
    if key is None:
        key = _explicit_key(
            source.get("HYMEM_EMBEDDING_API_KEY"),
            source="HYMEM_EMBEDDING_API_KEY",
        )
    # An insecure private-network opt-in must never make a cloud/provider key
    # eligible.  Local OpenAI-compatible servers conventionally require a
    # syntactic bearer value even when authentication is disabled.
    internal_plaintext = (
        endpoint.scheme == "http"
        and endpoint.is_internal_service
        and not endpoint.is_loopback
    )
    if key is None and not internal_plaintext and endpoint.official_provider == "openai":
        key = _explicit_key(
            source.get("OPENAI_API_KEY"), source="OPENAI_API_KEY"
        )
    if key is None and (endpoint.is_loopback or internal_plaintext):
        key = "local"
    if key is None:
        raise EnvironmentError(
            "No embedding API key is authorized for the configured endpoint. "
            "Set HYMEM_EMBEDDING_API_KEY or pass an explicit key; "
            "OPENAI_API_KEY is inherited only by the exact official HTTPS "
            "OpenAI origin."
        )
    return endpoint, key


# Captured by the owning module itself, before a rolling deploy can replace
# this file while an older process continues executing its loaded functions.
ENDPOINT_POLICY_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)
