"""Secret-free identity for the Phase-1 LLM producer and wire request.

The extraction contract identifies prompts, parsing, validation, and recovery.
It deliberately does not identify the client that answers those prompts.  A
durable Phase-1 cache hit requires both halves: otherwise a store built with a
cheap screening model can be silently presented as output from a later target
model.

Clients maintained by HyMem expose an explicit, credential-free declaration.
Third-party clients may expose the same declaration through
``phase1_producer_declaration()``.  Clients without one receive an opaque
process-instance identity: it permits idempotence while that exact object is
alive, but cannot authorize reuse after a restart and is explicitly labelled
non-exact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import dis
import functools
import json
import re
import secrets
import sqlite3
import threading
import types
import sys
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from hymem.contrib.endpoint_policy import (
    secret_free_endpoint_identity,
    validate_public_attestation,
    validate_http_endpoint,
)
from hymem.extraction.contract import (
    EXTRACTION_CACHE_SCHEMA,
    extraction_cache_key,
)


PHASE1_PRODUCER_DECLARATION_SCHEMA = "hymem-phase1-producer-declaration-v3"
AGGREGATION_PRODUCER_DECLARATION_SCHEMA = (
    "hymem-aggregation-producer-declaration-v1"
)
PHASE1_PRODUCER_BINDING_SCHEMA = "hymem-phase1-producer-binding-v1"
PHASE1_GENERATION_SCHEMA = "hymem-phase1-generation-v1"
PHASE1_NESTED_IDENTITY_SCHEMA = "hymem-phase1-nested-identity-sha256-v1"
_GENERATION_KEY_RE = re.compile(
    rf"{re.escape(PHASE1_GENERATION_SCHEMA)}:[0-9a-f]{{64}}\Z"
)
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9_.:/@+-]{1,256}\Z")
_SENSITIVE_KEY_COMPONENTS = frozenset({
    "auth", "authorization", "bearer", "cookie", "credential",
    "credentials", "key", "password", "secret", "session", "token",
})
_NON_SECRET_TOKEN_FIELDS = frozenset({"max_tokens", "max_tokens_source"})
_SENSITIVE_COMPOUND_KEYS = frozenset({
    "accesstoken", "apikey", "authtoken", "bearertoken", "clientsecret",
    "privatekey", "secretkey", "sessioncookie", "sessionkey", "xauthtoken",
})
_SENSITIVE_KEY_ROOTS = (
    "auth", "bearer", "cookie", "credential", "passwd", "password",
    "secret", "session", "token",
)
_MAX_DECLARATION_BYTES = 32_768
_MAX_EXTRACTION_CACHE_KEY_BYTES = 8_192
_module_digest_lock = threading.Lock()
_module_digest_cache: dict[tuple[tuple[object, ...], ...], str] = {}
_SUPPORTED_EXTRACTION_CACHE_SCHEMAS = frozenset({
    "hymem-extraction-cache-v1",
    EXTRACTION_CACHE_SCHEMA,
})
_LOADED_CODE_RUNTIME_DOMAIN = {
    "implementation": sys.implementation.name,
    "cache_tag": sys.implementation.cache_tag,
    "version": list(sys.version_info[:5]),
    "bytecode_magic": importlib.util.MAGIC_NUMBER.hex(),
}


@dataclass(frozen=True)
class Phase1ProducerDeclaration:
    """Explicit stable identity supplied by a producer implementation.

    The typed public metadata is persisted; caller-controlled
    ``effective_request`` and ``retry_policy`` mappings are canonicalized and
    retained only as stable digests. They must still describe request behavior,
    never credentials. ``endpoint`` is canonicalized and rejects URL userinfo,
    query strings, fragments, and unsafe cleartext origins.
    """

    client_id: str
    implementation: str
    model: str
    endpoint: str | None
    effective_request: Mapping[str, Any]
    retry_policy: Mapping[str, Any]


class _ProducerDeclarationCarrier:
    """Internal adapter for canonicalizing already-derived typed metadata."""

    def __init__(self, declaration: Phase1ProducerDeclaration) -> None:
        self.declaration = declaration

    def phase1_producer_declaration(self) -> Phase1ProducerDeclaration:
        return self.declaration

    def aggregation_producer_declaration(self) -> Phase1ProducerDeclaration:
        return self.declaration

    def memory_producer_declaration(self) -> Phase1ProducerDeclaration:
        return self.declaration


# Guard inspection/registry allocation can synchronously collect a different
# client or proxy. Its weakref cleanup re-enters this same registry lock on the
# collecting thread; a plain Lock would deadlock real provider dispatch. Keep
# cross-thread exclusion, but permit that same-thread cleanup. Registry reads
# retain their entry locally and cleanup checks the exact weakref before
# deletion; no operation iterates a live registry while callbacks can mutate it.
_unknown_lock = threading.RLock()
# Weak-referenceable clients do not stay alive merely because identity was
# requested.  A bounded strong fallback is unavoidable for classes that opt
# out of weak references; eviction only makes their old cache fail closed.
_unknown_instances: dict[int, tuple[weakref.ReferenceType[object], str]] = {}
_unknown_nonweak: OrderedDict[int, tuple[object, str]] = OrderedDict()
_unknown_generation_keys: dict[int, set[str]] = {}
_runtime_authorized_inexact_generations: set[str] = set()
# Short-lived, maintained execution-control wrappers (attempt counting,
# deadlines, heartbeats) must retain the producer identity of the real client
# they transparently invoke.  A weak registry avoids exposing a second
# caller-supplied identity parameter at the extraction boundary, where client B
# could otherwise be mislabeled with client A's binding.
_producer_proxies: dict[
    int, tuple[
        weakref.ReferenceType[object], object, str | None, object | None,
        type[object], tuple[object, ...], tuple[tuple[str, object], ...],
    ]
] = {}
_PRODUCER_PROXY_DISPATCH_NAMES = (
    "embed", "complete", "model", "dim", "backend", "quality",
    "network_free", "close", "__getattribute__", "__getattr__",
    "_embed_serialized", "_validated_vector", "_failure",
)
_PRODUCER_PROXY_POLICY_NAMES = (
    "expected_dimension", "_initial_model", "_deadline", "_heartbeat",
    "_error_type", "_owned_transport", "_owned_transport_target",
    "_lifecycle_lock",
)


def _proxy_policy_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return ("value", type(value), value)
    return ("identity", type(value), id(value))


def _maintained_proxy_type(proxy_type: type[object]) -> bool:
    """Allow transparent authority only for exact in-tree wrapper classes.

    The registration helper is importable Python infrastructure, not an
    authority capability.  A custom proxy calling it must never inherit an
    exact producer's durable identity.  The allowlist is keyed by exact class
    objects and module-initialized descriptor guards; forgeable class names,
    module labels, and source paths never confer authority.
    """

    try:
        from hymem.deadline import (
            _DEADLINE_PROXY_DISPATCH_NAMES,
            _DEADLINE_PROXY_ORIGINAL_GUARDS,
            deadline_proxy_support_integrity,
        )
        from hymem.extraction import embeddings as embeddings_module
        from hymem.extraction.embeddings import (
            PinnedEmbeddingClient,
            _CACHED_EMBEDDING_GUARD_NAMES,
            _CACHED_EMBEDDING_ORIGINAL_GUARD,
            _PINNED_EMBEDDING_GUARD_NAMES,
            _PINNED_EMBEDDING_ORIGINAL_GUARD,
        )
        CachedEmbeddingClient = (
            embeddings_module._CACHED_EMBEDDING_MAINTAINED_CLASS
        )
        PinnedEmbeddingClient = (
            embeddings_module._PINNED_EMBEDDING_MAINTAINED_CLASS
        )

        if proxy_type in _DEADLINE_PROXY_ORIGINAL_GUARDS:
            if not deadline_proxy_support_integrity():
                return False
            names = _DEADLINE_PROXY_DISPATCH_NAMES
            expected = _DEADLINE_PROXY_ORIGINAL_GUARDS[proxy_type]
        elif proxy_type is CachedEmbeddingClient:
            names = _CACHED_EMBEDDING_GUARD_NAMES
            expected = _CACHED_EMBEDDING_ORIGINAL_GUARD
        elif proxy_type is PinnedEmbeddingClient:
            names = _PINNED_EMBEDDING_GUARD_NAMES
            expected = _PINNED_EMBEDDING_ORIGINAL_GUARD
        else:
            # Runner types cannot be imported while producer itself is being
            # imported, but registration happens only after runner module
            # initialization.  Its mapping is keyed by the exact class object.
            import sys

            runner = sys.modules.get("hymem.dreaming.runner")
            guards = getattr(runner, "_RUNNER_PROXY_ORIGINAL_GUARDS", {})
            if proxy_type not in guards:
                return False
            names = getattr(runner, "_RUNNER_PROXY_DISPATCH_NAMES", ())
            expected = guards[proxy_type]
        return tuple(
            inspect.getattr_static(proxy_type, name, None) for name in names
        ) == expected
    except (ImportError, KeyError, TypeError):
        return False
_MAX_NONWEAK_UNKNOWN_CLIENTS = 1024
_MAX_INEXACT_GENERATIONS_PER_CLIENT = 256


def _reject_credential_shaped_endpoint_path(endpoint: str) -> None:
    """Keep aggregation declarations from persisting credential-like routes."""

    for segment in unquote(urlsplit(endpoint).path).casefold().split("/"):
        compact = re.sub(r"[^a-z0-9]+", "", segment)
        if (
            segment.startswith("sk-")
            or any(root in compact for root in (
                "apikey", "accesstoken", "authtoken", "bearertoken",
                "clientsecret", "credential", "password", "secretkey",
            ))
        ):
            raise ValueError(
                "aggregation producer endpoint path is credential-shaped"
            )


@dataclass(frozen=True, slots=True, eq=False)
class _CodeIdentity:
    """Identity key that never hashes/compares caller-supplied constants.

    Python code equality may consult objects inside synthetic ``co_consts``.
    Using the code object's identity avoids executing their hash/equality
    hooks before the cache's immutability check. The strong reference also
    prevents object-id recycling while this bounded cache entry exists.
    """

    code: types.CodeType

    def __hash__(self) -> int:
        return id(self.code)

    def __eq__(self, other: object) -> bool:
        return type(other) is _CodeIdentity and self.code is other.code


def _immutable_code_constant(
    value: object, *, depth: int = 0, verified: set[int] | None = None,
) -> bool:
    """Only genuine immutable built-ins may enter the code-piece cache.

    ``CodeType.replace`` accepts arbitrary objects in ``co_consts``. A code
    object is therefore not sufficient proof that its entire constant graph
    is immutable, even when that code object happens to be hashable.
    """

    # Cache eligibility must not exceed the serializer's existing traversal
    # bound or expand shared constant DAGs exponentially. Refuse caching on
    # unusually deep input; the ordinary depth-limited serializer still runs.
    if depth > 12:
        return False
    if value is None or type(value) in (bool, int, str, float, bytes):
        return True
    if type(value) not in (tuple, frozenset, types.CodeType):
        return False
    if verified is None:
        verified = set()
    identity = id(value)
    if identity in verified:
        return True
    items = value.co_consts if isinstance(value, types.CodeType) else value
    if not all(
        _immutable_code_constant(item, depth=depth + 1, verified=verified)
        for item in items
    ):
        return False
    verified.add(identity)
    return True


def _uncached_loaded_code_sha256(code: types.CodeType) -> str:
    constants = list(code.co_consts)
    # A docstring has no execution effect and historically did not move the
    # commitment.  Loaded bytecode, names, defaults, and nested code do.
    if constants and isinstance(constants[0], str):
        constants[0] = ("docstring-omitted",)
    record = {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "consts": [_loaded_identity_value(value) for value in constants],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
    }
    encoded = json.dumps(
        record, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "loaded-code-sha256-v1:" + hashlib.sha256(encoded).hexdigest()


@functools.lru_cache(maxsize=1024)
def _immutable_loaded_code_sha256(identity: _CodeIdentity) -> str:
    code = identity.code
    if not _immutable_code_constant(code):
        raise TypeError("code constants contain mutable or opaque state")
    return _uncached_loaded_code_sha256(code)


def _loaded_code_record(code: types.CodeType) -> str:
    """Commit code pieces once, while inspecting function state every time.

    Cached values are immutable digests, not caller-mutable record trees.
    Defaults, closures, classes, global containers, and module bindings are
    deliberately outside this cache. Replacing ``function.__code__`` selects
    a new immutable key; unusual mutable code constants bypass it entirely.
    """

    try:
        return _immutable_loaded_code_sha256(_CodeIdentity(code))
    except TypeError:
        return _uncached_loaded_code_sha256(code)


@functools.lru_cache(maxsize=1024)
def _immutable_code_global_names(identity: _CodeIdentity) -> tuple[str, ...]:
    code = identity.code
    # The same immutability check also excludes hashable mutable constants.
    if not _immutable_code_constant(code):
        raise TypeError("code constants contain mutable or opaque state")
    return _uncached_code_global_names(code)


def _uncached_code_global_names(code: types.CodeType) -> tuple[str, ...]:
    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and isinstance(instruction.argval, str)
    }
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_code_global_names(constant))
    return tuple(sorted(names))


def _code_global_names(code: types.CodeType) -> tuple[str, ...]:
    try:
        return _immutable_code_global_names(_CodeIdentity(code))
    except TypeError:
        return _uncached_code_global_names(code)


def _loaded_identity_value(value: object, *, depth: int = 0) -> object:
    if depth > 12:
        return ["depth-limit", type(value).__module__, type(value).__qualname__]
    if value is None or isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, types.CodeType):
        return ["code", _loaded_code_record(value)]
    if isinstance(value, types.FunctionType):
        closure: list[object] = []
        for cell in value.__closure__ or ():
            try:
                item = cell.cell_contents
            except ValueError:
                item = ("empty-cell",)
            closure.append(_loaded_identity_value(item, depth=depth + 1))
        return [
            "function", value.__module__, value.__qualname__,
            _loaded_code_record(value.__code__),
            _loaded_identity_value(value.__defaults__, depth=depth + 1),
            _loaded_identity_value(value.__kwdefaults__, depth=depth + 1),
            closure,
        ]
    if isinstance(value, (staticmethod, classmethod)):
        return [type(value).__name__, _loaded_identity_value(
            value.__func__, depth=depth + 1,
        )]
    if isinstance(value, property):
        return [
            "property",
            _loaded_identity_value(value.fget, depth=depth + 1),
            _loaded_identity_value(value.fset, depth=depth + 1),
            _loaded_identity_value(value.fdel, depth=depth + 1),
        ]
    if isinstance(value, type):
        members: list[list[object]] = []
        for name, member in sorted(vars(value).items()):
            if name in {"__dict__", "__doc__", "__module__", "__weakref__"}:
                continue
            if (
                isinstance(member, (
                    types.FunctionType, staticmethod, classmethod, property,
                ))
                or name == "__annotations__"
                or name.isupper()
            ):
                members.append([
                    name, _loaded_identity_value(member, depth=depth + 1),
                ])
        return [
            "class", value.__module__, value.__qualname__,
            [[base.__module__, base.__qualname__] for base in value.__bases__],
            members,
        ]
    if isinstance(value, Mapping):
        items = [
            [
                _loaded_identity_value(key, depth=depth + 1),
                _loaded_identity_value(item, depth=depth + 1),
            ]
            for key, item in value.items()
        ]
        items.sort(key=lambda item: json.dumps(
            item[0], ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ))
        return ["mapping", items]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [
            _loaded_identity_value(item, depth=depth + 1) for item in value
        ]]
    if isinstance(value, (set, frozenset)):
        items = [
            _loaded_identity_value(item, depth=depth + 1) for item in value
        ]
        items.sort(key=lambda item: json.dumps(
            item, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ))
        return [type(value).__name__, items]
    if isinstance(value, re.Pattern):
        return ["regex", value.pattern, value.flags]
    return ["typed-opaque", type(value).__module__, type(value).__qualname__]


def canonical_callable_sha256(*callables: Callable[..., object]) -> str:
    """Hash live loaded executable state, never mutable source files.

    Filename, line-table, and source layout are excluded.  This prevents an
    OLD long-running process from minting a NEW deployment's producer key
    after package files are atomically replaced during a rolling deploy.
    """

    payload = json.dumps(
        {
            "runtime": _LOADED_CODE_RUNTIME_DOMAIN,
            "callables": [
                _loaded_identity_value(function) for function in callables
            ],
        },
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def exact_callable_sha256(*callables: object) -> str:
    """Hash a closed callable state or reject unsupported execution state.

    This stricter form is used when a custom implementation asks for durable
    authority.  Plain functions and static/class methods are supported;
    arbitrary descriptors and opaque defaults, closures, or referenced module
    globals are rejected instead of being reduced to a forgeable type label.
    Referenced JSON-like global state is included by value.
    """

    active: set[int] = set()

    def value_record(value: object, *, depth: int = 0) -> object:
        if depth > 12:
            raise ValueError("callable identity state is too deeply nested")
        if value is None or type(value) in (bool, int, str):
            return [type(value).__name__, value]
        if type(value) is float:
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("callable identity contains a non-finite value")
            return ["float", value.hex()]
        if type(value) is bytes:
            return ["bytes", value.hex()]
        if isinstance(value, (staticmethod, classmethod)):
            return [type(value).__name__, function_record(
                value.__func__, depth=depth + 1,
            )]
        if isinstance(value, types.FunctionType):
            return function_record(value, depth=depth + 1)
        # Mutable defaults/closures/globals cannot be made ABA-safe by a
        # before/after content hash.  Exact custom execution may reference
        # only immutable built-in containers; maintained stateful clients use
        # their own versioned/frozen runtime guards instead.
        if type(value) in (dict, list, set):
            raise ValueError("callable identity contains mutable execution state")
        if type(value) is tuple:
            identity = id(value)
            if identity in active:
                raise ValueError("callable identity contains a cycle")
            active.add(identity)
            try:
                records = [
                    value_record(item, depth=depth + 1) for item in value
                ]
            finally:
                active.remove(identity)
            return [type(value).__name__, records]
        if type(value) is frozenset:
            records = [value_record(item, depth=depth + 1) for item in value]
            records.sort(key=lambda item: json.dumps(
                item, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            ))
            return [type(value).__name__, records]
        if isinstance(value, re.Pattern):
            return ["regex", value.pattern, value.flags]
        raise ValueError("callable identity contains unsupported opaque state")

    def function_record(
        function: object, *, depth: int = 0,
    ) -> object:
        if not isinstance(function, types.FunctionType):
            raise ValueError("exact callable must be a plain function descriptor")
        identity = id(function)
        if identity in active:
            return ["recursive-function", function.__module__, function.__qualname__]
        active.add(identity)
        try:
            closure = []
            freevars = function.__code__.co_freevars
            for index, cell in enumerate(function.__closure__ or ()):
                try:
                    cell_value = cell.cell_contents
                except ValueError as exc:
                    raise ValueError("callable identity has an empty closure") from exc
                freevar_name = freevars[index] if index < len(freevars) else None
                if freevar_name == "__class__" and isinstance(cell_value, type):
                    closure.append([
                        "class-cell", cell_value.__module__, cell_value.__qualname__,
                        [
                            [base.__module__, base.__qualname__]
                            for base in cell_value.__mro__
                        ],
                    ])
                else:
                    closure.append(value_record(cell_value, depth=depth + 1))
            referenced_globals = []
            namespace = function.__globals__
            for name in _code_global_names(function.__code__):
                if name not in namespace or name == "__builtins__":
                    continue
                global_value = namespace[name]
                # A function's own recursive name is already represented by
                # its loaded code record.
                if global_value is function:
                    continue
                referenced_globals.append([
                    name, value_record(global_value, depth=depth + 1),
                ])
            return [
                "exact-function", function.__module__, function.__qualname__,
                _loaded_code_record(function.__code__),
                value_record(function.__defaults__, depth=depth + 1),
                value_record(function.__kwdefaults__, depth=depth + 1),
                closure, referenced_globals,
            ]
        finally:
            active.remove(identity)

    records: list[object] = []
    for callable_object in callables:
        if isinstance(callable_object, (staticmethod, classmethod)):
            callable_object = callable_object.__func__
        records.append(function_record(callable_object))
    payload = json.dumps(
        {"runtime": _LOADED_CODE_RUNTIME_DOMAIN, "callables": records},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_module_sha256(*modules: object) -> str:
    """Hash the loaded code/constants of modules without reading disk."""

    normalized: list[object] = []
    for module in modules:
        name = getattr(module, "__name__", None)
        namespace = getattr(module, "__dict__", None)
        if not isinstance(name, str) or not isinstance(namespace, Mapping):
            raise ValueError("trusted loaded module is unavailable")
        members: list[list[object]] = []
        for member_name, value in sorted(namespace.items()):
            defined_here = getattr(value, "__module__", None) == name
            constant = member_name.lstrip("_").isupper()
            if defined_here or constant:
                members.append([
                    member_name, _loaded_identity_value(value),
                ])
        normalized.append(["module", name, members])
    payload = json.dumps(
        {"runtime": _LOADED_CODE_RUNTIME_DOMAIN, "modules": normalized},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_module_slice_sha256(module: object, *roots: str) -> str:
    """Hash a transitive slice of live loaded module state.

    Only module-owned globals reached by the requested roots are traversed.
    This mirrors the former AST slice without reading mutable package files at
    identity time and without coupling Phase-1 to unrelated prompt families.
    Imported helper aliases remain guarded by their consuming owner module.
    """

    module_name = getattr(module, "__name__", None)
    namespace = getattr(module, "__dict__", None)
    if not isinstance(module_name, str) or type(namespace) is not dict:
        raise ValueError("trusted loaded module is unavailable")
    missing = sorted(set(roots) - set(namespace))
    if missing:
        raise ValueError("loaded module roots are absent: " + ",".join(missing))

    def code_objects(value: object) -> list[types.CodeType]:
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if isinstance(value, types.FunctionType):
            return [value.__code__]
        if isinstance(value, property):
            return [
                accessor.__code__
                for accessor in (value.fget, value.fset, value.fdel)
                if isinstance(accessor, types.FunctionType)
            ]
        if isinstance(value, type):
            codes: list[types.CodeType] = []
            for member in vars(value).values():
                codes.extend(code_objects(member))
            return codes
        return []

    pending = list(roots)
    selected: dict[str, object] = {}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        value = namespace[name]
        selected[name] = _loaded_identity_value(value)
        for code in code_objects(value):
            for dependency in _code_global_names(code):
                if dependency not in namespace:
                    continue
                dependency_value = namespace[dependency]
                if (
                    getattr(dependency_value, "__module__", module_name)
                    == module_name
                    or dependency.isupper()
                    or dependency.startswith("_")
                ):
                    pending.append(dependency)
    payload = json.dumps(
        {
            "runtime": _LOADED_CODE_RUNTIME_DOMAIN,
            "module": module_name,
            "roots": sorted(set(roots)),
            "bindings": [[name, selected[name]] for name in sorted(selected)],
        },
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object, *, path: str = "declaration") -> Any:
    """Return a bounded JSON value while rejecting credential-shaped state."""

    if value is None or type(value) in (str, bool, int):
        if type(value) is str:
            if "\x00" in value or len(value) > 4096:
                raise ValueError(f"{path} contains malformed text")
            # A URL-shaped custom field cannot smuggle userinfo into the
            # persisted declaration under an innocuous key.
            if "://" in value:
                try:
                    parsed = validate_http_endpoint(value, label="Phase-1 producer")
                except ValueError as exc:
                    raise ValueError(
                        f"{path} contains an unsafe URL"
                    ) from exc
                secret_free_endpoint_identity(parsed.url, label=path)
                return parsed.url
            validate_public_attestation(value, label=path, max_bytes=4096)
        return value
    if type(value) is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key or len(key) > 128:
                raise ValueError(f"{path} contains a malformed key")
            # Split both punctuation-delimited and camelCase names. Without
            # the latter, an otherwise exact custom declaration could persist
            # obvious credentials under ``accessToken``/``clientSecret`` even
            # though ``access_token``/``client_secret`` were rejected.
            separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
            folded = separated.casefold().replace("-", "_")
            components = frozenset(
                part for part in re.split(r"[^a-z0-9]+", folded) if part
            )
            compact = "".join(components)
            credential_shaped = (
                bool(components.intersection(_SENSITIVE_KEY_COMPONENTS))
                or any(root in compact for root in _SENSITIVE_KEY_ROOTS)
                or any(value in compact for value in _SENSITIVE_COMPOUND_KEYS)
                or compact == "key"
                or compact.startswith("key")
                or compact.endswith("key")
            )
            if (
                folded not in _NON_SECRET_TOKEN_FIELDS
                and credential_shaped
            ):
                raise ValueError(
                    f"{path} contains a credential-shaped field"
                )
            result[key] = _canonical_json(item, path=f"{path}.{key}")
        return {key: result[key] for key in sorted(result)}
    if type(value) in (list, tuple):
        if len(value) > 256:
            raise ValueError(f"{path} contains too many items")
        return [
            _canonical_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} contains a non-JSON value")


def _safe_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
        raise ValueError(f"Phase-1 producer {label} is malformed")
    return validate_public_attestation(
        value, label=f"Phase-1 producer {label}", max_bytes=256,
    )


def _nested_identity_digest(value: object, *, label: str) -> dict[str, str]:
    """Hash caller-controlled declaration state before it becomes durable.

    Effective request and retry mappings may contain provider-specific fields,
    so no finite key-name denylist can prove that every nested scalar is safe to
    persist.  Canonicalize the complete mapping (which still rejects obvious
    credential fields and unsafe URLs), bind its exact bytes into the identity,
    and retain only the digest in SQLite, portable exports, status, and benchmark
    receipts.
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"Phase-1 producer {label} must be a mapping")
    canonical = _canonical_json(value, path=label)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_DECLARATION_BYTES:
        raise ValueError(f"Phase-1 producer {label} is too large")
    return {
        "schema": PHASE1_NESTED_IDENTITY_SCHEMA,
        "sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }


def _validate_nested_identity_digest(
    value: object, *, label: str,
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "sha256"}
        or value.get("schema") != PHASE1_NESTED_IDENTITY_SCHEMA
        or not isinstance(value.get("sha256"), str)
        or _DIGEST_RE.fullmatch(str(value["sha256"])) is None
    ):
        raise ValueError(f"Phase-1 producer {label} digest is invalid")
    return {"schema": str(value["schema"]), "sha256": str(value["sha256"])}


def _validate_historical_extraction_cache_key(value: object) -> str:
    """Validate a recorded cache-key envelope without executing today's code.

    The embedded digest identifies the historical extraction implementation.
    Recomputing it with the current implementation would turn an ordinary code
    upgrade into registry corruption and prevent the database from opening.
    Live extraction separately compares this preserved key with today's exact
    :func:`extraction_cache_key` result.
    """

    if not isinstance(value, str) or not value:
        raise ValueError("Phase-1 extraction cache key is invalid")
    if "\x00" in value or len(value.encode("utf-8")) > _MAX_EXTRACTION_CACHE_KEY_BYTES:
        raise ValueError("Phase-1 extraction cache key is malformed")
    schema, separator, remainder = value.partition(":")
    if not separator or schema not in _SUPPORTED_EXTRACTION_CACHE_SCHEMAS:
        raise ValueError("Phase-1 extraction cache key schema is unsupported")
    digest, separator, prompt_version = remainder.partition(":")
    if (
        not separator
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or not prompt_version
    ):
        raise ValueError("Phase-1 extraction cache key is malformed")
    return value


def _declared_producer(
    client: object, *, declaration_hook: str = "phase1_producer_declaration",
) -> dict[str, Any] | None:
    # Maintained clients receive durable authority only while their frozen
    # execution dispatch and state contract remain intact.  In particular an
    # instance-shadowed declaration hook must not bypass the class's own
    # validator by returning an old, still-well-shaped declaration.
    try:
        from hymem.extraction import llm as llm_module
    except ImportError:
        llm_module = None  # type: ignore[assignment]
    StubLLMClient = (
        getattr(llm_module, "_STUB_LLM_MAINTAINED_CLASS", None)
        if llm_module is not None else None
    )
    if isinstance(StubLLMClient, type) and type(client) is StubLLMClient:
        stub_guard = getattr(llm_module, "_STUB_LLM_INTEGRITY_FUNCTION", None)
        if (
            getattr(llm_module, "maintained_stub_llm_integrity", None)
            is not stub_guard
            or not callable(stub_guard)
            or not stub_guard(client)
        ):
            raise ValueError("maintained stub LLM integrity changed")
    try:
        from hymem.contrib import openai_client as openai_llm_module
    except ImportError:
        openai_llm_module = None  # type: ignore[assignment]
    OpenAICompatibleClient = (
        getattr(openai_llm_module, "_OPENAI_LLM_MAINTAINED_CLASS", None)
        if openai_llm_module is not None else None
    )
    if (
        isinstance(OpenAICompatibleClient, type)
        and type(client) is OpenAICompatibleClient
    ):
        openai_guard = getattr(
            openai_llm_module, "_OPENAI_LLM_INTEGRITY_FUNCTION", None,
        )
        if (
            getattr(openai_llm_module, "maintained_openai_llm_integrity", None)
            is not openai_guard
            or not callable(openai_guard)
            or not openai_guard(client)
        ):
            raise ValueError("maintained OpenAI LLM integrity changed")
    # Resolve declarations statically before invoking them.  An arbitrary
    # transparent wrapper may delegate unknown attributes to its wrapped
    # object via ``__getattr__``; that is execution convenience, not authority
    # to inherit the wrapped producer's durable identity.  Maintained wrappers
    # are handled separately by the private producer-proxy registry.
    try:
        inspect.getattr_static(client, declaration_hook)
    except AttributeError:
        return None
    try:
        factory = getattr(client, declaration_hook)
    except (AttributeError, TypeError):
        return None
    if not callable(factory):
        raise ValueError(f"{declaration_hook} must be callable")
    try:
        declaration = factory()
    except NotImplementedError:
        return None
    if not isinstance(declaration, Phase1ProducerDeclaration):
        raise ValueError(
            f"{declaration_hook} must return Phase1ProducerDeclaration"
        )
    raw = asdict(declaration)
    client_id = _safe_identifier(raw["client_id"], label="client_id")
    implementation = _safe_identifier(
        raw["implementation"], label="implementation"
    )
    model = _safe_identifier(raw["model"], label="model")
    endpoint = raw["endpoint"]
    if endpoint is not None:
        endpoint = validate_http_endpoint(
            endpoint, label="Phase-1 producer"
        ).url
        # A Phase-1 declaration historically retains its canonical request
        # endpoint.  Even before the artifact-v2 origin/digest projection,
        # never let credential-shaped route bytes enter that durable binding.
        secret_free_endpoint_identity(endpoint, label="Phase-1 producer")
    canonical: dict[str, Any] = {
        "client_id": client_id,
        "implementation": implementation,
        "model": model,
        "effective_request": _nested_identity_digest(
            raw["effective_request"], label="effective_request"
        ),
        "retry_policy": _nested_identity_digest(
            raw["retry_policy"], label="retry_policy"
        ),
    }
    if endpoint is None:
        endpoint_origin = None
        endpoint_sha256 = None
    else:
        endpoint_identity = secret_free_endpoint_identity(
            endpoint,
            label=(
                "aggregation producer"
                if declaration_hook == "aggregation_producer_declaration"
                else "Phase-1 producer"
            ),
        )
        endpoint_origin = endpoint_identity["endpoint_origin"]
        endpoint_sha256 = endpoint_identity["endpoint_sha256"]
    canonical.update({
        "endpoint_origin": endpoint_origin,
        "endpoint_sha256": endpoint_sha256,
    })
    if declaration_hook == "aggregation_producer_declaration":
        declaration_schema = AGGREGATION_PRODUCER_DECLARATION_SCHEMA
    else:
        declaration_schema = PHASE1_PRODUCER_DECLARATION_SCHEMA
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_DECLARATION_BYTES:
        raise ValueError("Phase-1 producer declaration is too large")
    return {
        "schema": PHASE1_PRODUCER_BINDING_SCHEMA,
        "identity_exact": True,
        "reuse_scope": "durable",
        "declaration": {
            "schema": declaration_schema,
            **canonical,
        },
        "identity_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }


def _unknown_producer(client: object) -> dict[str, Any]:
    identity = id(client)
    with _unknown_lock:
        registered = _unknown_instances.get(identity)
        if registered is not None and registered[0]() is client:
            nonce = registered[1]
        else:
            nonweak = _unknown_nonweak.get(identity)
            if nonweak is not None and nonweak[0] is client:
                _unknown_nonweak.move_to_end(identity)
                nonce = nonweak[1]
            else:
                nonce = secrets.token_hex(32)
                try:
                    reference = weakref.ref(
                        client,
                        lambda ref, object_id=identity: _drop_unknown_reference(
                            object_id, ref
                        ),
                    )
                except TypeError:
                    _unknown_nonweak[identity] = (client, nonce)
                    _unknown_nonweak.move_to_end(identity)
                    while len(_unknown_nonweak) > _MAX_NONWEAK_UNKNOWN_CLIENTS:
                        evicted_id, _ = _unknown_nonweak.popitem(last=False)
                        _drop_unknown_generation_keys_locked(evicted_id)
                else:
                    _unknown_instances[identity] = (reference, nonce)
    cls = type(client)
    class_name = f"{cls.__module__}.{cls.__qualname__}"
    if not class_name or "\x00" in class_name or len(class_name) > 512:
        class_name = "unknown-client"
    descriptor = {
        "client_class": class_name,
        "process_instance_nonce": nonce,
    }
    encoded = json.dumps(
        descriptor, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": PHASE1_PRODUCER_BINDING_SCHEMA,
        "identity_exact": False,
        "reuse_scope": "process_instance",
        "declaration": None,
        "identity_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }


def _drop_unknown_reference(
    object_id: int, reference: weakref.ReferenceType[object]
) -> None:
    """Remove only the matching dead weak entry; object ids may be reused."""

    with _unknown_lock:
        current = _unknown_instances.get(object_id)
        if current is not None and current[0] is reference:
            _unknown_instances.pop(object_id, None)
            _drop_unknown_generation_keys_locked(object_id)


def _drop_unknown_generation_keys_locked(object_id: int) -> None:
    for key in _unknown_generation_keys.pop(object_id, set()):
        _runtime_authorized_inexact_generations.discard(key)


def _drop_producer_proxy(
    object_id: int, reference: weakref.ReferenceType[object]
) -> None:
    with _unknown_lock:
        current = _producer_proxies.get(object_id)
        if current is not None and current[0] is reference:
            _producer_proxies.pop(object_id, None)


def _phase1_proxy_source(client: object) -> object | None:
    with _unknown_lock:
        current = _producer_proxies.get(id(client))
        if current is None or current[0]() is not client:
            return None
        (
            _reference, source, source_attr, direct_source, proxy_type,
            dispatch_guard, policy_guard,
        ) = current
        if type(client) is not proxy_type:
            return None
        if not _maintained_proxy_type(proxy_type):
            return None
        try:
            state = object.__getattribute__(client, "__dict__")
            if any(name in state for name in _PRODUCER_PROXY_DISPATCH_NAMES):
                return None
            if tuple(
                inspect.getattr_static(proxy_type, name, None)
                for name in _PRODUCER_PROXY_DISPATCH_NAMES
            ) != dispatch_guard:
                return None
            if any(
                _proxy_policy_value(object.__getattribute__(client, name))
                != expected
                for name, expected in policy_guard
            ):
                return None
            if state.get("_closed", False) is not False:
                return None
        except (AttributeError, TypeError, ValueError):
            return None
        if source_attr is not None:
            try:
                if object.__getattribute__(client, source_attr) is not direct_source:
                    # A maintained wrapper whose delegate was rebound is no
                    # longer the transparent execution proxy we registered.
                    # Fall back to conservative identity instead of labeling
                    # calls to producer B with producer A's authority.
                    return None
            except (AttributeError, TypeError):
                return None
        return source


def _registered_phase1_proxy_delegate(proxy: object) -> object | None:
    """Return a registered delegate for inexact diagnostics only.

    Unlike :func:`_verified_phase1_proxy_delegate`, this does not confer exact
    authority.  It lets a drifted maintained wrapper retain bounded model and
    dimension diagnostics while its execution path is disabled and its
    producer binding is downgraded to process-instance scope.
    """

    with _unknown_lock:
        current = _producer_proxies.get(id(proxy))
        if current is None or current[0]() is not proxy:
            return None
        return current[1]


def _verified_phase1_proxy_delegate(proxy: object) -> object:
    """Return the frozen direct delegate or reject a drifted proxy.

    Maintained wrapper methods call this once before dispatch and retain the
    returned object in a local.  A temporary ``_inner``/``delegate`` swap can
    therefore neither redirect the call nor hide via an ABA restore before a
    later producer-identity fence.
    """

    source = _phase1_proxy_source(proxy)
    if source is None:
        raise RuntimeError("maintained producer proxy identity changed")
    return source


def _register_phase1_producer_proxy(proxy: object, source: object) -> None:
    """Bind one maintained transparent wrapper to its invoked client.

    This is intentionally private infrastructure, not an identity-declaration
    hook for arbitrary clients. Custom clients use the closed, validated
    ``phase1_producer_declaration`` contract or conservative process identity.
    """

    if proxy is source:
        raise ValueError("Phase-1 producer proxy cannot wrap itself")
    if not _maintained_proxy_type(type(proxy)):
        raise ValueError("producer proxy type is not maintained")
    try:
        proxy_state = object.__getattribute__(proxy, "__dict__")
    except (AttributeError, TypeError) as exc:
        raise ValueError("maintained producer proxy has no delegate state") from exc
    source_attrs = tuple(
        name for name in ("_inner", "_delegate", "delegate")
        if name in proxy_state
    )
    if len(source_attrs) != 1:
        raise ValueError("maintained producer proxy delegate is ambiguous")
    source_attr = source_attrs[0]
    direct_source = proxy_state[source_attr]
    # Registration describes one execution hop.  Retaining the direct
    # delegate makes identity lookup recursively verify every maintained
    # wrapper in a nested chain.  Accepting a caller-supplied flattened root
    # would let drift in an intermediate wrapper go unnoticed.
    if source is not direct_source:
        raise ValueError("producer proxy source does not match its delegate")
    seen = {id(proxy)}
    chain_source = direct_source
    while True:
        if id(chain_source) in seen:
            raise ValueError("Phase-1 producer proxy chain is cyclic")
        seen.add(id(chain_source))
        nested = _phase1_proxy_source(chain_source)
        if nested is None:
            break
        chain_source = nested
    try:
        reference = weakref.ref(
            proxy,
            lambda ref, object_id=id(proxy): _drop_producer_proxy(
                object_id, ref
            ),
        )
    except TypeError as exc:
        raise TypeError("Phase-1 producer proxy must support weak references") from exc
    with _unknown_lock:
        proxy_type = type(proxy)
        dispatch_guard = tuple(
            inspect.getattr_static(proxy_type, name, None)
            for name in _PRODUCER_PROXY_DISPATCH_NAMES
        )
        policy_guard = tuple(
            (name, _proxy_policy_value(object.__getattribute__(proxy, name)))
            for name in _PRODUCER_PROXY_POLICY_NAMES
            if name in object.__getattribute__(proxy, "__dict__")
        )
        _producer_proxies[id(proxy)] = (
            # Retain the direct delegate, not only the flattened ultimate
            # producer. Identity lookup then revalidates every maintained hop
            # in a nested wrapper chain; drift in an inner proxy cannot be
            # skipped by an outer proxy that still points at it.
            reference, direct_source, source_attr, direct_source, proxy_type,
            dispatch_guard, policy_guard,
        )


def _authorize_inexact_generation(client: object, generation_key: str) -> None:
    """Authorize one opaque generation only while its source object is live."""

    identity = id(client)
    with _unknown_lock:
        weak = _unknown_instances.get(identity)
        strong = _unknown_nonweak.get(identity)
        if not (
            (weak is not None and weak[0]() is client)
            or (strong is not None and strong[0] is client)
        ):
            raise RuntimeError("unknown Phase-1 client identity was not retained")
        client_keys = _unknown_generation_keys.setdefault(identity, set())
        client_keys.add(generation_key)
        _runtime_authorized_inexact_generations.add(generation_key)
        # A pathological custom client that cycles arbitrary public prompt
        # labels cannot grow process authority without bound.  Eviction is
        # conservative: it hides an older generation and forces extraction.
        while len(client_keys) > _MAX_INEXACT_GENERATIONS_PER_CLIENT:
            evicted = next(iter(client_keys - {generation_key}), None)
            if evicted is None:
                break
            client_keys.remove(evicted)
            _runtime_authorized_inexact_generations.discard(evicted)


def phase1_generation_runtime_authorized(
    generation_key: object, identity_exact: object,
) -> int:
    """SQLite-safe current-process authority check for query publication."""

    if identity_exact in (1, True):
        return 1
    if not isinstance(generation_key, str):
        return 0
    with _unknown_lock:
        return int(generation_key in _runtime_authorized_inexact_generations)


def phase1_producer_binding(client: object) -> dict[str, Any]:
    """Return a canonical credential-free producer binding for ``client``."""

    return producer_binding_for_declaration(
        client, declaration_hook="phase1_producer_declaration"
    )


def producer_binding_for_declaration(
    client: object, *, declaration_hook: str,
) -> dict[str, Any]:
    """Resolve one purpose-specific producer declaration or a live nonce.

    The hook vocabulary is closed so an arbitrary caller-selected attribute
    cannot be elevated into durable identity. Aggregation deliberately uses a
    different hook from Phase-1: a router may send those tasks to different
    backends even when both implement the same ``complete`` protocol.
    """

    if declaration_hook not in {
        "phase1_producer_declaration", "aggregation_producer_declaration",
        "memory_producer_declaration",
    }:
        raise ValueError("unsupported producer declaration hook")

    source = _phase1_proxy_source(client)
    if source is not None:
        return producer_binding_for_declaration(
            source, declaration_hook=declaration_hook
        )
    declared = _declared_producer(
        client, declaration_hook=declaration_hook
    )
    return declared if declared is not None else _unknown_producer(client)


def producer_binding_from_typed_declaration(
    declaration: Phase1ProducerDeclaration, *, declaration_hook: str,
) -> dict[str, Any]:
    """Canonicalize a maintained, already-derived producer declaration."""

    return producer_binding_for_declaration(
        _ProducerDeclarationCarrier(declaration),
        declaration_hook=declaration_hook,
    )


def authorize_inexact_producer_generation(
    client: object, generation_key: str,
) -> None:
    """Authorize a non-durable generation for its real live client object.

    This is shared by Phase-1 and downstream LLM materializers. Maintained
    execution-control wrappers are unwrapped through the private proxy
    registry, so deadlines/counters cannot manufacture a second producer.
    """

    source = _phase1_proxy_source(client)
    if source is not None:
        authorize_inexact_producer_generation(source, generation_key)
        return
    _authorize_inexact_generation(client, generation_key)


def producer_generation_runtime_authorized(
    generation_key: object, identity_exact: object,
) -> int:
    """Generic name for the process-lifetime generation authority check."""

    return phase1_generation_runtime_authorized(generation_key, identity_exact)


def phase1_generation_binding(
    prompt_version: str,
    client: object,
) -> dict[str, Any]:
    """Bind the exact extraction contract to one effective producer."""

    source = _phase1_proxy_source(client)
    if source is not None:
        return phase1_generation_binding(prompt_version, source)
    producer = phase1_producer_binding(client)
    payload = {
        "schema": PHASE1_GENERATION_SCHEMA,
        "extraction_cache_key": extraction_cache_key(prompt_version),
        "producer": producer,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    binding = {
        **payload,
        "generation_key": (
            f"{PHASE1_GENERATION_SCHEMA}:"
            + hashlib.sha256(encoded).hexdigest()
        ),
    }
    if not producer["identity_exact"]:
        authorize_inexact_producer_generation(client, binding["generation_key"])
    return binding


def validate_phase1_producer_binding(value: object) -> dict[str, Any]:
    """Validate one historical secret-free producer binding envelope."""

    producer = value
    if not isinstance(producer, Mapping) or set(producer) != {
        "schema", "identity_exact", "reuse_scope", "declaration",
        "identity_sha256",
    }:
        raise ValueError("Phase-1 producer binding shape is invalid")
    if producer.get("schema") != PHASE1_PRODUCER_BINDING_SCHEMA:
        raise ValueError("Phase-1 producer binding schema is invalid")
    if not isinstance(producer.get("identity_exact"), bool):
        raise ValueError("Phase-1 producer exactness is invalid")
    expected_scope = (
        "durable" if producer["identity_exact"] else "process_instance"
    )
    if producer.get("reuse_scope") != expected_scope:
        raise ValueError("Phase-1 producer reuse scope is invalid")
    digest = producer.get("identity_sha256")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ValueError("Phase-1 producer digest is invalid")
    declaration = producer.get("declaration")
    if producer["identity_exact"]:
        expected_declaration_fields = {
            "schema", "client_id", "implementation", "model",
            "endpoint_origin", "endpoint_sha256",
            "effective_request", "retry_policy",
        }
        if (
            not isinstance(declaration, Mapping)
            or set(declaration) != expected_declaration_fields
            or declaration.get("schema")
            != PHASE1_PRODUCER_DECLARATION_SCHEMA
        ):
            raise ValueError("Phase-1 producer declaration is invalid")
        for field in ("client_id", "implementation", "model"):
            _safe_identifier(declaration.get(field), label=field)
        origin = declaration.get("endpoint_origin")
        endpoint_digest = declaration.get("endpoint_sha256")
        if (origin is None) != (endpoint_digest is None):
            raise ValueError("Phase-1 producer endpoint identity is incomplete")
        if origin is not None:
            if not isinstance(origin, str) or validate_http_endpoint(
                origin, label="Phase-1 producer"
            ).url != origin:
                raise ValueError("Phase-1 producer endpoint origin is invalid")
            parsed = urlsplit(origin)
            if parsed.path not in {"", "/"}:
                raise ValueError("Phase-1 producer endpoint origin carries a path")
            if not isinstance(endpoint_digest, str) or _DIGEST_RE.fullmatch(
                endpoint_digest
            ) is None:
                raise ValueError("Phase-1 producer endpoint digest is invalid")
        _validate_nested_identity_digest(
            declaration.get("effective_request"), label="effective request"
        )
        _validate_nested_identity_digest(
            declaration.get("retry_policy"), label="retry policy"
        )
        canonical = _canonical_json(
            {key: item for key, item in declaration.items() if key != "schema"}
        )
        encoded = json.dumps(
            canonical,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > _MAX_DECLARATION_BYTES:
            raise ValueError("Phase-1 producer declaration is too large")
        if "sha256:" + hashlib.sha256(encoded).hexdigest() != digest:
            raise ValueError("Phase-1 producer declaration digest disagrees")
    elif declaration is not None:
        raise ValueError("inexact Phase-1 producer must not claim a declaration")
    return dict(producer)


def validate_aggregation_producer_binding(value: object) -> dict[str, Any]:
    """Validate historical producer metadata under aggregation's policy."""

    producer = value
    if not isinstance(producer, Mapping) or set(producer) != {
        "schema", "identity_exact", "reuse_scope", "declaration",
        "identity_sha256",
    }:
        raise ValueError("aggregation producer binding shape is invalid")
    if producer.get("schema") != PHASE1_PRODUCER_BINDING_SCHEMA:
        raise ValueError("aggregation producer binding schema is invalid")
    if not isinstance(producer.get("identity_exact"), bool):
        raise ValueError("aggregation producer exactness is invalid")
    expected_scope = "durable" if producer["identity_exact"] else "process_instance"
    if producer.get("reuse_scope") != expected_scope:
        raise ValueError("aggregation producer reuse scope is invalid")
    digest = producer.get("identity_sha256")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ValueError("aggregation producer digest is invalid")
    declaration = producer.get("declaration")
    if not producer["identity_exact"]:
        if declaration is not None:
            raise ValueError("inexact aggregation producer claims a declaration")
        return dict(producer)
    expected_fields = {
        "schema", "client_id", "implementation", "model",
        "endpoint_origin", "endpoint_sha256", "effective_request",
        "retry_policy",
    }
    if (
        not isinstance(declaration, Mapping)
        or set(declaration) != expected_fields
        or declaration.get("schema") != AGGREGATION_PRODUCER_DECLARATION_SCHEMA
    ):
        raise ValueError("aggregation producer declaration is invalid")
    for field in ("client_id", "implementation", "model"):
        _safe_identifier(declaration.get(field), label=field)
    origin = declaration.get("endpoint_origin")
    endpoint_digest = declaration.get("endpoint_sha256")
    if (origin is None) != (endpoint_digest is None):
        raise ValueError("aggregation producer endpoint identity is incomplete")
    if origin is not None:
        if not isinstance(origin, str) or validate_http_endpoint(
            origin, label="aggregation producer"
        ).url != origin:
            raise ValueError("aggregation producer endpoint origin is invalid")
        parsed = urlsplit(origin)
        if parsed.path not in {"", "/"}:
            raise ValueError("aggregation producer endpoint origin carries a path")
        if not isinstance(endpoint_digest, str) or _DIGEST_RE.fullmatch(
            endpoint_digest
        ) is None:
            raise ValueError("aggregation producer endpoint digest is invalid")
    _validate_nested_identity_digest(
        declaration.get("effective_request"), label="effective request"
    )
    _validate_nested_identity_digest(
        declaration.get("retry_policy"), label="retry policy"
    )
    canonical = _canonical_json(
        {key: item for key, item in declaration.items() if key != "schema"}
    )
    encoded = json.dumps(
        canonical, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_DECLARATION_BYTES:
        raise ValueError("aggregation producer declaration is too large")
    if "sha256:" + hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("aggregation producer declaration digest disagrees")
    return dict(producer)


def validate_phase1_generation_binding(value: object) -> dict[str, Any]:
    """Require a self-consistent recorded generation binding.

    This is deliberately historical/structural validation. It proves the
    registry envelope and hashes agree without requiring the recorded contract
    digest to equal the code running today.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "schema", "extraction_cache_key", "producer", "generation_key",
    }:
        raise ValueError("Phase-1 generation binding shape is invalid")
    if value.get("schema") != PHASE1_GENERATION_SCHEMA:
        raise ValueError("Phase-1 generation binding schema is invalid")
    cache_key = _validate_historical_extraction_cache_key(
        value.get("extraction_cache_key")
    )
    producer = validate_phase1_producer_binding(value.get("producer"))

    payload = {
        "schema": value["schema"],
        "extraction_cache_key": cache_key,
        "producer": dict(producer),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_key = (
        f"{PHASE1_GENERATION_SCHEMA}:" + hashlib.sha256(encoded).hexdigest()
    )
    if value.get("generation_key") != expected_key:
        raise ValueError("Phase-1 generation key disagrees with its binding")
    return {
        "schema": value["schema"],
        "extraction_cache_key": cache_key,
        "producer": dict(producer),
        "generation_key": expected_key,
    }


def validate_current_phase1_generation_binding(
    value: object,
    *,
    prompt_version: str,
) -> dict[str, Any]:
    """Require a valid generation bound to today's executable contract."""

    binding = validate_phase1_generation_binding(value)
    expected = extraction_cache_key(prompt_version)
    if binding["extraction_cache_key"] != expected:
        raise ValueError("Phase-1 generation uses another extraction contract")
    return binding


def phase1_generation_key_is_shaped(value: object) -> bool:
    return isinstance(value, str) and _GENERATION_KEY_RE.fullmatch(value) is not None


def canonical_phase1_generation_json(value: object) -> str:
    """Validate and serialize a generation binding for durable storage."""

    binding = validate_phase1_generation_binding(value)
    return json.dumps(
        binding,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def register_phase1_generation(
    conn: sqlite3.Connection, value: object
) -> dict[str, Any]:
    """Idempotently register one validated binding in the caller's transaction."""

    binding = validate_phase1_generation_binding(value)
    producer = binding["producer"]
    # An opaque binding is meaningful only while the exact client instance
    # that minted its nonce is alive in this process.  Structural validation
    # alone is insufficient: a serialized binding copied from another process
    # would otherwise be able to recreate processed/cache rows even though it
    # is intentionally not durable identity.
    if not producer["identity_exact"] and not phase1_generation_runtime_authorized(
        binding["generation_key"], False
    ):
        raise ValueError(
            "inexact Phase-1 generation is not authorized by a live client"
        )
    encoded = canonical_phase1_generation_json(binding)
    key = binding["generation_key"]
    existing = conn.execute(
        "SELECT extraction_cache_key,producer_identity_sha256,identity_exact,"
        "reuse_scope,binding_json FROM phase1_generations WHERE generation_key=?",
        (key,),
    ).fetchone()
    expected = (
        binding["extraction_cache_key"],
        producer["identity_sha256"],
        1 if producer["identity_exact"] else 0,
        producer["reuse_scope"],
        encoded,
    )
    if existing is not None:
        if tuple(existing) != expected:
            raise ValueError("Phase-1 generation registry collision")
        return binding
    conn.execute(
        "INSERT INTO phase1_generations("
        "generation_key,extraction_cache_key,producer_identity_sha256,"
        "identity_exact,reuse_scope,binding_json) VALUES (?,?,?,?,?,?)",
        (key, *expected),
    )
    return binding


def phase1_generation_registry_row_is_valid(
    generation_key: object,
    extraction_cache_key_value: object,
    producer_identity_sha256: object,
    identity_exact: object,
    reuse_scope: object,
    binding_json: object,
) -> int:
    """SQLite guard for registry inserts; malformed rows fail closed."""

    try:
        if not isinstance(binding_json, str) or len(
            binding_json.encode("utf-8")
        ) > _MAX_DECLARATION_BYTES + 4096:
            return 0

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = item
            return result

        decoded = json.loads(
            binding_json,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
        binding = validate_phase1_generation_binding(decoded)
        producer = binding["producer"]
        expected = (
            binding["generation_key"],
            binding["extraction_cache_key"],
            producer["identity_sha256"],
            1 if producer["identity_exact"] else 0,
            producer["reuse_scope"],
            canonical_phase1_generation_json(binding),
        )
        actual = (
            generation_key, extraction_cache_key_value,
            producer_identity_sha256, identity_exact, reuse_scope,
            binding_json,
        )
        return int(actual == expected)
    except (TypeError, ValueError, UnicodeError, OverflowError):
        return 0


# Complete the producer <-> contract import cycle with one frozen live-code
# helper.  Contract identities continue to inspect the active loaded module
# graph, but never follow a later monkeypatch of this public helper name and
# never reread deployment files.
from hymem.extraction import contract as _contract_module

_contract_module._LOADED_MODULE_SLICE_SHA256 = canonical_module_slice_sha256
