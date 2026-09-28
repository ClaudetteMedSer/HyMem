"""Per-read sharing must preserve the ordinary depth-limited commitment."""

from __future__ import annotations

import gc
import hashlib
import json
import re
import types
import weakref
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from hymem.extraction import producer


def _ordinary_json(value, *, depth=0):
    return json.dumps(
        producer._loaded_identity_value(value, depth=depth),
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    )


def _encoded_json(value, *, depth=0):
    return producer._LoadedIdentityEncoder().value(value, depth=depth).text


def _graphs():
    shared = {"unicode": ["é", "\ud800", "\n"], "values": {1, 2}}
    cycle = [shared]
    cycle.extend([cycle, cycle])

    @dataclass(frozen=True)
    class Policy:
        value: int = 1

    def read(value=shared, *, option=cycle):
        return Policy, value, option

    empty = (lambda item: lambda: item)(1)
    del empty.__closure__[0].cell_contents
    return [
        [shared, shared, cycle], Policy, read, empty,
        {("é", 1): re.compile("abc", re.I), ("a", 2): {b"one", b"two"}},
        [None, True, -0.0, float("nan"), float("inf"), b"\x00\xff"],
        [staticmethod(read), classmethod(read), property(read)],
    ]


@pytest.mark.parametrize("depth", [0, 4, 12, 13])
@pytest.mark.parametrize("graph", range(7))
def test_fragment_matches_ordinary_depth_limited_json(graph, depth):
    value = _graphs()[graph]
    assert _encoded_json(value, depth=depth) == _ordinary_json(value, depth=depth)


def test_shared_graph_construction_and_encoding_are_not_exponential(monkeypatch):
    value = ["leaf"]
    for _ in range(6):
        value = [value] * 4
    expected = _ordinary_json(value)
    original = producer._loaded_identity_record
    visits = 0

    def counted(value, *, depth=0, visit=None):
        nonlocal visits
        visits += 1
        return original(value, depth=depth, visit=visit)

    monkeypatch.setattr(producer, "_loaded_identity_record", counted)
    assert _ordinary_json(value) == expected
    ordinary_visits = visits
    visits = 0
    assert _encoded_json(value) == expected
    # Eight unique object/depth pairs, versus thousands of unfolded records.
    assert ordinary_visits > 9000
    assert visits == 8


def test_dataclass_closure_cycle_is_shared_without_changing_json(monkeypatch):
    @dataclass(frozen=True)
    class Policy:
        value: int = 1

    original = producer._loaded_identity_record
    counts = {"ordinary": 0, "shared": 0}
    mode = "ordinary"

    def counted(value, *, depth=0, visit=None):
        counts[mode] += 1
        return original(value, depth=depth, visit=visit)

    monkeypatch.setattr(producer, "_loaded_identity_record", counted)
    expected = _ordinary_json(Policy)
    mode = "shared"
    assert _encoded_json(Policy) == expected
    assert counts["ordinary"] > 10 * counts["shared"]


def test_public_records_keep_independent_alias_copies():
    shared = [1]
    record = producer._loaded_identity_value([shared, shared])
    record[1][0][1].append(["int", 2])
    assert record[1][1] == ["list", [["int", 1]]]
    assert _ordinary_json(shared) == '["list",[["int",1]]]'


class _ChangingMapping(Mapping):
    def __init__(self, shared):
        self.shared = shared
        self.calls = 0

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(("value",))

    def __getitem__(self, key):
        return self.calls

    def items(self):
        self.calls += 1
        self.shared.append(self.calls)
        return [("value", self.calls)]


def test_dynamic_descendants_invalidate_prior_and_enclosing_fragments():
    shared = [0]
    dynamic = _ChangingMapping(shared)
    enclosing = [dynamic]
    graph = [shared, enclosing, shared, enclosing, shared]
    expected = _ordinary_json(graph)
    assert dynamic.calls == 2
    shared[:] = [0]
    dynamic.calls = 0
    assert _encoded_json(graph) == expected
    assert dynamic.calls == 2


@pytest.mark.parametrize("kind", [float, bytes])
def test_dynamic_hex_hooks_are_not_memoized(kind):
    class Dynamic(kind):
        calls = 0

        def hex(self):
            self.calls += 1
            return str(self.calls)

    value = Dynamic(1 if kind is float else b"a")
    graph = [value, value]
    expected = _ordinary_json(graph)
    value.calls = 0
    assert _encoded_json(graph) == expected
    assert value.calls == 2


def test_synthetic_mutable_code_constants_do_not_cache_their_function():
    shared = [0]
    dynamic = _ChangingMapping(shared)

    def read():
        return None

    read.__code__ = read.__code__.replace(co_consts=(dynamic,))
    graph = [shared, read, shared, read, shared]
    expected = _ordinary_json(graph)
    assert dynamic.calls == 2
    shared[:] = [0]
    dynamic.calls = 0
    assert _encoded_json(graph) == expected
    assert dynamic.calls == 2


def test_memo_eligibility_never_invokes_metaclass_equality():
    class Meta(type):
        def __eq__(cls, other):
            raise AssertionError("eligibility invoked user equality")

        __hash__ = type.__hash__

    class Opaque(metaclass=Meta):
        pass

    value = Opaque()
    assert not producer._immutable_code_constant(value)
    # Stop before the legacy Mapping ABC check, which may itself legitimately
    # consult metaclass equality. The new memo check must not consult it first.
    assert _encoded_json(value, depth=13) == _ordinary_json(value, depth=13)


def test_dynamic_hook_reentrant_identity_has_its_own_context():
    shared = [0]

    def read():
        return shared

    class Reentrant(_ChangingMapping):
        nested = []

        def items(self):
            result = super().items()
            self.nested.append(producer.canonical_callable_sha256(read))
            return result

    dynamic = Reentrant(shared)
    graph = [shared, dynamic, shared, dynamic, shared]
    expected = _ordinary_json(graph)
    nested = list(dynamic.nested)
    shared[:] = [0]
    dynamic.calls = 0
    dynamic.nested.clear()
    assert _encoded_json(graph) == expected
    assert dynamic.nested == nested
    assert len(set(nested)) == 2


def test_nested_mutable_code_constants_are_read_after_closure_hooks():
    shared = [0]
    dynamic = _ChangingMapping(shared)

    def read():
        return dynamic

    def constant_template():
        return None

    nested_code = constant_template.__code__.replace(co_consts=(shared,))
    read.__code__ = read.__code__.replace(co_consts=(nested_code,))
    expected = _ordinary_json([read, read])
    assert dynamic.calls == 2
    dynamic.calls = 0
    shared[:] = [0]
    assert _encoded_json([read, read]) == expected
    assert dynamic.calls == 2


def test_memo_limits_preserve_full_output_and_release_input_references(monkeypatch):
    monkeypatch.setattr(producer._LoadedIdentityEncoder, "_MAX_ENTRIES", 8)
    monkeypatch.setattr(producer._LoadedIdentityEncoder, "_MAX_BYTES", 256)
    encoder = producer._LoadedIdentityEncoder()
    values = ["a" * 1024, *range(100)]
    assert encoder.value(values).text == _ordinary_json(values)
    assert len(encoder._memo) <= 8
    assert encoder._memo_bytes <= 256
    assert all(len(fragment.text) <= 256 for _, fragment in encoder._memo.values())

    def read():
        return None

    reference = weakref.ref(read)
    encoder = producer._LoadedIdentityEncoder()
    # A bounded reference is held while an eligible fragment is reusable.
    monkeypatch.setattr(producer._LoadedIdentityEncoder, "_MAX_BYTES", 4096)
    encoder.value(read)
    del read
    gc.collect()
    assert reference() is not None
    del encoder
    gc.collect()
    assert reference() is None


def _digest_payload(payload):
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_module_discovery_hooks_cannot_hide_later_alias_mutation():
    module = types.ModuleType("identity_discovery")

    class Policy:
        OPTIONS = [1]

    Policy.__module__ = module.__name__

    class Mutator:
        def __getattribute__(self, name):
            if name == "__module__":
                Policy.OPTIONS.append(2)
                return "not_the_owner"
            return object.__getattribute__(self, name)

    module.aPolicy = Policy
    module.mutator = Mutator()
    module.zPolicy = Policy
    first = producer._loaded_identity_value(Policy)
    Policy.OPTIONS.append(2)
    second = producer._loaded_identity_value(Policy)
    expected = _digest_payload({
        "runtime": producer._LOADED_CODE_RUNTIME_DOMAIN,
        "modules": [["module", module.__name__, [
            ["aPolicy", first], ["zPolicy", second],
        ]]],
    })
    Policy.OPTIONS[:] = [1]
    assert producer.canonical_module_sha256(module) == expected
    assert Policy.OPTIONS == [1, 2]


def test_module_slice_discovery_hooks_cannot_hide_later_alias_mutation():
    module = types.ModuleType("identity_slice_discovery")
    exec("def root():\n    return MUTATOR\n", module.__dict__)

    class Policy:
        OPTIONS = [1]

    class Mutator:
        def __getattribute__(self, name):
            if name == "__module__":
                Policy.OPTIONS.append(2)
                return "not_the_owner"
            return object.__getattribute__(self, name)

    module.MUTATOR = Mutator()
    module.first = Policy
    module.second = Policy
    first = producer._loaded_identity_value(Policy)
    root = producer._loaded_identity_value(module.root)
    Policy.OPTIONS.append(2)
    mutation = producer._loaded_identity_value(module.MUTATOR)
    second = producer._loaded_identity_value(Policy)
    expected = _digest_payload({
        "runtime": producer._LOADED_CODE_RUNTIME_DOMAIN,
        "module": module.__name__, "roots": ["first", "root", "second"],
        "bindings": [
            ["MUTATOR", mutation], ["first", first], ["root", root],
            ["second", second],
        ],
    })
    Policy.OPTIONS[:] = [1]
    # LIFO discovery: first -> root -> MUTATOR -> second.
    assert producer.canonical_module_slice_sha256(
        module, "second", "root", "first",
    ) == expected
    assert Policy.OPTIONS == [1, 2]


@pytest.mark.parametrize("mapping", [{1: "one", 2: "two"}, {None: "none"}, {True: "bool"}])
def test_mutated_runtime_domain_keeps_json_key_conversion(monkeypatch, mapping):
    def read():
        return None

    monkeypatch.setitem(producer._LOADED_CODE_RUNTIME_DOMAIN, "custom", mapping)
    expected = _digest_payload({
        "runtime": producer._LOADED_CODE_RUNTIME_DOMAIN,
        "callables": [producer._loaded_identity_value(read)],
    })
    assert producer.canonical_callable_sha256(read) == expected
