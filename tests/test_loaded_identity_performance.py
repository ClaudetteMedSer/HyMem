"""Immutable-piece reuse must never turn live executable checks into cache hits."""

from __future__ import annotations

import gc
import types
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from hymem.extraction import contract, producer, triples
from hymem.dreaming import chunks


def _module(source: str) -> types.ModuleType:
    module = types.ModuleType("identity_piece_test")
    exec(compile(source, "identity_piece_test.py", "exec"), module.__dict__)
    return module


def test_warm_identity_reuses_code_and_disassembly_but_not_live_state(monkeypatch):
    module = _module("""
OPTIONS = {"allowed": {"first"}}
def select(value=[1], *, option=[2]):
    return OPTIONS, value, option
""")
    producer._immutable_loaded_code_sha256.cache_clear()
    producer._immutable_code_global_names.cache_clear()
    counts = {"code": 0, "disassembly": 0}
    original_code = producer._uncached_loaded_code_sha256
    original_names = producer._uncached_code_global_names

    def code_record(code):
        counts["code"] += 1
        return original_code(code)

    def names(code):
        counts["disassembly"] += 1
        return original_names(code)

    monkeypatch.setattr(producer, "_uncached_loaded_code_sha256", code_record)
    monkeypatch.setattr(producer, "_uncached_code_global_names", names)
    baseline = producer.canonical_module_slice_sha256(module, "select")
    cold_counts = dict(counts)
    assert all(count > 0 for count in cold_counts.values())
    for _ in range(5):
        assert producer.canonical_module_slice_sha256(module, "select") == baseline
    assert counts == cold_counts

    # All of these mutate existing objects, rather than merely rebinding a name.
    module.select.__defaults__[0].append(3)
    defaults_changed = producer.canonical_module_slice_sha256(module, "select")
    assert defaults_changed != baseline
    module.select.__kwdefaults__["option"].append(4)
    kwdefaults_changed = producer.canonical_module_slice_sha256(module, "select")
    assert kwdefaults_changed != defaults_changed
    module.OPTIONS["allowed"].add("second")
    assert producer.canonical_module_slice_sha256(module, "select") != kwdefaults_changed
    assert counts == cold_counts


def test_cached_code_does_not_cache_closure_or_class_state():
    def factory():
        values = {"items": [1]}

        def read():
            return values

        return read, values

    read, values = factory()
    original = producer.canonical_callable_sha256(read)
    values["items"].append(2)
    assert producer.canonical_callable_sha256(read) != original
    values["items"].pop()
    assert producer.canonical_callable_sha256(read) == original

    module = _module("""
class Policy:
    OPTIONS = {"allowed": [1]}
    def read(self):
        return self.OPTIONS
""")
    original = producer.canonical_module_slice_sha256(module, "Policy")
    module.Policy.OPTIONS["allowed"].append(2)
    assert producer.canonical_module_slice_sha256(module, "Policy") != original


def test_code_replacement_refreshes_dependencies_and_rebinding():
    module = _module("""
FIRST = {"value": 1}
SECOND = {"value": 2}
def read():
    return FIRST
def replacement():
    return SECOND
""")
    original = producer.canonical_module_slice_sha256(module, "read")
    module.read.__code__ = module.replacement.__code__
    changed = producer.canonical_module_slice_sha256(module, "read")
    assert changed != original
    module.SECOND["value"] = 3
    assert producer.canonical_module_slice_sha256(module, "read") != changed
    changed = producer.canonical_module_slice_sha256(module, "read")
    module.read = lambda: 4
    assert producer.canonical_module_slice_sha256(module, "read") != changed


@pytest.mark.parametrize("hashable_list", [False, True])
def test_mutable_code_constants_never_enter_piece_cache(hashable_list):
    class HashableList(list):
        def __hash__(self):
            raise AssertionError("identity caching invoked a constant hash hook")

        def __eq__(self, other):
            raise AssertionError("identity caching invoked a constant equality hook")

    value = HashableList([1]) if hashable_list else [1]

    def read():
        return None

    read.__code__ = read.__code__.replace(co_consts=(value,))
    baseline = producer.canonical_callable_sha256(read)
    value.append(2)
    assert producer.canonical_callable_sha256(read) != baseline


def test_deep_synthetic_constants_bypass_cache_without_recursive_hashing():
    constant = 1
    for _ in range(1500):
        constant = (constant,)

    def read():
        return None

    read.__code__ = read.__code__.replace(co_consts=(constant,))
    assert not producer._immutable_code_constant(read.__code__)
    assert producer.canonical_callable_sha256(read) == (
        producer.canonical_callable_sha256(read)
    )


def test_exact_callable_rechecks_referenced_globals_after_code_cache_hit():
    module = _module("OPTIONS = (1, 2)\ndef read():\n    return OPTIONS\n")
    original = producer.exact_callable_sha256(module.read)
    module.OPTIONS = (1, 3)
    assert producer.exact_callable_sha256(module.read) != original
    module.OPTIONS = [1, 3]
    with pytest.raises(ValueError, match="mutable execution state"):
        producer.exact_callable_sha256(module.read)


def test_cached_pieces_and_returned_records_cannot_be_poisoned():
    def read(value=[1]):
        return value

    original = producer.canonical_callable_sha256(read)
    assert isinstance(producer._loaded_code_record(read.__code__), str)
    assert isinstance(producer._code_global_names(read.__code__), tuple)
    record = producer._loaded_identity_value(read)
    record[3] = "forged code digest"
    record[4].append("forged default")
    assert producer.canonical_callable_sha256(read) == original


def test_piece_cache_retention_is_bounded_and_does_not_retain_modules():
    producer._immutable_loaded_code_sha256.cache_clear()
    producer._immutable_code_global_names.cache_clear()
    module = _module("def read():\n    return 1\n")
    module_reference = weakref.ref(module)
    function_reference = weakref.ref(module.read)
    base_code = module.read.__code__
    for index in range(1200):
        code = base_code.replace(co_consts=(None, index))
        producer._loaded_code_record(code)
        producer._code_global_names(code)
    assert producer._immutable_loaded_code_sha256.cache_info().currsize <= 1024
    assert producer._immutable_code_global_names.cache_info().currsize <= 1024
    producer.canonical_module_slice_sha256(module, "read")
    del module
    gc.collect()
    assert module_reference() is None
    assert function_reference() is None


def test_concurrent_identity_reads_are_deterministic():
    module = _module("OPTIONS = {1, 2}\ndef read():\n    return OPTIONS\n")
    expected = producer.canonical_module_slice_sha256(module, "read")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _: producer.canonical_module_slice_sha256(module, "read"),
            range(80),
        ))
    assert results == [expected] * 80


def test_warm_contract_rejects_old_key_after_in_place_vocabulary_mutation(
    monkeypatch,
):
    # A mutable vocabulary remains legal input to the canonicalizer. Caching
    # only code pieces must not conceal changes to this shared runtime value.
    valid_types = set(triples._VALID_TYPES)
    monkeypatch.setattr(triples, "_VALID_TYPES", valid_types)
    old_key = contract.extraction_cache_key()
    assert contract.extraction_cache_key(old_key) == old_key
    valid_types.add("new_runtime_type")
    with pytest.raises(ValueError, match="another contract"):
        contract.extraction_cache_key(old_key)


def test_warm_contract_avoids_repeated_code_and_disassembly_work(monkeypatch):
    contract.extraction_contract_identity()

    def rebuilt(_code):
        raise AssertionError("unchanged immutable code piece was rebuilt")

    monkeypatch.setattr(producer, "_uncached_loaded_code_sha256", rebuilt)
    monkeypatch.setattr(producer, "_uncached_code_global_names", rebuilt)
    expected = contract.extraction_contract_identity()
    for _ in range(3):
        assert contract.extraction_contract_identity() == expected


@pytest.mark.parametrize(("max_attempts", "generation"), [(0, "key"), (3, None)])
def test_noop_quarantine_does_not_hash_an_unused_contract(
    monkeypatch, max_attempts, generation,
):
    def unexpected(_prompt):
        raise AssertionError("no-op quarantine computed an unused identity")

    monkeypatch.setattr(chunks, "extraction_cache_key", unexpected)
    assert not chunks.chunk_extraction_is_quarantined(
        None, "unused", prompt_version="v20", max_attempts=max_attempts,
        phase1_generation_key=generation,
    )
