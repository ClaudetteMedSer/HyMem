from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("requests")
pytest.importorskip("ijson")
_BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS))

import beam_adapter as beam  # noqa: E402
import locomo_adapter as locomo  # noqa: E402
import longmemeval_adapter as lme  # noqa: E402
import msc_adapter as msc  # noqa: E402
from benchmarks.strictness import (  # noqa: E402
    AtomicCheckpoint,
    BenchmarkIntegrityError,
    build_manifest,
    content_hash,
)


_SOURCES = {
    "strictness": """
CODE_IDENTITY_VERSION = "test-code-v2"
try:
    import optional_lock_backend as LOCK_BACKEND
except ImportError:
    LOCK_BACKEND = None
if LOCK_BACKEND is None:
    LOCK_MODE = "fallback"
else:
    LOCK_MODE = "native"
def strict_helper(value):
    return value + 1 + (0 if LOCK_MODE else 0)
def strict_guard(value):
    return strict_helper(value)
def code_hash(paths, root):
    return strict_helper(len(tuple(paths)))
def strict_unused(value):
    return value - 99
""",
    "protocol": """
LME_PROTOCOL = "lme-v1"
BEAM_PROTOCOL = "beam-v1"
MSC_PROTOCOL = "msc-v1"
TRANSITIVE_PROTOCOL = "transitive-v1"
LOCOMO_PROTOCOL = "locomo-v1"
UNUSED_PROTOCOL = "unused-v1"
""",
    "run_registry": """
def is_retrieval_only(value):
    return bool(value)
def unused_registry_helper(value):
    return not value
""",
    "canary": """
from benchmarks.strictness import strict_guard
from hymem.canary_core import canary_value
CANARY_LIMIT = 2
if CANARY_LIMIT < 2:
    raise RuntimeError("invalid canary limit")
def canary_helper():
    return strict_guard(canary_value())
def run_canary():
    return canary_helper() + CANARY_LIMIT
def unused_canary():
    return "unused"
""",
    "store": """
from benchmarks.strictness import strict_guard
def attest(value):
    return strict_guard(value)
def unused_attestation(value):
    return value
""",
    "msc_registry": """
from benchmarks.strictness import strict_guard
def validate_msc_artifact(value):
    return strict_guard(value)
def unused_msc_registry(value):
    return value
""",
    "lme": """
from benchmarks.strictness import code_hash, strict_guard
from benchmarks.extraction_canary import run_canary
from benchmarks.lme_protocol import (
    LME_PROTOCOL, BEAM_PROTOCOL, MSC_PROTOCOL, TRANSITIVE_PROTOCOL,
    LOCOMO_PROTOCOL,
)
from hymem import shared_value
from hymem.lme_core import lme_value
MAX_CONTEXT_CHARS = 8000

def identity_decorator(**options):
    return lambda function: function

def lme_only():
    return LME_PROTOCOL + str(lme_value())

@identity_decorator(frozen=True)
def beam_shared():
    return BEAM_PROTOCOL + str(strict_guard(shared_value()))

def msc_lme_shared():
    return MSC_PROTOCOL

def transitive_lme_shared():
    return TRANSITIVE_PROTOCOL

def locomo_lme_shared():
    return LOCOMO_PROTOCOL

def registry_path(value):
    from run_registry import is_retrieval_only
    return is_retrieval_only(value)

if __name__ != "__main__" and BEAM_PROTOCOL == "":
    raise RuntimeError("invalid imported BEAM protocol")

if __name__ == "__main__":
    print(beam_shared())
""",
    "beam": """
from benchmarks.strictness import code_hash, strict_guard
from benchmarks.extraction_canary import run_canary
from longmemeval_adapter import beam_shared
from hymem import shared_value
from hymem.beam_core import beam_value
def beam_only():
    return beam_shared() + str(beam_value() + shared_value())
""",
    "msc": """
from benchmarks.strictness import code_hash, strict_guard
from benchmarks.extraction_canary import run_canary
from benchmarks.store_attestation import attest
from longmemeval_adapter import msc_lme_shared
from hymem import shared_value
from hymem.msc_core import msc_value
INDEXING_PROVENANCE_VERSION = "hymem-benchmark-indexing-v4"
STORE_BUILD_RECEIPT_VERSION = "hymem-benchmark-store-build-v7"
STORE_INDEXING_ATTESTATION_VERSION = "hymem-benchmark-store-indexing-attestation-v2"

def msc_shared():
    from longmemeval_adapter import transitive_lme_shared
    return transitive_lme_shared() + str(attest(shared_value())) + (
        INDEXING_PROVENANCE_VERSION + STORE_BUILD_RECEIPT_VERSION
        + STORE_INDEXING_ATTESTATION_VERSION
    )

def msc_only():
    return msc_lme_shared() + str(msc_value())

def validate_output(value):
    from benchmarks.msc_registry import validate_msc_artifact
    return validate_msc_artifact(value)
""",
    "locomo": """
from benchmarks.strictness import code_hash, strict_guard
from benchmarks.extraction_canary import run_canary
from msc_adapter import msc_shared
from longmemeval_adapter import locomo_lme_shared
import longmemeval_adapter as _lme
from hymem import shared_value
from hymem.locomo_core import locomo_value
def locomo_only():
    return (msc_shared() + locomo_lme_shared()
            + str(locomo_value() + shared_value() + _lme.MAX_CONTEXT_CHARS))
""",
    "hymem_init": "from hymem.shared_core import shared_value\n",
    "hymem_shared": """
from hymem.core import db
def shared_value():
    return db.VALUE
""",
    "hymem_db": "VALUE = 1\n",
    "hymem_canary": "def canary_value():\n    return 2\n",
    "hymem_lme": "def lme_value():\n    return 3\n",
    "hymem_beam": "def beam_value():\n    return 4\n",
    "hymem_msc": "def msc_value():\n    return 5\n",
    "hymem_locomo": "def locomo_value():\n    return 6\n",
}


def _write_source_tree(root: Path) -> dict[str, Path]:
    benchmark_dir = root / "benchmarks"
    hymem_dir = root / "hymem"
    core_dir = hymem_dir / "core"
    migrations = core_dir / "migrations"
    benchmark_dir.mkdir(parents=True)
    migrations.mkdir(parents=True)
    paths = {
        "root": root,
        "adapter": benchmark_dir / "longmemeval_adapter.py",
        "beam": benchmark_dir / "beam_adapter.py",
        "msc": benchmark_dir / "msc_adapter.py",
        "locomo": benchmark_dir / "locomo_adapter.py",
        "strictness": benchmark_dir / "strictness.py",
        "protocol": benchmark_dir / "lme_protocol.py",
        "run_registry": benchmark_dir / "run_registry.py",
        "canary": benchmark_dir / "extraction_canary.py",
        "store": benchmark_dir / "store_attestation.py",
        "msc_registry": benchmark_dir / "msc_registry.py",
        "hymem": hymem_dir,
        "hymem_init": hymem_dir / "__init__.py",
        "hymem_shared": hymem_dir / "shared_core.py",
        "hymem_db": core_dir / "db.py",
        "hymem_core_init": core_dir / "__init__.py",
        "hymem_schema": core_dir / "schema.sql",
        "hymem_migration": migrations / "001_test.sql",
        "hymem_migrations_init": migrations / "__init__.py",
        "hymem_canary": hymem_dir / "canary_core.py",
        "hymem_lme": hymem_dir / "lme_core.py",
        "hymem_beam": hymem_dir / "beam_core.py",
        "hymem_msc": hymem_dir / "msc_core.py",
        "hymem_locomo": hymem_dir / "locomo_core.py",
        "hymem_unrelated": hymem_dir / "unrelated.py",
        "hymem_markdown": hymem_dir / "notes.md",
        "hymem_text": hymem_dir / "notes.txt",
        "benchmark_unrelated": benchmark_dir / "unrelated_probe.py",
    }
    source_keys = {
        "adapter": "lme",
        "beam": "beam",
        "msc": "msc",
        "locomo": "locomo",
        "strictness": "strictness",
        "protocol": "protocol",
        "run_registry": "run_registry",
        "canary": "canary",
        "store": "store",
        "msc_registry": "msc_registry",
        "hymem_init": "hymem_init",
        "hymem_shared": "hymem_shared",
        "hymem_db": "hymem_db",
        "hymem_canary": "hymem_canary",
        "hymem_lme": "hymem_lme",
        "hymem_beam": "hymem_beam",
        "hymem_msc": "hymem_msc",
        "hymem_locomo": "hymem_locomo",
    }
    for path_key, source_key in source_keys.items():
        paths[path_key].write_text(_SOURCES[source_key].lstrip(), encoding="utf-8")
    paths["hymem_core_init"].write_text(
        "from hymem.core import db\nCORE_INIT = 1\n", encoding="utf-8"
    )
    paths["hymem_schema"].write_text(
        "CREATE TABLE shared_v1 (id INTEGER);\n", encoding="utf-8"
    )
    paths["hymem_migration"].write_text(
        "ALTER TABLE shared_v1 ADD COLUMN note TEXT;\n", encoding="utf-8"
    )
    paths["hymem_migrations_init"].write_text(
        "MIGRATION_INIT = 1\n", encoding="utf-8"
    )
    paths["hymem_unrelated"].write_text("UNRELATED = 1\n", encoding="utf-8")
    paths["hymem_markdown"].write_text("documentation v1\n", encoding="utf-8")
    paths["hymem_text"].write_text("prose v1\n", encoding="utf-8")
    paths["benchmark_unrelated"].write_text("UNRELATED = 1\n", encoding="utf-8")
    return paths


def _common_kwargs(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "strictness_path": paths["strictness"],
        "archive_evidence_path": paths.get("archive_evidence"),
        "lme_adapter_path": paths["adapter"],
        "lme_protocol_path": paths["protocol"],
        "extraction_canary_path": paths["canary"],
        "hymem_path": paths["hymem"],
        "root": paths["root"],
    }


def _lme_hash(paths: dict[str, Path]) -> str:
    return lme.longmemeval_code_hash(
        adapter_path=paths["adapter"],
        strictness_path=paths["strictness"],
        archive_evidence_path=paths.get("archive_evidence"),
        protocol_path=paths["protocol"],
        run_registry_path=paths["run_registry"],
        extraction_canary_path=paths["canary"],
        hymem_path=paths["hymem"],
        root=paths["root"],
    )


def _beam_hash(paths: dict[str, Path]) -> str:
    return beam.beam_code_hash(adapter_path=paths["beam"], **_common_kwargs(paths))


def _msc_hash(paths: dict[str, Path]) -> str:
    return msc.msc_code_hash(
        adapter_path=paths["msc"],
        store_attestation_path=paths["store"],
        registry_path=paths["msc_registry"],
        **_common_kwargs(paths),
    )


def _locomo_hash(paths: dict[str, Path]) -> str:
    return locomo.locomo_code_hash(
        adapter_path=paths["locomo"],
        msc_adapter_path=paths["msc"],
        store_attestation_path=paths["store"],
        **_common_kwargs(paths),
    )


_HASHERS = {
    "lme": _lme_hash,
    "beam": _beam_hash,
    "msc": _msc_hash,
    "locomo": _locomo_hash,
}


def _hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: hasher(paths) for name, hasher in _HASHERS.items()}


def _replace(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    assert old in source
    path.write_text(source.replace(old, new), encoding="utf-8")


def _assert_changed(
    before: dict[str, str], after: dict[str, str], expected: set[str],
) -> None:
    assert {name for name in before if before[name] != after[name]} == expected


def _manifest(benchmark: str, code_hash: str) -> dict:
    return build_manifest(
        benchmark=benchmark,
        code_sha256=code_hash,
        data_sha256=content_hash("fixed data"),
        config={
            "label_free_answer_path": True,
            "scored_run": True,
            "exploratory_label_steering": False,
            "exploratory_non_comparable": False,
        },
        models={"reader": "fixed-model"},
        seed=7,
        expected_ids=("question-1",),
        protocol_split="full",
    )


def _assert_identity_change_rejects_old_checkpoint(
    tmp_path: Path,
    *,
    benchmark: str,
    before_hash: str,
    after_hash: str,
) -> None:
    assert before_hash != after_hash
    before_manifest = _manifest(benchmark, before_hash)
    after_manifest = _manifest(benchmark, after_hash)
    assert before_manifest["run_id"] != after_manifest["run_id"]
    checkpoint = tmp_path / f"{benchmark.casefold()}.checkpoint.json"
    ledger = AtomicCheckpoint(
        checkpoint,
        manifest=before_manifest,
        expected_ids=("question-1",),
    )
    ledger.close()
    checkpoint_bytes = checkpoint.read_bytes()
    with pytest.raises(BenchmarkIntegrityError, match="run identity mismatch"):
        AtomicCheckpoint(
            checkpoint,
            manifest=after_manifest,
            expected_ids=("question-1",),
            resume=True,
        )
    assert checkpoint.read_bytes() == checkpoint_bytes


def test_identities_are_checkout_independent(tmp_path: Path):
    first = _write_source_tree(tmp_path / "checkout-a")
    second = _write_source_tree(tmp_path / "checkout-b")
    assert _hashes(first) == _hashes(second)


def test_reachable_archive_evidence_changes_all_four_identities(tmp_path: Path):
    paths = _write_source_tree(tmp_path / "archive-evidence")
    archive = paths["strictness"].with_name("archive_evidence.py")
    paths["archive_evidence"] = archive
    archive.write_text(
        "def checkpoint_attestation(value):\n    return value + 17\n"
        "def unused_archive_helper(value):\n    return value + 999\n",
        encoding="utf-8",
    )
    _replace(paths["strictness"], "def strict_guard(value):\n    return strict_helper(value)",
             "def strict_guard(value):\n    from benchmarks.archive_evidence import checkpoint_attestation\n    return checkpoint_attestation(strict_helper(value))")
    before = _hashes(paths)
    _replace(archive, "value + 17", "value + 18")
    after = _hashes(paths)
    _assert_changed(before, after, set(_HASHERS))
    _replace(archive, "value + 999", "value + 1000")
    assert _hashes(paths) == after


@pytest.mark.parametrize(
    ("path_key", "old", "new"),
    [
        ("hymem_markdown", "v1", "v2"),
        ("hymem_text", "v1", "v2"),
        ("hymem_unrelated", "1", "2"),
        ("benchmark_unrelated", "1", "2"),
        ("strictness", "value - 99", "value - 100"),
        ("canary", 'return "unused"', 'return "still-unused"'),
        ("store", "return value\n", "return value + 0\n"),
        ("msc_registry", "return value\n", "return value + 0\n"),
        ("run_registry", "return not value", "return bool(not value)"),
        (
            "protocol", 'UNUSED_PROTOCOL = "unused-v1"',
            'UNUSED_PROTOCOL = "unused-v2"',
        ),
    ],
)
def test_incidental_or_unreached_changes_do_not_invalidate(
    tmp_path: Path, path_key: str, old: str, new: str,
):
    paths = _write_source_tree(tmp_path / "sources")
    before = _hashes(paths)
    _replace(paths[path_key], old, new)
    assert _hashes(paths) == before


@pytest.mark.parametrize(
    ("path_key", "old", "new"),
    [
        ("strictness", "return value + 1", "return value + 2"),
        ("canary", "return canary_helper()", "return canary_helper() + 1"),
        ("strictness", "LOCK_BACKEND = None", "LOCK_BACKEND = False"),
        ("canary", "CANARY_LIMIT < 2", "CANARY_LIMIT < 1"),
        ("hymem_shared", "return db.VALUE", "return db.VALUE + 1"),
        ("hymem_core_init", "CORE_INIT = 1", "CORE_INIT = 2"),
        ("hymem_migrations_init", "= 1", "= 2"),
        ("hymem_schema", "shared_v1", "shared_v2"),
        ("hymem_migration", "note TEXT", "note BLOB"),
    ],
)
def test_shared_executable_changes_invalidate_all_benchmarks(
    tmp_path: Path, path_key: str, old: str, new: str,
):
    paths = _write_source_tree(tmp_path / "sources")
    before = _hashes(paths)
    _replace(paths[path_key], old, new)
    _assert_changed(before, _hashes(paths), set(_HASHERS))


@pytest.mark.parametrize(
    ("path_key", "old", "new", "expected"),
    [
        ("adapter", "return LME_PROTOCOL", "return LME_PROTOCOL + 'x'", {"lme"}),
        ("beam", "return beam_shared()", "return beam_shared() + 'x'", {"beam"}),
        ("msc", "return msc_lme_shared()", "return msc_lme_shared() + 'x'", {"msc"}),
        (
            "locomo", "return (msc_shared()", "return ('x' + msc_shared()",
            {"locomo"},
        ),
        ("hymem_lme", "return 3", "return 30", {"lme"}),
        ("hymem_beam", "return 4", "return 40", {"beam"}),
        ("hymem_msc", "return 5", "return 50", {"msc"}),
        ("hymem_locomo", "return 6", "return 60", {"locomo"}),
    ],
)
def test_own_executable_changes_only_invalidate_the_owner(
    tmp_path: Path,
    path_key: str,
    old: str,
    new: str,
    expected: set[str],
):
    paths = _write_source_tree(tmp_path / "sources")
    before = _hashes(paths)
    _replace(paths[path_key], old, new)
    _assert_changed(before, _hashes(paths), expected)


@pytest.mark.parametrize(
    ("path_key", "old", "new", "expected"),
    [
        (
            "adapter", "return BEAM_PROTOCOL", "return BEAM_PROTOCOL + 'b'",
            {"lme", "beam"},
        ),
        (
            "adapter", "return MSC_PROTOCOL", "return MSC_PROTOCOL + 'm'",
            {"lme", "msc"},
        ),
        (
            "adapter", "return TRANSITIVE_PROTOCOL",
            "return TRANSITIVE_PROTOCOL + 't'", {"lme", "msc", "locomo"},
        ),
        (
            "adapter", "return LOCOMO_PROTOCOL", "return LOCOMO_PROTOCOL + 'l'",
            {"lme", "locomo"},
        ),
        (
            "adapter", "MAX_CONTEXT_CHARS = 8000", "MAX_CONTEXT_CHARS = 9000",
            {"lme", "locomo"},
        ),
        (
            "adapter", "print(beam_shared())", "print(beam_shared(), flush=True)",
            {"lme"},
        ),
        (
            "adapter", 'BEAM_PROTOCOL == ""', "BEAM_PROTOCOL is None",
            {"lme", "beam"},
        ),
        (
            "msc", "return transitive_lme_shared()",
            "return transitive_lme_shared() + 's'", {"msc", "locomo"},
        ),
        (
            "store", "return strict_guard(value)",
            "return strict_guard(value) + 1", {"msc", "locomo"},
        ),
        (
            "msc_registry", "return strict_guard(value)",
            "return strict_guard(value) + 1", {"msc"},
        ),
        (
            "run_registry", "return bool(value)",
            "return bool(value) and True", {"lme"},
        ),
        (
            "protocol", 'LME_PROTOCOL = "lme-v1"',
            'LME_PROTOCOL = "lme-v2"', {"lme"},
        ),
        (
            "protocol", 'BEAM_PROTOCOL = "beam-v1"',
            'BEAM_PROTOCOL = "beam-v2"', {"lme", "beam"},
        ),
        (
            "protocol", 'MSC_PROTOCOL = "msc-v1"',
            'MSC_PROTOCOL = "msc-v2"', {"lme", "msc"},
        ),
        (
            "protocol", 'TRANSITIVE_PROTOCOL = "transitive-v1"',
            'TRANSITIVE_PROTOCOL = "transitive-v2"',
            {"lme", "msc", "locomo"},
        ),
        (
            "protocol", 'LOCOMO_PROTOCOL = "locomo-v1"',
            'LOCOMO_PROTOCOL = "locomo-v2"', {"lme", "locomo"},
        ),
    ],
)
def test_exact_cross_adapter_and_evidence_dependencies_are_transitive(
    tmp_path: Path,
    path_key: str,
    old: str,
    new: str,
    expected: set[str],
):
    paths = _write_source_tree(tmp_path / "sources")
    before = _hashes(paths)
    _replace(paths[path_key], old, new)
    _assert_changed(before, _hashes(paths), expected)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            'INDEXING_PROVENANCE_VERSION = "hymem-benchmark-indexing-v4"',
            'INDEXING_PROVENANCE_VERSION = "hymem-benchmark-indexing-v5"',
        ),
        (
            'STORE_BUILD_RECEIPT_VERSION = "hymem-benchmark-store-build-v7"',
            'STORE_BUILD_RECEIPT_VERSION = "hymem-benchmark-store-build-v8"',
        ),
        (
            '"hymem-benchmark-store-indexing-attestation-v2"',
            '"hymem-benchmark-store-indexing-attestation-v3"',
        ),
    ),
)
def test_msc_receipt_protocol_constants_are_transitive_code_identity(
    tmp_path: Path, old: str, new: str,
):
    paths = _write_source_tree(tmp_path / "sources")
    before = _hashes(paths)
    _replace(paths["msc"], old, new)
    after = _hashes(paths)

    _assert_changed(before, after, {"msc", "locomo"})
    for benchmark in ("msc", "locomo"):
        _assert_identity_change_rejects_old_checkpoint(
            tmp_path / benchmark,
            benchmark=benchmark,
            before_hash=before[benchmark],
            after_hash=after[benchmark],
        )


def test_shared_decorator_changes_are_executable_identity(tmp_path: Path):
    paths = _write_source_tree(tmp_path / "sources")
    before = _hashes(paths)
    _replace(
        paths["adapter"],
        "@identity_decorator(frozen=True)",
        "@identity_decorator(frozen=False)",
    )
    _assert_changed(before, _hashes(paths), {"lme", "beam"})


@pytest.mark.parametrize("name", sorted(_HASHERS))
def test_code_change_rejects_stale_checkpoint(tmp_path: Path, name: str):
    paths = _write_source_tree(tmp_path / "sources")
    before = _HASHERS[name](paths)
    _replace(
        paths["canary"], "return canary_helper()", "return canary_helper() + 1"
    )
    _assert_identity_change_rejects_old_checkpoint(
        tmp_path,
        benchmark=name,
        before_hash=before,
        after_hash=_HASHERS[name](paths),
    )


def test_missing_imported_shared_symbol_fails_closed(tmp_path: Path):
    paths = _write_source_tree(tmp_path / "sources")
    _replace(paths["beam"], "import beam_shared", "import missing_shared")
    with pytest.raises(BenchmarkIntegrityError, match="lacks requested symbols"):
        _beam_hash(paths)
