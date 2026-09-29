"""Offline Q1 package inspection/sealing; no execution, network or credentials.

Default is read-only inspection. Sealing requires an explicit approved manifest
pin and --seal-approved-source; neither option grants permission to run a test
against a provider. Only source files named by the accepted manifest are copied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import types

BASE = Path(__file__).resolve().parent
PENDING = BASE / "pending"
PREVIOUS = BASE / "previous-v3"
PREVIOUS_PIN = "e6be83fee57c238c88d989bc7d8f49fc2016e62e7642636dc4cfd240f5f1868d"
HELPER_PINS = {
    "RUNBOOK.md": "829779e710d9e1d4ee6d70e4dafcd6d07fd11ad3fc0efada2f9c7cc4fe342c60",
    "lme_q1_startup_preflight.py": "c5219eb1ca8589cabcbec03d8a9a4c731e98f790d1ba4b107dd7e16c48bcf79a",
    "q1_stock_host.py": "b2f87d3f368d78994e2210e3084bfeb6592748f090abe0af70f0238bad51ab21",
    "q1_stock_run.py": "9eb95e439037d22116796565511d99400e7b51edf3f24048279ac85c4f38ae26",
    "q1_stock_validate.py": "c4e3ccdef2d0a558d43be1d800192dd94f747bbf233a8fbfbe7c09154e601db3",
    "supervised_invocation.py": "9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc",
    "transport_common.py": "1c98c73cfb1b607113c726616058862fd6bb16a57901a57c8962ee4a0a2807a0",
}
REMOTE_BASE = "/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ"
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def need(value, label):
    if not value:
        raise ValueError("q1_seal_" + label)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: need(False, "nonfinite_json"))


def regular_bytes(path):
    need(path.is_absolute() and path.resolve() == path, "noncanonical_path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        need(stat.S_ISREG(info.st_mode) and info.st_size <= 32 * 1024 * 1024, "regular_file")
        raw = stream.read(32 * 1024 * 1024 + 1)
    need(len(raw) <= 32 * 1024 * 1024, "file_size")
    return raw


def mapping(value, *, source=False):
    need(type(value) is dict and bool(value), "empty_mapping")
    for name, digest in value.items():
        path = PurePosixPath(name) if type(name) is str else None
        need(path is not None and name == str(path) and not path.is_absolute()
             and ".." not in path.parts and all(not part.startswith(".") for part in path.parts)
             and type(digest) is str and DIGEST.fullmatch(digest), "mapping_entry")
        if source:
            need(path.suffix in {".py", ".sql", ".toml"}, "non_source_file")
    return value


def exact_tree(root, expected):
    """Inventory before reading contents: never open unexpected/secret files."""
    need(root.is_absolute() and root.resolve() == root and root.is_dir(), "tree_root")
    actual = set()
    def failed(_error):
        raise ValueError("q1_seal_tree_traversal")
    for directory, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in directories + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            need(not stat.S_ISLNK(mode), "tree_symlink")
            need(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "tree_special_file")
            if stat.S_ISREG(mode):
                actual.add(path.relative_to(root).as_posix())
    need(actual == set(expected), "tree_inventory")
    captured = {}
    for name, digest in expected.items():
        raw = regular_bytes(root / name)
        need(sha(raw) == digest, "tree_hash")
        captured[name] = raw
    return captured


def load_definitions(raw, name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def prepare(*, source_manifest, approved_manifest_sha256, tree, layout,
            remote_root, remote_source):
    need(type(approved_manifest_sha256) is str and DIGEST.fullmatch(approved_manifest_sha256), "approved_pin")
    raw_manifest = regular_bytes(source_manifest)
    need(sha(raw_manifest) == approved_manifest_sha256, "approved_manifest_drift")
    original = decode(raw_manifest)
    need(original.get("revision") == "r5", "final_revision_required")
    sources = mapping(original.get("source_sha256"), source=True)
    need(layout in {"source", "verification"}, "tree_layout")
    expected = dict(sources)
    if layout == "verification":
        for key in ("test_sha256", "auxiliary_sha256"):
            values = mapping(original.get(key))
            need(not (set(expected) & set(values)), "overlapping_maps")
            expected.update(values)
    captured = exact_tree(tree, expected)
    helpers = exact_tree(PENDING, HELPER_PINS)
    historical_raw = regular_bytes(PREVIOUS / "manifest.json")
    need(sha(historical_raw) == PREVIOUS_PIN, "previous_manifest_drift")
    historical = decode(historical_raw)
    runner = load_definitions(helpers["q1_stock_run.py"], "q1_seal_runner", PENDING / "q1_stock_run.py")
    host = load_definitions(helpers["q1_stock_host.py"], "q1_seal_host", PENDING / "q1_stock_host.py")
    # Only the independently byte-pinned label-blind selector is executed,
    # never arbitrary source modules or the benchmark entry point.
    runner.derive_seed(runner.selector_from_source(tree / "benchmarks/longmemeval_adapter.py"))
    need(runner.stock_arguments() == historical["stock_arguments"], "recipe_drift")
    need(runner.DATASET_SHA == historical["dataset_sha256"]
         and runner.MODEL == historical["requested_model"]
         and runner.ENDPOINT == historical["endpoint"]
         and host.IMAGE == historical["docker_image"], "fixed_contract_drift")
    remote_match = re.fullmatch(re.escape(REMOTE_BASE) + r"/q1-stock-v([1-9][0-9]{0,2})", remote_root) if type(remote_root) is str else None
    need(remote_match is not None and int(remote_match[1]) >= 4, "remote_root")
    need(remote_source == REMOTE_BASE + "/offline-r5/candidate", "remote_source")
    host.command(remote_root, remote_source, "0" * 64)  # Pure plan validation; never executes Docker.
    manifest = {
        "schema": historical["schema"], "status": "sealed_local_only_not_launched",
        "preparation_revision": int(remote_match[1]), "previous_preparation_manifest_sha256": PREVIOUS_PIN,
        "approved_source_manifest_sha256": approved_manifest_sha256,
        "source_revision": "r5", "source_sha256": sources, "source_files": len(sources),
        "source_mapping_sha256": sha(runner.canonical(sources)),
        "source_inventory_policy": "exact-manifest-regular-files-no-symlinks",
        "helper_sha256": HELPER_PINS, "dataset_sha256": runner.DATASET_SHA,
        "docker_image": host.IMAGE, "runtime_host_path": host.RUNTIME,
        "runtime_python": runner.PYTHON, "requested_model": runner.MODEL,
        "target_runtime_expected_versions": {"python": "3.11.2", "pytest": "9.1.1", "openai": "2.53.0",
                                             "requests": "2.34.2", "httpx": "0.28.1"},
        "runtime_verified_by_sealing": False,
        "endpoint": runner.ENDPOINT, "thinking": "disabled",
        "stock_arguments": runner.stock_arguments(), "supervision_seconds": runner.TIMEOUT,
        "cleanup_seconds": 10, "indexing_max_cycles": 100, "indexing_timeout_seconds": 3600,
        "seed": runner.SEED, "source_index": runner.SOURCE_INDEX, "question_id": runner.QUESTION_ID,
        "selector_sha256": runner.SELECTOR_SHA, "sessions": 44, "messages": 479,
        "source_question_count": 500, "remote_run_root": remote_root, "remote_source": remote_source,
        "global_paid_call_cap": None, "rerolls_allowed": False, "resume_allowed": False,
        "production_memory_mounted": False, "api_calls_performed_by_preparation": 0,
        "deployment_performed": False, "benchmark_executed": False,
        "full_suite_pass_claimed": False, "target_runtime_gate_pass_claimed": False,
        "model_version_immutably_pinned": False,
        "launch_gate": "requires_separate_parent_review_target_runtime_gate_and_authorization",
        "postvalidation": historical["postvalidation"],
        "verification_scope": "Exact bytes, selector and stock recipe only; not a passing-test, live-runtime or benchmark-readiness claim.",
    }
    return manifest, {name: captured[name] for name in sources}, helpers, raw_manifest


def exclusive(path, raw):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    need(path.parent.resolve() == path.parent, "output_parent")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def seal(prepared, *, output, input_tree):
    need(output.is_absolute() and output.resolve() == output
         and output.parent.is_dir() and not output.is_relative_to(input_tree)
         and not output.is_relative_to(BASE), "output_scope")
    # Refuse every existing target, including dangling links. Keep any partial
    # new output on failure for inspection; never overwrite, retry or delete it.
    output.mkdir(mode=0o700)
    manifest, sources, helpers, raw_input_manifest = prepared
    payload = {"input-manifest.json": raw_input_manifest,
               "bundle/manifest.json": encoded(manifest),
               **{"candidate/" + name: raw for name, raw in sources.items()},
               **{"bundle/" + name: raw for name, raw in helpers.items()}}
    for name, raw in sorted(payload.items()):
        exclusive(output / name, raw)
    pins = {name: sha(raw) for name, raw in payload.items()}
    exact_tree(output, pins)
    receipt = {"schema": "stock-q1-local-seal-v1", "status": "sealed_local_only_not_launched",
               "package_manifest_sha256": pins["bundle/manifest.json"], "files_sha256": pins,
               "source_files": len(sources), "helper_files": len(helpers),
               "source_manifest_sha256": manifest["approved_source_manifest_sha256"],
               "paid_calls": 0, "remote_operations": 0, "benchmark_readiness_claimed": False}
    exclusive(output / "seal-receipt.json", encoded(receipt))
    exact_tree(output, {**pins, "seal-receipt.json": sha(encoded(receipt))})
    return receipt


def verify_sealed(*, output, expected_receipt_sha256):
    """Recheck an independently pinned seal receipt and every output byte."""
    need(type(expected_receipt_sha256) is str and DIGEST.fullmatch(expected_receipt_sha256), "receipt_pin")
    raw = regular_bytes(output / "seal-receipt.json")
    need(sha(raw) == expected_receipt_sha256, "receipt_drift")
    receipt = decode(raw)
    need(receipt.get("schema") == "stock-q1-local-seal-v1", "receipt_schema")
    pins = mapping(receipt.get("files_sha256"))
    exact_tree(output, {**pins, "seal-receipt.json": expected_receipt_sha256})
    manifest = decode(regular_bytes(output / "bundle/manifest.json"))
    source = mapping(manifest.get("source_sha256"), source=True)
    need(manifest.get("helper_sha256") == HELPER_PINS
         and pins["bundle/manifest.json"] == receipt.get("package_manifest_sha256")
         and pins["input-manifest.json"] == receipt.get("source_manifest_sha256"), "receipt_crosslinks")
    need(set(pins) == {"bundle/manifest.json", "input-manifest.json",
                      *("candidate/" + name for name in source),
                      *("bundle/" + name for name in HELPER_PINS)}, "sealed_inventory")
    need(all(pins["candidate/" + name] == digest for name, digest in source.items())
         and all(pins["bundle/" + name] == digest for name, digest in HELPER_PINS.items()), "sealed_maps")
    return {"status": "sealed_bytes_verified_only", "package_manifest_sha256": receipt["package_manifest_sha256"],
            "source_files": len(source), "paid_calls": 0, "remote_operations": 0,
            "benchmark_readiness_claimed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--approved-manifest-sha256")
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--layout", choices=("source", "verification"))
    parser.add_argument("--remote-root")
    parser.add_argument("--remote-source")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seal-approved-source", action="store_true")
    parser.add_argument("--verify-sealed", type=Path)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args()
    if args.verify_sealed is not None:
        need(not args.seal_approved_source and all(getattr(args, field) is None for field in (
            "source_manifest", "approved_manifest_sha256", "tree", "layout", "remote_root", "remote_source", "output")), "verify_mode")
        print(json.dumps(verify_sealed(output=args.verify_sealed, expected_receipt_sha256=args.receipt_sha256), sort_keys=True))
        return
    need(args.receipt_sha256 is None and all(getattr(args, field) is not None for field in (
        "source_manifest", "approved_manifest_sha256", "tree", "layout", "remote_root", "remote_source")), "preparation_arguments")
    need(args.seal_approved_source == (args.output is not None), "output_requires_explicit_seal")
    prepared = prepare(source_manifest=args.source_manifest, approved_manifest_sha256=args.approved_manifest_sha256,
                       tree=args.tree, layout=args.layout, remote_root=args.remote_root, remote_source=args.remote_source)
    if args.seal_approved_source:
        result = seal(prepared, output=args.output, input_tree=args.tree)
        print(json.dumps({key: value for key, value in result.items() if key != "files_sha256"}, sort_keys=True))
    else:
        print(json.dumps({"status": "inspected_only_no_writes", "source_files": len(prepared[1]),
                          "helper_files": len(prepared[2]), "source_manifest_sha256": args.approved_manifest_sha256,
                          "paid_calls": 0, "remote_operations": 0, "benchmark_readiness_claimed": False}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({"status": "offline_seal_failed", "exception_type": type(exc).__name__, "paid_calls": 0}))
        raise SystemExit(1)
