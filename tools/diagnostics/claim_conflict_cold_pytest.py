"""One-shot, network-none full suite for the reviewed cold-import candidate.

This adapter reuses the hash-pinned v2 suite runner. Its v3 proof receipt
authenticates the *parent* candidate only; the new candidate is independently
bound to that 481-file manifest plus one exact phase1 replacement. Preparation
and remote installation remain disabled until final review is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
ROOT = BASE / "cold-replay-pytest-v1"
SELF = ROOT / "claim_conflict_cold_pytest.py"
WORK = ROOT / "work"
OVERLAY = ROOT / "overlay"
CANDIDATE = ROOT / "candidate"
PHASE1_UPLOAD = ROOT / "overrides/hymem/dreaming/phase1.py"
BASE_SUITE = BASE / "proof-pytest-v2/claim_conflict_proof_pytest.py"
LOCAL_BASE_SUITE = Path(__file__).resolve().with_name("claim_conflict_proof_pytest.py")
BASE_SUITE_SHA = "3b4a1180dc50e176ea1964534f2562b4cc1a60425e04485ccc056f747b0f5a5e"
BASE_CONTAINER_SUITE = Path("/diag/base_suite.py")
PARENT_CANDIDATE = BASE / "proof-replay-v3/candidate"
PARENT_CANDIDATE_SHA = "cd6e7810a4c08d771c49658a5954ca71a24d9d1119bcfde3aaa4ccb86dafd694"
PARENT_PHASE1_SHA = "1037bf6d62c3981add3702f96d2ce5bd79d8ebc831f15b30f95bb3724cfd7217"
NEW_PHASE1_SHA = "bc47739973a7d5c4825505f83486951b11e6b1ca0d4eeec8ab450dd9fc3272ac"
BASELINE_PROOF_RESULT_SHA = "6e602bae89aee5f42be1f3f7efff87ba905a59bc134d09015ffe7a02a672c52d"
PARENT_OVERLAY_SHA = "978b687ae717fd4a5166b7710715105f4ab092dfab9ba6d19e886b9391b65247"
NEW_OVERLAY_SHA = "0321fd5419d8079a2cb6ea9802ef4c64df0294a3d00dc7f4204b237b67608344"
TEST_ROOT = Path("/private/tmp/hymem-r7-cold-replay-tests.4BJkt5/tests")
PHASE1_SOURCE = Path("/private/tmp/hymem-r7-cold-replay-tests.4BJkt5/hymem/dreaming/phase1.py")
PARENT_TEST_NAMES = (
    "test_fact_authority.py", "test_claim_semantic_dedup_guard.py",
    "test_alias_registration_idempotence.py", "test_alias_registration_root_controls.py",
    "test_claim_replay_binding_regressions.py", "test_claim_replay_proof_root.py",
    "test_claim_replay_local_proof_root.py", "test_shared_embedding_bounds.py",
    "test_embedding_bounds_root_controls.py", "test_chunk_embedding_batches.py",
    "test_chunk_embedding_batches_root.py", "test_claim_replay_r7_upgrade_root.py",
    "test_summary_recovery_v63.py", "test_r7_v64_summary_preservation.py",
)
NEW_TEST_NAMES = (
    "test_claim_cold_import_replay.py", "test_claim_cold_import_replay_root.py",
    "test_claim_cold_replay_reviewer_retired.py",
)
TEST_NAMES = PARENT_TEST_NAMES + NEW_TEST_NAMES
REVIEWED_FINAL_CANDIDATE = True  # Parent reviewed exact app/tests and 59 launcher controls.
FULL_SECONDS = 10800
HOST_SECONDS = 11200
MIN_COLLECTED = 7752


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError(code)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def load_base(*, local_prepare: bool = False):
    # A local prepare must use the reviewed adjacent source. Remote host and
    # container paths are separate explicit pins, never fallback candidates.
    path = (
        LOCAL_BASE_SUITE if local_prepare else
        BASE_CONTAINER_SUITE if BASE_CONTAINER_SUITE.is_file() else BASE_SUITE
    )
    need(path.is_file() and not path.is_symlink() and sha(path) == BASE_SUITE_SHA,
         "base_suite_pin_drift")
    spec = importlib.util.spec_from_file_location("cold_pinned_base_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def overlay_manifest(test_root: Path = TEST_ROOT) -> dict[str, str]:
    need(test_root == TEST_ROOT and test_root.is_dir() and not test_root.is_symlink(),
         "reviewed_test_root_required")
    manifest = {}
    for name in TEST_NAMES:
        path = test_root / name
        need(path.is_file() and not path.is_symlink(), "test_source_missing")
        manifest["tests/" + name] = sha(path)
    parent = {name: manifest[name] for name in ("tests/" + item for item in PARENT_TEST_NAMES)}
    need(digest(parent) == PARENT_OVERLAY_SHA
         and digest(manifest) == NEW_OVERLAY_SHA, "cold_test_overlay_pin_drift")
    return manifest


def derive_candidate_manifest(parent: dict[str, str]) -> dict[str, str]:
    need(len(parent) == 481 and digest(parent) == PARENT_CANDIDATE_SHA
         and parent.get("hymem/dreaming/phase1.py") == PARENT_PHASE1_SHA,
         "parent_candidate_manifest_drift")
    changed = dict(parent)
    changed["hymem/dreaming/phase1.py"] = NEW_PHASE1_SHA
    need(len(changed) == 481 and all(
        changed[name] == value for name, value in parent.items()
        if name != "hymem/dreaming/phase1.py"
    ), "cold_candidate_not_single_file_delta")
    return changed


def configure_base(*, local_prepare: bool = False):
    base = load_base(local_prepare=local_prepare)
    # The v3 replay receipt authenticates the unchanged parent candidate.
    # Keep its checker in a separate pinned module: new source_pins mutates
    # base.CANDIDATE_SHA to the cold candidate's digest during inherited pins.
    baseline_suite = load_base(local_prepare=local_prepare)
    base.baseline_suite = baseline_suite
    base.ROOT, base.SELF, base.WORK = ROOT, SELF, WORK
    base.OVERLAY, base.CANDIDATE = OVERLAY, CANDIDATE
    base.TEST_ROOT, base.FACT_FIX = TEST_ROOT, TEST_ROOT / "test_fact_authority.py"
    base.TEST_NAMES = TEST_NAMES
    base.REVIEWED_OVERLAY_SHA = NEW_OVERLAY_SHA
    base.CANDIDATE_SHA = PARENT_CANDIDATE_SHA  # Shape-valid until the exact new manifest is derived.
    base.FULL_SECONDS, base.HOST_SECONDS = FULL_SECONDS, HOST_SECONDS
    base.MIN_COLLECTED = MIN_COLLECTED
    base.__file__ = str(SELF if SELF.is_file() else Path(__file__))
    original_configure, original_pins = base.configure, base.pins

    def source_pins(shared, proof, helper):
        parent = shared.baseline_inventory(helper)
        parent.update(shared.OVERRIDE_SHAS)
        parent.update(proof.reviewed_overrides())
        changed = derive_candidate_manifest(parent)
        need(base.inventory(PARENT_CANDIDATE) == parent, "parent_candidate_pin_drift")
        need(base.inventory(CANDIDATE) == changed, "cold_candidate_pin_drift")
        base.CANDIDATE_SHA = digest(changed)
        return changed

    def pins(proof, shared, helper):
        receipt = original_pins(proof, shared, helper)
        need(receipt["proof_result_sha256"] == BASELINE_PROOF_RESULT_SHA,
             "baseline_proof_result_drift")
        return {**receipt, "baseline_proof_only": True,
                "new_candidate_replay_verified": False,
                "parent_candidate_sha256": PARENT_CANDIDATE_SHA,
                "candidate_phase1_sha256": NEW_PHASE1_SHA}

    def replay_gate(proof, helper):
        return baseline_suite.replay_gate(proof, helper)

    def configure(helper):
        command, mounts = original_configure(helper)
        extra = (str(BASE_SUITE), str(BASE_CONTAINER_SUITE), False)
        mounts.append(extra)
        position = command.index("--workdir")
        command[position:position] = [
            "--mount", "type=bind,src=" + extra[0] + ",dst=" + extra[1] + ",readonly"
        ]
        command[command.index("--name") + 1] = "hymem-cold-replay-pytest-v1"
        return command, mounts

    base.source_pins, base.pins = source_pins, pins
    base.replay_gate, base.configure = replay_gate, configure
    return base


def prepare(destination: Path):
    need(REVIEWED_FINAL_CANDIDATE, "final_cold_candidate_review_pending")
    need(PHASE1_SOURCE.is_file() and not PHASE1_SOURCE.is_symlink()
         and sha(PHASE1_SOURCE) == NEW_PHASE1_SHA, "cold_phase1_source_pin_drift")
    overlay_manifest()
    base = configure_base(local_prepare=True)
    result = base.prepare(destination, TEST_ROOT)
    target = destination / "overrides/hymem/dreaming/phase1.py"
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(PHASE1_SOURCE, target)
    target.chmod(0o400)
    need(sha(target) == NEW_PHASE1_SHA, "cold_phase1_copy_drift")
    return {**result, "phase1_sha256": NEW_PHASE1_SHA,
            "baseline_proof_only": True, "new_candidate_replay_verified": False}


def remote_install(base):
    need(REVIEWED_FINAL_CANDIDATE, "final_cold_candidate_review_pending")
    need(not CANDIDATE.exists() and not WORK.exists(), "cold_install_already_attempted")
    need(PHASE1_UPLOAD.is_file() and not PHASE1_UPLOAD.is_symlink()
         and sha(PHASE1_UPLOAD) == NEW_PHASE1_SHA,
         "cold_phase1_upload_pin_drift")
    proof, shared, helper = base.dependencies()
    parent = shared.baseline_inventory(helper)
    parent.update(shared.OVERRIDE_SHAS)
    parent.update(proof.reviewed_overrides())
    changed = derive_candidate_manifest(parent)
    need(base.inventory(PARENT_CANDIDATE) == parent, "parent_candidate_pin_drift")
    need(base.replay_gate(proof, helper) == BASELINE_PROOF_RESULT_SHA,
         "baseline_proof_not_passed")
    need(base.inventory(OVERLAY) == helper.read_json(ROOT / "test-overlay.json")
         and digest(base.inventory(OVERLAY)) == NEW_OVERLAY_SHA,
         "cold_test_overlay_pin_drift")
    shutil.copytree(PARENT_CANDIDATE, CANDIDATE, symlinks=False)
    target = CANDIDATE / "hymem/dreaming/phase1.py"
    target.chmod(0o600)
    shutil.copyfile(PHASE1_UPLOAD, target)
    target.chmod(0o400)
    need(base.inventory(CANDIDATE) == changed, "cold_candidate_copy_drift")
    return base.remote("remote-install")


def worker(base):
    manifest = json.loads(base.SOURCE_MANIFEST.read_text())
    parent = dict(manifest)
    parent["hymem/dreaming/phase1.py"] = PARENT_PHASE1_SHA
    need(derive_candidate_manifest(parent) == manifest,
         "worker_cold_candidate_manifest_drift")
    base.CANDIDATE_SHA = digest(manifest)
    return base.worker()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "prepare", "remote-install", "remote-launch", "remote-status",
        "supervise", "worker", "pytest-collect", "pytest-full",
    ))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            need(args.output is not None, "output_required")
            result = prepare(args.output)
        else:
            if args.action in ("remote-install", "remote-launch", "supervise"):
                need(REVIEWED_FINAL_CANDIDATE, "final_cold_candidate_review_pending")
            base = configure_base()
            if args.action == "remote-install":
                result = remote_install(base)
            elif args.action == "worker":
                result = worker(base)
            elif args.action.startswith("pytest-"):
                return base.pytest_stage(args.action.removeprefix("pytest-"))
            else:
                result = base.remote(args.action)
    except BaseException:
        result = {"status": "failed_inspect_private_artifacts"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in (
        "prepared_not_uploaded", "installed_not_launched",
        "detached_supervisor_started", "passed", "not_terminal",
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
