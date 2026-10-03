"""Sequential, restartable 500-row Luna no-dream LME campaign controller.

Preparing and status make no model calls. Running requires the SHA-256 of a
write-once plan, admits one one-shot capsule at a time, and never retries a
failed capsule. This controller produces diagnostic evidence, not an official
LongMemEval score.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

HOST_ROOT = Path("/home/atta")
CAMPAIGN_NAME = re.compile(r"\.hymem-siwc-lme-nodream-campaign-[a-z0-9-]{8,}\Z")
VERIFIER = Path(__file__).with_name("siwc_lme_aggregate_v1.py")
LAUNCHER = Path(__file__).with_name("siwc_lme_launch_nodream_v12.py").resolve()
RUNTIME = Path("/home/atta/.hymem-siwc-runtime-v1/bin/python")
SCHEMA = "siwc-lme-luna-nodream-campaign-v1"
VARIANT = "no_dream_diagnostic_v13"
TOTAL_ROWS = 500
HEX = re.compile(r"[0-9a-f]{64}\Z")
POLL_SECONDS = 30
MAX_HOST_IDLE_WAIT_SECONDS = 600
MAX_MANIFEST_BYTES = 262_144
MAX_RESULT_BYTES = 262_144
MAX_BUDGET_REVISIONS = 100

_launcher_spec = importlib.util.spec_from_file_location("siwc_lme_campaign_launcher", LAUNCHER)
if _launcher_spec is None or _launcher_spec.loader is None:
    raise RuntimeError("campaign_launcher_import_invalid")
launcher = importlib.util.module_from_spec(_launcher_spec)
sys.modules[_launcher_spec.name] = launcher
_launcher_spec.loader.exec_module(launcher)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, maximum: int | None = None) -> bool:
    try:
        info = path.lstat()
        return (stat.S_ISREG(info.st_mode) and not path.is_symlink()
            and (maximum is None or info.st_size <= maximum))
    except OSError:
        return False


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")


def write_once(path: Path, value: dict, maximum: int) -> None:
    encoded = canonical(value)
    require(len(encoded) <= maximum, "campaign_metadata_too_large")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def record_progress(root: Path, manifest_sha256: str, *, state: str,
                    reason: str | None, offset: int, rows_verified: int,
                    known_tokens: int, turns: int, wall_seconds: int) -> None:
    """Persist only bounded numeric progress and a safe controller reason code."""
    require(state in {"running", "paused", "stopped", "complete"}
        and (reason is None or re.fullmatch(r"[a-z0-9_]{1,80}", reason) is not None)
        and all(type(value) is int and 0 <= value <= 10**11 for value in
            (offset, rows_verified, known_tokens, turns, wall_seconds)),
        "campaign_progress_invalid")
    payload = {"schema": SCHEMA + "-progress-v1",
        "manifest_sha256": manifest_sha256, "state": state,
        "reason": reason, "next_offset": offset,
        "rows_verified": rows_verified, "known_tokens": known_tokens,
        "turns": turns, "observed_wall_seconds": wall_seconds}
    encoded = canonical(payload)
    require(len(encoded) <= 2048, "campaign_progress_invalid")
    pending = root / f"campaign-status.json.pending-{os.getpid()}"
    fd = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, root / "campaign-status.json")
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        pending.unlink(missing_ok=True)


def checked_root(root: Path, *, exists: bool) -> Path:
    require(root.is_absolute() and root.parent == HOST_ROOT
        and CAMPAIGN_NAME.fullmatch(root.name) is not None
        and ((root.is_dir() and not root.is_symlink()) if exists
             else (not root.exists() and not root.is_symlink())),
        "campaign_root_invalid")
    if exists:
        info = root.stat()
        require(info.st_uid == os.getuid() and not info.st_mode & 0o077,
            "campaign_root_permission_invalid")
    return root


def runtime_admission() -> None:
    require(os.getuid() == launcher.HOST_UID and os.geteuid() == launcher.HOST_UID,
        "host_user_invalid")
    require(Path(sys.executable).resolve() == RUNTIME.resolve(), "runtime_invalid")
    require(regular(LAUNCHER) and regular(VERIFIER), "controller_source_missing")
    require(regular(launcher.DATASET) and sha(launcher.DATASET) == launcher.DATASET_SHA256,
        "dataset_drift")
    require(regular(launcher.SOURCE_ROOT / "source-map.json")
        and sha(launcher.SOURCE_ROOT / "source-map.json") == launcher.INVENTORY_SHA256,
        "source_inventory_drift")
    require(regular(launcher.SOURCE_ROOT / launcher.RUNNER_RELATIVE)
        and sha(launcher.SOURCE_ROOT / launcher.RUNNER_RELATIVE) == launcher.RUNNER_SHA256,
        "runner_source_drift")


def _planned_windows(root: Path, batch_size: int, adopted_windows: list[dict],
                     prior: list[dict], allow_prior_overlap: bool) -> list[dict]:
    slug = hashlib.sha256(str(root).encode("ascii")).hexdigest()[:12]
    planned = [dict(item) for item in adopted_windows]
    offset = sum(item["count"] for item in planned)
    while offset < TOTAL_ROWS:
        count = 1 if offset == 0 else min(batch_size, TOTAL_ROWS - offset)
        capsule = HOST_ROOT / f".hymem-siwc-lme-nodream-c{slug}-o{offset:03d}-n{count}"
        overlaps = sorted(item["root"] for item in prior
            if offset < item["offset"] + item["count"]
            and item["offset"] < offset + count and item["root"] != str(capsule))
        require(allow_prior_overlap or not overlaps,
            "prior_window_overlap_requires_explicit_admission")
        require(not capsule.exists(), "planned_capsule_already_exists")
        require(launcher.ROOT_NAME.fullmatch(capsule.name) is not None
            and "_" not in capsule.name, "planned_capsule_name_invalid")
        planned.append({"offset": offset, "count": count, "workers": count,
            "root": str(capsule), "adopted": False,
            "repeat_window_authorized": bool(overlaps),
            "prior_overlap_roots": overlaps})
        offset += count
    require(sum(item["count"] for item in planned) == TOTAL_ROWS,
        "campaign_selection_invalid")
    return planned


def prepare(root: Path, *, batch_size: int, adopted_root: Path | None,
            resume_from: Path | None,
            allow_prior_overlap: bool, max_known_tokens: int, max_turns: int,
            max_wall_seconds: int) -> dict:
    checked_root(root, exists=False)
    runtime_admission()
    require(type(batch_size) is int and 1 <= batch_size <= 4
        and type(allow_prior_overlap) is bool
        and type(max_known_tokens) is int and 0 < max_known_tokens <= 10**10
        and type(max_turns) is int and 0 < max_turns <= 10**7
        and type(max_wall_seconds) is int and 0 < max_wall_seconds <= 365 * 86400,
        "campaign_limits_invalid")
    require(adopted_root is None or resume_from is None,
        "campaign_adoption_ambiguous")
    adopted_windows: list[dict] = []
    resumed_from_sha256 = None
    if adopted_root is not None:
        launcher.checked_root(adopted_root)
        selected = launcher.preflight(adopted_root)
        require(selected["source_offset"] == 0 and selected["selected_count"] == 1
            and selected["workers"] == 1
            and regular(adopted_root / "launch-attempt.json", 512),
            "adopted_smoke_invalid")
    if resume_from is not None:
        old = load_manifest(resume_from)
        source_admission(old)
        verifier = load_verifier(old)
        prefix, _, _, _ = _verified_prefix(old, verifier)
        require(prefix and len(prefix) < len(old["windows"]),
            "resume_prefix_invalid")
        verifier.verify_campaign(launcher.DATASET,
            [Path(item["root"]) for item in prefix], require_full=False)
        adopted_windows = [{**item, "adopted": True,
            "adopted_receipt_sha256": sha(Path(item["root"]) / "launch-receipt.json")}
            for item in prefix]
        resumed_from_sha256 = sha(resume_from / "manifest.json")
    with launcher.selection_lock():
        prior = launcher.launched_batches()
        if adopted_root is not None:
            receipt = json.loads((adopted_root / "launch-receipt.json").read_text(
                encoding="ascii"))
            overlaps = sorted(item["root"] for item in prior
                if item["offset"] < 1 and item["offset"] + item["count"] > 0
                and item["root"] != str(adopted_root))
            require(receipt.get("repeat_window_authorized") == bool(overlaps),
                "adopted_overlap_identity_invalid")
            adopted_windows = [{"offset": 0, "count": 1, "workers": 1,
                "root": str(adopted_root), "adopted": True,
                "repeat_window_authorized": bool(overlaps),
                "prior_overlap_roots": overlaps,
                "adopted_receipt_sha256": sha(adopted_root / "launch-receipt.json")}]
        windows = _planned_windows(root, batch_size, adopted_windows, prior,
            allow_prior_overlap)
        manifest = {"schema": SCHEMA, "campaign_root": str(root),
            "dataset_sha256": launcher.DATASET_SHA256,
            "inventory_sha256": launcher.INVENTORY_SHA256,
            "launcher_sha256": sha(LAUNCHER), "runner_sha256": launcher.RUNNER_SHA256,
            "controller_sha256": sha(Path(__file__)),
            "verifier_sha256": sha(VERIFIER),
            "runtime_path": str(RUNTIME), "model": "gpt-5.6-luna",
            "auth": "siwc_oauth", "endpoint": "https://api.openai.com/v1/responses",
            "api_fallback_allowed": False, "one_shot_windows": True,
            "billing_policy": "siwc_server_enforced_plan_or_existing_credits_v1",
            "variant": VARIANT,
            "canonical_r9_artifact": False, "official_model_score": False,
            "batch_size": batch_size, "max_known_tokens": max_known_tokens,
            "max_turns": max_turns, "max_wall_seconds": max_wall_seconds,
            "resumed_from_manifest_sha256": resumed_from_sha256,
            "adopted_prefix_rows": sum(item["count"] for item in adopted_windows),
            "windows": windows}
        root.mkdir(mode=0o700)
        write_once(root / "manifest.json", manifest, MAX_MANIFEST_BYTES)
    return {"prepared": True, "model_calls": 0, "campaign_root": str(root),
        "manifest_sha256": sha(root / "manifest.json"), "rows_planned": TOTAL_ROWS,
        "windows_planned": len(windows), "adopted_prefix_rows":
            sum(item["count"] for item in adopted_windows),
        "max_known_tokens": max_known_tokens, "max_turns": max_turns,
        "max_wall_seconds": max_wall_seconds, "official_model_score": False}


def load_manifest(root: Path, expected_sha256: str | None = None) -> dict:
    checked_root(root, exists=True)
    path = root / "manifest.json"
    require(regular(path, MAX_MANIFEST_BYTES), "campaign_manifest_missing")
    digest = sha(path)
    require(expected_sha256 is None or (HEX.fullmatch(expected_sha256) is not None
        and digest == expected_sha256), "campaign_manifest_pin_invalid")
    manifest = json.loads(path.read_text(encoding="ascii"))
    require(type(manifest) is dict and manifest.get("schema") == SCHEMA
        and manifest.get("campaign_root") == str(root)
        and manifest.get("dataset_sha256") == launcher.DATASET_SHA256
        and manifest.get("inventory_sha256") == launcher.INVENTORY_SHA256
        and manifest.get("runner_sha256") == launcher.RUNNER_SHA256
        and manifest.get("runtime_path") == str(RUNTIME)
        and manifest.get("model") == "gpt-5.6-luna"
        and manifest.get("auth") == "siwc_oauth"
        and manifest.get("api_fallback_allowed") is False
        and manifest.get("one_shot_windows") is True
        and manifest.get("variant") == VARIANT
        and manifest.get("official_model_score") is False,
        "campaign_manifest_invalid")
    windows = manifest.get("windows")
    require(type(windows) is list and 1 <= len(windows) <= TOTAL_ROWS,
        "campaign_windows_invalid")
    cursor = 0
    roots = set()
    new_window_seen = False
    for index, item in enumerate(windows):
        require(type(item) is dict and type(item.get("offset")) is int
            and item["offset"] == cursor and type(item.get("count")) is int
            and 1 <= item["count"] <= 4 and item.get("workers") == item["count"]
            and type(item.get("adopted")) is bool
            and item["adopted"] == ("adopted_receipt_sha256" in item)
            and type(item.get("repeat_window_authorized")) is bool
            and type(item.get("prior_overlap_roots")) is list
            and item["repeat_window_authorized"] == bool(item["prior_overlap_roots"])
            and type(item.get("root")) is str,
            "campaign_windows_invalid")
        if item["adopted"]:
            require(not new_window_seen
                and type(item["adopted_receipt_sha256"]) is str
                and HEX.fullmatch(item["adopted_receipt_sha256"]) is not None,
                "adopted_prefix_invalid")
        else:
            new_window_seen = True
        capsule = Path(item["root"])
        require(capsule.is_absolute() and capsule.parent == HOST_ROOT
            and launcher.ROOT_NAME.fullmatch(capsule.name) is not None
            and "_" not in capsule.name and str(capsule) not in roots,
            "campaign_capsule_path_invalid")
        roots.add(str(capsule))
        cursor += item["count"]
    require(cursor == TOTAL_ROWS and windows[0]["count"] == 1,
        "campaign_coverage_invalid")
    require(manifest.get("adopted_prefix_rows") == sum(item["count"]
        for item in windows if item["adopted"])
        and (manifest.get("resumed_from_manifest_sha256") is None
            or (type(manifest["resumed_from_manifest_sha256"]) is str
                and HEX.fullmatch(manifest["resumed_from_manifest_sha256"]) is not None)),
        "adopted_prefix_invalid")
    return manifest


def effective_budget(root: Path, manifest: dict) -> tuple[dict[str, int], str, int]:
    limits = {key: manifest[key] for key in (
        "max_known_tokens", "max_turns", "max_wall_seconds")}
    require(all(type(value) is int and value > 0 for value in limits.values()),
        "campaign_limits_invalid")
    previous = sha(root / "manifest.json")
    revision = 0
    while revision < MAX_BUDGET_REVISIONS:
        path = root / f"budget-{revision + 1:04d}.json"
        if not path.exists() and not path.is_symlink():
            break
        require(regular(path, 2048), "budget_revision_invalid")
        value = json.loads(path.read_text(encoding="ascii"))
        require(type(value) is dict and value.get("schema") == SCHEMA + "-budget-v1"
            and value.get("manifest_sha256") == sha(root / "manifest.json")
            and value.get("revision") == revision + 1
            and value.get("previous_sha256") == previous,
            "budget_revision_invalid")
        new_limits = {key: value.get(key) for key in limits}
        require(all(type(new_limits[key]) is int and new_limits[key] >= limits[key]
            for key in limits)
            and any(new_limits[key] > limits[key] for key in limits)
            and new_limits["max_known_tokens"] <= 10**10
            and new_limits["max_turns"] <= 10**7
            and new_limits["max_wall_seconds"] <= 365 * 86400,
            "budget_revision_invalid")
        limits = new_limits
        previous = sha(path)
        revision += 1
    require(not (root / f"budget-{MAX_BUDGET_REVISIONS + 1:04d}.json").exists(),
        "budget_revision_limit")
    return limits, previous, revision


def extend_budget(root: Path, manifest_sha256: str, previous_sha256: str,
                  *, max_known_tokens: int, max_turns: int,
                  max_wall_seconds: int) -> dict:
    manifest = load_manifest(root, manifest_sha256)
    lock_fd = _campaign_lock(root)
    try:
        limits, current_sha, revision = effective_budget(root, manifest)
        require(current_sha == previous_sha256 and revision < MAX_BUDGET_REVISIONS,
            "budget_revision_pin_invalid")
        new_limits = {"max_known_tokens": max_known_tokens,
            "max_turns": max_turns, "max_wall_seconds": max_wall_seconds}
        require(all(type(new_limits[key]) is int and new_limits[key] >= limits[key]
            for key in limits)
            and any(new_limits[key] > limits[key] for key in limits)
            and new_limits["max_known_tokens"] <= 10**10
            and new_limits["max_turns"] <= 10**7
            and new_limits["max_wall_seconds"] <= 365 * 86400,
            "budget_extension_invalid")
        amendment = {"schema": SCHEMA + "-budget-v1",
            "manifest_sha256": manifest_sha256, "revision": revision + 1,
            "previous_sha256": current_sha, **new_limits}
        path = root / f"budget-{revision + 1:04d}.json"
        write_once(path, amendment, 2048)
        return {"budget_extended": True, "model_calls": 0,
            "budget_revision": revision + 1, "budget_revision_sha256": sha(path),
            **new_limits}
    finally:
        os.close(lock_fd)


def source_admission(manifest: dict) -> None:
    runtime_admission()
    require(sha(LAUNCHER) == manifest["launcher_sha256"]
        and sha(Path(__file__)) == manifest["controller_sha256"]
        and sha(VERIFIER) == manifest["verifier_sha256"],
        "campaign_controller_source_drift")


def load_verifier(manifest: dict):
    require(regular(VERIFIER) and sha(VERIFIER) == manifest["verifier_sha256"],
        "campaign_verifier_drift")
    spec = importlib.util.spec_from_file_location("pinned_siwc_lme_campaign_verifier", VERIFIER)
    require(spec is not None and spec.loader is not None,
        "campaign_verifier_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(callable(getattr(module, "verify_capsule", None))
        and callable(getattr(module, "verify_campaign", None)),
        "campaign_verifier_interface_invalid")
    return module


def runner_receipt(window: dict) -> dict:
    capsule = launcher.checked_root(Path(window["root"]))
    path = capsule / "launch-receipt.json"
    require(regular(path, 8192), "window_receipt_missing")
    receipt = json.loads(path.read_text(encoding="ascii"))
    require(receipt.get("root") == str(capsule)
        and receipt.get("dataset_sha256") == launcher.DATASET_SHA256
        and receipt.get("candidate_map_sha256") is not None
        and receipt.get("inventory_sha256") == launcher.INVENTORY_SHA256
        and receipt.get("source_offset") == window["offset"]
        and receipt.get("selected_count") == window["count"]
        and receipt.get("workers") == window["workers"]
        and receipt.get("repeat_window_authorized") == window["repeat_window_authorized"]
        and receipt.get("dream_mode") == "no_dream"
        and receipt.get("mode") == "message_only_luna_diagnostic_v1"
        and receipt.get("model") == "gpt-5.6-luna"
        and receipt.get("auth") == "siwc_oauth"
        and receipt.get("endpoint") == "https://api.openai.com/v1/responses"
        and receipt.get("api_fallback_allowed") is False
        and receipt.get("one_shot") is True
        and receipt.get("billing_policy") ==
            "siwc_server_enforced_plan_or_existing_credits_v1",
        "window_receipt_policy_invalid")
    if window["adopted"]:
        require(sha(path) == window["adopted_receipt_sha256"],
            "adopted_receipt_drift")
    return receipt


def capsule_state(window: dict) -> str:
    capsule = Path(window["root"])
    if not capsule.exists():
        return "unprepared"
    if not regular(capsule / "launch-receipt.json", 8192):
        return "partial_capsule"
    runner_receipt(window)
    if not regular(capsule / "launch-attempt.json", 512):
        return "prepared"
    command_result = capsule / "launch-command-result.json"
    if regular(command_result, 8192):
        result = json.loads(command_result.read_text(encoding="ascii"))
        if result.get("returncode") != 0:
            return "launch_failed"
    result_path = capsule / "run" / "diagnostic-result.json"
    if regular(result_path, 128_000):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return "result_complete" if result.get("diagnostic_complete") is True else "result_failed"
    return "launched_no_result"


def invoke_launcher(*args: str) -> dict:
    done = subprocess.run([str(RUNTIME), "-I", "-B", str(LAUNCHER), *args],
        capture_output=True, text=True, timeout=300, check=False)
    require(len(done.stdout) <= 8192 and len(done.stderr) <= 8192,
        "launcher_output_invalid")
    try:
        result = json.loads(done.stdout)
    except (ValueError, TypeError):
        raise ValueError("launcher_output_invalid") from None
    require(done.returncode == 0 and type(result) is dict,
        "launcher_action_failed")
    return result


def verified_usage(result: dict) -> tuple[int, int]:
    budget = result.get("budget")
    require(type(budget) is dict and type(budget.get("known_tokens")) is int
        and type(budget.get("turns")) is int
        and budget["known_tokens"] >= 0 and budget["turns"] >= 0
        and budget.get("usage_complete") is True
        and budget.get("stopped") is False
        and result.get("diagnostic_complete") is True
        and result.get("campaign_stop") is None,
        "window_accounting_incomplete")
    return budget["known_tokens"], budget["turns"]


def window_elapsed(window: dict, *, current: bool = False) -> int:
    capsule = Path(window["root"])
    attempt = capsule / "launch-attempt.json"
    require(regular(attempt, 512), "window_attempt_missing")
    result = capsule / "run" / "diagnostic-result.json"
    end = time.time() if current else result.stat().st_mtime
    elapsed = end - attempt.stat().st_mtime
    require(0 <= elapsed <= 30 * 86400, "window_timing_invalid")
    return int(elapsed) + 1


def _verified_prefix(manifest: dict, verifier) -> tuple[list[dict], int, int, int]:
    completed = []
    tokens = turns = seconds = 0
    for window in manifest["windows"]:
        state = capsule_state(window)
        if state != "result_complete":
            break
        summary = verifier.verify_capsule(launcher.DATASET, Path(window["root"]))
        require(type(summary) is dict and summary.get("rows_verified") == window["count"],
            "window_offline_verification_failed")
        result_path = Path(window["root"]) / "run" / "diagnostic-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        window_tokens, window_turns = verified_usage(result)
        require(summary.get("known_tokens") == window_tokens
            and summary.get("turns") == window_turns,
            "window_accounting_verification_failed")
        tokens += window_tokens
        turns += window_turns
        seconds += window_elapsed(window)
        completed.append(window)
    require(all(capsule_state(window) not in {"result_complete", "result_failed"}
        for window in manifest["windows"][len(completed) + 1:]),
        "campaign_out_of_order_results")
    return completed, tokens, turns, seconds


def _unit_stopped(window: dict) -> bool:
    capsule = Path(window["root"])
    unit = launcher.unit_for(capsule)
    return launcher.prior_unit_stopped(unit, launcher.unit_state(unit))


def _await_window(window: dict, max_wall_seconds: int, prior_seconds: int) -> None:
    while True:
        state = capsule_state(window)
        if state in {"result_complete", "result_failed", "launch_failed"}:
            if _unit_stopped(window):
                require(state == "result_complete", "window_incomplete_no_auto_retry")
                return
            require(prior_seconds + window_elapsed(window, current=True) <= max_wall_seconds,
                "campaign_wall_ceiling_reached")
            time.sleep(POLL_SECONDS)
            continue
        require(state == "launched_no_result", "window_state_invalid")
        if _unit_stopped(window):
            raise ValueError("window_stopped_without_result_no_auto_retry")
        require(prior_seconds + window_elapsed(window, current=True) <= max_wall_seconds,
            "campaign_wall_ceiling_reached")
        time.sleep(POLL_SECONDS)


def _campaign_lock(root: Path):
    fd = os.open(root / "driver.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    metadata = os.fstat(fd)
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == launcher.HOST_UID
        and stat.S_IMODE(metadata.st_mode) == 0o600 and metadata.st_nlink == 1,
        "campaign_lock_invalid")
    try:
        require(fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB) is None,
            "campaign_driver_already_running")
    except BlockingIOError:
        os.close(fd)
        raise ValueError("campaign_driver_already_running") from None
    return fd


def run(root: Path, manifest_sha256: str) -> dict:
    manifest = load_manifest(root, manifest_sha256)
    source_admission(manifest)
    lock_fd = _campaign_lock(root)
    completed: list[dict] = []
    tokens = turns = seconds = 0
    current_offset = 0
    try:
        budget_limits, budget_sha256, budget_revision = effective_budget(root, manifest)
        verifier = load_verifier(manifest)
        completed, tokens, turns, seconds = _verified_prefix(manifest, verifier)
        current_offset = sum(item["count"] for item in completed)
        record_progress(root, manifest_sha256, state="running", reason=None,
            offset=current_offset, rows_verified=current_offset,
            known_tokens=tokens, turns=turns, wall_seconds=seconds)
        if completed:
            verifier.verify_campaign(launcher.DATASET,
                [Path(item["root"]) for item in completed], require_full=False)
        require(tokens <= budget_limits["max_known_tokens"]
            and turns <= budget_limits["max_turns"]
            and seconds <= budget_limits["max_wall_seconds"],
            "campaign_ceiling_already_exceeded")
        for window in manifest["windows"][len(completed):]:
            current_offset = window["offset"]
            capsule = Path(window["root"])
            state = capsule_state(window)
            require(state not in {"partial_capsule", "launch_failed", "result_failed"},
                "window_incomplete_no_auto_retry")
            if state == "unprepared":
                require(not window["adopted"], "adopted_capsule_missing")
                existing = sorted(item["root"] for item in launcher.launched_batches()
                    if window["offset"] < item["offset"] + item["count"]
                    and item["offset"] < window["offset"] + window["count"])
                require(existing == window["prior_overlap_roots"],
                    "window_overlap_changed_after_plan")
                args = ["--assemble-root", str(capsule), "--offset",
                    str(window["offset"]), "--questions", str(window["count"]),
                    "--workers", str(window["workers"])]
                if window["count"] > 1:
                    args.append("--multi-question-batch")
                if window["repeat_window_authorized"]:
                    args.append("--allow-repeat-window")
                invoke_launcher(*args)
                state = "prepared"
            receipt = runner_receipt(window)
            if state == "prepared":
                require(not window["adopted"], "adopted_capsule_unlaunched")
                verify = invoke_launcher("--preflight-root", str(capsule))
                require(verify.get("preflight_verified") is True
                    and verify.get("receipt_sha256") == sha(capsule / "launch-receipt.json"),
                    "window_preflight_invalid")
                window_limits = receipt.get("limits", {}).get("campaign")
                require(type(window_limits) is list and len(window_limits) == 3
                    and type(window_limits[0]) is int and type(window_limits[1]) is int,
                    "window_budget_invalid")
                require(tokens + window_limits[1] <= budget_limits["max_known_tokens"]
                    and turns + window_limits[0] <= budget_limits["max_turns"]
                    and type(window_limits[2]) is int
                    and seconds + window_limits[2] <= budget_limits["max_wall_seconds"],
                    "campaign_ceiling_paused_before_launch")
                host_wait_started = time.monotonic()
                while True:
                    try:
                        launcher.host_admission()
                        break
                    except ValueError as exc:
                        if str(exc) != "prior_benchmark_unit_running":
                            raise
                        require(time.monotonic() - host_wait_started < MAX_HOST_IDLE_WAIT_SECONDS,
                            "host_busy_retry_later")
                        time.sleep(POLL_SECONDS)
                launched = invoke_launcher("--launch-root", str(capsule),
                    "--receipt-sha256", sha(capsule / "launch-receipt.json"))
                require(launched.get("launch_command_returncode") == 0,
                    "window_launch_failed_no_auto_retry")
            _await_window(window, budget_limits["max_wall_seconds"], seconds)
            summary = verifier.verify_capsule(launcher.DATASET, capsule)
            require(summary.get("rows_verified") == window["count"],
                "window_offline_verification_failed")
            result = json.loads((capsule / "run" / "diagnostic-result.json").read_text(
                encoding="utf-8"))
            window_tokens, window_turns = verified_usage(result)
            require(summary.get("known_tokens") == window_tokens
                and summary.get("turns") == window_turns,
                "window_accounting_verification_failed")
            tokens += window_tokens
            turns += window_turns
            seconds += window_elapsed(window)
            require(tokens <= budget_limits["max_known_tokens"]
                and turns <= budget_limits["max_turns"]
                and seconds <= budget_limits["max_wall_seconds"],
                "campaign_ceiling_exceeded_after_window")
            completed.append(window)
            current_offset += window["count"]
            record_progress(root, manifest_sha256, state="running", reason=None,
                offset=current_offset, rows_verified=current_offset,
                known_tokens=tokens, turns=turns, wall_seconds=seconds)
        capsules = [Path(item["root"]) for item in completed]
        aggregate = verifier.verify_campaign(launcher.DATASET, capsules, require_full=True)
        require(aggregate.get("rows_verified") == TOTAL_ROWS
            and aggregate.get("known_tokens") == tokens
            and aggregate.get("turns") == turns,
            "campaign_aggregate_verification_failed")
        result = {"schema": SCHEMA, "campaign_manifest": str(root / "manifest.json"),
            "manifest_sha256": manifest_sha256, "capsules": [str(path) for path in capsules],
            "budget_revision": budget_revision,
            "budget_revision_sha256": budget_sha256,
            "rows_verified": TOTAL_ROWS, "correct_count": aggregate["correct_count"],
            "diagnostic_accuracy": aggregate["correct_count"] / TOTAL_ROWS,
            "known_tokens": tokens, "turns": turns, "observed_wall_seconds": seconds,
            "model": "gpt-5.6-luna", "auth": "siwc_oauth",
            "variant": manifest["variant"],
            "billing_policy": manifest["billing_policy"],
            "api_fallback_allowed": False, "diagnostic_complete": True,
            "canonical_r9_artifact": False, "official_model_score": False}
        output = root / "campaign-result.json"
        if output.exists():
            require(regular(output, MAX_RESULT_BYTES) and output.read_bytes() == canonical(result),
                "campaign_result_drift")
        else:
            write_once(output, result, MAX_RESULT_BYTES)
        record_progress(root, manifest_sha256, state="complete", reason=None,
            offset=TOTAL_ROWS, rows_verified=TOTAL_ROWS,
            known_tokens=tokens, turns=turns, wall_seconds=seconds)
        return {"complete": True, "campaign_result": str(output),
            "rows_verified": TOTAL_ROWS, "correct_count": aggregate["correct_count"],
            "known_tokens": tokens, "turns": turns, "official_model_score": False}
    except BaseException as exc:
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(
            r"[a-z0-9_]{1,80}", str(exc)) else "campaign_unverified"
        state = "paused" if code == "campaign_ceiling_paused_before_launch" else "stopped"
        try:
            record_progress(root, manifest_sha256, state=state, reason=code,
                offset=current_offset, rows_verified=sum(item["count"] for item in completed),
                known_tokens=tokens, turns=turns, wall_seconds=seconds)
        except (OSError, ValueError):
            pass
        raise
    finally:
        os.close(lock_fd)


def status(root: Path) -> dict:
    manifest = load_manifest(root)
    limits, budget_sha256, revision = effective_budget(root, manifest)
    states = []
    for window in manifest["windows"]:
        try:
            state = capsule_state(window)
        except (OSError, ValueError, TypeError, KeyError):
            state = "unverified"
        states.append({"offset": window["offset"], "count": window["count"],
            "root": window["root"], "state": state})
    raw_completed = 0
    for item in states:
        if item["state"] != "result_complete":
            break
        raw_completed += item["count"]
    result_path = root / "campaign-result.json"
    progress_path = root / "campaign-status.json"
    progress = (json.loads(progress_path.read_text(encoding="ascii"))
        if regular(progress_path, 2048) else None)
    if progress is not None:
        require(type(progress) is dict
            and progress.get("schema") == SCHEMA + "-progress-v1"
            and progress.get("manifest_sha256") == sha(root / "manifest.json"),
            "campaign_progress_invalid")
    return {"schema": SCHEMA, "campaign_root": str(root),
        "manifest_sha256": sha(root / "manifest.json"),
        "budget_revision": revision, "budget_revision_sha256": budget_sha256,
        "budget_limits": limits,
        "rows_with_complete_artifacts_unverified": raw_completed,
        "rows_total": TOTAL_ROWS,
        "next_offset": raw_completed if raw_completed < TOTAL_ROWS else None,
        "campaign_result_present": regular(result_path, MAX_RESULT_BYTES),
        "official_model_score": False, "progress": progress,
        "windows": states, "model_calls": 0}


def preflight_campaign(root: Path, manifest_sha256: str) -> dict:
    """Verify a pinned plan and adopted prefix without dispatching a model."""
    manifest = load_manifest(root, manifest_sha256)
    source_admission(manifest)
    verifier = load_verifier(manifest)
    completed, tokens, turns, seconds = _verified_prefix(manifest, verifier)
    require(sum(item["count"] for item in completed) ==
        manifest["adopted_prefix_rows"], "campaign_prefix_identity_invalid")
    for window in completed:
        require(window["adopted"], "campaign_unexpected_completed_window")
        check = launcher.preflight(Path(window["root"]))
        require(check.get("preflight_verified") is True
            and check.get("receipt_sha256") == window["adopted_receipt_sha256"],
            "adopted_capsule_preflight_invalid")
    if completed:
        aggregate = verifier.verify_campaign(launcher.DATASET,
            [Path(item["root"]) for item in completed], require_full=False)
        require(aggregate.get("rows_verified") == manifest["adopted_prefix_rows"]
            and aggregate.get("known_tokens") == tokens
            and aggregate.get("turns") == turns,
            "campaign_prefix_verification_invalid")
    budget, budget_sha, revision = effective_budget(root, manifest)
    require(tokens <= budget["max_known_tokens"]
        and turns <= budget["max_turns"]
        and seconds <= budget["max_wall_seconds"],
        "campaign_ceiling_already_exceeded")
    return {"preflight_verified": True, "model_calls": 0,
        "campaign_root": str(root), "manifest_sha256": manifest_sha256,
        "budget_revision": revision, "budget_revision_sha256": budget_sha,
        "rows_verified": manifest["adopted_prefix_rows"],
        "windows_planned": len(manifest["windows"]),
        "next_offset": manifest["adopted_prefix_rows"],
        "known_tokens": tokens, "turns": turns,
        "observed_wall_seconds": seconds,
        "variant": VARIANT, "official_model_score": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-campaign")
    action.add_argument("--run-campaign")
    action.add_argument("--status-campaign")
    action.add_argument("--preflight-campaign")
    action.add_argument("--extend-campaign")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--previous-budget-sha256")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--adopt-root")
    parser.add_argument("--resume-from-campaign")
    parser.add_argument("--allow-prior-overlap", action="store_true")
    parser.add_argument("--max-known-tokens", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-wall-seconds", type=int)
    args = parser.parse_args(argv)
    root = Path(args.prepare_campaign or args.run_campaign or args.status_campaign
        or args.preflight_campaign or args.extend_campaign)
    try:
        if args.prepare_campaign:
            require(args.manifest_sha256 is None and args.previous_budget_sha256 is None
                and args.batch_size is not None
                and args.max_known_tokens is not None and args.max_turns is not None
                and args.max_wall_seconds is not None,
                "campaign_prepare_arguments_invalid")
            result = prepare(root, batch_size=args.batch_size,
                adopted_root=Path(args.adopt_root) if args.adopt_root else None,
                resume_from=Path(args.resume_from_campaign)
                    if args.resume_from_campaign else None,
                allow_prior_overlap=args.allow_prior_overlap,
                max_known_tokens=args.max_known_tokens, max_turns=args.max_turns,
                max_wall_seconds=args.max_wall_seconds)
        elif args.extend_campaign:
            require(args.manifest_sha256 is not None
                and args.previous_budget_sha256 is not None
                and args.batch_size is None and args.adopt_root is None
                and args.resume_from_campaign is None
                and not args.allow_prior_overlap
                and args.max_known_tokens is not None and args.max_turns is not None
                and args.max_wall_seconds is not None,
                "campaign_extension_arguments_invalid")
            result = extend_budget(root, args.manifest_sha256,
                args.previous_budget_sha256,
                max_known_tokens=args.max_known_tokens, max_turns=args.max_turns,
                max_wall_seconds=args.max_wall_seconds)
        else:
            require(args.batch_size is None and args.adopt_root is None
                and args.resume_from_campaign is None
                and not args.allow_prior_overlap and args.max_known_tokens is None
                and args.max_turns is None and args.max_wall_seconds is None
                and args.previous_budget_sha256 is None,
                "campaign_unexpected_selection")
            if args.run_campaign:
                require(args.manifest_sha256 is not None, "campaign_manifest_pin_required")
                result = run(root, args.manifest_sha256)
            elif args.preflight_campaign:
                require(args.manifest_sha256 is not None, "campaign_manifest_pin_required")
                result = preflight_campaign(root, args.manifest_sha256)
            else:
                require(args.manifest_sha256 is None, "campaign_status_read_only")
                result = status(root)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(
            r"[a-z0-9_]{1,80}", str(exc)) else "campaign_unverified"
        print(json.dumps({"ok": False, "reason": code,
            "campaign_root": str(root), "automatic_retry": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
