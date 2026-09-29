"""One-shot private launcher for the source-pinned observed grounding pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile

import importlib.util
import sys

GROUND_LAUNCH_SHA256 = "f1683059fa92f47f7b9a6bcecc1a420489f3c0135212fc474ac83a1105c68475"
OBSERVED_V3_SHA256 = "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d"
GROUNDED_MAP_SHA256 = "217036b8089911c352cdf5994ef2b37915b5c68d2622a23c0642d264487dbe62"
RUNNER_NAME = "luna_observed_lme.py"
LAUNCHER_NAME = "luna_observed_lme_launch.py"
RUNNER_SHA256 = "2062a8816e8e22214a1e7febc4898322210c381abdf9bb0f34e802d61d111550"
HOST_ROOT = Path("/home/atta")
HOST_UID = 1000
ROOT_PREFIX = ".hymem-luna-lme-observed-"
UNIT_PREFIX = "hymem-luna-lme-observed-"
STAGED_NAME = re.compile(r"\.hymem-luna-observed-bundle-([A-Za-z0-9_-]{8,})\Z")
ROOT_NAME = re.compile(r"\.hymem-luna-lme-observed-([A-Za-z0-9_-]{8,})\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(ok, code):
    if not ok:
        raise ValueError(code)


def pinned(path, digest, identity):
    check(path.is_file() and not path.is_symlink() and sha(path) == digest,
          "source_pin_invalid")
    spec = importlib.util.spec_from_file_location(identity, path)
    check(spec is not None and spec.loader is not None, "source_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[identity] = module
    spec.loader.exec_module(module)
    return module


def grounding(root):
    return pinned(root / "luna_grounding_lme_launch.py", GROUND_LAUNCH_SHA256,
                  "pinned_observed_grounding_launch")


def expected_pins(base, root):
    pins = dict(base.PINS)
    pins["luna_subscription_capacity_launch.py"] = base.BASE_LAUNCH_SHA256
    pins["luna_grounding_lme_launch.py"] = GROUND_LAUNCH_SHA256
    pins["luna_grounding_lme_progress.py"] = "bce6e26197d9831a90ba1015a7b129254c5368fa2adedff29050381ac31a1710"
    pins[RUNNER_NAME] = RUNNER_SHA256
    pins[LAUNCHER_NAME] = sha(Path(__file__))
    pins["codex_subscription_warm_v3.py"] = OBSERVED_V3_SHA256
    return pins


def unit_for(root):
    match = ROOT_NAME.fullmatch(root.name)
    check(root.parent == HOST_ROOT and match is not None, "root_invalid")
    return UNIT_PREFIX + match.group(1) + ".service"


def verify_sources(root):
    base = grounding(root)
    pins = expected_pins(base, root)
    pins[base.DERIVED_STAMP] = sha(root / base.DERIVED_STAMP)
    for name, digest in pins.items():
        check((root / name).is_file() and not (root / name).is_symlink()
              and sha(root / name) == digest, "source_pin_invalid")
    builder = base.builder(root)
    proof = builder.verify_derived(base.ORIGINAL_CANDIDATE, root / "candidate",
                                   root / base.DERIVED_STAMP, root / base.ORIGINAL_STAMP)
    check(proof["source_files"] == 508 and
          proof["grounded_inventory_sha256"] == pins[base.DERIVED_STAMP] and
          proof["grounded_map_sha256"] == GROUNDED_MAP_SHA256,
          "candidate_invalid")
    runner = pinned(root / RUNNER_NAME, pins[RUNNER_NAME], "pinned_observed_runner_for_launch")
    grounded = pinned(root / "luna_grounding_lme.py", base.RUNNER_SHA256,
                      "pinned_grounded_runner_for_observed_launch")
    profile = pinned(root / "luna_subscription_lme_profiled_v2.py",
                     base.PINS["luna_subscription_lme_profiled_v2.py"],
                     "pinned_profile_for_observed_launch")
    grounded.bind_profile(profile, builder, root / "candidate", root / base.DERIVED_STAMP,
                          root / base.ORIGINAL_STAMP, pins[base.DERIVED_STAMP])
    runner.bind(grounded, profile)
    loaded = profile.warm_runner.load_verified(candidate=root / "candidate",
        inventory_stamp=root / base.DERIVED_STAMP,
        inventory_sha256=pins[base.DERIVED_STAMP], dataset=base.DATASET,
        dataset_sha256=base.DATASET_SHA256, binary=base.BINARY,
        base_path=root / "codex_subscription.py",
        concurrent_path=root / "codex_subscription_concurrent_v2.py",
        warm_path=root / "codex_subscription_warm_v2.py")
    check(loaded[0] == 508 and loaded[-1].WarmSubscriptionClient.__module__ ==
          "pinned_observed_warm_v3" and loaded[1] is loaded[-1].concurrent,
          "effective_transport_invalid")
    profile.collector.verify_candidate(root / "candidate")
    return base, proof, pins


def receipt_for(root, base, proof, pins):
    unit = unit_for(root)
    return {"schema": "luna-observed-launch-v1", "root": str(root), "unit": unit,
        "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "output": "run", "candidate": str(root / "candidate"),
        "original_candidate": str(base.ORIGINAL_CANDIDATE),
        "dataset": str(base.DATASET), "dataset_sha256": base.DATASET_SHA256,
        "inventory_stamp": str(root / base.DERIVED_STAMP),
        "inventory_sha256": proof["grounded_inventory_sha256"],
        "candidate_source_map_sha256": proof["grounded_map_sha256"],
        "runner_sha256": RUNNER_SHA256, "source_sha256": pins,
        "launcher_sha256": pins[LAUNCHER_NAME], "binary": str(base.BINARY),
        "binary_sha256": sha(base.BINARY), "runtime_max_seconds": 14530,
        "timeout_stop_seconds": 10, "memory_max_bytes": 4294967296,
        "cpu_quota_percent": 200, "tasks_max": 256, "oom_policy": "kill",
        "kill_mode": "control-group", "restart": "no", "remain_after_exit": True,
        "model": "gpt-6-luna", "subscription_only": True,
        "reported_quota_floor_percent": 25, "limits": base.LIMITS,
        "inherited_warm_transport_sha256": base.PINS["codex_subscription_warm_v2.py"],
        "effective_warm_transport_sha256": OBSERVED_V3_SHA256}


def command(root, base, receipt):
    base.RUNNER_NAME = RUNNER_NAME
    args = base.command(root, receipt)
    check(args[args.index(str(root / RUNNER_NAME))] == str(root / RUNNER_NAME),
          "command_runner_invalid")
    return args


def private_workspaces(root):
    for name in ("empty", "tmp"):
        path = root / name
        check(path.parent == root and not path.is_symlink(), "workspace_invalid")
        state = path.lstat()
        check(stat.S_ISDIR(state.st_mode) and state.st_uid == HOST_UID and
              not state.st_mode & 0o077 and not any(path.iterdir()), "workspace_invalid")


def main(argv=None):
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--launch-root")
    parser.add_argument("--staged-root")
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    root = None
    try:
        check((args.prepare and args.staged_root is not None and args.receipt_sha256 is None)
              or (not args.prepare and args.staged_root is None and
                  type(args.receipt_sha256) is str and HEX.fullmatch(args.receipt_sha256)),
              "arguments_invalid")
        if args.prepare:
            staged = Path(args.staged_root)
            check(staged.is_absolute() and staged.parent == HOST_ROOT and
                  STAGED_NAME.fullmatch(staged.name) and staged.is_dir() and
                  not staged.is_symlink() and staged.stat().st_uid == HOST_UID and
                  not staged.stat().st_mode & 0o077, "staged_root_invalid")
            root = Path(tempfile.mkdtemp(prefix=ROOT_PREFIX, dir=HOST_ROOT))
            root.chmod(0o700)
            stage_base = grounding(staged)
            pins = expected_pins(stage_base, staged)
            for name, digest in pins.items():
                src = staged / name
                check(src.is_file() and not src.is_symlink() and sha(src) == digest,
                      "staged_pin_invalid")
                shutil.copyfile(src, root / name)
                (root / name).chmod(0o600)
            for name in ("empty", "tmp"):
                (root / name).mkdir(mode=0o700)
            proof = stage_base.builder(root).prepare(stage_base.ORIGINAL_CANDIDATE,
                root / stage_base.ORIGINAL_STAMP, root / "candidate",
                root / stage_base.DERIVED_STAMP)
            check(proof["source_files"] == 508, "candidate_incomplete")
            host = stage_base.base_launcher(root)
            host.host_admission()
            base, proof, pins = verify_sources(root)
            receipt = receipt_for(root, base, proof, pins)
            host.write_once(root / "launch-receipt.json", receipt)
            print(json.dumps({"prepared": True, "launched": False, "root": str(root),
                "unit": receipt["unit"], "expected_cgroup": receipt["expected_cgroup"],
                "receipt_sha256": sha(root / "launch-receipt.json"), "model_calls": 0}))
            return 0
        root = Path(args.launch_root)
        check(root.parent == HOST_ROOT and ROOT_NAME.fullmatch(root.name) and
              root.is_dir() and not root.is_symlink() and root.stat().st_uid == HOST_UID and
              not root.stat().st_mode & 0o077, "root_invalid")
        base = grounding(root)
        host = base.base_launcher(root)
        host.host_admission()
        receipt_path = root / "launch-receipt.json"
        check(receipt_path.is_file() and not receipt_path.is_symlink() and
              sha(receipt_path) == args.receipt_sha256, "receipt_pin_invalid")
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        base, proof, pins = verify_sources(root)
        check(receipt == receipt_for(root, base, proof, pins), "receipt_invalid")
        check(not (root / "run").exists(), "output_already_exists")
        private_workspaces(root)
        host.write_once(root / "launch-attempt.json",
                        {"receipt_sha256": args.receipt_sha256, "one_shot": True})
        started = subprocess.run(command(root, base, receipt), capture_output=True, timeout=20)
        host.write_once(root / "launch-command-result.json", {"returncode": started.returncode})
        print(json.dumps({"launch_command_returncode": started.returncode,
            "root": str(root), "unit": receipt["unit"],
            "receipt_sha256": args.receipt_sha256, "never_retry": True}))
        return 0 if started.returncode == 0 else 1
    except Exception:
        print(json.dumps({"ok": False, "stage": "prepare" if args.prepare else "launch",
            "root": str(root) if root else None, "reason": "admission_or_launch_failed",
            "never_retry_launch": not args.prepare}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
