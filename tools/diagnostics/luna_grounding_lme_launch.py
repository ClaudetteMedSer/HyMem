"""One-shot launcher for the exact one-file grounding candidate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


BASE_LAUNCH_SHA256 = "b848ad37f66680c8e423876b0aed2f4f99bdd6ee896247f1faa9461a50e43983"
BUILDER_SHA256 = "51e03ff2bec29f7517db027b103163698c18b9d34603ab950a36f5d09255e96a"
RUNNER_SHA256 = "5956ebbed67b9e0a7bcbf812d7d819c1878f7fe91f8afd369447b539b2e81745"
ORIGINAL_INVENTORY_SHA256 = "852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb"
ORIGINAL_CANDIDATE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate")
DATASET = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json")
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
BINARY = Path("/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex")
HOST_ROOT = Path("/home/atta")
HOST_UID = 1000
ROOT_PREFIX = ".hymem-luna-lme-grounding-"
UNIT_PREFIX = "hymem-luna-lme-grounding-"
STAGED_PREFIX = ".hymem-luna-grounding-bundle-"
ROOT_NAME = re.compile(r"\.hymem-luna-lme-grounding-([A-Za-z0-9_-]{8,})\Z")
STAGED_NAME = re.compile(r"\.hymem-luna-grounding-bundle-([A-Za-z0-9_-]{8,})\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
RUNNER_NAME = "luna_grounding_lme.py"
BUILDER_NAME = "luna_grounding_candidate.py"
DERIVED_STAMP = "headless-grounding-source-map.json"
ORIGINAL_STAMP = "headless-source-map.json"
PINS = {
    RUNNER_NAME: RUNNER_SHA256,
    BUILDER_NAME: BUILDER_SHA256,
    "luna_subscription_lme_profiled_v2.py": "53628ac7e9c6107bb68d1cd4ebdf42d5a129b96c4d96360d4e7a6525be7bc739",
    "luna_subscription_lme_warm_v2.py": "3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567",
    "luna_stage_accounting.py": "800ef9baedc9d68093b3160cc324ef17528dd0b89f7a734f220217b07323fba2",
    "codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    "codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
    "luna_subscription_pilot.py": "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0",
    "codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
    "luna_subscription_capacity_progress.py": "2eed17065cf144a4f59d0af47ac2b6d937253f0b1c100b1ee134430a89789831",
    ORIGINAL_STAMP: ORIGINAL_INVENTORY_SHA256,
}
LIMITS = {
    "campaign-turns": 8012, "campaign-known-tokens": 48160000,
    "campaign-seconds": 14400, "question-turns": 2000,
    "question-known-tokens": 12000000, "question-seconds": 12600,
    "canary-turns": 12, "canary-known-tokens": 160000,
    "canary-seconds": 600, "indexing-seconds": 10800,
    "questions": 4, "workers": 4,
    "warm-max-requests": 16, "warm-max-age-seconds": 300,
}


def check(condition, code):
    if not condition:
        raise ValueError(code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pinned_module(path: Path, expected: str, name: str):
    check(path.is_file() and not path.is_symlink() and sha(path) == expected,
          "source_pin_invalid")
    spec = importlib.util.spec_from_file_location(name, path)
    check(spec is not None and spec.loader is not None, "source_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def base_launcher(root: Path):
    return pinned_module(root / "luna_subscription_capacity_launch.py",
                         BASE_LAUNCH_SHA256, "pinned_grounding_base_launcher")


def builder(root: Path):
    return pinned_module(root / BUILDER_NAME, BUILDER_SHA256,
                         "pinned_grounding_candidate_builder")


def unit_for(root: Path) -> str:
    match = ROOT_NAME.fullmatch(root.name)
    check(root.parent == HOST_ROOT and match is not None, "root_invalid")
    return UNIT_PREFIX + match.group(1) + ".service"


def source_pins(root: Path, *, include_derived: bool) -> dict:
    expected = dict(PINS)
    expected["luna_subscription_capacity_launch.py"] = BASE_LAUNCH_SHA256
    expected["luna_grounding_lme_launch.py"] = sha(Path(__file__))
    if include_derived:
        expected[DERIVED_STAMP] = sha(root / DERIVED_STAMP)
    for name, digest in expected.items():
        path = root / name
        check(path.is_file() and not path.is_symlink() and sha(path) == digest,
              "source_pin_invalid")
    return expected


def verify_sources(root: Path) -> dict:
    pins = source_pins(root, include_derived=True)
    proof = builder(root).verify_derived(ORIGINAL_CANDIDATE, root / "candidate",
                                         root / DERIVED_STAMP, root / ORIGINAL_STAMP)
    check(proof["source_files"] == 508 and
          proof["grounded_inventory_sha256"] == pins[DERIVED_STAMP],
          "grounding_candidate_invalid")
    profile = pinned_module(root / "luna_subscription_lme_profiled_v2.py",
        PINS["luna_subscription_lme_profiled_v2.py"], "pinned_grounding_profile_for_launch")
    runner = pinned_module(root / RUNNER_NAME, RUNNER_SHA256,
                           "pinned_grounding_runner_for_launch")
    check(runner.ORIGINAL_CANDIDATE == ORIGINAL_CANDIDATE and
          runner.BUILDER_SHA256 == BUILDER_SHA256, "runner_binding_invalid")
    runner.bind_profile(profile, builder(root), root / "candidate", root / DERIVED_STAMP,
                        root / ORIGINAL_STAMP, pins[DERIVED_STAMP])
    files, *_ = profile.warm_runner.load_verified(
        candidate=root / "candidate", inventory_stamp=root / DERIVED_STAMP,
        inventory_sha256=pins[DERIVED_STAMP], dataset=DATASET,
        dataset_sha256=DATASET_SHA256, binary=BINARY,
        base_path=root / "codex_subscription.py",
        concurrent_path=root / "codex_subscription_concurrent_v2.py",
        warm_path=root / "codex_subscription_warm_v2.py")
    check(files == 508, "inventory_incomplete")
    profile.collector.verify_candidate(root / "candidate")
    return proof


def receipt_for(root: Path, proof: dict) -> dict:
    unit = unit_for(root)
    pins = source_pins(root, include_derived=True)
    return {"schema": "luna-grounding-launch-v1", "root": str(root), "unit": unit,
        "expected_cgroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit,
        "output": "run", "candidate": str(root / "candidate"),
        "original_candidate": str(ORIGINAL_CANDIDATE),
        "dataset": str(DATASET), "dataset_sha256": DATASET_SHA256,
        "inventory_stamp": str(root / DERIVED_STAMP),
        "inventory_sha256": proof["grounded_inventory_sha256"],
        "candidate_source_map_sha256": proof["grounded_map_sha256"],
        "runner_sha256": RUNNER_SHA256, "source_sha256": pins,
        "launcher_sha256": sha(Path(__file__)), "binary": str(BINARY),
        "binary_sha256": sha(BINARY), "runtime_max_seconds": 14530,
        "timeout_stop_seconds": 10, "memory_max_bytes": 4294967296,
        "cpu_quota_percent": 200, "tasks_max": 256, "oom_policy": "kill",
        "kill_mode": "control-group", "restart": "no", "remain_after_exit": True,
        "model": "gpt-6-luna", "subscription_only": True,
        "reported_quota_floor_percent": 25, "limits": LIMITS}


def command(root: Path, receipt: dict) -> list[str]:
    args = ["/usr/bin/systemd-run", "--user", "--quiet", "--unit", receipt["unit"],
        "--property=Type=exec", "--property=Restart=no", "--property=KillMode=control-group",
        "--property=RemainAfterExit=yes", "--property=RuntimeMaxSec=14530s",
        "--property=TimeoutStopSec=10s", "--property=MemoryMax=4294967296",
        "--property=CPUQuota=200%", "--property=TasksMax=256",
        "--property=OOMPolicy=kill", "--property=UMask=0077",
        "--property=WorkingDirectory=" + str(root / "empty"),
        "--property=StandardOutput=file:" + str(root / "safe-terminal.json"),
        "--property=StandardError=file:" + str(root / "private-launch-stderr.log"),
        "/usr/bin/env", "-i", "HOME=/home/atta", "PATH=/usr/local/bin:/usr/bin:/bin",
        "TMPDIR=" + str(root / "tmp"), "/usr/bin/python3", "-I", "-B",
        str(root / RUNNER_NAME)]
    options = {"binary": str(BINARY), "base-transport": str(root / "codex_subscription.py"),
        "concurrent-transport": str(root / "codex_subscription_concurrent_v2.py"),
        "warm-transport": str(root / "codex_subscription_warm_v2.py"),
        "candidate": receipt["candidate"], "inventory-stamp": receipt["inventory_stamp"],
        "inventory-sha256": receipt["inventory_sha256"], "dataset": str(DATASET),
        "dataset-sha256": DATASET_SHA256, "output-dir": str(root / "run"), **LIMITS}
    for key, value in options.items():
        args.extend(["--" + key, str(value)])
    return args


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--launch-root")
    parser.add_argument("--staged-root")
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    root = None
    try:
        check((args.prepare and args.staged_root is not None and args.receipt_sha256 is None)
              or (not args.prepare and args.staged_root is None and
                  isinstance(args.receipt_sha256, str) and
                  HEX.fullmatch(args.receipt_sha256) is not None), "arguments_invalid")
        if args.prepare:
            staged = Path(args.staged_root)
            check(staged.is_absolute() and staged.parent == HOST_ROOT and
                  STAGED_NAME.fullmatch(staged.name) is not None and staged.is_dir() and
                  not staged.is_symlink() and staged.stat().st_uid == HOST_UID and
                  not staged.stat().st_mode & 0o077, "staged_root_invalid")
            root = Path(tempfile.mkdtemp(prefix=ROOT_PREFIX, dir=HOST_ROOT))
            root.chmod(0o700)
            for name, expected in {**PINS,
                    "luna_subscription_capacity_launch.py": BASE_LAUNCH_SHA256,
                    "luna_grounding_lme_launch.py": sha(Path(__file__))}.items():
                source = staged / name
                check(source.is_file() and not source.is_symlink() and sha(source) == expected,
                      "staged_pin_invalid")
                shutil.copyfile(source, root / name)
                (root / name).chmod(0o600)
            for name in ("empty", "tmp"):
                (root / name).mkdir(mode=0o700)
            proof = builder(root).prepare(ORIGINAL_CANDIDATE, root / ORIGINAL_STAMP,
                                          root / "candidate", root / DERIVED_STAMP)
            check(proof["source_files"] == 508, "candidate_incomplete")
            base_launcher(root).host_admission()
            verify_sources(root)
            receipt = receipt_for(root, proof)
            base_launcher(root).write_once(root / "launch-receipt.json", receipt)
            print(json.dumps({"prepared": True, "launched": False, "root": str(root),
                "unit": receipt["unit"], "expected_cgroup": receipt["expected_cgroup"],
                "receipt_sha256": sha(root / "launch-receipt.json"), "model_calls": 0}))
            return 0
        root = Path(args.launch_root)
        check(root.parent == HOST_ROOT and ROOT_NAME.fullmatch(root.name) is not None and
              root.is_dir() and not root.is_symlink() and root.stat().st_uid == HOST_UID and
              not root.stat().st_mode & 0o077, "root_invalid")
        base = base_launcher(root)
        base.host_admission()
        receipt_path = root / "launch-receipt.json"
        check(receipt_path.is_file() and not receipt_path.is_symlink() and
              sha(receipt_path) == args.receipt_sha256, "receipt_pin_invalid")
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        proof = verify_sources(root)
        check(receipt == receipt_for(root, proof), "receipt_invalid")
        check(not (root / "run").exists(), "output_already_exists")
        base.write_once(root / "launch-attempt.json",
                        {"receipt_sha256": args.receipt_sha256, "one_shot": True})
        started = subprocess.run(command(root, receipt), capture_output=True, timeout=20)
        base.write_once(root / "launch-command-result.json", {"returncode": started.returncode})
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
