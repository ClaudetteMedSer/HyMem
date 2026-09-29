"""Read-only, source-free observer for the one-file grounding LME run."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys


BASE_READER_SHA256 = "2eed17065cf144a4f59d0af47ac2b6d937253f0b1c100b1ee134430a89789831"
BUILDER_SHA256 = "51e03ff2bec29f7517db027b103163698c18b9d34603ab950a36f5d09255e96a"
RUNNER_SHA256 = "5956ebbed67b9e0a7bcbf812d7d819c1878f7fe91f8afd369447b539b2e81745"
LAUNCHER_SHA256 = "f1683059fa92f47f7b9a6bcecc1a420489f3c0135212fc474ac83a1105c68475"
GROUNDED_MAP_SHA256 = "217036b8089911c352cdf5994ef2b37915b5c68d2622a23c0642d264487dbe62"
SCHEMA = "luna-grounding-lme-v1"
ORIGINAL_CANDIDATE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate")
ROOT_NAME = re.compile(r"\.hymem-luna-lme-grounding-([A-Za-z0-9_-]{8,})\Z")
UNIT = re.compile(r"hymem-luna-lme-grounding-[A-Za-z0-9_-]{8,}\.service\Z")
ROOT_PARENT = Path("/home/atta")
OBSERVER_UID = 1000
ORIGINAL_STAMP = "headless-source-map.json"
DERIVED_STAMP = "headless-grounding-source-map.json"
PINS = {
    "luna_grounding_lme.py": RUNNER_SHA256,
    "luna_grounding_candidate.py": BUILDER_SHA256,
    "luna_grounding_lme_launch.py": LAUNCHER_SHA256,
    "luna_subscription_capacity_launch.py": "b848ad37f66680c8e423876b0aed2f4f99bdd6ee896247f1faa9461a50e43983",
    "luna_subscription_capacity_progress.py": BASE_READER_SHA256,
    "luna_subscription_lme_profiled_v2.py": "53628ac7e9c6107bb68d1cd4ebdf42d5a129b96c4d96360d4e7a6525be7bc739",
    "luna_subscription_lme_warm_v2.py": "3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567",
    "luna_stage_accounting.py": "800ef9baedc9d68093b3160cc324ef17528dd0b89f7a734f220217b07323fba2",
    "codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    "codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
    "luna_subscription_pilot.py": "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0",
    "codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
    ORIGINAL_STAMP: "852758cfa63902048669f78f2377db4354b2e196a16ccbb98a3e6cadca2590eb",
}


def _pinned_module(path: Path, expected: str, name: str):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("observer_dependency_path_invalid")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise RuntimeError("observer_dependency_source_drift")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("observer_dependency_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _base_reader_path() -> Path:
    source = Path(__file__)
    if source.name == "luna_grounding_lme_progress.py" and source.is_file():
        return source.resolve().with_name("luna_subscription_capacity_progress.py")
    args = sys.argv[1:]
    if args.count("--root") != 1:
        raise RuntimeError("observer_root_argument_invalid")
    index = args.index("--root")
    if index + 1 >= len(args):
        raise RuntimeError("observer_root_argument_invalid")
    root = Path(args[index + 1])
    if (not root.is_absolute() or root.parent != ROOT_PARENT or
            ROOT_NAME.fullmatch(root.name) is None):
        raise RuntimeError("observer_root_argument_invalid")
    state = root.lstat()
    if (not stat.S_ISDIR(state.st_mode) or state.st_uid != OBSERVER_UID or
            state.st_mode & 0o077):
        raise RuntimeError("observer_root_path_invalid")
    return root / "luna_subscription_capacity_progress.py"


base = _pinned_module(_base_reader_path(), BASE_READER_SHA256,
    "pinned_grounding_capacity_reader")
base.PROFILED = RUNNER_SHA256
base.SOURCE_MAP = GROUNDED_MAP_SHA256
base.UNIT = UNIT
accepted_verify_terminal = base.verify_terminal


def root_identity_valid(root, unit, cgroup):
    match = ROOT_NAME.fullmatch(root.name)
    return bool(root.is_absolute() and root.parent == ROOT_PARENT and match is not None
        and not root.is_symlink() and root.is_dir()
        and unit == "hymem-luna-lme-grounding-" + match.group(1) + ".service"
        and UNIT.fullmatch(unit)
        and cgroup == "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit)


def receipt_valid(receipt, root, unit, cgroup):
    if type(receipt) is not dict:
        return False
    source = receipt.get("source_sha256")
    if type(source) is not dict or set(source) != set(PINS) | {DERIVED_STAMP}:
        return False
    if any(source.get(name) != expected for name, expected in PINS.items()):
        return False
    derived_sha = source.get(DERIVED_STAMP)
    if type(derived_sha) is not str or base.HEX.fullmatch(derived_sha) is None:
        return False
    return bool(receipt.get("schema") == "luna-grounding-launch-v1"
        and receipt.get("root") == str(root) and receipt.get("unit") == unit
        and receipt.get("expected_cgroup") == cgroup and receipt.get("output") == "run"
        and receipt.get("candidate") == str(root / "candidate")
        and receipt.get("original_candidate") == str(ORIGINAL_CANDIDATE)
        and receipt.get("inventory_stamp") == str(root / DERIVED_STAMP)
        and receipt.get("inventory_sha256") == derived_sha
        and receipt.get("candidate_source_map_sha256") == GROUNDED_MAP_SHA256
        and receipt.get("runner_sha256") == RUNNER_SHA256
        and receipt.get("launcher_sha256") == LAUNCHER_SHA256
        and receipt.get("dataset_sha256") == base.DATASET
        and receipt.get("binary") == str(base.BINARY)
        and base.HEX.fullmatch(str(receipt.get("binary_sha256", "")))
        and receipt.get("runtime_max_seconds") == 14530
        and receipt.get("timeout_stop_seconds") == 10
        and receipt.get("memory_max_bytes") == 4294967296
        and receipt.get("cpu_quota_percent") == 200 and receipt.get("tasks_max") == 256
        and receipt.get("oom_policy") == "kill" and receipt.get("kill_mode") == "control-group"
        and receipt.get("restart") == "no" and receipt.get("remain_after_exit") is True
        and receipt.get("model") == "gpt-6-luna" and receipt.get("subscription_only") is True
        and receipt.get("reported_quota_floor_percent") == 25
        and receipt.get("limits") == base.LIMITS
        and Path(str(receipt.get("dataset", ""))).is_absolute())


def verify_live_pins(root, receipt):
    source = receipt["source_sha256"]
    if any(not base.regular(root / name) or base.digest(root / name) != expected
           for name, expected in source.items()):
        return {"source_pins_verified": False, "dataset_pin_verified": False,
                "inventory_verified": False}
    dataset = Path(receipt["dataset"])
    if not base.regular(dataset) or base.digest(dataset) != base.DATASET:
        return {"source_pins_verified": True, "dataset_pin_verified": False,
                "inventory_verified": False}
    try:
        builder = _pinned_module(root / "luna_grounding_candidate.py", BUILDER_SHA256,
                                 "pinned_grounding_reader_builder")
        proof = builder.verify_derived(ORIGINAL_CANDIDATE, root / "candidate",
                                       root / DERIVED_STAMP, root / ORIGINAL_STAMP)
        verified = (proof["source_files"] == 508 and
                    proof["grounded_map_sha256"] == GROUNDED_MAP_SHA256 and
                    proof["grounded_inventory_sha256"] == receipt["inventory_sha256"])
    except Exception:
        verified = False
    return {"source_pins_verified": True, "dataset_pin_verified": True,
            "inventory_verified": verified}


def verify_terminal(root, receipt, safe, result):
    # Reuse every accepted terminal gate while requiring the new runner label.
    if (type(safe) is not dict or safe.get("schema") != SCHEMA or
            type(result) is not dict or result.get("schema") != SCHEMA):
        return {"available": safe is not None, "validated": False,
                "questions": [{"index": index, "validated": False, "correct": None}
                              for index in range(4)],
                "stage_accounting_reconciled": False, "warm_metrics_valid": False}
    safe_for_base = dict(safe, schema="luna-subscription-lme-profiled-v2")
    result_for_base = dict(result, schema="luna-subscription-lme-profiled-v2")
    return accepted_verify_terminal(root, receipt, safe_for_base, result_for_base)


base.root_identity_valid = root_identity_valid
base.receipt_valid = receipt_valid
base.verify_live_pins = verify_live_pins
base.verify_terminal = verify_terminal


def main(argv=None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
