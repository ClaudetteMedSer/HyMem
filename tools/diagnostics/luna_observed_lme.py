"""Source-bound grounding runner with the accepted observation-only v3 client."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys


GROUNDED_RUNNER_SHA256 = "5956ebbed67b9e0a7bcbf812d7d819c1878f7fe91f8afd369447b539b2e81745"
WARM_V3_SHA256 = "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d"
WARM_V2_SHA256 = "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593"
SCHEMA = "luna-observed-lme-v1"


def pinned(name: str, digest: str, identity: str):
    path = Path(__file__).resolve().with_name(name)
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError("observed_dependency_drift")
    spec = importlib.util.spec_from_file_location(identity, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("observed_dependency_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[identity] = module
    spec.loader.exec_module(module)
    return module


def bind(grounding, profile):
    warm_runner = profile.warm_runner
    if (grounding.SCHEMA != "luna-grounding-lme-v1" or
            warm_runner.WARM_SHA256 != WARM_V2_SHA256):
        raise RuntimeError("observed_transport_binding_invalid")
    original_load = warm_runner.load_verified
    original_safe = warm_runner.safe_first_failure
    original_campaign = profile.run_campaign
    selected = {"module": None}

    def load_verified(**kwargs):
        if (Path(kwargs["warm_path"]).name != "codex_subscription_warm_v2.py" or
                Path(kwargs["warm_path"]).parent != Path(__file__).resolve().parent or
                hashlib.sha256(Path(kwargs["warm_path"]).read_bytes()).hexdigest() != WARM_V2_SHA256):
            raise RuntimeError("observed_transport_layout_invalid")
        loaded = original_load(**kwargs)
        v3 = pinned("codex_subscription_warm_v3.py", WARM_V3_SHA256,
                    "pinned_observed_warm_v3")
        if (v3.v2.__file__ != str(Path(v3.__file__).with_name("codex_subscription_warm_v2.py")) or
                v3.WarmSubscriptionClient.__module__ != v3.__name__):
            raise RuntimeError("observed_transport_binding_invalid")
        if loaded[-1].WarmSubscriptionClient is v3.WarmSubscriptionClient:
            raise RuntimeError("observed_transport_mix_invalid")
        selected["module"] = v3
        return (loaded[0], v3.concurrent, *loaded[2:-1], v3)

    def safe_failure(value, warm):
        v3 = selected["module"]
        if v3 is None or warm is not v3 or original_safe(value, v3.v2) is None:
            return None
        projected = v3.serialize_failure(value)
        if projected is None:
            return None
        required = original_safe(value, v3.v2)
        if projected.get("code") != required["code"]:
            return None
        return {**required, **projected}

    def campaign(**kwargs):
        if selected["module"] is None or kwargs.get("warm") is not selected["module"] or kwargs.get("concurrent") is not selected["module"].concurrent:
            raise RuntimeError("observed_transport_mix_invalid")
        user_progress = kwargs.get("progress")
        def progress(value):
            value["effective_warm_transport_sha256"] = WARM_V3_SHA256
            value["inherited_warm_transport_sha256"] = WARM_V2_SHA256
            if user_progress is not None:
                user_progress(value)
        kwargs["progress"] = progress
        result = original_campaign(**kwargs)
        result["effective_warm_transport_sha256"] = WARM_V3_SHA256
        result["inherited_warm_transport_sha256"] = WARM_V2_SHA256
        return result

    warm_runner.load_verified = load_verified
    warm_runner.safe_first_failure = safe_failure
    profile.run_campaign = campaign
    warm_runner.run_campaign = campaign
    warm_runner.RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    warm_runner.SCHEMA = SCHEMA


def main(argv=None) -> int:
    grounding = pinned("luna_grounding_lme.py", GROUNDED_RUNNER_SHA256,
                       "pinned_observed_grounding_runner")
    builder = grounding._pinned_module("luna_grounding_candidate.py",
                                      grounding.BUILDER_SHA256, "pinned_observed_builder")
    profile = grounding._pinned_module("luna_subscription_lme_profiled_v2.py",
                                      grounding.PROFILED_SHA256, "pinned_observed_profile")
    args = list(sys.argv[1:] if argv is None else argv)
    def option(name):
        flag = "--" + name
        if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
            raise RuntimeError("observed_argument_invalid")
        return args[args.index(flag) + 1]
    grounding.bind_profile(profile, builder, Path(option("candidate")),
        Path(option("inventory-stamp")), Path(__file__).resolve().with_name(
            grounding.ORIGINAL_INVENTORY_NAME), option("inventory-sha256"))
    bind(grounding, profile)
    # The grounding binding proves the exact 508-file one-prompt delta.
    captured = io.StringIO()
    with redirect_stdout(captured):
        code = profile.main(args)
    lines = captured.getvalue().splitlines()
    if len(lines) != 1:
        raise RuntimeError("observed_report_invalid")
    report = json.loads(lines[0])
    if type(report) is not dict or report.get("schema") != SCHEMA:
        raise RuntimeError("observed_report_invalid")
    report["effective_warm_transport_sha256"] = WARM_V3_SHA256
    report["inherited_warm_transport_sha256"] = WARM_V2_SHA256
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
