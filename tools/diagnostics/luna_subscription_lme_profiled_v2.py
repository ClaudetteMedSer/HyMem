"""Source-pinned stage accounting over the attributed warm LME runner.

This flat diagnostic bundle uses the versioned attributed transport and the
unchanged stage collector.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types


WARM_RUNNER_SHA256 = "3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567"
COLLECTOR_SHA256 = "800ef9baedc9d68093b3160cc324ef17528dd0b89f7a734f220217b07323fba2"
SCHEMA = "luna-subscription-lme-profiled-v2"


def _load_pinned(path: Path, expected: str, module_name: str):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise RuntimeError("profile_dependency_source_drift")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


_sibling_collector = Path(__file__).resolve().with_name("luna_stage_accounting.py")
_repository_collector = Path(__file__).resolve().parents[2] / "benchmarks/luna_stage_accounting.py"
_collector_path = _sibling_collector if _sibling_collector.is_file() else _repository_collector
collector = _load_pinned(_collector_path,
                         COLLECTOR_SHA256, "pinned_luna_stage_accounting")
warm_runner = _load_pinned(Path(__file__).resolve().with_name(
    "luna_subscription_lme_warm_v2.py"), WARM_RUNNER_SHA256,
    "pinned_luna_subscription_lme_warm_v2_runner")
_accepted_run_campaign = warm_runner.run_campaign

# The producer declaration must identify the diagnostic wrapper as well as
# its fixed call-site mapping. No prompt, route, model, or runtime policy changes.
PROFILE_RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
warm_runner.RUNNER_SHA256 = PROFILE_RUNNER_SHA256
warm_runner.SCHEMA = SCHEMA


def run_campaign(**kwargs):
    candidate = Path(kwargs["chunk"].__file__).resolve().parents[2]
    ledger = collector.StageLedger(candidate)
    supplied_factory = kwargs.pop("client_factory", None)
    supplied_progress = kwargs.pop("progress", None)

    def make_client(key, limits, budget):
        if supplied_factory is None:
            warm = kwargs["warm"]
            delegate = warm.WarmSubscriptionClient(
                kwargs["binary"], budget, key, limits,
                max_requests=kwargs.get("warm_max_requests", 16),
                max_age_seconds=kwargs.get("warm_max_age_seconds", 300))
        else:
            delegate = supplied_factory(key, limits, budget)
        return ledger.wrap(delegate, key)

    def progress(value):
        value["stage_accounting"] = ledger.snapshot()
        value["stage_accounting_schema"] = "source-free-stages-v1"
        value["stage_collector_sha256"] = COLLECTOR_SHA256
        if supplied_progress is not None:
            supplied_progress(value)

    result = _accepted_run_campaign(client_factory=make_client,
                                    progress=progress, **kwargs)
    result["stage_accounting"] = ledger.snapshot()
    result["stage_accounting_reconciled"] = ledger.reconcile(result["budget"] or {})
    if not result["stage_accounting_reconciled"]:
        result["campaign_stop"] = result["campaign_stop"] or "stage_accounting_mismatch"
    progress(result)
    return result


warm_runner.run_campaign = run_campaign


def main(argv=None) -> int:
    return warm_runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
