"""Offline source and projection checks for the versioned delta opt-out route."""
import hashlib
from pathlib import Path

from benchmarks import codex_subscription_staged_v5 as staged
from tools.diagnostics import luna_delta_source_install_v1 as install
from tools.diagnostics import luna_lme_diagnostic_bundle_v7 as bundle
from tools.diagnostics import luna_lme_diagnostic_host_preflight_v7 as host
from tools.diagnostics import luna_lme_diagnostic_launch_v7 as launch
from tools.diagnostics import luna_lme_diagnostic_progress_v8 as progress
from tools.diagnostics import luna_lme_diagnostic_v7 as runner


REPO = Path(__file__).resolve().parents[1]


def test_pinned_client_and_policy_lineage():
    warm = staged.warm
    assert warm.NOTIFICATION_POLICY == runner.NOTIFICATION_POLICY == progress.NOTIFICATION_POLICY
    assert warm.BILLING_POLICY == runner.BILLING_POLICY == progress.BILLING_POLICY
    assert warm.WarmSession.__bases__ == (warm.v7.WarmSession,)
    assert runner.PINS["benchmarks/codex_subscription_warm_v8.py"] == warm_source_hash()
    assert runner.PINS["benchmarks/codex_subscription_staged_v5.py"] == staged_source_hash()
    assert progress.PINS == {**runner.PINS,
        "benchmarks/lme_diagnostic.py": runner.DIAGNOSTIC_HELPER_SHA256,
        "tools/diagnostics/luna_lme_diagnostic_v7.py": launch.RUNNER_SHA256}
    assert launch.RUNNER_SHA256 == bundle.RUNNER_SHA256 == host.RUNNER_SHA
    assert launch.RUNNER_SHA256 == hashlib.sha256(
        (REPO / "tools/diagnostics/luna_lme_diagnostic_v7.py").read_bytes()).hexdigest()
    assert host.RUNNER_REL == "code/" + bundle.RUNNER_RELATIVE
    assert host.SOURCE.name == "hymem-lme-delta-offline-assembly-v1"


def warm_source_hash():
    return hashlib.sha256((REPO / "benchmarks/codex_subscription_warm_v8.py").read_bytes()).hexdigest()


def staged_source_hash():
    return hashlib.sha256((REPO / "benchmarks/codex_subscription_staged_v5.py").read_bytes()).hexdigest()


def test_inert_source_installer_pins_exact_two_files_and_finite_projection():
    assert set(install.FILES) == {"access-check-v2.py", "luna_lme_diagnostic_launch_v7.py"}
    for relative, digest in install.FILES.values():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == digest
    assert "luna_lme_diagnostic_launch_v7.py" in install.REMOTE
    assert "luna_lme_diagnostic_launch_v6.py" not in install.REMOTE
    root = "/home/atta/.hymem-lme-diagnostic-preflight-invented123"
    unit = "hymem-luna-lme-diagnostic-preflight-invented123.service"
    value = {"schema": install.SCHEMA, "prepared": True, "model_calls": 0,
             "root": root, "unit": unit, "kind": "pilot",
             "receipt_sha256": "a" * 64}
    assert install._success_projection(value, root=root, kind="pilot", returncode=0) == value
    for changed in ({**value, "model_calls": True}, {**value, "schema": "other"},
                    {**value, "extra": "private"}, {**value, "kind": "access"}):
        assert install._success_projection(changed, root=root, kind="pilot", returncode=0) is None
    assert install._success_projection(value, root=root, kind="pilot", returncode=1) is None
