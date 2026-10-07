"""Root-owned source closure and experiment-invariance checks; never inference."""
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

from tools.diagnostics import luna_lme_diagnostic_v7 as prior
from tools.diagnostics import luna_lme_diagnostic_v8 as runner
from tools.diagnostics import luna_lme_diagnostic_launch_v7 as old_launch
from tools.diagnostics import luna_lme_diagnostic_launch_v8 as launch
from tools.diagnostics import luna_lme_diagnostic_bundle_v8 as bundle
from tools.diagnostics import luna_lme_diagnostic_host_preflight_v8 as host
from tools.diagnostics import luna_lme_diagnostic_progress_v9 as progress

REPO = Path(__file__).resolve().parents[1]
ACCEPTED = Path("/private/tmp/hymem-lme-delta-offline-assembly-v1")


def test_experiment_and_headless_resource_command_are_unchanged():
    for name in ("MAX_LIMITS", "ACCEPTED_MAP_SHA256", "ACCEPTED_FILES", "DATASET_SHA256",
                 "CANDIDATE_PINS", "DIAGNOSTIC_HELPER_SHA256", "NOTIFICATION_POLICY"):
        assert getattr(runner, name) == getattr(prior, name)
    assert runner.BILLING_POLICY == "included_allowance_or_existing_finite_positive_credits_per_window_v2"
    root, receipt = Path("/invented"), {"unit": "invented.service"}
    assert launch.command(root, receipt, "a"*64) == [
        item.replace("luna_lme_diagnostic_v7.py", "luna_lme_diagnostic_v8.py")
        for item in old_launch.command(root, receipt, "a"*64)]
    assert runner.PINS["benchmarks/codex_subscription_timeout_v3.py"] == "2d59fb8c59c5e304e557b05dbd346998ed46a9c2ba14a726924dc0517d352778"
    assert runner.PINS["benchmarks/codex_subscription_staged_v6.py"] == "558a6cdee2c5562ff1bd4106b3abc2dadad77a6b3fd980e69420678248af2900"


def test_actual_source_bundle_remote_decoder_and_isolated_import(monkeypatch, tmp_path):
    root = tmp_path / ".hymem-lme-diagnostic-rootinstrumented"
    result = bundle.assemble(repo=REPO, accepted_code=ACCEPTED/"code",
        candidate=ACCEPTED/"candidate", map_path=ACCEPTED/"source-map.json", output=root)
    assert result["candidate_files"] == 514 and len(result["code_sha256"]) == 16
    assert result["model_calls"] == 0 and result["dataset_present"] is False
    assert result["launch_receipt_present"] is False and result["binary_present"] is False
    payload = host.archive_bytes(root)
    nodes = []
    for node in ast.parse(host.REMOTE).body:
        statement = ast.get_source_segment(host.REMOTE, node) or ""
        if statement.startswith("need(os.getuid()"):
            continue
        if statement.startswith("need(regular(DATASET)"):
            break
        nodes.append(node)
    else:
        raise AssertionError("Remote dataset/side-effect boundary absent")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<root-instrumented-archive>", "exec"), namespace)
    assert len(namespace["manifest"]) == 531
    assert sum(name.startswith("code/") for name in namespace["manifest"]) == 16
    assert sum(name.startswith("candidate/") for name in namespace["manifest"]) == 514
    script = """
import inspect,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'candidate'),str(root/'code')]
from benchmarks import codex_subscription_staged_v6 as staged
from benchmarks import codex_subscription_timeout_v3 as observer
from tools.diagnostics import luna_lme_diagnostic_v8 as runner
assert staged._ROOT==root/'candidate'
assert staged.observer is observer and staged.warm is observer.warm
assert observer.SharedBudget is staged.warm.SharedBudget
assert Path(observer.warm.__file__)==root/'code/benchmarks/codex_subscription_warm_v9.py'
assert staged.warm.BILLING_POLICY==runner.BILLING_POLICY
assert staged.warm.NOTIFICATION_POLICY==runner.NOTIFICATION_POLICY
assert inspect.signature(staged.StagedSubscriptionClient.__init__).parameters['session_factory'].default is observer.TimeoutSession
assert inspect.signature(observer.TimeoutSubscriptionClient.__init__).parameters['session_factory'].default is observer.TimeoutSession
print('source_only_import_verified')
"""
    checked = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(root)],
        capture_output=True, text=True, timeout=20, check=True)
    assert checked.stdout.strip() == "source_only_import_verified"
    monkeypatch.setattr(launch, "_root", lambda value: value)
    receipt = launch.receipt_for(root, runner)
    assert receipt["source_sha256"] == progress.PINS
    assert receipt["billing_policy"] == runner.BILLING_POLICY
    for relative, pin in result["code_sha256"].items():
        assert hashlib.sha256((root/"code"/relative).read_bytes()).hexdigest() == pin
    assert (ACCEPTED/"code/tools/diagnostics/luna_lme_diagnostic_v7.py").is_file()
    assert not (ACCEPTED/"code/tools/diagnostics/luna_lme_diagnostic_v8.py").exists()
