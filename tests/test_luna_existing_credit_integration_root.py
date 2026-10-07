"""Root-owned source, receipt and cap checks for the authorized billing policy."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

from benchmarks import codex_subscription_warm_v7 as warm
from tools.diagnostics import luna_lme_diagnostic_v5 as prior
from tools.diagnostics import luna_lme_diagnostic_v6 as runner
from tools.diagnostics import luna_lme_diagnostic_launch_v5 as old_launch
from tools.diagnostics import luna_lme_diagnostic_launch_v6 as launch
from tools.diagnostics import luna_lme_diagnostic_progress_v7 as progress
from tools.diagnostics import luna_lme_diagnostic_bundle_v6 as bundle
from tools.diagnostics import luna_lme_diagnostic_host_preflight_v6 as host
from tools.diagnostics import luna_subscription_access_check_v1 as old_access
from tools.diagnostics import luna_subscription_access_check_v2 as access


REPO = Path(__file__).resolve().parents[1]
ACCEPTED = Path("/private/tmp/hymem-lme-retry-offline-assembly-v1")


def test_billing_policy_only_changes_admission_not_experiment():
    assert runner.MAX_LIMITS == prior.MAX_LIMITS
    for key in ("ACCEPTED_MAP_SHA256", "ACCEPTED_FILES", "DATASET_SHA256",
                "CANDIDATE_PINS", "DIAGNOSTIC_HELPER_SHA256"):
        assert getattr(runner, key) == getattr(prior, key)
    assert runner.BILLING_POLICY == warm.BILLING_POLICY
    assert access.BILLING_POLICY == runner.BILLING_POLICY == progress.BILLING_POLICY
    assert access.LIMITS == old_access.LIMITS
    root, receipt = Path("/invented"), {"unit": "invented.service"}
    old_command = old_launch.command(root, receipt, "a" * 64)
    assert launch.command(root, receipt, "a" * 64) == [
        x.replace("luna_lme_diagnostic_v5.py", "luna_lme_diagnostic_v6.py") for x in old_command]
    assert access.command(root, receipt, "a" * 64) == [
        x.replace("access-check-v1.py", "access-check-v2.py")
        for x in old_access.command(root, receipt, "a" * 64)]


def test_actual_bundle_receipt_binds_new_policy_and_preserves_source(monkeypatch, tmp_path):
    assert ACCEPTED.is_dir(), "Reviewed source-only input must exist for root verification"
    root = tmp_path / ".hymem-lme-diagnostic-creditroot"
    result = bundle.assemble(repo=REPO, accepted_code=ACCEPTED / "code",
        candidate=ACCEPTED / "candidate", map_path=ACCEPTED / "source-map.json", output=root)
    assert result["candidate_files"] == 514
    assert len(result["code_sha256"]) == 13
    payload = host.archive_bytes(root)
    nodes = []
    for node in ast.parse(host.REMOTE).body:
        source = ast.get_source_segment(host.REMOTE, node) or ""
        if source.startswith("need(os.getuid()"):
            continue
        if source.startswith("need(regular(DATASET)"):
            break
        nodes.append(node)
    else:
        pytest.fail("Could not identify remote verification side-effect boundary")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<root-credit-archive>", "exec"), scope)
    assert len(scope["manifest"]) == 528
    assert len([name for name in scope["manifest"] if name.startswith("code/")]) == 13
    binary = tmp_path / "invented-binary"
    binary.write_bytes(b"invented binary; never executed")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    monkeypatch.setattr(launch, "_root", lambda value: value)
    monkeypatch.setattr(launch, "BINARY_SHA256", digest)
    receipt = launch.receipt_for(root, runner)
    assert receipt["billing_policy"] == warm.BILLING_POLICY
    assert receipt["source_sha256"] == progress.PINS
    loaded = {"root": root, "code": root / "code", "binary": binary,
              "questions": [{"question_id": str(index)} for index in range(4)]}
    def place(value):
        file = root / "launch-receipt.json"
        file.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
        pin = hashlib.sha256(file.read_bytes()).hexdigest()
        (root / "launch-attempt.json").write_text(json.dumps({"receipt_sha256": pin, "one_shot": True}))
        return pin
    assert runner.verify_launch_receipt(root, place(receipt), loaded) == receipt
    for replacement in (None, "subscription_only", True, {}, "unlimited_credit_purchases"):
        altered = copy.deepcopy(receipt)
        altered["billing_policy"] = replacement
        with pytest.raises(ValueError):
            runner.verify_launch_receipt(root, place(altered), loaded)
    altered = copy.deepcopy(receipt)
    altered.pop("billing_policy")
    with pytest.raises(ValueError):
        runner.verify_launch_receipt(root, place(altered), loaded)


@pytest.mark.parametrize("route", ["ordinary", "staged"])
@pytest.mark.parametrize("outcome", ["success", "terminal", "missing_usage"])
def test_new_routes_preserve_root_retry_capture_and_accounting(monkeypatch, tmp_path, route, outcome):
    # Reuse root-owned invented-stream controls, changing only module bindings
    # and the added v7 -> v6 lineage level. No provider or shell process exists.
    path = REPO / "tests/test_luna_retry_integration_root.py"
    source = path.read_text()
    source = source.replace("codex_subscription_staged_v3 as staged", "codex_subscription_staged_v4 as staged")
    source = source.replace("luna_lme_diagnostic_v5 as runner", "luna_lme_diagnostic_v6 as runner")
    source = source.replace("warm.v5.v4.v3.WarmSession", "warm.v6.v5.v4.v3.WarmSession")
    scope = {"__file__": str(path), "__name__": "root_credit_stream_controls"}
    exec(compile(source, str(path), "exec"), scope)
    scope["test_actual_runner_routes_retry_and_capture_without_second_turn"](
        monkeypatch, tmp_path, route, outcome)
