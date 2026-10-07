"""Root-owned integration checks for the immutable delta opt-out pilot."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_warm_v8 as warm
from tools.diagnostics import luna_lme_diagnostic_v6 as prior
from tools.diagnostics import luna_lme_diagnostic_v7 as runner
from tools.diagnostics import luna_lme_diagnostic_launch_v6 as old_launch
from tools.diagnostics import luna_lme_diagnostic_launch_v7 as launch
from tools.diagnostics import luna_lme_diagnostic_progress_v8 as progress
from tools.diagnostics import luna_lme_diagnostic_bundle_v7 as bundle
from tools.diagnostics import luna_lme_diagnostic_host_preflight_v7 as host

REPO = Path(__file__).resolve().parents[1]
ACCEPTED = Path("/private/tmp/hymem-lme-existing-credit-offline-assembly-v1")


def test_root_notification_policy_is_not_a_budget_or_semantic_change():
    for key in ("MAX_LIMITS", "ACCEPTED_MAP_SHA256", "ACCEPTED_FILES", "DATASET_SHA256",
                "CANDIDATE_PINS", "DIAGNOSTIC_HELPER_SHA256", "BILLING_POLICY"):
        assert getattr(runner, key) == getattr(prior, key)
    assert runner.NOTIFICATION_POLICY == warm.NOTIFICATION_POLICY == progress.NOTIFICATION_POLICY
    root, receipt = Path("/invented"), {"unit": "invented.service"}
    assert launch.command(root, receipt, "a" * 64) == [
        item.replace("luna_lme_diagnostic_v6.py", "luna_lme_diagnostic_v7.py")
        for item in old_launch.command(root, receipt, "a" * 64)]


def test_root_actual_bundle_and_receipt_bind_both_policies(monkeypatch, tmp_path):
    assert ACCEPTED.is_dir(), "Reviewed source-only input required"
    root = tmp_path / ".hymem-lme-diagnostic-deltaroot"
    result = bundle.assemble(repo=REPO, accepted_code=ACCEPTED / "code",
        candidate=ACCEPTED / "candidate", map_path=ACCEPTED / "source-map.json", output=root)
    assert result["candidate_files"] == 514 and len(result["code_sha256"]) == 14
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
        pytest.fail("Remote verification side-effect boundary missing")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<root-delta-archive>", "exec"), scope)
    assert len(scope["manifest"]) == 529
    assert len([name for name in scope["manifest"] if name.startswith("code/")]) == 14
    imported = subprocess.run([sys.executable, "-I", "-B", "-c", """
import inspect,sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path[:0]=[str(root/'candidate'),str(root/'code')]
from benchmarks import codex_subscription_staged_v5 as staged
from tools.diagnostics import luna_lme_diagnostic_v7 as runner
assert staged._ROOT==root/'candidate'
assert Path(staged.warm.__file__)==root/'code/benchmarks/codex_subscription_warm_v8.py'
assert staged.warm.NOTIFICATION_POLICY==runner.NOTIFICATION_POLICY
assert staged.warm.BILLING_POLICY==runner.BILLING_POLICY
assert inspect.signature(staged.warm.WarmSubscriptionClient.__init__).parameters['session_factory'].default is staged.warm.WarmSession
print('source_only_import_verified')
""", str(root)], capture_output=True, text=True, check=True, timeout=20)
    assert imported.stdout.strip() == "source_only_import_verified"
    binary = tmp_path / "invented-binary"
    binary.write_bytes(b"invented binary; never executed")
    monkeypatch.setattr(launch, "_root", lambda value: value)
    monkeypatch.setattr(launch, "BINARY_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest())
    receipt = launch.receipt_for(root, runner)
    assert receipt["billing_policy"] == warm.BILLING_POLICY
    assert receipt["notification_policy"] == warm.NOTIFICATION_POLICY
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
    for policy in ("notification_policy", "billing_policy"):
        for replacement in (None, "unapproved_policy", True, {}, [warm.NOTIFICATION_POLICY]):
            altered = copy.deepcopy(receipt)
            altered[policy] = replacement
            with pytest.raises(ValueError):
                runner.verify_launch_receipt(root, place(altered), loaded)
        altered = copy.deepcopy(receipt)
        altered.pop(policy)
        with pytest.raises(ValueError):
            runner.verify_launch_receipt(root, place(altered), loaded)


@pytest.mark.parametrize("route", ["ordinary", "staged"])
@pytest.mark.parametrize("outcome", ["success", "terminal", "missing_usage"])
def test_root_real_routes_preserve_retry_failure_capture_accounting(monkeypatch, tmp_path, route, outcome):
    path = REPO / "tests/test_luna_retry_integration_root.py"
    source = path.read_text()
    source = source.replace("codex_subscription_staged_v3 as staged", "codex_subscription_staged_v5 as staged")
    source = source.replace("luna_lme_diagnostic_v5 as runner", "luna_lme_diagnostic_v7 as runner")
    source = source.replace("warm.v5.v4.v3.WarmSession", "warm.v7.v6.v5.v4.v3.WarmSession")
    scope = {"__file__": str(path), "__name__": "root_delta_stream_controls"}
    exec(compile(source, str(path), "exec"), scope)
    scope["test_actual_runner_routes_retry_and_capture_without_second_turn"](
        monkeypatch, tmp_path, route, outcome)


def test_root_actual_checkpoint_writer_and_reader_reject_policy_drift(tmp_path):
    questions = [{"question_id": "invented-" + str(i)} for i in range(4)]
    loaded = {"strictness": SimpleNamespace(content_hash=progress._canonical_hash),
              "diagnostic": SimpleNamespace(MODE=progress.MODE)}
    limits = {key: dict(zip(("turns", "known_tokens", "seconds"), values))
              for key, values in runner.MAX_LIMITS.items()}
    limits.update(indexing_seconds=10800, workers=4)
    manifest = runner._identity_manifest(loaded, questions, limits, runner.DIAGNOSTIC_HELPER_SHA256)
    ids = [question["question_id"] for question in questions]
    path = tmp_path / "checkpoint.json"
    script = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from benchmarks.strictness import AtomicCheckpoint
p=json.loads(sys.argv[2])
with AtomicCheckpoint(Path(p['path']),manifest=p['manifest'],expected_ids=p['ids'],scored=True,
                      resume=False,retry_failures=False) as cp:
    for qid in p['ids']:
        cp.record(qid,row=None,failure='question_failure')
    cp.finalize()
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ACCEPTED / "candidate"),
        json.dumps({"path": str(path), "manifest": manifest, "ids": ids})],
        capture_output=True, text=True, check=True, timeout=20)
    checkpoint = json.loads(path.read_text())
    assert progress._checkpoint(checkpoint, {"selected_count": 4}) == checkpoint
    for policy in ("notification_policy", "billing_policy"):
        for value in (None, True, {}, "unapproved_policy"):
            changed = copy.deepcopy(checkpoint)
            altered = changed["manifest"]
            altered[policy] = value
            altered.pop("run_id")
            altered["run_id"] = progress._canonical_hash(altered)
            changed["run_id"] = altered["run_id"]
            with pytest.raises(ValueError, match="checkpoint_identity_invalid"):
                progress._checkpoint(changed, {"selected_count": 4})


def test_root_notification_failure_is_finite_and_untrusted_detail_does_not_escape():
    assert progress._safe_stop("notification_optout_unverified")
    assert not progress._safe_stop("notification_optout_unverified:private")
    budget = {"stop_code": "notification_optout_unverified", "first_failure": {
        "code": "notification_optout_unverified", "phase": "run", "rpc": "turn/events",
        "process_index": 1, "request_index": 1, "queue_count": 0, "retired_count": 0,
        "turn_admitted": True, "known_usage": False, "usage_complete": False,
        "resource_observation": {"current": 128, "peak": 130, "limit": 256, "denials": 0}}}
    safe = progress._failure_projection(budget)
    assert safe["code"] == "notification_optout_unverified"
    assert safe["usage_complete"] is False and safe["known_usage"] is False
    budget["first_failure"]["code"] += ":private"
    with pytest.raises(ValueError):
        progress._failure_projection(budget)
