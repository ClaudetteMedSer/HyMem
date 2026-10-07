"""Independent policy and retained capacity checks for the attribution repair."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.diagnostics import luna_lme_diagnostic_v2 as previous
from tools.diagnostics import luna_lme_diagnostic_v3 as repaired
from tests import test_lme_diagnostic_capacity_root as capacity


def _node(module, name, method=None):
    tree = ast.parse(Path(module.__file__).read_text())
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    if method is not None:
        node = next(node for node in node.body if getattr(node, "name", None) == method)
    return ast.dump(node, include_attributes=False)


def test_repair_keeps_measurement_resources_and_accounting_exact():
    for name in ("MAX_LIMITS", "ACCEPTED_FILES", "ACCEPTED_MAP_SHA256",
                 "DATASET_SHA256", "PINS", "CANDIDATE_PINS", "DIAGNOSTIC_HELPER_SHA256"):
        assert getattr(repaired, name) == getattr(previous, name), name
    for name in ("run_live_canary", "_canary_gold", "validate_diagnostic_row",
                 "_identity_manifest", "_question_worker", "make_dual",
                 "_resource_sample", "run_campaign", "verify_live_containment",
                 "RegistrationAlias", "DualClient"):
        assert _node(repaired, name) == _node(previous, name), name
    for method in ("_call", "complete", "complete_stage", "reconcile"):
        assert _node(repaired, "AccountedClient", method) == _node(previous, "AccountedClient", method)


def test_reader_and_launcher_only_change_source_and_version_bindings():
    directory = Path(repaired.__file__).parent
    old_hash = __import__("hashlib").sha256(Path(previous.__file__).read_bytes()).hexdigest()
    new_hash = __import__("hashlib").sha256(Path(repaired.__file__).read_bytes()).hexdigest()
    for old_name, new_name in (
        ("luna_lme_diagnostic_progress_v3.py", "luna_lme_diagnostic_progress_v4.py"),
        ("luna_lme_diagnostic_launch_v2.py", "luna_lme_diagnostic_launch_v3.py"),
    ):
        expected = (directory / old_name).read_text().replace(old_hash, new_hash)
        expected = expected.replace("luna_lme_diagnostic_v2.py", "luna_lme_diagnostic_v3.py")
        expected = expected.replace("luna-lme-diagnostic-launch-v2", "luna-lme-diagnostic-launch-v3")
        expected = expected.replace("luna-lme-diagnostic-progress-v3", "luna-lme-diagnostic-progress-v4")
        assert (directory / new_name).read_text() == expected


def test_exact_resource_policy_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "_candidate", lambda: repaired)
    capacity.test_measured_task_bound_is_exact_and_other_limits_stay_fixed(tmp_path, monkeypatch)


@pytest.mark.parametrize("fault", ["rpc", "denial", "unreadable"])
def test_pre_cleanup_observer_and_dispatch_guards_retained(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(capacity, "_candidate", lambda: repaired)
    capacity.test_actual_campaign_observer_records_before_cleanup_and_stops(
        tmp_path, monkeypatch, fault)
