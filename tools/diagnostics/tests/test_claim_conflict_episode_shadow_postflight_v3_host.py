from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_episode_shadow_postflight_v3_host as host
from tools.diagnostics import claim_conflict_episode_shadow_postflight_v3_install as installer


def test_v3_pins_match_and_old_stage_is_separate():
    assert set(installer.files()) == set(installer.PINS)
    assert host.STAGE.name == "episode-shadow-postflight-v3"
    assert "postflight-v1" not in installer.STAGE
    assert installer.PINS["claim_conflict_episode_shadow_postflight_v3.py"] == host.CHECKER_SHA
    assert installer.PINS["claim_conflict_episode_shadow_embedding_public.json"] == host.PROFILE_SHA
    assert installer.PINS["claim_conflict_episode_shadow_semantic_public.json"] == host.SEMANTIC_SHA


def test_v3_offline_mounts_profile_readonly_and_explicit_pins():
    command, mounts = host.configure(SimpleNamespace(RUNTIME=Path("/runtime")), "a" * 64)
    assert command[command.index("--network") + 1] == "none"
    assert "--env" not in command and "--env-file" not in command
    assert [dst for _, dst, rw in mounts if rw] == ["/work"]
    assert (str(host.ROOT / "work/live"), "/private-dream", False) in mounts
    assert (str(host.STAGE / "claim_conflict_episode_shadow_embedding_public.json"), "/diag/embedding-public.json", False) in mounts
    assert command[command.index("--embedding-profile-sha256") + 1] == host.PROFILE_SHA
    assert command[command.index("--semantic-config-sha256") + 1] == host.SEMANTIC_SHA
    assert (str(host.STAGE / "claim_conflict_episode_shadow_semantic_public.json"), "/diag/semantic-config-hashes.json", False) in mounts


def test_v3_report_discloses_repair_and_unfinished_convergence_safely():
    report = {"status": "pass", "repair_passed": True, "convergence_verified": False,
        "bounded_work_remaining": True, "bounded_progress": {"valid": True, "pending_domains": 3},
        "worker_boolean_gates": {"budget_exhausted": True, "skipped_locked": False},
        "gates": {"budget_stop_explained": True, "worker_boolean_gates_clear": False}}
    assert host.safe_report(report) == report


@pytest.mark.parametrize("action", ["seal", "run"])
def test_v3_never_launches_without_explicit_ready(action, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("unexpected SSH")
    monkeypatch.setattr(installer.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="explicit_ready_required"):
        installer.dispatch(SimpleNamespace(action=action, ready=False))

