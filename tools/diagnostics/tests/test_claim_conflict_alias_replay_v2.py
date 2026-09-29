"""Network-free controls for the fresh corrected replay stage."""
from pathlib import Path
from unittest.mock import patch

from tools.diagnostics import claim_conflict_alias_replay_v2_host as host


def test_stage_is_fresh_and_old_host_remains_pinned():
    local = Path(__file__).resolve().parents[1]
    with patch.object(host, "V1_HOST", local / "claim_conflict_alias_replay_host.py"), \
         patch.object(host, "WORKER", local / "claim_conflict_instrumented_replay.py"):
        controller = host.v1_controller()
        host.pin_worker(controller)
        assert controller.ROOT == host.ROOT
        assert controller.ROOT.name == "alias-replay-v2"
        assert controller.SELF == host.SELF
        assert controller.CAPTURE == host.ROOT / "capture"
        assert controller.BASELINE_CANDIDATE == host.ROOT / "baseline"
        assert controller.FIXED_CANDIDATE == host.ROOT / "fixed"
        assert controller.WORK == host.ROOT / "work"
        assert host.V1_HOST_SHA == controller.sha(local / "claim_conflict_alias_replay_host.py")


def test_worker_drift_fails_before_remote_controller_action():
    local = Path(__file__).resolve().parents[1]
    with patch.object(host, "V1_HOST", local / "claim_conflict_alias_replay_host.py"), \
         patch.object(host, "WORKER", local / "claim_conflict_alias_replay_host.py"):
        controller = host.v1_controller()
        try:
            host.pin_worker(controller)
        except RuntimeError as exc:
            assert str(exc) == "replay_worker_pin_drift"
        else:
            raise AssertionError("worker drift accepted")
