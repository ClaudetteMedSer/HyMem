"""Network-free controls for the fresh v64 replay stage adapter."""
from pathlib import Path

from tools.diagnostics import claim_conflict_proof_replay_v2_host as host


def test_all_v64_override_and_diagnostic_pins_match_local_files():
    host.reviewed_overrides()
    assert len(host.OVERRIDE_SHAS) == 7
    assert "hymem/core/migrations/064_local_claim_replay_proof.sql" in host.OVERRIDE_SHAS
    assert "hymem/core/migrations/062_local_claim_replay_proof.sql" not in host.OVERRIDE_SHAS
    for relative, expected in host.OVERRIDE_SHAS.items():
        assert host.sha(host.LOCAL_FROZEN / relative) == expected
    diagnostics = Path(__file__).resolve().parents[1]
    assert host.sha(diagnostics / "claim_conflict_proof_replay_v2.py") == host.WORKER_SHA
    assert host.sha(diagnostics / "claim_conflict_proof_replay_host.py") == host.V1_HOST_SHA
    assert host.sha(diagnostics / "claim_conflict_instrumented_replay.py") == host.OLD_REPLAY_SHA


def test_fresh_stage_has_extra_pinned_v1_audit_mount_and_no_network():
    local_host = Path(__file__).resolve().parents[1] / "claim_conflict_proof_replay_host.py"
    module = host.controller(local_host)
    class Helper:
        IMAGE = "sha256:" + "f" * 64
        RUNTIME = Path("/runtime")
    command, mounts = module.configure(Helper())
    assert module.ROOT.name == "proof-replay-v2"
    assert module.SELF == host.SELF
    assert command[command.index("--name") + 1] == "hymem-proof-replay-v2"
    assert command[command.index("--network") + 1] == "none"
    assert (str(host.V1_AUDIT_WORKER), "/diag/claim_conflict_proof_replay_v1.py", False) in mounts
    assert sum(writable for _, _, writable in mounts) == 1
    assert not any("runtime-env.json" in str(value) for value in command)
    assert command[-1] == host.OVERRIDE_SHAS["hymem/dreaming/phase1.py"]
