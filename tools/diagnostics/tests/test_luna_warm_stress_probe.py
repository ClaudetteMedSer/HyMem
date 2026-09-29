"""Model-free checks for the one-shot stress harness and privacy boundary."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_warm_stress_probe as probe


def test_first_failure_drops_text_and_unapproved_fields():
    warm = SimpleNamespace(_safe_code=lambda exc: exc.args[0] if exc.args[0] == "timeout" else "fixed_other")
    result = probe._safe_failure({"code": "timeout", "route_phase": "turn_events",
        "last_rpc": "turn/start", "known_tokens": 42, "turn_admitted": True,
        "provider_message": "secret response", "arbitrary_count": 9,
        "thread_id": "secret-thread", "rpc": "private/custom",
        "phase": "provider says secret"}, warm)
    assert result == {"code": "timeout", "route_phase": "turn_events",
                      "last_rpc": "turn/start", "known_tokens": 42,
                      "turn_admitted": True}


def test_event_metadata_has_only_classifications():
    warm = SimpleNamespace(_EVENT_METHODS=frozenset({"thread/status/changed", "warning"}))
    session = SimpleNamespace(active_thread="secret-active", retired_threads={"secret-retired"})
    event = {"method": "thread/status/changed", "params": {
        "threadId": "secret-retired", "status": {"type": "notLoaded"},
        "message": "secret-prompt"}}
    metadata = probe._event_metadata(event, session, warm)
    assert metadata == {"event_method": "thread/status/changed",
                        "thread_relation": "retired", "lifecycle_status": "notLoaded"}
    assert "secret" not in json.dumps(metadata)
    rpc = probe._event_metadata({"id": 7, "error": {"code": -32000,
        "message": "secret provider error"}}, session, warm)
    assert rpc["rpc_error_code"] == -32000
    assert "secret" not in json.dumps(rpc)


def test_overlap_counts_four_simultaneous_invocations():
    assert probe._peak_overlap([(0, 4), (1, 3), (1.5, 2), (1.75, 2.5)]) == 4
    assert probe._peak_overlap([(0, 1), (1, 2)]) == 1


def test_private_result_is_atomic_and_one_shot(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    probe.claim_once(root, "a" * 64)
    with pytest.raises(FileExistsError):
        probe.claim_once(root, "a" * 64)
    probe.atomic_private(root, {"ok": False, "first_failure": {"code": "timeout"}})
    assert json.loads((root / probe.RESULT_NAME).read_text())["first_failure"]["code"] == "timeout"
    assert (root / probe.RESULT_NAME).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="result_exists"):
        probe.atomic_private(root, {"ok": True})


def test_bundle_pin_is_checked_before_candidate_import(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    binary = tmp_path / "codex"
    binary.write_bytes(b"placeholder")
    warm = tmp_path / "codex_subscription_warm_v2.py"
    warm.write_bytes(b"wrong source")
    with pytest.raises(ValueError, match="transport_pin_invalid"):
        probe.load_verified(candidate, binary, warm, "0" * 64)
