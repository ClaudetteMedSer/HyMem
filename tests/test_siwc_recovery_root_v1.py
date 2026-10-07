"""Independent recovery-worker integration checks, all provider/owner I/O mocked."""
import json
import time
from pathlib import Path

import pytest

from benchmarks import chatgpt_plan_lme_v1 as bridge
from hymem.extraction.llm import LLMRequest
from tools.diagnostics import siwc_lme_recovery_check_v1 as recovery


@pytest.mark.parametrize("mode", ["success", "503", "owner_denied", "overshoot", "wrong_output"])
def test_exact_bridge_worker_and_reader_reconcile(tmp_path, monkeypatch, mode):
    digest = "a" * 64
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(recovery, "UID", root.stat().st_uid)
    receipt = recovery.receipt_for(root, {recovery.SELF: "b" * 64})
    monkeypatch.setattr(recovery, "root_checked", lambda path: path)
    monkeypatch.setattr(recovery, "receipt_checked", lambda path, pin: receipt)
    monkeypatch.setattr(recovery, "execute_containment", lambda _: tmp_path)
    monkeypatch.setattr(recovery, "failure_codes_readonly", lambda _: bridge._CODES)
    monkeypatch.setattr(recovery, "source_only", lambda _: (None, {
        "siwc": bridge, "warm": bridge.warm, "request_type": LLMRequest}))
    monkeypatch.setattr(recovery, "resource", lambda _: {
        "peak": 8, "denials": 0, "oom": 0, "oom_kill": 0})
    model_calls, owner_calls, closes = [], [], []

    def owner_init(self, *args):
        self.identity_digest = recovery.GRANT_SHA

    def acquire(self, *, caller_deadline):
        owner_calls.append(caller_deadline)
        if mode == "owner_denied":
            raise bridge.owner.OwnerError("refresh_denied")
        return bridge.owner.CredentialLease("invented-offline", int(time.time()) + 900)

    def response(credentials, system, user, schema, *, timeout):
        model_calls.append(timeout)
        assert system == recovery.SYSTEM and user == recovery.USER and schema is None
        assert 0 < timeout <= 120
        if mode == "503":
            raise bridge.transport.TransportError(
                "subscription_sharing_user_unavailable", 503, "error_object", "missing")
        tokens = 160001 if mode == "overshoot" else 14
        return bridge.transport.Completed(
            "WRONG" if mode == "wrong_output" else recovery.EXPECTED,
            tokens - 2, 2, tokens, 0, 0)

    original_init = bridge.SIWCLMEClient.__init__

    def client_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs, response_call=response)

    monkeypatch.setattr(bridge.owner.CredentialBroker, "__init__", owner_init)
    monkeypatch.setattr(bridge.owner.CredentialBroker, "acquire", acquire)
    monkeypatch.setattr(bridge.owner.CredentialBroker, "close", lambda self: closes.append(True))
    monkeypatch.setattr(bridge.SIWCLMEClient, "__init__", client_init)
    recovery.write_once(root / "recovery-attempt.json", {
        "receipt_sha256": digest, "one_shot": True})
    exitcode = recovery.execute(root, digest)
    result = json.loads((root / "recovery-result.json").read_bytes())
    assert len(owner_calls) == 1 and closes == [True]
    assert len(model_calls) == (0 if mode == "owner_denied" else 1)
    assert result["in_flight"] == result["reserved"] == 0
    assert exitcode == (0 if mode == "success" else 1)
    monkeypatch.setattr(recovery, "service_state", lambda _: ("terminal", True, exitcode == 0))
    projected = recovery.inspect(root, digest)
    assert projected["runtime_cleanup_verified"] is True
    assert projected["recovery_verified"] is (mode == "success")
    assert projected["usage_complete"] is (mode != "503")
    if mode == "503":
        assert projected["first_failure_code"] == "subscription_sharing_user_unavailable"
        assert projected["failed_turn_usage_unknown"] is True
    elif mode == "overshoot":
        assert projected["known_tokens"] == 160001
    elif mode == "owner_denied":
        assert projected["admitted_turns"] == 0
        assert projected["failed_turn_usage_unknown"] is False
    assert recovery.EXPECTED not in json.dumps(projected)
    # A consumed execution marker blocks a second invocation before owner/model access.
    with pytest.raises(FileExistsError):
        recovery.execute(root, digest)
    assert len(owner_calls) == 1


def test_error_allowlist_is_exact_accepted_bridge():
    frozen = Path('/private/tmp/hymem-siwc-prompt-root-BM1kL3Kf/bundle')
    assert recovery.failure_codes_readonly(frozen) == bridge._CODES


@pytest.mark.parametrize('field,value', [('one_shot', 1), ('stream', 1), ('store', 0)])
def test_receipt_boolean_coercion_rejected(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(recovery, 'UID', tmp_path.stat().st_uid)
    sources = {recovery.SELF: 'c' * 64}
    monkeypatch.setattr(recovery, 'identity_files', lambda _: sources)
    monkeypatch.setattr(recovery, 'source_tree_readonly', lambda _: None)
    receipt = recovery.receipt_for(tmp_path, sources)
    receipt[field] = value
    path = tmp_path / 'recovery-receipt.json'
    recovery.write_once(path, receipt)
    with pytest.raises(ValueError, match='receipt_identity_invalid'):
        recovery.receipt_checked(tmp_path, recovery.sha(path))
