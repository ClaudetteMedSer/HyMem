"""Independent one-shot wire-diagnostic controls using invented state only."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from benchmarks import chatgpt_plan_responses_v3 as wire
from tools.diagnostics import lme_chatgpt_plan_probe_v3 as p
from tests.test_lme_chatgpt_plan_probe_v1 import FakeCatalog, FakeTransport


class Transport(FakeTransport):
    TransportError = wire.TransportError
    _sanitize_observation = staticmethod(wire._sanitize_observation)


def observation():
    return {"header_defect": "missing_separator", "parsed_header_count": 0,
        "transfer_encoding": "missing", "content_encoding": "missing",
        "body_prefix": "sse_prefix", "body_bytes": 123, "body_truncated": False,
        "sse_validation": "validated_completion"}


@pytest.fixture
def setup_probe():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
        root, state = Path(name) / "run", Path(name) / "state"
        root.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        catalog, transport = FakeCatalog(), Transport()
        with patch.object(p, "_modules", return_value=(catalog, transport)):
            sha = p.prepare(root, state)["receipt_sha256"]
            yield root, state, sha, catalog, transport


def fail_once(setup):
    root, state, sha, _, transport = setup
    transport.replies = [wire.TransportError("invalid_content_type", 200, "non_json",
                                             "missing", observation())]
    result = p.run(root, state, sha)
    assert result["status"] == "failed"
    assert result["usage_complete"] is False and result["known_usage"] == []
    assert len(transport.calls) == 1
    return result


def test_diagnostic_evidence_not_mislabelled_as_success(setup_probe):
    root, state, sha, _, transport = setup_probe
    result = fail_once(setup_probe)
    assert result["first_fault"]["wire_observation"] == observation()
    assert result["first_fault"]["code"] == "invalid_content_type"
    assert p.status(root, sha) == result
    with pytest.raises(p.ProbeError):
        p.run(root, state, sha)
    assert len(transport.calls) == 1


def test_success_still_one_exact_call_and_complete_usage(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("blue paper kite is ready", 15)]
    result = p.run(root, state, sha)
    assert result["status"] == "passed" and result["first_fault"] is None
    assert result["usage_complete"] and result["known_tokens"] == 15
    assert transport.calls == [(p.PLAIN_SYSTEM, p.PLAIN_USER, None, 120)]
    assert p.status(root, sha) == result


@pytest.mark.parametrize("change", [
    {"raw_body": "PRIVATE_SENTINEL"}, {"header_defect": ["PRIVATE_SENTINEL"]},
    {"parsed_header_count": True}, {"body_bytes": -1},
    {"body_prefix": "text"}, {"body_truncated": True},
    {"sse_validation": "PRIVATE_SENTINEL"},
])
def test_status_rejects_tampered_evidence_before_export(setup_probe, change):
    root, _, sha, _, _ = setup_probe
    result = fail_once(setup_probe)
    result["first_fault"]["wire_observation"].update(change)
    (root / "result.json").write_text(json.dumps(result))
    with pytest.raises(p.ProbeError) as caught:
        p.status(root, sha)
    assert "PRIVATE_SENTINEL" not in str(caught.value)


def test_local_fault_cannot_invent_upstream_observation(setup_probe):
    result = fail_once(setup_probe)
    result["first_fault"].update(code="internal_error", http_status=None,
                                 body_shape=None, media_type_class=None)
    with pytest.raises(p.ProbeError):
        p._validate_result(result)


def test_probe_pins_complete_transport_closure():
    hashes = p._source_hashes()
    for version in ("transport_v1", "transport_v2", "transport_v3"):
        assert len(hashes[version]) == 64
    assert hashes["transport_v3"] == "c5169d498c8f3f2160141d5d649ec7cf27b25b4ba1645d7e607f43b3e3074659"
    assert p.MAX_CALLS == 1 and p.CALL_SECONDS == 120
    assert p.MAX_TOKENS == 160000 and p.MAX_SECONDS == 300


def test_expired_credential_never_consumes_attempt(setup_probe):
    root, state, sha, catalog, transport = setup_probe
    with patch.object(catalog, "_validated_access_token", side_effect=RuntimeError("expired")):
        with pytest.raises(Exception):
            p.run(root, state, sha)
    assert not (root / "attempt.json").exists()
    assert transport.calls == []


def test_prefix_heuristic_does_not_override_valid_json_classification(setup_probe):
    result = fail_once(setup_probe)
    result["first_fault"].update(code="subscription_sharing_usage_unavailable",
                                 body_shape="error_object")
    result["first_fault"]["wire_observation"].update(body_prefix="text", body_bytes=1000,
                                                   sse_validation="not_checked")
    p._validate_result(result)
