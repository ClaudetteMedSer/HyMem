"""Independent one-call and finite-attribution controls; no real credentials."""
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from tools.diagnostics import lme_chatgpt_plan_probe_v2 as p
from tests.test_lme_chatgpt_plan_probe_v1 import FakeCatalog, FakeTransport


class Transport(FakeTransport):
    class TransportError(FakeTransport.TransportError):
        def __init__(self, code, status=None, shape=None, media=None):
            super().__init__(code, status, shape)
            self.media_type_class = media


@pytest.fixture
def setup_probe():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        root, state = Path(tmp) / "run", Path(tmp) / "state"
        root.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        catalog, transport = FakeCatalog(), Transport()
        with patch.object(p, "_modules", return_value=(catalog, transport)):
            sha = p.prepare(root, state)["receipt_sha256"]
            yield root, state, sha, catalog, transport


def test_exactly_one_request_and_consumed_attempt(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("blue paper kite is ready", 15)]
    result = p.run(root, state, sha)
    assert result["status"] == "passed" and result["model_calls"] == 1
    assert result["usage_complete"] is True and result["known_tokens"] == 15
    assert len(transport.calls) == 1
    assert transport.calls[0] == (p.PLAIN_SYSTEM, p.PLAIN_USER, None, 120)
    assert p.status(root, sha) == result
    with pytest.raises(p.ProbeError):
        p.run(root, state, sha)
    assert len(transport.calls) == 1


def test_non_sse_metadata_survives_without_retry(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [Transport.TransportError("invalid_content_type", 200, "detail", "json")]
    result = p.run(root, state, sha)
    assert result["status"] == "failed" and result["model_calls"] == 1
    assert result["known_tokens"] == 0 and result["usage_complete"] is False
    assert result["first_fault"] == {"code": "invalid_content_type", "phase": "call_1",
        "http_status": 200, "body_shape": "detail", "media_type_class": "json"}
    assert len(transport.calls) == 1
    assert p.status(root, sha) == result


def test_expiry_preflight_leaves_attempt_unconsumed(setup_probe):
    root, state, sha, catalog, transport = setup_probe
    with patch.object(catalog, "_validated_access_token", side_effect=RuntimeError("expired")):
        with pytest.raises(Exception):
            p.run(root, state, sha)
    assert not (root / "attempt.json").exists()
    assert transport.calls == []


def test_incomplete_usage_cannot_be_passed(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("blue paper kite is ready", 15)]
    result = p.run(root, state, sha)
    result.update(known_usage=[], known_tokens=0, usage_complete=False)
    with pytest.raises(p.ProbeError):
        p._validate_result(result)


@pytest.mark.parametrize("media", ["PRIVATE_HEADER", ["PRIVATE_HEADER"], True])
def test_arbitrary_media_rejected_in_status(setup_probe, media):
    root, state, sha, _, transport = setup_probe
    transport.replies = [Transport.TransportError("invalid_content_type", 200, "detail", "json")]
    result = p.run(root, state, sha)
    result["first_fault"]["media_type_class"] = media
    with pytest.raises(p.ProbeError):
        p._validate_result(result)


def test_attribution_does_not_change_known_output_failure_usage(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("wrong invented response", 15)]
    result = p.run(root, state, sha)
    assert result["status"] == "failed"
    assert result["known_tokens"] == 15 and result["usage_complete"] is True
    assert result["first_fault"]["code"] == "plain_output_mismatch"
