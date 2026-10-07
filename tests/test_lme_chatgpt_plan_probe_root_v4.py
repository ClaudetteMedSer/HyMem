"""Independent source-bound SSE compatibility probe controls, offline only."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from benchmarks import chatgpt_plan_responses_v4 as wire
from tools.diagnostics import lme_chatgpt_plan_probe_v4 as p
from tests.test_lme_chatgpt_plan_probe_v1 import FakeCatalog, FakeTransport


class Transport(FakeTransport):
    TransportError = wire.TransportError
    _v3 = wire._v3
    _sanitize_stream_observation = staticmethod(wire._sanitize_stream_observation)


def stream_observation():
    return {"event_type": "response.completed", "terminal_status": "completed",
            "terminal_model_matches": True, "terminal_output_kind": "message",
            "terminal_channel": "missing", "terminal_content_kind": "output_text"}


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


def failed(setup):
    root, state, sha, _, transport = setup
    transport.replies = [wire.TransportError("missing_usage", 200, "sse_event", "missing",
                                            stream_observation=stream_observation())]
    result = p.run(root, state, sha)
    assert result["status"] == "failed" and not result["usage_complete"]
    assert result["known_usage"] == [] and len(transport.calls) == 1
    assert result["first_fault"]["stream_observation"] == stream_observation()
    assert p.status(root, sha) == result
    return result


def test_semantic_failure_retained_without_retry(setup_probe):
    root, state, sha, _, transport = setup_probe
    failed(setup_probe)
    with pytest.raises(p.ProbeError):
        p.run(root, state, sha)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("change", [
    {"raw_text": "PRIVATE_SENTINEL"}, {"event_type": "PRIVATE_SENTINEL"},
    {"terminal_model_matches": 1}, {"terminal_channel": ["PRIVATE_SENTINEL"]},
    {"terminal_output_kind": "PRIVATE_SENTINEL"},
    {"terminal_status": {}}, {"terminal_content_kind": None},
])
def test_tampered_observation_rejected_before_export(setup_probe, change):
    root, _, sha, _, _ = setup_probe
    result = failed(setup_probe)
    result["first_fault"]["stream_observation"].update(change)
    (root / "result.json").write_text(json.dumps(result))
    with pytest.raises(p.ProbeError) as caught:
        p.status(root, sha)
    assert "PRIVATE_SENTINEL" not in str(caught.value)


@pytest.mark.parametrize("change", [
    {"http_status": 403}, {"media_type_class": "html"}, {"body_shape": "non_json"},
    {"code": "internal_error"},
])
def test_stream_observation_must_match_actual_fault_path(setup_probe, change):
    result = failed(setup_probe)
    result["first_fault"].update(change)
    with pytest.raises(p.ProbeError):
        p._validate_result(result)


def test_one_exact_fixture_and_complete_usage(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("blue paper kite is ready", 15)]
    result = p.run(root, state, sha)
    assert result["status"] == "passed" and result["first_fault"] is None
    assert result["usage_complete"] and result["known_tokens"] == 15
    assert transport.calls == [(p.PLAIN_SYSTEM, p.PLAIN_USER, None, 120)]
    assert p.status(root, sha) == result


def test_source_binding_and_caps():
    hashes = p._source_hashes()
    assert hashes["transport_v4"] == "4cdd173ed84e17f03a0a44b749e575cbd401150873948310837503875d6387fc"
    assert {"transport_v1", "transport_v2", "transport_v3", "transport_v4"} <= hashes.keys()
    assert (p.MAX_CALLS, p.MAX_TOKENS, p.MAX_SECONDS, p.CALL_SECONDS) == (1, 160000, 300, 120)


def test_expired_local_admission_never_consumes_receipt(setup_probe):
    root, state, sha, catalog, transport = setup_probe
    with patch.object(catalog, "_validated_access_token", side_effect=RuntimeError("expired")):
        with pytest.raises(Exception):
            p.run(root, state, sha)
    assert not (root / "attempt.json").exists() and transport.calls == []


@pytest.mark.parametrize("media", ["sse", "missing"])
@pytest.mark.parametrize("defect,encoding", [("missing_separator", "missing"), ("none", "gzip")])
def test_wire_observation_only_on_actual_rejected_header_path(setup_probe, media, defect, encoding):
    root, state, sha, _, transport = setup_probe
    obs = {"header_defect": defect, "parsed_header_count": 2,
           "transfer_encoding": "chunked", "content_encoding": encoding,
           "body_prefix": "sse_prefix", "body_bytes": 123, "body_truncated": False,
           "sse_validation": "invalid"}
    transport.replies = [wire.TransportError("invalid_content_type", 200, "non_json", media, obs)]
    result = p.run(root, state, sha)
    assert p.status(root, sha) == result
    result["first_fault"]["wire_observation"].update(header_defect="none", content_encoding="missing")
    with pytest.raises(p.ProbeError):
        p._validate_result(result)
