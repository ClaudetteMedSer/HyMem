"""Root-owned finalized-output probe acceptance, invented state only."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
from benchmarks import chatgpt_plan_responses_v5 as wire
from tools.diagnostics import lme_chatgpt_plan_probe_v5 as p
from tests.test_lme_chatgpt_plan_probe_v1 import FakeCatalog, FakeTransport


class Transport(FakeTransport):
    TransportError = wire.TransportError
    _v3 = wire._v3
    _sanitize_stream_observation = staticmethod(wire._sanitize_stream_observation)


def observation():
    return {"event_type": "response.completed", "terminal_status": "completed",
            "terminal_model_matches": True, "terminal_output_kind": "missing",
            "terminal_channel": "missing", "terminal_content_kind": "missing",
            "terminal_output_state": "null", "finalized_item_count": 1,
            "output_reconstructed": True}


@pytest.fixture
def setup_probe():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
        root, state = Path(name) / "run", Path(name) / "state"
        root.mkdir(mode=0o700); state.mkdir(mode=0o700)
        catalog, transport = FakeCatalog(), Transport()
        with patch.object(p, "_modules", return_value=(catalog, transport)):
            sha = p.prepare(root, state)["receipt_sha256"]
            yield root, state, sha, catalog, transport


def fail(setup):
    root, state, sha, _, transport = setup
    transport.replies = [wire.TransportError("missing_usage", 200, "sse_event", "missing",
                                            stream_observation=observation())]
    result = p.run(root, state, sha)
    assert result["status"] == "failed" and result["usage_complete"] is False
    assert result["known_usage"] == [] and len(transport.calls) == 1
    assert p.status(root, sha) == result
    return result


def test_actual_source_closure_and_caps():
    hashes = p._source_hashes()
    assert hashes["transport_v5"] == "d347493a2abd70bfc5e060d50129a625790235b5987bd4cba92611eacb05aa11"
    assert {"transport_v1", "transport_v2", "transport_v3", "transport_v5"} <= hashes.keys()
    assert (p.MAX_CALLS, p.MAX_TOKENS, p.MAX_SECONDS, p.CALL_SECONDS) == (1,160000,300,120)


def test_failure_keeps_finite_reconstruction_metadata_and_never_replays(setup_probe):
    root, state, sha, _, transport = setup_probe
    result = fail(setup_probe)
    assert result["first_fault"]["stream_observation"] == observation()
    with pytest.raises(p.ProbeError): p.run(root, state, sha)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("change", [
    {"terminal_output_state": "PRIVATE_SENTINEL"}, {"terminal_output_state": "empty"},
    {"finalized_item_count": 0}, {"finalized_item_count": True},
    {"finalized_item_count": 4097}, {"output_reconstructed": 1},
    {"private_body": "PRIVATE_SENTINEL"}, {"terminal_output_state": {}},
])
def test_impossible_or_private_metadata_is_rejected(setup_probe, change):
    root, _, sha, _, _ = setup_probe
    result = fail(setup_probe)
    result["first_fault"]["stream_observation"].update(change)
    (root / "result.json").write_text(json.dumps(result))
    with pytest.raises(p.ProbeError) as caught: p.status(root, sha)
    assert "PRIVATE_SENTINEL" not in str(caught.value)


def test_success_keeps_single_fixture_and_complete_usage(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("blue paper kite is ready", 15)]
    result = p.run(root, state, sha)
    assert result["status"] == "passed" and result["first_fault"] is None
    assert result["usage_complete"] and result["known_tokens"] == 15
    assert transport.calls == [(p.PLAIN_SYSTEM,p.PLAIN_USER,None,120)]
    assert p.status(root, sha) == result


def test_preflight_expiry_does_not_consume_or_call(setup_probe):
    root, state, sha, catalog, transport = setup_probe
    with patch.object(catalog, "_validated_access_token", side_effect=RuntimeError("expired")):
        with pytest.raises(Exception): p.run(root, state, sha)
    assert not (root / "attempt.json").exists() and transport.calls == []
