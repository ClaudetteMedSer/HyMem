"""Independent v6 probe controls, no network or private state."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
from benchmarks import chatgpt_plan_responses_v6 as wire
from tools.diagnostics import lme_chatgpt_plan_probe_v6 as p
from tools.diagnostics import lme_chatgpt_plan_probe_v5 as old
from tests.test_lme_chatgpt_plan_probe_v1 import FakeCatalog, FakeTransport
from tests.test_lme_chatgpt_plan_probe_root_v5 import observation


class Transport(FakeTransport):
    TransportError = wire.TransportError
    _v3 = wire._v3
    _sanitize_stream_observation = staticmethod(wire._sanitize_stream_observation)


@pytest.fixture
def setup_probe():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
        root, state = Path(name)/"run", Path(name)/"state"
        root.mkdir(mode=0o700); state.mkdir(mode=0o700)
        catalog, transport = FakeCatalog(), Transport()
        with patch.object(p, "_modules", return_value=(catalog,transport)):
            sha = p.prepare(root,state)["receipt_sha256"]
            yield root,state,sha,transport


def test_only_new_empty_reconstruction_state_added():
    obs = {**observation(), "terminal_output_state": "empty"}
    assert not old._valid_stream_observation(obs)
    assert p._valid_stream_observation(obs)
    for change in ({"finalized_item_count":0}, {"finalized_item_count":True},
                   {"terminal_output_state":"invalid"}, {"private_text":"PRIVATE_SENTINEL"}):
        assert not p._valid_stream_observation({**obs,**change})


def test_source_and_limits():
    assert p._source_hashes()["transport_v6"] == "811bff13ebc4b24ebd22cad16542c3b58597dc04538a1d7deb5085df7190a28f"
    assert (p.MAX_CALLS,p.MAX_TOKENS,p.MAX_SECONDS,p.CALL_SECONDS) == (1,160000,300,120)


def test_success_exactly_one_fixture_and_no_replay(setup_probe):
    root,state,sha,transport = setup_probe
    transport.replies = [("blue paper kite is ready",15)]
    result = p.run(root,state,sha)
    assert result["status"] == "passed" and result["usage_complete"] and result["known_tokens"] == 15
    assert p.status(root,sha) == result
    with pytest.raises(p.ProbeError): p.run(root,state,sha)
    assert transport.calls == [(p.PLAIN_SYSTEM,p.PLAIN_USER,None,120)]


def test_failed_reconstruction_metadata_preserved_not_success(setup_probe):
    root,state,sha,transport = setup_probe
    obs = {**observation(), "terminal_output_state":"empty"}
    transport.replies = [wire.TransportError("missing_usage",200,"sse_event","missing",stream_observation=obs)]
    result = p.run(root,state,sha)
    assert result["status"] == "failed" and not result["usage_complete"] and result["known_usage"] == []
    assert result["first_fault"]["stream_observation"] == obs
    assert p.status(root,sha) == result
    result["first_fault"]["stream_observation"]["private_text"] = "PRIVATE_SENTINEL"
    (root/"result.json").write_text(json.dumps(result))
    with pytest.raises(p.ProbeError) as caught: p.status(root,sha)
    assert "PRIVATE_SENTINEL" not in str(caught.value)
