"""Independent root controls: invented credentials/responses, no provider calls."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from tools.diagnostics import lme_chatgpt_plan_probe_v1 as p
from tests.test_lme_chatgpt_plan_probe_v1 import FakeCatalog, FakeTransport


@pytest.fixture
def setup_probe():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
        root, state = Path(temp) / "run", Path(temp) / "state"
        root.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        catalog, transport = FakeCatalog(), FakeTransport()
        with patch.object(p, "_modules", return_value=(catalog, transport)):
            sha = p.prepare(root, state)["receipt_sha256"]
            yield root, state, sha, catalog, transport


@pytest.mark.parametrize("text", ['{"ready":1}', '{"ready":1.0}', '{"ready":false}',
                                  '{"ready":true,"extra":null}', '{"ready":true,"ready":true}'])
def test_exact_schema_boolean(text):
    with pytest.raises(p.ProbeError):
        p._check_output(2, text)


def test_output_failure_keeps_known_complete_usage(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("incorrect invented output", 13)]
    result = p.run(root, state, sha)
    assert result["status"] == "failed"
    assert result["known_tokens"] == 13 and result["usage_complete"] is True
    assert len(transport.calls) == 1


def test_second_failure_retains_first_known_usage(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("blue paper kite is ready", 17), FakeTransport.TransportError("missing_usage")]
    result = p.run(root, state, sha)
    assert result["known_tokens"] == 17 and result["usage_complete"] is False
    assert result["model_calls"] == 2
    assert result["first_fault"]["phase"] == "call_2"
    assert p.status(root, sha) == result


def test_existing_attempt_never_calls_transport(setup_probe):
    root, state, sha, _, transport = setup_probe
    fd = p._private_dir(root)
    try:
        p._create(fd, "attempt.json", {"receipt_sha256": sha, "started_at": 1})
    finally:
        os.close(fd)
    with pytest.raises(p.ProbeError):
        p.run(root, state, sha)
    assert transport.calls == []


def test_success_requires_every_call_accounted(setup_probe):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("blue paper kite is ready", 17), ('{"ready":true}', 19)]
    result = p.run(root, state, sha)
    result["known_usage"].pop()
    result["known_tokens"] = 17
    result["usage_complete"] = False
    with pytest.raises(p.ProbeError):
        p._validate_result(result)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(private_text="DO_NOT_EXPORT"),
    lambda r: r.update(policy="DO_NOT_EXPORT"),
    lambda r: r.update(elapsed_seconds=float("nan")),
    lambda r: r.update(model_calls=True),
    lambda r: r.update(usage_complete=False),
    lambda r: r.update(first_fault={"code": "DO_NOT_EXPORT", "phase": "call_1", "http_status": None, "body_shape": None}),
])
def test_status_rejects_arbitrary_or_inconsistent_metadata(setup_probe, mutate):
    root, state, sha, _, transport = setup_probe
    transport.replies = [("blue paper kite is ready", 17), ('{"ready":true}', 19)]
    p.run(root, state, sha)
    path = root / "result.json"
    result = json.loads(path.read_text())
    mutate(result)
    path.write_text(json.dumps(result))
    with pytest.raises(p.ProbeError):
        p.status(root, sha)


def test_symlink_state_rejected(setup_probe):
    root, state, _, _, _ = setup_probe
    link = state.parent / "symlink"
    link.symlink_to(state, target_is_directory=True)
    with pytest.raises(p.ProbeError):
        p._private_dir(link)


def test_nonprivate_directory_rejected(setup_probe):
    root, _, _, _, _ = setup_probe
    root.chmod(0o755)
    with pytest.raises(p.ProbeError):
        p._private_dir(root)
