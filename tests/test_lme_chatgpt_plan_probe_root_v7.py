"""Independent structured-only readiness controls; no account/network access."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from tools.diagnostics import lme_chatgpt_plan_probe_v7 as p
from tests.test_lme_chatgpt_plan_probe_v6 import FakeCatalog, FakeTransport


@pytest.fixture
def prepared():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root, state = Path(directory) / "run", Path(directory) / "state"
        root.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        catalog, wire = FakeCatalog(), FakeTransport()
        with patch.object(p, "_modules", return_value=(catalog, wire)):
            digest = p.prepare(root, state)["receipt_sha256"]
            yield root, state, digest, wire, catalog


def test_actual_pinned_import_and_limits():
    catalog, wire = p._modules()
    assert wire.MODEL == "gpt-5.6-luna"
    assert p._source_hashes()["transport_v6"] == "811bff13ebc4b24ebd22cad16542c3b58597dc04538a1d7deb5085df7190a28f"
    assert (p.MAX_CALLS, p.MAX_TOKENS, p.MAX_SECONDS, p.CALL_SECONDS) == (1, 160000, 300, 120)
    result = subprocess.run([sys.executable, "-I", "-B", "-c",
        "import sys;sys.path.insert(0," + repr(str(p.ROOT)) + ");"
        "from tools.diagnostics import lme_chatgpt_plan_probe_v7 as p; p._modules()"],
        capture_output=True, timeout=15)
    assert result.returncode == 0


def test_structured_only_once_usage_and_replay(prepared):
    root, state, digest, wire, catalog = prepared
    assert catalog.credential_reads == 0
    wire.replies = [('{"ready":true}', 13)]
    result = p.run(root, state, digest)
    assert result["status"] == "passed"
    assert result["known_tokens"] == 13 and result["usage_complete"]
    assert result["child_cleanup"] == "verified"
    assert p.status(root, digest) == result
    assert wire.calls == [(p.STRUCTURED_SYSTEM, p.STRUCTURED_USER, p.SCHEMA, 120)]
    with pytest.raises(Exception):
        p.run(root, state, digest)
    assert len(wire.calls) == 1


@pytest.mark.parametrize("output", [
    '{"ready":1}', '{"ready":false}', '{"ready":true,"extra":"PRIVATE"}',
    '{"ready":true,"ready":true}', '[true]', 'true', 'null',
    '{"ready":NaN}', '```json\n{"ready":true}\n```',
])
def test_invalid_structured_output_retains_completed_usage(prepared, output):
    root, state, digest, wire, _ = prepared
    wire.replies = [(output, 13)]
    result = p.run(root, state, digest)
    assert result["status"] == "failed" and result["usage_complete"]
    assert result["known_tokens"] == 13
    assert result["first_fault"]["code"] in p.OUTPUT_CODES
    assert p.status(root, digest) == result
    assert "PRIVATE" not in (root / "result.json").read_text()
    assert not any(key in result for key in ("output", "text", "raw_output"))


@pytest.mark.parametrize("where,value", [("calls", True), ("schema", 0)])
def test_receipt_boolean_numeric_confusion_rejected(prepared, where, value):
    root, state, _, wire, _ = prepared
    receipt = json.loads((root / "receipt.json").read_bytes())
    if where == "calls":
        receipt["limits"]["calls"] = value
    else:
        receipt["fixtures"]["schema"]["additionalProperties"] = value
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "receipt.json").write_bytes(raw)
    with pytest.raises(Exception):
        p.run(root, state, hashlib.sha256(raw).hexdigest())
    assert not (root / "attempt.json").exists() and wire.calls == []


def test_explicit_denial_one_request_unknown_usage(prepared):
    root, state, digest, wire, _ = prepared
    wire.replies = [FakeTransport.TransportError("subscription_sharing_usage_limit_exceeded", 429)]
    result = p.run(root, state, digest)
    assert result["status"] == "failed" and not result["usage_complete"]
    assert result["known_usage"] == [] and len(wire.calls) == 1
    assert result["first_fault"]["code"] == "subscription_sharing_usage_limit_exceeded"
    assert p.status(root, digest) == result


def test_reader_rejects_private_fields_or_old_probe_fault(prepared):
    root, state, digest, wire, _ = prepared
    wire.replies = [('{"ready":false}', 13)]
    result = p.run(root, state, digest)
    for changed in (dict(result, raw_output="PRIVATE"),
                    dict(result, first_fault=dict(result["first_fault"], code="plain_output_mismatch"))):
        (root / "result.json").write_text(json.dumps(changed))
        with pytest.raises(Exception):
            p.status(root, digest)
