"""Independent controls for the probe's new fourth-call accounting."""
import json

import pytest

from benchmarks import episode_probe as probe
from hymem.dreaming import digest
from tests.test_episode_probe_multicall import _backend, _run, _verdict


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("outcome", ["supported", "unsupported", "malformed", "transport", "input_cap"])
def test_root_probe_distinguishes_adjudication_and_preserves_primary(tmp_path, monkeypatch, compact, outcome):
    original = _backend(compact=compact)
    def backend(system, user):
        if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            if outcome == "transport":
                raise OSError("private provider details")
            if outcome == "malformed":
                return "not a valid adjudication"
            return json.dumps({
                "summary_format": [{"index": 0, "verdict": outcome}],
                "episode_format": [{"index": 0, "verdict": "supported"}],
            })
        if system == digest._DIGEST_FIDELITY_SYSTEM:
            return _verdict(user)
        return original(system, user)
    if outcome == "input_cap":
        monkeypatch.setattr(digest, "_DIGEST_FORMAT_ADJUDICATION_MAX_INPUT_CHARS", 1)
    (row,), client = _run(tmp_path, backend)
    count = (3 if compact else 2) + int(outcome != "input_cap")
    assert row["calls"] == len(row["completion_records"]) == count
    assert row["extractor_input"] == client.sent[0]["user"]
    assert row["reply_chars"] == client.sent[0]["reply_chars"]
    assert row["digest_failed"] is (outcome != "supported")
    if outcome != "input_cap":
        assert row["completion_records"][-1]["stage"] == "format_adjudication"
    if outcome == "supported":
        assert row["failure_stage"] is None and row["episodes"]
    else:
        assert row["failure_stage"] == "format_adjudication"
        assert not row["episodes"]
        if outcome in {"input_cap", "transport"}:
            assert row["failure_reply_chars"] is None
        else:
            assert row["failure_reply_chars"] == client.sent[-1]["reply_chars"]
    assert "private provider details" not in json.dumps(row)
    probe.assert_full_source(row)
