"""Root-owned call-role and repeated-verifier attribution regressions."""
import json

import pytest

from benchmarks import episode_probe as probe
from hymem.dreaming import digest
from tests.digest_verification_fixtures import synthetic_summary_issues
from tests.test_episode_probe_multicall import _backend, _run, _verdict


def backend_for(monkeypatch, *, compact=False, outcome="supported"):
    primary = _backend(compact=compact)
    verifications = 0
    def backend(system, user):
        nonlocal verifications
        if system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            return json.dumps({"issues": synthetic_summary_issues(json.loads(user))})
        if system == digest._DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE.format(max_chars=500):
            if outcome == "transport":
                raise OSError("private provider transport detail")
            if outcome == "malformed":
                return "not a summary object"
            if outcome == "second_input_cap":
                monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", 1)
            return json.dumps({"summary": "Deployed release 2.4 in region eu-1." if outcome == "unchanged"
                               else "Release 2.4 was deployed in region eu-1."})
        if system == digest._DIGEST_FIDELITY_SYSTEM:
            verifications += 1
            value = json.loads(_verdict(user))
            value["summary_content"][0]["verdict"] = (
                "unsupported" if verifications == 1 or outcome == "second_rejection" else "supported")
            return json.dumps(value)
        if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            return json.dumps({"summary_format": [{"index": 0, "verdict": "unsupported" if outcome == "format_rejection" else "supported"}],
                               "episode_format": [{"index": 0, "verdict": "supported"}]})
        return primary(system, user)
    return backend


@pytest.mark.parametrize("compact", [False, True])
def test_root_probe_records_new_role_without_changing_primary(tmp_path, monkeypatch, compact):
    (row,), client = _run(tmp_path, backend_for(monkeypatch, compact=compact))
    assert not row["digest_failed"]
    assert [r["stage"] for r in row["completion_records"]] == [
        "primary", *(["summary_compaction"] if compact else []),
        "fidelity_verification", "summary_diagnosis", "summary_content_recovery", "fidelity_reverification",
        "format_adjudication",
    ]
    assert row["calls"] == 6 + compact
    assert row["extractor_input"] == client.sent[0]["user"]
    assert row["reply_chars"] == client.sent[0]["reply_chars"]
    recovery_input = json.loads(row["completion_records"][-3]["user"])
    assert recovery_input["original_generation_input"] == client.sent[0]["user"]
    probe.assert_full_source(row)


def test_root_second_verifier_input_cap_has_no_borrowed_first_response(tmp_path, monkeypatch):
    (row,), client = _run(tmp_path, backend_for(monkeypatch, outcome="second_input_cap"))
    assert row["digest_failed"] and row["calls"] == 4
    assert row["failure_reason"] == "fidelity_input_cap"
    assert row["failure_stage"] == "fidelity_reverification"
    assert row["failure_reply_chars"] is row["failure_reply_head"] is None
    assert client.sent[1]["reply_chars"] > 0


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("outcome,stage,calls", [
    ("malformed", "summary_content_recovery", 3),
    ("transport", "summary_content_recovery", 3),
    ("unchanged", "summary_content_recovery", 3),
    ("second_rejection", "fidelity_reverification", 4),
    ("format_rejection", "format_adjudication", 5),
])
def test_root_probe_names_actual_failed_invocation(tmp_path, monkeypatch, compact, outcome, stage, calls):
    (row,), client = _run(tmp_path, backend_for(monkeypatch, compact=compact, outcome=outcome))
    assert row["digest_failed"] and row["calls"] == calls + compact + 1
    assert row["failure_stage"] == stage
    assert row["failure_reply_chars"] == client.sent[-1]["reply_chars"]
    assert "private provider transport detail" not in json.dumps(row)
    stats = probe.summarize([row], [], faithfulness=None)
    assert stats["attempted_session_digests"] == 1
    assert stats["digest_failure_rate"] == 1


def test_root_probe_maximum_seven_call_success_counts_one_session(tmp_path, monkeypatch):
    (row,), _ = _run(tmp_path, backend_for(monkeypatch, compact=True, outcome="format_supported"))
    assert not row["digest_failed"] and row["calls"] == 7
    assert [r["stage"] for r in row["completion_records"]] == [
        "primary", "summary_compaction", "fidelity_verification",
        "summary_diagnosis", "summary_content_recovery", "fidelity_reverification", "format_adjudication",
    ]
    stats = probe.summarize([row], [], faithfulness=None)
    assert stats["attempted_session_digests"] == 1 and stats["calls"] == 7
    assert stats["digest_failure_rate"] == 0
