"""Offline controls for source-linked recovery and repeated-verifier accounting."""
import hashlib
import json

import pytest

from benchmarks import episode_probe as probe
from hymem.dreaming import digest
from hymem.extraction.llm import LLMRequest
from tests.digest_verification_fixtures import synthetic_summary_issues
from tests.test_episode_probe_multicall import _backend, _run, _verdict


_CONTENT_SYSTEM = digest._DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE.format(max_chars=500)


def _content_backend(monkeypatch, *, compact=False, outcome="supported"):
    primary = _backend(compact=compact)
    verifications = 0

    def backend(system, user):
        nonlocal verifications
        if system == _CONTENT_SYSTEM:
            if outcome == "recovery_transport":
                raise OSError("PRIVATE_RECOVERY_TRANSPORT")
            if outcome == "recovery_malformed":
                return "invalid recovery JSON"
            if outcome == "verification_input_cap":
                monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", 1)
            return json.dumps({"summary": "Release 2.4 was deployed in region eu-1."})
        if system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            return json.dumps({"issues": synthetic_summary_issues(json.loads(user))})
        if system == digest._DIGEST_FIDELITY_SYSTEM:
            verifications += 1
            if verifications == 2 and outcome == "verification_transport":
                raise OSError("PRIVATE_SECOND_VERIFIER_TRANSPORT")
            if verifications == 2 and outcome == "verification_malformed":
                return "invalid second verifier JSON"
            value = json.loads(_verdict(user))
            if verifications == 1:
                value["summary_content"][0]["verdict"] = "unsupported"
                if outcome == "recovery_input_cap":
                    monkeypatch.setattr(digest, "_DIGEST_SUMMARY_CONTENT_RECOVERY_MAX_INPUT_CHARS", 1)
            else:
                if outcome == "format_input_cap":
                    monkeypatch.setattr(digest, "_DIGEST_FORMAT_ADJUDICATION_MAX_INPUT_CHARS", 1)
            return json.dumps(value)
        if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            if outcome == "format_transport":
                raise OSError("PRIVATE_FORMAT_TRANSPORT")
            if outcome == "format_malformed":
                return "invalid format JSON"
            value = json.loads(probe.sim_backend(system, user))
            if outcome == "format_unsupported":
                value["summary_format"][0]["verdict"] = "unsupported"
            return json.dumps(value)
        return primary(system, user)

    return backend


@pytest.mark.parametrize("compact", [False, True])
def test_combined_content_recovery_and_format_adjudication_keep_all_exact_calls(
    tmp_path, monkeypatch, compact,
):
    (row,), client = _run(tmp_path, _content_backend(monkeypatch, compact=compact))
    records = row["completion_records"]
    assert not row["digest_failed"] and row["episodes"]
    assert row["record_version"] == "episode-probe-multicall-v5"
    assert row["calls"] == client.calls == len(records) == 6 + compact
    assert [r["stage"] for r in records] == [
        "primary", *(["summary_compaction"] if compact else []),
        "fidelity_verification", "summary_diagnosis", "summary_content_recovery", "fidelity_reverification",
        "format_adjudication",
    ]
    recovery_input = json.loads(records[-3]["user"])
    assert row["extractor_input"] == records[0]["user"] == recovery_input["original_generation_input"]
    assert row["reply_chars"] == records[0]["reply_chars"]
    assert row["reply_head"] == records[0]["reply_head"]
    assert row["failure_reply_chars"] is row["failure_reply_head"] is None
    first = json.loads(records[-5]["user"])
    repeated = json.loads(records[-2]["user"])
    assert recovery_input["rejection_diagnostics"]["candidate_summary"] == first["summary_item"]["candidate_summary"]
    assert recovery_input["rejection_diagnostics"]["issues"] == json.loads(records[-4]["reply"])["issues"]
    assert first["summary_item"]["candidate_summary"] != repeated["summary_item"]["candidate_summary"]
    for record in records:
        assert record["user_sha256"] == hashlib.sha256(record["user"].encode()).hexdigest()
        assert record["reply_sha256"] == hashlib.sha256(record["reply"].encode()).hexdigest()
        assert record["user_chars"] == len(record["user"])
        assert record["reply_chars"] == len(record["reply"])
    stats = probe.summarize([row], [], None)
    assert stats["calls"] == 6 + compact
    assert stats["attempted_session_digests"] == 1 and stats["digest_failures"] == 0
    probe.assert_full_source(row)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("outcome,stage,calls,reply_exists", [
    ("recovery_transport", "summary_content_recovery", 3, False),
    ("recovery_malformed", "summary_content_recovery", 3, True),
    ("recovery_input_cap", "summary_content_recovery", 2, False),
    ("verification_transport", "fidelity_reverification", 4, False),
    ("verification_malformed", "fidelity_reverification", 4, True),
    ("verification_input_cap", "fidelity_reverification", 3, False),
    ("format_transport", "format_adjudication", 5, False),
    ("format_malformed", "format_adjudication", 5, True),
    ("format_unsupported", "format_adjudication", 5, True),
    ("format_input_cap", "format_adjudication", 4, False),
])
def test_failure_reply_belongs_only_to_the_actual_final_invocation(
    tmp_path, monkeypatch, compact, outcome, stage, calls, reply_exists,
):
    (row,), client = _run(
        tmp_path, _content_backend(monkeypatch, compact=compact, outcome=outcome),
    )
    assert row["digest_failed"] and not row["episodes"]
    assert row["calls"] == calls + compact + 1 == len(row["completion_records"])
    assert row["failure_stage"] == stage
    if reply_exists:
        assert row["failure_reply_chars"] == client.sent[-1]["reply_chars"] > 0
        assert row["failure_reply_head"] == client.sent[-1]["reply_head"]
    else:
        assert row["failure_reply_chars"] is row["failure_reply_head"] is None
    assert row["extractor_input"] == client.sent[0]["user"]
    assert row["reply_chars"] == client.sent[0]["reply_chars"] > 0
    assert "PRIVATE_" not in json.dumps(row)
    if outcome == "verification_input_cap":
        assert row["failure_reason"] == "fidelity_input_cap"
        assert client.sent[-1]["stage"] == "summary_content_recovery"
        assert client.sent[-2]["stage"] == "summary_diagnosis"
        assert client.sent[-2]["reply_chars"] > 0  # Not the failed verifier's reply.
    if outcome == "recovery_input_cap":
        assert row["failure_reason"] == "summary_content_recovery_input_cap"
        assert client.sent[-1]["stage"] == "summary_diagnosis"
        assert client.sent[-1]["reply_chars"] > 0  # No recovery invocation exists.
    stats = probe.summarize([row], [], None)
    assert stats["attempted_session_digests"] == stats["digest_failures"] == 1
    assert stats["digest_failure_rate"] == 1.0
    assert stats["failure_replies_recorded"] == int(reply_exists)
    probe.assert_full_source(row)


@pytest.mark.parametrize("system,expected", [
    (_CONTENT_SYSTEM, "summary_content_recovery"),
    (_CONTENT_SYSTEM + " ", "unknown"),
    (digest._DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE.format(max_chars=501), "unknown"),
    ("User prose mentioning summary_content_recovery", "unknown"),
])
def test_content_recovery_requires_the_exact_production_task(system, expected):
    assert probe._request_stage(LLMRequest(system=system, user=_CONTENT_SYSTEM)) == expected


def test_unattributed_exception_after_content_recovery_stays_unknown(tmp_path, monkeypatch):
    def broken(conn, sid, client, **kwargs):
        client.complete(LLMRequest(system=_CONTENT_SYSTEM, user="Synthetic source"))
        raise RuntimeError("PRIVATE_POSTPROCESS_ERROR")
    monkeypatch.setattr(probe, "extract_session_digest", broken)
    (row,), _ = _run(tmp_path, lambda system, user: '{"summary":"Synthetic summary."}')
    assert row["calls"] == 1 and row["digest_failed"]
    assert row["completion_records"][0]["stage"] == "summary_content_recovery"
    assert row["failure_stage"] == "unknown"
    assert row["failure_reply_chars"] is row["failure_reply_head"] is None
    assert "PRIVATE_POSTPROCESS_ERROR" not in json.dumps(row)
