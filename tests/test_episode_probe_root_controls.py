"""Independent multicall probe controls using the real digest and stored source."""
from copy import deepcopy
import hashlib
import json

import pytest

from hymem.config import HyMemConfig
from hymem.dreaming import digest
from tests.digest_verification_fixtures import synthetic_fidelity_result, synthetic_format_result
from tests.test_episode_probe import _entry, _row, episode_probe_module as probe


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("verification", ["accepted", "unsupported", "malformed_diagnosis", "transport"])
def test_root_probe_keeps_primary_bytes_and_actual_failure_stage(tmp_path, repair, verification):
    sent = []
    final_summary = "The rollout was discussed and its settings recorded."

    def backend(system, user):
        sent.append((system, user))
        if system == digest._DIGEST_FIDELITY_SYSTEM:
            if verification == "transport":
                raise RuntimeError("PRIVATE_EXCEPTION_SENTINEL /private/example/key")
            verdicts = synthetic_fidelity_result()
            if verification in {"unsupported", "malformed_diagnosis"}:
                verdicts["summary_content"][0]["verdict"] = "unsupported"
            return json.dumps(verdicts)
        if system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            if verification == "malformed_diagnosis":
                # A candidate-shaped reply cannot substitute for bounded hints.
                return json.dumps({"summary": final_summary})
            return json.dumps({"issues": []})
        if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            return json.dumps(synthetic_format_result())
        if system.startswith("You compact one rolling conversation summary"):
            return json.dumps({"summary": final_summary})
        return json.dumps({"episodes": [], "procedures": [],
                           "summary": "Overlong candidate. " * 30 if repair else final_summary})

    config = HyMemConfig(root=tmp_path)
    entry = _entry("root-probe", "target", turns=2)
    conn = probe.build_store(tmp_path / "probe.sqlite", [entry], config)
    try:
        client = probe.CapturingLLM(backend)
        row = probe.extract_one(conn, entry, client, config, granular=True)
    finally:
        conn.close()
    # A rejection without source-linked findings is held without a generic reroll.
    assert row["calls"] == len(sent) == 2 + repair + (verification != "transport")
    verification_input = next(user for system, user in sent if system == digest._DIGEST_FIDELITY_SYSTEM)
    assert row["extractor_input"] == sent[0][1] and row["extractor_input"] != verification_input
    assert row["extractor_input_sha256"] == hashlib.sha256(sent[0][1].encode()).hexdigest()
    assert row["extractor_input_chars"] == len(sent[0][1])
    assert "[chunk msgcov_" in row["extractor_input"]
    assert row["extractor_input"].rstrip().endswith("Return the granular digest JSON object now.")
    probe.assert_full_source(row)
    stages = [record["stage"] for record in row["completion_records"]]
    assert stages == (["primary", *(["summary_compaction"] if repair else []), "fidelity_verification"]
                      + (["format_adjudication"] if verification == "accepted" else
                         ["summary_diagnosis"] if verification != "transport" else []))
    assert row["digest_attempted"] is True
    assert row["digest_failed"] is (verification != "accepted")
    if verification == "accepted":
        assert row["failure_stage"] is None and row["failure_reason"] is None
    else:
        assert row["failure_stage"] == ("fidelity_verification" if verification == "transport" else "summary_diagnosis")
        assert row["episodes"] == []
        assert row["failure_reason"] == {
            "unsupported": "summary_diagnosis_unactionable",
            "malformed_diagnosis": "summary_diagnosis_shape_failure",
            "transport": "execution_failure:DigestCompletionError",
        }[verification]
        if verification == "transport":
            assert row["failure_reply_chars"] is None
            assert row["backend_error"] == "execution_failure:RuntimeError"
            assert row["parse_failed"] is False
        else:
            diagnosis_record = row["completion_records"][-1]
            assert diagnosis_record["stage"] == "summary_diagnosis"
            assert row["failure_reply_chars"] == len(diagnosis_record["reply"]) > 0
            assert row["failure_reply_head"] == diagnosis_record["reply"][:240]
            assert row["parse_failed"] is True
    assert "PRIVATE_EXCEPTION_SENTINEL" not in json.dumps(row)
    assert row["reply_chars"] > 0  # A verifier failure must not replace the primary reply.


def test_root_probe_reused_client_cannot_reuse_previous_error_or_request(tmp_path):
    calls = []

    def backend(system, user):
        calls.append((system, user))
        if len(calls) == 1:
            raise RuntimeError("Private old failure")
        if system == digest._DIGEST_FIDELITY_SYSTEM:
            return json.dumps(synthetic_fidelity_result())
        if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            return json.dumps(synthetic_format_result())
        return json.dumps({"episodes": [], "summary": "The new session was discussed.", "procedures": []})

    entries = [_entry("old-probe", "target", turns=1), _entry("new-probe", "control", turns=1)]
    config = HyMemConfig(root=tmp_path)
    conn = probe.build_store(tmp_path / "probe.sqlite", entries, config)
    try:
        client = probe.CapturingLLM(backend)
        first = probe.extract_one(conn, entries[0], client, config, granular=False)
        saved_first = deepcopy(first)
        second = probe.extract_one(conn, entries[1], client, config, granular=False)
    finally:
        conn.close()
    assert first == saved_first and first["digest_failed"] is True and first["calls"] == 1
    assert second["digest_failed"] is False and second["backend_error"] is None
    assert second["calls"] == len(second["completion_records"]) == 3
    assert second["extractor_input"] == calls[1][1] != calls[0][1]
    assert second["failure_stage"] is None and second["failure_reason"] is None


@pytest.mark.parametrize("calls_per_attempt", [1, 2, 3, 12])
@pytest.mark.parametrize("failure_kind", ["parse", "execution"])
def test_root_extra_completions_cannot_dilute_the_frozen_failure_threshold(calls_per_attempt, failure_kind):
    rows = [_row("target" if index < 25 else "control", episodes=5 if index < 25 else 1,
                 calls=calls_per_attempt) for index in range(50)]
    for index, row in enumerate(rows):
        row["session_id"] = f"root-metric-{index}"
    for row in rows[-2:]:
        row["episodes"] = []
        if failure_kind == "parse":
            row["parse_failed"] = True
        else:
            row["error"] = "execution_failure:RuntimeError"
            # A contradictory optional false flag cannot erase observable error evidence.
            row["digest_failed"] = False
    summary = probe.summarize(rows[:25], rows[25:], 0.99)
    assert summary["calls"] == 50 * calls_per_attempt
    assert summary["attempted_session_digests"] == 50
    assert summary["digest_failures"] == 2 and summary["digest_failure_rate"] == pytest.approx(0.04)
    assert probe._MAX_PARSE_FAILURE_RATE == 0.02
    assert summary["gate"]["parse_failures_ok"] is False
    assert summary["verdict"].startswith("INCOMPLETE")
