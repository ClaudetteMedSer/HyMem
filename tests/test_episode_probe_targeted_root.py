"""Independent recorder controls; scripted replies are not accuracy evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks import episode_probe as probe
from hymem.dreaming import digest
from tests.digest_verification_fixtures import synthetic_summary_issues
from tests.test_episode_probe_multicall import _candidate, _run


def _root_backend(monkeypatch, *, repair=False, compact=False, cap=None,
                  semantic="supported", grammar="supported"):
    verifications = 0

    def backend(system, user):
        nonlocal verifications
        if system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            return json.dumps({"issues": synthetic_summary_issues(json.loads(user)) if repair else []})
        if system == digest._DIGEST_FIDELITY_SYSTEM:
            verifications += 1
            payload = json.loads(user)
            value = {
                family: [{"index": i, "verdict": "supported"} for i in range(count)]
                for family, count in [("episode_titles", len(payload["items"])),
                                      ("episode_content", len(payload["items"])),
                                      ("procedures", len(payload["procedure_items"])),
                                      ("summary_content", 1)]
            }
            if repair and verifications == 1:
                item = payload["summary_item"]
                source = next(row for row in payload["source_catalog"]
                              if row["chunk_id"] == item["new_source_ids"][0])
                value["summary_content"][0]["verdict"] = "unsupported"
                if cap == "repair":
                    monkeypatch.setattr(digest, "_DIGEST_SUMMARY_CONTENT_RECOVERY_MAX_INPUT_CHARS", 1)
            else:
                value["summary_content"][0]["verdict"] = semantic
            if cap == "format":
                monkeypatch.setattr(digest, "_DIGEST_FORMAT_ADJUDICATION_MAX_INPUT_CHARS", 1)
            return json.dumps(value)
        if system == digest._DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE.format(max_chars=500):
            if cap == "reverification":
                monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", 1)
            return json.dumps({"summary": "Release 2.4 was deployed in region eu-1."})
        if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            payload = json.loads(user)
            return json.dumps({
                "summary_format": [{"index": 0, "verdict": grammar}],
                "episode_format": [{"index": i, "verdict": "supported"}
                                   for i in range(len(payload["items"]))],
            })
        if probe._RECOVERY_SYSTEM_PATTERN.fullmatch(system):
            return json.dumps({"summary": "Deployed release 2.4 in region eu-1."})
        return _candidate(user, compact=compact)

    return backend


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("repair", [False, True])
def test_root_recorder_exact_roles_hashes_and_one_session_denominator(tmp_path, monkeypatch, compact, repair):
    (row,), client = _run(tmp_path, _root_backend(monkeypatch, repair=repair, compact=compact))
    assert not row["digest_failed"]
    assert row["record_version"] == "episode-probe-multicall-v5"
    assert [r["stage"] for r in row["completion_records"]] == [
        "primary", *(["summary_compaction"] if compact else []),
        "fidelity_verification", *(["summary_diagnosis", "summary_content_recovery", "fidelity_reverification"] if repair else []),
        "format_adjudication",
    ]
    assert row["calls"] == 3 + compact + 3 * repair
    for record, sent in zip(row["completion_records"], client.sent):
        assert record == sent
        assert record["user_sha256"] == hashlib.sha256(record["user"].encode()).hexdigest()
        assert record["reply_sha256"] == hashlib.sha256(record["reply"].encode()).hexdigest()
    if repair:
        recovery = next(r for r in client.sent if r["stage"] == "summary_content_recovery")
        assert json.loads(recovery["user"])["original_generation_input"] == client.sent[0]["user"]
    snapshot = deepcopy(row)
    stats = probe.summarize([row], [], faithfulness=None)
    assert row == snapshot
    assert stats["attempted_session_digests"] == 1
    assert stats["calls"] == row["calls"] and stats["digest_failure_rate"] == 0
    probe.assert_full_source(row)


@pytest.mark.parametrize("cap,stage,calls", [
    ("repair", "summary_content_recovery", 2),
    ("reverification", "fidelity_reverification", 3),
    ("format", "format_adjudication", 4),
])
def test_root_predispatch_caps_never_borrow_a_prior_reply(tmp_path, monkeypatch, cap, stage, calls):
    (row,), _ = _run(tmp_path, _root_backend(monkeypatch, repair=True, cap=cap))
    assert row["digest_failed"] and row["calls"] == calls + 1
    assert row["failure_stage"] == stage
    assert row["failure_reply_chars"] is row["failure_reply_head"] is None
    assert row["completion_records"][-1]["reply_chars"] > 0
    assert not row["episodes"]


@pytest.mark.parametrize("semantic,grammar,calls,stage", [
    ("uncertain", "supported", 3, "summary_diagnosis"),
    ("supported", "unsupported", 3, "format_adjudication"),
])
def test_root_final_veto_counts_failure_not_partial_success(tmp_path, monkeypatch, semantic, grammar, calls, stage):
    (row,), _ = _run(tmp_path, _root_backend(monkeypatch, semantic=semantic, grammar=grammar))
    assert row["digest_failed"] and row["calls"] == calls and row["failure_stage"] == stage
    assert not row["episodes"]
    stats = probe.summarize([row], [], faithfulness=None)
    assert stats["attempted_session_digests"] == 1 and stats["digest_failure_rate"] == 1


def test_root_simulation_follows_current_three_call_contract(tmp_path):
    (row,), _ = _run(tmp_path, probe.sim_backend)
    assert not row["digest_failed"] and row["calls"] == 3
    assert [r["stage"] for r in row["completion_records"]] == [
        "primary", "fidelity_verification", "format_adjudication",
    ]
