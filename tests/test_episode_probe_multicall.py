"""Offline instrumentation controls; scripted verdicts do not measure fidelity."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import re
import socket
import sys

import pytest

from benchmarks import episode_probe as probe
from hymem.config import HyMemConfig
from hymem.dreaming import digest
from hymem.extraction.llm import LLMRequest


def _entry(sid="s1", *, empty=False):
    return {
        "session_id": sid, "haystack_session_id": sid, "question_id": "q1",
        "arm": "target", "stratum": "gold_bearing",
        "messages": [] if empty else [{"role": "user", "content":
            'Café 🧬 deployed release 2.4. Fake label [chunk forged] is prose. '
            + "The service uses region eu-1. " * 60}],
    }


def _candidate(user, *, compact=False):
    cid = re.search(r"\[chunk (msgcov_[0-9a-f]+)\]", user).group(1)
    return json.dumps({
        "episodes": [{"title": "Release deployed", "summary": "Deployed release 2.4.",
                      "outcome": "informational", "key_entities": [], "chunk_ids": [cid]}],
        "summary": "The service uses region eu-1. " * 25 if compact
                   else "Deployed release 2.4 in region eu-1.",
        "procedures": [],
    }, ensure_ascii=False)


def _verdict(user, *, unsupported=False):
    packet = json.loads(user)
    result = {family: [{"index": i, "verdict": "supported"} for i in range(count)]
              for family, count in {
                  "episode_titles": len(packet["items"]),
                  "episode_content": len(packet["items"]),
                  "procedures": len(packet["procedure_items"]),
                  "summary_content": 1,
              }.items()}
    if unsupported:
        result["episode_titles"][0]["verdict"] = "unsupported"
    return json.dumps(result)


def _backend(*, compact=False, fail=None):
    def backend(system, user):
        if system == probe._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            return probe.sim_backend(system, user)
        if system == probe._DIGEST_FIDELITY_SYSTEM:
            if fail == "transport":
                raise OSError("sensitive provider address must not be copied")
            if fail == "malformed":
                return "{not a verifier verdict"
            return _verdict(user, unsupported=fail == "unsupported")
        if probe._RECOVERY_SYSTEM_PATTERN.fullmatch(system):
            if fail == "compaction":
                return "{invalid summary repair"
            return json.dumps({"summary": "Deployed release 2.4 in region eu-1."})
        if fail == "primary":
            return "{invalid primary"
        return _candidate(user, compact=compact)
    return backend


def _run(tmp_path, backend, *, entries=None):
    entries = entries or [_entry()]
    cfg = HyMemConfig(root=tmp_path)
    llm = probe.CapturingLLM(backend)
    with closing(probe.build_store(tmp_path / "probe.sqlite", entries, cfg)) as conn:
        rows = [probe.extract_one(conn, e, llm, cfg, granular=True) for e in entries]
    return rows, llm


@pytest.mark.parametrize("compact", [False, True])
def test_primary_exact_capture_survives_verification_and_compaction(tmp_path, compact):
    (row,), llm = _run(tmp_path, _backend(compact=compact))
    assert not row["digest_failed"] and not row["parse_failed"]
    assert row["calls"] == (4 if compact else 3)
    records = row["completion_records"]
    assert [r["stage"] for r in records] == (
        ["primary", "summary_compaction", "fidelity_verification", "format_adjudication"] if compact
        else ["primary", "fidelity_verification", "format_adjudication"])
    assert row["extractor_input"] == records[0]["user"] == llm.sent[0]["user"]
    assert row["extractor_input"] != records[-1]["user"]
    assert "Café 🧬" in row["extractor_input"]
    assert row["extractor_input_sha256"] == hashlib.sha256(
        records[0]["user"].encode()).hexdigest()
    if compact:
        assert records[0]["user"] == records[1]["user"]
    for record in records:
        assert record["reply_sha256"] == hashlib.sha256(record["reply"].encode()).hexdigest()
        assert record["reply_chars"] == len(record["reply"])
        assert record["user_sha256"] == hashlib.sha256(record["user"].encode()).hexdigest()
    assert row["reply_chars"] == records[0]["reply_chars"]
    probe.assert_full_source(row)


@pytest.mark.parametrize("fail,stage,calls,parse_failed", [
    ("primary", "primary", 1, True),
    ("compaction", "summary_compaction", 2, True),
    ("malformed", "fidelity_verification", 2, True),
    ("unsupported", "fidelity_verification", 2, True),
    ("transport", "fidelity_verification", 2, False),
])
def test_failure_attribution_and_attempts_survive_later_failures(
    tmp_path, fail, stage, calls, parse_failed,
):
    (row,), _ = _run(tmp_path, _backend(compact=fail == "compaction", fail=fail))
    assert row["calls"] == calls
    assert row["failure_stage"] == stage
    assert row["digest_failed"] and row["parse_failed"] is parse_failed
    assert not row["episodes"]
    assert len(row["completion_records"]) == calls
    assert row["failure_reply_chars"] == row["completion_records"][-1]["reply_chars"]
    if fail == "transport":
        assert row["failure_reply_chars"] is None
        assert row["backend_error"] == "execution_failure:OSError"
        assert "sensitive provider address" not in json.dumps(row)
    if fail == "unsupported":
        assert row["failure_reason"] == "episode_title_unsupported"
        assert row["failure_reply_chars"] != row["reply_chars"]
    probe.assert_full_source(row)
    row["extractor_input"] = row["extractor_input"][:-1]
    with pytest.raises(AssertionError, match="TRUNCATED"):
        probe.assert_full_source(row)


def test_reused_client_never_borrows_old_request_or_error(tmp_path):
    first = True
    good = _backend()
    def backend(system, user):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("first only")
        return good(system, user)
    rows, llm = _run(tmp_path, backend, entries=[_entry(), _entry("s2"), _entry("empty", empty=True)])
    assert [r["calls"] for r in rows] == [1, 3, 0]
    assert rows[0]["backend_error"] == "execution_failure:RuntimeError"
    assert rows[1]["backend_error"] is None and rows[1]["error"] is None
    assert rows[1]["completion_records"] == llm.sent[1:4]
    assert llm.last_error is None
    assert rows[2]["error"] == "no_digest_input"
    assert rows[2]["extractor_input"] is None
    assert rows[2]["completion_records"] == [] and rows[2]["backend_error"] is None


@pytest.mark.parametrize("after_call", [False, True])
def test_exception_outside_digest_parse_flag_is_counted(tmp_path, monkeypatch, after_call):
    def broken(conn, sid, llm, **kwargs):
        if after_call:
            llm.complete(LLMRequest(system=probe.SESSION_DIGEST_SYSTEM, user="literal input"))
        raise RuntimeError("private path and key must not be copied")
    monkeypatch.setattr(probe, "extract_session_digest", broken)
    (row,), _ = _run(tmp_path, lambda system, user: "{}")
    assert row["calls"] == int(after_call)
    assert row["parse_failed"] is False and row["digest_failed"] is True
    assert row["failure_stage"] == "unknown"
    assert row["failure_reply_chars"] is None, "last reply is not proof of failure stage"
    assert row["extractor_input"] == ("literal input" if after_call else None)
    summary = probe.summarize([row], [], 1.0)
    assert summary["digest_failure_rate"] == 1.0
    assert not summary["gate"]["parse_failures_ok"]


def _mechanical_row(index, *, failure=False, calls=2, error=False):
    return {
        "session_id": str(index), "arm": "target" if index < 10 else "control",
        "session_chars": 4000, "episodes": [
            {"summary": "Enabled feature 2.4.", "outcome": "informational"}
            for _ in range(5)],
        "calls": calls, "parse_failed": failure and not error,
        "error": "execution_failure:RuntimeError" if failure and error else None,
    }


@pytest.mark.parametrize("calls", [2, 3, 20])
@pytest.mark.parametrize("error", [False, True])
def test_extra_completions_cannot_dilute_failed_session_ceiling(calls, error):
    rows = [_mechanical_row(i, failure=i < 2, calls=calls, error=error) for i in range(20)]
    s = probe.summarize(rows[:10], rows[10:], 1.0)
    assert s["attempted_session_digests"] == 20
    assert s["calls"] == 20 * calls
    assert s["digest_failures"] == 2 and s["digest_failure_rate"] == 0.1
    assert s["parse_failure_rate"] == (0.0 if error else 0.1)
    assert not s["gate"]["parse_failures_ok"]
    assert s["verdict"].startswith("INCOMPLETE")
    assert s["thresholds"]["session_digest_failure_rate"] == 0.02


def test_unsupported_and_historical_replies_do_not_imply_truncation(tmp_path, capsys):
    (row,), _ = _run(tmp_path, _backend(fail="unsupported"))
    # Supply independently populated mechanical rows so failure diagnostics run.
    rows = [_mechanical_row(i) for i in range(4)] + [row]
    historical = _mechanical_row(6, failure=True)
    historical["reply_chars"] = 200
    rows.append(historical)
    s = probe.summarize(rows, [_mechanical_row(10)], 1.0)
    probe.report(s, {}, "granular", False, rows)
    output = capsys.readouterr().out
    assert "episode_title_unsupported" in output
    assert "Missing historical stage evidence remains UNKNOWN" in output
    assert "raising --max-tokens is the remedy" not in output
    assert s["failure_stage_counts"] == {"fidelity_verification": 1, "unknown": 1}
    assert s["failure_replies_recorded"] == 1


def test_verifier_input_cap_has_no_fabricated_verifier_reply(tmp_path, monkeypatch):
    def capped_payload(payload, *, system=None):
        assert system == digest._DIGEST_FIDELITY_SYSTEM
        return None

    monkeypatch.setattr(digest, "_encode_digest_fidelity_payload", capped_payload)
    (row,), _ = _run(tmp_path, _backend())
    assert row["calls"] == 1 and row["digest_failed"]
    assert row["failure_stage"] == "fidelity_verification"
    assert row["failure_reason"] == "fidelity_input_cap"
    assert row["reply_chars"] > 0 and row["failure_reply_chars"] is None


def test_cost_is_offline_and_prints_three_seven_completion_bounds(tmp_path, monkeypatch, capsys):
    # Cost selection is stubbed, but the real CLI must exit before constructing
    # an API client, a CapturingLLM or an extraction store.
    import longmemeval_adapter as adapter
    source = tmp_path / "run.json"
    source.write_text("{}")
    monkeypatch.setattr(adapter, "load_longmemeval_data", lambda *a, **k: [])
    monkeypatch.setattr(probe, "select_session_sets", lambda *a, **k: (
        [_entry()], [_entry("s2"), _entry("s3")], {}))
    def forbidden(*a, **k):
        raise AssertionError("cost must never construct a client or store")
    monkeypatch.setattr(adapter, "LLMClient", forbidden)
    monkeypatch.setattr(probe, "CapturingLLM", forbidden)
    monkeypatch.setattr(probe, "build_store", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(sys, "argv", ["episode_probe.py", "--source", str(source),
                                     "--dataset", "unused.json", "--cost"])
    probe.main()
    output = capsys.readouterr().out
    assert "3 attempted session digests" in output
    assert "normally 9 logical completions (3/session)" in output
    assert ("at most 21 (7/session with summary compaction, diagnosis, content recovery, "
            "repeated verification and format adjudication)") in output
    assert "provider HTTP retries are separate" in output


def test_cli_sim_uses_requested_declared_token_bound(tmp_path, monkeypatch):
    import longmemeval_adapter as adapter
    source = tmp_path / "run.json"
    source.write_text("{}")
    monkeypatch.setattr(adapter, "load_longmemeval_data", lambda *a, **k: [])
    monkeypatch.setattr(probe, "select_session_sets", lambda *a, **k: ([_entry()], [], {}))
    original = probe.extract_one
    seen = []
    def extract(conn, entry, llm, cfg, **kwargs):
        result = original(conn, entry, llm, cfg, **kwargs)
        seen.extend(result["completion_records"])
        assert cfg.dream_digest_max_tokens == llm.max_tokens == 4097
        return result
    monkeypatch.setattr(probe, "extract_one", extract)
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["episode_probe.py", "--source", str(source),
                                     "--dataset", "unused.json", "--sim", "--max-tokens", "4097"])
    with pytest.raises(SystemExit):
        probe.main()
    assert len(seen) == 3 and all(r["max_tokens"] == 4097 for r in seen)


def _format_backend(*, compact=False, outcome="supported"):
    original = _backend(compact=compact)
    def backend(system, user):
        if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            if outcome == "transport":
                raise OSError("private adjudicator endpoint and credentials")
            if outcome == "malformed":
                return "{broken adjudicator response"
            verdict = json.loads(probe.sim_backend(system, user))
            if outcome == "shape":
                verdict["unrecognized"] = []
            else:
                verdict["summary_format"][0]["verdict"] = outcome
            return json.dumps(verdict)
        if system == probe._DIGEST_FIDELITY_SYSTEM:
            return _verdict(user)
        return original(system, user)
    return backend


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("outcome,reason", [
    ("supported", None),
    ("unsupported", "summary_format_unsupported"),
    ("uncertain", "summary_format_uncertain"),
    ("malformed", "format_adjudication_parse_failure"),
    ("shape", "format_adjudication_shape_failure"),
    ("transport", "execution_failure:DigestCompletionError"),
])
def test_format_adjudication_preserves_exact_multicall_evidence(
    tmp_path, compact, outcome, reason,
):
    (row,), client = _run(tmp_path, _format_backend(compact=compact, outcome=outcome))
    assert row["record_version"] == "episode-probe-multicall-v5"
    records = row["completion_records"]
    assert row["calls"] == client.calls == len(records) == (4 if compact else 3)
    assert [r["stage"] for r in records] == (
        ["primary"] + (["summary_compaction"] if compact else [])
        + ["fidelity_verification", "format_adjudication"])
    for record in records:
        assert record["user_chars"] == len(record["user"])
        assert record["user_sha256"] == hashlib.sha256(record["user"].encode()).hexdigest()
        if record["reply"] is not None:
            assert record["reply_chars"] == len(record["reply"])
            assert record["reply_sha256"] == hashlib.sha256(record["reply"].encode()).hexdigest()
        else:
            assert record["reply_sha256"] is record["reply_chars"] is None
    assert row["reply_chars"] == records[0]["reply_chars"]
    assert row["reply_head"] == records[0]["reply_head"]
    assert row["failure_reason"] == reason
    assert row["failure_stage"] == (None if outcome == "supported" else "format_adjudication")
    assert row["digest_failed"] is (outcome != "supported")
    assert row["parse_failed"] is (outcome not in {"supported", "transport"})
    assert row["failure_reply_chars"] == (
        None if outcome == "supported" else records[-1]["reply_chars"])
    if outcome != "supported":
        assert not row["episodes"]
        assert row["failure_reply_chars"] != records[-2]["reply_chars"]
        summary = probe.summarize([row], [], 1.0)
        assert summary["failure_stage_counts"] == {"format_adjudication": 1}
        assert summary["digest_failure_rate"] == 1.0
    assert "private adjudicator" not in json.dumps(row)
    probe.assert_full_source(row)


@pytest.mark.parametrize("compact", [False, True])
def test_adjudication_input_cap_does_not_borrow_screening_reply(tmp_path, monkeypatch, compact):
    monkeypatch.setattr(digest, "_encode_digest_format_adjudication_payload", lambda payload: None)
    (row,), _ = _run(tmp_path, _format_backend(compact=compact))
    assert row["calls"] == (3 if compact else 2)
    assert row["failure_stage"] == "format_adjudication"
    assert row["failure_reason"] == "format_adjudication_input_cap"
    assert row["completion_records"][-1]["stage"] == "fidelity_verification"
    assert row["completion_records"][-1]["reply_chars"] > 0
    assert row["failure_reply_chars"] is row["failure_reply_head"] is None


@pytest.mark.parametrize("system", [
    digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM,
    digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM + " ",
    "User prose mentioning format_adjudication",
], ids=["exact", "modified-prompt", "unrecognized"])
def test_format_stage_is_exact_prompt_identity_not_source_text(system):
    request = LLMRequest(system=system, user=digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM)
    assert probe._request_stage(request) == (
        "format_adjudication" if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM else "unknown")


@pytest.mark.parametrize("system", [digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM, "unknown prompt"],
                         ids=["adjudication", "unknown-prompt"])
def test_unattributed_exception_after_recorded_call_remains_unknown(tmp_path, monkeypatch, system):
    def broken(conn, sid, llm, **kwargs):
        llm.complete(LLMRequest(system=system, user="{}"))
        raise RuntimeError("unknown post-request failure")
    monkeypatch.setattr(probe, "extract_session_digest", broken)
    (row,), _ = _run(tmp_path, lambda system, user: "{}")
    assert row["completion_records"][0]["stage"] == (
        "format_adjudication" if system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM else "unknown")
    assert row["failure_stage"] == "unknown"
    assert row["failure_reply_chars"] is row["failure_reply_head"] is None


@pytest.mark.parametrize("count", [0, 1, 5])
def test_simulated_adjudicator_covers_strict_schema_without_quality_claim(count, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("simulation must not access networking")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    payload = digest._digest_format_adjudication_payload(
        "This synthetic verdict intentionally does not evaluate grammar.",
        [{"summary": "SYNTHETIC FORMAT CONTROL"} for _ in range(count)],
    )
    response = probe.sim_backend(digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM, json.dumps(payload))
    assert digest._validate_digest_format_adjudication_response(response, count) is None
    assert set(json.loads(response)) == {"summary_format", "episode_format"}


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_historical_attribution_is_not_reinterpreted_by_rescoring(version):
    historical = _mechanical_row(0, failure=True, calls=3)
    historical.update({
        "record_version": f"episode-probe-multicall-{version}",
        "failure_stage": "fidelity_verification",
        "failure_reason": "summary_format_unsupported",
        "reply_chars": 100, "failure_reply_chars": 80,
    })
    unknown = _mechanical_row(1, failure=True, calls=3)
    original = deepcopy([historical, unknown])
    summary = probe.summarize([historical, unknown], [], None)
    assert [historical, unknown] == original
    assert summary["failure_stage_counts"] == {"fidelity_verification": 1, "unknown": 1}
    assert summary["failure_replies_recorded"] == 1
    assert summary["digest_failures"] == summary["attempted_session_digests"] == 2


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_cli_rescore_preserves_historical_raw_requests_replies_and_metadata(
    tmp_path, monkeypatch, capsys, version,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("rescore must not construct a client, store or network request")
    monkeypatch.setattr(probe, "CapturingLLM", forbidden)
    monkeypatch.setattr(probe, "build_store", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    row = _mechanical_row(0, failure=True, calls=3)
    source_text = 'Historical synthetic source: Café 🧬 said "not yet".\n'
    raw_reply = '{ "historical": "unparsed bytes stay exact" }\n'
    row.update({
        "record_version": f"episode-probe-multicall-{version}",
        "failure_stage": "fidelity_verification",
        "failure_reason": "summary_format_unsupported",
        "extractor_input": source_text,
        "extractor_input_chars": len(source_text),
        "extractor_input_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "completion_records": [{
            "stage": "fidelity_verification", "system": "historical six-family task",
            "user": source_text, "user_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "reply": raw_reply, "reply_sha256": hashlib.sha256(raw_reply.encode()).hexdigest(),
            "max_tokens": 3072, "temperature": 0.0, "response_format": "json",
        }],
        "failure_reply_chars": len(raw_reply), "reply_chars": 99,
    })
    artifact = {
        "per_session": [row], "prompt_arm": "historical-arm", "model": "historical-model",
        "summary": {"old_summary": True}, "sim": True,
        "selection": {"historical_seed": 17}, "custom_metadata": {"nested": [version]},
    }
    prior = tmp_path / "historical.json"
    output = tmp_path / "rescored.json"
    original = json.dumps(artifact, ensure_ascii=False).encode()
    prior.write_bytes(original)
    monkeypatch.setattr(sys, "argv", [
        "episode_probe.py", "--source", "unused-run.json", "--dataset", "unused-data.json",
        "--rescore", str(prior), "--out", str(output),
    ])
    with pytest.raises(SystemExit) as caught:
        probe.main()
    assert caught.value.code == 1  # A recorded failed session stays failed.
    rescored = json.loads(output.read_text())
    assert prior.read_bytes() == original
    for key in artifact.keys() - {"summary"}:
        assert rescored[key] == artifact[key]
    assert rescored["summary"]["failure_stage_counts"] == {"fidelity_verification": 1}
    assert rescored["rescored_from"] == {"artifact_sha256": probe.content_hash(artifact)}
    assert "ZERO LLM calls" in capsys.readouterr().out


def test_help_describes_optional_seventh_call_without_constructing_client(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("help must not access client, store or networking")
    monkeypatch.setattr(probe, "CapturingLLM", forbidden)
    monkeypatch.setattr(probe, "build_store", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(sys, "argv", ["episode_probe.py", "--help"])
    with pytest.raises(SystemExit) as caught:
        probe.main()
    assert caught.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert ("at most 7 with summary compaction, diagnosis, content recovery, repeated "
            "verification and format adjudication") in output
    assert "3/7 logical-completion budget" in output
    assert "provider HTTP retries are separate" in output
