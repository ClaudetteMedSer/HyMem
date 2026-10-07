"""Offline source-field, closed-output, and v1-gate controls for v2."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_failure_metadata_v2 as v2


def scope():
    result = {name: getattr(v2, name) for name in (
        "PENDING", "MALFORMED", "QUARANTINED", "SUMMARY_COUNTS", "MAX_COUNT", "SCHEMA")}
    result["V1_SCHEMA"] = v2.v1.SCHEMA
    prior = {"INDEXING_CODES": v2.v1.INDEXING_CODES,
        "EXCEPTION_TYPES": v2.v1.EXCEPTION_TYPES}
    exec(compile(v2.v1.PROJECTION, "<offline-v1-projection>", "exec"), prior)
    result["V1_FIELDS"] = prior["_fields"]
    exec(compile(v2.EXTENSION, "<offline-extension>", "exec"), result)
    return result


def final_status():
    return {"pending": {name: 0 for name in v2.PENDING},
        "malformed": {name: 0 for name in v2.MALFORMED},
        "quarantined": {name: 0 for name in v2.QUARANTINED},
        "terminal_loss": {"chunks": 0, "reasons": {"SECRET_REASON": 1}},
        "coverage_integrity": {"failures": 0, "details": ["SECRET_DETAIL"]},
        "summary_health": {"summary_degraded_sessions": 0,
            "summary_missing_sessions": 0, "malformed_summaries": 0,
            "summary_healthy": True},
        "private_id": "SECRET_HASH", "aggregation_material": {"secret": "SECRET"}}


def base():
    questions = {}
    for i in range(4):
        questions[f"q-{i:04d}"] = {"checkpoint_failure_code": "unspecified_failure",
            "private_row_present": False,
            "indexing": ({"present": True, "object": True,
                "outcome": "failure", "failure_code": "quarantined_extraction"}
                if i == 1 else {"present": False}),
            "diagnostic_indexing": {"present": False}}
    return {"schema": v2.v1.SCHEMA, "terminal_and_cleanup_verified": True,
        "source_receipt_verified": True, "questions": questions}


def test_exact_frozen_field_names_and_nonchunk_quarantine_projection():
    assert set(v2.MALFORMED) == {"malformed_source_materialization",
        "malformed_digests", "malformed_profiles", "malformed_facts",
        "malformed_summaries"}
    assert set(v2.QUARANTINED) == {"quarantined_chunks", "quarantined_digests",
        "quarantined_profiles", "quarantined_facts", "quarantined_facts_malformed"}
    raw = final_status()
    raw["quarantined"]["quarantined_digests"] = 3
    output = scope()["_final"](raw)
    assert output["quarantined"]["quarantined_chunks"] == 0
    assert output["quarantined"]["quarantined_digests"] == 3
    assert "SECRET" not in json.dumps(output)
    assert v2._validated_final(output)


def test_extension_reads_only_existing_fixed_indexing_path_after_gate(tmp_path):
    reads = []
    raw = {"final_status": final_status(), "private_text": "SECRET",
        "outcome": "failure", "failure": {"code": "quarantined_extraction"}}
    namespace = {"_read": lambda path, root, cap: (reads.append(path), raw)[1]}
    result = scope()["_extend"](tmp_path, namespace, {}, base())
    assert reads == [tmp_path / "run/q-0001/private-indexing.json"]
    assert "SECRET" not in json.dumps(result)
    assert result["questions"]["q-0000"]["indexing"]["final_status"] is None
    assert v2._validated(result) == result


def test_v1_gate_precedes_private_read(tmp_path):
    reads = []
    candidate = base()
    candidate["terminal_and_cleanup_verified"] = False
    namespace = {"_read": lambda *args: reads.append(args)}
    with pytest.raises(ValueError, match="v1_gate_invalid"):
        scope()["_extend"](tmp_path, namespace, {}, candidate)
    assert reads == []
    assert v2.REMOTE.index("namespace['inspect']") < v2.REMOTE.index("scope['_project']")
    assert v2.REMOTE.index("scope['_project']") < v2.REMOTE.index("extension['_extend']")


def test_changed_indexing_metadata_between_v1_and_v2_is_rejected(tmp_path):
    namespace = {"_read": lambda *args: {"outcome": "success",
        "final_status": final_status()}}
    with pytest.raises(ValueError, match="private_metadata_changed"):
        scope()["_extend"](tmp_path, namespace, {}, base())


@pytest.mark.parametrize("mutation", [
    lambda x: x["pending"].update(pending_digests="SECRET"),
    lambda x: x["pending"].update(SECRET_KEY=1),
    lambda x: x["quarantined"].update(quarantined_chunks=True),
    lambda x: x["coverage_integrity"].update(failures=-1),
    lambda x: x["terminal_loss"].update(chunks=2_147_483_648),
    lambda x: x["malformed"].update(malformed_summaries=1),
    lambda x: x["summary_health"].update(summary_healthy="SECRET"),
    lambda x: x["summary_health"].update(summary_missing_sessions=1),
])
def test_malformed_or_ambiguous_final_state_fails_closed(mutation):
    candidate = final_status()
    mutation(candidate)
    with pytest.raises(ValueError):
        scope()["_final"](candidate)


def test_stdout_validator_rejects_private_extras_and_malformed_fields():
    projected = scope()["_final"](final_status())
    for field, value in (("pending", {"SECRET_KEY": 2}),
                         ("summary_health", {"SECRET": 1}),
                         ("terminal_loss", {"chunks": 0, "reasons": "SECRET"})):
        candidate = copy.deepcopy(projected)
        candidate[field] = value
        assert not v2._validated_final(candidate)
    candidate = base()
    candidate["schema"] = v2.SCHEMA
    for question in candidate["questions"].values():
        question["indexing"]["final_status"] = None
    candidate["questions"]["q-0001"]["indexing"]["final_status"] = projected
    assert v2._validated(candidate) == candidate
    candidate["questions"]["q-0001"]["indexing"]["final_status"]["private"] = "SECRET"
    assert v2._validated(candidate) is None


def test_v1_pin_blocks_ssh(monkeypatch, capsys):
    monkeypatch.setattr(v2, "V1_SHA", "0" * 64)
    monkeypatch.setattr(v2.subprocess, "run", lambda *args, **kwargs:
        pytest.fail("SSH must not run after v1 pin mismatch"))
    assert v2.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": v2.SCHEMA, "status": "metadata_unavailable"}


def test_remote_stdout_sentinel_never_escapes(monkeypatch, capsys):
    monkeypatch.setattr(v2.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout='{"private":"SECRET"}',
            stderr="SECRET_STDERR"))
    assert v2.main() == 1
    assert "SECRET" not in capsys.readouterr().out
