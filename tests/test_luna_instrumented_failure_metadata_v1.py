"""Offline privacy, bounds, and gate controls for the stopped-pilot projector."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_failure_metadata_v1 as metadata


def projection():
    scope = {"INDEXING_CODES": metadata.INDEXING_CODES,
             "EXCEPTION_TYPES": metadata.EXCEPTION_TYPES,
             "RECEIPT_SHA": metadata.RECEIPT_SHA, "SCHEMA": metadata.SCHEMA}
    exec(compile(metadata.PROJECTION, "<test-projection>", "exec"), scope)
    return scope


def report():
    return {"status": "terminal_incomplete_or_unclean",
        "runtime_cleanup_verified": True, "campaign_stop": "question_failure",
        "budget_stop_code": "question_failure", "selected_denominator": 4,
        "scored_count": 0, "known_turns": 4586, "known_tokens": 33246386,
        "usage_complete": True, "first_failure": None,
        "resource_observation": {"denials": 0}}


def fixture(tmp_path):
    ids = [f"private-qid-{i}" for i in range(4)]
    checkpoint = {"expected_ids": ids,
        "entries": {qid: {"status": "failed", "row": {
            "benchmark_failure": "question_failure", "private": "SECRET_ROW"}}
            for qid in ids}}
    reads = []
    def read(path, root, cap):
        reads.append(path)
        if path.name == "diagnostic-checkpoint.json":
            return checkpoint
        if path.name == "private-indexing.json":
            return {"outcome": "failure", "healthy": False, "cycles": 7,
                "elapsed_s": 1.5, "failure": {"code": "quarantined_extraction",
                    "exception_type": "SECRET_EXCEPTION"},
                "reports": [{"private": "SECRET_REPORT"}],
                "cleanup_errors": ["SECRET_CLEANUP"]}
        if path.name == "private-diagnostic-indexing.json":
            return {"admitted": False, "kind": "rejected",
                "quarantined_chunks": 2, "semantic_failure_reasons": {
                    "SECRET_DYNAMIC_KEY": 1}}
        raise AssertionError(path)
    namespace = {"_read": read, "_receipt": lambda *args: {"selected_count": 4},
        "_checkpoint": lambda *args: checkpoint,
        "_counts": lambda *args: ({"failed": 4}, 0, 0, 0),
        "_file": lambda path, root, cap: path.is_file() and path.stat().st_size <= cap}
    for i in range(4):
        directory = tmp_path / "run" / f"q-{i:04d}"
        directory.mkdir(parents=True)
        for name in ("private-indexing.json", "private-diagnostic-indexing.json"):
            (directory / name).write_text("{}")
    return namespace, reads


def test_projection_is_closed_and_private_row_is_presence_only(tmp_path):
    scope = projection()
    namespace, reads = fixture(tmp_path)
    (tmp_path / "run/q-0000/private-row.json").write_text("SECRET_ROW_CONTENT")
    result = scope["_project"](tmp_path, namespace, report())
    wire = json.dumps(result)
    assert "SECRET" not in wire and "private-qid" not in wire
    assert result["questions"]["q-0000"]["private_row_present"] is True
    assert result["questions"]["q-0001"]["private_row_present"] is False
    assert result["questions"]["q-0000"]["indexing"]["failure_code"] == "quarantined_extraction"
    assert not any(path.name == "private-row.json" for path in reads)
    assert metadata._validated(result) == result


@pytest.mark.parametrize("change", [
    {"runtime_cleanup_verified": False}, {"known_turns": 4585},
    {"campaign_stop": "other"}, {"first_failure": {"code": "secret"}},
    {"resource_observation": {"denials": 1}},
])
def test_terminal_gate_precedes_all_private_reads(tmp_path, change):
    namespace, reads = fixture(tmp_path)
    candidate = report()
    candidate.update(change)
    with pytest.raises(ValueError, match="terminal_gate_invalid"):
        projection()["_project"](tmp_path, namespace, candidate)
    assert reads == []


def test_unknown_nonfinite_and_oversize_values_are_fixed_or_null():
    scope = projection()
    projected = scope["_fields"]({"status": "SECRET", "outcome": "SECRET",
        "cycles": float("nan"), "elapsed_s": float("inf"),
        "quarantined_chunks": 1_000_000_001, "healthy": 1,
        "failure": {"code": "SECRET"}, "cleanup_errors": ["SECRET"]})
    assert projected == {"object": True, "status": "other", "outcome": "other",
        "cycles": None, "elapsed_s": None, "quarantined_chunks": None,
        "failure_code": "other", "cleanup_error_count": 1}


def test_symlink_private_file_rejected(tmp_path):
    namespace, _ = fixture(tmp_path)
    target = tmp_path / "outside"
    target.write_text("SECRET")
    (tmp_path / "run/q-0000/private-row.json").symlink_to(target)
    with pytest.raises(ValueError, match="private_file_invalid"):
        projection()["_project"](tmp_path, namespace, report())


def test_stdout_validation_rejects_dynamic_fields_and_values(tmp_path):
    namespace, _ = fixture(tmp_path)
    value = projection()["_project"](tmp_path, namespace, report())
    value["questions"]["q-0000"]["indexing"]["private"] = "SECRET"
    assert metadata._validated(value) is None


@pytest.mark.parametrize("field,bad", [
    ("status", "SECRET_STATUS"), ("outcome", "SECRET_OUTCOME"),
    ("failure_code", "SECRET_CODE"), ("exception_type", "SECRET_EXCEPTION"),
    ("cycles", float("nan")), ("healthy", "SECRET_BOOL"),
    ("cleanup_error_count", 1_000_000_001),
])
def test_stdout_validation_rejects_malformed_projected_fields(tmp_path, field, bad):
    namespace, _ = fixture(tmp_path)
    value = projection()["_project"](tmp_path, namespace, report())
    value["questions"]["q-0000"]["indexing"][field] = bad
    assert metadata._validated(value) is None


def test_checkpoint_failure_is_closed_and_source_writer_code_is_preserved(tmp_path):
    namespace, _ = fixture(tmp_path)
    checkpoint = namespace["_read"](tmp_path / "run/diagnostic-checkpoint.json",
        tmp_path, 2_000_000)
    checkpoint["entries"][checkpoint["expected_ids"][0]]["row"][
        "benchmark_failure"] = "unspecified_failure"
    result = projection()["_project"](tmp_path, namespace, report())
    assert result["questions"]["q-0000"]["checkpoint_failure_code"] == "unspecified_failure"
    checkpoint["entries"][checkpoint["expected_ids"][0]]["row"][
        "benchmark_failure"] = "SECRET_FAILURE"
    result = projection()["_project"](tmp_path, namespace, report())
    assert result["questions"]["q-0000"]["checkpoint_failure_code"] == "other"


def test_local_reader_pin_blocks_ssh(monkeypatch, capsys):
    monkeypatch.setattr(metadata, "READER", Path(__file__))
    monkeypatch.setattr(metadata.subprocess, "run", lambda *args, **kwargs:
        pytest.fail("SSH must not run after reader pin mismatch"))
    assert metadata.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": metadata.SCHEMA, "status": "metadata_unavailable"}


def test_bad_remote_stdout_cannot_escape(monkeypatch, capsys):
    monkeypatch.setattr(metadata.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout='{"private":"SECRET"}',
            stderr="SECRET_STDERR"))
    assert metadata.main() == 1
    assert "SECRET" not in capsys.readouterr().out
