"""Offline controls for the stopped SIWC metadata projector; no host connection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import siwc_lme_failure_metadata_v1 as m
from tools.diagnostics import siwc_lme_diagnostic_progress_v3 as reader_module


def scope():
    values = {name: getattr(m, name) for name in (
        "RECEIPT_SHA", "SCHEMA", "INDEXING_SCHEMA", "INDEXING_CODES",
        "EXCEPTION_TYPES", "PENDING", "MALFORMED", "QUARANTINED", "SUMMARY",
        "CHECKPOINT_CODES", "MAX_COUNT", "MAX_SECONDS")}
    exec(compile(m.PROJECTION, "<offline-projection>", "exec"), values)
    return values


def gate():
    return {"status": "terminal_incomplete_or_unclean", "runtime_cleanup_verified": True,
        "completed_diagnostic_and_clean": False, "selected_denominator": 4,
        "scored_count": 0, "failed_count": 4, "known_turns": 3554,
        "known_tokens": 11837159, "usage_complete": True,
        "campaign_stop": "question_failure", "budget_stop_code": "question_failure",
        "first_failure": None, "owner_failure": None, "resource_fault": None,
        "resource_observation": {"denials": 0}}


def indexing():
    sentinel = "private account and question text: never exported"
    return {"schema": m.INDEXING_SCHEMA, "outcome": "failure", "cycles": 2,
        "max_cycles": 100, "elapsed_s": 18.5, "timeout_s": 10800,
        "complete": False, "healthy": False,
        "failure": {"code": "cycle_exception", "exception_type": "ValueError",
                    "traceback": sentinel},
        "cleanup_errors": [sentinel], "private_question": sentinel,
        "final_status": {
            "pending": {name: 0 for name in m.PENDING},
            "malformed": {name: 0 for name in m.MALFORMED},
            "quarantined": {name: 0 for name in m.QUARANTINED},
            "terminal_loss": {"chunks": 0, "reasons": {sentinel: 1}},
            "coverage_integrity": {"failures": 0, "details": [sentinel]},
            "summary_health": {"summary_degraded_sessions": 0,
                "summary_missing_sessions": 0, "malformed_summaries": 0,
                "summary_healthy": True},
            "account": sentinel}}


def fake_reader(root: Path, *, report=None, private=None):
    calls = []
    ids = [f"private-secret-id-{i}" for i in range(4)]
    check = {"expected_ids": ids, "entries": {qid: {"status": "failed",
        "failure": "indexing_rejected"} for qid in ids}}
    metadata = indexing() if private is None else private

    def read(path, _root, cap):
        calls.append((path, cap))
        if path.name == "diagnostic-checkpoint.json":
            return check
        assert path.name == "private-indexing.json"
        assert cap == 262144
        return metadata

    return {"inspect": lambda *_: gate() if report is None else report,
        "receipt": lambda *_: {"checked": True},
        "checkpoint": lambda *_: ({"expected": 4, "failed": 4, "missing": 0},
            0, 0, 0, check), "read": read}, calls


def directory(tmp_path):
    root = tmp_path / "root"
    (root / "run").mkdir(parents=True)
    for index in range(4):
        place = root / "run" / f"q-{index:04d}"
        place.mkdir()
        (place / "private-indexing.json").write_text("{}")
    return root


def test_gate_precedes_every_private_read(tmp_path):
    root = directory(tmp_path)
    bad = gate()
    bad["known_turns"] = 3553
    fake, calls = fake_reader(root, report=bad)
    with pytest.raises(ValueError, match="terminal_gate_invalid"):
        scope()["_project"](root, fake)
    assert calls == []
    bad = gate()
    bad["resource_observation"]["denials"] = False
    fake, calls = fake_reader(root, report=bad)
    with pytest.raises(ValueError, match="terminal_gate_invalid"):
        scope()["_project"](root, fake)
    assert calls == []


def test_finite_projection_uses_ordinal_labels_and_no_private_text(tmp_path):
    root = directory(tmp_path)
    fake, calls = fake_reader(root)
    value = scope()["_project"](root, fake)
    assert m._validated(value) == value
    assert set(value["questions"]) == {f"q-{i:04d}" for i in range(4)}
    exported = json.dumps(value)
    assert "private-secret-id" not in exported
    assert "private account and question text" not in exported
    assert "traceback" not in exported
    assert len(calls) == 5
    assert calls[0][0].name == "diagnostic-checkpoint.json"
    assert all(path.name == "private-indexing.json" for path, _ in calls[1:])
    assert value["questions"]["q-0000"]["indexing"]["failure_code"] == "cycle_exception"


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(schema="unknown"),
    lambda v: v.update(cycles=-1),
    lambda v: v.update(cycles=True),
    lambda v: v.update(elapsed_s=float("nan")),
    lambda v: v["final_status"]["pending"].update(pending_chunks=-1),
    lambda v: v["final_status"]["pending"].update(pending_chunks=True),
    lambda v: v["final_status"]["pending"].update(unexpected=0),
    lambda v: v["final_status"]["summary_health"].update(summary_healthy="yes"),
])
def test_malformed_metadata_fails_closed(mutation):
    value = indexing()
    mutation(value)
    with pytest.raises(ValueError):
        scope()["_indexing"](value)


def test_unknown_code_and_type_are_finite():
    value = indexing()
    value["failure"]["code"] = "private-custom-failure: details"
    value["failure"]["exception_type"] = "Sensitive.Exception: details"
    result = scope()["_indexing"](value)
    assert result["failure_code"] == "other"
    assert result["exception_type"] == "other"


def test_missing_private_file_is_unknown_not_zero(tmp_path):
    root = directory(tmp_path)
    (root / "run" / "q-0002" / "private-indexing.json").unlink()
    fake, calls = fake_reader(root)
    result = scope()["_project"](root, fake)
    assert result["questions"]["q-0002"]["indexing"] == {"present": False}
    assert len(calls) == 4


def test_run_and_private_symlinks_fail_closed(tmp_path):
    root = directory(tmp_path)
    moved = root / "other"
    (root / "run").rename(moved)
    (root / "run").symlink_to(moved, target_is_directory=True)
    fake, calls = fake_reader(root)
    with pytest.raises(ValueError, match="run_directory_invalid"):
        scope()["_project"](root, fake)
    assert calls == []
    (root / "run").unlink()
    moved.rename(root / "run")
    private = root / "run" / "q-0000" / "private-indexing.json"
    private.unlink()
    private.symlink_to(root / "run" / "q-0001" / "private-indexing.json")
    fake, calls = fake_reader(root)
    with pytest.raises(ValueError, match="private_file_invalid"):
        scope()["_project"](root, fake)
    assert len(calls) == 1


def test_pinned_reader_rejects_duplicate_nonfinite_and_oversize(tmp_path):
    root = tmp_path
    path = root / "private-indexing.json"
    for raw in ('{"a":1,"a":2}', '{"a":NaN}', "x" * 262145):
        path.write_text(raw)
        with pytest.raises(ValueError):
            reader_module.read(path, root, 262144)


def test_main_source_pin_and_unsafe_stdout_are_suppressed(tmp_path, monkeypatch, capsys):
    source = tmp_path / "reader.py"
    source.write_text("drift")
    monkeypatch.setattr(m, "READER", source)
    called = []
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: called.append((a, kw)))
    assert m.main() == 1
    assert called == []
    assert "drift" not in capsys.readouterr().out

    monkeypatch.setattr(m, "READER_SHA", hashlib.sha256(source.read_bytes()).hexdigest())
    sentinel = "private bearer token sentinel"
    def fake_run(*args, **kwargs):
        called.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"schema": m.SCHEMA,
            "questions": {"q-0000": {"arbitrary": sentinel}}}), stderr=sentinel)
    monkeypatch.setattr(m.subprocess, "run", fake_run)
    assert m.main() == 1
    assert sentinel not in capsys.readouterr().out
    command = called[-1][0][0]
    assert command[:7] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           "-o", "ConnectionAttempts=1"]
    assert called[-1][1]["capture_output"] is True


def test_local_validator_rejects_arbitrary_and_nonfinite_fields(tmp_path):
    root = directory(tmp_path)
    fake, _ = fake_reader(root)
    report = scope()["_project"](root, fake)
    report["questions"]["q-0000"]["indexing"]["unexpected"] = "private"
    assert m._validated(report) is None
    del report["questions"]["q-0000"]["indexing"]["unexpected"]
    report["questions"]["q-0000"]["indexing"]["elapsed_s"] = float("inf")
    assert m._validated(report) is None


def test_frozen_candidate_indexing_schema_pin():
    frozen = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle/candidate/benchmarks/lme_protocol.py")
    if not frozen.exists():
        pytest.skip("preflight copy not retained")
    import ast
    tree = ast.parse(frozen.read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value)
              for node in tree.body if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == "LME_INDEXING_SUMMARY_VERSION"}
    assert values["LME_INDEXING_SUMMARY_VERSION"] == m.INDEXING_SCHEMA
