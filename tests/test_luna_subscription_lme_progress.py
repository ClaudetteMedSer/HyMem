"""Offline controls for the metadata-only LME progress reader."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.diagnostics import luna_subscription_lme_progress as progress


def test_progress_masks_usage_in_flight_and_drops_private_fields():
    value = {"invocation_in_flight": True, "usage_complete": True,
             "observed_turns": 4, "known_tokens": 50,
             "question_started": True, "raw_model_output": "secret",
             "canary": {"passed": True, "response": "secret"}}
    safe = progress.summarize_progress(value)
    assert safe["usage_complete"] is False
    assert safe["known_tokens"] == 50
    assert safe["canary_passed"] is True
    assert "secret" not in json.dumps(safe)
    assert progress.summarize_progress({"known_tokens": 4_000_001})["known_tokens"] == 4_000_001


def test_unavailable_progress_is_unknown_not_zero():
    assert progress.summarize_progress(None) == {
        "available": False, "usage_complete": None, "known_tokens": None}
    assert progress.summarize_progress({"observed_turns": -1})["observed_turns"] is None


def test_bounded_json_reader_rejects_large_or_symlink(tmp_path):
    path = tmp_path / "private-progress.json"
    path.write_bytes(b"x" * (progress.MAX_JSON_BYTES + 1))
    assert progress.read_json(path) is None
    path.write_text('{"ok":true}')
    assert progress.read_json(path) == {"ok": True}
    link = tmp_path / "link"
    link.symlink_to(path)
    assert progress.read_json(link) is None


def test_systemd_state_only_exact_cgroup_is_accepted(monkeypatch):
    unit = "hymem-luna-lme-v2-20260928-yeds3hl.service"
    expected = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=(
            "ActiveState=active\nSubState=running\nResult=success\nMainPID=123\n"
            "ControlGroup=" + expected + "\nNRestarts=0\nExecMainStatus=0\n"
            "RuntimeMaxUSec=5400000000\nKillMode=control-group\n"))
    monkeypatch.setattr(progress.subprocess, "run", fake_run)
    monkeypatch.setattr(progress, "cgroup_process_count", lambda path: 2)
    assert progress.unit_state(unit, expected)["exact_cgroup"] is True
    assert progress.unit_state(unit, expected)["expected_cgroup_processes"] == 2
    assert progress.unit_state(unit, expected + "-other")["exact_cgroup"] is False
    def blank_run(*args, **kwargs):
        value = fake_run(*args, **kwargs)
        return SimpleNamespace(returncode=0, stdout=value.stdout.replace(expected, ""))
    monkeypatch.setattr(progress.subprocess, "run", blank_run)
    assert progress.unit_state(unit, expected)["exact_cgroup"] is False
    assert progress.unit_state(unit, expected)["expected_cgroup_processes"] == 2


def test_terminal_validation_requires_pins_before_import(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.py"
    transport = tmp_path / "transport.py"
    dataset = tmp_path / "dataset.json"
    for path in (pilot, transport, dataset):
        path.write_text("not pinned")
    verdict = progress.terminal_check(safe={"ok": True}, result={}, row={},
        pilot_path=pilot, transport_path=transport, candidate=tmp_path,
        inventory_stamp=tmp_path / "stamp", inventory_sha256="0" * 64,
        dataset=dataset)
    assert verdict["validated"] is False
    assert verdict["source_pins_verified"] is False


def test_no_private_text_fields_in_file_stats(tmp_path):
    path = tmp_path / "private-run.log"
    path.write_text("private prompt and answer")
    assert set(progress.file_stat(path)) == {"present", "bytes", "mtime_ns"}


def test_positive_terminal_validation_exports_only_correct_boolean(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.py"
    pilot.write_text("def verify_inventory(*args):\n    return 508\n")
    transport = tmp_path / "transport.py"
    dataset = tmp_path / "dataset.json"
    for path in (transport, dataset):
        path.write_text("pinned")
    monkeypatch.setattr(progress, "digest", lambda path: {
        pilot: progress.PILOT_SHA256, transport: progress.TRANSPORT_SHA256,
        dataset: progress.DATASET_SHA256}[path])
    fake_protocol = SimpleNamespace(_validate_versioned_indexing=lambda *args, **kwargs: True)
    monkeypatch.setitem(sys.modules, "benchmarks", SimpleNamespace(lme_protocol=fake_protocol))
    indexing = {"outcome": "success", "healthy": True, "summary_healthy": True,
                "raw_prompt": "private"}
    row = {"indexing": indexing, "benchmark_failure": None,
           "judge_parse_valid": True, "judge_error": False, "correct": True,
           "answer": "private"}
    summary = {"question_completed": True, "cleanup_ok": True,
               "usage_complete": True, "invocation_in_flight": False,
               "observed_turns": 8, "known_tokens": 4500,
               "benchmark_failure": None, "correct": True}
    safe = {"ok": True, "question_completed": True, "canary_passed": True,
            "observed_turns": 8, "known_tokens": 4500, "usage_complete": True,
            "invocation_in_flight": False, "correct": True}
    verdict = progress.terminal_check(safe=safe, result={"summary": summary, "row": row},
        row=row, pilot_path=pilot, transport_path=transport, candidate=tmp_path,
        inventory_stamp=tmp_path / "stamp", inventory_sha256="0" * 64,
        dataset=dataset)
    assert verdict["validated"] is True
    assert verdict["correct"] is True
    assert "private" not in json.dumps(verdict)


def test_main_uses_root_terminal_path_not_run_directory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "run-root"
    root.mkdir()
    output = root / "run-v2"
    output.mkdir()
    pilot, transport, stamp, receipt = (root / name for name in
        ("luna_subscription_pilot.py", "codex_subscription.py", "headless-source-map.json",
         "launch-receipt.json"))
    dataset = tmp_path / "dataset.json"
    for path in (pilot, transport, stamp, receipt, dataset):
        path.write_text("fixture")
    unit = "hymem-luna-lme-v2-20260928-yeds3hl.service"
    cgroup = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit
    receipt_value = {"schema": "luna-lme-launch-v2", "unit": unit,
        "cgroup": cgroup, "candidate": str(tmp_path), "dataset": str(dataset),
        "output": str(output), "dataset_sha256": progress.DATASET_SHA256,
        "model": "gpt-6-luna", "question_limit": 1, "source_question_index": 0,
        "max_turns": 1200, "max_observed_tokens": 4_000_000,
        "max_total_seconds": 5400, "automatic_retry": False,
        "production_changed": False,
        "source_sha256": {"luna_subscription_pilot.py": progress.PILOT_SHA256,
                          "codex_subscription.py": progress.TRANSPORT_SHA256,
                          "headless-source-map.json": "a" * 64}}
    original_read = progress.read_json
    seen = []
    def fake_read(path):
        seen.append(path)
        return receipt_value if path == receipt else original_read(path)
    monkeypatch.setattr(progress, "read_json", fake_read)
    monkeypatch.setattr(progress, "digest", lambda path: {
        pilot: progress.PILOT_SHA256, transport: progress.TRANSPORT_SHA256,
        stamp: "a" * 64, receipt: "b" * 64}[path])
    monkeypatch.setattr(progress, "unit_state", lambda *_: {"available": True})
    monkeypatch.setattr(progress, "terminal_check", lambda **kwargs:
                        {"available": False, "validated": False})
    args = ["--root", str(root), "--unit", unit, "--expected-cgroup", cgroup,
            "--pilot", str(pilot), "--transport", str(transport),
            "--candidate", str(tmp_path), "--inventory-stamp", str(stamp),
            "--inventory-sha256", "a" * 64, "--dataset", str(dataset),
            "--launch-receipt", str(receipt), "--launch-receipt-sha256", "b" * 64]
    assert progress.main(args) == 0
    assert root / "safe-terminal.json" in seen
    assert output / "safe-terminal.json" not in seen
    assert json.loads(capsys.readouterr().out)["files"]["safe-terminal.json"]["present"] is False
