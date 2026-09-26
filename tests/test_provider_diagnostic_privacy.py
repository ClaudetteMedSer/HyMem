"""Offline regression coverage for provider diagnostics and watcher control flow."""
import logging
import io
from pathlib import Path
import runpy
import sys

import pytest

from benchmarks import extraction_canary
from hymem.contrib.model_policy import RECOMMENDED_DEEPSEEK_MODEL
from hymem.deadline import DeadlineExceeded, MonotonicDeadline, use_deadline
from hymem.extraction import retry

PRIVATE = "private-provider-body-sk-credential"


class HostileError(Exception):
    def __str__(self):
        raise AssertionError("exception string was inspected")

    def __repr__(self):
        raise AssertionError("exception representation was inspected")


def test_retry_diagnostics_and_final_identity(monkeypatch, caplog, capsys, tmp_path):
    error = type(PRIVATE, (HostileError,), {})(PRIVATE)
    calls, sleeps = [], []
    def fail():
        calls.append(1)
        raise error
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    stream = io.StringIO()
    file_path = tmp_path / "retry.log"
    handlers = [logging.StreamHandler(stream), logging.FileHandler(file_path)]
    for handler in handlers:
        retry.log.addHandler(handler)
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(HostileError) as caught:
            retry.with_retry(fail, attempts=4, base_delay=0.5, max_delay=1, label=error)
    finally:
        for handler in handlers:
            retry.log.removeHandler(handler)
            handler.close()
    assert caught.value is error
    assert len(calls) == 4
    assert sleeps == [0.5, 1, 1]
    assert len(caplog.records) == 3
    captured = capsys.readouterr()
    assert PRIVATE not in stream.getvalue() + file_path.read_text() + captured.out + captured.err
    for record in caplog.records:
        assert PRIVATE not in str(record.__dict__)
        assert record.exc_info is None and record.stack_info is None
        assert len(record.getMessage()) < 100


def test_retry_deadline_and_interrupt(monkeypatch):
    now, calls, sleeps = [0.0], [], []
    def fail():
        calls.append(1)
        raise HostileError(PRIVATE)
    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay
    monkeypatch.setattr(retry.time, "sleep", sleep)
    with use_deadline(MonotonicDeadline(0.75, clock=lambda: now[0])):
        with pytest.raises(DeadlineExceeded):
            retry.with_retry(fail, attempts=5)
    assert len(calls) == 2 and sleeps == [0.5, 0.25]
    interrupt = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        retry.with_retry(lambda: (_ for _ in ()).throw(interrupt))
    assert caught.value is interrupt


def run_watcher(monkeypatch, tmp_path, outcomes, expected_exit):
    log_path = tmp_path / (PRIVATE + ".log")
    calls, sleeps = [], []
    sequence = iter(outcomes)
    def probe(**kwargs):
        calls.append(kwargs)
        outcome = next(sequence)
        if outcome is not None:
            raise outcome
    monkeypatch.setenv("DEEPSEEK_API_KEY", PRIVATE)
    monkeypatch.setattr(sys, "argv", ["lme_canary_watch.py", str(log_path)])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr("os.chdir", lambda path: None)
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    monkeypatch.setattr(extraction_canary, "run_configured_extraction_canary", probe)
    with pytest.raises((SystemExit, KeyboardInterrupt)) as caught:
        runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/lme_canary_watch.py"))
    if expected_exit is None:
        assert isinstance(caught.value, KeyboardInterrupt)
    else:
        assert caught.value.code == expected_exit
    assert all(call["api_key"] == PRIVATE for call in calls)
    assert all(call["base_url"] == "https://api.deepseek.com" and call["thinking"] == "auto" for call in calls)
    assert all(call["model"] == RECOMMENDED_DEEPSEEK_MODEL for call in calls)
    assert sleeps == [600] * max(0, len(calls) - 1)
    return log_path.read_text(), calls


@pytest.mark.parametrize("report", [
    {"failure_reason": PRIVATE, "execution_path": {"prose_claim_exact_context_emissions": PRIVATE, "table_claim_exact_context_emissions": HostileError(PRIVATE)}},
    {"failure_reason": HostileError(PRIVATE), "execution_path": PRIVATE},
    {"failure_reason": "clean_empty", "execution_path": {"prose_claim_exact_context_emissions": 2**10000, "table_claim_exact_context_emissions": True}},
])
def test_watcher_private_and_malformed_failures(monkeypatch, tmp_path, capsys, caplog, report):
    error = extraction_canary.ExtractionCanaryError(PRIVATE, report)
    text, calls = run_watcher(monkeypatch, tmp_path, [error] * 24, 4)
    captured = capsys.readouterr()
    assert PRIVATE not in text + captured.out + captured.err + caplog.text
    assert len(calls) == 24 and "WATCH_EXPIRED" in text
    assert all(len(line) < 200 for line in text.splitlines())


def test_watcher_errors_are_private(monkeypatch, tmp_path, capsys, caplog):
    error = type(PRIVATE, (HostileError,), {})(PRIVATE)
    text, calls = run_watcher(monkeypatch, tmp_path, [error] * 3, 5)
    captured = capsys.readouterr()
    assert PRIVATE not in text + captured.out + captured.err + str([r.__dict__ for r in caplog.records])
    assert len(calls) == 3 and text.count("reason=probe_error") == 3


def test_watcher_reset_and_pass(monkeypatch, tmp_path):
    fail = extraction_canary.ExtractionCanaryError(PRIVATE, {"failure_reason": "clean_empty", "execution_path": {"prose_claim_exact_context_emissions": 3, "table_claim_exact_context_emissions": 4}})
    text, calls = run_watcher(monkeypatch, tmp_path, [HostileError(), HostileError(), fail, HostileError(), HostileError(), None], 0)
    assert len(calls) == 6 and "WATCH_FIRED" in text
    assert "reason=clean_empty prose_em=3 table_em=4" in text


def test_watcher_interrupt(monkeypatch, tmp_path):
    text, calls = run_watcher(monkeypatch, tmp_path, [KeyboardInterrupt()], None)
    assert len(calls) == 1 and "PROBE" not in text
