"""Independent parent controls for the complete watcher/retry logging path."""
import logging
from pathlib import Path
import runpy
import sys

import pytest

from benchmarks import extraction_canary
from hymem.contrib.model_policy import RECOMMENDED_DEEPSEEK_MODEL
from hymem.extraction import retry


def test_watcher_real_retry_keeps_private_exception_out_of_all_logs(
    monkeypatch, tmp_path, capsys, caplog,
):
    marker = "SYNTHETIC_PRIVATE_PROVIDER_BODY_NOT_A_REAL_KEY"

    class PrivateError(RuntimeError):
        def __str__(self):
            raise AssertionError("private exception was stringified")

        def __repr__(self):
            raise AssertionError("private exception was represented")

    error = PrivateError(marker)
    probes, attempts, sleeps = [], [], []

    def fail():
        attempts.append(1)
        raise error

    def probe(**kwargs):
        probes.append(kwargs)
        return retry.with_retry(fail, attempts=3, base_delay=0, label=marker)

    log_path = tmp_path / (marker + ".log")
    monkeypatch.setenv("DEEPSEEK_API_KEY", marker)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["watcher", str(log_path)])
    monkeypatch.setattr("os.chdir", lambda _: None)
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    monkeypatch.setattr(extraction_canary, "run_configured_extraction_canary", probe)
    disabled_before = logging.root.manager.disable
    with caplog.at_level(logging.WARNING), pytest.raises(SystemExit) as exit_:
        runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/lme_canary_watch.py"))
    assert exit_.value.code == 5
    assert logging.root.manager.disable == disabled_before
    assert len(probes) == 3 and len(attempts) == 9
    assert sleeps == [0, 0, 600, 0, 0, 600, 0, 0]
    assert all(call == {
        "api_key": marker, "base_url": "https://api.deepseek.com",
        "model": RECOMMENDED_DEEPSEEK_MODEL, "thinking": "auto",
    } for call in probes)
    captured = capsys.readouterr()
    text = log_path.read_text()
    assert text.count("reason=probe_error") == 3 and "WATCH_ERROR_STREAK" in text
    assert len(caplog.records) == 6
    assert marker not in text + captured.out + captured.err + repr([
        record.__dict__ for record in caplog.records
    ])
    assert all(record.exc_info is None and record.stack_info is None for record in caplog.records)


def test_retry_logs_without_muting_other_loggers(monkeypatch, caplog):
    error = RuntimeError("SYNTHETIC_PRIVATE_PROVIDER_BODY")
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    disabled_before = logging.root.manager.disable
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as caught:
            retry.with_retry(lambda: (_ for _ in ()).throw(error), attempts=2)
        logging.getLogger("synthetic.other.component").warning("ordinary health event")
    assert caught.value is error
    assert logging.root.manager.disable == disabled_before
    assert "ordinary health event" in caplog.text
    assert "SYNTHETIC_PRIVATE_PROVIDER_BODY" not in caplog.text
    assert len([record for record in caplog.records if record.name == retry.log.name]) == 1
