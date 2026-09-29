"""Offline controls for the three-file R7 target receipt fetcher."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from tools.diagnostics import lme_r7_target_receipts as receipts


def _payload():
    raw = {
        "receipt.json": b'{"gate_passed":true,"private":"RAW_RECEIPT_SENTINEL"}',
        "supervisor.json": b'{"status":"passed","private":"RAW_SUPERVISOR_SENTINEL"}',
        "junit.xml": b'<testsuite>RAW_JUNIT_SENTINEL</testsuite>',
    }
    entries = {
        name: {
            "sha256": hashlib.sha256(value).hexdigest(),
            "bytes": len(value),
            "base64": base64.b64encode(value).decode("ascii"),
        }
        for name, value in raw.items()
    }
    return raw, entries


@pytest.fixture
def destination(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "docs" / "patches").mkdir(parents=True)
    monkeypatch.setattr(receipts, "REPO", root)
    return root / "docs" / "patches"


def _local_files(destination):
    return sorted(destination.iterdir())


def test_source_compiles_without_executing_or_contacting_remote():
    source = receipts.__file__
    with open(source, encoding="utf-8") as stream:
        compile(stream.read(), source, "exec")
    compile("ROOT=" + repr(receipts.ROOT) + "\n" + receipts.REMOTE, "<remote>", "exec")


def test_success_writes_only_allowlisted_verified_bytes_and_metadata(
    destination, monkeypatch, capsys,
):
    raw, entries = _payload()
    calls = []

    def ssh(command, *, capture_output, timeout):
        calls.append((command, capture_output, timeout))
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(entries).encode(),
            stderr=b"RAW_STDERR_SENTINEL",
        )

    monkeypatch.setattr(receipts.subprocess, "run", ssh)
    receipts.main()

    assert len(calls) == 1
    command, captured, timeout = calls[0]
    assert command[0] == "ssh" and command[-2] == "afrodite"
    assert captured is True and timeout == 120
    assert len(_local_files(destination)) == len(receipts.FILES)
    for name, value in raw.items():
        assert (destination / f"2026-09-25-lme-r7-target-{name}").read_bytes() == value
    output = capsys.readouterr()
    metadata = json.loads(output.out)
    assert metadata == {
        name: {"sha256": entry["sha256"], "bytes": entry["bytes"]}
        for name, entry in entries.items()
    }
    assert output.err == ""
    assert "RAW_" not in output.out
    assert "base64" not in output.out


def test_corrupt_hash_rejects_all_receipts(destination, monkeypatch, capsys):
    _, entries = _payload()
    entries["junit.xml"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        receipts.subprocess, "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout=json.dumps(entries).encode(), stderr=b"RAW_SECRET"
        ),
    )
    with pytest.raises(AssertionError):
        receipts.main()
    assert _local_files(destination) == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_exact_file_allowlist_rejects_extra_or_missing_entry(
    destination, monkeypatch, capsys, change,
):
    _, entries = _payload()
    if change == "extra":
        entries["unapproved.txt"] = entries["junit.xml"]
    else:
        del entries["junit.xml"]
    monkeypatch.setattr(
        receipts.subprocess, "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout=json.dumps(entries).encode(), stderr=b"RAW_SECRET"
        ),
    )
    with pytest.raises(AssertionError):
        receipts.main()
    assert _local_files(destination) == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_transport_failure_writes_nothing_and_hides_remote_output(
    destination, monkeypatch, capsys, failure,
):
    def ssh(command, *, capture_output, timeout):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout, output=b"RAW_SECRET")
        return SimpleNamespace(returncode=255, stdout=b"RAW_SECRET", stderr=b"RAW_SECRET")

    monkeypatch.setattr(receipts.subprocess, "run", ssh)
    with pytest.raises(SystemExit, match="receipt_download_"):
        receipts.main()
    assert _local_files(destination) == []
    output = capsys.readouterr()
    assert "RAW_SECRET" not in output.out + output.err


def test_existing_output_refuses_before_ssh(destination, monkeypatch):
    existing = destination / "2026-09-25-lme-r7-target-supervisor.json"
    existing.write_bytes(b"KEEP_EXISTING")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("SSH was called despite an existing output")

    monkeypatch.setattr(receipts.subprocess, "run", forbidden)
    with pytest.raises(AssertionError):
        receipts.main()
    assert existing.read_bytes() == b"KEEP_EXISTING"
    assert _local_files(destination) == [existing]
