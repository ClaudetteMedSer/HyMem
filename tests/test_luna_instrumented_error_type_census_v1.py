"""Invented local log controls; no pilot host or provider is contacted."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_error_type_census_v1 as census


SECRET = "PRIVATE_SENTINEL_DO_NOT_EXPORT"


def projection(*, max_bytes=None, max_line=None, deadline=None):
    scope = {"V1_SCHEMA": census.v1.SCHEMA, "SCHEMA": census.SCHEMA,
        "CHUNK_SHA": census.CHUNK_SHA, "ERROR_TYPES": census.ERROR_TYPES,
        "MAX_LOG_BYTES": max_bytes or census.MAX_LOG_BYTES,
        "MAX_LINE_BYTES": max_line or census.MAX_LINE_BYTES,
        "READ_BYTES": 7, "DEADLINE_SECONDS": deadline or census.DEADLINE_SECONDS}
    exec(compile(census.PROJECTION, "<invented-log-census>", "exec"), scope)
    return scope["_census"]


def fixture(tmp_path, log):
    candidate = tmp_path / "candidate/hymem/extraction/chunk.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("invented source")
    path = tmp_path / "private-diagnostic-run.log"
    path.write_bytes(log)
    reads = []
    def file_check(path, root, cap):
        reads.append(path.name)
        return (path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root)
                and path.stat().st_size <= cap)
    namespace = {"_file": file_check, "_sha": lambda _: census.CHUNK_SHA}
    base = {"schema": census.v1.SCHEMA, "terminal_and_cleanup_verified": True,
        "source_receipt_verified": True}
    return path, namespace, base, reads


def test_full_line_fixed_counters_and_private_text_never_exported(tmp_path):
    warning = b"chunk_extraction.call_failure error_type="
    path, namespace, base, _ = fixture(tmp_path, b"\n".join([
        warning + b"ValueError", warning + b"ProgrammingError",
        warning + b"DreamLeaseLost", warning + SECRET.encode(),
        b"prefix " + warning + b"TypeError",
        warning + b"RuntimeError private:" + SECRET.encode(),
        b"WARNING:hymem.extraction.chunk:" + warning + b"KeyError",
        b"unrelated private " + SECRET.encode(),
    ]) + b"\n")
    result = projection()(tmp_path, namespace, base)
    assert result["total_call_failure_warnings"] == 4
    assert result["error_type_counts"]["ValueError"] == 1
    assert result["error_type_counts"]["ProgrammingError"] == 1
    assert result["error_type_counts"]["DreamLeaseLost"] == 1
    assert result["error_type_counts"]["other"] == 1
    assert set(result["error_type_counts"]) == set(census.ERROR_TYPES)
    assert SECRET not in json.dumps(result)
    assert census._validated(result) == result
    assert path.read_bytes().count(warning) == 7


@pytest.mark.parametrize("change", [
    {"terminal_and_cleanup_verified": False},
    {"source_receipt_verified": False},
    {"schema": "wrong"},
])
def test_v1_gate_fails_before_log_lookup(tmp_path, change):
    _, namespace, base, reads = fixture(tmp_path, b"private")
    base.update(change)
    with pytest.raises(ValueError, match="v1_gate_invalid"):
        projection()(tmp_path, namespace, base)
    assert reads == []


def test_source_pin_fails_before_log_lookup(tmp_path):
    _, namespace, base, reads = fixture(tmp_path, b"private")
    namespace["_sha"] = lambda _: "wrong"
    with pytest.raises(ValueError, match="candidate_source_invalid"):
        projection()(tmp_path, namespace, base)
    assert reads == ["chunk.py"]


def test_symlink_outside_and_oversized_logs_rejected(tmp_path):
    path, namespace, base, _ = fixture(tmp_path, b"private")
    outside = tmp_path.parent / "outside-private-log"
    outside.write_bytes(b"private")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="log_invalid"):
        projection()(tmp_path, namespace, base)
    path.unlink()
    path.write_bytes(b"123456789")
    with pytest.raises(ValueError, match="log_invalid"):
        projection(max_bytes=8)(tmp_path, namespace, base)


def test_oversized_line_fails_without_partial_result(tmp_path):
    _, namespace, base, _ = fixture(tmp_path,
        b"chunk_extraction.call_failure error_type=ValueError\n" + b"x" * 100)
    with pytest.raises(ValueError, match="log_line_limit"):
        projection(max_line=64)(tmp_path, namespace, base)


def test_deadline_fails_closed(tmp_path):
    _, namespace, base, _ = fixture(tmp_path, b"chunk_extraction.call_failure error_type=ValueError\n")
    with pytest.raises(ValueError, match="log_deadline"):
        projection(deadline=-1)(tmp_path, namespace, base)


def test_output_validation_rejects_dynamic_keys_and_bad_totals(tmp_path):
    _, namespace, base, _ = fixture(tmp_path, b"chunk_extraction.call_failure error_type=ValueError\n")
    result = projection()(tmp_path, namespace, base)
    result["error_type_counts"][SECRET] = 1
    assert census._validated(result) is None
    del result["error_type_counts"][SECRET]
    result["total_call_failure_warnings"] = 2
    assert census._validated(result) is None


def test_local_pin_blocks_remote_call(monkeypatch, capsys):
    monkeypatch.setattr(census, "CHUNK", Path(__file__))
    monkeypatch.setattr(census.subprocess, "run", lambda *args, **kwargs:
        pytest.fail("remote call must not run"))
    assert census.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": census.SCHEMA, "status": "metadata_unavailable"}


def test_remote_private_stdout_and_stderr_suppressed(monkeypatch, capsys):
    monkeypatch.setattr(census.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout='{"private":"' + SECRET + '"}',
            stderr=SECRET))
    assert census.main() == 1
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
