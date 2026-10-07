"""Invented two-file warning controls; no pilot host or provider calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_error_type_census_v2 as census


SECRET = "PRIVATE_SENTINEL_DO_NOT_EXPORT"
WARNING = b"chunk_extraction.call_failure error_type="
PREFIX = b"WARNING:hymem.extraction.chunk:"


def projection(*, max_bytes=None, max_line=None, deadline=None):
    scope = {"SCHEMA": census.SCHEMA, "PRIOR_SCHEMA": census.prior.SCHEMA,
        "ERROR_TYPES": census.prior.ERROR_TYPES,
        "FILE_LABELS": census.FILE_LABELS, "FILE_NAMES": census.FILE_NAMES,
        "MAX_LOG_BYTES": max_bytes or census.prior.MAX_LOG_BYTES,
        "MAX_LINE_BYTES": max_line or census.prior.MAX_LINE_BYTES,
        "READ_BYTES": 5, "DEADLINE_SECONDS": deadline or 20}
    exec(compile(census.EXTENSION, "<invented-two-file-census>", "exec"), scope)
    return scope["_extend"]


def fixture(tmp_path, first=b"", second=b""):
    paths = [tmp_path / name for name in census.FILE_NAMES]
    paths[0].write_bytes(first)
    paths[1].write_bytes(second)
    checks = []
    def file_check(path, root, cap):
        checks.append(path.name)
        return (path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root)
            and path.stat().st_size <= cap)
    namespace = {"_file": file_check}
    counts = {name: 0 for name in census.prior.ERROR_TYPES}
    for line in first.splitlines():
        if line == WARNING + b"ValueError":
            counts["ValueError"] += 1
    baseline = {"schema": census.prior.SCHEMA,
        "terminal_and_cleanup_verified": True,
        "source_receipt_verified": True, "candidate_source_verified": True,
        "error_type_counts": counts}
    return paths, namespace, baseline, checks


def test_two_fixed_files_full_line_prefix_and_split_chunks(tmp_path):
    first = b"\n".join((WARNING + b"ValueError",
        PREFIX + WARNING + b"ProgrammingError",
        b"NOTICE:hymem.extraction.chunk:" + WARNING + b"TypeError",
        WARNING + b"RuntimeError private:" + SECRET.encode())) + b"\n"
    second = b"\n".join((PREFIX + WARNING + b"DreamLeaseLost",
        WARNING + SECRET.encode(),
        b"prefix " + PREFIX + WARNING + b"KeyError",
        PREFIX + WARNING + b"ValueError private")) + b"\n"
    paths, namespace, baseline, checks = fixture(tmp_path, first, second)
    result = projection()(tmp_path, namespace, baseline)
    assert checks == list(census.FILE_NAMES)
    assert result["files"]["diagnostic_run"]["total_call_failure_warnings"] == 2
    assert result["files"]["launch_stderr"]["total_call_failure_warnings"] == 2
    assert result["total_call_failure_warnings"] == 4
    assert result["error_type_counts"]["ValueError"] == 1
    assert result["error_type_counts"]["ProgrammingError"] == 1
    assert result["error_type_counts"]["DreamLeaseLost"] == 1
    assert result["error_type_counts"]["other"] == 1
    assert census._validated(result) == result
    assert SECRET not in json.dumps(result)
    assert all(path.read_bytes() for path in paths)


@pytest.mark.parametrize("field", ["terminal_and_cleanup_verified",
    "source_receipt_verified", "candidate_source_verified"])
def test_prior_gate_precedes_either_log_lookup(tmp_path, field):
    _, namespace, baseline, checks = fixture(tmp_path)
    baseline[field] = False
    with pytest.raises(ValueError, match="prior_gate_invalid"):
        projection()(tmp_path, namespace, baseline)
    assert checks == []


def test_baseline_bare_count_mismatch_fails_closed(tmp_path):
    _, namespace, baseline, checks = fixture(tmp_path, WARNING + b"ValueError\n")
    baseline["error_type_counts"]["ValueError"] = 0
    with pytest.raises(ValueError, match="prior_log_changed"):
        projection()(tmp_path, namespace, baseline)
    assert checks == [census.FILE_NAMES[0]]


@pytest.mark.parametrize("index", [0, 1])
def test_each_fixed_file_rejects_symlink_and_size_overflow(tmp_path, index):
    paths, namespace, baseline, _ = fixture(tmp_path)
    outside = tmp_path.parent / ("outside-private-" + str(index))
    outside.write_bytes(b"private")
    paths[index].unlink()
    paths[index].symlink_to(outside)
    with pytest.raises(ValueError, match="log_invalid"):
        projection()(tmp_path, namespace, baseline)
    paths[index].unlink()
    paths[index].write_bytes(b"123456789")
    with pytest.raises(ValueError, match="log_invalid"):
        projection(max_bytes=8)(tmp_path, namespace, baseline)


def test_oversized_line_and_deadline_fail_closed(tmp_path):
    _, namespace, baseline, _ = fixture(tmp_path, b"", b"x" * 100)
    with pytest.raises(ValueError, match="log_line_limit"):
        projection(max_line=64)(tmp_path, namespace, baseline)
    with pytest.raises(ValueError, match="log_deadline"):
        projection(deadline=-1)(tmp_path, namespace, baseline)


def test_output_shape_rejects_private_keys_and_inconsistent_totals(tmp_path):
    _, namespace, baseline, _ = fixture(tmp_path, WARNING + b"ValueError\n")
    result = projection()(tmp_path, namespace, baseline)
    result["files"]["launch_stderr"][SECRET] = 1
    assert census._validated(result) is None
    del result["files"]["launch_stderr"][SECRET]
    result["total_call_failure_warnings"] = 2
    assert census._validated(result) is None


@pytest.mark.parametrize("pin", ["prior", "launcher"])
def test_local_pins_block_remote_call(monkeypatch, capsys, pin):
    if pin == "prior":
        monkeypatch.setattr(census, "PRIOR_SHA", "0" * 64)
    else:
        monkeypatch.setattr(census, "LAUNCHER", Path(__file__))
    monkeypatch.setattr(census.subprocess, "run", lambda *args, **kwargs:
        pytest.fail("remote call must not run"))
    assert census.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": census.SCHEMA, "status": "metadata_unavailable"}


def test_remote_private_output_and_stderr_suppressed(monkeypatch, capsys):
    monkeypatch.setattr(census.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout='{"private":"' + SECRET + '"}',
            stderr=SECRET))
    assert census.main() == 1
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
