"""Independent checks for finite two-file log metadata output."""
import copy
import json
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_error_type_census_v2 as reader


def valid():
    zero = {"total_call_failure_warnings": 0,
            "error_type_counts": dict.fromkeys(reader.prior.ERROR_TYPES, 0)}
    return {"schema": reader.SCHEMA, "terminal_and_cleanup_verified": True,
            "source_receipt_verified": True, "candidate_source_verified": True,
            **copy.deepcopy(zero),
            "files": {key: copy.deepcopy(zero) for key in reader.FILE_LABELS}}


@pytest.mark.parametrize("label", reader.FILE_LABELS)
@pytest.mark.parametrize("field", reader.prior.ERROR_TYPES)
def test_all_perfile_types_are_closed(monkeypatch, capsys, label, field):
    payload = valid()
    payload["files"][label]["error_type_counts"][field] = "PRIVATE-TEXT"
    monkeypatch.setattr(reader.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="PRIVATE-STDERR"))
    assert reader.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": reader.SCHEMA, "status": "metadata_unavailable"}


def test_filenames_cannot_be_supplied_by_private_metadata():
    assert reader.FILE_NAMES == ("private-diagnostic-run.log", "private-launch-stderr.log")
    value = valid()
    assert reader._validated(value) == value
    value["files"]["PRIVATE-FILENAME"] = value["files"].pop("diagnostic_run")
    assert reader._validated(value) is None


def test_aggregate_must_reconcile_each_error_type():
    value = valid()
    value["files"]["launch_stderr"]["error_type_counts"]["ValueError"] = 1
    value["files"]["launch_stderr"]["total_call_failure_warnings"] = 1
    assert reader._validated(value) is None
    value["error_type_counts"]["ValueError"] = 1
    value["total_call_failure_warnings"] = 1
    assert reader._validated(value) == value
