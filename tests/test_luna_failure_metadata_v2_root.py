"""Independent privacy checks for the terminal final-status extension."""
import json
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_failure_metadata_v2 as reader


def report():
    final = {
        "pending": dict.fromkeys(reader.PENDING, 0),
        "malformed": dict.fromkeys(reader.MALFORMED, 0),
        "quarantined": dict.fromkeys(reader.QUARANTINED, 0),
        "terminal_loss": {"chunks": 0},
        "coverage_integrity": {"failures": 0},
        "summary_health": {**dict.fromkeys(reader.SUMMARY_COUNTS, 0),
                           "summary_healthy": True},
    }
    return {"schema": reader.SCHEMA, "terminal_and_cleanup_verified": True,
            "source_receipt_verified": True, "questions": {
                f"q-{index:04d}": {"checkpoint_failure_code": "unspecified_failure",
                    "private_row_present": False,
                    "diagnostic_indexing": {"present": False},
                    "indexing": {"present": True, "object": True,
                                 "final_status": final}}
                for index in range(4)}}


FIELDS = [(group, key) for group, keys in (
    ("pending", reader.PENDING), ("malformed", reader.MALFORMED),
    ("quarantined", reader.QUARANTINED), ("terminal_loss", ("chunks",)),
    ("coverage_integrity", ("failures",)), ("summary_health", reader.SUMMARY_COUNTS))
    for key in keys]


@pytest.mark.parametrize("group,key", FIELDS)
@pytest.mark.parametrize("invalid", ["PRIVATE-TEXT", {"PRIVATE-KEY": 1}, True, -1])
def test_every_numeric_field_rejects_noncount(monkeypatch, capsys, group, key, invalid):
    payload = report()
    payload["questions"]["q-0001"]["indexing"]["final_status"][group][key] = invalid
    monkeypatch.setattr(reader.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="PRIVATE-ERR"))
    assert reader.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": reader.SCHEMA, "status": "metadata_unavailable"}


def test_no_projection_executes_if_pinned_reader_rejects():
    scope = {"SOURCE": "from pathlib import Path\ndef inspect(*args):\n    raise ValueError('rejected')",
             "ROOT": reader.v1.ROOT, "RECEIPT_SHA": reader.v1.RECEIPT_SHA,
             "INDEXING_CODES": (), "EXCEPTION_TYPES": (),
             "V1_SCHEMA": reader.v1.SCHEMA,
             "V1_PROJECTION": "raise AssertionError('projection reached')"}
    with pytest.raises(ValueError, match="rejected"):
        exec(reader.REMOTE, scope)


def test_unknown_group_and_question_fail_closed():
    valid = report()
    assert reader._validated(valid) == valid
    valid["questions"]["q-0001"]["indexing"]["final_status"]["private"] = "PRIVATE"
    assert reader._validated(valid) is None
    valid = report()
    valid["questions"]["PRIVATE-ID"] = valid["questions"].pop("q-0001")
    assert reader._validated(valid) is None
