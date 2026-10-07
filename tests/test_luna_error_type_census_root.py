"""Independent closed-output and gate ordering checks for log counts."""
import json
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_error_type_census_v1 as census


def valid():
    return {"schema": census.SCHEMA, "terminal_and_cleanup_verified": True,
            "source_receipt_verified": True, "candidate_source_verified": True,
            "total_call_failure_warnings": 1,
            "error_type_counts": {**dict.fromkeys(census.ERROR_TYPES, 0), "ValueError": 1}}


@pytest.mark.parametrize("name", census.ERROR_TYPES)
@pytest.mark.parametrize("bad", ["PRIVATE-TEXT", {"PRIVATE-KEY": 1}, True, -1])
def test_every_output_counter_is_typed_and_private(monkeypatch, capsys, name, bad):
    payload = valid()
    payload["error_type_counts"][name] = bad
    monkeypatch.setattr(census.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="PRIVATE-STDERR"))
    assert census.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": census.SCHEMA, "status": "metadata_unavailable"}


def test_remote_integrity_rejection_precedes_projection():
    scope = {"SOURCE": "from pathlib import Path\ndef inspect(*args):\n    raise ValueError('rejected')",
             "ROOT": census.v1.ROOT, "RECEIPT_SHA": census.v1.RECEIPT_SHA,
             "V1_PROJECTION": "raise AssertionError('private projection reached')"}
    with pytest.raises(ValueError, match="rejected"):
        exec(census.REMOTE, scope)


def test_no_dynamic_keys_or_additional_text_allowed():
    value = valid()
    assert census._validated(value) == value
    value["error_type_counts"]["PRIVATE-NAME"] = 1
    assert census._validated(value) is None
    value = valid()
    value["private_log_line"] = "PRIVATE-TEXT"
    assert census._validated(value) is None
