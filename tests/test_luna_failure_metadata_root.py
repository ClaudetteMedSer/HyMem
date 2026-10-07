"""Independent root privacy controls for the terminal-only metadata reader."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_instrumented_failure_metadata_v1 as metadata


def valid():
    return {"schema": metadata.SCHEMA, "terminal_and_cleanup_verified": True,
        "source_receipt_verified": True, "questions": {f"q-{i:04d}": {
            "checkpoint_failure_code": "unspecified_failure", "private_row_present": False,
            "indexing": {"present": True, "object": True, "outcome": "failure",
                "complete": False, "failure_code": "cycle_exception",
                "exception_type": "ValueError", "cycles": 1},
            "diagnostic_indexing": {"present": False}} for i in range(4)}}


@pytest.mark.parametrize("field", ["object", "status", "outcome", "complete",
    "healthy", "summary_healthy", "admitted", "cycles", "max_cycles", "elapsed_s",
    "timeout_s", "quarantined_chunks", "summary_degraded_sessions",
    "summary_missing_sessions", "kind", "failure_code", "exception_type",
    "cleanup_error_count"])
@pytest.mark.parametrize("bad", ["PRIVATE-SENTINEL", ["PRIVATE-SENTINEL"],
                                {"PRIVATE-SENTINEL": 1}])
def test_no_field_can_export_arbitrary_text(monkeypatch, capsys, field, bad):
    value = valid()
    value["questions"]["q-0000"]["indexing"][field] = bad
    monkeypatch.setattr(metadata.subprocess, "run", lambda *a, **kw:
        SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="PRIVATE-STDERR"))
    assert metadata.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": metadata.SCHEMA, "status": "metadata_unavailable"}


def test_real_remote_gate_cannot_reach_private_projection():
    def reject(*args):
        raise ValueError("source_receipt_rejected")
    scope = {"SOURCE": "from pathlib import Path\ndef inspect(*args):\n    raise ValueError('source_receipt_rejected')\n",
        "ROOT": metadata.ROOT, "RECEIPT_SHA": metadata.RECEIPT_SHA,
        "SCHEMA": metadata.SCHEMA, "INDEXING_CODES": metadata.INDEXING_CODES,
        "EXCEPTION_TYPES": metadata.EXCEPTION_TYPES,
        "PROJECTION": "raise AssertionError('private_projection_reached')",
        "json": json}
    with pytest.raises(ValueError, match="source_receipt_rejected"):
        exec(metadata.REMOTE, scope)


def test_indexing_codes_match_frozen_candidate_ast():
    import ast
    from pathlib import Path
    path = Path("/private/tmp/hymem-lme-instrumented-IjmZdT/bundle/candidate/benchmarks/lme_protocol.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_INDEXING_FAILURE_CODES" for t in n.targets))
    assert metadata.INDEXING_CODES == frozenset(ast.literal_eval(node.value.args[0]))


def test_valid_closed_result_is_detached_from_private_keys():
    value = valid()
    assert metadata._validated(value) == value
    scope = {"INDEXING_CODES": metadata.INDEXING_CODES,
             "EXCEPTION_TYPES": metadata.EXCEPTION_TYPES}
    exec(metadata.PROJECTION, scope)
    projected = scope["_fields"]({"failure": {"code": "cycle_exception",
        "exception_type": "ValueError", "message": "PRIVATE-MESSAGE"},
        "reports": ["PRIVATE-TEXT"], "context": "PRIVATE-CONTEXT"})
    assert projected == {"object": True, "failure_code": "cycle_exception",
                         "exception_type": "ValueError"}
