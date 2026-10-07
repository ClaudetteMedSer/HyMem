"""Focused offline checks for staged replay's private boundary and stage grammar."""
from __future__ import annotations

import json

import pytest

from tools.diagnostics import luna_semantic_cases as cases
from tools.diagnostics import luna_staged_core_v1 as core
from tools.diagnostics import luna_staged_replay_v1 as replay


def _first_stage(raw="private malformed response"):
    unit = core.schedule()[0]
    triples, sources, _ = core._unit_input(unit, cases.cases(), ())
    request, batch = core.staged.build_original_request(triples, sources)
    before = replay._expected_before(core, request, batch, "original", False)
    returned = {"phase": "response_returned", "stage": "original",
                "recheck": False, "response": raw}
    return unit, triples, sources, before, returned


def test_exact_nested_json_types_and_stage_prefix():
    unit, triples, sources, before, returned = _first_stage()
    corrupted = json.loads(json.dumps(before))
    corrupted["recheck"] = 0
    with pytest.raises(ValueError):
        replay._same(corrupted, before, "type_alias")
    evaluated, stages, responses = replay._evaluate(
        core, unit, triples, sources, [before], complete=False)
    assert (evaluated, stages, responses) == (None, 1, 0)
    evaluated, stages, responses = replay._evaluate(
        core, unit, triples, sources, [before, returned], complete=False)
    assert (stages, responses) == (1, 1)
    assert evaluated["outcome"] == "malformed"
    with pytest.raises(ValueError):
        replay._evaluate(core, unit, triples, sources,
                         [before, returned, before], complete=True)


def test_alternatives_request_is_bound_to_private_original_response():
    unit, triples, sources, before, _ = _first_stage()
    request, batch = core.staged.build_original_request(triples, sources)
    raw = json.dumps({"schema": core.staged.ORIGINAL_SCHEMA,
        "batch_sha256": batch.batch_sha256, "complete": True,
        "originals": [{"index": 0, "original": {
            "state": "not_established", "support": None}}]})
    alt_request, alt_batch = core.staged.build_alternatives_request(batch, raw)
    alt_before = replay._expected_before(core, alt_request, alt_batch,
                                         "alternatives", False)
    events = [before, {"phase": "response_returned", "stage": "original",
                       "recheck": False, "response": raw}, alt_before]
    assert replay._evaluate(core, unit, triples, sources, events,
                            complete=False)[1:] == (2, 1)
    altered = json.loads(json.dumps(alt_before))
    altered["prior_response_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        replay._evaluate(core, unit, triples, sources,
                         events[:-1] + [altered], complete=False)


def test_entry_pin_fails_before_any_import(tmp_path):
    entry = tmp_path / "code/tools/diagnostics/luna_staged_run_v1.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("raise AssertionError('untrusted code executed')\n")
    with pytest.raises(ValueError, match="entry_pin_invalid"):
        replay.replay(tmp_path, "1" * 64, "0" * 64)
