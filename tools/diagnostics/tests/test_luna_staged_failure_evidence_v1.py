"""Offline finite evidence controls with invented stage responses."""
import json
import subprocess
import sys

from benchmarks import extraction_canary as canary
from hymem.extraction.grounding_v2 import GroundingContext, GroundingSource
from hymem.extraction.triples import Triple
from tools.diagnostics import luna_staged_core_v1 as core
from tools.diagnostics import luna_staged_failure_evidence_v1 as evidence


def _fixture(*, prefix=None, omit_left=False):
    left, right = canary._PROSE_BOUNDARY_LEFT, canary._PROSE_BOUNDARY_RIGHT
    owned = "Introduction.\n" + right + "\nAfterword."
    context = GroundingContext("boundary", "Other text." if omit_left else left,
        len(owned) if prefix is None else prefix)
    source = GroundingSource(9911, owned, (context,))
    triple = Triple("Avery Boundary Canary", "uses", "PostgreSQL", 1,
                    source_message_id=9911)
    _, batch = core.staged.build_original_request((triple,), (source,))
    original = {"schema": core.staged.ORIGINAL_SCHEMA,
        "batch_sha256": batch.batch_sha256, "complete": True,
        "originals": [{"index": 0, "original":
            {"state": "not_established", "support": None}}]}
    raw = json.dumps(original)
    _, alternative_batch = core.staged.build_alternatives_request(batch, raw)
    alternative = {"schema": core.staged.ALTERNATIVES_SCHEMA,
        "batch_sha256": batch.batch_sha256,
        "original_response_sha256": alternative_batch.original_response_sha256,
        "complete": True,
        "alternatives": [{"index": 0, "alternatives": {
            name: {"state": "not_established", "support": None}
            for name in core.v4.PREDICATE_ORDER if name != "uses"}}]}
    events = [{"phase": "before_dispatch"}, {"phase": "response_returned"},
              {"phase": "before_dispatch"},
              {"phase": "response_returned", "response": json.dumps(alternative)}]
    return batch, raw, events


def test_prose_cue_bound_and_mechanically_correctable():
    batch, raw, events = _fixture()
    result = evidence._prose(core, canary, batch, raw, events)
    assert result["left_cue_in_bound_context"]
    assert result["right_cue_in_owned"]
    assert result["owned_quote_within_context_prefix"]
    assert result["mechanical_candidate_valid"]
    assert result["mechanical_candidate_accepted"]
    assert not any(text in json.dumps(result) for text in
                   (canary._PROSE_BOUNDARY_LEFT, canary._PROSE_BOUNDARY_RIGHT,
                    "Avery Boundary Canary", "PostgreSQL"))


def test_prefix_or_left_cue_mutation_blocks_candidate():
    for options in ({"prefix": 1}, {"omit_left": True}):
        batch, raw, events = _fixture(**options)
        result = evidence._prose(core, canary, batch, raw, events)
        assert not result["mechanical_candidate_accepted"]


def test_cli_failure_emits_only_finite_fields(tmp_path):
    run = subprocess.run([sys.executable, "-I", "-B", evidence.__file__,
        "--root", str(tmp_path / "MISSING_PRIVATE"),
        "--receipt-sha256", "0" * 64, "--entry-sha256", "0" * 64,
        "--result-sha256", "0" * 64], capture_output=True, text=True, timeout=15)
    assert run.returncode == 1 and run.stderr == ""
    result = json.loads(run.stdout)
    assert result == {"schema": evidence.SCHEMA, "verified": False,
                      "new_model_calls": 0, "reason": "finite_evidence_unverified"}
