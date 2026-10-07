"""Independent offline controls for the noncanonical diagnostic runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_lme_diagnostic_v1 as runner


def _limits(turns: int, tokens: int, seconds: int) -> SimpleNamespace:
    return SimpleNamespace(turns=turns, known_tokens=tokens, seconds=seconds)


@pytest.mark.parametrize("verdict", [False, None, 0])
def test_containment_must_prove_true_before_dataset_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: object,
) -> None:
    output = tmp_path / "unstarted"
    monkeypatch.setattr(runner, "_sha", lambda _path: pytest.fail(
        "dataset/source was read after containment rejected"))
    with pytest.raises(ValueError, match="containment_unverified"):
        runner.run_campaign(
            {"questions": [{"question_id": "q1"}]}, output=output,
            campaign_limits=_limits(100, 1000, 100),
            canary_limits=_limits(10, 100, 20),
            question_limits=_limits(50, 500, 50),
            indexing_seconds=10, workers=1,
            helper_sha256="0" * 64, containment=lambda _loaded: verdict,
        )
    assert not output.exists()


def test_staged_canary_source_check_rejects_empty_and_wrong_context() -> None:
    expected = {11: "alpha table header\nalpha row", 12: "beta prose context"}
    base = {"source_ids": [11], "source_contents": ["alpha row"],
            "source_contexts": [["alpha table header"]]}
    ordinary_contexts = {(11, "alpha row"): ("alpha table header",)}
    assert runner._stage_source_context_bound([base], expected, set(expected), ordinary_contexts)
    for change in (
        {"source_contents": [""]},
        {"source_contexts": [[""]]},
        {"source_contexts": [["unrelated"]]},
        {"source_contexts": [[]]},
        {"source_ids": [12]},
        {"source_contents": ["alpha row", "beta prose"]},
    ):
        record = {**base, **change}
        assert not runner._stage_source_context_bound([record], expected, set(expected), ordinary_contexts)
    assert not runner._stage_source_context_bound([], expected, set(expected), ordinary_contexts)
    assert not runner._stage_source_context_bound([base], expected, set(expected), None)


def test_import_origins_bind_module_object_to_exact_pinned_path(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    code = tmp_path / "code"
    candidate.mkdir()
    code.mkdir()
    expected = candidate / "benchmarks" / "strictness.py"
    expected.parent.mkdir()
    expected.write_text("# pinned\n")
    wrong = code / "benchmarks" / "strictness.py"
    wrong.parent.mkdir()
    wrong.write_text("# shadow\n")
    runner._verify_import_origins(((SimpleNamespace(__file__=str(expected)), expected),))
    with pytest.raises(ValueError, match="import_origin_invalid"):
        runner._verify_import_origins(((SimpleNamespace(__file__=str(wrong)), expected),))
    with pytest.raises(ValueError, match="import_origin_invalid"):
        runner._verify_import_origins(((SimpleNamespace(), expected),))


def test_failed_question_keeps_full_denominator_and_nonzero_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No model client exists: only the coordinator's terminal logic runs."""
    captured: dict[str, object] = {}
    dataset = tmp_path / "source.jsonl"
    dataset.write_text("{}\n")

    class Budget:
        def __init__(self, *_args, **_kwargs):
            self.stop_code = None

        def halt(self, code):
            self.stop_code = code

        def snapshot(self):
            return {"stopped": self.stop_code is not None,
                    "stop_code": self.stop_code, "reserved": 0,
                    "in_flight": 0, "usage_complete": True}

    class Checkpoint:
        def __init__(self, *_args, **_kwargs):
            self.rows = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def record(self, qid, *, row, failure=None):
            self.rows[qid] = {"row": row, "failure": failure}

        def finalize(self):
            return {"counts": {"completed": sum(x["row"] is not None
                                               for x in self.rows.values()),
                               "failed": sum(x["row"] is None
                                             for x in self.rows.values())}}

    def content_hash(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    loaded = {
        "questions": [{"question_id": "q1"}], "dataset": dataset,
        "warm": SimpleNamespace(SharedBudget=Budget),
        "strictness": SimpleNamespace(AtomicCheckpoint=Checkpoint,
                                      content_hash=content_hash),
        "diagnostic": SimpleNamespace(MODE="semantic_diagnostic_v1"),
        "prior": SimpleNamespace(atomic_private=lambda path, value:
                                 captured.__setitem__(str(path), value)),
    }
    monkeypatch.setattr(runner, "_sha", lambda path:
        runner.DATASET_SHA256 if Path(path) == dataset else "1" * 64)
    monkeypatch.setattr(runner, "run_live_canary", lambda *_args: {
        "structural_valid": True, "model_gold_match": True,
        "quality_failure_reason": None, "semantic_failure_proved": False})
    def failed_worker(_loaded, budget, *_args):
        budget.halt("question_failure")
        return {"projection": None, "accounting": None,
                "stop_code": "question_failure"}
    monkeypatch.setattr(runner, "_question_worker", failed_worker)
    result = runner.run_campaign(
        loaded, output=tmp_path / "diagnostic-output",
        campaign_limits=_limits(100, 1000, 100),
        canary_limits=_limits(10, 100, 20),
        question_limits=_limits(50, 500, 50),
        indexing_seconds=10, workers=1,
        helper_sha256="0" * 64, containment=lambda _loaded: True,
    )
    assert result["selected_denominator"] == 1
    assert result["scored_count"] == 0
    assert result["failed_or_unscored_count"] == 1
    assert result["quality_accuracy_full_selected"] is None
    assert result["campaign_stop"] == "question_failure"
    assert result["checkpoint_counts"] == {"completed": 0, "failed": 1}
