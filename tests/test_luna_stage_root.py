"""Independent frozen-runtime and source-privacy checks for stage profiling."""
import os
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_actual_frozen_empty_verifier_is_counted_without_changing_result():
    frozen = os.environ.get("HYMEM_FROZEN_CANDIDATE")
    if not frozen:
        pytest.skip("HYMEM_FROZEN_CANDIDATE required")
    source = Path(__file__).resolve().parents[1] / "benchmarks/luna_stage_accounting.py"
    script = r'''
import importlib.util,sys,json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
spec=importlib.util.spec_from_file_location("root_stage_frozen",sys.argv[2])
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from hymem.extraction import chunk
class Delegate:
    observed_turns=0
    observed_tokens=0
    usage_complete=True
    budget=SimpleNamespace(halt=lambda code: None)
    def complete(self,request):
        self.observed_turns+=1
        self.observed_tokens+=5
        return '{"triples":[],"markers":[],"complete":true}'
ledger=m.StageLedger(Path(sys.argv[1]))
delegate=Delegate()
profiled=chunk.extract_chunk(ledger.wrap(delegate,"q-0000"),"A quiet afternoon.")
plain=chunk.extract_chunk(Delegate(),"A quiet afternoon.")
assert profiled==plain
stats=ledger.snapshot()["q-0000"]
assert stats["extraction_primary_or_other"]["admitted_turns"]==1
assert stats["extraction_empty_verifier"]["admitted_turns"]==1
assert delegate.observed_turns==2
assert "quiet" not in json.dumps(stats)
assert "triples" not in json.dumps(stats)
bad=m.StageLedger(Path(sys.argv[1]))
bad_delegate=Delegate()
def broken_record(*args,**kwargs): raise ValueError("invented counter fault")
bad.record=broken_record
try: chunk.extract_chunk(bad.wrap(bad_delegate,"q-0000"),"A quiet afternoon.")
except m.StageAccountingStop: pass
else: raise AssertionError("accounting failure was swallowed")
assert bad_delegate.observed_turns==1
print("frozen verifier counted; output unchanged; no text retained")
'''
    result = subprocess.run([sys.executable, "-c", script, frozen, str(source)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "output unchanged" in result.stdout


def test_profiled_five_question_runner_preserves_results_and_reconciles(tmp_path, monkeypatch):
    frozen = os.environ.get("HYMEM_FROZEN_CANDIDATE")
    if not frozen:
        pytest.skip("HYMEM_FROZEN_CANDIDATE required")
    from tools.diagnostics import luna_subscription_lme_profiled as profiled
    source = Path(__file__).with_name("test_luna_warm_root.py")
    spec = importlib.util.spec_from_file_location("root_profiled_coordinator_test", source)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    captured = []
    def invoke(**kwargs):
        kwargs["chunk"] = SimpleNamespace(__file__=str(Path(frozen) / "hymem/extraction/chunk.py"))
        value = profiled.run_campaign(**kwargs)
        captured.append(value)
        return value
    monkeypatch.setattr(fixture, "runner", profiled.warm_runner)
    monkeypatch.setattr(profiled.warm_runner, "run_campaign", invoke)
    fixture.test_five_questions_never_retain_more_than_four_clients(tmp_path, monkeypatch)
    assert len(captured) == 1
    result = captured[0]
    assert result["stage_accounting_reconciled"] is True
    stages = result["stage_accounting"]
    assert set(stages) == {"canary", *[f"q-{index:04d}" for index in range(5)]}
    assert all(set(entry) == {"unclassified"} for entry in stages.values())
    assert sum(entry["unclassified"]["admitted_turns"] for entry in stages.values()) == 6
