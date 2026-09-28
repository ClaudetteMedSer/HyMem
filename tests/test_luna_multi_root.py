"""Independent root controls for the versioned multi-question controller."""
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import threading
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_subscription_lme_multi as multi
from benchmarks import codex_subscription_concurrent as concurrent
from hymem.extraction.llm import LLMRequest


def test_existing_output_is_never_modified(tmp_path, capsys):
    output = tmp_path / "previous-run"
    output.mkdir()
    artifact = output / "private-failure.txt"
    artifact.write_text("previous evidence", encoding="utf-8")
    before = artifact.stat().st_mtime_ns
    args = []
    for name in ("binary", "base-transport", "concurrent-transport", "candidate",
                 "inventory-stamp", "inventory-sha256", "dataset", "dataset-sha256"):
        args.extend(["--" + name, "/unused"])
    for name in ("campaign-turns", "campaign-known-tokens", "campaign-seconds",
                 "question-turns", "question-known-tokens", "question-seconds",
                 "canary-turns", "canary-known-tokens", "canary-seconds",
                 "indexing-seconds"):
        args.extend(["--" + name, "1"])
    args.extend(["--output-dir", str(output)])
    args.extend(["--questions", "2", "--workers", "2"])
    assert multi.main(args) == 1
    assert artifact.read_text(encoding="utf-8") == "previous evidence"
    assert artifact.stat().st_mtime_ns == before
    assert json.loads(capsys.readouterr().out)["stop_code"] == "output_not_fresh"


def test_atomic_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    target = tmp_path / "private-progress.json"
    multi.atomic_private(target, {"revision": 1})
    real_replace = multi.os.replace
    def fail_replace(*_args, **_kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(multi.os, "replace", fail_replace)
    with pytest.raises(OSError):
        multi.atomic_private(target, {"revision": 2})
    assert json.loads(target.read_text()) == {"revision": 1}
    assert not list(tmp_path.glob("*.pending"))
    monkeypatch.setattr(multi.os, "replace", real_replace)
    multi.atomic_private(target, {"revision": 3})
    assert json.loads(target.read_text()) == {"revision": 3}
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("text", ['[{"a":1},,{"a":2}]', '[{"a":1}{"a":2}]', '[{"a":1},]'])
def test_stream_rejects_malformed_selected_sequence(tmp_path, text):
    path = tmp_path / "invented.json"
    path.write_text(text)
    with pytest.raises(multi.CampaignStop):
        multi.first_source_items(path, 2)


def test_actual_frozen_stores_have_no_cross_question_state(tmp_path):
    frozen = os.environ.get("HYMEM_FROZEN_CANDIDATE")
    if not frozen:
        pytest.skip("HYMEM_FROZEN_CANDIDATE required for actual frozen runtime")
    runner = Path(multi.__file__).resolve()
    # Clean process: import the verified runtime before any HyMem modules.
    script = r'''
import importlib.util,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,sys.argv[1])
spec=importlib.util.spec_from_file_location("root_multi_frozen",sys.argv[2])
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
from benchmarks import longmemeval_adapter as lme
assert str(Path(sys.argv[1])) in lme.__file__
class NoInference:
    def complete(self,request): raise AssertionError("no model call allowed")
def run(index):
    directory=Path(sys.argv[3])/str(index)
    directory.mkdir()
    memory=m.make_memory_client(NoInference())
    cls=m.old.make_adapter_class(lme,memory)
    adapter=cls(directory/"question.sqlite",embeddings=False,
                aggregation_nodes=False,episode_granularity=False,
                pipeline_model="gpt-6-luna")
    try:
        adapter.open()
        adapter.ingest_sessions([[{"role":"user","content":"invented-isolation-"+str(index)}]],
              ["same-source-session"],["2024-01-01"],namespace="same-question")
        rows=adapter.hy.conn.execute("SELECT content FROM messages").fetchall()
        assert [r[0] for r in rows]==["invented-isolation-"+str(index)]
        assert adapter.hy.config.db_path==directory/"hymem.sqlite"
        fork=adapter.hy.fork()
        assert fork._llm is memory
        fork.close()
        return adapter.hy._phase1_generation["generation_key"]
    finally: adapter.close()
with ThreadPoolExecutor(2) as pool:
    keys=list(pool.map(run,[0,1]))
assert keys[0]==keys[1]
print("two frozen stores isolated; common producer; no model calls")
'''
    completed = subprocess.run([sys.executable, "-c", script, frozen, str(runner), str(tmp_path)],
                               text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "two frozen stores isolated" in completed.stdout


@pytest.mark.parametrize("failure_on_publish", [1, 2])
def test_progress_failure_cannot_trigger_real_frozen_model_retries(failure_on_publish):
    frozen = os.environ.get("HYMEM_FROZEN_CANDIDATE")
    if not frozen:
        pytest.skip("HYMEM_FROZEN_CANDIDATE required")
    script = r'''
import importlib.util,sys,threading
sys.path.insert(0,sys.argv[1])
spec=importlib.util.spec_from_file_location("root_progress_frozen",sys.argv[2])
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
from hymem.extraction import chunk
class Budget:
    stopped=False
    def halt(self,code): self.stopped=True
class Delegate:
    budget=Budget()
    calls=0
    def complete(self,request):
        self.calls+=1
        return '{"triples":[],"markers":[]}'
delegate=Delegate()
publishes=[0]
def callback():
    publishes[0]+=1
    if publishes[0]==int(sys.argv[3]): raise OSError("invented disk error")
active=set()
client=m._ProgressClient(delegate,callback,active,"q-0",threading.RLock())
try: chunk.extract_chunk(client,"invented source content")
except m.CampaignStop: pass
else: raise AssertionError("write failure was swallowed")
assert delegate.calls==int(sys.argv[3])-1
assert not active and delegate.budget.stopped
print("progress failure escaped without retry")
'''
    completed = subprocess.run([sys.executable, "-c", script, frozen, multi.__file__, str(failure_on_publish)],
                               text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "progress failure escaped" in completed.stdout


@pytest.mark.parametrize("unhealthy_first", [False, True])
def test_three_question_campaign_with_actual_shared_budget(tmp_path, monkeypatch, unhealthy_first):
    """Real coordinator, fake sessions; ensures runner overlap and result semantics."""
    rendezvous = threading.Barrier(2, timeout=3)
    snapshots = []
    closed = []
    session_paths = []
    class Session:
        def __init__(self, binary, cwd, timeout):
            session_paths.append(cwd)
        def close(self):
            pass
    monkeypatch.setattr(concurrent.base, "inspect_preflight", lambda *a, **k: {
        "auth": "chatgpt", "model": "gpt-6-luna", "inference_enabled": False,
        "config_isolation_admitted": True, "_thread_id": "fake",
        "quota_windows": [{"remaining_percent": 80}]})
    def infer(session, thread, user):
        if user in ("q-0", "q-1"):
            rendezvous.wait()
        return "invented", 7
    monkeypatch.setattr(concurrent.base, "_run_turn", infer)
    def factory(key, limits, budget):
        return concurrent.ConcurrentSubscriptionClient("unused", budget, key, limits, session_factory=Session)
    def canary(_canary, _chunk, client, **kw):
        client.complete(LLMRequest("s", "canary"))
        return {"passed": True}
    monkeypatch.setattr(multi.old, "experimental_canary", canary)
    class Adapter:
        def __init__(self, path, **kw):
            self.path = path
            self.db_path = path
            self.hy = SimpleNamespace(dream_status=lambda: {"pending_chunks": 0, "summary_healthy": True})
            self.last_indexing_summary = None
        def open(self):
            with sqlite3.connect(self.path) as conn:
                conn.execute("CREATE TABLE dream_runs (ended_at TEXT, error TEXT)")
                conn.execute("CREATE TABLE run_lock (name TEXT)")
            return self
        def close(self):
            closed.append(self.path)
    monkeypatch.setattr(multi.old, "make_adapter_class", lambda *a: Adapter)
    def evaluate(reader, judge, adapter, question, **kw):
        assert kw["indexing_timeout_s"] == 10800
        assert kw["indexing_require_healthy"] is True
        reader.chat([{"role": "user", "content": f'q-{question["index"]}'}])
        indexing = {"outcome": "success", "healthy": True, "summary_healthy": True}
        if unhealthy_first and question["index"] == 0:
            indexing["healthy"] = False
        adapter.last_indexing_summary = indexing
        return {"indexing": indexing, "correct": question["index"] != 1,
                "benchmark_failure": None, "judge_error": False, "judge_parse_valid": True}
    lme = SimpleNamespace(evaluate_question=evaluate, IndexingConvergenceError=ValueError,
                          DEFAULT_MAX_INPUT_TOKENS=100, DEFAULT_MAX_INPUT_BYTES=1000)
    result = multi.run_campaign(concurrent=concurrent, request_type=LLMRequest,
        canary=None, chunk=None, lme=lme,
        protocol=SimpleNamespace(_validate_versioned_indexing=lambda *a, **kw: True),
        binary="unused", questions=[{"index": n} for n in range(3)], output=tmp_path,
        campaign_limits=concurrent.BudgetLimits(10, 1000, 14400),
        question_limits=concurrent.BudgetLimits(2, 100, 12600),
        canary_limits=concurrent.BudgetLimits(1, 100, 120),
        indexing_timeout_s=10800, workers=2, client_factory=factory,
        progress=lambda value: snapshots.append(json.loads(json.dumps(value))))
    assert result["campaign_stop"] is None
    assert result["budget"]["turns"] == 4
    assert result["budget"]["known_tokens"] == 28
    assert result["usage_complete_now"] is True
    assert result["budget"]["in_flight"] == 0
    assert max(s["active_invocations"] for s in snapshots) == 2
    assert any(s["usage_complete_now"] is False for s in snapshots)
    assert len(closed) == len(set(closed)) == 3
    assert len(session_paths) == len(set(session_paths)) == 4
    assert [q["index"] for q in result["questions"]] == [0, 1, 2]
    assert result["questions"][0]["question_completed"] is not unhealthy_first
    assert result["questions"][1]["question_completed"] is True
    assert result["questions"][1]["correct"] is False
    assert result["questions"][2]["question_completed"] is True


def test_real_frozen_aborted_dream_housekeeping_changes_no_memory(tmp_path):
    frozen = os.environ.get("HYMEM_FROZEN_CANDIDATE")
    if not frozen:
        pytest.skip("HYMEM_FROZEN_CANDIDATE required")
    script = r'''
import sys,importlib.util
from pathlib import Path
sys.path.insert(0,sys.argv[1])
spec=importlib.util.spec_from_file_location("root_frozen_abort",sys.argv[2])
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
from benchmarks import longmemeval_adapter as lme
class StopClient:
    def complete(self,request): raise m.CampaignStop("invented resource stop")
directory=Path(sys.argv[3])
memory=m.make_memory_client(StopClient())
adapter=m.old.make_adapter_class(lme,memory)(directory/"hymem.sqlite",embeddings=False,
    aggregation_nodes=False,episode_granularity=False,pipeline_model="gpt-6-luna")
adapter.open()
try:
    adapter.ingest_sessions([[{"role":"user","content":"The user uses invented app Orchidia every Tuesday."}]],
        ["invented"],["2024-01-01"])
    try: adapter.dream_and_wait(max_cycles=2,timeout=30,require_healthy=True)
    except m.CampaignStop: pass
    else: raise AssertionError("stop did not escape")
    conn=adapter.hy.conn
    assert conn.execute("SELECT COUNT(*) FROM dream_runs WHERE ended_at IS NULL").fetchone()[0]==1
    def memory_rows():
        return {table:[tuple(row) for row in conn.execute("SELECT * FROM "+table)]
                for table in ("messages","chunks","sessions","processed_chunks","kg_claim_observations")}
    before=memory_rows()
    receipt=m.terminalize_owned_dream_run(adapter,directory,abnormal=True)
    assert receipt["terminalized"]==1 and receipt["open_runs_after"]==0 and receipt["active_leases_after"]==0
    assert memory_rows()==before
    assert conn.execute("SELECT error FROM dream_runs").fetchone()[0]=="execution_interrupted:subscription_control"
    again=m.terminalize_owned_dream_run(adapter,directory,abnormal=True)
    assert again["terminalized"]==0
    print("aborted dream terminalized once, no memory changed")
finally: adapter.close()
'''
    completed = subprocess.run([sys.executable, "-c", script, frozen, multi.__file__, str(tmp_path)],
                               text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "no memory changed" in completed.stdout
