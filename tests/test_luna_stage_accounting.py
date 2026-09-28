from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

import pytest

from benchmarks import luna_stage_accounting as stage


class Frame:
    def __init__(self, root, relative, function, line, back=None):
        self.f_code = types.SimpleNamespace(
            co_filename=str(root / relative), co_name=function)
        self.f_lineno = line
        self.f_back = back


def ledger(tmp_path):
    result = stage.StageLedger.__new__(stage.StageLedger)
    result.candidate = tmp_path
    result._lock = threading.RLock()
    result._data = {}
    return result


@pytest.mark.parametrize(("stack", "expected"), [
    ([("hymem/extraction/chunk.py", "single_attempt", 2955)], "extraction_primary_or_other"),
    ([("hymem/extraction/chunk.py", "single_attempt", 2955),
      ("hymem/extraction/chunk.py", "attempt", 3152)], "extraction_contract_repair"),
    ([("hymem/extraction/chunk.py", "single_attempt", 2955),
      ("hymem/extraction/chunk.py", "attempt", 3140),
      ("hymem/extraction/chunk.py", "recover", 3309)], "extraction_terminal_retry"),
    ([("hymem/extraction/chunk.py", "single_attempt", 2955),
      ("hymem/extraction/chunk.py", "recover", 3254)], "extraction_empty_verifier"),
    ([("hymem/extraction/chunk.py", "single_attempt", 2955),
      ("hymem/extraction/chunk.py", "verify_nonempty", 3201)], "extraction_omission_verifier"),
    ([("hymem/dreaming/digest.py", "extract_session_digest", 711)], "digest_primary"),
    ([("hymem/dreaming/digest.py", "extract_session_digest", 742)], "digest_summary_repair"),
    ([("benchmarks/longmemeval_adapter.py", "answer_question_raw", 2254)], "reader"),
    ([("benchmarks/longmemeval_adapter.py", "judge_answer_raw", 2615)], "judge"),
    ([("benchmarks/longmemeval_adapter.py", "other", 2615)], "unclassified"),
])
def test_fixed_callsite_labels(tmp_path, stack, expected):
    frame = None
    for relative, function, line in reversed(stack):
        frame = Frame(tmp_path, relative, function, line, frame)
    assert stage.classify_stack(tmp_path, frame) == expected


class Budget:
    def __init__(self):
        self.stops = []
    def halt(self, code):
        self.stops.append(code)


class Delegate:
    def __init__(self, mode="return"):
        self.mode = mode
        self.observed_turns = 0
        self.observed_tokens = 0
        self.usage_complete = True
        self.budget = Budget()
        self.closed = False
    def complete(self, request):
        if self.mode == "before_admission":
            raise ValueError("private response text")
        self.observed_turns += 1
        if self.mode == "interrupt":
            self.usage_complete = False
            raise KeyboardInterrupt("private response text")
        if self.mode == "unknown_usage":
            self.usage_complete = False
            return "private response text"
        self.observed_tokens += 7
        return "private response text"
    def close(self):
        self.closed = True


@pytest.mark.parametrize(("mode", "turns", "tokens", "returned"), [
    ("return", 1, 7, 1),
    ("before_admission", 0, 0, 0),
    ("unknown_usage", 1, 0, 1),
    ("interrupt", 1, 0, 0),
])
def test_one_record_on_each_exit(tmp_path, mode, turns, tokens, returned):
    book = ledger(tmp_path)
    delegate = Delegate(mode)
    client = book.wrap(delegate, "q-0000")
    if mode in ("before_admission", "interrupt"):
        with pytest.raises(BaseException):
            client.complete("private request text")
    else:
        assert client.complete("private request text") == "private response text"
    client.close()
    assert delegate.closed
    value = book.snapshot()["q-0000"]["unclassified"]
    assert value["attempted_calls"] == 1
    assert value["admitted_turns"] == turns
    assert value["known_tokens"] == tokens
    assert value["returned_calls"] == returned
    assert value["rejected_calls"] == 1 - returned
    assert value["usage_complete"] is (mode in ("return", "before_admission"))
    assert "private" not in repr(book.snapshot())


def test_concurrent_per_question_reconciliation(tmp_path):
    book = ledger(tmp_path)
    clients = {key: book.wrap(Delegate(), key) for key in ("q-0000", "q-0001")}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda key: clients[key].complete("private"), clients))
    assert results == ["private response text"] * 2
    budget = {"turns": 2, "known_tokens": 14, "questions": {
        key: {"turns": 1, "known_tokens": 7} for key in clients}}
    assert book.reconcile(budget)
    budget["known_tokens"] = 15
    assert not book.reconcile(budget)


def test_candidate_mapping_drift_fails_closed(tmp_path):
    for relative in stage.SOURCE_SHA256:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="stage_callsite_source_drift"):
        stage.StageLedger(tmp_path)


def test_accounting_fault_is_base_exception_stop(tmp_path):
    book = ledger(tmp_path)
    book.record = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("private"))
    delegate = Delegate()
    with pytest.raises(stage.StageAccountingStop, match="stage_accounting_failure"):
        book.wrap(delegate, "q-0000").complete("private")
    assert delegate.budget.stops == ["stage_accounting_failure"]


def test_numeric_record_rejects_raw_keys_and_nonfinite_time(tmp_path):
    book = ledger(tmp_path)
    fixed = dict(turns=0, tokens=0, elapsed=0.1, returned=False,
                 usage_complete=True)
    with pytest.raises(RuntimeError, match="stage_counter_invalid"):
        book.record("private question", "reader", **fixed)
    with pytest.raises(RuntimeError, match="stage_counter_invalid"):
        book.record("q-0000", "private stage", **fixed)
    with pytest.raises(RuntimeError, match="stage_counter_invalid"):
        book.record("q-0000", "reader", **{**fixed, "elapsed": float("nan")})


@pytest.mark.parametrize(("responses", "expected"), [
    (["{\"triples\":[],\"markers\":[],\"complete\":true}"] * 2,
     ["extraction_primary_or_other", "extraction_empty_verifier"]),
    (["{\"triples\":[{\"subject\":\"user\",\"predicate\":\"uses\",\"object\":\"PostgreSQL\",\"polarity\":-1}],\"markers\":[],\"complete\":true}",
      "{\"triples\":[],\"markers\":[],\"complete\":true}"],
     ["extraction_primary_or_other", "extraction_omission_verifier"]),
    (["{\"triples\":[],\"markers\":[],\"complete\":\"yes\"}",
      "{\"triples\":[],\"markers\":[],\"complete\":true}"],
     ["extraction_primary_or_other", "extraction_contract_repair"]),
])
def test_actual_frozen_chunk_callgraph(responses, expected):
    candidate = Path("/private/tmp/hymem-r9-summary-20260927.XzIhDt/candidate")
    if not candidate.is_dir():
        pytest.skip("frozen local candidate unavailable")
    collector = Path(stage.__file__).resolve()
    script = '''
import importlib.util, json, sys
from pathlib import Path
candidate, collector, responses = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
sys.path.insert(0, str(candidate))
from hymem.extraction import chunk
spec = importlib.util.spec_from_file_location("profile_collector_test", collector)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class Budget:
    def halt(self, code):
        raise AssertionError(code)
class Delegate:
    observed_turns = 0
    observed_tokens = 0
    usage_complete = True
    budget = Budget()
    def complete(self, request):
        self.observed_turns += 1
        self.observed_tokens += 5
        return responses.pop(0)
book = module.StageLedger(candidate)
result = chunk.extract_chunk(book.wrap(Delegate(), "q-0000"), "User uses PostgreSQL")
print(json.dumps({"stages": book.snapshot()["q-0000"], "calls": result.completion_calls}))
'''
    run = subprocess.run([sys.executable, "-c", script, str(candidate),
                          str(collector), json.dumps(responses)],
                         capture_output=True, text=True, check=True)
    payload = json.loads(run.stdout)
    assert set(payload["stages"]) == set(expected), payload
    assert payload["calls"] == 2
