"""Offline-only checks of source order, shared ledger, gates and cleanup."""
import importlib.util
from contextlib import nullcontext
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("luna_multi_test_subject",
    TOOLS / "luna_subscription_lme_multi_v2.py")
multi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi)


class FakeBudget:
    def __init__(self, limits, *, max_in_flight):
        self.limits = limits
        self.started_at = time.monotonic()
        self.max_in_flight = max_in_flight
        self.questions = {}
        self.stopped = False
        self.stop_code = None
    def register(self, key, limits):
        self.questions[key] = {"turns": 0, "known_tokens": 0,
                               "in_flight": 0, "usage_complete": True,
                               "stopped": False}
    def halt(self, code):
        self.stopped = True
        self.stop_code = self.stop_code or code
    def snapshot(self):
        return {"turns": sum(q["turns"] for q in self.questions.values()),
                "known_tokens": sum(q["known_tokens"] for q in self.questions.values()),
                "usage_complete": all(q["usage_complete"] for q in self.questions.values()),
                "in_flight": 0, "stopped": self.stopped,
                "stop_code": self.stop_code, "questions": self.questions}


class FakeClient:
    def __init__(self, budget, key, limits):
        self.budget = budget
        self.key = key
        budget.register(key, limits)
        self.internal_http_attempts = None
    @property
    def observed_turns(self):
        return self.budget.questions[self.key]["turns"]
    @property
    def observed_tokens(self):
        return self.budget.questions[self.key]["known_tokens"]
    @property
    def usage_complete(self):
        return True
    def complete(self, request):
        self.budget.questions[self.key]["turns"] += 1
        self.budget.questions[self.key]["known_tokens"] += 5
        return "{}"


class Adapter:
    opened = closed = 0
    paths = []
    def __init__(self, path, **kwargs):
        self.path = path
        self.hy = SimpleNamespace(dream_status=lambda: {"pending_chunks": 0,
                                  "summary_healthy": True})
        self.last_indexing_summary = None
        type(self).paths.append(path)
    def open(self):
        type(self).opened += 1
        return self
    def close(self):
        type(self).closed += 1
        if self.path.parent.name == "q-0000":
            type(self).first_closed.set()


class MultiTests(unittest.TestCase):
    def setUp(self):
        Adapter.opened = Adapter.closed = 0
        Adapter.paths = []
        Adapter.first_closed = threading.Event()

    def test_source_order_and_short_input(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "items.json"
            source.write_text('[ {"question_id":"first"}, {"question_id":"second"}, {"question_id":"third"} ]')
            self.assertEqual([x["question_id"] for x in multi.first_source_items(source, 3)],
                             ["first", "second", "third"])
            source.write_text('[{"question_id":"first"}]')
            with self.assertRaises(multi.CampaignStop):
                multi.first_source_items(source, 2)

    def test_lazy_selected_questions_two_pass_validation(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "items.json"
            source.write_text(json.dumps([{"question_id": f"q{i}"} for i in range(5)]))
            calls = []
            def validate(rows, *, scale):
                calls.append(rows[0]["question_id"])
                return tuple(rows)
            selected = multi.SelectedQuestions(source, 3,
                SimpleNamespace(validate_lme_dataset=validate))
            self.assertEqual(calls, ["q0", "q1", "q2"])
            cursor = iter(selected)
            self.assertEqual(next(cursor)["question_id"], "q0")
            self.assertEqual(calls, ["q0", "q1", "q2", "q0"])
            self.assertEqual([item["question_id"] for item in cursor], ["q1", "q2"])
            source.write_text(json.dumps([{"question_id": "changed"},
                {"question_id": "q1"}, {"question_id": "q2"}]))
            with self.assertRaises(multi.CampaignStop):
                next(iter(selected))
            source.write_text('[{"question_id":"same"},{"question_id":"same"}]')
            with self.assertRaises(multi.CampaignStop):
                multi.SelectedQuestions(source, 2,
                    SimpleNamespace(validate_lme_dataset=validate))
            source.write_text('[{"question_id":"first"},,{"question_id":"second"}]')
            with self.assertRaises(multi.CampaignStop):
                multi.first_source_items(source, 2)

    def invoke(self, *, canary_pass=True, second_wrong=False, second_stop=False,
               count=2, workers=2, worker_interrupt=False,
               controller_interrupt=False, canary_interrupt=False):
        concurrency = SimpleNamespace(SharedBudget=FakeBudget,
            ConcurrentStop=type("ConcurrentStop", (BaseException,), {}))
        limits = SimpleNamespace(seconds=10000)
        campaign_limits = SimpleNamespace(seconds=20000)
        def factory(key, qlimits, budget):
            return FakeClient(budget, key, qlimits)
        def evaluate(reader, judge, adapter, question, **kw):
            self.assertEqual(kw["indexing_timeout_s"], 3600)
            self.assertTrue(kw["indexing_require_healthy"])
            self.assertEqual(kw["judge_protocol"], "legacy-custom")
            reader.chat([{"role": "system", "content": "s"},
                         {"role": "user", "content": "u"}])
            judge.chat([{"role": "user", "content": "u"}])
            if second_stop and question["index"] == 1:
                raise concurrency.ConcurrentStop("question_budget_exhausted")
            if worker_interrupt and question["index"] == 1:
                self.assertTrue(Adapter.first_closed.wait(5),
                                "first peer did not finish cleanup before interrupt")
                raise SystemExit("controlled test interrupt")
            indexing = {"outcome": "success", "healthy": True,
                        "summary_healthy": True, "final_status": {"pending_chunks": 0}}
            adapter.last_indexing_summary = indexing
            return {"indexing": indexing, "benchmark_failure": None,
                    "judge_error": False, "judge_parse_valid": True,
                    "correct": not(second_wrong and question["index"] == 1)}
        lme = SimpleNamespace(HyMemAdapter=Adapter, evaluate_question=evaluate,
                              IndexingConvergenceError=ValueError,
                              DEFAULT_MAX_INPUT_TOKENS=100,
                              DEFAULT_MAX_INPUT_BYTES=1000)
        protocol = SimpleNamespace(_validate_versioned_indexing=lambda *a, **kw: True)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            canary_effect = KeyboardInterrupt() if canary_interrupt else None
            with patch.object(multi.old, "experimental_canary", return_value={"passed": canary_pass},
                              side_effect=canary_effect):
                with patch.object(multi.old, "make_adapter_class", return_value=Adapter):
                    with patch.object(multi, "make_memory_client", side_effect=lambda client: client):
                        with patch.object(multi, "terminalize_owned_dream_run",
                                          return_value={"open_runs_before": 0,
                                                        "terminalized": 0,
                                                        "open_runs_after": 0,
                                                        "active_leases_after": 0}):
                            interrupt_context = (patch.object(multi, "wait", side_effect=KeyboardInterrupt)
                                                 if controller_interrupt else nullcontext())
                            with interrupt_context:
                                result = multi.run_campaign(
                        concurrent=concurrency, request_type=lambda **x: SimpleNamespace(**x),
                        canary=object(), chunk=object(), lme=lme, protocol=protocol,
                        binary="/no-binary", questions=[{"index": i} for i in range(count)],
                        output=output, campaign_limits=campaign_limits, question_limits=limits,
                        canary_limits=limits, indexing_timeout_s=3600,
                        workers=workers, client_factory=factory)
            files = sorted(p.relative_to(output).as_posix() for p in output.rglob("*.json"))
            result["_test_failure_text"] = [(p.read_text()[-1000:]) for p in output.rglob("private-failure.txt")]
            return result, files

    def test_canary_gate_prevents_ingestion(self):
        result, files = self.invoke(canary_pass=False)
        self.assertEqual(result["campaign_stop"], "canary_failed")
        self.assertEqual(Adapter.opened, 0)

    def test_canary_interrupt_retains_terminal_budget(self):
        result, _ = self.invoke(canary_interrupt=True)
        self.assertEqual(result["campaign_stop"], "canary_interrupted")
        self.assertEqual(Adapter.opened, 0)
        self.assertEqual(result["budget"]["in_flight"], 0)

    def test_distinct_stores_and_false_is_valid_score(self):
        result, files = self.invoke(second_wrong=True)
        self.assertIsNone(result["campaign_stop"], result["_test_failure_text"])
        self.assertEqual([x["correct"] for x in result["questions"]], [True, False])
        self.assertTrue(all(x["question_completed"] for x in result["questions"]))
        self.assertEqual(Adapter.opened, Adapter.closed)
        self.assertEqual(len(set(Adapter.paths)), 2)
        self.assertIn("q-0000/private-row.json", files)
        self.assertIn("q-0001/private-row.json", files)

    def test_local_budget_stop_preserves_peer(self):
        result, _ = self.invoke(second_stop=True)
        self.assertIsNone(result["campaign_stop"], result["_test_failure_text"])
        self.assertTrue(result["questions"][0]["question_completed"])
        self.assertFalse(result["questions"][1]["question_completed"])
        self.assertEqual(result["questions"][1]["stop_code"],
                         "question_budget_exhausted")
        self.assertEqual(Adapter.opened, Adapter.closed)

    def test_three_source_ordered_questions_with_two_workers(self):
        result, files = self.invoke(count=3, workers=2)
        self.assertEqual([item["index"] for item in result["questions"]], [0, 1, 2])
        self.assertEqual(len(set(Adapter.paths)), 3)
        self.assertTrue(all(item["question_completed"] for item in result["questions"]))

    def test_five_source_ordered_questions_on_four_workers(self):
        result, files = self.invoke(count=5, workers=4, second_wrong=True)
        self.assertEqual(result["schema"], "luna-subscription-lme-multi-v2")
        self.assertIsNone(result["campaign_stop"], result["_test_failure_text"])
        self.assertEqual([item["index"] for item in result["questions"]], list(range(5)))
        self.assertEqual([item["correct"] for item in result["questions"]],
                         [True, False, True, True, True])
        self.assertEqual(len(set(Adapter.paths)), 5)
        self.assertEqual(Adapter.opened, Adapter.closed)
        self.assertTrue(all(item["question_completed"] and item["cleanup_ok"]
                            for item in result["questions"]))
        self.assertEqual(len([name for name in files if name.endswith("private-row.json")]), 5)

    def test_worker_bounds_are_explicit(self):
        for invalid in (0, 5, True):
            with self.assertRaisesRegex(multi.CampaignStop, "worker_count_invalid"):
                self.invoke(count=1, workers=invalid)

    def test_worker_base_exception_preserves_peer_and_terminal_ledger(self):
        result, _ = self.invoke(worker_interrupt=True)
        self.assertEqual(result["campaign_stop"], "worker_interrupted")
        self.assertEqual(result["questions"][1]["stop_code"], "worker_interrupted")
        self.assertTrue(result["questions"][0]["question_completed"])
        self.assertEqual(Adapter.opened, Adapter.closed)
        self.assertEqual(result["active_invocations"], 0)
        self.assertEqual(result["budget"]["in_flight"], 0)

    def test_four_worker_interrupt_preserves_completed_peer(self):
        result, _ = self.invoke(count=5, workers=4, worker_interrupt=True)
        self.assertEqual(result["campaign_stop"], "worker_interrupted")
        self.assertTrue(result["questions"][0]["question_completed"])
        self.assertEqual(result["questions"][1]["stop_code"], "worker_interrupted")
        self.assertEqual(Adapter.opened, Adapter.closed)
        self.assertEqual(result["active_invocations"], 0)
        self.assertEqual(result["budget"]["in_flight"], 0)

    def test_controller_interrupt_halts_before_queue_replenish(self):
        result, _ = self.invoke(count=3, controller_interrupt=True)
        self.assertEqual(result["campaign_stop"], "controller_interrupted")
        self.assertIsNone(result["questions"][2])
        self.assertEqual(Adapter.opened, Adapter.closed)
        self.assertEqual(result["active_invocations"], 0)
        self.assertEqual(result["budget"]["in_flight"], 0)

    def test_progress_failure_before_and_after_turn(self):
        for fail_at, expected_turns in ((1, 0), (2, 1)):
            budget = FakeBudget(SimpleNamespace(seconds=100), max_in_flight=1)
            client = FakeClient(budget, "q", SimpleNamespace(seconds=100))
            count = [0]
            active = set()
            def publish():
                count[0] += 1
                if count[0] == fail_at:
                    raise OSError("private artifact unavailable")
            wrapped = multi._ProgressClient(client, publish, active, "q", __import__("threading").RLock())
            with self.assertRaises(multi.CampaignStop):
                wrapped.complete(SimpleNamespace())
            self.assertEqual(client.observed_turns, expected_turns)
            self.assertEqual(active, set())
            self.assertEqual(budget.stop_code, "artifact_write_failure")

    def test_owned_run_housekeeping_only_terminalizes_telemetry(self):
        with tempfile.TemporaryDirectory() as td:
            owned = Path(td) / "q-0000"
            owned.mkdir()
            db = owned / "hymem.sqlite"
            with closing(sqlite3.connect(db)) as connection:
                connection.executescript("CREATE TABLE run_lock(name TEXT);"
                    "CREATE TABLE dream_runs(id INTEGER PRIMARY KEY, ended_at TEXT, error TEXT);"
                    "CREATE TABLE source_rows(content TEXT);"
                    "INSERT INTO dream_runs(id) VALUES (1);"
                    "INSERT INTO source_rows VALUES ('unchanged');")
                connection.commit()
            adapter = SimpleNamespace(db_path=db)
            result = multi.terminalize_owned_dream_run(adapter, owned, abnormal=True)
            self.assertEqual(result["terminalized"], 1)
            self.assertEqual(result["open_runs_after"], 0)
            with closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute("SELECT content FROM source_rows").fetchone()[0],
                                 "unchanged")
                self.assertEqual(connection.execute("SELECT error FROM dream_runs").fetchone()[0],
                                 "execution_interrupted:subscription_control")
                connection.execute("INSERT INTO dream_runs(id) VALUES (2)")
                connection.execute("INSERT INTO run_lock VALUES ('dreaming')")
                connection.commit()
            with self.assertRaises(multi.CampaignStop):
                multi.terminalize_owned_dream_run(adapter, owned, abnormal=True)
            with closing(sqlite3.connect(db)) as connection:
                self.assertIsNone(connection.execute(
                    "SELECT ended_at FROM dream_runs WHERE id=2").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
