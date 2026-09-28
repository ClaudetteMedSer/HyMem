"""Offline pilot gate tests; no Codex process, provider call, or LME data."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "luna_subscription_pilot.py"
SPEC = importlib.util.spec_from_file_location("luna_pilot_test_subject", SOURCE)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class Request:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class Client:
    def __init__(self):
        self.inference_accepted = False
        self.observed_turns = 0
        self.observed_tokens = None
        self.usage_complete = True
        self.requested_controls = []

    def preflight(self):
        return {"config_isolation_admitted": True, "auth": "chatgpt",
                "model": "gpt-6-luna", "inference_enabled": False}

    def complete(self, request):
        assert self.inference_accepted
        self.observed_turns += 1
        self.observed_tokens = (self.observed_tokens or 0) + 5
        self.requested_controls.append({"temperature_effective": None})
        return "yes"


class FakeAdapter:
    opened = 0
    closed = 0

    def __init__(self, *args, **kwargs):
        pass

    def open(self):
        type(self).opened += 1
        return self

    def close(self):
        type(self).closed += 1


def fake_lme(*, fails=False):
    def evaluate(reader, judge, adapter, question, **kwargs):
        assert kwargs["top_k"] == 15
        assert kwargs["judge_protocol"] == "legacy-custom"
        assert kwargs["indexing_max_cycles"] == 100
        assert kwargs["indexing_timeout_s"] == 3600.0
        assert kwargs["indexing_require_healthy"] is True
        assert kwargs["permissive_default"] is True
        assert kwargs["no_dream"] is False
        assert reader is judge
        reader.chat([{"role": "system", "content": "reader"},
                     {"role": "user", "content": "question"}])
        judge.chat([{"role": "user", "content": "judge prompt"}])
        if fails:
            raise RuntimeError("indexing failed")
        return {"indexing": {"outcome": "success", "healthy": True,
                             "summary_healthy": True},
                "benchmark_failure": None, "correct": True,
                "judge_error": False, "judge_parse_valid": True}
    return SimpleNamespace(HyMemAdapter=FakeAdapter,
                           evaluate_question=evaluate,
                           DEFAULT_MAX_INPUT_TOKENS=16000,
                           DEFAULT_MAX_INPUT_BYTES=60000)


class PilotTests(unittest.TestCase):
    def setUp(self):
        FakeAdapter.opened = FakeAdapter.closed = 0

    def invoke(self, *, canary_pass=True, lme=None, client=None):
        client = client or Client()
        lme = lme or fake_lme()
        def canary_result(active_client):
            active_client.complete(Request(system="fixture", user="fixture",
                                           response_format="json", max_tokens=10,
                                           temperature=0.0))
            return {"passed": canary_pass}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pilot, "experimental_canary",
                              side_effect=lambda _a, _b, active, **kw: canary_result(active)):
                with patch.object(pilot, "make_adapter_class", return_value=FakeAdapter):
                    return pilot.run_pilot(
                        transport=None, llm_request=Request, canary=None, chunk=None,
                        lme=lme, protocol=SimpleNamespace(
                            _validate_versioned_indexing=lambda *a, **k: True),
                        binary="/unused",
                        question={"question_id": "first"},
                        store_root=Path(directory), client_factory=lambda: client)

    def test_one_question_after_canary_and_shared_client(self):
        client = Client()
        summary, row = self.invoke(client=client)
        self.assertTrue(summary["question_completed"])
        self.assertTrue(row["correct"])
        self.assertEqual(client.observed_turns, 3)
        self.assertEqual(FakeAdapter.opened, 1)
        self.assertEqual(FakeAdapter.closed, 1)

    def test_canary_gate_no_question(self):
        with self.assertRaisesRegex(pilot.PilotStop, "canary_failed"):
            self.invoke(canary_pass=False)
        self.assertEqual(FakeAdapter.opened, 0)

    def test_usage_gate_no_question(self):
        client = Client()
        original = client.complete
        def incomplete(request):
            response = original(request)
            client.usage_complete = False
            return response
        client.complete = incomplete
        with self.assertRaisesRegex(pilot.PilotStop, "usage_incomplete"):
            self.invoke(client=client)
        self.assertEqual(FakeAdapter.opened, 0)

    def test_indexing_failure_closes_adapter(self):
        with self.assertRaisesRegex(RuntimeError, "indexing failed"):
            self.invoke(lme=fake_lme(fails=True))
        self.assertEqual(FakeAdapter.closed, 1)

    def test_first_item_source_order_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.json"
            dataset.write_text(json.dumps([{"question_id": "first"},
                                           {"question_id": "second"}]),
                               encoding="utf-8")
            self.assertEqual(pilot.first_source_item(dataset)["question_id"], "first")
            dataset.write_text('{"question_id":"not-array"}', encoding="utf-8")
            with self.assertRaisesRegex(pilot.PilotStop, "dataset_shape_invalid"):
                pilot.first_source_item(dataset)

    def test_chat_bridge_preserves_one_user_judge_text(self):
        client = Client()
        requests = []
        def capture(request):
            requests.append(request)
            return "yes"
        client.complete = capture
        bridge = pilot.ChatBridge(client, Request)
        bridge.chat([{"role": "user", "content": "exact judge"}],
                    temperature=0.0, max_tokens=10)
        self.assertEqual((requests[0].system, requests[0].user,
                          requests[0].max_tokens), ("", "exact judge", 10))

    def test_in_flight_progress_brackets_completion_and_error(self):
        snapshots = []
        client = Client()
        client.inference_accepted = True
        wrapper = pilot.SubscriptionMemoryClient(
            client, progress=lambda active: snapshots.append(
                (active.invocation_in_flight, active.observed_turns,
                 active.observed_tokens)))
        wrapper.complete(Request(system="s", user="u"))
        self.assertEqual(snapshots, [(True, 0, None), (False, 1, 5)])
        def interrupted(_request):
            raise RuntimeError("private failure text")
        client.complete = interrupted
        with self.assertRaises(RuntimeError):
            wrapper.complete(Request(system="s", user="u"))
        self.assertEqual(snapshots[-2][0], True)
        self.assertEqual(snapshots[-1][0], False)

    def test_wrong_type_cannot_hide_behind_unrelated_invalid_optional(self):
        claim = {"subject": "HyMem Canary Relay", "predicate": "deploys_to",
                 "object": "Fly.io", "polarity": 1, "source_message_id": 1}
        bad = {**claim, "subject_type": "project", "unexpected_field": "x"}
        good = {**claim, "subject_type": "service"}
        canary = SimpleNamespace(
            _CANARY_EXPECTED_CLAIMS=[("HyMem Canary Relay", "service",
                "deploys_to", "Fly.io", "platform", 1, 1)],
            EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID=1,
            EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID=2,
            _TABLE_CLAIM_ROW="HyMem Canary Relay",
            _PROSE_BOUNDARY_RIGHT="prose",
            _EXPECTED_TABLE_CONTEXT={"type": "header"},
            _EXPECTED_PROSE_BOUNDARY_CONTEXT={},
            _strict_equal=lambda a, b: a == b,
            _request_source_payloads=lambda _request: ([{
                "source_message_id": 1,
                "content": "HyMem Canary Relay",
                "source_fragment_context": {"type": "header"}}], 0))
        response = json.dumps({"triples": [bad, good], "markers": []})
        path, slots = pilot._core_path_and_types(
            canary, [(object(), response)], {})
        self.assertTrue(slots["0_subject"]["wrong"])
        self.assertEqual(path["table_claim_exact_context_emissions"], 1)

    def test_exact_experimental_canary_contract(self):
        fields = ("subject", "subject_type", "predicate", "object", "object_type",
                  "polarity", "source_message_id")
        rows = [("A", "person", "prefers", "B", "database", 1, 1),
                ("C", "service", "runs_on", "D", "platform", 1, 2)]
        triples = [SimpleNamespace(**dict(zip(fields, row)), optional=None)
                   for row in rows]
        result = SimpleNamespace(failed=False, triples=triples, markers=[],
                                 duplicate_triples_collapsed=0,
                                 entity_type_hints={"A": "person", "B": "database",
                                                    "C": "service", "D": "platform"},
                                 entity_property_hints={}, completion_calls=0,
                                 provider_attempts=0,
                                 initial_prepartition_leaves=0)
        class Recorder:
            def __init__(self, delegate):
                self.requests = []
                self.responses = []
                self.provider_output_truncations = 0
        canary = SimpleNamespace(_CANARY_CONTENT="fixture", _source_records=lambda: (),
                                 EXTRACTION_CANARY_MAX_COMPLETION_CALLS=24,
                                 _CANARY_EXPECTED_CLAIMS=rows,
                                 _CANARY_OPTIONAL_TRIPLE_FIELDS=("optional",),
                                 EXTRACTION_CANARY_FIXTURE_SHA256="fixture-hash",
                                 _RecordingClient=Recorder,
                                 _request_execution_path=lambda req, resp: {
                                     "table_claim_exact_context_emissions": 0,
                                     "table_claim_wrong_context_emissions": 0,
                                     "prose_claim_exact_context_emissions": 0,
                                     "prose_claim_wrong_context_emissions": 0},
                                 _validate_execution_path=lambda *a, **k: None,
                                 extraction_canary_policy=lambda: {
                                     "normal_execution_path": {
                                         "provider_output_truncations": 0,
                                         "table_claim_exact_context_emissions": 0,
                                         "table_claim_wrong_context_emissions": 0,
                                         "prose_claim_exact_context_emissions": 0,
                                         "prose_claim_wrong_context_emissions": 0},
                                     "normal_pass_completion_calls": 0,
                                     "minimum_pass_completion_calls": 0,
                                     "max_completion_calls": 24,
                                     "max_provider_attempts": 72,
                                     "expected_prepartition_leaves": 0})
        chunk = SimpleNamespace(extract_chunk=lambda *a, **k: result)
        active = Client()
        active.observed_tokens = 0
        self.assertTrue(pilot.experimental_canary(canary, chunk, active)["passed"])
        result.entity_property_hints = {"A": {"unexpected": "value"}}
        self.assertFalse(pilot.experimental_canary(canary, chunk, active)["passed"])

    def test_inventory_pin_and_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            for name in ("hymem/extraction/chunk.py",
                         "benchmarks/extraction_canary.py",
                         "benchmarks/longmemeval_adapter.py"):
                path = candidate / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source", encoding="utf-8")
            files = {path.relative_to(candidate).as_posix(): pilot.digest(path)
                     for path in candidate.rglob("*") if path.is_file()}
            stamp = root / "stamp.json"
            stamp.write_text(json.dumps({"source_sha256": files}), encoding="utf-8")
            map_sha = __import__("hashlib").sha256(json.dumps(
                files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(pilot.verify_inventory(candidate, stamp,
                pilot.digest(stamp), expected_map_sha256=map_sha,
                expected_file_count=3), 3)
            (candidate / "hymem/extraction/chunk.py").write_text("drift")
            with self.assertRaisesRegex(pilot.PilotStop, "inventory_drift"):
                pilot.verify_inventory(candidate, stamp, pilot.digest(stamp),
                    expected_map_sha256=map_sha, expected_file_count=3)

    def test_progress_on_failed_canary_is_safe(self):
        snapshots = []
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pilot, "experimental_canary",
                              return_value={"passed": False,
                                            "failure_code": "canary_contract_failed"}):
                with self.assertRaisesRegex(pilot.PilotStop, "canary_failed"):
                    pilot.run_pilot(
                        transport=None, llm_request=Request, canary=None,
                        chunk=None, lme=fake_lme(), protocol=None, binary="/unused",
                        question={"question_id": "private-question"},
                        store_root=Path(directory), client_factory=lambda: client,
                        progress=lambda value: snapshots.append(json.loads(json.dumps(value))))
        self.assertEqual(snapshots[-1]["canary"]["failure_code"],
                         "canary_contract_failed")
        self.assertFalse(snapshots[-1]["question_started"])
        self.assertNotIn("private-question", json.dumps(snapshots))

    @unittest.skipUnless(os.environ.get("HYMEM_FROZEN_CANDIDATE"),
                         "set HYMEM_FROZEN_CANDIDATE for frozen-module test")
    def test_real_frozen_producer_and_canary_contract(self):
        candidate = Path(os.environ["HYMEM_FROZEN_CANDIDATE"])
        project = SOURCE.parents[2]
        script = """
from hymem.extraction.producer import phase1_producer_binding
from benchmarks import extraction_canary as c
from hymem.extraction import chunk
from tools.diagnostics.luna_subscription_pilot import (
    SubscriptionMemoryClient, experimental_canary)
import json
client = SubscriptionMemoryClient(object())
binding = phase1_producer_binding(client)
assert binding['identity_exact'] is True
assert binding['declaration']['endpoint_origin'] is None
assert binding['declaration']['model'] == 'gpt-6-luna'
assert client.memory_producer_declaration().model == 'gpt-6-luna'
p = c.extraction_canary_policy()
assert p['normal_pass_completion_calls'] == 8
assert c._validate_execution_path(p['normal_execution_path'],
    completion_calls=8, initial_leaves=p['expected_prepartition_leaves'],
    passed=True, policy=p) == 0
class Telemetry:
    def __init__(self, variant='typed'):
        self.variant = variant
        self.observed_turns = 0
        self.observed_tokens = 0
        self.usage_complete = True
    def complete(self, request):
        self.observed_turns += 1
        self.observed_tokens += 1
        items = []
        if 'OMISSION VERIFICATION PASS' not in request.system:
            payloads, failures = c._request_source_payloads(request)
            assert failures == 0
            for payload in payloads:
                content = payload.get('content', '')
                if (c._TABLE_CLAIM_ROW in content
                        and payload.get('source_fragment_context') ==
                            c._EXPECTED_TABLE_CONTEXT):
                    s, st, pred, obj, ot, polarity, source = c._CANARY_EXPECTED_CLAIMS[0]
                    items.append(dict(subject=s, subject_type=st, predicate=pred,
                                      object=obj, object_type=ot, polarity=polarity,
                                      source_message_id=source))
                if (c._PROSE_BOUNDARY_RIGHT in content
                        and payload.get('source_boundary_context') ==
                            c._EXPECTED_PROSE_BOUNDARY_CONTEXT):
                    s, st, pred, obj, ot, polarity, source = c._CANARY_EXPECTED_CLAIMS[1]
                    items.append(dict(subject=s, subject_type=st, predicate=pred,
                                      object=obj, object_type=ot, polarity=polarity,
                                      source_message_id=source))
        data = {'triples': items, 'markers': [], 'complete': True}
        for item in data.get('triples', []):
            if self.variant in ('all_untyped', 'table_untyped') and (
                self.variant == 'all_untyped' or item.get('subject') == 'HyMem Canary Relay'):
                item.pop('subject_type', None)
                item.pop('object_type', None)
            if item.get('subject') == 'HyMem Canary Relay':
                if self.variant == 'wrong_type':
                    item['subject_type'] = 'project'
                if self.variant == 'invalid_type':
                    item['subject_type'] = None
                if self.variant == 'wrong_source':
                    item['source_message_id'] = 999
        return json.dumps(data)
for variant, expected_pass, expected_present in (
    ('typed', True, 4), ('table_untyped', True, 2),
    ('all_untyped', True, 0), ('wrong_type', False, 4),
    ('invalid_type', False, 4), ('wrong_source', False, None)):
    result = experimental_canary(
        c, chunk, Telemetry(variant))
    assert result['passed'] is expected_pass, (variant, result)
    assert result['schema'] == 'luna-experimental-canary-v2'
    if variant == 'wrong_type':
        assert result['type_fields_wrong'] > 0
    if variant == 'invalid_type':
        assert result['type_fields_invalid'] > 0
    if expected_pass:
        assert result['type_fields_present'] == expected_present, (variant, result)
        assert result['completion_calls'] == 8, (variant, result)
        assert result['core_execution_path_exact'] is True
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(candidate), str(project),
                                             env.get("PYTHONPATH", "")))
        completed = subprocess.run([sys.executable, "-c", script], cwd=candidate,
                                   env=env, capture_output=True, text=True,
                                   timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
