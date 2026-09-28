"""Root-owned frozen-runtime verification; no model or network access."""
import os
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def pilot_fixture():
    source = Path(__file__).with_name("test_luna_subscription_pilot.py")
    spec = importlib.util.spec_from_file_location("root_pilot_fixtures", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("close_fails", [False, True])
def test_failure_preserves_known_usage_and_primary(monkeypatch, tmp_path,
                                                  pilot_fixture, close_fails):
    f = pilot_fixture
    snapshots = []
    client = f.Client()

    def canary(_canary, _chunk, active, **kwargs):
        active.complete(f.Request(system="s", user="u", max_tokens=10,
                                  temperature=0, response_format="json"))
        return {"passed": True}

    class Adapter(f.FakeAdapter):
        def close(self):
            if close_fails:
                raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(f.pilot, "experimental_canary", canary)
    monkeypatch.setattr(f.pilot, "make_adapter_class", lambda *args: Adapter)
    with pytest.raises(RuntimeError, match="indexing failed"):
        f.pilot.run_pilot(transport=None, llm_request=f.Request,
            canary=None, chunk=None, lme=f.fake_lme(fails=True), protocol=None,
            binary="/unused", question={}, store_root=tmp_path,
            client_factory=lambda: client,
            progress=lambda value: snapshots.append(dict(value)))
    assert snapshots[-1]["known_tokens"] == 15
    assert snapshots[-1]["observed_turns"] == 3
    assert snapshots[-1]["cleanup_ok"] is not close_fails
    assert snapshots[-1]["question_completed"] is False


def test_late_usage_loss_cannot_pass_question(pilot_fixture):
    f = pilot_fixture
    client = f.Client()
    complete = client.complete

    def incomplete(request):
        result = complete(request)
        if client.observed_turns == 3:
            client.usage_complete = False
        return result

    client.complete = incomplete
    with pytest.raises(f.pilot.PilotStop, match="usage_incomplete"):
        f.PilotTests().invoke(client=client)


@pytest.mark.skipif(not os.environ.get("HYMEM_FROZEN_CANDIDATE"),
                    reason="needs the retained frozen R9 candidate")
def test_real_frozen_canary_and_adapter_are_subscription_only(tmp_path):
    project = Path(__file__).resolve().parents[3]
    candidate = Path(os.environ["HYMEM_FROZEN_CANDIDATE"])
    script = r'''
import json, pathlib, socket, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[2])
from benchmarks import extraction_canary as c, longmemeval_adapter as lme
from hymem.extraction import chunk, producer
from tools.diagnostics import luna_subscription_pilot as p

def no_network(*args, **kwargs):
    raise AssertionError("network forbidden in root offline verification")
socket.create_connection = no_network
socket.socket.connect = no_network
original_table_context = dict(c._EXPECTED_TABLE_CONTEXT)
class SyntheticClient:
    observed_turns = 0
    observed_tokens = None
    usage_complete = True
    type_mode = "typed"
    def complete(self, request):
        self.observed_turns += 1
        self.observed_tokens = (self.observed_tokens or 0) + 15
        claims = []
        keys = ("subject", "subject_type", "predicate", "object",
                "object_type", "polarity", "source_message_id")
        expected = [dict(zip(keys, row)) for row in c._CANARY_EXPECTED_CLAIMS]
        if "OMISSION VERIFICATION PASS" not in request.system:
            payloads, errors = c._request_source_payloads(request)
            assert not errors
            for payload in payloads:
                if (c._TABLE_CLAIM_ROW in payload["content"] and
                    payload.get("source_fragment_context") == original_table_context):
                    claims.extend(row for row in expected
                        if row["source_message_id"] == c.EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID)
                if (c._PROSE_BOUNDARY_RIGHT in payload["content"] and
                    payload.get("source_boundary_context") == c._EXPECTED_PROSE_BOUNDARY_CONTEXT):
                    claims.extend(row for row in expected
                        if row["source_message_id"] == c.EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID)
        for row in claims:
            is_table = row["source_message_id"] == c.EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID
            if self.type_mode == "omit_all" or (self.type_mode == "omit_table" and is_table):
                row.pop("subject_type")
                row.pop("object_type")
            elif self.type_mode == "wrong_type" and is_table:
                row["subject_type"] = "tool"
            elif self.type_mode == "invalid_type" and is_table:
                row["subject_type"] = 42
            elif self.type_mode == "null_type" and is_table:
                row["object_type"] = None
            elif self.type_mode == "wrong_source" and is_table:
                row["source_message_id"] = c.EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID
        return json.dumps({"triples": claims, "markers": [], "complete": True})
client = SyntheticClient()
report = p.experimental_canary(c, chunk, client)
assert report["passed"] is True
assert report["completion_calls"] == 8
assert client.observed_turns == 8

for mode, should_pass in (("omit_table", True), ("omit_all", True),
                          ("wrong_type", False), ("invalid_type", False),
                          ("null_type", False), ("wrong_source", False)):
    control = SyntheticClient()
    control.type_mode = mode
    outcome = p.experimental_canary(c, chunk, control)
    assert outcome["passed"] is should_pass, (mode, outcome)

with patch.object(c, "_EXPECTED_TABLE_CONTEXT", {"wrong": "context"}):
    control = SyntheticClient()
    outcome = p.experimental_canary(c, chunk, control)
    assert outcome["passed"] is False

declared = p.SubscriptionMemoryClient(client)
binding = producer.phase1_producer_binding(declared)
assert binding["identity_exact"] is True
assert binding["declaration"]["endpoint_origin"] is None
assert binding["declaration"]["model"] == "gpt-6-luna"
assert producer.producer_binding_for_declaration(declared,
    declaration_hook="memory_producer_declaration")["identity_exact"] is True

adapter_cls = p.make_adapter_class(lme, declared)
adapter = adapter_cls(pathlib.Path(sys.argv[3]) / "root-offline.sqlite",
    embeddings=False, aggregation_nodes=False, episode_granularity=False,
    pipeline_model="gpt-6-luna")
with patch("hymem.contrib.openai_client.OpenAICompatibleClient", side_effect=no_network):
    adapter.open()
    try:
        assert adapter.hy._llm is declared
        fork = adapter.hy.fork()
        try:
            assert fork._llm is declared
        finally:
            fork.close()
        assert adapter.embedding_client is None
        assert adapter.pipeline_llm is declared
    finally:
        adapter.close()
assert client.observed_turns == 8
'''
    result = subprocess.run([sys.executable, "-B", "-c", script,
                             str(candidate), str(project), str(tmp_path)],
                            cwd=tmp_path, env=os.environ.copy(),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
