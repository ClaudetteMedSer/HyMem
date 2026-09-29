"""Offline physical and synthetic checks for the classification candidate."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import luna_classification_candidate as builder


SOURCE = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/candidate")
SOURCE_STAMP = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/headless-grounding-source-map.json")
ACCEPTED = Path("/private/tmp/hymem-semantic-step2-v2-candidate-20260929")
ACCEPTED_STAMP = Path("/private/tmp/hymem-semantic-step2-v2-map-20260929.json")
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("classification-candidate")
    target, stamp = root / "candidate", root / "map.json"
    result = builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                             target, stamp, REPO)
    return target, stamp, result


def _run(candidate: Path, script: str):
    proc = subprocess.run([sys.executable, "-B", "-c", script], cwd=candidate,
                          env=dict(os.environ, PYTHONPATH=str(candidate)),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_exact_inventory_and_original_ordinary_sources(candidate):
    target, stamp, result = candidate
    original = json.loads(SOURCE_STAMP.read_text())["source_sha256"]
    derived = json.loads(stamp.read_text())["source_sha256"]
    assert result["files"] == len(derived) == 513
    assert builder.old.inventory(target) == derived
    assert builder.old.mapping_sha(derived) == result["derived_map_sha256"]
    assert builder.old.sha(stamp.read_bytes()) == result["derived_stamp_sha256"]
    assert {key for key in original if derived[key] != original[key]} == {
        builder.old.CHUNK, builder.old.CONTRACT,
    }
    for relative, digest in builder.HELPERS.items():
        assert derived[relative] == digest
        assert (target / relative).read_bytes() == (REPO / relative).read_bytes()


def test_initial_and_recheck_have_distinct_pinned_callsites(candidate):
    target, _, result = candidate
    source = (target / builder.old.CHUNK).read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "complete_grounding"]
    assert len(calls) == 2
    lines = sorted(node.lineno for node in calls)
    assert lines[1] == lines[0] + 1
    assert source.splitlines()[lines[0] - 2].strip() == "if recheck:"
    assert builder.old.sha(source.encode()) == result["chunk_sha256"]


def test_chunk_preserves_ordinary_complete_and_uses_batch_bound_method(candidate):
    target, _, _ = candidate
    value = _run(target, r'''
import json
from hymem.extraction.chunk import extract_chunk
from hymem.extraction import grounding_classification_v1 as c
from hymem.extraction import contract

source = json.dumps({"source_record_version":"hymem-claim-source-v2",
    "source_message_id":1,"source_role":"user","source_peer_id":"Ada",
    "content":"I prefer tea."}, sort_keys=True)
claim = {"subject":"Ada","predicate":"prefers","object":"tea",
    "polarity":1,"source_message_id":1}
primary = json.dumps({"triples":[claim],"markers":[],"complete":True})
omission = json.dumps({"triples":[],"markers":[],"complete":True})

class Client:
    def __init__(self): self.ordinary=[]; self.ground=[]
    def complete(self, request):
        self.ordinary.append(request)
        return (primary, omission)[len(self.ordinary)-1]
    def complete_grounding(self, request, batch):
        c.validate_request(request, batch)
        self.ground.append((request,batch))
        assert "predicate" not in json.loads(request.user)["batch"]["candidates"][0]
        states = ["n"] * len(c.PREDICATE_ORDER)
        citations = [[] for _ in states]
        pos = c.PREDICATE_ORDER.index("prefers")
        states[pos] = "e"; citations[pos] = [0]
        return json.dumps({"schema":c.GROUNDING_CONTRACT_VERSION,
            "batch_sha256":batch.batch_sha256,"complete":True,
            "classifications":[{"index":0,"states":states,
                "evidence_pool":[{"source_message_id":1,"region":"owned",
                                  "quote":"I prefer tea."}],"citations":citations}]})

client = Client()
result = extract_chunk(client, source, source_records=((1,source),))
print(json.dumps({"failed":result.failed,"reason":result.failure_reason,
    "predicate":result.triples[0].predicate if result.triples else None,
    "ordinary":len(client.ordinary),"ground":len(client.ground),
    "calls":result.completion_calls,"attempts":result.provider_attempts,
    "ground_calls":result.grounding_calls,"rechecks":result.grounding_recheck_calls,
    "contract":contract.extraction_contract_identity()}))
''')
    assert value["failed"] is False
    assert value["predicate"] == "prefers"
    assert (value["ordinary"], value["ground"], value["calls"],
            value["attempts"], value["ground_calls"], value["rechecks"]) == (2, 1, 3, 3, 1, 0)


def test_contract_binding_and_policy_mutations_fail_closed(candidate):
    target, _, _ = candidate
    value = _run(target, r'''
import json
from hymem.extraction import contract, chunk, grounding_classification_v1 as c
from hymem.extraction import grounding_classification_gate_v1 as g

base = contract.extraction_contract_identity()
original = c._SYSTEM
c._SYSTEM += " changed"
changed = contract.extraction_contract_identity()
c._SYSTEM = original
assert contract.extraction_contract_identity() == base
original = c.build_output_schema
c.build_output_schema = lambda batch: {}
try: contract.extraction_contract_identity()
except RuntimeError: schema_binding = True
else: schema_binding = False
c.build_output_schema = original
original = chunk.ground_triples
chunk.ground_triples = lambda *args: []
try: contract.extraction_contract_identity()
except RuntimeError: gate_binding = True
else: gate_binding = False
chunk.ground_triples = original
original = g.parse_grounding_response
g.parse_grounding_response = lambda *args: None
try: contract.extraction_contract_identity()
except RuntimeError: parser_binding = True
else: parser_binding = False
g.parse_grounding_response = original
print(json.dumps({"changed":changed != base,"schema_binding":schema_binding,
                  "gate_binding":gate_binding,"parser_binding":parser_binding,
                  "restored":contract.extraction_contract_identity() == base}))
''')
    assert all(value.values())


def test_unique_correction_gets_one_full_recheck(candidate):
    target, _, _ = candidate
    value = _run(target, r'''
import json
from hymem.extraction.chunk import extract_chunk
from hymem.extraction import grounding_classification_v1 as c

source = json.dumps({"source_record_version":"hymem-claim-source-v2",
    "source_message_id":1,"source_role":"user","content":"I prefer tea."})
claim = {"subject":"Ada","predicate":"uses","object":"tea",
    "polarity":1,"source_message_id":1}
primary = json.dumps({"triples":[claim],"markers":[],"complete":True})
omission = json.dumps({"triples":[],"markers":[],"complete":True})

class Client:
    def __init__(self): self.ordinary=[]; self.ground=[]
    def complete(self, request):
        self.ordinary.append(request)
        return (primary, omission)[len(self.ordinary)-1]
    def complete_grounding(self, request, batch):
        c.validate_request(request, batch)
        self.ground.append(batch)
        states = ["n"] * len(c.PREDICATE_ORDER)
        citations = [[] for _ in states]
        pos = c.PREDICATE_ORDER.index("prefers")
        states[pos] = "e"; citations[pos] = [0]
        return json.dumps({"schema":c.GROUNDING_CONTRACT_VERSION,
            "batch_sha256":batch.batch_sha256,"complete":True,
            "classifications":[{"index":0,"states":states,
                "evidence_pool":[{"source_message_id":1,"region":"owned",
                                  "quote":"I prefer tea."}],"citations":citations}]})

client = Client()
result = extract_chunk(client, source, source_records=((1,source),))
print(json.dumps({"failed":result.failed,"reason":result.failure_reason,
    "predicate":result.triples[0].predicate if result.triples else None,
    "initial_predicate":client.ground[0].triples[0].predicate,
    "recheck_predicate":client.ground[1].triples[0].predicate,
    "calls":result.completion_calls,"ground_calls":result.grounding_calls,
    "rechecks":result.grounding_recheck_calls}))
''')
    assert value == {"failed": False, "reason": None, "predicate": "prefers",
                     "initial_predicate": "uses", "recheck_predicate": "prefers",
                     "calls": 4, "ground_calls": 2, "rechecks": 1}


def test_pins_and_freshness_fail_before_output(tmp_path, monkeypatch):
    target, stamp = tmp_path / "candidate", tmp_path / "map.json"
    monkeypatch.setitem(builder.HELPERS, builder.CLASSIFICATION, "0" * 64)
    with pytest.raises(ValueError, match="helper_source_drift"):
        builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                        target, stamp, REPO)
    assert not target.exists() and not stamp.exists()


def test_direct_isolated_cli_help():
    proc = subprocess.run([sys.executable, "-I", "-B", str(Path(builder.__file__)), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "--accepted-stamp" in proc.stdout
