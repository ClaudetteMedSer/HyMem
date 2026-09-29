"""Offline synthetic completions against the actual derived frozen runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import luna_semantic_candidate as builder

SOURCE = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/candidate")
STAMP = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/headless-grounding-source-map.json")
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("semantic-candidate")
    target = root / "candidate"
    stamp = root / "map.json"
    builder.prepare(SOURCE, STAMP, target, stamp, REPO)
    return target


SCRIPT = r'''
import json, sys
from hymem.extraction.chunk import extract_chunk

case = sys.argv[1]
triple = {"subject":"Ada","predicate":"uses","object":"tea","polarity":1,"source_message_id":1}
source = json.dumps({"source_record_version":"hymem-claim-source-v2","source_message_id":1,
    "source_role":"user","source_peer_id":"Ada","content":"I prefer tea."}, sort_keys=True)
if case == "legacy":
    triple.pop("source_message_id")
if case == "collision":
    first = dict(triple, predicate="prefers")
    triples = [first, triple]
elif case == "many":
    triples = [dict(triple, object="tea " + str(index)) for index in range(9)]
else:
    triples = [triple]
primary = json.dumps({"triples":triples,"markers":[],"complete":True})
omission = json.dumps({"triples":[],"markers":[],"complete":True})
marker = json.dumps({"triples":[],"markers":[{"kind":"preference","statement":"tea"}],"complete":True})
empty = omission

class Client:
    def __init__(self): self.requests=[]; self.ground=0
    def complete(self, request):
        self.requests.append(request)
        if "Return exactly one JSON object with schema \"source-grounding-v1\"" in request.system:
            self.ground += 1
            if case == "provider": raise RuntimeError("synthetic provider error")
            batch = json.loads(request.user)
            entries = []
            for index, claim in enumerate(batch["batch"]["candidates"]):
                status = "supported"
                predicate = claim["predicate"]
                if case in {"reject", "uncertain"}: status, predicate = ("uncertain" if case == "uncertain" else "unsupported"), None
                if case == "recheck_reject" and self.ground == 2: status, predicate = "unsupported", None
                if case == "second_correction" and self.ground == 2: status, predicate = "replace_predicate", "rejects"
                if case in {"correct", "collision", "recheck_reject", "second_correction"} and self.ground == 1 and claim["predicate"] == "uses":
                    status, predicate = "replace_predicate", "prefers"
                evidence = [] if status in {"unsupported", "uncertain"} else [{"source_message_id":claim["source_message_id"],
                    "region":"owned","quote":"I prefer tea."}]
                entries.append({"index":index,"status":status,"predicate":predicate,"evidence":evidence})
            result = {"schema":"source-grounding-v1","batch_sha256":batch["batch_sha256"],
                "complete":True,"verdicts":entries}
            if case == "malformed": result["batch_sha256"] = "0" * 64
            return json.dumps(result)
        number = len([r for r in self.requests if "source-grounding-v1" not in r.system])
        if case == "empty_recovery":
            return (empty, primary, omission)[number - 1]
        if case == "terminal_recovery":
            return ('{"triples":', primary, omission)[number - 1]
        if case == "marker": return (marker, omission)[number - 1]
        return (primary, omission)[number - 1]

client = Client()
result = extract_chunk(client, "I prefer tea." if case == "legacy" else source,
    source_records=None if case == "legacy" else ((1,source),),
    completion_call_limit=2 if case == "budget" else None)
print(json.dumps({"failed":result.failed,"reason":result.failure_reason,
    "details":result.failure_details,"triples":[t.__dict__ for t in result.triples],
    "markers":[m.__dict__ for m in result.markers],
    "calls":result.completion_calls,"provider_attempts":result.provider_attempts,
    "grounding_calls":result.grounding_calls,"recheck":result.grounding_recheck_calls,
    "grounding_attempts":result.grounding_provider_attempts}, sort_keys=True))
'''


def run_case(candidate: Path, case: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(candidate))
    proc = subprocess.run([sys.executable, "-c", SCRIPT, case], cwd=candidate,
                          env=env, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


@pytest.mark.parametrize("case", [
    "normal", "correct", "reject", "malformed", "collision", "budget",
    "legacy", "provider", "marker", "empty_recovery", "terminal_recovery",
    "uncertain", "recheck_reject", "second_correction", "many",
])
def test_derived_runtime_cases(candidate, case):
    value = run_case(candidate, case)
    if case in {"reject", "uncertain", "malformed", "collision", "budget",
                "provider", "recheck_reject", "second_correction"}:
        assert value["failed"]
        assert value["triples"] == []
        assert value["markers"] == []
    else:
        assert not value["failed"]
    if case == "correct":
        assert value["triples"][0]["predicate"] == "prefers"
        assert value["grounding_calls"] == 2
        assert value["recheck"] == 1
    if case == "legacy":
        assert value["triples"][0]["source_message_id"] is None
    if case == "marker":
        assert value["grounding_calls"] == 0
        assert len(value["markers"]) == 1
    if case in {"empty_recovery", "terminal_recovery"}:
        assert value["grounding_calls"] == 1
    if case == "many":
        assert len(value["triples"]) == 9
        assert value["grounding_calls"] == 2
    assert value["calls"] == value["provider_attempts"]
    assert value["grounding_calls"] == value["grounding_attempts"]


def test_builder_rejects_original_drift(tmp_path):
    with pytest.raises(ValueError, match="chunk_source_drift"):
        builder.derive_chunk(b"older checkout")
    with pytest.raises(ValueError, match="contract_source_drift"):
        builder.derive_contract(b"older checkout")


def test_cache_identity_and_helper_guards(candidate):
    script = r'''
from hymem.extraction import chunk, contract, grounding, grounding_gate
base = contract.extraction_contract_identity()
original = grounding_gate.GROUNDING_GATE_VERSION
grounding_gate.GROUNDING_GATE_VERSION += "-changed"
assert contract.extraction_contract_identity() != base
grounding_gate.GROUNDING_GATE_VERSION = original
assert contract.extraction_contract_identity() == base
original = chunk.ground_triples
chunk.ground_triples = lambda *args: []
try:
    contract.extraction_contract_identity()
except RuntimeError: pass
else: raise AssertionError("chunk alias rebind reused cache identity")
chunk.ground_triples = original
original = grounding_gate.parse_grounding_response
grounding_gate.parse_grounding_response = lambda *args: None
try:
    contract.extraction_contract_identity()
except RuntimeError: pass
else: raise AssertionError("gate alias rebind reused cache identity")
grounding_gate.parse_grounding_response = original
original = grounding.loads_exact_or_fenced
grounding.loads_exact_or_fenced = lambda *args: None
try:
    contract.extraction_contract_identity()
except RuntimeError: pass
else: raise AssertionError("contract alias rebind reused cache identity")
grounding.loads_exact_or_fenced = original
original = grounding.build_grounding_request
grounding.build_grounding_request = lambda *args: None
try:
    contract.extraction_contract_identity()
except RuntimeError: pass
else: raise AssertionError("definition rebind reused consumed helper identity")
grounding.build_grounding_request = original
assert contract.extraction_contract_identity() == base
'''
    proc = subprocess.run([sys.executable, "-c", script], cwd=candidate,
                          env=dict(os.environ, PYTHONPATH=str(candidate)),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_builder_rejects_output_aliases(tmp_path):
    helper_alias = tmp_path / "helper-link"
    helper_alias.symlink_to(REPO, target_is_directory=True)
    with pytest.raises(ValueError, match="path_alias_or_symlink"):
        builder.prepare(SOURCE, STAMP, tmp_path / "candidate", tmp_path / "map.json",
                        helper_alias)
    with pytest.raises(ValueError, match="output_inside_source"):
        builder.prepare(SOURCE, STAMP, tmp_path / "candidate", SOURCE / "new-map.json",
                        REPO)
    with pytest.raises(ValueError, match="path_alias_or_symlink"):
        builder.prepare(SOURCE, STAMP, tmp_path / "candidate", tmp_path / "alias" / ".." / "map.json",
                        REPO)


def test_nested_context_preserves_full_body_and_scopes_header(candidate):
    script = r'''
import json
from hymem.extraction.grounding_gate import _source
prior = {"source_message_id":9,"source_role":"assistant","source_peer_id":"bot",
    "content":"row one\nrow two", "source_content_start":100,"source_content_end":115,
    "context_for_source_message_id":1,"applies_through_source_content_end":8,
    "source_fragment_context":{"content":"| item | value |\n|---|---|\n",
        "applies_through_source_content_end":107}}
owned = {"source_message_id":1,"source_role":"user","source_peer_id":"Ada",
    "content":"I choose this.","source_content_start":0}
source = _source((1,json.dumps(owned)), ((9,json.dumps(prior)),))
assert source.contexts[0].region == "conversation_0"
assert source.contexts[0].content == prior["content"]
header = source.contexts[1]
assert header.region == "conversation_0_header"
assert header.applies_to_region == "conversation_0"
assert header.applies_to_prefix_chars == 7
assert header.owned_prefix_chars == 8
'''
    proc = subprocess.run([sys.executable, "-c", script], cwd=candidate,
                          env=dict(os.environ, PYTHONPATH=str(candidate)),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
