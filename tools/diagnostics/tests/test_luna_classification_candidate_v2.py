"""Physical, source-bound checks for the inactive classification-v2 candidate."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import luna_classification_candidate_v2 as builder


SOURCE = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/candidate")
SOURCE_STAMP = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/headless-grounding-source-map.json")
ACCEPTED = Path("/private/tmp/hymem-semantic-step2-v2-candidate-20260929")
ACCEPTED_STAMP = Path("/private/tmp/hymem-semantic-step2-v2-map-20260929.json")
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("classification-v2-candidate").resolve()
    target, stamp = root / "candidate", root / "map.json"
    result = builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                             target, stamp, REPO)
    return target, stamp, result


def _run(target: Path, script: str):
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(target)],
                          capture_output=True, text=True, timeout=40)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_exact_inventory_and_pinned_helper_bytes(candidate):
    target, stamp, result = candidate
    original = json.loads(SOURCE_STAMP.read_text())["source_sha256"]
    derived = json.loads(stamp.read_text())["source_sha256"]
    assert result["files"] == len(derived) == 513
    assert builder.prior.old.inventory(target) == derived
    assert builder.prior.old.mapping_sha(derived) == result["derived_map_sha256"]
    assert builder.prior.old.sha(stamp.read_bytes()) == result["derived_stamp_sha256"]
    assert {path for path in original if derived[path] != original[path]} == {
        builder.prior.old.CHUNK, builder.prior.old.CONTRACT}
    assert set(derived) - set(original) == set(builder.HELPERS)
    for path, digest in builder.HELPERS.items():
        assert derived[path] == digest
        assert (target / path).read_bytes() == (REPO / path).read_bytes()


def test_v1_to_v2_is_only_import_delta(candidate):
    target, _, _ = candidate
    for path, transform in ((builder.prior.old.CHUNK, builder.prior.derive_chunk),
                            (builder.prior.old.CONTRACT, builder.prior.derive_contract)):
        prior = transform((SOURCE / path).read_bytes())
        expected = builder.derive_chunk((SOURCE / path).read_bytes()) if path == builder.prior.old.CHUNK else builder.derive_contract((SOURCE / path).read_bytes())
        assert (target / path).read_bytes() == expected
        assert expected.replace(b"grounding_classification_gate_v2", b"grounding_classification_gate_v1").replace(
            b"grounding_classification_v2", b"grounding_classification_v1") == prior
    chunk = (target / builder.prior.old.CHUNK).read_text()
    tree = ast.parse(chunk)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "complete_grounding"]
    assert len(calls) == 2
    assert sorted(node.lineno for node in calls)[1] == sorted(node.lineno for node in calls)[0] + 1
    assert "def grounding_call(request: LLMRequest, batch: ClassificationBatch, recheck: bool)" in chunk


def test_actual_candidate_full_recheck_and_cache_identity(candidate):
    target, _, _ = candidate
    value = _run(target, r'''
import json, socket, sys
sys.path.insert(0, sys.argv[1])
from hymem.extraction.chunk import extract_chunk
from hymem.extraction import contract, grounding_classification_v2 as c
def deny(*args, **kwargs): raise AssertionError('offline only')
socket.socket.connect=deny
socket.create_connection=deny
source=json.dumps({'source_record_version':'hymem-claim-source-v2',
    'source_message_id':1,'source_role':'user','source_peer_id':'Ada','content':'I prefer tea.'})
claim={'subject':'Ada','predicate':'uses','object':'tea','polarity':1,'source_message_id':1}
primary=json.dumps({'triples':[claim],'markers':[],'complete':True})
omission=json.dumps({'triples':[],'markers':[],'complete':True})
class Client:
    def __init__(self): self.ordinary=[]; self.ground=[]
    def complete(self, request):
        self.ordinary.append(request)
        return (primary,omission)[len(self.ordinary)-1]
    def complete_grounding(self, request, batch):
        c.validate_request(request,batch)
        self.ground.append(batch)
        assert 'predicate' not in json.loads(request.user)['batch']['candidates'][0]
        states=['not_established']*len(c.PREDICATE_ORDER)
        states[c.PREDICATE_ORDER.index('prefers')]='supported'
        return json.dumps({'schema':c.GROUNDING_CONTRACT_VERSION,
            'batch_sha256':batch.batch_sha256,'complete':True,
            'classifications':[{'index':0,'states':states,
                'support_groups':[{'predicates':['prefers'],'evidence':[
                    {'source_message_id':1,'region':'owned','quote':'I prefer tea.'}]}]}]})
client=Client()
before=contract.extraction_cache_key()
result=extract_chunk(client,source,source_records=((1,source),))
assert contract.extraction_cache_key()==before
old=c._SYSTEM
c._SYSTEM+=' changed'
changed=contract.extraction_cache_key()!=before
c._SYSTEM=old
print(json.dumps({'failed':result.failed,'predicate':result.triples[0].predicate if result.triples else None,
    'batches':[b.triples[0].predicate for b in client.ground], 'ordinary':len(client.ordinary),
    'calls':result.completion_calls,'attempts':result.provider_attempts,
    'ground':result.grounding_calls,'rechecks':result.grounding_recheck_calls,
    'identity_changed':changed,'identity_restored':contract.extraction_cache_key()==before}))
''')
    assert value == {"failed": False, "predicate": "prefers", "batches": ["uses", "prefers"],
                     "ordinary": 2, "calls": 4, "attempts": 4, "ground": 2, "rechecks": 1,
                     "identity_changed": True, "identity_restored": True}


def test_pins_and_freshness_fail_before_output(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    target, stamp = root / "candidate", root / "map.json"
    monkeypatch.setitem(builder.HELPERS, builder.CLASSIFICATION, "0" * 64)
    with pytest.raises(ValueError, match="helper_source_drift"):
        builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                        target, stamp, REPO)
    assert not target.exists() and not stamp.exists()


@pytest.mark.parametrize("mutation", ["parser", "schema", "gate", "validator", "source_mapper"])
def test_consumed_helper_change_invalidates_identity(candidate, mutation):
    target, _, _ = candidate
    value = _run(target, r'''
import json, sys
sys.path.insert(0,sys.argv[1])
from hymem.extraction import contract, chunk
from hymem.extraction import grounding_classification_v2 as classification
from hymem.extraction import grounding_classification_gate_v2 as gate
from hymem.extraction import grounding_v2 as v2, grounding_gate as source_gate
before=contract.extraction_cache_key()
case=''' + repr(mutation) + r'''
if case=='parser': classification.parse_grounding_response=lambda *a,**k: None
elif case=='schema': classification.build_output_schema=lambda *a,**k: {}
elif case=='gate': chunk.ground_triples=lambda *a,**k: []
elif case=='validator': v2._validate.__code__=(lambda *a,**k: None).__code__
elif case=='source_mapper': source_gate._source.__code__=(lambda *a,**k: None).__code__
try: after=contract.extraction_cache_key()
except (RuntimeError,ValueError): print(json.dumps('closed'))
else: print(json.dumps('changed' if after!=before else 'reused'))
''')
    assert value in {"closed", "changed"}


def test_direct_isolated_cli_help():
    proc = subprocess.run([sys.executable, "-I", "-B", str(Path(builder.__file__)), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "--accepted-stamp" in proc.stdout
