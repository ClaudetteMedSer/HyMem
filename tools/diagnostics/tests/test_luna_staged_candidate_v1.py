"""Physical isolation checks for the inactive staged candidate builder."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import luna_staged_candidate_v1 as builder

SOURCE = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/candidate")
SOURCE_STAMP = Path("/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/headless-grounding-source-map.json")
ACCEPTED = Path("/private/tmp/hymem-semantic-step2-v2-candidate-20260929")
ACCEPTED_STAMP = Path("/private/tmp/hymem-semantic-step2-v2-map-20260929.json")
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("staged-candidate").resolve()
    target, stamp = root / "candidate", root / "map.json"
    result = builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                             target, stamp, REPO)
    return target, stamp, result


def _run(target: Path, script: str):
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(target)],
                          capture_output=True, text=True, timeout=40)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_exact_inventory_and_bytes(candidate):
    target, stamp, result = candidate
    original = json.loads(SOURCE_STAMP.read_text())["source_sha256"]
    derived = json.loads(stamp.read_text())["source_sha256"]
    assert result["files"] == len(derived) == 514
    assert builder.old.inventory(target) == derived
    assert builder.old.mapping_sha(derived) == result["derived_map_sha256"]
    assert builder.old.sha(stamp.read_bytes()) == result["derived_stamp_sha256"]
    assert {path for path in original if derived[path] != original[path]} == {
        builder.old.CHUNK, builder.old.CONTRACT}
    assert set(derived) - set(original) == set(builder.HELPERS)
    for path, digest in builder.HELPERS.items():
        assert derived[path] == digest
        assert (target / path).read_bytes() == (REPO / path).read_bytes()


def test_actual_staged_correction_accounting_and_identity(candidate):
    target, _, _ = candidate
    value = _run(target, r'''
import json, socket, sys
sys.path.insert(0, sys.argv[1])
from hymem.extraction.chunk import extract_chunk
from hymem.extraction import contract, grounding_staged_v1 as staged
from hymem.extraction import grounding_classification_v4 as v4
def deny(*args, **kwargs): raise AssertionError('offline only')
socket.socket.connect=deny
socket.create_connection=deny
source=json.dumps({'source_record_version':'hymem-claim-source-v2',
    'source_message_id':1,'source_role':'user','source_peer_id':'Ada','content':'I prefer tea.'})
claim={'subject':'Ada','predicate':'uses','object':'tea','polarity':1,'source_message_id':1}
primary=json.dumps({'triples':[claim],'markers':[],'complete':True})
omission=json.dumps({'triples':[],'markers':[],'complete':True})
quote={'source_message_id':1,'region':'owned','quote':'I prefer tea.'}
support={'evidence':[quote], 'checks':{
    'attribution_and_roles':{'state':'supported','evidence_indices':[0]},
    'relation_and_polarity':{'state':'supported','evidence_indices':[0]}}}
class Client:
    def __init__(self): self.ordinary=[]; self.stages=[]; self.wire=[]
    def complete(self, request):
        self.ordinary.append(request)
        return (primary,omission)[len(self.ordinary)-1]
    def complete_stage(self, request, batch, stage, recheck):
        self.stages.append((stage,recheck))
        assert request.max_tokens == 4096
        if stage=='original':
            staged.validate_original_request(request,batch)
            self.wire.append(json.loads(request.user)['batch']['candidates'][0])
            assessment={'state':'not_established','support':None} if not recheck else {'state':'supported','support':support}
            return json.dumps({'schema':staged.ORIGINAL_SCHEMA,'batch_sha256':batch.batch_sha256,
                'complete':True,'originals':[{'index':0,'original':assessment}]})
        staged.validate_alternatives_request(request,batch)
        values={p:{'state':'not_established','support':None} for p in v4.PREDICATE_ORDER if p!='uses'}
        values['prefers']={'state':'supported','support':support}
        return json.dumps({'schema':staged.ALTERNATIVES_SCHEMA,
            'batch_sha256':batch.classification_batch.batch_sha256,
            'original_response_sha256':batch.original_response_sha256,
            'complete':True,'alternatives':[{'index':0,'alternatives':values}]})
client=Client()
before=contract.extraction_cache_key()
result=extract_chunk(client,source,source_records=((1,source),))
staged._ORIGINAL_SYSTEM+=' changed'
changed=contract.extraction_cache_key()!=before
staged._ORIGINAL_SYSTEM=staged._ORIGINAL_SYSTEM[:-8]
print(json.dumps({'failed':result.failed,'predicate':result.triples[0].predicate if result.triples else None,
    'stages':client.stages,'wire':client.wire,'ordinary':len(client.ordinary),
    'calls':result.completion_calls,'attempts':result.provider_attempts,
    'ground':result.grounding_calls,'rechecks':result.grounding_recheck_calls,
    'identity_changed':changed,'identity_restored':contract.extraction_cache_key()==before}))
''')
    assert {key: value[key] for key in ("failed", "predicate", "stages", "ordinary", "calls",
                                         "attempts", "ground", "rechecks", "identity_changed",
                                         "identity_restored")} == {
        "failed": False, "predicate": "prefers",
        "stages": [["original", False], ["alternatives", False], ["original", True]],
        "ordinary": 2, "calls": 5, "attempts": 5, "ground": 3, "rechecks": 1,
        "identity_changed": True, "identity_restored": True}
    assert [item["predicate"] for item in value["wire"]] == ["uses", "prefers"]


def test_budget_failure_is_atomic(candidate):
    target, _, _ = candidate
    value = _run(target, r'''
import json,sys
sys.path.insert(0,sys.argv[1])
from hymem.extraction.chunk import extract_chunk
from hymem.extraction import grounding_staged_v1 as staged
source=json.dumps({'source_record_version':'hymem-claim-source-v2','source_message_id':1,
    'source_role':'user','source_peer_id':'Ada','content':'I prefer tea.'})
claim={'subject':'Ada','predicate':'uses','object':'tea','polarity':1,'source_message_id':1}
class Client:
    def __init__(self): self.n=0; self.stages=[]
    def complete(self,request):
        self.n+=1
        return json.dumps({'triples':[claim] if self.n==1 else [],'markers':[],'complete':True})
    def complete_stage(self,request,batch,stage,recheck):
        self.stages.append((stage,recheck))
        return json.dumps({'schema':staged.ORIGINAL_SCHEMA,'batch_sha256':batch.batch_sha256,
            'complete':True,'originals':[{'index':0,'original':{'state':'not_established','support':None}}]})
client=Client()
result=extract_chunk(client,source,source_records=((1,source),),completion_call_limit=3)
print(json.dumps({'failed':result.failed,'reason':result.failure_reason,'triples':len(result.triples),
    'markers':len(result.markers),'calls':result.completion_calls,'attempts':result.provider_attempts,
    'stages':client.stages}))
''')
    assert value == {"failed": True, "reason": "resource_limit", "triples": 0, "markers": 0,
                     "calls": 3, "attempts": 3, "stages": [["original", False]]}


@pytest.mark.parametrize("mode,expected,stages", [
    ("malformed", "contract:response_shape", [["original", False]]),
    ("collision", "correction:collision", [["original", False], ["alternatives", False]]),
])
def test_isolated_gate_failure_branches(candidate, mode, expected, stages):
    target, _, _ = candidate
    value = _run(target, r'''
import json,sys
sys.path.insert(0,sys.argv[1])
from hymem.extraction import grounding_staged_gate_v1 as gate, grounding_staged_v1 as staged
from hymem.extraction import grounding_classification_v4 as v4
from hymem.extraction.triples import Triple
mode=''' + repr(mode) + r'''
triples=[Triple('Ada','uses','tea',1,source_message_id=1),
         Triple('Ada','prefers','tea',1,source_message_id=1)]
source=json.dumps({'source_record_version':'hymem-claim-source-v2','source_message_id':1,
    'source_role':'user','source_peer_id':'Ada','content':'I prefer tea.'})
quote={'source_message_id':1,'region':'owned','quote':'I prefer tea.'}
support={'evidence':[quote], 'checks':{
    'attribution_and_roles':{'state':'supported','evidence_indices':[0]},
    'relation_and_polarity':{'state':'supported','evidence_indices':[0]}}}
calls=[]
def invoke(request,batch,stage,recheck):
    calls.append((stage,recheck))
    if mode=='malformed': return 'not json'
    if stage=='original':
        rows=[{'index':i,'original':{'state':'not_established','support':None} if i==0 else
               {'state':'supported','support':support}} for i in range(len(batch.triples))]
        return json.dumps({'schema':staged.ORIGINAL_SCHEMA,'batch_sha256':batch.batch_sha256,
                           'complete':True,'originals':rows})
    rows=[]
    for i in batch.negative_indices:
        values={p:{'state':'not_established','support':None}
                for p in v4.PREDICATE_ORDER if p!='uses'}
        values['prefers']={'state':'supported','support':support}
        rows.append({'index':i,'alternatives':values})
    return json.dumps({'schema':staged.ALTERNATIVES_SCHEMA,
        'batch_sha256':batch.classification_batch.batch_sha256,
        'original_response_sha256':batch.original_response_sha256,
        'complete':True,'alternatives':rows})
try: gate.ground_triples(triples,((1,source),),(),source,invoke)
except gate.GroundingGateError as exc: code=exc.code
else: code='accepted'
print(json.dumps({'code':code,'stages':calls,'unchanged':[triple.predicate for triple in triples]}))
''')
    assert value == {"code": expected, "stages": stages, "unchanged": ["uses", "prefers"]}


def test_pins_and_freshness_fail_before_output(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    target, stamp = root / "candidate", root / "map.json"
    monkeypatch.setitem(builder.HELPERS, builder.CLASSIFICATION, "0" * 64)
    with pytest.raises(ValueError, match="helper_source_drift"):
        builder.prepare(SOURCE, SOURCE_STAMP, ACCEPTED, ACCEPTED_STAMP,
                        target, stamp, REPO)
    assert not target.exists() and not stamp.exists()


@pytest.mark.parametrize("mutation", ["staged_parser", "staged_schema", "gate", "v4_schema", "source_mapper"])
def test_consumed_helper_change_invalidates_identity(candidate, mutation):
    target, _, _ = candidate
    value = _run(target, r'''
import json,sys
sys.path.insert(0,sys.argv[1])
from hymem.extraction import contract,chunk
from hymem.extraction import grounding_staged_v1 as staged
from hymem.extraction import grounding_classification_v4 as v4
from hymem.extraction import grounding_gate as source_gate
before=contract.extraction_cache_key()
case=''' + repr(mutation) + r'''
if case=='staged_parser': staged.parse_staged_responses=lambda *a,**k: None
elif case=='staged_schema': staged.build_original_output_schema=lambda *a,**k: {}
elif case=='gate': chunk.ground_triples=lambda *a,**k: []
elif case=='v4_schema': v4.build_output_schema=lambda *a,**k: {}
elif case=='source_mapper': source_gate._source.__code__=(lambda *a,**k: None).__code__
try: after=contract.extraction_cache_key()
except (RuntimeError,ValueError): print(json.dumps('closed'))
else: print(json.dumps('changed' if after!=before else 'reused'))
''')
    assert value in {"closed", "changed"}


def test_direct_cli_help():
    proc = subprocess.run([sys.executable, "-I", "-B", str(Path(builder.__file__)), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "--accepted-stamp" in proc.stdout
