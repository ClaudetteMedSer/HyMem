"""Independent source-copy/import and strict-gate controls, with invented data.

Mocked verdicts test contract mechanics, not model semantic accuracy.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tools.diagnostics import siwc_extraction_prompt_repair_v1 as repair


FROZEN = Path("/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle")
PROMPT = "hymem/extraction/prompts/__init__.py"


def test_actual_candidate_import_and_unchanged_strict_grounding(tmp_path):
    stamp = json.loads((FROZEN / "source-map.json").read_text())
    entries = stamp["source_sha256"]
    assert len(entries) == 514
    for relative, digest in entries.items():
        assert hashlib.sha256((FROZEN / "candidate" / relative).read_bytes()).hexdigest() == digest
    candidate = tmp_path / "candidate"
    shutil.copytree(FROZEN / "candidate", candidate)
    prompt = candidate / PROMPT
    prompt.write_bytes(repair.transform_prompt_source(prompt.read_bytes()))
    changed = [relative for relative, digest in entries.items()
               if hashlib.sha256((candidate / relative).read_bytes()).hexdigest() != digest]
    assert changed == [PROMPT]
    script = r'''
import sys,json,hashlib
from pathlib import Path
def audit(event,args):
    if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:
        raise AssertionError('external_action_forbidden')
sys.addaudithook(audit)
candidate=Path(sys.argv[1]); sys.path.insert(0,str(candidate))
from hymem.extraction import prompts,grounding_staged_gate_v1 as gate,grounding_staged_v1 as staged
from hymem.extraction.grounding_classification_v4 import PREDICATE_ORDER
from hymem.extraction.triples import Triple
from hymem.extraction.contract import extraction_contract_identity
assert Path(prompts.__file__).resolve()==candidate/'hymem/extraction/prompts/__init__.py'
primary=prompts.build_chunk_extraction_system()
assert 'I own a Ford F-150' in primary and 'including clearly entailed implicit wording' in primary
assert '"I drive a Ford F-150"             -> (user, owns, ford_f_150)' not in primary
assert prompts.build_chunk_empty_verification_system().startswith(primary)
assert prompts.build_chunk_omission_verification_system().startswith(primary)
identity=extraction_contract_identity()
assert identity!='hymem-extraction-contract-sha256-v1:94a1adfc028694e9b1868ec8712baff38d4d7a99a9d2e3547c66819711890f95'
def execute(content,subject,predicate,obj,state,quote=None):
    calls=[]
    triple=Triple(subject,predicate,obj,1,source_message_id=7)
    def assessment(s):
        if s!='supported': return {'state':s,'support':None}
        return {'state':s,'support':{'evidence':[{'source_message_id':7,'region':'owned','quote':quote or content}],
          'checks':{k:{'state':'supported','evidence_indices':[0]} for k in ('attribution_and_roles','relation_and_polarity')}}}
    def invoke(request,batch,stage,recheck):
        calls.append((stage,recheck))
        if stage=='original':
            staged.validate_original_request(request,batch)
            return json.dumps({'schema':staged.ORIGINAL_SCHEMA,'batch_sha256':batch.batch_sha256,'complete':True,
              'originals':[{'index':0,'original':assessment(state)}]})
        staged.validate_alternatives_request(request,batch)
        return json.dumps({'schema':staged.ALTERNATIVES_SCHEMA,'batch_sha256':batch.classification_batch.batch_sha256,
          'original_response_sha256':batch.original_response_sha256,'complete':True,
          'alternatives':[{'index':0,'alternatives':{p:assessment('not_established') for p in PREDICATE_ORDER if p!=predicate}}]})
    try:
        result=gate.ground_triples([triple],((7,json.dumps({'content':content,'source_role':'user'})),),(),'ignored',invoke)
        assert result==[triple]
        code='accepted'
    except gate.GroundingGateError as exc: code=exc.code
    return code,calls
for content,subject,predicate,obj in (
    ('Mira owns a bicycle.','Mira','owns','bicycle'),
    ('Soren is a member of the Cedar team.','Soren','part_of','Cedar team'),
    ('Package Cedar contains module Birch.','Package Cedar','contains','module Birch')):
    code,calls=execute(content,subject,predicate,obj,'supported')
    assert code=='accepted' and calls==[('original',False)]
    code,calls=execute(content,subject,predicate,obj,'supported','a quote absent from the source')
    assert code=='contract:evidence_quote_missing' and calls==[('original',False)]
for content,subject,predicate,obj in (
    ('Mira drives a rented bicycle.','Mira','owns','bicycle'),
    ('Soren maintains Cedar as an external contractor.','Soren','part_of','Cedar'),
    ('The Elm team is responsible for the Birch service.','Elm team','contains','Birch service')):
    code,calls=execute(content,subject,predicate,obj,'not_established')
    assert code=='verdict:unsupported' and calls==[('original',False),('alternatives',False)]
print(json.dumps({'candidate_contract_identity':identity,'fixture_controls':9,'model_calls':0,'external_actions':0}))
'''
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(candidate)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    print(json.dumps(result, sort_keys=True))
    assert result["fixture_controls"] == 9
    assert result["model_calls"] == result["external_actions"] == 0
    for relative, digest in entries.items():
        assert hashlib.sha256((FROZEN / "candidate" / relative).read_bytes()).hexdigest() == digest
