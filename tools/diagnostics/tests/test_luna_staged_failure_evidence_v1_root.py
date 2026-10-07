"""Root independent cue/applicability tests on the physical canary candidate."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.diagnostics import luna_staged_bundle_v1 as bundle
from tools.diagnostics import luna_staged_failure_evidence_v1 as helper


@pytest.mark.parametrize('fault',['none','missing_left','missing_right','prefix'])
def test_actual_derived_boundary_support_is_checked_not_assumed(tmp_path,fault):
    repo=Path(__file__).resolve().parents[3]
    candidate=tmp_path/'candidate';shutil.copytree(bundle.CANDIDATE,candidate)
    code=tmp_path/'code'
    for name,raw in bundle.collect_sources(repo).items():
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    tree=ast.parse(Path(__file__).with_name('test_luna_staged_core_v1_root.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and
        n.name=='test_actual_candidate_derivation_preserves_initial_batches_and_exact_ordinary_bytes')
    script=next(ast.literal_eval(n.value) for n in function.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
    script+=r'''
import importlib.util
from dataclasses import replace
spec=importlib.util.spec_from_file_location('root_evidence_helper',sys.argv[3])
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
batch=batches[1];source=batch.sources[0];fault=sys.argv[4]
if fault=='missing_left':
    source=replace(source,contexts=tuple(replace(ctx,content=ctx.content.replace(
        gold._PROSE_BOUNDARY_LEFT,' '*len(gold._PROSE_BOUNDARY_LEFT))) for ctx in source.contexts))
elif fault=='missing_right':
    source=replace(source,content=source.content.replace(gold._PROSE_BOUNDARY_RIGHT,
        ' '*len(gold._PROSE_BOUNDARY_RIGHT)))
elif fault=='prefix':
    source=replace(source,contexts=tuple(replace(ctx,owned_prefix_chars=1) for ctx in source.contexts))
_,batch=s.build_original_request(batch.triples,(source,))
original=json.dumps(dict(schema=s.ORIGINAL_SCHEMA,batch_sha256=batch.batch_sha256,
    complete=True,originals=[dict(index=0,original=dict(state='not_established',support=None))]))
_,alt=s.build_alternatives_request(batch,original)
raw=json.dumps(dict(schema=s.ALTERNATIVES_SCHEMA,batch_sha256=batch.batch_sha256,
    original_response_sha256=alt.original_response_sha256,complete=True,
    alternatives=[dict(index=0,alternatives={p:dict(state='not_established',support=None)
        for p in g.PREDICATE_ORDER if p!=batch.triples[0].predicate})]))
events=[{}, {}, dict(phase='before_dispatch'),dict(phase='response_returned',response=raw)]
out=helper._prose(c,gold,batch,original,events)
assert out['original_negative'] and out['alternatives_all_negative']
assert out['mechanical_candidate_valid']==out['mechanical_candidate_accepted']==(fault=='none')
if fault=='none':assert out['left_cue_in_bound_context'] and out['right_cue_in_owned'] and out['owned_quote_within_context_prefix']
if fault=='missing_left':assert not out['left_cue_in_bound_context']
if fault=='missing_right':assert not out['right_cue_in_owned']
if fault=='prefix':assert not out['owned_quote_within_context_prefix']
print(json.dumps(out,sort_keys=True))
'''
    out=subprocess.run([sys.executable,'-I','-B','-c',script,str(candidate),str(code),helper.__file__,fault],
        capture_output=True,text=True,timeout=45)
    assert out.returncode==0,out.stderr
    value=json.loads(out.stdout.splitlines()[-1])
    assert value['mechanical_candidate_accepted']==(fault=='none')


def test_bad_cli_never_emits_private_path_or_calls(tmp_path):
    out=subprocess.run([sys.executable,'-I','-B',helper.__file__,'--root',str(tmp_path/'PRIVATE'),
        '--receipt-sha256','0'*64,'--entry-sha256','0'*64,'--result-sha256','0'*64],
        capture_output=True,text=True,timeout=15)
    assert out.returncode==1 and out.stderr=='' and 'PRIVATE' not in out.stdout
    value=json.loads(out.stdout)
    assert value['verified'] is False and value['new_model_calls']==0


def test_complete_private_journal_path_uses_actual_replay_and_binding(monkeypatch,tmp_path):
    from tools.diagnostics import luna_staged_run_v1 as entry,luna_staged_replay_v1 as replay
    from tools.diagnostics.tests.test_luna_staged_replay_v1_root import execute_fixture
    from benchmarks import extraction_canary as canary
    root,entry_sha,terminal=execute_fixture(monkeypatch,tmp_path)
    receipt,loaded,proof,retained=entry.preflight(root,'1'*64)
    loaded=(*loaded[:4],canary,*loaded[5:])
    monkeypatch.setattr(builtins,'_staged_root_test_preflight',lambda *a,**k:(receipt,loaded,proof,retained))
    path=root/'code/tools/diagnostics/luna_staged_replay_v1.py'
    path.write_bytes(Path(replay.__file__).read_bytes())
    receipt['source_sha256']['tools/diagnostics/luna_staged_replay_v1.py']=hashlib.sha256(path.read_bytes()).hexdigest()
    digest=hashlib.sha256((root/'run/private-result.json').read_bytes()).hexdigest()
    out=helper.verify(root,'1'*64,entry_sha,digest)
    assert out['verified'] and out['new_model_calls']==0
    assert out['role_control']['original_states']==['not_established','not_established']
    # Synthetic fixture deliberately does not contain the physical boundary cues.
    assert not out['prose_canary']['mechanical_candidate_accepted']
    path=next(p for p in (root/'run/private-journal').glob('*unit-07.json')
        if json.loads(p.read_bytes())['phase']=='before_dispatch')
    value=json.loads(path.read_bytes());value['batch_sha256']='0'*64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        helper.verify(root,'1'*64,entry_sha,digest)
