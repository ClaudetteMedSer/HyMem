"""Root: real runner/core journals, invented canary inputs, offline replay faults."""
import builtins
import hashlib
import json
from pathlib import Path

import pytest

from tools.diagnostics import luna_staged_replay_v1 as replay
from tools.diagnostics import luna_staged_run_v1 as entry,luna_staged_core_v1 as core
from tools.diagnostics.tests.test_luna_staged_run_v1_root import setup_entry
from tools.diagnostics.tests.test_luna_staged_core_v1 import FakeClient,_canary_batches
from tools.diagnostics.tests.test_luna_staged_core_v1_root import H
from tools.diagnostics import luna_semantic_cases as cases


def execute_fixture(monkeypatch,tmp_path,*,fault='none',ordinal=3,at=1,mode='negative',journal_fault=None):
    root,host,reference,_=setup_entry(monkeypatch,tmp_path)
    batches=_canary_batches()
    def derive(**kw):
        for i in range(8):kw['record'](dict(phase='ordinary_replay',ordinal=i,synthetic=True))
        kw['record'](dict(phase='canary_batches_derived',ordinary_replays=8,
            synthetic_table_advancement=True,table_batch_sha256=batches[0].batch_sha256,
            prose_batch_sha256=batches[1].batch_sha256))
        return batches
    monkeypatch.setattr(core,'derive_canary_batches',derive)
    if journal_fault:
        record=core.PrivateJournal.record
        def maybe_record(self,key,value):
            if key==f'unit-{ordinal:02d}' and value['phase']==journal_fault:
                raise core.DiagnosticStop('private_evidence_write_failure')
            return record(self,key,value)
        monkeypatch.setattr(core.PrivateJournal,'record',maybe_record)
    class Client(FakeClient):
        def __init__(self,key,cap,budget):
            super().__init__(key,cap,budget);self.seen=0
        def complete_stage(self,request,batch,stage,recheck):
            self.seen+=1
            active=self.key==f'unit-{ordinal:02d}' and self.seen==at
            if active and fault in ('unknown_usage','overshoot'):
                warm=core.transport.warm
                self.budget.reserve(self.key)
                self.budget.before_turn(self.key,dict(auth='chatgpt',model=warm.base.MODEL,
                    config_isolation_admitted=True,inference_enabled=False,
                    quota_windows=[dict(remaining_percent=90)]))
                self.budget.settle(self.key,used=None if fault=='unknown_usage' else 500001,turn_started=True)
                if stage=='original':payload=H['original'](batch)
                else:_,_,payload=H['alternatives'](batch.classification_batch,json.loads(batch.original_response_canonical_json))
                raw=json.dumps(payload)
            else:raw=super().complete_stage(request,batch,stage,recheck)
            if active and fault=='transport':raise RuntimeError('PRIVATE_DO_NOT_EXPORT')
            if active and fault=='malformed':return 'PRIVATE_DO_NOT_EXPORT'
            if mode=='gold':
                bound=batch;batch=bound if stage=='original' else bound.classification_batch
                unit=core.schedule()[int(self.key.rsplit('-',1)[-1])]
                case=cases.cases()[unit.index] if unit.kind=='control' else None
                if case is not None:
                    predicates=[e.predicate if case.category=='correction' else t.predicate
                        if case.category=='supported' else None for t,e in zip(batch.triples,case.expected,strict=True)]
                    pools=[[H['quote'](t.source_message_id,text,region) for region,text in e.evidence]
                        for t,e in zip(batch.triples,case.expected,strict=True)]
                else:
                    predicates=['deploys_to' if unit.kind=='table_canary' else 'prefers']
                    pools=[[H['quote'](t.source_message_id,batch.sources[0].content)] for t in batch.triples]
                if stage=='original':
                    payload=H['original'](batch,['supported' if p==t.predicate else 'not_established'
                        for p,t in zip(predicates,batch.triples,strict=True)],pools)
                else:
                    _,rebound,payload=H['alternatives'](batch,json.loads(bound.original_response_canonical_json),
                        positive=[(i,p) for i,p in enumerate(predicates) if p is not None and p!=batch.triples[i].predicate],pools=pools)
                    assert rebound==bound
                raw=json.dumps(payload)
            return raw
        def close(self):
            super().close()
            if fault=='cleanup' and self.key==f'unit-{ordinal:02d}':raise RuntimeError('PRIVATE_CLEANUP')
    terminal=entry.execute(root,'1'*64,client_factory=Client,containment=lambda *a:None)
    monkeypatch.setattr(builtins,'_staged_root_test_preflight',entry.preflight,raising=False)
    shim=root/'code/tools/diagnostics/luna_staged_run_v1.py';shim.parent.mkdir(parents=True)
    shim.write_text('import builtins\npreflight=builtins._staged_root_test_preflight\n')
    return root,hashlib.sha256(shim.read_bytes()).hexdigest(),terminal


@pytest.mark.parametrize('fault',['none','malformed','transport','cleanup'])
@pytest.mark.parametrize('ordinal',[0,3,7])
@pytest.mark.parametrize('at',[1,2])
def test_real_stage_journal_replays_every_return_and_preserves_partial_failure(monkeypatch,tmp_path,fault,ordinal,at):
    root,digest,terminal=execute_fixture(monkeypatch,tmp_path,fault=fault,ordinal=ordinal,at=at)
    proof=replay.replay(root,'1'*64,digest)
    assert proof['verified'] and proof['new_model_calls']==0
    assert proof['complete']==terminal['diagnostic_completed']==(fault in ('none','malformed'))
    assert proof['returned_responses']==terminal['paid_budget']['turns']-(fault=='transport')
    assert 'PRIVATE_' not in json.dumps(proof)


@pytest.mark.parametrize('field',['request','schema','prior','recheck','stage','response','evaluation',
    'unit_tokens','prior_tokens','reserved','flight','stopped','extra_file','missing_file','extra_field','symlink',
    'request_bool','recheck_int','prior_usage_int','first_failure'])
def test_replay_rejects_one_record_substitution(monkeypatch,tmp_path,field):
    root,digest,terminal=execute_fixture(monkeypatch,tmp_path)
    assert replay.replay(root,'1'*64,digest)['complete']
    directory=root/'run/private-journal'
    files=sorted(directory.glob('*unit-03.json'))
    rows=[json.loads(p.read_bytes()) for p in files]
    if field in ('extra_file','missing_file','symlink'):
        if field=='extra_file':(directory/'9999-unit-08.json').write_text('{}')
        elif field=='missing_file':files[1].unlink()
        else:files[1].unlink();files[1].symlink_to(files[0])
    else:
        index=2 if field in ('schema','prior','stage','recheck') else 0
        if field=='request':rows[index]['request']['user']+=' '
        elif field=='schema':rows[index]['output_schema_sha256']='0'*64
        elif field=='prior':rows[index]['prior_response_sha256']='0'*64
        elif field=='recheck':rows[index]['recheck']=True
        elif field=='stage':rows[index]['stage']='original'
        elif field=='request_bool':rows[index]['request']['temperature']=False
        elif field=='recheck_int':rows[index]['recheck']=0
        elif field=='response':index=1;rows[index]['response']='{}'
        elif field=='evaluation':
            index=next(i for i,r in enumerate(rows) if r['phase']=='evaluated')
            rows[index]['evaluation']['outcome']='accepted'
        elif field=='extra_field':rows[index]['unreviewed']='PRIVATE'
        else:
            index=len(rows)-1;snapshot=rows[index]['budget']
            if field=='unit_tokens':snapshot['questions']['unit-03']['known_tokens']+=1
            elif field=='prior_tokens':snapshot['questions']['unit-01']['known_tokens']+=1
            elif field=='reserved':snapshot['reserved']=1
            elif field=='flight':snapshot['in_flight']=1
            elif field=='prior_usage_int':snapshot['questions']['unit-01']['usage_complete']=1
            elif field=='first_failure':snapshot['first_failure']={'raw':'PRIVATE'}
            else:snapshot['stopped']=True
        files[index].write_text(json.dumps(rows[index]))
    with pytest.raises((ValueError,core.DiagnosticStop)):replay.replay(root,'1'*64,digest)


def test_unit_blocks_must_follow_the_frozen_chronological_schedule(monkeypatch,tmp_path):
    root,digest,terminal=execute_fixture(monkeypatch,tmp_path)
    assert replay.replay(root,'1'*64,digest)['complete']
    directory=root/'run/private-journal'
    saved=[(p.name,p.read_bytes()) for p in sorted(directory.iterdir())]
    def group(name):return name.split('-',1)[1]
    blocks={}
    for name,raw in saved:blocks.setdefault(group(name),[]).append(raw)
    for path in directory.iterdir():path.unlink()
    ordinal=1
    for key in ['derivation.json','unit-01.json','unit-00.json']+[f'unit-{i:02d}.json' for i in range(2,8)]:
        for raw in blocks[key]:
            (directory/f'{ordinal:04d}-{key}').write_bytes(raw);ordinal+=1
    with pytest.raises((ValueError,core.DiagnosticStop)):replay.replay(root,'1'*64,digest)


@pytest.mark.parametrize('fault',['none','malformed','transport'])
def test_third_stage_recheck_and_correction_replayed_exactly(monkeypatch,tmp_path,fault):
    root,digest,terminal=execute_fixture(monkeypatch,tmp_path,mode='gold',fault=fault,ordinal=1,at=3)
    proof=replay.replay(root,'1'*64,digest)
    assert proof['verified'] and proof['new_model_calls']==0
    assert proof['complete']==(fault!='transport')
    assert proof['returned_responses']==terminal['paid_budget']['turns']-(fault=='transport')
    if fault=='none':
        assert terminal['paid_budget']['turns']==17 and terminal['expected_gold_matches']==8
    elif fault=='malformed':assert terminal['expected_gold_matches']==7


@pytest.mark.parametrize('field',['corrected_request','final_predicate','gold'])
def test_replayed_correction_cannot_be_substituted_or_regraded(monkeypatch,tmp_path,field):
    root,digest,terminal=execute_fixture(monkeypatch,tmp_path,mode='gold')
    assert replay.replay(root,'1'*64,digest)['complete']
    files=sorted((root/'run/private-journal').glob('*unit-01.json'))
    if field=='corrected_request':
        path=files[4];record=json.loads(path.read_bytes());record['request']['user']+=' '
    else:
        path=root/'run/private-result.json';record=json.loads(path.read_bytes())
        if field=='gold':record['units'][1]['expected_gold_match']=False
        else:record['units'][1]['final_predicates']=['uses']
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError,core.DiagnosticStop)):replay.replay(root,'1'*64,digest)


@pytest.mark.parametrize('fault',['unknown_usage','overshoot'])
@pytest.mark.parametrize('ordinal,at,mode',[(0,1,'negative'),(3,2,'negative'),(1,3,'gold')])
def test_stopped_usage_is_reconciled_without_assuming_dispatch_means_admission(monkeypatch,tmp_path,fault,ordinal,at,mode):
    root,digest,terminal=execute_fixture(monkeypatch,tmp_path,fault=fault,ordinal=ordinal,at=at,mode=mode)
    proof=replay.replay(root,'1'*64,digest)
    assert proof['verified'] and not proof['complete'] and proof['new_model_calls']==0
    assert not terminal['diagnostic_completed']
    assert proof['returned_responses']==terminal['paid_budget']['turns']
    assert terminal['paid_budget']['usage_complete']==(fault!='unknown_usage')
    if fault=='overshoot':assert terminal['paid_budget']['known_tokens']>=500001


@pytest.mark.parametrize('phase',['before_dispatch','response_returned','evaluated','unit_finished'])
def test_incomplete_journal_cannot_prove_completed_even_when_usage_is_known(monkeypatch,tmp_path,phase):
    root,digest,terminal=execute_fixture(monkeypatch,tmp_path,journal_fault=phase)
    assert not terminal['diagnostic_completed']
    assert terminal['stop_code']=='private_evidence_write_failure'
    try:proof=replay.replay(root,'1'*64,digest)
    except (ValueError,core.DiagnosticStop):return
    assert not proof['complete'] and proof['new_model_calls']==0


def test_isolated_cli_invalid_input_reports_no_raw_details(tmp_path):
    import subprocess,sys
    out=subprocess.run([sys.executable,'-I','-B',replay.__file__,'--root',str(tmp_path/'PRIVATE_MISSING'),
        '--receipt-sha256','0'*64,'--entry-sha256','0'*64],capture_output=True,text=True,timeout=15)
    assert out.returncode==1 and out.stderr=='' and 'PRIVATE_' not in out.stdout
    value=json.loads(out.stdout)
    assert value['verified'] is False and value['complete'] is False and value['new_model_calls']==0
