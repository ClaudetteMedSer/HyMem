import copy
import json
from pathlib import Path
import types

import pytest

HERE = Path(__file__).resolve().parents[1]
M = types.ModuleType('continuation_test')
M.__file__ = str(HERE/'lme_r8_headless_continuation.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
CFG = json.loads((HERE/'lme_r8_headless_continuation_config.json').read_text())

def setup(monkeypatch, *, suite_code=0, live_code=0, deny=False, ambiguous=False):
    cfg = copy.deepcopy(CFG)
    manifest = {'source_sha256':{str(i):'a'*64 for i in range(503)},
                'approved_source_manifest_sha256':cfg['source_pin']}
    calls, receipts = [], {}
    controller = types.SimpleNamespace(definitions=lambda:(manifest,object()),
        js=lambda p:{'container_id':cfg['preflight_cid']},
        candidate_evidence=lambda g,m: (_ for _ in ()).throw(RuntimeError('deny')) if deny else None,
        live_gates=lambda m,h:None)
    suite = types.SimpleNamespace(verify=lambda *a:None,
        inspect=lambda *a:{'status':'exited','pid':0,'exit_code':suite_code,'oom_killed':False})
    def read(p):
        if p.name == 'launch.json':
            return json.dumps({'manifest_sha256':cfg['evidence']['suite']['manifest_sha256'],
                               'container_id':cfg['suite_cid']}).encode()
        return b'{}'
    monkeypatch.setattr(M,'read',read)
    monkeypatch.setattr(M,'gate',lambda *a:{})
    monkeypatch.setattr(M,'save',lambda *a:None)
    def record(n,v):
        if n in receipts: raise FileExistsError(n)
        receipts[n] = v
    def waiting(cid,seconds):
        calls.append(('wait',cid,seconds))
        return suite_code if cid == cfg['suite_cid'] else live_code if cid == '1'*64 else 0
    def sending(host,cfg,action,mode,cid=None,gate_pin=None):
        calls.append((action,mode,cid))
        if ambiguous: raise RuntimeError('ambiguous')
        cid = cid or ('1'*64 if mode == 'live' else '2'*64)
        return {'container_id':cid,'status':'created' if action == 'create' else 'running' if action == 'start' else 'exited',
                'pid':0,'exit_code':live_code if mode == 'live' else 0,'oom_killed':False}
    return cfg,controller,suite,record,waiting,sending,calls,receipts

@pytest.mark.parametrize('failure',['suite','gate','ambiguous'])
def test_denial_never_retries(monkeypatch,failure):
    a=setup(monkeypatch,suite_code=1 if failure=='suite' else 0,
            deny=failure=='gate',ambiguous=failure=='ambiguous')
    result=M.pipeline(*a[:4],waiting=a[4],sending=a[5])
    assert result['status']=='operator_pending'
    dispatched=[c for c in a[6] if c[0]!='wait']
    assert dispatched == ([('create','live',None)] if failure=='ambiguous' else [])
    assert result['paid_runs_started']==0

def test_failed_live_still_validated_with_exact_deadlines(monkeypatch):
    a=setup(monkeypatch,live_code=1)
    result=M.pipeline(*a[:4],waiting=a[4],sending=a[5])
    assert result['status']=='offline_validation_finished'
    assert result['live_success'] is False and result['validation_success'] is True
    assert [c[2] for c in a[6] if c[0]=='wait']==[10860,32460,3600]
    assert [(c[0],c[1]) for c in a[6] if c[0]=='create']==[('create','live'),('create','validation')]

def test_repeat_intent_cannot_dispatch(monkeypatch):
    a=setup(monkeypatch)
    M.pipeline(*a[:4],waiting=a[4],sending=a[5])
    before=list(a[6])
    with pytest.raises(FileExistsError):
        M.pipeline(*a[:4],waiting=a[4],sending=a[5])
    assert a[6]==before

def test_duplicate_active_cannot_write_any_receipt(monkeypatch):
    a=setup(monkeypatch);a[3]('intent.json',{'original':True})
    before=copy.deepcopy(a[7])
    with pytest.raises(FileExistsError):
        M.pipeline(*a[:4],waiting=a[4],sending=a[5])
    assert a[7]==before and not a[6] and 'result.json' not in a[7]

@pytest.mark.parametrize('mode',['live','validation'])
def test_ambiguous_start_counts_unknown(monkeypatch,mode):
    a=setup(monkeypatch)
    def sending(*args,**kwargs):
        if args[2:4]==('start',mode):raise RuntimeError('ambiguous_start')
        return a[5](*args,**kwargs)
    result=M.pipeline(*a[:4],waiting=a[4],sending=sending)
    key='paid_runs_started' if mode=='live' else 'validation_runs_started'
    assert result[key] is None and result[mode+'_start_attempted'] is True
    assert mode+'-start-requested.json' in a[7]
    assert result['status']=='operator_pending' and M.exit_status(result)==1

def test_process_exit_honesty():
    assert M.exit_status({'status':'operator_pending'})==1
    assert M.exit_status({'status':'offline_validation_finished','live_success':False,'validation_success':True})==1
    assert M.exit_status({'status':'offline_validation_finished','live_success':True,'validation_success':True})==0

def test_timeout_marks_pending_no_more_dispatch(monkeypatch):
    a=setup(monkeypatch)
    def timeout(*args): raise TimeoutError()
    result=M.pipeline(*a[:4],waiting=timeout,sending=a[5])
    assert result['operator_inspection_required'] and not a[6]

@pytest.mark.parametrize('key,value',[
    ('repair_control_review_passed',False),('summary_control_review_passed',1),
    ('expected_collected',7871),('source_files',502),('suite_cid','bad'),
    ('root',str(M.BASE)),('target',str(M.BASE/'production'))])
def test_config_admission(key,value):
    cfg=copy.deepcopy(CFG);cfg[key]=value
    with pytest.raises(RuntimeError):M.checked_config(cfg)

def test_pin_loading_and_private_exclusive_receipts(tmp_path):
    p=tmp_path/'code.py'; p.write_bytes(b'x=1\n')
    with pytest.raises(RuntimeError):M.load(p,'0'*64,'bad_pin')
    out=tmp_path/'receipt.json'; M.save(out,{'safe':True})
    assert out.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):M.save(out,{})

@pytest.mark.parametrize('change',[{'pid':1},{'oom_killed':True},{'exit_code':1},{'status':'running'}])
def test_terminal_state(change):
    state={'pid':0,'oom_killed':False,'exit_code':0,'status':'exited'};state.update(change)
    with pytest.raises(RuntimeError):M.terminal(state,0)
