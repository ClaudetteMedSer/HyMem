import copy
from pathlib import Path
import subprocess
import types

import pytest

M=types.ModuleType('finalizer_tests')
M.__file__=str(Path(__file__).resolve().parents[1]/'lme_r9_full500_finalizer.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)

def test_long_wait_keeps_one_client_and_small_timeouts():
    calls=[];clock=[100.0]
    class Client:
        returncode=None
        def communicate(self,timeout):
            calls.append(timeout)
            assert 0<timeout<=30
            if len(calls)<4:
                clock[0]+=timeout
                raise subprocess.TimeoutExpired('docker wait',timeout)
            self.returncode=0
            return b'137\n',b''
        def poll(self):return self.returncode
    created=[]
    def popen(argv,**kwargs):
        created.append(argv);return Client()
    assert M.wait_existing(M.LIVE_CID,100+2592060,popen=popen,now=lambda:clock[0])==137
    assert created==[['docker','wait',M.LIVE_CID]] and len(calls)==4

def test_deadline_terminates_only_wait_client():
    clock=[0.0];actions=[]
    class Client:
        returncode=None
        def communicate(self,timeout):
            assert timeout<=30
            if actions:
                self.returncode=-15;return b'',b''
            clock[0]+=timeout
            raise subprocess.TimeoutExpired('docker wait',timeout)
        def poll(self):return self.returncode
        def terminate(self):actions.append('terminate_wait_client')
        def kill(self):actions.append('kill_wait_client')
    with pytest.raises(RuntimeError,match='wait_expired'):
        M.wait_existing(M.LIVE_CID,60,popen=lambda *a,**k:Client(),now=lambda:clock[0])
    assert actions==['terminate_wait_client']

def setup(*,code=0,oom=False,ambiguous=None):
    calls=[];records={}
    state={'status':'exited','pid':0,'exit_code':code,'oom_killed':oom}
    controller=types.SimpleNamespace(inspect=lambda *a:copy.deepcopy(state))
    def record(name,value):
        if name in records:raise FileExistsError(name)
        records[name]=value
    def waiting(cid,deadline):
        calls.append(('wait',cid));return code if cid==M.LIVE_CID else 0
    def sending(cont,cfg,action,cid=None):
        calls.append((action,'validation',cid))
        if action==ambiguous:raise RuntimeError('ambiguous_dispatch')
        return {'container_id':cid or 'a'*64,'status':'created' if action=='create' else 'running' if action=='start' else 'exited',
                'pid':0,'exit_code':0,'oom_killed':False}
    return controller,object(),record,lambda *a:({}, {}, 100),waiting,sending,calls,records

@pytest.mark.parametrize('code,oom',[(0,False),(1,False),(137,True),(0,True)])
def test_existing_paid_run_terminal_always_validated_once(code,oom):
    a=setup(code=code,oom=oom)
    out=M.pipeline(*a[:2],record=a[2],admitting=a[3],waiting=a[4],sending=a[5])
    assert out['status']=='offline_validation_finished' and out['paid_runs_started']==0
    assert out['live_success'] is (code==0 and not oom)
    assert out['validation_runs_started']==1
    assert [x[:2] for x in a[6] if x[0]!='wait']==[('create','validation'),('start','validation'),('status','validation')]

@pytest.mark.parametrize('action',['create','start'])
def test_ambiguous_dispatch_never_retries(action):
    a=setup(ambiguous=action)
    out=M.pipeline(*a[:2],record=a[2],admitting=a[3],waiting=a[4],sending=a[5])
    assert out['status']=='operator_pending' and out['automatic_retry'] is False
    assert len([x for x in a[6] if x[0]==action])==1
    assert out['validation_runs_started'] is (None if action=='start' else 0)

def test_repeat_intent_prevents_every_operation():
    a=setup();a[2]('intent.json',{})
    with pytest.raises(FileExistsError):
        M.pipeline(*a[:2],record=a[2],admitting=a[3],waiting=a[4],sending=a[5])
    assert not a[6]

def test_wait_timeout_stops_before_validation():
    a=setup()
    def waiting(*args):raise TimeoutError()
    out=M.pipeline(*a[:2],record=a[2],admitting=a[3],waiting=waiting,sending=a[5])
    assert out['operator_inspection_required'] is True and out['paid_runs_started']==0
    assert not a[6]

def test_dispatch_has_no_live_mode():
    calls=[]
    continuation=types.SimpleNamespace(dispatch=lambda *args:calls.append(args))
    M.validation_dispatch(continuation,{},'create')
    assert calls[0][3]=='validation'
    with pytest.raises(RuntimeError):M.validation_dispatch(continuation,{},'live')

def test_validation_oom_never_claims_success_even_exit_zero():
    a=setup()
    def sending(*args):
        state=a[5](*args)
        if args[2]=='status':state['oom_killed']=True
        return state
    out=M.pipeline(*a[:2],record=a[2],admitting=a[3],waiting=a[4],sending=sending)
    assert out['status']=='operator_pending' and out['error_type']=='RuntimeError'
    assert out.get('validation_success') is not True
    assert out['validation_runs_started']==1 and out['automatic_retry'] is False

@pytest.mark.parametrize('change',[{'pid':1},{'status':'running'},{'exit_code':1},{'oom_killed':0}])
def test_terminal_state_strict(change):
    state={'status':'exited','pid':0,'exit_code':0,'oom_killed':False};state.update(change)
    with pytest.raises(RuntimeError):M.terminal(state,0)
