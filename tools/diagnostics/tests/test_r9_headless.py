import copy
import hashlib
import json
from pathlib import Path
import types

import pytest

HERE = Path(__file__).resolve().parents[1]
M = types.ModuleType('continuation_test')
M.__file__ = str(HERE/'lme_r9_headless/continuation.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
# Private inert test pins/CIDs; never operational inputs.
CFG = {
    'schema': 'r9-headless-continuation-v1',
    'root': str(M.BASE/'lme-r9-headless-continuation-v1'),
    'target': str(M.BASE/'lme-r9-sample8-headless-v1'),
    'target_pin': 'a'*64, 'controller_pin': 'b'*64,
    'source_pin': '35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51',
    'source_files': 508, 'expected_collected': 7940,
    'suite_cid': 'c'*64, 'preflight_cid': 'd'*64,
    'semantic_review': copy.deepcopy(M.SEMANTIC_REVIEW),
    'evidence': {
        name: {'root': str(M.BASE/('lme-r9-full-suite-v1' if name == 'suite' else 'lme-r9-summary-candidate-v2')),
               'manifest_sha256': pin}
        for name, pin in M.EVIDENCE_PINS.items()
    },
}

def setup(monkeypatch, *, suite_code=0, live_code=0, deny=False, ambiguous=False):
    cfg = copy.deepcopy(CFG)
    manifest = {'source_sha256':{str(i):'a'*64 for i in range(508)},
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

@pytest.mark.parametrize('key,value',[
    ('schema','r8-headless-continuation-v1'),
    ('root',str(M.BASE/'lme-r8-headless-continuation-v1')),
    ('target',str(M.BASE/'lme-r8-sample8-headless-v2')),
    ('source_files',503),('expected_collected',7872)])
def test_old_campaign_admission_denied(key,value):
    cfg=copy.deepcopy(CFG);cfg[key]=value
    with pytest.raises(RuntimeError):M.checked_config(cfg)

def test_timeout_marks_pending_no_more_dispatch(monkeypatch):
    a=setup(monkeypatch)
    def timeout(*args): raise TimeoutError()
    result=M.pipeline(*a[:4],waiting=timeout,sending=a[5])
    assert result['operator_inspection_required'] and not a[6]

@pytest.mark.parametrize('key,value',[
    ('semantic_review',{}),
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


def module(path, name):
    obj = types.ModuleType(name)
    obj.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), obj.__dict__)
    return obj


R9 = HERE/'lme_r9_headless'
HOST = module(R9/'host_control.py', 'r9_host_tests')
RUN = module(R9/'bundle/q1_stock_run.py', 'r9_run_tests')
PLAN = module(R9/'bundle/q1_stock_host.py', 'r9_plan_tests')


def evidence():
    pins = {f'hymem/test_{i}.py': 'a'*64 for i in range(508)}
    manifest = {'source_sha256': pins, 'source_files': 508,
                'approved_source_manifest_sha256': HOST.SOURCE_PIN}
    records = {}
    gate = {'semantic_review': copy.deepcopy(HOST.SEMANTIC_REVIEW), 'evidence': {}}
    for name, pin in HOST.EVIDENCE_PINS.items():
        gate['evidence'][name] = {kind: {'path': name+'/'+kind, 'sha256': pin}
                                  for kind in ('manifest','receipt','supervisor')}
        records[name+'/manifest'] = {'source_sha256': copy.deepcopy(pins),
            'source_inventory_sha256': HOST.SOURCE_PIN}
        records[name+'/receipt'] = {'manifest_sha256': pin, 'source_unchanged': True}
        records[name+'/supervisor'] = {'manifest_sha256': pin,
            'package_and_source_unchanged': True, 'exception_type': None,
            'outcome': {'status':'completed','returncode':0,'safe_to_continue':True,
                'child_reaped':True,'group_absent_after_reap':True,
                'cleanup_complete':True,'errors':[]}}
    records['suite/manifest'].update(schema='r8-offline-full-suite-v1', expected_collected=7940)
    records['suite/receipt'].update(status='passed',tested_source_unchanged=True,
        full_runs_started=1,provider_calls=0,collect={'collected':7940},
        full={'collected':7940,'passed':7936,'skipped':4,'failed':0,'errors':0,'exit_code':0})
    bounds = {'completion_calls':12,'http_attempts':36,'invocation_timeout_seconds':120,'timeout_seconds':900}
    records['summary/manifest'].update(schema='r9-summary-output-cap-package-v1',
        bounds=bounds,rerolls_allowed=False,production_changes=False,
        model='deepseek-flash',endpoint='https://api.deepseek.com')
    receipt = records['summary/receipt']
    receipt.update(schema='r9-summary-output-cap-package-v1',status='review_pending',mode='live',
        completion_calls=9,provider_calls=9,http_attempts=9,
        usage={'calls':9,'request_attempts':9,'successful_responses':9,
            'calls_available':True,'request_attempts_available':True,'successful_responses_available':True},
        clients_closed=True,connections_closed=True,threads_clean=True,reference_unchanged=True,
        all_non_summary_clone_unchanged=True,target_summary_healthy=True,cleanup_errors=[],
        normal={'parse_failed':False,'failure_reason':None,'summary_failure_reason':None,'summary_chars':417},
        source_exact_repair={'failure_reason':None,'option_lengths':[282,233,191],'selected_chars':282},
        explicit_recovery={'calls':3,'provider_attempts':3,'published':1,'held':0,'exhausted':0},
        recovery_changes={'published_sessions':1,'recovered_sessions':1,'private_partial_sessions':0,'held_sessions':0})
    for i in range(4):
        receipt['control_'+str(i)] = {'failure_reason':None,'selected_chars':100}
    return gate, manifest, records


def test_exact_r9_targeted_evidence_admission(monkeypatch):
    gate, manifest, records = evidence()
    monkeypatch.setattr(HOST, 'evidence_record', lambda binding: records[binding['path']])
    HOST.candidate_evidence(gate, manifest)
    assert gate['semantic_review']['strong_coverage_checklist_passed'] is False
    assert gate['semantic_review']['r8_extraction_carried_forward'] is False
    assert 'summary_control_review_passed' not in gate


@pytest.mark.parametrize('mutation', [
    'suite_source','summary_source','suite_pin','summary_pin','old_schema','cleanup',
    'suite_failure','count_gap','bool_count','repair_failure','normal_degraded',
    'recovery_failure','accounting_unavailable','paid_limit','coverage_claim',
    'retention_omission','carried_extraction','extra_evidence','control_failure','reference_changed'])
def test_evidence_denials(monkeypatch, mutation):
    gate, manifest, records = evidence()
    r = records['summary/receipt']
    if mutation == 'suite_source': records['suite/manifest']['source_sha256']['hymem/test_0.py'] = 'b'*64
    if mutation == 'summary_source': records['summary/manifest']['source_sha256']['hymem/test_0.py'] = 'b'*64
    if mutation == 'suite_pin': gate['evidence']['suite']['manifest']['sha256'] = 'b'*64
    if mutation == 'summary_pin': gate['evidence']['summary']['manifest']['sha256'] = 'b'*64
    if mutation == 'old_schema': records['summary/manifest']['schema'] = 'r8-summary-repair-verification-package-v2'
    if mutation == 'cleanup': records['summary/supervisor']['outcome']['cleanup_complete'] = False
    if mutation == 'suite_failure': records['suite/receipt']['full']['failed'] = 1
    if mutation == 'count_gap': records['suite/receipt']['full']['passed'] = 7935
    if mutation == 'bool_count': records['suite/receipt']['full']['errors'] = False
    if mutation == 'repair_failure': r['source_exact_repair']['failure_reason'] = 'summary_output_cap'
    if mutation == 'normal_degraded': r['normal']['summary_failure_reason'] = 'summary_output_cap'
    if mutation == 'recovery_failure': r['explicit_recovery']['published'] = 0
    if mutation == 'accounting_unavailable': r['usage']['calls_available'] = False
    if mutation == 'paid_limit': r['usage']['request_attempts'] = 37
    if mutation == 'coverage_claim': gate['semantic_review']['strong_coverage_checklist_passed'] = True
    if mutation == 'retention_omission': del gate['semantic_review']['coverage_omission']
    if mutation == 'carried_extraction': gate['semantic_review']['r8_extraction_carried_forward'] = True
    if mutation == 'extra_evidence': gate['evidence']['retained_extraction'] = {}
    if mutation == 'control_failure': r['control_0']['failure_reason'] = 'shape_failure'
    if mutation == 'reference_changed': r['reference_unchanged'] = False
    monkeypatch.setattr(HOST, 'evidence_record', lambda binding: records[binding['path']])
    with pytest.raises((AssertionError,KeyError)):
        HOST.candidate_evidence(gate, manifest)


def test_recipe_and_helpers_remain_exact():
    old = module(HERE/'lme_r8_headless/bundle/q1_stock_run.py', 'old_r8_recipe')
    assert RUN.stock_arguments() == old.stock_arguments()
    for name in ('SOURCE_INDICES','QUESTION_IDS','QUESTION_CENSUS','DATASET_SHA',
                 'SELECTOR_SHA','CANARY_VERSION','SPLIT_VERSION','TIMEOUT','MODEL','ENDPOINT'):
        assert getattr(RUN,name) == getattr(old,name)
    for name in ('transport_common.py','supervised_invocation.py','lme_q1_startup_preflight.py'):
        assert (R9/'bundle'/name).read_bytes() == (HERE/'lme_r8_headless/bundle'/name).read_bytes()
    assert HOST.SEMANTIC_REVIEW == M.SEMANTIC_REVIEW and HOST.EVIDENCE_PINS == M.EVIDENCE_PINS


def test_runner_and_validator_only_package_admission_changed():
    old = (HERE/'lme_r8_headless/bundle/q1_stock_run.py').read_text()
    expected = old.replace('r8','r9').replace("manifest['source_files'] >= 481", "manifest['source_files'] == 508")
    expected = expected.replace("and len(manifest['source_sha256']) == manifest['source_files']",
        "and manifest['approved_source_manifest_sha256'] == '"+HOST.SOURCE_PIN+"'\n"
        "            and len(manifest['source_sha256']) == manifest['source_files']")
    assert (R9/'bundle/q1_stock_run.py').read_text().rstrip() == expected.rstrip()
    old_validation = (HERE/'lme_r8_headless/bundle/q1_stock_validate.py').read_text()
    assert (R9/'bundle/q1_stock_validate.py').read_text().rstrip() == old_validation.replace('r8','r9').rstrip()


def test_container_isolation():
    live = PLAN.command(str(HOST.RUN_ROOT), str(HOST.SOURCE_ROOT), 'a'*64, live=True)
    assert live['name'].startswith('hymem-r9-sample8-')
    assert live['mounts']['/candidate'] == (str(HOST.SOURCE_ROOT), False)
    pre = PLAN.command(str(HOST.RUN_ROOT), str(HOST.SOURCE_ROOT), 'a'*64)
    assert pre['network'] == 'none' and '/run/deepseek.env' not in pre['mounts']
    val = PLAN.validation_command(str(HOST.RUN_ROOT), str(HOST.SOURCE_ROOT), 'a'*64)
    assert val['network'] == 'none' and all(not value[1] for value in val['mounts'].values())


@pytest.mark.parametrize('live_code',[0,137])
def test_known_live_oom_still_runs_one_validation(monkeypatch, live_code):
    a = setup(monkeypatch, live_code=live_code)
    def sending(*args, **kwargs):
        state = a[5](*args, **kwargs)
        if args[2:4] == ('status','live'): state['oom_killed'] = True
        return state
    result = M.pipeline(*a[:4], waiting=a[4], sending=sending)
    assert result['status'] == 'offline_validation_finished'
    assert result['live_success'] is False and result['validation_runs_started'] == 1
    assert M.exit_status(result) == 1


def test_honest_caveat_config_rejects_bool_substitution():
    cfg = copy.deepcopy(CFG)
    cfg['semantic_review']['strong_coverage_checklist_passed'] = 0
    with pytest.raises(RuntimeError): M.checked_config(cfg)
