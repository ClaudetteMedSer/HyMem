import copy
import hashlib
import json
from pathlib import Path
import types

import pytest

HERE = Path(__file__).resolve().parents[1]
M = types.ModuleType('continuation_test')
M.__file__ = str(HERE/'lme_r9_full500/continuation.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
# Private inert test pins/CIDs; never operational inputs.
CFG = {
    'schema': 'r9-headless-continuation-v1',
    'root': str(M.BASE/'lme-r9-full500-continuation-v1'),
    'target': str(M.BASE/'lme-r9-full500-headless-v1'),
    'target_pin': 'a'*64, 'controller_pin': 'b'*64,
    'source_pin': '35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51',
    'source_files': 508, 'expected_collected': 7940,
    'suite_cid': 'c'*64, 'preflight_cid': 'd'*64,
    'semantic_review': copy.deepcopy(M.SEMANTIC_REVIEW),
    'evidence': {
        name: {'root': str(M.BASE/('lme-r9-full-suite-v1' if name == 'suite' else 'lme-r9-sample8-headless-v1')),
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
    assert [c[2] for c in a[6] if c[0]=='wait']==[10860,2592060,3600]
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


R9 = HERE/'lme_r9_full500'
HOST = module(R9/'host_control.py', 'r9_host_tests')
RUN = module(R9/'bundle/q1_stock_run.py', 'r9_run_tests')
PLAN = module(R9/'bundle/q1_stock_host.py', 'r9_plan_tests')

PREFLIGHT = module(R9/'bundle/lme_q1_startup_preflight.py', 'full500_preflight_tests')
VALIDATOR = module(R9/'bundle/q1_stock_validate.py', 'full500_validator_tests')
PREPARE = module(R9/'prepare.py', 'full500_prepare_tests')

def test_full_selection_and_unchanged_recipe():
    old = module(HERE/'lme_r9_headless/bundle/q1_stock_run.py', 'sample8_recipe_tests')
    expected = old.stock_arguments()
    expected[expected.index('--sample')+1] = '0'
    assert RUN.stock_arguments() == expected
    assert RUN.TIMEOUT == 2592000 and RUN.SAMPLE == 0
    assert RUN.SOURCE_INDICES == list(range(500))
    candidate = Path('/private/tmp/hymem-r9-summary-20260927.XzIhDt/candidate/benchmarks/longmemeval_adapter.py')
    if candidate.is_file():
        assert RUN.verify_selection(RUN.selector_from_source(candidate)) == list(range(500))
    PREFLIGHT.request_recipe(RUN.stock_arguments())
    for name in ('transport_common.py','supervised_invocation.py'):
        assert (R9/'bundle'/name).read_bytes() == (HERE/'lme_r9_headless/bundle'/name).read_bytes()
    assert HOST.SEMANTIC_REVIEW == M.SEMANTIC_REVIEW
    assert HOST.EVIDENCE_PINS == M.EVIDENCE_PINS

def full_manifest():
    questions = [{'source_index':i,'question_id':f'q{i}','question_type':'multi-session',
                  'sessions':1,'messages':2} for i in range(500)]
    return {'schema':'stock-lme-r9-full500-candidate-regression-v1','sample':0,'seed':0,
            'question_count':500,'source_indices':list(range(500)),
            'question_ids':[row['question_id'] for row in questions],'questions':questions}

def test_exact_500_ids_and_no_accidental_sample8():
    manifest=full_manifest()
    assert RUN.question_ids_from_manifest(manifest) == VALIDATOR.expected_ids(manifest,RUN)
    PREFLIGHT.validate_expected_ids(manifest['question_ids'])
    for ids in (manifest['question_ids'][:8], manifest['question_ids'][:-1], ['q0']*500):
        bad=copy.deepcopy(manifest);bad['question_ids']=ids
        with pytest.raises(RuntimeError): RUN.question_ids_from_manifest(bad)
        with pytest.raises(RuntimeError): VALIDATOR.expected_ids(bad,RUN)
        with pytest.raises(RuntimeError): PREFLIGHT.validate_expected_ids(ids)
    bad=copy.deepcopy(manifest);bad['sample']=8
    with pytest.raises(RuntimeError):VALIDATOR.expected_ids(bad,RUN)
    bad=RUN.stock_arguments();bad[bad.index('--sample')+1]='8'
    with pytest.raises(RuntimeError):PREFLIGHT.request_recipe(bad)

def test_all500_census_and_scalar_types():
    manifest=full_manifest()
    census={'schema':'lme-full500-census-v1','dataset_sha256':RUN.DATASET_SHA,
            'source_question_count':500,'sample':0,'seed':0,
            'source_indices':list(range(500)),'questions':manifest['questions'],
            'provider_calls':0,'selection_predeclared':True,'raw_content_exported':False}
    util=types.SimpleNamespace(need=M.need)
    assert len(PREPARE.checked_census(census,RUN,util))==500
    for key,value in [('sample',8),('sample',False),('source_indices',list(range(8))),
                      ('questions',manifest['questions'][:8]),('provider_calls',1)]:
        bad=copy.deepcopy(census);bad[key]=value
        with pytest.raises(RuntimeError):PREPARE.checked_census(bad,RUN,util)

def test_container_isolation():
    pre=PLAN.command(str(HOST.RUN_ROOT),str(HOST.SOURCE_ROOT),'a'*64)
    live=PLAN.command(str(HOST.RUN_ROOT),str(HOST.SOURCE_ROOT),'a'*64,live=True)
    val=PLAN.validation_command(str(HOST.RUN_ROOT),str(HOST.SOURCE_ROOT),'a'*64)
    assert pre['network']=='none' and '/run/deepseek.env' not in pre['mounts']
    assert live['name'].startswith('hymem-r9-full500-')
    assert val['network']=='none' and all(not mount[1] for mount in val['mounts'].values())
    assert live['mounts']['/candidate']==(str(HOST.SOURCE_ROOT),False)

def test_legacy_protocol_and_accounting_claims_are_honest():
    text=(R9/'bundle/q1_stock_validate.py').read_text()
    assert "'official_comparable': False" in text
    assert "'global_paid_call_cap': None" in text
    assert "'canary_accounted_separately': True" in text
    assert "'new_provider_calls': 0" in text
    assert "'legacy-custom'" in text
    meter={field:0 for field in VALIDATOR.USAGE_FIELDS}
    assert VALIDATOR.usage_projection(meter)==meter
    meter['cost_usd']=float('nan')
    with pytest.raises(RuntimeError): VALIDATOR.usage_projection(meter)

def sample8_gate_fixture(monkeypatch):
    pins={'source.py':'a'*64,**{str(i):'b'*64 for i in range(507)}}
    report={'schema':'stock-lme-sample8-postvalidation-v1','status':'scored_sample8_completed',
            'manifest_sha256':HOST.EVIDENCE_PINS['sample8'],'sample':8,
            'strict_scored_artifact_validated':True,'benchmark_completed_without_faults':True,
            'physical_checkpoint_bound':True,'process_completed_cleanly':True,
            'reader_judge_calls_measured':True,'question_ids':[f'q{i}' for i in range(8)],
            'counts':{'completed':8,'expected':8,'failed':0,'missing':0},
            'summary_degraded_questions':0,'summary_degraded_sessions':0,
            'new_provider_calls':0,'official_comparable':False,
            'per_question':[{'question_id':f'q{i}','summary_healthy':True,'indexing_complete':True,
                'item_indexing_healthy':True,'strict_failure':False,'summary_degraded_sessions':0,
                'summary_missing_sessions':0,'malformed_summaries':0} for i in range(8)]}
    prior_code=b"import types\ndef definitions(): return ({},types.SimpleNamespace(validation_command=lambda *a:{}))\ndef inspect(*args): return dict(exit_code=0,pid=0,oom_killed=False)\n"
    manifest={'source_sha256':pins,'source_files':508,'approved_source_manifest_sha256':HOST.SOURCE_PIN,
              'prior_sample8_validation_sha256':'0a8325295daea3be967b996c8fe7db0f97a71e4fad948f19ed573167505b255e'}
    gate={'semantic_review':copy.deepcopy(HOST.SEMANTIC_REVIEW),'evidence':{}}
    records={}
    HOST.ROOT=HOST.RUN_ROOT
    old=HOST.BASE/'lme-r9-sample8-headless-v1'
    for name,pin in HOST.EVIDENCE_PINS.items():
        paths={'manifest':str(old/'bundle/manifest.json') if name=='sample8' else 'suite/manifest',
               'receipt':str(HOST.ROOT/'prior-sample8-validation.json') if name=='sample8' else 'suite/receipt',
               'supervisor':str(old/'live-results/supervisor-summary.json') if name=='sample8' else 'suite/supervisor'}
        gate['evidence'][name]={kind:{'path':path,'sha256':manifest['prior_sample8_validation_sha256'] if name=='sample8' and kind=='receipt' else pin} for kind,path in paths.items()}
        records[paths['manifest']]={'source_sha256':pins,'source_inventory_sha256':HOST.SOURCE_PIN}
        records[paths['receipt']]={'manifest_sha256':pin,'source_unchanged':True}
        records[paths['supervisor']]={'manifest_sha256':pin,'source_and_dataset_unchanged':True,
            'exception_type':None,'postrun_exception_type':None,
            'outcome':{'status':'completed','returncode':0,'safe_to_continue':True,'child_reaped':True,
                'group_absent_after_reap':True,'cleanup_complete':True,'errors':[],
                'terminal_receipt_written':True,'cleanup_warnings':[]}}
    records['suite/manifest'].update(schema='r8-offline-full-suite-v1',expected_collected=7940)
    records['suite/receipt'].update(status='passed',tested_source_unchanged=True,full_runs_started=1,
        provider_calls=0,collect={'collected':7940},full={'collected':7940,'passed':7936,'skipped':4,'failed':0,'errors':0,'exit_code':0})
    records[str(old/'bundle/manifest.json')].update(schema='stock-lme-r9-sample8-candidate-regression-v1',
        question_ids=report['question_ids'],host_controller_sha256=hashlib.sha256(prior_code).hexdigest(),
        remote_source=str(HOST.SOURCE_ROOT))
    records[str(HOST.ROOT/'prior-sample8-validation.json')]=report
    monkeypatch.setattr(HOST,'evidence_record',lambda binding:records[binding['path']])
    monkeypatch.setattr(HOST,'read',lambda path:prior_code)
    monkeypatch.setattr(HOST.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout=json.dumps(report).encode(),stderr=b''))
    return gate,manifest,records,report

def test_successful_same_source_sample8_admission(monkeypatch):
    gate,manifest,records,report=sample8_gate_fixture(monkeypatch)
    HOST.candidate_evidence(gate,manifest)

@pytest.mark.parametrize('mutation',['missing_summary','degraded','wrong_id','extra_row','source','receipt_pin','official','terminal','bool_count'])
def test_sample8_evidence_denials(monkeypatch,mutation):
    gate,manifest,records,report=sample8_gate_fixture(monkeypatch)
    if mutation=='missing_summary':report['per_question'][0]['summary_missing_sessions']=1
    if mutation=='degraded':report['per_question'][0]['summary_healthy']=False
    if mutation=='wrong_id':report['per_question'][0]['question_id']='unknown'
    if mutation=='extra_row':report['per_question'].append(report['per_question'][0])
    if mutation=='source':records['suite/manifest']['source_sha256']={'wrong':'x'}
    if mutation=='receipt_pin':gate['evidence']['sample8']['receipt']['sha256']='f'*64
    if mutation=='official':report['official_comparable']=True
    if mutation=='terminal':records['suite/supervisor']['outcome']['terminal_receipt_written']=False
    if mutation=='bool_count':report['counts']['missing']=False
    with pytest.raises((AssertionError,KeyError)):HOST.candidate_evidence(gate,manifest)
