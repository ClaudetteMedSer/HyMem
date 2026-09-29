"""Offline recipe and candidate-evidence admission controls; no Docker calls."""
import hashlib
import json
from pathlib import Path
import types
import pytest

BASE=Path(__file__).resolve().parents[1]/'lme_r8_headless'
def load(path,name):
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module
runner=load(BASE/'bundle/q1_stock_run.py','r8_run_tests')
host=load(BASE/'host_control.py','r8_host_tests')
planner=load(BASE/'bundle/q1_stock_host.py','r8_plan_tests')

def test_recipe_and_support():
    args=runner.stock_arguments()
    assert runner.CANARY_VERSION=='hymem-phase1-extraction-canary-v20'
    assert runner.SPLIT_VERSION=='hymem-source-semantic-split-v11'
    assert runner.TIMEOUT==32400 and runner.SAMPLE==8 and runner.SEED==0
    old=load(BASE.parent/'lme_v64_headless_v1/bundle/q1_stock_run.py','old_recipe_tests')
    assert args==old.stock_arguments() and runner.QUESTION_IDS==old.QUESTION_IDS
    assert runner.SOURCE_INDICES==old.SOURCE_INDICES and runner.DATASET_SHA==old.DATASET_SHA
    for flag,value in [('--workers','1'),('--top-k','15'),('--indexing-max-cycles','100'),('--indexing-timeout-s','3600')]:
        assert args[args.index(flag)+1]==value
    for name in ('supervised_invocation.py','transport_common.py','lme_q1_startup_preflight.py'):
        assert (BASE/'bundle'/name).read_bytes()==(BASE.parent/'lme_v64_headless_v1/bundle'/name).read_bytes()

def test_plan_isolation():
    root=str(host.BASE/'lme-r8-test');source=str(host.BASE/'frozen-r8/candidate')
    plan=planner.command(root,source,'a'*64,live=True)
    assert plan['mounts']['/candidate']==(source,False) and plan['name'].startswith('hymem-r8-')
    pre=planner.command(root,source,'a'*64)
    assert pre['network']=='none' and '/run/deepseek.env' not in pre['mounts']
    validation=planner.validation_command(root,source,'a'*64)
    assert validation['network']=='none' and all(not v[1] for v in validation['mounts'].values())

def package(tmp_path):
    pins={f'tests/file{i}.py':'b'*64 for i in range(501)}
    manifest={'schema':'stock-lme-r8-sample8-candidate-regression-v1','source_revision':'r8',
        'source_sha256':pins,'source_files':501,'approved_source_manifest_sha256':hashlib.sha256(runner.canonical(pins)).hexdigest(),
        'candidate_only':True,'production_changes':False,'canary_version':runner.CANARY_VERSION,
        'source_split_policy_version':runner.SPLIT_VERSION,'rerolls_allowed':False,'resume_allowed':False,
        'dataset_sha256':runner.DATASET_SHA,'sample':8,'seed':0,'source_indices':runner.SOURCE_INDICES,
        'stock_arguments':runner.stock_arguments(),'supervision_seconds':32400,'global_paid_call_cap':None,
        'question_ids':runner.QUESTION_IDS,'helper_sha256':{}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));return manifest

def test_source_pin_derived_from_manifest(tmp_path,monkeypatch):
    manifest=package(tmp_path);monkeypatch.setattr(runner,'PACKAGE',tmp_path)
    assert runner.validate_package(runner.sha(tmp_path/'manifest.json'))['source_files']==501
    manifest['approved_source_manifest_sha256']='0'*64
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='manifest_schema'):runner.validate_package(runner.sha(tmp_path/'manifest.json'))

def evidence():
    pins={'hymem/a.py':'a'*64,'benchmarks/b.py':'b'*64,'tests/c.py':'c'*64}
    records={};gate={'summary_control_review_passed':True,'repair_control_review_passed':True,'retained_capture_inventory_sha256':'a'*64,'evidence':{}}
    for name,schema in [('suite','r8-offline-full-suite-v1'),('retained_extraction','r8-retained-chunks-v1'),('summary','r8-summary-repair-verification-package-v2')]:
        gate['evidence'][name]={kind:{'path':name+'/'+kind,'sha256':'f'*64} for kind in ('receipt','manifest','supervisor')}
        records[name+'/manifest']={'schema':schema,'source_sha256':dict(pins),'expected_collected':7799}
        records[name+'/supervisor']={'manifest_sha256':'f'*64,'outcome':{'returncode':0,'safe_to_continue':True,'child_reaped':True,'group_absent_after_reap':True,'cleanup_complete':True,'errors':[]}}
        records[name+'/receipt']={'manifest_sha256':'f'*64,'source_unchanged':True}
    records['suite/receipt'].update(status='passed',tested_source_unchanged=True,full_runs_started=1,collect={'collected':7799},full={'collected':7799,'exit_code':0,'failed':0,'errors':0,'passed':7795,'skipped':4})
    records['suite/receipt']['provider_calls']=0
    records['retained_extraction/receipt'].update(status='live_completed',mode='live',extraction_success=True,extraction_invocations=2,reference_unchanged=True,client_closed=True,threads_clean=True,chunks=[{'chunk_id':cid,'extraction':{'failed':False}} for cid in ('chk_aace7a800d186c4351ce09307040592724ff7239','chk_57346bc06395cd17a1f1bfd38b6701a3de5e5b8e')])
    records['summary/receipt'].update(status='effectiveness_passed_review_pending',mode='live',normal_retained={'complete_sessions':14},normal_controls={'complete_sessions':4},explicit_recovery={'recovered_all':True},clients_closed=True,connections_closed=True,reference_unchanged=True,threads_clean=True,cleanup_errors=[])
    records['summary/manifest'].update(previous_pins=dict(host.SUMMARY_PREVIOUS_PINS),repair_control_targets=4)
    records['summary/receipt'].update(schema='r8-summary-repair-verification-v2',
        previous_pins_verified=dict(host.SUMMARY_PREVIOUS_PINS),
        repair_contract_controls={'verification_kind':'direct_repair_prompt_contract_not_stock_invocation',
            'complete_sessions':4,'synthetic_primary_calls':0,'database_unchanged':True,
            'sessions':[{'complete':True,'windows':[{'failure_reason':None}]} for _ in range(4)]},
        retained_repair_case={'verification_kind':'direct_repair_prompt_on_exact_recorded_primary_input',
            'passed':True,'failure_reason':None,'completion_calls':1,'raw_content_exported':False,
            'database_writes':0,'previous_pins':dict(host.SUMMARY_PREVIOUS_PINS)})
    for name,calls,bounds in [('retained_extraction',2,{'completion_calls':96,'http_attempts':288,'timeout_seconds':1200}),('summary',3,{'completion_calls':192,'http_attempts':576,'timeout_seconds':2700})]:
        records[name+'/manifest']['bounds']=bounds
        records[name+'/receipt']['usage']={'calls':calls,'request_attempts':calls,'successful_responses':calls,
            'calls_available':True,'request_attempts_available':True,'successful_responses_available':True}
    records['summary/receipt'].update(completion_calls=3,provider_calls=3,http_attempts=3)
    for c in records['retained_extraction/receipt']['chunks']:
        c['extraction'].update(completion_calls=1,provider_attempts=1)
    gate['evidence']['retained_extraction']['audit']={'path':'audit','sha256':'f'*64}
    records['audit']={'status':'audit_passed','manifest_sha256':'f'*64,'capture_inventory_sha256':'a'*64,
        'exact_requests_verified':True,'private_results_verified':True,'capture_unchanged':True,
        'reference_unchanged':True,'source_unchanged':True,'container_removed':True,'provider_calls':0,
        'completion_calls':2,'recorded_http_attempts':2,'chunks':[dict(chunk_id=c['chunk_id'],**c['extraction'])
            for c in records['retained_extraction/receipt']['chunks']]}
    return gate,{'source_sha256':pins},records

def test_good_evidence_exact_final_source(monkeypatch):
    gate,manifest,records=evidence()
    monkeypatch.setattr(host,'evidence_record',lambda binding:records[binding['path']])
    host.candidate_evidence(gate,manifest)

@pytest.mark.parametrize('mutation',['suite_source','paid_source','summary_app','summary_test','cleanup','summary_review','summary_degraded','explicit_incomplete','failed_chunk','audit_source','audit_capture','audit_unclean','paid_cap','summary_accounting'])
def test_bad_evidence_fails_closed(monkeypatch,mutation):
    gate,manifest,records=evidence()
    if mutation=='suite_source':records['suite/manifest']['source_sha256']['tests/c.py']='d'*64
    if mutation=='paid_source':records['retained_extraction/manifest']['source_sha256']['hymem/a.py']='d'*64
    if mutation=='summary_app':records['summary/manifest']['source_sha256']['benchmarks/b.py']='d'*64
    if mutation=='summary_test':records['summary/manifest']['source_sha256']['tests/c.py']='d'*64
    if mutation=='cleanup':records['suite/supervisor']['outcome']['safe_to_continue']=False
    if mutation=='summary_review':gate['summary_control_review_passed']=False
    if mutation=='summary_degraded':records['summary/receipt']['status']='honestly_degraded'
    if mutation=='explicit_incomplete':records['summary/receipt']['explicit_recovery']['recovered_all']=False
    if mutation=='failed_chunk':records['retained_extraction/receipt']['extraction_success']=False
    if mutation=='audit_source':records['audit']['manifest_sha256']='e'*64
    if mutation=='audit_capture':records['audit']['capture_inventory_sha256']='e'*64
    if mutation=='audit_unclean':records['audit']['container_removed']=False
    if mutation=='paid_cap':records['retained_extraction/receipt']['usage']['request_attempts']=289
    if mutation=='summary_accounting':records['summary/receipt']['usage']['calls_available']=False
    monkeypatch.setattr(host,'evidence_record',lambda binding:records[binding['path']])
    with pytest.raises(AssertionError):host.candidate_evidence(gate,manifest)

def test_deployed_gate_not_admitted_as_candidate_gate(monkeypatch):
    raw=json.dumps({'schema':'lme-v64-root-reviewed-deployed-gate-v1'}).encode()
    monkeypatch.setattr(host,'ROOT',host.BASE/'r8-test')
    monkeypatch.setattr(host,'GATE_PIN',hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(host,'read',lambda _:raw)
    with pytest.raises(AssertionError):host.live_gates({},None)

@pytest.mark.parametrize('mutation',['old_package','old_receipt','absent_controls','absent_exact_case',
    'repair_review_false','repair_controls_incomplete','synthetic_primary','control_db_write',
    'exact_case_failed','extra_call','retained_export','retained_db_write','wrong_previous_pins'])
def test_new_repair_evidence_cannot_be_omitted_or_weakened(monkeypatch,mutation):
    gate,manifest,records=evidence()
    receipt=records['summary/receipt']
    if mutation=='old_package':records['summary/manifest']['schema']='r8-summary-replay-package-v1'
    if mutation=='old_receipt':receipt['schema']='r8-summary-replay-v1'
    if mutation=='absent_controls':del receipt['repair_contract_controls']
    if mutation=='absent_exact_case':del receipt['retained_repair_case']
    if mutation=='repair_review_false':gate['repair_control_review_passed']=False
    if mutation=='repair_controls_incomplete':receipt['repair_contract_controls']['complete_sessions']=3
    if mutation=='synthetic_primary':receipt['repair_contract_controls']['synthetic_primary_calls']=1
    if mutation=='control_db_write':receipt['repair_contract_controls']['database_unchanged']=False
    if mutation=='exact_case_failed':receipt['retained_repair_case']['passed']=False
    if mutation=='extra_call':receipt['retained_repair_case']['completion_calls']=2
    if mutation=='retained_export':receipt['retained_repair_case']['raw_content_exported']=True
    if mutation=='retained_db_write':receipt['retained_repair_case']['database_writes']=1
    if mutation=='wrong_previous_pins':receipt['retained_repair_case']['previous_pins']['returned_chars']=519
    monkeypatch.setattr(host,'evidence_record',lambda binding:records[binding['path']])
    with pytest.raises((AssertionError,KeyError)):host.candidate_evidence(gate,manifest)

@pytest.mark.parametrize('mutation',['skip_overflow','missing_passed','missing_skipped','count_gap','bool_passed'])
def test_suite_count_accounting_cannot_hide_tests(monkeypatch,mutation):
    gate,manifest,records=evidence();full=records['suite/receipt']['full']
    if mutation=='skip_overflow':full.update(passed=7794,skipped=5)
    if mutation=='missing_passed':del full['passed']
    if mutation=='missing_skipped':del full['skipped']
    if mutation=='count_gap':full['passed']=7794
    if mutation=='bool_passed':full['passed']=True
    monkeypatch.setattr(host,'evidence_record',lambda binding:records[binding['path']])
    with pytest.raises((AssertionError,KeyError)):host.candidate_evidence(gate,manifest)
