"""Single-use source-exact R9 diagnostic; private text stays on Afrodite."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

SCHEMA = 'r9-summary-output-cap-package-v1'
SOURCE, DIAG, RESULTS = Path('/candidate'), Path('/diag'), Path('/results')
REFERENCE, WORK = Path('/reference/hymem.sqlite'), Path('/work')
SUPPORT_SHA = 'bd8cc72c9bec26e391632af0f133d226e00f367807120b7a9a395e4dc55bf7a5'
SUPERVISOR_SHA = '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'

def sha(path):
    core.need(path.is_file() and not path.is_symlink(), 'regular_file')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path, name, pin):
    core.need(sha(path)==pin, 'support_pin')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def ref_hashes():
    return {p.name:sha(p) for p in (REFERENCE,Path(str(REFERENCE)+'-wal'),Path(str(REFERENCE)+'-shm')) if p.exists()}

def save(path,value):
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as handle: json.dump(value,handle,sort_keys=True,allow_nan=False)

def verify(pin):
    core.need(sha(DIAG/'manifest.json')==pin, 'manifest_pin')
    m=json.loads((DIAG/'manifest.json').read_text())
    core.need(m['schema']==SCHEMA and m['bounds']==core.BOUNDS and m['session_hash']==core.TARGET
              and m['variant'] in ('baseline','candidate'), 'manifest_scope')
    core.need(sha(DIAG/'worker.py')==m['worker_sha256'] and sha(DIAG/'core.py')==m['core_sha256'], 'worker_drift')
    actual={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*') if p.is_file()
            and not any(v in ('__pycache__','.pytest_cache') for v in p.parts) and p.suffix not in ('.pyc','.pyo')}
    core.need(actual==m['source_sha256'] and ref_hashes()==m['reference_sha256'], 'source_reference_drift')
    core.need(not any(p.is_symlink() for p in SOURCE.rglob('*')),'source_symlink')
    core.need(all(p.is_file() or p.is_dir() for p in SOURCE.rglob('*')),'source_special_file')
    for file,key in (('worker.py','support_sha256'),('r6_summary.py','r6_summary_sha256'),
                     ('r8_summary.py','r8_summary_sha256'),('supervised_invocation.py','supervisor_sha256')):
        core.need(sha(Path('/support')/file)==m[key],'support_inventory')
    return m

def environment():
    return dict(PATH='/home/node/hymem-env/bin:/usr/bin:/bin', HOME='/tmp',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',LANG='C.UTF-8')

def repair_parser(digest_module):
    # Candidate must expose its actual new normal-dispatch parser. Baseline
    # directly retains its shipped one-summary parser; no semantic override.
    return getattr(digest_module, '_validate_current_digest_summary_repair', digest_module._validate_digest_summary_repair)

def worker(mode,pin):
    os.umask(0o077)
    os.environ.clear(); os.environ.update(environment())
    logging.disable(logging.CRITICAL)
    result=dict(schema=SCHEMA,manifest_sha256=pin,mode=mode,bounds=core.BOUNDS,
                replay_kind='source-exact replay; not recorded-response replay',
                numeric_feedback_chars=503,numeric_builder_placeholder_is_model_response=False)
    client=budget=capture=support=None
    connections=[]
    threads={t.ident for t in threading.enumerate()}
    before_refs=ref_hashes()
    stage='preflight'
    try:
        core.need(os.geteuid()==1000,'isolated_user')
        stage='verify'
        m=verify(pin)
        result.update(capsule_root=m['capsule_root'],variant=m['variant'])
        stage='support_import'; support=load(Path('/support/worker.py'),'r9_support',SUPPORT_SHA)
        core.need({name:importlib.metadata.version(name) for name in support.RUNTIME}==support.RUNTIME,'runtime_drift')
        r6=load(Path('/support/r6_summary.py'),'r9_controls',m['r6_summary_sha256'])
        r8=load(Path('/support/r8_summary.py'),'r9_wire',m['r8_summary_sha256'])
        stage='source_import'; api=support.import_api(SOURCE)
        from hymem.dreaming import digest as producer
        from hymem.deadline import MonotonicDeadline,use_deadline
        deadline=MonotonicDeadline.after(900)
        if mode=='live':
            core.need(sys.stdin.buffer.read(256)==(pin+'\n').encode(),'execution_authority')
            proof=json.loads(Path('/preflight/offline.json').read_text())
            core.need(proof['status']=='offline_passed' and proof['manifest_sha256']==pin
                      and proof['completion_calls']==proof['http_attempts']==0,'offline_proof')
        normal_path=WORK/(mode+'-normal.sqlite'); recovery_path=WORK/(mode+'-recovery.sqlite')
        result['backup_progress']={}
        for stage,path in (('backup_normal',normal_path),('backup_recovery',recovery_path)):
            result['backup_progress'][stage]={}
            core.backup(REFERENCE,path,metrics=result['backup_progress'][stage])
        stage='connect_normal'; normal=api.db.connect(normal_path); connections.append(normal)
        stage='connect_recovery'; recovery=api.db.connect(recovery_path); connections.append(recovery)
        # Only clone initialization is permitted; reference never enters API.
        stage='initialize_normal'; api.db.initialize(normal)
        stage='initialize_recovery'; api.db.initialize(recovery)
        stage='select_target'; sid=core.select(normal); core.need(core.select(recovery)==sid,'clone_selection')
        stage='snapshot_clones'
        normal_before=support.snapshot(normal); recovery_before=support.snapshot(recovery)
        initial_states={target:api.classify_summary_state(recovery,target) for target in recovery_before['sessions']}
        r6.CONTROL=WORK/(mode+'-controls.sqlite')
        stage='seed_controls'; r6.seed_controls(api)
        controls=api.db.connect(r6.CONTROL); connections.append(controls)
        stage='snapshot_controls'; control_before=support.snapshot(controls)
        private=RESULTS/'private'; private.mkdir(mode=0o700)
        class Capture(r8.WireCapture):
            def select_recovery_session(self):
                pass
            def request(self,request):
                deadline.check()
                core.need(self.attempts<36 and capture.phase in core.PHASE_CAPS,'http_budget')
                core.need(__import__('time').monotonic()<budget.phase_end,'http_phase_deadline')
                body=request.read(); payload=json.loads(body)
                core.need(str(request.url)=='https://api.deepseek.com/chat/completions'
                    and payload['model']=='deepseek-flash' and payload['thinking']=={'type':'disabled'},'wire_route')
                self.attempts+=1
                request.extensions.update(r8_capture_index=self.attempts,r8_capture_phase=self.phase,r8_capture_session=self.session_sha256)
                self.save(self.attempts,'request',body)
        capture=Capture(private,deadline); capture.session_sha256=core.TARGET
        key='synthetic-not-a-key' if mode=='offline' else support.read_key(Path('/run/deepseek.env'))
        stage='client_create'; client=r8.capture_client(api,key,capture)
        core.need(api.retry_attempts==3 and client.transport_integrity_ok,'transport_contract')
        core.need(client.model=='deepseek-flash' and client.base_url=='https://api.deepseek.com'
                  and client.effective_extra_body=={'thinking':{'type':'disabled'}},'client_route')
        budget=core.Budget(client,deadline,m['worker_sha256'],capture)
        stage='producer_identity'; native=api.producer_binding(client,declaration_hook='memory_producer_declaration')
        diagnostic=api.producer_binding(budget,declaration_hook='memory_producer_declaration')
        core.need(native['identity_exact'] and diagnostic['identity_exact'], 'producer_identity')
        result.update(native_producer_identity_sha256=native['identity_sha256'],diagnostic_producer_identity_sha256=diagnostic['identity_sha256'])
        stage='capture_primary'; primary=core.capture_normal(normal,sid,client,producer.extract_session_digest)
        repair=core.source_exact_repair(primary,producer._build_digest_summary_repair_request)
        result.update(primary_request_sha256=core.digest(asdict(primary)),repair_request_sha256=core.digest(asdict(repair)))
        stage='capture_controls'; control_requests=[]
        for control_sid in sorted(r6.control_cases()):
            original=core.capture_normal(controls,control_sid,client,producer.extract_session_digest,max_chars=24000)
            for role,content in r6.control_cases()[control_sid]['messages']:
                core.need(content in original.user,'control_full_source_missing')
            control_requests.append((control_sid,core.source_exact_repair(original,producer._build_digest_summary_repair_request)))
        result['control_request_sha256']=[core.digest(asdict(r)) for _,r in control_requests]
        result['control_contract']=dict(kind='direct normal-format repair contract control',
            invented_feedback_chars=503,synthetic_primary_calls=0,max_chars=24000)
        if mode=='live':
            for field in ('native_producer_identity_sha256','diagnostic_producer_identity_sha256','primary_request_sha256','repair_request_sha256','control_request_sha256'):
                core.need(proof[field]==result[field],'preflight_context_drift')
            with use_deadline(deadline):
                stage='normal'; budget.start('normal')
                extracted=producer.extract_session_digest(normal,sid,budget,**core.NORMAL)
                result['normal']=dict(parse_failed=extracted.parse_failed,failure_reason=extracted.failure_reason,summary_failure_reason=getattr(extracted,'summary_failure_reason',None),summary_chars=len(extracted.summary or ''))
                stage='source_exact_repair'; budget.start('source_exact_repair')
                result['source_exact_repair']=core.parse_projection(budget.complete(repair),repair_parser(producer))
                stage='explicit_recovery'; budget.start('explicit_recovery')
                report=api.recovery.run_summary_recovery(recovery,budget,session_id=sid,max_calls=3,max_attempts=3,max_chars=8000,max_tokens=3072,timeout_seconds=120)
                result['explicit_recovery']=api.safe_report(report)
                for index,(control_sid,request) in enumerate(control_requests):
                    stage='control_'+str(index); budget.start(stage); capture.session_sha256=hashlib.sha256(control_sid.encode()).hexdigest()
                    result['control_'+str(index)]=core.parse_projection(budget.complete(request),repair_parser(producer),invented=True)
                    result['control_'+str(index)]['review_criteria']=[criterion for criterion in r6.control_cases()[control_sid]['review'] if 'window' not in criterion.lower()] + ['Preserve full-source update order; do not infer a deadline.']
        stage='audit'
        core.need(support.snapshot(normal)['full_sha256']==normal_before['full_sha256'],'normal_database_write')
        core.need(support.snapshot(controls)['full_sha256']==control_before['full_sha256'],'controls_database_write')
        support.assert_unchanged(recovery_before,support.snapshot(recovery))
        if mode=='live':
            result['recovery_changes']=support.audit_summary_changes(recovery,recovery_before,support.snapshot(recovery),initial_states,result['explicit_recovery'],api)
            state=api.classify_summary_state(recovery,sid)
            result['target_summary_healthy']=state['summary_healthy']
        for other_sid, row in recovery_before['sessions'].items():
            if other_sid != sid:
                core.need(dict(recovery.execute('SELECT * FROM sessions WHERE id=?',(other_sid,)).fetchone()) == row,
                          'unselected_session_changed')
        core.need(recovery.execute('SELECT 1 FROM run_lock').fetchone() is None,'recovery_lease_cleanup')
        if mode=='offline':
            core.need(support.snapshot(recovery)['full_sha256']==recovery_before['full_sha256'],'offline_recovery_changed')
            result['offline_clones_unchanged']=True
        result.update(all_non_summary_clone_unchanged=True,status='offline_passed' if mode=='offline' else 'review_pending')
    except BaseException as exc:
        result.update(status='error',exception_type='diagnostic_failed',failure_phase=stage if stage in core.FAILURE_STAGES else 'preflight',
                      failure_evidence=core.failure_evidence(exc))
    finally:
        result.update(completion_calls=budget.calls if budget else 0,http_attempts=capture.attempts if capture else 0,provider_calls=0)
        cleanup=[]
        if client:
            try:
                usage=api.usage_snapshot(client)
                result['usage']=usage; result['provider_calls']=usage['calls']
                core.need(usage['calls_available'] and usage['request_attempts_available']
                    and usage['successful_responses_available']
                    and 0<=usage['calls']==usage['successful_responses']<=result['completion_calls']<=12
                    and usage['request_attempts']==result['http_attempts']<=min(36,3*result['completion_calls'])
                    and capture.responses<=capture.attempts,'accounting')
            except BaseException: cleanup.append('accounting_failed')
        if capture:
            try: result['private_capture']=capture.public_inventory()
            except BaseException: cleanup.append('capture_failed')
        if client:
            try: client.close()
            except BaseException: cleanup.append('client_close_failed')
        for connection in connections:
            try: connection.close()
            except BaseException: cleanup.append('connection_close_failed')
        result.update(clients_closed='client_close_failed' not in cleanup,
            connections_closed='connection_close_failed' not in cleanup,
            threads_clean=not any(t.ident not in threads for t in threading.enumerate()),cleanup_errors=cleanup)
        if cleanup or not result['threads_clean']: result['status']='error'
        try:
            verify(pin); core.need(ref_hashes()==before_refs,'reference_changed')
            result.update(source_unchanged=True,reference_unchanged=True)
        except BaseException:
            result['status']='error'
        try: save(RESULTS/(mode+'.json'),result)
        except BaseException:
            result['status']='error'
    print(json.dumps({k:result[k] for k in ('status','completion_calls','http_attempts')}))
    return 0 if result['status'] in ('offline_passed','review_pending') else 1

def supervise(pin):
    verify(pin)
    supervisor=load(Path('/support/supervised_invocation.py'),'r9_supervisor',SUPERVISOR_SHA)
    def cancel(_signal,_frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT,cancel); signal.signal(signal.SIGTERM,cancel)
    outcome=None; failure=None; unchanged=False
    try:
        outcome=supervisor.supervise_invocation([sys.executable,'-I','-B','/diag/worker.py','live','--manifest-sha256',pin],cwd=SOURCE,env=environment(),output_dir=RESULTS/'invocation',timeout_seconds=900,cleanup_seconds=10,output_limit_bytes=1024*1024,stdin_bytes=(pin+'\n').encode())
    except BaseException: failure='supervision_failed'
    try: verify(pin); unchanged=True
    except BaseException: failure='postcheck_failed'
    save(RESULTS/'supervisor.json',dict(manifest_sha256=pin,outcome=asdict(outcome) if outcome else None,
         package_and_source_unchanged=unchanged,exception_type=failure))
    return 0 if failure is None and outcome and outcome.status=='completed' and outcome.returncode==0 and outcome.safe_to_continue else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('offline','live','supervise')); parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args(); raise SystemExit(supervise(args.manifest_sha256) if args.mode=='supervise' else worker(args.mode,args.manifest_sha256))
