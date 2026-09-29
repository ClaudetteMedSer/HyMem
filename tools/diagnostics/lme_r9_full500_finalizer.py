"""One-shot offline-validation rescue for the already running sealed full500 run."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

if sys.flags.optimize:
    raise RuntimeError('optimized_execution_forbidden')
BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky')
ROOT = BASE/'lme-r9-full500-finalizer-v1'
RUN = BASE/'lme-r9-full500-headless-v1'
CONT = BASE/'lme-r9-full500-continuation-v1'
MANIFEST_PIN = '494bd08f76ef86d71b10c23ecd799a6189e3d92b561ea3c5137f8295bcc73c89'
CONTROLLER_PIN = '08a885505b24532f2cc74f06f73c16424f885a51064c56cc20941fc4f4a53008'
CONTINUATION_PIN = 'e96f6b5a1d2ffe4146761bc802637542bb9543b072010fcb22a0c4b5060a87fc'
LIVE_CID = '523469c234e48ff01d944de22a149251d53f0477aa16133b438bfee2f7030588'
CONFIG_PIN = '038a5428da81dca78007e1573e39421d79c3af1b9e6454e9f0dfb77af6edb1a7'

def need(value, label):
    if not value:
        raise RuntimeError(label)

def read(path):
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink()
         and path.is_file() and path.stat().st_size<=32*1024*1024,'input_scope')
    return path.read_bytes()

def sha(raw): return hashlib.sha256(raw).hexdigest()
def js(path): return json.loads(read(path))

def save(name, value):
    path=ROOT/name
    need(path.parent==ROOT,'receipt_scope')
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:
        out.write(raw);out.flush();os.fsync(out.fileno())

def load(path,pin,name):
    raw=read(path);need(sha(raw)==pin,'code_pin')
    module=types.ModuleType(name);module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module

def wait_existing(cid,deadline,*,popen=subprocess.Popen,now=time.time):
    """Keep one read-only Docker wait client across bounded Linux-safe polls."""
    need(type(cid) is str and re.fullmatch('[0-9a-f]{64}',cid),'wait_cid')
    need(now()<deadline,'wait_expired')
    client=popen(['docker','wait',cid],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        while True:
            remaining=deadline-now()
            need(remaining>0,'wait_expired')
            try:
                stdout,stderr=client.communicate(timeout=min(30,remaining))
            except subprocess.TimeoutExpired:
                continue
            need(client.returncode==0 and not stderr
                 and re.fullmatch(rb'[0-9]{1,3}\n?',stdout),'wait_failed')
            code=int(stdout);need(0<=code<=255,'exit_range')
            return code
    finally:
        # This only reaps our local read-only Docker wait client, never a container.
        if client.poll() is None:
            client.terminate()
            try: client.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                client.kill();client.communicate(timeout=10)

def terminal(state,code):
    need(state['status']=='exited' and state['pid']==0 and state['exit_code']==code
         and type(state['oom_killed']) is bool,'terminal_state')

def started_deadline(cid):
    reply=subprocess.run(['docker','inspect','--format','{{.State.StartedAt}}',cid],
                         capture_output=True,timeout=30)
    need(reply.returncode==0 and not reply.stderr and len(reply.stdout)<128,'started_at')
    text=reply.stdout.decode().strip()
    need(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z',text),'started_at_format')
    # datetime accepts Docker's nanoseconds by truncating to microseconds.
    start=datetime.fromisoformat(text.replace('Z','+00:00')).timestamp()
    need(start>0 and start<=time.time()+60,'started_at_range')
    return start+2592060

def admit(controller,continuation):
    need(sha(read(RUN/'bundle/manifest.json'))==MANIFEST_PIN,'manifest_pin')
    controller.ROOT,controller.PIN=RUN,MANIFEST_PIN
    controller.SOURCE=Path(js(RUN/'bundle/manifest.json')['remote_source'])
    manifest,host=controller.definitions()
    need(manifest['host_controller_sha256']==CONTROLLER_PIN
         and manifest['continuation_sha256']==CONTINUATION_PIN,'sealed_pins')
    prior=js(CONT/'result.json')
    need(prior['status']=='operator_pending' and type(prior['paid_runs_started']) is int
         and prior['paid_runs_started']==1 and type(prior['validation_runs_started']) is int
         and prior['validation_runs_started']==0 and prior['error_type']=='OverflowError'
         and prior['automatic_retry'] is False and prior['operator_inspection_required'] is True,
         'prior_failure')
    intent=js(CONT/'intent.json')
    need(intent['live_runs_allowed']==intent['validation_runs_allowed']==1,'prior_allowance')
    cfg=intent['config'];continuation.checked_config(cfg)
    need(cfg['target_pin']==MANIFEST_PIN and cfg['controller_pin']==CONTROLLER_PIN,'prior_config')
    config_raw=read(CONT/'config.json')
    need(sha(config_raw)==CONFIG_PIN and json.loads(config_raw)==cfg,'config_pin')
    launch=js(CONT/'launch-intent.json');owned=js(CONT/'launch-owned.json')
    need(launch['schema']=='r9-detached-continuation-launch-v1'
         and launch['config_path']==str(CONT/'config.json') and launch['config_sha256']==CONFIG_PIN
         and launch['continuation_sha256']==CONTINUATION_PIN and launch['automatic_retry'] is False
         and type(launch['launches_allowed']) is int and launch['launches_allowed']==1,'prior_launch')
    argv=launch['command']
    need(type(argv) is list and len(argv)==10 and type(argv[0]) is str
         and Path(argv[0]).is_absolute()
         and argv[1:]==['-I','-B',str(RUN/'continuation.py'),'--config',str(CONT/'config.json'),
                        '--config-sha256',CONFIG_PIN,'--self-sha256',CONTINUATION_PIN],'prior_launch_command')
    need(all(owned[key]==value for key,value in launch.items())
         and owned['status']=='detached_process_owned'
         and owned['pid']==owned['process_group_id']==owned['session_id']==2011243
         and owned['start_ticks']==198499097,'prior_launch_owner')
    need(not (Path('/proc')/str(owned['pid'])).exists(),'prior_process_present')
    for path in (CONT/'live-owned.json',CONT/'live-start-requested.json',RUN/'live-container-id.json'):
        need(js(path)=={'container_id':LIVE_CID},'live_owner')
    need(js(RUN/'live-start-intent.json')=={'container_id':LIVE_CID,'manifest_sha256':MANIFEST_PIN},'live_start_intent')
    plan=host.command(str(RUN),str(controller.SOURCE),MANIFEST_PIN,live=True)
    need(js(RUN/'live-create-intent.json')=={'manifest_sha256':MANIFEST_PIN,'plan':json.loads(json.dumps(plan))},'live_create_intent')
    for parent,names in ((RUN,('validation-create-intent.json','validation-container-id.json',
                              'validation-created.json','validation-start-intent.json')),
                         (CONT,('validation-owned.json','validation-start-requested.json'))):
        need(all(not (parent/name).exists() and not (parent/name).is_symlink() for name in names),'prior_validation_present')
    state=controller.inspect(LIVE_CID,plan)
    need(state['status'] in ('running','exited'),'live_state')
    return cfg,plan,started_deadline(LIVE_CID)

def validation_dispatch(continuation,cfg,action,cid=None):
    need(action in ('create','start','status'),'validation_action')
    return continuation.dispatch(RUN/'host_control.py',cfg,action,'validation',cid)

def pipeline(controller,continuation,*,record=save,admitting=admit,waiting=wait_existing,
             sending=validation_dispatch):
    # Exclusive ownership precedes any subprocess or evidence mutation.
    record('intent.json',{'schema':'r9-full500-validation-rescue-v1','live_container_id':LIVE_CID,
        'manifest_sha256':MANIFEST_PIN,'paid_runs_allowed':0,'validation_runs_allowed':1})
    outcome={'status':'operator_pending','paid_runs_started':0,'validation_runs_started':0,
             'existing_live_container_id':LIVE_CID,'automatic_retry':False}
    try:
        cfg,plan,deadline=admitting(controller,continuation)
        record('admitted.json',{'live_container_id':LIVE_CID,'absolute_deadline_epoch':deadline})
        code=waiting(LIVE_CID,deadline)
        state=controller.inspect(LIVE_CID,plan)
        terminal(state,code)
        outcome.update(live_exit_code=code,live_success=code==0 and state['oom_killed'] is False)
        record('live-terminal.json',state)
        created=sending(continuation,cfg,'create')
        cid=created['container_id'];need(created['status']=='created','validation_create')
        record('validation-owned.json',{'container_id':cid})
        record('validation-start-requested.json',{'container_id':cid})
        outcome.update(validation_runs_started=None,validation_start_attempted=True)
        started=sending(continuation,cfg,'start',cid)
        need(started['status'] in ('running','exited'),'validation_start')
        outcome['validation_runs_started']=1
        vcode=waiting(cid,time.time()+3600)
        vstate=sending(continuation,cfg,'status',cid)
        terminal(vstate,vcode)
        need(vstate['oom_killed'] is False,'validation_oom')
        outcome.update(status='offline_validation_finished',validation_exit_code=vcode,
                       validation_success=vcode==0)
    except BaseException as exc:
        outcome.update(error_type=type(exc).__name__,operator_inspection_required=True)
    record('result.json',outcome)
    return outcome

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-sha256',required=True)
    args=parser.parse_args()
    need(os.geteuid()==1000,'host_uid')
    need(sha(read(Path(__file__).resolve()))==args.self_sha256,'self_pin')
    need(ROOT.resolve()==ROOT and ROOT.is_dir() and ROOT.stat().st_uid==1000
         and ROOT.stat().st_mode & 0o077==0,'private_root')
    controller=load(RUN/'host_control.py',CONTROLLER_PIN,'rescue_controller')
    continuation=load(RUN/'continuation.py',CONTINUATION_PIN,'rescue_continuation')
    result=pipeline(controller,continuation)
    return 0 if (result['status']=='offline_validation_finished' and result['live_success'] is True
                 and result['validation_success'] is True) else 1

if __name__=='__main__':sys.exit(main())
