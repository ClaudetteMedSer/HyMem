"""One-shot offline replay installer; no provider or production writes."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

CANDIDATE = Path('/private/tmp/hymem-episode-shadow-fix.RjDDmJ')
PINS = {
    'hymem/core/db.py': 'ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3',
    'hymem/dreaming/runner.py': '25387efe4ef6cf5ca6d6178f96a0c7872748cdb3758bbf68836c222027691f5a',
}
WORKER_SHA = 'faa28387ab543f24e5370aedf31ff3168ae59e897ec9594ccb1d1b121f4bfd66'

REMOTE = r'''
from pathlib import Path
import hashlib,importlib.util,json,os,re,shutil,stat,subprocess,sys
os.umask(0o077)
root=Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/cold-replay-dream-v1')
stage=root/'episode-shadow-replay-v1'
def need(ok,code):
    if not ok:raise RuntimeError(code)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value,mode=0o600):
    raw=json.dumps(value,sort_keys=True).encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
def inventory(path):
    out={}
    for entry in path.rglob('*'):
        need(not entry.is_symlink(),'candidate_symlink')
        if entry.is_file():out[entry.relative_to(path).as_posix()]=sha(entry)
    return out
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed_source():
    source=root/'work/live/hymem.sqlite'
    need(not source.is_symlink() and source.is_file(),'source_invalid')
    need(sha(source)=='f4f9cb8ec8247d27ab2981af58ec0c044fb76cc71356f2e6a757540d4eae59ae','source_changed')
    need(all(not os.path.lexists(str(source)+suffix) for suffix in ('-wal','-shm','-journal')),'source_sidecar_present')
    return source
cid=None;helper=None
try:
    data=json.load(sys.stdin)
    need(os.geteuid()==1000 and not stage.exists(),'stage_not_fresh')
    parent=inventory(root/'candidate')
    need(len(parent)==481 and digest(parent)=='ed889d8970c6d7827315de34342996ad55182773c238922fbbdf8510d73c43a4','parent_candidate_changed')
    sealed_source()
    hostpath=root/'claim_conflict_v64_dream_host.py'
    need(sha(hostpath)=='38df85589fbc561e287bc75329b3ab785af001dd0f6bc3024cbb09f629820f0c','parent_host_changed')
    spec=importlib.util.spec_from_file_location('sealed_dream_host',hostpath);host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
    shared,proof,helper=host.dependencies()
    livecid='8a726a4be865885910f20a5d396f659a1cc59f3dbbe8eaae25052e42c38bad5d'
    controller=host.controller();_,livemounts=controller.configure(helper,'live')
    live=controller.inspect(shared,helper,livecid,'live',livemounts)
    need(live['status']=='exited' and live['pid']==0 and live['exit_code']==0 and not live['oom_killed'],'source_writer_not_stopped')
    pins={'hymem/core/db.py':'ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3','hymem/dreaming/runner.py':'25387efe4ef6cf5ca6d6178f96a0c7872748cdb3758bbf68836c222027691f5a'}
    need(set(data['files'])==set(pins),'overlay_keys_invalid')
    need(hashlib.sha256(data['worker'].encode()).hexdigest()=='faa28387ab543f24e5370aedf31ff3168ae59e897ec9594ccb1d1b121f4bfd66','worker_pin_drift')
    for name,value in data['files'].items():need(hashlib.sha256(value.encode()).hexdigest()==pins[name],'overlay_pin_drift')
    stage.mkdir(mode=0o700);(stage/'work').mkdir(mode=0o700);(stage/'diag').mkdir(mode=0o700)
    shutil.copytree(root/'candidate',stage/'candidate')
    for name,value in data['files'].items():
        p=stage/'candidate'/name;p.chmod(0o600);p.write_text(value);p.chmod(0o400)
    expected=dict(parent);expected.update(pins)
    need(inventory(stage/'candidate')==expected,'candidate_overlay_invalid')
    worker=stage/'diag/claim_conflict_episode_shadow_replay.py';worker.write_text(data['worker']);worker.chmod(0o400)
    audit=root/'postflight-v2/claim_conflict_store_audit.py'
    need(sha(audit)=='e2efe365c5aedbbe88d86d521dd37b821759afc5fb21c21a6662e6a8ce567f42','audit_changed')
    shutil.copy2(audit,stage/'diag/claim_conflict_store_audit.py')
    image='sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
    need(helper.IMAGE==image,'image_changed')
    mounts=[(str(stage/'candidate'),'/candidate',False),(str(stage/'diag'),'/diag',False),(str(root/'work/live'),'/private-dream',False),(str(stage/'work'),'/work',True),(str(helper.RUNTIME),'/home/node/hymem-env',False)]
    command=['docker','create','--name','hymem-episode-shadow-replay-v1','--pull','never','--init','--network','none','--user','1000:1000','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','128','--memory','2g','--cpus','2','--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--env','HOME=/tmp','--env','TMPDIR=/tmp','--env','PYTHONDONTWRITEBYTECODE=1']
    for source,dest,rw in mounts:command+=['--mount','type=bind,src='+source+',dst='+dest+('' if rw else ',readonly')]
    command+=['--workdir','/candidate','--entrypoint','/home/node/hymem-env/bin/python3',image,'-I','-B','/diag/claim_conflict_episode_shadow_replay.py']
    shared.configure=lambda *_args:(command,mounts)
    save(stage/'intent.json',{'networked_runs_started':0,'runs_allowed':1,'candidate_sha256':digest(expected),'worker_sha256':sha(worker)})
    cid=helper.run(command,60,'shadow_replay_create').decode().strip()
    need(re.fullmatch('[0-9a-f]{64}',cid) is not None,'container_id_invalid')
    save(stage/'container.json',{'container_id':cid})
    need(shared.inspect(helper,cid,'offline',mounts,host.PHASE1_SHA)['status']=='created','not_created')
    need(helper.run(['docker','start',cid],60,'shadow_replay_start').decode().strip()==cid,'start_identity')
    exited=helper.run(['docker','wait',cid],240,'shadow_replay_wait').decode().strip()
    need(exited in ('0','1'),'exit_invalid')
    state=shared.inspect(helper,cid,'offline',mounts,host.PHASE1_SHA)
    need(state['status']=='exited' and state['pid']==0 and not state['oom_killed'] and state['exit_code']==int(exited),'terminal_invalid')
    result=json.loads(helper.run(['docker','logs',cid],30,'shadow_replay_logs'))
    allowed={'status','source_sha256','source_unchanged','before_actual','valid_vectors','removed_surplus','after_actual','exact_vectors_preserved','semantic_sha256','semantic_rows_unchanged','repeat_noop','reopen_aligned','health_clean','reason_code','failure_captured'}
    need(isinstance(result,dict) and set(result)<=allowed,'output_keys_invalid')
    need(all(type(v) in (bool,int) or isinstance(v,str) and (re.fullmatch('[0-9a-f]{64}',v) or v in ('verified','error','episode_shadow_replay_failed')) for v in result.values()),'output_values_invalid')
    need((result.get('status')=='verified')==(state['exit_code']==0),'exit_verdict_mismatch')
    sealed_source();need(inventory(root/'candidate')==parent and inventory(stage/'candidate')==expected,'code_changed')
    save(stage/'result.json',result)
    print(json.dumps({'stage':'episode-shadow-replay-v1','container_id':cid,'exit_code':state['exit_code'],'network':'none','configuration_verified':True,'candidate_sha256':digest(expected),'result_sha256':sha(stage/'result.json'),'report':result}))
except BaseException:
    if cid is not None and helper is not None and re.fullmatch('[0-9a-f]{64}',cid):
        try:
            helper.run(['docker','stop','--time','10',cid],40,'shadow_replay_cleanup')
            state=json.loads(helper.run(['docker','inspect',cid],30,'shadow_replay_cleanup_inspect'))[0]
            need(state['Id']==cid and state['State']['Pid']==0 and not state['State']['Running'],'cleanup_failed')
        except BaseException:
            print('{"status":"cleanup_failed"}');raise SystemExit(1)
    print('{"status":"error"}');raise SystemExit(1)
'''


if __name__ == '__main__':
    files = {}
    for name, pin in PINS.items():
        raw = (CANDIDATE / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin
        files[name] = raw.decode()
    raw = Path(__file__).with_name('claim_conflict_episode_shadow_replay.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == WORKER_SHA
    ssh = ['ssh', '-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
           '-o', 'ConnectionAttempts=1', 'afrodite']
    result = subprocess.run(ssh + [shlex.join(['python3', '-I', '-B', '-c', REMOTE])],
                            input=json.dumps({'files': files, 'worker': raw.decode()}),
                            text=True, capture_output=True)
    print(result.stdout, end='')
    # Remote failures never export arbitrary exception payloads or logs.
    if result.returncode:
        print('{"status":"remote_replay_failed"}')
    raise SystemExit(result.returncode)
