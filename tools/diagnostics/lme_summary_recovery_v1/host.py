"""Scoped recovery packaging/installation/container control. Never production."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tarfile

BASE='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ'
ROOT=BASE+'/summary-recovery-v2'
IMAGE='sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME='/opt/stacks/hermes/instance1/home/hymem-env'
SOURCE=BASE+'/offline-r5/candidate'
REFERENCE=BASE+'/q1-stock-v4/live-results/stores/hymem-lme-ktsn8xyj'
KEY='/opt/stacks/hermes/instance1/home/.hermes/.env'
R5_PIN='c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'
SUPERVISOR_PIN='9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'
PIN='6113dd9f513521f789392df934cb6735f0cd5b05f0432b4bad9d641e6b73d6e9'

def encoded(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()

def plan(mode,pin):
    assert mode in ('preflight','live') and re.fullmatch('[0-9a-f]{64}',pin)
    mounts={'/home/node/hymem-env':(RUNTIME,False),'/candidate':(SOURCE,False),
        '/diag':(ROOT+'/bundle',False),'/r5':(BASE+'/offline-r5/diag',False),
        '/reference':(REFERENCE,False),'/work':(ROOT+'/work',True),
        '/results':(ROOT+'/'+mode+'-results',True),'/home/node/.hermes':(ROOT+'/home',False)}
    if mode=='live': mounts['/run/deepseek.env']=(KEY,False)
    name='hymem-summary-recovery-'+pin[:12]+'-'+mode
    command=['docker','create','--name',name,'--pull','never','--init','--network',
        'bridge' if mode=='live' else 'none','--user','1000:1000','--read-only',
        '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','128',
        '--memory','2g','--cpus','2','--tmpfs','/tmp:rw,noexec,nosuid,size=64m']
    for target,(origin,writable) in mounts.items():
        command+=['--mount','type=bind,src='+origin+',dst='+target+('' if writable else ',readonly')]
    command+=['--workdir','/candidate','--entrypoint','/home/node/hymem-env/bin/python3',
        IMAGE,'-I','-B','/diag/protocol.py',mode,'--manifest-sha256',pin]
    return dict(command=command,mounts=mounts,name=name,image=IMAGE,
                network='bridge' if mode=='live' else 'none')

def inspect(cid,expected,state=None):
    assert re.fullmatch('[0-9a-f]{64}',cid)
    obj=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
    cfg,hc,current=obj['Config'],obj['HostConfig'],obj['State']
    assert obj['Image']==cfg['Image']==IMAGE and obj['Name']=='/'+expected['name']
    assert cfg['User']=='1000:1000' and cfg['WorkingDir']=='/candidate'
    assert cfg['Entrypoint']==['/home/node/hymem-env/bin/python3']
    assert obj['Path']=='/home/node/hymem-env/bin/python3'
    assert cfg['Cmd']==obj['Args']==expected['command'][expected['command'].index(IMAGE)+1:]
    assert {m['Destination']:(m['Source'],m['RW'],m['Type']) for m in obj['Mounts']}=={
        target:(origin,writable,'bind') for target,(origin,writable) in expected['mounts'].items()}
    assert hc['NetworkMode']==expected['network'] and hc['ReadonlyRootfs'] is True
    assert hc['Privileged'] is False and hc['CapDrop']==['ALL']
    assert hc['SecurityOpt']==['no-new-privileges'] and hc['Init'] is True
    assert hc['PidsLimit']==128 and hc['Memory']==2147483648 and hc['NanoCpus']==2000000000
    assert hc['Tmpfs']=={'/tmp':'rw,noexec,nosuid,size=64m'} and hc['RestartPolicy']['Name']=='no'
    assert not any(v.startswith(('DEEPSEEK_','OPENAI_','HYMEM_')) for v in cfg['Env'])
    if state is not None: assert current['Status']==state
    return dict(container_id=cid,status=current['Status'],exit_code=current['ExitCode'],
        oom_killed=current['OOMKilled'],pid=current['Pid'],configuration_verified=True,
        network=expected['network'],credential_mount='/run/deepseek.env' in expected['mounts'],
        manifest_sha256=PIN)

def read(path):
    assert path.resolve()==path and stat.S_ISREG(path.lstat().st_mode)
    return path.read_bytes()

def check_package():
    root=Path(ROOT); raw=read(root/'bundle/manifest.json'); assert sha(raw)==PIN
    manifest=json.loads(raw); pins=manifest['helper_sha256']
    assert set(pins)=={'worker.py','protocol.py','supervised_invocation.py'}
    assert {p.name for p in (root/'bundle').iterdir()}==set(pins)|{'manifest.json'}
    assert all(sha(read(root/'bundle'/name))==pin for name,pin in pins.items())
    assert pins['supervised_invocation.py']==SUPERVISOR_PIN
    assert manifest['schema']=='isolated-summary-recovery-package-v1'
    assert manifest['source_manifest_sha256']==R5_PIN and manifest['root']==ROOT
    assert sha(read(Path(BASE)/'offline-r5/diag/manifest.json'))==R5_PIN
    return manifest

def save(path,value):
    with path.open('x') as out: json.dump(value,out,sort_keys=True)

def checked_preflight(receipt,source_map):
    assert receipt['schema']=='summary-recovery-live-diagnostic-v1'
    assert receipt['status']=='preflight_passed' and receipt['r5_manifest_sha256']==R5_PIN
    assert receipt['model']=='deepseek-flash' and receipt['endpoint']=='https://api.deepseek.com'
    assert json.dumps(receipt['bounds'],sort_keys=True)==json.dumps(dict(
        max_calls=100,max_attempts=3,max_chars=8000,max_tokens=3072,timeout_seconds=1800),sort_keys=True)
    assert type(receipt['max_http_attempts']) is int and receipt['max_http_attempts']==300
    for field in ('stock_invocations','provider_calls'):
        assert type(receipt[field]) is int and receipt[field]==0
    for field in ('client_closed','reference_store_unchanged_verified',
                  'source_unchanged_verified','all_non_summary_state_unchanged'):
        assert receipt[field] is True
    for field in ('original_store_modified','benchmark_rerun','full_500_readiness_verified',
                  'semantic_quality_guaranteed'):
        assert receipt[field] is False
    assert re.fullmatch('[0-9a-f]{64}',receipt['clone_before_sha256'])
    assert receipt['clone_before_sha256']==receipt['clone_after_sha256']
    expected=sha(json.dumps(source_map,sort_keys=True,separators=(',',':'),
                           ensure_ascii=True,allow_nan=False).encode())
    assert receipt['source_inventory_sha256']==expected
    assert re.fullmatch('sha256:[0-9a-f]{64}',receipt['producer_identity_sha256'])
    health=receipt['health_before']
    assert set(health)=={'summary_degraded_sessions','summary_missing_sessions','malformed_summaries','summary_healthy'}
    assert type(health['summary_degraded_sessions']) is int and health['summary_degraded_sessions']==10
    assert type(health['malformed_summaries']) is int and health['malformed_summaries']==0
    assert type(health['summary_missing_sessions']) is int and 0<=health['summary_missing_sessions']<=10
    assert health['summary_healthy'] is False
    for field in ('calls','request_attempts','successful_responses'):
        assert type(receipt['usage'][field]) is int and receipt['usage'][field]==0
        assert receipt['usage'][field+'_available'] is True

def live_gates():
    root=Path(ROOT)
    cid=json.loads(read(root/'preflight-container-id.json'))['container_id']
    state=inspect(cid,plan('preflight',PIN),'exited')
    assert state['exit_code']==0 and state['oom_killed'] is False and state['pid']==0
    receipt=json.loads(read(root/'preflight-results/preflight.json'))
    source_map=json.loads(read(Path(BASE)/'offline-r5/diag/manifest.json'))['source_sha256']
    checked_preflight(receipt,source_map)
    info=Path(KEY).lstat()
    assert Path(KEY).resolve()==Path(KEY) and stat.S_ISREG(info.st_mode)
    assert info.st_uid==1000 and stat.S_IMODE(info.st_mode)==0o600

def remote(action,mode,cid):
    assert action in ('create','start','status') and mode in ('preflight','live')
    check_package(); root=Path(ROOT); expected=plan(mode,PIN)
    if action=='create':
        save(root/(mode+'-create-intent.json'),expected)
        cid=subprocess.check_output(expected['command'],text=True).strip()
        save(root/(mode+'-container-id.json'),{'container_id':cid})
        result=inspect(cid,expected,'created')
    else:
        assert json.loads(read(root/(mode+'-container-id.json')))['container_id']==cid
        if action=='start':
            inspect(cid,expected,'created')
            if mode=='live': live_gates()
            save(root/(mode+'-start-intent.json'),{'container_id':cid,'manifest_sha256':PIN})
            assert subprocess.check_output(['docker','start',cid],text=True).strip()==cid
        result=inspect(cid,expected)
    print(json.dumps(result,sort_keys=True))

INSTALL=r'''
import hashlib,io,json,os,pathlib,sys,tarfile
raw=sys.stdin.buffer.read(2097153); assert len(raw)<=2097152
assert hashlib.sha256(raw).hexdigest()==C['archive_pin']
files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:
    for member in tar.getmembers():
        p=pathlib.PurePosixPath(member.name)
        assert member.isfile() and not member.pax_headers and member.name not in files
        assert str(p)==member.name and len(p.parts)==2 and p.parts[0]=='bundle'
        assert not p.is_absolute() and '..' not in p.parts and member.size<=1048576
        files[member.name]=tar.extractfile(member).read()
assert set(files)==set(C['pins'])
assert all(hashlib.sha256(body).hexdigest()==C['pins'][name] for name,body in files.items())
assert hashlib.sha256(files['bundle/manifest.json']).hexdigest()==C['manifest_pin']
root=pathlib.Path(C['root']); assert root.parent.resolve()==root.parent and root.parent.is_dir()
assert os.geteuid()==1000
root.mkdir(mode=0o700)
for name in ('bundle','home','work','preflight-results','live-results'): (root/name).mkdir(mode=0o700)
for name,body in files.items():
    fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as out: out.write(body)
result={'status':'installed_not_started','manifest_sha256':C['manifest_pin'],
        'archive_sha256':C['archive_pin'],'bundle_files':len(files),'provider_calls':0}
with (root/'install.json').open('x') as out: json.dump(result,out,sort_keys=True)
print(json.dumps(result,sort_keys=True))
'''

def seal(output):
    base=Path(__file__).resolve().parent
    names=('worker.py','protocol.py','supervised_invocation.py')
    files={'bundle/'+n:read(base/n) for n in names}
    assert sha(files['bundle/supervised_invocation.py'])==SUPERVISOR_PIN
    manifest={'schema':'isolated-summary-recovery-package-v1','source_manifest_sha256':R5_PIN,
        'helper_sha256':{n:sha(files['bundle/'+n]) for n in names},'root':ROOT,
        'model':'deepseek-flash','endpoint':'https://api.deepseek.com',
        'logical_completion_limit':100,'http_attempt_limit':300,'stock_timeout_seconds':1800,
        'supervision_seconds':1860,'production_changes':False,'rerolls_allowed':False}
    files['bundle/manifest.json']=encoded(manifest)
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for name,body in sorted(files.items()):
            member=tarfile.TarInfo(name); member.size=len(body); member.mode=0o400
            member.uid=member.gid=1000; member.mtime=0; tar.addfile(member,io.BytesIO(body))
    output.mkdir(mode=0o700)
    for name,body in files.items():
        path=output/name; path.parent.mkdir(mode=0o700,exist_ok=True)
        with path.open('xb') as out: out.write(body)
    with (output/'upload.tar').open('xb') as out: out.write(data.getvalue())
    result={'root':ROOT,'manifest_pin':sha(files['bundle/manifest.json']),
        'archive_pin':sha(data.getvalue()),'pins':{n:sha(b) for n,b in files.items()}}
    save(output/'seal.json',result); print(json.dumps(result,sort_keys=True))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('seal','install','create','start','status'))
    parser.add_argument('--mode',choices=('preflight','live'))
    parser.add_argument('--container-id')
    parser.add_argument('--package',type=Path)
    parser.add_argument('--seal-pin')
    args=parser.parse_args()
    if args.action=='seal': return seal(args.package)
    if args.action=='install':
        raw=read(args.package/'seal.json'); assert sha(raw)==args.seal_pin
        config=json.loads(raw); assert config['root']==ROOT and config['manifest_pin']==PIN
        archive=read(args.package/'upload.tar'); assert sha(archive)==config['archive_pin']
        code='import json\nC=json.loads('+repr(json.dumps(config))+')\n'+INSTALL
        result=subprocess.run(['ssh','afrodite','python3 -I -B -c '+shlex.quote(code)],
            input=archive,capture_output=True,timeout=120)
        if result.returncode: raise SystemExit('isolated_install_failed')
        print(json.dumps(json.loads(result.stdout),sort_keys=True)); return
    remote(args.action,args.mode,args.container_id)

if __name__=='__main__': main()
