"""Fixed-scope R5 offline target gate. No production paths or provider calls."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[3]
TREE = Path('/private/tmp/hymem-parent-frozen-r5-20260924.dkx3z00n/verification-tree-r5')
MANIFEST = REPO / 'docs/patches/2026-09-24-lme-independent-summary-indexing-r5-manifest.json'
GATE = REPO / 'tools/diagnostics/lme_offline_gate.py'
MANIFEST_PIN = 'c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'
GATE_PIN = 'd53ac383414f077a2af29b506ed883acbcd64cb933da149df3ca5da8cfe6e01a'
REMOTE_ROOT = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r5'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
FOCUS = ['tests/' + name + '.py' for name in (
    'test_completion_response_admission', 'test_typed_response_recovery', 'test_canary_response_history',
    'test_openai_client', 'test_provider_attempt_accounting', 'test_indexing_deadlines',
    'test_benchmark_client_cleanup', 'test_benchmark_extraction_canary',
    'test_digest_summary_index_separation', 'test_digest_summary_separate_publication', 'test_digest_publication',
    'test_summary_frontier_v62', 'test_summary_recovery_v63', 'test_summary_recovery_integration_root',
    'test_summary_recovery_command_root', 'test_lme_protocol_hardening', 'test_lme_historical_admission',
    'test_lme_registry_cli_bootstrap', 'test_lme_current_model_startup', 'test_startup_schema_atomicity',
    'test_summary_portability_v18', 'test_benchmark_summary_degradation')]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def archive():
    raw_manifest, raw_gate = MANIFEST.read_bytes(), GATE.read_bytes()
    assert sha(raw_manifest) == MANIFEST_PIN and sha(raw_gate) == GATE_PIN
    manifest = json.loads(raw_manifest)
    expected = {}
    for key in ('source_sha256', 'test_sha256', 'auxiliary_sha256'):
        assert not set(expected) & set(manifest[key])
        expected.update(manifest[key])
    assert len(expected) == 467 and len(manifest['expected_nodeids']) == 7247
    assert all(name in manifest['test_sha256'] for name in FOCUS)
    nodes = [node for node in manifest['expected_nodeids'] if node.split('::', 1)[0] in FOCUS]
    assert len(nodes) == 1044 and set(manifest['expected_skip_nodeids']) <= set(nodes)
    actual = set()
    for path in TREE.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            actual.add(path.relative_to(TREE).as_posix())
    assert actual == set(expected)
    files = {'diag/manifest.json': raw_manifest, 'diag/lme_offline_gate.py': raw_gate}
    for name, digest in expected.items():
        raw = (TREE / name).read_bytes()
        assert sha(raw) == digest
        files['verification/' + name] = raw
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w', format=tarfile.USTAR_FORMAT) as writer:
        for name, raw in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.uid, member.gid, member.mtime = len(raw), 0o400, 1000, 1000, 0
            writer.addfile(member, io.BytesIO(raw))
    return output.getvalue()


INSTALL = r'''
import hashlib,io,json,os,pathlib,sys,tarfile
root=pathlib.Path(C['root'])
raw=sys.stdin.buffer.read(64*1024*1024+1)
assert len(raw)<=64*1024*1024 and hashlib.sha256(raw).hexdigest()==C['archive_sha256']
files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
    for item in archive.getmembers():
        assert item.isfile() and not item.pax_headers and item.name not in files
        assert pathlib.PurePosixPath(item.name).as_posix()==item.name
        assert not item.name.startswith('/') and '..' not in pathlib.PurePosixPath(item.name).parts
        files[item.name]=archive.extractfile(item).read()
assert hashlib.sha256(files['diag/manifest.json']).hexdigest()==C['manifest_pin']
assert hashlib.sha256(files['diag/lme_offline_gate.py']).hexdigest()==C['gate_pin']
m=json.loads(files['diag/manifest.json']); expected={}
for key in ('source_sha256','test_sha256','auxiliary_sha256'):
    assert not set(expected)&set(m[key]); expected.update(m[key])
pins={'verification/'+name:pin for name,pin in expected.items()}
pins.update({'diag/manifest.json':C['manifest_pin'],'diag/lme_offline_gate.py':C['gate_pin']})
assert set(files)==set(pins) and len(expected)==467
assert all(hashlib.sha256(files[name]).hexdigest()==pin for name,pin in pins.items())
assert os.geteuid()==1000 and root.parent.resolve()==root.parent and root.parent.is_dir()
root.mkdir(mode=0o700)
for name in ('verification','diag','results','results/tmp'):
    (root/name).mkdir(mode=0o700)
for name,raw in sorted(files.items()):
    path=root/name; path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream: stream.write(raw)
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==pin for name,pin in pins.items())
receipt={'status':'installed_not_started','archive_sha256':C['archive_sha256'],'manifest_sha256':C['manifest_pin'],
         'gate_sha256':C['gate_pin'],'verification_files':len(expected),'diag_files':2,'provider_calls':0,'candidate_created':False}
with (root/'results/install.json').open('x') as stream: json.dump(receipt,stream,sort_keys=True)
print(json.dumps(receipt,sort_keys=True))
'''


SUPERVISOR = r'''
import json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path('/results'); start=time.monotonic(); timed_out=False; errors=[]
args=json.loads(sys.argv[1])
with (root/'console.log').open('xb') as log:
    child=subprocess.Popen([sys.executable,'-I','-B','/diag/lme_offline_gate.py',*args],stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,env={'PATH':'/home/node/hymem-env/bin:/usr/bin:/bin','TMPDIR':'/results/tmp',
                                                       'HOME':'/results/tmp','PYTHONDONTWRITEBYTECODE':'1'})
    try: code=child.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        timed_out=True
        try: os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        try: code=child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            code=child.wait(timeout=15)
    try: os.killpg(child.pid,0)
    except ProcessLookupError: group_absent=True
    else:
        group_absent=False; errors.append('child_group_survived')
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
receipt={'returncode':code,'timed_out':timed_out,'child_reaped':child.poll() is not None,
         'group_absent':group_absent,'errors':errors,'elapsed_seconds':round(time.monotonic()-start,3),'provider_calls':0}
with (root/'supervisor.json').open('x') as stream: json.dump(receipt,stream,sort_keys=True)
raise SystemExit(0 if code==0 and not timed_out and group_absent else 1)
'''


REMOTE_DOCKER = r'''
import hashlib,json,pathlib,re,subprocess
root=pathlib.Path(C['root']); results=root/'results'
def exclusive(name,value):
    with (results/name).open('x') as stream: json.dump(value,stream,sort_keys=True)
def inspect(cid,expected_state=None):
    assert re.fullmatch('[0-9a-f]{64}',cid)
    obj=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
    cfg,h,state=obj['Config'],obj['HostConfig'],obj['State']
    mounts={v['Destination']:(v['Source'],v['RW'],v['Type']) for v in obj['Mounts']}
    assert mounts=={target:(origin,writable,'bind') for target,origin,writable in C['mounts']}
    assert obj['Image']==C['image'] and cfg['Image']==C['image'] and cfg['User']=='1000:1000'
    assert cfg['WorkingDir']=='/verification' and cfg['Entrypoint']==['/home/node/hymem-env/bin/python3']
    assert cfg['Cmd']==C['container_args'] and obj['Path']=='/home/node/hymem-env/bin/python3' and obj['Args']==C['container_args']
    assert h['NetworkMode']=='none' and h['ReadonlyRootfs'] is True and h['Privileged'] is False
    assert h['CapDrop']==['ALL'] and h['SecurityOpt']==['no-new-privileges'] and h['Init'] is True
    assert h['Memory']==2147483648 and h['NanoCpus']==2000000000 and h['PidsLimit']==128
    assert h['Tmpfs']=={'/tmp':'rw,noexec,nosuid,size=64m'} and h['RestartPolicy']['Name']=='no'
    assert 'TMPDIR=/results/tmp' in cfg['Env']
    assert not any(v.startswith(('DEEPSEEK_','OPENAI_','HYMEM_')) for v in cfg['Env'])
    if expected_state is not None: assert state['Status']==expected_state
    return {'container_id':cid,'status':state['Status'],'exit_code':state['ExitCode'],'oom_killed':state['OOMKilled'],
            'pid':state['Pid'],'configuration_verified':True,'network':'none','credential_mount':False,
            'selected':1044,'manifest_sha256':C['manifest_pin']}
action=C['action']
if action=='create':
    assert (results/'install.json').is_file()
    exclusive('create-intent.json',{'command_sha256':hashlib.sha256(json.dumps(C['command']).encode()).hexdigest(),'selected':1044})
    cid=subprocess.check_output(C['command'],text=True).strip()
    exclusive('container-id.json',{'container_id':cid})
    out=inspect(cid,'created'); exclusive('created.json',out)
elif action=='start':
    cid=C['container_id']; assert json.loads((results/'container-id.json').read_text())['container_id']==cid
    inspect(cid,'created'); exclusive('start-intent.json',{'container_id':cid})
    assert subprocess.check_output(['docker','start',cid],text=True).strip()==cid
    out=inspect(cid)
else:
    cid=C['container_id']; assert json.loads((results/'container-id.json').read_text())['container_id']==cid
    out=inspect(cid)
    for name in ('receipt.json','supervisor.json'):
        path=results/name
        if path.is_file(): out[name]=json.loads(path.read_text())
print(json.dumps(out,sort_keys=True))
'''


def config(action, container_id=None):
    args = ['--tree','/verification','--manifest','/diag/manifest.json','--manifest-sha256',MANIFEST_PIN,
            '--receipt','/results/receipt.json','--junit','/results/junit.xml']
    for name in FOCUS:
        args += ['--focus-file',name]
    container_args = ['-I','-B','-c',SUPERVISOR,json.dumps(args)]
    mounts = [('/home/node/hymem-env',RUNTIME,False),('/verification',REMOTE_ROOT+'/verification',False),
              ('/diag',REMOTE_ROOT+'/diag',False),('/results',REMOTE_ROOT+'/results',True)]
    command = ['docker','create','--name','hymem-r5-offline-target-c67a4ba8bd9c','--pull','never','--init',
               '--network','none','--user','1000:1000','--read-only','--cap-drop','ALL',
               '--security-opt','no-new-privileges','--pids-limit','128','--memory','2g','--cpus','2',
               '--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--env','TMPDIR=/results/tmp']
    for target,origin,writable in mounts:
        command += ['--mount','type=bind,src='+origin+',dst='+target+('' if writable else ',readonly')]
    command += ['--workdir','/verification','--entrypoint','/home/node/hymem-env/bin/python3',IMAGE,*container_args]
    return {'root':REMOTE_ROOT,'manifest_pin':MANIFEST_PIN,'gate_pin':GATE_PIN,'image':IMAGE,
            'action':action,'container_id':container_id,'mounts':mounts,'command':command,'container_args':container_args}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('install','create','start','status'))
    parser.add_argument('--container-id')
    args=parser.parse_args()
    if args.action in ('start','status'):
        assert args.container_id and re.fullmatch('[0-9a-f]{64}',args.container_id)
    else:
        assert args.container_id is None
    cfg=config(args.action,args.container_id)
    raw=b''
    if args.action=='install':
        raw=archive(); cfg['archive_sha256']=sha(raw)
    code='C=json.loads('+repr(json.dumps(cfg))+')\n'
    remote='import json\n'+code+(INSTALL if args.action=='install' else REMOTE_DOCKER)
    result=subprocess.run(['ssh','afrodite','python3 -c '+shlex.quote(remote)],input=raw,capture_output=True,timeout=120)
    if result.returncode:
        print(json.dumps({'status':'target_gate_operation_failed','action':args.action,'returncode':result.returncode,
                          'stderr':result.stderr.decode(errors='replace')[-2000:]}))
        raise SystemExit(1)
    value=json.loads(result.stdout)
    if args.action!='status':
        receipt=REPO/('docs/patches/2026-09-24-lme-r5-target-'+args.action+'.json')
        with receipt.open('x') as stream: json.dump(value,stream,sort_keys=True,indent=2); stream.write('\n')
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':
    main()
