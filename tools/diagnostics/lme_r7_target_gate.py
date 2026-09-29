"""Fixed-scope R7 offline target/startup gate. No production memory or provider calls."""
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
import sys
import tarfile

if sys.flags.optimize:
    raise RuntimeError('optimized_execution_forbidden')

REPO = Path(__file__).resolve().parents[2]
TREE = Path('/private/tmp/hymem-r7-integration-20260925.vDa9yP/frozen-r7-final')
MANIFEST = REPO / 'docs/patches/2026-09-25-lme-r7-final-frozen-manifest.json'
GATE = REPO / 'tools/diagnostics/lme_offline_gate.py'
MANIFEST_PIN = '1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8'
GATE_PIN = 'd53ac383414f077a2af29b506ed883acbcd64cb933da149df3ca5da8cfe6e01a'
REMOTE_ROOT = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
BASELINE = REMOTE_ROOT.rsplit('/', 1)[0] + '/r6-lme-failed-question-v1/candidate'
DATA = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json'
DATA_PIN = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
BASELINE_MANIFEST = REPO / 'docs/patches/2026-09-25-lme-r6-final-frozen-manifest.json'
BASELINE_PIN = 'bd8d0f3a8fb40bd6b77e7ca6579c8e5e8ee78733bea2bb22df71bc4b2c12eaa2'
SUPPORT = REPO / 'tools/diagnostics/lme_r6_regression_v1/bundle'
HELPERS = {'supervised_invocation.py': '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc',
           'lme_q1_startup_preflight.py': 'b17230379613453fd8d9927128f3f35f83d8fd941cc8be4a52e515eddfff7f50'}
FOCUS = ['tests/' + name + '.py' for name in (
    'test_completion_response_admission', 'test_typed_response_recovery', 'test_canary_response_history',
    'test_openai_client', 'test_provider_attempt_accounting', 'test_indexing_deadlines',
    'test_benchmark_client_cleanup', 'test_benchmark_extraction_canary',
    'test_digest_summary_index_separation', 'test_digest_summary_separate_publication', 'test_digest_publication',
    'test_summary_frontier_v62', 'test_summary_recovery_v63', 'test_summary_recovery_integration_root',
    'test_summary_recovery_command_root', 'test_lme_protocol_hardening', 'test_lme_historical_admission',
    'test_lme_registry_cli_bootstrap', 'test_lme_current_model_startup', 'test_startup_schema_atomicity',
    'test_summary_portability_v18', 'test_benchmark_summary_degradation',
    'test_loaded_identity_traversal', 'test_loaded_identity_performance',
    'test_phase1_producer_identity', 'test_extraction_contract_identity',
    'test_producer_gc_reentrancy', 'test_benchmark_code_identity',
    'test_benchmark_cli_imports', 'test_beam_cli_imports', 'test_registry_historical_models',
    'test_peer_provenance', 'test_dream_scheduler', 'test_dream_lease', 'test_embeddings',
    'test_aggregation_health', 'test_orphan_quarantine_rehearsal', 'test_honcho_server',
    'test_benchmark_adapter_strictness', 'test_store_attestation',
    'test_lme_collection_determinism', 'test_terminal_clean_empty_policy',
    'test_lme_embedding_setup_guidance', 'test_connection_lifecycle',
    'test_shared_statement_cache', 'test_locomo_judge_endpoint')]
SSH_OPTIONS = ('-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=2')
# Installation streams the sealed archive over SSH; the other operations send
# only a small command. A local timeout cannot establish whether remote work
# completed, so neither path may automatically retry.
ACTION_TIMEOUT_SECONDS = {'install': 300, 'create': 120, 'start': 120, 'status': 120}


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
    assert len(manifest['source_sha256']) == 231 and len(manifest['expected_nodeids']) > 7319
    assert all(name in manifest['test_sha256'] for name in FOCUS)
    nodes = [node for node in manifest['expected_nodeids'] if node.split('::', 1)[0] in FOCUS]
    assert nodes and set(manifest['expected_skip_nodeids']) <= set(nodes)
    actual = set()
    for path in TREE.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            actual.add(path.relative_to(TREE).as_posix())
    assert actual == set(expected)
    files = {'diag/manifest.json': raw_manifest, 'diag/lme_offline_gate.py': raw_gate,
             'diag/r6-manifest.json': BASELINE_MANIFEST.read_bytes()}
    assert sha(files['diag/r6-manifest.json']) == BASELINE_PIN
    for name, pin in HELPERS.items():
        files['diag/' + name] = (SUPPORT / name).read_bytes()
        assert sha(files['diag/' + name]) == pin
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
if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
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
pins.update({'diag/'+name:pin for name,pin in C['helpers'].items()})
pins['diag/r6-manifest.json']=C['baseline_pin']
assert set(files)==set(pins) and len(expected)==C['files']
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
         'gate_sha256':C['gate_pin'],'verification_files':len(expected),'diag_files':len(pins)-len(expected),'provider_calls':0,'candidate_created':False}
with (root/'results/install.json').open('x') as stream: json.dump(receipt,stream,sort_keys=True)
print(json.dumps(receipt,sort_keys=True))
'''


SUPERVISOR = r'''
import dataclasses,hashlib,json,pathlib,signal,sys,types
if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
diag=pathlib.Path('/diag'); diag_pins=json.loads(sys.argv[3])
def verify_diag():
    actual=set()
    for path in diag.rglob('*'):
        assert not path.is_symlink()
        if path.is_file(): actual.add(path.relative_to(diag).as_posix())
    assert actual==set(diag_pins)
    bodies={name:(diag/name).read_bytes() for name in diag_pins}
    assert all(hashlib.sha256(body).hexdigest()==diag_pins[name] for name,body in bodies.items())
    return bodies
def load_verified(name,body):
    module=types.ModuleType(name); module.__file__=str(diag/(name+'.py'))
    sys.modules[name]=module
    exec(compile(body,module.__file__,'exec'),module.__dict__)
    return module
bodies=verify_diag()
supervise_invocation=load_verified('supervised_invocation',bodies['supervised_invocation.py']).supervise_invocation
gate=load_verified('lme_offline_gate',bodies['lme_offline_gate.py'])
expected_files,verify_tree=gate.expected_files,gate.verify_tree
root=pathlib.Path('/results'); args=json.loads(sys.argv[1]); stages={}
def cancelled(signum,frame): raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,cancelled)
env={'PATH':'/home/node/hymem-env/bin:/usr/bin:/bin','TMPDIR':'/results/tmp',
     'HOME':'/results/tmp','PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0'}
baseline=json.loads(bodies['r6-manifest.json'])
current=json.loads(bodies['manifest.json'])
def verified():
    verify_diag()
    verify_tree(pathlib.Path('/baseline'),baseline['source_sha256'])
    verify_tree(pathlib.Path('/verification'),expected_files(current))
    assert hashlib.sha256(pathlib.Path('/data/longmemeval_s_cleaned.json').read_bytes()).hexdigest()==sys.argv[2]
def invoke(name,command,timeout):
    outcome=supervise_invocation(command,cwd='/verification',env=env,
        output_dir=root/('supervised-'+name),timeout_seconds=timeout,cleanup_seconds=10)
    stages[name]=dataclasses.asdict(outcome)
    assert outcome.safe_to_continue and outcome.status=='completed' and outcome.returncode==0
report={'provider_calls':0,'status':'failed','stages':stages}
try:
    verified()
    invoke('tests',[sys.executable,'-I','-B','/diag/lme_offline_gate.py',*args],3600)
    identity_code="import json,pathlib,sys;sys.path.insert(0,sys.argv[1]);from hymem.extraction.contract import extraction_contract_identity;pathlib.Path(sys.argv[2]).write_text(json.dumps({'contract':extraction_contract_identity()}))"
    for label,source in (('r6','/baseline'),('r7','/verification')):
        invoke('startup-'+label,[sys.executable,'-I','-B','/diag/lme_q1_startup_preflight.py',
            '--source',source,'--data-dir','/data','--output',str(root/('startup-'+label)),
            '--question-ids-json','["gpt4_483dd43c"]'],300)
        invoke('contract-'+label,[sys.executable,'-I','-B','-c',identity_code,source,
            str(root/('contract-'+label+'.json'))],120)
    startups={label:json.loads((root/('startup-'+label)/'startup-report.json').read_text()) for label in ('r6','r7')}
    contracts={label:json.loads((root/('contract-'+label+'.json')).read_text()) for label in ('r6','r7')}
    assert all(v['status']=='passed' and v['provider_completions']==v['provider_http_attempts']==0
               and v['runtime_clients_constructed']==v['runtime_clients_closed']==1
               and v['cli_checkpoint_handles_closed'] for v in startups.values())
    assert startups['r6']['producer_identity_sha256']==startups['r7']['producer_identity_sha256']
    assert startups['r6']['producer_identity_sha256']=='sha256:b7dcd2a8a3a5107c9b0d868ecd2f3c96ab7d9f02392e20c9b967c44805673cb5'
    assert contracts['r6']==contracts['r7']
    report.update(startup_reports=startups,contract_reports=contracts,
                  producer_identity_parity=True,extraction_contract_parity=True)
    verified()
    report['source_verified_before_and_after']=True
    report['status']='passed'
except BaseException as exc:
    report['status']='failed'
    report['error_type']=type(exc).__name__
finally:
    with (root/'supervisor.json').open('x') as stream:json.dump(report,stream,sort_keys=True)
raise SystemExit(0 if report['status']=='passed' else 1)
'''


REMOTE_DOCKER = r'''
import hashlib,json,pathlib,re,subprocess,sys
if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
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
            'selected':C['selected'],'manifest_sha256':C['manifest_pin']}
action=C['action']
if action=='create':
    assert (results/'install.json').is_file()
    exclusive('create-intent.json',{'command_sha256':hashlib.sha256(json.dumps(C['command']).encode()).hexdigest(),'selected':C['selected']})
    cid=subprocess.check_output(C['command'],text=True).strip()
    exclusive('container-id.json',{'container_id':cid})
    out=inspect(cid,'created'); exclusive('created.json',out)
elif action=='start':
    cid=C['container_id']; assert json.loads((results/'container-id.json').read_text())['container_id']==cid
    inspect(cid,'created'); exclusive('start-intent.json',{'container_id':cid})
    started=subprocess.check_output(['docker','start',cid],text=True).strip()
    assert started==cid
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
    diag_pins = {'manifest.json':MANIFEST_PIN,'r6-manifest.json':BASELINE_PIN,
                 'lme_offline_gate.py':GATE_PIN,**HELPERS}
    container_args = ['-I','-B','-c',SUPERVISOR,json.dumps(args),DATA_PIN,json.dumps(diag_pins,sort_keys=True)]
    mounts = [('/home/node/hymem-env',RUNTIME,False),('/verification',REMOTE_ROOT+'/verification',False),
              ('/diag',REMOTE_ROOT+'/diag',False),('/results',REMOTE_ROOT+'/results',True),
              ('/baseline',BASELINE,False),('/data/longmemeval_s_cleaned.json',DATA,False)]
    command = ['docker','create','--name','hymem-r7-offline-target-'+MANIFEST_PIN[:12],'--pull','never','--init',
               '--network','none','--user','1000:1000','--read-only','--cap-drop','ALL',
               '--security-opt','no-new-privileges','--pids-limit','128','--memory','2g','--cpus','2',
               '--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--env','TMPDIR=/results/tmp']
    for target,origin,writable in mounts:
        command += ['--mount','type=bind,src='+origin+',dst='+target+('' if writable else ',readonly')]
    command += ['--workdir','/verification','--entrypoint','/home/node/hymem-env/bin/python3',IMAGE,*container_args]
    raw = MANIFEST.read_bytes()
    assert sha(raw) == MANIFEST_PIN
    manifest = json.loads(raw)
    selected = sum(node.split('::',1)[0] in FOCUS for node in manifest['expected_nodeids'])
    files = sum(len(manifest[key]) for key in ('source_sha256','test_sha256','auxiliary_sha256'))
    return {'root':REMOTE_ROOT,'manifest_pin':MANIFEST_PIN,'gate_pin':GATE_PIN,'image':IMAGE,
            'selected':selected,'files':files,'baseline_pin':BASELINE_PIN,'helpers':HELPERS,
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
    timeout_seconds=ACTION_TIMEOUT_SECONDS[args.action]
    try:
        result=subprocess.run(
            ['ssh',*SSH_OPTIONS,'afrodite','python3 -I -B -c '+shlex.quote(remote)],
            input=raw,capture_output=True,timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # TimeoutExpired carries the entire inline command and possibly partial
        # output. Never render it: the remote operation may have completed.
        print(json.dumps({'status':'target_gate_timeout','action':args.action,
                          'outcome':'unknown','requires_inspection':True,
                          'retry_attempted':False,'timeout_seconds':timeout_seconds},
                         sort_keys=True))
        raise SystemExit(1)
    if result.returncode:
        print(json.dumps({'status':'target_gate_operation_failed','action':args.action,'returncode':result.returncode,
                          'stderr':result.stderr.decode(errors='replace')[-2000:]}))
        raise SystemExit(1)
    value=json.loads(result.stdout)
    if args.action!='status':
        receipt=REPO/('docs/patches/2026-09-25-lme-r7-target-'+args.action+'.json')
        with receipt.open('x') as stream: json.dump(value,stream,sort_keys=True,indent=2); stream.write('\n')
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':
    main()
