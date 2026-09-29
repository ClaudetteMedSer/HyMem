"""Read-only host census and upload-command preparation. Never invokes SSH."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hymem_v64_rollout as r
from hymem_v64_vector_check import CHECKER

PROCESS_SCRIPT = r'''
import hashlib,importlib.metadata as m,json,os,pathlib
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
roles={'honcho':[],'mcp':[]};envs={}
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdecimal():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    role=None
    if b'hymem.honcho' in args or any(a.rsplit(b'/',1)[-1]==b'hymem-honcho' for a in args):role='honcho'
    if b'hymem.server' in args or any(a.rsplit(b'/',1)[-1]==b'hymem-server' for a in args):role='mcp'
    if role:
        raw=(p/'environ').read_bytes()
        env={os.fsdecode(k):os.fsdecode(v) for item in raw.split(b'\0') if b'=' in item for k,v in [item.split(b'=',1)]}
        profile={k:v for k,v in env.items() if k.startswith('HYMEM_')}
        roles[role].append({'pid':int(p.name),'raw_environ_sha256':hashlib.sha256(raw).hexdigest(),'profile_sha256':digest(profile)})
        envs.setdefault(role,[]).append(env)
assert len(roles['honcho'])==1 and len(roles['mcp'])==2
assert len({x['profile_sha256'] for x in roles['mcp']})==1
h=envs['honcho'][0]
assert all(e.get('HYMEM_LLM_MODEL')=='deepseek-flash' for vals in envs.values() for e in vals)
keys=set(k for vals in envs.values() for e in vals for k in e if k.startswith(('HYMEM_LLM_','HYMEM_EMBEDDING_')))
assert all(h.get(k)==e.get(k) for e in envs['mcp'] for k in keys)
rows={d.metadata['Name']:d.version for d in m.distributions() if d.metadata.get('Name','').lower()!='hymem'}
distribution=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
print(json.dumps({'roles':roles,'llm_embedding_parity':True,'distribution_count':len(rows),'distribution_sha256':distribution}))
'''

def capture(extra_paths=()):
    r.need(os.geteuid()==1000 and r.HOME.is_dir(),'afrodite_host_identity_required')
    def run(cmd):
        p=subprocess.run(cmd,capture_output=True,timeout=60)
        r.need(p.returncode==0 and len(p.stdout)<2_000_000,'readonly_census_failed')
        return p.stdout
    obj=json.loads(run(['docker','inspect','hermes-1']))[0]
    r.need(obj['Name']=='/hermes-1' and obj['Image']==r.IMAGE and obj['State']['Running']
           and not obj['State']['OOMKilled'],'container_identity')
    processes=json.loads(run(['docker','exec','hermes-1','/home/node/hymem-env/bin/python3','-I','-B','-c',PROCESS_SCRIPT]))
    vector_script=CHECKER+r'''
import json,pathlib,sqlite3
from hymem.core import db
c=sqlite3.connect('file:/home/node/.hermes/hymem.sqlite?mode=ro',uri=True)
c.row_factory=sqlite3.Row
c.execute('PRAGMA query_only=ON')
try:
    try:result=strict_episode_vectors(c,db)
    except Exception as exc:
        result={'verifiable':False,'aligned':False,'failure_code':str(exc) if type(exc) is RuntimeError else type(exc).__name__}
finally:c.close()
print(json.dumps(result))
'''
    vectors=json.loads(run(['docker','exec','hermes-1','/home/node/hymem-env/bin/python3','-I','-B','-c',vector_script]))
    required=[r.RUNTIME,r.HOME/'.hermes/bin/hymem-server-wrapper',r.HOME/'.agent37/hooks/post-restart.sh']
    env_files=[p for p in (r.HOME/'.env',r.HOME/'.hermes/.env',r.HOME.parent/'.env') if p.exists()]
    r.need(bool(env_files),'env_config_inventory_missing')
    configs=[r.HOME/'.hermes/config.yaml',r.HOME/'.hermes/config.yml',r.HOME/'.agent37/config.yaml']
    paths=sorted(set(required+env_files+[p for p in configs if p.exists()]+list(extra_paths)))
    preserved={str(p):r.digest(r.inventory(p)) for p in paths}
    honcho=processes['roles']['honcho'][0]
    baseline=r.HOME/'.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/diag/manifest.json'
    files=r.manifest(baseline,r.BASELINE_PIN,479,grouped=True)
    files.update({'hymem/doctor.py':r.DOCTOR_PIN,'hymem/dreaming/phase1.py':r.PHASE1_PIN})
    r.verify_files(r.LIVE,files)
    conn=sqlite3.connect(r.DB.as_uri()+'?mode=ro',uri=True,timeout=30)
    try: schema=int(conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0])
    finally:conn.close()
    r.need(schema==63,'production_schema_drift')
    return {'status':'captured','root_reviewed':False,'schema':schema,'source_files_verified':479,
            'production_source_manifest_sha256':r.digest(files),
            'image':obj['Image'],'container_id':obj['Id'],'docker_config_sha256':r.digest(obj['Config']),
            'preserved_paths':preserved,'preserved_file_sha256':{str(p):r.sha(p) for p in paths if p.is_file()},
            'health_port':8765,'honcho_pid':honcho['pid'],
            'honcho_environ_sha256':honcho['raw_environ_sha256'],
            'role_profile_sha256':{role:sorted(set(x['profile_sha256'] for x in vals))
                                  for role,vals in processes['roles'].items()},
            'processes':processes,'episode_vector_baseline':vectors}

def runtime_seal(root=r.RUNTIME):
    return {'schema':'lme-v64-runtime-seal-v1','image':r.IMAGE,'runtime_root':str(root),
            'root_reviewed':False,'entries':r.inventory(root)}

def upload_plan(paths,remote_directory):
    """Return reviewable commands without invoking SSH/SCP or creating paths."""
    r.need(remote_directory.startswith(str(r.HOME/'.hermes/benchmarks/hymem-v64-rollout-'))
           and ':' not in remote_directory and '..' not in Path(remote_directory).parts,'upload_stage_invalid')
    rows=[]
    for p in paths:
        p=r.regular(p).resolve()
        rows.append({'file':str(p),'sha256':r.sha(p),'command':shlex.join(['scp','-p',str(p),'afrodite:'+remote_directory+'/'+p.name])})
    return {'status':'prepared_only','remote_directory_must_be_private':True,'files':rows}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    census=sub.add_parser('census');census.add_argument('--output',required=True,type=Path)
    census.add_argument('--preserved-path',action='append',type=Path,default=[])
    census.add_argument('--runtime-seal-output',type=Path)
    upload=sub.add_parser('upload-plan');upload.add_argument('--remote-directory',required=True)
    upload.add_argument('files',nargs='+',type=Path)
    args=parser.parse_args()
    if args.action=='upload-plan':print(json.dumps(upload_plan(args.files,args.remote_directory),sort_keys=True))
    else:
        value=capture(args.preserved_path);r.exclusive_json(args.output,value)
        if args.runtime_seal_output:
            # Shared LME contract: canonical pretty JSON with one newline.
            fd=os.open(args.runtime_seal_output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'w') as stream:
                stream.write(json.dumps(runtime_seal(),sort_keys=True,indent=2,allow_nan=False)+'\n')
                stream.flush();os.fsync(stream.fileno())
        print(json.dumps({'status':'captured','receipt_sha256':r.sha(args.output),'root_review_required':True}))

if __name__=='__main__':
    try:main()
    except Exception:
        print(json.dumps({'status':'failed','readonly_census_rejected':True}));raise SystemExit(1)
