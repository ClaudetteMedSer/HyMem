"""Explicit reviewed metadata upload only; never reviews, launches, or deploys."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hymem_v64_rollout as r

STAGE='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-v1'
NAMES=('candidate-manifest.json','fullsuite-gate.json','paid-postflight-gate.json',
       'census-reviewed.json','rollout-config.json')
REMOTE=r'''
import base64,hashlib,json,os,pathlib,stat
C=json.loads(CONFIG);stage=pathlib.Path(C['stage'])
def need(ok,code):
    if not ok:raise RuntimeError(code)
need(os.geteuid()==1000 and str(stage)=='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-20260926-v1','target_identity')
need(stage.is_dir() and not stage.is_symlink() and stat.S_IMODE(stage.stat().st_mode)==0o700,'private_existing_stage_required')
need(set(C['files'])=={'candidate-manifest.json','fullsuite-gate.json','paid-postflight-gate.json','census-reviewed.json','rollout-config.json'},'metadata_names')
for name in C['files']:need(not os.path.lexists(stage/name),'metadata_already_present')
for name,item in C['files'].items():
    body=base64.b64decode(item['body'],validate=True)
    need(hashlib.sha256(body).hexdigest()==item['sha256'],'metadata_bytes_drift')
for name,item in C['files'].items():
    fd=os.open(stage/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(base64.b64decode(item['body']));stream.flush();os.fsync(stream.fileno())
    need(hashlib.sha256((stage/name).read_bytes()).hexdigest()==item['sha256'],'metadata_postwrite_drift')
print(json.dumps({'status':'uploaded','metadata_sha256':{n:v['sha256'] for n,v in C['files'].items()},'launch_performed':False,'review_performed':False},sort_keys=True))
'''

def prepare(directory):
    files={name:r.regular(directory/name) for name in NAMES}
    r.manifest(files['candidate-manifest.json'],r.CANDIDATE_PIN,481)
    values={name:json.loads(path.read_bytes()) for name,path in files.items()}
    for name in ('fullsuite-gate.json','paid-postflight-gate.json','census-reviewed.json'):
        r.need(values[name].get('root_reviewed') is True,'external_root_review_required')
    census=values['census-reviewed.json']
    r.need(census.get('schema')==63 and census.get('image')==r.IMAGE,'census_identity')
    cfg=values['rollout-config.json']
    r.need(cfg.get('candidate_manifest_sha256')==r.CANDIDATE_PIN
           and cfg.get('candidate_manifest')==STAGE+'/candidate-manifest.json','config_candidate_identity')
    for role,name in (('fullsuite','fullsuite-gate.json'),('paid_postflight','paid-postflight-gate.json')):
        gate=cfg['gates'][role]
        r.need(gate['path']==STAGE+'/'+name and gate['sha256']==r.sha(files[name]),'gate_binding_drift')
    r.need(cfg['census']['path']==STAGE+'/census-reviewed.json'
           and cfg['census']['sha256']==r.sha(files['census-reviewed.json']),'census_binding_drift')
    # Validate gate semantics from local paths without changing their JSON.
    local=dict(cfg);local['gates']={role:{**cfg['gates'][role],'path':str(files[name])}
          for role,name in (('fullsuite','fullsuite-gate.json'),('paid_postflight','paid-postflight-gate.json'))}
    r.gates(local)
    def no_secrets(value):
        if isinstance(value,dict):
            for key,item in value.items():
                r.need(key.lower() not in ('env','environment','api_key','credentials','password','token','secret',
                                           'hymem_llm_api_key','hymem_embedding_api_key','extra_body'),'raw_secret_metadata_forbidden')
                no_secrets(item)
        elif isinstance(value,list):
            for item in value:no_secrets(item)
    for value in values.values():no_secrets(value)
    return {'stage':STAGE,'files':{name:{'sha256':r.sha(path),'body':base64.b64encode(path.read_bytes()).decode()}
                                for name,path in files.items()}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','upload'))
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    cfg=prepare(args.directory)
    pins={name:item['sha256'] for name,item in cfg['files'].items()}
    r.need(r.digest(pins)==args.manifest_sha256,'metadata_manifest_drift')
    if args.action=='prepare':
        print(json.dumps({'status':'prepared','metadata_sha256':pins,'launch_performed':False,'review_performed':False},sort_keys=True))
        return
    code='CONFIG='+repr(json.dumps(cfg))+'\n'+REMOTE
    command=['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=10','afrodite',
             'python3 -I -B -c '+shlex.quote(code)]
    reply=subprocess.run(command,capture_output=True,timeout=120)
    r.need(reply.returncode==0 and len(reply.stdout)<8192,'metadata_upload_failed_inspect_before_retry')
    result=json.loads(reply.stdout)
    r.need(result.get('status')=='uploaded' and result.get('metadata_sha256')==pins,'metadata_remote_verdict')
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':
    try:main()
    except Exception:
        print(json.dumps({'status':'failed','launch_performed':False,'review_performed':False}));raise SystemExit(1)
