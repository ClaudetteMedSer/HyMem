"""Explicit install/census/read transport for root invocation; no implicit actions."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

FILES=('hymem_v64_rollout.py','hymem_v64_census.py','hymem_v64_postdeploy.py',
       'hymem_v64_vector_check.py','lme_r7_postdeploy_verify.py')
SSH=('ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1',
     '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','afrodite')
REMOTE=r'''
import base64,hashlib,json,os,pathlib,re,stat,subprocess
C=json.loads(CONFIG);stage=pathlib.Path(C['stage'])
def need(ok,code):
    if not ok:raise RuntimeError(code)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,raw,mode):
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
    with os.fdopen(fd,'wb') as s:s.write(raw);s.flush();os.fsync(s.fileno())
def verify():
    need(stage.is_dir() and not stage.is_symlink() and stat.S_IMODE(stage.stat().st_mode)==0o700,'tools_stage_invalid')
    for name,pin in C['pins'].items():
        p=stage/name
        need(p.is_file() and not p.is_symlink() and stat.S_IMODE(p.stat().st_mode)==0o400 and sha(p)==pin,'tools_pin_drift')
need(os.geteuid()==1000,'host_identity')
need(stage.parent==pathlib.Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks')
     and re.fullmatch('hymem-v64-tools-[a-z0-9-]+',stage.name),'tools_stage_name')
need(not stage.parent.is_symlink(),'tools_parent_symlink')
if C['action']=='install':
    need(not stage.exists(),'tools_stage_exists');stage.mkdir(mode=0o700)
    for name,body in C['bodies'].items():write(stage/name,base64.b64decode(body),0o400)
    verify();print(json.dumps({'status':'installed','files':len(C['pins']),'manifest_sha256':C['manifest_sha256']}))
elif C['action']=='census':
    verify()
    p=subprocess.run(['python3','-I','-B',str(stage/'hymem_v64_census.py'),'census',
                      '--output',str(stage/'census.json'),'--runtime-seal-output',str(stage/'runtime-seal.json')],
                     capture_output=True,timeout=900)
    write(stage/'census-private-stdout.bin',p.stdout,0o600)
    write(stage/'census-private-stderr.bin',p.stderr,0o600)
    need(p.returncode==0,'census_failed_inspect_private_evidence')
    print(json.dumps({'status':'captured','census_sha256':sha(stage/'census.json'),
                      'runtime_seal_sha256':sha(stage/'runtime-seal.json')}))
else:
    verify();result={}
    for name in ('census.json','runtime-seal.json'):
        p=stage/name
        need(p.is_file() and not p.is_symlink() and stat.S_IMODE(p.stat().st_mode)==0o600,'receipt_invalid')
        # These scripts emit inventories/hashes only, never config/environment.
        result[name]={'sha256':sha(p),'body_base64':base64.b64encode(p.read_bytes()).decode()}
    print(json.dumps(result))
'''

def inputs():
    root=Path(__file__).resolve().parent
    bodies={name:(root/name).read_bytes() for name in FILES}
    pins={name:hashlib.sha256(body).hexdigest() for name,body in bodies.items()}
    manifest=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return bodies,pins,manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('install','census','read'))
    p.add_argument('--tools-stage',required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--output-directory',type=Path)
    args=p.parse_args()
    bodies,pins,manifest=inputs()
    if args.manifest_sha256!=manifest:raise RuntimeError('local_transport_seal_drift')
    if not re.fullmatch('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-[a-z0-9-]+',args.tools_stage):
        raise RuntimeError('tools_stage_invalid')
    if args.action=='read' and (args.output_directory is None or args.output_directory.exists()):
        raise RuntimeError('fresh_local_output_directory_required')
    cfg={'action':args.action,'stage':args.tools_stage,'pins':pins,'manifest_sha256':manifest}
    if args.action=='install':cfg['bodies']={name:base64.b64encode(body).decode() for name,body in bodies.items()}
    code='CONFIG='+repr(json.dumps(cfg))+'\n'+REMOTE
    result=subprocess.run([*SSH,'python3 -I -B -c '+shlex.quote(code)],capture_output=True,timeout=1000)
    if result.returncode!=0:raise RuntimeError('remote_action_failed_inspect_private_evidence')
    if len(result.stdout)>32*1024*1024:raise RuntimeError('receipt_output_oversized')
    value=json.loads(result.stdout)
    if args.action=='read':
        args.output_directory.mkdir(mode=0o700)
        pins={}
        for name,item in value.items():
            if name not in ('census.json','runtime-seal.json'):raise RuntimeError('unexpected_receipt')
            body=base64.b64decode(item['body_base64'],validate=True)
            if hashlib.sha256(body).hexdigest()!=item['sha256']:raise RuntimeError('download_receipt_drift')
            fd=os.open(args.output_directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as stream:stream.write(body)
            pins[name]=item['sha256']
        value={'status':'retrieved','receipt_sha256':pins}
    print(json.dumps(value,sort_keys=True))

if __name__=='__main__':
    try:main()
    except Exception:
        print(json.dumps({'status':'failed','inspect_private_evidence':True}));raise SystemExit(1)
