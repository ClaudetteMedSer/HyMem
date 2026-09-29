"""Upload and invoke one reviewed network-none postflight of a sealed run."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys

BASE='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/cold-replay-dream-v1'
SEAL={
    'host_sha256':'38df85589fbc561e287bc75329b3ab785af001dd0f6bc3024cbb09f629820f0c',
    'install_sha256':'26dedaa596d00f5370c1e1210ed3f053f1110ccd9afa7efc264800cb9cab5968',
    'result_sha256':'d19429e8da253445575594e6a907955569c0e1446fd13f9fae0d9ef2a0df75da',
    'live_container_id':'8a726a4be865885910f20a5d396f659a1cc59f3dbbe8eaae25052e42c38bad5d',
    'source_sha256':'f4f9cb8ec8247d27ab2981af58ec0c044fb76cc71356f2e6a757540d4eae59ae',
}
PINS={
    'claim_conflict_postflight_host.py':'6b1ad8ded38f76adba0b5005caa13ee95f5eec1a2026f8d84470eacb9a4f6aae',
    'claim_conflict_private_dream_postflight.py':'1ca6318fdac58d7b0715547baa8b80d4119c9a1a2012fb9f58bf96c3a443b117',
    'claim_conflict_store_audit.py':'e2efe365c5aedbbe88d86d521dd37b821759afc5fb21c21a6662e6a8ce567f42',
}
REMOTE=r'''
import hashlib,json,os,pathlib,sys
config=json.load(sys.stdin)
root=pathlib.Path(config['root']);stage=root/'postflight-v1'
assert os.geteuid()==1000 and not stage.exists()
for name,key in [('claim_conflict_v64_dream_host.py','host_sha256'),('install.json','install_sha256'),('result.json','result_sha256'),('work/live/hymem.sqlite','source_sha256')]:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==config['seal'][key]
stage.mkdir(mode=0o700)
for name,item in config['files'].items():
    raw=item['text'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256']
    fd=os.open(stage/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
raw=json.dumps(config['seal'],sort_keys=True).encode();seal_sha=hashlib.sha256(raw).hexdigest()
fd=os.open(stage/'seal.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'stage':str(stage),'seal_sha256':seal_sha,'status':'installed_not_launched'}))
'''

if __name__=='__main__':
    ssh=['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','afrodite']
    action=sys.argv[1]
    if action=='install':
        files={}
        for name,pin in PINS.items():
            raw=Path(__file__).with_name(name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=pin:raise SystemExit('local_postflight_pin_drift')
            files[name]={'sha256':pin,'text':raw.decode()}
        r=subprocess.run(ssh+[shlex.join(['python3','-I','-B','-c',REMOTE])],input=json.dumps({'root':BASE,'seal':SEAL,'files':files}),text=True,capture_output=True)
    elif action=='run':
        seal_sha=hashlib.sha256(json.dumps(SEAL,sort_keys=True).encode()).hexdigest()
        r=subprocess.run(ssh+[shlex.join(['python3','-I','-B',BASE+'/postflight-v1/claim_conflict_postflight_host.py','--seal',BASE+'/postflight-v1/seal.json','--seal-sha256',seal_sha])],text=True,capture_output=True)
    else:raise SystemExit('unknown_action')
    print(r.stdout,end='')
    if r.returncode:print(r.stderr[-2000:],file=sys.stderr)
    raise SystemExit(r.returncode)
