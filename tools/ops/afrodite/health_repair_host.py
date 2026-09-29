"""SHA-bound health-checker rehearsal/deployment; no service restarts."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys

ORIGINAL = 'c254b11eecd4322f5110edf2d6d458dd57a83922cef0f3e2416c4058cc9ca378'
CANDIDATE = 'b3de67297a3e24680c0bd02d227f4a10c75b9a1f6915a0478cde43e05b826e13'
SSH = ['ssh', '-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ConnectionAttempts=1', 'afrodite']
REMOTE = r'''
from pathlib import Path
import fcntl, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
package=json.load(sys.stdin); action=sys.argv[1]
root=Path('/opt/stacks/hermes/health-repair-20260926-v2')
live=Path('/opt/stacks/hermes/stack-health-check.sh')
state=Path('/opt/stacks/hermes/health')
original='c254b11eecd4322f5110edf2d6d458dd57a83922cef0f3e2416c4058cc9ca378'
candidate='b3de67297a3e24680c0bd02d227f4a10c75b9a1f6915a0478cde43e05b826e13'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok, code):
    if not ok: raise RuntimeError(code)
def atomic(path, data, mode):
    fd,name=tempfile.mkstemp(prefix='.health-repair-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(),mode)
        os.replace(name,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(name): os.unlink(name)
def report(path):
    d=json.loads(path.read_text())
    return {k:d[k] for k in ('state','ok_count','warn_count','fail_count')}|{'headlines':[f['headline'] for f in d['findings']]}
if action=='stage':
    need(sha(live)==original,'original_source_changed')
    data=package['source'].encode(); need(hashlib.sha256(data).hexdigest()==candidate,'upload_pin_changed')
    root.mkdir(mode=0o700)
    atomic(root/'original.sh',live.read_bytes(),0o700)
    atomic(root/'candidate.sh',data,0o700)
    home=root/'home';home.mkdir(mode=0o700)
    for name in ('cron','scripts'):
        (home/name).symlink_to(Path('/opt/stacks/hermes/instance1/home/.hermes')/name,target_is_directory=True)
    for name in ('original-state','candidate-state'): (root/name).mkdir(mode=0o700)
    syntax=subprocess.run(['bash','-n',str(root/'candidate.sh')],capture_output=True)
    need(syntax.returncode==0,'syntax_failed')
    print(json.dumps({'stage':str(root),'source_sha256':sha(root/'candidate.sh'),'syntax_passed':True}))
elif action=='rehearse':
    need(sha(root/'candidate.sh')==candidate and sha(root/'original.sh')==original,'stage_source_changed')
    results=[]
    for name in ('original','candidate','candidate'):
        p=subprocess.run(['bash',str(root/(name+'.sh')),'--no-notify','--state-dir',str(root/(name+'-state')),'--hermes-home',str(root/'home')],capture_output=True,timeout=120)
        item=report(root/(name+'-state')/'last-run.json')
        item.update(source=name,exit_code=p.returncode,stderr_present=bool(p.stderr))
        results.append(item)
    atomic(root/'rehearsal.json',json.dumps(results).encode(),0o600)
    print(json.dumps(results))
elif action=='deploy':
    need(sha(root/'candidate.sh')==candidate and sha(root/'original.sh')==original,'stage_source_changed')
    receipts=json.loads((root/'rehearsal.json').read_text())
    need(receipts[-1]['state']=='ok' and receipts[-1]['exit_code']==0,'rehearsal_not_clean')
    source=(root/'candidate.sh').read_text()
    template=re.search(r"docker inspect --format \\\n\s*'([^']+)'",source).group(1)
    observations=list((root/'candidate-state').glob('container-*.json'))
    need(len(observations)==14,'observation_inventory_changed')
    with (state/'.health.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(sha(live)==original,'live_source_changed')
        for p in observations:
            old=json.loads(p.read_text()); name=p.name[len('container-'):-len('.json')]
            cur=json.loads(subprocess.check_output(['docker','inspect','--format',template,name]))[0]
            need(cur['Id']==old['id'] and cur['RestartCount']==old['count'] and cur['State']['StartedAt']==old['started'],'container_changed_since_rehearsal')
            need(0<=time.time()-old['observed']<240,'rehearsal_too_old')
            need(not (state/p.name).exists(),'baseline_already_exists')
        # Preserve production alert bookkeeping. Only seed independently
        # verified lifecycle observations; next normal cron handles recovery.
        for p in observations: atomic(state/p.name,p.read_bytes(),0o600)
        atomic(root/'deployed-original.sh',live.read_bytes(),live.stat().st_mode&0o777)
        atomic(live,(root/'candidate.sh').read_bytes(),live.stat().st_mode&0o777)
        need(sha(live)==candidate,'installed_source_mismatch')
        atomic(root/'deploy.json',json.dumps({'source_sha256':candidate,'at':int(time.time()),'baselines':14,'restarts':0}).encode(),0o600)
    print((root/'deploy.json').read_text())
elif action=='status':
    print(json.dumps({'source_sha256':sha(live),'health':report(state/'last-run.json'),'finished_epoch':json.loads((state/'last-run.json').read_text())['finished_epoch']}))
else: raise RuntimeError('unknown_action')
'''

if __name__ == '__main__':
    action = sys.argv[1]
    if action not in ('stage','rehearse','deploy','status'):
        raise SystemExit('unknown_action')
    source = Path(__file__).with_name('stack-health-check.sh').read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=CANDIDATE:
        raise SystemExit('local_source_pin_changed')
    result = subprocess.run(SSH+[shlex.join(['python3','-c',REMOTE,action])],input=json.dumps({'source':source} if action=='stage' else {}),text=True,capture_output=True)
    print(result.stdout,end='')
    if result.returncode:
        print(result.stderr[-4000:],file=sys.stderr)
    raise SystemExit(result.returncode)
