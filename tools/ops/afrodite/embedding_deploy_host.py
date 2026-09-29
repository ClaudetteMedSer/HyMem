"""Install the rehearsed embedding image only; preserve rollback artifacts."""
import json
import shlex
import subprocess
import sys
from embedding_repair_host import SSH

REMOTE=r'''
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,time,urllib.request,math
action=sys.argv[1]
root=Path('/opt/stacks/hermes/embedding-repair-20260926');live=Path('/opt/stacks/embedding-server')
old='sha256:b69045eebd9f3ea874459a176d1316d67ebed05cb70cdb69fd454e704d989865'
new='sha256:4843a433a78681afe25a39e069f212b20c588d5f134621282c7c97872ec11669'
source='0bf31ebda3d13e2b99a68bbccf9dafa571714167a27458e435cf6b8ae17bb435'
old_source='05c369afa23bc246bde4af282050e17b3cc3cb1ffde100c7d76a5e4031b89232'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,code):
    if not ok:raise RuntimeError(code)
def run(args,timeout=120):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError('command_failed:'+p.stderr[-2000:])
    return p.stdout
def inspect(n):return json.loads(run(['docker','inspect',n]))[0]
def save(name,data):
    p=root/name;p.write_text(json.dumps(data));p.chmod(0o600)
def install_bytes(p,data):
    old_stat=p.stat();fd,temp=tempfile.mkstemp(prefix='.reviewed-repair-',dir=p.parent)
    with os.fdopen(fd,'wb') as f:
        f.write(data);f.flush();os.fchmod(f.fileno(),old_stat.st_mode&0o777);os.fsync(f.fileno())
    os.replace(temp,p)
def compose():return run(['docker','compose','-f',str(live/'compose.yaml'),'up','-d','--no-deps','--no-build','--pull','never','embedding-server'])
if action=='deploy':
    need(not (root/'deploy-intent.json').exists(),'deployment_already_attempted')
    for name in ('probe.json','mixed-probe.json'):
        need(json.loads((root/name).read_text())['status']=='passed','functional_gate_not_passed')
    offline=json.loads((root/'offline.json').read_text())
    need(offline['tests_passed']==9 and offline['image']==new,'offline_gate_not_passed')
    need(sha(live/'server.py')==old_source and sha(root/'server.py')==source,'source_pin_changed')
    need(sha(live/'compose.yaml')=='0718d94dcdd0dee70792479c8264221d7ed73006114a0cfb97a88bfd6560b820','compose_pin_changed')
    current=inspect('embedding-server')
    need(current['Image']==old and current['Id']=='cb06ae8d51553158c9c3cb83191d7717dfbfabe8685429e2e64468ab16ff1393' and current['RestartCount']==1 and current['State']['Running'],'unexpected_production_state')
    canary=inspect(json.loads((root/'canary.json').read_text())['container_id'])
    need(canary['Image']==new and canary['State']['Status']=='exited' and not canary['State']['OOMKilled'],'canary_not_stopped_cleanly')
    images={x:json.loads(run(['docker','image','inspect',x]))[0] for x in (old,new)}
    need(images[new]['RootFS']['Layers'][:-1]==images[old]['RootFS']['Layers'],'dependency_layers_changed')
    for key in ('Env','Cmd','Entrypoint','WorkingDir','User','ExposedPorts'):
        need(images[new]['Config'].get(key)==images[old]['Config'].get(key),'image_runtime_config_changed')
    others={n:inspect(n)['Id'] for n in ('hermes-1','hermes-2','hermes-3')}
    save('deploy-intent.json',{'at':int(time.time()),'old_image':old,'new_image':new,'old_container':current['Id'],'other_containers':others})
    # The parent tag already retains the exact rollback image.
    need(json.loads(run(['docker','image','inspect','embedding-server:repair-parent-20260926']))[0]['Id']==old,'rollback_image_changed')
    install_bytes(live/'server.py',(root/'server.py').read_bytes())
    run(['docker','tag',new,'embedding-server:local'])
    try:
        compose()
    except Exception:
        install_bytes(live/'server.py',(root/'original-server.py').read_bytes())
        run(['docker','tag',old,'embedding-server:local']);compose()
        save('rollback.json',{'reason':'compose_failed','at':int(time.time())});raise
    after=inspect('embedding-server')
    need(after['Image']==new,'wrong_image_started')
    need({n:inspect(n)['Id'] for n in others}==others,'other_container_changed')
    result={'at':int(time.time()),'container_id':after['Id'],'image':after['Image'],'source_sha256':sha(live/'server.py'),'other_containers_unchanged':True}
    save('deploy.json',result);print(json.dumps(result))
elif action=='status':
    d=inspect('embedding-server');s=d['State']
    print(json.dumps({'container_id':d['Id'],'image':d['Image'],'status':s['Status'],'health':s.get('Health',{}).get('Status'),'restarts':d['RestartCount'],'oom':s['OOMKilled'],'started':s['StartedAt']}))
elif action=='rollback':
    d=inspect('embedding-server');need(d['Image']==new,'unexpected_image_for_rollback')
    need(sha(live/'server.py')==source and sha(root/'original-server.py')==old_source,'rollback_source_changed')
    install_bytes(live/'server.py',(root/'original-server.py').read_bytes())
    run(['docker','tag',old,'embedding-server:local']);compose()
    result={'at':int(time.time()),'image':inspect('embedding-server')['Image']};save('rollback.json',result);print(json.dumps(result))
else:raise RuntimeError('unknown_action')
'''
if __name__=='__main__':
    action=sys.argv[1]
    if action not in ('deploy','status','rollback'):raise SystemExit('unknown_action')
    p=subprocess.run(SSH+[shlex.join(['python3','-c',REMOTE,action])],capture_output=True,text=True)
    print(p.stdout,end='')
    if p.returncode:print(p.stderr[-4000:],file=sys.stderr)
    raise SystemExit(p.returncode)
