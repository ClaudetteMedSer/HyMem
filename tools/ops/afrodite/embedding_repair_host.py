"""Pinned, isolated rehearsal of the Afrodite embedding-server repair."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

SSH=['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','afrodite']
REMOTE=r'''
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
root=Path('/opt/stacks/hermes/embedding-repair-20260926')
live=Path('/opt/stacks/embedding-server')
old_image='sha256:b69045eebd9f3ea874459a176d1316d67ebed05cb70cdb69fd454e704d989865'
old_source='05c369afa23bc246bde4af282050e17b3cc3cb1ffde100c7d76a5e4031b89232'
source='0bf31ebda3d13e2b99a68bbccf9dafa571714167a27458e435cf6b8ae17bb435'
tag='embedding-server:bounded-20260926'
parent='embedding-server:repair-parent-20260926'
name='embedding-server-bounded-canary-20260926'
cache='embedding-repair-cache-20260926'
action=sys.argv[1];package=json.load(sys.stdin)
def need(ok,code):
    if not ok:raise RuntimeError(code)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args,timeout=120):
    r=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError('command_failed:'+args[0]+':'+r.stderr[-1500:])
    return r.stdout
def inspect(container):return json.loads(run(['docker','inspect',container]))[0]
def receipt(d):
    (root/(action+'.json')).write_text(json.dumps(d));print(json.dumps(d))
if action=='stage':
    need(sha(live/'server.py')==old_source,'live_source_changed')
    need(inspect('embedding-server')['Image']==old_image,'live_image_changed')
    root.mkdir(mode=0o700)
    for rel,contents in package.items():
        need(rel in ('server.py','tests/test_server.py','tests/test_server_root.py'),'unexpected_file')
        p=root/rel;p.parent.mkdir(mode=0o700,exist_ok=True);p.write_text(contents);p.chmod(0o600)
    need(sha(root/'server.py')==source,'candidate_pin_changed')
    (root/'original-server.py').write_bytes((live/'server.py').read_bytes())
    (root/'original-compose.yaml').write_bytes((live/'compose.yaml').read_bytes())
    (root/'Dockerfile').write_text('FROM '+parent+'\nCOPY server.py /app/server.py\n')
    receipt({'source_sha256':source,'parent_image':old_image,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
elif action=='build':
    need(sha(root/'server.py')==source,'candidate_pin_changed')
    need(inspect('embedding-server')['Image']==old_image,'live_image_changed')
    existing=run(['docker','image','ls','--format','{{.Repository}}:{{.Tag}}'])
    need(tag not in existing.splitlines() and parent not in existing.splitlines(),'candidate_tags_already_exist')
    run(['docker','tag',old_image,parent])
    output=run(['docker','build','--network','none','--pull=false','-t',tag,str(root)],timeout=300)
    (root/'build.log').write_text(output)
    image=json.loads(run(['docker','image','inspect',tag]))[0]['Id']
    receipt({'image':image,'parent_image':old_image,'source_sha256':source})
elif action=='offline':
    image=json.loads((root/'build.json').read_text())['image']
    need(sha(root/'server.py')==source,'candidate_pin_changed')
    result=subprocess.run(['docker','run','--rm','--network','none','--memory','1g','--cpus','2','--env','PYTHONDONTWRITEBYTECODE=1','--mount','type=bind,src='+str(root)+',dst=/review,readonly','--entrypoint','python3',image,'-m','unittest','discover','-s','/review/tests','-v'],text=True,capture_output=True,timeout=120)
    output=result.stdout+result.stderr
    (root/'offline.log').write_text(output)
    need(result.returncode==0 and re.search(r'Ran 9 tests in ',output) and output.rstrip().endswith('OK'),'offline_tests_not_proven')
    receipt({'tests_passed':9,'network':'none','image':image,'source_sha256':source})
elif action=='canary':
    image=json.loads((root/'build.json').read_text())['image']
    need(json.loads((root/'offline.json').read_text())['tests_passed']==9,'offline_gate_missing')
    need(name not in run(['docker','ps','-a','--format','{{.Names}}']).splitlines(),'canary_already_exists')
    need(cache not in run(['docker','volume','ls','--format','{{.Name}}']).splitlines(),'canary_cache_already_exists')
    run(['docker','volume','create',cache])
    run(['docker','run','--rm','--network','none','--memory','1g','--cpus','2','--mount','type=volume,src=embedding-server_embedding-model-cache,dst=/cache,readonly','--mount','type=volume,src='+cache+',dst=/copy','--entrypoint','cp',old_image,'-a','/cache/.','/copy/'],timeout=180)
    production=inspect('embedding-server')
    # Copy only known non-secret model/runtime settings. No provider credentials.
    env=[]
    for item in production['Config']['Env']:
        key=item.split('=',1)[0]
        if key in ('EMBED_MODEL','EMBED_PORT','EMBED_DIM','EMBED_CACHE_DIR','EMBED_MAX_TOKENS','RERANK_MODEL','RERANK_PRELOAD','RERANK_BATCH_SIZE','RERANK_MAX_CHARS','RERANK_MAX_DOCS','RERANK_THREADS'):
            env.extend(['--env',item])
    cid=run(['docker','run','-d','--name',name,'--network','none','--memory','3g','--memory-swap','3g','--cpus','2','--restart','no','--mount','type=volume,src='+cache+',dst=/cache',*env,image]).strip()
    receipt({'container_id':cid,'image':image,'network':'none','cache':cache,'production_unchanged':production['Image']==old_image})
elif action=='status':
    d=inspect(name);st=d['State']
    print(json.dumps({'container_id':d['Id'],'image':d['Image'],'status':st['Status'],'pid':st['Pid'],'oom':st['OOMKilled'],'restarts':d['RestartCount'],'health':st.get('Health',{}).get('Status')}))
elif action=='stop-canary':
    expected=json.loads((root/'canary.json').read_text())['container_id']
    need(inspect(name)['Id']==expected,'canary_identity_changed')
    run(['docker','stop','--time','20',expected],timeout=40)
    d=inspect(expected)
    receipt({'container_id':expected,'status':d['State']['Status'],'exit_code':d['State']['ExitCode'],'oom':d['State']['OOMKilled']})
else:raise RuntimeError('unknown_action')
'''
if __name__=='__main__':
    action=sys.argv[1]
    if action not in ('stage','build','offline','canary','status','stop-canary'):raise SystemExit('unknown_action')
    base=Path(__file__).with_name('embedding-server')
    package={n:(base/n).read_text() for n in ('server.py','tests/test_server.py','tests/test_server_root.py')} if action=='stage' else {}
    r=subprocess.run(SSH+[shlex.join(['python3','-c',REMOTE,action])],input=json.dumps(package),text=True,capture_output=True)
    print(r.stdout,end='')
    if r.returncode:print(r.stderr[-4000:],file=sys.stderr)
    raise SystemExit(r.returncode)
