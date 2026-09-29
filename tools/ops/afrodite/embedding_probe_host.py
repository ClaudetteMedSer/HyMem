"""Synthetic pinned-image rehearsal; vectors never leave Afrodite."""
import json
import shlex
import subprocess
from embedding_repair_host import SSH

PROBE=r'''
import concurrent.futures,json,math,threading,time,urllib.request
base='http://127.0.0.1:8766'
def request(path,body=None):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=100) as r:return json.load(r)
def embedding(texts):
    d=request('/v1/embeddings',{'input':texts})
    assert len(d['data'])==len(texts)
    assert [r['index'] for r in d['data']]==list(range(len(texts)))
    assert all(len(r['embedding'])==384 and all(math.isfinite(v) for v in r['embedding']) for r in d['data'])
    return d
assert request('/health')['status']=='ok'
texts=['Synthetic embedding regression sample number '+str(i) for i in range(33)]
control=embedding(texts)
rank=request('/v1/rerank',{'query':'local memory search','documents':['Local memory search stores and retrieves facts.','Fresh bananas grow on a tree.'],'top_n':2})
assert len(rank['results'])==2 and sorted(v['index'] for v in rank['results'])==[0,1]
assert all(math.isfinite(v['relevance_score']) for v in rank['results'])
latencies=[]
long=['Synthetic bounded inference memory test. '*180+str(i) for i in range(16)]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    future=pool.submit(embedding,long)
    while not future.done():
        begin=time.monotonic();assert request('/health')['status']=='ok';latencies.append(time.monotonic()-begin);time.sleep(.1)
    bounded=future.result()
assert latencies and max(latencies)<2
health=request('/health');assert health['rerank']['loaded']
print(json.dumps({'control':control,'rank_count':len(rank['results']),'bounded_count':len(bounded['data']),'health_samples':len(latencies),'max_health_seconds':max(latencies),'rerank_loaded':True}))
'''
REMOTE=r'''
from pathlib import Path
import hashlib,json,math,subprocess,sys,urllib.request
root=Path('/opt/stacks/hermes/embedding-repair-20260926')
package=json.load(sys.stdin);code=package['probe']
canary=json.loads((root/'canary.json').read_text());cid=canary['container_id']
inspect=lambda n:json.loads(subprocess.check_output(['docker','inspect',n]))[0]
d=inspect(cid);assert d['Image']==canary['image'] and d['HostConfig']['NetworkMode']=='none'
assert d['State']['Running'] and d['RestartCount']==0
raw=subprocess.run(['docker','exec',cid,'python3','-c',code],capture_output=True,text=True,timeout=240)
if raw.returncode:
    print(json.dumps({'status':'probe_failed','exit_code':raw.returncode,'error_tail':raw.stderr[-2000:]}));raise SystemExit(1)
result=json.loads(raw.stdout)
texts=['Synthetic embedding regression sample number '+str(i) for i in range(33)]
req=urllib.request.Request('http://127.0.0.1:8766/v1/embeddings',data=json.dumps({'input':texts}).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=60) as response:baseline=json.load(response)
assert len(baseline['data'])==len(result['control']['data'])==33
worst=0.;least=1.
for a,b in zip(baseline['data'],result['control']['data']):
    assert a['index']==b['index'] and len(a['embedding'])==len(b['embedding'])==384
    av,bv=a['embedding'],b['embedding']
    worst=max(worst,max(abs(x-y) for x,y in zip(av,bv)))
    cosine=sum(x*y for x,y in zip(av,bv))/(sum(x*x for x in av)*sum(y*y for y in bv))**.5
    least=min(least,cosine)
assert worst<1e-5 and least>0.999999
d=inspect(cid);assert d['State']['Running'] and not d['State']['OOMKilled'] and d['RestartCount']==0
pid=d['State']['Pid'];cg=Path('/proc')/str(pid)/'cgroup'
relative=[l.split(':',2)[2] for l in cg.read_text().splitlines() if l.startswith('0::')][0]
memory=Path('/sys/fs/cgroup')/relative.lstrip('/')
events=dict(line.split() for line in (memory/'memory.events').read_text().splitlines())
assert int(events['oom_kill'])==0 and int(events['oom'])==0
result.pop('control')
result.update(status='passed',container_id=cid,image=d['Image'],control_vectors=33,max_absolute_difference=worst,min_cosine_similarity=least,memory_peak_bytes=int((memory/'memory.peak').read_text()),memory_current_bytes=int((memory/'memory.current').read_text()),memory_events={k:int(v) for k,v in events.items()},source_sha256='0bf31ebda3d13e2b99a68bbccf9dafa571714167a27458e435cf6b8ae17bb435')
(root/'probe.json').write_text(json.dumps(result));print(json.dumps(result))
'''
if __name__=='__main__':
    r=subprocess.run(SSH+[shlex.join(['python3','-c',REMOTE])],input=json.dumps({'probe':PROBE}),text=True,capture_output=True)
    print(r.stdout,end='')
    if r.returncode:print(r.stderr[-3000:],file=__import__('sys').stderr)
    raise SystemExit(r.returncode)
