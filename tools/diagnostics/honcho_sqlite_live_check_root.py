"""Read-only HTTP/SQLite recovery verification; private data never leaves host."""
import json
import subprocess

CODE = r'''
import concurrent.futures,json,sqlite3,time,urllib.request,urllib.parse
db=sqlite3.connect('file:/home/node/.hermes/hymem.sqlite?mode=ro',uri=True,timeout=5)
try:
    quick=db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    fk_count=len(db.execute('PRAGMA foreign_key_check').fetchall())
    row=db.execute("SELECT id,source_workspace_id FROM sessions WHERE source_workspace_id IS NOT NULL ORDER BY started_at DESC LIMIT 1").fetchone()
    dreams=[dict(zip(('id','started_at','ended_at','chunks_processed','provider_attempts','has_error'), item))
        for item in db.execute('SELECT id,started_at,ended_at,chunks_processed,chunk_extraction_provider_attempts,error IS NOT NULL FROM dream_runs ORDER BY id DESC LIMIT 3')]
finally: db.close()
def request(path,payload=None):
    start=time.monotonic()
    req=urllib.request.Request('http://127.0.0.1:8765'+path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=8) as response:
            raw=response.read(2_000_000)
            data=json.loads(raw)
            result={'ok':response.status==200,'seconds':round(time.monotonic()-start,3)}
            if path=='/dream-status':
                result['state']={k:v for k,v in data.items() if (
                    k.startswith(('pending_','malformed_','quarantined_','terminal_loss_'))
                    and type(v) is int) or (k in ('in_progress','coverage_integrity_failures','summary_healthy')
                    and type(v) in (int,bool))}
            elif type(data) is list: result['count']=len(data)
            return result
    except Exception as exc:
        return {'ok':False,'error_type':type(exc).__name__,
                'seconds':round(time.monotonic()-start,3)}
out={'quick_check_ok':quick,'foreign_key_violations':fk_count,
     'authorized_session_available':row is not None,'dream_runs':dreams,'requests':[]}
if row:
    session,workspace=row
    path='/v3/workspaces/'+urllib.parse.quote(workspace,safe='')+'/sessions/'+urllib.parse.quote(session,safe='')+'/search'
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for _ in range(5):
            pending=[pool.submit(request,path,{'query':'HyMem','limit':3}) for _ in range(4)]
            pending.append(pool.submit(request,'/health'))
            out['requests'].extend(f.result() for f in pending)
            time.sleep(.3)
    out['elapsed_seconds']=round(time.monotonic()-start,3)
out['dream_status']=request('/dream-status')
out['health']=request('/health')
out['all_http_ok']=bool(out['requests']) and all(v['ok'] for v in out['requests']) and out['dream_status']['ok'] and out['health']['ok']
print(json.dumps(out,sort_keys=True))
'''
try:
    result=subprocess.run(['docker','exec','-i','-u','node','hermes-1',
        '/usr/bin/python3.11','-B','-'],input=CODE,capture_output=True,text=True,timeout=55)
except subprocess.TimeoutExpired:
    print(json.dumps({'error':'verification_timeout'}))
else:
    if result.returncode:
        print(json.dumps({'error':'verification_failed','returncode':result.returncode}))
    else:
        print(json.dumps(json.loads(result.stdout),sort_keys=True))
