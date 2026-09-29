"""Scoped Hermes1 backup and offline migration rehearsal; never installs code."""
from __future__ import annotations
import json
import shlex
import subprocess
import sys

REMOTE_ROOT = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/r7-deploy-20260925'
CONTAINER_ROOT = '/home/node/.hermes/benchmarks/r7-deploy-20260925'
SOURCE = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/verification'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
REHEARSE = r'''
import sys,pathlib,json,hashlib,sqlite3
raw=pathlib.Path('/manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8'
manifest=json.loads(raw); expected={}
for key in ('source_sha256','test_sha256','auxiliary_sha256'): expected.update(manifest[key])
source=pathlib.Path('/candidate'); actual={}
for path in source.rglob('*'):
    assert not path.is_symlink()
    if path.is_file(): actual[path.relative_to(source).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
assert actual==expected and len(expected)==479
sys.path.insert(0,'/candidate')
from hymem.core import db
root=pathlib.Path('/results'); report={'status':'failed','production_modified':False,'provider_calls':0}
def quote(name): return '"'+name.replace('"','""')+'"'
def digest(conn,table,columns):
    hashes=[]
    for row in conn.execute('SELECT '+','.join(map(quote,columns))+' FROM '+quote(table)):
        values=[{'blob':v.hex()} if isinstance(v,bytes) else v for v in row]
        hashes.append(hashlib.sha256(json.dumps(values,separators=(',',':'),ensure_ascii=True).encode()).digest())
    return {'count':len(hashes),'sha256':hashlib.sha256(b''.join(sorted(hashes))).hexdigest()}
conn=None
try:
    conn=db.connect(root/'rehearsal.sqlite')
    report['before_schema']=db.schema_version(conn)
    columns={r[0]:[c[1] for c in conn.execute('PRAGMA table_info('+quote(r[0])+')')]
             for r in conn.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")
             if r[0]!='schema_meta' and not r[0].startswith('sqlite_')
             and r[1] and not r[1].upper().startswith('CREATE VIRTUAL TABLE')}
    before={name:digest(conn,name,cols) for name,cols in columns.items()}
    db.initialize(conn)
    report['after_schema']=db.schema_version(conn)
    assert report['after_schema']==63
    after={name:digest(conn,name,cols) for name,cols in columns.items()}
    report['changed_existing_tables']=sorted(n for n in before if before[n]!=after[n])
    assert before==after
    report['original_tables_verified']=len(before)
    report['counts']={name:v['count'] for name,v in before.items()}
    report['integrity_ok']=[r[0] for r in conn.execute('PRAGMA integrity_check')]==['ok']
    report['foreign_key_violations']=len(conn.execute('PRAGMA foreign_key_check').fetchall())
    assert report['integrity_ok'] and report['foreign_key_violations']==0
    conn.close(); conn=db.connect(root/'rehearsal.sqlite'); db.initialize(conn)
    assert db.schema_version(conn)==63
    assert {name:digest(conn,name,cols) for name,cols in columns.items()}==before
    report['reopen_verified']=True; report['status']='passed'
except BaseException as exc:
    report['error_type']=type(exc).__name__
finally:
    if conn is not None: conn.close()
    with (root/'rehearsal.json').open('x') as stream: json.dump(report,stream,sort_keys=True)
print(json.dumps(report,sort_keys=True))
raise SystemExit(0 if report['status']=='passed' else 1)
'''

BACKUP = r'''
import pathlib,sqlite3,json,os,shutil
root=pathlib.Path(ROOT); root.mkdir(mode=0o700)
backup=root/'pre-rehearsal.sqlite'
fd=os.open(backup,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.close(fd)
source=sqlite3.connect('file:/home/node/.hermes/hymem.sqlite?mode=ro',uri=True,timeout=30)
destination=sqlite3.connect(backup)
try: source.backup(destination,pages=1024,sleep=0.1)
finally: destination.close(); source.close()
shutil.copy2(backup,root/'rehearsal.sqlite')
print(json.dumps({'backup_created':True,'bytes':backup.stat().st_size,'production_modified':False}))
'''

def main():
    if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
    command = ['docker','create','--name','hymem-r7-production-migration-rehearsal',
        '--pull','never','--network','none','--user','1000:1000','--read-only',
        '--cap-drop','ALL','--security-opt','no-new-privileges','--memory','2g',
        '--cpus','2','--pids-limit','128','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
        '--mount','type=bind,src=/opt/stacks/hermes/instance1/home/hymem-env,dst=/home/node/hymem-env,readonly',
        '--mount','type=bind,src='+SOURCE+',dst=/candidate,readonly',
        '--mount','type=bind,src='+SOURCE.rsplit('/',1)[0]+'/diag/manifest.json,dst=/manifest.json,readonly',
        '--mount','type=bind,src='+REMOTE_ROOT+',dst=/results',
        '--entrypoint','/home/node/hymem-env/bin/python3',IMAGE,'-I','-B','-c',REHEARSE]
    code = 'import subprocess,json,pathlib\n'
    code += 'backup='+repr('ROOT='+repr(CONTAINER_ROOT)+'\n'+BACKUP)+'\n'
    code += "r=subprocess.run(['docker','exec','hermes-1','/home/node/hymem-env/bin/python3','-I','-B','-c',backup],capture_output=True,text=True,timeout=120)\n"
    code += "assert r.returncode==0,'backup_failed'; result=json.loads(r.stdout)\n"
    code += 'command='+repr(command)+'\n'
    code += "cid=subprocess.check_output(command,text=True,timeout=30).strip(); result['container_id']=cid\n"
    code += "obj=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]\n"
    code += "assert obj['State']['Status']=='created' and obj['HostConfig']['NetworkMode']=='none'\n"
    code += "assert subprocess.check_output(['docker','start',cid],text=True).strip()==cid\n"
    code += "with (pathlib.Path("+repr(REMOTE_ROOT)+")/'launch.json').open('x') as f: json.dump(result,f,sort_keys=True)\n"
    code += 'print(json.dumps(result,sort_keys=True))\n'
    try:
        r=subprocess.run(['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=10',
            '-o','ConnectionAttempts=1','afrodite','python3 -I -B -c '+shlex.quote(code)],
            capture_output=True,timeout=180)
    except subprocess.TimeoutExpired:
        raise SystemExit('rehearsal_setup_timeout_inspect_before_retry') from None
    if r.returncode: raise SystemExit('rehearsal_setup_failed_inspect_before_retry')
    print(r.stdout.decode())

if __name__=='__main__': main()
