"""Read-only bounded post-recovery metadata; run on Afrodite through stdin."""
import json
import subprocess

CODE = r'''
import hashlib,json,os,time,urllib.request
from pathlib import Path
stage=Path('/home/node/.hermes/repairs/honcho-sqlite-jd70_3xs')
out={'receipts':{}}
for name in ('install.json','recover-intent.json','recover.json','recover-failed.json',
             'recovery-source-intent.json','recovery-source-launch.json',
             'recovery-source-result.json','recovery-source-failed.json'):
    path=stage/name
    if path.is_file():
        value=json.loads(path.read_text())
        allowed={'action','phase','original_stopped','new_pid','new_identity',
                 'port_owned_by_new','term_escalated','cmdline_sha256','identity',
                 'pid','health_ok','argv_sha256'}
        out['receipts'][name]={key:val for key,val in value.items() if key in allowed}
intent=out['receipts'].get('recover-intent.json',{})
expected=intent.get('cmdline_sha256')
out['matching_processes']=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        raw=(proc/'cmdline').read_bytes()
        if not expected or hashlib.sha256(raw).hexdigest()!=expected: continue
        fields=(proc/'stat').read_text().rpartition(') ')[2].split()
        out['matching_processes'].append({'pid':int(proc.name),'state':fields[0],
            'start_ticks':int(fields[19]),'exe':os.readlink(proc/'exe'),
            'cwd':os.readlink(proc/'cwd')})
    except (FileNotFoundError,PermissionError,ProcessLookupError): pass
log=stage/'recovery-source-launch.private.log'
out['restart_log_exists']=log.exists()
out['restart_log_bytes']=log.stat().st_size if log.exists() else None
out['listeners']=[]
for table in ('/proc/net/tcp','/proc/net/tcp6'):
    for line in Path(table).read_text().splitlines()[1:]:
        fields=line.split()
        if fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16)==8765:
            inode=fields[9];owners=[]
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit(): continue
                try:
                    for fd in (proc/'fd').iterdir():
                        try:
                            if os.readlink(fd)=='socket:['+inode+']': owners.append(int(proc.name));break
                        except (FileNotFoundError,PermissionError,ProcessLookupError): pass
                except (FileNotFoundError,PermissionError,ProcessLookupError): pass
            out['listeners'].append({'inode':inode,'owners':owners})
start=time.monotonic()
try:
    with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=3) as response:
        out['health_status']=response.status
        response.read(4096)
except Exception as exc:
    out['health_status']=None
    out['health_error_type']=type(exc).__name__
out['health_seconds']=round(time.monotonic()-start,3)
print(json.dumps(out,sort_keys=True))
'''
result = subprocess.run(['docker','exec','-i','-u','node','hermes-1',
                         '/usr/bin/python3.11','-B','-'], input=CODE,
                        capture_output=True,text=True,timeout=15)
if result.returncode:
    print(json.dumps({'error':'metadata_read_failed','returncode':result.returncode}))
else:
    print(json.dumps(json.loads(result.stdout),sort_keys=True))
