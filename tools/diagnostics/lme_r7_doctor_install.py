"""Install the reviewed diagnostic-only probe fix; never touch benchmark source."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

OLD = '5aa54d3de99a0ed9bd1c3e7cebef6f2d7472f0482783b87a46aed8a2536193bc'
NEW = '2ac786c6590aa7fbece726e6380981906d514fefe9f664a3ca4f7b1d1de7c15b'
REMOTE = r'''
import base64,hashlib,json,os,pathlib,stat,sys,tempfile
root=pathlib.Path('/opt/stacks/hermes/instance1/home')
target=root/'HyMem/hymem/doctor.py'
stage=root/'.hermes/benchmarks/r7-deploy-20260925'
assert target.resolve()==target and stage.resolve()==stage and stage.is_dir()
info=target.stat();before=target.read_bytes()
assert hashlib.sha256(before).hexdigest()==OLD
assert info.st_uid==os.geteuid() and info.st_gid in os.getgroups()
body=base64.b64decode(sys.stdin.buffer.read(),validate=True)
assert hashlib.sha256(body).hexdigest()==NEW
assert not (stage/'doctor-probe.json').exists()
fd=os.open(stage/'doctor-probe-before.py',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as out:out.write(before);out.flush();os.fsync(out.fileno())
fd,name=tempfile.mkstemp(prefix='.doctor-probe-',dir=target.parent)
with os.fdopen(fd,'wb') as out:
    out.write(body);out.flush();os.fsync(out.fileno())
    os.fchmod(out.fileno(),stat.S_IMODE(info.st_mode))
    os.fchown(out.fileno(),info.st_uid,info.st_gid)
assert hashlib.sha256(target.read_bytes()).hexdigest()==OLD
os.replace(name,target)
assert hashlib.sha256(target.read_bytes()).hexdigest()==NEW
value={'status':'passed','path':'hymem/doctor.py','before_sha256':OLD,
       'after_sha256':NEW,'benchmark_source_changed':False,
       'restart_required':False}
fd=os.open(stage/'doctor-probe.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as out:json.dump(value,out,sort_keys=True)
print(json.dumps(value,sort_keys=True))
'''

if __name__ == '__main__':
    body=Path('/private/tmp/hymem-r7-doctor-fix-20260925/doctor.py').read_bytes()
    assert hashlib.sha256(body).hexdigest()==NEW
    code='OLD='+repr(OLD)+'\nNEW='+repr(NEW)+'\n'+REMOTE
    result=subprocess.run(['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=10',
        '-o','ConnectionAttempts=1','afrodite','python3 -I -B -c '+shlex.quote(code)],
        input=base64.b64encode(body),capture_output=True,timeout=90)
    if result.returncode:raise SystemExit('doctor_patch_failed_inspect_remote')
    value=json.loads(result.stdout)
    assert value['status']=='passed' and value['after_sha256']==NEW
    print(json.dumps(value,sort_keys=True))
