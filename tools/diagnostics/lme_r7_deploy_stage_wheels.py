"""Stage two pinned public build wheels on Afrodite; no installation or restart."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

FILES = {
    'setuptools-80.9.0-py3-none-any.whl': '062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922',
    'wheel-0.45.1-py3-none-any.whl': '708e7481cc80179af0e556bbf0cc00b8444c7321e2700b8d8580231d13017248',
}
REMOTE = r'''
import base64,hashlib,json,os,pathlib,sys
root=pathlib.Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/r7-deploy-20260925/build-wheels')
assert root.parent.resolve()==root.parent and root.parent.is_dir()
raw=sys.stdin.buffer.read(4*1024*1024+1); assert len(raw)<=4*1024*1024
packet=json.loads(raw); assert set(packet)==set(PINS)
files={name:base64.b64decode(value,validate=True) for name,value in packet.items()}
assert all(hashlib.sha256(raw).hexdigest()==PINS[name] for name,raw in files.items())
root.mkdir(mode=0o700)
for name,raw in files.items():
    fd=os.open(root/name,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as out: out.write(raw)
print(json.dumps({'status':'build_wheels_staged_not_installed','sha256':PINS},sort_keys=True))
'''

def main():
    packet={}
    for name,pin in FILES.items():
        raw=(Path('/private/tmp/hymem-r7-deploy-build-wheels-20260925')/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pin
        packet[name]=base64.b64encode(raw).decode('ascii')
    code='PINS='+repr(FILES)+'\n'+REMOTE
    try:
        result=subprocess.run(['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=10',
            '-o','ConnectionAttempts=1','afrodite','python3 -I -B -c '+shlex.quote(code)],
            input=json.dumps(packet).encode(),capture_output=True,timeout=120)
    except subprocess.TimeoutExpired:
        raise SystemExit('wheel_stage_timeout_inspect_before_retry') from None
    if result.returncode: raise SystemExit('wheel_stage_failed_inspect_before_retry')
    print(result.stdout.decode())

if __name__=='__main__': main()
