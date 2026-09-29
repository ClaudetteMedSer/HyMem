"""Stage the reviewed code-only full suite on Afrodite; launch is explicit."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys

UPLOAD = Path('/private/tmp/hymem-episode-shadow-pytest-v2-root-20260926')
ROOT = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/episode-shadow-pytest-v2'
CONTROLLER = 'claim_conflict_episode_shadow_pytest_v2.py'
PIN = 'b20c120ef5cdc0ccfaf85a8a48d05bc79fd9a563e79487064c7f1726e392c754'
SSH = ['ssh', '-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ConnectionAttempts=1', 'afrodite']
REMOTE = r'''
from pathlib import Path
import hashlib,json,os,subprocess,sys
os.umask(0o077)
data=json.load(sys.stdin);root=Path(data['root'])
assert str(root)=='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/episode-shadow-pytest-v2'
assert os.geteuid()==1000 and not root.exists() and not root.is_symlink()
files=data['files'];controller='claim_conflict_episode_shadow_pytest_v2.py'
assert hashlib.sha256(files[controller].encode()).hexdigest()=='b20c120ef5cdc0ccfaf85a8a48d05bc79fd9a563e79487064c7f1726e392c754'
overlay=json.loads(files['test-overlay.json'])
assert len(overlay)==21 and hashlib.sha256(json.dumps(overlay,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='acbb892d530f2426c8a81dc527eded867542d6ebb7e0858f8bb8e418163a02ab'
assert set(files)=={controller,'test-overlay.json','reviewed-candidate.json'}|{'overlay/'+name for name in overlay}
manifest=json.loads(files['reviewed-candidate.json'])
assert len(manifest)==481 and hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576'
for name,pin in overlay.items():
    p=Path(name);assert name==p.as_posix() and not p.is_absolute() and '..' not in p.parts
    assert hashlib.sha256(files['overlay/'+name].encode()).hexdigest()==pin
root.mkdir(mode=0o700)
for name,value in files.items():
    target=root/name;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:stream.write(value.encode());stream.flush();os.fsync(stream.fileno())
run=subprocess.run(['python3','-I','-B',str(root/controller),'remote-install'],capture_output=True,text=True,timeout=180)
result=json.loads(run.stdout);assert result.get('status')=='installed_not_launched'
print(json.dumps(result,sort_keys=True))
'''


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'install':
        files = {}
        for path in UPLOAD.rglob('*'):
            assert not path.is_symlink()
            if path.is_file():
                files[path.relative_to(UPLOAD).as_posix()] = path.read_text()
        assert hashlib.sha256(files[CONTROLLER].encode()).hexdigest() == PIN
        result = subprocess.run(SSH + [shlex.join(['python3', '-I', '-B', '-c', REMOTE])],
                                input=json.dumps({'root': ROOT, 'files': files}),
                                text=True, capture_output=True)
    elif action in ('launch', 'status'):
        code = ('import hashlib,subprocess; from pathlib import Path; '
                f'p=Path({(ROOT + "/" + CONTROLLER)!r}); '
                f'assert hashlib.sha256(p.read_bytes()).hexdigest()=={PIN!r}; '
                f'raise SystemExit(subprocess.call(["python3","-I","-B",str(p),"remote-{action}"]))')
        result = subprocess.run(SSH + [shlex.join(['python3', '-I', '-B', '-c', code])],
                                text=True, capture_output=True)
    else:
        raise SystemExit('invalid_action')
    print(result.stdout, end='')
    if result.returncode:
        print('{"status":"remote_suite_action_failed"}')
    raise SystemExit(result.returncode)
