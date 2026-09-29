"""Upload one reviewed helper archive into a fresh isolated Afrodite directory.

No test is launched, credential read or application source modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/sample8-stock-v1'
INSTALL=r'''
import hashlib,io,json,os,pathlib,sys,tarfile
root=pathlib.Path(C['root'])
raw=sys.stdin.buffer.read(2097153)
assert len(raw)<=2097152 and hashlib.sha256(raw).hexdigest()==C['archive_pin']
files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
    for item in archive.getmembers():
        assert item.isfile() and not item.pax_headers and item.name not in files
        p=pathlib.PurePosixPath(item.name)
        assert p.as_posix()==item.name and not p.is_absolute() and '..' not in p.parts
        assert len(p.parts)==2 and p.parts[0]=='bundle' and item.size<=1048576
        files[item.name]=archive.extractfile(item).read()
assert set(files)==set(C['files'])
assert all(hashlib.sha256(files[n]).hexdigest()==pin for n,pin in C['files'].items())
assert hashlib.sha256(files['bundle/manifest.json']).hexdigest()==C['manifest_pin']
m=json.loads(files['bundle/manifest.json'])
assert m['remote_run_root']==str(root) and m['sample']==8 and m['seed']==0
assert os.geteuid()==1000 and root.parent.resolve()==root.parent and root.parent.is_dir()
root.mkdir(mode=0o700)
for name in ('bundle','home','preflight-results','live-results'):
    (root/name).mkdir(mode=0o700)
for name,body in sorted(files.items()):
    fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as out: out.write(body)
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==pin for n,pin in C['files'].items())
receipt={'status':'installed_not_started','manifest_sha256':C['manifest_pin'],
 'archive_sha256':C['archive_pin'],'bundle_files':len(files),'provider_calls':0,'production_changes':False}
with (root/'install.json').open('x') as out: json.dump(receipt,out,sort_keys=True)
print(json.dumps(receipt,sort_keys=True))
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seal',type=Path,required=True)
    parser.add_argument('--seal-sha256',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    raw=args.seal.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.seal_sha256: raise RuntimeError('seal_pin')
    receipt=json.loads(raw); archive=Path(receipt['archive']).read_bytes()
    if hashlib.sha256(archive).hexdigest()!=receipt['archive_sha256']: raise RuntimeError('archive_pin')
    config={'root':ROOT,'manifest_pin':receipt['manifest_sha256'],
            'archive_pin':receipt['archive_sha256'],'files':receipt['files_sha256']}
    code='import json\nC=json.loads('+repr(json.dumps(config))+')\n'+INSTALL
    result=subprocess.run(['ssh','afrodite','python3 -I -B -c '+shlex.quote(code)],
        input=archive,capture_output=True,timeout=120)
    if result.returncode:
        print(json.dumps({'status':'sample8_install_failed','returncode':result.returncode}))
        raise SystemExit(1)
    value=json.loads(result.stdout)
    with args.receipt.open('x') as out: json.dump(value,out,sort_keys=True,indent=2)
    print(json.dumps(value,sort_keys=True))

if __name__=='__main__': main()
