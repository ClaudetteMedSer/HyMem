"""Install a pinned new regression package; never start a container or read a key."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/r6-lme-failed-question-v1'
SOURCE_PIN = 'bd8d0f3a8fb40bd6b77e7ca6579c8e5e8ee78733bea2bb22df71bc4b2c12eaa2'

INSTALL = r'''
import hashlib,io,json,os,pathlib,stat,sys,tarfile
def need(value):
    if not value: raise RuntimeError('r6_install_contract')
root=pathlib.Path(C['root'])
raw=sys.stdin.buffer.read(64*1024*1024+1)
need(len(raw)<=64*1024*1024 and hashlib.sha256(raw).hexdigest()==C['archive_pin'])
files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
    for item in archive.getmembers():
        need(item.isfile() and not item.pax_headers and item.name not in files)
        p=pathlib.PurePosixPath(item.name)
        need(p.as_posix()==item.name and not p.is_absolute() and '..' not in p.parts
             and all(not part.startswith('.') for part in p.parts) and item.size<=32*1024*1024)
        need(item.name in C['files'] and (p.parts[0] in ('bundle','candidate') or item.name=='input-manifest.json'))
        files[item.name]=archive.extractfile(item).read()
        need(sum(map(len,files.values()))<=128*1024*1024)
need(set(files)==set(C['files']))
need(all(hashlib.sha256(files[n]).hexdigest()==pin for n,pin in C['files'].items()))
need(hashlib.sha256(files['bundle/manifest.json']).hexdigest()==C['manifest_pin'])
need(hashlib.sha256(files['input-manifest.json']).hexdigest()==C['source_pin'])
m=json.loads(files['bundle/manifest.json']); source=json.loads(files['input-manifest.json'])
need(m['schema']=='r6-lme-failed-question-preparation-v1' and m['remote_run_root']==str(root)
     and m['remote_source']==str(root/'candidate') and type(m['sample']) is int and m['sample']==1
     and type(m['seed']) is int and m['seed']==0 and m['question_ids']==['gpt4_483dd43c']
     and m['source_indices']==[329] and m['source_sha256']==source['source_sha256']
     and m['approved_source_manifest_sha256']==C['source_pin'] and len(m['source_sha256'])==231)
expected={'input-manifest.json','bundle/manifest.json'}
expected.update('candidate/'+p for p in m['source_sha256'])
expected.update('bundle/'+p for p in m['helper_sha256'])
need(set(files)==expected)
need(all(C['files']['candidate/'+p]==pin for p,pin in m['source_sha256'].items()))
need(all(C['files']['bundle/'+p]==pin for p,pin in m['helper_sha256'].items()))
need(os.geteuid()==1000 and root.parent.resolve()==root.parent and root.parent.is_dir())
root.mkdir(mode=0o700)
for name in ('bundle','candidate','home','preflight-results','live-results','gates'):
    (root/name).mkdir(mode=0o700)
for name,body in sorted(files.items()):
    (root/name).parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as out:
        out.write(body);out.flush();os.fsync(out.fileno())
need(all(hashlib.sha256((root/n).read_bytes()).hexdigest()==pin for n,pin in C['files'].items()))
receipt={'status':'installed_not_started','manifest_sha256':C['manifest_pin'],
 'source_manifest_sha256':C['source_pin'],'archive_sha256':C['archive_pin'],
 'source_files':231,'provider_calls':0,'production_changes':False}
with (root/'install.json').open('x') as out:json.dump(receipt,out,sort_keys=True)
print(json.dumps(receipt,sort_keys=True))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--seal-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    raw = args.seal.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.seal_sha256:
        raise RuntimeError('r6_seal_pin')
    seal = json.loads(raw)
    if seal['schema'] != 'r6-lme-failed-question-seal-v1' or seal['source_manifest_sha256'] != SOURCE_PIN:
        raise RuntimeError('r6_seal_identity')
    archive = Path(seal['archive']).read_bytes()
    if hashlib.sha256(archive).hexdigest() != seal['archive_sha256']:
        raise RuntimeError('r6_archive_pin')
    config = {'root': ROOT, 'manifest_pin': seal['manifest_sha256'], 'source_pin': SOURCE_PIN,
              'archive_pin': seal['archive_sha256'], 'files': seal['files_sha256']}
    code = 'C=' + repr(config) + '\n' + INSTALL
    result = subprocess.run(['ssh', 'afrodite', 'python3 -I -B -c ' + shlex.quote(code)],
                            input=archive, capture_output=True, timeout=600)
    if result.returncode:
        print(json.dumps({'status': 'r6_install_failed', 'returncode': result.returncode}))
        raise SystemExit(1)
    receipt = json.loads(result.stdout)
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
