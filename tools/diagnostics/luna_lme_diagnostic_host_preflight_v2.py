"""Upload the accepted source bundle and run only the Luna LME host preflight.

No inference, launch receipt, run marker, service, credential copy, or dataset
copy is performed. The remote result is finite metadata; raw runner output stays
on the host only when a preflight fails.
"""
from __future__ import annotations

import argparse
import ast
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tarfile


SOURCE = Path('/private/tmp/hymem-lme-diagnostic-offline-assembly-v3')
RUNNER_REL = 'code/tools/diagnostics/luna_lme_diagnostic_v2.py'
RUNNER_SHA = '5a9b29456c8fcc96fd9df7d1532a457662eccba63e5ef39f737121fe349565d9'
MAP_SHA = '228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf'
DATASET = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json'
BINARY = '/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex'
SSH_BASE = ['ssh', '-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ConnectionAttempts=1']


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def source_manifest(source: Path = SOURCE) -> dict[str, str]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError('source_missing')
    runner_path = source / RUNNER_REL
    map_path = source / 'source-map.json'
    if not regular(runner_path) or sha(runner_path) != RUNNER_SHA:
        raise ValueError('runner_drift')
    if not regular(map_path) or sha(map_path) != MAP_SHA:
        raise ValueError('source_map_drift')
    tree = ast.parse(runner_path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('ACCEPTED_FILES','ACCEPTED_MAP_SHA256','PINS','DIAGNOSTIC_HELPER_SHA256'):
                constants[name]=ast.literal_eval(node.value)
    if set(constants)!=set(('ACCEPTED_FILES','ACCEPTED_MAP_SHA256','PINS','DIAGNOSTIC_HELPER_SHA256')):
        raise ValueError('runner_invalid')
    stamp = json.loads(map_path.read_text())
    entries = stamp.get('source_sha256')
    if type(entries) is not dict or len(entries) != constants['ACCEPTED_FILES']:
        raise ValueError('source_map_shape')
    encoded = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(encoded).hexdigest() != constants['ACCEPTED_MAP_SHA256']:
        raise ValueError('source_map_pin')
    manifest = {'source-map.json': MAP_SHA, RUNNER_REL: RUNNER_SHA}
    for relative, digest in entries.items():
        if (type(relative) is not str or not relative or Path(relative).is_absolute()
                or '..' in Path(relative).parts or type(digest) is not str
                or re.fullmatch(r'[0-9a-f]{64}', digest) is None):
            raise ValueError('candidate_map_entry_invalid')
        manifest['candidate/' + relative] = digest
    for relative, digest in constants['PINS'].items():
        manifest['code/' + relative] = digest
    manifest['code/benchmarks/lme_diagnostic.py'] = constants['DIAGNOSTIC_HELPER_SHA256']
    if len(manifest) != constants['ACCEPTED_FILES'] + 9 + 1:
        raise ValueError('file_count_invalid')
    expected = set(manifest)
    actual = {str(path.relative_to(source)) for path in source.rglob('*')
              if not path.is_dir() or path.is_symlink()}
    if actual != expected:
        raise ValueError('source_file_set_invalid')
    for relative, digest in manifest.items():
        path = source / relative
        if not regular(path) or path.is_symlink() or sha(path) != digest:
            raise ValueError('source_file_drift')
    return manifest


def archive_bytes(source: Path = SOURCE) -> bytes:
    manifest = source_manifest(source)
    output = BytesIO()
    with tarfile.open(fileobj=output, mode='w') as archive:
        data = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        info = tarfile.TarInfo('manifest.json')
        info.size = len(data)
        info.mode = 0o600
        archive.addfile(info, BytesIO(data))
        for relative in sorted(manifest):
            path = source / relative
            info = tarfile.TarInfo(relative)
            info.size = path.stat().st_size
            info.mode = 0o600
            with path.open('rb') as handle:
                archive.addfile(info, handle)
    return output.getvalue()


REMOTE = r'''
import hashlib,io,json,os,re,stat,subprocess,sys,tarfile,tempfile
from pathlib import Path

HOST_ROOT=Path('/home/atta')
DATASET=Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json')
BINARY=Path('/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex')
RUNNER='code/tools/diagnostics/luna_lme_diagnostic_v2.py'
RUNNER_SHA='5a9b29456c8fcc96fd9df7d1532a457662eccba63e5ef39f737121fe349565d9'
MAP_SHA='228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf'
DATASET_SHA='d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
MAP_PIN='9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae'
MAX_BYTES=64*1024*1024

def need(ok,code):
    if not ok: raise RuntimeError(code)
def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def regular(path):
    try:return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:return False

need(os.getuid()==1000 and HOST_ROOT.is_dir() and not HOST_ROOT.is_symlink(),'host_identity_invalid')
payload=sys.stdin.buffer.read(MAX_BYTES+1)
need(len(payload)<=MAX_BYTES,'archive_too_large')
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
    members=archive.getmembers()
    need(members and members[0].name=='manifest.json' and members[0].isfile(),
         'manifest_missing')
    manifest_data=archive.extractfile(members[0]).read()
    manifest=json.loads(manifest_data)
    need(type(manifest) is dict and len(manifest)==524 and
         manifest.get(RUNNER)==RUNNER_SHA and manifest.get('source-map.json')==MAP_SHA,
         'manifest_invalid')
    for name,digest in manifest.items():
        parts=Path(name).parts
        need(type(name) is str and ((parts and parts[0] in ('candidate','code')) or
             name=='source-map.json'),'manifest_path_invalid')
        need(not Path(name).is_absolute() and '..' not in parts and
             type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest) is not None,
             'manifest_entry_invalid')
    names=[member.name for member in members[1:]]
    need(len(names)==len(manifest) and set(names)==set(manifest) and len(names)==len(set(names)),
         'archive_file_set_invalid')
    need(sum(name.startswith('candidate/') for name in names)==514 and
         sum(name.startswith('code/') for name in names)==9,
         'archive_file_set_invalid')
    need(all(member.isfile() and not member.issym() and not member.islnk()
             for member in members),'archive_member_invalid')
    # Verify every byte before creating the one private output directory.
    content={}
    for member in members[1:]:
        raw=archive.extractfile(member).read()
        need(sha_bytes(raw)==manifest[member.name],'archive_hash_mismatch')
        content[member.name]=raw
    stamp=json.loads(content['source-map.json'])
    entries=stamp.get('source_sha256')
    need(type(entries) is dict and len(entries)==514 and
         sha_bytes(json.dumps(entries,sort_keys=True,separators=(',',':')).encode())==MAP_PIN and
         all(manifest.get('candidate/'+name)==digest for name,digest in entries.items()),
         'candidate_map_invalid')
need(regular(DATASET) and sha_file(DATASET)==DATASET_SHA,'dataset_identity_invalid')
need(regular(BINARY),'binary_missing')
binary_sha=sha_file(BINARY)
root=Path(tempfile.mkdtemp(prefix='.hymem-lme-diagnostic-preflight-',dir=HOST_ROOT))
os.chmod(root,0o700)
for name,raw in content.items():
    target=root/name
    target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    with target.open('xb') as handle:handle.write(raw)
    os.chmod(target,0o600)
    need(sha_file(target)==manifest[name],'copy_hash_mismatch')
command=['/usr/bin/python3','-I','-B',str(root/RUNNER),
    '--root',str(root),'--inventory',str(root/'source-map.json'),
    '--inventory-sha256',MAP_SHA,'--dataset',str(DATASET),
    '--binary',str(BINARY),'--binary-sha256',binary_sha,'--preflight-only']
try:
    result=subprocess.run(command,capture_output=True,timeout=120)
except subprocess.TimeoutExpired:
    raise RuntimeError('preflight_timeout') from None
if result.returncode:
    # Private diagnostic details remain in the private preflight directory.
    path=root/'preflight-stderr.private'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as handle:handle.write(result.stderr[:65536])
    known={'input_identity_invalid','transport_or_helper_drift','diagnostic_helper_drift',
           'candidate_source_drift','cached_source_mismatch','import_origin_invalid',
           'inventory_invalid','dataset_invalid','root_invalid','module_load_invalid'}
    matches=re.findall(rb'ValueError: ([a-z_]+)',result.stderr)
    reason=matches[-1].decode() if matches else 'unclassified'
    if reason in known: raise RuntimeError('preflight_failed_'+reason)
    modules=re.findall(rb"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'",result.stderr)
    missing=modules[-1].decode() if modules else ''
    if missing in ('requests','numpy','scipy','sqlite_vec'):
        raise RuntimeError('preflight_missing_module_'+missing)
    raise RuntimeError('preflight_failed')
parsed=json.loads(result.stdout)
need(type(parsed) is dict and parsed.get('preflight_verified') is True and
     parsed.get('model_calls')==0 and parsed.get('selected_count')==4,
     'preflight_result_invalid')
print(json.dumps({'schema':'luna-lme-diagnostic-host-preflight-v2',
    'root':str(root),'candidate_files':514,'code_files':9,
    'inventory_sha256':MAP_SHA,'binary_sha256':binary_sha,
    'dataset_sha256':DATASET_SHA,'preflight_verified':True,
    'selected_count':4,'model_calls':0},sort_keys=True))
'''


SAFE_ERRORS = frozenset({
    'host_identity_invalid','archive_too_large','manifest_missing','manifest_invalid',
    'manifest_path_invalid','manifest_entry_invalid','archive_file_set_invalid',
    'archive_member_invalid','archive_hash_mismatch','candidate_map_invalid',
    'dataset_identity_invalid','binary_missing','copy_hash_mismatch',
    'preflight_timeout','preflight_failed','preflight_result_invalid',
    'preflight_failed_input_identity_invalid',
    'preflight_failed_transport_or_helper_drift',
    'preflight_failed_diagnostic_helper_drift',
    'preflight_failed_candidate_source_drift',
    'preflight_failed_cached_source_mismatch',
    'preflight_failed_import_origin_invalid',
    'preflight_failed_inventory_invalid',
    'preflight_failed_dataset_invalid',
    'preflight_failed_root_invalid',
    'preflight_failed_module_load_invalid',
    'preflight_missing_module_requests','preflight_missing_module_numpy',
    'preflight_missing_module_scipy','preflight_missing_module_sqlite_vec',
})


def wrapped_remote() -> str:
    import textwrap
    return (
        'import json,sys\ntry:\n' + textwrap.indent(REMOTE, '    ')
        + '\nexcept RuntimeError as exc:\n'
        + f'    code=str(exc)\n    safe={repr(sorted(SAFE_ERRORS))}\n'
        + "    output={'error':code if code in safe else 'unclassified'}\n"
          "    if 'root' in globals() and str(root).startswith('/home/atta/.hymem-lme-diagnostic-preflight-'):\n"
          "        output['root']=str(root)\n"
          "    print(json.dumps(output))\n"
          '    sys.exit(1)\n'
        + 'except BaseException as exc:\n'
        + "    kind=type(exc).__name__\n"
          "    if kind not in ('ValueError','OSError','KeyError','TypeError','JSONDecodeError'): kind='Exception'\n"
          "    output={'error':'unclassified','type':kind}\n"
          "    if 'root' in globals() and str(root).startswith('/home/atta/.hymem-lme-diagnostic-preflight-'):\n"
          "        output['root']=str(root)\n"
          "    print(json.dumps(output))\n"
          '    sys.exit(1)\n'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='Reviewed SSH host alias for Luna')
    parser.add_argument('--source', required=True, help='Reviewed absolute v2 source-bundle root')
    args = parser.parse_args(argv)
    if re.fullmatch(r'[A-Za-z0-9_.-]+', args.host) is None:
        print(json.dumps({'error':'host_alias_invalid'}))
        return 1
    try:
        source=Path(args.source)
        if not source.is_absolute(): raise ValueError('source_missing')
        payload = archive_bytes(source)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        code = str(exc)
        print(json.dumps({'error':code if code in {
            'source_missing','runner_drift','source_map_drift','runner_invalid',
            'source_map_shape','source_map_pin','candidate_map_entry_invalid',
            'file_count_invalid','source_file_set_invalid','source_file_drift'}
            else 'source_invalid'}, sort_keys=True))
        return 1
    command = SSH_BASE + [args.host, shlex.join(['/usr/bin/python3','-I','-B',
        '-c',wrapped_remote()])]
    try:
        result = subprocess.run(command,input=payload,capture_output=True,timeout=190)
    except subprocess.TimeoutExpired:
        print(json.dumps({'error':'ssh_timeout'}))
        return 1
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(json.dumps({'error':'remote_receipt_invalid'}))
        return 1
    if result.returncode:
        code = value.get('error') if isinstance(value,dict) else None
        output={'error':code if code in SAFE_ERRORS else 'remote_failed'}
        if isinstance(value,dict):
            root=value.get('root')
            if isinstance(root,str) and re.fullmatch(
                    r'/home/atta/\.hymem-lme-diagnostic-preflight-[A-Za-z0-9_-]+',root):
                output['root']=root
        print(json.dumps(output,sort_keys=True))
        return 1
    if not isinstance(value,dict) or value.get('preflight_verified') is not True:
        print(json.dumps({'error':'remote_receipt_invalid'}))
        return 1
    print(json.dumps(value,sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
