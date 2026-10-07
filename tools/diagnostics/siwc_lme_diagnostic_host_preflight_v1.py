"""Stage and verify SIWC LME source on Afrodite without inference or dataset egress."""
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
import textwrap


RUNNER_REL = "code/tools/diagnostics/siwc_lme_diagnostic_v1.py"
RUNNER_SHA = "9366f40cf218581b51b877fd230e530c1e699cca0b76036e4304886c08c698f3"
MAP_SHA = "1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd"
DATASET = "/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json"
DATASET_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
RUNTIME = "/home/atta/.hymem-siwc-runtime-v1/bin/python"
SSH_BASE = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def runner_constants(raw: str) -> dict:
    wanted = {"ACCEPTED_FILES", "ACCEPTED_MAP_SHA256", "PINS", "SIWC_PINS",
              "DIAGNOSTIC_HELPER_SHA256"}
    values = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                values[name] = ast.literal_eval(node.value)
    if set(values) != wanted:
        raise ValueError("runner_constants_invalid")
    return values


def source_manifest(source: Path) -> dict[str, str]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source_missing")
    runner = source / RUNNER_REL
    map_path = source / "source-map.json"
    if not regular(runner) or sha(runner) != RUNNER_SHA or not regular(map_path) or sha(map_path) != MAP_SHA:
        raise ValueError("source_identity_invalid")
    constants = runner_constants(runner.read_text())
    stamp = json.loads(map_path.read_text())
    entries = stamp.get("source_sha256") if type(stamp) is dict else None
    if type(entries) is not dict or len(entries) != constants["ACCEPTED_FILES"]:
        raise ValueError("source_map_shape_invalid")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != constants["ACCEPTED_MAP_SHA256"]:
        raise ValueError("source_map_pin_invalid")
    manifest = {"source-map.json": MAP_SHA, RUNNER_REL: RUNNER_SHA}
    for relative, digest in entries.items():
        if (type(relative) is not str or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or type(digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("candidate_map_entry_invalid")
        manifest["candidate/" + relative] = digest
    for relative, digest in {**constants["PINS"], **constants["SIWC_PINS"]}.items():
        manifest["code/" + relative] = digest
    manifest["code/benchmarks/lme_diagnostic.py"] = constants["DIAGNOSTIC_HELPER_SHA256"]
    if len(manifest) != constants["ACCEPTED_FILES"] + len(constants["PINS"]) + len(constants["SIWC_PINS"]) + 3:
        raise ValueError("source_count_invalid")
    actual = {str(path.relative_to(source)) for path in source.rglob("*")
              if not path.is_dir() or path.is_symlink()}
    if actual != set(manifest):
        raise ValueError("source_file_set_invalid")
    for relative, digest in manifest.items():
        path = source / relative
        if not regular(path) or sha(path) != digest:
            raise ValueError("source_file_drift")
    return manifest


def archive_bytes(source: Path) -> bytes:
    manifest = source_manifest(source)
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size, info.mode = len(data), 0o600
        archive.addfile(info, BytesIO(data))
        for relative in sorted(manifest):
            path = source / relative
            info = tarfile.TarInfo(relative)
            info.size, info.mode = path.stat().st_size, 0o600
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return output.getvalue()


REMOTE = r'''
import ast,hashlib,io,json,os,re,stat,subprocess,sys,tarfile,tempfile
from pathlib import Path
HOST_ROOT=Path('/home/atta')
DATASET=Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json')
RUNTIME=Path('/home/atta/.hymem-siwc-runtime-v1/bin/python')
RUNNER='code/tools/diagnostics/siwc_lme_diagnostic_v1.py'
RUNNER_SHA='9366f40cf218581b51b877fd230e530c1e699cca0b76036e4304886c08c698f3'
MAP_SHA='1c56ea5806f629cf09655cc150610d338877204318f26835bcb696b0ccae24bd'
DATASET_SHA='d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
RUNTIME_SHA='17b78e0a93175e86f9ac03141924fd7a7f0c0c52e66b34bfa0de20ffef989df1'
SITE_SHA='2bed78ec3df853e3efe5052b30d514a2765a2097e183f4b3c32a3d53ef54806d'
GRANT='5f91fe05fb7d3b0552b7247d6a81f3fd29aae894dbcf45828b1556a0b11633dc'
MAX_BYTES=64*1024*1024
def need(ok,code):
    if not ok:raise RuntimeError(code)
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
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
    need(members and members[0].name=='manifest.json' and members[0].isfile(),'manifest_missing')
    manifest=json.loads(archive.extractfile(members[0]).read())
    need(type(manifest) is dict and manifest.get(RUNNER)==RUNNER_SHA
         and manifest.get('source-map.json')==MAP_SHA,'manifest_invalid')
    for name,digest in manifest.items():
        parts=Path(name).parts
        need(type(name) is str and name!='manifest.json' and
             ((parts and parts[0] in ('candidate','code')) or name=='source-map.json')
             and not Path(name).is_absolute() and '..' not in parts and
             type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest) is not None,
             'manifest_entry_invalid')
    names=[member.name for member in members[1:]]
    need(len(names)==len(manifest) and set(names)==set(manifest) and len(names)==len(set(names))
         and all(member.isfile() and not member.issym() and not member.islnk()
                 for member in members),'archive_file_set_invalid')
    content={}
    for member in members[1:]:
        raw=archive.extractfile(member).read()
        need(sha_bytes(raw)==manifest[member.name],'archive_hash_mismatch')
        content[member.name]=raw
    constants={}
    for node in ast.parse(content[RUNNER]).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('ACCEPTED_FILES','ACCEPTED_MAP_SHA256','PINS','SIWC_PINS','DIAGNOSTIC_HELPER_SHA256'):
                constants[name]=ast.literal_eval(node.value)
    need(set(constants)=={'ACCEPTED_FILES','ACCEPTED_MAP_SHA256','PINS','SIWC_PINS','DIAGNOSTIC_HELPER_SHA256'},'manifest_invalid')
    expected={'source-map.json':MAP_SHA,RUNNER:RUNNER_SHA,
              'code/benchmarks/lme_diagnostic.py':constants['DIAGNOSTIC_HELPER_SHA256']}
    expected.update({'code/'+name:digest for name,digest in
                     {**constants['PINS'],**constants['SIWC_PINS']}.items()})
    stamp=json.loads(content['source-map.json'])
    entries=stamp.get('source_sha256')
    need(type(entries) is dict and len(entries)==constants['ACCEPTED_FILES'] and
         sha_bytes(json.dumps(entries,sort_keys=True,separators=(',',':')).encode())==constants['ACCEPTED_MAP_SHA256'],
         'candidate_map_invalid')
    expected.update({'candidate/'+name:digest for name,digest in entries.items()})
    need(expected==manifest,'manifest_invalid')
need(regular(DATASET) and sha_file(DATASET)==DATASET_SHA,'dataset_identity_invalid')
need(regular(RUNTIME) and sha_file(RUNTIME)==RUNTIME_SHA,'runtime_identity_invalid')
site=RUNTIME.parent.parent/'lib/python3.13/site-packages'
need(site.is_dir() and not site.is_symlink(),'runtime_site_invalid')
files={}
for path in site.rglob('*'):
    need(not path.is_symlink(),'runtime_site_invalid')
    if path.suffix=='.pyc':continue
    need(path.is_dir() or regular(path),'runtime_site_invalid')
    if path.is_file():files[str(path.relative_to(site))]=sha_file(path)
need(len(files)==309 and sha_bytes(json.dumps(files,sort_keys=True,separators=(',',':')).encode())==SITE_SHA,
     'runtime_site_invalid')
root=Path(tempfile.mkdtemp(prefix='.hymem-siwc-lme-diagnostic-preflight-',dir=HOST_ROOT))
os.chmod(root,0o700)
for name,raw in content.items():
    target=root/name
    target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    with target.open('xb') as handle:handle.write(raw)
    os.chmod(target,0o600)
    need(sha_file(target)==manifest[name],'copy_hash_mismatch')
command=[str(RUNTIME),'-I','-B',str(root/RUNNER),'--root',str(root),
    '--inventory',str(root/'source-map.json'),'--inventory-sha256',MAP_SHA,
    '--dataset',str(DATASET),'--preflight-only']
try:result=subprocess.run(command,capture_output=True,timeout=180,
                          env={'HOME':'/home/atta','PATH':'/usr/bin:/bin','TMPDIR':str(root)})
except subprocess.TimeoutExpired:raise RuntimeError('preflight_timeout') from None
need(result.returncode==0,'preflight_failed')
parsed=json.loads(result.stdout)
need(type(parsed) is dict and parsed.get('preflight_verified') is True and
     parsed.get('model_calls')==0 and parsed.get('selected_count')==4 and
     parsed.get('candidate_map_sha256')==constants['ACCEPTED_MAP_SHA256'],
     'preflight_result_invalid')
print(json.dumps({'schema':'siwc-lme-diagnostic-host-preflight-v1','root':str(root),
    'candidate_files':len(entries),'code_files':len(constants['PINS'])+len(constants['SIWC_PINS'])+2,
    'inventory_sha256':MAP_SHA,'runtime_sha256':RUNTIME_SHA,'runtime_site_sha256':SITE_SHA,
    'dataset_sha256':DATASET_SHA,'grant_identity_sha256':GRANT,
    'preflight_verified':True,'selected_count':4,'model_calls':0},sort_keys=True))
'''

SAFE_ERRORS = frozenset({"host_identity_invalid", "archive_too_large", "manifest_missing",
    "manifest_invalid", "manifest_entry_invalid", "archive_file_set_invalid",
    "archive_hash_mismatch", "candidate_map_invalid", "dataset_identity_invalid",
    "runtime_identity_invalid", "runtime_site_invalid", "copy_hash_mismatch",
    "preflight_timeout", "preflight_failed", "preflight_result_invalid"})


def wrapped_remote() -> str:
    return ("import json,sys\ntry:\n" + textwrap.indent(REMOTE, "    ")
        + "\nexcept RuntimeError as exc:\n"
        + f"    code=str(exc)\n    safe={repr(sorted(SAFE_ERRORS))}\n"
        + "    output={'error':code if code in safe else 'unclassified'}\n"
          "    if 'root' in globals() and str(root).startswith('/home/atta/.hymem-siwc-lme-diagnostic-preflight-'):\n"
          "        output['root']=str(root)\n"
          "    print(json.dumps(output))\n"
          "    sys.exit(1)\n"
        + "except BaseException:\n"
          "    output={'error':'unclassified'}\n"
          "    if 'root' in globals() and str(root).startswith('/home/atta/.hymem-siwc-lme-diagnostic-preflight-'):\n"
          "        output['root']=str(root)\n"
          "    print(json.dumps(output))\n"
          "    sys.exit(1)\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--host", default="afrodite")
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9_.-]+", args.host) is None or args.host.startswith("-"):
        print(json.dumps({"status": "preflight_unverified", "reason": "host_alias_invalid"}))
        return 1
    try:
        archive = archive_bytes(Path(args.source))
    except (ValueError, OSError, json.JSONDecodeError):
        print(json.dumps({"status": "preflight_unverified", "reason": "source_invalid"}))
        return 1
    command = SSH_BASE + [args.host, shlex.join(["python3", "-I", "-B", "-c", wrapped_remote()])]
    try:
        remote = subprocess.run(command, input=archive, capture_output=True, timeout=240)
    except (OSError, subprocess.SubprocessError):
        print(json.dumps({"status": "preflight_unverified"}))
        return 1
    try:
        if len(remote.stdout) > 8192:
            raise ValueError("metadata_too_large")
        value = json.loads(remote.stdout, object_pairs_hook=lambda pairs:
            _unique_fields(pairs), parse_constant=lambda _value:
            (_ for _ in ()).throw(ValueError("nonfinite_metadata")))
    except (ValueError, TypeError):
        print(json.dumps({"status": "preflight_unverified"}))
        return 1
    if remote.returncode:
        code = value.get("error") if type(value) is dict else None
        output = {"status": "preflight_unverified",
            "reason": code if code in SAFE_ERRORS else "remote_failed"}
        candidate_root = value.get("root") if type(value) is dict else None
        if (type(candidate_root) is str and re.fullmatch(
                r"/home/atta/\.hymem-siwc-lme-diagnostic-preflight-[a-z0-9_-]{8,}", candidate_root)):
            output["root"] = candidate_root
        print(json.dumps(output))
        return 1
    if not _success_projection(value, source_manifest(Path(args.source))):
        print(json.dumps({"status": "preflight_unverified", "reason": "remote_receipt_invalid"}))
        return 1
    print(json.dumps(value, sort_keys=True))
    return 0


def _unique_fields(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_field")
        output[key] = value
    return output


def _success_projection(value, manifest):
    if type(value) is not dict or set(value) != {
            "schema", "root", "candidate_files", "code_files", "inventory_sha256",
            "runtime_sha256", "runtime_site_sha256", "dataset_sha256",
            "grant_identity_sha256", "preflight_verified", "selected_count", "model_calls"}:
        return False
    expected_root = re.fullmatch(r"/home/atta/\.hymem-siwc-lme-diagnostic-preflight-[a-z0-9_-]{8,}",
        value["root"]) if type(value["root"]) is str else None
    return bool(expected_root and value["schema"] == "siwc-lme-diagnostic-host-preflight-v1"
        and type(value["candidate_files"]) is int and value["candidate_files"] == 514
        and type(value["code_files"]) is int
        and value["code_files"] == sum(key.startswith("code/") for key in manifest)
        and value["inventory_sha256"] == MAP_SHA
        and value["runtime_sha256"] == "17b78e0a93175e86f9ac03141924fd7a7f0c0c52e66b34bfa0de20ffef989df1"
        and value["runtime_site_sha256"] == "2bed78ec3df853e3efe5052b30d514a2765a2097e183f4b3c32a3d53ef54806d"
        and value["dataset_sha256"] == DATASET_SHA
        and value["grant_identity_sha256"] == "5f91fe05fb7d3b0552b7247d6a81f3fd29aae894dbcf45828b1556a0b11633dc"
        and value["preflight_verified"] is True
        and type(value["selected_count"]) is int and value["selected_count"] == 4
        and type(value["model_calls"]) is int and value["model_calls"] == 0)


if __name__ == "__main__":
    raise SystemExit(main())
