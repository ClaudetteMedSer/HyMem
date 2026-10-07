"""Read-only metadata for reconstructing a stopped Honcho launch.

Runs inside hermes-1 only when explicitly invoked. It reads maintained source
files and sibling process environments privately, and returns booleans, counts,
file hashes, and a command-shape label. It never prints values, raw scripts,
arguments, process environments, or database content.
"""
from __future__ import annotations

import json
import shlex
import subprocess


SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite"]

REMOTE = r"""
import hashlib,json,os,re,shlex,stat,sys
from pathlib import Path

STAGE=Path('/home/node/.hermes/repairs/honcho-sqlite-jd70_3xs')
RECORDED='664846cdaa6d7ee5e16e2403a0a18461a004c27b676796b44ef340237f1c7429'
LIVE=Path('/home/node/HyMem')
HOOK=Path('/home/node/.agent37/hooks/post-restart.sh')
WRAPPER=Path('/home/node/.hermes/bin/hymem-server-wrapper')
ENV=Path('/home/node/.hermes/.env')
SCRIPT_DIR=Path('/home/node/.hermes/scripts')
REQUIRED=(
    'HYMEM_ROOT','HYMEM_LLM_MODEL','HYMEM_LLM_BASE_URL','HYMEM_LLM_API_KEY',
    'HYMEM_LLM_THINKING','HYMEM_EMBEDDING_BASE_URL','HYMEM_EMBEDDING_MODEL',
    'HYMEM_EMBEDDING_DIM','HYMEM_EMBEDDING_PIN_DIMENSION',
    'HYMEM_EMBEDDING_DEPLOYMENT_REVISION','HYMEM_EMBEDDING_DEPLOYMENT_TENANT',
)

def sha(data): return hashlib.sha256(data).hexdigest()
def need(ok,code):
    if not ok: raise RuntimeError(code)
def metadata(path):
    if not path.is_file() or path.is_symlink(): return {'present':False}
    s=path.stat()
    return {'present':True,'sha256':sha(path.read_bytes()),'bytes':s.st_size,
            'uid':s.st_uid,'mode':stat.S_IMODE(s.st_mode)}
def argv_match(digest):
    interpreters=(b'/usr/bin/python3.11',b'/home/node/hymem-env/bin/python',
      b'/home/node/hymem-env/bin/python3',b'/home/node/hymem-env/bin/python3.11')
    candidates=[]
    for python in interpreters:
        candidates.append(('console:'+os.fsdecode(python),
            (python,b'/home/node/hymem-env/bin/hymem-honcho')))
        candidates.append(('module:'+os.fsdecode(python),
            (python,b'-m',b'hymem.honcho.app')))
    matches=[]
    for label,argv in candidates:
        for trailing in (True,False):
            raw=b'\0'.join(argv)+(b'\0' if trailing else b'')
            if sha(raw)==digest:
                matches.append({'shape':label,'argc':len(argv),'trailing_nul':trailing})
    return matches
def parse_static_env(source):
    values={}; duplicate=[]; unparsed=0; dynamic=0
    for line in source.splitlines():
        stripped=line.strip()
        if not stripped or stripped.startswith('#'): continue
        match=re.fullmatch(r'(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)',stripped)
        if not match:
            unparsed+=1;continue
        key,raw=match.groups()
        if '$' in raw or '`' in raw or '\\' in raw:
            dynamic+=1;continue
        try: tokens=shlex.split(raw,comments=True,posix=True)
        except ValueError:
            unparsed+=1;continue
        if len(tokens)!=1:
            unparsed+=1;continue
        if key in values: duplicate.append(key)
        values[key]=tokens[0]
    return values,{'assignment_count':len(values),'duplicate_names':sorted(set(duplicate)),
                   'unparsed_lines':unparsed,'dynamic_lines':dynamic,
                   'required_present':{key:key in values for key in REQUIRED}}
def parse_proc_env(pid):
    raw=(Path('/proc')/str(pid)/'environ').read_bytes()
    values={}
    for item in raw.split(b'\0'):
        if b'=' in item:
            key,value=item.split(b'=',1)
            values[os.fsdecode(key)]=os.fsdecode(value)
    return values
def mcp_environments():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal(): continue
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            if not (b'hymem.server' in argv or any(a.rsplit(b'/',1)[-1]==b'hymem-server' for a in argv)):
                continue
            found.append((int(proc.name),parse_proc_env(proc.name)))
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    return found
def hook_shape(source):
    lines=source.splitlines()
    result={'line_count':len(lines)}
    for label,pattern in (
        ('honcho_lines',r'hymem-honcho|hymem\.honcho'),
        ('nohup_lines',r'\bnohup\b'),
        ('env_file_lines',r'\.hermes/\.env'),
        ('source_lines',r'\bsource\b|^\s*\.\s'),
        ('export_lines',r'^\s*export\s'),
        ('restart_lines',r'\brestart\b'),
    ):
        result[label]=[i+1 for i,line in enumerate(lines) if re.search(pattern,line)]
    return result

intent=json.loads((STAGE/'recover-intent.json').read_text())
need(intent.get('cmdline_sha256')==RECORDED,'intent_cmdline_hash_changed')
need((STAGE/'install.json').is_file(),'install_receipt_missing')
matches=argv_match(RECORDED)
files={'hook':metadata(HOOK),'wrapper':metadata(WRAPPER),'env':metadata(ENV)}
need(files['hook']['present'] and files['wrapper']['present'] and files['env']['present'],
     'maintained_source_missing')
hook=hook_shape(HOOK.read_text())
env_values,env_meta=parse_static_env(ENV.read_text())
mcp=mcp_environments()
comparisons={key:bool(mcp) and key in env_values and all(
    process.get(key)==env_values[key] for _,process in mcp) for key in REQUIRED}
scripts=[]
if SCRIPT_DIR.is_dir():
    for path in SCRIPT_DIR.iterdir():
        name=path.name.lower()
        if ('hymem' in name or 'honcho' in name) and not any(
            word in name for word in ('backup','before','.bak','.old')):
            scripts.append({'name':path.name,**metadata(path)})
print(json.dumps({'action':'inspect-recovery-sources','recorded_cmdline_matches':matches,
    'unique_cmdline_match':len(matches)==1,'sources':files,'hook_shape':hook,
    'env_structure':env_meta,'mcp_process_count':len(mcp),
    'required_env_matches_mcp':comparisons,
    'llm_model_expected':env_values.get('HYMEM_LLM_MODEL')=='deepseek-flash',
    'root_expected':env_values.get('HYMEM_ROOT')=='/home/node/.hermes',
    'active_script_metadata':scripts},sort_keys=True))
"""


def main() -> int:
    command = SSH + [shlex.join(["docker", "exec", "-i", "-u", "node", "hermes-1",
                                 "/usr/bin/python3.11", "-B", "-c", REMOTE])]
    try:
        result = subprocess.run(command, input="", text=True, capture_output=True,
                                timeout=25)
    except subprocess.TimeoutExpired:
        print(json.dumps({"error": "inspection_timeout"}))
        return 1
    if result.returncode:
        print(json.dumps({"error": "inspection_failed", "returncode": result.returncode}))
        return 1
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(json.dumps({"error": "invalid_metadata"}))
        return 1
    print(json.dumps(data, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
