"""One-shot Honcho start from pinned maintained configuration, without a hook.

Run `plan` first and review its boolean-only receipt. `start` repeats every
fence, creates a private intent, launches only the exact Honcho argv, records
the child PID immediately, then verifies health with a bounded poll. There is
no model call, database write, hook execution, container restart, or retry.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import textwrap


SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite"]

REMOTE = r"""
import hashlib,json,os,re,shlex,signal,stat,subprocess,sys,time,urllib.request
from pathlib import Path

ACTION=sys.argv[1]
STAGE=Path('/home/node/.hermes/repairs/honcho-sqlite-jd70_3xs')
LIVE=Path('/home/node/HyMem')
HOOK=Path('/home/node/.agent37/hooks/post-restart.sh')
WRAPPER=Path('/home/node/.hermes/bin/hymem-server-wrapper')
ENV_FILE=Path('/home/node/.hermes/.env')
EXE='/usr/bin/python3.11'
ARGV=(b'/home/node/hymem-env/bin/python3',b'/home/node/hymem-env/bin/hymem-honcho')
ARGV_SHA='664846cdaa6d7ee5e16e2403a0a18461a004c27b676796b44ef340237f1c7429'
ORIGINAL='ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3'
CANDIDATE='aa1a252c5a9b32e5b0f91962e5e0499f71525e7d725931ba16597f350f339fb9'
HELPER='b46335c9e2f8f0453204e4fc9d3f75bc63b13a26bc808e2b0923bc3ea31eca5e'
BACKUP='a83ef34da8f7ecc7828dd532c5f0909a15e1364bcfa18fd037eff8b45b5c68a9'
SOURCE_HASHES={
    HOOK:'4bd4cc0011f2bfa298cf073d823bce7aaf3a3a1fd3b4f801c010c89a8dcaef68',
    WRAPPER:'685b198a87e22c877d562e126b0f13780e500e30383b7a3f43ae2b6b73930402',
    ENV_FILE:'0fea90464e6f2ac8b6cf8cd907e43e6535c03eeeb3492f2737369b0aae8e19de',
}
NAMES=(
    'HYMEM_LLM_API_KEY','HYMEM_LLM_BASE_URL','HYMEM_LLM_MODEL',
    'HYMEM_EMBEDDING_API_KEY','HYMEM_EMBEDDING_BASE_URL',
    'HYMEM_EMBEDDING_MODEL','HYMEM_EMBEDDING_DIM',
    'HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP',
    'HYMEM_EMBEDDING_PIN_DIMENSION','HYMEM_EMBEDDING_DEPLOYMENT_REVISION',
    'HYMEM_EMBEDDING_DEPLOYMENT_TENANT','HYMEM_ROOT',
    'HYMEM_DREAM_COOLDOWN_SECONDS','HYMEM_AGGREGATION_NODES_ENABLED',
)
ALLOWED_REFS=frozenset(('DEEPSEEK_API_KEY','OPENAI_API_KEY','HOME','PATH',
    'HYMEM_ROOT','HYMEM_EMBEDDING_API_KEY','EMBEDDING_API_KEY'))
LAST_PARSE_CONTEXT={}
MISMATCH_NAMES=[]
CONFIG_DIAGNOSTICS={}

def need(ok,code):
    if not ok: raise RuntimeError(code)
def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(path.read_bytes())
def receipt(path,value):
    need(not path.exists(),'receipt_already_exists')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('x') as handle:
        os.chmod(temp,0o600)
        json.dump(value,handle,sort_keys=True,separators=(',',':'))
        handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path)
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
def read_receipt(path):
    need(path.is_file() and not path.is_symlink(),'receipt_missing')
    return json.loads(path.read_text())
def source_fences():
    need(os.geteuid()==1000,'uid_changed')
    need(STAGE.is_dir() and not STAGE.is_symlink(),'stage_missing')
    stage=read_receipt(STAGE/'stage.json')
    rehearsal=read_receipt(STAGE/'rehearse.json')
    installed=read_receipt(STAGE/'install.json')
    intent=read_receipt(STAGE/'recover-intent.json')
    need(stage['expected_original_sha256']==ORIGINAL and
         stage['candidate_db_sha256']==CANDIDATE and stage['helper_sha256']==HELPER and
         stage['backup_sha256']==BACKUP,'stage_receipt_changed')
    need(rehearsal['candidate_db_sha256']==CANDIDATE and
         rehearsal['helper_sha256']==HELPER and rehearsal['backup_sha256']==BACKUP and
         rehearsal['synthetic_concurrency']=='passed' and
         rehearsal['clone_token_overlap']=='passed' and rehearsal['quick_check']=='ok',
         'rehearsal_receipt_changed')
    need(installed['original_sha256']==ORIGINAL and installed['candidate_db_sha256']==CANDIDATE and
         installed['helper_sha256']==HELPER,'install_receipt_changed')
    need(intent['cmdline_sha256']==ARGV_SHA and intent['db_sha256']==CANDIDATE and
         intent['helper_sha256']==HELPER,'recover_intent_changed')
    need(file_sha(STAGE/'original-db.py')==ORIGINAL and
         file_sha(STAGE/'backup.sqlite')==BACKUP,'private_backup_changed')
    need(file_sha(LIVE/'hymem/core/db.py')==CANDIDATE and
         file_sha(LIVE/'hymem/core/serialized_sqlite.py')==HELPER,
         'installed_sources_changed')
    for path,digest in SOURCE_HASHES.items():
        need(path.is_file() and not path.is_symlink() and file_sha(path)==digest,
             'maintained_source_changed')
    need(sha(b'\0'.join(ARGV)+b'\0')==ARGV_SHA,'argv_hash_mismatch')
    need(Path(os.fsdecode(ARGV[1])).is_file() and not Path(os.fsdecode(ARGV[1])).is_symlink(),
         'launcher_missing')
def parse_literal(raw,base,allow_reference,source='unknown',name='unknown'):
    global LAST_PARSE_CONTEXT
    LAST_PARSE_CONTEXT={'source':source,'name':name,
        'backslash_count':raw.count('\\'),'dollar_count':raw.count('$'),
        'quote_type':'single' if raw.startswith("'") else
                     'double' if raw.startswith('"') else 'none',
        'has_space':any(char.isspace() for char in raw)}
    if raw=='': return ''
    need('$(' not in raw and '`' not in raw and '\n' not in raw,'dynamic_value')
    match=re.fullmatch(r'\$\{([A-Za-z_][A-Za-z0-9_]*):-\}|"\$\{([A-Za-z_][A-Za-z0-9_]*):-\}"',raw)
    if match:
        name=match.group(1) or match.group(2)
        need(allow_reference and name in ALLOWED_REFS and name in base,'reference_unapproved')
        return base[name]
    match=re.fullmatch(r'\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))|"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"',raw)
    if match:
        name=next(part for part in match.groups() if part is not None)
        need(allow_reference and name in ALLOWED_REFS and name in base,'reference_unapproved')
        return base[name]
    need('$' not in raw and '\\' not in raw,'dynamic_value')
    try: tokens=shlex.split(raw,comments=False,posix=True)
    except ValueError: raise RuntimeError('literal_syntax') from None
    need(len(tokens)==1,'literal_syntax')
    return tokens[0]
def env_file_values():
    values={}
    for line in ENV_FILE.read_text().splitlines():
        stripped=line.strip()
        if not stripped or stripped.startswith('#'): continue
        match=re.fullmatch(r'(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)',stripped)
        need(match is not None,'env_file_nonassignment')
        name,raw=match.groups()
        need(name not in values or name=='MISTRAL_API_KEY','env_file_duplicate')
        values[name]=parse_literal(raw,{},False,'env',name)
    need(bool(values.get('DEEPSEEK_API_KEY')),'deepseek_key_missing')
    return values
def hook_assignments(base):
    lines=HOOK.read_text().splitlines()
    need(len(lines)>=92,'hook_shape')
    prelude=lines[76].strip()
    need(prelude.endswith('\\'),'hook_nohup_shape')
    try: prelude_tokens=shlex.split(prelude[:-1].strip(),comments=True,posix=True)
    except ValueError: raise RuntimeError('hook_nohup_shape') from None
    prefix=prelude_tokens[:-2]
    need(prelude_tokens[-2:]==['nohup','env'] and prefix in (
         ['(cd',str(LIVE),'&&'],['cd',str(LIVE),'&&'],
         ['(','cd',str(LIVE),'&&'],[]), 'hook_nohup_shape')
    values={}
    for line,name in zip(lines[77:91],NAMES,strict=True):
        stripped=line.strip()
        need(stripped.endswith('\\'),'hook_continuation_missing')
        stripped=stripped[:-1].strip()
        match=re.fullmatch(r'([A-Z][A-Z0-9_]*)=(.*)',stripped)
        need(match is not None and match.group(1)==name,'hook_assignment_order')
        need(name not in values,'hook_assignment_duplicate')
        values[name]=parse_literal(match.group(2),{**base,**values},True,'hook',name)
    launch=lines[91].strip()
    need('$' not in launch and '`' not in launch and ';' not in launch and '|' not in launch,
         'hook_launcher_dynamic')
    try: tokens=shlex.split(launch,comments=True,posix=True)
    except ValueError: raise RuntimeError('hook_launcher_syntax') from None
    # The maintained shell has two redirects following the console command.
    # We do not run any shell text or carry over either redirection target.
    need(len(tokens)==5 and tokens[0]==os.fsdecode(ARGV[1]),'hook_launcher_mismatch')
    need(tokens[1] in ('>','>>','1>','1>>') and
         tokens[3] in ('2>','2>>','2>&1') and tokens[2] and tokens[4],
         'hook_redirection_shape')
    return values
def wrapper_assignments(base):
    values={}
    for line in WRAPPER.read_text().splitlines():
        match=re.fullmatch(r'\s*export\s+([A-Z][A-Z0-9_]*)=(.*)',line)
        if not match: continue
        name,raw=match.groups()
        if name not in NAMES: continue
        need(name not in values,'wrapper_duplicate')
        values[name]=parse_literal(raw,base,True,'wrapper',name)
    expected=set(NAMES)-{'HYMEM_DREAM_COOLDOWN_SECONDS'}
    need(set(values)==expected,'wrapper_assignment_set')
    return values
def mcp_environments():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal(): continue
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            if not (b'hymem.server' in argv or any(a.rsplit(b'/',1)[-1]==b'hymem-server' for a in argv)):
                continue
            env={}
            for item in (proc/'environ').read_bytes().split(b'\0'):
                if b'=' in item:
                    key,value=item.split(b'=',1)
                    env[os.fsdecode(key)]=os.fsdecode(value)
            found.append(env)
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    return found
def listener_count():
    count=0
    for table in ('/proc/net/tcp','/proc/net/tcp6'):
        for line in Path(table).read_text().splitlines()[1:]:
            fields=line.split()
            if len(fields)>9 and fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16)==8765:
                count+=1
    return count
def matching_processes():
    matches=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal(): continue
        try:
            if sha((proc/'cmdline').read_bytes())==ARGV_SHA: matches.append(int(proc.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    return matches
def runtime_fences():
    old=Path('/proc')/'12381'
    if old.exists():
        fields=(old/'stat').read_text().rpartition(') ')[2].split()
        need(len(fields)>19 and (fields[0]=='Z' or int(fields[19])!=215405984),
             'old_pid_present')
    need(not matching_processes(),'matching_process_present')
    need(listener_count()==0,'listener_present')
    time.sleep(.7)
    need(listener_count()==0 and not matching_processes(),'listener_not_stable')
    need(not (STAGE/'recover-failed.json').exists() and
         not (STAGE/'honcho-restart.private.log').exists() and
         not (STAGE/'recover.json').exists() and
         not (STAGE/'recovery-source-launch.private.log').exists() and
         not (STAGE/'recovery-source-launch.json').exists() and
         not (STAGE/'recovery-source-intent.json').exists(),
         'launch_already_attempted')
def configuration():
    global MISMATCH_NAMES,CONFIG_DIAGNOSTICS
    base=env_file_values()
    hook=hook_assignments({**os.environ,**base})
    wrapper=wrapper_assignments({**os.environ,**base})
    common=set(NAMES)-{'HYMEM_DREAM_COOLDOWN_SECONDS',
                       'HYMEM_AGGREGATION_NODES_ENABLED'}
    mcp=mcp_environments()
    aggregation={'honcho':hook['HYMEM_AGGREGATION_NODES_ENABLED'],
                 'wrapper':wrapper['HYMEM_AGGREGATION_NODES_ENABLED'],
                 'mcp':[process.get('HYMEM_AGGREGATION_NODES_ENABLED') for process in mcp]}
    def flag_category(value):
        if value is None: return 'missing'
        if value in ('0','1','true','false','yes','no','on','off','TRUE','FALSE'):
            return value
        return 'other'
    CONFIG_DIAGNOSTICS={
        'wrapper_mismatch_names':sorted(name for name in common if hook[name]!=wrapper[name]),
        'mcp_mismatch_names':sorted(name for name in common if any(
            process.get(name)!=hook[name] for process in mcp)),
        'aggregation_observed':{
            'honcho':flag_category(aggregation['honcho']),
            'wrapper':flag_category(aggregation['wrapper']),
            'mcp':sorted(flag_category(value) for value in aggregation['mcp'])},
        'mcp_count':len(mcp)}
    MISMATCH_NAMES=CONFIG_DIAGNOSTICS['wrapper_mismatch_names']
    need(not MISMATCH_NAMES,'wrapper_hook_config_mismatch')
    need(bool(mcp),'mcp_missing')
    normalized={'0':False,'1':True,'false':False,'true':True}
    values=[aggregation['honcho'],aggregation['wrapper'],*aggregation['mcp']]
    need(all(value in normalized for value in values),'aggregation_flag_invalid')
    need(len({normalized[value] for value in values})==1,'aggregation_flag_mismatch')
    MISMATCH_NAMES=CONFIG_DIAGNOSTICS['mcp_mismatch_names']
    need(not MISMATCH_NAMES,'mcp_hook_config_mismatch')
    need(hook['HYMEM_LLM_API_KEY']==base['DEEPSEEK_API_KEY'],'api_key_source_mismatch')
    need(hook['HYMEM_ROOT']=='/home/node/.hermes','root_changed')
    need(hook['HYMEM_EMBEDDING_DIM'].isdigit() and
         0<int(hook['HYMEM_EMBEDDING_DIM'])<=100000,'embedding_dim_invalid')
    aggregation['mcp']=sorted(aggregation['mcp'])
    return hook,base,len(mcp),aggregation
def plan():
    source_fences()
    config,base,mcp_count,aggregation=configuration()
    runtime_fences()
    return config,{'action':'plan','ready':True,'argv_sha256':ARGV_SHA,
        'candidate_db_sha256':CANDIDATE,'helper_sha256':HELPER,
        'hook_sha256':SOURCE_HASHES[HOOK],'wrapper_sha256':SOURCE_HASHES[WRAPPER],
        'env_sha256':SOURCE_HASHES[ENV_FILE],'assignment_count':len(config),
        'env_assignment_unique_count':len(base),
        'mcp_count':mcp_count,'old_pid_absent':True,'port_free_two_observations':True,
        'auth_matches_env':True,'common_config_matches_mcp':True,
        'aggregation_flags':aggregation,
        'aggregation_enabled':aggregation['honcho'] in ('1','true')}
def process_identity(pid):
    proc=Path('/proc')/str(pid)
    fields=(proc/'stat').read_text().rpartition(') ')[2].split()
    need(len(fields)>19,'new_proc_invalid')
    uid=int(next(line.split()[1] for line in (proc/'status').read_text().splitlines() if line.startswith('Uid:')))
    return {'pid':pid,'start_ticks':int(fields[19]),'uid':uid,
        'exe':os.readlink(proc/'exe'),'cwd':os.readlink(proc/'cwd')}
def start():
    config,plan_receipt=plan()
    need(not (STAGE/'recovery-source-failed.json').exists(),'prior_launch_failure')
    receipt(STAGE/'recovery-source-intent.json',{
        'action':'source-launch-intent','argv_sha256':ARGV_SHA,
        'candidate_db_sha256':CANDIDATE,'helper_sha256':HELPER,
        'hook_sha256':SOURCE_HASHES[HOOK],'assignment_count':len(config)})
    env=dict(os.environb)
    for name in NAMES: env[name.encode()]=config[name].encode()
    log_path=STAGE/'recovery-source-launch.private.log'
    fd=os.open(str(log_path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb',buffering=0) as log:
            child=subprocess.Popen(ARGV,executable=EXE.encode(),cwd=str(LIVE),env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
    except BaseException:
        receipt(STAGE/'recovery-source-failed.json',{'action':'source-launch-failed',
            'phase':'spawn','log_path':str(log_path)})
        raise
    pid=child.pid
    try:
        identity=process_identity(pid)
    except (FileNotFoundError,ProcessLookupError):
        identity={'pid':pid,'start_ticks':None,'uid':None,'exe':None,'cwd':None}
    receipt(STAGE/'recovery-source-launch.json',{'action':'source-launch','identity':identity,
        'argv_sha256':ARGV_SHA,'candidate_db_sha256':CANDIDATE,
        'helper_sha256':HELPER,'log_path':str(log_path)})
    end=time.monotonic()+30
    healthy=False;owner=False;settled=None
    while time.monotonic()<end:
        if child.poll() is not None: break
        try:
            actual=process_identity(pid)
            raw=(Path('/proc')/str(pid)/'cmdline').read_bytes()
            if (actual['uid']!=1000 or actual['exe']!=EXE or actual['cwd']!=str(LIVE)
                or sha(raw)!=ARGV_SHA): break
            settled=actual
            # A listener may outlive its fd entry briefly. Poll until ownership
            # resolves instead of treating a transient orphan inode as fatal.
            inodes=set()
            for table in ('/proc/net/tcp','/proc/net/tcp6'):
                for line in Path(table).read_text().splitlines()[1:]:
                    fields=line.split()
                    if len(fields)>9 and fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16)==8765:
                        inodes.add(fields[9])
            own=set()
            for fdpath in (Path('/proc')/str(pid)/'fd').iterdir():
                try:
                    target=os.readlink(fdpath)
                    if target.startswith('socket:['): own.add(target[8:-1])
                except (FileNotFoundError,PermissionError): pass
            owner=bool(inodes) and inodes<=own
            if owner:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=1.5) as response:
                        body=json.loads(response.read(4096))
                        healthy=response.status==200 and isinstance(body,dict) and body.get('status')=='ok' and body.get('backend')=='hymem'
                except Exception: pass
            if owner and healthy: break
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
        time.sleep(.3)
    result={'action':'source-launch-verified' if owner and healthy else 'source-launch-unverified',
        'pid':pid,'identity':settled,'argv_sha256':ARGV_SHA,
        'port_owned_by_new':owner,'health_ok':healthy,
        'log_path':str(log_path)}
    receipt(STAGE/'recovery-source-result.json',result)
    print(json.dumps(result,sort_keys=True))

if ACTION=='plan':
    _,result=plan();print(json.dumps(result,sort_keys=True))
elif ACTION=='start': start()
else: raise RuntimeError('unknown_action')
"""


SAFE_CODES = frozenset({
    'receipt_already_exists','receipt_missing','uid_changed','stage_missing',
    'stage_receipt_changed','rehearsal_receipt_changed','install_receipt_changed',
    'recover_intent_changed','private_backup_changed','installed_sources_changed',
    'maintained_source_changed','argv_hash_mismatch','launcher_missing',
    'dynamic_value','reference_unapproved','literal_syntax','env_file_nonassignment',
    'env_file_duplicate','deepseek_key_missing',
    'hook_shape','hook_nohup_shape','hook_continuation_missing',
    'hook_assignment_order','hook_assignment_duplicate','hook_launcher_dynamic',
    'hook_launcher_syntax','hook_launcher_mismatch','hook_redirection_shape',
    'wrapper_duplicate','wrapper_assignment_set','old_pid_present',
    'matching_process_present','listener_present','listener_not_stable',
    'launch_already_attempted','wrapper_hook_config_mismatch','mcp_missing',
    'mcp_hook_config_mismatch','aggregation_flag_invalid','aggregation_flag_mismatch',
    'api_key_source_mismatch','llm_model_changed',
    'root_changed','embedding_dim_invalid','new_proc_invalid','prior_launch_failure',
    'unknown_action',
})


def wrapped_remote() -> str:
    return (
        'import json,re,sys\ntry:\n' + textwrap.indent(REMOTE, '    ')
        + '\nexcept RuntimeError as exc:\n'
        + f'    code=str(exc)\n    safe={repr(sorted(SAFE_CODES))}\n'
        + "    result={'error':code if code in safe else 'unclassified','type':'RuntimeError'}\n"
          "    if code in ('dynamic_value','literal_syntax','reference_unapproved'):\n"
          "        result['parse_context']=LAST_PARSE_CONTEXT\n"
          "    if code in ('wrapper_hook_config_mismatch','mcp_hook_config_mismatch',"
          "'aggregation_flag_invalid','aggregation_flag_mismatch','mcp_missing'):\n"
          "        result['config_diagnostics']=CONFIG_DIAGNOSTICS\n"
          "    print(json.dumps(result))\n    sys.exit(1)\n"
        + 'except BaseException as exc:\n'
        + "    name=type(exc).__name__\n"
          "    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}',name): name='Exception'\n"
          "    print(json.dumps({'error':'unclassified','type':name}))\n    sys.exit(1)\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan','start'))
    args = parser.parse_args()
    command = SSH + [shlex.join(['docker','exec','-i','-u','node','hermes-1',
        '/usr/bin/python3.11','-B','-c',wrapped_remote(),args.action])]
    try:
        result = subprocess.run(command,input='',text=True,capture_output=True,timeout=70)
    except subprocess.TimeoutExpired:
        print(json.dumps({'action':args.action,'error':'remote_timeout'}));return 1
    try: value=json.loads(result.stdout)
    except json.JSONDecodeError:
        print(json.dumps({'action':args.action,'error':'invalid_receipt'}));return 1
    if result.returncode:
        code=value.get('error') if isinstance(value,dict) else None
        output={'action':args.action,'error':code if code in SAFE_CODES else 'unclassified'}
        if isinstance(value,dict) and code in ('dynamic_value','literal_syntax','reference_unapproved'):
            context=value.get('parse_context')
            if isinstance(context,dict):
                output['parse_context']=context
        if isinstance(value,dict) and code in ('wrapper_hook_config_mismatch','mcp_hook_config_mismatch',
                                               'aggregation_flag_invalid','aggregation_flag_mismatch','mcp_missing'):
            diagnostic=value.get('config_diagnostics')
            allowed=frozenset((
                'HYMEM_LLM_API_KEY','HYMEM_LLM_BASE_URL','HYMEM_LLM_MODEL',
                'HYMEM_EMBEDDING_API_KEY','HYMEM_EMBEDDING_BASE_URL',
                'HYMEM_EMBEDDING_MODEL','HYMEM_EMBEDDING_DIM',
                'HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP',
                'HYMEM_EMBEDDING_PIN_DIMENSION','HYMEM_EMBEDDING_DEPLOYMENT_REVISION',
                'HYMEM_EMBEDDING_DEPLOYMENT_TENANT','HYMEM_ROOT',
                'HYMEM_AGGREGATION_NODES_ENABLED'))
            categories=frozenset(('0','1','true','false','yes','no','on','off',
                                  'TRUE','FALSE','missing','other'))
            if isinstance(diagnostic,dict):
                wrapper_names=diagnostic.get('wrapper_mismatch_names')
                mcp_names=diagnostic.get('mcp_mismatch_names')
                flags=diagnostic.get('aggregation_observed')
                if (isinstance(wrapper_names,list) and isinstance(mcp_names,list) and
                    all(isinstance(name,str) and name in allowed for name in wrapper_names+mcp_names) and
                    isinstance(flags,dict) and flags.get('honcho') in categories and
                    flags.get('wrapper') in categories and isinstance(flags.get('mcp'),list) and
                    all(flag in categories for flag in flags['mcp'])):
                    output['config_diagnostics']={
                        'wrapper_mismatch_names':wrapper_names,
                        'mcp_mismatch_names':mcp_names,
                        'aggregation_observed':flags,
                        'mcp_count':diagnostic.get('mcp_count')}
        print(json.dumps(output,sort_keys=True))
        return 1
    print(json.dumps(value,sort_keys=True));return 0


if __name__ == '__main__':
    raise SystemExit(main())
