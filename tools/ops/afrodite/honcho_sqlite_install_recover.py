"""Reviewed second phase of the Afrodite Honcho SQLite repair.

No action runs on import. Install changes only two source files after the
offline-rehearsal receipt is verified. Recover signals only the staged Honcho
PID and starts its exact privately captured invocation and environment.
Output is finite metadata; process arguments, environment, and logs stay inside
the container. This helper has no automatic retry or unknown-state rollback.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import textwrap


SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite"]
DEFAULT_STAGE = "/home/node/.hermes/repairs/honcho-sqlite-jd70_3xs"

REMOTE = r"""
import hashlib,json,os,signal,stat,subprocess,sys,time
from pathlib import Path

ACTION=sys.argv[1]
REQUEST=json.load(sys.stdin)
STAGE=Path(REQUEST['stage'])
ROOT=Path('/home/node/.hermes/repairs')
LIVE=Path('/home/node/HyMem')
LIVE_DB=LIVE/'hymem/core/db.py'
LIVE_HELPER=LIVE/'hymem/core/serialized_sqlite.py'
PID=12381
START=215405984
ORIGINAL='ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3'
CANDIDATE='aa1a252c5a9b32e5b0f91962e5e0499f71525e7d725931ba16597f350f339fb9'
HELPER='b46335c9e2f8f0453204e4fc9d3f75bc63b13a26bc808e2b0923bc3ea31eca5e'
BACKUP='a83ef34da8f7ecc7828dd532c5f0909a15e1364bcfa18fd037eff8b45b5c68a9'
EXE='/usr/bin/python3.11'
UID=1000

def need(value,code):
    if not value: raise RuntimeError(code)
def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def sha(path): return sha_bytes(path.read_bytes())
def json_file(path):
    need(path.is_file() and not path.is_symlink(),'receipt_missing')
    return json.loads(path.read_text())
def receipt(path,value):
    need(not path.exists(),'receipt_already_exists')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('x') as handle:
        os.chmod(temp,0o600)
        json.dump(value,handle,sort_keys=True,separators=(',',':'))
        handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path)
    directory_fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
def validate_stage():
    need(STAGE.parent==ROOT and STAGE.name=='honcho-sqlite-jd70_3xs','stage_path_mismatch')
    need(STAGE.is_dir() and not STAGE.is_symlink(),'stage_missing')
    need(STAGE.stat().st_uid==UID and (STAGE.stat().st_mode & 0o077)==0,'stage_permissions')
    stage=json_file(STAGE/'stage.json')
    rehearsal=json_file(STAGE/'rehearse.json')
    need(stage['expected_original_sha256']==ORIGINAL,'stage_original_mismatch')
    need(stage['candidate_db_sha256']==CANDIDATE and stage['helper_sha256']==HELPER,'stage_candidate_mismatch')
    need(stage['backup_sha256']==BACKUP,'stage_backup_mismatch')
    need(rehearsal['candidate_db_sha256']==CANDIDATE and rehearsal['helper_sha256']==HELPER,'rehearsal_candidate_mismatch')
    need(rehearsal['backup_sha256']==BACKUP,'rehearsal_backup_mismatch')
    need(rehearsal['synthetic_concurrency']=='passed' and rehearsal['clone_token_overlap']=='passed' and rehearsal['quick_check']=='ok','rehearsal_gate_failed')
    need(sha(STAGE/'original-db.py')==ORIGINAL,'staged_original_changed')
    need(sha(STAGE/'hymem/core/db.py')==CANDIDATE,'staged_candidate_changed')
    need(sha(STAGE/'hymem/core/serialized_sqlite.py')==HELPER,'staged_helper_changed')
    need(sha(STAGE/'backup.sqlite')==BACKUP,'staged_backup_changed')
    need(stage['identity']=={'pid':PID,'start_ticks':START,'uid':UID,'exe':EXE,'cwd':str(LIVE)},'staged_process_identity_mismatch')
    need(os.geteuid()==UID,'runner_uid_mismatch')
    return stage,rehearsal
def identity(pid):
    proc=Path('/proc')/str(pid)
    stat_text=(proc/'stat').read_text()
    fields=stat_text.rpartition(') ')[2].split()
    need(len(fields)>19,'proc_stat_invalid')
    uid=int(next(line.split()[1] for line in (proc/'status').read_text().splitlines() if line.startswith('Uid:')))
    return {'pid':pid,'start_ticks':int(fields[19]),'uid':uid,
            'exe':os.readlink(proc/'exe'),'cwd':os.readlink(proc/'cwd')}
def original_identity():
    found=identity(PID)
    need(found=={'pid':PID,'start_ticks':START,'uid':UID,'exe':EXE,'cwd':str(LIVE)},'original_process_changed')
    return found
def same_original():
    try:
        proc=Path('/proc')/str(PID)
        fields=(proc/'stat').read_text().rpartition(') ')[2].split()
        return len(fields)>19 and fields[0] not in ('Z','X') and int(fields[19])==START
    except (FileNotFoundError,ProcessLookupError): return False
def valid_command(argv):
    python_names=(b'/usr/bin/python3.11',b'/home/node/hymem-env/bin/python',
                  b'/home/node/hymem-env/bin/python3',b'/home/node/hymem-env/bin/python3.11')
    launcher=b'/home/node/hymem-env/bin/hymem-honcho'
    if len(argv)==2 and argv[0] in python_names and argv[1]==launcher:
        return Path(os.fsdecode(launcher)).is_file() and not Path(os.fsdecode(launcher)).is_symlink()
    if len(argv)==3 and argv[0] in python_names and argv[1:]==[b'-m',b'hymem.honcho.app']:
        return True
    return False
def command_and_environment():
    proc=Path('/proc')/str(PID)
    raw_argv=(proc/'cmdline').read_bytes()
    raw_env=(proc/'environ').read_bytes()
    argv=raw_argv.rstrip(b'\0').split(b'\0')
    need(argv and all(argv) and len(argv)<=24,'cmdline_invalid')
    # A shebang launcher is represented by interpreter + script in /proc.
    # Also admit the maintained module entry point, without interpreter flags.
    need(valid_command(argv),'cmdline_shape_changed')
    env={}
    for item in raw_env.split(b'\0'):
        if not item: continue
        need(b'=' in item,'environment_invalid')
        key,value=item.split(b'=',1)
        need(key and b'\0' not in key and key not in env,'environment_invalid')
        env[key]=value
    need(env,'environment_empty')
    return argv,env,sha_bytes(raw_argv)
def listening_inodes(port):
    found=set()
    for table in ('/proc/net/tcp','/proc/net/tcp6'):
        for line in Path(table).read_text().splitlines()[1:]:
            fields=line.split()
            if len(fields)>9 and fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16)==port:
                found.add(fields[9])
    return found
def port_owners(port):
    inodes=listening_inodes(port)
    owners=set()
    if not inodes: return owners
    seen=set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal(): continue
        try:
            for fd in (proc/'fd').iterdir():
                try: target=os.readlink(fd)
                except (FileNotFoundError,PermissionError): continue
                if target.startswith('socket:[') and target[8:-1] in inodes:
                    seen.add(target[8:-1])
                    owners.add(int(proc.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    need(seen==inodes,'port_owner_unresolved')
    return owners
def expected_port(env):
    raw=env.get(b'HYMEM_HONCHO_PORT',b'8765')
    need(type(raw) is bytes and raw.isdigit(),'port_invalid')
    port=int(raw)
    need(port==8765,'port_changed')
    return port
def venv_replay_ok():
    python='/home/node/hymem-env/bin/python'
    need(Path(python).is_file(),'venv_python_missing')
    code='import json,sys;print(json.dumps([sys.prefix,sys.base_prefix,sys.executable]))'
    safe_env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'}
    direct=subprocess.run([python,'-c',code],stdin=subprocess.DEVNULL,
        capture_output=True,text=True,timeout=5,env=safe_env,cwd=str(LIVE))
    replay=subprocess.run([python,'-c',code],executable=EXE,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,timeout=5,env=safe_env,cwd=str(LIVE))
    if direct.returncode or replay.returncode: return False
    try: a,b=json.loads(direct.stdout),json.loads(replay.stdout)
    except json.JSONDecodeError: return False
    return a==b and len(a)==3 and a[0]!=a[1]
def atomic_source(path,contents,mode):
    temporary=path.with_name(path.name+'.sqlite-repair-tmp')
    fd=os.open(str(temporary),os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(contents);handle.flush();os.fsync(handle.fileno())
        os.chmod(temporary,mode)
        os.replace(temporary,path)
        directory_fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        if temporary.exists(): temporary.unlink()
def install():
    stage,rehearsal=validate_stage()
    original_identity()
    argv,env,argv_sha=command_and_environment()
    port=expected_port(env)
    need(port_owners(port)=={PID},'port_ownership_changed')
    need(venv_replay_ok(),'venv_replay_mismatch')
    need(not (STAGE/'install.json').exists() and not (STAGE/'install-intent.json').exists(),'install_already_attempted')
    need(sha(LIVE_DB)==ORIGINAL,'live_db_source_changed')
    need(not LIVE_HELPER.exists() and not LIVE_HELPER.is_symlink(),'live_helper_already_exists')
    need(LIVE_DB.stat().st_uid==UID and LIVE_DB.is_file() and not LIVE_DB.is_symlink(),'live_db_file_changed')
    db_mode=stat.S_IMODE(LIVE_DB.stat().st_mode)
    need(db_mode in (0o644,0o640,0o600),'live_db_mode_unexpected')
    original=(STAGE/'original-db.py').read_bytes()
    candidate=(STAGE/'hymem/core/db.py').read_bytes()
    helper=(STAGE/'hymem/core/serialized_sqlite.py').read_bytes()
    receipt(STAGE/'install-intent.json',{'action':'install-intent','original_sha256':ORIGINAL,
        'candidate_db_sha256':CANDIDATE,'helper_sha256':HELPER,'pid':PID,'start_ticks':START,
        'cmdline_sha256':argv_sha,'port':port})
    try:
        atomic_source(LIVE_HELPER,helper,db_mode)
        need(sha(LIVE_HELPER)==HELPER,'helper_write_failed')
        atomic_source(LIVE_DB,candidate,db_mode)
        need(sha(LIVE_DB)==CANDIDATE,'db_write_failed')
        receipt(STAGE/'install.json',{'action':'install','original_sha256':ORIGINAL,
            'candidate_db_sha256':CANDIDATE,'helper_sha256':HELPER,
            'pid':PID,'start_ticks':START,'db_mode':db_mode})
    except BaseException:
        # Restore only our exact candidate bytes. If anything else has touched
        # either file, fail closed and retain the original in the private stage.
        if LIVE_DB.exists() and sha(LIVE_DB)==CANDIDATE:
            atomic_source(LIVE_DB,original,db_mode)
        if LIVE_HELPER.exists() and sha(LIVE_HELPER)==HELPER:
            LIVE_HELPER.unlink()
        receipt(STAGE/'install-failed.json',{'action':'install-failed',
            'db_sha256':sha(LIVE_DB) if LIVE_DB.exists() else None,
            'helper_present':LIVE_HELPER.exists()})
        raise
    print(json.dumps({'action':'install','stage':str(STAGE),'db_sha256':sha(LIVE_DB),
                      'helper_sha256':sha(LIVE_HELPER),'original_pid_unchanged':same_original()},sort_keys=True))
def recover():
    stage,rehearsal=validate_stage()
    installed=json_file(STAGE/'install.json')
    need(installed['original_sha256']==ORIGINAL and installed['candidate_db_sha256']==CANDIDATE and installed['helper_sha256']==HELPER,'install_receipt_mismatch')
    need(not (STAGE/'recover-intent.json').exists() and not (STAGE/'recover.json').exists(),'recovery_already_attempted')
    need(sha(LIVE_DB)==CANDIDATE and sha(LIVE_HELPER)==HELPER,'live_sources_changed')
    before=original_identity()
    argv,env,argv_sha=command_and_environment()
    port=expected_port(env)
    need(port_owners(port)=={PID},'port_ownership_changed')
    need(venv_replay_ok(),'venv_replay_mismatch')
    receipt(STAGE/'recover-intent.json',{'action':'recover-intent','identity':before,
        'cmdline_sha256':argv_sha,'port':port,'db_sha256':CANDIDATE,'helper_sha256':HELPER})
    os.kill(PID,signal.SIGTERM)
    term_deadline=time.monotonic()+18
    while same_original() and time.monotonic()<term_deadline: time.sleep(.2)
    escalated=False
    if same_original():
        need(original_identity()==before,'original_identity_changed_before_kill')
        os.kill(PID,signal.SIGKILL);escalated=True
        kill_deadline=time.monotonic()+6
        while same_original() and time.monotonic()<kill_deadline: time.sleep(.2)
    need(not same_original(),'original_still_running')
    need(not port_owners(port),'port_still_owned')
    log_path=STAGE/'honcho-restart.private.log'
    fd=os.open(str(log_path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb',buffering=0) as log:
            child=subprocess.Popen(argv,executable=EXE.encode(),cwd=str(LIVE),env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
    except BaseException:
        receipt(STAGE/'recover-failed.json',{'action':'recover-failed',
            'phase':'launch','original_stopped':True,'log_path':str(log_path)})
        raise
    new_pid=child.pid
    started=time.monotonic()+25
    launched=None
    while time.monotonic()<started:
        if child.poll() is not None: break
        try:
            found=identity(new_pid)
            raw=(Path('/proc')/str(new_pid)/'cmdline').read_bytes()
            if (found['uid']==UID and found['exe']==EXE and found['cwd']==str(LIVE)
                and sha_bytes(raw)==argv_sha and port_owners(port)=={new_pid}):
                launched=found;break
        except (FileNotFoundError,ProcessLookupError): pass
        time.sleep(.25)
    if launched is None:
        receipt(STAGE/'recover-failed.json',{'action':'recover-failed',
            'phase':'verification','original_stopped':True,'new_pid':new_pid,
            'log_path':str(log_path),'port_owned_by_new':new_pid in port_owners(port)})
        raise RuntimeError('new_process_unverified')
    result={'action':'recover','old_pid':PID,'old_start_ticks':START,
            'new_identity':launched,'cmdline_sha256':argv_sha,'port':port,
            'port_owned_by_new':True,'term_escalated':escalated,
            'db_sha256':CANDIDATE,'helper_sha256':HELPER,'log_path':str(log_path)}
    receipt(STAGE/'recover.json',result)
    print(json.dumps(result,sort_keys=True))
def preflight():
    validate_stage()
    source_matches=LIVE_DB.is_file() and not LIVE_DB.is_symlink() and sha(LIVE_DB)==ORIGINAL
    helper_absent=not LIVE_HELPER.exists() and not LIVE_HELPER.is_symlink()
    intent_absent=not (STAGE/'install-intent.json').exists() and not (STAGE/'install.json').exists()
    try: process_matches=original_identity() is not None
    except (FileNotFoundError,ProcessLookupError,RuntimeError): process_matches=False
    launcher_shape=False;sole_port_owner=False;venv_equivalent=False;port=None
    if process_matches:
        try:
            argv,env,argv_sha=command_and_environment()
            launcher_shape=True
            port=expected_port(env)
            sole_port_owner=port_owners(port)=={PID}
            venv_equivalent=venv_replay_ok()
        except (FileNotFoundError,ProcessLookupError,RuntimeError): pass
    checks={'source_matches':source_matches,'helper_absent':helper_absent,
            'intent_absent':intent_absent,'process_identity_matches':process_matches,
            'launcher_shape_valid':launcher_shape,'sole_port_owner':sole_port_owner,
            'venv_replay_equivalent':venv_equivalent}
    print(json.dumps({'action':'preflight','ready':all(checks.values()),
                      'checks':checks,'port':port},sort_keys=True))
def status():
    stage,rehearsal=validate_stage()
    installed=(STAGE/'install.json').is_file()
    recovered=(STAGE/'recover.json').is_file()
    output={'action':'status','stage':str(STAGE),'installed':installed,'recovered':recovered,
            'db_sha256':sha(LIVE_DB),'helper_sha256':sha(LIVE_HELPER) if LIVE_HELPER.is_file() else None}
    if recovered:
        r=json_file(STAGE/'recover.json')
        try: output['new_identity_unchanged']=identity(r['new_identity']['pid'])==r['new_identity']
        except (FileNotFoundError,ProcessLookupError): output['new_identity_unchanged']=False
        output['new_port_ownership']=port_owners(r['port'])=={r['new_identity']['pid']}
    else:
        output['original_identity_unchanged']=same_original()
    print(json.dumps(output,sort_keys=True))

if ACTION=='preflight': preflight()
elif ACTION=='install': install()
elif ACTION=='recover': recover()
elif ACTION=='status': status()
else: raise RuntimeError('unknown_action')
"""

# Guard failures expose only fixed operational codes. Arbitrary exception text,
# tracebacks, argv, environment, and child logs never cross the SSH boundary.
SAFE_GUARD_CODES = frozenset({
    "receipt_already_exists", "receipt_missing", "stage_path_mismatch",
    "stage_missing", "stage_permissions", "stage_original_mismatch",
    "stage_candidate_mismatch", "stage_backup_mismatch",
    "rehearsal_candidate_mismatch", "rehearsal_backup_mismatch",
    "rehearsal_gate_failed", "staged_original_changed",
    "staged_candidate_changed", "staged_helper_changed",
    "staged_backup_changed", "staged_process_identity_mismatch",
    "runner_uid_mismatch", "proc_stat_invalid", "original_process_changed",
    "cmdline_invalid", "cmdline_shape_changed", "environment_invalid",
    "environment_empty", "port_invalid", "port_changed",
    "port_owner_unresolved", "install_already_attempted",
    "live_db_source_changed", "live_helper_already_exists",
    "live_db_file_changed", "live_db_mode_unexpected",
    "port_ownership_changed", "helper_write_failed", "db_write_failed",
    "install_receipt_mismatch", "recovery_already_attempted",
    "live_sources_changed", "port_still_owned", "original_still_running",
    "original_identity_changed_before_kill", "new_process_unverified",
    "venv_python_missing", "venv_replay_mismatch",
    "unknown_action",
})


def wrapped_remote() -> str:
    safe_codes = repr(sorted(SAFE_GUARD_CODES))
    return (
        "import json,re,sys\ntry:\n" + textwrap.indent(REMOTE, "    ")
        + "\nexcept RuntimeError as exc:\n"
        + f"    code=str(exc)\n    safe={safe_codes}\n"
        + "    print(json.dumps({'error':code if code in safe else 'unclassified',"
          "'type':'RuntimeError'}))\n    sys.exit(1)\n"
        + "except BaseException as exc:\n"
        + "    name=type(exc).__name__\n"
          "    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}',name): name='Exception'\n"
          "    print(json.dumps({'error':'unclassified','type':name}))\n    sys.exit(1)\n"
    )


def run_remote(action: str, stage: str) -> int:
    command = SSH + [shlex.join(["docker", "exec", "-i", "-u", "node", "hermes-1",
                                 "/usr/bin/python3.11", "-c", wrapped_remote(), action])]
    try:
        result = subprocess.run(command, input=json.dumps({"stage": stage}), text=True,
                                capture_output=True, timeout=100)
    except subprocess.TimeoutExpired:
        print(json.dumps({"action": action, "error": "remote_timeout"}))
        return 1
    if result.returncode:
        try:
            remote_failure = json.loads(result.stdout)
            if not isinstance(remote_failure, dict) or remote_failure.get("error") not in (SAFE_GUARD_CODES | {"unclassified"}):
                raise ValueError("untrusted_failure")
            failure = {"action": action, "error": remote_failure["error"]}
            if failure["error"] == "unclassified":
                name = remote_failure.get("type")
                if isinstance(name, str) and name.isidentifier() and len(name) <= 64:
                    failure["type"] = name
            print(json.dumps(failure, sort_keys=True))
        except (json.JSONDecodeError, ValueError):
            print(json.dumps({"action": action, "error": "remote_failed",
                              "returncode": result.returncode}))
        return 1
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(json.dumps({"action": action, "error": "invalid_remote_receipt"}))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "install", "recover", "status"))
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    args = parser.parse_args()
    return run_remote(args.action, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
