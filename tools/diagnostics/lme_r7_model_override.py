"""Guarded Hermes1 R7 model override. Run only after operator review.

Changes the pinned startup wrapper and hook (and an active .env assignment, if
present) from deepseek-v4-flash to deepseek-flash. Does not restart anything.
Only metadata is returned over SSH; backups remain private on the remote host.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys


TASK_HOME = "/opt/stacks/hermes/instance1/home"
STAGE = TASK_HOME + "/.hermes/benchmarks/r7-deploy-20260925"
PINS = {
    ".hermes/bin/hymem-server-wrapper": "f5b69c6b397c2519c11e8bfc0d457a7e6a2cf9747d5c9d8632cbf6698a561844",
    ".agent37/hooks/post-restart.sh": "84a737b339ada8f1b5b16c94b9da92364bc5eb5144797d76e9b7e0b4dca99159",
}
SSH_OPTIONS = (
    "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
    "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
)

REMOTE = r'''
import hashlib,json,os,pathlib,re,stat,sys,tempfile

def need(ok, code):
    if not ok: raise RuntimeError(code)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def check_pin(data, pin):
    need(digest(data)==pin, 'target_hash_mismatch')

# Match complete shell/.env assignments only. Anything else that begins with
# the active key is ambiguous and is rejected, including a second assignment.
KEY = re.compile(rb'^[ \t]*(?:export[ \t]+)?HYMEM_LLM_MODEL\b')
ASSIGN = re.compile(
    rb'(?P<prefix>^[ \t]*(?:export[ \t]+)?HYMEM_LLM_MODEL[ \t]*=[ \t]*)'
    rb'(?P<quote>[\x22\x27]?)(?P<value>deepseek-v4-flash|deepseek-flash)(?P=quote)'
    rb'(?P<tail>[ \t]*(?:\#[^\r\n]*|\\[ \t]*)?)(?P<eol>\r?\n|$)'
)

def transform(data, required):
    changed=[]; assignments=0; count=0
    for line in data.splitlines(keepends=True):
        if KEY.match(line):
            match=ASSIGN.fullmatch(line)
            need(match is not None, 'other_or_ambiguous_model_assignment')
            assignments+=1
            if match.group('value')==b'deepseek-v4-flash':
                count+=1
                changed.append(match.group('prefix')+match.group('quote')+
                               b'deepseek-flash'+match.group('quote')+
                               match.group('tail')+match.group('eol'))
            else:
                changed.append(line)
        else:
            changed.append(line)
    need(assignments == 1 if required else assignments <= 1,
         'missing_or_duplicate_model_assignment')
    need(not required or count == 1, 'target_model_already_changed')
    return b''.join(changed),count

def regular(path):
    need(path.resolve()==path, 'unsafe_parent')
    need(path.parent.is_dir() and not path.parent.is_symlink(), 'unsafe_parent')
    need(path.is_file() and not path.is_symlink(), 'unsafe_target')
    st=path.stat()
    need(stat.S_ISREG(st.st_mode) and st.st_nlink==1, 'unsafe_target_metadata')
    need(os.geteuid()==0 or
         (st.st_uid==os.geteuid() and st.st_gid in os.getgroups()),
         'cannot_preserve_owner')
    return st

def exclusive(path, data, mode=0o600):
    fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, mode)
    try:
        os.fchmod(fd,mode)
        with os.fdopen(fd,'wb',closefd=False) as stream:
            stream.write(data);stream.flush();os.fsync(fd)
    finally:
        os.close(fd)

def main():
    need(not sys.flags.optimize, 'optimized_execution_forbidden')
    cfg=json.loads(CONFIG)
    task_home=pathlib.Path(cfg['home']); stage=pathlib.Path(cfg['stage'])
    need(str(task_home)=='/opt/stacks/hermes/instance1/home', 'wrong_home')
    need(str(stage)==str(task_home/'.hermes/benchmarks/r7-deploy-20260925'),
         'wrong_stage')
    need(task_home.is_dir() and task_home.resolve()==task_home and
         stage.is_dir() and stage.resolve()==stage, 'missing_or_unsafe_stage')
    backup=stage/'model-config-backup'
    need(not backup.exists() and not backup.is_symlink(), 'backup_already_exists')
    targets=[]
    for rel,pin in cfg['pins'].items():
        need(rel in ('.hermes/bin/hymem-server-wrapper',
                     '.agent37/hooks/post-restart.sh'), 'wrong_target')
        path=task_home/rel; st=regular(path); before=path.read_bytes()
        check_pin(before,pin)
        after,count=transform(before,True)
        targets.append((rel,path,st,before,after,count))
    need(len(targets)==2, 'target_count')
    env=task_home/'.hermes/.env'
    need(not env.is_symlink(), 'unsafe_env')
    if env.exists():
        st=regular(env); before=env.read_bytes()
        after,count=transform(before,False)
        targets.append(('.hermes/.env',env,st,before,after,count))
    # All files, assignments, identities and destination freshness are checked
    # before any target write. Retest identity immediately before each replace.
    backup.mkdir(mode=0o700)
    os.chmod(backup,0o700)
    rows={}; pending=[]
    for rel,path,st,before,after,count in targets:
        name=rel.replace('/','__').lstrip('.')+'.before'
        saved=backup/name
        exclusive(saved,before)
        need(digest(saved.read_bytes())==digest(before), 'backup_hash_mismatch')
        rows[rel]={'before_sha256':digest(before),'after_sha256':digest(after),
                   'replacements':count,'mode':stat.S_IMODE(st.st_mode),
                   'uid':st.st_uid,'gid':st.st_gid,'backup':name}
    index=backup/'index.json'
    exclusive(index,json.dumps(rows,sort_keys=True).encode()+b'\n')
    for rel,path,st,before,after,count in targets:
        if not count: continue
        fd,name=tempfile.mkstemp(prefix='.r7-model-',dir=path.parent)
        tmp=pathlib.Path(name)
        try:
            os.fchmod(fd,stat.S_IMODE(st.st_mode))
            if os.fstat(fd).st_uid!=st.st_uid or os.fstat(fd).st_gid!=st.st_gid:
                os.fchown(fd,st.st_uid,st.st_gid)
            with os.fdopen(fd,'wb',closefd=False) as stream:
                stream.write(after);stream.flush();os.fsync(fd)
        finally:
            os.close(fd)
        pending.append((path,tmp,st,digest(before),digest(after)))
    for path,tmp,st,before_pin,after_pin in pending:
        current=regular(path)
        need((current.st_dev,current.st_ino,current.st_uid,current.st_gid,
              stat.S_IMODE(current.st_mode))==
             (st.st_dev,st.st_ino,st.st_uid,st.st_gid,stat.S_IMODE(st.st_mode))
             and digest(path.read_bytes())==before_pin, 'target_changed_during_apply')
        os.replace(tmp,path)
        need(digest(path.read_bytes())==after_pin, 'write_hash_mismatch')
    receipt={'status':'passed','target_count':len(targets),
             'changed_count':len(pending),'files':rows,
             'backup_index_sha256':digest(index.read_bytes())}
    exclusive(backup/'receipt.json',json.dumps(receipt,sort_keys=True).encode()+b'\n')
    print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        code=str(exc) if type(exc) is RuntimeError and re.fullmatch('[a-z_]+',str(exc)) else 'apply_failed_inspect_remote'
        print(json.dumps({'status':'failed','failure_code':code},sort_keys=True))
        raise SystemExit(1)
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if sys.flags.optimize:
        raise RuntimeError("optimized_execution_forbidden")
    cfg = {"home": TASK_HOME, "stage": STAGE, "pins": PINS}
    script = "CONFIG=" + repr(json.dumps(cfg, sort_keys=True)) + "\n" + REMOTE
    command = ["ssh", *SSH_OPTIONS, "afrodite",
               "python3 -I -B -c " + shlex.quote(script)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        raise SystemExit("model_override_timeout_inspect_remote_before_retry") from None
    if result.returncode:
        raise SystemExit("model_override_failed_inspect_remote_before_retry")
    try:
        receipt = json.loads(result.stdout)
        if receipt.get("status") != "passed" or receipt.get("target_count") not in (2, 3):
            raise ValueError("invalid_receipt")
    except (ValueError, AttributeError):
        raise SystemExit("model_override_invalid_receipt_inspect_remote") from None
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
