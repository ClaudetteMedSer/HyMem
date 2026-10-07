"""Fenced, code-only staging and offline rehearsal for the Honcho SQLite fix.

This tool never prints provider configuration, process environment, log contents,
or database rows.  It does not contact Afrodite unless a subcommand is run.
Install and process recovery intentionally require a separate reviewed step.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
DB_REL = "hymem/core/db.py"
HELPER_REL = "hymem/core/serialized_sqlite.py"
TEST_REL = "tests/test_serialized_sqlite.py"
EXPECTED_DB_SHA256 = "ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3"
SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite"]


def definition(source: str, name: str) -> str:
    """Select exactly one top-level function, including its decorators."""
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"function_count:{name}")
    node = matches[0]
    first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
    return "".join(source.splitlines(keepends=True)[first - 1:node.end_lineno])


def candidate_payload(db_path: str) -> dict[str, str]:
    original = subprocess.run(
        ["git", "show", f"HEAD:{DB_REL}"], cwd=REPO,
        text=True, capture_output=True, timeout=10, check=True,
    ).stdout
    current = (REPO / DB_REL).read_text()
    original_txn = definition(original, "transaction")
    candidate_txn = definition(current, "transaction")
    if original_txn == candidate_txn or "with operation_scope(conn):" not in candidate_txn:
        raise ValueError("local_transaction_candidate_missing")
    old_connect = definition(original, "connect")
    new_connect = definition(current, "connect")
    marker = "        cached_statements=0 if sys.version_info >= (3, 12) else 128,\n"
    if old_connect.count(marker) != 1 or new_connect.count(marker + "        factory=SerializedConnection,\n") != 1:
        raise ValueError("local_factory_candidate_mismatch")
    if new_connect != old_connect.replace(marker, marker + "        factory=SerializedConnection,\n"):
        raise ValueError("local_connect_has_other_changes")
    helper = (REPO / HELPER_REL).read_text()
    test = (REPO / TEST_REL).read_text()
    ast.parse(helper)
    ast.parse(test)
    return {
        "original_transaction": original_txn,
        "candidate_transaction": candidate_txn,
        "serialized_sqlite": helper,
        "test_serialized_sqlite": test,
        "db_path": db_path,
    }


REMOTE = r"""
import ast,hashlib,json,os,shutil,sqlite3,subprocess,sys,tempfile,time
from pathlib import Path

ACTION=sys.argv[1]
PACKAGE=json.load(sys.stdin)
LIVE=Path('/home/node/HyMem')
LIVE_DB=LIVE/'hymem/core/db.py'
ROOT=Path('/home/node/.hermes/repairs')
PID=12381
EXPECTED='ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3'

def need(ok, code):
    if not ok: raise RuntimeError(code)
def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def sha(path): return sha_bytes(path.read_bytes())
def atomic_json(path, value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x') as handle:
        os.chmod(temporary,0o600)
        json.dump(value,handle,sort_keys=True,separators=(',',':'))
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
def proc_identity():
    proc=Path('/proc')/str(PID)
    stat=(proc/'stat').read_text()
    fields=stat.rpartition(') ')[2].split()
    need(len(fields)>19,'pid_stat_invalid')
    uid=next(line.split()[1] for line in (proc/'status').read_text().splitlines() if line.startswith('Uid:'))
    return {'pid':PID,'start_ticks':int(fields[19]),'uid':int(uid),
            'exe':os.readlink(proc/'exe'),'cwd':os.readlink(proc/'cwd')}
def check_identity():
    identity=proc_identity()
    need(identity['uid']==os.geteuid(),'user_mismatch')
    need(identity['exe']=='/usr/bin/python3.11','exe_mismatch')
    need(identity['cwd']==str(LIVE),'cwd_mismatch')
    return identity
def definition(source,name):
    tree=ast.parse(source)
    matches=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    need(len(matches)==1,'function_count_'+name)
    n=matches[0]
    first=min([n.lineno,*[d.lineno for d in n.decorator_list]])
    return ''.join(source.splitlines(keepends=True)[first-1:n.end_lineno])
def replace_one(source,old,new,code):
    need(source.count(old)==1,code)
    return source.replace(old,new,1)
def transform(source,package):
    need(sha_bytes(source.encode())==EXPECTED,'live_source_changed')
    old_txn=package['original_transaction']
    new_txn=package['candidate_transaction']
    need(definition(source,'transaction')==old_txn,'transaction_source_changed')
    source=replace_one(source,old_txn,new_txn,'transaction_match_count')
    anchor='from hymem.deadline import check_current_deadline\n'
    source=replace_one(source,anchor,anchor+'from hymem.core.serialized_sqlite import SerializedConnection, operation_scope\n','import_match_count')
    connect=definition(source,'connect')
    marker='        cached_statements=0 if sys.version_info >= (3, 12) else 128,\n'
    need(connect.count(marker)==1 and 'factory=SerializedConnection' not in connect,'connect_factory_mismatch')
    modified=connect.replace(marker,marker+'        factory=SerializedConnection,\n',1)
    source=replace_one(source,connect,modified,'connect_match_count')
    ast.parse(source)
    need(definition(source,'transaction')==new_txn,'transaction_result_mismatch')
    need(definition(source,'connect')==modified,'connect_result_mismatch')
    return source
def stage_path():
    stage=Path(PACKAGE.get('stage',''))
    need(stage.parent==ROOT and stage.name.startswith('honcho-sqlite-'),'stage_path_invalid')
    need(stage.is_dir() and not stage.is_symlink(),'stage_missing')
    return stage
def staged_receipt(stage):
    value=json.loads((stage/'stage.json').read_text())
    need(value['expected_original_sha256']==EXPECTED,'receipt_original_mismatch')
    need(sha(stage/'hymem/core/db.py')==value['candidate_db_sha256'],'candidate_db_changed')
    need(sha(stage/'hymem/core/serialized_sqlite.py')==value['helper_sha256'],'helper_changed')
    return value
def private_backup(db_path,target):
    # Child process owns the SQLite backup; its parent has a finite deadline.
    code='''
import os,sqlite3,sys
source=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True,timeout=5)
target=sqlite3.connect(sys.argv[2],timeout=5)
try: source.backup(target,pages=256,sleep=.05)
finally: target.close();source.close()
os.chmod(sys.argv[2],0o600)
check=sqlite3.connect('file:'+sys.argv[2]+'?mode=ro',uri=True)
try:
    if check.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise RuntimeError('integrity')
finally: check.close()
'''
    p=subprocess.run(['/usr/bin/python3.11','-c',code,str(db_path),str(target)],
                     stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                     timeout=90)
    need(p.returncode==0,'backup_failed')
def code_only(directory,names):
    ignored=[]
    for name in names:
        path=Path(directory)/name
        if path.is_symlink() or name=='__pycache__': ignored.append(name)
        elif path.is_file() and path.suffix not in ('.py','.sql','.md'): ignored.append(name)
    return ignored

if ACTION=='inspect':
    identity=check_identity()
    db_path=Path(PACKAGE['db_path'])
    need(db_path.is_absolute() and db_path.is_file() and not db_path.is_symlink(),'db_path_invalid')
    need(db_path.name=='hymem.sqlite','db_name_invalid')
    db_stat=db_path.stat()
    print(json.dumps({'action':'inspect','identity':identity,
                      'db_sha256':sha(LIVE_DB),'expected_db_sha256':EXPECTED,
                      'store':{'path':str(db_path),'bytes':db_stat.st_size,
                               'inode':db_stat.st_ino,'owner_uid':db_stat.st_uid}},sort_keys=True))
elif ACTION=='stage':
    identity=check_identity()
    need(sha(LIVE_DB)==EXPECTED,'live_source_changed')
    db_path=Path(PACKAGE['db_path'])
    need(db_path.is_absolute() and db_path.is_file() and not db_path.is_symlink(),'db_path_invalid')
    need(db_path.name=='hymem.sqlite','db_name_invalid')
    ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
    need(ROOT.stat().st_uid==os.geteuid() and (ROOT.stat().st_mode & 0o077)==0,'repair_root_permissions')
    stage=Path(tempfile.mkdtemp(prefix='honcho-sqlite-',dir=ROOT))
    os.chmod(stage,0o700)
    original=LIVE_DB.read_text()
    candidate=transform(original,PACKAGE)
    source_root=LIVE/'hymem'
    shutil.copytree(source_root,stage/'hymem',ignore=code_only)
    (stage/'hymem/core/db.py').write_text(candidate)
    (stage/'hymem/core/serialized_sqlite.py').write_text(PACKAGE['serialized_sqlite'])
    (stage/'tests').mkdir(mode=0o700)
    (stage/'tests/test_serialized_sqlite.py').write_text(PACKAGE['test_serialized_sqlite'])
    (stage/'original-db.py').write_text(original)
    for path in (stage/'hymem/core/db.py',stage/'hymem/core/serialized_sqlite.py',
                 stage/'tests/test_serialized_sqlite.py',stage/'original-db.py'):
        os.chmod(path,0o600)
    private_backup(db_path,stage/'backup.sqlite')
    receipt={'action':'stage','stage':str(stage),'identity':identity,
             'expected_original_sha256':EXPECTED,'candidate_db_sha256':sha(stage/'hymem/core/db.py'),
             'helper_sha256':sha(stage/'hymem/core/serialized_sqlite.py'),
             'backup_sha256':sha(stage/'backup.sqlite'),'db_path':str(db_path),
             'test_sha256':sha(stage/'tests/test_serialized_sqlite.py')}
    atomic_json(stage/'stage.json',receipt)
    print(json.dumps(receipt,sort_keys=True))
elif ACTION=='rehearse':
    stage=stage_path(); receipt=staged_receipt(stage)
    need(sha(stage/'backup.sqlite')==receipt['backup_sha256'],'backup_changed')
    rehearsal_db=stage/'rehearsal.sqlite'
    need(not rehearsal_db.exists(),'rehearsal_already_exists')
    shutil.copy2(stage/'backup.sqlite',rehearsal_db)
    os.chmod(rehearsal_db,0o600)
    # No provider construction or network. A subprocess gives this rehearsal a
    # hard parent deadline even if the Python/SQLite lock inversion recurs.
    code='''
import json,sqlite3,sys,threading,time
from pathlib import Path
from hymem.core.serialized_sqlite import SerializedConnection
from hymem.core import db
from hymem.query.augment import build_token_overlap_index
barrier=threading.Event();errors=[]
c=sqlite3.connect(':memory:',factory=SerializedConnection,check_same_thread=False,isolation_level=None)
c.create_function('pause_udf',0,lambda:(barrier.set(),time.sleep(.02),1)[2])
def slow():
    try:
        for _ in range(150): c.execute('SELECT pause_udf()').fetchone()
    except BaseException as exc: errors.append(type(exc).__name__)
def concurrent():
    barrier.wait(2)
    try:
        for _ in range(150): c.execute('SELECT 1').fetchone()
    except BaseException as exc: errors.append(type(exc).__name__)
a=threading.Thread(target=slow);b=threading.Thread(target=concurrent)
a.start();b.start();a.join(12);b.join(12)
if a.is_alive() or b.is_alive() or errors: raise RuntimeError('synthetic_concurrency_failed')
c.close()
store=db.connect(Path(sys.argv[1]));store.execute('PRAGMA query_only=ON')
tables=('messages','chunks','knowledge_graph','token_overlap_index')
def counts():
    return {name:store.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in tables}
before=counts()
errors=[]
def scan():
    try: build_token_overlap_index(store)
    except BaseException as exc: errors.append(type(exc).__name__)
a=threading.Thread(target=scan);b=threading.Thread(target=scan)
a.start();b.start();a.join(20);b.join(20)
if a.is_alive() or b.is_alive() or errors: raise RuntimeError('clone_scan_failed:'+','.join(errors))
if counts()!=before: raise RuntimeError('clone_counts_changed')
if store.execute('PRAGMA quick_check').fetchone()[0]!='ok': raise RuntimeError('clone_quick_check_failed')
store.close();print(json.dumps({'status':'PASS','counts':before},sort_keys=True))
'''
    env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(stage),'PYTHONDONTWRITEBYTECODE':'1'}
    command=['/home/node/hymem-env/bin/python','-c',code,str(rehearsal_db)]
    try:
        p=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,text=True,timeout=55,env=env,cwd=stage)
    except subprocess.TimeoutExpired as exc:
        (stage/'rehearsal-stderr.private').write_bytes(exc.stderr or b'')
        os.chmod(stage/'rehearsal-stderr.private',0o600)
        raise RuntimeError('rehearsal_timeout')
    (stage/'rehearsal-stderr.private').write_text(p.stderr)
    os.chmod(stage/'rehearsal-stderr.private',0o600)
    need(p.returncode==0,'rehearsal_failed')
    result_data=json.loads(p.stdout)
    need(result_data.get('status')=='PASS','rehearsal_result_invalid')
    result={'action':'rehearse','stage':str(stage),'candidate_db_sha256':receipt['candidate_db_sha256'],
            'helper_sha256':receipt['helper_sha256'],'backup_sha256':receipt['backup_sha256'],
            'synthetic_concurrency':'passed','clone_token_overlap':'passed',
            'logical_counts':result_data['counts'],'quick_check':'ok'}
    atomic_json(stage/'rehearse.json',result)
    print(json.dumps(result,sort_keys=True))
elif ACTION=='status':
    stage=stage_path(); receipt=staged_receipt(stage)
    identity=check_identity()
    rehearsed=(stage/'rehearse.json').is_file()
    print(json.dumps({'action':'status','stage':str(stage),'source_sha256':sha(LIVE_DB),
                      'identity_unchanged':identity==receipt['identity'],
                      'rehearsed':rehearsed,'candidate_db_sha256':receipt['candidate_db_sha256']},sort_keys=True))
else: raise RuntimeError('unknown_action')
"""


def run_remote(action: str, package: dict[str, str], timeout: int) -> int:
    command = SSH + [shlex.join(["docker", "exec", "-i", "-u", "node", "hermes-1",
                                 "/usr/bin/python3.11", "-c", REMOTE, action])]
    try:
        result = subprocess.run(command, input=json.dumps(package), text=True,
                                capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(json.dumps({"action": action, "error": "remote_timeout"}))
        return 1
    if result.returncode:
        # SSH/Docker/Python stderr may include private paths or environment.
        print(json.dumps({"action": action, "error": "remote_failed",
                          "returncode": result.returncode}))
        return 1
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(json.dumps({"action": action, "error": "invalid_remote_receipt"}))
        return 1
    print(json.dumps(output, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "stage", "rehearse", "status"))
    parser.add_argument("--stage", help="private stage path returned by stage")
    parser.add_argument("--db-path", default="/home/node/.hermes/hymem.sqlite",
                        help="in-container HyMem SQLite path for stage")
    args = parser.parse_args()
    if args.action in ("rehearse", "status") and not args.stage:
        parser.error("--stage is required")
    package = ({"stage": args.stage} if args.stage else {"db_path": args.db_path})
    if args.action == "stage":
        package = candidate_payload(args.db_path)
    return run_remote(args.action, package, 180 if args.action == "stage" else 90)


if __name__ == "__main__":
    raise SystemExit(main())
