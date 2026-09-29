"""Guarded Hermes1 R7 deployment stages. Run only after operator review.

Each invocation performs exactly one remote action. A failure never restores a
database or restarts the service automatically. Receipts contain metadata only.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys


MANIFEST_PIN = "1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8"
IMAGE = "sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5"
TASK_HOME = "/opt/stacks/hermes/instance1/home"
LIVE = TASK_HOME + "/HyMem"
VENV = TASK_HOME + "/hymem-env"
DB = TASK_HOME + "/.hermes/hymem.sqlite"
SOURCE = TASK_HOME + "/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/verification"
MANIFEST = TASK_HOME + "/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/diag/manifest.json"
STAGE = TASK_HOME + "/.hermes/benchmarks/r7-deploy-20260925"
CONTAINER = "hermes-1"
WHEELS = {
    "setuptools-80.9.0-py3-none-any.whl",
    "wheel-0.45.1-py3-none-any.whl",
}
SSH_OPTIONS = (
    "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
    "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
)
TIMEOUT = {"stop": 900, "install": 900, "migrate": 600, "start": 180}


REMOTE = r"""
import hashlib,json,os,pathlib,re,shutil,sqlite3,stat,subprocess,sys
from urllib.parse import quote

if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
C=json.loads(CONFIG)
home=pathlib.Path(C['home']); live=pathlib.Path(C['live'])
source=pathlib.Path(C['source']); manifest_path=pathlib.Path(C['manifest'])
venv=pathlib.Path(C['venv']); db_path=pathlib.Path(C['db'])
stage=pathlib.Path(C['stage']); action=C['action']; container=C['container']
assert action in ('stop','install','migrate','start')
assert stage.is_dir() and not stage.is_symlink()
assert home.is_dir() and live.is_dir() and venv.is_dir()
assert db_path.is_file() and not db_path.is_symlink()
assert source.is_dir() and not source.is_symlink()

def need(ok, code):
    if not ok: raise RuntimeError(code)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def safe_rel(name):
    p=pathlib.PurePosixPath(name)
    need(name==p.as_posix() and not p.is_absolute() and '..' not in p.parts,
         'unsafe_manifest_path')
    need(not any(part in ('.hermes','.env','venv','hymem-env') for part in p.parts),
         'protected_manifest_path')
    return p

def manifest():
    need(sha(manifest_path)==C['manifest_pin'],'manifest_pin_mismatch')
    raw=json.loads(manifest_path.read_bytes()); files={}
    for group in ('source_sha256','test_sha256','auxiliary_sha256'):
        for name,pin in raw[group].items():
            safe_rel(name); need(name not in files and re.fullmatch('[0-9a-f]{64}',pin),
                                'manifest_duplicate_or_pin')
            files[name]=pin
    need(len(files)==479,'manifest_file_count')
    actual={}
    for path in source.rglob('*'):
        need(not path.is_symlink(),'source_symlink')
        if path.is_file():
            name=path.relative_to(source).as_posix()
            actual[name]=sha(path)
    need(actual==files,'candidate_tree_mismatch')
    return files

def receipt(name, body):
    path=stage/(name+'.json')
    with path.open('x',encoding='utf-8') as stream:
        json.dump(body,stream,sort_keys=True)
    return body

def prior(name):
    path=stage/(name+'.json')
    need(path.is_file() and not path.is_symlink(),'missing_'+name+'_receipt')
    value=json.loads(path.read_bytes())
    need(value.get('status')=='passed' and value.get('manifest_sha256')==C['manifest_pin'],
         'invalid_'+name+'_receipt')
    return value

def run(command,timeout,code):
    try:
        result=subprocess.run(command,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(code+'_timeout_inspect_before_retry') from None
    need(result.returncode==0,code+'_failed_inspect_before_retry')
    return result.stdout

def inspect():
    raw=run(['docker','inspect',container],30,'docker_inspect')
    rows=json.loads(raw)
    need(len(rows)==1 and rows[0]['Name']=='/'+container,'container_identity')
    return rows[0]

def stopped():
    state=inspect()['State']
    need(state['Status']=='exited' and not state['Running'] and not state['OOMKilled'],
         'service_not_safely_stopped')
    need(state['ExitCode']!=137,'forced_stop_detected')
    return state

def backup_db():
    target=stage/'pre-deploy.sqlite'
    need(not target.exists(),'fresh_backup_already_exists')
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    os.close(fd)
    src=sqlite3.connect('file:'+quote(str(db_path),safe='/')+'?mode=ro',uri=True,timeout=30)
    dst=sqlite3.connect(target)
    try: src.backup(dst,pages=1024,sleep=0.1)
    finally: dst.close();src.close()
    check=r'''
import json,pathlib,sys
sys.path.insert(0,'/home/node/HyMem')
from hymem.core import db
conn=db.connect(pathlib.Path('/backup/pre-deploy.sqlite'))
try:
    schema=db.schema_version(conn)
    assert schema==61
    assert [row[0] for row in conn.execute('PRAGMA integrity_check')]==['ok']
    assert conn.execute('PRAGMA foreign_key_check').fetchone() is None
    print(json.dumps({'schema':schema,'integrity_ok':True,'foreign_keys_ok':True}))
finally: conn.close()
'''
    result=json.loads(docker_python(check,label='stop-backup',source_host=source,
                                    rw_stage=True,timeout=120))
    need(result=={'schema':61,'integrity_ok':True,'foreign_keys_ok':True},
         'backup_integrity_or_schema')
    return {'sha256':sha(target),'bytes':target.stat().st_size,
            'schema_version':result['schema']}

def backup_source(files):
    backup=stage/'source-backup'
    backup.mkdir(mode=0o700)
    items={}
    for name in sorted(files):
        rel=safe_rel(name); current=live.joinpath(*rel.parts)
        for parent in current.parents:
            need(not parent.is_symlink(),'live_path_symlink')
            if parent==live: break
        need(not current.is_symlink(),'live_file_symlink')
        if current.exists():
            need(current.is_file(),'live_path_not_file')
            target=backup.joinpath(*rel.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(current,target)
            items[name]={'present':True,'sha256':sha(current),
                         'mode':stat.S_IMODE(current.stat().st_mode)}
            need(sha(target)==items[name]['sha256'],'source_backup_mismatch')
        else: items[name]={'present':False}
    with (stage/'source-backup-index.json').open('x') as stream:
        json.dump(items,stream,sort_keys=True)
    need(len(items)==479,'source_backup_incomplete')
    return {'present':sum(v['present'] for v in items.values()),
            'missing':sum(not v['present'] for v in items.values()),
            'index_sha256':sha(stage/'source-backup-index.json')}

def backup_venv():
    backup=stage/'venv-backup';backup.mkdir(mode=0o700)
    candidates=[]
    for site in sorted(venv.glob('lib/python*/site-packages')):
        for pattern in ('hymem*','__editable__*hymem*'):
            candidates.extend(site.glob(pattern))
    for pattern in ('hymem*','honcho*'):
        candidates.extend((venv/'bin').glob(pattern))
    candidates=sorted(set(candidates))
    need(bool(candidates),'venv_distribution_not_found')
    items={}
    for path in candidates:
        need(not path.is_symlink(),'venv_target_symlink')
        rel=path.relative_to(venv);target=backup/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        if path.is_dir():
            shutil.copytree(path,target,symlinks=True)
            need(not any(p.is_symlink() for p in target.rglob('*')),'venv_backup_symlink')
            hashes={p.relative_to(path).as_posix():sha(p) for p in path.rglob('*') if p.is_file()}
            need(all(sha(target/name)==pin for name,pin in hashes.items()),
                 'venv_backup_mismatch')
            items[rel.as_posix()]={'directory':True,'files':hashes}
        else:
            need(path.is_file(),'venv_target_not_file')
            shutil.copy2(path,target)
            items[rel.as_posix()]={'directory':False,'sha256':sha(path)}
            need(sha(target)==items[rel.as_posix()]['sha256'],'venv_backup_mismatch')
    with (stage/'venv-backup-index.json').open('x') as stream:
        json.dump(items,stream,sort_keys=True)
    return {'entries':len(items),'index_sha256':sha(stage/'venv-backup-index.json')}

def verify_backups(stop):
    need(sha(stage/'pre-deploy.sqlite')==stop['db_backup']['sha256'],
         'db_backup_changed')
    need(sha(stage/'source-backup-index.json')==stop['source_backup']['index_sha256'],
         'source_backup_index_changed')
    need(sha(stage/'venv-backup-index.json')==stop['venv_backup']['index_sha256'],
         'venv_backup_index_changed')
    source_items=json.loads((stage/'source-backup-index.json').read_bytes())
    need(len(source_items)==479,'source_backup_inventory_changed')
    for name,item in source_items.items():
        path=stage/'source-backup'/safe_rel(name)
        need((not path.exists()) if not item['present'] else
             (path.is_file() and sha(path)==item['sha256']),
             'source_backup_content_changed')
    venv_items=json.loads((stage/'venv-backup-index.json').read_bytes())
    for name,item in venv_items.items():
        path=stage/'venv-backup'/safe_rel(name)
        if item['directory']:
            actual={p.relative_to(path).as_posix():sha(p)
                    for p in path.rglob('*') if p.is_file()}
            need(actual==item['files'],'venv_backup_content_changed')
        else:
            need(path.is_file() and sha(path)==item['sha256'],
                 'venv_backup_content_changed')

def confirm_idle():
    port=C['health_port']
    need(type(port) is int and 1<=port<=65535,'health_port_required')
    process_table=run(['docker','top',container,'-eo','pid,args'],15,'honcho_process')
    processes=[]
    for line in process_table.decode('utf-8').splitlines()[1:]:
        columns=line.strip().split(None,1)
        if len(columns)==2 and columns[0].isdigit():
            processes.append((int(columns[0]),columns[1]))
    need(any(pid==C['idle_confirmed_pid'] and
             ('hymem' in command.lower() or 'honcho' in command.lower())
             for pid,command in processes),'honcho_pid_changed')
    probe=("import json,urllib.request;"+
           "b='http://127.0.0.1:"+str(port)+"';"+
           "h=json.load(urllib.request.urlopen(b+'/health',timeout=5));"+
           "v=json.load(urllib.request.urlopen(b+'/dream-status',timeout=5));"+
           "print(json.dumps({'health':h,'in_progress':v['in_progress']}))")
    output=run(['docker','exec',container,'/home/node/hymem-env/bin/python3',
                '-I','-B','-c',probe],15,'idle_status')
    status=json.loads(output)
    need(status=={'health':{'status':'ok','backend':'hymem'},
                  'in_progress':False},'service_not_idle')

def copy_release(files):
    previous=json.loads((stage/'source-backup-index.json').read_bytes())
    need(set(previous)==set(files),'backup_path_set_changed')
    for name,pin in sorted(files.items()):
        rel=safe_rel(name); src=source.joinpath(*rel.parts)
        dst=live.joinpath(*rel.parts)
        for parent in dst.parents:
            need(not parent.is_symlink(),'live_path_symlink')
            if parent==live: break
        need(not dst.is_symlink(),'live_file_symlink')
        dst.parent.mkdir(parents=True,exist_ok=True)
        temp=dst.with_name(dst.name+'.r7-install-tmp')
        need(not temp.exists(),'install_temp_exists')
        with src.open('rb') as incoming, temp.open('xb') as outgoing:
            shutil.copyfileobj(incoming,outgoing)
        os.chmod(temp,previous[name]['mode'] if previous[name]['present'] else 0o644)
        need(sha(temp)==pin,'installed_temp_hash_mismatch')
        os.replace(temp,dst)
    need(all(sha(live/name)==pin for name,pin in files.items()),
         'installed_tree_hash_mismatch')

def docker_python(script,*,label,rw_source=False,rw_venv=False,rw_db=False,wheels=False,
                  rw_stage=False,source_host=None,timeout=600):
    need(label in ('stop-backup','install','migrate','start-check'),
         'unknown_offline_container_role')
    command=['docker','run','--rm','--name','hymem-r7-deploy-'+label,
             '--pull','never','--network','none',
             '--user','1000:1000','--read-only','--cap-drop','ALL',
             '--security-opt','no-new-privileges','--pids-limit','128',
             '--memory','2g','--cpus','2',
             '--tmpfs','/tmp:rw,nosuid,size=512m',
             '--env','TMPDIR=/tmp',
             '--env','PYTHONDONTWRITEBYTECODE=1']
    mounts=[(source_host or live,'/home/node/HyMem',rw_source),
            (venv,'/home/node/hymem-env',rw_venv)]
    if rw_db: mounts.append((db_path.parent,'/home/node/.hermes',True))
    if wheels: mounts.append((stage/'build-wheels','/build-wheels',False))
    if rw_stage: mounts.append((stage,'/backup',True))
    for host,destination,rw in mounts:
        need(host.is_dir() and not host.is_symlink(),'mount_source_invalid')
        command.extend(['--mount','type=bind,src='+str(host)+',dst='+destination+
                        ('' if rw else ',readonly')])
    command.extend(['--workdir','/home/node/HyMem',
                    '--entrypoint','/home/node/hymem-env/bin/python3',C['image'],
                    '-I','-B','-c',script])
    return run(command,timeout,'offline_container')

def do_stop(files):
    need(not (stage/'stop.json').exists() and not (stage/'stop-intent.json').exists(),
         'stop_already_attempted')
    need(not (stage/'pre-deploy.sqlite').exists() and
         not (stage/'source-backup').exists() and
         not (stage/'venv-backup').exists(),
         'deployment_backup_already_exists')
    rehearsal=json.loads((stage/'rehearsal.json').read_bytes())
    need(rehearsal.get('status')=='passed' and rehearsal.get('after_schema')==63
         and rehearsal.get('original_tables_verified')==117 and
         rehearsal.get('changed_existing_tables')==[], 'rehearsal_not_passed')
    need((stage/'pre-rehearsal.sqlite').is_file() and (stage/'rehearsal.sqlite').is_file(),
         'rehearsal_backups_absent')
    need(C['idle_confirmed_pid']>0,'idle_confirmation_required')
    before=inspect();need(before['State']['Running'],'service_not_running')
    confirm_idle()
    receipt('stop-intent',{'status':'attempted','manifest_sha256':C['manifest_pin'],
                           'idle_confirmed_pid':C['idle_confirmed_pid'],
                           'container_id':before['Id']})
    run(['docker','stop','-t','120',container],150,'graceful_stop')
    after=stopped();need(after['ExitCode']!=137,'forced_stop_detected')
    fresh=backup_db()
    source_backup=backup_source(files)
    venv_backup=backup_venv()
    return receipt('stop',{'status':'passed','manifest_sha256':C['manifest_pin'],
                           'container_id':before['Id'],'exit_code':after['ExitCode'],
                           'idle_status':{'in_progress':False},
                           'db_backup':fresh,'source_backup':source_backup,
                           'venv_backup':venv_backup})

def do_install(files):
    need(not (stage/'install.json').exists() and not (stage/'install-intent.json').exists(),
         'install_already_attempted')
    stop=prior('stop');stopped();verify_backups(stop)
    wheels=stage/'build-wheels'
    need(wheels.is_dir() and not wheels.is_symlink(),'build_wheels_missing')
    actual={p.name for p in wheels.iterdir()}
    need(actual==set(C['wheel_pins'])==set(C['wheel_names']),
         'build_wheel_set_mismatch')
    for name,pin in C['wheel_pins'].items():
        path=wheels/name
        need(path.is_file() and not path.is_symlink() and sha(path)==pin,
             'build_wheel_hash_mismatch')
    receipt('install-intent',{'status':'attempted','manifest_sha256':C['manifest_pin'],
                              'db_backup_sha256':stop['db_backup']['sha256'],
                              'wheel_sha256':C['wheel_pins']})
    copy_release(files)
    pip="import subprocess,sys;raise SystemExit(subprocess.run([sys.executable,'-I','-B','-m','pip','--isolated','install','--no-deps','--no-index','--no-cache-dir','--find-links','/build-wheels','--disable-pip-version-check','-e','/home/node/HyMem'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=600).returncode)"
    docker_python(pip,label='install',rw_source=True,rw_venv=True,
                  wheels=True,timeout=660)
    need(all(sha(live/name)==pin for name,pin in files.items()),
         'post_install_source_hash_mismatch')
    return receipt('install',{'status':'passed','manifest_sha256':C['manifest_pin'],
                              'source_files_verified':len(files),
                              'db_backup_sha256':stop['db_backup']['sha256'],
                              'wheel_sha256':C['wheel_pins']})

MIGRATE=r'''
import json,pathlib,sys
sys.path.insert(0,'/home/node/HyMem')
from hymem.core import db
path=pathlib.Path('/home/node/.hermes/hymem.sqlite')
conn=None
try:
    conn=db.connect(path); before=db.schema_version(conn)
    assert before in (61,62,63)
    db.initialize(conn); after=db.schema_version(conn)
    assert after==63
    assert [row[0] for row in conn.execute('PRAGMA integrity_check')]==['ok']
    assert conn.execute('PRAGMA foreign_key_check').fetchone() is None
    conn.close();conn=None
    conn=db.connect(path);db.initialize(conn)
    assert db.schema_version(conn)==63
    assert [row[0] for row in conn.execute('PRAGMA integrity_check')]==['ok']
    assert conn.execute('PRAGMA foreign_key_check').fetchone() is None
    print(json.dumps({'status':'passed','schema_before':before,'schema_after':63,
                      'integrity_ok':True,'foreign_keys_ok':True,'reopen_verified':True}))
finally:
    if conn is not None: conn.close()
'''

def do_migrate(files):
    need(not (stage/'migrate.json').exists() and not (stage/'migrate-intent.json').exists(),
         'migrate_already_attempted')
    stop=prior('stop');install=prior('install');stopped();verify_backups(stop)
    need(install['db_backup_sha256']==stop['db_backup']['sha256'],
         'installation_backup_binding')
    need(all(sha(live/name)==pin for name,pin in files.items()),
         'pre_migration_source_hash_mismatch')
    receipt('migrate-intent',{'status':'attempted','manifest_sha256':C['manifest_pin'],
                              'db_backup_sha256':stop['db_backup']['sha256']})
    output=docker_python(MIGRATE,label='migrate',rw_db=True,timeout=480)
    data=json.loads(output)
    need(data=={'status':'passed','schema_before':data['schema_before'],
                'schema_after':63,'integrity_ok':True,'foreign_keys_ok':True,
                'reopen_verified':True},'migration_result_invalid')
    return receipt('migrate',{'status':'passed','manifest_sha256':C['manifest_pin'],
                              'db_backup_sha256':stop['db_backup']['sha256'],
                              'schema_before':data['schema_before'],
                              'schema_after':63,'integrity_ok':True,
                              'foreign_keys_ok':True,'reopen_verified':True})

def do_start(files):
    need(not (stage/'start.json').exists() and not (stage/'start-intent.json').exists(),
         'start_already_attempted')
    stop=prior('stop');install=prior('install');migrate=prior('migrate')
    stopped();verify_backups(stop)
    need(install['db_backup_sha256']==migrate['db_backup_sha256']==stop['db_backup']['sha256'],
         'stage_backup_binding')
    need(migrate['schema_after']==63 and migrate['integrity_ok'] and
         migrate['foreign_keys_ok'] and migrate['reopen_verified'],
         'migration_not_verified')
    need(all(sha(live/name)==pin for name,pin in files.items()),
         'pre_start_source_hash_mismatch')
    check=r'''
import json,pathlib,sys
sys.path.insert(0,'/home/node/HyMem')
from hymem.core import db
conn=db.connect(pathlib.Path('/home/node/.hermes/hymem.sqlite'))
try:
    schema=db.schema_version(conn)
    assert schema==63
    assert [row[0] for row in conn.execute('PRAGMA integrity_check')]==['ok']
    assert conn.execute('PRAGMA foreign_key_check').fetchone() is None
    print(json.dumps({'schema':schema,'integrity_ok':True,'foreign_keys_ok':True}))
finally: conn.close()
'''
    result=json.loads(docker_python(check,label='start-check',rw_db=True,timeout=120))
    need(result=={'schema':63,'integrity_ok':True,'foreign_keys_ok':True},
         'pre_start_integrity_or_schema')
    receipt('start-intent',{'status':'attempted','manifest_sha256':C['manifest_pin'],
                            'container_id':stop['container_id']})
    output=run(['docker','start',container],90,'service_start')
    need(output.strip().decode()==container,'service_start_identity')
    after=inspect()
    need(after['Id']==stop['container_id'] and after['State']['Running'],
         'service_not_running_after_start')
    return receipt('start',{'status':'passed','manifest_sha256':C['manifest_pin'],
                            'container_id':after['Id'],'schema':63,
                            'source_files_verified':len(files)})

try:
    files=manifest()
    result={'stop':do_stop,'install':do_install,'migrate':do_migrate,
            'start':do_start}[action](files)
    print(json.dumps(result,sort_keys=True))
except BaseException as exc:
    # Never emit subprocess output, environment, database values, or traceback.
    failure={'status':'failed','action':action,'error_type':type(exc).__name__,
             'manifest_sha256':C['manifest_pin']}
    if type(exc) is RuntimeError and re.fullmatch('[a-z_]+',str(exc)):
        failure['failure_code']=str(exc)
    try:
        with (stage/(action+'-failed.json')).open('x') as stream:
            json.dump(failure,stream,sort_keys=True)
    except BaseException: pass
    print(json.dumps(failure,sort_keys=True))
    raise SystemExit(1)
"""


def configuration(args: argparse.Namespace) -> dict[str, object]:
    wheel_pins: dict[str, str] = {}
    if args.action == "stop":
        if args.idle_confirmed_pid is None or args.idle_confirmed_pid <= 0:
            raise ValueError("stop_requires_independently_confirmed_idle_pid")
        if args.health_port is None or not 1 <= args.health_port <= 65535:
            raise ValueError("stop_requires_live_health_port")
    if args.action == "install":
        wheel_pins = {
            "setuptools-80.9.0-py3-none-any.whl": args.setuptools_sha256,
            "wheel-0.45.1-py3-none-any.whl": args.wheel_sha256,
        }
        if set(wheel_pins) != WHEELS or any(
            not isinstance(pin, str) or re.fullmatch("[0-9a-f]{64}", pin) is None
            for pin in wheel_pins.values()
        ):
            raise ValueError("install_requires_pinned_offline_build_wheels")
    return {
        "action": args.action, "home": TASK_HOME, "live": LIVE, "venv": VENV,
        "db": DB, "source": SOURCE, "manifest": MANIFEST, "stage": STAGE,
        "container": CONTAINER, "manifest_pin": MANIFEST_PIN, "image": IMAGE,
        "idle_confirmed_pid": args.idle_confirmed_pid or 0,
        "health_port": args.health_port or 0,
        "wheel_names": sorted(WHEELS), "wheel_pins": wheel_pins,
    }


def main(argv: list[str] | None = None) -> int:
    if sys.flags.optimize:
        raise RuntimeError("optimized_execution_forbidden")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=tuple(TIMEOUT))
    parser.add_argument("--idle-confirmed-pid", type=int)
    parser.add_argument("--health-port", type=int)
    parser.add_argument("--setuptools-sha256")
    parser.add_argument("--wheel-sha256")
    args = parser.parse_args(argv)
    try:
        config = configuration(args)
    except ValueError as exc:
        parser.error(str(exc))
    script = "CONFIG=" + repr(json.dumps(config, sort_keys=True)) + "\n" + REMOTE
    command = ["ssh", *SSH_OPTIONS, "afrodite",
               "python3 -I -B -c " + shlex.quote(script)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=TIMEOUT[args.action])
    except subprocess.TimeoutExpired:
        raise SystemExit("deployment_stage_timeout_inspect_receipts_before_retry") from None
    if result.returncode:
        raise SystemExit("deployment_stage_failed_inspect_receipts_before_retry")
    try:
        body = json.loads(result.stdout)
        if body.get("status") != "passed" or body.get("manifest_sha256") != MANIFEST_PIN:
            raise ValueError("invalid_receipt")
    except (ValueError, AttributeError):
        raise SystemExit("deployment_stage_receipt_invalid_inspect_remote") from None
    print(json.dumps(body, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
