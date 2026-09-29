"""Pinned single-file Hermes1 claim-conflict deployment preflight and apply.

No restart is performed here. Both actions require the service container to be
normally exited. Preflight creates private backups and verification receipts;
apply atomically replaces only hymem/dreaming/phase1.py.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys

SNAP = "/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky"
OLD = "bea40b7a6565542861fadf9683dcbfd2bc70fe51dcf3f6b5bfc5ece5be6488ee"
NEW = "31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136"
DOCTOR = "2ac786c6590aa7fbece726e6380981906d514fefe9f664a3ca4f7b1d1de7c15b"
MANIFEST = "1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8"
IMAGE = "sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5"
SSH = ("ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=2", "afrodite")

REMOTE = r"""
import hashlib,json,os,pathlib,shutil,sqlite3,stat,subprocess,sys,tempfile
from urllib.parse import quote
if sys.flags.optimize: raise RuntimeError('optimized_execution_forbidden')
C=json.loads(CONFIG)
home=pathlib.Path('/opt/stacks/hermes/instance1/home')
live=home/'HyMem'; target=live/'hymem/dreaming/phase1.py'
snapshot=pathlib.Path(C['snapshot']); stage=snapshot/'deploy-v1'
candidate=snapshot/'offline-compare-v1/candidate/hymem/dreaming/phase1.py'
compare=snapshot/'offline-compare-v1/result.json'
store=snapshot/'offline-compare-v1/fixed-store-verification.json'
manifest_path=home/'.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/diag/manifest.json'
db=home/'.hermes/hymem.sqlite'; runtime=home/'hymem-env'

def need(ok,code):
    if not ok: raise RuntimeError(code)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def regular(path):
    info=path.lstat()
    need(stat.S_ISREG(info.st_mode) and not path.is_symlink(),'unsafe_file')
    return info

def receipt(name,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    need(len(raw)<8192,'oversized_receipt')
    fd=os.open(stage/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return value

def read(name):
    path=stage/name;regular(path)
    value=json.loads(path.read_bytes())
    need(isinstance(value,dict),'invalid_receipt')
    return value

def stopped():
    raw=subprocess.run(['docker','inspect','hermes-1'],capture_output=True,timeout=30)
    need(raw.returncode==0 and len(raw.stdout)<200000,'service_inspect_failed')
    obj=json.loads(raw.stdout)[0];state=obj['State']
    need(obj['Name']=='/hermes-1' and obj['Image']==C['image'] and state['Status']=='exited'
         and not state['Running'] and state['Pid']==0
         and not state['OOMKilled'] and state['ExitCode']!=137,
         'service_not_normally_stopped')
    return {'status':state['Status'],'exit_code':state['ExitCode'],
            'container_id':obj['Id']}

def source_pins(expected_phase1):
    need(sha(manifest_path)==C['manifest'],'manifest_pin_mismatch')
    manifest=json.loads(manifest_path.read_bytes());pins={}
    for group in ('source_sha256','test_sha256','auxiliary_sha256'):
        for name,pin in manifest[group].items():
            rel=pathlib.PurePosixPath(name)
            need(name==rel.as_posix() and not rel.is_absolute()
                 and '..' not in rel.parts and name not in pins,'unsafe_manifest_entry')
            pins[name]=pin
    need(len(pins)==479 and pins['hymem/dreaming/phase1.py']==C['old'],
         'source_manifest_shape')
    for name,pin in pins.items():
        if name=='hymem/dreaming/phase1.py':pin=expected_phase1
        if name=='hymem/doctor.py':pin=C['doctor']
        path=live/name;regular(path)
        need(sha(path)==pin,'production_source_drift')
    return len(pins)

def gates():
    regular(candidate);need(sha(candidate)==C['new'],'candidate_pin_drift')
    regular(compare);value=json.loads(compare.read_bytes())
    need(value['status']=='completed' and value['paid_calls']==0,
         'offline_compare_not_complete')
    arms=value['arms']
    statuses={arm:{dedup:arms[arm]['metadata'][dedup]['status']
                   for dedup in ('dedup_on','dedup_off')}
              for arm in ('baseline','fixed')}
    need(statuses=={'baseline':{'dedup_on':'rejected','dedup_off':'persisted'},
                    'fixed':{'dedup_on':'persisted','dedup_off':'persisted'}},
         'offline_compare_outcome_drift')
    regular(store);proof=json.loads(store.read_bytes())
    # Closed, metadata-only approval gate: the independent fixed-store receipt
    # must identify the expected 25 observations and one publication.
    need(proof.get('current_observations')==25
         and proof.get('current_publications')==1
         and proof.get('integrity_ok') is True
         and proof.get('foreign_key_violations')==0
         and proof.get('ledger_count_mismatches')==0
         and proof.get('same_generation_conflicting_groups')==0
         and proof.get('new_provider_calls')==0
         and proof.get('network')=='none',
         'fixed_store_verification_missing')
    return {'baseline':statuses['baseline'],'fixed':statuses['fixed'],
            'fixed_observations':25,'fixed_publications':1,
            'compare_sha256':sha(compare),'store_verification_sha256':sha(store)}

def backup():
    backup_path=stage/'pre-deploy.sqlite'
    need(not backup_path.exists(),'backup_exists')
    fd=os.open(backup_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    os.close(fd)
    source=sqlite3.connect('file:'+quote(str(db),safe='/')+'?mode=ro',uri=True,timeout=30)
    dest=sqlite3.connect(backup_path)
    try:source.backup(dest,pages=1024,sleep=0.1)
    finally:dest.close();source.close()
    check=r'''
import json,pathlib,sys
sys.path.insert(0,'/source')
from hymem.core import db
c=db.connect(pathlib.Path('/backup/pre-deploy.sqlite'))
c.execute('PRAGMA query_only=ON')
try:
    assert db._load_vec_extension(c)
    v=db.schema_version(c)
    ok=c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    fk=c.execute('PRAGMA foreign_key_check').fetchone() is None
    print(json.dumps({'schema':v,'integrity_ok':ok,'foreign_keys_ok':fk}))
finally:c.close()
'''
    command=['docker','run','--rm','--name','hymem-claim-deploy-backup-check',
             '--pull','never','--network','none','--user','1000:1000',
             '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
             '--pids-limit','128','--memory','2g','--cpus','2',
             '--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
             '--mount','type=bind,src='+str(live)+',dst=/source,readonly',
             '--mount','type=bind,src='+str(runtime)+',dst=/home/node/hymem-env,readonly',
             '--mount','type=bind,src='+str(stage)+',dst=/backup',
             '--entrypoint','/home/node/hymem-env/bin/python3',C['image'],
             '-I','-B','-c',check]
    process=subprocess.run(command,capture_output=True,timeout=180)
    need(process.returncode==0 and len(process.stdout)<1024,'backup_check_failed')
    verdict=json.loads(process.stdout)
    need(verdict=={'schema':63,'integrity_ok':True,'foreign_keys_ok':True},
         'backup_integrity_or_schema')
    return {'sha256':sha(backup_path),'bytes':backup_path.stat().st_size,
            **verdict}

def preflight():
    need(os.geteuid()==1000 and snapshot.is_dir() and not snapshot.is_symlink(),
         'host_identity_or_snapshot')
    need(not stage.exists(),'deploy_stage_already_exists')
    state=stopped();source_pins(C['old']);comparison=gates()
    need(runtime.is_dir() and not runtime.is_symlink(),'runtime_missing')
    regular(db)
    stage.mkdir(mode=0o700)
    before=target.read_bytes();need(hashlib.sha256(before).hexdigest()==C['old'],
                                   'phase1_before_drift')
    fd=os.open(stage/'phase1-before.py',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:out.write(before);out.flush();os.fsync(out.fileno())
    backup_info=backup()
    value={'status':'passed','service':state,'source_files':479,
           'before_phase1_sha256':C['old'],'after_phase1_sha256':C['new'],
           'phase1_backup_sha256':sha(stage/'phase1-before.py'),
           'production_db_sha256':sha(db),'db_backup':backup_info,
           'comparison':comparison,'manifest_sha256':C['manifest']}
    return receipt('preflight.json',value)

def apply():
    need(os.geteuid()==1000 and stage.is_dir() and not stage.is_symlink()
         and stat.S_IMODE(stage.stat().st_mode)==0o700,'deploy_stage_invalid')
    need(not (stage/'apply-intent.json').exists(),'apply_already_attempted')
    prior=read('preflight.json')
    need(prior['status']=='passed' and prior['manifest_sha256']==C['manifest'],
         'preflight_receipt_invalid')
    state=stopped();need(state['container_id']==prior['service']['container_id'],
                         'service_identity_changed')
    source_pins(C['old']);comparison=gates()
    need(comparison==prior['comparison'] and sha(db)==prior['production_db_sha256']
         and sha(stage/'pre-deploy.sqlite')==prior['db_backup']['sha256']
         and sha(stage/'phase1-before.py')==C['old'], 'preapply_state_drift')
    info=regular(target)
    need(info.st_uid==os.geteuid() and info.st_gid in os.getgroups()
         and sha(target)==C['old'],'target_owner_or_pin')
    receipt('apply-intent.json',{'status':'authorized_to_apply',
            'target':'hymem/dreaming/phase1.py','before_sha256':C['old'],
            'after_sha256':C['new']})
    body=candidate.read_bytes()
    fd,temp=tempfile.mkstemp(prefix='.claim-conflict-phase1-',dir=target.parent)
    with os.fdopen(fd,'wb') as out:
        out.write(body);out.flush();os.fsync(out.fileno())
        os.fchmod(out.fileno(),stat.S_IMODE(info.st_mode))
        os.fchown(out.fileno(),info.st_uid,info.st_gid)
    need(sha(target)==C['old'] and sha(pathlib.Path(temp))==C['new'],
         'atomic_replace_precondition_drift')
    os.replace(temp,target)
    dirfd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(dirfd)
    finally:os.close(dirfd)
    need(sha(target)==C['new'],'atomic_replace_failed')
    source_pins(C['new'])
    value={'status':'passed','target':'hymem/dreaming/phase1.py',
           'before_sha256':C['old'],'after_sha256':C['new'],
           'db_backup_sha256':prior['db_backup']['sha256'],
           'restart_required':True,'restart_performed':False,
           'source_files_verified':479}
    return receipt('apply.json',value)

value=preflight() if C['action']=='preflight' else apply()
print(json.dumps(value,sort_keys=True))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "apply"))
    args = parser.parse_args()
    cfg = {"action": args.action, "snapshot": SNAP, "old": OLD, "new": NEW,
           "doctor": DOCTOR, "manifest": MANIFEST, "image": IMAGE}
    code = "CONFIG=" + repr(json.dumps(cfg)) + "\n" + REMOTE
    try:
        reply = subprocess.run([*SSH, "python3 -I -B -c " + shlex.quote(code)],
                               capture_output=True,
                               timeout=300 if args.action == "preflight" else 120)
    except subprocess.TimeoutExpired:
        print(json.dumps({"status": "remote_timeout", "action": args.action,
                          "outcome": "unknown", "requires_inspection": True}))
        return 1
    if reply.returncode or len(reply.stdout) > 8192:
        print(json.dumps({"status": "remote_action_failed", "action": args.action,
                          "outcome": "unknown", "requires_inspection": True}))
        return 1
    value = json.loads(reply.stdout)
    print(json.dumps(value, sort_keys=True))
    return 0 if value.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
