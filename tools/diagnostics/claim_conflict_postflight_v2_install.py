"""Install only the reviewed SQLite-reader correction; never rerun paid work."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys

HERE=Path(__file__).parent
INSTALLER_SHA='3748be1ad6d5984da65bd6e6e86a283a37b8a0812533f448d4d847a2d53ad103'
FILES={
    'claim_conflict_postflight_v2_host.py':('claim_conflict_postflight_v2_host.py','03b58304e143823ed18dd5769fb6f42cf314eb27f0ed4d794821648d8c99ea9e'),
    'postflight_host_v1.py':('claim_conflict_postflight_host.py','6b1ad8ded38f76adba0b5005caa13ee95f5eec1a2026f8d84470eacb9a4f6aae'),
    'claim_conflict_private_dream_postflight.py':('claim_conflict_private_dream_postflight_v2.py','abc1293423616ccb042ed18325b367a4af69b050c91b527fe6884d747b541d15'),
    'postflight_v1.py':('claim_conflict_private_dream_postflight.py','1ca6318fdac58d7b0715547baa8b80d4119c9a1a2012fb9f58bf96c3a443b117'),
    'claim_conflict_store_audit.py':('claim_conflict_store_audit.py','e2efe365c5aedbbe88d86d521dd37b821759afc5fb21c21a6662e6a8ce567f42'),
}

if __name__=='__main__':
    p=HERE/'claim_conflict_postflight_install.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==INSTALLER_SHA
    spec=importlib.util.spec_from_file_location('pinned_postflight_install',p)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    action=sys.argv[1]
    ssh=['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','afrodite']
    if action=='install':
        files={}
        for remote,(local,pin) in FILES.items():
            raw=(HERE/local).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==pin
            files[remote]={'sha256':pin,'text':raw.decode()}
        needle="stage=root/'postflight-v1'"
        assert base.REMOTE.count(needle)==1
        remote=base.REMOTE.replace(needle,"stage=root/'postflight-v2'")
        result=subprocess.run(ssh+[shlex.join(['python3','-I','-B','-c',remote])],input=json.dumps({'root':base.BASE,'seal':base.SEAL,'files':files}),text=True,capture_output=True)
    elif action=='run':
        stage=base.BASE+'/postflight-v2'
        seal_sha=hashlib.sha256(json.dumps(base.SEAL,sort_keys=True).encode()).hexdigest()
        result=subprocess.run(ssh+[shlex.join(['python3','-I','-B',stage+'/claim_conflict_postflight_v2_host.py','--seal',stage+'/seal.json','--seal-sha256',seal_sha])],text=True,capture_output=True)
    else:raise SystemExit('unknown_action')
    print(result.stdout,end='')
    if result.returncode:print(result.stderr[-2000:],file=sys.stderr)
    raise SystemExit(result.returncode)
