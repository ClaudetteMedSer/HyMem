"""Read-only finite diagnosis of the stopped first semantic probe; no inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/atta/.hymem-luna-semantic-probe-8wftffh9')
ENTRY_SHA='60fc7fca900ef8320a410af2f5f01415f973f01907cf7c0456c0b4936611030a'
RECEIPT_SHA='b7814a637ab7468a29150be0dca9d0195013f6684c8f24742c161ccb1991b4c7'
path=ROOT/'code/tools/diagnostics/luna_semantic_probe_run.py'
assert not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==ENTRY_SHA
spec=importlib.util.spec_from_file_location('root_semantic_startup_entry',path)
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
receipt,loaded,proof,retained=entry.preflight(ROOT,RECEIPT_SHA)
metadata={'preflight_verified':True,'candidate_files':proof['candidate_files'],
    'run_directory_exists':(ROOT/'run').exists(),
    'terminal_exists':(ROOT/'safe-terminal.json').exists(),'status_probes':[]}
for bus in (False,True):
    env={'HOME':'/home/atta','PATH':'/usr/local/bin:/usr/bin:/bin','TMPDIR':str(ROOT/'tmp')}
    if bus:
        env['XDG_RUNTIME_DIR']='/run/user/1000'
        env['DBUS_SESSION_BUS_ADDRESS']='unix:path=/run/user/1000/bus'
    result=subprocess.run(['/usr/bin/systemctl','--user','show',receipt['unit'],
        '--property=MainPID,ControlGroup,ActiveState,SubState','--no-pager'],
        env=env,capture_output=True,text=True,timeout=10)
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    metadata['status_probes'].append({'explicit_user_bus':bus,'returncode':result.returncode,
        'missing_user_bus_error':'Failed to connect to user scope bus' in result.stderr or
             'XDG_RUNTIME_DIR not defined' in result.stderr or
             '$DBUS_SESSION_BUS_ADDRESS and $XDG_RUNTIME_DIR not defined' in result.stderr,
        'main_pid_zero':fields.get('MainPID')=='0',
        'control_group_empty':fields.get('ControlGroup')=='',
        'unit_failed':fields.get('ActiveState')=='failed'})
group=Path('/sys/fs/cgroup'+receipt['expected_cgroup'])
metadata['expected_cgroup_absent']=not group.exists()
print(json.dumps(metadata,sort_keys=True))
