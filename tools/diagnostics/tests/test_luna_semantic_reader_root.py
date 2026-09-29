"""Independent metadata integrity faults. No network, inference or service start."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_semantic_probe_host as host
from tools.diagnostics import luna_semantic_probe_progress as reader
from tools.diagnostics import luna_semantic_probe_run as runner


def failed_result():
    return {'schema':'luna-semantic-probe-v1', 'control_results':[], 'hybrid':None,
        'false_support_claims':0,'false_rejections':0,'missed_recoveries':0,
        'recheck_failures':0,'malformed_units':0,'completed_units':0,
        'paid_budget':{'turns':1,'known_tokens':0,'usage_complete':False,
                       'in_flight':0,'reserved':0},
        'first_failure':None,'client_cleanup_ok':True,
        'stop_code':'transport_or_budget_stop','core_completed':False,
        'completed_and_clean':False,'process_cleanup_verified':False,
        'all_semantic_checks_passed':False}


def pair():
    result = failed_result()
    terminal = runner._safe_result(result,'1'*64,'2'*64,
                                  hashlib.sha256(b'[]').hexdigest(),True,1.0)
    return result,terminal


def test_root_partial_unknown_usage_is_reportable_not_success():
    result,terminal = pair()
    assert reader._terminal_valid(terminal,result,'1'*64,'2'*64)
    assert not terminal['completed_and_clean']
    assert not terminal['paid_budget']['usage_complete']


@pytest.mark.parametrize('mutation',[
    lambda r,t:t.update(stop_code='PRIVATE_UNBOUNDED_TEXT'),
    lambda r,t:t.update(first_failure={'private':'PRIVATE_UNBOUNDED_TEXT'}),
    lambda r,t:r.update(core_completed=0),
    lambda r,t:r.update(all_semantic_checks_passed=0),
    lambda r,t:r.update(client_cleanup_ok=1),
    lambda r,t:r['paid_budget'].update(turns=1.0),
    lambda r,t:t.update(all_semantic_checks_passed=True),
    lambda r,t:(t.update(all_semantic_checks_passed=True),r.update(all_semantic_checks_passed=True)),
    lambda r,t:t.update(stop_code=None),
])
def test_root_malformed_or_inconsistent_terminal_is_not_validated(mutation):
    result,terminal = pair()
    # Isolate terminal and private-result objects as independent JSON files.
    terminal = copy.deepcopy(terminal)
    mutation(result,terminal)
    assert not reader._terminal_valid(terminal,result,'1'*64,'2'*64)


def test_root_boolean_process_index_is_not_integer_zero(tmp_path):
    folder=tmp_path/'run/private-owned-processes'
    folder.mkdir(parents=True)
    identity={'pid':2147483647,'pgid':2147483647,'starttime':1,'index':False}
    (folder/'0000.json').write_text(json.dumps(identity))
    digest=hashlib.sha256(json.dumps([identity],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert not reader._owned(tmp_path,digest)['verified']


@pytest.mark.parametrize('mutation',['alter','extra','missing','symlink'])
def test_root_physical_candidate_drift_precedes_import_or_dispatch(tmp_path,monkeypatch,mutation):
    repo=Path(__file__).resolve().parents[3]
    monkeypatch.setattr(host,'TASK_HOME_ROOT',tmp_path)
    monkeypatch.setattr(host,'UID',os.getuid())
    monkeypatch.setattr(host,'OLD_CANDIDATE',Path('/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/candidate'))
    monkeypatch.setattr(host,'OLD_MAP',Path('/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi/headless-grounding-source-map.json'))
    root=tmp_path/'.hymem-luna-semantic-probe-physical1'
    root.mkdir(mode=0o700)
    code=root/'code'
    for relative in [*host.ACCEPTED,*host.NEW]:
        dest=code/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/relative,dest)
    source=Path('/private/tmp/hymem-semantic-step2-v2-candidate-20260929')
    shutil.copytree(source,root/'candidate')
    shutil.copyfile('/private/tmp/hymem-semantic-step2-v2-map-20260929.json',root/'candidate-source-map.json')
    target=root/'candidate/hymem/extraction/grounding.py'
    if mutation=='alter': target.write_text('raise RuntimeError("MUST_NOT_IMPORT")\n')
    elif mutation=='missing': target.unlink()
    elif mutation=='extra': (root/'candidate/unsealed.py').write_text('raise RuntimeError("MUST_NOT_IMPORT")\n')
    else:
        target.unlink()
        target.symlink_to(source/'hymem/extraction/grounding.py')
    receipt=host.receipt_for(root,host.source_pins(code),'3'*64)
    with pytest.raises(ValueError,match='candidate_'):
        host.verify_bundle(root,receipt)


@pytest.mark.parametrize('field,value', [
    ('TasksMax','512'),('MemoryMax','infinity'),('CPUQuotaPerSecUSec','infinity'),
    ('KillMode','process'),('Restart','always'),('OOMPolicy','continue'),
    ('RuntimeMaxUSec','infinity'),('TimeoutStopUSec','1min 30s'),
    ('RemainAfterExit','no'),('MainPID','123'),('NRestarts','1'),
    ('Result','oom-kill'),('ExecMainStatus','1'),('SubState','running'),
    ('ControlGroup','/unrelated'),
])
def test_root_actual_unit_policy_drift_never_clean(monkeypatch,field,value):
    rows={'ActiveState':'active','SubState':'exited','MainPID':'0',
        'ControlGroup':'','NRestarts':'0','Result':'success','OOMPolicy':'kill',
        'ExecMainStatus':'0','MemoryMax':'4294967296','TasksMax':'256',
        'CPUQuotaPerSecUSec':'2s','KillMode':'control-group','Restart':'no',
        'RemainAfterExit':'yes','RuntimeMaxUSec':'32min 10s','TimeoutStopUSec':'10s'}
    def output(*args,**kwargs):
        return SimpleNamespace(returncode=0,stdout='\n'.join(f'{k}={v}' for k,v in rows.items()))
    monkeypatch.setattr(reader.subprocess,'run',output)
    group='/hymem-root-offline-nonexistent'
    assert reader._systemd('invented.service',group)['clean']
    rows[field]=value
    assert not reader._systemd('invented.service',group)['clean']
