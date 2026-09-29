"""Independent one-shot dispatch and receipt controls; never run a service."""
from pathlib import Path
import os
import subprocess
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_semantic_probe_host as host


@pytest.fixture
def sealed(tmp_path,monkeypatch):
    home_name = 'TASK_HOME_ROOT' if hasattr(host,'TASK_HOME_ROOT') else 'HOME'
    monkeypatch.setattr(host,home_name,tmp_path)
    monkeypatch.setattr(host,'UID',os.getuid())
    root = tmp_path / '.hymem-luna-semantic-probe-root1234'
    root.mkdir(mode=0o700)
    pins = dict(host.ACCEPTED)
    pins.update({name:'0'*64 for name in host.NEW})
    receipt = host.receipt_for(root,pins,'1'*64)
    host.write_once(root/'launch-receipt.json',receipt)
    monkeypatch.setattr(host,'verify_bundle',lambda *args:None)
    monkeypatch.setattr(host,'host_admission',lambda:None)
    return root,receipt


def test_root_one_shot_even_after_success_and_private_dispatch_output(sealed):
    root,receipt = sealed
    calls = []
    def dispatch(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0,stdout=b'PRIVATE_STDOUT',stderr=b'PRIVATE_STDERR')
    sha = host.sha(root/'launch-receipt.json')
    result = host.launch(root,sha,dispatch=dispatch)
    assert result['launched'] and result['never_retry']
    assert 'PRIVATE_' not in str(result)
    assert (root/'private-dispatch-stdout.bin').read_bytes() == b'PRIVATE_STDOUT'
    assert (root/'private-dispatch-stdout.bin').stat().st_mode & 0o077 == 0
    with pytest.raises(BaseException):
        host.launch(root,sha,dispatch=dispatch)
    assert len(calls) == 1


def test_root_ambiguous_timeout_permanently_consumes_dispatch(sealed):
    root,_ = sealed
    calls = []
    def dispatch(argv,**kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv,20,output=b'PRIVATE_OUTPUT')
    sha = host.sha(root/'launch-receipt.json')
    result = host.launch(root,sha,dispatch=dispatch)
    assert not result['launched'] and result['dispatch_ambiguous'] and result['never_retry']
    with pytest.raises(BaseException):
        host.launch(root,sha,dispatch=dispatch)
    assert len(calls) == 1


def test_root_fixed_containment_and_clean_environment(sealed):
    root,receipt = sealed
    argv = host.command(root,receipt)
    expected = {'--property=Restart=no','--property=KillMode=control-group',
        '--property=RuntimeMaxSec=1930s','--property=TimeoutStopSec=10s',
        '--property=TasksMax=256','--property=MemoryMax=4294967296',
        '--property=CPUQuota=200%','--property=OOMPolicy=kill','--property=UMask=0077'}
    assert expected.issubset(argv)
    assert argv[argv.index('/usr/bin/env')+1] == '-i'
    assert '-I' in argv and '-B' in argv
    assert all('API_KEY' not in part for part in argv)
    assert argv.count(receipt['unit']) == 1


@pytest.mark.parametrize('mutation', [
    lambda r:r['limits'].update(turns=30),
    lambda r:r['limits'].update(turns=29.0),
    lambda r:r['limits'].update(workers=True),
    lambda r:r.update(model='different'),
    lambda r:r.update(unit='unrelated.service'),
    lambda r:r['policy'].update(restart='always'),
    lambda r:r.update(subscription_only=1),
    lambda r:r.update(extra='unreviewed'),
])
def test_root_receipt_drift_rejected_before_dispatch(sealed,mutation):
    import copy,json
    root,receipt = sealed
    changed = copy.deepcopy(receipt)
    mutation(changed)
    # Simulate a newly supplied hash to prove field equality is also strict.
    (root/'launch-receipt.json').write_text(json.dumps(changed))
    calls = []
    def dispatch(*a,**k):
        calls.append(True)
        return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
    with pytest.raises(BaseException):
        host.launch(root,host.sha(root/'launch-receipt.json'),dispatch=dispatch)
    assert not calls
