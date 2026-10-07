"""Root startup controls; no control-plane or model process is executed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_staged_startup_bundle_v1 as builder


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


def startup_fixture(tmp_path, mode='inference'):
    root = tmp_path/'root'; root.mkdir(mode=0o700)
    source = tmp_path/'source.py'; source.write_bytes(builder.derive_adapter())
    adapter = module(source, 'root_staged_adapter_fixture')
    receipt = dict(unit='hymem-luna-invented.service', expected_cgroup='/invented',
        binary_sha256='2'*64, source_sha256={
            'tools/diagnostics/luna_classification_progress_reference_v3.py': '3'*64})
    (root/'launch-receipt.json').write_bytes(canonical(receipt))
    def write_once(path, value):
        with path.open('x') as stream:
            json.dump(value, stream, sort_keys=True)
    host = SimpleNamespace(root_valid=lambda r: r == root, regular=lambda p:p.is_file() and not p.is_symlink(),
        write_once=write_once, strict_equal=lambda a,b: canonical(a)==canonical(b),
        receipt_for=lambda *a: receipt, verify_bundle=lambda *a,**k: None)
    sidecar = adapter.seal(root, source, host, mode)
    # Execute the actual sealed copy, like server entry rather than prepare.
    adapter = module(root/builder.ADAPTER, 'root_staged_adapter_fixture_sealed')
    digest = hashlib.sha256((root/'launch-receipt.json').read_bytes()).hexdigest()
    (root/'launch-attempt.json').write_bytes(canonical(dict(receipt_sha256=digest, one_shot=True)))
    (root/'launch-admission.json').write_bytes(canonical(dict(receipt_sha256=digest,
        old_luna_stopped=True, old_deepseek_stopped=True, memory_floor_bytes=6*1024**3,
        disk_floor_bytes=20*1024**3, memory_floor_met=True, disk_floor_met=True)))
    return adapter, root, receipt, host, sidecar, digest


@pytest.mark.parametrize('fault', ['none', 'mode', 'adapter', 'receipt', 'sidecar', 'extra', 'scope', 'bool'])
def test_exact_sidecar_and_source_binding(tmp_path, fault):
    adapter, root, receipt, host, sidecar, digest = startup_fixture(tmp_path)
    adapter_hash, sidecar_hash = sidecar['adapter_sha256'], sidecar['sidecar_sha256']
    if fault == 'adapter':
        (root/builder.ADAPTER).write_text('PRIVATE')
    elif fault == 'receipt':
        digest = '0'*64
    elif fault == 'sidecar':
        sidecar_hash = '0'*64
    elif fault in ('extra', 'scope', 'bool'):
        value = json.loads((root/builder.SIDECAR).read_bytes())
        if fault == 'extra':
            value['extra'] = 'PRIVATE'
        elif fault == 'scope':
            value['bus_scope'] = 'all'
        else:
            value['entry_action'] = True
        (root/builder.SIDECAR).write_bytes(canonical(value))
        sidecar_hash = hashlib.sha256(canonical(value)).hexdigest()
    def verify():
        return adapter.verify(root, digest, adapter_hash, 'smoke' if fault=='mode' else 'inference', sidecar_hash)
    if fault == 'none':
        assert verify()['entry_action'] == 'inference'
    else:
        with pytest.raises(ValueError):
            verify()


@pytest.mark.parametrize('mode,fault', [('smoke','none'), ('inference','none'),
    ('inference','before'), ('inference','after'), ('inference','terminal_exists')])
def test_startup_failure_and_unknown_admission_are_distinct(monkeypatch, tmp_path, mode, fault):
    adapter, root, receipt, host, sidecar, digest = startup_fixture(tmp_path, mode)
    calls = []
    def execute(*args, **kw):
        calls.append('execute')
        if fault == 'before':
            raise RuntimeError('PRIVATE')
        (root/'run').mkdir()
        if fault == 'terminal_exists':
            host.write_once(root/'safe-terminal.json', dict(existing=True, known_tokens=123))
        if fault in ('after','terminal_exists'):
            raise RuntimeError('PRIVATE')
        return dict(diagnostic_completed=True)
    run = SimpleNamespace(execute=execute, preflight=lambda *a,**k: (receipt,(host,),None,None))
    monkeypatch.setattr(adapter, 'load_source', lambda p,*a: host if p.name=='luna_staged_host_v1.py' else run)
    monkeypatch.setattr(adapter, 'contained', lambda *a: calls.append('contained'))
    code = adapter.entry(root, digest, mode, sidecar['adapter_sha256'], sidecar['sidecar_sha256'])
    assert code == (fault != 'none')
    if mode == 'smoke':
        value = json.loads((root/'safe-terminal.json').read_bytes())
        assert calls == ['contained'] and value['model_calls'] == 0 and not (root/'run').exists()
    elif fault == 'before':
        value = json.loads((root/'safe-terminal.json').read_bytes())
        assert value['zero_admission_proof'] == 'run_directory_never_created' and value['paid_turns'] == 0
    elif fault == 'after':
        value = json.loads((root/'safe-terminal.json').read_bytes())
        assert value['paid_budget'] is None and not value['completed_and_clean']
        assert 'paid_turns' not in value
    elif fault == 'terminal_exists':
        assert json.loads((root/'safe-terminal.json').read_bytes()) == dict(existing=True, known_tokens=123)


@pytest.mark.parametrize('fault', ['none', 'unit', 'property', 'env', 'exception'])
def test_direct_containment_proxy_and_restoration(monkeypatch, tmp_path, fault):
    adapter, root, receipt, host, sidecar, digest = startup_fixture(tmp_path)
    seen = []
    def process(args, **kw):
        seen.append((args,kw))
        if fault == 'exception':
            raise RuntimeError('controlled')
    original = SimpleNamespace(run=process)
    run = SimpleNamespace(subprocess=original)
    monkeypatch.setattr(adapter, 'bus_environment', lambda: dict(DBUS_SESSION_BUS_ADDRESS='fixed-test-bus'))
    def contain(*args):
        command = ['/usr/bin/systemctl','--user','show',receipt['unit'],
            '--property=ActiveState,SubState,MainPID,ControlGroup,NRestarts,MemoryMax,TasksMax,CPUQuotaPerSecUSec,KillMode,Restart,RemainAfterExit,OOMPolicy,RuntimeMaxUSec,TimeoutStopUSec','--no-pager']
        if fault == 'unit':
            command[3] = 'unrelated.service'
        elif fault == 'property':
            command[4] += ',Environment'
        return run.subprocess.run(command, **({'env':{}} if fault=='env' else {}))
    run.verify_live_containment = contain
    if fault != 'none':
        with pytest.raises((ValueError,RuntimeError)):
            adapter.contained(root, receipt, run)
    else:
        adapter.contained(root, receipt, run)
        assert seen[0][1]['env'] == dict(DBUS_SESSION_BUS_ADDRESS='fixed-test-bus')
    assert run.subprocess is original
    if fault in ('unit','property','env'):
        assert seen == []


def test_real_host_one_shot_dispatch_preserves_env_isolation(monkeypatch, tmp_path):
    adapter, root, receipt, fake_host, sidecar, digest = startup_fixture(tmp_path)
    (root/'launch-attempt.json').unlink(); (root/'launch-admission.json').unlink()
    host = module(builder.BASE/'code/tools/diagnostics/luna_staged_host_v1.py', 'root_staged_real_host')
    monkeypatch.setattr(host,'root_valid', fake_host.root_valid)
    monkeypatch.setattr(host,'receipt_for', fake_host.receipt_for)
    monkeypatch.setattr(host,'verify_bundle', fake_host.verify_bundle)
    monkeypatch.setattr(host,'host_admission', lambda: None)
    monkeypatch.setattr(adapter,'bus_environment', lambda: dict(DBUS_SESSION_BUS_ADDRESS='fixed-test-bus'))
    seen=[]
    monkeypatch.setattr(adapter,'subprocess',SimpleNamespace(run=lambda cmd,**kw:
        (seen.append((cmd,kw)) or SimpleNamespace(returncode=0,stdout=b'',stderr=b''))))
    result=adapter.launch(root,digest,sidecar['adapter_sha256'],sidecar['sidecar_sha256'],host,'inference')
    assert result['launched'] and len(seen)==1
    command, kwargs=seen[0]
    assert kwargs['env']==dict(DBUS_SESSION_BUS_ADDRESS='fixed-test-bus')
    child=command[command.index('/usr/bin/env')+1:]
    assert child[0]=='-i' and not any('DBUS' in word or 'XDG_RUNTIME' in word for word in child)
    assert str(root/builder.ADAPTER) in command and command.count('inference')==1
    with pytest.raises(FileExistsError):
        adapter.launch(root,digest,sidecar['adapter_sha256'],sidecar['sidecar_sha256'],host,'inference')
    assert len(seen)==1


@pytest.mark.parametrize('fault',['none','bytes','extra','missing','symlink'])
def test_source_bundle_is_exact_and_never_rewritten(tmp_path,fault):
    base=tmp_path/'base';shutil.copytree(builder.BASE,base)
    source=base/'code/tools/diagnostics/luna_staged_core_v1.py'
    if fault=='bytes':source.write_text('PRIVATE')
    elif fault=='extra':(base/'extra').write_text('PRIVATE')
    elif fault=='missing':source.unlink()
    elif fault=='symlink':source.unlink();source.symlink_to(builder.BASE/'code/tools/diagnostics/luna_staged_core_v1.py')
    if fault=='none':
        assert len(builder.base_inventory(base))==37
        out=builder.prepare(tmp_path/'sealed',base)
        assert not out['launched'] and out['model_calls']==0
        assert (tmp_path/'sealed/derivation-receipt.json').read_bytes()==(base/'derivation-receipt.json').read_bytes()
    else:
        with pytest.raises(ValueError):builder.base_inventory(base)


@pytest.mark.parametrize('fault', ['none', 'running', 'oom', 'policy', 'marker', 'terminal_bool'])
def test_smoke_requires_independent_service_cleanup(monkeypatch, tmp_path, fault):
    from tools.diagnostics import luna_staged_progress_v1 as reader
    adapter, root, receipt, host, sidecar, digest = startup_fixture(tmp_path, 'smoke')
    terminal = dict(schema='luna-staged-probe-containment-smoke-staged-v1',
        receipt_sha256=digest, adapter_sha256=sidecar['adapter_sha256'],
        policy_verified=True, model_calls=0, paid_turns=0, known_tokens=0,
        semantic_accuracy_accepted=False, full_lme_ready=False)
    if fault == 'terminal_bool':
        terminal['policy_verified'] = 1
    (root/'safe-terminal.json').write_bytes(canonical(terminal))
    if fault == 'marker':
        (root/'launch-attempt.json').write_bytes(canonical(dict(receipt_sha256=digest, one_shot=1)))
    reference = module(builder.BASE/'code/tools/diagnostics/luna_classification_progress_reference_v3.py',
        'root_staged_service_reference')
    values = dict(ActiveState='active', SubState='exited', MainPID='0', ControlGroup='',
        NRestarts='0', Result='success', OOMPolicy='kill', ExecMainStatus='0',
        MemoryMax='4294967296', TasksMax='256', CPUQuotaPerSecUSec='2s',
        KillMode='control-group', Restart='no', RemainAfterExit='yes',
        RuntimeMaxUSec='1930s', TimeoutStopUSec='10s')
    if fault == 'running':
        values.update(SubState='running',MainPID='123')
    elif fault == 'oom':
        values.update(ActiveState='failed',SubState='failed',Result='oom-kill',ExecMainStatus='9')
    elif fault == 'policy':
        values['TasksMax'] = 'infinity'
    seen = []
    def systemctl(command, **kw):
        seen.append((command, kw))
        assert kw['env'] == dict(DBUS_SESSION_BUS_ADDRESS='fixed-test-bus')
        return SimpleNamespace(returncode=0,stdout='\n'.join(k+'='+v for k,v in values.items()))
    original = SimpleNamespace(run=systemctl)
    monkeypatch.setattr(reader,'subprocess',original)
    monkeypatch.setattr(adapter,'load_source',lambda p,*a: reference if 'reference' in p.name else host)
    monkeypatch.setattr(adapter,'bus_environment',lambda:dict(DBUS_SESSION_BUS_ADDRESS='fixed-test-bus'))
    if fault in ('marker','terminal_bool'):
        with pytest.raises(ValueError):
            adapter.observe(root,digest,sidecar['adapter_sha256'],sidecar['sidecar_sha256'],reader)
    else:
        out=adapter.observe(root,digest,sidecar['adapter_sha256'],sidecar['sidecar_sha256'],reader)
        assert out['unit_cleanup_verified'] == (fault=='none')
        assert out['phase'] == ('containment_smoke_passed' if fault=='none' else 'containment_smoke_cleanup_pending')
        assert out['model_calls']==0 and not out['completed_and_clean']
    assert reader.subprocess is original
