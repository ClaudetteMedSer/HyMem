"""Host tests are local and never run Docker, providers, or remote commands."""
import importlib.util
import json
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1]/'lme_r9_summary/host.py'
spec = importlib.util.spec_from_file_location('r9_host_test',PATH)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
PIN = 'a'*64

def source(tmp_path):
    tree = tmp_path/'tree'
    for rel in ('hymem/dreaming/digest.py','hymem/core/schema.sql','resources/example.json'):
        path = tree/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('offline fixture')
    return tree

def test_bounds_and_mounts():
    assert host.BOUNDS == dict(completion_calls=12,http_attempts=36,timeout_seconds=900,invocation_timeout_seconds=120)
    for root in host.ROOTS.values():
        for mode in ('offline','live'):
            plan = host.plan(root,mode,PIN)
            assert {p for p,(_,rw) in plan['mounts'].items() if rw} == {'/work','/results'}
            assert plan['network'] == ('none' if mode == 'offline' else 'bridge')
            assert ('/run/deepseek.env' in plan['mounts']) == (mode == 'live')
            assert '--read-only' in plan['command'] and '--cap-drop' in plan['command']
            assert 'DEEPSEEK_API_KEY' not in ' '.join(plan['command'])
            assert all('docker.sock' not in p for p,_ in plan['mounts'].values())

def test_denies_arbitrary_remote_root_before_docker(tmp_path,monkeypatch):
    monkeypatch.setattr(host,'run',lambda _:pytest.fail('Docker called'))
    with pytest.raises(RuntimeError,match='launch_scope'):
        host.plan(tmp_path,'offline',PIN)
    with pytest.raises(RuntimeError,match='manifest_pin'):
        host.action(tmp_path,'live',PIN)
    for variant in ('baseline','candidate'):
        assert host.ROOTS[variant] == host.BASE/('lme-r9-summary-'+variant+'-v2')
        with pytest.raises(RuntimeError,match='launch_scope'):
            host.plan(host.BASE/('lme-r9-summary-'+variant+'-v1'),'offline',PIN)

def test_sealing_full_inventory_freshness_and_tamper(tmp_path):
    tree = source(tmp_path)
    source_pin = host.digest(host.inventory(tree))
    root = tmp_path/'capsule'
    result = host.seal(tree,root,{'hymem.sqlite':PIN},'baseline',source_pin)
    m = json.loads((root/'diag/manifest.json').read_text())
    assert m['variant'] == 'baseline' and m['source_inventory_sha256'] == source_pin
    assert m['capsule_root'] == str(host.ROOTS['baseline'])
    assert host.sha(root/'diag/core.py') == m['core_sha256']
    assert host.sha(root/'support/r8_summary.py') == m['r8_summary_sha256']
    assert host.sha(root/'diag/manifest.json') == result['manifest_sha256']
    assert result['api_calls'] == 0 and result['source_files'] == 3
    with pytest.raises(RuntimeError,match='new_package_required'):
        host.seal(tree,root,{'hymem.sqlite':PIN},'baseline',source_pin)
    (tree/'resources/example.json').write_text('tampered')
    with pytest.raises(RuntimeError,match='source_inventory_pin'):
        host.seal(tree,tmp_path/'other',{'hymem.sqlite':PIN},'baseline',source_pin)

def test_symlink_source_denial(tmp_path):
    tree = source(tmp_path)
    (tree/'alias').symlink_to(tree/'resources/example.json')
    with pytest.raises(RuntimeError,match='source_special_file'):
        host.inventory(tree)

def test_status_privacy_closed_projection():
    raw = dict(status='review_pending',completion_calls=12,http_attempts=36,manifest_sha256=PIN,
               arbitrary='private body',response={'content':'secret'},exception_type='private',
               pid=True,source_unchanged=True,container_id='secret',provider_calls=-1)
    assert host.project(raw) == dict(status='review_pending',completion_calls=12,http_attempts=36,
                                    manifest_sha256=PIN,source_unchanged=True)
    assert host.project({'status':'secret response','manifest_sha256':'secret'}) == {}

def test_intent_is_exclusive_private_and_not_overwritten(tmp_path):
    path = tmp_path/'live-intent.json'
    host.save(path,{'manifest_sha256':PIN,'mode':'live'})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        host.save(path,{'manifest_sha256':'b'*64})
    assert json.loads(path.read_text())['manifest_sha256'] == PIN

def test_live_no_offline_receipt_before_docker(tmp_path,monkeypatch):
    root = tmp_path/'root'; root.mkdir()
    monkeypatch.setattr(host,'ROOTS',{'baseline':root})
    monkeypatch.setattr(host,'verify',lambda *_:{'variant':'baseline'})
    monkeypatch.setattr(host,'run',lambda _:pytest.fail('Docker called'))
    with pytest.raises(FileNotFoundError):
        host.action(root,'live',PIN)

def test_stale_intent_refuses_retry_before_docker(tmp_path,monkeypatch):
    root = tmp_path/'root'; root.mkdir()
    host.save(root/'offline-intent.json',{'manifest_sha256':PIN})
    monkeypatch.setattr(host,'ROOTS',{'baseline':root})
    monkeypatch.setattr(host,'verify',lambda *_:{})
    monkeypatch.setattr(host,'run',lambda _:pytest.fail('Docker called'))
    with pytest.raises(RuntimeError,match='already_launched'):
        host.action(root,'offline',PIN)

@pytest.mark.parametrize('fault', ['image','mount','network','privileged','memory'])
def test_real_inspection_denies_container_drift(tmp_path,monkeypatch,fault):
    tree = source(tmp_path)
    capsule = tmp_path/'capsule'
    host.seal(tree,capsule,{'hymem.sqlite':PIN},'baseline',host.digest(host.inventory(tree)))
    m = json.loads((capsule/'diag/manifest.json').read_text())
    expected = host.plan(host.ROOTS['baseline'],'offline',PIN)
    args = expected['command'][expected['command'].index(host.IMAGE)+1:]
    data = dict(Id=PIN,Image=host.IMAGE,Name='/'+expected['name'],Path='/home/node/hymem-env/bin/python3',Args=args,
        Config=dict(Image=host.IMAGE,User='1000:1000',WorkingDir='/candidate',
                    Entrypoint=['/home/node/hymem-env/bin/python3'],Cmd=args,Env=['PYTHONDONTWRITEBYTECODE=1']),
        HostConfig=dict(NetworkMode='none',ReadonlyRootfs=True,Privileged=False,CapDrop=['ALL'],
                        SecurityOpt=['no-new-privileges'],Init=True,PidsLimit=128,Memory=2147483648,NanoCpus=2000000000,
                        Tmpfs={'/tmp':'rw,noexec,nosuid,size=128m'},RestartPolicy={'Name':'no'}),
        State=dict(Status='created',ExitCode=0,OOMKilled=False,Pid=0),
        Mounts=[dict(Destination=target,Source=origin,RW=rw,Type='bind')
                for target,(origin,rw) in expected['mounts'].items()])
    if fault == 'image': data['Image'] = 'wrong'
    elif fault == 'mount': data['Mounts'][0]['RW'] = True
    elif fault == 'network': data['HostConfig']['NetworkMode'] = 'bridge'
    elif fault == 'privileged': data['HostConfig']['Privileged'] = True
    elif fault == 'memory': data['HostConfig']['Memory'] *= 2
    monkeypatch.setattr(host,'run',lambda command:json.dumps([data]))
    with pytest.raises(AssertionError):
        host.inspect(capsule,PIN,expected,m)
