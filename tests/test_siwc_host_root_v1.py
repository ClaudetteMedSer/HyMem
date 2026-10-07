"""Root-owned host-chain controls. Invented metadata only, no host/account I/O."""
import ast
import copy
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import siwc_lme_diagnostic_bundle_v1 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v1 as host
from tools.diagnostics import siwc_lme_diagnostic_launch_v1 as launch
from tools.diagnostics import siwc_lme_diagnostic_v1 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v3 as reader
from benchmarks import chatgpt_plan_lme_v1 as bridge

REPO = Path(__file__).resolve().parents[1]
ACCEPTED = Path('/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle')
CODE = Path('/private/tmp/hymem-repaired-four-root-dtBv2p/bundle/code')


def assembled(tmp_path):
    root = tmp_path / '.hymem-siwc-lme-diagnostic-rootverify0001'
    summary = bundle.assemble(repo=REPO, accepted_code=CODE,
        candidate=ACCEPTED/'candidate', map_path=ACCEPTED/'source-map.json', output=root)
    assert summary['candidate_files'] == 514 and summary['code_files'] == 25
    assert summary['model_calls'] == 0 and not summary['credential_present']
    return root


def test_actual_540_file_archive_and_remote_decoder(tmp_path, monkeypatch):
    root = assembled(tmp_path)
    payload = host.archive_bytes(root)
    statements = []
    for node in ast.parse(host.REMOTE).body:
        text = ast.get_source_segment(host.REMOTE, node) or ''
        if text.startswith('need(os.getuid()'):
            continue
        if text.startswith('need(regular(DATASET)'):
            break
        statements.append(node)
    else:
        raise AssertionError('safe decoder boundary absent')
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(payload)))
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                 '<offline-remote-decoder>', 'exec'), scope)
    assert len(scope['manifest']) == 540
    assert len(scope['content']) == 540
    assert not (root/'launch-receipt.json').exists()
    (root/'extra').write_text('invented')
    with pytest.raises(ValueError):
        host.archive_bytes(root)


def test_actual_isolated_import_and_real_receipt_size_without_owner_or_dataset(tmp_path):
    root = assembled(tmp_path)
    script = r'''
import importlib.util,json,pathlib,sys
from types import SimpleNamespace
root=pathlib.Path(sys.argv[1]);path=root/'code/tools/diagnostics/siwc_lme_diagnostic_v1.py'
spec=importlib.util.spec_from_file_location('isolated_siwc_root',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
questions=[{'question_id':f'invented-{i}'} for i in range(4)]
dataset=root/'absent-invented-dataset.json';original=runner._sha
runner._sha=lambda p:runner.DATASET_SHA256 if p==dataset else original(p)
loaded.update(source_only=False,dataset=dataset,questions=questions,
    prior=SimpleNamespace(SelectedQuestions=lambda *_:questions))
receipt=runner.receipt_for(root,loaded)
assert receipt['unit']=='hymem-siwc-lme-diagnostic-rootverify0001.service'
assert len(receipt['source_sha256'])==25 and receipt['workers']==4
assert receipt['model']=='gpt-5.6-luna' and receipt['store'] is False and receipt['stream'] is True
assert receipt['grant_identity_sha256']==runner.GRANT_IDENTITY_SHA256
assert len(runner._canonical(receipt))<=8192
print(json.dumps({'source_only_import':True,'receipt_bytes':len(runner._canonical(receipt)),'model_calls':0}))
'''
    checked = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(root)],
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result['source_only_import'] and result['model_calls'] == 0
    assert 4000 < result['receipt_bytes'] < 8192


def test_launcher_command_fixed_caps_runtime_and_no_binary_route():
    root=Path('/home/atta/.hymem-siwc-lme-diagnostic-preflight-invented')
    receipt={'unit':'hymem-siwc-lme-diagnostic-preflight-invented.service',
             'runtime_path':str(runner.RUNTIME_PATH)}
    command=launch.command(root,receipt,'a'*64)
    for arg in ('--property=RuntimeMaxSec=14530s','--property=TimeoutStopSec=10s',
                '--property=TasksMax=256','--property=MemoryMax=4294967296',
                '--property=CPUQuota=200%','--property=KillMode=control-group',
                '--property=Restart=no','--property=OOMPolicy=kill','-I','-B'):
        assert arg in command
    assert str(runner.RUNTIME_PATH) in command
    assert '--binary' not in command and not any('codex' in arg for arg in command)
    assert command[command.index('--questions')+1]=='4'
    assert command[command.index('--workers')+1]=='4'
    assert command[command.index('--receipt-sha256')+1]=='a'*64


def test_current_and_prior_unit_exclusion_includes_siwc():
    tree=ast.parse(Path(launch.__file__).read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='host_admission')
    literals={node.value for node in ast.walk(function) if isinstance(node,ast.Constant) and isinstance(node.value,str)}
    assert {'hymem-luna*','hymem-lme*','hymem-deepseek*','hymem-siwc-lme-diagnostic-*'} <= literals


def test_recursive_prior_unit_cleanup_includes_nested_threads(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'CGROUP_ROOT',tmp_path)
    unit='hymem-siwc-lme-diagnostic-prior.service'
    expected='/user.slice/user-1000.slice/user@1000.service/app.slice/'+unit
    group=tmp_path/expected.lstrip('/');group.mkdir(parents=True)
    def empty(path):
        (path/'cgroup.procs').write_text('')
        (path/'cgroup.threads').write_text('')
        (path/'cgroup.events').write_text('populated 0\n')
    empty(group)
    stopped=('active','exited','0','')
    assert launch.prior_unit_stopped(unit,stopped)
    child=group/'child';child.mkdir();empty(child)
    (child/'cgroup.threads').write_text('123\n')
    assert not launch.prior_unit_stopped(unit,stopped)


@pytest.mark.parametrize('key,value', [
    ('wire_observation', {'secret': 'private provider text'}),
    ('stream_observation', {'event_type': 'private provider text'}),
    ('unknown_usage', False), ('turn_admitted', 1), ('phase', 'private'),
    ('code', 'private provider text'), ('http_status', True)])
def test_reader_rejects_unbounded_or_untruthful_failure_metadata(key,value):
    first={'code':'http_failure','phase':'http','turn_admitted':True,'unknown_usage':True}
    first[key]=value
    with pytest.raises((ValueError, TypeError)):
        reader.first_failure(first)


def test_reader_finite_projection_matches_bridge():
    wire={'header_defect':'none','parsed_header_count':1,'transfer_encoding':'chunked',
        'content_encoding':'identity','body_prefix':'sse_prefix','body_bytes':10,
        'body_truncated':False,'sse_validation':'invalid'}
    stream={'event_type':'response.incomplete','terminal_status':'incomplete',
        'terminal_model_matches':True,'terminal_output_kind':'message',
        'terminal_channel':'final','terminal_content_kind':'output_text',
        'terminal_output_state':'list','finalized_item_count':1,'output_reconstructed':False}
    first={'code':'incomplete_response','phase':'http','turn_admitted':True,
        'unknown_usage':True,'wire_observation':wire,'stream_observation':stream}
    assert reader.first_failure(first)==bridge.project_first_failure(first)
    for name,observation in [('wire_observation',wire),('stream_observation',stream)]:
        for key in observation:
            changed=copy.deepcopy(first);changed[name][key]='private provider text'
            with pytest.raises((ValueError, TypeError)):
                reader.first_failure(changed)


def test_ambiguous_dispatch_consumes_one_shot(tmp_path,monkeypatch):
    root=tmp_path/'.hymem-siwc-lme-diagnostic-invented';root.mkdir()
    for name in ('empty','tmp'):(root/name).mkdir()
    receipt={'unit':'hymem-siwc-lme-diagnostic-invented.service',
             'runtime_path':str(runner.RUNTIME_PATH)}
    launch.write_once(root/'launch-receipt.json',receipt)
    digest=launch.sha(root/'launch-receipt.json')
    monkeypatch.setattr(launch,'HOST_UID',__import__('os').getuid())
    monkeypatch.setattr(launch,'checked_root',lambda root:root)
    monkeypatch.setattr(launch,'host_admission',lambda:None)
    fake=__import__('types').SimpleNamespace(receipt_for=lambda *_:receipt,
        EXECUTION_MARKER=runner.EXECUTION_MARKER)
    monkeypatch.setattr(launch,'verify_sources',lambda _: (fake,{}))
    monkeypatch.setattr(launch,'bus_env',lambda:{})
    dispatched=[]
    def timeout(command,**options):
        dispatched.append(command)
        assert (root/'launch-attempt.json').is_file()
        raise subprocess.TimeoutExpired(command,20)
    monkeypatch.setattr(launch.subprocess,'run',timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        launch.launch(root,digest)
    with pytest.raises(ValueError,match='launch_already_attempted'):
        launch.launch(root,digest)
    assert len(dispatched)==1


@pytest.mark.parametrize('state',['clean_exit','failed_exit_cleaned'])
def test_cleanup_without_terminal_result_is_never_success(tmp_path,monkeypatch,state):
    root=tmp_path/'.hymem-siwc-lme-diagnostic-invented';root.mkdir()
    (root/'launch-attempt.json').write_text(json.dumps({'receipt_sha256':'a'*64,'one_shot':True}))
    monkeypatch.setattr(reader,'checked_root',lambda path:path)
    monkeypatch.setattr(reader,'receipt',lambda *_:{})
    monkeypatch.setattr(reader,'runtime',lambda _: (state,True))
    output=reader.inspect(root,'a'*64)
    assert output['runtime_cleanup_verified'] is True
    assert output['completed_diagnostic_and_clean'] is False
    assert output['status']=='terminal_failure_without_result'


def test_reader_has_no_source_exec_or_auth_import():
    tree=ast.parse(Path(reader.__file__).read_text())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(alias.name for alias in node.names)
        if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            assert node.func.id not in {'exec','eval','compile','__import__'}
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'acquire','refresh','complete','complete_stage','urlopen'}
    assert not any(name.startswith(('benchmarks','hymem','tools','requests','http','urllib','importlib')) for name in imports)


@pytest.mark.parametrize('failure',[False,True])
def test_reader_consumes_actual_runner_campaign_result_and_checkpoint(tmp_path,monkeypatch,failure):
    path=Path(__file__).with_name('test_siwc_lme_runner_root_v1.py')
    spec=importlib.util.spec_from_file_location('siwc_root_campaign_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    run,_,_,_=fixture.rig(tmp_path,monkeypatch,failure=failure,observed=True)
    result=run()
    assert result['budget']['resource_observation']=={'current':1,'peak':9,'limit':256,'denials':0}
    assert result['budget']['resource_fault'] is None
    check=json.loads((tmp_path/'run/diagnostic-checkpoint.json').read_text())
    receipt={'selected_row_sha256':check['manifest']['selected_row_sha256'],
        'source_sha256':{**runner.PINS,**runner.SIWC_PINS}}
    counts,completed,unhealthy,degraded,verified=reader.checkpoint(check,receipt)
    assert reader.terminal(result,verified,receipt,counts,completed,unhealthy) is result
    assert result['diagnostic_complete'] is (not failure)
    assert completed==(3 if failure else 4) and degraded==completed*2
    for field in ['known_tokens','turns']:
        changed=copy.deepcopy(result);changed['budget'][field]+=1
        with pytest.raises(ValueError):
            reader.terminal(changed,verified,receipt,counts,completed,unhealthy)
    changed=copy.deepcopy(result);changed['siwc_observations']['question.0.ordinary']['summary']['known_tokens']+=1
    with pytest.raises(ValueError):
        reader.terminal(changed,verified,receipt,counts,completed,unhealthy)
