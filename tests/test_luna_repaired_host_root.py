"""Root-owned new source-closure, receipt and recursive-cleanup controls."""
import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_lme_diagnostic_v10 as runner
from tools.diagnostics import luna_lme_diagnostic_bundle_v9 as bundle
from tools.diagnostics import luna_lme_diagnostic_host_preflight_v9 as host
from tools.diagnostics import luna_lme_diagnostic_launch_v9 as launch
from tools.diagnostics import luna_lme_diagnostic_progress_v10 as reader
from tools.diagnostics import luna_lme_diagnostic_launch_v8 as old_launch
from tools.diagnostics import luna_instrumented_source_install_v2 as installer

REPO=Path(__file__).resolve().parents[1]
FROZEN=Path('/private/tmp/hymem-lme-instrumented-IjmZdT/bundle')
REPAIRED=Path('/private/tmp/hymem-staged-proxy-v1-8_ac647k/bundle')


@pytest.fixture(scope='module')
def assembly(tmp_path_factory):
    path=tmp_path_factory.mktemp('root-repaired-four')/'.hymem-lme-diagnostic-rootfixture'
    result=bundle.assemble(repo=REPO,accepted_code=FROZEN/'code',
        candidate=REPAIRED/'candidate',map_path=REPAIRED/'source-map.json',output=path)
    assert result['candidate_files']==514 and len(result['code_sha256'])==16
    assert result['model_calls']==0 and result['dataset_present'] is False
    assert result['binary_present'] is False and result['launch_receipt_present'] is False
    return path


def test_source_closure_and_actual_isolated_import(assembly):
    manifest=host.source_manifest(assembly)
    assert len(manifest)==531
    assert sum(p.startswith('candidate/') for p in manifest)==514
    assert sum(p.startswith('code/') for p in manifest)==16
    assert {p.removeprefix('code/'):h for p,h in manifest.items() if p.startswith('code/')}==reader.PINS
    for relative,digest in manifest.items():
        assert hashlib.sha256((assembly/relative).read_bytes()).hexdigest()==digest
    script=r'''
import importlib.util,sys
from pathlib import Path
root=Path(sys.argv[1]);path=root/'code/tools/diagnostics/luna_lme_diagnostic_v10.py'
spec=importlib.util.spec_from_file_location('root_new_four',path)
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
loaded=r.import_source_only(root,root/'source-map.json',r.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and loaded['questions']==[]
assert loaded['warm'].BILLING_POLICY==r.BILLING_POLICY
assert loaded['staged'].observer is loaded['observer']
assert loaded['staged'].warm is loaded['warm']
assert hasattr(loaded['dream_runner']._CountingPhase1LLM,'complete_stage')
assert hasattr(loaded['dream_runner']._HeartbeatLLMClient,'complete_stage')
assert not (root/'launch-receipt.json').exists()
print('verified')
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',script,str(assembly)],
        capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='verified'


def test_remote_decoder_matches_exact_source_archive(assembly,monkeypatch):
    nodes=[]
    for node in ast.parse(host.REMOTE).body:
        statement=ast.get_source_segment(host.REMOTE,node) or ''
        if statement.startswith('need(os.getuid()'): continue
        if statement.startswith('need(regular(DATASET)'): break
        nodes.append(node)
    else: raise AssertionError('No dataset/side-effect boundary')
    monkeypatch.setattr(sys,'stdin',io.TextIOWrapper(io.BytesIO(host.archive_bytes(assembly))))
    namespace={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
        '<root-source-decoder>','exec'),namespace)
    assert namespace['manifest']==host.source_manifest(assembly)


def test_headless_command_preserves_all_resource_and_account_limits():
    root=Path('/invented');receipt={'unit':'invented.service'}
    assert launch.command(root,receipt,'a'*64)==[
        arg.replace('luna_lme_diagnostic_v8.py','luna_lme_diagnostic_v10.py').replace(
            old_launch.INVENTORY_SHA256,runner.ACCEPTED_INVENTORY_SHA256)
        for arg in old_launch.command(root,receipt,'a'*64)]
    assert launch.RUNNER_SHA256==reader.RUNNER_SHA256==hashlib.sha256(
        (REPO/'tools/diagnostics/luna_lme_diagnostic_v10.py').read_bytes()).hexdigest()


def test_real_source_receipt_roundtrip_and_typed_mutations(assembly,monkeypatch):
    spec=importlib.util.spec_from_file_location('root_receipt_in_bundle',
        assembly/'code/tools/diagnostics/luna_lme_diagnostic_v10.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows=[{'question_id':f'invented-{i}'} for i in range(4)]
    dataset=assembly/'unused-dataset';binary=assembly/'unused-binary'
    dataset.write_text('invented dataset bytes')
    binary.write_text('invented runtime bytes')
    original=runner._sha
    monkeypatch.setattr(runner,'_sha',lambda p: runner.DATASET_SHA256 if p==dataset
        else runner.BINARY_SHA256 if p==binary else original(p))
    original_reader_sha=reader._sha
    monkeypatch.setattr(reader,'DATASET',dataset)
    monkeypatch.setattr(reader,'BINARY',binary)
    monkeypatch.setattr(reader,'_sha',lambda p: reader.DATASET_SHA256 if p==dataset
        else reader.BINARY_SHA256 if p==binary else original_reader_sha(p))
    loaded={'source_only':False,'root':assembly,'questions':rows,
        'dataset':dataset,'binary':binary,'protocol':None,
        'prior':SimpleNamespace(SelectedQuestions=lambda *args:rows)}
    receipt=runner.receipt_for(assembly,loaded)
    monkeypatch.setattr(launch,'HOST_UID',os.getuid())
    monkeypatch.setattr(launch,'HOST_ROOT',assembly.parent)
    assert launch.receipt_for(assembly,runner,loaded)==receipt
    path=assembly/'launch-receipt.json'
    def write(value):
        data=runner._canonical(value);path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    digest=write(receipt)
    assert reader._receipt(assembly,digest)==receipt
    for field,value in [('workers',4.0),('selected_count',True),('indexing_seconds',10800.0),
                        ('one_shot',1),('model','different'),('extra','PRIVATE')]:
        changed={**receipt,field:value}
        with pytest.raises(ValueError): reader._receipt(assembly,write(changed))
    write(receipt)
    with pytest.raises(ValueError,match='launch_selection_invalid'):
        runner.receipt_for(assembly,{**loaded,'questions':list(reversed(rows))})


def test_recursive_reader_cleanup_rejects_live_descendants(tmp_path,monkeypatch):
    monkeypatch.setattr(reader,'CGROUP_ROOT',tmp_path)
    group=tmp_path/'test.service';group.mkdir()
    def empty(path):
        (path/'cgroup.procs').write_text('')
        (path/'cgroup.threads').write_text('')
        (path/'cgroup.events').write_text('populated 0\nfrozen 0\n')
    empty(group)
    assert reader._cgroup_empty(group)
    child=group/'child';child.mkdir();empty(child)
    assert reader._cgroup_empty(group)
    (child/'cgroup.threads').write_text('123\n')
    assert not reader._cgroup_empty(group)
    (child/'cgroup.threads').write_text('')
    (child/'cgroup.events').write_text('populated 1\n')
    assert not reader._cgroup_empty(group)
    (child/'cgroup.events').write_text('populated 0\n')
    (group/'escape').symlink_to(tmp_path)
    assert not reader._cgroup_empty(group)


@pytest.mark.parametrize('mutation',['none','missing_execution','row_order','cleanup','usage','observer'])
def test_real_four_row_reader_completion_gate(tmp_path,monkeypatch,mutation):
    spec=importlib.util.spec_from_file_location('root_repaired_fixture',
        Path(__file__).with_name('test_luna_instrumented_reader_root.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fixture.reader=reader;fixture.runner=runner
    summary=fixture.summary.__wrapped__(monkeypatch)
    root,terminal=fixture.finished.__wrapped__(tmp_path,monkeypatch,summary)
    checkpoint_path=root/'run/diagnostic-checkpoint.json'
    checkpoint=json.loads(checkpoint_path.read_text())
    receipt={'selected_count':4,'selected_row_sha256':checkpoint['manifest']['selected_row_sha256']}
    monkeypatch.setattr(reader,'_receipt',lambda *args:receipt)
    marker=root/runner.EXECUTION_MARKER
    marker.write_bytes(runner._canonical({'receipt_sha256':'a'*64,'execution_started':True}))
    if mutation=='missing_execution': marker.unlink()
    elif mutation=='row_order':
        receipt['selected_row_sha256']=list(reversed(receipt['selected_row_sha256']))
    elif mutation=='cleanup': monkeypatch.setattr(reader,'_runtime',lambda _: 'unverified')
    elif mutation=='usage': terminal['budget']['usage_complete']=False
    elif mutation=='observer':
        terminal['timeout_observations']['question.0.ordinary']={'status':'unknown'}
    fixture.prior._json(root/'run/diagnostic-result.json',terminal)
    if mutation in {'missing_execution','row_order'}:
        with pytest.raises(ValueError): reader.inspect(root,'a'*64)
    else:
        result=reader.inspect(root,'a'*64)
        assert result['completed_diagnostic_and_clean'] is (mutation=='none')
        assert result['scored_count']==4 and result['correct_count']==2
        assert result['strict_indexing_healthy_for_all'] is False
        assert result['canary_model_gold_match'] is False
        assert result['summary_degraded_sessions_total']==1


def test_no_terminal_can_still_prove_cleanup_but_not_success(tmp_path,monkeypatch):
    root=tmp_path/'invented';root.mkdir()
    monkeypatch.setattr(reader,'_root',lambda r:r)
    monkeypatch.setattr(reader,'_receipt',lambda *args:{'selected_count':4})
    monkeypatch.setattr(reader,'_runtime',lambda *args:'unverified')
    monkeypatch.setattr(reader,'_failed_exit_cleanup',lambda *args:True)
    (root/'launch-attempt.json').write_bytes(runner._canonical({'receipt_sha256':'a'*64,'one_shot':True}))
    (root/runner.EXECUTION_MARKER).write_bytes(runner._canonical({'receipt_sha256':'a'*64,'execution_started':True}))
    result=reader.inspect(root,'a'*64)
    assert result['runtime_cleanup_verified'] is True
    assert result['completed_diagnostic_and_clean'] is False
    assert result['scored_count'] is result['known_turns'] is result['known_tokens'] is None


def test_preflight_success_projection_never_exports_unknown_fields():
    value={'schema':'luna-lme-diagnostic-host-preflight-v9',
        'root':'/home/atta/.hymem-lme-diagnostic-preflight-invented',
        'candidate_files':514,'code_files':16,'inventory_sha256':runner.ACCEPTED_INVENTORY_SHA256,
        'binary_sha256':runner.BINARY_SHA256,'dataset_sha256':runner.DATASET_SHA256,
        'preflight_verified':True,'selected_count':4,'model_calls':0}
    assert host._success_projection(value)==value
    for change in ({'extra':'PRIVATE'},{'model_calls':False},{'code_files':16.0},
                   {'root':'PRIVATE'},{'binary_sha256':'a'*64}):
        assert host._success_projection({**value,**change}) is None


def test_installer_pins_and_offline_remote_only_prepare(tmp_path,monkeypatch,capsys):
    assert set(installer.FILES)=={'luna_lme_diagnostic_launch_v9.py'}
    files={}
    for name,(relative,digest) in installer.FILES.items():
        data=(REPO/relative).read_bytes()
        assert hashlib.sha256(data).hexdigest()==digest
        files[name]={'sha256':digest,'data':base64.b64encode(data).decode('ascii')}
    root=tmp_path/'.hymem-lme-diagnostic-preflight-abcdefgh';root.mkdir(mode=0o700)
    (root/'candidate').mkdir();(root/'code').mkdir();(root/'source-map.json').write_text('{}')
    commands=[]
    def prepare(args,**kwargs):
        commands.append(args)
        assert '--prepare-root' in args and '--launch-root' not in args and '--run' not in args
        assert Path(args[3]).read_bytes()==(REPO/'tools/diagnostics/luna_lme_diagnostic_launch_v9.py').read_bytes()
        value={'prepared':True,'model_calls':0,'root':str(root),
            'unit':'hymem-luna-lme-diagnostic-preflight-abcdefgh.service','receipt_sha256':'a'*64}
        return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='PRIVATE')
    monkeypatch.setattr(subprocess,'run',prepare)
    remote=installer.REMOTE.replace('/home/atta/',str(tmp_path)+'/').replace(
        'os.getuid()==1000',f'os.getuid()=={os.getuid()}').replace(
        'st_uid==1000',f'st_uid=={os.getuid()}')
    exec(compile(remote,'<root-offline-installer>','exec'),{'PAYLOAD':{'root':str(root),'files':files}})
    assert len(commands)==1
    output=capsys.readouterr().out
    assert 'PRIVATE' not in output
    assert json.loads(output)['prepared'] is True
    with pytest.raises((AssertionError,FileExistsError)):
        exec(compile(remote,'<root-offline-installer>','exec'),{'PAYLOAD':{'root':str(root),'files':files}})
    assert len(commands)==1
