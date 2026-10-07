"""Independent inactive host/provenance checks; no external command is run."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import types

import pytest

from tools.diagnostics import luna_staged_bundle_v1 as b

REPO=Path(__file__).resolve().parents[3]
CANDIDATE=Path('/private/tmp/hymem-staged-v1-root-fZvNIRgs/candidate')
STAMP=CANDIDATE.with_name('map.json')
PENDING=tuple('tools/diagnostics/luna_staged_'+name+'_v1.py'
              for name in ('run','progress','replay'))
HOST='tools/diagnostics/luna_staged_host_v1.py'


def digest(raw):return hashlib.sha256(raw).hexdigest()


def host_fixture():
    code=b.collect_sources(REPO)
    pins={name:digest(raw) for name,raw in code.items()}
    # These are hashes only, never executable placeholder modules.
    pins.update({name:digest(('metadata-only test '+name).encode()) for name in PENDING})
    raw=b.derive_host(pins)
    module=types.ModuleType('root_staged_host')
    module.__file__='/nonexistent/code/'+HOST
    exec(compile(raw,module.__file__,'exec'),module.__dict__)
    return code,pins,raw,module


def test_candidate_inventory_exact_and_helpers_bound():
    mapping=b.validate_candidate()
    assert len(mapping)==514
    assert mapping==json.loads(STAMP.read_bytes())['source_sha256']
    assert digest(STAMP.read_bytes())=='228c76399395323c10be9a2107d4d5bf6004da64cda8db556a8d2ff4b0b0dfdf'
    for name in ('grounding_classification_v4','grounding_staged_v1','grounding_staged_gate_v1'):
        relative='hymem/extraction/'+name+'.py'
        assert mapping[relative]==digest((REPO/relative).read_bytes())


@pytest.mark.parametrize('fault',['extra','missing','changed','symlink','stamp'])
def test_candidate_substitution_fails_before_any_execution(tmp_path,fault):
    candidate=tmp_path/'candidate';stamp=tmp_path/'map.json'
    shutil.copytree(CANDIDATE,candidate);shutil.copyfile(STAMP,stamp)
    target=candidate/'hymem/extraction/chunk.py'
    if fault=='extra':(candidate/'unreviewed.py').write_text('raise RuntimeError()')
    elif fault=='missing':target.unlink()
    elif fault=='changed':target.write_bytes(target.read_bytes()+b'\n# drift\n')
    elif fault=='symlink':target.unlink();target.symlink_to(CANDIDATE/'hymem/extraction/chunk.py')
    else:stamp.write_bytes(stamp.read_bytes()+b' ')
    with pytest.raises(ValueError):b.validate_candidate(candidate,stamp)


def test_real_pending_dependencies_required_no_bundle_or_placeholders(tmp_path):
    target=tmp_path/'not-sealed'
    repo=tmp_path/'source';repo.mkdir()
    for name in b.LOCAL:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((REPO/name).read_bytes())
    with pytest.raises((ValueError,FileNotFoundError)):b.prepare(repo,target)
    assert not target.exists()


def test_exact_staged_transport_rebind_and_pinned_core():
    code=b.collect_sources(REPO)
    path='benchmarks/codex_subscription_staged_v1.py'
    original=(REPO/path).read_bytes()
    assert code[path]==original.replace(b'_ROOT = Path(__file__).resolve().parents[1]',
        b'_ROOT = Path(__file__).resolve().parents[2] / "candidate"')
    path='tools/diagnostics/luna_staged_core_v1.py'
    assert digest(code[path])=='a389f6e8a522d878a140234ac9c040f2e3daab77bfc7cbaeadf0f4d5b12676bd'
    assert all(name not in code for name in PENDING)


def test_new_host_receipt_preserves_limits_and_separates_eight_units(monkeypatch,tmp_path):
    code,pins,raw,host=host_fixture()
    monkeypatch.setattr(host,'root_valid',lambda _:True)
    root=Path('/home/atta/.hymem-luna-staged-probe-root1234')
    receipt=host.receipt_for(root,pins,'1'*64)
    assert receipt['model']=='gpt-6-luna' and receipt['subscription_only'] is True
    assert receipt['quota_floor_percent']==25
    assert receipt['inventory_sha256']==digest(STAMP.read_bytes())
    assert receipt['extraction_identity']=='hymem-extraction-contract-sha256-v1:94a1adfc028694e9b1868ec8712baff38d4d7a99a9d2e3547c66819711890f95'
    assert receipt['limits']['units']==8
    assert {k:receipt['limits'][k] for k in ('turns','known_tokens','seconds')}==dict(turns=29,known_tokens=500000,seconds=1800)
    assert host.POLICY==dict(runtime_max_seconds=1930,timeout_stop_seconds=10,
        tasks_max=256,memory_max_bytes=4294967296,cpu_quota_percent=200,
        oom_policy='kill',kill_mode='control-group',restart='no',remain_after_exit=True,
        memory_admission_bytes=6*1024**3,disk_admission_bytes=20*1024**3)
    assert receipt['source_sha256']==pins
    assert 'luna-claim-task-probe' not in receipt['schema']+receipt['unit']
    tree=ast.parse(raw)
    body={node.name:ast.dump(node,include_attributes=False)
          for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old=ast.parse((Path('/private/tmp/hymem-claim-task-accepted-KXzUiPyJ/bundle/code/tools/diagnostics/luna_semantic_probe_host.py')).read_bytes())
    old_body={node.name:ast.dump(node,include_attributes=False)
          for node in old.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    # These safety functions must remain byte-semantically identical.
    for name in ('launch','host_admission','write_once','root_valid','strict_equal'):
        assert body[name]==old_body[name],name


@pytest.mark.parametrize('fault',['missing','invalid_hash','traversal'])
def test_host_source_manifest_rejects_unbound_paths(fault):
    _,pins,_,_=host_fixture()
    if fault=='missing':pins.pop(PENDING[0])
    elif fault=='invalid_hash':pins[PENDING[0]]='z'*64
    else:pins['../unexpected']='1'*64
    with pytest.raises(ValueError):b.derive_host(pins)


@pytest.mark.parametrize('relative',PENDING)
def test_pending_source_bytes_cannot_change_after_host_derivation(tmp_path,relative):
    code,pins,raw,host=host_fixture()
    for name in PENDING:code[name]=('metadata-only test '+name).encode()
    code[HOST]=raw
    for name,content in code.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    initial=host.source_pins(tmp_path)
    assert all(initial[name]==digest(content) for name,content in code.items())
    path=tmp_path/relative
    path.write_bytes(path.read_bytes()+b'\nUNREVIEWED\n')
    with pytest.raises(ValueError):host.source_pins(tmp_path)


def test_host_derives_exact_fresh_514_candidate_without_external_commands(monkeypatch,tmp_path):
    code,pins,raw,host=host_fixture()
    stage=tmp_path/'stage';stage.mkdir(mode=0o700)
    for name in PENDING:code[name]=('metadata-only test '+name).encode()
    code[HOST]=raw
    for name,content in code.items():
        path=stage/'code'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    monkeypatch.setattr(host,'TASK_HOME_ROOT',tmp_path)
    monkeypatch.setattr(host,'UID',os.getuid())
    monkeypatch.setattr(host.sys,'platform','linux')
    monkeypatch.setattr(host,'host_admission',lambda:None)
    observed=Path('/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi')
    monkeypatch.setattr(host,'OLD_CANDIDATE',observed/'candidate')
    monkeypatch.setattr(host,'OLD_MAP',observed/'headless-grounding-source-map.json')
    monkeypatch.setattr(host,'ACCEPTED_CANDIDATE',Path('/private/tmp/hymem-semantic-step2-v2-candidate-20260929'))
    monkeypatch.setattr(host,'ACCEPTED_MAP',Path('/private/tmp/hymem-semantic-step2-v2-map-20260929.json'))
    # No private records are used: substitute three separately hashed invented files.
    evidence=tmp_path/'invented-observed';(evidence/'run').mkdir(parents=True)
    hashes={}
    for key,path in [('evidence',evidence/'run/private-canary-evidence.json'),
                     ('receipt',evidence/'launch-receipt.json'),('result',evidence/'run/private-result.json')]:
        path.write_bytes(b'{}');hashes[key]=digest(b'{}')
    monkeypatch.setattr(host,'OBSERVED',evidence)
    monkeypatch.setattr(host,'RETAINED',hashes)
    binary=tmp_path/'not-executable';binary.write_bytes(b'not an executable')
    monkeypatch.setattr(host,'BINARY',binary)
    def deny(*a,**k):raise AssertionError('external command forbidden')
    monkeypatch.setattr(host.subprocess,'run',deny)
    result=host.prepare(stage)
    assert result['prepared'] and not result['launched'] and result['model_calls']==0
    root=Path(result['root'])
    mapping=b.validate_candidate(root/'candidate',root/'candidate-source-map.json')
    assert len(mapping)==514
    assert not (root/'run').exists() and not (root/'launch-attempt.json').exists()
    receipt=json.loads((root/'launch-receipt.json').read_bytes())
    assert receipt['source_sha256']==host.source_pins(root/'code')
    assert host.sha(root/'launch-receipt.json')==result['receipt_sha256']
