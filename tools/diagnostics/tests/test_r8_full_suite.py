"""Offline admission and Docker command controls, never launch Docker."""
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('r8_suite',Path(__file__).resolve().parents[1]/'lme_r8_full_suite.py')
suite=importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def fixture_tree(tmp_path):
    tree=tmp_path/'tree'
    (tree/'hymem').mkdir(parents=True)
    (tree/'tests').mkdir()
    (tree/'hymem/__init__.py').write_text('# source')
    (tree/'tests/test_summary_overview_policy.py').write_text('def test_ok(): pass')
    (tree/'pyproject.toml').write_text('[project]\nname="test"')
    return tree


def test_seal_and_inventory_admission(tmp_path):
    tree=fixture_tree(tmp_path)
    pin=suite.digest(suite.inventory(tree,clean=True))
    root=tmp_path/'package'
    result=suite.seal(tree,root,pin,7798)
    assert result['launched'] is False and result['provider_calls']==0
    assert suite.verify(root,result['manifest_sha256'])['timeout_seconds']==10800
    (root/'candidate/hymem/__init__.py').write_text('# changed')
    with pytest.raises(RuntimeError,match='source_drift'):
        suite.verify(root,result['manifest_sha256'])


def test_wrong_source_pin_before_creation(tmp_path):
    tree=fixture_tree(tmp_path)
    root=tmp_path/'package'
    with pytest.raises(RuntimeError,match='source_inventory_pin'):
        suite.seal(tree,root,'0'*64,7798)
    assert not root.exists()


@pytest.mark.parametrize('count',[None,True,0,7763])
def test_collection_bounds(tmp_path,count):
    with pytest.raises(RuntimeError,match='collection_admission'):
        suite.seal(fixture_tree(tmp_path),tmp_path/'package','0'*64,count)


def test_network_none_no_credentials_configuration(tmp_path):
    command,mounts=suite.configuration(tmp_path,'a'*64)
    assert command[command.index('--network')+1]=='none'
    assert command[command.index('--user')+1]=='1000:1000'
    assert '--read-only' in command and command[command.index('--pull')+1]=='never'
    assert command[-2:]==['--pin','a'*64]
    assert set(mounts)=={'/candidate','/diag','/work','/home/node/hymem-env'}
    assert mounts['/candidate'][1] is False and mounts['/work'][1] is True
    assert not any('DEEPSEEK' in arg or 'OPENAI' in arg for arg in command)


def test_manifest_timeout_cannot_drift(tmp_path):
    tree=fixture_tree(tmp_path)
    root=tmp_path/'package'
    result=suite.seal(tree,root,suite.digest(suite.inventory(tree,clean=True)),7798)
    path=root/'diag/manifest.json'
    manifest=json.loads(path.read_text())
    manifest['timeout_seconds']=10801
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='manifest_bounds'):
        suite.verify(root,suite.sha(path))
