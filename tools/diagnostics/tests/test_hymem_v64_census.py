"""Census and postdeploy preparation controls; never execute remote actions."""
import json
from pathlib import Path
import pytest
from tools.diagnostics import hymem_v64_census as census
from tools.diagnostics import hymem_v64_postdeploy as post
from tools.diagnostics import hymem_v64_rollout as r

def test_readonly_process_script_compiles_and_exports_hashes_only():
    compile(census.PROCESS_SCRIPT,'<process>','exec')
    assert "'profile_sha256':digest(profile)" in census.PROCESS_SCRIPT
    assert "raw_environ_sha256" in census.PROCESS_SCRIPT
    assert "'env':env" not in census.PROCESS_SCRIPT
    assert "'HYMEM_AGGREGATION_NODES_ENABLED'" not in census.PROCESS_SCRIPT

def test_upload_plan_never_executes_commands(tmp_path,monkeypatch):
    monkeypatch.setattr(census.subprocess,'run',lambda *a,**kw:pytest.fail('unexpected process'))
    source=tmp_path/'helper.py';source.write_text('reviewed')
    value=census.upload_plan([source],str(r.HOME/'.hermes/benchmarks/hymem-v64-rollout-test'))
    assert value['status']=='prepared_only'
    assert value['files'][0]['sha256']==r.sha(source)

@pytest.mark.parametrize('target',['/tmp/stage','/opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-rollout-test/../x'])
def test_upload_plan_rejects_broad_targets(tmp_path,target):
    source=tmp_path/'helper.py';source.write_text('reviewed')
    with pytest.raises(RuntimeError):census.upload_plan([source],target)

def test_postdeploy_binds_full481_schema64_and_separate_role_profiles():
    baseline={'role_profile_sha256':{'honcho':['h'],'mcp':['m']},
              'processes':{'distribution_count':86,'distribution_sha256':'d'},
              'preserved_file_sha256':{str(r.HOME/'.hermes/bin/hymem-server-wrapper'):'w',
                                      str(r.HOME/'.agent37/hooks/post-restart.sh'):'k'}}
    script=post.container_script({'hymem/example.py':'x'},baseline)
    compile(script,'<post>','exec')
    assert "version==64" in script and "'schema_version':64" in script
    assert "len(EXPECTED_FILES)==481" in script
    assert "actual_roles==EXPECTED_ROLES" in script
    assert "version==63" not in script
    assert 'strict_episode_vectors(conn,db)' in script
    assert 'db.vec_episodes_aligned(conn)' not in script
    assert 'find_canonical_drift(conn)' in script
    assert 'count_mismatches(conn)' in script
    assert "EXPECTED_ROLES={'honcho': ['h'], 'mcp': ['m']}" in script

def test_postdeploy_template_pin_cannot_drift(monkeypatch):
    monkeypatch.setattr(post,'TEMPLATE_SHA256','0'*64)
    with pytest.raises(RuntimeError,match='postdeploy_template_pin_drift'):
        post.container_script({}, {})

def strict_scope(monkeypatch,expected,actual,extension=True,table=True):
    import sys
    from types import ModuleType,SimpleNamespace
    from tools.diagnostics.hymem_v64_vector_check import CHECKER
    aggregate=ModuleType('hymem.dreaming.aggregate')
    aggregate.load_clusterable_episodes=lambda *a,**kw:[{'rowid':key,'vector':value} for key,value in expected.items()]
    monkeypatch.setitem(sys.modules,'hymem.dreaming.aggregate',aggregate)
    class Conn:
        def execute(self,sql):
            if 'vec_dim' in sql:return SimpleNamespace(fetchone=lambda:(1,))
            if 'vec_model' in sql:return SimpleNamespace(fetchone=lambda:('hymem-embedding-producer-v1:'+'0'*64,))
            return list(actual.items())
    db=SimpleNamespace(_load_vec_extension=lambda c:extension,has_vec_table=lambda *a,**kw:table,
                       _finite_vec=lambda value,dim:value,_pack_vector=lambda value:value)
    scope={};exec(CHECKER,scope)
    return scope['strict_episode_vectors'](Conn(),db)

@pytest.mark.parametrize('extension,table',[(False,True),(True,False)])
def test_strict_vector_unverifiable_fails_closed(monkeypatch,extension,table):
    with pytest.raises(RuntimeError):
        strict_scope(monkeypatch,{1:b'a'},{1:b'a'},extension,table)

def test_strict_vector_reports_surplus_without_modifying_store(monkeypatch):
    expected={1:b'a'};actual={1:b'a',2:b'b'}
    result=strict_scope(monkeypatch,expected,actual)
    assert result['verifiable'] is True and result['aligned'] is False
    assert result['surplus_count']==1 and result['missing_count']==0 and result['different_count']==0
    assert expected=={1:b'a'} and actual=={1:b'a',2:b'b'}

def test_strict_vector_reports_missing_and_different(monkeypatch):
    result=strict_scope(monkeypatch,{1:b'a',2:b'b'},{1:b'c'})
    assert result['missing_count']==1 and result['different_count']==1

def test_strict_vector_reports_unverifiable_episodes_and_remains_fail_closed(monkeypatch):
    result=strict_scope(monkeypatch,{1:b'a',2:None},{1:b'a',3:b'b'})
    assert result['available'] is True and result['verifiable'] is False and result['aligned'] is False
    assert result['unverifiable_episodes']==1 and result['expected_count']==1
    assert result['actual_count']==2 and result['surplus_count']==1
