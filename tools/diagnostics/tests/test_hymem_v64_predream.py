"""Local pre-dream and metadata upload guards; no network/production actions."""
import json
from pathlib import Path
import pytest
from tools.diagnostics import hymem_v64_predream as pre
from tools.diagnostics import hymem_v64_metadata_upload as upload
from tools.diagnostics import hymem_v64_rollout as r

def census():
    return {'episode_vector_baseline':{**pre.BASELINE,'expected_sha256':'a'*64,'actual_sha256':'b'*64},
            'role_profile_sha256':{'honcho':['h'],'mcp':['m']},
            'processes':{'roles':{'honcho':[{}],'mcp':[{},{}]},
                         'distribution_count':86,'distribution_sha256':'d'},
            'preserved_file_sha256':{str(r.HOME/'.hermes/bin/hymem-server-wrapper'):'w',
                                    str(r.HOME/'.agent37/hooks/post-restart.sh'):'k'}}

def test_predream_is_separate_readonly_dispatch_and_preserves_hard_checks():
    script=pre.container_script({'hymem/x.py':'x'},census())
    compile(script,'<pre>','exec')
    dispatch=script[script.index("\ntry:\n    source();health();preservation()"):]
    assert 'doctor(env)' not in dispatch and 'mcp_probe(env)' not in dispatch
    assert 'vectors==EXPECTED_BASELINE' in script
    assert "'episode_vectors':False" in script
    assert 'mode=ro' in script and 'PRAGMA query_only=ON' in script
    assert 'find_canonical_drift(conn)' in script and 'count_mismatches(conn)' in script
    assert "actual_roles==EXPECTED_ROLES" in script
    assert "len(mcp)==2" in script and "open_dream_run" in script
    assert 'vector_health_passed=False' in script

@pytest.mark.parametrize('field,value',[('surplus_count',9),('missing_count',1),('different_count',1),
                                       ('unverifiable_episodes',16),('actual_sha256',None)])
def test_predream_rejects_unsealed_or_different_baseline(field,value):
    c=census();c['episode_vector_baseline'][field]=value
    with pytest.raises(RuntimeError):pre.container_script({},c)

def test_metadata_remote_compile_exclusive_and_no_launch():
    compile(upload.REMOTE,'<metadata>','exec')
    assert 'os.path.lexists(stage/name)' in upload.REMOTE
    assert 'os.O_EXCL|os.O_NOFOLLOW,0o600' in upload.REMOTE
    assert 'docker' not in upload.REMOTE
    assert "'review_performed':False" in upload.REMOTE

def metadata_files(tmp_path):
    # Real exact candidate flat manifest is safe metadata.
    manifest=Path('/Users/attavanwestreenen/AGprojects/HyMem/docs/patches/2026-09-26-episode-shadow-manifest.json')
    (tmp_path/'candidate-manifest.json').write_bytes(manifest.read_bytes())
    values={
        'fullsuite-gate.json':{'status':'passed','role':'fullsuite','candidate_manifest_sha256':r.CANDIDATE_PIN,
          'root_reviewed':True,'collected':1,'expected_collected':1,'passed':1,'failed':0,'errors':0,'exit_code':0,'cleanup_verified':True},
        'paid-postflight-gate.json':{'status':'passed','role':'paid_postflight','candidate_manifest_sha256':r.CANDIDATE_PIN,
          'root_reviewed':True,'checks':dict.fromkeys(('claims','ledger','canonical','same_generation','integrity','foreign_keys','episode_vectors'),True),
          'aggregation_passed':True,'paid_calls':1},
        'census-reviewed.json':{'root_reviewed':True,'schema':63,'image':r.IMAGE}}
    for name,value in values.items():(tmp_path/name).write_text(json.dumps(value))
    cfg={'candidate_manifest_sha256':r.CANDIDATE_PIN,'candidate_manifest':upload.STAGE+'/candidate-manifest.json',
         'gates':{role:{'path':upload.STAGE+'/'+name,'sha256':r.sha(tmp_path/name)}
                  for role,name in (('fullsuite','fullsuite-gate.json'),('paid_postflight','paid-postflight-gate.json'))},
         'census':{'path':upload.STAGE+'/census-reviewed.json','sha256':r.sha(tmp_path/'census-reviewed.json')}}
    (tmp_path/'rollout-config.json').write_text(json.dumps(cfg))

def test_metadata_prepare_preserves_bytes_and_never_calls_network(tmp_path,monkeypatch):
    metadata_files(tmp_path)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(upload.subprocess,'run',lambda *a,**kw:pytest.fail('unexpected network'))
    result=upload.prepare(tmp_path)
    assert set(result['files'])==set(upload.NAMES)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}

def test_metadata_uploader_cannot_review_receipts(tmp_path):
    metadata_files(tmp_path)
    path=tmp_path/'census-reviewed.json'
    value=json.loads(path.read_text());value['root_reviewed']=False
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='external_root_review_required'):upload.prepare(tmp_path)
