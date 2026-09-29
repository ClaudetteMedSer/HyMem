"""Offline controls; no SSH, database, credentials, Docker or provider calls."""
import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


replay = module('lme_r8_chunk_replay')
host = module('lme_r8_chunk_host')


def fixture_api(spans=((0,4),(4,8)), scope='s', content=None):
    original = {'source_message_id':7, 'content':'abcdefgh', 'source_role':'user',
                'source_record_version':'original', 'source_session_id':'s'}
    records = ((7,json.dumps(original)),)
    leaves = []
    for start,end in spans:
        payload = {**original,'content':original['content'][start:end] if content is None else content,
                   'source_content_start':start,'source_content_end':end,'source_session_id':scope}
        leaves.append((SimpleNamespace(text='x',source_records=((7,json.dumps(payload)),)),1))
    api = SimpleNamespace(_ExtractionUnit=lambda **kw:SimpleNamespace(**kw),
        _prepartition=lambda _: (leaves,None), _source_payload=lambda r:json.loads(r[1]),
        _MAX_SPLIT_DEPTH=8,_MAX_UNSPLITTABLE_INPUT_CHARS=8000,_MAX_PREPARTITION_LEAVES=32)
    return records,api


def test_exact_partition():
    assert len(replay.partition_proof(*fixture_api())) == 2


@pytest.mark.parametrize('kwargs',[{'spans':((0,4),(5,8))},{'spans':((0,5),(4,8))},
    {'spans':((0,4),)}, {'spans':((4,8),(0,4))}, {'scope':'other'}, {'content':'invented'}])
def test_partition_rejects_drift(kwargs):
    with pytest.raises(RuntimeError):
        replay.partition_proof(*fixture_api(**kwargs))


def test_wire_capture_private_bodies_no_headers(tmp_path):
    capture = replay.WireCapture(tmp_path, SimpleNamespace(check=lambda:None))
    body = json.dumps({'model':replay.MODEL,'thinking':{'type':'disabled'},'messages':[{'content':'private'}]}).encode()
    request = SimpleNamespace(url=replay.ENDPOINT+'/chat/completions',extensions={},
                              read=lambda:body,headers={'Authorization':'SECRET'})
    capture.request(request)
    response = SimpleNamespace(request=request,status_code=200,read=lambda:b'{"choices":[]}')
    capture.response(response)
    saved = json.loads((tmp_path/'001-request.json').read_text())
    assert base64.b64decode(saved['body_base64']) == body
    assert 'SECRET' not in ''.join(p.read_text() for p in tmp_path.iterdir())
    assert capture.attempts == 1


def test_wire_cap_before_body_or_network(tmp_path):
    capture = replay.WireCapture(tmp_path,SimpleNamespace(check=lambda:None))
    capture.attempts = 288
    request = SimpleNamespace(url=replay.ENDPOINT+'/chat/completions',extensions={},
                              read=lambda:pytest.fail('cap must precede read'))
    with pytest.raises(RuntimeError,match='http_cap'):
        capture.request(request)


def test_wrong_endpoint_rejected(tmp_path):
    capture = replay.WireCapture(tmp_path,SimpleNamespace(check=lambda:None))
    with pytest.raises(RuntimeError,match='wire_endpoint'):
        capture.request(SimpleNamespace(url='https://other.invalid/chat/completions'))


def test_seal_has_fresh_pins_and_no_caches(tmp_path):
    tree=tmp_path/'source'
    (tree/'hymem/extraction').mkdir(parents=True)
    (tree/'benchmarks').mkdir()
    (tree/'hymem/extraction/chunk.py').write_text('# source')
    (tree/'hymem/extraction/cache.pyc').write_bytes(b'cache')
    (tree/'pyproject.toml').write_text('[project]\nname="test"\n')
    records={c:'a'*64 for c in reversed(replay.CHUNKS)}
    root=tmp_path/'sealed'
    result=host.seal(tree,root,'candidate',records,{'hymem.sqlite':'b'*64})
    manifest=json.loads((root/'diag/manifest.json').read_text())
    assert result['api_calls']==0 and manifest['source_label']=='candidate'
    assert set(manifest['chunks'])==set(replay.CHUNKS)
    assert 'hymem/extraction/cache.pyc' not in manifest['source_sha256']
    assert replay.sha(root/'diag/manifest.json') == result['manifest_sha256']
    with pytest.raises(RuntimeError,match='new_package_required'):
        host.seal(tree,root,'baseline',records,{'hymem.sqlite':'b'*64})
