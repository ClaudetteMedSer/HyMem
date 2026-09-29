"""No-network replay negative controls."""
import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('r8_audit',Path(__file__).resolve().parents[1]/'lme_r8_chunk_capture_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def request():
    return SimpleNamespace(system='system',user='user',temperature=0,max_tokens=100,response_format='json')


def wire():
    return {'model':'deepseek-flash','messages':[{'role':'system','content':'system'},
        {'role':'user','content':'user'}],'temperature':0,'max_tokens':100,
        'thinking':{'type':'disabled'},'response_format':{'type':'json_object'}}


def answer(finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':'{"triples":[]}'}}]}


def captured(path,payload,**metadata):
    raw=json.dumps(payload).encode()
    path.write_text(json.dumps({'body_base64':base64.b64encode(raw).decode(),
                               'body_sha256':audit.digest(raw),**metadata}))


def test_exact_request_only():
    client=audit.ReplayClient([(audit.CHUNKS[0],wire(),answer())])
    client.chunk_id=audit.CHUNKS[0]
    assert client.complete(request()) == '{"triples":[]}'
    assert client.position == client.successful_responses == 1
    with pytest.raises(audit.AuditMismatch,match='replay_extra_request'):
        client.complete(request())


@pytest.mark.parametrize('field,value',[('system','changed'),('user','changed'),
    ('temperature',1),('max_tokens',101),('response_format','text')])
def test_changed_request_not_blindly_fed(field,value):
    client=audit.ReplayClient([(audit.CHUNKS[0],wire(),answer())])
    client.chunk_id=audit.CHUNKS[0]
    req=request()
    setattr(req,field,value)
    with pytest.raises(audit.AuditMismatch,match='replay_request_mismatch'):
        client.complete(req)
    assert client.position == 0


@pytest.mark.parametrize('finish',['length','content_filter',None])
def test_non_stop_cannot_simulate_wrong_extraction_path(finish):
    client=audit.ReplayClient([(audit.CHUNKS[0],wire(),answer(finish))])
    client.chunk_id=audit.CHUNKS[0]
    with pytest.raises(audit.AuditMismatch,match='unsupported_recorded_finish_reason'):
        client.complete(request())
    assert client.position == 1 and client.successful_responses == 0


def test_capture_hash_altered(tmp_path):
    path=tmp_path/'body.json'
    captured(path,wire())
    wrapper=json.loads(path.read_text())
    wrapper['body_sha256']='0'*64
    path.write_text(json.dumps(wrapper))
    with pytest.raises(audit.AuditMismatch,match='capture_body_hash'):
        audit.body(path)


def test_pair_order_and_truncated_sequence(tmp_path):
    for index,cid in enumerate(audit.CHUNKS,1):
        captured(tmp_path/f'{index:03d}-request.json',wire(),chunk_id=cid)
        captured(tmp_path/f'{index:03d}-response.json',answer(),status_code=200)
    assert len(audit.sequence(tmp_path)) == 2
    (tmp_path/'002-response.json').unlink()
    with pytest.raises(audit.AuditMismatch,match='capture_pair_count'):
        audit.sequence(tmp_path)


def test_counter_gap(tmp_path):
    captured(tmp_path/'002-request.json',wire(),chunk_id=audit.CHUNKS[0])
    captured(tmp_path/'002-response.json',answer(),status_code=200)
    with pytest.raises(audit.AuditMismatch,match='capture_order'):
        audit.sequence(tmp_path)
