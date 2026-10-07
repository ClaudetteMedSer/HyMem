"""Finite metadata helpers tested against actual synthetic runner journals."""
import hashlib
import json

import pytest

from tools.diagnostics import luna_staged_metadata_root as helper
from tools.diagnostics.tests.test_luna_staged_replay_v1_root import execute_fixture


@pytest.mark.parametrize('mode',['negative','gold'])
def test_stage_metadata_counters_are_finite(monkeypatch,tmp_path,mode):
    root,entry_sha,terminal=execute_fixture(monkeypatch,tmp_path,mode=mode)
    result_sha=hashlib.sha256((root/'run/private-result.json').read_bytes()).hexdigest()
    out=helper.verify(root,'1'*64,entry_sha,result_sha)
    assert out['verified'] and out['new_model_calls']==0 and len(out['units'])==8
    assert sum(row['admitted_turns'] for row in out['units'])==terminal['paid_budget']['turns']
    assert sum(row['known_tokens'] for row in out['units'])==terminal['paid_budget']['known_tokens']
    assert not any(word in json.dumps(out) for word in ('PRIVATE','quote','request','response','content','checks'))
    if mode=='gold':
        assert out['units'][1]['stages'][1]['positive_alternatives']==[dict(index=0,predicate='prefers')]


def test_changed_result_does_not_export_fields(monkeypatch,tmp_path):
    root,entry_sha,_=execute_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='result_pin_invalid'):
        helper.verify(root,'1'*64,entry_sha,'0'*64)
