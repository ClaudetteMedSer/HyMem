"""Parent-owned replay-proof controls; no database, provider or credentials."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    'r6_chunk_replay_control', Path(__file__).resolve().parents[1]/'lme_r6_chunk_replay.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def fixture_api(*, spans=((0, 4), (4, 8)), context=None, changed_scope=False,
                changed_text=False, failure=False, depth=2, size=40):
    original = {'source_message_id': 7, 'content': 'abcdefgh', 'source_role': 'user',
                'source_record_version': 'hymem-claim-source-v2', 'source_session_id': 's'}
    records = ((7, json.dumps(original)),)
    leaves = []
    for start, end in spans:
        payload = {**original, 'content': original['content'][start:end],
                   'source_content_start': start, 'source_content_end': end,
                   'source_record_version': 'fragment'}
        if context is not None:
            payload['source_fragment_context'] = context
        if changed_scope:
            payload['source_session_id'] = 'other'
        if changed_text:
            payload['content'] = 'invented'
        leaves.append((SimpleNamespace(text='x'*size, source_records=((7, json.dumps(payload)),)), depth))
    api = SimpleNamespace(_ExtractionUnit=lambda **kw: SimpleNamespace(**kw),
                          _prepartition=lambda _u: (None, object()) if failure else (leaves, None),
                          _source_payload=lambda record: json.loads(record[1]),
                          _MAX_SPLIT_DEPTH=8, _MAX_UNSPLITTABLE_INPUT_CHARS=8000,
                          _MAX_PREPARTITION_LEAVES=32)
    return records, api


def test_exact_contiguous_partition_passes():
    records, api = fixture_api()
    result = replay.partition_proof(records, api)
    assert [r['parts'][0]['start'] for r in result] == [0, 4]
    assert [r['parts'][0]['end'] for r in result] == [4, 8]


@pytest.mark.parametrize('kwargs', [
    {'spans': ((0, 4), (5, 8))}, {'spans': ((0, 5), (4, 8))},
    {'spans': ((0, 4),)}, {'spans': ((4, 8), (0, 4))},
    {'changed_scope': True}, {'changed_text': True}, {'failure': True},
    {'depth': 9}, {'size': 8001},
    {'context': {'content': 'fake', 'source_content_start': 0, 'source_content_end': 4}},
    {'context': {'content': 'abcd', 'source_content_start': 0, 'source_content_end': 4,
                 'prelude_content': 'fake', 'prelude_source_content_start': 0,
                 'prelude_source_content_end': 4}},
])
def test_unsafe_partition_never_claims_proof(kwargs):
    records, api = fixture_api(**kwargs)
    with pytest.raises(RuntimeError):
        replay.partition_proof(records, api)
