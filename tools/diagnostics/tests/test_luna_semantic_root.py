"""Independent publication and accounting controls; no model or network calls."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

from tools.diagnostics import luna_semantic_candidate as builder


REPO = Path(__file__).resolve().parents[3]
BASE = Path('/private/tmp/hymem-luna-grounding-v1.ZNG5XIVi')


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp('root-semantic')
    proof = builder.prepare(BASE / 'candidate', BASE / 'headless-grounding-source-map.json',
                            root / 'candidate', root / 'map.json', REPO)
    assert proof['files'] == 510
    return root / 'candidate'


SCRIPT = r'''
import sys, json, socket
sys.path.insert(0, sys.argv[1])
from hymem.extraction import chunk
def deny(*args, **kwargs): raise AssertionError('offline only')
socket.socket.connect = deny
socket.create_connection = deny
scenario = sys.argv[2]
source_text = 'These are invented publication controls, not semantic accuracy labels.'
record = json.dumps(dict(source_message_id=71, source_role='user', source_peer_id='Mira',
    source_created_at='2026-02-03T12:00:00Z', source_record_version='hymem-claim-source-v2',
    content=source_text))
claims = [dict(subject='person_' + str(i), predicate='prefers', object='database_' + str(i),
    polarity=1, source_message_id=71, value_text='five', value_numeric=5,
    value_unit='units', temporal_scope='2026', subject_type='person') for i in range(9)]
claims[0]['predicate'] = 'uses'
if scenario == 'collision_across_batches':
    claims[-1].update(subject=claims[0]['subject'], object=claims[0]['object'])
elif scenario == 'conflict_across_batches':
    claims[-1].update(subject=claims[0]['subject'], object=claims[0]['object'], polarity=-1)
marker = dict(kind='preference', statement='An independent invented marker.')

class Client:
    request_attempts = 0
    def __init__(self): self.ground = []; self.ordinary = 0
    def complete(self, request):
        try: wire = json.loads(request.user)
        except ValueError: wire = None
        if not isinstance(wire, dict) or 'batch' not in wire:
            self.request_attempts += 1
            self.ordinary += 1
            if self.ordinary == 1:
                return json.dumps(dict(complete=True, triples=claims, markers=[marker]))
            assert self.ordinary == 2, 'semantic rejection must not trigger extraction recovery'
            return json.dumps(dict(complete=True, triples=[], markers=[]))
        self.request_attempts += 3
        candidates = wire['batch']['candidates']
        self.ground.append(candidates)
        if scenario == 'provider_after_one_batch' and len(self.ground) == 2:
            raise RuntimeError('private-synthetic-provider-message')
        verdicts = []
        for i, item in enumerate(candidates):
            status, predicate = 'supported', item['predicate']
            if len(self.ground) <= 2 and item['subject'] == 'person_0' and predicate == 'uses':
                status, predicate = 'replace_predicate', 'prefers'
            if scenario == 'negative_late_recheck' and len(self.ground) == 4:
                status, predicate = 'unsupported', None
            if scenario == 'second_correction' and len(self.ground) == 3 and i == 0:
                status, predicate = 'replace_predicate', 'avoids'
            evidence = [] if status == 'unsupported' else [dict(source_message_id=71,
                region='owned', quote=source_text)]
            verdicts.append(dict(index=i, status=status, predicate=predicate, evidence=evidence))
        return json.dumps(dict(schema='source-grounding-v1', batch_sha256=wire['batch_sha256'],
                               complete=True, verdicts=verdicts))

client = Client()
limit = {'budget_second_batch': 3, 'budget_recheck': 4}.get(scenario)
result = chunk.extract_chunk(client, record, source_records=((71,record),), completion_call_limit=limit)
out = dict(failed=result.failed, reason=result.failure_reason, details=result.failure_details,
    calls=result.completion_calls, attempts=result.provider_attempts,
    grounding=result.grounding_calls, rechecks=result.grounding_recheck_calls,
    grounding_attempts=result.grounding_provider_attempts, batches=client.ground,
    triples=[t.__dict__ for t in result.triples], markers=[m.__dict__ for m in result.markers],
    types=result.entity_type_hints)
assert 'private-synthetic-provider-message' not in json.dumps(out)
if not result.failed:
    expected = [{k:v for k,v in item.items() if k != 'subject_type'} for item in claims]
    expected[0]['predicate'] = 'prefers'
    assert out['triples'] == expected
    assert out['markers'] == [marker]
    assert len(out['types']) == 9
print(json.dumps(out))
'''


def run(candidate, case):
    proc = subprocess.run([sys.executable, '-I', '-B', '-c', SCRIPT, str(candidate), case],
                          text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_whole_result_rechecked_with_unchanged_qualifiers_and_hints(candidate):
    out = run(candidate, 'positive')
    assert not out['failed']
    assert [len(b) for b in out['batches']] == [8, 1, 8, 1]
    assert out['calls'] == 6
    assert out['attempts'] == 14
    assert out['grounding'] == 4 and out['rechecks'] == 2
    assert out['grounding_attempts'] == 12
    assert out['batches'][1] == out['batches'][3]  # unchanged batch still rechecked


@pytest.mark.parametrize('case,calls,grounding,rechecks', [
    ('collision_across_batches', 4, 2, 0), ('conflict_across_batches', 4, 2, 0),
    ('negative_late_recheck', 6, 4, 2), ('second_correction', 5, 3, 1),
    ('provider_after_one_batch', 4, 2, 0), ('budget_second_batch', 3, 1, 0),
    ('budget_recheck', 4, 2, 0),
])
def test_whole_failure_has_no_partial_claim_marker_or_hint(candidate, case, calls, grounding, rechecks):
    out = run(candidate, case)
    assert out['failed']
    assert not out['triples'] and not out['markers'] and not out['types']
    assert out['calls'] == calls
    assert out['grounding'] == grounding and out['rechecks'] == rechecks
    assert out['grounding_attempts'] == grounding * 3
    assert out['attempts'] == 2 + grounding * 3


def test_derivation_delta_is_only_accepted_module_and_gate_integration(candidate):
    old = builder.inventory(BASE / 'candidate')
    new = builder.inventory(candidate)
    assert set(new) - set(old) == {builder.GROUNDING, builder.GATE}
    assert {key for key in old if old[key] != new[key]} == {builder.CHUNK, builder.CONTRACT}
    assert len(old) == 508 and len(new) == 510


def test_output_stamp_cannot_write_inside_original_source(tmp_path):
    # A lightweight stand-in catches validation before any source inventory read.
    source = tmp_path / 'original'
    source.mkdir()
    with pytest.raises(ValueError, match='output_inside_source'):
        builder.prepare(source, tmp_path / 'absent.json', tmp_path / 'candidate',
                        source / 'new-stamp.json', REPO)
    assert list(source.iterdir()) == []


CACHE_SCRIPT = r'''
import sys, socket
sys.path.insert(0, sys.argv[1])
from hymem.extraction import grounding as g, grounding_gate as gate, contract as c
def deny(*args, **kwargs): raise AssertionError('offline only')
socket.socket.connect = deny
before = c.extraction_cache_key()
case = sys.argv[2]
if case == 'prompt': g._SYSTEM += '\nIndependent mutation control.'
elif case == 'quote_bound': g.MAX_QUOTE_CHARS -= 1
elif case == 'nested_validator_code':
    g._validate.__code__ = (lambda a, b: (a, b)).__code__
elif case == 'mapper_code':
    gate._source.__code__ = (lambda a, b: None).__code__
elif case == 'correction_identity_code':
    gate._canonical_key.__code__ = (lambda a: None).__code__
elif case == 'gate_import':
    gate.build_grounding_request = lambda a, b: None
elif case == 'definition_then_old_consumed_code':
    consumed = gate.parse_grounding_response
    g.parse_grounding_response = lambda a, b: None
    consumed.__code__ = (lambda a, b: None).__code__
try:
    after = c.extraction_cache_key()
except (RuntimeError, ValueError):
    print('closed')
else:
    assert after != before, 'changed execution silently reused prior cache'
    print('changed')
'''


@pytest.mark.parametrize('case', ['prompt', 'quote_bound', 'nested_validator_code', 'mapper_code',
    'correction_identity_code', 'gate_import', 'definition_then_old_consumed_code'])
def test_cache_tracks_active_grounding_execution(candidate, case):
    proc = subprocess.run([sys.executable, '-I', '-B', '-c', CACHE_SCRIPT, str(candidate), case],
                          text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() in {'closed', 'changed'}


PUBLICATION_SCRIPT = r'''
import sys, json, socket
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, sys.argv[1])
from hymem import HyMem, HyMemConfig
from hymem.core import db
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.triples import Triple
from hymem.extraction.grounding_gate import _source
from hymem.extraction.grounding import build_grounding_request
def deny(*a, **k): raise AssertionError('no network allowed')
socket.socket.connect = deny
scenario = sys.argv[3]
cfg = HyMemConfig(root=Path(sys.argv[2]))
hy = HyMem(cfg, llm=StubLLMClient(default='[]'))
try:
    hy.conn.execute("INSERT INTO sessions(id) VALUES ('semantic-root')")
    cursor = hy.conn.execute("INSERT INTO messages(session_id,role,content) VALUES (?,?,?)",
        ('semantic-root', 'user', 'I prefer CairnDB.'))
    mid = int(cursor.lastrowid)
    chunk = Chunk(id='semantic-root-chunk', session_id='semantic-root', start_message_id=mid,
        end_message_id=mid, salience_reason='long_user_turn', text='user: I prefer CairnDB.',
        source_message_ids=(mid,))
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, chunk.session_id)
        persist_chunks(hy.conn, [chunk])
    original = Triple('user', 'uses', 'CairnDB', 1, source_message_id=mid)
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    records = tuple((s.message_id, phase1._claim_source_record(s)) for s in sources)
    owned = _source(records[0], ())
    def review(triple, status, predicate):
        _, batch = build_grounding_request((triple,), (owned,))
        witnesses = [] if status == 'unsupported' else [dict(source_message_id=mid,
            region='owned', quote='I prefer CairnDB.')]
        return json.dumps(dict(schema='source-grounding-v1', batch_sha256=batch.batch_sha256,
            complete=True, verdicts=[dict(index=0, status=status, predicate=predicate,
                evidence=witnesses)]))
    first = review(original, 'unsupported', None) if scenario == 'reject' else review(
        original, 'replace_predicate', 'prefers')
    corrected = replace(original, predicate='prefers')
    primary = dict(subject='user', predicate='uses', object='CairnDB', polarity=1,
                   source_message_id=mid)
    client = StubLLMClient(fixtures={
        'OMISSION VERIFICATION PASS': json.dumps(dict(complete=True, triples=[], markers=[])),
        '"predicate":"uses"': first,
        '"predicate":"prefers"': review(corrected, 'supported', 'prefers'),
    }, default=json.dumps(dict(complete=True, triples=[primary], markers=[])))
    result = phase1.extract_chunk_results(hy.conn, chunk, client, prompt_version=cfg.prompt_version)
    assert result is not None
    assert result.failed == (scenario == 'reject')
    with db.transaction(hy.conn):
        phase1.persist_chunk_results(hy.conn, chunk, result, prompt_version=cfg.prompt_version, cfg=cfg)
    rows = [r[0] for r in hy.conn.execute('SELECT predicate FROM knowledge_graph')]
    processed = hy.conn.execute('SELECT count(*) FROM processed_chunks').fetchone()[0]
    observations = hy.conn.execute('SELECT count(*) FROM kg_claim_observations').fetchone()[0]
    if scenario == 'reject':
        assert rows == [] and processed == 0 and observations == 0
        row = hy.conn.execute('SELECT last_failure_reason, attempts FROM chunk_extraction_attempts').fetchone()
        assert tuple(row) == ('grounding_failure', 1)
    else:
        assert rows == ['prefers'] and processed == 1 and observations == 1
    print(json.dumps(dict(failed=result.failed, graph_rows=len(rows), processed=processed,
                         observations=observations, calls=result.completion_calls)))
finally:
    hy.close()
'''


@pytest.mark.parametrize('scenario', ['reject', 'correct'])
def test_real_phase1_publication_respects_grounding(candidate, tmp_path, scenario):
    proc = subprocess.run([sys.executable, '-I', '-B', '-c', PUBLICATION_SCRIPT,
                           str(candidate), str(tmp_path / 'store'), scenario],
                          text=True, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out['calls'] == (3 if scenario == 'reject' else 4)
