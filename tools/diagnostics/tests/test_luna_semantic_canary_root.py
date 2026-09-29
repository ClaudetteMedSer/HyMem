"""Independent finite-oracle and real-stack checks. All model output is invented."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import luna_semantic_canary as semantic
from benchmarks import luna_semantic_stage_accounting as stages

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = Path('/private/tmp/hymem-semantic-step2-v2-candidate-20260929')

SCRIPT = r'''
import sys, socket, json
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
def deny(*a, **k): raise AssertionError('offline only')
socket.socket.connect = deny
socket.create_connection = deny
from benchmarks import extraction_canary as gold, luna_semantic_canary as check
from benchmarks import luna_semantic_stage_accounting as stage
from hymem.extraction import chunk
scenario = sys.argv[3]
class Budget:
    def halt(self, code): raise AssertionError(code)
class Client:
    observed_turns = 0
    observed_tokens = 0
    usage_complete = True
    request_attempts = 0
    budget = Budget()
    def __init__(self): self.seen = {}
    def complete(self, request):
        self.observed_turns += 1
        self.observed_tokens += 7
        self.request_attempts += 1
        try: wire = json.loads(request.user)
        except ValueError: wire = None
        if type(wire) is dict and 'batch' in wire:
            item = wire['batch']['candidates'][0]
            mid = item['source_message_id']
            i = 0 if mid == gold._CANARY_EXPECTED_CLAIMS[0][6] else 1
            self.seen[mid] = self.seen.get(mid, 0) + 1
            predicate = item['predicate']
            status = 'supported'
            if predicate != gold._CANARY_EXPECTED_CLAIMS[i][2]:
                status, predicate = 'replace_predicate', gold._CANARY_EXPECTED_CLAIMS[i][2]
            if scenario == 'false_support' and i == 1:
                status, predicate = 'supported', item['predicate']
            if scenario == 'repeated_correction' and self.seen[mid] > 1:
                status, predicate = 'replace_predicate', 'avoids'
            if scenario == 'negative_second' and i == 1:
                status, predicate = 'unsupported', None
            if scenario == 'provider_failure' and i == 1:
                self.usage_complete = False
                raise RuntimeError('synthetic secret must not reach metadata')
            quote = gold._TABLE_CLAIM_ROW if i == 0 else gold._PROSE_BOUNDARY_RIGHT
            evidence = [] if predicate is None else [dict(source_message_id=mid,region='owned',quote=quote)]
            return json.dumps(dict(schema='source-grounding-v1',complete=True,
                batch_sha256=wire['batch_sha256'],verdicts=[dict(index=0,status=status,
                predicate=predicate,evidence=evidence)]))
        payloads, failures = gold._request_source_payloads(request)
        assert not failures
        triples = []
        if 'VERIFICATION PASS' not in request.system:
            for i, expected in enumerate(gold._CANARY_EXPECTED_CLAIMS):
                fragment = next((p for p in payloads if p['source_message_id'] == expected[6]), None)
                owned = gold._TABLE_CLAIM_ROW if i == 0 else gold._PROSE_BOUNDARY_RIGHT
                if not fragment or owned not in fragment['content']: continue
                predicate = expected[2]
                if (scenario in {'both', 'repeated_correction'} or
                    scenario in {'second', 'false_support'} and i == 1): predicate = 'uses'
                t = dict(subject=expected[0],predicate=predicate,object=expected[3],
                    polarity=expected[5],source_message_id=expected[6],
                    subject_type=expected[1],object_type=expected[4])
                if scenario == 'extra_qualifier': t['temporal_scope'] = 'invented'
                if scenario == 'wrong_type': t['subject_type'] = 'service'
                triples.append(t)
        markers = [dict(kind='preference',statement='invented marker')] if scenario == 'extra_marker' else []
        return json.dumps(dict(complete=True,triples=triples,markers=markers))
delegate = Client()
ledger = stage.StageLedger(Path(sys.argv[1]))
client = ledger.wrap(delegate, 'canary')
report = check.run_canary(gold, chunk, client, candidate=Path(sys.argv[1]))
budget = dict(turns=delegate.observed_turns, known_tokens=delegate.observed_tokens,
    questions={'canary':dict(turns=delegate.observed_turns,known_tokens=delegate.observed_tokens)})
assert ledger.reconcile(budget)
safe = dict(report=report,ledger=ledger.snapshot())
assert 'synthetic secret' not in json.dumps(safe)
print(json.dumps(safe))
'''


def replay(case):
    result = subprocess.run([sys.executable, '-I', '-B', '-c', SCRIPT,
                             str(CANDIDATE), str(ROOT), case],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize('case,rechecks', [('healthy', 0), ('second', 1), ('both', 2)])
def test_root_oracle_and_real_wrapped_stack(case, rechecks):
    data = replay(case)
    report, ledger = data['report'], data['ledger']['canary']
    assert report['passed']
    assert report['completion_calls'] == 10 + rechecks
    assert report['observed_token_delta'] == 7 * (10 + rechecks)
    assert ledger['grounding_initial']['returned_calls'] == 2
    assert ledger.get('grounding_recheck', {}).get('returned_calls', 0) == rechecks
    assert ledger['extraction_primary_or_other']['returned_calls'] == 4
    assert ledger['extraction_omission_verifier']['returned_calls'] == 2
    assert ledger['extraction_empty_verifier']['returned_calls'] == 2
    assert 'unclassified' not in ledger


@pytest.mark.parametrize('case', ['false_support', 'repeated_correction', 'negative_second',
    'provider_failure', 'extra_qualifier', 'extra_marker', 'wrong_type'])
def test_root_strict_oracle_still_rejects(case):
    assert replay(case)['report']['passed'] is False


@pytest.fixture(scope='module')
def healthy_report():
    return replay('healthy')['report']


@pytest.mark.parametrize('key,value', [
    ('initial_prepartition_leaves', 0), ('ordinary_calls', True),
    ('completion_calls', 11), ('provider_attempts', 0),
    ('grounding_provider_attempts', 1000), ('grounding_provider_attempts', 10),
    ('observed_token_delta', None),
    ('raw_exact_emissions', [True, 1]), ('corrected_claim_indexes', [{}]),
    ('source_sha256', {}), ('failure_code', 'private message'),
])
def test_root_report_tampering_fails_with_finite_value_error(healthy_report, key, value):
    report = copy.deepcopy(healthy_report)
    report[key] = value
    with pytest.raises(ValueError):
        semantic.validate_report(report, fixture_sha256=healthy_report['fixture_sha256'])


@pytest.mark.parametrize('key,value', [('status', []), ('claim_index', True),
    ('batch_sha256', 'X' * 64), ('candidate_predicate_expected', False)])
def test_root_trace_tampering_fails(healthy_report, key, value):
    report = copy.deepcopy(healthy_report)
    report['grounding_trace'][0][key] = value
    with pytest.raises(ValueError):
        semantic.validate_report(report, fixture_sha256=healthy_report['fixture_sha256'])


def test_classifier_never_inspects_frame_locals():
    class Frame:
        class f_code:
            co_filename = str(CANDIDATE / 'hymem/extraction/grounding_gate.py')
            co_name = 'ground_triples'
        f_lineno = 193
        f_back = None
        @property
        def f_locals(self): raise AssertionError('private locals read')
    assert stages.classify_stack(CANDIDATE, Frame()) == 'unclassified'


def test_real_contract_repair_callsite_is_classified():
    script = r'''
import sys, json, socket
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
def deny(*a, **k): raise AssertionError('offline only')
socket.socket.connect = deny
from hymem.extraction import chunk
from benchmarks import luna_semantic_stage_accounting as stages
owned = 'I use QuillDB.'
record = json.dumps(dict(source_message_id=11,source_role='user',content=owned,
    source_record_version='hymem-claim-source-v2'))
claim = dict(subject='Elena',predicate='uses',object='QuillDB',polarity=1,source_message_id=11)
class Client:
    observed_turns = observed_tokens = request_attempts = 0
    usage_complete = True
    def complete(self, request):
        self.observed_turns += 1
        self.observed_tokens += 1
        self.request_attempts += 1
        try: wire = json.loads(request.user)
        except ValueError: wire = None
        if type(wire) is dict and 'batch' in wire:
            return json.dumps(dict(schema='source-grounding-v1',complete=True,
                batch_sha256=wire['batch_sha256'],verdicts=[dict(index=0,status='supported',
                predicate='uses',evidence=[dict(source_message_id=11,region='owned',quote=owned)])]))
        if self.observed_turns == 1:
            return json.dumps(dict(complete=True,triples=[dict(claim,predicate='invented_predicate')],markers=[]))
        return json.dumps(dict(complete=True,triples=[] if 'OMISSION VERIFICATION PASS' in request.system else [claim],markers=[]))
ledger = stages.StageLedger(Path(sys.argv[1]))
result = chunk.extract_chunk(ledger.wrap(Client(), 'q-0000'), record,source_records=((11,record),))
assert not result.failed and result.completion_calls == 4, (result.failure_reason,result.failure_details,result.completion_calls)
snapshot = ledger.snapshot()['q-0000']
assert snapshot['extraction_contract_repair']['returned_calls'] == 1
assert snapshot['extraction_primary_or_other']['returned_calls'] == 1
assert snapshot['extraction_omission_verifier']['returned_calls'] == 1
assert snapshot['grounding_initial']['returned_calls'] == 1
assert 'unclassified' not in snapshot
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script,
                             str(CANDIDATE), str(ROOT)], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
