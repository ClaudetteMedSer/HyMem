"""Root-authored diagnostic scoring/privacy controls, without model calls."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hymem.extraction import grounding
from hymem.extraction.llm import LLMRequest
from benchmarks import codex_subscription_concurrent_v2 as concurrent
from benchmarks import luna_semantic_stage_accounting as stage
from tools.diagnostics import luna_semantic_cases as fixtures
from tools.diagnostics import luna_semantic_probe as probe


class Judge:
    def __init__(self, case, behavior='expected'):
        self.case, self.behavior, self.requests = case, behavior, []

    def complete(self, request):
        self.requests.append(request)
        wire = json.loads(request.user)
        verdicts = []
        repeat = len(self.requests) > 1
        for i, (candidate, label) in enumerate(zip(wire['batch']['candidates'], self.case.expected)):
            status = 'supported' if repeat else sorted(label.statuses)[0]
            predicate = (label.predicate if status == 'replace_predicate' else
                         candidate['predicate'] if status == 'supported' else None)
            evidence = [dict(source_message_id=candidate['source_message_id'], region=region, quote=quote)
                        for region, quote in label.evidence] if predicate is not None else []
            if self.behavior == 'false_support':
                status, predicate = 'supported', candidate['predicate']
                evidence = [dict(source_message_id=candidate['source_message_id'], region='owned',
                                 quote=self.case.sources[0].content[:192])]
            elif self.behavior == 'false_rejection' or self.behavior == 'negative_recheck' and repeat:
                status, predicate, evidence = 'uncertain', None, []
            elif self.behavior == 'wrong_replacement':
                status, predicate = 'replace_predicate', 'avoids'
            elif self.behavior == 'repeated_correction' and repeat:
                status, predicate = 'replace_predicate', 'avoids'
            verdicts.append(dict(index=i,status=status,predicate=predicate,evidence=evidence))
        if self.behavior == 'private_malformed':
            return 'PRIVATE_PROVIDER_CANARY_DO_NOT_EXPORT'
        return json.dumps(dict(schema='source-grounding-v1',batch_sha256=wire['batch_sha256'],
                               complete=True,verdicts=verdicts))


@pytest.mark.parametrize('case', fixtures.cases(), ids=lambda c:c.case_id)
def test_root_fixed_case_scoring_and_exact_request_no_labels(case):
    judge = Judge(case)
    events = []
    result = probe.run_control(case,judge,grounding,record=events.append)
    assert result['passed']
    expected_calls = 2 if case.category == 'correction' else 1
    assert len(judge.requests) == expected_calls
    original, _ = grounding.build_grounding_request(case.triples,case.sources)
    assert asdict(judge.requests[0]) == asdict(original)
    # Labels, rationales and exemplar evidence must not become an extra prompt.
    for label in case.expected:
        assert label.rationale not in judge.requests[0].system + judge.requests[0].user
    assert len(events) == 2 * expected_calls
    if expected_calls == 2:
        before = json.loads(judge.requests[0].user)['batch']
        after = json.loads(judge.requests[1].user)['batch']
        assert before['sources'] == after['sources']
        for old,new,label in zip(before['candidates'],after['candidates'],case.expected):
            assert new['predicate'] == label.predicate
            assert {k:v for k,v in old.items() if k != 'predicate'} == {
                k:v for k,v in new.items() if k != 'predicate'}


@pytest.mark.parametrize('category,behavior,field,count', [
    ('reject','false_support','false_support_indexes',1),
    ('supported','false_rejection','false_rejection_indexes',1),
    ('correction','wrong_replacement','missed_recovery_indexes',1),
])
def test_root_wrong_initial_verdict_fails_without_another_model_call(category,behavior,field,count):
    case = next(c for c in fixtures.cases() if c.category == category)
    judge = Judge(case,behavior)
    result = probe.run_control(case,judge,grounding,record=lambda x:None)
    assert not result['passed'] and len(result[field]) == count
    assert len(judge.requests) == 1


@pytest.mark.parametrize('behavior', ['negative_recheck','repeated_correction'])
def test_root_recheck_cannot_trigger_a_third_call(behavior):
    case = next(c for c in fixtures.cases() if c.category == 'correction')
    judge = Judge(case,behavior)
    result = probe.run_control(case,judge,grounding,record=lambda x:None)
    assert not result['passed'] and len(judge.requests) == 2


def test_root_malformed_provider_text_is_not_exported():
    case = fixtures.cases()[0]
    judge = Judge(case,'private_malformed')
    result = probe.run_control(case,judge,grounding,record=lambda x:None)
    assert not result['passed'] and result['malformed_code']
    assert 'PRIVATE_PROVIDER_CANARY_DO_NOT_EXPORT' not in json.dumps(result)


def test_root_predispatch_evidence_failure_makes_zero_model_calls():
    case = fixtures.cases()[0]
    judge = Judge(case)
    def fail(value): raise OSError('private disk location must not be exported')
    with pytest.raises(BaseException):
        probe.run_control(case,judge,grounding,record=fail)
    assert not judge.requests


def test_root_retained_evidence_hash_is_mandatory():
    with pytest.raises(probe.ProbeStop):
        probe.verify_retained_evidence(b'{"requests":[],"responses":[]}')


def test_root_actual_hybrid_uses_only_three_new_judgments_and_zero_replay_tokens():
    from tools.diagnostics.tests.test_luna_semantic_canary_root import SCRIPT, CANDIDATE, ROOT
    # Reuse this root's independent synthetic canary client, not Sol's fixture
    # or the real private retained outputs. Build the eight ordinary records
    # offline, then verify a new hybrid only sends its three judgments onward.
    script = SCRIPT.split('delegate = Client()')[0] + r'''
from dataclasses import asdict
from tools.diagnostics import luna_semantic_probe as probe
retained = []
first = Client()
original = first.complete
def capture(request):
    raw = original(request)
    try: wire = json.loads(request.user)
    except ValueError: wire = None
    if type(wire) is not dict or 'batch' not in wire:
        retained.append((asdict(request),raw))
    return raw
first.complete = capture
initial = check.run_canary(gold,chunk,first,candidate=Path(sys.argv[1]))
assert initial['passed'] and first.observed_turns == 11 and len(retained) == 8
paid = Client()
events = []
out = probe.run_hybrid(canary_module=gold,chunk_module=chunk,semantic_module=check,
    candidate=Path(sys.argv[1]),paid=paid,retained=tuple(retained),record=events.append,
    stage_module=stage)
assert out['passed'] and paid.observed_turns == 3 and paid.observed_tokens == 21
assert out['replayed_ordinary_calls'] == 8
assert out['canary']['observed_turn_delta'] == 11
assert out['canary']['observed_token_delta'] == 21
assert out['logical_stage_reconciled']
bad = [(dict(retained[0][0],user='wrong retained request'),retained[0][1]),*retained[1:]]
blocked = Client()
try:
    probe.run_hybrid(canary_module=gold,chunk_module=chunk,semantic_module=check,
        candidate=Path(sys.argv[1]),paid=blocked,retained=tuple(bad),record=lambda x:None,
        stage_module=stage)
except BaseException:
    pass
else:
    raise AssertionError('mismatched replay accepted')
assert blocked.observed_turns == 0
'''
    result = subprocess.run([sys.executable,'-I','-B','-c',script,
        str(CANDIDATE),str(ROOT),'second'],capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stderr


@pytest.mark.parametrize('fault', ['provider', 'unknown_usage', 'write_before', 'write_after', 'cleanup'])
def test_root_campaign_failures_keep_paid_accounting_and_cleanup(fault):
    closed,clients = [],[]
    class Journal:
        def record(self,key,value):
            if (fault == 'write_before' and value['phase'] == 'initial_before_dispatch' or
                fault == 'write_after' and value['phase'] == 'initial_returned'):
                raise probe.ProbeStop('private_evidence_write_failure')
    class Client:
        def __init__(self,key,cap,budget):
            self.key,self.budget = key,budget
            budget.register(key,concurrent.BudgetLimits(*cap))
            self.judge = Judge(fixtures.cases()[int(key.split('-')[-1])])
        @property
        def observed_turns(self): return self.budget.snapshot()['questions'][self.key]['turns']
        @property
        def observed_tokens(self): return self.budget.snapshot()['questions'][self.key]['known_tokens']
        @property
        def usage_complete(self): return self.budget.snapshot()['questions'][self.key]['usage_complete']
        def complete(self,request):
            self.budget.reserve(self.key)
            self.budget.before_turn(self.key,dict(auth='chatgpt',model=concurrent.base.MODEL,
                config_isolation_admitted=True,inference_enabled=False,
                quota_windows=[dict(remaining_percent=90)]))
            used = None if fault in {'provider','unknown_usage'} else 9
            self.budget.settle(self.key,used=used,turn_started=True)
            if fault == 'provider': raise RuntimeError('PRIVATE_PROVIDER_MESSAGE')
            return self.judge.complete(request)
        def close(self):
            closed.append(self.key)
            if fault == 'cleanup': raise RuntimeError('PRIVATE_CLEANUP_PATH')
    def factory(key,cap,budget):
        client = Client(key,cap,budget)
        clients.append(client)
        return client
    out = probe.run_campaign(concurrent=concurrent,warm=SimpleNamespace(),binary='unused',
        cases_module=fixtures,grounding_module=grounding,journal=Journal(),retained=tuple([({},'')]*8),
        canary_module=None,chunk_module=None,semantic_module=None,stage_module=None,
        candidate=Path('/unused'),client_factory=factory)
    assert len(clients) == 1 and closed == ['control-00']
    assert out['paid_budget']['turns'] == (0 if fault == 'write_before' else 1)
    assert out['paid_budget']['known_tokens'] == (9 if fault in {'write_after','cleanup'} else 0)
    assert out['paid_budget']['usage_complete'] is (fault not in {'provider','unknown_usage'})
    assert out['stop_code'] and not out['core_completed']
    assert not out['completed_and_clean'] and not out['process_cleanup_verified']
    assert 'PRIVATE_' not in json.dumps(out)


def test_root_hybrid_stage_counter_failure_halts_owned_budget():
    from tools.diagnostics.tests.test_luna_semantic_canary_root import CANDIDATE
    request = LLMRequest(system='invented',user='invented')
    halted = []
    paid = SimpleNamespace(observed_turns=0,observed_tokens=0,usage_complete=True,
                           budget=SimpleNamespace(halt=halted.append))
    retained = tuple([(asdict(request),'{}')]*8)
    hybrid = probe.HybridReplayClient(paid,retained,record=lambda v:None,source_ids=(1,2))
    ledger = stage.StageLedger(CANDIDATE)
    def fail(*args,**kwargs): raise RuntimeError('PRIVATE_COUNTER_EXCEPTION')
    ledger.record = fail
    with pytest.raises(stage.StageAccountingStop):
        ledger.wrap(hybrid,'canary').complete(request)
    assert halted == ['stage_accounting_failure']
