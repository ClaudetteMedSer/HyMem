"""Independent root checks of repaired four-question pilot; no inference."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from tools.diagnostics import luna_lme_diagnostic_v9 as previous
from tools.diagnostics import luna_lme_diagnostic_v10 as runner
from benchmarks import codex_subscription_warm_v9 as warm


def test_candidate_and_measurement_functions_are_frozen():
    for name in ('PINS', 'CANDIDATE_PINS', 'ACCEPTED_MAP_SHA256',
                 'ACCEPTED_INVENTORY_SHA256', 'MAX_LIMITS', 'BILLING_POLICY',
                 'NOTIFICATION_POLICY', 'DATASET_SHA256', 'DIAGNOSTIC_HELPER_SHA256'):
        assert getattr(runner, name) == getattr(previous, name)
    for name in ('_load_verified', 'import_source_only', 'make_dual',
                 'run_live_canary', 'AccountedClient', '_memory_client',
                 'validate_diagnostic_row', '_question_worker', 'run_campaign',
                 'ObservationRegistry', '_resource_sample'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(runner, name)))) == ast.dump(
            ast.parse(inspect.getsource(getattr(previous, name)))), name
    assert runner.MAX_LIMITS == {'campaign': (8012, 48160000, 14400),
        'canary': (12, 160000, 600), 'question': (2000, 12000000, 12600)}


def test_source_only_cannot_run(tmp_path):
    with pytest.raises(ValueError, match='runnable_source_binding_required'):
        runner.run_campaign({'source_only': True}, output=tmp_path/'never',
            campaign_limits=None, canary_limits=None, question_limits=None,
            indexing_seconds=10800, workers=4, helper_sha256='', containment=lambda _: True)
    assert not (tmp_path/'never').exists()


@pytest.mark.parametrize('reject', ['receipt','containment','none'])
def test_cli_gates_and_durable_consumption_before_campaign(tmp_path,monkeypatch,reject):
    root=tmp_path/'.hymem-lme-diagnostic-invented'
    root.mkdir()
    calls=[]
    loaded={'root':root,'questions':[{} for _ in range(4)],'warm':warm}
    monkeypatch.setattr(runner,'load_verified',lambda *args:loaded)
    def receipt(*args):
        calls.append('receipt')
        if reject=='receipt': raise ValueError('PRIVATE')
        return {'expected_cgroup':'invented'}
    def containment(*args):
        calls.append('containment')
        if reject=='containment': raise ValueError('PRIVATE')
        return True
    def campaign(*args,**kwargs):
        calls.append('campaign')
        marker=root/runner.EXECUTION_MARKER
        assert marker.is_file() and json.loads(marker.read_text())=={
            'receipt_sha256':'a'*64,'execution_started':True}
        assert kwargs['workers']==4 and kwargs['indexing_seconds']==10800
        return {'run_id':'invented','selected_denominator':4,'scored_count':4,'campaign_stop':None}
    monkeypatch.setattr(runner,'verify_launch_receipt',receipt)
    monkeypatch.setattr(runner,'verify_live_containment',containment)
    monkeypatch.setattr(runner,'run_campaign',campaign)
    args=['--root',str(root),'--inventory',str(root/'source-map.json'),
        '--inventory-sha256',runner.ACCEPTED_INVENTORY_SHA256,
        '--dataset','/unused','--binary','/unused','--binary-sha256',runner.BINARY_SHA256,
        '--receipt-sha256','a'*64,'--output-dir',str(root/'run'),'--run']
    result=runner.main(args)
    if reject=='none':
        assert result==0 and calls==['receipt','containment','campaign']
        assert runner.main(args)==1
        assert calls.count('campaign')==1
    else:
        assert result==1 and 'campaign' not in calls
        assert not (root/runner.EXECUTION_MARKER).exists()


def test_actual_four_worker_coordinator_preserves_denominator_and_quality(tmp_path, monkeypatch):
    """A barrier proves overlap; invented workers settle real shared budgets."""
    barrier = threading.Barrier(4)
    captured = {}
    thread_ids = set()
    lock = threading.Lock()
    class Client:
        def diagnostic_summary(self):
            return {'calls': 1, 'successes': 1, 'failures': 0,
                    'counts_saturated': False, 'timing_saturated': False}
    class Checkpoint:
        def __init__(self, path, *, manifest, expected_ids, scored, resume, retry_failures):
            assert len(expected_ids) == 4 and scored and not resume and not retry_failures
            self.ids = expected_ids
            self.rows = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def record(self, qid, *, row, failure=None):
            assert qid in self.ids and qid not in self.rows
            self.rows[qid] = row
        def finalize(self):
            assert set(self.rows) == set(self.ids)
            return {'counts': {'completed': sum(x is not None for x in self.rows.values()),
                'failed': sum(x is None for x in self.rows.values()), 'missing': 0}}
    dataset = tmp_path/'invented.json'
    dataset.write_text('[]')
    original_sha = runner._sha
    monkeypatch.setattr(runner, '_sha', lambda p: runner.DATASET_SHA256 if p == dataset else original_sha(p))
    def hash_value(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    loaded = {'source_only': False, 'dataset': dataset,
        'questions': [{'question_id': f'invented-{i}'} for i in range(4)],
        'warm': warm, 'observer': SimpleNamespace(TimeoutSubscriptionClient=Client,
            _project_summary=lambda value: value),
        'strictness': SimpleNamespace(AtomicCheckpoint=Checkpoint, content_hash=hash_value),
        'diagnostic': SimpleNamespace(MODE='semantic_diagnostic_v1'),
        'prior': SimpleNamespace(atomic_private=lambda p,v: captured.__setitem__(p.name,v))}
    admission = {'auth':'chatgpt','model':'gpt-6-luna','config_isolation_admitted':True,
        'inference_enabled':False,'quota_windows':[{'remaining_percent':100}]}
    def settle_pair(budget, limits, registry, key, prefix):
        budget.register(key, limits)
        for route in ('ordinary','structured'):
            budget.reserve(key)
            budget.before_turn(key, admission)
            budget.settle(key, used=7, turn_started=True)
            slot=prefix+'.'+route
            registry.register(slot,Client())
            registry.capture(slot)
    def canary(_loaded,budget,limits,_path,registry):
        settle_pair(budget,limits,registry,'canary','canary')
        return {'structural_valid':True,'model_gold_match':False,
            'quality_failure_reason':None,'semantic_failure_proved':True,'completion_calls':2}
    def worker(_loaded,budget,limits,question,index,output,indexing,registry):
        with lock: thread_ids.add(threading.get_ident())
        barrier.wait(timeout=5)
        settle_pair(budget,limits,registry,f'q-{index:04d}',f'question.{index}')
        return {'projection': {'correct': index%2==0, 'strict_indexing_healthy':index%2==1},
                'accounting': {}, 'stop_code':None}
    monkeypatch.setattr(runner,'run_live_canary',canary)
    monkeypatch.setattr(runner,'_question_worker',worker)
    result=runner.run_campaign(loaded,output=tmp_path/'output',
        campaign_limits=warm.BudgetLimits(*runner.MAX_LIMITS['campaign']),
        canary_limits=warm.BudgetLimits(*runner.MAX_LIMITS['canary']),
        question_limits=warm.BudgetLimits(*runner.MAX_LIMITS['question']),
        indexing_seconds=10800,workers=4,helper_sha256=runner.DIAGNOSTIC_HELPER_SHA256,
        containment=lambda _:True)
    assert len(thread_ids)==4
    assert result['selected_denominator']==result['scored_count']==4
    assert result['correct_count']==2 and result['strict_unhealthy_count']==2
    assert result['quality_accuracy_full_selected']==0.5
    assert result['campaign_stop'] is None
    assert result['budget']['turns']==10 and result['budget']['known_tokens']==70
    assert result['budget']['in_flight']==result['budget']['reserved']==0
    assert len(result['timeout_observations'])==10
