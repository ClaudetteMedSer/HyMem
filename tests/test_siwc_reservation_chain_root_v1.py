"""Root-independent source-chain and assembled deadline controls, no inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import chatgpt_plan_lme_v6 as bridge
from tools.diagnostics import siwc_lme_diagnostic_v8 as old_runner
from tools.diagnostics import siwc_lme_diagnostic_v9 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v11 as reader
from tools.diagnostics import siwc_lme_diagnostic_bundle_v8 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v8 as host
from tools.diagnostics import siwc_lme_diagnostic_launch_v8 as launch

REPO = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle')


def helper():
    spec = importlib.util.spec_from_file_location('root_reservation_fixture', REPO / 'tests/test_siwc_indexing6h_root_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.runner, module.reader, module.bridge = runner, reader, bridge
    module.bundle, module.host, module.launch = bundle, host, launch
    return module


def test_runner_only_changes_identity_and_bridge_binding():
    text = Path(runner.__file__).read_text()
    for new, old in [('diagnostic-v9', 'diagnostic-v8'), ('diagnostic_v9.py', 'diagnostic_v8.py'),
        ('chatgpt_plan_lme_v6', 'chatgpt_plan_lme_v5'),
        ('be773c6d9891e89331a9ba814570a02f456d267b1946c7d141a273b935680af8',
         '34b51c18a28bd6a09b64a1592cb13edae50716f7694f5a271ec54a27a65010e4')]:
        text = text.replace(new, old)
    assert text == Path(old_runner.__file__).read_text()
    assert runner.MAX_LIMITS == old_runner.MAX_LIMITS


@pytest.mark.parametrize('failure', [False, True])
def test_actual_four_worker_campaign_and_strict_reader(tmp_path, monkeypatch, failure):
    helper().test_actual_four_worker_campaign_uses_new_wall_policy_and_reconciles(tmp_path, monkeypatch, failure)


@pytest.mark.parametrize('name', ['test_recursive_prior_unit_cleanup_includes_nested_threads',
                                'test_ambiguous_dispatch_consumes_one_shot'])
def test_recursive_cleanup_and_consumed_dispatch(tmp_path, monkeypatch, name):
    helper().test_recursive_cleanup_and_no_repeat(tmp_path, monkeypatch, name)


def test_actual_assembly_isolation_and_deadline_in_selected_source(tmp_path):
    root = tmp_path / '.hymem-siwc-lme-diagnostic-reservationroot'
    value = bundle.assemble(repo=REPO, accepted_code=FROZEN / 'code',
        candidate=FROZEN / 'candidate', map_path=FROZEN / 'source-map.json', output=root)
    assert value['candidate_files'] == 514 and value['code_files'] == 26
    assert value['model_calls'] == 0 and not value['credential_present']
    assert len(host.source_manifest(root)) == 541
    assert hashlib.sha256((root / 'code' / runner.RUNNER_RELATIVE).read_bytes()).hexdigest() == reader.RUNNER_SHA256
    code = '''
import sys, pathlib, importlib.util, json, time
from types import SimpleNamespace
def deny(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
        raise AssertionError('external forbidden')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if path.endswith('/auth.json') or '/.codex/' in path or '/.hymem-chatgpt-plan-lme/' in path:
            raise AssertionError('auth forbidden')
sys.addaudithook(deny)
root = pathlib.Path(sys.argv[1])
path = root / 'code/tools/diagnostics/siwc_lme_diagnostic_v9.py'
spec = importlib.util.spec_from_file_location('isolated_reservation_root', path)
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
loaded = runner.import_source_only(root, root / 'source-map.json', runner.ACCEPTED_INVENTORY_SHA256)
assert loaded['source_only'] and 'broker' not in loaded
bridge = loaded['siwc']
assert pathlib.Path(bridge.__file__).name == 'chatgpt_plan_lme_v6.py'
assert pathlib.Path(bridge.transport.__file__).name == 'chatgpt_plan_responses_v10.py'
broker = object.__new__(bridge.owner.CredentialBroker); broker.identity_digest = 'a' * 64
bridge.owner.CredentialBroker.acquire = lambda self, **kw: bridge.owner.CredentialLease('invented-not-a-token',int(time.time())+900)
seen=[]
def response(*a, timeout):
    seen.append(timeout); return bridge.transport.Completed('invented',2,1,3,0,0)
limits = bridge.warm.BudgetLimits(*runner.MAX_LIMITS['question'])
budget = bridge.SharedBudget(bridge.warm.BudgetLimits(*runner.MAX_LIMITS['campaign']))
clients=[bridge.SIWCLMEClient(broker,budget,'q',limits,response_call=response),
    bridge.SIWCLMEClient(broker,runner.RegistrationAlias(budget,'q',limits),'q',limits,response_call=response)]
for c in clients:
    assert c.complete(bridge.LLMRequest('invented s','invented u'))=='invented'
assert len(seen)==2 and all(299 < x <= 300 for x in seen)
assert budget.snapshot()['turns']==2 and budget.snapshot()['known_tokens']==6
assert budget.snapshot()['usage_complete'] and budget.snapshot()['in_flight']==0
questions=[{'question_id':f'invented-{i}'} for i in range(4)]
dataset=root/'absent-invented-dataset.json'; original=runner._sha
runner._sha=lambda p:runner.DATASET_SHA256 if p==dataset else original(p)
loaded.update(source_only=False,dataset=dataset,questions=questions,
    prior=SimpleNamespace(SelectedQuestions=lambda *_: questions))
receipt=runner.receipt_for(root,loaded)
assert receipt['invocation_seconds']==300 and receipt['indexing_seconds']==21600
assert receipt['limits']['campaign']==[8012,48160000,25200]
assert receipt['limits']['question']==[2000,12000000,23400]
assert receipt['limits']['canary']==[12,160000,600]
assert len(receipt['selected_row_sha256'])==4
assert 4000 < len(runner._canonical(receipt)) <= 8192
print(json.dumps({'isolated_import':True,'real_forwarding_verified':True,'model_calls':0}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(root)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'isolated_import': True, 'real_forwarding_verified': True, 'model_calls': 0}
