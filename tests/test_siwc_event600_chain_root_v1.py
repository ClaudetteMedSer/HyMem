"""Root source-only assembled owner-to-bridge deadline verification."""
import importlib.util
import ast
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import chatgpt_plan_lme_v8 as bridge
from tools.diagnostics import siwc_lme_diagnostic_v11 as runner
from tools.diagnostics import siwc_lme_diagnostic_progress_v13 as reader
from tools.diagnostics import siwc_lme_diagnostic_bundle_v10 as bundle
from tools.diagnostics import siwc_lme_diagnostic_host_preflight_v10 as host
from tools.diagnostics import siwc_lme_diagnostic_launch_v10 as launch
from tests.test_lme_chatgpt_plan_owner_v1 import fixture_vm_state

REPO = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle')


def test_remote_decoder_embedded_pin_matches_real_541_file_archive(tmp_path, monkeypatch):
    root = tmp_path / 'decoder'
    bundle.assemble(repo=REPO, accepted_code=FROZEN / 'code',
        candidate=FROZEN / 'candidate', map_path=FROZEN / 'source-map.json', output=root)
    archive = host.archive_bytes(root)
    statements = []
    for node in ast.parse(host.REMOTE).body:
        text = ast.get_source_segment(host.REMOTE, node) or ''
        if text.startswith('need(os.getuid()'):
            continue
        if text.startswith('need(regular(DATASET)'):
            break
        statements.append(node)
    else:
        raise AssertionError('remote_decoder_boundary_missing')
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(archive)))
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                 '<root-offline-decoder600>', 'exec'), scope)
    assert scope['RUNNER_SHA'] == host.RUNNER_SHA == reader.RUNNER_SHA256
    assert len(scope['content']) == len(scope['manifest']) == 541
    assert reader.LIMITS == {key:list(value) for key,value in runner.MAX_LIMITS.items()}
    assert bridge.MAX_INVOCATION == bridge.transport.MAX_WALL_SECONDS == 600
    assert runner.MAX_LIMITS == {'campaign':(8012,48160000,25200),
        'question':(2000,12000000,23400),'canary':(12,160000,600)}


def fixture():
    spec = importlib.util.spec_from_file_location('root_ownerhorizon_fixture', REPO / 'tests/test_siwc_indexing6h_root_v1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.runner, module.reader, module.bridge = runner, reader, bridge
    module.bundle, module.host, module.launch = bundle, host, launch
    return module


@pytest.mark.parametrize('failed', [False, True])
def test_actual_four_worker_campaign_and_reader(tmp_path, monkeypatch, failed):
    fixture().test_actual_four_worker_campaign_uses_new_wall_policy_and_reconciles(tmp_path, monkeypatch, failed)


@pytest.mark.parametrize('name', ['test_recursive_prior_unit_cleanup_includes_nested_threads',
    'test_ambiguous_dispatch_consumes_one_shot'])
def test_recursive_cleanup_and_no_repeat(tmp_path, monkeypatch, name):
    fixture().test_recursive_cleanup_and_no_repeat(tmp_path, monkeypatch, name)


def test_frozen_source_actual_owner_admission_no_acquire_stub(tmp_path):
    root = tmp_path / '.hymem-siwc-lme-diagnostic-ownerhorizon'
    value = bundle.assemble(repo=REPO, accepted_code=FROZEN / 'code',
        candidate=FROZEN / 'candidate', map_path=FROZEN / 'source-map.json', output=root)
    state = fixture_vm_state(tmp_path)
    assert value['model_calls'] == 0 and value['candidate_files'] == 514 and value['code_files'] == 26
    assert len(host.source_manifest(root)) == 541
    code = '''
import sys,pathlib,importlib.util,json
def deny(event,args):
    if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
        raise AssertionError('external operation forbidden')
    if event=='open' and isinstance(args[0],(str,bytes)) and ('/.codex/' in str(args[0]) or '/.hymem-chatgpt-plan-lme/' in str(args[0])):
        raise AssertionError('real credential state forbidden')
sys.addaudithook(deny)
root=pathlib.Path(sys.argv[1]);state=pathlib.Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('root_isolated_owner',root/'code/tools/diagnostics/siwc_lme_diagnostic_v11.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
loaded=runner.import_source_only(root,root/'source-map.json',runner.ACCEPTED_INVENTORY_SHA256)
bridge=loaded['siwc']
assert loaded['source_only'] and 'broker' not in loaded
assert pathlib.Path(bridge.__file__).name=='chatgpt_plan_lme_v8.py'
assert pathlib.Path(bridge.owner.__file__).name=='lme_chatgpt_plan_owner_v3.py'
assert pathlib.Path(bridge.owner._pinned_refresh().__file__).name=='lme_chatgpt_plan_refresh_v3.py'
seen=[]
def response(credentials,system,user,schema,*,timeout):
    assert credentials.access_token=='invented-access'
    seen.append((timeout,schema is not None))
    return bridge.transport.Completed('invented',2,1,3,0,0)
with bridge.owner.CredentialBroker(state,pathlib.Path(sys.executable)) as broker:
    budget=bridge.SharedBudget(bridge.warm.BudgetLimits(*runner.MAX_LIMITS['campaign']))
    limits=bridge.warm.BudgetLimits(*runner.MAX_LIMITS['question'])
    ordinary=bridge.SIWCLMEClient(broker,budget,'q',limits,response_call=response)
    structured=bridge.SIWCLMEClient(broker,runner.RegistrationAlias(budget,'q',limits),'q',limits,response_call=response)
    assert ordinary.complete(bridge.LLMRequest('invented s','invented u'))=='invented'
    source=bridge.staged_v6.classification.GroundingSource(7,'Mira uses CairnDB.',source_role='user',source_peer_id='invented',source_created_at='2026-09-30')
    triple=bridge.staged_v6.classification.Triple('Mira','uses','CairnDB',1,source_message_id=7)
    request,batch=bridge.staged_v6.staged.build_original_request((triple,),(source,))
    assert structured.complete_stage(request,batch,'original',False)=='invented'
    assert [v[1] for v in seen]==[False,True] and all(599<v[0]<=600 for v in seen)
    snap=budget.snapshot()
    assert snap['turns']==2 and snap['known_tokens']==6 and snap['usage_complete']
    assert snap['in_flight']==snap['reserved']==0
print(json.dumps({'real_owner_admission':True,'forwarded_deadline':600,'model_calls':0}))
'''
    done = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(root), str(state)],
        capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {'real_owner_admission': True, 'forwarded_deadline': 600, 'model_calls': 0}
