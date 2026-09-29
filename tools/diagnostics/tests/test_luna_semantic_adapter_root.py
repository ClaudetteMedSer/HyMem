"""Independent v2 control-plane isolation and pre-inference failure controls."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import codex_subscription_warm_v3 as warm
from tools.diagnostics import luna_semantic_probe_adapter_v2 as adapter
from tools.diagnostics import luna_semantic_probe_host as host


@pytest.mark.parametrize('directory,endpoint',[
    ((0o40700,999),(0o140600,1000)),
    ((0o40750,1000),(0o140600,1000)),
    ((0o120700,1000),(0o140600,1000)),
    ((0o40700,1000),(0o100600,1000)),
    ((0o40700,1000),(0o120600,1000)),
    ((0o40700,1000),(0o140600,999)),
])
def test_root_bus_rejects_aliases_wrong_owner_and_non_socket(monkeypatch,directory,endpoint):
    monkeypatch.setattr(adapter.os,'getuid',lambda:1000)
    def lstat(path):
        mode,uid=directory if path==adapter.RUNTIME else endpoint
        return SimpleNamespace(st_mode=mode,st_uid=uid)
    monkeypatch.setattr(Path,'lstat',lstat)
    with pytest.raises(ValueError,match='trusted_user_bus_invalid'):
        adapter.bus_environment()


def test_root_bus_context_never_enters_inference_environment(monkeypatch):
    monkeypatch.setattr(adapter.os,'getuid',lambda:1000)
    def lstat(path):
        return SimpleNamespace(st_mode=0o40700 if path==adapter.RUNTIME else 0o140600,st_uid=1000)
    monkeypatch.setattr(Path,'lstat',lstat)
    env=adapter.bus_environment()
    assert set(env)=={'HOME','PATH','XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS'}
    assert env['DBUS_SESSION_BUS_ADDRESS']=='unix:path=/run/user/1000/bus'
    for key,value in env.items(): monkeypatch.setenv(key,value)
    monkeypatch.setenv('OPENAI_API_KEY','SYNTHETIC_NEVER_EXPORT')
    child=warm.base.sanitized_environment()
    assert 'DBUS_SESSION_BUS_ADDRESS' not in child and 'XDG_RUNTIME_DIR' not in child
    assert 'OPENAI_API_KEY' not in child


def test_root_scoped_query_restores_even_on_error(monkeypatch):
    module=SimpleNamespace(run=lambda *a,**k:None)
    called=[]
    def verify(root,receipt):
        called.append(True)
        raise RuntimeError('invented_status_failure')
    runner=SimpleNamespace(subprocess=module,verify_live_containment=verify)
    before=dict(os.environ)
    with pytest.raises(RuntimeError):
        adapter.contained(Path('/unused'),{},runner)
    assert called==[True] and runner.subprocess is module and dict(os.environ)==before


def test_root_source_hash_precedes_execution(tmp_path):
    sentinel=tmp_path/'sentinel'
    source=tmp_path/'source.py'
    source.write_text(f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n')
    with pytest.raises(ValueError,match='accepted_source_pin_invalid'):
        adapter.load_source(source,'root_must_not_load','0'*64)
    assert not sentinel.exists()
    alias=tmp_path/'alias.py'
    alias.symlink_to(source)
    with pytest.raises(ValueError,match='accepted_source_pin_invalid'):
        adapter.load_source(alias,'root_must_not_load',adapter.digest(source))
    assert not sentinel.exists()


@pytest.mark.parametrize('failure,preexisting_run',[(False,False),(True,False),(True,True)])
def test_root_smoke_never_calls_campaign_and_zero_claim_requires_no_run(tmp_path,monkeypatch,failure,preexisting_run):
    root=tmp_path
    (root/'launch-attempt.json').write_text('{}')
    if preexisting_run: (root/'run').mkdir()
    calls=[]
    def execute(*a,**k):
        calls.append(True)
        raise AssertionError('inference_unreachable_in_smoke')
    run=SimpleNamespace(preflight=lambda *a:({'unit':'test'},(),{},()),execute=execute)
    fake_host=SimpleNamespace(root_valid=lambda _:True,write_once=host.write_once)
    def source(path,*a):
        return run if path.name=='luna_semantic_probe_run.py' else fake_host
    monkeypatch.setattr(adapter,'load_source',source)
    monkeypatch.setattr(adapter,'verify',lambda *a:{})
    def contained(*a):
        if failure: raise ValueError('invented_missing_bus')
    monkeypatch.setattr(adapter,'contained',contained)
    rc=adapter.entry(root,'1'*64,'smoke','2'*64,'3'*64)
    assert not calls and rc==int(failure)
    if failure and preexisting_run:
        assert not(root/'safe-terminal.json').exists()
    else:
        terminal=json.loads((root/'safe-terminal.json').read_bytes())
        assert terminal['model_calls']==terminal['paid_turns']==terminal['known_tokens']==0
        if failure:
            assert not terminal['completed_and_clean']
            assert terminal['zero_admission_proof']=='run_directory_never_created'
        else: assert terminal['policy_verified']
