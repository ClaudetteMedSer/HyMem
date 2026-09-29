"""Local controls for prepared transport; no actual SSH calls."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.diagnostics import hymem_v64_transport as transport

STAGE='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/hymem-v64-tools-test'

def test_remote_compile_and_expected_file_modes():
    compile(transport.REMOTE,'<remote>','exec')
    assert "write(stage/name,base64.b64decode(body),0o400)" in transport.REMOTE
    assert "'--runtime-seal-output'" in transport.REMOTE
    assert "'hymem-v64-tools-[a-z0-9-]+'" in transport.REMOTE
    assert 'raw_environ' not in transport.REMOTE

def test_manifest_binds_all_helper_and_template_bytes():
    bodies,pins,manifest=transport.inputs()
    assert set(pins)==set(transport.FILES)
    assert 'hymem_v64_vector_check.py' in pins and 'lme_r7_postdeploy_verify.py' in pins
    assert all(hashlib.sha256(bodies[name]).hexdigest()==pin for name,pin in pins.items())
    assert len(manifest)==64

def test_bad_seal_rejects_before_any_ssh(monkeypatch):
    monkeypatch.setattr(transport.sys if hasattr(transport,'sys') else __import__('sys'),'argv',
                        ['transport','install','--tools-stage',STAGE,'--manifest-sha256','0'*64])
    monkeypatch.setattr(transport.subprocess,'run',lambda *a,**kw:pytest.fail('unexpected SSH'))
    with pytest.raises(RuntimeError,match='local_transport_seal_drift'):transport.main()

def test_explicit_install_only_uses_pinned_remote_payload(monkeypatch,capsys):
    import sys
    manifest=transport.inputs()[2];calls=[]
    monkeypatch.setattr(sys,'argv',['transport','install','--tools-stage',STAGE,'--manifest-sha256',manifest])
    monkeypatch.setattr(transport.subprocess,'run',lambda cmd,**kw:calls.append(cmd) or SimpleNamespace(returncode=0,stdout=b'{"status":"installed"}'))
    transport.main()
    assert len(calls)==1 and calls[0][0]=='ssh'
    assert 'chmod' not in calls[0][-1]
    assert json.loads(capsys.readouterr().out)=={'status':'installed'}
