"""Source-only host-preflight packaging checks; no SSH or inference."""
from __future__ import annotations

from io import BytesIO
import importlib.util
import json
from pathlib import Path
import tarfile


SOURCE = Path(__file__).resolve().parents[1] / 'luna_lme_diagnostic_host_preflight.py'
spec = importlib.util.spec_from_file_location('luna_lme_diagnostic_host_preflight', SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pinned_source_snapshot_and_archive():
    manifest = module.source_manifest()
    assert len(manifest) == 524
    assert manifest[module.RUNNER_REL] == module.RUNNER_SHA
    assert manifest['source-map.json'] == module.MAP_SHA
    payload = module.archive_bytes()
    with tarfile.open(fileobj=BytesIO(payload), mode='r:') as archive:
        members = archive.getmembers()
        assert members[0].name == 'manifest.json'
        assert all(member.isfile() for member in members)
        assert {member.name for member in members[1:]} == set(manifest)
        assert json.loads(archive.extractfile(members[0]).read()) == manifest


def test_remote_code_syntax_and_no_run_action():
    compile(module.REMOTE, '<host-remote>', 'exec')
    compile(module.wrapped_remote(), '<host-remote-wrapped>', 'exec')
    assert '--preflight-only' in module.REMOTE
    assert '--run' not in module.REMOTE


def test_main_uploads_once_without_printing_runner_data(monkeypatch, capsys):
    class Result:
        returncode = 0
        stdout = json.dumps({'preflight_verified': True, 'model_calls': 0,
            'root': '/home/atta/.hymem-lme-diagnostic-preflight-test'}).encode()
    calls = []
    monkeypatch.setattr(module, 'archive_bytes', lambda: b'private-source-archive')
    monkeypatch.setattr(module.subprocess, 'run',
        lambda command, **kwargs: calls.append((command, kwargs)) or Result())
    assert module.main(['--host', 'luna']) == 0
    assert len(calls) == 1
    assert calls[0][1]['input'] == b'private-source-archive'
    assert 'private-source-archive' not in capsys.readouterr().out
