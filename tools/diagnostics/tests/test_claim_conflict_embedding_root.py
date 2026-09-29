"""Independent parent controls for the isolated internal embedding verifier."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / filename)
    value = importlib.util.module_from_spec(spec)
    before = sys.path[:]
    try:
        spec.loader.exec_module(value)
    finally:
        sys.path[:] = before
    return value


worker = load('embedding_root_worker', 'claim_conflict_embedding_verify.py')
host = load('embedding_root_host', 'claim_conflict_embedding_host.py')


def test_provider_boundary_is_sixteen_and_preserves_original_text():
    calls = []
    def provider(texts):
        calls.append(texts)
    probe = worker.Admission(provider.__code__)
    payload = ['  exact 🪷 text  '] * 16
    try:
        sys.setprofile(probe.profile)
        provider(payload)
        with pytest.raises(worker.BudgetStop):
            provider(payload + ['one too many'])
    finally:
        sys.setprofile(None)
    assert calls == [payload]
    assert probe.http_attempts == 1


def test_invalid_utf8_is_rejected_before_transport():
    calls = []
    def provider(texts):
        calls.append(texts)
    probe = worker.Admission(provider.__code__)
    try:
        sys.setprofile(probe.profile)
        with pytest.raises(worker.BudgetStop):
            provider(['\ud800'])
    finally:
        sys.setprofile(None)
    assert not calls and probe.http_attempts == 0


def test_only_clone_work_is_writable_and_no_llm_environment_in_args():
    h = SimpleNamespace(RUNTIME=Path('/runtime'), RUNTIME_ENV=Path('/private/runtime.json'), IMAGE='sha256:pinned')
    for mode in ('offline', 'live'):
        command, mounts = host.configure(h, mode, 'a' * 64)
        assert [m for m in mounts if m[2]] == [(str(host.WORK), '/work', True)]
        assert command[command.index('--network') + 1] == ('hermes-net' if mode == 'live' else 'none')
        assert not any('LLM_API_KEY' in value or 'DEEPSEEK' in value for value in command)


def test_metadata_projection_discards_private_extras():
    value = host.projection({'status': 'captured_failure', 'http_attempts': 1,
                             'provider_request_attempts': 1,
                             'request': 'private', 'exception': 'private'})
    assert 'private' not in repr(value)
