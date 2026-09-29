"""Independent parent controls using the real guarded client with a fake SDK."""
import importlib.util
from pathlib import Path
import sys

import pytest

from hymem.contrib.openai_client import OpenAICompatibleClient
from hymem.contrib.openai_embedding_client import OpenAICompatibleEmbeddingClient
from hymem.dreaming.aggregation_material import embedding_producer_binding
from hymem.extraction.llm import LLMRequest
from hymem.extraction.producer import phase1_generation_binding
from tests.test_openai_client import _RecordingOpenAI, _RecordingCompletions
from tests.test_message_semantic import _fake_openai_module

spec = importlib.util.spec_from_file_location('instrumented_root_worker', Path(__file__).parents[1] / 'claim_conflict_instrumented_dream.py')
worker = importlib.util.module_from_spec(spec)
prior_path = sys.path[:]
spec.loader.exec_module(worker)
sys.path[:] = prior_path


def prepare(monkeypatch, tmp_path, *, completions=1, attempts=3):
    import openai
    for key in tuple(__import__('os').environ):
        if key.startswith(('HYMEM_', 'OPENAI_', 'DEEPSEEK_')):
            monkeypatch.delenv(key)
    calls = []
    original_create = _RecordingCompletions.create
    def create(self, **kwargs):
        response = original_create(self, **kwargs)
        response.model_dump = lambda **_kw: {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}
        return response
    monkeypatch.setattr(_RecordingCompletions, 'create', create)
    monkeypatch.setattr(openai, 'OpenAI', _RecordingOpenAI(calls, []))
    client = OpenAICompatibleClient(api_key='dummy-not-a-real-key', model='deepseek-flash', base_url='https://api.deepseek.com')
    journal = worker.PrivateJournal(tmp_path)
    complete, attempt, embedding, extraction, persist = worker.code_points()
    probe = worker.Probe(journal, tmp_path / 'unused.sqlite', completion_code=complete,
                         attempt_code=attempt, embedding_code=embedding, extraction_code=extraction,
                         persist_code=persist, max_completions=completions, max_attempts=attempts)
    return calls, client, journal, probe


def test_profile_preserves_guarded_producer_and_blocks_second_completion(monkeypatch, tmp_path):
    calls, client, journal, probe = prepare(monkeypatch, tmp_path)
    before = phase1_generation_binding('fixture-prompt-v1', client)
    try:
        sys.setprofile(probe.profile)
        assert client.complete(LLMRequest(system='synthetic', user='synthetic')) == 'ok'
        assert phase1_generation_binding('fixture-prompt-v1', client) == before
        with pytest.raises(worker.BudgetStop):
            client.complete(LLMRequest(system='synthetic', user='synthetic'))
    finally:
        sys.setprofile(None)
        journal.close()
        client.close()
    assert len(calls) == 1
    assert probe.completions == 1 and probe.llm_attempts == 1


def test_http_cap_is_checked_before_provider_dispatch(monkeypatch, tmp_path):
    calls, client, journal, probe = prepare(monkeypatch, tmp_path, attempts=0)
    try:
        sys.setprofile(probe.profile)
        with pytest.raises(worker.BudgetStop):
            client.complete(LLMRequest(system='synthetic', user='synthetic'))
    finally:
        sys.setprofile(None)
        journal.close()
        client.close()
    assert calls == []
    assert probe.completions == 1 and probe.llm_attempts == 0


def test_journal_failure_cannot_continue_unmetered(monkeypatch, tmp_path):
    calls, client, journal, probe = prepare(monkeypatch, tmp_path)
    def fail(*args):
        raise OSError('private-synthetic-message')
    monkeypatch.setattr(journal, 'append', fail)
    try:
        sys.setprofile(probe.profile)
        with pytest.raises(worker.InstrumentationStop):
            # A broad application Exception handler must not swallow this.
            try:
                client.complete(LLMRequest(system='synthetic', user='synthetic'))
            except Exception:
                pytest.fail('instrumentation error is swallowable')
    finally:
        sys.setprofile(None)
        journal.close()
        client.close()
    assert calls == []


def test_real_embedding_client_exact_code_rejects_before_fake_sdk(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pytest

    calls = []
    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.embeddings = SimpleNamespace(create=lambda **request: calls.append(request))
        def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'openai', _fake_openai_module(FakeOpenAI))
    client = OpenAICompatibleEmbeddingClient(
        api_key='synthetic', base_url='https://embed.example/v1', model='m', dim=3,
        pin_dimension=True, deployment_revision='revision-v1',
        deployment_tenant='tenant-v1')
    journal = worker.PrivateJournal(tmp_path)
    complete, attempt, embedding, extraction, persist = worker.code_points()
    assert embedding is OpenAICompatibleEmbeddingClient._embed_with_locked_transport.__code__
    probe = worker.Probe(journal, tmp_path / 'unused.sqlite',
                         completion_code=complete, attempt_code=attempt,
                         embedding_code=embedding, extraction_code=extraction,
                         persist_code=persist)
    before = embedding_producer_binding(client)
    try:
        sys.setprofile(probe.profile)
        with pytest.raises(worker.BudgetStop):
            client.embed(['synthetic'] * 17)
    finally:
        sys.setprofile(None)
        after = embedding_producer_binding(client)
        journal.close()
        client.close()
    assert calls == []
    assert probe.attempts == probe.embedding_attempts == 0
    assert after == before
