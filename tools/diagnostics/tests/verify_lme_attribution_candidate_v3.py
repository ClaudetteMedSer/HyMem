"""Real frozen HyMem dream/retrieval accounting with an invented-input fake provider."""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
CANDIDATE = Path('/private/tmp/hymem-lme-diagnostic-offline-assembly-v3/candidate')
if not CANDIDATE.is_dir():
    raise SystemExit('frozen_candidate_unavailable')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CANDIDATE))

from tools.diagnostics.luna_lme_diagnostic_v3 import AccountedClient  # noqa: E402
from hymem import HyMem, HyMemConfig  # noqa: E402
from hymem.extraction.llm import StubLLMClient, LLMRequest  # noqa: E402
from hymem.dreaming.facts import reextract_fact_outcome  # noqa: E402
from hymem.extraction import chunk  # noqa: E402
from benchmarks import extraction_canary as canary, longmemeval_adapter as lme  # noqa: E402


class Budget:
    def __init__(self):
        self.turns = 0
        self.tokens = 0
        self.halted = None

    def snapshot(self):
        return {'questions': {'q': {'turns': self.turns, 'known_tokens': self.tokens}}}

    def halt(self, code):
        self.halted = code


class FakeProvider:
    def __init__(self):
        self.budget = Budget()
        self.key = 'q'
        self.stub = StubLLMClient(fixtures={
            'Return the JSON object now': '{"episodes":[],"summary":"","procedures":[]}',
            'Return the JSON object of narrative facts now': '[]',
        }, default='[]')

    def complete(self, request):
        self.budget.turns += 1
        self.budget.tokens += 13
        return self.stub.complete(request)

    def complete_stage(self, request, batch, stage, recheck):
        self.budget.turns += 1
        self.budget.tokens += 13
        return '{}'


class RoutedClient:
    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete(self, request):
        return self.client.complete(request)

    def complete_stage(self, request, batch, stage, recheck):
        return self.client.complete_stage(request, batch, stage, recheck)


class ChatBridge:
    def __init__(self, client):
        self.client = client

    def chat(self, messages, *, temperature=0.0, max_tokens=1024):
        if len(messages) == 2:
            system, user = messages[0]['content'], messages[1]['content']
        else:
            system, user = '', messages[0]['content']
        return self.client.complete(LLMRequest(system=system, user=user,
            response_format='text', temperature=temperature, max_tokens=max_tokens))


def main():
    logging.disable(logging.CRITICAL)
    provider = FakeProvider()
    accounted = AccountedClient(provider, CANDIDATE)
    with tempfile.TemporaryDirectory(prefix='hymem-attribution-v3-') as scratch:
        routed = RoutedClient(accounted)
        hy = HyMem(HyMemConfig(root=Path(scratch)), llm=routed)
        try:
            hy.open_session('s')
            for index in range(10):
                hy.log_message('s', 'user',
                    f'MedFlow deploy location {index} is fly.io region {index}.')
            hy.close_session('s')
            hy.dream()
            hy.augment('Where does MedFlow deploy?')
            outcome = hy.conn.execute(
                'SELECT slice_key FROM fact_extraction_outcomes ORDER BY slice_key LIMIT 1').fetchone()
            assert outcome is not None
            before_replay = provider.budget.turns
            reextract_fact_outcome(hy.conn, outcome[0], routed, hy.config)
            assert provider.budget.turns == before_replay + 1
        finally:
            hy.close()
    bridge = ChatBridge(routed)
    lme.answer_question_raw(bridge, [], 'Where does MedFlow deploy?')
    lme.judge_answer_raw(bridge, 'single-session-user', 'Where does MedFlow deploy?', 'fly.io', 'fly.io')
    expected = {'extraction', 'digest', 'profile', 'facts', 'rerank', 'reader', 'judge'}
    assert expected.issubset(accounted.counts), accounted.counts
    assert sum(slot['turns'] for slot in accounted.counts.values()) == provider.budget.turns
    assert sum(slot['known_tokens'] for slot in accounted.counts.values()) == provider.budget.tokens
    assert accounted.reconcile()
    assert provider.budget.halted is None

    # The frozen chunk extractor calls the staged transport from grounding_call.
    # Local representative ordinary replies reach that branch; the invalid stage
    # reply is contained in the disposable result and cannot be published.
    import importlib.util
    spec = importlib.util.spec_from_file_location('frozen_canary_helpers',
        CANDIDATE / 'tests/test_benchmark_extraction_canary.py')
    assert spec and spec.loader
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    class GroundingProvider(FakeProvider):
        def complete(self, request):
            self.budget.turns += 1
            self.budget.tokens += 13
            return helpers._representative_response(request)

    staged_provider = GroundingProvider()
    staged = AccountedClient(staged_provider, CANDIDATE)
    chunk.extract_chunk(RoutedClient(staged), canary._CANARY_CONTENT,
        source_records=canary._source_records(),
        completion_call_limit=canary.EXTRACTION_CANARY_MAX_COMPLETION_CALLS)
    assert 'extraction' in staged.counts
    assert 'grounding_original_initial' in staged.counts
    assert staged.reconcile()
    assert staged_provider.budget.halted is None


if __name__ == '__main__':
    main()
