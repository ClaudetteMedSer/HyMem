"""Keep the accepted short doctor probe in packaged application source."""
from types import SimpleNamespace
import pytest
from hymem import doctor
from hymem.contrib import openai_client
from hymem.extraction import llm

class LLMOutputTruncatedError(RuntimeError):
    """Synthetic admission error for checkouts predating the typed client."""

TruncatedError = getattr(llm, 'LLMOutputTruncatedError', LLMOutputTruncatedError)

@pytest.mark.parametrize('reply,status,error', [
    ('OK', doctor.OK, None),
    (TruncatedError(), doctor.FAIL, 'LLMOutputTruncatedError'),
    (ConnectionError('secret-key'), doctor.FAIL, 'ConnectionError'),
])
def test_probe_preserves_budget_admission_and_cleanup(monkeypatch, reply, status, error):
    calls = []
    class Probe:
        def complete(self, request):
            calls.append(request)
            if isinstance(reply, Exception): raise reply
            return reply
        def close(self): calls.append('close')
    monkeypatch.setattr(openai_client, 'OpenAICompatibleClient', lambda **_: Probe())
    cfg = SimpleNamespace(llm_base_url='https://api.deepseek.com/v1',
                          llm_model='deepseek-flash', has_llm_key=True,
                          llm_api_key='secret-key')
    result = doctor._check_llm(cfg)
    assert result.status == status
    assert len(calls) == 2 and calls[1] == 'close'
    request = calls[0]
    assert request.system == '' and request.user == 'Reply with exactly OK.'
    assert request.response_format == 'text' and request.max_tokens == 32
    if error:
        assert result.detail.endswith('unreachable: ' + error)
        assert 'secret-key' not in result.detail
