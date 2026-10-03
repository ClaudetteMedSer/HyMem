"""Independent regressions for the summary/index separation review."""
import pytest

from hymem.dreaming import runner
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from tests.test_digest_publication import PublicationLLM


@pytest.mark.parametrize("alias", [
    "load_digest_staged_summary_state", "classify_summary_state",
    "mark_summary_current", "record_summary_failure",
])
def test_digest_generation_binds_summary_helpers_at_their_runner_call_sites(monkeypatch, alias):
    client = PublicationLLM()
    original = getattr(runner, alias)
    before = semantic_generation_suffix("digest", client)

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(runner, alias, replacement)
        assert semantic_generation_suffix("digest", client) != before
    assert getattr(runner, alias) is original
    assert semantic_generation_suffix("digest", client) == before
