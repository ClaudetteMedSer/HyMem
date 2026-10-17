"""The Hermes fact cap can be switched without changing its default."""

import pytest

from hymem import bootstrap


@pytest.mark.parametrize("raw,expected", [
    (None, None), ("1", 1), ("16", 16), ("32", 32), ("256", 256),
])
def test_fact_cap_override_accepts_only_contract_values(raw, expected):
    assert bootstrap._validated_facts_cap_override(raw) == expected


@pytest.mark.parametrize("raw", ["", "0", "257", "9999", "016", " 16", "16 ", "x"])
def test_fact_cap_override_rejects_invalid_values_without_echo(raw):
    with pytest.raises(RuntimeError, match="integer from 1 to 256") as error:
        bootstrap._validated_facts_cap_override(raw)
    if raw:
        assert raw not in str(error.value)


def _env(tmp_path, raw):
    return bootstrap.EnvConfig(
        root=tmp_path, llm_api_key="fixture-key",
        llm_base_url="https://api.deepseek.com", llm_model="deepseek-flash",
        embedding_api_key=None,
        embedding_base_url=bootstrap.DEFAULT_EMBEDDING_BASE_URL,
        embedding_model=bootstrap.DEFAULT_EMBEDDING_MODEL,
        embedding_dim=384, embedding_backend="local_feature_hash",
        embedding_fallback_reason=None,
        aggregation_nodes_enabled=None, aggregation_digest_enabled=None,
        dream_max_facts_per_session_raw=raw,
    )


@pytest.mark.parametrize("raw,expected", [(None, 8), ("16", 16), ("32", 32)])
def test_build_from_env_applies_fact_cap_and_retains_default(
    monkeypatch, tmp_path, raw, expected,
):
    import hymem.contrib.openai_client as llm_module

    class FakeLLM:
        def close(self):
            pass

    class FakeStore:
        def __init__(self, config, *, llm, embedding_client):
            self.config = config
            self.conn = object()

        def close(self):
            pass

    monkeypatch.setattr(bootstrap, "resolve_env", lambda: _env(tmp_path, raw))
    monkeypatch.setattr(llm_module, "OpenAICompatibleClient", lambda **_: FakeLLM())
    monkeypatch.setattr(bootstrap, "HyMem", FakeStore)
    instance = bootstrap.build_from_env()
    assert instance.config.dream_max_facts_per_session == expected


def test_invalid_fact_cap_stops_before_client_construction(monkeypatch, tmp_path):
    import hymem.contrib.openai_client as llm_module

    monkeypatch.setattr(bootstrap, "resolve_env", lambda: _env(tmp_path, "999"))
    monkeypatch.setattr(
        llm_module, "OpenAICompatibleClient",
        lambda **_: pytest.fail("invalid cap reached LLM construction"),
    )
    with pytest.raises(RuntimeError, match="integer from 1 to 256"):
        bootstrap.build_from_env()


def test_resolve_env_keeps_cap_raw_for_fail_closed_startup(monkeypatch):
    monkeypatch.setenv("HYMEM_DREAM_MAX_FACTS_PER_SESSION", "not-a-number")
    assert bootstrap.resolve_env().dream_max_facts_per_session_raw == "not-a-number"


@pytest.mark.parametrize("raw,expected", [
    (None, None), ("1", 1), ("3072", 3072), ("6144", 6144),
    ("32768", 32768),
])
def test_fact_token_override_accepts_only_contract_values(raw, expected):
    assert bootstrap._validated_facts_tokens_override(raw) == expected


@pytest.mark.parametrize("raw", ["", "0", "32769", "03072", " 6144", "6144 ", "x"])
def test_fact_token_override_rejects_invalid_values_without_echo(raw):
    with pytest.raises(RuntimeError, match="integer from 1 to 32768") as error:
        bootstrap._validated_facts_tokens_override(raw)
    if raw:
        assert raw not in str(error.value)


def test_build_from_env_applies_fact_only_token_budget(monkeypatch, tmp_path):
    import hymem.contrib.openai_client as llm_module

    class FakeLLM:
        def close(self):
            pass

    class FakeStore:
        def __init__(self, config, *, llm, embedding_client):
            self.config = config
            self.conn = object()

        def close(self):
            pass

    env = _env(tmp_path, "64")
    from dataclasses import replace
    env = replace(env, facts_extraction_max_tokens_raw="6144")
    monkeypatch.setattr(bootstrap, "resolve_env", lambda: env)
    monkeypatch.setattr(llm_module, "OpenAICompatibleClient", lambda **_: FakeLLM())
    monkeypatch.setattr(bootstrap, "HyMem", FakeStore)
    instance = bootstrap.build_from_env()
    assert instance.config.dream_max_facts_per_session == 64
    assert instance.config.facts_extraction_max_tokens == 6144
    assert instance.config.dream_digest_max_tokens == 3072
