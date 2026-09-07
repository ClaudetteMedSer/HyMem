from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import hymem.bootstrap as bootstrap
from hymem.dreaming.scheduler import DreamScheduler


class _Closable:
    model = "lifecycle-test"
    dim = 2

    def __init__(
        self,
        label: str,
        events: list[str],
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.label = label
        self.events = events
        self.failure = failure
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.events.append(f"{self.label}:close")
        if self.failure is not None:
            raise self.failure


class _FakeHyMem:
    def __init__(
        self,
        _config,
        *,
        llm,
        embedding_client,
        events: list[str] | None = None,
        close_failure: BaseException | None = None,
    ) -> None:
        self.llm = llm
        self.embedding_client = embedding_client
        self.events = events if events is not None else []
        self.close_failure = close_failure
        self.close_calls = 0
        self.conn = object()

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("store:close")
        if self.close_failure is not None:
            raise self.close_failure


@pytest.fixture(autouse=True)
def _isolated_singleton(monkeypatch):
    monkeypatch.setattr(bootstrap, "_instance", None)


def _env(tmp_path: Path, *, remote_embedding: bool = True) -> bootstrap.EnvConfig:
    return bootstrap.EnvConfig(
        root=tmp_path,
        llm_api_key="purpose-key",
        llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-v4-flash",
        embedding_api_key="embedding-key" if remote_embedding else None,
        embedding_base_url=(
            "https://api.openai.com/v1"
            if remote_embedding
            else bootstrap.DEFAULT_EMBEDDING_BASE_URL
        ),
        embedding_model=(
            "text-embedding-3-small"
            if remote_embedding
            else bootstrap.DEFAULT_EMBEDDING_MODEL
        ),
        embedding_dim=2,
        embedding_backend=(
            "openai_compatible" if remote_embedding else "local_feature_hash"
        ),
        embedding_fallback_reason=None,
        aggregation_nodes_enabled=None,
        aggregation_digest_enabled=None,
        embedding_pin_dimension=remote_embedding,
        embedding_deployment_revision=(
            "fixture-release-2026-09" if remote_embedding else None
        ),
        embedding_deployment_tenant=(
            "fixture-tenant" if remote_embedding else None
        ),
    )


def _patch_successful_build(
    monkeypatch,
    tmp_path: Path,
    events: list[str],
    *,
    llm: object | None = None,
    embedding: object | None = None,
    hy_factory=None,
):
    import hymem.contrib.openai_client as llm_module
    import hymem.contrib.openai_embedding_client as embedding_module

    llm = llm if llm is not None else _Closable("llm", events)
    embedding = (
        embedding if embedding is not None else _Closable("embedding", events)
    )
    monkeypatch.setattr(bootstrap, "resolve_env", lambda: _env(tmp_path))
    monkeypatch.setattr(
        llm_module, "OpenAICompatibleClient", lambda **_kwargs: llm
    )
    monkeypatch.setattr(
        embedding_module,
        "OpenAICompatibleEmbeddingClient",
        lambda **_kwargs: embedding,
    )
    if hy_factory is None:
        hy_factory = lambda config, *, llm, embedding_client: _FakeHyMem(
            config,
            llm=llm,
            embedding_client=embedding_client,
            events=events,
        )
    monkeypatch.setattr(bootstrap, "HyMem", hy_factory)
    return llm, embedding


def test_owned_shutdown_orders_scheduler_store_embedding_then_llm_and_is_idempotent(
    monkeypatch, tmp_path
):
    events: list[str] = []
    llm, embedding = _patch_successful_build(monkeypatch, tmp_path, events)
    instance = bootstrap.get_instance()

    assert bootstrap.shutdown_instance(
        stop_background=lambda: events.append("scheduler:stop")
    ) is True
    assert events == [
        "scheduler:stop",
        "store:close",
        "embedding:close",
        "llm:close",
    ]
    assert instance.close_calls == embedding.close_calls == llm.close_calls == 1
    assert bootstrap.shutdown_instance(instance) is False
    assert instance.close_calls == embedding.close_calls == llm.close_calls == 1


def test_shared_embedding_and_llm_transport_is_closed_once(monkeypatch, tmp_path):
    events: list[str] = []
    shared = _Closable("shared", events)
    _patch_successful_build(
        monkeypatch, tmp_path, events, llm=shared, embedding=shared
    )
    instance = bootstrap.get_instance()

    assert bootstrap.shutdown_instance() is True
    assert events == ["store:close", "shared:close"]
    assert shared.close_calls == 1
    assert instance.embedding_client._closed is True


def test_background_stop_failure_defers_every_dependent_close(monkeypatch, tmp_path):
    events: list[str] = []
    llm, embedding = _patch_successful_build(monkeypatch, tmp_path, events)
    instance = bootstrap.get_instance()

    def blocked_stop():
        events.append("scheduler:timeout")
        raise TimeoutError("still running")

    with pytest.raises(TimeoutError, match="still running"):
        bootstrap.shutdown_instance(stop_background=blocked_stop)

    assert events == ["scheduler:timeout"]
    assert instance.close_calls == embedding.close_calls == llm.close_calls == 0
    assert bootstrap.get_instance() is instance

    assert bootstrap.shutdown_instance(
        stop_background=lambda: events.append("scheduler:stopped")
    ) is True
    assert events == [
        "scheduler:timeout",
        "scheduler:stopped",
        "store:close",
        "embedding:close",
        "llm:close",
    ]


def test_blocked_real_scheduler_prevents_singleton_dependency_teardown(
    monkeypatch, tmp_path
):
    events: list[str] = []
    entered = threading.Event()
    release = threading.Event()
    llm, embedding = _patch_successful_build(monkeypatch, tmp_path, events)
    instance = bootstrap.get_instance()

    class Fork:
        def dream(self):
            entered.set()
            assert release.wait(timeout=5.0)
            events.append("dream:return")

        def close(self):
            events.append("fork:close")

    class SchedulerRoot:
        def fork(self):
            return Fork()

        def invalidate_query_caches(self):
            events.append("root:invalidate")

    scheduler = DreamScheduler(SchedulerRoot(), cooldown=0.0)
    scheduler.start()
    scheduler.kick()
    assert entered.wait(timeout=2.0)

    with pytest.raises(TimeoutError, match="did not stop"):
        bootstrap.shutdown_instance(
            stop_background=lambda: scheduler.stop(timeout=0.01)
        )
    assert instance.close_calls == embedding.close_calls == llm.close_calls == 0

    release.set()
    assert bootstrap.shutdown_instance(
        stop_background=lambda: scheduler.stop(timeout=2.0)
    ) is True
    assert events == [
        "dream:return",
        "root:invalidate",
        "fork:close",
        "store:close",
        "embedding:close",
        "llm:close",
    ]


def test_injected_instance_and_clients_remain_caller_owned(monkeypatch):
    events: list[str] = []
    injected = _FakeHyMem(
        object(),
        llm=_Closable("llm", events),
        embedding_client=_Closable("embedding", events),
        events=events,
    )
    bootstrap.set_instance(injected)

    assert bootstrap.shutdown_instance(
        stop_background=lambda: events.append("scheduler:stop")
    ) is False
    assert events == ["scheduler:stop"]
    assert bootstrap.get_instance() is injected


def test_live_owned_singleton_cannot_be_replaced_without_shutdown(
    monkeypatch, tmp_path
):
    events: list[str] = []
    _patch_successful_build(monkeypatch, tmp_path, events)
    owned = bootstrap.get_instance()
    injected = _FakeHyMem(object(), llm=None, embedding_client=None)

    with pytest.raises(RuntimeError, match="shutdown_instance"):
        bootstrap.set_instance(injected)
    assert bootstrap.get_instance() is owned

    bootstrap.shutdown_instance()
    bootstrap.set_instance(injected)
    assert bootstrap.get_instance() is injected


def test_explicit_non_singleton_shutdown_does_not_perturb_other_singleton(
    monkeypatch, tmp_path
):
    events: list[str] = []
    _patch_successful_build(monkeypatch, tmp_path, events)
    owned = bootstrap.build_from_env()
    singleton = _FakeHyMem(object(), llm=None, embedding_client=None)
    bootstrap.set_instance(singleton)

    assert bootstrap.shutdown_instance(
        owned,
        stop_background=lambda: events.append("wrong-scheduler:stop"),
    ) is True
    assert bootstrap.get_instance() is singleton
    assert owned.close_calls == 1
    assert "wrong-scheduler:stop" not in events


def test_getter_cannot_observe_singleton_during_blocked_shutdown(
    monkeypatch, tmp_path
):
    events: list[str] = []
    close_entered = threading.Event()
    release_close = threading.Event()

    class BlockingHy(_FakeHyMem):
        def close(self):
            self.close_calls += 1
            events.append("store:close-enter")
            close_entered.set()
            assert release_close.wait(timeout=5.0)
            events.append("store:close-return")

    _patch_successful_build(
        monkeypatch,
        tmp_path,
        events,
        hy_factory=lambda config, *, llm, embedding_client: BlockingHy(
            config,
            llm=llm,
            embedding_client=embedding_client,
            events=events,
        ),
    )
    old = bootstrap.get_instance()
    replacement = _FakeHyMem(object(), llm=None, embedding_client=None)
    monkeypatch.setattr(bootstrap, "build_from_env", lambda: replacement)

    shutdown_thread = threading.Thread(target=bootstrap.shutdown_instance)
    shutdown_thread.start()
    assert close_entered.wait(timeout=2.0)

    observed: list[object] = []
    getter = threading.Thread(target=lambda: observed.append(bootstrap.get_instance()))
    getter.start()
    time.sleep(0.05)
    assert getter.is_alive(), "getter must block while the old singleton closes"

    release_close.set()
    shutdown_thread.join(timeout=2.0)
    getter.join(timeout=2.0)
    assert not shutdown_thread.is_alive() and not getter.is_alive()
    assert observed == [replacement]
    assert observed[0] is not old


def test_shutdown_close_faults_preserve_first_control_flow_and_attempt_once(
    monkeypatch, tmp_path
):
    events: list[str] = []
    store_fault = KeyboardInterrupt("store-secret-/private/store")
    embedding_fault = SystemExit("embedding-secret-/private/embed")
    llm_fault = RuntimeError("llm-secret-/private/llm")
    llm = _Closable("llm", events, failure=llm_fault)
    embedding = _Closable("embedding", events, failure=embedding_fault)
    _patch_successful_build(
        monkeypatch,
        tmp_path,
        events,
        llm=llm,
        embedding=embedding,
        hy_factory=lambda config, *, llm, embedding_client: _FakeHyMem(
            config,
            llm=llm,
            embedding_client=embedding_client,
            events=events,
            close_failure=store_fault,
        ),
    )
    instance = bootstrap.get_instance()

    with pytest.raises(KeyboardInterrupt) as caught:
        bootstrap.shutdown_instance()

    assert caught.value is store_fault
    notes = " ".join(getattr(store_fault, "__notes__", ()))
    assert "SystemExit" in notes and "RuntimeError" in notes
    assert "secret" not in notes and "/private" not in notes
    assert events == ["store:close", "embedding:close", "llm:close"]
    assert bootstrap.shutdown_instance(instance) is False
    assert instance.close_calls == embedding.close_calls == llm.close_calls == 1


def test_llm_construction_failure_has_no_partial_resource(monkeypatch, tmp_path):
    import hymem.contrib.openai_client as llm_module

    primary = RuntimeError("llm construction")
    monkeypatch.setattr(bootstrap, "resolve_env", lambda: _env(tmp_path))
    monkeypatch.setattr(
        llm_module,
        "OpenAICompatibleClient",
        lambda **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(RuntimeError) as caught:
        bootstrap.build_from_env()
    assert caught.value is primary


def test_embedding_control_flow_construction_failure_closes_llm(
    monkeypatch, tmp_path
):
    import hymem.contrib.openai_client as llm_module
    import hymem.contrib.openai_embedding_client as embedding_module

    events: list[str] = []
    llm = _Closable("llm", events)
    primary = SystemExit("embedding construction")
    monkeypatch.setattr(bootstrap, "resolve_env", lambda: _env(tmp_path))
    monkeypatch.setattr(llm_module, "OpenAICompatibleClient", lambda **_kwargs: llm)
    monkeypatch.setattr(
        embedding_module,
        "OpenAICompatibleEmbeddingClient",
        lambda **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(SystemExit) as caught:
        bootstrap.build_from_env()
    assert caught.value is primary
    assert events == ["llm:close"]


def test_cache_construction_failure_closes_raw_embedding_then_llm(
    monkeypatch, tmp_path
):
    import hymem.contrib.openai_client as llm_module
    import hymem.contrib.openai_embedding_client as embedding_module
    import hymem.extraction.embeddings as embeddings_module

    events: list[str] = []
    llm = _Closable("llm", events)
    embedding = _Closable("embedding", events)
    primary = RuntimeError("cache construction")

    class BrokenCache:
        def __init__(self, _inner):
            raise primary

    monkeypatch.setattr(bootstrap, "resolve_env", lambda: _env(tmp_path))
    monkeypatch.setattr(llm_module, "OpenAICompatibleClient", lambda **_kwargs: llm)
    monkeypatch.setattr(
        embedding_module,
        "OpenAICompatibleEmbeddingClient",
        lambda **_kwargs: embedding,
    )
    monkeypatch.setattr(embeddings_module, "CachedEmbeddingClient", BrokenCache)

    with pytest.raises(RuntimeError) as caught:
        bootstrap.build_from_env()
    assert caught.value is primary
    assert events == ["embedding:close", "llm:close"]


@pytest.mark.parametrize("stage", ["config", "store", "eager_connection"])
def test_late_bootstrap_stage_failure_closes_every_constructed_resource(
    monkeypatch, tmp_path, stage
):
    events: list[str] = []
    llm, embedding = _patch_successful_build(monkeypatch, tmp_path, events)
    primary = KeyboardInterrupt(f"{stage} construction")

    if stage == "config":
        monkeypatch.setattr(
            bootstrap,
            "HyMemConfig",
            lambda **_kwargs: (_ for _ in ()).throw(primary),
        )
    elif stage == "store":
        monkeypatch.setattr(
            bootstrap,
            "HyMem",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )
    else:
        class EagerConnectionFailure(_FakeHyMem):
            @property
            def conn(self):
                raise primary

            @conn.setter
            def conn(self, value):
                self._constructed_conn = value

        monkeypatch.setattr(
            bootstrap,
            "HyMem",
            lambda config, *, llm, embedding_client: EagerConnectionFailure(
                config,
                llm=llm,
                embedding_client=embedding_client,
                events=events,
            ),
        )

    with pytest.raises(KeyboardInterrupt) as caught:
        bootstrap.build_from_env()

    assert caught.value is primary
    expected = ["embedding:close", "llm:close"]
    if stage == "eager_connection":
        expected.insert(0, "store:close")
    assert events == expected
    assert embedding.close_calls == llm.close_calls == 1


def test_primary_build_failure_outweighs_all_cleanup_control_flow(
    monkeypatch, tmp_path
):
    events: list[str] = []
    primary = KeyboardInterrupt("primary-secret-/private/primary")
    llm = _Closable(
        "llm", events, failure=SystemExit("cleanup-secret-/private/llm")
    )
    embedding = _Closable(
        "embedding", events, failure=RuntimeError("cleanup-secret-/private/embed")
    )
    _patch_successful_build(
        monkeypatch, tmp_path, events, llm=llm, embedding=embedding
    )
    monkeypatch.setattr(
        bootstrap,
        "HyMem",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        bootstrap.build_from_env()

    assert caught.value is primary
    notes = " ".join(getattr(primary, "__notes__", ()))
    assert "RuntimeError" in notes and "SystemExit" in notes
    assert "secret" not in notes and "/private" not in notes
    assert events == ["embedding:close", "llm:close"]
