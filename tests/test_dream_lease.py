from __future__ import annotations

import contextlib
import sqlite3
import threading

import pytest

import hymem.bootstrap as bootstrap
from hymem import DreamLeaseLost, HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import runner
from hymem.extraction.llm import LLMRequest, StubLLMClient


def _open_pair(tmp_path):
    path = tmp_path / "lease.sqlite"
    first = core_db.connect(path)
    core_db.initialize(first)
    second = core_db.connect(path)
    return first, second


def test_lease_tokens_are_unique_per_invocation():
    first = runner._new_lease_token()
    second = runner._new_lease_token()

    assert first != second
    assert len(first.rsplit(":", 1)[-1]) == 32
    assert len(second.rsplit(":", 1)[-1]) == 32


def test_second_bootstrap_preserves_an_active_cross_process_lease(
    tmp_path, monkeypatch,
):
    import hymem.contrib.openai_client as llm_module

    env = bootstrap.EnvConfig(
        root=tmp_path,
        llm_api_key="test-purpose-key",
        llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-v4-flash",
        embedding_api_key=None,
        embedding_base_url=bootstrap.DEFAULT_EMBEDDING_BASE_URL,
        embedding_model=bootstrap.DEFAULT_EMBEDDING_MODEL,
        embedding_dim=bootstrap.DEFAULT_EMBEDDING_DIM,
        embedding_backend="local_feature_hash",
        embedding_fallback_reason=None,
        aggregation_nodes_enabled=False,
        aggregation_digest_enabled=False,
    )
    monkeypatch.setattr(bootstrap, "resolve_env", lambda: env)
    monkeypatch.setattr(
        llm_module,
        "OpenAICompatibleClient",
        lambda **_kwargs: StubLLMClient(default="[]"),
    )

    first = bootstrap.build_from_env()
    second = None
    try:
        assert runner._acquire_lock(first.conn, "live-owner-token") is True
        second = bootstrap.build_from_env()
        row = second.conn.execute(
            "SELECT holder FROM run_lock WHERE name='dreaming'"
        ).fetchone()
        assert row is not None
        assert row["holder"] == "live-owner-token"
    finally:
        if second is not None:
            second.conn.execute("DELETE FROM run_lock WHERE name='dreaming'")
            bootstrap.shutdown_instance(second)
        bootstrap.shutdown_instance(first)


def test_simultaneous_stale_takeover_has_exactly_one_winner(tmp_path):
    first, second = _open_pair(tmp_path)
    try:
        first.execute(
            "INSERT INTO run_lock(name,acquired_at,holder) "
            "VALUES ('dreaming',datetime('now','-1 hour'),'dead-owner')"
        )
        barrier = threading.Barrier(3)
        outcomes: list[tuple[str, bool]] = []
        failures: list[BaseException] = []

        def attempt(conn, token):
            try:
                barrier.wait()
                outcomes.append((token, runner._acquire_lock(conn, token)))
            except BaseException as exc:  # preserve thread failures for parent
                failures.append(exc)

        threads = [
            threading.Thread(target=attempt, args=(first, "candidate-a")),
            threading.Thread(target=attempt, args=(second, "candidate-b")),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5.0)

        assert not failures
        assert all(not thread.is_alive() for thread in threads)
        assert sum(int(acquired) for _, acquired in outcomes) == 1
        winner = next(token for token, acquired in outcomes if acquired)
        row = first.execute(
            "SELECT holder FROM run_lock WHERE name='dreaming'"
        ).fetchone()
        assert row["holder"] == winner
    finally:
        first.close()
        second.close()


def test_renew_and_release_are_exact_token_conditional(tmp_path):
    first, second = _open_pair(tmp_path)
    try:
        assert runner._acquire_lock(first, "owner-a") is True
        with pytest.raises(core_db.LeaseOwnershipLost):
            runner._refresh_lock(second, "intruder")
        assert runner._release_lock(second, "intruder") is False

        first.execute(
            "UPDATE run_lock SET acquired_at=datetime('now','-1 hour') "
            "WHERE name='dreaming'"
        )
        assert runner._acquire_lock(second, "owner-b") is True
        assert runner._release_lock(first, "owner-a") is False
        assert first.execute(
            "SELECT holder FROM run_lock WHERE name='dreaming'"
        ).fetchone()["holder"] == "owner-b"
        runner._refresh_lock(second, "owner-b")
        assert runner._release_lock(second, "owner-b") is True
    finally:
        first.close()
        second.close()


def test_transaction_fence_rolls_back_after_forced_takeover(tmp_path):
    first, second = _open_pair(tmp_path)
    first.execute("CREATE TABLE semantic_probe(value TEXT)")
    try:
        assert runner._acquire_lock(first, "owner-a") is True
        fence = core_db.activate_transaction_lease_fence(
            first, name="dreaming", holder="owner-a"
        )
        try:
            # The second proof immediately before COMMIT also catches an
            # accidental same-transaction deletion and rolls the whole unit
            # back, including the lease row itself.
            with pytest.raises(core_db.LeaseOwnershipLost):
                with core_db.transaction(first):
                    first.execute(
                        "INSERT INTO semantic_probe(value) VALUES ('also-forbidden')"
                    )
                    first.execute(
                        "DELETE FROM run_lock WHERE name='dreaming' AND holder='owner-a'"
                    )
            assert first.execute(
                "SELECT holder FROM run_lock WHERE name='dreaming'"
            ).fetchone()["holder"] == "owner-a"

            second.execute(
                "UPDATE run_lock SET holder='owner-b',"
                "acquired_at=CURRENT_TIMESTAMP WHERE name='dreaming'"
            )
            with pytest.raises(core_db.LeaseOwnershipLost):
                with core_db.transaction(first):
                    first.execute(
                        "INSERT INTO semantic_probe(value) VALUES ('forbidden')"
                    )
        finally:
            core_db.deactivate_transaction_lease_fence(fence)

        assert first.in_transaction is False
        assert first.execute("SELECT * FROM semantic_probe").fetchall() == []
        assert runner._release_lock(first, "owner-a") is False
        assert first.execute(
            "SELECT holder FROM run_lock WHERE name='dreaming'"
        ).fetchone()["holder"] == "owner-b"
    finally:
        second.execute("DELETE FROM run_lock WHERE name='dreaming'")
        first.close()
        second.close()


def test_commit_failure_rolls_back_and_does_not_strand_the_lease(tmp_path):
    owner, observer = _open_pair(tmp_path)
    owner.execute("CREATE TABLE semantic_probe(value TEXT)")
    commit_failure = sqlite3.OperationalError("injected commit failure")

    class CommitFailureProxy:
        @property
        def in_transaction(self):
            return owner.in_transaction

        def execute(self, sql, parameters=()):
            if sql == "COMMIT":
                raise commit_failure
            return owner.execute(sql, parameters)

    proxy = CommitFailureProxy()
    try:
        assert runner._acquire_lock(owner, "owner-token") is True
        fence = core_db.activate_transaction_lease_fence(
            proxy, name="dreaming", holder="owner-token"
        )
        try:
            with pytest.raises(sqlite3.OperationalError) as caught:
                with core_db.transaction(proxy):
                    proxy.execute(
                        "INSERT INTO semantic_probe(value) VALUES ('forbidden')"
                    )
        finally:
            core_db.deactivate_transaction_lease_fence(fence)

        assert caught.value is commit_failure
        assert owner.in_transaction is False
        assert owner.execute("SELECT * FROM semantic_probe").fetchall() == []
        assert runner._release_lock(proxy, "owner-token") is True
        assert runner._acquire_lock(observer, "successor-token") is True
    finally:
        observer.execute("DELETE FROM run_lock WHERE name='dreaming'")
        owner.close()
        observer.close()


def test_rollback_failure_never_masks_original_transaction_exception(tmp_path):
    owner, observer = _open_pair(tmp_path)
    primary = core_db.LeaseOwnershipLost("dreaming lease ownership lost")

    class RollbackFailureProxy:
        @property
        def in_transaction(self):
            return owner.in_transaction

        def execute(self, sql, parameters=()):
            result = owner.execute(sql, parameters)
            if sql == "ROLLBACK":
                raise KeyboardInterrupt("injected rollback cleanup failure")
            return result

    proxy = RollbackFailureProxy()
    try:
        with pytest.raises(core_db.LeaseOwnershipLost) as caught:
            with core_db.transaction(proxy):
                raise primary

        assert caught.value is primary
        assert owner.in_transaction is False
        assert "KeyboardInterrupt" in " ".join(
            getattr(primary, "__notes__", ())
        )
    finally:
        owner.close()
        observer.close()


class _BlockingLLM:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def complete(self, _request: LLMRequest) -> str:
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise AssertionError("test provider was not released")
        return "[]"


def test_forced_takeover_while_provider_blocked_publishes_nothing(
    tmp_path, monkeypatch,
):
    # Keep the periodic timer out of this deterministic forced-takeover test;
    # the post-completion transaction fence is the behavior under test.
    monkeypatch.setattr(runner, "_LOCK_REFRESH_INTERVAL_SECONDS", 3600)
    cfg = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        rules_extraction_enabled=False,
        dream_budget=1,
    )
    llm = _BlockingLLM()
    hy = HyMem(cfg, llm=llm)
    hy.open_session("lease-loss")
    hy.log_message(
        "lease-loss",
        "user",
        "We definitively switched the production dependency from Docker to uv "
        "and system Python; remember this operational decision for all future "
        "deployments because the old container workflow is retired.",
    )
    hy.close_session("lease-loss")
    hy._token_overlap_index = {"stale": ["cached-canonical"]}
    contender = core_db.connect(cfg.db_path)
    caught: list[BaseException] = []

    def dream():
        try:
            hy.dream()
        except BaseException as exc:
            caught.append(exc)

    thread = threading.Thread(target=dream)
    thread.start()
    try:
        assert llm.entered.wait(timeout=5.0)
        row = contender.execute(
            "SELECT holder FROM run_lock WHERE name='dreaming'"
        ).fetchone()
        assert row is not None
        old_token = row["holder"]
        successor = "forced-successor"
        contender.execute(
            "UPDATE run_lock SET holder=?,acquired_at=CURRENT_TIMESTAMP "
            "WHERE name='dreaming' AND holder=?",
            (successor, old_token),
        )
        llm.release.set()
        thread.join(timeout=5.0)

        assert not thread.is_alive()
        assert len(caught) == 1
        assert isinstance(caught[0], DreamLeaseLost)
        assert isinstance(caught[0].__cause__, core_db.LeaseOwnershipLost)
        assert hy._token_overlap_index is None
        assert hy.conn.execute("SELECT * FROM processed_chunks").fetchall() == []
        assert hy.conn.execute("SELECT * FROM knowledge_graph").fetchall() == []
        assert hy.conn.execute("SELECT * FROM behavioral_markers").fetchall() == []
        assert hy.conn.execute("SELECT * FROM episodes").fetchall() == []
        assert hy.conn.execute("SELECT * FROM narrative_facts").fetchall() == []
        lock = contender.execute(
            "SELECT holder FROM run_lock WHERE name='dreaming'"
        ).fetchone()
        assert lock["holder"] == successor
        run = contender.execute(
            "SELECT ended_at,error FROM dream_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run["ended_at"] is not None
        assert run["error"] == "lease_lost"
    finally:
        llm.release.set()
        thread.join(timeout=5.0)
        contender.execute("DELETE FROM run_lock WHERE name='dreaming'")
        contender.close()
        hy.close()


def test_periodic_renewal_prevents_takeover_during_one_blocked_provider_call(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(runner, "_LOCK_TTL_SECONDS", 1)
    monkeypatch.setattr(runner, "_LOCK_REFRESH_INTERVAL_SECONDS", 0.01)
    cfg = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        rules_extraction_enabled=False,
        dream_budget=1,
    )
    llm = _BlockingLLM()
    hy = HyMem(cfg, llm=llm)
    owner_connection = hy.conn
    allow_periodic_renewal = threading.Event()
    lease_backdated = threading.Event()
    renewed_after_backdate = threading.Event()
    real_refresh = runner._refresh_lock

    def observe_refresh(connection, holder):
        # Hold only the dedicated connection at its renewal boundary. The
        # foreground connection must still perform its initial pre-provider
        # refresh so the dream enters the call normally.
        if connection is not owner_connection:
            assert allow_periodic_renewal.wait(timeout=5.0)
        result = real_refresh(connection, holder)
        if connection is not owner_connection and lease_backdated.is_set():
            renewed_after_backdate.set()
        return result

    monkeypatch.setattr(runner, "_refresh_lock", observe_refresh)
    hy.open_session("slow-provider")
    hy.log_message(
        "slow-provider",
        "user",
        "This deliberately long operational decision triggers extraction while "
        "the test provider remains blocked beyond the shortened lease TTL.",
    )
    hy.close_session("slow-provider")
    observer = core_db.connect(cfg.db_path)
    caught: list[BaseException] = []

    def dream():
        try:
            hy.dream()
        except BaseException as exc:
            caught.append(exc)

    thread = threading.Thread(target=dream)
    thread.start()
    try:
        assert llm.entered.wait(timeout=5.0)

        # Deterministically move the blocked call beyond its shortened TTL;
        # relying on a 1.25-second sleep is insufficient because SQLite's
        # CURRENT_TIMESTAMP has one-second precision and staleness uses strict
        # `<`. The gated dedicated heartbeat cannot repair this row until the
        # stale state has first been observed below.
        observer.execute(
            "UPDATE run_lock SET acquired_at=datetime('now','-1 hour') "
            "WHERE name='dreaming'"
        )
        stale = observer.execute(
            "SELECT 1 FROM run_lock WHERE name='dreaming' "
            "AND acquired_at < datetime('now','-1 second')"
        ).fetchone()
        assert stale is not None
        lease_backdated.set()
        allow_periodic_renewal.set()
        assert renewed_after_backdate.wait(timeout=2.0)

        assert thread.is_alive()
        assert runner._acquire_lock(observer, "would-be-successor") is False
        stale = observer.execute(
            "SELECT 1 FROM run_lock WHERE name='dreaming' "
            "AND acquired_at < datetime('now','-1 second')"
        ).fetchone()
        assert stale is None

        llm.release.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert caught == []
        assert observer.execute(
            "SELECT * FROM run_lock WHERE name='dreaming'"
        ).fetchall() == []
    finally:
        llm.release.set()
        thread.join(timeout=5.0)
        observer.execute("DELETE FROM run_lock WHERE name='dreaming'")
        observer.close()
        hy.close()


def test_final_success_report_is_lease_fenced(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_LOCK_REFRESH_INTERVAL_SECONDS", 3600)
    cfg = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        rules_extraction_enabled=False,
        redact_secrets=False,
        vacuum_after_prune=False,
    )
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    _ = hy.conn  # finish schema initialization before intercepting transactions
    contender = core_db.connect(cfg.db_path)
    real_transaction = core_db.transaction
    transaction_count = 0

    @contextlib.contextmanager
    def force_takeover_before_final(connection):
        nonlocal transaction_count
        transaction_count += 1
        # With no sessions/embedding/aggregation, phase 2 and phase 3 are the
        # first two transactions; the third is the final success publication.
        if transaction_count == 3:
            current = contender.execute(
                "SELECT holder FROM run_lock WHERE name='dreaming'"
            ).fetchone()
            assert current is not None
            contender.execute(
                "UPDATE run_lock SET holder='final-successor',"
                "acquired_at=CURRENT_TIMESTAMP WHERE name='dreaming' AND holder=?",
                (current["holder"],),
            )
        with real_transaction(connection) as transaction_connection:
            yield transaction_connection

    monkeypatch.setattr(core_db, "transaction", force_takeover_before_final)
    try:
        with pytest.raises(DreamLeaseLost) as caught:
            hy.dream()
        assert isinstance(caught.value.__cause__, core_db.LeaseOwnershipLost)
        assert transaction_count == 3
        run = contender.execute(
            "SELECT ended_at,error,sessions_processed FROM dream_runs "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run["ended_at"] is not None
        assert run["error"] == "lease_lost"
        assert run["sessions_processed"] == 0
        assert contender.execute(
            "SELECT holder FROM run_lock WHERE name='dreaming'"
        ).fetchone()["holder"] == "final-successor"
    finally:
        contender.execute("DELETE FROM run_lock WHERE name='dreaming'")
        contender.close()
        hy.close()


def test_periodic_heartbeat_uses_a_dedicated_connection(
    tmp_path, monkeypatch,
):
    owner, observer = _open_pair(tmp_path)
    assert runner._acquire_lock(owner, "owner-token") is True
    renewed = threading.Event()
    calls: list[object] = []
    real_refresh = runner._refresh_lock

    def refresh(connection, holder):
        calls.append(connection)
        result = real_refresh(connection, holder)
        renewed.set()
        return result

    monkeypatch.setattr(runner, "_refresh_lock", refresh)
    heartbeat = runner._DreamLeaseHeartbeat(
        owner, "owner-token", interval_seconds=0.01
    )
    heartbeat.start()
    try:
        assert renewed.wait(timeout=2.0)
        heartbeat.check()
        assert calls
        assert all(connection is not owner for connection in calls)
        assert observer.execute(
            "SELECT holder FROM run_lock WHERE name='dreaming'"
        ).fetchone()["holder"] == "owner-token"
    finally:
        heartbeat.stop()
        runner._release_lock(owner, "owner-token")
        owner.close()
        observer.close()


def test_llm_progress_hook_renews_after_each_logical_completion(tmp_path):
    owner, observer = _open_pair(tmp_path)
    try:
        assert runner._acquire_lock(owner, "owner-token") is True

        class BackdatingLLM:
            def complete(self, _request):
                owner.execute(
                    "UPDATE run_lock SET acquired_at=datetime('now','-1 hour') "
                    "WHERE holder='owner-token'"
                )
                return "[]"

        refreshes = 0

        def heartbeat():
            nonlocal refreshes
            runner._refresh_lock(owner, "owner-token")
            refreshes += 1

        wrapped = runner._HeartbeatLLMClient(BackdatingLLM(), heartbeat)
        request = LLMRequest(system="test", user="test", max_tokens=8)
        wrapped.complete(request)
        wrapped.complete(request)

        assert refreshes == 4
        assert observer.execute(
            "SELECT 1 FROM run_lock WHERE name='dreaming' "
            "AND acquired_at < datetime('now','-120 seconds')"
        ).fetchone() is None
    finally:
        runner._release_lock(owner, "owner-token")
        owner.close()
        observer.close()


def test_heartbeat_start_failure_closes_its_dedicated_connection(
    tmp_path, monkeypatch,
):
    owner, observer = _open_pair(tmp_path)
    heartbeat = runner._DreamLeaseHeartbeat(
        owner, "owner-token", interval_seconds=1.0
    )
    dedicated = heartbeat._conn
    assert dedicated is not None

    def fail_start(_thread):
        raise KeyboardInterrupt("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    try:
        with pytest.raises(KeyboardInterrupt, match="thread start failed"):
            heartbeat.start()
        assert heartbeat._conn is None
        with pytest.raises(Exception, match="closed database"):
            dedicated.execute("SELECT 1")
        heartbeat.stop()  # idempotent cleanup after partial start
    finally:
        owner.close()
        observer.close()


def test_cleanup_failure_cannot_mask_primary_or_leak_lease_fence(
    tmp_path, monkeypatch,
):
    cfg = HyMemConfig(
        root=tmp_path,
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        rules_extraction_enabled=False,
        redact_secrets=False,
        vacuum_after_prune=False,
    )
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    primary = RuntimeError("primary dream failure")
    real_stop = runner._DreamLeaseHeartbeat.stop

    def stop_then_fail(self):
        real_stop(self)
        raise KeyboardInterrupt("cleanup failure")

    def fail_phase2(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(runner._DreamLeaseHeartbeat, "stop", stop_then_fail)
    # Leave the exact profile materializer intact: v57 deliberately detects
    # executable drift before allowing it to publish.  Inject the primary
    # failure at the following Phase-2 step so this test continues to exercise
    # the cleanup/masking contract after all producer-integrity fences passed.
    monkeypatch.setattr(runner.phase2, "consolidate_insights", fail_phase2)
    try:
        with pytest.raises(RuntimeError) as caught:
            hy.dream()
        assert caught.value is primary
        assert hy.conn.execute("SELECT * FROM run_lock").fetchall() == []

        # Reset ran despite the cleanup failure: the same execution context can
        # install a fresh fence instead of tripping the nested-fence guard.
        token = core_db.activate_transaction_lease_fence(
            hy.conn, name="dreaming", holder="fresh-token"
        )
        core_db.deactivate_transaction_lease_fence(token)
    finally:
        hy.close()
