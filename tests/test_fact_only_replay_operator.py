"""Safety checks for the bounded, fact-only repair operator."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import aggregation_material, facts, runner
from hymem.dreaming import lossless
from hymem.extraction.llm import StubLLMClient
from tools.diagnostics import fact_only_replay as operator


def _process(root, pid, argv, values):
    proc = root / str(pid)
    proc.mkdir()
    (proc / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")
    (proc / "environ").write_bytes(
        b"\0".join(key.encode() + b"=" + value.encode()
                   for key, value in values.items()) + b"\0"
    )


def _server_env(cap="16", tokens=None):
    values = {
        "HYMEM_LLM_API_KEY": "fixture-secret",
        "HYMEM_EMBEDDING_API_KEY": "fixture-embedding-secret",
        "HYMEM_EMBEDDING_BASE_URL": "http://embedding-server:8766/v1",
        "HYMEM_DREAM_MAX_FACTS_PER_SESSION": cap,
        "HYMEM_ROOT": "/home/node/.hermes",
        "UNRELATED_SECRET": "must-not-copy",
    }
    if tokens is not None:
        values["HYMEM_FACTS_EXTRACTION_MAX_TOKENS"] = tokens
    return values


def test_server_env_loader_rejects_unrelated_process_and_copies_only_allowed(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HYMEM_LLM_API_KEY", "stale-inherited")
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    _process(tmp_path, 1, ["python", "unrelated"], _server_env("32"))
    _process(tmp_path, 2, ["/home/node/hymem-env/bin/hymem-honcho"], _server_env())
    _process(tmp_path, 3, ["/home/node/hymem-env/bin/hymem-server"], _server_env())
    operator._load_server_process_env(tmp_path)
    assert os.environ["HYMEM_LLM_API_KEY"] == "fixture-secret"
    assert os.environ["HYMEM_DREAM_MAX_FACTS_PER_SESSION"] == "16"
    assert "UNRELATED_SECRET" not in os.environ


def test_server_env_loader_requires_matching_live_producer_envs(tmp_path, monkeypatch):
    monkeypatch.setenv("HYMEM_LLM_API_KEY", "inherited")
    _process(tmp_path, 2, ["/home/node/hymem-env/bin/hymem-honcho"], _server_env())
    _process(tmp_path, 3, ["/home/node/hymem-env/bin/hymem-server"], _server_env("32"))
    with pytest.raises(operator.RepairStopped, match="matching running Honcho/MCP"):
        operator._load_server_process_env(tmp_path)
    assert os.environ["HYMEM_LLM_API_KEY"] == "inherited"


def test_server_env_loader_rejects_duplicate_same_role_even_with_same_env(tmp_path):
    values = _server_env("64", "6144")
    _process(tmp_path, 2, ["/home/node/hymem-env/bin/hymem-honcho"], values)
    _process(tmp_path, 3, ["/home/node/hymem-env/bin/hymem-honcho"], values)
    _process(tmp_path, 4, ["/home/node/hymem-env/bin/hymem-server"], values)
    with pytest.raises(operator.RepairStopped, match="matching running Honcho/MCP"):
        operator._load_server_process_env(tmp_path)


def test_server_env_loader_rejects_missing_or_wrong_embedding_route(tmp_path):
    wrong = _server_env()
    wrong["HYMEM_EMBEDDING_BASE_URL"] = "http://localhost:8766/v1"
    _process(tmp_path, 2, ["/home/node/hymem-env/bin/hymem-honcho"], wrong)
    with pytest.raises(operator.RepairStopped, match="matching running Honcho/MCP"):
        operator._load_server_process_env(tmp_path)


def test_server_env_loader_copies_matching_fact_token_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HYMEM_FACTS_EXTRACTION_MAX_TOKENS", "stale")
    values = _server_env("64", "6144")
    _process(tmp_path, 2, ["/home/node/hymem-env/bin/hymem-honcho"], values)
    _process(tmp_path, 3, ["/home/node/hymem-env/bin/hymem-server"], values)
    operator._load_server_process_env(tmp_path)
    assert os.environ["HYMEM_DREAM_MAX_FACTS_PER_SESSION"] == "64"
    assert os.environ["HYMEM_FACTS_EXTRACTION_MAX_TOKENS"] == "6144"


def test_fact_replay_configuration_accepts_only_reviewed_pair(tmp_path):
    cfg = operator._configuration(tmp_path / "hymem.sqlite", 64, 6144)
    assert (cfg.dream_max_facts_per_session, cfg.facts_extraction_max_tokens) == (
        64, 6144,
    )
    with pytest.raises(operator.RepairStopped, match="reviewed"):
        operator._configuration(tmp_path / "hymem.sqlite", 64, 3072)


def test_new_fact_target_accepts_only_old_3072_token_v4_field():
    target = "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=6144|items=64"
    allowed = operator._allowed_v4_fields(target)
    assert allowed == {
        "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=6144",
        "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=3072",
    }
    assert "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=8192" not in allowed
    for legacy in operator.REVIEWED_3072_FACT_VERSIONS:
        assert operator._v4_version_allowed(legacy, target)
    assert operator._v4_version_allowed(target, target)
    assert not operator._v4_version_allowed(
        "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=3072|items=256",
        target,
    )


def test_server_env_loader_requires_both_process_roles(tmp_path):
    _process(tmp_path, 2, ["/home/node/hymem-env/bin/hymem-honcho"], _server_env())
    with pytest.raises(operator.RepairStopped, match="matching running Honcho/MCP"):
        operator._load_server_process_env(tmp_path)


def test_live_store_guard_includes_hardlink_alias(tmp_path, monkeypatch):
    live = tmp_path / "live.sqlite"
    live.write_bytes(b"fixture")
    alias = tmp_path / "alias.sqlite"
    os.link(live, alias)
    another = tmp_path / "another.sqlite"
    another.write_bytes(b"fixture")
    monkeypatch.setattr(operator, "LIVE_CONTAINER_DB", live)
    monkeypatch.setattr(operator, "LIVE_HOST_DB", tmp_path / "missing.sqlite")
    assert operator._is_live_store(alias)
    assert not operator._is_live_store(another)


def _held_fixture(tmp_path):
    store = HyMem(
        HyMemConfig(root=Path(tmp_path)), llm=StubLLMClient(default="[]"),
    )
    store.open_session("held")
    store.log_message("held", "user", "Fixture source A")
    with core_db.transaction(store.conn):
        lossless.materialize_message_coverage(store.conn, "held")
        store.conn.execute(
            "UPDATE sessions SET facts_retry_count=1 WHERE id='held'"
        )
    return store


def _apply_fixture(tmp_path, monkeypatch, *, llm=None, retry_count=1):
    store = _held_fixture(tmp_path)
    cfg = HyMemConfig(root=Path(tmp_path))
    llm = llm or StubLLMClient(default="[]")
    model = "hymem-embedding-producer-v1:" + "0" * 64
    with core_db.transaction(store.conn):
        store.conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('vec_model',?)",
            (model,),
        )
        store.conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('vec_dim','384')"
        )
        store.conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_facts(rowid INTEGER PRIMARY KEY)"
        )
        store.conn.execute(
            "UPDATE sessions SET facts_retry_count=? WHERE id='held'",
            (retry_count,),
        )
    monkeypatch.setattr(aggregation_material, "embedding_storage_identity",
                        lambda _client: (model, 384))
    monkeypatch.setattr(core_db, "_load_vec_extension", lambda _conn: True)
    target = facts.facts_config_version(cfg, client=llm)
    state = operator._state(store.conn, target)
    backup_parent = Path(tmp_path) / "private-backup"
    backup_parent.mkdir(mode=0o700)
    kwargs = dict(
        expected_state=state["state_sha256"], expected_target=target,
        expected_outcomes=state["outcomes"], expected_sessions=state["sessions"],
        expected_stale=state["stale"], backup_path=backup_parent / "backup.sqlite",
        max_logical=1 if retry_count == 2 else 0,
        max_http=3 if retry_count == 2 else 0,
        max_embed_batches=0, max_embed_texts=0,
        include_fresh=retry_count == 2,
    )
    return store, cfg, llm, kwargs


def test_preview_fingerprint_refuses_covered_append(tmp_path):
    store = _held_fixture(tmp_path)
    try:
        before = operator._state(store.conn, "fixture-target")
        store.log_message("held", "assistant", "Fixture source B")
        with core_db.transaction(store.conn):
            lossless.materialize_message_coverage(store.conn, "held")
        after = operator._state(store.conn, "fixture-target")
        assert before["pending_fresh"] == after["pending_fresh"] == 1
        assert before["state_sha256"] != after["state_sha256"]
    finally:
        store.close()


def test_preview_fingerprint_binds_source_bytes_without_cursor_change(
    tmp_path, monkeypatch,
):
    store = _held_fixture(tmp_path)
    try:
        before = operator._state(store.conn, "fixture-target")
        original = lossless.covered_messages_after

        def changed_source(*args, **kwargs):
            source = original(*args, **kwargs)
            return [replace(source[0], content="Fixture changed source")]

        monkeypatch.setattr(lossless, "covered_messages_after", changed_source)
        after = operator._state(store.conn, "fixture-target")
        assert before["state_sha256"] != after["state_sha256"]
    finally:
        store.close()


def test_heartbeat_constructor_failure_releases_acquired_dream_lease(
    tmp_path, monkeypatch,
):
    store, cfg, llm, kwargs = _apply_fixture(tmp_path, monkeypatch)
    try:
        observed = []

        def failed_heartbeat(*_args, **_kwargs):
            observed.append(True)
            raise OSError("injected heartbeat constructor failure")

        monkeypatch.setattr(runner, "_DreamLeaseHeartbeat", failed_heartbeat)
        with pytest.raises(OSError, match="injected heartbeat"):
            operator.apply(
                store.conn, Path(tmp_path) / "hymem.sqlite", cfg, llm, object(),
                **kwargs,
            )
        assert observed == [True]
        assert store.conn.execute(
            "SELECT 1 FROM run_lock WHERE name='dreaming'"
        ).fetchone() is None
    finally:
        store.close()


def test_fence_deactivation_failure_still_releases_dream_lease(
    tmp_path, monkeypatch,
):
    store, cfg, llm, kwargs = _apply_fixture(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(core_db, "activate_transaction_lease_fence",
                            lambda *_a, **_k: object())
        monkeypatch.setattr(core_db, "deactivate_transaction_lease_fence",
                            lambda _fence: (_ for _ in ()).throw(
                                OSError("injected fence cleanup failure")
                            ))
        monkeypatch.setattr(operator, "_backup_under_lease",
                            lambda *_a: (_ for _ in ()).throw(
                                operator.RepairStopped("injected pre-replay failure")
                            ))
        with pytest.raises(OSError, match="injected fence cleanup"):
            operator.apply(
                store.conn, Path(tmp_path) / "hymem.sqlite", cfg, llm, object(),
                **kwargs,
            )
        assert store.conn.execute(
            "SELECT 1 FROM run_lock WHERE name='dreaming'"
        ).fetchone() is None
    finally:
        store.close()


def test_fresh_provider_append_cannot_publish_unreviewed_tail(
    tmp_path, monkeypatch,
):
    seen = []

    class AppendingLLM:
        request_attempts = 0

        def complete(self, request):
            self.request_attempts += 1
            seen.append(request.user)
            store.log_message("held", "assistant", "New unreviewed covered turn")
            with core_db.transaction(store.conn):
                lossless.materialize_message_coverage(store.conn, "held")
            return '[]'

    llm = AppendingLLM()
    store, cfg, llm, kwargs = _apply_fixture(
        tmp_path, monkeypatch, llm=llm, retry_count=2,
    )
    try:
        with pytest.raises(operator.RepairStopped,
                           match="held fresh source changed before publication"):
            operator.apply(
                store.conn, Path(tmp_path) / "hymem.sqlite", cfg, llm, object(),
                **kwargs,
            )
        assert len(seen) == 1
        assert "Fixture source A" in seen[0]
        assert "New unreviewed covered turn" not in seen[0]
        assert store.conn.execute(
            "SELECT 1 FROM fact_extraction_outcomes"
        ).fetchone() is None
        assert store.conn.execute(
            "SELECT 1 FROM run_lock WHERE name='dreaming'"
        ).fetchone() is None
    finally:
        store.close()
