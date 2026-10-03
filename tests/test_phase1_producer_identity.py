"""Phase-1 producer identity is a hard cache/publication boundary."""

from __future__ import annotations

import gc
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field, replace

import pytest

from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.core.graph import live_edge_predicate
from hymem.dreaming import phase1, phase2
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.extraction.producer import (
    PHASE1_GENERATION_SCHEMA,
    Phase1ProducerDeclaration,
    canonical_callable_sha256,
    phase1_generation_binding,
    register_phase1_generation,
    validate_current_phase1_generation_binding,
    validate_phase1_generation_binding,
)
from hymem.rules import route_markers_to_rules, suggest_rules_from_markers


@dataclass
class _DeclaredLLM:
    model: str
    object_name: str
    endpoint: str = "https://models.example/v1"
    thinking: str = "disabled"
    effective_body: dict = field(
        default_factory=lambda: {"thinking": {"type": "disabled"}}
    )
    api_key: str = "test-key-never-persisted"
    fail: bool = False
    marker_statement: str | None = None
    calls: list[LLMRequest] = field(default_factory=list)

    def phase1_producer_declaration(self) -> Phase1ProducerDeclaration:
        return Phase1ProducerDeclaration(
            client_id="tests.phase1.DeclaredLLM",
            implementation=canonical_callable_sha256(type(self).complete),
            model=self.model,
            endpoint=self.endpoint,
            effective_request={
                "messages": ["system", "user"],
                "temperature_source": "LLMRequest.temperature",
                "max_tokens_source": "LLMRequest.max_tokens",
                "response_format_source": "LLMRequest.response_format",
                "thinking_mode": self.thinking,
                "effective_body": self.effective_body,
            },
            retry_policy={"owner": "test-client", "attempts": 1},
        )

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if self.fail:
            return "not-json"
        source_ids = [
            int(value)
            for value in re.findall(r'"source_message_id"\s*:\s*(\d+)', request.user)
        ]
        triples = []
        if source_ids:
            triples.append({
                "subject": "app",
                "predicate": "uses",
                "object": self.object_name,
                "polarity": 1,
                "source_message_id": source_ids[-1],
            })
        markers = (
            [{"kind": "style", "statement": self.marker_statement}]
            if self.marker_statement is not None else []
        )
        return json.dumps({
            "triples": triples, "markers": markers, "complete": True,
        })


class _UnknownLLM:
    def complete(self, _request: LLMRequest) -> str:
        return '{"triples":[],"markers":[],"complete":true}'


class _AllStandingJudge:
    def complete(self, request: LLMRequest) -> str:
        markers = json.loads(
            request.user.split("Markers:", 1)[1]
            .rsplit("Return", 1)[0]
            .strip()
        )
        return json.dumps([
            {
                "index": item["index"],
                "standing": True,
                "confidence": 1.0,
                "rule": item["statement"],
            }
            for item in markers
        ])


@dataclass
class _MutatingDeclaredLLM(_DeclaredLLM):
    def complete(self, request: LLMRequest) -> str:
        self.model = f"{self.model}-mutated"
        return super().complete(request)


def _seed_chunk(hy: HyMem, *, chunk_id: str = "producer-chunk") -> Chunk:
    conn = hy.conn
    conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES ('producer-session')")
    message_id = int(conn.execute(
        "INSERT INTO messages(session_id,role,content,created_at) "
        "VALUES ('producer-session','user','App chooses a datastore.',"
        "'2026-09-06T10:00:00.000Z')"
    ).lastrowid)
    chunk = Chunk(
        id=chunk_id,
        session_id="producer-session",
        start_message_id=message_id,
        end_message_id=message_id,
        salience_reason="long_user_turn",
        text="user: App chooses a datastore.",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "producer-session")
        persist_chunks(conn, [chunk])
    return chunk


def _extract_and_persist(
    hy: HyMem, chunk: Chunk, client: _DeclaredLLM,
):
    result = phase1.extract_chunk_results(
        hy.conn, chunk, client, prompt_version=hy.config.prompt_version
    )
    if result is not None:
        with core_db.transaction(hy.conn):
            phase1.persist_chunk_results(
                hy.conn, chunk, result,
                prompt_version=hy.config.prompt_version,
                cfg=hy.config,
            )
    return result


def _current_objects(hy: HyMem) -> set[str]:
    rows = hy.conn.execute(
        f"SELECT object_canonical FROM knowledge_graph "
        f"WHERE {live_edge_predicate()}"
    ).fetchall()
    return {str(row[0]) for row in rows}


def test_canonical_identity_covers_effective_request_but_not_api_key(cfg):
    baseline = _DeclaredLLM("model-a", "redis", api_key="alpha-secret")
    same = _DeclaredLLM("model-a", "redis", api_key="beta-secret")
    different_model = _DeclaredLLM("model-b", "redis")
    different_endpoint = _DeclaredLLM(
        "model-a", "redis", endpoint="https://other.example/v1"
    )
    different_thinking = _DeclaredLLM(
        "model-a", "redis", thinking="off", effective_body={}
    )
    changed_body = _DeclaredLLM(
        "model-a", "redis",
        effective_body={"thinking": {"type": "disabled"}, "mode": "strict"},
    )

    bindings = [
        phase1_generation_binding(cfg.prompt_version, client)
        for client in (
            baseline, same, different_model, different_endpoint,
            different_thinking, changed_body,
        )
    ]
    assert bindings[0] == bindings[1]
    assert len({item["generation_key"] for item in bindings[0:1] + bindings[2:]}) == 5
    encoded = json.dumps(bindings[0], sort_keys=True)
    assert "alpha-secret" not in encoded
    assert "beta-secret" not in encoded
    assert bindings[0]["producer"]["identity_exact"] is True
    assert bindings[0]["producer"]["reuse_scope"] == "durable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint", "https://operator:password@models.example/v1"),
        ("endpoint", "https://models.example/v1?token=credential"),
        ("endpoint", "https://models.example/v1#credential"),
        ("endpoint", "http://models.example/v1"),
        ("endpoint", "https://models.example/v1/sk_live_lowentropy"),
        ("endpoint", "https://models.example/v1/api%252dkey/credential"),
        ("secret_field", {"nested": {"x_auth_token": "credential"}}),
        ("secret_field", {"nested": {"Authorization": "credential"}}),
        ("secret_field", {"nested": {"bearer-token": "credential"}}),
        ("secret_field", {"nested": {"apiKey": "credential"}}),
        ("secret_field", {"nested": {"accessToken": "credential"}}),
        ("secret_field", {"nested": {"clientSecret": "credential"}}),
        ("secret_field", {"nested": {"secretKey": "credential"}}),
        ("secret_field", {"nested": {"sessionCookie": "credential"}}),
        ("secret_field", {"nested": {"bearerToken": "credential"}}),
        ("secret_field", {"nested": {"refreshtoken": "credential"}}),
        ("secret_field", {"nested": {"clientpassword": "credential"}}),
        ("secret_field", {"nested": {"credentialvalue": "credential"}}),
        ("secret_field", {"nested": {"cookievalue": "credential"}}),
        ("secret_field", {"nested": {"authorizationheader": "credential"}}),
        ("secret_field", {"nested": {"authheader": "credential"}}),
        ("secret_field", {"nested": {"sessiontokenvalue": "credential"}}),
        ("secret_field", {"nested": {"passwd": "credential"}}),
        ("secret_field", {"nested": {"privkey": "credential"}}),
        ("secret_field", {"nested": {"credentials": "credential"}}),
        ("secret_field", {"nested": {"cookie": "credential"}}),
        ("secret_field", {"nested": {"session_key": "credential"}}),
        ("secret_field", {"nested": {"key": "credential"}}),
    ],
)
def test_credential_shaped_declarations_are_rejected(cfg, field, value):
    client = _DeclaredLLM("model-a", "redis")
    if field == "endpoint":
        client.endpoint = value
    else:
        client.effective_body = value
    with pytest.raises(ValueError):
        phase1_generation_binding(cfg.prompt_version, client)


def test_stub_is_content_derived_and_arbitrary_custom_clients_fail_closed(cfg):
    stub_a = StubLLMClient(default="[]")
    stub_a_restart = StubLLMClient(default="[]")
    stub_b = StubLLMClient(default="{}")
    assert phase1_generation_binding(cfg.prompt_version, stub_a) == (
        phase1_generation_binding(cfg.prompt_version, stub_a_restart)
    )
    assert phase1_generation_binding(
        cfg.prompt_version, stub_a
    )["generation_key"] != phase1_generation_binding(
        cfg.prompt_version, stub_b
    )["generation_key"]

    unknown_a = _UnknownLLM()
    unknown_b = _UnknownLLM()
    first = phase1_generation_binding(cfg.prompt_version, unknown_a)
    assert first == phase1_generation_binding(cfg.prompt_version, unknown_a)
    second = phase1_generation_binding(cfg.prompt_version, unknown_b)
    assert first["generation_key"] != second["generation_key"]
    assert first["producer"]["identity_exact"] is False
    assert first["producer"]["reuse_scope"] == "process_instance"
    assert first["producer"]["declaration"] is None


def test_stub_identity_binds_first_match_order_and_rejects_dispatch_drift(cfg):
    first = StubLLMClient(fixtures={"a": "A", "ab": "B"})
    second = StubLLMClient(fixtures={"ab": "B", "a": "A"})
    request = LLMRequest(system="", user="ab")
    assert first.complete(request) == "A"
    assert second.complete(request) == "B"
    assert phase1_generation_binding(cfg.prompt_version, first) != (
        phase1_generation_binding(cfg.prompt_version, second)
    )

    baseline = phase1_generation_binding(cfg.prompt_version, first)
    first.complete = lambda _request: "forced"
    with pytest.raises(ValueError, match="integrity"):
        phase1_generation_binding(cfg.prompt_version, first)
    assert baseline["producer"]["identity_exact"] is True


def test_stub_route_aba_cannot_restore_prior_exact_identity(cfg):
    client = StubLLMClient(fixtures={"route": "A"})
    before = phase1_generation_binding(cfg.prompt_version, client)
    client.fixtures["route"] = "B"
    assert client.complete(LLMRequest(system="", user="route")) == "B"
    client.fixtures["route"] = "A"
    after = phase1_generation_binding(cfg.prompt_version, client)
    assert after["generation_key"] != before["generation_key"]


def test_stub_integrity_helper_rebind_cannot_bless_dispatch_shadow(
    cfg, monkeypatch,
):
    from hymem.extraction import llm as llm_module

    client = StubLLMClient(default="A")
    monkeypatch.setattr(llm_module, "maintained_stub_llm_integrity", lambda _: True)
    client.complete = lambda _request: "B"
    with pytest.raises(ValueError, match="integrity"):
        phase1_generation_binding(cfg.prompt_version, client)


@pytest.mark.parametrize(
    "fixtures,default",
    [
        ({"x": 1}, None),
        ({1: "x"}, None),
        ({}, 1),
    ],
)
def test_stub_rejects_non_string_execution_payloads(cfg, fixtures, default):
    with pytest.raises(TypeError, match="exact string"):
        StubLLMClient(fixtures=fixtures, default=default)


def test_supplied_generation_cannot_mislabel_another_client(cfg):
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False))
    try:
        chunk = _seed_chunk(hy)
        client_a = _DeclaredLLM("model-a", "redis")
        client_b = _DeclaredLLM("model-b", "postgres")
        binding_a = phase1_generation_binding(cfg.prompt_version, client_a)
        with pytest.raises(ValueError, match="does not match the effective client"):
            phase1.extract_chunk_results(
                hy.conn,
                chunk,
                client_b,
                prompt_version=cfg.prompt_version,
                phase1_generation=binding_a,
            )
        assert client_b.calls == []
    finally:
        hy.close()


def test_serialized_inexact_binding_dies_with_its_client(cfg):
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False))
    try:
        chunk = _seed_chunk(hy)
        client = _UnknownLLM()
        result = phase1.extract_chunk_results(
            hy.conn, chunk, client, prompt_version=hy.config.prompt_version
        )
        assert result is not None and result.failed is False
        del client
        gc.collect()
        with core_db.transaction(hy.conn), pytest.raises(
            ValueError, match="not authorized by a live client"
        ):
            phase1.persist_chunk_results(
                hy.conn, chunk, result,
                prompt_version=hy.config.prompt_version,
                cfg=hy.config,
            )
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM processed_chunks WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == 0
    finally:
        hy.close()


def test_unknown_client_registry_turnover_is_bounded_and_preserves_refs(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False)
    hy = HyMem(config)
    held_chunk = _seed_chunk(hy, chunk_id="producer-held-inexact")
    held_client = _UnknownLLM()
    assert _extract_and_persist(hy, held_chunk, held_client) is not None
    held_key = hy.conn.execute(
        "SELECT phase1_generation_key FROM kg_claim_extraction_outcomes "
        "WHERE chunk_id=?", (held_chunk.id,),
    ).fetchone()[0]

    turnover = _seed_chunk(hy, chunk_id="producer-turnover-inexact")
    latest_key = None
    for _ in range(24):
        client = _UnknownLLM()
        assert _extract_and_persist(hy, turnover, client) is not None
        latest_key = hy.conn.execute(
            "SELECT phase1_generation_key FROM kg_claim_extraction_outcomes "
            "WHERE chunk_id=?", (turnover.id,),
        ).fetchone()[0]
        # One key per referenced chunk; every superseded process nonce is
        # removed in the same successful reconciliation transaction.
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM phase1_generations"
        ).fetchone()[0] == 2

    orphan_client = _UnknownLLM()
    orphan = phase1_generation_binding(config.prompt_version, orphan_client)
    with core_db.transaction(hy.conn):
        register_phase1_generation(hy.conn, orphan)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        hy.conn.execute(
            "DELETE FROM phase1_generations WHERE generation_key=?",
            (orphan["generation_key"],),
        )
    hy.close()

    reopened = HyMem(config)
    try:
        keys = {
            row[0] for row in reopened.conn.execute(
                "SELECT generation_key FROM phase1_generations"
            )
        }
        assert keys == {held_key, latest_key}
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "first_model,second_model,second_is_cache_hit",
    [
        ("cheap", "cheap", True),
        ("target", "target", True),
        ("cheap", "target", False),
        ("target", "cheap", False),
    ],
)
def test_cheap_target_2x2_never_cross_contaminates_cache(
    cfg, first_model, second_model, second_is_cache_hit,
):
    config = replace(cfg, aggregation_nodes_enabled=False)
    first = _DeclaredLLM(first_model, first_model)
    hy = HyMem(config, llm=first)
    try:
        chunk = _seed_chunk(hy)
        assert _extract_and_persist(hy, chunk, first) is not None
        second = _DeclaredLLM(second_model, second_model)
        hy.set_llm(second)
        result = _extract_and_persist(hy, chunk, second)
        assert (result is None) is second_is_cache_hit
        assert len(second.calls) == (0 if second_is_cache_hit else 2)
        assert _current_objects(hy) == {second_model}
        outcome_key = hy.conn.execute(
            "SELECT phase1_generation_key FROM kg_claim_extraction_outcomes "
            "WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0]
        assert outcome_key == phase1_generation_binding(
            cfg.prompt_version, second
        )["generation_key"]
    finally:
        hy.close()


def test_set_llm_failure_and_transaction_rollback_never_relabel_old_output(cfg):
    client_a = _DeclaredLLM("model-a", "redis")
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False), llm=client_a)
    try:
        chunk = _seed_chunk(hy)
        assert _extract_and_persist(hy, chunk, client_a) is not None
        key_a = phase1_generation_binding(
            cfg.prompt_version, client_a
        )["generation_key"]

        failed_b = _DeclaredLLM("model-b", "postgres", fail=True)
        hy.set_llm(failed_b)
        failed = _extract_and_persist(hy, chunk, failed_b)
        assert failed is not None and failed.failed is True
        status = hy.dream_status()
        assert status["phase1_generation_key"] != key_a
        assert status["pending_chunks"] == 1
        assert _current_objects(hy) == set()
        assert {
            row[0] for row in hy.conn.execute(
                "SELECT object_canonical FROM knowledge_graph WHERE derived=0"
            )
        } == {"redis"}
        assert hy.conn.execute(
            "SELECT phase1_generation_key FROM kg_claim_extraction_outcomes "
            "WHERE chunk_id=?", (chunk.id,),
        ).fetchone()[0] == key_a

        healthy_b = _DeclaredLLM("model-b", "postgres")
        hy.set_llm(healthy_b)
        result = phase1.extract_chunk_results(
            hy.conn, chunk, healthy_b,
            prompt_version=hy.config.prompt_version,
        )
        assert result is not None and result.failed is False
        with pytest.raises(RuntimeError, match="simulated crash"):
            with core_db.transaction(hy.conn):
                phase1.persist_chunk_results(
                    hy.conn, chunk, result,
                    prompt_version=hy.config.prompt_version,
                    cfg=hy.config,
                )
                raise RuntimeError("simulated crash")
        assert _current_objects(hy) == set()
        assert hy.dream_status()["pending_chunks"] == 1

        retry = _extract_and_persist(hy, chunk, healthy_b)
        assert retry is not None and retry.failed is False
        assert _current_objects(hy) == {"postgres"}
        assert hy.dream_status()["pending_chunks"] == 0

        # Returning to A has an exact stable key, but its removed observations
        # are not resurrected as a cache hit.
        replay_a = _DeclaredLLM("model-a", "redis")
        hy.set_llm(replay_a)
        assert hy.dream_status()["pending_chunks"] == 1
        assert _current_objects(hy) == set()
        replay = _extract_and_persist(hy, chunk, replay_a)
        assert replay is not None and replay.failed is False
        assert len(replay_a.calls) == 2
        assert _current_objects(hy) == {"redis"}
        assert hy.dream_status()["pending_chunks"] == 0
        assert hy.conn.execute(
            "SELECT phase1_generation_key FROM kg_claim_extraction_outcomes "
            "WHERE chunk_id=?", (chunk.id,),
        ).fetchone()[0] == key_a
    finally:
        hy.close()


def test_producer_mutation_during_complete_aborts_without_publication(cfg):
    client = _MutatingDeclaredLLM("model-a", "redis")
    hy = HyMem(replace(
        cfg,
        aggregation_nodes_enabled=False,
        rules_extraction_enabled=False,
        facts_extraction_enabled=False,
    ), llm=client)
    try:
        chunk = _seed_chunk(hy)
        with pytest.raises(
            RuntimeError, match="producer identity changed during extraction"
        ):
            hy.dream(session_ids=[chunk.session_id])
        for table in (
            "processed_chunks",
            "kg_claim_extraction_outcomes",
            "kg_claim_observations",
            "behavioral_markers",
        ):
            assert hy.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE chunk_id=?", (chunk.id,)
            ).fetchone()[0] == 0
    finally:
        hy.close()


def test_claim_identical_replay_replaces_omitted_marker(cfg):
    client = _DeclaredLLM(
        "stable-model", "redis", marker_statement="Always use the old format"
    )
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False), llm=client)
    try:
        chunk = _seed_chunk(hy)
        assert _extract_and_persist(hy, chunk, client) is not None
        assert hy.conn.execute(
            "SELECT statement FROM behavioral_markers WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == "Always use the old format"

        # Import recovery and explicit cache invalidation both use this shape:
        # claim evidence remains exact, but a complete Phase-1 replay is due.
        hy.conn.execute(
            "DELETE FROM processed_chunks WHERE chunk_id=?", (chunk.id,)
        )
        client.marker_statement = None
        replay = _extract_and_persist(hy, chunk, client)
        assert replay is not None and replay.failed is False
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM behavioral_markers WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM processed_chunks WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == 1
    finally:
        hy.close()


def test_fresh_marker_consumers_require_current_nonnull_generation(cfg):
    config = replace(
        cfg, aggregation_nodes_enabled=False, rules_extraction_mode="lexical"
    )
    old_client = _DeclaredLLM(
        "model-a", "redis", marker_statement="Always use the old format"
    )
    hy = HyMem(config, llm=old_client)
    try:
        old_chunk = _seed_chunk(hy, chunk_id="producer-old-marker")
        assert _extract_and_persist(hy, old_chunk, old_client) is not None
        old_key = phase1_generation_binding(
            config.prompt_version, old_client
        )["generation_key"]

        current_client = _DeclaredLLM(
            "model-b", "postgres", marker_statement="Always use the new format"
        )
        hy.set_llm(current_client)
        current_chunk = _seed_chunk(hy, chunk_id="producer-current-marker")
        assert _extract_and_persist(hy, current_chunk, current_client) is not None
        current_key = phase1_generation_binding(
            config.prompt_version, current_client
        )["generation_key"]
        assert current_key != old_key
        hy.conn.execute(
            "INSERT INTO behavioral_markers("
            "kind,statement,chunk_id,phase1_generation_key) VALUES (?,?,?,NULL)",
            ("style", "Always use the legacy format", old_chunk.id),
        )

        # A neutral/unscoped helper call cannot choose safely among current,
        # historical, and legacy-NULL producer rows, so every public default
        # fails closed until the caller supplies the selected generation.
        assert suggest_rules_from_markers(
            hy.conn, config, _AllStandingJudge()
        ) == []
        assert route_markers_to_rules(hy.conn, config) == 0
        phase2.consolidate_profile(hy.conn, config)
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM profile_entries"
        ).fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM rules WHERE source='agent_inferred'"
        ).fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM behavioral_markers "
            "WHERE consolidated_at IS NOT NULL"
        ).fetchone()[0] == 0

        candidates = suggest_rules_from_markers(
            hy.conn,
            config,
            _AllStandingJudge(),
            phase1_generation_key=current_key,
        )
        assert [candidate.text for candidate in candidates] == [
            "Always use the new format"
        ]
        assert route_markers_to_rules(
            hy.conn, config, phase1_generation_key=current_key
        ) == 1
        phase2.consolidate_profile(
            hy.conn, config, phase1_generation_key=current_key
        )
        assert {
            row[0] for row in hy.conn.execute("SELECT text FROM profile_entries")
        } == {"Always use the new format"}
        assert {
            row[0] for row in hy.conn.execute(
                "SELECT text FROM rules WHERE source='agent_inferred'"
            )
        } == {"Always use the new format"}
        consolidated = {
            row["statement"]: row["consolidated_at"]
            for row in hy.conn.execute(
                "SELECT statement,consolidated_at FROM behavioral_markers"
            )
        }
        assert consolidated["Always use the new format"] is not None
        assert consolidated["Always use the old format"] is None
        assert consolidated["Always use the legacy format"] is None
        # The generation-scoped suggestion path must honor the same consumed
        # marker boundary as legacy suggestions; otherwise it repeatedly pays
        # for tagger calls and re-proposes already materialized signal.
        assert suggest_rules_from_markers(
            hy.conn,
            config,
            _AllStandingJudge(),
            phase1_generation_key=current_key,
        ) == []
    finally:
        hy.close()


def test_generation_registry_is_canonical_immutable_and_tamper_evident(cfg):
    client = _DeclaredLLM("registry-model", "redis")
    hy = HyMem(cfg, llm=client)
    binding = phase1_generation_binding(cfg.prompt_version, client)
    key = binding["generation_key"]
    try:
        with core_db.transaction(hy.conn):
            register_phase1_generation(hy.conn, binding)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            hy.conn.execute(
                "UPDATE phase1_generations SET reuse_scope='process_instance' "
                "WHERE generation_key=?", (key,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            hy.conn.execute(
                "DELETE FROM phase1_generations WHERE generation_key=?", (key,)
            )
    finally:
        hy.close()

    raw = sqlite3.connect(cfg.db_path)
    try:
        raw.execute("DROP TRIGGER phase1_generations_update_guard")
        raw.execute(
            "UPDATE phase1_generations SET binding_json='{}' "
            "WHERE generation_key=?", (key,),
        )
        raw.commit()
    finally:
        raw.close()
    reopened = HyMem(cfg)
    try:
        with pytest.raises(
            RuntimeError, match="generation registry integrity check failed"
        ):
            _ = reopened.conn
    finally:
        reopened.close()


def test_nested_secret_never_reaches_registry_export_or_status(cfg, tmp_path):
    secret = "PHASE1_PRIVATE_SENTINEL_82741"
    client = _DeclaredLLM(
        "secretless-model", "redis", api_key=f"key-{secret}",
        effective_body={"metadata": {"foo": secret}},
    )
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False), llm=client)
    try:
        chunk = _seed_chunk(hy)
        assert _extract_and_persist(hy, chunk, client) is not None
        database_text = "\n".join(hy.conn.iterdump())
        assert secret not in database_text
        status_text = json.dumps(hy.dream_status(), sort_keys=True)
        assert secret not in status_text
        nested = phase1_generation_binding(
            cfg.prompt_version, client
        )["producer"]["declaration"]["effective_request"]
        assert set(nested) == {"schema", "sha256"}
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", nested["sha256"])
        export_path = tmp_path / "phase1-portable.jsonl"
        hy.export(export_path)
        assert secret not in export_path.read_text(encoding="utf-8")
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM processed_chunks WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == 1
        hy.import_(export_path)
        # An exact self-reimport is a no-op and must not schedule a paid replay.
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM processed_chunks WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == 1
    finally:
        hy.close()


def test_exact_generation_reuses_after_restart_and_legacy_null_is_stale(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False)
    first = _DeclaredLLM("restart-model", "redis")
    hy = HyMem(config, llm=first)
    chunk = _seed_chunk(hy)
    assert _extract_and_persist(hy, chunk, first) is not None
    hy.close()

    restarted_client = _DeclaredLLM("restart-model", "redis")
    restarted = HyMem(config, llm=restarted_client)
    try:
        assert restarted.dream_status()["pending_chunks"] == 0
        assert _extract_and_persist(
            restarted, chunk, restarted_client
        ) is None
        restarted.conn.execute(
            "UPDATE processed_chunks SET phase1_generation_key=NULL "
            "WHERE chunk_id=?", (chunk.id,),
        )
        assert restarted.dream_status()["pending_chunks"] == 1
    finally:
        restarted.close()


def test_historical_contract_row_survives_reopen_but_is_never_current(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False)
    client = _DeclaredLLM("contract-drift-model", "redis")
    hy = HyMem(config, llm=client)
    chunk = _seed_chunk(hy)
    current = phase1_generation_binding(config.prompt_version, client)
    historical = json.loads(json.dumps(current))
    historical["extraction_cache_key"] = (
        "hymem-extraction-cache-v1:" + "0" * 64 + ":v20"
    )
    payload = {
        "schema": historical["schema"],
        "extraction_cache_key": historical["extraction_cache_key"],
        "producer": historical["producer"],
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    historical["generation_key"] = (
        PHASE1_GENERATION_SCHEMA + ":" + hashlib.sha256(encoded).hexdigest()
    )
    assert validate_phase1_generation_binding(historical) == historical
    with pytest.raises(ValueError, match="another extraction contract"):
        validate_current_phase1_generation_binding(
            historical, prompt_version=config.prompt_version
        )

    with core_db.transaction(hy.conn):
        register_phase1_generation(hy.conn, historical)
        with core_db.evidence_mutation(hy.conn):
            hy.conn.execute(
                "INSERT INTO kg_claim_extraction_outcomes("
                "chunk_id,prompt_version,prompt_generation,result_hash,"
                "phase1_generation_key) VALUES (?,?,?,?,?)",
                (
                    chunk.id, historical["extraction_cache_key"], 20,
                    "sha256:" + "0" * 64, historical["generation_key"],
                ),
            )
        hy.conn.execute(
            "INSERT INTO processed_chunks("
            "chunk_id,prompt_version,phase1_generation_key) VALUES (?,?,?)",
            (
                chunk.id, historical["extraction_cache_key"],
                historical["generation_key"],
            ),
        )
    hy.close()

    restarted_client = _DeclaredLLM("contract-drift-model", "redis")
    reopened = HyMem(config, llm=restarted_client)
    try:
        assert reopened.conn.execute(
            "SELECT COUNT(*) FROM phase1_generations WHERE generation_key=?",
            (historical["generation_key"],),
        ).fetchone()[0] == 1
        assert reopened.dream_status()["pending_chunks"] == 1
        extracted = phase1.extract_chunk_results(
            reopened.conn, chunk, restarted_client,
            prompt_version=config.prompt_version,
        )
        assert extracted is not None and extracted.failed is False
        assert len(restarted_client.calls) == 2
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("second_model", "replacement_expected"),
    [("model-b", True), ("model-a", False)],
)
def test_prompt_monotonicity_is_scoped_to_producer_identity(
    cfg, second_model, replacement_expected,
):
    high_config = replace(
        cfg, prompt_version="v999", aggregation_nodes_enabled=False
    )
    producer_a = _DeclaredLLM("model-a", "redis")
    first = HyMem(high_config, llm=producer_a)
    chunk = _seed_chunk(first)
    assert _extract_and_persist(first, chunk, producer_a) is not None
    high_key = first.conn.execute(
        "SELECT phase1_generation_key FROM kg_claim_extraction_outcomes "
        "WHERE chunk_id=?", (chunk.id,),
    ).fetchone()[0]
    first.close()

    low_config = replace(
        cfg, prompt_version="v21", aggregation_nodes_enabled=False
    )
    producer_b = _DeclaredLLM(second_model, "postgres")
    second = HyMem(low_config, llm=producer_b)
    try:
        if replacement_expected:
            attempted = _extract_and_persist(second, chunk, producer_b)
            assert attempted is not None and attempted.failed is False
            assert len(producer_b.calls) == 2
        else:
            with pytest.raises(
                phase1.Phase1ProducerGenerationConflictError,
                match="newer prompt generation",
            ):
                second.dream(session_ids=[chunk.session_id])
            assert producer_b.calls == []
        outcome_key = second.conn.execute(
            "SELECT phase1_generation_key FROM kg_claim_extraction_outcomes "
            "WHERE chunk_id=?", (chunk.id,),
        ).fetchone()[0]
        if replacement_expected:
            assert outcome_key != high_key
            assert _current_objects(second) == {"postgres"}
            assert second.dream_status()["pending_chunks"] == 0
            # The successful target publication owns the cache gate; the next
            # pass neither pays again nor duplicates evidence.
            assert _extract_and_persist(second, chunk, producer_b) is None
            assert len(producer_b.calls) == 2
            assert second.conn.execute(
                "SELECT COUNT(*) FROM kg_claim_observations WHERE chunk_id=?",
                (chunk.id,),
            ).fetchone()[0] == 1
        else:
            assert outcome_key == high_key
            assert _current_objects(second) == set()
            assert second.dream_status()["pending_chunks"] >= 1
    finally:
        second.close()
