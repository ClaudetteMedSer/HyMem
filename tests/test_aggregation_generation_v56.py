from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
import pytest

from benchmarks.store_attestation import material_store_state
from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming.aggregate import (
    aggregation_config_version,
    build_aggregation_nodes,
    load_digest,
)
from hymem.dreaming.aggregation_generation import (
    AGGREGATION_GENERATION_SCHEMA,
    aggregation_generation_binding,
    register_aggregation_generation,
    validate_aggregation_generation_binding,
)
from hymem.dreaming.aggregation_provenance import (
    aggregation_llm_request,
    aggregation_llm_request_hash,
    aggregation_output_hash,
    load_current_aggregation_publication,
)
from hymem.dreaming.aggregation_health import (
    begin_aggregation_build,
    complete_aggregation_build,
)
from hymem.extraction.llm import LLMRequest, StubLLMClient
from hymem.extraction.producer import (
    Phase1ProducerDeclaration,
    phase1_producer_binding,
    validate_phase1_producer_binding,
)
from hymem.extraction.producer import _register_phase1_producer_proxy
from hymem.deadline import DeadlineBoundLLMClient, MonotonicDeadline
from hymem.query.augment import augment as query_augment
from tests.test_aggregation_provenance import (
    _aggregation_cfg,
    _seed_native_episode,
)


def _producer(label: str) -> StubLLMClient:
    response = json.dumps({"title": label, "summary": f"{label} material"})
    return StubLLMClient(
        fixtures={
            "fuse several related episodes": response,
            "combined summary that loses no thread": response,
            "standing digest of everything known": response,
        },
        default="[]",
    )


def _seed(conn) -> None:
    _seed_native_episode(
        conn, "generation-a", title="One", summary="shared one",
        entity="shared-thread",
    )
    _seed_native_episode(
        conn, "generation-b", title="Two", summary="shared two",
        entity="shared-thread",
    )


def test_core_generation_identity_uses_frozen_loaded_hash_helpers(
    cfg, monkeypatch,
):
    """Identity lookup never follows mutable public hashing helper names."""

    from hymem.dreaming import aggregation_generation as generation_module
    from hymem.extraction import producer as producer_module

    enabled = _aggregation_cfg(cfg)
    client = _producer("loaded-code")
    before = aggregation_generation_binding(enabled, client)

    def replaced_helper(*_args, **_kwargs):
        raise AssertionError("runtime identity followed a replaced helper")

    monkeypatch.setattr(
        generation_module, "canonical_callable_sha256", replaced_helper,
    )
    monkeypatch.setattr(
        generation_module, "canonical_module_sha256", replaced_helper,
    )
    monkeypatch.setattr(
        producer_module, "canonical_callable_sha256", replaced_helper,
    )
    monkeypatch.setattr(
        producer_module, "canonical_module_sha256", replaced_helper,
    )
    assert aggregation_generation_binding(enabled, client) == before


def _acknowledged_build(hy: HyMem, cfg, client) -> object:
    binding = aggregation_generation_binding(cfg, client)
    version = aggregation_config_version(cfg)
    with core_db.transaction(hy.conn):
        attempt_token = begin_aggregation_build(
            hy.conn, version, generation_binding=binding,
        )
    result = build_aggregation_nodes(
        hy.conn, cfg, client, generation_binding=binding,
        health_managed=True, health_attempt_token=attempt_token,
    )
    assert result.fusion_failures == 0
    with core_db.transaction(hy.conn):
        complete_aggregation_build(
            hy.conn, version, str(binding["generation_key"]),
            attempt_token,
            expected_node_count=result.nodes,
        )
    return result


def test_exact_a_b_lifecycle_and_no_llm_reader(cfg):
    enabled = _aggregation_cfg(cfg, digest=True)
    a = _producer("producer A")
    hy = HyMem(enabled, llm=a)
    try:
        _seed(hy.conn)
        first = _acknowledged_build(hy, enabled, a)
        assert a.calls
        a_equivalent = _producer("producer A")
        second = _acknowledged_build(hy, enabled, a_equivalent)
        assert second.reused == first.nodes
        assert a_equivalent.calls == []

        old_id = hy.digest().node_id
        b = _producer("producer B")
        hy.set_llm(b)
        assert hy.digest() is None
        assert hy.expand_node(old_id) is None
        assert hy.dream_status()["pending_aggregation"] == 1

        b_result = _acknowledged_build(hy, enabled, b)
        assert b_result.reused == 0
        assert b_result.keying_residual == 0
        assert b.calls
        assert hy.digest().title == "producer B"
        assert hy.digest().node_id != old_id

        a_again = _producer("producer A")
        hy.set_llm(a_again)
        assert hy.digest() is None
        _acknowledged_build(hy, enabled, a_again)
        assert a_again.calls
        assert hy.digest().title == "producer A"
    finally:
        hy.close()

    reader = HyMem(enabled)
    try:
        digest = reader.digest()
        assert digest is not None and digest.title == "producer A"
        status = reader.dream_status()
        assert status["pending_aggregation"] == 0
        assert status["aggregation_generation_key"] is None
        assert status["aggregation_publication_generation_key"]
        assert status["aggregation_publication_generation"]["producer"]
    finally:
        reader.close()


class _UnknownClient:
    def __init__(self, label: str = "unknown") -> None:
        self.label = label
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        return json.dumps({"title": self.label, "summary": self.label})


def test_unknown_client_is_same_object_only(cfg):
    enabled = _aggregation_cfg(cfg, digest=False)
    hy = HyMem(enabled)
    try:
        _seed(hy.conn)
        one = _UnknownClient()
        first = build_aggregation_nodes(hy.conn, enabled, one)
        assert first.nodes == 1 and len(one.calls) == 1
        one.calls.clear()
        second = build_aggregation_nodes(hy.conn, enabled, one)
        assert second.reused == 1 and one.calls == []

        other = _UnknownClient()
        third = build_aggregation_nodes(hy.conn, enabled, other)
        assert third.reused == 0 and len(other.calls) == 1
    finally:
        hy.close()


class _Phase1OnlyRouter(_UnknownClient):
    def phase1_producer_declaration(self):
        return Phase1ProducerDeclaration(
            client_id="tests.phase1-router",
            implementation="sha256:" + "1" * 64,
            model="phase1-only",
            endpoint=None,
            effective_request={"format": "json"},
            retry_policy={"attempts": 1},
        )


def test_phase1_only_router_cannot_claim_durable_aggregation(cfg):
    binding = aggregation_generation_binding(_aggregation_cfg(cfg), _Phase1OnlyRouter())
    assert binding["producer"]["identity_exact"] is False
    assert binding["producer"]["reuse_scope"] == "process_instance"


class _DeclaredClient(_UnknownClient):
    def __init__(self, model: str, *, endpoint: str = "https://models.example/v1"):
        super().__init__(model)
        self.model = model
        self.endpoint = endpoint
        self.request_mode = "json"

    def aggregation_producer_declaration(self):
        return Phase1ProducerDeclaration(
            client_id="tests.aggregation-client",
            implementation="sha256:" + "2" * 64,
            model=self.model,
            endpoint=self.endpoint,
            effective_request={"format": self.request_mode},
            retry_policy={"attempts": 1},
        )


class _SameOutputDeclared(_DeclaredClient):
    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        return json.dumps({"title": "same", "summary": "same bytes"})


def test_identical_output_from_different_models_never_reuses(cfg):
    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        _seed(hy.conn)
        a = _SameOutputDeclared("model-a")
        build_aggregation_nodes(hy.conn, enabled, a)
        first_id = hy.conn.execute(
            "SELECT id FROM aggregation_nodes"
        ).fetchone()[0]
        b = _SameOutputDeclared("model-b")
        result = build_aggregation_nodes(hy.conn, enabled, b)
        second_id = hy.conn.execute(
            "SELECT id FROM aggregation_nodes"
        ).fetchone()[0]
        assert result.reused == 0 and b.calls
        assert first_id != second_id
    finally:
        hy.close()


class _TransientFailure(_DeclaredClient):
    def __init__(self, model: str):
        super().__init__(model)
        self.fail = True

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("ordinary temporary failure")
        return json.dumps({"title": self.model, "summary": self.model})


def test_failed_new_generation_withdraws_old_publication(cfg):
    enabled = _aggregation_cfg(cfg, digest=True)
    a = _producer("producer A")
    hy = HyMem(enabled, llm=a)
    try:
        _seed(hy.conn)
        _acknowledged_build(hy, enabled, a)
        old_rows = hy.conn.execute(
            "SELECT COUNT(*) FROM aggregation_nodes"
        ).fetchone()[0]
        b = _TransientFailure("model-b")
        hy.set_llm(b)
        result = build_aggregation_nodes(hy.conn, enabled, b)
        assert result.fusion_failures > 0
        assert hy.digest() is None
        assert hy.conn.execute(
            "SELECT 1 FROM aggregation_publication_state"
        ).fetchone() is None
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM aggregation_nodes"
        ).fetchone()[0] == old_rows

        b.fail = False
        healed = build_aggregation_nodes(hy.conn, enabled, b)
        assert healed.fusion_failures == 0
        assert hy.digest().title == "model-b"
    finally:
        hy.close()


def test_mutable_live_declaration_hides_old_publication(cfg):
    enabled = _aggregation_cfg(cfg, digest=True)
    client = _DeclaredClient("model-a")
    hy = HyMem(enabled, llm=client)
    try:
        _seed(hy.conn)
        _acknowledged_build(hy, enabled, client)
        assert hy.digest() is not None
        client.model = "model-b"
        assert hy.digest() is None
        assert hy.dream_status()["pending_aggregation"] == 1
    finally:
        hy.close()


def test_declared_model_endpoint_and_request_settings_change_identity(cfg):
    enabled = _aggregation_cfg(cfg)
    client = _DeclaredClient("model-a")
    baseline = aggregation_generation_binding(enabled, client)["generation_key"]
    client.model = "model-b"
    model_key = aggregation_generation_binding(enabled, client)["generation_key"]
    client.model = "model-a"
    client.endpoint = "https://models.example/v2"
    endpoint_key = aggregation_generation_binding(enabled, client)["generation_key"]
    client.endpoint = "https://models.example/v1"
    client.request_mode = "strict-json"
    request_key = aggregation_generation_binding(enabled, client)["generation_key"]
    assert len({baseline, model_key, endpoint_key, request_key}) == 4


def test_prompt_and_parser_code_changes_change_identity(cfg, monkeypatch):
    from hymem.dreaming import aggregate as aggregate_mod

    enabled = _aggregation_cfg(cfg)
    client = _producer("stable")
    baseline = aggregation_generation_binding(enabled, client)["generation_key"]
    monkeypatch.setattr(
        aggregate_mod, "AGGREGATE_SYSTEM",
        aggregate_mod.AGGREGATE_SYSTEM + "\nExact revision.",
    )
    prompt_key = aggregation_generation_binding(enabled, client)["generation_key"]

    def revised_parser(value, *, expect):
        return None

    monkeypatch.setattr(aggregate_mod, "loads_lenient", revised_parser)
    parser_key = aggregation_generation_binding(enabled, client)["generation_key"]
    assert len({baseline, prompt_key, parser_key}) == 3


class _MutatesDuringCall(_DeclaredClient):
    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        self.model = "model-after-call"
        return json.dumps({"title": "late", "summary": "late"})


def test_identity_is_rechecked_after_completion_before_publication(cfg):
    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        _seed(hy.conn)
        client = _MutatesDuringCall("model-before-call")
        result = build_aggregation_nodes(hy.conn, enabled, client)
        assert result.fusion_failures == 1
        assert hy.conn.execute(
            "SELECT 1 FROM aggregation_publication_state"
        ).fetchone() is None
    finally:
        hy.close()


def test_late_identity_change_rolls_back_candidate_and_embeddings(
    cfg, monkeypatch,
):
    from hymem.dreaming import aggregate as aggregate_mod
    from hymem import StubEmbeddingClient

    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        _seed(hy.conn)
        client = _DeclaredClient("model-before-persist")
        original = aggregate_mod.persist_aggregation_source_manifest
        changed = False

        def mutate_after_proof(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed:
                changed = True
                client.model = "model-after-persist"
            return result

        monkeypatch.setattr(
            aggregate_mod, "persist_aggregation_source_manifest",
            mutate_after_proof,
        )
        with pytest.raises(RuntimeError, match="producer identity changed"):
            build_aggregation_nodes(
                hy.conn, enabled, client, StubEmbeddingClient(),
            )
        assert hy.conn.execute(
            "SELECT 1 FROM aggregation_publication_state"
        ).fetchone() is None
        assert hy.conn.execute(
            "SELECT 1 FROM aggregation_node_embeddings"
        ).fetchone() is None
    finally:
        hy.close()


def test_endpoint_route_is_exact_but_only_origin_and_digest_are_persisted(cfg):
    enabled = _aggregation_cfg(cfg)
    private_route = "unpredictable-private-route-73d91"
    private = aggregation_generation_binding(
        enabled,
        _DeclaredClient(
            "m", endpoint=f"https://models.example/v1/{private_route}",
        ),
    )
    one = aggregation_generation_binding(
        enabled, _DeclaredClient("m", endpoint="https://models.example/v1/alpha"),
    )
    two = aggregation_generation_binding(
        enabled, _DeclaredClient("m", endpoint="https://models.example/v1/other"),
    )
    assert one["generation_key"] != two["generation_key"]
    assert private_route not in json.dumps(private, sort_keys=True)
    declaration = private["producer"]["declaration"]
    assert declaration["endpoint_origin"] == "https://models.example"
    assert declaration["endpoint_sha256"].startswith("sha256:")
    hy = HyMem(enabled)
    try:
        with core_db.transaction(hy.conn):
            register_aggregation_generation(hy.conn, private)
        stored = hy.conn.execute(
            "SELECT binding_json FROM aggregation_generations"
        ).fetchone()[0]
        assert private_route not in stored
    finally:
        hy.close()


class _HistoricalEndpointClient(_DeclaredClient):
    def phase1_producer_declaration(self):
        return self.aggregation_producer_declaration()


def _aggregation_binding_with_producer(cfg, producer):
    contract = aggregation_generation_binding(
        cfg, _DeclaredClient("template"),
    )["contract"]
    payload = {
        "schema": AGGREGATION_GENERATION_SCHEMA,
        "contract": contract,
        "producer": producer,
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "generation_key": AGGREGATION_GENERATION_SCHEMA + ":"
        + hashlib.sha256(encoded).hexdigest(),
    }


def test_phase1_and_aggregation_endpoint_bindings_are_secret_free(cfg):
    safe = _HistoricalEndpointClient(
        "model", endpoint="https://models.example/deployments/stable/v1",
    )
    phase1 = phase1_producer_binding(safe)
    assert phase1["declaration"]["endpoint_origin"] == "https://models.example"
    assert phase1["declaration"]["endpoint_sha256"].startswith("sha256:")
    assert safe.endpoint not in json.dumps(phase1)
    assert validate_phase1_producer_binding(phase1) == phase1

    sensitive = _HistoricalEndpointClient(
        "model", endpoint="https://models.example/v1/sk-private-value",
    )
    with pytest.raises(ValueError, match="credential-shaped"):
        phase1_producer_binding(sensitive)


def test_historical_sensitive_aggregation_endpoint_fails_registry_and_reopen(cfg):
    enabled = _aggregation_cfg(cfg)
    sensitive = _HistoricalEndpointClient(
        "model", endpoint="https://models.example/v1/sk-private-value",
    )
    # Model a historical v2 declaration directly.  Current Phase-1 producers
    # reject this route before they can emit a binding, so using the live
    # helper here would no longer reach the registry corruption boundary this
    # regression is intended to exercise.
    legacy_declaration = {
        "schema": "hymem-phase1-producer-declaration-v2",
        "client_id": "tests.aggregation-client",
        "implementation": "sha256:" + "2" * 64,
        "model": "model",
        "endpoint": sensitive.endpoint,
        "effective_request": {"format": "json"},
        "retry_policy": {"attempts": 1},
    }
    legacy_producer = {
        "schema": "hymem-phase1-producer-binding-v1",
        "identity_exact": True,
        "reuse_scope": "durable",
        "declaration": legacy_declaration,
        "identity_sha256": "sha256:" + "9" * 64,
    }
    binding = _aggregation_binding_with_producer(
        enabled, legacy_producer,
    )
    hy = HyMem(cfg)
    with pytest.raises(ValueError, match="declaration"):
        register_aggregation_generation(hy.conn, binding)
    hy.conn.execute("DROP TRIGGER aggregation_generations_insert_guard")
    producer = binding["producer"]
    hy.conn.execute(
        "INSERT INTO aggregation_generations VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        (
            binding["generation_key"],
            binding["contract"]["material_config_version"],
            producer["identity_sha256"], 1, "durable",
            json.dumps(binding, sort_keys=True, separators=(",", ":")),
        ),
    )
    hy.close()
    reopened = core_db.connect(cfg.db_path)
    with pytest.raises(RuntimeError, match="v56 aggregation generation"):
        core_db.initialize(reopened)
    reopened.close()


def test_registry_is_material_and_immutable(cfg):
    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        binding = aggregation_generation_binding(enabled, _producer("one"))
        hy.conn
        before = material_store_state(cfg.db_path)
        with core_db.transaction(hy.conn):
            register_aggregation_generation(hy.conn, binding)
        after = material_store_state(cfg.db_path)
        assert before != after
        with pytest.raises(Exception):
            hy.conn.execute(
                "UPDATE aggregation_generations SET reuse_scope='process_instance'"
            )
    finally:
        hy.close()


def test_exact_requests_roundtrip_for_every_tree_kind(cfg):
    from tests.test_aggregation_provenance_v55 import _seed_tree

    hy = HyMem(cfg)
    try:
        _enabled, publication = _seed_tree(hy, cfg)
        assert {proof.row["node_kind"] for proof in publication.nodes.values()} == {
            "cluster", "rollup", "root",
        }
        for proof in publication.nodes.values():
            request = aggregation_llm_request(
                proof.inputs, node_kind=str(proof.row["node_kind"]),
            )
            assert proof.row["aggregation_request_hash"] == (
                aggregation_llm_request_hash(request)
            )
            assert proof.row["output_hash"] == aggregation_output_hash(
                str(proof.row["node_kind"]), str(proof.row["title"]),
                str(proof.row["summary"]),
                str(proof.row["aggregation_request_hash"]),
            )
    finally:
        hy.close()


def test_copied_request_hash_is_not_a_valid_publication(cfg):
    from tests.test_aggregation_provenance_v55 import _seed_tree

    hy = HyMem(cfg)
    try:
        _enabled, publication = _seed_tree(hy, cfg)
        root = publication.nodes[publication.root_node_id]
        cluster = next(
            proof for proof in publication.nodes.values()
            if proof.row["node_kind"] == "cluster"
        )
        hy.conn.execute("DROP TRIGGER aggregation_generation_node_update_guard")
        hy.conn.execute("DROP TRIGGER aggregation_source_bound_update_guard")
        copied = root.row["aggregation_request_hash"]
        hy.conn.execute(
            "UPDATE aggregation_nodes SET aggregation_request_hash=?,output_hash=? "
            "WHERE id=?",
            (
                copied,
                aggregation_output_hash(
                    "cluster", str(cluster.row["title"]),
                    str(cluster.row["summary"]), str(copied),
                ),
                cluster.row["id"],
            ),
        )
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


def test_empty_publication_is_generation_bound_without_provider_call(cfg):
    enabled = _aggregation_cfg(cfg)
    client = _producer("unused")
    hy = HyMem(enabled)
    try:
        result = build_aggregation_nodes(hy.conn, enabled, client)
        assert result.nodes == 0 and client.calls == []
        publication = load_current_aggregation_publication(hy.conn)
        assert publication is not None and publication.nodes == {}
        assert publication.generation_key == aggregation_generation_binding(
            enabled, client
        )["generation_key"]
    finally:
        hy.close()


class _Transparent:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, request):
        return self.inner.complete(request)


def test_only_registered_internal_wrappers_preserve_identity(cfg):
    enabled = _aggregation_cfg(cfg)
    source = _producer("source")
    baseline = aggregation_generation_binding(enabled, source)
    arbitrary = _Transparent(source)
    assert aggregation_generation_binding(
        enabled, arbitrary
    )["producer"]["identity_exact"] is False

    deadline = DeadlineBoundLLMClient(
        source, MonotonicDeadline.after(60),
    )
    _register_phase1_producer_proxy(deadline, source)
    assert aggregation_generation_binding(enabled, deadline) == baseline


def test_direct_query_with_other_llm_does_not_serve_old_nodes(cfg):
    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        _seed(hy.conn)
        build_aggregation_nodes(hy.conn, enabled, _producer("A"))
        context = query_augment(
            hy.conn, enabled, "shared-thread", llm=_producer("B"),
            ability="TR",
        )
        assert context.aggregation_nodes == []
    finally:
        hy.close()


def test_direct_query_rejects_generation_key_from_other_llm(cfg):
    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        _seed(hy.conn)
        producer_a = _producer("A")
        build_aggregation_nodes(hy.conn, enabled, producer_a)
        key_a = aggregation_generation_binding(
            enabled, producer_a,
        )["generation_key"]
        producer_b = _producer("B")
        with pytest.raises(ValueError, match="differs from the supplied LLM"):
            query_augment(
                hy.conn, enabled, "shared-thread", llm=producer_b,
                ability="TR", aggregation_generation_key=str(key_a),
            )
        assert producer_b.calls == []
    finally:
        hy.close()


def test_no_llm_reader_does_not_adopt_live_inexact_publication(cfg):
    enabled = _aggregation_cfg(cfg, digest=False)
    writer = HyMem(enabled)
    producer = _UnknownClient("live unknown")
    try:
        _seed(writer.conn)
        build_aggregation_nodes(writer.conn, enabled, producer)
        assert load_current_aggregation_publication(
            writer.conn,
            expected_generation_key=str(
                aggregation_generation_binding(enabled, producer)["generation_key"]
            ),
        ) is not None
        live_key = str(
            aggregation_generation_binding(enabled, producer)["generation_key"]
        )
        live = query_augment(
            writer.conn, enabled, "live unknown", llm=producer, ability="TR",
        )
        assert live.aggregation_nodes
        historical = query_augment(
            writer.conn, enabled, "live unknown", llm=None, ability="TR",
            aggregation_generation_key=live_key,
        )
        assert historical.aggregation_nodes == []
        reader = HyMem(enabled)
        try:
            assert load_current_aggregation_publication(reader.conn) is None
            status = reader.dream_status()
            assert status["pending_aggregation"] == 1
            assert status["aggregation_publication_generation_key"] is None
        finally:
            reader.close()
    finally:
        writer.close()


def test_api_caches_contract_but_not_mutable_producer(cfg, monkeypatch):
    from hymem import api as api_mod

    enabled = _aggregation_cfg(cfg)
    client = _DeclaredClient("model-a")
    hy = HyMem(enabled, llm=client)
    try:
        monkeypatch.setattr(
            api_mod, "aggregation_generation_contract",
            lambda _cfg: (_ for _ in ()).throw(AssertionError("rehash")),
        )
        first = hy._current_aggregation_generation_key()
        assert hy._current_aggregation_generation_key() == first
        client.request_mode = "changed"
        assert hy._current_aggregation_generation_key() != first
    finally:
        hy.close()


def test_disabled_aggregation_does_not_require_source_identity(cfg, monkeypatch):
    from hymem import api as api_mod

    disabled = replace(cfg, aggregation_nodes_enabled=False)
    monkeypatch.setattr(
        api_mod, "aggregation_generation_contract",
        lambda _cfg: (_ for _ in ()).throw(ValueError("source unavailable")),
    )
    hy = HyMem(disabled, llm=_producer("unused"))
    try:
        assert hy._aggregation_contract is None
        assert hy._current_aggregation_generation_key() is None
    finally:
        hy.close()


def test_parser_helper_change_rotates_generation_contract(cfg, monkeypatch):
    from hymem.extraction import jsonio

    enabled = _aggregation_cfg(cfg)
    client = _DeclaredClient("model")
    baseline = aggregation_generation_binding(enabled, client)["generation_key"]
    original = jsonio._candidate_spans

    def changed_candidate_spans(text, expect):
        return original(text, expect)

    monkeypatch.setattr(jsonio, "_candidate_spans", changed_candidate_spans)
    assert aggregation_generation_binding(enabled, client)["generation_key"] != baseline


def test_cache_proof_loader_change_rotates_generation_contract(cfg, monkeypatch):
    from hymem.dreaming import aggregate as aggregate_mod

    enabled = _aggregation_cfg(cfg)
    client = _DeclaredClient("model")
    baseline = aggregation_generation_binding(enabled, client)["generation_key"]
    original = aggregate_mod.load_aggregation_node_proof

    def changed_loader(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(
        aggregate_mod, "load_aggregation_node_proof", changed_loader,
    )
    assert aggregation_generation_binding(enabled, client)["generation_key"] != baseline


@pytest.mark.parametrize(
    "llm_request",
    (
        LLMRequest(system="s", user="u", max_tokens=True),
        LLMRequest(system="s", user="u", max_tokens=0),
        LLMRequest(system="s", user="u", temperature=True),
        LLMRequest(system="s", user="u", temperature=float("nan")),
        LLMRequest(system="s", user="u", temperature=float("inf")),
        LLMRequest(system="s", user="u", response_format="xml"),
    ),
)
def test_request_hash_rejects_noncanonical_scalars(llm_request):
    with pytest.raises(ValueError, match="aggregation request"):
        aggregation_llm_request_hash(llm_request)


def test_credential_value_is_not_identity_or_persisted(cfg):
    enabled = _aggregation_cfg(cfg)
    one = _DeclaredClient("model")
    two = _DeclaredClient("model")
    one.api_key = "first-private-value"
    two.api_key = "second-private-value"
    left = aggregation_generation_binding(enabled, one)
    right = aggregation_generation_binding(enabled, two)
    assert left == right
    encoded = json.dumps(left, sort_keys=True)
    assert one.api_key not in encoded and two.api_key not in encoded


def test_unreferenced_inexact_registry_rows_are_bounded(cfg):
    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        for index in range(8):
            client = _UnknownClient(str(index))
            binding = aggregation_generation_binding(enabled, client)
            with core_db.transaction(hy.conn):
                register_aggregation_generation(hy.conn, binding)
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM aggregation_generations WHERE identity_exact=0"
        ).fetchone()[0] == 1
    finally:
        hy.close()


def test_v56_restart_heals_owned_trigger(cfg):
    hy = HyMem(cfg)
    hy.conn.execute("DROP TRIGGER aggregation_generations_insert_guard")
    hy.close()
    reopened = core_db.connect(cfg.db_path)
    try:
        core_db.initialize(reopened)
        assert core_db._v56_generation_bindings_present(
            reopened, allow_v57=True,
        )
    finally:
        reopened.close()


def _downgrade_empty_v56_store_to_v55(conn: sqlite3.Connection) -> None:
    for trigger in (
        "aggregation_health_attempt_insert_guard",
        "aggregation_health_attempt_update_guard",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    for trigger in (
        "aggregation_generations_insert_guard",
        "aggregation_generations_update_guard",
        "aggregation_generations_delete_guard",
        "aggregation_generation_node_update_guard",
        "aggregation_generation_publication_insert_guard",
    ):
        conn.execute(f"DROP TRIGGER {trigger}")
    for table, columns in (
        ("aggregation_nodes", (
            "aggregation_request_hash", "aggregation_generation_key",
        )),
        ("aggregation_publication_state", (
            "request_contract_sha256", "aggregation_generation_key",
        )),
        ("aggregation_build_health", (
            "last_failure_generation_key", "pending_generation_key",
            "last_success_generation_key",
        )),
        ("dream_runs", ("aggregation_generation_key",)),
    ):
        for column in columns:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    conn.execute("DROP TABLE aggregation_generations")
    conn.execute(
        "DELETE FROM schema_meta WHERE key='aggregation_generation_schema'"
    )
    conn.execute(
        "UPDATE schema_meta SET value='55' WHERE key='schema_version'"
    )


@pytest.mark.parametrize(
    "table,column",
    (
        ("aggregation_build_health", "pending_generation_key"),
        ("dream_runs", "aggregation_generation_key"),
    ),
)
def test_v56_incompatible_precreated_generation_column_rolls_back(
    cfg, table, column,
):
    hy = HyMem(cfg)
    _downgrade_empty_v56_store_to_v55(hy.conn)
    hy.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} INTEGER")
    hy.close()

    reopened = core_db.connect(cfg.db_path)
    with pytest.raises(RuntimeError, match="v56 aggregation generation domain"):
        core_db._run_migrations(reopened)
    assert core_db.schema_version(reopened) == 55
    assert reopened.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='aggregation_generations'"
    ).fetchone() is None
    malformed = {
        row["name"]: row["type"]
        for row in reopened.execute(f"PRAGMA table_info({table})")
    }
    assert malformed[column].upper() == "INTEGER"
    reopened.close()


def test_duplicate_key_registry_json_is_not_loadable(cfg):
    hy = HyMem(cfg)
    binding = aggregation_generation_binding(_aggregation_cfg(cfg), _producer("A"))
    producer = binding["producer"]
    duplicate = json.dumps(binding, sort_keys=True, separators=(",", ":"))
    duplicate = duplicate[:-1] + ',"schema":"hymem-aggregation-generation-v1"}'
    hy.conn.execute("DROP TRIGGER aggregation_generations_insert_guard")
    hy.conn.execute(
        "INSERT INTO aggregation_generations VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        (
            binding["generation_key"],
            binding["contract"]["material_config_version"],
            producer["identity_sha256"], 1, "durable", duplicate,
        ),
    )
    from hymem.dreaming.aggregation_generation import (
        load_registered_aggregation_generation,
    )
    assert load_registered_aggregation_generation(
        hy.conn, binding["generation_key"],
    ) is None
    hy.close()


def test_registry_binding_swap_hides_publication(cfg):
    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        _seed(hy.conn)
        producer_a = _producer("A")
        build_aggregation_nodes(hy.conn, enabled, producer_a)
        publication = load_current_aggregation_publication(hy.conn)
        assert publication is not None
        binding_b = aggregation_generation_binding(enabled, _producer("B"))
        producer_b = binding_b["producer"]
        hy.conn.execute("DROP TRIGGER aggregation_generations_update_guard")
        hy.conn.execute(
            "UPDATE aggregation_generations SET material_config_version=?,"
            "producer_identity_sha256=?,identity_exact=?,reuse_scope=?,binding_json=? "
            "WHERE generation_key=?",
            (
                binding_b["contract"]["material_config_version"],
                producer_b["identity_sha256"], 1, "durable",
                json.dumps(binding_b, sort_keys=True, separators=(",", ":")),
                publication.generation_key,
            ),
        )
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()


def test_missing_publication_generation_is_invisible(cfg):
    enabled = _aggregation_cfg(cfg)
    hy = HyMem(enabled)
    try:
        _seed(hy.conn)
        build_aggregation_nodes(hy.conn, enabled, _producer("A"))
        hy.conn.execute("DROP TRIGGER aggregation_publication_update_guard")
        hy.conn.execute(
            "UPDATE aggregation_publication_state "
            "SET aggregation_generation_key=NULL,request_contract_sha256=NULL"
        )
        assert load_current_aggregation_publication(hy.conn) is None
    finally:
        hy.close()
