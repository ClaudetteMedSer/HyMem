"""Producer authority regressions for Phase-1 auxiliary/materialized state."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import canonicalize, phase1, phase2
from hymem.dreaming.phase1_auxiliary import (
    AUXILIARY_CONTRACT_V0,
    AUXILIARY_CONTRACT_V1,
    AUXILIARY_CONTRACT_V2,
    AUXILIARY_POLICY_SHA256,
    CURRENT_AUXILIARY_CONTRACT_KEY,
    auxiliary_policy_sha256,
    canonical_auxiliary_result,
)
from hymem.extraction.markers import Marker
from hymem.extraction.producer import phase1_generation_binding
from hymem.extraction.triples import Triple
from hymem.rules import route_markers_to_rules
from tests.test_phase1_producer_identity import _DeclaredLLM, _seed_chunk


def _persist_projection(
    hy: HyMem,
    chunk,
    client: _DeclaredLLM,
    *,
    type_hints: dict[str, str] | None = None,
    property_hints: dict[str, dict[str, str]] | None = None,
    entity_mentions: list[str] | None = None,
    markers: list[tuple[str, str]] | None = None,
    failed: bool = False,
) -> str:
    hy.set_llm(client)
    binding = phase1_generation_binding(hy.config.prompt_version, client)
    result = phase1.ChunkExtraction(
        triples=(
            [
                Triple(
                    subject=entity,
                    predicate="uses",
                    object=entity,
                    polarity=1,
                    source_message_id=chunk.start_message_id,
                )
                for entity in entity_mentions or []
            ]
            if not failed else []
        ),
        markers=[Marker(kind=kind, statement=text) for kind, text in markers or []],
        entity_type_hints=type_hints or {},
        entity_property_hints=property_hints or {},
        failed=failed,
        failure_reason="parse_failure" if failed else None,
        failure_details=("response",) if failed else (),
        source_validated=True,
        phase1_generation=binding,
        claim_sources={
            source.message_id: source
            for source in phase1._claim_sources_for_chunk(hy.conn, chunk)
        },
    )
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn,
            chunk,
            result,
            prompt_version=hy.config.prompt_version,
            cfg=hy.config,
        )
    return str(binding["generation_key"])


def _materialize_markers(hy: HyMem, generation_key: str) -> None:
    with core_db.transaction(hy.conn):
        route_markers_to_rules(
            hy.conn,
            hy.config,
            phase1_generation_key=generation_key,
        )
        phase2.consolidate_profile(
            hy.conn,
            hy.config,
            phase1_generation_key=generation_key,
        )


def _materialize_profile_only(hy: HyMem, generation_key: str) -> None:
    with core_db.transaction(hy.conn):
        phase2.consolidate_profile(
            hy.conn,
            hy.config,
            phase1_generation_key=generation_key,
        )


def _rows(conn, query: str) -> list[tuple]:
    return [tuple(row) for row in conn.execute(query).fetchall()]


def _align_portable_seed_parent_clocks(*stores: HyMem) -> None:
    """Make `_seed_chunk`'s shared source parent byte-identical in test stores."""
    for store in stores:
        store.conn.execute(
            "UPDATE sessions SET started_at='2026-09-06T09:00:00Z' "
            "WHERE id='producer-session'"
        )
        store.conn.execute(
            "UPDATE chunks SET created_at='2026-09-06T10:00:00Z' "
            "WHERE chunk_kind='coverage'"
        )
        store.conn.execute(
            "UPDATE chunks SET created_at='2026-09-06T12:00:00Z' "
            "WHERE chunk_kind<>'coverage'"
        )
        # Normalize only the operational publication clock, then immediately
        # restore both immutable coverage guards used by normal code.
        store.conn.execute("DROP TRIGGER message_retention_coverage_update_guard")
        store.conn.execute("DROP TRIGGER message_lossless_stream_update_guard")
        store.conn.execute(
            "UPDATE message_retention_coverage "
            "SET created_at='2026-09-06T11:00:00Z'"
        )
        core_db._install_external_peer_guards(store.conn)
        store.conn.execute(
            "CREATE TRIGGER message_lossless_stream_update_guard "
            "BEFORE UPDATE ON message_retention_coverage "
            "WHEN old.coverage_version='dream-lossless-message-v1' BEGIN "
            "SELECT RAISE(ABORT,'ordered digest coverage is immutable'); END"
        )


def _remove_v54_auxiliary_domain(conn: sqlite3.Connection) -> None:
    """Test-only downgrade helper retaining v53 claim/marker history."""
    core_db._drop_phase1_auxiliary_views(conn)
    base_trigger_names = (
        "behavioral_marker_producer_insert_guard",
        "behavioral_marker_semantic_update_guard",
        "behavioral_marker_delete_guard",
        "linked_profile_semantic_update_guard",
        "linked_profile_delete_guard",
        "linked_rule_semantic_update_guard",
        "linked_rule_delete_guard",
        "entity_type_authority_insert_guard",
        "entity_type_authority_update_guard",
        "entity_property_authority_insert_guard",
        "entity_property_authority_update_guard",
        "profile_entry_domain_insert_guard",
        "profile_entry_domain_update_guard",
        "rule_domain_insert_guard",
        "rule_domain_update_guard",
    )
    for name in base_trigger_names:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP INDEX IF EXISTS idx_behavioral_marker_generation_identity")
    for table in (
        "profile_marker_decisions",
        "profile_entry_marker_evidence",
        "rule_marker_decisions",
        "rule_marker_evidence",
        "entity_type_observations",
        "entity_property_observations",
        "entity_mention_observations",
        "phase1_auxiliary_outcomes",
    ):
        conn.execute(f"DROP TABLE {table}")


def _replace_type_observation_with_extra_check(conn: sqlite3.Connection) -> None:
    core_db._drop_phase1_auxiliary_views(conn)
    for name in (
        "entity_type_observation_insert_guard",
        "entity_type_observation_update_guard",
        "entity_type_observation_delete_guard",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    for name in (
        "idx_entity_type_observations_lookup",
        "idx_entity_type_observations_entity",
    ):
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.execute("DROP TABLE entity_type_observations")
    conn.execute(
        "CREATE TABLE entity_type_observations("
        "chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,"
        "entity_canonical TEXT NOT NULL CHECK("
        "length(trim(entity_canonical))>0 AND "
        "hymem_entity_canonical_is_normalized(entity_canonical)=1),"
        "type TEXT NOT NULL CHECK(length(trim(type))>0),"
        "confidence REAL NOT NULL DEFAULT 1.0 CHECK("
        "typeof(confidence) IN ('integer','real') AND "
        "confidence>=0.0 AND confidence<=1.0),"
        "phase1_generation_key TEXT NOT NULL REFERENCES "
        "phase1_generations(generation_key) ON DELETE RESTRICT,"
        "observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "PRIMARY KEY(chunk_id,entity_canonical,type,phase1_generation_key),"
        "CHECK(type='person'))"
    )


def _replace_entity_types_with_generated_gate(conn: sqlite3.Connection) -> None:
    core_db._drop_phase1_auxiliary_views(conn)
    for name in (
        "entity_type_authority_insert_guard",
        "entity_type_authority_update_guard",
        "idx_entity_types_type",
        "idx_entity_types_entity",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.execute("DROP TABLE entity_types")
    conn.execute(
        "CREATE TABLE entity_types("
        "entity_canonical TEXT NOT NULL,type TEXT NOT NULL,"
        "confidence REAL NOT NULL DEFAULT 1.0,"
        "source_chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,"
        "origin TEXT NOT NULL DEFAULT 'legacy_unattributed' "
        "CHECK(origin IN ('user','legacy_unattributed')),"
        "gate INTEGER GENERATED ALWAYS AS ("
        "CASE WHEN type='person' THEN 1 END) STORED NOT NULL,"
        "PRIMARY KEY(entity_canonical,type))"
    )


def _replace_entity_properties_with_nocase_value(conn: sqlite3.Connection) -> None:
    core_db._drop_phase1_auxiliary_views(conn)
    for name in (
        "entity_property_authority_insert_guard",
        "entity_property_authority_update_guard",
        "idx_entity_properties_key",
        "idx_entity_properties_value",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.execute("DROP TABLE entity_properties")
    conn.execute(
        "CREATE TABLE entity_properties("
        "entity_canonical TEXT NOT NULL,key TEXT NOT NULL,"
        "value TEXT NOT NULL COLLATE NOCASE,"
        "source_chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,"
        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        "origin TEXT NOT NULL DEFAULT 'legacy_unattributed' "
        "CHECK(origin IN ('user','legacy_unattributed')),"
        "PRIMARY KEY(entity_canonical,key))"
    )


def _rewrite_jsonl(path: Path, mutate) -> None:
    objects = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    body, end = objects[:-1], objects[-1]
    mutate(body)
    end["counts"] = {
        kind: sum(item.get("type") == kind for item in body)
        for kind in end["counts"]
    }
    encoded = [json.dumps(item, ensure_ascii=False) + "\n" for item in body]
    end["sha256"] = hashlib.sha256("".join(encoded).encode("utf-8")).hexdigest()
    path.write_text(
        "".join(encoded) + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_auxiliary_a_failure_b_success_return_a_and_idempotent_replay(cfg):
    config = replace(
        cfg,
        aggregation_nodes_enabled=False,
        rules_extraction_mode="lexical",
    )
    client_a = _DeclaredLLM("aux-a", "unused")
    hy = HyMem(config, llm=client_a)
    try:
        chunk = _seed_chunk(hy, chunk_id="aux-switch")
        key_a = _persist_projection(
            hy,
            chunk,
            client_a,
            type_hints={"Tool": "tool"},
            property_hints={"Tool": {"language": "python"}},
            entity_mentions=["Tool"],
            markers=[("style", "Always answer tersely")],
        )
        _materialize_markers(hy, key_a)
        assert _rows(
            hy.conn,
            "SELECT entity_canonical,type FROM current_entity_types",
        ) == [("tool", "tool")]
        assert _rows(
            hy.conn,
            "SELECT entity_canonical,key,value FROM current_entity_properties",
        ) == [("tool", "language", "python")]
        assert _rows(
            hy.conn, "SELECT entity_canonical FROM current_entity_mentions",
        ) == [("tool",)]
        assert _rows(
            hy.conn, "SELECT text,source FROM current_profile_entries"
        ) == [("Always answer tersely", "agent_inferred")]
        assert _rows(
            hy.conn, "SELECT text,source FROM current_rules"
        ) == [("Always answer tersely", "agent_inferred")]

        physical_a = {
            table: _rows(hy.conn, f"SELECT * FROM {table} ORDER BY rowid")
            for table in (
                "entity_type_observations",
                "entity_property_observations",
                "entity_mention_observations",
                "behavioral_markers",
                "phase1_auxiliary_outcomes",
                "profile_entry_marker_evidence",
                "profile_marker_decisions",
                "rule_marker_evidence",
                "rule_marker_decisions",
            )
        }

        failed_b = _DeclaredLLM("aux-b", "unused", fail=True)
        _persist_projection(hy, chunk, failed_b, failed=True)
        # A is retained physically, but configured B reads fail closed.
        for table in (
            "current_entity_types", "current_entity_properties",
            "current_entity_mentions",
            "current_profile_entries", "current_rules",
        ):
            assert hy.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        for table, rows in physical_a.items():
            assert _rows(hy.conn, f"SELECT * FROM {table} ORDER BY rowid") == rows

        healthy_b = _DeclaredLLM("aux-b", "unused")
        key_b = _persist_projection(
            hy,
            chunk,
            healthy_b,
            type_hints={"Tool": "service"},
            property_hints={"Tool": {"runtime": "rust"}},
            entity_mentions=["Redis"],
            markers=[("style", "Always include examples")],
        )
        assert key_b != key_a
        _materialize_markers(hy, key_b)
        assert _rows(
            hy.conn,
            "SELECT entity_canonical,type FROM current_entity_types",
        ) == [("tool", "service")]
        assert _rows(
            hy.conn,
            "SELECT entity_canonical,key,value FROM current_entity_properties",
        ) == [("tool", "runtime", "rust")]
        assert _rows(
            hy.conn, "SELECT entity_canonical FROM current_entity_mentions",
        ) == [("redis",)]
        assert _rows(
            hy.conn, "SELECT text FROM current_profile_entries"
        ) == [("Always include examples",)]
        assert _rows(hy.conn, "SELECT text FROM current_rules") == [
            ("Always include examples",)
        ]

        # Returning to A is not a stale cache hit: a successful replay switches
        # both claim and auxiliary publication back to A.
        hy.set_llm(client_a)
        assert hy.dream_status()["pending_chunks"] == 1
        key_a_replay = _persist_projection(
            hy,
            chunk,
            client_a,
            type_hints={"Tool": "tool"},
            property_hints={"Tool": {"language": "python"}},
            entity_mentions=["Tool"],
            markers=[("style", "Always answer tersely")],
        )
        assert key_a_replay == key_a
        assert _rows(
            hy.conn,
            "SELECT entity_canonical,key,value FROM current_entity_properties",
        ) == [("tool", "language", "python")]
        assert _rows(
            hy.conn, "SELECT entity_canonical FROM current_entity_mentions",
        ) == [("tool",)]
        assert _rows(
            hy.conn, "SELECT text FROM current_profile_entries"
        ) == [("Always answer tersely",)]

        stable = {
            table: _rows(hy.conn, f"SELECT * FROM {table} ORDER BY rowid")
            for table in (
                "entity_type_observations",
                "entity_property_observations",
                "entity_mention_observations",
                "behavioral_markers",
                "phase1_auxiliary_outcomes",
                "profile_entry_marker_evidence",
                "profile_marker_decisions",
                "rule_marker_evidence",
                "rule_marker_decisions",
            )
        }
        hy.conn.execute("DELETE FROM processed_chunks WHERE chunk_id=?", (chunk.id,))
        _persist_projection(
            hy,
            chunk,
            client_a,
            type_hints={"Tool": "tool"},
            property_hints={"Tool": {"language": "python"}},
            entity_mentions=["Tool"],
            markers=[("style", "Always answer tersely")],
        )
        _materialize_markers(hy, key_a)
        for table, rows in stable.items():
            assert _rows(hy.conn, f"SELECT * FROM {table} ORDER BY rowid") == rows
    finally:
        hy.close()

    restarted = HyMem(config, llm=_DeclaredLLM("aux-a", "unused"))
    try:
        assert _rows(
            restarted.conn,
            "SELECT entity_canonical,key,value FROM current_entity_properties",
        ) == [("tool", "language", "python")]
        assert restarted.dream_status()["pending_chunks"] == 0
    finally:
        restarted.close()


def test_v53_producer_marker_upgrades_as_unscoped_history_then_replays(cfg):
    config = replace(
        cfg,
        aggregation_nodes_enabled=False,
        rules_extraction_mode="lexical",
    )
    client = _DeclaredLLM("v53-marker-upgrade", "unused")
    hy = HyMem(config, llm=client)
    chunk = _seed_chunk(hy, chunk_id="v53-marker-upgrade")
    key = _persist_projection(
        hy,
        chunk,
        client,
        markers=[("style", "Always include a short example")],
    )
    _materialize_markers(hy, key)
    assert hy.conn.execute(
        "SELECT phase1_generation_key FROM behavioral_markers"
    ).fetchone()[0] == key

    # Recreate the real authority boundary of v53: producer-bound claims and
    # markers existed, but no whole-response auxiliary outcome or marker-link
    # ledger could prove those markers were part of the accepted result.
    with core_db.evidence_mutation(hy.conn):
        _remove_v54_auxiliary_domain(hy.conn)
        hy.conn.execute(
            "UPDATE profile_entries SET source='legacy_unattributed'"
        )
        hy.conn.execute(
            "UPDATE schema_meta SET value='53' WHERE key='schema_version'"
        )
    hy.close()

    upgraded = HyMem(config, llm=_DeclaredLLM("v53-marker-upgrade", "unused"))
    try:
        assert core_db.schema_version(upgraded.conn) == core_db.EXPECTED_SCHEMA_VERSION
        assert _rows(
            upgraded.conn,
            "SELECT statement,phase1_generation_key FROM behavioral_markers",
        ) == [("Always include a short example", None)]
        assert upgraded.conn.execute(
            "SELECT COUNT(*) FROM current_profile_entries"
        ).fetchone()[0] == 0
        assert upgraded.conn.execute(
            "SELECT COUNT(*) FROM current_rules"
        ).fetchone()[0] == 0
        assert upgraded.dream_status()["pending_chunks"] == 1

        replay_key = _persist_projection(
            upgraded,
            chunk,
            client,
            markers=[("style", "Always include a short example")],
        )
        assert replay_key == key
        _materialize_markers(upgraded, replay_key)
        assert _rows(
            upgraded.conn,
            "SELECT phase1_generation_key FROM behavioral_markers ORDER BY id",
        ) == [(None,), (key,)]
        assert _rows(
            upgraded.conn,
            "SELECT text,source FROM current_profile_entries",
        ) == [("Always include a short example", "agent_inferred")]
        assert _rows(
            upgraded.conn,
            "SELECT text,source FROM current_rules",
        ) == [("Always include a short example", "agent_inferred")]
        assert upgraded.dream_status()["pending_chunks"] == 0
    finally:
        upgraded.close()


def test_orphan_auxiliary_outcome_rejects_export_and_reopen(cfg, tmp_path):
    config = replace(cfg, aggregation_nodes_enabled=False)
    client = _DeclaredLLM("orphan-auxiliary", "unused")
    hy = HyMem(config, llm=client)
    chunk = _seed_chunk(hy, chunk_id="orphan-auxiliary")
    _persist_projection(
        hy,
        chunk,
        client,
        type_hints={"app": "service"},
        property_hints={"app": {"runtime": "rust"}},
        entity_mentions=["app"],
        markers=[("style", "Always show a short example")],
    )
    with core_db.evidence_mutation(hy.conn):
        hy.conn.execute(
            "DELETE FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (chunk.id,),
        )
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM phase1_auxiliary_outcomes"
    ).fetchone()[0] == 1
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM entity_type_observations"
    ).fetchone()[0] == 1
    assert hy.conn.execute(
        "SELECT COUNT(*) FROM behavioral_markers"
    ).fetchone()[0] == 1

    export_path = tmp_path / "must-not-replace.jsonl"
    export_path.write_text("sentinel", encoding="utf-8")
    with pytest.raises(RuntimeError, match="auxiliary current claim lineage"):
        hy.export(export_path)
    assert export_path.read_text(encoding="utf-8") == "sentinel"
    hy.close()

    reopened = HyMem(config, llm=_DeclaredLLM("orphan-auxiliary", "unused"))
    with pytest.raises(RuntimeError, match="auxiliary current claim lineage"):
        _ = reopened.conn


def test_manual_hints_validate_redact_export_and_dominate_inference(cfg, tmp_path):
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False), llm=_DeclaredLLM("m", "x"))
    try:
        with pytest.raises(ValueError):
            hy.set_entity_type(" ", "tool")
        with pytest.raises(ValueError):
            hy.set_entity_type("thing", " ")
        for confidence in (True, -0.1, 1.1, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                hy.set_entity_type("thing", "tool", confidence=confidence)
        with pytest.raises(ValueError):
            hy.set_entity_property("thing", " ", "value")
        with pytest.raises(ValueError):
            hy.set_entity_property("thing", "key", 3)  # type: ignore[arg-type]

        hy.set_entity_type("The Tool", "manual_type", confidence=0.8)
        hy.set_entity_property(
            "The Tool", "token", "api_key=abcdefghijk123456789"
        )
        chunk = _seed_chunk(hy, chunk_id="manual-wins")
        key = _persist_projection(
            hy,
            chunk,
            _DeclaredLLM("m", "x"),
            type_hints={"Tool": "manual_type"},
            property_hints={"Tool": {"token": "model-value"}},
        )
        assert key
        assert _rows(
            hy.conn,
            "SELECT confidence,origin FROM current_entity_types "
            "WHERE entity_canonical='tool' AND type='manual_type'",
        ) == [(0.8, "user")]
        value, origin = hy.conn.execute(
            "SELECT value,origin FROM current_entity_properties "
            "WHERE entity_canonical='tool' AND key='token'"
        ).fetchone()
        assert value == "api_key=[REDACTED-SECRET]"
        assert origin == "user"

        export_path = tmp_path / "manual.jsonl"
        hy.export(export_path)
        assert "abcdefghijk123456789" not in export_path.read_text(encoding="utf-8")
    finally:
        hy.close()


def test_legacy_unattributed_null_rows_never_become_manual_or_portable(cfg, tmp_path):
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False))
    try:
        chunk = _seed_chunk(hy, chunk_id="legacy-hint-source")
        hy.conn.execute(
            "INSERT INTO entity_types(entity_canonical,type,confidence,"
            "source_chunk_id) VALUES ('legacy','tool',0.5,?)",
            (chunk.id,),
        )
        hy.conn.execute(
            "INSERT INTO entity_properties(entity_canonical,key,value,"
            "source_chunk_id) VALUES ('legacy','bad','value',?)",
            (chunk.id,),
        )
        hy.conn.execute("DELETE FROM chunks WHERE id=?", (chunk.id,))
        assert tuple(hy.conn.execute(
            "SELECT source_chunk_id,origin FROM entity_types WHERE "
            "entity_canonical='legacy'"
        ).fetchone()) == (None, "legacy_unattributed")
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM current_entity_types WHERE "
            "entity_canonical='legacy'"
        ).fetchone()[0] == 0
        path = tmp_path / "legacy-filtered.jsonl"
        hy.export(path)
        assert '"entity_canonical":"legacy"' not in path.read_text(encoding="utf-8")
    finally:
        hy.close()


def test_historical_aux_contract_is_valid_history_but_pending_until_replay(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False)
    client = _DeclaredLLM("old-aux", "unused")
    hy = HyMem(config, llm=client)
    chunk = _seed_chunk(hy, chunk_id="old-aux-contract")
    generation_key = _persist_projection(
        hy,
        chunk,
        client,
        type_hints={"Tool": "tool"},
        property_hints={"Tool": {"language": "python"}},
    )
    cache_key = str(hy.conn.execute(
        "SELECT prompt_version FROM processed_chunks WHERE chunk_id=?",
        (chunk.id,),
    ).fetchone()[0])
    v0 = canonical_auxiliary_result(
        chunk_id=chunk.id,
        phase1_generation_key=generation_key,
        extraction_cache_key=cache_key,
        auxiliary_contract_key=AUXILIARY_CONTRACT_V0,
        entity_types=[("tool", "tool", 1.0)],
        entity_properties=[("tool", "language", "python")],
        entity_mentions=[],
        markers=[],
    )
    with core_db.evidence_mutation(hy.conn):
        hy.conn.execute(
            "UPDATE phase1_auxiliary_outcomes SET auxiliary_contract_key=?,"
            "result_hash=?,entity_mention_count=0 WHERE chunk_id=? AND "
            "phase1_generation_key=?",
            (
                AUXILIARY_CONTRACT_V0, v0["result_hash"], chunk.id,
                generation_key,
            ),
        )
    assert hy.conn.execute("SELECT COUNT(*) FROM current_phase1_publications").fetchone()[0] == 0
    hy.close()

    reopened = HyMem(config, llm=_DeclaredLLM("old-aux", "unused"))
    try:
        assert reopened.dream_status()["pending_chunks"] == 1
        _persist_projection(
            reopened,
            chunk,
            _DeclaredLLM("old-aux", "unused"),
            type_hints={"Tool": "tool"},
            property_hints={"Tool": {"language": "python"}},
        )
        assert reopened.conn.execute(
            "SELECT auxiliary_contract_key FROM current_phase1_publications"
        ).fetchone()[0] == CURRENT_AUXILIARY_CONTRACT_KEY
    finally:
        reopened.close()


def test_auxiliary_contract_dispatch_and_policy_sentinel_are_frozen(monkeypatch):
    kwargs = dict(
        chunk_id="chunk",
        phase1_generation_key="generation",
        extraction_cache_key="cache",
        entity_types=[("entity", "tool", 1.0)],
        entity_properties=[("entity", "key", "value")],
        entity_mentions=["entity"],
        markers=[("style", "Always concise")],
    )
    v0 = canonical_auxiliary_result(
        **kwargs, auxiliary_contract_key=AUXILIARY_CONTRACT_V0,
    )
    v1 = canonical_auxiliary_result(
        **kwargs, auxiliary_contract_key=AUXILIARY_CONTRACT_V1,
    )
    v2 = canonical_auxiliary_result(
        **kwargs, auxiliary_contract_key=AUXILIARY_CONTRACT_V2,
    )
    assert v0["result_hash"] == (
        "sha256:c9a28e2ea5faaa19b2e66347873c53efa2496ac7fca033b2ac1c38bcabad8e09"
    )
    assert v1["result_hash"] == (
        "sha256:80372c58cb1f207364f4f7fa76a8d4735d42ebf7bb03ff70207121c48e4776a2"
    )
    assert v2["result_hash"] == (
        "sha256:8404d354e299f4ccdf86da773d179eff3189363fad4d4c4c6557eda1d4bdeeb1"
    )
    assert v0["entity_mention_count"] == 0
    assert v1["entity_mention_count"] == 1
    import hymem.dreaming.phase1_auxiliary as auxiliary

    monkeypatch.setattr(auxiliary, "CURRENT_AUXILIARY_CONTRACT_KEY", "future-v3")
    assert canonical_auxiliary_result(
        **kwargs, auxiliary_contract_key=AUXILIARY_CONTRACT_V0,
    )["result_hash"] == v0["result_hash"]
    assert canonical_auxiliary_result(
        **kwargs, auxiliary_contract_key=AUXILIARY_CONTRACT_V1,
    )["result_hash"] == v1["result_hash"]
    assert canonical_auxiliary_result(
        **kwargs, auxiliary_contract_key=AUXILIARY_CONTRACT_V2,
    )["result_hash"] == v2["result_hash"]
    with pytest.raises(ValueError, match="unsupported"):
        canonical_auxiliary_result(**kwargs, auxiliary_contract_key="unknown")
    assert auxiliary_policy_sha256() == AUXILIARY_POLICY_SHA256
    assert phase2.profile_materialization_policy_sha256() == (
        phase2.PROFILE_MATERIALIZATION_POLICY_SHA256
    )


def test_phase2_policy_identities_never_reread_source_after_import(monkeypatch):
    """An OLD worker keeps its import-captured policy after files roll forward."""

    import inspect
    import hymem.dreaming.phase1_auxiliary as auxiliary

    auxiliary_before = auxiliary.auxiliary_policy_sha256()
    profile_before = phase2.profile_materialization_policy_sha256()

    def late_source_read(*_args, **_kwargs):
        raise AssertionError("runtime policy identity reread mutable source")

    monkeypatch.setattr(inspect, "getsource", late_source_read)
    assert auxiliary.auxiliary_policy_sha256() == auxiliary_before
    assert phase2.profile_materialization_policy_sha256() == profile_before


def test_profile_cap_is_render_only_and_missing_decision_replays(cfg):
    config = replace(
        cfg,
        aggregation_nodes_enabled=False,
        profile_max_entries=1,
        rules_extraction_mode="lexical",
    )
    client = _DeclaredLLM("profile-cap", "unused")
    hy = HyMem(config, llm=client)
    try:
        chunk = _seed_chunk(hy, chunk_id="profile-cap")
        key = _persist_projection(
            hy,
            chunk,
            client,
            markers=[
                ("style", "Always answer one"),
                ("style", "Always answer two"),
                ("preference", "Prefers answer three"),
            ],
        )
        _materialize_markers(hy, key)
        assert hy.conn.execute("SELECT COUNT(*) FROM profile_entries").fetchone()[0] == 3
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM profile_entry_marker_evidence"
        ).fetchone()[0] == 3
        assert phase2.current_profile_body(hy.conn, config).count("\n") == 0

        marker = hy.conn.execute(
            "SELECT marker_id FROM profile_marker_decisions ORDER BY marker_id LIMIT 1"
        ).fetchone()[0]
        with core_db.evidence_mutation(hy.conn):
            hy.conn.execute(
                "DELETE FROM profile_marker_decisions WHERE marker_id=?", (marker,)
            )
            hy.conn.execute(
                "DELETE FROM profile_entry_marker_evidence WHERE marker_id=?", (marker,)
            )
        assert hy.conn.execute(
            "SELECT consolidated_at FROM behavioral_markers WHERE id=?", (marker,)
        ).fetchone()[0] is not None
        with core_db.transaction(hy.conn):
            phase2.consolidate_profile(
                hy.conn, config, phase1_generation_key=key,
            )
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM profile_marker_decisions WHERE marker_id=?",
            (marker,),
        ).fetchone()[0] == 1
        assert hy.conn.execute("SELECT COUNT(*) FROM profile_entries").fetchone()[0] == 3
    finally:
        hy.close()


def test_manual_profile_and_rule_are_not_reinforced_by_markers(cfg):
    config = replace(
        cfg,
        aggregation_nodes_enabled=False,
        rules_extraction_mode="lexical",
    )
    client = _DeclaredLLM("manual-materialization", "unused")
    hy = HyMem(config, llm=client)
    try:
        chunk = _seed_chunk(hy, chunk_id="manual-materialization")
        statement = "Always answer tersely"
        hy.conn.execute(
            "INSERT INTO profile_entries(kind,text,source) "
            "VALUES ('style',?,'user')", (statement,),
        )
        rule_id = hy.add_rule(statement)
        key = _persist_projection(
            hy, chunk, client, markers=[("style", statement)],
        )
        _materialize_markers(hy, key)
        assert tuple(hy.conn.execute(
            "SELECT pos_evidence,source FROM profile_entries WHERE text=?",
            (statement,),
        ).fetchone()) == (1, "user")
        assert tuple(hy.conn.execute(
            "SELECT pos_evidence,source FROM rules WHERE id=?", (rule_id,),
        ).fetchone()) == (1, "user")
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM profile_entry_marker_evidence"
        ).fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM rule_marker_evidence"
        ).fetchone()[0] == 0
    finally:
        hy.close()


def test_register_alias_rejects_owned_identity_and_manual_merge_conflict_is_atomic(
    cfg, tmp_path,
):
    hy = HyMem(replace(cfg, aggregation_nodes_enabled=False))
    try:
        hy.set_entity_type("App", "tool")
        with pytest.raises(ValueError, match="already owns state"):
            hy.register_alias("app", "application")
        assert canonicalize.resolve(hy.conn, "app") == "app"

        before_invalid_identity = "\n".join(hy.conn.iterdump())
        with pytest.raises(ValueError, match="normalized canonical"):
            hy.register_alias("Foo Surface", "Not Normal")
        with pytest.raises(ValueError, match="normalized canonical"):
            hy.merge_canonical("Not Normal", "drop")
        assert "\n".join(hy.conn.iterdump()) == before_invalid_identity

        hy.register_alias("friendly tool", "tool")
        with pytest.raises(ValueError, match="must not itself be an alias"):
            hy.register_alias("second surface", "friendly_tool")
        hy.export(tmp_path / "canonical-api.jsonl")

        hy.set_entity_property("keep", "env", "prod")
        hy.set_entity_property("drop", "env", "dev")
        before = "\n".join(hy.conn.iterdump())
        with pytest.raises(ValueError, match="conflicting manual"):
            hy.merge_canonical("keep", "drop")
        assert "\n".join(hy.conn.iterdump()) == before
    finally:
        hy.close()


def test_v0_outcome_with_mentions_is_rejected_on_reopen(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False)
    client = _DeclaredLLM("v0-mention", "unused")
    hy = HyMem(config, llm=client)
    chunk = _seed_chunk(hy, chunk_id="v0-mention")
    key = _persist_projection(
        hy, chunk, client, entity_mentions=["Tool"],
    )
    outcome = hy.conn.execute(
        "SELECT extraction_cache_key FROM phase1_auxiliary_outcomes "
        "WHERE chunk_id=? AND phase1_generation_key=?", (chunk.id, key),
    ).fetchone()
    v0 = canonical_auxiliary_result(
        chunk_id=chunk.id,
        phase1_generation_key=key,
        extraction_cache_key=str(outcome["extraction_cache_key"]),
        auxiliary_contract_key=AUXILIARY_CONTRACT_V0,
        entity_types=[],
        entity_properties=[],
        entity_mentions=[],
        markers=[],
    )
    with core_db.evidence_mutation(hy.conn):
        hy.conn.execute(
            "UPDATE phase1_auxiliary_outcomes SET auxiliary_contract_key=?,"
            "result_hash=?,entity_mention_count=0 WHERE chunk_id=? AND "
            "phase1_generation_key=?",
            (AUXILIARY_CONTRACT_V0, v0["result_hash"], chunk.id, key),
        )
    hy.close()

    with pytest.raises(RuntimeError, match="auxiliary publication integrity"):
        HyMem(config, llm=_DeclaredLLM("v0-mention", "unused")).conn


def test_authenticated_invalid_auxiliary_value_is_blocked_and_rejected_on_reopen(
    cfg,
):
    config = replace(cfg, aggregation_nodes_enabled=False)
    client = _DeclaredLLM("invalid-aux-domain", "unused")
    hy = HyMem(config, llm=client)
    chunk = _seed_chunk(hy, chunk_id="invalid-aux-domain")
    key = _persist_projection(
        hy, chunk, client, type_hints={"Tool": "tool"},
    )
    with core_db.evidence_mutation(hy.conn):
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute(
                "UPDATE entity_type_observations SET confidence=7 "
                "WHERE chunk_id=? AND phase1_generation_key=?",
                (chunk.id, key),
            )

    # Simulate an offline writer that disabled CHECK enforcement and removed
    # the live guard, then forged a matching hash. Reopen must validate the
    # semantic domain independently of hash authenticity.
    hy.conn.execute("PRAGMA ignore_check_constraints=ON")
    hy.conn.execute("DROP TRIGGER entity_type_observation_update_guard")
    with core_db.evidence_mutation(hy.conn):
        hy.conn.execute(
            "UPDATE entity_type_observations SET confidence=7 "
            "WHERE chunk_id=? AND phase1_generation_key=?", (chunk.id, key),
        )
        outcome = hy.conn.execute(
            "SELECT extraction_cache_key,auxiliary_contract_key "
            "FROM phase1_auxiliary_outcomes WHERE chunk_id=? AND "
            "phase1_generation_key=?", (chunk.id, key),
        ).fetchone()
        forged = canonical_auxiliary_result(
            chunk_id=chunk.id,
            phase1_generation_key=key,
            extraction_cache_key=str(outcome["extraction_cache_key"]),
            auxiliary_contract_key=str(outcome["auxiliary_contract_key"]),
            entity_types=[("tool", "tool", 7.0)],
            entity_properties=[],
            entity_mentions=[],
            markers=[],
        )
        hy.conn.execute(
            "UPDATE phase1_auxiliary_outcomes SET result_hash=? WHERE "
            "chunk_id=? AND phase1_generation_key=?",
            (forged["result_hash"], chunk.id, key),
        )
    hy.conn.execute("PRAGMA ignore_check_constraints=OFF")
    hy.close()

    with pytest.raises(RuntimeError, match="auxiliary publication integrity"):
        HyMem(config, llm=_DeclaredLLM("invalid-aux-domain", "unused")).conn


@pytest.mark.parametrize(
    ("table", "column", "guard", "error"),
    [
        (
            "profile_marker_decisions", "profile_policy_key",
            "profile_marker_decision_update_guard", "profile marker decision",
        ),
        (
            "rule_marker_decisions", "routing_key",
            "rule_marker_decision_update_guard", "rule marker decision",
        ),
    ],
)
def test_empty_materialization_policy_key_is_rejected_on_reopen(
    cfg, tmp_path, table, column, guard, error,
):
    config = replace(
        cfg,
        root=tmp_path / table,
        aggregation_nodes_enabled=False,
        rules_extraction_mode="lexical",
    )
    client = _DeclaredLLM("empty-policy", "unused")
    hy = HyMem(config, llm=client)
    chunk = _seed_chunk(hy, chunk_id=f"empty-{table}")
    key = _persist_projection(
        hy, chunk, client, markers=[("style", "Always answer tersely")],
    )
    _materialize_markers(hy, key)
    hy.conn.execute("PRAGMA ignore_check_constraints=ON")
    hy.conn.execute(f"DROP TRIGGER {guard}")
    hy.conn.execute(f"UPDATE {table} SET {column}='' ")
    hy.conn.execute("PRAGMA ignore_check_constraints=OFF")
    hy.close()

    with pytest.raises(RuntimeError, match=error):
        HyMem(config, llm=_DeclaredLLM("empty-policy", "unused")).conn


def test_forged_manual_hint_domain_is_rejected_on_reopen(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False)
    hy = HyMem(config)
    hy.set_entity_type("tool", "tool", confidence=0.5)
    hy.conn.execute("DROP TRIGGER entity_type_authority_update_guard")
    hy.conn.execute(
        "UPDATE entity_types SET confidence=7 WHERE entity_canonical='tool'"
    )
    hy.close()

    with pytest.raises(RuntimeError, match="entity hint authority row"):
        HyMem(config).conn


def test_public_rule_trigger_validation_self_exports_and_reopens(cfg, tmp_path):
    config = replace(cfg, aggregation_nodes_enabled=False)
    hy = HyMem(config)
    for triggers in ([], ["!!!"], [3]):
        with pytest.raises(ValueError, match="trigger"):
            hy.add_rule(
                "Always test",
                scope="contextual",
                trigger_entities=triggers,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="always-on"):
        hy.add_rule("Always test", trigger_entities=["redis"])
    hy.add_rule(
        "Always test",
        scope="contextual",
        trigger_entities=["Redis Service"],
    )
    first = tmp_path / "valid-rule.jsonl"
    hy.export(first)
    hy.close()

    reopened = HyMem(config)
    try:
        assert _rows(
            reopened.conn, "SELECT trigger_entities FROM current_rules",
        ) == [('["redis_service"]',)]
        reopened.export(tmp_path / "valid-rule-reopen.jsonl")
    finally:
        reopened.close()


def test_direct_invalid_rule_trigger_is_rejected_on_reopen(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False)
    hy = HyMem(config)
    rule_id = hy.add_rule(
        "Always test",
        scope="contextual",
        trigger_entities=["redis"],
    )
    hy.conn.execute("DROP TRIGGER rule_domain_update_guard")
    hy.conn.execute(
        "UPDATE rules SET trigger_entities='[\"\"]' WHERE id=?", (rule_id,),
    )
    hy.close()
    with pytest.raises(RuntimeError, match="rule domain"):
        HyMem(config).conn


def test_old_profile_policy_is_hidden_then_replayed(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False)
    client = _DeclaredLLM("profile-policy-replay", "unused")
    hy = HyMem(config, llm=client)
    try:
        chunk = _seed_chunk(hy, chunk_id="profile-policy-replay")
        key = _persist_projection(
            hy, chunk, client,
            markers=[("style", "Always answer tersely")],
        )
        with core_db.transaction(hy.conn):
            phase2.consolidate_profile(
                hy.conn, config, phase1_generation_key=key,
            )
        marker_id = hy.conn.execute(
            "SELECT id FROM behavioral_markers WHERE chunk_id=?", (chunk.id,),
        ).fetchone()[0]
        with core_db.evidence_mutation(hy.conn):
            hy.conn.execute(
                "DELETE FROM profile_marker_decisions WHERE marker_id=?",
                (marker_id,),
            )
            hy.conn.execute(
                "INSERT INTO profile_marker_decisions("
                "marker_id,phase1_generation_key,profile_policy_key,decision,"
                "profile_entry_id) SELECT marker_id,phase1_generation_key,"
                "'marker-profile-materialization-v3',decision,profile_entry_id "
                "FROM (SELECT ? AS marker_id,? AS phase1_generation_key,"
                "'materialized' AS decision,profile_entry_id FROM "
                "profile_entry_marker_evidence WHERE marker_id=?)",
                (marker_id, key, marker_id),
            )
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM current_profile_entries"
        ).fetchone()[0] == 0
        with core_db.transaction(hy.conn):
            phase2.consolidate_profile(
                hy.conn, config, phase1_generation_key=key,
            )
        assert hy.conn.execute(
            "SELECT profile_policy_key FROM profile_marker_decisions "
            "WHERE marker_id=?", (marker_id,),
        ).fetchone()[0] == phase2.PROFILE_MATERIALIZATION_POLICY_KEY
        assert _rows(
            hy.conn, "SELECT text FROM current_profile_entries",
        ) == [("Always answer tersely",)]
    finally:
        hy.close()


def test_stamped_v54_rejects_extra_unique_auxiliary_index(cfg):
    hy = HyMem(cfg)
    hy.conn.execute(
        "CREATE UNIQUE INDEX evil_aux_unique ON "
        "entity_type_observations(entity_canonical)"
    )
    hy.close()
    with pytest.raises(RuntimeError, match="auxiliary domain is incomplete"):
        HyMem(cfg).conn


@pytest.mark.parametrize("table", ["profile_entries", "rules"])
def test_stamped_v54_rejects_nocase_text_unique_identity(cfg, tmp_path, table):
    config = replace(cfg, root=tmp_path / f"nocase-{table}")
    hy = HyMem(config)
    conn = hy.conn
    core_db._drop_phase1_auxiliary_views(conn)
    conn.execute("PRAGMA foreign_keys=OFF")
    if table == "profile_entries":
        conn.execute("DROP TABLE profile_marker_decisions")
        conn.execute("DROP TABLE profile_entry_marker_evidence")
        conn.execute("DROP TABLE profile_entries")
        conn.execute(
            "CREATE TABLE profile_entries("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "kind TEXT NOT NULL CHECK(kind IN "
            "('preference','avoidance','style','context')),"
            "text TEXT NOT NULL COLLATE NOCASE UNIQUE,"
            "pos_evidence INTEGER NOT NULL DEFAULT 1,"
            "neg_evidence INTEGER NOT NULL DEFAULT 0,"
            "first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "source TEXT NOT NULL DEFAULT 'legacy_unattributed' CHECK(source IN "
            "('user','agent_inferred','legacy_unattributed')))"
        )
    else:
        conn.execute("DROP TABLE rule_marker_decisions")
        conn.execute("DROP TABLE rule_marker_evidence")
        conn.execute("DROP TABLE rules")
        conn.execute(
            "CREATE TABLE rules("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "text TEXT NOT NULL COLLATE NOCASE UNIQUE,"
            "scope TEXT NOT NULL DEFAULT 'always_on' CHECK(scope IN "
            "('always_on','contextual')),"
            "trigger_entities TEXT NOT NULL DEFAULT '[]',"
            "source TEXT NOT NULL DEFAULT 'user' CHECK(source IN "
            "('user','agent_inferred')),"
            "pos_evidence INTEGER NOT NULL DEFAULT 1,"
            "neg_evidence INTEGER NOT NULL DEFAULT 0,"
            "valid_at TIMESTAMP,invalid_at TIMESTAMP,"
            "status TEXT NOT NULL DEFAULT 'active' CHECK(status IN "
            "('active','retracted')),"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
    conn.execute("PRAGMA foreign_keys=ON")
    hy.close()

    with pytest.raises(RuntimeError, match="auxiliary domain is incomplete"):
        HyMem(config).conn


def test_stamped_v54_rejects_nocase_composite_primary_identity(cfg):
    hy = HyMem(cfg)
    conn = hy.conn
    core_db._drop_phase1_auxiliary_views(conn)
    for name in (
        "entity_type_observation_insert_guard",
        "entity_type_observation_update_guard",
        "entity_type_observation_delete_guard",
    ):
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute("DROP TABLE entity_type_observations")
    conn.execute(
        "CREATE TABLE entity_type_observations("
        "chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,"
        "entity_canonical TEXT NOT NULL COLLATE NOCASE CHECK("
        "length(trim(entity_canonical))>0 AND "
        "hymem_entity_canonical_is_normalized(entity_canonical)=1),"
        "type TEXT NOT NULL CHECK(length(trim(type))>0),"
        "confidence REAL NOT NULL DEFAULT 1.0 CHECK(typeof(confidence) IN "
        "('integer','real') AND confidence>=0.0 AND confidence<=1.0),"
        "phase1_generation_key TEXT NOT NULL REFERENCES "
        "phase1_generations(generation_key) ON DELETE RESTRICT,"
        "observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "PRIMARY KEY(chunk_id,entity_canonical,type,phase1_generation_key))"
    )
    hy.close()
    with pytest.raises(RuntimeError, match="auxiliary domain is incomplete"):
        HyMem(cfg).conn


def test_stamped_v54_missing_base_table_fails_before_bootstrap_recreates_it(cfg):
    hy = HyMem(cfg)
    conn = hy.conn
    core_db._drop_phase1_auxiliary_views(conn)
    conn.execute("DROP TABLE entity_types")
    hy.close()

    reopened = core_db.connect(cfg.db_path)
    try:
        with pytest.raises(RuntimeError, match="auxiliary domain is incomplete"):
            core_db.initialize(reopened)
        assert reopened.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity_types'"
        ).fetchone() is None
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("incoming_decision", "incoming_kind"),
    [
        ("materialized", "style"),
        ("manual_authority", "style"),
        ("identity_conflict", "context"),
    ],
)
def test_v14_profile_collision_maps_every_incoming_decision_to_user_authority(
    cfg, tmp_path, incoming_decision, incoming_kind,
):
    statement = "Always answer tersely"
    donor_config = replace(
        cfg,
        root=tmp_path / f"donor-{incoming_decision}",
        aggregation_nodes_enabled=False,
    )
    target_config = replace(
        cfg,
        root=tmp_path / f"target-{incoming_decision}",
        aggregation_nodes_enabled=False,
    )
    client = _DeclaredLLM("portable-profile-collision", "unused")
    donor = HyMem(donor_config, llm=client)
    target = HyMem(
        target_config,
        llm=_DeclaredLLM("portable-profile-collision", "unused"),
    )
    path = tmp_path / f"{incoming_decision}.jsonl"
    try:
        if incoming_decision == "manual_authority":
            donor.conn.execute(
                "INSERT INTO profile_entries(kind,text,source) "
                "VALUES ('style',?,'user')", (statement,),
            )
        elif incoming_decision == "identity_conflict":
            donor.conn.execute(
                "INSERT INTO profile_entries(kind,text,source) "
                "VALUES ('context',?,'agent_inferred')", (statement,),
            )
        chunk = _seed_chunk(donor, chunk_id="portable-profile-collision")
        key = _persist_projection(
            donor, chunk, client, markers=[("style", statement)],
        )
        _materialize_profile_only(donor, key)
        assert donor.conn.execute(
            "SELECT decision FROM profile_marker_decisions"
        ).fetchone()[0] == incoming_decision
        donor.export(path)

        target.conn.execute(
            "INSERT INTO profile_entries(kind,text,source) VALUES (?,?,'user')",
            (incoming_kind, statement),
        )
        target.import_(path)
        assert _rows(
            target.conn,
            "SELECT decision FROM profile_marker_decisions",
        ) == [("manual_authority",)]
        assert target.conn.execute(
            "SELECT COUNT(*) FROM profile_entry_marker_evidence"
        ).fetchone()[0] == 0
        assert _rows(
            target.conn,
            "SELECT kind,text,source FROM profile_entries",
        ) == [(incoming_kind, statement, "user")]
        assert not any(target.import_(path).values())
    finally:
        donor.close()
        target.close()


def test_v14_current_profile_semantic_collision_rejects_atomically(cfg, tmp_path):
    statement = "Always answer tersely"
    client_a = _DeclaredLLM("profile-divergence", "unused")
    donor = HyMem(
        replace(cfg, root=tmp_path / "profile-divergence-donor"),
        llm=client_a,
    )
    target = HyMem(
        replace(cfg, root=tmp_path / "profile-divergence-target"),
        llm=_DeclaredLLM("profile-divergence", "unused"),
    )
    path = tmp_path / "profile-divergence.jsonl"
    try:
        donor.conn.execute(
            "INSERT INTO profile_entries(kind,text,source) "
            "VALUES ('context',?,'agent_inferred')", (statement,),
        )
        donor_chunk = _seed_chunk(donor, chunk_id="profile-divergence")
        donor_key = _persist_projection(
            donor, donor_chunk, client_a, markers=[("style", statement)],
        )
        _materialize_profile_only(donor, donor_key)
        assert donor.conn.execute(
            "SELECT decision FROM profile_marker_decisions"
        ).fetchone()[0] == "identity_conflict"

        target_chunk = _seed_chunk(target, chunk_id="profile-divergence")
        target_key = _persist_projection(
            target,
            target_chunk,
            _DeclaredLLM("profile-divergence", "unused"),
            markers=[("style", statement)],
        )
        _materialize_profile_only(target, target_key)
        assert target.conn.execute(
            "SELECT decision FROM profile_marker_decisions"
        ).fetchone()[0] == "materialized"
        _align_portable_seed_parent_clocks(donor, target)
        donor.export(path)
        before = "\n".join(target.conn.iterdump())
        with pytest.raises(ValueError, match="profile_entry collides"):
            target.import_(path)
        assert "\n".join(target.conn.iterdump()) == before
    finally:
        donor.close()
        target.close()


def test_v14_equal_manual_property_merges_latest_clock_both_directions(
    cfg, tmp_path,
):
    left = HyMem(replace(cfg, root=tmp_path / "manual-clock-left"))
    right = HyMem(replace(cfg, root=tmp_path / "manual-clock-right"))
    left_path = tmp_path / "manual-clock-left.jsonl"
    right_path = tmp_path / "manual-clock-right.jsonl"
    try:
        left.set_entity_property("app", "env", "prod")
        right.set_entity_property("app", "env", "prod")
        left.conn.execute(
            "UPDATE entity_properties SET updated_at='2026-01-01T00:00:00Z'"
        )
        right.conn.execute(
            "UPDATE entity_properties SET updated_at='2026-02-01T00:00:00Z'"
        )
        left.export(left_path)
        right.export(right_path)

        assert not any(left.import_(right_path).values())
        assert not any(right.import_(left_path).values())
        expected = [("app", "env", "prod", "2026-02-01T00:00:00Z", "user")]
        query = (
            "SELECT entity_canonical,key,value,updated_at,origin "
            "FROM entity_properties"
        )
        assert _rows(left.conn, query) == expected
        assert _rows(right.conn, query) == expected
        assert not any(left.import_(right_path).values())
        assert not any(right.import_(left_path).values())
    finally:
        left.close()
        right.close()


def test_v14_same_manual_profile_and_rule_converge_commutatively(cfg, tmp_path):
    left = HyMem(replace(cfg, root=tmp_path / "manual-authority-left"))
    right = HyMem(replace(cfg, root=tmp_path / "manual-authority-right"))
    left_initial = tmp_path / "manual-authority-left-initial.jsonl"
    right_initial = tmp_path / "manual-authority-right-initial.jsonl"
    left_final = tmp_path / "manual-authority-left-final.jsonl"
    right_final = tmp_path / "manual-authority-right-final.jsonl"
    try:
        left.conn.execute(
            "INSERT INTO profile_entries(kind,text,pos_evidence,neg_evidence,"
            "first_seen,last_updated,source) VALUES "
            "('style','Prefer terse replies',2,0,"
            "'2026-02-01T00:00:00Z','2026-02-02T00:00:00Z','user')"
        )
        right.conn.execute(
            "INSERT INTO profile_entries(kind,text,pos_evidence,neg_evidence,"
            "first_seen,last_updated,source) VALUES "
            "('style','Prefer terse replies',1,3,"
            "'2026-01-01T00:00:00Z','2026-03-01T00:00:00Z','user')"
        )
        left.add_rule("Always test")
        right.add_rule("Always test")
        left.conn.execute(
            "UPDATE rules SET pos_evidence=4,neg_evidence=0,"
            "valid_at='2026-02-01T00:00:00Z',created_at='2026-02-01T00:00:00Z'"
        )
        right.conn.execute(
            "UPDATE rules SET pos_evidence=1,neg_evidence=2,"
            "valid_at='2026-01-01T00:00:00Z',"
            "invalid_at='2026-03-01T00:00:00Z',status='retracted',"
            "created_at='2026-01-01T00:00:00Z'"
        )
        left.export(left_initial)
        right.export(right_initial)

        left.import_(right_initial)
        right.import_(left_initial)
        assert _rows(
            left.conn,
            "SELECT kind,text,pos_evidence,neg_evidence,first_seen,"
            "last_updated,source FROM profile_entries",
        ) == _rows(
            right.conn,
            "SELECT kind,text,pos_evidence,neg_evidence,first_seen,"
            "last_updated,source FROM profile_entries",
        ) == [(
            "style", "Prefer terse replies", 2, 3,
            "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z", "user",
        )]
        assert _rows(
            left.conn,
            "SELECT text,scope,trigger_entities,source,pos_evidence,"
            "neg_evidence,valid_at,invalid_at,status,created_at FROM rules",
        ) == _rows(
            right.conn,
            "SELECT text,scope,trigger_entities,source,pos_evidence,"
            "neg_evidence,valid_at,invalid_at,status,created_at FROM rules",
        ) == [(
            "Always test", "always_on", "[]", "user", 4, 2,
            "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z",
            "retracted", "2026-01-01T00:00:00Z",
        )]

        left.export(left_final)
        right.export(right_final)
        assert left_final.read_bytes() == right_final.read_bytes()
        assert not any(left.import_(right_final).values())
        assert not any(right.import_(left_final).values())
    finally:
        left.close()
        right.close()


def test_v14_same_user_rule_semantic_divergence_rejects_atomically(cfg, tmp_path):
    donor = HyMem(replace(cfg, root=tmp_path / "rule-divergence-donor"))
    target = HyMem(replace(cfg, root=tmp_path / "rule-divergence-target"))
    path = tmp_path / "rule-divergence.jsonl"
    try:
        donor.add_rule(
            "Follow project conventions",
            scope="contextual",
            trigger_entities=["project"],
        )
        target.add_rule("Follow project conventions")
        donor.export(path)
        before = "\n".join(target.conn.iterdump())
        with pytest.raises(ValueError, match="rule collides"):
            target.import_(path)
        assert "\n".join(target.conn.iterdump()) == before
    finally:
        donor.close()
        target.close()


def test_v14_inferred_profile_and_rule_union_distinct_marker_histories(
    cfg, tmp_path,
):
    config_left = replace(
        cfg,
        root=tmp_path / "inferred-union-left",
        aggregation_nodes_enabled=False,
        rules_extraction_mode="lexical",
    )
    config_right = replace(
        cfg,
        root=tmp_path / "inferred-union-right",
        aggregation_nodes_enabled=False,
        rules_extraction_mode="lexical",
    )
    client_left = _DeclaredLLM("inferred-union", "unused")
    client_right = _DeclaredLLM("inferred-union", "unused")
    left = HyMem(config_left, llm=client_left)
    right = HyMem(config_right, llm=client_right)
    left_path = tmp_path / "inferred-union-left.jsonl"
    right_path = tmp_path / "inferred-union-right.jsonl"
    try:
        left_chunk = _seed_chunk(left, chunk_id="inferred-union-left")
        left_key = _persist_projection(
            left,
            left_chunk,
            client_left,
            markers=[("style", "Always answer tersely")],
        )
        _materialize_markers(left, left_key)

        right_chunk = _seed_chunk(right, chunk_id="inferred-union-right")
        right_key = _persist_projection(
            right,
            right_chunk,
            client_right,
            markers=[("style", "Always answer tersely")],
        )
        _materialize_markers(right, right_key)
        _align_portable_seed_parent_clocks(left, right)
        left.export(left_path)
        right.export(right_path)

        left.import_(right_path)
        right.import_(left_path)
        for store in (left, right):
            assert _rows(
                store.conn,
                "SELECT text,pos_evidence,source FROM current_profile_entries",
            ) == [("Always answer tersely", 1, "agent_inferred")]
            assert _rows(
                store.conn,
                "SELECT text,pos_evidence,source FROM current_rules",
            ) == [("Always answer tersely", 1, "agent_inferred")]
            assert store.conn.execute(
                "SELECT COUNT(*) FROM profile_entry_marker_evidence"
            ).fetchone()[0] == 2
            assert store.conn.execute(
                "SELECT COUNT(*) FROM rule_marker_evidence"
            ).fetchone()[0] == 2
            assert store.dream_status()["pending_chunks"] == 1

        # Import preserves exact history but cannot mint the destination's
        # processed acknowledgement. Each side must execute the other chunk
        # locally before both independently sourced markers become current.
        _persist_projection(
            left,
            right_chunk,
            client_left,
            markers=[("style", "Always answer tersely")],
        )
        _materialize_markers(left, left_key)
        _persist_projection(
            right,
            left_chunk,
            client_right,
            markers=[("style", "Always answer tersely")],
        )
        _materialize_markers(right, right_key)
        for store in (left, right):
            assert _rows(
                store.conn,
                "SELECT text,pos_evidence,source FROM current_profile_entries",
            ) == [("Always answer tersely", 2, "agent_inferred")]
            assert _rows(
                store.conn,
                "SELECT text,pos_evidence,source FROM current_rules",
            ) == [("Always answer tersely", 2, "agent_inferred")]
            assert store.dream_status()["pending_chunks"] == 0

        assert not any(left.import_(right_path).values())
        assert not any(right.import_(left_path).values())
        assert left.dream_status()["pending_chunks"] == 0
        assert right.dream_status()["pending_chunks"] == 0
    finally:
        left.close()
        right.close()


def test_v14_auxiliary_roundtrip_reexport_is_exact_and_import_idempotent(
    cfg, tmp_path,
):
    source_config = replace(
        cfg,
        root=tmp_path / "v14-roundtrip-source",
        aggregation_nodes_enabled=False,
        rules_extraction_mode="lexical",
    )
    target_config = replace(
        cfg,
        root=tmp_path / "v14-roundtrip-target",
        aggregation_nodes_enabled=False,
        rules_extraction_mode="lexical",
    )
    source_client = _DeclaredLLM("v14-roundtrip", "unused")
    source = HyMem(source_config, llm=source_client)
    target = HyMem(
        target_config, llm=_DeclaredLLM("v14-roundtrip", "unused")
    )
    first = tmp_path / "v14-roundtrip-first.jsonl"
    second = tmp_path / "v14-roundtrip-second.jsonl"
    try:
        source.set_entity_type("operator", "person", confidence=0.75)
        source.set_entity_property("operator", "timezone", "Europe/Amsterdam")
        chunk = _seed_chunk(source, chunk_id="v14-roundtrip")
        key = _persist_projection(
            source,
            chunk,
            source_client,
            type_hints={"App": "service"},
            property_hints={"App": {"runtime": "rust"}},
            entity_mentions=["App"],
            markers=[("style", "Always answer tersely")],
        )
        _materialize_markers(source, key)
        source.export(first)

        target.import_(first)
        # Import restores exact history, never a counterfeit local cache ack.
        assert target.dream_status()["pending_chunks"] == 1
        assert not any(target.import_(first).values())
        target.export(second)
        assert second.read_bytes() == first.read_bytes()
    finally:
        source.close()
        target.close()


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("entity_type_observation", "entity_canonical"),
        ("entity_property_observation", "entity_canonical"),
        ("entity_mention_observation", "entity_canonical"),
        ("rule", "trigger_entities"),
    ],
)
def test_v14_noncanonical_wire_identity_rolls_back_before_mutation(
    cfg, tmp_path, kind, field,
):
    source_config = replace(
        cfg,
        root=tmp_path / f"wire-source-{kind}",
        aggregation_nodes_enabled=False,
    )
    target_config = replace(cfg, root=tmp_path / f"wire-target-{kind}")
    client = _DeclaredLLM("noncanonical-wire", "unused")
    source = HyMem(source_config, llm=client)
    target = HyMem(target_config)
    path = tmp_path / f"noncanonical-{kind}.jsonl"
    try:
        source.add_rule(
            "Use app conventions",
            scope="contextual",
            trigger_entities=["app"],
        )
        chunk = _seed_chunk(source, chunk_id="noncanonical-wire")
        _persist_projection(
            source,
            chunk,
            client,
            type_hints={"app": "service"},
            property_hints={"app": {"runtime": "rust"}},
            entity_mentions=["app"],
        )
        source.export(path)

        def mutate(body):
            record = next(item["record"] for item in body if item["type"] == kind)
            record[field] = (
                json.dumps(["Bad ID"])
                if field == "trigger_entities" else "Bad ID"
            )

        _rewrite_jsonl(path, mutate)
        before = "\n".join(target.conn.iterdump())
        with pytest.raises(ValueError, match="canonical|trigger|lineage"):
            target.import_(path)
        assert "\n".join(target.conn.iterdump()) == before
    finally:
        source.close()
        target.close()


def test_v54_wrong_partial_index_heals_and_malformed_migration_never_stamps(cfg):
    conn = core_db.connect(cfg.db_path)
    core_db.initialize(conn)
    conn.execute("DROP INDEX idx_behavioral_marker_generation_identity")
    conn.execute(
        "CREATE UNIQUE INDEX idx_behavioral_marker_generation_identity ON "
        "behavioral_markers(chunk_id,phase1_generation_key,kind,statement) "
        "WHERE phase1_generation_key IS NULL"
    )
    conn.close()
    healed = core_db.connect(cfg.db_path)
    core_db.initialize(healed)
    sql = healed.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND "
        "name='idx_behavioral_marker_generation_identity'"
    ).fetchone()[0]
    assert "WHERE phase1_generation_key IS NOT NULL" in sql

    core_db._drop_phase1_auxiliary_views(healed)
    for name in (
        "entity_type_observation_insert_guard",
        "entity_type_observation_update_guard",
        "entity_type_observation_delete_guard",
    ):
        healed.execute(f"DROP TRIGGER IF EXISTS {name}")
    for name in (
        "idx_entity_type_observations_lookup",
        "idx_entity_type_observations_entity",
    ):
        healed.execute(f"DROP INDEX IF EXISTS {name}")
    healed.execute("DROP TABLE entity_type_observations")
    healed.execute(
        "CREATE TABLE entity_type_observations("
        "chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,"
        "entity_canonical TEXT NOT NULL,type TEXT NOT NULL,"
        "confidence REAL NOT NULL DEFAULT 1.0,"
        "phase1_generation_key TEXT NOT NULL REFERENCES "
        "phase1_generations(generation_key) ON DELETE RESTRICT,"
        "observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    healed.execute("UPDATE schema_meta SET value='53' WHERE key='schema_version'")
    with pytest.raises(RuntimeError, match="v54.*malformed"):
        core_db._run_migrations(healed)
    assert core_db.schema_version(healed) == 53
    assert [row["pk"] for row in healed.execute(
        "PRAGMA table_info(entity_type_observations)"
    )] == [0, 0, 0, 0, 0, 0]
    healed.close()


@pytest.mark.parametrize("stamped", [False, True], ids=["pre-stamp", "stamped"])
@pytest.mark.parametrize(
    "corruption",
    ["extra-check", "extra-trigger", "generated-column", "nocase-value"],
)
def test_v54_rejects_narrowing_checks_and_unowned_authority_triggers(
    cfg, tmp_path, stamped, corruption,
):
    path = tmp_path / f"v54-{corruption}-{stamped}.sqlite"
    conn = core_db.connect(path)
    core_db.initialize(conn)
    if corruption == "extra-check":
        _replace_type_observation_with_extra_check(conn)
    elif corruption == "generated-column":
        _replace_entity_types_with_generated_gate(conn)
    elif corruption == "nocase-value":
        _replace_entity_properties_with_nocase_value(conn)
        conn.execute(
            "INSERT INTO entity_properties("
            "entity_canonical,key,value,origin) VALUES ('app','runtime','Rust','user')"
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM entity_properties WHERE value='rust'"
        ).fetchone()[0] == 1
    else:
        conn.execute(
            "CREATE TRIGGER evil_entity_type_side_effect "
            "AFTER INSERT ON entity_types BEGIN SELECT 1; END"
        )
    if not stamped:
        conn.execute(
            "UPDATE schema_meta SET value='53' WHERE key='schema_version'"
        )
    conn.close()

    reopened = core_db.connect(path)
    with pytest.raises(RuntimeError, match="v54.*(?:malformed|incomplete)"):
        core_db.initialize(reopened)
    assert core_db.schema_version(reopened) == (
        core_db.EXPECTED_SCHEMA_VERSION if stamped else 53
    )
    reopened.close()


def test_v14_alias_split_fails_before_mutation(cfg, tmp_path):
    donor_cfg = replace(
        cfg, root=tmp_path / "donor", aggregation_nodes_enabled=False,
    )
    target_cfg = replace(cfg, root=tmp_path / "target")
    client = _DeclaredLLM("alias-split", "unused")
    donor = HyMem(donor_cfg, llm=client)
    target = HyMem(target_cfg)
    path = tmp_path / "alias-split.jsonl"
    try:
        donor.set_entity_type("app", "tool")
        donor.set_entity_property("app", "owner", "team")
        donor.add_rule(
            "Use app conventions",
            scope="contextual",
            trigger_entities=["app"],
        )
        chunk = _seed_chunk(donor, chunk_id="alias-split-aux")
        _persist_projection(
            donor,
            chunk,
            client,
            type_hints={"app": "service"},
            property_hints={"app": {"runtime": "rust"}},
            entity_mentions=["app"],
        )
        donor.export(path)
        target.register_alias("app", "application")
        before = "\n".join(target.conn.iterdump())
        with pytest.raises(
            ValueError,
            match="(?:collides with target|resolves differently)",
        ):
            target.import_(path)
        assert "\n".join(target.conn.iterdump()) == before
        assert target.conn.execute(
            "SELECT COUNT(*) FROM current_entity_types"
        ).fetchone()[0] == 0
    finally:
        donor.close()
        target.close()


@pytest.mark.parametrize(
    "owner", ["manual-property", "manual-type", "rule-trigger", "observation"]
)
def test_v14_incoming_alias_cannot_capture_target_canonical_owner(
    cfg, tmp_path, owner,
):
    donor = HyMem(replace(cfg, root=tmp_path / f"alias-capture-donor-{owner}"))
    target = HyMem(
        replace(
            cfg,
            root=tmp_path / f"alias-capture-target-{owner}",
            aggregation_nodes_enabled=False,
        )
    )
    path = tmp_path / f"alias-capture-{owner}.jsonl"
    try:
        donor.register_alias("foo", "bar")
        donor.export(path)
        if owner == "manual-property":
            target.set_entity_property("foo", "env", "prod")
        elif owner == "manual-type":
            target.set_entity_type("foo", "service")
        elif owner == "rule-trigger":
            target.add_rule(
                "Use Foo conventions",
                scope="contextual",
                trigger_entities=["foo"],
            )
        else:
            client = _DeclaredLLM("alias-owner", "unused")
            chunk = _seed_chunk(target, chunk_id="alias-owner-observation")
            _persist_projection(
                target,
                chunk,
                client,
                type_hints={"foo": "service"},
                entity_mentions=["foo"],
            )
        before = "\n".join(target.conn.iterdump())
        with pytest.raises(ValueError, match="(?:captures target|collides with target)"):
            target.import_(path)
        assert "\n".join(target.conn.iterdump()) == before
        assert canonicalize.resolve(target.conn, "foo") == "foo"
        self_export = tmp_path / f"alias-capture-target-self-{owner}.jsonl"
        target.export(self_export)
        assert self_export.exists()
    finally:
        donor.close()
        target.close()


def test_v14_stale_aux_history_does_not_invalidate_live_target_gate(cfg, tmp_path):
    donor_cfg = replace(cfg, root=tmp_path / "donor-live", aggregation_nodes_enabled=False)
    target_cfg = replace(cfg, root=tmp_path / "target-live", aggregation_nodes_enabled=False)
    donor_client = _DeclaredLLM("producer-a", "unused")
    target_client = _DeclaredLLM("producer-b", "unused")
    donor = HyMem(donor_cfg, llm=donor_client)
    target = HyMem(target_cfg, llm=target_client)
    path = tmp_path / "stale-history.jsonl"
    try:
        donor_chunk = _seed_chunk(donor, chunk_id="portable-shared-chunk")
        _persist_projection(
            donor,
            donor_chunk,
            donor_client,
            type_hints={"app": "project"},
        )
        donor.export(path)

        # First restore gives both stores byte-identical portable parents.  It
        # deliberately leaves the chunk pending; B then earns a local current
        # gate before the older A snapshot is imported again.
        target.import_(path)
        target_chunk = donor_chunk
        key_b = _persist_projection(
            target,
            target_chunk,
            target_client,
            type_hints={"app": "service"},
        )
        with core_db.evidence_mutation(target.conn):
            target.conn.execute(
                "UPDATE kg_claim_extraction_outcomes SET "
                "succeeded_at=strftime('%Y-%m-%dT%H:%M:%fZ',"
                "succeeded_at,'+1 second') WHERE chunk_id=?",
                (target_chunk.id,),
            )
        assert target.dream_status()["pending_chunks"] == 0
        target.import_(path)
        processed = target.conn.execute(
            "SELECT phase1_generation_key FROM processed_chunks WHERE chunk_id=?",
            (target_chunk.id,),
        ).fetchone()
        assert processed is not None and processed[0] == key_b
        assert target.dream_status()["pending_chunks"] == 0
        assert _rows(
            target.conn,
            "SELECT entity_canonical,type FROM current_entity_types",
        ) == [("app", "service")]
    finally:
        donor.close()
        target.close()
