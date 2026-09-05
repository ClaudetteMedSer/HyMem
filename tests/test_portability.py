"""Tests for memory export / import (improv item G).

`hy.export(path)` writes the canonical state as JSON Lines; `hy.import_(path)`
loads it back, additive and idempotent, with sessions ahead of their
dependents and FTS shadow tables kept in sync.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from hymem import HyMem, HyMemConfig, redaction, portability
from hymem.core import db as core_db
from hymem.core.message_records import message_content_hash
from hymem.dreaming.canonicalize import normalize
from hymem.dreaming import evidence as evidence_ledger
from hymem.dreaming.bitemporal import record_lifecycle_event
from hymem.dreaming.value_supersession import supersede_competing_values
from hymem.dreaming.message_coverage import (
    encode_message_record,
    record_message_coverage,
)
from hymem.dreaming.lossless import (
    LOSSLESS_COVERAGE_VERSION,
    coverage_chunk_id,
    validate_message_coverage_artifact,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming import phase1
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.triples import Triple
from hymem.dreaming.user_profile import (
    PROFILE_PROMPT_VERSION,
    ProfileExtraction,
    persist_user_profile,
    profile_config_version,
    profile_retry_policy_version,
)
from hymem.extraction.llm import StubLLMClient

_EXPECTED = {
    "session": 1, "peer": 0, "session_peer": 0,
    "chunk": 1, "message_retention_coverage": 1, "episode": 1,
    "episode_source_occurrence": 0,
    "user_profile_fact": 1,
    "procedure": 1, "edge": 1, "profile_entry": 1,
    "entity_alias": 0,
    "chunk_source_manifest": 0, "chunk_message_source": 0,
    "claim_extraction_outcome": 0,
    "edge_evidence": 0, "edge_evidence_signal": 1,
    "claim_observation": 0, "edge_lifecycle": 1,
    "lifecycle_dependency": 0,
    "fact_extraction_outcome": 0,
    "fact_extraction_source_occurrence": 0,
    "fact_extraction_revision": 0,
    "narrative_fact": 0,
    "narrative_fact_lifecycle": 0,
}


def _rewrite_v6_export(path, mutate) -> None:
    """Mutate body records and replace the v6 checksum over their exact bytes."""
    objects = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert objects[-1]["type"] == "_end"
    body, end = objects[:-1], objects[-1]
    mutate(body)
    if isinstance(end.get("counts"), dict):
        end["counts"] = {
            kind: sum(obj.get("type") == kind for obj in body)
            for kind in end["counts"]
        }
    encoded_body = [
        json.dumps(obj, ensure_ascii=False) + "\n" for obj in body
    ]
    end["sha256"] = hashlib.sha256(
        "".join(encoded_body).encode("utf-8")
    ).hexdigest()
    path.write_text(
        "".join(encoded_body) + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _downgrade_to_true_v6(path) -> None:
    objects = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_kinds = {row[0] for row in portability._V6_EXPORT_SPEC}
    body = []
    for obj in objects[:-1]:
        if obj["type"] == "_meta":
            obj["version"] = 6
            obj["schema_version"] = 39
            body.append(obj)
        elif obj["type"] in base_kinds:
            obj["record"] = {
                column: obj["record"][column]
                for column in portability._V6_COLS_BY_KIND[obj["type"]]
            }
            body.append(obj)
    encoded = [json.dumps(obj, ensure_ascii=False) + "\n" for obj in body]
    counts = {
        kind: sum(obj["type"] == kind for obj in body) for kind in base_kinds
    }
    end = {
        "type": "_end", "counts": counts,
        "sha256": hashlib.sha256("".join(encoded).encode("utf-8")).hexdigest(),
    }
    path.write_text(
        "".join(encoded) + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _seed(hy: HyMem) -> None:
    conn = hy.conn
    conn.execute("INSERT INTO sessions(id, summary) VALUES ('s1', 'did stuff')")
    content = "we deploy postgres to prod"
    message_id = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s1', 'user', ?)",
        (content,),
    ).lastrowid
    record = encode_message_record(
        message_id=message_id,
        role="user",
        content=content,
    )
    durable_chunk_id = coverage_chunk_id("s1", message_id)
    conn.execute(
        "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
        "salience_reason, text, chunk_kind) "
        "VALUES (?, 's1', ?, ?, 'lossless', ?, 'coverage')",
        (durable_chunk_id, message_id, message_id, record),
    )
    record_message_coverage(
        conn,
        message_id=message_id,
        chunk_id=durable_chunk_id,
        coverage_version=LOSSLESS_COVERAGE_VERSION,
    )
    profile_generation = (
        profile_config_version(max_chars=12000, max_items=16)
        + "|walk="
        + ("a" * 32)
    )
    conn.execute(
        "UPDATE sessions SET coverage_message_id = ?, "
        "profile_prompt_version = ?, profile_cursor_message_id = ?, "
        "profile_cursor_prompt_version = ?, profile_published_generation = ? "
        "WHERE id = 's1'",
        (
            message_id,
            PROFILE_PROMPT_VERSION,
            message_id,
            profile_generation,
            profile_generation,
        ),
    )
    persist_user_profile(
        conn,
        ProfileExtraction(items=[{
            "slot": "location",
            "slot_key": None,
            "value": "Utrecht",
            "evidence_message_id": message_id,
            "confidence": 0.9,
        }]),
    )
    edge_id = int(conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, valid_at, invalid_at) "
        "VALUES ('app', 'uses', 'postgres', 3, '2024-03-15 00:00:00', NULL)"
    ).lastrowid)
    evidence_ledger.capture_unattributed_counts(
        conn, [edge_id], reason="portable test legacy edge"
    )
    record_lifecycle_event(
        conn, edge_id=edge_id, event_key="legacy-state",
        event_kind="legacy_state", direction=1,
        event_at="2024-03-15 00:00:00",
        details="portable test legacy edge",
    )
    conn.execute(
        "UPDATE sessions SET digest_published_generation = 'portable-gen' "
        "WHERE id = 's1'"
    )
    conn.execute(
        "INSERT INTO episodes(id, session_id, title, summary, digest_generation) "
        "VALUES ('e1', 's1', 'Setup', "
        "'Configured the postgres connection pool', 'portable-gen')"
    )
    conn.execute(
        "INSERT INTO procedures(id, session_id, name, description, steps) "
        "VALUES ('p1', 's1', 'Deploy to staging', 'build and push', '[]')"
    )
    conn.execute(
        "INSERT INTO profile_entries(kind, text) VALUES ('preference', 'prefers postgres')"
    )


def _portable_claim_chunk(
    hy: HyMem,
    *,
    chunk_id: str,
    message_id: int,
    role: str,
    content: str,
) -> Chunk:
    chunk = Chunk(
        id=chunk_id,
        session_id="claim-session",
        start_message_id=message_id,
        end_message_id=message_id,
        salience_reason="portable-test",
        text=f"{role}: {content}",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [chunk])
    return chunk


def _persist_portable_claim(
    hy: HyMem,
    chunk: Chunk,
    triples: list[Triple],
    *,
    prompt_version: str,
) -> None:
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn,
            chunk,
            ChunkExtraction(
                triples=triples,
                markers=[],
                claim_sources={source.message_id: source for source in sources},
                source_validated=True,
            ),
            prompt_version=prompt_version,
            cfg=hy.config,
        )


def _seed_named_portable_claim(
    hy: HyMem, *, session_id: str, chunk_id: str, created_at: str,
    subject: str = "app", predicate: str = "uses", object_: str = "redis",
) -> None:
    hy.open_session(session_id)
    content = "The app uses Redis"
    message_id = hy.log_message(
        session_id, "user", content, created_at=created_at
    )
    with core_db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, session_id)
    chunk = Chunk(
        id=chunk_id,
        session_id=session_id,
        start_message_id=message_id,
        end_message_id=message_id,
        salience_reason="portable-merge-test",
        text=f"user: {content}",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        persist_chunks(hy.conn, [chunk])
    sources = phase1._claim_sources_for_chunk(hy.conn, chunk)
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn,
            chunk,
            ChunkExtraction(
                triples=[Triple(
                    subject, predicate, object_, 1,
                    source_message_id=message_id,
                )],
                markers=[],
                claim_sources={source.message_id: source for source in sources},
                source_validated=True,
            ),
            prompt_version="v13",
            cfg=hy.config,
        )


def _seed_shared_claim_artifact(hy: HyMem) -> tuple[Chunk, int]:
    """Create one portable source/manfiest without publishing a claim."""
    hy.open_session("claim-session")
    content = "The service target is 65 percent"
    message_id = hy.log_message(
        "claim-session", "user", content,
        created_at="2025-01-02T01:04:05+00:00",
    )
    with core_db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, "claim-session")
    chunk = _portable_claim_chunk(
        hy, chunk_id="shared-claim", message_id=message_id,
        role="user", content=content,
    )
    return chunk, int(message_id)


def _portable_claim_state(hy: HyMem) -> tuple[list[tuple], ...]:
    return (
        [tuple(row) for row in hy.conn.execute(
            "SELECT subject_canonical,predicate,object_canonical,pos_evidence,"
            "neg_evidence,status,valid_at,invalid_at FROM knowledge_graph "
            "WHERE derived=0 ORDER BY subject_canonical,predicate,object_canonical"
        )],
        [tuple(row) for row in hy.conn.execute(
            "SELECT source_session_id,source_message_id,evidence_kind,revision,"
            "polarity,interpretation_key,is_current,superseded_at,superseded_reason "
            "FROM kg_evidence ORDER BY source_session_id,source_message_id,"
            "evidence_kind,revision"
        )],
        [tuple(row) for row in hy.conn.execute(
            "SELECT chunk_id,source_session_id,source_message_id,evidence_kind,"
            "polarity,prompt_version,prompt_generation,interpretation_key "
            "FROM kg_claim_observations ORDER BY chunk_id"
        )],
        [tuple(row) for row in hy.conn.execute(
            "SELECT lifecycle.event_key,lifecycle.event_kind,lifecycle.direction,"
            "lifecycle.event_at,source.revision FROM kg_edge_lifecycle lifecycle "
            "LEFT JOIN kg_evidence source ON source.id=lifecycle.source_evidence_id "
            "ORDER BY lifecycle.event_key"
        )],
        [tuple(row) for row in hy.conn.execute(
            "SELECT chunk_id,prompt_version,prompt_generation,result_hash "
            "FROM kg_claim_extraction_outcomes ORDER BY chunk_id"
        )],
    )


def test_export_import_roundtrip(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(src)
    out = tmp_path / "export.jsonl"
    assert src.export(out) == _EXPECTED
    src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "dst"))
    try:
        assert dst.import_(out) == _EXPECTED
        # Content survived.
        assert dst.conn.execute(
            "SELECT summary FROM sessions WHERE id = 's1'"
        ).fetchone()["summary"] == "did stuff"
        # Edge is queryable through the timeline API.
        assert any(e.object == "postgres" for e in dst.timeline("app"))
        # Bi-temporal validity interval survived the round-trip (schema v15).
        edge = dst.conn.execute(
            "SELECT valid_at, invalid_at FROM knowledge_graph "
            "WHERE subject_canonical='app' AND object_canonical='postgres'"
        ).fetchone()
        assert edge["valid_at"] == "2024-03-15T00:00:00.000Z"
        assert edge["invalid_at"] is None
        # The raw message is not part of the v3 portable memory export, but its
        # durable proof survives and keeps the lossless backing chunk protected.
        assert dst.conn.execute(
            "SELECT COUNT(*) AS c FROM messages"
        ).fetchone()["c"] == 0
        assert dst.conn.execute(
            "SELECT COUNT(*) AS c FROM message_retention_coverage"
        ).fetchone()["c"] == 1
        profile = dst.profile()
        assert [(entry.slot, entry.value, entry.evidence_message_id) for entry in profile] == [
            ("location", "Utrecht", 1)
        ]
        provenance = dst.conn.execute(
            "SELECT evidence_message_id, source_message_id, source_session_id, "
            "source_created_at, valid_at FROM user_profile"
        ).fetchone()
        assert provenance["evidence_message_id"] is None
        assert provenance["source_message_id"] == 1
        assert provenance["source_session_id"] == "s1"
        assert provenance["source_created_at"] is not None
        assert provenance["valid_at"] is not None
        # Privacy-preserving imports rewind free-form control-plane generation
        # markers instead of persisting potentially secret-bearing text.
        assert dst.conn.execute(
            "SELECT digest_published_generation FROM sessions WHERE id = 's1'"
        ).fetchone()["digest_published_generation"] is None
        assert dst.conn.execute(
            "SELECT COUNT(*) AS c FROM episodes_fts "
            "WHERE episodes_fts MATCH 'configured'"
        ).fetchone()["c"] == 1
        # Coverage artifacts are intentionally filtered from chunk retrieval;
        # normal portable procedures remain searchable.
        ctx = dst.augment("postgres deploy staging")
        assert ctx.fts_hits == []
        assert any(p.name == "Deploy to staging" for p in ctx.procedures)
    finally:
        dst.close()


def test_export_writes_meta_header(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(hy)
    out = tmp_path / "export.jsonl"
    hy.export(out)
    meta = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert meta["type"] == "_meta"
    assert meta["format"] == "hymem-jsonl"
    assert meta["schema_version"] == core_db.EXPECTED_SCHEMA_VERSION
    hy.close()


def test_import_is_idempotent(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(src)
    out = tmp_path / "export.jsonl"
    src.export(out)
    src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "dst"))
    try:
        dst.import_(out)
        second = dst.import_(out)
        assert sum(second.values()) == 0  # nothing new on re-import
        assert dst.conn.execute(
            "SELECT COUNT(*) AS c FROM knowledge_graph"
        ).fetchone()["c"] == 1
        assert dst.conn.execute(
            "SELECT COUNT(*) AS c FROM user_profile"
        ).fetchone()["c"] == 1
    finally:
        dst.close()


def test_v7_claim_history_roundtrips_through_natural_id_maps(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "v7-claim-source"))
    try:
        src.open_session("claim-session")
        content = "The service target is 65 percent"
        message_id = src.log_message(
            "claim-session", "user", content,
            created_at="2025-01-02T03:04:05+02:00",
        )
        with core_db.transaction(src.conn):
            materialize_message_coverage(src.conn, "claim-session")
        older = _portable_claim_chunk(
            src, chunk_id="claim-old", message_id=message_id,
            role="user", content=content,
        )
        newer = _portable_claim_chunk(
            src, chunk_id="claim-new", message_id=message_id,
            role="user", content=content,
        )
        _persist_portable_claim(
            src, older,
            [Triple("service", "has_attribute", "65_percent", 1,
                    temporal_scope="2025", source_message_id=message_id)],
            prompt_version="v13",
        )
        _persist_portable_claim(
            src, newer,
            [Triple("service", "has_attribute", "65_percent", 1,
                    temporal_scope="2026", source_message_id=message_id)],
            prompt_version="v14",
        )
        before = {
            table: [tuple(row) for row in src.conn.execute(f"SELECT * FROM {table}")]
            for table in (
                "kg_evidence", "kg_claim_observations", "kg_edge_lifecycle",
                "kg_lifecycle_dependencies",
            )
        }
        assert len(before["kg_evidence"]) == 2
        assert [row[0] for row in src.conn.execute(
            "SELECT is_current FROM kg_evidence ORDER BY revision"
        ).fetchall()] == [0, 1]
        out = tmp_path / "v7-claim.jsonl"
        counts = src.export(out)
        assert counts["edge_evidence"] == 2
        assert counts["claim_observation"] == 2
        assert counts["edge_lifecycle"] == 2
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "v7-claim-target"))
    try:
        # Shift both local rowid domains. Wire ids must never be inserted as
        # destination ids or used as durable cross-store identity.
        dst.conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical) "
            "VALUES ('unrelated','uses','dummy')"
        )
        dst.conn.execute("INSERT INTO sessions(id) VALUES ('dummy-session')")
        dst.conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text,chunk_kind) VALUES "
            "('dummy-chunk','dummy-session',99,99,'test','dummy','extraction')"
        )
        dummy_edge = dst.conn.execute(
            "SELECT id FROM knowledge_graph WHERE subject_canonical='unrelated'"
        ).fetchone()[0]
        dummy_key = evidence_ledger._interpretation_key(
            polarity=1, evidence_weight=1, weight_source="legacy_default",
            source_role=None, surface_subject=None, surface_object=None,
            value_text=None, value_numeric=None, value_unit=None,
            temporal_scope=None,
        )
        with core_db.evidence_mutation(dst.conn):
            dummy_evidence_id = int(dst.conn.execute(
                "INSERT INTO kg_evidence(edge_id,chunk_id,polarity,interpretation_key) "
                "VALUES (?,?,1,?)", (dummy_edge, "dummy-chunk", dummy_key),
            ).lastrowid)

        first = dst.import_(out)
        assert first["edge_evidence"] == 2
        edge = dst.conn.execute(
            "SELECT id,pos_evidence,valid_at FROM knowledge_graph "
            "WHERE subject_canonical='service'"
        ).fetchone()
        assert edge["id"] != 1
        evidence_rows = dst.conn.execute(
            "SELECT id,revision,is_current,temporal_scope,source_event_at "
            "FROM kg_evidence WHERE edge_id=? ORDER BY revision", (edge["id"],)
        ).fetchall()
        assert all(row["id"] > dummy_evidence_id for row in evidence_rows)
        assert [tuple(row[1:]) for row in evidence_rows] == [
            (1, 0, "2025", "2025-01-02T01:04:05.000Z"),
            (2, 1, "2026", "2025-01-02T01:04:05.000Z"),
        ]
        assert dst.conn.execute(
            "SELECT COUNT(*) FROM kg_claim_observations WHERE edge_id=?",
            (edge["id"],),
        ).fetchone()[0] == 2
        second = dst.import_(out)
        assert sum(second.values()) == 0
    finally:
        dst.close()


def test_v7_same_natural_edge_merges_distinct_histories_commutatively(tmp_path):
    left = HyMem(HyMemConfig(root=tmp_path / "merge-left"))
    right = HyMem(HyMemConfig(root=tmp_path / "merge-right"))
    try:
        _seed_named_portable_claim(
            left, session_id="left-session", chunk_id="left-chunk",
            created_at="2024-01-01T00:00:00Z",
        )
        _seed_named_portable_claim(
            right, session_id="right-session", chunk_id="right-chunk",
            created_at="2025-01-01T00:00:00Z",
        )
        left_wire = tmp_path / "left.jsonl"
        right_wire = tmp_path / "right.jsonl"
        left.export(left_wire)
        right.export(right_wire)

        assert right.import_(left_wire)["edge"] == 0
        assert left.import_(right_wire)["edge"] == 0
        assert right.import_(left_wire)["edge_evidence"] == 0
        assert left.import_(right_wire)["edge_evidence"] == 0

        def semantic_state(hy: HyMem):
            edge = tuple(hy.conn.execute(
                "SELECT subject_canonical,predicate,object_canonical,pos_evidence,"
                "neg_evidence,status,valid_at,invalid_at FROM knowledge_graph "
                "WHERE derived=0"
            ).fetchone())
            revisions = [tuple(row) for row in hy.conn.execute(
                "SELECT source_session_id,source_created_at,polarity,evidence_weight,"
                "is_current FROM kg_evidence ORDER BY source_session_id"
            ).fetchall()]
            events = [tuple(row) for row in hy.conn.execute(
                "SELECT event_key,event_kind,direction,event_at FROM kg_edge_lifecycle "
                "ORDER BY event_key"
            ).fetchall()]
            return edge, revisions, events

        assert semantic_state(left) == semantic_state(right)
        assert semantic_state(left)[0][3:] == (
            4, 0, "active", "2024-01-01T00:00:00.000Z", None,
        )
    finally:
        left.close()
        right.close()


def test_v7_evolving_snapshot_union_is_order_independent(tmp_path):
    source = HyMem(HyMemConfig(root=tmp_path / "lineage-source"))
    try:
        chunk, message_id = _seed_shared_claim_artifact(source)
        _persist_portable_claim(
            source, chunk,
            [Triple("service", "has_attribute", "65_percent", 1,
                    temporal_scope="old", source_message_id=message_id)],
            prompt_version="v13",
        )
        low = tmp_path / "lineage-low.jsonl"
        source.export(low)
        _persist_portable_claim(
            source, chunk,
            [Triple("service", "has_attribute", "65_percent", -1,
                    temporal_scope="new", source_message_id=message_id)],
            prompt_version="v14",
        )
        high = tmp_path / "lineage-high.jsonl"
        source.export(high)
    finally:
        source.close()

    low_high = HyMem(HyMemConfig(root=tmp_path / "lineage-low-high"))
    high_low = HyMem(HyMemConfig(root=tmp_path / "lineage-high-low"))
    try:
        low_high.import_(low)
        low_high.import_(high)
        assert sum(low_high.import_(high).values()) == 0

        high_low.import_(high)
        assert sum(high_low.import_(low).values()) == 0
        assert sum(high_low.import_(high).values()) == 0

        assert _portable_claim_state(low_high) == _portable_claim_state(high_low)
        evidence = low_high.conn.execute(
            "SELECT revision,polarity,is_current FROM kg_evidence ORDER BY revision"
        ).fetchall()
        assert [tuple(row) for row in evidence] == [(1, 1, 0), (2, -1, 1)]
    finally:
        low_high.close()
        high_low.close()


def test_v7_successful_empty_outcome_supersedes_stale_claim_in_both_orders(
    tmp_path,
):
    base = HyMem(HyMemConfig(root=tmp_path / "empty-base"))
    try:
        _chunk, message_id = _seed_shared_claim_artifact(base)
        base_wire = tmp_path / "empty-base.jsonl"
        base.export(base_wire)
    finally:
        base.close()

    asserted = HyMem(HyMemConfig(root=tmp_path / "empty-asserted"))
    empty = HyMem(HyMemConfig(root=tmp_path / "empty-authority"))
    try:
        asserted.import_(base_wire)
        empty.import_(base_wire)
        chunk = Chunk(
            id="shared-claim", session_id="claim-session",
            start_message_id=message_id, end_message_id=message_id,
            salience_reason="portable-test",
            text="user: The service target is 65 percent",
            source_message_ids=(message_id,),
        )
        _persist_portable_claim(
            asserted, chunk,
            [Triple("service", "has_attribute", "65_percent", 1,
                    source_message_id=message_id)],
            prompt_version="v13",
        )
        _persist_portable_claim(
            empty, chunk, [], prompt_version="v14",
        )
        asserted_wire = tmp_path / "empty-asserted.jsonl"
        empty_wire = tmp_path / "empty-authority.jsonl"
        asserted.export(asserted_wire)
        empty.export(empty_wire)
    finally:
        asserted.close()
        empty.close()

    low_high = HyMem(HyMemConfig(root=tmp_path / "empty-low-high"))
    high_low = HyMem(HyMemConfig(root=tmp_path / "empty-high-low"))
    try:
        low_high.import_(asserted_wire)
        low_high.conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,derived) "
            "VALUES ('stale','depends_on','closure',1,'active',1)"
        )
        low_high.import_(empty_wire)
        assert low_high.conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0] == 0
        high_low.import_(empty_wire)
        high_low.import_(asserted_wire)

        for store in (low_high, high_low):
            assert store.conn.execute(
                "SELECT COUNT(*) FROM kg_claim_observations"
            ).fetchone()[0] == 0
            evidence = store.conn.execute(
                "SELECT polarity,is_current FROM kg_evidence"
            ).fetchone()
            assert tuple(evidence) == (1, 0)
            edge = store.conn.execute(
                "SELECT pos_evidence,status,invalid_at FROM knowledge_graph "
                "WHERE derived=0"
            ).fetchone()
            assert edge["pos_evidence"] == 0
            assert edge["status"] == "retracted"
            assert edge["invalid_at"] is not None
            outcome = store.conn.execute(
                "SELECT prompt_version,prompt_generation,result_hash "
                "FROM kg_claim_extraction_outcomes"
            ).fetchone()
            assert tuple(outcome) == (
                "v14", 14, evidence_ledger.claim_result_hash([]),
            )
            assert store.conn.execute(
                "SELECT COUNT(*) FROM processed_chunks "
                "WHERE chunk_id='shared-claim' AND prompt_version='v14'"
            ).fetchone()[0] == 1
            assert sum(store.import_(empty_wire).values()) == 0
        assert _portable_claim_state(low_high) == _portable_claim_state(high_low)
    finally:
        low_high.close()
        high_low.close()


def test_v7_empty_chunk_replay_preserves_overlapping_chunk_authority(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "overlap-outcome-source"))
    try:
        base_chunk, message_id = _seed_shared_claim_artifact(src)
        other_chunk = _portable_claim_chunk(
            src, chunk_id="shared-claim-other", message_id=message_id,
            role="user", content="The service target is 65 percent",
        )
        claim = [Triple(
            "service", "has_attribute", "65_percent", 1,
            source_message_id=message_id,
        )]
        _persist_portable_claim(
            src, base_chunk, claim, prompt_version="v13"
        )
        _persist_portable_claim(
            src, other_chunk, claim, prompt_version="v13"
        )
        low = tmp_path / "overlap-outcome-low.jsonl"
        src.export(low)
        _persist_portable_claim(
            src, base_chunk, [], prompt_version="v14"
        )
        assert src.conn.execute(
            "SELECT pos_evidence,status FROM knowledge_graph"
        ).fetchone()[:] == (2, "active")
        high = tmp_path / "overlap-outcome-high.jsonl"
        src.export(high)
    finally:
        src.close()

    low_high = HyMem(HyMemConfig(root=tmp_path / "overlap-low-high"))
    high_low = HyMem(HyMemConfig(root=tmp_path / "overlap-high-low"))
    try:
        low_high.import_(low)
        low_high.import_(high)
        high_low.import_(high)
        high_low.import_(low)
        for store in (low_high, high_low):
            assert [row[0] for row in store.conn.execute(
                "SELECT chunk_id FROM kg_claim_observations"
            ).fetchall()] == ["shared-claim-other"]
            assert store.conn.execute(
                "SELECT COUNT(*) FROM kg_evidence WHERE is_current=1"
            ).fetchone()[0] == 1
            assert store.conn.execute(
                "SELECT pos_evidence,status,invalid_at FROM knowledge_graph"
            ).fetchone()[:] == (2, "active", None)
            assert sum(store.import_(high).values()) == 0
        assert _portable_claim_state(low_high) == _portable_claim_state(high_low)
    finally:
        low_high.close()
        high_low.close()


def test_v7_same_generation_empty_and_nonempty_outcomes_conflict_atomically(
    tmp_path,
):
    base = HyMem(HyMemConfig(root=tmp_path / "outcome-conflict-base"))
    try:
        _chunk, message_id = _seed_shared_claim_artifact(base)
        base_wire = tmp_path / "outcome-conflict-base.jsonl"
        base.export(base_wire)
    finally:
        base.close()
    claimed = HyMem(HyMemConfig(root=tmp_path / "outcome-conflict-claim"))
    empty = HyMem(HyMemConfig(root=tmp_path / "outcome-conflict-empty"))
    try:
        claimed.import_(base_wire)
        empty.import_(base_wire)
        chunk = Chunk(
            id="shared-claim", session_id="claim-session",
            start_message_id=message_id, end_message_id=message_id,
            salience_reason="portable-test",
            text="user: The service target is 65 percent",
            source_message_ids=(message_id,),
        )
        _persist_portable_claim(
            claimed, chunk,
            [Triple("service", "has_attribute", "65_percent", 1,
                    source_message_id=message_id)],
            prompt_version="v14",
        )
        _persist_portable_claim(empty, chunk, [], prompt_version="v14")
        claimed_wire = tmp_path / "outcome-conflict-claim.jsonl"
        claimed.export(claimed_wire)
        before = _portable_claim_state(empty)
        with pytest.raises(ValueError, match="prompt generation"):
            empty.import_(claimed_wire)
        assert _portable_claim_state(empty) == before
        assert empty.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        claimed.close()
        empty.close()


def test_v7_rejects_positive_evidence_without_exact_assertion_lifecycle(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "missing-assertion-source"))
    try:
        chunk, message_id = _seed_shared_claim_artifact(src)
        _persist_portable_claim(
            src, chunk,
            [Triple("service", "has_attribute", "65_percent", 1,
                    source_message_id=message_id)],
            prompt_version="v13",
        )
        wire = tmp_path / "missing-assertion.jsonl"
        src.export(wire)
    finally:
        src.close()

    _rewrite_v6_export(
        wire,
        lambda body: body.__setitem__(
            slice(None),
            [
                item for item in body
                if not (
                    item["type"] == "edge_lifecycle"
                    and item["record"]["event_kind"] == "claim_assertion"
                )
            ],
        ),
    )
    dst = HyMem(HyMemConfig(root=tmp_path / "missing-assertion-target"))
    try:
        with pytest.raises(ValueError, match="assertion coverage mismatches"):
            dst.import_(wire)
        assert dst.conn.execute("SELECT COUNT(*) FROM knowledge_graph").fetchone()[0] == 0
        assert dst.conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 0
    finally:
        dst.close()


def test_v7_future_source_clock_rejects_and_rolls_back_before_claim_mutation(
    tmp_path,
):
    src = HyMem(HyMemConfig(root=tmp_path / "future-clock-source"))
    try:
        _seed_named_portable_claim(
            src,
            session_id="future-source",
            chunk_id="future-chunk",
            created_at="2024-01-01T00:00:00Z",
        )
        wire = tmp_path / "future-clock-v7.jsonl"
        src.export(wire)

        def move_valid_source_to_future(body):
            future_raw = "2100-01-01T00:00:00Z"
            future_normalized = "2100-01-01T00:00:00.000Z"
            for item in body:
                record = item.get("record", {})
                if item["type"] == "message_retention_coverage":
                    record["source_created_at"] = future_raw
                elif item["type"] == "edge":
                    record["valid_at"] = future_normalized
                elif item["type"] == "edge_evidence":
                    record["source_created_at"] = future_raw
                    record["source_event_at"] = future_normalized
                elif item["type"] == "edge_lifecycle":
                    record["event_at"] = future_normalized

        _rewrite_v6_export(wire, move_valid_source_to_future)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "future-clock-target"))
    try:
        dst.log_message(
            "preserved", "user", "local state", created_at="2024-01-01"
        )
        before = {
            table: dst.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "sessions",
                "messages",
                "message_retention_coverage",
                "knowledge_graph",
                "kg_evidence",
                "kg_edge_lifecycle",
            )
        }
        before_dump = list(dst.conn.iterdump())
        with pytest.raises(ValueError, match="portable canonical evidence valid time"):
            dst.import_(wire)
        after = {
            table: dst.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }
        assert after == before
        assert list(dst.conn.iterdump()) == before_dump
        assert not dst.conn.in_transaction
        assert dst.conn.execute(
            "SELECT 1 FROM sessions WHERE id='future-source'"
        ).fetchone() is None
        assert dst.conn.execute(
            "SELECT 1 FROM knowledge_graph WHERE subject_canonical='app'"
        ).fetchone() is None
    finally:
        dst.close()


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("edge_evidence", "extracted_at"),
        ("edge_lifecycle", "created_at"),
        ("edge", "last_seen"),
        ("chunk", "created_at"),
    ],
)
def test_v7_future_transaction_clocks_roll_back_entire_import(
    tmp_path, kind, field
):
    src = HyMem(HyMemConfig(root=tmp_path / f"future-{kind}-source"))
    try:
        _seed_named_portable_claim(
            src, session_id="incoming", chunk_id="incoming-chunk",
            created_at="2024-01-01T00:00:00Z",
        )
        wire = tmp_path / f"future-{kind}.jsonl"
        src.export(wire)
    finally:
        src.close()

    def poison(body):
        record = next(item["record"] for item in body if item["type"] == kind)
        record[field] = "2100-01-01T00:00:00.000Z"

    _rewrite_v6_export(wire, poison)
    dst = HyMem(HyMemConfig(root=tmp_path / f"future-{kind}-target"))
    try:
        dst.log_message(
            "preserved", "user", "local sequence state",
            created_at="2024-01-01T00:00:00Z",
        )
        before = list(dst.conn.iterdump())
        with pytest.raises(ValueError, match="portable .* transaction"):
            dst.import_(wire)
        assert list(dst.conn.iterdump()) == before
        assert not dst.conn.in_transaction
    finally:
        dst.close()


def test_v7_rejects_compounded_source_and_transaction_skew(tmp_path):
    """Two individually chained +300s allowances must not become +600s."""
    from datetime import UTC, datetime, timedelta

    src = HyMem(HyMemConfig(root=tmp_path / "compound-skew-source"))
    try:
        _seed_named_portable_claim(
            src, session_id="compound", chunk_id="compound-chunk",
            created_at="2024-01-01T00:00:00Z",
        )
        wire = tmp_path / "compound-skew.jsonl"
        src.export(wire)
    finally:
        src.close()
    base = datetime.now(UTC).replace(microsecond=0)
    extracted = (base + timedelta(seconds=300)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    source = (base + timedelta(seconds=600)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )

    def poison(body):
        for item in body:
            record = item.get("record", {})
            if item["type"] == "message_retention_coverage":
                record["source_created_at"] = source
            elif item["type"] == "edge_evidence":
                record["source_created_at"] = source
                record["source_event_at"] = source
                record["extracted_at"] = extracted
            elif item["type"] == "edge_lifecycle":
                record["event_at"] = source
                record["created_at"] = extracted
            elif item["type"] == "edge":
                record["valid_at"] = source

    _rewrite_v6_export(wire, poison)
    dst = HyMem(HyMemConfig(root=tmp_path / "compound-skew-target"))
    try:
        before = list(dst.conn.iterdump())
        with pytest.raises(ValueError, match="acceptance|message source"):
            dst.import_(wire)
        assert list(dst.conn.iterdump()) == before
        assert not dst.conn.in_transaction
    finally:
        dst.close()


def test_behavioral_merge_removes_member_and_roundtrips_portably(tmp_path):
    from hymem.dreaming.behavioral_dedup import (
        DuplicateMember,
        ProposedMerge,
        apply_behavioral_merges,
    )

    src = HyMem(HyMemConfig(root=tmp_path / "behavioral-portable-source"))
    try:
        src.open_session("claim-session")
        messages = []
        for index, phrase in enumerate(("concise", "concise mode")):
            content = f"I prefer {phrase}"
            message_id = src.log_message(
                "claim-session", "user", content,
                created_at=f"2026-01-0{index + 1}T00:00:00Z",
            )
            messages.append((message_id, content))
        with core_db.transaction(src.conn):
            materialize_message_coverage(src.conn, "claim-session")
        edge_ids = []
        for index, (message_id, content) in enumerate(messages):
            obj = "concise" if index == 0 else "concise_mode"
            chunk = _portable_claim_chunk(
                src, chunk_id=f"behavioral-portable-{index}",
                message_id=message_id, role="user", content=content,
            )
            _persist_portable_claim(
                src, chunk,
                [Triple("user", "prefers", obj, 1,
                        source_message_id=message_id)],
                prompt_version="v13",
            )
            edge_ids.append(int(src.conn.execute(
                "SELECT id FROM knowledge_graph WHERE object_canonical=?", (obj,)
            ).fetchone()[0]))
        with core_db.transaction(src.conn):
            apply_behavioral_merges(src.conn, [ProposedMerge(
                subject="user", predicate="prefers",
                survivor_id=edge_ids[0], survivor_object="concise",
                survivor_pos=2, survivor_neg=0,
                members=[DuplicateMember(
                    edge_ids[1], "concise_mode", 2, 0, 0.99,
                )],
            )])
        assert src.conn.execute(
            "SELECT 1 FROM knowledge_graph WHERE id=?", (edge_ids[1],)
        ).fetchone() is None
        assert src.conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes"
        ).fetchone()[0] == 2
        wire = tmp_path / "behavioral-portable.jsonl"
        src.export(wire)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "behavioral-portable-target"))
    try:
        dst.import_(wire)
        assert sum(dst.import_(wire).values()) == 0
        assert [tuple(row) for row in dst.conn.execute(
            "SELECT subject_canonical,predicate,object_canonical,pos_evidence "
            "FROM knowledge_graph WHERE derived=0"
        ).fetchall()] == [("user", "prefers", "concise", 4)]
        assert dst.conn.execute(
            "SELECT canonical FROM entity_aliases WHERE alias='concise_mode'"
        ).fetchone()[0] == "concise"
        assert dst.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        dst.close()


def test_v7_independent_local_revision_ones_merge_by_prompt_authority(tmp_path):
    base = HyMem(HyMemConfig(root=tmp_path / "branch-base"))
    try:
        _chunk, message_id = _seed_shared_claim_artifact(base)
        base_wire = tmp_path / "branch-base.jsonl"
        base.export(base_wire)
    finally:
        base.close()

    left = HyMem(HyMemConfig(root=tmp_path / "branch-left"))
    right = HyMem(HyMemConfig(root=tmp_path / "branch-right"))
    try:
        left.import_(base_wire)
        right.import_(base_wire)
        chunk = Chunk(
            id="shared-claim", session_id="claim-session",
            start_message_id=message_id, end_message_id=message_id,
            salience_reason="portable-test",
            text="user: The service target is 65 percent",
            source_message_ids=(message_id,),
        )
        _persist_portable_claim(
            left, chunk,
            [Triple("service", "has_attribute", "65_percent", 1,
                    temporal_scope="left", source_message_id=message_id)],
            prompt_version="v13",
        )
        _persist_portable_claim(
            right, chunk,
            [Triple("service", "has_attribute", "65_percent", -1,
                    temporal_scope="right", source_message_id=message_id)],
            prompt_version="v14",
        )
        assert left.conn.execute(
            "SELECT revision FROM kg_evidence"
        ).fetchone()[0] == 1
        assert right.conn.execute(
            "SELECT revision FROM kg_evidence"
        ).fetchone()[0] == 1
        left_wire = tmp_path / "branch-left.jsonl"
        right_wire = tmp_path / "branch-right.jsonl"
        left.export(left_wire)
        right.export(right_wire)

        left.import_(right_wire)
        right.import_(left_wire)
        assert sum(left.import_(right_wire).values()) == 0
        assert sum(right.import_(left_wire).values()) == 0
        assert _portable_claim_state(left) == _portable_claim_state(right)
        assert [tuple(row) for row in left.conn.execute(
            "SELECT revision,polarity,is_current FROM kg_evidence ORDER BY revision"
        )] == [(1, 1, 0), (2, -1, 1)]
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    ("winner_polarity", "winner_version", "loser_polarity", "loser_version"),
    [
        (1, "v15", -1, "v14"),
        (-1, "v16", 1, "v15"),
    ],
)
def test_v7_reconciled_lower_generation_snapshot_reimports_without_growth(
    tmp_path,
    winner_polarity,
    winner_version,
    loser_polarity,
    loser_version,
):
    source = HyMem(HyMemConfig(root=tmp_path / "reconciled-source"))
    try:
        winner_chunk, message_id = _seed_shared_claim_artifact(source)
        loser_chunk = _portable_claim_chunk(
            source,
            chunk_id="shared-claim-overlap",
            message_id=message_id,
            role="user",
            content="The service target is 65 percent",
        )
        def claim(polarity):
            return [Triple(
                "service", "has_attribute", "65_percent", polarity,
                source_message_id=message_id,
            )]

        _persist_portable_claim(
            source, winner_chunk, claim(winner_polarity),
            prompt_version=winner_version,
        )
        # The lower-generation overlapping result temporarily creates the
        # opposite revision; finalization restores the already-published
        # global winner as a new current revision.
        _persist_portable_claim(
            source, loser_chunk, claim(loser_polarity),
            prompt_version=loser_version,
        )
        wire = tmp_path / "reconciled.jsonl"
        source.export(wire)
    finally:
        source.close()

    target = HyMem(HyMemConfig(root=tmp_path / "reconciled-target"))
    try:
        target.import_(wire)
        before_state = _portable_claim_state(target)
        before = tmp_path / "reconciled-before.jsonl"
        after = tmp_path / "reconciled-after.jsonl"
        target.export(before)
        second = target.import_(wire)
        target.export(after)
        assert sum(second.values()) == 0
        assert _portable_claim_state(target) == before_state
        assert after.read_bytes() == before.read_bytes()
    finally:
        target.close()


def test_v7_exact_branch_intervals_converge_at_outcome_boundary(tmp_path):
    base = HyMem(HyMemConfig(root=tmp_path / "interval-base"))
    try:
        _chunk, message_id = _seed_shared_claim_artifact(base)
        base_wire = tmp_path / "interval-base.jsonl"
        base.export(base_wire)
    finally:
        base.close()

    historical = HyMem(HyMemConfig(root=tmp_path / "interval-historical"))
    winner = HyMem(HyMemConfig(root=tmp_path / "interval-winner"))
    try:
        historical.import_(base_wire)
        winner.import_(base_wire)
        old_chunk = _portable_claim_chunk(
            historical,
            chunk_id="interval-old",
            message_id=message_id,
            role="user",
            content="The service target is 65 percent",
        )
        middle_chunk = _portable_claim_chunk(
            historical,
            chunk_id="interval-middle",
            message_id=message_id,
            role="user",
            content="The service target is 65 percent",
        )
        winner_chunk = _portable_claim_chunk(
            winner,
            chunk_id="interval-winner",
            message_id=message_id,
            role="user",
            content="The service target is 65 percent",
        )
        old = Triple(
            "service", "has_attribute", "65_percent", 1,
            temporal_scope="old", source_message_id=message_id,
        )
        middle = dataclasses.replace(old, temporal_scope="middle")
        _persist_portable_claim(
            historical, old_chunk, [old], prompt_version="v13"
        )
        _persist_portable_claim(
            historical, middle_chunk, [middle], prompt_version="v14"
        )
        _persist_portable_claim(
            winner, winner_chunk, [old], prompt_version="v15"
        )
        historical_wire = tmp_path / "interval-historical.jsonl"
        winner_wire = tmp_path / "interval-winner.jsonl"
        historical.export(historical_wire)
        winner.export(winner_wire)
    finally:
        historical.close()
        winner.close()

    def backdate(body, clocks):
        evidence_by_id = {}
        for obj in body:
            record = obj.get("record", {})
            kind = obj.get("type")
            if kind == "session":
                record["started_at"] = "2025-01-01T00:00:00.000Z"
            elif kind == "chunk":
                if record["chunk_kind"] == "coverage":
                    record["created_at"] = "2025-01-02T01:04:30.000Z"
                elif record["id"] in clocks:
                    record["created_at"] = clocks[record["id"]][0]
                else:
                    record["created_at"] = "2025-01-02T01:05:00.000Z"
            elif kind == "message_retention_coverage":
                record["created_at"] = "2025-01-02T01:05:00.000Z"
            elif kind == "edge_evidence":
                extracted, lifecycle, observed, published, superseded = clocks[
                    record["chunk_id"]
                ]
                record["extracted_at"] = extracted
                record["published_at"] = published
                record["superseded_at"] = superseded
                evidence_by_id[int(record["id"])] = lifecycle
            elif kind == "claim_observation":
                record["observed_at"] = clocks[record["chunk_id"]][2]
            elif kind == "claim_extraction_outcome":
                record["succeeded_at"] = clocks[record["chunk_id"]][3]
            elif kind == "edge":
                record["first_seen"] = min(value[0] for value in clocks.values())
                record["last_seen"] = max(value[3] for value in clocks.values())
                record["last_reinforced"] = record["last_seen"]
        for obj in body:
            if obj.get("type") == "edge_lifecycle":
                source_id = obj["record"]["source_evidence_id"]
                if source_id is not None:
                    obj["record"]["created_at"] = evidence_by_id[int(source_id)]

    historical_clocks = {
        "interval-old": (
            "2025-02-01T00:00:00.000Z",
            "2025-02-01T00:01:00.000Z",
            "2025-02-01T00:02:00.000Z",
            "2025-02-01T00:03:00.000Z",
            "2025-03-01T00:03:00.000Z",
        ),
        "interval-middle": (
            "2025-03-01T00:00:00.000Z",
            "2025-03-01T00:01:00.000Z",
            "2025-03-01T00:02:00.000Z",
            "2025-03-01T00:03:00.000Z",
            None,
        ),
    }
    winner_clocks = {
        "interval-winner": (
            "2025-04-01T00:00:00.000Z",
            "2025-04-01T00:01:00.000Z",
            "2025-04-01T00:02:00.000Z",
            "2025-04-01T00:03:00.000Z",
            None,
        ),
    }
    _rewrite_v6_export(
        historical_wire, lambda body: backdate(body, historical_clocks)
    )
    _rewrite_v6_export(
        winner_wire, lambda body: backdate(body, winner_clocks)
    )

    old_new = HyMem(HyMemConfig(root=tmp_path / "interval-old-new"))
    new_old = HyMem(HyMemConfig(root=tmp_path / "interval-new-old"))
    try:
        old_new.import_(historical_wire)
        old_new.import_(winner_wire)
        new_old.import_(winner_wire)
        new_old.import_(historical_wire)
        for store in (old_new, new_old):
            assert [tuple(row) for row in store.conn.execute(
                "SELECT revision,temporal_scope,is_current FROM kg_evidence "
                "ORDER BY revision"
            ).fetchall()] == [
                (1, "old", 0),
                (2, "middle", 0),
                (3, "old", 1),
            ]
            # The v14 row remains authoritative until the v15 chunk succeeds;
            # using observed_at as its close would leave this cutoff empty.
            assert len(store.facts_at(
                "2025-01-03T00:00:00Z",
                recorded_at="2025-04-01T00:02:30Z",
            )) == 1
            assert len(store.facts_at(
                "2025-01-03T00:00:00Z",
                recorded_at="2025-04-01T00:04:00Z",
            )) == 1
            assert sum(store.import_(historical_wire).values()) == 0
            assert sum(store.import_(winner_wire).values()) == 0
        assert _portable_claim_state(old_new) == _portable_claim_state(new_old)
        left = tmp_path / "interval-old-new-export.jsonl"
        right = tmp_path / "interval-new-old-export.jsonl"
        old_new.export(left)
        new_old.export(right)
        # Local surrogate IDs intentionally remain stable once assigned, so
        # opposite import orders need not produce byte-identical wire IDs.  The
        # portable history itself (including deterministic revisions and
        # transaction intervals) must converge, and each wire must be a fixed
        # point when re-imported into its originating store.
        assert sum(old_new.import_(left).values()) == 0
        assert sum(new_old.import_(right).values()) == 0
    finally:
        old_new.close()
        new_old.close()


def test_v7_exact_shared_occurrence_replay_is_audit_stable_and_cites_once(
    tmp_path,
):
    """A stale reinterpretation must not rewrite a merged clone on replay.

    Both branches inherit the exact same r1 occurrence. One republishes that
    interpretation at v15; the other changes the same chunk to a negative r2
    at v14 and closes r1 with the specific ``source_reinterpreted`` reason.
    The union needs two append-only interval handles for r1, but they still
    represent one source occurrence and therefore one citation.
    """
    base = HyMem(HyMemConfig(root=tmp_path / "clone-base"))
    try:
        shared_chunk, message_id = _seed_shared_claim_artifact(base)
        old = Triple(
            "service", "has_attribute", "65_percent", 1,
            temporal_scope="old", source_message_id=message_id,
        )
        _persist_portable_claim(
            base, shared_chunk, [old], prompt_version="v13"
        )
        base_wire = tmp_path / "clone-base.jsonl"
        base.export(base_wire)
    finally:
        base.close()

    positive = HyMem(HyMemConfig(root=tmp_path / "clone-positive"))
    negative = HyMem(HyMemConfig(root=tmp_path / "clone-negative"))
    try:
        positive.import_(base_wire)
        negative.import_(base_wire)
        _persist_portable_claim(
            positive, shared_chunk, [old], prompt_version="v15"
        )
        _persist_portable_claim(
            negative, shared_chunk,
            [dataclasses.replace(old, polarity=-1)], prompt_version="v14",
        )
        positive_wire = tmp_path / "clone-positive.jsonl"
        negative_wire = tmp_path / "clone-negative.jsonl"
        positive.export(positive_wire)
        negative.export(negative_wire)
    finally:
        positive.close()
        negative.close()

    def backdate(body, *, positive_branch):
        evidence_lifecycle = {}
        outcome_at = (
            "2025-04-01T00:03:00.000Z"
            if positive_branch else "2025-06-01T00:03:00.000Z"
        )
        observed_at = (
            "2025-04-01T00:02:00.000Z"
            if positive_branch else "2025-06-01T00:02:00.000Z"
        )
        for obj in body:
            record = obj.get("record", {})
            kind = obj.get("type")
            if kind == "session":
                record["started_at"] = "2024-12-31T23:50:00.000Z"
            elif kind == "chunk":
                record["created_at"] = (
                    "2024-12-31T23:54:30.000Z"
                    if record["chunk_kind"] == "coverage"
                    else "2025-01-01T00:00:00.000Z"
                )
            elif kind == "message_retention_coverage":
                record["source_created_at"] = "2024-12-31T23:54:05.000Z"
                record["created_at"] = "2024-12-31T23:55:00.000Z"
            elif kind == "edge":
                record["first_seen"] = "2025-01-01T00:00:00.000Z"
                record["last_seen"] = outcome_at
                record["last_reinforced"] = outcome_at
                record["valid_at"] = "2024-12-31T23:54:05.000Z"
                if record["status"] == "retracted":
                    record["invalid_at"] = "2024-12-31T23:54:05.000Z"
            elif kind == "edge_evidence":
                record["source_created_at"] = "2024-12-31T23:54:05.000Z"
                record["source_event_at"] = "2024-12-31T23:54:05.000Z"
                if int(record["polarity"]) == 1:
                    record["extracted_at"] = "2025-01-01T00:00:00.000Z"
                    record["published_at"] = "2025-01-01T00:03:00.000Z"
                    record["superseded_at"] = (
                        None if positive_branch
                        else "2025-06-01T00:03:00.000Z"
                    )
                    evidence_lifecycle[int(record["id"])] = (
                        "2025-01-01T00:01:00.000Z"
                    )
                else:
                    record["extracted_at"] = "2025-06-01T00:00:00.000Z"
                    record["published_at"] = "2025-06-01T00:03:00.000Z"
            elif kind == "claim_observation":
                record["observed_at"] = observed_at
            elif kind == "claim_extraction_outcome":
                record["succeeded_at"] = outcome_at
        for obj in body:
            if obj.get("type") != "edge_lifecycle":
                continue
            source_id = obj["record"]["source_evidence_id"]
            if source_id is not None:
                obj["record"]["event_at"] = "2024-12-31T23:54:05.000Z"
                obj["record"]["created_at"] = evidence_lifecycle[int(source_id)]

    _rewrite_v6_export(
        positive_wire, lambda body: backdate(body, positive_branch=True)
    )
    _rewrite_v6_export(
        negative_wire, lambda body: backdate(body, positive_branch=False)
    )

    positive_negative = HyMem(
        HyMemConfig(root=tmp_path / "clone-positive-negative")
    )
    negative_positive = HyMem(
        HyMemConfig(root=tmp_path / "clone-negative-positive")
    )
    try:
        positive_negative.import_(positive_wire)
        positive_negative.import_(negative_wire)
        negative_positive.import_(negative_wire)
        negative_positive.import_(positive_wire)

        assert _portable_claim_state(positive_negative) == _portable_claim_state(
            negative_positive
        )
        for store in (positive_negative, negative_positive):
            assert all(
                row["superseded_at"] is None
                or row["published_at"] <= row["superseded_at"]
                for row in store.conn.execute(
                    "SELECT published_at,superseded_at FROM kg_evidence "
                    "WHERE provenance_status='canonical'"
                ).fetchall()
            )
            late_negative = store.conn.execute(
                "SELECT published_at,superseded_at FROM kg_evidence "
                "WHERE polarity=-1"
            ).fetchone()
            assert tuple(late_negative) == (
                "2025-06-01T00:03:00.000Z",
                "2025-06-01T00:03:00.000Z",
            )
            retired_positive = store.conn.execute(
                "SELECT superseded_at,superseded_reason FROM kg_evidence "
                "WHERE polarity=1 AND is_current=0"
            ).fetchall()
            assert [tuple(row) for row in retired_positive] == [(
                "2025-06-01T00:03:00.000Z", "source_reinterpreted",
            )]
            snapshot = store.facts_at(
                "2025-01-02T00:00:00Z",
                recorded_at="2025-05-01T00:00:00Z",
            )
            assert len(snapshot) == 1
            assert len(snapshot[0].citations) == 1

            before = tmp_path / f"{id(store)}-before.jsonl"
            after = tmp_path / f"{id(store)}-after.jsonl"
            store.export(before)
            assert sum(store.import_(negative_wire).values()) == 0
            store.export(after)
            assert before.read_bytes() == after.read_bytes()
    finally:
        positive_negative.close()
        negative_positive.close()


def test_v7_higher_authority_recovers_past_unpublished_staged_revision(tmp_path):
    base = HyMem(HyMemConfig(root=tmp_path / "staged-base"))
    try:
        shared_chunk, message_id = _seed_shared_claim_artifact(base)
        base_wire = tmp_path / "staged-base.jsonl"
        base.export(base_wire)
    finally:
        base.close()

    staged = HyMem(HyMemConfig(root=tmp_path / "staged-target"))
    winner = HyMem(HyMemConfig(root=tmp_path / "staged-winner"))
    try:
        staged.import_(base_wire)
        winner.import_(base_wire)
        negative = Triple(
            "service", "has_attribute", "65_percent", -1,
            source_message_id=message_id,
        )
        positive = dataclasses.replace(negative, polarity=1)
        # Stop at the supported pre-publication boundary instead of publishing
        # and then corrupting history: evidence registration precedes the
        # observation/outcome in the normal phase-1 transaction. A crash or a
        # migrated v41 orphan can therefore leave exactly this hidden state.
        source = phase1._claim_sources_for_chunk(
            staged.conn, shared_chunk
        )[0]
        with core_db.transaction(staged.conn):
            edge_id = int(staged.conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical,predicate,"
                "object_canonical,status,derived) "
                "VALUES ('service','has_attribute','65_percent','active',0)"
            ).lastrowid)
            evidence_ledger.record_chunk_evidence(
                staged.conn,
                edge_id=edge_id,
                chunk_id=shared_chunk.id,
                evidence_kind="extraction",
                polarity=-1,
                evidence_weight=2,
                weight_source="configured_role:user",
                prompt_version="v13",
                source_role=source.role,
                surface_subject="service",
                surface_object="65_percent",
                source_message_id=source.message_id,
                source_session_id=source.session_id,
                source_created_at=source.source_created_at,
                source_event_at=phase1._normalized_source_event_at(
                    staged.conn, source.source_created_at
                ),
                source_coverage_chunk_id=source.chunk_id,
                source_coverage_version=LOSSLESS_COVERAGE_VERSION,
            )
        _persist_portable_claim(
            winner, shared_chunk, [positive], prompt_version="v15"
        )
        winner_wire = tmp_path / "staged-winner.jsonl"
        winner.export(winner_wire)

        # The staged revision has never acquired publication authority, and
        # the production immutability guard remains installed throughout.
        assert staged.conn.execute(
            "SELECT published_at FROM kg_evidence WHERE is_current=1"
        ).fetchone()[0] is None
        assert staged.conn.execute(
            "SELECT COUNT(*) FROM kg_claim_observations"
        ).fetchone()[0] == 0
        assert staged.conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes"
        ).fetchone()[0] == 0
        assert staged.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' "
            "AND name='kg_evidence_published_at_update_guard'"
        ).fetchone() is not None

        assert staged.facts_at("2025-01-03T00:00:00Z") == []
        staged.import_(winner_wire)
        rows = staged.conn.execute(
            "SELECT polarity,is_current,published_at,superseded_at,"
            "superseded_reason FROM kg_evidence ORDER BY revision"
        ).fetchall()
        unpublished = next(row for row in rows if int(row["polarity"]) == -1)
        current = next(row for row in rows if int(row["polarity"]) == 1)
        assert tuple(unpublished) == (
            -1, 0, None, current["published_at"], "lower_prompt_authority",
        )
        assert current["is_current"] == 1
        assert current["published_at"] is not None
        assert len(staged.facts_at("2025-01-03T00:00:00Z")) == 1
        cache = staged.conn.execute(
            "SELECT pos_evidence,neg_evidence,status,invalid_at "
            "FROM knowledge_graph WHERE subject_canonical='service' "
            "AND predicate='has_attribute' AND object_canonical='65_percent'"
        ).fetchone()
        assert tuple(cache) == (2, 0, "active", None)
        from hymem.query.entities import match_known_entities

        assert match_known_entities(staged.conn, "service") == ["service"]
        assert staged.conn.execute("PRAGMA foreign_key_check").fetchall() == []

        before = "\n".join(staged.conn.iterdump())
        replay = staged.import_(winner_wire)
        after = "\n".join(staged.conn.iterdump())
        assert sum(replay.values()) == 0
        assert before == after
        assert not staged.conn.in_transaction
    finally:
        staged.close()
        winner.close()


def test_v7_same_semantic_revision_uses_commutative_authoritative_audit_metadata(
    tmp_path,
):
    base = HyMem(HyMemConfig(root=tmp_path / "metadata-base"))
    try:
        _chunk, message_id = _seed_shared_claim_artifact(base)
        base_wire = tmp_path / "metadata-base.jsonl"
        base.export(base_wire)
    finally:
        base.close()

    old = HyMem(HyMemConfig(root=tmp_path / "metadata-old"))
    new = HyMem(HyMemConfig(root=tmp_path / "metadata-new"))
    try:
        old.import_(base_wire)
        new.import_(base_wire)
        old_chunk = _portable_claim_chunk(
            old, chunk_id="metadata-old-chunk", message_id=message_id,
            role="user", content="The service target is 65 percent",
        )
        new_chunk = _portable_claim_chunk(
            new, chunk_id="metadata-new-chunk", message_id=message_id,
            role="user", content="The service target is 65 percent",
        )
        _persist_portable_claim(
            old, old_chunk,
            [Triple("Service", "has_attribute", "65 Percent", 1,
                    source_message_id=message_id)],
            prompt_version="v13",
        )
        _persist_portable_claim(
            new, new_chunk,
            [Triple("service", "has_attribute", "65_percent", 1,
                    source_message_id=message_id)],
            prompt_version="v14",
        )
        old_wire = tmp_path / "metadata-old.jsonl"
        new_wire = tmp_path / "metadata-new.jsonl"
        old.export(old_wire)
        new.export(new_wire)
    finally:
        old.close()
        new.close()

    # Force the hard tie: both independently published occurrences have the
    # exact same normalized extraction/publication clocks. Selection must then
    # use the remaining audit tuple, never whichever row was imported first.
    def tie_audit_clocks(body):
        for obj in body:
            record = obj.get("record", {})
            kind = obj.get("type")
            if kind == "session":
                record["started_at"] = "2025-01-01T00:00:00.000Z"
            elif kind == "chunk":
                record["created_at"] = (
                    "2025-01-02T01:04:30.000Z"
                    if record["chunk_kind"] == "coverage"
                    else "2025-02-01T00:00:00.000Z"
                )
            elif kind == "message_retention_coverage":
                record["created_at"] = "2025-01-02T01:05:00.000Z"
            elif kind == "edge":
                record["first_seen"] = "2025-02-01T00:01:00.000Z"
                record["last_seen"] = "2025-02-01T00:04:00.000Z"
                record["last_reinforced"] = "2025-02-01T00:04:00.000Z"
            elif kind == "edge_evidence":
                record["extracted_at"] = "2025-02-01T00:01:00.000Z"
                record["published_at"] = "2025-02-01T00:04:00.000Z"
            elif kind == "edge_lifecycle":
                record["created_at"] = "2025-02-01T00:02:00.000Z"
            elif kind == "claim_observation":
                record["observed_at"] = "2025-02-01T00:03:00.000Z"
            elif kind == "claim_extraction_outcome":
                record["succeeded_at"] = "2025-02-01T00:04:00.000Z"

    _rewrite_v6_export(old_wire, tie_audit_clocks)
    _rewrite_v6_export(new_wire, tie_audit_clocks)

    old_new = HyMem(HyMemConfig(root=tmp_path / "metadata-old-new"))
    new_old = HyMem(HyMemConfig(root=tmp_path / "metadata-new-old"))
    try:
        old_new.import_(old_wire)
        old_new.import_(new_wire)
        new_old.import_(new_wire)
        new_old.import_(old_wire)

        def audit(store):
            return [tuple(row) for row in store.conn.execute(
                "SELECT chunk_id,surface_subject,surface_object,"
                "extraction_prompt_version,extracted_at,revision,interpretation_key "
                "FROM kg_evidence ORDER BY source_session_id,source_message_id,"
                "evidence_kind,revision"
            ).fetchall()]

        assert audit(old_new) == audit(new_old)
        assert audit(old_new)[0][0:4] == (
            "metadata-new-chunk", "service", "65_percent", "v14",
        )
        assert sum(old_new.import_(new_wire).values()) == 0
        assert sum(new_old.import_(old_wire).values()) == 0
        assert audit(old_new) == audit(new_old)
        old_new_wire = tmp_path / "metadata-old-new-export.jsonl"
        new_old_wire = tmp_path / "metadata-new-old-export.jsonl"
        old_new.export(old_new_wire)
        new_old.export(new_old_wire)
        assert old_new_wire.read_bytes() == new_old_wire.read_bytes()
    finally:
        old_new.close()
        new_old.close()


def test_v7_same_generation_branch_divergence_rolls_back(tmp_path):
    base = HyMem(HyMemConfig(root=tmp_path / "same-generation-base"))
    try:
        _chunk, message_id = _seed_shared_claim_artifact(base)
        base_wire = tmp_path / "same-generation-base.jsonl"
        base.export(base_wire)
    finally:
        base.close()

    left = HyMem(HyMemConfig(root=tmp_path / "same-generation-left"))
    right = HyMem(HyMemConfig(root=tmp_path / "same-generation-right"))
    try:
        left.import_(base_wire)
        right.import_(base_wire)
        chunk = Chunk(
            id="shared-claim", session_id="claim-session",
            start_message_id=message_id, end_message_id=message_id,
            salience_reason="portable-test",
            text="user: The service target is 65 percent",
            source_message_ids=(message_id,),
        )
        _persist_portable_claim(
            left, chunk,
            [Triple("service", "has_attribute", "65_percent", 1,
                    temporal_scope="left", source_message_id=message_id)],
            prompt_version="v14",
        )
        _persist_portable_claim(
            right, chunk,
            [Triple("service", "has_attribute", "65_percent", -1,
                    temporal_scope="right", source_message_id=message_id)],
            prompt_version="v14",
        )
        right_wire = tmp_path / "same-generation-right.jsonl"
        right.export(right_wire)
        before = _portable_claim_state(left)
        with pytest.raises(ValueError, match="prompt generation"):
            left.import_(right_wire)
        assert _portable_claim_state(left) == before
        assert left.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        left.close()
        right.close()


def test_v7_omits_untrusted_derived_rows_and_rebuilds_with_destination_policy(
    tmp_path,
):
    from hymem.dreaming.inference import infer_transitive_edges

    low_cfg = HyMemConfig(root=tmp_path / "derived-source", retract_threshold=0.15)
    src = HyMem(low_cfg)
    try:
        _seed_named_portable_claim(
            src, session_id="derived-a", chunk_id="derived-a-chunk",
            created_at="2024-01-01T00:00:00Z",
            subject="app", predicate="uses", object_="db",
        )
        _seed_named_portable_claim(
            src, session_id="derived-b", chunk_id="derived-b-chunk",
            created_at="2024-01-02T00:00:00Z",
            subject="db", predicate="depends_on", object_="disk",
        )
        assert infer_transitive_edges(src.conn, low_cfg) == 1
        derived = src.conn.execute(
            "SELECT id FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0]
        src.conn.execute(
            "UPDATE knowledge_graph SET pos_evidence=999,valid_at='2099-01-01',"
            "first_seen='2099-01-01' WHERE id=?", (derived,),
        )
        wire = tmp_path / "derived-v7.jsonl"
        counts = src.export(wire)
        assert counts["edge"] == 2
        assert all(
            not item["record"]["derived"]
            for item in (
                json.loads(line) for line in wire.read_text().splitlines()
                if '"type": "edge"' in line
            )
        )
    finally:
        src.close()

    low = HyMem(HyMemConfig(
        root=tmp_path / "derived-low", retract_threshold=0.15
    ))
    try:
        low.import_(wire)
        rebuilt = low.conn.execute(
            "SELECT id,pos_evidence,first_seen FROM knowledge_graph "
            "WHERE derived=1 AND subject_canonical='app' "
            "AND object_canonical='disk'"
        ).fetchone()
        assert rebuilt is not None
        assert rebuilt["pos_evidence"] == 1
        assert rebuilt["first_seen"] != "2099-01-01"
        stable_id = rebuilt["id"]
        assert sum(low.import_(wire).values()) == 0
        assert low.conn.execute(
            "SELECT id FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0] == stable_id
        high_policy = dataclasses.replace(low.config, retract_threshold=0.9)
        assert sum(portability.import_jsonl(
            low.conn, wire, config=high_policy
        ).values()) == 0
        assert low.conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0] == 0
        assert sum(portability.import_jsonl(
            low.conn, wire, config=low.config
        ).values()) == 0
        assert low.conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0] == 1
    finally:
        low.close()

    high = HyMem(HyMemConfig(
        root=tmp_path / "derived-high", retract_threshold=0.9
    ))
    try:
        high.import_(wire)
        assert high.conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0] == 0
    finally:
        high.close()

    addon = HyMem(HyMemConfig(root=tmp_path / "derived-addon"))
    try:
        _seed_named_portable_claim(
            addon, session_id="derived-addon", chunk_id="derived-addon-chunk",
            created_at="2024-01-03T00:00:00Z",
            subject="app", predicate="uses", object_="db",
        )
        addon_wire = tmp_path / "derived-addon.jsonl"
        addon.export(addon_wire)
    finally:
        addon.close()
    configless = HyMem(HyMemConfig(
        root=tmp_path / "derived-configless", retract_threshold=0.15
    ))
    try:
        configless.import_(wire)
        assert configless.conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0] == 1
        portability.import_jsonl(configless.conn, addon_wire, config=None)
        assert configless.conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE derived=1"
        ).fetchone()[0] == 0
    finally:
        configless.close()


def test_true_v6_import_materializes_legacy_ledger_then_exports_v7(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "true-v6-source"))
    try:
        _seed(src)
        old = tmp_path / "true-v6.jsonl"
        src.export(old)
        _downgrade_to_true_v6(old)
        header = json.loads(old.read_text(encoding="utf-8").splitlines()[0])
        assert header["version"] == 6
    finally:
        src.close()

    middle = HyMem(HyMemConfig(root=tmp_path / "true-v6-middle"))
    try:
        imported = middle.import_(old)
        assert set(imported) >= {row[0] for row in portability._V6_EXPORT_SPEC}
        assert evidence_ledger.count_mismatches(middle.conn) == []
        assert middle.conn.execute(
            "SELECT COUNT(*) FROM kg_evidence_signals"
        ).fetchone()[0] == 1
        assert middle.conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE event_kind='legacy_state'"
        ).fetchone()[0] == 1
        upgraded = tmp_path / "true-v6-upgraded-v7.jsonl"
        middle.export(upgraded)
        assert json.loads(upgraded.read_text(encoding="utf-8").splitlines()[0])[
            "version"
        ] == portability.EXPORT_VERSION
    finally:
        middle.close()

    final = HyMem(HyMemConfig(root=tmp_path / "true-v6-final"))
    try:
        final.import_(upgraded)
        assert evidence_ledger.count_mismatches(final.conn) == []
        assert final.conn.execute(
            "SELECT pos_evidence,status,valid_at FROM knowledge_graph "
            "WHERE subject_canonical='app'"
        ).fetchone()[:] == (3, "active", "2024-03-15T00:00:00.000Z")
    finally:
        final.close()


def test_true_v6_offset_valid_time_is_exactly_idempotent(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "v6-offset-source"))
    try:
        _seed(src)
        wire = tmp_path / "v6-offset.jsonl"
        src.export(wire)
        _downgrade_to_true_v6(wire)
        _rewrite_v6_export(
            wire,
            lambda body: next(
                item for item in body if item["type"] == "edge"
            )["record"].update({
                "valid_at": "2024-03-15 01:00:00+01:00",
            }),
        )
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "v6-offset-target"))
    try:
        dst.import_(wire)
        before = tuple(dst.conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph"
        ).fetchone())
        second = dst.import_(wire)
        assert sum(second.values()) == 0
        assert tuple(dst.conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph"
        ).fetchone()) == before == (
            "active", "2024-03-15T00:00:00.000Z", None,
        )
    finally:
        dst.close()


def test_true_v6_retracted_offsets_preserve_open_and_close_through_v7(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "v6-closed-offset-source"))
    try:
        _seed(src)
        wire = tmp_path / "v6-closed-offset.jsonl"
        src.export(wire)
        _downgrade_to_true_v6(wire)

        def close_edge(body):
            edge = next(item for item in body if item["type"] == "edge")["record"]
            edge.update({
                "status": "retracted",
                "valid_at": "2024-03-15 01:00:00+01:00",
                "invalid_at": "2024-03-17 03:00:00+02:00",
                "last_seen": "2024-03-18 00:00:00+00:00",
            })

        _rewrite_v6_export(wire, close_edge)
    finally:
        src.close()

    middle = HyMem(HyMemConfig(root=tmp_path / "v6-closed-offset-middle"))
    try:
        middle.import_(wire)
        expected = (
            "retracted", "2024-03-15T00:00:00.000Z",
            "2024-03-17T01:00:00.000Z",
        )
        assert tuple(middle.conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph"
        ).fetchone()) == expected
        assert [tuple(row) for row in middle.conn.execute(
            "SELECT event_key,direction,event_at FROM kg_edge_lifecycle "
            "WHERE event_kind='legacy_state' ORDER BY event_key"
        ).fetchall()] == [
            ("portable-v6-legacy-0-open", 1, expected[1]),
            ("portable-v6-legacy-1-close", -1, expected[2]),
        ]
        assert sum(middle.import_(wire).values()) == 0
        upgraded = tmp_path / "v6-closed-offset-v7.jsonl"
        middle.export(upgraded)
    finally:
        middle.close()

    final = HyMem(HyMemConfig(root=tmp_path / "v6-closed-offset-final"))
    try:
        final.import_(upgraded)
        assert tuple(final.conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph"
        ).fetchone()) == expected
        assert sum(final.import_(upgraded).values()) == 0
        assert final.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        final.close()


def test_v7_value_chain_restores_cross_edge_dependencies_and_reimports(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "v7-values-source"))
    try:
        src.open_session("claim-session")
        messages = []
        for value, event_at in (
            ("65", "2020-01-01T00:00:00Z"),
            ("78", "2025-01-01T00:00:00Z"),
            ("90", "2026-01-01T00:00:00Z"),
        ):
            content = f"Target is {value} percent"
            mid = src.log_message(
                "claim-session", "user", content, created_at=event_at
            )
            messages.append((value, content, mid))
        with core_db.transaction(src.conn):
            materialize_message_coverage(src.conn, "claim-session")
        for index, (value, content, mid) in enumerate(messages[:2]):
            chunk = _portable_claim_chunk(
                src, chunk_id=f"value-{index}", message_id=mid,
                role="user", content=content,
            )
            _persist_portable_claim(
                src, chunk,
                [Triple("service", "has_attribute", f"{value}_percent", 1,
                        source_message_id=mid)],
                prompt_version="v13",
            )
        assert supersede_competing_values(src.conn, src.config) == 1
        value, content, mid = messages[2]
        chunk = _portable_claim_chunk(
            src, chunk_id="value-2", message_id=mid, role="user", content=content,
        )
        _persist_portable_claim(
            src, chunk,
            [Triple("service", "has_attribute", "90_percent", 1,
                    source_message_id=mid)],
            prompt_version="v13",
        )
        assert supersede_competing_values(src.conn, src.config) == 1
        out = tmp_path / "v7-value-chain.jsonl"
        counts = src.export(out)
        assert counts["edge_lifecycle"] == 5
        assert counts["lifecycle_dependency"] == 2
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "v7-values-target"))
    try:
        first = dst.import_(out)
        assert first["lifecycle_dependency"] == 2
        assert [tuple(row) for row in dst.conn.execute(
            "SELECT object_canonical,status,invalid_at FROM knowledge_graph "
            "ORDER BY object_canonical"
        )] == [
            ("65_percent", "retracted", "2025-01-01T00:00:00.000Z"),
            ("78_percent", "retracted", "2026-01-01T00:00:00.000Z"),
            ("90_percent", "active", None),
        ]
        assert sum(dst.import_(out).values()) == 0
        assert dst.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        dst.close()


def test_v7_restores_terminal_edge_after_value_cause_is_retired(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "retired-value-source"))
    try:
        src.open_session("claim-session")
        chunks = []
        for index, (value, event_at) in enumerate((
            ("65", "2020-01-01T00:00:00Z"),
            ("78", "2025-01-01T00:00:00Z"),
        )):
            content = f"Target is {value} percent"
            message_id = src.log_message(
                "claim-session", "user", content, created_at=event_at
            )
            with core_db.transaction(src.conn):
                materialize_message_coverage(src.conn, "claim-session")
            chunk = _portable_claim_chunk(
                src, chunk_id=f"retired-value-{index}",
                message_id=message_id, role="user", content=content,
            )
            _persist_portable_claim(
                src, chunk,
                [Triple("service", "has_attribute", f"{value}_percent", 1,
                        source_message_id=message_id)],
                prompt_version="v13",
            )
            chunks.append(chunk)
        assert supersede_competing_values(src.conn, src.config) == 1
        _persist_portable_claim(src, chunks[1], [], prompt_version="v14")
        assert [tuple(row) for row in src.conn.execute(
            "SELECT object_canonical,status FROM knowledge_graph "
            "ORDER BY object_canonical"
        )] == [("65_percent", "active"), ("78_percent", "retracted")]
        retired_cause = int(src.conn.execute(
            "SELECT id FROM kg_evidence WHERE edge_id=(SELECT id FROM knowledge_graph "
            "WHERE object_canonical='78_percent') ORDER BY revision DESC LIMIT 1"
        ).fetchone()[0])
        content = "Target is 55 percent"
        third_id = src.log_message(
            "claim-session", "user", content,
            created_at="2019-01-01T00:00:00Z",
        )
        with core_db.transaction(src.conn):
            materialize_message_coverage(src.conn, "claim-session")
        third_chunk = _portable_claim_chunk(
            src, chunk_id="retired-value-runtime-check",
            message_id=third_id, role="user", content=content,
        )
        _persist_portable_claim(
            src, third_chunk,
            [Triple("service", "has_attribute", "55_percent", 1,
                    source_message_id=third_id)],
            prompt_version="v13",
        )
        loser = int(src.conn.execute(
            "SELECT id FROM knowledge_graph WHERE object_canonical='55_percent'"
        ).fetchone()[0])
        event_at = "2025-01-01T00:00:00.000Z"
        with pytest.raises(ValueError, match="current evidence"):
            record_lifecycle_event(
                src.conn, edge_id=loser,
                event_key=evidence_ledger.value_supersession_event_key(
                    src.conn, loser_edge_id=loser,
                    winner_evidence_id=retired_cause, event_at=event_at,
                ),
                event_kind="value_supersession", direction=-1,
                event_at=event_at, dependency_evidence_ids=[retired_cause],
                details="newer typed value superseded this edge",
            )
        out = tmp_path / "retired-value.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "retired-value-target"))
    try:
        dst.import_(out)
        assert [tuple(row) for row in dst.conn.execute(
            "SELECT object_canonical,status FROM knowledge_graph "
            "ORDER BY object_canonical"
        )] == [
            ("55_percent", "active"), ("65_percent", "active"),
            ("78_percent", "retracted"),
        ]
        dependency = dst.conn.execute(
            "SELECT ev.is_current FROM kg_lifecycle_dependencies dep "
            "JOIN kg_evidence ev ON ev.id=dep.evidence_id"
        ).fetchone()
        assert dependency["is_current"] == 0
        assert sum(dst.import_(out).values()) == 0
        assert dst.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        dst.close()


def test_v7_restores_inactive_phase3_event_with_retired_cause(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "v7-retired-cause-source"))
    try:
        src.open_session("claim-session")
        positive_text = "The app uses Redis"
        negative_text = "The app no longer uses Redis"
        positive_id = src.log_message(
            "claim-session", "user", positive_text,
            created_at="2024-01-01T00:00:00Z",
        )
        negative_id = src.log_message(
            "claim-session", "user", negative_text,
            created_at="2025-01-01T00:00:00Z",
        )
        with core_db.transaction(src.conn):
            materialize_message_coverage(src.conn, "claim-session")
        positive_chunk = _portable_claim_chunk(
            src, chunk_id="retired-positive", message_id=positive_id,
            role="user", content=positive_text,
        )
        negative_chunk = _portable_claim_chunk(
            src, chunk_id="retired-negative", message_id=negative_id,
            role="user", content=negative_text,
        )
        _persist_portable_claim(
            src, positive_chunk,
            [Triple("app", "uses", "redis", 1, source_message_id=positive_id)],
            prompt_version="v13",
        )
        _persist_portable_claim(
            src, negative_chunk,
            [Triple("app", "uses", "redis", -1, source_message_id=negative_id)],
            prompt_version="v13",
        )
        edge_id = src.conn.execute(
            "SELECT id FROM knowledge_graph WHERE subject_canonical='app'"
        ).fetchone()[0]
        cause_id = src.conn.execute(
            "SELECT id FROM kg_evidence WHERE edge_id=? AND polarity=-1",
            (edge_id,),
        ).fetchone()[0]
        record_lifecycle_event(
            src.conn, edge_id=edge_id,
            event_key=evidence_ledger.phase3_retraction_event_key(
                src.conn, [cause_id]
            ),
            event_kind="phase3_retraction", direction=-1,
            event_at="2025-01-01T00:00:00Z",
            dependency_evidence_ids=[cause_id],
            details="confidence_or_negative_dominance",
        )
        assert src.conn.execute(
            "SELECT status FROM knowledge_graph WHERE id=?", (edge_id,)
        ).fetchone()[0] == "retracted"
        _persist_portable_claim(src, negative_chunk, [], prompt_version="v14")
        assert src.conn.execute(
            "SELECT status FROM knowledge_graph WHERE id=?", (edge_id,)
        ).fetchone()[0] == "active"
        # Append-only history can be replayed exactly after its cause retires;
        # only a new event is forbidden from citing a retired cause.
        assert not record_lifecycle_event(
            src.conn, edge_id=edge_id,
            event_key=evidence_ledger.phase3_retraction_event_key(
                src.conn, [cause_id]
            ),
            event_kind="phase3_retraction", direction=-1,
            event_at="2025-01-01T00:00:00Z",
            dependency_evidence_ids=[cause_id],
            details="confidence_or_negative_dominance",
        )
        out = tmp_path / "v7-retired-cause.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "v7-retired-cause-target"))
    try:
        dst.import_(out)
        assert dst.conn.execute(
            "SELECT status FROM knowledge_graph WHERE subject_canonical='app'"
        ).fetchone()[0] == "active"
        cause = dst.conn.execute(
            "SELECT ev.is_current FROM kg_lifecycle_dependencies dep "
            "JOIN kg_evidence ev ON ev.id=dep.evidence_id"
        ).fetchone()
        assert cause["is_current"] == 0
        assert sum(dst.import_(out).values()) == 0
    finally:
        dst.close()


def test_v7_alias_semantics_survive_artifact_backed_claim_replay(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "v7-alias-source"))
    try:
        src.open_session("claim-session")
        src.register_alias("Postgres", "postgresql")
        content = "The app uses Postgres"
        message_id = src.log_message(
            "claim-session", "user", content,
            created_at="2026-01-01T00:00:00Z",
        )
        with core_db.transaction(src.conn):
            materialize_message_coverage(src.conn, "claim-session")
        chunk = _portable_claim_chunk(
            src, chunk_id="alias-claim", message_id=message_id,
            role="user", content=content,
        )
        _persist_portable_claim(
            src, chunk,
            [Triple("app", "uses", "Postgres", 1,
                    source_message_id=message_id)],
            prompt_version="v13",
        )
        out = tmp_path / "v7-alias.jsonl"
        assert src.export(out)["entity_alias"] == 2
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "v7-alias-target"))
    try:
        dst.import_(out)
        assert dst.conn.execute(
            "SELECT canonical FROM entity_aliases WHERE alias='postgres'"
        ).fetchone()[0] == "postgresql"
        replay = Chunk(
            id="alias-claim", session_id="claim-session",
            start_message_id=message_id, end_message_id=message_id,
            salience_reason="portable-test", text="user: " + content,
            source_message_ids=(message_id,),
        )
        _persist_portable_claim(
            dst, replay,
            [Triple("app", "uses", "Postgres", 1,
                    source_message_id=message_id)],
            prompt_version="v14",
        )
        assert [tuple(row) for row in dst.conn.execute(
            "SELECT object_canonical,status FROM knowledge_graph "
            "WHERE subject_canonical='app'"
        )] == [("postgresql", "active")]
        assert dst.conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE is_current=1"
        ).fetchone()[0] == 1
    finally:
        dst.close()


def test_incomplete_profile_stage_is_not_exported_and_cursor_replays(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(src)
    old_generation = src.conn.execute(
        "SELECT profile_published_generation FROM sessions WHERE id = 's1'"
    ).fetchone()[0]
    incomplete_generation = (
        profile_config_version(max_chars=12000, max_items=16)
        + "|walk="
        + ("b" * 32)
    )
    discarded_retry = profile_retry_policy_version(
        profile_config_version(max_chars=12000, max_items=16),
        max_attempts=4,
        rebuild_from=old_generation,
        invalidated_stamp=PROFILE_PROMPT_VERSION,
    )
    src.conn.execute(
        "INSERT INTO profile_staging(session_id, generation, slice_key, "
        "items_json, start_message_id, start_message_offset, end_message_id) "
        "VALUES ('s1', ?, 'partial', '[]', 1, 5, 1)",
        (incomplete_generation,),
    )
    src.conn.execute(
        "UPDATE sessions SET profile_cursor_message_id = NULL, "
        "profile_cursor_partial_message_id = 1, profile_cursor_offset = 5, "
        "profile_cursor_prompt_version = ?, profile_retry_count = 4, "
        "profile_retry_config_version = ?, "
        "profile_quarantined = 1 WHERE id = 's1'",
        (incomplete_generation, discarded_retry),
    )
    out = tmp_path / "incomplete.jsonl"
    src.export(out)
    wire = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert not any(row["type"] == "profile_staging" for row in wire)
    exported_session = next(row["record"] for row in wire if row["type"] == "session")
    assert exported_session["profile_cursor_message_id"] is None
    assert exported_session["profile_cursor_partial_message_id"] is None
    assert exported_session["profile_cursor_offset"] == 0
    assert exported_session["profile_cursor_prompt_version"] is None
    assert exported_session["profile_published_generation"] == old_generation
    assert exported_session["profile_retry_count"] == 0
    assert exported_session["profile_retry_config_version"] is None
    assert exported_session["profile_quarantined"] == 0
    src.close()

    payload = json.dumps({"episodes": [], "summary": "", "procedures": []})
    llm = StubLLMClient(
        fixtures={
            "typed user-profile facts": '{"items": []}',
            "Return the JSON object now": payload,
            "single pass": '{"triples": [], "markers": []}',
        },
        default="[]",
    )
    config = dataclasses.replace(
        HyMemConfig(root=tmp_path / "dst"),
        aggregation_nodes_enabled=False,
        facts_extraction_enabled=False,
    )
    dst = HyMem(config, llm=llm)
    try:
        dst.import_(out)
        assert [(entry.slot, entry.value) for entry in dst.profile()] == [
            ("location", "Utrecht")
        ]
        imported = dst.conn.execute(
            "SELECT profile_cursor_message_id, profile_cursor_prompt_version, "
            "profile_published_generation, profile_retry_count, "
            "profile_retry_config_version, profile_quarantined "
            "FROM sessions WHERE id = 's1'"
        ).fetchone()
        assert imported["profile_cursor_message_id"] is None
        assert imported["profile_cursor_prompt_version"] is None
        assert imported["profile_published_generation"] == old_generation
        assert imported["profile_retry_count"] == 0
        assert imported["profile_retry_config_version"] is None
        assert imported["profile_quarantined"] == 0
        assert dst.conn.execute("SELECT COUNT(*) FROM profile_staging").fetchone()[0] == 0

        report = dst.dream()
        assert report.profile_failures == 0
        assert len([
            call for call in llm.calls if "typed user-profile facts" in call.system
        ]) == 1
        state = dst.conn.execute(
            "SELECT profile_cursor_message_id, profile_prompt_version "
            "FROM sessions WHERE id = 's1'"
        ).fetchone()
        assert state["profile_cursor_message_id"] == 1
        assert state["profile_prompt_version"] == PROFILE_PROMPT_VERSION
        assert [(entry.slot, entry.value) for entry in dst.profile()] == [
            ("location", "Utrecht")
        ], "valid empty replay uses the documented conservative over-keep policy"
    finally:
        dst.close()


def test_profile_import_never_attaches_to_unrelated_live_message_id(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(src)
    out = tmp_path / "profile.jsonl"
    src.export(out)
    src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "dst"))
    try:
        unrelated = dst.log_message("other", "assistant", "unrelated id collision")
        assert unrelated == 1
        dst.import_(out)
        row = dst.conn.execute(
            "SELECT evidence_message_id, source_message_id, source_session_id "
            "FROM user_profile"
        ).fetchone()
        assert row["evidence_message_id"] is None
        assert row["source_message_id"] == 1
        assert row["source_session_id"] == "s1"
        assert dst.profile()[0].evidence_message_id == 1
    finally:
        dst.close()


def test_v3_import_normalizes_coverage_reserves_ids_and_continues(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src"))
    _seed(src)
    out = tmp_path / "export.jsonl"
    src.export(out)
    src.close()

    # Reproduce the v3 wire contract: the ledger existed, but chunk purpose,
    # digest cursors, summary provenance, and episode generations did not.
    downgraded = []
    for line in out.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj["type"] == "_meta":
            obj["version"] = 3
            obj["schema_version"] = 37
        elif obj["type"] == "session":
            obj["record"] = {
                key: obj["record"][key]
                for key in ("id", "started_at", "ended_at", "summary")
            }
        elif obj["type"] == "chunk":
            obj["record"].pop("chunk_kind", None)
        elif obj["type"] == "message_retention_coverage":
            # v37 accepted exact independent proofs but did not establish the
            # ordered-stream contract, so import must preserve it without
            # feeding it to the v38 digest reader.
            obj["record"]["coverage_version"] = "caller-defined-v37-proof"
        elif obj["type"] == "episode":
            obj["record"].pop("digest_slice_key", None)
            obj["record"].pop("digest_generation", None)
        elif obj["type"] == "user_profile_fact":
            continue
        downgraded.append(json.dumps(obj))
    out.write_text("\n".join(downgraded) + "\n", encoding="utf-8")

    payload = json.dumps({
        "episodes": [],
        "summary": "Combined the imported history with the appended tail.",
        "procedures": [],
    })
    base = HyMemConfig(root=tmp_path / "dst")
    cfg = dataclasses.replace(
        base,
        aggregation_nodes_enabled=False,
        facts_extraction_enabled=False,
        profile_extraction_enabled=False,
    )
    dst = HyMem(
        cfg,
        llm=StubLLMClient(
            fixtures={"Return the JSON object now": payload},
            default='{"triples":[],"markers":[]}',
        ),
    )
    try:
        dst.import_(out)
        new_id = dst.log_message("s1", "assistant", "new portable tail")
        assert new_id > 1
        dst.close_session("s1")
        dst.dream()

        assert dst.conn.execute(
            "SELECT COUNT(*) AS c FROM message_retention_coverage "
            "WHERE source_session_id = 's1'"
        ).fetchone()["c"] == 2
        row = dst.conn.execute(
            "SELECT coverage_message_id, digest_cursor_message_id, "
            "digested_prompt_version FROM sessions WHERE id = 's1'"
        ).fetchone()
        assert row["coverage_message_id"] == new_id
        assert row["digest_cursor_message_id"] == new_id
        assert row["digested_prompt_version"] == cfg.prompt_version
        imported_chunk_id = coverage_chunk_id("s1", 1)
        assert dst.conn.execute(
            "SELECT chunk_kind FROM chunks WHERE id = ?", (imported_chunk_id,)
        ).fetchone()["chunk_kind"] == "extraction"
    finally:
        dst.close()


def test_redaction_enabled_v6_import_never_writes_raw_profile_value_or_key(
    tmp_path,
):
    raw_values = (
        "friend first.secret@example.com",
        "colleague sk-ABCD1234efgh5678ijkl",
        "friend second.secret@example.net",
    )
    raw_keys = ("alice.private@example.com", "bob.private@example.com")
    src = HyMem(
        HyMemConfig(root=tmp_path / "src-secret", redact_secrets=False)
    )
    try:
        first = src.log_message(
            "secret-profile", "user", "first relationship assertion",
            created_at="2026-01-01T00:00:00Z",
        )
        second = src.log_message(
            "secret-profile", "user", "second relationship assertion",
            created_at="2026-02-01T00:00:00Z",
        )
        third = src.log_message(
            "secret-profile", "user", "third relationship assertion",
            created_at="2026-03-01T00:00:00Z",
        )
        with core_db.transaction(src.conn):
            persist_user_profile(
                src.conn,
                ProfileExtraction(items=[
                    {
                        "slot": "relationship",
                        "slot_key": raw_keys[0],
                        "value": raw_values[0],
                        "evidence_message_id": first,
                        "confidence": 0.9,
                    },
                    {
                        "slot": "relationship",
                        "slot_key": raw_keys[0],
                        "value": raw_values[1],
                        "evidence_message_id": second,
                        "confidence": 0.9,
                    },
                    {
                        "slot": "relationship",
                        "slot_key": raw_keys[1],
                        "value": raw_values[2],
                        "evidence_message_id": third,
                        "confidence": 0.9,
                    },
                ]),
                redact_values=False,
            )
        out = tmp_path / "secret-profile-v6.jsonl"
        src.export(out)
        assert all(secret in out.read_text(encoding="utf-8") for secret in (
            *raw_values,
            *raw_keys,
        ))
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "dst-secret", redact_secrets=True))
    traced_sql: list[str] = []
    try:
        dst.conn.set_trace_callback(traced_sql.append)
        dst.import_(out)
        dst.conn.set_trace_callback(None)

        rows = dst.conn.execute(
            "SELECT value, slot_key, invalid_at FROM user_profile ORDER BY id"
        ).fetchall()
        stored = json.dumps([dict(row) for row in rows])
        writes = "\n".join(traced_sql)
        for secret in (*raw_values, *raw_keys):
            assert secret not in writes
            assert secret not in stored
        assert len(rows) == 3
        assert sum(row["invalid_at"] is None for row in rows) == 2
        assert "[REDACTED-EMAIL]" in stored
        assert "[REDACTED-API-KEY]" in stored
        keys = {row["slot_key"] for row in rows}
        assert len(keys) == 2
        assert all(key.startswith("[redacted-email]#") for key in keys)
    finally:
        dst.conn.set_trace_callback(None)
        dst.close()


def test_redacted_import_scrubs_all_text_before_sql_and_keeps_coverage_valid(
    tmp_path,
):
    secret = "alice.private@example.com"
    src = HyMem(HyMemConfig(root=tmp_path / "all-text-source", redact_secrets=False))
    try:
        mid = src.log_message("all-text", "user", f"Contact {secret} for deploys.")
        src.conn.execute(
            "UPDATE sessions SET summary = ?, auto_summary = ? WHERE id = 'all-text'",
            (f"Summary for {secret}", f"Automatic summary for {secret}"),
        )
        src.conn.execute(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text, chunk_kind) VALUES "
            "('ordinary-secret', 'all-text', ?, ?, 'salient', ?, 'extraction')",
            (mid, mid, f"ordinary memory mentioning {secret}"),
        )
        src.conn.execute(
            "INSERT INTO episodes(id, session_id, title, summary, participants, "
            "key_entities) VALUES ('secret-episode', 'all-text', ?, ?, ?, ?)",
            (
                f"Email {secret}", f"Discussed {secret}",
                json.dumps([secret]), json.dumps([secret]),
            ),
        )
        src.conn.execute(
            "INSERT INTO procedures(id, session_id, name, description, steps, "
            "triggers, entities_involved) VALUES "
            "('secret-procedure', 'all-text', ?, ?, ?, ?, ?)",
            (
                f"Contact {secret}", f"Notify {secret}",
                json.dumps([{"order": 1, "action": f"email {secret}", "tool": None}]),
                json.dumps([secret]), json.dumps([secret]),
            ),
        )
        secret_edge_id = int(src.conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical) "
            "VALUES (?, 'uses', 'email')",
            ("contact_identity",),
        ).lastrowid)
        evidence_ledger.capture_unattributed_counts(
            src.conn, [secret_edge_id], reason="redaction test legacy edge"
        )
        secret_valid_at = src.conn.execute(
            "SELECT valid_at FROM knowledge_graph WHERE id=?", (secret_edge_id,)
        ).fetchone()[0]
        record_lifecycle_event(
            src.conn, edge_id=secret_edge_id, event_key="legacy-state",
            event_kind="legacy_state", direction=1, event_at=secret_valid_at,
            details="redaction test legacy edge",
        )
        src.conn.execute(
            "INSERT INTO profile_entries(kind, text) VALUES ('preference', ?)",
            (f"contact {secret}",),
        )
        out = tmp_path / "all-text-unredacted.jsonl"
        src.export(out)
        assert secret in out.read_text(encoding="utf-8")
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "all-text-target", redact_secrets=True))
    statements: list[str] = []
    try:
        dst.conn.set_trace_callback(statements.append)
        dst.import_(out)
        dst.conn.set_trace_callback(None)
        assert secret not in "\n".join(statements)
        assert secret not in "\n".join(dst.conn.iterdump())

        proof_row = dst.conn.execute(
            "SELECT message_id, chunk_id, coverage_version "
            "FROM message_retention_coverage WHERE source_session_id = 'all-text'"
        ).fetchone()
        proof = validate_message_coverage_artifact(
            dst.conn,
            message_id=proof_row["message_id"],
            chunk_id=proof_row["chunk_id"],
            coverage_version=proof_row["coverage_version"],
        )
        assert secret not in proof.content
        assert "[REDACTED-EMAIL]" in proof.content
        dst.export(tmp_path / "all-text-redacted-roundtrip.jsonl")
    finally:
        dst.conn.set_trace_callback(None)
        dst.close()


def test_v6_truncated_manifest_rejects_without_mutating_destination(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src-truncated"))
    try:
        _seed(src)
        out = tmp_path / "truncated-v6.jsonl"
        src.export(out)
    finally:
        src.close()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["type"] == "_end"
    out.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    dst = HyMem(HyMemConfig(root=tmp_path / "dst-truncated"))
    try:
        local_mid = dst.log_message("local", "user", "must survive rejection")
        tables = (
            "sessions",
            "chunks",
            "message_retention_coverage",
            "user_profile",
            "episodes",
        )
        before = {
            table: dst.conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in tables
        }
        with pytest.raises(ValueError, match="truncated|manifest"):
            dst.import_(out)
        after = {
            table: dst.conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in tables
        }
        assert after == before
        assert dst.conn.in_transaction is False
        assert dst.conn.execute(
            "SELECT content FROM messages WHERE id = ?", (local_mid,)
        ).fetchone()[0] == "must survive rejection"
    finally:
        dst.close()


@pytest.mark.parametrize(
    "poisoned_message_id",
    [
        pytest.param(2**63 - 1, id="sqlite-max"),
        pytest.param(2**63 - 1 - 999_999, id="inside-reserved-headroom"),
    ],
)
def test_v6_import_rejects_rowid_exhaustion_and_log_message_still_works(
    tmp_path, poisoned_message_id,
):
    src = HyMem(HyMemConfig(root=tmp_path / f"src-rowid-{poisoned_message_id}"))
    try:
        _seed(src)
        out = tmp_path / f"rowid-{poisoned_message_id}.jsonl"
        src.export(out)
    finally:
        src.close()

    def poison_coverage(body):
        coverage = next(
            obj for obj in body if obj["type"] == "message_retention_coverage"
        )
        coverage["record"]["message_id"] = poisoned_message_id

    _rewrite_v6_export(out, poison_coverage)

    dst = HyMem(HyMemConfig(root=tmp_path / f"dst-rowid-{poisoned_message_id}"))
    try:
        with pytest.raises(ValueError, match="rowid headroom"):
            dst.import_(out)
        assert dst.conn.execute(
            "SELECT COUNT(*) FROM message_retention_coverage"
        ).fetchone()[0] == 0
        assert dst.log_message("after-rejection", "user", "normal append") == 1
        assert dst.conn.execute(
            "SELECT content FROM messages WHERE id = 1"
        ).fetchone()[0] == "normal append"
    finally:
        dst.close()


def test_v6_import_rewinds_digest_cursor_that_is_not_a_sparse_session_member(
    tmp_path,
):
    src = HyMem(HyMemConfig(root=tmp_path / "src-sparse-cursor"))
    try:
        first = src.log_message("sparse", "user", "first sparse member")
        foreign = src.log_message("other", "user", "global id belongs elsewhere")
        last = src.log_message("sparse", "assistant", "last sparse member")
        assert (first, foreign, last) == (1, 2, 3)
        generation = (
            "lossless-digest-v2|prompt=test|episodes=blob|chars=12000|"
            "tokens=1200|episode-cap=blob|walk=" + ("d" * 32)
        )
        src.conn.execute(
            "UPDATE sessions SET digest_cursor_message_id = ?, "
            "digest_cursor_partial_message_id = NULL, digest_cursor_offset = 0, "
            "digest_cursor_prompt_version = ?, digest_published_generation = ?, "
            "digest_retry_count = 0, digest_retry_config_version = NULL, "
            "digest_quarantined = 0 WHERE id = 'sparse'",
            (foreign, generation, generation),
        )
        out = tmp_path / "sparse-cursor-v6.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "dst-sparse-cursor"))
    try:
        dst.import_(out)
        state = dst.conn.execute(
            "SELECT coverage_message_id, digest_cursor_message_id, "
            "digest_cursor_partial_message_id, digest_cursor_offset, "
            "digest_cursor_prompt_version, digest_published_generation, "
            "digest_retry_count, digest_retry_config_version, digest_quarantined "
            "FROM sessions WHERE id = 'sparse'"
        ).fetchone()
        assert state["coverage_message_id"] == last
        assert state["digest_cursor_message_id"] is None
        assert state["digest_cursor_partial_message_id"] is None
        assert state["digest_cursor_offset"] == 0
        assert state["digest_cursor_prompt_version"] is None
        assert state["digest_published_generation"] == generation
        assert state["digest_retry_count"] == 0
        assert state["digest_retry_config_version"] is None
        assert state["digest_quarantined"] == 0
    finally:
        dst.close()


def test_future_portable_version_is_rejected_before_mutation(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src-future-wire"))
    try:
        _seed(src)
        out = tmp_path / "future-wire.jsonl"
        src.export(out)
    finally:
        src.close()

    def future(body):
        body[0]["version"] = 999

    _rewrite_v6_export(out, future)
    dst = HyMem(HyMemConfig(root=tmp_path / "dst-future-wire"))
    try:
        with pytest.raises(ValueError, match="unsupported.*version"):
            dst.import_(out)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        dst.close()


@pytest.mark.parametrize("mutation", ["missing_column", "extra_column", "unknown_kind"])
def test_manifest_backed_v6_requires_its_exact_record_schema(tmp_path, mutation):
    src = HyMem(HyMemConfig(root=tmp_path / f"src-schema-{mutation}"))
    try:
        _seed(src)
        out = tmp_path / f"schema-{mutation}.jsonl"
        src.export(out)
    finally:
        src.close()

    def corrupt(body):
        session = next(obj for obj in body if obj["type"] == "session")
        if mutation == "missing_column":
            session["record"].pop("coverage_message_id")
        elif mutation == "extra_column":
            session["record"]["id) VALUES ('injected'); --"] = "ignored?"
        else:
            body.append({"type": "future_memory", "record": {"value": "lost"}})

    _rewrite_v6_export(out, corrupt)
    dst = HyMem(HyMemConfig(root=tmp_path / f"dst-schema-{mutation}"))
    try:
        with pytest.raises(ValueError, match="schema|unknown record"):
            dst.import_(out)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        dst.close()


def test_manifest_backed_invalid_profile_fact_rolls_back_whole_import(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "src-invalid-profile"))
    try:
        _seed(src)
        out = tmp_path / "invalid-profile.jsonl"
        src.export(out)
    finally:
        src.close()

    def corrupt(body):
        profile = next(obj for obj in body if obj["type"] == "user_profile_fact")
        profile["record"]["source_message_id"] = True

    _rewrite_v6_export(out, corrupt)
    dst = HyMem(HyMemConfig(root=tmp_path / "dst-invalid-profile"))
    try:
        with pytest.raises(ValueError, match="profile fact.*provenance"):
            dst.import_(out)
        for table in ("sessions", "chunks", "message_retention_coverage", "user_profile"):
            assert dst.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        dst.close()


def test_v6_export_rejects_corrupt_coverage_without_replacing_destination(
    tmp_path,
):
    src = HyMem(HyMemConfig(root=tmp_path / "src-corrupt-export"))
    destination = tmp_path / "existing-export.jsonl"
    original = b"pre-existing portable backup\x00must remain byte-exact\n"
    destination.write_bytes(original)
    try:
        _seed(src)
        chunk_id = src.conn.execute(
            "SELECT chunk_id FROM message_retention_coverage"
        ).fetchone()[0]
        # Simulate disk corruption beneath the normal immutable-artifact guard.
        src.conn.execute(
            "DROP TRIGGER message_retention_covered_chunk_update_guard"
        )
        src.conn.execute(
            "UPDATE chunks SET text = text || '\ncorrupt trailing bytes' "
            "WHERE id = ?",
            (chunk_id,),
        )

        with pytest.raises(
            (ValueError, RuntimeError), match="coverage|artifact|corrupt"
        ):
            src.export(destination)
        assert destination.read_bytes() == original
        assert src.conn.in_transaction is False
    finally:
        src.close()


def test_v6_export_rejects_pruned_profile_without_durable_provenance_atomically(
    tmp_path,
):
    src = HyMem(HyMemConfig(root=tmp_path / "src-orphan-profile"))
    destination = tmp_path / "existing-profile-export.jsonl"
    original = b"known-good backup bytes\n"
    destination.write_bytes(original)
    try:
        src.conn.execute("INSERT INTO sessions(id) VALUES ('orphan-profile')")
        # A legacy row can predate both the raw FK and durable source columns.
        # Such an unattributed row remains queryable locally but cannot be
        # authenticated by a v6 importer, so export must fail closed.
        # Simulate an irrecoverable pre-v39 row. Current v39 writes reject new
        # unattributed assertions at the SQL boundary.
        src.conn.execute("DROP TRIGGER user_profile_source_insert_guard")
        src.conn.execute(
            "INSERT INTO user_profile("
            "slot, value, confidence"
            ") VALUES ('location', 'Utrecht', 0.9)"
        )
        core_db._ensure_profile_active_invariants(src.conn)
        orphan = src.conn.execute(
            "SELECT evidence_message_id, source_message_id, source_session_id, "
            "source_created_at FROM user_profile"
        ).fetchone()
        assert tuple(orphan) == (None, None, None, None)

        with pytest.raises(ValueError, match="profile|provenance|source"):
            src.export(destination)
        assert destination.read_bytes() == original
        assert src.conn.in_transaction is False
    finally:
        src.close()


def test_export_inside_caller_transaction_preserves_state_and_destination(tmp_path):
    hy = HyMem(HyMemConfig(root=tmp_path / "nested-export-source"))
    destination = tmp_path / "existing-nested-export.jsonl"
    original = b"previous known-good export\n"
    destination.write_bytes(original)
    try:
        hy.conn.execute("BEGIN")
        hy.conn.execute("INSERT INTO sessions(id) VALUES ('uncommitted-session')")

        with pytest.raises(RuntimeError, match="caller-owned transaction"):
            hy.export(destination)

        assert hy.conn.in_transaction is True
        assert hy.conn.execute(
            "SELECT 1 FROM sessions WHERE id = 'uncommitted-session'"
        ).fetchone() is not None
        assert destination.read_bytes() == original
        assert list(tmp_path.glob(".existing-nested-export.jsonl.*.tmp")) == []

        hy.conn.execute("ROLLBACK")
        assert hy.conn.execute(
            "SELECT 1 FROM sessions WHERE id = 'uncommitted-session'"
        ).fetchone() is None
    finally:
        if hy.conn.in_transaction:
            hy.conn.execute("ROLLBACK")
        hy.close()


def test_generic_v37_exact_proof_roundtrips_without_becoming_ordered(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "generic-proof-source"))
    try:
        src.conn.execute("INSERT INTO sessions(id) VALUES ('generic-proof')")
        message_id = src.conn.execute(
            "INSERT INTO messages(session_id, role, content) "
            "VALUES ('generic-proof', 'user', 'exact generic source')"
        ).lastrowid
        record = encode_message_record(
            message_id=message_id, role="user", content="exact generic source"
        )
        src.conn.execute(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text, chunk_kind) VALUES "
            "('generic-chunk', 'generic-proof', ?, ?, 'caller-proof', ?, 'extraction')",
            (message_id, message_id, record),
        )
        record_message_coverage(
            src.conn,
            message_id=message_id,
            chunk_id="generic-chunk",
            coverage_version="caller-proof-v1",
        )
        out = tmp_path / "generic-proof.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "generic-proof-target"))
    try:
        dst.import_(out)
        row = dst.conn.execute(
            "SELECT c.chunk_kind, mc.coverage_version, s.coverage_message_id "
            "FROM message_retention_coverage mc "
            "JOIN chunks c ON c.id = mc.chunk_id "
            "JOIN sessions s ON s.id = mc.source_session_id"
        ).fetchone()
        assert tuple(row) == ("extraction", "caller-proof-v1", None)
    finally:
        dst.close()


def test_profile_quarantine_roundtrips_when_its_cursor_position_is_coherent(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "profile-quarantine-source"))
    retry_key = profile_retry_policy_version(
        profile_config_version(max_chars=12000, max_items=16),
        max_attempts=6,
    )
    try:
        src.log_message("profile-held", "user", "I live in Utrecht.")
        src.conn.execute(
            "UPDATE sessions SET profile_retry_count = 6, "
            "profile_retry_config_version = ?, profile_quarantined = 1 "
            "WHERE id = 'profile-held'",
            (retry_key,),
        )
        out = tmp_path / "profile-quarantine.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "profile-quarantine-target"))
    try:
        dst.import_(out)
        state = dst.conn.execute(
            "SELECT profile_cursor_message_id, profile_retry_count, "
            "profile_retry_config_version, profile_quarantined "
            "FROM sessions WHERE id = 'profile-held'"
        ).fetchone()
        assert tuple(state) == (None, 6, retry_key, 1)
    finally:
        dst.close()


def test_null_source_time_profile_history_roundtrips_without_reordering(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "null-time-profile-source"))
    try:
        src.open_session("null-time-profile")
        unknown = src.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) "
            "VALUES ('null-time-profile', 'user', 'I live in Utrecht.', NULL)"
        ).lastrowid
        from hymem.dreaming.lossless import materialize_message_coverage

        materialize_message_coverage(src.conn, "null-time-profile")
        persist_user_profile(src.conn, ProfileExtraction(items=[{
            "slot": "location", "value": "Utrecht",
            "evidence_message_id": unknown, "confidence": 0.9,
        }]), redact_values=False)
        known = src.log_message(
            "null-time-profile", "user", "I now live in Amsterdam.",
            created_at="2026-05-01T12:00:00Z",
        )
        persist_user_profile(src.conn, ProfileExtraction(items=[{
            "slot": "location", "value": "Amsterdam",
            "evidence_message_id": known, "confidence": 0.9,
        }]), redact_values=False)
        before = [tuple(row) for row in src.conn.execute(
            "SELECT value, source_message_id, source_created_at, valid_at, "
            "invalid_at FROM user_profile ORDER BY source_message_id"
        ).fetchall()]
        out = tmp_path / "null-time-profile.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "null-time-profile-target"))
    try:
        dst.import_(out)
        after = [tuple(row) for row in dst.conn.execute(
            "SELECT value, source_message_id, source_created_at, valid_at, "
            "invalid_at FROM user_profile ORDER BY source_message_id"
        ).fetchall()]
        assert after == before
        assert [row[0] for row in after if row[4] is None] == ["Amsterdam"]
    finally:
        dst.close()


@pytest.mark.parametrize("legacy_created_at", [None, "not-a-time"])
def test_v7_legacy_unknown_claim_source_roundtrips_exactly_and_idempotently(
    tmp_path, legacy_created_at
):
    src = HyMem(HyMemConfig(root=tmp_path / "legacy-claim-source"))
    try:
        src.open_session("claim-session")
        message_id = int(src.conn.execute(
            "INSERT INTO messages(session_id,role,content,created_at) "
            "VALUES ('claim-session','user','Legacy app uses SQLite',?)",
            (legacy_created_at,),
        ).lastrowid)
        with core_db.transaction(src.conn):
            materialize_message_coverage(src.conn, "claim-session")
        chunk = _portable_claim_chunk(
            src, chunk_id="legacy-unknown-claim", message_id=message_id,
            role="user", content="Legacy app uses SQLite",
        )
        _persist_portable_claim(
            src, chunk,
            [Triple("legacy_app", "uses", "sqlite", 1,
                    source_message_id=message_id)],
            prompt_version="v13",
        )
        out = tmp_path / "legacy-unknown-claim.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "legacy-claim-target"))
    try:
        dst.import_(out)
        evidence = dst.conn.execute(
            "SELECT source_created_at,source_event_at,published_at "
            "FROM kg_evidence"
        ).fetchone()
        expected_source_time = (
            None if legacy_created_at is None else "0001-01-01T00:00:00.000Z"
        )
        assert evidence["source_created_at"] == expected_source_time
        assert evidence["source_event_at"] == "0001-01-01T00:00:00.000Z"
        assert evidence["published_at"] is not None
        proof = dst.conn.execute(
            "SELECT source_created_at FROM message_retention_coverage "
            "WHERE message_id=?",
            (message_id,),
        ).fetchone()
        assert proof["source_created_at"] == expected_source_time
        before = _portable_claim_state(dst)
        assert sum(dst.import_(out).values()) == 0
        assert _portable_claim_state(dst) == before
    finally:
        dst.close()


def test_import_inside_caller_transaction_fails_without_touching_caller_state(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "nested-import-source"))
    try:
        _seed(src)
        out = tmp_path / "nested-import.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "nested-import-target"))
    try:
        dst.conn.execute("BEGIN")
        dst.conn.execute("INSERT INTO sessions(id) VALUES ('caller-owned')")
        with pytest.raises(RuntimeError, match="caller-owned transaction"):
            dst.import_(out)
        assert dst.conn.in_transaction is True
        assert dst.conn.execute(
            "SELECT 1 FROM sessions WHERE id = 'caller-owned'"
        ).fetchone() is not None
        dst.conn.execute("ROLLBACK")
    finally:
        if dst.conn.in_transaction:
            dst.conn.execute("ROLLBACK")
        dst.close()


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("procedure_status", "procedure.*status"),
        ("procedure_confidence", "strict JSON|non-finite"),
        ("boolean_manifest_count", "manifest|truncated"),
    ],
)
def test_v6_rejects_invalid_scalars_before_any_destination_write(
    tmp_path, mutation, error,
):
    src = HyMem(HyMemConfig(root=tmp_path / f"scalar-source-{mutation}"))
    try:
        _seed(src)
        out = tmp_path / f"scalar-{mutation}.jsonl"
        src.export(out)
    finally:
        src.close()

    if mutation == "procedure_status":
        _rewrite_v6_export(
            out,
            lambda body: next(
                row for row in body if row["type"] == "procedure"
            )["record"].__setitem__("status", "bogus"),
        )
    elif mutation == "procedure_confidence":
        _rewrite_v6_export(
            out,
            lambda body: next(
                row for row in body if row["type"] == "procedure"
            )["record"].__setitem__("confidence", float("inf")),
        )
    else:
        _rewrite_v6_export(
            out,
            lambda body: None,
        )
        objects = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
        ]
        objects[-1]["counts"]["procedure"] = True
        out.write_text(
            "\n".join(json.dumps(obj) for obj in objects) + "\n",
            encoding="utf-8",
        )

    dst = HyMem(HyMemConfig(root=tmp_path / f"scalar-target-{mutation}"))
    try:
        with pytest.raises(ValueError, match=error):
            dst.import_(out)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        dst.close()


def test_v6_duplicate_json_key_is_rejected_even_with_matching_manifest(tmp_path):
    src = HyMem(HyMemConfig(root=tmp_path / "duplicate-json-source"))
    try:
        _seed(src)
        out = tmp_path / "duplicate-json.jsonl"
        src.export(out)
    finally:
        src.close()

    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    body = lines[:-1]
    body[1] = body[1].replace(
        '"type": "session"',
        '"type": "session", "type": "session"',
        1,
    )
    end = json.loads(lines[-1])
    end["sha256"] = hashlib.sha256("".join(body).encode("utf-8")).hexdigest()
    out.write_text(
        "".join(body) + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dst = HyMem(HyMemConfig(root=tmp_path / "duplicate-json-target"))
    try:
        with pytest.raises(ValueError, match="strict JSON"):
            dst.import_(out)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        dst.close()


@pytest.mark.parametrize("first_content", ["alpha source", "beta source"])
def test_v6_coverage_collision_never_rebinds_profile_provenance(
    tmp_path, first_content,
):
    src = HyMem(HyMemConfig(root=tmp_path / f"collision-source-{first_content[0]}"))
    try:
        _seed(src)
        base = tmp_path / f"collision-{first_content[0]}.jsonl"
        src.export(base)
    finally:
        src.close()

    def rewrite_content(path, content):
        def mutate(body):
            chunk = next(row for row in body if row["type"] == "chunk")
            proof = next(
                row for row in body if row["type"] == "message_retention_coverage"
            )
            chunk["record"]["text"] = encode_message_record(
                message_id=1, role="user", content=content
            )
            proof["record"]["message_content_hash"] = message_content_hash(
                "user", content
            )
        _rewrite_v6_export(path, mutate)

    alpha = tmp_path / f"collision-alpha-{first_content[0]}.jsonl"
    beta = tmp_path / f"collision-beta-{first_content[0]}.jsonl"
    alpha.write_bytes(base.read_bytes())
    beta.write_bytes(base.read_bytes())
    rewrite_content(alpha, "alpha source")
    rewrite_content(beta, "beta source")
    first, second = (alpha, beta) if first_content == "alpha source" else (beta, alpha)

    dst = HyMem(HyMemConfig(root=tmp_path / f"collision-target-{first_content[0]}"))
    try:
        dst.import_(first)
        before = list(dst.conn.iterdump())
        with pytest.raises(ValueError, match="chunk.*collides"):
            dst.import_(second)
        assert list(dst.conn.iterdump()) == before
        assert dst.profile()[0].evidence_message_id == 1
    finally:
        dst.close()


@pytest.mark.parametrize("first_generation", ["portable-gen", "replacement-gen"])
def test_v6_session_publication_collision_aborts_without_hidden_children(
    tmp_path, first_generation,
):
    src = HyMem(HyMemConfig(
        root=tmp_path / f"generation-source-{first_generation}",
        redact_secrets=False,
    ))
    try:
        _seed(src)
        original = tmp_path / f"generation-original-{first_generation}.jsonl"
        replacement = tmp_path / f"generation-replacement-{first_generation}.jsonl"
        src.export(original)
        replacement.write_bytes(original.read_bytes())
    finally:
        src.close()

    def change_generation(body):
        next(row for row in body if row["type"] == "session")["record"][
            "digest_published_generation"
        ] = "replacement-gen"
        next(row for row in body if row["type"] == "episode")["record"][
            "digest_generation"
        ] = "replacement-gen"

    _rewrite_v6_export(replacement, change_generation)
    first, second = (
        (original, replacement)
        if first_generation == "portable-gen"
        else (replacement, original)
    )
    dst = HyMem(HyMemConfig(
        root=tmp_path / f"generation-target-{first_generation}",
        redact_secrets=False,
    ))
    try:
        dst.import_(first)
        before = [tuple(row) for row in dst.conn.execute(
            "SELECT id, digest_generation FROM episodes"
        ).fetchall()]
        with pytest.raises(ValueError, match="session.*collides"):
            dst.import_(second)
        after = [tuple(row) for row in dst.conn.execute(
            "SELECT id, digest_generation FROM episodes"
        ).fetchall()]
        assert after == before
        assert len(after) == 1
    finally:
        dst.close()


@pytest.mark.parametrize("stream", ["digest", "profile"])
def test_v6_retry_poison_is_rejected_before_mutation(tmp_path, stream):
    src = HyMem(HyMemConfig(root=tmp_path / f"retry-poison-source-{stream}"))
    try:
        _seed(src)
        out = tmp_path / f"retry-poison-{stream}.jsonl"
        src.export(out)
    finally:
        src.close()

    def poison(body):
        session = next(row for row in body if row["type"] == "session")["record"]
        session[f"{stream}_retry_count"] = 0
        session[f"{stream}_retry_config_version"] = None
        session[f"{stream}_quarantined"] = 1

    _rewrite_v6_export(out, poison)
    dst = HyMem(HyMemConfig(root=tmp_path / f"retry-poison-target-{stream}"))
    try:
        with pytest.raises(ValueError, match=f"invalid {stream} retry state"):
            dst.import_(out)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        dst.close()


def test_redacted_generic_multirecord_pem_proof_remains_exact_and_idempotent(
    tmp_path,
):
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEverysecretlineone\n"
        "MIIEverysecretlinetwo\n"
        "-----END PRIVATE KEY-----"
    )
    src = HyMem(HyMemConfig(root=tmp_path / "pem-source", redact_secrets=False))
    try:
        src.conn.execute("INSERT INTO sessions(id) VALUES ('pem')")
        mids = []
        records = []
        for content in ("first durable record", "second durable record"):
            mid = src.conn.execute(
                "INSERT INTO messages(session_id, role, content) "
                "VALUES ('pem', 'user', ?)",
                (content,),
            ).lastrowid
            mids.append(mid)
            records.append(encode_message_record(
                message_id=mid, role="user", content=content
            ))
        chunk_text = "\n".join([pem, records[0], "boundary text", records[1]])
        src.conn.execute(
            "INSERT INTO chunks(id, session_id, start_message_id, end_message_id, "
            "salience_reason, text, chunk_kind) VALUES "
            "('pem-generic', 'pem', ?, ?, 'caller-proof', ?, 'extraction')",
            (mids[0], mids[-1], chunk_text),
        )
        for mid in mids:
            record_message_coverage(
                src.conn, message_id=mid, chunk_id="pem-generic",
                coverage_version="caller-pem-proof-v1",
            )
        out = tmp_path / "pem.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "pem-target", redact_secrets=True))
    try:
        dst.import_(out)
        safe_text = dst.conn.execute(
            "SELECT text FROM chunks WHERE id = 'pem-generic'"
        ).fetchone()[0]
        assert "MIIEverysecret" not in safe_text
        assert "BEGIN PRIVATE KEY" not in safe_text
        assert "[REDACTED-PRIVATE-KEY]" in safe_text
        assert redaction.redact(safe_text) == safe_text
        proofs = dst.conn.execute(
            "SELECT message_id, chunk_id, coverage_version "
            "FROM message_retention_coverage ORDER BY message_id"
        ).fetchall()
        assert len(proofs) == 2
        for proof in proofs:
            validate_message_coverage_artifact(
                dst.conn,
                message_id=proof["message_id"],
                chunk_id=proof["chunk_id"],
                coverage_version=proof["coverage_version"],
            )
    finally:
        dst.close()


def test_redacted_import_preserves_distinct_natural_keys(tmp_path):
    first = "first_private_identity"
    second = "second_private_identity"
    profile_secrets = (
        "first.private@example.com", "second.private@example.com",
    )
    src = HyMem(HyMemConfig(root=tmp_path / "natural-redact-source", redact_secrets=False))
    try:
        for identity, profile_secret in zip((first, second), profile_secrets):
            edge_id = int(src.conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical, predicate, "
                "object_canonical) VALUES (?, 'uses', 'mail')",
                (identity,),
            ).lastrowid)
            evidence_ledger.capture_unattributed_counts(
                src.conn, [edge_id], reason="natural-key redaction test"
            )
            valid_at = src.conn.execute(
                "SELECT valid_at FROM knowledge_graph WHERE id=?", (edge_id,)
            ).fetchone()[0]
            record_lifecycle_event(
                src.conn, edge_id=edge_id, event_key="legacy-state",
                event_kind="legacy_state", direction=1, event_at=valid_at,
                details="natural-key redaction test",
            )
            src.conn.execute(
                "INSERT INTO profile_entries(kind, text) VALUES ('context', ?)",
                (profile_secret,),
            )
        out = tmp_path / "natural-redact.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(HyMemConfig(root=tmp_path / "natural-redact-target", redact_secrets=True))
    try:
        counts = dst.import_(out)
        assert counts["edge"] == 2
        assert counts["profile_entry"] == 2
        subjects = [row[0] for row in dst.conn.execute(
            "SELECT subject_canonical FROM knowledge_graph ORDER BY subject_canonical"
        )]
        entries = [row[0] for row in dst.conn.execute(
            "SELECT text FROM profile_entries ORDER BY text"
        )]
        assert len(set(subjects)) == len(set(entries)) == 2
        assert all("@" not in value for value in [*subjects, *entries])
        assert all(normalize(value) == value for value in subjects)
    finally:
        dst.close()


def test_redacted_v7_claim_scrubs_normalized_secret_graph_identities(tmp_path):
    secrets = (
        "alice.private@example.com",
        "bob.private@example.com",
    )
    src = HyMem(HyMemConfig(
        root=tmp_path / "normalized-secret-source", redact_secrets=False
    ))
    try:
        src.open_session("claim-session")
        for index, secret in enumerate(secrets):
            content = f"{secret} uses Redis"
            message_id = src.log_message(
                "claim-session", "user", content,
                created_at=f"2026-01-0{index + 1}T00:00:00Z",
            )
            with core_db.transaction(src.conn):
                materialize_message_coverage(src.conn, "claim-session")
            chunk = _portable_claim_chunk(
                src, chunk_id=f"normalized-secret-{index}",
                message_id=message_id, role="user", content=content,
            )
            _persist_portable_claim(
                src, chunk,
                [Triple(secret, "uses", "redis", 1,
                        source_message_id=message_id)],
                prompt_version="v13",
            )
        assert {
            row[0] for row in src.conn.execute(
                "SELECT subject_canonical FROM knowledge_graph"
            )
        } == {normalize(secret) for secret in secrets}
        wire = tmp_path / "normalized-secret.jsonl"
        src.export(wire)
    finally:
        src.close()

    target_root = tmp_path / "normalized-secret-target"
    dst = HyMem(HyMemConfig(root=target_root, redact_secrets=True))
    db_path = dst.config.db_path
    try:
        dst.import_(wire)
        assert sum(dst.import_(wire).values()) == 0
        subjects = [row[0] for row in dst.conn.execute(
            "SELECT subject_canonical FROM knowledge_graph ORDER BY subject_canonical"
        )]
        aliases = [tuple(row) for row in dst.conn.execute(
            "SELECT alias,canonical FROM entity_aliases ORDER BY alias"
        )]
        assert len(set(subjects)) == 2
        dump = "\n".join(dst.conn.iterdump())
        for secret in secrets:
            assert secret not in dump
            assert normalize(secret) not in dump
        assert all("private_example_com" not in value for row in aliases for value in row)
        dst.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        dst.close()
    raw = db_path.read_bytes()
    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists():
        raw += wal.read_bytes()
    for secret in secrets:
        assert secret.encode() not in raw
        assert normalize(secret).encode() not in raw
