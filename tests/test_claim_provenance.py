from __future__ import annotations

from pathlib import Path

import pytest
import sqlite3

from hymem import HyMem
from hymem.config import HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming import phase3
from hymem.dreaming import evidence as evidence_ledger
from hymem.dreaming import canonicalize
from hymem.dreaming.inference import infer_transitive_edges
from hymem.dreaming.retention import prune_chunks, prune_retracted_edges
from hymem.dreaming.value_supersession import supersede_competing_values
from hymem.dreaming.bitemporal import record_lifecycle_event
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.phase1 import ChunkExtraction
from hymem.extraction.triples import Triple
from hymem.extraction.markers import Marker


def _open(tmp_path: Path):
    conn = core_db.connect(tmp_path / "claim-provenance.sqlite")
    core_db.initialize(conn)
    conn.execute("INSERT INTO sessions(id) VALUES ('s')")
    return conn


def _messages(conn, rows: list[tuple[str, str, str]]) -> list[int]:
    ids: list[int] = []
    for role, content, created_at in rows:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) "
            "VALUES ('s', ?, ?, ?)",
            (role, content, created_at),
        )
        ids.append(int(cur.lastrowid))
    with core_db.transaction(conn):
        materialize_message_coverage(conn, "s")
    return ids


def _seed_manual_event(
    conn, *, edge_id: int, signal_key: str, event_at: str, details: str | None = None
) -> None:
    """Build a valid historical manual signal/event pair at a chosen instant."""
    with core_db.evidence_mutation(conn):
        conn.execute(
            "INSERT INTO kg_evidence_signals(edge_id,signal_key,signal_kind,"
            "polarity,evidence_weight,counts_toward_confidence,details,created_at) "
            "VALUES (?,?,'manual_retraction',-1,1,1,?,?)",
            (edge_id, signal_key, details, event_at),
        )
    evidence_ledger.reconcile_edge_counts(conn, [edge_id])
    record_lifecycle_event(
        conn, edge_id=edge_id,
        event_key=evidence_ledger.manual_retraction_event_key(signal_key),
        event_kind="manual_retraction", direction=-1, event_at=event_at,
        details=details,
    )


def _chunk(conn, chunk_id: str, message_ids: list[int]) -> Chunk:
    records = conn.execute(
        "SELECT id, role, content FROM messages WHERE id IN (%s) ORDER BY id"
        % ",".join("?" for _ in message_ids),
        message_ids,
    ).fetchall()
    chunk = Chunk(
        id=chunk_id,
        session_id="s",
        start_message_id=message_ids[0],
        end_message_id=message_ids[-1],
        salience_reason="test",
        text="\n".join(f"{row['role']}: {row['content']}" for row in records),
        source_message_ids=tuple(message_ids),
    )
    with core_db.transaction(conn):
        persist_chunks(conn, [chunk])
    return chunk


def _persist(
    conn,
    chunk: Chunk,
    triples: list[Triple],
    *,
    prompt_version: str,
    cfg: HyMemConfig,
    dedup_vectors: dict[str, list[float]] | None = None,
) -> None:
    sources = phase1._claim_sources_for_chunk(conn, chunk)
    extraction = ChunkExtraction(
        triples=triples,
        markers=[],
        claim_sources={source.message_id: source for source in sources},
        source_validated=True,
    )
    with core_db.transaction(conn):
        phase1.persist_chunk_results(
            conn,
            chunk,
            extraction,
            prompt_version=prompt_version,
            cfg=cfg,
            dedup_vectors=dedup_vectors,
        )


def test_mixed_role_sources_are_independent_and_weight_both_polarities(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        assistant_id, user_id = _messages(
            conn,
            [
                ("assistant", "The app uses Redis", "2026-01-03T10:00:00+02:00"),
                ("user", "No, it does not use Redis", "2026-01-03T09:00:00Z"),
            ],
        )
        chunk = _chunk(conn, "mixed", [assistant_id, user_id])
        _persist(
            conn,
            chunk,
            [
                Triple("app", "uses", "redis", 1, source_message_id=assistant_id),
                Triple("app", "uses", "redis", -1, source_message_id=user_id),
            ],
            prompt_version="v13",
            cfg=HyMemConfig(root=tmp_path),
        )

        edge = conn.execute(
            "SELECT pos_evidence, neg_evidence FROM knowledge_graph"
        ).fetchone()
        assert tuple(edge) == (1, 2)
        evidence = conn.execute(
            "SELECT source_message_id, source_role, polarity, evidence_weight, "
            "source_created_at, source_event_at FROM kg_evidence "
            "WHERE is_current = 1 ORDER BY source_message_id"
        ).fetchall()
        assert [tuple(row) for row in evidence] == [
            (
                assistant_id,
                "assistant",
                1,
                1,
                "2026-01-03T10:00:00+02:00",
                "2026-01-03T08:00:00.000Z",
            ),
            (
                user_id,
                "user",
                -1,
                2,
                "2026-01-03T09:00:00Z",
                "2026-01-03T09:00:00.000Z",
            ),
        ]
    finally:
        conn.close()


def test_successful_replay_is_exact_and_empty_retires_old_claim(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "The app uses Redis", "2026-02-01T12:00:00Z")]
        )
        chunk = _chunk(conn, "replay", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        claim = Triple("app", "uses", "redis", 1, source_message_id=message_id)
        _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)
        _persist(conn, chunk, [claim], prompt_version="v14", cfg=cfg)

        edge = conn.execute(
            "SELECT id, pos_evidence, status, valid_at, invalid_at "
            "FROM knowledge_graph"
        ).fetchone()
        assert (edge["pos_evidence"], edge["status"], edge["invalid_at"]) == (
            2,
            "active",
            None,
        )
        # One USER source contributes its configured weight once, despite replay.
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE is_current = 1"
        ).fetchone()[0] == 1

        _persist(conn, chunk, [], prompt_version="v15", cfg=cfg)
        edge = conn.execute(
            "SELECT pos_evidence, status, valid_at, invalid_at FROM knowledge_graph"
        ).fetchone()
        assert edge["pos_evidence"] == 0
        assert edge["status"] == "retracted"
        assert edge["invalid_at"] == edge["valid_at"]
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE is_current = 1"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def _overlap_state(tmp_path: Path, order: tuple[str, str]) -> tuple:
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "The app uses Redis", "2026-03-01T12:00:00Z")]
        )
        chunks = {
            name: _chunk(conn, f"overlap-{name}", [message_id])
            for name in ("a", "b")
        }
        cfg = HyMemConfig(root=tmp_path)
        claim = Triple("app", "uses", "redis", 1, source_message_id=message_id)
        for name in order:
            _persist(
                conn,
                chunks[name],
                [claim],
                prompt_version="v13",
                cfg=cfg,
            )
        # Both chunks now assert the same exact proof. Replaying B empty removes
        # only B's authority; A must keep the one globally deduped contribution.
        _persist(conn, chunks["b"], [], prompt_version="v14", cfg=cfg)
        edge = conn.execute(
            "SELECT pos_evidence, neg_evidence, status, valid_at, invalid_at "
            "FROM knowledge_graph"
        ).fetchone()
        current = conn.execute(
            "SELECT polarity, evidence_weight, revision, source_event_at "
            "FROM kg_evidence WHERE is_current = 1"
        ).fetchall()
        observations = conn.execute(
            "SELECT chunk_id, polarity, prompt_generation "
            "FROM kg_claim_observations ORDER BY chunk_id"
        ).fetchall()
        return tuple(edge), tuple(tuple(row) for row in current), tuple(
            tuple(row) for row in observations
        )
    finally:
        conn.close()


def test_overlapping_chunks_form_one_order_independent_source_union(tmp_path: Path):
    forward = _overlap_state(tmp_path / "forward", ("a", "b"))
    reverse = _overlap_state(tmp_path / "reverse", ("b", "a"))
    assert forward == reverse
    assert forward[0][0:3] == (2, 0, "active")
    assert len(forward[1]) == 1
    assert len(forward[2]) == 1


def test_prompt_authority_restores_full_older_interpretation_append_only(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "The target was 65 percent", "2026-04-01T12:00:00Z")]
        )
        older = _chunk(conn, "scope-old", [message_id])
        newer = _chunk(conn, "scope-new", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn,
            older,
            [Triple(
                "service", "has_attribute", "65_percent", 1,
                temporal_scope="2025", value_numeric=65, value_unit="percent",
                source_message_id=message_id,
            )],
            prompt_version="v13",
            cfg=cfg,
        )
        _persist(
            conn,
            newer,
            [Triple(
                "service", "has_attribute", "65_percent", 1,
                temporal_scope="2026", value_numeric=65, value_unit="percent",
                source_message_id=message_id,
            )],
            prompt_version="v14",
            cfg=cfg,
        )
        assert conn.execute(
            "SELECT temporal_scope FROM kg_evidence WHERE is_current = 1"
        ).fetchone()[0] == "2026"
        retired_before = {
            int(row[0]) for row in conn.execute(
                "SELECT id FROM kg_evidence WHERE is_current = 0"
            ).fetchall()
        }

        _persist(conn, newer, [], prompt_version="v15", cfg=cfg)

        current = conn.execute(
            "SELECT temporal_scope, value_numeric, value_unit, revision "
            "FROM kg_evidence WHERE is_current = 1"
        ).fetchone()
        assert tuple(current[:3]) == ("2025", 65.0, "percent")
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE is_current = 0"
        ).fetchone()[0] > len(retired_before)
        # Retirement is append-only: no historical revision was reactivated.
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE superseded_at IS NOT NULL"
        ).fetchone()[0] >= 2
        assert not retired_before.intersection({
            int(row[0]) for row in conn.execute(
                "SELECT id FROM kg_evidence WHERE is_current = 1"
            ).fetchall()
        })
    finally:
        conn.close()


def _generation_state(tmp_path: Path, order: tuple[str, str]) -> tuple[int, int, int]:
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "The app relation is disputed", "2026-05-01T12:00:00Z")]
        )
        chunks = {
            name: _chunk(conn, f"generation-{name}", [message_id])
            for name in ("old", "new")
        }
        cfg = HyMemConfig(root=tmp_path)
        payloads = {
            "old": (
                "v13",
                Triple("app", "uses", "redis", 1, source_message_id=message_id),
            ),
            "new": (
                "v14",
                Triple("app", "uses", "redis", -1, source_message_id=message_id),
            ),
        }
        for name in order:
            version, triple = payloads[name]
            _persist(conn, chunks[name], [triple], prompt_version=version, cfg=cfg)
        row = conn.execute(
            "SELECT kg.pos_evidence, kg.neg_evidence, ev.polarity "
            "FROM knowledge_graph kg JOIN kg_evidence ev ON ev.edge_id = kg.id "
            "WHERE ev.is_current = 1"
        ).fetchone()
        return tuple(row)
    finally:
        conn.close()


def test_highest_prompt_generation_wins_independent_of_processing_order(tmp_path: Path):
    assert _generation_state(tmp_path / "forward", ("old", "new")) == (0, 2, -1)
    assert _generation_state(tmp_path / "reverse", ("new", "old")) == (0, 2, -1)


def test_lower_generation_exact_replay_does_not_churn_restored_revision(tmp_path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "The app relation is disputed", "2024-01-01")]
        )
        chunks = {
            name: _chunk(conn, f"stable-replay-{name}", [message_id])
            for name in ("winner", "loser")
        }
        cfg = HyMemConfig(root=tmp_path)
        positive = Triple(
            "app", "uses", "redis", 1, source_message_id=message_id
        )
        negative = Triple(
            "app", "uses", "redis", -1, source_message_id=message_id
        )
        _persist(
            conn, chunks["winner"], [positive], prompt_version="v15", cfg=cfg
        )
        _persist(
            conn, chunks["loser"], [negative], prompt_version="v14", cfg=cfg
        )
        before_rows = list(conn.iterdump())
        from hymem import portability

        before_wire = tmp_path / "stable-before.jsonl"
        after_wire = tmp_path / "stable-after.jsonl"
        portability.export_jsonl(conn, before_wire)

        _persist(
            conn, chunks["loser"], [negative], prompt_version="v14", cfg=cfg
        )

        portability.export_jsonl(conn, after_wire)
        assert list(conn.iterdump()) == before_rows
        assert after_wire.read_bytes() == before_wire.read_bytes()
        current = conn.execute(
            "SELECT polarity,published_at FROM kg_evidence WHERE is_current=1"
        ).fetchone()
        assert current["polarity"] == 1
        assert current["published_at"] is not None
    finally:
        conn.close()


def test_exact_claim_replay_persists_new_auxiliary_projections_without_churn(
    tmp_path,
):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "The app uses Redis", "2024-01-01")]
        )
        chunk = _chunk(conn, "auxiliary-replay", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        claim = Triple("app", "uses", "redis", 1, source_message_id=message_id)
        _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)
        evidence_before = [tuple(row) for row in conn.execute(
            "SELECT * FROM kg_evidence ORDER BY id"
        ).fetchall()]
        outcome_before = tuple(conn.execute(
            "SELECT * FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone())
        sources = phase1._claim_sources_for_chunk(conn, chunk)
        replay = ChunkExtraction(
            triples=[claim],
            markers=[Marker("preference", "Prefer Redis")],
            entity_type_hints={"app": "software"},
            entity_property_hints={"app": {"tier": "critical"}},
            claim_sources={source.message_id: source for source in sources},
            source_validated=True,
        )
        with core_db.transaction(conn):
            phase1.persist_chunk_results(
                conn, chunk, replay, prompt_version="v13", cfg=cfg
            )

        assert [tuple(row) for row in conn.execute(
            "SELECT * FROM kg_evidence ORDER BY id"
        ).fetchall()] == evidence_before
        assert tuple(conn.execute(
            "SELECT * FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()) == outcome_before
        assert conn.execute(
            "SELECT COUNT(*) FROM behavioral_markers WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT type FROM entity_types WHERE entity_canonical='app'"
        ).fetchone()[0] == "software"
        assert conn.execute(
            "SELECT value FROM entity_properties "
            "WHERE entity_canonical='app' AND key='tier'"
        ).fetchone()[0] == "critical"
    finally:
        conn.close()


def test_exact_replay_repairs_staged_publication_and_missing_assertion(tmp_path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "The app uses Redis", "2024-01-01")]
        )
        chunk = _chunk(conn, "repair-staged", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        claim = Triple("app", "uses", "redis", 1, source_message_id=message_id)
        _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)
        evidence_id = int(conn.execute("SELECT id FROM kg_evidence").fetchone()[0])
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "UPDATE kg_evidence SET extracted_at=?,published_at=? WHERE id=?",
                (
                    "2024-01-02T00:00:00.000Z",
                    "2024-01-02T00:03:00.000Z",
                    evidence_id,
                ),
            )
            conn.execute(
                "UPDATE kg_edge_lifecycle SET created_at=? "
                "WHERE source_evidence_id=?",
                ("2024-01-02T00:01:00.000Z", evidence_id),
            )
            conn.execute(
                "UPDATE kg_claim_observations SET observed_at=? "
                "WHERE evidence_id=?",
                ("2024-01-02T00:02:00.000Z", evidence_id),
            )
            conn.execute(
                "UPDATE kg_claim_extraction_outcomes SET succeeded_at=? "
                "WHERE chunk_id=?",
                ("2024-01-02T00:03:00.000Z", chunk.id),
            )
        conn.execute("DROP TRIGGER kg_evidence_published_at_update_guard")
        with core_db.evidence_history_mutation(conn):
            conn.execute(
                "DELETE FROM kg_edge_lifecycle WHERE source_evidence_id=?",
                (evidence_id,),
            )
            conn.execute(
                "UPDATE kg_evidence SET published_at=NULL WHERE id=?",
                (evidence_id,),
            )
        core_db._ensure_post_migration_runtime_guards(conn)

        _persist(conn, chunk, [claim], prompt_version="v13", cfg=cfg)

        assert conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 1
        assert conn.execute(
            "SELECT published_at FROM kg_evidence WHERE id=?", (evidence_id,)
        ).fetchone()[0] > "2024-01-02T00:03:00.000Z"
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle "
            "WHERE source_evidence_id=? AND event_kind='claim_assertion'",
            (evidence_id,),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_same_generation_semantic_divergence_rolls_back_atomically(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "The service target changed", "2026-06-01T12:00:00Z")]
        )
        first = _chunk(conn, "divergence-a", [message_id])
        second = _chunk(conn, "divergence-b", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        base = dict(
            subject="service",
            predicate="has_attribute",
            object="65_percent",
            polarity=1,
            source_message_id=message_id,
        )
        _persist(
            conn, first, [Triple(**base, temporal_scope="2025")],
            prompt_version="v14", cfg=cfg,
        )
        with pytest.raises(ValueError, match="same-generation"):
            _persist(
                conn, second, [Triple(**base, temporal_scope="2026")],
                prompt_version="v14", cfg=cfg,
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_claim_observations"
        ).fetchone()[0] == 1
        current = conn.execute(
            "SELECT temporal_scope FROM kg_evidence WHERE is_current = 1"
        ).fetchone()
        assert current[0] == "2025"
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("first_closes", "second_closes"),
    [
        (("feb", "mar"), ("may", "june")),
        (("mar", "feb"), ("june", "may")),
    ],
)
def test_lifecycle_reducer_keeps_first_close_until_a_later_reopen(
    tmp_path: Path,
    first_closes: tuple[str, str],
    second_closes: tuple[str, str],
):
    conn = _open(tmp_path)
    try:
        first_id, second_id = _messages(
            conn,
            [
                ("user", "The app uses Redis", "2026-01-01T00:00:00Z"),
                ("user", "The app again uses Redis", "2026-04-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        first = _chunk(conn, "interval-first", [first_id])
        second = _chunk(conn, "interval-second", [second_id])
        _persist(
            conn, first,
            [Triple("app", "uses", "redis", 1, source_message_id=first_id)],
            prompt_version="v13", cfg=cfg,
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        close_times = {
            "feb": "2026-02-01T00:00:00Z",
            "mar": "2026-03-01T00:00:00Z",
            "may": "2026-05-01T00:00:00Z",
            "june": "2026-06-01T00:00:00Z",
        }
        for label in first_closes:
            _seed_manual_event(
                conn, edge_id=edge_id, signal_key=f"manual:{label}",
                event_at=close_times[label],
            )
        closed = conn.execute(
            "SELECT valid_at, invalid_at FROM knowledge_graph WHERE id = ?",
            (edge_id,),
        ).fetchone()
        assert tuple(closed) == (
            "2026-01-01T00:00:00.000Z", "2026-02-01T00:00:00.000Z"
        )

        _persist(
            conn, second,
            [Triple("app", "uses", "redis", 1, source_message_id=second_id)],
            prompt_version="v13", cfg=cfg,
        )
        for label in second_closes:
            _seed_manual_event(
                conn, edge_id=edge_id, signal_key=f"manual:{label}",
                event_at=close_times[label],
            )
        closed_again = conn.execute(
            "SELECT valid_at, invalid_at FROM knowledge_graph WHERE id = ?",
            (edge_id,),
        ).fetchone()
        assert tuple(closed_again) == (
            "2026-04-01T00:00:00.000Z", "2026-05-01T00:00:00.000Z"
        )
    finally:
        conn.close()


def test_phase3_decision_is_revoked_when_its_negative_cause_is_replayed_empty(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        positive_id, negative_id = _messages(
            conn,
            [
                ("user", "The app uses Redis", "2026-01-01T00:00:00Z"),
                ("user", "The app does not use Redis", "2026-02-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        positive = _chunk(conn, "cause-positive", [positive_id])
        negative = _chunk(conn, "cause-negative", [negative_id])
        _persist(
            conn, positive,
            [Triple("app", "uses", "redis", 1, source_message_id=positive_id)],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, negative,
            [Triple("app", "uses", "redis", -1, source_message_id=negative_id)],
            prompt_version="v13", cfg=cfg,
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        conn.execute(
            "UPDATE knowledge_graph SET last_reinforced = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (edge_id,),
        )
        phase3.decay(
            conn,
            HyMemConfig(
                root=tmp_path,
                retract_threshold=0.6,
                zombie_neg_threshold=100,
            ),
        )
        assert conn.execute(
            "SELECT status FROM knowledge_graph WHERE id = ?", (edge_id,)
        ).fetchone()[0] == "retracted"
        lifecycle = conn.execute(
            "SELECT id, dependency_count FROM kg_edge_lifecycle "
            "WHERE edge_id = ? AND event_kind = 'phase3_retraction'",
            (edge_id,),
        ).fetchone()
        assert lifecycle["dependency_count"] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_lifecycle_dependencies "
            "WHERE lifecycle_id = ?",
            (lifecycle["id"],),
        ).fetchone()[0] == 1
        dependency_id = int(conn.execute(
            "SELECT evidence_id FROM kg_lifecycle_dependencies "
            "WHERE lifecycle_id = ?",
            (lifecycle["id"],),
        ).fetchone()[0])
        for statement, parameters in (
            (
                "INSERT INTO kg_lifecycle_dependencies(lifecycle_id,evidence_id) "
                "VALUES (?, ?)",
                (lifecycle["id"], dependency_id),
            ),
            (
                "UPDATE kg_lifecycle_dependencies SET evidence_id = evidence_id "
                "WHERE lifecycle_id = ? AND evidence_id = ?",
                (lifecycle["id"], dependency_id),
            ),
            (
                "DELETE FROM kg_lifecycle_dependencies "
                "WHERE lifecycle_id = ? AND evidence_id = ?",
                (lifecycle["id"], dependency_id),
            ),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(statement, parameters)

        _persist(conn, negative, [], prompt_version="v14", cfg=cfg)
        reopened = conn.execute(
            "SELECT status, valid_at, invalid_at FROM knowledge_graph WHERE id = ?",
            (edge_id,),
        ).fetchone()
        assert tuple(reopened) == (
            "active", "2026-01-01T00:00:00.000Z", None,
        )
    finally:
        conn.close()


def test_lifecycle_helper_rolls_back_partial_dependencies_and_exact_retry(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        positive_id, negative_id = _messages(
            conn,
            [
                ("user", "App uses Redis", "2026-01-01T00:00:00Z"),
                ("user", "App does not use Redis", "2026-02-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, _chunk(conn, "atomic-positive", [positive_id]),
            [Triple("app", "uses", "redis", 1, source_message_id=positive_id)],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, _chunk(conn, "atomic-negative", [negative_id]),
            [Triple("app", "uses", "redis", -1, source_message_id=negative_id)],
            prompt_version="v13", cfg=cfg,
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        cause_id = int(conn.execute(
            "SELECT id FROM kg_evidence WHERE polarity=-1 AND is_current=1"
        ).fetchone()[0])
        before = tuple(conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph WHERE id=?",
            (edge_id,),
        ).fetchone())
        with pytest.raises(ValueError, match="missing"):
            record_lifecycle_event(
                conn, edge_id=edge_id, event_key="atomic-outside",
                event_kind="phase3_retraction", direction=-1,
                event_at="2026-02-01T00:00:00Z",
                dependency_evidence_ids=[cause_id, 999999],
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE event_key='atomic-outside'"
        ).fetchone()[0] == 0
        assert tuple(conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph WHERE id=?",
            (edge_id,),
        ).fetchone()) == before
        assert record_lifecycle_event(
            conn, edge_id=edge_id,
            event_key=evidence_ledger.phase3_retraction_event_key(conn, [cause_id]),
            event_kind="phase3_retraction", direction=-1,
            event_at="2026-02-01T00:00:00Z",
            dependency_evidence_ids=[cause_id],
            details="confidence_or_negative_dominance",
        )

        with core_db.transaction(conn):
            with pytest.raises(ValueError, match="missing"):
                record_lifecycle_event(
                    conn, edge_id=edge_id, event_key="atomic-nested",
                    event_kind="phase3_retraction", direction=-1,
                    event_at="2026-02-01T00:00:00Z",
                    dependency_evidence_ids=[cause_id, 999999],
                )
            assert conn.execute(
                "SELECT COUNT(*) FROM kg_edge_lifecycle "
                "WHERE event_key='atomic-nested'"
            ).fetchone()[0] == 0
            assert not record_lifecycle_event(
                conn, edge_id=edge_id,
                event_key=evidence_ledger.phase3_retraction_event_key(conn, [cause_id]),
                event_kind="phase3_retraction", direction=-1,
                event_at="2026-02-01T00:00:00Z",
                dependency_evidence_ids=[cause_id],
                details="confidence_or_negative_dominance",
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_lifecycle_dependencies dep "
            "JOIN kg_edge_lifecycle lifecycle ON lifecycle.id=dep.lifecycle_id "
            "WHERE lifecycle.event_kind='phase3_retraction'"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_lifecycle_helper_rejects_noncanonical_times_and_details(tmp_path: Path):
    from hymem.dreaming.bitemporal import evidence_event_at

    conn = _open(tmp_path)
    try:
        positive_id, negative_id, old_value_id, new_value_id = _messages(
            conn,
            [
                ("user", "App uses Redis", "2026-01-01T00:00:00Z"),
                ("user", "App does not use Redis", "2026-02-01T00:00:00Z"),
                ("user", "Target is 65 percent", "2026-03-01T00:00:00Z"),
                ("user", "Target is 78 percent", "2026-04-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, _chunk(conn, "canonical-positive", [positive_id]),
            [Triple("app", "uses", "redis", 1,
                    source_message_id=positive_id)],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, _chunk(conn, "canonical-negative", [negative_id]),
            [Triple("app", "uses", "redis", -1,
                    source_message_id=negative_id)],
            prompt_version="v13", cfg=cfg,
        )
        for message_id, value in (
            (old_value_id, "65_percent"), (new_value_id, "78_percent")
        ):
            _persist(
                conn, _chunk(conn, f"canonical-{value}", [message_id]),
                [Triple("service", "has_attribute", value, 1,
                        source_message_id=message_id)],
                prompt_version="v13", cfg=cfg,
            )

        redis_edge = int(conn.execute(
            "SELECT id FROM knowledge_graph WHERE subject_canonical='app'"
        ).fetchone()[0])
        negative_evidence = int(conn.execute(
            "SELECT id FROM kg_evidence WHERE edge_id=? AND polarity=-1",
            (redis_edge,),
        ).fetchone()[0])
        phase_key = evidence_ledger.phase3_retraction_event_key(
            conn, [negative_evidence]
        )
        for event_at, details in (
            ("2026-02-02T00:00:00Z", "confidence_or_negative_dominance"),
            (evidence_event_at(conn, negative_evidence), "forged decision"),
        ):
            with pytest.raises(ValueError, match="phase3 lifecycle semantics"):
                record_lifecycle_event(
                    conn, edge_id=redis_edge, event_key=phase_key,
                    event_kind="phase3_retraction", direction=-1,
                    event_at=event_at,
                    dependency_evidence_ids=[negative_evidence], details=details,
                )

        loser_edge = int(conn.execute(
            "SELECT id FROM knowledge_graph WHERE object_canonical='65_percent'"
        ).fetchone()[0])
        winner_evidence = int(conn.execute(
            "SELECT ev.id FROM kg_evidence ev JOIN knowledge_graph kg "
            "ON kg.id=ev.edge_id WHERE kg.object_canonical='78_percent'"
        ).fetchone()[0])
        wrong_value_at = "2026-04-02T00:00:00.000Z"
        value_key = evidence_ledger.value_supersession_event_key(
            conn, loser_edge_id=loser_edge,
            winner_evidence_id=winner_evidence, event_at=wrong_value_at,
        )
        with pytest.raises(ValueError, match="value lifecycle semantics"):
            record_lifecycle_event(
                conn, edge_id=loser_edge, event_key=value_key,
                event_kind="value_supersession", direction=-1,
                event_at=wrong_value_at,
                dependency_evidence_ids=[winner_evidence],
                details="newer typed value superseded this edge",
            )
        correct_value_at = evidence_event_at(conn, winner_evidence)
        correct_value_key = evidence_ledger.value_supersession_event_key(
            conn, loser_edge_id=loser_edge,
            winner_evidence_id=winner_evidence, event_at=correct_value_at,
        )
        with pytest.raises(ValueError, match="value lifecycle semantics"):
            record_lifecycle_event(
                conn, edge_id=loser_edge, event_key=correct_value_key,
                event_kind="value_supersession", direction=-1,
                event_at=correct_value_at,
                dependency_evidence_ids=[winner_evidence], details="forged decision",
            )

        _seed_manual_event(
            conn, edge_id=redis_edge, signal_key="canonical-host",
            event_at="2026-05-01T00:00:00Z", details="operator",
        )
        with pytest.raises(ValueError, match="matching signal"):
            record_lifecycle_event(
                conn, edge_id=redis_edge,
                event_key=evidence_ledger.manual_retraction_event_key(
                    "canonical-host"
                ),
                event_kind="manual_retraction", direction=-1,
                event_at="2026-05-02T00:00:00Z", details="operator",
            )
        assertion = conn.execute(
            "SELECT lifecycle.event_key,lifecycle.source_evidence_id "
            "FROM kg_edge_lifecycle lifecycle WHERE lifecycle.edge_id=? "
            "AND lifecycle.event_kind='claim_assertion'",
            (redis_edge,),
        ).fetchone()
        with pytest.raises(ValueError, match="details must be empty"):
            record_lifecycle_event(
                conn, edge_id=redis_edge, event_key=assertion["event_key"],
                event_kind="claim_assertion", direction=1,
                event_at="2026-01-01T00:00:00Z",
                source_evidence_id=assertion["source_evidence_id"],
                details="forged assertion",
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE event_kind IN "
            "('phase3_retraction','value_supersession')"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_evidence_reinterpret_failure_rolls_back_retirement_and_retries(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2026-01-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "atomic-evidence", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, chunk,
            [Triple("app", "uses", "redis", 1, temporal_scope="2025",
                    source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
        )
        row = conn.execute(
            "SELECT * FROM kg_evidence WHERE is_current=1"
        ).fetchone()
        edge_before = tuple(conn.execute(
            "SELECT pos_evidence,neg_evidence,status,valid_at,invalid_at "
            "FROM knowledge_graph WHERE id=?", (row["edge_id"],)
        ).fetchone())
        with pytest.raises(sqlite3.IntegrityError, match="provenance mismatch"):
            evidence_ledger.record_chunk_evidence(
                conn,
                edge_id=int(row["edge_id"]), chunk_id=row["chunk_id"],
                evidence_kind=row["evidence_kind"], polarity=1,
                evidence_weight=int(row["evidence_weight"]),
                weight_source=row["weight_source"], prompt_version="v14",
                source_role="assistant", surface_subject=row["surface_subject"],
                surface_object=row["surface_object"], temporal_scope="2026",
                source_message_id=int(row["source_message_id"]),
                source_session_id=row["source_session_id"],
                source_created_at=row["source_created_at"],
                source_event_at=row["source_event_at"],
                source_coverage_chunk_id=row["source_coverage_chunk_id"],
                source_coverage_version=row["source_coverage_version"],
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE is_current=1"
        ).fetchone()[0] == 1
        assert tuple(conn.execute(
            "SELECT pos_evidence,neg_evidence,status,valid_at,invalid_at "
            "FROM knowledge_graph WHERE id=?", (row["edge_id"],)
        ).fetchone()) == edge_before
        mutation = evidence_ledger.record_chunk_evidence(
            conn,
            edge_id=int(row["edge_id"]), chunk_id=row["chunk_id"],
            evidence_kind=row["evidence_kind"], polarity=1,
            evidence_weight=int(row["evidence_weight"]),
            weight_source=row["weight_source"], prompt_version="v14",
            source_role="user", surface_subject=row["surface_subject"],
            surface_object=row["surface_object"], temporal_scope="2026",
            source_message_id=int(row["source_message_id"]),
            source_session_id=row["source_session_id"],
            source_created_at=row["source_created_at"],
            source_event_at=row["source_event_at"],
            source_coverage_chunk_id=row["source_coverage_chunk_id"],
            source_coverage_version=row["source_coverage_version"],
        )
        assert mutation.inserted
        assert conn.execute(
            "SELECT temporal_scope FROM kg_evidence WHERE is_current=1"
        ).fetchone()[0] == "2026"
    finally:
        conn.close()


def test_signal_lifecycle_collision_rolls_back_signal_and_retries(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2026-01-01T00:00:00Z")]
        )
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, _chunk(conn, "atomic-signal", [message_id]),
            [Triple("app", "uses", "redis", 1, source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        _seed_manual_event(
            conn, edge_id=edge_id, signal_key="collision",
            event_at="2026-02-01T00:00:00Z", details="preexisting",
        )
        before = tuple(conn.execute(
            "SELECT pos_evidence,neg_evidence,status,valid_at,invalid_at "
            "FROM knowledge_graph WHERE id=?", (edge_id,)
        ).fetchone())
        with pytest.raises(ValueError, match="collides"):
            evidence_ledger.record_signal(
                conn, edge_id=edge_id, signal_key="collision",
                signal_kind="manual_retraction", polarity=-1,
                details="different",
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence_signals WHERE signal_key='collision'"
        ).fetchone()[0] == 1
        assert tuple(conn.execute(
            "SELECT pos_evidence,neg_evidence,status,valid_at,invalid_at "
            "FROM knowledge_graph WHERE id=?", (edge_id,)
        ).fetchone()) == before
        lifecycle_id = int(conn.execute(
            "SELECT id FROM kg_edge_lifecycle "
            "WHERE event_key='manual-retraction:collision'"
        ).fetchone()[0])
        with core_db.evidence_history_mutation(conn):
            conn.execute("DELETE FROM kg_edge_lifecycle WHERE id=?", (lifecycle_id,))
            conn.execute(
                "DELETE FROM kg_evidence_signals WHERE edge_id=? AND signal_key='collision'",
                (edge_id,),
            )
        evidence_ledger.reconcile_edge_counts(conn, [edge_id])
        from hymem.dreaming.bitemporal import recompute_edge_interval
        recompute_edge_interval(conn, edge_id)
        assert evidence_ledger.record_signal(
            conn, edge_id=edge_id, signal_key="collision",
            signal_kind="manual_retraction", polarity=-1,
            details="different",
        )
        assert conn.execute(
            "SELECT neg_evidence FROM knowledge_graph WHERE id=?", (edge_id,)
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_signal_exact_idempotence_and_semantic_collision_are_atomic(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical) "
            "VALUES ('app','uses','redis')"
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        assert evidence_ledger.record_signal(
            conn, edge_id=edge_id, signal_key="host", signal_kind="host_signal",
            polarity=1, evidence_weight=2, details="exact",
        )
        assert not evidence_ledger.record_signal(
            conn, edge_id=edge_id, signal_key="host", signal_kind="host_signal",
            polarity=1, evidence_weight=2, details="exact",
        )
        for changed in (
            {"polarity": -1}, {"evidence_weight": 3}, {"details": "changed"},
        ):
            payload = {
                "polarity": 1, "evidence_weight": 2, "details": "exact", **changed,
            }
            with pytest.raises(ValueError, match="collides"):
                evidence_ledger.record_signal(
                    conn, edge_id=edge_id, signal_key="host",
                    signal_kind="host_signal", **payload,
                )
        with pytest.raises(ValueError, match="negative polarity"):
            evidence_ledger.record_signal(
                conn, edge_id=edge_id, signal_key="bad-manual",
                signal_kind="manual_retraction", polarity=1,
            )
        assert conn.execute(
            "SELECT pos_evidence,neg_evidence FROM knowledge_graph WHERE id=?",
            (edge_id,),
        ).fetchone()[:] == (2, 0)
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence_signals"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_lifecycle_sql_guards_reject_forged_or_mutated_events(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        message_ids = _messages(
            conn,
            [
                ("user", "A uses B", "2026-01-01T00:00:00Z"),
                ("user", "C uses D", "2026-01-02T00:00:00Z"),
                ("user", "A does not use B", "2026-01-03T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        for index, (subject, obj) in enumerate((("a", "b"), ("c", "d"))):
            chunk = _chunk(conn, f"guard-{index}", [message_ids[index]])
            _persist(
                conn, chunk,
                [Triple(
                    subject, "uses", obj, 1,
                    source_message_id=message_ids[index],
                )],
                prompt_version="v13", cfg=cfg,
            )
        negative_chunk = _chunk(conn, "guard-negative", [message_ids[2]])
        _persist(
            conn, negative_chunk,
            [Triple("a", "uses", "b", -1, source_message_id=message_ids[2])],
            prompt_version="v13", cfg=cfg,
        )
        copy_chunk = _chunk(conn, "guard-copy", [message_ids[0]])
        edges = conn.execute("SELECT id FROM knowledge_graph ORDER BY id").fetchall()
        evidence_row = conn.execute(
            "SELECT * FROM kg_evidence WHERE edge_id = ? AND is_current = 1 "
            "AND polarity=1",
            (edges[0][0],),
        ).fetchone()
        evidence_id = int(evidence_row["id"])
        # The authorization gate rejects even otherwise-valid mutations: each
        # helper owns the matching cache/interval reconciliation.
        with pytest.raises(sqlite3.IntegrityError, match="internally managed"):
            conn.execute(
                "INSERT INTO kg_evidence(edge_id,chunk_id,evidence_kind,polarity,"
                "evidence_weight,weight_source,provenance_status,interpretation_key) "
                "VALUES (?, 'guard-0', 'decay', -1, 1, 'direct', "
                "'legacy_unattributed', 'direct-valid')",
                (edges[0][0],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="internally managed"):
            conn.execute(
                "INSERT INTO kg_evidence_signals(edge_id,signal_key,signal_kind,"
                "polarity,evidence_weight) VALUES (?, 'raw', 'manual', 1, 1)",
                (edges[0][0],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="internally managed"):
            conn.execute(
                "INSERT INTO kg_claim_observations(chunk_id,edge_id,"
                "source_session_id,source_message_id,evidence_kind,polarity,"
                "prompt_version,prompt_generation,evidence_id,interpretation_key) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    copy_chunk.id, int(edges[0][0]), "s", message_ids[0],
                    "extraction", 1, "v13", 13, evidence_id,
                    evidence_row["interpretation_key"],
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,"
                "direction,event_at) VALUES (?, 'valid-manual', "
                "'manual_retraction', -1, '2026-01-03T00:00:00.000Z')",
                (edges[0][0],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,"
                "direction,event_at,source_evidence_id) "
                "VALUES (?, 'forged', 'claim_assertion', 1, "
                "'2026-01-01T00:00:00.000Z', ?)",
                (edges[1][0], evidence_id),
            )
        # Authorized internal code still cannot forge a cross-edge citation;
        # authorization bypasses only ownership, never provenance shape.
        with core_db.evidence_mutation(conn):
            with pytest.raises(sqlite3.IntegrityError, match="invalid knowledge"):
                conn.execute(
                    "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,"
                    "direction,event_at,source_evidence_id) "
                    "VALUES (?, 'forged-authorized', 'claim_assertion', 1, "
                    "'2026-01-01T00:00:00.000Z', ?)",
                    (edges[1][0], evidence_id),
                )
        negative_id = int(conn.execute(
            "SELECT id FROM kg_evidence WHERE edge_id=? AND polarity=-1 "
            "AND is_current=1",
            (edges[0][0],),
        ).fetchone()[0])
        conn.execute("SAVEPOINT invalid_direct_dependency")
        try:
            with core_db.evidence_mutation(conn):
                cur = conn.execute(
                    "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,"
                    "direction,event_at,dependency_count) VALUES "
                    "(?, 'dependency-parent', 'phase3_retraction', -1, "
                    "'2026-01-03T00:00:00.000Z', 1)",
                    (edges[0][0],),
                )
            with pytest.raises(sqlite3.IntegrityError, match="internally managed"):
                conn.execute(
                    "INSERT INTO kg_lifecycle_dependencies(lifecycle_id,evidence_id) "
                    "VALUES (?,?)",
                    (int(cur.lastrowid), negative_id),
                )
        finally:
            conn.execute("ROLLBACK TO invalid_direct_dependency")
            conn.execute("RELEASE invalid_direct_dependency")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO kg_edge_lifecycle(edge_id,event_key,event_kind,"
                "direction,event_at,source_evidence_id) "
                "VALUES (?, 'bad-time', 'claim_assertion', 1, 'not-a-time', ?)",
                (edges[0][0], evidence_id),
            )
        lifecycle_id = int(conn.execute(
            "SELECT id FROM kg_edge_lifecycle WHERE edge_id = ? ORDER BY id LIMIT 1",
            (edges[0][0],),
        ).fetchone()[0])
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE kg_edge_lifecycle SET direction = -1 WHERE id = ?",
                (lifecycle_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM kg_edge_lifecycle WHERE id = ?", (lifecycle_id,))
        with core_db.evidence_mutation(conn), pytest.raises(
            sqlite3.IntegrityError, match="lifecycle history is immutable"
        ):
            conn.execute(
                "UPDATE kg_edge_lifecycle SET details=details WHERE id=?",
                (lifecycle_id,),
            )
        with core_db.evidence_mutation(conn), pytest.raises(
            sqlite3.IntegrityError, match="lifecycle history is immutable"
        ):
            conn.execute(
                "DELETE FROM kg_edge_lifecycle WHERE id=?", (lifecycle_id,)
            )

        from hymem.dreaming.bitemporal import evidence_event_at

        cause_at = evidence_event_at(conn, negative_id)
        record_lifecycle_event(
            conn,
            edge_id=int(edges[0][0]),
            event_key=evidence_ledger.phase3_retraction_event_key(
                conn, [negative_id]
            ),
            event_kind="phase3_retraction",
            direction=-1,
            event_at=cause_at,
            dependency_evidence_ids=[negative_id],
            details="confidence_or_negative_dominance",
        )
        dependency = conn.execute(
            "SELECT lifecycle_id,evidence_id FROM kg_lifecycle_dependencies "
            "WHERE evidence_id=?",
            (negative_id,),
        ).fetchone()
        with core_db.evidence_mutation(conn), pytest.raises(
            sqlite3.IntegrityError, match="dependency history is immutable"
        ):
            conn.execute(
                "UPDATE kg_lifecycle_dependencies SET evidence_id=evidence_id "
                "WHERE lifecycle_id=? AND evidence_id=?",
                tuple(dependency),
            )
        with core_db.evidence_mutation(conn), pytest.raises(
            sqlite3.IntegrityError, match="dependency history is immutable"
        ):
            conn.execute(
                "DELETE FROM kg_lifecycle_dependencies "
                "WHERE lifecycle_id=? AND evidence_id=?",
                tuple(dependency),
            )

        # A valid helper mutation succeeds and converges the cached ledger.
        assert evidence_ledger.record_signal(
            conn,
            edge_id=int(edges[0][0]),
            signal_key="host-confirmed",
            signal_kind="host_confirmation",
            polarity=1,
        )
        signal_id = int(conn.execute(
            "SELECT id FROM kg_evidence_signals WHERE signal_key='host-confirmed'"
        ).fetchone()[0])
        observation = conn.execute(
            "SELECT rowid FROM kg_claim_observations ORDER BY rowid LIMIT 1"
        ).fetchone()
        for statement, parameters in (
            ("UPDATE kg_evidence SET polarity=polarity WHERE id=?", (evidence_id,)),
            ("DELETE FROM kg_evidence WHERE id=?", (evidence_id,)),
            ("UPDATE kg_evidence_signals SET details=details WHERE id=?", (signal_id,)),
            ("DELETE FROM kg_evidence_signals WHERE id=?", (signal_id,)),
            (
                "UPDATE kg_claim_observations SET polarity=polarity WHERE rowid=?",
                (observation["rowid"],),
            ),
            ("DELETE FROM kg_claim_observations WHERE rowid=?", (observation["rowid"],)),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(statement, parameters)
        assert not evidence_ledger.count_mismatches(conn)
    finally:
        conn.close()


def test_value_supersession_dependency_reopens_when_winner_is_removed(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        old_id, new_id = _messages(
            conn,
            [
                ("user", "The target is 65 percent", "2026-01-01T00:00:00Z"),
                ("user", "The target is now 78 percent", "2026-02-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        old_chunk = _chunk(conn, "value-old", [old_id])
        new_chunk = _chunk(conn, "value-new", [new_id])
        _persist(
            conn, old_chunk,
            [Triple(
                "service", "has_attribute", "65_percent", 1,
                value_numeric=65, value_unit="percent", source_message_id=old_id,
            )],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, new_chunk,
            [Triple(
                "service", "has_attribute", "78_percent", 1,
                value_numeric=78, value_unit="percent", source_message_id=new_id,
            )],
            prompt_version="v13", cfg=cfg,
        )
        assert supersede_competing_values(conn, cfg) == 1
        old_edge = conn.execute(
            "SELECT status, invalid_at FROM knowledge_graph "
            "WHERE object_canonical = '65_percent'"
        ).fetchone()
        assert tuple(old_edge) == ("retracted", "2026-02-01T00:00:00.000Z")

        _persist(conn, new_chunk, [], prompt_version="v14", cfg=cfg)
        old_edge = conn.execute(
            "SELECT status, valid_at, invalid_at FROM knowledge_graph "
            "WHERE object_canonical = '65_percent'"
        ).fetchone()
        assert tuple(old_edge) == (
            "active", "2026-01-01T00:00:00.000Z", None,
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_value_supersession_recloses_on_winner_revision_change(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        old_id, new_id = _messages(
            conn,
            [
                ("user", "The target is 65 percent", "2026-01-01T00:00:00Z"),
                ("user", "The target is 78 percent", "2026-02-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        old_chunk = _chunk(conn, "revision-old", [old_id])
        new_chunk = _chunk(conn, "revision-new", [new_id])
        _persist(
            conn, old_chunk,
            [Triple("service", "has_attribute", "65_percent", 1,
                    temporal_scope="2025", source_message_id=old_id)],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, new_chunk,
            [Triple("service", "has_attribute", "78_percent", 1,
                    temporal_scope="2025", source_message_id=new_id)],
            prompt_version="v13", cfg=cfg,
        )
        assert supersede_competing_values(conn, cfg) == 1
        _persist(
            conn, new_chunk,
            [Triple("service", "has_attribute", "78_percent", 1,
                    temporal_scope="2026", source_message_id=new_id)],
            prompt_version="v14", cfg=cfg,
        )
        assert conn.execute(
            "SELECT status FROM knowledge_graph WHERE object_canonical='65_percent'"
        ).fetchone()[0] == "active"
        assert supersede_competing_values(conn, cfg) == 1
        assert conn.execute(
            "SELECT status FROM knowledge_graph WHERE object_canonical='65_percent'"
        ).fetchone()[0] == "retracted"
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle "
            "WHERE event_kind='value_supersession'"
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_value_supersession_uses_latest_current_assertion_not_interval_start(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        first_65, value_78, reaffirm_65 = _messages(
            conn,
            [
                ("user", "Target is 65", "2020-01-01T00:00:00Z"),
                ("user", "Target is 78", "2023-01-01T00:00:00Z"),
                ("user", "Target is again 65", "2024-01-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        for label, message_id, value in (
            ("first", first_65, "65_percent"),
            ("middle", value_78, "78_percent"),
            ("reaffirm", reaffirm_65, "65_percent"),
        ):
            _persist(
                conn, _chunk(conn, f"batch-{label}", [message_id]),
                [Triple("service", "has_attribute", value, 1,
                        source_message_id=message_id)],
                prompt_version="v13", cfg=cfg,
            )
        assert supersede_competing_values(conn, cfg) == 1
        active = conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph "
            "WHERE object_canonical='65_percent'"
        ).fetchone()
        closed = conn.execute(
            "SELECT id,status,valid_at,invalid_at FROM knowledge_graph "
            "WHERE object_canonical='78_percent'"
        ).fetchone()
        assert tuple(active) == ("active", "2020-01-01T00:00:00.000Z", None)
        assert tuple(closed[1:]) == (
            "retracted", "2023-01-01T00:00:00.000Z",
            "2024-01-01T00:00:00.000Z",
        )
        from hymem.dreaming.bitemporal import recompute_edge_interval
        before = tuple(closed[1:])
        recompute_edge_interval(conn, int(closed["id"]))
        assert tuple(conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph WHERE id=?",
            (closed["id"],),
        ).fetchone()) == before
    finally:
        conn.close()


def test_public_ingress_rejects_future_dated_assertion_atomically(
    tmp_path: Path,
):
    cfg = HyMemConfig(root=tmp_path)
    hy = HyMem(cfg)
    try:
        with pytest.raises(ValueError, match="message source valid time"):
            hy.log_message(
                "s", "user", "The service uses Redis",
                created_at="2100-01-01T00:00:00Z",
            )
        assert hy.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM message_retention_coverage"
        ).fetchone()[0] == 0
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph"
        ).fetchone()[0] == 0
    finally:
        hy.close()


def test_future_dated_legacy_lifecycle_event_is_rejected_atomically(tmp_path: Path):
    cfg = HyMemConfig(root=tmp_path)
    hy = HyMem(cfg)
    try:
        edge_id = int(hy.conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,status,valid_at) "
            "VALUES ('legacy_service','uses','legacy_db',3,'active',"
            "'2031-04-05T00:00:00.000Z')"
        ).lastrowid)
        before = tuple(hy.conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph WHERE id=?",
            (edge_id,),
        ).fetchone())
        with pytest.raises(ValueError, match="lifecycle event valid time"):
            record_lifecycle_event(
                hy.conn,
                edge_id=edge_id,
                event_key="legacy-state",
                event_kind="legacy_state",
                direction=1,
                event_at="2031-04-05T00:00:00.000Z",
            )
        assert hy.conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE edge_id=?", (edge_id,)
        ).fetchone()[0] == 0
        assert tuple(hy.conn.execute(
            "SELECT status,valid_at,invalid_at FROM knowledge_graph WHERE id=?",
            (edge_id,),
        ).fetchone()) == before
    finally:
        hy.close()


def test_observed_claim_promotes_exact_derived_edge_before_inference_rebuild(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,derived,status) "
            "VALUES ('app','uses','redis',1,1,'active')"
        )
        [message_id] = _messages(
            conn, [("user", "The app uses Redis", "2026-03-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "derived-direct", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, chunk,
            [Triple("app", "uses", "redis", 1, source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
        )
        promoted = conn.execute(
            "SELECT id, derived, pos_evidence FROM knowledge_graph "
            "WHERE subject_canonical = 'app' AND object_canonical = 'redis'"
        ).fetchone()
        assert (promoted["derived"], promoted["pos_evidence"]) == (0, 2)
        infer_transitive_edges(conn, cfg)
        assert conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE id = ? AND derived = 0",
            (promoted["id"],),
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_semantic_dedup_never_attaches_claim_to_derived_candidate(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        cur = conn.execute(
            "INSERT INTO knowledge_graph(subject_canonical,predicate,"
            "object_canonical,pos_evidence,derived,status) "
            "VALUES ('app','uses','redis_db',1,1,'active')"
        )
        derived_id = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO edge_embeddings(edge_text,vector_json,model,dim) "
            "VALUES ('app uses redis_db','[1.0,0.0]','test',2)"
        )
        [message_id] = _messages(
            conn,
            [("user", "The app uses Redis Database", "2026-03-01T00:00:00Z")],
        )
        chunk = _chunk(conn, "derived-semantic", [message_id])
        cfg = HyMemConfig(
            root=tmp_path,
            triple_dedup_enabled=True,
            triple_dedup_cosine_threshold=0.9,
        )
        _persist(
            conn, chunk,
            [Triple("app", "uses", "redis_database", 1,
                    source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
            dedup_vectors={"app uses redis_database": [1.0, 0.0]},
        )
        direct = conn.execute(
            "SELECT id FROM knowledge_graph WHERE object_canonical='redis_database' "
            "AND derived=0"
        ).fetchone()
        assert direct is not None and int(direct["id"]) != derived_id
        infer_transitive_edges(conn, cfg)
        assert conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE id=?", (derived_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE id=? AND derived=0",
            (direct["id"],),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def _equal_time_state(
    tmp_path: Path,
    source_polarities: tuple[int, int],
    processing_order: tuple[int, int],
) -> tuple[str, str | None, str | None]:
    conn = _open(tmp_path)
    try:
        ids = _messages(
            conn,
            [
                ("user", "First claim", "2026-03-01T00:00:00Z"),
                ("user", "Second claim", "2026-03-01T00:00:00+00:00"),
            ],
        )
        chunks = [_chunk(conn, f"equal-{index}", [message_id])
                  for index, message_id in enumerate(ids)]
        cfg = HyMemConfig(
            root=tmp_path,
            retract_threshold=0.6,
            zombie_neg_threshold=100,
        )
        for index in processing_order:
            _persist(
                conn, chunks[index],
                [Triple("app", "uses", "redis", source_polarities[index],
                        source_message_id=ids[index])],
                prompt_version="v13", cfg=cfg,
            )
        conn.execute(
            "UPDATE knowledge_graph SET last_reinforced=CURRENT_TIMESTAMP"
        )
        phase3.decay(conn, cfg)
        row = conn.execute(
            "SELECT status, valid_at, invalid_at FROM knowledge_graph"
        ).fetchone()
        return tuple(row)
    finally:
        conn.close()


@pytest.mark.parametrize("processing_order", [(0, 1), (1, 0)])
def test_equal_time_source_order_is_portable_and_arrival_independent(
    tmp_path: Path, processing_order: tuple[int, int]
):
    # A contradiction from the earlier source precedes the later assertion.
    assert _equal_time_state(
        tmp_path / "negative-first", (-1, 1), processing_order
    ) == ("active", "2026-03-01T00:00:00.000Z", None)
    # A contradiction from the later source closes the earlier assertion.
    assert _equal_time_state(
        tmp_path / "positive-first", (1, -1), processing_order
    ) == (
        "retracted",
        "2026-03-01T00:00:00.000Z",
        "2026-03-01T00:00:00.000Z",
    )


def test_edge_merge_moves_revision_and_lifecycle_history_before_member_delete(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        first_id, second_id = _messages(
            conn,
            [
                ("user", "App uses Redis", "2026-01-01T00:00:00Z"),
                ("user", "App uses Redis DB", "2026-02-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        first = _chunk(conn, "merge-first", [first_id])
        second = _chunk(conn, "merge-second", [second_id])
        _persist(
            conn, first,
            [Triple("app", "uses", "redis", 1, source_message_id=first_id)],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, second,
            [Triple("app", "uses", "redis_db", 1, source_message_id=second_id)],
            prompt_version="v13", cfg=cfg,
        )
        rows = conn.execute(
            "SELECT id, object_canonical FROM knowledge_graph ORDER BY id"
        ).fetchall()
        survivor = int(rows[0]["id"])
        member = int(rows[1]["id"])
        evidence_ledger.move_edge_provenance(conn, survivor, [member])
        conn.execute("DELETE FROM knowledge_graph WHERE id = ?", (member,))

        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE edge_id = ?",
            (survivor,),
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_edge_lifecycle WHERE edge_id = ?",
            (survivor,),
        ).fetchone()[0] == 2
        edge = conn.execute(
            "SELECT pos_evidence, status, valid_at, invalid_at "
            "FROM knowledge_graph WHERE id = ?",
            (survivor,),
        ).fetchone()
        assert tuple(edge) == (4, "active", "2026-01-01T00:00:00.000Z", None)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_alias_merge_dedupes_surface_variants_and_preserves_external_dependency(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        old_id, winner_id = _messages(
            conn,
            [
                ("user", "Svc target is 65 percent", "2026-01-01T00:00:00Z"),
                ("user", "Svc/service target is 78 percent", "2026-02-01T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        old_chunk = _chunk(conn, "alias-old", [old_id])
        winner_chunk = _chunk(conn, "alias-winner", [winner_id])
        _persist(
            conn, old_chunk,
            [Triple("svc", "has_attribute", "65_percent", 1,
                    source_message_id=old_id)],
            prompt_version="v13", cfg=cfg,
        )
        # One exact source expresses alias-only variants. They are independent
        # edges before canonicalization but the semantic interpretation is the
        # same and must dedupe safely when `svc` folds into `service`.
        _persist(
            conn, winner_chunk,
            [
                Triple("svc", "has_attribute", "78_percent", 1,
                       source_message_id=winner_id),
                Triple("service", "has_attribute", "78_percent", 1,
                       source_message_id=winner_id),
            ],
            prompt_version="v13", cfg=cfg,
        )
        assert supersede_competing_values(conn, cfg) == 1
        loser_id = int(conn.execute(
            "SELECT id FROM knowledge_graph WHERE subject_canonical='svc' "
            "AND object_canonical='65_percent'"
        ).fetchone()[0])
        canonicalize.merge(conn, keep="service", drop="svc")

        survivor = conn.execute(
            "SELECT id FROM knowledge_graph WHERE subject_canonical='service' "
            "AND object_canonical='78_percent'"
        ).fetchone()
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE edge_id=? AND is_current=1",
            (survivor["id"],),
        ).fetchone()[0] == 1
        lifecycle = conn.execute(
            "SELECT id, dependency_count FROM kg_edge_lifecycle "
            "WHERE edge_id=? AND event_kind='value_supersession'",
            (loser_id,),
        ).fetchone()
        assert lifecycle["dependency_count"] == 1
        dependency = conn.execute(
            "SELECT ev.edge_id FROM kg_lifecycle_dependencies dep "
            "JOIN kg_evidence ev ON ev.id=dep.evidence_id "
            "WHERE dep.lifecycle_id=?",
            (lifecycle["id"],),
        ).fetchone()
        assert int(dependency["edge_id"]) == int(survivor["id"])
        assert conn.execute(
            "SELECT status FROM knowledge_graph WHERE id=?", (loser_id,)
        ).fetchone()[0] == "retracted"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_merge_keeps_distinct_same_key_lifecycle_dependencies(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        ids = _messages(
            conn,
            [
                ("user", "Redis DB is disputed", "2026-01-01T00:00:00Z"),
                ("user", "Redis database is disputed", "2026-01-02T00:00:00Z"),
            ],
        )
        cfg = HyMemConfig(root=tmp_path)
        edge_ids: list[int] = []
        cause_ids: list[int] = []
        for index, obj in enumerate(("redis_db", "redis_database")):
            chunk = _chunk(conn, f"collision-{index}", [ids[index]])
            _persist(
                conn, chunk,
                [Triple("app", "uses", obj, -1, source_message_id=ids[index])],
                prompt_version="v13", cfg=cfg,
            )
            edge = conn.execute(
                "SELECT id FROM knowledge_graph WHERE object_canonical=?", (obj,)
            ).fetchone()
            edge_ids.append(int(edge["id"]))
            cause_ids.append(int(conn.execute(
                "SELECT id FROM kg_evidence WHERE edge_id=? AND is_current=1",
                (edge["id"],),
            ).fetchone()[0]))
            from hymem.dreaming.bitemporal import evidence_event_at

            record_lifecycle_event(
                conn, edge_id=int(edge["id"]),
                event_key=evidence_ledger.phase3_retraction_event_key(
                    conn, [cause_ids[-1]]
                ),
                event_kind="phase3_retraction", direction=-1,
                event_at=evidence_event_at(conn, cause_ids[-1]),
                dependency_evidence_ids=[cause_ids[-1]],
                details="confidence_or_negative_dominance",
            )
        canonicalize.merge(conn, keep="redis_db", drop="redis_database")
        survivor = int(conn.execute(
            "SELECT id FROM knowledge_graph WHERE object_canonical='redis_db'"
        ).fetchone()[0])
        parents = conn.execute(
            "SELECT id,dependency_count FROM kg_edge_lifecycle "
            "WHERE edge_id=? AND event_kind='phase3_retraction' ORDER BY event_key",
            (survivor,),
        ).fetchall()
        assert len(parents) == 2
        assert all(int(parent["dependency_count"]) == 1 for parent in parents)
        assert all(conn.execute(
            "SELECT COUNT(*) FROM kg_lifecycle_dependencies WHERE lifecycle_id=?",
            (parent["id"],),
        ).fetchone()[0] == 1 for parent in parents)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_merge_never_lets_retired_member_revision_overwrite_current_survivor(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "Redis alias claim", "2026-01-01T00:00:00Z")]
        )
        survivor_chunk = _chunk(conn, "history-survivor", [message_id])
        member_old = _chunk(conn, "history-member-old", [message_id])
        member_new = _chunk(conn, "history-member-new", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, survivor_chunk,
            [Triple("app", "uses", "redis_db", 1,
                    source_message_id=message_id)],
            prompt_version="v14", cfg=cfg,
        )
        _persist(
            conn, member_old,
            [Triple("app", "uses", "redis_database", -1,
                    source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, member_new,
            [Triple("app", "uses", "redis_database", 1,
                    source_message_id=message_id)],
            prompt_version="v14", cfg=cfg,
        )
        retired_negative_id = int(conn.execute(
            "SELECT ev.id FROM kg_evidence ev JOIN knowledge_graph kg "
            "ON kg.id=ev.edge_id WHERE kg.object_canonical='redis_database' "
            "AND ev.polarity=-1 AND ev.is_current=0"
        ).fetchone()[0])
        canonicalize.merge(conn, keep="redis_db", drop="redis_database")
        edge = conn.execute(
            "SELECT id,pos_evidence,neg_evidence,status,invalid_at "
            "FROM knowledge_graph WHERE object_canonical='redis_db'"
        ).fetchone()
        assert tuple(edge[1:]) == (2, 0, "active", None)
        assert conn.execute(
            "SELECT is_current,polarity,edge_id FROM kg_evidence WHERE id=?",
            (retired_negative_id,),
        ).fetchone()[:] == (0, -1, edge["id"])
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE edge_id=? AND is_current=1",
            (edge["id"],),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_merge_observation_conflict_rolls_back_whole_move_and_cleanly_retries(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "Alias target", "2026-01-01T00:00:00Z")]
        )
        survivor_chunk = _chunk(conn, "atomic-merge-survivor", [message_id])
        member_chunk = _chunk(conn, "atomic-merge-member", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, survivor_chunk,
            [Triple("app", "uses", "redis_db", 1, temporal_scope="2025",
                    source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, member_chunk,
            [Triple("app", "uses", "redis_database", 1, temporal_scope="2026",
                    source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
        )
        rows = conn.execute(
            "SELECT id,object_canonical FROM knowledge_graph ORDER BY id"
        ).fetchall()
        survivor = int(rows[0]["id"])
        member = int(rows[1]["id"])
        before_evidence = [tuple(row) for row in conn.execute(
            "SELECT id,edge_id,is_current,revision,interpretation_key "
            "FROM kg_evidence ORDER BY id"
        ).fetchall()]
        before_observations = [tuple(row) for row in conn.execute(
            "SELECT chunk_id,edge_id,evidence_id,interpretation_key "
            "FROM kg_claim_observations ORDER BY chunk_id"
        ).fetchall()]
        with pytest.raises(ValueError, match="same-generation"):
            evidence_ledger.move_edge_provenance(conn, survivor, [member])
        assert [tuple(row) for row in conn.execute(
            "SELECT id,edge_id,is_current,revision,interpretation_key "
            "FROM kg_evidence ORDER BY id"
        ).fetchall()] == before_evidence
        assert [tuple(row) for row in conn.execute(
            "SELECT chunk_id,edge_id,evidence_id,interpretation_key "
            "FROM kg_claim_observations ORDER BY chunk_id"
        ).fetchall()] == before_observations

        _persist(
            conn, member_chunk,
            [Triple("app", "uses", "redis_database", 1, temporal_scope="2025",
                    source_message_id=message_id)],
            prompt_version="v14", cfg=cfg,
        )
        evidence_ledger.move_edge_provenance(conn, survivor, [member])
        conn.execute("DELETE FROM knowledge_graph WHERE id=?", (member,))
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_evidence WHERE edge_id=? AND is_current=1",
            (survivor,),
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_manifest_nonowner_cannot_be_unpublished_and_keeps_shared_authority(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2026-01-01T00:00:00Z")]
        )
        chunks = {
            name: _chunk(conn, f"manifest-{name}", [message_id])
            for name in ("a", "b")
        }
        cfg = HyMemConfig(root=tmp_path)
        claim = Triple("app", "uses", "redis", 1, source_message_id=message_id)
        _persist(conn, chunks["a"], [claim], prompt_version="v13", cfg=cfg)
        _persist(conn, chunks["b"], [claim], prompt_version="v13", cfg=cfg)
        assert conn.execute(
            "SELECT chunk_id FROM kg_evidence WHERE is_current=1"
        ).fetchone()[0] == chunks["a"].id
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE chunks SET source_manifest_version=NULL, "
                "source_manifest_count=NULL WHERE id=?",
                (chunks["b"].id,),
            )
        _persist(conn, chunks["a"], [], prompt_version="v14", cfg=cfg)
        assert conn.execute(
            "SELECT pos_evidence,status,invalid_at FROM knowledge_graph"
        ).fetchone()[:] == (2, "active", None)
    finally:
        conn.close()


def test_retention_preserves_observation_and_evidence_chunks_but_prunes_empty(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2026-01-01T00:00:00Z")]
        )
        first = _chunk(conn, "retention-a", [message_id])
        second = _chunk(conn, "retention-b", [message_id])
        empty = _chunk(conn, "retention-empty", [message_id])
        published_empty = _chunk(conn, "retention-published-empty", [message_id])
        cfg = HyMemConfig(root=tmp_path, max_chunks=1, retention_days=1)
        claim = Triple("app", "uses", "redis", 1, source_message_id=message_id)
        _persist(conn, first, [claim], prompt_version="v13", cfg=cfg)
        _persist(conn, second, [claim], prompt_version="v13", cfg=cfg)
        _persist(conn, published_empty, [], prompt_version="v14", cfg=cfg)
        conn.execute(
            "UPDATE chunks SET created_at='2000-01-01' WHERE id IN (?,?,?,?)",
            (first.id, second.id, empty.id, published_empty.id),
        )
        assert prune_chunks(conn, cfg) == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE id IN (?,?)", (first.id, second.id)
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE id=?", (empty.id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes WHERE chunk_id=?",
            (published_empty.id,),
        ).fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM chunks WHERE id=?", (published_empty.id,))
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_outcome_helper_rejects_wrong_prompt_authority_and_unmanifested_chunk(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2026-01-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "outcome-helper", [message_id])
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, chunk,
            [Triple("app", "uses", "redis", 1,
                    source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
        )
        before = tuple(conn.execute(
            "SELECT prompt_version,prompt_generation,result_hash "
            "FROM kg_claim_extraction_outcomes WHERE chunk_id=?", (chunk.id,),
        ).fetchone())
        with pytest.raises(ValueError, match="observation authority"):
            evidence_ledger.record_claim_extraction_outcome(
                conn, chunk_id=chunk.id, prompt_version="v14"
            )
        assert tuple(conn.execute(
            "SELECT prompt_version,prompt_generation,result_hash "
            "FROM kg_claim_extraction_outcomes WHERE chunk_id=?", (chunk.id,),
        ).fetchone()) == before

        conn.execute(
            "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
            "salience_reason,text,chunk_kind) VALUES "
            "('unmanifested','s',?,?, 'test','user: App uses Redis','extraction')",
            (message_id, message_id),
        )
        with pytest.raises(ValueError, match="published source manifest"):
            evidence_ledger.record_claim_extraction_outcome(
                conn, chunk_id="unmanifested", prompt_version="v14"
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM kg_claim_extraction_outcomes "
            "WHERE chunk_id='unmanifested'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_retention_authorizes_canonical_tombstone_cascade(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "App uses Redis", "2026-01-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "tombstone", [message_id])
        cfg = HyMemConfig(root=tmp_path, tombstone_retention_days=1)
        _persist(
            conn, chunk,
            [Triple("app", "uses", "redis", 1, source_message_id=message_id)],
            prompt_version="v13", cfg=cfg,
        )
        edge_id = int(conn.execute("SELECT id FROM knowledge_graph").fetchone()[0])
        _seed_manual_event(
            conn, edge_id=edge_id, signal_key="host-delete",
            event_at="2026-02-01T00:00:00Z",
        )
        conn.execute(
            "UPDATE knowledge_graph SET last_seen='2000-01-01' WHERE id=?",
            (edge_id,),
        )
        assert prune_retracted_edges(conn, cfg) == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM knowledge_graph WHERE id=?", (edge_id,)
        ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_value_lifecycle_key_rebinds_after_subject_canonical_merge(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        first, second = _messages(conn, [
            ("user", "Svc target is 65 percent", "2024-01-01T00:00:00Z"),
            ("user", "Svc target is 78 percent", "2025-01-01T00:00:00Z"),
        ])
        cfg = HyMemConfig(root=tmp_path)
        _persist(
            conn, _chunk(conn, "value-merge-old", [first]),
            [Triple("svc", "has_attribute", "65_percent", 1,
                    source_message_id=first)],
            prompt_version="v13", cfg=cfg,
        )
        _persist(
            conn, _chunk(conn, "value-merge-new", [second]),
            [Triple("svc", "has_attribute", "78_percent", 1,
                    source_message_id=second)],
            prompt_version="v13", cfg=cfg,
        )
        assert supersede_competing_values(conn, cfg) == 1
        canonicalize.merge(conn, keep="service", drop="svc")
        event = conn.execute(
            "SELECT event_key FROM kg_edge_lifecycle "
            "WHERE event_kind='value_supersession'"
        ).fetchone()[0]
        assert event.startswith("value-supersession:service:has_attribute:")
        winner = conn.execute(
            "SELECT id FROM kg_evidence WHERE polarity=1 AND is_current=1 "
            "AND edge_id=(SELECT id FROM knowledge_graph "
            "WHERE object_canonical='78_percent')"
        ).fetchone()[0]
        loser = conn.execute(
            "SELECT id FROM knowledge_graph WHERE object_canonical='65_percent'"
        ).fetchone()[0]
        from hymem.dreaming.bitemporal import normalized_event_at

        assert event == evidence_ledger.value_supersession_event_key(
            conn, loser_edge_id=loser, winner_evidence_id=winner,
            event_at=normalized_event_at(conn, "2025-01-01T00:00:00Z"),
        )
        from hymem import portability

        portability.export_jsonl(conn, tmp_path / "value-merge.jsonl")
    finally:
        conn.close()


def test_manual_signal_and_lifecycle_rekey_together_on_edge_merge(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        ids = _messages(conn, [
            ("user", "App uses Redis DB", "2024-01-01T00:00:00Z"),
            ("user", "App uses Redis database", "2024-01-02T00:00:00Z"),
        ])
        cfg = HyMemConfig(root=tmp_path)
        edge_ids = []
        for index, obj in enumerate(("redis_db", "redis_database")):
            _persist(
                conn, _chunk(conn, f"manual-merge-{index}", [ids[index]]),
                [Triple("app", "uses", obj, 1, source_message_id=ids[index])],
                prompt_version="v13", cfg=cfg,
            )
            edge_id = int(conn.execute(
                "SELECT id FROM knowledge_graph WHERE object_canonical=?", (obj,)
            ).fetchone()[0])
            edge_ids.append(edge_id)
            evidence_ledger.record_signal(
                conn, edge_id=edge_id, signal_key="shared",
                signal_kind="manual_retraction", polarity=-1,
                details=f"operator-{index}",
            )
        canonicalize.merge(conn, keep="redis_db", drop="redis_database")
        rows = conn.execute(
            "SELECT signal_key,details FROM kg_evidence_signals "
            "WHERE signal_kind='manual_retraction' ORDER BY signal_key"
        ).fetchall()
        assert len(rows) == 2
        events = {
            row["event_key"]: row["details"]
            for row in conn.execute(
                "SELECT event_key,details FROM kg_edge_lifecycle "
                "WHERE event_kind='manual_retraction'"
            ).fetchall()
        }
        assert events == {
            evidence_ledger.manual_retraction_event_key(row["signal_key"]):
                row["details"]
            for row in rows
        }
        from hymem import portability

        wire = tmp_path / "manual-merge.jsonl"
        portability.export_jsonl(conn, wire)
        restored = core_db.connect(tmp_path / "manual-merge-restored.sqlite")
        try:
            core_db.initialize(restored)
            portability.import_jsonl(restored, wire)
            assert restored.execute(
                "SELECT COUNT(*) FROM kg_evidence_signals "
                "WHERE signal_kind='manual_retraction'"
            ).fetchone()[0] == 2
        finally:
            restored.close()
    finally:
        conn.close()


def test_manual_merge_keeps_same_payload_at_distinct_times_as_two_pairs(
    tmp_path: Path,
):
    conn = _open(tmp_path)
    try:
        ids = _messages(conn, [
            ("user", "App uses Redis DB", "2024-01-01T00:00:00Z"),
            ("user", "App uses Redis database", "2024-01-02T00:00:00Z"),
        ])
        cfg = HyMemConfig(root=tmp_path)
        edge_ids = []
        for index, obj in enumerate(("redis_db", "redis_database")):
            _persist(
                conn, _chunk(conn, f"manual-time-merge-{index}", [ids[index]]),
                [Triple("app", "uses", obj, 1, source_message_id=ids[index])],
                prompt_version="v13", cfg=cfg,
            )
            edge_ids.append(int(conn.execute(
                "SELECT id FROM knowledge_graph WHERE object_canonical=?", (obj,)
            ).fetchone()[0]))
        _seed_manual_event(
            conn, edge_id=edge_ids[0], signal_key="shared",
            event_at="2026-02-01T00:00:00Z", details="same operator action",
        )
        _seed_manual_event(
            conn, edge_id=edge_ids[1], signal_key="shared",
            event_at="2026-03-01T00:00:00Z", details="same operator action",
        )

        canonicalize.merge(conn, keep="redis_db", drop="redis_database")
        signals = conn.execute(
            "SELECT signal_key,created_at FROM kg_evidence_signals "
            "WHERE signal_kind='manual_retraction' ORDER BY created_at"
        ).fetchall()
        events = conn.execute(
            "SELECT event_key,event_at FROM kg_edge_lifecycle "
            "WHERE event_kind='manual_retraction' ORDER BY event_at"
        ).fetchall()
        assert len(signals) == len(events) == 2
        assert {
            row["event_key"] for row in events
        } == {
            evidence_ledger.manual_retraction_event_key(row["signal_key"])
            for row in signals
        }

        from hymem import portability

        wire = tmp_path / "manual-time-merge.jsonl"
        portability.export_jsonl(conn, wire)
        restored = core_db.connect(tmp_path / "manual-time-merge-restored.sqlite")
        try:
            core_db.initialize(restored)
            portability.import_jsonl(restored, wire)
            result = portability.import_jsonl(restored, wire)
            assert sum(result.values()) == 0
            assert restored.execute("PRAGMA foreign_key_check").fetchall() == []
        finally:
            restored.close()
    finally:
        conn.close()


def test_phase3_lifecycle_key_rebinds_when_merge_renumbers_cause(tmp_path: Path):
    conn = _open(tmp_path)
    try:
        [message_id] = _messages(
            conn, [("user", "Redis aliases are disputed", "2024-01-01T00:00:00Z")]
        )
        chunk = _chunk(conn, "phase-revision-merge", [message_id])
        edges = []
        for index, obj in enumerate(("redis_db", "redis_database")):
            edge_id = int(conn.execute(
                "INSERT INTO knowledge_graph(subject_canonical,predicate,object_canonical) "
                "VALUES ('app','uses',?)", (obj,),
            ).lastrowid)
            evidence_ledger.capture_unattributed_counts(
                conn, [edge_id], reason="phase merge seed"
            )
            valid_at = conn.execute(
                "SELECT valid_at FROM knowledge_graph WHERE id=?", (edge_id,)
            ).fetchone()[0]
            record_lifecycle_event(
                conn, edge_id=edge_id, event_key="legacy-state",
                event_kind="legacy_state", direction=1, event_at=valid_at,
                details="phase merge seed",
            )
            mutation = evidence_ledger.record_chunk_evidence(
                conn, edge_id=edge_id, chunk_id=chunk.id,
                evidence_kind="decay", polarity=-1, evidence_weight=1,
                weight_source="phase merge test", temporal_scope=str(index),
            )
            from hymem.dreaming.bitemporal import evidence_event_at

            cause_at = evidence_event_at(conn, mutation.evidence_id)
            record_lifecycle_event(
                conn, edge_id=edge_id,
                event_key=evidence_ledger.phase3_retraction_event_key(
                    conn, [mutation.evidence_id]
                ),
                event_kind="phase3_retraction", direction=-1,
                event_at=cause_at,
                dependency_evidence_ids=[mutation.evidence_id],
                details="confidence_or_negative_dominance",
            )
            edges.append(edge_id)

        canonicalize.merge(conn, keep="redis_db", drop="redis_database")
        causes = conn.execute(
            "SELECT lifecycle.event_key,ev.id,ev.revision "
            "FROM kg_edge_lifecycle lifecycle "
            "JOIN kg_lifecycle_dependencies dep ON dep.lifecycle_id=lifecycle.id "
            "JOIN kg_evidence ev ON ev.id=dep.evidence_id "
            "WHERE lifecycle.event_kind='phase3_retraction' ORDER BY ev.revision"
        ).fetchall()
        assert [row["revision"] for row in causes] == [1, 2]
        assert all(
            row["event_key"] == evidence_ledger.phase3_retraction_event_key(
                conn, [row["id"]]
            )
            for row in causes
        )
    finally:
        conn.close()
