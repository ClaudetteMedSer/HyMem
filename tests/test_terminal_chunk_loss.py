"""Permanent scheduling semantics for unrecoverable legacy chunk input."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming.chunks import (
    BASELINE_SALIENCE_REASON,
    Chunk,
    extract_baseline_chunks,
    extract_high_salience_chunks,
    load_pending_persisted_chunks,
    persist_chunks,
    record_unrecoverable_chunk_losses,
)
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.retention import prune_chunks
from hymem.extraction.contract import extraction_cache_key
from hymem.extraction.llm import StubLLMClient


def _insert_unmanifested_chunk(
    hy: HyMem,
    *,
    session_id: str,
    chunk_id: str,
    start: int = 1,
    end: int = 1,
    text: str = "legacy bytes whose source is gone",
) -> None:
    hy.conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (session_id,))
    hy.conn.execute(
        "INSERT INTO chunks(id,session_id,start_message_id,end_message_id,"
        "salience_reason,text,chunk_kind) VALUES (?,?,?,?,?,?,'extraction')",
        (chunk_id, session_id, start, end, "legacy", text),
    )


def test_v47_migration_recovers_live_source_before_terminalizing_true_loss(
    cfg, caplog
):
    hy = HyMem(cfg)
    path = hy.config.db_path
    try:
        live_session = "legacy-live"
        hy.open_session(live_session)
        message_id = int(hy.conn.execute(
            "INSERT INTO messages(session_id,role,content) "
            "VALUES (?, 'user', ?)",
            (live_session, "I prefer the exact recoverable source."),
        ).lastrowid)
        hy.close_session(live_session)
        _insert_unmanifested_chunk(
            hy,
            session_id=live_session,
            chunk_id="legacy-recoverable",
            start=message_id,
            end=message_id,
            text="user: I prefer the exact recoverable source.",
        )
        _insert_unmanifested_chunk(
            hy,
            session_id="legacy-pruned",
            chunk_id="legacy-unrecoverable",
        )

        for trigger in (
            "processed_chunks_terminal_loss_insert_guard",
            "processed_chunks_terminal_loss_update_guard",
            "chunk_extraction_terminal_loss_insert_guard",
            "chunk_extraction_terminal_loss_update_guard",
            "chunk_extraction_terminal_loss_manifest_guard",
        ):
            hy.conn.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
        hy.conn.execute("DROP TABLE chunk_extraction_terminal_losses")
        hy.conn.execute(
            "UPDATE schema_meta SET value='46' WHERE key='schema_version'"
        )
    finally:
        hy.close()

    conn = core_db.connect(path)
    try:
        with caplog.at_level("WARNING"):
            core_db.initialize(conn)
        assert core_db.schema_version(conn) == core_db.EXPECTED_SCHEMA_VERSION
        recovered = conn.execute(
            "SELECT source_manifest_version,source_manifest_count FROM chunks "
            "WHERE id='legacy-recoverable'"
        ).fetchone()
        assert tuple(recovered) == ("claim-source-manifest-v1", 1)
        assert conn.execute(
            "SELECT 1 FROM chunk_extraction_terminal_losses "
            "WHERE chunk_id='legacy-recoverable'"
        ).fetchone() is None

        loss = conn.execute(
            "SELECT reason FROM chunk_extraction_terminal_losses "
            "WHERE chunk_id='legacy-unrecoverable'"
        ).fetchone()
        assert loss["reason"] == "source_manifest_unrecoverable"
        assert conn.execute(
            "SELECT 1 FROM processed_chunks "
            "WHERE chunk_id='legacy-unrecoverable'"
        ).fetchone() is None
        assert "source_manifest_terminal_loss" in caplog.text
    finally:
        conn.close()


def test_terminal_loss_is_prompt_independent_visible_and_never_fake_success(cfg):
    hy = HyMem(cfg)
    try:
        _insert_unmanifested_chunk(
            hy, session_id="lost", chunk_id="lost-forever"
        )
        hy.conn.execute(
            "INSERT INTO processed_chunks(chunk_id,prompt_version) "
            "VALUES ('lost-forever','legacy-ambiguous-result')"
        )
        with core_db.transaction(hy.conn):
            assert record_unrecoverable_chunk_losses(hy.conn, "lost") == 1

        status = hy.dream_status()
        assert status["pending_chunks"] == 0
        assert status["quarantined_chunks"] == 0
        assert status["terminal_loss_chunks"] == 1
        assert status["terminal_loss_reasons"] == {
            "source_manifest_unrecoverable": 1
        }
        assert hy.conn.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_id='lost-forever'"
        ).fetchone() is None
        with core_db.transaction(hy.conn):
            assert prune_chunks(hy.conn, replace(hy.config, max_chunks=0)) == 0
        assert hy.conn.execute(
            "SELECT 1 FROM chunks WHERE id='lost-forever'"
        ).fetchone() is not None
        assert load_pending_persisted_chunks(
            hy.conn,
            "lost",
            prompt_version="brand-new-prompt",
            limit=1,
            max_attempts=0,
        ) == []
        bumped = HyMem(replace(
            hy.config, prompt_version=hy.config.prompt_version + "-next"
        ))
        try:
            bumped_status = bumped.dream_status()
            assert bumped_status["pending_chunks"] == 0
            assert bumped_status["terminal_loss_chunks"] == 1
        finally:
            bumped.close()
        with pytest.raises(
            sqlite3.IntegrityError,
            match="terminal extraction loss cannot be marked processed",
        ):
            hy.conn.execute(
                "INSERT INTO processed_chunks(chunk_id,prompt_version) "
                "VALUES ('lost-forever','brand-new-prompt')"
            )
    finally:
        hy.close()


def test_old_terminal_loss_cannot_hide_new_recoverable_chunk_at_limit_one(cfg):
    llm = StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [], "summary": "tail complete", "procedures": [],
            }),
        },
        default=json.dumps({
            "triples": [], "markers": [], "complete": True,
        }),
    )
    hy = HyMem(replace(
        cfg,
        dream_budget=1,
        dream_baseline_budget=1,
        salience_min_chars=1,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
    ), llm=llm)
    try:
        _insert_unmanifested_chunk(
            hy, session_id="a-old-loss", chunk_id="a-old-lost"
        )
        with core_db.transaction(hy.conn):
            record_unrecoverable_chunk_losses(hy.conn, "a-old-loss")

        hy.open_session("z-new-live")
        message_id = hy.log_message(
            "z-new-live", "user", "This new source must reach extraction."
        )
        hy.close_session("z-new-live")
        report = hy.dream()

        chunk = hy.conn.execute(
            "SELECT id FROM chunks WHERE session_id='z-new-live' "
            "AND chunk_kind='extraction'"
        ).fetchone()
        assert chunk is not None
        assert hy.conn.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_id=? "
            "AND prompt_version=?",
            (chunk["id"], extraction_cache_key(hy.config.prompt_version)),
        ).fetchone()
        # The one actionable chunk exactly consumed the one-item allowance;
        # the older terminal source-loss row is not schedulable pending work.
        assert report.budget_exhausted is False
        assert hy.conn.execute(
            "SELECT digested_prompt_version FROM sessions "
            "WHERE id='z-new-live'"
        ).fetchone()[0] == hy.config.prompt_version
        assert message_id > 0
    finally:
        hy.close()


def test_targeted_exact_drain_ignores_unrelated_pending_session(cfg):
    llm = StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [],
                "summary": "Targeted session indexing completed.",
                "procedures": [],
            }),
        },
        default=json.dumps({
            "triples": [], "markers": [], "complete": True,
        }),
    )
    hy = HyMem(replace(
        cfg,
        dream_budget=1,
        dream_baseline_budget=1,
        salience_min_chars=1,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
    ), llm=llm)
    try:
        hy.open_session("outside-scope")
        outside_message = hy.log_message(
            "outside-scope", "user", "Unrelated actionable memory remains."
        )
        hy.close_session("outside-scope")
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "outside-scope")
            persist_chunks(hy.conn, [Chunk(
                id="outside-scope-pending",
                session_id="outside-scope",
                start_message_id=outside_message,
                end_message_id=outside_message,
                salience_reason="persisted_backlog",
                text="user: Unrelated actionable memory remains.",
                source_message_ids=(outside_message,),
            )])

        hy.open_session("target-scope")
        hy.log_message(
            "target-scope", "user", "The only targeted memory drains exactly."
        )
        hy.close_session("target-scope")

        report = hy.dream(session_ids=["target-scope"])

        assert report.budget_exhausted is False
        assert hy.dream_status()["pending_chunks"] == 1
        assert hy.conn.execute(
            "SELECT 1 FROM processed_chunks "
            "WHERE chunk_id='outside-scope-pending'"
        ).fetchone() is None
    finally:
        hy.close()


@pytest.mark.parametrize(
    ("stored_reason", "current_min_chars", "expected_exhausted"),
    [
        ("long_user_turn", 10_000, False),
        (BASELINE_SALIENCE_REASON, 1, True),
    ],
)
def test_budget_completion_refreshes_and_uses_current_tier(
    cfg, stored_reason, current_min_chars, expected_exhausted,
):
    llm = StubLLMClient(
        fixtures={
            "Return the JSON object now": json.dumps({
                "episodes": [],
                "summary": "Current scheduling tier was checked.",
                "procedures": [],
            }),
        },
        default=json.dumps({
            "triples": [], "markers": [], "complete": True,
        }),
    )
    hy = HyMem(replace(
        cfg,
        dream_budget=0,
        dream_baseline_budget=0,
        salience_min_chars=current_min_chars,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
    ), llm=llm)
    try:
        session_id = "tier-changed"
        hy.open_session(session_id)
        hy.log_message(session_id, "user", "Plain factual sentence.")
        hy.close_session(session_id)
        with core_db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, session_id)
            if current_min_chars > 1:
                candidate = extract_baseline_chunks(
                    hy.conn,
                    session_id,
                    prompt_version=hy.config.prompt_version,
                    limit=None,
                    min_chars=current_min_chars,
                )[0]
            else:
                candidate = extract_high_salience_chunks(
                    hy.conn, session_id, min_chars=current_min_chars
                )[0]
            persist_chunks(
                hy.conn, [replace(candidate, salience_reason=stored_reason)]
            )

        report = hy.dream()

        assert report.chunk_extraction_completion_calls == 0
        assert report.budget_exhausted is expected_exhausted
        assert hy.dream_status()["pending_chunks"] == 1
        assert hy.conn.execute(
            "SELECT salience_reason FROM chunks WHERE id=?", (candidate.id,)
        ).fetchone()[0] == candidate.salience_reason
    finally:
        hy.close()


def test_chunk_budget_exhaustion_does_not_abort_later_session_tail_work(cfg):
    digest_payload = json.dumps({
        "episodes": [], "summary": "tail complete", "procedures": [],
    })
    llm = StubLLMClient(
        fixtures={"Return the JSON object now": digest_payload},
        default=json.dumps({
            "triples": [], "markers": [], "complete": True,
        }),
    )
    hy = HyMem(replace(
        cfg,
        dream_budget=1,
        dream_baseline_budget=1,
        salience_min_chars=1,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
    ), llm=llm)
    try:
        for session_id in ("a-old", "z-new"):
            hy.open_session(session_id)
            hy.log_message(session_id, "user", f"substantive {session_id} input")
            hy.close_session(session_id)

        report = hy.dream()
        states = hy.conn.execute(
            "SELECT id,digested_prompt_version FROM sessions "
            "WHERE id IN ('a-old','z-new') ORDER BY id"
        ).fetchall()
        assert [tuple(row) for row in states] == [
            ("a-old", hy.config.prompt_version),
            ("z-new", hy.config.prompt_version),
        ]
        digest_calls = [
            call for call in llm.calls if "Return the JSON object now" in call.user
        ]
        assert len(digest_calls) == 2
        assert report.sessions_processed == 2
        assert report.budget_exhausted is True
    finally:
        hy.close()


def test_terminal_loss_roundtrips_without_becoming_processed(cfg, tmp_path):
    source = HyMem(cfg)
    export_path = tmp_path / "terminal-loss.jsonl"
    try:
        _insert_unmanifested_chunk(
            source, session_id="portable-loss", chunk_id="portable-lost"
        )
        with core_db.transaction(source.conn):
            record_unrecoverable_chunk_losses(source.conn, "portable-loss")
        counts = source.export(export_path)
        assert counts["chunk_extraction_terminal_loss"] == 1
    finally:
        source.close()

    target = HyMem(replace(cfg, root=tmp_path / "target"))
    try:
        imported = target.import_(export_path)
        assert imported["chunk_extraction_terminal_loss"] == 1
        assert target.conn.execute(
            "SELECT reason FROM chunk_extraction_terminal_losses "
            "WHERE chunk_id='portable-lost'"
        ).fetchone()[0] == "source_manifest_unrecoverable"
        assert target.conn.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_id='portable-lost'"
        ).fetchone() is None
        assert target.import_(export_path)["chunk_extraction_terminal_loss"] == 0
    finally:
        target.close()
