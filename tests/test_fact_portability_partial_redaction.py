"""Offset-stable redaction regressions for portable authoritative facts."""

from __future__ import annotations

import dataclasses
import json

from hymem import HyMem, HyMemConfig
from hymem.core import db as core_db
from hymem.dreaming import facts
from hymem.dreaming.lossless import validate_message_coverage_artifact
from hymem.extraction.llm import StubLLMClient
from hymem import redaction


def _config(root, *, redact_secrets: bool) -> HyMemConfig:
    return dataclasses.replace(
        HyMemConfig(root=root, redact_secrets=redact_secrets),
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        dream_digest_max_chars=256,
    )


def test_partial_fact_source_redaction_preserves_offsets_and_benign_entities(
    tmp_path,
):
    email = "secret.person@example.com"
    normalized_email = "secret_person_example_com"
    pem_body = "super-private-body-material"
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        f"{pem_body}\n"
        "-----END PRIVATE KEY-----"
    )
    content = (
        f"BENIGN-PREFIX medflow; contact {email}. {pem} "
        "BENIGN-SUFFIX remains useful. " + ("ordinary tail words " * 30)
    )
    expected_safe_content = redaction.redact_preserving_length(content)
    assert len(expected_safe_content) == len(content)
    assert "BENIGN-PREFIX medflow" in expected_safe_content
    assert "BENIGN-SUFFIX remains useful" in expected_safe_content
    assert email not in expected_safe_content
    assert pem_body not in expected_safe_content

    src = HyMem(_config(tmp_path / "partial-src", redact_secrets=False))
    wire = tmp_path / "partial-redaction.jsonl"
    try:
        message_id = src.log_message(
            "partial-redaction", "user", content,
            created_at="2025-01-02T03:04:05.000Z",
        )
        extraction = facts.extract_facts(
            src.conn,
            "partial-redaction",
            StubLLMClient(default=json.dumps([{
                "text": f"Medflow support is available at {email}.",
                "entities": ["medflow", normalized_email],
            }])),
            src.config,
        )
        assert extraction is not None
        assert extraction.partial_message_id == message_id
        assert 0 < extraction.next_message_offset < len(content)
        with core_db.transaction(src.conn):
            facts.persist_facts(src.conn, "partial-redaction", extraction)
        first_offset = extraction.next_message_offset
        cursor = src.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset FROM sessions WHERE id='partial-redaction'"
        ).fetchone()
        continuation = facts.extract_facts(
            src.conn,
            "partial-redaction",
            StubLLMClient(default="[]"),
            src.config,
            since_message_id=cursor["facts_cursor_message_id"],
            partial_message_id=cursor["facts_cursor_partial_message_id"],
            start_offset=cursor["facts_cursor_offset"],
        )
        assert continuation is not None
        assert continuation.cursor_before_partial_message_id == message_id
        assert continuation.cursor_before_offset == first_offset
        assert continuation.partial_message_id == message_id
        assert continuation.next_message_offset > first_offset
        with core_db.transaction(src.conn):
            facts.persist_facts(src.conn, "partial-redaction", continuation)
        assert src.conn.execute(
            "SELECT COUNT(*) FROM fact_extraction_outcomes "
            "WHERE session_id='partial-redaction'"
        ).fetchone()[0] == 2
        source_cursor = tuple(src.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset FROM sessions WHERE id='partial-redaction'"
        ).fetchone())
        src.export(wire)
    finally:
        src.close()

    dst = HyMem(_config(tmp_path / "partial-dst", redact_secrets=True))
    try:
        dst.import_(wire)
        target_cursor = tuple(dst.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset FROM sessions WHERE id='partial-redaction'"
        ).fetchone())
        assert target_cursor == source_cursor
        offsets = [tuple(row) for row in dst.conn.execute(
            "SELECT cursor_before_offset,cursor_after_offset "
            "FROM fact_extraction_outcomes WHERE session_id='partial-redaction' "
            "ORDER BY cursor_before_offset"
        )]
        assert offsets == [(0, first_offset), (first_offset, source_cursor[2])]

        proof_row = dst.conn.execute(
            "SELECT message_id,chunk_id,coverage_version "
            "FROM message_retention_coverage WHERE message_id=? "
            "ORDER BY coverage_version LIMIT 1",
            (message_id,),
        ).fetchone()
        artifact = validate_message_coverage_artifact(
            dst.conn,
            message_id=proof_row["message_id"],
            chunk_id=proof_row["chunk_id"],
            coverage_version=proof_row["coverage_version"],
        )
        assert artifact.content == expected_safe_content

        fact = dst.conn.execute(
            "SELECT text,entities FROM narrative_facts"
        ).fetchone()
        entities = json.loads(fact["entities"])
        assert "medflow" in entities
        assert normalized_email not in entities
        assert email not in fact["text"]
        assert any(value.startswith("redacted_entity_") for value in entities)
        assert facts.fact_session_authority_is_valid(
            dst.conn, "partial-redaction"
        )

        # A redacted snapshot is a fixed point: re-exporting and importing it
        # cannot re-mask bytes, move offsets, or mint another fact identity.
        redacted_wire = tmp_path / "partial-redacted-again.jsonl"
        before = "\n".join(dst.conn.iterdump())
        dst.export(redacted_wire)
        assert sum(dst.import_(redacted_wire).values()) == 0
        assert "\n".join(dst.conn.iterdump()) == before
    finally:
        db_path = dst.config.root / "hymem.sqlite"
        dst.close()

    raw = db_path.read_bytes()
    wal_path = db_path.with_name(db_path.name + "-wal")
    if wal_path.exists():
        raw += wal_path.read_bytes()
    assert email.encode() not in raw
    assert normalized_email.encode() not in raw
    assert pem_body.encode() not in raw
