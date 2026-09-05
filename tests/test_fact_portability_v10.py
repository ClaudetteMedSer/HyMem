"""Portable v10 authority, merge, corruption, and privacy regressions."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sqlite3

import pytest

from hymem import HyMem, HyMemConfig, portability
from hymem.core import db as core_db
from hymem.dreaming import facts
from hymem.dreaming.message_coverage import record_message_coverage
from hymem.extraction.llm import StubLLMClient
from hymem.session import register_session_peer


def _config(root, *, redact: bool = False, **changes) -> HyMemConfig:
    return dataclasses.replace(
        HyMemConfig(root=root, redact_secrets=redact),
        aggregation_nodes_enabled=False,
        profile_extraction_enabled=False,
        **changes,
    )


def _publish(
    hy: HyMem, session_id: str, items: list[dict],
    *, text: str = "The service uses a quasar cache.",
) -> str:
    if hy.conn.execute(
        "SELECT 1 FROM sessions WHERE id=?", (session_id,)
    ).fetchone() is None:
        hy.log_message(
            session_id, "user", text,
            created_at="2025-01-02T03:04:05.000Z",
        )
        hy.conn.execute(
            "UPDATE sessions SET started_at='2025-01-02T03:04:00.000Z' "
            "WHERE id=?", (session_id,),
        )
    cursor = hy.conn.execute(
        "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
        "facts_cursor_offset FROM sessions WHERE id=?", (session_id,),
    ).fetchone()
    extraction = facts.extract_facts(
        hy.conn, session_id, StubLLMClient(default=json.dumps(items)), hy.config,
        since_message_id=cursor["facts_cursor_message_id"],
        partial_message_id=cursor["facts_cursor_partial_message_id"],
        start_offset=int(cursor["facts_cursor_offset"] or 0),
    )
    assert extraction is not None
    with core_db.transaction(hy.conn):
        facts.persist_facts(
            hy.conn, session_id, extraction,
            max_items=hy.config.dream_max_facts_per_session,
        )
    assert extraction.slice_key is not None
    return extraction.slice_key


def _replay(hy: HyMem, slice_key: str, items: list[dict]) -> None:
    extraction = facts.reextract_fact_outcome(
        hy.conn, slice_key,
        StubLLMClient(default=json.dumps(items)), hy.config,
    )
    with core_db.transaction(hy.conn):
        facts.persist_facts(
            hy.conn, "portable-facts", extraction,
            max_items=hy.config.dream_max_facts_per_session,
        )


def _fact_state(hy: HyMem) -> tuple[list[tuple], ...]:
    return (
        [tuple(row) for row in hy.conn.execute(
            "SELECT slice_key,session_id,prompt_version,input_hash,generation,"
            "outcome_status,result_hash,source_manifest_count,"
            "source_manifest_hash,source_manifest_complete,succeeded_at "
            "FROM fact_extraction_outcomes ORDER BY slice_key"
        )],
        [tuple(row) for row in hy.conn.execute(
            "SELECT slice_key,generation,prompt_version,outcome_status,"
            "result_hash,succeeded_at FROM fact_extraction_revisions "
            "ORDER BY slice_key,generation"
        )],
        [tuple(row) for row in hy.conn.execute(
            "SELECT source_outcome_key,fact_key,text,fact_date,entities,"
            "valid_at,invalid_at,current_generation,lifecycle_status,created_at "
            "FROM narrative_facts ORDER BY source_outcome_key,fact_key"
        )],
        [tuple(row) for row in hy.conn.execute(
            "SELECT f.source_outcome_key,f.fact_key,l.generation,l.direction,"
            "l.event_at,l.prompt_version,l.result_hash,l.recorded_at "
            "FROM narrative_fact_lifecycle l JOIN narrative_facts f "
            "ON f.id=l.fact_id ORDER BY f.source_outcome_key,f.fact_key,l.generation"
        )],
    )


def _rewrite(path, mutate) -> None:
    objects = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    body, end = objects[:-1], objects[-1]
    mutate(body)
    encoded = [json.dumps(row, ensure_ascii=False) + "\n" for row in body]
    end["counts"] = {
        kind: sum(row.get("type") == kind for row in body)
        for kind in end["counts"]
    }
    end["sha256"] = hashlib.sha256("".join(encoded).encode()).hexdigest()
    path.write_text(
        "".join(encoded) + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _downgrade_wire_to_v9(path) -> None:
    """Project a generated v10 snapshot onto the exact frozen v9 wire."""

    objects = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[dict] = []
    for envelope in objects[:-1]:
        kind = envelope["type"]
        if kind == "_meta":
            envelope = dict(envelope)
            envelope["version"] = 9
            records.append(envelope)
            continue
        if kind not in portability._V9_TABLE_BY_KIND:
            continue
        record = envelope["record"]
        records.append({
            "type": kind,
            "record": {
                column: record[column]
                for column in portability._V9_COLS_BY_KIND[kind]
            },
        })
    encoded = [
        json.dumps(envelope, ensure_ascii=False) + "\n" for envelope in records
    ]
    counts = {
        kind: sum(envelope.get("type") == kind for envelope in records)
        for kind in portability._V9_TABLE_BY_KIND
    }
    end = {
        "type": "_end",
        "counts": counts,
        "sha256": hashlib.sha256("".join(encoded).encode()).hexdigest(),
    }
    path.write_text(
        "".join(encoded) + json.dumps(end, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_v10_fact_history_roundtrips_exactly_and_idempotently(tmp_path):
    src = HyMem(_config(tmp_path / "src"))
    try:
        slice_key = _publish(src, "portable-facts", [{
            "text": "The service uses a quasar cache.",
            "date": "2025-01-02", "entities": ["quasar"],
        }])
        _replay(src, slice_key, [])
        _replay(src, slice_key, [{
            "text": "The service uses the corrected quasar cache.",
            "date": "2025-01-03", "entities": ["quasar cache"],
        }])
        expected = _fact_state(src)
        out = tmp_path / "facts-v10.jsonl"
        counts = src.export(out)
        assert counts["fact_extraction_outcome"] == 1
        assert counts["fact_extraction_revision"] == 3
        assert counts["narrative_fact"] == 2
        assert counts["narrative_fact_lifecycle"] == 3
    finally:
        src.close()

    dst = HyMem(_config(tmp_path / "dst"))
    try:
        dst.import_(out)
        assert _fact_state(dst) == expected
        row = dst.conn.execute(
            "SELECT slice_key FROM fact_extraction_outcomes"
        ).fetchone()
        assert facts.load_fact_outcome_source_manifest(
            dst.conn, row["slice_key"], verify_result=True
        ) is not None
        assert sum(dst.import_(out).values()) == 0
        assert _fact_state(dst) == expected
    finally:
        dst.close()


def test_v10_zero_outcome_retry_and_caught_up_markers_roundtrip(tmp_path):
    src = HyMem(_config(tmp_path / "zero-src"))
    try:
        src.open_session("held")
        retry = facts.facts_retry_policy_version(src.config)
        src.conn.execute(
            "UPDATE sessions SET facts_retry_count=2,"
            "facts_retry_config_version=?,facts_quarantined=0 WHERE id='held'",
            (retry,),
        )
        src.open_session("caught")
        generation = facts.facts_config_version(src.config)
        src.conn.execute(
            "UPDATE sessions SET facts_cursor_prompt_version=? WHERE id='caught'",
            (generation,),
        )
        out = tmp_path / "zero.jsonl"
        src.export(out)
    finally:
        src.close()
    dst = HyMem(_config(tmp_path / "zero-dst"))
    try:
        dst.import_(out)
        held = dst.conn.execute(
            "SELECT facts_retry_count,facts_retry_config_version,"
            "facts_quarantined FROM sessions WHERE id='held'"
        ).fetchone()
        assert tuple(held) == (2, retry, 0)
        caught = dst.conn.execute(
            "SELECT facts_cursor_prompt_version FROM sessions WHERE id='caught'"
        ).fetchone()[0]
        assert caught == generation
    finally:
        dst.close()


def test_v10_prefix_merge_is_monotonic_and_divergence_is_atomic(tmp_path):
    src = HyMem(_config(tmp_path / "prefix-src"))
    try:
        slice_key = _publish(src, "portable-facts", [{"text": "Version one."}])
        older = tmp_path / "older.jsonl"
        src.export(older)
        _replay(src, slice_key, [{"text": "Version two."}])
        newer = tmp_path / "newer.jsonl"
        src.export(newer)
        newer_state = _fact_state(src)
    finally:
        src.close()

    old_target = HyMem(_config(tmp_path / "old-target"))
    try:
        old_target.import_(older)
        old_target.import_(newer)
        assert _fact_state(old_target) == newer_state
        assert sum(old_target.import_(newer).values()) == 0
    finally:
        old_target.close()
    new_target = HyMem(_config(tmp_path / "new-target"))
    try:
        new_target.import_(newer)
        before = _fact_state(new_target)
        assert sum(new_target.import_(older).values()) == 0
        assert _fact_state(new_target) == before
    finally:
        new_target.close()

    # Branch the same immutable generation-1 snapshot and publish two distinct
    # generation-2 histories. They are incomparable, not last-writer-wins.
    branches = []
    for name, text in (("a", "Branch A."), ("b", "Branch B.")):
        branch = HyMem(_config(tmp_path / f"branch-{name}"))
        branch.import_(older)
        _replay(branch, slice_key, [{"text": text}])
        path = tmp_path / f"branch-{name}.jsonl"
        branch.export(path)
        branch.close()
        branches.append(path)
    target = HyMem(_config(tmp_path / "divergent-target"))
    try:
        target.import_(branches[0])
        before = _fact_state(target)
        with pytest.raises(ValueError, match="diverg|collid|prefix"):
            target.import_(branches[1])
        assert _fact_state(target) == before
    finally:
        target.close()


def test_v10_equal_history_preflight_holds_write_lock_and_target_controls_win(
    tmp_path, monkeypatch,
):
    """Relation and operational state are read under the same write lock.

    The second connection is synchronized inside fact preflight. It cannot
    advance the target between relation selection and import, and an equal
    donor never overwrites target-local retry state.
    """

    source = HyMem(_config(tmp_path / "locked-source"))
    wire = tmp_path / "equal-under-lock.jsonl"
    try:
        slice_key = _publish(
            source, "portable-facts", [{"text": "Locked history."}]
        )
        source.export(wire)
    finally:
        source.close()

    target = HyMem(_config(tmp_path / "locked-target"))
    racer = None
    try:
        target.import_(wire)
        retry_version = facts.facts_retry_policy_version(
            target.config, replay_slice_key=slice_key
        )
        target.conn.execute(
            "UPDATE sessions SET facts_retry_count=2,"
            "facts_retry_config_version=?,facts_quarantined=0 "
            "WHERE id='portable-facts'",
            (retry_version,),
        )
        db_path = target.config.root / "hymem.sqlite"
        racer = core_db.connect(db_path)
        racer.execute("PRAGMA busy_timeout=1")
        original = portability._preflight_v10_fact_target_collisions
        observed = {"locked": False}

        def barrier(conn, grouped):
            assert conn.in_transaction
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                racer.execute(
                    "UPDATE sessions SET facts_retry_count=3 "
                    "WHERE id='portable-facts'"
                )
            observed["locked"] = True
            return original(conn, grouped)

        monkeypatch.setattr(
            portability, "_preflight_v10_fact_target_collisions", barrier
        )
        assert sum(target.import_(wire).values()) == 0
        assert observed["locked"] is True
        controls = target.conn.execute(
            "SELECT facts_retry_count,facts_retry_config_version,"
            "facts_quarantined FROM sessions WHERE id='portable-facts'"
        ).fetchone()
        assert tuple(controls) == (2, retry_version, 0)
        assert facts.fact_session_authority_is_valid(
            target.conn, "portable-facts"
        )
    finally:
        if racer is not None:
            racer.close()
        target.close()


def test_v10_older_donor_rejects_corrupt_target_only_suffix_atomically(tmp_path):
    source = HyMem(_config(tmp_path / "suffix-source"))
    older = tmp_path / "suffix-older.jsonl"
    newer = tmp_path / "suffix-newer.jsonl"
    try:
        first_key = _publish(
            source, "portable-facts", [{"text": "First committed unit."}],
            text="First committed source.",
        )
        source.export(older)
        source.log_message(
            "portable-facts", "user", "Second committed source.",
            created_at="2025-01-02T03:05:05.000Z",
        )
        second_key = _publish(
            source, "portable-facts", [{"text": "Second committed unit."}]
        )
        assert second_key != first_key
        source.export(newer)
    finally:
        source.close()

    target = HyMem(_config(tmp_path / "suffix-target"))
    try:
        target.import_(newer)
        with core_db.evidence_destructive_mutation(target.conn):
            target.conn.execute(
                "UPDATE fact_extraction_outcomes SET source_manifest_hash=? "
                "WHERE slice_key=?",
                ("sha256:" + "0" * 64, second_key),
            )
        before = "\n".join(target.conn.iterdump())
        with pytest.raises(ValueError, match="corrupt target"):
            target.import_(older)
        assert "\n".join(target.conn.iterdump()) == before
        assert target.conn.execute(
            "SELECT source_manifest_hash FROM fact_extraction_outcomes "
            "WHERE slice_key=?", (second_key,),
        ).fetchone()[0] == "sha256:" + "0" * 64
    finally:
        target.close()


@pytest.mark.parametrize("mutation", ["bool", "huge", "orphan", "clock", "causal"])
def test_v10_fact_wire_corruption_rejects_before_mutation(tmp_path, mutation):
    src = HyMem(_config(tmp_path / "corrupt-src"))
    try:
        _publish(src, "portable-facts", [{"text": "Authority matters."}])
        original = tmp_path / "original.jsonl"
        src.export(original)
    finally:
        src.close()
    poisoned = tmp_path / f"{mutation}.jsonl"
    shutil.copyfile(original, poisoned)

    def mutate(body):
        outcome = next(row["record"] for row in body if row["type"] == "fact_extraction_outcome")
        if mutation == "bool":
            outcome["source_manifest_complete"] = True
        elif mutation == "huge":
            outcome["generation"] = 10 ** 1000
        elif mutation == "orphan":
            session = next(row["record"] for row in body if row["type"] == "session")
            session["facts_message_id"] = None
            session["facts_cursor_message_id"] = None
            session["facts_cursor_partial_message_id"] = None
            session["facts_cursor_offset"] = 0
            session["facts_cursor_prompt_version"] = None
        elif mutation == "clock":
            noncanonical = "2026-01-01T01:00:00+01:00"
            outcome["succeeded_at"] = noncanonical
            for row in body:
                if row["type"] == "fact_extraction_revision":
                    row["record"]["succeeded_at"] = noncanonical
                elif row["type"] == "narrative_fact":
                    row["record"]["created_at"] = noncanonical
                elif row["type"] == "narrative_fact_lifecycle":
                    row["record"]["recorded_at"] = noncanonical
        else:
            proof = next(row["record"] for row in body if row["type"] == "message_retention_coverage")
            proof["created_at"] = "2099-01-01T00:00:00.000Z"

    _rewrite(poisoned, mutate)
    dst = HyMem(_config(tmp_path / f"corrupt-dst-{mutation}"))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            dst.import_(poisoned)
        assert dst.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert dst.conn.execute(
            "SELECT COUNT(*) FROM fact_extraction_outcomes"
        ).fetchone()[0] == 0
    finally:
        dst.close()


def test_v10_late_import_failure_rolls_back_everything(tmp_path, monkeypatch):
    src = HyMem(_config(tmp_path / "rollback-src"))
    try:
        _publish(src, "portable-facts", [{"text": "Incoming rollback mutation."}])
        out = tmp_path / "rollback.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(_config(tmp_path / "rollback-dst"))
    try:
        target_key = _publish(
            dst, "target-existing", [{"text": "Existing rollback sentinel."}],
            text="Existing rollback sentinel source.",
        )
        retry_version = facts.facts_retry_policy_version(
            dst.config, replay_slice_key=target_key
        )
        dst.conn.execute(
            "UPDATE sessions SET facts_retry_count=2,"
            "facts_retry_config_version=?,facts_quarantined=0 "
            "WHERE id='target-existing'",
            (retry_version,),
        )
        # Retire a high id while leaving no projection row behind. A failed
        # import that rolls back facts but not sqlite_sequence could silently
        # reuse or skip identities on the next publication.
        dst.conn.execute(
            "INSERT INTO narrative_facts("
            "id,session_id,start_message_id,end_message_id,text,prompt_version) "
            "VALUES (1000,'target-existing',1000,1000,'retired high water',"
            "'facts.v2')"
        )
        dst.conn.execute("DELETE FROM narrative_facts WHERE id=1000")

        def snapshot() -> dict[str, object]:
            return {
                "sessions": [tuple(row) for row in dst.conn.execute(
                    "SELECT * FROM sessions ORDER BY id"
                )],
                "facts": _fact_state(dst),
                "fact_sources": [tuple(row) for row in dst.conn.execute(
                    "SELECT * FROM fact_extraction_source_occurrences "
                    "ORDER BY slice_key,ordinal"
                )],
                "chunks": [tuple(row) for row in dst.conn.execute(
                    "SELECT * FROM chunks ORDER BY id"
                )],
                "coverage": [tuple(row) for row in dst.conn.execute(
                    "SELECT * FROM message_retention_coverage "
                    "ORDER BY message_id,chunk_id,coverage_version"
                )],
                "sequence": dst.conn.execute(
                    "SELECT seq FROM sqlite_sequence "
                    "WHERE name='narrative_facts'"
                ).fetchone()[0],
                "fts": [tuple(row) for row in dst.conn.execute(
                    "SELECT rowid,text FROM narrative_facts_fts "
                    "WHERE narrative_facts_fts MATCH 'rollback' ORDER BY rowid"
                )],
                "foreign_keys": [tuple(row) for row in dst.conn.execute(
                    "PRAGMA foreign_key_check"
                )],
            }

        before = snapshot()
        assert before["sequence"] == 1000
        assert len(before["fts"]) == 1
        assert "Existing rollback sentinel." in before["fts"][0]
        assert before["foreign_keys"] == []
        original = portability._import_v10_fact_state
        reached_late_state = {"value": False}

        def fail_late(conn, grouped, inserted, *, session_relations):
            original(
                conn, grouped, inserted, session_relations=session_relations
            )
            reached_late_state["value"] = True
            assert conn.execute(
                "SELECT COUNT(*) FROM fact_extraction_outcomes"
            ).fetchone()[0] == 2
            assert conn.execute(
                "SELECT rowid FROM narrative_facts_fts "
                "WHERE narrative_facts_fts MATCH 'incoming'"
            ).fetchone() is not None
            assert conn.execute(
                "SELECT seq FROM sqlite_sequence "
                "WHERE name='narrative_facts'"
            ).fetchone()[0] > 1000
            raise RuntimeError("injected late fact import failure")

        monkeypatch.setattr(portability, "_import_v10_fact_state", fail_late)
        with pytest.raises(RuntimeError, match="injected late"):
            dst.import_(out)
        assert reached_late_state["value"] is True
        assert snapshot() == before
        assert dst.conn.execute(
            "SELECT rowid FROM narrative_facts_fts "
            "WHERE narrative_facts_fts MATCH 'incoming'"
        ).fetchall() == []
        # rank=1 asks FTS5 to compare its index against the external-content
        # table as well as checking the index's internal structure.
        dst.conn.execute(
            "INSERT INTO narrative_facts_fts(narrative_facts_fts,rank) "
            "VALUES ('integrity-check',1)"
        )
        controls = dst.conn.execute(
            "SELECT facts_cursor_message_id,facts_cursor_partial_message_id,"
            "facts_cursor_offset,facts_cursor_prompt_version,facts_retry_count,"
            "facts_retry_config_version,facts_quarantined "
            "FROM sessions WHERE id='target-existing'"
        ).fetchone()
        assert tuple(controls)[-3:] == (2, retry_version, 0)
        assert facts.fact_session_authority_is_valid(
            dst.conn, "target-existing"
        )
        assert dst.conn.execute(
            "SELECT 1 FROM sessions WHERE id='portable-facts'"
        ).fetchone() is None
    finally:
        dst.close()


def test_v9_missing_middle_coverage_rejects_existing_v10_chain_atomically(
    tmp_path,
):
    """An old donor cannot retroactively make a proved v10 interval lossy."""

    session_id = "portable-facts"
    base = HyMem(_config(tmp_path / "middle-base"))
    try:
        for minute, content in enumerate((
            "First source turn.",
            "Previously absent middle turn.",
            "Third source turn.",
        ), start=4):
            base.log_message(
                session_id, "user", content,
                created_at=f"2025-01-02T03:{minute:02d}:05.000Z",
            )
        base.conn.execute(
            "UPDATE sessions SET started_at='2025-01-02T03:04:00.000Z' "
            "WHERE id=?", (session_id,),
        )
        full_wire = tmp_path / "missing-middle-full.jsonl"
        base.export(full_wire)
    finally:
        base.close()

    sparse_wire = tmp_path / "missing-middle-sparse-v10.jsonl"
    donor_wire = tmp_path / "missing-middle-v9.jsonl"
    shutil.copyfile(full_wire, sparse_wire)
    shutil.copyfile(full_wire, donor_wire)

    def remove_middle(body):
        middle_proof = next(
            envelope["record"] for envelope in body
            if envelope["type"] == "message_retention_coverage"
            and envelope["record"]["message_id"] == 2
        )
        chunk_id = middle_proof["chunk_id"]
        body[:] = [
            envelope for envelope in body
            if not (
                envelope["type"] == "message_retention_coverage"
                and envelope["record"]["message_id"] == 2
            ) and not (
                envelope["type"] == "chunk"
                and envelope["record"]["id"] == chunk_id
            )
        ]

    _rewrite(sparse_wire, remove_middle)
    _downgrade_wire_to_v9(donor_wire)

    target = HyMem(_config(tmp_path / "middle-target"))
    try:
        target.import_(sparse_wire)
        slice_key = _publish(
            target, session_id, [{"text": "First and third were visible."}]
        )
        assert facts.load_fact_outcome_source_manifest(
            target.conn, slice_key, verify_result=True
        ) is not None
        assert [row[0] for row in target.conn.execute(
            "SELECT source_message_id FROM fact_extraction_source_occurrences "
            "WHERE slice_key=? ORDER BY ordinal", (slice_key,),
        )] == [1, 3]

        def snapshot() -> tuple:
            return (
                _fact_state(target),
                [tuple(row) for row in target.conn.execute(
                    "SELECT * FROM sessions ORDER BY id"
                )],
                [tuple(row) for row in target.conn.execute(
                    "SELECT * FROM chunks ORDER BY id"
                )],
                [tuple(row) for row in target.conn.execute(
                    "SELECT * FROM message_retention_coverage "
                    "ORDER BY message_id,chunk_id,coverage_version"
                )],
                [tuple(row) for row in target.conn.execute(
                    "PRAGMA foreign_key_check"
                )],
            )

        before = snapshot()
        with pytest.raises(ValueError, match="skip|corrupt|invalid"):
            target.import_(donor_wire)
        assert snapshot() == before
        assert target.conn.execute(
            "SELECT 1 FROM message_retention_coverage WHERE message_id=2"
        ).fetchone() is None
        assert facts.fact_session_authority_is_valid(target.conn, session_id)
    finally:
        target.close()


def test_v10_redaction_scrubs_logical_json_entities_controls_and_raw_db(tmp_path):
    email = "secret.person@example.com"
    normalized_email = "secret_person_example_com"
    pem = "-----BEGIN PRIVATE KEY-----\nverysecretmaterial\n-----END PRIVATE KEY-----"
    src = HyMem(_config(tmp_path / "redact-src", redact=False))
    try:
        with core_db.transaction(src.conn):
            register_session_peer(
                src.conn, "portable-facts", "workspace", "peer", "user",
                configuration={"nested": {email: email, "pem": pem}},
            )
            src.conn.execute(
                "UPDATE peers SET metadata=? WHERE id='peer' AND workspace_id='workspace'",
                (json.dumps({"escaped": email, "pem": pem}),),
            )
        src.log_message(
            "portable-facts", "user", f"Contact {email} for access.",
            created_at="2025-01-02T03:04:05.000Z",
            source_peer_id="peer", source_workspace_id="workspace",
        )
        extraction = facts.extract_facts(
            src.conn, "portable-facts",
            StubLLMClient(default=json.dumps([{
                "text": f"Contact {email} for access.",
                "entities": [normalized_email],
            }])), src.config,
        )
        assert extraction is not None
        with core_db.transaction(src.conn):
            facts.persist_facts(src.conn, "portable-facts", extraction)
        proof = src.conn.execute(
            "SELECT message_id,chunk_id FROM message_retention_coverage "
            "WHERE coverage_version='dream-lossless-message-v1'"
        ).fetchone()
        with core_db.transaction(src.conn):
            record_message_coverage(
                src.conn, message_id=proof["message_id"],
                chunk_id=proof["chunk_id"], coverage_version=email,
            )
        src.conn.execute(
            "UPDATE chunks SET salience_reason=? WHERE id=?", (email, proof["chunk_id"])
        )
        src.conn.execute(
            "UPDATE sessions SET digest_published_generation=?,"
            "digested_prompt_version=?,profile_prompt_version=?,"
            "profile_published_generation=?,episodes_prompt_version=? "
            "WHERE id='portable-facts'", (email, email, email, email, email),
        )
        out = tmp_path / "redact-v10.jsonl"
        src.export(out)
    finally:
        src.close()

    dst = HyMem(_config(tmp_path / "redact-dst", redact=True))
    try:
        dst.import_(out)
        fact = dst.conn.execute("SELECT text,entities FROM narrative_facts").fetchone()
        assert email not in fact["text"]
        assert normalized_email not in fact["entities"]
        assert all(
            email not in json.dumps(json.loads(row[0]), ensure_ascii=False)
            and pem not in json.dumps(json.loads(row[0]), ensure_ascii=False)
            for row in (
                dst.conn.execute("SELECT metadata FROM peers").fetchall()
                + dst.conn.execute("SELECT configuration FROM session_peers").fetchall()
            )
        )
        controls = dst.conn.execute(
            "SELECT digest_published_generation,digested_prompt_version,"
            "profile_prompt_version,profile_published_generation,"
            "episodes_prompt_version FROM sessions WHERE id='portable-facts'"
        ).fetchone()
        assert tuple(controls) == (None, None, None, None, None)
        versions = [row[0] for row in dst.conn.execute(
            "SELECT coverage_version FROM message_retention_coverage"
        )]
        assert email not in versions
        assert "dream-lossless-message-v1" in versions
        assert facts.fact_session_authority_is_valid(dst.conn, "portable-facts")
        dump = "\n".join(dst.conn.iterdump())
        assert email not in dump and normalized_email not in dump and pem not in dump
    finally:
        db_path = dst.config.root / "hymem.sqlite"
        dst.close()
    raw = db_path.read_bytes()
    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists():
        raw += wal.read_bytes()
    assert email.encode() not in raw
    assert normalized_email.encode() not in raw
    assert b"verysecretmaterial" not in raw


@pytest.mark.parametrize(
    ("external", "wire_source_time", "expected_source_time"),
    [
        (
            False,
            "malformed-secret.person@example.com",
            "0001-01-01T00:00:00.000Z",
        ),
        (
            True,
            "2025-01-02T04:04:05+01:00",
            "2025-01-02T03:04:05.000Z",
        ),
    ],
    ids=("native-malformed", "external-noncanonical"),
)
def test_v10_redaction_rewrites_multi_version_source_clock_once(
    tmp_path, external, wire_source_time, expected_source_time,
):
    """Every proof/reference follows one canonical source-clock rewrite."""

    secret = "secret.person@example.com"
    normalized_secret = "secret_person_example_com"
    session_id = "portable-facts"
    src = HyMem(_config(tmp_path / f"clock-src-{external}"))
    wire = tmp_path / f"clock-{external}.jsonl"
    try:
        if external:
            with core_db.transaction(src.conn):
                register_session_peer(
                    src.conn, session_id, "workspace", "peer", "user"
                )
        src.log_message(
            session_id, "user", f"Contact {secret} for the migration.",
            created_at="2025-01-02T03:04:05.000Z",
            source_peer_id="peer" if external else None,
            source_workspace_id="workspace" if external else None,
        )
        src.conn.execute(
            "UPDATE sessions SET started_at='2025-01-02T03:04:00.000Z' "
            "WHERE id=?", (session_id,),
        )
        _publish(src, session_id, [{
            "text": f"Contact {secret} for the migration.",
            "date": "2025-01-02",
            "entities": [normalized_secret],
        }])
        proof = src.conn.execute(
            "SELECT message_id,chunk_id FROM message_retention_coverage "
            "WHERE coverage_version='dream-lossless-message-v1'"
        ).fetchone()
        with core_db.transaction(src.conn):
            record_message_coverage(
                src.conn, message_id=proof["message_id"],
                chunk_id=proof["chunk_id"],
                coverage_version="legacy-clock-copy-v1",
            )
        src.export(wire)
    finally:
        src.close()

    def rewrite_source_clock(body):
        proofs = [
            envelope["record"] for envelope in body
            if envelope["type"] == "message_retention_coverage"
            and envelope["record"]["message_id"] == proof["message_id"]
        ]
        assert len(proofs) == 2
        chunk = next(
            envelope["record"] for envelope in body
            if envelope["type"] == "chunk"
            and envelope["record"]["id"] == proof["chunk_id"]
        )
        if external:
            payload = json.loads(chunk["text"])
            payload["source_created_at"] = wire_source_time
            chunk["text"] = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            source_hash = hashlib.sha256(
                chunk["text"].encode("utf-8")
            ).hexdigest()
        else:
            source_hash = proofs[0]["message_content_hash"]
        for coverage in proofs:
            coverage["source_created_at"] = wire_source_time
            coverage["message_content_hash"] = source_hash

        source = next(
            envelope["record"] for envelope in body
            if envelope["type"] == "fact_extraction_source_occurrence"
        )
        source["source_created_at"] = wire_source_time
        source["source_content_hash"] = source_hash
        occurrence = facts.BoundSourceOccurrence(
            message_id=source["source_message_id"],
            session_id=source["source_session_id"],
            role=source["source_role"],
            source_peer_id=source["source_peer_id"],
            source_workspace_id=source["source_workspace_id"],
            source_created_at=source["source_created_at"],
            coverage_chunk_id=source["source_coverage_chunk_id"],
            coverage_version=source["source_coverage_version"],
            content_hash=source["source_content_hash"],
        )
        outcome = next(
            envelope["record"] for envelope in body
            if envelope["type"] == "fact_extraction_outcome"
        )
        outcome["source_manifest_hash"] = facts.source_manifest_hash(
            facts.FACT_SOURCE_MANIFEST_VERSION, (occurrence,)
        )

    _rewrite(wire, rewrite_source_clock)

    dst = HyMem(_config(tmp_path / f"clock-dst-{external}", redact=True))
    try:
        dst.import_(wire)
        coverage_rows = dst.conn.execute(
            "SELECT message_id,chunk_id,coverage_version,source_created_at,"
            "message_content_hash FROM message_retention_coverage "
            "ORDER BY coverage_version"
        ).fetchall()
        assert len(coverage_rows) == 2
        assert {row["source_created_at"] for row in coverage_rows} == {
            expected_source_time
        }
        assert len({row["message_content_hash"] for row in coverage_rows}) == 1
        source = dst.conn.execute(
            "SELECT source_created_at,source_content_hash "
            "FROM fact_extraction_source_occurrences"
        ).fetchone()
        assert source["source_created_at"] == expected_source_time
        assert source["source_content_hash"] == coverage_rows[0][
            "message_content_hash"
        ]
        chunk_text = dst.conn.execute(
            "SELECT text FROM chunks WHERE id=?", (coverage_rows[0]["chunk_id"],)
        ).fetchone()[0]
        payload = json.loads(chunk_text)
        assert secret not in payload["content"]
        if external:
            assert payload["source_created_at"] == expected_source_time
        else:
            assert "source_created_at" not in payload
        for row in coverage_rows:
            from hymem.dreaming.lossless import validate_message_coverage_artifact

            assert validate_message_coverage_artifact(
                dst.conn, message_id=row["message_id"],
                chunk_id=row["chunk_id"],
                coverage_version=row["coverage_version"],
            ).source_created_at == expected_source_time
        assert facts.fact_session_authority_is_valid(dst.conn, session_id)
        assert sum(dst.import_(wire).values()) == 0
        dump = "\n".join(dst.conn.iterdump())
        assert secret not in dump
        assert normalized_secret not in dump
        assert wire_source_time not in dump
    finally:
        db_path = dst.config.root / "hymem.sqlite"
        dst.close()
    raw = db_path.read_bytes()
    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists():
        raw += wal.read_bytes()
    assert secret.encode() not in raw
    assert normalized_secret.encode() not in raw
    assert wire_source_time.encode() not in raw
