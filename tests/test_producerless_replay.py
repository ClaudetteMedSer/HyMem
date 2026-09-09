"""Legacy prompt-only history must replay without inventing authority or rows."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from hymem import HyMem, HyMemConfig, portability
from hymem.core import extraction_audit as audit
from hymem.extraction.triples import Triple
from tests.test_extraction_audit import _old_format_claim_wire
from tests.test_portability import (
    _persist_portable_claim,
    _portable_claim_chunk,
    _seed_shared_claim_artifact,
)


def _freeze_format(wire, version):
    """Use the released columns, omitting unavailable producer declarations."""
    if version == portability.EXPORT_VERSION:
        return
    spec = getattr(portability, f"_V{version}_EXPORT_SPEC")
    columns = {kind: fields for kind, _table, fields in spec}
    objects = [json.loads(line) for line in wire.read_text().splitlines()]
    objects[0]["version"] = version
    body = [objects[0], *[
        {"type": row["type"], "record": {
            field: row["record"][field] for field in columns[row["type"]]
        }}
        for row in objects[1:-1] if row["type"] in columns
    ]]
    encoded = "".join(json.dumps(row) + "\n" for row in body)
    end = {
        "type": "_end",
        "counts": {kind: sum(row["type"] == kind for row in body) for kind in columns},
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }
    wire.write_text(encoded + json.dumps(end) + "\n")


def _originals(conn):
    originals = set()
    for row in conn.execute("SELECT id FROM kg_evidence"):
        stored = conn.execute(
            "SELECT payload_json FROM kg_evidence_extraction_audit WHERE evidence_id=?",
            (row["id"],),
        ).fetchall()
        if stored:
            originals.update(item[0] for item in stored)
        else:
            originals.add(audit.carrier_record(conn, row["id"])["payload_json"])
    return originals


def _wire_originals(wire):
    records = [json.loads(line) for line in wire.read_text().splitlines()]
    edges = {
        row["record"]["id"]: [row["record"][field] for field in (
            "subject_canonical", "predicate", "object_canonical"
        )]
        for row in records if row["type"] == "edge"
    }
    snapshots = {}
    for row in records:
        if row["type"] == "edge_evidence_extraction_audit":
            record = row["record"]
            snapshots.setdefault(record["evidence_id"], set()).add(record["payload_json"])
    originals = set()
    for row in records:
        if row["type"] != "edge_evidence":
            continue
        record = {"source_peer_id": None, "source_workspace_id": None, **row["record"]}
        originals.update(snapshots.get(record["id"], {
            audit.encode(audit.original(edges[record["edge_id"]], record))
        }))
    return originals


def _provenance(conn):
    return tuple(tuple(row) for row in conn.execute(
        "SELECT DISTINCT source_session_id,source_message_id,source_created_at,"
        "source_event_at,source_coverage_chunk_id,source_coverage_version,"
        "source_role,source_peer_id,source_workspace_id,provenance_status "
        "FROM kg_evidence ORDER BY source_session_id,source_message_id"
    ))


def _counts(conn):
    return tuple(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                 for table in ("kg_evidence", "kg_edge_lifecycle", "kg_lifecycle_dependencies"))


def _assert_replays_unchanged(store, wires):
    before = tuple(store.conn.iterdump())
    originals = _originals(store.conn)
    provenance = _provenance(store.conn)
    for wire in [*wires, *reversed(wires)]:
        assert sum(store.import_(wire).values()) == 0
        assert tuple(store.conn.iterdump()) == before
        assert _originals(store.conn) == originals
        assert _provenance(store.conn) == provenance
    assert not store.conn.execute("PRAGMA foreign_key_check").fetchall()
    assert store.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("version", range(7, 13))
def test_producerless_replay_and_reopen_are_exact_noops(tmp_path, version):
    wire = _old_format_claim_wire(tmp_path, version)
    cfg = HyMemConfig(root=tmp_path / "target", redact_secrets=False)
    target = HyMem(cfg)
    try:
        target.import_(wire)
        assert _counts(target.conn) == (1, 1, 0)
        assert tuple(target.conn.execute(
            "SELECT pos_evidence,neg_evidence,status FROM knowledge_graph WHERE derived=0"
        ).fetchone()) == (0, 0, "retracted")
        assert target.conn.execute("SELECT SUM(is_current) FROM kg_evidence").fetchone()[0] == 0
        assert target.conn.execute("SELECT COUNT(*) FROM phase1_generations").fetchone()[0] == 0
        assert target.conn.execute("SELECT COUNT(*) FROM kg_evidence_extraction_audit").fetchone()[0] == 0
        _assert_replays_unchanged(target, [wire])
        before = tuple(target.conn.iterdump())
    finally:
        target.close()
    reopened = HyMem(cfg)
    try:
        # Reopening reinstalls guards, which can reorder sqlite_master rows.
        assert sorted(reopened.conn.iterdump()) == sorted(before)
        _assert_replays_unchanged(reopened, [wire])
    finally:
        reopened.close()


@pytest.mark.parametrize("version", range(7, 13))
@pytest.mark.parametrize("newer_result", ["positive", "empty"])
def test_stale_producerless_history_cannot_replace_newer_authority(
    tmp_path, version, newer_result,
):
    source = HyMem(HyMemConfig(root=tmp_path / "source", redact_secrets=False))
    old_wire, new_wire = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    try:
        chunk, mid = _seed_shared_claim_artifact(source)
        old = Triple("service", "uses", "database", -1, source_message_id=mid)
        _persist_portable_claim(source, chunk, [old], prompt_version="v13")
        source.export(old_wire)
        _freeze_format(old_wire, version)
        triples = [dataclasses.replace(old, polarity=1)] if newer_result == "positive" else []
        _persist_portable_claim(source, chunk, triples, prompt_version="v14")
        source.export(new_wire)
        expected_originals = _wire_originals(old_wire) | _wire_originals(new_wire)
        expected_provenance = _provenance(source.conn)
        expected_counts = _counts(source.conn)
        expected_projection = tuple(source.conn.execute(
            "SELECT pos_evidence,neg_evidence,status FROM knowledge_graph WHERE derived=0"
        ).fetchone())
    finally:
        source.close()
    histories = []
    for index, wires in enumerate(((old_wire, new_wire), (new_wire, old_wire))):
        cfg = HyMemConfig(root=tmp_path / f"target{index}", redact_secrets=False)
        target = HyMem(cfg)
        try:
            for wire in wires:
                target.import_(wire)
            assert _counts(target.conn) == expected_counts
            assert _originals(target.conn) == expected_originals
            assert _provenance(target.conn) == expected_provenance
            assert tuple(target.conn.execute(
                "SELECT pos_evidence,neg_evidence,status FROM knowledge_graph WHERE derived=0"
            ).fetchone()) == expected_projection
            outcome = target.conn.execute(
                "SELECT prompt_generation,phase1_generation_key FROM kg_claim_extraction_outcomes"
            ).fetchone()
            assert outcome["prompt_generation"] == 14
            assert outcome["phase1_generation_key"] is not None
            assert target.conn.execute("SELECT COUNT(*) FROM kg_claim_observations").fetchone()[0] == int(newer_result == "positive")
            _assert_replays_unchanged(target, [old_wire, new_wire])
            histories.append(tuple(tuple(row) for row in target.conn.execute(
                "SELECT revision,polarity,is_current FROM kg_evidence ORDER BY revision"
            )))
        finally:
            target.close()
        reopened = HyMem(cfg)
        try:
            _assert_replays_unchanged(reopened, [old_wire, new_wire])
        finally:
            reopened.close()
    assert histories[0] == histories[1]


@pytest.mark.parametrize("version", [*range(7, 13), portability.EXPORT_VERSION])
def test_same_millisecond_revival_keeps_distinct_declared_intervals(tmp_path, version):
    source = HyMem(HyMemConfig(root=tmp_path / "source", redact_secrets=False))
    wire = tmp_path / "revival.jsonl"
    try:
        clock = source.conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
        source.conn.create_function("current_timestamp", 0, lambda: clock)
        chunk, mid = _seed_shared_claim_artifact(source)
        overlap = _portable_claim_chunk(
            source, chunk_id="overlap", message_id=mid, role="user",
            content="The service target is 65 percent",
        )
        winning = Triple("service", "uses", "database", 1, source_message_id=mid)
        _persist_portable_claim(source, chunk, [winning], prompt_version="v15")
        _persist_portable_claim(
            source, overlap, [dataclasses.replace(winning, polarity=-1)], prompt_version="v14",
        )
        positive = source.conn.execute(
            "SELECT * FROM kg_evidence WHERE polarity=1 ORDER BY revision"
        ).fetchall()
        assert len(positive) == 2
        assert [row["is_current"] for row in positive] == [0, 1]
        assert portability._normalized_wire_event(positive[0]["extracted_at"]) == portability._normalized_wire_event(positive[1]["extracted_at"])
        assert positive[0]["published_at"] == positive[1]["published_at"]
        assert positive[0]["chunk_id"] == positive[1]["chunk_id"]
        assert positive[0]["extraction_prompt_version"] == positive[1]["extraction_prompt_version"]
        source.export(wire)
        _freeze_format(wire, version)
        expected_provenance = _provenance(source.conn)
        expected_originals = _wire_originals(wire)
    finally:
        source.close()
    target = HyMem(HyMemConfig(root=tmp_path / "target", redact_secrets=False))
    try:
        target.import_(wire)
        assert target.conn.execute("SELECT COUNT(*) FROM kg_evidence").fetchone()[0] == 3
        assert target.conn.execute("SELECT COUNT(*) FROM kg_evidence WHERE polarity=1").fetchone()[0] == 2
        assert target.conn.execute("SELECT SUM(is_current) FROM kg_evidence").fetchone()[0] == int(version >= 13)
        assert _provenance(target.conn) == expected_provenance
        assert _originals(target.conn) == expected_originals
        _assert_replays_unchanged(target, [wire])
    finally:
        target.close()
