"""Search-shadow invariants for authoritative narrative facts."""

from __future__ import annotations

import json

from hymem import HyMem
from hymem.core import db as core_db
from hymem.extraction.llm import StubLLMClient


_FACTS_CLOSER = "Return the JSON array of narrative facts now"
_TOKEN = "quasarneedle"


def _seed_authoritative_fact(hy: HyMem, session_id: str) -> int:
    hy.open_session(session_id)
    hy.log_message(session_id, "user", f"The {_TOKEN} service stays on fly.io.")
    hy.close_session(session_id)
    hy.dream()
    row = hy.conn.execute(
        "SELECT id FROM narrative_facts "
        "WHERE session_id=? AND lifecycle_status='active'",
        (session_id,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _hy_with_fact(cfg, session_id: str) -> tuple[HyMem, int]:
    payload = [{
        "text": f"The {_TOKEN} service stays on fly.io.",
        "date": None,
        "entities": ["fly.io"],
    }]
    llm = StubLLMClient(
        fixtures={_FACTS_CLOSER: json.dumps(payload)},
        default="[]",
    )
    hy = HyMem(cfg, llm=llm)
    return hy, _seed_authoritative_fact(hy, session_id)


def _matches(conn, token: str) -> list[tuple[int, float]]:
    return [
        (int(row["rowid"]), float(row["rank"]))
        for row in conn.execute(
            "SELECT rowid, bm25(narrative_facts_fts) AS rank "
            "FROM narrative_facts_fts "
            "WHERE narrative_facts_fts MATCH ? ORDER BY rank, rowid",
            (token,),
        ).fetchall()
    ]


def test_legacy_and_retracted_spam_never_changes_fact_bm25(cfg):
    """Non-current projections are absent from both hits and corpus stats."""

    hy, fact_id = _hy_with_fact(cfg, "fact-fts-membership")
    try:
        conn = hy.conn
        before = _matches(conn, _TOKEN)
        assert [rowid for rowid, _ in before] == [fact_id]

        fact = conn.execute(
            "SELECT session_id,start_message_id,end_message_id,prompt_version,"
            "source_outcome_key FROM narrative_facts WHERE id=?",
            (fact_id,),
        ).fetchone()
        assert fact is not None

        # Legacy imports are supported historical state but carry no exact
        # source proof. A large corpus of them must not alter active BM25 IDF.
        conn.executemany(
            "INSERT INTO narrative_facts("
            "session_id,start_message_id,end_message_id,text,prompt_version"
            ") VALUES (?,?,?,?,?)",
            [
                (
                    fact["session_id"],
                    fact["start_message_id"],
                    fact["end_message_id"],
                    f"legacy {_TOKEN} spam {index}",
                    fact["prompt_version"],
                )
                for index in range(300)
            ],
        )

        # Retracted authoritative projections are equally historical. These
        # rows use a real complete source outcome so the test exercises the
        # membership predicate rather than relying only on NULL provenance.
        with core_db.evidence_mutation(conn):
            conn.executemany(
                "INSERT INTO narrative_facts("
                "session_id,start_message_id,end_message_id,text,prompt_version,"
                "invalid_at,source_outcome_key,fact_key,current_generation,"
                "lifecycle_status"
                ") VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        fact["session_id"],
                        fact["start_message_id"],
                        fact["end_message_id"],
                        f"retracted {_TOKEN} spam {index}",
                        fact["prompt_version"],
                        "2026-01-02T00:00:00.000Z",
                        fact["source_outcome_key"],
                        f"sha256:{index + 1:064x}",
                        1,
                        "retracted",
                    )
                    for index in range(300)
                ],
            )

        after = _matches(conn, _TOKEN)
        assert after == before

        # The update trigger follows lifecycle transitions in both directions.
        with core_db.evidence_mutation(conn):
            conn.execute(
                "UPDATE narrative_facts SET lifecycle_status='retracted', "
                "invalid_at='2026-01-03T00:00:00.000Z' WHERE id=?",
                (fact_id,),
            )
        assert _matches(conn, _TOKEN) == []

        with core_db.evidence_mutation(conn):
            conn.execute(
                "UPDATE narrative_facts SET lifecycle_status='active', "
                "invalid_at=NULL WHERE id=?",
                (fact_id,),
            )
        assert _matches(conn, _TOKEN) == before
    finally:
        hy.close()


def test_startup_heals_unconditional_triggers_and_stale_fts_rows(cfg):
    """Opening a stamped v46 store removes stale docs and old trigger drift."""

    hy, fact_id = _hy_with_fact(cfg, "fact-fts-heal")
    try:
        conn = hy.conn
        fact = conn.execute(
            "SELECT session_id,start_message_id,end_message_id,prompt_version "
            "FROM narrative_facts WHERE id=?",
            (fact_id,),
        ).fetchone()
        assert fact is not None
        conn.execute(
            "INSERT INTO narrative_facts("
            "session_id,start_message_id,end_message_id,text,prompt_version"
            ") VALUES (?,?,?,?,?)",
            (
                fact["session_id"],
                fact["start_message_id"],
                fact["end_message_id"],
                "staleftsmarker legacy projection",
                fact["prompt_version"],
            ),
        )
        legacy_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

        # Simulate an older/broken binary: poison the index directly and leave
        # behind an unconditional same-named trigger that schema IF NOT EXISTS
        # alone cannot replace.
        conn.execute(
            "INSERT INTO narrative_facts_fts(rowid,text) VALUES (?,?)",
            (legacy_id, "staleftsmarker legacy projection"),
        )
        conn.execute("DROP TRIGGER narrative_facts_fts_insert")
        conn.execute(
            "CREATE TRIGGER narrative_facts_fts_insert "
            "AFTER INSERT ON narrative_facts BEGIN "
            "INSERT INTO narrative_facts_fts(rowid,text) "
            "VALUES(new.id,new.text); END"
        )
        assert [rowid for rowid, _ in _matches(conn, "staleftsmarker")] == [
            legacy_id
        ]

        core_db.initialize(conn)

        assert _matches(conn, "staleftsmarker") == []
        assert [rowid for rowid, _ in _matches(conn, _TOKEN)] == [fact_id]

        conn.execute(
            "INSERT INTO narrative_facts("
            "session_id,start_message_id,end_message_id,text,prompt_version"
            ") VALUES (?,?,?,?,?)",
            (
                fact["session_id"],
                fact["start_message_id"],
                fact["end_message_id"],
                "posthealmarker legacy projection",
                fact["prompt_version"],
            ),
        )
        assert _matches(conn, "posthealmarker") == []

        trigger_sql = " ".join(
            str(row["sql"] or "").lower()
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'narrative_facts_fts_%' ORDER BY name"
            ).fetchall()
        )
        assert "source_outcome_key is not null" in trigger_sql
        assert "lifecycle_status = 'active'" in trigger_sql
        assert "invalid_at is null" in trigger_sql
    finally:
        hy.close()
