"""Capture is one validated, closed source batch before embedding or dreaming."""

from __future__ import annotations

import json
import sqlite3

import pytest

from hymem import server
from hymem.dreaming.runner import DreamReport
from hymem.extraction.embeddings import MappedStubEmbeddingClient


_TURNS = [
    {"role": "user", "content": "First accepted batch turn."},
    {"role": "assistant", "content": "FAIL_SECOND"},
]


def _snapshot(hy):
    # Includes source/coverage/FTS tables, peer effects, and session clocks.
    return "\n".join(hy.conn.iterdump())


@pytest.mark.parametrize("bad", [
    42, None, [], "private invalid turn", True,
    {"role": [], "content": "private role"},
    {"role": None, "content": "private role"},
    {"role": 42, "content": "private role"},
    {"role": "user", "content": ["private", "content"]},
    {"role": "user", "content": {"private": "content"}},
    {"role": "user", "content": None},
    {"role": "user", "content": False},
    {"role": "unsupported", "content": 42},
])
def test_capture_validates_every_item_before_initializing_memory(monkeypatch, bad):
    def forbidden_init():
        raise AssertionError("invalid capture initialized memory")

    monkeypatch.setattr(server, "_get_hy", forbidden_init)
    for _ in range(2):
        result = server._do_capture("capture", json.dumps([_TURNS[0], bad]))
        assert result.startswith("error:")
        assert len(result) <= 160
        assert "private" not in result


@pytest.mark.parametrize(("session_id", "messages", "dream"), [
    ("capture", None, False), ("capture", [], False),
    ("capture", "not JSON", False), ("capture", "{}", False),
    ("capture", "null", False), ("capture", "[", False),
    (None, "[]", False), (42, "[]", False), ("", "[]", False),
    ("  ", "[]", False), ("capture", "[]", "false"),
])
def test_capture_rejects_invalid_envelope_before_memory_init(
    monkeypatch, session_id, messages, dream,
):
    def forbidden_init():
        raise AssertionError("invalid capture initialized memory")

    monkeypatch.setattr(server, "_get_hy", forbidden_init)
    result = server._do_capture(session_id, messages, dream=dream)
    assert result.startswith("error:") and len(result) <= 160


_FAULTS = {
    "insert": """CREATE TEMP TRIGGER capture_fault BEFORE INSERT ON messages
        WHEN NEW.content='FAIL_SECOND'
        BEGIN SELECT RAISE(ABORT, 'forced insert fault'); END""",
    "coverage": """CREATE TEMP TRIGGER capture_fault BEFORE INSERT ON message_retention_coverage
        WHEN NEW.message_id=(SELECT max(id) FROM messages)
        BEGIN SELECT RAISE(ABORT, 'forced coverage fault'); END""",
    "close": """CREATE TEMP TRIGGER capture_fault BEFORE UPDATE OF ended_at ON sessions
        WHEN NEW.id='capture' AND NEW.ended_at IS NOT NULL
        BEGIN SELECT RAISE(ABORT, 'forced close fault'); END""",
}


@pytest.mark.parametrize(("state", "fault"), [
    (state, fault) for state in ("new", "open", "closed")
    for fault in _FAULTS if not (state == "closed" and fault == "close")
])
def test_capture_sql_fault_rolls_back_complete_batch_and_failed_retries(
    hy, monkeypatch, state, fault,
):
    if state != "new":
        hy.log_message("capture", "user", "Previously committed source.")
        if state == "closed":
            hy.close_session("capture")
    before = _snapshot(hy)
    hy.conn.execute(_FAULTS[fault])
    calls = []
    monkeypatch.setattr(server, "_get_hy", lambda: hy)
    monkeypatch.setattr(hy, "_embed_pending_messages_best_effort", lambda ids: calls.append("embedding"))
    monkeypatch.setattr(hy, "dream", lambda **kwargs: calls.append("dream"))
    for _ in range(2):
        with pytest.raises(sqlite3.IntegrityError, match="forced"):
            server._do_capture("capture", json.dumps(_TURNS), dream=True)
        assert _snapshot(hy) == before
        assert calls == []
        assert not hy.conn.in_transaction
    hy.conn.execute("DROP TRIGGER capture_fault")
    result = server._do_capture("capture", json.dumps(_TURNS), dream=False)
    assert result.startswith("logged 2 turns")
    assert calls == ["embedding"]
    assert hy.conn.execute("SELECT count(*) FROM messages WHERE session_id='capture'").fetchone()[0] == (
        2 if state == "new" else 3
    )
    assert hy.conn.execute("SELECT ended_at FROM sessions WHERE id='capture'").fetchone()[0] is not None


@pytest.mark.parametrize("failure", [None, "FAIL_SECOND"])
def test_real_embedding_and_targeted_dream_observe_committed_closed_batch(
    hy, monkeypatch, failure,
):
    observed = []
    with sqlite3.connect(hy.config.db_path) as observer:
        def state():
            ended = observer.execute("SELECT ended_at FROM sessions WHERE id='capture'").fetchone()
            return (
                bool(hy.conn.in_transaction),
                ended is not None and ended[0] is not None,
                observer.execute("SELECT count(*) FROM messages").fetchone()[0],
                observer.execute("SELECT count(*) FROM message_retention_coverage").fetchone()[0],
            )

        class TransactionProbe:
            @property
            def in_transaction(self):
                observed.append(("embedding", state()))
                return hy.conn.in_transaction

        embedder = MappedStubEmbeddingClient(conn=TransactionProbe(), fail_on=failure)
        hy.set_embedding_client(embedder)
        monkeypatch.setattr(server, "_get_hy", lambda: hy)

        def dream(*, session_ids):
            assert session_ids == ["capture"]
            observed.append(("dream", state()))
            return DreamReport()

        monkeypatch.setattr(hy, "dream", dream)
        result = server._do_capture("capture", json.dumps(_TURNS), dream=True)
    assert observed == [
        ("embedding", (False, True, 2, 2)),
        ("dream", (False, True, 2, 2)),
    ]
    assert len(embedder.calls) == 1 and len(embedder.calls[0]) == 2
    assert "logged 2 turns" in result and "store-wide" in result
    assert hy.conn.execute("SELECT count(*) FROM message_embeddings").fetchone()[0] == (
        2 if failure is None else 0
    )


def test_capture_preserves_compatible_filtering_and_empty_batch_closure(hy, monkeypatch):
    monkeypatch.setattr(server, "_get_hy", lambda: hy)
    result = server._do_capture("capture", json.dumps([
        {"role": "unknown", "content": "ignored"}, {"role": "user", "content": ""},
        {}, {"content": "missing role"}, {"role": "user"},
        {"role": "system", "content": "system context"},
        {"role": "tool", "content": "tool context"},
    ]), dream=False)
    assert result.startswith("logged 2 turns")
    assert [row[0] for row in hy.conn.execute("SELECT role FROM messages ORDER BY id")] == ["system", "tool"]
    assert server._do_capture("empty", "[]", dream=False).startswith("logged 0 turns")
    assert hy.conn.execute("SELECT ended_at FROM sessions WHERE id='empty'").fetchone()[0] is not None


def test_log_messages_default_remains_open_and_closure_is_opt_in(hy):
    hy.log_messages("ordinary", [("user", "ordinary open batch")])
    assert hy.conn.execute("SELECT ended_at FROM sessions WHERE id='ordinary'").fetchone()[0] is None
    ids = hy.log_messages("closed", [("user", "closed batch")], close_session=True)
    assert len(ids) == 1
    assert hy.conn.execute("SELECT ended_at FROM sessions WHERE id='closed'").fetchone()[0] is not None


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_batch_closure_flag_must_be_boolean_without_mutation(hy, invalid):
    before = _snapshot(hy)
    with pytest.raises(ValueError, match="close_session must be a boolean"):
        hy.log_messages("capture", [("user", "not accepted")], close_session=invalid)
    assert _snapshot(hy) == before


@pytest.mark.parametrize("state", ["new", "existing"])
def test_closed_peer_batch_rolls_back_membership_coverage_and_session(hy, state):
    if state == "existing":
        hy.open_session("capture", source_workspace_id="workspace")
    before = _snapshot(hy)
    hy.conn.execute(_FAULTS["close"])
    for _ in range(2):
        with pytest.raises(sqlite3.IntegrityError, match="forced close"):
            hy.log_messages("capture", [(t["role"], t["content"]) for t in _TURNS],
                source_peer_ids=["alice", "bot"], source_workspace_id="workspace", close_session=True)
        assert _snapshot(hy) == before
    hy.conn.execute("DROP TRIGGER capture_fault")
    hy.log_messages("capture", [(t["role"], t["content"]) for t in _TURNS],
        source_peer_ids=["alice", "bot"], source_workspace_id="workspace", close_session=True)
    assert hy.conn.execute("SELECT count(*) FROM peers").fetchone()[0] == 2
    assert hy.conn.execute("SELECT count(*) FROM session_peers").fetchone()[0] == 2
