"""Markdown compatibility sidecars never publish rolled-back DB state."""
from dataclasses import replace
import sqlite3

import pytest

from hymem import HyMem
from hymem.core import db, markdown_io
from hymem.dreaming import phase2
from hymem.deadline import DeadlineExceeded, MonotonicDeadline, use_deadline
from tests.conftest import make_routed_llm


def _hy(cfg):
    config = replace(cfg, aggregation_nodes_enabled=False,
                     profile_extraction_enabled=False, facts_extraction_enabled=False)
    llm = make_routed_llm([], [{"kind": "preference", "statement": "user prefers uv"}])
    llm = type(llm)(fixtures={
        "You analyze one conversation session": '{"episodes":[],"summary":"","procedures":[]}',
        **llm.fixtures,
    }, default="[]")
    hy = HyMem(config, llm=llm)
    hy.log_message("x", "user", "I prefer uv for Python tooling in every project.")
    hy.close_session("x")
    return hy


def test_second_file_failure_keeps_committed_profile_and_next_dream_repairs(cfg, monkeypatch):
    hy = _hy(cfg)
    try:
        actual = markdown_io.write_section
        observed = []
        def fail_memory(path, *args, **kwargs):
            observed.append((path.name, hy.conn.in_transaction))
            if path == hy.config.memory_md_path:
                raise OSError("second file failed")
            return actual(path, *args, **kwargs)
        monkeypatch.setattr(markdown_io, "write_section", fail_memory)
        with pytest.raises(OSError, match="second file failed"):
            hy.dream()
        assert observed and all(not active for _, active in observed)
        assert hy.conn.execute("SELECT COUNT(*) FROM current_profile_entries").fetchone()[0] == 1
        assert "user prefers uv" in hy.config.user_md_path.read_text()
        monkeypatch.setattr(markdown_io, "write_section", actual)
        calls_before = len(hy._llm.calls)
        hy.dream()
        assert len(hy._llm.calls) == calls_before
        assert markdown_io.read_section(hy.config.memory_md_path, "project_insights")
        assert not hy.conn.in_transaction
    finally:
        hy.close()


def test_direct_profile_inside_rolled_back_transaction_never_writes(cfg, monkeypatch):
    hy = _hy(cfg)
    try:
        calls = []
        monkeypatch.setattr(markdown_io, "write_section", lambda *args, **kwargs: calls.append(args))
        with pytest.raises(RuntimeError, match="rollback"):
            with db.transaction(hy.conn):
                phase2.consolidate_profile(hy.conn, hy.config)
                raise RuntimeError("rollback")
        assert calls == []
    finally:
        hy.close()


def _hold_phase2_sql(hy):
    hy.conn.execute("CREATE TEMP TRIGGER fail_profile_decision BEFORE INSERT ON profile_marker_decisions "
                    "BEGIN SELECT RAISE(ABORT,'profile SQL failed'); END")
    with pytest.raises(sqlite3.IntegrityError, match="profile SQL failed"):
        hy.dream()
    hy.conn.execute("DROP TRIGGER fail_profile_decision")
    assert hy.conn.execute("SELECT COUNT(*) FROM profile_entries").fetchone()[0] == 0
    return hy.conn.execute("SELECT phase1_generation_key FROM behavioral_markers LIMIT 1").fetchone()[0]


def test_phase2_sql_rollback_cannot_change_either_file(cfg):
    hy = _hy(cfg)
    try:
        hy.config.user_md_path.write_text("# Curated user\nKeep this exact text.\n")
        hy.config.memory_md_path.write_text("# Curated memory\nKeep this exact text too.\n")
        before = (hy.config.user_md_path.read_bytes(), hy.config.memory_md_path.read_bytes())
        _hold_phase2_sql(hy)
        assert (hy.config.user_md_path.read_bytes(), hy.config.memory_md_path.read_bytes()) == before
        assert not hy.conn.in_transaction
        hy.dream()
        assert "user prefers uv" in hy.config.user_md_path.read_text()
    finally:
        hy.close()


def test_standalone_profile_commits_before_failed_publication(cfg, monkeypatch):
    hy = _hy(cfg)
    try:
        generation = _hold_phase2_sql(hy)
        observed = []
        def fail(*args, **kwargs):
            observed.append(hy.conn.in_transaction)
            raise OSError("profile file failed")
        monkeypatch.setattr(markdown_io, "write_section", fail)
        with pytest.raises(OSError, match="profile file failed"):
            phase2.consolidate_profile(hy.conn, hy.config, phase1_generation_key=generation)
        assert observed == [False]
        assert hy.conn.execute("SELECT COUNT(*) FROM current_profile_entries").fetchone()[0] == 1
        assert hy.read_conn.execute("SELECT COUNT(*) FROM profile_entries").fetchone()[0] == 1
        assert not hy.conn.in_transaction
    finally:
        hy.close()


def test_successful_phase3_refresh_and_manual_sections_are_preserved(cfg, monkeypatch):
    hy = _hy(cfg)
    try:
        for path, section in ((hy.config.user_md_path, "behavioral_profile"),
                              (hy.config.memory_md_path, "project_insights")):
            markdown_io.write_section(path, section, "stale auto content")
            path.write_text("# Manual preface\n" + path.read_text() + "\nManual footer\n")
        actual = markdown_io.write_section
        observed = []
        def observe(path, *args, **kwargs):
            observed.append((path.name, hy.conn.in_transaction))
            return actual(path, *args, **kwargs)
        monkeypatch.setattr(markdown_io, "write_section", observe)
        hy.dream()
        assert observed == [("USER.md", False), ("MEMORY.md", False), ("MEMORY.md", False)]
        for path in (hy.config.user_md_path, hy.config.memory_md_path):
            text = path.read_text()
            assert text.startswith("# Manual preface\n")
            assert text.endswith("\nManual footer\n")
            assert "stale auto content" not in text
    finally:
        hy.close()


def test_phase3_sql_rollback_does_not_publish_its_refresh(cfg, monkeypatch):
    hy = _hy(cfg)
    try:
        hy.conn.execute("INSERT INTO token_overlap_index(token,canonical) VALUES ('keep','keep')")
        hy.conn.execute("CREATE TEMP TRIGGER fail_phase3 BEFORE DELETE ON token_overlap_index "
                        "BEGIN SELECT RAISE(ABORT,'phase3 SQL failed'); END")
        actual = markdown_io.write_section
        observed = []
        def observe(path, *args, **kwargs):
            observed.append((path.name, hy.conn.in_transaction))
            return actual(path, *args, **kwargs)
        monkeypatch.setattr(markdown_io, "write_section", observe)
        with pytest.raises(sqlite3.IntegrityError, match="phase3 SQL failed"):
            hy.dream()
        assert observed == [("USER.md", False), ("MEMORY.md", False)]
        assert hy.conn.execute("SELECT COUNT(*) FROM token_overlap_index WHERE token='keep'").fetchone()[0] == 1
        assert hy.conn.execute("SELECT COUNT(*) FROM current_profile_entries").fetchone()[0] == 1
        hy.conn.execute("DROP TRIGGER fail_phase3")
        observed.clear()
        hy.dream()
        assert observed == [("USER.md", False), ("MEMORY.md", False), ("MEMORY.md", False)]
    finally:
        hy.close()


@pytest.mark.parametrize("publish", [phase2.publish_profile, phase2.consolidate_insights])
def test_direct_sidecar_publish_rejects_uncommitted_state(cfg, publish):
    hy = _hy(cfg)
    try:
        with db.transaction(hy.conn):
            with pytest.raises(RuntimeError, match="outside a transaction"):
                publish(hy.conn, hy.config)
        assert not hy.config.user_md_path.exists()
        assert not hy.config.memory_md_path.exists()
    finally:
        hy.close()


@pytest.mark.parametrize("failure", ["lease", "deadline"])
def test_final_replacement_guard_holds_old_file_and_cleans_temp(cfg, monkeypatch, failure):
    hy = _hy(cfg)
    token = None
    try:
        path = hy.config.user_md_path
        path.write_text("old committed snapshot\n")
        actual = markdown_io.write_section
        clock = [0.0]
        if failure == "lease":
            hy.conn.execute("INSERT INTO run_lock(name,holder,acquired_at) "
                            "VALUES ('dreaming','owner',CURRENT_TIMESTAMP)")
            token = db.activate_transaction_lease_fence(hy.conn, name="dreaming", holder="owner")
        def interrupt_before_temp_write(*args, **kwargs):
            if failure == "lease":
                hy.conn.execute("UPDATE run_lock SET holder='successor' WHERE name='dreaming'")
            else:
                clock[0] = 2.0
            return actual(*args, **kwargs)
        monkeypatch.setattr(markdown_io, "write_section", interrupt_before_temp_write)
        expected = db.LeaseOwnershipLost if failure == "lease" else DeadlineExceeded
        with use_deadline(MonotonicDeadline(1.0, clock=lambda: clock[0])):
            with pytest.raises(expected):
                phase2.publish_profile(hy.conn, hy.config)
        assert path.read_text() == "old committed snapshot\n"
        assert list(path.parent.glob("USER.md.*")) == []
        assert not hy.conn.in_transaction
    finally:
        if token is not None:
            db.deactivate_transaction_lease_fence(token)
        hy.close()
