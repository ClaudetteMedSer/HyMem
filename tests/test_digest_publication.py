"""All digest surfaces publish together, never from unfinished source walks."""
from dataclasses import replace
import json
import sqlite3

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import digest, runner
from hymem.dreaming.summary import effective_session_summary
from hymem.dreaming.status import durable_dream_work_status
from hymem.session import _session_is_pristine
from tests.test_lossless_digest import RollingLLM, _quiet_cfg


class PublicationLLM(RollingLLM):
    label = "oldpublished"

    def complete(self, request):
        raw = super().complete(request)
        if request.system.startswith(("You analyze one conversation session",
                                      "You re-read one conversation session")) and raw != "not-json":
            value = json.loads(raw)
            value["summary"] = self.label + " summary for this generation."
            for procedure in value["procedures"]:
                procedure["steps"][0]["action"] = self.label + " replacement action"
            return json.dumps(value)
        return raw


def _state(hy, sid="x"):
    row = hy.conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return {
        "marker": row["digest_published_generation"],
        "summary": effective_session_summary(row),
        "auto_position": (row["auto_summary_message_id"],
                          row["auto_summary_partial_message_id"],
                          row["auto_summary_message_offset"]),
        "procedures": [(item[0], item[1], json.dumps(json.loads(item[2]), sort_keys=True), item[3]) for item in hy.conn.execute(
            "SELECT id,name,steps,status FROM procedures WHERE session_id=? ORDER BY id", (sid,))],
        "episodes": [tuple(item) for item in hy.conn.execute(
            "SELECT e.id,e.title,e.summary FROM episodes e JOIN sessions s ON s.id=e.session_id "
            "WHERE e.session_id=? AND (e.digest_generation IS NULL OR "
            "e.digest_generation=s.digest_published_generation) ORDER BY e.id", (sid,))],
    }


def _finish(hy, sid="x", limit=25):
    for _ in range(limit):
        hy.dream()
        state = hy.conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if (state["digest_cursor_prompt_version"] == state["digest_published_generation"]
                and state["digest_cursor_message_id"] == state["coverage_message_id"]
                and not state["digest_cursor_offset"]):
            return
    pytest.fail("digest failed to complete its bounded walk")


def _initial(cfg, *, long=True):
    llm = PublicationLLM(emit_slice_artifacts=True)
    hy = HyMem(_quiet_cfg(cfg, dream_digest_max_chars=300), llm=llm)
    hy.log_message("x", "user", "alpha " + ("long content " * 50 if long else "source"))
    hy.close_session("x")
    _finish(hy)
    return hy, llm


def _rebuild(hy, llm):
    hy.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
    llm.label = llm.artifact_label = "newstaged"
    assert hy.dream().budget_exhausted


def test_replacement_keeps_all_public_surfaces_through_restart_and_failure(cfg):
    hy, llm = _initial(cfg)
    try:
        old = _state(hy)
        _rebuild(hy, llm)
        assert _state(hy) == old
        config = hy.config
        hy.close()
        hy = HyMem(config, llm=llm)
        assert _state(hy) == old
        llm.fail_next_digest = True
        assert hy.dream().digest_failures == 1
        assert _state(hy) == old
        hy.dream()
        assert "newstaged summary" in llm.successful_digest_calls[-1].user
        assert _state(hy) == old
        _finish(hy)
        current = _state(hy)
        assert current["marker"] != old["marker"]
        assert "newstaged" in current["summary"]
        assert any("newstaged" in row[2] for row in current["procedures"])
        assert all("newstaged" in row[1] for row in current["episodes"])
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        hy.dream()
        assert _state(hy) == current
    finally:
        hy.close()


def test_forward_tail_is_staged_including_episodes_and_preserves_old_history(cfg):
    hy, llm = _initial(cfg, long=False)
    try:
        old = _state(hy)
        llm.label = llm.artifact_label = "newstaged"
        hy.log_message("x", "assistant", "beta " + "long tail " * 60)
        hy.dream()
        assert _state(hy) == old
        assert hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == len(old["episodes"])
        _finish(hy)
        final = _state(hy)
        assert final["marker"] == old["marker"]
        assert set(old["episodes"]).issubset(set(final["episodes"]))
        assert len(final["episodes"]) > len(old["episodes"])
        assert "newstaged" in final["summary"]
        hy.dream()
        assert _state(hy) == final
    finally:
        hy.close()


def test_failed_publication_rolls_back_summary_procedures_episodes_and_last_slice(cfg):
    hy, llm = _initial(cfg)
    try:
        old = _state(hy)
        _rebuild(hy, llm)
        hy.conn.execute("CREATE TRIGGER fail_digest_publication BEFORE UPDATE OF "
                        "digest_published_generation ON sessions WHEN new.id='x' "
                        "BEGIN SELECT RAISE(ABORT,'injected publication failure'); END")
        for _ in range(25):
            before = [tuple(row) for row in hy.conn.execute("SELECT * FROM digest_staging ORDER BY slice_key")]
            cursor = hy.conn.execute("SELECT digest_cursor_offset FROM sessions WHERE id='x'").fetchone()[0]
            try:
                hy.dream()
            except sqlite3.IntegrityError as exc:
                assert "injected publication failure" in str(exc)
                assert [tuple(row) for row in hy.conn.execute("SELECT * FROM digest_staging ORDER BY slice_key")] == before
                assert hy.conn.execute("SELECT digest_cursor_offset FROM sessions WHERE id='x'").fetchone()[0] == cursor
                break
            assert _state(hy) == old
        else:
            pytest.fail("publication failure was never exercised")
        assert _state(hy) == old
        hy.conn.execute("DROP TRIGGER fail_digest_publication")
        _finish(hy)
        assert "newstaged" in _state(hy)["summary"]
    finally:
        hy.close()


def test_manual_summary_and_other_session_are_not_changed_by_partial_rebuild(cfg):
    hy, llm = _initial(cfg)
    try:
        hy.log_message("other", "user", "alpha independent source")
        hy.close_session("other")
        _finish(hy, "other")
        other = _state(hy, "other")
        manual = "Operator-curated summary remains exact."
        hy.conn.execute("UPDATE sessions SET summary=?,summary_source='operator' WHERE id='x'", (manual,))
        old = _state(hy)
        _rebuild(hy, llm)
        assert _state(hy) == old
        assert _state(hy, "other") == other
        _finish(hy)
        row = hy.conn.execute("SELECT * FROM sessions WHERE id='x'").fetchone()
        assert row["summary"] == manual and row["summary_source"] == "operator"
        assert "newstaged" in row["auto_summary"]
        assert _state(hy, "other") == other
    finally:
        hy.close()


@pytest.mark.parametrize("damage", ["missing", "chain", "source", "payload", "cursor", "range", "orphan"])
def test_damaged_staging_is_health_visible_and_rewinds_without_publishing(cfg, damage):
    hy, llm = _initial(cfg)
    try:
        old = _state(hy)
        _rebuild(hy, llm)
        first_gen = hy.conn.execute("SELECT generation FROM digest_staging").fetchone()[0]
        if damage == "missing":
            hy.conn.execute("DELETE FROM digest_staging")
        elif damage == "chain":
            hy.conn.execute("UPDATE digest_staging SET cursor_before_offset=1")
        elif damage == "source":
            hy.conn.execute("UPDATE digest_staging SET source_sha256=?", ("0" * 64,))
        elif damage == "payload":
            hy.conn.execute("UPDATE digest_staging SET episodes_json='[42]'")
        elif damage == "cursor":
            hy.conn.execute("UPDATE sessions SET digest_cursor_message_id=coverage_message_id,"
                            "digest_cursor_partial_message_id=NULL,digest_cursor_offset=0 WHERE id='x'")
        elif damage == "orphan":
            hy.conn.execute("UPDATE sessions SET digest_cursor_message_id=auto_summary_message_id,"
                            "digest_cursor_partial_message_id=NULL,digest_cursor_offset=0,"
                            "digest_cursor_prompt_version=digest_published_generation,"
                            "digested_prompt_version=? WHERE id='x'", (hy.config.prompt_version,))
        else:
            # Matching row/session cursors are still insufficient: an edited
            # character range must disagree with the pre-call source proof.
            hy.conn.execute("UPDATE digest_staging SET cursor_after_offset=cursor_after_offset+1")
            hy.conn.execute("UPDATE sessions SET digest_cursor_offset=digest_cursor_offset+1 WHERE id='x'")
        status = durable_dream_work_status(hy.conn, hy.config, client=llm)
        assert status["pending_digests"] == 1
        assert status["malformed_digests"] == 1
        hy.dream()
        assert _state(hy) == old
        assert "chars=0:" in llm.successful_digest_calls[-1].user
        assert hy.conn.execute("SELECT generation FROM digest_staging").fetchone()[0] != first_gen
        _finish(hy)
        assert "newstaged" in _state(hy)["summary"]
    finally:
        hy.close()


def test_unpublished_digest_output_is_not_portable_and_restore_replays(cfg, tmp_path):
    hy, llm = _initial(cfg)
    try:
        old = _state(hy)
        _rebuild(hy, llm)
        path = tmp_path / "paused.jsonl"
        hy.export(path)
        wire = path.read_text()
        assert "newstaged" not in wire
        records = [json.loads(line) for line in wire.splitlines()]
        session = next(row["record"] for row in records if row["type"] == "session")
        assert session["digest_cursor_prompt_version"] is None
        assert session["digest_cursor_message_id"] is None
        assert session["digest_published_generation"] == old["marker"]
        restored = HyMem(replace(hy.config, root=tmp_path / "restore"), llm=llm)
        try:
            restored.import_(path)
            assert _state(restored) == old
            assert restored.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
            restored.dream()
            assert "chars=0:" in llm.successful_digest_calls[-1].user
            assert _state(restored) == old
            _finish(restored)
            assert "newstaged" in _state(restored)["summary"]
        finally:
            restored.close()
    finally:
        hy.close()


def test_v57_partial_cursor_without_staging_replays_after_upgrade(cfg):
    hy, llm = _initial(cfg)
    config = hy.config
    try:
        old = _state(hy)
        _rebuild(hy, llm)
        hy.conn.execute("DROP TRIGGER digest_staging_workspace_guard")
        hy.conn.execute("DROP TABLE digest_staging")
        hy.conn.execute("UPDATE schema_meta SET value='57' WHERE key='schema_version'")
        hy.close()
        hy = HyMem(config, llm=llm)
        assert db.schema_version(hy.conn) == db.EXPECTED_SCHEMA_VERSION
        hy.dream()
        assert "chars=0:" in llm.successful_digest_calls[-1].user
        assert _state(hy) == old
        _finish(hy)
        assert "newstaged" in _state(hy)["summary"]
    finally:
        hy.close()


@pytest.mark.parametrize("mutation", ["citation", "private_marker", "oversized", "cursor"])
def test_staging_rejects_invalid_payload_or_changed_cursor_before_advancing(cfg, monkeypatch, mutation):
    hy, llm = _initial(cfg, long=False)
    try:
        hy.log_message("x", "user", "beta " + "tail " * 150)
        original = runner.extract_session_digest

        def extract(*args, **kwargs):
            value = original(*args, **kwargs)
            if mutation == "citation":
                value.episodes.items[0]["chunk_ids"] = ["unowned-source"]
            elif mutation == "private_marker":
                value.episodes.items[0]["_source_citations_invalid"] = True
            elif mutation == "oversized":
                value.episodes.items[0]["summary"] = "x" * (digest._MAX_DIGEST_STAGE_JSON_BYTES + 1)
            else:
                hy.conn.execute("UPDATE sessions SET digest_cursor_offset=1 WHERE id='x'")
            return value

        monkeypatch.setattr(runner, "extract_session_digest", extract)
        # Capture after rebinding; semantic-generation fencing intentionally
        # schedules a full walk under this exact loaded implementation.
        old = _state(hy)
        with pytest.raises(RuntimeError, match="staging|source"):
            hy.dream()
        assert _state(hy) == old
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
    finally:
        hy.close()


def test_staging_prevents_workspace_adoption_and_cascades_on_session_delete(cfg):
    hy, llm = _initial(cfg)
    try:
        _rebuild(hy, llm)
        assert not _session_is_pristine(hy.conn, "x")
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute("UPDATE sessions SET source_workspace_id='other' WHERE id='x'")
        hy.conn.execute("INSERT INTO sessions(id) VALUES ('empty')")
        row = dict(hy.conn.execute("SELECT * FROM digest_staging LIMIT 1").fetchone())
        row["session_id"] = "empty"
        columns = ",".join(row)
        hy.conn.execute(f"INSERT INTO digest_staging({columns}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))
        assert not _session_is_pristine(hy.conn, "empty")
        with pytest.raises(sqlite3.IntegrityError):
            hy.conn.execute("UPDATE sessions SET source_workspace_id='other' WHERE id='empty'")
        hy.conn.execute("DELETE FROM sessions WHERE id='empty'")
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging WHERE session_id='empty'").fetchone()[0] == 0
    finally:
        hy.close()


def test_store_attestation_covers_staged_digest_content(cfg):
    from benchmarks.store_attestation import material_store_state
    hy, llm = _initial(cfg)
    try:
        _rebuild(hy, llm)
        before = material_store_state(hy.config.db_path)
        hy.conn.execute("UPDATE digest_staging SET summary='changed private summary'")
        after = material_store_state(hy.config.db_path)
        changed = {key for key in before["tables"]
                   if before["tables"][key] != after["tables"][key]}
        assert changed == {"digest_staging"}
        assert before["sha256"] != after["sha256"]
    finally:
        hy.close()


def test_replacement_publication_reconstructs_episodes_from_validated_stage(cfg):
    hy, llm = _initial(cfg)
    try:
        old = _state(hy)
        _rebuild(hy, llm)
        generation = hy.conn.execute("SELECT generation FROM digest_staging").fetchone()[0]
        with db.transaction(hy.conn):
            hy.conn.execute("DELETE FROM episodes WHERE digest_generation=?", (generation,))
            hy.conn.execute("INSERT INTO episodes(id,session_id,title,summary,digest_generation) "
                            "VALUES ('injected','x','Injected episode','Unproved staging.',?)", (generation,))
        assert _state(hy) == old
        _finish(hy)
        assert hy.conn.execute("SELECT 1 FROM episodes WHERE id='injected'").fetchone() is None
        assert any(row[1] == "newstaged Slice 0" for row in _state(hy)["episodes"])
    finally:
        hy.close()
