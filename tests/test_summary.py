"""Tests for the per-session summarization layer:

  * `extract_session_summary` skips sessions that already have a summary
    and rejects suspiciously short LLM outputs.
  * `persist_session_summary` writes to `sessions.summary` so the
    Honcho context endpoint can prefer it over the MEMORY.md dump.
  * Quote-wrapped LLM output is stripped; over-cap output is held for retry
  without advancing the durable digest cursor.
"""

from __future__ import annotations

import pytest

from hymem import HyMem
from hymem.dreaming.summary import (
    extract_session_summary,
    persist_session_summary,
)
from hymem.extraction.llm import StubLLMClient


# --- helpers ---------------------------------------------------------------


def _summary_llm(summary: str) -> StubLLMClient:
    """Stub that returns ``summary`` as the summary section of the batched
    session-digest response (episodes/procedures empty), and an empty array for
    any other extraction call. Keyed on the unique digest user-prompt closer
    ``Return the JSON object now``."""
    import json

    digest = {"episodes": [], "summary": summary, "procedures": []}
    return StubLLMClient(
        fixtures={"Return the JSON object now": json.dumps(digest)},
        default="[]",
    )


class _SequencedSummaryLLM:
    def __init__(self, summaries: list[str]):
        self.summaries = list(summaries)
        self.calls = []

    def complete(self, request) -> str:
        import json

        self.calls.append(request)
        if "Return the JSON object now" not in request.user:
            return "[]"
        assert self.summaries, "unexpected extra digest retry"
        return json.dumps({
            "episodes": [],
            "summary": self.summaries.pop(0),
            "procedures": [],
        })


def _seed_session(hy: HyMem, sid: str, turns: list[tuple[str, str]]) -> None:
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)


# --- happy path ------------------------------------------------------------


def test_session_summary_persisted_after_dream(cfg):
    """End-to-end: dream() runs the summarizer and writes
    ``sessions.summary``."""
    expected = (
        "Diagnosed Postgres connection-pool exhaustion and resolved it by "
        "bumping pool_size and adding health checks."
    )
    hy = HyMem(cfg, llm=_summary_llm(expected))
    try:
        sid = "s_happy"
        _seed_session(hy, sid, [
            ("assistant", "what broke last night?"),
            ("user", "Postgres pool exhaustion on prod — bumping pool_size and adding health checks fixed it."),
        ])
        hy.dream()

        row = hy.conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert row is not None
        assert row["summary"] == expected
    finally:
        hy.close()


# --- idempotency / no rework when summary already exists -------------------


def test_existing_summary_is_not_overwritten(cfg, stub_llm):
    """If a session already has a summary, extract_session_summary returns
    None and does not call the LLM. This protects against re-dreams
    overwriting an operator-curated summary."""
    hy = HyMem(cfg, llm=stub_llm)
    try:
        sid = "s_existing"
        _seed_session(hy, sid, [
            ("assistant", "kickoff turn"),
            ("user", "We talked about something here long enough to clear the salience minimum threshold."),
        ])
        preset = "operator-authored summary that must survive"
        with hy.conn:
            hy.conn.execute(
                "UPDATE sessions SET summary = ? WHERE id = ?",
                (preset, sid),
            )

        result = extract_session_summary(hy.conn, sid, hy._llm)
        assert result is None, "must short-circuit when summary already set"
        assert stub_llm.calls == [], "LLM must not be called on this path"

        row = hy.conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert row["summary"] == preset
    finally:
        hy.close()


# --- validation: short / quoted / long -------------------------------------


@pytest.mark.parametrize("raw", ["", "tiny", "  short  "])
def test_short_llm_output_rejected(cfg, raw):
    """Only the explicit empty digest no-op advances; short prose is invalid."""
    hy = HyMem(cfg, llm=_summary_llm(raw))
    try:
        sid = "s_short"
        _seed_session(hy, sid, [
            ("assistant", "ok"),
            ("user", "A user turn long enough to be chunked as a salient message in this test."),
        ])
        hy.dream()
        row = hy.conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        if raw == "":
            assert row["summary"] == ""
        else:
            assert row["summary"] is None
    finally:
        hy.close()


def test_quoted_summary_is_unwrapped(cfg):
    """LLMs sometimes return summaries wrapped in quotes despite the prompt
    forbidding them; the summarizer strips leading and trailing quotes
    before persisting."""
    quoted = '"User reviewed the deployment runbook and renamed the staging step."'
    hy = HyMem(cfg, llm=_summary_llm(quoted))
    try:
        sid = "s_quoted"
        _seed_session(hy, sid, [
            ("assistant", "what did we touch in the runbook?"),
            ("user", "We renamed the staging step in the deployment runbook to be more explicit."),
        ])
        hy.dream()
        row = hy.conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert row["summary"] is not None
        assert not row["summary"].startswith('"')
        assert not row["summary"].endswith('"')
        assert "deployment runbook" in row["summary"]
    finally:
        hy.close()


def test_long_summary_holds_cursor_and_heals_on_retry(cfg):
    """An over-cap response is a failed outcome, never silent truncation."""
    long = "a" * 1200
    healed = "Reviewed the deployment runbook and recorded the exact outcome."
    llm = _SequencedSummaryLLM([long, healed])
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_long"
        _seed_session(hy, sid, [
            ("assistant", "ok"),
            ("user", "A user turn long enough to clear the salience minimum threshold for chunking."),
        ])
        failed = hy.dream()
        row = hy.conn.execute(
            "SELECT summary, digest_cursor_message_id, digest_retry_count, "
            "digest_quarantined FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert row["summary"] is None
        assert row["digest_cursor_message_id"] is None
        assert row["digest_retry_count"] == 1
        assert row["digest_quarantined"] == 0
        assert failed.digest_failures == 1
        assert failed.budget_exhausted is True

        succeeded = hy.dream()
        healed_row = hy.conn.execute(
            "SELECT summary, coverage_message_id, digest_cursor_message_id, "
            "digest_retry_count, digest_retry_config_version, digest_quarantined "
            "FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert healed_row["summary"] == healed
        assert healed_row["digest_cursor_message_id"] == healed_row["coverage_message_id"]
        assert healed_row["digest_retry_count"] == 0
        assert healed_row["digest_retry_config_version"] is None
        assert healed_row["digest_quarantined"] == 0
        assert succeeded.digest_failures == 0
    finally:
        hy.close()


# --- direct persist path ---------------------------------------------------


def test_persist_session_summary_updates_existing_row(cfg, stub_llm):
    """`persist_session_summary` is a plain UPDATE; verify it overwrites
    whatever was there (callers gate idempotency, not the writer)."""
    hy = HyMem(cfg, llm=stub_llm)
    try:
        sid = "s_persist"
        hy.open_session(sid)
        hy.close_session(sid)

        with hy.conn:
            persist_session_summary(hy.conn, sid, "first summary")
        first = hy.conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (sid,),
        ).fetchone()["summary"]
        assert first == "first summary"

        with hy.conn:
            persist_session_summary(hy.conn, sid, "second summary")
        second = hy.conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (sid,),
        ).fetchone()["summary"]
        assert second == "second summary"
    finally:
        hy.close()


# --- honcho /context surface ----------------------------------------------


def test_honcho_context_prefers_session_summary_over_memory_md(cfg, tmp_path):
    """When `sessions.summary` is set, the context endpoint returns it
    in the `summary` field instead of dumping MEMORY.md. This is the
    operational payoff of the summarizer."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import hymem.honcho.app as hsrv

    # Memory.md exists with placeholder content so we can distinguish which
    # branch the context endpoint took.
    cfg.memory_md_path.write_text("# memory dump\n- some marker\n", encoding="utf-8")

    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    try:
        sid = "s_ctx"
        hy.log_message(
            sid,
            "user",
            "hello there",
            source_peer_id="user",
            source_workspace_id="hermes",
        )
        with hy.conn:
            hy.conn.execute(
                "UPDATE sessions SET summary = ? WHERE id = ?",
                ("Discussed Postgres pool sizing and resolved the prod outage.", sid),
            )

        hsrv.set_hy(hy)
        if hsrv._scheduler is not None:
            hsrv._scheduler.stop()
            hsrv.set_scheduler(None)

        with TestClient(hsrv.app) as client:
            r = client.get(f"/v3/workspaces/hermes/sessions/{sid}/context")
            assert r.status_code == 200
            body = r.json()
            assert body["summary"] is not None
            assert "Postgres pool sizing" in body["summary"]["content"]
            # MEMORY.md marker must NOT have leaked into the summary payload
            # — the session summary wins.
            assert "some marker" not in body["summary"]["content"]
    finally:
        hy.close()
