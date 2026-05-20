"""Tests for the per-session summarization layer:

  * `extract_session_summary` skips sessions that already have a summary
    and rejects suspiciously short LLM outputs.
  * `persist_session_summary` writes to `sessions.summary` so the
    Honcho context endpoint can prefer it over the MEMORY.md dump.
  * Quote-wrapped LLM output is stripped; long output is truncated to
    500 characters.
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
    """Stub that returns ``summary`` for the session-summary prompt and an
    empty array for any other extraction call. Substring keyed on the unique
    opener of `SESSION_SUMMARY_SYSTEM`."""
    return StubLLMClient(
        fixtures={"one-sentence summary": summary},
        default="[]",
    )


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
    """The summarizer rejects outputs that are too short to be useful
    (after stripping whitespace and surrounding quotes). After a dream
    pass the sessions row carries no summary."""
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


def test_long_summary_is_truncated(cfg):
    """The summarizer caps the persisted summary at 500 chars so an
    overlong LLM response doesn't poison the context endpoint."""
    long = "a" * 1200
    hy = HyMem(cfg, llm=_summary_llm(long))
    try:
        sid = "s_long"
        _seed_session(hy, sid, [
            ("assistant", "ok"),
            ("user", "A user turn long enough to clear the salience minimum threshold for chunking."),
        ])
        hy.dream()
        row = hy.conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert row["summary"] is not None
        assert len(row["summary"]) == 500
        assert set(row["summary"]) == {"a"}
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
        hy.open_session(sid)
        hy.log_message(sid, "user", "hello there")
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
