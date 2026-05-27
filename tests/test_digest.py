"""Tests for the batched per-session digest (the dream cycle's tail).

  * One LLM call produces episodes + summary + procedures together, replacing
    the three separate per-session calls.
  * The skip-guard (``sessions.digested_prompt_version``) suppresses the digest
    call entirely on a re-dream of an unchanged session — the cost win for
    steady-state dreaming over many already-processed sessions.
  * A prompt_version bump (new extraction contract) forces a re-digest.
"""

from __future__ import annotations

import json

from hymem import HyMem
from hymem.dreaming.digest import extract_session_digest
from hymem.extraction.llm import StubLLMClient


# --- helpers ---------------------------------------------------------------


def _digest_llm(
    *,
    episodes: list[dict] | None = None,
    summary: str = "",
    procedures: list[dict] | None = None,
) -> StubLLMClient:
    """Stub returning one combined digest object for the batched call, and an
    empty array for triple/marker chunk calls. Keyed on the digest user-prompt
    closer ``Return the JSON object now``."""
    payload = {
        "episodes": episodes or [],
        "summary": summary,
        "procedures": procedures or [],
    }
    return StubLLMClient(
        fixtures={"Return the JSON object now": json.dumps(payload)},
        default="[]",
    )


def _digest_calls(llm: StubLLMClient) -> list:
    """The subset of recorded calls that hit the batched digest prompt."""
    return [c for c in llm.calls if "Return the JSON object now" in c.user]


def _seed_session(hy: HyMem, sid: str, turns: list[tuple[str, str]]) -> None:
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)


_TURNS = [
    ("assistant", "how do we deploy to staging?"),
    ("user", "Build the docker image then kubectl apply the staging manifests; that ships it."),
]


# --- one call produces all three ------------------------------------------


def test_single_digest_call_persists_episodes_summary_procedures(cfg):
    """A single batched call writes episodes, the session summary, and
    procedures — and the digest prompt is hit exactly once."""
    episode = {
        "title": "Staging deploy",
        "summary": "Walked through shipping a build to staging.",
        "outcome": "resolved",
        "key_entities": ["docker", "kubernetes"],
        "chunk_ids": [],
    }
    procedure = {
        "name": "Deploy to staging",
        "description": "Ship the current build to the staging cluster.",
        "steps": [
            {"order": 1, "action": "build the docker image", "tool": "docker"},
            {"order": 2, "action": "kubectl apply the staging manifests", "tool": "kubectl"},
        ],
        "triggers": ["deploy staging"],
        "entities_involved": ["docker", "kubernetes"],
    }
    summary = "Documented the staging deploy: build the docker image and kubectl apply the manifests."

    llm = _digest_llm(episodes=[episode], summary=summary, procedures=[procedure])
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_digest"
        _seed_session(hy, sid, _TURNS)
        hy.dream()

        ep = hy.conn.execute(
            "SELECT title FROM episodes WHERE session_id = ?", (sid,)
        ).fetchall()
        assert [r["title"] for r in ep] == ["Staging deploy"]

        srow = hy.conn.execute(
            "SELECT summary FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert srow["summary"] == summary

        pr = hy.conn.execute(
            "SELECT name FROM procedures WHERE session_id = ?", (sid,)
        ).fetchall()
        assert [r["name"] for r in pr] == ["Deploy to staging"]

        # The whole tail cost exactly one LLM call.
        assert len(_digest_calls(llm)) == 1
    finally:
        hy.close()


# --- skip-guard: unchanged re-dream makes no tail call --------------------


def test_redream_unchanged_session_skips_digest_call(cfg):
    """A second dream over a session with no new chunks and a matching
    digested_prompt_version must not issue another digest call."""
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_skip"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        after_first = len(_digest_calls(llm))
        assert after_first == 1

        marker = hy.conn.execute(
            "SELECT digested_prompt_version FROM sessions WHERE id = ?", (sid,)
        ).fetchone()["digested_prompt_version"]
        assert marker == hy.config.prompt_version

        hy.dream()  # nothing changed
        assert len(_digest_calls(llm)) == after_first, "re-dream must skip the digest"
    finally:
        hy.close()


# --- prompt_version bump forces a re-digest --------------------------------


def test_prompt_version_bump_forces_redigest(cfg):
    """A new prompt_version invalidates the marker (and re-extracts chunks),
    so the digest runs again. Config is frozen, so this opens a second HyMem
    over the same DB with a bumped version."""
    import dataclasses

    sid = "s_bump"
    llm1 = _digest_llm(summary="A short but valid session summary about deploys.")
    hy1 = HyMem(cfg, llm=llm1)
    try:
        _seed_session(hy1, sid, _TURNS)
        hy1.dream()
        assert len(_digest_calls(llm1)) == 1
    finally:
        hy1.close()

    cfg2 = dataclasses.replace(cfg, prompt_version=cfg.prompt_version + "x")
    llm2 = _digest_llm(summary="A short but valid session summary about deploys.")
    hy2 = HyMem(cfg2, llm=llm2)
    try:
        hy2.dream()
        assert len(_digest_calls(llm2)) == 1, "version bump must re-run the digest"
    finally:
        hy2.close()


# --- malformed payloads degrade gracefully ---------------------------------


def test_non_object_digest_payload_yields_empty(cfg):
    """If the LLM returns a bare array (or other non-object), the digest is
    empty rather than raising — nothing is persisted, marker still set."""
    llm = StubLLMClient(
        fixtures={"Return the JSON object now": "[]"},
        default="[]",
    )
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_bad"
        _seed_session(hy, sid, _TURNS)
        hy.dream()  # materialize chunks (digest returns empty for the "[]" payload)
        digest = extract_session_digest(
            hy.conn, sid, hy._llm,
            max_tokens=hy.config.dream_digest_max_tokens,
            max_chars=hy.config.dream_digest_max_chars,
        )
        assert digest is not None
        assert digest.episodes.items == []
        assert digest.summary is None
        assert digest.procedures.items == []
    finally:
        hy.close()
