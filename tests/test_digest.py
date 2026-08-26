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
    """The subset of recorded calls that hit the batched SESSION digest prompt.

    Discriminated on the system prompt, not on the "Return the JSON object now"
    user-tail marker: that marker is shared by the RAPTOR ROOT digest fusion
    ("You write the standing digest ..."), which became a default-on call when
    `aggregation_nodes_enabled` flipped True on 2026-08-26. These tests measure
    session-digest BATCHING (one tail call per session, not three); counting an
    unrelated subsystem's call against that bar reads as a batching regression
    that never happened.
    """
    return [
        c for c in llm.calls
        if "Return the JSON object now" in c.user
        and c.system.startswith("You analyze one conversation session")
    ]


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


# --- the digest watermark (schema v24) --------------------------------------


def _append_turns(hy: HyMem, sid: str, turns: list[tuple[str, str]]) -> None:
    """Add traffic to an already-closed session, the way a long-lived Hermes
    session grows between dreams."""
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)


_MORE_TURNS = [
    ("assistant", "and how do we roll back a bad staging release?"),
    ("user", "Run helm rollback staging to the previous revision; that reverts it."),
]


def test_new_traffic_in_digested_session_reopens_the_digest(cfg):
    """The regression this fixes: messages landing in an ALREADY-digested
    session must re-open the digest. Before the watermark the guard only asked
    `had_new_chunk_work`, so traffic whose chunks were already processed left
    the session skipped forever and its episodes froze (2026-07-30: 184
    messages, zero episodes, six days)."""
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_append"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        after_first = len(_digest_calls(llm))
        assert after_first == 1

        hy.dream()  # unchanged: still skipped, the cost win must survive
        assert len(_digest_calls(llm)) == after_first

        _append_turns(hy, sid, _MORE_TURNS)
        hy.dream()
        assert len(_digest_calls(llm)) == after_first + 1, (
            "new traffic in a digested session must re-open the digest"
        )
    finally:
        hy.close()


def test_watermark_advances_and_restricts_input_to_the_tail(cfg):
    """The watermark tracks the highest message actually covered, and the next
    digest sees only chunks above it — the fix for `combined[:max_chars]`
    keeping the OLDEST slice of a growing session."""
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_tail"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        first = hy.conn.execute(
            "SELECT digested_message_id FROM sessions WHERE id = ?", (sid,)
        ).fetchone()["digested_message_id"]
        assert first is not None and first > 0

        _append_turns(hy, sid, _MORE_TURNS)
        hy.dream()
        second = hy.conn.execute(
            "SELECT digested_message_id FROM sessions WHERE id = ?", (sid,)
        ).fetchone()["digested_message_id"]
        assert second > first, "watermark must advance over the new tail"

        # The tail digest must not re-send the head: the last digest call sees
        # the rollback turn and not the original deploy turn.
        last_call = _digest_calls(llm)[-1].user
        assert "helm rollback" in last_call
        assert "kubectl apply the staging manifests" not in last_call
    finally:
        hy.close()


def test_session_larger_than_max_chars_walks_forward_across_dreams(cfg):
    """The original defect, reproduced at the size that caused it: a session
    whose chunks exceed `dream_digest_max_chars`. The old code joined every
    chunk and kept `combined[:max_chars]` — the OLDEST slice — so the tail was
    unreachable forever. Successive dreams must now cover strictly more of the
    session, and the final digest must reach turns the first one could not see."""
    import dataclasses

    # Small cap so a handful of ordinary turns overflows it several times over.
    small = dataclasses.replace(cfg, dream_digest_max_chars=400)
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(small, llm=llm)
    try:
        sid = "s_big"
        turns: list[tuple[str, str]] = []
        for i in range(12):
            turns.append(("assistant", f"question {i}: what do we do in step {i}?"))
            turns.append((
                "user",
                f"For step {i} you run the step-{i} playbook end to end; "
                f"marker-{i} confirms it finished cleanly.",
            ))
        _seed_session(hy, sid, turns)

        newest = hy.conn.execute(
            "SELECT MAX(id) AS m FROM messages WHERE session_id = ?", (sid,)
        ).fetchone()["m"]

        marks: list[int] = []
        call_counts: list[int] = []
        for _ in range(10):
            hy.dream()
            marks.append(hy.conn.execute(
                "SELECT digested_message_id FROM sessions WHERE id = ?", (sid,)
            ).fetchone()["digested_message_id"])
            call_counts.append(len(_digest_calls(llm)))

        assert marks[0] is not None
        assert marks == sorted(marks), "coverage must never go backwards"
        assert marks[-1] == newest, (
            "coverage must reach the end of the session — with the old "
            "combined[:max_chars] the tail was unreachable at any dream count"
        )
        # ...and once caught up, the digest stops costing calls.
        assert call_counts[-1] == call_counts[-2] == call_counts[-3]

        # The head is digested first and the tail later: no single call carries
        # both ends, and the tail turns do get read.
        calls = [c.user for c in _digest_calls(llm)]
        assert "marker-0" in calls[0]
        assert "marker-0" not in calls[-1]
        assert any("marker-11" in c for c in calls), "the tail must eventually be read"
    finally:
        hy.close()


def test_parse_failure_does_not_advance_the_watermark(cfg):
    """A malformed digest reply reports no coverage, so the slice is retried
    rather than silently skipped — advancing here would recreate the exact
    starvation mode migration 024 fixes."""
    llm = StubLLMClient(
        fixtures={"Return the JSON object now": "not json at all"},
        default="[]",
    )
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_badjson"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        mark = hy.conn.execute(
            "SELECT digested_message_id FROM sessions WHERE id = ?", (sid,)
        ).fetchone()["digested_message_id"]
        assert mark is None, "a parse failure must not claim coverage"

        hy.dream()
        assert len(_digest_calls(llm)) == 2, "the uncovered slice must be retried"
    finally:
        hy.close()


def test_since_message_id_selects_only_the_tail(cfg):
    """Unit-level: `since_message_id` restricts the chunk set, and a fully
    digested tail yields None instead of a pointless LLM call."""
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_since"
        _seed_session(hy, sid, _TURNS)
        hy.dream()

        newest = hy.conn.execute(
            "SELECT MAX(id) AS m FROM messages WHERE session_id = ?", (sid,)
        ).fetchone()["m"]
        assert extract_session_digest(
            hy.conn, sid, hy._llm,
            max_tokens=hy.config.dream_digest_max_tokens,
            max_chars=hy.config.dream_digest_max_chars,
            since_message_id=newest,
        ) is None, "nothing above the watermark means no call"
    finally:
        hy.close()


# --- an oversized head chunk must not pin the watermark ---------------------


def _oversized_turns(cap: int) -> list[tuple[str, str]]:
    """A session whose FIRST chunk alone overflows `cap`, followed by ordinary
    tail turns. The loop in extract_session_digest admits the head part without
    a size check, so this is the only shape that can reach the truncation
    branch."""
    filler = "the deploy playbook step repeats here to pad the turn. " * 40
    turns = [
        ("assistant", "head-marker: how do we deploy to staging?"),
        ("user", f"Build the image then apply the manifests. {filler} tail-marker."),
    ]
    for i in range(4):
        turns.append(("assistant", f"question {i}: what runs in step {i}?"))
        turns.append((
            "user",
            f"For step {i} you run the step-{i} playbook end to end; "
            f"marker-{i} confirms it finished cleanly.",
        ))
    assert len(filler) > cap, "the head turn must overflow the cap on its own"
    return turns


def test_oversized_head_chunk_advances_the_watermark(cfg):
    """A single chunk longer than the whole cap must be truncated AND claim
    coverage. Holding the watermark here re-sent the identical slice on every
    dream forever: the session's episodes froze while its messages kept
    arriving, and the repeated digest upserted onto the same deterministic
    episode ids, so the store never grew (2026-08-05, watermark 547 vs message
    2093). Truncation keeps the HEAD, so no later dream could read more of this
    chunk than this one did — claiming coverage loses nothing a retry recovers.
    Mirrors facts.oversized_turn in dreaming/facts.py."""
    import dataclasses

    small = dataclasses.replace(cfg, dream_digest_max_chars=400)
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(small, llm=llm)
    try:
        sid = "s_oversized"
        _seed_session(hy, sid, _oversized_turns(400))

        newest = hy.conn.execute(
            "SELECT MAX(id) AS m FROM messages WHERE session_id = ?", (sid,)
        ).fetchone()["m"]

        marks: list[int | None] = []
        for _ in range(10):
            hy.dream()
            marks.append(hy.conn.execute(
                "SELECT digested_message_id FROM sessions WHERE id = ?", (sid,)
            ).fetchone()["digested_message_id"])

        assert marks[0] is not None, (
            "the oversized head chunk must claim coverage — reporting None pins "
            "the watermark and re-sends this slice on every dream forever"
        )
        assert marks == sorted(marks), "coverage must never go backwards"
        assert marks[-1] == newest, "the tail beyond the oversized chunk must be reached"

        calls = [c.user for c in _digest_calls(llm)]
        assert "head-marker" in calls[0], "the head of the oversized chunk is kept"
        assert "tail-marker" not in calls[0], "its tail is truncated at the cap"
        assert any("marker-3" in c for c in calls), "later turns must still be read"
    finally:
        hy.close()


def test_oversized_head_chunk_is_logged(cfg, caplog):
    """The truncation drops text, so it must be audible rather than silent —
    the pre-fix branch reported no coverage with no log at all, which is what
    made the stall invisible for five days."""
    import dataclasses
    import logging

    small = dataclasses.replace(cfg, dream_digest_max_chars=400)
    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(small, llm=llm)
    try:
        sid = "s_oversized_log"
        _seed_session(hy, sid, _oversized_turns(400))
        with caplog.at_level(logging.WARNING, logger="hymem.dreaming.digest"):
            hy.dream()
        assert "digest.oversized_chunk" in caplog.text
        assert "cap=400" in caplog.text
        assert "(tail not read)" in caplog.text
        # The message id is the coverage now being claimed, not None.
        assert "message_id=None" not in caplog.text
    finally:
        hy.close()


def test_normal_session_logs_no_oversized_warning(cfg, caplog):
    """The warning must stay quiet on ordinary traffic — a truncation notice on
    every dream would be noise, and noise is how the real one gets missed."""
    import logging

    llm = _digest_llm(summary="A short but valid session summary about deploys.")
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_normal"
        _seed_session(hy, sid, _TURNS)
        with caplog.at_level(logging.WARNING, logger="hymem.dreaming.digest"):
            hy.dream()
        assert "digest.oversized_chunk" not in caplog.text
    finally:
        hy.close()


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


def test_digest_parses_fenced_reply(cfg):
    """Dream 1013 verbatim: a complete, valid payload wrapped in ```json
    fences from a call that already had json_object mode set. It used to be
    dropped, taking episodes + summary + procedures with it and stalling the
    watermark for a formatting quirk."""
    payload = {
        "episodes": [{"title": "Deploy chat", "summary": "How to ship to staging.",
                      "chunk_ids": []}],
        "summary": "A short but valid session summary about deploys.",
        "procedures": [{"name": "Deploy to staging",
                        "steps": [{"order": 1, "action": "build the image"}]}],
    }
    fenced = "Here is the JSON:\n```json\n" + json.dumps(payload) + "\n```\nHope that helps!"
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    try:
        sid = "s_fenced_digest"
        _seed_session(hy, sid, _TURNS)
        hy.dream()  # materialize chunks
        hy.set_llm(StubLLMClient(
            fixtures={"Return the JSON object now": fenced}, default="[]",
        ))
        digest = extract_session_digest(
            hy.conn, sid, hy._llm,
            max_tokens=hy.config.dream_digest_max_tokens,
            max_chars=hy.config.dream_digest_max_chars,
        )
        assert digest is not None
        assert digest.parse_failed is False
        assert [i["title"] for i in digest.episodes.items] == ["Deploy chat"]
        assert [i["name"] for i in digest.procedures.items] == ["Deploy to staging"]
        assert digest.summary == payload["summary"]
    finally:
        hy.close()


def test_digest_refusal_sets_parse_failed_and_logs(cfg, caplog):
    """An unparseable reply must still take the `_empty()` contract —
    parse_failed=True, covered_message_id=None so the watermark does not
    advance — and must now say so in the log."""
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    try:
        sid = "s_refusal_digest"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        hy.set_llm(StubLLMClient(
            fixtures={"Return the JSON object now": "I'm sorry, I can't help with that."},
            default="[]",
        ))
        with caplog.at_level("WARNING"):
            digest = extract_session_digest(
                hy.conn, sid, hy._llm,
                max_tokens=hy.config.dream_digest_max_tokens,
                max_chars=hy.config.dream_digest_max_chars,
            )
        assert digest is not None
        assert digest.parse_failed is True
        assert digest.covered_message_id is None
        assert digest.episodes.items == [] and digest.procedures.items == []
        assert digest.summary is None
        assert any(
            "digest.parse_failure" in r.message and sid in r.getMessage()
            for r in caplog.records
        )
    finally:
        hy.close()


def test_digest_wrong_shape_holds_the_watermark_and_is_audible(cfg, caplog):
    """Valid JSON of the wrong shape. _empty() sets parse_failed, which HOLDS
    the watermark, so a model that persistently returns this re-sends the same
    slice every dream forever — the log line is the only way that surfaces."""
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    try:
        sid = "s_shape_digest"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        hy.set_llm(StubLLMClient(
            fixtures={"Return the JSON object now": json.dumps(["not", "an", "object"])},
            default="[]",
        ))
        with caplog.at_level("WARNING"):
            digest = extract_session_digest(
                hy.conn, sid, hy._llm,
                max_tokens=hy.config.dream_digest_max_tokens,
                max_chars=hy.config.dream_digest_max_chars,
            )
        assert digest.parse_failed is True
        assert digest.covered_message_id is None
        assert any(
            "digest.shape_failure" in r.message and sid in r.getMessage()
            for r in caplog.records
        )
    finally:
        hy.close()


def test_digest_stays_quiet_on_the_stub_empty_array(cfg, caplog):
    """`[]` is StubLLMClient's documented default and a routine "nothing here".
    Warning on it would fire on every stub-configured dream — the no-LLM
    default this project ships — and drown the real signal."""
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    try:
        sid = "s_stub_digest"
        _seed_session(hy, sid, _TURNS)
        hy.dream()
        hy.set_llm(StubLLMClient(default="[]"))
        with caplog.at_level("WARNING"):
            caplog.clear()  # ignore anything the dream above logged
            extract_session_digest(
                hy.conn, sid, hy._llm,
                max_tokens=hy.config.dream_digest_max_tokens,
                max_chars=hy.config.dream_digest_max_chars,
            )
        assert not any("shape_failure" in r.message for r in caplog.records)
    finally:
        hy.close()
