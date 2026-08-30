"""Offline tests for the Plan C front-run probe (`benchmarks/episode_probe.py`).

No LLM, no network, no dataset file: synthetic fixtures and the `--sim` backend
throughout. Three things are pinned, and they are the three ways this particular
probe could lie:

  * **The verdict arithmetic.** A missing faithfulness hand-score must read
    INCOMPLETE, never PASS, however good the mechanical criteria look — "four of
    five criteria" is not the gate. And an EMPTY target arm must read INCOMPLETE
    too: every criterion of the form `median <= cap` or `share >= floor` is
    trivially satisfied at n=0, so a probe that extracted nothing would
    otherwise print the same banner as one that extracted well.
  * **The full-source guarantee.** `assert_full_source` must REJECT a dump whose
    recorded extractor input was sliced. This is the `[:4000]` defect that
    turned 50 faithful facts into apparent inventions on the E1 gate; here the
    recorded string is hashed at send time, so a slice is a hard failure rather
    than an invisible one.
  * **The plumbing**, end to end on the production digest path: sessions →
    HyMem store → production chunkers → `extract_session_digest` → validated
    episodes → criteria. The `--sim` arm exists precisely so this is exercisable
    before any spend.

The module is imported WITHOUT `requests` being available (episode_probe defers
its `longmemeval_adapter`/`fact_probe` imports into the functions that need
them), which is why this file is collectable where `test_fact_probe.py` is not.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from episode_probe import (  # noqa: E402
    _MAX_CONTROL_MEDIAN_EPISODES,
    _MIN_SUBSTANTIVE_CHARS,
    CapturingLLM,
    arm_stats,
    assert_full_source,
    build_faithfulness_sample,
    build_store,
    carries_concrete_value,
    extract_one,
    sim_backend,
    summarize,
)
from hymem.config import HyMemConfig  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────

def _row(arm: str, *, episodes: int = 5, chars: int = 4000,
         concrete: bool = True, stratum: str = "gold_bearing",
         calls: int = 1, parse_failed: bool = False) -> dict:
    """A synthetic dump row. `concrete=False` writes summaries with no number,
    date, version or path — the blob shape criterion 3 exists to reject."""
    text = ("Pinned pandas to 2.1.4 after the regression." if concrete
            else "Resolved a dependency issue and moved on.")
    return {
        "session_id": f"{arm}_{id(text)}_{episodes}_{chars}",
        "haystack_session_id": "h1",
        "question_id": "q1",
        "arm": arm,
        "stratum": stratum,
        "session_chars": chars,
        "episodes": [
            {"title": f"E{i}", "summary": text, "outcome": "resolved",
             "key_entities": [], "chunk_ids": []}
            for i in range(episodes)
        ],
        "parse_failed": parse_failed,
        "calls": calls,
        "extractor_input": "chunk text",
        "extractor_input_sha256": hashlib.sha256(b"chunk text").hexdigest(),
        "extractor_input_chars": len("chunk text"),
        "error": None,
    }


def _passing_arms() -> tuple[list[dict], list[dict]]:
    """Rows that satisfy every MECHANICAL criterion: median 5 episodes on
    substantive sessions, concrete values, full coverage, a control arm well
    under the over-extraction ceiling."""
    target = [_row("target") for _ in range(4)]
    control = [_row("control", episodes=4) for _ in range(4)]
    # Distinct ids so nothing collapses silently.
    for i, r in enumerate(target + control):
        r["session_id"] = f"{r['arm']}_{i}"
    return target, control


# ── the verdict arithmetic ──────────────────────────────────────────────────

def test_missing_faithfulness_reads_incomplete_not_pass():
    """The property this probe is built around: the hand-read is a criterion,
    not a footnote. All mechanical criteria pass here — asserted explicitly, so
    the INCOMPLETE below cannot be coming from a mechanical failure."""
    target, control = _passing_arms()
    s = summarize(target, control, None)
    mechanical = {k: v for k, v in s["gate"].items() if k != "faithfulness_ok"}
    assert all(mechanical.values()), mechanical
    assert s["gate"]["faithfulness_ok"] is False
    assert s["verdict"].startswith("INCOMPLETE")
    assert "PASS" not in s["verdict"]


def test_full_gate_passes_only_with_a_supplied_hand_score():
    target, control = _passing_arms()
    assert summarize(target, control, 0.95)["verdict"] == "PASS"
    # ...and a hand-score BELOW the bar is a FAIL, not an INCOMPLETE: the score
    # was read, it just did not clear.
    assert summarize(target, control, 0.80)["verdict"] == "FAIL"


def test_empty_target_arm_cannot_pass():
    """The vacuity guard. With no target sessions, `median <= cap` and
    `share >= floor` are both satisfied by an arm that extracted NOTHING; a gate
    that read them would report a clean pass on a broken run."""
    s = summarize([], [_row("control")], 1.0)
    assert s["verdict"].startswith("INCOMPLETE")
    assert s["target"]["n_sessions"] == 0


def test_blob_shaped_cut_fails_granularity_and_concreteness():
    """A re-cut that produces one value-free blob per session must FAIL even at
    perfect faithfulness — faithfulness alone is satisfied by copying the old
    prompt's output."""
    target = [_row("target", episodes=1, concrete=False) for _ in range(4)]
    control = [_row("control", episodes=1, concrete=False) for _ in range(4)]
    for i, r in enumerate(target + control):
        r["session_id"] = f"{r['arm']}_{i}"
    s = summarize(target, control, 1.0)
    assert s["gate"]["granularity_ok"] is False
    assert s["gate"]["concrete_ok"] is False
    assert s["verdict"] == "FAIL"


def test_runaway_extraction_fails_the_upper_granularity_bound():
    """3-8 is an interval, not a floor: one episode per turn is a different
    failure from one blob per session, and the gate must catch both."""
    target = [_row("target", episodes=30) for _ in range(4)]
    control = [_row("control", episodes=30) for _ in range(4)]
    for i, r in enumerate(target + control):
        r["session_id"] = f"{r['arm']}_{i}"
    s = summarize(target, control, 1.0)
    assert s["gate"]["granularity_ok"] is False
    assert s["gate"]["control_ok"] is False, (
        f"30 episodes/session must breach the control ceiling of "
        f"{_MAX_CONTROL_MEDIAN_EPISODES}"
    )
    assert s["verdict"] == "FAIL"


def test_parse_failure_ceiling_reads_incomplete_never_fail():
    """Pre-registered: truncation biases the criteria in OPPOSITE directions, so
    a truncation-heavy run looks like a sparse, clean, honest FAIL. It must be
    reported as unreadable instead."""
    target = [_row("target", parse_failed=True, episodes=0) for _ in range(4)]
    control = [_row("control") for _ in range(4)]
    for i, r in enumerate(target + control):
        r["session_id"] = f"{r['arm']}_{i}"
    s = summarize(target, control, 1.0)
    assert s["parse_failure_rate"] > 0.02
    assert s["verdict"].startswith("INCOMPLETE")
    assert "FAIL" not in s["verdict"]


def test_thin_sessions_are_excluded_from_granularity_but_still_counted():
    """LME haystacks are mostly padding. A thin session yielding no episodes is
    not a coverage failure, but its episodes still count toward concreteness —
    over-extraction on filler competes for the same retrieval slots."""
    thin = _row("target", episodes=0, chars=_MIN_SUBSTANTIVE_CHARS - 1)
    thick = _row("target", episodes=5, chars=_MIN_SUBSTANTIVE_CHARS + 1)
    stats = arm_stats([thin, thick])
    assert stats["n_substantive"] == 1 and stats["n_thin"] == 1
    assert stats["coverage"] == 1.0, "the thin session must not drag coverage down"
    assert stats["median_episodes"] == 5.0
    assert stats["episodes"] == 5


# ── the full-source guarantee ───────────────────────────────────────────────

def test_assert_full_source_accepts_an_intact_row():
    row = _row("target")
    assert_full_source(row)  # must not raise


def test_assert_full_source_rejects_a_truncated_source():
    """The [:4000] defect, reproduced: the recorded source is sliced after the
    call. The hash was taken at send time, so the slice cannot hide."""
    row = _row("target")
    row["extractor_input"] = row["extractor_input"][:4]
    with pytest.raises(AssertionError, match="TRUNCATED"):
        assert_full_source(row)


def test_assert_full_source_rejects_a_rewritten_source():
    """Same length, different content — a re-render rather than a slice. The
    length check alone would pass it; the hash is what catches it."""
    row = _row("target")
    row["extractor_input"] = "x" * row["extractor_input_chars"]
    with pytest.raises(AssertionError, match="does not hash"):
        assert_full_source(row)


# ── plumbing, end to end on the production digest path ─────────────────────

def _long_turn(i: int) -> str:
    return (
        f"For step {i} we pinned the service to version 2.{i}.4 and moved the "
        f"deploy to fly.io; the run took {i * 3} minutes and the error "
        f"'ECONNRESET at pool.py:{i}' stopped appearing afterwards."
    )


def _entry(session_id: str, arm: str, turns: int = 6) -> dict:
    messages = []
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"what about step {i}?"})
        messages.append({"role": "user", "content": _long_turn(i)})
    return {
        "session_id": session_id,
        "haystack_session_id": session_id,
        "question_id": "q1",
        "arm": arm,
        "stratum": "gold_bearing",
        "messages": messages,
    }


def test_sim_pipeline_extracts_through_the_production_digest(tmp_path):
    """Selection → store → production chunkers → `extract_session_digest` →
    validated episodes. The assertions are about the INPUT as much as the
    output: the recorded source must be the chunk-tagged digest prompt (the
    shape the feature actually sees), not a re-render of the raw turns."""
    cfg = HyMemConfig(root=tmp_path)
    entries = [_entry("s1", "target"), _entry("s2", "control")]
    conn = build_store(tmp_path / "probe.sqlite", entries, cfg)
    try:
        chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        assert chunks > 0, "precondition: the chunkers must have produced input"

        rows = []
        for entry in entries:
            llm = CapturingLLM(sim_backend)
            row = extract_one(conn, entry, llm, cfg, granular=True)
            assert_full_source(row)
            rows.append(row)
    finally:
        conn.close()

    assert all(r["error"] is None for r in rows), [r["error"] for r in rows]
    assert all(r["calls"] == 1 for r in rows), "one digest call per session"
    assert all(r["episodes"] for r in rows), "the sim backend must yield episodes"
    # The gated input shape: chunk-tagged, and carrying the granular closer
    # rather than the shipping one.
    src = rows[0]["extractor_input"]
    assert "[chunk chk_" in src
    assert src.rstrip().endswith("Return the granular digest JSON object now.")
    # And the hash-verified length is the real one, not a rendering's.
    assert rows[0]["extractor_input_chars"] == len(src)


def test_blob_arm_sends_the_shipping_prompt(tmp_path):
    """`--prompt-arm blob` is the granularity contrast's baseline, so it must
    genuinely run the SHIPPING prompt over the same sessions."""
    cfg = HyMemConfig(root=tmp_path)
    entries = [_entry("s1", "target")]
    conn = build_store(tmp_path / "probe.sqlite", entries, cfg)
    try:
        llm = CapturingLLM(sim_backend)
        row = extract_one(conn, entries[0], llm, cfg, granular=False)
    finally:
        conn.close()
    assert row["extractor_input"].rstrip().endswith("Return the JSON object now.")
    assert "granular" not in llm.sent[-1]["system"][:80].lower()


def test_capturing_llm_records_what_was_sent_not_what_was_asked_for(tmp_path):
    """The capture happens INSIDE the client, so nothing between the digest and
    the dump can substitute a shorter string for the prompt."""
    seen = {}

    def backend(system: str, user: str) -> str:
        seen["user"] = user
        return json.dumps({"episodes": [], "summary": "", "procedures": []})

    cfg = HyMemConfig(root=tmp_path)
    entries = [_entry("s1", "target")]
    conn = build_store(tmp_path / "probe.sqlite", entries, cfg)
    try:
        llm = CapturingLLM(backend)
        row = extract_one(conn, entries[0], llm, cfg, granular=True)
    finally:
        conn.close()
    assert row["extractor_input"] == seen["user"]
    assert row["extractor_input_sha256"] == hashlib.sha256(
        seen["user"].encode("utf-8")).hexdigest()


def test_backend_failure_is_a_parse_failure_not_a_crash(tmp_path):
    """A probe row must never abort the run — the failure has to surface as a
    counted parse failure so the pre-registered ceiling can read it."""
    def backend(system: str, user: str) -> str:
        raise RuntimeError("upstream 500")

    cfg = HyMemConfig(root=tmp_path)
    entries = [_entry("s1", "target")]
    conn = build_store(tmp_path / "probe.sqlite", entries, cfg)
    try:
        llm = CapturingLLM(backend)
        row = extract_one(conn, entries[0], llm, cfg, granular=True)
    finally:
        conn.close()
    assert row["parse_failed"] is True
    assert row["episodes"] == []
    assert llm.last_error and "upstream 500" in llm.last_error


# ── smaller units ───────────────────────────────────────────────────────────

def test_concrete_value_detector_is_conservative():
    assert carries_concrete_value("Pinned pandas to 2.1.4")
    assert carries_concrete_value("Moved the deploy to fly.io")
    assert carries_concrete_value("Patched pool.py")
    # Bare proper nouns do NOT count: a capitalized word is weak evidence of a
    # concrete value, and over-counting would let a blob-shaped cut through.
    assert not carries_concrete_value("Discussed the deployment with Atta")
    assert not carries_concrete_value("Resolved a dependency issue")


def test_faithfulness_sample_prefers_gold_bearing_sessions():
    """LongMemEval pads each haystack ~10:1 with UltraChat/ShareGPT filler, so a
    uniform sample would hand-score generic chat instead of the dated, numeric,
    name-bearing material a verbatim-value error actually costs an answer on."""
    rows = [_row("target", stratum="gold_bearing") for _ in range(3)]
    rows += [_row("target", stratum="distractor") for _ in range(30)]
    for i, r in enumerate(rows):
        r["session_id"] = f"s{i}"
    sample = build_faithfulness_sample(rows, size=8, seed=0)
    assert len(sample) == 8
    assert sum(1 for e in sample if e["stratum"] == "gold_bearing") == 3, (
        "every available gold-bearing session must be in the sample; the "
        "unfilled gold budget rolls over to filler"
    )
    # The hand-read must be self-contained: episodes AND their complete source.
    assert all(e["extractor_input"] for e in sample)


def test_sample_rows_carry_no_error_and_at_least_one_episode():
    """A row with no episodes has nothing to hand-score; including it would
    inflate the sample size while measuring nothing."""
    rows = [_row("target"), _row("target", episodes=0)]
    rows[1]["session_id"] = "empty"
    sample = build_faithfulness_sample(rows, size=8, seed=0)
    assert [e["session_id"] for e in sample] == [rows[0]["session_id"]]


# ── the reply is recorded, so a parse failure names its own cause ───────────

def test_an_empty_reply_is_recorded_as_empty_not_inferred_as_truncation(tmp_path):
    """The instrument gap the first real G-EP1 invocation exposed.

    A reasoning model with thinking left enabled burns the whole output budget
    and returns "" — content_len 0, finish_reason length. The probe discarded
    the reply the moment it failed to parse, so a 52.5% failure rate was
    attributed to TRUNCATION by inference and the emitted remedy (raise
    --max-tokens) could not have worked. The row must carry the true reply
    length, which is the one number that separates the two causes.
    """
    def backend(system: str, user: str) -> str:
        return ""

    cfg = HyMemConfig(root=tmp_path)
    entries = [_entry("s1", "target")]
    conn = build_store(tmp_path / "probe.sqlite", entries, cfg)
    try:
        row = extract_one(conn, entries[0], CapturingLLM(backend), cfg,
                          granular=True)
    finally:
        conn.close()
    assert row["parse_failed"] is True
    assert row["reply_chars"] == 0          # empty, NOT unrecorded
    assert row["reply_head"] == ""
    # And a non-empty reply that still fails to parse must NOT read as empty,
    # or the two causes collapse back into one and the remedy is a coin flip.
    conn = build_store(tmp_path / "probe2.sqlite", entries, cfg)
    try:
        row2 = extract_one(conn, entries[0],
                           CapturingLLM(lambda sysm, usr: "{not json"), cfg,
                           granular=True)
    finally:
        conn.close()
    assert row2["parse_failed"] is True
    assert row2["reply_chars"] == len("{not json")


def test_a_dump_without_recorded_replies_reads_unknown_not_zero():
    """A missing field must not answer the question it cannot answer.

    Dumps written before reply recording carry no `reply_chars`. Counting those
    as "0 empty replies" would report a confident cause for a run whose replies
    were thrown away — the failure shape the diagnostic-controls memo calls a
    device that returns a constant when it is broken.
    """
    old = [_row("target", parse_failed=True, episodes=0) for _ in range(4)]
    for r in old:
        r.pop("reply_chars", None)
    s_old = summarize(old, [_row("control")], None)
    assert s_old["replies_recorded"] == 0
    assert s_old["empty_replies"] == 0      # unknown, and the report says so

    new = [_row("target", parse_failed=True, episodes=0) for _ in range(4)]
    for r in new:
        r["reply_chars"] = 0
    s_new = summarize(new, [_row("control")], None)
    assert s_new["replies_recorded"] == 4
    assert s_new["empty_replies"] == 4

    # And the counts are over the FAILING rows only. A first cut counted every
    # row, so a single recorded SUCCESS made an unrecorded set of failures look
    # diagnosed and the report printed a confident, wrong remedy.
    mixed = [_row("target", parse_failed=True, episodes=0) for _ in range(3)]
    for r in mixed:
        r.pop("reply_chars", None)
    ok_row = _row("target")
    ok_row["reply_chars"] = 500
    s_mixed = summarize(mixed + [ok_row], [_row("control")], None)
    assert s_mixed["parse_failures"] == 3
    assert s_mixed["replies_recorded"] == 0     # not 1, and not 5
    assert s_mixed["empty_replies"] == 0
