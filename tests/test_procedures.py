"""Tests for the procedural-memory layer:

  * `extract_procedures_for_session` validates LLM output (drops items
    with no name or no valid steps, reorders step `order` to 1..N).
  * `persist_procedures` JSON-encodes the three list columns and
    INSERT-OR-IGNOREs into `procedures`.
  * Dream-loop integration: `hy.dream()` writes procedures and the FTS
    shadow row that backs `_procedure_search`.
  * `augment()` surfaces matching procedures via `ProcedureHit` with
    steps decoded back to a list of dicts.
"""

from __future__ import annotations

import json
import re

from hymem import HyMem
from hymem.dreaming.procedures import (
    ProceduresExtraction,
    extract_procedures_for_session,
    persist_procedures,
)
from hymem.extraction.llm import StubLLMClient


# --- helpers ---------------------------------------------------------------


class _ProcedureDigestLLM:
    """Produce strict digest procedures backed by ids in the actual window."""

    def __init__(self, items: list[dict]):
        self.items = items
        self.calls = []
        self.cited_chunk_ids: list[list[str]] = []

    def complete(self, request) -> str:
        self.calls.append(request)
        if "Return the JSON object now" not in request.user:
            return "[]"
        # The strict digest contract admits only producer-rendered NEW
        # msgcov ids. Deriving these from the exact prompt keeps the fixture
        # valid when global message ids and coverage hashes change.
        chunk_ids = list(dict.fromkeys(re.findall(
            r"\[chunk (msgcov_[^\]\s]+)\]", request.user
        )))
        assert chunk_ids, "digest prompt must expose current coverage provenance"
        self.cited_chunk_ids.append(chunk_ids)
        procedures = []
        for item in self.items:
            prepared = json.loads(json.dumps(item))
            prepared["chunk_ids"] = chunk_ids
            procedures.append(prepared)
        return json.dumps({
            "episodes": [], "summary": "", "procedures": procedures,
        })


def _procedure_llm(items: list[dict]) -> _ProcedureDigestLLM:
    """Stub that returns ``items`` as the procedures section of the batched
    session-digest response (episodes/summary empty), and an empty array for
    every other extraction call (triples, markers). Keyed on the unique digest
    user-prompt closer ``Return the JSON object now``."""
    return _ProcedureDigestLLM(items)


def _assert_current_coverage_was_cited(
    hy: HyMem, sid: str, llm: _ProcedureDigestLLM
) -> None:
    assert llm.cited_chunk_ids
    actual = {
        row["chunk_id"]
        for row in hy.conn.execute(
            "SELECT chunk_id FROM message_retention_coverage "
            "WHERE source_session_id = ? AND source_role IN ('user','assistant')",
            (sid,),
        ).fetchall()
    }
    assert set(llm.cited_chunk_ids[-1]) == actual


def _seed_session(hy: HyMem, sid: str, turns: list[tuple[str, str]]) -> None:
    hy.open_session(sid)
    for role, content in turns:
        hy.log_message(sid, role, content)
    hy.close_session(sid)


# --- extraction validation -------------------------------------------------


def test_extract_returns_none_for_empty_session(cfg, stub_llm):
    """No chunks and no episodes → don't call the LLM, return None."""
    hy = HyMem(cfg, llm=stub_llm)
    try:
        hy.open_session("s_empty")
        hy.close_session("s_empty")
        ext = extract_procedures_for_session(hy.conn, "s_empty", hy._llm)
        assert ext is None
        # And no spurious LLM call happened either.
        assert stub_llm.calls == []
    finally:
        hy.close()


def test_digest_rejects_malformed_procedure_siblings_atomically(cfg):
    """Malformed non-empty items hold the whole digest slice for retry.

    Persisting the valid sibling while advancing would silently discard the
    malformed procedures and falsely claim complete source coverage.
    """
    llm = _procedure_llm([
        {"name": "", "steps": [{"order": 1, "action": "do thing"}]},          # no name
        {"name": "no steps", "steps": []},                                    # empty steps
        {"name": "non-list steps", "steps": "do everything"},                 # steps not a list
        {"name": "garbage steps", "steps": [{"order": "x", "action": ""}]},   # no valid step
        {
            "name": "Deploy to staging",
            "description": "Push the current build to the staging cluster.",
            "steps": [
                {"order": 1, "action": "build the docker image", "tool": "docker"},
                {"order": 2, "action": "push the image to the registry"},
            ],
            "triggers": ["deploy", "ship to staging"],
            "entities_involved": ["docker", "staging"],
        },
    ])
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_filter"
        _seed_session(hy, sid, [
            ("assistant", "let's document our deploy"),
            ("user", "Sure — to push to staging we build the docker image and then push it to the registry."),
        ])
        hy.dream()

        rows = hy.conn.execute(
            "SELECT name, description, steps, triggers, entities_involved "
            "FROM procedures WHERE session_id = ?",
            (sid,),
        ).fetchall()
        assert rows == []
    finally:
        hy.close()


def test_extract_renormalizes_step_order(cfg):
    """LLM may emit steps out of order or with non-contiguous order numbers.
    ``persist_procedures`` (via extraction) sorts by order and reassigns
    1..N so downstream consumers never see gaps."""
    llm = _procedure_llm([
        {
            "name": "Run integration tests",
            "description": "Run the pytest integration suite locally.",
            "steps": [
                {"order": 7, "action": "interpret coverage report", "tool": None},
                {"order": 3, "action": "spin up the docker compose stack", "tool": "docker compose"},
                {"order": 5, "action": "invoke pytest", "tool": "pytest"},
            ],
            "triggers": ["test", "integration test"],
            "entities_involved": ["pytest", "docker"],
        },
    ])
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_reorder"
        _seed_session(hy, sid, [
            ("assistant", "what does the integration test setup look like?"),
            ("user", "We spin up docker compose, run pytest, then read the coverage report."),
        ])
        hy.dream()
        _assert_current_coverage_was_cited(hy, sid, llm)

        row = hy.conn.execute(
            "SELECT steps FROM procedures WHERE session_id = ?", (sid,),
        ).fetchone()
        assert row is not None
        steps = json.loads(row["steps"])
        # Sorted by original `order`, then renumbered to 1..N.
        assert [s["order"] for s in steps] == [1, 2, 3]
        assert [s["action"] for s in steps] == [
            "spin up the docker compose stack",
            "invoke pytest",
            "interpret coverage report",
        ]


    finally:
        hy.close()


def test_extract_handles_invalid_json(cfg):
    """Non-JSON LLM output must not raise. The extractor returns an empty
    ProceduresExtraction, which persist treats as a no-op. We seed chunks
    via a first `dream()` pass (so the extractor has content to feed the
    LLM), then call extract directly with the garbage-returning stub."""
    bad_llm = StubLLMClient(
        fixtures={"step-by-step procedures": "this is not valid json"},
        default="[]",
    )
    hy = HyMem(cfg, llm=bad_llm)
    try:
        sid = "s_garbage"
        _seed_session(hy, sid, [
            ("assistant", "kickoff"),
            ("user", "Pretend there's a deploy procedure here long enough to clear the salience minimum."),
        ])
        # First dream exercises the runner's swallow-and-log path.
        hy.dream()
        # And no procedures should have been persisted from garbage JSON.
        count = hy.conn.execute(
            "SELECT COUNT(*) AS c FROM procedures WHERE session_id = ?", (sid,),
        ).fetchone()["c"]
        assert count == 0

        # Direct call now that chunks exist: returns an empty extraction.
        ext = extract_procedures_for_session(hy.conn, sid, hy._llm)
        assert isinstance(ext, ProceduresExtraction)
        assert ext.items == []
        inserted = persist_procedures(hy.conn, sid, ext)
        assert inserted == 0
    finally:
        hy.close()


# --- augment / FTS surface -------------------------------------------------


def test_procedure_surfaces_via_augment_fts(cfg):
    """After dream, augment() returns the procedure for a query that matches
    its name / description via the FTS index."""
    llm = _procedure_llm([
        {
            "name": "Deploy to staging",
            "description": "Push the latest build to the staging Kubernetes cluster.",
            "steps": [
                {"order": 1, "action": "build the docker image", "tool": "docker"},
                {"order": 2, "action": "kubectl apply the staging manifests", "tool": "kubectl"},
            ],
            "triggers": ["deploy staging", "ship to staging"],
            "entities_involved": ["docker", "kubernetes", "staging"],
        },
    ])
    hy = HyMem(cfg, llm=llm)
    try:
        sid = "s_aug"
        _seed_session(hy, sid, [
            ("assistant", "how do we deploy to staging again?"),
            ("user", "Build the docker image and kubectl apply the staging manifests."),
        ])
        hy.dream()
        _assert_current_coverage_was_cited(hy, sid, llm)

        ctx = hy.augment("how do I deploy to staging?")
        assert ctx.procedures, "expected a procedure hit for a staging-deploy query"
        top = ctx.procedures[0]
        assert top.name == "Deploy to staging"
        # Steps come back as a parsed list of dicts, not raw JSON.
        assert isinstance(top.steps, list) and top.steps
        assert top.steps[0]["action"] == "build the docker image"
        assert top.steps[0]["tool"] == "docker"
        assert top.session_id == sid
    finally:
        hy.close()


# --- procedural feedback loop (mark_procedure_stale) -----------------------


def _seed_one_procedure(cfg) -> tuple[HyMem, str]:
    """Dream a single 'Deploy to staging' procedure into a fresh HyMem and
    return (hy, procedure_id) for the surfaced hit."""
    llm = _procedure_llm([
        {
            "name": "Deploy to staging",
            "description": "Push the latest build to the staging Kubernetes cluster.",
            "steps": [
                {"order": 1, "action": "build the docker image", "tool": "docker"},
                {"order": 2, "action": "kubectl apply the staging manifests", "tool": "kubectl"},
            ],
            "triggers": ["deploy staging"],
            "entities_involved": ["docker", "kubernetes", "staging"],
        },
    ])
    hy = HyMem(cfg, llm=llm)
    _seed_session(hy, "s_stale", [
        ("assistant", "how do we deploy to staging again?"),
        ("user", "Build the docker image and kubectl apply the staging manifests."),
    ])
    hy.dream()
    _assert_current_coverage_was_cited(hy, "s_stale", llm)
    hit = hy.augment("how do I deploy to staging?").procedures[0]
    return hy, hit.procedure_id


def test_mark_procedure_stale_hides_from_search(cfg):
    """A procedure marked stale stops surfacing in augment()."""
    hy, pid = _seed_one_procedure(cfg)
    try:
        assert hy.mark_procedure_stale(pid) is True

        ctx = hy.augment("how do I deploy to staging?")
        assert all(p.procedure_id != pid for p in ctx.procedures), (
            "stale procedure must not surface via _procedure_search"
        )

        row = hy.conn.execute(
            "SELECT status FROM procedures WHERE id = ?", (pid,)
        ).fetchone()
        assert row["status"] == "stale"
    finally:
        hy.close()


def test_mark_procedure_stale_downgrades_confidence(cfg):
    """Marking stale knocks confidence down by the configured factor."""
    hy, pid = _seed_one_procedure(cfg)
    try:
        before = hy.conn.execute(
            "SELECT confidence FROM procedures WHERE id = ?", (pid,)
        ).fetchone()["confidence"]
        assert before == 1.0  # default for a freshly persisted procedure

        hy.mark_procedure_stale(pid)

        after = hy.conn.execute(
            "SELECT confidence FROM procedures WHERE id = ?", (pid,)
        ).fetchone()["confidence"]
        assert after == before * hy.config.procedure_stale_confidence_factor
    finally:
        hy.close()


def test_mark_procedure_stale_is_idempotent(cfg):
    """Second call on an already-stale procedure returns False and does not
    double-discount confidence."""
    hy, pid = _seed_one_procedure(cfg)
    try:
        assert hy.mark_procedure_stale(pid) is True
        once = hy.conn.execute(
            "SELECT confidence FROM procedures WHERE id = ?", (pid,)
        ).fetchone()["confidence"]

        assert hy.mark_procedure_stale(pid) is False
        twice = hy.conn.execute(
            "SELECT confidence FROM procedures WHERE id = ?", (pid,)
        ).fetchone()["confidence"]
        assert twice == once
    finally:
        hy.close()


def test_mark_procedure_stale_unknown_id_returns_false(cfg, stub_llm):
    hy = HyMem(cfg, llm=stub_llm)
    try:
        assert hy.mark_procedure_stale("does-not-exist@proc0") is False
    finally:
        hy.close()


# --- fenced replies (dream 1013) -------------------------------------------


_PROC_TURNS = [
    ("assistant", "how do we deploy?"),
    ("user", "Pretend there's a deploy procedure here long enough to clear the salience minimum."),
]


def _hy_with_chunks(cfg, sid: str) -> HyMem:
    """HyMem whose `sid` has real chunks, ready for a direct extract call."""
    hy = HyMem(cfg, llm=StubLLMClient(default="[]"))
    _seed_session(hy, sid, _PROC_TURNS)
    hy.dream()
    return hy


def test_extract_procedures_parses_fenced_reply(cfg):
    """The procedure call sets response_format="json"; dream 1013 proved a
    provider will fence the reply anyway, and this path had no retry and no
    log — a fenced reply was silent, permanent data loss."""
    sid = "s_fenced_proc"
    hy = _hy_with_chunks(cfg, sid)
    try:
        fenced = "```json\n" + json.dumps([{
            "name": "Deploy to staging",
            "description": "Ship the build.",
            "steps": [{"order": 1, "action": "build the image"}],
        }]) + "\n```"
        hy.set_llm(StubLLMClient(
            fixtures={"identify step-by-step procedures": fenced}, default="[]",
        ))
        ext = extract_procedures_for_session(hy.conn, sid, hy._llm)
        assert ext is not None
        assert [i["name"] for i in ext.items] == ["Deploy to staging"]
    finally:
        hy.close()


def test_extract_procedures_refusal_yields_empty_extraction(cfg, caplog):
    """An unparseable reply keeps the documented empty ProceduresExtraction,
    now with a warning instead of silence."""
    sid = "s_refusal_proc"
    hy = _hy_with_chunks(cfg, sid)
    try:
        hy.set_llm(StubLLMClient(
            fixtures={"identify step-by-step procedures": "I cannot do that."},
            default="[]",
        ))
        with caplog.at_level("WARNING"):
            ext = extract_procedures_for_session(hy.conn, sid, hy._llm)
        assert ext == ProceduresExtraction()
        assert any(
            "procedures.parse_failure" in r.message and sid in r.getMessage()
            for r in caplog.records
        )
    finally:
        hy.close()


def test_extract_procedures_wrong_shape_yields_empty_extraction(cfg, caplog):
    """validate_procedure_items() absorbs a non-list into [], which reads as
    "no procedures in this session" — the one thing a dropped reply must not
    be mistaken for."""
    sid = "s_shape_proc"
    hy = _hy_with_chunks(cfg, sid)
    try:
        hy.set_llm(StubLLMClient(
            fixtures={"identify step-by-step procedures": '{"procedures": "none"}'},
            default="[]",
        ))
        with caplog.at_level("WARNING"):
            ext = extract_procedures_for_session(hy.conn, sid, hy._llm)
        assert ext == ProceduresExtraction()
        assert any(
            "procedures.shape_failure" in r.message and sid in r.getMessage()
            for r in caplog.records
        )
    finally:
        hy.close()
