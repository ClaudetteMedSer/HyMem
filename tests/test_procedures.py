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

from hymem import HyMem
from hymem.dreaming.procedures import (
    ProceduresExtraction,
    extract_procedures_for_session,
    persist_procedures,
)
from hymem.extraction.llm import StubLLMClient


# --- helpers ---------------------------------------------------------------


def _procedure_llm(items: list[dict]) -> StubLLMClient:
    """Stub that returns ``items`` for the procedure-extraction prompt and an
    empty array for every other extraction call (triples, markers, episodes,
    session-summary). Substring keyed on the unique opener of
    `PROCEDURE_SYSTEM`."""
    return StubLLMClient(
        fixtures={"step-by-step procedures": json.dumps(items)},
        default="[]",
    )


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


def test_extract_filters_items_missing_name_or_steps(cfg):
    """An item with no name, no steps list, or no valid step objects is
    silently dropped. The valid sibling still comes through."""
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
        assert len(rows) == 1, "only the well-formed procedure should survive"
        row = rows[0]
        assert row["name"] == "Deploy to staging"
        assert row["description"].startswith("Push the current build")
        assert json.loads(row["triggers"]) == ["deploy", "ship to staging"]
        assert json.loads(row["entities_involved"]) == ["docker", "staging"]
        steps = json.loads(row["steps"])
        assert [s["order"] for s in steps] == [1, 2]
        assert steps[0]["tool"] == "docker"
        assert steps[1]["tool"] is None
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
                {"order": 7, "action": "interpret coverage report"},
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
