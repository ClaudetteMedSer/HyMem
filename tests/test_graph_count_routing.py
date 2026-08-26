"""Tests for the in-domain graph-count layer wired into augment(ability="MR").

Covers: (1) the pure routing heuristic in `count_routing.plan_count` — the
subject-vs-object side decision for the anchored and unanchored shapes, and the
abstain cases (out-of-vocab, ambiguous, un-typed); (2) augment() surfacing an
EXACT `graph_count` alongside the lexical `total_message_matches` candidate; and
(3) a TRUE end-to-end test: seed raw turns, run a REAL dream that extracts
in-vocab triples + entity types, then assert augment() reports the exact count.
"""

from __future__ import annotations

from hymem.query.count_routing import plan_count
from tests.conftest import make_routed_llm


# ---------------------------------------------------------------------------
# Pure routing heuristic (no DB).
# ---------------------------------------------------------------------------

def test_plan_anchored_counts_typed_subjects():
    # "how many <T:services> depend on <E:redis>" -> distinct subjects of type T
    # anchored on object E.
    plan = plan_count("how many services depend on redis?", ["redis"])
    assert plan is not None
    assert plan.count == "subject"
    assert plan.subject_type == "service"
    assert plan.object == "redis"
    assert plan.object_type is None
    assert plan.predicates == ["depends_on"]


def test_plan_unanchored_counts_typed_objects():
    # "how many <T:databases> do we use" with no anchor entity -> distinct
    # objects of type T.
    plan = plan_count("how many databases do we use?", [])
    assert plan is not None
    assert plan.count == "object"
    assert plan.object_type == "database"
    assert plan.subject_type is None
    assert plan.object is None


def test_plan_without_predicate_still_routes_typed():
    # No predicate keyword ("have"), but a clear type -> typed all-predicate count.
    plan = plan_count("how many databases do we have?", [])
    assert plan is not None
    assert plan.count == "object"
    assert plan.object_type == "database"
    assert plan.predicates is None


def test_plan_out_of_vocab_type_returns_none():
    # "shirts" is not an in-vocab entity type -> no graph count.
    assert plan_count("how many shirts do I own?", []) is None


def test_plan_ambiguous_multiple_types_returns_none():
    # Two in-vocab types named -> ambiguous which column to constrain -> abstain.
    assert plan_count("how many databases and frameworks do we use?", []) is None


def test_plan_no_type_returns_none():
    # An entity but no type keyword -> nothing typed to count.
    assert plan_count("how many times did we use redis?", ["redis"]) is None


# ---------------------------------------------------------------------------
# augment() integration: graph_count coexists with the keyword candidate.
# ---------------------------------------------------------------------------

def _seed_edge(hy, subj, pred, obj, *, subj_type=None, obj_type=None):
    hy.conn.execute(
        "INSERT INTO knowledge_graph(subject_canonical, predicate, object_canonical, "
        "pos_evidence, status) VALUES (?, ?, ?, 1, 'active')",
        (subj, pred, obj),
    )
    if subj_type:
        hy.conn.execute(
            "INSERT OR IGNORE INTO entity_types(entity_canonical, type) VALUES (?, ?)",
            (subj, subj_type),
        )
    if obj_type:
        hy.conn.execute(
            "INSERT OR IGNORE INTO entity_types(entity_canonical, type) VALUES (?, ?)",
            (obj, obj_type),
        )


def test_augment_mr_sets_exact_graph_count_anchored(hy):
    # Three distinct services depend on redis; one extra edge re-uses a subject
    # via a different predicate (must not double-count).
    _seed_edge(hy, "billing", "depends_on", "redis", subj_type="service")
    _seed_edge(hy, "auth", "depends_on", "redis", subj_type="service")
    _seed_edge(hy, "gateway", "depends_on", "redis", subj_type="service")
    _seed_edge(hy, "billing", "uses", "redis", subj_type="service")
    # A non-service subject must be excluded by the type filter.
    _seed_edge(hy, "cronjob", "depends_on", "redis")

    ctx = hy.augment("how many services depend on redis?", ability="MR")

    assert ctx.graph_count is not None
    assert ctx.graph_count.count == 3
    assert ctx.graph_count.counted == "subject"
    assert set(ctx.graph_count.entities) == {"billing", "auth", "gateway"}


def test_augment_mr_sets_exact_graph_count_unanchored(hy):
    _seed_edge(hy, "app", "uses", "postgres", obj_type="database")
    _seed_edge(hy, "app", "uses", "redis", obj_type="database")
    _seed_edge(hy, "app", "uses", "react", obj_type="framework")

    ctx = hy.augment("how many databases do we use?", ability="MR")

    assert ctx.graph_count is not None
    assert ctx.graph_count.count == 2
    assert ctx.graph_count.counted == "object"
    assert set(ctx.graph_count.entities) == {"postgres", "redis"}


def test_augment_mr_recovers_reverse_direction_when_type_is_object(hy):
    # "how many databases does billing use" types the OBJECT (databases are the
    # objects of `billing uses redis`), the mirror of the router's subject-typed
    # default. The primary orientation (subject_type=database, object=billing)
    # finds nothing, so the mirror orientation (object_type=database,
    # subject=billing) must recover the exact count instead of reporting 0.
    _seed_edge(hy, "billing", "uses", "redis", obj_type="database")
    _seed_edge(hy, "billing", "uses", "postgres", obj_type="database")
    _seed_edge(hy, "billing", "uses", "react", obj_type="framework")

    ctx = hy.augment("how many databases does billing use?", ability="MR")

    assert ctx.graph_count is not None
    assert ctx.graph_count.count == 2  # redis + postgres, not react
    assert ctx.graph_count.counted == "object"
    assert set(ctx.graph_count.entities) == {"redis", "postgres"}


def test_augment_mr_suppresses_misleading_zero_graph_count(hy):
    # An in-vocab mapping that finds no edges in EITHER orientation must not
    # surface an "exact 0" — it falls back to None so the keyword candidate (run
    # over the raw turns) is the host's answer instead.
    sid = "z"
    hy.open_session(sid)
    hy.log_message(sid, "user", "How many services depend on kafka, I wonder.")
    hy.close_session(sid)

    ctx = hy.augment("how many services depend on kafka?", ability="MR")

    assert ctx.graph_count is None  # zero-with-no-evidence suppressed
    assert ctx.total_message_matches >= 1  # keyword fallback still answers


def test_augment_mr_out_of_vocab_leaves_graph_count_none_but_keyword_runs(hy):
    # Seed raw user turns so the keyword aggregate path has something to count,
    # but the question maps to no in-vocab type -> graph_count stays None.
    sid = "shirts"
    hy.open_session(sid)
    hy.log_message(sid, "user", "I bought a red shirt.")
    hy.log_message(sid, "user", "I bought a blue shirt.")
    hy.close_session(sid)

    ctx = hy.augment("how many shirts did I buy?", ability="MR")

    assert ctx.graph_count is None
    # The lexical candidate still works as the fallback.
    assert ctx.total_message_matches == 2


# ---------------------------------------------------------------------------
# TRUE end-to-end: seed turns, run a REAL dream, then count via augment().
# ---------------------------------------------------------------------------

def test_end_to_end_real_dream_then_exact_graph_count(hy):
    # 1. Seed raw turns describing three services depending on redis.
    sid = "deps"
    hy.open_session(sid)
    hy.log_message(sid, "user", "The billing service depends on redis.")
    hy.log_message(sid, "user", "The auth service depends on redis.")
    hy.log_message(sid, "user", "The gateway service depends on redis.")
    hy.close_session(sid)

    # 2. Inject in-vocab triples + entity types and run a REAL dream so the
    #    knowledge_graph and entity_types tables are populated by the pipeline
    #    (not hand-inserted), exactly as production would.
    triples = [
        {"subject": "billing", "predicate": "depends_on", "object": "redis",
         "subject_type": "service", "object_type": "database", "polarity": 1},
        {"subject": "auth", "predicate": "depends_on", "object": "redis",
         "subject_type": "service", "object_type": "database", "polarity": 1},
        {"subject": "gateway", "predicate": "depends_on", "object": "redis",
         "subject_type": "service", "object_type": "database", "polarity": 1},
    ]
    hy.set_llm(make_routed_llm(triples, markers=[]))
    hy.dream()

    # 3. Ask the counting question through the public MR path.
    ctx = hy.augment("how many services depend on redis?", ability="MR")

    assert ctx.graph_count is not None
    assert ctx.graph_count.count == 3
    assert ctx.graph_count.counted == "subject"
    assert set(ctx.graph_count.entities) == {"billing", "auth", "gateway"}
