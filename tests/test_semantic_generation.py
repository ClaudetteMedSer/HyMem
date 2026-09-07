"""Producer/loaded-code changes must replay lossless memory, not just Phase-1."""
from dataclasses import replace
from contextlib import closing
import json

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import digest, facts, runner, user_profile
from hymem.dreaming.retention import prune_messages
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import StubLLMClient


def _client(label="Alpha"):
    return StubLLMClient(fixtures={
        "typed user-profile facts": json.dumps({"items": [{
            "slot": "role", "value": label + " engineer",
            "evidence_message_id": 1, "confidence": 0.9,
        }]}),
        "Return the JSON array of narrative facts now": json.dumps([{
            "text": label + " attended the engineering meeting.",
            "date": None, "entities": [label],
        }]),
        "You analyze one conversation session": json.dumps({
            "episodes": [], "summary": label + " engineering session.",
            "procedures": [],
        }),
        "single pass": '{"triples":[],"markers":[],"complete":true}',
    }, default="[]")


def _config(cfg):
    return replace(cfg, aggregation_nodes_enabled=False,
                   profile_extraction_enabled=True, facts_extraction_enabled=True)


def _state(hy):
    return tuple(hy.conn.execute(
        "SELECT digest_published_generation,profile_published_generation,"
        "facts_cursor_prompt_version FROM sessions WHERE id='semantic'"
    ).fetchone())


def _seed(hy):
    hy.log_message("semantic", "user", "I am an engineer and attended a meeting.",
                   created_at="2020-01-01")
    hy.close_session("semantic")


def test_changed_producer_replays_retained_source_and_unchanged_reopen_skips(cfg):
    config = _config(cfg)
    alpha = _client()
    with closing(HyMem(config, llm=alpha)) as hy:
        _seed(hy)
        hy.dream()
        original = _state(hy)
        with db.transaction(hy.conn):
            assert prune_messages(hy.conn, replace(config, message_retention_days=1)) == 1
        assert hy.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    beta = _client("Beta")
    with closing(HyMem(config, llm=beta)) as hy:
        status = hy.dream_status()
        assert [status["pending_" + tier] for tier in ("digests", "profiles", "facts")] == [1, 1, 1]
        hy.dream()
        assert all(old != new for old, new in zip(original, _state(hy)))
        assert [entry.value for entry in hy.profile()] == ["Beta engineer"]
        assert [row["text"] for row in hy.conn.execute(
            "SELECT text FROM narrative_facts WHERE lifecycle_status='active' "
            "AND invalid_at IS NULL"
        )] == [
            "Beta attended the engineering meeting."
        ]
        status = hy.dream_status()
        for tier in ("digests", "profiles", "facts"):
            assert status["pending_" + tier] == status["malformed_" + tier] == 0
        calls = len(beta.calls)
        hy.dream()
        assert len(beta.calls) == calls
    same_beta = _client("Beta")
    with closing(HyMem(config, llm=same_beta)) as hy:
        hy.dream()
        assert same_beta.calls == []


@pytest.mark.parametrize("tier,module,name", [
    ("digest", digest, "validate_episode_items"),
    ("digest", runner, "extract_session_digest"),
    ("digest", runner, "_run_dreaming"),
    ("profile", user_profile, "loads_exact_or_fenced"),
    ("profile", runner, "extract_user_profile"),
    ("facts", facts, "normalize"),
    ("facts", runner, "extract_facts"),
])
def test_loaded_consuming_alias_changes_identity(monkeypatch, tier, module, name):
    client = _client()
    before = semantic_generation_suffix(tier, client)
    original = getattr(module, name)

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, replacement)
    assert semantic_generation_suffix(tier, client) != before


def test_rebound_imported_alias_still_tracks_its_mutable_closure(monkeypatch):
    original = digest.validate_episode_items
    policy = {"enabled": True}

    def replacement(*args, **kwargs):
        if policy["enabled"]:
            return original(*args, **kwargs)
        raise RuntimeError("changed policy")

    monkeypatch.setattr(digest, "validate_episode_items", replacement)
    client = _client()
    before = semantic_generation_suffix("digest", client)
    policy["enabled"] = False
    assert semantic_generation_suffix("digest", client) != before


class _ChangingClient:
    def __init__(self, tier):
        self.delegate = _client()
        self.tier = tier
        self.changed = False

    def phase1_producer_declaration(self):
        return self.delegate.phase1_producer_declaration()

    def memory_producer_declaration(self):
        return self.delegate.memory_producer_declaration()

    def complete(self, request):
        response = self.delegate.complete(request)
        needles = {
            "digest": "You analyze one conversation session",
            "profile": "typed user-profile facts",
            "facts": "Return the JSON array of narrative facts now",
        }
        if not self.changed and needles[self.tier] in request.system + request.user:
            self.changed = True
            self.delegate.default = "producer changed while request was in flight"
        return response


@pytest.mark.parametrize("tier,index", [("digest", 0), ("profile", 1), ("facts", 2)])
def test_inflight_producer_change_cannot_publish(cfg, tier, index):
    client = _ChangingClient(tier)
    with closing(HyMem(_config(cfg), llm=client)) as hy:
        _seed(hy)
        report = hy.dream()
        assert client.changed
        assert _state(hy)[index] is None
        assert getattr(report, {"digest": "digest_failures", "profile": "profile_failures",
                                "facts": "fact_failures"}[tier]) == 1


def test_memory_route_identity_is_independent_of_phase1():
    class Routed(_ChangingClient):
        def __init__(self):
            super().__init__("unused")
            self.memory = _client("Beta")

        def memory_producer_declaration(self):
            return self.memory.memory_producer_declaration()

    client = Routed()
    phase1 = client.phase1_producer_declaration()
    before = semantic_generation_suffix("digest", client)
    client.memory = _client("Gamma")
    assert client.phase1_producer_declaration() == phase1
    assert semantic_generation_suffix("digest", client) != before


def test_undeclared_memory_route_has_process_instance_scope():
    class Undeclared:
        def phase1_producer_declaration(self):
            return _client().phase1_producer_declaration()

    first, second = Undeclared(), Undeclared()
    assert semantic_generation_suffix("digest", first) == semantic_generation_suffix("digest", first)
    assert semantic_generation_suffix("digest", first) != semantic_generation_suffix("digest", second)


@pytest.mark.parametrize("tier,module,name,index", [
    ("digest", digest, "validate_episode_items", 0),
    ("profile", user_profile, "loads_exact_or_fenced", 1),
    ("facts", facts, "normalize", 2),
])
def test_loaded_helper_drift_schedules_and_completes_replay(
    cfg, monkeypatch, tier, module, name, index,
):
    client = _client()
    with closing(HyMem(_config(cfg), llm=client)) as hy:
        _seed(hy)
        hy.dream()
        before = _state(hy)
        original = getattr(module, name)

        def replacement(*args, **kwargs):
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, replacement)
        assert hy.dream_status()["pending_" + {
            "digest": "digests", "profile": "profiles", "facts": "facts",
        }[tier]] == 1
        hy.dream()
        assert _state(hy)[index] != before[index]


@pytest.mark.parametrize("replacement_label", ["Alpha", "Beta"])
def test_local_profile_replay_replaces_interpretation_not_later_source(cfg, replacement_label):
    with closing(HyMem(_config(cfg), llm=_client())) as hy:
        _seed(hy)
        hy.dream()
        later_id = hy.log_message("later", "user", "I now work as a manager.",
                                   created_at="2021-01-01")
        with db.transaction(hy.conn):
            materialize_message_coverage(hy.conn, "later")
            user_profile.persist_user_profile(
                hy.conn, user_profile.ProfileExtraction(items=[{
                    "slot": "role", "value": "Later manager", "confidence": 0.8,
                    "evidence_message_id": later_id,
                }]),
            )
        beta = _client(replacement_label)
        item = json.loads(beta.fixtures["typed user-profile facts"])
        item["items"][0]["confidence"] = 0.25
        beta.fixtures["typed user-profile facts"] = json.dumps(item)
        hy.set_llm(beta)
        hy.dream(session_ids=["semantic"])
        assert [entry.value for entry in hy.profile()] == ["Later manager"]
        prior = hy.conn.execute(
            "SELECT value,confidence,invalid_at FROM user_profile "
            "WHERE source_session_id='semantic'"
        ).fetchall()
        assert len(prior) == 1
        assert prior[0]["value"] == replacement_label + " engineer"
        assert prior[0]["confidence"] == 0.25
        assert prior[0]["invalid_at"] is not None


def test_earlier_tier_producer_change_does_not_quarantine_new_facts_producer(cfg):
    client = _ChangingClient("digest")
    config = replace(_config(cfg), profile_extraction_enabled=False,
                     facts_extraction_max_attempts=1)
    with closing(HyMem(config, llm=client)) as hy:
        _seed(hy)
        first = hy.dream()
        assert first.digest_failures == first.fact_failures == 1
        assert _state(hy)[2] is None
        assert hy.dream_status()["quarantined_facts"] == 0
        assert hy.dream_status()["pending_facts"] == 1
        second = hy.dream()
        assert second.fact_failures == 0
        assert _state(hy)[2] is not None


def test_captured_fact_retry_version_rejects_ambiguous_or_mismatched_config(cfg):
    client = _client()
    version = facts.facts_config_version(cfg, client=client)
    assert facts.facts_retry_policy_version(cfg, publication_version=version) == (
        facts.facts_retry_policy_version(cfg, client=client)
    )
    with pytest.raises(ValueError, match="both captured and live"):
        facts.facts_retry_policy_version(cfg, client=client, publication_version=version)
    with pytest.raises(ValueError, match="does not match configuration"):
        facts.facts_retry_policy_version(
            replace(cfg, dream_digest_max_chars=cfg.dream_digest_max_chars + 1),
            publication_version=version,
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_same_source_items_cannot_choose_by_model_order(cfg, reverse):
    with closing(HyMem(_config(cfg), llm=_client())) as hy:
        _seed(hy)
        hy.dream()
        before = _state(hy)[1]
        beta = _client("Beta")
        payload = json.loads(beta.fixtures["typed user-profile facts"])
        payload["items"].append({**payload["items"][0], "value": "Gamma engineer"})
        if reverse:
            payload["items"].reverse()
        beta.fixtures["typed user-profile facts"] = json.dumps(payload)
        hy.set_llm(beta)
        report = hy.dream()
        assert report.profile_failures == 1
        assert _state(hy)[1] == before
        assert [entry.value for entry in hy.profile()] == ["Alpha engineer"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("replacement_label", ["Alpha", "Beta"])
def test_equal_source_assertions_fold_only_new_generation_confidence(
    cfg, reverse, replacement_label,
):
    with closing(HyMem(_config(cfg), llm=_client())) as hy:
        _seed(hy)
        hy.dream()
        assert hy.profile()[0].confidence == 0.9
        replacement = _client(replacement_label)
        payload = json.loads(replacement.fixtures["typed user-profile facts"])
        payload["items"][0]["confidence"] = 0.2
        payload["items"].append({**payload["items"][0], "confidence": 0.4})
        if reverse:
            payload["items"].reverse()
        replacement.fixtures["typed user-profile facts"] = json.dumps(payload)
        hy.set_llm(replacement)
        report = hy.dream()
        assert report.profile_failures == 0
        assert len(hy.profile()) == 1
        assert hy.profile()[0].value == replacement_label + " engineer"
        assert hy.profile()[0].confidence == 0.4
