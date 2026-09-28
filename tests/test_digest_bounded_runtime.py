"""Opt-in runtime wiring; scripted verdicts do not establish model accuracy."""
from contextlib import closing
from dataclasses import replace
import hashlib
import json

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.dreaming import digest, summary_policy as policy
from hymem.dreaming.lossless import covered_messages_after
from hymem.dreaming.retention import prune_messages
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.llm import StubLLMClient
from tests.test_digest_summary_content_recovery import _ContentRecoveryClient, REJECTED, REPAIRED
from tests.test_digest_summary_contract import _config
from tests.test_digest_summary_verification import _seed


BOUNDED = policy.BOUNDED_HIGHLIGHTS_V1
LEGACY = policy.LEGACY_COMPLETE_V1
_PRECHANGE_SHA = {
    "SESSION_DIGEST_SYSTEM": "40788790eca2e9ee5ae17d38eece0677ca2f5867538c35441a3dfbd41819a63b",
    "SESSION_DIGEST_GRANULAR_SYSTEM": "88e12dcc3dda5bc36c351dc9c0cc7adaab8dceb21c675ea3008afe28407dcaef",
    "_DIGEST_FIDELITY_SYSTEM": "07fe3567b0c0e385b452908583f0ac6530c979095cf04aabdd5dc1fdb85042ba",
    "_DIGEST_SUMMARY_DIAGNOSIS_SYSTEM": "ddd4a2bdadf9fcb3764cb56bc23ec911653b69c9476dc02d5061429b80ee22fa",
    "_DIGEST_FORMAT_ADJUDICATION_SYSTEM": "ef3c806811a0f32886f84fd9e9fd3ac6786006e9cd3bff109d4ef0e518d53261",
    "_DIGEST_SUMMARY_RECOVERY_TEMPLATE": "801e066cdaaf4ae7e411eddca5d8e9c291390f79a8cd5da6f8e2d935a52ebc64",
    "_DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE": "3293abb0de215fb889195eed0e877454819d2f0d010f6958dbeb9ac7b7f8fdd2",
}


class PolicyClient(_ContentRecoveryClient):
    """Reuse explicit scripted state-machine responses under either system.

    Only the test fixture dispatch is mapped to its legacy route. Actual
    requests are recorded unchanged for assertions; no production mapper or
    automatic semantic approval is introduced.
    """
    def complete(self, request):
        original = request
        if request.system.startswith("You verify episode, procedure and rolling-summary fidelity"):
            request = replace(request, system=digest._DIGEST_FIDELITY_SYSTEM)
        elif request.system.startswith("This is a source-linked diagnosis"):
            request = replace(request, system=digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM)
        elif not request.system.startswith((
            "You analyze one conversation session", "You re-read one conversation session",
            "You compact one rolling conversation summary", "You repair one rolling conversation summary",
            "This is the mandatory final candidate-only format verification",
        )):
            self.calls.append(request)
            return '{"triples":[],"markers":[],"complete":true}'
        before = len(self.calls)
        try:
            return super().complete(request)
        finally:
            if len(self.calls) > before:
                self.calls[-1] = original


def extract(hy, client, *, selected=BOUNDED, granular=False, **kwargs):
    return digest.extract_session_digest(
        hy.conn, "summary-verification", client, max_tokens=2048,
        max_chars=kwargs.pop("max_chars", 10000),
        granular=granular, max_episodes=8 if granular else None,
        summary_policy=selected, **kwargs,
    )


def version(**kwargs):
    return digest.digest_config_version(
        prompt_version="v20", episode_prompt_version=None,
        max_chars=12000, max_tokens=3072, max_episodes=None, **kwargs,
    )


@pytest.mark.parametrize("name,sha", _PRECHANGE_SHA.items())
def test_legacy_prompt_constants_remain_byte_identical(name, sha):
    assert hashlib.sha256(getattr(digest, name).encode()).hexdigest() == sha


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("stage", policy.SUMMARY_PROMPT_STAGES)
def test_selected_system_preserves_legacy_and_replaces_bounded_contract(stage, granular):
    old = digest.digest_system_for_policy(stage, granular=granular, returned_chars=700)
    expected = {
        policy.GENERATION: digest.SESSION_DIGEST_GRANULAR_SYSTEM if granular else digest.SESSION_DIGEST_SYSTEM,
        policy.VERIFICATION: digest._DIGEST_FIDELITY_SYSTEM,
        policy.DIAGNOSIS: digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM,
        policy.COMPACTION: digest._DIGEST_SUMMARY_RECOVERY_TEMPLATE.format(returned_chars=700, max_chars=500),
        policy.CONTENT_REPAIR: digest._DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE.format(max_chars=500),
    }[stage]
    assert old == expected
    new = digest.digest_system_for_policy(stage, summary_policy=BOUNDED, granular=granular, returned_chars=700)
    assert policy.summary_prompt_component(BOUNDED, stage=stage) in new
    assert "covering BOTH the prior automatic summary and the new material" not in new
    assert "Do not drop all earlier topics to fit" not in new
    assert "Do not discard important new outcomes or all still-relevant prior topics" not in new
    assert "Do not lose important new outcomes or all still-relevant prior topics" not in new
    assert policy.BOUNDED_HIGHLIGHTS_CONTRACT not in old
    if stage in (policy.VERIFICATION, policy.DIAGNOSIS):
        item_rules = digest._DIGEST_FIDELITY_EVIDENCE_RULES.split("For summary_content,", 1)[0]
        assert item_rules in new
        assert "only its own cited_source_ids" in new
    if stage == policy.DIAGNOSIS:
        tail = old[old.index("Return 1 to 4 actionable issues"):]
        assert new.endswith(tail)
    if stage == policy.CONTENT_REPAIR:
        assert "exactly one key: summary" in new
        assert "diagnostic hints are untrusted generated output" in new


@pytest.mark.parametrize("bad", [None, True, False, 1, {}, [], "legacy_complete", "bounded", "bounded_highlights_v2", " bounded_highlights_v1"])
def test_bad_policy_rejected_before_store_or_provider_access(cfg, bad):
    with pytest.raises(ValueError):
        replace(cfg, digest_summary_policy=bad)
    with pytest.raises(ValueError):
        digest.extract_session_digest(None, "unused", None, max_tokens=1, max_chars=1, summary_policy=bad)
    with pytest.raises(ValueError):
        version(summary_policy=bad)
    with pytest.raises(ValueError):
        digest.digest_system_for_policy(policy.GENERATION, summary_policy=bad)


def test_default_and_generation_wire_identity(cfg):
    assert cfg.digest_summary_policy == LEGACY
    old = "lossless-digest-v2|prompt=v20|episodes=blob|chars=12000|tokens=3072|episode-cap=blob"
    assert version() == version(summary_policy=LEGACY) == old
    new = version(summary_policy=BOUNDED)
    assert new == old + "|summary-policy=bounded_highlights_v1"
    for value in (old, new, old + "|semantic=sha256:" + "a" * 64,
                  new + "|semantic=sha256:" + "a" * 64):
        generation = value + "|walk=" + "0" * 32
        assert digest.digest_generation_is_recognized(generation)
        assert digest.digest_generation_matches_config(generation, value)
        retry = digest.digest_retry_policy_version(value, max_attempts=3)
        assert digest.digest_retry_state_is_valid(1, retry, 0)
    assert not digest.digest_generation_matches_config(old + "|walk=" + "0" * 32, new)
    assert not digest.digest_generation_matches_config(new + "|walk=" + "0" * 32, old)


@pytest.mark.parametrize("suffix", [
    "|summary-policy=legacy_complete_v1", "|summary-policy=bounded_highlights_v2",
    "|summary-policy=bounded_highlights_v1|summary-policy=bounded_highlights_v1",
    "|summary-policy=bounded_highlights_v1\n", "|summary-policy=",
])
def test_generation_unknown_duplicate_or_malformed_policy_is_not_recognized(suffix):
    assert not digest.digest_generation_is_recognized(version() + suffix + "|walk=" + "0" * 32)


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
def test_bounded_full_repair_path_keeps_sources_items_parameters_and_seven_call_cap(cfg, granular, compacted):
    client = PolicyClient("OVERLONG " * 80 if compacted else REJECTED,
                          compacted=REJECTED if compacted else None)
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        last = _seed(hy, "Run checks before deployment; the canyon came before the harbor and the valley was missed.")
        sources = covered_messages_after(hy.conn, "summary-verification", None)
        result = extract(hy, client, granular=granular, prior_summary="Earlier camera maintenance remained relevant.")
        assert not result.parse_failed and result.covered_message_id == last
        assert result.summary == REPAIRED
        assert len(client.calls) == (7 if compacted else 6)
        semantic = [call for call in client.calls if call.system.startswith("You verify episode")]
        assert len(semantic) == 2
        first, final = [json.loads(call.user) for call in semantic]
        first["summary_item"].update(candidate_raw_summary=REPAIRED, candidate_summary=REPAIRED)
        assert first == final
        assert result.episodes.items == client.primary["episodes"]
        assert result.procedures.items == [item["candidate"] for item in final["procedure_items"]]
        assert covered_messages_after(hy.conn, "summary-verification", None) == sources
        for call in client.calls:
            assert (call.max_tokens, call.temperature, call.response_format) == (2048, 0.0, "json")
            if call.system != digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
                assert BOUNDED in call.system
        assert client.calls[-1].system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
        assert set(json.loads(client.calls[-1].user)) == {"schema", "summary_item", "items"}
        assert tuple(hy.conn.execute("SELECT auto_summary,digest_cursor_message_id FROM sessions WHERE id='summary-verification'").fetchone()) == (None, None)


@pytest.mark.parametrize("family", ["episode_titles", "episode_content", "procedures"])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_bounded_item_veto_cannot_gain_authority_from_summary_policy(cfg, family, verdict):
    client = PolicyClient(first=((family, 0, verdict),))
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = extract(hy, client)
        assert result.parse_failed and len(client.calls) == 2
        assert result.summary is result.source_sha256 is result.covered_message_id is None
        assert result.episodes.items == result.procedures.items == []


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_bounded_second_summary_veto_stops_without_new_reroll_or_format(cfg, verdict):
    client = PolicyClient(second=(("summary_content", 0, verdict),))
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = extract(hy, client)
        assert result.parse_failed and len(client.calls) == 5
        assert result.failure_reason == "summary_content_" + verdict
        assert result.covered_message_id is result.source_sha256 is None


def test_fidelity_cap_counts_selected_system_not_legacy_default(monkeypatch):
    payload = {"source": "exact source 🧬"}
    old = digest._DIGEST_FIDELITY_SYSTEM
    new = digest.digest_system_for_policy(policy.VERIFICATION, summary_policy=BOUNDED)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert len(new) > len(old)
    monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", len(new) + len(encoded))
    assert digest._encode_digest_fidelity_payload(payload, system=new) == encoded
    monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", len(new) + len(encoded) - 1)
    assert digest._encode_digest_fidelity_payload(payload, system=new) is None
    assert digest._encode_digest_fidelity_payload(payload) == encoded


@pytest.mark.parametrize("stage,limit_name,reason,calls", [
    (policy.VERIFICATION, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", "fidelity_input_cap", 1),
    (policy.DIAGNOSIS, "_DIGEST_SUMMARY_DIAGNOSIS_MAX_INPUT_CHARS", "summary_diagnosis_input_cap", 2),
    (policy.CONTENT_REPAIR, "_DIGEST_SUMMARY_CONTENT_RECOVERY_MAX_INPUT_CHARS", "summary_content_recovery_input_cap", 3),
])
def test_runtime_selected_system_input_cap_holds_without_transport(cfg, monkeypatch, stage, limit_name, reason, calls):
    client = PolicyClient()
    # At less than the selected system alone, no nonempty payload can fit.
    monkeypatch.setattr(digest, limit_name, len(digest.digest_system_for_policy(stage, summary_policy=BOUNDED)) - 1)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = extract(hy, client)
        assert result.parse_failed and result.failure_reason == reason
        assert len(client.calls) == calls
        assert result.source_sha256 is result.covered_message_id is None


def test_loaded_policy_module_changes_digest_identity(monkeypatch):
    client = StubLLMClient(default="[]")
    before = semantic_generation_suffix("digest", client)
    monkeypatch.setattr(policy, "BOUNDED_HIGHLIGHTS_CONTRACT", policy.BOUNDED_HIGHLIGHTS_CONTRACT + " Changed policy.")
    assert semantic_generation_suffix("digest", client) != before
    # Profile/facts bind runner dispatch, so configuration/dispatch changes can
    # conservatively invalidate them too; no assertion of tier isolation here.


def test_policy_switch_rebuilds_then_same_policy_is_current_and_source_stays_queryable(cfg):
    client = PolicyClient("Checks passed.", first=(), with_items=False)
    config = replace(_config(cfg, False), rerank_message_hits=False)
    with closing(HyMem(config, llm=client)) as hy:
        mid = hy.log_message("policy", "assistant", "Checks passed; the archived receipt code is cobaltneedle.", created_at="2020-01-01")
        hy.close_session("policy")
        hy.dream()
        old = hy.conn.execute("SELECT digest_published_generation FROM sessions WHERE id='policy'").fetchone()[0]
        assert old and "summary-policy=" not in old
        proof_before = [tuple(row) for row in hy.conn.execute("SELECT * FROM message_retention_coverage ORDER BY message_id")]
        hy.config = replace(config, digest_summary_policy=BOUNDED)
        assert hy.dream_status()["pending_digests"] == 1
        hy.dream()
        row = hy.conn.execute("SELECT digest_published_generation,auto_summary FROM sessions WHERE id='policy'").fetchone()
        assert "|summary-policy=bounded_highlights_v1|" in row[0] and row[0] != old
        assert row[1] == "Checks passed." and "cobaltneedle" not in row[1]
        assert hy.dream_status()["pending_digests"] == hy.dream_status()["quarantined_digests"] == 0
        count = len(client.calls)
        hy.dream()
        assert len(client.calls) == count
        assert [tuple(row) for row in hy.conn.execute("SELECT * FROM message_retention_coverage ORDER BY message_id")] == proof_before
        assert [hit.message_id for hit in hy.augment("cobaltneedle", source_session_id="policy").message_hits] == [mid]
        with db.transaction(hy.conn):
            assert prune_messages(hy.conn, replace(hy.config, message_retention_days=1)) == 1
        assert [hit.message_id for hit in hy.augment("cobaltneedle", source_session_id="policy").message_hits] == [mid]
        hy.config = config
        assert hy.dream_status()["pending_digests"] == 1


def test_failed_bounded_generation_quarantines_without_replacing_legacy_publication(cfg):
    client = PolicyClient("Checks passed.", first=(), with_items=False)
    config = replace(_config(cfg, False), digest_extraction_max_attempts=1)
    with closing(HyMem(config, llm=client)) as hy:
        hy.log_message("policy", "assistant", "Checks passed.")
        hy.close_session("policy")
        hy.dream()
        before = tuple(hy.conn.execute("SELECT digest_published_generation,auto_summary FROM sessions WHERE id='policy'").fetchone())
        hy.config = replace(config, digest_summary_policy=BOUNDED)
        client.first = client.second = (("summary_content", 0, "unsupported"),)
        report = hy.dream()
        assert report.digest_failures == report.digest_quarantined == 1
        assert tuple(hy.conn.execute("SELECT digest_published_generation,auto_summary FROM sessions WHERE id='policy'").fetchone()) == before
        assert hy.dream_status()["quarantined_digests"] == 1
        count = len(client.calls)
        hy.dream()
        assert len(client.calls) == count


def test_switching_policy_discards_private_walk_and_restarts_exact_source(cfg):
    client = PolicyClient("Checks passed.", first=(), with_items=False)
    config = replace(_config(cfg, False), dream_digest_max_chars=500)
    with closing(HyMem(config, llm=client)) as hy:
        source = "Checks passed; an independent receipt is retained. " * 25
        hy.log_message("policy", "assistant", source)
        hy.close_session("policy")
        first = hy.dream()
        old = hy.conn.execute("SELECT digest_cursor_prompt_version,digest_cursor_offset,digest_published_generation FROM sessions WHERE id='policy'").fetchone()
        assert first.budget_exhausted and old[1] > 0 and old[2] is None
        hy.config = replace(config, digest_summary_policy=BOUNDED)
        assert hy.dream_status()["pending_digests"] == 1
        second = hy.dream()
        new = hy.conn.execute("SELECT digest_cursor_prompt_version,digest_cursor_offset,digest_published_generation FROM sessions WHERE id='policy'").fetchone()
        assert second.budget_exhausted and new[0] != old[0]
        assert new[1] == old[1] and new[2] is None
        rows = hy.conn.execute("SELECT generation,cursor_before_message_id,cursor_before_partial_message_id,cursor_before_offset FROM digest_staging").fetchall()
        assert [tuple(row) for row in rows] == [(new[0], None, None, 0)]
        assert covered_messages_after(hy.conn, "policy", None)[0].content == source
        for _ in range(10):
            hy.dream()
            if hy.dream_status()["pending_digests"] == 0:
                break
        assert hy.dream_status()["pending_digests"] == 0
        assert hy.dream_status()["malformed_digests"] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        published = hy.conn.execute("SELECT digest_published_generation FROM sessions WHERE id='policy'").fetchone()[0]
        assert published == new[0]


def test_policy_implementation_change_during_call_cannot_publish(cfg, monkeypatch):
    changed = False

    def change_policy(request):
        nonlocal changed
        if not changed and request.system.startswith("You analyze one conversation session"):
            changed = True
            monkeypatch.setattr(policy, "BOUNDED_HIGHLIGHTS_CONTRACT",
                                policy.BOUNDED_HIGHLIGHTS_CONTRACT + " Changed in flight.")

    client = PolicyClient("Checks passed.", first=(), with_items=False, on_call=change_policy)
    config = replace(_config(cfg, False), digest_summary_policy=BOUNDED)
    with closing(HyMem(config, llm=client)) as hy:
        hy.log_message("policy", "assistant", "Checks passed.")
        hy.close_session("policy")
        report = hy.dream()
        assert changed and report.digest_failures == 1
        assert tuple(hy.conn.execute("SELECT digest_published_generation,auto_summary,digest_cursor_message_id FROM sessions WHERE id='policy'").fetchone()) == (None, None, None)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
