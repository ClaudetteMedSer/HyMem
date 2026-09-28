"""Independent policy wiring controls; scripted verdicts do not prove semantics."""
from contextlib import closing
from dataclasses import replace
import hashlib
import json

import pytest

from hymem import HyMem
from hymem.dreaming import digest
from hymem.dreaming.summary_policy import (
    LEGACY_COMPLETE_V1 as LEGACY, BOUNDED_HIGHLIGHTS_V1 as BOUNDED,
    GENERATION, COMPACTION, CONTENT_REPAIR, VERIFICATION, DIAGNOSIS,
)
from tests.test_digest_summary_content_recovery import (
    _ContentRecoveryClient, REJECTED, REPAIRED,
)
from tests.test_digest_summary_contract import _config
from tests.test_digest_summary_verification import _seed


# Captured by root before the runtime integration, not calculated from new code.
LEGACY_HASHES = {
    "SESSION_DIGEST_SYSTEM": "40788790eca2e9ee5ae17d38eece0677ca2f5867538c35441a3dfbd41819a63b",
    "SESSION_DIGEST_GRANULAR_SYSTEM": "88e12dcc3dda5bc36c351dc9c0cc7adaab8dceb21c675ea3008afe28407dcaef",
    "_DIGEST_FIDELITY_SYSTEM": "07fe3567b0c0e385b452908583f0ac6530c979095cf04aabdd5dc1fdb85042ba",
    "_DIGEST_SUMMARY_DIAGNOSIS_SYSTEM": "ddd4a2bdadf9fcb3764cb56bc23ec911653b69c9476dc02d5061429b80ee22fa",
    "_DIGEST_SUMMARY_RECOVERY_TEMPLATE": "801e066cdaaf4ae7e411eddca5d8e9c291390f79a8cd5da6f8e2d935a52ebc64",
    "_DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE": "3293abb0de215fb889195eed0e877454819d2f0d010f6958dbeb9ac7b7f8fdd2",
    "_DIGEST_FORMAT_ADJUDICATION_SYSTEM": "ef3c806811a0f32886f84fd9e9fd3ac6786006e9cd3bff109d4ef0e518d53261",
}


@pytest.mark.parametrize("name,expected", LEGACY_HASHES.items())
def test_preintegration_legacy_system_bytes_are_unchanged(name, expected):
    assert hashlib.sha256(getattr(digest, name).encode()).hexdigest() == expected


class PolicyScript:
    """Map only systems to an existing deterministic state-machine script.

    Actual requests are recorded separately. No approval depends on candidate
    meaning; this deliberately tests plumbing, not model performance.
    """
    def __init__(self, *, policy, granular=False, compact=False, **kwargs):
        self.policy = policy
        self.granular = granular
        candidate = "overlong primary " * 60 if compact else REJECTED
        self.inner = _ContentRecoveryClient(
            candidate, compacted=REJECTED if compact else None, **kwargs)
        self.calls = []
        self.system_map = {
            digest.digest_system_for_policy(stage, summary_policy=policy,
                granular=granular, returned_chars=len(candidate.strip())):
            digest.digest_system_for_policy(stage, summary_policy=LEGACY,
                granular=granular, returned_chars=len(candidate.strip()))
            for stage in (GENERATION, COMPACTION, CONTENT_REPAIR, VERIFICATION, DIAGNOSIS)
        }

    def complete(self, request):
        self.calls.append(request)
        mapped = self.system_map.get(request.system)
        if mapped is None:
            assert request.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
            mapped = request.system
        return self.inner.complete(replace(request, system=mapped))


def extract(hy, client, *, prior="Earlier backup checks remain enabled."):
    return digest.extract_session_digest(
        hy.conn, "summary-verification", client, max_tokens=2048,
        max_chars=10000, granular=client.granular,
        max_episodes=8 if client.granular else None,
        prior_summary=prior, summary_policy=client.policy,
    )


@pytest.mark.parametrize("policy", [LEGACY, BOUNDED])
@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("repair", [False, True])
def test_every_stage_uses_selected_policy_without_altering_sources_or_items(
    cfg, policy, granular, compact, repair,
):
    client = PolicyScript(policy=policy, granular=granular, compact=compact,
        first=(("summary_content", 0, "unsupported"),) if repair else ())
    with closing(HyMem(replace(_config(cfg, granular), digest_summary_policy=policy), llm=client)) as hy:
        last = _seed(hy, "The canyon trip preceded the harbor stop; the valley was missed and route advice was supplied.")
        source_before = [tuple(row) for row in hy.conn.execute("SELECT * FROM message_retention_coverage")]
        result = extract(hy, client)
        assert not result.parse_failed
        assert result.summary == (REPAIRED if repair else REJECTED)
        assert result.covered_message_id == last and result.source_sha256
        assert len(client.calls) == 3 + int(compact) + 3 * int(repair)
        assert len(client.calls) <= 7
        assert client.calls[-1].system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
        for actual, scripted in zip(client.calls, client.inner.calls):
            assert replace(actual, system=scripted.system) == scripted
            if actual is not client.calls[-1]:
                assert (BOUNDED in actual.system) == (policy == BOUNDED)
        assert [tuple(row) for row in hy.conn.execute("SELECT * FROM message_retention_coverage")] == source_before
        assert result.episodes.items == client.inner.primary["episodes"]
        verifications = [json.loads(c.user) for c in client.calls
            if c.system == digest.digest_system_for_policy(VERIFICATION, summary_policy=policy)]
        assert len(verifications) == 1 + int(repair)
        assert all(v["schema"] == "digest-fidelity-decisions-v9" for v in verifications)
        assert all(v["items"] == verifications[0]["items"] for v in verifications)
        assert all(v["procedure_items"] == verifications[0]["procedure_items"] for v in verifications)
        assert all(v["source_catalog"] == verifications[0]["source_catalog"] for v in verifications)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert tuple(hy.conn.execute("SELECT auto_summary,digest_cursor_message_id FROM sessions WHERE id='summary-verification'").fetchone()) == (None, None)


@pytest.mark.parametrize("policy", [LEGACY, BOUNDED])
@pytest.mark.parametrize("family", ["episode_titles", "episode_content", "procedures", "summary_content"])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_post_repair_veto_never_acquires_cursor_or_publication(cfg, policy, family, verdict):
    client = PolicyScript(policy=policy, second=((family, 0, verdict),))
    with closing(HyMem(replace(_config(cfg, False), digest_summary_policy=policy), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = extract(hy, client)
        assert result.parse_failed and result.failure_stage == "fidelity_reverification"
        assert result.failure_reason.endswith("_" + verdict)
        assert result.summary is result.source_sha256 is result.covered_message_id is None
        assert result.episodes.items == result.procedures.items == []
        assert len(client.calls) == 5
        assert not any(c.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM for c in client.calls)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


@pytest.mark.parametrize("policy", [LEGACY, BOUNDED])
def test_fidelity_cap_counts_actual_selected_system_exactly(monkeypatch, policy):
    packet = {"text": "Unicode 🧬 and escaped quote \""}
    system = digest.digest_system_for_policy(VERIFICATION, summary_policy=policy)
    encoded = digest._encode_digest_fidelity_payload(packet, system=system)
    assert json.loads(encoded) == packet
    total = len(system) + len(encoded)
    monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", total)
    assert digest._encode_digest_fidelity_payload(packet, system=system) == encoded
    monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", total - 1)
    assert digest._encode_digest_fidelity_payload(packet, system=system) is None


@pytest.mark.parametrize("bad", [None, "", "bounded", "bounded_highlights_v2", " legacy_complete_v1", 1, True])
def test_unknown_policy_fails_before_store_or_provider_use(bad):
    with pytest.raises(ValueError):
        digest.extract_session_digest(None, "not-opened", None,
            max_tokens=2048, max_chars=10000, summary_policy=bad)


def test_generation_and_item_rules_are_scoped_replacements():
    marker = "For summary_content, inspect summary_item.candidate_summary, "
    for stage in (VERIFICATION, DIAGNOSIS):
        old = digest.digest_system_for_policy(stage, summary_policy=LEGACY)
        new = digest.digest_system_for_policy(stage, summary_policy=BOUNDED)
        assert new.split(marker)[0] == old.split(marker)[0]
        assert "Do not lose important new outcomes or all still-relevant prior topics" not in new
        assert "Do NOT" in new and "semantic" in new
    for granular in (False, True):
        old = digest.digest_system_for_policy(GENERATION, granular=granular)
        new = digest.digest_system_for_policy(GENERATION, summary_policy=BOUNDED, granular=granular)
        before, after = old.split(digest._SESSION_DIGEST_SUMMARY_CONTRACT)
        assert new.startswith(before) and new.endswith(after)
        assert "covering BOTH the prior automatic summary and the new material" not in new


def test_historical_and_bounded_generation_shapes_do_not_match_each_other():
    args = dict(prompt_version="v1", episode_prompt_version=None,
                max_chars=12000, max_tokens=3072, max_episodes=None)
    legacy = digest.digest_config_version(**args)
    bounded = digest.digest_config_version(**args, summary_policy=BOUNDED)
    assert legacy == "lossless-digest-v2|prompt=v1|episodes=blob|chars=12000|tokens=3072|episode-cap=blob"
    assert bounded == legacy + "|summary-policy=" + BOUNDED
    for config in (legacy, bounded):
        generation = config + "|walk=" + "a" * 32
        assert digest.digest_generation_is_recognized(generation)
        assert digest.digest_generation_matches_config(generation, config)
        assert not digest.digest_generation_matches_config(generation, bounded if config == legacy else legacy)
    assert not digest.digest_generation_is_recognized(bounded.replace(BOUNDED, "bounded_highlights_v2") + "|walk=" + "a" * 32)
