"""Bounded source-linked summary repair; synthetic verdicts are not accuracy proof."""
from contextlib import closing
from dataclasses import asdict, replace
import json

import pytest

from hymem import HyMem
from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline, current_deadline
from hymem.dreaming import digest, summary
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.producer import canonical_module_sha256
from tests.digest_verification_fixtures import synthetic_fidelity_result, synthetic_summary_issues
from tests.test_digest_summary_contract import _config
from tests.test_digest_summary_verification import _SummaryClient, _extract, _seed


_DEFAULT = object()
REJECTED = "The user visited the canyon and harbor, then received route advice."
REPAIRED = "The user visited the canyon before the harbor, missed the valley and received route advice."


class _ContentRecoveryClient(_SummaryClient):
    def __init__(self, candidate=REJECTED, *, repaired=_DEFAULT, compacted=None,
                 first=(("summary_content", 0, "unsupported"),), second=(),
                 first_raw=_DEFAULT, second_raw=_DEFAULT, with_items=True, on_call=None):
        super().__init__(candidate, compacted=compacted, with_items=with_items)
        self.repaired = {"summary": REPAIRED} if repaired is _DEFAULT else repaired
        self.first = first
        self.second = second
        self.first_raw = first_raw
        self.second_raw = second_raw
        self.on_call = on_call
        self.deadlines = []
        self.verifications = 0

    def complete(self, request):
        self.deadlines.append(current_deadline())
        if self.on_call is not None:
            self.on_call(request)
        if request.system == digest._DIGEST_SUMMARY_DIAGNOSIS_SYSTEM:
            self.calls.append(request)
            return json.dumps({"issues": synthetic_summary_issues(json.loads(request.user))})
        if request.system.startswith("You repair one rolling conversation summary"):
            self.calls.append(request)
            if isinstance(self.repaired, BaseException):
                raise self.repaired
            return json.dumps(self.repaired) if isinstance(self.repaired, (dict, list)) else self.repaired
        if request.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM:
            self.calls.append(request)
            payload = json.loads(request.user)
            return json.dumps({
                "summary_format": [{"index": 0, "verdict": "supported"}],
                "episode_format": [{"index": i, "verdict": "supported"} for i in range(len(payload["items"]))],
            })
        raw = super().complete(request)
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            self.verifications += 1
            changes = self.first if self.verifications == 1 else self.second
            explicit = self.first_raw if self.verifications == 1 else self.second_raw
            if explicit is not _DEFAULT:
                return explicit
            response = json.loads(raw)
            for family, index, verdict in changes:
                response[family][index]["verdict"] = verdict
            return json.dumps(response)
        return raw


def _is_recovery(request):
    return request.system.startswith("You repair one rolling conversation summary")


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("first_verdict", ["unsupported", "uncertain"])
@pytest.mark.parametrize("compacted", [False, True])
def test_source_linked_repair_preserves_items_and_reverifies_every_family(
    cfg, granular, first_verdict, compacted,
):
    client = _ContentRecoveryClient(
        "REJECTED_PRIMARY_MARKER " * 40 if compacted else REJECTED,
        compacted=REJECTED if compacted else None,
        first=(("summary_content", 0, first_verdict),),
    )
    prior = "Earlier photography and winery topics remain relevant."
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        last = _seed(hy, "The canyon trip came before the harbor stop; the valley was missed. Route advice was supplied.")
        result = _extract(hy, client, granular=granular, prior=prior)
        assert not result.parse_failed and result.summary == REPAIRED
        assert result.covered_message_id == last and result.source_sha256 is not None
        assert len(client.calls) == (7 if compacted else 6)
        assert client.verifications == 2
        recovery = [call for call in client.calls if _is_recovery(call)]
        assert len(recovery) == 1
        repair_payload = json.loads(recovery[0].user)
        assert repair_payload["original_generation_input"] == client.calls[0].user
        assert replace(recovery[0], system=client.calls[0].system, user=client.calls[0].user) == client.calls[0]
        assert repair_payload["rejection_diagnostics"]["candidate_summary"] == REJECTED
        assert "REJECTED_PRIMARY_MARKER" not in recovery[0].user
        verification_calls = [call for call in client.calls if call.system == digest._DIGEST_FIDELITY_SYSTEM]
        first, final = [json.loads(call.user) for call in verification_calls]
        assert first["summary_item"]["candidate_summary"] == REJECTED
        assert final["summary_item"]["candidate_summary"] == REPAIRED
        first["summary_item"]["candidate_summary"] = REPAIRED
        first["summary_item"]["candidate_raw_summary"] = REPAIRED
        assert final == first
        assert result.episodes.items == client.primary["episodes"]
        assert result.procedures.items == [item["candidate"] for item in final["procedure_items"]]
        assert tuple(hy.conn.execute(
            "SELECT auto_summary,digest_cursor_message_id FROM sessions WHERE id='summary-verification'",
        ).fetchone()) == (None, None)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


@pytest.mark.parametrize("granular", [False, True])
def test_recomposition_keeps_exact_partial_window_and_source_authority(cfg, granular):
    client = _ContentRecoveryClient()
    control = _ContentRecoveryClient(REPAIRED, first=())
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        _seed(hy, "The canyon came before the harbor. " * 100)
        kwargs = dict(max_tokens=2048, max_chars=650, granular=granular,
                      max_episodes=8 if granular else None)
        result = digest.extract_session_digest(hy.conn, "summary-verification", client, **kwargs)
        expected = digest.extract_session_digest(hy.conn, "summary-verification", control, **kwargs)
        assert not result.parse_failed and not result.caught_up
        assert result.partial_message_id is not None and result.next_message_offset > 0
        assert asdict(result) == asdict(expected)
        assert len(client.calls) == 6 and len(control.calls) == 3


@pytest.mark.parametrize("family,reason", [
    ("episode_titles", "episode_title"), ("episode_content", "episode_content"),
    ("procedures", "procedure_content"), ("summary_content", "summary_content"),
])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_second_semantic_failure_holds_without_repair_loop_or_format_override(cfg, family, reason, verdict):
    client = _ContentRecoveryClient(second=((family, 0, verdict),))
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = _extract(hy, client)
        assert result.parse_failed and result.failure_reason == reason + "_" + verdict
        assert result.failure_stage == "fidelity_reverification" and len(client.calls) == 5
        assert result.episodes.items == result.procedures.items == []
        assert result.summary is result.covered_message_id is result.source_sha256 is None
        assert result.episode_input_items == result.episode_rejected_items == 1
        assert result.procedure_input_items == result.procedure_rejected_items == 1
        assert not any(call.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM for call in client.calls)


@pytest.mark.parametrize("family", ["episode_titles", "episode_content", "procedures"])
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_any_initial_item_failure_blocks_summary_recomposition(cfg, family, verdict):
    client = _ContentRecoveryClient(first=(("summary_content", 0, "unsupported"), (family, 0, verdict)))
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = _extract(hy, client)
        assert result.parse_failed and len(client.calls) == 2
        assert not any(_is_recovery(call) for call in client.calls)


@pytest.mark.parametrize("family", list(synthetic_fidelity_result(1, 1)))
@pytest.mark.parametrize("phase", ["first", "second"])
def test_malformed_verdict_group_never_gains_authority_from_recomposition(cfg, family, phase):
    response = synthetic_fidelity_result(1, 1)
    response["summary_content"][0]["verdict"] = "unsupported"
    response[family] = []
    client = _ContentRecoveryClient(**{phase + "_raw": json.dumps(response)})
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = _extract(hy, client)
        assert result.failure_reason == "fidelity_shape_failure"
        assert len(client.calls) == (2 if phase == "first" else 5)
        assert result.covered_message_id is result.source_sha256 is None


@pytest.mark.parametrize("raw,reason", [
    ("not JSON", "parse_failure"), ([], "shape_failure"),
    ({"summary": REPAIRED, "episodes": []}, "shape_failure"),
    ({"summary": None}, "summary_shape_failure"),
    ({"summary": ""}, "summary_validation_failure"),
    ({"summary": " \n "}, "summary_validation_failure"),
    ({"summary": "tiny"}, "summary_validation_failure"),
    ({"summary": "🧭" * 501}, "summary_output_cap"),
])
def test_invalid_recomposition_holds_without_reverification_or_truncation(cfg, raw, reason):
    client = _ContentRecoveryClient(repaired=raw)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = _extract(hy, client)
        assert result.failure_reason == "summary_content_recovery_" + reason
        assert result.failure_stage == "summary_content_recovery"
        assert len(client.calls) == 4 and client.verifications == 1
        assert result.summary is result.covered_message_id is result.source_sha256 is None
        assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)


@pytest.mark.parametrize("candidate,prior,repaired", [
    (REJECTED, None, REJECTED), (REJECTED, None, "  " + REJECTED + "  "),
    ("", REJECTED, REJECTED), (" \n ", REJECTED + "  ", REJECTED),
])
def test_effectively_unchanged_recomposition_is_not_rerolled_again(cfg, candidate, prior, repaired):
    client = _ContentRecoveryClient(candidate, repaired={"summary": repaired})
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        result = _extract(hy, client, prior=prior)
        assert result.failure_reason == "summary_content_unsupported"
        assert len(client.calls) == 4 and client.verifications == 1
        assert result.covered_message_id is result.source_sha256 is None


def test_empty_primary_may_be_replaced_only_after_new_summary_is_fully_verified(cfg):
    client = _ContentRecoveryClient("", with_items=False)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        last = _seed(hy, "The canyon came before the harbor; route advice was supplied.")
        result = _extract(hy, client, prior="Earlier photography advice.")
        assert result.summary == REPAIRED and result.covered_message_id == last
        first, final = [json.loads(call.user)["summary_item"] for call in client.calls
                        if call.system == digest._DIGEST_FIDELITY_SYSTEM]
        assert first["candidate_is_noop"] and not final["candidate_is_noop"]
        assert first["prior_derived_summary"] == final["prior_derived_summary"]


def test_length_recovery_diagnosis_recomposition_reverification_and_format_are_bounded_at_seven(cfg):
    client = _ContentRecoveryClient("too long " * 100, compacted=REJECTED)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "The canyon came before the harbor; route advice was supplied.")
        result = _extract(hy, client)
        assert not result.parse_failed and result.summary == REPAIRED
        assert len(client.calls) == 7
        assert sum(_is_recovery(call) for call in client.calls) == 1
        assert client.verifications == 2
        assert client.calls[-1].system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM
        assert json.loads(client.calls[-1].user)["summary_item"]["candidate_summary"] == REPAIRED


@pytest.mark.parametrize("error", [RuntimeError("offline transport failure"), DeadlineExceeded("expired"), KeyboardInterrupt()])
def test_recomposition_transport_errors_keep_exact_stage_and_controls_escape(cfg, error):
    client = _ContentRecoveryClient(repaired=error)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        if isinstance(error, Exception):
            with pytest.raises(digest.DigestCompletionError) as raised:
                _extract(hy, client)
            assert raised.value.failure_stage == "summary_content_recovery" and raised.value.__cause__ is error
        else:
            with pytest.raises(type(error)) as raised:
                _extract(hy, client)
            assert raised.value is error
        assert len(client.calls) == 4


@pytest.mark.parametrize("expires,expected", [(2., 2), (3., 3), (4., 4), (5., 5), (6., 6), (7., 6)])
def test_content_recovery_has_no_new_deadline_and_never_accepts_a_late_result(cfg, expires, expected):
    clock = [0.]
    deadline = MonotonicDeadline(expires, clock=lambda: clock[0])
    def advance(_request):
        clock[0] += 1.
    inner = _ContentRecoveryClient(on_call=advance)
    client = DeadlineBoundLLMClient(inner, deadline)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy, "Run checks before deployment.")
        if expires <= 6.:
            with pytest.raises(DeadlineExceeded):
                _extract(hy, client)
        else:
            assert not _extract(hy, client).parse_failed
        assert len(inner.calls) == expected
        assert all(seen is deadline for seen in inner.deadlines)
        assert current_deadline() is None


@pytest.mark.parametrize("field", ["version", "prompt"])
def test_content_recovery_policy_invalidates_digest_generation_only(monkeypatch, field):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    standalone = canonical_module_sha256(summary)
    name = "DIGEST_SUMMARY_CONTENT_RECOVERY_VERSION" if field == "version" else "_DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE"
    monkeypatch.setattr(digest, name, "changed")
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert before["digest"] != after["digest"]
    assert before["facts"] == after["facts"] and before["profile"] == after["profile"]
    assert extraction_contract_identity("v20") == phase1
    assert canonical_module_sha256(summary) == standalone


def test_recomposition_prompt_reuses_all_shared_summary_contracts_without_incident_nouns():
    prompt = digest._DIGEST_SUMMARY_CONTENT_RECOVERY_TEMPLATE
    for text in (digest.SESSION_DIGEST_SUMMARY_ALLOCATION, digest.SESSION_DIGEST_CATEGORY_RELATIONS,
                 digest.SESSION_DIGEST_CLAIM_SCOPE, digest.SESSION_DIGEST_SUMMARY_SENTENCE):
        assert text in prompt
    for text in ("material temporal sequence", "corrections", "actual", "intended", "not source evidence"):
        assert text in prompt
    for forbidden in ("Big Sur", "Monterey", "Santa Ynez", "canyon", "harbor"):
        assert forbidden not in prompt
