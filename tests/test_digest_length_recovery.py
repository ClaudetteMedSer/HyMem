"""Bounded output-side correction cannot weaken the digest publication gate."""
from contextlib import closing
from dataclasses import asdict, replace
import json

import pytest

from hymem import HyMem
from hymem.core import db
from hymem.deadline import DeadlineExceeded, MonotonicDeadline
from hymem.dreaming import digest, runner
from hymem.dreaming.lossless import covered_messages_after
from hymem.dreaming.retention import prune_messages
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import StubLLMClient
from tests.digest_verification_fixtures import synthetic_fidelity_result, synthetic_format_result
from tests.test_digest_publication import _finish, _state
from tests.test_digest_summary_contract import _config, _extract, _payload, _seed
from tests.test_lossless_digest import RollingLLM, _quiet_cfg


class ScriptedDigestLLM(RollingLLM):
    """A stable-identity client with bounded, synthetic per-call responses."""

    def __init__(self, outputs=()):
        super().__init__(emit_slice_artifacts=True)
        self.outputs = list(outputs)
        self.digest_requests = []

    def complete(self, request):
        if request.system.startswith("You compact one rolling conversation summary"):
            self.calls.append(request)
            self.digest_requests.append(request)
            raw = _summary_payload(json.loads(self.last_digest_raw)["summary"])
            if self.outputs:
                output = self.outputs.pop(0)
                if isinstance(output, BaseException):
                    raise output
                return output(raw) if callable(output) else output
            return raw
        raw = super().complete(request)
        if request.system.startswith(("You analyze one conversation session",
                                      "You re-read one conversation session")):
            self.last_digest_raw = raw
            self.digest_requests.append(request)
            if self.outputs:
                output = self.outputs.pop(0)
                if isinstance(output, BaseException):
                    raise output
                return output(raw) if callable(output) else output
        return raw


def _overcap(raw):
    data = json.loads(raw)
    data["summary"] = "Long synthetic summary. " * 24
    return json.dumps(data)


def _summary_payload(text):
    return json.dumps({"summary": text})


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("length", [500, 501])
@pytest.mark.parametrize("character", ["s", "🧬", "\u0301"], ids=["ascii", "emoji", "combining"])
def test_exact_unicode_bound_and_one_length_correction(cfg, granular, length, character):
    original = " \n" + character * length + "\t "
    corrected = "Rephrased Alpha history and new Beta work concisely."
    llm = ScriptedDigestLLM([_payload(original), _summary_payload(corrected)])
    prior = "Prior Alpha topic with exact whitespace.  \nDo not erase it."
    source = "New Beta topic with Unicode 🧬 and exact whitespace.  \nKeep it."
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy, source)
        before = [asdict(row) for row in covered_messages_after(hy.conn, "bounded-summary", None)]
        result = _extract(hy, llm, granular=granular, prior_summary=prior)
        assert result is not None and not result.parse_failed
        assert result.summary == (character * length if length == 500 else corrected)
        assert result.covered_message_id == last_id and result.caught_up
        assert len(llm.digest_requests) == (1 if length == 500 else 2)
        assert [asdict(row) for row in covered_messages_after(hy.conn, "bounded-summary", None)] == before
        if length == 501:
            first, second = llm.digest_requests
            assert second.user == first.user
            assert prior in second.user and source in second.user
            assert "501 Unicode code points" in second.system
            assert "hard maximum is 500" in second.system
            assert "Aim for 350 characters" in second.system
            assert "Previous model output is not source evidence" in second.system
            assert "Preserve earlier topics" in second.system
            assert character * length not in second.system + second.user
            assert second.system.startswith("You compact one rolling conversation summary")
            assert not second.system.startswith(first.system)
            assert replace(second, system=first.system) == first


def _bad_payload(kind, raw):
    value = json.loads(raw)
    if kind == "parse":
        return "not JSON"
    if kind == "truncated":
        return '{"episodes":[],"summary":"unfinished'
    if kind == "array":
        return "[]"
    if kind == "extra-key":
        value["error"] = "unexpected"
    elif kind == "summary-type":
        value["summary"] = 42
    elif kind == "empty":
        value["summary"] = " \n\t "
    elif kind == "episode-type":
        value["episodes"][0]["outcome"] = "invented"
    elif kind == "episode-citation":
        value["episodes"][0]["chunk_ids"] = ["msgcov_not_in_source"]
    elif kind == "episode-key":
        value["episodes"][0]["invented"] = "extra"
    elif kind == "procedure-type":
        value["procedures"][0]["steps"][0]["order"] = True
    elif kind == "procedure-citation":
        value["procedures"][0]["chunk_ids"] = ["msgcov_not_in_source"]
    elif kind == "procedure-key":
        value["procedures"][0]["invented"] = "extra"
    elif kind == "overcap":
        return _overcap(raw)
    return json.dumps(value)


_BAD_CASES = [
    ("parse", "parse_failure"), ("truncated", "output_truncated"),
    ("array", "shape_failure"), ("extra-key", "shape_failure"),
    ("summary-type", "summary_shape_failure"),
    ("episode-type", "episode_validation_failure"),
    ("episode-citation", "episode_validation_failure"),
    ("episode-key", "episode_validation_failure"),
    ("procedure-type", "procedure_validation_failure"),
    ("procedure-citation", "procedure_validation_failure"),
    ("procedure-key", "procedure_validation_failure"),
]


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("correction,reason", [
    ("not JSON", "parse_failure"),
    ('{"summary":"unfinished', "output_truncated"),
    ("[]", "shape_failure"), ("{}", "shape_failure"),
    (_summary_payload(42), "summary_shape_failure"),
    (_summary_payload(None), "summary_shape_failure"),
    (_summary_payload(" \n\t "), "summary_validation_failure"),
    (_summary_payload('"tiny"'), "summary_validation_failure"),
    (_summary_payload("s" * 501), "summary_output_cap"),
    (_payload("Must not replace valid primary items."), "shape_failure"),
    ('{"summary":"One valid summary.","episodes":[]}', "shape_failure"),
    ('{"summary":"One valid summary.","procedures":[]}', "shape_failure"),
    ('{"summary":"One valid summary.","error":"none"}', "shape_failure"),
    ('{"summary":"One valid summary.","summary":"Duplicate key."}', "parse_failure"),
    ('Here is JSON: {"summary":"One valid summary."}', "parse_failure"),
])
def test_compaction_rejects_invalid_summary_or_replacement_items_without_a_third_call(
    cfg, granular, correction, reason,
):
    llm = ScriptedDigestLLM([_overcap, correction, _payload("Must not be used.")])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        _seed(hy)
        result = _extract(hy, llm, granular=granular)
        assert result is not None and result.parse_failed
        assert result.failure_reason == reason
        assert result.summary is None and result.source_sha256 is None
        assert result.episodes.items == result.procedures.items == []
        assert result.covered_message_id is None and result.next_message_offset == 0
        assert not result.caught_up
        assert len(llm.digest_requests) == 2 and len(llm.outputs) == 1


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("kind,reason", _BAD_CASES)
def test_original_non_length_failure_does_not_spend_a_correction(cfg, granular, kind, reason):
    # The long summary deliberately coexists with the item/schema failure.
    # Summary repair must not hide a malformed procedure validated afterwards.
    llm = ScriptedDigestLLM([lambda raw: _bad_payload(kind, _overcap(raw))])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        _seed(hy)
        result = _extract(hy, llm, granular=granular)
        assert result is not None and result.parse_failed
        assert result.failure_reason == reason
        assert len(llm.digest_requests) == 1


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
def test_original_valid_empty_is_not_repaired(cfg, granular):
    llm = ScriptedDigestLLM([_payload("")])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        last_id = _seed(hy)
        result = _extract(hy, llm, granular=granular)
        assert result is not None and not result.parse_failed and result.summary is None
        assert result.covered_message_id == last_id and len(llm.digest_requests) == 1


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("text", ['"' * 501, '"' * 251 + "tiny" + '"' * 251, '"tiny"', '"' * 500])
def test_normalization_invalid_summary_is_not_a_length_only_repair(cfg, granular, text):
    llm = ScriptedDigestLLM([_payload(text), _payload("Must not be requested.")])
    with closing(HyMem(_config(cfg, granular), llm=llm)) as hy:
        _seed(hy)
        result = _extract(hy, llm, granular=granular)
        assert result is not None and result.parse_failed
        assert result.failure_reason == "summary_validation_failure"
        assert result.summary is None and result.covered_message_id is None
        assert len(llm.digest_requests) == 1 and len(llm.outputs) == 1


@pytest.mark.parametrize("overcap_summary", [False, True])
def test_granular_episode_cap_is_not_masked_by_summary_repair(cfg, overcap_summary):
    llm = ScriptedDigestLLM([_overcap] if overcap_summary else [])
    with closing(HyMem(_config(cfg, True), llm=llm)) as hy:
        _seed(hy)
        result = digest.extract_session_digest(
            hy.conn, "bounded-summary", llm, max_tokens=2048, max_chars=10000,
            granular=True, max_episodes=0,
        )
        assert result is not None and result.failure_reason == "episode_output_cap"
        assert len(llm.digest_requests) == 1


def _cursor(hy):
    return tuple(hy.conn.execute(
        "SELECT digest_cursor_message_id,digest_cursor_partial_message_id,"
        "digest_cursor_offset,digest_cursor_prompt_version,digested_message_id "
        "FROM sessions WHERE id='x'",
    ).fetchone())


def _staging(hy):
    return [tuple(row) for row in hy.conn.execute("SELECT * FROM digest_staging ORDER BY slice_key")]


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("failure", ["original-exception", "correction-exception", "overcap", "empty", "deadline", "deadline-before-correction"])
def test_failed_or_expired_correction_preserves_cursor_and_existing_stage(cfg, granular, failure):
    llm = ScriptedDigestLLM()
    config = _quiet_cfg(cfg, dream_digest_max_chars=400, episode_granularity_enabled=granular)
    with closing(HyMem(config, llm=llm)) as hy:
        hy.log_message("x", "assistant", "alpha durable history " + "old material " * 45)
        hy.close_session("x")
        _finish(hy)
        published = _state(hy)
        hy.log_message("x", "assistant", "beta new work " + "new material " * 60)
        assert hy.dream().budget_exhausted
        before_cursor, before_stage = _cursor(hy), _staging(hy)
        before_source = covered_messages_after(hy.conn, "x", None)
        assert before_stage and _state(hy) == published
        before_calls = len(llm.digest_requests)
        clock = [0.0]
        if failure == "original-exception":
            llm.outputs = [RuntimeError("synthetic provider failure")]
        elif failure == "correction-exception":
            llm.outputs = [_overcap, RuntimeError("synthetic correction failure")]
        elif failure.startswith("deadline"):
            def expire(raw):
                clock[0] = 2.0
                return _overcap(raw) if failure == "deadline-before-correction" else raw
            llm.outputs = [expire] if failure == "deadline-before-correction" else [_overcap, expire]
        else:
            llm.outputs = [_overcap, lambda raw: _bad_payload(failure, raw)]
        if failure.startswith("deadline"):
            with pytest.raises(DeadlineExceeded):
                hy.dream(deadline=MonotonicDeadline(expires_at=1.0, clock=lambda: clock[0]))
        else:
            report = hy.dream()
            assert report.digest_failures == 1
            assert hy.conn.execute("SELECT digest_retry_count FROM sessions WHERE id='x'").fetchone()[0] == 1
        assert len(llm.digest_requests) - before_calls == (
            1 if failure in {"original-exception", "deadline-before-correction"} else 2
        )
        assert _cursor(hy) == before_cursor and _staging(hy) == before_stage
        assert _state(hy) == published
        assert covered_messages_after(hy.conn, "x", None) == before_source
        # An ordinary next dream with valid output recovers; no partial output
        # from either failed call was ever published or staged.
        _finish(hy)
        assert _staging(hy) == []
        assert "alpha" in _state(hy)["summary"] and "beta" in _state(hy)["summary"]
        assert hy.dream_status()["pending_digests"] == 0
        assert hy.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
@pytest.mark.parametrize("mode", ["forward", "rebuild"])
def test_successful_correction_stages_one_slice_and_publishes_only_complete_walk(cfg, granular, mode):
    llm = ScriptedDigestLLM()
    config = _quiet_cfg(cfg, dream_digest_max_chars=400, episode_granularity_enabled=granular)
    with closing(HyMem(config, llm=llm)) as hy:
        hy.log_message("x", "assistant", "alpha durable history " + "old material " * 45)
        hy.close_session("x")
        _finish(hy)
        published = _state(hy)
        if mode == "forward":
            hy.log_message("x", "assistant", "beta new work " + "new material " * 60)
        else:
            hy.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
        assert hy.dream().budget_exhausted
        before_cursor, before_stage = _cursor(hy), _staging(hy)
        before_source = covered_messages_after(hy.conn, "x", None)
        before_calls = len(llm.digest_requests)
        llm.outputs = [_overcap]
        report = hy.dream()
        assert report.digest_failures == 0 and report.digest_quarantined == 0
        assert report.budget_exhausted
        assert len(llm.digest_requests) - before_calls == 2
        first, second = llm.digest_requests[-2:]
        assert first.user == second.user
        assert _cursor(hy) != before_cursor
        assert len(_staging(hy)) == len(before_stage) + 1
        assert _state(hy) == published
        assert covered_messages_after(hy.conn, "x", None) == before_source
        _finish(hy)
        assert _staging(hy) == []
        assert "alpha" in _state(hy)["summary"]
        if mode == "forward":
            assert "beta" in _state(hy)["summary"]
        assert hy.dream_status()["pending_digests"] == 0
        assert hy.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
def test_persistent_length_failure_still_quarantines_at_bounded_attempts(cfg, granular):
    llm = ScriptedDigestLLM([_overcap] * 12)
    config = _quiet_cfg(cfg, dream_digest_max_chars=1200,
                        episode_granularity_enabled=granular, digest_extraction_max_attempts=6)
    with closing(HyMem(config, llm=llm)) as hy:
        hy.log_message("x", "assistant", "alpha source " + "long material " * 150)
        hy.close_session("x")
        initial = _cursor(hy)
        for attempt in range(1, 7):
            report = hy.dream()
            assert report.digest_failures == 1
            assert _cursor(hy) == initial and _staging(hy) == []
            assert hy.conn.execute("SELECT digest_retry_count FROM sessions WHERE id='x'").fetchone()[0] == attempt
        assert len(llm.digest_requests) == 12
        assert hy.dream_status()["quarantined_digests"] == 1
        assert hy.dream().digest_quarantined == 1
        assert len(llm.digest_requests) == 12 and _state(hy)["marker"] is None


@pytest.mark.parametrize("granular", [False, True], ids=["blob", "granular"])
def test_length_policy_generation_replays_retained_source_without_phase1_drift(cfg, monkeypatch, granular):
    llm = StubLLMClient(fixtures={
        digest._DIGEST_FIDELITY_SYSTEM: json.dumps(synthetic_fidelity_result()),
        digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM: json.dumps(synthetic_format_result()),
        "You analyze one conversation session": _payload("Alpha history."),
        "You re-read one conversation session": _payload("Alpha history."),
        "single pass": '{"triples":[],"markers":[],"complete":true}',
    }, default="[]")
    config = _config(cfg, granular)
    phase1 = extraction_contract_identity("v20")
    with closing(HyMem(config, llm=llm)) as hy:
        _seed(hy)
        with monkeypatch.context() as old:
            old.setattr(digest, "DIGEST_SUMMARY_RECOVERY_VERSION", "digest-summary-length-recovery-previous")
            hy.dream()
            previous = hy.conn.execute("SELECT digest_published_generation FROM sessions WHERE id='bounded-summary'").fetchone()[0]
            assert previous is not None
            assert sum(call.system == digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM for call in llm.calls) == 1
            with db.transaction(hy.conn):
                assert prune_messages(hy.conn, replace(config, message_retention_days=1)) == 1
        assert extraction_contract_identity("v20") == phase1
        assert hy.dream_status()["pending_digests"] == 1
        before_calls = len(llm.calls)
        hy.dream()
        current = hy.conn.execute("SELECT digest_published_generation FROM sessions WHERE id='bounded-summary'").fetchone()[0]
        assert current != previous
        assert any("Documented the Alpha service rollout." in request.user for request in llm.calls[before_calls:])
        assert hy.dream_status()["pending_digests"] == 0
        before_calls = len(llm.calls)
        hy.dream()
        assert len(llm.calls) == before_calls


@pytest.mark.parametrize("change", ["policy", "prompt", "dispatch"])
def test_digest_recovery_generation_changes_are_tier_local(monkeypatch, change):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    if change == "policy":
        monkeypatch.setattr(digest, "DIGEST_SUMMARY_RECOVERY_VERSION", "changed-recovery-policy")
    elif change == "prompt":
        monkeypatch.setattr(digest, "_DIGEST_SUMMARY_RECOVERY_TEMPLATE", "changed output-length feedback")
    else:
        original = digest.extract_session_digest

        def changed(*args, **kwargs):
            return original(*args, **kwargs)

        monkeypatch.setattr(digest, "extract_session_digest", changed)
        monkeypatch.setattr(runner, "extract_session_digest", changed)
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert before["digest"] != after["digest"]
    assert before["facts"] == after["facts"] and before["profile"] == after["profile"]
    assert extraction_contract_identity("v20") == phase1
