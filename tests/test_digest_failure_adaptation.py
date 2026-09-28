"""Durable attempt budgets are independent of input-reduction history."""
from contextlib import closing
from dataclasses import replace
import asyncio
import json
import re
import sqlite3

import pytest

from hymem import HyMem
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import digest, runner
from hymem.dreaming.lossless import covered_messages_after
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import StubLLMClient
from tests.test_digest_length_recovery import ScriptedDigestLLM, _overcap, _summary_payload
from tests.test_digest_publication import _finish, _state
from tests.test_digest_summary_contract import _extract, _payload, _seed
from tests.test_lossless_digest import _quiet_cfg
from tests.test_portability import _rewrite_v6_export


def _policy(maximum=3, *, rebuild=False):
    config = digest.digest_config_version(
        prompt_version="v20", episode_prompt_version=None,
        max_chars=1800, max_tokens=2048, max_episodes=None,
    )
    return digest.digest_retry_policy_version(
        config, max_attempts=maximum,
        rebuild_from=config + "|walk=" + "a" * 32 if rebuild else None,
        invalidated_stamp="old|nested|stamp" if rebuild else None,
    )


def _wrapped(count, policy):
    return f"digest-retry-state-v2|input-retries={count}|{policy}"


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("maximum", [-1, 0, 1, 3, 10])
@pytest.mark.parametrize("attempts,input_retries", [(1, 0), (1, 1), (3, 0), (3, 1), (3, 3)])
def test_current_and_historical_retry_wire_formats(rebuild, maximum, attempts, input_retries):
    policy = _policy(maximum, rebuild=rebuild)
    flag = int(maximum > 0 and attempts >= maximum)
    for key, expected_input in [(policy, attempts), (_wrapped(input_retries, policy), input_retries)]:
        assert digest.digest_retry_state_is_valid(attempts, key, flag)
        assert not digest.digest_retry_state_is_valid(attempts, key, 1 - flag)
        assert digest.digest_retry_counts_for_policy(attempts, key, retry_key=policy) == (attempts, expected_input)
        assert digest.digest_retry_is_quarantined(attempts, key, retry_key=policy, max_attempts=maximum) == bool(flag)
        assert digest.digest_retry_counts_for_policy(attempts, key, retry_key=_policy(maximum + 1, rebuild=rebuild)) == (0, 0)


_BAD_COUNT_TEXTS = ["", "-1", "+1", "01", "1.0", " 1", "1 ", "true", "١", "3", "9" * 5000]
_BAD_KEYS = [
    None, "", "digest.v9", "digest-retry-state-v3|input-retries=1|" + _policy(),
    "digest-retry-state-v2|" + _policy(),
    _wrapped(1, _wrapped(1, _policy())),
    _wrapped(1, _policy()) + "\n", _policy().replace("retry-max=3", "retry-max=" + "9" * 5000),
] + [_wrapped(text, _policy(rebuild=True)) for text in _BAD_COUNT_TEXTS]


@pytest.mark.parametrize("key", _BAD_KEYS)
def test_malformed_metadata_is_neither_a_policy_change_nor_a_retry_reset(key):
    assert not digest.digest_retry_state_is_valid(2, key, 0)
    assert not digest.digest_retry_is_quarantined(2, key, retry_key=_policy(), max_attempts=1)
    with pytest.raises(ValueError, match="invalid digest retry"):
        digest.digest_retry_counts_for_policy(2, key, retry_key=_policy())


@pytest.mark.parametrize("count", [True, False, -1, 1.0, "1", None, 1 << 63])
def test_malformed_total_count_is_rejected(count):
    assert not digest.digest_retry_state_is_valid(count, _wrapped(0, _policy()), 0)
    with pytest.raises(ValueError):
        digest.digest_retry_counts_for_policy(count, _wrapped(0, _policy()), retry_key=_policy())


def test_zero_state_and_sqlite_integer_bound_are_explicit():
    assert digest.digest_retry_state_is_valid(0, None, 0)
    assert not digest.digest_retry_state_is_valid(0, None, 1)
    assert not digest.digest_retry_state_is_valid(0, _wrapped(0, _policy()), 0)
    assert digest.digest_retry_counts_for_policy(0, None, retry_key=_policy()) == (0, 0)
    maximum = (1 << 63) - 1
    assert digest.digest_retry_state_is_valid(maximum, _wrapped(maximum, _policy()), 1)
    assert not digest.digest_retry_state_is_valid(1, _wrapped(0, _policy()), True)


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_recording_mixed_failures_keeps_total_budget_and_durable_input_count(rebuild, legacy):
    policy = _policy(4, rebuild=rebuild)
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, digest_retry_count INTEGER, digest_retry_config_version TEXT, digest_quarantined INTEGER)")
        conn.execute("INSERT INTO sessions VALUES ('x',?,?,0)", (1, policy if legacy else _wrapped(0, policy)))
        for attempts, input_failure in [(2, False), (3, True), (4, False)]:
            quarantined = digest.record_digest_failure(conn, "x", max_attempts=4,
                                                      retry_config_version=policy, input_failure=input_failure)
            row = conn.execute("SELECT * FROM sessions").fetchone()
            assert row["digest_retry_count"] == attempts and quarantined == (attempts == 4)
            expected_input = int(legacy) + int(attempts >= 3)
            assert row["digest_retry_config_version"] == _wrapped(expected_input, policy)
            assert digest.digest_retry_state_is_valid(attempts, row["digest_retry_config_version"], row["digest_quarantined"])
        new_policy = _policy(5, rebuild=rebuild)
        assert not digest.record_digest_failure(conn, "x", max_attempts=5,
                                                retry_config_version=new_policy, input_failure=False)
        assert tuple(conn.execute("SELECT digest_retry_count,digest_retry_config_version,digest_quarantined FROM sessions").fetchone()) == (1, _wrapped(0, new_policy), 0)


@pytest.mark.parametrize("key", _BAD_KEYS)
def test_record_failure_cannot_overwrite_malformed_metadata(key):
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, digest_retry_count INTEGER, digest_retry_config_version TEXT, digest_quarantined INTEGER)")
        conn.execute("INSERT INTO sessions VALUES ('x',2,?,0)", (key,))
        before = tuple(conn.execute("SELECT * FROM sessions").fetchone())
        with pytest.raises(ValueError):
            digest.record_digest_failure(conn, "x", max_attempts=3, retry_config_version=_policy(), input_failure=False)
        assert tuple(conn.execute("SELECT * FROM sessions").fetchone()) == before


@pytest.mark.parametrize("stage", ["primary", "summary_compaction", "unknown", None])
@pytest.mark.parametrize("reason,source_related", [
    ("completion_failure", True), ("parse_failure", True), ("output_truncated", True),
    ("shape_failure", True), ("episode_output_cap", True), ("episode_validation_failure", True),
    ("procedure_validation_failure", True), ("summary_shape_failure", False),
    ("summary_validation_failure", False), ("summary_output_cap", False), (None, False), ("unknown", False),
])
def test_failure_classification_is_explicit_and_conservative(stage, reason, source_related):
    assert digest.digest_failure_requires_input_shrink(reason, stage) == (stage == "primary" and source_related)


def _primary_calls(llm):
    return [call for call in llm.digest_requests if not call.system.startswith("You compact")]


def _retry_state(hy):
    return tuple(hy.conn.execute("SELECT digest_retry_count,digest_retry_config_version,digest_quarantined FROM sessions WHERE id='x'").fetchone())


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("failure,shrink", [
    ("not JSON", True), ('{"episodes":[],"summary":"unfinished', True),
    (_payload(42), False), (_payload('"tiny"'), False),
    (RuntimeError("synthetic primary transport failure"), True),
])
def test_primary_failures_adapt_only_the_source_related_kinds_after_reopen(cfg, granular, failure, shrink):
    llm = ScriptedDigestLLM([failure, failure])
    config = _quiet_cfg(cfg, dream_digest_max_chars=1800, digest_extraction_max_attempts=2,
                        episode_granularity_enabled=granular)
    hy = HyMem(config, llm=llm)
    try:
        hy.log_message("x", "assistant", "alpha " + "source material " * 300)
        hy.close_session("x")
        assert hy.dream().digest_failures == 1
        hy.close()
        hy = HyMem(config, llm=llm)
        assert hy.dream().digest_failures == 1
        first, second = _primary_calls(llm)
        assert (len(second.user) < len(first.user)) == shrink
        if not shrink:
            assert second.user == first.user
        assert _retry_state(hy)[0::2] == (2, 1)
        assert hy.dream_status()["quarantined_digests"] == 1
        assert hy.dream().digest_quarantined == 1
        assert len(llm.digest_requests) == 2
    finally:
        hy.close()


@pytest.mark.parametrize("mode", ["forward", "rebuild"])
def test_success_resets_both_counters_without_publishing_a_partial_walk(cfg, mode):
    llm = ScriptedDigestLLM()
    config = _quiet_cfg(cfg, dream_digest_max_chars=500, digest_extraction_max_attempts=5)
    with closing(HyMem(config, llm=llm)) as hy:
        hy.log_message("x", "assistant", "alpha " + "older history " * 35)
        hy.close_session("x")
        _finish(hy)
        published = _state(hy)
        if mode == "forward":
            hy.log_message("x", "assistant", "beta " + "new content " * 60)
        else:
            hy.conn.execute("UPDATE sessions SET digested_prompt_version=NULL WHERE id='x'")
        llm.outputs = ["not JSON", _overcap, _summary_payload("S" * 501)]
        assert hy.dream().digest_failures == 1
        assert hy.dream().digest_failures == 1
        assert _retry_state(hy)[0] == 2
        before = len(_primary_calls(llm))
        assert hy.dream().digest_failures == 0
        assert _retry_state(hy) == (0, None, 0)
        assert _state(hy) == published
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 1
        assert hy.dream().digest_failures == 0
        first_success, second_success = _primary_calls(llm)[before:before + 2]
        def width(call):
            start, end = map(int, re.search(r"chars=(\d+):(\d+)/", call.user).groups())
            return end - start
        assert width(second_success) > width(first_success)
        _finish(hy)
        assert hy.dream_status()["pending_digests"] == 0
        assert hy.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert hy.conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("key", [_wrapped(2, _policy()), "digest-retry-state-v9|unknown", _wrapped("01", _policy())])
def test_runner_holds_malformed_metadata_without_calls_or_state_reset(cfg, key):
    llm = ScriptedDigestLLM()
    with closing(HyMem(_quiet_cfg(cfg), llm=llm)) as hy:
        hy.log_message("x", "assistant", "alpha exact source")
        hy.close_session("x")
        hy.conn.execute("UPDATE sessions SET digest_retry_count=1,digest_retry_config_version=? WHERE id='x'", (key,))
        before = _retry_state(hy)
        assert hy.dream_status()["malformed_digests"] == 1
        report = hy.dream()
        assert report.digest_failures == 1 and report.digest_quarantined == 0
        assert llm.digest_requests == [] and _retry_state(hy) == before
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


@pytest.mark.parametrize("stage", ["primary", "summary_compaction"])
@pytest.mark.parametrize("error", [DeadlineExceeded("synthetic deadline"), KeyboardInterrupt(), SystemExit(3)])
def test_control_flow_exceptions_escape_without_wrapping_or_retry_mutation(cfg, stage, error):
    llm = ScriptedDigestLLM([error] if stage == "primary" else [_overcap, error])
    with closing(HyMem(_quiet_cfg(cfg), llm=llm)) as hy:
        _seed(hy)
        with pytest.raises(type(error)) as caught:
            _extract(hy, llm, granular=False)
        assert caught.value is error
        assert hy.conn.execute("SELECT digest_retry_count FROM sessions WHERE id='bounded-summary'").fetchone()[0] == 0


@pytest.mark.parametrize("stage", ["primary", "summary_compaction"])
def test_ordinary_completion_exceptions_keep_stage_and_original_cause(cfg, stage):
    original = RuntimeError("synthetic transport failure")
    llm = ScriptedDigestLLM([original] if stage == "primary" else [_overcap, original])
    with closing(HyMem(_quiet_cfg(cfg), llm=llm)) as hy:
        _seed(hy)
        with pytest.raises(digest.DigestCompletionError) as caught:
            _extract(hy, llm, granular=False)
        assert caught.value.failure_stage == stage and caught.value.__cause__ is original


@pytest.mark.parametrize("legacy", [False, True])
def test_retry_wire_state_survives_portable_round_trip_and_preserves_next_source(cfg, tmp_path, legacy):
    llm = ScriptedDigestLLM([_overcap, _summary_payload("S" * 501), "not JSON"])
    config = _quiet_cfg(cfg, dream_digest_max_chars=1800, digest_extraction_max_attempts=4,
                        redact_secrets=False)
    with closing(HyMem(config, llm=llm)) as source:
        source.log_message("x", "assistant", "alpha " + "exact source " * 300)
        source.close_session("x")
        assert source.dream().digest_failures == 1
        assert source.dream().digest_failures == 1
        before = _retry_state(source)
        if legacy:
            bare = before[1].split("|", 2)[2]
            source.conn.execute("UPDATE sessions SET digest_retry_config_version=? WHERE id='x'", (bare,))
            before = _retry_state(source)
        export = tmp_path / "state.jsonl"
        source.export(export)
        with closing(HyMem(replace(config, root=tmp_path / "restored"), llm=llm)) as restored:
            restored.import_(export)
            assert _retry_state(restored) == before
            assert restored.dream_status()["malformed_digests"] == 0
            # A successful attempt uses the persisted input count (legacy
            # defaults to all two failures), then clears both counters.
            assert restored.dream().digest_failures == 0
            request = _primary_calls(llm)[-1]
            start, end, total = map(int, re.search(r"chars=(\d+):(\d+)/(\d+)", request.user).groups())
            message = covered_messages_after(restored.conn, "x", None)[0]
            expected_cap = 450 if legacy else 900
            assert start == 0 and total == len(message.content) and end < total
            # Prove the actual imported runner request fills exactly the
            # expected framed source budget, not merely that metadata survived.
            assert len(digest._render_message_part(message, 0, end)) <= expected_cap
            assert len(digest._render_message_part(message, 0, end + 1)) > expected_cap
            assert message.content[:end] in request.user
            assert _retry_state(restored) == (0, None, 0)
            assert restored.conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_export_and_import_reject_corrupt_retry_metadata_atomically(cfg, tmp_path):
    with closing(HyMem(_quiet_cfg(cfg), llm=ScriptedDigestLLM())) as hy:
        hy.log_message("x", "assistant", "alpha source")
        hy.close_session("x")
        good = tmp_path / "good.jsonl"
        hy.export(good)
        hy.conn.execute("UPDATE sessions SET digest_retry_count=1,digest_retry_config_version=? WHERE id='x'", (_wrapped(2, _policy()),))
        with pytest.raises(ValueError, match="invalid digest retry"):
            hy.export(tmp_path / "bad-export.jsonl")
    def corrupt(body):
        for item in body:
            if item.get("type") == "session":
                item["record"].update(digest_retry_count=1, digest_retry_config_version=_wrapped(2, _policy()))
    _rewrite_v6_export(good, corrupt)
    with closing(HyMem(replace(_quiet_cfg(cfg), root=tmp_path / "destination"), llm=ScriptedDigestLLM())) as destination:
        with pytest.raises(ValueError, match="invalid digest retry"):
            destination.import_(good)
        assert destination.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_runner_policy_identity_invalidates_all_auxiliary_tiers_not_phase1(monkeypatch):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    original = runner._run_dreaming
    def changed(*args, **kwargs):
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "_run_dreaming", changed)
    assert all(semantic_generation_suffix(tier, client) != version for tier, version in before.items())
    assert extraction_contract_identity("v20") == phase1


def _inject_compaction_failure(monkeypatch, point, error):
    if point == "request_builder":
        class FailingTemplate(str):
            def format(self, *args, **kwargs):
                raise error
        monkeypatch.setattr(digest, "_DIGEST_SUMMARY_RECOVERY_TEMPLATE",
                            FailingTemplate(digest._DIGEST_SUMMARY_RECOVERY_TEMPLATE))
    elif point == "repair_validator":
        def fail(raw):
            raise error
        monkeypatch.setattr(digest, "_validate_digest_summary_repair", fail)
    else:
        original = digest._validate_digest_response
        def fail(raw, *args, **kwargs):
            if raw is None:
                raise error
            return original(raw, *args, **kwargs)
        monkeypatch.setattr(digest, "_validate_digest_response", fail)


@pytest.mark.parametrize("point", ["request_builder", "repair_validator", "assembled_validator"])
@pytest.mark.parametrize("error", [RuntimeError("synthetic processing failure"),
                                   DeadlineExceeded("synthetic deadline"),
                                   KeyboardInterrupt(), SystemExit(3), asyncio.CancelledError()])
def test_entire_compaction_boundary_preserves_stage_causes_and_control_flow(cfg, monkeypatch, point, error):
    llm = ScriptedDigestLLM([_overcap])
    with closing(HyMem(_quiet_cfg(cfg), llm=llm)) as hy:
        _seed(hy)
        _inject_compaction_failure(monkeypatch, point, error)
        expected = digest.DigestCompletionError if isinstance(error, Exception) else type(error)
        with pytest.raises(expected) as caught:
            _extract(hy, llm, granular=False)
        if isinstance(error, Exception):
            assert caught.value.failure_stage == "summary_compaction"
            assert caught.value.__cause__ is error
        else:
            assert caught.value is error
        assert len(llm.digest_requests) == (1 if point == "request_builder" else 2)
        assert hy.conn.execute("SELECT digest_retry_count FROM sessions WHERE id='bounded-summary'").fetchone()[0] == 0


@pytest.mark.parametrize("granular", [False, True])
def test_real_deep_json_parser_error_retains_compaction_stage_and_original_cause(cfg, granular):
    # 10k levels reaches the C JSON parser bound on Python3.13; Python3.11
    # already raises at ~1k. The malformed provider output is never accepted.
    raw = "[" * 10000 + "0" + "]" * 10000
    with pytest.raises(RecursionError):
        digest._validate_digest_summary_repair(raw)
    llm = ScriptedDigestLLM([_overcap, raw])
    with closing(HyMem(_quiet_cfg(cfg), llm=llm)) as hy:
        _seed(hy)
        with pytest.raises(digest.DigestCompletionError) as caught:
            _extract(hy, llm, granular=granular)
        assert caught.value.failure_stage == "summary_compaction"
        assert isinstance(caught.value.__cause__, RecursionError)
        assert len(llm.digest_requests) == 2


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("point", ["deep_json", "request_builder", "repair_validator", "assembled_validator"])
def test_unexpected_compaction_processing_failures_preserve_source_after_reopen(cfg, monkeypatch, granular, point):
    raw = "[" * 10000 + "0" + "]" * 10000
    if point == "deep_json":
        outputs = [_overcap, raw, _overcap, raw]
    elif point == "request_builder":
        outputs = [_overcap, _overcap]
    else:
        outputs = [_overcap, _summary_payload("Compact valid summary."),
                   _overcap, _summary_payload("Compact valid summary.")]
    llm = ScriptedDigestLLM(outputs)
    config = _quiet_cfg(cfg, dream_digest_max_chars=1800, digest_extraction_max_attempts=2,
                        episode_granularity_enabled=granular)
    hy = HyMem(config, llm=llm)
    try:
        hy.log_message("x", "assistant", "alpha exact source " + "durable material " * 250)
        hy.close_session("x")
        if point != "deep_json":
            _inject_compaction_failure(monkeypatch, point, RuntimeError("synthetic processing failure"))
        for attempt in (1, 2):
            assert hy.dream().digest_failures == 1
            assert _retry_state(hy)[0::2] == (attempt, int(attempt == 2))
            hy.close()
            hy = HyMem(config, llm=llm)
        first, second = _primary_calls(llm)
        assert second.user == first.user
        assert hy.dream_status()["quarantined_digests"] == 1
        assert hy.dream().digest_quarantined == 1
        assert len(llm.digest_requests) == (2 if point == "request_builder" else 4)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert hy.conn.execute("SELECT digest_cursor_message_id FROM sessions WHERE id='x'").fetchone()[0] is None
    finally:
        hy.close()
