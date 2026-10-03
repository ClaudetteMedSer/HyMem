"""Free measuring-device controls; no provider calls or canonical mutations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import extraction_texture_probe as probe


def _old_fresh_hex(length, seed):
    # Reproduce the actual broken instrument, not an imagined bad control.
    return "".join(probe.HEX[ord(hashlib.sha256(f"{seed}:{i}".encode())
                               .hexdigest()[0]) % 16] for i in range(length))


def _request():
    context = "0123456789abcdef" * 8 + "\nLeft-side claim: café prefers the following."
    records = [
        {"content": "The table control is unchanged.", "source_message_id": 9},
        {"content": "PostgreSQL is that database.", "source_message_id": 10,
         "source_peer_id": "synthetic-only", "source_content_start": 170,
         "source_boundary_context": {
             "content": context, "kind": "preceding_adjacent_prose",
             "source_content_start": 0, "source_content_end": 170,
             "applies_through_source_content_end": 205, "version": "test-v1"},
         "unfamiliar_future_field": {"keep": [False, None, 3]}},
    ]
    body = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) for record in records)
    return {"system": "Unchanged synthetic system.",
            "user": 'Excerpt:\n"""\n' + body + '\n"""\nUnchanged suffix.',
            "temperature": 0.73, "max_tokens": 5713, "response_format": "json",
            "model": "synthetic-not-a-provider", "thinking": "off",
            "extra_body": {"unfamiliar": [7, {"nested": "value"}]}}


def _mutate(request, **overrides):
    kwargs = {"record_index": 1,
              "field_path": ("source_boundary_context", "content"),
              "padding_span": (0, 128),
              "protected_claim": "Left-side claim: café prefers the following.",
              "mode": "fresh_hex", "seed": "control"}
    kwargs.update(overrides)
    return probe.mutate_request(request, **kwargs)


def test_original_generator_control_fails_as_hex_instrument():
    # Exhaustive over its intermediate alphabet: deterministic, no flaky
    # statistical assertion about whether a random sample happens to mix classes.
    assert set(probe.HEX[ord(char) % 16] for char in probe.HEX) == set("0123456789")
    old = _old_fresh_hex(256, "instrument-control-v1")
    assert set(old) <= set("0123456789")
    assert not set(old) & set("abcdef")


def test_fresh_hex_uses_direct_digest_and_stable_full_alphabet():
    expected = "".join(hashlib.sha256(f"instrument-control-v1:{i}".encode())
                       .hexdigest() for i in range(4))
    actual = probe.fresh_hex(256, seed="instrument-control-v1")
    assert actual == expected
    assert set(actual) == set(probe.HEX)
    assert probe.fresh_hex(65, seed="instrument-control-v1") == expected[:65]
    assert probe.fresh_hex(0, seed="instrument-control-v1") == ""
    assert actual != probe.fresh_hex(256, seed="different")


@pytest.mark.parametrize("length,seed", [(-1, "x"), (True, "x"), (1.5, "x"),
                                       (probe.MAX_CHARS + 1, "x"), (10, None),
                                       (10, "x" * (probe.MAX_SEED_CHARS + 1))])
def test_fresh_hex_rejects_invalid_inputs(length, seed):
    with pytest.raises(probe.ProbeInputError):
        probe.fresh_hex(length, seed=seed)


@pytest.mark.parametrize("mode,expected", [
    ("baseline", "0123456789abcdef"),
    ("digits_to_hash", "##########abcdef"),
    ("letters_to_a", "0123456789aaaaaa"),
    ("digits_to_a", "aaaaaaaaaaabcdef"),
    ("flat_a", "aaaaaaaaaaaaaaaa"),
])
def test_padding_mutations_preserve_every_other_byte(mode, expected):
    claim = "Protected café claim."
    text = "prefix!0123456789abcdef\n" + claim + " suffix"
    actual = probe.mutate_padding(text, padding_span=(7, 23),
                                  protected_claim=claim, mode=mode)
    assert actual == "prefix!" + expected + "\n" + claim + " suffix"
    assert len(actual) == len(text)
    assert len(actual.encode()) == len(text.encode())


@pytest.mark.parametrize("text,span,claim", [
    ("abcdef claim claim", (0, 6), "claim"),
    ("abcdef aaa", (0, 6), "aa"),  # overlapping occurrences
    ("abcdef claim", (0, 6), "missing"),
    ("abcdef claim", (0, 6), ""),
    ("abcdef claim", (0, 6), "abc"),
    ("ABCDEF claim", (0, 6), "claim"),
    ("abcdef claim", (0, 7), "claim"),
    ("abcdef claim", (0, 0), "claim"),
    ("abcdef claim", (0, 100), "claim"),
    ("abcdef claim", (True, 6), "claim"),
    ("abcdef claim", (0, 6.0), "claim"),
])
def test_ambiguous_or_nonpadding_targets_rejected(text, span, claim):
    with pytest.raises(probe.ProbeInputError):
        probe.mutate_padding(text, padding_span=span, protected_claim=claim,
                             mode="fresh_hex")


def test_baseline_exact_roundtrip_all_fields_no_defaults():
    request = _request()
    frozen = deepcopy(request)
    result = _mutate(request, mode="baseline")
    assert result == request == frozen
    assert result is not request
    assert result["extra_body"] is not request["extra_body"]


def test_mutation_changes_only_one_span_preserves_parameters_and_source_metadata():
    request = _request()
    frozen = deepcopy(request)
    result = _mutate(request)
    assert request == frozen
    original_hex = "0123456789abcdef" * 8
    assert request["user"].count(original_hex) == 1
    assert result["user"] == request["user"].replace(
        original_hex, probe.fresh_hex(128, seed="control"))
    assert len(result["user"].encode()) == len(request["user"].encode())
    assert {k: v for k, v in result.items() if k != "user"} == {
        k: v for k, v in request.items() if k != "user"}
    result["extra_body"]["unfamiliar"].append("new")
    assert request == frozen


def test_current_canonical_request_bytes_are_accepted_without_mutating_fixture():
    # Construct the maintained fixture/prompt without invoking extraction or
    # any client. This is an instrument compatibility check, not a scored arm.
    from benchmarks import extraction_canary as canary
    from hymem.extraction.llm import LLMRequest
    from hymem.extraction.prompts import CHUNK_EXTRACTION_USER_TEMPLATE

    originals = canary._source_records()
    request = asdict(LLMRequest(
        system="synthetic-control", max_tokens=4096, temperature=0.0,
        user=CHUNK_EXTRACTION_USER_TEMPLATE.format(text="\n".join(text for _, text in originals)),
    ))
    content = json.loads(originals[1][1])["content"]
    pad = hashlib.sha256(b"hymem-canary-prose-prefix").hexdigest() * 20
    start = content.index(pad)
    result = probe.mutate_request(
        request, record_index=1, field_path=("content",),
        padding_span=(start, start + len(pad)),
        protected_claim="PostgreSQL is that database.", mode="baseline",
    )
    assert result == request
    assert canary._source_records() == originals


@pytest.mark.parametrize("user", [
    'Excerpt:\n"""\n{"a":1,"a":2}\n"""',
    'Excerpt:\n"""\n{"content": "spaces"}\n"""',
    'Excerpt:\n"""\n{"content":NaN}\n"""',
    'Excerpt:\n"""\n{"content":Infinity}\n"""',
    'Excerpt:\n"""\n[]\n"""',
    'Excerpt:\n"""\nnot-json\n"""',
    'Excerpt:\n"""\n{"content":"x"}\n"""\n"""',
    'Different excerpt:\n"""\n{"content":"x"}\n"""',
])
def test_mutation_refuses_serialization_or_prompt_drift(user):
    with pytest.raises(probe.ProbeInputError):
        _mutate({"user": user}, record_index=0)


@pytest.mark.parametrize("overrides", [
    {"record_index": True}, {"record_index": -1}, {"record_index": 2},
    {"field_path": ("source_message_id",)},
    {"field_path": ("source_boundary_context", "source_content_start")},
    {"record_index": 0}, {"mode": "unsafe_mode"},
])
def test_mutation_refuses_invalid_field_selection(overrides):
    with pytest.raises(probe.ProbeInputError):
        _mutate(_request(), **overrides)


def test_randomized_schedule_is_deterministic_balanced_and_explicit_about_spacing():
    args = {"arms": probe.ARMS, "repetitions": 15, "seed": 31, "spacing_seconds": 0.8}
    schedule = probe.build_schedule(**args)
    assert schedule == probe.build_schedule(**args)
    assert schedule != probe.build_schedule(**{**args, "seed": 32})
    orders = []
    for repetition in range(15):
        order = [item.arm for item in schedule if item.repetition == repetition]
        assert sorted(order) == sorted(probe.ARMS)
        orders.append(tuple(order))
    assert len(set(orders)) > 1  # fixed fixture, not a probabilistic run
    assert schedule[0].planned_delay_before_seconds == 0
    assert all(item.planned_delay_before_seconds == 0.8 for item in schedule[1:])


@pytest.mark.parametrize("overrides", [
    {"arms": []}, {"arms": ["baseline", "baseline"]}, {"arms": ["private-source"]},
    {"repetitions": 0}, {"repetitions": True}, {"repetitions": 10001},
    {"seed": False}, {"seed": 1 << 257}, {"spacing_seconds": -1},
    {"spacing_seconds": 1 << 10000}, {"spacing_seconds": float("nan")},
    {"spacing_seconds": float("inf")}, {"spacing_seconds": True},
])
def test_schedule_rejects_unsafe_or_unbounded_spec(overrides):
    args = {"arms": probe.ARMS, "repetitions": 3, "seed": 31, "spacing_seconds": 0.8}
    with pytest.raises(probe.ProbeInputError):
        probe.build_schedule(**{**args, **overrides})


def _observe(**overrides):
    values = {"contract_valid": True, "complete": True, "triple_count": 1,
              "marker_count": 0, "matched_claim_indexes": [0], "expected_claim_index": 0}
    return probe.classify_observation(**{**values, **overrides})


def test_control_claim_index_not_globally_hardcoded_to_prose():
    assert _observe().classification == "emitted_expected"
    assert _observe(expected_claim_index=1).classification == "emitted_other"
    assert _observe(expected_claim_index=1, matched_claim_indexes=[1]).expected_claim_present


def test_classification_does_not_confuse_emission_empty_invalid_or_correctness():
    observations = [_observe(), _observe(matched_claim_indexes=[]),
                    _observe(triple_count=0, matched_claim_indexes=[]),
                    _observe(contract_valid=False), _observe(complete=False)]
    assert probe.aggregate_observations(observations) == {
        "observations": 5, "emitted_expected": 1, "emitted_other": 1,
        "empty": 1, "contract_failure": 1, "incomplete": 1}
    assert _observe(triple_count=0, marker_count=1,
                    matched_claim_indexes=[]).classification == "emitted_other"


@pytest.mark.parametrize("overrides", [
    {"contract_valid": 1}, {"complete": 1}, {"triple_count": True},
    {"marker_count": -1}, {"expected_claim_index": True},
    {"matched_claim_indexes": [True]}, {"matched_claim_indexes": [0, 1]},
    {"matched_claim_indexes": "private text"},
])
def test_invalid_observation_shapes_cannot_earn_credit(overrides):
    with pytest.raises(probe.ProbeInputError):
        _observe(**overrides)


def test_aggregates_and_errors_cannot_copy_response_text(capsys):
    private = "secret response payload"
    with pytest.raises(probe.ProbeInputError, match="^invalid_observation$"):
        probe.aggregate_observations([probe.Observation(private, False)])
    with pytest.raises(probe.ProbeInputError, match="^invalid_observation$"):
        probe.aggregate_observations([probe.Observation("emitted_expected", False)])
    assert probe.main([private]) == 2
    output = capsys.readouterr()
    assert private not in output.out + output.err
    assert json.loads(output.out)["model_calls"] == 0


def test_cli_defaults_to_free_offline_synthetic_control():
    script = Path(probe.__file__).resolve()
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            check=False, timeout=5)
    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "status": "pass", "mode": "offline_instrument_check", "model_calls": 0,
        "hex_alphabet_symbols": 16}
