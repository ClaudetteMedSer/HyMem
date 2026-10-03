"""Regression pins for the lenient LLM-JSON parser (hymem/extraction/jsonio.py).

The shapes below are not hypothetical: dream 1013 dropped a complete 4660-char
rollup because the provider fenced it despite json_object mode. Each case here
is a wrapper a chat-tuned model actually emits around otherwise-valid JSON.
"""
from __future__ import annotations

import pytest

from hymem.extraction.jsonio import (
    is_ceiling_cut,
    loads_exact_or_fenced,
    loads_lenient,
)

_OBJ = '{"title": "t", "summary": "s"}'
_EXPECTED = {"title": "t", "summary": "s"}


@pytest.mark.parametrize("raw", [
    _OBJ,
    f"```json\n{_OBJ}\n```",
    f"```JSON\n{_OBJ}\n```",
    f"```\n{_OBJ}\n```",
])
def test_advancement_parser_accepts_only_exact_json_or_a_whole_fence(raw):
    assert loads_exact_or_fenced(raw) == _EXPECTED


@pytest.mark.parametrize("raw", [
    f"I cannot comply. {_OBJ}",
    f"Here is an example:\n```json\n{_OBJ}\n```",
    f"```JSON\n{_OBJ}\n```\nHope that helps!",
])
def test_advancement_parser_never_salvages_json_from_prose(raw):
    assert loads_exact_or_fenced(raw) is None


@pytest.mark.parametrize("raw", [
    '{"triples": [], "triples": [1], "markers": []}',
    '{"outer": {"value": 1, "value": 2}}',
    '{"value": NaN}',
    '{"value": Infinity}',
    '```JSON\n{"value": -Infinity}\n```',
])
def test_advancement_parser_rejects_ambiguous_or_nonstandard_json(raw):
    assert loads_exact_or_fenced(raw) is None


@pytest.mark.parametrize("raw", [
    _OBJ,                                        # bare object (the strict path)
    f"```json\n{_OBJ}\n```",                     # lowercase language tag
    f"```\n{_OBJ}\n```",                         # bare fence
    f"```JSON\n{_OBJ}\n```",                     # uppercase tag
    f"Here is the JSON:\n```json\n{_OBJ}\n```",  # prose before
    f"```json\n{_OBJ}\n```\nHope that helps!",   # chatter after
    f"Sure!\n```JSON\n{_OBJ}\n```\nLet me know.",  # both
    f"  \n\n{_OBJ}\n\n  ",                       # stray whitespace only
])
def test_object_survives_its_wrapper(raw):
    assert loads_lenient(raw) == _EXPECTED


@pytest.mark.parametrize("raw", ["", "   \n\t ", None, 42, {"already": "decoded"},
                                 b'{"bytes": 1}', "no json here at all",
                                 "{ not: valid, json }"])
def test_unusable_input_returns_none_never_raises(raw):
    assert loads_lenient(raw) is None
    assert loads_lenient(raw, expect="array") is None
    assert loads_lenient(raw, expect="any") is None


def test_array_shape():
    assert loads_lenient('[{"text": "a"}, {"text": "b"}]', expect="array") == [
        {"text": "a"}, {"text": "b"}]
    assert loads_lenient('```json\n[{"text": "a"}]\n```', expect="array") == [
        {"text": "a"}]


def test_strict_first_keeps_a_top_level_array_whole():
    # The negative that strict-first exists for: bracket-scanning a well-formed
    # array with expect="object" would find the first `{` and the last `}` and
    # silently return the inner objects joined — a wrong answer, not a failure.
    # Strict parse wins before the scan ever runs.
    raw = '[{"text": "a"}, {"text": "b"}]'
    parsed = loads_lenient(raw, expect="object")
    assert isinstance(parsed, list) and len(parsed) == 2
    assert loads_lenient(raw, expect="any") == parsed


@pytest.mark.parametrize("wrapper", ["{body}", "```json\n{body}\n```",
                                     "Here you go:\n```JSON\n{body}\n```\nEnjoy!"])
@pytest.mark.parametrize("key", ["facts", "items", "results"])
def test_single_list_envelope_is_unwrapped_however_it_is_wrapped(wrapper, key):
    # json_object mode pushes providers to a top-level OBJECT while the prompt
    # asks for a bare array; {"facts": [...]} is how a model satisfies both.
    # Bare and fenced MUST agree — otherwise survival depends on formatting.
    raw = wrapper.format(body='{"%s": [{"text": "a"}, {"text": "b"}]}' % key)
    assert loads_lenient(raw, expect="array") == [{"text": "a"}, {"text": "b"}]


def test_envelope_with_a_sibling_scalar_still_unwraps():
    # One list among the values is still one candidate, not a guess.
    assert loads_lenient('{"count": 2, "facts": [1, 2]}', expect="array") == [1, 2]


def test_ambiguous_envelope_is_not_unwrapped():
    # Two lists = two candidates. Guessing here would silently drop half the
    # payload; hand the dict back and let the caller's shape check reject it.
    raw = '{"facts": [1, 2], "notes": [3]}'
    assert loads_lenient(raw, expect="array") == {"facts": [1, 2], "notes": [3]}
    assert loads_lenient(f"```json\n{raw}\n```", expect="array") == {
        "facts": [1, 2], "notes": [3]}


def test_envelope_whose_value_is_not_a_list_is_not_unwrapped():
    assert loads_lenient('{"facts": "none"}', expect="array") == {"facts": "none"}


def test_object_containing_a_list_is_not_narrowed_under_expect_object():
    # Mirror image of the unwrap: with expect="object" the dict IS the answer,
    # so nothing is narrowed to the inner array.
    assert loads_lenient('{"items": [1, 2]}', expect="object") == {"items": [1, 2]}
    assert loads_lenient('{"items": [1, 2]}', expect="any") == {"items": [1, 2]}


def test_expect_any_recovers_either_shape_from_a_wrapper():
    assert loads_lenient(f"chatter\n```json\n{_OBJ}\n```", expect="any") == _EXPECTED
    assert loads_lenient('chatter\n```\n[1, 2]\n```', expect="any") == [1, 2]


def test_unknown_expect_falls_back_to_object_delimiters():
    assert loads_lenient(f"```json\n{_OBJ}\n```", expect="nonsense") == _EXPECTED


@pytest.mark.parametrize("raw", [
    '{"items": [1, {"name": "Ada"}',
    '[{"name": "Ada"}, false, nul',
    '{"message": "unfinished',
    '{"message": "unfinished\\\\',
    '{"message": "unfinished\\\" quote and slash \\\\',
    '{"message": "partial unicode \\u12',
    '{"count": -',
    '{"count": 1.',
    '{"count": 1e',
    '{"count": 1e-',
    '{"enabled": tru',
    ' \t\r\n {"items": [1, 2',
    'Here is the result:\n{"items": [1, 2',
    'A [note] precedes the result: {"items": [1, 2',
    '```json\n{"items": [1, 2',
    '```JSON   \r\n ["café", {"emoji": "🧠"}',
    '```   \n {"items": [1, 2',
    '``` json \n {"items": [1, 2',
])
def test_ceiling_cut_accepts_only_unfinished_legal_json_prefixes(raw):
    assert is_ceiling_cut(raw) is True


@pytest.mark.parametrize("raw", [
    None,
    42,
    b'{"items": [1',
    "",
    "plain prose",
    "```json",
    "```json\n",
    "```JSON\n```",
    '{"message": "a } brace and [ bracket in a value"}',
    '{"message": "escaped quote: \\\" and backslash: \\\\"}',
    '{"message": "café 🧠"}',
    '{"items": [1, {"name": "Ada"}]}',
    '[1, 2, {"name": "Ada"}]',
    '{"items": [1]} trailing prose',
    '```json\n{"items": [1]}\n```',
    # A complete JSON root with only its Markdown fence cut is not evidence
    # that the structured response itself hit the model's token ceiling.
    '```json\n{"items": [1]}',
    '{ not: "json"',
    '{"a" "missing colon"',
    '{"a": ???',
    '{"a": 1,}',
    '[1,]',
    '{"a": 01',
    '{"a": 1.}',
    '{"a": 1e+}',
    '{"a": trux',
    '{"a": 1, "a":',
    '{"a": 1, "\\u0061":',
    '{"outer": {"a": 1, "a":',
    'Here is JSON: {"a": 1, "a": [1',
    '{"a": 1e999',
    '{"a": "bad\\q escape',
    '{"a": "raw\nnewline',
    '{"a": [1}',
])
def test_ceiling_cut_rejects_complete_or_malformed_payloads(raw):
    assert is_ceiling_cut(raw) is False


def test_ceiling_cut_ignores_structural_characters_inside_a_large_string():
    # This also guards the scanner's linear path: every byte is visited once,
    # without repeated substring parsing or delimiter recounting.
    raw = '{"payload":"' + ('}🧠[{' * 250_000)
    assert len(raw) > 1_000_000
    assert is_ceiling_cut(raw) is True
