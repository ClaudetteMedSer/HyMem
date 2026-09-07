"""Lenient parsing of an LLM reply that is SUPPOSED to be pure JSON.

Every JSON-shaped call in HyMem already sets ``response_format="json"``, which
the OpenAI-compatible client turns into ``response_format={"type":
"json_object"}``. That is a request, not a contract: dream 1013 logged
``kind=rollup stage=parse raw_len=4660`` — a complete, valid rollup wrapped in
```json fences — from a call that had json_object mode set. Providers behind an
OpenAI-compatible facade vary in whether they honour it at all, and a model that
slips back into chat habits will also prepend "Here is the JSON:" or append
"Hope that helps!".

What that costs is not a crash but a SILENT DROP: `json.loads` raises, the
caller logs a warning and returns None, and a perfectly good LLM result — one
we already paid for — is thrown away. For the dreaming parsers a dropped
fusion is retried on the next dream and costs reuse (each fail→heal transition
re-fuses a node), so tolerating the wrapper is strictly cheaper than re-earning
the answer.

Strict-first is deliberate: a well-formed reply parses exactly as it does today
and the bracket scan only ever runs on a payload that already failed. That
keeps a top-level ARRAY an array — bracket-scanning it for an object would
silently narrow it to its first element.
"""
from __future__ import annotations

from hymem.contrib.implementation_identity import import_time_source_sha256

EXTRACTION_IMPLEMENTATION_SHA256 = import_time_source_sha256(__file__)

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

_DELIMS = {"object": ("{", "}"), "array": ("[", "]")}

# This policy is embedded in the Phase-1 extraction contract.  Changing the
# classifier therefore invalidates durable extraction cache hits and benchmark
# canary reports even when the human-facing prompt version is unchanged.
JSON_CEILING_CUT_POLICY_VERSION = "hymem-json-ceiling-cut-grammar-v1"

_JSON_WHITESPACE = frozenset(" \t\r\n")
_JSON_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')
_JSON_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_OPENING_JSON_FENCE = re.compile(
    r"```[ \t]*(?:json[ \t]*)?(?:\r\n|\r|\n)",
    flags=re.IGNORECASE,
)

_PREFIX_COMPLETE = "complete"
_PREFIX_INCOMPLETE = "incomplete"
_PREFIX_INVALID = "invalid"


@dataclass
class _JSONContainerFrame:
    kind: str
    state: str
    keys: set[str] = field(default_factory=set)


def _new_json_container_frame(opener: str) -> _JSONContainerFrame:
    if opener == "{":
        return _JSONContainerFrame("object", "key_or_end")
    return _JSONContainerFrame("array", "value_or_end")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def loads_strict_json(text: str) -> Any:
    """RFC-style JSON: reject duplicate keys and Python's NaN extensions."""
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_object_keys,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_finite_float,
    )


def loads_exact_or_fenced(raw: object) -> Any | None:
    """Parse exact JSON, allowing only a whole-response Markdown fence.

    This is the advancement-authorizing parser for durable extraction
    cursors/processed markers.  It deliberately does *not* scan surrounding
    prose: a refusal or example containing a valid-looking empty object must
    not become an authoritative successful empty.  Providers may still wrap
    an otherwise pure answer in one complete ``json`` fence.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return loads_strict_json(text)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.fullmatch(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        return loads_strict_json(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def is_ceiling_cut(raw: object) -> bool:
    """Return whether *raw* ends while still a legal JSON-container prefix.

    This is deliberately narrower than "has more opening than closing
    braces".  Braces inside strings are data, arrays are valid roots, and a
    syntactically impossible fragment is malformed rather than evidence that
    the provider hit its output ceiling.  The scanner accepts the same useful
    envelopes seen from chat-tuned providers: leading whitespace/prose and a
    whole-response Markdown JSON fence.  It does not require a closing fence,
    because a ceiling cut necessarily may remove it; the JSON value itself
    must still be incomplete.  Conversely, reaching a complete root makes the
    answer non-truncated even if prose follows or a fence remains unclosed.

    Only object/array roots are relevant to HyMem's structured calls.  The
    iterative recognizer is deterministic and linear in the reply length; it
    tracks container grammar, strings and escapes, literals, and JSON numbers.
    """
    if not isinstance(raw, str):
        return False
    for start in _json_container_starts(raw):
        state = _scan_json_container_prefix(raw, start)
        if state == _PREFIX_COMPLETE:
            return False
        if state == _PREFIX_INCOMPLETE:
            return True
    return False


def _json_container_starts(text: str) -> tuple[int, ...]:
    """Locate at most one object and one array candidate, in source order."""
    offset = len(text) - len(text.lstrip())
    if offset >= len(text):
        return ()
    if text[offset] in "{[":
        return (offset,)

    # A Markdown fence is an envelope, not reply content.  Permit a bare fence
    # or a case-insensitive ``json`` info string, with horizontal whitespace;
    # this is at least as tolerant as ``loads_exact_or_fenced`` while remaining
    # unambiguous about where the payload begins.
    fence = _OPENING_JSON_FENCE.match(text, offset)
    if fence is not None:
        body = fence.end()
        while body < len(text) and text[body] in _JSON_WHITESPACE:
            body += 1
        if body < len(text) and text[body] in "{[":
            return (body,)
        offset = body

    # ``loads_lenient`` tolerates prose before a structured payload.  Use the
    # first opener of each supported root kind, just as its outermost-span
    # recovery does. All current ceiling-aware calls request objects, so once
    # an object begins it is authoritative; an array inside a malformed object
    # must not be reinterpreted as a second reply. An earlier prose ``[note]``
    # may fail before a later object begins. Trying at most those two candidates
    # preserves O(n) behavior. A same-kind brace in prose still fails closed
    # rather than authorizing an ambiguous recovery classification.
    object_start = text.find("{", offset)
    array_start = text.find("[", offset)
    if object_start < 0:
        return (array_start,) if array_start >= 0 else ()
    if array_start < 0 or object_start < array_start:
        return (object_start,)
    return (array_start, object_start)


def _scan_json_container_prefix(text: str, start: int) -> str:
    """Classify a top-level object/array as complete, incomplete, or invalid."""
    root = text[start]
    if root not in "{[":
        return _PREFIX_INVALID

    # Object states: key_or_end, key, colon, value, comma_or_end.
    # Array states: value_or_end, value, comma_or_end.
    stack = [_new_json_container_frame(root)]
    index = start + 1
    length = len(text)

    while True:
        while index < length and text[index] in _JSON_WHITESPACE:
            index += 1
        if index >= length:
            return _PREFIX_INCOMPLETE

        frame = stack[-1]
        kind, state = frame.kind, frame.state
        char = text[index]

        if kind == "object" and state in {"key_or_end", "key"}:
            if char == "}" and state == "key_or_end":
                index += 1
                if _close_container(stack):
                    return _PREFIX_COMPLETE
                continue
            if char != '"':
                return _PREFIX_INVALID
            key_start = index
            token, index = _scan_json_string(text, index)
            if token != _PREFIX_COMPLETE:
                return token
            # ``loads_exact_or_fenced`` rejects duplicate keys. Decode each
            # completed key once so escape-equivalent spellings (``"a"`` and
            # ``"\u0061"``) cannot make an irreparable contract violation look
            # like an output-ceiling cut.
            try:
                key = loads_strict_json(text[key_start:index])
            except ValueError:
                return _PREFIX_INVALID
            if key in frame.keys:
                return _PREFIX_INVALID
            frame.keys.add(key)
            frame.state = "colon"
            continue

        if kind == "object" and state == "colon":
            if char != ":":
                return _PREFIX_INVALID
            frame.state = "value"
            index += 1
            continue

        if state in {"value_or_end", "value"}:
            if kind == "array" and char == "]" and state == "value_or_end":
                index += 1
                if _close_container(stack):
                    return _PREFIX_COMPLETE
                continue

            if char in "{[":
                stack.append(_new_json_container_frame(char))
                index += 1
                continue
            if char == '"':
                token, index = _scan_json_string(text, index)
            elif char in "tfn":
                token, index = _scan_json_literal(text, index)
            elif char == "-" or char.isdigit() and char.isascii():
                number_start = index
                token, index = _scan_json_number(text, index)
                if token == _PREFIX_COMPLETE:
                    try:
                        loads_strict_json(text[number_start:index])
                    except ValueError:
                        return _PREFIX_INVALID
            else:
                return _PREFIX_INVALID
            if token != _PREFIX_COMPLETE:
                return token
            frame.state = "comma_or_end"
            continue

        if state == "comma_or_end":
            closer = "}" if kind == "object" else "]"
            if char == closer:
                index += 1
                if _close_container(stack):
                    return _PREFIX_COMPLETE
                continue
            if char != ",":
                return _PREFIX_INVALID
            frame.state = "key" if kind == "object" else "value"
            index += 1
            continue

        return _PREFIX_INVALID


def _close_container(stack: list[_JSONContainerFrame]) -> bool:
    """Pop a completed container; return True when the root is complete."""
    stack.pop()
    if not stack:
        return True
    # A nested container can only have been opened where its parent expected a
    # value.  Mark that value complete without re-reading any input.
    if stack[-1].state not in {"value", "value_or_end"}:
        return False
    stack[-1].state = "comma_or_end"
    return False


def _scan_json_string(text: str, start: int) -> tuple[str, int]:
    """Scan one JSON string, distinguishing a cut prefix from bad escaping."""
    index = start + 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == '"':
            return _PREFIX_COMPLETE, index + 1
        if char == "\\":
            index += 1
            if index >= length:
                return _PREFIX_INCOMPLETE, index
            escape = text[index]
            if escape in _JSON_SIMPLE_ESCAPES:
                index += 1
                continue
            if escape != "u":
                return _PREFIX_INVALID, index
            for _ in range(4):
                index += 1
                if index >= length:
                    return _PREFIX_INCOMPLETE, index
                if text[index] not in _JSON_HEX_DIGITS:
                    return _PREFIX_INVALID, index
            index += 1
            continue
        if ord(char) < 0x20:
            return _PREFIX_INVALID, index
        index += 1
    return _PREFIX_INCOMPLETE, index


def _scan_json_literal(text: str, start: int) -> tuple[str, int]:
    expected = {"t": "true", "f": "false", "n": "null"}[text[start]]
    for offset, char in enumerate(expected):
        index = start + offset
        if index >= len(text):
            return _PREFIX_INCOMPLETE, index
        if text[index] != char:
            return _PREFIX_INVALID, index
    return _PREFIX_COMPLETE, start + len(expected)


def _scan_json_number(text: str, start: int) -> tuple[str, int]:
    """Scan the RFC 8259 number grammar, retaining valid EOF prefixes."""
    index = start
    length = len(text)
    if text[index] == "-":
        index += 1
        if index >= length:
            return _PREFIX_INCOMPLETE, index

    if text[index] == "0":
        index += 1
    elif text[index] in "123456789":
        index += 1
        while index < length and text[index].isdigit() and text[index].isascii():
            index += 1
    else:
        return _PREFIX_INVALID, index

    if index < length and text[index] == ".":
        index += 1
        if index >= length:
            return _PREFIX_INCOMPLETE, index
        if not (text[index].isdigit() and text[index].isascii()):
            return _PREFIX_INVALID, index
        while index < length and text[index].isdigit() and text[index].isascii():
            index += 1

    if index < length and text[index] in "eE":
        index += 1
        if index >= length:
            return _PREFIX_INCOMPLETE, index
        if text[index] in "+-":
            index += 1
            if index >= length:
                return _PREFIX_INCOMPLETE, index
        if not (text[index].isdigit() and text[index].isascii()):
            return _PREFIX_INVALID, index
        while index < length and text[index].isdigit() and text[index].isascii():
            index += 1

    return _PREFIX_COMPLETE, index


def loads_lenient(raw: str, *, expect: str = "object") -> Any | None:
    """Parse *raw* as JSON, tolerating markdown fences and surrounding prose.

    Returns the parsed value, or None when nothing usable can be recovered.
    Never raises — a non-str (or None) `raw` is a malformed reply like any
    other, and callers of this module are all on a "log and move on" path.

    `expect` is the shape the caller asked the model for: "object", "array", or
    "any". It drives two things, in this order:

    1. ENVELOPE UNWRAP, applied to whatever parsed — strict or scanned. An
       ``expect="array"`` call that gets a dict back with exactly one list
       among its values returns that list. json_object mode pushes providers
       toward a top-level object even when the prompt asks for a bare array,
       so ``{"facts": [...]}`` is the shape a model reaches for to satisfy
       both; unwrapping it is recovery, not guessing, because there is a
       single candidate. Two or more lists is ambiguous and is left alone, as
       is a dict whose lone value is not a list. Anything not unwrapped is
       returned as-is for the caller's own shape check to reject.
    2. FALLBACK SCAN, only when the strict parse raised. It takes the
       OUTERMOST span — first opener to last closer — of the delimiters that
       could carry the expected shape, which is what strips a fence or a
       sentence on either side without needing to model markdown. An
       ``expect="array"`` scan considers ``{``…``}`` as well as ``[``…``]``,
       outermost span first, so a FENCED envelope reaches the same unwrap rule
       the bare one does — the two must not disagree, or whether a fact
       survives would depend on the model's formatting mood.

    Never falls back to the scan on a payload that parsed strictly: that is
    what keeps a well-formed top-level array from being narrowed to its first
    inner object under ``expect="object"``.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return _unwrap_envelope(json.loads(text), expect)
    except json.JSONDecodeError:
        pass

    salvaged: Any | None = None
    for start, end in _candidate_spans(text, expect):
        try:
            parsed = _unwrap_envelope(json.loads(text[start : end + 1]), expect)
        except json.JSONDecodeError:
            continue
        if _matches(parsed, expect):
            return parsed
        if salvaged is None:
            salvaged = parsed
    return salvaged


def _candidate_spans(text: str, expect: str) -> list[tuple[int, int]]:
    """Spans worth trying for *expect*, outermost (earliest opener) first.

    "array" includes the object delimiters because a fenced envelope hides its
    array inside braces; "object" stays narrow — nothing useful hides an object
    inside a bare array here.
    """
    wanted = ("object", "array") if expect in ("array", "any") else ("object",)
    spans = [_span(text, *_DELIMS[w]) for w in wanted]
    return sorted(s for s in spans if s is not None)


def _span(text: str, open_ch: str, close_ch: str) -> tuple[int, int] | None:
    """Outermost `open_ch`…`close_ch` span in *text*, or None if absent."""
    start, end = text.find(open_ch), text.rfind(close_ch)
    if start == -1 or end <= start:
        return None
    return start, end


def _matches(value: Any, expect: str) -> bool:
    if expect == "array":
        return isinstance(value, list)
    if expect == "any":
        return isinstance(value, (list, dict))
    return isinstance(value, dict)


def _unwrap_envelope(value: Any, expect: str) -> Any:
    """``{"facts": [...]}`` → ``[...]`` when an array was asked for and there is
    exactly one list to choose from. Every other shape passes through."""
    if expect != "array" or not isinstance(value, dict):
        return value
    lists = [v for v in value.values() if isinstance(v, list)]
    return lists[0] if len(lists) == 1 else value
