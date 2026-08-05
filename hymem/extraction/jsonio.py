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

import json
from typing import Any

_DELIMS = {"object": ("{", "}"), "array": ("[", "]")}


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
