"""Best-effort secret/PII scrubbing for text headed into the on-disk store.

This is a *defensive* redactor, not a guarantee. It targets high-confidence,
low-false-positive patterns (provider API keys, JWTs, private-key blocks,
bearer/credential strings, credentials embedded in URLs, email addresses) so
the SQLite store and the derived chunks don't hold raw secrets. It deliberately
avoids generic high-entropy heuristics, which would shred ordinary prose.

Applied at the single ingest chokepoint (`HyMem.log_message`/`log_messages`),
so chunks — which are built from already-redacted messages — inherit the
scrubbing without a second pass.
"""
from __future__ import annotations

import re

# Each entry: (compiled pattern, replacement). Order matters — the most
# specific / structural patterns run first so a key embedded in a URL or an
# Authorization header is caught by its structural rule before the looser
# generic rules see it.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # PEM private key blocks (RSA / EC / OpenSSH / generic).
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
            r".*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED-PRIVATE-KEY]",
    ),
    # JSON Web Tokens: header.payload.signature, each base64url.
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        "[REDACTED-JWT]",
    ),
    # OpenAI / Anthropic / Google style prefixed keys (sk-, sk-ant-, AIza, etc.).
    (re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}\b"), "[REDACTED-API-KEY]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"), "[REDACTED-API-KEY]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "[REDACTED-API-KEY]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED-API-KEY]"),
    # AWS access key id.
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED-AWS-KEY]"),
    # Authorization: Bearer <token> / Basic <token>.
    (
        re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
        r"\1 [REDACTED-TOKEN]",
    ),
    # Credentials embedded in a URL (scheme://user:pass@host).
    (
        re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@]+):[^\s:/@]+@"),
        r"\1:[REDACTED-CREDENTIAL]@",
    ),
    # key/secret/token/password = "value" assignments (json, env, kwargs).
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|secret|token|password|passwd|access[_-]?token))"
            r"(\s*[:=]\s*)"
            r"['\"]?[A-Za-z0-9._~+/=-]{8,}['\"]?"
        ),
        r"\1\2[REDACTED-SECRET]",
    ),
    # Email addresses.
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED-EMAIL]",
    ),
]


def redact(text: str) -> str:
    """Return *text* with recognised secrets/PII replaced by ``[REDACTED-*]``
    markers. Idempotent: re-running on already-redacted text is a no-op because
    the markers contain no secret-shaped substrings."""
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _changed_span(match: re.Match[str], replacement: str) -> tuple[int, int]:
    """Return the part of one match changed by its normal replacement.

    Several rules intentionally preserve structural prefixes (``Bearer ``, an
    assignment name, or a URL username). Longest common prefix/suffix isolates
    only the secret-bearing middle instead of masking that benign structure.
    """

    original = match.group(0)
    expanded = match.expand(replacement)
    prefix = 0
    limit = min(len(original), len(expanded))
    while prefix < limit and original[prefix] == expanded[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(original) - prefix
        and suffix < len(expanded) - prefix
        and original[len(original) - suffix - 1]
        == expanded[len(expanded) - suffix - 1]
    ):
        suffix += 1
    return prefix, len(original) - suffix


def redact_preserving_length(text: str) -> str:
    """Mask only recognised sensitive spans without changing character offsets.

    This is for exact source artifacts whose published cursors may point inside
    the content. Ordinary ingest/export should prefer :func:`redact`, whose
    visible markers are more useful to humans.
    """

    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        def mask(match: re.Match[str], replacement: str = replacement) -> str:
            start, end = _changed_span(match, replacement)
            original = match.group(0)
            candidate = (
                original[:start] + ("*" * (end - start)) + original[end:]
            )
            # The URL-credential rule deliberately accepts almost every
            # non-whitespace password character, including ``*``.  Fall back
            # to same-width spaces when the rule would otherwise recognise
            # its own mask; this keeps both coordinates and redaction fixed.
            if pattern.search(candidate) is not None:
                candidate = (
                    original[:start] + (" " * (end - start)) + original[end:]
                )
            return candidate

        text = pattern.sub(mask, text)
    return text


def sensitive_fragments(text: str) -> tuple[str, ...]:
    """Return the exact secret-bearing fragments recognized in ``text``.

    The values are intended only for in-memory identity rebinding during a
    redacted export; callers must never persist or log them.
    """

    if not text:
        return ()
    found: list[str] = []
    seen: set[str] = set()
    for pattern, replacement in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = _changed_span(match, replacement)
            fragment = match.group(0)[start:end]
            if fragment and fragment not in seen:
                seen.add(fragment)
                found.append(fragment)
    return tuple(found)
