#!/usr/bin/env python3
"""Best-effort, deliberately over-eager terminal redaction (stdlib only).

Derived from the deployed 2026-08-18 standalone script. Sensitive key names
hide their entire JSON value/subtree or the remainder of a non-JSON line.
Credential-shaped tokens are also hidden at any line position. Detection uses
the same rules, without the old detector-only identifier exemption.

This is not a security boundary or a guarantee that arbitrary secrets are
recognized. Encoded/fragmented secrets, unfamiliar formats, multiline shell
constructs and non-JSON structured formats may evade these heuristics. Inspect
only synthetic fixtures when testing; never treat a passing self-test as
permission to disclose raw configuration. Over-redaction is intentional.

Usage: redact.py [FILE ...]             # no files: read stdin
       redact.py --selftest [FILE ...]  # built-in canaries always run
Self-test output contains counts only, never paths, input or exception values.
"""

import json
import re
import sys


REDACTED = "<redacted>"
SECRET_KEY = re.compile(
    r"key|token|password|passwd|pass|secret|credential|auth|bearer|"
    r"salt|signature|session", re.I,
)
CRED_VALUE = re.compile(
    r"^(sk-|pk-|xox[abprs]-|ghp_|gho_|ghu_|ghs_|github_pat_|AKIA|ASIA|"
    r"eyJ[A-Za-z0-9_-]{6,}\.)"
    r"|^[A-Za-z0-9_\-]{28,}$"
    r"|^[A-Fa-f0-9]{32,}$"
    r"|^[A-Za-z0-9+/]{40,}={0,2}$"
)
_ASSIGNMENT = re.compile(
    r'''(?<![\w.\-])(?P<key>"[^"\r\n]+"|'[^'\r\n]+'|[A-Za-z_][A-Za-z0-9_.\-]*)[ \t]*[:=][ \t]*'''
)
# '=' is a separator, never part of an assignment's credential token.
# A separate base64 matcher retains padding/slash support without swallowing
# KEY= or path prefixes in front of recognizable sk-/ghp_/JWT tokens.
_TOKEN = re.compile(r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*")
_BASE64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


def _secret_spans(text):
    spans = []
    for match in _ASSIGNMENT.finditer(text):
        if not SECRET_KEY.search(match.group("key").strip("\"'")):
            continue
        end = text.find("\n", match.end())
        end = len(text) if end < 0 else end
        if end and text[end - 1:end] == "\r":
            end -= 1
        value = text[match.end():end]
        if value.strip() and value.strip().strip("\"'") != REDACTED:
            spans.append((match.end(), end))
    for pattern in (_TOKEN, _BASE64):
        for match in pattern.finditer(text):
            if CRED_VALUE.match(match.group()):
                spans.append(match.span())
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _redact_plain(text):
    spans = _secret_spans(text)
    parts, cursor = [], 0
    for start, end in spans:
        parts.extend((text[cursor:start], REDACTED))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), len(spans)


def _transform_json(obj, key=None):
    if isinstance(key, str) and SECRET_KEY.search(key) and obj != REDACTED:
        return REDACTED, 1
    if isinstance(obj, dict):
        result, count = {}, 0
        for name, value in obj.items():
            safe_name, name_count = _redact_plain(name)
            safe_value, value_count = _transform_json(value, name)
            result[safe_name] = safe_value
            count += name_count + value_count
        return result, count
    if isinstance(obj, list):
        results = [_transform_json(value) for value in obj]
        return [value for value, _ in results], sum(count for _, count in results)
    if isinstance(obj, str):
        return _redact_plain(obj)
    return obj, 0


def redact_json(obj, key=None):
    return _transform_json(obj, key)[0]


def redact_lines(text):
    return _redact_plain(text)[0]


def _transform_text(text):
    if text.lstrip().startswith(("{", "[")):
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            pass
        else:
            safe, count = _transform_json(obj)
            return json.dumps(safe, indent=2, ensure_ascii=False), count
    return _redact_plain(text)


def redact_text(text):
    return _transform_text(text)[0]


def count_secrets(text):
    """Count detected regions (not necessarily distinct credentials)."""
    return _transform_text(text)[1]


def _canaries():
    token = "sk-" + "0123456789abcdef" * 2
    short = "synthetic-short-value"
    for raw in (
        token, "\t" + token, '"' + token + '"',
        "export KEY=" + token, "env PUBLIC=yes API_KEY=" + token + " command",
        "prefix " + token + " suffix", "export TOKEN=" + short,
        json.dumps({"neutral": ["prefix " + token + " suffix"]}),
        json.dumps({"PASSWORD": [short, {"arbitrary": short}]}),
        json.dumps({token: "ordinary"}),
    ):
        yield raw, (token, short)


def _selftest(paths):
    checks, failures = 0, 0
    for raw, known_secrets in _canaries():
        checks += 1
        try:
            safe = redact_text(raw)
            # Independent literal oracle: even jointly broken redaction and
            # heuristic detection cannot certify a surviving known canary.
            passed = (
                isinstance(safe, str)
                and not any(secret in safe for secret in known_secrets)
                and count_secrets(raw) > 0
                and count_secrets(safe) == 0
                and redact_text(safe) == safe
            )
        except Exception:
            passed = False
        failures += not passed
    print(f"{'FAIL' if failures else 'PASS'} built-in checks={checks} failures={failures}")
    rc = 1 if failures else 0
    for index, path in enumerate(paths, 1):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                raw = handle.read()
            before = count_secrets(raw)
            after = count_secrets(redact_text(raw))
        except Exception:
            print(f"FAIL input={index} unreadable-or-unprocessable")
            rc = 2
            continue
        print(f"{'FAIL' if after else 'PASS'} input={index} before={before} after={after}")
        if after and rc == 0:
            rc = 1
    return rc


def main():
    args = sys.argv[1:]
    if any(arg.startswith("-") and arg != "--selftest" for arg in args) or args.count("--selftest") > 1:
        print("ERROR unsupported option", file=sys.stderr)
        return 2
    paths = [arg for arg in args if arg != "--selftest"]
    if "--selftest" in args:
        return _selftest(paths)
    try:
        inputs = []
        for path in paths:
            with open(path, encoding="utf-8", errors="replace") as handle:
                inputs.append(handle.read())
        if not paths:
            inputs.append(sys.stdin.read())
        # Finish reading/redacting every input before publishing any output.
        outputs = [redact_text(text) for text in inputs]
        for output in outputs:
            sys.stdout.write(output.rstrip("\n") + "\n")
    except Exception:
        print("ERROR input or output unavailable", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
