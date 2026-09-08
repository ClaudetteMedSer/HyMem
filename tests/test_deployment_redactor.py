"""Standalone deployment redactor checks; all inputs are synthetic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "tools/deployment/redact.py"
_TOKEN = "sk-" + "a1b2c3d4" * 4
_SHORT = "synthetic-short-value"


@pytest.fixture
def redactor():
    spec = importlib.util.spec_from_file_location("deployment_redactor", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_no_secrets(text, *secrets):
    if any(secret in text for secret in secrets):
        pytest.fail("synthetic secret survived (value intentionally omitted)")


def _cli(*args, text=""):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *map(str, args)], input=text,
        text=True, capture_output=True, timeout=5, check=False,
    )


@pytest.mark.parametrize("case", [
    "bare", "indent", "double-quoted", "single-quoted", "brackets", "path",
    "assignment", "export", "quoted-export", "env", "shell-prefix", "prose",
    "short-prefix", "short-key", "known-scalar", "opaque", "hex", "base64", "jwt",
])
def test_plain_redaction_detection_and_idempotence(redactor, case):
    cases = {
        "bare": (_TOKEN, _TOKEN),
        "indent": ("\t" + _TOKEN, _TOKEN),
        "double-quoted": ('"' + _TOKEN + '"', _TOKEN),
        "single-quoted": ("'" + _TOKEN + "'", _TOKEN),
        "brackets": ("prefix(" + _TOKEN + ")suffix", _TOKEN),
        "path": ("/synthetic/" + _TOKEN, _TOKEN),
        "assignment": ("KEY=" + _TOKEN, _TOKEN),
        "export": ("export KEY=" + _TOKEN, _TOKEN),
        "quoted-export": ('export API_KEY="' + _TOKEN + '"', _TOKEN),
        "env": ("env PUBLIC=yes API_KEY=" + _TOKEN + " command", _TOKEN),
        "shell-prefix": ("sudo -E env TOKEN='" + _SHORT + "' command", _SHORT),
        "prose": ("prefix\t" + _TOKEN + " suffix", _TOKEN),
        "short-prefix": ("neutral=sk-small", "sk-small"),
        "short-key": ("export KEY=" + _SHORT, _SHORT),
        "known-scalar": ("PASSWORD: 1234", "1234"),
        "opaque": ("gateway_auto_continue_freshness", "gateway_auto_continue_freshness"),
        "hex": ("neutral=" + "fedcba98" * 4, "fedcba98" * 4),
        "base64": ("neutral=" + "aB9/" * 12 + "==", "aB9/" * 12 + "=="),
        "jwt": ("eyJsynthetic.payload.signature", "eyJsynthetic.payload.signature"),
    }
    raw, secret = cases[case]
    safe = redactor.redact_text(raw)
    _assert_no_secrets(safe, secret)
    assert redactor.count_secrets(raw) > 0
    assert redactor.count_secrets(safe) == 0
    assert redactor.redact_text(safe) == safe


def test_assignment_prefix_and_benign_content_are_preserved(redactor):
    assert redactor.redact_text("export KEY=" + _TOKEN) == "export KEY=<redacted>"
    assert redactor.redact_text("env PUBLIC=yes TOKEN=" + _SHORT) == "env PUBLIC=yes TOKEN=<redacted>"
    benign = "host=localhost\r\nport=8766\nfeature=true\n"
    assert redactor.redact_text(benign) == benign
    assert redactor.count_secrets(benign) == 0


@pytest.mark.parametrize("case", ["nested", "subtree", "scalar", "key", "array"])
def test_json_structure_sensitive_subtrees_and_embedded_tokens(redactor, case):
    source = {
        "nested": {"service": {"api_key": _SHORT, "note": "prefix " + _TOKEN + " suffix"}},
        "subtree": {"PASSWORD": [_SHORT, {"arbitrary": _SHORT}]},
        "scalar": {"api_key": 1234, "auth": False, "secret": None},
        "key": {_TOKEN: "ordinary"},
        "array": ["prefix " + _TOKEN, {"ordinary": "safe"}],
    }[case]
    raw = json.dumps(source)
    safe = redactor.redact_text(raw)
    _assert_no_secrets(safe, _TOKEN, _SHORT, "1234")
    parsed = json.loads(safe)
    if case == "subtree":
        assert parsed == {"PASSWORD": "<redacted>"}
    if case == "scalar":
        assert set(parsed.values()) == {"<redacted>"}
    assert redactor.count_secrets(raw) > 0
    assert redactor.count_secrets(safe) == 0
    assert redactor.redact_text(safe) == safe


def test_malformed_json_uses_conservative_line_redaction(redactor):
    raw = '{"API_KEY": "' + _SHORT + '"'
    safe = redactor.redact_text(raw)
    _assert_no_secrets(safe, _SHORT)
    assert redactor.count_secrets(raw) > 0
    assert redactor.count_secrets(safe) == 0


def test_real_cli_stdin_and_json():
    for raw in ("export KEY=" + _TOKEN, json.dumps({"nested": {"auth": _SHORT}})):
        result = _cli(text=raw)
        assert result.returncode == 0
        assert result.stderr == ""
        _assert_no_secrets(result.stdout, _TOKEN, _SHORT)
        assert "<redacted>" in result.stdout
        repeated = _cli(text=result.stdout)
        assert repeated.returncode == 0
        assert repeated.stdout == result.stdout
    assert json.loads(result.stdout) == {"nested": {"auth": "<redacted>"}}


def test_real_cli_multiple_files_and_selftest_never_print_filenames(tmp_path):
    first = tmp_path / (_TOKEN + ".txt")
    second = tmp_path / "ordinary.json"
    first.write_text("env TOKEN=" + _SHORT)
    second.write_text(json.dumps({"note": _TOKEN}))
    for flags in ((), ("--selftest",)):
        result = _cli(*flags, first, second)
        assert result.returncode == 0
        assert result.stderr == ""
        _assert_no_secrets(result.stdout, _TOKEN, _SHORT, str(tmp_path))
        if flags:
            assert "built-in checks=" in result.stdout
            assert "input=1" in result.stdout and "input=2" in result.stdout


def test_no_file_selftest_runs_canaries_without_reading_stdin():
    result = _cli("--selftest", text=_TOKEN)
    assert result.returncode == 0
    assert "PASS built-in checks=" in result.stdout
    assert "checks=0" not in result.stdout
    assert result.stderr == ""
    _assert_no_secrets(result.stdout, _TOKEN, _SHORT)


@pytest.mark.parametrize("fault", ["redactor", "detector", "both", "exception"])
def test_builtin_selftest_independently_detects_broken_components(
    redactor, monkeypatch, capsys, fault,
):
    if fault in {"redactor", "both"}:
        monkeypatch.setattr(redactor, "redact_text", lambda text: text)
    if fault in {"detector", "both"}:
        monkeypatch.setattr(redactor, "count_secrets", lambda text: 0)
    if fault == "exception":
        def fail(_text):
            raise RuntimeError(_TOKEN)
        monkeypatch.setattr(redactor, "redact_text", fail)
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--selftest"])
    assert redactor.main() == 1
    output = capsys.readouterr()
    assert "FAIL built-in checks=" in output.out
    _assert_no_secrets(output.out + output.err, _TOKEN, _SHORT)
    # The canary's literal secret is not the synthetic input used by these
    # external tests; ensure no internal canary was printed either.
    for _raw, secrets in redactor._canaries():
        _assert_no_secrets(output.out + output.err, *secrets)


@pytest.mark.parametrize("case", ["unknown-flag", "read-error", "selftest-read-error"])
def test_cli_errors_are_nonzero_and_do_not_echo_sensitive_arguments(tmp_path, case):
    missing = tmp_path / (_TOKEN + ".missing")
    args = {
        "unknown-flag": ["--" + _TOKEN],
        "read-error": [str(missing)],
        "selftest-read-error": ["--selftest", str(missing)],
    }[case]
    result = _cli(*args, text=_TOKEN)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    _assert_no_secrets(output, _TOKEN, str(tmp_path))
    assert "Traceback" not in output
    if case != "selftest-read-error":
        assert result.stdout == ""
    else:
        assert "built-in checks=" in result.stdout


def test_read_failure_does_not_publish_partial_output(tmp_path):
    valid = tmp_path / "valid.txt"
    valid.write_text("safe content")
    result = _cli(valid, tmp_path / (_TOKEN + ".missing"))
    assert result.returncode != 0
    assert result.stdout == ""
    _assert_no_secrets(result.stderr, _TOKEN, str(tmp_path))


def test_many_credential_spans_remain_complete_and_idempotent(redactor):
    raw = ("before " + _TOKEN + " after\n") * 4000
    safe = redactor.redact_text(raw)
    _assert_no_secrets(safe, _TOKEN)
    assert safe.count("<redacted>") == 4000
    assert redactor.count_secrets(safe) == 0
    assert redactor.redact_text(safe) == safe
