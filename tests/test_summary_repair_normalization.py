"""Actual dispatch preserves the compatibility cleaner's one-pass semantics."""
import json

import pytest

from hymem.dreaming import summary_recovery
from hymem.dreaming.summary import clean_summary
from tests.test_digest_bounded_summary_repair import SequenceLLM, extract, payload, source


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("value,accepted", [
    ('\'"12345678"\'', True),
    ('\'"1234567"\'', False),
    ('"\'12345678\'"', False),
    ('\'"1234567890"\'', True),
    ('"\'1234567890\'"', True),
    ('\'"\'12345678\'"\'', True),
    ('\'"\'1234567\'"\'', True),
    ('\'"\'12345\'"\'', False),
])
def test_actual_normal_dispatch_normalizes_selected_option_once(source, value, accepted, separate):
    raw = json.dumps({"alternatives": [value] * 3})
    expected = clean_summary(value)
    result = extract(source, SequenceLLM(payload(source, "x" * 501), raw),
                     separate_summary=separate)
    if accepted:
        assert not result.parse_failed
        assert result.summary_failure_reason is None
        assert result.summary == expected
        assert summary_recovery._parse_repair_alternatives(raw) == (expected, None)
    else:
        assert result.summary is None
        assert (result.summary_failure_reason if separate else result.failure_reason) == "summary_validation_failure"
        assert summary_recovery._parse_repair_alternatives(raw) == (None, "summary_validation_failure")


def test_cap_is_checked_before_quote_normalization(source):
    over = "'" + '"' + "x" * 497 + '"' + "'"
    fitting = "A complete fallback summary."
    raw = json.dumps({"alternatives": [over, fitting, fitting]})
    assert len(over) == 501 and len(clean_summary(over)) == 499
    result = extract(source, SequenceLLM(payload(source, "x" * 501), raw))
    assert not result.parse_failed and result.summary == fitting
    assert summary_recovery._parse_repair_alternatives(raw) == (fitting, None)


@pytest.mark.parametrize("bad,reason", [(None, "summary_shape_failure"),
                                       ("tiny", "summary_validation_failure")])
def test_valid_nested_quotes_cannot_hide_invalid_unused_option(source, bad, reason):
    raw = json.dumps({"alternatives": ['\'"12345678"\'', "A complete alternative.", bad]})
    result = extract(source, SequenceLLM(payload(source, "x" * 501), raw))
    assert result.parse_failed and result.failure_reason == reason
    assert summary_recovery._parse_repair_alternatives(raw) == (None, reason)
