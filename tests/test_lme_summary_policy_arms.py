"""Summary-policy contrasts discount one validated duplicate, not a config."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import lme_registry as registry, run_registry as shared
from hymem.dreaming.summary_policy import LEGACY_COMPLETE_V1 as LEGACY, BOUNDED_HIGHLIGHTS_V1 as BOUNDED


FIELD = "digest_summary_policy"


def config(policy):
    return {
        FIELD: policy, "top_k": 15, "seed": 0,
        "effective_hymem_config": {
            FIELD: policy, "message_fts_top_k": 15,
            "other": {"nested": [True, 2], "retained": None},
        },
    }


def compare(a, b):
    return registry.lme_arm_evidence(a, b, FIELD)


def test_clean_policy_pair_is_evidenced_without_false_duplicate_confound():
    a, b = config(LEGACY), config(BOUNDED)
    before = deepcopy((a, b))
    assert shared.arm_evidence(a, b, FIELD)[2] == ["effective_hymem_config"]
    verdict, note, confounds = compare(a, b)
    assert verdict == shared.ARM_EVIDENCED
    assert LEGACY in note and BOUNDED in note
    assert confounds == []
    assert (a, b) == before


@pytest.mark.parametrize("policy", [LEGACY, BOUNDED])
def test_same_recorded_policy_is_not_an_ab(policy):
    assert compare(config(policy), config(policy))[::2] == (shared.ARM_SAME, [])


@pytest.mark.parametrize("location", ["top", "nested"])
@pytest.mark.parametrize("before,after", [
    (True, 1), (False, 0), (1, 1.0), (0, 0.0), (1, "1"),
    ({"value": True}, {"value": 1}), ([True], [1]),
    ([1, 2], [2, 1]), ({"a": 1}, {"a": 2}),
])
def test_every_other_value_difference_remains_a_confound(location, before, after):
    a, b = config(LEGACY), config(BOUNDED)
    target_a = a if location == "top" else a["effective_hymem_config"]
    target_b = b if location == "top" else b["effective_hymem_config"]
    target_a["real_difference"], target_b["real_difference"] = before, after
    old_bytes = [json.dumps(c, sort_keys=True) for c in (a, b)]
    verdict, _, confounds = compare(a, b)
    assert verdict == shared.ARM_EVIDENCED
    assert confounds == (["real_difference"] if location == "top" else ["effective_hymem_config"])
    assert [json.dumps(c, sort_keys=True) for c in (a, b)] == old_bytes


@pytest.mark.parametrize("location", ["top", "nested"])
@pytest.mark.parametrize("side", [0, 1])
def test_missing_and_explicit_null_are_different(location, side):
    arms = [config(LEGACY), config(BOUNDED)]
    target = arms[side] if location == "top" else arms[side]["effective_hymem_config"]
    target["new_setting"] = None
    assert compare(*arms)[2] == (["new_setting"] if location == "top" else ["effective_hymem_config"])


def test_only_nested_policy_is_discounted_and_top_timing_exclusion_is_unchanged():
    a, b = config(LEGACY), config(BOUNDED)
    a.update(elapsed_s=10, total_tokens=100)
    b.update(elapsed_s=12, total_tokens=101)
    assert compare(a, b)[2] == []
    a["effective_hymem_config"]["elapsed_s"] = 1
    b["effective_hymem_config"]["elapsed_s"] = 2
    b["top_k"] = 20
    assert compare(a, b)[2] == ["effective_hymem_config", "top_k"]


def test_mapping_key_order_does_not_create_a_confound():
    a, b = config(LEGACY), config(BOUNDED)
    b = dict(reversed(list(b.items())))
    b["effective_hymem_config"] = dict(reversed(list(b["effective_hymem_config"].items())))
    assert compare(a, b)[2] == []


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("missing", ["top", "nested", "effective", "both"])
def test_missing_disclosure_is_not_imputed_from_history(side, missing):
    arms = [config(LEGACY), config(BOUNDED)]
    if missing in ("top", "both"):
        arms[side].pop(FIELD)
    if missing in ("nested", "both"):
        arms[side]["effective_hymem_config"].pop(FIELD)
    if missing == "effective":
        arms[side].pop("effective_hymem_config")
    before = deepcopy(arms)
    verdict, note, _ = compare(*arms)
    assert verdict == shared.ARM_UNEVIDENCED
    assert "absent" in note if missing == "both" else "disclosure" in note
    assert arms == before


def test_both_historical_configs_remain_unevidenced_without_inventing_policy():
    a = {"effective_hymem_config": {"message_fts_top_k": 15}}
    b = deepcopy(a)
    assert compare(a, b)[0] == shared.ARM_UNEVIDENCED
    assert FIELD not in a and FIELD not in b
    assert FIELD not in a["effective_hymem_config"]
    assert compare(None, None)[0] == shared.ARM_UNEVIDENCED


@pytest.mark.parametrize("location", ["top", "nested", "both"])
@pytest.mark.parametrize("bad", [None, True, 1, [], {}, "bounded", "legacy_complete", "bounded_highlights_v2", "bounded_highlights_v1 "])
def test_invalid_recorded_policy_is_nonpassing(location, bad):
    a, b = config(LEGACY), config(BOUNDED)
    if location in ("top", "both"):
        b[FIELD] = bad
    if location in ("nested", "both"):
        b["effective_hymem_config"][FIELD] = bad
    assert compare(a, b)[0] == shared.ARM_UNEVIDENCED


@pytest.mark.parametrize("top,nested", [(LEGACY, BOUNDED), (BOUNDED, LEGACY)])
def test_valid_but_mismatched_recorded_policies_are_nonpassing(top, nested):
    a, b = config(LEGACY), config(top)
    b["effective_hymem_config"][FIELD] = nested
    assert compare(a, b)[0] == shared.ARM_UNEVIDENCED


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object(), {1, 2}])
def test_nonfinite_or_nonjson_other_config_values_cannot_gain_comparison_assurance(bad):
    a, b = config(LEGACY), config(BOUNDED)
    a["other"] = b["other"] = bad
    assert compare(a, b)[0] == shared.ARM_UNEVIDENCED


@pytest.mark.parametrize("bad", [[], True, 1, "config"])
def test_malformed_config_block_is_nonpassing(bad):
    assert compare(config(LEGACY), bad)[0] == shared.ARM_UNEVIDENCED


def test_nonpolicy_levers_delegate_without_changing_shared_behavior(monkeypatch):
    a, b = config(LEGACY), config(BOUNDED)
    calls = []

    def original(left, right, lever):
        calls.append((left, right, lever))
        return "unchanged", "shared note", ["shared confound"]

    monkeypatch.setattr(shared, "arm_evidence", original)
    assert registry.lme_arm_evidence(a, b, "other_lever") == ("unchanged", "shared note", ["shared confound"])
    assert calls == [(a, b, "other_lever")]
    assert calls[0][0] is a and calls[0][1] is b


@pytest.mark.parametrize("scenario,expected_code", [
    ("clean", 0), ("confounded", 0), ("same", 1),
    ("historical", 1), ("partial", 1), ("mismatched", 1),
])
@pytest.mark.parametrize("invocation", ["absolute", "module"])
def test_actual_cli_reports_policy_evidence_without_duplicate_warning(tmp_path, scenario, expected_code, invocation):
    a, b = config(LEGACY), config(BOUNDED)
    if scenario == "confounded":
        b["effective_hymem_config"]["other"]["nested"][0] = 1
        b["top_k"] = 20
    elif scenario == "same":
        b = config(LEGACY)
    elif scenario == "historical":
        a.pop(FIELD)
        a["effective_hymem_config"].pop(FIELD)
    elif scenario == "partial":
        a["effective_hymem_config"].pop(FIELD)
    elif scenario == "mismatched":
        a["effective_hymem_config"][FIELD] = BOUNDED
    paths = [tmp_path / "left.json", tmp_path / "right.json"]
    for path, value in zip(paths, (a, b)):
        path.write_text(json.dumps({"benchmark": "LongMemEval", "config": value}), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    target = ([str(root / "benchmarks/lme_registry.py")] if invocation == "absolute" else
              ["-m", "benchmarks.lme_registry"])
    result = subprocess.run(
        [sys.executable, "-E", "-s", "-S", "-B", *target, "arms", *map(str, paths), "--lever", FIELD],
        cwd=tmp_path if invocation == "absolute" else root,
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    if scenario == "clean":
        assert "[EVIDENCED]" in result.stdout and "confounded" not in result.stdout
    elif scenario == "confounded":
        assert "confounded on 2 other key(s): effective_hymem_config, top_k" in result.stdout
    elif scenario == "same":
        assert "[SAME_ARM]" in result.stdout
    else:
        assert "[UNEVIDENCED]" in result.stdout
