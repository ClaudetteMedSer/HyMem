"""An A/B is two files and a claim that they differ in one lever.

The claim has to come from somewhere. A filename is not somewhere:
`guard-epg-on` / `guard-epg-off` is the operator's memory of what they typed,
written into a stem. A `--set` in the registry is the same memory, recorded
later, by the same person.

The worked example is real and is why this exists. Both arms of the
2026-08-30/31 episode-granularity guard ran BEFORE 6543ee6 taught the adapter
to write the lever into its config block, so the two blocks are byte-identical
except `elapsed_s`/`total_tokens`, and nothing inside either file says which
arm it is. Both scored OVERALL 71.0. That is what a real null looks like, and
it is also what two runs of the SAME configuration look like -- the artifacts
cannot separate the readings, so the pair cannot discharge a non-regression
gate on that lever however clean the numbers are.

Same shape as the unreachable-code-path control in docs/diagnostic_controls.md,
one level up: not an instrument that never touched the lever, but a pair of
results that cannot show whether it did.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import benchmarks.run_registry as rr  # noqa: E402

LEVER = "episode_granularity_enabled"


def _cfg(**kw):
    base = {"scale": "S", "sample": 0, "seed": 0, "top_k": 15,
            "elapsed_s": 9807.5, "total_tokens": 1744256}
    base.update(kw)
    return base


# ------------------------------------------------------------- refuses ----

def test_a_pair_that_records_nothing_is_unevidenced():
    """THE case. Byte-identical config blocks, opposite filenames."""
    v, note, _ = rr.arm_evidence(_cfg(), _cfg(elapsed_s=11216.7), LEVER)
    assert v == rr.ARM_UNEVIDENCED
    assert "absent from the config block of arm(s) A, B" in note
    assert "not a record of what ran" in note


def test_one_arm_recording_it_is_still_not_enough():
    """Half a record is not a contrast: the other file could be either arm."""
    v, note, _ = rr.arm_evidence(_cfg(**{LEVER: True}), _cfg(), LEVER)
    assert v == rr.ARM_UNEVIDENCED
    assert "arm(s) B" in note
    assert "A, B" not in note


def test_two_arms_recording_the_SAME_value_are_not_an_ab():
    """A pair that names itself on/off and records off/off. The filenames are
    the only thing claiming a contrast, and they are not evidence."""
    v, note, _ = rr.arm_evidence(_cfg(**{LEVER: False}),
                                 _cfg(**{LEVER: False}), LEVER)
    assert v == rr.ARM_SAME
    assert "whatever it is named" in note


# ------------------------------------------------------------- accepts ----

def test_a_recorded_difference_is_evidenced():
    v, note, confounds = rr.arm_evidence(_cfg(**{LEVER: False}),
                                         _cfg(**{LEVER: True}), LEVER)
    assert v == rr.ARM_EVIDENCED
    assert "A=False B=True" in note
    assert confounds == []


def test_timing_keys_are_not_confounds():
    """Two runs of anything differ in wall clock and token count. Reporting
    those as confounds would make every real pair look dirty, and a warning
    that always fires is not read."""
    _, _, confounds = rr.arm_evidence(
        _cfg(**{LEVER: False}),
        _cfg(elapsed_s=11216.7, total_tokens=1779172, **{LEVER: True}), LEVER)
    assert confounds == []


def test_a_second_moved_lever_is_reported_as_a_confound():
    """Evidenced is not the same as clean: the contrast is still attributable
    to either lever."""
    v, _, confounds = rr.arm_evidence(
        _cfg(**{LEVER: False, "aggregation_nodes_enabled": False}),
        _cfg(**{LEVER: True, "aggregation_nodes_enabled": True}), LEVER)
    assert v == rr.ARM_EVIDENCED
    assert confounds == ["aggregation_nodes_enabled"]


def test_a_key_present_in_only_one_block_is_a_confound():
    _, _, confounds = rr.arm_evidence(_cfg(**{LEVER: False}),
                                      _cfg(**{LEVER: True, "distill": True}),
                                      LEVER)
    assert confounds == ["distill"]


def test_an_empty_config_does_not_crash():
    v, _, _ = rr.arm_evidence(None, None, LEVER)
    assert v == rr.ARM_UNEVIDENCED


# ----------------------------------------------------------------- CLI ----

def _artifact(path, cfg):
    path.write_text(json.dumps(
        {"benchmark": "LongMemEval", "date": "2026-08-31T03:11:30+00:00",
         "config": cfg, "scores": {}, "per_question": []}))
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO / "benchmarks" / "lme_registry.py"),
         "arms", *args, "--lever", LEVER],
        capture_output=True, text=True)


def test_the_cli_exits_nonzero_on_an_unevidenced_pair(tmp_path):
    """Nonzero because this is a refusal, not a note. The guard pair on the box
    returns exactly this."""
    a = _artifact(tmp_path / "off.json", _cfg())
    b = _artifact(tmp_path / "on.json", _cfg(elapsed_s=11216.7))
    r = _run(str(a), str(b))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[UNEVIDENCED]" in r.stdout
    assert "cannot discharge a gate on that lever" in r.stdout
    assert "The scores may still be" in r.stdout, "not a claim the numbers are wrong"


def test_the_cli_exits_zero_on_an_evidenced_pair(tmp_path):
    a = _artifact(tmp_path / "off.json", _cfg(**{LEVER: False}))
    b = _artifact(tmp_path / "on.json", _cfg(**{LEVER: True}))
    r = _run(str(a), str(b))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[EVIDENCED]" in r.stdout


def test_the_cli_names_confounds(tmp_path):
    a = _artifact(tmp_path / "off.json", _cfg(**{LEVER: False, "top_k": 15}))
    b = _artifact(tmp_path / "on.json", _cfg(**{LEVER: True, "top_k": 30}))
    r = _run(str(a), str(b))
    assert "confounded on 1 other key(s): top_k" in r.stdout
