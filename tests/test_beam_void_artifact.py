"""A voided run must keep its rows, and must never be readable as a result.

B2 (2026-09-01) judged all 160 rows, hit one silent-0, and exited before the
artifact was written -- so 219 seconds of judged data was discarded and the
only way to see what the judge had said was to buy it again. The refusal was
correct; throwing the evidence away was a second defect wearing the first
one's clothes.

The fix has two halves and BOTH are load-bearing. Persisting the rows is
useless if a reader can mistake them for a verdict, and marking them void is
useless if the marking is easy to miss -- so the marker is in the metadata
AND the filename, and the readers refuse on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import beam_adapter as ba  # noqa: E402

SILENT0 = [("IF", "What are some common responses when something goes wrong",
            '{"scores": [1], "total_score": 1.0, "explanation": "The response inclu',
            "length")]


def test_a_clean_run_records_void_as_null_not_as_an_absent_key():
    """An artifact predating this field and a void artifact must not look
    alike. Absence is something a reader has to notice; null is something
    they can test."""
    assert ba.void_record([]) is None


def test_a_voided_run_records_the_rule_it_broke_not_just_that_it_broke_one():
    v = ba.void_record(SILENT0)
    assert v["n_silent0"] == 1
    assert "silent-0" in v["reason"]
    assert "4.7" in v["rule"]


def test_the_offending_judge_output_is_preserved_for_diagnosis():
    """The whole point of keeping the rows. B2's silent-0 turned out to be a
    TRUNCATION -- the judge had scored the row 1.0 and run out of tokens
    mid-explanation -- which is only visible in the raw head."""
    v = ba.void_record(SILENT0)
    assert '"scores": [1]' in v["rows"][0]["judge_raw_head"]
    assert v["rows"][0]["ability"] == "IF"
    # The structural half of the same fact: "length" says the judge ran out of
    # tokens, without anyone having to read the prose to work that out.
    assert v["rows"][0]["finish_reason"] == "length"


def test_voidness_is_in_the_filename_not_only_the_metadata():
    """Artifacts get cited by name in commits, pre-registrations and chat. A
    void run whose name reads like every other result is how it gets quoted
    as one by someone who never opened it."""
    src = Path("/tmp/results_20260831T165039Z.json")
    clean = ba.rejudge_dest(src, "deepseek-chat", "20260901T090000Z", None)
    voided = ba.rejudge_dest(src, "deepseek-chat", "20260901T090000Z", ba.void_record(SILENT0))
    assert not clean.name.endswith("-VOID.json")
    assert voided.name.endswith("-VOID.json")
    assert voided.name.replace("-VOID.json", ".json") == clean.name


def test_a_slash_in_the_model_name_cannot_escape_the_directory():
    dest = ba.rejudge_dest(Path("/tmp/r.json"), "vendor/model", "S", None)
    assert "/" not in dest.name and dest.parent == Path("/tmp")


@pytest.mark.parametrize("script", ["step1_pin_compare", "b2_alias_churn"])
def test_both_readers_refuse_a_void_artifact(script, capsys):
    """Preserving the evidence only helps if reading it as a result stays
    impossible -- otherwise the fix that saved the data becomes the reason
    someone quotes it."""
    mod = __import__(script)
    with pytest.raises(SystemExit) as e:
        mod.refuse_void("B2", {"void": ba.void_record(SILENT0)})
    assert e.value.code == 4
    assert "VOID" in capsys.readouterr().out


@pytest.mark.parametrize("script", ["step1_pin_compare", "b2_alias_churn"])
@pytest.mark.parametrize("meta", [{}, {"void": None}])
def test_both_readers_pass_a_clean_or_pre_field_artifact(script, meta):
    __import__(script).refuse_void("B", meta)
